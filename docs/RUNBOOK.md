<!-- doc-meta
scope: Pre-launch operations for ViolaWake production.
authority: LIVING
code_paths: console/backend/app/health.py, scripts/backup_to_r2.py, docker-compose.production.yml, console/backend/app/routes/inbound_email.py
staleness_signals: health endpoint semantics, backup bucket, restore command, or human launch handoff changes
-->

# ViolaWake Launch Runbook

Production backend: `wakeword-backend-1`.
Production database: `wakeword-postgres-1`.
Production URL: `https://api.violawake.com`.

## Health monitoring

Monitor URL:

```bash
https://api.violawake.com/api/health
```

Expected response: HTTP 200 only when the app is ready and all configured checks pass. The response includes `failed_checks` and returns HTTP 503 when Postgres, storage, the training queue, billing config, or startup readiness fails.

UptimeRobot setup:

1. Open `https://uptimerobot.com/` and create or log into the account.
2. Add a new HTTPS monitor.
3. URL: `https://api.violawake.com/api/health`.
4. Interval: `5 minutes`.
5. Alert contact: the launch/operator email address.
6. Expected status: HTTP `200`.

Local check:

```bash
curl -i https://api.violawake.com/api/health
```

Failure drill:

```bash
cd /j/CLAUDE/PROJECTS/Wakeword
cp .env.production .env.production.health-test.bak
python -c "from pathlib import Path; p=Path('.env.production'); lines=p.read_text().splitlines(); p.write_text('\n'.join(('POSTGRES_PASSWORD=wrong-healthcheck-password' if line.startswith('POSTGRES_PASSWORD=') else line) for line in lines) + '\n')"
docker compose -f docker-compose.production.yml up -d backend
curl -i https://api.violawake.com/api/health
mv .env.production.health-test.bak .env.production
docker compose -f docker-compose.production.yml up -d backend
```

Expected during the drill: HTTP 503 with `database` in `failed_checks`.

## Stripe webhook end-to-end

Fast local proof is covered by `console/tests/test_billing.py`: it signs a fake Stripe event with the configured webhook secret and verifies the subscription tier moves from `free` to `developer`.

Live test with Stripe CLI:

```bash
stripe trigger checkout.session.completed --override 'data.object.client_reference_id=<your_user_id>'
```

If Stripe CLI is not installed, use a real-card launch test:

1. Register a throwaway account.
2. Buy the Developer plan with your own card.
3. Verify the Billing page and `/api/billing/subscription` show `tier=developer`.
4. Verify the confirmation email is sent.
5. Refund the charge in Stripe.

## Backups

Nightly backup target: private Cloudflare R2 bucket `violawake-backups`.

The backup script uploads two objects per run:

- `r2://violawake-backups/postgres/YYYY-MM-DD.sql.gz`
- `r2://violawake-backups/app-data/YYYY-MM-DD.tar.gz`

Required credentials:

- `CLOUDFLARE_ACCOUNT_ID`
- `CLOUDFLARE_API_TOKEN` with R2 bucket read/create permissions for privacy verification
- `VIOLAWAKE_BACKUP_R2_ACCESS_KEY_ID`
- `VIOLAWAKE_BACKUP_R2_SECRET_ACCESS_KEY`

If the script exits with a Cloudflare R2 `403`, create or extend the API token
with R2 bucket read/create permissions. Do not use `--allow-unverified-privacy`
for production backups unless you have manually confirmed the bucket has no
public domain or custom-domain access.

Manual run:

```bash
cd /j/CLAUDE/PROJECTS/Wakeword
python scripts/backup_to_r2.py --env-file .env.production --env-file /j/CLAUDE/PROJECTS/FewerJobs/.env
```

Access check without dumping user data:

```bash
python scripts/backup_to_r2.py --check-only --env-file .env.production --env-file /j/CLAUDE/PROJECTS/FewerJobs/.env
```

Windows Task Scheduler:

```powershell
schtasks /Create /TN "ViolaWake Nightly R2 Backup" /XML "J:\CLAUDE\PROJECTS\Wakeword\scripts\backup-task.xml"
```

## Restore procedure

Set the same R2 env vars used by `scripts/backup_to_r2.py`, then download the target date:

```bash
DATE=YYYY-MM-DD
mkdir -p restore
aws --endpoint-url "https://${CLOUDFLARE_ACCOUNT_ID}.r2.cloudflarestorage.com" s3 cp "s3://violawake-backups/postgres/${DATE}.sql.gz" "restore/${DATE}.sql.gz"
aws --endpoint-url "https://${CLOUDFLARE_ACCOUNT_ID}.r2.cloudflarestorage.com" s3 cp "s3://violawake-backups/app-data/${DATE}.tar.gz" "restore/${DATE}.tar.gz"
```

Restore Postgres:

```bash
gunzip -c "restore/${DATE}.sql.gz" | docker exec -i wakeword-postgres-1 psql -U violawake -d violawake
```

Restore `/app/data`:

```bash
docker exec wakeword-backend-1 sh -lc 'mkdir -p /app/data && find /app/data -mindepth 1 -maxdepth 1 -exec rm -rf {} +'
cat "restore/${DATE}.tar.gz" | docker exec -i wakeword-backend-1 tar -xzf - -C /app/data
docker compose -f docker-compose.production.yml restart backend
```

## Live mic test

Human-required:

1. Open `https://violawake.com`.
2. Log in as user 45.
3. Navigate to a wake-word demo page, or download the `.onnx` model and run:

```bash
violawake-cli stream-mic --threshold 0.5
```

4. Say `viola`.
5. Confirm the detector fires.
6. If it does not fire, retrain the model before launch.

## Support inbox auto-reply

Webhook URL:

```bash
https://api.violawake.com/api/email/inbound
```

Required environment variable:

```bash
VIOLAWAKE_EMAIL_INBOUND_WEBHOOK_SECRET=<random shared secret>
```

The endpoint expects JSON with a sender in `from`, `sender`, `reply_to`, or the same fields under `message`. It sends one Resend auto-reply per sender per 24 hours:

```text
Thanks for contacting ViolaWake. We received your message and aim to respond within 48 hours.
```

Cloudflare Email Routing handoff:

1. In Cloudflare, enable Email Routing for `violawake.com`. A DNS MX lookup currently returns no configured MX records, so inbound mail is not connected yet.
2. Create a Worker named `violawake-support-inbound`.
3. Add a Worker secret named `VIOLAWAKE_EMAIL_INBOUND_WEBHOOK_SECRET` with the same value configured on the backend.
4. Worker email handler:

```js
export default {
  async email(message, env, ctx) {
    const subject = message.headers.get("subject") || "";
    const messageId = message.headers.get("message-id") || "";
    const body = {
      from: message.from,
      to: message.to,
      subject,
      message_id: messageId
    };

    ctx.waitUntil(fetch("https://api.violawake.com/api/email/inbound", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "X-ViolaWake-Email-Secret": env.VIOLAWAKE_EMAIL_INBOUND_WEBHOOK_SECRET
      },
      body: JSON.stringify(body)
    }));
  }
}
```

5. In Email Routing, add a custom address route for `hello@violawake.com` and set the action to `Send to Worker` -> `violawake-support-inbound`.

## Human-required launch actions

- UptimeRobot: create the external monitor described above.
- Cloudflare R2: grant the API token R2 bucket permissions and create private R2 S3 access keys for backups.
- USPTO trademark search: use USPTO TESS manually before launch.
- Docker Desktop autostart: configure Windows boot behavior manually.
- Stripe brand-collision decision: decide whether shared Stripe account branding is acceptable.
- Cloudflare Email Routing: enable MX records and add the Worker route above for `hello@violawake.com`.
- Tier 3 container hardening: run `docker compose -f docker-compose.production.yml up -d --build` when ready to rebuild.
- Search Console: submit the final site after launch.
