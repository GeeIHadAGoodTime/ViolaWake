# Operations Runbook

Procedures for live operational changes that require dashboards or external accounts. Anything in this file is something only the human operator can do — agents can't.

Cross-reference: `docs/DEPLOYMENT.md` (mechanics), `docs/PRODUCTION_STATUS.md` (current state).

---

## Configure Resend (turn on real email)

**Status as of 2026-05-07:** Resend NOT configured. `VIOLAWAKE_RESEND_API_KEY` is unset; backend's email service runs in `enabled=False` mode → new users are auto-verified at first login and no real email is ever sent. `send_verification_email`, `send_team_invite`, `send_training_complete`, `send_quota_warning`, `send_password_reset`, `send_existing_account_notice` all silently no-op.

### Why turn it on

Without Resend:
- Users register but never receive a "Verify your email" link → workaround is the silent auto-verify, which is fine for dev but misleading in production.
- Team invites return the invite token in the JSON body; the inviter has to manually paste it to the invitee.
- Password reset returns success but no link is sent → users can't reset their own password.
- Training-complete and quota-warning emails never fire → users learn about completion only by polling the UI.

### Steps

1. **Create the Resend account** (skip if you already have one)
   - https://resend.com → Sign up with the same Google / email account that owns `violawake.com`.

2. **Add the sending domain**
   - Resend dashboard → **Domains** → **Add Domain** → enter `violawake.com`.
   - Resend will display 3-4 DNS records to add (DKIM `resend._domainkey`, SPF `_resend`, MX, optionally a `Return-Path` record).

3. **Add the DNS records in Cloudflare**
   - Cloudflare dashboard → `violawake.com` → **DNS** → **Records**.
   - Add each record from Resend exactly as shown. **Set proxy status to DNS only (grey cloud), not Proxied (orange cloud)** — Resend needs to read these records directly.
   - DKIM record TTL: leave on Auto.

4. **Verify the domain in Resend**
   - Back in the Resend dashboard, click **Verify DNS Records**.
   - Wait for all records to show ✅. Can take 1–60 minutes for DNS propagation.

5. **Create an API key**
   - Resend → **API Keys** → **Create API Key**.
   - Name: `violawake-production`.
   - Permission: **Full access** (or **Sending access** restricted to the verified domain).
   - Copy the key (`re_...`) once — Resend won't show it again.

6. **Set the env var**
   ```bash
   # On the host machine running wakeword-backend-1
   cd /j/CLAUDE/PROJECTS/Wakeword
   # Append (or replace existing) line in .env.production:
   echo "VIOLAWAKE_RESEND_API_KEY=re_<paste_key_here>" >> .env.production
   ```
   Make sure no other `VIOLAWAKE_RESEND_API_KEY=` line exists in `.env.production`. If there's an old empty one, delete it.

7. **Restart the backend**
   ```bash
   cd /j/CLAUDE/PROJECTS/Wakeword
   docker compose -f docker-compose.production.yml up -d backend
   docker inspect wakeword-backend-1 --format='{{.State.Health.Status}}'   # expect: healthy
   ```

8. **Verify a real email actually sends**
   ```bash
   # Register a fresh account using YOUR real inbox, not a fake address
   curl -sS -X POST -H "Content-Type: application/json" \
     -d '{"email":"YOUR_REAL_EMAIL@example.com","password":"TestPass123!","name":"Verify Test"}' \
     https://api.violawake.com/api/auth/register
   ```
   Check your inbox within 60 seconds for "ViolaWake — Verify your email". If the email lands, Resend is configured correctly.

   If it doesn't land:
   - Check Resend dashboard → **Logs** for the most recent send attempt + error.
   - Check the backend logs: `docker logs wakeword-backend-1 --tail 100 | grep -i resend`.
   - Common failure: domain not yet verified (DNS still propagating). Wait 30 min and re-try.

9. **Update production status**
   - Edit `docs/PRODUCTION_STATUS.md`:
     - Move the Resend bullet from "NOT verified" to "Verified end-to-end" with today's date.
     - Update the "Operational levers" table row.

### Rotation

When a Resend key leaks (e.g., gets dumped in a transcript), rotate immediately:
1. Resend → API Keys → revoke the leaked key.
2. Create a new one (same name, same permissions).
3. Update `.env.production` with the new key.
4. `docker compose up -d backend` to pick up the new env.

---

## Decide Stripe mode (TEST → LIVE)

**Status as of 2026-05-07:** Stripe is configured in **TEST mode**. `VIOLAWAKE_STRIPE_SECRET_KEY` is `sk_test_*`, prices are test-mode prices, the checkout URLs returned are `cs_test_*`. No real money moves.

### When to flip

Flip to LIVE when ALL of these are true:
- You have a verified business identity in Stripe (Stripe → Settings → Account details, fully filled in for your country).
- You've taxed/legally registered for the jurisdictions you'll sell in (or are using Stripe's tax handling explicitly).
- Your terms of service + privacy policy + refund policy are public on `violawake.com` (already are: `/terms` and `/privacy`).
- You've test-driven the full flow at least once in TEST mode end-to-end (register → checkout → quota update → cancel) and confirmed it works.
- You're prepared for real customers to be able to register, pay, and demand support.

If any of those are no, **stay in TEST mode**. Going live with broken billing is harder to clean up than waiting another week.

### Pre-flight (do these in TEST mode first to confirm flow works)

1. **Drive a real test checkout** end-to-end with Stripe's test card.
   - Register a fresh account at `https://violawake.com/register`.
   - Click "Get Started" on the Developer tier.
   - In Stripe Checkout, fill: card `4242 4242 4242 4242`, exp `12/34`, CVC `123`, ZIP `12345`. Submit.
   - You should redirect back to the dashboard. Within ~30 seconds, `GET /api/billing/subscription` (or the Billing page in the UI) should show `tier=developer`.
   - If it doesn't, the webhook isn't firing or isn't being processed — fix that BEFORE going live.

2. **Confirm webhook is reachable.**
   - Stripe dashboard → Developers → Webhooks → confirm there's an endpoint pointing to `https://api.violawake.com/api/billing/webhook` and it's enabled for at least these events:
     - `checkout.session.completed`
     - `customer.subscription.created`
     - `customer.subscription.updated`
     - `customer.subscription.deleted`
     - `invoice.payment_failed`
   - The webhook signing secret must match `VIOLAWAKE_STRIPE_WEBHOOK_SECRET` in `.env.production`. If they don't match, all webhook posts fail signature verification with HTTP 400.

3. **Confirm test mode actually works on the live site.** (See step 1 above.)

### Steps to flip to LIVE

1. **Activate the Stripe account.** Stripe → top-right toggle "Test mode" → switch off → Stripe will require business identity verification if not already done. Complete it.

2. **Create LIVE-mode products and prices.**
   - Stripe → Products → Add product (separately for each tier you want to charge).
   - Developer: $29/month, recurring monthly.
   - Business: $99/month, recurring monthly.
   - Save each product and copy its **Price ID** (starts with `price_...`).
   - These are SEPARATE from your test-mode price IDs. You cannot use a test `price_*` in live mode and vice versa.

3. **Get LIVE API keys.**
   - Stripe → Developers → API keys (in LIVE mode).
   - Copy the **Secret key** (`sk_live_...`). This appears once.
   - Copy the **Publishable key** (`pk_live_...`).

4. **Create a LIVE webhook endpoint.**
   - Stripe → Developers → Webhooks (LIVE mode) → Add endpoint.
   - URL: `https://api.violawake.com/api/billing/webhook`.
   - Listen to the same events as the test webhook (see Pre-flight #2).
   - After creating, click "Reveal signing secret" — copy it (`whsec_...`).

5. **Update `.env.production`.** Replace four env vars at once. Bad partial updates leak charges to wrong customers.
   ```
   VIOLAWAKE_STRIPE_SECRET_KEY=sk_live_<your_live_secret_key>
   VIOLAWAKE_STRIPE_WEBHOOK_SECRET=whsec_<your_live_webhook_signing_secret>
   VIOLAWAKE_STRIPE_PRICE_DEVELOPER=price_<live_developer_price_id>
   VIOLAWAKE_STRIPE_PRICE_BUSINESS=price_<live_business_price_id>
   ```

6. **Restart the backend.**
   ```bash
   cd /j/CLAUDE/PROJECTS/Wakeword
   docker compose -f docker-compose.production.yml up -d backend
   ```

7. **Verify with a $0.50 real charge** before announcing.
   - Use your own real card.
   - Go through full checkout on `violawake.com`.
   - Confirm: Stripe dashboard shows the live charge, your account in Console shows tier=developer, the subscription auto-cancels in Stripe (so you don't get billed again).
   - Refund yourself in Stripe.

8. **Update `docs/PRODUCTION_STATUS.md`:**
   - "Stripe mode" row: TEST → LIVE.
   - Add a "Verified end-to-end" line: "$0.50 self-test on YYYY-MM-DD: charge captured, subscription created, refund processed."

### Rollback

If LIVE billing breaks within hours:
1. Disable the LIVE webhook endpoint (Stripe → Webhooks → toggle off). New events queue at Stripe but won't hit the broken backend.
2. Switch `.env.production` back to TEST keys.
3. Restart backend.
4. Manually refund / void any charges that landed during the broken window.
5. Diagnose with `docker logs wakeword-backend-1 --tail 500 | grep -i stripe` and Stripe → Webhooks → recent deliveries (each shows the request body + your response).

### Rotation

If a Stripe LIVE key leaks:
1. Stripe → Developers → API keys → roll the key (Stripe gives you 24h overlap to update env).
2. Update `.env.production` with the new key.
3. `docker compose up -d backend`.
4. Old key remains valid for 24h then auto-revokes.

For webhook signing secret rotation: create a NEW webhook endpoint with the new secret, update `.env.production`, restart, then disable the old endpoint after confirming the new one is delivering.

---

## When to update this runbook

Add a section here when an operational change requires a dashboard the user must visit (Cloudflare, Resend, Stripe, GitHub, PyPI, registrar, Sentry, etc.). Don't put runbook steps in `docs/DEPLOYMENT.md` — that file is for repeatable code/CLI deploys; this file is for human-mediated config changes.
