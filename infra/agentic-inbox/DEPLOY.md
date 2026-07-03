# Deploy runbook — violawake-agentic-inbox (ViolaWake support inbox)

Self-hosted Cloudflare Worker on the founder's account
(`368f46caaf71208619e7734b1823c0e1`), zone `violawake.com`
(`22a294f4dbc4b4fc5245cdeb2d3ba42b`). **Standalone** — separate worker / bucket /
Access app / service token from NOVVIOLA's `agentic-inbox` (see `VENDORED.md`).
First deployed + cut over live 2026-07-03.

## Deployed resources (for reference + rollback)

| Resource | Value |
|---|---|
| Worker | `violawake-agentic-inbox` |
| Custom hostname | `support-inbox.violawake.com` |
| R2 bucket | `violawake-agentic-inbox` |
| Access app id | `1d9223ff-eb94-491a-9ed0-b9a2da439795` |
| Access app AUD (`POLICY_AUD`) | `0ebf81cacaf8322a241af86ac776a59f6e96ce2a0ca0c70eb081059f152437d6` |
| Access service token id | `321bef77-ed3f-4070-9415-15e2bf98bbee` |
| Access policy id | `8026f9ba-8ac9-40cf-bcc5-e1a72fc0e822` |
| Team domain | `https://violavoice.cloudflareaccess.com` |
| Mailbox | `hello@violawake.com` |
| Gmail copy (`EMAIL_FORWARD_COPY_TO`) | `violavoiceassistant@gmail.com` |
| Email Routing rule (hello@) | id `5d6083a078794d4bb98d5e10a007b3cc` |
| Rollback target worker | `violawake-support-email` (old lightweight worker, left deployed) |

Service-token client-id/secret live in the repo `.env` as
`VIOLAWAKE_INBOX_WORKER_CF_CLIENT_ID` / `..._SECRET` (never committed).

## Prerequisites

- `CLOUDFLARE_API_TOKEN` in `.env` (Workers/R2/Email-Routing/DNS — present).
- An Access-scoped token for the Access step: the base token has
  `User:API Tokens:Edit`, so mint one via `POST /user/tokens` with permission
  groups Access Apps&Policies Write `1e13c5124ca64b72b1969a67e8829049`, Service
  Tokens Write `a1c0fec57cf94af79479a6d827fa518c`, Organizations Write
  `aed5acd922ae4fa68560cf0094e3e517`. Stored as `CLOUDFLARE_ACCESS_API_TOKEN`.
- `cd infra/agentic-inbox && npm install`.
- Cloudflare's WAF blocks the default tool User-Agent → always send a browser UA
  (`-H "User-Agent: Mozilla/5.0 ..."`) on API calls.

## 1. Bucket + deploy

```bash
export CLOUDFLARE_API_TOKEN=...; export CLOUDFLARE_ACCOUNT_ID=368f46caaf71208619e7734b1823c0e1
npx wrangler r2 bucket create violawake-agentic-inbox    # once
npm run deploy                                           # react-router build + wrangler deploy
```
Deploy binds: Durable Objects (MAILBOX/EMAIL_AGENT/EMAIL_MCP), `EMAIL`
(unrestricted Send Email — CF Email Service), `BUCKET` (R2), `AI` (Workers AI),
vars. It also creates the `support-inbox.violawake.com` custom domain.

## 2. Cloudflare Access (the worker fails closed without it)

HTTP + `/mcp` require a valid `cf-access-jwt-assertion`; only the inbound `email()`
handler is exempt (`workers/app.ts`). The self-hosted Access app over
`support-inbox.violawake.com` + a **service token** (non-identity policy) are
already provisioned (ids above); `POLICY_AUD`/`TEAM_DOMAIN` are baked into
`wrangler.jsonc`. To recreate:

1. `POST /accounts/<acct>/access/apps` `{type:self_hosted, domain:support-inbox.violawake.com}` → capture `aud` → set `POLICY_AUD`.
2. `POST /accounts/<acct>/access/service_tokens` `{name}` → capture `client_id`/`client_secret` (secret shown once) → `.env`.
3. `POST /accounts/<acct>/access/apps/<app_id>/policies` `{decision:non_identity, include:[{service_token:{token_id}}]}`.
4. Redeploy if `POLICY_AUD`/`TEAM_DOMAIN` changed.

Drive MCP/REST with `CF-Access-Client-Id` / `CF-Access-Client-Secret` headers.

## 3. Email Routing (inbound) — cut over hello@, keep Gmail alive

Zone `22a294f4dbc4b4fc5245cdeb2d3ba42b`. The existing rule
`5d6083a078794d4bb98d5e10a007b3cc` (`hello@violawake.com`) is flipped from the old
`violawake-support-email` worker to this one. The worker forwards the Gmail copy
itself (D4), so nothing to Gmail is lost.

```bash
ZONE=22a294f4dbc4b4fc5245cdeb2d3ba42b
# create the mailbox object first (so inbound is accepted + Gmail copy default applies)
printf '%s' '{"fromName":"ViolaWake Support","forwarding":{"enabled":false,"email":""},"signature":{"enabled":false,"text":""},"autoReply":{"enabled":false,"subject":"","message":""}}' > /tmp/mb.json
npx wrangler r2 object put "violawake-agentic-inbox/mailboxes/hello@violawake.com.json" --file=/tmp/mb.json --content-type=application/json --remote
# flip the hello@ rule -> violawake-agentic-inbox
curl -X PUT -H "Authorization: Bearer $CLOUDFLARE_API_TOKEN" -H "Content-Type: application/json" -H "User-Agent: Mozilla/5.0" \
  "https://api.cloudflare.com/client/v4/zones/$ZONE/email/routing/rules/5d6083a078794d4bb98d5e10a007b3cc" \
  --data '{"name":"Support inbox (agentic-inbox)","enabled":true,"matchers":[{"type":"literal","field":"to","value":"hello@violawake.com"}],"actions":[{"type":"worker","value":["violawake-agentic-inbox"]}]}'
```

> Note: violawake.com Email Routing MX (`route1/2/3.mx.cloudflare.net`) is already
> live; the zone `status:unconfigured` flag is cosmetic (mail flows). Do not "enable"
> via the wizard blindly — it can reset rules. Only touch the specific rule.

For **all-address** capture (any `@violawake.com` → inbox + Gmail), flip the zone
catch-all (`dfae6a6aa3b042708571af5c3e9af8d2`) to `{"actions":[{"type":"worker",
"value":["violawake-agentic-inbox"]}],"enabled":true}` — the worker funnels any
recipient into the primary mailbox and forwards the Gmail copy. Left as **drop**
today (only `hello@` is captured) — enabling catch-all also captures spam to random
addresses. Founder decision if/when desired.

## 4. Verify

- Send a real test email to `hello@violawake.com`; assert (a) it appears via
  `GET /api/v1/mailboxes/hello%40violawake.com/emails?folder=inbox` (with the service
  token headers) and (b) a Gmail copy lands in `violavoiceassistant@gmail.com`.
- Auth check: no token → `403`; service token → `200`.
- Outbound: `POST .../emails/{id}/reply` to a founder-controlled address → `{"status":"sent"}`.

## Rollback

Flip rule `5d6083a078794d4bb98d5e10a007b3cc` back to
`{"actions":[{"type":"worker","value":["violawake-support-email"]}]}` — the old
worker + Console `/api/email/inbound` are still deployed and resume immediately. To
fully remove: `npx wrangler delete violawake-agentic-inbox`; the Gmail forward is
preserved by the old worker.
