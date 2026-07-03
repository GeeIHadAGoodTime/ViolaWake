---
name: support-inbox
description: Read inbound customer support email captured by the ViolaWake agentic-inbox Cloudflare Worker (hello@violawake.com), draft context-aware replies grounded in the customer's account/Stripe state, show the batch, and send ONLY the ones the founder approves — via the worker's audited outbound, never auto-send. Triggers on "check the support inbox", "reply to customers", "any support emails", "draft replies to the inbox", "/support-inbox".
---

# ViolaWake Support Inbox — draft & batch-reply to customers

## What this is

Inbound mail to **hello@violawake.com** is captured by the vendored **agentic-inbox
Cloudflare Worker** (`infra/agentic-inbox/`, live at `support-inbox.violawake.com`):
full bodies live in per-mailbox Durable Objects + R2. This skill is how the founder —
through you (Claude Code) — reads those threads and replies, threaded into the
customer's original conversation. Every inbound is also forwarded to the founder's
Gmail (`violavoiceassistant@gmail.com`) as a safety-net copy by the worker itself.

**This is a FOUNDER OPS tool, not a product feature.** ViolaWake (the SDK/console we
ship) never reads, drafts, or sends customer-support email. This skill is you
operating the support inbox on the founder's behalf.

**Standalone.** This is a ViolaWake-only deployment — a separate worker / R2 bucket /
Access app / service token from NOVVIOLA's `agentic-inbox`. Never reach into
NOVVIOLA's inbox, env vars (`VIOLA_INBOX_WORKER_*`), or systems from here. All creds
here are `VIOLAWAKE_INBOX_WORKER_*` in the repo `.env`.

**Autonomy = batch-send-on-go-ahead.** You draft, you show the batch, and you send
ONLY what the founder explicitly approves. Customer email is outward-facing and cannot
be unsent — there is no auto-send path. The founder's go-ahead IS the gate.

## Auth — every worker call

The worker HTTP API sits behind **Cloudflare Access**; authenticate with the service
token. Cloudflare's WAF blocks the default tool User-Agent, so always send a browser UA.

- Base URL: `https://support-inbox.violawake.com`
- Mailbox: `hello@violawake.com` (URL-encode the `@` → `hello%40violawake.com`)
- Headers (read the values from the repo `.env`, never hardcode):
  - `CF-Access-Client-Id: $VIOLAWAKE_INBOX_WORKER_CF_CLIENT_ID`
  - `CF-Access-Client-Secret: $VIOLAWAKE_INBOX_WORKER_CF_CLIENT_SECRET`
  - `User-Agent: Mozilla/5.0 (...)`

## The loop

### 1. Read new threads

```bash
MB=hello%40violawake.com
curl -s -H "CF-Access-Client-Id: $VIOLAWAKE_INBOX_WORKER_CF_CLIENT_ID" \
        -H "CF-Access-Client-Secret: $VIOLAWAKE_INBOX_WORKER_CF_CLIENT_SECRET" \
        -H "User-Agent: Mozilla/5.0" \
  "https://support-inbox.violawake.com/api/v1/mailboxes/$MB/emails?folder=inbox&limit=50"
```

Each summary has `id`, `subject`, `sender`, `date`. Full body of one thread:
`GET /api/v1/mailboxes/$MB/emails/{id}`.

### 2. Attribute the sender (who is this customer?)

Give the draft real context. ViolaWake's own systems are the truth — never NOVVIOLA's:

- **Console Postgres** (`wakeword-postgres-1`): look up the account by sender email
  (users table) for plan tier, verification state, training-job history. Query through
  the Console's admin surface or a read-only `docker compose exec postgres psql`, never
  by writing to prod.
- **Stripe** (test mode today — `VIOLAWAKE_STRIPE_*`): verify live billing state before
  stating anything about a subscription, refund, or invoice. Use the Stripe MCP if
  configured, else the Stripe dashboard for the ViolaWake account.

A paying Developer/Business customer, a churned one, and a free/trial user each warrant
a different tone and priority — reflect it in the draft.

### 3. Draft a reply per thread

Ground every reply in the **actual message** + the customer's **real account state** +
repo knowledge (the SDK docs, `docs/PRD.md`, the benchmark numbers with their corpus +
SHA + threshold caveats — see CLAUDE.md "Don't manufacture accuracy claims"). Never
fabricate account facts — look them up or flag for the founder. Common classes:

- **SDK / install** ("pip install violawake fails", "ModelNotFoundError") — point to
  the `[oww]` extra + `download_models()` quickstart (CLAUDE.md Deploy paths → SDK).
- **Console / training** (sign-up, sample upload, training job stuck, model download) —
  check the live Console + job state before promising a fix.
- **Billing / subscription** — Stripe is the source of truth.
- **Benchmark / comparison questions** — only quote numbers with their corpus + SHA +
  threshold; never cross-quote the production-eval `d'=8.577` as a public-benchmark
  number.

### 4. Show the batch, wait for go-ahead

Present a compact table — `id | customer (plan) | subject | one-line summary of your
draft` — then each full draft. **Stop and wait.** The founder replies with `send all`,
`send 1,3,5`, edits, or skips. Send nothing until that explicit instruction.

### 5. Send approved replies (audited, threaded, kill-switch)

```bash
curl -s -X POST \
  -H "CF-Access-Client-Id: $VIOLAWAKE_INBOX_WORKER_CF_CLIENT_ID" \
  -H "CF-Access-Client-Secret: $VIOLAWAKE_INBOX_WORKER_CF_CLIENT_SECRET" \
  -H "User-Agent: Mozilla/5.0" -H "Content-Type: application/json" \
  "https://support-inbox.violawake.com/api/v1/mailboxes/$MB/emails/{id}/reply" \
  --data '{
    "to": "<customer@example.com>",
    "from": {"email": "hello@violawake.com", "name": "ViolaWake Support"},
    "subject": "Re: <original subject>",
    "html": "<p>...</p>",
    "text": "..."
  }'
```

The worker owns threading (sets `In-Reply-To`/`References`), sends via **Cloudflare
Email Service** through `sendEmailAudited` (every send written to the outbound audit log
and gated by the kill-switch), and files the reply in the `sent` folder. Success returns
`{"status": "sent"}`.

**Kill-switch:** if an operator wrote `{"paused": true}` to
`config/outbound-paused.json` in the worker's R2 bucket (`violawake-agentic-inbox`), the
send fails with `OutboundPausedError` — surface it, do not retry blindly.

### 6. Confirm

Report which ids sent and surface any failure **verbatim**.

## Guardrails

- **Never auto-send.** No "send all without asking", no scheduled send. The founder's
  explicit per-batch go-ahead is mandatory. Customer email is outward-facing + unsendable.
- **Founder ops tool, not a product feature.** Never route any of this through the SDK,
  the Console product code, or present it as a ViolaWake capability.
- **Standalone.** Never touch NOVVIOLA's inbox/creds/systems from here.
- **Don't widen scope.** Reply to existing inbound threads only — no cold-email or
  bulk-mail.
- **Don't fabricate account facts.** ViolaWake's Postgres + Stripe are the truth.
- **Enablement.** The pipeline is live when `VIOLAWAKE_INBOX_WORKER_CF_CLIENT_ID` /
  `..._SECRET` are set and `hello@violawake.com` Email Routing delivers to the
  `violawake-agentic-inbox` worker. If the inbox is empty and you expect mail, verify
  routing before assuming "no new email."

## Notes

- The worker also exposes an **MCP** endpoint (`/mcp`, same service token) and a web
  inbox UI at `support-inbox.violawake.com` for quick manual replies.
- Deploy / rollback runbook + all provisioned resource ids: `infra/agentic-inbox/DEPLOY.md`.
- Rollback (restore the old lightweight forwarder): flip the `hello@violawake.com` Email
  Routing rule back to the `violawake-support-email` worker — see DEPLOY.md.
