# Vendored: cloudflare/agentic-inbox (ViolaWake support inbox)

This is a **vendored fork** of Cloudflare's open-source agentic inbox, adopted as
ViolaWake's customer-support inbox (drives `hello@violawake.com`). We take the
inbox/transport/approve-UI and draft replies with repo context via the founder-side
`support-inbox` skill hitting the worker's REST API / `/mcp` endpoint.

## Standalone — shares nothing with NOVVIOLA

ViolaWake is standalone (repo-root `CLAUDE.md` → "Relationship to NOVVIOLA"). This
deployment is a **separate worker instance** from NOVVIOLA's `agentic-inbox`:

| Thing | NOVVIOLA (useviola) | ViolaWake (this) |
|---|---|---|
| Worker name | `agentic-inbox` | `violawake-agentic-inbox` |
| Hostname | `support-inbox.useviola.com` | `support-inbox.violawake.com` |
| R2 bucket | `agentic-inbox` | `violawake-agentic-inbox` |
| Mailbox | `hello@useviola.com` | `hello@violawake.com` |
| Access app + service token | useviola AUD `cc7b3f…` | ViolaWake AUD `0ebf81ca…` (own token) |
| Creds env prefix | `VIOLA_INBOX_WORKER_*` | `VIOLAWAKE_INBOX_WORKER_*` |

The Cloudflare account (`368f46caaf71208619e7734b1823c0e1`) and Zero Trust org
(`violavoice.cloudflareaccess.com`) are the founder's and host both zones — that is
not shared *infrastructure*; the worker, bucket, tokens, and DOs are all distinct.
Never hardcode a NOVVIOLA hostname, bucket, or env var here.

## Upstream

- Repo: https://github.com/cloudflare/agentic-inbox
- Pinned commit: `48039bb6785af34e592c2966f87cde2b255c4c80` (2026-04-17)
- License: Apache-2.0 (see `LICENSE`) — Cloudflare, Inc.

Source re-vendored from NOVVIOLA's proven fork (which carries the modifications
below) at adoption time; the two copies are intentionally identical at the source
level and diverge only in `wrangler.jsonc` config.

## Why vendored (not a submodule)

We modify it, so it lives in-repo and tracked under `master` — no stealth untracked
infra. `node_modules` and build artifacts are gitignored; the source is committed.

## Modifications carried from upstream (all already in this source)

- **Outbound provider** — Cloudflare Email Service (`env.EMAIL.send()`).
- **Mail-loop / auto-responder suppression** — `classifyAutoMail()` in
  `workers/index.ts` (RFC 3834 Auto-Submitted, Precedence bulk/list, bounce/vacation
  subjects). Email always stored; only the agent auto-trigger is skipped.
- **Outbound audit log + kill-switch** — `workers/email-sender.ts`
  `sendEmailAudited()` at every send site. Kill-switch: write `{"paused":true}` to
  `config/outbound-paused.json` in R2 to halt all sends. Audit to
  `audit/outbound/<mailboxId>/YYYY-MM.jsonl`.
- **Full mailbox purge (GDPR erasure)** — `MailboxDO.purgeAllData()` + DELETE
  `/api/v1/mailboxes/:mailboxId` wipes DO SQLite + R2 blobs + settings object.
- **D4 Gmail copy-forward on receive** — `workers/app.ts` `email()` handler forwards
  every inbound to `EMAIL_FORWARD_COPY_TO` (a worker var) unless a per-mailbox
  `forwarding` setting overrides. This preserves the Gmail safety-net copy the old
  `violawake-support-email` worker provided (Cloudflare Email Routing allows one
  action per rule, so the rule routes to the worker and the worker re-forwards).
- **`.gitignore`** — negates `workers/lib/` and `app/lib/` defensively (the
  Python-convention `lib/` swept up the worker's TS source in NOVVIOLA; ViolaWake's
  root `.gitignore` has no bare `lib/`, but the negation is kept so re-vendoring is
  mechanical).
- **`wrangler.jsonc`** — ViolaWake production config (name, hostname, bucket,
  DOMAINS=violawake.com, EMAIL_ADDRESSES=[hello@violawake.com], Access AUD, Gmail
  copy). This is the only intentional source-level divergence from NOVVIOLA's fork.

## Relationship to the old `workers/support-email/` worker

`workers/support-email/` is the previous lightweight worker: it POSTed inbound
*metadata* to the Console `/api/email/inbound` (deduped auto-acknowledgement via
Resend) and forwarded the full message to Gmail. This agentic-inbox worker
supersedes it for `hello@violawake.com` — it captures full bodies into a
founder-readable inbox AND keeps the Gmail forward. The old worker + its Console
endpoint are left deployed (not deleted) as an instant rollback target; flipping the
Email Routing rule back to `violawake-support-email` restores the old behavior.

## Updating from upstream

1. `git clone --depth 1 https://github.com/cloudflare/agentic-inbox <tmp>`
2. Diff `<tmp>` against this dir; re-apply the modifications above.
3. Bump the pinned commit here and run the oracle before deploy.

## Deploy

See `DEPLOY.md` for the wrangler + Cloudflare Access + Email Routing runbook.
