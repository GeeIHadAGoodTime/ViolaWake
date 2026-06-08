---
name: plain-status
description: Structured plain-English status reply when Jay asks "where are we", "plain English", "progress report", "I'm confused", "how close", "are we going in circles". Leads with product reality, not lane/oracle/gate jargon. 52 user messages with "plain English" across 15 sessions — this is a top friction pattern.
---

# Plain Status

**Triggers** (any of these in Jay's message means he wants plain-English status, not jargon):
- "plain english" / "plain English"
- "where are we" / "where we're at"
- "progress report" / "status report"
- "how close are we" / "how much longer"
- "I'm confused" / "what does this mean"
- "are we going in circles" / "is this ritual"
- "what just happened"
- "summarize where things are"

## Required structure

Reply in EXACTLY this order. Translate any internal terms (lane, oracle, gate,
ratchet, convergence, ground-truth, etc.) into product reality AFTER the
plain-English claim — never before.

### 1. WHAT WORKS NOW (the product reality)

One paragraph in product terms: what can a user do today that works.
NOT: lane names. NOT: gate ids. NOT: "we converged on round 4."
DO: "Sign-in works. Music plays. Phone calls connect. Email sending works for
Gmail; Outlook is broken."

### 2. WHAT STILL FAILS

One paragraph in product terms: what a real user would see broken if they
tried it. Be specific — don't say "some auth issues", say "Apple sign-in fails
with a 502 in production."

### 3. WHAT SUCCESS TEST DECIDES "DONE"

Name the ONE oracle/test that, when it passes clean, proves the work is done.
Quote it verbatim from the source (e.g. the LLC live run, the launch readiness
checklist's final row). Don't invent your own definition.

### 4. WHY THE CURRENT WORK ISN'T RITUAL

One paragraph addressing Jay's recurring concern that "we're optimizing for our
test instead of real results." If you can't name the user-visible thing the
current work changes, say so — that itself is the answer.

### 5. NEXT PHYSICAL ACTION

ONE concrete action — file edit, command, dispatch, decision. Not three. Not a
phase plan. The next button to press.

## Banned in a plain-status reply

- Lane labels before the plain claim ("Lane B is at round 4 of 6")
- Oracle/gate/ratchet/convergence/launch-ready as headline words
- Stacked acronyms (LLC, OAC, CONT, etc.) without first defining each in plain
  language
- Round counts ("R10", "round 4 of 6") in the headline
- "Standing by" / "holding" / "want me to" anywhere (that's a different skill,
  but those phrases are categorically wrong in a plain-status reply too)

## Pattern for the lookup

Each line in the plain-status reply should map to one of these source-of-truth
queries (read fresh, don't paraphrase from memory):

| Section | Source |
|---|---|
| 1. What works | the live service + viola-runner command path |
| 2. What fails | the live behavioral trace, the most recent gate run |
| 3. Success test | `docs/LANE_LEDGER.md` for the lane's oracle |
| 4. Why not ritual | the user-visible thing the current code changes |
| 5. Next action | the next entry on the lane's todo OR the dispatch queue |

## Example

**BAD reply:**

> Lane L13 is at round 4 of the launch-readiness convergence cycle. The
> external-action policy registry is green; the trusted-proxy-ip ratchet
> shipped at SHA 4216eab5. R8-D residuals are at 2/45.

**GOOD reply:**

> **What works now:** Sign-in, music, phone calls, calendar, email (Gmail
> only). Desktop installer ships and runs clean.
>
> **What still fails:** Outlook email sending returns 502 in production.
> Spotify playback on Linux desktop doesn't start (Widevine carve-out is
> known, we're shipping without it).
>
> **Success test:** `docs/LANE_LEDGER.md` says lane L13 is done when the
> live regression run (`/live-regression`) shows green and the desktop
> installer passes the publish-update fresh-VM verification.
>
> **Why this isn't ritual:** The Outlook fix changes whether real users on
> the most common corporate email provider can use Viola at all. Without
> it, ~40% of installs are functionally broken.
>
> **Next action:** Read `docs/LANE_LEDGER.md` for the Outlook lane owner and
> ping them (or take it myself if unassigned).
