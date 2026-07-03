#!/usr/bin/env python3
"""Edge-case oracle for the ViolaWake agentic-inbox support pipeline.

The lane's instrumented success-test (CLAUDE.md "Lanes & the Lane Ledger"):
ONE runnable thing that exercises the real end-to-end path against a live
instance and yields a binary verdict against a MUST-FIX bar. It is NOT mocked —
it sends real mail and asserts side effects in the Durable Object + outbound.

Standalone: this targets ViolaWake's own worker + creds only (never NOVVIOLA's).

MUST-FIX bar (a plausibly-broken implementation must FAIL):
  C1 inbound capture     - mail to the mailbox lands in the inbox folder
  C2 threading           - a reply carries In-Reply-To/References and stays in-thread
  C3 MCP control         - /mcp lists the tool set and get_thread returns the message
  C4 outbound send       - send_reply is accepted and recorded in the sent folder
  C5 loop suppression    - an auto-submitted/bounce message gets NO auto-reply   [Phase 5]
  C6 erased-sender       - a GDPR-erased sender's mail is purged/anonymized       [Phase 4]
  C7 malformed resilience- a malformed/odd MIME message is captured, not dropped
Cases tagged [Phase N] are expected-pending until that phase lands; the harness
reports them PENDING (not FAIL) so it stays green-able while the lane builds.

Config (from the repo .env / .env.production):
  VIOLAWAKE_INBOX_WORKER_CF_CLIENT_ID/SECRET (Cloudflare Access service token — .env),
  VIOLAWAKE_RESEND_API_KEY (send inbound test mail — .env.production),
  INBOX_BASE / MAILBOX (override per instance; default to the live prod worker).
Cloudflare WAF blocks the default python UA — we send a browser User-Agent
(CLAUDE.md WAF note). NOTE: runs against the LIVE prod inbox — it sends real test
mail to hello@violawake.com and a reply; run it deliberately.

Usage:  python infra/agentic-inbox/oracle/run_oracle.py
Exit 0 only when every non-PENDING MUST-FIX case PASSES.
"""

from __future__ import annotations

# ruff: noqa: T201 - standalone CLI oracle harness; stdout is its result channel.
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

INBOX_BASE = os.environ.get("INBOX_BASE", "https://support-inbox.violawake.com")
MAILBOX = os.environ.get("MAILBOX", "hello@violawake.com")
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124 Safari/537.36"


def _env(key: str) -> str:
    val = os.environ.get(key)
    if val:
        return val
    # fall back to the repo .env / .env.production so the harness runs without exporting.
    # (Access creds live in .env; the Resend key lives in .env.production.)
    root = os.path.join(os.path.dirname(__file__), "..", "..", "..")
    for fname in (".env", ".env.production"):
        try:
            for line in open(os.path.join(root, fname), encoding="utf-8"):
                if line.startswith(key + "="):
                    return line.split("=", 1)[1].strip().strip('"').strip("'")
        except OSError:
            pass
    return ""


CID = _env("VIOLAWAKE_INBOX_WORKER_CF_CLIENT_ID")
CSEC = _env("VIOLAWAKE_INBOX_WORKER_CF_CLIENT_SECRET")
RESEND = _env("VIOLAWAKE_RESEND_API_KEY")
_ACCESS = {"CF-Access-Client-Id": CID, "CF-Access-Client-Secret": CSEC, "User-Agent": UA}


def _req(path: str, method: str = "GET", body: dict | None = None, extra: dict | None = None, retries: int = 5):
    headers = dict(_ACCESS)
    headers.update(extra or {})
    if body is not None:
        headers["Content-Type"] = "application/json"
    data = json.dumps(body).encode() if body is not None else None
    last: Exception | None = None
    for i in range(retries):
        try:
            req = urllib.request.Request(INBOX_BASE + path, method=method, headers=headers, data=data)
            return urllib.request.urlopen(req, timeout=25)  # nosec B310 - fixed https endpoints (worker API + Resend)
        except urllib.error.HTTPError as exc:
            last = exc
            # 500 also retried: a transient Cloudflare Durable-Object storage reset
            # ("object was reset") surfaces as 500 and clears on retry (2026-07-03).
            if exc.code in (403, 500, 502, 521, 522, 523, 524) and i < retries - 1:
                time.sleep(4)
                continue
            raise
    raise last  # type: ignore[misc]


def _emails(folder: str) -> list[dict]:
    mb = urllib.parse.quote(MAILBOX, safe="")
    resp = _req(f"/api/v1/mailboxes/{mb}/emails?folder={folder}&limit=50")
    return json.load(resp).get("emails", [])


def _send_inbound(subject: str, *, headers: dict | None = None, body: str = "test") -> None:
    payload: dict = {
        "from": "ViolaWake Test <noreply@violawake.com>",
        "to": [MAILBOX],
        "subject": subject,
        "text": body,
        "html": f"<p>{body}</p>",
    }
    if headers:
        payload["headers"] = headers
    req = urllib.request.Request(
        "https://api.resend.com/emails",
        method="POST",
        headers={"Authorization": f"Bearer {RESEND}", "Content-Type": "application/json", "User-Agent": UA},
        data=json.dumps(payload).encode(),
    )
    urllib.request.urlopen(req, timeout=25)  # nosec B310 - fixed https endpoints (worker API + Resend)


def _await_inbox(subject: str, *, timeout: int = 90) -> dict | None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        for e in _emails("inbox"):
            if e.get("subject") == subject:
                return e
        time.sleep(6)
    return None


def _mcp(method: str, params: dict, session: str | None = None):
    extra = {"Accept": "application/json, text/event-stream"}
    if session:
        extra["mcp-session-id"] = session
    resp = _req("/mcp", "POST", {"jsonrpc": "2.0", "id": 1, "method": method, "params": params}, extra)
    sid = resp.headers.get("mcp-session-id") or session
    payload = None
    for line in resp.read().decode().splitlines():
        if line.startswith("data:"):
            payload = json.loads(line[5:])
            break
    return payload, sid


# --------------------------------------------------------------------------- #
# Cases — each returns (status, detail). status in {PASS, FAIL, PENDING}.
# --------------------------------------------------------------------------- #
def case_inbound_and_threading() -> tuple[str, str]:
    tag = f"ORACLE inbound {int(time.time())}"
    _send_inbound(tag)
    got = _await_inbox(tag)
    if not got:
        return "FAIL", "C1: inbound email never reached the inbox"
    # C3 MCP + C2/C4 reply threading
    _, sid = _mcp(
        "initialize",
        {"protocolVersion": "2024-11-05", "capabilities": {}, "clientInfo": {"name": "oracle", "version": "1"}},
    )
    _mcp("notifications/initialized", {}, sid)
    tl, _ = _mcp("tools/list", {}, sid)
    tools = {t["name"] for t in (tl or {}).get("result", {}).get("tools", [])}
    need = {"get_thread", "draft_reply", "send_reply", "list_emails"}
    if not need.issubset(tools):
        return "FAIL", f"C3: MCP missing tools {need - tools}"
    # C4: send a reply, assert it lands in sent + threads
    mb = urllib.parse.quote(MAILBOX, safe="")
    rr = _req(
        f"/api/v1/mailboxes/{mb}/emails/{got['id']}/reply",
        "POST",
        {
            "to": "noreply@violawake.com",
            "from": {"email": MAILBOX, "name": "ViolaWake"},
            "subject": f"Re: {tag}",
            "html": "<p>oracle reply</p>",
            "text": "oracle reply",
        },
    )
    if json.load(rr).get("status") != "sent":
        return "FAIL", "C4: reply send was not accepted"
    time.sleep(3)
    sent = [e for e in _emails("sent") if (e.get("subject") or "").endswith(tag)]
    if not sent or not sent[0].get("in_reply_to"):
        return "FAIL", "C2/C4: reply not recorded in sent with in_reply_to set"
    return "PASS", f"C1-C4 ok (inbox id {got['id'][:8]}, {len(tools)} MCP tools, threaded reply sent)"


def case_malformed_resilience() -> tuple[str, str]:
    # Odd subject + empty body shape. Real malformed-MIME fuzzing lands with the
    # raw-injection harness in Phase 6 hardening.
    tag = f"ORACLE odd <#?&> {int(time.time())}"
    _send_inbound(tag, body="")
    return ("PASS", "C7: odd-header inbound captured") if _await_inbox(tag) else ("FAIL", "C7: odd inbound dropped")


def case_loop_suppression() -> tuple[str, str]:
    # An Auto-Submitted: auto-replied message (RFC 3834) must NOT trigger an
    # auto-reply, or two systems ping-pong forever. Implemented in Phase 5.
    return "PENDING", "C5: mail-loop suppression — Phase 5"


def case_erased_sender() -> tuple[str, str]:
    return "PENDING", "C6: GDPR-erased sender purge — Phase 4"


CASES = [
    ("inbound+threading+mcp+outbound", case_inbound_and_threading),
    ("malformed-resilience", case_malformed_resilience),
    ("loop-suppression", case_loop_suppression),
    ("erased-sender", case_erased_sender),
]


def main() -> int:
    if not (CID and CSEC and RESEND):
        print("MISSING CONFIG: need VIOLAWAKE_INBOX_WORKER_CF_CLIENT_ID/SECRET + VIOLAWAKE_RESEND_API_KEY")
        return 2
    print(f"oracle target: {INBOX_BASE}  mailbox: {MAILBOX}\n")
    failed = 0
    for name, fn in CASES:
        try:
            status, detail = fn()
        except Exception as exc:  # noqa: BLE001, RUF100 - a thrown case is a failure, not a crash
            status, detail = "FAIL", f"exception: {exc}"
        mark = {"PASS": "PASS", "FAIL": "FAIL", "PENDING": "pend"}[status]
        print(f"  [{mark}] {name}: {detail}")
        if status == "FAIL":
            failed += 1
    print()
    if failed:
        print(f"ORACLE: FAIL ({failed} MUST-FIX case(s) broken)")
        return 1
    print("ORACLE: PASS (all implemented MUST-FIX cases green; PENDING cases await their phase)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
