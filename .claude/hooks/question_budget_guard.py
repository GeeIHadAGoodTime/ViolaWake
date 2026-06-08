#!/usr/bin/env python3
"""Stop hook: refuse to end a turn with a question Claude could have answered itself.

Pattern (top friction in Jay's transcripts, 274 messages across 20 sessions):
    Claude ends the turn with "want me to / should I / standing by / holding"
    instead of just choosing the best option and acting. Jay has typed
    "stop asking me questions with obvious answers" 3+ times across sessions.

What this hook does:
    1. Read the last assistant message text from the session transcript.
    2. Check whether the TAIL of the message contains a question-shaped phrase.
    3. If yes AND no legitimate-blocker context is present anywhere in the
       message, block the Stop with feedback telling Claude to choose and act
       (or add an explicit `[blocker: <reason>]` marker if it's a real blocker).

Allowlist (any one bypasses the block):
    - Explicit override markers: `[blocker: <reason>]` or `[blocked-on: <reason>]`
    - Irreversible ops: git push --force, git reset --hard, drop table/database, rm -rf
    - Spend: $-amount, "spend", "subscription change", "paid plan", "billing"
    - Legal/product/founder/business decision, compliance, regulatory
    - Credential the USER must provide: "your API key", "your credentials", "OAuth"
    - Physical user action: card on file, Windows Hello, biometric, physical presence,
      "your phone", "your machine", "your browser"

Kill-switch (env var, if the hook misbehaves):
    QUESTION_BUDGET_GUARD_OFF=1

Block-cap: after N consecutive blocks in the same session, the hook gives up and
passes (avoids the infinite-loop case where Claude can't find an acceptable
phrasing). N defaults to 3, override with QUESTION_BUDGET_GUARD_MAX_BLOCKS.

Per `feedback_stop_hook_shared_main_emergency.md`: this hook is SAFE in main
checkouts because it reads only the per-session transcript, not shared state.
The gatekeeper deferral does not apply.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path

if sys.platform == "win32":
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    except (AttributeError, OSError, ValueError):
        # Old Python / non-TextIOWrapper stream / already-detached stream.
        # Hook continues with default encoding; output may have unicode escapes.
        pass


# Question-shaped phrases that indicate Claude is asking instead of acting
# Each pattern is intentionally narrow to limit false positives
QUESTION_PATTERNS = [
    r"\bwant me to\b",
    r"\bshould i\b",
    r"\bwould you like\b",
    r"\bdo you want\b",
    r"\bshall i\b",
    r"\bcan i (?:proceed|continue|go|move|start|begin)\b",
    r"\bstanding by\b",
    r"\bawaiting (?:your|input|confirmation|decision|approval|response)\b",
    r"\bplease confirm\b",
    r"\bplease (?:let me know|advise|decide|specify|clarify)\b",
    r"\blet me know (?:if|whether|when|what|which|how|your)\b",
    r"\bwhich (?:option|approach|path|one|do you|would you)\b",
    r"\byour call\b",
    r"\bup to you\b",
    r"\bholding (?:until|on|for|pending)\b",
    r"\bblocked on (?:your|user|founder|jay)\b",
    r"\bneed your (?:input|decision|approval|go-?ahead|sign-?off)\b",
]
QUESTION_RE = re.compile("|".join(QUESTION_PATTERNS), re.IGNORECASE)

# Phrases that legitimize asking — any one anywhere in the message bypasses the block
BLOCKER_PATTERNS = [
    # Explicit overrides
    r"\[blocker:",
    r"\[blocked-on:",
    # Irreversible ops Claude shouldn't unilaterally do
    r"\bgit\s+push\s+--force\b",
    r"\bgit\s+reset\s+--hard\b",
    r"\bforce[-\s]push(?:ing|ed)?\s+to\s+main\b",
    r"\bdrop\s+(?:table|database|schema)\b",
    r"\brm\s+-rf\b",
    r"\bdelete\s+(?:the\s+)?(?:repo|repository|database|production|prod)\b",
    r"\brevert\s+(?:the\s+)?(?:merge|release|deploy|production)\b",
    # Spend
    r"\$\s*\d",
    r"\bspend\b",
    r"\bsubscription\s+change\b",
    r"\bpaid\s+plan\b",
    r"\bbilling\b",
    r"\bcredit\s+card\b",
    # Legal / product / business judgment
    r"\b(?:legal|compliance|regulatory|gdpr|hipaa|pci)\b",
    r"\b(?:product|founder|business|policy)\s+decision\b",
    # Credentials the user must provide
    r"\byour\s+(?:api\s+key|credentials?|password|oauth)\b",
    r"\bneed.*credential\b",
    r"\b(?:provide|share|paste)\s+(?:your\s+)?(?:api\s+key|token|password)\b",
    # Physical user action only the human can do
    r"\bcard\s+on\s+file\b",
    r"\bwindows\s+hello\b",
    r"\bbiometric\b",
    r"\bphysical\s+presence\b",
    r"\bat\s+(?:your|the)\s+machine\b",
    r"\byour\s+phone\b",
    r"\byour\s+browser\b",
    r"\bin\s+person\b",
    r"\bsms\s+code\b",
    r"\btwo-?factor\b",
    r"\bmfa\b",
]
BLOCKER_RE = re.compile("|".join(BLOCKER_PATTERNS), re.IGNORECASE)

# A question pattern is a real question only if it's the ENDING clause of the
# message — i.e. followed within ~80 chars by sentence-terminating punctuation
# (or end-of-text), with NO comma in between. A comma after the phrase signals
# a continuation clause ("...I can add X if you want me to, but I shipped Y").
QUESTION_TAIL = r"[^,]{0,80}?(?:[.!?]+\s*$|$)"
QUESTION_ENDING_RE = re.compile(
    "(?:" + "|".join(QUESTION_PATTERNS) + ")" + QUESTION_TAIL,
    re.IGNORECASE | re.MULTILINE,
)
MARKER_TTL = 12 * 60 * 60  # 12 hours
DEFAULT_MAX_BLOCKS = 3


def _kill_switched() -> bool:
    return bool(os.environ.get("QUESTION_BUDGET_GUARD_OFF", "").strip())


def _max_blocks() -> int:
    raw = os.environ.get("QUESTION_BUDGET_GUARD_MAX_BLOCKS", "").strip()
    if not raw:
        return DEFAULT_MAX_BLOCKS
    try:
        return max(1, int(raw))
    except ValueError:
        return DEFAULT_MAX_BLOCKS


def _read_last_assistant_text(transcript_path: str) -> str:
    """Return the text content of the most recent assistant message in the transcript."""
    p = Path(transcript_path)
    if not p.exists():
        return ""
    last_text = ""
    try:
        with p.open(encoding="utf-8", errors="replace") as f:
            for line in f:
                try:
                    o = json.loads(line)
                except (json.JSONDecodeError, ValueError):
                    continue
                if o.get("type") != "assistant":
                    continue
                msg = o.get("message") or {}
                if msg.get("role") != "assistant":
                    continue
                content = msg.get("content")
                if isinstance(content, str):
                    if content.strip():
                        last_text = content
                elif isinstance(content, list):
                    parts = []
                    for item in content:
                        if isinstance(item, dict) and item.get("type") == "text":
                            parts.append(item.get("text", "") or "")
                    joined = "\n".join(p for p in parts if p.strip())
                    if joined:
                        last_text = joined
    except OSError:
        return ""
    return last_text


def _marker_dir(cwd: Path) -> Path:
    return cwd / ".viola" / "agents" / "question_budget_markers"


def _marker_path(cwd: Path, session_id: str) -> Path | None:
    if not session_id:
        return None
    safe = re.sub(r"[^A-Za-z0-9_-]+", "_", session_id)
    return _marker_dir(cwd) / f"{safe}.json"


def _load_block_count(marker: Path | None) -> int:
    if marker is None or not marker.exists():
        return 0
    try:
        data = json.loads(marker.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return 0
        # Reset if stale
        if time.time() - float(data.get("updated", 0)) > MARKER_TTL:
            return 0
        return int(data.get("count", 0))
    except (OSError, json.JSONDecodeError, ValueError, TypeError):
        return 0


def _save_block_count(marker: Path | None, count: int) -> None:
    if marker is None:
        return
    try:
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(
            json.dumps({"count": count, "updated": time.time()}),
            encoding="utf-8",
        )
    except OSError:
        pass


def _cleanup_old_markers(directory: Path) -> None:
    if not directory.exists():
        return
    cutoff = time.time() - MARKER_TTL
    for path in directory.glob("*.json"):
        try:
            if path.stat().st_mtime < cutoff:
                path.unlink()
        except OSError:
            continue


def evaluate(text: str) -> tuple[bool, str]:
    """Return (should_block, matched_phrase).

    Block when: the message ENDS with a question pattern (the phrase is the
    closing clause, followed within ~80 chars by sentence-terminating
    punctuation or end-of-text, with no comma continuation in between) AND
    no BLOCKER context appears anywhere in the message.
    """
    if not text:
        return False, ""
    stripped = text.rstrip()
    if not stripped:
        return False, ""
    m = QUESTION_ENDING_RE.search(stripped)
    if not m:
        return False, ""
    if BLOCKER_RE.search(text):
        return False, ""
    # Extract the canonical short phrase that matched (for the error message)
    matched_short = QUESTION_RE.search(m.group(0))
    return True, (matched_short.group(0) if matched_short else m.group(0)[:40])


def _build_feedback(matched_phrase: str) -> str:
    return (
        "QUESTION_BUDGET_GUARD blocked this turn.\n"
        "\n"
        f"Your message ends with a question-shaped phrase: {matched_phrase!r}\n"
        "\n"
        "This is the most-corrected pattern in Jay's transcripts (274 user messages\n"
        'across 20 sessions; "stop asking me questions with obvious answers", "work\n'
        'autonomously" repeatedly). When the answer is obvious from the goal, repo\n'
        "state, or your previous authorization, just choose the best option and act.\n"
        "\n"
        "How to fix this turn:\n"
        "  - Pick the best default and DO it. Report what you did, not what you might do.\n"
        "  - If you genuinely need Jay's input, add an explicit marker anywhere in the\n"
        "    message: [blocker: <one-line-reason>]   e.g.  [blocker: $200 spend]\n"
        "\n"
        "Legitimate blocker classes (these bypass this hook automatically):\n"
        "  - Irreversible ops (git push --force, git reset --hard, drop table, rm -rf)\n"
        "  - Spend ($ amount / subscription change / paid plan / billing)\n"
        "  - Legal / compliance / product / founder / business decision\n"
        "  - Credentials only Jay can provide (his API key, OAuth)\n"
        "  - Physical user action (Windows Hello, card on file, biometric, MFA)\n"
        "\n"
        "Kill-switch (if this hook is wrong for your case):\n"
        "  set QUESTION_BUDGET_GUARD_OFF=1 in your environment.\n"
    )


def handle(payload: dict) -> tuple[int, str]:
    if _kill_switched():
        return 0, ""
    event_name = str(payload.get("hook_event_name") or "").lower()
    if event_name and event_name != "stop":
        return 0, ""
    transcript_path = str(payload.get("transcript_path") or "")
    if not transcript_path:
        return 0, ""
    text = _read_last_assistant_text(transcript_path)
    block, matched = evaluate(text)
    if not block:
        return 0, ""

    cwd = Path(str(payload.get("cwd") or Path.cwd())).resolve()
    session_id = str(payload.get("session_id") or "")
    marker = _marker_path(cwd, session_id)
    _cleanup_old_markers(_marker_dir(cwd))
    count = _load_block_count(marker) + 1
    _save_block_count(marker, count)

    if count > _max_blocks():
        # Give up to avoid loops; let Claude through with a soft notice on stderr
        sys.stderr.write(f"question_budget_guard: block-cap reached ({count} > {_max_blocks()}); passing.\n")
        return 0, ""

    return 2, _build_feedback(matched)


def main() -> int:
    try:
        raw = sys.stdin.read()
        payload = json.loads(raw) if raw.strip() else {}
    except (json.JSONDecodeError, ValueError, OSError):
        return 0
    if not isinstance(payload, dict):
        return 0
    rc, stderr_text = handle(payload)
    if stderr_text:
        sys.stderr.write(stderr_text)
    return rc


if __name__ == "__main__":
    sys.exit(main())
