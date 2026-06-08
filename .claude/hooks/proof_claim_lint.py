"""Stop hook: refuse a turn that asserts proof claims without evidence anchors.

Pattern (from Jay's transcripts: `root cause` 81×, `full trace` 17×, plus
repeated "n=1 is not a pattern" corrections, "your stale-bundle conclusion is
incomplete... cloud_app.py:3272..." style corrections):

    Claude writes "root cause is X" or "tests are green" or "we're done"
    without an evidence anchor (current SHA, file:line, command run this
    session, trace id, artifact path). Jay has to push back and demand the
    proof. This hook makes the anchor mandatory in the same paragraph.

What this hook does:
    1. Read the last assistant message text.
    2. For each high-confidence proof claim pattern that fires, check the
       SAME PARAGRAPH for at least one evidence anchor.
    3. If no anchor, block with feedback naming the offending claim and the
       expected anchor categories.

Allowlist (any one anywhere in the paragraph bypasses):
    - 7-40 hex char SHA (`abc1234`, `4216eab5`)
    - `file/path.ext:LINENO` (covers Python, JS, MD, etc.)
    - Backtick-quoted commands containing shell exec verbs
      (pytest, python, bash, npm, curl, git, gh, codex, docker, ruff, mypy)
    - Trace/UUID anchor: `trace:` `task:` `inv_` or UUID-shape
    - Artifact path anchor: `_diag/.../...md`, `.out`, `.log`, `.txt`
    - URL evidence: http(s)://...
    - "merged into <branch>" / "shipped as <SHA>" phrasings

Hedged claims (downgraders that bypass): "candidate", "likely", "appears to",
"probably", "I suspect", "tentative", "needs verification", "unconfirmed".

Kill-switch: PROOF_CLAIM_LINT_OFF=1
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
        pass

# HIGH-CONFIDENCE PROOF CLAIM PATTERNS
# Each must be a phrasing that asserts something IS proven/done/fixed/working,
# not a hypothesis or future claim. Patterns include forbidden context to reduce
# false positives (e.g. "would be done", "should be green" should NOT fire).
CLAIM_PATTERNS = [
    # Causation
    r"\bthe root cause is\b",
    r"\broot cause:\s*\w",
    r"\bthe (?:real|actual) (?:cause|problem|bug) is\b",
    r"\bthat's the bug\b",
    r"\bthat is the bug\b",
    # Done / shipped
    r"\bthis is (?:now\s+)?(?:done|complete|fixed|shipped|merged|deployed|landed)\b",
    r"\bit'?s (?:now\s+)?(?:done|complete|fixed|shipped|merged|deployed|landed)\b",
    r"\bnow (?:done|complete|fixed|shipped|merged|deployed|landed)\b",
    r"\bsuccessfully (?:done|complete|fixed|shipped|merged|deployed|landed)\b",
    # Tests / gates green
    r"\b(?:tests?|gates?|checks?) (?:are\s+)?(?:now\s+)?(?:green|passing|passed)\b",
    r"\ball (?:tests?|gates?|checks?) pass\b",
    r"\bgreen across the board\b",
    # Proven / verified
    r"\b(?:proven|verified|confirmed) (?:that\s+|in\s+|with\s+|to\s+|by\s+)",
    r"\bI (?:proved|verified|confirmed) (?:that|this|it)\b",
    # Convergence
    r"\bwe (?:have\s+)?converged\b",
    r"\bconvergence reached\b",
    r"\b(?:two|2|three|3) (?:consecutive|adversarial) (?:rounds?\s+)?clean\b",
    # Launch-ready
    r"\blaunch[- ]ready\b",
    r"\bready to ship\b",
    r"\bready to launch\b",
]
CLAIM_RE = re.compile("(?:" + "|".join(CLAIM_PATTERNS) + ")", re.IGNORECASE)

# HEDGE PATTERNS — if the paragraph contains any of these, the claim is treated as
# tentative and passes (Claude is honestly hedging, not over-claiming).
HEDGE_PATTERNS = [
    r"\bcandidate\b",
    r"\blikely\b",
    r"\bappears? to\b",
    r"\bprobably\b",
    r"\bI suspect\b",
    r"\btentative\b",
    r"\bneeds? verification\b",
    r"\bunconfirmed\b",
    r"\bnot (?:yet )?(?:proven|verified|confirmed)\b",
    r"\bhypothes(?:is|izing)\b",
    r"\bmight be\b",
    r"\bcould be\b",
    r"\bworking theory\b",
    r"\bwould be\b",
    r"\bshould be\b",
    r"\bif\s+\w",  # conditional clause weakens claims
]
HEDGE_RE = re.compile("(?:" + "|".join(HEDGE_PATTERNS) + ")", re.IGNORECASE)

# EVIDENCE ANCHOR PATTERNS — any one in the same paragraph as the claim bypasses
EVIDENCE_PATTERNS = [
    r"\b[0-9a-f]{7,40}\b",  # SHA-like
    r"[/\\][A-Za-z0-9_./\\-]+\.[A-Za-z0-9]+:\d+\b",  # path/file.ext:LINE
    r"\b[A-Za-z0-9_.-]+\.(?:py|js|jsx|ts|tsx|md|yaml|yml|json|sh|sql|toml|cfg|ini):\d+\b",
    r"`[^`]*\b(?:pytest|python|bash|npm|curl|git|gh|codex|docker|ruff|mypy|node|playwright)\b[^`]*`",
    r"\btrace[_:][A-Za-z0-9_-]+",
    r"\btask[_:][A-Za-z0-9_-]+",
    r"\binv_\d+",
    r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b",  # UUID
    r"_diag[/\\][^\s)]+",
    r"\.(?:out|log|txt|jsonl)\b",
    r"https?://\S+",
    r"\bmerged (?:into|to)\s+\w",
    r"\bshipped as\s+[0-9a-f]{6,}",
    r"\bcommit\s+[0-9a-f]{6,}",
    r"\bsee\s+`[^`]+`",
]
EVIDENCE_RE = re.compile("(?:" + "|".join(EVIDENCE_PATTERNS) + ")", re.IGNORECASE)

MARKER_TTL = 12 * 60 * 60
DEFAULT_MAX_BLOCKS = 3


def _kill_switched() -> bool:
    return bool(os.environ.get("PROOF_CLAIM_LINT_OFF", "").strip())


def _max_blocks() -> int:
    raw = os.environ.get("PROOF_CLAIM_LINT_MAX_BLOCKS", "").strip()
    if not raw:
        return DEFAULT_MAX_BLOCKS
    try:
        return max(1, int(raw))
    except ValueError:
        return DEFAULT_MAX_BLOCKS


def _read_last_assistant_text(transcript_path: str) -> str:
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
                    parts = [
                        item.get("text", "") or ""
                        for item in content
                        if isinstance(item, dict) and item.get("type") == "text"
                    ]
                    joined = "\n".join(p for p in parts if p.strip())
                    if joined:
                        last_text = joined
    except OSError:
        return ""
    return last_text


def _marker_dir(cwd: Path) -> Path:
    return cwd / ".viola" / "agents" / "proof_claim_markers"


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


def _is_markdown_heading_only(para: str) -> bool:
    """True if the paragraph is JUST a markdown heading (one or more `#` lines).

    Headings are intentionally short, often phrase-shaped (`## Verified in each repo`,
    `### Tests green`), and don't carry the supporting evidence — that lives in the
    paragraph after. Treating a heading as a standalone claim creates false-positives
    where the lint blocks legitimate section labels.
    """
    lines = [ln.strip() for ln in para.splitlines() if ln.strip()]
    if not lines:
        return False
    return all(re.match(r"^#{1,6}\s+\S", ln) for ln in lines)


def evaluate(text: str) -> tuple[bool, str, str]:
    """Return (should_block, matched_claim, offending_paragraph).

    Block when: any paragraph contains a high-confidence claim, no hedge phrase
    in the same paragraph, and no evidence anchor in the same paragraph.
    """
    if not text:
        return False, "", ""
    # Split into paragraphs (blank-line separated). Markdown lists count as
    # part of the surrounding paragraph for our purposes.
    paragraphs = re.split(r"\n\s*\n", text)
    for paragraph in paragraphs:
        para = paragraph.strip()
        if not para:
            continue
        # Markdown heading-only paragraphs aren't standalone claims — their
        # evidence lives in the following body paragraph.
        if _is_markdown_heading_only(para):
            continue
        claim_match = CLAIM_RE.search(para)
        if not claim_match:
            continue
        # Hedged? skip — Claude is being honest about uncertainty
        if HEDGE_RE.search(para):
            continue
        # Evidence anchor present in same paragraph? skip
        if EVIDENCE_RE.search(para):
            continue
        return True, claim_match.group(0), para[:400]
    return False, "", ""


def _build_feedback(claim: str, paragraph: str) -> str:
    return (
        "PROOF_CLAIM_LINT blocked this turn.\n"
        "\n"
        f"You asserted: {claim!r}\n"
        "\n"
        "...but the same paragraph contains no evidence anchor. CLAUDE.md says\n"
        '"no root cause without a code location" and "any time I write \'root cause\',\n'
        "'proven', 'green', 'done', or 'converged', I must attach the evidence in the\n"
        'same paragraph."\n'
        "\n"
        "Offending paragraph (first 400 chars):\n"
        f"  {paragraph}\n"
        "\n"
        "Add at least ONE of these in the same paragraph:\n"
        "  - Current SHA or deployed SHA (7-40 hex)\n"
        "  - file/path.ext:LINENO  (the exact code location for causal claims)\n"
        "  - A backticked command you ran this session\n"
        "    (pytest/python/bash/git/codex/curl/...)\n"
        "  - A trace id, task id, or inv_NNN reference\n"
        "  - An artifact path under _diag/ or a *.out / *.log / *.jsonl file\n"
        "  - A URL to live evidence\n"
        "  - `merged into <branch>` or `shipped as <SHA>` or `commit <SHA>`\n"
        "\n"
        "If the claim is genuinely tentative, hedge it explicitly — words like\n"
        "'candidate', 'likely', 'appears to', 'probably', 'needs verification', or\n"
        "'not yet proven' downgrade the claim and pass the lint.\n"
        "\n"
        "Kill-switch (if this hook is wrong for your case):\n"
        "  set PROOF_CLAIM_LINT_OFF=1 in your environment.\n"
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
    block, claim, paragraph = evaluate(text)
    if not block:
        return 0, ""

    cwd = Path(str(payload.get("cwd") or Path.cwd())).resolve()
    session_id = str(payload.get("session_id") or "")
    marker = _marker_path(cwd, session_id)
    _cleanup_old_markers(_marker_dir(cwd))
    count = _load_block_count(marker) + 1
    _save_block_count(marker, count)

    if count > _max_blocks():
        sys.stderr.write(f"proof_claim_lint: block-cap reached ({count} > {_max_blocks()}); passing.\n")
        return 0, ""

    return 2, _build_feedback(claim, paragraph)


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
