#!/usr/bin/env python3
# PreToolUse(matcher=Bash) guard for codex dispatch.
#
# Blocks the failure modes that strand Claude or corrupt codex behavior:
#   --full-auto                : overrides config.toml sandbox, breaks Windows file writes
#   --sandbox*                 : config.toml already sets sandbox_mode = "danger-full-access"
#   --approval*                : config.toml already sets approval_policy = "never"
#   raw `codex exec`           : bypasses dispatch_codex.sh -> no .pid, no DONE marker, no self-wake
#   trailing `&` detach        : detaches from harness -> no completion notification -> Jay wakes us
#
# Allowed:
#   bash .../dispatch_codex.sh <prompt> <out> [timeout_s] [model]
#   codex exec resume <UUID> ...   (per feedback_codex_exec_resume_via_session_id.md)
#
# Override marker (rare-but-necessary cases): include literal token
#     [codex-guard-override: <reason>]
# anywhere in the command. The guard logs the override but does not block.
import json
import re
import sys


def main() -> int:
    try:
        payload = json.load(sys.stdin)
    except (json.JSONDecodeError, ValueError, OSError):
        return 0

    command = (payload.get("tool_input") or {}).get("command", "") or ""
    if "codex" not in command:
        return 0

    # Explicit override marker
    if "[codex-guard-override:" in command:
        sys.stderr.write("codex_guard: override marker present, passing through.\n")
        return 0

    # Forbidden flags (existing rules)
    forbidden = None
    if re.search(r"--full-auto", command, re.IGNORECASE):
        forbidden = "--full-auto"
    elif re.search(r"--sandbox", command, re.IGNORECASE):
        forbidden = "--sandbox*"
    elif re.search(r"--approval", command, re.IGNORECASE):
        forbidden = "--approval*"

    if forbidden is not None:
        sys.stderr.write(
            f"BLOCKED: codex called with forbidden flag: {forbidden}\n"
            "\n"
            "config.toml already has:\n"
            '  approval_policy = "never"\n'
            '  sandbox_mode = "danger-full-access"\n'
            "\n"
            f"{forbidden} overrides these and BREAKS file writes on Windows.\n"
            "\n"
            "Correct: dispatch through .claude/skills/codex-dispatch/scripts/dispatch_codex.sh\n"
        )
        return 2

    # Trailing `&` detach check FIRST — applies whether the command invokes codex
    # directly (codex exec ...) or through the wrapper (dispatch_codex.sh ...).
    # The detach kills the harness self-wake either way.
    detach_pattern = re.compile(r"(?:dispatch_codex\.sh|codex\s+exec)\b[^&|;`]*&\s*(?:$|;|#)")
    if detach_pattern.search(command):
        sys.stderr.write(
            "BLOCKED: trailing `&` detach detected on codex dispatch.\n"
            "\n"
            "Trailing `&` detaches the process from the harness, which means:\n"
            "  - no task-notification when codex finishes\n"
            "  - no auto-wake -> Claude strands itself, Jay has to wake us\n"
            "\n"
            "Use Bash(run_in_background=true, ...) instead. That IS the self-wake.\n"
            "See: feedback-codex-dispatch-run-in-background-never-detach.\n"
        )
        return 2

    # Identify whether this command actually invokes codex (not just mentions it in a path/arg)
    invokes_codex = bool(re.search(r"(?:^|[\s;&|`(])codex\s+exec\b", command))
    if not invokes_codex:
        return 0

    # `codex exec resume <uuid>` is the explicit recovery path (per memory)
    is_resume = bool(re.search(r"codex\s+exec\s+resume\b", command))
    if is_resume:
        return 0

    # Must go through dispatch_codex.sh
    uses_wrapper = "dispatch_codex.sh" in command
    if not uses_wrapper:
        sys.stderr.write(
            "BLOCKED: raw `codex exec` detected.\n"
            "\n"
            "Raw codex bypasses dispatch_codex.sh, losing:\n"
            "  - .pid sidecar (process-id tracking)\n"
            "  - DONE rc=<n> marker (completion signal)\n"
            "  - run_in_background self-wake (background-task notification)\n"
            "\n"
            "Use the wrapper:\n"
            "  bash .claude/skills/codex-dispatch/scripts/dispatch_codex.sh \\\n"
            "       <prompt_file> <out_file> 0\n"
            "\n"
            "Invoke via Bash(run_in_background=true, ...) for auto-wake on completion.\n"
            "See: feedback-codex-dispatch-use-skill-wrapper.\n"
            "\n"
            "If you genuinely need raw codex (rare), include\n"
            "    [codex-guard-override: <one-line-reason>]\n"
            "in the command.\n"
        )
        return 2

    return 0


if __name__ == "__main__":
    sys.exit(main())
