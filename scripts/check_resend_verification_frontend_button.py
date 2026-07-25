"""Detect whether the console frontend exposes a self-service resend-verification
action on the unverified-email banner.

Regression this guards against: the backend POST /api/auth/resend-verification
endpoint (quality/gates.yaml: resend-verification-endpoint) shipped in PR #3
with no frontend caller. Layout.tsx's unverified-email banner rendered static
text only, so a user whose first verification email was lost had no
self-service recovery path even though the backend supported one (GH #2153).

This is a static source check (no build/browser required) so it can run fast
in CI: it greps the frontend source for (a) an API client function that POSTs
to /auth/resend-verification and (b) a banner control wired to call it with a
human-readable "resend" affordance. It intentionally does not require an
exact string match so refactors don't spuriously break it, but it does
require enough evidence that clicking something in the unverified banner
actually reaches the endpoint.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

API_CALL_RE = re.compile(
    r"resendVerification[\s\S]{0,400}?/auth/resend-verification", re.I
)
BANNER_BUTTON_RE = re.compile(
    r"verification-banner[\s\S]{0,2000}?<button[\s\S]{0,400}?resendVerification",
    re.I,
)
VISIBLE_LABEL_RE = re.compile(r"resend[^\"'<]{0,20}verification", re.I)


def check(frontend_src: Path) -> tuple[bool, list[str]]:
    api_ts = frontend_src / "api.ts"
    layout_tsx = frontend_src / "components" / "Layout.tsx"
    messages: list[str] = []
    ok = True

    if not api_ts.exists():
        return False, [f"FAIL: {api_ts} does not exist"]
    if not layout_tsx.exists():
        return False, [f"FAIL: {layout_tsx} does not exist"]

    api_text = api_ts.read_text(encoding="utf-8")
    layout_text = layout_tsx.read_text(encoding="utf-8")

    if API_CALL_RE.search(api_text):
        messages.append("PASS: api.ts has a resendVerification() call to /auth/resend-verification")
    else:
        ok = False
        messages.append(
            "FAIL: api.ts has no client function calling POST /auth/resend-verification "
            "(the backend endpoint has no frontend caller)"
        )

    if BANNER_BUTTON_RE.search(layout_text):
        messages.append("PASS: Layout.tsx wires a <button> inside the verification banner to resendVerification()")
    else:
        ok = False
        messages.append(
            "FAIL: Layout.tsx's verification banner does not contain a <button> that calls "
            "resendVerification() -- the banner is static/non-actionable"
        )

    if VISIBLE_LABEL_RE.search(layout_text):
        messages.append("PASS: Layout.tsx has a human-readable resend-verification label")
    else:
        ok = False
        messages.append("FAIL: Layout.tsx has no visible 'resend ... verification' label for the user")

    return ok, messages


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--frontend-src",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "console" / "frontend" / "src",
        help="Path to console/frontend/src (defaults to the repo's own checkout)",
    )
    args = parser.parse_args()

    ok, messages = check(args.frontend_src)
    for message in messages:
        print(message)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
