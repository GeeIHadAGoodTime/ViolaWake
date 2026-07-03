"""Gate: spa-deep-link-rewrite-target.

Cloudflare Pages 308-canonicalizes any ".html" rewrite target to its clean URL
(/app.html -> /app, /app/index.html -> /app/) BEFORE serving. When a 200
rewrite's target canonicalizes to a path DIFFERENT from the requested route,
the rewrite degrades into a redirect and the original SPA path (and with it
the route the SPA must mount) is stripped. That exact shape silently broke
/verify-email and /reset-password deep links from 2026-05-08 to 2026-07-03 —
users clicked a valid verification link and landed on /app/ with the token but
no route, so the verify API was never called.

Rule enforced here: an SPA-fallback rewrite target must be a canonical
directory path (ends with "/", no ".html"). Per-page static rewrites of the
form "/pricing /pricing/index.html 200" stay allowed: their target
canonicalizes back to the source route itself, so nothing is lost.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
GENERATOR = REPO_ROOT / "console" / "frontend" / "scripts" / "generate-marketing.mjs"


def spa_rewrite_target_violations(redirect_lines: list[str]) -> list[str]:
    """Return _redirects lines whose rewrite target loses the SPA route.

    A target ending in ".html" is only safe when it is the source route's own
    directory index ("<route>/index.html"); anything else canonicalizes to a
    different URL and Pages emits a 308 that strips the requested path.
    """
    violations: list[str] = []
    for raw in redirect_lines:
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 3 or parts[2] != "200":
            continue
        source, target = parts[0], parts[1]
        if not target.endswith(".html"):
            continue
        source_route = source.rstrip("*").rstrip("/")
        expected_self_index = f"{source_route}/index.html"
        if target == expected_self_index:
            continue  # canonicalizes back to the source route; path survives
        violations.append(line)
    return violations


def _read_generator_target() -> str:
    match = re.search(
        r'const appShellRedirectTarget\s*=\s*"([^"]+)"', GENERATOR.read_text(encoding="utf-8")
    )
    assert match, "appShellRedirectTarget constant not found in generate-marketing.mjs"
    return match.group(1)


def test_app_shell_rewrite_target_is_canonical_directory() -> None:
    target = _read_generator_target()
    assert target.endswith("/") and not target.endswith(".html"), (
        f"appShellRedirectTarget is {target!r}; it must be the canonical directory "
        "form (e.g. '/app/'). File-path targets get 308-canonicalized by Cloudflare "
        "Pages, stripping the SPA deep-link path (2026-05-08 verify-email incident)."
    )


def test_detector_fails_on_the_pre_fix_shapes() -> None:
    old_shape_may_2026 = ["/verify-email /app/index.html 200"]
    older_shape = ["/verify-email /app.html 200"]
    assert spa_rewrite_target_violations(old_shape_may_2026), (
        "detector must flag the /app/index.html target that shipped the incident"
    )
    assert spa_rewrite_target_violations(older_shape), (
        "detector must flag the /app.html target (same class, clean-URL variant)"
    )


def test_detector_passes_on_the_fixed_and_benign_shapes() -> None:
    fixed = [
        "/verify-email /app/ 200",
        "/reset-password /app/ 200",
        "/pricing /pricing/index.html 200",  # self-canonical static page rewrite
        "/blog/some-post /blog/some-post/index.html 200",
        "/* /index.html 200",  # root SPA fallback: canonicalizes to "/", the source itself
        "/old-page /new-page 301",  # explicit redirects are out of scope
    ]
    assert spa_rewrite_target_violations(fixed) == []


def test_generated_redirects_have_no_violations_when_dist_exists() -> None:
    dist_redirects = REPO_ROOT / "console" / "frontend" / "dist" / "_redirects"
    if not dist_redirects.exists():
        return  # dist is a build artifact; the generator-constant test above still gates
    lines = dist_redirects.read_text(encoding="utf-8").splitlines()
    assert spa_rewrite_target_violations(lines) == []
