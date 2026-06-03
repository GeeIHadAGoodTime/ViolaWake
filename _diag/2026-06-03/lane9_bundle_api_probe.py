"""Check whether a Vite bundle has the expected ViolaWake API base URL."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


PRODUCTION_API = "https://api.violawake.com/api"
SAME_ORIGIN_API_RE = re.compile(r"""["']\/api["']""")


def read_js_bundle(dist_dir: Path) -> tuple[str, list[Path]]:
    js_files = sorted((dist_dir / "assets").glob("*.js"))
    if not js_files:
        raise SystemExit(f"FAIL: no JS assets found under {dist_dir / 'assets'}")
    return "\n".join(path.read_text(encoding="utf-8") for path in js_files), js_files


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("dist_dir", type=Path)
    parser.add_argument(
        "--expect",
        choices=("production", "absent-env-detected"),
        required=True,
    )
    args = parser.parse_args()

    bundle, js_files = read_js_bundle(args.dist_dir)
    has_production_api = PRODUCTION_API in bundle
    has_same_origin_api = bool(SAME_ORIGIN_API_RE.search(bundle))
    asset_list = ", ".join(str(path) for path in js_files)

    if args.expect == "production":
        if has_production_api and not has_same_origin_api:
            print(f"PASS: production API base is baked into JS assets: {asset_list}")
            return 0
        print(
            "FAIL: production bundle API base mismatch "
            f"(has_production_api={has_production_api}, has_same_origin_api={has_same_origin_api}) "
            f"in {asset_list}"
        )
        return 1

    if has_same_origin_api and not has_production_api:
        print(f"PASS: absent VITE_API_URL fallback detected in JS assets: {asset_list}")
        return 0
    print(
        "FAIL: absent-env broken shape was not detected "
        f"(has_production_api={has_production_api}, has_same_origin_api={has_same_origin_api}) "
        f"in {asset_list}"
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
