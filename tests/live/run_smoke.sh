#!/usr/bin/env bash
set -u
set -o pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUTPUT_FILE="$ROOT_DIR/tests/live/.smoke_last_output.txt"

cd "$ROOT_DIR" || exit 1

if [[ "${VIOLAWAKE_LIVE:-}" != "1" ]]; then
  echo "VIOLAWAKE_LIVE=1 is required for live smoke tests."
  exit 2
fi

echo "ViolaWake live smoke"
echo "API: ${VIOLAWAKE_API_BASE_URL:-https://api.violawake.com}"
echo "Site: ${VIOLAWAKE_SITE_URL:-https://violawake.com}"
echo

python -m pytest tests/live -m "live and smoke" --no-cov -ra -vv 2>&1 | tee "$OUTPUT_FILE"
STATUS=${PIPESTATUS[0]}

echo
echo "Smoke summary"
grep -E "tests/live/.*::.* (PASSED|FAILED|SKIPPED|XFAILED|XPASSED|ERROR)" "$OUTPUT_FILE" || true
echo
echo "Full smoke output: tests/live/.smoke_last_output.txt"

exit "$STATUS"
