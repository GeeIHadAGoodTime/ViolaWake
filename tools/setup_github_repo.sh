#!/usr/bin/env bash
set -euo pipefail

# Configure GitHub repository metadata for ViolaWake.
# This script is idempotent: re-running it applies the same desired settings.

# Fail early with a clear message if the GitHub CLI is unavailable.
if ! command -v gh >/dev/null 2>&1; then
  echo "Error: gh CLI is required but was not found in PATH." >&2
  exit 1
fi

REPO="${1:-$(gh repo view --json nameWithOwner -q '.nameWithOwner')}"

DESCRIPTION=$'Open-source wake word engine \u2014 Apache 2.0 alternative to Porcupine. ONNX inference, custom training, bundled TTS/STT pipeline.'
HOMEPAGE_URL="https://violawake.com"

TOPICS_JSON=$(cat <<'JSON'
{
  "names": [
    "wake-word",
    "voice-assistant",
    "speech-recognition",
    "onnx",
    "python",
    "tts",
    "stt",
    "machine-learning",
    "open-source",
    "porcupine-alternative"
  ]
}
JSON
)

# Verify that the target repository is accessible before attempting updates.
gh repo view "$REPO" >/dev/null

# Set the description, homepage, and discussions flag in one call.
# Re-applying the same values is safe.
gh repo edit "$REPO" \
  --description "$DESCRIPTION" \
  --homepage "$HOMEPAGE_URL" \
  --enable-discussions

# Replace the repo topics with the exact desired set.
# PUT is safe to run repeatedly and avoids accumulating stale topics.
gh api \
  --method PUT \
  -H "Accept: application/vnd.github+json" \
  "repos/$REPO/topics" \
  --input - <<<"$TOPICS_JSON" >/dev/null

echo "GitHub metadata configured for $REPO"
