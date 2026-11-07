#!/bin/sh
set -e

# VIOLAWAKE_BOOT_CHECK=1: do only what can fail on the data volume, then stop.
# scripts/deploy_backend.py runs this on a new image against the real volume, under
# the real capabilities, while the old container is still serving.
if [ "${VIOLAWAKE_BOOT_CHECK:-0}" != "1" ]; then
    # Run database migrations before serving. Idempotent — alembic handles
    # already-applied migrations gracefully. Tolerates connection blips by
    # retrying briefly while postgres warms up.
    echo "[entrypoint] running alembic upgrade head"
    for i in 1 2 3 4 5; do
        if alembic upgrade head; then
            break
        fi
        echo "[entrypoint] alembic try $i failed; retrying in 3s..."
        [ "$i" = "5" ] && exit 1
        sleep 3
    done

    # Ensure OpenWakeWord backbone resources are present. The openwakeword
    # wheel ships without ONNX/TFLite resources; download_models() fetches
    # them on first call. Idempotent — short-circuits if files exist.
    # Required for the training pipeline (OWWModel construction in
    # violawake_sdk.tools.train).
    echo "[entrypoint] ensuring openwakeword backbone models are downloaded"
    python -c "from openwakeword.utils import download_models; download_models()"
fi

DATA_DIR="${VIOLAWAKE_ENTRYPOINT_DATA_DIR:-/app/data}"

if [ "$(id -u)" = "0" ]; then
    # This runs with every capability dropped except CHOWN, SETUID and SETGID, so root
    # cannot read a folder another user locked to 700, and the app makes those: ONNX
    # Runtime's cache, and every temp folder from mkdtemp. chown -R skips what it cannot
    # read and exits 1. Under `set -e` that used to end the boot, so one such folder
    # stopped every restart (2026-11-07, api.violawake.com down for hours). A folder
    # root cannot read is one the app already owns, so skipping it is correct. What
    # must hold is checked directly below.
    chown -R app:app "$DATA_DIR" || echo "[entrypoint] WARNING: could not chown everything under $DATA_DIR; carrying on"

    for dir in "$DATA_DIR" "$DATA_DIR/recordings" "$DATA_DIR/models"; do
        if [ -d "$dir" ] && ! gosu app test -w "$dir"; then
            echo "[entrypoint] FATAL: the app user cannot write $dir"
            exit 1
        fi
    done

    [ "${VIOLAWAKE_BOOT_CHECK:-0}" = "1" ] && { echo "[entrypoint] boot check passed"; exit 0; }
    exec gosu app python run.py
else
    [ "${VIOLAWAKE_BOOT_CHECK:-0}" = "1" ] && { echo "[entrypoint] boot check passed"; exit 0; }
    exec python run.py
fi
