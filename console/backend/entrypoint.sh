#!/bin/sh
set -e

# Run database migrations before serving. Idempotent — alembic handles
# already-applied migrations gracefully. Tolerates connection blips by
# retrying briefly while postgres warms up.
echo "[entrypoint] running alembic upgrade head"
for i in 1 2 3 4 5; do
    if alembic upgrade head; then
        break
    fi
    echo "[entrypoint] alembic try $i failed; retrying in 3s..."
    sleep 3
done

# Ensure OpenWakeWord backbone resources are present. The openwakeword
# wheel ships without ONNX/TFLite resources; download_models() fetches
# them on first call. Idempotent — short-circuits if files exist.
# Required for the training pipeline (OWWModel construction in
# violawake_sdk.tools.train).
echo "[entrypoint] ensuring openwakeword backbone models are downloaded"
python -c "from openwakeword.utils import download_models; download_models()" 2>&1 \
    | tail -3 || true

if [ "$(id -u)" = "0" ]; then
    chown -R app:app /app/data
    exec gosu app python run.py
else
    exec python run.py
fi
