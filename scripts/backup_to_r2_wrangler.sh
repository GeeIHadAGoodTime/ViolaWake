#!/usr/bin/env bash
# Nightly backup of ViolaWake Postgres + /app/data to Cloudflare R2.
#
# Uses wrangler (with the user's existing OAuth) instead of boto3 + S3 keys,
# so no S3-compatible credentials need to be minted. The wrangler login
# persists across reboots in ~/.wrangler/config/default.toml.
#
# Run nightly via Windows Task Scheduler (see scripts/backup-task.xml) or
# any cron equivalent. Requires Docker Desktop running.
#
# Retention: keeps the last 30 daily backups in each prefix; older ones are
# deleted after a successful upload.

set -Eeuo pipefail

BUCKET="${VIOLAWAKE_BACKUP_R2_BUCKET:-violawake-backups}"
POSTGRES_CONTAINER="${VIOLAWAKE_POSTGRES_CONTAINER:-wakeword-postgres-1}"
BACKEND_CONTAINER="${VIOLAWAKE_BACKEND_CONTAINER:-wakeword-backend-1}"
RETENTION_DAYS="${VIOLAWAKE_BACKUP_RETENTION_DAYS:-30}"

TODAY="$(date -u +%Y-%m-%d)"
TMPDIR="${TMPDIR:-/tmp}"
PG_FILE="${TMPDIR}/violawake-backup-${TODAY}.sql.gz"
DATA_FILE="${TMPDIR}/violawake-backup-${TODAY}-app-data.tar.gz"

cleanup() { rm -f "$PG_FILE" "$DATA_FILE"; }
trap cleanup EXIT

require_container() {
    docker inspect --format '{{.State.Running}}' "$1" 2>/dev/null | grep -q true || {
        echo "ERROR: container $1 is not running" >&2
        exit 1
    }
}

require_container "$POSTGRES_CONTAINER"
require_container "$BACKEND_CONTAINER"

echo "[$(date -u +%FT%TZ)] dumping postgres -> $PG_FILE"
docker exec "$POSTGRES_CONTAINER" sh -c 'pg_dump -U "$POSTGRES_USER" "$POSTGRES_DB"' 2>/dev/null | gzip > "$PG_FILE"

echo "[$(date -u +%FT%TZ)] tar /app/data -> $DATA_FILE"
docker exec "$BACKEND_CONTAINER" sh -c 'cd /app/data && tar --warning=no-file-changed -czf - . 2>/dev/null' > "$DATA_FILE"

echo "[$(date -u +%FT%TZ)] upload postgres/${TODAY}.sql.gz"
wrangler r2 object put "${BUCKET}/postgres/${TODAY}.sql.gz" --file "$PG_FILE" --remote

echo "[$(date -u +%FT%TZ)] upload app-data/${TODAY}.tar.gz"
wrangler r2 object put "${BUCKET}/app-data/${TODAY}.tar.gz" --file "$DATA_FILE" --remote

echo "[$(date -u +%FT%TZ)] retention pass: keep last ${RETENTION_DAYS} per prefix"
prune_prefix() {
    local prefix="$1"
    # Wrangler 4.x doesn't have a native list-objects; use Cloudflare API directly.
    # CLOUDFLARE_API_TOKEN + CLOUDFLARE_ACCOUNT_ID must be in env or in
    # an .env.production loaded by the parent.
    local token="${CLOUDFLARE_API_TOKEN:-}"
    local account="${CLOUDFLARE_ACCOUNT_ID:-}"
    if [[ -z "$token" || -z "$account" ]]; then
        echo "  retention skipped (CLOUDFLARE_API_TOKEN / CLOUDFLARE_ACCOUNT_ID not set)"
        return 0
    fi
    local keys
    keys=$(curl -fsS "https://api.cloudflare.com/client/v4/accounts/${account}/r2/buckets/${BUCKET}/objects?prefix=${prefix}/&limit=1000" \
        -H "Authorization: Bearer ${token}" \
        | python -c "
import json,sys
d = json.load(sys.stdin)
if not d.get('success'):
    sys.exit(0)
items = sorted(d.get('result', []), key=lambda x: x.get('key',''))
for o in items[:-${RETENTION_DAYS}]:
    print(o['key'])
")
    if [[ -z "$keys" ]]; then return 0; fi
    while IFS= read -r key; do
        [[ -z "$key" ]] && continue
        echo "  delete ${key}"
        wrangler r2 object delete "${BUCKET}/${key}" --remote >/dev/null 2>&1 || true
    done <<< "$keys"
}

# Optional: load CLOUDFLARE_* from .env.production if present
if [[ -f .env.production ]]; then
    set +e
    set -a
    # shellcheck disable=SC1091
    source .env.production 2>/dev/null
    set +a
    set -e
fi

prune_prefix postgres
prune_prefix app-data

echo "[$(date -u +%FT%TZ)] backup OK"
