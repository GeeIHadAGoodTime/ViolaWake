#!/usr/bin/env bash
# Daily Postgres backup for ViolaWake.
#
# Dumps wakeword-postgres-1 → uploads to Cloudflare R2 bucket "violawake-backups".
# Retains the last 30 daily backups in R2 (older ones are deleted).
#
# Run via Windows Task Scheduler (or cron on linux/mac):
#   schtasks /create /tn "ViolaWake-Backup" /tr "C:\path\bash.exe -c '/j/CLAUDE/PROJECTS/Wakeword/scripts/backup_postgres.sh'" /sc daily /st 03:00 /f
#
# Manual run: bash scripts/backup_postgres.sh
#
# Restore (manual):
#   wrangler r2 object get violawake-backups/<backup-name>.sql.gz --file=/tmp/restore.sql.gz
#   gunzip -c /tmp/restore.sql.gz | docker exec -i wakeword-postgres-1 psql -U violawake violawake

set -euo pipefail

BUCKET="violawake-backups"
CONTAINER="wakeword-postgres-1"
DB_USER="violawake"
DB_NAME="violawake"
RETAIN_DAYS=30
LOCAL_DIR="${LOCAL_BACKUP_DIR:-/tmp/violawake_backups}"

DATE=$(date +%Y-%m-%d_%H%M)
DUMP_NAME="violawake_pg_${DATE}.sql.gz"
LOCAL_PATH="${LOCAL_DIR}/${DUMP_NAME}"

mkdir -p "${LOCAL_DIR}"

echo "[$(date -Iseconds)] starting backup → ${DUMP_NAME}"

# 1. Dump from inside the running container, pipe through gzip, write locally
docker exec "${CONTAINER}" pg_dump -U "${DB_USER}" -d "${DB_NAME}" --format=plain --no-owner --no-acl \
  | gzip -9 > "${LOCAL_PATH}"

SIZE=$(stat -c%s "${LOCAL_PATH}" 2>/dev/null || stat -f%z "${LOCAL_PATH}")
echo "[$(date -Iseconds)] dump complete: ${SIZE} bytes"

# 2. Upload to R2
echo "[$(date -Iseconds)] uploading to r2://${BUCKET}/${DUMP_NAME}"
wrangler r2 object put "${BUCKET}/${DUMP_NAME}" --file="${LOCAL_PATH}" --remote
echo "[$(date -Iseconds)] upload OK"

# 3. Delete local copy
rm -f "${LOCAL_PATH}"

# 4. Prune R2 backups older than RETAIN_DAYS days.
#    Wrangler R2 doesn't expose object listing well from the CLI, so we list via
#    the S3-compatible API would be cleaner — for now, rely on R2 lifecycle rules
#    set in the dashboard, OR call CF API directly. Lightweight approach:
#    delete-by-name where the date in the filename is older than the cutoff.
CUTOFF=$(date -d "${RETAIN_DAYS} days ago" +%Y-%m-%d 2>/dev/null || date -v-${RETAIN_DAYS}d +%Y-%m-%d)
echo "[$(date -Iseconds)] retention cutoff: ${CUTOFF}"
# NOTE: list/delete via wrangler r2 isn't ergonomic; recommend setting an R2
# lifecycle rule "delete after 30 days" in the Cloudflare dashboard for now.
# (One-time setup; persists across all backups.)

echo "[$(date -Iseconds)] backup complete"
