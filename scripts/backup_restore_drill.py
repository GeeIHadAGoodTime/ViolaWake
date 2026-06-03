#!/usr/bin/env python3
"""Restore the latest ViolaWake R2 Postgres backup into scratch Postgres.

This drill is intentionally read-only against production:
- downloads a backup object from Cloudflare R2 through the Cloudflare API
- starts a temporary Postgres container with a generated name
- restores the dump into that scratch database
- runs a small query to prove the restored database is readable

It never writes to wakeword-postgres-1 or any production container.
"""

from __future__ import annotations

import argparse
import datetime as dt
import gzip
import json
import os
import re
import secrets
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any


DEFAULT_BUCKET = "violawake-backups"
DEFAULT_PREFIX = "postgres"
DEFAULT_IMAGE = "postgres:16-alpine"
DATE_RE = re.compile(r"(\d{4}-\d{2}-\d{2})")


def load_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def merged_env(paths: list[Path]) -> dict[str, str]:
    env = dict(os.environ)
    for path in paths:
        env.update(load_env_file(path))
    return env


def env_first(env: dict[str, str], *names: str, default: str = "") -> str:
    for name in names:
        value = env.get(name, "").strip()
        if value:
            return value
    return default


def cloudflare_request(env: dict[str, str], path: str, *, binary: bool = False) -> Any:
    token = env_first(env, "CLOUDFLARE_API_TOKEN")
    if not token:
        raise RuntimeError("CLOUDFLARE_API_TOKEN is required")

    request = urllib.request.Request(
        f"https://api.cloudflare.com/client/v4{path}",
        headers={"Authorization": f"Bearer {token}"},
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            payload = response.read()
            if binary:
                return payload
            return json.loads(payload.decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Cloudflare API GET {path} failed: HTTP {exc.code} {detail}") from exc


def require_account(env: dict[str, str]) -> str:
    account_id = env_first(env, "CLOUDFLARE_ACCOUNT_ID")
    if not account_id:
        raise RuntimeError("CLOUDFLARE_ACCOUNT_ID is required")
    return account_id


def parse_object_date(key: str) -> dt.date | None:
    match = DATE_RE.search(Path(key).name)
    if not match:
        return None
    return dt.date.fromisoformat(match.group(1))


def list_objects(env: dict[str, str], bucket: str, prefix: str) -> list[dict[str, Any]]:
    account_id = require_account(env)
    encoded_prefix = urllib.parse.quote(f"{prefix.rstrip('/')}/", safe="")
    response = cloudflare_request(
        env,
        f"/accounts/{account_id}/r2/buckets/{bucket}/objects?prefix={encoded_prefix}&limit=1000",
    )
    if not response.get("success"):
        raise RuntimeError(f"Cloudflare object list failed: {response}")
    objects = [item for item in response.get("result", []) if parse_object_date(str(item.get("key", "")))]
    objects.sort(key=lambda item: parse_object_date(str(item["key"])) or dt.date.min, reverse=True)
    return objects


def download_object(env: dict[str, str], bucket: str, key: str) -> bytes:
    account_id = require_account(env)
    encoded_key = urllib.parse.quote(key, safe="/")
    return cloudflare_request(
        env,
        f"/accounts/{account_id}/r2/buckets/{bucket}/objects/{encoded_key}",
        binary=True,
    )


def run(command: list[str], *, input_bytes: bytes | None = None, timeout: int = 120) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        command,
        input=input_bytes,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
        check=False,
    )


def require_docker() -> None:
    if shutil.which("docker") is None:
        raise RuntimeError("docker CLI is required for the scratch restore drill")


def start_scratch_postgres(image: str, container: str, password: str) -> None:
    result = run(
        [
            "docker",
            "run",
            "-d",
            "--rm",
            "--name",
            container,
            "-e",
            "POSTGRES_USER=violawake",
            "-e",
            "POSTGRES_DB=violawake",
            "-e",
            f"POSTGRES_PASSWORD={password}",
            image,
        ],
        timeout=180,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.decode("utf-8", errors="replace").strip())


def stop_container(container: str) -> None:
    run(["docker", "rm", "-f", container], timeout=60)


def wait_for_postgres(container: str) -> None:
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        result = run(
            ["docker", "exec", container, "pg_isready", "-U", "violawake", "-d", "violawake"],
            timeout=15,
        )
        if result.returncode == 0:
            return
        time.sleep(2)
    raise RuntimeError(f"scratch Postgres container {container} did not become ready")


def restore_sql(container: str, sql: bytes) -> None:
    result = run(
        [
            "docker",
            "exec",
            "-i",
            container,
            "psql",
            "-v",
            "ON_ERROR_STOP=1",
            "-U",
            "violawake",
            "-d",
            "violawake",
        ],
        input_bytes=sql,
        timeout=300,
    )
    if result.returncode != 0:
        stderr = result.stderr.decode("utf-8", errors="replace")
        raise RuntimeError(f"restore failed: {stderr[-1000:]}")


def query_restored_db(container: str) -> str:
    sql = (
        "SELECT 'tables=' || COUNT(*) "
        "FROM information_schema.tables WHERE table_schema='public'; "
        "SELECT 'alembic_versions=' || COUNT(*) FROM public.alembic_version;"
    )
    result = run(
        ["docker", "exec", container, "psql", "-t", "-A", "-U", "violawake", "-d", "violawake", "-c", sql],
        timeout=60,
    )
    if result.returncode != 0:
        stderr = result.stderr.decode("utf-8", errors="replace")
        raise RuntimeError(f"restore query failed: {stderr[-1000:]}")
    return result.stdout.decode("utf-8", errors="replace").strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", action="append", type=Path, default=[Path(".env.production")])
    parser.add_argument("--bucket", default=DEFAULT_BUCKET)
    parser.add_argument("--prefix", default=DEFAULT_PREFIX)
    parser.add_argument("--postgres-image", default=DEFAULT_IMAGE)
    parser.add_argument("--max-age-hours", type=int, default=36)
    parser.add_argument("--inspect-only", action="store_true", help="Download/decompress the latest dump but skip Docker restore")
    parser.add_argument("--keep-container", action="store_true", help="Leave the scratch container running after restore")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    env = merged_env(args.env_file)
    try:
        objects = list_objects(env, args.bucket, args.prefix)
        if not objects:
            raise RuntimeError(f"No backup objects found under r2://{args.bucket}/{args.prefix}/")

        latest = objects[0]
        key = str(latest["key"])
        backup_date = parse_object_date(key)
        if backup_date is None:
            raise RuntimeError(f"Could not parse backup date from {key}")

        age_hours = (dt.datetime.now(dt.timezone.utc) - dt.datetime.combine(backup_date, dt.time.min, tzinfo=dt.timezone.utc)).total_seconds() / 3600
        print(f"Latest backup: r2://{args.bucket}/{key}")
        print(f"Backup age: {age_hours:.1f} hours")
        if age_hours > args.max_age_hours:
            raise RuntimeError(f"Latest backup is older than {args.max_age_hours} hours")

        gzipped = download_object(env, args.bucket, key)
        sql = gzip.decompress(gzipped)
        print(f"Downloaded {len(gzipped)} compressed bytes; decompressed to {len(sql)} SQL bytes")

        if args.inspect_only:
            create_count = sql.count(b"CREATE TABLE ")
            copy_count = sql.count(b"COPY public.")
            print(f"SQL inspection: create_tables={create_count} copy_sections={copy_count}")
            if create_count == 0 or copy_count == 0:
                raise RuntimeError(
                    "Backup SQL inspection failed: expected at least one CREATE TABLE "
                    "and one COPY public.* section in the downloaded dump, got "
                    f"create_tables={create_count} copy_sections={copy_count}. An "
                    "empty-but-gzipped or non-dump artifact would pass without this check."
                )
            print("SQL inspection OK")
            return 0

        require_docker()
        container = f"violawake-restore-drill-{int(time.time())}-{secrets.token_hex(3)}"
        password = secrets.token_urlsafe(18)
        print(f"Starting scratch Postgres container: {container}")
        try:
            start_scratch_postgres(args.postgres_image, container, password)
            wait_for_postgres(container)
            restore_sql(container, sql)
            query_output = query_restored_db(container)
            print("Restore query OK:")
            print(query_output)
        finally:
            if args.keep_container:
                print(f"Keeping scratch container for inspection: {container}")
            else:
                stop_container(container)
                print(f"Removed scratch container: {container}")
        return 0
    except Exception as exc:
        print(f"backup_restore_drill failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
