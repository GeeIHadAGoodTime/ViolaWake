#!/usr/bin/env python3
"""Restore the latest ViolaWake R2 Postgres backup into scratch Postgres.

Archive format (the Hetzner box's nightly job since 2026-07-08,
NOVVIOLA deploy/backup/dump_violawake_prod.sh):
    r2://violawake-backups/prod/violawake-prod-YYYYMMDDTHHMMSSZ.tar.gz.enc
    = openssl enc -aes-256-cbc -pbkdf2 (passphrase file) of a tar.gz holding
      db.dump (pg_dump -Fc) + globals.sql (roles)

This drill is intentionally read-only against production:
- lists and downloads the newest archive from Cloudflare R2 through the Cloudflare API
- decrypts it with the backup passphrase and checks db.dump is a real pg_dump archive
- starts a throwaway Postgres container (prod's image, --network none, generated name),
  loads the roles, pg_restore --exit-on-error into it
- runs a small query to prove the restored database is readable, then removes the container

It never writes to wakeword-postgres-1 or any production container.
Any failure exits 1; with --alert it also pages Telegram through the box's
scripts/ops_alert.py (the same path the nightly backup uses).
"""

from __future__ import annotations

import argparse
import datetime as dt
import io
import json
import os
import re
import secrets
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any


DEFAULT_BUCKET = "violawake-backups"
DEFAULT_PREFIX = "prod"
DEFAULT_PASSFILE = "/root/.config/violawake/backup_passphrase"
DEFAULT_PROD_CONTAINER = "wakeword-postgres-1"
DEFAULT_IMAGE = "postgres:16-alpine"  # used only when the prod container cannot be inspected
DEFAULT_DB = "violawake"
DEFAULT_OWNER = "violawake"
DEFAULT_ALERT_PY = "/opt/viola/NOVVIOLA/scripts/ops_alert.py"
STAMP_RE = re.compile(r"violawake-prod-(\d{8}T\d{6}Z)\.tar\.gz\.enc$")


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
        raise RuntimeError("CLOUDFLARE_API_TOKEN is required (R2 read on the backup bucket)")

    request = urllib.request.Request(
        f"https://api.cloudflare.com/client/v4{path}",
        headers={"Authorization": f"Bearer {token}"},
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            payload = response.read()
            if binary:
                return payload
            return json.loads(payload.decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Cloudflare API GET {path} failed: HTTP {exc.code} {detail[:300]}") from exc


def require_account(env: dict[str, str]) -> str:
    account_id = env_first(env, "CLOUDFLARE_ACCOUNT_ID")
    if not account_id:
        raise RuntimeError("CLOUDFLARE_ACCOUNT_ID is required")
    return account_id


def parse_object_time(key: str) -> dt.datetime | None:
    match = STAMP_RE.search(Path(key).name)
    if not match:
        return None
    return dt.datetime.strptime(match.group(1), "%Y%m%dT%H%M%SZ").replace(tzinfo=dt.timezone.utc)


def list_objects(env: dict[str, str], bucket: str, prefix: str) -> list[dict[str, Any]]:
    account_id = require_account(env)
    encoded_prefix = urllib.parse.quote(f"{prefix.rstrip('/')}/", safe="")
    objects: list[dict[str, Any]] = []
    cursor = ""
    while True:
        path = f"/accounts/{account_id}/r2/buckets/{bucket}/objects?prefix={encoded_prefix}&per_page=1000"
        if cursor:
            path += f"&cursor={urllib.parse.quote(cursor, safe='')}"
        response = cloudflare_request(env, path)
        if not response.get("success"):
            raise RuntimeError(f"Cloudflare object list failed: {response.get('errors')}")
        objects.extend(item for item in response.get("result", []) if parse_object_time(str(item.get("key", ""))))
        info = response.get("result_info") or {}
        cursor = info.get("cursor") or ""
        if not info.get("is_truncated") or not cursor:
            break
    objects.sort(key=lambda item: parse_object_time(str(item["key"])) or dt.datetime.min.replace(tzinfo=dt.timezone.utc), reverse=True)
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


def decrypt_archive(blob: bytes, passfile: Path) -> dict[str, bytes]:
    """openssl-decrypt the archive and return {member name: bytes} for db.dump + globals.sql."""
    if not passfile.is_file():
        raise RuntimeError(f"backup passphrase file not found: {passfile}")
    if shutil.which("openssl") is None:
        raise RuntimeError("openssl is required to decrypt the backup")
    with tempfile.TemporaryDirectory(prefix="violawake-drill-") as tmp:
        enc = Path(tmp) / "archive.enc"
        enc.write_bytes(blob)
        result = run(
            ["openssl", "enc", "-d", "-aes-256-cbc", "-pbkdf2", "-pass", f"file:{passfile}", "-in", str(enc)],
            timeout=300,
        )
    if result.returncode != 0:
        raise RuntimeError("decrypt failed (wrong passphrase or corrupt archive): "
                           + result.stderr.decode("utf-8", errors="replace").strip()[-300:])
    members: dict[str, bytes] = {}
    with tarfile.open(fileobj=io.BytesIO(result.stdout), mode="r:gz") as tar:
        for name in ("db.dump", "globals.sql"):
            try:
                handle = tar.extractfile(name)
            except KeyError:
                handle = None
            if handle is None:
                raise RuntimeError(f"archive is missing {name}")
            members[name] = handle.read()
    if not members["db.dump"].startswith(b"PGDMP"):
        raise RuntimeError("db.dump is not a pg_dump custom-format archive (no PGDMP header)")
    return members


def require_docker() -> None:
    if shutil.which("docker") is None:
        raise RuntimeError("docker CLI is required for the scratch restore drill")


def scratch_image(prod_container: str, override: str) -> str:
    if override:
        return override
    result = run(["docker", "inspect", "-f", "{{.Config.Image}}", prod_container], timeout=30)
    image = result.stdout.decode("utf-8", errors="replace").strip()
    return image if result.returncode == 0 and image else DEFAULT_IMAGE


def start_scratch_postgres(image: str, container: str) -> None:
    result = run(
        [
            "docker", "run", "-d", "--name", container, "--network", "none",
            "--label", "viola.backup.throwaway=1",
            "-e", "POSTGRES_HOST_AUTH_METHOD=trust",
            image,
        ],
        timeout=180,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.decode("utf-8", errors="replace").strip())


def stop_container(container: str) -> None:
    run(["docker", "rm", "-f", "-v", container], timeout=60)


def wait_for_postgres(container: str) -> None:
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline:
        result = run(["docker", "exec", container, "pg_isready", "-h", "127.0.0.1", "-U", "postgres", "-q"], timeout=15)
        if result.returncode == 0:
            return
        time.sleep(2)
    raise RuntimeError(f"scratch Postgres container {container} did not become ready")


def restore_dump(container: str, members: dict[str, bytes], db: str, owner: str) -> None:
    # Roles first, so owners and grants restore exactly; "already exists" notices are expected.
    run(["docker", "exec", "-i", container, "psql", "-U", "postgres", "-q", "-d", "postgres"],
        input_bytes=members["globals.sql"], timeout=120)
    created = run(["docker", "exec", container, "createdb", "-U", "postgres", "-O", owner, db], timeout=60)
    if created.returncode != 0:
        raise RuntimeError("createdb failed in the scratch container: "
                           + created.stderr.decode("utf-8", errors="replace").strip()[-300:])
    result = run(["docker", "exec", "-i", container, "pg_restore", "-U", "postgres", "-d", db, "--exit-on-error"],
                 input_bytes=members["db.dump"], timeout=900)
    if result.returncode != 0:
        raise RuntimeError("restore failed: " + result.stderr.decode("utf-8", errors="replace")[-1000:])


def query_restored_db(container: str, db: str) -> str:
    sql = (
        "SELECT 'tables=' || COUNT(*) "
        "FROM information_schema.tables WHERE table_schema='public'; "
        "SELECT 'alembic_versions=' || COUNT(*) FROM public.alembic_version;"
    )
    result = run(["docker", "exec", container, "psql", "-t", "-A", "-U", "postgres", "-d", db, "-c", sql], timeout=60)
    if result.returncode != 0:
        stderr = result.stderr.decode("utf-8", errors="replace")
        raise RuntimeError(f"restore query failed: {stderr[-1000:]}")
    output = result.stdout.decode("utf-8", errors="replace").strip()
    counts = dict(line.split("=", 1) for line in output.splitlines() if "=" in line)
    if int(counts.get("tables", "0")) < 1 or int(counts.get("alembic_versions", "0")) < 1:
        raise RuntimeError(f"restored database looks empty: {output!r}")
    return output


def send_alert(alert_py: str, message: str) -> None:
    if not Path(alert_py).is_file():
        print(f"ALERT NOT DELIVERED: {alert_py} not found", file=sys.stderr)
        return
    result = run([sys.executable, alert_py, message], timeout=60)
    print("Telegram alert " + ("SENT" if result.returncode == 0 else "NOT delivered"), file=sys.stderr)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--env-file", action="append", type=Path, default=[Path(".env.production")],
                        help="env file(s) with CLOUDFLARE_API_TOKEN + CLOUDFLARE_ACCOUNT_ID (later files win)")
    parser.add_argument("--bucket", default=DEFAULT_BUCKET)
    parser.add_argument("--prefix", default=DEFAULT_PREFIX)
    parser.add_argument("--passfile", type=Path,
                        default=Path(os.environ.get("VIOLAWAKE_BACKUP_PASSFILE", DEFAULT_PASSFILE)))
    parser.add_argument("--prod-container", default=DEFAULT_PROD_CONTAINER,
                        help="only inspected for its image tag; never written to")
    parser.add_argument("--postgres-image", default="", help="scratch image (default: the prod container's image)")
    parser.add_argument("--max-age-hours", type=int, default=36)
    parser.add_argument("--inspect-only", action="store_true",
                        help="Download, decrypt and check the archive but skip the Docker restore")
    parser.add_argument("--keep-container", action="store_true", help="Leave the scratch container running after restore")
    parser.add_argument("--alert", action="store_true", help="On failure, page Telegram via ops_alert.py")
    parser.add_argument("--alert-py", default=os.environ.get("VIOLA_OPS_ALERT_PY", DEFAULT_ALERT_PY))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    env = merged_env(args.env_file)
    try:
        objects = list_objects(env, args.bucket, args.prefix)
        if not objects:
            raise RuntimeError(f"No violawake-prod-*.tar.gz.enc backups found under r2://{args.bucket}/{args.prefix}/")

        latest = objects[0]
        key = str(latest["key"])
        taken = parse_object_time(key)
        assert taken is not None
        age_hours = (dt.datetime.now(dt.timezone.utc) - taken).total_seconds() / 3600
        print(f"Latest backup: r2://{args.bucket}/{key}")
        print(f"Backup age: {age_hours:.1f} hours")
        if age_hours > args.max_age_hours:
            raise RuntimeError(f"Latest backup is older than {args.max_age_hours} hours")

        blob = download_object(env, args.bucket, key)
        members = decrypt_archive(blob, args.passfile)
        print(f"Downloaded {len(blob)} encrypted bytes; db.dump={len(members['db.dump'])} bytes, "
              f"globals.sql={len(members['globals.sql'])} bytes")

        if args.inspect_only:
            print("Archive inspection OK (decrypts; db.dump is a pg_dump archive; globals.sql present)")
            return 0

        require_docker()
        image = scratch_image(args.prod_container, args.postgres_image)
        container = f"violawake-restore-drill-{int(time.time())}-{secrets.token_hex(3)}"
        print(f"Starting scratch Postgres container: {container} ({image}, --network none)")
        try:
            start_scratch_postgres(image, container)
            wait_for_postgres(container)
            restore_dump(container, members, DEFAULT_DB, DEFAULT_OWNER)
            query_output = query_restored_db(container, DEFAULT_DB)
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
        if args.alert:
            send_alert(args.alert_py, f"ViolaWake backup RESTORE DRILL FAILED: {exc}"[:900])
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
