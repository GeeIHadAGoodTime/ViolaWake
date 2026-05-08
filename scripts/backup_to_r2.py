#!/usr/bin/env python
"""Back up ViolaWake Postgres and app data to Cloudflare R2.

The script intentionally verifies bucket privacy before creating local backup
artifacts that may contain user data.
"""

from __future__ import annotations

import argparse
import datetime as dt
import gzip
import os
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


DEFAULT_BUCKET = "violawake-backups"
DEFAULT_POSTGRES_CONTAINER = "wakeword-postgres-1"
DEFAULT_BACKEND_CONTAINER = "wakeword-backend-1"
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


def cloudflare_api(
    *,
    token: str,
    method: str,
    path: str,
    body: bytes | None = None,
) -> dict[str, Any]:
    req = urllib.request.Request(
        f"https://api.cloudflare.com/client/v4{path}",
        data=body,
        method=method,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            import json

            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Cloudflare API {method} {path} failed: HTTP {exc.code} {detail}") from exc


def ensure_bucket_private(env: dict[str, str], bucket: str) -> None:
    account_id = env_first(env, "CLOUDFLARE_ACCOUNT_ID")
    token = env_first(env, "CLOUDFLARE_API_TOKEN")
    if not account_id or not token:
        raise RuntimeError("CLOUDFLARE_ACCOUNT_ID and CLOUDFLARE_API_TOKEN are required for bucket privacy verification")

    bucket_path = f"/accounts/{account_id}/r2/buckets/{bucket}"
    try:
        response = cloudflare_api(token=token, method="GET", path=bucket_path)
    except RuntimeError as exc:
        if "HTTP 404" not in str(exc):
            raise
        response = cloudflare_api(token=token, method="PUT", path=bucket_path, body=b"{}")

    if not response.get("success"):
        raise RuntimeError(f"Cloudflare did not confirm bucket {bucket}: {response}")

    result = response.get("result") or {}
    public_domain = result.get("public_domain") or result.get("publicDomain") or {}
    if isinstance(public_domain, dict) and public_domain.get("enabled"):
        raise RuntimeError(f"R2 bucket {bucket} has public domain access enabled")

    domains = result.get("domains") or result.get("custom_domains") or result.get("customDomains") or []
    if isinstance(domains, list):
        enabled_domains = [domain for domain in domains if isinstance(domain, dict) and domain.get("enabled")]
        if enabled_domains:
            raise RuntimeError(f"R2 bucket {bucket} has enabled custom domains")


def r2_client(env: dict[str, str], bucket: str):
    try:
        import boto3
        from botocore.config import Config as BotoConfig
    except ImportError as exc:
        raise RuntimeError("boto3 is required. Install console/backend requirements or run `python -m pip install boto3`.") from exc

    account_id = env_first(env, "CLOUDFLARE_ACCOUNT_ID")
    endpoint = env_first(
        env,
        "VIOLAWAKE_BACKUP_R2_ENDPOINT",
        "CLOUDFLARE_R2_ENDPOINT",
        "VIOLAWAKE_R2_ENDPOINT",
        default=f"https://{account_id}.r2.cloudflarestorage.com" if account_id else "",
    )
    access_key_id = env_first(
        env,
        "VIOLAWAKE_BACKUP_R2_ACCESS_KEY_ID",
        "CLOUDFLARE_R2_ACCESS_KEY_ID",
        "VIOLAWAKE_R2_ACCESS_KEY_ID",
        "AWS_ACCESS_KEY_ID",
    )
    secret_access_key = env_first(
        env,
        "VIOLAWAKE_BACKUP_R2_SECRET_ACCESS_KEY",
        "CLOUDFLARE_R2_SECRET_ACCESS_KEY",
        "VIOLAWAKE_R2_SECRET_ACCESS_KEY",
        "AWS_SECRET_ACCESS_KEY",
    )

    missing = [
        name
        for name, value in {
            "R2 endpoint": endpoint,
            "R2 access key id": access_key_id,
            "R2 secret access key": secret_access_key,
        }.items()
        if not value
    ]
    if missing:
        raise RuntimeError(f"Missing {', '.join(missing)} for private R2 object upload")

    client = boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=access_key_id,
        aws_secret_access_key=secret_access_key,
        region_name="auto",
        config=BotoConfig(signature_version="s3v4"),
    )
    client.head_bucket(Bucket=bucket)
    return client


def require_container(name: str) -> None:
    result = subprocess.run(
        ["docker", "inspect", "-f", "{{.State.Running}}", name],
        check=False,
        text=True,
        capture_output=True,
    )
    if result.returncode != 0 or result.stdout.strip().lower() != "true":
        raise RuntimeError(f"Docker container {name} is not running")


def dump_postgres(container: str, output_path: Path) -> None:
    command = [
        "docker",
        "exec",
        container,
        "pg_dump",
        "--clean",
        "--if-exists",
        "-U",
        "violawake",
        "violawake",
    ]
    process = subprocess.Popen(command, stdout=subprocess.PIPE)
    assert process.stdout is not None
    with gzip.open(output_path, "wb") as handle:
        shutil.copyfileobj(process.stdout, handle)
    rc = process.wait()
    if rc != 0:
        raise RuntimeError(f"pg_dump failed with exit code {rc}")


def archive_app_data(container: str, output_path: Path) -> None:
    command = ["docker", "exec", container, "tar", "-czf", "-", "-C", "/app/data", "."]
    with output_path.open("wb") as handle:
        result = subprocess.run(command, check=False, stdout=handle)
    if result.returncode != 0:
        raise RuntimeError(f"/app/data archive failed with exit code {result.returncode}")


def upload_file(client: Any, bucket: str, key: str, path: Path) -> None:
    client.upload_file(
        str(path),
        bucket,
        key,
        ExtraArgs={"ContentType": "application/gzip"},
    )


def retention_delete_old(client: Any, bucket: str, prefix: str, keep: int) -> list[str]:
    paginator = client.get_paginator("list_objects_v2")
    objects: list[tuple[dt.date, str]] = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for item in page.get("Contents", []):
            key = str(item["Key"])
            match = DATE_RE.search(Path(key).name)
            if not match:
                continue
            objects.append((dt.date.fromisoformat(match.group(1)), key))

    objects.sort(reverse=True)
    deleted: list[str] = []
    for _, key in objects[keep:]:
        client.delete_object(Bucket=bucket, Key=key)
        deleted.append(key)
    return deleted


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", action="append", type=Path, default=[Path(".env.production")])
    parser.add_argument("--bucket", default=DEFAULT_BUCKET)
    parser.add_argument("--postgres-container", default=DEFAULT_POSTGRES_CONTAINER)
    parser.add_argument("--backend-container", default=DEFAULT_BACKEND_CONTAINER)
    parser.add_argument("--retention-days", type=int, default=30)
    parser.add_argument("--check-only", action="store_true", help="Verify Cloudflare/R2 access without dumping data")
    parser.add_argument(
        "--allow-unverified-privacy",
        action="store_true",
        help="Allow backup when Cloudflare API privacy verification is unavailable",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    env = merged_env(args.env_file)

    try:
        try:
            ensure_bucket_private(env, args.bucket)
        except RuntimeError:
            if not args.allow_unverified_privacy:
                raise

        client = r2_client(env, args.bucket)
        if args.check_only:
            print(f"R2 backup check OK for bucket {args.bucket}")
            return 0

        require_container(args.postgres_container)
        require_container(args.backend_container)

        today = dt.datetime.now(dt.timezone.utc).date().isoformat()
        with tempfile.TemporaryDirectory(prefix="violawake-backup-") as tmp:
            tmp_path = Path(tmp)
            pg_path = tmp_path / f"{today}.sql.gz"
            app_path = tmp_path / f"{today}.tar.gz"

            dump_postgres(args.postgres_container, pg_path)
            archive_app_data(args.backend_container, app_path)

            pg_key = f"postgres/{today}.sql.gz"
            app_key = f"app-data/{today}.tar.gz"
            upload_file(client, args.bucket, pg_key, pg_path)
            upload_file(client, args.bucket, app_key, app_path)
            deleted = []
            deleted.extend(retention_delete_old(client, args.bucket, "postgres/", args.retention_days))
            deleted.extend(retention_delete_old(client, args.bucket, "app-data/", args.retention_days))

        print(f"Uploaded r2://{args.bucket}/{pg_key}")
        print(f"Uploaded r2://{args.bucket}/{app_key}")
        print(f"Deleted {len(deleted)} object(s) beyond {args.retention_days}-day retention")
        return 0
    except Exception as exc:
        print(f"backup_to_r2 failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
