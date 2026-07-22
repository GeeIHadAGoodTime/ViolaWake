#!/usr/bin/env python3
"""Assert docker-compose.production.yml self-carries the wakeword-backend alias.

Why: viola-api reaches this repo's backend over Docker DNS at
http://wakeword-backend:8000 (VIOLA_WAKEWORD_API_URL). That alias used to
live ONLY in the separate docker-compose.viola-bridge.yml overlay, which is
NOT part of the documented deploy command in docs/DEPLOYMENT.md
(`docker compose -f docker-compose.production.yml up -d backend`). A
routine base-only deploy silently dropped the alias while /health stayed
green -- the same base-only-deploy-drops-out-of-band-wiring class as
#1722's Postgres-host outage (twice-recurred, see #2305).

This gate parses docker-compose.production.yml ALONE (no overlay merged
in) and fails unless the `backend` service's `novviola_viola-local` network
entry declares `wakeword-backend` as an alias -- so the alias can never
again silently regress to living only in the optional overlay.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_COMPOSE_FILE = REPO_ROOT / "docker-compose.production.yml"
TARGET_NETWORK = "novviola_viola-local"
REQUIRED_ALIAS = "wakeword-backend"
TARGET_SERVICE = "backend"


class ComposeAliasError(Exception):
    """Raised when the base compose file does not self-carry the alias."""


def _load_compose(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ComposeAliasError(f"compose file not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ComposeAliasError(f"{path} did not parse to a mapping")
    return data


def check_alias_baked(path: Path = DEFAULT_COMPOSE_FILE) -> None:
    """Raise ComposeAliasError unless the base file alone carries the alias."""
    data = _load_compose(path)

    services = data.get("services")
    if not isinstance(services, dict) or TARGET_SERVICE not in services:
        raise ComposeAliasError(f"{path} has no '{TARGET_SERVICE}' service -- cannot verify alias")

    service = services[TARGET_SERVICE]
    networks = service.get("networks")

    if networks is None:
        raise ComposeAliasError(f"{TARGET_SERVICE} service in {path} declares no networks at all")

    if isinstance(networks, list):
        # Old shape: bare list of network names, e.g. `- novviola_viola-local`.
        # This form cannot carry an alias, which is exactly the regression
        # this gate exists to catch.
        raise ComposeAliasError(
            f"{TARGET_SERVICE}.networks in {path} is a bare list "
            f"({networks!r}) and cannot declare an alias for "
            f"'{TARGET_NETWORK}'. Use the mapping form with an "
            f"'aliases: [{REQUIRED_ALIAS}]' entry."
        )

    if not isinstance(networks, dict) or TARGET_NETWORK not in networks:
        raise ComposeAliasError(
            f"{TARGET_SERVICE}.networks in {path} does not attach "
            f"'{TARGET_NETWORK}' at all: {networks!r}"
        )

    net_config = networks[TARGET_NETWORK]
    if not isinstance(net_config, dict):
        raise ComposeAliasError(
            f"{TARGET_SERVICE}.networks.{TARGET_NETWORK} in {path} is "
            f"{net_config!r}, not a mapping -- cannot carry an "
            f"'aliases' key, so '{REQUIRED_ALIAS}' cannot be declared."
        )

    aliases = net_config.get("aliases")
    if not aliases or REQUIRED_ALIAS not in aliases:
        raise ComposeAliasError(
            f"{TARGET_SERVICE}.networks.{TARGET_NETWORK}.aliases in {path} "
            f"is {aliases!r} and does not include '{REQUIRED_ALIAS}'. "
            "A base-only `docker compose -f docker-compose.production.yml "
            "up -d backend` deploy (docs/DEPLOYMENT.md) would silently "
            "leave viola-api unable to resolve wakeword-backend."
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--compose-file",
        type=Path,
        default=DEFAULT_COMPOSE_FILE,
        help="Path to the base compose file to check (default: %(default)s)",
    )
    args = parser.parse_args(argv)

    try:
        check_alias_baked(args.compose_file)
    except ComposeAliasError as exc:
        print(f"WAKEWORD-ALIAS-NOT-BAKED: {exc}", file=sys.stderr)
        return 1

    print(f"OK: {args.compose_file} self-carries the '{REQUIRED_ALIAS}' alias")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
