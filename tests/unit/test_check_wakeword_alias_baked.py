"""Unit tests for scripts/check_wakeword_alias_baked.py (gate: wakeword-backend-alias-baked-in-base).

Proves the detector fails on every pre-fix compose shape (bare list, missing
network, mapping without the alias) and passes only when
docker-compose.production.yml self-carries the wakeword-backend alias --
without needing docker-compose.viola-bridge.yml merged in.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "scripts"))

from check_wakeword_alias_baked import (  # noqa: E402
    ComposeAliasError,
    check_alias_baked,
)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def _write_compose(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "docker-compose.production.yml"
    path.write_text(body, encoding="utf-8")
    return path


def test_real_base_compose_file_passes() -> None:
    """The actual repo file, post-fix, must pass with no overlay merged in."""
    check_alias_baked(REPO_ROOT / "docker-compose.production.yml")


def test_bare_list_networks_shape_fails(tmp_path: Path) -> None:
    """The exact pre-fix shape: alias lived only in the bridge overlay."""
    compose = _write_compose(
        tmp_path,
        """
services:
  backend:
    networks:
      - default
      - decoder-net
      - novviola_viola-local
""",
    )
    with pytest.raises(ComposeAliasError, match="bare list"):
        check_alias_baked(compose)


def test_network_missing_entirely_fails(tmp_path: Path) -> None:
    compose = _write_compose(
        tmp_path,
        """
services:
  backend:
    networks:
      default: {}
      decoder-net: {}
""",
    )
    with pytest.raises(ComposeAliasError, match="does not attach"):
        check_alias_baked(compose)


def test_mapping_without_alias_key_fails(tmp_path: Path) -> None:
    compose = _write_compose(
        tmp_path,
        """
services:
  backend:
    networks:
      default: {}
      decoder-net: {}
      novviola_viola-local: {}
""",
    )
    with pytest.raises(ComposeAliasError, match="does not include"):
        check_alias_baked(compose)


def test_alias_present_but_wrong_name_fails(tmp_path: Path) -> None:
    compose = _write_compose(
        tmp_path,
        """
services:
  backend:
    networks:
      default: {}
      decoder-net: {}
      novviola_viola-local:
        aliases:
          - some-other-name
""",
    )
    with pytest.raises(ComposeAliasError, match="does not include"):
        check_alias_baked(compose)


def test_fixed_shape_passes(tmp_path: Path) -> None:
    """The fixed shape: alias baked directly into the base file."""
    compose = _write_compose(
        tmp_path,
        """
services:
  backend:
    networks:
      default: {}
      decoder-net: {}
      novviola_viola-local:
        aliases:
          - wakeword-backend
""",
    )
    check_alias_baked(compose)


def test_missing_file_fails(tmp_path: Path) -> None:
    with pytest.raises(ComposeAliasError, match="not found"):
        check_alias_baked(tmp_path / "does-not-exist.yml")


def test_missing_backend_service_fails(tmp_path: Path) -> None:
    compose = _write_compose(
        tmp_path,
        """
services:
  postgres:
    image: postgres:16-alpine
""",
    )
    with pytest.raises(ComposeAliasError, match="no 'backend' service"):
        check_alias_baked(compose)
