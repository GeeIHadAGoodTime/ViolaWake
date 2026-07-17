"""Gate: compose-db-host-unambiguous-on-shared-net.

The ViolaWake backend container joins two Docker networks: its own project
network (where the real ``wakeword-postgres-1`` container lives) and the shared
``novviola_viola-local`` network, joined so the backend can reach the shared
self-hosted GlitchTip at ``glitchtip-web:8080``. Viola's stack ALSO runs a
Postgres container aliased ``postgres`` on ``novviola_viola-local``.

When ``VIOLAWAKE_DB_URL`` addresses the database by the bare, unqualified host
``postgres``, Docker's embedded DNS resolves that name across ALL of the
container's attached networks -- and the answer is order-dependent on the
network-attachment timing at container-create. On an unlucky recreate it
resolves to Viola's ``postgres`` (172.18.0.2), a DIFFERENT server with no
``violawake`` role, so asyncpg opens a real TCP connection with a correctly
formed but foreign-server password and fails with ``InvalidPasswordError``.

This latent landmine took ViolaWake's backend down twice in two days --
2026-07-15 ~08:09 UTC (GlitchTip violawake issues 35/36/37) and again during
the 2026-07-16 17:16-17:27 UTC recalibration redeploy (~11 min of
api.violawake.com 502). Both times pooled connections kept serving while every
NEW physical connection auth-failed, so the outage looked partial and
intermittent. Root cause + durable fix (pin the host to the container name) is
commit 541590e; this gate is the ratchet that keeps the ambiguous shape from
coming back.

Rule enforced here: in any compose file that attaches the backend to the shared
``novviola_viola-local`` network, ``VIOLAWAKE_DB_URL`` must address Postgres by
its unambiguous container name (``wakeword-postgres-1``), which Docker
guarantees resolves to the one intended container on every network it joins.
The bare ``postgres`` alias is forbidden there. A standalone compose that does
NOT join ``novviola_viola-local`` may still use ``postgres`` -- the alias is
only ambiguous once the shared network is in play.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# The shared cross-stack network whose own `postgres` alias collides with ours.
SHARED_COLLIDING_NETWORK = "novviola_viola-local"
# The bare alias that Docker cannot disambiguate once the shared network is joined.
AMBIGUOUS_DB_HOST = "postgres"
# The unambiguous container name Docker always resolves to the intended server.
UNAMBIGUOUS_DB_HOST = "wakeword-postgres-1"

COMPOSE_FILES = (
    "docker-compose.production.yml",
    "docker-compose.viola-bridge.yml",
)

# postgresql+asyncpg://<user>:<pw>@<host>:<port>/<db>  -- capture <host>.
_DB_URL_HOST_RE = re.compile(r"VIOLAWAKE_DB_URL\s*=\s*[^@\s]*@([A-Za-z0-9_.-]+):\d+/")


def db_url_hosts(compose_text: str) -> list[str]:
    """Return every VIOLAWAKE_DB_URL host declared in the compose text."""
    return _DB_URL_HOST_RE.findall(compose_text)


def joins_shared_colliding_network(compose_text: str) -> bool:
    """True when the compose file wires anything onto the shared network."""
    return SHARED_COLLIDING_NETWORK in compose_text


def db_host_collision_violations(compose_text: str) -> list[str]:
    """Return VIOLAWAKE_DB_URL hosts that are ambiguous on the shared network.

    A host is a violation only when the compose file joins the shared
    ``novviola_viola-local`` network AND the DB host is the bare ``postgres``
    alias that Docker cannot disambiguate across networks. Outside the shared
    network the alias is unambiguous and allowed.
    """
    if not joins_shared_colliding_network(compose_text):
        return []
    return [host for host in db_url_hosts(compose_text) if host == AMBIGUOUS_DB_HOST]


def _read_compose(rel: str) -> str:
    return (REPO_ROOT / rel).read_text(encoding="utf-8")


def test_shipped_compose_files_pin_unambiguous_db_host() -> None:
    for rel in COMPOSE_FILES:
        text = _read_compose(rel)
        hosts = db_url_hosts(text)
        assert hosts, f"{rel}: expected at least one VIOLAWAKE_DB_URL host"
        assert db_host_collision_violations(text) == [], (
            f"{rel}: VIOLAWAKE_DB_URL uses the ambiguous {AMBIGUOUS_DB_HOST!r} alias "
            f"while the backend joins {SHARED_COLLIDING_NETWORK}; pin it to "
            f"{UNAMBIGUOUS_DB_HOST!r} (the 2026-07-15/16 InvalidPasswordError outages)."
        )


def test_detector_fails_on_the_pre_fix_shape() -> None:
    pre_fix = (
        "services:\n"
        "  backend:\n"
        "    environment:\n"
        "      - VIOLAWAKE_DB_URL=postgresql+asyncpg://violawake:"
        "${POSTGRES_PASSWORD}@postgres:5432/violawake\n"
        "    networks:\n"
        "      - default\n"
        f"      - {SHARED_COLLIDING_NETWORK}\n"
        "networks:\n"
        f"  {SHARED_COLLIDING_NETWORK}:\n"
        "    external: true\n"
    )
    assert db_host_collision_violations(pre_fix) == [AMBIGUOUS_DB_HOST], (
        "detector must flag the bare 'postgres' DB host when the backend also "
        "joins the shared novviola_viola-local network (the outage shape)."
    )


def test_detector_passes_on_the_fixed_and_benign_shapes() -> None:
    fixed = (
        "services:\n"
        "  backend:\n"
        "    environment:\n"
        "      - VIOLAWAKE_DB_URL=postgresql+asyncpg://violawake:"
        f"${{POSTGRES_PASSWORD}}@{UNAMBIGUOUS_DB_HOST}:5432/violawake\n"
        "    networks:\n"
        "      - default\n"
        f"      - {SHARED_COLLIDING_NETWORK}\n"
        "networks:\n"
        f"  {SHARED_COLLIDING_NETWORK}:\n"
        "    external: true\n"
    )
    standalone_benign = (
        "services:\n"
        "  backend:\n"
        "    environment:\n"
        "      - VIOLAWAKE_DB_URL=postgresql+asyncpg://violawake:"
        "${POSTGRES_PASSWORD}@postgres:5432/violawake\n"
        "    networks:\n"
        "      - default\n"
        "networks:\n"
        "  default:\n"
    )
    assert db_host_collision_violations(fixed) == [], (
        "the pinned wakeword-postgres-1 host must not be flagged"
    )
    assert db_host_collision_violations(standalone_benign) == [], (
        "a standalone compose that does not join the shared network may use the "
        "'postgres' alias safely"
    )
