"""Gate: console-sqlalchemy-asyncio-install.

The console imports SQLAlchemy's asyncio extension. SQLAlchemy 2.1 stopped
installing greenlet by default, so the backend manifest must request the
asyncio extra even when another installed package happens to provide greenlet.
This packaging check needs no console dependencies or live services.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from packaging.requirements import Requirement

REPO_ROOT = Path(__file__).resolve().parents[2]
BACKEND_REQUIREMENTS = REPO_ROOT / "console" / "backend" / "requirements.txt"


def has_sqlalchemy_asyncio_requirement(requirements: str) -> bool:
    """Check active PEP 508 declarations for SQLAlchemy's asyncio extra."""
    for raw_line in requirements.splitlines():
        line = raw_line.partition("#")[0].strip()
        if not line:
            continue
        requirement = Requirement(line)
        if requirement.name.lower() != "sqlalchemy":
            continue
        if requirement.marker is not None and not requirement.marker.evaluate():
            continue
        if "asyncio" in {extra.lower() for extra in requirement.extras}:
            return True
    return False


def test_backend_requests_sqlalchemy_asyncio_extra() -> None:
    assert has_sqlalchemy_asyncio_requirement(BACKEND_REQUIREMENTS.read_text(encoding="utf-8")), (
        "The console uses sqlalchemy.ext.asyncio; declare sqlalchemy[asyncio] "
        "in console/backend/requirements.txt so fresh installs include greenlet."
    )


@pytest.mark.parametrize(
    "requirements",
    [
        "fastapi>=0.110\naiosqlite>=0.19\n",
        "sqlalchemy>=2.0\n",
        "sqlalchemy[postgresql]>=2.0\n",
        "# sqlalchemy[asyncio]>=2.0\nsqlalchemy>=2.0\n",
        'sqlalchemy[asyncio]>=2.0; python_version < "0"\n',
    ],
)
def test_detector_rejects_missing_asyncio_dependency(requirements: str) -> None:
    assert not has_sqlalchemy_asyncio_requirement(requirements)


@pytest.mark.parametrize(
    "requirements",
    [
        "sqlalchemy[asyncio]>=2.0\n",
        "SQLAlchemy[asyncio]>=2.0 # required by the async database\n",
        "SQLAlchemy[AsyncIO]>=2.0\n",
        "sqlalchemy[asyncio,postgresql]>=2.0\n",
        "# backend dependencies\naiosqlite>=0.19\nsqlalchemy[asyncio]>=2.0\n",
        'sqlalchemy[asyncio]>=2.0; python_version >= "3.10"\n',
    ],
)
def test_detector_accepts_asyncio_dependency(requirements: str) -> None:
    assert has_sqlalchemy_asyncio_requirement(requirements)
