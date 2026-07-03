"""Packaging hygiene gate: the published sdist must never contain secrets or
agent debris.

Root cause this gate exists for (2026-05-08 incident): violawake 0.2.4 shipped
its source distribution with ten stray ``.codex_log_*.txt`` codex-agent
transcripts in the repo root. Those transcripts contained a **live** Stripe
secret key, a Stripe webhook secret, a Cloudflare tunnel token, a Postgres
password, and the app secret key. The files were only gitignored starting at
the 0.2.5 release, so 0.2.4 leaked them publicly on PyPI.

Two independent failure modes let that happen and both are covered here:

1. A secret-bearing file whose *name* looks like debris (``.codex_log*``,
   ``.env``, ``.env.*.bak``) is not ignored and hatchling sweeps it into the
   sdist.
2. A secret sits inside a file with an *innocuous* name — a filename allowlist
   would miss it, so we also scan file **content** for secret shapes.

The gate builds the real sdist (the true artifact users download) and scans it,
and the ``scan_tree_for_leaks`` primitive is unit-tested against synthetic trees
so we prove it catches the broken shape and stays quiet on the fixed one.
"""

from __future__ import annotations

import re
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

# --- Forbidden file names (basename match, case-insensitive) --------------
# Debris / credential file shapes that must never be packaged. ``.env.example``
# and ``.env.*.example`` are the deliberate, secret-free exceptions.
_FORBIDDEN_NAME = re.compile(
    r"""^(
        \.codex .*            # codex agent transcripts / prompts
      | \.env (?! \.[a-z0-9_]*example$) (\.|$).*  # any .env* except *.example
      | secrets\.json
      | .*\.pem
      | id_rsa.*
      | .*\.p12
      | .*\.pfx
    )$""",
    re.IGNORECASE | re.VERBOSE,
)

# --- Forbidden content (secret shapes, scanned in every text member) ------
_FORBIDDEN_CONTENT = {
    "stripe_secret_key": re.compile(r"sk_(?:live|test)_[A-Za-z0-9]{20,}"),
    "stripe_restricted_key": re.compile(r"rk_(?:live|test)_[A-Za-z0-9]{20,}"),
    "stripe_webhook_secret": re.compile(r"whsec_[A-Za-z0-9]{20,}"),
    "postgres_url_with_creds": re.compile(r"postgres(?:ql)?://[^\s:/@]+:[^\s:/@]+@"),
    "aws_access_key_id": re.compile(r"AKIA[0-9A-Z]{16}"),
    "openai_key": re.compile(r"sk-(?:proj-)?[A-Za-z0-9]{32,}"),
    "anthropic_key": re.compile(r"sk-ant-[A-Za-z0-9_-]{24,}"),
    "google_api_key": re.compile(r"AIza[0-9A-Za-z_-]{35}"),
    "slack_token": re.compile(r"xox[baprs]-[0-9A-Za-z-]{10,}"),
    "private_key_block": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |PGP )?PRIVATE KEY-----"),
    # A real Cloudflare tunnel token is a long base64 JSON blob (``eyJ...``).
    # Require that shape so documentation placeholders like
    # ``CLOUDFLARE_TUNNEL_TOKEN=...`` or ``=<your-token>`` do not trip the gate.
    "cloudflare_tunnel_token": re.compile(
        r"CLOUDFLARE_TUNNEL_TOKEN\s*[=:]\s*[\"']?eyJ[A-Za-z0-9+/=_-]{40,}"
    ),
}

# Members with a known-binary extension are skipped for content scanning — a
# secret would not survive as an exact ASCII match there and models/audio are
# large. Everything else (including odd names like ``.env.production.bak_x``
# that have no clean extension) IS content-scanned, because that is exactly
# where the 0.2.4-style leak hid.
_BINARY_SUFFIXES = {
    ".onnx", ".tflite", ".pt", ".pth", ".npy", ".npz", ".bin", ".h5",
    ".wav", ".mp3", ".flac", ".ogg", ".opus", ".m4a",
    ".png", ".jpg", ".jpeg", ".gif", ".ico", ".webp", ".pdf",
    ".zip", ".gz", ".tar", ".whl", ".7z", ".xz",
    ".so", ".pyd", ".dll", ".dylib", ".pyc",
    ".woff", ".woff2", ".ttf", ".eot",
}


def scan_tree_for_leaks(members: list[tuple[str, bytes | None]]) -> list[str]:
    """Return a list of human-readable violations for the given archive members.

    ``members`` is a list of ``(arcname, data_or_None)``. ``data`` may be None
    for members we could not or need not read (directories); the name check
    still applies.
    """
    violations: list[str] = []
    for arcname, data in members:
        base = arcname.replace("\\", "/").rsplit("/", 1)[-1]
        if base and _FORBIDDEN_NAME.match(base):
            violations.append(f"forbidden filename: {arcname}")
        if data is None:
            continue
        suffix = ("." + base.rsplit(".", 1)[-1].lower()) if "." in base else ""
        if suffix in _BINARY_SUFFIXES:
            continue
        if b"\x00" in data[:4096]:  # looks binary — skip
            continue
        try:
            text = data.decode("utf-8", "replace")
        except Exception:  # pragma: no cover - defensive
            continue
        for label, pat in _FORBIDDEN_CONTENT.items():
            if pat.search(text):
                violations.append(f"{label} found in: {arcname}")
    return violations


def _tarball_members(path: Path) -> list[tuple[str, bytes | None]]:
    out: list[tuple[str, bytes | None]] = []
    with tarfile.open(path) as tf:
        for m in tf.getmembers():
            if not m.isfile():
                out.append((m.name, None))
                continue
            f = tf.extractfile(m)
            out.append((m.name, f.read() if f else None))
    return out


def _build_sdist(tmp: Path) -> Path:
    """Build the real repo sdist with the release backend, offline."""
    try:
        subprocess.run(
            [sys.executable, "-m", "build", "--sdist", "--no-isolation",
             "--outdir", str(tmp), str(REPO_ROOT)],
            check=True, capture_output=True, text=True, cwd=str(REPO_ROOT),
        )
    except FileNotFoundError:
        pytest.fail("`build` is not installed — it is required to run the sdist "
                    "hygiene gate. Install the [dev] extra.")
    except subprocess.CalledProcessError as e:
        pytest.fail(f"sdist build failed:\n{e.stdout}\n{e.stderr}")
    tarballs = sorted(tmp.glob("*.tar.gz"))
    assert tarballs, "sdist build produced no tarball"
    return tarballs[-1]


# --- The gate: the real published artifact must be clean ------------------

def test_release_sdist_contains_no_secrets_or_debris(tmp_path: Path) -> None:
    """FIXED-SHAPE: the actual sdist users would download is clean."""
    sdist = _build_sdist(tmp_path)
    violations = scan_tree_for_leaks(_tarball_members(sdist))
    assert not violations, (
        "Secrets/debris leaked into the published sdist:\n  "
        + "\n  ".join(violations)
    )


# --- Own-tests: prove the scanner catches the broken shape ----------------

def test_scanner_flags_codex_log_debris_by_name() -> None:
    """BROKEN-SHAPE #1: a debris-named file is flagged even if 'empty'."""
    members = [("violawake-9.9.9/.codex_log_inv2.txt", b"routine agent chatter\n")]
    assert any("forbidden filename" in v for v in scan_tree_for_leaks(members))


def test_scanner_flags_live_stripe_key_in_innocuous_file() -> None:
    """BROKEN-SHAPE #2: a secret inside a normally-named file is caught by
    the content scan, which a filename allowlist alone would miss."""
    # Assemble the fake key from fragments so the literal never appears in the
    # source. A real-looking `sk_live_…` string here would (correctly) be blocked
    # by GitHub push protection and secret scanners — the value only needs to
    # exist at runtime to exercise the scanner.
    fake_key = "sk_" + "live_" + "FAKE" + ("deadbeef" * 4)  # sk_live_ + 36 chars
    payload = (f"notes\nVIOLAWAKE_STRIPE_SECRET_KEY={fake_key}\n").encode()
    members = [("violawake-9.9.9/README.md", payload)]
    violations = scan_tree_for_leaks(members)
    assert any("stripe_secret_key" in v for v in violations), violations


def test_scanner_flags_env_backup_and_postgres_and_tunnel() -> None:
    """The exact 0.2.4 leak surface: .env.*.bak name + pg-with-creds +
    tunnel token content are all caught."""
    fake_tunnel = "eyJ" + "A1b2C3d4E5f6G7h8I9j0" * 3  # eyJ + 60 base64-ish chars
    fake_pg = "postgresql://viola:" + "hunter2" + "@db.internal:5432/wake"  # assembled, not a literal
    members = [
        (".env.production.bak_sentry_dsn",
         (f"CLOUDFLARE_TUNNEL_TOKEN={fake_tunnel}\n"
          f"DATABASE_URL={fake_pg}\n").encode()),
    ]
    violations = scan_tree_for_leaks(members)
    joined = " ".join(violations)
    assert "forbidden filename" in joined
    assert "cloudflare_tunnel_token" in joined
    assert "postgres_url_with_creds" in joined


def test_scanner_stays_quiet_on_clean_tree() -> None:
    """FIXED-SHAPE (unit): ordinary package files raise nothing — including a
    permitted .env.example and source code that merely mentions the var name."""
    members = [
        ("violawake-9.9.9/src/violawake_sdk/__init__.py", b'__version__ = "9.9.9"\n'),
        ("violawake-9.9.9/.env.example",
         ("VIOLAWAKE_STRIPE_SECRET_KEY=sk_" + "live_your_key_here\n").encode()),
        ("violawake-9.9.9/README.md", b"Set `VIOLAWAKE_STRIPE_SECRET_KEY` in your env.\n"),
        ("violawake-9.9.9/PKG-INFO", None),
    ]
    # NOTE: the .env.example placeholder uses `sk_live_your_key_here` which is
    # < 20 alnum chars after the prefix, so it does not match the secret shape;
    # a real key would. The filename is the *.example exception.
    assert scan_tree_for_leaks(members) == []
