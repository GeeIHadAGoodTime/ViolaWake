"""Regression tests for `tools/update_model_registry.py`.

The bug this locks in: the "Update model registry" release job
(`.github/workflows/release.yml`) hard-failed on two consecutive tagged
releases -- v0.2.9 (2026-06-04, run 26957961171) and v0.2.10 (2026-07-31, run
30672702094) -- with

    RuntimeError: Could not find release asset 'temporal_cnn.onnx' in local
    models or GitHub Release v<version>

Root cause: `resolve_asset_info` treated "this model has no matching asset in
the *current* release" as fatal, even though most releases (metadata-only
fixes, docs, non-model features) ship no `.onnx` assets at all, and several
registry entries (`oww_backbone`, the Kokoro TTS models) are never hosted as
assets on this repo's releases in the first place -- they would fail this
check on *every* release, forever.

The fix: a model with no local file and no matching asset on the current
release keeps its existing pinned registry entry untouched, instead of
raising. These tests exercise that behavior directly against the real
resolve/render functions (network calls mocked), plus the `quoted()`
ASCII-escaping bug the same code path exposed.
"""

from __future__ import annotations

import importlib.util
import sys
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "tools" / "update_model_registry.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("update_model_registry_under_test", SCRIPT)
    assert spec and spec.loader, f"cannot load {SCRIPT}"
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


umr = _load_module()


@dataclass(frozen=True)
class _FakeSpec:
    """Stand-in for `violawake_sdk.models.ModelSpec` — same field shape."""

    name: str
    url: str
    sha256: str
    size_bytes: int
    description: str
    version: str = "latest"


def _specs(*entries: _FakeSpec) -> "OrderedDict[str, _FakeSpec]":
    return OrderedDict((entry.name, entry) for entry in entries)


# ---------------------------------------------------------------------------
# resolve_asset_info: the shape that broke v0.2.9 and v0.2.10
# ---------------------------------------------------------------------------


def test_resolve_asset_info_skips_model_not_in_this_release(monkeypatch, tmp_path):
    """A metadata-only release (no .onnx assets at all) must not raise.

    This is the exact v0.2.9/v0.2.10 shape: no local file, and the current
    release's asset list doesn't contain the model's filename.
    """
    specs = _specs(
        _FakeSpec(
            name="temporal_cnn",
            url="https://github.com/GeeIHadAGoodTime/ViolaWake/releases/download/v0.1.0/temporal_cnn.onnx",
            sha256="deadbeef",
            size_bytes=102378,
            description="pinned to v0.1.0",
            version="0.1.0",
        )
    )

    monkeypatch.setattr(umr, "fetch_release_asset_metadata", lambda version: {})

    result = umr.resolve_asset_info("0.2.10", tmp_path / "empty_models", specs)

    assert result == {}, "model absent from this release must be skipped, not raise"


def test_resolve_asset_info_never_finds_a_release_only_hosted_elsewhere(monkeypatch, tmp_path):
    """oww_backbone / Kokoro-style entries (hosted on a different repo/never
    downloadable) must be skipped on every release, not just this one."""
    specs = _specs(
        _FakeSpec(
            name="oww_backbone",
            url="https://github.com/dscripka/openWakeWord/tree/main/openwakeword/resources",
            sha256="cafef00d",
            size_bytes=1_326_578,
            description="package-managed, never a release asset",
            version="0.6.0",
        )
    )

    # Even a release that DOES have assets shouldn't matter -- "resources"
    # (the URL's basename) will never be among them.
    monkeypatch.setattr(
        umr,
        "fetch_release_asset_metadata",
        lambda version: {"temporal_cnn.onnx": {"browser_download_url": "https://example.invalid/x"}},
    )

    result = umr.resolve_asset_info("0.2.10", tmp_path / "empty_models", specs)

    assert result == {}


def test_resolve_asset_info_uses_local_file_when_present(tmp_path):
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    payload = b"fake onnx bytes"
    (models_dir / "temporal_cnn.onnx").write_bytes(payload)

    specs = _specs(
        _FakeSpec(
            name="temporal_cnn",
            url="https://github.com/GeeIHadAGoodTime/ViolaWake/releases/download/v0.1.0/temporal_cnn.onnx",
            sha256="stale",
            size_bytes=1,
            description="d",
            version="0.1.0",
        )
    )

    result = umr.resolve_asset_info("0.3.0", models_dir, specs)

    assert "temporal_cnn" in result
    assert result["temporal_cnn"].size_bytes == len(payload)
    assert result["temporal_cnn"].sha256 == umr.hash_file(models_dir / "temporal_cnn.onnx")


def test_resolve_asset_info_hashes_remote_asset_actually_shipped_this_release(monkeypatch, tmp_path):
    specs = _specs(
        _FakeSpec(
            name="temporal_cnn",
            url="https://github.com/GeeIHadAGoodTime/ViolaWake/releases/download/v0.1.0/temporal_cnn.onnx",
            sha256="stale",
            size_bytes=1,
            description="d",
            version="0.1.0",
        )
    )

    monkeypatch.setattr(
        umr,
        "fetch_release_asset_metadata",
        lambda version: {
            "temporal_cnn.onnx": {
                "browser_download_url": "https://example.invalid/temporal_cnn.onnx",
                "size": 999,
            }
        },
    )
    monkeypatch.setattr(umr, "hash_remote_asset", lambda url: ("freshsha", 999))

    result = umr.resolve_asset_info("0.3.0", tmp_path / "empty_models", specs)

    assert result["temporal_cnn"].sha256 == "freshsha"
    assert result["temporal_cnn"].size_bytes == 999


# ---------------------------------------------------------------------------
# render_registry_block: skipped models must keep their OWN url/version, not
# get stamped with the new release tag they were never part of.
# ---------------------------------------------------------------------------


def test_render_registry_block_preserves_unresolved_entries_verbatim():
    specs = _specs(
        _FakeSpec(
            name="temporal_cnn",
            url="https://github.com/GeeIHadAGoodTime/ViolaWake/releases/download/v0.1.0/temporal_cnn.onnx",
            sha256="oldsha",
            size_bytes=102378,
            description="unchanged",
            version="0.1.0",
        )
    )

    # No assets resolved this release (the v0.2.10 shape).
    rendered = umr.render_registry_block("0.2.10", specs, assets={})

    assert "v0.1.0/temporal_cnn.onnx" in rendered, "must keep pointing at the release it was actually published in"
    assert "oldsha" in rendered
    assert '"0.1.0"' in rendered, "version field must stay pinned, not get stamped with the new tag"
    assert "v0.2.10" not in rendered


def test_render_registry_block_stamps_new_version_only_for_resolved_entries():
    specs = _specs(
        _FakeSpec(
            name="temporal_cnn",
            url="https://github.com/GeeIHadAGoodTime/ViolaWake/releases/download/v0.1.0/temporal_cnn.onnx",
            sha256="oldsha",
            size_bytes=102378,
            description="updated in this release",
            version="0.1.0",
        )
    )
    assets = {
        "temporal_cnn": umr.AssetInfo(filename="temporal_cnn.onnx", sha256="newsha", size_bytes=200_000)
    }

    rendered = umr.render_registry_block("0.3.0", specs, assets)

    assert "v0.3.0/temporal_cnn.onnx" in rendered
    assert "newsha" in rendered
    assert '"0.3.0"' in rendered
    assert "oldsha" not in rendered


# ---------------------------------------------------------------------------
# quoted(): non-ASCII descriptions (every entry uses an em-dash) must not get
# mangled into \uXXXX escapes on the next successful automated run.
# ---------------------------------------------------------------------------


def test_quoted_preserves_non_ascii_characters():
    rendered = umr.quoted("Temporal CNN — production default")
    assert "—" in rendered
    assert "\\u2014" not in rendered


# ---------------------------------------------------------------------------
# End-to-end: the real registry file is fully recoverable (no raise) when
# resolving against a release that ships no matching assets at all — the
# literal shape of v0.2.10.
# ---------------------------------------------------------------------------


def test_end_to_end_metadata_only_release_does_not_raise(monkeypatch, tmp_path):
    real_registry = REPO_ROOT / "src" / "violawake_sdk" / "models.py"
    scratch_registry = tmp_path / "models.py"
    scratch_registry.write_text(real_registry.read_text(encoding="utf-8"), encoding="utf-8")

    monkeypatch.setattr(umr, "fetch_release_asset_metadata", lambda version: {})
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "update_model_registry.py",
            "--version",
            "0.2.10",
            "--models-dir",
            str(tmp_path / "empty_models"),
            "--registry-path",
            str(scratch_registry),
        ],
    )

    exit_code = umr.main()

    assert exit_code == 0
    # Every entry in the real registry must have survived, semantically
    # unchanged (this is the exact registry file that broke CI twice).
    # Reuse the script's own loader (`registry_specs_in_order`) rather than a
    # second ad hoc import — it's the exact code path `main()` itself uses.
    original_specs = umr.registry_specs_in_order(real_registry)
    rewritten_specs = umr.registry_specs_in_order(scratch_registry)
    for name, spec in original_specs.items():
        new_spec = rewritten_specs[name]
        assert new_spec.sha256 == spec.sha256
        assert new_spec.size_bytes == spec.size_bytes
        assert new_spec.url == spec.url
        assert new_spec.version == spec.version


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
