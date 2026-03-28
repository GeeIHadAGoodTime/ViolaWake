"""Tests for violawake_sdk.training.temporal_model.

Covers:
  - Forward pass shape verification for all 3 architectures
  - Parameter counting
  - ONNX export function argument validation
  - Stub fallback when torch is unavailable
"""
from __future__ import annotations

import importlib
import sys
from unittest import mock

import pytest

torch = pytest.importorskip("torch")

from violawake_sdk.training.temporal_model import (  # noqa: E402
    TemporalCNN,
    TemporalConvGRU,
    TemporalGRU,
    count_parameters,
    export_temporal_onnx,
)


# ── Forward pass shape tests ─────────────────────────────────────────────────


class TestTemporalCNNForward:
    """TemporalCNN forward pass and shape verification."""

    def test_output_shape_default(self) -> None:
        """Default params: (batch=4, seq=9, emb=96) -> (4, 1)."""
        model = TemporalCNN()
        model.eval()
        x = torch.randn(4, 9, 96)
        out = model(x)
        assert out.shape == (4, 1)

    def test_output_shape_custom(self) -> None:
        """Custom embedding_dim and seq_len."""
        model = TemporalCNN(embedding_dim=64, seq_len=5, dropout=0.1)
        model.eval()
        x = torch.randn(2, 5, 64)
        out = model(x)
        assert out.shape == (2, 1)

    def test_output_range(self) -> None:
        """Output should be in [0, 1] due to sigmoid."""
        model = TemporalCNN()
        model.eval()
        x = torch.randn(8, 9, 96)
        out = model(x)
        assert out.min().item() >= 0.0
        assert out.max().item() <= 1.0

    def test_variable_seq_len(self) -> None:
        """Should accept seq_len different from the init value (>= 3 for kernel)."""
        model = TemporalCNN(embedding_dim=96, seq_len=9)
        model.eval()
        # seq_len=15, different from default 9
        x = torch.randn(2, 15, 96)
        out = model(x)
        assert out.shape == (2, 1)

    def test_single_sample_batch(self) -> None:
        """Batch size 1 should work."""
        model = TemporalCNN()
        model.eval()
        x = torch.randn(1, 9, 96)
        out = model(x)
        assert out.shape == (1, 1)


class TestTemporalGRUForward:
    """TemporalGRU forward pass and shape verification."""

    def test_output_shape_default(self) -> None:
        """Default params: (batch=4, seq=9, emb=96) -> (4, 1)."""
        model = TemporalGRU()
        model.eval()
        x = torch.randn(4, 9, 96)
        out = model(x)
        assert out.shape == (4, 1)

    def test_output_shape_custom(self) -> None:
        """Custom hidden_dim."""
        model = TemporalGRU(embedding_dim=48, hidden_dim=16, dropout=0.1)
        model.eval()
        x = torch.randn(3, 7, 48)
        out = model(x)
        assert out.shape == (3, 1)

    def test_output_range(self) -> None:
        """Output should be in [0, 1] due to sigmoid."""
        model = TemporalGRU()
        model.eval()
        x = torch.randn(8, 9, 96)
        out = model(x)
        assert out.min().item() >= 0.0
        assert out.max().item() <= 1.0

    def test_variable_seq_len(self) -> None:
        """GRU should handle variable sequence lengths."""
        model = TemporalGRU()
        model.eval()
        for seq_len in [3, 9, 20]:
            x = torch.randn(2, seq_len, 96)
            out = model(x)
            assert out.shape == (2, 1)


class TestTemporalConvGRUForward:
    """TemporalConvGRU forward pass and shape verification."""

    def test_output_shape_default(self) -> None:
        """Default params: (batch=4, seq=9, emb=96) -> (4, 1)."""
        model = TemporalConvGRU()
        model.eval()
        x = torch.randn(4, 9, 96)
        out = model(x)
        assert out.shape == (4, 1)

    def test_output_shape_custom(self) -> None:
        """Custom conv_channels and gru_hidden."""
        model = TemporalConvGRU(embedding_dim=64, conv_channels=32, gru_hidden=16, dropout=0.1)
        model.eval()
        x = torch.randn(2, 7, 64)
        out = model(x)
        assert out.shape == (2, 1)

    def test_output_range(self) -> None:
        """Output should be in [0, 1] due to sigmoid."""
        model = TemporalConvGRU()
        model.eval()
        x = torch.randn(8, 9, 96)
        out = model(x)
        assert out.min().item() >= 0.0
        assert out.max().item() <= 1.0

    def test_variable_seq_len(self) -> None:
        """Hybrid model should handle variable sequence lengths."""
        model = TemporalConvGRU()
        model.eval()
        for seq_len in [3, 9, 20]:
            x = torch.randn(2, seq_len, 96)
            out = model(x)
            assert out.shape == (2, 1)


# ── Parameter counting ───────────────────────────────────────────────────────


class TestCountParameters:
    """Verify parameter counts are within expected ranges (< 500K target)."""

    def test_temporal_cnn_params(self) -> None:
        """TemporalCNN should have ~12K params (docstring says ~12K)."""
        model = TemporalCNN()
        n = count_parameters(model)
        assert 5_000 < n < 50_000, f"TemporalCNN has {n} params, expected ~12K"

    def test_temporal_gru_params(self) -> None:
        """TemporalGRU should have ~28K params."""
        model = TemporalGRU()
        n = count_parameters(model)
        assert 10_000 < n < 100_000, f"TemporalGRU has {n} params, expected ~28K"

    def test_temporal_conv_gru_params(self) -> None:
        """TemporalConvGRU should have ~18K params."""
        model = TemporalConvGRU()
        n = count_parameters(model)
        assert 5_000 < n < 100_000, f"TemporalConvGRU has {n} params, expected ~18K"

    def test_all_under_500k(self) -> None:
        """All models must be under the 500K param budget."""
        for cls in [TemporalCNN, TemporalGRU, TemporalConvGRU]:
            model = cls()
            n = count_parameters(model)
            assert n < 500_000, f"{cls.__name__} has {n} params, exceeds 500K budget"


# ── ONNX export ──────────────────────────────────────────────────────────────


class TestExportTemporalOnnx:
    """Test ONNX export function."""

    def test_export_creates_file(self, tmp_path) -> None:
        """export_temporal_onnx should create an .onnx file."""
        model = TemporalCNN()
        out_path = tmp_path / "model.onnx"
        export_temporal_onnx(model, str(out_path), seq_len=9, embedding_dim=96)
        assert out_path.exists()
        assert out_path.stat().st_size > 0

    def test_export_creates_parent_dirs(self, tmp_path) -> None:
        """Should create parent directories if they don't exist."""
        model = TemporalGRU()
        out_path = tmp_path / "sub" / "dir" / "model.onnx"
        export_temporal_onnx(model, str(out_path))
        assert out_path.exists()

    def test_export_all_architectures(self, tmp_path) -> None:
        """All 3 architectures should export successfully."""
        for i, cls in enumerate([TemporalCNN, TemporalGRU, TemporalConvGRU]):
            model = cls()
            out_path = tmp_path / f"model_{i}.onnx"
            export_temporal_onnx(model, str(out_path))
            assert out_path.exists(), f"Failed to export {cls.__name__}"


# ── Stub fallback ────────────────────────────────────────────────────────────


class TestStubFallback:
    """Test that stubs raise ImportError when torch is unavailable."""

    def _reload_without_torch(self):
        """Reload temporal_model with torch hidden."""
        original_modules = {}
        for key in list(sys.modules.keys()):
            if key == "torch" or key.startswith("torch."):
                original_modules[key] = sys.modules.pop(key)
        return original_modules

    def _restore_torch(self, original_modules):
        """Restore torch modules and reload."""
        sys.modules.update(original_modules)
        import violawake_sdk.training.temporal_model as tm
        importlib.reload(tm)

    def test_stubs_raise_import_error(self) -> None:
        """All stubs should raise ImportError when torch is unavailable."""
        original_modules = self._reload_without_torch()
        try:
            with mock.patch.dict(sys.modules, {"torch": None, "torch.nn": None}):
                import violawake_sdk.training.temporal_model as tm
                reloaded = importlib.reload(tm)

                with pytest.raises(ImportError, match="PyTorch required"):
                    reloaded.TemporalCNN()
                with pytest.raises(ImportError, match="PyTorch required"):
                    reloaded.TemporalGRU()
                with pytest.raises(ImportError, match="PyTorch required"):
                    reloaded.TemporalConvGRU()
                with pytest.raises(ImportError, match="PyTorch required"):
                    reloaded.count_parameters(None)
                with pytest.raises(ImportError, match="PyTorch required"):
                    reloaded.export_temporal_onnx(None, "x.onnx")
        finally:
            self._restore_torch(original_modules)
