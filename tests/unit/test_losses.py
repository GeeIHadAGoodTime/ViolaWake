"""Tests for violawake_sdk.training.losses — FocalLoss.

Covers:
  - Forward pass with known inputs/outputs (torch path)
  - Various gamma/alpha/label_smoothing values
  - Edge cases: all-zeros, all-ones inputs
  - Stub fallback when torch is unavailable
"""
from __future__ import annotations

import importlib
import sys
from unittest import mock

import numpy as np
import pytest

# ── Torch-available tests ────────────────────────────────────────────────────

torch = pytest.importorskip("torch")

from violawake_sdk.training.losses import FocalLoss  # noqa: E402


class TestFocalLossForward:
    """Forward pass correctness with the real PyTorch implementation."""

    def test_basic_shape(self) -> None:
        """Output should be a scalar (0-dim tensor)."""
        loss_fn = FocalLoss()
        inputs = torch.tensor([0.9, 0.1, 0.8, 0.2])
        targets = torch.tensor([1.0, 0.0, 1.0, 0.0])
        loss = loss_fn(inputs, targets)
        assert loss.dim() == 0, "FocalLoss should return a scalar"

    def test_perfect_predictions_low_loss(self) -> None:
        """Perfect predictions should yield a very small loss."""
        loss_fn = FocalLoss(gamma=2.0, alpha=0.75, label_smoothing=0.0)
        inputs = torch.tensor([0.999, 0.001, 0.999, 0.001])
        targets = torch.tensor([1.0, 0.0, 1.0, 0.0])
        loss = loss_fn(inputs, targets)
        assert loss.item() < 0.01, f"Perfect predictions should have near-zero loss, got {loss.item()}"

    def test_wrong_predictions_high_loss(self) -> None:
        """Completely wrong predictions should yield high loss."""
        loss_fn = FocalLoss(gamma=2.0, alpha=0.75, label_smoothing=0.0)
        inputs = torch.tensor([0.01, 0.99, 0.01, 0.99])
        targets = torch.tensor([1.0, 0.0, 1.0, 0.0])
        loss = loss_fn(inputs, targets)
        # Even with focal weighting, wrong predictions should have noticeable loss
        assert loss.item() > 0.1

    def test_known_value_no_smoothing(self) -> None:
        """Verify FocalLoss matches manual computation for a single sample.

        With gamma=0, alpha=1.0, label_smoothing=0.0, FocalLoss reduces to
        standard weighted BCE: -alpha * [t*log(p) + (1-t)*log(1-p)]
        For t=1, p=0.8: loss = -1.0 * log(0.8) ≈ 0.2231
        """
        loss_fn = FocalLoss(gamma=0.0, alpha=1.0, label_smoothing=0.0)
        p = torch.tensor([0.8])
        t = torch.tensor([1.0])
        loss = loss_fn(p, t)
        expected = -np.log(0.8)
        np.testing.assert_allclose(loss.item(), expected, rtol=1e-5)

    def test_gamma_effect(self) -> None:
        """Higher gamma should down-weight easy examples (reduce loss for high-confidence correct predictions)."""
        inputs = torch.tensor([0.9, 0.1])  # confident correct predictions
        targets = torch.tensor([1.0, 0.0])

        loss_g0 = FocalLoss(gamma=0.0, alpha=0.5, label_smoothing=0.0)(inputs, targets)
        loss_g2 = FocalLoss(gamma=2.0, alpha=0.5, label_smoothing=0.0)(inputs, targets)
        loss_g5 = FocalLoss(gamma=5.0, alpha=0.5, label_smoothing=0.0)(inputs, targets)

        # Higher gamma should give lower loss for confident correct predictions
        assert loss_g2.item() < loss_g0.item()
        assert loss_g5.item() < loss_g2.item()

    def test_alpha_effect(self) -> None:
        """Alpha controls positive vs negative class weighting.

        With asymmetric predictions, changing alpha should change the loss.
        Use only positives so alpha directly scales the loss.
        """
        # All positive targets with imperfect predictions
        inputs = torch.tensor([0.7, 0.6, 0.8])
        targets = torch.tensor([1.0, 1.0, 1.0])

        loss_a025 = FocalLoss(gamma=0.0, alpha=0.25, label_smoothing=0.0)(inputs, targets)
        loss_a075 = FocalLoss(gamma=0.0, alpha=0.75, label_smoothing=0.0)(inputs, targets)

        # Higher alpha means higher weight on positive class loss
        assert loss_a075.item() > loss_a025.item()

    def test_label_smoothing_raises_loss(self) -> None:
        """Label smoothing should increase loss compared to no smoothing for perfect predictions."""
        inputs = torch.tensor([0.99, 0.01])
        targets = torch.tensor([1.0, 0.0])

        loss_no_smooth = FocalLoss(gamma=2.0, alpha=0.75, label_smoothing=0.0)(inputs, targets)
        loss_smooth = FocalLoss(gamma=2.0, alpha=0.75, label_smoothing=0.1)(inputs, targets)

        # Smoothing moves targets away from 0/1, making "perfect" predictions less perfect
        assert loss_smooth.item() > loss_no_smooth.item()


class TestFocalLossEdgeCases:
    """Edge case inputs."""

    def test_all_zeros_inputs(self) -> None:
        """All-zero inputs (predicting 0.0 for everything)."""
        loss_fn = FocalLoss()
        inputs = torch.zeros(4)
        targets = torch.tensor([1.0, 0.0, 1.0, 0.0])
        loss = loss_fn(inputs, targets)
        assert torch.isfinite(loss), "Loss should be finite for all-zero inputs"

    def test_all_ones_inputs(self) -> None:
        """All-one inputs (predicting 1.0 for everything)."""
        loss_fn = FocalLoss()
        inputs = torch.ones(4)
        targets = torch.tensor([1.0, 0.0, 1.0, 0.0])
        loss = loss_fn(inputs, targets)
        assert torch.isfinite(loss), "Loss should be finite for all-one inputs"

    def test_all_same_target(self) -> None:
        """All targets are the same class."""
        loss_fn = FocalLoss()
        inputs = torch.tensor([0.9, 0.8, 0.7])
        targets = torch.ones(3)
        loss = loss_fn(inputs, targets)
        assert torch.isfinite(loss)
        assert loss.item() >= 0.0

    def test_single_sample(self) -> None:
        """Single sample should work."""
        loss_fn = FocalLoss()
        loss = loss_fn(torch.tensor([0.5]), torch.tensor([1.0]))
        assert torch.isfinite(loss)

    def test_batch_of_mixed_predictions(self) -> None:
        """Larger batch with diverse predictions."""
        loss_fn = FocalLoss()
        rng = np.random.default_rng(42)
        inputs = torch.tensor(rng.uniform(0, 1, 100).astype(np.float32))
        targets = torch.tensor(rng.choice([0.0, 1.0], 100).astype(np.float32))
        loss = loss_fn(inputs, targets)
        assert torch.isfinite(loss)
        assert loss.item() >= 0.0

    def test_gradient_flows(self) -> None:
        """Gradients should flow through the loss."""
        loss_fn = FocalLoss()
        inputs = torch.tensor([0.5, 0.5], requires_grad=True)
        targets = torch.tensor([1.0, 0.0])
        loss = loss_fn(inputs, targets)
        loss.backward()
        assert inputs.grad is not None
        assert torch.all(torch.isfinite(inputs.grad))


class TestFocalLossStubFallback:
    """Test the stub class that is used when torch is not available.

    We simulate the no-torch path by reloading the module with torch
    import patched out.
    """

    def test_stub_raises_import_error(self) -> None:
        """When torch is unavailable, FocalLoss() should raise ImportError."""
        # Temporarily hide torch and reload the losses module
        original_modules = {}
        for key in list(sys.modules.keys()):
            if key == "torch" or key.startswith("torch."):
                original_modules[key] = sys.modules.pop(key)

        try:
            with mock.patch.dict(sys.modules, {"torch": None, "torch.nn": None}):
                # Force reload to trigger the try/except path
                import violawake_sdk.training.losses as losses_mod
                reloaded = importlib.reload(losses_mod)
                with pytest.raises(ImportError, match="PyTorch required"):
                    reloaded.FocalLoss()
        finally:
            # Restore torch modules
            sys.modules.update(original_modules)
            # Reload to restore normal state
            import violawake_sdk.training.losses as losses_mod2
            importlib.reload(losses_mod2)
