"""Tests for SWA/EMA weight averaging (J4).

Verifies that:
  - EMATracker maintains correct shadow parameters
  - EMATracker apply/restore roundtrips correctly
  - EMATracker warmup-adjusted decay works
  - SWACollector computes correct running mean
  - SWACollector apply replaces model weights
  - auto_select_averaging picks the method with lowest val loss
  - State dict save/load works for both EMA and SWA
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch", reason="torch required for weight averaging tests")
import torch.nn as nn

from violawake_sdk.training.weight_averaging import (
    EMATracker,
    SWACollector,
    auto_select_averaging,
)


@pytest.fixture
def simple_model() -> nn.Module:
    """A small MLP matching the wake word architecture."""
    return nn.Sequential(
        nn.Linear(8, 4),
        nn.ReLU(),
        nn.Linear(4, 1),
        nn.Sigmoid(),
    )


@pytest.fixture
def two_layer_model() -> nn.Module:
    """Another small model for comparison tests."""
    return nn.Sequential(
        nn.Linear(8, 4),
        nn.ReLU(),
        nn.Linear(4, 1),
        nn.Sigmoid(),
    )


# ── EMATracker Tests ──────────────────────────────────────────────────────────


class TestEMATracker:
    """Test the Exponential Moving Average tracker."""

    def test_initialization(self, simple_model: nn.Module) -> None:
        """EMA should initialize shadow params as copies of model params."""
        ema = EMATracker(simple_model, decay=0.999)
        for name, param in simple_model.named_parameters():
            torch.testing.assert_close(ema.shadow[name], param.data)
        assert ema.num_updates == 0

    def test_update_changes_shadow(self, simple_model: nn.Module) -> None:
        """After updating model params and calling update(), shadow should change."""
        ema = EMATracker(simple_model, decay=0.99)
        # Store initial shadow
        initial_shadow = {k: v.clone() for k, v in ema.shadow.items()}

        # Modify model parameters
        with torch.no_grad():
            for param in simple_model.parameters():
                param.add_(torch.randn_like(param))

        ema.update()
        assert ema.num_updates == 1

        # Shadow should have moved toward new params
        for name in initial_shadow:
            assert not torch.equal(ema.shadow[name], initial_shadow[name])

    def test_high_decay_warmup_moves_fast(self, simple_model: nn.Module) -> None:
        """Warmup-adjusted decay causes fast early movement toward new params.

        The warmup formula adjusts decay to min(0.999, (1+n)/(10+n)),
        so early updates use a much lower effective decay (~0.18 for n=1).
        This means the shadow moves rapidly toward the target, even with a
        nominal decay of 0.999.  After 2 updates the shadow should be
        closer to the target (100.0) than to the initial values.
        """
        ema = EMATracker(simple_model, decay=0.999)
        initial_shadow = {k: v.clone() for k, v in ema.shadow.items()}

        # Make a large change
        with torch.no_grad():
            for param in simple_model.parameters():
                param.fill_(100.0)

        # 2 warmup updates -- effective decay is ~0.18 then ~0.25
        for _ in range(2):
            ema.update()

        # Shadow should have moved most of the way toward 100.0
        for name in initial_shadow:
            dist_to_initial = (ema.shadow[name] - initial_shadow[name]).abs().mean()
            dist_to_100 = (ema.shadow[name] - 100.0).abs().mean()
            # After 2 low-decay warmup updates, shadow is closer to 100 than to initial
            assert dist_to_100 < dist_to_initial

    def test_apply_replaces_model_params(self, simple_model: nn.Module) -> None:
        """apply() should replace model params with shadow params."""
        ema = EMATracker(simple_model, decay=0.99)

        # Modify model params away from shadow
        with torch.no_grad():
            for param in simple_model.parameters():
                param.add_(torch.randn_like(param) * 5.0)

        # Shadow is still the original; apply should restore
        ema.apply()
        for name, param in simple_model.named_parameters():
            torch.testing.assert_close(param.data, ema.shadow[name])

    def test_restore_after_apply(self, simple_model: nn.Module) -> None:
        """restore() should put back the pre-apply params."""
        ema = EMATracker(simple_model, decay=0.99)

        # Save original params
        original_params = {
            name: param.data.clone() for name, param in simple_model.named_parameters()
        }

        # Modify params
        with torch.no_grad():
            for param in simple_model.parameters():
                param.add_(1.0)

        modified_params = {
            name: param.data.clone() for name, param in simple_model.named_parameters()
        }

        ema.apply()
        ema.restore()

        # Should be back to the modified params (not original, not EMA)
        for name, param in simple_model.named_parameters():
            torch.testing.assert_close(param.data, modified_params[name])

    def test_restore_without_apply_raises(self, simple_model: nn.Module) -> None:
        """restore() before apply() should raise RuntimeError."""
        ema = EMATracker(simple_model)
        with pytest.raises(RuntimeError, match="apply.*not called"):
            ema.restore()

    def test_state_dict_roundtrip(self, simple_model: nn.Module) -> None:
        """state_dict and load_state_dict should roundtrip correctly."""
        ema = EMATracker(simple_model, decay=0.99)

        # Do some updates
        with torch.no_grad():
            for param in simple_model.parameters():
                param.add_(torch.randn_like(param))
        ema.update()
        ema.update()

        # Save and reload
        saved = ema.state_dict()
        ema2 = EMATracker(simple_model, decay=0.99)
        ema2.load_state_dict(saved)

        for name in saved:
            torch.testing.assert_close(ema2.shadow[name], saved[name])

    def test_warmup_adjusted_decay(self, simple_model: nn.Module) -> None:
        """Early updates should use lower effective decay (warmup)."""
        ema = EMATracker(simple_model, decay=0.999)

        # On first update, effective decay = min(0.999, 2/11) = 0.1818
        # which means shadow moves significantly toward new params
        initial_shadow = {k: v.clone() for k, v in ema.shadow.items()}

        with torch.no_grad():
            for param in simple_model.parameters():
                param.fill_(10.0)

        ema.update()

        # With warmup decay ~0.18, shadow should move significantly toward 10.0
        for name in initial_shadow:
            # Should have moved more than if using full 0.999 decay
            movement = (ema.shadow[name] - initial_shadow[name]).abs().mean()
            assert movement > 0.1  # significant movement


# ── SWACollector Tests ─────────────────────────────────────────────────────────


class TestSWACollector:
    """Test the Stochastic Weight Averaging collector."""

    def test_single_collect(self, simple_model: nn.Module) -> None:
        """Single collection should just copy the params."""
        swa = SWACollector()
        swa.collect(simple_model)
        assert swa.n_collected == 1
        for name, param in simple_model.named_parameters():
            torch.testing.assert_close(swa._avg_params[name], param.data)

    def test_two_collects_averages(self, simple_model: nn.Module) -> None:
        """Two collections should produce the arithmetic mean."""
        swa = SWACollector()

        # First snapshot
        with torch.no_grad():
            for param in simple_model.parameters():
                param.fill_(2.0)
        swa.collect(simple_model)

        # Second snapshot
        with torch.no_grad():
            for param in simple_model.parameters():
                param.fill_(4.0)
        swa.collect(simple_model)

        assert swa.n_collected == 2
        # Average of 2.0 and 4.0 should be 3.0
        for name in swa._avg_params:
            torch.testing.assert_close(
                swa._avg_params[name],
                torch.full_like(swa._avg_params[name], 3.0),
            )

    def test_three_collects_running_mean(self, simple_model: nn.Module) -> None:
        """Three collections should compute correct running mean."""
        swa = SWACollector()
        values = [1.0, 4.0, 7.0]

        for val in values:
            with torch.no_grad():
                for param in simple_model.parameters():
                    param.fill_(val)
            swa.collect(simple_model)

        expected_mean = sum(values) / len(values)  # 4.0
        assert swa.n_collected == 3
        for name in swa._avg_params:
            torch.testing.assert_close(
                swa._avg_params[name],
                torch.full_like(swa._avg_params[name], expected_mean),
                atol=1e-5,
                rtol=1e-5,
            )

    def test_apply_replaces_model_weights(self, simple_model: nn.Module) -> None:
        """apply() should set model params to the SWA average."""
        swa = SWACollector()

        with torch.no_grad():
            for param in simple_model.parameters():
                param.fill_(2.0)
        swa.collect(simple_model)

        with torch.no_grad():
            for param in simple_model.parameters():
                param.fill_(6.0)
        swa.collect(simple_model)

        # Model currently has 6.0, SWA avg is 4.0
        swa.apply(simple_model)

        for name, param in simple_model.named_parameters():
            torch.testing.assert_close(
                param.data,
                torch.full_like(param.data, 4.0),
                atol=1e-5,
                rtol=1e-5,
            )

    def test_apply_without_collect_raises(self, simple_model: nn.Module) -> None:
        """apply() before any collect() should raise RuntimeError."""
        swa = SWACollector()
        with pytest.raises(RuntimeError, match="No parameters collected"):
            swa.apply(simple_model)

    def test_val_loss_tracking(self, simple_model: nn.Module) -> None:
        """val_loss property should return mean of tracked losses."""
        swa = SWACollector()
        swa.collect(simple_model, val_loss=0.5)
        swa.collect(simple_model, val_loss=0.3)
        swa.collect(simple_model, val_loss=0.1)
        assert swa.val_loss == pytest.approx(0.3, abs=1e-6)

    def test_val_loss_none_when_empty(self) -> None:
        """val_loss should be None if no losses tracked."""
        swa = SWACollector()
        assert swa.val_loss is None

    def test_state_dict(self, simple_model: nn.Module) -> None:
        """state_dict should return a copy of averaged params."""
        swa = SWACollector()
        with torch.no_grad():
            for param in simple_model.parameters():
                param.fill_(3.0)
        swa.collect(simple_model)

        state = swa.state_dict()
        assert len(state) > 0
        for name in state:
            torch.testing.assert_close(
                state[name],
                torch.full_like(state[name], 3.0),
            )


# ── Auto-Selection Tests ──────────────────────────────────────────────────────


class TestAutoSelectAveraging:
    """Test the auto_select_averaging function."""

    def test_raw_wins(self) -> None:
        """Should select raw when it has lowest loss."""
        result = auto_select_averaging(
            raw_val_loss=0.1, ema_val_loss=0.5, swa_val_loss=0.3,
        )
        assert result == "raw"

    def test_ema_wins(self) -> None:
        """Should select ema when it has lowest loss."""
        result = auto_select_averaging(
            raw_val_loss=0.5, ema_val_loss=0.1, swa_val_loss=0.3,
        )
        assert result == "ema"

    def test_swa_wins(self) -> None:
        """Should select swa when it has lowest loss."""
        result = auto_select_averaging(
            raw_val_loss=0.5, ema_val_loss=0.3, swa_val_loss=0.1,
        )
        assert result == "swa"

    def test_ema_none_ignored(self) -> None:
        """Should handle None ema_val_loss gracefully."""
        result = auto_select_averaging(
            raw_val_loss=0.5, ema_val_loss=None, swa_val_loss=0.1,
        )
        assert result == "swa"

    def test_swa_none_ignored(self) -> None:
        """Should handle None swa_val_loss gracefully."""
        result = auto_select_averaging(
            raw_val_loss=0.5, ema_val_loss=0.1, swa_val_loss=None,
        )
        assert result == "ema"

    def test_both_none_returns_raw(self) -> None:
        """With both None, should return raw."""
        result = auto_select_averaging(
            raw_val_loss=0.5, ema_val_loss=None, swa_val_loss=None,
        )
        assert result == "raw"

    def test_equal_losses_stable(self) -> None:
        """Equal losses should not crash (any valid method is fine)."""
        result = auto_select_averaging(
            raw_val_loss=0.3, ema_val_loss=0.3, swa_val_loss=0.3,
        )
        assert result in ("raw", "ema", "swa")
