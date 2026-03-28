"""
Weight Averaging for Wake Word Models
=======================================

Implements Stochastic Weight Averaging (SWA) and Exponential Moving Average
(EMA) for improving wake word model generalization.

SWA (Izmailov et al. 2018): Averages model weights collected at the end of
each epoch during a flat or cyclical LR phase. Finds wider optima that
generalize better.

EMA: Maintains a running exponential average of model weights during training.
The decay parameter controls how much history is retained (higher = more
smoothing).

Usage::

    from violawake_sdk.training.weight_averaging import EMATracker, SWACollector

    # EMA during training
    ema = EMATracker(model, decay=0.999)
    for batch in train_loader:
        loss = train_step(batch)
        ema.update()
    ema.apply()  # Replace model weights with EMA weights

    # SWA after main training
    swa = SWACollector()
    for epoch in swa_epochs:
        train_one_epoch()
        swa.collect(model)
    swa.apply(model)  # Replace model weights with SWA average
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

_TORCH_AVAILABLE = False
try:
    import torch
    import torch.nn as nn

    _TORCH_AVAILABLE = True
except ImportError:
    pass


if _TORCH_AVAILABLE:

    class EMATracker:
        """Exponential Moving Average of model parameters.

        Maintains a shadow copy of model parameters that is updated each
        step as: ema_param = decay * ema_param + (1 - decay) * param.

        Higher decay values produce smoother averages with more history.
        Typical values: 0.999 (slow, stable), 0.99 (faster adaptation).

        Args:
            model: The PyTorch model to track.
            decay: EMA decay factor. Default 0.999.

        Example::

            model = nn.Sequential(nn.Linear(96, 64), nn.ReLU(), nn.Linear(64, 1))
            ema = EMATracker(model, decay=0.999)

            for epoch in range(50):
                for batch_x, batch_y in train_loader:
                    optimizer.zero_grad()
                    loss = criterion(model(batch_x), batch_y)
                    loss.backward()
                    optimizer.step()
                    ema.update()

            # Apply EMA weights for inference
            ema.apply()
        """

        def __init__(self, model: nn.Module, decay: float = 0.999) -> None:
            self.model = model
            self.decay = decay
            self.shadow: dict[str, torch.Tensor] = {
                name: param.data.clone() for name, param in model.named_parameters()
            }
            self._backup: dict[str, torch.Tensor] | None = None
            self.num_updates: int = 0

        def update(self) -> None:
            """Update EMA parameters with current model parameters."""
            self.num_updates += 1
            # Use a warmup-adjusted decay: lower decay early on prevents
            # the initial random weights from dominating.
            decay = min(self.decay, (1.0 + self.num_updates) / (10.0 + self.num_updates))
            for name, param in self.model.named_parameters():
                self.shadow[name].mul_(decay).add_(param.data, alpha=1.0 - decay)

        def apply(self) -> None:
            """Replace model parameters with EMA parameters.

            Saves the original parameters so they can be restored with
            ``restore()``.
            """
            self._backup = {
                name: param.data.clone() for name, param in self.model.named_parameters()
            }
            for name, param in self.model.named_parameters():
                param.data.copy_(self.shadow[name])

        def restore(self) -> None:
            """Restore original (non-EMA) model parameters.

            Only valid after ``apply()`` has been called.

            Raises:
                RuntimeError: If ``apply()`` was not called before ``restore()``.
            """
            if self._backup is None:
                raise RuntimeError("Cannot restore: apply() was not called first")
            for name, param in self.model.named_parameters():
                param.data.copy_(self._backup[name])
            self._backup = None

        def state_dict(self) -> dict[str, torch.Tensor]:
            """Return the EMA shadow parameters as a dict."""
            return {k: v.clone() for k, v in self.shadow.items()}

        def load_state_dict(self, state_dict: dict[str, torch.Tensor]) -> None:
            """Load EMA parameters from a state dict."""
            for k, v in state_dict.items():
                if k in self.shadow:
                    self.shadow[k].copy_(v)

        @property
        def val_loss(self) -> float | None:
            """Validation loss tracked externally. Used for auto-selection."""
            return getattr(self, "_val_loss", None)

        @val_loss.setter
        def val_loss(self, value: float) -> None:
            self._val_loss = value

    class SWACollector:
        """Stochastic Weight Averaging collector.

        Collects model snapshots at the end of each epoch during the SWA
        phase and maintains their running average. After collection,
        call ``apply()`` to replace the model weights with the averaged
        weights.

        SWA typically uses a high constant or cyclical learning rate to
        explore the loss surface, then averages the traversed points.
        This finds wider optima that generalize better than SGD alone.

        Args:
            n_epochs: Expected number of SWA epochs. Used only for
                informational purposes.

        Example::

            swa = SWACollector()

            # SWA phase: constant LR, collect at end of each epoch
            for epoch in range(swa_start, swa_start + swa_epochs):
                train_one_epoch(model, optimizer)
                swa.collect(model)

            swa.apply(model)
            # Now model has averaged weights
        """

        def __init__(self, n_epochs: int = 0) -> None:
            self.n_collected: int = 0
            self._avg_params: dict[str, torch.Tensor] | None = None
            self.n_epochs = n_epochs
            self._val_losses: list[float] = []

        def collect(self, model: nn.Module, val_loss: float | None = None) -> None:
            """Collect current model parameters into the running average.

            Args:
                model: The model whose parameters to collect.
                val_loss: Optional validation loss at this collection point.
            """
            if self._avg_params is None:
                self._avg_params = {
                    name: param.data.clone() for name, param in model.named_parameters()
                }
                self.n_collected = 1
            else:
                self.n_collected += 1
                for name, param in model.named_parameters():
                    # Running mean: avg = avg + (new - avg) / n
                    self._avg_params[name].add_(
                        (param.data - self._avg_params[name]) / self.n_collected
                    )

            if val_loss is not None:
                self._val_losses.append(val_loss)

        def apply(self, model: nn.Module) -> None:
            """Replace model parameters with SWA-averaged parameters.

            Raises:
                RuntimeError: If no parameters have been collected yet.
            """
            if self._avg_params is None or self.n_collected == 0:
                raise RuntimeError("No parameters collected. Call collect() first.")
            for name, param in model.named_parameters():
                if name in self._avg_params:
                    param.data.copy_(self._avg_params[name])

        def state_dict(self) -> dict[str, torch.Tensor]:
            """Return the SWA averaged parameters as a dict."""
            if self._avg_params is None:
                return {}
            return {k: v.clone() for k, v in self._avg_params.items()}

        @property
        def val_loss(self) -> float | None:
            """Mean validation loss across all collected snapshots."""
            if not self._val_losses:
                return None
            return sum(self._val_losses) / len(self._val_losses)

    def auto_select_averaging(
        raw_val_loss: float,
        ema_val_loss: float | None,
        swa_val_loss: float | None,
    ) -> str:
        """Auto-select the best weight averaging method based on validation loss.

        Compares the validation loss of the raw model, EMA model, and SWA
        model, returning the name of the method with the lowest loss.

        Args:
            raw_val_loss: Validation loss of the raw (non-averaged) model.
            ema_val_loss: Validation loss of the EMA model, or None if
                EMA was not used.
            swa_val_loss: Validation loss of the SWA model, or None if
                SWA was not used.

        Returns:
            One of "raw", "ema", or "swa".
        """
        candidates = [("raw", raw_val_loss)]
        if ema_val_loss is not None:
            candidates.append(("ema", ema_val_loss))
        if swa_val_loss is not None:
            candidates.append(("swa", swa_val_loss))

        candidates.sort(key=lambda x: x[1])
        return candidates[0][0]

else:

    class EMATracker:  # type: ignore[no-redef]
        """Stub when PyTorch is not available."""

        def __init__(self, *args, **kwargs):
            raise ImportError(
                "PyTorch required for EMATracker. Install with: pip install 'violawake[training]'"
            )

    class SWACollector:  # type: ignore[no-redef]
        """Stub when PyTorch is not available."""

        def __init__(self, *args, **kwargs):
            raise ImportError(
                "PyTorch required for SWACollector. Install with: pip install 'violawake[training]'"
            )

    def auto_select_averaging(*args, **kwargs) -> str:
        """Stub when PyTorch is not available."""
        return "raw"
