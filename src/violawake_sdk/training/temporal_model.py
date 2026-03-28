"""
Multi-frame temporal wake word model (J5).

Instead of mean-pooling N frames of 96-dim OWW embeddings into a single vector,
this module provides models that operate on the full temporal sequence, preserving
frame ordering and temporal dynamics of the wake word utterance.

Architecture choices (all targeting <500K params, <5ms inference):

1. **TemporalCNN**: 1D convolution over the frame axis. Two conv layers with
   kernel sizes tuned for the 3-10 frame typical wake word duration, followed
   by adaptive pooling and a small MLP head. ~15K params.

2. **TemporalGRU**: Bidirectional GRU over the embedding sequence, taking the
   final hidden state through an MLP head. Captures directional temporal patterns.
   ~30K params.

3. **TemporalConvGRU**: Hybrid -- 1D conv feature extraction followed by a GRU
   for sequence modeling, then MLP head. Best of both worlds. ~40K params.

All models:
  - Input: (batch, seq_len, 96) -- variable-length sequences of OWW embeddings
  - Output: (batch, 1) -- wake word probability (after sigmoid)
  - Support ONNX export with fixed sequence length
  - Support variable sequence lengths via padding at inference time

Usage::

    from violawake_sdk.training.temporal_model import TemporalCNN, TemporalGRU, TemporalConvGRU

    model = TemporalCNN(embedding_dim=96, seq_len=9)
    score = model(embeddings)  # (batch, 9, 96) -> (batch, 1)
"""

from __future__ import annotations

_TORCH_AVAILABLE = False
try:
    import torch
    import torch.nn as nn

    _TORCH_AVAILABLE = True
except ImportError:
    pass


if _TORCH_AVAILABLE:

    class TemporalCNN(nn.Module):
        """1D CNN over OWW embedding frames.

        Applies 1D convolutions along the time axis to capture local temporal
        patterns (e.g., the phoneme sequence "vee-oh-lah"), followed by
        adaptive max pooling and a small MLP classifier.

        Architecture:
            Conv1d(96, 64, k=3) -> BN -> ReLU -> Dropout
            Conv1d(64, 32, k=3) -> BN -> ReLU -> AdaptiveMaxPool(1)
            Linear(32, 16) -> ReLU -> Dropout -> Linear(16, 1) -> Sigmoid

        Total params: ~12K (well under 500K target).

        Args:
            embedding_dim: Dimension of each OWW embedding frame (default 96).
            seq_len: Expected sequence length (used for ONNX export; runtime
                accepts any length >= 3).
            dropout: Dropout rate (default 0.3).
        """

        def __init__(
            self,
            embedding_dim: int = 96,
            seq_len: int = 9,
            dropout: float = 0.3,
        ) -> None:
            super().__init__()
            self.embedding_dim = embedding_dim
            self.seq_len = seq_len

            # Conv layers operate on (batch, channels=embedding_dim, time=seq_len)
            self.conv1 = nn.Conv1d(embedding_dim, 64, kernel_size=3, padding=1)
            self.bn1 = nn.BatchNorm1d(64)
            self.conv2 = nn.Conv1d(64, 32, kernel_size=3, padding=1)
            self.bn2 = nn.BatchNorm1d(32)
            self.pool = nn.AdaptiveMaxPool1d(1)
            self.drop1 = nn.Dropout(dropout)

            self.head = nn.Sequential(
                nn.Linear(32, 16),
                nn.ReLU(),
                nn.Dropout(dropout * 0.7),
                nn.Linear(16, 1),
                nn.Sigmoid(),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Forward pass.

            Args:
                x: (batch, seq_len, embedding_dim) tensor of OWW embeddings.

            Returns:
                (batch, 1) wake word probability.
            """
            # (batch, seq, emb) -> (batch, emb, seq) for Conv1d
            x = x.transpose(1, 2)
            x = self.drop1(torch.relu(self.bn1(self.conv1(x))))
            x = torch.relu(self.bn2(self.conv2(x)))
            x = self.pool(x).squeeze(-1)  # (batch, 32)
            return self.head(x)

    class TemporalGRU(nn.Module):
        """Bidirectional GRU over OWW embedding frames.

        Uses a small bidirectional GRU to capture temporal ordering of
        the embedding sequence, then classifies using the concatenated
        final hidden states from both directions.

        Architecture:
            BiGRU(96, hidden=32, layers=1) -> Dropout
            Linear(64, 16) -> ReLU -> Dropout -> Linear(16, 1) -> Sigmoid

        Total params: ~28K.

        Args:
            embedding_dim: Dimension of each OWW embedding frame (default 96).
            hidden_dim: GRU hidden dimension per direction (default 32).
            dropout: Dropout rate (default 0.3).
        """

        def __init__(
            self,
            embedding_dim: int = 96,
            hidden_dim: int = 32,
            dropout: float = 0.3,
        ) -> None:
            super().__init__()
            self.embedding_dim = embedding_dim
            self.hidden_dim = hidden_dim

            self.gru = nn.GRU(
                input_size=embedding_dim,
                hidden_size=hidden_dim,
                num_layers=1,
                batch_first=True,
                bidirectional=True,
            )
            self.drop = nn.Dropout(dropout)
            self.head = nn.Sequential(
                nn.Linear(hidden_dim * 2, 16),
                nn.ReLU(),
                nn.Dropout(dropout * 0.7),
                nn.Linear(16, 1),
                nn.Sigmoid(),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Forward pass.

            Args:
                x: (batch, seq_len, embedding_dim) tensor of OWW embeddings.

            Returns:
                (batch, 1) wake word probability.
            """
            # GRU output: (batch, seq_len, hidden*2), hidden: (2, batch, hidden)
            _, hidden = self.gru(x)
            # Concatenate forward and backward final hidden states
            combined = torch.cat([hidden[0], hidden[1]], dim=-1)  # (batch, hidden*2)
            combined = self.drop(combined)
            return self.head(combined)

    class TemporalConvGRU(nn.Module):
        """Hybrid 1D CNN + GRU temporal model.

        Applies a 1D convolution layer for local feature extraction, then
        feeds the sequence through a GRU for temporal modeling. This captures
        both local spectral transitions (conv) and longer-range temporal
        dependencies (GRU).

        Architecture:
            Conv1d(96, 48, k=3) -> BN -> ReLU -> Dropout
            GRU(48, hidden=24, layers=1) -> Dropout
            Linear(24, 16) -> ReLU -> Dropout -> Linear(16, 1) -> Sigmoid

        Total params: ~18K.

        Args:
            embedding_dim: Dimension of each OWW embedding frame (default 96).
            conv_channels: Number of conv output channels (default 48).
            gru_hidden: GRU hidden dimension (default 24).
            dropout: Dropout rate (default 0.3).
        """

        def __init__(
            self,
            embedding_dim: int = 96,
            conv_channels: int = 48,
            gru_hidden: int = 24,
            dropout: float = 0.3,
        ) -> None:
            super().__init__()
            self.embedding_dim = embedding_dim

            self.conv = nn.Conv1d(embedding_dim, conv_channels, kernel_size=3, padding=1)
            self.bn = nn.BatchNorm1d(conv_channels)
            self.conv_drop = nn.Dropout(dropout)

            self.gru = nn.GRU(
                input_size=conv_channels,
                hidden_size=gru_hidden,
                num_layers=1,
                batch_first=True,
            )
            self.gru_drop = nn.Dropout(dropout)

            self.head = nn.Sequential(
                nn.Linear(gru_hidden, 16),
                nn.ReLU(),
                nn.Dropout(dropout * 0.7),
                nn.Linear(16, 1),
                nn.Sigmoid(),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Forward pass.

            Args:
                x: (batch, seq_len, embedding_dim) tensor of OWW embeddings.

            Returns:
                (batch, 1) wake word probability.
            """
            # Conv: (batch, seq, emb) -> (batch, emb, seq) -> conv -> (batch, ch, seq)
            conv_in = x.transpose(1, 2)
            conv_out = self.conv_drop(torch.relu(self.bn(self.conv(conv_in))))
            # Back to (batch, seq, ch) for GRU
            gru_in = conv_out.transpose(1, 2)
            _, hidden = self.gru(gru_in)
            hidden = self.gru_drop(hidden.squeeze(0))  # (batch, gru_hidden)
            return self.head(hidden)

    def count_parameters(model: nn.Module) -> int:
        """Count total trainable parameters in a model."""
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    def export_temporal_onnx(
        model: nn.Module,
        output_path: str,
        seq_len: int = 9,
        embedding_dim: int = 96,
        opset_version: int = 11,
    ) -> None:
        """Export a temporal model to ONNX format.

        Args:
            model: Trained temporal model (TemporalCNN, TemporalGRU, or TemporalConvGRU).
            output_path: Path to save the .onnx file.
            seq_len: Fixed sequence length for the exported model.
            embedding_dim: Embedding dimension (default 96).
            opset_version: ONNX opset version (default 11).
        """
        from pathlib import Path

        model = model.cpu()
        model.eval()
        dummy_input = torch.zeros(1, seq_len, embedding_dim)

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        torch.onnx.export(
            model,
            dummy_input,
            str(output_path),
            input_names=["embeddings"],
            output_names=["score"],
            dynamic_axes={
                "embeddings": {0: "batch"},
                "score": {0: "batch"},
            },
            opset_version=opset_version,
        )

else:
    # Stubs when PyTorch is not available
    class TemporalCNN:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch required. pip install 'violawake[training]'")

    class TemporalGRU:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch required. pip install 'violawake[training]'")

    class TemporalConvGRU:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch required. pip install 'violawake[training]'")

    def count_parameters(model) -> int:
        raise ImportError("PyTorch required. pip install 'violawake[training]'")

    def export_temporal_onnx(model, output_path, **kwargs) -> None:
        raise ImportError("PyTorch required. pip install 'violawake[training]'")
