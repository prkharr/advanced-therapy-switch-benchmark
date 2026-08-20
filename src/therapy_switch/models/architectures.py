"""Compact PyTorch architectures used by the neural model runners."""

from __future__ import annotations

from typing import Sequence

try:  # Optional dependency: importing the package must remain safe without it.
    import torch
    from torch import nn
    from torch.nn import functional as functional

    TORCH_AVAILABLE = True
    TORCH_IMPORT_ERROR: Exception | None = None
except (ImportError, OSError) as exc:  # pragma: no cover - minimal/broken installs
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    functional = None  # type: ignore[assignment]
    TORCH_AVAILABLE = False
    TORCH_IMPORT_ERROR = exc


if TORCH_AVAILABLE:

    class FocalLoss(nn.Module):
        """Binary focal loss with optional positive-class weighting."""

        def __init__(
            self,
            gamma: float = 2.0,
            alpha: float | None = None,
            pos_weight: "torch.Tensor | None" = None,
        ) -> None:
            super().__init__()
            if gamma < 0:
                raise ValueError("focal gamma must be non-negative")
            if alpha is not None and not 0 <= alpha <= 1:
                raise ValueError("focal alpha must be between zero and one")
            self.gamma = gamma
            self.alpha = alpha
            self.register_buffer("pos_weight", pos_weight)

        def forward(self, logits: "torch.Tensor", target: "torch.Tensor") -> "torch.Tensor":
            base = functional.binary_cross_entropy_with_logits(
                logits,
                target,
                reduction="none",
                pos_weight=self.pos_weight,
            )
            probability = torch.sigmoid(logits)
            p_t = probability * target + (1.0 - probability) * (1.0 - target)
            modulation = (1.0 - p_t).pow(self.gamma)
            if self.alpha is not None:
                alpha_t = self.alpha * target + (1.0 - self.alpha) * (1.0 - target)
                modulation = modulation * alpha_t
            return (modulation * base).mean()

    class TabularMLP(nn.Module):
        def __init__(
            self,
            input_dim: int,
            hidden_dims: Sequence[int] = (128, 64),
            dropout: float = 0.25,
        ) -> None:
            super().__init__()
            if not hidden_dims or any(width < 1 for width in hidden_dims):
                raise ValueError("hidden_dims must contain positive layer widths")
            layers: list[nn.Module] = []
            previous = input_dim
            for width in hidden_dims:
                layers.extend(
                    [
                        nn.Linear(previous, int(width)),
                        nn.BatchNorm1d(int(width)),
                        nn.ReLU(),
                        nn.Dropout(dropout),
                    ]
                )
                previous = int(width)
            layers.append(nn.Linear(previous, 1))
            self.network = nn.Sequential(*layers)

        def forward(self, values: "torch.Tensor") -> "torch.Tensor":
            return self.network(values).squeeze(-1)

    class RecurrentSequenceClassifier(nn.Module):
        def __init__(
            self,
            input_dim: int,
            cell: str = "lstm",
            hidden_size: int = 48,
            num_layers: int = 1,
            dropout: float = 0.2,
            bidirectional: bool = False,
        ) -> None:
            super().__init__()
            if cell not in {"lstm", "gru"}:
                raise ValueError("cell must be 'lstm' or 'gru'")
            recurrent_class = nn.LSTM if cell == "lstm" else nn.GRU
            # Elapsed time is concatenated as an explicit temporal feature.
            self.recurrent = recurrent_class(
                input_size=input_dim + 1,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
                bidirectional=bidirectional,
            )
            output_size = hidden_size * (2 if bidirectional else 1)
            self.head = nn.Sequential(
                nn.LayerNorm(output_size),
                nn.Dropout(dropout),
                nn.Linear(output_size, 1),
            )
            self.bidirectional = bidirectional
            self.num_layers = num_layers

        def forward(
            self,
            values: "torch.Tensor",
            mask: "torch.Tensor",
            times: "torch.Tensor",
        ) -> "torch.Tensor":
            elapsed = torch.log1p(torch.clamp(times, min=0.0)).unsqueeze(-1)
            sequence = torch.cat([values, elapsed], dim=-1)
            lengths = mask.sum(dim=1).to(dtype=torch.long).cpu()
            packed = nn.utils.rnn.pack_padded_sequence(
                sequence,
                lengths,
                batch_first=True,
                enforce_sorted=False,
            )
            if isinstance(self.recurrent, nn.LSTM):
                _, (hidden, _) = self.recurrent(packed)
            else:
                _, hidden = self.recurrent(packed)
            if self.bidirectional:
                representation = torch.cat([hidden[-2], hidden[-1]], dim=-1)
            else:
                representation = hidden[-1]
            return self.head(representation).squeeze(-1)

    class TemporalTransformerClassifier(nn.Module):
        def __init__(
            self,
            input_dim: int,
            max_length: int,
            d_model: int = 48,
            nhead: int = 4,
            num_layers: int = 2,
            dim_feedforward: int = 96,
            dropout: float = 0.2,
        ) -> None:
            super().__init__()
            if d_model % nhead != 0:
                raise ValueError("d_model must be divisible by nhead")
            self.event_embedding = nn.Linear(input_dim, d_model)
            self.position_embedding = nn.Embedding(max_length, d_model)
            self.time_embedding = nn.Sequential(
                nn.Linear(1, d_model), nn.Tanh(), nn.Linear(d_model, d_model)
            )
            layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
            self.head = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Dropout(dropout),
                nn.Linear(d_model, 1),
            )

        def forward(
            self,
            values: "torch.Tensor",
            mask: "torch.Tensor",
            times: "torch.Tensor",
        ) -> "torch.Tensor":
            batch, length, _ = values.shape
            positions = torch.arange(length, device=values.device).unsqueeze(0)
            positions = positions.expand(batch, length)
            elapsed = torch.log1p(torch.clamp(times, min=0.0)).unsqueeze(-1)
            embedded = (
                self.event_embedding(values)
                + self.position_embedding(positions)
                + self.time_embedding(elapsed)
            )
            encoded = self.encoder(embedded, src_key_padding_mask=~mask)
            weights = mask.unsqueeze(-1).to(encoded.dtype)
            pooled = (encoded * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
            return self.head(pooled).squeeze(-1)


else:

    class _TorchRequired:
        def __init__(self, *_: object, **__: object) -> None:
            raise ImportError(
                "optional dependency 'torch' is unavailable; install the neural extra"
            ) from TORCH_IMPORT_ERROR

    FocalLoss = _TorchRequired
    TabularMLP = _TorchRequired
    RecurrentSequenceClassifier = _TorchRequired
    TemporalTransformerClassifier = _TorchRequired


__all__ = [
    "FocalLoss",
    "RecurrentSequenceClassifier",
    "TORCH_AVAILABLE",
    "TORCH_IMPORT_ERROR",
    "TabularMLP",
    "TemporalTransformerClassifier",
]
