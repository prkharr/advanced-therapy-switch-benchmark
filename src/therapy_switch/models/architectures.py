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

    class EventEmbedding(nn.Module):
        def __init__(self, input_dim, d_model, categorical_sizes=()):
            super().__init__()
            self.count = len(categorical_sizes)
            self.embeddings = nn.ModuleList(
                [nn.Embedding(size, 8, padding_idx=0) for size in categorical_sizes]
            )
            self.projection = nn.Linear(8 * self.count + input_dim - self.count + 1, d_model)

        def forward(self, values, times):
            pieces = [
                embedding(values[:, :, i].long()) for i, embedding in enumerate(self.embeddings)
            ]
            numerical = values[:, :, self.count :]
            if self.count:
                numerical = torch.log1p(numerical.clamp_min(0)) / 6
            pieces.extend([numerical, torch.log1p(times.clamp_min(0)).unsqueeze(-1) / 6])
            return self.projection(torch.cat(pieces, dim=-1))

    class RecurrentSequenceClassifier(nn.Module):
        def __init__(
            self,
            input_dim,
            cell="lstm",
            hidden_size=48,
            num_layers=1,
            dropout=0.2,
            bidirectional=False,
            categorical_sizes=(),
        ):
            super().__init__()
            self.event_embedding = EventEmbedding(input_dim, hidden_size, categorical_sizes)
            recurrent_class = nn.GRU if cell == "gru" else nn.LSTM
            self.recurrent = recurrent_class(
                hidden_size,
                hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0,
                bidirectional=bidirectional,
            )
            self.output_size = hidden_size * (2 if bidirectional else 1)
            self.bidirectional = bidirectional
            self.head = nn.Sequential(
                nn.LayerNorm(self.output_size), nn.Dropout(dropout), nn.Linear(self.output_size, 1)
            )

        def encode(self, values, mask, times):
            sequence = self.event_embedding(values, times)
            sequence = sequence.masked_fill(~mask.unsqueeze(-1), 0)
            lengths = mask.sum(1).clamp_min(1).long().cpu()
            packed = nn.utils.rnn.pack_padded_sequence(
                sequence, lengths, batch_first=True, enforce_sorted=False
            )
            if isinstance(self.recurrent, nn.LSTM):
                _, (hidden, _) = self.recurrent(packed)
            else:
                _, hidden = self.recurrent(packed)
            representation = (
                torch.cat([hidden[-2], hidden[-1]], dim=-1) if self.bidirectional else hidden[-1]
            )
            return representation * mask.any(1).unsqueeze(-1)

        def forward(self, values, mask, times):
            return self.head(self.encode(values, mask, times)).squeeze(-1)

    class TemporalTransformerClassifier(nn.Module):
        def __init__(
            self,
            input_dim,
            max_length,
            d_model=48,
            nhead=4,
            num_layers=2,
            dim_feedforward=96,
            dropout=0.2,
            categorical_sizes=(),
        ):
            super().__init__()
            if d_model % nhead:
                raise ValueError("d_model must be divisible by nhead")
            self.event_embedding = EventEmbedding(input_dim, d_model, categorical_sizes)
            self.position_embedding = nn.Embedding(max_length, d_model)
            layer = nn.TransformerEncoderLayer(
                d_model,
                nhead,
                dim_feedforward,
                dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(layer, num_layers, enable_nested_tensor=False)
            self.output_size = d_model
            self.head = nn.Sequential(
                nn.LayerNorm(d_model), nn.Dropout(dropout), nn.Linear(d_model, 1)
            )

        def encode(self, values, mask, times):
            positions = torch.arange(values.shape[1], device=values.device).unsqueeze(0)
            embedded = self.event_embedding(values, times) + self.position_embedding(positions)
            embedded = embedded.masked_fill(~mask.unsqueeze(-1), 0)
            safe_mask = mask.clone()
            safe_mask[~safe_mask.any(1), 0] = True
            encoded = self.encoder(embedded, src_key_padding_mask=~safe_mask)
            encoded = encoded.masked_fill(~mask.unsqueeze(-1), 0)
            return encoded.sum(1) / mask.sum(1).clamp_min(1).unsqueeze(-1)

        def forward(self, values, mask, times):
            return self.head(self.encode(values, mask, times)).squeeze(-1)

    class HybridSequenceClassifier(nn.Module):
        def __init__(
            self, input_dim, wide_dim, categorical_sizes=(), hidden_size=48, dropout=0.2, **kwargs
        ):
            super().__init__()
            self.input_dim = input_dim
            self.sequence = RecurrentSequenceClassifier(
                input_dim,
                cell="gru",
                hidden_size=hidden_size,
                dropout=dropout,
                categorical_sizes=categorical_sizes,
            )
            self.wide = nn.Sequential(
                nn.Linear(wide_dim, 32), nn.LayerNorm(32), nn.ReLU(), nn.Dropout(dropout)
            )
            self.head = nn.Sequential(
                nn.Linear(hidden_size + 32, 32), nn.ReLU(), nn.Dropout(dropout), nn.Linear(32, 1)
            )

        def forward(self, values, mask, times):
            sequence = self.sequence.encode(values[:, :, : self.input_dim], mask, times)
            wide = self.wide(values[:, 0, self.input_dim :])
            return self.head(torch.cat([sequence, wide], dim=-1)).squeeze(-1)


else:

    class _TorchRequired:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch is required for neural models")

    FocalLoss = TabularMLP = RecurrentSequenceClassifier = TemporalTransformerClassifier = (
        HybridSequenceClassifier
    ) = _TorchRequired
