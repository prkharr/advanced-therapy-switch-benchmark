"""Train-only EHR pretraining and capacity-selected neural ranking models.

These compact CPU architectures are inspired by structured-EHR transformers and
reverse-time attention. They do not load or reproduce Med-BERT/BEHRT checkpoints.
"""

from __future__ import annotations

import copy
from collections import Counter

import numpy as np
import torch
from torch import nn

from therapy_switch.features.history import eligible_events
from therapy_switch.models.common import build_preprocessor
from therapy_switch.patient_lists import latest_patient_indices, patient_capture_metrics


class EventTokenizer:
    """Training-only event vocabulary; PAD=0, unknown=1, mask=2."""

    def __init__(self, max_length=96):
        self.max_length = max_length

    def fit(self, events, frame):
        data = eligible_events(events, frame)
        counts = Counter(data.history_token)
        self.vocabulary_ = {
            token: i + 3
            for i, (token, _) in enumerate(
                sorted(counts.items(), key=lambda pair: (-pair[1], pair[0]))
            )
        }
        return self

    def transform(self, events, frame):
        data = eligible_events(events, frame)
        groups = {key: part for key, part in data.groupby("snapshot_id", sort=False)}
        codes = np.zeros((len(frame), self.max_length), dtype=np.int64)
        times = np.zeros((len(frame), self.max_length, 2), dtype=np.float32)
        visits = np.zeros_like(codes)
        for i, snapshot in enumerate(frame.snapshot_id):
            part = groups.get(snapshot)
            if part is None:
                continue
            part = part.tail(self.max_length)
            size = len(part)
            codes[i, :size] = [self.vocabulary_.get(token, 1) for token in part.history_token]
            days = part.days_before_index.to_numpy(dtype=float)
            times[i, :size, 0] = np.log1p(days) / np.log(367)
            gaps = np.r_[0, np.maximum(0, days[:-1] - days[1:])]
            times[i, :size, 1] = np.log1p(gaps) / np.log(367)
            # Alternating visit segments, derived only from observed event dates.
            visit = (part.event_date != part.event_date.shift()).cumsum().to_numpy()
            visits[i, :size] = visit % 2
        return codes, times, visits


class PatientNetwork(nn.Module):
    def __init__(self, family, wide_size, vocab_size, length, width=64, dropout=0.2, hybrid=True):
        super().__init__()
        self.family = family
        self.hybrid = hybrid or family == "rank_mlp"
        self.wide = nn.Sequential(
            nn.Linear(wide_size, width * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(width * 2, width),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        if family != "rank_mlp":
            self.embedding = nn.Embedding(vocab_size, width, padding_idx=0)
            self.time = nn.Linear(2, width)
            self.segment = nn.Embedding(2, width)
            if family == "ehr_transformer":
                self.position = nn.Embedding(length + 1, width)
                self.cls = nn.Parameter(torch.zeros(1, 1, width))
                layer = nn.TransformerEncoderLayer(
                    width, 4, width * 2, dropout, batch_first=True, norm_first=True
                )
                self.encoder = nn.TransformerEncoder(layer, 2, enable_nested_tensor=False)
                self.masked_head = nn.Linear(width, vocab_size)
            else:
                self.alpha = nn.GRU(width, width, batch_first=True)
                self.beta = nn.GRU(width, width, batch_first=True)
                self.alpha_head = nn.Linear(width, 1)
                self.beta_head = nn.Linear(width, width)
        branches = 1 if family == "rank_mlp" or not self.hybrid else 2
        self.head = nn.Sequential(
            nn.Linear(width * branches, width), nn.ReLU(), nn.Dropout(dropout), nn.Linear(width, 1)
        )

    def encode(self, codes, times, visits):
        mask = codes != 0
        embedding = self.embedding(codes) + self.time(times) + self.segment(visits)
        if self.family == "ehr_transformer":
            cls = self.cls.expand(len(codes), -1, -1)
            embedding = torch.cat([cls, embedding], dim=1)
            position = torch.arange(embedding.shape[1], device=codes.device)
            embedding = embedding + self.position(position)[None, :, :]
            padding = torch.cat([torch.zeros((len(codes), 1), dtype=torch.bool), ~mask], dim=1)
            hidden = self.encoder(embedding, src_key_padding_mask=padding)
            return hidden[:, 0], hidden[:, 1:]
        # Reverse valid events, then pack; padding cannot affect recurrent states.
        lengths = mask.sum(1).clamp(min=1)
        reverse_index = (lengths[:, None] - 1 - torch.arange(codes.shape[1])[None]).clamp(min=0)
        reversed_events = embedding.gather(
            1, reverse_index[..., None].expand(-1, -1, embedding.shape[-1])
        )
        packed = nn.utils.rnn.pack_padded_sequence(
            reversed_events, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        a, _ = self.alpha(packed)
        b, _ = self.beta(packed)
        a, _ = nn.utils.rnn.pad_packed_sequence(a, batch_first=True, total_length=codes.shape[1])
        b, _ = nn.utils.rnn.pad_packed_sequence(b, batch_first=True, total_length=codes.shape[1])
        valid = torch.arange(codes.shape[1])[None] < lengths[:, None]
        alpha = torch.softmax(self.alpha_head(a).squeeze(-1).masked_fill(~valid, -1e9), dim=1)
        beta = torch.tanh(self.beta_head(b))
        context = (alpha[..., None] * beta * reversed_events).sum(1)
        context = context * mask.any(1)[:, None]
        return context, None

    def forward(self, wide, codes=None, times=None, visits=None):
        if self.family == "rank_mlp":
            return self.head(self.wide(wide)).squeeze(-1)
        context, _ = self.encode(codes, times, visits)
        if self.hybrid:
            context = torch.cat([context, self.wide(wide)], dim=1)
        return self.head(context).squeeze(-1)


class PatientNeuralEstimator:
    def __init__(self, family, options, random_state=42):
        self.family = family
        self.options = options
        self.random_state = random_state

    def _arrays(self, matrix, frame, events):
        wide = torch.as_tensor(
            np.nan_to_num(self.preprocessor_.transform(matrix)).astype(np.float32)
        )
        if self.family == "rank_mlp":
            return (wide,)
        codes, times, visits = self.tokenizer_.transform(events, frame)
        return wide, torch.from_numpy(codes), torch.from_numpy(times), torch.from_numpy(visits)

    def _predict_arrays(self, arrays):
        self.network_.eval()
        with torch.no_grad():
            pieces = [
                torch.sigmoid(self.network_(*[a[i : i + 256] for a in arrays])).numpy()
                for i in range(0, len(arrays[0]), 256)
            ]
        return np.concatenate(pieces).astype(float)

    def fit(self, xt, train, train_events, xv, validation, validation_events):
        torch.set_num_threads(2)
        torch.manual_seed(self.random_state)
        rng = np.random.default_rng(self.random_state)
        options = copy.deepcopy(self.options)
        width = int(options.get("width", 64))
        length = int(options.get("max_length", 96))
        self.preprocessor_ = build_preprocessor(xt, scale_numeric=True).fit(xt)
        self.tokenizer_ = None
        vocab_size = 3
        if self.family != "rank_mlp":
            self.tokenizer_ = EventTokenizer(length).fit(train_events, train)
            vocab_size += len(self.tokenizer_.vocabulary_)
        at, av = (
            self._arrays(xt, train, train_events),
            self._arrays(xv, validation, validation_events),
        )
        self.network_ = PatientNetwork(
            self.family,
            at[0].shape[1],
            vocab_size,
            length,
            width,
            float(options.get("dropout", 0.2)),
            bool(options.get("hybrid", True)),
        )
        self.history_ = []
        batch_size = int(options.get("batch_size", 128))
        pretrain_epochs = int(options.get("pretrain_epochs", 0))
        if pretrain_epochs and self.family != "ehr_transformer":
            raise ValueError("Masked pretraining requires the EHR transformer")
        optimizer = torch.optim.AdamW(
            self.network_.parameters(),
            lr=float(options.get("learning_rate", 0.001)),
            weight_decay=float(options.get("weight_decay", 0.001)),
        )
        # One history per training patient avoids overweighting duplicated events.
        pretrain_ids = latest_patient_indices(train)
        for epoch in range(pretrain_epochs):
            self.network_.train()
            losses = []
            for start in range(0, len(pretrain_ids), batch_size):
                if start == 0:
                    ordered = rng.permutation(pretrain_ids)
                ids = ordered[start : start + batch_size]
                codes, times, visits = [a[ids] for a in at[1:]]
                chosen = (torch.rand(codes.shape) < 0.15) & (codes >= 3)
                if not chosen.any():
                    continue
                corrupted = codes.clone()
                policy = torch.rand(codes.shape)
                corrupted[chosen & (policy < 0.8)] = 2
                random_ids = chosen & (policy >= 0.8) & (policy < 0.9)
                corrupted[random_ids] = torch.randint(3, vocab_size, (int(random_ids.sum()),))
                _, hidden = self.network_.encode(corrupted, times, visits)
                loss = nn.functional.cross_entropy(
                    self.network_.masked_head(hidden)[chosen], codes[chosen]
                )
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network_.parameters(), 1.0)
                optimizer.step()
                losses.append(float(loss.detach()))
            self.history_.append(
                {
                    "stage": "train_only_masked_event_pretraining",
                    "epoch": epoch + 1,
                    "loss": float(np.mean(losses)) if losses else None,
                }
            )
        optimizer = torch.optim.AdamW(
            self.network_.parameters(),
            lr=float(options.get("learning_rate", 0.001)),
            weight_decay=float(options.get("weight_decay", 0.001)),
        )
        y = torch.tensor(train.label.to_numpy(), dtype=torch.float32)
        ratio = float((len(y) - y.sum()) / y.sum())
        positive_weight = torch.tensor(ratio ** float(options.get("weight_power", 0.0)))
        rank_weight = float(options.get("rank_weight", 0.0))
        best_key, state, stale = (-1, -1), None, 0
        self.best_epoch_ = 0
        for epoch in range(int(options.get("max_epochs", 60))):
            self.network_.train()
            order = rng.permutation(len(y))
            losses = []
            for start in range(0, len(y), batch_size):
                ids = order[start : start + batch_size]
                logits = self.network_(*[a[ids] for a in at])
                target = y[ids]
                loss = nn.functional.binary_cross_entropy_with_logits(
                    logits, target, pos_weight=positive_weight
                )
                if rank_weight and target.min() != target.max():
                    positives, negatives = logits[target == 1], logits[target == 0]
                    # Hard negatives concentrate learning near the highest-risk list.
                    negatives = torch.topk(negatives, max(1, int(len(negatives) * 0.25))).values
                    pair_loss = nn.functional.softplus(
                        negatives[None, :] - positives[:, None]
                    ).mean()
                    loss = loss + rank_weight * pair_loss
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network_.parameters(), 1.0)
                optimizer.step()
                losses.append(float(loss.detach()))
            probability = self._predict_arrays(av)
            metrics = patient_capture_metrics(validation, probability)
            key = metrics["recall"], metrics["ap"]
            self.history_.append(
                {
                    "stage": "supervised",
                    "epoch": epoch + 1,
                    "loss": float(np.mean(losses)),
                    "validation_recall": key[0],
                    "validation_ap": key[1],
                }
            )
            if key > best_key:
                best_key = key
                self.best_epoch_ = epoch + 1
                state = copy.deepcopy(self.network_.state_dict())
                stale = 0
            else:
                stale += 1
            if stale >= int(options.get("patience", 12)):
                break
        if state is None:
            raise ValueError("No neural training epochs completed")
        self.network_.load_state_dict(state)
        self.network_.eval()
        return self

    def predict_scores(self, matrix, frame, events):
        return self._predict_arrays(self._arrays(matrix, frame, events))
