"""Tabular and longitudinal deep-learning runners.

PyTorch is optional.  When absent these runners return an explicit
``NOT APPLICABLE`` result rather than disappearing from a benchmark.
"""

from __future__ import annotations

import copy
import json
import re
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from .architectures import (
    TORCH_AVAILABLE,
    FocalLoss,
    RecurrentSequenceClassifier,
    TabularMLP,
    TemporalTransformerClassifier,
)
from .classical import BaseModelRunner
from .common import (
    DependencyUnavailable,
    average_precision,
    build_preprocessor,
    ensure_two_training_classes,
    json_safe,
    positive_class_weight,
    require_module,
    select_threshold,
)
from .contracts import SUCCESS, ModelResult, ModelRun, SequenceSplit


def _require_torch() -> Any:
    if not TORCH_AVAILABLE:
        raise DependencyUnavailable(
            "optional dependency 'torch' is unavailable; install the neural extra"
        )
    return require_module("torch")


def _set_seed(seed: int) -> None:
    torch = _require_torch()
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _device_from(options: Mapping[str, Any]) -> Any:
    torch = _require_torch()
    requested = options.get("device")
    if requested is None:
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if str(requested).startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("a CUDA device was requested but CUDA is unavailable")
    return torch.device(str(requested))


def _loss_function(
    loss_name: str,
    class_ratio: float,
    device: Any,
    options: Mapping[str, Any],
) -> Any:
    torch = _require_torch()
    pos_weight = torch.tensor([class_ratio], dtype=torch.float32, device=device)
    if loss_name == "bce":
        return torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    if loss_name == "focal":
        alpha = options.get("focal_alpha")
        return FocalLoss(
            gamma=float(options.get("focal_gamma", 2.0)),
            alpha=None if alpha is None else float(alpha),
            pos_weight=pos_weight,
        )
    raise ValueError("loss must be 'bce' or 'focal'")


def _forward(model: Any, batch: Sequence[Any], sequence: bool) -> Any:
    if sequence:
        values, mask, times, _ = batch
        return model(values, mask, times)
    values, _, _, _ = batch
    return model(values)


def _loader(
    values: np.ndarray,
    target: np.ndarray,
    batch_size: int,
    *,
    mask: Optional[np.ndarray] = None,
    times: Optional[np.ndarray] = None,
    shuffle: bool = False,
    seed: int = 42,
    protect_batch_norm: bool = False,
) -> Any:
    torch = _require_torch()
    values_tensor = torch.as_tensor(values, dtype=torch.float32)
    if mask is None:
        mask = np.ones((len(values), 1), dtype=bool)
    if times is None:
        times = np.zeros(np.asarray(mask).shape, dtype=np.float32)
    dataset = torch.utils.data.TensorDataset(
        values_tensor,
        torch.as_tensor(mask, dtype=torch.bool),
        torch.as_tensor(times, dtype=torch.float32),
        torch.as_tensor(target, dtype=torch.float32),
    )
    effective_batch = max(2 if protect_batch_norm else 1, min(batch_size, len(dataset)))
    drop_last = protect_batch_norm and len(dataset) % effective_batch == 1
    generator = torch.Generator().manual_seed(seed)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=effective_batch,
        shuffle=shuffle,
        drop_last=drop_last,
        generator=generator,
    )


def _evaluate_network(
    model: Any,
    loader: Any,
    criterion: Any,
    device: Any,
    sequence: bool,
) -> tuple[float, np.ndarray]:
    torch = _require_torch()
    model.eval()
    total_loss = 0.0
    observations = 0
    probabilities: list[np.ndarray] = []
    with torch.no_grad():
        for raw_batch in loader:
            batch = tuple(item.to(device) for item in raw_batch)
            target = batch[-1]
            logits = _forward(model, batch, sequence)
            loss = criterion(logits, target)
            total_loss += float(loss.item()) * len(target)
            observations += len(target)
            probabilities.append(torch.sigmoid(logits).cpu().numpy())
    if observations == 0:
        raise ValueError("evaluation loader contained no observations")
    return total_loss / observations, np.concatenate(probabilities).astype(float)


def _train_network(
    model: Any,
    train_arrays: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    val_arrays: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    *,
    sequence: bool,
    loss_name: str,
    class_ratio: float,
    options: Mapping[str, Any],
    random_state: int,
) -> tuple[Any, dict[str, Any], np.ndarray]:
    torch = _require_torch()
    _set_seed(random_state)
    device = _device_from(options)
    model = model.to(device)
    batch_size = int(options.get("batch_size", 64))
    max_epochs = int(options.get("max_epochs", 50))
    patience = int(options.get("patience", 7))
    if batch_size < 2 or max_epochs < 1 or patience < 1:
        raise ValueError("batch_size >= 2, max_epochs >= 1, and patience >= 1 are required")

    train_values, train_mask, train_times, y_train = train_arrays
    val_values, val_mask, val_times, y_val = val_arrays
    train_loader = _loader(
        train_values,
        y_train,
        batch_size,
        mask=train_mask,
        times=train_times,
        shuffle=True,
        seed=random_state,
        protect_batch_norm=not sequence,
    )
    val_loader = _loader(
        val_values,
        y_val,
        batch_size,
        mask=val_mask,
        times=val_times,
        shuffle=False,
        seed=random_state,
    )
    criterion = _loss_function(loss_name, class_ratio, device, options)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(options.get("learning_rate", 1e-3)),
        weight_decay=float(options.get("weight_decay", 1e-4)),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=float(options.get("lr_factor", 0.5)),
        patience=int(options.get("lr_patience", 2)),
    )

    history: dict[str, Any] = {
        "loss": loss_name,
        "train_loss": [],
        "val_loss": [],
        "val_pr_auc": [],
        "learning_rate": [],
    }
    best_state = copy.deepcopy(model.state_dict())
    best_loss = float("inf")
    stale_epochs = 0
    min_delta = float(options.get("min_delta", 1e-5))

    for _epoch in range(max_epochs):
        model.train()
        running_loss = 0.0
        observations = 0
        for raw_batch in train_loader:
            batch = tuple(item.to(device) for item in raw_batch)
            target = batch[-1]
            optimizer.zero_grad(set_to_none=True)
            logits = _forward(model, batch, sequence)
            loss = criterion(logits, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(options.get("gradient_clip", 5.0))
            )
            optimizer.step()
            running_loss += float(loss.item()) * len(target)
            observations += len(target)
        if observations == 0:
            raise ValueError("training loader contained no observations")

        train_loss = running_loss / observations
        val_loss, val_probability = _evaluate_network(
            model, val_loader, criterion, device, sequence
        )
        val_pr_auc = average_precision(y_val, val_probability)
        scheduler.step(val_loss)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_pr_auc"].append(val_pr_auc)
        history["learning_rate"].append(float(optimizer.param_groups[0]["lr"]))

        if val_loss < best_loss - min_delta:
            best_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break

    model.load_state_dict(best_state)
    final_loss, final_probability = _evaluate_network(
        model, val_loader, criterion, device, sequence
    )
    history["epochs_trained"] = len(history["train_loss"])
    history["best_val_loss"] = best_loss
    history["restored_val_loss"] = final_loss
    history["early_stopped"] = len(history["train_loss"]) < max_epochs
    history["device"] = str(device)
    return model, history, final_probability


def _save_history(run: ModelRun, model_name: str, history: Mapping[str, Any]) -> Optional[str]:
    directory = run.artifacts
    if directory is None:
        return None
    directory.mkdir(parents=True, exist_ok=True)
    slug = re.sub(r"[^a-z0-9]+", "_", model_name.lower()).strip("_")
    path = directory / f"{slug}_training_history.json"
    path.write_text(json.dumps(json_safe(history), indent=2), encoding="utf-8")
    return str(path)


class TorchTabularEstimator:
    """Small sklearn-like inference wrapper for a fitted PyTorch MLP."""

    def __init__(self, preprocessor: Any, network: Any, device: Any) -> None:
        self.preprocessor = preprocessor
        self.network = network
        self.device = device

    def predict_proba(self, frame: Any) -> np.ndarray:
        torch = _require_torch()
        values = np.asarray(self.preprocessor.transform(frame), dtype=np.float32)
        self.network.eval()
        outputs: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(values), 1024):
                tensor = torch.as_tensor(
                    values[start : start + 1024], dtype=torch.float32, device=self.device
                )
                outputs.append(torch.sigmoid(self.network(tensor)).cpu().numpy())
        positive = np.concatenate(outputs).astype(float)
        return np.column_stack([1.0 - positive, positive])


@dataclass
class SequenceScaler:
    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, split: SequenceSplit) -> "SequenceScaler":
        assert split.mask is not None
        valid = np.asarray(split.values, dtype=float)[split.mask]
        if not np.isfinite(valid).all():
            raise ValueError("valid sequence event features must be finite")
        mean = valid.mean(axis=0)
        scale = valid.std(axis=0)
        scale[scale < 1e-8] = 1.0
        return cls(mean=mean.astype(np.float32), scale=scale.astype(np.float32))

    def transform(self, split: SequenceSplit) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        split.validated(expected_rows=len(split.values))
        assert split.mask is not None
        values = (np.asarray(split.values, dtype=np.float32) - self.mean) / self.scale
        values[~split.mask] = 0.0
        times = (
            np.zeros(split.mask.shape, dtype=np.float32)
            if split.times is None
            else np.asarray(split.times, dtype=np.float32)
        )
        times[~split.mask] = 0.0
        return values, split.mask.astype(bool), times


class TorchSequenceEstimator:
    def __init__(
        self, network: Any, scaler: SequenceScaler, device: Any, batch_size: int = 1024
    ) -> None:
        self.network = network
        self.scaler = scaler
        self.device = device
        self.batch_size = batch_size

    def predict_proba(self, split: SequenceSplit) -> np.ndarray:
        torch = _require_torch()
        values, mask, times = self.scaler.transform(split)
        self.network.eval()
        probabilities: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(values), self.batch_size):
                stop = start + self.batch_size
                batch_values = torch.as_tensor(
                    values[start:stop], dtype=torch.float32, device=self.device
                )
                batch_mask = torch.as_tensor(mask[start:stop], dtype=torch.bool, device=self.device)
                batch_times = torch.as_tensor(
                    times[start:stop], dtype=torch.float32, device=self.device
                )
                logits = self.network(batch_values, batch_mask, batch_times)
                probabilities.append(torch.sigmoid(logits).cpu().numpy())
        positive = np.concatenate(probabilities).astype(float)
        return np.column_stack([1.0 - positive, positive])


def _configured_losses(options: Mapping[str, Any], default: Sequence[str]) -> tuple[str, ...]:
    configured = options.get("losses", options.get("loss", default))
    if isinstance(configured, str):
        losses = (configured.lower(),)
    else:
        losses = tuple(str(item).lower() for item in configured)
    if not losses or not set(losses).issubset({"bce", "focal"}):
        raise ValueError("losses must contain only 'bce' and/or 'focal'")
    return tuple(dict.fromkeys(losses))


class MLPRunner(BaseModelRunner):
    key = "mlp"
    model_name = "MLP"
    category = "Tabular DL"

    def run(self, run: ModelRun) -> ModelResult:
        try:
            _require_torch()
            run.validate_tabular()
            y_train, y_val, _ = run.targets()
            ensure_two_training_classes(y_train)
            if len(y_train) < 2:
                raise ValueError("MLP training requires at least two patients")
            options = run.options_for(self.key)
            class_ratio = positive_class_weight(y_train)
            started = perf_counter()
            preprocessor = build_preprocessor(run.X_train, scale_numeric=True)
            X_train = np.asarray(preprocessor.fit_transform(run.X_train), dtype=np.float32)
            X_val = np.asarray(preprocessor.transform(run.X_val), dtype=np.float32)
            empty_train_mask = np.ones((len(X_train), 1), dtype=bool)
            empty_val_mask = np.ones((len(X_val), 1), dtype=bool)
            train_arrays = (
                X_train,
                empty_train_mask,
                np.zeros_like(empty_train_mask, dtype=np.float32),
                y_train,
            )
            val_arrays = (
                X_val,
                empty_val_mask,
                np.zeros_like(empty_val_mask, dtype=np.float32),
                y_val,
            )

            hidden_dims = tuple(int(value) for value in options.get("hidden_dims", (128, 64)))
            dropout = float(options.get("dropout", 0.25))
            losses = _configured_losses(options, default=("bce", "focal"))
            candidates: dict[str, tuple[Any, dict[str, Any], np.ndarray, float]] = {}
            for loss_name in losses:
                _set_seed(run.random_state)
                network = TabularMLP(
                    input_dim=X_train.shape[1],
                    hidden_dims=hidden_dims,
                    dropout=dropout,
                )
                trained, history, val_probability = _train_network(
                    network,
                    train_arrays,
                    val_arrays,
                    sequence=False,
                    loss_name=loss_name,
                    class_ratio=class_ratio,
                    options=options,
                    random_state=run.random_state,
                )
                score = average_precision(y_val, val_probability)
                candidates[loss_name] = (trained, history, val_probability, score)
            selected_loss = max(
                candidates,
                key=lambda name: -np.inf if np.isnan(candidates[name][3]) else candidates[name][3],
            )
            network, selected_history, val_probability, val_score = candidates[selected_loss]
            training_time = perf_counter() - started
            device = _device_from(options)
            estimator = TorchTabularEstimator(preprocessor, network, device)
            threshold = select_threshold(
                y_val,
                val_probability,
                strategy=str(options.get("threshold_strategy", "f1")),
                fixed_threshold=float(options.get("fixed_threshold", 0.5)),
            )
            inference_started = perf_counter()
            test_probability = estimator.predict_proba(run.X_test)[:, 1]
            inference_time = perf_counter() - inference_started
            all_history = {
                "selected_loss": selected_loss,
                "runs": {name: values[1] for name, values in candidates.items()},
            }
            history_path = _save_history(run, self.model_name, all_history)
            return ModelResult(
                model=self.model_name,
                category=self.category,
                status=SUCCESS,
                estimator=estimator,
                validation_probabilities=val_probability,
                test_probabilities=test_probability,
                test_predictions=(test_probability >= threshold).astype(np.int8),
                threshold=threshold,
                validation_score=val_score,
                best_params={
                    "hidden_dims": hidden_dims,
                    "dropout": dropout,
                    "loss": selected_loss,
                },
                training_time_seconds=training_time,
                inference_time_seconds=inference_time,
                history=all_history,
                metadata={
                    "loss_comparison": {name: values[3] for name, values in candidates.items()},
                    "positive_class_weight": class_ratio,
                    "history_path": history_path,
                    "n_transformed_features": int(X_train.shape[1]),
                    "threshold_selected_on": "validation",
                },
            )
        except Exception as exc:
            return self._failed(exc)


class SequenceModelRunner(BaseModelRunner):
    category = "Longitudinal DL"
    requires_sequence = True
    architecture: str

    def _network(
        self,
        input_dim: int,
        max_length: int,
        options: Mapping[str, Any],
    ) -> Any:
        if self.architecture in {"lstm", "gru", "bilstm"}:
            return RecurrentSequenceClassifier(
                input_dim=input_dim,
                cell="gru" if self.architecture == "gru" else "lstm",
                hidden_size=int(options.get("hidden_size", 48)),
                num_layers=int(options.get("num_layers", 1)),
                dropout=float(options.get("dropout", 0.2)),
                bidirectional=self.architecture == "bilstm",
            )
        if self.architecture == "transformer":
            return TemporalTransformerClassifier(
                input_dim=input_dim,
                max_length=max_length,
                d_model=int(options.get("d_model", 48)),
                nhead=int(options.get("nhead", 4)),
                num_layers=int(options.get("num_layers", 2)),
                dim_feedforward=int(options.get("dim_feedforward", 96)),
                dropout=float(options.get("dropout", 0.2)),
            )
        raise ValueError(f"unsupported sequence architecture {self.architecture!r}")

    def run(self, run: ModelRun) -> ModelResult:
        if any(
            sequence is None
            for sequence in (run.sequence_train, run.sequence_val, run.sequence_test)
        ):
            return ModelResult.not_applicable(
                self.model_name,
                self.category,
                "event-level train/validation/test sequence inputs were not supplied",
                sequence_required=True,
            )
        try:
            run.validate_tabular()
            run.validate_sequences()
            # Validate temporal safety before dependency checks so a leakage
            # violation can never be disguised as a missing-PyTorch skip.
            _require_torch()
            y_train, y_val, _ = run.targets()
            ensure_two_training_classes(y_train)
            options = run.options_for(self.key)
            class_ratio = positive_class_weight(y_train)
            assert run.sequence_train is not None
            assert run.sequence_val is not None
            assert run.sequence_test is not None
            started = perf_counter()
            scaler = SequenceScaler.fit(run.sequence_train)
            train_values, train_mask, train_times = scaler.transform(run.sequence_train)
            val_values, val_mask, val_times = scaler.transform(run.sequence_val)
            max_length = run.sequence_train.values.shape[1]
            if (
                run.sequence_val.values.shape[1] > max_length
                or run.sequence_test.values.shape[1] > max_length
            ):
                raise ValueError(
                    "validation/test sequence length exceeds the training-defined "
                    "maximum; truncate upstream without inspecting outcomes"
                )
            losses = _configured_losses(options, default=("bce",))
            candidates: dict[str, tuple[Any, dict[str, Any], np.ndarray, float]] = {}
            for loss_name in losses:
                _set_seed(run.random_state)
                network = self._network(train_values.shape[2], max_length, options)
                trained, history, val_probability = _train_network(
                    network,
                    (train_values, train_mask, train_times, y_train),
                    (val_values, val_mask, val_times, y_val),
                    sequence=True,
                    loss_name=loss_name,
                    class_ratio=class_ratio,
                    options=options,
                    random_state=run.random_state,
                )
                score = average_precision(y_val, val_probability)
                candidates[loss_name] = (trained, history, val_probability, score)
            selected_loss = max(
                candidates,
                key=lambda name: -np.inf if np.isnan(candidates[name][3]) else candidates[name][3],
            )
            network, _selected_history, val_probability, val_score = candidates[selected_loss]
            training_time = perf_counter() - started
            estimator = TorchSequenceEstimator(
                network,
                scaler,
                _device_from(options),
                batch_size=int(options.get("inference_batch_size", 1024)),
            )
            threshold = select_threshold(
                y_val,
                val_probability,
                strategy=str(options.get("threshold_strategy", "f1")),
                fixed_threshold=float(options.get("fixed_threshold", 0.5)),
            )
            inference_started = perf_counter()
            test_probability = estimator.predict_proba(run.sequence_test)[:, 1]
            inference_time = perf_counter() - inference_started
            all_history = {
                "selected_loss": selected_loss,
                "runs": {name: values[1] for name, values in candidates.items()},
            }
            history_path = _save_history(run, self.model_name, all_history)
            architecture_params = {
                key: options[key]
                for key in (
                    "hidden_size",
                    "d_model",
                    "nhead",
                    "num_layers",
                    "dim_feedforward",
                    "dropout",
                )
                if key in options
            }
            return ModelResult(
                model=self.model_name,
                category=self.category,
                status=SUCCESS,
                estimator=estimator,
                validation_probabilities=val_probability,
                test_probabilities=test_probability,
                test_predictions=(test_probability >= threshold).astype(np.int8),
                threshold=threshold,
                validation_score=val_score,
                best_params={
                    "architecture": self.architecture,
                    "loss": selected_loss,
                    **architecture_params,
                },
                training_time_seconds=training_time,
                inference_time_seconds=inference_time,
                history=all_history,
                metadata={
                    "loss_comparison": {name: values[3] for name, values in candidates.items()},
                    "positive_class_weight": class_ratio,
                    "history_path": history_path,
                    "uses_only_pre_index_events": True,
                    "temporal_encoding": (
                        "elapsed-time feature"
                        if self.architecture != "transformer"
                        else "event projection + learned position + elapsed-time embedding"
                    ),
                    "threshold_selected_on": "validation",
                },
            )
        except Exception as exc:
            return self._failed(exc)


class LSTMRunner(SequenceModelRunner):
    key = "lstm"
    model_name = "LSTM"
    architecture = "lstm"


class GRURunner(SequenceModelRunner):
    key = "gru"
    model_name = "GRU"
    architecture = "gru"


class BiLSTMRunner(SequenceModelRunner):
    key = "bilstm"
    model_name = "BiLSTM"
    architecture = "bilstm"

    def run(self, run: ModelRun) -> ModelResult:
        result = super().run(run)
        if result.succeeded:
            result.metadata["bidirectional_safety"] = (
                "attention/recurrent context is bidirectional only within the verified "
                "pre-index history; no prediction-window events are present"
            )
        return result


class TransformerRunner(SequenceModelRunner):
    key = "transformer"
    model_name = "Transformer"
    architecture = "transformer"
