"""Event-level patient history representation for longitudinal neural models."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from ._config import config_value, timeline_days

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"


@dataclass(frozen=True)
class EventSequenceDataset:
    """Padded, chronological event arrays plus their auditable long-form events.

    Array dimensions are ``(n_patients, max_sequence_length)``. Positions where
    ``attention_mask`` is false are right padding. ``events`` contains only the
    retained (possibly recent-truncated) pre-index events.
    """

    patient_ids: np.ndarray
    index_dates: np.ndarray
    labels: np.ndarray
    event_type_ids: np.ndarray
    code_ids: np.ndarray
    therapy_class_ids: np.ndarray
    specialty_ids: np.ndarray
    time_delta_days: np.ndarray
    days_before_index: np.ndarray
    event_dates: np.ndarray
    attention_mask: np.ndarray
    lengths: np.ndarray
    events: pd.DataFrame
    vocabularies: dict[str, dict[str, int]]

    def __len__(self) -> int:
        return int(len(self.patient_ids))

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Return a framework-neutral single-patient training example."""

        return {
            "patient_id": self.patient_ids[index],
            "index_date": self.index_dates[index],
            "label": self.labels[index],
            "event_type_ids": self.event_type_ids[index],
            "code_ids": self.code_ids[index],
            "therapy_class_ids": self.therapy_class_ids[index],
            "specialty_ids": self.specialty_ids[index],
            "time_delta_days": self.time_delta_days[index],
            "days_before_index": self.days_before_index[index],
            "event_dates": self.event_dates[index],
            "attention_mask": self.attention_mask[index],
            "length": self.lengths[index],
        }

    def as_dict(self) -> dict[str, Any]:
        """Return all fields in a serialization-friendly mapping."""

        return {
            field: getattr(self, field)
            for field in (
                "patient_ids",
                "index_dates",
                "labels",
                "event_type_ids",
                "code_ids",
                "therapy_class_ids",
                "specialty_ids",
                "time_delta_days",
                "days_before_index",
                "event_dates",
                "attention_mask",
                "lengths",
                "events",
                "vocabularies",
            )
        }

    def to_sequence_split(self) -> Any:
        """Adapt this dataset to ``therapy_switch.models.SequenceSplit``.

        The import is intentionally lazy so data preparation never requires the
        optional deep-learning stack. Token identifiers remain separate columns
        in ``values``; elapsed days are supplied through ``times`` for the model's
        temporal encoder. Event and index dates are retained for a second leakage
        validation at model ingress.
        """

        from therapy_switch.models.contracts import SequenceSplit

        values = np.stack(
            [
                self.event_type_ids,
                self.code_ids,
                self.therapy_class_ids,
                self.specialty_ids,
                self.days_before_index,
            ],
            axis=-1,
        ).astype(np.float32)
        return SequenceSplit(
            values=values,
            mask=self.attention_mask,
            times=self.time_delta_days,
            event_dates=self.event_dates,
            index_dates=self.index_dates,
            pre_index_verified=True,
        )


def _validate_inputs(tables: Mapping[str, pd.DataFrame], cohort: pd.DataFrame) -> None:
    for table_name in ("medical_claims", "pharmacy_claims"):
        if table_name not in tables:
            raise ValueError(f"tables is missing {table_name!r}")
    if not {"patient_id", "index_date", "label"}.issubset(cohort.columns):
        raise ValueError("cohort must contain patient_id, index_date, and label")
    if cohort["patient_id"].duplicated().any():
        raise ValueError("cohort must have exactly one row per patient")


def build_event_frame(
    tables: Mapping[str, pd.DataFrame],
    cohort: pd.DataFrame,
    config: Mapping[str, Any] | Any | None = None,
) -> pd.DataFrame:
    """Return chronological, long-form events available on/before index.

    A medical row with both a diagnosis and procedure becomes two events. This
    avoids conflating code systems in a single token. Pharmacy fills become
    ``pharmacy`` events. Event times are always bounded by the configured
    observation window and index date.
    """

    config = {} if config is None else config
    _validate_inputs(tables, cohort)
    observation_days, _ = timeline_days(config)
    medical = tables["medical_claims"].copy()
    pharmacy = tables["pharmacy_claims"].copy()
    providers = tables.get("providers", pd.DataFrame()).copy()
    required_medical = {
        "patient_id",
        "claim_date",
        "diagnosis_code",
        "procedure_code",
        "provider_id",
        "place_of_service",
    }
    required_pharmacy = {
        "patient_id",
        "fill_date",
        "drug_id",
        "therapy_class",
        "prescriber_id",
    }
    if not required_medical.issubset(medical.columns):
        raise ValueError(
            f"medical_claims missing {sorted(required_medical - set(medical.columns))}"
        )
    if not required_pharmacy.issubset(pharmacy.columns):
        raise ValueError(
            f"pharmacy_claims missing {sorted(required_pharmacy - set(pharmacy.columns))}"
        )

    specialty_map: dict[Any, str] = {}
    if not providers.empty and {"provider_id", "specialty"}.issubset(providers.columns):
        specialty_map = (
            providers.drop_duplicates("provider_id")
            .set_index("provider_id")["specialty"]
            .astype("string")
            .to_dict()
        )

    landmarks = cohort[["patient_id", "index_date"]].copy()
    landmarks["index_date"] = pd.to_datetime(landmarks["index_date"], errors="raise")
    medical["claim_date"] = pd.to_datetime(medical["claim_date"], errors="raise")
    pharmacy["fill_date"] = pd.to_datetime(pharmacy["fill_date"], errors="raise")
    medical = medical.merge(landmarks, on="patient_id", how="inner", validate="many_to_one")
    pharmacy = pharmacy.merge(landmarks, on="patient_id", how="inner", validate="many_to_one")
    medical["days_before_index"] = (medical["index_date"] - medical["claim_date"]).dt.days
    pharmacy["days_before_index"] = (pharmacy["index_date"] - pharmacy["fill_date"]).dt.days
    medical = medical.loc[medical["days_before_index"].between(0, observation_days)]
    pharmacy = pharmacy.loc[pharmacy["days_before_index"].between(0, observation_days)]

    common_columns = [
        "patient_id",
        "index_date",
        "event_date",
        "event_type",
        "code",
        "therapy_class",
        "provider_specialty",
        "place_of_service",
        "days_before_index",
    ]
    frames: list[pd.DataFrame] = []
    diagnoses = medical.loc[medical["diagnosis_code"].notna()].copy()
    if not diagnoses.empty:
        diagnoses["event_date"] = diagnoses["claim_date"]
        diagnoses["event_type"] = "diagnosis"
        diagnoses["code"] = diagnoses["diagnosis_code"].astype("string")
        diagnoses["therapy_class"] = "none"
        diagnoses["provider_specialty"] = (
            diagnoses["provider_id"].map(specialty_map).fillna("unknown").astype("string")
        )
        frames.append(diagnoses[common_columns])
    procedures = medical.loc[medical["procedure_code"].notna()].copy()
    if not procedures.empty:
        procedures["event_date"] = procedures["claim_date"]
        procedures["event_type"] = "procedure"
        procedures["code"] = procedures["procedure_code"].astype("string")
        procedures["therapy_class"] = "none"
        procedures["provider_specialty"] = (
            procedures["provider_id"].map(specialty_map).fillna("unknown").astype("string")
        )
        frames.append(procedures[common_columns])
    if not pharmacy.empty:
        rx = pharmacy.copy()
        rx["event_date"] = rx["fill_date"]
        rx["event_type"] = "pharmacy"
        rx["code"] = rx["drug_id"].fillna("unknown").astype("string")
        rx["therapy_class"] = rx["therapy_class"].fillna("unknown").astype("string")
        rx["provider_specialty"] = (
            rx["prescriber_id"].map(specialty_map).fillna("unknown").astype("string")
        )
        rx["place_of_service"] = "pharmacy"
        frames.append(rx[common_columns])
    if not frames:
        return pd.DataFrame(columns=common_columns + ["time_since_previous_event", "event_order"])
    events = pd.concat(frames, ignore_index=True)
    events.sort_values(
        ["patient_id", "event_date", "event_type", "code"], kind="stable", inplace=True
    )
    events["time_since_previous_event"] = (
        events.groupby("patient_id")["event_date"].diff().dt.days.fillna(0).clip(lower=0)
    ).astype("float32")
    events["event_order"] = events.groupby("patient_id").cumcount().astype("int32")
    events.reset_index(drop=True, inplace=True)
    return events


def fit_sequence_vocabularies(events: pd.DataFrame) -> dict[str, dict[str, int]]:
    """Fit deterministic token maps, ideally using training-patient events only."""

    source_columns = {
        "event_type": "event_type",
        "code": "code",
        "therapy_class": "therapy_class",
        "specialty": "provider_specialty",
    }
    vocabularies: dict[str, dict[str, int]] = {}
    for vocabulary_name, column in source_columns.items():
        values = sorted(events[column].fillna("unknown").astype(str).unique().tolist())
        vocabulary = {PAD_TOKEN: 0, UNK_TOKEN: 1}
        vocabulary.update({value: offset + 2 for offset, value in enumerate(values)})
        vocabularies[vocabulary_name] = vocabulary
    return vocabularies


def _validate_vocabularies(vocabularies: Mapping[str, Mapping[str, int]]) -> None:
    for name in ("event_type", "code", "therapy_class", "specialty"):
        if name not in vocabularies:
            raise ValueError(f"Sequence vocabularies missing {name!r}")
        if vocabularies[name].get(PAD_TOKEN) != 0 or UNK_TOKEN not in vocabularies[name]:
            raise ValueError(f"Vocabulary {name!r} must define <PAD>=0 and <UNK>")


def build_event_sequences(
    tables: Mapping[str, pd.DataFrame],
    cohort: pd.DataFrame,
    config: Mapping[str, Any] | Any | None = None,
    *,
    vocabularies: Mapping[str, Mapping[str, int]] | None = None,
) -> EventSequenceDataset:
    """Create padded pre-index sequences for LSTM/GRU/Transformer models.

    ``features.sequence.max_length`` (or ``max_sequence_length``) controls recent
    truncation. For leakage-minimal evaluation, call :func:`build_event_frame` on
    training patients, fit vocabularies there, then pass those vocabularies for
    validation/test construction. Unseen tokens map to ``<UNK>``.
    """

    config = {} if config is None else config
    max_length = int(
        config_value(
            config,
            "max_sequence_length",
            "sequence.max_length",
            "features.sequence.max_length",
            "features.sequence.max_sequence_length",
            "features.sequence.max_events",
            default=256,
        )
    )
    if max_length <= 0:
        raise ValueError("max_sequence_length must be positive")
    events = build_event_frame(tables, cohort, config)
    configured_vocabularies = config_value(
        config, "sequence.vocabularies", "features.sequence.vocabularies", default=None
    )
    selected_vocabularies: Mapping[str, Mapping[str, int]]
    if vocabularies is not None:
        selected_vocabularies = vocabularies
    elif configured_vocabularies is not None:
        selected_vocabularies = configured_vocabularies
    else:
        selected_vocabularies = fit_sequence_vocabularies(events)
    _validate_vocabularies(selected_vocabularies)
    vocab = {name: dict(values) for name, values in selected_vocabularies.items()}

    n_patients = len(cohort)
    shapes = (n_patients, max_length)
    event_type_ids = np.zeros(shapes, dtype=np.int64)
    code_ids = np.zeros(shapes, dtype=np.int64)
    therapy_class_ids = np.zeros(shapes, dtype=np.int64)
    specialty_ids = np.zeros(shapes, dtype=np.int64)
    time_delta_days = np.zeros(shapes, dtype=np.float32)
    days_before_index = np.zeros(shapes, dtype=np.float32)
    event_dates = np.full(shapes, np.datetime64("NaT"), dtype="datetime64[ns]")
    attention_mask = np.zeros(shapes, dtype=bool)
    lengths = np.zeros(n_patients, dtype=np.int32)
    event_groups = {
        patient_id: group for patient_id, group in events.groupby("patient_id", sort=False)
    }
    retained_frames: list[pd.DataFrame] = []

    def encoded(values: pd.Series, vocabulary: Mapping[str, int]) -> np.ndarray:
        unknown = int(vocabulary[UNK_TOKEN])
        return (
            values.fillna("unknown")
            .astype(str)
            .map(vocabulary)
            .fillna(unknown)
            .to_numpy(dtype=np.int64)
        )

    cohort_ordered = cohort[["patient_id", "index_date", "label"]].reset_index(drop=True)
    for row_index, patient in enumerate(cohort_ordered.itertuples(index=False)):
        patient_events = event_groups.get(patient.patient_id)
        if patient_events is None:
            continue
        retained = patient_events.tail(max_length).copy().reset_index(drop=True)
        retained["event_order"] = np.arange(len(retained), dtype=np.int32)
        retained_frames.append(retained)
        length = len(retained)
        lengths[row_index] = length
        attention_mask[row_index, :length] = True
        event_type_ids[row_index, :length] = encoded(retained["event_type"], vocab["event_type"])
        code_ids[row_index, :length] = encoded(retained["code"], vocab["code"])
        therapy_class_ids[row_index, :length] = encoded(
            retained["therapy_class"], vocab["therapy_class"]
        )
        specialty_ids[row_index, :length] = encoded(
            retained["provider_specialty"], vocab["specialty"]
        )
        time_delta_days[row_index, :length] = retained["time_since_previous_event"].to_numpy(
            dtype=np.float32
        )
        days_before_index[row_index, :length] = retained["days_before_index"].to_numpy(
            dtype=np.float32
        )
        event_dates[row_index, :length] = pd.to_datetime(retained["event_date"]).to_numpy(
            dtype="datetime64[ns]"
        )

    retained_events = (
        pd.concat(retained_frames, ignore_index=True)
        if retained_frames
        else events.iloc[0:0].copy()
    )
    return EventSequenceDataset(
        patient_ids=cohort_ordered["patient_id"].to_numpy(),
        index_dates=pd.to_datetime(cohort_ordered["index_date"]).to_numpy(),
        labels=cohort_ordered["label"].to_numpy(dtype=np.int8),
        event_type_ids=event_type_ids,
        code_ids=code_ids,
        therapy_class_ids=therapy_class_ids,
        specialty_ids=specialty_ids,
        time_delta_days=time_delta_days,
        days_before_index=days_before_index,
        event_dates=event_dates,
        attention_mask=attention_mask,
        lengths=lengths,
        events=retained_events,
        vocabularies=vocab,
    )
