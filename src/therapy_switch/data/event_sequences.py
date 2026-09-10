"""Chronological snapshot event histories and training-only token encoding."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ._config import config_value

PAD_TOKEN, UNK_TOKEN = "<PAD>", "<UNK>"
VOCAB_COLUMNS = {
    "event_type": "event_type",
    "code": "code_token",
    "therapy_class": "therapy_class",
    "specialty": "provider_specialty",
}


def build_event_frame(tables, cohort, config=None):
    config = config or {}
    if "prepared_events" in tables:
        return (
            tables["prepared_events"]
            .loc[tables["prepared_events"].snapshot_id.isin(cohort.snapshot_id)]
            .copy()
        )
    providers = tables["providers"].set_index("provider_id").specialty
    landmarks = cohort[
        ["snapshot_id", "patient_id", "index_date", "feature_cutoff", "lookback_start"]
    ]
    frames = []
    for name, date_col, provider_col, code_cols in [
        (
            "medical_claims",
            "claim_date",
            "provider_id",
            [("diagnosis", "diagnosis_code"), ("procedure", "procedure_code")],
        ),
        ("pharmacy_claims", "fill_date", "prescriber_id", [("pharmacy", "drug_id")]),
    ]:
        source = tables[name].merge(
            landmarks, on="patient_id", how="inner", validate="many_to_many"
        )
        source = source.loc[
            source[date_col].le(source.feature_cutoff)
            & source[date_col].ge(source.lookback_start)
            & source.available_date.le(source.index_date)
        ].copy()
        if "reversal_date" in source:
            source.loc[source.reversal_date.le(source.index_date), "status"] = "reversed"
        for kind, code_col in code_cols:
            part = source.loc[source[code_col].notna()].copy()
            part["event_date"] = part[date_col]
            part["event_type"] = kind + "_" + part.status
            part["code_system"] = kind
            part["code"] = part[code_col].astype(str)
            part["event_id"] = part.claim_id.astype(str) + "__" + kind
            part["hcp_id"] = part[provider_col]
            part["provider_specialty"] = part[provider_col].map(providers).fillna("unknown")
            if name == "medical_claims":
                part["therapy_class"], part["product_id"] = "none", ""
            else:
                part["product_id"] = part.drug_id
                part["place_of_service"] = "pharmacy"
            frames.append(
                part[
                    [
                        "snapshot_id",
                        "patient_id",
                        "index_date",
                        "event_id",
                        "event_date",
                        "available_date",
                        "event_type",
                        "code_system",
                        "code",
                        "hcp_id",
                        "product_id",
                        "therapy_class",
                        "provider_specialty",
                        "status",
                        "place_of_service",
                    ]
                ]
            )
    events = pd.concat(frames, ignore_index=True)
    return normalize_events(events)


def normalize_events(events):
    out = events.copy()
    out["code_token"] = out.code_system.astype(str) + "::" + out.code.astype(str)
    out.sort_values(["snapshot_id", "event_date", "event_id"], kind="stable", inplace=True)
    out["days_before_index"] = (out.index_date - out.event_date).dt.days
    out["time_since_previous_event"] = (
        out.groupby("snapshot_id").event_date.diff().dt.days.fillna(0)
    )
    out["event_order"] = out.groupby("snapshot_id").cumcount()
    return out.reset_index(drop=True)


def fit_sequence_vocabularies(events, min_frequency=1):
    vocabularies = {}
    for key, col in VOCAB_COLUMNS.items():
        counts = events[col].fillna("unknown").astype(str).value_counts()
        values = sorted(
            v for v in counts[counts >= min_frequency].index if v not in {PAD_TOKEN, UNK_TOKEN}
        )
        vocabularies[key] = {PAD_TOKEN: 0, UNK_TOKEN: 1, **{v: i + 2 for i, v in enumerate(values)}}
    return vocabularies


@dataclass(frozen=True)
class EventSequenceDataset:
    snapshot_ids: np.ndarray
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
    available_dates: np.ndarray
    attention_mask: np.ndarray
    lengths: np.ndarray
    events: pd.DataFrame
    vocabularies: dict

    def __len__(self):
        return len(self.snapshot_ids)

    def to_sequence_split(self):
        from therapy_switch.models.contracts import SequenceSplit

        return SequenceSplit(
            values=np.stack(
                [
                    self.event_type_ids,
                    self.code_ids,
                    self.therapy_class_ids,
                    self.specialty_ids,
                    self.days_before_index,
                ],
                axis=-1,
            ).astype(np.float32),
            mask=self.attention_mask,
            times=self.time_delta_days,
            event_dates=self.event_dates,
            index_dates=self.index_dates,
            available_dates=self.available_dates,
            categorical_sizes=tuple(len(self.vocabularies[key]) for key in VOCAB_COLUMNS),
            pre_index_verified=True,
        )


def encode_event_sequences(events, cohort, config, vocabularies):
    maximum = int(config_value(config, "features.sequence.max_length", default=96))
    if maximum < 1:
        raise ValueError("Sequence max_length must be positive")
    for key in VOCAB_COLUMNS:
        vocabulary = vocabularies[key]
        if (
            vocabulary.get(PAD_TOKEN) != 0
            or vocabulary.get(UNK_TOKEN) != 1
            or sorted(vocabulary.values()) != list(range(len(vocabulary)))
        ):
            raise ValueError("Vocabularies require contiguous IDs with PAD=0 and UNK=1")
    shape = (len(cohort), maximum)
    tokens = [np.zeros(shape, dtype=np.int64) for _ in range(4)]
    delta, recency = np.zeros(shape, dtype=np.float32), np.zeros(shape, dtype=np.float32)
    dates = np.full(shape, np.datetime64("NaT"), dtype="datetime64[ns]")
    available = dates.copy()
    mask = np.zeros(shape, dtype=bool)
    groups = dict(tuple(events.groupby("snapshot_id", sort=False)))
    retained = []
    for i, row in enumerate(cohort.itertuples(index=False)):
        group = groups.get(row.snapshot_id)
        if group is None or group.empty:
            continue
        group = group.sort_values(["event_date", "event_id"], kind="stable").tail(maximum).copy()
        group["time_since_previous_event"] = group.event_date.diff().dt.days.fillna(0)
        length = len(group)
        mask[i, :length] = True
        for array, (key, col) in zip(tokens, VOCAB_COLUMNS.items()):
            array[i, :length] = (
                group[col].fillna("unknown").astype(str).map(vocabularies[key]).fillna(1).to_numpy()
            )
        recency[i, :length] = group.days_before_index
        delta[i, :length] = group.time_since_previous_event
        dates[i, :length] = group.event_date
        available[i, :length] = group.available_date
        retained.append(group)
    return EventSequenceDataset(
        cohort.snapshot_id.to_numpy(),
        cohort.patient_id.to_numpy(),
        cohort.index_date.to_numpy(),
        cohort.label.to_numpy(),
        *tokens,
        delta,
        recency,
        dates,
        available,
        mask,
        mask.sum(axis=1),
        pd.concat(retained, ignore_index=True) if retained else events.iloc[:0],
        vocabularies,
    )


def build_event_sequences(tables, cohort, config=None, *, vocabularies=None):
    config = config or {}
    events = build_event_frame(tables, cohort, config)
    if vocabularies is None:
        vocabularies = fit_sequence_vocabularies(
            events, int(config_value(config, "features.sequence.vocab_min_frequency", default=1))
        )
    return encode_event_sequences(events, cohort, config, vocabularies)
