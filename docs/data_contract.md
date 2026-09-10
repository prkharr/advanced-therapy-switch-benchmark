# Data contract

The unit of modeling is a patient snapshot, uniquely identified by snapshot_id and by the pair patient_id/index_date. A patient may have multiple monthly snapshots. Identifiers are opaque strings, including leading zeros. Dates are timezone-naive calendar dates after explicit upstream timezone normalization.

## Canonical raw inputs

| Table | Required fields |
|---|---|
| patients | patient_id, birth_year, gender, geography, observation_start, observation_end |
| medical_claims | claim_id, patient_id, claim_date, available_date, status, diagnosis_code, procedure_code, provider_id, place_of_service |
| pharmacy_claims | claim_id, patient_id, fill_date, available_date, status, drug_id, therapy_class, quantity, days_supply, prescriber_id, plan_id, patient_cost |
| providers | provider_id, specialty, geography, organization |
| plans | plan_id, payer_type |
| enrollment | patient_id, coverage_start, coverage_end, plan_id |
| therapy_mapping | drug_id, therapy_class, effective_start, effective_end |

Raw CSV and Parquet files use these table names by default. Explicit source-to-canonical column renames and file overrides belong in data.tables. Uppercase canonical names are accepted. CSV input preserves identifiers as strings and casts declared numeric fields.

Entity and claim keys must be unique and non-null. Claims and enrollment must reference known patients; non-null provider and plan references must exist. Supply must be nonnegative whole days; costs and quantities must be nonnegative. Exactly one effective therapy mapping must cover each pharmacy claim. Source class and effective mapping must agree; overlapping mapping intervals fail.

Each canonical claim is an immutable line. status is paid, final, rejected or reversed. available_date means first analytical availability and cannot precede service. Optional reversal_date is the availability time of a later reversal; retain the original paid/final status so historical state can be reconstructed. A source containing only current corrected state must be reconstructed upstream; this package cannot infer missing historical states. Rejected/reversed claims remain events and rejection features, but never count as paid exposure or outcomes.

Providers, patient attributes and plan dictionaries are treated as static within an extract. Enrollment supplies effective plan intervals; concurrent plans require an approved primary-plan resolution. Real inputs require point-in-time dimension snapshots or upstream effective-date reconstruction. The software cannot establish those semantics from column names.

## Index and cohort construction

Supply snapshot_candidates(patient_id, index_date), a patient index_date plus a configured snapshot count, or a monthly calendar using timeline.index_date_start/end. Candidate calendars must be independent of future outcomes. Eligibility applies a fixed extraction as_of_date, observable history, continuous enrollment through follow-up, diagnosis confirmation, conventional coverage and known prior advanced exposure.

The default synthetic rules are 365 days of history, a 90-day horizon, 7-day service lag, 30-day label runout, two qualifying diagnosis dates more than 90 days apart, and at least 135 covered days in 270. These are configurable engineering assumptions, not approved clinical definitions. require_conventional_on_index is false and maximum_gap_days is unset by default.

Exposure coverage unions fill intervals [fill_date, fill_date + days_supply) with carry-in clipping and no stockpiling. Windows use inclusive calendar endpoints. Paid/final conventional history must exist even when the configured minimum covered days is zero.

Features use service_date <= index_date - claims_lag_days and available_date <= index_date. Known prior advanced exposure under those cutoffs excludes a snapshot. This is an as-known exclusion: late-arriving prior exposure can remain unobserved at index. Real outcome policy must explicitly resolve such cases.

A positive label requires a valid advanced fill in (index_date, index_date + prediction_window_days], available by prediction_end + label_runout_days. Negative labels require the same mature follow-up. Claims arriving after this frozen maturity date do not retroactively change that benchmark label.

## Three prepared tables

| Input | Required contract |
|---|---|
| snapshots | snapshot_id, patient_id, cohort_id, start_dt, index_date, resp, outcome_date, lookback_start, feature_cutoff, prediction_end, label_available_date, eligible, followup_complete |
| wide | snapshot_id, feature_as_of and exactly the configured feature_columns |
| events | snapshot_id, patient_id, event_id, event_date, available_date, event_type, code_system, code, hcp_id, product_id, therapy_class, provider_specialty, status |

RESP may be uppercase and becomes resp. label is an internal alias and must agree if supplied. start_dt is the first known qualifying conventional fill; it must not exceed feature_cutoff. Labels are binary; outcome_date is present exactly for positives and strictly within the future window. Every benchmark row must be eligible with complete follow-up, and label_available_date must be mature by data.as_of_date.

The wide table must align one-to-one with snapshots. Every predictor needs data.feature_lineage with a definition, available_at_index: true and a dtype of numeric or categorical. Datetime, identifier, split and target-like predictors are rejected. Lineage is a required declaration, not automatic proof against a semantically disguised target; upstream feature logic still requires review.

Prepared events must agree with snapshot patient IDs and fit the declared history/cutoff and availability dates. Duplicate event IDs within snapshots fail. HCP/product fields may be blank. Empty event histories are supported. Date-derived ordering, recency and deltas are recalculated after validation. HCP context fields have to be supplied even when blank so attribution behavior is explicit.

The prepared path does not recompute client features or rerun raw cohort engineering. It validates the supplied canonical contract and feeds the common split/train/evaluate workflow. Training vocabulary maps PAD to 0 and unseen values to UNK 1. Both medical code kinds are separate events; sequence truncation retains the latest events using stable event-date/event-ID order.

## Optional Snowpark transport

SnowflakeAdapter accepts an existing session. Explicit use_active_session=True lazily imports get_active_session. It does not build a credential-based connection. Each table read uses limit(max_rows + 1) and rejects oversize extracts. Table identifiers are restricted to unquoted one-, two- or three-part names.

Implementation and mocks are tested. Integration state is IMPLEMENTED / NOT VERIFIED IN SENTINEL. Package availability, session behavior, actual schemas, extract size and datatype compatibility require environment validation.

Official API references: [active session](https://docs.snowflake.com/en/developer-guide/snowpark/reference/python/latest/snowpark/api/snowflake.snowpark.context.get_active_session), [bounded DataFrame limit](https://docs.snowflake.com/en/developer-guide/snowpark/reference/python/latest/snowpark/api/snowflake.snowpark.DataFrame.limit).
