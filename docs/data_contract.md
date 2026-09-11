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

## Independent cohort and target

All clinical and timing rules must be explicitly configured. Do not copy labels, eligibility flags, engineered features or scores from an existing model table. Construct the population from raw diagnosis, treatment and observation histories using separately reviewed definitions.

Monthly candidate dates come from `timeline.index_date_start/end`, independent of future outcomes. Eligibility applies history, enrollment, diagnosis confirmation, conventional coverage and known prior advanced exposure. No diagnosis code list, therapy list or clinical time window is assumed. Feature window sizes and missing-value handling are engineering settings that must also be reviewed.

Exposure coverage unions fill intervals [fill_date, fill_date + days_supply), with clipping and no stockpiling. Paid/final conventional history must exist even if minimum covered days is zero. Features use service_date <= index_date - claims_lag_days and available_date <= index_date. Prior advanced exposure known under those cutoffs excludes a snapshot. Late-arriving prior exposure needs an explicit source-policy review.

A positive label requires a valid advanced fill in (index_date, index_date + prediction_window_days], available by prediction_end + label_runout_days. Negative labels require mature observable follow-up. This measures initiation; proving replacement of conventional treatment would require an additional discontinuation rule.

## Coding and source mapping

NDC/product identifiers are strings. Do not remove leading zeros or blindly pad an ambiguous 10-digit NDC. Preserve the original code and code system in the upstream mapping; normalize only with a verified segment format or reference crosswalk. `therapy_mapping.csv` is an effective-dated product-to-class dictionary, independently reviewed for the study target. Every pharmacy transaction needs one applicable mapping, including an explicit category for unrelated products.

Diagnosis normalization must preserve the ICD version and meaning, specify whether matching is exact or prefix based, and resolve multi-code arrays using verified source encoding. Canonical `diagnosis_code` holds one diagnosis per row. When exploding diagnoses, derive a unique line key and retain a procedure only once per original event. Medical activity counts are row-level proxies, not necessarily distinct encounters. Separate claim encounters before deriving encounter-based features if that is the agreed definition.

The raw availability, status, observation and mapping semantics must be documented under `source_definitions` and verified against saved query results. An event date alone does not establish when the event became available.

## Files and evidence

Place the seven canonical CSVs directly in `actual_raw_data/`. Manual downloads do not require an exporter manifest. If an export manifest is present, real-data loading verifies its completion state, as-of date and all file hashes. The Python exporter writes a completed manifest only after all seven files pass validation; failed exports cannot be loaded through that manifest.

The internal snapshot/wide/event tables are built by this pipeline and validated before fitting. They are not external model inputs. Successful preflight saves aggregate input profiles. A completed model run saves split and performance evidence separately. Source metadata and execution evidence are distinct from software unit checks.
