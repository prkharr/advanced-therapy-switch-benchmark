# Leakage controls

The benchmark separates data availability, target observability and model fitting. Source rows may contain future events; tests deliberately retain and mutate them.

## Snapshot boundaries

Each snapshot declares index_date, feature_cutoff, lookback_start, prediction_end and label_available_date. Service lag and claim availability are checked separately. Later reversals affect exposure only when their reversal timestamp is observable. Labels require the strictly future horizon and configured runout; inadequate observation or enrollment excludes the candidate.

Wide features and event sequences derive from the same snapshot history. HCP historical prescribing features query prior claim activity under the focal cutoff and exclude the focal patient; they never aggregate snapshot labels. Demographic/provider/plan dimensions require approved historical semantics upstream.

## Predictors and preprocessing

Structural validation forbids RESP, label, outcome date, future/post-index fields, target aggregates, partition fields and patient/snapshot/HCP identifiers as predictors. Prepared inputs require an explicit allowlist and reviewed availability lineage. A benign name cannot prove an upstream value is leakage-free.

Numeric imputation, scaling and categorical encoding fit training rows only. Event vocabulary fits only training events; validation/test unseen values map to UNK. Sequence tensors carry original service/availability dates and masks for an additional audit. Truncation retains recent history deterministically; the first retained delta resets to zero. Empty rows remain fully masked.

## Strict evaluation

Patient assignments are disjoint. Temporal partitions are based on each patient's first candidate index, without using labels. Training snapshots are purged unless their labels are available by the training boundary. Validation labels must be available before the test period; configured gaps can add separation. Excluded snapshots stay in a reusable split manifest.

Random patient-disjoint stratification is a development sensitivity analysis. It does not establish temporal performance. Neither splitter implements operational recurring-patient backtesting. That requires rolling retraining, as-of feature reconstruction, embargoes for overlapping horizons and explicit reuse rules.

Validation AP controls candidate selection and neural early stopping. Validation predictions control thresholds. Calibration methods are compared on patient-disjoint validation folds, then the selected method is refit on validation predictions. Test labels contribute only reported evaluation and post-hoc diagnostics, including explanations; they cannot select or refit the candidate.

## Adversarial checks

Tests mutate future/late-arriving claims, relabel snapshots, inject forbidden predictors, move events after cutoff, supply immature labels, duplicate keys, mismatch patients, and remove lineage. Further tests verify stable ordering, PAD/UNK behavior, model reload equality, patient-cluster resampling and deterministic pipeline outputs.

These controls establish behavior on canonical and synthetic inputs. They do not establish source-system timestamp accuracy, full channel capture, historical correction recovery or clinical target validity.
