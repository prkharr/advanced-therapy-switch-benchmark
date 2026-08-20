# Leakage-control strategy

Target leakage is treated as a pipeline invariant, not a modeling convention.

## Time boundaries

For each patient:

```text
[index - observation_window, index] -> features and sequence events
(index, index + prediction_window]  -> label only
```

An event on the index date may be used only if the operational scoring workflow
would have received it by score time. If claim latency makes this unrealistic,
configure or implement a data-lag cutoff and test it explicitly.

## Enforced controls

- Cohort construction derives outcome from advanced-therapy fills strictly after
  index and excludes any advanced exposure at or before index.
- Every feature extraction join carries the patient-specific index date and
  filters service/fill dates before aggregation.
- Recency, trend, adherence, utilization, therapy history, provider, and sequence
  fields are computed only from the observation slice.
- Split functions assign a patient once and assert disjoint identifiers.
- The out-of-time experiment orders index dates and records partition boundaries.
- Preprocessing learns imputation, encoding, and scaling from training only.
- Hyperparameter selection, early stopping, thresholds, and calibration use
  validation only. Test labels are used exclusively for final evaluation.
- Oversampling, if added to a tabular training experiment, must occur after the
  split and inside the training pipeline. Sequence validation/test data are never
  oversampled.
- HCP attribution uses pre-index interactions. Future HCP interactions are not
  valid attribution evidence.

`tests/test_leakage.py` injects distinctive post-index claims and confirms they do
not change the feature vector or sequence. It also checks boundary semantics and
patient partition disjointness.

## Operational checks before launch

1. Verify source timestamps and claim latency against the intended score date.
2. Review every newly added column for creation time, not merely service time.
3. Re-run the leakage test suite after any data adapter or feature change.
4. Compare random and out-of-time performance; investigate implausibly large gaps.
5. Retain feature lineage and the exact configuration with every scored artifact.
