# Benchmark design

The research question is whether ordered longitudinal events add useful signal beyond wide claims features. The target is advanced-therapy initiation/use in a configured future window. A synthetic LightGBM reference is not a reproduction of a client model or an external AutoML baseline.

## Controlled comparisons

| Run | Representation | Purpose |
|---|---|---|
| Logistic Regression | Full wide | Linear reference |
| LightGBM | Full wide including recency | Main nonlinear reference |
| MLP | Full wide | Tabular neural comparison |
| GRU | Ordered code/type/class/specialty embeddings plus time | Recurrent sequence comparison |
| Transformer | Same event vocabulary with temporal and position inputs | Attention encoder comparison |
| Hybrid GRU Wide | GRU representation plus preprocessed wide features | Incremental sequence information |
| LightGBM Without Recency | Reduced wide feature list | Engineered recency ablation |
| GRU Without Time | Ordered events with recency/deltas zeroed | Timing ablation |
| GRU Shuffled Without Time | Deterministically shuffled events with time zeroed | Ordering ablation against no-time GRU |
| Naive Baseline | Training prevalence | Constant-score reference |

Removing time before shuffling prevents timestamps from reconstructing original order. Event counts and the multiset remain; shuffled inference is deterministic per sequence. The reduced LightGBM run removes short rolling windows, recency, trend and coverage-change features but retains coarse history; persisted feature lists define the exact comparison.

Random Forest, XGBoost, CatBoost, LSTM and BiLSTM remain optional. A disabled or unavailable model has a reason and blank metrics; a failed model is explicitly FAILED. Optional dependencies cannot be interpreted as a negative result.

## Split and fitting policy

Default evaluation is patient-disjoint temporal, with labels mature before subsequent scoring periods. Fractions determine patient date boundaries; date ties and maturity purging mean realized row fractions differ. All models receive identical retained partitions, with modality-specific representations. The full snapshot manifest also records purged rows.

All fitting uses training data, with validation-only stopping, thresholds and selection. Neural stopping monitors AP; the learning-rate scheduler monitors validation loss. The default uses fixed compact architectures and weighted BCE; optional focal loss is supported. Hyperparameter tuning is disabled in the reported default run. This is a practical common-data comparison, not a claim of equal hyperparameter-search effort or optimal tuning for every family.

No test-performance fallback is allowed for selection. Missing validation evidence yields no recommendation. The candidate is an offline research choice, not a deployment approval.

## Metrics and uncertainty

PR-AUC means scikit-learn average precision, not trapezoidal integration. Capacity metrics rank scores descending with stable input-order ties and choose ceil(N * capacity) rows. Recall divides captured positives by all positives; precision divides by selected rows; lift divides precision by prevalence. At top 5%, 10% and 20%, challengers also report additional captured true positives relative to the same-capacity LightGBM ranking.

Evaluation is over snapshot opportunities. Repeated horizons may contain the same initiation event; this is not a count of unique clinical initiations. HCP outputs use a separate period deduplication rule. Stable tie order makes the naive model's capacity capture arbitrary; its expected capture is the selected population fraction.

ROC-AUC, Brier score, calibration curves, threshold precision/recall/F1 and timings are secondary diagnostics. Deciles and cumulative gains show capture across capacity.

Paired patient-cluster bootstrap draws patients with replacement and retains all their snapshots, including duplicate-cluster multiplicity. Both models use each identical resample. Single-class draws are skipped and successful/requested counts recorded. Percentile 95% intervals are descriptive, conditional on fitted models; they exclude training uncertainty and calendar/provider clustering. Comparisons across many models/capacities are not multiplicity-adjusted. A small point difference is insufficient evidence of superiority.

## External baseline

evaluation.external_reference_file accepts snapshot_id and reference_score. It must contain exactly one valid score for every retained test snapshot and no extra population. Before comparison, independently establish that the external model used compatible target definitions and training/validation/test exclusions. Alignment validates population, not training provenance.

## Reproducibility and interpretation

Resolved configuration, split manifest, train vocabulary, model/preprocessor artifacts, thresholds, calibration selection, training histories, source/input hashes and runtime versions accompany each run. Estimators are reloaded and their test predictions checked numerically.

Synthetic findings establish functioning engineering and recovery of deliberately simulated signals. They cannot establish clinical validity, performance on governed data, superiority over an unavailable real baseline or production readiness.
