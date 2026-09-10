# TAK861 Advanced Therapy Switch Benchmark

Engineering report for an offline claims modeling framework

10 September 2026

**SYNTHETIC BENCHMARK RESULTS**

**REAL CLIENT DATA RESULTS — NOT YET RUN**

## Purpose and conclusion

The benchmark identifies eligible NT1 patient snapshots associated with advanced-therapy initiation or use in the following 90 days. TAK861 is the program context; the outcome does not mean receipt of TAK861. Initiation can represent an add-on to conventional treatment and does not, by itself, establish a treatment replacement.

The engineering workflow now links raw claims to repeated snapshots, wide features, chronological events, model fitting, held-out evaluation and HCP opportunity aggregation. It also accepts prepared snapshot, feature and event tables directly, preserving an existing feature-engineering investment. All six priority model families and three ablations executed on the same synthetic temporal partitions.

Logistic regression was selected by validation average precision of 0.337. Its held-out average precision was 0.415. The synthetic LightGBM reference had the highest observed test average precision, 0.448, while the hybrid captured 65 positive snapshots in the top 10% versus 63 for LightGBM. Test results did not change the validation-selected candidate.

The experiment establishes that the architecture functions and can recover deliberately simulated signals. It does not establish superiority over a real client baseline, clinical validity or operational readiness. The small capacity differences must be interpreted with the paired patient-cluster intervals in this report.

### Reader and scope

This report is for engineering, analytics and model-validation reviewers. It covers implemented data contracts, assumptions, models, leakage protections, actual synthetic evidence, explanations and limitations. The code supports offline research and scoring. Production services, scheduled pipelines, deployment infrastructure and model promotion are outside scope.

Generic Snowpark support is **IMPLEMENTED** and mock-tested. It is **NOT VERIFIED IN SENTINEL**. Real source schemas, clinical mappings, feature lineage, claim observability and external-baseline comparability remain validation dependencies.

<!-- pagebreak -->

## Data architecture

Seven canonical raw tables represent patients, medical claims, pharmacy claims, providers, plans, enrollment and effective-dated therapy mappings. Entity and claim keys are unique; patient/provider/plan joins are checked; identifiers remain strings. Each pharmacy claim must match exactly one effective therapy classification.

The raw-source adapter applies explicit column mappings, normalizes canonical names and validates datatypes and relationships. The prepared-input adapter accepts the three modeling tables below and validates their alignment and timing without rerunning raw cohort or feature engineering.

| Modeling input | Key | Main content |
|---|---|---|
| Snapshot and label | snapshot_id | Patient, cohort, conventional start, index, RESP, outcome and maturity |
| Wide features | snapshot_id | Reviewed predictors and feature availability timestamp |
| Event history | snapshot_id and event_id | Patient, service/availability dates, event type, code, therapy and HCP context |

A patient may contribute several snapshots. Every event is linked to a particular snapshot, so joins cannot multiply a patient's histories across unqualified index dates. Prepared wide inputs require an explicit predictor allowlist and a definition, datatype and availability declaration for every feature. These declarations require upstream review; they cannot prove that a disguised target-derived value is safe.

### Synthetic generation

The reported run generated 3,000 patients, 150 providers, 3 plans, 91,690 medical claims and 57,750 pharmacy claims. Artificial dictionaries and identifiers are independent of proprietary code lists. Longitudinal histories include diagnosis/procedure events, emergency and urgent care, conventional refills, gaps, changes, patient costs, rejected claims, delayed availability and later reversals.

Risk is a noisy function of generated pre-index behavior, including recent versus prior utilization, treatment changes, refill gaps and rejection patterns. Labels are sampled probabilistically, not deterministically assigned or made completely random. Future events remain in the raw tables to exercise leakage filtering.

The generator targets approximately 12% patient initiation probability after the last anchor. Earlier snapshots have shorter overlap with that generated outcome period; the resulting eligible snapshot prevalence is 7.13%. Neither rate is an estimate of real patient behavior.

### Transport boundary

Optional Snowpark support uses a supplied session or an explicit active-session request. It does not create a credential-based connection. Reads are bounded and oversize inputs fail rather than silently truncate. CSV, Parquet and supplied dataframes support local adapter validation.

<!-- pagebreak -->

## Cohort and point in time features

The default synthetic cohort uses a 365-day historical window, 90-day prediction horizon, 7-day claims service lag and 30-day label runout. It requires a qualifying NT1 diagnosis followed by another configured NT1/NT2/IH diagnosis more than 90 days later, and at least 135 covered conventional-therapy days in the preceding 270 days. These are engineering assumptions awaiting real-data approval.

Continuous enrollment and observation must cover the required history and future follow-up. At least one valid conventional exposure is required. The default does not require coverage on the index day and does not impose a maximum-gap exclusion; both policies are configurable. The result contains 8,981 eligible snapshots from 2,997 patients; 19 candidate snapshots were excluded.

### Exposure and label semantics

Coverage unions fill intervals from fill date through days_supply minus one, clips carry-in fills to the window and does not stockpile overlapping supply. Rejected and reversed claims do not count as exposure. A later reversal changes the historical state only once its reversal availability date is known.

Predictors require service dates no later than index minus lag and analytical availability no later than index. Known prior advanced exposure under these cutoffs excludes a snapshot. Late-arriving prior exposure is therefore an unresolved real-data policy question, not an assumed observed exclusion.

Positive labels require a valid advanced fill strictly after index and on or before prediction end, available by the frozen maturity date. Mature negative labels use the same observation requirements. Later arrivals do not retroactively update the frozen benchmark outcome.

### Wide representation

The run produced 70 predictors: demographics and payer, therapy coverage/gaps/changes, number tried and observed duration, diagnosis/procedure/pharmacy counts, emergency/urgent/specialist activity, costs and claim-status history, recency, utilization acceleration, coverage change, symptom/comorbidity counts and historical HCP prescribing.

Rolling windows are 30, 60, 90, 180 and 365 days. HCP history uses only prior available claims and excludes the focal patient; it never aggregates snapshot labels. Raw identifiers, outcome fields, partition metadata and target-like fields cannot enter predictors.

Age uses index year minus birth year. Patient, provider and plan dictionaries are static within the extract; real changing attributes require historical snapshots or upstream effective-date reconstruction. Claims column checks cannot establish the availability of those dimensions.

<!-- pagebreak -->

## Models and sequence representation

The central comparison asks whether event order and timing add value to already engineered features. All priority families execute real estimators with persisted preprocessing; the temporal models consume event tensors rather than text serialized from the wide table.

| Model | Input | Implementation |
|---|---|---|
| Logistic regression | Full wide | Scaled numeric and encoded categorical features with class weighting |
| LightGBM reference | Full wide | Weighted gradient-boosted trees with validation AP stopping |
| MLP | Full wide | Hidden layers of 64 and 32 units with dropout and weighted BCE |
| GRU | Ordered history | Learned categorical embeddings and a 32-unit recurrent representation |
| Temporal Transformer | Ordered history | Embeddings, time, learned position, masked attention and pooling |
| Hybrid GRU Wide | History and full wide | GRU representation concatenated with a learned wide-feature branch |
| Naive reference | Training prevalence | Constant probability with explicit tie-order behavior |

The sequence vocabulary contains event type/status, code-system-plus-code, therapy class and provider specialty. Each category has a separate learned embedding. Recency and inter-event deltas enter as transformed numeric time inputs. Vocabularies are fitted on training events only, with PAD 0 and UNK 1 reserved.

Histories are sorted by event date and event ID. The most recent 96 events are retained with deterministic ties; the first retained delta resets to zero. Right padding and masks distinguish real events from padding. Empty sequences have a neutral masked representation. Sequence contracts retain service and availability dates for an additional audit.

The Transformer uses one encoder layer, model width 32, four heads and a 64-unit feed-forward layer. Padding is excluded from attention and pooling. All visible events precede the cutoff; a future-token causal mask is unnecessary for classification over a completed historical window.

Neural training uses weighted binary cross-entropy, validation AP early stopping and at most 25 epochs, with dropout and a validation-loss learning-rate scheduler. Focal loss is also supported. Hyperparameter search was disabled for this report. Random Forest, XGBoost, CatBoost, LSTM and BiLSTM remain supported but disabled in the main run.

<!-- pagebreak -->

## Evaluation design

Patients are assigned by their first snapshot date into disjoint chronological partitions. Labels are not used to choose temporal boundaries. Snapshots are purged if their labels would not have been observable by the next decision boundary. Date ties and purging mean realized fractions differ from the nominal 60/20/20 split.

| Partition | Snapshots and patients | Positive snapshots |
|---|---|---|
| Training | 4,690 snapshots and 1,633 patients | 344 |
| Validation | 864 snapshots and 348 patients | 52 |
| Test | 1,644 snapshots and 549 patients | 126 |

Training index dates end on 1 June 2023 and labels mature by 29 September 2023. Validation starts on 1 November 2023 and ends on 1 April 2024, with labels mature by 30 July 2024. Test index dates span 1 September 2024 through 1 July 2025. The split manifest retains 1,783 purged eligible snapshots with their disposition.

### Selection and untouched test policy

Imputation, numeric scaling, categorical encoding and sequence vocabularies fit training data only. Validation controls stopping, thresholds and candidate selection. Calibration candidates are compared on patient-disjoint validation folds; the chosen calibration is then refit on validation predictions. Missing validation evidence cannot fall back to test performance.

Test scores are evaluated and explained after model fitting. Test metrics, paired intervals and post-hoc explanations cannot change the frozen candidate. The strict patient-disjoint design differs from recurring-patient operational backtesting, which would require rolling refits, historical as-of reconstruction and explicit controls for overlapping horizons.

### Metrics and counting unit

PR-AUC here means average precision. At 5%, 10% and 20% capacity, recall measures the share of positives captured, precision measures the positive fraction of selected rows, and lift divides that precision by prevalence. Selection uses the ceiling of population times capacity, with stable ordering for tied scores.

The test population has 7.66% positive snapshots. Top 10% selects 165 snapshot opportunities. These are not necessarily 165 unique patients or unique treatment initiations; repeated horizons can share an outcome. HCP prioritization applies separate patient-period deduplication.

ROC-AUC, Brier score, calibration, threshold metrics and timings supplement ranking. Deciles and cumulative gains describe capture across the score distribution. External baseline scores require exact test-snapshot alignment and separately verified training provenance.

<!-- pagebreak -->

## Synthetic benchmark findings

**SYNTHETIC BENCHMARK RESULTS — seed 42, strict temporal test**

| Model | Val AP | Test AP | Recall 10% | Lift 10% |
| --- | --- | --- | --- | --- |
| Naive Baseline | 0.060 | 0.077 | 11.1% | 1.11 |
| Logistic Regression | 0.337 | 0.415 | 47.6% | 4.74 |
| LightGBM | 0.327 | 0.448 | 50.0% | 4.98 |
| MLP | 0.315 | 0.432 | 47.6% | 4.74 |
| GRU | 0.272 | 0.314 | 46.8% | 4.67 |
| Transformer | 0.263 | 0.355 | 48.4% | 4.82 |
| Hybrid GRU Wide | 0.298 | 0.413 | 51.6% | 5.14 |
| LightGBM Without Recency | 0.257 | 0.352 | 42.1% | 4.19 |
| GRU Without Time | 0.275 | 0.311 | 44.4% | 4.43 |
| GRU Shuffled Without Time | 0.280 | 0.299 | 45.2% | 4.51 |

Average precision is shown for validation and test. Recall and lift use the top 165 of 1,644 test snapshots. The naive model's capacity result depends on the stable ordering of tied scores; it is not a learned targeting rule.

Logistic regression remains the validation-selected candidate. LightGBM had the highest test average precision, but selecting it because of that result would reuse the final test for model selection. The reported choice therefore remains logistic regression.

The hybrid's test ROC-AUC was 0.823 versus 0.821 for LightGBM. Its slightly higher top-decile capture did not translate into higher average precision. Different metrics answer different questions, and this single synthetic run does not justify a general deep-learning superiority claim.

The priority models, naive baseline and all three ablations completed. Five retained optional model entries were explicitly marked disabled with blank metrics. Every fitted estimator was saved, reloaded and checked against its original held-out predictions. Neural training histories and training vocabularies were also persisted.

The complete output tables include precision/recall/lift at 5%, 10% and 20%, additional captured true positives, threshold metrics, Brier score, calibration, gains, deciles and training/inference time. The committed evidence snapshot contains only aggregate synthetic results.

<!-- pagebreak -->

## Paired uncertainty and capacity value

All challengers were compared with the synthetic LightGBM reference on identical test snapshots. Each of 500 bootstrap draws resampled patients with replacement and retained all snapshots from each sampled patient, including duplicate-cluster multiplicity. The two models used the same draw.

| Challenger | AP diff | AP 95% CI | TP diff 10% | TP 95% CI |
| --- | --- | --- | --- | --- |
| Logistic Regression | -0.033 | [-0.095, 0.022] | -3 | [-11.0, 5.0] |
| MLP | -0.017 | [-0.065, 0.034] | -3 | [-11.0, 5.0] |
| GRU | -0.134 | [-0.190, -0.067] | -4 | [-15.0, 3.0] |
| Transformer | -0.093 | [-0.142, -0.039] | -2 | [-13.0, 5.5] |
| Hybrid GRU Wide | -0.035 | [-0.086, 0.014] | +2 | [-5.0, 7.5] |

Differences are challenger minus LightGBM. The capacity difference is additional positive snapshots captured at top 10%. Brackets show paired percentile 95% intervals. The point difference of two captured positives for the hybrid should not be treated as a reliable operational gain when its interval spans zero.

These intervals are conditional on the fitted models. They do not include retraining variation and do not model shared provider or calendar shocks. Comparisons across multiple models and capacities are not adjusted for multiple testing. A second seed or a different time split can change the result.

The AP intervals for GRU and Transformer lie below zero against LightGBM in this simulation. Logistic regression, MLP and hybrid AP differences include zero. All five priority-challenger top-decile capture intervals include zero. This describes the tested data and representations; it is not a general conclusion about model families.

### Calibration and probability interpretation

Patient-fold validation selected sigmoid calibration for logistic regression. Held-out Brier score decreased from 0.155 to 0.062; lower is better. The calibration choice was made using validation predictions, and its saved transform was reloaded and verified.

Calibration is separate from ranking. A model can order patients usefully while producing poorly calibrated probabilities, especially after class weighting. HCP expected-switcher sums depend on calibration and should not be interpreted as validated real-world expected counts from this synthetic experiment.

The exported patient score table contains only held-out identifiers, dates, raw/selected-calibration scores, model and score scope. Evaluation labels remain in separate benchmark artifacts.

<!-- pagebreak -->

## Ablation and explanation findings

| Comparison | Test AP values | Point difference |
| --- | --- | --- |
| Full vs reduced LightGBM | 0.448 vs 0.352 | +0.096 |
| GRU with vs without time | 0.314 vs 0.311 | +0.003 |
| No-time GRU ordered vs shuffled | 0.311 vs 0.299 | +0.012 |

The LightGBM comparison removes short rolling windows, recency, trend and coverage-change predictors; exact feature lists are persisted. The GRU time ablation zeros both recency and inter-event deltas. Its shuffled comparison removes time in both arms, preserves the event multiset and uses deterministic shuffling so timestamps cannot restore order.

These point differences suggest useful engineered recency in this simulation. The smaller GRU time and order changes remain exploratory. They are not isolated causal effects and cannot establish that a particular representation will improve governed claims performance.

### Explainability

Compatible tabular models use SHAP over their fitted preprocessing pipeline; native or permutation importance is an explicit fallback. Sequence explanations replace selected code families with UNK while retaining other event context, then measure score changes. This is code-occlusion sensitivity, not attention attribution, complete temporal attribution or a causal explanation.

SHAP completed for logistic regression and both LightGBM runs. MLP used held-out permutation AP importance. All five sequence/hybrid runs produced code-occlusion artifacts. These are post-hoc diagnostics and were not used to select or retune models.

### HCP opportunity layer

The patient model and attribution layer are independent. Within each quarterly targeting period, aggregation keeps the latest snapshot for each patient, then uses only events linked to that snapshot and available at scoring. The default attributes to the most recent relevant prescriber; specialty/relevance and score weights are configurable.

The default opportunity score is the sum of calibrated patient propensities, with eligible/high-propensity counts and exact-capacity bands also available. The output contains 914 distinct patient-quarter attributions across 5 quarters and 467 HCP-quarter rows. The same patient can legitimately appear in a later quarter, so cross-quarter totals must not be interpreted as a unique-patient total.

HCP ranking supports commercial prioritization. It is not a treatment recommendation, clinical ranking of providers or estimate of causal prescribing influence.

<!-- pagebreak -->

## Validation evidence and remaining limits

The full automated suite passed **92 tests with zero failures**. Tests cover deterministic synthetic generation, schemas, coverage union, cohort eligibility and maturity, future labels, adversarial leakage, patient-disjoint splits, sequence ordering/truncation/PAD/UNK/empty cases, neural models, hybrid behavior, metrics, patient-cluster bootstrap, serialization, prepared CSV/Parquet inputs, mocked Snowpark sessions and pipeline reproducibility.

The end-to-end 3,000-patient synthetic run executed all priority models and ablations, persisted outputs, performed model reload checks, generated explanations and HCP tables, and completed uncertainty/calibration analyses. Lint and dependency consistency checks also passed. Runtime and byte-level source/input hashes accompany the local run manifest; source control contains a small sanitized aggregate evidence copy.

### Boundaries of the evidence

Synthetic histories simplify disease progression, benefit design, channel completeness, coding ambiguity and adjudication. Supply union assumes no stockpiling. Fixed-runout labels freeze later corrections. Overlapping snapshots are correlated and may share the same future initiation. Static extract dimensions require historical reconstruction on real inputs.

The reported models use compact fixed architectures, a small artificial code vocabulary and one main seed. This is a practical controlled comparison, not exhaustive tuning or an estimate of the best attainable performance for each family. Sequence truncation and rare/unseen tokens require review at real scale. Subgroup fairness, temporal drift, clinical validity and recurring-patient backtesting have not been established.

Prepared feature lineage requires semantic review beyond field-name checks. Snowpark mocks cannot establish corporate runtime compatibility or actual source availability. The synthetic LightGBM implementation does not reproduce an unavailable client AutoML version.

### Real data validation still required

Approve the population, diagnostic confirmation, therapy mapping, exposure semantics, lag/runout, channel observability and outcome definition. Confirm actual prepared/raw schemas and historical feature availability. Establish the external baseline's training exclusions and exact held-out population before comparing scores.

Validate approved attribution rules and targeting cadence, inspect independent cohort/label reconciliations, and assess calibration, subgroup/time stability and operational costs. Governed data and fitted artifacts require approved storage and access policies.

**REAL SENTINEL / CLIENT-DATA VALIDATION REMAINS PENDING.**

The reusable integration is implemented; successful execution and model validity in the target environment remain to be demonstrated.
