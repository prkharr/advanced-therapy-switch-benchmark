# Patient-list optimization workflow

This workflow targets future advanced-therapy initiation captured in a list containing the top 10% of distinct eligible patients. It complements the historical snapshot/AP benchmark. The patient-list model is selected by mean validation recall at 10%, with mean patient-level AP breaking ties.

The executable study remains entirely synthetic. Replacing synthetic inputs with real data requires approved eligibility, therapy mappings, availability lineage and independent validation. No remote warehouse access is needed for this workflow.

For strict replay of the sealed study, use commit **649ac654ed1d9f4178c2a5ffb598a154e4bfb83c**. Later delivery-pipeline additions change the source fingerprint but do not alter this frozen study. The [raw-to-HCP pipeline](field_delivery_pipeline.md) now provides a separate current-eligibility path and offline client deliverables.

## What a patient list means

Keep the latest supplied eligible assessment for each patient, then rank by score. Do not take each patient's maximum score or maximum label across historical assessments. Break score ties by patient ID, independently of outcomes. Select ceil(0.10 × patient count) patients.

The synthetic evaluation uses the latest eligible **historical assessment available in each partition**. Its index dates differ across patients. This is not a list of patients known to be eligible at a common current date. The upstream data producer must establish current eligibility and complete observable history for an operational list. An as-of cutoff only excludes assessments after that date; it cannot infer continued eligibility from stale records.

Risk scores support ranking. Class-weighted neural outputs and monotone LambdaRank scores are not calibrated probabilities, and they must not be summed as expected switcher counts. The existing calibrated HCP workflow is a separate evaluation path.

## Dependencies

Install the reusable package with the optional boosting and research dependencies:

    python -m pip install -e ".[dev,benchmark,boosting,research]"

The checked-in recipe records the actual package versions. For numerical reproduction, use those exact versions and the matching source revision. Run this CPU study with bounded worker counts; individual models use two CPU threads.

The 55 historical candidates include optional pretrained tabular models with local checkpoint paths. Follow [the neural study dependency and checkpoint instructions](neural_study_workflow.md). There are no implicit checkpoint downloads in the study runner. Do not include checkpoint binaries in version control.

## Prepare and search

The [candidate bank](../configs/patient_search.json) contains 98 configurations: the 55 historical specifications and 43 additions. The [protocol](../configs/patient_study_protocol.json) declares development seeds 42–44 and five newly reserved confirmation seeds 9101–9105. Previously scored seeds 7301–7303 are not reused for confirmation.

For each seed, prepare the unchanged 3,000-patient generator. Example:

    python -m therapy_switch.patient_study prepare --dataset artifacts/patient_search/datasets/42 --seed 42 --config configs/default.yaml

Prepare the other seven declared seeds in the same way. Preparation seals train, validation and test snapshot/event partitions with SHA-256 manifests. The model search reads only train and validation:

    python -m therapy_switch.patient_study search --dataset artifacts/patient_search/datasets/42 --candidates configs/patient_search.json --output artifacts/patient_search/discovery/42

Repeat for 43 and 44. Search runs every candidate and records failures rather than treating a partial run as completed. Resolve a failed dependency or execution before freezing; use --retry-failed to retry those records. A changed specification or model implementation cannot silently reuse a cached result.

When exact historical artifacts exist, --legacy-cache artifacts/dl_search/discovery/42 reuses their development predictions after checking the specification and byte-identical train/validation inputs. Historical models need not be refitted for a new metric on unchanged development patients. These records are explicitly marked as reused; they are not counted as new training runs. Every selected model is refitted from training data on fresh confirmation cohorts.

## Model comparisons

All models receive the same eligible evaluation patients. Variants alter training representation or objective, not outcomes or synthetic generation:

- Original LightGBM and logistic settings, tuned tabular baselines, RealMLP, TabM, MLP, residual MLP and optional pretrained tabular candidates.
- CatBoost, XGBoost and small LightGBM models selected at validation checkpoints by patient recall and then AP.
- Training on all eligible snapshots versus only the latest eligible training assessment.
- Train-vocabulary code counts in 14-, 30-, 90- and 365-day windows, code recency, visit-date counts, code diversity and visit gaps. These added predictors are offered to classical and neural models.
- Quarter-grouped LambdaRank and a neural BCE objective with an optional hard-negative pairwise ranking loss.
- A compact EHR transformer, with and without train-only masked-event pretraining, each tested with and without a wide-feature branch.
- A reverse-time attention model inspired by RETAIN.
- Fixed probability/score blends across model families, selected solely on the three development cohorts.

The transformer uses learned event/code, position and alternating-visit embeddings, elapsed-time features, two layers, width 64 and a 64-event limit in this bank. Five masked-event pretraining epochs use one history per training patient. Fifteen percent of eligible tokens are selected, with an 80% mask / 10% random / 10% unchanged corruption policy. Validation histories do not train the tokenizer or the pretraining objective. Supervised early stopping uses the distinct-patient list objective.

These are original compact **Med-BERT/BEHRT-inspired** and **RETAIN-inspired** implementations, not reproductions of their published checkpoints or clinical results. Med-BERT's official repository states that its pretrained weights can no longer be shared. Synthetic SYN codes also have no validated mapping to a real ICD vocabulary. Consequently this study cannot measure the transfer value of the original Med-BERT model.

Sources: [Med-BERT official repository](https://github.com/ZhiGroup/Med-BERT), [BEHRT official repository](https://github.com/deepmedicine/BEHRT), [RETAIN paper](https://papers.nips.cc/paper_files/paper/2016/hash/231141b34c82aa95e48810a9d1b33a79-Abstract.html), [official TabM implementation](https://github.com/yandex-research/tabm).

## Freeze before confirmation

Once all candidates have completed:

    python -m therapy_switch.patient_study freeze --root artifacts/patient_search --candidates configs/patient_search.json --protocol configs/patient_study_protocol.json --previous-recipe docs/neural_study/frozen_recipe.json --output artifacts/patient_search/frozen_recipe.json

Freeze compares individual candidates, equal blends, fixed pair blends and a bounded deterministic Dirichlet weight search. All cohorts use identical members and weights. It records source hashes, runtime versions, checkpoint hashes, all prepared manifests, the exact selected recipe and the complete development selection trace. It refuses to overwrite an existing recipe.

Three primary controls are fixed: original LightGBM, original logistic regression and the previous five-member neural ensemble. The strongest development classical model and EHR model are fixed secondary controls. Their test results cannot replace the selected candidate.

Fit each reserved cohort before evaluating any test partition:

    python -m therapy_switch.patient_study fit-confirmation --root artifacts/patient_search --recipe artifacts/patient_search/frozen_recipe.json --seed 9101

Repeat for 9102–9105. Every model is serialized, reloaded and checked on validation predictions. The evaluator refuses to open a test partition until every required model file for all five cohorts matches its sealed fit manifest:

    python -m therapy_switch.patient_study evaluate-confirmation --root artifacts/patient_search --recipe artifacts/patient_search/frozen_recipe.json

The primary gate requires mean recall gain of at least 0.03 and a positive lower confidence limit against all three primary controls. The 3,000 paired patient bootstrap draws resample independently within each cohort, rebuild each model's top-10% list on every draw and average cohort differences equally. Bonferroni adjustment gives 98.333% intervals per primary comparison at family alpha 0.05. Intervals condition on fitted models; they do not estimate uncertainty from retraining or generalization beyond this synthetic generator.

Top-5% and top-20% results, precision, lift, AP, per-cohort counts and the two secondary comparisons provide context. They do not change the primary criterion. The secondary comparisons use their own two-comparison family and cannot substitute for a failed primary gate.

## Produce a reviewable list

Use a trusted locally created model artifact. For example, the first reserved cohort's model can demonstrate scoring its held-out synthetic assessments:

    python -m therapy_switch.patient_study score --model artifacts/patient_search/confirmation/9101/champion.joblib --snapshots artifacts/patient_search/datasets/9101/test.parquet --events artifacts/patient_search/datasets/9101/test_events.parquet --output outputs/patient_capture/ranked_synthetic_patients.csv

This writes all distinct patients with patient_id, snapshot_id, index_date, risk_score, rank and selected, plus a .selected.csv file containing only the capacity-limited list. Outcomes are not required or consumed for inference.

To refit the fixed recipe on a new approved local training/validation dataset that implements the same sealed partition contract:

    python -m therapy_switch.patient_study fit-list --dataset artifacts/new_local_dataset --recipe docs/patient_study/frozen_recipe.json --output artifacts/new_local_model/champion.joblib

Training and validation patients must be disjoint. This command does not open a test partition. Refit alone does not establish performance on a new population: keep a separate untouched evaluation cohort, assess availability drift and calibrate probabilities separately if needed.

The local partition contract consists of train.parquet, validation.parquet, train_events.parquet, validation_events.parquet and ready.json. Snapshot files contain patient_id, unique snapshot_id, index_date, feature_cutoff, lookback_start, a binary label and the explicit predictor columns. Normalized event files contain snapshot_id, event_id, event_date, available_date, event_type and code_token. Additional normalized event columns may be retained. All histories must respect feature_cutoff, index-time availability and lookback_start.

The ready.json object provides seed, features (the predictor allowlist), and hashes (SHA-256 values keyed by train, validation, train_events and validation_events). The study's synthetic preparer also supplies test partitions and hashes for confirmation. Existing raw/prepared adapters and the temporal patient splitter can produce the same frames from reviewed local inputs. Do not include identifiers, outcomes, split fields or post-index information in the predictor allowlist.

Generated datasets, patient lists, row-level predictions, fitted models and environment-specific material remain ignored. Only reusable code, configurations, tests and reviewed aggregate synthetic evidence belong in Git.
