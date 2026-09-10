# Advanced Therapy Switch Benchmark

An offline benchmark for predicting advanced-therapy initiation within 90 days of an eligible NT1 patient snapshot. The program context is TAK861; the label does **not** mean receipt of TAK861.

Seven canonical source tables produce repeated patient snapshots, wide features and chronological events. The same benchmark also accepts prepared client inputs directly. Patient prediction and HCP prioritization are separate layers.

**SYNTHETIC BENCHMARK RESULTS** are documented in the [project report](docs/TAK861_Advanced_Therapy_Switch_Project_Report.md) and [Word report](docs/TAK861_Advanced_Therapy_Switch_Project_Report.docx). **REAL CLIENT DATA RESULTS — NOT YET RUN.** Snowpark support is implemented and mock-tested; it is **NOT VERIFIED IN SENTINEL**.

## Raw data to field-ready HCP outputs

The [implementation workbook](docs/implementation_review/Project_data_features_and_analysis_plain.xlsx) lists dataset decisions, NDC/ICD handling, features, analysis evidence and remaining dependencies. The [raw-folder workflow](docs/raw_folder_workflow.md) explains how to map the seven [SQL extraction templates](sql/raw_extract), export Snowflake data to CSV, and run the same pipeline with `--raw-dir`. Actual source mappings must be completed before extraction; no real-data performance is claimed.

Run the modular delivery pipeline with:

~~~sh
python -m pip install -e ".[deep-learning]"
python run_pipeline.py --config configs/delivery_demo.yaml
~~~

It generates synthetic raw inputs, builds a label-free cohort at a common assessment date, fits the fixed experimental neural candidate, selects the top 10% of eligible patients, and exports an **HCP-only CSV** plus a **self-contained HTML report**. Open the HTML directly in a browser; it supports filtering, sorting and CSV download without a server or internet. The default candidate is not a statistically confirmed winner.

Use score mode to apply a saved model to new raw data. Data adapters, model trainers/recipes and HCP rules are replaceable through configuration while the pipeline architecture remains the same. The [delivery guide](docs/field_delivery_pipeline.md) covers raw files, prepared inputs, model replacement, output columns, and the distinction between current eligibility and mature historical labels.

## Install and run the benchmark

Python 3.10 or newer is required. Create and activate a virtual environment using the commands appropriate to the operating system, then run:

~~~sh
python -m pip install -e ".[benchmark,dev]"
therapy-switch run --config configs/quickstart.yaml
pytest -q
~~~

The benchmark extra installs LightGBM, PyTorch and SHAP. The core package supports logistic regression and the naive reference; unavailable optional models receive explicit NOT APPLICABLE rows. Install the all extra for the retained XGBoost, CatBoost and optimization integrations. The snowflake extra is optional and is unnecessary for synthetic or file inputs.

~~~sh
therapy-switch generate --config configs/quickstart.yaml --output-dir data/synthetic
therapy-switch validate-data --config configs/quickstart.yaml
therapy-switch run --config configs/default.yaml
~~~

Quickstart uses 600 synthetic patients and shorter training. Default uses 3,000 patients, up to three monthly snapshots, a 96-event limit, up to 25 neural epochs and 500 patient-cluster bootstrap draws. CPU threads are bounded; elapsed time depends on hardware. Both configurations use strict temporal evaluation. Cohort rules and coding dictionaries are explicit synthetic engineering assumptions.

## Models and experiments

The default executes logistic regression, synthetic LightGBM reference, MLP, GRU, temporal Transformer, hybrid GRU plus wide features, and a naive prevalence reference. Three additional runs remove recency features from LightGBM, remove timing from GRU, and shuffle a GRU sequence with timing removed. Random Forest, XGBoost, CatBoost, LSTM and BiLSTM remain configurable registry entries.

Optional research models add TabM, RealMLP, residual MLP, local TabICL and TabPFN classifiers, and fixed neural ensembles. The [neural study](docs/neural_model_study.md) reports a 55-configuration search across three development cohorts and frozen confirmation on three independent synthetic cohorts. The [reproduction workflow](docs/neural_study_workflow.md) describes dependencies, local checkpoints, selection and statistical criteria. The [challenger configuration](configs/neural_challenger.yaml) runs the selected neural recipe alongside the original references.

The frozen RealMLP/TabM/MLP ensemble achieved mean reserved-test AP **0.4079**, versus **0.3728** for the original LightGBM and **0.3578** for original logistic regression, passing the predefined AP criterion. It did not significantly beat tuned logistic regression, and its top-10% capture was lower than LightGBM's. These are synthetic, conditional comparisons; the report gives the full intervals and cohort results.

In the original snapshot benchmark, all enabled models face identical retained snapshots. Preprocessing and vocabularies fit training data only. Validation selects models and optional parameters; neural stopping uses validation AP except the optional RealMLP interface, which uses validation cross-entropy. Thresholds use validation data; calibration uses patient-disjoint validation folds. Test comparisons cannot change the selected candidate.

## Patient lists at 10% capacity

The [patient capture study](docs/patient_capture_study.md) targets the future switchers captured in the top 10% of **distinct eligible patients**. It evaluates 98 configurations across three development cohorts, then freezes one recipe before confirmation on five fresh synthetic cohorts. The search includes added claims-history features, latest-assessment training, CatBoost, XGBoost, LambdaRank, neural ranking losses, EHR transformers with local masked-event pretraining, reverse-time attention and mixed ensembles.

The validation-selected recipe combines two neural networks, with 75% weight on a history-enhanced ranking model and 25% on a model using the original wide features. The report gives the complete fresh-cohort results, adjusted uncertainty and promotion criterion. The EHR implementations are explicitly Med-BERT/BEHRT-inspired; the original medical checkpoints were not used.

On five fresh synthetic test cohorts, the candidate captured **141/325 switchers**, compared with **138/325** for LightGBM and **126/325** for logistic regression, at the same 286 list places. Mean cohort recall was 43.9%, 42.9% and 39.0%, respectively. The improvement was **not statistically confirmed**; the candidate remains experimental because the prespecified improvement gate failed.

Use `therapy-switch-patients` or `python -m therapy_switch.patient_study` to prepare, search, freeze, confirm, refit a fixed recipe and write the ranked and selected patient lists. The [reproduction and scoring workflow](docs/patient_study_workflow.md) includes exact commands and input requirements. Establish current eligibility upstream: a patient's latest historical assessment alone does not prove eligibility today.

## Data interfaces

- Raw source path: canonical patients, medical claims, pharmacy claims, providers, plans, enrollment and effective-dated therapy mapping.
- Prepared path: snapshot/label, wide-feature and event-history tables; explicit predictor allowlist and reviewed availability lineage are required.
- Optional Snowpark transport: a supplied session or explicit active-session request, validated identifiers and bounded materialization. Oversized tables fail rather than being silently truncated.

The [data contract](docs/data_contract.md) defines exact fields and limitations. Sanitized [raw Snowpark](configs/snowflake_example.yaml) and [prepared-file](configs/prepared_example.yaml) configurations contain placeholders. They require approved mappings and dates before use.

## Evidence and outputs

Results are written under the configured output directory; trained models and canonical inputs are stored separately under the artifact directory. Default locations are outputs/synthetic_benchmark and artifacts/synthetic_benchmark. Generated data, row-level scores, fitted models and logs are excluded from Git.

Outputs include all-model metrics, executive comparison, same-capacity capture and lift, additional true positives against LightGBM, clustered intervals, calibration, gains/deciles, held-out patient scores, period-specific HCP outputs, SHAP or fallback tabular explanations and sequence code-occlusion sensitivity. Artifacts include split manifests, vocabulary, training histories, fitted preprocessors/models, selected calibration, reload checks and source/input hashes.

Scores and HCP rankings support commercial analysis. They are associations, not clinical recommendations or causal effects. The original benchmark counts snapshot opportunities; the patient-list study counts distinct patients. HCP aggregation retains the latest patient snapshot per targeting period. Patient-list ranking scores are not calibrated expected-switcher counts.

## Technical documentation

- [Raw-to-HCP pipeline and offline report](docs/field_delivery_pipeline.md)
- [Benchmark design](docs/benchmark_design.md)
- [Leakage controls](docs/leakage_controls.md)
- [Data contract](docs/data_contract.md)
- [Model card](docs/model_card.md)
- [Project report](docs/TAK861_Advanced_Therapy_Switch_Project_Report.md)
- [Neural search and confirmation results](docs/neural_model_study.md)
- [Neural study reproduction](docs/neural_study_workflow.md)
- [Patient capture results](docs/patient_capture_study.md)
- [Patient-list reproduction and scoring](docs/patient_study_workflow.md)
- [Focused LightGBM versus GRU example](examples/test_lightgbm_vs_gru.py)

Run ruff check src tests and pytest -q for validation. Rebuild the Word report from its editable Markdown with python scripts/build_project_document.py. The implementation is an offline research workflow; production deployment is outside its scope.

**Built with PriorLabs-TabPFN.** Optional TabPFN-v2 integration and benchmark trials use the [Prior Labs License 1.1](docs/licenses/TabPFN_LICENSE.txt). Pretrained weights are not included in this repository.
