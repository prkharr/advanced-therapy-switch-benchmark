# Advanced Therapy Switch Benchmark

Build an independent patient cohort and compare fresh logistic-regression and LightGBM baselines using raw claims. Rank the top 10% of eligible patients and aggregate them to relevant HCPs. No existing cohort, feature, label or score table is required.

Actual-data performance has not been measured here. Source mappings and population, therapy, outcome and timing definitions must be completed before execution. Advanced-therapy initiation is the implemented outcome; it does not by itself establish that conventional treatment stopped.

## Set up and run

From the repository folder in a terminal:

```sh
python -m pip install -e .
python setup_real_data.py
```

1. Complete the null values and code lists in `configs/private/benchmark.yaml`. Record the rationale and evidence for each rule under `source_definitions`. Complete `delivery.scoring_date` and HCP specialties in `configs/private/delivery.yaml`.
2. Review the queries in `sql_extraction/`. Map the actual source fields, create an independent raw-source population filter and resolve every placeholder. The source-mapping examples require review; they are not verified warehouse queries.
3. Execute the seven canonical SELECT queries in Snowflake and download each result as CSV into `actual_raw_data/`. Use exactly these names: `patients.csv`, `medical_claims.csv`, `pharmacy_claims.csv`, `providers.csv`, `plans.csv`, `enrollment.csv`, `therapy_mapping.csv`. Keep identifier and NDC fields as text. See [the contract](docs/data_contract.md).
4. Validate the files, then execute:

```sh
python run_pipeline.py --check
python run_pipeline.py
```

The check writes an aggregate `raw_profile.json` under `artifacts/field_delivery/input_check/` after successful input validation. Missing definitions or invalid data stop the run; the program does not substitute data or infer missing clinical rules.

For automatic extraction, install `python -m pip install -e ".[snowflake]"`, configure a local named Snowflake connection and run `python run_pipeline.py --check --extract`, followed by `python run_pipeline.py --extract`. This uses the same seven SQL files and raw folder. SQL alone cannot write into a local folder; download through the warehouse UI or use the Python exporter. See [extraction instructions](sql_extraction/README.md).

## Outputs and model comparison

Each run has a unique folder under `outputs/field_delivery/` containing `hcp_targets.csv` and `field_report.html`. Open the HTML directly in a browser; no server is needed. Restricted patient-level records and fitted models are kept under the corresponding `artifacts/field_delivery/` run folder.

Fresh baselines share the same raw-derived features and patient-disjoint temporal partitions. Preprocessing is fitted on training data. Validation recall at the requested capacity selects the model; average precision breaks ties. Both models are evaluated on the held-out test partition after selection. `baseline_comparison.csv` and `training_audit.json` record the comparison, parameters, runtime versions, partition counts and reload check. These files provide execution evidence only after a successful run.

The initial settings are explicit engineering defaults, not a tuned winner. Review temporal partitions and class counts before interpreting results. No improvement or statistically significant result is claimed before actual-data evaluation.

## Project structure

```text
actual_raw_data/          Empty input folder for the seven downloaded CSVs
sql_extraction/           Canonical SELECT templates and evidence queries
  source_mapping/        Raw medical and pharmacy mapping scaffolds
configs/                 Public templates; private settings stay local
src/therapy_switch/      Validation, cohort, features, baselines and HCP outputs
docs/Features_and_analysis.xlsx  Plain feature dictionary and analysis evidence
tests/                   Contract, leakage, ranking and model checks
```

Change source mappings in SQL, feature definitions in the feature module, and the trainer through `delivery.model.trainer`. Each trainer returns `predict_scores(frame, events)`. The raw-input and delivery interfaces stay the same. Changing population, timing or feature meaning requires retraining a compatible model.

Run checks with `python -m pip install -e ".[dev]"`, `python -m ruff check src tests *.py` and `python -m pytest -q`. Warehouse integration and outcome validity still require an actual-data run and review. Downloads, credentials and fitted artifacts are excluded from Git.
