# Advanced Therapy Switch Prediction

A reproducible benchmarking framework for asking a deliberately neutral question:
can longitudinal deep learning materially improve advanced-therapy switch
prediction over strong classical claims models?

For a short summary of the confirmed client workflow and proposed deep-learning
evaluation, see the
[Word project brief](docs/NT1_Advanced_Therapy_Current_Findings_and_DL_Plan.docx)
or its [editable Markdown source](docs/current_project_findings_and_dl_plan.md).
The focused [LightGBM-versus-GRU test](examples/test_lightgbm_vs_gru.py) shows
how to run the first comparison on one leakage-safe temporal cohort.

The project supports synthetic claims for development and canonicalized real
claims later. It keeps two comparisons separate:

1. **Tabular:** logistic regression, random forest, XGBoost, LightGBM, CatBoost,
   and an MLP on the same leakage-safe aggregate features.
2. **Longitudinal:** the best aggregate classical model versus LSTM, GRU,
   bidirectional LSTM, and a compact temporal Transformer on pre-index events.

It is a commercial analytics and HCP-opportunity workflow—not a treatment
recommendation system. Scores are predictive associations and must not be used
to direct clinical care.

## Quick start

Python 3.10 or newer is required.

```powershell
py -3.10 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
therapy-switch run --config configs/quickstart.yaml
pytest
```

Install the full model suite when compute and platform policy permit:

```powershell
python -m pip install -e ".[all,dev]"
therapy-switch run --config configs/default.yaml
```

Core execution requires scikit-learn. Optional model packages are imported only
when needed. If XGBoost, LightGBM, CatBoost, PyTorch, Optuna, or SHAP is missing,
the run records `NOT APPLICABLE` and a reason; it never invents a score or
silently removes the model.

Useful commands:

```powershell
# Generate canonical synthetic input tables without training
therapy-switch generate --config configs/quickstart.yaml --output-dir data/synthetic

# Validate canonical real-data files and temporal coverage
therapy-switch validate-data --config configs/real_claims.yaml

# Run one or both split experiments
therapy-switch run --config configs/default.yaml
```

## Repository layout

```text
configs/                         Run settings and non-proprietary therapy mappings
docs/                            Data contract, leakage, design, operations
src/therapy_switch/
  data/                          Synthetic generator, cohort, splits, sequences
  features/                      Leakage-safe aggregate feature engineering
  models/                        Classical, tabular DL, sequence DL, tuning
  evaluation/                    Metrics, deciles, bootstrap, calibration, plots
  hcp/                           Patient-to-HCP attribution and opportunity ranking
  pipeline.py                    End-to-end experiment orchestration
  cli.py                         Command-line entry point
tests/                           Unit and integration tests, including leakage tests
outputs/                         Reproducible reports (ignored by default)
artifacts/                       Fitted models and metadata (ignored by default)
```

## Canonical data inputs

Real claims are adapted into four tables:

- `patients`: patient identifier, demographics, geography, observable start/end.
- `medical_claims`: service date, diagnosis, procedure, provider, place of service.
- `pharmacy_claims`: fill date, product, mapped therapy class, supply, prescriber.
- `providers`: provider identifier, specialty, geography, organization.

CSV and Parquet are supported. File names and source-to-canonical column renames
can be declared under `data.tables` in YAML. Therapy classification is always a
configuration mapping; proprietary product definitions never belong in model
code. See [the data contract](docs/data_contract.md).

## Temporal design and leakage controls

For every patient, the observation window ends on the index date and the label is
determined only in the subsequent prediction window. Eligibility requires enough
history and follow-up, conventional therapy exposure, and no prior advanced
therapy. Features and sequences filter each event at or before the index date.

Both patient-stratified and out-of-time splits are available. The temporal split
orders patients by index date and is the primary decision view by default. Model
selection, threshold choice, and calibration use validation data only; the test
population stays untouched until final scoring. See
[leakage controls](docs/leakage_controls.md).

## Evaluation and outputs

Every successful model is evaluated with discrimination, calibration, operating
point, and field-capacity metrics. The outputs include:

- `model_benchmark.csv` — required full benchmark, timing, and explicit NA rows.
- `executive_benchmark.csv` — presentation view of PR-AUC, capture, lift,
  complexity, and the evidence-based recommendation.
- `decile_analysis.csv` — per-model decile performance and cumulative lift.
- `cumulative_gains.csv` — population targeted versus switchers captured.
- `bootstrap_confidence_intervals.csv` and `paired_model_comparison.csv`.
- `patient_propensity_scores.csv` and `hcp_targeting_output.csv`.
- ROC/PR, calibration, gains/lift, decile, and model-comparison charts.
- run manifest, cohort statistics, selected thresholds, failure reasons, and
  training histories under the artifact directory.

Ranking metrics are primary: PR-AUC, Recall@Top-X%, Precision@Top-X%, lift,
deciles, and cumulative gains. Accuracy is intentionally not a selection target.
Small point-estimate differences are not treated as meaningful without paired
bootstrap uncertainty, stability, cost, and explainability context.

## Real-data configuration example

```yaml
data:
  source: files
  input_dir: D:/governed/claims_snapshot
  file_format: parquet
  tables:
    patients:
      file: patient_dimension
      columns:
        member_token: patient_id
    pharmacy_claims:
      file: rx_claims
      columns:
        service_date: fill_date
        product_code: drug_id
        mapped_class: therapy_class
```

Copy `configs/default.yaml`, replace only environment-specific paths/mappings,
and validate before training. Do not commit row-level real claims, identifiers,
secrets, or governed therapy mappings.

## Reproducibility and governance

- Fixed seeds are applied to Python, NumPy, scikit-learn, and PyTorch when present.
- Patient identifiers never cross partitions; temporal boundaries are recorded.
- Runtime and dependency failures are retained as benchmark evidence.
- Patient scoring and HCP prioritization are separate, auditable layers.
- HCP attribution and score weights are configurable business rules.
- SHAP or fallback importance describes association, not causality.
- Generated data, model binaries, and row-level outputs are ignored by Git.

The synthetic generator is designed only for engineering and test coverage. It
does not reproduce, infer, or reverse-engineer Komodo Healthcare Map data.
