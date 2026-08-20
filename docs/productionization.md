# Productionization checklist

## Data and privacy

- Run inside the approved claims environment with minimum necessary access.
- Use tokenized patient/provider identifiers and permitted geography only.
- Keep source claims, row-level scores, model binaries, and logs out of Git.
- Validate adjudication/reversal rules, claim run-out, enrollment, and refresh lag.
- Version the governed code mappings and record their effective dates.

## Model operations

- Materialize a point-in-time feature snapshot and immutable cohort manifest.
- Record configuration, package lock, code commit, random seed, temporal bounds,
  feature names, training duration, and model-selection evidence.
- Add schema, volume, missingness, prevalence, and score-distribution monitors.
- Monitor PR-AUC/capture when labels mature, plus calibration and HCP concentration.
- Define retraining, rollback, artifact retention, and human approval procedures.
- Benchmark inference throughput on the actual deployment substrate.

## Governance

- Review intended use, prohibited use, privacy, fairness, and geographic fields.
- Label outputs as commercial analytics, not medical advice.
- Validate association explanations with claims and domain experts; do not use SHAP
  or feature importance as causal evidence.
- Confirm that HCP ranking and field capacity rules are independently auditable.
- Require a documented go/no-go decision comparing the simplest adequate model to
  any higher-complexity alternative.
