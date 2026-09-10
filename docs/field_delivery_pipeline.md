# Raw data to HCP field delivery

Run one script to acquire raw data, create features, fit or load a fixed model, select the highest-scoring 10% of currently eligible patients, attribute patients to HCPs, and export an HCP CSV with an offline HTML report.

The default model is the development-selected two-network candidate from the patient capture study. **There is no statistically confirmed winner.** Its fresh-cohort improvement gate failed. The pipeline therefore labels the default model **experimental** in both deliverables. The original logistic reference is supplied as a replacement example. This delivery demo does not reopen model selection or change the frozen study.

## Run the complete demonstration

From the repository root, after creating a Python 3.10+ environment:

~~~sh
python -m pip install -e ".[deep-learning]"
python run_pipeline.py --config configs/delivery_demo.yaml
~~~

The default generates 1,200 synthetic patients and 100 synthetic HCPs, with an assessment date of 2024-12-01. It uses CPU training and requires no warehouse, external checkpoint, visualization server or browser automation package. Raw generation and feature engineering account for most runtime.

Equivalent installed command:

~~~sh
therapy-switch-deliver --config configs/delivery_demo.yaml
~~~

The script prints the completed run ID and absolute paths to:

| Deliverable | Purpose |
| --- | --- |
| hcp_targets.csv | One row per HCP with at least the configured number of priority patients |
| client_report.html | Self-contained, searchable HCP table, counts and bars; open directly in a browser |
| hcp_targets.schema.json | CSV grain, columns and score interpretation |
| model.joblib | Fitted model, preprocessing and input-policy metadata |
| run_manifest.json | Source/input/model hashes, eligibility exclusions, attribution coverage and historical metrics |

CSV, schema and HTML go into outputs/field_delivery/RUN_ID. Models, resolved configuration, audits and patient-level intermediates go into artifacts/field_delivery/RUN_ID. These directories are ignored by Git. The artifact directory is a storage separation, **not an access-control mechanism**; configure an appropriately restricted location for real data. The report and CSV are the files intended for reviewed field/client delivery. The pipeline does not send or publish them.

Every run has a unique directory. A failed run leaves a FAILED manifest and must not be treated as a completed deliverable.

## Use raw files

Edit configs/delivery_files.yaml with the extract directory, extraction as-of date, assessment date, cohort rules and therapy/code mappings. Seven files must satisfy the [canonical data contract](data_contract.md):

- patients.csv
- medical_claims.csv
- pharmacy_claims.csv
- providers.csv
- plans.csv
- enrollment.csv
- therapy_mapping.csv

Set file_format to parquet to use the same seven Parquet filenames. Source-column mappings use the existing data.tables configuration. Paths in delivery YAML are resolved relative to that YAML file, so the working directory does not alter which files are read.

~~~sh
python run_pipeline.py --config configs/delivery_files.yaml
~~~

The repository defaults use artificial SYN codes and engineering assumptions. Replace them with reviewed definitions for the intended population. A raw extraction must contain historical training landmarks as patient index_date, a supplied snapshot_candidates table from a custom adapter, or a configured historical monthly calendar. Current scoring always creates one new assessment per patient at delivery.scoring_date.

To reuse a trained artifact with a later extract:

~~~sh
python run_pipeline.py --config configs/delivery_files.yaml --mode score --model-artifact artifacts/field_delivery/RUN_ID/model.joblib
~~~

Score mode only prepares the current cohort. It does not rebuild training labels, fit a model or read a historical test set. Load trusted local joblib artifacts only. Keep the fitted estimator's package versions available.

## Stable architecture

~~~mermaid
flowchart LR
  A[Raw files / synthetic / approved adapter] --> B[Canonical tables]
  B --> C[Current eligibility + wide features + events]
  B --> D[Mature historical training data]
  D --> E[Configured trainer]
  E --> F[Versioned model artifact]
  F --> G[Current patient scores]
  C --> G
  G --> H[Top 10% distinct patients]
  H --> I[One HCP attribution per patient]
  I --> J[HCP CSV + offline HTML]
~~~

The orchestrator is therapy_switch.delivery.pipeline. Replacement points are configured Python module:function references:

| What changes | Change here | Stable contract |
| --- | --- | --- |
| Raw source, transport or feature preparation | delivery.adapter | Return DeliveryInputs with a ScoringBatch and optional PreparedInputs |
| Model family, loss, ensemble or parameters | delivery.model.recipe, or delivery.model.trainer | Return a fitted object with predict_scores(frame, events) |
| Feature meaning or eligibility rules | Source preparation, features/cohort/timeline settings and feature_contract_version | Refit; the old artifact must not silently score changed definitions |
| Patient-list capacity | delivery.patient_fraction | One assessment per eligible patient; deterministic score ties |
| Attribution, minimum HCP count and tiers | delivery.hcp | HCP-only output schema |
| Client presentation | delivery/report.py | HCP aggregate table and run metadata |

Acquisition, training, scoring and exports remain the same when a plugin is replaced. Plugins are trusted Python code installed in the runtime; the orchestrator imports the configured callable and does not execute arbitrary source files from raw data.

### Model replacement

A fixed recipe uses selected.members and selected.weights. It can name any implemented patient-model family; install that family's optional dependency. This command switches to the original logistic model and preserves the pipeline and outputs:

~~~sh
python run_pipeline.py --config configs/delivery_demo.yaml --model-recipe configs/field_model_logistic.json
~~~

The logistic example requires only the core package: python -m pip install -e ".". Its report is marked reference. The neural recipe requires PyTorch. Recipe metadata may declare evidence_status; the supplied trainer carries that status into the artifact and deliverables.

A new trainer has this interface:

~~~python
def fit_model(train, validation, train_events, validation_events, features, *, settings, seed):
    # Fit preprocessing and parameters from train only.
    # Validation may select checkpoints; no test frame is supplied.
    return fitted_model

class FittedModel:
    def predict_scores(self, frame, events):
        # Return an aligned vector of finite ranking scores in [0, 1].
        # All encoders and preprocessing state belong to the fitted object.
        return scores
~~~

These scores need not be calibrated probabilities. For a model producing unrestricted margins, include a fixed training-derived score transformation in the fitted object. The pipeline never sums ranking scores as expected switchers.

Training validates the prepared input contract, partitions patients temporally, purges unavailable labels, fits on training data with validation-only stopping, checks serialization/reload predictions, and reports historical validation/test metrics. These demo metrics are descriptive execution checks, not a new independent confirmation of the selected model. The fitted model retains the historical training subset; it is not silently refitted on the reported test patients.

### Adapter replacement

An extractor has this interface:

~~~python
def load_inputs(config, *, include_training, expected_features=None, session=None):
    # Read once, canonicalize, prepare features using only known information.
    return DeliveryInputs(scoring=scoring_batch, training=training_inputs, provenance={})
~~~

ScoringBatch contains snapshots, wide, events, providers and feature_columns. It validates:

- Exactly one unique patient and assessment at the declared assessment date; no outcome fields.
- Eligible patients only, a valid lookback/service cutoff, feature allowlist and aligned wide rows.
- Feature availability on or before assessment; finite numeric inputs.
- Events with matching patient/assessment keys, unique event IDs and valid service/availability dates.
- HCP attribution fields and a unique provider directory.

The raw adapter establishes eligibility from claims. A custom/prepared producer must establish eligibility under the same reviewed rules; declaring eligible=True is not a substitute for upstream cohort logic. Wide aggregate lineage likewise requires reviewed source calculations; date metadata alone cannot prove that a hidden future claim was excluded.

The built-in load_prepared replacement reads four scoring files: snapshots, wide, events and providers. Configure delivery.prepared.scoring_dir and format, plus data.feature_columns and data.feature_lineage. Each lineage entry needs available_at_index: true, a definition and dtype. Training mode additionally reads snapshots, wide and events from delivery.prepared.training_dir using the existing mature prepared-input contract. See [the adapter source](../src/therapy_switch/delivery/adapters.py) and [integration tests](../tests/test_delivery.py) for a complete tested example.

The existing bounded Snowpark transport can be selected explicitly through source: snowflake_raw and a supplied session. No connection is created by importing the pipeline or running the defaults. This transport is implemented and mock-tested; **NOT VERIFIED IN SENTINEL**. Materialization is local and bounded, not a distributed training system. Larger extracts should be prepared upstream and supplied through the same adapter contract.

## Current eligibility and leakage controls

Current scoring is different from the historical benchmark cohort. It requires observation and enrollment through the assessment date, without requiring future follow-up or labels. Conventional exposure, diagnosis confirmation, covered days and gaps follow the configured rules. Every paid/final advanced exposure already known at assessment excludes the patient, including an exposure within the predictor service lag.

Features use the configured historical lookback, service lag and claim availability date. A reversal changes exposure status only once it is known. Future outcomes do not enter live eligibility, features or ranking.

The artifact records its predictor schema, training-label cutoff, recipe and input-policy hash. Scoring rejects changed feature names, changed configured feature/cohort/timing rules, or a date earlier than the labels used to train/select that model. Increment delivery.feature_contract_version when changing feature calculation semantics that keep the same names, and refit. Changes to transport paths or the assessment date alone do not invalidate a compatible model.

## Field CSV and attribution

At capacity 10%, the pipeline selects ceil(0.10 × eligible patient count), sorting score descending and patient ID ascending for ties. All selected patients remain in that denominator, including those lacking a usable HCP attribution.

By default, each patient is assigned to the most recent relevant paid/final prescriber within the validated history. Relevant means conventional therapy or a configured relevant specialty. Existing attribution code resolves ties by claim frequency and HCP ID. The other supported claim-based rules are most_frequent_relevant_specialist and plurality_of_relevant_claims.

HCP rows rank by priority-patient count descending, priority-share percentage descending, then HCP ID ascending. Only HCPs meeting minimum_priority_patients are exported. Default tiers cover the first 20%, next 30%, and remaining ranked HCPs using ceiling boundaries; small lists can have empty tiers. Tiers express workload priority, not clinical categories.

| CSV field | Meaning |
| --- | --- |
| hcp_rank, hcp_id | Deterministic priority rank and provider identifier |
| specialty, geography, organization | Provider directory context |
| eligible_patient_count | Distinct currently eligible patients attributed to the HCP |
| priority_patient_count | Attributed patients selected in the global top 10% |
| priority_share_percent | Priority count divided by attributed eligible count, multiplied by 100 |
| priority_tier | Configurable tier based on HCP rank |
| targeting_reason | Plain-language explanation of counts |
| assessment_date | Common date for the current batch |
| data_scope, model_evidence | Visible provenance and validation status |

Patient identifiers, individual risk scores and outcome labels are excluded from the field CSV and HTML. Text is escaped for HTML and spreadsheet formulas. Manifests separately report unattributed patients and priority patients withheld by the HCP minimum-count rule. Empty eligible cohorts produce header-only CSVs and an explanatory report.

## View results without extra tools

Open client_report.html directly in a modern browser. It embeds its HCP summary data, styles and vanilla JavaScript. It requires no server, internet, CDNs, dashboard service or visualization package. Search by HCP/organization, filter by specialty/geography, sort columns, or download the filtered view as CSV. With JavaScript disabled, the full static table remains readable and hcp_targets.csv is available separately. Browser Print can produce a PDF if needed.

These local capabilities have been tested separately from any client environment. Browser policy and file access in the intended environment still need verification.

## Study reproducibility

The patient capacity study and its sealed source fingerprint belong to commit 649ac654ed1d9f4178c2a5ffb598a154e4bfb83c. Check out that revision for strict study replay. The delivery pipeline adds current eligibility and changes the repository fingerprint; it does not rewrite the original freeze or treat previously scored cohorts as unseen. The earlier AP study is pinned separately in its workflow.

Run ruff check src tests and pytest -q for regression checks. See the [patient capture report](patient_capture_study.md) for the actual model comparisons and uncertainty.

## Recorded local execution

The [aggregate validation record](field_delivery_validation.json) covers 1,200 synthetic patients, 37,072 medical claim rows and 23,099 pharmacy claim rows. At the common assessment date, 253 patients were eligible and 26 selected. The neural recipe produced 20 HCP targets; the logistic replacement produced 18 from the same eligible population and patient capacity. HCP count differences are not measures of predictive quality.

Direct generation and raw CSV transport produced identical neural HCP outputs. Reloaded score-only execution reproduced every patient score and the complete HCP CSV exactly. All 141 tests and lint passed locally; the browser report made zero external requests and passed filter, sort, CSV export, mobile and no-JavaScript checks. These are engineering verification results, not validation on client data.
