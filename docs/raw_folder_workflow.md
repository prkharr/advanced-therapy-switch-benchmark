# Actual Snowflake data to raw CSVs and HCP delivery

The main script now defaults to `configs/private/delivery.yaml`. It never falls
back to the synthetic demo. The workflow is:

```text
Reviewed Snowflake SELECTs -> data/raw/EXTRACT_ID/*.csv
-> validated raw contract -> historical training and current scoring
-> top 10% eligible patients -> HCP CSV and offline HTML
```

## 1. Initialize once

From the cloned repository in your approved Python environment:

```sh
python -m pip install -e ".[snowflake,benchmark]"
python setup_real_data.py
```

This creates the files below without replacing existing edits:

| File | What to enter |
| --- | --- |
| `configs/private/benchmark.yaml` | Actual extract date/version, historical calendar, cohort and therapy rules, source definitions |
| `configs/private/delivery.yaml` | Actual scoring date, HCP specialties, named connection, model recipe and output paths |
| `configs/private/sql/*.sql` | Seven reviewed SELECT queries returning canonical columns |

There are no synthetic diagnosis/drug codes in the real-data template. Null dates
and rules deliberately require actual definitions. Empty symptom/comorbidity code
lists disable those optional groups until mapped; they are not approved code lists.
The `conventional` and `advanced` class labels must match the approved effective-dated
therapy dictionary; neither label is inferred from generic/brand status.

`source_definitions` records the source SQL/version and meaning of cohort, target,
claim availability, status, observation coverage, product mapping and diagnosis
normalization. Describe the actual implementation; these fields are not substitutes
for reviewing the source logic. `data.extract_version` identifies one consistent
source release, not merely the date when Python ran.

Keep history long enough for the earliest assessment and all coverage rules. The
historical index end plus outcome horizon/runout must precede the scoring date.
The scoring date must not exceed the extraction as-of date. Do not substitute the
maximum service date for analytical availability or observation completeness.

## 2. Inspect source columns and finish SQL mappings

Use the database/schema already identified in your environment:

```sh
python inspect_snowflake.py --connection-name YOUR_CONNECTION --database YOUR_DATABASE --schema YOUR_SCHEMA
```

This exports metadata only to `configs/private/discovery/source_columns.csv`.
It looks for the seven LATEST source families already discussed. Missing results
can mean a different object name or insufficient metadata visibility. The command
uses documented [Snowpark SQL parameter binding](https://docs.snowflake.com/en/developer-guide/snowpark/reference/python/latest/snowpark/api/snowflake.snowpark.Session.sql).

`sql/raw_extract/` supports already-canonical, cohort/date-filtered views.
`sql/source_extract/` supplies medical/pharmacy mapping scaffolds using known source
columns. See its README before substituting these for the private queries. Resolve
every placeholder. Demographics, enrollment, provider and plan mappings still need
exact source fields; the code does not invent them.

The seven required output files are patients, medical_claims, pharmacy_claims,
providers, plans, enrollment and therapy_mapping, each with a `.csv` extension.
See [the data contract](data_contract.md). All pharmacy records require one effective
mapping, including an explicit other/non-target class where appropriate. Do not
silently drop rejected/reversed or non-target records to make joins pass.

Filter all queries to a common approved patient population and appropriate history
and outcome dates. Include nonswitchers. Do not download the entire claims warehouse.
Reference tables must cover every non-null provider/plan key in the extract. HCP
historical features reflect the exported population unless prepared upstream from
an explicitly documented broader population.

Use consistent frozen sources or reviewed point-in-time reconstruction. Independent
reads of changing LATEST views do not guarantee a consistent extract. Preserve
identifier/NDC strings, diagnosis code-system versions, historical availability and
reversal timing. Do not invent availability from service date or observation from
first/last claims.

## 3. Check and run

Validate configuration and query placeholders without connecting or training:

```sh
python run_pipeline.py --check --extract
```

When it reports `CONFIGURATION_READY`, export and run the complete pipeline:

```sh
python run_pipeline.py --extract
```

Set `extraction.connection_name` once or pass `--connection-name YOUR_CONNECTION`.
The script opens one session, writes seven CSVs, validates the export and processes
that exact folder. The session is closed after completion or failure.

To process an existing export instead:

```sh
python run_pipeline.py --raw-dir data/raw/EXTRACT_ID --check
python run_pipeline.py --raw-dir data/raw/EXTRACT_ID
```

`--check` on files validates the manifest, hashes and raw schema; it does not establish
cohort correctness, model performance or HCP actionability. The default input path
is a placeholder location; use the exact EXTRACT_ID printed by the exporter.

The completed run prints `raw_csv_dir`, `hcp_csv`, `html_report` and `model_artifact`.
HCP CSV and offline HTML go to `outputs/client_delivery/RUN_ID/`; patient-level
intermediates and models go to `artifacts/client_delivery/RUN_ID/`. Configure approved
access permissions for these locations. Open `client_report.html` directly in a browser.

For extraction only, the existing standalone bridge remains available:

```sh
python export_raw_data.py --config configs/private/delivery.yaml --sql-dir configs/private/sql --connection-name YOUR_CONNECTION --output-dir data/raw
```

In a notebook with an already-authenticated `session`, call `export_raw_data(session,
config, sql_dir, output_dir)`, assign its returned `raw_dir` to `config["data"]["input_dir"]`,
then call `run_delivery(config)`. Load `config` with `load_delivery_config` first.
A notebook session is not inherited by a separate terminal process.

## 4. Compare models and score later extracts

The existing benchmark now accepts the same real-data configuration and CSV folder:

```sh
therapy-switch validate-data --config configs/private/benchmark.yaml --raw-dir data/raw/EXTRACT_ID
therapy-switch run --config configs/private/benchmark.yaml --raw-dir data/raw/EXTRACT_ID
```

The delivery model recipe remains replaceable in `delivery.model.recipe`; source SQL,
features and trainers keep the same interfaces. The shipped neural recipe is an
experimental starting candidate, not a demonstrated real-data winner. Train and compare
on actual mature data before selecting a final model. Model selection should preserve
an untouched test period and the agreed top-10% capacity.

For a subsequent compatible extract:

```sh
python run_pipeline.py --raw-dir data/raw/NEW_EXTRACT_ID --mode score --model-artifact artifacts/client_delivery/TRAIN_RUN_ID/model.joblib
```

Update the private extract date/version and scoring date for that extract. Real mode
rejects synthetic or legacy models without real-training provenance, as well as changed
feature/cohort/timing/source definitions. Refit when those meanings change.

## 5. Export boundaries and migration

Exports use unique folders and RUNNING/COMPLETED/FAILED manifests. Real mode verifies
completion, declared source kind, extract version/as-of date and every CSV hash.
The default ceiling is 1,000,000 rows per table; oversize results fail instead of being
silently truncated. Downloads are batched, but preparation/training remain local-memory
operations. Larger data need upstream preparation or a replacement adapter.

The older file config no longer inherits synthetic defaults. Initialize the new private
configuration and move reviewed settings into it. Existing unverified raw exports need
re-export with the real configuration to establish the manifest. A stage/GET download
alone does not establish this manifest; the SQL stage example is an alternative transport,
not a drop-in verified export. The main bridge is the supported path for this setup.

Synthetic generators and research examples remain explicit engineering utilities and
reject real-mode configuration. Run a demo only with `--config configs/delivery_demo.yaml`.
Private configurations, metadata extracts, raw CSVs and trained artifacts are ignored by Git.
Actual warehouse execution and client-model results have not been verified here.
