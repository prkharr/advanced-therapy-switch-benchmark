# Snowflake to raw folder to HCP delivery

The pipeline consumes seven canonical CSVs from one raw-data folder. Extraction
is separate from model training, so source SQL can change while the remaining
workflow stays the same. The model can change through the existing recipe/trainer.

```text
Reviewed, consistently versioned Snowflake source queries
    -> export_raw_data.py
    -> data/raw/EXTRACT_ID/{seven CSVs + export_manifest.json}
    -> run_pipeline.py --raw-dir data/raw/EXTRACT_ID
    -> HCP CSV + offline HTML report + model and audit artifacts
```

## 1. Establish source mappings

Use `sql/raw_extract/` as templates. Copy the seven SQL files into an ignored
directory such as `configs/private/sql/`. Replace the `{{...}}` placeholders with
reviewed canonical views, or replace each SELECT with your reviewed source query
that returns the same canonical columns. Templates intentionally do not guess
claim-availability timestamps, treatment classes, diagnosis codes or statuses.

The views/queries must already restrict patients and dates consistently. Do not
export an entire claims warehouse. Extract a common cohort across all seven
tables, preserve all required patient history, and retain the provider/plan
references and effective therapy mappings used by the claims. Historical HCP
features are measured within the exported population unless separately prepared
from an approved broader population; document that scope.

Use one frozen extract/version or equivalent time-consistent reads. Running seven
queries on changing LATEST views does not itself establish a consistent snapshot.
Training extracts need outcomes through the approved horizon/runout and sufficient
observation/enrollment. Do not prefilter outcomes to only successful switchers.

Medical events can contain several diagnosis codes in a single source field.
The canonical medical adapter expects one diagnosis code per record. Review the
delimiter and event grain before splitting: assign unique line/code keys, avoid
duplicating the same procedure across exploded diagnosis rows, and understand
that diagnosis-count-based visit features are proxies rather than encounter counts.
An existing prepared-input adapter may be preferable for a richer event schema.

Keep IDs/NDCs as strings. Preserve leading zeros, code-system/version metadata
upstream, effective mappings, paid/rejected/reversed semantics and actual
analytical availability. Do not set available_date equal to service date as a
substitute for missing history. See `docs/data_contract.md` and the workbook.

## 2. Prepare the delivery configuration

Copy `configs/delivery_files.yaml` to `configs/private/delivery.yaml`. Because
paths resolve relative to the YAML, change `benchmark_config` to `../default.yaml`,
the model recipe to `../field_model.json`, and output/artifact directories to
`../../outputs/client_delivery` and `../../artifacts/client_delivery`.

Set the actual extraction as-of date, scoring date, historical index calendar,
reviewed diagnosis/product definitions, lag/runout, coverage rules and specialties
through `overrides`. Set `delivery.data_scope` to the actual source description.
The shipped dates and SYN codes are demonstration assumptions and must be replaced.
The maximum service date is not the extraction as-of date or proof of claim latency.

## 3. Export using an authenticated session

Install in your approved Python environment:

```sh
python -m pip install -e ".[snowflake,deep-learning]"
```

In the same Python process as an existing authenticated Snowpark session:

```python
from therapy_switch.delivery.pipeline import load_delivery_config, run_delivery
from therapy_switch.delivery.raw_export import export_raw_data

config = load_delivery_config("configs/private/delivery.yaml")
export = export_raw_data(session, config, "configs/private/sql", "data/raw")
config["data"].update(source="files", input_dir=export["raw_dir"], file_format="csv")
result = run_delivery(config, mode="train-score")
print(result["hcp_csv"])
print(result["html_report"])
```

Alternatively, if your environment has an approved named Snowflake connection:

```sh
python export_raw_data.py --config configs/private/delivery.yaml --sql-dir configs/private/sql --connection-name YOUR_APPROVED_CONNECTION --output-dir data/raw
python run_pipeline.py --config configs/private/delivery.yaml --raw-dir data/raw/EXTRACT_ID
```

Replace EXTRACT_ID with the completed export directory printed by the exporter.
The terminal does not inherit an authenticated session from a separate notebook.
The exporter uses Snowflake's documented named-connection and batch-download APIs:
[creating sessions](https://docs.snowflake.com/en/developer-guide/snowpark/python/creating-session),
[batch downloads](https://docs.snowflake.com/en/developer-guide/snowpark/reference/python/latest/snowpark/api/snowflake.snowpark.DataFrame.to_pandas_batches).

SQL SELECT statements return data. They cannot directly write to an arbitrary
folder on your computer. This Python bridge executes the seven SELECTs and writes
the local CSVs. A SQL-only warehouse unload writes to a stage; downloading from
that stage requires a supported client. A separate example is supplied at
`sql/snowflake_stage_export.sql`.

## 4. What export validation means

Each export has a unique folder and a RUNNING, COMPLETED or FAILED manifest.
Columns are normalized to lowercase; files contain one header and no dataframe
index. Per-table row counts, file hashes and SQL hashes are recorded. All seven
files pass the canonical data checks before export completion. The pipeline
rejects an export whose manifest is not COMPLETED.

The default ceiling is 1,000,000 rows per table. An extra-row probe rejects larger
results instead of silently training on a truncated sample. Downloads are batched,
but final validation, feature preparation and model fitting are local-memory
operations. Increase the ceiling only after sizing the environment; for larger
data prepare upstream and use the prepared/custom adapter contract.

Structural validation cannot prove that the cohort, feature timing, product
mapping or clinical definitions are correct. Those require source-logic review.
The actual source session and real-data model performance remain unverified.

## 5. Reuse the fitted model

After training on actual historical data, score a compatible later completed extract:

```sh
python run_pipeline.py --config configs/private/delivery.yaml --raw-dir data/raw/NEW_EXTRACT_ID --mode score --model-artifact artifacts/client_delivery/TRAIN_RUN_ID/model.joblib
```

Do not use a synthetic-trained artifact for client targeting. Change the feature
contract version and refit when feature meaning or cohort rules change. Raw data,
populated connection configuration and fitted artifacts remain ignored by Git.
