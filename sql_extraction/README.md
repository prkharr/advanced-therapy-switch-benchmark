# SQL extraction

This folder is separate from `actual_raw_data/`, which receives downloaded results.

The seven root SELECT templates define the exact canonical output columns. Replace each placeholder with reviewed raw-source logic or an independently constructed canonical view. No existing model, score, feature or cohort table should be used. Build the population filter from raw diagnosis/treatment records using the reviewed code lists, and retain all relevant medical and pharmacy history for that population; do not extract only the qualifying diagnosis or therapy rows.

`source_mapping/` contains medical and pharmacy scaffolds showing how raw events can map to the contract. Field names, diagnosis encoding, claim grain, status expressions, availability timestamps and dimensions must be verified from metadata and saved aggregate query results. Populate `INDEPENDENT_PATIENT_SCOPE_VIEW` from the new raw-source rules. Do not concatenate dated snapshots unless their disjointness or deduplication is established. Pin consistent source versions for all extracts and retain point-in-time availability.

For manual extraction:

`source_mapping/independent_patient_scope.sql` defines a broad raw-diagnosis population filter. Review and use it as a CTE or a separately materialized view in the final extracts. Python subsequently applies the full index-date eligibility rules; the extraction scope is not a supplied modeling cohort.

1. Run the metadata section in `evidence_checks.sql` with the actual database/schema. Save its output privately.
2. Resolve the seven queries and review the source definitions and extraction window. The window must cover carry-in treatment, feature history and mature outcome follow-up.
3. Run each final SELECT in Snowflake. Download results as CSV with headers into `actual_raw_data/`, using the SQL filename stem. Check the UI export row limit and reconcile row counts; use the Python exporter or an approved stage download for larger extracts.
4. Run `python run_pipeline.py --check`, then `python run_pipeline.py` from the repository root.

For automatic extraction, `extraction.sql_dir` points here by default. Keep environment-specific SQL in the ignored `sql_extraction/private/` folder and set that path in the private delivery config. It must contain exactly the seven named dataset SQL files. Install the Snowflake extra, configure a named local connection, then run with `--extract`. The exporter executes read-only SELECT/WITH statements, rejects unresolved placeholders and oversized results, validates all seven tables and creates a hash manifest. Do not commit downloaded data or credentials.

The evidence queries are templates; their presence is not evidence that a warehouse check has passed. Record query text, query ID, execution time, source version and aggregate result in the approved environment. Resolve code lists and clinical definitions before creating the cohort.
