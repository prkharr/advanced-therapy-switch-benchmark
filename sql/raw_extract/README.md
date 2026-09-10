# Seven extraction queries

Each filename matches a canonical raw CSV consumed by the pipeline. These are templates, not completed source mappings. Replace the view placeholder, or supply a reviewed SELECT that maps the actual source fields to the listed aliases. The exporter rejects unresolved placeholders.

Source view names alone do not establish claim availability, status semantics, diagnosis splitting, population boundaries or treatment class. Follow [the raw-folder workflow](../../docs/raw_folder_workflow.md) and [data contract](../../docs/data_contract.md). Preserve all needed references and historical data across the same frozen extract.

SELECT queries return data; the Python exporter writes the local raw folder. The separate stage-unload example is for supported warehouse/file-transfer clients.
