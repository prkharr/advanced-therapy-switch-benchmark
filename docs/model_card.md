# Model card

**Status:** offline synthetic engineering benchmark. Real client results have not been run. The original configuration and findings are in the project report; the subsequent [neural study](neural_model_study.md) reports development search and reserved-cohort confirmation.

**Intended use:** rank eligible patient snapshots by association with future advanced-therapy initiation and aggregate held-out scores into HCP opportunity estimates. Clinical recommendation, causal interpretation and automated treatment decisions are outside scope.

**Target:** any approved advanced-therapy exposure in the future prediction window; program context does not make this a TAK861-specific outcome.

**Training population:** artificial longitudinal patients with synthetic identifiers and dictionaries. Default generation uses 3,000 patients, three candidate monthly snapshots, noisy risk and imbalanced outcomes. Real disease codes and product dictionaries are absent.

**Inputs:** engineered pre-index wide features and/or chronological event embeddings. Age is index year minus birth year, not exact birthday age. Dimensions are static within an extract; real temporal dimensions require upstream validation.

**Evaluation:** strict patient-disjoint temporal partitions, mature labels, train-only transforms, validation-selected model/threshold/calibration and held-out same-capacity comparisons. Patient-cluster bootstrap handles repeated snapshots within its documented assumptions.

**Explanations:** tabular SHAP where compatible, otherwise native/permutation importance; sequence code-occlusion sensitivity. Neither attention nor predictive importance establishes causality.

**Limitations:** simulated pathways, simple supply union without stockpiling, fixed-runout labels, incomplete representation of adjudication/cash/specialty channels, short sequence truncation, overlapping horizons and static extract dimensions. The original report uses one main synthetic seed. The neural study adds three development and three reserved cohorts from the same generator, with overlap at the original development seed; its intervals condition on fitted models. Recurring-patient prospective evaluation, retraining uncertainty, subgroup stability, clinical mapping review and external baseline validation remain outstanding.

**Integration:** generic file/prepared/Snowpark adapters are implemented. Snowpark is tested with mocks and is NOT VERIFIED IN SENTINEL. No production infrastructure or serving system is included.

**Artifacts:** generated datasets, row-level outputs, fitted models and manifests are local ignored artifacts. Source control contains reusable code, sanitized configurations, tests, documentation and a small aggregate synthetic result snapshot.
