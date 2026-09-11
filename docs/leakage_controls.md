# Temporal and evaluation controls

- Cohort and features are rebuilt from raw inputs using explicit diagnosis, therapy and timing rules.
- Service-date cutoffs and first-availability dates restrict predictors to information known at each assessment. Future outcomes are used only for labels.
- Negative labels require observable, mature follow-up. Incomplete future observation is not coded as a negative.
- Static dimensions need point-in-time source snapshots or upstream effective-date reconstruction. Current attributes alone cannot prove historical availability.
- Temporal splitting assigns patients to disjoint train, validation and test partitions. Label maturity and purge checks apply before fitting. Review resulting class and patient counts.
- Imputation, categorical encoding and scaling are fitted on training data only. Both fresh baselines use the same partitions and features.
- Validation recall at the top 10% of distinct eligible patients selects the baseline; average precision breaks ties. The test set is evaluated after selection and must not be reused to tune the next model.
- The latest assessment per patient is used for patient-capacity evaluation. HCP outputs aggregate selected patients after scoring.
- Model reload, feature-schema, source-policy and training-label-date checks guard subsequent scoring.

`tests/test_raw_workflow.py` and `tests/test_patient_lists.py` verify specified software behaviors on small hand-written test cases. They do not establish warehouse timestamp accuracy, historical correction recovery, population completeness or clinical target validity. Actual-data evidence is produced by `raw_profile.json`, `training_audit.json` and `baseline_comparison.csv` when those stages run successfully.
