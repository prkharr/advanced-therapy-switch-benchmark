# Synthetic aggregate evidence

Entirely synthetic default-configuration experiment, seed 42. No row-level identifiers or claims are included.

model_benchmark.csv reports all ten completed runs. paired_summary.csv gives AP and top-10% capture differences against LightGBM. run_summary.json records population/splits, model selection, calibration, explanation status, test count and runtime/source hashes. Five optional model entries were disabled.

Training and explanation postprocessing hashes are recorded separately; the saved model predictions were verified unchanged during explanation generation. These results do not reproduce a real client baseline or establish Sentinel compatibility.
