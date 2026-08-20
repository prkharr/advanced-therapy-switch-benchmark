# Model card template

Complete this document for the selected production candidate. Synthetic
quickstart results are engineering evidence only and must not populate a real
model approval package.

## Intended use

Rank eligible patients by association with an advanced-therapy initiation during
the configured future window, then aggregate those probabilities into a separate,
transparent HCP opportunity workflow for commercial analytics.

## Prohibited use

- Treatment selection, clinical decision support, diagnosis, or medical advice.
- Causal claims about any feature, provider, therapy, or patient outcome.
- Use beyond approved populations, geographies, fields, and privacy controls.
- Automated adverse decisions about patients or providers.

## Training and evaluation data

Record delivery/version, observation and prediction windows, inclusion/exclusion
criteria, claim run-out, code mappings, population counts, prevalence, temporal
boundaries, and permitted demographics. Do not include row-level identifiers in
this document.

## Model and selection evidence

Record the validation-selected candidate, preprocessing, class weighting/loss,
hyperparameters, threshold, calibration method, implementation version, training
time, inference throughput, and artifact hash.

Attach random and out-of-time results for PR-AUC, ROC-AUC, Recall/Lift at field
capacity, deciles, calibration, and 95% confidence intervals. Include the paired
classical-versus-longitudinal comparison and explain why any added complexity is
materially justified.

## Limitations and risks

Claims measure billing activity rather than complete clinical state. Missing
coverage, channel gaps, delayed adjudication, coding changes, access constraints,
and unobserved cash/assistance fills may alter both inputs and labels. Explainability
describes predictive association and may reflect healthcare access or documentation
patterns. Review subgroup stability and permitted-use policy before launch.

## Monitoring and ownership

Name the model owner, data owner, business owner, privacy reviewer, validation
owner, approval date, retraining cadence, performance/drift thresholds, rollback
procedure, and incident contact. Record every HCP attribution rule and opportunity
score weight separately from the patient model.
