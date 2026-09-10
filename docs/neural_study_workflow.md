# Reproduce the neural model study

This optional study keeps the existing synthetic generator, cohort rules, temporal splits and 70 wide predictors. It searches neural models on three development cohorts, then evaluates one frozen recipe against the original LightGBM and logistic regression configurations on three independently generated cohorts. All fitting and inference are local. The default benchmark remains available separately.

## Install

~~~sh
python -m pip install -e ".[benchmark,research,dev]"
~~~

The research extra adds the official TabM, RealMLP, TabICL and TabPFN libraries. These can be substantially slower than the original baselines on CPU. The published study used CPU execution with bounded model threads. Model weights and generated data are excluded from source control.

The optional pretrained trials require these explicit local files:

- `work/model_cache/tabicl/tabicl-classifier-v2-20260212.ckpt`, from the public [TabICL model repository](https://huggingface.co/jingang/TabICL).
- `work/model_cache/tabpfn/tabpfn-v2-classifier-v2_default.ckpt`, from the public [TabPFN-v2 classifier repository](https://huggingface.co/Prior-Labs/TabPFN-v2-clf).

Obtain those exact public checkpoints through their publishers before running the full search, or use a separate candidate bank without the corresponding trials. The wrappers require explicit paths and do not automatically acquire missing checkpoints. Removing trials changes the study; document that change. No gated checkpoint, remote prediction service or authentication is needed for the documented models.

**Built with PriorLabs-TabPFN.** The optional TabPFN integration and its reported trials use TabPFN-v2 under the [Prior Labs License 1.1](licenses/TabPFN_LICENSE.txt). The selected model's identity is stated separately in the result report. An ensemble containing TabPFN uses the `tabpfn_ensemble` registry entry to retain attribution.

## Execute in order

Run from the repository root in a fresh artifact directory. Keep all three confirmation seeds reserved until the freeze step.

~~~sh
python scripts/run_neural_study.py prepare --seeds 42 43 44 7301 7302 7303
python scripts/run_neural_study.py search --seeds 42 43 44
python scripts/run_neural_study.py freeze --seeds 42 43 44 --confirmation-seeds 7301 7302 7303
python scripts/run_neural_study.py confirm
~~~

The default candidate bank is [neural_search.json](../configs/neural_search.json). It contains 55 fixed model configurations, including original and tuned classical controls. Search can resume completed trials with the same specification. Results marked FAILED stay visible; they are not silently rerun with different settings. `--root` selects a different artifact directory, `--bank` an alternative bank and `--names` a subset for development search.

The freeze step ranks neural recipes by mean validation average precision (AP). It compares single recipes, equal-weight ensembles of the top 2, 3 and 5, and 512 deterministic convex probability blends of the top 10. The blend search uses Dirichlet concentrations 0.2 and 1.0, 256 draws each, seed 20260910, and removes weights below 0.025 before renormalizing. Every ensemble uses the same members and weights across cohorts. These choices are development tuning, not independent evidence of superiority.

The frozen recipe records selected members, weights, reference configurations, reserved seeds, package source fingerprint, runtime versions, checkpoint hashes and prepared dataset manifest hashes. An existing recipe cannot be replaced through this command. Confirmation rejects changed inputs, source, dependencies, checkpoints or unbound cached models. All reserved-cohort training and validation fits finish before the first reserved test partition is opened. Reference parameters are frozen too; validation stopping remains part of the predefined fitting procedure.

## Evaluation and interpretation

The primary criterion requires mean held-out AP improvement of at least **0.02 absolute** against **both original references**, with the lower confidence bound above zero for both comparisons. Two thousand paired patient-cluster bootstrap draws preserve repeated snapshots within each cohort. Cohorts have equal weight; the two primary comparisons use Bonferroni-adjusted 97.5% intervals, targeting family alpha 0.05. The 0.02 threshold applies to the point estimate, not the lower confidence bound.

The intervals condition on the fitted models and these three synthetic cohorts. They do not quantify uncertainty from retraining, establish performance across all possible cohorts, or imply clinical benefit. Top-10% capture, cohort consistency and comparison against validation-selected tuned classical controls are secondary analyses. Original seed-42 test results, if reported later, are a legacy diagnostic because that test had already been examined before this search.

Do not alter the recipe or restart selection based on confirmation outcomes. A failed gate is a valid study result. A subsequent search needs a new, prospectively reserved confirmation design and must disclose earlier test access.

## Model implementations

- **TabM:** official parameter-efficient MLP ensemble; member-level binary loss before averaging probabilities, validation-AP stopping, optional piecewise-linear numerical embeddings. Numerical bins, imputation, feature selection and normalization fit training rows only. [Official implementation](https://github.com/yandex-research/tabm).
- **RealMLP:** official `RealMLP_TD_Classifier`, one training/validation fit without refitting on validation, cross-entropy stopping. Outer configuration selection uses AP. [Official implementation](https://github.com/dholzmueller/pytabkit), [paper](https://arxiv.org/abs/2407.04491).
- **TabICL-v2 / TabPFN-v2:** local pretrained in-context classifiers; training labels form the context. The common wrapper fits imputation, encoding and optional feature selection on training rows only. [TabICL](https://github.com/soda-inria/tabicl), [TabPFN](https://github.com/PriorLabs/TabPFN).
- **Residual MLP:** a frozen training-fitted logistic skip plus a nonlinear correction. Epoch zero is an eligible validation checkpoint; choosing it means the candidate reduces to its linear baseline and must not be described as a nonlinear improvement.
- **Neural ensemble:** a fixed nonnegative average of configured neural probability outputs. No LightGBM predictions are blended into this candidate. The existing pipeline handles threshold selection, calibration, explanations, held-out scores, HCP outputs and joblib reload.

The study broadens tabular neural search. It does not exhaust architecture space or retune the original sequence models. Improved access to longitudinal signal, larger training populations and external validation remain separate research questions.

## Local evidence

Under `artifacts/dl_search`, development trials retain parameters, validation AP, predictions and fitted estimators. `frozen_recipe.json` and its SHA-256 companion define the confirmation recipe. `confirmation/confirmation_report.json` contains the primary and secondary results; `capacity.csv` contains aggregate capacity metrics. Row-level scores, model files and input manifests remain ignored artifacts. Only reviewed synthetic aggregates belong in published reports.

Some early exploratory trial records predate the final provenance guards. They retain exact candidate specifications, saved validation predictions and fitted estimators; they do not all carry immutable source hashes. The final confirmation run uses the complete frozen guards.
