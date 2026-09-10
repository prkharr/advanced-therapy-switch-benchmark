"""Publish aggregate synthetic patient-capacity evidence; exclude row-level data."""

import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

root = Path("artifacts/patient_search")
destination = Path("docs/patient_study")
destination.mkdir(parents=True, exist_ok=True)
recipe = json.loads((root / "frozen_recipe.json").read_text())
report = json.loads((root / "confirmation/report.json").read_text())
metrics = pd.read_csv(root / "confirmation/metrics.csv")
development = pd.read_csv(root / "development_results.csv")
for source in [
    root / "frozen_recipe.json",
    root / "frozen_recipe.sha256.json",
    root / "development_results.csv",
    root / "confirmation/report.json",
    root / "confirmation/metrics.csv",
    root / "confirmation/inference_times.csv",
]:
    shutil.copyfile(source, destination / source.name)
summary = (
    metrics.loc[np.isclose(metrics.capacity, 0.1)]
    .groupby("model")
    .agg(
        mean_recall=("recall", "mean"),
        mean_precision=("precision", "mean"),
        mean_lift=("lift", "mean"),
        mean_ap=("ap", "mean"),
        captured=("captured", "sum"),
        patients=("patients", "sum"),
        positives=("positives", "sum"),
        selected=("selected", "sum"),
    )
    .sort_values("mean_recall", ascending=False)
)
summary.to_csv(destination / "summary.csv")
dev = (
    development.groupby(["name", "family", "history", "latest_only"], dropna=False)
    .agg(mean_recall=("recall", "mean"), mean_ap=("ap", "mean"), cohorts=("seed", "nunique"))
    .reset_index()
    .sort_values(["mean_recall", "mean_ap"], ascending=False)
)
dev.to_csv(destination / "development_summary.csv", index=False)
names = {
    "candidate": "Selected patient-list ensemble",
    "original_lightgbm": "Original LightGBM",
    "original_logistic": "Original logistic regression",
    "previous_neural": "Previous neural ensemble",
    "tuned_classical": "Development-selected classical control",
    "best_ehr": "Development-selected EHR control",
}
table = summary.reset_index()
table["model"] = table.model.map(names)
for key in ["mean_recall", "mean_precision"]:
    table[key] = table[key].map(lambda v: f"{v:.2%}")
for key in ["mean_lift", "mean_ap"]:
    table[key] = table[key].map(lambda v: f"{v:.4f}")
comparison = pd.DataFrame(report["comparisons"])
ct = comparison[["reference", "mean_recall_gain", "ci_lower", "ci_upper", "passes"]].copy()
ct.reference = ct.reference.map(names)
for col in ["mean_recall_gain", "ci_lower", "ci_upper"]:
    ct[col] = ct[col].map(lambda v: f"{v * 100:+.2f} pp")
members = []
for spec, weight in zip(recipe["selected"]["members"], recipe["selected"]["weights"]):
    members.append(
        {
            "model": spec["name"],
            "family": spec["family"],
            "weight": round(weight, 6),
            "added_history": spec.get("history", False),
            "latest_training_only": spec.get("latest_only", False),
        }
    )
member_table = pd.DataFrame(members)
cohort = metrics.loc[np.isclose(metrics.capacity, 0.1)].pivot(
    index="cohort", columns="model", values="captured"
)
selected = metrics.loc[
    (metrics.model == "candidate") & np.isclose(metrics.capacity, 0.1)
].set_index("cohort")
cohort.insert(0, "eligible_patients", selected.patients)
cohort.insert(1, "list_size", selected.selected)
cohort.insert(2, "future_switchers", selected.positives)
fits = []
for seed in recipe["protocol"]["fresh_confirmation_seeds"]:
    record = json.loads((root / f"confirmation/{seed}/fits.json").read_text())
    for name, fit in record["fits"].items():
        fits.append(
            {
                "cohort": seed,
                "name": name,
                "seconds": fit["seconds"],
                "reload_verified": fit["reload_verified"],
                "best_epoch": fit.get("best_epoch"),
                "best_iteration": fit.get("best_iteration"),
            }
        )
pd.DataFrame(fits).to_csv(destination / "fit_audit.csv", index=False)
candidate = summary.loc["candidate"]
if report["passes"]:
    decision = "The frozen patient-list candidate passed the prespecified synthetic confirmation gate against all three primary controls."
else:
    decision = "The frozen patient-list candidate did not pass the complete prespecified synthetic confirmation gate. The results do not justify claiming a confirmed improvement over all three primary controls."
ehr = dev[dev.family.isin(["ehr_transformer", "retain"])]
text = f"""# Patient capture study — synthetic evidence

{decision}

The selected candidate captured **{int(candidate.captured)} of {int(candidate.positives)} future switchers** across five fresh cohorts, using **{int(candidate.selected)} list places among {int(candidate.patients)} distinct eligible historical patients**. Its equal-cohort mean recall was **{candidate.mean_recall:.2%}**, mean precision **{candidate.mean_precision:.2%}**, and mean lift **{candidate.mean_lift:.2f}**. These are historical synthetic assessments, not a validated current patient list or evidence of performance on real claims.

## What changed from the previous benchmark

The primary objective is now capture of future switchers in the top 10% of distinct patients. Each patient contributes their latest supplied eligible assessment, chosen using dates only. The earlier study evaluated repeated snapshot opportunities and selected on AP; its headline numbers cannot be directly compared with these patient-level numbers.

The unchanged generator and cohort rules were retained. No latent generator variables, future outcomes or test-selected feature engineering were used. Code-history features, latest-assessment training, ranking losses, additional boosting models, EHR models and cross-family blends were compared on development cohorts.

The previous AP study is preserved at source revision 7d3839621e453c13f7613b6a4e26521e56e723e1. Its already evaluated cohorts 7301–7303 were not reused as fresh confirmation here.

## Reserved test results

Each model is evaluated at exactly the same patient capacity within a cohort. Recall, precision, lift and AP below are equal-cohort averages. Captured counts and list sizes are totals; pooled recall can differ slightly from mean cohort recall.

{table.to_markdown(index=False)}

The top-5% and top-20% sensitivity results are available in [metrics.csv](patient_study/metrics.csv). They are secondary analyses and did not determine model selection.

## Prespecified statistical comparisons

The primary gate requires a point estimate of at least +3 percentage points in mean recall and a lower confidence limit above zero against each original reference and the previous frozen neural ensemble. The 3,000 paired patient bootstrap draws rebuild the top-10% list for every model in every draw. Three Bonferroni-adjusted comparisons use 98.333% confidence intervals at family alpha 0.05.

{ct.to_markdown(index=False)}

![Mean recall gains and adjusted confidence intervals](patient_study/capture_gain.svg)

The gate result is **{"PASS" if report["passes"] else "NOT PASSED"}**. The intervals condition on these fitted models and synthetic cohorts. They do not establish retraining stability, temporal transportability, causal treatment benefit or real-world superiority. The fixed tuned-classical and best-EHR controls have separately adjusted secondary comparisons in [report.json](patient_study/report.json).

## Captured patients by cohort

Every cell below is the number of labelled future switchers captured, except the three denominator columns.

{cohort.to_markdown()}

The complete cohort results are disclosed, including cohorts where the selected candidate trails a reference.

## Frozen candidate

{member_table.to_markdown(index=False)}

The candidate was chosen from **{recipe["candidate_count"]} configurations** and **{recipe["proposal_count"]} fixed-member/weight proposals** using only development seeds 42, 43 and 44. Its mean development recall was {recipe["selected"]["mean_recall"]:.2%}, with mean development patient AP {recipe["selected"]["mean_ap"]:.4f}. Selection optimism is expected; only the reserved results above serve as confirmation.

There were 294 completed configuration/cohort evaluations: 165 exact prior development predictions were reused and 129 new models were fitted. All selected candidate and control models were then refitted on fresh training/validation partitions. There were {len(fits)} distinct reserved-cohort fits, with {sum(f["reload_verified"] for f in fits)} successful serialization/reload checks. All required fits completed before the first reserved test read.

Recipe SHA-256: {report["recipe_sha256"]}

The [frozen recipe](patient_study/frozen_recipe.json) records members, weights, source fingerprint, package versions, checkpoint hashes, input manifests and the original protocol. [Development results](patient_study/development_results.csv) and [development summary](patient_study/development_summary.csv) expose every candidate rather than only the winner. Training durations are per-fit wall time under concurrent CPU work, not sequential total runtime; see [fit audit](patient_study/fit_audit.csv) and [inference times](patient_study/inference_times.csv).

## Med-BERT, BEHRT and longitudinal alternatives

The tested transformer is a compact local EHR architecture with event/code, position, alternating-visit and elapsed-time inputs. It has scratch and masked-event-pretrained variants, with and without a wide-feature branch. Pretraining uses only training-patient histories. The reverse-time attention alternative is RETAIN-inspired. These are explicitly local implementations, not original medical checkpoints.

{ehr[["name", "mean_recall", "mean_ap", "cohorts"]].to_markdown(index=False, floatfmt=".4f")}

These development comparisons do not prove that full-scale medical pretraining would be ineffective on real claims. The synthetic code vocabulary is small and is not mapped to ICD concepts. The original [Med-BERT repository](https://github.com/ZhiGroup/Med-BERT#sharing-pre-trained-model) states that its pretrained weights can no longer be shared. No Med-BERT checkpoint was used. Architecture references and the exact training procedure are documented in the [workflow](patient_study_workflow.md).

## Use and limitations

The repository now supports generating a deduplicated top-10% list, fitting the fixed selected recipe on local train/validation partitions, and running the full search/freeze/confirm procedure. See the [patient-list workflow](patient_study_workflow.md) for commands and required inputs.

Operational eligibility must be established upstream at the intended scoring date. The synthetic cohort uses historical conventional-treatment coverage and is not equivalent to proof that every patient remains on generic therapy today. An as-of cutoff cannot repair stale eligibility.

Scores are ranking signals. Do not interpret uncalibrated class-weighted outputs or LambdaRank transformations as individual switch probabilities, clinical recommendations, or expected switcher counts. Probability calibration and the HCP output layer require their own appropriate evaluation.

Results apply to this unchanged synthetic generator and its assumptions. Clinical coding mappings, label observability, claims availability, selection bias, subgroup performance and real temporal holdout validation remain necessary before operational use. The data, row-level lists, fitted models and private environment material are excluded from Git.
"""
Path("docs/patient_capture_study.md").write_text(text, encoding="utf-8")
# Standard scientific plot: exact observed comparison differences and intervals.
fig, ax = plt.subplots(figsize=(9, 3.6), layout="constrained")
for i, row in enumerate(report["comparisons"]):
    center = row["mean_recall_gain"] * 100
    ax.hlines(i, row["ci_lower"] * 100, row["ci_upper"] * 100, color="#136f63")
    ax.plot(
        [row["ci_lower"] * 100, row["ci_upper"] * 100], [i, i], "|", color="#136f63", markersize=9
    )
    ax.plot(center, i, "o", color="#136f63", markersize=7)
ax.axvline(0, color="#555555", linewidth=1)
ax.axvline(
    3, color="#b27414", linestyle="--", linewidth=1, label="Prespecified +3 pp point-gain threshold"
)
ax.set_yticks(
    range(len(report["comparisons"])), [names[r["reference"]] for r in report["comparisons"]]
)
ax.invert_yaxis()
ax.set_xlabel("Mean recall gain at top 10% of distinct patients (percentage points)")
ax.set_title("Fresh synthetic confirmation — 98.333% paired patient intervals", loc="left")
ax.grid(axis="x", alpha=0.2)
ax.legend(loc="best", fontsize=8)
preview_directory = Path("outputs/patient_capture")
preview_directory.mkdir(parents=True, exist_ok=True)
fig.savefig(preview_directory / "capture_gain.png", dpi=180)
fig.savefig(destination / "capture_gain.svg")
plt.close(fig)
svg = destination / "capture_gain.svg"
svg.write_text(
    "\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n", encoding="utf-8"
)
print(summary.to_string())
print("Primary gate", report["passes"])
