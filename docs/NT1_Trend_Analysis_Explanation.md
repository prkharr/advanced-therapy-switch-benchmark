# NT1 model improvement: explanation of the numbers

The accompanying workbook summarizes the supplied NT1 analysis and proposes features for validation. It predicts **first advanced-therapy initiation within 90 days among eligible NT1 patients using generic therapy**. It does not predict TAK-861 prescribing, and it contains no measured model-performance improvement.

## Sources and scope

- All cohort counts, RESP aggregates and descriptive signal ratings come from the supplied NT1 brief dated September 11, 2026. They were not recomputed from warehouse data.
- `docs/Features_and_analysis.xlsx`, **Features** sheet, supplies the additional repository benchmark inventory. Its source scope is identified separately in the workbook.
- Diagnosis/procedure raw-code and specialty trend exports were not found. Their detail tables are intentionally unpopulated. No code descriptions, specialty percentages or missing cohort-stage counts have been invented.
- The repository inventory describes benchmark features using conventional-therapy and assessment-date definitions. These are not automatically equivalent to the brief's `IS_GENERIC` and `EFFECTIVE_DATA_END` definitions. Reconcile those definitions before feature reuse.

## How to read RESP, snapshots and patients

**RESP 1** identifies a snapshot followed by the patient's first advanced therapy during the specified 90-day outcome window. **RESP 0** requires no advanced therapy in that window and the required future medical observation.

A **snapshot** is one patient assessed at a month-end index date. A **patient count** counts distinct people within that particular analysis. For therapy sequence, RESP 0 has **1,282,233 snapshots from 42,787 patients**, whereas RESP 1 has **9,040 snapshots from 9,040 patients**. Keeping only the first positive snapshot per patient explains why the positive counts match. Multiple negative snapshots can belong to one person; these are not independent people.

The supplied domain counts differ:

| Analysis | RESP 0 snapshots | RESP 0 patients | RESP 1 snapshots / patients |
|---|---:|---:|---:|
| Therapy sequence | 1,282,233 | 42,787 | 9,040 |
| Medical utilization | 1,260,959 | 42,500 | 8,747 |
| Provider | 1,224,415 | 42,807 | 8,489 |
| Plan/access | 1,266,881 | 42,635 | 8,980 |

These source values are preserved. Their coverage and join differences have not been reconciled. Do not add patient counts across domains or treat them as a single final cohort. No overall patient event rate is inferred from these counts.

## Timing and eligibility

The supplied cutoff is `EFFECTIVE_DATA_END = INDEX_DATE − 40 days`. All features use data at or before this cutoff. The generic eligibility lookback is 270 days, with at least 135 covered days and maximum gap no more than 270 days. At least two qualifying NT1 claim dates must be at least 90 days apart. Prior advanced therapy at the cutoff excludes the snapshot.

The outcome window is cutoff +1 through cutoff +90 inclusive. With the 40-day index lag, it starts at **INDEX_DATE −39 days** and ends at **INDEX_DATE +50 days**. Preserve this supplied definition and confirm its alignment with the intended operational scoring date before modeling. The reported latest advanced fill date does not establish complete follow-up for every snapshot.

## Cohort numbers

- **1,102,929 NT1 claim rows** represent recorded events; **176,552 patients** represent distinct people with NT1 claims.
- **85,585 confirmed NT1 patients** satisfy the confirmation rule.
- **34,791 confirmed NT1 patients ever observed on advanced therapy** are a historical subset, not the final eligible advanced-naïve cohort and not the positive training count.
- Observed advanced fill dates range from **2015-01-01** through **2026-06-05**.
- The market basket contains **5,166,799,412 event rows** for **143,445,550 patients**. Its **4,842,227,660 generic rows** plus **324,571,752 advanced rows** reconcile exactly to the total.

The market basket is broader than the NT1 modeling population. Generic and advanced rows are event counts, not numbers of people. Missing eligibility, exclusion and final-population counts cannot be inferred by subtracting the historical advanced subset. The cohort chart therefore shows only the two available sequential patient stages, using a bar chart supported by Excel export.

## Therapy sequence: strongest descriptive separation

The workbook's difference column always means **RESP 1 minus RESP 0**.

| Measure | RESP 0 | RESP 1 | Difference | Reading |
|---|---:|---:|---:|---|
| Average distinct generics, 270D | 1.85 | 2.23 | +0.38 | Greater observed treatment diversity |
| Average therapy changes, 270D | 4.38 | 5.60 | +1.22 | More changes in the longer window |
| Average therapy changes, 90D | 1.55 | 2.14 | +0.59 | More recent changes |
| Average therapy changes, 30D | 0.51 | 0.75 | +0.24 | More changes close to cutoff |
| Any change in last 30D | 31.3% | 43.2% | +11.9 pp | Higher share with a recent change |
| Any change in last 90D | 47.4% | 60.6% | +13.2 pp | Higher share with a recent change |
| Median days since last generic fill | 18 | 15 | −3 days | More recent fill activity |
| Average covered days, 270D | 239.9 | 234.4 | −5.5 days | Slightly lower mean coverage |
| Median maximum gap | 8 | 10 | +2 days | Slightly larger median maximum gap |

**Percentage points are absolute differences between percentages.** Here 43.2% − 31.3% = 11.9 percentage points; it is not an 11.9% relative increase. In Excel, percentages are stored as fractions (0.432 and 0.313), so the pp formula is `(RESP1 − RESP0) × 100`.

Average generic fill-date counts are also higher in RESP 1: **11.85 vs 10.52** over 270 days, **4.12 vs 3.46** over 90 days and **1.39 vs 1.14** over 30 days. Counts of fill dates do not establish medication consumption. The proposed features capture diversity, changes and recency; validate same-day ordering, change semantics and lookback boundaries before implementation.

## Medical utilization: limited separation

Average medical event counts are **79.13 vs 72.54** over 270 days, **26.63 vs 24.40** over 90 days and **8.91 vs 8.23** over 30 days (RESP 0 vs RESP 1). Active-day means are **22.49 vs 21.30**, **7.57 vs 7.29** and **2.53 vs 2.50**, respectively. Median recency is **13 vs 12 days**.

The supplied recent 30-day activity ratio is **1.01 vs 1.02**. Its exact denominator was not provided, so the workbook preserves the ratio without reconstructing it. These aggregate comparisons provide little evidence of a broad utilization surge. They do not prove that every patient's utilization remained stable. Specific types of clinical activity remain candidates for testing.

## Diagnosis/procedure and specialty: reported opportunities

The brief reports promising diagnosis/procedure separation and higher recent Psychiatry exposure in RESP 1. Without the exports, code-level rankings, effect sizes and specialty percentages cannot be verified. On receiving the diagnosis/procedure export, sort by `DIFF_90D_PP` descending, then `DIFF_30D_PP` descending. Sort specialty exposure by `DIFF_90D_PP` descending.

Candidate features include presence, frequency and recency of selected DX/PX codes and specialty exposure. Select codes and specialties using development data; evaluate the selected features on reserved future periods.

## Provider count and plan dynamics: lower priority

Average unique providers are **8.35 vs 8.25** over 270 days, **3.64 vs 3.59** over 90 days and **1.54 vs 1.63** over 30 days. Median provider recency is approximately **19 vs 16 days**. These supplied approximate values remain marked approximate.

Average distinct plans are **2.68 vs 2.67**, **1.70 vs 1.71** and **1.16 vs 1.18** over 270, 90 and 30 days. Mean plan changes are **5.23 vs 4.98**, **1.81 vs 1.78** and **0.60 vs 0.60**. The 90-day plan-change shares are **43.0% vs 43.2%** (+0.2 pp), and 30-day shares are **27.1% vs 26.8%** (−0.3 pp). Median recency of plan-linked RX is **8 days in both groups**. These values show little descriptive separation; no payer identity or access mechanism is inferred.

## What the findings do and do not establish

**Descriptive trend** means an observed difference in supplied aggregates. **Candidate feature** means a proposed predictor to test. **Validated model lift** requires incremental out-of-sample performance relative to a fixed baseline on a comparable evaluation population. The workbook contains the first two, with missing exports explicitly identified, and no evidence of the third.

Begin with the baseline LightGBM and compare cumulative additions of therapy sequence, diagnosis/procedure, specialty exposure and plan/access features. Use out-of-time evaluation, a defined patient-overlap policy, mature outcomes and the same evaluation cohort. Report average precision or a clearly defined PR-AUC method, precision/recall/lift at fixed ranking capacity, calibration and uncertainty. SHAP supports model interpretation, not causal conclusions. Benchmark MLP, GRU and Transformer only after testing incremental engineered-feature value.

The workbook's Next Steps statuses remain blank because no action-completion statuses were supplied. The proposed feature work has not been represented as implemented model improvement.
