# Mapping the supplied event views

These two scaffolds use the previously supplied medical/pharmacy field names.
They are intentionally incomplete until the unresolved expressions are mapped.
They do not create or alter warehouse objects. Copy a reviewed query into the
corresponding private SQL file; otherwise use `sql/raw_extract/` with canonical views.

| Canonical field | Known candidate source | Still to establish |
| --- | --- | --- |
| Medical claim/event key | MEDICAL_EVENT_ID | Source line grain, revisions, uniqueness |
| Patient key | PATIENT_ID | Consistent identifier type across all seven tables |
| Medical event date | SERVICE_DATE | Actual analytical availability separately |
| Medical diagnosis | DIAGNOSIS_CODES | Array/delimiter, code system and normalization |
| Medical procedure | PROCEDURE_CODE | Code type, deduplication after diagnosis expansion |
| Provider | RENDERING_NPI / BILLING_NPI | Approved role/fallback and directory mapping |
| Place of service | PLACE_OF_SERVICE | Mapping to model categories, e.g. emergency/urgent |
| Pharmacy key | PHARMACY_EVENT_ID, TRANSACTION_NUMBER | Unique source line and reversal/revision handling |
| Fill date / product | FILL_DATE / NDC11 | Availability and effective therapy classification |
| Pharmacy status | TRANSACTION_RESULT / TRANSACTION_STATUS | Observed value-to-status rules and reversal timing |
| Supply / quantity | DAYS_SUPPLY / QUANTITY | Null/invalid-value rules |
| Prescriber | PRESCRIBER_NPI | Provider-directory joins and specialty normalization |
| Plan | PRIMARY_KH_PLAN_ | Exact identifier spelling and plan lookup |
| Cost | PATIENT_RESPONSIBILITY / PATIENT_OOP | Choose the approved measure, not their unverified sum |

Do not supply constant paid status, guessed availability dates or guessed disease
codes to make a query executable. Where reversal history exists, include canonical
`reversal_date` with its reviewed availability meaning in both extraction and file.
No proof of continuity follows from a patient's first and last observed claims.

The medical scaffold emits one diagnosis per row and retains a procedure on only
the first expanded row. The diagnosis expression must produce an ordered array;
review null/blank elements and repeated codes upstream. Visit features currently
count qualifying diagnosis rows, not distinct encounters. If source event/claim
grain cannot fit this contract, use a reviewed canonical view or replace the adapter.

The cohort view must contain the common eligible candidate population, not only
future switchers, and cover the needed historical/current assessment populations.
Resolve extraction date placeholders to preserve history, future outcome observation
and claim runout. LATEST is a source-family example; substitute a consistent approved
release or point-in-time source for all seven tables.
