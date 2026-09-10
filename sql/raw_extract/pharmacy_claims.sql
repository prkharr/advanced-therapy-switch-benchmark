-- Replace the placeholder with a reviewed, consistently versioned canonical view.
-- That view/query must already filter the approved cohort and extraction window.
-- Optional historical index_date (patients) and reversal_date (claims) may be added.
SELECT
    CLAIM_ID AS claim_id,
    PATIENT_ID AS patient_id,
    FILL_DATE AS fill_date,
    AVAILABLE_DATE AS available_date,
    STATUS AS status,
    DRUG_ID AS drug_id,
    THERAPY_CLASS AS therapy_class,
    QUANTITY AS quantity,
    DAYS_SUPPLY AS days_supply,
    PRESCRIBER_ID AS prescriber_id,
    PLAN_ID AS plan_id,
    PATIENT_COST AS patient_cost
FROM {{PHARMACY_CLAIMS_CANONICAL_VIEW}};
