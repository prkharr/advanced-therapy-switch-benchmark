-- Replace the placeholder with a reviewed, consistently versioned canonical view.
-- That view/query must already filter the approved cohort and extraction window.
-- Optional historical index_date (patients) and reversal_date (claims) may be added.
SELECT
    CLAIM_ID AS claim_id,
    PATIENT_ID AS patient_id,
    CLAIM_DATE AS claim_date,
    AVAILABLE_DATE AS available_date,
    STATUS AS status,
    DIAGNOSIS_CODE AS diagnosis_code,
    PROCEDURE_CODE AS procedure_code,
    PROVIDER_ID AS provider_id,
    PLACE_OF_SERVICE AS place_of_service
FROM {{MEDICAL_CLAIMS_CANONICAL_VIEW}};
