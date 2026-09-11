-- Replace the placeholder with a reviewed, consistently versioned canonical view.
-- That view/query must already filter the approved cohort and extraction window.
-- Optional historical index_date (patients) and reversal_date (claims) may be added.
SELECT
    PATIENT_ID AS patient_id,
    COVERAGE_START AS coverage_start,
    COVERAGE_END AS coverage_end,
    PLAN_ID AS plan_id
FROM {{ENROLLMENT_CANONICAL_VIEW}};
