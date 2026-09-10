-- Replace the placeholder with a reviewed, consistently versioned canonical view.
-- That view/query must already filter the approved cohort and extraction window.
-- Optional historical index_date (patients) and reversal_date (claims) may be added.
SELECT
    PATIENT_ID AS patient_id,
    BIRTH_YEAR AS birth_year,
    GENDER AS gender,
    GEOGRAPHY AS geography,
    OBSERVATION_START AS observation_start,
    OBSERVATION_END AS observation_end
FROM {{PATIENTS_CANONICAL_VIEW}};
