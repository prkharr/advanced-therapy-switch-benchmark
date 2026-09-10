-- Replace the placeholder with a reviewed, consistently versioned canonical view.
-- That view/query must already filter the approved cohort and extraction window.
-- Optional historical index_date (patients) and reversal_date (claims) may be added.
SELECT
    PROVIDER_ID AS provider_id,
    SPECIALTY AS specialty,
    GEOGRAPHY AS geography,
    ORGANIZATION AS organization
FROM {{PROVIDERS_CANONICAL_VIEW}};
