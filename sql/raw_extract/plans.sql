-- Replace the placeholder with a reviewed, consistently versioned canonical view.
-- That view/query must already filter the approved cohort and extraction window.
-- Optional historical index_date (patients) and reversal_date (claims) may be added.
SELECT
    PLAN_ID AS plan_id,
    PAYER_TYPE AS payer_type
FROM {{PLANS_CANONICAL_VIEW}};
