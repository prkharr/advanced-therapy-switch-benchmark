-- Replace the placeholder with a reviewed, consistently versioned canonical view.
-- That view/query must already filter the approved cohort and extraction window.
-- Optional historical index_date (patients) and reversal_date (claims) may be added.
SELECT
    DRUG_ID AS drug_id,
    THERAPY_CLASS AS therapy_class,
    EFFECTIVE_START AS effective_start,
    EFFECTIVE_END AS effective_end
FROM {{THERAPY_MAPPING_CANONICAL_VIEW}};
