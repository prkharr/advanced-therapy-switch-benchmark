-- Read-only independent extraction scope from raw diagnosis history.
-- Resolve the diagnosis encoding, code system, code list and date expressions.
-- This is a broad extraction filter, not the final eligible modeling cohort.
-- Do not select patients using future advanced treatment, model labels or scores.
SELECT DISTINCT s.PATIENT_ID::VARCHAR AS patient_id
FROM {{SOURCE_DATABASE}}.{{SOURCE_SCHEMA}}.MEDICAL_EVENTS_LATEST s,
LATERAL FLATTEN(INPUT => {{DIAGNOSIS_ARRAY_EXPRESSION}}, OUTER => TRUE) d
WHERE {{NORMALIZED_DIAGNOSIS_EXPRESSION}} IN ({{REVIEWED_DIAGNOSIS_SQL_LITERALS}})
  AND s.SERVICE_DATE BETWEEN '{{EXTRACT_START_DATE}}'::DATE AND '{{SCORING_DATE}}'::DATE
  AND {{MEDICAL_AVAILABILITY_EXPRESSION}}::DATE <= '{{SCORING_DATE}}'::DATE;
