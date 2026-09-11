-- Source mapping scaffold. Copy to sql_extraction/private/medical_claims.sql ONLY
-- after resolving every placeholder and reviewing event grain and source versions.
-- Diagnosis array expression must reflect the actual source delimiter/encoding.
-- Preserve one procedure per source event when exploding diagnoses.
SELECT
    s.MEDICAL_EVENT_ID::VARCHAR || ':' || COALESCE(d.INDEX::VARCHAR, 'none') AS claim_id,
    s.PATIENT_ID::VARCHAR AS patient_id,
    s.SERVICE_DATE::DATE AS claim_date,
    {{MEDICAL_AVAILABILITY_EXPRESSION}}::DATE AS available_date,
    {{MEDICAL_STATUS_EXPRESSION}} AS status,
    {{NORMALIZED_DIAGNOSIS_EXPRESSION}} AS diagnosis_code,
    IFF(d.INDEX = 0 OR d.INDEX IS NULL, s.PROCEDURE_CODE, NULL) AS procedure_code,
    {{MEDICAL_PROVIDER_EXPRESSION}}::VARCHAR AS provider_id,
    {{NORMALIZED_PLACE_OF_SERVICE_EXPRESSION}} AS place_of_service
FROM {{SOURCE_DATABASE}}.{{SOURCE_SCHEMA}}.MEDICAL_EVENTS_LATEST s,
LATERAL FLATTEN(INPUT => {{DIAGNOSIS_ARRAY_EXPRESSION}}, OUTER => TRUE) d
WHERE s.SERVICE_DATE BETWEEN '{{EXTRACT_START_DATE}}'::DATE AND '{{EXTRACT_END_DATE}}'::DATE
  AND EXISTS (SELECT 1 FROM {{INDEPENDENT_PATIENT_SCOPE_VIEW}} c WHERE c.PATIENT_ID = s.PATIENT_ID);
