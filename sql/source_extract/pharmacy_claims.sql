-- Resolve source status/availability, effective therapy class and patient-cost
-- expressions explicitly. Do not treat all records as paid or fill date as availability.
-- The approved cohort includes both future switchers and nonswitchers.
-- NDC11 is already text: never cast it to a number or pad without a validated crosswalk.
SELECT
    {{UNIQUE_PHARMACY_LINE_ID_EXPRESSION}}::VARCHAR AS claim_id,
    s.PATIENT_ID::VARCHAR AS patient_id,
    s.FILL_DATE::DATE AS fill_date,
    {{PHARMACY_AVAILABILITY_EXPRESSION}}::DATE AS available_date,
    {{PHARMACY_STATUS_EXPRESSION}} AS status,
    s.NDC11::VARCHAR AS drug_id,
    {{EFFECTIVE_THERAPY_CLASS_EXPRESSION}} AS therapy_class,
    s.QUANTITY AS quantity,
    s.DAYS_SUPPLY AS days_supply,
    s.PRESCRIBER_NPI::VARCHAR AS prescriber_id,
    s.PRIMARY_KH_PLAN_::VARCHAR AS plan_id,
    {{PATIENT_COST_EXPRESSION}} AS patient_cost
FROM {{SOURCE_DATABASE}}.{{SOURCE_SCHEMA}}.PHARMACY_EVENTS_LATEST s
{{REVIEWED_EFFECTIVE_THERAPY_MAPPING_JOIN}}
WHERE s.FILL_DATE BETWEEN '{{EXTRACT_START_DATE}}'::DATE AND '{{EXTRACT_END_DATE}}'::DATE
  AND EXISTS (SELECT 1 FROM {{APPROVED_PATIENT_COHORT_VIEW}} c WHERE c.PATIENT_ID = s.PATIENT_ID);
