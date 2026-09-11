-- Run blocks separately after replacing every placeholder.
-- Retain query IDs, timestamps, source versions and aggregate outputs privately.

-- E01: Inventory and datatype evidence. Metadata does not prove business meaning.
SELECT TABLE_NAME, COLUMN_NAME, ORDINAL_POSITION, DATA_TYPE, IS_NULLABLE
FROM {{SOURCE_DATABASE}}.INFORMATION_SCHEMA.COLUMNS
WHERE TABLE_SCHEMA = '{{SOURCE_SCHEMA}}'
ORDER BY TABLE_NAME, ORDINAL_POSITION;

-- E02: Canonical medical grain, dates and completeness, within the reviewed extract.
SELECT COUNT(*) AS rows, COUNT(DISTINCT claim_id) AS distinct_claim_lines,
       COUNT(DISTINCT patient_id) AS patients,
       COUNT_IF(patient_id IS NULL) AS missing_patient,
       COUNT_IF(claim_date IS NULL) AS missing_service_date,
       COUNT_IF(available_date IS NULL) AS missing_availability,
       COUNT_IF(available_date < claim_date) AS availability_before_service,
       MIN(claim_date) AS first_service_date, MAX(claim_date) AS last_service_date
FROM {{MEDICAL_CLAIMS_CANONICAL_VIEW}};

-- E03: Diagnosis frequency after reviewed normalization; review ICD version separately.
SELECT diagnosis_code, COUNT(*) AS rows, COUNT(DISTINCT patient_id) AS patients
FROM {{MEDICAL_CLAIMS_CANONICAL_VIEW}}
GROUP BY diagnosis_code ORDER BY rows DESC;

-- E04: Pharmacy grain and status. A status label needs source documentation.
SELECT status, COUNT(*) AS rows, COUNT(DISTINCT claim_id) AS distinct_claim_lines,
       COUNT(DISTINCT patient_id) AS patients,
       COUNT_IF(drug_id IS NULL) AS missing_product,
       COUNT_IF(days_supply IS NULL OR days_supply < 0) AS invalid_supply,
       MIN(fill_date) AS first_fill_date, MAX(fill_date) AS last_fill_date
FROM {{PHARMACY_CLAIMS_CANONICAL_VIEW}}
GROUP BY status ORDER BY rows DESC;

-- E05: Product identifier shape; lengths alone do not establish the code system.
SELECT LENGTH(drug_id::VARCHAR) AS product_code_length, COUNT(*) AS rows
FROM {{PHARMACY_CLAIMS_CANONICAL_VIEW}}
GROUP BY product_code_length ORDER BY product_code_length;

-- E06: Zero or multiple effective mappings for a pharmacy line must be resolved.
SELECT p.claim_id, COUNT(m.drug_id) AS applicable_mappings
FROM {{PHARMACY_CLAIMS_CANONICAL_VIEW}} p
LEFT JOIN {{THERAPY_MAPPING_CANONICAL_VIEW}} m
  ON p.drug_id = m.drug_id
 AND p.fill_date BETWEEN m.effective_start AND m.effective_end
GROUP BY p.claim_id HAVING COUNT(m.drug_id) <> 1;

-- E07: Enrollment interval quality; continuity is checked per assessment in Python.
SELECT COUNT(*) AS rows, COUNT(DISTINCT patient_id) AS patients,
       COUNT_IF(coverage_start IS NULL OR coverage_end IS NULL) AS missing_dates,
       COUNT_IF(coverage_end < coverage_start) AS reversed_intervals
FROM {{ENROLLMENT_CANONICAL_VIEW}};
