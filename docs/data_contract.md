# Canonical claims data contract

The framework is vendor-neutral at the model boundary. Source-specific ingestion
must produce the four canonical tables below. Identifiers may be opaque strings or
integers but must be stable within a snapshot. Dates are parsed as calendar dates.

## Patients

Required: `patient_id`, `gender`, `geography`, `observation_start`,
`observation_end`. Include either `age` or `birth_year`; when both exist, the age
at index is preferred. Observation bounds describe data availability, not disease
onset.

## Medical claims

Required: `claim_id`, `patient_id`, `claim_date`, `diagnosis_code`,
`procedure_code`, `provider_id`, `place_of_service`. Multiple diagnosis lines may
be retained as separate claims if claim identifiers remain unique or namespaced.
The framework treats codes as categorical signals and does not attach clinical
meaning unless an approved mapping is configured.

## Pharmacy claims

Required: `claim_id`, `patient_id`, `fill_date`, `drug_id`, `therapy_class`,
`quantity`, `days_supply`, `prescriber_id`. `therapy_class` must use the configured
conventional/advanced vocabulary. The raw product identifier may remain more
granular for history features and sequence tokenization.

## Providers

Required: `provider_id`, `specialty`, `geography`, `organization`. Provider data
supports pre-index specialist features and the later attribution layer. A missing
organization can be represented by a documented unknown value.

## Mapping a production delivery

Use `data.tables.<table>.file` and `data.tables.<table>.columns` to rename source
fields without modifying feature/model code. The loader validates required fields,
date parseability, and null patient identifiers. Add delivery-specific validation
upstream for adjudication status, reversals, duplicate lines, code systems, and
enrollment completeness.

Therapy definitions belong under `therapy_mapping` in a protected environment
configuration. They must be versioned, reviewed by an authorized business/clinical
owner, and checked for overlap. Never embed proprietary definitions in source code.

## Minimum longitudinal coverage

Eligibility is calculated per patient. The default requires 365 observable days
before the index date and 90 days afterward. A conventional exposure anchors the
index; any advanced exposure on or before index excludes the patient. An advanced
fill strictly after index and within the prediction horizon produces label 1.

For production use, confirm whether coverage/enrollment, claim run-out, rejected
claims, reversals, and cash fills change what “observable” means. Encode those
rules in the adapter or cohort module and add tests before benchmarking.
