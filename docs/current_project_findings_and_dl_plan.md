# NT1 Advanced-Therapy Model: Current Findings and Deep-Learning Plan

## Purpose

The project identifies patients with narcolepsy type 1 (NT1) who are using
generic therapy and are likely to move to an advanced therapy within the next
90 days. Patients are scored regularly so field teams can focus on a smaller,
higher-value group of patients and healthcare providers (HCPs).

This is an advanced-therapy readiness model built in the TAK-861 context. The
predicted event is advanced-therapy use, rather than TAK-861 use itself.

## Current approach

The current production approach is a LightGBM classification model, version
V63, trained through a Snowflake AutoML workflow.

The main steps are:

1. Identify eligible NT1 patients.
2. Create a patient snapshot for each eligible month.
3. Describe treatment, diagnosis, healthcare-use, plan, and HCP history before
   each snapshot.
4. Predict whether the patient will use advanced therapy in the following 90
   days.
5. Rank patients by score and measure lift in the highest-scoring groups.
6. Use SHAP values to show the main factors associated with each score.
7. Aggregate high-scoring patients to HCP-level outreach lists.

## Cohort and label

An eligible patient has:

- at least one NT1 claim;
- another NT1, NT2, or idiopathic-hypersomnia claim more than 90 days away;
- generic therapy coverage on at least 135 of the previous 270 days; and
- no advanced-therapy use before the snapshot.

The label is:

- `1`: one or more advanced-therapy claims during the next 90 days;
- `0`: no advanced-therapy claim during that period.

The working data is organized by patient and snapshot date. The current patient
universe table contains 23,151 rows, while the backtest universe contains
44,949 rows. Both use the same general feature structure.

## Data and feature groups

The model uses medical claims, pharmacy claims, therapy mappings, plan data,
and HCP information. Confirmed feature groups include:

- generic therapy coverage, gaps, changes, and discontinuations;
- time since first generic fill and number of generic therapies tried;
- NT1 diagnosis history and recency;
- symptom and comorbidity claims over recent 3- and 12-month windows;
- emergency and urgent-care activity;
- sleep-medicine, neurology, pulmonology, and psychiatry visits;
- plan payment, rejection, patient-cost, and therapy-control measures;
- HCP history of prescribing advanced therapies;
- gender indicators; and
- short sequences of diagnosis, procedure, and pharmacy codes.

## Suitable deep-learning models

### 1. Multilayer perceptron (MLP)

Use the existing wide feature table as input. This is the simplest neural
baseline and tests whether nonlinear feature combinations improve on LightGBM.

### 2. FT-Transformer

Use a tabular Transformer for the existing numeric and categorical features.
This is suitable for testing interactions between treatment, plan, symptom,
and HCP variables without rebuilding the data as an event sequence.

### 3. GRU or LSTM

Represent each patient's claims in date order. These models can learn how the
order and timing of diagnoses, procedures, and prescriptions relate to later
therapy escalation. A GRU is the preferred first sequence model because it is
smaller and faster than an LSTM.

### 4. Temporal Transformer

Create a sequence containing claim code, claim type, event date, and time since
the previous event. A compact Transformer can learn longer patterns that are
difficult to express through fixed 3-, 6-, or 12-month counts.

### 5. Hybrid model

Combine a GRU or Transformer representation of the claims sequence with the
existing engineered features. This is the strongest candidate because it keeps
the useful business features from V63 while adding detailed longitudinal
information.

## Implementation plan

1. **Freeze the benchmark.** Use the same cohort, labels, training period,
   backtest period, and patient-level partitions for every model.
2. **Reproduce V63.** Record LightGBM performance and scoring time as the
   reference result.
3. **Build two model inputs.** Prepare the current wide feature table and a
   pre-snapshot chronological claims sequence.
4. **Train in stages.** Compare MLP and FT-Transformer on the wide data, then
   GRU and a compact Transformer on sequences, followed by the hybrid model.
5. **Handle the rare outcome.** Use class weighting, carefully tuned sampling,
   and validation-based early stopping.
6. **Evaluate field value.** Compare PR-AUC, lift, precision, and recall in the
   top 5%, 10%, and 20% of scores. Also review calibration, stability by month,
   and bootstrap confidence intervals.
7. **Keep explanations usable.** Retain SHAP explanations for the tabular
   component and use feature ablation or integrated gradients for sequence
   models.
8. **Select on evidence.** Adopt deep learning only if it provides stable
   out-of-time lift or captures more true escalators at the same outreach
   capacity without creating unreasonable scoring or maintenance cost.

## Recommended first experiment

Run three models on exactly the same cohort and backtest:

1. existing LightGBM V63;
2. GRU using chronological claims; and
3. hybrid GRU plus the current engineered features.

This provides the quickest clear answer to the main question: whether learning
directly from the order of claims adds useful information beyond the current
LightGBM feature set.
