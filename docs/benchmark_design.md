# Benchmark design and decision policy

## Comparisons

The tabular experiment holds cohort, split, aggregate inputs, and evaluation fixed
while comparing a prevalence baseline, logistic regression, random forest,
gradient-boosting libraries, and MLP. This answers whether generic tabular deep
learning adds value over mature structured-data methods.

The longitudinal experiment holds cohort and split fixed but supplies only
pre-index event sequences to compact LSTM, GRU, bidirectional LSTM, and Transformer
models. Its classical comparator is the best validation-selected aggregate model.
A bidirectional recurrent encoder is safe here because both directions traverse a
completed historical window; no event after index is present.

## Selection hierarchy

1. Select hyperparameters and threshold using validation PR-AUC/ranking behavior,
   never test performance.
2. Establish test PR-AUC, Recall@10/20%, and Lift@10/20% with 95% bootstrap
   intervals.
3. Pair bootstrap resamples when comparing the best classical and best sequence-DL
   scores on the same patients.
4. Treat a difference as credible only when uncertainty, out-of-time stability,
   and field-capacity metrics support it.
5. Weigh calibration, interpretability, training/inference cost, maintenance, and
   scoring infrastructure before recommending deployment.

A neural model is not recommended simply because its point estimate is higher.
Likewise, an MLP loss does not answer whether temporal representation learning is
useful.

## Calibration

Raw, Platt/sigmoid, and isotonic candidates are fit against validation data. A
calibrator is selected using validation Brier score and then applied once to test
scores. Isotonic should be rejected when validation positives are too sparse to
support a stable monotonic fit.

## HCP layer

The patient model outputs `advanced_therapy_propensity_score`. A separately
configured attribution rule associates each eligible patient with one relevant
HCP using pre-index history. HCP metrics aggregate patient counts, capacity bands,
and `expected_switchers = sum(patient probabilities)`.

The opportunity score is a transparent weighted combination of normalized
expected switchers, high-propensity patients, and eligible patient volume. Its
weights are business rules, not learned clinical effects. The output prioritizes
commercial opportunity for review and does not recommend treatment.
