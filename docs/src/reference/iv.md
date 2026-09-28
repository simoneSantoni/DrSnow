# Instrumental Variables: API

```@meta
CurrentModule = DrSnow
```

Reference documentation for the functions and types described in [Instrumental Variables and LATE](../iv.md), [IV Designs: Many Instruments, Judges, Shift-Share, MTE](../iv_designs.md), [IV with Machine-Learning First Stages](../iv_ml.md).

## Two-stage least squares and the LATE

```@docs
iv_regression
late_2sls
IVEstimate
```

## First-stage strength

```@docs
first_stage_diagnostics
WeakIVDiagnostics
FirstStageResult
tf_confint
```

## Weak-instrument-robust inference

```@docs
weak_iv_test
weak_iv_confidence_set
WeakIVConfidenceSet
```

## Compliers and conditional instrument validity

```@docs
estimate_compliance
ComplianceAnalysis
complier_characteristics
ComplierProfile
late_ipw
IPWLATEEstimate
complier_outcome_distribution
```

## Instrument validity and specification tests

```@docs
instrument_balance
first_stage_sign_test
zero_first_stage_test
instrument_validity_test
huber_mellace_test
overidentification_test
endogeneity_test
```

## Sensitivity and external validity

```@docs
plausibly_exogenous
PlausiblyExogenousResult
late_extrapolation
LATEExtrapolation
```

## Many instruments: k-class and jackknife IV

```@docs
kclass_iv
KClassEstimate
jive
JIVEEstimate
```

## Judge and examiner designs

```@docs
judge_iv
JudgeIVEstimate
judge_leniency
judge_balance_test
judge_validity_test
judge_subsample_monotonicity
```

## Shift-share instruments

```@docs
shift_share_instrument
shift_share_iv
ShiftShareIVEstimate
rotemberg_weights
RotembergDecomposition
```

## Marginal treatment effects

```@docs
mte_propensity
mte
MTEEstimate
mte_bounds
MTEBounds
```

## Several discrete instruments

```@docs
multiple_iv_weights
MultipleIVWeights
```

## Machine-learning specification tests

```@docs
residual_prediction_test
residual_prediction_confidence_set
```

## Double machine learning with instruments

```@docs
ml_first_stage
MLFirstStage
dml_weak_iv_test
dml_weak_iv_confidence_set
```

## Distributional effects for compliers

```@docs
dml_lqte
LQTEEstimate
dml_complier_cdf
ComplierDistribution
```
