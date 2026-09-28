# Difference-in-Differences: API

```@meta
CurrentModule = DrSnow
```

Reference documentation for the functions and types described in [Difference-in-Differences](../did.md).
The panel container `TreatmentPanel` and its helpers `validate_panel` and
`preprocess_panel` are documented with the core interface.

## Treatment timing

```@docs
treatment_timing
TreatmentTiming
FirstTreated
```

## Two-way fixed effects and its diagnostics

```@docs
did_twfe
twfe_weights
TWFEWeights
bacon_decomposition
BaconDecomposition
```

## Event studies

```@docs
event_study
did_sun_abraham
did_imputation
relative_periods
event_study_average
estimate(::EventStudyEstimate)
confint(::EventStudyEstimate)
```

## Group-time effects under staggered adoption

```@docs
did_drdid
did_callaway_santanna
aggregate_att
estimate(::CallawaySantAnnaEstimate)
confint(::CallawaySantAnnaEstimate)
confint(::AggregatedATT)
did_etwfe
estimate(::ETWFEEstimate)
```

## Non-binary, non-absorbing and continuous treatments

```@docs
did_multiplegt_dyn
did_continuous
confint(::ContinuousDiDEstimate)
```

## Pre-trends and covariate balance

```@docs
pre_trend_test
parallel_trends_test
pretreatment_balance
```

## Sensitivity analysis for parallel trends

```@docs
honest_did
honest_breakdown
confint(::HonestDiDResult)
```

## Result types

```@docs
DiDEstimate
EventStudyEstimate
AggregatedATT
CallawaySantAnnaEstimate
ETWFEEstimate
ContinuousDiDEstimate
HonestDiDResult
```
