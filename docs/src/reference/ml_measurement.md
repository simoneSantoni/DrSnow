# ML-Measured Variables: API

```@meta
CurrentModule = DrSnow
```

Reference documentation for the functions and types described in
[Prediction-Powered Inference and ML-Measured Variables](../ml_measurement.md).
Nuisance learners used for cross-fitting are on the
[main causal machine learning API page](ml.md).

## Prediction-powered inference

```@docs
PPIEstimate
ppi_mean
ppi_ols
ppi_logistic
```

## ML-measured variables

```@docs
MeasurementEstimate
MeasuredOutcomeEstimate
dsl_pseudo_outcome
dsl_regression
dsl_proportions
ppi_ate
ppi_regression
cross_ppi
did_with_predicted_outcome
rd_with_predicted_outcome
regression_calibration
differential_error_test
```
