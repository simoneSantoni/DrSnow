# Causal Machine Learning: API

```@meta
CurrentModule = DrSnow
```

Reference documentation for the causal machine learning area, described in
[Causal machine learning](../ml.md), [Heterogeneous Effects and Policy
Learning](../ml_hte.md) and [Prediction-Powered Inference and ML-Measured
Variables](../ml_measurement.md). The area's API is split over three pages:

- this page: nuisance learners, cross-fitting and double/debiased machine learning;
- [Heterogeneous effects and policy learning](ml_hte.md): CATE learners, generic ML
  inference, policy trees, generalized random forests, meta-learners and subgroup
  analysis;
- [ML-measured variables](ml_measurement.md): prediction-powered inference and
  design-based supervised learning for variables measured by ML models or LLMs.

The flexible covariate adjustment for regression discontinuity designs
([`rd_flex`](@ref)) is documented with the [regression-discontinuity API](rd.md).

## Nuisance learners and cross-fitting

```@docs
NuisanceLearner
fitpredict
fitpredict_proba
OLSLearner
RidgeLearner
LassoLearner
LogisticLearner
PenalizedLogisticLearner
KNNLearner
MeanLearner
MLJLearner
ForestLearner
crossfit_folds
```

## Double/debiased machine learning

```@docs
DMLEstimate
dml_plr
dml_irm
dml_pliv
dml_iivm
dml_did
dml_did_multi
simultaneous_confint
nuisance_loss
```
