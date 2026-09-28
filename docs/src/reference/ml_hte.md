# Heterogeneous Effects and Policy Learning: API

```@meta
CurrentModule = DrSnow
```

Reference documentation for the functions and types described in [Heterogeneous
Effects and Policy Learning](../ml_hte.md). Nuisance learners and the DML
estimators are on the [main causal machine learning API page](ml.md).

## CATE learners and generic ML inference

```@docs
CATEPredictor
cate_dr_learner
predict(::CATEPredictor, ::AbstractDataFrame)
CATEProjection
cate_projection
GenericMLInference
generic_ml
blp
blp_test
gates
clan
```

## Policy learning

```@docs
PolicyTree
PolicyLearningResult
policy_tree
predict(::PolicyTree, ::AbstractDataFrame)
```

## Generalized random forests

```@docs
GeneralizedRandomForest
RegressionForest
CausalForest
InstrumentalForest
regression_forest
causal_forest
instrumental_forest
predict(::GeneralizedRandomForest)
predict_interval
```

## Forest-based effect summaries

```@docs
HTEEstimate
heterogeneity_test
get_scores
double_robust_scores
average_treatment_effect
best_linear_projection
cate_projection(::Union{CausalForest,InstrumentalForest})
test_calibration
RATEEstimate
rank_average_treatment_effect
policy_value
split_frequencies
variable_importance
```

## Meta-learners and classical subgroup analysis

```@docs
MetaLearner
s_learner
t_learner
x_learner
r_learner
predict(::MetaLearner, ::AbstractDataFrame)
metalearner_bootstrap
subgroup_effects
interaction_effects
```
