# Causal machine learning: double/debiased ML, heterogeneous effects (DR-learner,
# generalized random forests, meta-learners, subgroups), policy learning,
# prediction-powered inference. Non-exported helpers are prefixed `_ml_`.

include("glmnet.jl")
include("learners.jl")
include("crossfit.jl")
include("dml.jl")
include("dml_did.jl")
include("dml_did_multi.jl")
include("rd_flex.jl")
include("cate.jl")
include("generic_ml.jl")
include("policy.jl")
include("ppi.jl")
include("grf_core.jl")
include("grf.jl")
include("grf_analysis.jl")
include("metalearners.jl")
include("hte_classical.jl")
include("measurement.jl")
include("dsl.jl")
include("ppi_causal.jl")
include("measured_designs.jl")
include("measurement_error.jl")

export NuisanceLearner, fitpredict, fitpredict_proba
export OLSLearner, RidgeLearner, LassoLearner, LogisticLearner
export PenalizedLogisticLearner, KNNLearner, MeanLearner, MLJLearner
export crossfit_folds
export DMLEstimate, dml_plr, dml_irm, dml_pliv, dml_iivm, dml_did
export dml_did_multi, RDFlexEstimate, rd_flex
export simultaneous_confint, nuisance_loss
export predict   # StatsAPI.predict, extended for the CATE learners, forests, trees
export CATEPredictor, CATEProjection, cate_dr_learner, cate_projection
export GenericMLInference, generic_ml, blp, blp_test, gates, clan
export PolicyTree, PolicyLearningResult, policy_tree
export PPIEstimate, ppi_mean, ppi_ols, ppi_logistic
export GeneralizedRandomForest, RegressionForest, CausalForest, InstrumentalForest
export regression_forest, causal_forest, instrumental_forest, predict_interval
export HTEEstimate, heterogeneity_test, get_scores, average_treatment_effect
export best_linear_projection, test_calibration, split_frequencies, variable_importance
export RATEEstimate, rank_average_treatment_effect, double_robust_scores, policy_value
export ForestLearner, MetaLearner, s_learner, t_learner, x_learner, r_learner
export metalearner_bootstrap, subgroup_effects, interaction_effects
export MeasurementEstimate, MeasuredOutcomeEstimate, dsl_pseudo_outcome
export dsl_regression, dsl_proportions, ppi_ate, ppi_regression, cross_ppi
export did_with_predicted_outcome, rd_with_predicted_outcome
export regression_calibration, differential_error_test
