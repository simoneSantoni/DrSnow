# Instrumental variables / LATE.
#
# design.jl          internal engine: estimation sample, partialling-out of fixed
#                    effects / covariates / weights, HC1 and multiway-cluster
#                    covariance, OLS and 2SLS on the partialled design
# estimator.jl       2SLS on FixedEffectModels with honest estimand labels
# first_stage.jl     first-stage strength, weak-IV pre-tests, tF
# weak_iv_robust.jl  Anderson–Rubin / CLR / K (homoskedastic and Kleibergen 2005
#                    robust) / jackknife AR tests and sets
# kclass.jl          LIML, Fuller, k-class, HLIM, HFUL; Bekker, Hansen–Hausman–Newey
#                    and many-IV robust SEs
# jive.jl            JIVE1, JIVE2, UJIVE, CJIVE; leave-one-out design; jackknife AR
# judge.jl           judge / examiner designs: leniency, estimation, diagnostics
#                    (incl. the Frandsen–Lefgren–Leslie test)
# shift_share.jl     shift-share IV: AKM / AKM0 / BHJ inference, Rotemberg weights
#                    (cross-section, panels, 2SLS), recentering and randomization
#                    inference
# mte.jl             marginal treatment effects: propensity, support, local IV
#                    (semiparametric / polynomial / normal), MTE-weighted parameters
# mte_bounds.jl      linear-programming bounds on treatment-effect parameters from
#                    IV-like estimands and shape-restricted MTRs (Mogstad, Santos &
#                    Torgovitsky 2018)
# huber_mellace.jl   Huber & Mellace (2015) moment-inequality validity test
# compliance.jl      compliance types, complier profiles, IPW (κ) LATE
# diagnostics.jl     balance, monotonicity / exclusion falsification, overid,
#                    endogeneity, Kitagawa instrument-validity test (with discrete
#                    covariates)
# sensitivity.jl     plausibly exogenous (Conley, Hansen & Rossi 2012)
# extrapolation.jl   reweighting cell LATEs (nonparametric) or a linear LATE(x)
#                    (parametric) to other populations (Angrist & Fernández-Val 2013)
# ml_spec_test.jl    machine-learning residual prediction specification tests and
#                    weak-IV-robust confidence sets (Scheidegger, Londschien &
#                    Bühlmann 2025)
# ml_first_stage.jl  cross-fitted first-stage strength for DML IV models and the DML
#                    Anderson–Rubin test / confidence set
# multiple_iv_weights.jl  2SLS with several discrete instruments: weights on response
#                    groups under partial / vector / IA monotonicity (Mogstad,
#                    Torgovitsky & Walters 2021)
# dml_lqte.jl        local quantile treatment effects and complier distributions with
#                    cross-fitted ML nuisances (DoubleML LPQ / QTE)
#
# The ML-era files (ml_spec_test.jl, ...) call learners and cross-fitting helpers of
# the ml area, which is included later; those functions are only resolved at run
# time, so no ml type appears in a signature or struct field here.

include("design.jl")
include("estimator.jl")
include("first_stage.jl")
include("weak_iv_robust.jl")
include("kclass.jl")
include("jive.jl")
include("judge.jl")
include("shift_share.jl")
include("mte.jl")
include("mte_bounds.jl")
include("compliance.jl")
include("diagnostics.jl")
include("huber_mellace.jl")
include("sensitivity.jl")
include("extrapolation.jl")
include("ml_spec_test.jl")
include("ml_first_stage.jl")
include("multiple_iv_weights.jl")
include("dml_lqte.jl")

export IVEstimate, FirstStageResult, WeakIVDiagnostics, WeakIVConfidenceSet
export ComplianceAnalysis, ComplierProfile, IPWLATEEstimate, PlausiblyExogenousResult
export iv_regression, late_2sls, first_stage_diagnostics, tf_confint
export weak_iv_test, weak_iv_confidence_set
export estimate_compliance, complier_characteristics, late_ipw
export complier_outcome_distribution
export instrument_balance, first_stage_sign_test, zero_first_stage_test
export overidentification_test, endogeneity_test, instrument_validity_test
export plausibly_exogenous
export late_extrapolation, LATEExtrapolation
export kclass_iv, KClassEstimate, jive, JIVEEstimate
export judge_iv, JudgeIVEstimate, judge_leniency, judge_balance_test
export judge_validity_test, judge_subsample_monotonicity
export shift_share_instrument, shift_share_iv, ShiftShareIVEstimate
export rotemberg_weights, RotembergDecomposition
export mte, MTEEstimate, mte_propensity, huber_mellace_test
export mte_bounds, MTEBounds
export residual_prediction_test, residual_prediction_confidence_set
export MLFirstStage, ml_first_stage, dml_weak_iv_test, dml_weak_iv_confidence_set
export MultipleIVWeights, multiple_iv_weights
export LQTEEstimate, dml_lqte, ComplierDistribution, dml_complier_cdf
