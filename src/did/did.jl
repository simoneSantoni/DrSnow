# Difference-in-Differences.
#
# Layout:
#   results.jl           DiDEstimate, EventStudyEstimate, AggregatedATT, confint with
#                        uniform bands, event_study_average, pre_trend_test
#   timing.jl            cohort / treatment-timing layer (FirstTreated, TreatmentTiming)
#   decomposition.jl     Goodman-Bacon decomposition, dCDH TWFE weights
#   twoway_fe.jl         did_twfe, parallel_trends_test
#   drdid.jl             Sant'Anna–Zhao 2×2 kernels and did_drdid
#   callaway_santanna.jl ATT(g,t) and aggregations
#   sun_abraham.jl       interaction-weighted event study
#   imputation.jl        Borusyak–Jaravel–Spiess imputation
#   event_study.jl       event_study dispatcher and TWFE event study
#   balance.jl           pretreatment_balance
#   honest_solvers.jl    dense LP / lasso-path kernels for honest_did
#   honest_did.jl        Rambachan–Roth sensitivity analysis (honest_did)
#   multiplegt_dyn.jl    de Chaisemartin–D'Haultfœuille DID_ℓ (did_multiplegt_dyn)
#   etwfe.jl             Wooldridge extended TWFE (did_etwfe) and its aggregations
#   continuous.jl        continuous-treatment DiD (did_continuous)

include("results.jl")
include("timing.jl")
include("decomposition.jl")
include("twoway_fe.jl")
include("drdid.jl")
include("callaway_santanna.jl")
include("sun_abraham.jl")
include("imputation.jl")
include("event_study.jl")
include("balance.jl")
include("honest_solvers.jl")
include("honest_did.jl")
include("multiplegt_dyn.jl")
include("etwfe.jl")
include("continuous.jl")

export DiDEstimate, EventStudyEstimate, AggregatedATT, CallawaySantAnnaEstimate
export TreatmentTiming, FirstTreated, treatment_timing
export BaconDecomposition, TWFEWeights, bacon_decomposition, twfe_weights
export did_twfe, event_study, did_sun_abraham, did_imputation, did_drdid
export did_callaway_santanna, aggregate_att
export relative_periods, event_study_average, pre_trend_test, parallel_trends_test
export pretreatment_balance
export HonestDiDResult, honest_did, honest_breakdown
export did_multiplegt_dyn, did_etwfe, ETWFEEstimate
export did_continuous, ContinuousDiDEstimate
