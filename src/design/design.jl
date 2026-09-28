# Design-stage tools: blocking on predicted outcomes, analytic and simulation-based
# power, minimum detectable effects and design optimization. Non-exported helpers are
# prefixed `_des_`.

include("power_analytic.jl")
include("power_rd.jl")
include("matching.jl")
include("prognostic.jl")
include("blocking.jl")
include("analysis.jl")
include("simulation.jl")
include("optimize.jl")

export PowerAnalysis, power_means, power_proportions, power_cluster, power_blocked
export power_did, power_iv, power_rd
export PrognosticScore, prognostic_score
export BlockingDesign, block_design, assign_treatment, variance_reduction
export ExperimentEstimate, experiment_estimate
export DeclaredDesign, declare_design, DesignDiagnosis, diagnose_design, diagnose_grid
export DesignOptimization, optimize_design
