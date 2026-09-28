# Tests of the adaptive-experiments area (src/adaptive): assignment policies, the
# experiment runner, adaptively weighted inference, off-policy evaluation and
# micro-randomized trials (WCLS / EMEE against MRTAnalysis).

import CSV
using Distributions: TDist, quantile
using FixedEffectModels: reg, @formula

const AD_VALIDATION = joinpath(@__DIR__, "..", "validation", "adaptive")

"""Read a committed validation CSV."""
ad_read_csv(file) = CSV.read(joinpath(AD_VALIDATION, file), DataFrame)

"""Binomial tolerance for Monte Carlo coverage checks with `reps` replications."""
ad_cover_tol(reps; level=0.95) = 3.5 * sqrt(level * (1 - level) / reps) + 0.01

include("test_policies.jl")
include("test_experiment.jl")
include("test_weighting.jl")
include("test_ope.jl")
include("test_mrt.jl")
include("test_montecarlo.jl")
