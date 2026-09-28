# Design area: analytic power, prognostic blocking and matched pairs, experiment
# analysis, simulation-based diagnosis and surrogate design optimization.

import CSV

const DES_VALDIR = joinpath(@__DIR__, "..", "validation", "design")

include("test_power.jl")
include("test_matching.jl")
include("test_blocking.jl")
include("test_analysis.jl")
include("test_simulation.jl")
include("test_optimize.jl")
