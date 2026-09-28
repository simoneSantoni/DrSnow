# Randomization inference: assignment mechanisms, Fisher tests, CI inversion,
# regression RI, balance and multiple testing.

include("helpers.jl")
include(joinpath(@__DIR__, "..", "validation", "ri", "ri2_reference.jl"))

include("test_mechanisms.jl")
include("test_fisher.jl")
include("test_confint.jl")
include("test_regression.jl")
include("test_multiple_balance.jl")
include("test_size.jl")
