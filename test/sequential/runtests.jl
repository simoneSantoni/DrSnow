# Sequential inference: confidence sequences, anytime-valid tests, group-sequential
# designs. Reference values in test/validation/sequential/ (confseq, gsDesign, rpact).

using CSV
using Distributions: Beta, Normal, cdf, quantile

const SEQ_VALDIR = joinpath(@__DIR__, "..", "validation", "sequential")

include("test_confseq.jl")
include("test_ate_msprt.jl")
include("test_group_sequential.jl")
include("test_monte_carlo.jl")
