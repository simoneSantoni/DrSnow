# Regression discontinuity area tests.
#
# Validation against rdrobust / rddensity reference values (test/validation/rd, produced
# by test/validation/rd/generate_reference.R), unit tests, and Monte Carlo checks.

using Distributions: Beta, Normal, ccdf, quantile

include("helpers.jl")

@testset "validation: rdrobust" begin
    include("test_validation_rdrobust.jl")
end
@testset "validation: rddensity" begin
    include("test_validation_density.jl")
end
@testset "validation: rdplot" begin
    include("test_validation_plot.jl")
end
@testset "validation: RDHonest, rdd, stdvars" begin
    include("test_validation_honest.jl")
end
@testset "unit" begin
    include("test_rd.jl")
end
@testset "honest inference, McCrary, stdvars" begin
    include("test_honest.jl")
end
@testset "Monte Carlo" begin
    include("test_montecarlo.jl")
end
@testset "flexible covariate adjustment (rd_flex)" begin
    include("test_flex.jl")
end
