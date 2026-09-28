# DrSnow test runner.
#
# Environment variables:
#   DRSNOW_TEST_GROUP  comma-separated subset of groups to run (default: all), e.g.
#                      DRSNOW_TEST_GROUP=did,iv julia --project=. -e 'using Pkg; Pkg.test()'
#   DRSNOW_SLOW_TESTS  set to "true" to run Monte Carlo size/coverage checks with full
#                      replication counts (CI runs them with reduced counts).

using DrSnow
using Test
using DataFrames
using Random
using Statistics
using LinearAlgebra
using StableRNGs

const GROUPS = let g = get(ENV, "DRSNOW_TEST_GROUP", "all")
    g == "all" ? :all : Set(strip.(split(g, ",")))
end
const SLOW_TESTS = lowercase(get(ENV, "DRSNOW_SLOW_TESTS", "false")) == "true"

rungroup(name) = GROUPS === :all || name in GROUPS

"""Monte Carlo replication count: `full` when DRSNOW_SLOW_TESTS=true, else `fast`."""
mc_reps(full::Int, fast::Int) = SLOW_TESTS ? full : fast

const AREAS = ["quality", "core", "ri", "sequential", "did", "iv", "rd", "sutva", "synth",
               "ml", "adaptive", "design", "viz", "gui", "validation"]

@testset "DrSnow.jl" begin
    for area in AREAS
        file = joinpath(@__DIR__, area, "runtests.jl")
        if rungroup(area) && isfile(file)
            @testset "$area" begin
                include(file)
            end
        end
    end
end
