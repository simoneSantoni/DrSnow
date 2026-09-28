# DrSnow benchmark suite (BenchmarkTools / PkgBenchmark compatible).
#
# `SUITE` groups representative estimators by area, each at a "small" and a "medium"
# problem size, on simulated data drawn once with fixed StableRNG seeds (so every
# run times the same problem). Stochastic estimators get a fresh StableRNG inside the
# benchmarked expression, so every sample does the same work.
#
#     julia --project=benchmark benchmark/run.jl            # run and print a table
#     julia --project=benchmark -e 'using PkgBenchmark, DrSnow; benchmarkpkg(DrSnow)'
#
# See benchmark/README.md.

using BenchmarkTools
using DataFrames
using DrSnow
using Random
using StableRNGs
using Statistics

const SUITE = BenchmarkGroup()

# Short, bounded runs: the suite is meant for spotting regressions of 2x, not 5%.
BenchmarkTools.DEFAULT_PARAMETERS.seconds = 2.0
BenchmarkTools.DEFAULT_PARAMETERS.samples = 50

quiet(f) = Base.CoreLogging.with_logger(f, Base.CoreLogging.NullLogger())

const SIZES = (small=1, medium=2)

# ---------------------------------------------------------------------------------
# Data-generating processes
# ---------------------------------------------------------------------------------

"""Staggered-adoption panel: N units × T periods, cohorts at T/2 and 3T/4, never."""
function bench_panel(rng, N, T)
    cohorts = [0, T ÷ 2, (3 * T) ÷ 4]
    g = rand(rng, cohorts, N)
    α = randn(rng, N)
    λ = cumsum(randn(rng, T)) .* 0.3
    rows = [(unit=i, time=t, y=α[i] + λ[t] + (g[i] > 0 && t >= g[i] ? 1.0 : 0.0) +
                             randn(rng),
             d=Int(g[i] > 0 && t >= g[i]))
            for i in 1:N for t in 1:T]
    return DataFrame(rows)
end

function bench_iv(rng, n)
    z = randn(rng, n)
    x = randn(rng, n)
    u = randn(rng, n)
    d = 0.4 .* z .+ 0.3 .* x .+ 0.6 .* u .+ randn(rng, n)
    return DataFrame(y=d .+ 0.5 .* x .+ u .+ randn(rng, n), d=d, z=z, x=x)
end

function bench_rd(rng, n)
    x = 2 .* rand(rng, n) .- 1
    y = 0.4 .+ 0.8 .* x .- 0.5 .* x .^ 2 .+ 0.6 .* (x .>= 0) .+ 0.3 .* randn(rng, n)
    return DataFrame(y=y, x=x)
end

"""Block panel for synthetic DiD: the last `n_treated` units treated for `T_post`."""
function bench_synth(rng, N, T; n_treated=5, T_post=5)
    f = cumsum(randn(rng, T))
    load = 0.5 .+ rand(rng, N)
    α = 5 .+ randn(rng, N)
    tr(i, t) = i > N - n_treated && t > T - T_post
    rows = [(unit=i, time=t, d=Int(tr(i, t)),
             y=α[i] + load[i] * f[t] + (tr(i, t) ? 2.0 : 0.0) + 0.5 * randn(rng))
            for i in 1:N for t in 1:T]
    return DataFrame(rows)
end

function bench_experiment(rng, n)
    d = shuffle(rng, vcat(ones(Int, n ÷ 2), zeros(Int, n - n ÷ 2)))
    x1, x2 = randn(rng, n), randn(rng, n)
    return DataFrame(y=0.5 .* x1 .+ d .* 0.5 .+ randn(rng, n), d=d, x1=x1, x2=x2)
end

function bench_observational(rng, n)
    x1, x2 = randn(rng, n), randn(rng, n)
    p = 1 ./ (1 .+ exp.(-(0.5 .* x1 .- 0.5 .* x2)))
    d = Int.(rand(rng, n) .< p)
    return DataFrame(y=x1 .+ x2 .+ d .+ randn(rng, n), d=d, x1=x1, x2=x2)
end

"""Random geometric network with complete randomization of half the units."""
function bench_network(rng, n)
    xs, ys = rand(rng, n), rand(rng, n)
    r = sqrt(4 / (π * n))                      # about four neighbors on average
    src, dst = Int[], Int[]
    for i in 1:n, j in (i + 1):n
        if (xs[i] - xs[j])^2 + (ys[i] - ys[j])^2 < r^2
            push!(src, i)
            push!(dst, j)
        end
    end
    g = NetworkStructure(collect(1:n), DataFrame(source=src, target=dst))
    design = CompleteRandomization(n, n ÷ 2)
    z = shuffle(rng, vcat(ones(Int, n ÷ 2), zeros(Int, n - n ÷ 2)))
    df = DataFrame(unit=1:n, z=z, y=randn(rng, n) .+ z)
    return df, g, design
end

# ---------------------------------------------------------------------------------
# Suite
# ---------------------------------------------------------------------------------

const PANEL = Dict(:small => bench_panel(StableRNG(1), 200, 10),
                   :medium => bench_panel(StableRNG(2), 2_000, 12))
const IVD = Dict(:small => bench_iv(StableRNG(3), 1_000),
                 :medium => bench_iv(StableRNG(4), 20_000))
const RDD = Dict(:small => bench_rd(StableRNG(5), 1_000),
                 :medium => bench_rd(StableRNG(6), 20_000))
const SYN = Dict(:small => bench_synth(StableRNG(7), 30, 20),
                 :medium => bench_synth(StableRNG(8), 100, 40))
const EXP = Dict(:small => bench_experiment(StableRNG(9), 200),
                 :medium => bench_experiment(StableRNG(10), 2_000))
const OBS = Dict(:small => bench_observational(StableRNG(11), 1_000),
                 :medium => bench_observational(StableRNG(12), 10_000))
const NET = Dict(:small => bench_network(StableRNG(13), 200),
                 :medium => bench_network(StableRNG(14), 1_000))

for area in ("did", "iv", "rd", "synth", "ri", "ml", "sutva")
    SUITE[area] = BenchmarkGroup([area])
end

for size in keys(SIZES)
    s = string(size)
    p, iv, rd, sy = PANEL[size], IVD[size], RDD[size], SYN[size]
    ex, ob = EXP[size], OBS[size]
    net, g, design = NET[size]

    SUITE["did"]["did_twfe", s] =
        @benchmarkable did_twfe($p, :y, :d, :unit, :time; warn_heterogeneity=false)
    SUITE["did"]["did_callaway_santanna", s] =
        @benchmarkable did_callaway_santanna($p, :y, :d, :unit, :time;
                                             rng=StableRNG(1))
    SUITE["did"]["event_study", s] =
        @benchmarkable quiet(() -> event_study($p, :y, :d, :unit, :time;
                                               estimator=:twfe))

    r_iv = iv_regression(iv, :y, :d, :z; covariates=[:x])
    SUITE["iv"]["iv_regression", s] =
        @benchmarkable iv_regression($iv, :y, :d, :z; covariates=[:x])
    SUITE["iv"]["weak_iv_confidence_set_ar", s] =
        @benchmarkable weak_iv_confidence_set($r_iv; method=:ar)
    r_iv0 = iv_regression(iv, :y, :d, :z; covariates=[:x], vcov=Vcov.simple())
    SUITE["iv"]["weak_iv_confidence_set_clr", s] =
        @benchmarkable weak_iv_confidence_set($r_iv0; method=:clr)

    SUITE["rd"]["rd_estimate", s] = @benchmarkable rd_estimate($rd, :y, :x)
    SUITE["rd"]["rd_bandwidth", s] = @benchmarkable rd_bandwidth($rd, :y, :x)

    SUITE["synth"]["synthetic_did", s] =
        @benchmarkable synthetic_did($sy, :y, :d, :unit, :time; se_method=:none)

    SUITE["ri"]["randomization_test", s] =
        @benchmarkable randomization_test($ex, :y, :d; nperm=1_000, rng=StableRNG(1),
                                          threaded=false)

    SUITE["ml"]["dml_irm", s] =
        @benchmarkable dml_irm($ob, :y, :d; covariates=[:x1, :x2],
                               outcome_learner=OLSLearner(),
                               propensity_learner=LogisticLearner(),
                               rng=StableRNG(1))

    P = exposure_probabilities(g, design; draws=2_000, rng=StableRNG(2))
    SUITE["sutva"]["exposure_probabilities", s] =
        @benchmarkable exposure_probabilities($g, $design; draws=2_000,
                                              rng=StableRNG(2))
    SUITE["sutva"]["exposure_effects", s] =
        @benchmarkable exposure_effects($net, :y, :z, $g, $P; unit=:unit,
                                        positivity=:restrict)
end
