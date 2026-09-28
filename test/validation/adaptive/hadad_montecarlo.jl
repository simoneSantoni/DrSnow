# Monte Carlo replication of the multi-armed bandit simulations of Hadad, Hirshberg,
# Zhan, Wager & Athey (2021, PNAS, Figs. 2-4): K = 3 arms with means
#   no signal (1, 1, 1), low signal (0.9, 1, 1.1), high signal (0.5, 1, 1.5),
# uniform noise on [-1, 1], Gaussian Thompson sampling with estimated noise variance
# and N(0, 1) priors, assignment-probability floor (1/3) t^-0.7, an equal-allocation
# burn-in of 15 units, and 90% intervals. (Differences from the authors' code: the
# burn-in is randomized rather than round robin, and the Thompson probabilities are
# computed by numerical integration rather than from 20 Monte Carlo draws.)
#
# Writes hadad_montecarlo.csv: design, T, method, target, coverage, bias, rmse,
# mean CI width, and the number of replications.
#
# Run (multithreaded; results do not depend on the number of threads):
#   julia -t auto --project=<env with DrSnow, DataFrames, CSV, StableRNGs> \
#       hadad_montecarlo.jl [reps] [T1,T2,...] [output.csv]
# The committed tables: `2000 1000,5000` -> hadad_montecarlo.csv and
# `1000 20000 hadad_montecarlo_T20000.csv`.

using DrSnow, DataFrames, CSV, StableRNGs, Random, Statistics

const REPS = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 1000
const TS = length(ARGS) >= 2 ? parse.(Int, split(ARGS[2], ",")) : [1000, 5000]
const OUT = length(ARGS) >= 3 ? ARGS[3] : "hadad_montecarlo.csv"
const DESIGNS = ["no signal" => [1.0, 1.0, 1.0], "low signal" => [0.9, 1.0, 1.1],
                 "high signal" => [0.5, 1.0, 1.5]]
const LEVEL = 0.9

function one_rep(truth, T, seed)
    rng = Random.Xoshiro(seed)
    K = length(truth)
    Y = truth' .+ (2 .* rand(rng, T, K) .- 1)
    pol = GaussianThompson(K; floor=1 / K, floor_decay=0.7, burnin=5K)
    log = run_adaptive_experiment(pol, Y; rng=rng)
    fits = ["two_point" => adaptive_arm_values(log; weights=:two_point, reference=3),
            "constant_allocation" => adaptive_arm_values(log;
                                         weights=:constant_allocation, reference=3),
            "uniform" => adaptive_arm_values(log; weights=:uniform, reference=3),
            "sample_mean" => naive_arm_means(log; reference=3)]
    # targets: arm values and the contrasts arm 1 - arm 3, arm 2 - arm 3
    tv = vcat(truth, truth[1] - truth[3], truth[2] - truth[3])
    out = NamedTuple[]
    for (m, r) in fits
        b = coef(r)
        ci = confint(r; level=LEVEL)
        for j in eachindex(tv)
            push!(out, (method=m, target=coefnames(r)[j], err=b[j] - tv[j],
                        cover=ci[j, 1] <= tv[j] <= ci[j, 2], width=ci[j, 2] - ci[j, 1]))
        end
    end
    return out
end

rows = DataFrame()
for (di, (dname, truth)) in enumerate(DESIGNS), T in TS
    seeds = DrSnow.task_seeds(StableRNG(1_000_000 * di + T), REPS)
    res = Vector{Vector{NamedTuple}}(undef, REPS)
    Threads.@threads for i in 1:REPS
        res[i] = one_rep(truth, T, seeds[i])
    end
    df = DataFrame(reduce(vcat, res))
    g = combine(groupby(df, [:method, :target]), :cover => mean => :coverage,
                :err => mean => :bias, :err => (e -> sqrt(mean(abs2, e))) => :rmse,
                :width => mean => :ci_width)
    g.design .= dname
    g.T .= T
    g.reps .= REPS
    append!(rows, g)
    println(dname, " T=", T, " done")
end
select!(rows, :design, :T, :method, :target, :coverage, :bias, :rmse, :ci_width, :reps)
CSV.write(joinpath(@__DIR__, OUT), rows)
show(stdout, MIME"text/plain"(), rows; allrows=true)
