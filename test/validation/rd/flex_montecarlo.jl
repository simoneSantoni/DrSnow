# Monte Carlo study of rd_flex (flexible ML covariate adjustment in RD): bias,
# standard deviation, mean standard error, coverage and length of the robust 95% CI,
# relative to rd_estimate without covariates and with linear covariate adjustment.
#
#   julia -t 4 --project=<env with DrSnow, StableRNGs> test/validation/rd/flex_montecarlo.jl
#
# Writes flex_montecarlo.csv (the numbers quoted in docs/src/rd.md).

using DrSnow, DataFrames, Statistics, StableRNGs, Printf

const REPS = parse(Int, get(ENV, "FLEX_MC_REPS", "1000"))

# Sharp design: the covariates act mostly non-linearly (effect 0.5 at the cutoff).
function dgp_sharp(rng, n)
    x = 2 .* rand(rng, n) .- 1
    Z = randn(rng, n, 3)
    y = 0.5 .* (x .>= 0) .+ 0.8 .* x .+ 1.5 .* sin.(2 .* Z[:, 1]) .+
        (Z[:, 2] .^ 2 .- 1) .+ 0.5 .* Z[:, 3] .+ 0.5 .* randn(rng, n)
    return DataFrame(x=x, y=y, z1=Z[:, 1], z2=Z[:, 2], z3=Z[:, 3])
end

# Sharp design with linear covariate effects (linear adjustment is already optimal).
function dgp_linear(rng, n)
    x = 2 .* rand(rng, n) .- 1
    Z = randn(rng, n, 3)
    y = 0.5 .* (x .>= 0) .+ 0.8 .* x .+ Z * [1.0, -0.8, 0.5] .+ 0.5 .* randn(rng, n)
    return DataFrame(x=x, y=y, z1=Z[:, 1], z2=Z[:, 2], z3=Z[:, 3])
end

# Fuzzy design (LATE 1.0): take-up depends on z1; outcome non-linear in the covariates.
function dgp_fuzzy(rng, n)
    x = 2 .* rand(rng, n) .- 1
    Z = randn(rng, n, 3)
    v = randn(rng, n)
    d = Float64.(0.2 .+ 0.6 .* (x .>= 0) .+ 0.3 .* Z[:, 1] .+ 0.5 .* v .> 0.5)
    y = 1.0 .* d .+ 0.8 .* x .+ 1.5 .* sin.(2 .* Z[:, 1]) .+ (Z[:, 2] .^ 2 .- 1) .+
        0.5 .* Z[:, 3] .+ 0.3 .* v .+ 0.5 .* randn(rng, n)
    return DataFrame(x=x, y=y, d=d, z1=Z[:, 1], z2=Z[:, 2], z3=Z[:, 3])
end

const ZS = [:z1, :z2, :z3]

function methods(df, rng, fuzzy)
    t = fuzzy ? :d : nothing
    return [
        "rd_estimate, no covariates" => rd_estimate(df, :y, :x; treatment=t),
        "rd_estimate, linear covariates" =>
            rd_estimate(df, :y, :x; treatment=t, covariates=ZS),
        "rd_flex, OLS" => rd_flex(df, :y, :x; covariates=ZS, treatment=t,
                                  outcome_learner=OLSLearner(),
                                  treatment_learner=LogisticLearner(), rng=rng),
        "rd_flex, random forest" =>
            rd_flex(df, :y, :x; covariates=ZS, treatment=t,
                    outcome_learner=ForestLearner(num_trees=300),
                    treatment_learner=ForestLearner(num_trees=300), rng=rng)]
end

rows = DataFrame(design=String[], n=Int[], method=String[], bias=Float64[],
                 sd=Float64[], mean_se=Float64[], coverage=Float64[],
                 ci_length=Float64[])
for (design, dgp, truth, fuzzy, n) in (("sharp, non-linear", dgp_sharp, 0.5, false, 1000),
                                       ("sharp, non-linear", dgp_sharp, 0.5, false, 4000),
                                       ("sharp, linear", dgp_linear, 0.5, false, 1000),
                                       ("fuzzy, non-linear", dgp_fuzzy, 1.0, true, 3000))
    res = Dict{String,Vector{NTuple{3,Float64}}}()
    labels = String[]
    for rep in 1:REPS
        rng = StableRNG(hash((design, n, rep)))
        for (lab, r) in methods(dgp(rng, n), rng, fuzzy)
            haskey(res, lab) || (res[lab] = NTuple{3,Float64}[]; push!(labels, lab))
            ci = confint(r)
            push!(res[lab], (coef(r)[1], stderror(r)[1], ci[1, 2] - ci[1, 1]))
        end
    end
    for lab in labels
        e = first.(res[lab])
        s = getindex.(res[lab], 2)
        L = last.(res[lab])
        cov = mean(abs.(e .- truth) .<= L ./ 2)
        push!(rows, (design, n, lab, mean(e) - truth, std(e), mean(s), cov, mean(L)))
    end
end
show(stdout, MIME"text/plain"(), rows; allrows=true)
println()
open(joinpath(@__DIR__, "flex_montecarlo.csv"), "w") do io
    println(io, join(names(rows), ","))
    for r in eachrow(rows)
        println(io, join(["\"$(r.design)\"", r.n, "\"$(r.method)\"",
                          (@sprintf("%.4f", v) for v in (r.bias, r.sd, r.mean_se,
                                                         r.coverage, r.ci_length))...],
                         ","))
    end
end
