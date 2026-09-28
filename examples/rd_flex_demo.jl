# Regression discontinuity with flexible machine-learning covariate adjustment
# (rd_flex; Noack, Olma & Rothe 2024) on simulated data.
#
#   julia --project=. examples/rd_flex_demo.jl
#
# The covariates are predetermined (no jump at the cutoff) and explain much of the
# outcome, mostly through non-linear terms: linear adjustment (rd_estimate with
# `covariates`) gains little, a flexible learner gains a lot.

using DrSnow
using DataFrames
using Random
using Statistics

rng = Xoshiro(2028)
n = 3000
x = 2 .* rand(rng, n) .- 1
Z = randn(rng, n, 3)
df = DataFrame(x=x, z1=Z[:, 1], z2=Z[:, 2], z3=Z[:, 3])
df.y = 0.5 .* (x .>= 0) .+ 0.8 .* x .+ 1.5 .* sin.(2 .* df.z1) .+ (df.z2 .^ 2 .- 1) .+
       0.5 .* df.z3 .+ 0.5 .* randn(rng, n)
zs = [:z1, :z2, :z3]

println("=" ^ 72)
println("1. Covariates are predetermined: no jump at the cutoff")
println("=" ^ 72)
show(stdout, MIME"text/plain"(), rd_covariate_balance(df, zs, :x))
println()

println("\n", "=" ^ 72)
println("2. Sharp RD (true effect 0.5): no, linear and flexible adjustment")
println("=" ^ 72)
fits = ["rd_estimate, no covariates" => rd_estimate(df, :y, :x),
        "rd_estimate, linear covariates" => rd_estimate(df, :y, :x; covariates=zs),
        "rd_flex, lasso" => rd_flex(df, :y, :x; covariates=zs,
                                    outcome_learner=LassoLearner(), rng=Xoshiro(1)),
        "rd_flex, random forest" =>
            rd_flex(df, :y, :x; covariates=zs,
                    outcome_learner=ForestLearner(num_trees=500), rng=Xoshiro(1))]
for (lab, r) in fits
    ci = confint(r)
    println(rpad(lab, 34), "estimate ", round(coef(r)[1]; digits=3), "  robust se ",
            round(stderror(r)[1]; digits=3), "  95% CI [", round(ci[1, 1]; digits=3),
            ", ", round(ci[1, 2]; digits=3), "]")
end
println()
show(stdout, MIME"text/plain"(), last(fits[end]))
println()

println("\n", "=" ^ 72)
println("3. Fuzzy RD (true LATE 1.0) with forest adjustment of outcome and take-up")
println("=" ^ 72)
v = randn(rng, n)
df.d = Float64.(0.2 .+ 0.6 .* (x .>= 0) .+ 0.3 .* df.z1 .+ 0.5 .* v .> 0.5)
df.yf = df.d .+ 0.8 .* x .+ 1.5 .* sin.(2 .* df.z1) .+ (df.z2 .^ 2 .- 1) .+
        0.5 .* df.z3 .+ 0.3 .* v .+ 0.5 .* randn(rng, n)
rf = rd_flex(df, :yf, :x; covariates=zs, treatment=:d,
             outcome_learner=ForestLearner(num_trees=500),
             treatment_learner=ForestLearner(num_trees=500), rng=Xoshiro(2))
r0 = rd_estimate(df, :yf, :x; treatment=:d)
println("rd_estimate (no covariates): ", round(coef(r0)[1]; digits=3), " (se ",
        round(stderror(r0)[1]; digits=3), ")")
println("rd_flex (random forests):    ", round(coef(rf)[1]; digits=3), " (se ",
        round(stderror(rf)[1]; digits=3), ")")
show(stdout, MIME"text/plain"(), rd_inference_table(rf))
println()
