# Generates the dataset used to validate DrSnow's generalized random forest
# post-estimation against the R package grf (see grf_reference.R), together with
# forest inputs (nuisance estimates, out-of-bag CATEs, compliance scores and
# debiasing weights) produced by DrSnow. grf's deterministic post-estimation
# formulas (AIPW average effects, best linear projection, calibration test, doubly
# robust scores, RATE / TOC point estimates) are evaluated in R on these same inputs,
# so the two implementations must agree to numerical precision.
#
#   julia --project=. test/validation/ml/make_grf_data.jl
#
# The CSV file is committed; this script documents how it was produced.

using DrSnow
using DataFrames
using Random

const DIR = @__DIR__

rng = Xoshiro(20260927)
n = 600
X = rand(rng, n, 5)
df = DataFrame(X, [:x1, :x2, :x3, :x4, :x5])
e = 0.2 .+ 0.6 .* X[:, 3]
df.w = Float64.(rand(rng, n) .< e)
df.y = X[:, 3] .+ (1 .+ 2 .* X[:, 1]) .* df.w .+ randn(rng, n)
df.cl = Float64.(rand(rng, 1:40, n))
df.sw = 0.5 .+ 1.5 .* rand(rng, n)
# continuous treatment
df.wc = X[:, 3] .+ randn(rng, n)
df.yc = X[:, 2] .+ (1 .+ X[:, 1]) .* df.wc .+ randn(rng, n)
# instrument with one-sided-ish non-compliance
df.z = Float64.(rand(rng, n) .< 0.5)
u = rand(rng, n)
df.d = Float64.(ifelse.(df.z .== 1, u .< 0.4 .+ 0.4 .* X[:, 2], u .< 0.1))
df.yiv = X[:, 3] .+ (1 .+ X[:, 1]) .* df.d .+ randn(rng, n)

xs = [:x1, :x2, :x3, :x4, :x5]
cf = causal_forest(df, :y, :w; covariates=xs, num_trees=500, rng=Xoshiro(1))
df.y_hat = cf.Y_hat
df.w_hat = cf.W_hat
df.tau = predict(cf)
cc = causal_forest(df, :yc, :wc; covariates=xs, num_trees=500, rng=Xoshiro(2))
df.yc_hat = cc.Y_hat
df.wc_hat = cc.W_hat
df.tauc = predict(cc)
vf = regression_forest(X, (df.wc .- df.wc_hat) .^ 2;
                       num_trees=200, ci_group_size=1, rng=Xoshiro(3))
df.gammac = (df.wc .- df.wc_hat) ./ predict(vf)
ivf = instrumental_forest(df, :yiv, :d, :z; covariates=xs, num_trees=500,
                          rng=Xoshiro(4))
df.yiv_hat = ivf.Y_hat
df.d_hat = ivf.W_hat
df.z_hat = ivf.Z_hat
df.tauiv = predict(ivf)
comp = causal_forest(df, :d, :z; covariates=xs, y_hat=ivf.W_hat, w_hat=ivf.Z_hat,
                     num_trees=500, rng=Xoshiro(5))
df.compliance = predict(comp)
df.prio2 = round.(4 .* X[:, 1])

open(joinpath(DIR, "grf_data.csv"), "w") do io
    println(io, join(names(df), ","))
    for r in eachrow(df)
        println(io, join((repr(Float64(v)) for v in r), ","))
    end
end

# Dataset for the statistical comparison of full forest fits (grf_forest_reference.R):
# confounded binary treatment with a smooth but strongly nonlinear CATE
# (Wager & Athey 2018, Section 5.2), plus a binary instrument, and test points.
rng = Xoshiro(20260928)
n, p = 1500, 6
X = rand(rng, n, p)
ζ(u) = 1 + 1 / (1 + exp(-20 * (u - 1 / 3)))
fd = DataFrame(X, [Symbol("x$j") for j in 1:p])
fd.w = Float64.(rand(rng, n) .< 0.3 .+ 0.4 .* X[:, 3])
fd.y = 2 .* X[:, 3] .+ X[:, 4] .+ ζ.(X[:, 1]) .* ζ.(X[:, 2]) .* fd.w .+ randn(rng, n)
fd.z = Float64.(rand(rng, n) .< 0.5)
u = rand(rng, n)
fd.d = Float64.(ifelse.(fd.z .== 1, u .< 0.5 .+ 0.3 .* X[:, 2], u .< 0.15))
fd.yiv = 2 .* X[:, 3] .+ (1 .+ X[:, 1]) .* fd.d .+ randn(rng, n)
Xt = rand(rng, 60, p)
tst = DataFrame(Xt, [Symbol("x$j") for j in 1:p])
tst.tau = ζ.(Xt[:, 1]) .* ζ.(Xt[:, 2])
for (name, d) in (("grf_forest_data.csv", fd), ("grf_forest_test.csv", tst))
    open(joinpath(DIR, name), "w") do io
        println(io, join(names(d), ","))
        for r in eachrow(d)
            println(io, join((repr(Float64(v)) for v in r), ","))
        end
    end
end
