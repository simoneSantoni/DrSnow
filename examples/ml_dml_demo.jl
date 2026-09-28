# Causal machine learning with DrSnow: a tour of the `ml` area on simulated data.
#
#   julia --project=. examples/ml_dml_demo.jl
#
# Uses only the built-in pure-Julia learners. If MLJ (or MLJModelInterface plus a
# model package such as MLJDecisionTreeInterface) is installed in the active
# environment, the last section also fits a random-forest nuisance via MLJLearner.

using DrSnow
using DataFrames
using Random
using Statistics

rng = Xoshiro(2026)
logistic(x) = 1 / (1 + exp(-x))

println("=" ^ 72)
println("1. Double/debiased ML: partially linear and interactive models")
println("=" ^ 72)

n = 2000
X = randn(rng, n, 10)
xs = [Symbol("x", j) for j in 1:10]
df = DataFrame(X, xs)
# confounded binary treatment with heterogeneous effect τ(x) = 1 + x1
m = logistic.(0.6 .* X[:, 1] .- 0.4 .* X[:, 2] .+ 0.3 .* X[:, 3])
df.d = Float64.(rand(rng, n) .< m)
df.y = df.d .* (1 .+ X[:, 1]) .+ X[:, 1] .+ 0.5 .* X[:, 2] .^ 2 .+ sin.(X[:, 3]) .+
       randn(rng, n)
# continuous treatment for the partially linear model (θ = 0.5)
df.dc = 0.7 .* X[:, 1] .+ 0.3 .* X[:, 4] .+ randn(rng, n)
df.yc = 0.5 .* df.dc .+ X[:, 1] .+ 0.5 .* X[:, 2] .^ 2 .+ randn(rng, n)

plr = dml_plr(df, :yc, :dc; covariates=xs, n_rep=3, rng=Xoshiro(1))
show(stdout, MIME"text/plain"(), plr)
println("\n")

irm = dml_irm(df, :y, :d; covariates=xs, score=:ATE, n_rep=3, rng=Xoshiro(2))
show(stdout, MIME"text/plain"(), irm)
println("\nNuisance fit (out of fold):")
println(nuisance_loss(irm))
println()

println("=" ^ 72)
println("2. LATE with machine-learned nuisances (interactive IV model)")
println("=" ^ 72)
z = Float64.(rand(rng, n) .< logistic.(0.5 .* X[:, 1]))
v = randn(rng, n)
df.z = z
df.dz = Float64.((-0.3 .+ 1.3 .* z .+ 0.4 .* X[:, 2] .+ v) .> 0.4)
df.yz = 0.8 .* df.dz .+ X[:, 1] .+ 0.5 .* X[:, 3] .+ 0.6 .* v .+ randn(rng, n)
late = dml_iivm(df, :yz, :dz, :z; covariates=xs, rng=Xoshiro(3))
show(stdout, MIME"text/plain"(), late)
println("\n")

println("=" ^ 72)
println("3. DML difference-in-differences (2×2 panel)")
println("=" ^ 72)
nu = 1000
Xd = randn(rng, nu, 3)
grp = Float64.(rand(rng, nu) .< logistic.(0.6 .* Xd[:, 1]))
α = randn(rng, nu)
y0 = α .+ Xd[:, 2] .+ randn(rng, nu)
y1 = α .+ Xd[:, 2] .+ 1.0 .+ 0.8 .* Xd[:, 1] .+ 1.5 .* grp .+ randn(rng, nu)
panel = vcat(DataFrame(id=1:nu, year=2020, y=y0, treated=grp, x1=Xd[:, 1],
                       x2=Xd[:, 2], x3=Xd[:, 3]),
             DataFrame(id=1:nu, year=2021, y=y1, treated=grp, x1=Xd[:, 1],
                       x2=Xd[:, 2], x3=Xd[:, 3]))
did = dml_did(panel, :y, :treated; time=:year, unit=:id, covariates=[:x1, :x2, :x3],
              rng=Xoshiro(4))
show(stdout, MIME"text/plain"(), did)
println("\n")

println("=" ^ 72)
println("4. Heterogeneous effects: DR-learner and its linear projection")
println("=" ^ 72)
cate = cate_dr_learner(df, :y, :d; covariates=xs, effect_modifiers=[:x1, :x2],
                       rng=Xoshiro(5))
show(stdout, MIME"text/plain"(), cate)
println("Predicted CATE at x1 = -1, 0, 1 (x2 = 0): ",
        round.(predict(cate, DataFrame(x1=[-1.0, 0.0, 1.0], x2=0.0)); digits=3))
show(stdout, MIME"text/plain"(), coeftable(cate_projection(cate)))
println("\n")

println("=" ^ 72)
println("5. Generic ML inference in a randomized experiment (BLP / GATES / CLAN)")
println("=" ^ 72)
rct = DataFrame(X[:, 1:5], xs[1:5])
rct.d = Float64.(rand(rng, n) .< 0.5)
rct.y = rct.d .* (1 .+ X[:, 1]) .+ X[:, 2] .+ randn(rng, n)
gml = generic_ml(rct, :y, :d; covariates=xs[1:5], n_splits=30, rng=Xoshiro(6))
show(stdout, MIME"text/plain"(), gml)
println()
show(stdout, MIME"text/plain"(), blp_test(gml))
println()

println("=" ^ 72)
println("6. Policy learning with a depth-2 doubly-robust tree")
println("=" ^ 72)
pol = policy_tree(df, :y, :d; covariates=xs, policy_covariates=[:x1, :x2, :x3],
                  depth=2, split_step=10, rng=Xoshiro(7))
show(stdout, MIME"text/plain"(), pol)
println("\n")

println("=" ^ 72)
println("7. Prediction-powered inference for an ML-coded outcome in an experiment")
println("=" ^ 72)
rng = Xoshiro(107)
N = 5000
nl = 300
treat = Float64.(rand(rng, N + nl) .< 0.5)
outcome = 1 .+ 0.4 .* treat .+ randn(rng, N + nl)
predicted = 0.8 .* outcome .+ 0.3 .+ 0.4 .* randn(rng, N + nl)   # biased classifier
labeled = DataFrame(treat=treat[1:nl], y=outcome[1:nl], y_hat=predicted[1:nl])
unlabeled = DataFrame(treat=treat[(nl + 1):end], y_hat=predicted[(nl + 1):end])
ppi = ppi_ols(labeled, unlabeled, :y, :y_hat; covariates=[:treat])
show(stdout, MIME"text/plain"(), ppi)
println("\n")

println("=" ^ 72)
println("8. Plugging in an MLJ model (optional)")
println("=" ^ 72)
has_mlj = Base.find_package("MLJModelInterface") !== nothing &&
          Base.find_package("MLJDecisionTreeInterface") !== nothing
if has_mlj
    # `import` (rather than `using`) keeps MLJ's own `predict` out of Main
    import MLJModelInterface, MLJDecisionTreeInterface
    forest = Base.invokelatest(() -> MLJLearner(
        MLJDecisionTreeInterface.RandomForestRegressor(n_trees=100)))
    rf = Base.invokelatest(dml_plr, df, :yc, :dc; covariates=xs,
                           outcome_learner=forest, treatment_learner=forest,
                           rng=Xoshiro(8))
    show(stdout, MIME"text/plain"(), rf)
    println()
else
    println("MLJModelInterface / MLJDecisionTreeInterface not installed; skipping.")
    println("With MLJ: `using MLJ; Tree = @load RandomForestRegressor pkg=DecisionTree;`")
    println("`dml_plr(df, :y, :d; covariates=xs, outcome_learner=MLJLearner(Tree()))`")
end
