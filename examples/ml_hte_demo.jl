# Heterogeneous treatment effects with DrSnow: causal forests (generalized random
# forests), meta-learners, and pre-specified subgroup / interaction analyses.
#
#   julia --threads=auto --project=. examples/ml_hte_demo.jl
#
# Everything is pure Julia. If a Makie backend is installed (`using CairoMakie`), the
# last section saves plots of variable importance, CATEs and the TOC curve.

using DrSnow
using DataFrames
using Random
using Statistics

show_result(x) = (show(stdout, MIME"text/plain"(), x); println("\n"))
section(t) = (println("=" ^ 76); println(t); println("=" ^ 76))

# Observational study: confounded treatment, effect increasing in x1 and x2
# (Wager & Athey 2018), outcome depending on x3 and x4, two nuisance covariates.
rng = Xoshiro(2026)
n = 3000
X = rand(rng, n, 6)
xs = [Symbol("x", j) for j in 1:6]
df = DataFrame(X, xs)
ζ(u) = 1 + 1 / (1 + exp(-20 * (u - 1 / 3)))
df.tau = ζ.(X[:, 1]) .* ζ.(X[:, 2])
df.d = Float64.(rand(rng, n) .< 0.2 .+ 0.6 .* X[:, 3])
df.y = 2 .* X[:, 3] .+ X[:, 4] .+ df.tau .* df.d .+ randn(rng, n)
df.school = rand(rng, 1:150, n)
df.region = ifelse.(X[:, 1] .> 0.5, "north", "south")
println("True ATE: ", round(mean(df.tau); digits=3))

section("1. Causal forest (honest, clustered by school)")
cf = causal_forest(df, :y, :d; covariates=xs, cluster=:school, num_trees=2000,
                   rng=Xoshiro(1))
show_result(cf)
show_result(average_treatment_effect(cf))
show_result(average_treatment_effect(cf; target=:treated))
show_result(average_treatment_effect(cf; target=:overlap))
println("Correlation of out-of-bag CATEs with the truth: ",
        round(cor(predict(cf), df.tau); digits=3))
grid = DataFrame(x1=0.1:0.2:0.9, x2=0.5, x3=0.5, x4=0.5, x5=0.5, x6=0.5)
pi = predict_interval(cf, grid)
pi.truth = ζ.(grid.x1) .* ζ.(0.5)
println("CATE along x1 with pointwise 95% intervals:")
println(pi[:, [:estimate, :conf_low, :conf_high, :truth]], "\n")

section("2. Is there heterogeneity? Calibration, projection, importance, RATE")
show_result(test_calibration(cf))
show_result(best_linear_projection(cf, [:x1, :x2, :x3]))
println(sort(variable_importance(cf), :importance; rev=true), "\n")
# RATE: learn priorities on one half, evaluate on the other
half = isodd.(1:n)
cf_a = causal_forest(df[half, :], :y, :d; covariates=xs, num_trees=1000,
                     rng=Xoshiro(2))
cf_b = causal_forest(df[.!half, :], :y, :d; covariates=xs, num_trees=1000,
                     rng=Xoshiro(3))
rate = rank_average_treatment_effect(cf_b, predict(cf_a, df[.!half, :]); R=200,
                                     rng=Xoshiro(4))
show_result(rate)

section("3. Policy: treat when the estimated effect exceeds 2.5")
show_result(policy_value(cf_b, predict(cf_a, df[.!half, :]) .> 2.5))

section("4. Meta-learners (point predictions, cross-fitted)")
for (name, f) in (("S", s_learner), ("T", t_learner), ("X", x_learner),
                  ("R", r_learner))
    m = f(df, :y, :d; covariates=xs, rng=Xoshiro(5))
    println(rpad("$name-learner", 10), " plug-in ATE ", round(m.ate; digits=3),
            ", corr(cross-fitted CATE, truth) ", round(cor(m.cate_oof, df.tau); digits=3))
end
mr = r_learner(df, :y, :d; covariates=xs, effect_modifiers=[:x1, :x2],
               final_learner=RidgeLearner(), rng=Xoshiro(6))
println("\nR-learner with a ridge CATE in (x1, x2):")
show_result(mr)
b = metalearner_bootstrap(t_learner(df, :y, :d; covariates=xs,
                                    outcome_learner=OLSLearner(), rng=Xoshiro(7));
                          newdata=grid[[1, 5], :], B=100, rng=Xoshiro(8))
show_result(b)

section("5. Pre-specified subgroups and interactions")
sg = subgroup_effects(df, :y, :d, :region; method=:aipw, covariates=xs,
                      outcome_learner=ForestLearner(num_trees=200),
                      propensity_learner=ForestLearner(num_trees=200), cluster=:school,
                      rng=Xoshiro(9))
show_result(sg)
show_result(heterogeneity_test(sg))
show_result(subgroup_effects(cf, df.region))
ie = interaction_effects(df, :y, :d, [:x1, :x2]; method=:aipw, covariates=xs,
                         cluster=:school, rng=Xoshiro(10))
show_result(ie)

section("6. Instrumental forest: heterogeneous LATE")
z = Float64.(rand(rng, n) .< 0.5)
u = rand(rng, n)
df.z = z
df.take = Float64.(ifelse.(z .== 1, u .< 0.5 .+ 0.3 .* X[:, 2], u .< 0.1))
df.yiv = 2 .* X[:, 3] .+ (1 .+ X[:, 1]) .* df.take .+ randn(rng, n)
ivf = instrumental_forest(df, :yiv, :take, :z; covariates=xs, num_trees=1000,
                          rng=Xoshiro(11))
show_result(average_treatment_effect(ivf))
show_result(best_linear_projection(ivf, [:x1]))

if Base.find_package("CairoMakie") !== nothing
    section("7. Plots")
    @eval using CairoMakie
    outdir = mktempdir()
    Base.invokelatest() do
        save(joinpath(outdir, "variable_importance.png"), plot_variable_importance(cf))
        save(joinpath(outdir, "cate_x1.png"), plot_cate(cf; modifier=:x1))
        save(joinpath(outdir, "toc.png"), plot_rate(rate))
    end
    println("Plots written to ", outdir)
end
