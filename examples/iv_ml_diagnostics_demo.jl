# Machine-learning-era IV diagnostics with DrSnow: residual prediction specification
# tests, cross-fitted first-stage strength and weak-IV-robust DML inference, 2SLS
# weights with several instruments (Mogstad, Torgovitsky & Walters 2021), and
# distributional effects for compliers (DML local quantile treatment effects).
#
# Run from the repository root:
#     julia --project=. examples/iv_ml_diagnostics_demo.jl
#
# Every data set below is simulated, so the true parameters are known and printed
# next to the estimates. Printed test results state statistics and p-values only.

using DrSnow, DataFrames, Random, Statistics, Printf

rng = Xoshiro(20260928)
section(title) = (println(); println("="^78); println(title); println("="^78))
show_plain(x) = (show(stdout, MIME"text/plain"(), x); println())

# ----------------------------------------------------------------------------
# 1. Residual prediction specification tests (Scheidegger et al. 2025)
# ----------------------------------------------------------------------------
section("1. Is the linear IV model well specified? (true β = −1)")
function spec_data(rng; n=1000, violation=0.0, pi=1.0)
    Z = randn(rng, n, 2)
    c = 0.5 .* Z[:, 1] .+ randn(rng, n)
    h = randn(rng, n)                                     # unobserved confounder
    x = pi .* tanh.((Z[:, 1] .+ Z[:, 2]) ./ sqrt(2)) .+ 0.3 .* c .+ h .+ randn(rng, n)
    e = randn(rng, n) .* abs.(Z[:, 1])                     # heteroskedastic
    y = 2 .- x .+ 0.5 .* c .- h .+ e .+ violation .* Z[:, 1] .^ 2
    return DataFrame(y=y, x=x, c=c, z1=Z[:, 1], z2=Z[:, 2])
end
good = spec_data(rng)
bad = spec_data(rng; violation=0.5)        # the instrument enters Y nonlinearly
for (label, df) in (("valid instruments", good), ("direct effect 0.5 z1²", bad))
    t = residual_prediction_test(df, :y, :x, [:z1, :z2]; covariates=[:c],
                                 learner=ForestLearner(num_trees=200), rng=Xoshiro(1))
    @printf("%-24s T = %6.2f, p = %.3g (2SLS on main sample: %.3f)\n", label,
            t.statistic, t.pvalue, t.details.beta[1])
end
println("\nWeak-IV-robust confidence set by inversion (an empty set rejects the model):")
for (label, df) in (("valid instruments", good), ("direct effect 0.5 z1²", bad))
    cs = residual_prediction_confidence_set(df, :y, :x, [:z1, :z2]; covariates=[:c],
                                            learner=ForestLearner(num_trees=200),
                                            weight_update=:linear, rng=Xoshiro(2))
    print(label, ": ")
    show_plain(cs)
end

# ----------------------------------------------------------------------------
# 2. Cross-fitted first stage and the DML Anderson–Rubin set
# ----------------------------------------------------------------------------
section("2. DML partially linear IV with a weak instrument (true θ = 1)")
n = 1000
x1, x2 = randn(rng, n), randn(rng, n)
z = 0.5 .* x1 .+ randn(rng, n)
v = randn(rng, n)
u = 0.9 .* v .+ sqrt(1 - 0.81) .* randn(rng, n)
d = 0.08 .* z .+ sin.(x1) .+ v                            # weak first stage
y = 1.0 .* d .+ x1 .^ 2 .- x2 .+ u
pl = DataFrame(y=y, d=d, z=z, x1=x1, x2=x2)
r = dml_pliv(pl, :y, :d, :z; covariates=[:x1, :x2], outcome_learner=ForestLearner(),
             treatment_learner=ForestLearner(), instrument_learner=ForestLearner(),
             rng=Xoshiro(3))
@printf("DML estimate %.3f, Wald 95%% CI [%.3f, %.3f]\n", coef(r)[1], confint(r)...)
show_plain(ml_first_stage(r, pl; instrument=:z))
show_plain(dml_weak_iv_confidence_set(r))
show_plain(dml_weak_iv_test(r; beta0=1.0))

# ----------------------------------------------------------------------------
# 3. Two binary instruments: which compliers get negative 2SLS weight?
# ----------------------------------------------------------------------------
section("3. 2SLS with two negatively correlated binary instruments (MTW 2021)")
# Two encouragement arms that are rarely combined: Z = (1,0) or (0,1) mostly.
cells = [0 0; 0 1; 1 0; 1 1]
cellp = cumsum([0.03, 0.47, 0.47, 0.03])
groups = [(name="always", D=[1, 1, 1, 1], share=0.10, effect=0.0),
          (name="never", D=[0, 0, 0, 0], share=0.20, effect=0.0),
          (name="eager", D=[0, 1, 1, 1], share=0.10, effect=1.0),
          (name="reluctant", D=[0, 0, 0, 1], share=0.05, effect=1.0),
          (name="Z1 compliers", D=[0, 0, 1, 1], share=0.35, effect=0.5),
          (name="Z2 compliers", D=[0, 1, 0, 1], share=0.20, effect=3.0)]
gp = cumsum([g.share for g in groups])
n = 20_000
k = [findfirst(>=(rand(rng)), cellp) for _ in 1:n]
g = [findfirst(>=(rand(rng)), gp) for _ in 1:n]
dd = [Float64(groups[g[i]].D[k[i]]) for i in 1:n]
yy = randn(rng, n) .+ dd .* [groups[g[i]].effect for i in 1:n]
mtw = DataFrame(y=yy, d=dd, z1=Float64.(cells[k, 1]), z2=Float64.(cells[k, 2]))
w = multiple_iv_weights(mtw, :d, [:z1, :z2]; outcome=:y, rng=Xoshiro(4))
show_plain(w)
complier_late = sum(gr.share * gr.effect for gr in groups[3:end]) /
                sum(gr.share for gr in groups[3:end])
@printf("Complier-share-weighted average effect: %.3f; 2SLS: %.3f\n", complier_late,
        w.tsls)

println("\nJudge design (five judges, one instrument): PM = IA monotonicity")
nj = 5000
judge = rand(rng, 1:5, nj)
lenient = [0.2, 0.35, 0.5, 0.6, 0.8]
uj = rand(rng, nj)
dj = Float64.(uj .< lenient[judge])
jd = DataFrame(y=dj .* (1 .+ uj) .+ randn(rng, nj), d=dj, judge=judge)
wj = multiple_iv_weights(jd, :d, :judge; outcome=:y, rng=Xoshiro(5))
println("Threshold groups: ", join(wj.groups.pattern, ", "),
        "; negative weight possible: ", wj.negative_weight_possible)

# ----------------------------------------------------------------------------
# 4. Distributional effects for compliers (DML LQTE and complier CDFs)
# ----------------------------------------------------------------------------
section("4. Local quantile treatment effects (true LQTE rises with τ)")
n = 3000
x1, x2 = randn(rng, n), randn(rng, n)
zb = Float64.(rand(rng, n) .< 1 ./ (1 .+ exp.(-0.6 .* x1)))
ut = rand(rng, n)
at, nt = ut .< 0.15, ut .> 0.75
db = Float64.(at .| (.!(at .| nt) .& (zb .== 1)))
y0 = 0.5 .* x1 .- 0.3 .* x2 .+ randn(rng, n)
y1 = y0 .+ 1.0 .+ 0.8 .* randexp(rng, n)                # skewed gains
qd = DataFrame(y=ifelse.(db .== 1, y1, y0), d=db, z=zb, x1=x1, x2=x2)
lq = dml_lqte(qd, :y, :d, :z; covariates=[:x1, :x2], quantiles=[0.1, 0.25, 0.5, 0.75, 0.9],
              outcome_learner=LogisticLearner(), treatment_learner=LogisticLearner(),
              instrument_learner=LogisticLearner(), rng=Xoshiro(6))
show_plain(lq)
bands = confint(lq; uniform=true)
println("95% uniform band over quantiles:")
for (j, τ) in enumerate(lq.quantiles)
    @printf("  τ = %.2f: [%.3f, %.3f]\n", τ, bands[j, 1], bands[j, 2])
end
cdfs = dml_complier_cdf(qd, :y, :d, :z; covariates=[:x1, :x2], grid=-2.0:1.0:4.0,
                        outcome_learner=LogisticLearner(),
                        treatment_learner=LogisticLearner(),
                        instrument_learner=LogisticLearner(), rng=Xoshiro(7))
show_plain(cdfs)
