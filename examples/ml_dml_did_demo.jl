# Staggered difference-in-differences with machine-learning nuisances
# (dml_did_multi) on simulated data.
#
#   julia --project=. examples/ml_dml_did_demo.jl
#
# Cohorts adopt a treatment in periods 3, 4 and 5 (plus never-treated units).
# Selection into cohorts and the untreated outcome trends depend non-linearly on two
# covariates, so parallel trends hold only conditionally on them and linear working
# models for the outcome trend and the cohort propensity are misspecified.

using DrSnow
using DataFrames
using Random
using Statistics

rng = Xoshiro(2027)
att(g, t) = t >= g ? 1.0 + 0.5 * (t - g) : 0.0

N, T = 2000, 5
x1 = randn(rng, N)
x2 = randn(rng, N)
f = 0.8 .* (x1 .^ 2 .- 1) .+ sin.(2 .* x2)
cohort = map(f) do fi
    s = [0.0, 0.5fi - 0.2, 0.4fi, 0.5fi - 0.1]
    p = cumsum(exp.(s) ./ sum(exp.(s)))
    (0, 3, 4, 5)[min(4, searchsortedfirst(p, rand(rng)))]
end
α = x1 .+ randn(rng, N)
df = DataFrame([(id=i, year=t, first_treat=cohort[i], x1=x1[i], x2=x2[i],
                 y=α[i] + 0.2t + 0.3t * f[i] +
                   (cohort[i] > 0 ? att(cohort[i], t) : 0.0) + randn(rng))
                for i in 1:N for t in 1:T])
ft = FirstTreated(:first_treat)

println("=" ^ 72)
println("1. ATT(g,t) with random-forest nuisances (5-fold cross-fitting)")
println("=" ^ 72)
r = dml_did_multi(df, :y, ft, :id, :year; covariates=[:x1, :x2],
                  outcome_learner=ForestLearner(num_trees=300),
                  propensity_learner=ForestLearner(num_trees=300), rng=Xoshiro(1))
show(stdout, MIME"text/plain"(), r)
println()
println("\nOut-of-fold nuisance losses per cell (RMSE of g0, log loss of m):")
show(stdout, MIME"text/plain"(), r.settings.nuisance_loss)
println()

println("\n", "=" ^ 72)
println("2. Aggregations (influence-function SEs, multiplier-bootstrap bands)")
println("=" ^ 72)
es = aggregate_att(r, :dynamic; rng=Xoshiro(2))
show(stdout, MIME"text/plain"(), es)
println("\nSimultaneous 95% band for the event study:")
display(round.(confint(es; uniform=true); digits=3))
for typ in (:simple, :group, :calendar)
    a = aggregate_att(r, typ; bootstrap=false)
    println("\n", typ, ": ", join(["$(n) = $(round(b; digits=3)) (se $(round(s; digits=3)))"
                                  for (n, b, s) in zip(coefnames(a), coef(a),
                                                       stderror(a))], "; "))
end
println("\n", pre_trend_test(r))

println("\n", "=" ^ 72)
println("3. Comparison: linear working models vs forests (true simple ATT)")
println("=" ^ 72)
truth = let rt = r
    b = att.(rt.groups, rt.times)
    s = CallawaySantAnnaEstimate((k === :coef ? b : getfield(rt, k)
                                  for k in fieldnames(CallawaySantAnnaEstimate))...)
    coef(aggregate_att(s, :simple; bootstrap=false))[1]
end
fits = ["Callaway–Sant'Anna DR (linear/logit)" =>
            did_callaway_santanna(df, :y, ft, :id, :year; covariates=[:x1, :x2],
                                  bootstrap=false),
        "dml_did_multi (OLS/logit)" =>
            dml_did_multi(df, :y, ft, :id, :year; covariates=[:x1, :x2],
                          outcome_learner=OLSLearner(),
                          propensity_learner=LogisticLearner(), rng=Xoshiro(3),
                          bootstrap=false),
        "dml_did_multi (random forests)" => r]
println("true simple ATT (cohort-share weights): ", round(truth; digits=3))
for (lab, fit) in fits
    a = aggregate_att(fit, :simple; bootstrap=false)
    ci = confint(a)
    println(rpad(lab, 40), round(coef(a)[1]; digits=3), "  95% CI [",
            round(ci[1, 1]; digits=3), ", ", round(ci[1, 2]; digits=3), "]")
end

println("\n", "=" ^ 72)
println("4. Repeated cross-sections and not-yet-treated comparison units")
println("=" ^ 72)
rc = df[df.year .== rand(Xoshiro(4), 1:T, N)[df.id], :]     # one period per unit
r_rc = dml_did_multi(rc, :y, ft, nothing, :year; covariates=[:x1, :x2],
                     control_group=:not_yet_treated,
                     outcome_learner=ForestLearner(num_trees=200),
                     propensity_learner=ForestLearner(num_trees=200), rng=Xoshiro(5),
                     bootstrap=false)
show(stdout, MIME"text/plain"(), aggregate_att(r_rc, :group; bootstrap=false))
println()
