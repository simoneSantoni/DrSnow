# Difference-in-differences with DrSnow: a guided example.
#
#   julia --project=. examples/basic_did_demo.jl
#
# Part 1: a single treatment date (2×2 design): TWFE, event study, pre-trend test and
#         the doubly robust Sant'Anna–Zhao estimator.
# Part 2: staggered adoption with effects that grow over time and differ by cohort:
#         why TWFE fails (Goodman-Bacon decomposition, negative weights) and the
#         heterogeneity-robust estimators (Callaway–Sant'Anna, Sun–Abraham,
#         Borusyak–Jaravel–Spiess imputation).
#
# The data are simulated, so every printed estimate can be compared with the truth.

using DrSnow
using DataFrames
using Random
using Statistics

rng = Xoshiro(2026)
section(title) = println("\n", "="^78, "\n", title, "\n", "="^78)

# ---------------------------------------------------------------------------
# Simulated panels
# ---------------------------------------------------------------------------

"""
Panel of `N` states over `T` years. `cohort[i]` is the first treated year index of
state `i` (0 = never treated). `effect(g, e)` is the effect `e` years after adoption.
A covariate `x` shifts both adoption and outcome trends when `confounded = true`.
"""
function simulate(rng, cohort, T, effect; confounded=false)
    N = length(cohort)
    x = randn(rng, N) .+ (confounded ? 0.8 .* (cohort .> 0) : 0.0)
    α = randn(rng, N)
    λ = cumsum(0.3 .* randn(rng, T))
    rows = [(state=i, year=2000 + t, x=x[i],
             first_treat=cohort[i] == 0 ? 0 : 2000 + cohort[i],
             policy=Int(cohort[i] > 0 && t >= cohort[i]),
             earnings=α[i] + λ[t] + (confounded ? 0.5 * x[i] * t : 0.0) +
                      (cohort[i] > 0 && t >= cohort[i] ? effect(cohort[i], t - cohort[i]) :
                       0.0) + randn(rng))
            for i in 1:N for t in 1:T]
    return DataFrame(rows)
end

# ---------------------------------------------------------------------------
# Part 1: one treatment date
# ---------------------------------------------------------------------------
section("Part 1. Single adoption date (half of 200 states treated from year 6)")
cohort1 = vcat(fill(6, 100), zeros(Int, 100))
effect1(g, e) = 2.0 + 0.5e                     # grows by 0.5 per year of exposure
df1 = simulate(rng, cohort1, 10, effect1)
true_att1 = mean(effect1(6, e) for e in 0:4)
println("True ATT (average over treated state-years): ", true_att1)

r = did_twfe(df1, :earnings, :policy, :state, :year)
show(stdout, MIME"text/plain"(), r)
println("\n90% confidence interval (t with G - 1 df): ", confint(r; level=0.90))

es = event_study(df1, :earnings, :policy, :state, :year)
println()
show(stdout, MIME"text/plain"(), es)
println("\nTrue effects by event time: ",
        [e < 0 ? 0.0 : effect1(6, e) for e in relative_periods(es)])
println("Simultaneous 95% band (sup-t):")
display(confint(es; uniform=true, rng=Xoshiro(1)))
println()
show(stdout, MIME"text/plain"(), pre_trend_test(es))

# Conditional parallel trends: adoption and trends both depend on x.
section("Part 1b. Two periods, parallel trends only conditional on x")
df2 = simulate(rng, vcat(fill(2, 400), zeros(Int, 600)), 2, (g, e) -> 1.0;
               confounded=true)
naive = did_drdid(df2, :earnings, :policy, :state, :year)
dr = did_drdid(df2, :earnings, :policy, :state, :year; covariates=[:x])
println("True ATT = 1.0")
println("Unconditional DiD:             ", round(coef(naive)[1]; digits=3),
        " (SE ", round(stderror(naive)[1]; digits=3), ")")
println("Doubly robust DiD (x-adjusted): ", round(coef(dr)[1]; digits=3),
        " (SE ", round(stderror(dr)[1]; digits=3), ")")
println("Pre-treatment covariate balance by cohort:")
show(stdout, MIME"text/plain"(),
     pretreatment_balance(df2, :policy, :state, :year; covariates=[:x]))
println()

# ---------------------------------------------------------------------------
# Part 2: staggered adoption with heterogeneous, dynamic effects
# ---------------------------------------------------------------------------
section("Part 2. Staggered adoption (cohorts treated in years 3, 5 and 7)")
cohort3 = vcat(zeros(Int, 60), fill(3, 60), fill(5, 60), fill(7, 60))
effect3(g, e) = 1.0 + 0.8e + 0.3 * (g - 3)   # dynamic and cohort-specific
df3 = simulate(rng, cohort3, 8, effect3)
tr = df3[df3.policy .== 1, :]
true_att3 = mean(effect3(r.first_treat - 2000, r.year - r.first_treat) for r in eachrow(tr))
println("True ATT (average over treated state-years): ", round(true_att3; digits=3))
show(stdout, MIME"text/plain"(), treatment_timing(df3, :policy, :state, :year))

tw = did_twfe(df3, :earnings, :policy, :state, :year)   # warns: staggered adoption
println("TWFE coefficient: ", round(coef(tw)[1]; digits=3), " (SE ",
        round(stderror(tw)[1]; digits=3), ")")

println("\nGoodman-Bacon decomposition of the TWFE coefficient:")
bd = bacon_decomposition(df3, :earnings, :policy, :state, :year)
show(stdout, MIME"text/plain"(), bd)
println("\nTWFE weights on treated state-years:")
show(stdout, MIME"text/plain"(), twfe_weights(df3, :earnings, :policy, :state, :year))

section("Heterogeneity-robust estimators")
cs = did_callaway_santanna(df3, :earnings, FirstTreated(:first_treat), :state, :year;
                           rng=Xoshiro(7))
simple = aggregate_att(cs, :simple)
println("Callaway–Sant'Anna simple ATT: ", round(coef(simple)[1]; digits=3),
        " (SE ", round(stderror(simple)[1]; digits=3), ")")
dyn = aggregate_att(cs, :dynamic; min_e=-3, max_e=3)
show(stdout, MIME"text/plain"(), dyn)
println()
show(stdout, MIME"text/plain"(), aggregate_att(cs, :group))
println()

sa = did_sun_abraham(df3, :earnings, :policy, :state, :year; max_pre=3, max_post=3)
println("\nSun–Abraham event study:")
show(stdout, MIME"text/plain"(), sa)

bjs = did_imputation(df3, :earnings, :policy, :state, :year)
println("\nImputation (Borusyak–Jaravel–Spiess) ATT: ", round(coef(bjs)[1]; digits=3),
        " (SE ", round(stderror(bjs)[1]; digits=3), ")")
bjs_es = did_imputation(df3, :earnings, :policy, :state, :year; horizons=0:3,
                        pretrends=2)
show(stdout, MIME"text/plain"(), pre_trend_test(bjs_es))

println("\nTrue event-time effects (cohort-size weighted): ",
        [round(mean(effect3(g, e) for g in (3, 5, 7) if g + e <= 8); digits=2)
         for e in 0:3])
