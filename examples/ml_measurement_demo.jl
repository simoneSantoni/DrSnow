# Causal inference with an LLM-coded outcome: a guided example.
#
#   julia --project=. examples/ml_measurement_demo.jl
#
# A research team studies online hostility. Posts are coded as hostile / not hostile
# by a large language model (cheap, every post) and by trained experts (expensive, a
# random subsample with known probabilities). The LLM's mistakes are not random: it
# misses hostility phrased in the coded language that the interventions studied here
# encourage, so its error differs between treated and control posts.
#
# Part 1: a randomized experiment (a "civility nudge" shown to half the users).
#         Naive analysis of the LLM labels vs design-based supervised learning (DSL)
#         and prediction-powered inference (PPI++) with the expert subsample.
# Part 2: a natural experiment (a moderation policy adopted by half of the forums in
#         month 4), analysed by difference-in-differences with the LLM-coded share of
#         hostile posts; corrected TWFE and Callaway–Sant'Anna estimates, the bias
#         test, the held-out rule for a fine-tuned classifier, and what happens when
#         expert labels exist only before the policy.
#
# The data are simulated, so every printed estimate can be compared with the truth.

using DrSnow
using DataFrames
using Random
using Statistics

rng = Xoshiro(2027)
section(title) = println("\n", "="^78, "\n", title, "\n", "="^78)
line(name, r; j=1) = println(rpad(name, 46), round(coef(r)[j]; digits=3), "  [",
                             join(round.(confint(r)[j, :]; digits=3), ", "), "]")
logistic(x) = 1 / (1 + exp(-x))

# ---------------------------------------------------------------------------
# Part 1: randomized experiment with an LLM-coded binary outcome
# ---------------------------------------------------------------------------
section("Part 1: randomized civility nudge, outcome coded by an LLM")

n = 4000
activity = randn(rng, n)                          # pre-treatment covariate
nudge = Float64.(rand(rng, n) .< 0.5)
# true hostility: the nudge lowers the probability of a hostile post by ≈ 0.10
p_hostile = logistic.(-0.4 .+ 0.6 .* activity .- 0.5 .* nudge)
hostile = Float64.(rand(rng, n) .< p_hostile)
# LLM coding: misses 30% of hostile posts in general and 60% among nudged users,
# whose hostility is more often sarcastic; 5% false positives
miss_rate = ifelse.(nudge .== 1, 0.6, 0.3)
llm = Float64.(ifelse.(hostile .== 1, rand(rng, n) .> miss_rate, rand(rng, n) .< 0.05))
# expert coding: 10% of posts at random, 25% among very active users (known design)
p_label = ifelse.(activity .> 1, 0.25, 0.10)
coded = rand(rng, n) .< p_label
rct = DataFrame(hostile=Union{Missing,Float64}[c ? h : missing
                                               for (c, h) in zip(coded, hostile)],
                llm=llm, nudge=nudge, activity=activity, p_label=p_label)
true_ate = mean(logistic.(-0.4 .+ 0.6 .* activity .- 0.5)) -
           mean(logistic.(-0.4 .+ 0.6 .* activity))
println("Posts: $n, expert-coded: $(count(coded)). ",
        "True ATE on the probability of hostility: ", round(true_ate; digits=3))

r_dsl = dsl_regression(rct, :hostile; covariates=[:nudge, :activity], prediction=:llm,
                       label_prob=:p_label, rng=rng)
naive = r_dsl.naive_coef[2]
println(rpad("Naive: LLM labels treated as truth", 46), round(naive; digits=3))
line("DSL linear probability model", r_dsl; j=2)
r_ppi = ppi_ate(rct, :hostile, :nudge, :llm; label_prob=:p_label)
line("PPI++ ATE (difference in means)", r_ppi)
r_lin = ppi_ate(rct, :hostile, :nudge, :llm; label_prob=:p_label, covariates=[:activity])
line("PPI++ ATE (Lin regression adjustment)", r_lin)
println("Power-tuning weight on the LLM labels: λ = ",
        round(r_lin.details.lambda; digits=2))
r_ipw = ppi_ate(rct, :hostile, :nudge, :llm; label_prob=:p_label, lambda=0)
line("Expert-coded posts only (IPW, λ = 0)", r_ipw)
println("Standard error with vs without the LLM labels: ",
        round(stderror(r_ppi)[1]; digits=4), " vs ", round(stderror(r_ipw)[1]; digits=4))
r_logit = dsl_regression(rct, :hostile; covariates=[:nudge, :activity],
                         prediction=:llm, family=:binomial, label_prob=:p_label,
                         rng=rng)
line("DSL logistic regression, nudge coefficient", r_logit; j=2)

println("\nDoes the LLM's error differ between arms? (expert-coded posts)")
t = differential_error_test(rct, :hostile, :llm; by=[:nudge], label_prob=:p_label)
show(stdout, MIME"text/plain"(), t)
println("Mean error by arm: ", round.(t.details.mean_error; digits=3))

# ---------------------------------------------------------------------------
# Part 2: difference-in-differences with an LLM-coded outcome
# ---------------------------------------------------------------------------
section("Part 2: moderation policy (natural experiment), forum-month panel")

F, T = 160, 6
forum = repeat(1:F, inner=T)
month = repeat(1:T, outer=F)
adopter = repeat(Float64.(rand(rng, F) .< 0.5), inner=T)
policy = adopter .* (month .>= 4)
level = repeat(0.3 .+ 0.08 .* randn(rng, F), inner=T)
# true share of hostile posts: common trend, policy effect −0.06
share = clamp.(level .+ 0.01 .* month .- 0.06 .* policy .+ 0.03 .* randn(rng, F * T),
               0, 1)
# LLM-coded share: attenuated, and after the policy users in adopting forums move
# to coded language the LLM misses (−0.04 extra)
llm_share = 0.05 .+ 0.7 .* share .- 0.04 .* policy .+ 0.02 .* randn(rng, F * T)
# experts code a random 25% of forum-months (all months, both groups)
coded = rand(rng, F * T) .< 0.25
panel = DataFrame(forum=forum, month=month, policy=policy,
                  first=ifelse.(adopter .== 1, 4, 0), llm_share=llm_share,
                  share=Union{Missing,Float64}[c ? s : missing
                                               for (c, s) in zip(coded, share)])
println("Forum-months: $(F * T), expert-coded: $(count(coded)). True ATT: -0.06")

r = did_with_predicted_outcome(panel, :share, :policy, :forum, :month;
                               prediction=:llm_share, rng=rng)
line("Naive TWFE on the LLM-coded share", r.naive)
line("Corrected TWFE (design-based)", r)
println("Bias test: naive − corrected = ", round(r.bias_test.details.difference;
                                                 digits=3),
        ", p = ", round(r.bias_test.pvalue; digits=4))
r_cs = did_with_predicted_outcome(panel, :share, FirstTreated(:first), :forum, :month;
                                  prediction=:llm_share, estimator=:cs, rng=rng)
line("Corrected Callaway–Sant'Anna (simple ATT)",
     aggregate_att(r_cs.corrected, :simple; bootstrap=false))

panel.adopter = adopter
panel.post = Float64.(month .>= 4)
t2 = differential_error_test(panel, :share, :llm_share; by=[:adopter, :post],
                             cluster=:forum)
println("\nMean LLM error by adopter × post cell:")
for (c, m) in zip(t2.details.cells, t2.details.mean_error)
    println("  ", rpad(c, 26), round(m; digits=3))
end

println("\nHeld-out rule: a classifier fine-tuned on the expert-coded months of forums",
        " 1-30 must not score those forums in the causal analysis.")
panel.trained_on = Int.(panel.forum .<= 30)
r_ho = did_with_predicted_outcome(panel, :share, :policy, :forum, :month;
                                  prediction=:llm_share, measure_training=:trained_on,
                                  rng=rng)
println("Forums excluded: ", r_ho.n_dropped_training, "; rows used: ", nobs(r_ho))
line("Corrected TWFE, held-out forums", r_ho)

println("\nExpert labels only before the policy (months 1-3):")
pre = copy(panel)
pre.share = [m <= 3 ? s : missing for (m, s) in zip(month, share)]
try
    did_with_predicted_outcome(pre, :share, :policy, :forum, :month;
                               prediction=:llm_share)
catch err
    println("  design-based mode refuses: ", first(sprint(showerror, err), 110), "...")
end
r_st = did_with_predicted_outcome(pre, :share, :policy, :forum, :month;
                                  prediction=:llm_share, assume_stable_error=true,
                                  bootstrap_reps=199, rng=rng)
line("Stable-error calibration", r_st)
println("  The calibration learned before the policy cannot see the policy-induced ",
        "error,\n  so this estimate inherits it: labelling in every period is what ",
        "makes the\n  design-based correction assumption-free.")
