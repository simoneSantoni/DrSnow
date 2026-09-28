# Micro-randomized trials with DrSnow: causal excursion effects (WCLS, EMEE).
#
#   julia --project=. examples/adaptive_mrt_demo.jl
#
# A simulated mobile-health trial in the spirit of HeartSteps (Klasnja et al. 2019):
# 40 participants, 5 decision points a day for 30 days. At each available decision
# point an activity prompt is sent with a probability that depends on the context
# (at home or at work: 0.4, elsewhere: 0.7). The proximal outcome is the log step
# count in the next 30 minutes; the prompt helps early in the study and the effect
# fades with time (habituation), and more so at home.
#
# Part 1: marginal and moderated excursion effects with WCLS, and why the naive
#         regression that ignores the randomization probabilities is not the same
#         estimand.
# Part 2: a binary proximal outcome (any walk in the next 30 minutes) with EMEE on the
#         relative-risk scale.

using DrSnow
using DataFrames
using Random
using Statistics

section(title) = println("\n", "="^78, "\n", title, "\n", "="^78)
rng = Xoshiro(42)

n, days, per_day = 40, 30, 5
rows = NamedTuple[]
for id in 1:n
    base = randn(rng)
    ylag = 0.0
    for d in 1:days, k in 1:per_day
        day = (d - 1) / (days - 1)                 # 0 … 1 over the study
        home = rand(rng) < 0.5
        avail = rand(rng) < 0.85
        prob = home ? 0.4 : 0.7
        a = avail && rand(rng) < prob ? 1.0 : 0.0
        effect = 0.5 - 0.4 * day - 0.2 * home          # true excursion effect
        μ = 2.0 + base + 0.3 * ylag - 0.3 * home + a * effect
        y = μ + randn(rng)
        walk = rand(rng) < 0.25 * exp(0.2 * ylag / 3) * exp(a * (0.4 - 0.3 * day))
        push!(rows, (id=id, day=day, home=Float64(home), avail=Int(avail), prob=prob,
                     prompt=a, logsteps=y, ylag=ylag, walk=Float64(walk)))
        ylag = y
    end
end
df = DataFrame(rows)
println("Person-decision points: ", nrow(df), "; available: ", sum(df.avail))

# ---------------------------------------------------------------------------
section("Part 1. Continuous proximal outcome: WCLS")
# ---------------------------------------------------------------------------
m = wcls(df, :logsteps, :prompt, :id; rand_prob=:prob, controls=[:ylag, :home, :day],
         availability=:avail)
show(stdout, MIME"text/plain"(), m)
# Marginal truth: E[0.5 - 0.4 day - 0.2 home | available]. With a constant numerator
# probability (default 0.5) WCLS targets this unweighted average.
av = df[df.avail .== 1, :]
println("\nTrue marginal excursion effect ≈ ",
        round(mean(0.5 .- 0.4 .* av.day .- 0.2 .* av.home); digits=3))

r = wcls(df, :logsteps, :prompt, :id; rand_prob=:prob, moderators=[:day, :home],
         controls=[:day, :home, :ylag], availability=:avail,
         numerator_prob=:prob)
show(stdout, MIME"text/plain"(), r)
println("\nTrue moderated effect: 0.5 - 0.4 day - 0.2 home")
w = wald_test(coef(r)[2:3], vcov(r)[2:3, 2:3]; dof=dof_residual(r))
println("Joint test of no moderation by day and location: ", w)
println("(numerator_prob = :prob is allowed here because the randomization " *
        "probability depends only on `home`, a moderator.)")

# ---------------------------------------------------------------------------
section("Part 2. Binary proximal outcome: EMEE (log relative risk)")
# ---------------------------------------------------------------------------
e = emee(df, :walk, :prompt, :id; rand_prob=:prob, moderators=[:day],
         controls=[:day, :ylag], availability=:avail)
show(stdout, MIME"text/plain"(), e)
println("\nTrue log relative risk: 0.4 - 0.3 day")
rr = exp.(confint(e)[1, :])
println("Relative risk at the start of the study: ", round(exp(coef(e)[1]); digits=3),
        " (95% CI ", round.(rr; digits=3), ")")

# Plots (need a Makie backend):
#   using CairoMakie
#   save("wcls_day.png", plot_excursion_effect(r; moderator=:day, at=(home=0.0,)))
#   save("emee_day.png", plot_excursion_effect(e; moderator=:day, relative_risk=true))
