# Adaptive experiments with DrSnow: design, simulation and inference.
#
#   julia --project=. examples/adaptive_bandit_demo.jl
#
# Part 1: plan a three-arm Thompson-sampling experiment by simulation: regret,
#         allocation and the effect of a burn-in and a probability floor.
# Part 2: analyse one experiment: naive sample means versus adaptively weighted
#         AIPW (Hadad et al. 2021), and a small Monte Carlo of their coverage.
# Part 3: deploy step by step (assign a batch, observe it) with a Bernoulli outcome.
# Part 4: a contextual bandit (linear Thompson sampling), evaluation of policies with
#         contextual adaptive weights (Zhan et al. 2021), and policy learning with
#         doubly-robust scores and a policy tree.
#
# Everything is simulated, so estimates can be compared with the truth.

using DrSnow
using DataFrames
using Random
using Statistics

section(title) = println("\n", "="^78, "\n", title, "\n", "="^78)
rng = Xoshiro(2026)

# ---------------------------------------------------------------------------
section("Part 1. Planning: regret and allocation of candidate designs")
# ---------------------------------------------------------------------------
truth = [0.9, 1.0, 1.1]                      # arm means; noise uniform on [-1, 1]
T = 2000
designs = [
    "uniform randomization" => EpsilonGreedy(3; epsilon=1.0),
    "Thompson, no burn-in, no floor" => GaussianThompson(3),
    "Thompson, burn-in 30, floor (1/3) t^-0.7" =>
        GaussianThompson(3; floor=1 / 3, floor_decay=0.7, burnin=30),
    "Top-two Thompson, same floor" =>
        TopTwoThompson(GaussianThompson(3; floor=1 / 3, floor_decay=0.7, burnin=30)),
]
println(rpad("design", 44), "regret  share best  min P(assign)  SE(Q3 - Q1)")
for (name, pol) in designs
    regs = Float64[]
    shares = Float64[]
    minp = Float64[]
    ses = Float64[]
    for r in 1:50
        Y = truth' .+ (2 .* rand(rng, T, 3) .- 1)
        log = run_adaptive_experiment(pol, Y; means=repeat(truth', T), rng=rng)
        push!(regs, cumulative_regret(log)[end])
        push!(shares, mean(log.arms .== 3))
        push!(minp, minimum(log.probabilities))
        est = adaptive_arm_values(log; weights=:constant_allocation)
        push!(ses, stderror(est)[5])
    end
    println(rpad(name, 44), rpad(round(mean(regs); digits=1), 8),
            rpad(round(mean(shares); digits=2), 11),
            rpad(round(mean(minp); sigdigits=2), 15), round(median(ses); digits=3))
end
println("Adaptive designs lower regret; the price is precision for the inferior arms.")
println("Without a floor the smallest probabilities come much closer to zero (min P")
println("column), which inflates inverse-probability weights; top-two sampling keeps")
println("the runner-up in play and gives the most precise contrast at more regret.")

# ---------------------------------------------------------------------------
section("Part 2. Analysis of one experiment")
# ---------------------------------------------------------------------------
pol = GaussianThompson(3; floor=1 / 3, floor_decay=0.7, burnin=30)
log = run_adaptive_experiment(pol, GaussianBandit(truth; sd=0.6), T; batch_size=50,
                              rng=Xoshiro(7))
show(stdout, MIME"text/plain"(), log)
println("\n\nNaive sample means (biased, invalid standard errors):")
show(stdout, MIME"text/plain"(), naive_arm_means(log))
println("\n\nAdaptively weighted AIPW with two-point weights:")
r = adaptive_arm_values(log; weights=:two_point)
show(stdout, MIME"text/plain"(), r)
println("\n\n90% intervals: ", round.(confint(r; level=0.9); digits=3))

println("\nSmall Monte Carlo (200 experiments of 1000 units, no-signal design).")
println("Sample means are biased downward and under-cover; see the documentation")
println("for the weighted versus unweighted AIPW comparison over longer horizons.")
cover = Dict(k => 0 for k in ("two_point", "uniform", "sample_mean"))
bias = Dict(k => 0.0 for k in keys(cover))
for rep in 1:200
    Y = 1.0 .+ (2 .* rand(rng, 1000, 3) .- 1)
    lg = run_adaptive_experiment(GaussianThompson(3; floor=1 / 3, floor_decay=0.7,
                                                  burnin=15), Y; rng=rng)
    for (k, est) in ("two_point" => adaptive_arm_values(lg; contrasts=:none),
                     "uniform" => adaptive_arm_values(lg; weights=:uniform,
                                                      contrasts=:none),
                     "sample_mean" => naive_arm_means(lg; contrasts=:none))
        ci = confint(est; level=0.9)
        cover[k] += ci[1, 1] <= 1.0 <= ci[1, 2]
        bias[k] += coef(est)[1] - 1.0
    end
end
for k in ("two_point", "uniform", "sample_mean")
    println(rpad(k, 13), " coverage of arm 1 (nominal 0.90): ", cover[k] / 200,
            "   mean bias: ", round(bias[k] / 200; digits=3))
end

# ---------------------------------------------------------------------------
section("Part 3. Deployment: assign a batch, record its outcomes")
# ---------------------------------------------------------------------------
exp = AdaptiveExperiment(BetaBernoulliThompson(2; floor=0.1, burnin=100))
for batch in 1:12
    arms = assign!(exp, 50; rng=rng)                    # 50 new participants
    # ... run the intervention; here the conversion rates are 10% and 13%
    y = [rand(rng) < (a == 2 ? 0.13 : 0.10) for a in arms]
    observe!(exp, y)
end
blog = experiment_log(exp)
println(blog)
show(stdout, MIME"text/plain"(), adaptive_arm_values(blog; weights=:constant_allocation))
println()

# ---------------------------------------------------------------------------
section("Part 4. Contextual bandit, policy evaluation and policy learning")
# ---------------------------------------------------------------------------
mean_fn(x) = [0.0, x[1], -x[1], 0.5 * x[2]]
env = ContextualBandit(mean_fn, r -> randn(r, 2), 4)
# A constant floor of 0.1 keeps every arm at probability ≥ 10% for every covariate
# value, which keeps later evaluation of other policies precise (at some regret).
cpol = LinearThompson(4, 2; floor=0.1, burnin=200)
clog = run_adaptive_experiment(cpol, env, 2000; batch_size=100, rng=Xoshiro(3))
oracle = x -> argmax(mean_fn(x))
v = adaptive_policy_value(clog, (oracle, 1); names=["oracle rule", "always arm 1"],
                          contrasts=:reference, reference=2)
show(stdout, MIME"text/plain"(), v)
truth_oracle = mean(maximum(mean_fn(randn(rng, 2))) for _ in 1:100_000)
println("\nTrue value of the oracle rule ≈ ", round(truth_oracle; digits=3),
        "; of arm 1 = 0.")

# Learn a depth-2 tree on the first half, evaluate it on the second half.
first_half = DataFrame(clog)[1:1000, :]
Γ = bandit_dr_scores(clog)[1:1000, :]
tree = policy_tree(Γ, Matrix(first_half[:, [:x1, :x2]]); depth=2, actions=1:4,
                   covariates=[:x1, :x2])
show(stdout, MIME"text/plain"(), tree)
second = DataFrame(clog)[1001:2000, :]
second.t .= 1:1000
learned = predict(tree, second)
# The second half is itself adaptive: evaluate with the logged probabilities and
# per-batch probabilities (the snapshots of the second half's batches).
snaps = clog.snapshots[11:20]
starts = clog.batch_start[11:20] .- 1000
pf = (t, x) -> assignment_probabilities(snaps[findfirst(==(t), starts)], t + 1000;
                                        context=x)
ev = adaptive_policy_value(second, :outcome, :arm, (learned, 1);
                           probabilities=[:p1, :p2, :p3, :p4], time=:t, batch=:batch,
                           covariates=[:x1, :x2], probability_fn=pf,
                           names=["learned tree", "always arm 1"])
show(stdout, MIME"text/plain"(), ev)
println()
