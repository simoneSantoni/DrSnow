# Designing an experiment with DrSnow: power, blocking on predicted outcomes,
# simulation-based diagnosis and surrogate design optimization.
#
#   julia --project=. examples/design_power_demo.jl
#
# Part 1: analytic power and minimum detectable effects for common designs.
# Part 2: a prognostic score fitted on pilot data, matched pairs on the score, and an
#         analysis that reuses the same score; expected and realized precision gains.
# Part 3: declare the design, diagnose it by simulation (power, bias, coverage,
#         type-S/M errors), compare with complete randomization, and find the
#         cheapest sample size reaching 80% power with a surrogate model.

using DrSnow
using DataFrames
using Random
using Statistics

rng = Xoshiro(2026)
section(title) = println("\n", "="^78, "\n", title, "\n", "="^78)

# ---------------------------------------------------------------------------
section("Part 1. Analytic power, MDE and sample sizes")
# ---------------------------------------------------------------------------

show(stdout, MIME"text/plain"(), power_means(effect=0.25, power=0.8)); println()
println("\nWith an adjustment covariate explaining 40% of the variance:")
show(stdout, MIME"text/plain"(), power_means(effect=0.25, power=0.8, r2=0.4)); println()
println("\nCluster-randomized trial (ICC 0.1, 20 pupils per school): schools needed")
show(stdout, MIME"text/plain"(), power_cluster(effect=0.25, icc=0.1, cluster_size=20,
                                               power=0.8)); println()
println("\nOne baseline and three follow-up rounds (rho = 0.4): ANCOVA vs DiD MDE")
for est in (:ancova, :did, :post)
    r = power_did(n=400, pre_periods=1, post_periods=3, rho=0.4, estimator=est,
                  power=0.8)
    println("  ", rpad(est, 8), "MDE = ", round(r.effect; digits=4))
end
println("\nEncouragement design, 40% compliance, 2000 units: MDE of the LATE = ",
        round(power_iv(compliance=0.4, n=2000, power=0.8).effect; digits=4))

# ---------------------------------------------------------------------------
section("Part 2. Matched pairs on a prognostic score")
# ---------------------------------------------------------------------------

# outcome model: nonlinear in baseline covariates
f(x1, x2, x3) = 2 .* x1 .+ x2 .^ 2 .- x1 .* x3
covs(m) = DataFrame(x1=randn(rng, m), x2=randn(rng, m), x3=randn(rng, m))
pilot = covs(1000)
pilot.y = f(pilot.x1, pilot.x2, pilot.x3) .+ randn(rng, 1000)

n = 120
sample = covs(n)
sample.id = ["unit$(i)" for i in 1:n]

ps = prognostic_score(pilot, :y; covariates=[:x1, :x2, :x3],
                      learner=ForestLearner(num_trees=300), target=sample, id=:id,
                      rng=Xoshiro(1))
show(stdout, MIME"text/plain"(), ps); println()
bd = block_design(sample, ps; id=:id)
println(); show(stdout, MIME"text/plain"(), bd); println()
vr = variance_reduction(bd)
println("\nExpected variance ratio (pairs vs complete randomization): ",
        round(vr.outcome_variance_ratio; digits=3),
        "; regression adjustment on the score: ",
        round(vr.regression_adjustment_ratio; digits=3))

# one realized experiment, analysed with the design's blocks and score
exp_ = assign_treatment(bd, sample; rng=Xoshiro(2))
exp_.y = f(exp_.x1, exp_.x2, exp_.x3) .+ randn(rng, n) .+ 0.5 .* exp_.treated
for m in (:difference, :lin)
    r = experiment_estimate(exp_, :y, :treated, bd; method=m)
    println(rpad(method_name(r), 55), round(coef(r)[1]; digits=3), "  (se ",
            round(stderror(r)[1]; digits=3), ")")
end
r0 = experiment_estimate(exp_, :y, :treated)
println(rpad("Ignoring the design (Neyman, no blocks)", 55), round(coef(r0)[1]; digits=3),
        "  (se ", round(stderror(r0)[1]; digits=3), ")")
rt = randomization_test(exp_, :y, :treated; mechanism=bd.mechanism, id=:id,
                        nperm=2000, rng=Xoshiro(3))
println("Randomization test with the pair mechanism: p = ", round(rt.pvalue; digits=4))

# ---------------------------------------------------------------------------
section("Part 3. Simulation-based diagnosis and surrogate optimization")
# ---------------------------------------------------------------------------

# The pilot model is fixed at the design stage. To keep the simulation fast, a large
# pool of covariate profiles is scored once with it; each simulated experiment samples
# its units from the pool and draws fresh outcomes.
pool = covs(20_000)
pool.id = 1:20_000
pool.score = prognostic_score(pilot, :y; covariates=[:x1, :x2, :x3],
                              learner=ForestLearner(num_trees=300), target=pool, id=:id,
                              rng=Xoshiro(6)).score
function population(rng, p)
    df = pool[rand(rng, 1:nrow(pool), p.n), [:x1, :x2, :x3, :score]]
    df.id = 1:p.n
    y0 = f(df.x1, df.x2, df.x3) .+ randn(rng, p.n)
    df.Y0 = y0
    df.Y1 = y0 .+ p.effect
    df.pair = block_design(df, :score; id=:id).blocks
    return df
end
ests = ["pairs, difference" => (d, p) -> experiment_estimate(d, :Y, :Z; blocks=:pair),
        "pairs, Lin + score" => (d, p) -> experiment_estimate(d, :Y, :Z; method=:lin,
                                                                blocks=:pair,
                                                                covariates=[:score])]
paired = declare_design(population; params=(n=80, effect=0.5), estimand=0.5,
                        assignment=(d, p) -> MatchedPairsRandomization(d.pair),
                        estimators=ests, name="matched pairs on a prognostic score")
complete = declare_design(population; params=(n=80, effect=0.5), estimand=0.5,
                          assignment=(d, p) -> CompleteRandomization(p.n, p.n ÷ 2),
                          estimators="complete, difference" =>
                              (d, p) -> experiment_estimate(d, :Y, :Z),
                          name="complete randomization")
for d in (paired, complete)
    dx = diagnose_design(d; sims=500, rng=Xoshiro(4))
    t = unstack(dx.diagnosands[in.(dx.diagnosands.diagnosand,
                                   Ref([:power, :bias, :rmse, :coverage, :type_m])),
                               [:estimator, :diagnosand, :value]],
                :diagnosand, :value)
    println("\n", d.name); show(stdout, MIME"text/plain"(), t); println()
end

opt = optimize_design(paired; space=(n=20:4:600,), target_power=0.8,
                      estimator="pairs, Lin + score", sims_per_point=100,
                      max_sims=2000, verify_sims=500, rng=Xoshiro(5))
println(); show(stdout, MIME"text/plain"(), opt); println()
println("Analytic benchmark (complete randomization, no adjustment): n = ",
        ceil(Int, power_means(effect=0.5, sd=std(pilot.y), power=0.8).parameters.n))
