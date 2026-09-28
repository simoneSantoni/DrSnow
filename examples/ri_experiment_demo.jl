# Randomization inference for randomized experiments with DrSnow.
#
# Run with:  julia --project=. examples/ri_experiment_demo.jl
#
# The script simulates three experiments (complete, stratified and
# cluster-randomized), then runs Fisher randomization tests, inverts them into
# confidence intervals, applies regression-based randomization inference
# (Young 2019), checks covariate balance, and adjusts for multiple outcomes. The
# data are simulated, so the true effects are known; outputs report estimates,
# intervals and p-values only.

using DrSnow
using DataFrames
using Random
using Statistics

rng = Xoshiro(2026)

println("="^78)
println("1. Completely randomized experiment (n = 60, 30 treated)")
println("="^78)
n = 60
age = round.(30 .+ 8 .* randn(rng, n))
z = shuffle(rng, [trues(30); falses(30)])
y = 0.05 .* age .+ 0.6 .* z .+ randn(rng, n)          # true constant effect 0.6
df = DataFrame(y=y, treated=Int.(z), age=age)

# Sharp null of no effect for any unit; 10,000 draws (C(60, 30) is far too many
# assignments to enumerate).
r = randomization_test(df, :y, :treated; nperm=10_000, rng=Xoshiro(1))
show(stdout, MIME"text/plain"(), r); println("\n")

# Studentized statistic: exact for the sharp null and asymptotically valid for the
# weak null of zero average effect (Wu & Ding 2021).
rs = randomization_test(df, :y, :treated; statistic=:studentized, nperm=10_000,
                        rng=Xoshiro(1))
println("Studentized statistic: t = ", round(rs.observed; digits=3),
        ", p = ", round(rs.pvalue; digits=4), " (MC s.e. ", round(rs.mc_se; digits=4),
        ")\n")

# Confidence interval for a constant additive effect by test inversion. With the
# (linear) difference in means the inversion is exact over the reference set.
ci = ri_confint(df, :y, :treated; nperm=5_000, rng=Xoshiro(2))
show(stdout, MIME"text/plain"(), ci); println("\n")

# Covariate adjustment (Lin 2013) usually narrows the interval.
ci_lin = ri_confint(df, :y, :treated; statistic=:lin, covariates=[:age], nperm=5_000,
                    rng=Xoshiro(2))
println("Lin-adjusted 95% interval: [", round(ci_lin.lower; digits=3), ", ",
        round(ci_lin.upper; digits=3), "], estimate ", round(ci_lin.estimate; digits=3))
ci_rank = ri_confint(df, :y, :treated; statistic=:rank_sum, nperm=2_000,
                     rng=Xoshiro(2))
println("Rank-sum 95% interval:     [", round(ci_rank.lower; digits=3), ", ",
        round(ci_rank.upper; digits=3), "], Hodges–Lehmann ",
        round(ci_rank.estimate; digits=3), "\n")

println("="^78)
println("2. Stratified experiment: 4 blocks of 6 units, 3 treated per block")
println("="^78)
blocks = repeat(1:4, inner=6)
zb = reduce(vcat, [shuffle(rng, [true, true, true, false, false, false]) for _ in 1:4])
yb = 1.0 .* blocks .+ 0.8 .* zb .+ randn(rng, 24)
dfb = DataFrame(y=yb, treated=Int.(zb), block=blocks)
println("Assignments under the design: ",
        n_assignments(StratifiedRandomization(dfb.block, dfb.treated)))
# 20^4 = 160,000 assignments: small enough to enumerate exactly.
rb = randomization_test(dfb, :y, :treated; strata=:block, exact=true)
show(stdout, MIME"text/plain"(), rb); println()
cib = ri_confint(dfb, :y, :treated; strata=:block, exact=true)
println("Exact 95% interval (stratified difference in means): [",
        round(cib.lower; digits=3), ", ", round(cib.upper; digits=3), "]\n")

println("="^78)
println("3. Cluster-randomized experiment: 12 schools, 6 treated")
println("="^78)
sizes = rand(rng, 8:25, 12)
school = reduce(vcat, [fill(s, k) for (s, k) in enumerate(sizes)])
zs = shuffle(rng, [trues(6); falses(6)])
ns = length(school)
x = randn(rng, ns)
score = 0.3 .* zs[school] .+ 0.5 .* x .+ 0.8 .* randn(rng, 12)[school] .+ randn(rng, ns)
attend = 0.2 .* x .+ 0.5 .* randn(rng, 12)[school] .+ randn(rng, ns)
dfc = DataFrame(score=score, attend=attend, treated=Int.(zs[school]), x=x,
                school=school)
# C(12, 6) = 924 assignments: exact enumeration.
rc = ri_regression(dfc, [:score, :attend], :treated; covariates=[:x], cluster=:school,
                   nperm=1_000)
show(stdout, MIME"text/plain"(), rc); println("\n")

bt = ri_balance_test(dfc, :treated, [:x]; cluster=:school, nperm=1_000)
show(stdout, MIME"text/plain"(), bt); println()
println(bt.details.per_covariate, "\n")

println("="^78)
println("4. Several outcomes: Westfall–Young step-down adjustment")
println("="^78)
f = randn(rng, n)
df.y2 = 0.6 .* f .+ randn(rng, n)
df.y3 = 0.6 .* f .+ 0.5 .* z .+ randn(rng, n)
df.y4 = 0.6 .* f .+ randn(rng, n)
mt = ri_multiple_testing(df, [:y, :y2, :y3, :y4], :treated; statistic=:studentized,
                         nperm=5_000, rng=Xoshiro(3))
show(stdout, MIME"text/plain"(), mt); println()
