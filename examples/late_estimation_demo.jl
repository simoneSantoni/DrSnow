# LATE / IV estimation with DrSnow
#
# Run from the repository root:
#     julia --project=. examples/late_estimation_demo.jl
#
# Simulated setting: a job-training program is offered by lottery (instrument Z,
# binary). Take-up (treatment D, binary) is voluntary: some applicants never enrol,
# some find a way to enrol without an offer, and the rest follow their offer
# (compliers). Unobserved motivation raises both take-up and earnings, so OLS of
# earnings on D is biased. The effect of training differs across people, and the
# lottery odds depend on an observed covariate (priority groups), so the instrument
# is only valid conditional on that covariate in the last part of the demo.
#
# Every number below comes from the simulated data; the true values are printed
# alongside so the estimators can be compared with the truth.

using DrSnow, DataFrames, Random, Statistics, Printf

rng = Xoshiro(20260927)

section(title) = (println(); println("="^78); println(title); println("="^78))
show_plain(x) = (show(stdout, MIME"text/plain"(), x); println())

# ----------------------------------------------------------------------------
# 1. Data-generating process
# ----------------------------------------------------------------------------
n = 4000
age = 20 .+ 25 .* rand(rng, n)
female = Float64.(rand(rng, n) .< 0.5)
prior_earn = 15 .+ 0.3 .* (age .- 30) .+ 3 .* randn(rng, n)
motivation = randn(rng, n)                           # unobserved
site = rand(rng, 1:40, n)                            # 40 training sites
site_shock = 2 .* randn(rng, 40)

# compliance types depend on motivation (always-takers are the most motivated)
u = rand(rng, n)
p_always = 0.10 .+ 0.10 .* (motivation .> 1)
p_never = 0.30 .+ 0.10 .* (motivation .< -1)
always = u .< p_always
never = u .> 1 .- p_never
complier = .!(always .| never)

offer = Float64.(rand(rng, n) .< 0.5)                # randomized lottery
train = Float64.(always .| (complier .& (offer .== 1)))

# heterogeneous effect: larger for younger applicants
effect = 3.0 .- 0.08 .* (age .- 30)
earn0 = 10 .+ 0.3 .* prior_earn .+ 2.5 .* motivation .- 1.0 .* female .+
        site_shock[site] .+ 3 .* randn(rng, n)
earn = earn0 .+ effect .* train

df = DataFrame(earn=earn, train=train, offer=offer, age=age, female=female,
               prior_earn=prior_earn, site=site)

true_late = mean(effect[complier])
true_ate = mean(effect)
@printf("True complier LATE: %.3f   (population ATE: %.3f)\n", true_late, true_ate)
@printf("True shares — compliers %.3f, always-takers %.3f, never-takers %.3f\n",
        mean(complier), mean(always), mean(never))

# ----------------------------------------------------------------------------
# 2. OLS versus 2SLS
# ----------------------------------------------------------------------------
section("2. OLS is biased by unobserved motivation; 2SLS uses the lottery")
ols = DrSnow.FixedEffectModels.reg(df, make_formula(:earn, [:train, :age, :female]),
                                   Vcov.cluster(:site))
@printf("OLS coefficient on train: %.3f (se %.3f)\n", coef(ols)[2], stderror(ols)[2])

r = late_2sls(df, :earn, :train, :offer; cluster=:site)
show_plain(r)
tf = tf_confint(r)
@printf("tF interval (Lee et al. 2022): [%.3f, %.3f] with critical value %.2f\n",
        tf.lower, tf.upper, tf.critical_value)

# ----------------------------------------------------------------------------
# 3. Weak-instrument diagnostics and robust inference
# ----------------------------------------------------------------------------
section("3. First-stage strength and weak-IV-robust confidence sets")
show_plain(first_stage_diagnostics(r))
show_plain(weak_iv_confidence_set(r))

# A deliberately weak instrument: the offer is recorded correctly for only 8% of
# applicants and replaced by a coin flip for the rest, which dilutes the first stage.
weak = copy(df)
flip = rand(rng, n) .< 0.92
weak.offer_weak = ifelse.(flip, Float64.(rand(rng, n) .< 0.5), weak.offer)
rw = late_2sls(weak, :earn, :train, :offer_weak; cluster=:site)
@printf("\nWeak instrument: 2SLS %.3f (se %.3f), effective F = %.2f\n",
        estimate(rw), stderror(rw)[1], rw.first_stage.effective_F)
println("  t-based 95% CI:  ", round.(confint(rw)[1, :]; digits=3))
tfw = tf_confint(rw)
@printf("  tF 95%% CI:       [%.3f, %.3f] (critical value %.2f)\n", tfw.lower, tfw.upper,
        tfw.critical_value)
show_plain(weak_iv_confidence_set(rw))

# ----------------------------------------------------------------------------
# 4. Who are the compliers?
# ----------------------------------------------------------------------------
section("4. Compliance types and complier characteristics")
show_plain(estimate_compliance(df, :train, :offer; cluster=:site))
prof = complier_characteristics(df, :train, :offer, [:age, :female, :prior_earn];
                                cluster=:site)
show_plain(prof)
@printf("True mean age — compliers %.2f, always-takers %.2f, never-takers %.2f\n",
        mean(age[complier]), mean(age[always]), mean(age[never]))
lw = late_ipw(df, :earn, :train, :offer; cluster=:site)
show_plain(lw)

# ----------------------------------------------------------------------------
# 5. Instrument valid only conditional on a covariate
# ----------------------------------------------------------------------------
section("5. Conditional instrument: offer odds depend on age (priority groups)")
p_offer = ifelse.(age .< 30, 0.75, 0.35)
offer_c = Float64.(rand(rng, n) .< p_offer)
df.offer_c = offer_c
df.train_c = Float64.(always .| (complier .& (offer_c .== 1)))
df.earn_c = earn0 .+ effect .* df.train_c
df.young = Float64.(age .< 30)
println("Unadjusted Wald estimate (confounded by age):")
@printf("  %.3f\n", estimate(late_2sls(df, :earn_c, :train_c, :offer_c; cluster=:site)))
rs = late_2sls(df, :earn_c, :train_c, :offer_c; fe=[:young], cluster=:site)
@printf("2SLS with saturated age-group fixed effects: %.3f (%s)\n", estimate(rs),
        estimand(rs))
ri = late_ipw(df, :earn_c, :train_c, :offer_c; covariates=[:young], cluster=:site)
@printf("IPW (κ-weighted) LATE: %.3f (se %.3f); true complier LATE %.3f\n",
        estimate(ri), stderror(ri)[1], true_late)
println("The saturated 2SLS estimand weights the two age groups by the variance of")
println("the offer within group; the IPW estimand weights them by complier shares.")

# ----------------------------------------------------------------------------
# 6. Falsification and specification tests
# ----------------------------------------------------------------------------
section("6. Falsification tests (non-rejection is not evidence of validity)")
show_plain(instrument_balance(df, :offer, [:age, :female, :prior_earn]; cluster=:site))
show_plain(first_stage_sign_test(df, :train, :offer, [:female, :site]))
show_plain(instrument_validity_test(df, :earn, :train, :offer; n_bootstrap=499,
                                    rng=Xoshiro(1)))
show_plain(endogeneity_test(late_2sls(df, :earn, :train, :offer;
                                      covariates=[:age, :female], cluster=:site)))

# ----------------------------------------------------------------------------
# 7. Sensitivity to violations of the exclusion restriction
# ----------------------------------------------------------------------------
section("7. Plausibly exogenous: allowing a direct effect of the offer")
println("γ is the direct effect of receiving an offer on earnings (earnings units),")
println("e.g. an encouragement effect of being selected that does not work through")
println("training.")
show_plain(plausibly_exogenous(r; method=:uci, gamma=(0.0, 0.3)))
show_plain(plausibly_exogenous(r; method=:ltz, gamma_mean=0.15, gamma_vcov=0.1^2))
