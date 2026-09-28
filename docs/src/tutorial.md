# Tutorial

```@meta
CurrentModule = DrSnow
```

This tutorial works through four designs with the current API. Every code block on
this page is executed when the documentation is built, so the output shown is what
DrSnow returns for these data.

1. [Staggered difference-in-differences](@ref tut-did): the TWFE warning, the
   Goodman-Bacon decomposition, Callaway–Sant'Anna, an event study, a pre-trend test
   and HonestDiD sensitivity analysis.
2. [Instrumental variables](@ref tut-iv): 2SLS, weak-instrument diagnostics, the
   Anderson–Rubin confidence set, compliance shares and a complier profile.
3. [Sharp regression discontinuity](@ref tut-rd): robust bias-corrected estimation,
   the density test and covariate balance at the cutoff.
4. [Randomization inference](@ref tut-ri): Fisher tests, a randomization confidence
   interval and a balance test for a stratified experiment.

The DiD, IV and RD sections use datasets that the test suite also uses to validate
DrSnow against R (see [Validation](validation.md)). They ship in `test/validation/` in the
repository.

```@example tut
using DrSnow
using CSV, DataFrames, Random, StableRNGs

validation_dir = joinpath(pkgdir(DrSnow), "test", "validation")
nothing # hide
```

## [Staggered difference-in-differences](@id tut-did)

`mpdta` (Callaway & Sant'Anna 2021) is a balanced panel of 500 US counties observed
from 2003 to 2007. The outcome `lemp` is log teen employment; counties are treated by
a minimum-wage increase in 2004, 2006 or 2007 (`first_treat`, `0` for counties never
treated in the sample).

```@example tut
mpdta = CSV.read(joinpath(validation_dir, "did", "mpdta.csv"), DataFrame)
combine(groupby(mpdta, :first_treat), :countyreal => (x -> length(unique(x))) => :counties)
```

### What TWFE estimates here

With several adoption dates, the two-way fixed effects coefficient is a weighted
average of cohort-period effects in which some weights can be negative. `did_twfe`
detects the staggered design and warns. The column `d` is the treatment indicator,
`1` from `first_treat` onwards.

```@example tut
twfe = did_twfe(mpdta, :lemp, :d, :countyreal, :year)
```

The Goodman-Bacon decomposition writes this coefficient as a weighted average of all
2×2 comparisons. The `later_vs_earlier` comparisons use already-treated counties as
controls, which is only valid if their effects are constant over time.

```@example tut
bacon = bacon_decomposition(mpdta, :lemp, FirstTreated(:first_treat),
                            :countyreal, :year)
```

```@example tut
twfe_weights(mpdta, :lemp, FirstTreated(:first_treat), :countyreal, :year)
```

Here the problematic comparisons carry little weight, so TWFE and the
heterogeneity-robust estimators below end up close. That is a property of these
data, not something to assume.

### Callaway–Sant'Anna

[`did_callaway_santanna`](@ref) estimates every group-time effect ATT(g, t) against
never-treated counties (doubly robust by default). `FirstTreated(:first_treat)` tells
DrSnow that the column holds the first treatment period, with `0` for never treated.

```@example tut
cs = did_callaway_santanna(mpdta, :lemp, FirstTreated(:first_treat),
                           :countyreal, :year; rng=StableRNG(1))
```

[`aggregate_att`](@ref) summarizes the group-time effects: an overall ATT, effects by
cohort, or effects by event time.

```@example tut
aggregate_att(cs, :simple)
```

```@example tut
aggregate_att(cs, :group)
```

### Event study and pre-trends

For an event study that can be fed to a sensitivity analysis, every coefficient must
be measured against the same reference period (`base_period = :universal`, period
`e = -1`). The dynamic aggregation carries bootstrap draws for simultaneous (sup-t)
bands.

```@example tut
cs_u = did_callaway_santanna(mpdta, :lemp, FirstTreated(:first_treat),
                             :countyreal, :year; base_period=:universal,
                             rng=StableRNG(1))
es = aggregate_att(cs_u, :dynamic; rng=StableRNG(2))
```

```@example tut
confint(es; uniform=true)   # simultaneous 95% bands, one row per event time
```

The joint pre-trend test uses the full covariance of the pre-period coefficients.

```@example tut
pre_trend_test(es)
```

The test does not reject, but that is weak evidence: pre-trend tests often have low
power, and the pre-period estimates for `e = -3` and `e = -2` are about as large as the
first post-period effect. What the data can support is better expressed by asking how
large a violation of parallel trends would overturn the conclusion.

### Sensitivity to violations of parallel trends

[`honest_did`](@ref) (Rambachan & Roth 2023) computes confidence sets for the
average effect over event times 0–3 under the relative-magnitudes restriction: each
period-to-period change in the difference in trends after treatment may be up to
`M̄` times the largest such change between consecutive pre-treatment periods. The
restriction bounds changes in the violation, not its level.

```@example tut
hd = honest_did(es; restriction=:relative_magnitudes, M=0:0.5:2, target=0:3,
                rng=StableRNG(3))
```

```@example tut
honest_breakdown(es; restriction=:relative_magnitudes, target=0:3, rng=StableRNG(3))
```

The negative average effect survives post-treatment changes in the trend difference
up to about half the largest pre-treatment change, and no further. A careful write-up would
report the event-study estimates, the pre-trend test with this caveat and the
breakdown value, rather than the TWFE coefficient alone.

The same event study can be plotted with a Makie backend (see
[Results, tables and plots](@ref)); this block is not executed here:

```julia
using CairoMakie
plot_event_study(es; uniform=true)
```

## [Instrumental variables](@id tut-iv)

### 2SLS with weak-instrument diagnostics

Card (1995) instruments years of schooling with growing up near a four-year college
(`nearc4`), controlling for experience, race and region.

```@example tut
card = CSV.read(joinpath(validation_dir, "iv", "card.csv"), DataFrame)
controls = [:exper, :expersq, :black, :smsa, :south]
iv = iv_regression(card, :lwage, :educ, :nearc4; covariates=controls)
```

The printout states what 2SLS estimates here: with years of schooling (a multi-valued
treatment), a binary instrument and non-saturated controls, it is a weighted
combination of conditional average causal responses, whose weights are guaranteed to
be non-negative only if the instrument's conditional mean is linear in the controls.
It also reports the first stage. The detailed weak-instrument diagnostics are in
`iv.first_stage`:

```@example tut
iv.first_stage
```

The effective F (17.5) is below the Olea–Pflueger critical value for a 10% worst-case
bias (23.1), so the t-based interval is not reliable. The Anderson–Rubin test and
confidence set have correct size whatever the instrument strength:

```@example tut
weak_iv_test(iv; beta0=0.0)
```

```@example tut
weak_iv_confidence_set(iv)
```

The AR set is bounded and excludes zero, but it is noticeably wider on the upper side
than the Wald interval.

### LATE, compliance and compliers

With a binary instrument and binary treatment, 2SLS estimates a local average
treatment effect for compliers. The simulated encouragement design below has a
randomized offer, three compliance types and a treatment effect of 2.

```@example tut
rng = StableRNG(7)
n = 2_000
enc = DataFrame(age = rand(rng, 18:65, n), female = rand(rng, 0:1, n),
                offer = rand(rng, 0:1, n))
u = rand(rng, n)
ctype = [x < 0.2 ? :always : x < 0.6 ? :complier : :never for x in u]
enc.training = [t == :always ? 1 : t == :never ? 0 : z
                for (t, z) in zip(ctype, enc.offer)]
enc.earnings = 10 .+ 0.05 .* enc.age .+ 2.0 .* enc.training .+ randn(rng, n)

late = late_2sls(enc, :earnings, :training, :offer)
```

```@example tut
estimate_compliance(enc, :training, :offer)
```

[`complier_characteristics`](@ref) reports the mean characteristics of compliers,
always-takers and never-takers (Abadie's κ-weighting), which describes the population
to which the LATE applies.

```@example tut
profile = complier_characteristics(enc, :training, :offer, [:age, :female])
profile.table[:, [:variable, :population_mean, :complier_mean, :complier_se,
                  :difference_pvalue]]
```

## [Sharp regression discontinuity](@id tut-rd)

The US Senate data of Cattaneo, Frandsen & Titiunik (2015) relate the Democratic
margin of victory in one election (`margin`) to the Democratic vote share in the
next election for the same seat (`vote`). Winning is determined by `margin ≥ 0`.

```@example tut
senate = CSV.read(joinpath(validation_dir, "rd", "senate.csv"), DataFrame;
                  missingstring="")
rd = rd_estimate(senate, :vote, :margin)
```

The headline row follows `rdrobust`: `coef`, `tidy` and `regtable` report the
conventional point estimate with the robust bias-corrected standard error, p-value and
interval (MSE-optimal bandwidth). The robust interval is centred on the bias-corrected
estimate, `rd.tau_bias_corrected`, so it is not symmetric around the conventional
estimate. [`rd_inference_table`](@ref) lists the conventional, bias-corrected and
robust rows.

```@example tut
(conventional = coef(rd)[1], bias_corrected = rd.tau_bias_corrected,
 robust_ci = confint(rd))
```

```@example tut
rd_inference_table(rd)
```

The density test checks one implication of the design, the absence of sorting
around the cutoff:

```@example tut
rd_density_test(senate, :margin)
```

Predetermined covariates should not jump at the cutoff:

```@example tut
rd_covariate_balance(senate, [:demvoteshlag1, :population], :margin)
```

Neither check rejects. As the printouts say, this is consistent with the design but
does not establish the continuity assumption.

## [Randomization inference](@id tut-ri)

In an experiment the assignment mechanism is known, and randomization inference uses
it as the only source of randomness. Here 240 students in 12 schools are randomized
within school, half to treatment.

```@example tut
rng = StableRNG(2026)
expt = DataFrame(school = repeat(1:12, inner=20))
expt.pretest = randn(rng, nrow(expt)) .+ 0.2 .* expt.school
expt.treated = reduce(vcat, [shuffle(rng, [fill(1, 10); fill(0, 10)]) for _ in 1:12])
expt.score = expt.pretest .+ 0.4 .* expt.treated .+ randn(rng, nrow(expt))
nothing # hide
```

The Fisher test of the sharp null of no effect for any student, using the
stratified mechanism (`strata`) and the difference in means:

```@example tut
ri = randomization_test(expt, :score, :treated; strata=:school, nperm=5_000,
                        rng=StableRNG(1))
```

A studentized, regression-adjusted statistic (Lin 2013) uses the pretest to gain
precision and is also asymptotically valid for the weak null of a zero average
effect:

```@example tut
randomization_test(expt, :score, :treated; strata=:school,
                   statistic=:lin_studentized, covariates=[:pretest],
                   nperm=5_000, rng=StableRNG(1))
```

Inverting the tests of constant additive effects gives a confidence interval:

```@example tut
ri_confint(expt, :score, :treated; strata=:school, nperm=2_000, rng=StableRNG(2))
```

And a balance test compares the pretest imbalance with its randomization
distribution:

```@example tut
ri_balance_test(expt, :treated, [:pretest]; strata=:school, nperm=2_000,
                rng=StableRNG(4))
```

## Where to go next

- The methods guides ([DiD](did.md), [IV](iv.md), [RD](rd.md),
  [synthetic control](synth.md), [randomization inference](ri.md),
  [interference](sutva.md), [causal ML](ml.md)) describe every estimator, its
  assumptions and its options.
- [Results, tables and plots](@ref) covers `tidy`, `glance`, regression tables and
  the Makie plotting functions.
- The `examples/` directory of the repository has longer scripts for each area.
