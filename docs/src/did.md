# Difference-in-Differences

```@meta
CurrentModule = DrSnow
```

DrSnow's DiD tools cover the canonical two-group design, event studies, and
staggered adoption with heterogeneous treatment effects. Every estimator returns a
`CausalEstimate` with the full covariance matrix, so `coef`, `vcov`, `stderror`,
`confint(r; level)`, `coeftable` and `nobs` work uniformly. Joint tests use the full
covariance, and diagnostic tests return a `DiagnosticTest`.

```julia
using DrSnow
r  = did_twfe(df, :y, :d, :unit, :year)            # 2×2 / single adoption date
es = event_study(df, :y, :d, :unit, :year)         # dynamic effects
cs = did_callaway_santanna(df, :y, :d, :unit, :year; covariates=[:x])
aggregate_att(cs, :dynamic)                        # event-study summary
```

## Data layout and treatment timing

All functions take a long panel `(data, outcome, treatment, unit, time)`, one row
per unit and period. The treatment can be given in two ways:

- a **0/1 indicator** `D_it` (or `Bool`);
- a **first-treatment period** column, wrapped as [`FirstTreated`](@ref)`(:col)`.
  Never-treated units are coded `0` (as in R's `did`), `missing` or `Inf`; another
  code can be passed as `FirstTreated(:col; never=-1)`. Repeated cross-sections
  (unit = `nothing`) require this form.

[`treatment_timing`](@ref) builds the shared cohort layer ([`TreatmentTiming`](@ref))
used by every estimator:

- time is mapped to an **ordered period index** `1, …, T` (sorted unique values of
  the time column), so gaps (e.g. biennial data) and `Date`s are handled and event
  time `e = t - G_i` is counted in periods;
- the cohort `G_i` is the period index of the first treated period, with **`0` for
  never-treated units** (i.e. `G_i = ∞`);
- it records whether treatment is absorbing (no unit leaves treatment), whether the
  panel is balanced, and the number of anticipation periods.

Results never depend on the row order of the data: units and periods are matched by
key.

## Which estimator?

| Design | Recommended | Notes |
|:--|:--|:--|
| One adoption date, never-treated comparison group | [`did_twfe`](@ref), [`event_study`](@ref) | TWFE equals the 2×2 DiD. |
| Two periods, parallel trends conditional on covariates | [`did_drdid`](@ref) | Doubly robust (Sant'Anna & Zhao 2020). |
| Staggered adoption | [`did_callaway_santanna`](@ref), [`did_imputation`](@ref), [`did_sun_abraham`](@ref) | TWFE can be badly biased; see diagnostics below. |
| Staggered adoption, covariates needed for parallel trends | [`did_callaway_santanna`](@ref) with `covariates` | DR/IPW/OR kernels. |
| Most efficient under homoskedastic errors and parallel trends in all periods | [`did_imputation`](@ref) | Uses all pre-periods; stronger pre-trend assumption. |
| Staggered adoption, regression framework (covariates, Poisson/logit outcomes) | [`did_etwfe`](@ref) | Wooldridge's extended TWFE; equals imputation (not-yet-treated) or Sun–Abraham (never-treated). |
| Diagnosing an existing TWFE estimate | [`bacon_decomposition`](@ref), [`twfe_weights`](@ref) | Share and sum of negative weights. |
| Treatment switches on and off, or takes many values | [`did_multiplegt_dyn`](@ref) | de Chaisemartin & D'Haultfœuille (2026). |
| Continuous dose, two periods | [`did_continuous`](@ref) | `ATT(d|d)`, `ACRT`; strong parallel trends for comparisons across doses. |
| How robust is the conclusion to non-parallel trends? | [`honest_did`](@ref), [`honest_breakdown`](@ref) | Rambachan & Roth (2023) sensitivity analysis on an event study. |

`event_study(...; estimator=:auto)` uses TWFE with a single cohort and Sun–Abraham
with several cohorts; `estimator=:imputation` and `:callaway_santanna` are also
available and all return an [`EventStudyEstimate`](@ref).

## Two-way fixed effects

```math
Y_{it} = \alpha_i + \lambda_t + \tau D_{it} + X_{it}'\beta + \varepsilon_{it}
```

[`did_twfe`](@ref) estimates this regression with `FixedEffectModels`, clustering by
unit by default (`cluster`, or any `vcov` estimator). Confidence intervals use
`t(G − 1)` critical values under clustering. With a single adoption date and parallel
trends, `τ` is the ATT.

**Staggered adoption.** With several adoption dates, TWFE compares newly treated
units with already-treated ones. `τ` is then a weighted average of cohort-period
effects in which some weights can be negative (Goodman-Bacon 2021; de Chaisemartin &
D'Haultfœuille 2020; Sun & Abraham 2021; Borusyak, Jaravel & Spiess 2024). When
effects change over time, `τ` can even have the opposite sign of every underlying
effect. `did_twfe` detects staggered and non-absorbing designs and warns, reporting
the number and sum of negative weights.

- [`bacon_decomposition`](@ref) writes `τ` as a weighted average of all 2×2
  comparisons: treated vs never treated, earlier vs later treated, and later vs
  earlier treated. In the last type, already-treated units serve as controls.
- [`twfe_weights`](@ref) gives the weight of each treated unit-period effect, the
  sum of negative weights, and `σ_fe`: the smallest standard deviation of effects
  under which the ATT could be zero given `τ`.

## Event studies and pre-trends

```math
Y_{it} = \alpha_i + \lambda_t + \sum_{e \neq -1} \beta_e \, 1\{t - G_i = e\} + \varepsilon_{it}
```

[`event_study`](@ref) estimates dynamic effects by event time. For the TWFE
specification:

- `max_pre = max_post = nothing` (default) estimates every observed relative period
  (fully dynamic). This needs never-treated units, and non-identified coefficients
  raise an error rather than returning `0`/`NaN`.
- With a window `[-max_pre, max_post]`, `endpoints = :bin` pools periods beyond the
  window into the endpoint coefficients, and `endpoints = :trim` drops those
  observations of treated units. Periods outside the window are never pooled into
  the reference period.
- `omit_period` (default `-1 - anticipation`) must be inside the window, observed,
  and not a binned endpoint.

With several cohorts, TWFE event-study coefficients are contaminated by effects from
other periods (Sun & Abraham 2021). The default `estimator = :auto` therefore
switches to [`did_sun_abraham`](@ref).

[`pre_trend_test`](@ref) is a joint Wald test that all pre-period coefficients are
zero, using their full covariance: `F(q, G − 1)` for regression estimators, `χ²(q)`
for influence-function estimators. A rejection signals differential pre-trends or
anticipation. A non-rejection is **not** evidence of parallel trends. Such tests
often have low power, and conditioning an analysis on passing them distorts
inference (Roth 2022). Report the pre-period estimates and consider sensitivity
analysis with [`honest_did`](@ref) (Rambachan & Roth 2023, below), or an
equivalence-type approach that asks whether the data rule out economically relevant
pre-trends (Bilinski & Hatfield 2026).
[`parallel_trends_test`](@ref) is a convenience
wrapper that fits a fully dynamic event study and applies `pre_trend_test`; despite
its name it tests pre-trends only.

`confint(es; uniform=true)` gives simultaneous (sup-t) bands that cover all
event-time effects jointly. It uses the multiplier bootstrap for Callaway–Sant'Anna
and simulation from the estimated covariance otherwise (Montiel Olea &
Plagborg-Møller 2019). [`event_study_average`](@ref) averages event-time effects,
with a standard error from the full covariance.

## Doubly robust two-period DiD (Sant'Anna & Zhao 2020)

For two periods and a treatment group `D`, [`did_drdid`](@ref) estimates the ATT
under **conditional** parallel trends,
``E[Y_1(0) - Y_0(0) \mid X, D=1] = E[Y_1(0) - Y_0(0) \mid X, D=0]``, and overlap. It
works for balanced panels and repeated cross-sections (`unit = nothing`).

- `:dr_improved` (default) is the improved locally efficient DR estimator. It uses an
  inverse-probability-tilting (calibrated) propensity score and a weighted outcome
  regression, and is doubly robust for both estimation and inference.
- `:dr` is the traditional locally efficient DR estimator (logit + OLS).
- `:ipw` is normalized IPW, `:ipw_unnormalized` is Abadie (2005), and `:reg` is
  outcome regression.

Standard errors come from the estimators' influence functions, including the
estimation effect of the propensity score and outcome regressions. They can be
clustered with `cluster`. Controls with propensity score ≥ `trim_level` (0.995) are
trimmed, as in R's `DRDID`.

## Callaway & Sant'Anna (2021)

[`did_callaway_santanna`](@ref) estimates group-time effects

```math
ATT(g, t) = E[Y_t(g) - Y_t(\infty) \mid G = g]
```

Each is a 2×2 DiD of cohort `g` against a comparison group between period `t` and a
base period, computed with the kernels above (`method = :dr` by default, as in R).

- **Comparison group.** `control_group = :never_treated` uses never-treated units.
  `:not_yet_treated` also uses units not yet treated by `max(t, base) + anticipation`.
- **Anticipation.** `anticipation = δ` makes `g − 1 − δ` the last clean pre-period.
- **Base period.** With `base_period = :varying`, pre-treatment cells compare
  consecutive periods (placebo effects) and post-treatment cells use `g − 1 − δ`.
  With `:universal`, all cells use `g − 1 − δ`, and that reference cell is omitted.
  Post-treatment estimates are identical, but the pre-treatment coefficients differ:
  varying-base placebos are short differences, not deviations from a common
  reference period, so state which was used (Roth 2026).
- **Covariates.** Values are taken from the earlier period of each comparison.
  Sampling weights are supported with `weights`, and `cluster` sets a unit-invariant
  cluster variable.
- **Inference.** Analytic standard errors come from influence functions.
  `bootstrap = true` (default) runs a multiplier bootstrap with Rademacher weights at
  the cluster level, used for sup-t simultaneous bands. Pass `rng` for
  reproducibility.

[`aggregate_att`](@ref) reproduces R's `aggte`. Its influence-function standard
errors account for the estimated cohort-share weights.

- `:simple`: the average of post-treatment `ATT(g,t)`, weighted by cohort size.
- `:group`: `θ(g)` per cohort, plus the cohort-size weighted overall effect.
- `:dynamic`: `θ(e)` by event time, returned as an [`EventStudyEstimate`](@ref). It
  supports `balance_e` (a fixed cohort composition), `min_e` and `max_e`, and
  `details.overall` holds the average over `e ≥ 0`.
- `:calendar`: `θ(t)` by calendar period.

`pre_trend_test(cs)` tests all pre-treatment `ATT(g,t)` jointly.

Units treated in the first period (after anticipation) are dropped, as are units not
observed in every period. With `unit = nothing` and a `FirstTreated` cohort column,
the estimator handles repeated cross-sections (every observation is its own unit, as
R's `did` with `panel = FALSE`). Without never-treated units, the periods from the
last cohort's treatment onwards are dropped, following R's `did`.

## Sun & Abraham (2021)

[`did_sun_abraham`](@ref) estimates a fully saturated regression with cohort ×
relative-period indicators, using never-treated units (or, failing those, the last
treated cohort) as controls. It then averages the cohort-specific effects at each
event time with cohort-share weights. The covariance of the aggregated effects is
`W V W'`, treating the shares as fixed, as in `fixest::sunab`. `details.att` holds
the average post-treatment effect weighted by cell size (fixest's `agg = "ATT"`),
and `details.cohort_effects` holds the `CATT(g, e)`.

## Borusyak, Jaravel & Spiess (2024) imputation

[`did_imputation`](@ref) proceeds in three steps:

1. Fit unit and time effects (and covariates) on untreated observations only.
2. Impute `Y_it(0)` for treated observations.
3. Average `τ̂_it = Y_it − Ŷ_it(0)`: over all treated observations for the ATT, or by
   horizon `h` for event-time effects.

Standard errors use the paper's conservative clustered variance, with `τ̂` averaged
within cohort × period cells.

`pretrends = K` adds BJS Test 1. It regresses `Y` on the fixed effects and `K` lead
indicators, using untreated observations only. The leads enter the returned event
study as negative relative periods, with their covariance joint with the horizon
effects, so [`pre_trend_test`](@ref) applies. The estimator is efficient under
homoskedastic errors when parallel trends holds in all periods. It relies on that
assumption more heavily than Callaway–Sant'Anna with a varying base period does.

## Extended two-way fixed effects (Wooldridge 2023, 2025)

[`did_etwfe`](@ref) keeps the regression framework but saturates it with one
treatment dummy per treated cohort × period cell,

```math
Y_{it} = \alpha_{g} + \lambda_t + \sum_{(g,s):\, s \ge g} \tau_{gs}\, 1\{G_i = g,\ t = s\}
         + \varepsilon_{it},
```

so each `τ_gs` is an `ATT(g, s)` under parallel trends and no anticipation, and the
negative weights of the single-coefficient TWFE regression disappear.

- **Equivalences.** With not-yet-treated controls this pooled OLS is numerically
  the imputation estimator of Borusyak, Jaravel & Spiess (2024) (Wooldridge 2025);
  cohort or unit fixed effects (`fe`) give the same `τ̂` in a balanced panel. With
  `control_group = :never_treated` every period except `g − 1` gets a cell, which
  reproduces the Sun–Abraham interaction-weighted event study. The test suite
  checks both equalities.
- **Covariates** are demeaned by cohort and interacted with every treatment cell,
  and enter with covariate × cohort and covariate × period terms (as in R's
  `etwfe`). This allows parallel trends conditional on covariates with effects that
  vary with them.
- **Nonlinear models** (Wooldridge 2023). `family = :poisson` or `:logit` fits an
  exponential or logistic conditional mean with cohort and period dummies by
  quasi-maximum likelihood. Parallel trends is then assumed on the index scale
  (e.g. proportional trends for Poisson), which suits counts, non-negative and
  binary outcomes. Cell effects are average differences of predicted responses
  with and without the cell's treatment terms.
- **Aggregation.** [`aggregate_att`](@ref) returns observation-weighted averages
  of the cell effects (`:simple`, `:group`, `:calendar`, `:dynamic`) with
  delta-method standard errors, as `etwfe::emfx` does.

## Non-binary and non-absorbing treatments (de Chaisemartin & D'Haultfœuille 2026)

When units switch treatment on and off, or the treatment takes several values,
cohorts and event time are not defined, and TWFE regressions (static or dynamic)
mix effects with negative weights. [`did_multiplegt_dyn`](@ref) estimates the
effect of having switched from one's period-one treatment for `ℓ` periods,

```math
\text{DID}_\ell = \frac{1}{N_\ell} \sum_{g:\, F_g - 1 + \ell \le T_g} S_g
  \Big[ (Y_{g,F_g-1+\ell} - Y_{g,F_g-1})
  - \overline{(Y_{g',F_g-1+\ell} - Y_{g',F_g-1})} \Big],
```

where `F_g` is the first period in which the treatment of `g` differs from its
period-one value, `S_g = ±1` for switchers in and out, and the average is over
groups `g'` with the same period-one treatment that have not switched by
`F_g − 1 + ℓ` (or never switch, with `only_never_switchers = true`).

- **Assumptions.** No anticipation, and parallel trends in the absence of any
  treatment change for groups with the same period-one treatment. Effects may be
  heterogeneous and dynamic: `DID_ℓ` is the effect of each switcher's actual
  treatment path, including later changes.
- `placebo = k` adds placebo estimators comparing `Y_{F_g−1−ℓ}` with `Y_{F_g−1}`.
  They are returned at negative event times, so [`pre_trend_test`](@ref) is the
  joint placebo test.
- `normalized = true` divides `DID_ℓ` by the average cumulative treatment change,
  giving an average effect of a one-unit change in current and lagged treatments.
  `details.average_total_effect` is the average total effect per unit of
  treatment.
- Standard errors follow the R package: group-level influence functions, demeaned
  within cohorts of switchers and of comparison groups (conservative), clustered at
  the group level or by `cluster`.

The result is an [`EventStudyEstimate`](@ref) with `e = ℓ − 1` for effect `ℓ` and
`e = −1 − ℓ` for placebo `ℓ`, so it works with `confint(...; uniform=true)`,
[`event_study_average`](@ref) and [`honest_did`](@ref). Time-varying `controls`
(residualized first differences, as in the R package) and non-parametric trends
(`trends_nonparam`) are supported; the `trends_lin`, `continuous`,
`predict_het` and bootstrap options of the R package are not.

## Continuous treatments (Callaway, Goodman-Bacon & Sant'Anna, forthcoming)

With two periods and a dose `D ≥ 0` received in the second period (`D = 0`:
untreated), [`did_continuous`](@ref) estimates the dose-response of the outcome
change among treated units with a B-spline and compares it with the change of the
untreated group.

- `ATT(d|d) = E[ΔY | D = d] − E[ΔY | D = 0]`, the effect of dose `d` for the units
  that chose it, is identified under **parallel trends** across all doses. Here
  `ATT(d|d') = E[Y_2(d) − Y_2(0) | D = d']`.
- The causal response to a marginal change in the dose among units that chose `d`
  is `ACRT(d|d) = ∂ATT(l|d)/∂l` at `l = d`. The slope of the estimated level curve,
  `∂E[ΔY | D = d]/∂d`, equals `ACRT(d|d)` plus a selection-bias term
  `∂ATT(d|l)/∂l` at `l = d`, so under parallel trends alone comparisons across doses
  are not causal. Under **strong parallel trends** (no selection into doses on
  treatment effects) the level curve identifies `ATE(d) = E[Y_2(d) − Y_2(0)]` and its
  slope the average causal response `ACR(d) = ∂ATE(d)/∂d`.
- The aggregates average the curves over the treated units' doses: `ATT^o` is the
  average of `ATT(D|D)`, and `ACRT^o` is the average estimated slope, which equals
  the average `ACR(D)` under strong parallel trends and otherwise contains the
  selection bias.

Standard errors treat the spline as a fixed sieve without bias correction. Choose
the degree and knots before looking at the results, and report the curves with
their simultaneous bands (`confint(r; curve=:att, uniform=true)`).

## Sensitivity to violations of parallel trends (Rambachan & Roth 2023)

Pre-trend tests cannot establish parallel trends. [`honest_did`](@ref) instead
asks how large post-treatment violations `δ_post` would have to be to overturn a
conclusion, bounding them by the pre-treatment ones. For a target `θ = l'τ_post`
(one event time, or an average with `target = 0:3`) it reports confidence sets
valid for every `δ` in one of these sets:

- **relative magnitudes** `Δ^RM(M̄)`: each post-treatment change in the trend
  difference is at most `M̄` times the largest pre-treatment change (`M̄ = 1`
  allows violations as large as the worst pre-period movement).
  `bound = :linear_trend` applies the bound to deviations from a linear pre-trend;
- **smoothness** `Δ^SD(M)`: the slope of the trend difference changes by at most
  `M` per period (`M = 0` allows exactly linear trends);
- either one intersected with a **sign** (`bias_sign`) or **monotonicity**
  (`monotonicity`) restriction.

The methods follow the HonestDiD R package: the optimal fixed-length confidence
interval (FLCI) for `Δ^SD`, and the conditional and hybrid tests of Andrews, Roth
& Pakes (2023) inverted over a θ grid. The default is the hybrid with a
least-favorable first stage for `Δ^RM`, and with an FLCI first stage for `Δ^SD`
with sign or shape restrictions. The linear programs are solved by a small
built-in dense simplex, so no optimization package is needed.
[`honest_breakdown`](@ref) returns the breakdown value: the smallest `M` (or `M̄`)
at which a zero effect is no longer rejected.

```julia
cs = did_callaway_santanna(mpdta, :lemp, FirstTreated(:first_treat), :countyreal,
                           :year; base_period=:universal)
es = aggregate_att(cs, :dynamic)
honest_did(es; restriction=:relative_magnitudes, M=0:0.5:2)
honest_did(es; restriction=:smoothness, M=[0, 0.01, 0.02], target=0:3)
honest_breakdown(es; restriction=:relative_magnitudes)
```

The event study must normalize a reference period and have consecutive relative
periods: TWFE, Sun–Abraham, Callaway–Sant'Anna with `base_period = :universal`,
`did_multiplegt_dyn`, and ETWFE with never-treated controls qualify; binned
endpoints and the imputation estimator (whose pre-trend coefficients have no
common reference period) are rejected. Inference relies on `β̂ ≈ N(β, Σ)` with the
estimated covariance.

## Covariate balance

[`pretreatment_balance`](@ref) compares covariate means of each cohort with its
comparison group, in the cohort's own pre-treatment periods, and reports
standardized differences. It is descriptive: large differences suggest conditioning
parallel trends on covariates, while small ones do not establish comparability on
unobservables.

## Validation

The test suite compares DrSnow with R reference implementations on published data
(`mpdta`, LaLonde/CPS, DRDID's simulated cross-sections). The files and the script
that generated them are in `test/validation/did/`.

| DrSnow | Reference | Agreement |
|:--|:--|:--|
| `did_drdid` (5 methods; panel and RC; weighted) | DRDID 1.3.0 | estimates ≤ 4e-9 relative; SEs ≤ 1e-9 |
| `did_callaway_santanna` + `aggregate_att` (9 settings × 6 aggregations) | did 2.5.1 | ATT ≤ 5e-11; SEs ≤ 2e-10 relative |
| `did_callaway_santanna`, repeated cross-sections (4 settings × 4 aggregations) | did 2.5.1 (`panel = FALSE`) | ATT ≤ 5e-11; SEs ≤ 2e-10 relative |
| `did_twfe`, binned TWFE event study, `did_sun_abraham` | fixest 0.14.2 | estimates ≤ 1e-14; SEs within 0.02% |
| `did_imputation` (ATT, horizons, pre-trends) | didimputation 0.5.1 | estimates ≤ 5e-9; SEs ≤ 1e-10 relative |
| `bacon_decomposition` | bacondecomp 0.1.1 | exact (≤ 1e-15) |
| `twfe_weights` | TwoWayFEWeights 2.1.0 | weights and sums exact |
| `honest_did`: conditional and C-F sets (ΔSD, ΔSD with sign/shape, ΔRM, ΔRM with sign/shape, ΔSDRM; 1 and 4 post periods) | HonestDiD 0.2.8 | identical accepted θ-grid points |
| `honest_did`: C-LF hybrid sets | HonestDiD 0.2.8 | within 2 grid steps (simulated least-favorable critical values) |
| `honest_did`: FLCI | HonestDiD 0.2.8 | half-length within 0.2% (R simulates folded-normal quantiles); DrSnow's optimum is never longer under the exact criterion |
| `did_multiplegt_dyn` (17 settings: clustering, weights, normalized, in/out switchers, never-switcher controls, same switchers, `trends_nonparam`, `controls`, missing cells) | DIDmultiplegtDYN 2.4.0 | effects, placebos, SEs and average total effect ≤ 1e-12; joint tests |
| `did_etwfe` (linear, covariates, never-treated, unit FE, Poisson, logit) + `aggregate_att` | etwfe 0.6.2 (`emfx`) | linear estimates ≤ 1e-13, SEs within 0.02%; Poisson/logit estimates ≤ 1e-9 relative, SEs within 1e-4 |

Four convention differences are known:

- Cluster-robust regression standard errors differ from `fixest` by a constant
  factor of about 1.0002. FixedEffectModels counts one more parameter in the
  `(n − 1)/(n − K)` correction.
- `twfe_weights().sigma_fe` uses the paper's population standard deviation of the
  weights, whereas the R package divides by `n − 1`.
- For Poisson/logit ETWFE, `fixest` multiplies the model-based covariance by
  `(n − 1)/(n − K)`; DrSnow reports the unadjusted maximum-likelihood covariance.
- DIDmultiplegtDYN 2.4.0 returns an effect/placebo covariance (`coef$vcov`) whose
  off-diagonal terms are miscomputed; DrSnow's covariance agrees with the one the
  package uses for its joint tests.

Monte Carlo checks in the test suite cover:

- coverage of the confidence intervals;
- size of the pre-trend tests;
- coverage of the uniform bands.

Set `DRSNOW_SLOW_TESTS=true` for full replication counts.

## References

- Andrews, I., Roth, J., & Pakes, A. (2023). Inference for linear conditional moment
  inequalities. *Review of Economic Studies*, 90(6), 2763–2791.
- Armstrong, T. B., & Kolesár, M. (2018). Optimal inference in a class of regression
  models. *Econometrica*, 86(2), 655–683.
- Abadie, A. (2005). Semiparametric difference-in-differences estimators. *Review of
  Economic Studies*, 72(1), 1–19.
- Borusyak, K., Jaravel, X., & Spiess, J. (2024). Revisiting event-study designs: Robust
  and efficient estimation. *Review of Economic Studies*, 91(6), 3253–3285.
- Bilinski, A., & Hatfield, L. A. (2026). Nothing to see here? A non-inferiority
  approach to parallel trends. *Statistics in Medicine*, 45(3–5), e70296.
- Callaway, B., Goodman-Bacon, A., & Sant'Anna, P. H. C. (forthcoming).
  Difference-in-differences with a continuous treatment. *American Economic Review*.
  Earlier version: NBER Working Paper 32117 (2024).
- Callaway, B., & Sant'Anna, P. H. C. (2021). Difference-in-differences with multiple
  time periods. *Journal of Econometrics*, 225(2), 200–230.
- de Chaisemartin, C., & D'Haultfœuille, X. (2020). Two-way fixed effects estimators
  with heterogeneous treatment effects. *American Economic Review*, 110(9), 2964–2996.
- de Chaisemartin, C., & D'Haultfœuille, X. (2026). Difference-in-differences estimators
  of intertemporal treatment effects. *Review of Economics and Statistics*, 108(4),
  863–880.
- Goodman-Bacon, A. (2021). Difference-in-differences with variation in treatment
  timing. *Journal of Econometrics*, 225(2), 254–277.
- Graham, B. S., Pinto, C. C. de X., & Egel, D. (2012). Inverse probability tilting for
  moment condition models with missing data. *Review of Economic Studies*, 79(3),
  1053–1079.
- Montiel Olea, J. L., & Plagborg-Møller, M. (2019). Simultaneous confidence bands:
  Theory, implementation, and an application to SVARs. *Journal of Applied
  Econometrics*, 34(1), 1–17.
- Rambachan, A., & Roth, J. (2023). A more credible approach to parallel trends. *Review
  of Economic Studies*, 90(5), 2555–2591.
- Roth, J. (2022). Pretest with caution: Event-study estimates after testing for
  parallel trends. *American Economic Review: Insights*, 4(3), 305–322.
- Roth, J. (2026). Interpreting event-studies from recent difference-in-differences
  methods. *The Japanese Economic Review*, 77(2), 275–288.
- Sant'Anna, P. H. C., & Zhao, J. (2020). Doubly robust difference-in-differences
  estimators. *Journal of Econometrics*, 219(1), 101–122.
- Sun, L., & Abraham, S. (2021). Estimating dynamic treatment effects in event studies
  with heterogeneous treatment effects. *Journal of Econometrics*, 225(2), 175–199.
- Wooldridge, J. M. (2023). Simple approaches to nonlinear difference-in-differences
  with panel data. *The Econometrics Journal*, 26(3), C31–C66.
- Wooldridge, J. M. (2025). Two-way fixed effects, the two-way Mundlak regression, and
  difference-in-differences estimators. *Empirical Economics*, 69(5), 2545–2587.

The functions and types described on this page are documented in the [API reference](reference/did.md).
