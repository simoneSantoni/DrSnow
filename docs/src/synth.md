# Synthetic control methods

```@meta
CurrentModule = DrSnow
```

This page covers comparative case studies and panels with few treated units:

| Function | Method | Treated units | Inference |
|:--|:--|:--|:--|
| [`synthetic_did`](@ref) | Synthetic DiD; synthdid's SC and DiD (`method`) | one or more, block or staggered | placebo, bootstrap, jackknife |
| [`synthetic_control`](@ref) | Classic Abadie–Diamond–Hainmueller SC with predictors and V | exactly one | in-space placebo permutation test |
| [`augmented_synthetic_control`](@ref) | Ridge-augmented SC (augsynth) | one or more, common adoption | placebo, unit jackknife; conformal |
| [`matrix_completion`](@ref) | MC-NNM (nuclear-norm matrix completion) | any absorbing pattern | placebo / bootstrap (approximate) |

All estimators return a `CausalEstimate` with a single coefficient `"ATT"` and share
the accessors [`synth_weights`](@ref), [`synth_gaps`](@ref) (treated vs. synthetic
paths) and, for synthetic DiD, [`synth_time_weights`](@ref) and [`synth_cohorts`](@ref).

## Data preparation

[`synth_panel`](@ref) turns long data (`outcome`, a 0/1 `treatment` indicator, `unit`,
`time`) into a balanced units × periods matrix. Cells are matched by the
`(unit, time)` key, so results do not depend on the order of the input rows. Rows are
ordered never-treated units first (sorted by id), then treated units by adoption
period. Treatment must be *absorbing*: once a unit is treated it stays treated, and
every treated unit needs at least one untreated period. The adoption period of a unit
is its first treated period; units never treated form the donor pool.

Unbalanced panels are rejected with an error that reports the missing cells. No
imputation is attempted: drop units with gaps or restrict the period range before
estimation (matrix completion can handle missing *untreated* cells in principle, but
DrSnow keeps a single balanced-panel contract across the area). Covariates may contain
missing values; the classic SC averages predictors ignoring missing values, while
synthetic DiD requires complete time-varying covariates.

All estimators accept either a `SynthPanel` or `(data, outcome, treatment, unit, time)`.

## Synthetic difference-in-differences

[`synthetic_did`](@ref) implements Arkhangelsky et al. (2021) exactly as the authors'
R package `synthdid`. With `N₀` never-treated units, `T₀` pre-periods and treated
averages over `N₁` units and `T₁` post-periods, unit weights `ω` (on the simplex, with
an intercept) solve

```math
\min_{\omega_0,\omega}\; \sum_{t\le T_0}\Big(\omega_0 + \sum_{i\le N_0}\omega_i Y_{it}
  - \bar Y_{\text{tr},t}\Big)^2 + \zeta_\omega^2 T_0 \lVert\omega\rVert_2^2 ,
```

time weights `λ` solve the analogous problem across control units with a tiny ridge
`ζ_λ`, and the estimate is

```math
\hat\tau = \Big(\bar Y_{\text{tr},\text{post}} - \sum_t \lambda_t \bar Y_{\text{tr},t}\Big)
 - \sum_i \omega_i\Big(\bar Y_{i,\text{post}} - \sum_t \lambda_t Y_{it}\Big).
```

The regularisation is `ζ_ω = (N₁T₁)^{1/4} σ̂` and `ζ_λ = 10⁻⁶ σ̂`, with `σ̂` the
standard deviation of first differences of control outcomes before treatment. Weights
are computed by Frank–Wolfe with synthdid's sparsification (weights below a quarter of
the largest are zeroed and the problem re-solved). `method = :sc` gives synthdid's
penalised synthetic control (no time weights, no intercept, `ζ_ω = 10⁻⁶ σ̂`) and
`method = :did` the difference-in-differences estimator (uniform weights).

**Staggered adoption.** Each adoption cohort is compared with the never-treated units
over all periods (not-yet-treated units are never controls), and cohort estimates are
averaged with weights proportional to their number of treated unit-periods
(Arkhangelsky et al. 2021, appendix; Clarke et al. 2024). [`synth_cohorts`](@ref)
reports the cohort estimates, weights and regularisation.

**Covariates.** `covariate_method = :optimized` (default) estimates the coefficients
jointly with the weights, as synthdid does; `:projected` residualises the outcome on
the covariates with unit and time fixed effects estimated on never-treated units
(Kranz 2021), then applies SDID. Covariate adjustment only helps if the covariates are
not themselves affected by the treatment.

**Variance.** Three estimators from the paper:

- `:placebo` (Algorithm 4, default): the treated units' adoption pattern is assigned
  to randomly chosen never-treated units and the estimator is re-run on controls only.
  It is the only option with a single treated unit, needs more controls than treated
  units, and assumes treated and control units have the same noise distribution
  (homoskedasticity across units).
- `:bootstrap` (Algorithm 2): units are resampled; needs at least two treated units
  and is reliable only with a moderate number of treated units.
- `:jackknife` (Algorithm 3): leave-one-unit-out with the weights held fixed; needs at
  least two treated units in every cohort. It can be conservative.

Replicates re-estimate the weights warm-started from the full-sample weights and keep
the full-sample regularisation, like synthdid. Confidence intervals use the normal
approximation.

## Classic synthetic control

[`synthetic_control`](@ref) implements Abadie & Gardeazabal (2003) and Abadie,
Diamond & Hainmueller (2010, 2015) for one treated unit. Donor weights `W ≥ 0`,
`ΣW = 1`, minimise the `V`-weighted distance between the treated unit's predictors
`X₁` and the synthetic unit's `X₀W`:

```math
W^*(V) = \arg\min_{W\in\Delta} (X_1 - X_0 W)' V (X_1 - X_0 W),\qquad
V^* = \arg\min_{V} \frac{1}{|\mathcal T|}\sum_{t\in\mathcal T}
      \big(Y_{1t} - Y_{0t}'W^*(V)\big)^2 ,
```

where `𝒯` are the `fit_periods`. Predictors are variables averaged over
`predictor_periods` (`predictors`) or over their own windows (`special_predictors`,
e.g. `:gdpcap => 1960:1969`), standardised by their standard deviation across units as
in the R package `Synth`. The inner problem is solved exactly; the outer search uses
Nelder–Mead from Synth's two starting points (equal weights and a regression-based
`V`), with optional random starts (`v_starts`). A user-supplied `v` skips the search.

The nested problem is not convex. In the Basque data the default starts reproduce
Synth's weights (Cataluña 0.85, Madrid 0.15), but random starts find a `V` with half the
pre-treatment MSPE and different donors, so weight-based narratives should be checked
for sensitivity to `V`. Without predictors, the outcome in every pre-period is used
with equal weights (outcome-only SC).

**Diagnostics and inference.**

- Pre-treatment fit: `pre_rmspe`, `predictor_balance`, [`synth_gaps`](@ref). Abadie
  et al. (2015) and Abadie (2021) advise against using SC when the pre-treatment fit is
  poor.
- [`synth_in_space_placebo`](@ref): each donor is treated in turn (re-optimising `V`)
  and the treated unit's post/pre RMSPE ratio (or average gap) is ranked among the
  placebos; `p = (1 + #placebos at least as extreme)/(1 + J)`. The smallest attainable
  p-value is `1/(J+1)`. `pre_rmspe_cutoff` drops placebos with poor pre-fit (ADH 2010).
  With `placebo_pool = :all` the treated unit is a potential donor in every placebo fit,
  which makes the test a symmetric permutation test of the sharp null.
- `vcov` returns the placebo variance of the ATT (variance of the placebo ATTs), which
  assumes equal noise variance for treated and donor units.
- [`synth_leave_one_out`](@ref) drops each donor with positive weight;
  [`synth_in_time_placebo`](@ref) backdates the intervention.

## Augmented synthetic control and conformal inference

[`augmented_synthetic_control`](@ref) implements the ridge-augmented SCM of
Ben-Michael, Feller & Rothstein (2021) following augsynth. SCM weights fitted to all
pre-treatment outcomes are corrected by a ridge regression of outcomes on pre-treatment
outcomes, `γ̂_aug = γ̂ + X₀(X₀'X₀ + λI)⁻¹(x₁ − X₀'γ̂)`; the correction lets weights be
negative, extrapolating outside the convex hull of donors when the synthetic control
alone fits poorly. `λ` is chosen by leave-one-period-out cross-validation with the
one-standard-error rule. `ridge = false` gives plain outcome-only SCM;
`fixed_effects = true` first removes unit pre-period means.

The default standard error is the placebo variance (the treated units' role is given
to randomly chosen never-treated units, `λ` held fixed), as for synthetic DiD.
`se_method = :jackknife` gives augsynth's leave-one-unit-out jackknife over donors with
non-zero weight (and over treated units when there are several). With a single treated
unit the treated unit is never dropped, so the jackknife ignores the treated unit's own
noise: in DrSnow's Monte Carlo its 95% interval covered about two thirds of the time
with one treated unit, and was conservative with five treated units. Use it only with
several treated units.

[`synth_conformal_inference`](@ref) implements Chernozhukov, Wüthrich & Zhu (2021)
for an augmented or plain SC fit: for each post-period, the null effect is imposed, the
model is refitted on the pre-periods plus that period, and its residual is ranked among
all residuals under moving-block (`:block`) or random (`:iid`) permutations.
Intervals invert the tests over a grid; `truncated = true` flags intervals that reach
the grid edge. Because p-values are multiples of `1/(T₀+1)`, nothing can be rejected
unless `1/(T₀+1) < 1 − level`: with 19 pre-periods, 95% intervals are the whole grid.
Validity requires residuals that are stationary and weakly dependent (block) or
exchangeable over time (iid).

## Matrix completion

[`matrix_completion`](@ref) implements MC-NNM (Athey et al. 2021) with the authors'
MCPanel algorithm: untreated potential outcomes are modelled as
`L + u1' + 1v'` with unpenalised unit and time effects and a nuclear-norm penalty on
`L`, fitted on untreated cells by coordinate descent with singular-value
soft-thresholding along a warm-started path of penalties. `λ` is chosen by
cross-validation on randomly held-out untreated cells (`n_folds`, `cv_ratio`, `rng`).
The method uses untreated periods of treated units and handles staggered adoption.

Inference is approximate: `:placebo` reassigns the treated units' adoption patterns to
random never-treated units, `:bootstrap` resamples units; both hold `λ` at its
full-sample value, ignoring its selection.

## Assumptions and guidance

- **No anticipation and no interference.** Outcomes before adoption are untreated, and
  donors are unaffected by the treatment (no spillovers; see the SUTVA tools).
- **Pre-treatment fit and the convex hull.** Classic SC requires the treated unit's
  pre-treatment path (and predictors) to be approximately a convex combination of the
  donors'. Check `pre_rmspe` and the gap plot. ASCM and SDID relax this by allowing
  extrapolation or an intercept shift, respectively.
- **Enough pre-periods.** Bias bounds for SC shrink with the number of pre-periods
  relative to the noise (Abadie et al. 2010); SDID and conformal inference need
  several pre-periods too.
- **Donor pool.** Restrict donors to units plausibly driven by the same factors and
  unaffected by the intervention (Abadie 2021).
- **Few treated units.** With one treated unit, inference relies on exchangeability
  (placebo tests, placebo variance) or on stationarity over time (conformal); the
  attainable p-values are bounded below by `1/(J+1)` or `1/(T₀+1)`. Report these
  bounds, compare several estimators, and do not read a non-rejection as evidence of no
  effect.
- **Staggered adoption.** Use [`synthetic_did`](@ref) or [`matrix_completion`](@ref);
  classic SC and ASCM here are for a single adoption date.

## Relation to other Julia packages

SynthControl.jl (with TreatmentPanels.jl) provides a simple SCM and a synthetic DiD for
a single treated unit (placebo SE only), and a port of `fect`. DrSnow implements the
estimators natively to cover several and staggered treated units, synthdid's full
variance menu, predictor-based ADH synthetic control with placebo inference, ASCM,
conformal inference and MC-NNM without a JuMP/HiGHS dependency. Loading TreatmentPanels
activates an extension: `synth_panel(bp)` converts a `TreatmentPanels.BalancedPanel`,
the estimators accept one directly, and `TreatmentPanels.BalancedPanel(p)` converts a
`SynthPanel` back for use with SynthControl.jl.

## Validation

`test/validation/synth/generate_references.R` produces the reference values used by the
test suite:

- **synthdid** (Prop 99 and simulated designs): SDID, SC and DiD estimates, weights,
  effect curves, regularisation, jackknife SEs and replicate-by-replicate placebo and
  bootstrap estimates agree to numerical precision (`≤ 1e-7` relative); Monte Carlo SEs
  agree within simulation error. Prop 99: SDID −15.60, SC −19.62, DiD −27.35
  (Arkhangelsky et al. 2021, Table 1).
- **augsynth**: ridge ASCM `λ`, weights, ATT path, jackknife SEs and block conformal
  p-values and intervals on Prop 99.
- **Synth**: Basque Country predictors and, given Synth's `V`, identical weights; the
  nested search reaches Synth's pre-treatment MSPE.
- **MCPanel**: MC-NNM fits at fixed `λ` on Prop 99.

Monte Carlo checks (`DRSNOW_SLOW_TESTS=true` for full replication counts) cover the
coverage of SDID placebo, bootstrap and jackknife intervals, the size of the in-space
placebo and conformal tests, and the coverage of MC-NNM placebo intervals.

## References

- Abadie, A. (2021). Using synthetic controls: Feasibility, data requirements, and
  methodological aspects. *Journal of Economic Literature*, 59(2), 391–425.
- Abadie, A., Diamond, A., & Hainmueller, J. (2010). Synthetic control methods for
  comparative case studies: Estimating the effect of California's tobacco control
  program. *Journal of the American Statistical Association*, 105(490), 493–505.
- Abadie, A., Diamond, A., & Hainmueller, J. (2015). Comparative politics and the
  synthetic control method. *American Journal of Political Science*, 59(2), 495–510.
- Abadie, A., & Gardeazabal, J. (2003). The economic costs of conflict: A case study of
  the Basque Country. *American Economic Review*, 93(1), 113–132.
- Arkhangelsky, D., Athey, S., Hirshberg, D. A., Imbens, G. W., & Wager, S. (2021).
  Synthetic difference-in-differences. *American Economic Review*, 111(12), 4088–4118.
- Athey, S., Bayati, M., Doudchenko, N., Imbens, G., & Khosravi, K. (2021). Matrix
  completion methods for causal panel data models. *Journal of the American Statistical
  Association*, 116(536), 1716–1730.
- Ben-Michael, E., Feller, A., & Rothstein, J. (2021). The augmented synthetic control
  method. *Journal of the American Statistical Association*, 116(536), 1789–1803.
- Chernozhukov, V., Wüthrich, K., & Zhu, Y. (2021). An exact and robust conformal
  inference method for counterfactual and synthetic controls. *Journal of the American
  Statistical Association*, 116(536), 1849–1864.
- Clarke, D., Pailañir, D., Athey, S., & Imbens, G. (2024). On synthetic
  difference-in-differences and related estimation methods in Stata. *The Stata
  Journal*, 24(4), 557–598.
- Kranz, S. (2021). Synthetic difference-in-differences with time-varying covariates.
  Mimeo.

The functions and types described on this page are documented in the [API reference](reference/synth.md).
