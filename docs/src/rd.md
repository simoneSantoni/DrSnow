# Regression discontinuity designs

```@meta
CurrentModule = DrSnow
```

DrSnow implements local polynomial regression discontinuity (RD) methods in pure Julia,
following the algorithms of the R/Stata packages `rdrobust`, `rddensity` and
`rdlocrand` (Calonico, Cattaneo, Farrell & Titiunik; Cattaneo, Jansson & Ma;
Cattaneo, Titiunik & Vazquez-Bare), and of the R package `RDHonest` (Armstrong &
Kolesár; Kolesár & Rothe) for honest (bias-aware) inference. Estimates, standard errors,
confidence intervals, bandwidths, density tests and RD plot bins reproduce `rdrobust`
4.0.0 and `rddensity` 3.0 to about 1e-9 relative error, and honest inference reproduces
`RDHonest` to floating-point accuracy at a given bandwidth. See
[Validation](@ref rd-validation).

| Task | Function | R/Stata analogue |
|---|---|---|
| Point estimation and inference (sharp, fuzzy, kink) | [`rd_estimate`](@ref) | `rdrobust` |
| Bandwidth selection | [`rd_bandwidth`](@ref) | `rdbwselect` |
| Plot data (binned means, global polynomial) | [`rd_plot_data`](@ref) | `rdplot` |
| Manipulation (density) test | [`rd_density_test`](@ref), [`rd_density_bandwidth`](@ref) | `rddensity`, `rdbwdensity` |
| McCrary (2008) binned density test | [`rd_mccrary_test`](@ref) | `rdd::DCdensity` |
| Honest (bias-aware) CIs under a bound on the second derivative (sharp, fuzzy, at a point) | [`rd_honest`](@ref) | `RDHonest` |
| Bias-aware Anderson–Rubin set for fuzzy RD | [`rd_honest_ar_confidence_set`](@ref) | Noack & Rothe (2024) |
| Discrete running variable: bounded misspecification CIs, lower bound on `M` | [`rd_honest_bme`](@ref), [`rd_smoothness_bound`](@ref) | `RDHonestBME`, `RDSmoothnessBound` |
| Covariate balance, placebo cutoffs, donut hole, bandwidth sensitivity | [`rd_covariate_balance`](@ref), [`rd_placebo_cutoffs`](@ref), [`rd_donut`](@ref), [`rd_bandwidth_sensitivity`](@ref) | workflow of Cattaneo, Idrobo & Titiunik (2020) |
| Fuzzy RD inference robust to a weak first stage | [`rd_weak_iv_confidence_set`](@ref) | Feir, Lemieux & Marmer (2016) |
| Local randomization inference | [`rd_randomization_test`](@ref), [`rd_window_selection`](@ref) | `rdrandinf`, `rdwinselect` |

## Design and estimands

Units have a score (running variable) ``X`` and are treated when ``X \ge c``.

- **Sharp RD** (default). Treatment is determined by the score. The estimand is the
  average treatment effect at the cutoff,
  ``\tau = \lim_{x \downarrow c} E[Y \mid X = x] - \lim_{x \uparrow c} E[Y \mid X = x]``.
  It is identified if ``E[Y(1) \mid X = x]`` and ``E[Y(0) \mid X = x]`` are continuous at
  ``c``.
- **Fuzzy RD** (`treatment = :d`). The probability of treatment jumps at the cutoff. The
  estimand is the ratio of the outcome jump to the treatment jump. Under continuity,
  monotonicity (no defiers) and a non-zero first stage it is a local average treatment
  effect for compliers at the cutoff.
- **Kink designs** (`deriv = 1`). The slope, not the level, of the treatment rule
  changes at the cutoff. The sharp kink estimand is the change in the slope of
  ``E[Y \mid X]``. The fuzzy kink estimand is the ratio of the slope changes in the
  outcome and treatment regressions. `scalepar` rescales the outcome estimand, for
  example to divide by the known kink in a policy formula.

The effect is local. It refers to units at the cutoff and does not extrapolate to units
far from it without further assumptions.

## Estimation and inference

[`rd_estimate`](@ref) fits weighted local polynomials of order `p` (default 1) on each
side of the cutoff, using observations within the bandwidth `h` and kernel weights
(triangular by default). The conventional estimator has a smoothing bias of the same
order as its standard error at the MSE-optimal bandwidth, so conventional confidence
intervals under-cover. Calonico, Cattaneo & Titiunik (2014) therefore:

1. estimate the leading bias with a local polynomial of order `q = p + 1` and a pilot
   bandwidth `b`, and subtract it (*bias-corrected* estimate);
2. compute a *robust* standard error that includes the variability of the estimated
   bias.

The headline result of an [`RDEstimate`](@ref) follows the reporting convention of
`rdrobust`: `coef` is the *conventional* point estimate, while `stderror`, `pvalues`
and `confint` are robust bias-corrected. The robust interval is centred on the
bias-corrected estimate (the field `r.tau_bias_corrected`), so it is generally not
symmetric around `coef(r)`; the two estimates differ by the estimated bias. `tidy`,
`coeftable` and `regtable` show the same layout: the conventional estimate next to the
robust standard error, p-value and interval, which is the recommended way to report
RD results (Cattaneo, Idrobo & Titiunik 2020). [`rd_inference_table`](@ref) lists the
conventional, bias-corrected and robust rows, as `rdrobust` does. Do not report the
conventional interval at MSE-optimal bandwidths.

Variance estimators (`vce`):

- `:nn` (default): heteroskedasticity-robust nearest-neighbour estimator with
  `nnmatch = 3` neighbours, with the tie handling of `rdrobust`;
- `:hc0`, `:hc1`, `:hc2`, `:hc3`: plug-in residuals from the local fits;
- `cluster = :id`: cluster-robust CR1 (the default when a cluster variable is given),
  or the small-sample corrections CR2 (`vce = :cr2`, Bell–McCaffrey-type) and CR3
  (`vce = :cr3`, jackknife-type) of `rdrobust` 4.0, which are preferable with few
  clusters.

*Covariates* (`covariates = [...]`) enter linearly with coefficients common to both
sides (Calonico, Cattaneo, Farrell & Titiunik 2019). If the covariates are
predetermined and balanced at the cutoff, the estimand does not change and precision
usually improves. Covariate adjustment does not fix a design in which covariates jump at
the cutoff. *Weights* (`weights = :w`) multiply the kernel weights.

*Mass points.* With a discrete running variable (`masspoints = :adjust`, the default),
the bandwidth selector counts distinct values of the running variable. If at least 20%
of observations on a side are repeated values, bandwidths are constrained to contain at
least 10 distinct values on each side (`bwcheck`). With few mass points, inference that
relies on the bandwidth shrinking is fragile. Use the honest methods of Kolesár & Rothe
(2018) described in [Discrete running variables](@ref rd-discrete), or local
randomization methods.

## Bandwidth selection

[`rd_bandwidth`](@ref) implements the ten selectors of `rdbwselect`:

- MSE-optimal: `:mserd` (one common `h`, the default), `:msetwo` (different `h` on each
  side), `:msesum` (for the sum of the regression functions), `:msecomb1`
  (``\min(\text{mserd}, \text{msesum})``), `:msecomb2` (the median of mserd, msesum and
  msetwo, side by side);
- CER-optimal: `:cerrd`, `:certwo`, `:cersum`, `:cercomb1`, `:cercomb2`. These multiply
  the MSE-optimal `h` by ``n^{-p/((3+p)(3+2p))}`` (the number of clusters when
  clustering), which minimises the coverage error of the robust interval (Calonico,
  Cattaneo & Farrell 2020).

The plug-in constants are estimated in three steps with a regularisation term
(`scaleregul = 1`), as in `rdrobust`. With `stdvars = true` (in [`rd_estimate`](@ref)
and [`rd_bandwidth`](@ref)) the outcome and the running variable are divided by their
standard deviations before the bandwidths are selected, and the bandwidths are
rescaled to the units of the running variable, as `rdrobust(..., stdvars = TRUE)`
does; estimation always uses the original data. MSE-optimal bandwidths are the natural choice for
point estimation. CER-optimal bandwidths give intervals with smaller coverage error, and
the robust interval is valid with either choice.

## RD plots

[`rd_plot_data`](@ref) returns binned sample means and a global polynomial fit of order
`p = 4` on each side, as plain `DataFrame`s for any plotting layer. With Makie loaded
(for example `using CairoMakie`), `plot_rd(pd)` draws the RD plot from an
[`RDPlotData`](@ref). The number of bins is chosen by the IMSE-optimal (`:es`, `:qs`) or
mimicking-variance (`:esmv`, the default, `:qsmv`) methods of Calonico, Cattaneo &
Titiunik (2015), with evenly spaced (`es`) or quantile-spaced (`qs`) bins. RD plots are
descriptive. The global polynomial fit should not be used to estimate the effect
(Gelman & Imbens 2019).

## Falsification and sensitivity workflow

These checks assess implications of the design. None of them can establish that the
continuity assumption holds, and a non-rejection is never evidence that it does. The
[`DiagnosticTest`](@ref) results and tables word their output accordingly.

1. **Manipulation of the running variable.** [`rd_density_test`](@ref) tests continuity
   of the density of ``X`` at the cutoff with the local polynomial density estimator of
   Cattaneo, Jansson & Ma (2020): robust bias-corrected, jackknife standard errors, and
   bandwidths from [`rd_density_bandwidth`](@ref). `details.binomial` adds exact
   binomial tests of the share of observations on each side in small windows. The test
   has low power against small or two-sided sorting. [`rd_mccrary_test`](@ref)
   implements the original test of McCrary (2008) (histogram with bin width
   ``2\hat\sigma_X n^{-1/2}``, local linear smoothing of the bin heights on each side
   with McCrary's rule-of-thumb bandwidth, and a test of equal log densities), as
   `DCdensity` of the R package `rdd`. It depends on the binning and has no bias
   correction; use it to compare with studies that report it, and prefer
   [`rd_density_test`](@ref) otherwise.
2. **Covariate balance.** [`rd_covariate_balance`](@ref) runs [`rd_estimate`](@ref) on
   each predetermined covariate, each with its own bandwidth. Holm-adjusted p-values
   account for testing several covariates.
3. **Placebo cutoffs.** [`rd_placebo_cutoffs`](@ref) estimates effects at artificial
   cutoffs, using only control units below the true cutoff and only treated units above
   it.
4. **Donut hole.** [`rd_donut`](@ref) drops observations closest to the cutoff, where
   sorting or heaping is most likely.
5. **Bandwidth sensitivity.** [`rd_bandwidth_sensitivity`](@ref) re-estimates with `h`
   and `b` scaled around the data-driven choice. Larger bandwidths trade variance for
   bias.

### Weak first stage in fuzzy designs

When the jump in take-up is small, the fuzzy RD estimator behaves like a weak-IV
estimator and its Wald-type interval can under-cover. [`rd_weak_iv_confidence_set`](@ref)
inverts robust bias-corrected tests of ``H_0 : \tau = \tau_0``, based on the jump in
``Y - \tau_0 D`` (Feir, Lemieux & Marmer 2016), using the bandwidths of the fitted
model. The set can be a bounded interval, two unbounded rays, or the whole real line.
The whole real line means the data are uninformative about ``\tau``. For a bias-aware
version, which bounds the smoothing bias instead of correcting it, see
[`rd_honest_ar_confidence_set`](@ref) (Noack & Rothe 2024) in
[Honest inference](@ref rd-honest).

### Local randomization

Under the alternative assumption that the score is as-if randomly assigned within a small
window around the cutoff, [`rd_randomization_test`](@ref) runs a Fisher randomization
test of the sharp null of no effect, reusing DrSnow's complete-randomization machinery.
[`rd_window_selection`](@ref) chooses the window by testing covariate balance in nested
windows (Cattaneo, Frandsen & Titiunik 2015). Conclusions depend on the window. Report
results for several windows.

## [Honest (bias-aware) inference](@id rd-honest)

Robust bias-corrected intervals ([`rd_estimate`](@ref)) are valid asymptotically for a
given regression function, as the bandwidth shrinks. Honest intervals (Armstrong &
Kolesár 2018, 2020) instead bound the smoothing bias over a class of functions and
widen the interval accordingly, so that coverage holds uniformly over the class and at
any bandwidth. [`rd_honest`](@ref) implements the approach of the R package `RDHonest`.

**Smoothness class.** The conditional mean ``f`` of the outcome is assumed to have a
second derivative bounded by `M` on each side of the cutoff (Hölder class,
`sclass = :holder`, the default), or to deviate from its first-order Taylor
approximation at the cutoff by at most ``M x^2 / 2`` (Taylor class, `:taylor`). The
local linear estimator is linear in the outcomes, ``\hat\tau = \sum_i k_i Y_i``, so its
worst-case bias over the Hölder class has the closed form

```math
\bar b = \frac{M}{2}\Big|\sum_{x_i < c} k_i (x_i - c)^2
- \sum_{x_i \ge c} k_i (x_i - c)^2\Big|.
```

**Interval.** With standard error ``\text{se}`` (nearest-neighbour, `vce = :nn`, or
Eicker–Huber–White, `vce = :ehw`; cluster-robust with `cluster`), the honest interval
is ``\hat\tau \pm \text{cv}_{1-\alpha}(\bar b / \text{se})\,\text{se}``, where
``\text{cv}_{1-\alpha}(B)`` is the ``1-\alpha`` quantile of ``|N(B, 1)|`` (the folded
non-central normal). It is wider than ``\hat\tau \pm z_{1-\alpha/2}\,\text{se}``, and it
is **not** a symmetric t-based interval: `confint(r)` returns it (at any `level`), and
`pvalues(r)` returns the matching honest p-value. One-sided intervals use
``\bar b / \text{se} + z_{1-\alpha}`` standard errors. The point estimate is the local
linear estimate itself (no bias correction).

**Bandwidth.** By default the bandwidth minimises the worst-case MSE
(`opt_criterion = :mse`); `:flci` minimises the length of the honest interval and
`:oci` a quantile of the excess length of one-sided intervals. The optimisation uses
preliminary homoskedastic variances on each side of the cutoff, from a local linear fit
with the Imbens & Kalyanaraman (2012) bandwidth, as in `RDHonest`.

**Choosing `M`.** `M` cannot be estimated consistently from the data without further
restrictions: honesty holds for the class defined by the chosen `M`, so it should come
from subject knowledge (for example, how much the slope of ``f`` can change over the
range of the running variable). When `M` is not given, the rule of thumb of Armstrong &
Kolesár (2020) is used: the largest absolute second derivative of global quartic fits
on each side of the cutoff (`r.M_rule_of_thumb` is then `true`). Report results for a
range of `M`. [`rd_smoothness_bound`](@ref) gives a data-driven *lower* bound on `M`
(Kolesár & Rothe 2018, online appendix); values of `M` below it are inconsistent with
the data.

**Fuzzy designs.** With `treatment`, `M = (M_Y, M_D)` bounds the second derivatives of
the outcome and treatment regressions. [`rd_honest`](@ref) reports the ratio estimate
with the delta-method standard error and the linearised worst-case bias
``(M_Y + |\hat\theta| M_D)\,\bar b_1 / |\hat\tau_D|``, where ``\bar b_1`` is the bias
per unit of `M`, as `RDHonest` does; the bandwidth uses a preliminary estimate `T0` of
the effect. This requires a strong first stage.
[`rd_honest_ar_confidence_set`](@ref) remains valid with a weak first stage (Noack &
Rothe 2024): it inverts bias-aware tests of ``H_0 : \theta = \theta_0`` based on the
jump in ``Y - \theta_0 D``, whose worst-case bias is
``(M_Y + |\theta_0| M_D)\,\bar b_1``. The set can be unbounded.

**Covariates** enter linearly. When `M` or `h` is not supplied, a bandwidth is first
selected without covariates, the outcome is adjusted with the covariate coefficients
from that local fit, and `M` and the bandwidth are then computed from the adjusted
outcome; the final estimate is the local linear regression with the covariates. This
reproduces `RDHonest`. **Inference at a point** (`point_inference = true`) targets
``E[Y \mid X = x_0]`` with `x0 = cutoff`; away from a boundary the worst-case bias is an
integral, which DrSnow evaluates exactly.

A maximal leverage (`r.leverage`, the largest ``k_i^2 / \sum_j k_j^2``) above 0.1
triggers a warning: the normal approximation may then be poor. `r.eff_obs` is the
number of effective observations relative to a uniform kernel.

## [Discrete running variables](@id rd-discrete)

When the running variable takes few distinct values (age in years, test scores with
few points, income brackets), the local polynomial approximation does not improve as
the sample grows, because there are no observations arbitrarily close to the cutoff.
Robust bias-corrected inference, which relies on the bandwidth shrinking, is then not
justified, and clustering standard errors by the value of the running variable does not
fix the problem (Kolesár & Rothe 2018). Two honest alternatives are available.

- **Bounded second derivative** ([`rd_honest`](@ref)). The worst-case bias bound and
  the honest interval are valid at any bandwidth, so they apply unchanged with discrete
  support; the bandwidth search respects the support (for the uniform kernel it is a
  search over the distinct values of ``|X - c|``). This uses all support points and
  requires `M`.
- **Bounded misspecification error** ([`rd_honest_bme`](@ref)). A polynomial of order
  `order` is fitted with a uniform kernel within `h`. The BME class assumes that the
  specification error of the polynomial at the cutoff is no larger than the largest
  error at the support points in the window on each side. The interval takes the union,
  over support points on each side and signs, of intervals shifted by the estimated
  specification errors, with standard errors that account for their estimation. It
  needs no `M`, but it needs support points on each side beyond the polynomial order,
  and it can be conservative.

Use these methods when the number of support points near the cutoff is small (say,
fewer than 10–20 on each side within any reasonable bandwidth). With many distinct
values, [`rd_estimate`](@ref) (with `masspoints = :adjust`) and [`rd_honest`](@ref)
both apply. [`rd_smoothness_bound`](@ref) estimates lower bounds on `M` from second
differences of cell means at adjacent support points, which helps to judge whether the
`M` passed to [`rd_honest`](@ref) is plausible.

## [Flexible covariate adjustment (machine learning)](@id rd-flex)

The linear adjustment of `rd_estimate(...; covariates)` gains precision only through
the part of the outcome that is linear in the covariates. [`rd_flex`](@ref)
implements the flexible adjustment of Noack, Olma & Rothe (2026), as in DoubleML's
`RDFlex`:

1. A learner estimates `μ±(X) = E[Y | X, running = cutoff±]` from the covariates, the
   side indicator (and optionally the centered running variable), with kernel
   weights around the cutoff (initial bandwidth `h_fs`: the largest MSE-optimal `h`
   and `b` without covariates). With **cross-fitting**, each observation's
   adjustment `η(X) = (μ₊(X) + μ₋(X)) / 2` comes from a fit on the other folds.
2. Local linear RD with robust bias-corrected inference ([`rd_estimate`](@ref)) is
   applied to the adjusted outcome `Y − η(X)`. With the default `n_iterations = 2`,
   the bandwidth selected on the adjusted data replaces `h_fs` in the kernel weights,
   `η` is re-estimated, and the final estimate uses that bandwidth.
3. In a fuzzy design (`treatment = :d`), the treatment is adjusted in the same way
   with a probabilistic learner, and the estimate is the ratio of the two jumps.

```julia
r = rd_flex(df, :y, :x; covariates = [:z1, :z2, :z3],
            outcome_learner = ForestLearner(), rng = StableRNG(1))
confint(r)                  # robust bias-corrected interval
rd_inference_table(r)
```

**Assumptions.** The usual continuity assumptions of the RD design, and
predetermined covariates: their distribution must not change discontinuously at the
cutoff (check with [`rd_covariate_balance`](@ref)). Then `η(X)` has no jump, the
adjustment leaves the estimand unchanged for *any* function `η`, and the estimate
is consistent even if the learner is poor. Cross-fitting makes the estimation of `η`
asymptotically negligible for inference, so the robust standard errors of
`rd_estimate` on the adjusted data remain valid (Noack, Olma & Rothe 2026). The
adjustment does not rescue a design in which covariates jump at the cutoff.

**When does ML adjustment help?** The variance of the RD estimator is driven by the
conditional variance of the outcome near the cutoff; the adjustment removes the
part explained by the covariates. The gain is large when the covariates explain
much of the outcome (lagged outcomes, baseline test scores, prior earnings) and
especially when they act non-linearly, which linear adjustment cannot capture. When
covariate effects are linear, linear adjustment is already (near-)optimal, and a
flexible learner loses a little precision to estimation noise. When the covariates
explain little, neither adjustment matters. Adjustment never changes the estimand,
so it is a choice about precision, to be made before looking at the estimates.

Monte Carlo (`test/validation/rd/flex_montecarlo.jl`, 1,000 replications; true
effect 0.5 in the sharp designs and 1.0 in the fuzzy design; robust 95% intervals):

| Design | n | Method | Bias | SD | Mean SE | Coverage | CI length |
|:--|--:|:--|--:|--:|--:|--:|--:|
| sharp, non-linear | 1000 | rd_estimate, no covariates | 0.025 | 0.599 | 0.572 | 0.951 | 2.24 |
| sharp, non-linear | 1000 | rd_estimate, linear covariates | 0.019 | 0.558 | 0.530 | 0.951 | 2.08 |
| sharp, non-linear | 1000 | rd_flex, OLS | 0.018 | 0.569 | 0.549 | 0.956 | 2.15 |
| sharp, non-linear | 1000 | rd_flex, random forest | 0.021 | 0.395 | 0.359 | 0.935 | 1.41 |
| sharp, non-linear | 4000 | rd_estimate, no covariates | 0.013 | 0.297 | 0.284 | 0.950 | 1.12 |
| sharp, non-linear | 4000 | rd_estimate, linear covariates | 0.008 | 0.276 | 0.266 | 0.951 | 1.04 |
| sharp, non-linear | 4000 | rd_flex, OLS | 0.008 | 0.277 | 0.268 | 0.952 | 1.05 |
| sharp, non-linear | 4000 | rd_flex, random forest | 0.007 | 0.132 | 0.123 | 0.948 | 0.48 |
| sharp, linear | 1000 | rd_estimate, no covariates | -0.009 | 0.469 | 0.442 | 0.944 | 1.73 |
| sharp, linear | 1000 | rd_estimate, linear covariates | 0.006 | 0.157 | 0.149 | 0.937 | 0.58 |
| sharp, linear | 1000 | rd_flex, OLS | 0.005 | 0.158 | 0.152 | 0.941 | 0.59 |
| sharp, linear | 1000 | rd_flex, random forest | -0.001 | 0.240 | 0.231 | 0.943 | 0.90 |
| fuzzy, non-linear | 3000 | rd_estimate, no covariates | -0.012 | 0.963 | 0.889 | 0.959 | 3.48 |
| fuzzy, non-linear | 3000 | rd_estimate, linear covariates | -0.004 | 0.906 | 0.827 | 0.951 | 3.24 |
| fuzzy, non-linear | 3000 | rd_flex, OLS | 0.002 | 0.894 | 0.840 | 0.955 | 3.29 |
| fuzzy, non-linear | 3000 | rd_flex, random forest | 0.001 | 0.472 | 0.441 | 0.948 | 1.73 |

With non-linear covariate effects, the forest adjustment reduces the standard
deviation by about 30% at n = 1000 and by half at n = 4000 relative to linear
adjustment (which gains little), with coverage close to nominal throughout. With
linear covariate effects, linear adjustment (or `rd_flex` with `OLSLearner()`) is
best; the forest adjustment is less precise, though still far better than no
adjustment.

`rd_flex` uses DrSnow's learners ([`OLSLearner`](@ref), [`LassoLearner`](@ref),
[`ForestLearner`](@ref), [`MLJLearner`](@ref), ...; the learner must accept
observation weights). Folds are drawn on the observations sorted by running variable
and outcome, so results do not depend on the row order of the data; with `cluster`
the folds are unions of clusters and the RD variance is cluster-robust. Repeated
cross-fitting (`n_rep`) reports medians with variances `median(se_r² + (θ_r − θ̃)²)`
(DoubleML scales the dispersion term by the effective sample size instead).

## [Validation](@id rd-validation)

`test/validation/rd/generate_reference.R` saves the Senate data from `rdrobust` and
simulated designs (with covariates, clusters, weights, fuzzy and one-sided take-up,
kinks and a discrete running variable) as CSV files. It then records reference output
from `rdrobust` 4.0.0, `rdbwselect`, `rdplot`, `rddensity` 3.0 and `rdbwdensity`.
`test/validation/rd/generate_reference_honest.R` saves the Lee (2008) data and subsets
of the Head Start, retirement-consumption and Oreopoulos (2006) data shipped with
`RDHonest`, and records reference output from `RDHonest` (1.0.2.9000, GitHub
`kolesarm/RDHonest`), `rdd` 0.57 (`DCdensity`) and `rdrobust` 4.0.0 with
`stdvars = TRUE`. The RD test group compares DrSnow against these values:

| Component | Cases | Largest relative difference |
|---|---|---|
| `rd_estimate` vs `rdrobust`: estimates, SEs, CIs, p-values, `h`, `b`, bias, effective N, covariate coefficients (NN, HC0–HC3, CR1–CR3) | 75 | < 1e-9 |
| `rd_bandwidth` vs `rdbwselect(all = TRUE)` | 9 × 10 selectors | < 1e-9 |
| `rd_density_test` vs `rddensity` (incl. binomial tests) | 17 | < 1e-9 |
| `rd_density_bandwidth` vs `rdbwdensity` | 4 | < 1e-9 |
| `rd_plot_data` vs `rdplot` | 15 | < 1e-9 (bin counts exact) |
| `stdvars = true` vs `rdrobust` / `rdbwselect(all = TRUE)` | 5 + 2 × 10 selectors | < 1e-9 |
| `rd_honest` vs `RDHonest` (sharp, fuzzy, point; kernels, criteria, NN/EHW, clusters, weights, covariates, discrete support) at a fixed or grid-searched bandwidth, or at the bandwidth selected by `RDHonest` | 28 | < 1e-9 |
| `rd_honest` with the bandwidth optimised by DrSnow vs `RDHonest` | 22 | < 2e-6 (optimiser tolerance) |
| `rd_honest_bme` vs `RDHonestBME` | 4 | < 1e-9 |
| `rd_smoothness_bound` vs `RDSmoothnessBound` (single curvature estimate) | 3 | < 1e-8 |
| `rd_mccrary_test` vs `rdd::DCdensity` | 5 | < 1e-9 |
| `rd_flex` vs DoubleML `RDFlex` (Python 0.11.4, rdrobust 2.1.0; OLS / logit learners, identical folds): adjusted outcomes and treatments, `h_fs`, `h`, `b`, estimates, SEs, effective N; sharp (three first-stage specifications, one or two iterations, uniform kernel) and fuzzy | 6 | < 1e-9 (fuzzy: < 1e-7, sklearn solver tolerance) |

Known deviations from the R implementations:

- With covariates and `deriv ≥ 2`, DrSnow includes the factor ``\text{deriv}!`` in the
  point estimate, consistent with the variance. `rdrobust` omits it, which matters only
  for derivatives of order 2 or higher.
- When some left-side bins are empty, `rdplot` misreports the left and right edges of
  left-side bins. DrSnow reports each bin's own edges.
- The kernel moment matrices of `rddensity` are computed exactly (rational arithmetic)
  instead of by numerical integration.
- `rdrobust`'s `subset`/formula interfaces are not available (use DataFrame
  subsetting).
- `rd_honest` selects bandwidths with a transcription of R's `optimize` (Brent's
  method). Objective values that differ in the last bits can stop it at a slightly
  different point, so optimised bandwidths agree with `RDHonest` to about 1e-8
  (relative) and the resulting intervals to about 1e-6. At the same bandwidth all
  quantities agree to about 1e-12.
- For inference at an interior point, `RDHonest` integrates the worst-case bias
  numerically (relative tolerance about 1e-4); DrSnow integrates the piecewise-linear
  integrand exactly, so the bias bounds differ by up to about 1e-6.
- `rd_honest` breaks ties in the running variable by the other columns before
  computing, so that results do not depend on the row order of the data (`RDHonest`
  keeps the input order of ties; this changes only the floating-point summation
  order).
- `rd_honest_bme` reports the p-value obtained by inverting the BME interval
  (`RDHonestBME` prints a p-value that treats the bias bound as a number of standard
  errors).
- Not available from `RDHonest`: the finite-sample optimal (non-local-polynomial)
  estimator `kern = "optimal"` with its efficiency bounds (`RDTEfficiencyBound`), and
  user-supplied conditional variances (`se.method = "supplied.var"`).
- `rd_smoothness_bound` with several curvature estimates (`multiple = true`) uses
  simulated critical values drawn from `rng`, so it matches `RDSmoothnessBound` only up
  to simulation noise.

Monte Carlo checks (`test/rd/test_montecarlo.jl`, full counts with
`DRSNOW_SLOW_TESTS=true`) cover the following:

- Coverage of the robust interval in a smooth design (≈ 0.94 for sharp and kink
  designs).
- Coverage in CCT (2014) model 1, where `rdrobust` itself attains 0.90 at ``n = 500``
  (0.902 in 1,500 R replications; 0.909 and 0.924 in two DrSnow runs of 3,000 and
  2,000 replications).
- Fuzzy and weak-first-stage coverage of the Anderson–Rubin set.
- The size of the density and randomization tests, and of the McCrary test.
- Coverage of the honest interval of [`rd_honest`](@ref) at the least favourable
  function of the Hölder class (``f(x) = \pm M x^2 / 2`` with opposite signs on the two
  sides; `M` known; MSE- and FLCI-optimal and fixed bandwidths; continuous and discrete
  running variable; `test/rd/test_honest.jl`): 0.92–0.95 at 300 replications of
  ``n = 500`` (nominal 0.95), while the naive interval ``\hat\tau \pm 1.96\,\text{se}``
  covers 0.90–0.92. Coverage of the bias-aware Anderson–Rubin set is 0.96 with strong and
  weak first stages (100 replications).

## References

- Armstrong, T. B., & Kolesár, M. (2018). Optimal inference in a class of regression
  models. *Econometrica*, 86(2), 655–683.
- Armstrong, T. B., & Kolesár, M. (2020). Simple and honest confidence intervals in
  nonparametric regression. *Quantitative Economics*, 11(1), 1–39.
- Calonico, S., Cattaneo, M. D., & Farrell, M. H. (2018). On the effect of bias
  estimation on coverage accuracy in nonparametric inference. *Journal of the American
  Statistical Association*, 113(522), 767–779.
- Calonico, S., Cattaneo, M. D., & Farrell, M. H. (2020). Optimal bandwidth choice for
  robust bias-corrected inference in regression discontinuity designs. *The Econometrics
  Journal*, 23(2), 192–210.
- Calonico, S., Cattaneo, M. D., Farrell, M. H., & Titiunik, R. (2017). rdrobust:
  Software for regression-discontinuity designs. *The Stata Journal*, 17(2), 372–404.
- Calonico, S., Cattaneo, M. D., Farrell, M. H., & Titiunik, R. (2019). Regression
  discontinuity designs using covariates. *Review of Economics and Statistics*, 101(3),
  442–451.
- Calonico, S., Cattaneo, M. D., & Titiunik, R. (2014). Robust nonparametric confidence
  intervals for regression-discontinuity designs. *Econometrica*, 82(6), 2295–2326.
- Calonico, S., Cattaneo, M. D., & Titiunik, R. (2015). Optimal data-driven regression
  discontinuity plots. *Journal of the American Statistical Association*, 110(512),
  1753–1769.
- Cattaneo, M. D., Frandsen, B. R., & Titiunik, R. (2015). Randomization inference in
  the regression discontinuity design: An application to party advantages in the U.S.
  Senate. *Journal of Causal Inference*, 3(1), 1–24.
- Cattaneo, M. D., Idrobo, N., & Titiunik, R. (2020). *A Practical Introduction to
  Regression Discontinuity Designs: Foundations*. Cambridge University Press.
- Cattaneo, M. D., Jansson, M., & Ma, X. (2020). Simple local polynomial density
  estimators. *Journal of the American Statistical Association*, 115(531), 1449–1455.
- Cattaneo, M. D., Titiunik, R., & Vazquez-Bare, G. (2016). Inference in regression
  discontinuity designs under local randomization. *The Stata Journal*, 16(2), 331–367.
- Feir, D., Lemieux, T., & Marmer, V. (2016). Weak identification in fuzzy regression
  discontinuity designs. *Journal of Business & Economic Statistics*, 34(2), 185–196.
- Gelman, A., & Imbens, G. (2019). Why high-order polynomials should not be used in
  regression discontinuity designs. *Journal of Business & Economic Statistics*, 37(3),
  447–456.
- Imbens, G., & Kalyanaraman, K. (2012). Optimal bandwidth choice for the regression
  discontinuity estimator. *Review of Economic Studies*, 79(3), 933–959.
- Kolesár, M., & Rothe, C. (2018). Inference in regression discontinuity designs with a
  discrete running variable. *American Economic Review*, 108(8), 2277–2304.
- McCrary, J. (2008). Manipulation of the running variable in the regression
  discontinuity design: A density test. *Journal of Econometrics*, 142(2), 698–714.
- Noack, C., Olma, T., & Rothe, C. (2026). Flexible covariate adjustments in regression
  discontinuity designs. *Journal of Econometrics*, 257, 106298.
- Noack, C., & Rothe, C. (2024). Bias-aware inference in fuzzy regression discontinuity
  designs. *Econometrica*, 92(3), 687–711.

The functions and types described on this page are documented in the [API reference](reference/rd.md).
