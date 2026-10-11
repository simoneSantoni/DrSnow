# Longitudinal treatment regimes

```@meta
CurrentModule = DrSnow
```

Many policies are not applied once but period after period, and whether a unit is
treated in a period depends on how it is doing: a firm receives support when its
sales fall, a patient is treated when a biomarker crosses a threshold, a household
re-enrols in a programme when its income drops. The covariate that drives the next
treatment decision is itself a consequence of the previous treatment. This is
**treatment–confounder feedback**, and it defeats the usual adjustment strategies.
Regressing the outcome on the whole treatment history while controlling for the
time-varying covariates blocks the part of the early treatments' effect that runs
through those covariates and opens collider paths; leaving the covariates out leaves
the later treatments confounded. The `longitudinal` area implements the **g-methods**
of Robins (1986), which identify the effect of a *treatment regime*, a strategy for
the whole sequence of decisions, by standardizing sequentially over the covariate
process:

- **regimes**: [`StaticRegime`](@ref) (a fixed sequence of actions, e.g.
  [`always_treat`](@ref), [`never_treat`](@ref)) and [`DynamicRegime`](@ref) (a rule
  that chooses each period's action from the history);
- **estimators** of the mean counterfactual outcome ``E[Y^d]`` under each regime with
  [`longitudinal_regime`](@ref): iterated-regression g-computation (`:gcomp`),
  inverse probability of treatment weighting (`:ipw`), longitudinal targeted minimum
  loss-based estimation (`:tmle`) and the sequentially doubly robust estimator
  (`:sdr`), all with optional cross-fitting of machine-learned nuisance functions;
- **summaries**: differences, risk ratios and odds ratios between regimes with
  [`regime_contrast`](@ref), and the coefficients of a working marginal structural
  model projected from several regimes with [`regime_msm`](@ref);
- **diagnostics** of positivity and weights with [`regime_weights`](@ref) and
  [`plot_regime_weights`](@ref).

The methods here are observational: they replace sequential randomization by the
assumption that every treatment decision is unconfounded given the observed past.
For micro-randomized trials, where the randomization probabilities are known, the
adaptive area's [`wcls`](@ref) and [`emee`](@ref) estimate proximal excursion effects
(which are not effects of sustained regimes); for staggered adoption under parallel
trends rather than sequential unconfoundedness, see the
[difference-in-differences](did.md) estimators; the [sequential inference](sequential.md)
area is about *when* data may be analysed (monitoring), not about sequential
treatments.

## Setting, estimand and identification

Each unit ``i`` is observed over ``K`` periods. In period ``t`` the time-varying
covariates ``L_t`` are measured, then the binary treatment ``A_t`` is decided; the
outcome ``Y`` is measured after period ``K``; baseline covariates ``W`` are measured
before period 1. The history at the time of the ``t``-th decision is
``H_t = (W, L_1, …, L_t, A_1, …, A_{t-1})``. A regime ``d = (d_1, …, d_K)`` prescribes
an action in every period: a constant ``a_t`` for a static regime, a function
``d_t(H_t)`` of the history for a dynamic one. The estimand for a regime is the mean
of the counterfactual outcome ``Y^d`` that would be observed if every unit's
treatment had been set by ``d`` at every period,

```math
ψ_d = E[Y^d],
```

and the quantities of interest are contrasts such as ``E[Y^{\bar 1}] - E[Y^{\bar 0}]``,
the effect of treating in every period versus never, or the coefficients of a
marginal structural model ``E[Y^{\bar a}] = β_0 + β_1 \sum_t a_t`` over several static
regimes.

Identification requires three conditions (Hernán and Robins 2020, ch. 19–21):

**Sequential exchangeability (no unmeasured time-varying confounding).** For every
period, ``Y^d`` is independent of ``A_t`` given ``H_t`` among units that have followed
``d`` so far. Treatment in each period may depend on everything observed before it,
including past treatments and covariates affected by them, but on nothing else that
also affects the outcome. This is the longitudinal version of selection on
observables; it is untestable, and the data cannot distinguish it from its failure.

**Positivity.** Every history that can arise under the regime must have a positive
probability of receiving the regime's action: ``P(A_t = d_t(h_t) \mid H_t = h_t) > 0``.
Unlike exchangeability this is partly checkable, and the longitudinal setting makes
it demanding: the cumulative probability of following a regime through ``t`` is a
product of ``t`` conditional probabilities and shrinks with ``t``. The
[weight diagnostics](@ref longitudinal-diagnostics) below describe the support of
each regime.

**Consistency.** ``Y = Y^{\bar A}`` and ``L_t = L_t^{\bar A_{t-1}}``: the observed outcome and
covariates are the counterfactual ones under the treatment actually received, which
presupposes well-defined treatments and no interference between units.

Under these conditions ``ψ_d`` equals Robins' **g-formula**, the mean of the outcome
regression under the regime averaged over the covariate process that the regime
would generate, and equivalently (Bang and Robins 2005) the **iterated conditional
expectation**

```math
Q_{K+1} = Y, \qquad
Q_t(h_t) = E\bigl[Q_{t+1}(H_{t+1}) \mid H_t = h_t, A_t = d_t(h_t)\bigr], \qquad
ψ_d = E[Q_1(H_1)],
```

and equivalently the **inverse-probability-weighted mean**

```math
ψ_d = E\Bigl[ Y \prod_{t=1}^{K} \frac{1\{A_t = d_t(H_t)\}}{g_t(d_t(H_t) \mid H_t)} \Bigr],
\qquad g_t(a \mid h_t) = P(A_t = a \mid H_t = h_t).
```

The three representations suggest the three classes of estimators; the doubly
robust ones combine them.

## Estimators

All estimators work with the outcome scaled to ``[0, 1]`` by its observed range (or
`outcome_bounds`), a treatment mechanism ``\hat g_t`` estimated by `treatment_learner`
from ``H_t`` in every period, cumulative probabilities
``\hat g_{1:t} = \prod_{s ≤ t} \hat g_s(d_s \mid H_s)`` bounded below by `g_bound`
(0.01 by default, ltmle's convention), and cumulative weights
``\hat W_t = 1\{\bar A_t = \bar d_t\} / \hat g_{1:t}`` for the **followers** through
``t``, the units whose observed treatments equal the regime's actions in every period
up to ``t``.

| `estimator` | construction | consistent when | variance |
|:--|:--|:--|:--|
| `:gcomp` | backward recursion: regress ``\hat Q_{t+1}`` on ``(H_t, A_t)``, predict with the treatment history set to the regime; ``\hat ψ = \bar{\hat Q}_1`` | every outcome regression is correct | unit bootstrap only |
| `:ipw` | Hájek mean ``\sum_i \hat W_{K,i} Y_i / \sum_i \hat W_{K,i}`` (or Horvitz–Thompson with `normalize = false`) | the treatment mechanism is correct | influence function, weights taken as known (conservative under a correct parametric mechanism) |
| `:tmle` | the `:gcomp` recursion with a logistic fluctuation of each ``\hat Q_t`` fitted among the followers with weights ``\hat W_t`` (van der Laan and Gruber 2012) | at every ``t``, the outcome regression or the treatment mechanism is correct | efficient influence function |
| `:sdr` | sequential regressions of the pseudo-outcomes ``\sum_{s ≥ t} (\hat W_s / \hat W_{t-1})(\hat Q_{s+1} - \hat Q_s) + \hat Q_t``; ``\hat ψ`` is the mean of the resulting influence function (Díaz et al. 2023) | for every ``t``, all later outcome regressions or all earlier treatment mechanisms are correct (sequential double robustness) | efficient influence function |

The sequentially doubly robust condition is strictly weaker than the doubly robust
one (Luedtke et al. 2017; Rotnitzky, Robins and Babino 2017), which makes `:sdr` the
default. Its price is that the regression targets are not confined to ``[0, 1]``, so
`outcome_learner` must accept unbounded targets ([`OLSLearner`](@ref),
[`LassoLearner`](@ref), [`RidgeLearner`](@ref), [`ForestLearner`](@ref), MLJ
regressors; not [`FractionalLogitLearner`](@ref)), and that the estimate is not a
substitution estimator, so it can leave the outcome's range when the weights are
extreme. `:tmle` keeps the estimate within the range and is the right choice when
positivity is weak; `:gcomp` and `:ipw` are the classical single-robust estimators,
useful as comparisons and as a check that the nuisance models agree.

**Cross-fitting.** With `crossfit = true` (default) the units are split into
`n_folds` folds; for each fold the treatment mechanism and the whole backward
recursion are fitted on the other folds and evaluated on the held-out units, so that
every unit's nuisance values are out of fold (Chernozhukov et al. 2018; the scheme of
lmtp). This is what allows flexible learners without Donsker conditions. With
parametric learners and `crossfit = false` the classical estimators are recovered:
with [`FractionalLogitLearner`](@ref) for the outcome regressions (quasibinomial
regression on the scaled outcome) and [`LogisticLearner`](@ref) for the treatment
mechanism, `:tmle`, `:ipw` and `:gcomp` reproduce the R package ltmle (Lendle et al.
2017). Repeated splits (`n_rep > 1`) are aggregated by the median rule of
[`DMLEstimate`](@ref).

**Inference.** For `:ipw`, `:tmle` and `:sdr` the covariance of the regime means is
the sample covariance of the estimated efficient influence functions divided by
``n`` (clustered, with a ``t_{G-1}`` reference, when `cluster` is given). Because the
influence functions of all regimes are stored, [`regime_contrast`](@ref) and
[`regime_msm`](@ref) obtain the covariance of contrasts and projected marginal
structural model coefficients by the delta method with the correct dependence
between regimes. `bootstrap = B` replaces the influence-function covariance by the
unit (or cluster) bootstrap, which refits everything in each replicate, and is the
only inference offered for `:gcomp`, whose plug-in influence function is not the
influence function of the substitution estimator.

```julia
using DrSnow, DataFrames, StableRNGs
# long data: one row per unit and period; y measured in the last period
r = longitudinal_regime(df, :y, :a, :id, :t;
                        regime=[always_treat(3), never_treat(3)],
                        covariates=[:l1, :l2], baseline=[:w1, :w2],
                        estimator=:sdr, n_folds=5, rng=StableRNG(1))
regime_contrast(r; reference="never treat")          # E[Y^(1,1,1)] - E[Y^(0,0,0)]
regime_contrast(r; reference=2, type=:ratio)         # log risk ratio (binary y)
regime_weights(r)                                    # support of each regime
```

A dynamic regime is a rule applied to the history; the rule sees the period, the
baseline covariates, the time-varying covariates up to the current period and the
treatments before it:

```julia
d = DynamicRegime(h -> h.covariates[end].l1 > 0 ? 1 : 0; label="treat when l1 > 0")
longitudinal_regime(df, :y, :a, :id, :t; regime=[d, never_treat(3)],
                    covariates=[:l1, :l2], baseline=[:w1, :w2])
```

Several static regimes can be summarized by a working marginal structural model;
``E[Y^{\bar a}] = β_0 + β_1 \sum_t a_t`` for the regimes
``(0,0,0), (1,0,0), (1,1,0), (1,1,1)`` is

```julia
rs = [StaticRegime([0, 0, 0]), StaticRegime([1, 0, 0]), StaticRegime([1, 1, 0]),
      StaticRegime([1, 1, 1])]
r = longitudinal_regime(df, :y, :a, :id, :t; regime=rs, covariates=[:l1, :l2],
                        baseline=[:w1, :w2])
regime_msm(r, [1 0; 1 1; 1 2; 1 3]; names=["intercept", "per treated period"])
```

## [Positivity and weight diagnostics](@id longitudinal-diagnostics)

[`regime_weights`](@ref) tabulates, for every regime and period, the number and share
of followers, the smallest estimated probabilities of the regime's action among
them, the distribution of the cumulative weights, and how many followers had their
cumulative probability raised to `g_bound`; [`plot_regime_weights`](@ref) draws the
weight distributions. Few followers in the last periods, probabilities near zero,
weights near ``1 / g_{bound}`` and a heavy right tail are the signatures of weak
support (Cole and Hernán 2008; Petersen et al. 2012). In that situation the
inverse-probability estimators are unstable, the targeted and sequentially doubly
robust estimators lean on a handful of units, and g-computation extrapolates from
its parametric form. The remedies are substantive, not statistical: a regime with
more support, fewer periods, a coarser time scale, or a larger `g_bound` accepted as
a biased but stable compromise. None of the diagnostics speaks to sequential
exchangeability.

## Data requirements and conventions

- Long format, one row per unit and period, every unit observed in every period
  (an `ArgumentError` lists the missing cells otherwise); rows in any order. Units
  are matched by their identifier and periods by the sorted values of the time
  column, so results are invariant to row order.
- `treatment` is 0/1 and must vary within every period. `covariates` are the
  time-varying covariates measured in each period *before* that period's treatment;
  `baseline` covariates must be constant within units. The `outcome` is read from
  each unit's last period and may be missing in earlier rows. Categorical variables
  are dummy-encoded by the user.
- The outcome regressions use the whole history (baseline, all covariates up to the
  period, all treatments up to the period) and predict with the full treatment
  history replaced by the regime's actions, as ltmle does; the treatment mechanism
  uses the history before the period's treatment. With cross-fitting, folds are
  drawn over units (whole clusters with `cluster`).
- A [`TreatmentPanel`](@ref) can be passed instead of the column names.

Not implemented: censoring and competing events (every unit must be observed
through the last period), repeated outcomes and survival outcomes, continuous or
multi-valued treatments and modified treatment policies, the parametric g-formula by
Monte Carlo simulation of the covariate process (the iterated-regression form is the
g-formula estimator provided), g-estimation of structural nested models, and the
learning of optimal regimes.

## Validation

- **ltmle 1.3.0 parity** (`test/validation/longitudinal/ltmle_reference.R`): on the
  simulated three-period data with treatment–confounder feedback (continuous and
  binary outcomes, two baseline and two time-varying covariates), `:tmle`, `:ipw`
  and `:gcomp` with `FractionalLogitLearner`, `LogisticLearner` and no cross-fitting
  reproduce ltmle's estimates for static regimes (always, never, treat once), a
  dynamic regime and the two-regime effect measures (difference, log risk ratio, log
  odds ratio) to about `1e-8`, and the influence-curve standard errors to the same
  precision after the two conventions are aligned (ltmle divides the variance by
  ``n - 1`` and omits the Hájek normalization in the IPW influence function).
- **lmtp 1.5.4 cross-check**: `lmtp_sdr` and `lmtp_tmle` without cross-fitting and
  with `glm` learners give estimates within about 4% of `:sdr` and `:tmle` (standard
  errors within 5%); lmtp
  estimates the density ratios by a classification trick rather than from the
  propensity scores, so the two are not expected to coincide.
- **Monte Carlo** (`test/longitudinal/test_montecarlo.jl`): on a design with
  treatment–confounder feedback and known truth, `:tmle` and `:sdr` cover at the
  nominal rate with correctly specified parametric nuisances and with cross-fitted
  learners; the double robustness is checked by misspecifying either the outcome
  regressions or the treatment mechanism, under which `:gcomp` or `:ipw`
  respectively lose consistency while `:tmle` and `:sdr` do not.

## References

- Bang, H., & Robins, J. M. (2005). Doubly robust estimation in missing data and
  causal inference models. *Biometrics*, 61(4), 962–973.
- Chernozhukov, V., Chetverikov, D., Demirer, M., Duflo, E., Hansen, C., Newey, W., &
  Robins, J. (2018). Double/debiased machine learning for treatment and structural
  parameters. *The Econometrics Journal*, 21(1), C1–C68.
- Cole, S. R., & Hernán, M. A. (2008). Constructing inverse probability weights for
  marginal structural models. *American Journal of Epidemiology*, 168(6), 656–664.
- Díaz, I., Williams, N., Hoffman, K. L., & Schenck, E. J. (2023). Nonparametric
  causal effects based on longitudinal modified treatment policies. *Journal of the
  American Statistical Association*, 118(542), 846–857.
- Hernán, M. A., & Robins, J. M. (2020). *Causal Inference: What If*. Chapman &
  Hall/CRC.
- Lendle, S. D., Schwab, J., Petersen, M. L., & van der Laan, M. J. (2017). ltmle: An
  R package implementing targeted minimum loss-based estimation for longitudinal
  data. *Journal of Statistical Software*, 81(1), 1–21.
- Luedtke, A. R., Sofrygin, O., van der Laan, M. J., & Carone, M. (2017). Sequential
  double robustness in right-censored longitudinal models. arXiv:1705.02459.
- Murphy, S. A. (2003). Optimal dynamic treatment regimes. *Journal of the Royal
  Statistical Society: Series B (Statistical Methodology)*, 65(2), 331–355.
- Petersen, M. L., Porter, K. E., Gruber, S., Wang, Y., & van der Laan, M. J. (2012).
  Diagnosing and responding to violations in the positivity assumption. *Statistical
  Methods in Medical Research*, 21(1), 31–54.
- Petersen, M., Schwab, J., Gruber, S., Blaser, N., Schomaker, M., & van der Laan,
  M. (2014). Targeted maximum likelihood estimation for dynamic and static
  longitudinal marginal structural working models. *Journal of Causal Inference*,
  2(2), 147–185.
- Robins, J. M. (1986). A new approach to causal inference in mortality studies with a
  sustained exposure period—application to control of the healthy worker survivor
  effect. *Mathematical Modelling*, 7(9–12), 1393–1512.
- Robins, J. M., Hernán, M. A., & Brumback, B. (2000). Marginal structural models and
  causal inference in epidemiology. *Epidemiology*, 11(5), 550–560.
- Rotnitzky, A., Robins, J., & Babino, L. (2017). On the multiply robust estimation
  of the mean of a longitudinal outcome. arXiv:1705.08582.
- van der Laan, M. J., & Gruber, S. (2012). Targeted minimum loss based estimation
  of causal effects of multiple time point interventions. *The International Journal
  of Biostatistics*, 8(1), Article 9.
- Williams, N. T., & Díaz, I. (2023). lmtp: An R package for estimating the causal
  effects of modified treatment policies. *Observational Studies*, 9(2), 103–122.

The functions and types described on this page are documented in the [API reference](reference/longitudinal.md).
