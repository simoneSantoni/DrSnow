# IV Designs: Many Instruments, Judges, Shift-Share, MTE

```@meta
CurrentModule = DrSnow
```

Estimators and diagnostics for specific instrumental-variable designs. They build on the core tools in the [main IV guide](iv.md).

## Many instruments: k-class and jackknife estimators

With many instruments 2SLS is biased toward OLS (the bias grows with K/F). DrSnow
provides the two standard families of alternatives; both use the same estimation
sample, controls, fixed effects and weights as [`iv_regression`](@ref) (whose fit is
kept in `r.tsls`), and report only the endogenous coefficients.

[`kclass_iv`](@ref) ([`KClassEstimate`](@ref)): with ``P`` the projection on the
(partialled) instruments and ``M = I − P``,
``\hat β(κ) = [D'(I − κM)D]^{-1} D'(I − κM) y``.

| `method` | κ | properties |
|:--|:--|:--|
| `:liml` | smallest root of ``\det(Ȳ'Ȳ − κ Ȳ'MȲ) = 0``, ``Ȳ = [y, D]`` | median-unbiased-ish, no moments; consistent under many-instrument asymptotics with homoskedastic errors (Bekker 1994) |
| `:fuller` | ``κ_{LIML} − α/(n − L)``, α = `fuller_alpha` (1 by default; 4 minimizes MSE) | finite moments (Fuller 1977) |
| `:kclass` | user `kappa` | κ = 0 is OLS, κ = 1 is 2SLS |
| `:hlim`, `:hful` | jackknife versions: ``P`` replaced by ``P − \operatorname{diag}(P)``; HFUL with ``α̂ = [α̃ − (1−α̃)C/n]/[1 − (1−α̃)C/n]`` | consistent with many instruments **and** heteroskedasticity (Hausman, Newey, Woutersen, Chao & Swanson 2012); LIML and Fuller are not |

Standard errors (`se`): `:standard` (the `cluster`/`vcov` type, κ treated as fixed —
usual fixed-K asymptotics), `:bekker` for LIML/Fuller (Bekker 1994; the Σ_B term of
Hansen, Hausman & Newey 2008, homoskedastic many-instrument asymptotics), `:hhn` for
LIML/Fuller (the full Hansen, Hausman & Newey 2008 variance, below) and
`:many_robust` for HLIM/HFUL (the Hausman et al. 2012 variance
``H^{-1}ΣH^{-1}`` with ``Σ = \sum_i ε̂_i^2 (Ẋ_i − P_{ii}X̂_i)(Ẋ_i − P_{ii}X̂_i)' +
\sum_{i\ne j} P_{ij}^2 X̂_i ε̂_i ε̂_j X̂_j'``, ``X̂ = D − ε̂ ε̂'D/ε̂'ε̂``, ``Ẋ = PX̂``).

**HHN variance.** Bekker's variance assumes normal errors. Hansen, Hausman & Newey
(2008) add the terms that third and fourth moments contribute under many-instrument
asymptotics: with all regressors ``X = [D, W]``, the projection ``P`` on all
instruments ``[Z, W]`` (diagonal ``p_{ii}``, rank ``K``), ``τ = K/n``,
``κ = \sum_i p_{ii}^2/K``, ``\hat Υ = PX``, ``\hat V = (I − P)\tilde X``,

```math
\hat Σ = \hat Σ_B + \hat A + \hat A' + \hat B,\quad
\hat A = \sum_i (p_{ii} − τ)\hat Υ_i \Bigl(\sum_j \hat u_j^2 \hat V_j/n\Bigr)',\quad
\hat B = \frac{K(κ − τ)\sum_i(\hat u_i^2 − \hat σ^2)\hat V_i\hat V_i'}
               {n(1 − 2τ + κτ)},
```

and ``\hat Λ = \hat H^{-1}\hat Σ\hat H^{-1}``. The corrections vanish when the
leverages ``p_{ii}`` are balanced (``p_{ii} ≈ τ``) and are small when ``K/n`` is
small; they matter with unbalanced instruments (e.g. group indicators of very
different sizes) and skewed or heavy-tailed errors. The variance still assumes
homoskedasticity (use HLIM/HFUL with `:many_robust` otherwise).

[`jive`](@ref) ([`JIVEEstimate`](@ref)) builds the first-stage prediction of each
unit from the *other* units: JIVE1 and JIVE2 (Angrist, Imbens & Krueger 1999) use the
leave-one-out prediction from the full first stage (instruments plus controls);
**UJIVE** (Kolesár 2013, the default) subtracts the leave-one-out prediction from the
controls alone, ``X̂ = [(I − D_Z)^{-1}(P_Z − D_Z) − (I − D_W)^{-1}(P_W − D_W)]D``,
``β̂ = (X̂'D)^{-1}X̂'y``, which stays consistent when the number of covariates (e.g.
fixed-effect levels) also grows. One fixed effect is handled with a sparse basis;
several fixed effects use a dense dummy basis (limited to moderate numbers of
levels). `se = :standard` treats the jackknife instrument as fixed (HC1 / cluster);
`se = :many_robust` (UJIVE) adds the many-instrument term
``\sum_{i\ne j} G_{ij}G_{ji}(D_iε̂_i)(D_jε̂_j)`` (Chao et al. 2012).

**Clustered data: CJIVE.** When errors are correlated within clusters, JIVE and
UJIVE do not remove the many-instrument bias: the leave-one-out prediction of unit
``i`` still uses the other units of its cluster, whose first-stage errors are
correlated with ``i``'s structural error. The cluster jackknife IV estimator
(`method = :cjive`, Frandsen, Leslie & McIntyre 2025) predicts the treatment of
every unit of cluster ``g`` from the first stage fitted **without cluster ``g``**,
``\hat D_g = \hat X_g − H_g(I − H_g)^{-1}\hat e_g`` (``H_g`` the cluster block of
the instrument projection), and uses ``\hat β = (\hat D'D)^{-1}\hat D'y`` with
cluster-robust standard errors that treat the leave-cluster-out instrument as fixed
(``G/(G − 1)`` correction, t(G − 1) reference), as in the authors' R package
`clusterIV`. Controls, fixed effects and weights are partialled out on the full
sample first (as in `clusterIV`); every instrument must vary outside each cluster.
These standard errors do not include a many-instrument correction term.

**Estimands.** With heterogeneous effects, UJIVE (like 2SLS) estimates a
non-negatively weighted average of LATEs under monotonicity (Kolesár 2013), and its
estimand label is taken from the 2SLS fit. LIML, Fuller, HLIM and HFUL target the
coefficient of a constant-effects model; their probability limits need not lie in
the range of the instrument-specific LATEs, and they are labelled accordingly.

Guidance: report the effective F; with many instruments prefer UJIVE or HFUL with
many-instrument robust standard errors (or LIML/Fuller with Bekker SEs under
homoskedasticity), and the jackknife AR set when instruments are also weak.

## Judge and examiner designs

Cases are randomly assigned (possibly within strata such as court × period) to
decision makers who differ in their propensity to treat. [`judge_leniency`](@ref)
computes the leave-one-out leniency of the assigned judge,
``Z_i = \sum_{k \in j(i), k \ne i} \tilde D_k / (n_{j(i)} − 1)``, from decisions
residualized on the strata fixed effects (and covariates; Dobbie, Goldin & Yang
2018). Leaving the own case out avoids the mechanical own-observation bias; judges
with a single case are dropped.

[`judge_iv`](@ref) ([`JudgeIVEstimate`](@ref)) estimates the effect of the decision by
2SLS on this instrument with strata fixed effects, or (`method = :ujive`) by UJIVE with
one indicator per judge (Kolesár 2013), with standard errors clustered by judge by
default.

**Estimand.** Under conditional random assignment and exclusion, the estimate is a
weighted average of treatment effects for *marginal* cases — those whose decision
depends on the judge drawn. The weights are non-negative under strict monotonicity
(a case treated by one judge is treated by every more lenient judge; Imbens & Angrist
1994) or under the weaker **average monotonicity** of Frandsen, Lefgren & Leslie
(2023). It is not the average effect of the decision on all cases.

Diagnostics:

| Function | What it checks |
|:--|:--|
| [`judge_balance_test`](@ref) | case characteristics do not predict the assigned judge's leniency (judge-clustered Wald F; the regression of the decision itself is reported for contrast) |
| [`judge_validity_test`](@ref) | joint implication of exclusion and monotonicity: judge mean outcomes are a continuous function of judge propensity with slope bounded by the outcome range (Frandsen, Lefgren & Leslie 2023) |
| [`judge_subsample_monotonicity`](@ref) | positive first stages within subgroups, with the leniency built from all cases ("standard") and from the judge's cases outside the subgroup ("reverse sample"; Bhuller et al. 2020) |

`judge_validity_test` implements **FLL's test** (their Section 3 and appendix).
After residualizing outcome and treatment on the covariates, ``p̂_i`` is the treatment
rate of the case's judge.
1. *Fit component*: regress ``Y_i`` on a quadratic B-spline in ``p̂_i`` (`n_knots`
   knots at quantiles of the judge propensities; strata indicators included), and
   test with a Wald statistic ``T = n\hat γ'\hat Ω^{+}\hat γ`` that the judge
   means ``\hat γ_j`` of the residuals are zero, ``χ^2`` with ``J − m − 1`` degrees
   of freedom (minus the number of strata − 1). ``\hat Ω`` accounts for the
   estimated propensities through the linearization of FLL's appendix; we also
   propagate the effect of ``p̂`` on the spline coefficients, which makes ``\hat Ω``
   singular in exactly the directions in which ``\hat γ`` is identically zero (hence
   the pseudo-inverse and the degrees of freedom).
2. *Slope component*: the spline slopes at the knots,
   ``φ'(t_l) = 2(δ_{l+1} − δ_l)/(t_{l+1} − t_{l−1})``, must lie in ``[−K, K]``,
   ``K = y_{max} − y_{min}``; the ``2m`` inequalities are tested with Andrews &
   Soares' (2010) generalized moment selection (MMM statistic, moments with
   standardized slack above ``\sqrt{\ln n}`` dropped, simulated p-value).
3. *Joint p-value* (weighted Bonferroni): ``\min\{p_f/ω, p_s/(1 − ω)\}``. FLL use
   ``ω = 1`` (fit only) with many judges and recommend ``ω`` near 0 with very few;
   the default is ``ω = 0.9``.

In simulations with 48 judges and 60 cases per judge the fit component is somewhat
conservative (rejection rates of 2–4% at the 5% level, consistent with FLL's and
`ivcheck`'s reports that estimated propensities compress the statistic). With few
cases per judge the test has limited power; with continuous outcomes the slope bound
is weak. The earlier minimum-distance approximation is available as
`method = :minimum_distance`.

## Shift-share instruments

A shift-share (Bartik) instrument combines exposure shares ``s_{ik}`` of region `i`
to sectors `k` with sector shocks ``g_k``: ``B_i = \sum_k s_{ik} g_k``
([`shift_share_instrument`](@ref)). [`shift_share_iv`](@ref)
([`ShiftShareIVEstimate`](@ref)) estimates the IV regression and reports, in
`r.inference`, region-level HC1 / cluster standard errors (for comparison) and
exposure-robust inference:

- **AKM** (Adão, Kolesár & Morales 2019), the default:
  ``\widehat{se}^2 = \sum_k (\hat h_k \sum_i s_{ik} \hat ε_i)^2 /
  (\sum_i \tilde x_i \tilde B_i)^2``
  with ``\hat h`` the coefficients of the partialled instrument on the shares; sums
  over sectors can be clustered (`shock_clusters`). The null-imposed **AKM0**
  confidence set (`r.akm0_set`) has better coverage with few or concentrated shocks
  and can be unbounded when the shift-share first stage is weak.
- **BHJ** (Borusyak, Hull & Jaravel 2022): the numerically equivalent shock-level IV
  regression of exposure-weighted residualized outcomes on treatments, instrumented by
  the shocks, weighted by total exposure ``s_k = \sum_i w_i s_{ik}``, with shock-level
  controls (an intercept and `shock_covariates`) and robust or `shock_clusters`
  standard errors (`se = :bhj`; `r.shock_level` holds the aggregated data). The
  equivalence requires region-level controls for ``\sum_k s_{ik} q_k`` for each
  shock-level control ``q_k`` — with incomplete shares, the sum of shares — which are
  added automatically.
- **Recentering** (Borusyak & Hull 2023): when exposure to shocks is not random
  (e.g. some regions are systematically more exposed), supply counterfactual shock
  draws (`shock_draws`, K × R, from the assignment process — e.g. permutations within
  clusters). The instrument is recentered by its expected value
  ``μ_i = \sum_k s_{ik} \bar g_k`` and randomization inference compares
  ``\sum_i \tilde B_i(\tilde y_i − β_0 \tilde x_i)`` with its counterfactual
  values; the confidence set in `r.ri.set` is computed exactly (the p-value function
  only changes at the crossing points of the counterfactual statistics).

[`rotemberg_weights`](@ref) ([`RotembergDecomposition`](@ref)) decomposes the Bartik
estimate into just-identified share-instrument estimates, ``\hat β = \sum_k \hat α_k
\hat β_k`` (Goldsmith-Pinkham, Sorkin & Swift 2020, GPSS). Under the exogenous-*shares*
view, the sectors with the largest ``|\hat α_k|`` are the ones whose exogeneity matters
most; negative weights mean the estimate is not a convex combination of the
share-specific estimands.

- **Panels** (`period = :year`, one row per region × period, period-specific shocks
  as a dictionary `period => shocks` or a ``K × T`` matrix): the instrument
  ``B_{it} = \sum_k s_{ikt} g_{kt}`` uses the ``K·T`` sector-period instruments
  ``s_k × 1\{t\}``; their weights and just-identified estimates are reported in
  `r.by_period` and aggregated by sector as in GPSS's replication code:
  ``α_k = \sum_t α_{kt}``, ``β_k = \sum_t α_{kt}β_{kt}/α_k``,
  ``g_k = \sum_t α_{kt}g_{kt}/α_k``.
- **Overidentified 2SLS** (`estimator = :tsls`): the decomposition of the 2SLS
  estimator that uses the shares (× periods) as separate instruments, with weights
  ``α_k ∝ \hat π_k s_k'Mx`` (``\hat π`` the first-stage coefficients) instead of
  ``g_k s_k'Mx`` (GPSS, Section IV).

**Estimand.** Under many as-good-as-random shocks, the shift-share IV estimates a
weighted average of region-level effects whose weights are non-negative only when the
first stage is monotone in the shocks (BHJ 2022).

## Marginal treatment effects

With a latent-index selection model ``D = 1\{U_D \le P(Z, X)\}``, ``U_D \sim U(0,1)``,
and additively separable potential outcomes, the marginal treatment effect
``\mathrm{MTE}(x, u) = E[Y_1 − Y_0 \mid X = x, U_D = u]`` is identified by local IV,
``\partial E[Y \mid X = x, P = p]/\partial p`` at ``p = u``, on the support of the
propensity score (Heckman & Vytlacil 2005).

[`mte_propensity`](@ref) fits the propensity (probit, logit or LPM) and reports the
common support ``[\max(\min P_{D=1}, \min P_{D=0}), \min(\max P_{D=1}, \max P_{D=0})]``,
quantiles by treatment status and a histogram. [`mte`](@ref) ([`MTEEstimate`](@ref))
estimates the MTE by
- `:semiparametric` (Carneiro, Heckman & Vytlacil 2011): Robinson's double-residual
  regression for the covariate part, then local quadratic regression in `P`
  (Gaussian kernel); identified only on the common support;
- `:polynomial` (Brinch, Mogstad & Wiswall 2017): a polynomial MTE in `u`;
- `:normal`: the joint-normal selection model, ``k(u) = c + s\,Φ^{-1}(u)``, by the
  local-IV regression on ``φ(Φ^{-1}(p))`` (not by maximum likelihood).

Treatment-effect parameters are weighted averages of the MTE (sample analogues):
ATE, ATT ``= \sum_i \int_0^{P_i}\mathrm{MTE}(X_i,u)du/\sum_i P_i``, ATU (parametric
methods only: they need the MTE on all of [0, 1], so outside the support they are
functional-form extrapolations), the average MTE over ``[a, b]`` (LATE), and the
policy-relevant treatment effect of a policy shifting propensities from ``P_i`` to
``P_i'`` (PRTE). Standard errors and pointwise MTE bands come from a nonparametric
bootstrap that re-estimates the propensity score (`rng`, per-draw seeds).

### Bounds for partially identified parameters

Parameters such as the ATE, ATT or the effect of policies that move propensities
outside the observed range are usually not point identified. Mogstad, Santos &
Torgovitsky (2018, MST) show that every IV-like estimand ``β_s = E[s(D, Z, X)Y]``
(the coefficients of any OLS / IV / 2SLS regression) and every such target
parameter are linear functionals of the marginal treatment response functions
``m_d(u, x) = E[Y(d) ∣ U = u, X = x]``:

```math
β_s = E\Bigl[s(1, Z, X)\int_0^{p(Z,X)} m_1(u, X)\,du +
          s(0, Z, X)\int_{p(Z,X)}^1 m_0(u, X)\,du\Bigr].
```

[`mte_bounds`](@ref) ([`MTEBounds`](@ref)) restricts the MTRs to a finite-dimensional
space — Bernstein polynomials (`basis = :bernstein`, `degree`) or B-splines
(`basis = :spline`, `degree`, `knots`; constant splines with knots at the
propensity values give the nonparametric case) in ``u``, plus additive covariates
and, for `interact`, covariate effects that vary with ``u`` — and computes the
smallest and largest target value over MTRs that satisfy the shape restrictions
(MTR bounds, by default the outcome range; MTE bounds; monotonicity of ``m_0``,
``m_1`` or the MTE in ``u``) and reproduce the estimated IV-like estimands. As in MST
and the R package `ivmte`, the ℓ₁ criterion ``\sum_s |Γ_s(θ) − \hat β_s|`` is first
minimized, and the bounds are taken over MTRs whose criterion is within
`criterion_tol` (relative) of the minimum. Shape restrictions are imposed on a
grid of ``u`` (`u_grid`) and on every observed covariate value. Targets: ATE, ATT,
ATU, the LATE of an instrument change (`late_from`, `late_to`), the average MTE
over ``[a, b]`` (`:genlate`) and the PRTE of a change in propensities.

The linear programs are solved with DrSnow's own simplex solver (shared with
`honest_did`), applied to the dual problem so that the basis size equals the number
of MTR coefficients. With `ivmte`'s audit grid the bounds reproduce `ivmte` to
``10^{-7}`` or better on its example data (`test/iv/test_mte_bounds.jl`).

Guidance: include IV-like estimands that are informative about the target (with a
discrete instrument, a saturated regression of ``Y`` on ``D``, the instrument
indicators and their interactions uses all the information in ``E[Y ∣ D, Z]``);
report how the bounds change with the basis, the shape restrictions and the grid.
Only estimated bounds are reported: no confidence interval accounts for the
sampling error of the propensity score and of the IV-like estimates.


Full references are listed at the end of the [main IV guide](iv.md).

The functions and types described on this page are documented in the [API reference](reference/iv.md).
