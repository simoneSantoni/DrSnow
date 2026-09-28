# Instrumental Variables and LATE

```@meta
CurrentModule = DrSnow
```

DrSnow's IV tools estimate linear IV models by two-stage least squares, report what
the estimate identifies under heterogeneous effects, measure instrument strength,
provide inference that remains valid with weak instruments, describe compliers, and
probe the identifying assumptions. Estimates subtype `CausalEstimate`, so `coef`,
`vcov`, `stderror`, `confint(r; level)`, `coeftable` and `nobs` work uniformly; tests
return a `DiagnosticTest`.

```julia
using DrSnow
r = late_2sls(df, :earnings, :training, :offer; covariates=[:age], cluster=:site)
r.first_stage                       # F statistics, effective F, critical values
weak_iv_confidence_set(r)           # Anderson–Rubin set (analytic)
tf_confint(r)                       # tF interval (just-identified)
estimate_compliance(df, :training, :offer)
complier_characteristics(df, :training, :offer, [:age, :female])
plausibly_exogenous(r; method=:uci, gamma=(0.0, 0.3))
```

A worked example with simulated data is in `examples/late_estimation_demo.jl`;
`examples/iv_designs_demo.jl` covers many instruments, judge designs, shift-share
instruments and marginal treatment effects.

## Estimation

[`iv_regression`](@ref)`(data, y, endogenous, instruments; ...)` fits

```math
y_i = D_i'\beta + W_i'\delta + \alpha_{g(i)} + u_i, \qquad
D_i = Z_i'\Pi + W_i'\Gamma + \mu_{g(i)} + v_i
```

by 2SLS through `FixedEffectModels.reg`, with any number of endogenous regressors
`D` and excluded instruments `Z` (at least as many), included covariates `W`
(`covariates`, categorical columns are dummy-coded), absorbed fixed effects `α_g`
(`fe`), analytic weights (`weights`), and covariance `vcov = Vcov.simple()`,
`Vcov.robust()` (HC1, **the default**) or `Vcov.cluster(:a, :b)` (one- or multi-way;
`cluster = :a` is shorthand). Small-sample corrections follow Stata/`fixest`:
`n/(n−K)` for HC1 and `(n−1)/(n−K)·G/(G−1)` for clustering with `t(G−1)` reference
distributions. [`late_2sls`](@ref)`(data, y, d, z; ...)` is the same estimator with
one endogenous treatment.

Coefficients are ordered with the endogenous regressors first, so `estimate(r)` is
the treatment coefficient. `r.first_stage` holds a [`WeakIVDiagnostics`](@ref).
Rows with missing values in used columns are dropped; singleton fixed-effect groups
are dropped as in `reg`. If an endogenous regressor is collinear with the
instruments, controls and other endogenous regressors (e.g. experience = age −
education − 6 with age as an instrument) the model is not identified and an error is
raised.

## What does 2SLS estimate?

With heterogeneous effects, 2SLS estimates a weighted average of causal effects whose
weights depend on the specification. [`late_2sls`](@ref) labels the estimand
(`estimand(r)`, explanation in `r.estimand_note`):

| Specification | Estimand | Key conditions |
|:--|:--|:--|
| binary `D`, binary `Z`, no controls | LATE for compliers (Imbens & Angrist 1994) | independence, exclusion, monotonicity, relevance |
| binary `D`, binary `Z`, controls = one set of cell fixed effects | non-negatively weighted average of within-cell LATEs; weights ∝ Var(Z∣cell) × cell first stage | the above within cells (Angrist & Imbens 1995) |
| binary `D`, binary `Z`, other covariates | weighted combination of conditional LATEs; **weights can be negative** unless E[Z∣X] is linear in the controls | Blandhol, Bonney, Mogstad & Torgovitsky (2026); Słoczyński (2020) |
| multi-valued / continuous `D`, binary `Z`, no controls | average causal response (ACR) | Angrist & Imbens (1995) |
| multi-valued / continuous `D`, binary `Z`, controls = one set of cell fixed effects | weighted average of within-cell ACRs; weights non-negative only if the cell first stages share a sign | Angrist & Imbens (1995) within cells |
| multi-valued / continuous `D`, binary `Z`, other covariates | weighted combination of conditional ACRs; **weights can be negative** unless E[Z∣X] is linear in the controls | Blandhol, Bonney, Mogstad & Torgovitsky (2026); Słoczyński (2020) |
| several instruments or multi-valued `Z` | weighted average of instrument-specific LATEs/ACRs | weights non-negative only under monotonicity of the combined first-stage index (Mogstad, Torgovitsky & Walters 2021); see [`multiple_iv_weights`](@ref) |
| several endogenous regressors | structural coefficients of a linear constant-effects model | no general LATE interpretation |

When the instrument is valid only conditional on covariates, [`late_ipw`](@ref)
estimates the **unconditional** LATE by reweighting the instrument arms with a logit
propensity score (the Wald form of Abadie's 2003 κ-weighting; Frölich 2007), and also
reports the complier potential-outcome means `E[Y(1)∣C]` and `E[Y(0)∣C]`. Saturating
discrete controls (a single `fe` of cells) is the 2SLS alternative; its weights differ
from complier shares.

## Instrument strength

`r.first_stage` (or [`first_stage_diagnostics`](@ref)) reports, per endogenous
regressor ([`FirstStageResult`](@ref)): the first-stage coefficients and covariance,
the Wald F with the model's covariance estimator (robust/cluster by default), the
conventional homoskedastic F, the partial R² and the Sanderson–Windmeijer (2016)
conditional F (homoskedastic form, relevant with several endogenous regressors); and
jointly the Cragg–Donald and Kleibergen–Paap rk Wald F statistics.

- **Effective F** (Montiel Olea & Pflueger 2013), one endogenous regressor:
  ``F_\text{eff} = \hat\pi' Q_{ZZ} \hat\pi / \operatorname{tr}(\hat V_\pi Q_{ZZ})``
  with the robust/cluster covariance ``\hat V_\pi``. It reduces to the robust F with
  one instrument and to the conventional F under homoskedasticity. The *simplified*
  (conservative) critical values test whether the worst-case Nagar bias of 2SLS
  exceeds a fraction τ of the OLS benchmark at the 5% level:
  ``c = \chi^2_{K_\text{eff}}(x K_\text{eff})^{-1}_{0.95}/K_\text{eff}``, ``x = 1/τ``;
  e.g. 23.11 for τ = 10% with one instrument. This is the recommended pre-test under
  heteroskedasticity, clustering or serial correlation (Andrews, Stock & Sun 2019).
- **Stock–Yogo** critical values (size and relative bias, one endogenous regressor,
  ≤ 10 instruments) are provided only as a reference for the Cragg–Donald F under
  **homoskedastic** errors. They are not valid for robust or clustered F statistics,
  and the "F > 10" rule of thumb is not a Stock–Yogo critical value.
- [`tf_confint`](@ref) (Lee, McCrary, Moreira & Porter 2022): for a just-identified
  model, replaces 1.96 by a critical value c(F) that depends on the first-stage F
  (same covariance estimator as the t-ratio); c(10) ≈ 3.43, c(F) = 1.96 for
  F > 104.7, unbounded for F < 3.84.

Screening specifications on the first-stage F and dropping the weak ones distorts
inference; report weak-IV-robust sets instead.

## Weak-instrument-robust inference

[`weak_iv_test`](@ref) and [`weak_iv_confidence_set`](@ref) invert tests whose size
does not depend on instrument strength ([`WeakIVConfidenceSet`](@ref)).

- **Anderson–Rubin (AR)**: regress `y − Dβ₀` on the instruments and controls and test
  that the instrument coefficients are zero, with the model's covariance estimator
  (homoskedastic, HC1 or cluster); reference F(k, dof). The confidence set is
  computed **analytically**:
  - homoskedastic (any k) and robust/cluster with k = 1: the acceptance region is the
    quadratic inequality ``(\hat\gamma - β\hat\pi)' A (\hat\gamma - β\hat\pi) \le
    c\,k\,σ^2(β)``, giving a bounded interval, the union of two rays, the whole line,
    or (overidentified) the empty set;
  - robust/cluster with k > 1: ``AR(β) = g(β)'V(β)^{-1}g(β)/k`` with `g` affine and
    `V` quadratic in β, so ``\det V(β)\,(AR(β) − c)`` is a polynomial of degree ≤ 2k
    whose real roots (Chebyshev colleague-matrix eigenvalues, polished by bisection
    on the exact statistic) are the set boundaries.

  The set is bounded if and only if the first-stage Wald statistic exceeds the
  critical value; an empty set in overidentified models signals that the
  overidentifying restrictions are rejected. With k > 1 the AR test also has power
  against invalid instruments, so AR sets can be conservative-looking or empty for
  that reason.
- **CLR** (Moreira 2003) and **K** (Kleibergen 2002) for one endogenous regressor.
  With `Vcov.simple()` these are Moreira's homoskedastic statistics: the CLR p-value
  is conditional on ``Q_T`` (Andrews, Moreira & Stock 2006, one-dimensional
  integral). With a robust or cluster covariance the **heteroskedasticity- /
  cluster-robust versions of Kleibergen (2005)** are used, as in Stata's `weakiv`
  (Finlay, Magnusson & Schaffer): from the reduced-form estimates ``\hat γ, \hat π``
  and their robust covariance, ``\hat g = \hat γ − β_0 \hat π``,
  ``Ω = \operatorname{Var}(\hat g)``,
  ``\tilde D = \hat π − \operatorname{Cov}(\hat π, \hat g)Ω^{-1}\hat g`` and
  ``Ψ = \operatorname{Var}(\tilde D)``,

  ```math
  AR = \hat g'Ω^{-1}\hat g,\quad
  K = \frac{(\tilde D'Ω^{-1}\hat g)^2}{\tilde D'Ω^{-1}\tilde D},\quad
  rk = \tilde D'Ψ^{-1}\tilde D,\quad
  LR = \tfrac12\bigl[AR − rk + \sqrt{(AR + rk)^2 − 4(AR − K)\,rk}\bigr],
  ```

  and the CLR p-value is Moreira's conditional p-value with ``Q_T`` replaced by
  ``rk``. With the homoskedastic covariance these reduce exactly to Moreira's
  ``Q_S``, ``Q_{ST}^2/Q_T`` and ``Q_T``; with k = 1, CLR = K = AR. Under
  heteroskedasticity this CLR is not the efficient test (Andrews 2016; Moreira &
  Moreira 2019 propose alternatives that are not implemented), but its size is
  robust to weak instruments. The robust CLR and K use χ² reference distributions
  (unlike AR, no F / t(G − 1) small-sample reference is available), so with few
  clusters they over-reject somewhat: in our simulations with three weak
  instruments, rejection rates at the 5% level were 7.7% (CLR and K) with 60
  clusters, 6.4% / 5.4% with 200 and 5.7% / 6.1% with 500 (AR: 5.9%, 6.0%, 4.7%).
  Confidence sets are obtained by numerically inverting
  the p-value on the compactified line (2001 points plus exact limits at ±∞,
  bisection refinement). The K test can have non-monotone power, which shows up as
  spurious extra pieces of its set.

- **Jackknife AR** (Mikusheva & Sun 2022), `method = :jackknife_ar`, one endogenous
  regressor: for many (possibly weak) instruments under heteroskedasticity,
  ``AR(β_0) = K^{-1/2}\sum_{i \ne j} P_{ij} e_i e_j / \sqrt{\hat Φ}`` with
  ``e = y − Dβ_0`` (controls partialled out) and the cross-fit variance
  ``\hat Φ = \frac{2}{K}\sum_{i\ne j} \frac{P_{ij}^2}{M_{ii}M_{jj} + M_{ij}^2}
  e_i (Me)_i e_j (Me)_j``; one-sided N(0, 1) reference. It is designed for a
  growing number of instruments (not a handful), assumes independent observations
  and costs O(n²K) operations. The confidence set is found by numerical inversion.
  The forwarding methods `weak_iv_test(r)` / `weak_iv_confidence_set(r)` also accept
  [`KClassEstimate`](@ref) and [`JIVEEstimate`](@ref) results.

In the just-identified case the AR test is efficient among unbiased tests and is the
recommended default (Andrews, Stock & Sun 2019); `DrSnow.pvalue(set, β₀)` (StatsAPI's
`pvalue`) returns the p-value function and `β₀ in set` tests membership.

## Compliers

For a binary instrument and binary treatment (Angrist, Imbens & Rubin 1996):

- [`estimate_compliance`](@ref): shares of compliers ``E[D∣Z=1]−E[D∣Z=0]``,
  always-takers ``E[D∣Z=0]`` and never-takers ``1−E[D∣Z=1]`` with their joint
  covariance ([`ComplianceAnalysis`](@ref)). If the instrument lowers take-up it is
  recoded as `1 − Z` and this is reported.
- [`complier_characteristics`](@ref): mean covariates of compliers
  ``(μ_1(XD) − μ_0(XD))/(μ_1(D) − μ_0(D))`` (Abadie's κ-weighted mean; the Wald ratio
  with `XD` as outcome), always-takers ``μ_0(XD)/μ_0(D)`` and never-takers
  ``μ_1(X(1−D))/μ_1(1−D)``, plus the complier-minus-population difference
  ([`ComplierProfile`](@ref)). Instrument-arm means ``μ_z`` are reweighted by a logit
  propensity when `covariates` are given.
- [`late_ipw`](@ref) ([`IPWLATEEstimate`](@ref)) and
  [`complier_outcome_distribution`](@ref) (complier CDFs of `Y(1)` and `Y(0)`,
  Imbens & Rubin 1997; Abadie 2002).

Standard errors come from influence functions that include the logit estimation step,
aggregated by observation or cluster. All complier quantities rely on independence,
exclusion and monotonicity. The v0.1 `complier_characteristics` regressed `X` on `Z`,
which is a balance test ([`instrument_balance`](@ref)), not a complier description.

## Beyond compliers: extrapolating LATEs

[`late_extrapolation`](@ref) ([`LATEExtrapolation`](@ref)) implements the covariate
reweighting of Angrist & Fernández-Val (2013) nonparametrically over cells of
discrete covariates. Within each cell the Wald ratio identifies the cell LATE. Under
**conditional effect ignorability** (within a cell, compliers, always-takers and
never-takers have the same average effect) the cell LATE is the cell's average
effect for everybody, and effects for other populations are reweighted averages:
the population (ATE), treated, untreated, always-takers, never-takers, or an
external population given by its cell distribution (`target_data`). The complier
target needs no extra assumption and equals [`late_ipw`](@ref) with the cells as
saturated covariates. Conditional effect ignorability cannot be tested with a single
instrument; the results are extrapolations. Cells with weak first stages make the
cell LATEs (ratios) and their delta-method SEs unreliable — a warning is issued when
a cell first-stage F is below 10.

The **parametric version**, `late_extrapolation(data, y, d, z; covariates)`
(Angrist & Fernández-Val 2013, Section 4), handles continuous covariates by
modelling the covariate-specific LATE as linear, ``\mathrm{LATE}(x) = x'δ``, with
``E[Y(0) ∣ X] = x'α``: under conditional effect ignorability the interacted 2SLS of
``Y`` on ``(x, D·x)`` with instruments ``(x, Z·x)`` estimates ``(α, δ)``, and the
interacted linear first stage ``E[D ∣ X, Z] = x'π_0 + Z·x'π_1`` gives the complier
share ``x'π_1`` (always-takers ``x'π_0``). Targets are weighted averages
``E[s(X)\,x'δ]/E[s(X)]`` with ``s`` the complier share (compliers), 1 (ATE), ``D``
(treated), ``1 − D`` (untreated), the always-/never-taker shares, or the covariate
distribution of `target_data`. Delta-method standard errors come from the stacked
influence functions of the first stage, the interacted 2SLS and the target
averages. With a saturated set of cell indicators the parametric and cell versions
coincide exactly. A warning is issued when the fitted complier share is not
positive somewhere (the linear first stage then contradicts monotonicity or is
misspecified); `r.cells` reports the coefficients of ``\mathrm{LATE}(x)`` and of
the complier share.

## Falsification and specification tests

None of the IV assumptions can be verified; the tests below can only detect certain
violations. A non-rejection is never evidence that an assumption holds.

| Test | Null hypothesis | What it can detect | Blind spots |
|:--|:--|:--|:--|
| [`instrument_balance`](@ref) | Z unrelated to pre-determined covariates / placebo outcomes (joint Wald, cross-equation covariance) | non-random assignment of Z | exclusion restriction; imbalance on unobservables |
| [`first_stage_sign_test`](@ref) | first stage has the pooled sign in every subgroup (one-sided t tests, Holm) | subgroups dominated by defiers | defiers offset by compliers within subgroups |
| [`zero_first_stage_test`](@ref) | no reduced-form effect in a subsample without compliers (van Kippersluis & Rietveld 2018) | direct effects of Z / confounding | requires a credible zero-first-stage group |
| [`instrument_validity_test`](@ref) | complier outcome densities are non-negative (Kitagawa 2015; variance-weighted KS, pooled bootstrap), or within cells of discrete `covariates` (κ-weighted moment inequalities, recentred bootstrap) | joint violations of independence (given covariates), exclusion, monotonicity | violations that keep densities non-negative; iid only |
| [`overidentification_test`](@ref) | all instruments valid given that a just-identifying subset is (Sargan / Hansen J) | instruments that disagree | all instruments invalid alike; heterogeneous LATEs also reject |
| [`endogeneity_test`](@ref) | regressor exogenous (Durbin–Wu–Hausman, control-function form, robust) | OLS–IV difference | presumes valid instruments; low power when weak |
| [`huber_mellace_test`](@ref) | always-/never-taker mean outcomes lie within the bounds implied by the mixtures with compliers (Huber & Mellace 2015; 4 moment inequalities, bootstrap with moment selection) | direct effects of Z, defiers, non-random Z that move these means | violations that keep the means within the bounds |

**Kitagawa's test with covariates** (Kitagawa 2015, Section 3.2): when the
instrument is valid only given discrete covariates ``X`` (cells), the implications
hold within cells and are tested through the unconditional moment inequalities
``E[κ_d(D, Z, X)\,1\{Y ∈ B, X = x\}] ≤ 0`` with
``κ_1 = D(p(X) − Z)/(p(X)(1 − p(X)))`` and
``κ_0 = (1 − D)(Z − p(X))/(p(X)(1 − p(X)))``, ``p(X) = P(Z = 1 ∣ X)`` estimated by
cell frequencies (no functional form). The statistic is the variance-weighted KS
supremum ``\sqrt N \max E_N[\hat κ_d g]/\max(ξ, \hat σ_d(g))`` over outcome
intervals and cells; critical values come from the nonparametric bootstrap of the
recentred statistic, studentized with the full-sample ``\hat σ_d(g)`` (Kitagawa
uses the bootstrap-sample standard deviation, which with κ-weights inflates the
critical value when a narrow interval loses its few observations; see the
docstring). Like Kitagawa's, the bootstrap ignores the estimation of ``p(X)`` and
selects no moments, so the test is conservative when many inequalities are slack;
at the least-favourable null its size is nominal (Monte Carlo in
`test/iv/test_kitagawa_covariates.jl`). The conditional moment-inequality
(intersection-bounds) version of Mourifié & Wan (2017), which allows continuous
covariates, is not implemented.

The v0.1 `test_monotonicity` (sign of the pooled first stage),
`test_exclusion_restriction` (hard-coded pass) and `external_validity_test`
(placeholder) were removed; [`late_extrapolation`](@ref) replaces the latter with a
genuine (assumption-explicit) extrapolation method.

## Designs and machine-learning diagnostics

Many-instrument estimators, judge and shift-share designs and marginal treatment effects are covered in [IV designs](iv_designs.md). Specification tests and first-stage diagnostics for machine-learning first stages, and distributional LATE, are covered in [IV with machine-learning first stages](iv_ml.md).

## Sensitivity to the exclusion restriction

[`plausibly_exogenous`](@ref) (Conley, Hansen & Rossi 2012) allows a direct effect
``γ`` of the instruments on the outcome, ``y = Dβ + Zγ + Wδ + u``. ``γ`` is measured
in **outcome units per unit of the instrument** (like a reduced-form coefficient).
2SLS then converges to ``β + Aγ`` with ``A = (\hat D'\hat D)^{-1}\hat D'Z``, so a
positive direct effect biases 2SLS upward.

- `method = :uci`: union over a box of γ values of the 2SLS confidence intervals for
  ``y − Zγ``; the interval endpoints are concave/convex in γ, so the union is attained
  at the box vertices and is computed exactly. Coverage is at least `level` for every
  γ in the support.
- `method = :ltz`: local-to-zero prior ``γ ∼ N(μ, Ω)``:
  ``\hat β − Aμ ± c\sqrt{\hat V + AΩA'}``.

A reduced-form estimate from [`zero_first_stage_test`](@ref) is a natural source for
the prior (van Kippersluis & Rietveld 2018).

## Validation

`test/iv/test_validation.jl` compares DrSnow with R on the Card (1995) and Mroz (1987)
data (`test/validation/iv/make_reference.R`): 2SLS coefficients and homoskedastic,
HC1, one-way and two-way cluster standard errors (`AER::ivreg`, `sandwich`,
`fixest`, including fixed effects and weights), AER's weak-instrument F, Wu–Hausman
and Sargan statistics, `ivmodel`'s AR test/set and CLR test/set, robust and cluster
AR sets by brute-force root finding, effective F and Olea–Pflueger critical values,
ivDiag's tF critical value, Hansen J from two-step GMM, Cragg–Donald and
Sanderson–Windmeijer statistics, LTZ/UCI intervals, compliance shares and complier
means, the IPW LATE (point estimates exactly; SEs against the bootstrap), and the
reweighted LATE targets (point estimates and delta-method SEs exactly).

The second wave (`test/iv/test_validation_manyiv.jl`, `test_judge.jl`,
`test_shift_share.jl`, `test_mte.jl`; R scripts `make_reference_manyiv.R`,
`make_reference_judge.R`, `make_reference_shiftshare.R`, `make_reference_mte.R`)
checks: LIML and Fuller point estimates, κ, homoskedastic, HC0 and CR0 SEs against
`ivmodel`; Bekker SEs, HLIM/HFUL and their many-instrument variance, JIVE1/JIVE2
(through `AER::ivreg` + `sandwich`), UJIVE with EHW and many-instrument variances and
the Mikusheva–Sun statistic against explicit n × n matrix code, with fixed effects and
weights; the judge-design leniency, 2SLS (`fixest`), UJIVE with judge indicators and
the balance test; EHW, AKM and AKM0 inference against `ShiftShareSE`, the BHJ
shock-level regression, Rotemberg weights and randomization inference coded from the
papers; and the propensity score, polynomial / normal MTE parameters and the
semiparametric local-IV steps (cross-checked against `KernSmooth::locpoly`).
Monte Carlo tests check the coverage or size of most inferential procedures.

The third wave (`test_weak_iv_robust.jl`, `test_extrapolation_parametric.jl`,
`test_kitagawa_covariates.jl`, `test_hhn_cjive.jl`, `test_rotemberg_panel.jl`,
`test_fll.jl`, `test_mte_bounds.jl`; R scripts `make_reference_robust_extrap.R`,
`make_reference_hhn_cjive.R`, `make_reference_rotemberg.R`,
`make_reference_mtebounds.R`) checks: the robust CLR / K statistics and p-values
against an independent R implementation (conditional p-value by numerical
integration) and their exact reduction to Moreira's statistics under
homoskedasticity; the parametric extrapolation against a just-identified GMM
sandwich in R and against the cell version with saturated covariates; the κ-moments
of the conditional Kitagawa test against the conditional probabilities; the HHN
variance against the formula coded with explicit n × n matrices (and within 3% of
`ManyIV`'s minimum-distance standard error); CJIVE against `clusterIV::cjive`
(estimates and standard errors, with covariates and weights); panel Rotemberg
weights against GPSS's `bartik.weight` on the ADH data and their aggregation code,
and the 2SLS decomposition against `AER::ivreg`; the FLL spline, slope formula and
degrees of freedom by construction; and MTE bounds against `ivmte` (seven
specifications: ATT / ATE / ATU / LATE / generalized LATE; Bernstein polynomials,
linear, quadratic and constant splines; covariates, u-varying covariates, monotone
MTRs and MTE, several IV-like specifications, a positive minimum criterion, logit
and probit propensities) and closed-form nonparametric bounds.

The fourth wave (`test_ml_spec_test.jl`, `test_ml_first_stage.jl`,
`test_multiple_iv_weights.jl`, `test_dml_lqte.jl`; scripts `make_reference_rpiv.R`,
`make_reference_lqte.py`) checks: the residual prediction tests (strong version with
heteroskedastic, homoskedastic and cluster variances; weak-IV-robust version at
several β₀) against the authors' R package `RPIV` 1.1.1, whose calls are replayed step
by step so that the sample split and random-forest predictions can be passed to
DrSnow (statistics agree to 1e-10); the cross-fitted PLIV first-stage F, coefficients,
covariance and partial R² against `FixedEffectModels` on the out-of-fold residuals
(HC1 and cluster) and the IIVM first stage against [`dml_irm`](@ref) of ``D`` on
``Z``; the DML Anderson–Rubin statistic, set endpoints and repetition rule by
construction; the MTW decomposition on exact "population" datasets (the 2SLS
estimand equals ``Σ_g ω_g Δ_g`` to 1e-10 for saturated and additive first stages;
Propositions 5–7; the saturated estimate equals `FixedEffectModels` 2SLS with
interacted instruments); and the local potential quantiles, LQTEs and their DoubleML
standard errors against Python `DoubleML` 0.11.4 (`DoubleMLLPQ`, framework
difference as in `DoubleMLQTE`) with unpenalized logistic learners, identical folds
and the recorded nested splits (estimates to 1e-9, standard errors to 1e-6, nuisance
predictions to 1e-6). Monte Carlo tests check size, power or coverage of these
procedures, including the complier-CDF bands against the true complier distributions.

## Not covered

The Mourifié & Wan (2017) conditional moment-inequality version of Kitagawa's test
(continuous covariates), the Andrews (2016) / Moreira & Moreira (2019) efficient
robust conditional tests, confidence intervals for MTE bounds, and a
many-instrument correction to the CJIVE standard errors are not implemented.
The MTW decomposition does not handle covariates or continuous instruments, and its
test uses a least-favorable critical value rather than the Romano–Shaikh–Wolf
two-step procedure; the DML Anderson–Rubin test does not cover several instruments;
the DML complier distributions and quantile effects do not support clustering.

## References

- Abadie, A. (2002). Bootstrap tests for distributional treatment effects in
  instrumental variable models. *Journal of the American Statistical Association*,
  97(457), 284–292.
- Abadie, A. (2003). Semiparametric instrumental variable estimation of treatment
  response models. *Journal of Econometrics*, 113(2), 231–263.
- Adão, R., Kolesár, M., & Morales, E. (2019). Shift-share designs: Theory and
  inference. *Quarterly Journal of Economics*, 134(4), 1949–2010.
- Anderson, T. W., & Rubin, H. (1949). Estimation of the parameters of a single equation
  in a complete system of stochastic equations. *Annals of Mathematical Statistics*,
  20(1), 46–63.
- Andrews, D. W. K., & Shi, X. (2013). Inference based on conditional moment
  inequalities. *Econometrica*, 81(2), 609–666.
- Andrews, D. W. K., & Soares, G. (2010). Inference for parameters defined by moment
  inequalities using generalized moment selection. *Econometrica*, 78(1), 119–157.
- Andrews, D. W. K., Moreira, M. J., & Stock, J. H. (2006). Optimal two-sided invariant
  similar tests for instrumental variables regression. *Econometrica*, 74(3), 715–752.
- Andrews, I. (2016). Conditional linear combination tests for weakly identified models.
  *Econometrica*, 84(6), 2155–2182.
- Andrews, I., Stock, J. H., & Sun, L. (2019). Weak instruments in instrumental
  variables regression: Theory and practice. *Annual Review of Economics*, 11, 727–753.
- Angrist, J. D., & Fernández-Val, I. (2013). ExtrapoLATE-ing: External validity and
  overidentification in the LATE framework. In D. Acemoglu, M. Arellano, & E. Dekel
  (Eds.), *Advances in Economics and Econometrics: Tenth World Congress* (Vol. III, pp.
  401–434). Cambridge University Press.
- Angrist, J. D., & Imbens, G. W. (1995). Two-stage least squares estimation of average
  causal effects in models with variable treatment intensity. *Journal of the American
  Statistical Association*, 90(430), 431–442.
- Angrist, J. D., Imbens, G. W., & Krueger, A. B. (1999). Jackknife instrumental
  variables estimation. *Journal of Applied Econometrics*, 14(1), 57–67.
- Angrist, J. D., Imbens, G. W., & Rubin, D. B. (1996). Identification of causal effects
  using instrumental variables. *Journal of the American Statistical Association*,
  91(434), 444–455.
- Bekker, P. A. (1994). Alternative approximations to the distributions of instrumental
  variable estimators. *Econometrica*, 62(3), 657–681.
- Belloni, A., Chernozhukov, V., Fernández-Val, I., & Hansen, C. (2017). Program
  evaluation and causal inference with high-dimensional data. *Econometrica*, 85(1),
  233–298.
- Bhuller, M., Dahl, G. B., Løken, K. V., & Mogstad, M. (2020). Incarceration,
  recidivism, and employment. *Journal of Political Economy*, 128(4), 1269–1324.
- Blandhol, C., Bonney, J., Mogstad, M., & Torgovitsky, A. (2026). When is TSLS actually
  LATE? *Review of Economic Studies*, advance online publication.
  https://doi.org/10.1093/restud/rdag029
- Borusyak, K., & Hull, P. (2023). Nonrandom exposure to exogenous shocks.
  *Econometrica*, 91(6), 2155–2185.
- Borusyak, K., Hull, P., & Jaravel, X. (2022). Quasi-experimental shift-share research
  designs. *Review of Economic Studies*, 89(1), 181–213.
- Brinch, C. N., Mogstad, M., & Wiswall, M. (2017). Beyond LATE with a discrete
  instrument. *Journal of Political Economy*, 125(4), 985–1039.
- Carneiro, P., Heckman, J. J., & Vytlacil, E. J. (2011). Estimating marginal returns to
  education. *American Economic Review*, 101(6), 2754–2781.
- Chan, D. C., Gentzkow, M., & Yu, C. (2022). Selection with variation in diagnostic
  skill: Evidence from radiologists. *Quarterly Journal of Economics*, 137(2), 729–783.
- Chao, J. C., Swanson, N. R., Hausman, J. A., Newey, W. K., & Woutersen, T. (2012).
  Asymptotic distribution of JIVE in a heteroskedastic IV regression with many
  instruments. *Econometric Theory*, 28(1), 42–86.
- Chernozhukov, V., Fernández-Val, I., & Galichon, A. (2010). Quantile and probability
  curves without crossing. *Econometrica*, 78(3), 1093–1125.
- Chernozhukov, V., Hansen, C., & Spindler, M. (2015). Post-selection and
  post-regularization inference in linear models with many controls and instruments.
  *American Economic Review*, 105(5), 486–490.
- Conley, T. G., Hansen, C. B., & Rossi, P. E. (2012). Plausibly exogenous. *Review of
  Economics and Statistics*, 94(1), 260–272.
- Dobbie, W., Goldin, J., & Yang, C. S. (2018). The effects of pretrial detention on
  conviction, future crime, and employment: Evidence from randomly assigned judges.
  *American Economic Review*, 108(2), 201–240.
- Finlay, K., Magnusson, L. M., & Schaffer, M. E. (2013). WEAKIV: Stata module to
  perform weak-instrument-robust tests and confidence intervals for
  instrumental-variable (IV) estimation of linear, probit and tobit models.
  Statistical Software Components S457684, Boston College Department of Economics.
- Frandsen, B., Lefgren, L., & Leslie, E. (2023). Judging judge fixed effects. *American
  Economic Review*, 113(1), 253–277.
- Frandsen, B., Leslie, E., & McIntyre, S. (2025). Cluster jackknife instrumental
  variables estimation. *Review of Economics and Statistics*, advance online
  publication. https://doi.org/10.1162/rest.a.263
- Frölich, M. (2007). Nonparametric IV estimation of local average treatment effects
  with covariates. *Journal of Econometrics*, 139(1), 35–75.
- Frölich, M., & Melly, B. (2013). Unconditional quantile treatment effects under
  endogeneity. *Journal of Business & Economic Statistics*, 31(3), 346–357.
- Fuller, W. A. (1977). Some properties of a modification of the limited information
  estimator. *Econometrica*, 45(4), 939–953.
- Goff, L. (2024). A vector monotonicity assumption for multiple instruments. *Journal
  of Econometrics*, 241(1), 105735.
- Goldsmith-Pinkham, P., Sorkin, I., & Swift, H. (2020). Bartik instruments: What, when,
  why, and how. *American Economic Review*, 110(8), 2586–2624.
- Hansen, C., Hausman, J., & Newey, W. (2008). Estimation with many instrumental
  variables. *Journal of Business & Economic Statistics*, 26(4), 398–422.
- Hansen, L. P. (1982). Large sample properties of generalized method of moments
  estimators. *Econometrica*, 50(4), 1029–1054.
- Hausman, J. A. (1978). Specification tests in econometrics. *Econometrica*, 46(6),
  1251–1271.
- Hausman, J. A., Newey, W. K., Woutersen, T., Chao, J. C., & Swanson, N. R. (2012).
  Instrumental variable estimation with heteroskedasticity and many instruments.
  *Quantitative Economics*, 3(2), 211–255.
- Heckman, J. J., & Vytlacil, E. (2005). Structural equations, treatment effects, and
  econometric policy evaluation. *Econometrica*, 73(3), 669–738.
- Huber, M., & Mellace, G. (2015). Testing instrument validity for LATE identification
  based on inequality moment constraints. *Review of Economics and Statistics*, 97(2),
  398–411.
- Imbens, G. W., & Angrist, J. D. (1994). Identification and estimation of local average
  treatment effects. *Econometrica*, 62(2), 467–475.
- Imbens, G. W., & Rubin, D. B. (1997). Estimating outcome distributions for compliers
  in instrumental variables models. *Review of Economic Studies*, 64(4), 555–574.
- van Kippersluis, H., & Rietveld, C. A. (2018). Beyond plausibly exogenous. *The
  Econometrics Journal*, 21(3), 316–331.
- Kitagawa, T. (2015). A test for instrument validity. *Econometrica*, 83(5), 2043–2063.
- Kleibergen, F. (2002). Pivotal statistics for testing structural parameters in
  instrumental variables regression. *Econometrica*, 70(5), 1781–1803.
- Kleibergen, F. (2005). Testing parameters in GMM without assuming that they are
  identified. *Econometrica*, 73(4), 1103–1123.
- Kleibergen, F., & Paap, R. (2006). Generalized reduced rank tests using the singular
  value decomposition. *Journal of Econometrics*, 133(1), 97–126.
- Kling, J. R. (2006). Incarceration length, employment, and earnings. *American
  Economic Review*, 96(3), 863–876.
- Kolesár, M. (2013). Estimation in an instrumental variables model with treatment
  effect heterogeneity (Working Paper No. 2013-2). Princeton University, Department of
  Economics.
- Kolesár, M. (2018). Minimum distance approach to inference with many instruments.
  *Journal of Econometrics*, 204(1).
- Lee, D. S., McCrary, J., Moreira, M. J., & Porter, J. (2022). Valid t-ratio inference
  for IV. *American Economic Review*, 112(10), 3260–3290.
- Ma, Y. (2026). Identification-robust inference for the LATE with high-dimensional
  covariates. *Journal of Econometrics*, 257, 106302.
- Mikusheva, A., & Sun, L. (2022). Inference with many weak instruments. *Review of
  Economic Studies*, 89(5), 2663–2686.
- Mogstad, M., Santos, A., & Torgovitsky, A. (2018). Using instrumental variables for
  inference about policy relevant treatment parameters. *Econometrica*, 86(5),
  1589–1619.
- Mogstad, M., Torgovitsky, A., & Walters, C. R. (2021). The causal interpretation of
  two-stage least squares with multiple instrumental variables. *American Economic
  Review*, 111(11), 3663–3698.
- Montiel Olea, J. L., & Pflueger, C. (2013). A robust test for weak instruments.
  *Journal of Business & Economic Statistics*, 31(3), 358–369.
- Moreira, H., & Moreira, M. J. (2019). Optimal two-sided tests for instrumental
  variables regression with heteroskedastic and autocorrelated errors. *Journal of
  Econometrics*, 213(2).
- Moreira, M. J. (2003). A conditional likelihood ratio test for structural models.
  *Econometrica*, 71(4), 1027–1048.
- Mourifié, I., & Wan, Y. (2017). Testing local average treatment effect assumptions.
  *Review of Economics and Statistics*, 99(2), 305–313.
- Sanderson, E., & Windmeijer, F. (2016). A weak instrument F-test in linear IV models
  with multiple endogenous variables. *Journal of Econometrics*, 190(2), 212–221.
- Sargan, J. D. (1958). The estimation of economic relationships using instrumental
  variables. *Econometrica*, 26(3), 393–415.
- Scheidegger, C., Londschien, M., & Bühlmann, P. (2025). Machine-learning-powered
  specification testing in linear instrumental variable models. arXiv:2506.12771.
- Shea, J., & Torgovitsky, A. (2023). ivmte: An R package for extrapolating instrumental
  variable estimates away from compliers. *Observational Studies*, 9(2), 1–42.
- Słoczyński, T. (2020). When should we (not) interpret linear IV estimands as LATE?
  arXiv:2011.06695.
- Staiger, D., & Stock, J. H. (1997). Instrumental variables regression with weak
  instruments. *Econometrica*, 65(3), 557–586.
- Stock, J. H., & Yogo, M. (2005). Testing for weak instruments in linear IV regression.
  In D. W. K. Andrews & J. H. Stock (Eds.), *Identification and Inference for
  Econometric Models: Essays in Honor of Thomas Rothenberg* (pp. 80–108). Cambridge
  University Press.

The functions and types described on this page are documented in the [API reference](reference/iv.md).
