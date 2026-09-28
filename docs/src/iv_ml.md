# IV with Machine-Learning First Stages

```@meta
CurrentModule = DrSnow
```

## Machine-learning-era diagnostics and distributional LATE

These tools combine IV designs with machine-learned (cross-fitted) nuisance
functions, or use machine learning to look for misspecification. Every learner is a
[`NuisanceLearner`](@ref): `ForestLearner`, `LassoLearner`, `KNNLearner`, … or any
MLJ model through [`MLJLearner`](@ref). A runnable tour is
`examples/iv_ml_diagnostics_demo.jl`.

```julia
residual_prediction_test(df, :y, :d, [:z1, :z2]; covariates=[:x])     # specification
residual_prediction_confidence_set(df, :y, :d, [:z1, :z2]; covariates=[:x],
                                   weight_update=:linear)              # weak-IV robust
r = dml_pliv(df, :y, :d, :z; covariates=[:x1, :x2])
ml_first_stage(r, df; instrument=:z)            # cross-fitted first-stage F
dml_weak_iv_confidence_set(r)                   # DML Anderson–Rubin set
multiple_iv_weights(df, :d, [:z1, :z2]; outcome=:y)   # MTW 2SLS weights
dml_lqte(df, :y, :d, :z; covariates=[:x1, :x2], quantiles=0.1:0.1:0.9)
dml_complier_cdf(df, :y, :d, :z; covariates=[:x1, :x2])
```

### Residual prediction specification tests

[`residual_prediction_test`](@ref) implements the tests of Scheidegger, Londschien &
Bühlmann (2025) for the null that the linear IV model is well specified,
``E[Y − X'β − C'θ ∣ Z, C] = 0`` for some ``(β, θ)``. The sample is split; on the
auxiliary part a learner predicts the IV residuals from ``(Z, C)``, and its clipped
predictions ``ŵ`` (``|ŵ| ≤ 1``) are correlated with the residuals of the main part:
``T = Σ ŵᵢ rᵢ / \sqrt{n₀ σ̂²}``, compared with the one-sided ``N(0, 1)``. Because
``ŵ`` comes from the other part of the data, the size does not depend on how well
the learner does (a variance floor `gamma` guards against degenerate weights); power
does.

- Strong-identification version (default): 2SLS residuals, with a variance
  correction for estimating ``β``.
- Weak-IV-robust version (`beta0 = β₀`): residuals ``Y − X'β₀`` after partialling out
  the covariates, an Anderson–Rubin-type construction valid for any instrument
  strength. [`residual_prediction_confidence_set`](@ref) inverts it; **an empty set
  rejects the model**, so the set is a joint test of specification and a confidence
  set for ``β``.

What it can and cannot detect: a violation is detectable only if it is not in the
span of ``E[X ∣ Z]`` (Lemma 1 of the paper). With one instrument, 2SLS residuals are
linearly uncorrelated with ``Z`` by construction, so a linear direct effect of the
instrument is undetectable and only nonlinear violations (``Z²``, thresholds,
interactions with ``C``, nonlinear structural functions) can be found. Heteroskedastic
(default), homoskedastic and cluster variances are available; with `cluster` the split
is drawn by cluster. The result depends on the random split; report the seed or pass
`aux_sample`. In simulations following the paper's design (``n = 500``, two
instruments, heteroskedastic errors, 500 replications, forest weights) the rejection
rate is 0.058 under the null, 0.996 against a ``0.5 Z₁²`` violation, 0.068 against a
linear direct effect in the just-identified model (undetectable, as it should be),
and 0.060 for the weak-IV-robust test at the true ``β`` with nearly irrelevant
instruments; the inverted set covers the true ``β`` in 95.0% of replications.

### First-stage strength with machine-learned nuisances

Neyman orthogonality makes DML scores insensitive to first-order nuisance errors, but
**it does nothing for weak identification**: the DML IV estimator divides by the
orthogonalized first stage, and its Wald interval is unreliable when that is small.
[`ml_first_stage`](@ref) measures it on the out-of-fold residualized data, either with
its own cross-fitting or reusing the nuisances of a fitted [`dml_pliv`](@ref) /
[`dml_iivm`](@ref):

| Model | Statistic |
|:--|:--|
| PLIV | regression of ``D − \hat r(X)`` on ``Z − \hat m(X)``: robust (HC1/cluster) F, homoskedastic F, partial R², Montiel Olea–Pflueger effective F and critical values; with several instruments the F of a cross-fitted linear optimal instrument |
| IIVM | orthogonalized first stage ``E[r₁ − r₀ + Z(D − r₁)/m − (1−Z)(D − r₀)/(1−m)]`` (the complier share) and ``F = t²`` |

Printed reference thresholds are ``F ≥ 10`` (Staiger & Stock 1997), the
Olea–Pflueger critical value, and ``F ≥ 104.7`` for 5% t-tests with conventional
critical values (Lee et al. 2022). They were derived for linear 2SLS and are
heuristics for DML. The in-sample optimal-instrument projection that `dml_pliv` uses
with several instruments would overstate strength; the diagnostic uses a fold-wise
out-of-sample projection instead.

### Weak-instrument-robust DML inference

The PLIV partialling-out and IIVM scores are linear in the parameter,
``ψ(θ) = ψ_b + θψ_a``, and none of their nuisances depends on ``θ``. So
``ψ(β₀)`` has mean zero under ``H₀: θ = β₀`` whatever the instrument strength, and the
score statistic with the null imposed,
``AR(β₀) = (Σψᵢ(β₀))² / Σψᵢ(β₀)²`` (cluster sums with `cluster`), is ``χ²(1)``
(``F(1, G−1)`` with clusters): the DML version of the Anderson–Rubin test
(Chernozhukov, Hansen & Spindler 2015; Ma 2026 for the LATE). [`dml_weak_iv_test`](@ref)
reports it and [`dml_weak_iv_confidence_set`](@ref) inverts it in closed form (a
quadratic inequality; bounded, union of two rays, whole line or empty). With repeated
cross-fitting a value is in the set when at least half of the repetitions accept it.
In simulations with ``n = 400`` and ``ρ = 0.9`` (2,000 replications) the DML-AR test
rejects the true value in 4.9% of replications at every instrument strength, while the
DML Wald interval rejects it in 28% without identification, 15% with a median
cross-fitted F of 1.5 and 8.5% with a median F of 9. Several-instrument PLIV (whose
score uses an in-sample projection on the treatment) is not supported; use one
instrument or an index.

### Multiple instruments: when is 2SLS a positively weighted average?

With several discrete instruments and heterogeneous effects, 2SLS is
``β = Σ_g ω_g Δ_g`` over response groups ``g`` (vectors of potential treatments
``(D(z))_z``), with ``ω_g = π_g\,\mathrm{Cov}(D_g(Z), ψ(Z)) / \mathrm{Cov}(D, ψ(Z))``
and ``ψ`` the first-stage fitted value (Mogstad, Torgovitsky & Walters 2021).
Imbens–Angrist monotonicity with several instruments forces effectively homogeneous
choice behaviour; MTW's **partial monotonicity** (each instrument moves everybody in
the same direction, holding the others fixed) allows, with two binary instruments,
always- and never-takers, ``Z₁`` and ``Z₂`` compliers, and eager and reluctant
compliers. Under partial monotonicity 2SLS can put negative weight on the less common
single-instrument complier group; this requires negatively correlated instruments
(MTW Propositions 5–6).

[`multiple_iv_weights`](@ref) enumerates the groups admissible under `assumption =
:pm` (partial), `:vm` (vector monotonicity: one direction per instrument, Goff 2024)
or `:iam`, computes ``c_g = \mathrm{Cov}(D_g(Z), ψ(Z))`` for the saturated
(`first_stage = :saturated`, MTW's case) or additive (`:linear`) first stage,
reports MTW's complier/defier sets ``𝒞_g, 𝒟_g`` in propensity order, bounds each
group share and weight by linear programming over the shares that reproduce the
observed propensities, and tests ``H₀: c_g ≥ 0`` for every group that can be present
(least-favorable max-t bootstrap). Covariates are not supported: split the sample or
saturate them into the instrument cells. For a judge design (one multivalued
instrument) partial monotonicity equals Imbens–Angrist monotonicity and the saturated
(judge-dummy) 2SLS weights are non-negative; a non-saturated first stage or a second
instrument negatively correlated with leniency can create negative weights, which the
function reports.

### Distributional effects for compliers

[`dml_lqte`](@ref) estimates local quantile treatment effects ``q₁(τ) − q₀(τ)`` for
compliers with a binary instrument and cross-fitted classifiers, following DoubleML's
`DoubleMLLPQ` / `DoubleMLQTE` (Belloni, Chernozhukov, Fernández-Val & Hansen 2017;
Frölich & Melly 2013): the local potential quantile ``q_d(τ)`` solves the orthogonal
moment ``s_d E[g₁ − g₀ + Z(1\{D=d, Y≤q\} − g₁)/m − (1−Z)(1\{D=d, Y≤q\} − g₀)/(1−m)]
/ p_C = τ``, with ``g_z = P(D = d, Y ≤ q ∣ Z = z, X)`` fitted at a preliminary IPW
quantile from a nested split of each training fold, normalized IPW weights, and the
density at the quantile from a weighted Gaussian kernel. DoubleML treats the complier
share ``p_C`` in the denominator as known; the default `variance = :influence` adds
its estimation error to the influence function (`variance = :doubleml` reproduces
DoubleML). `confint(r; uniform=true)` gives sup-t bands over the quantiles. In
simulations (``n = 1500``, 500 replications, logistic learners, true LQTEs from the
design) the 95% intervals covered in 95.2–96.0% of replications at τ = 0.25, 0.5,
0.75 and the uniform band in 95.6%; in that design the complier-share term changed
the standard errors by less than 2%.

[`dml_complier_cdf`](@ref) estimates the complier distributions ``F_{Y(0)∣C}`` and
``F_{Y(1)∣C}`` on a grid with the same orthogonal score, uniform multiplier-bootstrap
bands and optional monotone rearrangement. It is the machine-learning counterpart of
[`complier_outcome_distribution`](@ref) (logit κ-weighting).


Full references are listed at the end of the [main IV guide](iv.md).

The functions and types described on this page are documented in the [API reference](reference/iv.md).
