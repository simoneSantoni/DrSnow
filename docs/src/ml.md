# Causal machine learning

```@meta
CurrentModule = DrSnow
```

The `ml` area combines flexible prediction methods with estimands and inference
procedures whose validity does not depend on the prediction method being correct:

- **Double/debiased machine learning (DML)** for the partially linear model, the
  interactive (binary-treatment) model, the partially linear IV model, the
  interactive IV model (LATE with machine-learned nuisances) and the 2×2
  difference-in-differences design ([`dml_plr`](@ref), [`dml_irm`](@ref),
  [`dml_pliv`](@ref), [`dml_iivm`](@ref), [`dml_did`](@ref)).
- **Heterogeneous effects**: generalized random forests (causal, instrumental and
  regression forests with honest splitting and pointwise confidence intervals, a
  pure-Julia port of the R package grf: [`causal_forest`](@ref),
  [`instrumental_forest`](@ref), [`regression_forest`](@ref)) with AIPW average
  effects, calibration tests, best linear projections, variable importance and rank
  average treatment effects; the DR-learner ([`cate_dr_learner`](@ref)) with
  inference on the best linear projection of the CATE ([`cate_projection`](@ref));
  S-, T-, X- and R-learners ([`s_learner`](@ref), [`t_learner`](@ref),
  [`x_learner`](@ref), [`r_learner`](@ref)); generic ML inference for randomized
  experiments ([`generic_ml`](@ref): BLP, GATES, CLAN); and pre-specified subgroup
  and interaction analyses ([`subgroup_effects`](@ref),
  [`interaction_effects`](@ref)).
- **Policy learning** with doubly-robust scores and shallow decision trees
  ([`policy_tree`](@ref)).
- **Prediction-powered inference** for outcomes that are observed only in a small
  labeled sample but predicted (by an ML model or an LLM) in a large unlabeled
  sample ([`ppi_mean`](@ref), [`ppi_ols`](@ref), [`ppi_logistic`](@ref)).

## Nuisance learners

Every estimator takes learners for its nuisance functions (conditional means and
propensity scores). A learner is a [`NuisanceLearner`](@ref) implementing
[`fitpredict`](@ref) (and [`fitpredict_proba`](@ref) for probabilities). Built-in,
pure-Julia learners:

| Learner | Method | Tuning |
|:--|:--|:--|
| [`OLSLearner`](@ref) | least squares (pivoted QR) | none; deterministic |
| [`RidgeLearner`](@ref) | ridge on standardized covariates | exact leave-one-out CV (deterministic) or fixed λ |
| [`LassoLearner`](@ref) | lasso / elastic net, coordinate descent (glmnet parameterization) | K-fold CV (`:min` or `:one_se`) or fixed λ |
| [`LogisticLearner`](@ref) | logistic MLE (GLM.jl) | none; deterministic |
| [`PenalizedLogisticLearner`](@ref) | L1 / elastic-net / ridge logistic | K-fold CV on deviance or fixed λ |
| [`KNNLearner`](@ref) | k nearest neighbours | `k` |
| [`MeanLearner`](@ref) | training mean (e.g. known randomization) | none |
| [`ForestLearner`](@ref) | honest regression forest (grf algorithm) | forest parameters; multithreaded |

Any MLJ model can be used through [`MLJLearner`](@ref) once `MLJModelInterface` is
loaded (e.g. `using MLJ`), which activates the `DrSnowMLJExt` package extension.
Regressors work with `MLJModelInterface` alone; probabilistic classifiers need MLJ's
full data interface (`using MLJ` or `using MLJBase`). A custom learner only needs a
`fitpredict` method:

```julia
struct MyLearner <: NuisanceLearner end
DrSnow.fitpredict(::MyLearner, X, y, Xnew; rng, weights=nothing) = ...
```

Choose learners by out-of-fold prediction quality ([`nuisance_loss`](@ref) reports
the RMSE / log loss of every nuisance). DML is valid when the product of the
nuisance estimation errors vanishes faster than `n^{-1/2}`; flexible learners with
poor out-of-fold fit (in particular overfitted propensity scores near 0 or 1)
produce unstable estimates.

## Cross-fitting

All nuisances are *cross-fitted*: the sample is split into `K` folds and the
predictions for fold `k` come from learners trained on the other folds.
Repeating the split `R` times (`n_rep`) and aggregating reduces the dependence on a
particular split.

- **Folds**: drawn by [`crossfit_folds`](@ref); stratified by treatment (IRM, IIVM,
  DR-learner, policy tree), by treatment × instrument (IIVM) or by group × period
  (DiD for repeated cross-sections). A `folds` keyword accepts a vector, an
  `n × R` matrix or a column name, so the same splits can be reused across
  estimators or matched with another implementation.
- **Clusters**: with `cluster = :g`, all observations of a cluster are in the same
  fold, fold assignment is made on the sorted cluster labels (so it does not
  depend on row order), the variance is cluster-robust
  `G/(G-1) · Σ_g (Σ_{i∈g} ψᵢ)² / (Σᵢ ψ_a,i)²`, and intervals use `t(G - 1)`.
- **Aggregation over repetitions**: `θ̃ = median_r θ_r` and
  `σ̃² = median_r(σ_r² + (θ_r - θ̃)²)` (Chernozhukov et al. 2018, §3.4). DoubleML
  (R, 1.0.2) instead divides the squared deviation by `n`, which makes it nearly
  negligible; DrSnow's rule is the more conservative of the two. With `n_rep = 1`
  the two coincide.
- **Propensity clipping**: estimated propensities are clipped to
  `[trim, 1 - trim]` (default `trim = 0.01`); the number of clipped predictions is
  reported. Many clipped values indicate weak overlap, in which case the estimand is
  poorly identified and results should not be trusted.

## Reproducibility

Every stochastic function takes `rng`. The folds are drawn first, then one seed per
(repetition, nuisance, fold) task is drawn up front with `task_seeds`; each task
builds its own `Xoshiro(seed)`. Tasks run on threads when `parallel = true`, and the
results are bit-for-bit identical with or without threads. The seeds are stored in
the result (`r.seeds`). Results are invariant to the row order of the data when the
folds are keyed to the data (a `folds` column, or `cluster`), and the panel DML-DiD
estimator keys folds to sorted unit identifiers, so it is always invariant.

## Double/debiased ML estimators

All DML estimators use Neyman-orthogonal scores that are linear in the target
parameter, `ψ = θ ψ_a + ψ_b`; `θ` solves `mean(ψ) = 0` over the pooled folds ("DML2")
and `Var(θ̂) = mean(ψ²) / mean(ψ_a)² / n`. With deterministic learners and identical
folds, point estimates and standard errors reproduce the R package DoubleML to
machine precision (see *Validation* below).

| Function | Model | Estimand | Key assumptions |
|:--|:--|:--|:--|
| [`dml_plr`](@ref) | `Y = θD + g(X) + ε` | `θ` (constant effect / variance-weighted average) | `E[ε | D, X] = 0` |
| [`dml_irm`](@ref) | `Y = g(D, X) + U`, binary `D` | ATE or ATT | unconfoundedness, overlap |
| [`dml_pliv`](@ref) | `Y = θD + g(X) + ε` | `θ` | `E[ε | Z, X] = 0`, relevance |
| [`dml_iivm`](@ref) | binary `Z`, binary `D` | LATE | independence, exclusion, monotonicity, first stage, overlap |
| [`dml_did`](@ref) | 2×2 DiD (panel or repeated cross-sections) | ATT | conditional parallel trends, no anticipation, overlap |
| [`dml_did_multi`](@ref) | staggered adoption (panel or repeated cross-sections) | ATT(g,t) and their aggregations | conditional parallel trends per cohort, limited anticipation, overlap |

**Unconfoundedness and overlap (IRM, policy learning, DR-learner).** Potential
outcomes must be independent of treatment given `X`, and every unit must have a
propensity strictly between 0 and 1. Neither assumption is testable; covariates
must be pre-treatment.

**Instrument conditions (PLIV, IIVM).** The instrument must be as good as randomly
assigned given `X`, affect the outcome only through the treatment (exclusion), and
shift the treatment (relevance). For the LATE, monotonicity rules out defiers.
Only relevance and overlap are informed by the data; the estimand is the effect for
compliers. Use `always_takers = false` / `never_takers = false` under one-sided
non-compliance.

**Parallel trends (DML-DiD).** Absent treatment, the average outcome change of
treated units would have equalled that of control units with the same covariates.
Covariates should be pre-treatment (for panels they are taken from the first
period); with two periods parallel trends cannot be assessed. The scores are those
of Sant'Anna & Zhao (2020) with Hájek-normalized weights; the inference uses the
exact influence function of the normalized estimator. With in-sample OLS / logistic
nuisances the scores reproduce `DRDID::drdid_panel` and `DRDID::drdid_rc`.

**Multiple treatments and simultaneous inference.** [`dml_plr`](@ref) accepts
several treatments (each is partialled out with the others as controls, as in
DoubleML). [`simultaneous_confint`](@ref) returns sup-t intervals and max-t adjusted
p-values from a multiplier bootstrap of the scores.

### Staggered adoption: DML group-time effects

[`dml_did_multi`](@ref) is the machine-learning counterpart of
[`did_callaway_santanna`](@ref) (Callaway & Sant'Anna 2021), as DoubleML's
`DoubleMLDIDMulti`. Each group-time effect `ATT(g,t)` is a 2×2 comparison of cohort
`g` with its comparison group (never-treated units, or units not yet treated by
`t + anticipation`) between period `t` and a base period (`g − 1 − anticipation` for
post-treatment cells; the previous period for pre-treatment cells with the default
varying base period). Within each comparison the estimator uses the doubly-robust
score of Sant'Anna & Zhao (2020) with Hájek-normalized weights, as in [`dml_did`](@ref):

- panel data: an outcome regression of the change `Y_t − Y_base` in the comparison
  group and the generalized propensity score `P(G = g | X)` among cohort-`g` and
  comparison units;
- repeated cross-sections (`unit = nothing`): four outcome regressions (cohort and
  comparison group × the two periods) and the propensity score (locally efficient
  score).

The nuisances of every cell are cross-fitted on that cell's observations, using
folds drawn **once** over units (stratified by cohort, grouped by `cluster`), so all
cells share one sample split. Standard errors come from the influence functions,
which are stacked over cells; this is exactly the representation used by
`did_callaway_santanna`. The result is therefore a `CallawaySantAnnaEstimate`
(`settings.method = :dml`), and everything downstream works unchanged:
[`aggregate_att`](@ref) (`:simple`, `:group`, `:calendar`, `:dynamic`, with the
estimation of the cohort-share weights in the influence function),
multiplier-bootstrap sup-t bands (`confint(es; uniform = true)`),
[`pre_trend_test`](@ref), [`honest_did`](@ref) (with `base_period = :universal`) and
the plotting recipes. `r.settings` also records the learners, folds, per-repetition
estimates and a `nuisance_loss` table (out-of-fold RMSE and log loss per cell).

```julia
r = dml_did_multi(df, :y, FirstTreated(:first_treat), :id, :year;
                  covariates = [:x1, :x2], outcome_learner = ForestLearner(),
                  propensity_learner = ForestLearner(), rng = StableRNG(1))
es = aggregate_att(r, :dynamic)
confint(es; uniform = true)
aggregate_att(r, :group)
```

**Assumptions.** For every cohort, parallel trends conditional on the covariates
with respect to its comparison group; no anticipation beyond `anticipation`
periods; overlap (`P(G = g | X)` bounded away from one within each comparison);
covariates measured before treatment (the value in the earlier period of each
comparison is used, as in R's `did`). The learners must estimate the outcome
regressions and the propensity score consistently, with the product of their errors
vanishing faster than `n^{-1/2}` (the double robustness of the score).

**When does ML help?** Conditional parallel trends are credible only if the
covariates capture why cohorts differ in their untreated trends. When selection
into cohorts and the trends depend on the covariates non-linearly (thresholds,
interactions, U-shapes), the linear / logit working models of the parametric DR
estimator are both misspecified and its estimates are biased; flexible learners
remove most of that bias. When the linear working models are adequate,
`dml_did_multi` with `OLSLearner()` / `LogisticLearner()` gives estimates close to
`did_callaway_santanna(...; method = :dr)` (with `crossfit = false` it reproduces
them exactly), and ML learners mainly cost precision in small cohorts. With few
units per cohort, cross-fitting leaves little data for each nuisance fit: prefer
simple learners, fewer folds, or the parametric estimator.

Repeated cross-fitting (`n_rep > 1`) averages cell estimates and influence
functions over the repetitions (the mean rule, so that every aggregation is a
function of one set of cell estimates); the reported variance does not add the
between-split dispersion.

## Heterogeneous effects, policy learning and measurement

Heterogeneous treatment effects (forests, meta-learners, generic ML inference, subgroups) and policy learning are covered in [Heterogeneous effects](ml_hte.md). Prediction-powered inference and inference with machine-learned outcomes, covariates and labels are covered in [ML-measured variables](ml_measurement.md).

## Validation

- **DoubleML parity** (`test/validation/ml/`): simulated data and fold assignments
  are exported to R, where DoubleML 1.0.2 is run with `regr.lm` and
  `classif.log_reg` learners and the same splits. PLR (partialling-out, IV-type, two
  treatments), IRM (ATE, ATTE), PLIV (one and two instruments, IV-type) and IIVM
  (two- and one-sided) coefficients and standard errors agree to within `1e-10`
  relative error.
- **DRDID parity**: the DiD scores evaluated at in-sample OLS / logistic nuisances
  reproduce `DRDID::drdid_panel` and `DRDID::drdid_rc` point estimates exactly.
- **DoubleMLDIDMulti parity** (`doubleml_did_multi_reference.py`, Python DoubleML
  0.11.4 with scikit-learn `LinearRegression` / unpenalized `LogisticRegression`, each
  cell's folds set to DrSnow's unit-level folds): 54 `ATT(g,t)` in five cases (panel
  and repeated cross-sections; never-treated and not-yet-treated controls;
  anticipation) agree to `1e-7` (sklearn's solver tolerance), and so do the group,
  calendar and event-study aggregates. Standard errors equal, to `1e-8`, those of
  the influence function of the Hájek-normalized estimator computed from DoubleML's
  own nuisance predictions; DoubleML's linear score treats the normalizing constant
  of the comparison weights as known, and its cell standard errors differ by at most
  2.5% (aggregates: DoubleML also omits the estimation of the cohort-share weights).
  With anticipation, DoubleML uses long differences for pre-treatment cells, so only
  post-treatment cells are compared. Without cross-fitting (`crossfit = false`), OLS
  / logit nuisances reproduce `did_callaway_santanna(...; method = :dr)` on `mpdta`
  to `1e-10` (panel and repeated cross-sections, both control groups, anticipation,
  universal base period).
- **Learners**: closed forms (OLS, ridge, leave-one-out CV), KKT conditions (lasso,
  elastic net, penalized logistic), GLM (logistic).
- **grf parity, deterministic parts** (`grf_reference.R`): grf 2.6.1's
  `average_treatment_effect` (all four targets, subsets), `best_linear_projection`
  (HC3, HC1, overlap), `test_calibration`, `get_scores` and
  `rank_average_treatment_effect` (AUTOC, Qini, TOC, ties, two rules, custom grids)
  evaluated in R on DrSnow's nuisance estimates and out-of-bag predictions agree
  with DrSnow to `1e-8` relative error, for plain, clustered, sample-weighted and
  equalized-cluster forests, a continuous treatment and an instrumental forest.
- **grf parity, forest fits** (`grf_forest_reference.R`): with 20 seeds each on a
  common dataset (n = 1500, p = 6), causal, regression and instrumental forests of
  grf and DrSnow give the same distribution of average effects and standard
  errors, calibration and projection coefficients, variable importance, RATE, and
  predictions and variances at 60 test points (all 36 standardized differences of
  scalar summaries `|z| < 2.7`; single-fit deviations from grf's seed average are
  0.6–1.2 times grf's own seed-to-seed variance). Tests use `mc_reps(5, 1)` seeds.
- **Algorithmic checks**: forest weights reproduce predictions, honest leaves hold
  only in-bag units and none is empty, subsamples are unions of clusters, split
  constraints hold, a causal forest equals an instrumental forest with `Z = W`, and
  fits do not depend on threads or row order.
- **Forest coverage** (200 replications; grf 2.6.1 with 100 replications in
  brackets). Wager–Athey (2018) design with confounding and no effect, n = 1000,
  d = 2: ATE 0.955 [0.97], ATT 0.945 [0.96], overlap 0.950 [0.96], pointwise CATE
  0.941; d = 6, n = 2000: ATE 0.950 [0.91], ATT 0.965 [0.91], pointwise CATE 0.958.
  Randomized design with `τ(x) = ζ(x₁)ζ(x₂)` (steep logistic `ζ`), mean pointwise
  coverage at 100 test points: d = 2, n = 2000: 0.928 [0.925]; d = 2, n = 5000:
  0.938 [0.937]; d = 6, n = 2000: 0.866 [0.849] — the known small-sample
  undercoverage of forest intervals for sharp CATEs in more dimensions, reproduced
  by both implementations.
- **Meta-learners and subgroups**: closed forms with linear learners (S, T, X, R),
  equality with FixedEffectModels (subgroup and interaction regressions) and with
  the DR-learner scores (AIPW subgroup effects, projection); Monte Carlo coverage
  of AIPW subgroup effects, size of the equality test, and bootstrap coverage.
- **Performance**: a causal forest with n = 5000, p = 10 and 2000 trees (plus two
  500-tree nuisance forests) takes about 14 s on one thread and 3–4 s on 8
  threads; groups of trees and predictions run on threads.
- **Monte Carlo** coverage for most estimators (reduced counts in CI;
  `DRSNOW_SLOW_TESTS=true` for full counts), exhaustive-search optimality of policy
  trees against brute force, and row-order invariance.

**Monte Carlo for `dml_did_multi`** (`test/validation/ml/did_multi_montecarlo.jl`,
300 replications, T = 5, cohorts 3/4/5/never): cohort selection and untreated trends depend on
`0.8(x₁² − 1) + sin(2x₂)`, so the linear / logit working models are misspecified
and conditional parallel trends hold only given the non-linear index. Bias, SD
and mean SE of the simple aggregated ATT; coverage of the pointwise 95% intervals
and of the sup-t event-study band:

| N | Method | Bias | SD | Mean SE | Coverage (simple ATT) | Coverage (ATT(g,t)) | Coverage (uniform band) |
|--:|:--|--:|--:|--:|--:|--:|--:|
| 2000 | CS, no covariates | 0.307 | 0.055 | 0.055 | 0.000 | 0.345 | 0.000 |
| 2000 | CS, DR (linear/logit) | 0.292 | 0.056 | 0.055 | 0.000 | 0.377 | 0.000 |
| 2000 | DML, OLS/logit | 0.291 | 0.055 | 0.056 | 0.000 | 0.381 | 0.000 |
| 2000 | DML, random forests | 0.067 | 0.057 | 0.056 | 0.757 | 0.902 | 0.770 |
| 5000 | CS, no covariates | 0.311 | 0.035 | 0.035 | 0.000 | 0.075 | 0.000 |
| 5000 | CS, DR (linear/logit) | 0.296 | 0.035 | 0.035 | 0.000 | 0.098 | 0.000 |
| 5000 | DML, OLS/logit | 0.295 | 0.035 | 0.035 | 0.000 | 0.098 | 0.000 |
| 5000 | DML, random forests | 0.039 | 0.035 | 0.036 | 0.827 | 0.910 | 0.803 |

The parametric DR estimators (and `dml_did_multi` with linear learners) are
biased by 5–8 standard deviations; random-forest nuisances remove most of
the bias. The remaining forest regularization bias (about one standard deviation,
shrinking with N) makes the simple-ATT coverage 0.76–0.83 in this hard design,
while the standard errors match the sampling SD. In the correctly specified
design of the test suite (`test/ml/test_dml_did_multi.jl`) cell, simple-ATT and
band coverage are nominal.

## References

- Angelopoulos, A. N., Bates, S., Fannjiang, C., Jordan, M. I., & Zrnic, T. (2023).
  Prediction-powered inference. *Science*, 382(6671), 669–674.
- Angelopoulos, A. N., Duchi, J. C., & Zrnic, T. (2023). PPI++: Efficient
  prediction-powered inference. arXiv:2311.01453.
- Athey, S., Tibshirani, J., & Wager, S. (2019). Generalized random forests. *The Annals
  of Statistics*, 47(2), 1148–1178.
- Athey, S., & Wager, S. (2019). Estimating treatment effects with causal forests: An
  application. *Observational Studies*, 5(2), 37–51.
- Athey, S., & Wager, S. (2021). Policy learning with observational data.
  *Econometrica*, 89(1), 133–161.
- Bach, P., Kurz, M. S., Chernozhukov, V., Spindler, M., & Klaassen, S. (2024).
  DoubleML: An object-oriented implementation of double machine learning in R. *Journal
  of Statistical Software*, 108(3), 1–56.
- Callaway, B., & Sant'Anna, P. H. C. (2021). Difference-in-differences with multiple
  time periods. *Journal of Econometrics*, 225(2), 200–230.
- Chang, N.-C. (2020). Double/debiased machine learning for difference-in-differences
  models. *The Econometrics Journal*, 23(2), 177–191.
- Chernozhukov, V., Chetverikov, D., Demirer, M., Duflo, E., Hansen, C., Newey, W., &
  Robins, J. (2018). Double/debiased machine learning for treatment and structural
  parameters. *The Econometrics Journal*, 21(1), C1–C68.
- Chernozhukov, V., Demirer, M., Duflo, E., & Fernández-Val, I. (2025). Fisher–Schultz
  lecture: Generic machine learning inference on heterogeneous treatment effects in
  randomized experiments. *Econometrica*, 93(4), 1121–1164.
- Friedman, J., Hastie, T., & Tibshirani, R. (2010). Regularization paths for
  generalized linear models via coordinate descent. *Journal of Statistical Software*,
  33(1), 1–22.
- Holm, S. (1979). A simple sequentially rejective multiple test procedure.
  *Scandinavian Journal of Statistics*, 6(2), 65–70.
- Kennedy, E. H. (2023). Towards optimal doubly robust estimation of heterogeneous
  causal effects. *Electronic Journal of Statistics*, 17(2), 3008–3049.
- Künzel, S. R., Sekhon, J. S., Bickel, P. J., & Yu, B. (2019). Metalearners for
  estimating heterogeneous treatment effects using machine learning. *Proceedings of the
  National Academy of Sciences*, 116(10), 4156–4165.
- Nie, X., & Wager, S. (2021). Quasi-oracle estimation of heterogeneous treatment
  effects. *Biometrika*, 108(2), 299–319.
- Sant'Anna, P. H. C., & Zhao, J. (2020). Doubly robust difference-in-differences
  estimators. *Journal of Econometrics*, 219(1), 101–122.
- Semenova, V., & Chernozhukov, V. (2021). Debiased machine learning of conditional
  average treatment effects and other causal functions. *The Econometrics Journal*,
  24(2), 264–289.
- Wager, S., & Athey, S. (2018). Estimation and inference of heterogeneous treatment
  effects using random forests. *Journal of the American Statistical Association*,
  113(523), 1228–1242.
- Yadlowsky, S., Fleming, S., Shah, N., Brunskill, E., & Wager, S. (2025). Evaluating
  treatment prioritization rules via rank-weighted average treatment effects. *Journal
  of the American Statistical Association*, 120(549), 38–51.

The functions and types described on this page are documented in the [API reference](reference/ml.md).
