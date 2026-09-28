# Prediction-Powered Inference and ML-Measured Variables

```@meta
CurrentModule = DrSnow
```

## Prediction-powered inference

When an outcome (e.g. a coded text variable) is measured on a small labeled sample
and predicted by an ML model or LLM on a large unlabeled sample, naive use of the
predictions biases estimates, and dropping the unlabeled data wastes information.
Prediction-powered inference (Angelopoulos et al. 2023) combines both: the
estimator uses the predictions on the unlabeled data and corrects their bias with
the labeled data. PPI++ power tuning (Angelopoulos, Duchi & Zrnic 2023) chooses the
weight `λ` of the predictions to minimize variance, so the interval is never
asymptotically wider than the labeled-only interval. Validity requires the labeled
units to be a random sample of the same population (or randomly selected for
labeling); the prediction model may be arbitrarily wrong. In a randomized
experiment with an ML-coded outcome, `ppi_ols(...; covariates = [:treated])`
estimates the treatment effect.

## Inference with ML-measured variables

Text, images and audio are increasingly coded by machine-learning models and large
language models (LLMs) rather than by people. Using such a prediction `f` as if it
were the variable it measures is safe only when its error is unrelated to everything
in the analysis. The damaging case is **differential prediction error**: the model
errs differently for treated and control units, after a policy, or on one side of a
cutoff (for instance because the intervention changes the wording the model reacts
to, or because the error depends on the true outcome that the treatment moves). The
naive estimate is then biased and its confidence interval is centred on the wrong
value; more data make this worse, not better.

The tools below combine the cheap measurement on every unit with gold-standard
(expert) labels on a subsample. They are valid *whatever the quality of the
measurement* as long as the labelling design is known: the labelled units are drawn
with known probabilities `πᵢ > 0` (a simple random sample, or probabilities that
depend on observed variables — oversampling rare categories, one arm, or units near a
cutoff is allowed). Better predictions only shorten the intervals.

### Design-based supervised learning

[`dsl_regression`](@ref) implements design-based supervised learning (Egami, Hinck,
Stewart & Wei 2023). A model `ĝ` of each ML-measured variable (outcome and/or
covariates) given the prediction(s) and optional `features` is fitted on labelled
rows and **cross-fitted**, and the regression solves the doubly robust moment

```math
\frac1n\sum_i \Big[\big(1-\tfrac{R_i}{\pi_i}\big)\, m(\hat D_i;\beta)
      + \tfrac{R_i}{\pi_i}\, m(D_i;\beta)\Big] = 0,
```

where `m` is the least-squares or logistic score, `D̂` the data with `ĝ` in place of
the measured variables and `D` the data with the gold standard. For an ML-measured
outcome in a linear model this is least squares on the pseudo-outcome
`Ỹ = ĝ + (R/π)(Y − ĝ)` ([`dsl_pseudo_outcome`](@ref)); with `fe` it is the
fixed-effects regression of `Ỹ`. Sandwich standard errors (clustered with `cluster`)
follow R's `dsl`. [`dsl_proportions`](@ref) estimates category proportions of a
coded variable, optionally by group (e.g. by year).

### Prediction-powered inference for causal targets

In a randomized experiment with an ML-predicted outcome, [`ppi_ate`](@ref) estimates
the average treatment effect from the pseudo-outcome `Ỹ(λ) = λf + (R/π)(Y − λf)`
(difference in means, Lin's regression adjustment with `covariates`, or absorbed
fixed effects with `fe`); [`ppi_regression`](@ref) does the same for a general
(fixed-effects) regression. The weight `λ` on the predictions is chosen to minimize
the estimated variance of the target coefficient (PPI++; Angelopoulos, Duchi & Zrnic
2023): `λ = 1` is prediction-powered inference and `λ = 0` ignores the predictions.
These functions use a single data frame in which labelled units are a subsample of
the analysed units, so the labelling may depend on treatment and covariates;
[`ppi_mean`](@ref), [`ppi_ols`](@ref) and [`ppi_logistic`](@ref) cover the
two-sample setting with a separate random labelled sample. When no pre-trained
predictor exists and the model is trained on the labelled data themselves,
[`cross_ppi`](@ref) implements cross-prediction-powered inference (Zrnic & Candès
2024): each labelled unit is predicted by a model that did not see it and the
unlabelled units by the average of the fold models.

### Natural experiments with an ML-measured outcome

[`did_with_predicted_outcome`](@ref) and [`rd_with_predicted_outcome`](@ref) bring
the correction to difference-in-differences (TWFE, Callaway–Sant'Anna, two-period
DR-DiD) and sharp regression discontinuity designs.

- **Design-based mode (default).** The DiD or RD estimator is applied to the DSL
  pseudo-outcome, with `ĝ` cross-fitted **by unit** (or by `cluster`), so the model
  predicting a unit's outcome never saw that unit's labels, and with cohort, period
  and treatment indicators (side of the cutoff and running variable for RD) among its
  inputs. DiD and local-polynomial estimators are linear in the outcome given the
  design, so the estimand is the one defined with the true outcome and the
  estimators' own cluster-robust / robust bias-corrected inference, computed on the
  pseudo-outcome, accounts for the labelling. No assumption is made on how
  prediction errors vary between groups, periods or sides of the cutoff; the
  identifying assumptions of the design (parallel trends, continuity) are those of
  the **true** outcome.
- **Labelling requirement.** Every group × period cell (both sides of the cutoff)
  must have a positive labelling probability. If labels are missing in some cells and
  no `label_prob` is given, the functions stop with an explanation instead of
  silently assuming the error is stable.
- **Stable-error mode** (`assume_stable_error = true`) for labels available only in
  some periods (e.g. before treatment) or on one side of the cutoff: the linear
  measurement model `E[f | Y, group, features] = a_g + bY + γ'features` is assumed to
  be the same in all periods (on both sides), estimated on the labelled rows and
  inverted; a cluster bootstrap refits it. This corrects attenuation and group
  shifts, but by construction it cannot detect a treatment-induced change in how the
  measure errs (see the Monte Carlo below).
- **Held-out discipline.** A classifier fine-tuned on units of the study must not
  score those units: `measure_training` flags the rows used for training and every
  unit (cluster) containing one is excluded. Alternatively, pass the raw inputs as
  `features` with `prediction = nothing`: the measure is then learned from the labels
  by cross-fitting by unit (cross-prediction), and the naive comparison uses the
  out-of-fold measure.
- **Diagnostics.** Each result carries `bias_test`, the design estimator applied to
  the pseudo-outcome minus the prediction: it estimates the bias of the naive
  estimate, with a valid standard error. The corrected estimate does not depend on
  this test, and a non-rejection is not evidence that the naive estimate is unbiased.

### Measurement-error diagnostics and corrections

[`differential_error_test`](@ref) compares the mean prediction error across treatment
arms, periods, group × period cells or sides of a cutoff on the labelled rows
(weighted by `1/π`, cluster-robust Wald test). Differences in mean error across the
cells that a design contrasts are what bias the naive estimate; note that an
attenuated but otherwise perfect measure (`f = a + bY`, `b < 1`) is differential in
this sense whenever the treatment moves `Y`.

[`regression_calibration`](@ref) corrects the attenuation of a regression coefficient
when a covariate is observed with classical (non-differential) error on every row and
without error on a validation subsample (Carroll et al. 2006): the calibration model
`E[X | X*, Z]` replaces `X`, and the variance comes from the stacked estimating
equations. Its assumptions (surrogacy, linear calibration) are stronger than DSL's;
with an ML-predicted covariate whose error may be differential, use
`dsl_regression(...; predicted_vars = [:x])` instead (Fong & Tyler 2021 discuss the
problem of ML predictions as covariates).

### Which method for ML-measured variables?

| Situation | Function |
|:--|:--|
| Regression / logit / FE with ML-coded outcome or covariates, known labelling design | [`dsl_regression`](@ref) |
| Share of documents per category (by period) | [`dsl_proportions`](@ref) |
| Randomized experiment, ML-predicted outcome | [`ppi_ate`](@ref) (or `dsl_regression`) |
| Predictor must be trained on the labelled data | [`cross_ppi`](@ref), or `features` in the functions above |
| DiD / RD with an ML-coded outcome, labels in every cell | `did_…` / `rd_with_predicted_outcome` |
| Labels only before treatment / on one side | the same with `assume_stable_error = true` (strong assumption) |
| Covariate with classical error and a validation sample | [`regression_calibration`](@ref) |

### Validation and Monte Carlo

- **R `dsl` parity** (`test/validation/ml/dsl_reference.R`): on the package's example
  datasets, `dsl` 0.1.0 is run with a deterministic internal model (`sl_method =
  "lm"`) and one sample split; the cross-fitting folds it draws are reconstructed and
  reused by DrSnow with `OLSLearner()`. Coefficients, standard errors and covariance
  matrices agree to 2e-5 (relative) for the linear model with and without clustering,
  a linear and a logistic model with an ML-measured covariate, linear and logistic
  models with unequal labelling probabilities and the one-way fixed-effects model.
  In the two-way fixed-effects model, dsl's numerical optimizer stops about 1e-3
  (0.002 standard errors) from the root; DrSnow matches the exact within estimator
  on the same pseudo-outcome (R `lm`) to 1e-9.
- **Closed forms**: least squares on the pseudo-outcome with HC0/CR0 sandwich; the
  LSDV sandwich for fixed effects; the logistic moment and sandwich; PPI with `λ = 1`
  equals [`ppi_ols`](@ref)'s point estimate; the PPI ATE is the difference in means
  of the pseudo-outcome and `λ̂` minimizes the estimated variance exactly; cross-PPI
  reproduces [`ppi_mean`](@ref) on the out-of-fold predictions; regression calibration
  reduces to OLS with HC0 variance without measurement error; the DiD and RD wrappers
  equal the design estimators applied to the pseudo-outcome, and the naive-minus-
  corrected difference equals the contrast estimate. Results are invariant to row
  order (with a fold column).
- **Monte Carlo** (`test/validation/ml/measurement_montecarlo.jl`; every design has
  differential error, so the naive plug-in is biased):

| Design (replications) | Estimator | Bias | SD | Mean SE | 95% coverage |
|:--|:--|--:|--:|--:|--:|
| RCT, n = 1000, ~25% labelled with probabilities depending on `x` and `d`; ATE = 1 (2000) | naive (LLM outcome as truth) | +0.301 | 0.062 | 0.063 | 0.002 |
| | `dsl_regression` (OLS recalibration) | 0.000 | 0.096 | 0.095 | 0.949 |
| | `ppi_ate` (difference in means) | +0.002 | 0.099 | 0.099 | 0.948 |
| | `ppi_ate` (Lin adjustment) | +0.002 | 0.095 | 0.094 | 0.948 |
| | labelled rows only (`lambda = 0`) | +0.012 | 0.195 | 0.195 | 0.949 |
| Mean, 300 labelled + 3000 unlabelled, predictor trained on the labels (2000) | `cross_ppi` (5 folds) | −0.002 | 0.054 | 0.054 | 0.945 |
| DiD, 200 units × 4 periods, 30% of unit-periods labelled, LLM error +0.6 when treated; ATT = 1 (1000) | naive TWFE | +0.496 | 0.148 | 0.146 | 0.068 |
| | corrected TWFE | −0.007 | 0.179 | 0.183 | 0.951 |
| | corrected TWFE, `learner = nothing` | −0.002 | 0.197 | 0.194 | 0.948 |
| | corrected Callaway–Sant'Anna (simple ATT) | −0.004 | 0.235 | 0.222 | 0.931 |
| same, 800 units (600) | corrected Callaway–Sant'Anna (simple ATT) | | 0.110 | 0.111 | 0.952 |
| DiD, labels only in periods 1–2, stable-error mode (200) | error shift after treatment (assumption violated) | +0.651 | 0.156 | 0.165 | 0.015 |
| | attenuated, stable measure (assumption holds) | −0.002 | 0.162 | 0.163 | 0.950 |
| Sharp RD, n = 2000, labels 50% within 0.3 of the cutoff and 10% elsewhere, error jump +0.4; effect = 1 (1000) | naive RD | +0.297 | 0.121 | 0.121 | 0.322 |
| | corrected RD | −0.004 | 0.134 | 0.130 | 0.949 |
| Classical error in a covariate, n = 1000, 20% validation; β = 0.8 (2000) | naive | −0.356 | 0.029 | 0.029 | 0.000 |
| | `regression_calibration` | +0.003 | 0.066 | 0.066 | 0.956 |

The corrected Callaway–Sant'Anna intervals under-cover slightly with 200 units (the
cross-fitted calibration adds a finite-sample variance component the influence
function ignores, and each group-time cell uses a single base period); they are
accurate with 800 units. The stable-error rows show what the assumption buys and what
it costs: the correction is exact when the measurement model is stable and has no
protection against a treatment-induced change in the error.

The tests repeat these checks with `mc_reps(full, fast)` replications, plus the size
of the bias test and of [`differential_error_test`](@ref) under non-differential
error and its power under differential error.

### References (ML-measured variables)

- Angelopoulos, A. N., Bates, S., Fannjiang, C., Jordan, M. I. and Zrnic, T. (2023).
  Prediction-powered inference. *Science* 382(6671), 669–674.
- Angelopoulos, A. N., Duchi, J. C. and Zrnic, T. (2023). PPI++: Efficient
  prediction-powered inference. arXiv:2311.01453.
- Carroll, R. J., Ruppert, D., Stefanski, L. A. and Crainiceanu, C. M. (2006).
  *Measurement Error in Nonlinear Models: A Modern Perspective* (2nd ed.). Chapman &
  Hall/CRC.
- Egami, N., Hinck, M., Stewart, B. M. and Wei, H. (2023). Using imperfect surrogates
  for downstream inference: Design-based supervised learning for social science
  applications of large language models. *Advances in Neural Information Processing
  Systems* 36.
- Egami, N., Hinck, M., Stewart, B. M. and Wei, H. (2024). Using large language model
  annotations for the social sciences: A general framework of using predicted
  variables in downstream analyses. Working paper.
- Fong, C. and Tyler, M. (2021). Machine learning predictions as regression
  covariates. *Political Analysis* 29(4), 467–484.
- Lin, W. (2013). Agnostic notes on regression adjustments to experimental data:
  Reexamining Freedman's critique. *Annals of Applied Statistics* 7(1), 295–318.
- Wang, S., McCormick, T. H. and Leek, J. T. (2020). Methods for correcting inference
  based on outcomes predicted by machine learning. *PNAS* 117(48), 30266–30275.
- Zrnic, T. and Candès, E. J. (2024). Cross-prediction-powered inference. *PNAS*
  121(15), e2322083121.

### API (ML-measured variables)



Full references are listed at the end of the [main causal ML guide](ml.md).

The functions and types described on this page are documented in the [API reference](reference/ml.md).
