# Heterogeneous Effects and Policy Learning

```@meta
CurrentModule = DrSnow
```

## Heterogeneous treatment effects

### Choosing a method

| Question | Tool | Inference |
|:--|:--|:--|
| Effects in pre-specified groups, effect modification by a few variables | [`subgroup_effects`](@ref), [`interaction_effects`](@ref) | regression / AIPW scores, robust or cluster SEs; multiplicity adjustment |
| Is there heterogeneity, and along which variables? (experiment) | [`generic_ml`](@ref) (BLP, GATES, CLAN) | valid for any ML proxy (sample splitting) |
| Is there heterogeneity? (observational or experimental) | [`causal_forest`](@ref) + [`test_calibration`](@ref), [`best_linear_projection`](@ref), [`rank_average_treatment_effect`](@ref) | AIPW scores, HC3 / bootstrap |
| CATE at a point `x`, with a confidence interval | [`causal_forest`](@ref) + [`predict_interval`](@ref) | pointwise, asymptotically normal (little bags) |
| Best CATE predictions for targeting | [`causal_forest`](@ref), [`r_learner`](@ref), [`x_learner`](@ref), [`cate_dr_learner`](@ref) | none per point; evaluate the ranking with RATE on held-out data |
| Linear summary of the CATE | [`best_linear_projection`](@ref), [`cate_projection`](@ref) | valid without a linear CATE |
| Heterogeneity in an IV (LATE) design | [`instrumental_forest`](@ref) | pointwise intervals; AIPW average LATE |
| Treatment rules | [`policy_tree`](@ref), [`double_robust_scores`](@ref), [`policy_value`](@ref) | DR value estimates |

Pre-specify subgroups and moderators when the goal is confirmatory: a data-driven
search followed by tests on the same data overstates significance. The forest,
generic-ML and RATE tools are designed for exploration with valid inference, but
their targets are features of the estimated CATE (its projection, its ranking),
not the CATE function itself. Meta-learners give point predictions only.

### Generalized random forests

[`causal_forest`](@ref), [`instrumental_forest`](@ref) and
[`regression_forest`](@ref) implement the generalized random forests of Athey,
Tibshirani & Wager (2019) as in the R package grf 2.6.1, whose C++ training and
prediction core is ported line by line (only the random-number streams differ):

- **Local centering.** Causal forests work with `Ỹ = Y - Ŷ(X)` and
  `W̃ = W - Ŵ(X)`, where `Ŷ`, `Ŵ` are out-of-bag regression-forest predictions (with
  `max(50, num_trees ÷ 4)` trees), supplied values, or any
  [`NuisanceLearner`](@ref) cross-fitted in `n_folds` folds (`y_hat`, `w_hat`).
  Instrumental forests also center the instrument (`z_hat`).
- **Gradient-based ("local moment") splits.** Each node solves the local
  estimating equation (for a causal forest the residual-on-residual regression
  `τ̂ = Σ(W̃ - W̄)(Ỹ - Ȳ) / Σ(W̃ - W̄)²`), relabels its units with the pseudo-outcomes
  `ρᵢ = (Z̃ᵢ - Z̄)(Ỹᵢ - Ȳ - τ̂(W̃ᵢ - W̄))` (`Z̃ = W̃` for a causal forest), and chooses
  the CART split that best separates them. With `stabilize_splits = true` each child
  must contain `min_node_size` units with `Z̃` below and above the parent mean
  (treated and controls) and an `alpha` share of the parent's `Z̃` variation; an
  `imbalance_penalty` discourages edge splits. `mtry` candidate variables are drawn
  per split (a Poisson number with mean `mtry`).
- **Honesty and subsampling.** Every tree is grown on a subsample of
  `sample_fraction · n` units (whole clusters with `cluster`, optionally the same
  number per cluster with `equalize_cluster_weights`); one part
  (`honesty_fraction`) chooses the splits, the other populates the leaves, and
  splits with an empty honest child are pruned (`honesty_prune_leaves`).
- **Estimation.** Forest weights `αᵢ(x)` (the share of trees in which unit `i`
  shares a leaf with `x`, averaged over trees) define the local moment
  `Σ αᵢ(x) ψ(Oᵢ; τ, μ) = 0`; for a causal forest
  `τ̂(x) = Cov_α(W̃, Ỹ) / Var_α(W̃)`. Predictions for training units are
  out-of-bag (`predict(f)`).
- **Variance.** Trees are grown in groups of `ci_group_size = 2` sharing a
  half-sample; the between-group variance of the per-tree moments, debiased for
  within-group Monte Carlo noise with grf's objective-Bayes correction, estimates
  `Var(τ̂(x))` (bootstrap of little bags). [`predict_interval`](@ref) returns
  pointwise normal intervals. They are asymptotically valid under honesty,
  subsampling, overlap and smoothness (Wager & Athey 2018), but undercover where the
  CATE is steep relative to the sample size and the number of covariates (see the
  Monte Carlo results below; grf behaves identically).
- **Sample weights, clusters, missing covariates.** `weights` enter the splits and
  the leaf moments; `cluster` makes subsampling and all post-estimation standard
  errors cluster-level; `missing` covariate values are sent to the better side at
  each split (grf's MIA rule). Rows are sorted into a canonical order before
  sampling, so results do not depend on row order, and one seed per group of trees
  is drawn from `rng`, so results are identical with any number of threads.
- **Tuning** (`tune_parameters`): small forests at random draws of
  `sample_fraction`, `mtry`, `min_node_size`, `honesty_fraction`,
  `honesty_prune_leaves`, `alpha`, `imbalance_penalty` (grf's parameter
  distributions) are compared by their debiased out-of-bag error; the best draw is
  kept if, refitted with more trees, it beats the defaults. grf additionally smooths
  the error surface by kriging, which is omitted.

### Average effects, calibration and heterogeneity

All post-estimation functions mirror grf and use the forest's doubly robust scores
[`get_scores`](@ref): `Γᵢ = τ̂(Xᵢ) + γᵢ(Yᵢ - Ŷᵢ - τ̂(Xᵢ)(Wᵢ - Ŵᵢ))`, with
`γᵢ = (Wᵢ - Ŵᵢ)/(Ŵᵢ(1 - Ŵᵢ))` for a binary treatment (so the mean of `Γ` is the AIPW
ATE), `γ` from a variance forest for a continuous treatment, and
`γᵢ = (Zᵢ - Ẑᵢ)/(Ẑᵢ(1 - Ẑᵢ))/Δ̂(Xᵢ)` (compliance score `Δ̂`) for an instrumental
forest.

- [`average_treatment_effect`](@ref): ATE (`target = :all`), ATT (`:treated`), ATC
  (`:control`) and the overlap-weighted effect (`:overlap`, well defined when
  propensities approach 0 or 1), with cluster-robust standard errors; `subset`
  gives conditional average effects for a group. Estimated propensities near 0 or 1
  trigger a warning.
- [`best_linear_projection`](@ref) (or [`cate_projection`](@ref) on a forest):
  regression of `Γ` on covariates with HC3 cluster-robust standard errors
  (as `sandwich::vcovCL`).
- [`test_calibration`](@ref): regression of `Y - Ŷ` on `(W - Ŵ)τ̄` and
  `(W - Ŵ)(τ̂(X) - τ̄)`; a differential coefficient significantly above zero means
  the forest detects heterogeneity, and 1 means its predictions are well calibrated.
- [`rank_average_treatment_effect`](@ref): the RATE of Yadlowsky et al. (2025),
  AUTOC or Qini, with its TOC curve and half-sample bootstrap standard errors;
  with two priority rules the difference is tested in a paired bootstrap. Use
  priorities that were not fitted on the evaluation sample (e.g. a forest trained
  on another half) for a clean test.
- [`variable_importance`](@ref) and [`split_frequencies`](@ref): how often each
  covariate is used for splitting (a descriptive measure, not a test).
- [`double_robust_scores`](@ref) and [`policy_value`](@ref): arm-specific scores
  for [`policy_tree`](@ref) and doubly robust values of a treatment rule.

Plots: [`plot_cate`](@ref) (out-of-bag CATEs with pointwise intervals against a
covariate), [`plot_variable_importance`](@ref) and [`plot_rate`](@ref) (TOC curve).

```julia
cf = causal_forest(df, :y, :d; covariates=[:x1, :x2, :x3], cluster=:school,
                   rng=StableRNG(1))
average_treatment_effect(cf; target=:treated)
test_calibration(cf)
best_linear_projection(cf, [:x1, :x2])
predict_interval(cf, newdata; level=0.9)
```

### DR-learner

[`cate_dr_learner`](@ref) cross-fits outcome regressions and the [`cate_dr_learner`](@ref) cross-fits outcome regressions and the
propensity score, forms doubly-robust pseudo-outcomes and regresses them on effect
modifiers with any learner (Kennedy 2023). `predict` evaluates the fitted
CATE at new points. Generic learners do not deliver pointwise confidence intervals;
[`cate_projection`](@ref) provides valid inference on the best linear projection of
the CATE on a basis (Semenova & Chernozhukov 2021).

### Meta-learners

The S-, T- and X-learners (Künzel et al. 2019) and the R-learner (Nie & Wager 2021)
combine any [`NuisanceLearner`](@ref) (including [`ForestLearner`](@ref) and MLJ
models through [`MLJLearner`](@ref)):

| Learner | Construction | Notes |
|:--|:--|:--|
| [`s_learner`](@ref) | `μ̂(x, w)` on `(X, W)`; `τ̂ = μ̂(x,1) - μ̂(x,0)` | regularization can shrink `τ̂` to zero |
| [`t_learner`](@ref) | separate `μ̂₁`, `μ̂₀`; `τ̂ = μ̂₁ - μ̂₀` | spurious heterogeneity with unbalanced arms |
| [`x_learner`](@ref) | impute individual effects from the other arm's fit, regress them per arm, combine with `ê(x)` | good with unbalanced arms |
| [`r_learner`](@ref) | minimize the R-loss `Σ((Y - m̂) - τ(V)(W - ê))²` with cross-fitted `m̂`, `ê` | Neyman-orthogonal (quasi-oracle) |

Each returns a [`MetaLearner`](@ref) with in-sample (`cate`) and cross-fitted
(`cate_oof`) predictions; `predict(m, newdata)` refits on all data with the stored
seeds. They give point predictions only: inference comes from causal forests
(pointwise), from doubly robust scores ([`best_linear_projection`](@ref),
[`cate_projection`](@ref), [`generic_ml`](@ref) GATES), or, heuristically, from
[`metalearner_bootstrap`](@ref) (a pairs bootstrap of the whole learner, whose
validity is not guaranteed for regularized or tree-based learners).

### Generic ML inference

For randomized experiments with known assignment
probabilities, [`generic_ml`](@ref) implements the BLP, GATES and CLAN analyses of
Chernozhukov, Demirer, Duflo & Fernández-Val: ML proxies are trained on an
auxiliary half, the analyses are run on the main half, and results are aggregated
over many random splits by the median. Reported intervals at level `1 - α` use
per-split intervals at level `1 - α/2`, and p-values are `min(1, 2 × median p)`;
both are conservative. The targets are features of the CATE *as captured by the
proxy*: failure to reject no heterogeneity ([`blp_test`](@ref)) may reflect a weak
proxy. [`gates`](@ref) and [`clan`](@ref) describe the most and least affected
groups. The fit criteria `Λ` and `Λ̄` help choose among proxy learners.

### Subgroups and interactions

[`subgroup_effects`](@ref) estimates the ATE within pre-specified subgroups: in an
experiment by a FixedEffectModels regression of the outcome on treatment × subgroup
indicators with subgroup fixed effects (`method = :regression`), under
unconfoundedness by regressing cross-fitted AIPW scores on the subgroup indicators
(`method = :aipw`), or from a causal forest's scores. [`heterogeneity_test`](@ref)
gives the Wald test that all subgroup effects are equal, and
`r.details.p_adjusted` holds Holm (family-wise error), Benjamini–Hochberg (false
discovery rate) or Bonferroni adjusted p-values for the per-subgroup tests.
[`interaction_effects`](@ref) regresses the outcome on the treatment, its
interactions with centered moderators, the moderators and covariates
(`:regression`), or AIPW scores on the moderators (`:aipw`, the best linear
projection of the CATE); `heterogeneity_test` is the joint test of no effect
modification. Standard errors are robust, or cluster-robust with `cluster`
(`vcov` accepts any FixedEffectModels estimator). Results are
[`HTEEstimate`](@ref)s.

## Policy learning

[`policy_tree`](@ref) finds the depth-1 or depth-2 decision tree maximizing the sum
of cross-fitted doubly-robust scores (Athey & Wager 2021), by exhaustive search as in
the R package policytree. The value of the learned rule is estimated out of fold
(each fold is assigned by a tree learned on the other folds) and compared with
treating everyone and no one. Depth-2 search costs `O(p² n²)`; `split_step`
coarsens the root split points for large samples.


Full references are listed at the end of the [main causal ML guide](ml.md).

The functions and types described on this page are documented in the [API reference](reference/ml.md).
