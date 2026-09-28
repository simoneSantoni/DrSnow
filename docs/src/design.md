# Experimental design and power

```@meta
CurrentModule = DrSnow
```

The design area covers the decisions made *before* data are collected: how many units
(or clusters, periods, observations near a cutoff) are needed, which units to pair or
block before randomizing, and how the analysis will use the same information. It has
four parts:

1. **Analytic power, minimum detectable effects and sample sizes** for the standard
   designs: two-sample comparisons of means and proportions, blocked and
   cluster-randomized experiments, repeated measurements / difference-in-differences,
   encouragement designs (IV) and regression discontinuity.
2. **Blocking and matched pairs on predicted outcomes**: a prognostic score fitted
   with any [`NuisanceLearner`](@ref), blocks or optimal matched pairs formed on it, an
   `AssignmentMechanism` shared with randomization inference, and an analysis helper
   that reuses the same scores.
3. **Simulation-based diagnosis** of any declared design (data-generating process,
   assignment, estimand, DrSnow estimators): power, bias, RMSE, coverage, type-S and
   type-M errors, with Monte Carlo standard errors, over grids of design parameters.
4. **Surrogate-model optimization**: the cheapest design that reaches a target power,
   found with a small number of simulations.

## What machine learning buys at the design stage

Machine-learning predictions of the outcome from baseline covariates improve
**precision, not identification**. In a randomized experiment the treatment effect is
identified by randomization alone; blocking on a prognostic score or adjusting for it
removes outcome variance that the covariates explain, so the same experiment has
smaller standard errors (equivalently, a larger effective sample). Nothing about what
is estimated changes, and a poor model costs little: with random assignment the
estimators below remain unbiased (or consistent) for the average effect whatever the
quality of the predictions. Two rules keep it that way:

- the score may use only **pre-treatment** information, and it must be fixed before
  assignment (fit it on pilot, historical or baseline data);
- the analysis must respect the design: estimate within the blocks that were used and
  use the design's assignment mechanism for randomization inference.

The gain is governed by the *out-of-sample* R² of the score. With a score that
explains a share `R²` of the outcome variance, regression adjustment on the score
reduces the variance of the estimated effect by the factor `1 - R²`; tight blocks or
pairs on the score achieve about the same reduction in the design itself, and help
most in small samples, where they also balance the realized assignment. Bai (2022)
shows that, among stratified designs that treat each unit with probability one half,
the precision-maximizing design pairs units on an index of the covariates; a
well-predicting score is a feasible stand-in. [`prognostic_score`](@ref) reports the
cross-fitted R², and [`variance_reduction`](@ref) translates it into an expected
variance ratio for a given blocked design. Feed the same R² into the analytic
calculators (`r2` in [`power_means`](@ref), [`power_blocked`](@ref),
[`power_cluster`](@ref), [`power_iv`](@ref)) to size the experiment.

## Analytic power

Every calculator solves for whichever of `power`, the effect, or a size parameter is
left as `nothing`:

```julia
power_means(effect=0.3, n=200)                  # power
power_means(n=200, power=0.8)                   # minimum detectable effect
power_means(effect=0.3, power=0.8, r2=0.4)      # total sample size with adjustment
```

Power is the exact power of the level-`alpha` t test (noncentral t, both rejection
tails, as in R's `pwr` and `PowerUpR`) for an estimator with standard error `se` and
the design's degrees of freedom; `distribution = :normal` gives the z test. Solved
sizes are continuous (round them up; the printed result shows the rounded value). The
result also reports Bloom's (1995) multiplier approximation of the MDE,
`(t₁₋α/₂ + t_power) × se`.

| Function | Design | Standard error / notes | Reference implementation |
|:--|:--|:--|:--|
| [`power_means`](@ref) | two arms, completely randomized, optional ANCOVA (`r2`) | `sd √((1-R²)/(p(1-p)n))`, t(`n-2-k`) | `pwr.t.test`, `pwr.t2n.test` |
| [`power_proportions`](@ref) | binary outcome | arcsine `h` or pooled z test | `pwr.2p(2n).test`, `power.prop.test(strict=TRUE)` |
| [`power_blocked`](@ref) | blocked / matched pairs, block fixed effects | `sd √((1-R²)/(p(1-p)Jm))`, t(`J(m-1)-k-1`) | `PowerUpR::bira2c1` |
| [`power_cluster`](@ref) | cluster randomization, ICC, unequal sizes, covariates at both levels | Bloom (2006) with the Eldridge et al. (2006) design effect | `PowerUpR::cra2r2` |
| [`power_did`](@ref) | `m` pre / `r` post rounds: POST, DiD, ANCOVA | McKenzie (2012) for equicorrelated periods, any serial correlation matrix | closed forms, simulation |
| [`power_iv`](@ref) | encouragement design, compliance rate | ITT test; MDE of the LATE = ITT MDE / compliance | simulation with `late_2sls` |
| [`power_rd`](@ref) | sharp regression discontinuity from pilot data or variance constants | `rdpower` formula, robust bias-corrected test | `rdpower` |

Notes on the designs:

- **Clusters.** Power is bounded as clusters grow: adding individuals per cluster
  cannot remove the between-cluster variance `ρ(1-R²₂)`. Unequal cluster sizes enter
  through their coefficient of variation (`cv` or the planned `cluster_sizes`).
- **Repeated measurements.** With few baseline rounds and low autocorrelation,
  ANCOVA is much more efficient than DiD (McKenzie 2012). The equicorrelation
  assumption can misstate power when correlations decay (Burlig, Preonas & Woerman
  2020): pass the full correlation matrix, e.g. AR(1), with `corr`.
- **IV.** The calculation is the normal-theory approximation of the reduced-form test;
  with low compliance or strong heterogeneity, check by simulation.
- **RD.** With pilot data, `power_rd` runs [`rd_estimate`](@ref) and scales its
  variance to a new sample size and bandwidth, exactly as `rdpower` does; the new
  sample is assumed to have the pilot's distribution of the running variable.

## Blocking and matched pairs on predicted outcomes

```julia
using DrSnow, StableRNGs
# 1. prognostic score: fitted on pilot data, predicted for the experimental sample
ps = prognostic_score(pilot, :earnings; covariates=[:age, :educ, :earn_lag],
                      learner=LassoLearner(), target=sample, id=:pid,
                      rng=StableRNG(1))
# 2. optimal matched pairs on the score (or blocks of 4: method=:blocks, block_size=4)
bd = block_design(sample, ps; id=:pid)
variance_reduction(bd)                      # expected variance ratio vs complete RA
# 3. randomize with the design's mechanism
expt = assign_treatment(bd, sample; rng=StableRNG(2))
# ... collect outcomes ...
# 4. analyse with the same blocks and the same score
experiment_estimate(expt, :earnings, :treated, bd; method=:lin)
randomization_test(expt, :earnings, :treated; mechanism=bd.mechanism, id=:pid)
```

**Prognostic scores.** [`prognostic_score`](@ref) fits any learner (OLS, ridge,
lasso, random forests, any MLJ model) to pilot, historical or control-only data and
predicts the experimental units, or cross-fits it on the experimental sample itself
when the outcome is a pre-treatment measurement. The out-of-sample R² is computed by
cross-fitting in the training data. Aufenanger (2017) and Gui & Kim (2025) study
stratification on machine-learning (and LLM) predictions of the outcome.

**Forming blocks.** [`block_design`](@ref) forms

- *matched pairs* (`method = :pairs`): on a scalar score, pairing adjacent units in
  sorted order minimizes the total within-pair distance; with the Mahalanobis distance
  of several covariates the minimum-distance non-bipartite matching is found exactly by
  Edmonds' blossom algorithm (Greevy et al. 2004), or greedily (`algorithm = :greedy`,
  which reproduces `blockTools`' `optGreedy`);
- *blocks* (`method = :blocks`): consecutive quantile groups of the sorted score of a
  chosen size, or greedy multivariate blocks.

Units are identified by an `id` column, so the design does not depend on the row
order of the data and can be joined back by key. The returned [`BlockingDesign`](@ref)
holds a `MatchedPairsRandomization` or `StratifiedRandomization`, used by
[`assign_treatment`](@ref), by `randomization_test` / `ri_regression` (`mechanism`),
and by [`experiment_estimate`](@ref).

**Analysis with the same scores.** [`experiment_estimate`](@ref) estimates the
average effect with

- `:difference`: within-block differences in means with the Neyman variance, or the
  matched-pair variance of Imai, King & Nall (2009), with a t reference on `J − 1`
  degrees of freedom for `J` pairs. When the pairs are formed on covariates that
  predict the outcome, this test is conservative (Bai, Romano & Shaikh 2022);
- `:block_fe`: block fixed effects plus covariates, heteroskedasticity-robust;
- `:lin`: Lin's (2013) interacted adjustment with block fixed effects;
- `:dml`: cross-fitted AIPW with the *known* assignment probabilities (no propensity
  model), outcome models from any learner.

Given the design, the regression and DML methods add the design's prognostic score as
a covariate: the same predictions are used to block and to adjust. `:difference`,
`:lin` without blocks and `:block_fe` reproduce `estimatr` exactly.

**How much blocking buys.** [`variance_reduction`](@ref) computes the exact design
variance of the difference in means of the score under the blocked design and under
complete randomization, and, using the score's R², the expected variance ratio for the
outcome. Blocking and adjustment are complements: with a good score, pairs and Lin
adjustment on the same score reach about the same precision, and blocking guarantees
balance on the score in the realized assignment.

## Simulation-based power and design diagnosis

Analytic formulas exist only for simple designs and estimators. For anything else —
staggered DiD, RD with a specific bandwidth rule, clustered IV, estimators with
covariate selection — declare the design and simulate it (Blair, Cooper, Coppock &
Humphreys 2019):

```julia
using DrSnow, DataFrames, StableRNGs, Statistics
pop(rng, p) = (x = randn(rng, p.n); u = x .+ randn(rng, p.n);
               DataFrame(x=x, Y0=u, Y1=u .+ p.effect))
d = declare_design(pop; params=(n=200, effect=0.25),
                   assignment=(data, p) -> CompleteRandomization(p.n, p.n ÷ 2),
                   estimand=(data, p) -> mean(data.Y1 .- data.Y0),
                   estimators=["DiM" => (data, p) -> experiment_estimate(data, :Y, :Z),
                               "Lin" => (data, p) -> experiment_estimate(
                                   data, :Y, :Z; method=:lin, covariates=[:x])])
dx = diagnose_design(d; sims=2000, rng=StableRNG(1))
g = diagnose_grid(d, (n=100:100:600,); sims=1000, rng=StableRNG(2))
```

One simulation draws the population (with potential outcomes), computes the
estimand, draws an assignment from the mechanism, reveals the outcomes and runs every
estimator; any DrSnow estimator returning a `CausalEstimate` can be used. Natural
experiments are declared without an assignment step, the population function
producing the observed data (e.g. a panel analysed with `did_twfe`).

Diagnosands ([`DesignDiagnosis`](@ref)): mean estimate and estimand, bias, standard
deviation of the estimates, RMSE, mean standard error, power, coverage, the type-S
rate (share of significant estimates with the wrong sign) and type-M error
(exaggeration of significant estimates; Gelman & Carlin 2014). Monte Carlo standard
errors come from bootstrapping the simulations. Every simulation has its own
pre-drawn seed, so results are reproducible and identical with or without threads.

## Surrogate-model design optimization

[`optimize_design`](@ref) finds the cheapest design that reaches a target power
without simulating a full grid (Zimmer & Debelak 2025; Zimmer, Henninger & Debelak
2024). A few designs are simulated, a smooth monotone model of power is fitted —
by default a probit regression of rejections on `√n`, the shape of the normal-theory
power curve `Φ(a + b√n)`; alternatively logistic, isotonic, or user-supplied features
such as the standardized noncentrality — and the cheapest design predicted to reach
the target is simulated next, together with other designs near the power boundary,
until the budget is used. A final verification run at the chosen design is
recommended (`verify_sims`).

```julia
opt = optimize_design(d; space=(n=50:10:1000,), target_power=0.8, max_sims=4000,
                      verify_sims=2000, rng=StableRNG(3))
# two parameters with a cost: clusters are 10 times as expensive as individuals
optimize_design(dc; space=(n_clusters=10:2:80, cluster_size=[5, 10, 20, 40]),
                cost=p -> 10p.n_clusters + p.n_clusters * p.cluster_size,
                transform=p -> [sqrt(p.n_clusters / (0.1 + 0.9 / p.cluster_size))])
```

## Plots

With a Makie backend loaded, [`plot_power_curve`](@ref) draws simulated power over a
grid (with Monte Carlo intervals), a surrogate optimization (evaluated designs, fitted
curve, target and chosen design) or a vector of analytic results, and
[`plot_blocks`](@ref) shows the prognostic score of the units in each block.

## Guidance

- Size the experiment with the analytic calculator of the closest design, using a
  conservative (out-of-sample) R²; then simulate the planned estimator on the planned
  design, which also checks coverage and type-S/M errors at the planned sample size.
- Block or pair on a prognostic score when the sample is small or moderate and a
  good pre-treatment predictor exists; otherwise complete randomization with Lin
  adjustment is nearly as efficient in large samples.
- Pairs maximize balance but leave no within-pair variance estimate; the matched-pair
  variance is conservative. Blocks of four or more allow the blocked Neyman variance.
- Always analyse within the design's blocks, and use the design's mechanism for
  randomization inference.
- Power from a surrogate is an estimate: report the verification run.

## Validation

Reference values and generating scripts are in `test/validation/design/`
(`make_reference_power.R`, `make_reference_experiments.R`); the tests read the stored
CSV files.

| Component | Reference | Agreement |
|:--|:--|:--|
| `power_means`, `power_proportions` | `pwr` 1.3.0, `stats::power.prop.test` | power to 1e-12; solved n / MDE to `uniroot`'s tolerance |
| `power_cluster`, `power_blocked` | `PowerUpR` 1.1.0 (`cra2r2`, `bira2c1`) | power and multiplier MDE to 1e-12; required clusters equal |
| `power_did` | McKenzie (2012) closed forms; AR(1) Monte Carlo | exact; simulated SD within Monte Carlo error |
| `power_iv` | Monte Carlo with `late_2sls` | within Monte Carlo error |
| `power_rd` | `rdpower` 3.0 / `rdrobust` 4.0.0 (Senate data) | power and SEs to 1e-8 relative |
| `experiment_estimate` | `estimatr` 2.0.0 (`difference_in_means`, `lm_lin`, `lm_robust` FE) | estimates, SEs and dof to 1e-10 |
| greedy pairs | `blockTools` 0.6.6 `optGreedy` | identical pairs |
| optimal pairs | exhaustive search (150 random instances) | identical total distance |
| `diagnose_design` | `DeclareDesign` 1.1.1 (20 000 simulations) | all diagnosands within Monte Carlo error |

## References

- Athey, S., & Imbens, G. W. (2017). The econometrics of randomized experiments. In
  A. V. Banerjee & E. Duflo (Eds.), *Handbook of Economic Field Experiments* (Vol. 1,
  pp. 73–140). North-Holland.
- Aufenanger, T. (2017). *Machine learning to improve experimental design* (FAU
  Discussion Papers in Economics No. 16/2017). Friedrich-Alexander University
  Erlangen-Nuremberg.
- Bai, Y. (2022). Optimality of matched-pair designs in randomized controlled trials.
  *American Economic Review*, 112(12), 3911–3940.
- Bai, Y., Romano, J. P., & Shaikh, A. M. (2022). Inference in experiments with matched
  pairs. *Journal of the American Statistical Association*, 117(540), 1726–1737.
- Blair, G., Cooper, J., Coppock, A., & Humphreys, M. (2019). Declaring and diagnosing
  research designs. *American Political Science Review*, 113(3), 838–859.
- Bloom, H. S. (1995). Minimum detectable effects: A simple way to report the
  statistical power of experimental designs. *Evaluation Review*, 19(5), 547–556.
- Bloom, H. S. (2006). *The core analytics of randomized experiments for social
  research* (MDRC Working Papers on Research Methodology). MDRC.
- Burlig, F., Preonas, L., & Woerman, M. (2020). Panel data and experimental design.
  *Journal of Development Economics*, 144, 102458.
- Cattaneo, M. D., Titiunik, R., & Vazquez-Bare, G. (2019). Power calculations for
  regression-discontinuity designs. *Stata Journal*, 19(1), 210–245.
- Dong, N., & Maynard, R. (2013). PowerUp!: A tool for calculating minimum detectable
  effect sizes and minimum required sample sizes for experimental and quasi-experimental
  design studies. *Journal of Research on Educational Effectiveness*, 6(1), 24–67.
- Duflo, E., Glennerster, R., & Kremer, M. (2007). Using randomization in development
  economics research: A toolkit. In T. P. Schultz & J. Strauss (Eds.), *Handbook of
  Development Economics* (Vol. 4, pp. 3895–3962). Elsevier.
- Eldridge, S. M., Ashby, D., & Kerry, S. (2006). Sample size for cluster randomized
  trials: Effect of coefficient of variation of cluster size and analysis method.
  *International Journal of Epidemiology*, 35(5), 1292–1300.
- Gelman, A., & Carlin, J. (2014). Beyond power calculations: Assessing type S (sign)
  and type M (magnitude) errors. *Perspectives on Psychological Science*, 9(6), 641–651.
- Greevy, R., Lu, B., Silber, J. H., & Rosenbaum, P. (2004). Optimal multivariate
  matching before randomization. *Biostatistics*, 5(2), 263–275.
- Gui, G., & Kim, S. (2025). *Leveraging LLMs to improve experimental design: A
  generative stratification approach* (arXiv:2509.25709). arXiv.
- Hansen, B. B. (2008). The prognostic analogue of the propensity score. *Biometrika*,
  95(2), 481–488.
- Imai, K., King, G., & Nall, C. (2009). The essential role of pair matching in
  cluster-randomized experiments, with application to the Mexican universal health
  insurance evaluation. *Statistical Science*, 24(1), 29–53.
- Lin, W. (2013). Agnostic notes on regression adjustments to experimental data:
  Reexamining Freedman's critique. *Annals of Applied Statistics*, 7(1), 295–318.
- McKenzie, D. (2012). Beyond baseline and follow-up: The case for more T in
  experiments. *Journal of Development Economics*, 99(2), 210–221.
- Morgan, K. L., & Rubin, D. B. (2012). Rerandomization to improve covariate balance in
  experiments. *Annals of Statistics*, 40(2), 1263–1282.
- Zimmer, F., & Debelak, R. (2025). Simulation-based design optimization for statistical
  power: Utilizing machine learning. *Psychological Methods*, 30(3), 513–536.
- Zimmer, F., Henninger, M., & Debelak, R. (2024). Sample size planning for complex
  study designs: A tutorial for the mlpwr package. *Behavior Research Methods*, 56(5),
  5246–5263.

The functions and types described on this page are documented in the [API reference](reference/design.md).
