# Randomization inference

```@meta
CurrentModule = DrSnow
```

Randomization inference (RI) uses the known assignment mechanism of an experiment
(or an assumed one, in a natural experiment) as the only source of randomness. It
needs no model for the outcomes and no large-sample approximation. DrSnow
implements Fisher randomization tests, confidence intervals obtained by inverting
them, randomization inference for regression coefficients (Young 2019), covariate
balance tests and multiplicity adjustments. Everything is built on the
[`AssignmentMechanism`](@ref) abstraction shared with the interference area.

## Concepts

**Sharp versus weak nulls.** A *sharp* null specifies every unit's missing
potential outcome. Examples are *no effect for any unit*, `Yᵢ(1) = Yᵢ(0)`, or a
constant additive effect, `Yᵢ(1) = Yᵢ(0) + τ₀`. Under a sharp null the outcome
that would have been observed under any other assignment is known. The
distribution of any test statistic over re-randomized assignments is then known
exactly, and the Fisher randomization test is exact in finite samples, whatever
the statistic.

A *weak* null, such as *the average effect is zero*, does not pin down the missing
outcomes, and a Fisher test of it is not exact in general. With heterogeneous
effects the difference-in-means test can over-reject the weak null badly when arm
sizes and variances differ. Studentized statistics restore asymptotic validity for
the weak null while keeping exactness for the sharp null (Wu & Ding 2021; Zhao &
Ding 2021). Use `statistic = :studentized` or `:lin_studentized` whenever the
interest is in average effects.

**What RI tests, and what it does not.** A randomization p-value measures how
unusual the observed statistic is among the assignments the design could have
produced. It is valid only for the mechanism supplied. Using the wrong mechanism,
for example ignoring blocking, clustering or rerandomization, gives a wrong
reference distribution. A large p-value is not evidence that the null holds.

**Exact versus Monte Carlo.** When the number of possible assignments is at most
`nperm` (or with `exact = true`), every assignment is enumerated and p-values are
exact. Otherwise `nperm` assignments are drawn, the p-value is
`(1 + #{draws at least as extreme}) / (1 + nperm)`, which is itself a valid
p-value, and its Monte Carlo standard error is reported. Monte Carlo draws use
per-chunk seeds derived from `rng`, so results are reproducible and identical for
any number of threads (`threaded = true`).

## Assignment mechanisms

| Mechanism | Design |
|:--|:--|
| [`CompleteRandomization`](@ref) | exactly `k` of `n` units treated |
| [`BernoulliAssignment`](@ref) | independent coin flips (unit-specific probabilities allowed) |
| [`StratifiedRandomization`](@ref) | complete randomization within strata |
| [`MatchedPairsRandomization`](@ref) | one unit of each pair treated |
| [`ClusterRandomization`](@ref) | whole clusters treated |
| [`BlockClusterRandomization`](@ref) | cluster randomization within blocks |
| [`Rerandomization`](@ref) | redraw until a balance criterion holds (Morgan & Rubin 2012) |
| [`CustomAssignment`](@ref) | any sampler (Monte Carlo only) |

The data-level functions build the mechanism from columns. The default is complete
randomization with the observed number treated; `strata = :col` gives stratified
randomization, `cluster = :col` cluster randomization, and both together blocked
cluster randomization. The observed treated counts are always used. Alternatively
pass `mechanism = m`, whose units are the rows of the data in order, or the rows
sorted by `id` when `id = :col` is given. When the design is given by columns the
units are put in a canonical order internally, so results do not depend on the
order of the data rows. The observed assignment is checked against the mechanism:
an assignment with probability zero is an error.

## Fisher randomization tests

```julia
using DrSnow, DataFrames, StableRNGs

r = randomization_test(df, :y, :treated; strata=:block, nperm=10_000,
                       rng=StableRNG(1))
r.pvalue                              # also DrSnow.pvalue(r) (StatsAPI)
v, w = randomization_distribution(r)   # reference distribution for plotting
```

Test statistics (`statistic =`):

| Statistic | Description |
|:--|:--|
| `:diff_means` | difference in means; stratum-size-weighted within-stratum differences in stratified designs |
| `:studentized` | difference in means over its Neyman standard error (cluster-level in cluster designs; paired variance in matched pairs) |
| `:rank_sum` | Wilcoxon rank-sum (difference in mean within-stratum ranks) |
| `:ks` | Kolmogorov–Smirnov (upper tail; sensitive to distributional differences) |
| `:lin` | Lin (2013) regression-adjusted difference using `covariates` |
| `:lin_studentized` | its HC2 / cluster-robust t-statistic |
| `f(y, z)` | any function of the outcome vector and a `BitVector` assignment |

Sharp nulls other than zero are set with `tau0`: a number for a constant
additive effect, or a column of unit-specific effects. The statistic is computed
on the adjusted outcomes `Y - τ₀Z`, which are the control potential outcomes under
the null.

## Confidence intervals and point estimates

[`ri_confint`](@ref) collects the constant effects `τ₀` that are not rejected. The
two-sided interval is equal-tailed, and one-sided bounds are available. For the
linear statistics (`:diff_means`, `:lin`) the p-value is a step function of `τ₀`
with known breakpoints, and the interval is computed exactly by a sorted sweep with
no grid. For other statistics the end points are found by bracketing and
bisection on a fixed reference set. The Hodges–Lehmann-type point estimate is the
`τ₀` at which the observed statistic equals the mean of its randomization
distribution. For the rank-sum statistic this is the classical Hodges–Lehmann
estimate, the median of treated-minus-control differences.

A constant-effect interval describes the set of constant shifts compatible with
the data. When effects are heterogeneous, prefer studentized statistics, whose
intervals have asymptotic coverage for the average effect.

## Regression coefficients (Young 2019)

[`ri_regression`](@ref) re-randomizes the treatment according to the design and
refits `outcome ~ treatment + covariates + fe(…)` with FixedEffectModels for each
draw. It reports:

- **randomization-c**: the p-value from the coefficient's randomization
  distribution;
- **randomization-t**: the p-value from the distribution of the robust or
  clustered t-statistic. Young recommends this one, because it is exact for the
  sharp null and asymptotically valid for the weak null;
- joint randomization-t tests of several treatment coefficients, using the Wald
  statistic with the full covariance;
- an omnibus test across outcomes;
- Westfall–Young adjusted p-values across all coefficients.

Several treatment columns (multi-arm designs) are permuted jointly across
assignment units within strata.

## Balance

[`ri_balance_test`](@ref) compares covariate mean differences with their exact
randomization distribution. The omnibus statistic is a Mahalanobis distance or the
maximum absolute standardized difference. Per-covariate p-values come with
Westfall–Young adjustment. A small p-value flags imbalance that is unusual under
the stated mechanism. A large one does not establish that assignment followed the
mechanism, nor anything about unobserved characteristics.

## Multiple outcomes

[`ri_multiple_testing`](@ref) tests several outcomes on a common set of assignments
and reports Westfall–Young step-down p-values (`:minp` or `:maxt`; with studentized
statistics `:maxt` is the randomization analogue of Romano–Wolf). These control the
family-wise error rate using the joint randomization distribution, so they are
less conservative than Holm when outcomes are correlated. Holm
([`holm_adjust`](@ref)) and Benjamini–Hochberg ([`bh_adjust`](@ref), false
discovery rate) adjustments of plain p-values are also available.
[`westfall_young_adjust`](@ref) works on any matrix of joint draws.

## Guidance

- Always describe the actual design: strata, clusters, pairs and rerandomization
  criteria all change the reference distribution.
- Pick the statistic before seeing the results. Rank statistics are robust to
  outliers; studentized statistics protect weak-null inference.
- Report the number of draws, or state that inference is exact. Use enough draws
  that the Monte Carlo standard error is small relative to the distance between
  the p-value and the chosen significance level.
- Covariate adjustment through `:lin` changes the statistic, not the design, so the
  test stays exact.

## Validation

Exact p-values are checked against independent brute-force enumeration for every
statistic and design, against the R package ri2 (complete, blocked, unequal-block,
cluster and blocked-cluster designs, one- and two-sided, non-zero sharp nulls,
studentized), and against R's exact `wilcox.test` (p-value, Hodges–Lehmann
estimate and interval). Holm and Benjamini–Hochberg adjustments are checked
against `p.adjust`. Monte Carlo checks cover size under the sharp null, size of the
studentized test under a heterogeneous weak null, interval coverage, the
family-wise error rate of the Westfall–Young adjustment, and the size of regression
RI with few clusters. The reference script is `test/validation/ri/make_ri2_reference.R`.

## References

- Fisher, R. A. (1935). *The Design of Experiments*. Oliver & Boyd.
- Imbens, G. W., & Rubin, D. B. (2015). *Causal Inference for Statistics, Social, and
  Biomedical Sciences: An Introduction*, ch. 5. Cambridge University Press.
- Rosenbaum, P. R. (2002). *Observational Studies* (2nd ed.), ch. 2. Springer.
- Young, A. (2019). Channeling Fisher: Randomization tests and the statistical
  insignificance of seemingly significant experimental results. *Quarterly Journal of
  Economics*, 134(2), 557–598.
- Wu, J., & Ding, P. (2021). Randomization tests for weak null hypotheses in randomized
  experiments. *Journal of the American Statistical Association*, 116(536), 1898–1913.
- Zhao, A., & Ding, P. (2021). Covariate-adjusted Fisher randomization tests for the
  average treatment effect. *Journal of Econometrics*, 225(2), 278–294.
- Lin, W. (2013). Agnostic notes on regression adjustments to experimental data:
  Reexamining Freedman's critique. *Annals of Applied Statistics*, 7(1), 295–318.
- Morgan, K. L., & Rubin, D. B. (2012). Rerandomization to improve covariate balance in
  experiments. *Annals of Statistics*, 40(2), 1263–1282.
- Hodges, J. L., & Lehmann, E. L. (1963). Estimates of location based on rank tests.
  *Annals of Mathematical Statistics*, 34(2), 598–611.
- Westfall, P. H., & Young, S. S. (1993). *Resampling-Based Multiple Testing: Examples
  and Methods for p-Value Adjustment*. Wiley.
- Romano, J. P., & Wolf, M. (2005). Exact and approximate stepdown methods for multiple
  hypothesis testing. *Journal of the American Statistical Association*, 100(469),
  94–108.
- Holm, S. (1979). A simple sequentially rejective multiple test procedure.
  *Scandinavian Journal of Statistics*, 6(2), 65–70.
- Benjamini, Y., & Hochberg, Y. (1995). Controlling the false discovery rate: A
  practical and powerful approach to multiple testing. *Journal of the Royal Statistical
  Society: Series B (Methodological)*, 57(1), 289–300.

The functions and types described on this page are documented in the [API reference](reference/ri.md).
