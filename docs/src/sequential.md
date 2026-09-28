# Sequential inference

```@meta
CurrentModule = DrSnow
```

Experiments are often analysed while the data are still arriving: an online A/B test
is watched on a dashboard, a trial has interim analyses by a monitoring committee. A
fixed-sample confidence interval or p-value computed repeatedly on accumulating data
is **not** valid: checking a 95% interval after every observation and stopping the
first time it excludes zero rejects a true null far more often than 5% of the time
(about 50% within 600 observations in the simulations below). This area provides the
two families of methods that remain valid under monitoring:

- **Anytime-valid inference** — confidence sequences, e-processes and always-valid
  p-values. Valid at *every* sample size simultaneously, so the data may be looked at
  continuously and the experiment stopped (or extended) at any time, for any
  reason.
- **Group-sequential designs** — a small number of pre-scheduled interim looks with
  boundaries chosen so that the overall type-I error is exactly `α`.

## Which one to use

| Situation | Use |
|:--|:--|
| Continuous monitoring (dashboards, streaming data), no fixed schedule of looks, possible early stopping *or* extension | confidence sequences ([`confseq_mean`](@ref), [`confseq_ate`](@ref)) or the mSPRT ([`msprt_test`](@ref)) |
| A few interim analyses planned in advance (clinical trials, costly batches), regulatory-style error control, power and sample size planning | group-sequential design ([`gs_design`](@ref), [`gs_analysis`](@ref)) |
| Bounded outcome (conversion, rating, capped revenue) and exact finite-sample guarantees wanted | bounded-data confidence sequences (`:betting`, `:empirical_bernstein`) |

What peeking freedom costs. A confidence sequence at level `1-α` is wider than the
fixed-sample interval at every `n`: in the examples below by a factor of about 1.4–1.6
near the sample size it was tuned for (`t_opt`) and more elsewhere (the width shrinks
like `√(log n / n)` instead of `1/√n`). A group-sequential design with few looks costs
far less — an O'Brien–Fleming-type design with three looks needs about 1% more
information than the fixed design — because it only protects the looks that were
scheduled. Checking a group-sequential design more often than planned, or a
confidence sequence less often than possible, gives up the advantage of each.

## Confidence sequences

A `(1-α)` confidence sequence `(Cₜ)` satisfies
`P(θ ∈ Cₜ for all t ≥ 1) ≥ 1 - α`. Each is built from a family of nonnegative
(super)martingales — e-processes — `Eₜ(m)` indexed by candidate values `m`, and
`Cₜ = {m : Eₜ(m) < 1/α}`. Ville's inequality `P(∃t: Eₜ(θ) ≥ 1/α) ≤ α` gives the
time-uniform guarantee. The always-valid p-value for `H₀: θ = θ₀` is
`pₜ = min(1, 1/max_{s≤t} Eₛ(θ₀))`: rejecting when `pₜ ≤ α` at any data-dependent time
is a level-`α` test. The intersection `∩_{s≤t} Cₛ` is also a confidence sequence
and is reported by default (`running_intersection = true`).

[`confseq_mean`](@ref) (batch) and [`MeanMonitor`](@ref) (streaming) implement five
boundaries:

| `method` | data | guarantee | reference |
|:--|:--|:--|:--|
| `:asymptotic` | finite variance | asymptotic (like a CLT interval) | Waudby-Smith et al. (2024) |
| `:normal_mixture` | σ-sub-Gaussian, σ known | exact | Robbins (1970); Howard et al. (2021) |
| `:hoeffding` | bounded in `bounds` | exact | Waudby-Smith & Ramdas (2024) |
| `:empirical_bernstein` | bounded | exact, variance-adaptive | Waudby-Smith & Ramdas (2024) |
| `:betting` | bounded | exact, tightest | Waudby-Smith & Ramdas (2024) |

**Tuning for a sample size.** No confidence sequence is uniformly tightest; each is
tuned to be tightest around one sample size, `t_opt`. For the Gaussian boundaries the
normal-mixture parameter is `ρ = t_opt / (2 log(1/α) + log(1 + 2 log(1/α)))`, which
minimizes the width at `n = t_opt` (Howard et al. 2021, eq. (14)); set `t_opt` to the
planned sample size or to the size at which a decision is most likely. The bounded
boundaries use bets `λₜ ∝ 1/√(t log(1+t))` by default (good over a wide range of `n`)
or bets tuned to `t_opt` when it is given.

**Asymptotic confidence sequences** replace `σ` by the running standard deviation.
Their guarantee is asymptotic in the start time: with `σ̂` computed from very few
points the early intervals are too narrow. Monitoring therefore starts at `min_n = 20`
observations by default. Starting at `n = 1` gives a time-uniform coverage of about
0.91 for normal data and 0.83 for exponential data (nominal 0.95), against 0.97 and
0.95 when starting at `n = 20`.

## A/B tests and randomized experiments

[`confseq_ate`](@ref) / [`ATEMonitor`](@ref) give an asymptotic confidence sequence
for the average treatment effect from AIPW pseudo-outcomes
`φᵢ = μ̂₁(xᵢ) - μ̂₀(xᵢ) + dᵢ(yᵢ - μ̂₁(xᵢ))/π̂(xᵢ) - (1-dᵢ)(yᵢ - μ̂₀(xᵢ))/(1-π̂(xᵢ))`
whose nuisance estimates are fitted on the units that arrived *before* unit `i`
(predictable plug-ins; Waudby-Smith et al. 2024, Section 3). Any DrSnow
[`NuisanceLearner`](@ref) can be used for the outcome regressions and the
propensity. With known assignment probabilities (`propensity = 0.5`, or a column of
known probabilities) each `φᵢ` has conditional mean equal to the ATE whatever the
outcome model, so regression adjustment can only change the width — by about 30% in
the example below — not the validity. With learners, the models are refitted every
`refit_every` units, and the units of a block all use models fitted on earlier
blocks.

[`msprt_test`](@ref) / [`MSPRTMonitor`](@ref) implement the mixture sequential
probability ratio test of Johari et al. (2022) for the difference in means or in
conversion rates, with always-valid p-values and intervals. The mixture ratio is an
exact martingale for Gaussian outcomes with known variance arriving in pairs, and an
approximation (through the CLT) otherwise, including for conversions and with
estimated variances.

[`sequential_test`](@ref) turns any confidence sequence into a test of its `null`,
and [`stopping`](@ref) summarizes the first time the test rejected. A non-rejection
is reported as such; it is not evidence for the null.

### Streaming

Every monitor is updated with `fit!` and read with [`snapshot`](@ref) at any time:

```julia
m = ATEMonitor(; propensity=0.5, t_opt=10_000)
for (y, d) in incoming_units
    fit!(m, y, d)
    s = snapshot(m)                   # s.lower, s.upper, s.pvalue, s.n, ...
    (s.lower > 0 || s.upper < 0) && break
end
cs = confidence_sequence(m)           # the whole path, for tables and plots
sequence_path(cs)                     # DataFrame: n, estimate, lower, upper, ...
```

The batch functions feed the same monitors, so a batch result equals the streaming
result on the same data.

## Group-sequential designs

[`gs_design`](@ref) computes boundaries for `k` analyses at information fractions
`t₁ < … < t_k = 1` by the recursive numerical integration of Armitage, McPherson &
Rowe (1969) with the grid of Jennison & Turnbull (2000, ch. 19) — the algorithm of
gsDesign:

- classical boundaries: Pocock (1977) constant, O'Brien & Fleming (1979) `c/√tₖ`,
  Haybittle–Peto (3 at interim looks, final bound adjusted to spend exactly `α`);
- Lan & DeMets (1983) error spending with [`OBFSpending`](@ref),
  [`PocockSpending`](@ref), [`PowerSpending`](@ref) (Kim & DeMets 1987) and
  [`HSDSpending`](@ref) (Hwang, Shih & DeCani 1990);
- optional β-spending futility bounds, non-binding by default;
- one-sided designs and two-sided symmetric designs.

The design reports the inflation factor `I_max / I_fixed` — multiply the fixed-design
sample size by it (`n_fixed` does this) — the cumulative α spent, crossing
probabilities and the expected sample size under `H₀` and `H₁`.

[`gs_analysis`](@ref) analyses the looks performed so far from the effect estimates
and their standard errors. Spending boundaries are recomputed at the observed
information, so looks may be added, dropped or moved as long as the decision to look
does not depend on the observed effect. It returns the stopping decision, repeated
confidence intervals (Jennison & Turnbull 1989) and repeated p-values, and after
stopping the stagewise-ordering p-value, median-unbiased estimate and confidence
interval (Tsiatis, Rosner & Mehta 1984). Futility bounds are non-binding: they are
advisory, and continuing after crossing one does not inflate the type-I error.

```julia
d = gs_design(; k=3, alpha=0.025, beta=0.1, efficacy=OBFSpending(),
              futility=HSDSpending(-2), n_fixed=500)
d.n                                  # cumulative sample size per look
a = gs_analysis(d, [0.18, 0.31], [0.14, 0.10])
a.decision, confint(a), pvalues(a)
```

## Plotting

With a Makie backend loaded, [`plot_confidence_sequence`](@ref) draws the path of a
confidence sequence (or of an mSPRT interval) with the pointwise fixed-sample
interval for comparison, and [`plot_gs_boundaries`](@ref) draws the boundaries of a
group-sequential design with the observed statistics of an analysis.

## Validation

- Confidence sequences: the Hoeffding, empirical-Bernstein and betting paths and the
  normal-mixture boundary agree with the Python package confseq 0.0.11 to 1e-12 (two
  data streams, two levels, tuned and untuned, with and without running
  intersection).
- Group-sequential designs: 44 gsDesign 3.11.0 designs (spending families, classical
  boundaries, one- and two-sided, binding and non-binding β-spending futility,
  unequal timing) and 28 rpact 4.4.0 designs (including Haybittle–Peto): boundaries
  within 1e-6, α spent within 1e-7, inflation factors and expected sample sizes within
  1e-6.
- Group-sequential analysis: rpact's critical values, repeated p-values,
  stagewise-ordering p-values, median-unbiased estimates and final confidence
  intervals are reproduced to 1e-5 or better.
- Monte Carlo (in the test suite): time-uniform coverage of every confidence
  sequence, type-I error of the mSPRT under continuous monitoring, and type-I error
  of simulated group-sequential trials, each contrasted with naive repeated
  fixed-sample intervals or tests.

## References

- Armitage, P., McPherson, C. K., & Rowe, B. C. (1969). Repeated significance tests on
  accumulating data. *Journal of the Royal Statistical Society: Series A*, 132(2),
  235–244.
- Howard, S. R., Ramdas, A., McAuliffe, J., & Sekhon, J. (2021). Time-uniform,
  nonparametric, nonasymptotic confidence sequences. *Annals of Statistics*, 49(2),
  1055–1080.
- Hwang, I. K., Shih, W. J., & DeCani, J. S. (1990). Group sequential designs using a
  family of type I error probability spending functions. *Statistics in Medicine*,
  9(12), 1439–1445.
- Jennison, C., & Turnbull, B. W. (1989). Interim analyses: The repeated confidence
  interval approach. *Journal of the Royal Statistical Society: Series B*, 51(3),
  305–334.
- Jennison, C., & Turnbull, B. W. (2000). *Group Sequential Methods with Applications to
  Clinical Trials*. Chapman & Hall/CRC.
- Johari, R., Koomen, P., Pekelis, L., & Walsh, D. (2022). Always valid inference:
  Continuous monitoring of A/B tests. *Operations Research*, 70(3), 1806–1821.
- Kim, K., & DeMets, D. L. (1987). Design and analysis of group sequential tests based
  on the type I error spending rate function. *Biometrika*, 74(1), 149–154.
- Lan, K. K. G., & DeMets, D. L. (1983). Discrete sequential boundaries for clinical
  trials. *Biometrika*, 70(3), 659–663.
- O'Brien, P. C., & Fleming, T. R. (1979). A multiple testing procedure for clinical
  trials. *Biometrics*, 35(3), 549–556.
- Pocock, S. J. (1977). Group sequential methods in the design and analysis of clinical
  trials. *Biometrika*, 64(2), 191–199.
- Ramdas, A., Grünwald, P., Vovk, V., & Shafer, G. (2023). Game-theoretic statistics and
  safe anytime-valid inference. *Statistical Science*, 38(4), 576–601.
- Robbins, H. (1970). Statistical methods related to the law of the iterated logarithm.
  *Annals of Mathematical Statistics*, 41(5), 1397–1409.
- Tsiatis, A. A., Rosner, G. L., & Mehta, C. R. (1984). Exact confidence intervals
  following a group sequential test. *Biometrics*, 40(3), 797–803.
- Waudby-Smith, I., & Ramdas, A. (2024). Estimating means of bounded random variables by
  betting. *Journal of the Royal Statistical Society: Series B*, 86(1), 1–27.
- Waudby-Smith, I., Arbour, D., Sinha, R., Kennedy, E. H., & Ramdas, A. (2024).
  Time-uniform central limit theory and asymptotic confidence sequences. *Annals of
  Statistics*, 52(6), 2613–2640.

The functions and types described on this page are documented in the [API reference](reference/sequential.md).
