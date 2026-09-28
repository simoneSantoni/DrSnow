# Adaptive experiments

```@meta
CurrentModule = DrSnow
```

In an adaptive experiment the probability with which the next participant receives
each treatment arm depends on the outcomes observed so far. Response-adaptive
designs (multi-armed bandits) assign more participants to arms that look better as
the experiment runs. This lowers the cost of experimenting (regret) and can speed up
finding the best arm. It also breaks the textbook analysis: sample means are biased,
and the usual standard errors are wrong. The `adaptive` area covers the whole
workflow:

- **Designing and running** adaptive experiments with assignment policies that
  record the exact assignment probabilities used: Thompson sampling
  ([`BetaBernoulliThompson`](@ref), [`GaussianThompson`](@ref),
  [`TopTwoThompson`](@ref)), [`EpsilonGreedy`](@ref), [`SoftmaxPolicy`](@ref),
  [`UCBPolicy`](@ref) and contextual [`LinearThompson`](@ref). Every policy
  supports probability floors (optionally decaying), an equal-allocation burn-in and
  batched updates. Experiments can be simulated with [`run_adaptive_experiment`](@ref)
  or deployed step by step with [`AdaptiveExperiment`](@ref),
  [`assign!`](@ref) and [`observe!`](@ref).
- **Inference after adaptive data collection**: adaptively weighted AIPW estimators
  of arm values, arm contrasts and policy values that remain asymptotically normal
  under adaptivity. These are [`adaptive_arm_values`](@ref) (Hadad et al. 2021) and
  [`adaptive_policy_value`](@ref), including the contextual weighting of Zhan et al.
  (2021).
- **Off-policy evaluation** from logged bandit data with IPW, self-normalized IPW
  and doubly robust estimators ([`off_policy_value`](@ref)). DR scores can be passed
  to [`policy_tree`](@ref) for policy learning ([`bandit_dr_scores`](@ref)).
- **Micro-randomized trials** (mobile health): causal excursion effects with
  [`wcls`](@ref) (continuous outcomes; Boruvka et al. 2018) and [`emee`](@ref)
  (binary outcomes; Qian et al. 2021). Both allow moderators, availability and the
  small-sample corrected variance, and match the R package MRTAnalysis.

## Identification: what adaptive designs require

Write `A_t` for the arm assigned to unit `t`, `X_t` for its covariates, `Y_t(w)`
for its potential outcomes and `H_{t-1}` for everything observed before unit `t`.
The methods on this page rely on three conditions.

**Sequential ignorability (known propensities).** The arm is drawn from
probabilities `e_t(X_t, w) = P(A_t = w | X_t, H_{t-1})` that depend only on the
observed past and the unit's covariates, never on its potential outcomes. The
analysis must use exactly these probabilities. The policies here compute the
probability vector first, draw the arm from it and store the vector in the
[`AdaptiveLog`](@ref). For Thompson sampling this vector is a numerical
approximation of the posterior probability that each arm is best. It is still the
exact distribution the arm was drawn from, so it is the correct propensity score.
Reconstructing Thompson probabilities after the fact from a separate Monte Carlo
run would not be.

**Positivity (floors).** Inverse-probability scores divide by `e_t`. If an arm's
probability goes to zero, the few units that still receive it get enormous weights,
estimators become unstable, and the central limit theorems below fail. A floor
`e_t(w) ≥ c t^{-α}` (keywords `floor = c`, `floor_decay = α` on every policy) keeps
the design analysable. A decaying floor (`α > 0`) lets the design concentrate on
the best arm while exploring less and less. Hadad et al. (2021) require the floor
to decay slowly enough (`α < 1`; their simulations use `α = 0.7`), and the rate `α`
enters their two-point weights. Assignment rules without randomization, such as
UCB without a floor, cannot be analysed with these tools.

**Stable outcomes.** Units do not interfere with each other, and the outcome
distribution does not drift over the experiment. If outcomes arrive with a delay,
use batches: within a batch the policy is not updated, which is also how most
field experiments run (Offer-Westort, Coppock & Green 2021).

## Choosing a design: regret, power and burn-in

Adaptive allocation trades two goals against each other. Assigning more units to
the arm that currently looks best lowers **regret**, the outcome given up by not
always assigning the best arm ([`cumulative_regret`](@ref) computes it for
simulations). But the inferior arms then get few units, so their means, and
contrasts involving them, are estimated with little **power**.

- **Thompson sampling** balances the two well for cumulative outcomes, and is the
  usual choice in social-science adaptive experiments (Offer-Westort et al. 2021).
- **Top-two Thompson sampling** (Russo 2020) deliberately keeps sampling the
  runner-up. It is preferable when the aim is to *identify* the best arm, or to
  estimate its advantage over the runner-up, rather than to maximize outcomes during
  the experiment.
- **ε-greedy and softmax** are simple and keep every arm's probability explicitly
  bounded away from zero.
- **UCB** is deterministic, so it needs a floor to be analysable.
- **Contextual policies** ([`LinearThompson`](@ref)) learn which arm works for whom.
  Athey et al. (2022) ran a contextual-bandit survey experiment with probability
  floors so that the data remain usable for inference afterwards.

**Burn-in.** Kaibel & Biemann (2021) show that response-adaptive randomization with
very few initial observations can lock allocation onto an arm by chance. They
recommend an equal-allocation burn-in before adaptation (keyword `burnin`, in
units) and moderate floors. Burn-in and floors cost some regret, but they protect
both the power for comparisons and the stability of the adaptive estimators.

Simulate the planned design with [`run_adaptive_experiment`](@ref) under plausible
arm means. That shows the regret, the allocation ([`plot_assignment_probabilities`](@ref))
and the width of the resulting confidence intervals before the experiment runs.

```julia
using DrSnow, StableRNGs
p = GaussianThompson(3; floor=1/3, floor_decay=0.7, burnin=15)
log = run_adaptive_experiment(p, GaussianBandit([0.9, 1.0, 1.1]), 2000;
                              batch_size=100, rng=StableRNG(1))
cumulative_regret(log)[end]
adaptive_arm_values(log)
```

For a live deployment, draw each batch's arms with [`assign!`](@ref) and feed back
the outcomes with [`observe!`](@ref). [`experiment_log`](@ref) returns the log.

## Inference on arm values and contrasts

The **sample mean** of an arm is biased under adaptive assignment. An arm with
unluckily low early outcomes is sampled less, so its low draws are never averaged
out. Its naive standard error also ignores the adaptivity.
[`naive_arm_means`](@ref) computes these numbers for comparison only.

**AIPW scores** fix the bias. With the recorded `e_t` and an outcome model
`μ̂_t(w)` built from units before `t` only (by default each arm's running mean),

`Γ_t(w) = μ̂_t(w) + 1{A_t = w} / e_t(w) · (Y_t - μ̂_t(w))`

has conditional mean `Q(w) = E[Y(w)]` given the past. Their average is unbiased.
But when `e_t(w)` shrinks, a few terms dominate the average and its t-statistic is
not normal. The **adaptively weighted** estimator
`Q̂(w) = Σ_t h_t Γ_t(w) / Σ_t h_t` uses variance-stabilizing weights `h_t`. These
give each period a comparable share of the variance and restore asymptotic normality
(Hadad et al. 2021, following Luedtke & van der Laan 2016). The standard error is
`sqrt(Σ h_t² (Γ_t - Q̂)²) / Σ h_t`, and contrasts use the joint covariance of the
separately weighted arm estimates.

| `weights` | Weights | Use |
|:--|:--|:--|
| `:two_point` (default) | two-point allocation: anticipates that an arm ends up either best (`e_t → 1`) or at the floor `c t^{-α}` | arm values and contrasts, non-contextual designs with floor rate `α` |
| `:constant_allocation` | `h_t = sqrt(e_t)` (StableVar) | the same designs when `α` is unknown |
| `:contextual` | contextual StableVar (Zhan et al. 2021) | designs whose probabilities depend on covariates |
| `:uniform` | `h_t = 1` (plain AIPW) | comparison only; intervals can under-cover |

The outcome model only affects efficiency. It must use earlier units only
(`:running_mean`, `:none` for IPW, or any [`NuisanceLearner`](@ref) refitted on all
earlier units at `n_blocks` points: sequential cross-fitting).

**Monte Carlo evidence** (design of Hadad et al. 2021: three arms with uniform noise,
Gaussian Thompson sampling, floor `(1/3) t^{-0.7}`, burn-in 15, 90% intervals; 2000
replications at `T = 1000` and `5000`, 1000 at `T = 20000`; full tables in
`test/validation/adaptive/hadad_montecarlo*.csv`):

| Design, `T` | two-point | constant allocation | uniform AIPW | sample mean |
|:--|:--|:--|:--|:--|
| no signal, 1000 | 0.89 / 0.89 | 0.88–0.90 / 0.89–0.90 | 0.89–0.90 / 0.91–0.92 | 0.82–0.84 / 0.79–0.81 |
| no signal, 5000 | 0.88–0.91 / 0.90 | 0.89–0.90 / 0.89–0.90 | 0.90–0.91 / 0.93–0.94 | 0.82–0.84 / 0.78–0.79 |
| no signal, 20000 | 0.89–0.91 / 0.91 | 0.88–0.89 / 0.91 | 0.90–0.91 / 0.95 | 0.82–0.84 / 0.81 |
| low signal, 1000 | 0.87–0.90 / 0.86–0.88 | 0.86–0.89 / 0.86 | 0.84–0.91 / 0.84–0.86 | 0.85–0.88 / 0.84–0.87 |
| low signal, 5000 | 0.88–0.89 / 0.88–0.89 | 0.87–0.89 / 0.87 | 0.85–0.90 / 0.85–0.86 | 0.86–0.89 / 0.86–0.89 |
| low signal, 20000 | 0.88–0.90 / 0.88–0.89 | 0.89–0.91 / 0.89 | 0.87–0.92 / 0.86–0.87 | 0.88–0.91 / 0.88–0.89 |
| high signal, 1000 | 0.85–0.89 / 0.86 | 0.83–0.90 / 0.83–0.85 | 0.81–0.90 / 0.81–0.82 | 0.86–0.90 / 0.86–0.88 |
| high signal, 5000 | 0.87–0.90 / 0.86–0.87 | 0.85–0.90 / 0.85–0.86 | 0.84–0.89 / 0.84 | 0.86–0.89 / 0.86–0.89 |
| high signal, 20000 | 0.86–0.92 / 0.87–0.89 | 0.85–0.90 / 0.84–0.88 | 0.85–0.90 / 0.85–0.87 | 0.87–0.90 / 0.87–0.90 |

Each cell gives the coverage of the three arm values (range) / of the two contrasts
with the best arm (range); Monte Carlo standard errors are about 0.007 (2000
replications) and 0.009 (1000). What the table shows:

- **No signal** (all arms equal, so allocation keeps fluctuating): the weighted
  estimators are close to nominal at every horizon, while the naive sample means
  under-cover badly (0.78–0.84) because of their downward bias and too-small standard
  errors. Uniform AIPW is valid here and conservative for contrasts.
- **Low and high signal** (the design concentrates on the best arm, and the worst
  arm receives only a few dozen units at the floor `(1/3) t^{-0.7}`): every method
  under-covers the values of the inferior arms somewhat at these horizons
  (0.83–0.90). Among the AIPW estimators, two-point weights come closest to nominal
  and uniform weights are the worst. The naive sample means have similar coverage in
  these two designs, but they are biased: in the low-signal design the bias of the
  inferior arms is -0.036 at `T = 20000`, compared with -0.014 for two-point weights.
- The weighted estimators carry a small finite-sample bias, because they are ratios
  with random weights. It shrinks with `T` (no signal, arm 1: -0.024, -0.012 and
  -0.008 at `T = 1000`, `5000` and `20000`).

Asymptotic normality is a large-`T` guarantee. With very small floors and few units
on an arm, treat the intervals for that arm as approximate. Confidence sequences do
not remove this problem (see the section on monitoring below): the AIPW scores are
unbounded when the floor decays, so the finite-sample guarantees of the
bounded-outcome confidence sequences do not apply to them.

## Policy values (including contextual bandits)

[`adaptive_policy_value`](@ref) estimates the value `E[Σ_w π(X, w) Y(w)]` of fixed
target policies (an arm, a function of covariates, a probability matrix or a
[`PolicyTree`](@ref)) from DR scores `Γ_t(π) = Σ_w π(X_t, w) Γ_t(w)`.

With a contextual design, the non-contextual weights of Hadad et al. are not enough,
because the probabilities vary with `X_t`. Zhan et al. (2021) use weights
`h_t(x) = 1/sqrt(Σ_w π(x,w)²/e_t(x,w))` that depend on the covariates, normalized
context by context (`weights = :contextual`, the default). This needs the
probability the design *would have* used in every batch for every unit's covariates.
Logs from [`LinearThompson`](@ref) keep a snapshot of the policy at the start of
each batch for this purpose. For tables, pass `probability_fn`. The cost is
`O(B T K)` for `B` batches, so batched designs are much cheaper to analyse.

Policies must be fixed in advance or learned on other data. Evaluating a policy on
the data it was learned from gives an optimistic value.

## Off-policy evaluation and policy learning from logged data

When a logging policy with known propensities did *not* adapt to outcomes, for
example a deployed rule with randomized exploration, [`off_policy_value`](@ref)
estimates the value of a new policy with
- IPW: `mean(π(A|X)/p · Y)`, unbiased but high-variance;
- self-normalized IPW: `Σ w Y / Σ w`, with a small bias and usually much lower
  variance (Swaminathan & Joachims 2015);
- doubly robust (default): a cross-fitted outcome model plus an IPW correction
  (Dudík, Langford & Li 2011).

The standard errors treat units as independent. For data from an adaptive
experiment use [`adaptive_policy_value`](@ref) instead.

For policy learning, [`bandit_dr_scores`](@ref) returns the `n × K` matrix of DR
scores, which [`policy_tree`](@ref) maximizes over shallow trees:

```julia
Γ = bandit_dr_scores(df, :y, :a; propensity=:p, covariates=[:x1, :x2])
tree = policy_tree(Γ, Matrix(df[:, [:x1, :x2]]); depth=2, actions=1:3,
                   covariates=[:x1, :x2])
```

For an adaptive log, `bandit_dr_scores(log)` returns scores whose outcome model uses
only earlier units. Learning from them without adaptive weights is consistent, but
the scores are heavy-tailed when the probabilities were small. Zhan, Ren, Athey &
Zhou (2024) study weighted policy learning, which is not implemented here.

## Micro-randomized trials

In a micro-randomized trial each participant is randomized many times, at every
decision point `t` where they are *available* (`I_t = 1`), with a known probability
`p_t(H_t)` (Klasnja et al. 2019). The causal **excursion effect**

`β(S_t) = E[Y_{t+1}(Ā_{t-1}, 1) - Y_{t+1}(Ā_{t-1}, 0) | S_t, I_t = 1]`

is the effect of treating at `t` rather than not, averaged over the trial's own
treatment policy for everything outside the moderators `S_t`. It is modelled as
`β'(1, S_t)`.

- [`wcls`](@ref) (Boruvka et al. 2018) uses weighted and centered least squares.
  The treatment is centered at a numerator probability `p̃_t` that depends on `S_t`
  only, and each point is weighted by `(p̃/p)^A ((1-p̃)/(1-p))^{1-A}`. `β̂` is
  consistent even when the control model is wrong.
- [`emee`](@ref) (Qian et al. 2021) estimates the effect on the log relative-risk
  scale for binary outcomes.

The variance clusters by participant. The small-sample correction multiplies each
participant's residuals by `(I - H_ii)^{-1}`, and intervals use `t(n - p - q)`. This
matches MRTAnalysis exactly. Note that MRTAnalysis's `wcls` applies the correction
only with at most 50 participants: set `small_sample = n ≤ 50` to reproduce that
rule. Unavailable decision points are dropped, and their outcomes may be missing.
Interpreting the control coefficients (`r.control_coef`) is not recommended.

```julia
r = wcls(df, :steps, :prompt, :id; rand_prob=:prob, moderators=[:day],
         controls=[:day, :steps_lag], availability=:avail)
plot_excursion_effect(r; moderator=:day)
```

## Which estimator to use

| Data | Target | Estimator |
|:--|:--|:--|
| bandit experiment, non-contextual | arm means, arm contrasts | [`adaptive_arm_values`](@ref), `weights = :two_point` (or `:constant_allocation`) |
| bandit experiment, any | value of a fixed policy | [`adaptive_policy_value`](@ref) |
| contextual bandit experiment | policy values, arm values | [`adaptive_policy_value`](@ref), [`adaptive_arm_values`](@ref) with `weights = :contextual` |
| logs of a non-adaptive randomized policy | value of a new policy | [`off_policy_value`](@ref) (`:dr`) |
| logged or experimental data | learn a policy | [`bandit_dr_scores`](@ref) + [`policy_tree`](@ref) |
| micro-randomized trial, continuous outcome | excursion effects | [`wcls`](@ref) |
| micro-randomized trial, binary outcome | excursion effects (relative risk) | [`emee`](@ref) |

## Monitoring during the experiment: always-valid inference

All intervals on this page are *fixed-horizon* intervals. They are valid when the
analysis happens once, at a sample size fixed in advance (or at a stopping time that
does not depend on the estimates). Looking at them repeatedly and stopping when they
exclude zero inflates the error rate. To monitor an adaptive experiment
continuously, use the time-uniform confidence sequences and always-valid p-values of
the [sequential inference](sequential.md) area, which are valid at data-dependent
stopping times under their own assumptions. The AIPW scores `Γ_t(w)` above are
martingale increments (they have conditional mean `Q(w)` given the past because
`μ̂_t` and `e_t` are predictable), but they are unbounded, with a variance that grows
as the floor decays. The finite-sample guarantees of the bounded-outcome confidence
sequences therefore do not apply to them, and the asymptotic confidence sequences
give at best an asymptotic guarantee. DrSnow does not implement or validate this
combination: in an exploratory Monte Carlo run during the documentation review (not part
of the test suite; 300 replications), the time-uniform coverage of a confidence
sequence on the AIPW scores was as low as 0.81 for an inferior arm in a high-signal
design. Nor is it valid to monitor with a confidence sequence and then,
after stopping, report the fixed-horizon adaptively weighted interval: when the
stopping time depends on the confidence sequence, and hence on the outcomes, that
interval loses its coverage guarantee.

## Validation

- **Adaptively weighted AIPW** agrees to `1e-10` with the reference code of Hadad et
  al. (gsbDBI/adaptive-confidence-intervals). One Thompson-sampling experiment is
  generated with the authors' Python code, and all arm values, contrasts and
  standard errors are compared for two-point, constant-allocation and uniform
  weights, with AIPW and IPW scores (`test/validation/adaptive/hadad_reference.py`).
- **Contextual weighting** agrees exactly with a brute-force `T × T` implementation
  of the estimator and variance of Zhan et al. (2021), as in their code
  (gsbDBI/contextual_bandits_evaluation).
- **WCLS and EMEE** agree with MRTAnalysis 0.4.1 to `1e-7` or better (estimates,
  standard errors and degrees of freedom). The comparison uses the package's
  HeartSteps-mimic and binary example data plus a simulated trial with 120
  participants (`generate_mrt_references.R`).
- **Monte Carlo** coverage checks cover the Hadad et al. design (table above), WCLS
  (95% coverage with 30 participants), EMEE (conservative, about 97% with 40
  participants), off-policy evaluation, and contextual weighting (a
  linear-Thompson design with four arms).

## References

- Athey, S., Byambadalai, U., Hadad, V., Krishnamurthy, S. K., Leung, W., & Williams,
  J. J. (2022). Contextual bandits in a survey experiment on charitable giving:
  Within-experiment outcomes versus policy learning. arXiv:2211.12004.
- Boruvka, A., Almirall, D., Witkiewitz, K., & Murphy, S. A. (2018). Assessing
  time-varying causal effect moderation in mobile health. *Journal of the American
  Statistical Association*, 113(523), 1112–1121.
- Dudík, M., Langford, J., & Li, L. (2011). Doubly robust policy evaluation and
  learning. In *Proceedings of the 28th International Conference on Machine Learning*
  (pp. 1097–1104).
- Hadad, V., Hirshberg, D. A., Zhan, R., Wager, S., & Athey, S. (2021). Confidence
  intervals for policy evaluation in adaptive experiments. *Proceedings of the National
  Academy of Sciences*, 118(15), e2014602118.
- Kaibel, C., & Biemann, T. (2021). Rethinking the gold standard with multi-armed
  bandits: Machine learning allocation algorithms for experiments. *Organizational
  Research Methods*, 24(1), 78–103.
- Klasnja, P., Smith, S., Seewald, N. J., Lee, A., Hall, K., Luers, B., Hekler, E. B., &
  Murphy, S. A. (2019). Efficacy of contextually tailored suggestions for physical
  activity: A micro-randomized optimization trial of HeartSteps. *Annals of Behavioral
  Medicine*, 53(6), 573–582.
- Luedtke, A. R., & van der Laan, M. J. (2016). Statistical inference for the mean
  outcome under a possibly non-unique optimal treatment strategy. *Annals of
  Statistics*, 44(2), 713–742.
- Nie, X., Tian, X., Taylor, J., & Zou, J. (2018). Why adaptively collected data have
  negative bias and how to correct for it. In *Proceedings of the 21st International
  Conference on Artificial Intelligence and Statistics*, PMLR 84, 1261–1269.
- Offer-Westort, M., Coppock, A., & Green, D. P. (2021). Adaptive experimental design:
  Prospects and applications in political science. *American Journal of Political
  Science*, 65(4), 826–844.
- Qian, T., Yoo, H., Klasnja, P., Almirall, D., & Murphy, S. A. (2021). Estimating
  time-varying causal excursion effects in mobile health with binary outcomes.
  *Biometrika*, 108(3), 507–527.
- Russo, D. (2020). Simple Bayesian algorithms for best-arm identification. *Operations
  Research*, 68(6), 1625–1647.
- Russo, D., Van Roy, B., Kazerouni, A., Osband, I., & Wen, Z. (2018). A tutorial on
  Thompson sampling. *Foundations and Trends in Machine Learning*, 11(1), 1–96.
- Swaminathan, A., & Joachims, T. (2015). The self-normalized estimator for
  counterfactual learning. *Advances in Neural Information Processing Systems*, 28,
  3231–3239.
- Zhan, R., Hadad, V., Hirshberg, D. A., & Athey, S. (2021). Off-policy evaluation via
  adaptive weighting with data from contextual bandits. In *Proceedings of the 27th ACM
  SIGKDD Conference on Knowledge Discovery and Data Mining* (pp. 2125–2135).
- Zhan, R., Ren, Z., Athey, S., & Zhou, Z. (2024). Policy learning with adaptively
  collected data. *Management Science*, 70(8), 5270–5297.

The functions and types described on this page are documented in the [API reference](reference/adaptive.md).
