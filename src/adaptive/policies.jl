# Response-adaptive assignment policies (bandit algorithms) with assignment-probability
# floors, equal-allocation burn-in and exact recording of the probabilities used.
#
# Interface of every `AdaptivePolicy` `p` (arms are indexed 1:K):
#     assignment_probabilities(p, t; context) :: Vector{Float64}   (sums to one)
#     update_policy!(p, arm, outcome; context)                      (learn from one unit)
# Internally a policy implements `_ad_raw_probabilities(p, context, rng)` (before floor
# and burn-in) and `_ad_update!(p, arm, y, x)`. `t` is the index of the first unit of
# the batch the probabilities are used for; floors and burn-in depend on it only, so
# the probability *function* is fixed within a batch.

"""
    AdaptivePolicy

Abstract supertype of the response-adaptive assignment policies (multi-armed and
contextual bandit algorithms) that DrSnow can deploy, simulate and analyse:
[`BetaBernoulliThompson`](@ref), [`GaussianThompson`](@ref),
[`TopTwoThompson`](@ref), [`EpsilonGreedy`](@ref), [`SoftmaxPolicy`](@ref),
[`UCBPolicy`](@ref) and the contextual [`LinearThompson`](@ref).

A policy maps the history of an experiment to a probability distribution over its
`K` arms for the next unit or batch of units. Writing ``H_{t-1}`` for the arms,
outcomes and covariates of the units that arrived before unit ``t``, the policy
defines the assignment probabilities (propensity scores)

```math
e_t(x, w) = P(A_t = w \\mid X_t = x, H_{t-1}), \\qquad w = 1, \\dots, K,
```

and the arm ``A_t`` of unit ``t`` is drawn from ``e_t(X_t, \\cdot)``. Because
``e_t`` depends only on the observed past and on the unit's own covariates, never on
its potential outcomes, the design is sequentially ignorable by construction. This
is the property that inference after adaptive data collection
([`adaptive_arm_values`](@ref), [`adaptive_policy_value`](@ref)) relies on, and it
can only be exploited if the probabilities actually used are recorded exactly;
[`AdaptiveExperiment`](@ref) and [`run_adaptive_experiment`](@ref) do so.

**Positivity is not guaranteed by default.** Every policy has `floor = 0` by
default, so nothing prevents an arm's assignment probability from falling towards
zero (Thompson sampling, softmax) or being exactly zero (UCB, and ε-greedy with
`epsilon = 0`). The adaptive estimators divide by ``e_t``: without a floor the
inverse-probability weights can become arbitrarily large, an arm may be sampled only
finitely often, and the central limit theorems behind the confidence intervals of
the analysis functions do not apply. Whenever the data will be analysed, set a
floor. With `floor = c` and `floor_decay = α`, every arm receives probability at
least ``c\\, t^{-α}`` at unit ``t``; the policy's own probabilities are raised to the
floor and the excess is removed proportionally from the arms above it, as in the
reference code of Hadad et al. (2021). Hadad, Hirshberg, Zhan, Wager and Athey
(2021) prove asymptotic normality of their adaptively weighted arm-value estimators
when ``e_t(w) \\ge C t^{-α}`` for some ``α \\in [0, 1)``; Zhan, Hadad, Hirshberg and
Athey (2021) assume the stronger ``α \\in [0, 1/2)`` for policy evaluation with
contextual-bandit data. A constant floor (``α = 0``) gives the most stable inference;
a decaying floor lets the design concentrate on the best arm, which lowers regret,
at the price of fewer observations on inferior arms and slower convergence of their
estimates.

An equal-allocation burn-in (`burnin` units assigned with probability ``1/K`` each)
protects against the design locking onto an arm on the strength of a few noisy early
outcomes, and it guarantees every arm a minimum sample. Which algorithm to use
depends on the goal. Thompson sampling (Thompson 1933; Russo et al. 2018) balances
cumulative outcomes and learning and is the usual choice in social-science adaptive
experiments (Offer-Westort, Coppock and Green 2021; Kaibel and Biemann 2021).
Top-two Thompson sampling targets identification of the best arm (Russo 2020); the
exploration sampling of Kasy and Sautmann (2021), designed for choosing a policy at
the end of the experiment, is not implemented. ε-greedy and softmax rules are simple
heuristics with explicit exploration, UCB is deterministic, and
[`LinearThompson`](@ref) learns which arm works best for which covariate profile.
Lattimore and Szepesvári (2020) give a textbook treatment of these algorithms.

# Interface
- [`assignment_probabilities`](@ref)`(p, t; context)`: the probabilities the policy
  would use for a batch starting at unit `t`.
- [`update_policy!`](@ref)`(p, arm, outcome; context)`: learn from one outcome.
- [`n_arms`](@ref)`(p)`: the number of arms.

# Keywords shared by every policy
- `floor::Real = 0`: minimum assignment probability ``c`` of every arm, with
  `K * floor ≤ 1`. The default `0` provides no positivity guarantee.
- `floor_decay::Real = 0`: decay rate ``α \\in [0, 1)`` of the floor ``c\\, t^{-α}``;
  `0` keeps the floor constant.
- `burnin::Integer = 0`: number of initial units assigned with equal probabilities
  `1/K` before the algorithm takes over.

# References
- Thompson, W. R. (1933). On the likelihood that one unknown probability exceeds
  another in view of the evidence of two samples. *Biometrika*, 25(3/4), 285–294.
- Hadad, V., Hirshberg, D. A., Zhan, R., Wager, S., & Athey, S. (2021). Confidence
  intervals for policy evaluation in adaptive experiments. *Proceedings of the
  National Academy of Sciences*, 118(15), e2014602118.
- Zhan, R., Hadad, V., Hirshberg, D. A., & Athey, S. (2021). Off-policy evaluation
  via adaptive weighting with data from contextual bandits. In *Proceedings of the
  27th ACM SIGKDD Conference on Knowledge Discovery and Data Mining* (pp. 2125–2135).
- Offer-Westort, M., Coppock, A., & Green, D. P. (2021). Adaptive experimental
  design: Prospects and applications in political science. *American Journal of
  Political Science*, 65(4), 826–844.
- Kaibel, C., & Biemann, T. (2021). Rethinking the gold standard with multi-armed
  bandits: Machine learning allocation algorithms for experiments. *Organizational
  Research Methods*, 24(1), 78–103.
- Kasy, M., & Sautmann, A. (2021). Adaptive treatment assignment in experiments for
  policy choice. *Econometrica*, 89(1), 113–132.
- Lattimore, T., & Szepesvári, C. (2020). *Bandit Algorithms*. Cambridge University
  Press.
"""
abstract type AdaptivePolicy end

# Settings shared by every policy.
struct _AdSettings
    K::Int
    floor::Float64
    floor_decay::Float64
    burnin::Int
end

function _ad_settings(K, floor, floor_decay, burnin)
    K >= 2 || throw(ArgumentError("need at least two arms, got K = $K"))
    0 <= floor || throw(ArgumentError("floor must be non-negative, got $floor"))
    K * floor <= 1 + 1e-12 ||
        throw(ArgumentError("floor must satisfy K * floor ≤ 1 (K = $K, floor = $floor)"))
    0 <= floor_decay < 1 ||
        throw(ArgumentError("floor_decay must be in [0, 1), got $floor_decay"))
    burnin >= 0 || throw(ArgumentError("burnin must be non-negative"))
    return _AdSettings(Int(K), float(floor), float(floor_decay), Int(burnin))
end

_ad_floor(s::_AdSettings, t::Integer) = s.floor * float(t)^(-s.floor_decay)

"""
    _ad_apply_floor(p, amin)

Raise every probability to at least `amin` and remove the excess proportionally from
the arms above the floor (Hadad et al. 2021, `apply_floor`), so the result sums to one.
"""
function _ad_apply_floor(p::AbstractVector{<:Real}, amin::Real)
    K = length(p)
    amin <= 0 && return collect(float.(p))
    amin * K >= 1 - 1e-12 && return fill(1.0 / K, K)
    new = max.(float.(p), amin)
    slack = sum(new) - 1
    indiv = new .- amin
    tot = sum(indiv)
    tot <= 0 && return fill(1.0 / K, K)
    out = new .- (slack / tot) .* indiv
    out ./= sum(out)
    return out
end

_ad_is_contextual(::AdaptivePolicy) = false
_ad_dim(::AdaptivePolicy) = 0

"""
    n_arms(p::AdaptivePolicy) -> Int

Number of treatment arms `K` of an adaptive assignment policy.

Arms are always indexed `1:K` inside a policy and in an [`AdaptiveLog`](@ref);
analysis functions label them by these integers.

# Arguments
- `p::AdaptivePolicy`: any policy, including a [`TopTwoThompson`](@ref) wrapper
  (which reports the arms of its base policy).

# Returns
- `Int`: the number of arms.

# Examples
```julia
using DrSnow
n_arms(GaussianThompson(4))          # 4
```
"""
n_arms(p::AdaptivePolicy) = p.settings.K

"""
    assignment_probabilities(p::AdaptivePolicy, t=1; context=nothing,
                             rng=Random.default_rng()) -> Vector{Float64}

Probabilities with which policy `p`, in its current state, assigns each of its `K`
arms to a unit of a batch that starts at unit `t`.

The returned vector is the propensity score ``e_t(x, \\cdot)`` of the design: during
the burn-in (`t ≤ burnin`) every arm receives ``1/K``; afterwards the algorithm's
own probabilities are computed from its current statistics and raised to the floor
``c\\, t^{-α}`` (keywords `floor` and `floor_decay` of the policy), with the excess
removed proportionally from the arms above the floor (see [`AdaptivePolicy`](@ref)).
The floor and burn-in depend on `t` only, so the probability *function* is fixed
within a batch, which is what the batched analyses of
[`adaptive_arm_values`](@ref) assume.

For Thompson-sampling policies the ideal probability that an arm is optimal under the
posterior has no closed form. It is computed by deterministic numerical integration
over a grid of 256 equal-probability cells or, for [`BetaBernoulliThompson`](@ref)
with `ndraws > 0`, by Monte Carlo with `rng`. Either way the returned vector is
exactly the distribution from which [`assign!`](@ref) then draws the arm, so it is
the correct propensity score for inference even though it only approximates the
ideal Thompson probabilities. Reconstructing these probabilities after the fact with
a separate Monte Carlo run would not give the propensities actually used.

# Arguments
- `p::AdaptivePolicy`: the policy, in its current state.
- `t::Integer`: index (1-based) of the first unit of the batch for which the
  probabilities are used; it determines the burn-in and the floor.

# Keywords
- `context`: the unit's covariate vector, required by contextual policies
  ([`LinearThompson`](@ref)) and ignored by the others.
- `rng::AbstractRNG = Random.default_rng()`: random-number generator used only by
  Monte Carlo probability computations (`BetaBernoulliThompson` with `ndraws > 0`).

# Returns
- `Vector{Float64}` of length `K`, non-negative and summing to one.

# Examples
```julia
using DrSnow
p = GaussianThompson(3; floor=0.05)
update_policy!(p, 1, 0.4)
assignment_probabilities(p, 2)
```

# References
- Russo, D., Van Roy, B., Kazerouni, A., Osband, I., & Wen, Z. (2018). A tutorial
  on Thompson sampling. *Foundations and Trends in Machine Learning*, 11(1), 1–96.
- Hadad, V., Hirshberg, D. A., Zhan, R., Wager, S., & Athey, S. (2021). Confidence
  intervals for policy evaluation in adaptive experiments. *Proceedings of the
  National Academy of Sciences*, 118(15), e2014602118.
"""
function assignment_probabilities(p::AdaptivePolicy, t::Integer=1; context=nothing,
                                  rng::AbstractRNG=Random.default_rng())
    t >= 1 || throw(ArgumentError("t must be ≥ 1, got $t"))
    s = p.settings
    x = _ad_check_context(p, context)
    t <= s.burnin && return fill(1.0 / s.K, s.K)
    raw = _ad_raw_probabilities(p, x, rng)
    all(isfinite, raw) ||
        throw(ArgumentError("assignment probabilities are not finite; check the " *
                            "policy's data and parameters"))
    return _ad_apply_floor(raw ./ sum(raw), _ad_floor(s, t))
end

function _ad_check_context(p::AdaptivePolicy, context)
    _ad_is_contextual(p) || return nothing
    context === nothing &&
        throw(ArgumentError("$(nameof(typeof(p))) is contextual: pass `context`"))
    length(context) == _ad_dim(p) ||
        throw(DimensionMismatch("context must have length $(_ad_dim(p)), got " *
                                "$(length(context))"))
    x = Float64.(collect(context))
    all(isfinite, x) || throw(ArgumentError("context must be finite"))
    return x
end

"""
    update_policy!(p::AdaptivePolicy, arm::Integer, outcome::Real;
                   context=nothing) -> p

Update the sufficient statistics (or posterior) of policy `p` with the observed
outcome of one unit that was assigned to `arm`.

Thompson-sampling policies update their conjugate posteriors (Beta for binary
outcomes, normal for continuous outcomes, a Bayesian linear regression per arm for
[`LinearThompson`](@ref)); ε-greedy, softmax and UCB policies update the running
count and sum of each arm. The update changes the probabilities returned by the next
call of [`assignment_probabilities`](@ref). In a batched experiment the policy is
updated only after the whole batch has been observed ([`observe!`](@ref) does this),
which is how delayed outcomes are handled.

# Arguments
- `p::AdaptivePolicy`: the policy to update (mutated).
- `arm::Integer`: the arm the unit received, in `1:K`.
- `outcome::Real`: the unit's observed outcome; it must be finite, and `0` or `1`
  for [`BetaBernoulliThompson`](@ref).

# Keywords
- `context`: the unit's covariate vector, required by contextual policies and
  ignored by the others.

# Returns
- The updated policy `p`.

# Examples
```julia
using DrSnow
p = BetaBernoulliThompson(2)
update_policy!(p, 2, 1)
assignment_probabilities(p)
```
"""
function update_policy!(p::AdaptivePolicy, arm::Integer, outcome::Real; context=nothing)
    1 <= arm <= n_arms(p) || throw(ArgumentError("arm must be in 1:$(n_arms(p))"))
    isfinite(outcome) || throw(ArgumentError("outcome must be finite"))
    x = _ad_check_context(p, context)
    _ad_update!(p, Int(arm), float(outcome), x)
    return p
end

# ---------------------------------------------------------------------------
# Probability that each arm has the largest draw
# ---------------------------------------------------------------------------

const _AD_NGRID = 256
# Standard normal quantiles at the midpoints of 256 equal-probability cells.
const _AD_ZGRID = [quantile(Normal(), (i - 0.5) / _AD_NGRID)
                   for i in 1:_AD_NGRID]

"""
    _ad_prob_best_normal(m, s)

`P(arm k has the largest value)` for independent `N(m_k, s_k²)` variables, by
integrating `Π_{j≠k} Φ((m_k + s_k z - m_j)/s_j)` over `z ~ N(0, 1)` on a grid of 256
equal-probability cells (absolute error of order 1/256 at most for step-like
integrands, much smaller for smooth ones). Degenerate `s_k = 0` is handled.
"""
function _ad_prob_best_normal(m::AbstractVector, s::AbstractVector)
    K = length(m)
    out = zeros(K)
    for k in 1:K
        acc = 0.0
        for z in _AD_ZGRID
            v = m[k] + s[k] * z
            pr = 1.0
            for j in 1:K
                j == k && continue
                pr *= s[j] > 0 ? cdf(Normal(), (v - m[j]) / s[j]) :
                      (v > m[j] ? 1.0 : (v == m[j] ? 0.5 : 0.0))
                pr == 0 && break
            end
            acc += pr
        end
        out[k] = acc / _AD_NGRID
    end
    tot = sum(out)
    tot > 0 || return fill(1.0 / K, K)
    return out ./ tot
end

"""Monte Carlo `P(arm k is best)` from a `ndraws × K` matrix of posterior draws."""
function _ad_prob_best_draws(D::AbstractMatrix)
    n, K = size(D)
    cnt = zeros(K)
    for i in 1:n
        cnt[argmax(view(D, i, :))] += 1
    end
    return cnt ./ n
end

# Unit-level running statistics used by the non-Bayesian policies.
function _ad_running_means(n::Vector{Int}, s::Vector{Float64})
    obs = n .> 0
    return obs, [n[k] > 0 ? s[k] / n[k] : NaN for k in eachindex(n)]
end

"""Indices of the best arms; arms never observed count as best (forced exploration)."""
function _ad_best_arms(n::Vector{Int}, s::Vector{Float64})
    obs, μ = _ad_running_means(n, s)
    all(obs) || return findall(.!obs)
    mx = maximum(μ)
    return findall(==(mx), μ)
end

# ---------------------------------------------------------------------------
# Beta-Bernoulli Thompson sampling
# ---------------------------------------------------------------------------

"""
    BetaBernoulliThompson(K; a=1.0, b=1.0, ndraws=1000, floor=0.0, floor_decay=0.0,
                          burnin=0) -> BetaBernoulliThompson

Thompson sampling for binary outcomes with independent Beta priors on the success
probabilities of `K` arms.

Thompson (1933) proposed assigning each new unit to an arm with the posterior
probability that the arm is the best one. With success probabilities
``θ_1, \\dots, θ_K``, independent priors ``θ_k \\sim \\mathrm{Beta}(a_k, b_k)`` and
``s_k`` successes and ``f_k`` failures observed on arm ``k``, the posterior of
``θ_k`` is ``\\mathrm{Beta}(a_k + s_k, b_k + f_k)`` and the policy assigns arm ``k``
with probability

```math
α_k = P\\bigl(θ_k = \\max_j θ_j \\mid \\text{data}\\bigr).
```

Arms that look promising are assigned more often, but every arm keeps positive
probability as long as its posterior overlaps with that of the leader, so the design
balances exploitation (outcomes of the participants in the experiment, i.e. low
regret) and exploration (learning). Russo et al. (2018) review its properties;
Agrawal and Goyal (2012) gave the first logarithmic regret bound, and Kaufmann, Korda
and Munos (2012) showed that it attains the Lai-Robbins lower bound on regret for
Bernoulli bandits. For inference,
however, the same concentration is a liability: the posterior probability of a
clearly inferior arm goes to zero quickly, so a `floor` is needed whenever the data
will be analysed (see [`AdaptivePolicy`](@ref); the default `floor = 0` gives no
positivity guarantee).

The probabilities ``α_k`` are computed from `ndraws` Monte Carlo posterior draws
(using the `rng` passed to [`assignment_probabilities`](@ref) or
[`assign!`](@ref)) or, with `ndraws = 0`, by deterministic numerical integration of
``\\int_0^1 \\prod_{j \\ne k} F_j(F_k^{-1}(u))\\, du`` on a midpoint grid of 256 points.
In both cases the arm is then drawn from exactly the recorded vector, so the log
contains the true propensity scores. Uniform priors (`a = b = 1`) are the usual
default; informative priors shift early allocation and should be pre-registered.

# Arguments
- `K::Integer`: number of arms (at least 2).

# Keywords
- `a`, `b`: prior parameters of the Beta distributions, scalars (common to all arms)
  or length-`K` vectors; both must be positive. Default `1.0` (uniform prior).
- `ndraws::Integer = 1000`: number of Monte Carlo posterior draws used to compute the
  assignment probabilities; `0` switches to deterministic numerical integration.
- `floor`, `floor_decay`, `burnin`: probability floor ``c\\, t^{-α}`` and
  equal-allocation burn-in; see [`AdaptivePolicy`](@ref). The defaults (`0`) give no
  floor and no burn-in.

# Returns
- A `BetaBernoulliThompson <: AdaptivePolicy`. Its fields hold the configuration
  and the success and failure counts; they are internal, and the policy is queried
  through [`assignment_probabilities`](@ref) and [`n_arms`](@ref).

# Examples
```julia
using DrSnow, StableRNGs
p = BetaBernoulliThompson(3; floor=0.1, burnin=30)
lg = run_adaptive_experiment(p, BernoulliBandit([0.3, 0.4, 0.5]), 500;
                             rng=StableRNG(1))
naive_arm_means(lg)
```

# References
- Thompson, W. R. (1933). On the likelihood that one unknown probability exceeds
  another in view of the evidence of two samples. *Biometrika*, 25(3/4), 285–294.
- Agrawal, S., & Goyal, N. (2012). Analysis of Thompson sampling for the multi-armed
  bandit problem. In *Proceedings of the 25th Annual Conference on Learning Theory*,
  PMLR 23, 39.1–39.26.
- Kaufmann, E., Korda, N., & Munos, R. (2012). Thompson sampling: An asymptotically
  optimal finite-time analysis. In *Algorithmic Learning Theory* (Lecture Notes in
  Computer Science, Vol. 7568, pp. 199–213). Springer.
- Russo, D., Van Roy, B., Kazerouni, A., Osband, I., & Wen, Z. (2018). A tutorial
  on Thompson sampling. *Foundations and Trends in Machine Learning*, 11(1), 1–96.
- Offer-Westort, M., Coppock, A., & Green, D. P. (2021). Adaptive experimental
  design: Prospects and applications in political science. *American Journal of
  Political Science*, 65(4), 826–844.
"""
mutable struct BetaBernoulliThompson <: AdaptivePolicy
    settings::_AdSettings
    a::Vector{Float64}
    b::Vector{Float64}
    ndraws::Int
    successes::Vector{Float64}
    failures::Vector{Float64}
end

function BetaBernoulliThompson(K::Integer; a=1.0, b=1.0, ndraws::Integer=1000,
                               floor::Real=0.0, floor_decay::Real=0.0,
                               burnin::Integer=0)
    s = _ad_settings(K, floor, floor_decay, burnin)
    av = a isa Real ? fill(float(a), K) : float.(collect(a))
    bv = b isa Real ? fill(float(b), K) : float.(collect(b))
    length(av) == K && length(bv) == K ||
        throw(DimensionMismatch("a and b must be scalars or have length K"))
    all(>(0), av) && all(>(0), bv) || throw(ArgumentError("a and b must be positive"))
    ndraws >= 0 || throw(ArgumentError("ndraws must be non-negative"))
    return BetaBernoulliThompson(s, av, bv, Int(ndraws), zeros(K), zeros(K))
end

function _ad_update!(p::BetaBernoulliThompson, arm::Int, y::Float64, _)
    (y == 0 || y == 1) ||
        throw(ArgumentError("BetaBernoulliThompson needs 0/1 outcomes, got $y"))
    y == 1 ? (p.successes[arm] += 1) : (p.failures[arm] += 1)
    return p
end

_ad_posteriors(p::BetaBernoulliThompson) =
    [Beta(p.a[k] + p.successes[k], p.b[k] + p.failures[k]) for k in 1:p.settings.K]

function _ad_raw_probabilities(p::BetaBernoulliThompson, _, rng)
    post = _ad_posteriors(p)
    K = length(post)
    if p.ndraws > 0
        D = Matrix{Float64}(undef, p.ndraws, K)
        for k in 1:K
            D[:, k] .= rand(rng, post[k], p.ndraws)
        end
        return _ad_prob_best_draws(D)
    end
    # P(k best) = ∫₀¹ Π_{j≠k} F_j(F_k⁻¹(u)) du on a midpoint grid.
    out = zeros(K)
    for k in 1:K, i in 1:_AD_NGRID
        v = quantile(post[k], (i - 0.5) / _AD_NGRID)
        pr = 1.0
        for j in 1:K
            j == k || (pr *= cdf(post[j], v))
        end
        out[k] += pr / _AD_NGRID
    end
    return sum(out) > 0 ? out ./ sum(out) : fill(1.0 / K, K)
end

_ad_policy_name(::BetaBernoulliThompson) = "Beta-Bernoulli Thompson sampling"

# ---------------------------------------------------------------------------
# Gaussian Thompson sampling
# ---------------------------------------------------------------------------

"""
    GaussianThompson(K; prior_mean=0.0, prior_var=1.0, noise_var=nothing,
                     floor=0.0, floor_decay=0.0, burnin=0) -> GaussianThompson

Thompson sampling for continuous outcomes with independent normal priors on the arm
means and normal outcome noise.

Each arm ``k`` has mean ``μ_k`` with prior ``N(m_0, v_0)`` (`prior_mean`,
`prior_var`), and outcomes are modelled as ``Y \\mid A = k \\sim N(μ_k, σ_k^2)``.
After ``n_k`` outcomes with mean ``\\bar y_k`` the posterior of ``μ_k`` is normal with

```math
\\text{precision } τ_k = \\frac{1}{v_0} + \\frac{n_k}{σ_k^2}, \\qquad
\\text{mean } \\frac{m_0 / v_0 + n_k \\bar y_k / σ_k^2}{τ_k},
```

and the policy assigns arm ``k`` with its posterior probability of having the largest
mean, computed by deterministic numerical integration over 256 equal-probability
cells. With `noise_var = nothing` the noise variance ``σ_k^2`` of each arm is its
sample variance (pooled over arms while an arm has fewer than two outcomes, and `1`
before any arm has two); the resulting posterior is then an approximation that
ignores the uncertainty in ``σ_k^2``. This is the agent used in the simulation
studies of Hadad et al. (2021).

The normal model is a working model for allocation only: the validity of the
downstream analysis ([`adaptive_arm_values`](@ref)) does not depend on it, because
that analysis uses only the recorded assignment probabilities. Misspecification
affects how efficiently the design allocates units, not the validity of inference.
As with every policy, the default `floor = 0` provides no positivity guarantee;
Hadad et al. (2021) pair this agent with a floor decaying as ``t^{-0.7}`` (see
[`AdaptivePolicy`](@ref)). The prior should be on the scale of the outcome: with
outcomes far from `0` and `prior_var = 1`, early allocation is driven by the prior.

# Arguments
- `K::Integer`: number of arms (at least 2).

# Keywords
- `prior_mean = 0.0`, `prior_var = 1.0`: prior mean and variance of the arm means,
  scalars or length-`K` vectors; `prior_var` must be positive.
- `noise_var = nothing`: known outcome noise variance (a positive scalar common to
  all arms), or `nothing` to estimate it from the data as described above.
- `floor`, `floor_decay`, `burnin`: probability floor ``c\\, t^{-α}`` and
  equal-allocation burn-in; see [`AdaptivePolicy`](@ref). The defaults (`0`) give no
  floor and no burn-in.

# Returns
- A `GaussianThompson <: AdaptivePolicy`. Its fields hold the configuration and
  each arm's count, sum and sum of squares; they are internal.

# Examples
```julia
using DrSnow, StableRNGs
p = GaussianThompson(3; floor=1/3, floor_decay=0.7, burnin=15)
lg = run_adaptive_experiment(p, GaussianBandit([0.9, 1.0, 1.1]), 1000;
                             rng=StableRNG(7))
adaptive_arm_values(lg)
```

# References
- Thompson, W. R. (1933). On the likelihood that one unknown probability exceeds
  another in view of the evidence of two samples. *Biometrika*, 25(3/4), 285–294.
- Russo, D., Van Roy, B., Kazerouni, A., Osband, I., & Wen, Z. (2018). A tutorial
  on Thompson sampling. *Foundations and Trends in Machine Learning*, 11(1), 1–96.
- Hadad, V., Hirshberg, D. A., Zhan, R., Wager, S., & Athey, S. (2021). Confidence
  intervals for policy evaluation in adaptive experiments. *Proceedings of the
  National Academy of Sciences*, 118(15), e2014602118.
- Lattimore, T., & Szepesvári, C. (2020). *Bandit Algorithms*. Cambridge University
  Press.
"""
mutable struct GaussianThompson <: AdaptivePolicy
    settings::_AdSettings
    prior_mean::Vector{Float64}
    prior_var::Vector{Float64}
    noise_var::Union{Nothing,Float64}
    n::Vector{Int}
    sum::Vector{Float64}
    sumsq::Vector{Float64}
end

function GaussianThompson(K::Integer; prior_mean=0.0, prior_var=1.0, noise_var=nothing,
                          floor::Real=0.0, floor_decay::Real=0.0, burnin::Integer=0)
    s = _ad_settings(K, floor, floor_decay, burnin)
    pm = prior_mean isa Real ? fill(float(prior_mean), K) : float.(collect(prior_mean))
    pv = prior_var isa Real ? fill(float(prior_var), K) : float.(collect(prior_var))
    length(pm) == K && length(pv) == K ||
        throw(DimensionMismatch("prior_mean and prior_var must be scalars or length K"))
    all(>(0), pv) || throw(ArgumentError("prior_var must be positive"))
    nv = noise_var === nothing ? nothing : float(noise_var)
    (nv === nothing || nv > 0) || throw(ArgumentError("noise_var must be positive"))
    return GaussianThompson(s, pm, pv, nv, zeros(Int, K), zeros(K), zeros(K))
end

function _ad_update!(p::GaussianThompson, arm::Int, y::Float64, _)
    p.n[arm] += 1
    p.sum[arm] += y
    p.sumsq[arm] += y^2
    return p
end

function _ad_noise_vars(p::GaussianThompson)
    K = p.settings.K
    p.noise_var === nothing || return fill(p.noise_var, K)
    ok = p.n .>= 2
    ss(k) = p.sumsq[k] - p.sum[k]^2 / p.n[k]
    pooled = any(ok) ? max(sum(ss(k) for k in 1:K if ok[k]) /
                           sum(p.n[k] - 1 for k in 1:K if ok[k]), 1e-12) : 1.0
    return [ok[k] ? max(ss(k) / (p.n[k] - 1), 1e-12) : pooled for k in 1:K]
end

function _ad_posterior_normal(p::GaussianThompson)
    σ2 = _ad_noise_vars(p)
    prec = 1 ./ p.prior_var .+ p.n ./ σ2
    m = (p.prior_mean ./ p.prior_var .+ p.sum ./ σ2) ./ prec
    return m, sqrt.(1 ./ prec)
end

_ad_raw_probabilities(p::GaussianThompson, _, _rng) =
    _ad_prob_best_normal(_ad_posterior_normal(p)...)

_ad_policy_name(::GaussianThompson) = "Gaussian Thompson sampling"

# ---------------------------------------------------------------------------
# Top-two Thompson sampling
# ---------------------------------------------------------------------------

"""
    TopTwoThompson(base; beta=0.5) -> TopTwoThompson

Top-two Thompson sampling (Russo 2020), a variant of Thompson sampling designed to
identify the best arm rather than to maximize outcomes during the experiment.

Plain Thompson sampling concentrates almost all units on the arm that currently
looks best, so the runner-up, whose comparison with the leader decides which arm is
best, receives few units. Top-two sampling plays the posterior leader with
probability ``β`` and otherwise re-samples from the posterior until a *different*
arm is best and plays that arm. With ``α_k`` the posterior probability that arm
``k`` is best (computed by the `base` policy), the implied assignment probability is

```math
e(k) = β\\, α_k + (1 - β)\\, α_k \\sum_{j \\ne k} \\frac{α_j}{1 - α_j},
```

which DrSnow computes exactly (no re-sampling loop), so the recorded probabilities
are the true propensity scores. Russo (2020) shows that, with an appropriately tuned
``β``, the posterior probability that some arm other than the true best is optimal
converges to zero at the best exponential rate achievable by any allocation rule;
``β = 1/2`` is the customary default. The design therefore spends more of the sample
on the runner-up than plain Thompson sampling, which improves power for comparing
the leading arms at the cost of more regret during the experiment.

Use top-two sampling when the goal of the experiment is to learn which arm is best
(or to estimate the best arm's advantage over the runner-up) for a decision taken
after the experiment; Kasy and Sautmann (2021) study a closely related modification
of Thompson sampling for policy choice. Floors and burn-in are those of `base`
(the default `floor = 0` of the base policy provides no positivity guarantee; see
[`AdaptivePolicy`](@ref)). When one arm has posterior probability of being best
equal to one, the rule falls back to the base probabilities.

# Arguments
- `base`: a [`BetaBernoulliThompson`](@ref) or [`GaussianThompson`](@ref) policy
  that supplies the posterior, the floor and the burn-in.

# Keywords
- `beta::Real = 0.5`: probability of playing the posterior leader, in `(0, 1)`.
  Larger values move the design towards plain Thompson sampling.

# Returns
- A `TopTwoThompson <: AdaptivePolicy` wrapping `base` (fields `base`, `beta` and the
  shared settings).

# Examples
```julia
using DrSnow, StableRNGs
p = TopTwoThompson(GaussianThompson(4; floor=0.02); beta=0.5)
lg = run_adaptive_experiment(p, GaussianBandit([0.0, 0.1, 0.2, 0.3]), 400;
                             batch_size=20, rng=StableRNG(2))
lg.probabilities[end, :]
```

# References
- Russo, D. (2020). Simple Bayesian algorithms for best-arm identification.
  *Operations Research*, 68(6), 1625–1647.
- Kasy, M., & Sautmann, A. (2021). Adaptive treatment assignment in experiments for
  policy choice. *Econometrica*, 89(1), 113–132.
- Russo, D., Van Roy, B., Kazerouni, A., Osband, I., & Wen, Z. (2018). A tutorial
  on Thompson sampling. *Foundations and Trends in Machine Learning*, 11(1), 1–96.
"""
mutable struct TopTwoThompson{B<:AdaptivePolicy} <: AdaptivePolicy
    base::B
    beta::Float64
    settings::_AdSettings
end

function TopTwoThompson(base::Union{BetaBernoulliThompson,GaussianThompson};
                        beta::Real=0.5)
    0 < beta < 1 || throw(ArgumentError("beta must be in (0, 1), got $beta"))
    return TopTwoThompson(base, float(beta), base.settings)
end

_ad_update!(p::TopTwoThompson, arm::Int, y::Float64, x) = _ad_update!(p.base, arm, y, x)

function _ad_raw_probabilities(p::TopTwoThompson, x, rng)
    α = _ad_raw_probabilities(p.base, x, rng)
    K = length(α)
    maximum(α) >= 1 - 1e-12 && return α          # resampling would never stop
    out = [p.beta * α[k] + (1 - p.beta) * α[k] *
           sum(α[j] / (1 - α[j]) for j in 1:K if j != k) for k in 1:K]
    return out ./ sum(out)
end

_ad_policy_name(p::TopTwoThompson) = "Top-two " * _ad_policy_name(p.base)

# ---------------------------------------------------------------------------
# ε-greedy, softmax, UCB
# ---------------------------------------------------------------------------

"""
    EpsilonGreedy(K; epsilon=0.1, floor=0.0, floor_decay=0.0, burnin=0)
        -> EpsilonGreedy

ε-greedy allocation: the arm with the highest sample mean is assigned with
probability ``1 - ε`` and a uniformly random arm with probability ``ε``.

With ``\\bar y_k`` the running sample mean of arm ``k`` and ``\\mathcal{B}`` the set of
arms attaining ``\\max_k \\bar y_k`` (ties are split equally), the assignment
probabilities are

```math
e(k) = \\frac{ε}{K}
+ (1 - ε)\\, \\frac{\\mathbf{1}\\{k \\in \\mathcal{B}\\}}{|\\mathcal{B}|}.
```

Arms without outcomes yet are treated as best, so each arm is tried early. The rule
is the simplest way to trade off exploitation and exploration (Sutton and Barto 2018,
Ch. 2): unlike Thompson sampling it does not adapt the amount of exploration to the
uncertainty about the arm means, so it keeps spending ``ε (K-1)/K`` of the sample on
arms known to be inferior (linear regret for fixed ``ε``), but it gives every arm the
explicit probability bound ``e(k) \\ge ε / K``. With `epsilon > 0` positivity
therefore holds by construction even with the default `floor = 0`; with
`epsilon = 0` the rule is purely greedy and deterministic given the data, so
inverse-probability estimators are undefined unless a floor is set (see
[`AdaptivePolicy`](@ref)).

# Arguments
- `K::Integer`: number of arms (at least 2).

# Keywords
- `epsilon::Real = 0.1`: exploration probability, in `[0, 1]`; `1` is uniform
  random assignment, `0` is greedy.
- `floor`, `floor_decay`, `burnin`: additional probability floor ``c\\, t^{-α}`` and
  equal-allocation burn-in; see [`AdaptivePolicy`](@ref).

# Returns
- An `EpsilonGreedy <: AdaptivePolicy`. Its fields hold the configuration and each
  arm's count and sum; they are internal.

# Examples
```julia
using DrSnow, StableRNGs
p = EpsilonGreedy(3; epsilon=0.2)
lg = run_adaptive_experiment(p, GaussianBandit([0.0, 0.2, 0.4]), 600;
                             rng=StableRNG(3))
adaptive_arm_values(lg; weights=:constant_allocation)
```

# References
- Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction*
  (2nd ed.). MIT Press.
- Lattimore, T., & Szepesvári, C. (2020). *Bandit Algorithms*. Cambridge University
  Press.
"""
mutable struct EpsilonGreedy <: AdaptivePolicy
    settings::_AdSettings
    epsilon::Float64
    n::Vector{Int}
    sum::Vector{Float64}
end

function EpsilonGreedy(K::Integer; epsilon::Real=0.1, floor::Real=0.0,
                       floor_decay::Real=0.0, burnin::Integer=0)
    0 <= epsilon <= 1 || throw(ArgumentError("epsilon must be in [0, 1]"))
    return EpsilonGreedy(_ad_settings(K, floor, floor_decay, burnin), float(epsilon),
                         zeros(Int, K), zeros(K))
end

"""
    SoftmaxPolicy(K; temperature=0.1, floor=0.0, floor_decay=0.0, burnin=0)
        -> SoftmaxPolicy

Softmax (Boltzmann) allocation: each arm is assigned with probability proportional to
the exponential of its sample mean divided by a temperature.

With ``\\bar y_k`` the running sample mean of arm ``k`` and temperature ``τ``,

```math
e(k) = \\frac{\\exp(\\bar y_k / τ)}{\\sum_{j=1}^K \\exp(\\bar y_j / τ)}.
```

Arms without outcomes yet are given the largest observed mean (`0` before any
outcome), so they are tried early. The temperature is in outcome units: as
``τ \\to 0`` the rule becomes greedy, and as ``τ \\to \\infty`` it approaches uniform
random assignment (Sutton and Barto 2018, Ch. 2). Unlike ε-greedy, softmax explores
arms in proportion to how good they look, but, like ε-greedy, it ignores how
uncertain the sample means are.

Every arm has positive probability, but the probability of an arm whose mean is
``Δ`` below the leader is of order ``\\exp(-Δ/τ)`` and can be tiny for small
temperatures or large outcome scales; with the default `floor = 0` there is no
explicit positivity bound, so set a floor when the data will be analysed (see
[`AdaptivePolicy`](@ref)).

# Arguments
- `K::Integer`: number of arms (at least 2).

# Keywords
- `temperature::Real = 0.1`: positive temperature ``τ``, in the units of the
  outcome; smaller values are greedier. Choose it relative to the plausible
  differences between arm means.
- `floor`, `floor_decay`, `burnin`: probability floor ``c\\, t^{-α}`` and
  equal-allocation burn-in; see [`AdaptivePolicy`](@ref). The defaults (`0`) give no
  floor and no burn-in.

# Returns
- A `SoftmaxPolicy <: AdaptivePolicy`. Its fields hold the configuration and each
  arm's count and sum; they are internal.

# Examples
```julia
using DrSnow, StableRNGs
p = SoftmaxPolicy(3; temperature=0.05, floor=0.02)
lg = run_adaptive_experiment(p, BernoulliBandit([0.3, 0.35, 0.45]), 500;
                             rng=StableRNG(4))
lg.probabilities[end, :]
```

# References
- Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction*
  (2nd ed.). MIT Press.
- Lattimore, T., & Szepesvári, C. (2020). *Bandit Algorithms*. Cambridge University
  Press.
"""
mutable struct SoftmaxPolicy <: AdaptivePolicy
    settings::_AdSettings
    temperature::Float64
    n::Vector{Int}
    sum::Vector{Float64}
end

function SoftmaxPolicy(K::Integer; temperature::Real=0.1, floor::Real=0.0,
                       floor_decay::Real=0.0, burnin::Integer=0)
    temperature > 0 || throw(ArgumentError("temperature must be positive"))
    return SoftmaxPolicy(_ad_settings(K, floor, floor_decay, burnin), float(temperature),
                         zeros(Int, K), zeros(K))
end

"""
    UCBPolicy(K; c=2.0, floor=0.0, floor_decay=0.0, burnin=0) -> UCBPolicy

Upper-confidence-bound allocation (UCB1): the arm with the largest optimistic index
``\\bar y_k + \\sqrt{c \\log(n) / n_k}`` is assigned.

Here ``\\bar y_k`` and ``n_k`` are the sample mean and number of outcomes of arm
``k`` and ``n = \\sum_k n_k``; arms without outcomes are assigned first, and ties are
split equally. The index follows the principle of optimism in the face of
uncertainty: an arm is played either because its mean is high or because it has been
sampled so rarely that its mean might be high. With `c = 2` and outcomes in
``[0, 1]`` this is the UCB1 rule of Auer, Cesa-Bianchi and Fischer (2002), whose
regret grows logarithmically in the number of units; for outcomes on another scale
`c` should grow with the square of the outcome range. Lattimore and Szepesvári (2020)
treat the family in depth.

UCB is **deterministic given the data**: without a floor every arm other than the
chosen one has assignment probability exactly zero, so inverse-probability and
doubly-robust estimators ([`adaptive_arm_values`](@ref),
[`adaptive_policy_value`](@ref), [`off_policy_value`](@ref)) are not defined. Since
the default is `floor = 0`, set `floor > 0` whenever the data will be analysed;
the floor randomizes the assignment and makes the recorded probabilities valid
propensity scores (see [`AdaptivePolicy`](@ref)).

# Arguments
- `K::Integer`: number of arms (at least 2).

# Keywords
- `c::Real = 2.0`: non-negative exploration constant; `0` is greedy.
- `floor`, `floor_decay`, `burnin`: probability floor ``c_f\\, t^{-α}`` (keyword
  `floor`) and equal-allocation burn-in; see [`AdaptivePolicy`](@ref). The defaults
  (`0`) give no floor and no burn-in.

# Returns
- A `UCBPolicy <: AdaptivePolicy`. Its fields hold the configuration and each arm's
  count and sum; they are internal.

# Examples
```julia
using DrSnow, StableRNGs
p = UCBPolicy(3; floor=0.05)
lg = run_adaptive_experiment(p, BernoulliBandit([0.3, 0.4, 0.5]), 500;
                             rng=StableRNG(5))
last(cumulative_regret(lg))
```

# References
- Auer, P., Cesa-Bianchi, N., & Fischer, P. (2002). Finite-time analysis of the
  multiarmed bandit problem. *Machine Learning*, 47(2–3), 235–256.
- Lattimore, T., & Szepesvári, C. (2020). *Bandit Algorithms*. Cambridge University
  Press.
"""
mutable struct UCBPolicy <: AdaptivePolicy
    settings::_AdSettings
    c::Float64
    n::Vector{Int}
    sum::Vector{Float64}
end

function UCBPolicy(K::Integer; c::Real=2.0, floor::Real=0.0, floor_decay::Real=0.0,
                   burnin::Integer=0)
    c >= 0 || throw(ArgumentError("c must be non-negative"))
    return UCBPolicy(_ad_settings(K, floor, floor_decay, burnin), float(c),
                     zeros(Int, K), zeros(K))
end

const _AdMeanPolicy = Union{EpsilonGreedy,SoftmaxPolicy,UCBPolicy}

function _ad_update!(p::_AdMeanPolicy, arm::Int, y::Float64, _)
    p.n[arm] += 1
    p.sum[arm] += y
    return p
end

function _ad_raw_probabilities(p::EpsilonGreedy, _, _rng)
    K = p.settings.K
    best = _ad_best_arms(p.n, p.sum)
    out = fill(p.epsilon / K, K)
    out[best] .+= (1 - p.epsilon) / length(best)
    return out
end

function _ad_raw_probabilities(p::SoftmaxPolicy, _, _rng)
    obs, μ = _ad_running_means(p.n, p.sum)
    fill_value = any(obs) ? maximum(μ[obs]) : 0.0
    v = [obs[k] ? μ[k] : fill_value for k in eachindex(μ)] ./ p.temperature
    w = exp.(v .- maximum(v))
    return w ./ sum(w)
end

function _ad_raw_probabilities(p::UCBPolicy, _, _rng)
    K = p.settings.K
    obs, μ = _ad_running_means(p.n, p.sum)
    idx = if !all(obs)
        findall(.!obs)
    else
        N = sum(p.n)
        u = μ .+ sqrt.(p.c * log(N) ./ p.n)
        findall(==(maximum(u)), u)
    end
    out = zeros(K)
    out[idx] .= 1 / length(idx)
    return out
end

_ad_policy_name(p::EpsilonGreedy) = "ε-greedy (ε = $(p.epsilon))"
_ad_policy_name(p::SoftmaxPolicy) = "Softmax (temperature = $(p.temperature))"
_ad_policy_name(p::UCBPolicy) = "UCB1 (c = $(p.c))"

# ---------------------------------------------------------------------------
# Linear (contextual) Thompson sampling
# ---------------------------------------------------------------------------

"""
    LinearThompson(K, d; prior_var=1.0, noise_var=1.0, intercept=true, floor=0.0,
                   floor_decay=0.0, burnin=0) -> LinearThompson

Contextual Thompson sampling with a Bayesian linear outcome model for each arm: each
unit is assigned arm ``k`` with the posterior probability that ``k`` has the highest
expected outcome at the unit's covariates.

For covariates ``x \\in \\mathbb{R}^d`` let ``z = (1, x')'`` (or ``z = x`` with
`intercept = false`). Arm ``k`` is modelled as ``Y = z'θ_k + ε``,
``ε \\sim N(0, σ^2)`` with known ``σ^2`` (`noise_var`) and prior
``θ_k \\sim N(0, v_0 I)`` (`prior_var`), independently across arms. The posterior of
``θ_k`` is normal with precision ``(σ^{-2}) (Z_k'Z_k + (σ^2 / v_0) I)``, where ``Z_k``
stacks the ``z`` of the units assigned to ``k``, so at context ``x`` the posterior
arm means ``z'θ_k`` are independent normals and the policy assigns

```math
e_t(x, k) = P\\bigl(z'θ_k = \\max_j z'θ_j \\mid \\text{data before the batch}\\bigr),
```

computed by deterministic numerical integration. The algorithm is the linear-payoff
contextual Thompson sampling of Agrawal and Goyal (2013), with the exact probability
of being best in place of a single posterior draw so that the propensity score is
available in closed form.

Because the probabilities are deterministic functions of the policy state and of
``x``, DrSnow can evaluate the probability the design *would have* used at any
covariate value; logs of contextual experiments store a snapshot of the policy at
the start of every batch for this purpose, which the contextual adaptive weights of
[`adaptive_policy_value`](@ref) (Zhan et al. 2021) require. The linear model is a
working model for allocation only; inference uses the recorded probabilities and
does not assume it. As with every policy, the default `floor = 0` provides no
positivity guarantee: probabilities can become very small for covariate profiles
where one arm clearly dominates. Zhan et al. (2021) assume a floor decaying no faster
than ``t^{-α}`` with ``α < 1/2`` (see [`AdaptivePolicy`](@ref)); a burn-in of a few
times ``K (d + 1)`` units stabilizes the early regressions. Athey et al. (2022)
report a survey experiment run with a contextual bandit whose assignment
probabilities were bounded below by slowly decaying floors, balancing outcomes
within the experiment against learning a targeting policy afterwards.

# Arguments
- `K::Integer`: number of arms (at least 2).
- `d::Integer`: number of covariates in the context vector (at least 1).

# Keywords
- `prior_var::Real = 1.0`: prior variance ``v_0`` of each regression coefficient.
- `noise_var::Real = 1.0`: known outcome noise variance ``σ^2``; it scales how fast
  the posterior concentrates.
- `intercept::Bool = true`: include an intercept in each arm's regression.
- `floor`, `floor_decay`, `burnin`: probability floor ``c\\, t^{-α}`` and
  equal-allocation burn-in; see [`AdaptivePolicy`](@ref). The defaults (`0`) give no
  floor and no burn-in.

# Returns
- A `LinearThompson <: AdaptivePolicy`. Its fields hold the configuration and each
  arm's posterior precision and score vector; they are internal.

# Examples
```julia
using DrSnow
p = LinearThompson(4, 3; floor=0.25, floor_decay=0.4, burnin=200)
assignment_probabilities(p, 1; context=[0.1, -0.3, 1.2])     # burn-in: 1/4 each
update_policy!(p, 2, 1.5; context=[0.1, -0.3, 1.2])
assignment_probabilities(p, 201; context=[0.1, -0.3, 1.2])
```

# References
- Agrawal, S., & Goyal, N. (2013). Thompson sampling for contextual bandits with
  linear payoffs. In *Proceedings of the 30th International Conference on Machine
  Learning*, PMLR 28(3), 127–135.
- Zhan, R., Hadad, V., Hirshberg, D. A., & Athey, S. (2021). Off-policy evaluation
  via adaptive weighting with data from contextual bandits. In *Proceedings of the
  27th ACM SIGKDD Conference on Knowledge Discovery and Data Mining* (pp. 2125–2135).
- Athey, S., Byambadalai, U., Hadad, V., Krishnamurthy, S. K., Leung, W., &
  Williams, J. J. (2022). Contextual bandits in a survey experiment on charitable
  giving: Within-experiment outcomes versus policy learning. arXiv:2211.12004.
- Russo, D., Van Roy, B., Kazerouni, A., Osband, I., & Wen, Z. (2018). A tutorial
  on Thompson sampling. *Foundations and Trends in Machine Learning*, 11(1), 1–96.
"""
mutable struct LinearThompson <: AdaptivePolicy
    settings::_AdSettings
    d::Int
    prior_var::Float64
    noise_var::Float64
    intercept::Bool
    A::Vector{Matrix{Float64}}     # posterior precision (times noise_var)
    b::Vector{Vector{Float64}}     # Σ z y
end

function LinearThompson(K::Integer, d::Integer; prior_var::Real=1.0,
                        noise_var::Real=1.0, intercept::Bool=true, floor::Real=0.0,
                        floor_decay::Real=0.0, burnin::Integer=0)
    s = _ad_settings(K, floor, floor_decay, burnin)
    d >= 1 || throw(ArgumentError("d must be at least 1"))
    prior_var > 0 && noise_var > 0 ||
        throw(ArgumentError("prior_var and noise_var must be positive"))
    q = d + intercept
    A = [Matrix{Float64}(I, q, q) .* (noise_var / prior_var) for _ in 1:K]
    return LinearThompson(s, Int(d), float(prior_var), float(noise_var), intercept, A,
                          [zeros(q) for _ in 1:K])
end

_ad_is_contextual(::LinearThompson) = true
_ad_dim(p::LinearThompson) = p.d
_ad_z(p::LinearThompson, x) = p.intercept ? vcat(1.0, x) : x

function _ad_update!(p::LinearThompson, arm::Int, y::Float64, x)
    z = _ad_z(p, x)
    p.A[arm] .+= z * z'
    p.b[arm] .+= z .* y
    return p
end

function _ad_raw_probabilities(p::LinearThompson, x, _rng)
    z = _ad_z(p, x)
    K = p.settings.K
    m = zeros(K)
    s = zeros(K)
    for k in 1:K
        F = cholesky(Symmetric(p.A[k]))
        m[k] = dot(z, F \ p.b[k])
        s[k] = sqrt(max(p.noise_var * dot(z, F \ z), 0.0))
    end
    return _ad_prob_best_normal(m, s)
end

_ad_policy_name(::LinearThompson) = "Linear Thompson sampling (contextual)"

function Base.show(io::IO, p::AdaptivePolicy)
    s = p.settings
    print(io, _ad_policy_name(p), " with K = ", s.K, " arms")
    s.floor > 0 && print(io, ", floor ", s.floor,
                         s.floor_decay > 0 ? " t^-$(s.floor_decay)" : "")
    s.burnin > 0 && print(io, ", burn-in ", s.burnin)
end
