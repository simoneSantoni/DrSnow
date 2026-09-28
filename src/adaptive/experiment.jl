# Running adaptive experiments: bandit environments for simulation, a step-through
# deployment object (`AdaptiveExperiment`: assign a batch, observe its outcomes), a
# simulation runner and the resulting log (`AdaptiveLog`) that inference uses.

"""
    BanditEnvironment

Abstract supertype of the simulated outcome models used by
[`run_adaptive_experiment`](@ref) to study adaptive designs before they are fielded:
[`BernoulliBandit`](@ref), [`GaussianBandit`](@ref) and [`ContextualBandit`](@ref).

An environment specifies, for every unit, the expected outcome ``μ(x, w)`` of each
arm ``w`` at the unit's covariates ``x`` and the distribution of the realized outcome
around it. Units are independent and identically distributed, and each unit's
outcome depends only on its own arm (no interference, no drift), which are the
conditions under which the estimators of the adaptive area are justified. Because
the true means are known, a simulated [`AdaptiveLog`](@ref) stores them, so that
regret ([`cumulative_regret`](@ref)), bias and coverage of the analysis can be
computed. Simulating the planned design under several plausible configurations of
arm means is a practical way to choose a policy, a floor and a burn-in;
Offer-Westort, Coppock and Green (2021) compare adaptive and static designs in this
way.

# References
- Offer-Westort, M., Coppock, A., & Green, D. P. (2021). Adaptive experimental
  design: Prospects and applications in political science. *American Journal of
  Political Science*, 65(4), 826–844.
- Lattimore, T., & Szepesvári, C. (2020). *Bandit Algorithms*. Cambridge University
  Press.
"""
abstract type BanditEnvironment end

"""
    BernoulliBandit(means) -> BernoulliBandit

Simulated environment with binary outcomes: a unit assigned arm `k` succeeds
(outcome `1`) with probability `means[k]`, independently of all other units.

This is the canonical environment for experiments with a binary response (a click,
a donation, a vote). The arm values ``Q(k) = E[Y(k)]`` are the success probabilities,
so the estimates of [`adaptive_arm_values`](@ref) can be compared with `means`
directly. Pair it with [`BetaBernoulliThompson`](@ref) or any other policy.

# Arguments
- `means::AbstractVector{<:Real}`: success probability of each arm, in `[0, 1]`;
  at least two arms.

# Returns
- A `BernoulliBandit <: BanditEnvironment` with field `means::Vector{Float64}`.

# Examples
```julia
using DrSnow, StableRNGs
env = BernoulliBandit([0.3, 0.35, 0.5])
lg = run_adaptive_experiment(BetaBernoulliThompson(3; floor=0.05), env, 300;
                             rng=StableRNG(1))
```
"""
struct BernoulliBandit <: BanditEnvironment
    means::Vector{Float64}
    function BernoulliBandit(means::AbstractVector{<:Real})
        length(means) >= 2 || throw(ArgumentError("need at least two arms"))
        all(m -> 0 <= m <= 1, means) ||
            throw(ArgumentError("success probabilities must be in [0, 1]"))
        return new(float.(collect(means)))
    end
end

"""
    GaussianBandit(means; sd=1.0) -> GaussianBandit

Simulated environment with continuous outcomes: a unit assigned arm `k` yields
`means[k] + sd * Z` with `Z ~ N(0, 1)` drawn independently for every unit.

The arm values are ``Q(k) =`` `means[k]` and the noise is homoskedastic across arms.
The ratio of the gaps between arm means to `sd` governs how quickly an adaptive
design separates the arms, and hence how fast the probabilities of inferior arms
fall to their floor; simulating with a few signal-to-noise ratios shows how regret,
allocation and the width of the adaptive confidence intervals trade off.

# Arguments
- `means::AbstractVector{<:Real}`: expected outcome of each arm; at least two arms.

# Keywords
- `sd::Real = 1.0`: non-negative standard deviation of the outcome noise.

# Returns
- A `GaussianBandit <: BanditEnvironment` with fields `means::Vector{Float64}` and
  `sd::Float64`.

# Examples
```julia
using DrSnow, StableRNGs
env = GaussianBandit([0.5, 1.0, 1.5]; sd=1.0)
lg = run_adaptive_experiment(GaussianThompson(3; floor=0.05), env, 300;
                             rng=StableRNG(1))
```
"""
struct GaussianBandit <: BanditEnvironment
    means::Vector{Float64}
    sd::Float64
    function GaussianBandit(means::AbstractVector{<:Real}; sd::Real=1.0)
        length(means) >= 2 || throw(ArgumentError("need at least two arms"))
        sd >= 0 || throw(ArgumentError("sd must be non-negative"))
        return new(float.(collect(means)), float(sd))
    end
end

"""
    ContextualBandit(mean_fn, context_fn, K; noise_sd=1.0, outcome=:normal)
        -> ContextualBandit

Simulated environment with covariate-dependent arm means, for contextual policies
such as [`LinearThompson`](@ref).

Each unit's covariates are drawn as `x = context_fn(rng)`, independently across
units, and arm `k` has expected outcome ``μ(x, k) =`` `mean_fn(x)[k]`. The realized
outcome is ``μ(x, k)`` plus `noise_sd * N(0, 1)` noise (`outcome = :normal`) or a
Bernoulli draw with success probability ``μ(x, k)`` (`outcome = :bernoulli`). The
environment defines heterogeneous treatment effects, so the value of a targeting
policy ``π``, ``Q(π) = E[\\sum_w π(X, w) μ(X, w)]``, differs from the value of any
single arm; the optimal policy assigns ``\\arg\\max_k μ(x, k)``, and its value can be
estimated from the simulated log with [`adaptive_policy_value`](@ref).

# Arguments
- `mean_fn`: function `x -> Vector` returning the `K` expected outcomes at
  covariates `x` (probabilities for Bernoulli outcomes).
- `context_fn`: function `rng -> Vector` drawing one unit's covariate vector.
- `K::Integer`: number of arms (at least 2).

# Keywords
- `noise_sd::Real = 1.0`: standard deviation of normal outcome noise.
- `outcome::Symbol = :normal`: `:normal` or `:bernoulli`.

# Returns
- A `ContextualBandit <: BanditEnvironment` with fields `mean_fn`, `context_fn`,
  `K`, `noise_sd` and `outcome`.

# Examples
```julia
using DrSnow, StableRNGs
env = ContextualBandit(x -> [0.0, x[1], -x[1]], rng -> randn(rng, 2), 3)
lg = run_adaptive_experiment(LinearThompson(3, 2; floor=0.1, burnin=60), env, 300;
                             batch_size=50, rng=StableRNG(1))
```

# References
- Zhan, R., Hadad, V., Hirshberg, D. A., & Athey, S. (2021). Off-policy evaluation
  via adaptive weighting with data from contextual bandits. In *Proceedings of the
  27th ACM SIGKDD Conference on Knowledge Discovery and Data Mining* (pp. 2125–2135).
"""
struct ContextualBandit{F,G} <: BanditEnvironment
    mean_fn::F
    context_fn::G
    K::Int
    noise_sd::Float64
    outcome::Symbol
    function ContextualBandit(mean_fn::F, context_fn::G, K::Integer; noise_sd::Real=1.0,
                              outcome::Symbol=:normal) where {F,G}
        K >= 2 || throw(ArgumentError("need at least two arms"))
        outcome in (:normal, :bernoulli) ||
            throw(ArgumentError("outcome must be :normal or :bernoulli"))
        noise_sd >= 0 || throw(ArgumentError("noise_sd must be non-negative"))
        return new{F,G}(mean_fn, context_fn, Int(K), float(noise_sd), outcome)
    end
end

_ad_env_K(e::Union{BernoulliBandit,GaussianBandit}) = length(e.means)
_ad_env_K(e::ContextualBandit) = e.K
_ad_env_context(::Union{BernoulliBandit,GaussianBandit}, _rng) = nothing
function _ad_env_context(e::ContextualBandit, rng)
    x = Float64.(collect(e.context_fn(rng)))
    all(isfinite, x) || throw(ArgumentError("context_fn returned non-finite values"))
    return x
end
_ad_env_means(e::Union{BernoulliBandit,GaussianBandit}, _x) = e.means
function _ad_env_means(e::ContextualBandit, x)
    m = Float64.(collect(e.mean_fn(x)))
    length(m) == e.K || throw(DimensionMismatch("mean_fn must return $(e.K) means"))
    e.outcome === :bernoulli && !all(v -> 0 <= v <= 1, m) &&
        throw(ArgumentError("mean_fn must return probabilities for Bernoulli outcomes"))
    return m
end
_ad_env_draw(::BernoulliBandit, rng, μ) = float(rand(rng) < μ)
_ad_env_draw(e::GaussianBandit, rng, μ) = μ + e.sd * randn(rng)
_ad_env_draw(e::ContextualBandit, rng, μ) =
    e.outcome === :bernoulli ? float(rand(rng) < μ) : μ + e.noise_sd * randn(rng)

# ---------------------------------------------------------------------------
# The log
# ---------------------------------------------------------------------------

"""
    AdaptiveLog

Complete record of an adaptive experiment, produced by
[`run_adaptive_experiment`](@ref) or [`experiment_log`](@ref), and the input of the
adaptive analysis functions.

Valid inference after adaptive data collection needs more than arms and outcomes:
it needs, for every unit, the *full vector* of assignment probabilities from which
its arm was drawn, the order of arrival, and the batch structure (units of a batch
share one probability function). For contextual designs it also needs the
probabilities the design would have used at *other* covariate values, which is why
the log keeps a copy of the policy at the start of each batch. With this
information [`adaptive_arm_values`](@ref), [`adaptive_policy_value`](@ref),
[`bandit_dr_scores`](@ref) and [`off_policy_value`](@ref) can be applied directly;
[`naive_arm_means`](@ref) shows the biased sample means for comparison.

An `AdaptiveLog` is a Tables.jl table: `DataFrame(lg)` has columns `t`, `batch`,
`arm`, `outcome`, `p1 … pK` (assignment probabilities) and, for contextual runs, the
covariates `x1 … xd`. Saving this table is the minimal reproducible record of an
adaptive experiment; the table methods of the analysis functions accept it back.

# Fields
- `arms::Vector{Int}`: arm (in `1:K`) of each unit, in order of arrival.
- `outcomes::Vector{Float64}`: observed outcome of each unit.
- `probabilities::Matrix{Float64}`: `T × K`; row `t` is the distribution from which
  unit `t`'s arm was drawn.
- `contexts::Union{Nothing,Matrix{Float64}}`: `T × d` covariates (contextual runs).
- `batch::Vector{Int}`: batch of each unit.
- `batch_start::Vector{Int}`: index of the first unit of each batch.
- `snapshots::Vector`: copy of the policy at the start of each batch (contextual
  policies only), used to evaluate the probabilities the design would have used at
  other contexts.
- `true_means::Union{Nothing,Matrix{Float64}}`: `T × K` expected outcome of every arm
  for every unit (simulated experiments only).
- `policy::String`: description of the policy.
- `K::Int`: number of arms.
- `floor::Float64`, `floor_decay::Float64`, `burnin::Int`: the policy's floor
  ``c\\, t^{-α}`` and burn-in; `floor_decay` is the default decay rate used by the
  two-point weights of [`adaptive_arm_values`](@ref).

`nobs(lg)` returns the number of units.
"""
struct AdaptiveLog
    arms::Vector{Int}
    outcomes::Vector{Float64}
    probabilities::Matrix{Float64}
    contexts::Union{Nothing,Matrix{Float64}}
    batch::Vector{Int}
    batch_start::Vector{Int}
    snapshots::Vector{Any}
    true_means::Union{Nothing,Matrix{Float64}}
    policy::String
    K::Int
    floor::Float64
    floor_decay::Float64
    burnin::Int
end

StatsAPI.nobs(l::AdaptiveLog) = length(l.arms)

function Base.show(io::IO, ::MIME"text/plain", l::AdaptiveLog)
    T = nobs(l)
    println(io, "AdaptiveLog: ", T, " units, ", l.K, " arms, ", length(l.batch_start),
            " batch(es)")
    println(io, "Policy: ", l.policy)
    l.contexts === nothing || println(io, "Covariates: ", size(l.contexts, 2))
    counts = [count(==(k), l.arms) for k in 1:l.K]
    means = [counts[k] > 0 ? mean(l.outcomes[l.arms .== k]) : NaN for k in 1:l.K]
    println(io, "Arm  n     share   mean outcome  final P(assign)")
    for k in 1:l.K
        @printf(io, "%-4d %-5d %.3f   %-13.4g %.3f\n", k, counts[k], counts[k] / T,
                means[k], T > 0 ? l.probabilities[end, k] : NaN)
    end
    print(io, "Sample means are biased under adaptive assignment; see " *
              "adaptive_arm_values.")
end

Base.show(io::IO, l::AdaptiveLog) =
    print(io, "AdaptiveLog(", nobs(l), " units, ", l.K, " arms)")

function _ad_log_table(l::AdaptiveLog)
    T = nobs(l)
    df = DataFrame(t=collect(1:T), batch=l.batch, arm=l.arms, outcome=l.outcomes)
    for k in 1:l.K
        df[!, Symbol("p", k)] = l.probabilities[:, k]
    end
    if l.contexts !== nothing
        for j in axes(l.contexts, 2)
            df[!, Symbol("x", j)] = l.contexts[:, j]
        end
    end
    return df
end

Tables.istable(::Type{AdaptiveLog}) = true
Tables.columnaccess(::Type{AdaptiveLog}) = true
Tables.columns(l::AdaptiveLog) = Tables.columns(_ad_log_table(l))

# ---------------------------------------------------------------------------
# Step-through experiment
# ---------------------------------------------------------------------------

"""
    AdaptiveExperiment(policy::AdaptivePolicy) -> AdaptiveExperiment

A running adaptive experiment, for field deployment or step-by-step simulation, that
records everything needed for valid inference afterwards.

Participants arrive in batches. [`assign!`](@ref) computes the assignment
probabilities of the batch from the policy's current state (for contextual policies,
at each unit's covariates), draws each unit's arm from them and stores the full
probability vectors. [`observe!`](@ref) records the batch's outcomes and only then
updates the policy, so the probability function is constant within a batch, which
is how experiments with delayed outcomes are run in practice (Offer-Westort, Coppock
and Green 2021) and what the batched analyses assume. [`experiment_log`](@ref)
returns the [`AdaptiveLog`](@ref) for analysis.

Recording the probabilities at the moment of assignment is essential: they are the
propensity scores of the design, and inference after adaptive data collection
(Hadad et al. 2021; Zhan et al. 2021) is valid only when the analysis uses the
probabilities with which the arms were actually drawn. The experiment works on a copy
of `policy`, so the object passed in is not modified. As with every policy, set a
probability floor when the data will be analysed (the default `floor = 0` gives no
positivity guarantee; see [`AdaptivePolicy`](@ref)).

# Arguments
- `policy::AdaptivePolicy`: the assignment policy (copied).

# Returns
- An `AdaptiveExperiment`. `nobs(ex)` counts the units whose outcomes have been
  observed.

# Examples
```julia
using DrSnow, StableRNGs
ex = AdaptiveExperiment(BetaBernoulliThompson(2; floor=0.1))
rng = StableRNG(1)
for b in 1:20
    arms = assign!(ex, 25; rng=rng)                   # 25 participants per batch
    observe!(ex, [rand(rng) < (a == 2 ? 0.6 : 0.5) for a in arms])
end
lg = experiment_log(ex)
adaptive_arm_values(lg; weights=:constant_allocation)
```

# References
- Hadad, V., Hirshberg, D. A., Zhan, R., Wager, S., & Athey, S. (2021). Confidence
  intervals for policy evaluation in adaptive experiments. *Proceedings of the
  National Academy of Sciences*, 118(15), e2014602118.
- Zhan, R., Hadad, V., Hirshberg, D. A., & Athey, S. (2021). Off-policy evaluation
  via adaptive weighting with data from contextual bandits. In *Proceedings of the
  27th ACM SIGKDD Conference on Knowledge Discovery and Data Mining* (pp. 2125–2135).
- Offer-Westort, M., Coppock, A., & Green, D. P. (2021). Adaptive experimental
  design: Prospects and applications in political science. *American Journal of
  Political Science*, 65(4), 826–844.
"""
mutable struct AdaptiveExperiment{P<:AdaptivePolicy}
    policy::P
    arms::Vector{Int}
    outcomes::Vector{Float64}
    probabilities::Vector{Vector{Float64}}
    contexts::Vector{Vector{Float64}}
    batch::Vector{Int}
    batch_start::Vector{Int}
    snapshots::Vector{P}
    true_means::Vector{Vector{Float64}}
    n_pending::Int
end

AdaptiveExperiment(policy::AdaptivePolicy) =
    AdaptiveExperiment(deepcopy(policy), Int[], Float64[], Vector{Float64}[],
                       Vector{Float64}[], Int[], Int[], typeof(policy)[],
                       Vector{Float64}[], 0)

StatsAPI.nobs(e::AdaptiveExperiment) = length(e.outcomes)

function Base.show(io::IO, e::AdaptiveExperiment)
    print(io, "AdaptiveExperiment(", _ad_policy_name(e.policy), "; ", nobs(e),
          " outcomes, ", length(e.batch_start), " batches, ", e.n_pending, " pending)")
end

function _ad_context_rows(contexts, n)
    contexts === nothing && return fill(nothing, n)
    if contexts isa AbstractMatrix
        return [Float64.(collect(view(contexts, i, :))) for i in axes(contexts, 1)]
    end
    return [Float64.(collect(c)) for c in contexts]
end

"""
    assign!(ex::AdaptiveExperiment, n=1; contexts=nothing, rng=Random.default_rng())
        -> Vector{Int}

Assign arms to a new batch of `n` units and record the assignment probabilities.

The probabilities of every unit of the batch are computed from the policy's state at
the start of the batch, with [`assignment_probabilities`](@ref) evaluated at the
index of the batch's first unit (so burn-in and floor refer to that index); for
contextual policies they are evaluated at each unit's covariates and a snapshot of
the policy is stored. Each unit's arm is then drawn independently from its
probability vector. The outcomes of the batch must be recorded with
[`observe!`](@ref) before the next batch can be assigned; this enforces the batch
structure that the analysis relies on.

# Arguments
- `ex::AdaptiveExperiment`: the running experiment (mutated).
- `n::Integer = 1`: batch size; inferred from `contexts` when they are given.

# Keywords
- `contexts`: `n × d` matrix or vector of covariate vectors, one per unit; required
  for contextual policies and, once used, for every later batch.
- `rng::AbstractRNG = Random.default_rng()`: random-number generator for the arm
  draws (and Monte Carlo probabilities); pass a seeded generator for reproducible
  assignments.

# Returns
- `Vector{Int}`: the assigned arms, in the order of the units of the batch.

# Examples
```julia
using DrSnow, StableRNGs
ex = AdaptiveExperiment(GaussianThompson(3; floor=0.05))
arms = assign!(ex, 10; rng=StableRNG(2))
```
"""
function assign!(e::AdaptiveExperiment, n::Integer=1; contexts=nothing,
                 rng::AbstractRNG=Random.default_rng())
    e.n_pending == 0 || throw(ArgumentError(
        "assign!: $(e.n_pending) outcomes of the previous batch are pending; call " *
        "observe! first"))
    xs = _ad_context_rows(contexts, n)
    contexts === nothing || (n = length(xs))
    n >= 1 || throw(ArgumentError("assign!: batch size must be at least 1"))
    ctx = _ad_is_contextual(e.policy)
    ctx && contexts === nothing &&
        throw(ArgumentError("assign!: the policy is contextual; pass `contexts`"))
    if !isempty(e.contexts) && contexts === nothing
        throw(ArgumentError("assign!: earlier units had contexts; pass `contexts`"))
    end
    if isempty(e.contexts) && !isempty(e.arms) && contexts !== nothing
        throw(ArgumentError("assign!: earlier units had no contexts"))
    end
    t0 = length(e.arms) + 1
    b = length(e.batch_start) + 1
    push!(e.batch_start, t0)
    ctx && push!(e.snapshots, deepcopy(e.policy))
    K = n_arms(e.policy)
    arms = Vector{Int}(undef, n)
    # Non-contextual policies: one probability vector for the whole batch.
    pb = ctx ? nothing : assignment_probabilities(e.policy, t0; rng=rng)
    for i in 1:n
        p = ctx ? assignment_probabilities(e.policy, t0; context=xs[i], rng=rng) : pb
        a = _ad_draw_arm(rng, p)
        arms[i] = a
        push!(e.arms, a)
        push!(e.probabilities, p)
        push!(e.batch, b)
        xs[i] === nothing || push!(e.contexts, xs[i])
        length(p) == K || error("internal: probability vector of wrong length")
    end
    e.n_pending = n
    return arms
end

function _ad_draw_arm(rng::AbstractRNG, p::AbstractVector)
    u = rand(rng)
    c = 0.0
    for k in eachindex(p)
        c += p[k]
        u < c && return k
    end
    return findlast(>(0), p)
end

"""
    observe!(ex::AdaptiveExperiment, outcomes; means=nothing) -> ex

Record the outcomes of the pending batch and update the policy with them.

Outcomes are given in the order in which [`assign!`](@ref) returned the arms. The
policy is updated unit by unit with [`update_policy!`](@ref) only now, after the
whole batch was assigned, so the assignment probabilities of the batch did not depend
on any of its outcomes. In simulations, `means` may pass the expected outcome of
every arm for each unit; the log stores them for regret and bias computations
([`cumulative_regret`](@ref)).

# Arguments
- `ex::AdaptiveExperiment`: the running experiment with a pending batch (mutated).
- `outcomes::AbstractVector{<:Real}`: one finite outcome per pending unit.

# Keywords
- `means = nothing`: optional `n × K` matrix of true arm means for the units of the
  batch; pass it for every batch or for none.

# Returns
- The experiment `ex`.

# Examples
```julia
using DrSnow, StableRNGs
rng = StableRNG(3)
ex = AdaptiveExperiment(GaussianThompson(2; floor=0.1))
arms = assign!(ex, 5; rng=rng)
observe!(ex, [a == 2 ? 1.0 + randn(rng) : randn(rng) for a in arms])
```
"""
function observe!(e::AdaptiveExperiment, outcomes::AbstractVector; means=nothing)
    n = e.n_pending
    n > 0 || throw(ArgumentError("observe!: no pending assignments"))
    length(outcomes) == n ||
        throw(DimensionMismatch("observe!: expected $n outcomes, got $(length(outcomes))"))
    all(y -> y isa Real && isfinite(y), outcomes) ||
        throw(ArgumentError("observe!: outcomes must be finite numbers"))
    first_new = length(e.arms) - n + 1
    for i in 1:n
        t = first_new + i - 1
        x = isempty(e.contexts) ? nothing : e.contexts[t]
        update_policy!(e.policy, e.arms[t], outcomes[i]; context=x)
        push!(e.outcomes, float(outcomes[i]))
    end
    if means !== nothing
        size(means) == (n, n_arms(e.policy)) ||
            throw(DimensionMismatch("observe!: means must be n × K"))
        (length(e.true_means) == first_new - 1) ||
            throw(ArgumentError("observe!: pass `means` for every batch or none"))
        append!(e.true_means, [Float64.(collect(view(means, i, :))) for i in 1:n])
    end
    e.n_pending = 0
    return e
end

"""
    experiment_log(ex::AdaptiveExperiment) -> AdaptiveLog

The [`AdaptiveLog`](@ref) of a running experiment: arms, outcomes, the full
assignment-probability vectors, batches and (for contextual policies) policy
snapshots of all units whose outcomes have been observed.

A pending batch (assigned but not yet observed) is excluded. The log can be analysed
at any time, but the fixed-horizon confidence intervals of
[`adaptive_arm_values`](@ref) and [`adaptive_policy_value`](@ref) are valid only when
the analysis time does not depend on the estimates; repeatedly analysing an
interim log and stopping when an interval excludes zero inflates the error rate.

# Arguments
- `ex::AdaptiveExperiment`: an experiment with at least one observed outcome.

# Returns
- An [`AdaptiveLog`](@ref).

# Examples
```julia
using DrSnow, StableRNGs
rng = StableRNG(4)
ex = AdaptiveExperiment(BetaBernoulliThompson(2; floor=0.1))
for b in 1:10
    arms = assign!(ex, 20; rng=rng)
    observe!(ex, [rand(rng) < (a == 2 ? 0.6 : 0.4) for a in arms])
end
lg = experiment_log(ex)
adaptive_arm_values(lg; weights=:constant_allocation)
```
"""
function experiment_log(e::AdaptiveExperiment)
    T = length(e.outcomes)
    T > 0 || throw(ArgumentError("experiment_log: no outcomes observed yet"))
    K = n_arms(e.policy)
    P = Matrix{Float64}(undef, T, K)
    for t in 1:T
        P[t, :] = e.probabilities[t]
    end
    X = isempty(e.contexts) ? nothing : reduce(vcat, (c' for c in e.contexts[1:T]))
    nb = e.batch[T]
    M = length(e.true_means) == T ? reduce(vcat, (m' for m in e.true_means)) : nothing
    s = e.policy.settings
    snaps = Vector{Any}(e.snapshots[1:min(nb, length(e.snapshots))])
    return AdaptiveLog(e.arms[1:T], copy(e.outcomes), P, X, e.batch[1:T],
                       e.batch_start[1:nb], snaps, M, _ad_policy_name(e.policy) *
                       string(" (K = ", K, ")"), K, s.floor, s.floor_decay, s.burnin)
end

"""
    run_adaptive_experiment(policy, env::BanditEnvironment, T; batch_size=1,
                            rng=Random.default_rng()) -> AdaptiveLog
    run_adaptive_experiment(policy, Y::AbstractMatrix; contexts=nothing, means=nothing,
                            batch_size=1, rng=Random.default_rng()) -> AdaptiveLog

Simulate an adaptive experiment of `T` units under a given policy and return its
[`AdaptiveLog`](@ref).

Simulation is the main design tool for adaptive experiments: before fielding a
design, run it under plausible configurations of arm means to see the regret
([`cumulative_regret`](@ref)), how quickly allocation concentrates
([`plot_assignment_probabilities`](@ref)), how many units inferior arms receive under
the chosen floor and burn-in, and the width and coverage of the intervals of
[`adaptive_arm_values`](@ref) or [`adaptive_policy_value`](@ref). Offer-Westort,
Coppock and Green (2021) and Kaibel and Biemann (2021) evaluate adaptive designs for
social-science experiments by simulation of this kind; Hadad et al. (2021) use it to
assess coverage of their adaptively weighted estimators.

Units arrive in batches of `batch_size` (an integer, or a vector of batch sizes
summing to `T`). Within a batch the policy is not updated, which mimics delayed
outcomes; every unit's arm is drawn from the recorded probability vector through an
[`AdaptiveExperiment`](@ref), so simulation and deployment share one code path. The
first method draws covariates and outcomes from `env`. The second takes a `T × K`
matrix of potential outcomes `Y` (unit `t` yields `Y[t, k]` under arm `k`),
optionally with a `T × d` matrix of `contexts` and a `T × K` matrix of true `means`,
as in the simulation code of Hadad et al. (2021), where the potential outcomes are
drawn before the experiment so that different designs can be compared on the same
units. Simulated logs store the true means, which makes regret and bias computable.

# Arguments
- `policy::AdaptivePolicy`: the assignment policy (copied; not modified).
- `env::BanditEnvironment`, `T::Integer`: outcome model and number of units.
- `Y::AbstractMatrix`: alternatively, a `T × K` matrix of potential outcomes.

# Keywords
- `batch_size = 1`: an `Integer` batch size (the last batch may be smaller) or a
  vector of batch sizes summing to `T`.
- `contexts = nothing`, `means = nothing`: `T × d` covariates and `T × K` true arm
  means for the potential-outcome method.
- `rng::AbstractRNG = Random.default_rng()`: generator for contexts, outcomes, arm
  draws and Monte Carlo probabilities, used in a fixed order so that a seeded
  generator reproduces the experiment exactly.

# Returns
- An [`AdaptiveLog`](@ref).

# Examples
```julia
using DrSnow, StableRNGs
p = GaussianThompson(3; floor=1/3, floor_decay=0.7, burnin=15)
lg = run_adaptive_experiment(p, GaussianBandit([0.9, 1.0, 1.1]), 1000;
                             batch_size=50, rng=StableRNG(7))
last(cumulative_regret(lg))
adaptive_arm_values(lg; weights=:two_point)
```

# References
- Hadad, V., Hirshberg, D. A., Zhan, R., Wager, S., & Athey, S. (2021). Confidence
  intervals for policy evaluation in adaptive experiments. *Proceedings of the
  National Academy of Sciences*, 118(15), e2014602118.
- Offer-Westort, M., Coppock, A., & Green, D. P. (2021). Adaptive experimental
  design: Prospects and applications in political science. *American Journal of
  Political Science*, 65(4), 826–844.
- Kaibel, C., & Biemann, T. (2021). Rethinking the gold standard with multi-armed
  bandits: Machine learning allocation algorithms for experiments. *Organizational
  Research Methods*, 24(1), 78–103.
- Lattimore, T., & Szepesvári, C. (2020). *Bandit Algorithms*. Cambridge University
  Press.
"""
function run_adaptive_experiment(policy::AdaptivePolicy, env::BanditEnvironment,
                                 T::Integer; batch_size=1,
                                 rng::AbstractRNG=Random.default_rng())
    _ad_env_K(env) == n_arms(policy) ||
        throw(DimensionMismatch("policy has $(n_arms(policy)) arms, environment " *
                                "$(_ad_env_K(env))"))
    sizes = _ad_batch_sizes(batch_size, T)
    e = AdaptiveExperiment(policy)
    for n in sizes
        xs = [_ad_env_context(env, rng) for _ in 1:n]
        ctx = xs[1] === nothing ? nothing : xs
        arms = assign!(e, n; contexts=ctx, rng=rng)
        M = reduce(vcat, (_ad_env_means(env, x)' for x in xs))
        y = [_ad_env_draw(env, rng, M[i, arms[i]]) for i in 1:n]
        observe!(e, y; means=M)
    end
    return experiment_log(e)
end

function run_adaptive_experiment(policy::AdaptivePolicy, Y::AbstractMatrix;
                                 contexts=nothing, means=nothing, batch_size=1,
                                 rng::AbstractRNG=Random.default_rng())
    T, K = size(Y)
    K == n_arms(policy) ||
        throw(DimensionMismatch("policy has $(n_arms(policy)) arms, Y has $K columns"))
    all(isfinite, Y) || throw(ArgumentError("Y must be finite"))
    contexts === nothing || size(contexts, 1) == T ||
        throw(DimensionMismatch("contexts must have T rows"))
    means === nothing || size(means) == (T, K) ||
        throw(DimensionMismatch("means must be T × K"))
    sizes = _ad_batch_sizes(batch_size, T)
    e = AdaptiveExperiment(policy)
    t = 0
    for n in sizes
        rows = (t + 1):(t + n)
        ctx = contexts === nothing ? nothing : contexts[rows, :]
        arms = assign!(e, n; contexts=ctx, rng=rng)
        observe!(e, [Y[rows[i], arms[i]] for i in 1:n];
                 means=means === nothing ? nothing : means[rows, :])
        t += n
    end
    return experiment_log(e)
end

function _ad_batch_sizes(batch_size, T)
    T >= 1 || throw(ArgumentError("T must be at least 1"))
    if batch_size isa Integer
        batch_size >= 1 || throw(ArgumentError("batch_size must be at least 1"))
        nb = cld(T, batch_size)
        return [min(batch_size, T - (b - 1) * batch_size) for b in 1:nb]
    end
    s = collect(Int, batch_size)
    all(>(0), s) && sum(s) == T ||
        throw(ArgumentError("batch sizes must be positive and sum to T = $T"))
    return s
end

"""
    cumulative_regret(lg::AdaptiveLog; expected=true) -> Vector{Float64}

Cumulative regret of a simulated adaptive experiment: the total expected outcome lost
relative to an oracle that assigns every unit its best arm.

With ``μ_s(k)`` the expected outcome of arm ``k`` for unit ``s`` (which depends on
the unit's covariates in contextual simulations), the cumulative regret after ``t``
units is

```math
R_t = \\sum_{s \\le t} \\Bigl(\\max_k μ_s(k) - μ_s(A_s)\\Bigr),
```

and with `expected = true` the realized arm is replaced by its expectation under the
assignment probabilities, ``\\sum_k e_s(k)\\, μ_s(k)``, which removes the noise of the
arm draws (pseudo-regret). Regret measures the cost of experimentation to the
participants; it is the quantity bandit algorithms minimize (Lattimore and Szepesvári
2020) and trades off against the precision with which inferior arms are estimated.
Reporting both, from simulations of the planned design, makes that trade-off
explicit. Only simulated logs store the true means.

# Arguments
- `lg::AdaptiveLog`: a log from [`run_adaptive_experiment`](@ref) (or from an
  [`AdaptiveExperiment`](@ref) whose outcomes were recorded with `means`).

# Keywords
- `expected::Bool = true`: use the expected regret of each assignment (`true`) or
  the regret of the realized arm (`false`).

# Returns
- `Vector{Float64}` of length `T` with ``R_1, \\dots, R_T``.

# Examples
```julia
using DrSnow, StableRNGs
lg = run_adaptive_experiment(BetaBernoulliThompson(3; floor=0.02),
                             BernoulliBandit([0.3, 0.4, 0.5]), 800; rng=StableRNG(8))
last(cumulative_regret(lg))
```

# References
- Lattimore, T., & Szepesvári, C. (2020). *Bandit Algorithms*. Cambridge University
  Press.
"""
function cumulative_regret(l::AdaptiveLog; expected::Bool=true)
    l.true_means === nothing &&
        throw(ArgumentError("cumulative_regret: the log has no true means " *
                            "(only simulated experiments record them)"))
    M = l.true_means
    best = vec(maximum(M; dims=2))
    got = expected ? vec(sum(l.probabilities .* M; dims=2)) :
          [M[t, l.arms[t]] for t in axes(M, 1)]
    return cumsum(best .- got)
end
