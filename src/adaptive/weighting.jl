# Inference after adaptive data collection: adaptively weighted AIPW estimators of arm
# values, arm contrasts and policy values (Hadad et al. 2021; Zhan et al. 2021).
#
# Notation: unit t ∈ 1:T arrives in batch b(t); e_t(x, w) is the probability the
# design would assign arm w at covariates x in that batch (a function fixed within a
# batch); A_t, Y_t the assigned arm and outcome. With a plug-in outcome model μ̂_t(w)
# that uses only units before t, the AIPW score
#     Γ_t(w) = μ̂_t(w) + 1{A_t = w} / e_t(X_t, w) · (Y_t - μ̂_t(w))
# is conditionally unbiased for the arm value given the past, so Σ_t h_t Γ_t / Σ_t h_t
# is unbiased for predictable weights h_t. Uniform weights (h_t = 1) are consistent
# but their t-statistic need not be asymptotically normal when e_t → 0; the
# variance-stabilizing weights below restore normality.

# ---------------------------------------------------------------------------
# Canonical data
# ---------------------------------------------------------------------------

struct _AdData
    y::Vector{Float64}
    arms::Vector{Int}
    e::Matrix{Float64}                       # recorded probabilities (T × K)
    X::Union{Nothing,Matrix{Float64}}
    batch::Vector{Int}
    batch_start::Vector{Int}
    batch_units::Vector{Vector{Int}}
    probfn::Any                              # nothing, or (b, x) -> Vector (K)
    floor_decay::Union{Nothing,Float64}
    labels::Vector{String}
    covnames::Vector{Symbol}
end

_ad_T(d::_AdData) = length(d.y)
_ad_K(d::_AdData) = size(d.e, 2)
_ad_contextual(d::_AdData) = d.probfn !== nothing

function _ad_batch_units(batch::Vector{Int}, B::Int)
    u = [Int[] for _ in 1:B]
    for t in eachindex(batch)
        push!(u[batch[t]], t)
    end
    return u
end

function _ad_data(l::AdaptiveLog)
    T = nobs(l)
    T > 0 || throw(ArgumentError("the log is empty"))
    probfn = isempty(l.snapshots) ? nothing :
             (b, x) -> assignment_probabilities(l.snapshots[b], l.batch_start[b];
                                                context=x, rng=Random.Xoshiro(b))
    covs = l.contexts === nothing ? Symbol[] :
           [Symbol("x", j) for j in axes(l.contexts, 2)]
    return _AdData(l.outcomes, l.arms, l.probabilities, l.contexts, l.batch,
                   l.batch_start, _ad_batch_units(l.batch, length(l.batch_start)),
                   probfn, l.floor_decay, string.(1:l.K), covs)
end

function _ad_data(data, outcome::Symbol, arm::Symbol; probabilities, time,
                  batch=nothing, covariates=Symbol[], arm_levels=nothing,
                  probability_fn=nothing, floor_decay=nothing,
                  context::AbstractString)
    probabilities === nothing &&
        throw(ArgumentError("$context: pass `probabilities`, the columns holding the " *
                            "assignment probability of every arm"))
    time === nothing &&
        throw(ArgumentError("$context: pass `time`, the column ordering the units " *
                            "(adaptive designs depend on the order of arrival)"))
    pcols = Symbol.(collect(probabilities))
    K = length(pcols)
    K >= 2 || throw(ArgumentError("$context: need probability columns for ≥ 2 arms"))
    covs = Symbol.(collect(covariates))
    cols = vcat(outcome, arm, time, pcols, covs)
    batch === nothing || push!(cols, batch)
    require_columns(data, unique(cols); context=context)
    tv = data[!, time]
    any(ismissing, tv) && throw(ArgumentError("$context: missing values in $time"))
    ord = sortperm(collect(tv))
    for i in 2:length(ord)
        tv[ord[i]] == tv[ord[i - 1]] &&
            throw(ArgumentError("$context: duplicated values in the time column $time; " *
                                "units must have a strict order (use `batch` for " *
                                "units assigned simultaneously)"))
    end
    df = data[ord, :]
    y = _ml_column(df, outcome; context=context)
    levels = arm_levels === nothing ? collect(1:K) : collect(arm_levels)
    length(levels) == K ||
        throw(ArgumentError("$context: need one arm level per probability column"))
    lev = Dict(levels[k] => k for k in 1:K)
    arms = map(df[!, arm]) do a
        (ismissing(a) || !haskey(lev, a)) &&
            throw(ArgumentError("$context: arm value $a is not one of $(levels)"))
        lev[a]
    end
    E = _ml_matrix(df, pcols; context=context)
    all(p -> 0 <= p <= 1, E) ||
        throw(ArgumentError("$context: probabilities must be in [0, 1]"))
    all(abs.(sum(E; dims=2) .- 1) .< 1e-6) ||
        throw(ArgumentError("$context: the probabilities of each unit must sum to one"))
    all(E[t, arms[t]] > 0 for t in eachindex(arms)) ||
        throw(ArgumentError("$context: a unit was assigned an arm with probability 0"))
    X = isempty(covs) ? nothing : _ml_matrix(df, covs; context=context)
    bv = if batch === nothing
        collect(1:length(y))
    else
        raw = df[!, batch]
        any(ismissing, raw) && throw(ArgumentError("$context: missing batch values"))
        ids = Dict{Any,Int}()
        out = Vector{Int}(undef, length(raw))
        prev = 0
        for (t, r) in enumerate(raw)
            if !haskey(ids, r)
                ids[r] = length(ids) + 1
            end
            out[t] = ids[r]
            out[t] >= prev ||
                throw(ArgumentError("$context: batches must be contiguous in time"))
            prev = out[t]
        end
        out
    end
    B = maximum(bv)
    starts = [findfirst(==(b), bv) for b in 1:B]
    units = _ad_batch_units(bv, B)
    if probability_fn === nothing
        for u in units
            all(E[t, :] ≈ E[u[1], :] for t in u) || throw(ArgumentError(
                "$context: assignment probabilities vary within a batch; if they " *
                "depend on covariates pass `probability_fn`"))
        end
        pf = nothing
    else
        X === nothing && throw(ArgumentError("$context: `probability_fn` needs " *
                                             "`covariates`"))
        pf = (b, x) -> Float64.(collect(probability_fn(starts[b], x)))
        for t in eachindex(y)
            p = pf(bv[t], X[t, :])
            length(p) == K && isapprox(p, E[t, :]; atol=1e-6) || throw(ArgumentError(
                "$context: probability_fn does not reproduce the recorded " *
                "probabilities of unit $t"))
        end
    end
    fd = floor_decay === nothing ? nothing : float(floor_decay)
    return _AdData(y, arms, E, X, bv, starts, units, pf, fd, string.(levels), covs)
end

# ---------------------------------------------------------------------------
# Outcome model and AIPW scores
# ---------------------------------------------------------------------------

"""Running mean of each arm's outcomes over units before t (0 before any outcome)."""
function _ad_running_mean(d::_AdData)
    T, K = _ad_T(d), _ad_K(d)
    μ = zeros(T, K)
    s = zeros(K)
    n = zeros(Int, K)
    for t in 1:T
        for k in 1:K
            μ[t, k] = n[k] > 0 ? s[k] / n[k] : 0.0
        end
        s[d.arms[t]] += d.y[t]
        n[d.arms[t]] += 1
    end
    return μ
end

"""
Plug-in outcome model `μ̂_t(w)` using only units before `t`: `:none` (zero, giving
IPW scores), `:running_mean` or a `NuisanceLearner` refitted on all earlier units at
the start of each of `n_blocks` blocks of consecutive units (sequential
cross-fitting); within the first block, and for arms with fewer than `min_obs`
earlier units, the running mean is used.
"""
function _ad_muhat(d::_AdData, model, n_blocks::Integer, rng::AbstractRNG,
                   context::AbstractString)
    T, K = _ad_T(d), _ad_K(d)
    model === :none && return zeros(T, K)
    μ = _ad_running_mean(d)
    model === :running_mean && return μ
    model isa NuisanceLearner || throw(ArgumentError(
        "$context: outcome_model must be :none, :running_mean or a NuisanceLearner"))
    d.X === nothing && throw(ArgumentError(
        "$context: a learner outcome model needs covariates"))
    n_blocks >= 1 || throw(ArgumentError("$context: n_blocks must be at least 1"))
    nb = min(Int(n_blocks), T)
    edges = [1 + div((j - 1) * T, nb) for j in 1:(nb + 1)]
    edges[end] = T + 1
    seeds = task_seeds(rng, nb * K)
    min_obs = 2 * (size(d.X, 2) + 1)
    for j in 2:nb
        rows = edges[j]:(edges[j + 1] - 1)
        isempty(rows) && continue
        past = 1:(edges[j] - 1)
        for k in 1:K
            tr = [s for s in past if d.arms[s] == k]
            length(tr) >= min_obs || continue
            μ[rows, k] = fitpredict(model, d.X[tr, :], d.y[tr], d.X[rows, :];
                                    rng=Random.Xoshiro(seeds[(j - 1) * K + k]))
        end
    end
    return μ
end

function _ad_aipw_scores(d::_AdData, μ::AbstractMatrix)
    Γ = Matrix{Float64}(μ)
    for t in 1:_ad_T(d)
        a = d.arms[t]
        Γ[t, a] += (d.y[t] - μ[t, a]) / d.e[t, a]
    end
    return Γ
end

# ---------------------------------------------------------------------------
# Weights
# ---------------------------------------------------------------------------

"""
Two-point allocation ratios λ_t (Hadad et al. 2021): the arm is anticipated to be
either the best (e → 1, λ = 1/(T - t + 1)) or a bad arm assigned at the decaying floor
(e ∝ t^-α); λ interpolates with weight e_t. Then h²/e is obtained by stick breaking
and h_t = sqrt(e_t h²/e).
"""
function _ad_twopoint_weights(e::AbstractVector, α::Real)
    T = length(e)
    h2e = zeros(T)
    used = 0.0
    for t in 1:T
        bad = (1 - α) / ((1 - α) + T * (t / T)^α - t)
        good = 1 / (1 + T - t)
        λ = clamp((1 - e[t]) * bad + e[t] * good, 0.0, 1.0)
        h2e[t] = λ * (1 - used)
        used += h2e[t]
    end
    return sqrt.(max.(h2e .* e, 0.0))
end

"""
Function `b -> e_b(X_s, ·)` for all units `s` (`T × K`), caching the matrices when
`B T K` is moderate (contextual designs evaluate the policy snapshots).
"""
function _ad_batch_prob_fn(d::_AdData)
    d.probfn === nothing && return b -> _ad_batch_probs(d, b)
    B = length(d.batch_units)
    B * _ad_T(d) * _ad_K(d) <= 20_000_000 || return b -> _ad_batch_probs(d, b)
    cache = Vector{Union{Nothing,Matrix{Float64}}}(nothing, B)
    return function (b)
        cache[b] === nothing && (cache[b] = _ad_batch_probs(d, b))
        return cache[b]
    end
end

"""Probabilities e_b(X_s, ·) for all units s under batch b (`T × K`)."""
function _ad_batch_probs(d::_AdData, b::Int)
    if d.probfn === nothing
        return repeat(d.e[d.batch_units[b][1], :]', _ad_T(d), 1)
    end
    T, K = _ad_T(d), _ad_K(d)
    P = Matrix{Float64}(undef, T, K)
    for s in 1:T
        P[s, :] = d.probfn(b, view(d.X, s, :))
    end
    return P
end

# π²/e with 0/0 = 0 (arms the target policy does not use may have probability 0).
_ad_ratio(π, e) = π == 0 ? 0.0 : π^2 / e

# Σ_w π(s,w)² / e(s,w) for all rows (the conditional variance factor).
_ad_condvar(Π::AbstractMatrix, P::AbstractMatrix) =
    vec(sum(_ad_ratio.(Π, P); dims=2))

"""
Estimates and influence rows for policies `Πs` (each `T × K`, rows are distributions
over arms) with scores `Γ`:
- non-contextual weights `h_t` (`:uniform`, `:constant_allocation`, `:two_point`):
  `Q̂ = Σ h_t Γ_t(π) / Σ h_t`, influence `a_t = h_t (Γ_t(π) - Q̂) / Σ h`;
- contextual weights (`:contextual`, Zhan et al. 2021):
  `Q̂ = Σ_t h_t(X_t)/Z(X_t) Γ_t(π)`, `Z(x) = Σ_s h_s(x)`, influence
  `a_t = B_t - Σ_s h_t(X_s)/Z(X_s) B_s`.
The covariance of the estimates is `Σ_t a_t a_t'`.
"""
function _ad_weighted_estimates(d::_AdData, Γ::AbstractMatrix,
                                Πs::Vector{<:AbstractMatrix}, weights::Symbol,
                                arm_index::Vector{Int}, context::AbstractString)
    T = _ad_T(d)
    J = length(Πs)
    G = hcat([vec(sum(Π .* Γ; dims=2)) for Π in Πs]...)       # T × J policy scores
    θ = zeros(J)
    A = zeros(T, J)
    H = zeros(T, J)
    Pb = _ad_batch_prob_fn(d)
    # Without covariate-dependent probabilities and with policies that do not depend
    # on the covariates, contextual weighting coincides with constant allocation.
    if weights === :contextual && d.probfn === nothing &&
       all(Π -> all(r -> r == view(Π, 1, :), eachrow(Π)), Πs)
        weights = :constant_allocation
    end
    if weights === :contextual
        B = length(d.batch_units)
        Z = zeros(T, J)
        hself = zeros(T, J)
        for b in 1:B
            P = Pb(b)
            nb = length(d.batch_units[b])
            for j in 1:J
                h = 1 ./ sqrt.(_ad_condvar(Πs[j], P))
                Z[:, j] .+= nb .* h
                for t in d.batch_units[b]
                    hself[t, j] = h[t]
                end
            end
        end
        all(>(0), sum(hself; dims=1)) || throw(ArgumentError(
            "$context: the target policy has zero weight in every period (it only " *
            "uses arms that were never assignable)"))
        Z = max.(Z, floatmin(Float64))           # Z(x) = 0 only where h(x) = 0
        Bt = hself ./ Z .* G
        θ = vec(sum(Bt; dims=1))
        for b in 1:B
            P = Pb(b)
            for j in 1:J
                h = 1 ./ sqrt.(_ad_condvar(Πs[j], P))
                C = sum(h ./ Z[:, j] .* Bt[:, j])
                for t in d.batch_units[b]
                    A[t, j] = Bt[t, j] - C
                end
            end
        end
        H .= hself
    else
        for j in 1:J
            h = if weights === :uniform
                ones(T)
            elseif weights === :two_point
                α = d.floor_decay
                α === nothing && throw(ArgumentError(
                    "$context: two-point weights need `floor_decay`, the decay rate α " *
                    "of the assignment-probability floor c·t^-α"))
                0 <= α < 1 || throw(ArgumentError("$context: floor_decay must be in [0,1)"))
                _ad_twopoint_weights(d.e[:, arm_index[j]], α)
            else # :constant_allocation
                _ad_stablevar_weights(d, Πs[j], Pb)
            end
            all(isfinite, h) && sum(h) > 0 ||
                throw(ArgumentError("$context: the adaptive weights are degenerate"))
            Hs = sum(h)
            θ[j] = sum(h .* G[:, j]) / Hs
            A[:, j] = h .* (G[:, j] .- θ[j]) ./ Hs
            H[:, j] = h
        end
    end
    return θ, Symmetric(A' * A) |> Matrix, A, G, H
end

"""
Constant-allocation ("StableVar") weights `h_t = 1 / sqrt(E_x Σ_w π(x,w)²/e_t(x,w))`,
the expectation taken over the covariates of units in earlier batches (the first
batch uses its own units); without covariate dependence this is
`1/sqrt(Σ_w π(w)²/e_t(w))`, i.e. `sqrt(e_t(w))` for an arm.
"""
function _ad_stablevar_weights(d::_AdData, Π::AbstractMatrix, Pb)
    T = _ad_T(d)
    h = zeros(T)
    # prefix sums of π(X_s, w)² (non-contextual designs need only these: O(T K))
    C = cumsum(Π .^ 2; dims=1)
    for (b, u) in enumerate(d.batch_units)
        s0 = d.batch_start[b]
        past = s0 > 1 ? (1:(s0 - 1)) : u
        v = if d.probfn === nothing
            e = d.e[u[1], :]
            n = length(past)
            m2 = s0 > 1 ? C[s0 - 1, :] ./ n : vec(sum(Π[u, :] .^ 2; dims=1)) ./ n
            sum(_ad_ratio(sqrt(m2[w]), e[w]) for w in axes(Π, 2))
        else
            P = Pb(b)
            mean(sum(_ad_ratio(Π[s, w], P[s, w]) for w in axes(Π, 2))
                 for s in past)
        end
        h[u] .= 1 / sqrt(v)
    end
    return h
end

# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

"""
    AdaptiveEstimate <: CausalEstimate

Arm values, arm contrasts or policy values estimated from adaptively collected data
([`adaptive_arm_values`](@ref), [`adaptive_policy_value`](@ref),
[`naive_arm_means`](@ref)) or from logged bandit data ([`off_policy_value`](@ref)).

The object stores the point estimates of ``J`` targets (arms or policies) followed by
any requested contrasts, their joint covariance, and the per-unit ingredients of the
estimator: the doubly-robust (or IPW) scores ``\\hat Γ_t``, the evaluation weights
``h_t`` and the influence terms ``a_t`` with ``\\widehat{\\mathrm{Var}} = \\sum_t a_t
a_t'``. Keeping the scores and weights makes the estimator auditable (for instance,
by plotting weights against time or checking which units dominate) and allows
custom contrasts. Inference is asymptotically normal (`dof_residual = Inf`); its
justification and limits are described in the docstring of the function that
produced the estimate.

# Fields
- `coef::Vector{Float64}`: estimates of the targets, then the contrasts.
- `vcov::Matrix{Float64}`: their joint covariance matrix.
- `names::Vector{String}`: coefficient names, e.g. `"value(arm 2)"` or
  `"value(arm 2) - value(arm 1)"`.
- `n::Int`: number of units.
- `weights::Symbol`: weighting scheme (`:two_point`, `:constant_allocation`,
  `:contextual`, `:uniform`) or estimator (`:ipw`, `:snipw`, `:dr`,
  `:sample_mean`).
- `scores::Matrix{Float64}`: `T × J` scores ``\\hat Γ_t(π_j)`` of the `J` targets
  (before contrasts).
- `evaluation_weights::Matrix{Float64}`: `T × J` weights ``h_t`` (at each unit's own
  covariates for contextual weighting; empty for off-policy evaluation).
- `influence::Matrix{Float64}`: `T × J` terms ``a_t`` with `vcov = Σ_t a_t a_t'`
  before contrasts.
- `estimand::String`, `method::String`: descriptions used in printed output.
- `details::NamedTuple`: outcome model, notes and arm labels.

The StatsAPI accessors `coef`, `vcov`, `stderror`, `confint`, `coeftable`,
`coefnames` and `nobs` apply.
"""
struct AdaptiveEstimate <: CausalEstimate
    coef::Vector{Float64}
    vcov::Matrix{Float64}
    names::Vector{String}
    n::Int
    weights::Symbol
    scores::Matrix{Float64}
    evaluation_weights::Matrix{Float64}
    influence::Matrix{Float64}
    estimand::String
    method::String
    details::NamedTuple
end

StatsAPI.coef(r::AdaptiveEstimate) = r.coef
StatsAPI.vcov(r::AdaptiveEstimate) = r.vcov
StatsAPI.coefnames(r::AdaptiveEstimate) = r.names
StatsAPI.nobs(r::AdaptiveEstimate) = r.n
estimand(r::AdaptiveEstimate) = r.estimand
method_name(r::AdaptiveEstimate) = r.method

function show_details(io::IO, r::AdaptiveEstimate)
    haskey(r.details, :note) && !isempty(r.details.note) &&
        print(io, "\nNote: ", r.details.note)
    return nothing
end

const _AD_WEIGHT_NAMES = Dict(
    :two_point => "two-point allocation (Hadad et al. 2021)",
    :constant_allocation => "constant allocation / StableVar (Hadad et al. 2021)",
    :contextual => "contextual StableVar (Zhan et al. 2021)",
    :uniform => "uniform (unweighted)")

function _ad_check_weights(w::Symbol, d::_AdData, context)
    haskey(_AD_WEIGHT_NAMES, w) ||
        throw(ArgumentError("$context: weights must be one of " *
                            "$(sort(collect(keys(_AD_WEIGHT_NAMES))))"))
    if _ad_contextual(d) && w in (:two_point, :constant_allocation)
        w === :two_point && throw(ArgumentError(
            "$context: two-point weights assume assignment probabilities that do " *
            "not depend on covariates; with a contextual design use " *
            "weights = :contextual (or :constant_allocation)"))
    end
    return w
end

function _ad_contrast_matrix(J::Int, contrasts, reference::Int, context)
    contrasts === :none && return zeros(0, J), Tuple{Int,Int}[]
    1 <= reference <= J || throw(ArgumentError("$context: reference must be in 1:$J"))
    pairs = if contrasts === :reference
        [(j, reference) for j in 1:J if j != reference]
    elseif contrasts === :pairwise
        [(j, i) for i in 1:J for j in (i + 1):J]
    else
        throw(ArgumentError("$context: contrasts must be :none, :reference or " *
                            ":pairwise"))
    end
    L = zeros(length(pairs), J)
    for (r, (j, i)) in enumerate(pairs)
        L[r, j] = 1
        L[r, i] = -1
    end
    return L, pairs
end

function _ad_finish(θ, V, names, L, pairs, n, weights, G, H, A, estimand, method, details)
    M = vcat(Matrix{Float64}(I, length(θ), length(θ)), L)
    cn = vcat(names, ["$(names[j]) - $(names[i])" for (j, i) in pairs])
    Vf = M * V * M'
    return AdaptiveEstimate(M * θ, Matrix(Symmetric((Vf + Vf') / 2)), cn, n, weights,
                            G, H, A, estimand, method, details)
end

# ---------------------------------------------------------------------------
# Arm values and contrasts
# ---------------------------------------------------------------------------

"""
    adaptive_arm_values(lg::AdaptiveLog; weights=:two_point,
                        outcome_model=:running_mean, contrasts=:reference, reference=1,
                        floor_decay=lg.floor_decay, n_blocks=10,
                        rng=Random.default_rng()) -> AdaptiveEstimate
    adaptive_arm_values(data, outcome, arm; probabilities, time, batch=nothing,
                        covariates=Symbol[], arm_levels=nothing, probability_fn=nothing,
                        floor_decay=nothing, kwargs...) -> AdaptiveEstimate

Mean outcome of every arm and contrasts between arms, estimated from data collected
by a response-adaptive design with the adaptively weighted augmented
inverse-probability-weighted (AIPW) estimator of Hadad, Hirshberg, Zhan, Wager and
Athey (2021).

**Estimand and the problem.** The arm value is ``Q(w) = E[Y_t(w)]``, the mean
potential outcome under arm ``w`` in the population of units, and contrasts are
differences ``Q(j) - Q(i)``. Under adaptive assignment the sample mean of an arm is
biased, typically downward: an arm that looks bad by chance is sampled less, so its
unlucky early draws are not averaged out (Nie et al. 2018), and its usual standard
error ignores the adaptivity ([`naive_arm_means`](@ref)). With the recorded
propensities ``e_t(w)`` and an outcome model ``\\hat μ_t(w)`` fitted on units before
``t`` only, the AIPW score

```math
\\hat Γ_t(w) = \\hat μ_t(w) + \\frac{\\mathbf{1}\\{A_t = w\\}}{e_t(w)}
\\bigl(Y_t - \\hat μ_t(w)\\bigr)
```

has conditional mean ``Q(w)`` given the past, so any average with weights ``h_t``
that are predictable (functions of the past) is unbiased. The unweighted average is
nonetheless unreliable: when ``e_t(w)`` shrinks, the conditional variance of
``\\hat Γ_t`` (of order ``1/e_t``) explodes, a few units dominate, and the
t-statistic need not be asymptotically normal. The adaptively weighted estimator

```math
\\hat Q^h(w) = \\frac{\\sum_{t=1}^T h_t(w)\\, \\hat Γ_t(w)}{\\sum_{t=1}^T h_t(w)},
\\qquad
\\widehat{\\mathrm{se}} = \\frac{\\sqrt{\\sum_t h_t^2(w)
\\bigl(\\hat Γ_t(w) - \\hat Q^h(w)\\bigr)^2}}{\\sum_t h_t(w)},
```

uses variance-stabilizing weights that give every period a comparable share of the
variance, following the martingale argument of Luedtke and van der Laan (2016).

**Weights.** `:two_point` (default) builds ``h_t`` by the stick-breaking recursion
``h_t^2 / e_t = λ_t (1 - \\sum_{s<t} h_s^2 / e_s)`` with an allocation rate ``λ_t``
that interpolates, with weight ``e_t``, between the rates that would be optimal if
the arm ended up best (``e_t \\to 1``) or were assigned at a floor decaying as
``t^{-α}``; it needs ``α`` (`floor_decay`, taken from the log). `:constant_allocation`
uses ``h_t = \\sqrt{e_t(w)}`` (the "StableVar" weights), appropriate when ``α`` is
unknown. `:contextual` applies the contextual weighting of Zhan et al. (2021) (see
[`adaptive_policy_value`](@ref)) and reduces to `:constant_allocation` when the
probabilities do not depend on covariates. `:uniform` is plain AIPW, for comparison
only. Contrasts use each arm's own weights and the joint covariance of the separately
weighted estimates.

**Assumptions and inference.** Hadad et al. (2021) show that the studentized
estimator is asymptotically standard normal (``T \\to \\infty``) when (i) the
recorded probabilities are the true assignment probabilities and depend only on the
past (sequential ignorability, guaranteed by the logs of this package); (ii) the
potential outcomes are independent and identically distributed across units with
finite ``2 + δ`` moments and non-zero variance (no interference, no drift);
(iii) the probabilities satisfy the floor condition ``e_t(w) \\ge C t^{-α}`` with
``α \\in [0, 1)``, for the variance-stabilizing weights; and (iv) the outcome model
is bounded and converges, with either the model consistent for ``Q(w)`` or
``e_t(w)`` converging. The policies of this package have `floor = 0` by default, in
which case condition (iii) is not guaranteed. The data can check the recorded
floors but not the i.i.d. assumption. The guarantee is asymptotic: in the package's
Monte Carlo study of the design of Hadad et al. (three arms, floor
``(1/3) t^{-0.7}``, nominal 90%), all weighted estimators were close to nominal when
the arms were equal, but coverage for arms pushed to the floor ranged from about 0.83
to 0.90 at horizons up to ``T = 20000``, with two-point weights closest to nominal
and uniform weights the worst; the ratio form also carries a small finite-sample
bias that shrinks with ``T`` (see the method guide for the full tables).

**Practical guidance.** The intervals are fixed-horizon: they are valid when the
analysis happens once, at a sample size (or stopping time) that does not depend on
the estimates. For continuous monitoring, the confidence sequences of the sequential
area can be applied to the AIPW scores, but because the scores are unbounded when the
floor decays their guarantee is only asymptotic, not a finite-sample one, and it can
under-cover markedly when inferior arms are sampled rarely. Report the design
(policy, floor, burn-in, batches), the weights and the outcome model together with
the estimates, and compare with [`naive_arm_means`](@ref) to show the size of the
adaptivity bias. For batched designs, Zhang, Janson and Murphy (2020) show that
ordinary least squares (arm sample means) is not asymptotically normal when there is
no unique best arm and propose a batched OLS estimator (not implemented here). For
policy values use [`adaptive_policy_value`](@ref).

# Arguments
- `lg::AdaptiveLog`: log from [`run_adaptive_experiment`](@ref) or
  [`experiment_log`](@ref).
- `data`: alternatively, a table with one row per unit.
- `outcome::Symbol`, `arm::Symbol`: outcome column and arm column (values in
  `arm_levels`, default `1:K`).

# Keywords
- `weights::Symbol = :two_point`: `:two_point`, `:constant_allocation`,
  `:contextual` or `:uniform`, as described above.
- `outcome_model = :running_mean`: plug-in model ``\\hat μ_t``: `:running_mean` (each
  arm's mean over earlier units, as in Hadad et al.), `:none` (IPW scores) or a
  [`NuisanceLearner`](@ref) on `covariates`, refitted on all earlier units at the
  start of each of `n_blocks` blocks of consecutive units. The model affects
  efficiency only, provided it uses earlier units only.
- `contrasts::Symbol = :reference`: `:reference` (every arm minus `reference`),
  `:pairwise` or `:none`.
- `reference::Integer = 1`: the reference arm for `contrasts = :reference`.
- `floor_decay`: decay rate ``α \\in [0, 1)`` of the floor, required by `:two_point`
  (default: the value stored in the log; `nothing` for tables).
- `n_blocks::Integer = 10`: number of refitting blocks of a learner outcome model.
- `rng::AbstractRNG = Random.default_rng()`: seeds for learner fits.
- `probabilities` (tables): the columns holding the assignment probability of every
  arm, in the order of `arm_levels`.
- `time` (tables): column giving the strict order of arrival.
- `batch` (tables): column of batch labels; units of a batch were assigned with the
  same probability function (default: every unit its own batch).
- `covariates` (tables): covariate columns, for a learner outcome model or
  contextual weights.
- `arm_levels` (tables): arm labels in the order of `probabilities`.
- `probability_fn` (tables, contextual designs): function `(t, x) -> Vector` giving
  the probabilities the design used for the batch starting at unit `t` at covariates
  `x`; it must reproduce the recorded probabilities.

# Returns
- An [`AdaptiveEstimate`](@ref) with coefficients `value(arm k)` followed by the
  contrasts; `coef`, `stderror`, `confint` and `coeftable` use normal reference
  distributions.

# Examples
```julia
using DrSnow, StableRNGs
p = GaussianThompson(3; floor=1/3, floor_decay=0.7, burnin=15)
lg = run_adaptive_experiment(p, GaussianBandit([0.9, 1.0, 1.1]), 2000;
                             rng=StableRNG(1))
r = adaptive_arm_values(lg)                          # two-point weights
confint(r; level=0.9)
adaptive_arm_values(lg; weights=:uniform)            # plain AIPW, for comparison
```

# References
- Hadad, V., Hirshberg, D. A., Zhan, R., Wager, S., & Athey, S. (2021). Confidence
  intervals for policy evaluation in adaptive experiments. *Proceedings of the
  National Academy of Sciences*, 118(15), e2014602118.
- Luedtke, A. R., & van der Laan, M. J. (2016). Statistical inference for the mean
  outcome under a possibly non-unique optimal treatment strategy. *Annals of
  Statistics*, 44(2), 713–742.
- Nie, X., Tian, X., Taylor, J., & Zou, J. (2018). Why adaptively collected data
  have negative bias and how to correct for it. In *Proceedings of the 21st
  International Conference on Artificial Intelligence and Statistics*, PMLR 84,
  1261–1269.
- Zhan, R., Hadad, V., Hirshberg, D. A., & Athey, S. (2021). Off-policy evaluation
  via adaptive weighting with data from contextual bandits. In *Proceedings of the
  27th ACM SIGKDD Conference on Knowledge Discovery and Data Mining* (pp. 2125–2135).
- Zhang, K. W., Janson, L., & Murphy, S. A. (2020). Inference for batched bandits.
  *Advances in Neural Information Processing Systems*, 33, 9818–9829.
"""
function adaptive_arm_values(l::AdaptiveLog; floor_decay=l.floor_decay, kwargs...)
    d = _ad_data(l)
    d = _AdData(d.y, d.arms, d.e, d.X, d.batch, d.batch_start, d.batch_units, d.probfn,
                floor_decay === nothing ? nothing : float(floor_decay), d.labels,
                d.covnames)
    return _ad_arm_values(d; kwargs...)
end

function adaptive_arm_values(data, outcome::Symbol, arm::Symbol; probabilities=nothing,
                             time=nothing, batch=nothing, covariates=Symbol[],
                             arm_levels=nothing, probability_fn=nothing,
                             floor_decay=nothing, kwargs...)
    d = _ad_data(data, outcome, arm; probabilities=probabilities, time=time,
                 batch=batch, covariates=covariates, arm_levels=arm_levels,
                 probability_fn=probability_fn, floor_decay=floor_decay,
                 context="adaptive_arm_values")
    return _ad_arm_values(d; kwargs...)
end

function _ad_arm_values(d::_AdData; weights::Symbol=:two_point,
                        outcome_model=:running_mean, contrasts::Symbol=:reference,
                        reference::Integer=1, n_blocks::Integer=10,
                        rng::AbstractRNG=Random.default_rng())
    ctx = "adaptive_arm_values"
    _ad_check_weights(weights, d, ctx)
    T, K = _ad_T(d), _ad_K(d)
    μ = _ad_muhat(d, outcome_model, n_blocks, rng, ctx)
    Γ = _ad_aipw_scores(d, μ)
    Πs = [repeat(Float64.((1:K) .== k)', T, 1) for k in 1:K]
    θ, V, A, G, H = _ad_weighted_estimates(d, Γ, Πs, weights, collect(1:K), ctx)
    L, pairs = _ad_contrast_matrix(K, contrasts, Int(reference), ctx)
    names = ["value(arm $(d.labels[k]))" for k in 1:K]
    om = outcome_model isa Symbol ? string(outcome_model) :
         _ml_learner_name(outcome_model)
    note = weights === :uniform ?
           "uniform weights: the AIPW t-statistic need not be asymptotically normal " *
           "under adaptive assignment; intervals can under-cover." : ""
    return _ad_finish(θ, V, names, L, pairs, T, weights, G, H, A,
                      "mean outcome under each arm (and contrasts)",
                      "Adaptively weighted AIPW, " * _AD_WEIGHT_NAMES[weights],
                      (outcome_model=om, note=note, arms=d.labels))
end

"""
    naive_arm_means(lg::AdaptiveLog; contrasts=:reference, reference=1)
        -> AdaptiveEstimate
    naive_arm_means(data, outcome, arm; arm_levels=nothing, contrasts=:reference,
                    reference=1) -> AdaptiveEstimate

Sample mean of every arm with the textbook standard errors ``s_w / \\sqrt{n_w}``
(independent across arms), reported for comparison only: these are not valid
estimates or intervals under adaptive assignment.

The sample mean ``\\bar Y_w`` estimates ``Q(w) = E[Y(w)]`` without bias when the
number of units on each arm is fixed in advance. Under response-adaptive assignment
the number of draws of an arm depends on its own past outcomes: an arm that starts
with unlucky draws is sampled less, so the unlucky draws are not averaged out, and
Nie et al. (2018) show that under natural conditions on the design the sample means
are biased downward. The standard errors ignore the dependence created by adaptivity
as well, and in batched designs the sample means need not be asymptotically normal
(Zhang, Janson and Murphy 2020). In the package's Monte Carlo study of the Hadad et
al. (2021) design with equal arms, nominal 90% intervals from these formulas
covered only 0.78–0.84 of the time.

Use [`adaptive_arm_values`](@ref) for inference; reporting both shows readers how
large the adaptivity bias is in the data at hand.

# Arguments
- `lg::AdaptiveLog`: a log of an adaptive experiment.
- `data`, `outcome::Symbol`, `arm::Symbol`: alternatively, a table with outcome and
  arm columns.

# Keywords
- `arm_levels` (tables): arm labels, in order (default: sorted unique values).
- `contrasts::Symbol = :reference`, `reference::Integer = 1`: contrasts as in
  [`adaptive_arm_values`](@ref).

# Returns
- An [`AdaptiveEstimate`](@ref) with `weights = :sample_mean`. Every arm needs at
  least two outcomes.

# Examples
```julia
using DrSnow, StableRNGs
lg = run_adaptive_experiment(GaussianThompson(3; floor=0.05),
                             GaussianBandit([0.9, 1.0, 1.1]), 1000; rng=StableRNG(1))
naive_arm_means(lg)
```

# References
- Nie, X., Tian, X., Taylor, J., & Zou, J. (2018). Why adaptively collected data
  have negative bias and how to correct for it. In *Proceedings of the 21st
  International Conference on Artificial Intelligence and Statistics*, PMLR 84,
  1261–1269.
- Zhang, K. W., Janson, L., & Murphy, S. A. (2020). Inference for batched bandits.
  *Advances in Neural Information Processing Systems*, 33, 9818–9829.
- Hadad, V., Hirshberg, D. A., Zhan, R., Wager, S., & Athey, S. (2021). Confidence
  intervals for policy evaluation in adaptive experiments. *Proceedings of the
  National Academy of Sciences*, 118(15), e2014602118.
"""
function naive_arm_means(l::AdaptiveLog; contrasts::Symbol=:reference,
                         reference::Integer=1)
    return _ad_naive(l.outcomes, l.arms, string.(1:l.K), contrasts, reference)
end

function naive_arm_means(data, outcome::Symbol, arm::Symbol; arm_levels=nothing,
                         contrasts::Symbol=:reference, reference::Integer=1)
    ctx = "naive_arm_means"
    require_columns(data, [outcome, arm]; context=ctx)
    y = _ml_column(data, outcome; context=ctx)
    raw = data[!, arm]
    levels = arm_levels === nothing ? sort(unique(skipmissing(raw))) : collect(arm_levels)
    lev = Dict(levels[k] => k for k in eachindex(levels))
    a = map(v -> (ismissing(v) || !haskey(lev, v)) ?
                 throw(ArgumentError("$ctx: arm value $v not in $levels")) : lev[v], raw)
    return _ad_naive(y, a, string.(levels), contrasts, reference)
end

function _ad_naive(y, arms, labels, contrasts, reference)
    K = length(labels)
    T = length(y)
    θ = zeros(K)
    A = zeros(T, K)
    for k in 1:K
        idx = findall(==(k), arms)
        length(idx) >= 2 ||
            throw(ArgumentError("naive_arm_means: arm $(labels[k]) has fewer than two " *
                                "outcomes"))
        θ[k] = mean(y[idx])
        n = length(idx)
        A[idx, k] = (y[idx] .- θ[k]) ./ n
    end
    V = A' * A
    L, pairs = _ad_contrast_matrix(K, contrasts, Int(reference), "naive_arm_means")
    return _ad_finish(θ, V, ["value(arm $(l))" for l in labels], L, pairs, T,
                      :sample_mean, zeros(T, 0), zeros(T, 0), A,
                      "mean outcome under each arm (and contrasts)",
                      "Naive sample means (not valid under adaptive assignment)",
                      (note="sample means are biased and their standard errors are " *
                            "not valid when assignment adapted to earlier outcomes.",))
end

# ---------------------------------------------------------------------------
# Policy values
# ---------------------------------------------------------------------------

"""
Evaluate a target policy on `T` units with covariates `X` (or `nothing`): a `T × K`
matrix of arm probabilities. Arms are identified by their labels (`labels`, compared
as strings), so a scalar policy, a per-unit vector, a function returning an arm, or a
`PolicyTree` all name arms the way the data do.
"""
function _ad_policy_matrix(pol, X, covnames::Vector{Symbol}, labels::Vector{String},
                           T::Int, context)
    K = length(labels)
    idx(a) = begin
        k = findfirst(==(string(a)), labels)
        k === nothing && throw(ArgumentError("$context: arm $a is not one of the arm " *
                                             "labels $(labels)"))
        k
    end
    onehot(ks) = (M = zeros(T, K); for t in 1:T; M[t, ks[t]] = 1; end; M)
    Π = if pol isa AbstractMatrix
        size(pol) == (T, K) ||
            throw(DimensionMismatch("$context: a policy matrix must be $T × $K"))
        Matrix{Float64}(pol)
    elseif pol isa AbstractVector
        length(pol) == T || throw(DimensionMismatch("$context: a vector policy needs " *
                                                    "one arm per unit ($T)"))
        onehot(idx.(pol))
    elseif pol isa PolicyTree
        X === nothing && throw(ArgumentError("$context: a PolicyTree needs covariates"))
        js = map(pol.covariates) do c
            j = findfirst(==(c), covnames)
            j === nothing && throw(ArgumentError("$context: policy covariate $c is not " *
                                                 "among the covariates $(covnames)"))
            j
        end
        onehot(idx.(StatsAPI.predict(pol, X[:, js])))
    elseif pol isa Function
        M = zeros(T, K)
        for t in 1:T
            r = pol(X === nothing ? nothing : X[t, :])
            if r isa AbstractVector
                length(r) == K || throw(DimensionMismatch(
                    "$context: policy must return an arm or a length-$K vector"))
                M[t, :] = r
            else
                M[t, idx(r)] = 1
            end
        end
        M
    else
        onehot(fill(idx(pol), T))
    end
    all(>=(-1e-12), Π) && all(abs.(sum(Π; dims=2) .- 1) .< 1e-8) ||
        throw(ArgumentError("$context: policy rows must be probability distributions " *
                            "over the arms"))
    return Π
end

_ad_policy_matrix(pol, d::_AdData, context) =
    _ad_policy_matrix(pol, d.X, d.covnames, d.labels, _ad_T(d), context)

"""
    adaptive_policy_value(lg::AdaptiveLog, policies; names=nothing,
                          weights=:contextual, outcome_model=:running_mean,
                          contrasts=:none, reference=1, n_blocks=10,
                          rng=Random.default_rng()) -> AdaptiveEstimate
    adaptive_policy_value(data, outcome, arm, policies; probabilities, time,
                          batch=nothing, covariates=Symbol[], arm_levels=nothing,
                          probability_fn=nothing, kwargs...) -> AdaptiveEstimate

Value of one or more fixed treatment-assignment policies, estimated from data
collected by an adaptive (possibly contextual-bandit) experiment with adaptively
weighted doubly-robust scores (Zhan, Hadad, Hirshberg and Athey 2021).

**Estimand.** A target policy ``π`` assigns arm ``w`` to a unit with covariates
``x`` with probability ``π(x, w)``; its value is the mean outcome if it were applied
to the population,

```math
Q(π) = E\\Bigl[\\sum_{w=1}^K π(X, w)\\, Y(w)\\Bigr].
```

A single arm is the special case ``π(x, w) = \\mathbf{1}\\{w = k\\}``; a targeting
rule learned elsewhere (a function of covariates or a [`PolicyTree`](@ref)) is the
typical application. Contrasts ``Q(π_j) - Q(π_i)`` compare policies. The estimator
averages the doubly-robust scores ``\\hat Γ_t(π) = \\sum_w π(X_t, w)\\, \\hat Γ_t(w)``,
with the AIPW scores ``\\hat Γ_t(w)`` of [`adaptive_arm_values`](@ref), under
adaptive weights.

**Weights.** With `:contextual` (default), each unit's weight depends on the
covariates, ``h_t(x) = 1/\\sqrt{\\sum_w π(x, w)^2 / e_t(x, w)}``, the inverse
standard deviation of the score at ``x`` under the design of period ``t``, and the
scores are normalized context by context:

```math
\\hat Q^C(π) = \\sum_{t=1}^T \\frac{h_t(X_t)}{\\sum_{s=1}^T h_s(X_t)}\\, \\hat Γ_t(π).
```

This needs the probabilities the design *would have* used in every batch at every
unit's covariates, which logs of contextual policies provide through their policy
snapshots (or `probability_fn` for tables); the cost is ``O(B T K)`` for ``B``
batches, so batched designs are much cheaper to analyse. `:constant_allocation`
uses the non-contextual StableVar weights
``h_t = 1/\\sqrt{E_x \\sum_w π(x, w)^2 / e_t(x, w)}``, with the expectation over the
covariates of earlier units. `:uniform` is the unweighted doubly-robust estimator,
for comparison only. When neither the design nor the policies depend on covariates,
`:contextual` coincides with `:constant_allocation`.

**Assumptions and inference.** Zhan et al. (2021) prove consistency and asymptotic
normality of the studentized estimators under sequential ignorability with the
recorded probabilities, i.i.d. units with bounded conditional fourth moments and
conditional variances bounded away from zero, a bounded and convergent outcome
model, a floor ``e_t(x, w) \\ge C t^{-α}`` with ``α \\in [0, 1/2)`` (stricter than
the ``α < 1`` of Hadad et al. 2021 for arm values), and a stability condition
requiring that the inverse probabilities behave like their expectations in the long
run (it can fail when the design keeps switching between equally good arms). Their
central limit theorem for contextual weighting is stated for a discrete covariate
space and a policy that always assigns one arm; for continuous covariates and
general policies the intervals rest on the same argument without a formal proof, and
the package checks their coverage by Monte Carlo (a four-arm linear Thompson design).
The policies of this package have `floor = 0` by default, in which case the floor
condition is not guaranteed. The intervals are fixed-horizon (see
[`adaptive_arm_values`](@ref) on monitoring).

**Practical guidance.** The target policies must be fixed in advance or learned on
other data; evaluating a policy on the data from which it was learned gives an
optimistically biased value (for policy learning from adaptive data see
[`bandit_dr_scores`](@ref)). For logs from a design that did *not* adapt to
outcomes, [`off_policy_value`](@ref) is simpler.

# Arguments
- `lg::AdaptiveLog`, or `data` with columns `outcome` and `arm` (see
  [`adaptive_arm_values`](@ref) for the table keywords `probabilities`, `time`,
  `batch`, `covariates`, `arm_levels` and `probability_fn`).
- `policies`: one policy or a tuple of policies. A policy is an arm label (always
  assign that arm), a vector of arm labels (one per unit, in time order), a `T × K`
  matrix of arm probabilities, a function `x -> arm` or `x -> probability vector`
  (called with `nothing` when there are no covariates), or a [`PolicyTree`](@ref)
  over covariates `x1 … xd` (the column names of `DataFrame(lg)`) whose actions are
  arm labels. Arms of an [`AdaptiveLog`](@ref) are labelled `1:K`.

# Keywords
- `names = nothing`: labels of the policies (default `"policy 1"`, …).
- `weights::Symbol = :contextual`: `:contextual`, `:constant_allocation` or
  `:uniform`; two-point weights are defined for arm values only.
- `outcome_model = :running_mean`: `:running_mean`, `:none` or a
  [`NuisanceLearner`](@ref) refitted on earlier units in `n_blocks` blocks.
- `contrasts::Symbol = :none`: `:none`, `:reference` or `:pairwise` differences of
  policy values; `reference::Integer = 1` selects the reference policy.
- `n_blocks::Integer = 10`, `rng::AbstractRNG = Random.default_rng()`: refitting
  blocks and seeds of a learner outcome model.

# Returns
- An [`AdaptiveEstimate`](@ref) with coefficients `value(<name>)` followed by the
  contrasts.

# Examples
```julia
using DrSnow, StableRNGs
env = ContextualBandit(x -> [0.0, x[1], -x[1], 0.5x[2]], rng -> randn(rng, 2), 4)
p = LinearThompson(4, 2; floor=0.25, floor_decay=0.4, burnin=200)
lg = run_adaptive_experiment(p, env, 2000; batch_size=100, rng=StableRNG(3))
oracle = x -> argmax([0.0, x[1], -x[1], 0.5x[2]])
adaptive_policy_value(lg, (oracle, 1); names=["oracle", "arm 1"],
                      contrasts=:reference)
```

# References
- Zhan, R., Hadad, V., Hirshberg, D. A., & Athey, S. (2021). Off-policy evaluation
  via adaptive weighting with data from contextual bandits. In *Proceedings of the
  27th ACM SIGKDD Conference on Knowledge Discovery and Data Mining* (pp. 2125–2135).
- Hadad, V., Hirshberg, D. A., Zhan, R., Wager, S., & Athey, S. (2021). Confidence
  intervals for policy evaluation in adaptive experiments. *Proceedings of the
  National Academy of Sciences*, 118(15), e2014602118.
- Dudík, M., Langford, J., & Li, L. (2011). Doubly robust policy evaluation and
  learning. In *Proceedings of the 28th International Conference on Machine
  Learning* (pp. 1097–1104).
- Athey, S., & Wager, S. (2021). Policy learning with observational data.
  *Econometrica*, 89(1), 133–161.
"""
function adaptive_policy_value(l::AdaptiveLog, policies; kwargs...)
    return _ad_policy_value(_ad_data(l), policies; kwargs...)
end

function adaptive_policy_value(data, outcome::Symbol, arm::Symbol, policies;
                               probabilities=nothing, time=nothing, batch=nothing,
                               covariates=Symbol[], arm_levels=nothing,
                               probability_fn=nothing, kwargs...)
    d = _ad_data(data, outcome, arm; probabilities=probabilities, time=time,
                 batch=batch, covariates=covariates, arm_levels=arm_levels,
                 probability_fn=probability_fn, floor_decay=nothing,
                 context="adaptive_policy_value")
    return _ad_policy_value(d, policies; kwargs...)
end

# A tuple lists several policies; anything else is a single policy.
_ad_policy_list(p) = p isa Tuple ? collect(p) : [p]

function _ad_policy_value(d::_AdData, policies; names=nothing, weights::Symbol=:contextual,
                          outcome_model=:running_mean, contrasts::Symbol=:none,
                          reference::Integer=1, n_blocks::Integer=10,
                          rng::AbstractRNG=Random.default_rng())
    ctx = "adaptive_policy_value"
    weights === :two_point && throw(ArgumentError(
        "$ctx: two-point weights are defined for arm values; use " *
        "adaptive_arm_values or weights = :contextual / :constant_allocation"))
    _ad_check_weights(weights, d, ctx)
    pols = _ad_policy_list(policies)
    J = length(pols)
    labels = names === nothing ? ["policy $j" for j in 1:J] : string.(collect(names))
    length(labels) == J || throw(ArgumentError("$ctx: need one name per policy"))
    Πs = [_ad_policy_matrix(p, d, ctx) for p in pols]
    μ = _ad_muhat(d, outcome_model, n_blocks, rng, ctx)
    Γ = _ad_aipw_scores(d, μ)
    θ, V, A, G, H = _ad_weighted_estimates(d, Γ, Πs, weights, zeros(Int, J), ctx)
    L, pairs = _ad_contrast_matrix(J, contrasts, Int(reference), ctx)
    om = outcome_model isa Symbol ? string(outcome_model) :
         _ml_learner_name(outcome_model)
    note = weights === :uniform ?
           "uniform weights: the DR t-statistic need not be asymptotically normal " *
           "under adaptive assignment." : ""
    return _ad_finish(θ, V, ["value($l)" for l in labels], L, pairs, _ad_T(d), weights,
                      G, H, A, "value of the target policies (and contrasts)",
                      "Adaptively weighted DR policy evaluation, " *
                      _AD_WEIGHT_NAMES[weights], (outcome_model=om, note=note))
end
