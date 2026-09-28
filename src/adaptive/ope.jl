# Off-policy evaluation from logged bandit data (IPW, self-normalized IPW, doubly
# robust) and doubly-robust scores for policy learning (hand-off to `policy_tree`).

struct _AdLogged
    y::Vector{Float64}
    a::Vector{Int}
    p::Vector{Float64}          # logged propensity of the chosen action
    X::Union{Nothing,Matrix{Float64}}
    labels::Vector{String}
    covnames::Vector{Symbol}
end

function _ad_logged(data, outcome::Symbol, action::Symbol; propensity, covariates,
                    action_levels, context)
    propensity === nothing && throw(ArgumentError(
        "$context: pass `propensity`, the column with the logged probability of the " *
        "action taken"))
    covs = Symbol.(collect(covariates))
    require_columns(data, unique(vcat(outcome, action, propensity, covs));
                    context=context)
    y = _ml_column(data, outcome; context=context)
    p = _ml_column(data, propensity; context=context)
    all(v -> 0 < v <= 1, p) ||
        throw(ArgumentError("$context: logged propensities must be in (0, 1]"))
    raw = data[!, action]
    any(ismissing, raw) && throw(ArgumentError("$context: missing actions"))
    levels = action_levels === nothing ? sort(unique(raw)) : collect(action_levels)
    length(levels) >= 2 || throw(ArgumentError("$context: need at least two actions"))
    lev = Dict(levels[k] => k for k in eachindex(levels))
    a = map(v -> haskey(lev, v) ? lev[v] :
                 throw(ArgumentError("$context: action $v not in $levels")), raw)
    X = isempty(covs) ? nothing : _ml_matrix(data, covs; context=context)
    return _AdLogged(y, a, p, X, string.(levels), covs)
end

function _ad_logged(l::AdaptiveLog)
    T = nobs(l)
    p = [l.probabilities[t, l.arms[t]] for t in 1:T]
    covs = l.contexts === nothing ? Symbol[] :
           [Symbol("x", j) for j in axes(l.contexts, 2)]
    return _AdLogged(l.outcomes, l.arms, p, l.contexts, string.(1:l.K), covs)
end

"""Cross-fitted outcome model `m̂(x, a)` for every unit and action (`n × K`)."""
function _ad_crossfit_outcomes(L::_AdLogged, learner, n_folds, folds, rng, context)
    n = length(L.y)
    K = length(L.labels)
    L.X === nothing && throw(ArgumentError("$context: the doubly-robust method needs " *
                                           "`covariates` for the outcome model"))
    learner isa NuisanceLearner ||
        throw(ArgumentError("$context: outcome_learner must be a NuisanceLearner"))
    fold = if folds === nothing
        vec(crossfit_folds(n, n_folds, 1; rng=rng, strata=L.a))
    else
        f = collect(Int, folds)
        length(f) == n || throw(DimensionMismatch("$context: folds must have length $n"))
        f
    end
    F = maximum(fold)
    F >= 2 || throw(ArgumentError("$context: need at least two folds"))
    seeds = task_seeds(rng, F * K)
    M = zeros(n, K)
    for k in 1:F
        te = findall(==(k), fold)
        isempty(te) && continue
        for a in 1:K
            tr = findall(i -> fold[i] != k && L.a[i] == a, 1:n)
            length(tr) >= 2 || throw(ArgumentError(
                "$context: action $(L.labels[a]) has fewer than two observations " *
                "outside fold $k"))
            M[te, a] = fitpredict(learner, L.X[tr, :], L.y[tr], L.X[te, :];
                                  rng=Random.Xoshiro(seeds[(k - 1) * K + a]))
        end
    end
    return M
end

"""
    off_policy_value(data, outcome, action, policy; propensity, covariates=Symbol[],
                     method=:dr, outcome_learner=RidgeLearner(), n_folds=5,
                     folds=nothing, action_levels=nothing, reference=nothing,
                     rng=Random.default_rng()) -> AdaptiveEstimate
    off_policy_value(lg::AdaptiveLog, policy; kwargs...) -> AdaptiveEstimate

Value of a target treatment-assignment policy estimated from logged bandit data
collected under a different, *non-adaptive* logging policy with known propensities
(off-policy evaluation).

**Estimand and identification.** The data are ``(X_i, A_i, Y_i)`` together with the
logged probability ``p_i = p(A_i \\mid X_i)`` with which action ``A_i`` was taken. The
value of a target policy ``π`` is
``V(π) = E[\\sum_a π(a \\mid X)\\, Y(a)]``, the mean outcome if ``π`` were deployed.
It is identified when the logged propensities are the true assignment probabilities
given ``X`` (the logging policy is unconfounded given the recorded covariates) and
``p(a \\mid x) > 0`` wherever the target policy puts mass (overlap). Neither
condition can be verified from the outcome data; the propensities must come from the
logging system, not from a fitted model.

**Estimators.** With importance weights ``w_i = π(A_i \\mid X_i) / p_i``,
- `:ipw`: ``\\hat V = n^{-1} \\sum_i w_i Y_i`` (Horvitz and Thompson 1952), unbiased
  but with high variance when the target and logging policies differ;
- `:snipw`: the self-normalized ``\\sum_i w_i Y_i / \\sum_i w_i``, with bias of order
  ``1/n`` and usually much lower variance (Swaminathan and Joachims 2015),
  delta-method standard error;
- `:dr` (default): the doubly-robust estimator of Dudík, Langford and Li (2011),

```math
\\hat V_{DR} = \\frac{1}{n} \\sum_{i=1}^n \\Bigl[\\sum_a π(a \\mid X_i)\\, \\hat m(X_i, a)
+ w_i \\bigl(Y_i - \\hat m(X_i, A_i)\\bigr)\\Bigr],
```

with an outcome model ``\\hat m`` cross-fitted over `n_folds` folds. With known
propensities the DR estimator is unbiased whatever ``\\hat m``, and a good outcome
model reduces its variance.

**Inference and scope.** Standard errors treat the logged units as independent
draws and the logging policy as fixed (for example, a randomized rule or a deployed
rule with fixed exploration). For data collected by an *adaptive* experiment, in
which the logging probabilities depend on earlier outcomes, these standard errors are
not justified: use [`adaptive_policy_value`](@ref) instead. The `AdaptiveLog`
method is provided for logs of non-adaptive designs and for comparison. With
`reference`, the result also reports the value of a second policy and the difference,
using their joint covariance.

# Arguments
- `data`, `outcome::Symbol`, `action::Symbol`: table and columns; actions are labels
  in `action_levels`.
- `lg::AdaptiveLog`: alternatively, a log (propensities of the arms taken).
- `policy`: an action label, a vector of labels (one per row), an `n × K` matrix of
  action probabilities, a function of the covariate vector returning a label or a
  probability vector, or a [`PolicyTree`](@ref) over `covariates`.

# Keywords
- `propensity::Symbol` (required for tables): column with the logged probability of
  the action taken, in `(0, 1]`.
- `covariates::Vector{Symbol} = Symbol[]`: covariates of the outcome model and of
  `policy`; required by `:dr`.
- `method::Symbol = :dr`: `:dr`, `:ipw` or `:snipw`.
- `outcome_learner::NuisanceLearner = RidgeLearner()`: outcome model for `:dr`, fitted
  separately for each action.
- `n_folds::Integer = 5`, `folds = nothing`: number of cross-fitting folds (stratified
  by action) or explicit fold ids.
- `action_levels = nothing`: action labels in order (default: sorted unique values).
- `reference = nothing`: optional second policy for a difference in values.
- `rng::AbstractRNG = Random.default_rng()`: fold assignment and learner seeds.

# Returns
- An [`AdaptiveEstimate`](@ref) with coefficient `value(policy)` (and
  `value(reference)` and `value(policy) - value(reference)`).

# Examples
```julia
using DrSnow, DataFrames, StableRNGs
rng = StableRNG(1)
n = 2000
x1, x2 = randn(rng, n), randn(rng, n)
P = hcat(fill(1.0, n), exp.(0.5 .* x1), exp.(-0.5 .* x1))
P ./= sum(P; dims=2)                          # logging policy: softmax in x1
a = [findfirst(cumsum(P[i, :]) .> rand(rng)) for i in 1:n]
mu = hcat(zeros(n), x1, 0.5 .* x2)
y = [mu[i, a[i]] for i in 1:n] .+ randn(rng, n)
df = DataFrame(x1=x1, x2=x2, a=a, y=y, p=[P[i, a[i]] for i in 1:n])
off_policy_value(df, :y, :a, x -> x[1] > 0 ? 2 : 1; propensity=:p,
                 covariates=[:x1, :x2], reference=1, rng=StableRNG(2))
```

# References
- Horvitz, D. G., & Thompson, D. J. (1952). A generalization of sampling without
  replacement from a finite universe. *Journal of the American Statistical
  Association*, 47(260), 663–685.
- Dudík, M., Langford, J., & Li, L. (2011). Doubly robust policy evaluation and
  learning. In *Proceedings of the 28th International Conference on Machine
  Learning* (pp. 1097–1104).
- Swaminathan, A., & Joachims, T. (2015). The self-normalized estimator for
  counterfactual learning. *Advances in Neural Information Processing Systems*, 28,
  3231–3239.
- Zhan, R., Hadad, V., Hirshberg, D. A., & Athey, S. (2021). Off-policy evaluation
  via adaptive weighting with data from contextual bandits. In *Proceedings of the
  27th ACM SIGKDD Conference on Knowledge Discovery and Data Mining* (pp. 2125–2135).
"""
function off_policy_value(data, outcome::Symbol, action::Symbol, policy;
                          propensity=nothing, covariates=Symbol[], action_levels=nothing,
                          kwargs...)
    ctx = "off_policy_value"
    L = _ad_logged(data, outcome, action; propensity=propensity, covariates=covariates,
                   action_levels=action_levels, context=ctx)
    return _ad_ope(L, policy; kwargs...)
end

off_policy_value(l::AdaptiveLog, policy; kwargs...) =
    _ad_ope(_ad_logged(l), policy; kwargs...)

function _ad_ope(L::_AdLogged, policy; method::Symbol=:dr,
                 outcome_learner=RidgeLearner(), n_folds::Integer=5, folds=nothing,
                 reference=nothing, rng::AbstractRNG=Random.default_rng())
    ctx = "off_policy_value"
    method in (:dr, :ipw, :snipw) ||
        throw(ArgumentError("$ctx: method must be :dr, :ipw or :snipw"))
    n = length(L.y)
    K = length(L.labels)
    pols = reference === nothing ? [policy] : [policy, reference]
    Πs = [_ad_policy_matrix(p, L.X, L.covnames, L.labels, n, ctx) for p in pols]
    M = method === :dr ? _ad_crossfit_outcomes(L, outcome_learner, n_folds, folds, rng,
                                               ctx) : nothing
    J = length(Πs)
    θ = zeros(J)
    A = zeros(n, J)
    G = zeros(n, J)
    for j in 1:J
        Π = Πs[j]
        w = [Π[i, L.a[i]] / L.p[i] for i in 1:n]
        if method === :ipw
            ψ = w .* L.y
            θ[j] = mean(ψ)
            G[:, j] = ψ
            A[:, j] = (ψ .- θ[j]) ./ n
        elseif method === :snipw
            sum(w) > 0 || throw(ArgumentError("$ctx: the target policy puts no mass on " *
                                              "any logged action"))
            θ[j] = sum(w .* L.y) / sum(w)
            G[:, j] = w .* L.y
            A[:, j] = w .* (L.y .- θ[j]) ./ sum(w)
        else
            ψ = [sum(Π[i, a] * M[i, a] for a in 1:K) + w[i] * (L.y[i] - M[i, L.a[i]])
                 for i in 1:n]
            θ[j] = mean(ψ)
            G[:, j] = ψ
            A[:, j] = (ψ .- θ[j]) ./ n
        end
    end
    V = A' * A
    names = reference === nothing ? ["value(policy)"] :
            ["value(policy)", "value(reference)"]
    Lm, pairs = reference === nothing ? (zeros(0, 1), Tuple{Int,Int}[]) :
                (reshape([1.0, -1.0], 1, 2), [(1, 2)])
    mname = Dict(:ipw => "Inverse propensity weighting",
                 :snipw => "Self-normalized inverse propensity weighting",
                 :dr => "Doubly robust (cross-fitted)")[method]
    res = _ad_finish(θ, V, names, Lm, Tuple{Int,Int}[], n, method, G, zeros(n, 0), A,
                     "value of the target policy", "Off-policy evaluation: " * mname,
                     (note="standard errors assume independent units and a logging " *
                           "policy that did not adapt to earlier outcomes.",
                      outcome_learner=method === :dr ? _ml_learner_name(outcome_learner) :
                                      ""))
    isempty(pairs) && return res
    b = vcat(res.coef, res.coef[1] - res.coef[2])
    Mx = [1.0 0.0; 0.0 1.0; 1.0 -1.0]
    return AdaptiveEstimate(b, Matrix(Symmetric(Mx * V * Mx')),
                            vcat(names, "value(policy) - value(reference)"), n, method,
                            G, zeros(n, 0), A, res.estimand, res.method, res.details)
end

"""
    bandit_dr_scores(data, outcome, action; propensity, covariates,
                     outcome_learner=RidgeLearner(), n_folds=5, folds=nothing,
                     action_levels=nothing, rng=Random.default_rng()) -> Matrix{Float64}
    bandit_dr_scores(lg::AdaptiveLog; outcome_model=:running_mean, n_blocks=10,
                     rng=Random.default_rng()) -> Matrix{Float64}

Doubly-robust scores for every unit and every action, the input to policy learning
from logged or experimental bandit data.

For unit ``i`` and action ``a`` the score is

```math
\\hat Γ_i(a) = \\hat m(X_i, a) + \\frac{\\mathbf{1}\\{A_i = a\\}}{p_i(a)}
\\bigl(Y_i - \\hat m(X_i, a)\\bigr),
```

so that ``\\sum_a π(a \\mid X_i)\\, \\hat Γ_i(a)`` is the doubly-robust score of any
policy ``π`` (Dudík, Langford and Li 2011). The column means estimate the action
values, and maximizing the mean policy score over a class of policies is the
doubly-robust policy-learning approach of Athey and Wager (2021): pass the matrix to
[`policy_tree`](@ref) to learn a shallow decision tree,
`policy_tree(Γ, X; actions=1:K)`.

The table method is for logged data with known propensities of the action taken;
the outcome model is cross-fitted, and the standard justification treats units as
independent draws under a fixed logging policy. The `AdaptiveLog` method returns
scores from an adaptive experiment with an outcome model fitted on earlier units only
(as in [`adaptive_arm_values`](@ref)), so every score is conditionally unbiased
given the past. Learning a policy by maximizing the unweighted mean of these scores
is consistent, but with decaying assignment probabilities the scores are heavy
tailed and the learned policy can be unstable; Zhan, Ren, Athey and Zhou (2024)
propose re-weighting the scores for policy learning, which is not implemented here.
The value of a learned policy should be evaluated on data not used to learn it.

# Arguments
- `data`, `outcome::Symbol`, `action::Symbol`: table and columns, as in
  [`off_policy_value`](@ref).
- `lg::AdaptiveLog`: alternatively, the log of an adaptive experiment.

# Keywords
- `propensity::Symbol` (tables): column with the logged probability of the action
  taken.
- `covariates::Vector{Symbol}` (tables): covariates of the outcome model.
- `outcome_learner = RidgeLearner()`, `n_folds = 5`, `folds = nothing`,
  `action_levels = nothing` (tables): cross-fitting settings, as in
  [`off_policy_value`](@ref).
- `outcome_model = :running_mean`, `n_blocks = 10` (logs): sequential outcome model,
  as in [`adaptive_arm_values`](@ref).
- `rng::AbstractRNG = Random.default_rng()`: folds and learner seeds.

# Returns
- `Matrix{Float64}` of size `n × K`, rows in the order of `data` (time order for a
  log), columns in the order of the action levels.

# Examples
```julia
using DrSnow, DataFrames, StableRNGs
rng = StableRNG(1)
n = 1500
x1, x2 = randn(rng, n), randn(rng, n)
a = rand(rng, 1:3, n)                               # uniform logging policy
y = [(0.0, x1[i], 0.5 * x2[i])[a[i]] for i in 1:n] .+ randn(rng, n)
df = DataFrame(x1=x1, x2=x2, a=a, y=y, p=fill(1/3, n))
Γ = bandit_dr_scores(df, :y, :a; propensity=:p, covariates=[:x1, :x2],
                     rng=StableRNG(2))
tree = policy_tree(Γ, Matrix(df[:, [:x1, :x2]]); depth=2, actions=1:3,
                   covariates=[:x1, :x2])
```

# References
- Dudík, M., Langford, J., & Li, L. (2011). Doubly robust policy evaluation and
  learning. In *Proceedings of the 28th International Conference on Machine
  Learning* (pp. 1097–1104).
- Athey, S., & Wager, S. (2021). Policy learning with observational data.
  *Econometrica*, 89(1), 133–161.
- Zhan, R., Ren, Z., Athey, S., & Zhou, Z. (2024). Policy learning with adaptively
  collected data. *Management Science*, 70(8), 5270–5297.
"""
function bandit_dr_scores(data, outcome::Symbol, action::Symbol; propensity=nothing,
                          covariates=Symbol[], outcome_learner=RidgeLearner(),
                          n_folds::Integer=5, folds=nothing, action_levels=nothing,
                          rng::AbstractRNG=Random.default_rng())
    ctx = "bandit_dr_scores"
    L = _ad_logged(data, outcome, action; propensity=propensity, covariates=covariates,
                   action_levels=action_levels, context=ctx)
    M = _ad_crossfit_outcomes(L, outcome_learner, n_folds, folds, rng, ctx)
    Γ = copy(M)
    for i in eachindex(L.y)
        a = L.a[i]
        Γ[i, a] += (L.y[i] - M[i, a]) / L.p[i]
    end
    return Γ
end

function bandit_dr_scores(l::AdaptiveLog; outcome_model=:running_mean,
                          n_blocks::Integer=10, rng::AbstractRNG=Random.default_rng())
    d = _ad_data(l)
    return _ad_aipw_scores(d, _ad_muhat(d, outcome_model, n_blocks, rng,
                                        "bandit_dr_scores"))
end
