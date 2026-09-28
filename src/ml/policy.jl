# Policy learning with doubly-robust scores (Athey & Wager 2021): exhaustive search
# over depth-1 / depth-2 decision trees (as in the R package policytree), with the
# value of the learned policy evaluated out of fold.

struct _MLPolicyNode
    leaf::Bool
    action::Int          # leaf: action index (1-based column of the score matrix)
    var::Int             # split variable
    threshold::Float64   # go left when x[var] ≤ threshold
    left::Int
    right::Int
end

"""
    PolicyTree

A depth-1 or depth-2 treatment-assignment rule learned by [`policy_tree`](@ref).

A policy tree is a shallow decision tree ``\\pi: \\mathcal{X} \\to \\mathcal{A}`` that
maps covariates to one of a finite set of actions (for a binary treatment: `0` =
control, `1` = treat). Each internal node splits on a single covariate at a threshold
(go left when ``x_j \\le c``) and each leaf assigns an action. Trees of this depth are
the interpretable policy class studied by Athey and Wager (2021) and Zhou, Athey and
Wager (2023) and implemented in the R package policytree (Sverdrup et al. 2020).
`predict(tree, newdata)` returns the recommended action for each row; the tree is
printed as nested rules.

# Fields
- `nodes::Vector`: the tree's nodes in storage order, root first (internal nodes
  store the split variable index, threshold and child indices; leaves store the
  action index). The node type is internal; use `predict` and `show` rather than
  reading nodes directly.
- `depth::Int`: requested depth (1 or 2); a tree may have fewer splits when no split
  improves the objective.
- `covariates::Vector{Symbol}`: splitting variables, in the column order expected by
  `predict` with a matrix.
- `actions::Vector{Int}`: action labels, in the column order of the score matrix.
- `reward::Float64`: in-sample mean score of the tree's assignments,
  ``n^{-1}\\sum_i \\hat\\Gamma_i(\\pi(X_i))``; an optimistic estimate of the policy
  value because the tree was chosen to maximize it.
"""
struct PolicyTree
    nodes::Vector{_MLPolicyNode}
    depth::Int
    covariates::Vector{Symbol}
    actions::Vector{Int}
    reward::Float64
end

function _ml_tree_lines(io, t::PolicyTree, i, indent)
    nd = t.nodes[i]
    pad = "  "^indent
    if nd.leaf
        println(io, pad, "leaf: action = ", t.actions[nd.action])
    else
        @printf(io, "%s%s ≤ %.6g:\n", pad, t.covariates[nd.var], nd.threshold)
        _ml_tree_lines(io, t, nd.left, indent + 1)
        @printf(io, "%s%s > %.6g:\n", pad, t.covariates[nd.var], nd.threshold)
        _ml_tree_lines(io, t, nd.right, indent + 1)
    end
end

function Base.show(io::IO, ::MIME"text/plain", t::PolicyTree)
    println(io, "PolicyTree (depth $(t.depth))")
    _ml_tree_lines(io, t, 1, 1)
end

Base.show(io::IO, t::PolicyTree) = print(io, "PolicyTree(depth ", t.depth, ")")

"""
    predict(tree::PolicyTree, newdata) -> Vector{Int}

Recommended action for each row of `newdata` under a learned policy tree.

Each row is passed down the tree (left when the split covariate is at most the
threshold) and receives the action of the leaf it reaches. The method also accepts a
[`PolicyLearningResult`](@ref), in which case the full-sample tree `r.tree` is used.
The recommendation is only as good as the scores and covariates the tree was learned
from; its estimated value is reported by [`policy_tree`](@ref).

# Arguments
- `tree::PolicyTree`: a tree from [`policy_tree`](@ref) (or a
  [`PolicyLearningResult`](@ref)).
- `newdata::Union{AbstractDataFrame,AbstractMatrix}`: a `DataFrame` containing the
  tree's `covariates`, or a numeric matrix with those columns in the same order.

# Returns
- `Vector{Int}` of action labels (binary treatment: `0` control, `1` treat).

# Examples
```julia
using DrSnow, DataFrames, StableRNGs
rng = StableRNG(1)
n = 1000
df = DataFrame(x1=randn(rng, n), x2=randn(rng, n))
df.d = Float64.(rand(rng, n) .< 0.5)
df.y = df.x2 .+ df.x1 .* df.d .+ randn(rng, n)
r = policy_tree(df, :y, :d; covariates=[:x1, :x2], depth=1, rng=StableRNG(2))
predict(r.tree, DataFrame(x1=[-1.0, 1.0], x2=[0.0, 0.0]))
```
"""
function StatsAPI.predict(t::PolicyTree, newdata::AbstractDataFrame)
    require_columns(newdata, t.covariates; context="predict(PolicyTree)")
    return StatsAPI.predict(t, _ml_matrix(newdata, t.covariates;
                                          context="predict(PolicyTree)"))
end

function StatsAPI.predict(t::PolicyTree, X::AbstractMatrix)
    size(X, 2) == length(t.covariates) ||
        throw(DimensionMismatch("expected $(length(t.covariates)) covariate columns"))
    return [t.actions[_ml_tree_leaf(t, view(X, i, :))] for i in axes(X, 1)]
end

function _ml_tree_leaf(t::PolicyTree, x)
    i = 1
    while !t.nodes[i].leaf
        nd = t.nodes[i]
        i = x[nd.var] <= nd.threshold ? nd.left : nd.right
    end
    return t.nodes[i].action
end

# Best depth-1 split (or leaf) on the observations i with member[i] == want.
# Returns (reward, var, threshold, left action, right action); var = 0 for a leaf.
function _ml_best_depth1(Γ, X, orders, member, want, min_node)
    n, A = size(Γ)
    tot = zeros(A)
    m = 0
    @inbounds for i in 1:n
        if member[i] == want
            m += 1
            for a in 1:A
                tot[a] += Γ[i, a]
            end
        end
    end
    m == 0 && return (-Inf, 0, 0.0, 1, 1)
    la = argmax(tot)
    best = (tot[la], 0, 0.0, la, la)
    left = zeros(A)
    for j in axes(X, 2)
        ord = orders[j]
        fill!(left, 0.0)
        c = 0
        prev = 0
        @inbounds for t in 1:n
            i = ord[t]
            member[i] == want || continue
            if prev != 0 && c >= min_node && m - c >= min_node && X[i, j] > X[prev, j]
                bl = 1
                br = 1
                vl = left[1]
                vr = tot[1] - left[1]
                for a in 2:A
                    left[a] > vl && (vl = left[a]; bl = a)
                    tot[a] - left[a] > vr && (vr = tot[a] - left[a]; br = a)
                end
                if vl + vr > best[1] + 1e-12
                    best = (vl + vr, j, X[prev, j], bl, br)
                end
            end
            for a in 1:A
                left[a] += Γ[i, a]
            end
            c += 1
            prev = i
        end
    end
    return best
end

function _ml_depth1_nodes!(nodes, b)
    if b[2] == 0
        push!(nodes, _MLPolicyNode(true, b[4], 0, 0.0, 0, 0))
        return length(nodes)
    end
    root = length(nodes) + 1
    push!(nodes, _MLPolicyNode(false, 0, b[2], b[3], root + 1, root + 2))
    push!(nodes, _MLPolicyNode(true, b[4], 0, 0.0, 0, 0))
    push!(nodes, _MLPolicyNode(true, b[5], 0, 0.0, 0, 0))
    return root
end

"""Exhaustive depth-1 or depth-2 policy tree maximizing Σᵢ Γ[i, π(Xᵢ)]."""
function _ml_fit_policy_tree(Γ::AbstractMatrix, X::AbstractMatrix, depth::Int,
                             min_node::Int, split_step::Int)
    n = size(Γ, 1)
    orders = [sortperm(view(X, :, j)) for j in axes(X, 2)]
    member = fill(true, n)
    nodes = _MLPolicyNode[]
    if depth == 1
        b = _ml_best_depth1(Γ, X, orders, member, true, min_node)
        _ml_depth1_nodes!(nodes, b)
        return nodes, b[1]
    end
    # depth 2: every root split, best depth-1 subtree on each side
    inleft = fill(false, n)
    best = (-Inf, 0, 0.0, nothing, nothing)
    for j in axes(X, 2)
        ord = orders[j]
        fill!(inleft, false)
        for t in 1:(n - 1)
            i = ord[t]
            inleft[i] = true
            X[ord[t + 1], j] > X[i, j] || continue
            (t >= min_node && n - t >= min_node) || continue
            (t % split_step == 0) || continue
            bl = _ml_best_depth1(Γ, X, orders, inleft, true, min_node)
            br = _ml_best_depth1(Γ, X, orders, inleft, false, min_node)
            v = bl[1] + br[1]
            if v > best[1] + 1e-12
                best = (v, j, X[i, j], bl, br)
            end
        end
    end
    if best[2] == 0
        b = _ml_best_depth1(Γ, X, orders, member, true, min_node)
        _ml_depth1_nodes!(nodes, b)
        return nodes, b[1]
    end
    push!(nodes, _MLPolicyNode(false, 0, best[2], best[3], 0, 0))
    l = _ml_depth1_nodes!(nodes, best[4])
    r = _ml_depth1_nodes!(nodes, best[5])
    nodes[1] = _MLPolicyNode(false, 0, best[2], best[3], l, r)
    return nodes, best[1]
end

"""
    PolicyLearningResult <: CausalEstimate

Result of [`policy_tree`](@ref) on data: the learned treatment rule, the doubly
robust scores used to learn it, and out-of-fold estimates of its value.

The three coefficients are, for the out-of-fold rule ``\\hat\\pi`` (the rule applied
to fold ``k`` is learned without fold ``k``):

1. `value(policy)`: ``V(\\hat\\pi) = E[Y(\\hat\\pi(X))]``, the mean outcome if the rule
   were applied, estimated by ``n^{-1}\\sum_i \\hat\\Gamma_i(\\hat\\pi_{-k(i)}(X_i))``;
2. `value(policy) - value(treat all)`: ``V(\\hat\\pi) - E[Y(1)]``;
3. `value(policy) - value(treat none)`: ``V(\\hat\\pi) - E[Y(0)]``.

The covariance is the sample (or cluster-robust) covariance of the corresponding
score contrasts divided by ``n``. The object supports `coef`, `vcov`, `stderror`,
`confint(r; level)`, `coeftable`, `nobs`, `dof_residual` (`G - 1` with clusters,
otherwise `Inf`) and `predict(r, newdata)` (the full-sample tree).

# Fields
- `tree::PolicyTree`: rule learned on all observations; use it for deployment.
- `fold_trees::Vector{PolicyTree}`: the tree learned without fold `k`, applied to
  fold `k` for the out-of-fold evaluation.
- `scores::Matrix{Float64}`: cross-fitted doubly robust scores ``\\hat\\Gamma``
  (`n × 2`: column 1 control, column 2 treatment).
- `oof_actions::Vector{Int}`: out-of-fold recommended actions (`0`/`1`).
- `coef::Vector{Float64}`, `vcov::Matrix{Float64}`: the three value estimates and
  their covariance.
- `folds::Vector{Int}`: fold id of each observation.
- `n_trimmed::Int`: number of propensity predictions clipped to `[trim, 1 - trim]`.
- `cluster::Union{Nothing,Vector{Int}}`, `n_clusters::Int`: cluster index of each
  observation (or `nothing`) and the number of clusters.
- `learners::Vector{Pair{Symbol,String}}`: description of the nuisance learners.
"""
struct PolicyLearningResult <: CausalEstimate
    tree::PolicyTree
    fold_trees::Vector{PolicyTree}
    scores::Matrix{Float64}
    oof_actions::Vector{Int}
    coef::Vector{Float64}
    vcov::Matrix{Float64}
    folds::Vector{Int}
    n_trimmed::Int
    cluster::Union{Nothing,Vector{Int}}
    n_clusters::Int
    learners::Vector{Pair{Symbol,String}}
end

StatsAPI.coef(r::PolicyLearningResult) = r.coef
StatsAPI.vcov(r::PolicyLearningResult) = r.vcov
StatsAPI.coefnames(::PolicyLearningResult) =
    ["value(policy)", "value(policy) - value(treat all)",
     "value(policy) - value(treat none)"]
StatsAPI.nobs(r::PolicyLearningResult) = size(r.scores, 1)
StatsAPI.dof_residual(r::PolicyLearningResult) =
    r.cluster === nothing ? Inf : r.n_clusters - 1.0
estimand(::PolicyLearningResult) = "value of the learned treatment rule (out of fold)"
method_name(r::PolicyLearningResult) =
    "Doubly-robust policy tree (Athey–Wager), depth $(r.tree.depth)"

function show_details(io::IO, r::PolicyLearningResult)
    println(io)
    show(io, MIME"text/plain"(), r.tree)
    @printf(io, "Share assigned to treatment (out of fold): %.3f\n",
            mean(r.oof_actions .== 1))
    return nothing
end

StatsAPI.predict(r::PolicyLearningResult, newdata) = StatsAPI.predict(r.tree, newdata)

"""
    policy_tree(data, outcome, treatment; covariates=Symbol[],
                policy_covariates=covariates, depth=2, outcome_learner=LassoLearner(),
                propensity_learner=PenalizedLogisticLearner(), trim=0.01,
                min_node_size=1, split_step=1, stratify=true, n_folds=5, folds=nothing,
                cluster=nothing, rng=Random.default_rng(),
                parallel=true) -> PolicyLearningResult
    policy_tree(scores::AbstractMatrix, X::AbstractMatrix; depth=2, min_node_size=1,
                split_step=1, covariates=[:x1, ...],
                actions=0:(size(scores, 2) - 1)) -> PolicyTree

Learn an interpretable treatment-assignment rule by exact search over shallow
decision trees that maximize a doubly robust estimate of the policy value (Athey &
Wager 2021; Zhou, Athey & Wager 2023).

The research question is which units should be treated when treatment can be targeted
on observed characteristics. For a policy ``\\pi`` mapping covariates to actions, the
estimand is its value, the mean outcome if everyone were assigned by ``\\pi``, and the
learner seeks the best rule within the class ``\\Pi`` of depth-1 or depth-2 trees:

```math
V(\\pi) = E\\big[Y(\\pi(X))\\big], \\qquad
\\pi^\\ast = \\arg\\max_{\\pi \\in \\Pi} V(\\pi).
```

With a binary treatment, identification requires unconfoundedness given `covariates`,
overlap and SUTVA, as in [`dml_irm`](@ref); in a randomized experiment these hold by
design. The value is identified by the doubly robust (AIPW) scores

```math
\\Gamma_i(1) = \\mu_1(X_i) + \\frac{D_i\\{Y_i - \\mu_1(X_i)\\}}{e(X_i)}, \\qquad
\\Gamma_i(0) = \\mu_0(X_i) + \\frac{(1 - D_i)\\{Y_i - \\mu_0(X_i)\\}}{1 - e(X_i)},
```

with ``V(\\pi) = E[\\Gamma(\\pi(X))]``, where ``\\mu_d(x) = E[Y \\mid D = d, X = x]``
and ``e(x) = P(D = 1 \\mid X = x)`` is the propensity score. The nuisances are
cross-fitted and the tree maximizing ``\\sum_i \\hat\\Gamma_i(\\pi(X_i))`` is found by
exhaustive search, as in the R package policytree (Sverdrup et al. 2020). This is the
doubly robust empirical welfare maximization of Athey and Wager (2021), whose regret
relative to the best rule in the class is of order ``\\sqrt{\\mathrm{VC}(\\Pi)/n}``
under their rate conditions; Kitagawa and Tetenov (2018) study the known-propensity
case. The rule is only as good as the class searched and the `policy_covariates`
offered, and ``\\pi^\\ast`` is optimal for the mean outcome, not for other welfare
criteria or budget constraints.

The tree fitted on all observations is returned for use, but its in-sample reward is
optimistic. Its value is therefore evaluated out of fold: for each fold ``k`` a tree is
learned on the other folds and applied to fold ``k``, and the mean score of these
assignments estimates ``V(\\hat\\pi)``, which is compared with treating everyone and no
one; standard errors come from the sample (or cluster-robust) variance of the score
contrasts. Because scores in different folds share cross-fitted nuisance fits, this
evaluation is approximately, not exactly, honest; for a formal evaluation hold out a
separate sample. Search cost is ``O(p\\,n)`` for depth 1 and ``O(p^2 n^2)`` for depth
2, so use `split_step` for large samples. For a smooth prioritization rule instead of
a tree, rank units by a CATE estimate and evaluate it with
[`rank_average_treatment_effect`](@ref) or [`policy_value`](@ref).

The matrix method runs only the tree search on user-supplied rewards `scores`
(`n × A`, one column per action), which allows multi-action problems and scores from
other estimators (e.g. [`double_robust_scores`](@ref) of a causal forest).

# Arguments
- `data`: a `DataFrame` with one row per unit.
- `outcome::Symbol`: numeric outcome column ``Y`` (larger is better).
- `treatment::Symbol`: treatment column ``D``, coded `0`/`1`.
- `scores::AbstractMatrix` (matrix method): `n × A` rewards of each of the `A`
  actions for each unit.
- `X::AbstractMatrix` (matrix method): `n × p` splitting covariates.

# Keywords
- `covariates::Vector{Symbol} = Symbol[]`: confounders used by the nuisance learners
  (data method); in the matrix method, names for the columns of `X` (default `:x1`,
  `:x2`, …).
- `policy_covariates::Vector{Symbol} = covariates`: variables the rule may split on;
  restrict them to variables that can legitimately be used for targeting.
- `depth::Integer = 2`: tree depth, 1 or 2.
- `outcome_learner = LassoLearner()`: [`NuisanceLearner`](@ref) for
  ``\\mu_0`` and ``\\mu_1``.
- `propensity_learner = PenalizedLogisticLearner()`: learner for the propensity
  score (must implement [`fitpredict_proba`](@ref)).
- `trim::Real = 0.01`: propensity clipping threshold; clipping keeps the scores
  finite but does not repair limited overlap.
- `min_node_size::Integer = 1`: minimum number of observations per leaf.
- `split_step::Integer = 1`: consider only every `split_step`-th candidate root split
  point in depth-2 search (faster, approximate).
- `stratify::Bool = true`, `n_folds::Integer = 5`, `folds = nothing`,
  `cluster = nothing`: cross-fitting folds (stratified by treatment, grouped by
  cluster), as in [`dml_irm`](@ref) with one repetition; `cluster` also makes the
  standard errors cluster-robust.
- `rng::AbstractRNG = Random.default_rng()`, `parallel::Bool = true`: random-number
  generator (per-task seeds drawn up front) and threading.
- `actions = 0:(size(scores, 2) - 1)` (matrix method): action labels, one per column
  of `scores`.

# Returns
- Data method: [`PolicyLearningResult`](@ref), with the full-sample tree in `tree`
  and the out-of-fold value estimates as coefficients.
- Matrix method: a [`PolicyTree`](@ref).

# Examples
```julia
using DrSnow, DataFrames, StableRNGs
rng = StableRNG(1)
n = 1000
df = DataFrame(x1=randn(rng, n), x2=randn(rng, n), x3=randn(rng, n))
df.d = Float64.(rand(rng, n) .< 0.5)
df.y = df.x2 .+ df.x1 .* df.d .+ randn(rng, n)
r = policy_tree(df, :y, :d; covariates=[:x1, :x2, :x3], depth=2,
                split_step=10, rng=StableRNG(2))
r.tree
coeftable(r)
predict(r, DataFrame(x1=[-1.0, 1.0], x2=[0.0, 0.0], x3=[0.0, 0.0]))
```

# References
- Athey, S., & Wager, S. (2021). Policy learning with observational data.
  *Econometrica*, 89(1), 133–161.
- Zhou, Z., Athey, S., & Wager, S. (2023). Offline multi-action policy learning:
  Generalization and optimization. *Operations Research*, 71(1), 148–183.
- Kitagawa, T., & Tetenov, A. (2018). Who should be treated? Empirical welfare
  maximization methods for treatment choice. *Econometrica*, 86(2), 591–616.
- Sverdrup, E., Kanodia, A., Zhou, Z., Athey, S., & Wager, S. (2020). policytree:
  Policy learning via doubly robust empirical welfare maximization over trees.
  *Journal of Open Source Software*, 5(50), 2232.
- Robins, J. M., Rotnitzky, A., & Zhao, L. P. (1994). Estimation of regression
  coefficients when some regressors are not always observed. *Journal of the
  American Statistical Association*, 89(427), 846–866.
"""
function policy_tree(scores::AbstractMatrix, X::AbstractMatrix; depth::Integer=2,
                     min_node_size::Integer=1, split_step::Integer=1,
                     covariates=[Symbol("x", j) for j in 1:size(X, 2)],
                     actions=collect(0:(size(scores, 2) - 1)))
    size(scores, 1) == size(X, 1) ||
        throw(DimensionMismatch("scores and X must have the same number of rows"))
    depth in (1, 2) || throw(ArgumentError("depth must be 1 or 2"))
    min_node_size >= 1 || throw(ArgumentError("min_node_size must be at least 1"))
    split_step >= 1 || throw(ArgumentError("split_step must be at least 1"))
    size(scores, 2) >= 2 || throw(ArgumentError("need at least two actions"))
    size(X, 2) >= 1 || throw(ArgumentError("need at least one splitting covariate"))
    length(actions) == size(scores, 2) ||
        throw(ArgumentError("need one action label per column of scores"))
    all(isfinite, scores) && all(isfinite, X) ||
        throw(ArgumentError("scores and X must be finite"))
    Γ = Matrix{Float64}(scores)
    nodes, reward = _ml_fit_policy_tree(Γ, Matrix{Float64}(X), Int(depth),
                                        Int(min_node_size), Int(split_step))
    return PolicyTree(nodes, Int(depth), Symbol.(collect(covariates)),
                      collect(Int, actions), reward / size(Γ, 1))
end

function policy_tree(data, outcome::Symbol, treatment::Symbol; covariates=Symbol[],
                     policy_covariates=covariates, depth::Integer=2,
                     outcome_learner=LassoLearner(),
                     propensity_learner=PenalizedLogisticLearner(), trim::Real=0.01,
                     min_node_size::Integer=1, split_step::Integer=1,
                     stratify::Bool=true, n_folds::Integer=5, folds=nothing,
                     cluster=nothing, rng::AbstractRNG=Random.default_rng(),
                     parallel::Bool=true)
    ctx = "policy_tree"
    trim = _ml_check_trim(trim)
    pcovs = Symbol.(collect(policy_covariates))
    isempty(pcovs) && throw(ArgumentError("$ctx: policy_covariates must not be empty"))
    require_columns(data, vcat(treatment, pcovs); context=ctx)
    d = _ml_column(data, treatment; context=ctx)
    _ml_check_binary_col(d, treatment, ctx)
    covs, X, cid, G, F = _ml_setup(data, [outcome, treatment], covariates, cluster, folds,
                                   n_folds, 1, rng, stratify ? d : nothing; context=ctx)
    size(F, 2) == 1 || throw(ArgumentError("$ctx: supply a single column of fold ids"))
    y = _ml_column(data, outcome; context=ctx)
    P = _ml_matrix(data, pcovs; context=ctx)
    fold = F[:, 1]
    K = maximum(fold)
    seeds = _ml_seeds(rng, K, 3, 1)
    μ0, μ1, π, nt, _ = _ml_aipw_nuisances(y, d, X, fold, view(seeds, :, :, 1),
                                          outcome_learner, propensity_learner, trim,
                                          parallel, ctx)
    Γ = hcat(μ0 .+ (1 .- d) .* (y .- μ0) ./ (1 .- π), μ1 .+ d .* (y .- μ1) ./ π)
    tree = policy_tree(Γ, P; depth=depth, min_node_size=min_node_size,
                       split_step=split_step, covariates=pcovs, actions=[0, 1])
    oof = zeros(Int, length(y))
    fold_trees = Vector{PolicyTree}(undef, K)
    for k in 1:K
        tr = fold .!= k
        te = .!tr
        tk = policy_tree(Γ[tr, :], P[tr, :]; depth=depth, min_node_size=min_node_size,
                         split_step=split_step, covariates=pcovs, actions=[0, 1])
        oof[te] .= StatsAPI.predict(tk, P[te, :])
        fold_trees[k] = tk
    end
    v = [Γ[i, oof[i] + 1] for i in eachindex(oof)]
    comps = hcat(v, v .- Γ[:, 2], v .- Γ[:, 1])
    θ = vec(mean(comps; dims=1))
    V = _ml_score_cov(comps .- θ', -ones(3), cid, G)
    learners = [:ml_g0 => _ml_learner_name(outcome_learner),
                :ml_g1 => _ml_learner_name(outcome_learner),
                :ml_m => _ml_learner_name(propensity_learner)]
    return PolicyLearningResult(tree, fold_trees, Γ, oof, θ, V, fold, nt, cid, G,
                                learners)
end
