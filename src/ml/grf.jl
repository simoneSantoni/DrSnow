# Generalized random forests: user-facing regression, causal and instrumental forests
# (grf's `regression_forest`, `causal_forest`, `instrumental_forest`), prediction with
# little-bags variance estimates, and parameter tuning. The engine is in grf_core.jl.

"""
    GeneralizedRandomForest

Abstract supertype of the fitted generalized random forests [`RegressionForest`](@ref),
[`CausalForest`](@ref) and [`InstrumentalForest`](@ref).

A generalized random forest (Athey, Tibshirani & Wager 2019) estimates a parameter
``\\theta(x)`` defined by a local moment condition
``E[\\psi_{\\theta(x)}(O_i) \\mid X_i = x] = 0`` by solving the sample moment weighted by
forest weights ``\\alpha_i(x)``, the share of trees in which training unit ``i`` falls into
the same leaf as ``x``. The trees are honest (one half of each subsample places the splits,
the other half populates the leaves), are grown on subsamples drawn without replacement,
and use splitting rules targeted at heterogeneity in ``\\theta(x)`` rather than at
prediction of the outcome. DrSnow's engine is a line-by-line port of the C++ core of the R
package grf 2.6.1, and the post-estimation formulas are validated against grf evaluated on
the same nuisance estimates.

Every forest supports `predict(f)` (out-of-bag predictions for the training rows) and
`predict(f, newdata)` (predictions at new points), [`predict_interval`](@ref) (predictions
with bootstrap-of-little-bags standard errors and pointwise confidence intervals),
[`variable_importance`](@ref), [`split_frequencies`](@ref) and `nobs`. Tuning parameters
that were actually used are stored in the `params` field of every concrete forest.

# References
- Athey, S., Tibshirani, J., & Wager, S. (2019). Generalized random forests. *The Annals of
  Statistics*, 47(2), 1148–1178.
- Breiman, L. (2001). Random forests. *Machine Learning*, 45(1), 5–32.
"""
abstract type GeneralizedRandomForest end

"""
    RegressionForest <: GeneralizedRandomForest

Honest regression forest estimating the conditional mean ``\\mu(x) = E[Y \\mid X = x]``,
returned by [`regression_forest`](@ref).

The object stores the trained trees, the training data (in the order supplied), the
out-of-bag predictions ``\\hat\\mu(X_i)`` and the parameters actually used, so that it can
predict at new points, produce little-bags variance estimates and be passed to
[`test_calibration`](@ref) or [`variable_importance`](@ref).

# Fields
- `forest`: the trained trees (internal engine type; not part of the public API).
- `order::Vector{Int}`: canonical (lexicographic) row order used internally, which makes
  the fit invariant to the row order of the data.
- `covariates::Vector{Symbol}`, `outcome::Symbol`: column names (`:x1, :x2, …` and `:y` for
  the matrix method).
- `X::Matrix{Float64}`, `Y::Vector{Float64}`: training data in the order supplied (`NaN`
  marks a missing covariate).
- `sample_weights::Union{Nothing,Vector{Float64}}`: observation weights, if any.
- `cluster::Union{Nothing,Vector{Int}}`, `n_clusters::Int`: cluster index `1:G` per row (or
  `nothing`) and `G`.
- `equalize_cluster_weights::Bool`: whether every cluster contributed the same number of
  units to each subsample.
- `predictions::Vector{Float64}`: out-of-bag predictions ``\\hat\\mu(X_i)``.
- `debiased_error::Vector{Float64}`: out-of-bag squared error minus its estimated Monte
  Carlo excess from using finitely many trees (grf's `debiased.error`).
- `params::NamedTuple`: the forest parameters actually used, after tuning: `num_trees`
  (rounded up to a multiple of `ci_group_size`), `sample_fraction`, `mtry`,
  `min_node_size`, `honesty`, `honesty_fraction`, `honesty_prune_leaves`, `alpha`,
  `imbalance_penalty`, `ci_group_size` and `stabilize_splits`.
- `tuning::Union{Nothing,NamedTuple}`: `nothing` without tuning, otherwise
  `(status, params, error)` with `status` `"tuned"`, `"default"` or `"failure"`.
- `seed::UInt64`: seed drawn from `rng` that generated the forest.
"""
struct RegressionForest <: GeneralizedRandomForest
    forest::_GRFForest
    order::Vector{Int}
    covariates::Vector{Symbol}
    outcome::Symbol
    X::Matrix{Float64}
    Y::Vector{Float64}
    sample_weights::Union{Nothing,Vector{Float64}}
    cluster::Union{Nothing,Vector{Int}}
    n_clusters::Int
    equalize_cluster_weights::Bool
    predictions::Vector{Float64}
    debiased_error::Vector{Float64}
    params::NamedTuple
    tuning::Union{Nothing,NamedTuple}
    seed::UInt64
end

"""
    CausalForest <: GeneralizedRandomForest

Causal forest estimating the conditional average treatment effect
``\\tau(x) = E[Y(1) - Y(0) \\mid X = x]`` (for a continuous treatment, the conditional
average partial effect), returned by [`causal_forest`](@ref).

The object holds the trained forest together with the locally centred data it was grown on:
the out-of-bag (or cross-fitted, or user-supplied) nuisance estimates
``\\hat Y(X_i) = \\hat E[Y \\mid X_i]`` and ``\\hat W(X_i) = \\hat E[W \\mid X_i]`` and the
out-of-bag CATE estimates ``\\hat\\tau(X_i)``. These are the inputs of all post-estimation
tools: [`get_scores`](@ref), [`average_treatment_effect`](@ref),
[`best_linear_projection`](@ref), [`test_calibration`](@ref),
[`rank_average_treatment_effect`](@ref), [`double_robust_scores`](@ref) and
[`policy_value`](@ref).

# Fields
- `forest`: the trained trees (internal engine type; not part of the public API).
- `order::Vector{Int}`: canonical row order used internally.
- `covariates::Vector{Symbol}`, `outcome::Symbol`, `treatment::Symbol`: column names.
- `X::Matrix{Float64}`, `Y::Vector{Float64}`, `W::Vector{Float64}`: training data in the
  order supplied.
- `Y_hat::Vector{Float64}`, `W_hat::Vector{Float64}`: estimates of ``E[Y \\mid X]`` and
  ``E[W \\mid X]`` used for local centering (``W_hat`` is the propensity score for a binary
  treatment).
- `sample_weights`, `cluster`, `n_clusters`, `equalize_cluster_weights`: as in
  [`RegressionForest`](@ref).
- `predictions::Vector{Float64}`: out-of-bag CATE estimates ``\\hat\\tau(X_i)``.
- `debiased_error::Vector{Float64}`: out-of-bag R-loss error estimates, debiased for the
  Monte Carlo error of a finite forest.
- `params::NamedTuple`: forest parameters actually used (`num_trees`, `sample_fraction`,
  `mtry`, `min_node_size`, `honesty`, `honesty_fraction`, `honesty_prune_leaves`, `alpha`,
  `imbalance_penalty`, `ci_group_size`, `stabilize_splits`); there is no separate
  `num_trees` field.
- `tuning::Union{Nothing,NamedTuple}`: tuning summary, as in [`RegressionForest`](@ref).
- `seed::UInt64`: seed drawn from `rng`.
- `nuisance::Vector{Pair{Symbol,String}}`: how `Y_hat` and `W_hat` were obtained
  (out-of-bag regression forest, cross-fitted learner, supplied vector, …).
"""
struct CausalForest <: GeneralizedRandomForest
    forest::_GRFForest
    order::Vector{Int}
    covariates::Vector{Symbol}
    outcome::Symbol
    treatment::Symbol
    X::Matrix{Float64}
    Y::Vector{Float64}
    W::Vector{Float64}
    Y_hat::Vector{Float64}
    W_hat::Vector{Float64}
    sample_weights::Union{Nothing,Vector{Float64}}
    cluster::Union{Nothing,Vector{Int}}
    n_clusters::Int
    equalize_cluster_weights::Bool
    predictions::Vector{Float64}
    debiased_error::Vector{Float64}
    params::NamedTuple
    tuning::Union{Nothing,NamedTuple}
    seed::UInt64
    nuisance::Vector{Pair{Symbol,String}}
end

"""
    InstrumentalForest <: GeneralizedRandomForest

Instrumental forest estimating the conditional local average treatment effect
``\\tau(x) = \\mathrm{Cov}(Y, Z \\mid X = x) / \\mathrm{Cov}(W, Z \\mid X = x)`` (Athey,
Tibshirani & Wager 2019, Section 7), returned by [`instrumental_forest`](@ref).

The object stores the trained forest, the out-of-bag conditional LATE estimates and the
centering nuisances ``\\hat Y``, ``\\hat W`` and ``\\hat Z``; with a binary instrument it
supports [`average_treatment_effect`](@ref), [`best_linear_projection`](@ref) and
[`rank_average_treatment_effect`](@ref) through the doubly robust scores of
[`get_scores`](@ref).

# Fields
- `forest`, `order`: trained trees and internal row order (not public API).
- `covariates::Vector{Symbol}`, `outcome::Symbol`, `treatment::Symbol`,
  `instrument::Symbol`: column names.
- `X::Matrix{Float64}`, `Y`, `W`, `Z::Vector{Float64}`: training data in the order
  supplied.
- `Y_hat`, `W_hat`, `Z_hat::Vector{Float64}`: centering estimates of ``E[Y \\mid X]``,
  ``E[W \\mid X]`` and ``E[Z \\mid X]``.
- `sample_weights`, `cluster`, `n_clusters`, `equalize_cluster_weights`: as in
  [`RegressionForest`](@ref).
- `predictions::Vector{Float64}`: out-of-bag conditional LATE estimates.
- `debiased_error::Vector{Float64}`: debiased out-of-bag error estimates.
- `params::NamedTuple`, `tuning`, `seed::UInt64`: forest parameters actually used
  (including `num_trees`), tuning summary and seed, as in [`CausalForest`](@ref).
- `nuisance::Vector{Pair{Symbol,String}}`: how `Y_hat`, `W_hat` and `Z_hat` were obtained.
"""
struct InstrumentalForest <: GeneralizedRandomForest
    forest::_GRFForest
    order::Vector{Int}
    covariates::Vector{Symbol}
    outcome::Symbol
    treatment::Symbol
    instrument::Symbol
    X::Matrix{Float64}
    Y::Vector{Float64}
    W::Vector{Float64}
    Z::Vector{Float64}
    Y_hat::Vector{Float64}
    W_hat::Vector{Float64}
    Z_hat::Vector{Float64}
    sample_weights::Union{Nothing,Vector{Float64}}
    cluster::Union{Nothing,Vector{Int}}
    n_clusters::Int
    equalize_cluster_weights::Bool
    predictions::Vector{Float64}
    debiased_error::Vector{Float64}
    params::NamedTuple
    tuning::Union{Nothing,NamedTuple}
    seed::UInt64
    nuisance::Vector{Pair{Symbol,String}}
end

StatsAPI.nobs(f::GeneralizedRandomForest) = length(f.Y)

_ml_grf_label(::RegressionForest) = "Regression forest"
_ml_grf_label(::CausalForest) = "Causal forest"
_ml_grf_label(::InstrumentalForest) = "Instrumental forest"
_ml_grf_target(::RegressionForest) = "E[Y | X]"
_ml_grf_target(::CausalForest) = "CATE"
_ml_grf_target(::InstrumentalForest) = "conditional LATE"

function Base.show(io::IO, ::MIME"text/plain", f::GeneralizedRandomForest)
    println(io, _ml_grf_label(f), " (generalized random forest, grf algorithm)")
    println(io, "Observations: ", nobs(f), ", covariates: ", length(f.covariates),
            ", trees: ", length(f.forest.trees))
    f.cluster === nothing || println(io, "Clusters: ", f.n_clusters)
    pr = f.params
    @printf(io, "sample_fraction = %.3g, mtry = %d, min_node_size = %d, ",
            pr.sample_fraction, pr.mtry, pr.min_node_size)
    @printf(io, "honesty_fraction = %.3g, alpha = %.3g, imbalance_penalty = %.3g\n",
            pr.honesty_fraction, pr.alpha, pr.imbalance_penalty)
    if hasproperty(f, :nuisance)
        for (nm, s) in f.nuisance
            println(io, "  ", nm, ": ", s)
        end
    end
    ok = filter(isfinite, f.predictions)
    if !isempty(ok)
        q = quantile(ok, [0.1, 0.5, 0.9])
        @printf(io, "Out-of-bag %s predictions: 10%% %.4g, median %.4g, 90%% %.4g\n",
                _ml_grf_target(f), q...)
    end
    f.tuning === nothing ||
        println(io, "Tuning: ", f.tuning.status, " (", join(string.(keys(f.tuning.params)),
                                                            ", "), ")")
    return nothing
end

Base.show(io::IO, f::GeneralizedRandomForest) =
    print(io, nameof(typeof(f)), "(n = ", nobs(f), ", trees = ",
          length(f.forest.trees), ")")

# ------------------------------------------------------------------ data helpers

"""Covariate matrix for forests: `missing` becomes `NaN` (handled by the splits)."""
function _ml_grf_matrix(data, cols::Vector{Symbol}; context)
    n = nrow(data)
    X = Matrix{Float64}(undef, n, length(cols))
    for (j, c) in enumerate(cols)
        v = data[!, c]
        T = nonmissingtype(eltype(v))
        T <: Real || throw(ArgumentError("$(context): column $(c) must be numeric " *
                                         "(got $(T)); encode categories as dummies"))
        for i in 1:n
            x = v[i]
            X[i, j] = ismissing(x) ? NaN : Float64(x)
        end
        any(isinf, view(X, :, j)) &&
            throw(ArgumentError("$(context): column $(c) contains infinite values"))
    end
    return X
end

function _ml_grf_check_X(X::AbstractMatrix, context)
    size(X, 2) >= 1 || throw(ArgumentError("$(context): at least one covariate is " *
                                           "required"))
    any(isinf, X) && throw(ArgumentError("$(context): covariates contain infinite values"))
    return Matrix{Float64}(X)
end

function _ml_grf_vector(v, n, name, context)
    v === nothing && return nothing
    length(v) == n ||
        throw(DimensionMismatch("$(context): $(name) must have length $n"))
    any(ismissing, v) && throw(ArgumentError("$(context): $(name) has missing values"))
    out = Float64.(v)
    all(isfinite, out) || throw(ArgumentError("$(context): $(name) must be finite"))
    return out
end

"""Lexicographic row order of the data (makes forests invariant to row order)."""
function _ml_grf_canonical_order(cols::Vector{<:AbstractVector})
    n = length(cols[1])
    lt = function (a, b)
        for c in cols
            x, y = c[a], c[b]
            isequal(x, y) && continue
            return isless(x, y)
        end
        return false
    end
    return sort!(collect(1:n); lt=lt, alg=MergeSort)
end

# ------------------------------------------------------------------ parameters

_ml_grf_default_mtry(p) = min(ceil(Int, sqrt(p) + 20), p)

function _ml_grf_params(p; num_trees, sample_fraction, mtry, min_node_size, honesty,
                        honesty_fraction, honesty_prune_leaves, alpha, imbalance_penalty,
                        ci_group_size, stabilize_splits=true, context)
    mtry = mtry === nothing ? _ml_grf_default_mtry(p) : Int(mtry)
    1 <= mtry <= p || throw(ArgumentError("$(context): mtry must be in 1:$(p)"))
    num_trees >= 1 || throw(ArgumentError("$(context): num_trees must be positive"))
    ci_group_size >= 1 || throw(ArgumentError("$(context): ci_group_size must be ≥ 1"))
    0 < sample_fraction <= 1 ||
        throw(ArgumentError("$(context): sample_fraction must be in (0, 1]"))
    (ci_group_size > 1 && sample_fraction > 0.5) &&
        throw(ArgumentError("$(context): sample_fraction must be at most 0.5 when " *
                            "ci_group_size > 1 (variance estimates use half-samples)"))
    min_node_size >= 1 || throw(ArgumentError("$(context): min_node_size must be ≥ 1"))
    0 < honesty_fraction < 1 ||
        throw(ArgumentError("$(context): honesty_fraction must be in (0, 1)"))
    0 <= alpha < 0.25 || throw(ArgumentError("$(context): alpha must be in [0, 0.25)"))
    imbalance_penalty >= 0 ||
        throw(ArgumentError("$(context): imbalance_penalty must be non-negative"))
    nt = Int(num_trees)
    nt += nt % ci_group_size        # as grf's ForestOptions
    return (num_trees=nt, sample_fraction=Float64(sample_fraction), mtry=mtry,
            min_node_size=Int(min_node_size), honesty=Bool(honesty),
            honesty_fraction=Float64(honesty_fraction),
            honesty_prune_leaves=Bool(honesty_prune_leaves), alpha=Float64(alpha),
            imbalance_penalty=Float64(imbalance_penalty),
            ci_group_size=Int(ci_group_size), stabilize_splits=Bool(stabilize_splits))
end

function _ml_grf_options(kind, pr::NamedTuple, clusters, spc; reduced_form_weight=0.0)
    return _GRFOptions(kind, pr.num_trees, pr.ci_group_size, pr.sample_fraction, pr.mtry,
                       pr.min_node_size, pr.honesty, pr.honesty_fraction,
                       pr.honesty_prune_leaves, pr.alpha, pr.imbalance_penalty,
                       pr.stabilize_splits, Float64(reduced_form_weight), clusters, spc)
end

# Clusters (ids 1:G in internal row order) → samples by cluster and samples per cluster.
function _ml_grf_clusters(cid, equalize::Bool, has_weights::Bool, context)
    if cid === nothing
        equalize && throw(ArgumentError("$(context): equalize_cluster_weights requires " *
                                        "`cluster`"))
        return Vector{Int}[], 0
    end
    equalize && has_weights &&
        throw(ArgumentError("$(context): sample weights cannot be combined with " *
                            "equalize_cluster_weights = true"))
    G = maximum(cid)
    cl = [Int[] for _ in 1:G]
    for (i, g) in enumerate(cid)
        push!(cl[g], i)
    end
    sizes = length.(cl)
    return cl, equalize ? minimum(sizes) : maximum(sizes)
end

"""Train a forest on internally ordered data; returns (forest, OOB preds, OOB errors)."""
function _ml_grf_train(kind, X, y, pr, clusters, spc, rng; w=Float64[], z=Float64[],
                       wt=nothing, parallel=true, error=true, reduced_form_weight=0.0,
                       context)
    n = size(X, 1)
    sf = pr.sample_fraction
    if floor(Int, n * sf) < 1 ||
       (pr.honesty && (n * sf * pr.honesty_fraction < 1 ||
                       n * sf * (1 - pr.honesty_fraction) < 1))
        throw(ArgumentError("$(context): too few observations for sample_fraction = " *
                            "$(sf) and honesty_fraction = $(pr.honesty_fraction)"))
    end
    opts = _ml_grf_options(kind, pr, clusters, spc; reduced_form_weight=reduced_form_weight)
    data = _GRFData(X, y; w=w, z=z, wt=wt)
    seeds = task_seeds(rng, pr.num_trees ÷ pr.ci_group_size)
    forest = _grf_train(data, opts, seeds; parallel=parallel)
    L = _grf_leaf_matrix(forest, data.X, true, parallel)
    pred, _, err = _grf_collect(forest, L; estimate_error=error, yerr=data.y,
                                zerr=kind === :regression ? nothing : data.z,
                                parallel=parallel)
    return forest, pred, err
end

# ---------------------------------------------------------------------- tuning

const _ML_GRF_TUNABLE = (:sample_fraction, :mtry, :min_node_size, :honesty_fraction,
                         :honesty_prune_leaves, :alpha, :imbalance_penalty)

# grf's get_params_from_draw: map uniform draws to parameter values.
function _ml_grf_param_from_draw(name::Symbol, u::Float64, n::Int, p::Int)
    name === :min_node_size && return floor(Int, 2^(u * (log2(n) - 4)))
    name === :sample_fraction && return 0.05 + 0.45 * u
    name === :mtry && return max(1, ceil(Int, min(p, sqrt(p) + 20) * u))
    name === :alpha && return u / 4
    name === :imbalance_penalty && return -log(u)
    name === :honesty_fraction && return 0.5 + 0.3 * u
    name === :honesty_prune_leaves && return u < 0.5
    throw(ArgumentError("unknown tuning parameter $(name)"))
end

function _ml_grf_tune_list(tune, honesty, context)
    tune === :none && return Symbol[]
    names = tune === :all ? collect(_ML_GRF_TUNABLE) : Symbol.(collect(tune))
    for nm in names
        nm in _ML_GRF_TUNABLE ||
            throw(ArgumentError("$(context): cannot tune $(nm); tunable parameters are " *
                                join(_ML_GRF_TUNABLE, ", ")))
    end
    honesty || filter!(nm -> !occursin("honesty", string(nm)), names)
    return unique(names)
end

"""
Tuning by out-of-bag error (grf's `tune_forest` without the kriging smoother): small
forests at random parameter draws; the draw with the lowest mean debiased OOB error
is refitted with 4× the trees and kept if it beats the defaults refitted likewise.
"""
function _ml_grf_tune(kind, X, y, pr, names, clusters, spc, rng; w=Float64[],
                      z=Float64[], wt=nothing, tune_num_trees, tune_num_reps, parallel,
                      reduced_form_weight=0.0, context)
    n, p = size(X)
    small = merge(pr, (num_trees=tune_num_trees, ci_group_size=1))
    U = rand(rng, tune_num_reps, length(names))
    errs = fill(NaN, tune_num_reps)
    cand = Vector{NamedTuple}(undef, tune_num_reps)
    for r in 1:tune_num_reps
        vals = NamedTuple{Tuple(names)}(Tuple(_ml_grf_param_from_draw(nm, U[r, j], n, p)
                                              for (j, nm) in enumerate(names)))
        cand[r] = vals
        prr = merge(small, vals)
        e = try
            _, _, err = _ml_grf_train(kind, X, y, prr, clusters, spc, rng; w=w, z=z,
                                      wt=wt, parallel=parallel,
                                      reduced_form_weight=reduced_form_weight,
                                      context=context)
            ok = filter(isfinite, err)
            isempty(ok) ? NaN : mean(ok)
        catch e
            e isa ArgumentError || rethrow()
            NaN
        end
        errs[r] = e
    end
    keep = findall(isfinite, errs)
    defaults = NamedTuple{Tuple(names)}(Tuple(getfield(pr, nm) for nm in names))
    length(keep) < 2 && return (status="failure", params=defaults, error=NaN)
    best = keep[argmin(errs[keep])]
    big = merge(small, (num_trees=4 * tune_num_trees,))
    evalerr(vals) = begin
        _, _, err = _ml_grf_train(kind, X, y, merge(big, vals), clusters, spc, rng;
                                  w=w, z=z, wt=wt, parallel=parallel,
                                  reduced_form_weight=reduced_form_weight,
                                  context=context)
        ok = filter(isfinite, err)
        isempty(ok) ? NaN : mean(ok)
    end
    e_tuned = evalerr(cand[best])
    e_default = evalerr(defaults)
    if isnan(e_tuned) || e_default < e_tuned
        return (status="default", params=defaults, error=e_default)
    end
    return (status="tuned", params=cand[best], error=e_tuned)
end

# ------------------------------------------------------------------ nuisances

const _ML_GRF_LPM = Union{OLSLearner,RidgeLearner,LassoLearner}

"""
Local-centering nuisance `E[target | X]` in internal row order: a regression
forest (OOB), a constant, a vector (original row order), a column, or a
`NuisanceLearner` (cross-fitted). Returns (values, description).
"""
function _ml_grf_nuisance(spec, name, target, X, order, pr, clusters, spc, cid, wt, rng,
                          data; ntrees, n_folds, parallel, tune, tune_num_trees,
                          tune_num_reps, context)
    n = length(target)
    if spec === nothing
        prn = merge(pr, (num_trees=ntrees, ci_group_size=1, min_node_size=5,
                         honesty=true, honesty_fraction=0.5))
        tuned = nothing
        if !isempty(tune)
            tuned = _ml_grf_tune(:regression, X, target, prn, tune, clusters, spc, rng;
                                 wt=wt, tune_num_trees=tune_num_trees,
                                 tune_num_reps=tune_num_reps, parallel=parallel,
                                 context=context)
            prn = merge(prn, tuned.params)
        end
        _, pred, _ = _ml_grf_train(:regression, X, target, prn, clusters, spc, rng;
                                   wt=wt, parallel=parallel, error=false, context=context)
        any(isnan, pred) &&
            throw(ArgumentError("$(context): some out-of-bag predictions of $(name) are " *
                                "undefined; increase the number of trees"))
        return pred, "regression forest ($(ntrees) trees, out-of-bag)"
    elseif spec isa Real
        return fill(Float64(spec), n), "fixed at $(spec)"
    elseif spec isa Symbol
        data === nothing && throw(ArgumentError("$(context): $(name) given as a column " *
                                                "name requires the data-frame method"))
        require_columns(data, [spec]; context=context)
        return _ml_column(data, spec; context=context)[order], "column $(spec)"
    elseif spec isa AbstractVector
        v = _ml_grf_vector(spec, n, string(name), context)
        return v[order], "supplied vector"
    elseif spec isa NuisanceLearner
        any(isnan, X) && throw(ArgumentError("$(context): covariates have missing " *
                                             "values; $(name) must be a forest or given"))
        binary = all(v -> v == 0 || v == 1, target) && length(unique(target)) == 2
        proba = binary && !(spec isa _ML_GRF_LPM)
        F = crossfit_folds(n, n_folds, 1; rng=rng, strata=binary ? target : nothing,
                           groups=cid)
        seeds = _ml_seeds(rng, n_folds, 1, 1)
        P = _ml_crossfit([_MLNuisance(name, spec, target, X, proba)], F[:, 1],
                         view(seeds, :, :, 1); parallel=parallel, context=context)
        return P[:, 1], "$(_ml_learner_name(spec)) ($(n_folds)-fold cross-fitted)"
    end
    throw(ArgumentError("$(context): $(name) must be nothing, a number, a vector, a " *
                        "column name or a NuisanceLearner"))
end

# ------------------------------------------------------------------ front ends

# Shared extraction for the data-frame methods.
function _ml_grf_extract(data, cols, covariates, weights, cluster, context)
    covs = Symbol.(collect(covariates))
    isempty(covs) && throw(ArgumentError("$(context): covariates must not be empty"))
    wcol = weights === nothing ? Symbol[] : [Symbol(weights)]
    ccol = cluster === nothing ? Symbol[] : [Symbol(cluster)]
    require_columns(data, vcat(cols, covs, wcol, ccol); context=context)
    X = _ml_grf_matrix(data, covs; context=context)
    vals = [_ml_column(data, c; context=context) for c in cols]
    wt = weights === nothing ? nothing : _ml_column(data, Symbol(weights); context=context)
    cl = cluster === nothing ? nothing : data[!, Symbol(cluster)]
    return covs, X, vals, wt, cl
end

function _ml_grf_common_setup(X, vecs, wt, cl, equalize, context)
    n = size(X, 1)
    n >= 2 || throw(ArgumentError("$(context): need at least two observations"))
    if wt !== nothing
        all(>=(0), wt) || throw(ArgumentError("$(context): weights must be non-negative"))
        sum(wt) > 0 || throw(ArgumentError("$(context): weights sum to zero"))
    end
    cid, G = if cl === nothing
        nothing, 0
    else
        length(cl) == n || throw(DimensionMismatch("$(context): cluster must have " *
                                                   "length $n"))
        any(ismissing, cl) && throw(ArgumentError("$(context): cluster has missing values"))
        _ml_group_index(cl)
    end
    keycols = AbstractVector[]
    cid === nothing || push!(keycols, cid)
    append!(keycols, [view(X, :, j) for j in axes(X, 2)])
    append!(keycols, vecs)
    wt === nothing || push!(keycols, wt)
    order = _ml_grf_canonical_order(keycols)
    cidi = cid === nothing ? nothing : cid[order]
    clusters, spc = _ml_grf_clusters(cidi, equalize, wt !== nothing, context)
    return cid, G, order, cidi, clusters, spc
end

_ml_grf_unorder(v, order) = (out = similar(v); out[order] = v; out)

"""
    regression_forest(data, outcome; covariates, num_trees=2000, weights=nothing,
                      cluster=nothing, equalize_cluster_weights=false,
                      sample_fraction=0.5, mtry=nothing, min_node_size=5,
                      honesty=true, honesty_fraction=0.5, honesty_prune_leaves=true,
                      alpha=0.05, imbalance_penalty=0.0, ci_group_size=2,
                      tune_parameters=:none, tune_num_trees=50, tune_num_reps=100,
                      rng=Random.default_rng(), parallel=true) -> RegressionForest
    regression_forest(X::AbstractMatrix, Y::AbstractVector; weights=nothing,
                      cluster=nothing, kwargs...) -> RegressionForest

Fit an honest regression forest for the conditional mean ``\\mu(x) = E[Y \\mid X = x]``
(grf's `regression_forest`).

The forest is the regression special case of a generalized random forest (Athey, Tibshirani
& Wager 2019), itself a variant of Breiman's (2001) random forest with the honesty and
subsampling devices of Wager and Athey (2018). Each tree is grown on a subsample of size
`sample_fraction · n` drawn without replacement (by cluster when `cluster` is given). With
honesty, the subsample is split into one part that chooses the splits and another
(`honesty_fraction` determines the split) that estimates the leaf means, and leaves left
empty in the estimation part are pruned (`honesty_prune_leaves`). Splits maximize the
between-child variance of `Y` subject to each child holding at least a share `alpha` of the
parent and to an `imbalance_penalty`; at each split a Poisson number of candidate variables
with mean `mtry` is drawn, as in grf. The prediction at `x` is the forest-weighted mean
``\\hat\\mu(x) = \\sum_i \\alpha_i(x) Y_i``.

Honesty and subsampling make ``\\hat\\mu(x)`` asymptotically normal and centred on
``\\mu(x)`` under smoothness conditions (Wager & Athey 2018). Trees are grown in groups of
`ci_group_size` that share a half-sample, which yields the bootstrap-of-little-bags
variance estimates reported by [`predict_interval`](@ref). In causal work, regression
forests are mainly used as nuisance estimators (propensity scores, outcome regressions; see
also [`ForestLearner`](@ref)) and for calibration checks with [`test_calibration`](@ref).
Results are invariant to the row order of the data and, because one seed per group of trees
is drawn up front from `rng`, to the number of threads.

# Arguments
- `data`: a `DataFrame` (or other Tables.jl source) holding the outcome and the covariates.
  The matrix method takes the covariates `X` (`n × p`) and the outcome `Y` directly and
  names the covariates `:x1, :x2, …`.
- `outcome::Symbol`: numeric outcome column, which must be complete.

# Keywords
- `covariates::Vector{Symbol}`: numeric covariates (required, non-empty); `missing` values
  are allowed and handled by the splitting rule. Encode categories as dummies.
- `num_trees::Integer = 2000`: number of trees; more trees reduce the Monte Carlo error of
  predictions and variance estimates.
- `weights = nothing`: column of non-negative sample weights (a vector for the matrix
  method), used in the leaf means and the post-estimation averages.
- `cluster = nothing`: cluster column; subsamples are then drawn by cluster, so out-of-bag
  predictions and variance estimates respect the dependence.
- `equalize_cluster_weights::Bool = false`: draw the same number of units from every
  cluster, so each cluster receives equal weight; incompatible with `weights`.
- `sample_fraction::Real = 0.5`: subsample share per tree; must be at most 0.5 when
  `ci_group_size > 1`.
- `mtry = nothing`: mean number of candidate split variables; the default is
  ``\\min(\\lceil\\sqrt{p} + 20\\rceil, p)``.
- `min_node_size::Integer = 5`: target minimum number of observations per leaf.
- `honesty::Bool = true`, `honesty_fraction::Real = 0.5`,
  `honesty_prune_leaves::Bool = true`: honest splitting and its options; disabling honesty
  invalidates the confidence intervals.
- `alpha::Real = 0.05`: minimum share of the parent in each child (in `[0, 0.25)`).
- `imbalance_penalty::Real = 0.0`: penalty on unbalanced splits.
- `ci_group_size::Integer = 2`: trees per little bag; `1` disables variance estimation.
- `tune_parameters = :none`: `:all` or a vector of names among `:sample_fraction`, `:mtry`,
  `:min_node_size`, `:honesty_fraction`, `:honesty_prune_leaves`, `:alpha`,
  `:imbalance_penalty`, tuned by out-of-bag error as described under
  [`causal_forest`](@ref); `tune_num_trees::Integer = 50` and
  `tune_num_reps::Integer = 100` set the size and number of the trial forests.
- `rng::AbstractRNG = Random.default_rng()`: random-number generator.
- `parallel::Bool = true`: grow trees on threads (results do not depend on it).

# Returns
- [`RegressionForest`](@ref), whose out-of-bag predictions are `predict(rf)`.

# Examples
```julia
using DrSnow, DataFrames, StableRNGs
rng = StableRNG(1)
n = 400
df = DataFrame(x1=rand(rng, n), x2=rand(rng, n))
df.y = sin.(3 .* df.x1) .+ df.x2 .+ 0.5 .* randn(rng, n)
rf = regression_forest(df, :y; covariates=[:x1, :x2], num_trees=200,
                       rng=StableRNG(2))
predict(rf)[1:3]                                   # out-of-bag predictions
predict_interval(rf, DataFrame(x1=[0.2, 0.8], x2=[0.5, 0.5]))
```

# References
- Athey, S., Tibshirani, J., & Wager, S. (2019). Generalized random forests. *The Annals of
  Statistics*, 47(2), 1148–1178.
- Breiman, L. (2001). Random forests. *Machine Learning*, 45(1), 5–32.
- Wager, S., & Athey, S. (2018). Estimation and inference of heterogeneous treatment
  effects using random forests. *Journal of the American Statistical Association*,
  113(523), 1228–1242.
"""
function regression_forest(data, outcome::Symbol; covariates=Symbol[], weights=nothing,
                           cluster=nothing, kwargs...)
    ctx = "regression_forest"
    covs, X, (y,), wt, cl = _ml_grf_extract(data, [outcome], covariates, weights, cluster,
                                            ctx)
    return _ml_regression_forest(X, y, wt, cl, covs, outcome; kwargs...)
end

function regression_forest(X::AbstractMatrix, Y::AbstractVector; weights=nothing,
                           cluster=nothing, kwargs...)
    ctx = "regression_forest"
    Xm = _ml_grf_check_X(X, ctx)
    n = size(Xm, 1)
    y = _ml_grf_vector(Y, n, "Y", ctx)
    wt = _ml_grf_vector(weights, n, "weights", ctx)
    covs = [Symbol("x", j) for j in 1:size(Xm, 2)]
    return _ml_regression_forest(Xm, y, wt, cluster, covs, :y; kwargs...)
end

function _ml_regression_forest(X, y, wt, cl, covs, outcome; num_trees::Integer=2000,
                               equalize_cluster_weights::Bool=false,
                               sample_fraction::Real=0.5, mtry=nothing,
                               min_node_size::Integer=5, honesty::Bool=true,
                               honesty_fraction::Real=0.5,
                               honesty_prune_leaves::Bool=true, alpha::Real=0.05,
                               imbalance_penalty::Real=0.0, ci_group_size::Integer=2,
                               tune_parameters=:none, tune_num_trees::Integer=50,
                               tune_num_reps::Integer=100,
                               rng::AbstractRNG=Random.default_rng(),
                               parallel::Bool=true)
    ctx = "regression_forest"
    cid, G, order, cidi, clusters, spc = _ml_grf_common_setup(X, [y], wt, cl,
                                                              equalize_cluster_weights, ctx)
    pr = _ml_grf_params(size(X, 2); num_trees, sample_fraction, mtry, min_node_size,
                        honesty, honesty_fraction, honesty_prune_leaves, alpha,
                        imbalance_penalty, ci_group_size, context=ctx)
    Xi, yi = X[order, :], y[order]
    wti = wt === nothing ? nothing : wt[order]
    seed = task_seeds(rng, 1)[1]
    trng = Random.Xoshiro(seed)
    tune = _ml_grf_tune_list(tune_parameters, honesty, ctx)
    tuning = nothing
    if !isempty(tune)
        tuning = _ml_grf_tune(:regression, Xi, yi, pr, tune, clusters, spc, trng; wt=wti,
                              tune_num_trees=tune_num_trees, tune_num_reps=tune_num_reps,
                              parallel=parallel, context=ctx)
        pr = merge(pr, tuning.params)
    end
    forest, pred, err = _ml_grf_train(:regression, Xi, yi, pr, clusters, spc, trng;
                                      wt=wti, parallel=parallel, context=ctx)
    return RegressionForest(forest, order, covs, outcome, X, y, wt, cid, G,
                            equalize_cluster_weights, _ml_grf_unorder(pred, order),
                            _ml_grf_unorder(err, order), pr, tuning, seed)
end

"""
    causal_forest(data, outcome, treatment; covariates, y_hat=nothing, w_hat=nothing,
                  num_trees=2000, weights=nothing, cluster=nothing,
                  equalize_cluster_weights=false, sample_fraction=0.5, mtry=nothing,
                  min_node_size=5, honesty=true, honesty_fraction=0.5,
                  honesty_prune_leaves=true, alpha=0.05, imbalance_penalty=0.0,
                  stabilize_splits=true, ci_group_size=2, tune_parameters=:none,
                  tune_num_trees=200, tune_num_reps=50, n_folds=5,
                  rng=Random.default_rng(), parallel=true) -> CausalForest
    causal_forest(X::AbstractMatrix, Y::AbstractVector, W::AbstractVector;
                  weights=nothing, cluster=nothing, kwargs...) -> CausalForest

Estimate conditional average treatment effects with a causal forest (grf's `causal_forest`;
Wager & Athey 2018; Athey, Tibshirani & Wager 2019).

The estimand is the conditional average treatment effect
```math
\\tau(x) = E[Y_i(1) - Y_i(0) \\mid X_i = x],
```
or, for a continuous treatment, the conditional average partial effect under a locally
linear model ``E[Y \\mid X = x, W] = \\mu(x) + \\tau(x) W``. Identification requires
unconfoundedness, ``\\{Y_i(0), Y_i(1)\\} \\perp W_i \\mid X_i``, and overlap,
``0 < e(x) = P(W_i = 1 \\mid X_i = x) < 1``. Neither is testable: the data can reveal
limited overlap through estimated propensities near 0 or 1, but not unmeasured confounding.
The partially linear representation behind the forest is that of Robinson (1988); the
forest solves, locally in ``x``, the residual-on-residual moment
```math
E\\big[(W_i - e(X_i))\\{Y_i - m(X_i) - \\tau(x)(W_i - e(X_i))\\} \\mid X_i = x\\big] = 0,
\\qquad m(x) = E[Y_i \\mid X_i = x].
```

The algorithm is as follows. (1) **Local centering**: ``\\hat Y = \\hat E[Y \\mid X]`` and
``\\hat W = \\hat E[W \\mid X]`` are estimated out of bag by regression forests with
`max(50, num_trees ÷ 4)` trees unless supplied through `y_hat` / `w_hat`; centering removes
the confounding through ``m`` and ``e`` from the splitting and makes the estimator
insensitive to first-order errors in the nuisances (the R-learner orthogonality of Nie &
Wager 2021). (2) **Gradient-based splitting**: in each node the moment is solved for a
node-level ``\\hat\\tau_P`` and the node is split to maximize the heterogeneity of the
pseudo-outcomes
```math
\\rho_i = (\\tilde W_i - \\bar W)
  \\{\\tilde Y_i - \\bar Y - \\hat\\tau_P(\\tilde W_i - \\bar W)\\},
```
with ``\\tilde Y = Y - \\hat Y`` and ``\\tilde W = W - \\hat W`` (bars denote node means);
with `stabilize_splits = true` each child must contain at least `min_node_size` units on
each side of the node mean of ``\\tilde W`` (treated and controls for a binary treatment).
(3) **Honesty, subsampling and little bags** as in [`regression_forest`](@ref). (4)
**Estimation**: ``\\hat\\tau(x)`` solves the moment with forest weights ``\\alpha_i(x)``,
```math
\\hat\\tau(x) = \\frac{\\sum_i \\alpha_i(x)(\\tilde W_i - \\bar W_\\alpha)
                (\\tilde Y_i - \\bar Y_\\alpha)}
            {\\sum_i \\alpha_i(x)(\\tilde W_i - \\bar W_\\alpha)^2}.
```

Pointwise confidence intervals for ``\\tau(x)`` ([`predict_interval`](@ref)) rely on the
asymptotic normality results of Wager and Athey (2018) and Athey, Tibshirani and Wager
(2019), which require honesty, subsampling rates with `sample_fraction` shrinking slowly,
overlap and smoothness of ``\\tau``; in finite samples they can undercover where ``\\tau``
changes sharply. Individual CATE estimates are noisy, so summaries with valid inference are
usually more informative: the AIPW average effect ([`average_treatment_effect`](@ref)), the
best linear projection ([`best_linear_projection`](@ref)), the calibration test
([`test_calibration`](@ref)) and the rank-weighted average treatment effect
([`rank_average_treatment_effect`](@ref)), all built on the doubly robust scores of
[`get_scores`](@ref) (Athey & Wager 2019). For CATEs with a parametric summary and
cross-fitted inference see also [`cate_dr_learner`](@ref) and [`generic_ml`](@ref); for
learner-agnostic CATE estimation see [`r_learner`](@ref).

Tuning (`tune_parameters = :all` or a vector of names among `:sample_fraction`, `:mtry`,
`:min_node_size`, `:honesty_fraction`, `:honesty_prune_leaves`, `:alpha`,
`:imbalance_penalty`) fits `tune_num_reps` small forests of `tune_num_trees` trees at
random parameter draws, and keeps the draw with the lowest mean debiased out-of-bag R-loss
if, refitted with four times the trees, it beats the defaults. grf additionally smooths the
error surface by kriging; that step is omitted, so tuned parameters can differ from grf's.
Rows are put in a canonical order before sampling, so results are invariant to the row
order of the data.

# Arguments
- `data`: a `DataFrame` (or Tables.jl source). The matrix method takes the covariates `X`,
  outcome `Y` and treatment `W` as arrays.
- `outcome::Symbol`: numeric outcome column (complete).
- `treatment::Symbol`: treatment column, binary 0/1 or continuous; it must vary.

# Keywords
- `covariates::Vector{Symbol}`: numeric covariates (required). `missing` values are allowed
  when `y_hat` / `w_hat` are forests or supplied.
- `y_hat = nothing`, `w_hat = nothing`: centering nuisances. `nothing` fits out-of-bag
  regression forests; alternatively a number, a vector, a column name, or a
  [`NuisanceLearner`](@ref), which is then cross-fitted in `n_folds` folds
  (`n_folds::Integer = 5`). Supplying a known randomization propensity through `w_hat` is
  appropriate in experiments.
- `num_trees::Integer = 2000`, `sample_fraction`, `mtry`, `min_node_size`, `honesty`,
  `honesty_fraction`, `honesty_prune_leaves`, `alpha`, `imbalance_penalty`,
  `ci_group_size`: forest parameters, with the defaults and meaning documented in
  [`regression_forest`](@ref).
- `stabilize_splits::Bool = true`: impose the treatment-balance constraint on splits
  described above.
- `weights = nothing`, `cluster = nothing`, `equalize_cluster_weights = false`: sample
  weights and clusters, used for subsampling, the nuisance forests and all post-estimation
  inference (cluster-robust with `G - 1` degrees of freedom).
- `tune_parameters = :none`, `tune_num_trees::Integer = 200`,
  `tune_num_reps::Integer = 50`: parameter tuning, as described above.
- `rng::AbstractRNG = Random.default_rng()`, `parallel::Bool = true`: one seed per group of
  trees is drawn up front, so results are reproducible and independent of the number of
  threads.

# Returns
- [`CausalForest`](@ref); `predict(cf)` gives the out-of-bag CATEs.

# Examples
```julia
using DrSnow, DataFrames, StableRNGs
rng = StableRNG(1)
n = 500
df = DataFrame(x1=rand(rng, n), x2=rand(rng, n), x3=rand(rng, n))
df.d = Int.(rand(rng, n) .< 0.3 .+ 0.4 .* df.x2)            # confounded by x2
df.y = df.x2 .+ (1 .+ 2 .* df.x1) .* df.d .+ randn(rng, n)  # τ(x) = 1 + 2 x1
cf = causal_forest(df, :y, :d; covariates=[:x1, :x2, :x3], num_trees=200,
                   rng=StableRNG(2))
predict(cf)[1:3]                              # out-of-bag CATEs
average_treatment_effect(cf)                  # AIPW ATE (truth 2)
average_treatment_effect(cf; target=:treated) # ATT
test_calibration(cf)
```

# References
- Athey, S., Tibshirani, J., & Wager, S. (2019). Generalized random forests. *The Annals of
  Statistics*, 47(2), 1148–1178.
- Athey, S., & Wager, S. (2019). Estimating treatment effects with causal forests: An
  application. *Observational Studies*, 5(2), 37–51.
- Athey, S., & Imbens, G. (2016). Recursive partitioning for heterogeneous causal effects.
  *Proceedings of the National Academy of Sciences*, 113(27), 7353–7360.
- Nie, X., & Wager, S. (2021). Quasi-oracle estimation of heterogeneous treatment effects.
  *Biometrika*, 108(2), 299–319.
- Robinson, P. M. (1988). Root-N-consistent semiparametric regression. *Econometrica*,
  56(4), 931–954.
- Wager, S., & Athey, S. (2018). Estimation and inference of heterogeneous treatment
  effects using random forests. *Journal of the American Statistical Association*,
  113(523), 1228–1242.
"""
function causal_forest(data, outcome::Symbol, treatment::Symbol; covariates=Symbol[],
                       weights=nothing, cluster=nothing, kwargs...)
    ctx = "causal_forest"
    covs, X, (y, w), wt, cl = _ml_grf_extract(data, [outcome, treatment], covariates,
                                              weights, cluster, ctx)
    return _ml_causal_forest(X, y, w, wt, cl, covs, outcome, treatment, data; kwargs...)
end

function causal_forest(X::AbstractMatrix, Y::AbstractVector, W::AbstractVector;
                       weights=nothing, cluster=nothing, kwargs...)
    ctx = "causal_forest"
    Xm = _ml_grf_check_X(X, ctx)
    n = size(Xm, 1)
    y = _ml_grf_vector(Y, n, "Y", ctx)
    w = _ml_grf_vector(W, n, "W", ctx)
    wt = _ml_grf_vector(weights, n, "weights", ctx)
    covs = [Symbol("x", j) for j in 1:size(Xm, 2)]
    return _ml_causal_forest(Xm, y, w, wt, cluster, covs, :y, :w, nothing; kwargs...)
end

function _ml_causal_forest(X, y, w, wt, cl, covs, outcome, treatment, data;
                           y_hat=nothing, w_hat=nothing, num_trees::Integer=2000,
                           equalize_cluster_weights::Bool=false,
                           sample_fraction::Real=0.5, mtry=nothing,
                           min_node_size::Integer=5, honesty::Bool=true,
                           honesty_fraction::Real=0.5, honesty_prune_leaves::Bool=true,
                           alpha::Real=0.05, imbalance_penalty::Real=0.0,
                           stabilize_splits::Bool=true, ci_group_size::Integer=2,
                           tune_parameters=:none, tune_num_trees::Integer=200,
                           tune_num_reps::Integer=50, n_folds::Integer=5,
                           rng::AbstractRNG=Random.default_rng(), parallel::Bool=true)
    ctx = "causal_forest"
    length(unique(w)) >= 2 || throw(ArgumentError("$(ctx): the treatment is constant"))
    cid, G, order, cidi, clusters, spc = _ml_grf_common_setup(X, [y, w], wt, cl,
                                                              equalize_cluster_weights, ctx)
    pr = _ml_grf_params(size(X, 2); num_trees, sample_fraction, mtry, min_node_size,
                        honesty, honesty_fraction, honesty_prune_leaves, alpha,
                        imbalance_penalty, ci_group_size, stabilize_splits, context=ctx)
    Xi, yi, wi = X[order, :], y[order], w[order]
    wti = wt === nothing ? nothing : wt[order]
    seed = task_seeds(rng, 1)[1]
    trng = Random.Xoshiro(seed)
    tune = _ml_grf_tune_list(tune_parameters, honesty, ctx)
    nk = (ntrees=max(50, pr.num_trees ÷ 4), n_folds=n_folds, parallel=parallel, tune=tune,
          tune_num_trees=tune_num_trees, tune_num_reps=tune_num_reps, context=ctx)
    yh, ydesc = _ml_grf_nuisance(y_hat, :Y_hat, yi, Xi, order, pr, clusters, spc, cidi,
                                 wti, trng, data; nk...)
    wh, wdesc = _ml_grf_nuisance(w_hat, :W_hat, wi, Xi, order, pr, clusters, spc, cidi,
                                 wti, trng, data; nk...)
    yc = yi .- yh
    wc = wi .- wh
    tuning = nothing
    if !isempty(tune)
        tuning = _ml_grf_tune(:instrumental, Xi, yc, pr, tune, clusters, spc, trng; w=wc,
                              z=wc, wt=wti, tune_num_trees=tune_num_trees,
                              tune_num_reps=tune_num_reps, parallel=parallel, context=ctx)
        pr = merge(pr, tuning.params)
    end
    forest, pred, err = _ml_grf_train(:instrumental, Xi, yc, pr, clusters, spc, trng;
                                      w=wc, z=wc, wt=wti, parallel=parallel, context=ctx)
    un(v) = _ml_grf_unorder(v, order)
    return CausalForest(forest, order, covs, outcome, treatment, X, y, w, un(yh), un(wh),
                        wt, cid, G, equalize_cluster_weights, un(pred), un(err), pr,
                        tuning, seed, [:Y_hat => ydesc, :W_hat => wdesc])
end

"""
    instrumental_forest(data, outcome, treatment, instrument; covariates,
                        y_hat=nothing, w_hat=nothing, z_hat=nothing,
                        reduced_form_weight=0.0, kwargs...) -> InstrumentalForest
    instrumental_forest(X::AbstractMatrix, Y, W, Z; weights=nothing, cluster=nothing,
                        kwargs...) -> InstrumentalForest

Estimate conditional local average treatment effects with an instrumental forest (grf's
`instrumental_forest`; Athey, Tibshirani & Wager 2019, Section 7).

With a binary instrument ``Z``, binary treatment ``W`` and potential treatments ``W_i(z)``,
the estimand is the conditional LATE, the average effect for units at covariate value ``x``
whose treatment is moved by the instrument,
```math
\\tau(x) = E[Y_i(1) - Y_i(0) \\mid W_i(1) > W_i(0), X_i = x]
     = \\frac{\\mathrm{Cov}(Y_i, Z_i \\mid X_i = x)}
            {\\mathrm{Cov}(W_i, Z_i \\mid X_i = x)}.
```
Identification requires, conditionally on ``X``, instrument independence and the exclusion
restriction, monotonicity (no defiers) and a non-zero first stage at ``x``. Exclusion and
monotonicity cannot be verified from the data; the strength of the conditional first stage
can be inspected (for instance with a causal forest of ``W`` on ``Z``). With a continuous
treatment or instrument ``\\tau(x)`` is the conditional IV slope of the local linear
structural model, and its causal interpretation rests on stronger homogeneity conditions.

The forest solves, with forest weights ``\\alpha_i(x)``, the locally centred IV moment
```math
\\sum_i \\alpha_i(x)(\\tilde Z_i - \\bar Z_\\alpha)
    \\{\\tilde Y_i - \\bar Y_\\alpha - \\tau(x)(\\tilde W_i - \\bar W_\\alpha)\\} = 0,
```
where ``\\tilde Y = Y - \\hat Y``, ``\\tilde W = W - \\hat W`` and
``\\tilde Z = Z - \\hat Z`` are centred by out-of-bag regression forests with
`min(500, num_trees)` trees unless supplied. Splits use the pseudo-outcomes (gradients) of
this moment, as in [`causal_forest`](@ref); `reduced_form_weight ∈ [0, 1]` shifts the
splitting criterion towards the reduced form, which regularizes splits when the conditional
first stage is weak. Honesty, subsampling and the little-bags variance estimates of
[`predict_interval`](@ref) work as for the other forests.

Pointwise intervals are unreliable where the conditional first stage is weak, because the
ratio estimator is then badly approximated by its normal limit. For the average conditional
LATE with a binary instrument use [`average_treatment_effect`](@ref), which averages doubly
robust scores in the spirit of Frölich (2007); for a global parametric LATE see
[`dml_iivm`](@ref).

# Arguments
- `data`: a `DataFrame` (or Tables.jl source); the matrix method takes `X`, `Y`, `W` and
  `Z` as arrays.
- `outcome::Symbol`: numeric outcome.
- `treatment::Symbol`: treatment (binary or continuous).
- `instrument::Symbol`: instrument (binary or continuous); it must vary.

# Keywords
- `covariates::Vector{Symbol}`: numeric covariates (required).
- `y_hat = nothing`, `w_hat = nothing`, `z_hat = nothing`: centering nuisances, in any of
  the forms accepted by [`causal_forest`](@ref).
- `reduced_form_weight::Real = 0.0`: weight on the reduced form in the splitting rule, in
  `[0, 1]`.
- All other keywords (`num_trees = 2000`, forest parameters, `weights`, `cluster`,
  `equalize_cluster_weights`, `stabilize_splits`, `ci_group_size`, tuning options,
  `n_folds`, `rng`, `parallel`) as in [`causal_forest`](@ref).

# Returns
- [`InstrumentalForest`](@ref); `predict(ivf)` gives out-of-bag conditional LATEs.

# Examples
```julia
using DrSnow, DataFrames, StableRNGs
rng = StableRNG(1)
n = 500
df = DataFrame(x1=rand(rng, n), x2=rand(rng, n))
df.z = Int.(rand(rng, n) .< 0.5)
u = randn(rng, n)
df.d = Int.(0.8 .* df.z .+ 0.3 .* u .+ 0.3 .* rand(rng, n) .> 0.5)
df.y = (1 .+ df.x1) .* df.d .+ u .+ randn(rng, n)
ivf = instrumental_forest(df, :y, :d, :z; covariates=[:x1, :x2], num_trees=200,
                          rng=StableRNG(2))
average_treatment_effect(ivf)    # average conditional LATE
```

# References
- Athey, S., Tibshirani, J., & Wager, S. (2019). Generalized random forests. *The Annals of
  Statistics*, 47(2), 1148–1178.
- Abadie, A. (2003). Semiparametric instrumental variable estimation of treatment response
  models. *Journal of Econometrics*, 113(2), 231–263.
- Frölich, M. (2007). Nonparametric IV estimation of local average treatment effects with
  covariates. *Journal of Econometrics*, 139(1), 35–75.
"""
function instrumental_forest(data, outcome::Symbol, treatment::Symbol, instrument::Symbol;
                             covariates=Symbol[], weights=nothing, cluster=nothing,
                             kwargs...)
    ctx = "instrumental_forest"
    covs, X, (y, w, z), wt, cl = _ml_grf_extract(data, [outcome, treatment, instrument],
                                                 covariates, weights, cluster, ctx)
    return _ml_instrumental_forest(X, y, w, z, wt, cl, covs, outcome, treatment,
                                   instrument, data; kwargs...)
end

function instrumental_forest(X::AbstractMatrix, Y::AbstractVector, W::AbstractVector,
                             Z::AbstractVector; weights=nothing, cluster=nothing,
                             kwargs...)
    ctx = "instrumental_forest"
    Xm = _ml_grf_check_X(X, ctx)
    n = size(Xm, 1)
    y = _ml_grf_vector(Y, n, "Y", ctx)
    w = _ml_grf_vector(W, n, "W", ctx)
    z = _ml_grf_vector(Z, n, "Z", ctx)
    wt = _ml_grf_vector(weights, n, "weights", ctx)
    covs = [Symbol("x", j) for j in 1:size(Xm, 2)]
    return _ml_instrumental_forest(Xm, y, w, z, wt, cluster, covs, :y, :w, :z, nothing;
                                   kwargs...)
end

function _ml_instrumental_forest(X, y, w, z, wt, cl, covs, outcome, treatment, instrument,
                                 data; y_hat=nothing, w_hat=nothing, z_hat=nothing,
                                 num_trees::Integer=2000,
                                 equalize_cluster_weights::Bool=false,
                                 sample_fraction::Real=0.5, mtry=nothing,
                                 min_node_size::Integer=5, honesty::Bool=true,
                                 honesty_fraction::Real=0.5,
                                 honesty_prune_leaves::Bool=true, alpha::Real=0.05,
                                 imbalance_penalty::Real=0.0, stabilize_splits::Bool=true,
                                 ci_group_size::Integer=2, reduced_form_weight::Real=0.0,
                                 tune_parameters=:none, tune_num_trees::Integer=200,
                                 tune_num_reps::Integer=50, n_folds::Integer=5,
                                 rng::AbstractRNG=Random.default_rng(),
                                 parallel::Bool=true)
    ctx = "instrumental_forest"
    0 <= reduced_form_weight <= 1 ||
        throw(ArgumentError("$(ctx): reduced_form_weight must be in [0, 1]"))
    length(unique(z)) >= 2 || throw(ArgumentError("$(ctx): the instrument is constant"))
    cid, G, order, cidi, clusters, spc = _ml_grf_common_setup(X, [y, w, z], wt, cl,
                                                              equalize_cluster_weights, ctx)
    pr = _ml_grf_params(size(X, 2); num_trees, sample_fraction, mtry, min_node_size,
                        honesty, honesty_fraction, honesty_prune_leaves, alpha,
                        imbalance_penalty, ci_group_size, stabilize_splits, context=ctx)
    Xi, yi, wi, zi = X[order, :], y[order], w[order], z[order]
    wti = wt === nothing ? nothing : wt[order]
    seed = task_seeds(rng, 1)[1]
    trng = Random.Xoshiro(seed)
    tune = _ml_grf_tune_list(tune_parameters, honesty, ctx)
    nk = (ntrees=min(500, pr.num_trees), n_folds=n_folds, parallel=parallel, tune=tune,
          tune_num_trees=tune_num_trees, tune_num_reps=tune_num_reps, context=ctx)
    yh, ydesc = _ml_grf_nuisance(y_hat, :Y_hat, yi, Xi, order, pr, clusters, spc, cidi,
                                 wti, trng, data; nk...)
    wh, wdesc = _ml_grf_nuisance(w_hat, :W_hat, wi, Xi, order, pr, clusters, spc, cidi,
                                 wti, trng, data; nk...)
    zh, zdesc = _ml_grf_nuisance(z_hat, :Z_hat, zi, Xi, order, pr, clusters, spc, cidi,
                                 wti, trng, data; nk...)
    yc, wc, zc = yi .- yh, wi .- wh, zi .- zh
    tuning = nothing
    rfw = Float64(reduced_form_weight)
    if !isempty(tune)
        tuning = _ml_grf_tune(:instrumental, Xi, yc, pr, tune, clusters, spc, trng; w=wc,
                              z=zc, wt=wti, tune_num_trees=tune_num_trees,
                              tune_num_reps=tune_num_reps, parallel=parallel,
                              reduced_form_weight=rfw, context=ctx)
        pr = merge(pr, tuning.params)
    end
    forest, pred, err = _ml_grf_train(:instrumental, Xi, yc, pr, clusters, spc, trng;
                                      w=wc, z=zc, wt=wti, parallel=parallel,
                                      reduced_form_weight=rfw, context=ctx)
    un(v) = _ml_grf_unorder(v, order)
    return InstrumentalForest(forest, order, covs, outcome, treatment, instrument, X, y, w,
                              z, un(yh), un(wh), un(zh), wt, cid, G,
                              equalize_cluster_weights, un(pred), un(err), pr, tuning,
                              seed, [:Y_hat => ydesc, :W_hat => wdesc, :Z_hat => zdesc])
end

# ------------------------------------------------------------------ prediction

function _ml_grf_newdata(f::GeneralizedRandomForest, newdata, context)
    if newdata isa AbstractDataFrame
        require_columns(newdata, f.covariates; context=context)
        return _ml_grf_matrix(newdata, f.covariates; context=context)
    elseif newdata isa AbstractMatrix
        size(newdata, 2) == length(f.covariates) ||
            throw(DimensionMismatch("$(context): expected $(length(f.covariates)) " *
                                    "covariate columns"))
        return _ml_grf_check_X(newdata, context)
    end
    throw(ArgumentError("$(context): newdata must be a DataFrame or a matrix"))
end

# (predictions, variances) at newdata, or out of bag for the training rows.
function _ml_grf_predict(f::GeneralizedRandomForest, newdata; estimate_variance::Bool,
                         parallel::Bool=true, context)
    if newdata === nothing
        Xi = f.X[f.order, :]
        L = _grf_leaf_matrix(f.forest, Xi, true, parallel)
        pred, vars, _ = _grf_collect(f.forest, L; estimate_variance=estimate_variance,
                                     parallel=parallel)
        return _ml_grf_unorder(pred, f.order),
               estimate_variance ? _ml_grf_unorder(vars, f.order) : vars
    end
    Xn = _ml_grf_newdata(f, newdata, context)
    L = _grf_leaf_matrix(f.forest, Xn, false, parallel)
    pred, vars, _ = _grf_collect(f.forest, L; estimate_variance=estimate_variance,
                                 parallel=parallel)
    return pred, vars
end

"""
    predict(f::GeneralizedRandomForest) -> Vector{Float64}
    predict(f::GeneralizedRandomForest, newdata) -> Vector{Float64}

Point predictions of a fitted forest, out of bag for the training rows or at new covariate
values.

Without `newdata`, each training row is predicted only by trees whose subsample did not
contain it (out-of-bag prediction), which avoids the overfitting of in-sample forest
predictions and is what the post-estimation tools use. With `newdata` the full forest is
used. The quantity predicted depends on the forest: estimates of ``E[Y \\mid X = x]`` for a
[`RegressionForest`](@ref), CATE estimates ``\\hat\\tau(x)`` for a [`CausalForest`](@ref),
and conditional LATE estimates for an [`InstrumentalForest`](@ref). `NaN` marks a point
that no tree could predict, which can happen out of bag with honesty and very few trees.
Standard errors are available from [`predict_interval`](@ref).

# Arguments
- `f::GeneralizedRandomForest`: a fitted forest.
- `newdata`: a `DataFrame` containing the forest's covariate columns, or a matrix with the
  same columns in the same order; omit it for out-of-bag predictions.

# Returns
- `Vector{Float64}`: one prediction per training row or per row of `newdata`.

# Examples
```julia
using DrSnow, DataFrames, StableRNGs
rng = StableRNG(1)
df = DataFrame(x1=rand(rng, 300), x2=rand(rng, 300))
df.d = Int.(rand(rng, 300) .< 0.5)
df.y = df.x1 .* df.d .+ randn(rng, 300)
cf = causal_forest(df, :y, :d; covariates=[:x1, :x2], num_trees=200,
                   rng=StableRNG(2))
predict(cf)[1:3]                                        # out of bag
predict(cf, DataFrame(x1=[0.2, 0.8], x2=[0.5, 0.5]))    # new points
```
"""
StatsAPI.predict(f::GeneralizedRandomForest) = copy(f.predictions)

function StatsAPI.predict(f::GeneralizedRandomForest, newdata::AbstractDataFrame)
    return _ml_grf_predict(f, newdata; estimate_variance=false,
                           context="predict($(nameof(typeof(f))))")[1]
end

function StatsAPI.predict(f::GeneralizedRandomForest, newdata::AbstractMatrix)
    return _ml_grf_predict(f, newdata; estimate_variance=false,
                           context="predict($(nameof(typeof(f))))")[1]
end

"""
    predict_interval(f::GeneralizedRandomForest, newdata=nothing; level=0.95,
                     parallel=true) -> DataFrame

Forest predictions with bootstrap-of-little-bags standard errors and pointwise normal
confidence intervals (grf's `predict(..., estimate.variance = TRUE)`).

The variance of a forest prediction is estimated by the bootstrap of little bags (Athey,
Tibshirani & Wager 2019, Section 4; building on Sexton & Laake 2009): the trees are grown
in groups of `ci_group_size` that share a half-sample, and the variance of the
forest-weighted estimating equation is estimated from the between-group variability of the
group-level estimates, debiased for the within-group Monte Carlo noise. grf's
objective-Bayes correction keeps the estimate non-negative. For a causal or instrumental
forest the variance of the estimating equation is converted to a variance of
``\\hat\\tau(x)`` by linearization. The interval is
``\\hat\\theta(x) \\pm z_{1-(1-\\text{level})/2} \\, \\widehat{\\mathrm{se}}(x)``.

The intervals are pointwise, not simultaneous over ``x``, and are asymptotically valid for
the forest's target (for example the CATE) under the conditions of Wager and Athey (2018):
honesty, subsampling, overlap and smoothness. They may undercover where the target function
is not smooth relative to the sample size, and the variance estimates are noisy with few
trees, so use at least several thousand trees for reported intervals.

# Arguments
- `f::GeneralizedRandomForest`: a forest fitted with `ci_group_size ≥ 2` (the default).
- `newdata`: a `DataFrame` or matrix of prediction points; `nothing` (default) gives
  out-of-bag results for the training rows.

# Keywords
- `level::Real = 0.95`: confidence level of the pointwise intervals.
- `parallel::Bool = true`: compute on threads (results do not depend on it).

# Returns
- `DataFrame` with columns `estimate`, `variance`, `std_error`, `conf_low` and `conf_high`,
  one row per prediction point.

# Examples
```julia
using DrSnow, DataFrames, StableRNGs
rng = StableRNG(1)
df = DataFrame(x1=rand(rng, 400), x2=rand(rng, 400))
df.y = 2 .* df.x1 .+ randn(rng, 400)
rf = regression_forest(df, :y; covariates=[:x1, :x2], num_trees=200,
                       rng=StableRNG(2))
predict_interval(rf, DataFrame(x1=0:0.25:1, x2=fill(0.5, 5)); level=0.9)
```

# References
- Athey, S., Tibshirani, J., & Wager, S. (2019). Generalized random forests. *The Annals of
  Statistics*, 47(2), 1148–1178.
- Sexton, J., & Laake, P. (2009). Standard errors for bagged and random forest estimators.
  *Computational Statistics & Data Analysis*, 53(3), 801–811.
- Wager, S., & Athey, S. (2018). Estimation and inference of heterogeneous treatment
  effects using random forests. *Journal of the American Statistical Association*,
  113(523), 1228–1242.
"""
function predict_interval(f::GeneralizedRandomForest, newdata=nothing; level::Real=0.95,
                          parallel::Bool=true)
    ctx = "predict_interval"
    f.params.ci_group_size >= 2 ||
        throw(ArgumentError("$(ctx): variance estimates require a forest fitted with " *
                            "ci_group_size ≥ 2"))
    c = critical_value(level)
    pred, vars = _ml_grf_predict(f, newdata; estimate_variance=true, parallel=parallel,
                                 context=ctx)
    se = sqrt.(max.(vars, 0.0))
    return DataFrame(estimate=pred, variance=vars, std_error=se, conf_low=pred .- c .* se,
                     conf_high=pred .+ c .* se)
end

"""Copy of a forest with some fields replaced (used to evaluate post-estimation
formulas on given nuisance estimates and predictions, e.g. in validation tests)."""
function _ml_grf_replace(f::T; kwargs...) where {T<:GeneralizedRandomForest}
    vals = [haskey(kwargs, k) ? kwargs[k] : getfield(f, k) for k in fieldnames(T)]
    return T(vals...)
end

# ------------------------------------------------------------ forest learner

"""
    ForestLearner(; num_trees=500, sample_fraction=0.5, mtry=nothing, min_node_size=5,
                  honesty=true, honesty_fraction=0.5, alpha=0.05,
                  imbalance_penalty=0.0)

An honest regression forest (the grf algorithm of [`regression_forest`](@ref)) as a
[`NuisanceLearner`](@ref): a nonparametric learner for conditional means and, for a
binary target, probabilities (forest predictions clipped to `[0, 1]`). Uses the task
RNG, supports observation weights, and runs its trees on threads.

# Examples
```julia
t = t_learner(df, :y, :d; covariates=[:x1, :x2], outcome_learner=ForestLearner())
dml_irm(df, :y, :d; covariates=[:x1, :x2], outcome_learner=ForestLearner(),
        propensity_learner=ForestLearner(num_trees=300))
```
"""
Base.@kwdef struct ForestLearner <: NuisanceLearner
    num_trees::Int = 500
    sample_fraction::Float64 = 0.5
    mtry::Union{Nothing,Int} = nothing
    min_node_size::Int = 5
    honesty::Bool = true
    honesty_fraction::Float64 = 0.5
    alpha::Float64 = 0.05
    imbalance_penalty::Float64 = 0.0
end

function fitpredict(l::ForestLearner, X::AbstractMatrix, y::AbstractVector,
                    Xnew::AbstractMatrix; rng::AbstractRNG=Random.default_rng(),
                    weights=nothing)
    _ml_check_xy(X, y, Xnew)
    n, p = size(X)
    wbar = weights === nothing ? mean(y) : sum(weights .* y) / sum(weights)
    p == 0 && return fill(wbar, size(Xnew, 1))
    ctx = "ForestLearner"
    pr = _ml_grf_params(p; num_trees=l.num_trees, sample_fraction=l.sample_fraction,
                        mtry=l.mtry === nothing ? nothing : min(l.mtry, p),
                        min_node_size=l.min_node_size, honesty=l.honesty,
                        honesty_fraction=l.honesty_fraction, honesty_prune_leaves=true,
                        alpha=l.alpha, imbalance_penalty=l.imbalance_penalty,
                        ci_group_size=1, context=ctx)
    m = n * pr.sample_fraction
    if floor(Int, m) < 1 || (pr.honesty && (floor(m * pr.honesty_fraction) < 1 ||
                                            floor(m * (1 - pr.honesty_fraction)) < 1))
        throw(ArgumentError("$(ctx): too few observations ($n) to grow honest trees"))
    end
    data = _GRFData(Matrix{Float64}(X), Vector{Float64}(y); wt=weights)
    forest = _grf_train(data, _ml_grf_options(:regression, pr, Vector{Int}[], 0),
                        task_seeds(rng, pr.num_trees))
    L = _grf_leaf_matrix(forest, Matrix{Float64}(Xnew), false, true)
    pred = _grf_collect(forest, L)[1]
    pred[isnan.(pred)] .= wbar
    return pred
end

function fitpredict_proba(l::ForestLearner, X::AbstractMatrix, y::AbstractVector,
                          Xnew::AbstractMatrix; rng::AbstractRNG=Random.default_rng(),
                          weights=nothing)
    _ml_check_binary(y)
    return clamp.(fitpredict(l, X, y, Xnew; rng=rng, weights=weights), 0.0, 1.0)
end
