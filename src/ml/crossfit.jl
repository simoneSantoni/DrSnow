# Cross-fitting infrastructure shared by every estimator of the ml area: data
# extraction, fold assignment (stratified / cluster-grouped / user-supplied), per-task
# seeds, the parallel out-of-fold prediction engine, propensity clipping, and the
# linear-score estimation and variance formulas of DoubleML.

# ------------------------------------------------------------------- data access

"""Numeric `Float64` column; errors on missing or non-numeric values."""
function _ml_column(data, col::Symbol; context::AbstractString)
    v = data[!, col]
    if any(ismissing, v)
        throw(ArgumentError("$(context): column $(col) contains missing values; drop " *
                            "or impute them before estimation"))
    end
    T = nonmissingtype(eltype(v))
    if !(T <: Real)
        throw(ArgumentError("$(context): column $(col) must be numeric (got $(T)); " *
                            "encode categorical variables as dummy columns"))
    end
    out = Float64.(v)
    all(isfinite, out) ||
        throw(ArgumentError("$(context): column $(col) contains non-finite values"))
    return out
end

function _ml_matrix(data, cols::AbstractVector{Symbol}; context::AbstractString)
    n = nrow(data)
    X = Matrix{Float64}(undef, n, length(cols))
    for (j, c) in enumerate(cols)
        X[:, j] .= _ml_column(data, c; context=context)
    end
    return X
end

function _ml_check_binary_col(v, col, context)
    all(x -> x == 0 || x == 1, v) ||
        throw(ArgumentError("$(context): $(col) must be binary (0/1)"))
    (any(==(1), v) && any(==(0), v)) ||
        throw(ArgumentError("$(context): $(col) must take both values 0 and 1"))
    return nothing
end

"""
Cluster indices `1:G` for column `cluster` (or `nothing`). Groups are numbered in
sorted order of the cluster values when they are sortable, so fold assignment by
cluster does not depend on the row order of the data.
"""
function _ml_cluster_ids(data, cluster; context::AbstractString)
    cluster === nothing && return nothing, 0
    cl = cluster isa AbstractVector ? cluster : [cluster]
    length(cl) == 1 ||
        throw(ArgumentError("$(context): only one-way clustering is supported; got " *
                            "$(length(cl)) cluster variables"))
    c = Symbol(cl[1])
    v = data[!, c]
    any(ismissing, v) &&
        throw(ArgumentError("$(context): cluster column has missing values"))
    return _ml_group_index(v)
end

function _ml_group_index(v::AbstractVector)
    u = unique(v)
    try
        sort!(u)
    catch
    end
    idx = Dict(x => i for (i, x) in enumerate(u))
    g = [idx[x] for x in v]
    length(u) >= 2 || throw(ArgumentError("need at least two clusters"))
    return g, length(u)
end

# ----------------------------------------------------------------------- folds

"""
    crossfit_folds(n, n_folds=5, n_rep=1; rng=Random.default_rng(), strata=nothing,
                   groups=nothing) -> Matrix{Int}

Draw the random sample splits used for cross-fitting: an `n × n_rep` matrix whose
column `r` assigns each observation to one of `n_folds` folds in repetition `r`.

Cross-fitting (Chernozhukov et al., 2018) partitions the sample into ``K`` folds; for
each fold the nuisance functions are trained on the other ``K - 1`` folds and
evaluated on the held-out fold, so every observation receives a nuisance prediction
from a model that did not see it. This removes the own-observation
overfitting bias that arises when flexible learners are fitted and evaluated on the
same data, without the efficiency loss of a single sample split, and allows learners
whose complexity grows with ``n`` without Donsker-type conditions. The estimators of
this package solve the orthogonal moment condition once over the pooled out-of-fold
scores (the "DML2" variant). Because the estimate depends on the random partition,
Chernozhukov et al. (2018, Section 3.4) recommend repeating the split and aggregating
by the median; each column of the returned matrix is one repetition, and the
estimators aggregate repetitions as described for [`DMLEstimate`](@ref).

Fold assignment is by random permutation from `rng`. With `strata`, the permutation
is drawn within each stratum and observations are dealt to folds in turn, so every
fold contains close to a ``1/K`` share of each stratum; stratifying by a binary
treatment or instrument keeps both arms present in every training sample, which
propensity and arm-specific outcome models need. With `groups`, whole clusters are
assigned to folds, which is what cross-fitting requires when observations are
dependent within clusters: if observations of one cluster appeared both in a training
fold and in the evaluation fold, within-cluster dependence would reintroduce the
overfitting bias that cross-fitting removes. Estimators with a `cluster` keyword build
their folds this way automatically. The number of folds trades off the size of the
training samples (larger ``K`` gives nuisance fits closer to the full-sample fit)
against computation; ``K = 5`` is the DoubleML default.

# Arguments
- `n::Integer`: number of observations (rows of the data the folds will be used with).
- `n_folds::Integer = 5`: number of folds ``K`` per repetition; at least 2, and at
  most `n` (or the number of clusters with `groups`).
- `n_rep::Integer = 1`: number of independent repetitions of the split.

# Keywords
- `rng::AbstractRNG = Random.default_rng()`: random-number generator for the
  permutations; pass a seeded generator (e.g. `StableRNG`) for reproducible splits.
- `strata = nothing`: optional vector of length `n`; folds are balanced within each
  of its distinct values (e.g. the treatment indicator).
- `groups = nothing`: optional vector of cluster labels of length `n`; all
  observations of a cluster share a fold. The assignment is made on the sorted
  cluster labels, so it does not depend on the row order of the data (without
  `groups`, folds are tied to row positions). `strata` is then applied at the cluster
  level when it is constant within clusters, and ignored otherwise.

# Returns
- `Matrix{Int}` of size `n × n_rep` with entries in `1:n_folds`. Pass it (or one
  column) to the `folds` keyword of any cross-fitted estimator to reuse the same
  sample splits, for example to compare specifications or learners on identical
  partitions.

# Examples
```julia
using DrSnow, DataFrames, StableRNGs
rng = StableRNG(1)
n = 400
df = DataFrame(x1=randn(rng, n), x2=randn(rng, n))
df.d = Float64.(rand(rng, n) .< 1 ./ (1 .+ exp.(-df.x1)))
df.y = df.d .+ df.x1 .+ 0.5 .* df.x2 .+ randn(rng, n)
F = crossfit_folds(nrow(df), 5, 3; rng=StableRNG(2), strata=df.d)
r1 = dml_plr(df, :y, :d; covariates=[:x1, :x2], folds=F)
r2 = dml_irm(df, :y, :d; covariates=[:x1, :x2], folds=F)
```

# References
- Chernozhukov, V., Chetverikov, D., Demirer, M., Duflo, E., Hansen, C., Newey, W.,
  & Robins, J. (2018). Double/debiased machine learning for treatment and structural
  parameters. *The Econometrics Journal*, 21(1), C1–C68.
- Bach, P., Kurz, M. S., Chernozhukov, V., Spindler, M., & Klaassen, S. (2024).
  DoubleML: An object-oriented implementation of double machine learning in R.
  *Journal of Statistical Software*, 108(3), 1–56.
"""
function crossfit_folds(n::Integer, n_folds::Integer=5, n_rep::Integer=1;
                        rng::AbstractRNG=Random.default_rng(), strata=nothing,
                        groups=nothing)
    n_folds >= 2 || throw(ArgumentError("n_folds must be at least 2"))
    n_rep >= 1 || throw(ArgumentError("n_rep must be at least 1"))
    strata === nothing || length(strata) == n ||
        throw(DimensionMismatch("strata must have length $n"))
    F = zeros(Int, n, n_rep)
    if groups === nothing
        n >= n_folds ||
            throw(ArgumentError("need at least n_folds = $n_folds observations"))
        for r in 1:n_rep
            F[:, r] .= _ml_simple_folds(rng, n, n_folds, strata)
        end
    else
        length(groups) == n || throw(DimensionMismatch("groups must have length $n"))
        g, G = _ml_group_index(groups)
        G >= n_folds ||
            throw(ArgumentError("need at least n_folds = $n_folds clusters, got $G"))
        gstrata = nothing
        if strata !== nothing
            first_s = Dict{Int,Any}()
            constant = true
            for i in 1:n
                s = get!(first_s, g[i], strata[i])
                s == strata[i] || (constant = false; break)
            end
            constant && (gstrata = [first_s[k] for k in 1:G])
        end
        for r in 1:n_rep
            fg = _ml_simple_folds(rng, G, n_folds, gstrata)
            F[:, r] .= fg[g]
        end
    end
    return F
end

"""Validate / construct the fold matrix for an estimator call."""
function _ml_resolve_folds(data, folds, n, n_folds, n_rep, rng, strata, groups;
                           context)
    if folds === nothing
        return crossfit_folds(n, n_folds, n_rep; rng=rng, strata=strata, groups=groups)
    end
    F = if folds isa Symbol
        require_columns(data, [folds]; context=context)
        reshape(Int.(data[!, folds]), :, 1)
    elseif folds isa AbstractVector
        reshape(Int.(folds), :, 1)
    elseif folds isa AbstractMatrix
        Matrix{Int}(folds)
    else
        throw(ArgumentError("$(context): folds must be a column name, vector or matrix"))
    end
    size(F, 1) == n ||
        throw(DimensionMismatch("$(context): folds must have $n rows, got $(size(F, 1))"))
    for r in axes(F, 2)
        ks = sort(unique(F[:, r]))
        (length(ks) >= 2 && ks == collect(1:length(ks))) ||
            throw(ArgumentError("$(context): fold ids in each column must be 1:K " *
                                "with K ≥ 2"))
        length(ks) == length(unique(F[:, 1])) ||
            throw(ArgumentError("$(context): all repetitions must use the same K"))
    end
    if groups !== nothing
        for r in axes(F, 2)
            seen = Dict{eltype(groups),Int}()
            for (gi, fi) in zip(groups, view(F, :, r))
                get!(seen, gi, fi) == fi ||
                    throw(ArgumentError("$(context): with `cluster`, all observations " *
                                        "of a cluster must share a fold"))
            end
        end
    end
    return F
end

# ------------------------------------------------------------ nuisance engine

# One nuisance function to cross-fit: a learner, its target, the covariate matrix,
# whether probabilities are needed, and an optional training restriction.
struct _MLNuisance
    name::Symbol
    learner::Any
    target::Vector{Float64}
    X::Matrix{Float64}
    proba::Bool
    train_mask::Union{Nothing,BitVector}
end

_MLNuisance(name, learner, target, X, proba) =
    _MLNuisance(name, learner, target, X, proba, nothing)

"""
Cross-fit the nuisances `specs` over the folds in column `r` of `F`. `seeds[k, s]`
seeds task (fold `k`, nuisance `s`). Returns out-of-fold predictions (n × S).
Tasks run on separate threads when `parallel` and results are identical either way.
"""
function _ml_crossfit(specs::Vector{_MLNuisance}, fold::AbstractVector{Int},
                      seeds::AbstractMatrix{UInt64}; parallel::Bool, context)
    n = length(fold)
    K = maximum(fold)
    S = length(specs)
    size(seeds, 1) >= K && size(seeds, 2) >= S ||
        throw(ArgumentError("internal: seed array too small"))
    out = Matrix{Float64}(undef, n, S)
    jobs = [(k, s) for s in 1:S for k in 1:K]
    run = function (job)
        k, s = job
        sp = specs[s]
        test = fold .== k
        train = .!test
        sp.train_mask === nothing || (train .&= sp.train_mask)
        ntr = count(train)
        ntr > 0 || throw(ArgumentError("$(context): no training observations for " *
                                       "nuisance $(sp.name) in fold $k"))
        ytr = sp.target[train]
        if sp.proba && (all(==(0), ytr) || all(==(1), ytr))
            throw(ArgumentError("$(context): the training sample of fold $k for " *
                                "nuisance $(sp.name) contains a single class; use " *
                                "fewer folds or stratified folds"))
        end
        Xtr = sp.X[train, :]
        Xte = sp.X[test, :]
        trng = Random.Xoshiro(seeds[k, s])
        pred = sp.proba ?
               fitpredict_proba(sp.learner, Xtr, ytr, Xte; rng=trng) :
               fitpredict(sp.learner, Xtr, ytr, Xte; rng=trng)
        length(pred) == count(test) ||
            throw(DimensionMismatch("$(context): learner for $(sp.name) returned " *
                                    "$(length(pred)) predictions for $(count(test)) rows"))
        all(isfinite, pred) ||
            throw(ArgumentError("$(context): learner for $(sp.name) returned " *
                                "non-finite predictions in fold $k"))
        out[test, s] .= pred
        return nothing
    end
    if parallel && Threads.nthreads() > 1 && length(jobs) > 1
        tasks = [Threads.@spawn run(j) for j in jobs]
        for t in tasks
            try
                fetch(t)
            catch e
                e isa TaskFailedException ? throw(e.task.exception) : rethrow()
            end
        end
    else
        foreach(run, jobs)
    end
    return out
end

"""Seeds for (fold, nuisance slot, repetition) drawn up front from `rng`."""
_ml_seeds(rng::AbstractRNG, K::Integer, S::Integer, R::Integer) =
    reshape(task_seeds(rng, K * S * R), K, S, R)

"""Clip propensities to `[trim, 1 - trim]` in place; returns the number clipped."""
function _ml_clip!(m::AbstractVector, trim::Real)
    trim == 0 && return 0
    c = 0
    for i in eachindex(m)
        if m[i] < trim
            m[i] = trim
            c += 1
        elseif m[i] > 1 - trim
            m[i] = 1 - trim
            c += 1
        end
    end
    return c
end

function _ml_check_trim(trim)
    0 <= trim < 0.5 || throw(ArgumentError("trim must be in [0, 0.5)"))
    return Float64(trim)
end

# --------------------------------------------------------- linear-score algebra

"""
Solve the linear moment `mean(ψ_a) θ + mean(ψ_b) = 0` (DML2) and return
`(θ, ψ, var)` where the variance is `mean(ψ²) / mean(ψ_a)² / n` (DoubleML) or, with
clusters, `c · Σ_g (Σ_{i∈g} ψᵢ)² / (Σᵢ ψ_a,i)²` with `c = G/(G-1)`.
"""
function _ml_solve_score(psi_a::AbstractVector, psi_b::AbstractVector, cluster, G)
    sa = sum(psi_a)
    abs(sa) > 0 || throw(ArgumentError("the score is not identified: mean(ψ_a) = 0"))
    θ = -sum(psi_b) / sa
    psi = psi_a .* θ .+ psi_b
    v = _ml_score_cov(reshape(psi, :, 1), [sa / length(psi)], cluster, G)[1, 1]
    return θ, psi, v
end

"""
Covariance of the DML estimators of several parameters from their scores `Ψ`
(n × T) and Jacobians `J` (mean ψ_a per parameter): `J⁻¹ Ω J⁻¹ / n`.
"""
function _ml_score_cov(Ψ::AbstractMatrix, J::AbstractVector, cluster, G)
    n = size(Ψ, 1)
    Ω = if cluster === nothing
        (Ψ' * Ψ) ./ n
    else
        S = zeros(G, size(Ψ, 2))
        for i in 1:n
            @views S[cluster[i], :] .+= Ψ[i, :]
        end
        (S' * S) ./ n .* (G / (G - 1))
    end
    Jinv = 1 ./ J
    return (Jinv .* Ω .* Jinv') ./ n
end

"""
Aggregate estimates over repeated cross-fitting with the median rule of Chernozhukov
et al. (2018, §3.4): θ̃_j = median_r θ_{r,j} and
Var(θ̃_j) = median_r(se²_{r,j} + (θ_{r,j} - θ̃_j)²), where se²_{r,j} is the estimated
variance of θ_{r,j} itself. The dispersion term is added on the Var(θ̂) scale, not to
the √n-scaled asymptotic variance as in the literal §3.4 rule and in DoubleML (where
it enters divided by n), so it carries n times more weight and the standard error is
at least as large (more conservative). The off-diagonal covariances use the average
correlation across repetitions, which keeps the matrix positive semi-definite.
"""
function _ml_aggregate(all_coef::AbstractMatrix, all_vcov::Vector{<:AbstractMatrix})
    T, R = size(all_coef)
    θ = [median(all_coef[j, :]) for j in 1:T]
    R == 1 && return θ, Matrix(Symmetric(all_vcov[1]))
    var = [median([all_vcov[r][j, j] + (all_coef[j, r] - θ[j])^2 for r in 1:R])
           for j in 1:T]
    C = zeros(T, T)
    for r in 1:R
        s = sqrt.(max.(diag(all_vcov[r]), 0.0))
        s[s .== 0] .= 1.0
        C .+= all_vcov[r] ./ (s .* s')
    end
    C ./= R
    for j in 1:T
        C[j, j] = 1.0
    end
    sd = sqrt.(var)
    return θ, Matrix(Symmetric(C .* (sd .* sd')))
end

"""Dimension checks shared by estimator front ends."""
function _ml_check_common(n_folds, n_rep)
    n_folds >= 2 || throw(ArgumentError("n_folds must be at least 2"))
    n_rep >= 1 || throw(ArgumentError("n_rep must be at least 1"))
    return nothing
end
