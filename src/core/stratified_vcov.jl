# Stratified sandwich covariance: scores treated as independent draws within strata
# (optionally summed within clusters first) and centred at their stratum means, with
# strata independent of one another. Plugs into `FixedEffectModels.reg` through the
# `Vcov` interface (`completecases`, `materialize`, `S_hat`, `vcov`, `dof_residual`),
# and is available as a matrix function for fits obtained elsewhere.

"""
    StratifiedVcov(strata::Symbol; cluster=nothing, small_sample=true)

Stratified sandwich covariance estimator: a stratified variance wrapper around the
independent-observations (or cluster-robust) sandwich, for use as the `vcov`
argument of `FixedEffectModels.reg` and of every DrSnow estimator that fits its
regression through it (`ri_regression`, `experiment_estimate` with `:block_fe` or
`:lin`, the regression-based DiD estimators, `spillover_did`, ...).

Let ``ψ_i = x_i e_i`` be the score of observation ``i`` (the regressor vector, after
partialling out absorbed fixed effects, times the residual), and let the variance
units be the observations, or the within-cluster sums ``ψ_g = ∑_{i ∈ g} ψ_i`` when
`cluster` is given. Within stratum ``h``, which holds ``n_h`` variance units with
scores ``ψ_{h1}, …, ψ_{h n_h}`` and mean ``\\bar ψ_h``, the units are treated as
independent draws from a stratum-specific distribution; strata are independent of
one another. The variance of the total score is then estimated by

```math
\\hat Ω = ∑_{h=1}^{H} \\frac{n_h}{n_h - 1}
    ∑_{j=1}^{n_h} (ψ_{hj} - \\bar ψ_h)(ψ_{hj} - \\bar ψ_h)',
```

the with-replacement linearization variance of a stratified design (Binder 1983;
Särndal, Swensson & Wretman 1992, §5.10; `survey::svydesign(strata = …)` in Lumley
2004), and the coefficient covariance is ``(X'X)^{-1} \\hat Ω (X'X)^{-1}``. With one
stratum the estimator reduces to the cluster-robust estimator (`Vcov.cluster`) or,
without clusters, to the heteroskedasticity-robust estimator scaled by
``n / (n - 1)`` instead of ``n / (n - k)``, because the scores of a least-squares fit
sum to zero. The difference from those estimators is the centring at the stratum
means: it removes the between-stratum component of the score variation, which is
exactly the component that stratified sampling or stratified (block) randomization
eliminates. In a regression with stratum fixed effects and a treatment randomized
within strata, the estimator is asymptotically equivalent to the design-based
(Neyman) variance of the stratum-weighted treatment–control comparison, whereas the
heteroskedasticity-robust estimator overstates the variance when treatment effects
differ across strata (Bugni, Canay & Shaikh 2018); with a constant effect the two
agree.

Every stratum needs at least two variance units, and a cluster must not span several
strata; both conditions raise an `ArgumentError` rather than yielding a silently
deflated variance. With `small_sample = true` (default) the meat is multiplied by
``(N - 1) / (N - K)``, the convention of `Vcov.cluster` and reghdfe, where ``N`` is
the number of observations and ``K`` the number of estimated parameters including
absorbed fixed effects; with `small_sample = false` only the ``n_h / (n_h - 1)``
factors apply, which reproduces the `survey` package. The residual degrees of
freedom reported by the fit (`dof_residual`, used for t statistics and intervals)
are the design degrees of freedom ``∑_h (n_h - 1)``, the number of variance units
minus the number of strata.

# Arguments
- `strata::Symbol`: column holding the stratum of each observation.

# Keywords
- `cluster::Union{Nothing,Symbol}`: column holding the variance unit (cluster) of
  each observation; `nothing` (default) treats observations as the units.
- `small_sample::Bool`: multiply by ``(N - 1) / (N - K)``; default `true`.

# Returns
- `StratifiedVcov`, a `StatsBase.CovarianceEstimator` accepted by
  `FixedEffectModels.reg`. Rows with a missing stratum or cluster are dropped from
  the estimation sample.

# Examples
```julia
using DrSnow, DataFrames, FixedEffectModels, StableRNGs
rng = StableRNG(1)
H, n = 20, 10
df = DataFrame(block=repeat(1:H, inner=n))
df.z = vcat([shuffle(rng, [ones(5); zeros(5)]) for _ in 1:H]...)
df.y = 0.3 .* (df.block .- 10) .* df.z .+ df.block ./ 4 .+ randn(rng, H * n)
m = reg(df, @formula(y ~ z + fe(block)), StratifiedVcov(:block))
stderror(m)            # smaller than with Vcov.robust(): effects differ across blocks
dof_residual(m)        # 180 = 200 units - 20 strata
```

# References
- Binder, D. A. (1983). On the variances of asymptotically normal estimators from
  complex surveys. *International Statistical Review*, 51(3), 279–292.
- Bugni, F. A., Canay, I. A., & Shaikh, A. M. (2018). Inference under
  covariate-adaptive randomization. *Journal of the American Statistical
  Association*, 113(524), 1784–1796.
- Cameron, A. C., & Miller, D. L. (2015). A practitioner's guide to cluster-robust
  inference. *Journal of Human Resources*, 50(2), 317–372.
- Lumley, T. (2004). Analysis of complex survey samples. *Journal of Statistical
  Software*, 9(8), 1–19.
- Särndal, C.-E., Swensson, B., & Wretman, J. (1992). *Model Assisted Survey
  Sampling*. Springer.
"""
struct StratifiedVcov <: StatsBase.CovarianceEstimator
    strata::Symbol
    cluster::Union{Nothing,Symbol}
    small_sample::Bool
end

StratifiedVcov(strata::Symbol; cluster::Union{Nothing,Symbol}=nothing,
               small_sample::Bool=true) = StratifiedVcov(strata, cluster, small_sample)

Base.show(io::IO, v::StratifiedVcov) =
    print(io, "Stratified covariance estimator (strata: ", v.strata,
          v.cluster === nothing ? "" : ", clusters: $(v.cluster)", ")")

# Materialized estimator for the rows of the estimation sample.
struct _StratifiedMaterialized <: StatsBase.CovarianceEstimator
    unit::Vector{Int}               # row → variance unit (1..n_units)
    unit_stratum::Vector{Int}       # variance unit → stratum (1..n_strata)
    units_per_stratum::Vector{Int}
    small_sample::Bool
end

_strat_n_units(m::_StratifiedMaterialized) = length(m.unit_stratum)
_strat_n_strata(m::_StratifiedMaterialized) = length(m.units_per_stratum)

# Integer codes by first appearance (results never depend on the code values).
function _strat_codes(col)
    pos = Dict{Any,Int}()
    return Int[get!(pos, u, length(pos) + 1) for u in col]
end

# Build the materialized estimator from stratum labels and optional unit labels.
function _strat_materialize(strata::AbstractVector, cluster, small_sample::Bool,
                            context::AbstractString)
    n = length(strata)
    any(ismissing, strata) &&
        throw(ArgumentError("$context: strata have missing values"))
    scode = _strat_codes(strata)
    if cluster === nothing
        unit = collect(1:n)
        unit_stratum = scode
    else
        length(cluster) == n ||
            throw(DimensionMismatch("$context: strata and cluster lengths differ"))
        any(ismissing, cluster) &&
            throw(ArgumentError("$context: clusters have missing values"))
        unit = _strat_codes(cluster)
        G = isempty(unit) ? 0 : maximum(unit)
        unit_stratum = zeros(Int, G)
        for i in 1:n
            g = unit[i]
            if unit_stratum[g] == 0
                unit_stratum[g] = scode[i]
            elseif unit_stratum[g] != scode[i]
                throw(ArgumentError("$context: cluster $(cluster[i]) spans several " *
                                    "strata; clusters must be nested in strata"))
            end
        end
    end
    H = isempty(scode) ? 0 : maximum(scode)
    per = zeros(Int, H)
    for h in unit_stratum
        per[h] += 1
    end
    lonely = findall(<(2), per)
    if !isempty(lonely)
        labels = unique(strata)[lonely]
        throw(ArgumentError("$context: every stratum needs at least two " *
                            (cluster === nothing ? "observations" : "clusters") *
                            "; stratum " * join(repr.(labels), ", ") *
                            (length(labels) == 1 ? " has" : " have") * " one. " *
                            "Merge such strata before estimation"))
    end
    return _StratifiedMaterialized(unit, unit_stratum, per, small_sample)
end

# Scores x_i e_i; for matrix residuals (IV rank tests) one block of columns per
# residual column, the layout used by Vcov's robust and cluster estimators.
function _strat_scores(X::AbstractMatrix, r::AbstractArray)
    r isa AbstractVector && return Matrix{Float64}(X .* r)
    return reduce(hcat, [Matrix{Float64}(X .* view(r, :, k)) for k in 1:size(r, 2)])
end

# Σ_h n_h/(n_h - 1) Σ_j (ψ_hj - ψ̄_h)(ψ_hj - ψ̄_h)' for row scores S (n × p).
function _stratified_meat(S::AbstractMatrix{Float64}, m::_StratifiedMaterialized)
    n, p = size(S)
    length(m.unit) == n ||
        throw(DimensionMismatch("stratified covariance: $(length(m.unit)) rows " *
                                "expected, got $n"))
    G, H = _strat_n_units(m), _strat_n_strata(m)
    U = zeros(G, p)
    for j in 1:p
        @inbounds for i in 1:n
            U[m.unit[i], j] += S[i, j]
        end
    end
    M = zeros(H, p)
    for j in 1:p, g in 1:G
        M[m.unit_stratum[g], j] += U[g, j]
    end
    M ./= m.units_per_stratum
    for j in 1:p, g in 1:G
        h = m.unit_stratum[g]
        U[g, j] = (U[g, j] - M[h, j]) * sqrt(m.units_per_stratum[h] /
                                             (m.units_per_stratum[h] - 1))
    end
    return Matrix(Symmetric(transpose(U) * U))
end

function FixedEffectModels.Vcov.completecases(table, v::StratifiedVcov)
    Tables.istable(table) ||
        throw(ArgumentError("completecases requires a table input"))
    cols = Tables.columns(table)
    out = trues(length(Tables.rows(table)))
    for c in _vcov_columns(v)
        col = Tables.getcolumn(cols, c)
        Missing <: eltype(col) && (out .&= .!ismissing.(col))
    end
    return out
end

function FixedEffectModels.Vcov.materialize(table, v::StratifiedVcov)
    Tables.istable(table) || throw(ArgumentError("materialize requires a table input"))
    cols = Tables.columns(table)
    strata = collect(Tables.getcolumn(cols, v.strata))
    cluster = v.cluster === nothing ? nothing : collect(Tables.getcolumn(cols, v.cluster))
    return _strat_materialize(strata, cluster, v.small_sample, "StratifiedVcov")
end

function FixedEffectModels.Vcov.S_hat(x::FixedEffectModels.Vcov.VcovData,
                                      m::_StratifiedMaterialized)
    X = StatsAPI.modelmatrix(x)
    meat = _stratified_meat(_strat_scores(X, StatsAPI.residuals(x)), m)
    m.small_sample && rmul!(meat, (size(X, 1) - 1) / StatsAPI.dof_residual(x))
    return Symmetric(meat)
end

StatsAPI.dof_residual(::FixedEffectModels.Vcov.VcovData, m::_StratifiedMaterialized) =
    _strat_n_units(m) - _strat_n_strata(m)

function StatsAPI.vcov(x::FixedEffectModels.Vcov.VcovData, m::_StratifiedMaterialized)
    xtx = FixedEffectModels.Vcov.invcrossmodelmatrix(x)
    S = FixedEffectModels.Vcov.S_hat(x, m)
    return FixedEffectModels.Vcov.pinvertible(Symmetric(xtx * S * xtx))
end

"""
    stratified_vcov(X, residuals, strata; cluster=nothing, small_sample=true,
                    dof=size(X, 1) - size(X, 2)) -> Matrix{Float64}

Stratified sandwich covariance ``(X'X)^{-1} \\hat Ω (X'X)^{-1}`` of the coefficients
of a least-squares fit, computed from its design matrix and residuals.

This is the matrix interface to the estimator described in [`StratifiedVcov`](@ref),
for fits obtained outside `FixedEffectModels` (with `FixedEffectModels`, pass
`StratifiedVcov` as the `vcov` argument instead). Row ``i`` of `X` belongs to stratum
`strata[i]` and, when given, to variance unit `cluster[i]`; units are matched by
label, never by position. If fixed effects were absorbed, `X` must be the demeaned
design matrix, so that the scores are those of the Frisch–Waugh–Lovell regression,
and `dof` the residual degrees of freedom net of the absorbed effects. The
requirements of [`StratifiedVcov`](@ref) apply: at least two units per stratum, and
clusters nested in strata.

# Arguments
- `X::AbstractMatrix`: ``n × k`` design matrix.
- `residuals::AbstractVector`: the ``n`` residuals.
- `strata::AbstractVector`: stratum label of each row.

# Keywords
- `cluster::Union{Nothing,AbstractVector}`: variance-unit label of each row.
- `small_sample::Bool`: multiply by ``(n - 1) / \\text{dof}``; default `true`.
- `dof::Integer`: residual degrees of freedom used by the small-sample factor;
  default ``n - k``.

# Returns
- `Matrix{Float64}`: the ``k × k`` covariance matrix.

# Examples
```julia
using DrSnow, StableRNGs
rng = StableRNG(1)
n = 200
strata = repeat(1:20, inner=10)
X = [ones(n) randn(rng, n)]
y = X * [1.0, 0.5] .+ 0.2 .* strata .+ randn(rng, n)
b = X \\ y
V = stratified_vcov(X, y .- X * b, strata)
sqrt.(diag(V))
```

# References
- Binder, D. A. (1983). On the variances of asymptotically normal estimators from
  complex surveys. *International Statistical Review*, 51(3), 279–292.
"""
function stratified_vcov(X::AbstractMatrix, residuals::AbstractVector,
                         strata::AbstractVector; cluster=nothing,
                         small_sample::Bool=true,
                         dof::Integer=size(X, 1) - size(X, 2))
    n = size(X, 1)
    length(residuals) == n ||
        throw(DimensionMismatch("stratified_vcov: $n residuals expected"))
    length(strata) == n ||
        throw(DimensionMismatch("stratified_vcov: strata must have length $n"))
    dof > 0 || throw(ArgumentError("stratified_vcov: dof must be positive"))
    m = _strat_materialize(strata, cluster, small_sample, "stratified_vcov")
    meat = _stratified_meat(_strat_scores(X, Float64.(residuals)), m)
    small_sample && rmul!(meat, (n - 1) / dof)
    Xf = Matrix{Float64}(X)
    invXX = inv(cholesky(Symmetric(transpose(Xf) * Xf)))
    V = invXX * meat * invXX
    return Matrix(Symmetric((V + transpose(V)) / 2))
end

# Data columns a covariance estimator reads, so that estimators building a reduced
# data frame for `reg` can carry them over.
_vcov_columns(v::StratifiedVcov) =
    Symbol[c for c in (v.strata, v.cluster) if c !== nothing]
_vcov_columns(v::Vcov.ClusterCovariance) = Symbol[Symbol(c) for c in Vcov.names(v)]
_vcov_columns(::Any) = Symbol[]
