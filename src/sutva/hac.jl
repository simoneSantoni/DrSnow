# Dependence-robust variance estimators for regressions with spatially or network
# correlated errors: Conley (1999) spatial HAC and network HAC (Kojevnikov, Marmer &
# Song 2021; Leung 2022).
#
# Both are `Vcov` covariance estimators, so they plug into `FixedEffectModels.reg`
# (`reg(df, f, ConleyVcov(...))`): FixedEffectModels passes the estimation-sample rows
# (in model-matrix order) to `Vcov.materialize`, so coordinates / unit ids are read
# from the same rows as the scores and no alignment by position is involved. Fixed
# effects are handled by Frisch–Waugh–Lovell: the scores use the demeaned regressors.
#
# Meat: Σ_{a,b} k(a, b) s_a s_b', s_a = x̃_a e_a, with
#   cross-section / pooled:  k = K(dist(loc_a, loc_b))                (all pairs)
#   panel (`time` given):    k = K(dist) 1{t_a = t_b}                   (spatial part)
#                              + L(|t_a − t_b|) 1{unit_a = unit_b, t_a ≠ t_b} (serial)

const _SV_KERNELS = (:uniform, :bartlett)

"""
    ConleyVcov(; lat, lon, cutoff, kernel=:uniform, units=:km, earth_radius=nothing,
               time=nothing, unit=nothing, lag_cutoff=Inf, lag_kernel=:bartlett,
               small_sample=false, fix_psd=true)
    ConleyVcov(; x, y, cutoff, ...)
    ConleyVcov(s::SpatialStructure; unit, cutoff, ...)

Conley (1999) spatial heteroskedasticity- and autocorrelation-consistent (HAC)
covariance estimator, usable as the `vcov` argument of `FixedEffectModels.reg` and
of the DrSnow spillover regressions.

When outcomes of nearby units share unobserved shocks (weather, local labour
markets, and spillovers themselves), regression scores are spatially correlated and
heteroskedasticity-robust or unit-clustered standard errors are too small. The
estimator is the sandwich ``(X'X)^{-1} \\hat Ω (X'X)^{-1}`` with
```math
\\hat Ω = \\sum_{a} \\sum_{b} K(d_{ab}) \\, \\tilde x_a \\hat e_a \\hat e_b \\tilde x_b',
```
where ``\\tilde x_a`` are the regressors (demeaned by the fixed effects, by
Frisch–Waugh–Lovell), ``\\hat e_a`` the residuals, ``d_{ab}`` the distance between
the locations of observations ``a`` and ``b``, and ``K`` a kernel that is 1 at
distance zero and vanishes beyond `cutoff`: `:uniform` (``1\\{d ≤ c\\}``, as in
Conley 1999 and fixest) or `:bartlett` (``\\max(0, 1 - d/c)``). Consistency
requires that dependence decay with distance and that the cutoff be large relative
to the range of dependence but small relative to the extent of the sample; in
practice the cutoff is a judgement call, and results should be reported for several
cutoffs. Colella, Lalive, Sakalli and Thoenig (2019) extend the estimator to
arbitrary, possibly non-spatial, dependence structures and to 2SLS.

Neither kernel guarantees a positive-semidefinite matrix with two-dimensional
distances. With
`fix_psd = true` negative eigenvalues are set to zero (Cameron, Gelbach and Miller
2011) and a warning is issued; a warning signals that the cutoff or kernel is
poorly suited to the data. The estimator is a large-sample (sampling-based)
approximation: with few effectively independent spatial clusters it can understate
uncertainty, and it does not account for the design-based variance of randomized
treatments (see [`exposure_effects`](@ref) for design-based inference).

For panels, without `time` all pairs of observations within `cutoff` are correlated
regardless of period (repeated observations of a location are fully correlated, as
in fixest's `vcov_conley`). With `time`, spatial correlation is allowed only within
periods, and observations of the same unit (`unit`, or the location) in different
periods are correlated with lag weight `lag_kernel` up to `lag_cutoff` periods
(`Inf` gives unit clustering; the Bartlett lag weight is
``1 - |Δt| / (\\text{lag\\_cutoff} + 1)``), as in Hsiang (2010).

# Keywords
- `lat`, `lon` (degrees, haversine distance in `units`) or `x`, `y` (Euclidean):
  coordinate column names. Alternatively pass a [`SpatialStructure`](@ref) `s` and
  the `unit` column that maps rows to its units.
- `cutoff::Real`: distance beyond which scores are uncorrelated (in `units`, or in
  coordinate units for `x` / `y`).
- `kernel::Symbol`: `:uniform` (default) or `:bartlett`.
- `units::Symbol`: `:km` (default) or `:mi`; `earth_radius`: sphere radius (see
  [`SpatialStructure`](@ref)).
- `time`, `unit`: period and unit columns for panels (see above).
- `lag_cutoff::Real`: largest lag with serial correlation; default `Inf`.
- `lag_kernel::Symbol`: `:bartlett` (default) or `:uniform`.
- `small_sample::Bool`: multiply by ``n / (n - k)``; default `false`.
- `fix_psd::Bool`: repair a non-positive-semidefinite matrix; default `true`.

# Returns
- `ConleyVcov`, a `StatsBase.CovarianceEstimator` accepted by
  `FixedEffectModels.reg`.

# Examples
```julia
using DrSnow, DataFrames, FixedEffectModels, StableRNGs
rng = StableRNG(1)
n = 300
df = DataFrame(lat=45 .+ rand(rng, n), lon=9 .+ rand(rng, n), x=randn(rng, n))
df.y = 0.5 .* df.x .+ sin.(3 .* df.lat) .+ randn(rng, n)     # spatially correlated error
m = reg(df, @formula(y ~ x), ConleyVcov(; lat=:lat, lon=:lon, cutoff=25))
stderror(m)
```

# References
- Conley, T. G. (1999). GMM estimation with cross sectional dependence. *Journal of
  Econometrics*, 92(1), 1–45.
- Hsiang, S. M. (2010). Temperatures and cyclones strongly associated with economic
  production in the Caribbean and Central America. *Proceedings of the National
  Academy of Sciences*, 107(35), 15367–15372.
- Cameron, A. C., Gelbach, J. B., & Miller, D. L. (2011). Robust inference with
  multiway clustering. *Journal of Business & Economic Statistics*, 29(2), 238–249.
- Colella, F., Lalive, R., Sakalli, S. O., & Thoenig, M. (2019). Inference with
  arbitrary clustering. IZA Discussion Paper No. 12584.
"""
struct ConleyVcov{S} <: StatsBase.CovarianceEstimator
    cols::Union{Nothing,Tuple{Symbol,Symbol}}
    metric::Symbol
    units::Symbol
    radius::Union{Nothing,Float64}
    structure::S
    unit::Union{Nothing,Symbol}
    cutoff::Float64
    kernel::Symbol
    time::Union{Nothing,Symbol}
    lag_cutoff::Float64
    lag_kernel::Symbol
    small_sample::Bool
    fix_psd::Bool
end

function _sv_check_hac_options(cutoff, kernel, lag_cutoff, lag_kernel)
    cutoff > 0 || throw(ArgumentError("cutoff must be positive"))
    kernel in _SV_KERNELS || throw(ArgumentError("kernel must be one of $(_SV_KERNELS)"))
    lag_cutoff >= 0 || throw(ArgumentError("lag_cutoff must be non-negative"))
    lag_kernel in _SV_KERNELS ||
        throw(ArgumentError("lag_kernel must be one of $(_SV_KERNELS)"))
end

function ConleyVcov(; lat::Union{Nothing,Symbol}=nothing,
                    lon::Union{Nothing,Symbol}=nothing,
                    x::Union{Nothing,Symbol}=nothing, y::Union{Nothing,Symbol}=nothing,
                    cutoff::Real, kernel::Symbol=:uniform, units::Symbol=:km,
                    earth_radius=nothing,
                    time::Union{Nothing,Symbol}=nothing,
                    unit::Union{Nothing,Symbol}=nothing,
                    lag_cutoff::Real=Inf, lag_kernel::Symbol=:bartlett,
                    small_sample::Bool=false, fix_psd::Bool=true)
    geo = lat !== nothing && lon !== nothing
    cart = x !== nothing && y !== nothing
    geo ⊻ cart || throw(ArgumentError("ConleyVcov: pass either `lat` and `lon` or `x` " *
                                      "and `y` column names"))
    geo && !(units in (:km, :mi)) && throw(ArgumentError("units must be :km or :mi"))
    _sv_check_hac_options(cutoff, kernel, lag_cutoff, lag_kernel)
    return ConleyVcov(geo ? (lat, lon) : (x, y), geo ? :haversine : :euclidean,
                      geo ? units : :coordinate,
                      earth_radius === nothing ? nothing : Float64(earth_radius), nothing,
                      unit, Float64(cutoff), kernel,
                      time, Float64(lag_cutoff), lag_kernel, small_sample, fix_psd)
end

function ConleyVcov(s::SpatialStructure; unit::Symbol, cutoff::Real,
                    kernel::Symbol=:uniform, time::Union{Nothing,Symbol}=nothing,
                    lag_cutoff::Real=Inf, lag_kernel::Symbol=:bartlett,
                    small_sample::Bool=false, fix_psd::Bool=true)
    _sv_check_hac_options(cutoff, kernel, lag_cutoff, lag_kernel)
    return ConleyVcov(nothing, s.metric, s.units, s.radius, s, unit, Float64(cutoff),
                      kernel, time,
                      Float64(lag_cutoff), lag_kernel, small_sample, fix_psd)
end

function Base.show(io::IO, v::ConleyVcov)
    print(io, "Conley spatial HAC covariance estimator (", v.kernel, " kernel, cutoff ",
          _sv_fmt(v.cutoff), v.units === :coordinate ? "" : " " * string(v.units),
          v.time === nothing ? "" : ", within-period + serial", ")")
end

"""
    NetworkHACVcov(s::NetworkStructure; unit, bandwidth, kernel=:bartlett,
                   time=nothing, lag_cutoff=Inf, lag_kernel=:bartlett,
                   small_sample=false, fix_psd=true)

Network HAC covariance estimator (Kojevnikov, Marmer and Song 2021; Leung 2022),
usable as the `vcov` argument of `FixedEffectModels.reg` and of the DrSnow spillover
regressions.

When outcomes of connected units are correlated (through common shocks, peer effects
or spillovers of treatment), the scores of units that are close in the network are
dependent. The estimator has the same sandwich form as [`ConleyVcov`](@ref), with
the distance ``d_{ab}`` replaced by the undirected shortest-path distance between
the nodes of observations ``a`` and ``b`` and the kernel weights
``K(d) = 1\\{d ≤ b\\}`` (`:uniform`) or ``1 - d/(b + 1)`` (`:bartlett`) for
``d ≤ b``, with ``b`` the `bandwidth`. `bandwidth = 0` gives the
heteroskedasticity-robust estimator (unit clustering in panels). Kojevnikov, Marmer
and Song (2021) establish consistency for network-dependent processes whose
dependence decays with network distance, with a bandwidth growing slowly with the
network; Leung (2022) uses this estimator for inference on exposure effects under
approximate neighbourhood interference. The bandwidth should be related to the
typical path lengths of the network; in practice, report results for a few
bandwidths. As with the spatial version, the matrix need not be positive
semidefinite (see `fix_psd`), and the approximation is asymptotic in the number of
nodes. Panel options (`time`, `lag_cutoff`, `lag_kernel`) work as in
[`ConleyVcov`](@ref).

# Arguments
- `s::NetworkStructure`: the network; rows are mapped to nodes through `unit`.

# Keywords
- `unit::Symbol`: column holding the node identifier of each row.
- `bandwidth::Integer`: largest network distance with correlated scores (``≥ 0``).
- `kernel::Symbol`: `:bartlett` (default) or `:uniform`.
- `time`, `lag_cutoff`, `lag_kernel`, `small_sample`, `fix_psd`: as in
  [`ConleyVcov`](@ref).

# Returns
- `NetworkHACVcov`, a `StatsBase.CovarianceEstimator` accepted by
  `FixedEffectModels.reg`.

# Examples
```julia
using DrSnow, DataFrames, StableRNGs
rng = StableRNG(2)
n = 150
A = zeros(Int, n, n)
for i in 1:n, j in (i + 1):n
    rand(rng) < 0.03 && (A[i, j] = A[j, i] = 1)
end
g = NetworkStructure(1:n, A)
z = draw_assignment(rng, CompleteRandomization(n, 50))
df = DataFrame(id=1:n, z=Int.(z), y=0.5 .* z .+ randn(rng, n))
r = exposure_regression(df, :y, :z, g; unit=:id,
                        exposure=NeighborExposure(:share; isolates=:zero),
                        vcov=NetworkHACVcov(g; unit=:id, bandwidth=2))
stderror(r)
```

# References
- Kojevnikov, D., Marmer, V., & Song, K. (2021). Limit theorems for network
  dependent random variables. *Journal of Econometrics*, 222(2), 882–908.
- Leung, M. P. (2022). Causal inference under approximate neighborhood interference.
  *Econometrica*, 90(1), 267–293.
- Conley, T. G. (1999). GMM estimation with cross sectional dependence. *Journal of
  Econometrics*, 92(1), 1–45.
"""
struct NetworkHACVcov{S} <: StatsBase.CovarianceEstimator
    structure::S
    unit::Symbol
    bandwidth::Int
    kernel::Symbol
    time::Union{Nothing,Symbol}
    lag_cutoff::Float64
    lag_kernel::Symbol
    small_sample::Bool
    fix_psd::Bool
end

function NetworkHACVcov(s::NetworkStructure; unit::Symbol, bandwidth::Integer,
                        kernel::Symbol=:bartlett, time::Union{Nothing,Symbol}=nothing,
                        lag_cutoff::Real=Inf, lag_kernel::Symbol=:bartlett,
                        small_sample::Bool=false, fix_psd::Bool=true)
    bandwidth >= 0 || throw(ArgumentError("bandwidth must be non-negative"))
    _sv_check_hac_options(1.0, kernel, lag_cutoff, lag_kernel)
    return NetworkHACVcov(s, unit, Int(bandwidth), kernel, time, Float64(lag_cutoff),
                          lag_kernel, small_sample, fix_psd)
end

Base.show(io::IO, v::NetworkHACVcov) =
    print(io, "Network HAC covariance estimator (", v.kernel, " kernel, bandwidth ",
          v.bandwidth, v.time === nothing ? "" : ", within-period + serial", ")")

# Materialized estimator: everything needed to compute the meat for the sample rows.
struct _SvHACMaterialized <: StatsBase.CovarianceEstimator
    loc::Vector{Int}                       # row → location
    K::SparseMatrixCSC{Float64,Int}        # location kernel (diagonal = 1)
    tcode::Union{Nothing,Vector{Int}}      # row → period code
    tval::Vector{Float64}                  # numeric period (for lags)
    ucode::Vector{Int}                     # row → unit (serial correlation)
    lag_cutoff::Float64
    lag_kernel::Symbol
    small_sample::Bool
    fix_psd::Bool
end

_sv_kernel_weight(kernel, d, c) =
    kernel === :uniform ? Float64(d <= c) : max(0.0, 1 - d / c)

function _sv_spatial_kernel(s::SpatialStructure, cutoff, kernel)
    ii, jj, dd = _sv_pairs_within(s, cutoff)
    w = [_sv_kernel_weight(kernel, d, cutoff) for d in dd]
    n = n_units(s)
    keep = w .> 0
    return sparse(vcat(ii[keep], 1:n), vcat(jj[keep], 1:n), vcat(w[keep], ones(n)), n, n)
end

function _sv_network_kernel(s::NetworkStructure, bandwidth, kernel)
    n = n_units(s)
    bandwidth == 0 && return sparse(1.0I, n, n)
    H = shortest_path_hops(s; max_hops=bandwidth, direction=:undirected)
    ii, jj, dd = findnz(H)
    w = [kernel === :uniform ? 1.0 : 1 - d / (bandwidth + 1) for d in dd]
    return sparse(vcat(ii, 1:n), vcat(jj, 1:n), vcat(w, ones(n)), n, n)
end

function _sv_hac_columns(v::ConleyVcov)
    cols = v.cols === nothing ? Symbol[] : Symbol[v.cols...]
    for c in (v.unit, v.time)
        c === nothing || push!(cols, c)
    end
    return cols
end
_sv_hac_columns(v::NetworkHACVcov) = Symbol[c for c in (v.unit, v.time) if c !== nothing]

function FixedEffectModels.Vcov.completecases(table, v::Union{ConleyVcov,NetworkHACVcov})
    cols = Tables.columns(table)
    n = length(Tables.rows(table))
    out = trues(n)
    for c in _sv_hac_columns(v)
        col = Tables.getcolumn(cols, c)
        out .&= .!ismissing.(col)
    end
    return out
end

function _sv_time_codes(table, time, context)
    time === nothing && return nothing, Float64[]
    tcol = collect(Tables.getcolumn(Tables.columns(table), time))
    periods = sort(unique(tcol))
    pos = Dict(t => k for (k, t) in enumerate(periods))
    tval = eltype(tcol) <: Real ? Float64.(tcol) : Float64[pos[t] for t in tcol]
    return Int[pos[t] for t in tcol], tval
end

function _sv_codes(col)
    pos = Dict{Any,Int}()
    return Int[get!(pos, u, length(pos) + 1) for u in col]
end

function FixedEffectModels.Vcov.materialize(table, v::ConleyVcov)
    context = "ConleyVcov"
    cols = Tables.columns(table)
    tcode, tval = _sv_time_codes(table, v.time, context)
    if v.structure === nothing
        a = Float64.(collect(Tables.getcolumn(cols, v.cols[1])))
        b = Float64.(collect(Tables.getcolumn(cols, v.cols[2])))
        loc = _sv_codes(collect(zip(a, b)))
        L = maximum(loc)
        la = zeros(L)
        lb = zeros(L)
        la[loc] .= a
        lb[loc] .= b
        s = v.metric === :haversine ?
            SpatialStructure(1:L; lat=la, lon=lb, units=v.units, earth_radius=v.radius) :
            SpatialStructure(1:L; x=la, y=lb)
        K = _sv_spatial_kernel(s, v.cutoff, v.kernel)
        ucode = v.unit === nothing ? loc : _sv_codes(Tables.getcolumn(cols, v.unit))
    else
        loc = _sv_rows_to_index(v.structure, collect(Tables.getcolumn(cols, v.unit)),
                                context)
        K = _sv_spatial_kernel(v.structure, v.cutoff, v.kernel)
        ucode = loc
    end
    return _SvHACMaterialized(loc, K, tcode, tval, ucode, v.lag_cutoff, v.lag_kernel,
                              v.small_sample, v.fix_psd)
end

function FixedEffectModels.Vcov.materialize(table, v::NetworkHACVcov)
    cols = Tables.columns(table)
    tcode, tval = _sv_time_codes(table, v.time, "NetworkHACVcov")
    loc = _sv_rows_to_index(v.structure, collect(Tables.getcolumn(cols, v.unit)),
                            "NetworkHACVcov")
    K = _sv_network_kernel(v.structure, v.bandwidth, v.kernel)
    return _SvHACMaterialized(loc, K, tcode, tval, loc, v.lag_cutoff, v.lag_kernel,
                              v.small_sample, v.fix_psd)
end

# Σ_{a,b} k(a,b) s_a s_b' for scores S (n × p).
function _sv_hac_meat(S::AbstractMatrix{Float64}, m::_SvHACMaterialized)
    n, p = size(S)
    L = size(m.K, 1)
    if m.tcode === nothing
        G = zeros(L, p)
        for i in 1:n, j in 1:p
            G[m.loc[i], j] += S[i, j]
        end
        return transpose(G) * (m.K * G)
    end
    meat = zeros(p, p)
    T = maximum(m.tcode)
    byt = [Int[] for _ in 1:T]
    for i in 1:n
        push!(byt[m.tcode[i]], i)
    end
    G = zeros(L, p)
    for rows in byt
        isempty(rows) && continue
        fill!(G, 0.0)
        for i in rows, j in 1:p
            G[m.loc[i], j] += S[i, j]
        end
        meat .+= transpose(G) * (m.K * G)
    end
    # serial correlation within unit across different periods
    byu = Dict{Int,Vector{Int}}()
    for i in 1:n
        push!(get!(byu, m.ucode[i], Int[]), i)
    end
    for rows in values(byu), a in eachindex(rows), b in (a + 1):length(rows)
        i, j = rows[a], rows[b]
        m.tcode[i] == m.tcode[j] && continue
        lag = abs(m.tval[i] - m.tval[j])
        w = if isinf(m.lag_cutoff)
            1.0
        elseif m.lag_kernel === :uniform
            Float64(lag <= m.lag_cutoff)
        else
            max(0.0, 1 - lag / (m.lag_cutoff + 1))
        end
        w == 0 && continue
        for c in 1:p, r in 1:p
            meat[r, c] += w * (S[i, r] * S[j, c] + S[j, r] * S[i, c])
        end
    end
    return meat
end

# Scores s_a = x̃_a e_a; for IV-type matrix residuals, one block of columns per
# residual column (the layout used by Vcov's robust estimator).
function _sv_scores(X::AbstractMatrix, r::AbstractArray)
    r isa AbstractVector && return X .* r
    return reduce(hcat, [X .* view(r, :, k) for k in 1:size(r, 2)])
end

function _sv_bread_meat(invXX::AbstractMatrix, meat::AbstractMatrix, fix_psd::Bool)
    V = invXX * meat * invXX
    V = Matrix(Symmetric((V + transpose(V)) / 2))
    if fix_psd
        F = eigen(Symmetric(V))
        tol = eps() * max(1.0, maximum(abs, F.values))
        if any(<(-tol), F.values)
            @warn "HAC covariance matrix is not positive semidefinite; negative " *
                  "eigenvalues set to zero (Cameron, Gelbach & Miller 2011)"
            V = Matrix(Symmetric(F.vectors * Diagonal(max.(F.values, 0.0)) *
                                 transpose(F.vectors)))
        end
    end
    return V
end

function FixedEffectModels.Vcov.S_hat(x::FixedEffectModels.Vcov.VcovData,
                                      m::_SvHACMaterialized)
    X = StatsAPI.modelmatrix(x)
    meat = _sv_hac_meat(_sv_scores(X, StatsAPI.residuals(x)), m)
    m.small_sample && (meat .*= size(X, 1) / StatsAPI.dof_residual(x))
    return Symmetric(meat)
end

function StatsAPI.vcov(x::FixedEffectModels.Vcov.VcovData, m::_SvHACMaterialized)
    StatsAPI.residuals(x) isa AbstractVector ||
        throw(ArgumentError("HAC covariance: vector residuals expected"))
    meat = FixedEffectModels.Vcov.S_hat(x, m)
    invXX = Matrix(FixedEffectModels.Vcov.invcrossmodelmatrix(x))
    return Symmetric(_sv_bread_meat(invXX, Matrix(meat), m.fix_psd))
end

"""
    conley_vcov(X, residuals, s::SpatialStructure, ids; cutoff, kernel=:uniform,
                time=nothing, lag_cutoff=Inf, lag_kernel=:bartlett,
                small_sample=false, fix_psd=true) -> Matrix{Float64}

Conley (1999) spatial HAC sandwich covariance ``(X'X)^{-1} \\hat Ω (X'X)^{-1}`` of
the coefficients of a least-squares fit, computed from its design matrix and
residuals.

This is the matrix interface to the estimator described in [`ConleyVcov`](@ref),
for fits obtained outside `FixedEffectModels` (with `FixedEffectModels`, pass
`ConleyVcov` as the `vcov` argument instead). Row ``a`` of `X` belongs to unit
`ids[a]` of `s`, matched by identifier, and, for panels, to period `time[a]`. If
fixed effects were absorbed, `X` must be the demeaned design matrix, so that the
scores are those of the Frisch–Waugh–Lovell regression. The cutoff, kernel and
positive-semidefiniteness caveats of [`ConleyVcov`](@ref) apply.

# Arguments
- `X::AbstractMatrix`: ``n × k`` regressors (demeaned if fixed effects were
  absorbed).
- `residuals::AbstractVector`: least-squares residuals, one per row of `X`.
- `s::SpatialStructure`: locations of the units.
- `ids::AbstractVector`: unit identifier of every row.

# Keywords
- `cutoff`, `kernel`, `lag_cutoff`, `lag_kernel`, `small_sample`, `fix_psd`: as in
  [`ConleyVcov`](@ref).
- `time`: vector with the period of every row, or `nothing` for a cross-section.

# Returns
- `Matrix{Float64}`: ``k × k`` covariance matrix of the coefficients.

# Examples
```julia
using DrSnow, StableRNGs
rng = StableRNG(3)
n = 200
s = SpatialStructure(1:n; x=10 .* rand(rng, n), y=10 .* rand(rng, n))
X = hcat(ones(n), randn(rng, n))
yv = X * [1.0, 0.5] .+ randn(rng, n)
b = X \\ yv
V = conley_vcov(X, yv .- X * b, s, 1:n; cutoff=1.5)
sqrt.([V[1, 1], V[2, 2]])
```

# References
- Conley, T. G. (1999). GMM estimation with cross sectional dependence. *Journal of
  Econometrics*, 92(1), 1–45.
- Hsiang, S. M. (2010). Temperatures and cyclones strongly associated with economic
  production in the Caribbean and Central America. *Proceedings of the National
  Academy of Sciences*, 107(35), 15367–15372.
- Cameron, A. C., Gelbach, J. B., & Miller, D. L. (2011). Robust inference with
  multiway clustering. *Journal of Business & Economic Statistics*, 29(2), 238–249.
- Colella, F., Lalive, R., Sakalli, S. O., & Thoenig, M. (2019). Inference with
  arbitrary clustering. IZA Discussion Paper No. 12584.
"""
function conley_vcov(X::AbstractMatrix, residuals::AbstractVector, s::SpatialStructure,
                     ids::AbstractVector; cutoff::Real, kernel::Symbol=:uniform,
                     time=nothing, lag_cutoff::Real=Inf, lag_kernel::Symbol=:bartlett,
                     small_sample::Bool=false, fix_psd::Bool=true)
    _sv_check_hac_options(cutoff, kernel, lag_cutoff, lag_kernel)
    loc = _sv_rows_to_index(s, ids, "conley_vcov")
    K = _sv_spatial_kernel(s, Float64(cutoff), kernel)
    return _sv_matrix_hac(X, residuals, loc, K, time, lag_cutoff, lag_kernel,
                          small_sample, fix_psd)
end

"""
    network_hac_vcov(X, residuals, s::NetworkStructure, ids; bandwidth,
                     kernel=:bartlett, time=nothing, lag_cutoff=Inf,
                     lag_kernel=:bartlett, small_sample=false, fix_psd=true)
        -> Matrix{Float64}

Network HAC sandwich covariance of the coefficients of a least-squares fit, computed
from its design matrix and residuals; the matrix interface to the estimator of
[`NetworkHACVcov`](@ref).

Row ``a`` of `X` belongs to node `ids[a]` of `s`, matched by identifier. If fixed
effects were absorbed, `X` must be the demeaned design matrix. The bandwidth and
positive-semidefiniteness caveats of [`NetworkHACVcov`](@ref) apply.

# Arguments
- `X::AbstractMatrix`: ``n × k`` regressors (demeaned if fixed effects were
  absorbed).
- `residuals::AbstractVector`: least-squares residuals.
- `s::NetworkStructure`: the network.
- `ids::AbstractVector`: node identifier of every row.

# Keywords
- `bandwidth`, `kernel`, `lag_cutoff`, `lag_kernel`, `small_sample`, `fix_psd`: as
  in [`NetworkHACVcov`](@ref).
- `time`: vector with the period of every row, or `nothing` for a cross-section.

# Returns
- `Matrix{Float64}`: ``k × k`` covariance matrix of the coefficients.

# Examples
```julia
using DrSnow, StableRNGs
rng = StableRNG(4)
n = 100
A = zeros(Int, n, n)
for i in 1:n, j in (i + 1):n
    rand(rng) < 0.04 && (A[i, j] = A[j, i] = 1)
end
g = NetworkStructure(1:n, A)
X = hcat(ones(n), randn(rng, n))
yv = X * [1.0, 0.5] .+ randn(rng, n)
b = X \\ yv
network_hac_vcov(X, yv .- X * b, g, 1:n; bandwidth=2)
```

# References
- Kojevnikov, D., Marmer, V., & Song, K. (2021). Limit theorems for network
  dependent random variables. *Journal of Econometrics*, 222(2), 882–908.
- Leung, M. P. (2022). Causal inference under approximate neighborhood interference.
  *Econometrica*, 90(1), 267–293.
"""
function network_hac_vcov(X::AbstractMatrix, residuals::AbstractVector,
                          s::NetworkStructure, ids::AbstractVector; bandwidth::Integer,
                          kernel::Symbol=:bartlett, time=nothing, lag_cutoff::Real=Inf,
                          lag_kernel::Symbol=:bartlett, small_sample::Bool=false,
                          fix_psd::Bool=true)
    bandwidth >= 0 || throw(ArgumentError("bandwidth must be non-negative"))
    _sv_check_hac_options(1.0, kernel, lag_cutoff, lag_kernel)
    loc = _sv_rows_to_index(s, ids, "network_hac_vcov")
    K = _sv_network_kernel(s, bandwidth, kernel)
    return _sv_matrix_hac(X, residuals, loc, K, time, lag_cutoff, lag_kernel,
                          small_sample, fix_psd)
end

function _sv_matrix_hac(X, residuals, loc, K, time, lag_cutoff, lag_kernel,
                        small_sample, fix_psd)
    n = size(X, 1)
    (length(residuals) == n && length(loc) == n) ||
        throw(DimensionMismatch("X, residuals and ids must have the same number of rows"))
    if time === nothing
        tcode, tval = nothing, Float64[]
    else
        length(time) == n || throw(DimensionMismatch("time must have one entry per row"))
        tcode, tval = _sv_time_codes(DataFrame(t=collect(time)), :t, "HAC")
    end
    m = _SvHACMaterialized(loc, K, tcode, tval, loc, Float64(lag_cutoff), lag_kernel,
                           small_sample, fix_psd)
    Xf = Matrix{Float64}(X)
    invXX = Matrix(inv(Symmetric(transpose(Xf) * Xf)))
    meat = _sv_hac_meat(Xf .* Float64.(residuals), m)
    small_sample && (meat .*= n / (n - size(Xf, 2)))
    return _sv_bread_meat(invXX, meat, fix_psd)
end
