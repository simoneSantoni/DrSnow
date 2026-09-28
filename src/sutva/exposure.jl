# Exposure mappings: how the treatment vector translates into each unit's exposure.
#
# Every specification is "prepared" once for a structure into sparse matrices; the
# exposure for an N × T treatment matrix Z is then a handful of sparse products, so
# panel exposures cost O(nnz · T) and Monte Carlo draws O(nnz) each.

"""
    ExposureSpec

Abstract supertype of exposure specifications: rules that summarize the treatments
of a unit's neighbours into one or more numeric *exposure* variables.

An exposure mapping ``f_i(z)`` reduces the ``2^N`` possible assignment vectors to a
low-dimensional summary that is assumed (or, following Sävje 2024, used to define
effects as if) to capture how the assignment affects unit ``i``'s outcome, so that
``Y_i(z) = Y_i(z_i, f_i(z))`` (Manski 2013's "effective treatments"; Aronow and
Samii 2017). The concrete specifications are [`NeighborExposure`](@ref) (shares,
counts or indicators of treated neighbours), [`RingExposure`](@ref) (distance
bands), [`HopExposure`](@ref) (network shells by shortest-path distance) and
[`CustomExposure`](@ref). Evaluate one with [`compute_exposure`](@ref), discretize it
into conditions with [`ExposureMapping`](@ref), or use it directly as regressors in
[`spillover_did`](@ref) and [`exposure_regression`](@ref).

# Examples
```julia
using DrSnow
spec = RingExposure([10.0, 20.0])
spec isa ExposureSpec                 # true
exposure_columns(spec)                # ["ring_0_10", "ring_10_20"]
```

# References
- Manski, C. F. (2013). Identification of treatment response with social
  interactions. *Econometrics Journal*, 16(1), S1–S23.
- Aronow, P. M., & Samii, C. (2017). Estimating average causal effects under general
  interference, with application to a social network experiment. *Annals of Applied
  Statistics*, 11(4), 1912–1947.
- Sävje, F. (2024). Causal inference with misspecified exposure mappings: Separating
  definitions and assumptions. *Biometrika*, 111(1), 1–15.
"""
abstract type ExposureSpec end

const _SV_NEIGHBOR_STATS = (:share, :count, :any, :weighted_sum, :weighted_share)
const _SV_BAND_STATS = (:nearest, :any, :count, :share)

"""
    NeighborExposure(stat=:share; threshold=nothing, radius=nothing, decay=nothing,
                     isolates=:missing)

Exposure to treated neighbours: a summary of the treatments of the units that can
affect unit ``i`` (network in-neighbours, units within `radius` of a spatial
structure, or the other members of a partition group).

With neighbour set ``N_i`` and weights ``w_{ij}``, the available summaries are the
share ``\\sum_{j ∈ N_i} z_j / |N_i|`` (`:share`), the count ``\\sum_{j ∈ N_i} z_j``
(`:count`), the indicator of at least one treated neighbour (`:any`), the weighted
sum ``\\sum_j w_{ij} z_j`` (`:weighted_sum`) and the weighted share
``\\sum_j w_{ij} z_j / \\sum_j w_{ij}`` (`:weighted_share`). Weights are the network
edge weights, or `decay(distance)` for spatial structures (default 1). With
`threshold = c` the exposure becomes the indicator that the statistic is at least
``c``, e.g. `NeighborExposure(:share; threshold=0.5)` for "at least half of the
neighbours treated", a common "fractional ``q``-neighbourhood" exposure.

The choice between shares and counts is substantive: shares assume that a unit with
many neighbours is affected by the proportion treated, counts that each treated
neighbour adds to the exposure. Units without any neighbour (isolates) get a
`missing` exposure by default: they can never be exposed, so their exposure is not
"zero treated neighbours" in any meaningful sense; `isolates=:zero` codes them as
unexposed instead.

# Arguments
- `stat::Symbol`: `:share` (default), `:count`, `:any`, `:weighted_sum` or
  `:weighted_share`.

# Keywords
- `threshold`: if given, the exposure is the indicator `stat ≥ threshold`.
- `radius`: neighbourhood radius for spatial structures (required there; not
  allowed otherwise).
- `decay`: function of distance giving spatial weights (spatial structures only),
  e.g. `d -> exp(-d / 10)`.
- `isolates::Symbol`: `:missing` (default) or `:zero`.

# Returns
- `NeighborExposure`.

# Examples
```julia
using DrSnow
g = NetworkStructure(1:4, [0 1 1 0; 1 0 0 0; 1 0 0 0; 0 0 0 0])
z = [false, true, true, false]
compute_exposure(g, z, NeighborExposure(:any))        # unit 4 is an isolate
s = SpatialStructure(1:4; x=[0.0, 1.0, 3.0, 9.0], y=zeros(4))
compute_exposure(s, z, NeighborExposure(:weighted_share; radius=5.0,
                                        decay=d -> exp(-d / 2)))
```

# References
- Aronow, P. M., & Samii, C. (2017). Estimating average causal effects under general
  interference, with application to a social network experiment. *Annals of Applied
  Statistics*, 11(4), 1912–1947.
- Ugander, J., Karrer, B., Backstrom, L., & Kleinberg, J. (2013). Graph cluster
  randomization: Network exposure to multiple universes. *Proceedings of the 19th
  ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*,
  329–337.
"""
struct NeighborExposure <: ExposureSpec
    stat::Symbol
    threshold::Union{Nothing,Float64}
    radius::Union{Nothing,Float64}
    decay::Any
    isolates::Symbol
end

function NeighborExposure(stat::Symbol=:share; threshold=nothing, radius=nothing,
                          decay=nothing, isolates::Symbol=:missing)
    stat in _SV_NEIGHBOR_STATS ||
        throw(ArgumentError("NeighborExposure: stat must be one of $(_SV_NEIGHBOR_STATS)"))
    isolates in (:missing, :zero) ||
        throw(ArgumentError("isolates must be :missing or :zero"))
    radius === nothing || radius > 0 || throw(ArgumentError("radius must be positive"))
    return NeighborExposure(stat, threshold === nothing ? nothing : Float64(threshold),
                            radius === nothing ? nothing : Float64(radius), decay,
                            isolates)
end

"""
    RingExposure(radii; stat=:nearest)

Distance-band ("ring") exposure for spatial structures, following Butts (2021).

The increasing outer radii ``r_1 < r_2 < … < r_K`` define bands: band ``k`` holds
the other units at distance ``d ∈ (r_{k-1}, r_k]``, with band 1 equal to
``[0, r_1]``. Exposure column ``k`` records treated units in band ``k``. Units
farther than ``r_K`` from every treated unit are unexposed; they are the "clean
controls" of a ring design, and ``r_K`` is the maintained assumption on the maximal
reach of spillovers. With `:nearest`, the rings are mutually exclusive (a unit is
assigned to the band of its nearest treated unit), which gives the specification of
Butts (2021) in which each ring coefficient is the average spillover at that
distance.

# Arguments
- `radii::AbstractVector{<:Real}`: positive, strictly increasing outer radii, in the
  distance units of the structure.

# Keywords
- `stat::Symbol`: `:nearest` (default; indicator that the *nearest* treated unit
  lies in band ``k``), `:any` (at least one treated unit in band ``k``), `:count`
  (number of treated units in band ``k``) or `:share` (fraction of band-``k`` units
  that are treated; `missing` if the band is empty).

# Returns
- `RingExposure`.

# Examples
```julia
using DrSnow
s = SpatialStructure(1:5; x=[0.0, 4.0, 8.0, 15.0, 40.0], y=zeros(5))
compute_exposure(s, [true, false, false, false, false], RingExposure([5.0, 10.0, 20.0]))
```

# References
- Butts, K. (2021). Difference-in-differences estimation with spatial spillovers.
  arXiv:2105.03737.
"""
struct RingExposure <: ExposureSpec
    radii::Vector{Float64}
    stat::Symbol
end

function RingExposure(radii::AbstractVector{<:Real}; stat::Symbol=:nearest)
    r = Float64.(collect(radii))
    isempty(r) && throw(ArgumentError("RingExposure: at least one radius is required"))
    (r[1] > 0 && all(diff(r) .> 0)) ||
        throw(ArgumentError("RingExposure: radii must be positive and increasing"))
    stat in _SV_BAND_STATS ||
        throw(ArgumentError("RingExposure: stat must be one of $(_SV_BAND_STATS)"))
    return RingExposure(r, stat)
end

"""
    HopExposure(max_hops; stat=:nearest, isolates=:missing)

Network exposure by shortest-path distance: column ``k`` refers to the units whose
shortest path to unit ``i`` has exactly ``k`` edges, ``k = 1, …, `` `max_hops`.

Distances are computed by breadth-first search ([`shortest_path_hops`](@ref)), not
by powers of the adjacency matrix: a unit is counted at a single distance, so the
2-hop term does not pick up first-order spillovers through the many walks of length
two that exist in clustered graphs. Truncating at `max_hops` is the maintained
assumption that treatments farther away have no effect; Leung (2022) studies
inference when such effects are small but non-zero (approximate neighbourhood
interference).

# Arguments
- `max_hops::Integer`: number of distance shells (at least 1).

# Keywords
- `stat::Symbol`: `:nearest` (default; indicator that the closest treated unit is at
  exactly ``k`` hops), `:any`, `:count` or `:share` (fraction of the units at
  distance ``k`` that are treated; `missing` if there are none).
- `isolates::Symbol`: `:missing` (default) or `:zero` for units without
  in-neighbours.

# Returns
- `HopExposure`.

# Examples
```julia
using DrSnow
g = NetworkStructure(1:5, [0 1 0 0 0; 1 0 1 0 0; 0 1 0 1 0; 0 0 1 0 1; 0 0 0 1 0])
compute_exposure(g, [true, false, false, false, false], HopExposure(2; stat=:any))
```

# References
- Leung, M. P. (2022). Causal inference under approximate neighborhood interference.
  *Econometrica*, 90(1), 267–293.
"""
struct HopExposure <: ExposureSpec
    max_hops::Int
    stat::Symbol
    isolates::Symbol
end

function HopExposure(max_hops::Integer; stat::Symbol=:nearest, isolates::Symbol=:missing)
    max_hops >= 1 || throw(ArgumentError("HopExposure: max_hops must be ≥ 1"))
    stat in _SV_BAND_STATS ||
        throw(ArgumentError("HopExposure: stat must be one of $(_SV_BAND_STATS)"))
    isolates in (:missing, :zero) ||
        throw(ArgumentError("isolates must be :missing or :zero"))
    return HopExposure(Int(max_hops), stat, isolates)
end

"""
    CustomExposure(f, names)

User-defined exposure specification: `f(structure, z)` receives the interference
structure and the treatment vector `z` and returns the exposure of every unit.

`z` is in the order of [`structure_units`](@ref), with `Float64` entries and
`missing` where a unit's treatment is unobserved (e.g. absent in a period). `f`
returns a vector (one exposure column) or an ``N × `` `length(names)` matrix;
`missing` entries mark undefined exposures. Use it for exposures that combine
several structures or covariates (e.g. population-weighted shares).

# Arguments
- `f`: function `(structure, z) -> AbstractVector` or `AbstractMatrix`.
- `names`: a `String` or a vector of `String`s naming the exposure columns.

# Returns
- `CustomExposure`.

# Examples
```julia
using DrSnow
pop = [100.0, 50.0, 200.0, 80.0]
s = SpatialStructure(1:4; x=[0.0, 1.0, 2.0, 3.0], y=zeros(4))
f(s, z) = (W = neighbor_matrix(s; radius=1.5); (W * (z .* pop)) ./ (W * pop))
compute_exposure(s, [1, 0, 0, 1], CustomExposure(f, "pop_share"))
```
"""
struct CustomExposure{F} <: ExposureSpec
    f::F
    names::Vector{String}
end
CustomExposure(f, name::AbstractString) = CustomExposure(f, [String(name)])
CustomExposure(f, names::AbstractVector) = CustomExposure(f, String.(collect(names)))

_sv_fmt(x) = @sprintf("%g", x)

"""
    exposure_columns(spec::ExposureSpec) -> Vector{String}

Names of the exposure columns produced by `spec`, as used by
[`compute_exposure`](@ref) and in the coefficient names of
[`spillover_did`](@ref) and [`exposure_regression`](@ref).

# Arguments
- `spec::ExposureSpec`: an exposure specification.

# Returns
- `Vector{String}`, e.g. `["ring_0_10", "ring_10_20"]`, `["hop1", "hop2"]`,
  `["share"]` or `["share_ge_0.5"]`.

# Examples
```julia
using DrSnow
exposure_columns(RingExposure([10.0, 20.0]))               # ["ring_0_10", "ring_10_20"]
exposure_columns(NeighborExposure(:share; threshold=0.5))  # ["share_ge_0.5"]
```
"""
function exposure_columns(spec::NeighborExposure)
    base = string(spec.stat)
    spec.threshold === nothing && return [base]
    return [base * "_ge_" * _sv_fmt(spec.threshold)]
end
function exposure_columns(spec::RingExposure)
    lo = vcat(0.0, spec.radii[1:(end - 1)])
    return ["ring_" * _sv_fmt(a) * "_" * _sv_fmt(b) for (a, b) in zip(lo, spec.radii)]
end
exposure_columns(spec::HopExposure) = ["hop$k" for k in 1:spec.max_hops]
exposure_columns(spec::CustomExposure) = copy(spec.names)

# ---------------------------------------------------------------------------------
# Preparation: turn (structure, spec) into sparse matrices
# ---------------------------------------------------------------------------------

struct _SVPrepared{S,F}
    kind::Symbol                                  # :neighbor, :band, :custom
    stat::Symbol
    threshold::Union{Nothing,Float64}
    isolates::Symbol
    pattern::Vector{SparseMatrixCSC{Float64,Int}}  # binary neighbour sets per column
    weights::Union{Nothing,SparseMatrixCSC{Float64,Int}}
    names::Vector{String}
    structure::S
    f::F
    n::Int
end

function _sv_binary(M::SparseMatrixCSC)
    B = SparseMatrixCSC{Float64,Int}(M)
    fill!(nonzeros(B), 1.0)
    return B
end

function _sv_prepare(s::InterferenceStructure, spec::NeighborExposure)
    n = n_units(s)
    if s isa SpatialStructure
        spec.radius === nothing && throw(ArgumentError("NeighborExposure on a spatial " *
                                                       "structure requires `radius`"))
        I, J, D = _sv_pairs_within(s, spec.radius)
        B = sparse(I, J, ones(length(I)), n, n)
        W = if spec.decay === nothing
            B
        else
            w = Float64[spec.decay(d) for d in D]
            all(x -> isfinite(x) && x >= 0, w) ||
                throw(ArgumentError("decay(d) must be finite and non-negative"))
            sparse(I, J, w, n, n)
        end
    else
        spec.radius === nothing ||
            throw(ArgumentError("`radius` applies only to spatial structures"))
        spec.decay === nothing ||
            throw(ArgumentError("`decay` applies only to spatial structures"))
        B = neighbor_matrix(s)
        W = s isa NetworkStructure ? copy(s.A) : B
    end
    return _SVPrepared(:neighbor, spec.stat, spec.threshold, spec.isolates, [B], W,
                       exposure_columns(spec), s, nothing, n)
end

function _sv_prepare(s::InterferenceStructure, spec::RingExposure)
    s isa SpatialStructure ||
        throw(ArgumentError("RingExposure requires a SpatialStructure"))
    n = n_units(s)
    I, J, D = _sv_pairs_within(s, spec.radii[end])
    lo = vcat(-Inf, spec.radii[1:(end - 1)])
    mats = SparseMatrixCSC{Float64,Int}[]
    for (a, b) in zip(lo, spec.radii)
        m = (D .> a) .& (D .<= b)
        push!(mats, sparse(I[m], J[m], ones(count(m)), n, n))
    end
    return _SVPrepared(:band, spec.stat, nothing, :none, mats, nothing,
                       exposure_columns(spec), s, nothing, n)
end

function _sv_prepare(s::InterferenceStructure, spec::HopExposure)
    s isa NetworkStructure || throw(ArgumentError("HopExposure requires a " *
                                                  "NetworkStructure"))
    n = n_units(s)
    H = shortest_path_hops(s; max_hops=spec.max_hops)
    I, J, V = findnz(H)
    mats = [sparse(I[V .== k], J[V .== k], ones(count(==(k), V)), n, n)
            for k in 1:spec.max_hops]
    return _SVPrepared(:band, spec.stat, nothing, spec.isolates, mats, nothing,
                       exposure_columns(spec), s, nothing, n)
end

function _sv_prepare(s::InterferenceStructure, spec::CustomExposure)
    return _SVPrepared(:custom, :custom, nothing, :none,
                       SparseMatrixCSC{Float64,Int}[], nothing, spec.names, s, spec.f,
                       n_units(s))
end

# ---------------------------------------------------------------------------------
# Evaluation on an N × T treatment matrix
# ---------------------------------------------------------------------------------

# Z: N × T Float64 (0/1, unobserved entries set to 0); O: N × T observed mask or
# `nothing` (all observed). Returns a K-vector of N × T matrices (NaN = missing).
function _sv_eval(p::_SVPrepared, Z::AbstractMatrix{Float64},
                  O::Union{Nothing,AbstractMatrix{Bool}})
    p.kind === :custom && return _sv_eval_custom(p, Z, O)
    N, T = size(Z)
    U = O === nothing ? nothing : Float64.(.!O)      # unobserved indicator
    if p.kind === :neighbor
        B = p.pattern[1]
        deg = vec(sum(B; dims=2))
        cnt = B * Z
        val = if p.stat === :count
            cnt
        elseif p.stat === :any
            Float64.(cnt .> 0)
        elseif p.stat === :share
            cnt ./ deg
        elseif p.stat === :weighted_sum
            p.weights * Z
        else
            wdeg = vec(sum(p.weights; dims=2))
            (p.weights * Z) ./ wdeg
        end
        if p.threshold !== nothing
            val = map(v -> isnan(v) ? NaN : Float64(v >= p.threshold), val)
        end
        iso = deg .== 0
        if any(iso)
            fillv = p.isolates === :missing ? NaN :
                    p.threshold === nothing ? 0.0 : Float64(0.0 >= p.threshold)
            val[iso, :] .= fillv
        end
        if U !== nothing
            val[(B * U) .> 0] .= NaN
        end
        return [val]
    end
    # distance bands / hop shells
    K = length(p.pattern)
    cnts = [M * Z for M in p.pattern]
    degs = [vec(sum(M; dims=2)) for M in p.pattern]
    unob = U === nothing ? nothing : [(M * U) .> 0 for M in p.pattern]
    out = Vector{Matrix{Float64}}(undef, K)
    if p.stat === :nearest
        seen = zeros(Bool, N, T)
        for k in 1:K
            out[k] = Float64.((cnts[k] .> 0) .& .!seen)
            seen .|= cnts[k] .> 0
        end
        if unob !== nothing
            anyunob = reduce(.|, unob)
            for k in 1:K
                out[k][anyunob] .= NaN
            end
        end
    else
        for k in 1:K
            out[k] = p.stat === :count ? copy(cnts[k]) :
                     p.stat === :any ? Float64.(cnts[k] .> 0) :
                     cnts[k] ./ degs[k]                      # :share, 0/0 = NaN
            unob === nothing || (out[k][unob[k]] .= NaN)
        end
    end
    if p.isolates === :missing
        iso = vec(sum(p.pattern[1]; dims=2)) .== 0          # no in-neighbours at all
        for k in 1:K
            out[k][iso, :] .= NaN
        end
    end
    return out
end

function _sv_eval_custom(p::_SVPrepared, Z, O)
    N, T = size(Z)
    K = length(p.names)
    out = [fill(NaN, N, T) for _ in 1:K]
    for t in 1:T
        zt = Union{Missing,Float64}[(O === nothing || O[i, t]) ? Z[i, t] : missing
                                    for i in 1:N]
        r = p.f(p.structure, zt)
        M = r isa AbstractVector ? reshape(collect(r), :, 1) : collect(r)
        size(M) == (N, K) || throw(DimensionMismatch("CustomExposure: function " *
                                                     "returned size $(size(M)), " *
                                                     "expected ($N, $K)"))
        for k in 1:K, i in 1:N
            v = M[i, k]
            out[k][i, t] = ismissing(v) ? NaN : Float64(v)
        end
    end
    return out
end

# Exposure of a single assignment vector, as an N × K matrix with NaN = undefined.
function _sv_eval_vector(p::_SVPrepared, z::AbstractVector)
    E = _sv_eval(p, reshape(Float64.(z), :, 1), nothing)
    return reduce(hcat, [vec(e) for e in E])
end

_sv_nan_to_missing(v) = isnan(v) ? missing : v

# ---------------------------------------------------------------------------------
# Treatment validation and public entry points
# ---------------------------------------------------------------------------------

function _sv_binary_value(v, context)
    ismissing(v) && return missing
    (v isa Bool || (v isa Real && (v == 0 || v == 1))) ||
        throw(ArgumentError("$context: treatment must be binary (0/1 or Bool); got " *
                            "$(repr(v))"))
    return Float64(v)
end

"""
    compute_exposure(s::InterferenceStructure, z::AbstractVector, spec) -> DataFrame
    compute_exposure(data, treatment, s, spec; unit, time=nothing) -> DataFrame

Evaluate the exposure specification `spec` ([`NeighborExposure`](@ref),
[`RingExposure`](@ref), [`HopExposure`](@ref) or [`CustomExposure`](@ref)) for a
treatment vector or for the treatments recorded in a table.

The vector method takes `z` in the order of [`structure_units`](@ref)`(s)` and
returns a `DataFrame` with a `:unit` column and one column per exposure
([`exposure_columns`](@ref)). The table method takes one row per unit
(cross-section, `time = nothing`) or per unit and period (panel); units are matched
to `s` by the `unit` column, and the set of units in `data` must equal the set in
`s`, because every unit's treatment enters its neighbours' exposures. Panel
exposures are time-varying: ``E_{it}`` depends on the period-``t`` treatments of
the other units. The result is aligned with the rows of `data`.

An exposure is `missing` when it is undefined (isolates, empty bands for `:share`)
or when it depends on a treatment that is missing or on a unit absent from that
period; it is never silently set to zero. Tabulating the exposures before
estimation shows how many units populate each exposure cell, which governs the
precision of the corresponding effects.

# Arguments
- `s::InterferenceStructure`, `z::AbstractVector`: structure and binary treatment
  vector (vector method; `missing` entries allowed).
- `data`: table with one row per unit, or per unit and period (table method).
- `treatment::Symbol`: binary treatment column (table method).
- `spec::ExposureSpec`: the exposure specification.

# Keywords
- `unit::Symbol`: unit-identifier column (table method).
- `time`: period column for panels, or `nothing` (default) for a cross-section.

# Returns
- `DataFrame`: `unit` (and `time`) columns followed by the exposure columns.

# Examples
```julia
using DrSnow, DataFrames
g = NetworkStructure(1:4, [0 1 1 0; 1 0 0 0; 1 0 0 1; 0 0 1 0])
people = DataFrame(id=1:4, treated=[1, 0, 0, 0])
compute_exposure(people, :treated, g, NeighborExposure(:share); unit=:id)
panel = DataFrame(id=repeat(1:4; inner=2), year=repeat([1, 2]; outer=4),
                  d=[0, 1, 0, 0, 0, 0, 0, 1])
compute_exposure(panel, :d, g, NeighborExposure(:any); unit=:id, time=:year)
```
"""
function compute_exposure(s::InterferenceStructure, z::AbstractVector,
                          spec::ExposureSpec)
    n = n_units(s)
    length(z) == n || throw(DimensionMismatch("compute_exposure: z must have one entry " *
                                              "per structure unit ($n)"))
    zz = [_sv_binary_value(v, "compute_exposure") for v in z]
    O = reshape(.!ismissing.(zz), :, 1)
    Z = reshape(Float64[coalesce(v, 0.0) for v in zz], :, 1)
    p = _sv_prepare(s, spec)
    E = _sv_eval(p, Z, all(O) ? nothing : O)
    out = DataFrame(:unit => copy(s.ids))
    for (name, e) in zip(p.names, E)
        out[!, name] = _sv_nan_to_missing.(vec(e))
    end
    return out
end

function compute_exposure(data, treatment::Symbol, s::InterferenceStructure,
                          spec::ExposureSpec; unit::Symbol,
                          time::Union{Nothing,Symbol}=nothing)
    require_columns(data, [treatment, unit, time]; context="compute_exposure")
    df = DataFrame(data; copycols=false)
    p = _sv_prepare(s, spec)
    rows, E = _sv_panel_exposure(df, treatment, s, p; unit=unit, time=time,
                                 context="compute_exposure")
    out = time === nothing ? DataFrame(unit => df[!, unit]) :
          DataFrame(unit => df[!, unit], time => df[!, time])
    for (name, e) in zip(p.names, E)
        out[!, name] = [_sv_nan_to_missing(e[u, t]) for (u, t) in rows]
    end
    return out
end

# Build the N × T treatment matrix from a (possibly panel) table and evaluate the
# prepared exposure. Returns the (unit index, period index) of every row and the
# exposure matrices.
function _sv_panel_exposure(df::AbstractDataFrame, treatment::Symbol,
                            s::InterferenceStructure, p::_SVPrepared; unit::Symbol,
                            time::Union{Nothing,Symbol}, context::AbstractString)
    uidx = _sv_require_units(s, df[!, unit], context)
    if time === nothing
        tidx = ones(Int, nrow(df))
        T = 1
    else
        tcol = df[!, time]
        any(ismissing, tcol) &&
            throw(ArgumentError("$context: `$time` has missing values"))
        periods = sort(unique(tcol))
        tpos = Dict(t => k for (k, t) in enumerate(periods))
        tidx = [tpos[t] for t in tcol]
        T = length(periods)
    end
    N = n_units(s)
    Z = zeros(N, T)
    O = falses(N, T)
    seen = falses(N, T)
    for (r, (u, t)) in enumerate(zip(uidx, tidx))
        if seen[u, t]
            what = time === nothing ? "unit $(repr(s.ids[u]))" :
                   "unit $(repr(s.ids[u])) in period $(repr(df[r, time]))"
            throw(ArgumentError("$context: duplicated row for $what"))
        end
        seen[u, t] = true
        v = _sv_binary_value(df[r, treatment], context)
        if !ismissing(v)
            Z[u, t] = v
            O[u, t] = true
        end
    end
    E = _sv_eval(p, Z, all(O) ? nothing : Matrix(O))
    return collect(zip(uidx, tidx)), E
end
