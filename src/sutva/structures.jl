# Interference structures: unit-keyed spatial, network and partition structures.
#
# Every structure stores the unit identifiers once (`ids`) together with an
# `id => index` dictionary. All data are matched to structures by identifier, never by
# row position; `_sv_rows_to_index` and `_sv_require_units` perform (and validate)
# that matching. Internally, units are indexed 1:N in the order of `ids`.

"""
    InterferenceStructure

Abstract supertype of the objects that describe *who can affect whom* when the
stable unit treatment value assumption (SUTVA) fails because of interference between
units.

Without interference, unit ``i``'s potential outcome depends only on its own
treatment, ``Y_i(z) = Y_i(z_i)``. With interference it may depend on the whole
assignment vector, ``Y_i(z)``, and causal effects can only be defined and estimated
after restricting that dependence (Hudgens and Halloran 2008; Manski 2013; Aronow and
Samii 2017). An interference structure provides the geometry used for these
restrictions: [`SpatialStructure`](@ref) (point locations and distances),
[`NetworkStructure`](@ref) (a directed or undirected, possibly weighted graph) and
[`PartitionStructure`](@ref) (groups, i.e. partial interference). Exposure
specifications ([`ExposureSpec`](@ref)) then summarize the treatments of a unit's
neighbours into its exposure.

The structure encodes a maintained assumption, not a finding: spillovers that
operate outside it (beyond the radius, along unrecorded ties) are ignored by
estimators that rely on it. Structures are keyed by unit identifiers, and data are
always matched to them by identifier, never by row position.

# Accessors
- [`structure_units`](@ref), [`n_units`](@ref), [`neighbor_matrix`](@ref); for
  spatial structures [`pairwise_distances`](@ref); for networks
  [`shortest_path_hops`](@ref).

# Examples
```julia
using DrSnow
s = SpatialStructure(["a", "b", "c"]; x=[0.0, 1.0, 5.0], y=[0.0, 0.0, 0.0])
s isa InterferenceStructure        # true
```

# References
- Hudgens, M. G., & Halloran, M. E. (2008). Toward causal inference with
  interference. *Journal of the American Statistical Association*, 103(482),
  832–842.
- Manski, C. F. (2013). Identification of treatment response with social
  interactions. *Econometrics Journal*, 16(1), S1–S23.
- Aronow, P. M., & Samii, C. (2017). Estimating average causal effects under general
  interference, with application to a social network experiment. *Annals of Applied
  Statistics*, 11(4), 1912–1947.
"""
abstract type InterferenceStructure end

const _SV_EARTH_RADIUS = Dict(:km => 6371.0, :mi => 3958.7613)

function _sv_build_index(ids::AbstractVector, context::AbstractString)
    isempty(ids) && throw(ArgumentError("$context: no units supplied"))
    any(ismissing, ids) && throw(ArgumentError("$context: unit identifiers contain " *
                                               "missing values"))
    idv = collect(ids)
    index = Dict{eltype(idv),Int}()
    for (i, id) in enumerate(idv)
        haskey(index, id) && throw(ArgumentError("$context: duplicated unit identifier " *
                                                 "`$(repr(id))`"))
        index[id] = i
    end
    return idv, index
end

"""
    SpatialStructure(ids; lat, lon, units=:km, earth_radius=nothing)
    SpatialStructure(ids; x, y)
    SpatialStructure(data, id; lat=:lat, lon=:lon, units=:km, earth_radius=nothing)
    SpatialStructure(data, id; x=:x, y=:y)

Unit-keyed point locations, used to define spatial neighbourhoods, distance-band
exposures ([`RingExposure`](@ref)) and spatial HAC variances ([`ConleyVcov`](@ref)).

With `lat` / `lon` (decimal degrees) distances are great-circle (haversine)
distances on a sphere, in kilometres or miles; with `x` / `y` they are Euclidean in
the units of the coordinates (projected coordinates). Coordinates are always passed
*by name*, so latitude and longitude cannot be silently swapped, and latitudes
outside ``[-90, 90]`` or longitudes outside ``[-180, 360]`` raise an error. The
haversine distance treats the Earth as a sphere; its error relative to ellipsoidal
geodesic distance is small (well under 1%) and immaterial for choosing spillover
radii, but the radius constant differs across software, which matters when
reproducing numbers exactly.

The table methods take one row per unit, or a panel in which each unit's coordinates
are constant over rows (this is checked).

# Arguments
- `ids::AbstractVector`: unique unit identifiers (vector methods).
- `data`, `id::Symbol`: a table and its unit-identifier column (table methods).

# Keywords
- `lat`, `lon` or `x`, `y`: coordinate vectors (vector methods) or column names
  (table methods; default `:lat` / `:lon` when none is given).
- `units::Symbol`: `:km` (default) or `:mi` for haversine distances.
- `earth_radius`: sphere radius in `units`; default the mean Earth radius
  (6371.0 km or 3958.76 mi). Other software uses other constants (fixest uses
  6376 km).

# Returns
- `SpatialStructure`.

# Examples
```julia
using DrSnow
s = SpatialStructure(["a", "b", "c"]; lat=[45.0, 45.1, 46.0], lon=[9.0, 9.2, 9.1])
pairwise_distances(s)                       # kilometres
p = SpatialStructure(1:3; x=[0.0, 3.0, 0.0], y=[0.0, 4.0, 1.0])
pairwise_distances(p)[1, 2]                  # 5.0
```
"""
struct SpatialStructure{T} <: InterferenceStructure
    ids::Vector{T}
    index::Dict{T,Int}
    coords::Matrix{Float64}        # N × 2: (lat, lon) in degrees or (x, y)
    metric::Symbol                 # :haversine or :euclidean
    units::Symbol                  # :km, :mi or :coordinate
    radius::Float64                # sphere radius for haversine (NaN otherwise)
end

function SpatialStructure(ids::AbstractVector; lat=nothing, lon=nothing, x=nothing,
                          y=nothing, units::Symbol=:km, earth_radius=nothing)
    geo = lat !== nothing || lon !== nothing
    cart = x !== nothing || y !== nothing
    geo && cart && throw(ArgumentError("SpatialStructure: pass either `lat`/`lon` or " *
                                       "`x`/`y`, not both"))
    geo || cart || throw(ArgumentError("SpatialStructure: coordinates are required: " *
                                       "pass `lat` and `lon`, or `x` and `y`"))
    idv, index = _sv_build_index(ids, "SpatialStructure")
    n = length(idv)
    a, b = geo ? (lat, lon) : (x, y)
    (a === nothing || b === nothing) &&
        throw(ArgumentError("SpatialStructure: both coordinates must be supplied"))
    (length(a) == n && length(b) == n) ||
        throw(DimensionMismatch("SpatialStructure: coordinate vectors must have one " *
                                "entry per unit ($n)"))
    (any(ismissing, a) || any(ismissing, b)) &&
        throw(ArgumentError("SpatialStructure: coordinates contain missing values"))
    coords = hcat(Float64.(collect(a)), Float64.(collect(b)))
    all(isfinite, coords) || throw(ArgumentError("SpatialStructure: coordinates must " *
                                                 "be finite"))
    if geo
        units in (:km, :mi) || throw(ArgumentError("units must be :km or :mi"))
        if any(v -> !(-90 <= v <= 90), coords[:, 1])
            throw(ArgumentError("SpatialStructure: latitude outside [-90, 90]; check " *
                                "that `lat` and `lon` are not swapped"))
        end
        if any(v -> !(-180 <= v <= 360), coords[:, 2])
            throw(ArgumentError("SpatialStructure: longitude outside [-180, 360]"))
        end
        R = earth_radius === nothing ? _SV_EARTH_RADIUS[units] : Float64(earth_radius)
        R > 0 || throw(ArgumentError("earth_radius must be positive"))
        return SpatialStructure(idv, index, coords, :haversine, units, R)
    end
    earth_radius === nothing ||
        throw(ArgumentError("SpatialStructure: earth_radius applies to lat/lon only"))
    return SpatialStructure(idv, index, coords, :euclidean, :coordinate, NaN)
end

function _sv_unit_table(data, id::Symbol, cols::Vector{Symbol}, context)
    require_columns(data, [id; cols]; context=context)
    df = DataFrame(data)
    ids = df[!, id]
    any(ismissing, ids) && throw(ArgumentError("$context: `$id` contains missing values"))
    first_row = Dict{Any,Int}()
    order = Int[]
    for (r, u) in enumerate(ids)
        if !haskey(first_row, u)
            first_row[u] = r
            push!(order, r)
        else
            r0 = first_row[u]
            for c in cols
                isequal(df[r, c], df[r0, c]) ||
                    throw(ArgumentError("$context: column `$c` varies within unit " *
                                        "`$(repr(u))`"))
            end
        end
    end
    return df[order, :]
end

function SpatialStructure(data, id::Symbol; lat::Union{Nothing,Symbol}=nothing,
                          lon::Union{Nothing,Symbol}=nothing,
                          x::Union{Nothing,Symbol}=nothing,
                          y::Union{Nothing,Symbol}=nothing, units::Symbol=:km,
                          earth_radius=nothing)
    if lat === nothing && lon === nothing && x === nothing && y === nothing
        lat, lon = :lat, :lon
    end
    cols = Symbol[c for c in (lat, lon, x, y) if c !== nothing]
    u = _sv_unit_table(data, id, cols, "SpatialStructure")
    col(c) = c === nothing ? nothing : u[!, c]
    return SpatialStructure(u[!, id]; lat=col(lat), lon=col(lon), x=col(x), y=col(y),
                            units=units, earth_radius=earth_radius)
end

"""
    NetworkStructure(ids, A; directed=false)
    NetworkStructure(ids, edges; source=:source, target=:target, weight=nothing,
                     directed=false)

Unit-keyed, possibly directed and weighted network through which treatments can
spill over.

The convention is that rows of the adjacency matrix are *receivers*: ``A_{ij} ≠ 0``
means that the treatment of unit `ids[j]` can affect the outcome of unit `ids[i]`,
with weight ``A_{ij}``. In an edge table, each row `source → target` means that the
treatment of `source` can affect `target`; edges are matched to units by
identifier, so the table can be in any order. Undirected networks must have a
symmetric adjacency matrix (edges given as a table are symmetrized). Self-loops,
negative weights and duplicated edges are rejected. Units without neighbours
(isolates) are allowed and must be listed in `ids`; by default they receive a
`missing` exposure, because "no treated neighbour" is not a meaningful exposure level
for a unit that has no neighbour.

The recorded network is itself an assumption. Missing or mismeasured ties make the
exposure mapping misspecified; see [`ExposureMapping`](@ref) for how estimands are
then interpreted.

# Arguments
- `ids::AbstractVector`: unique unit identifiers, including isolates.
- `A::AbstractMatrix`: ``N × N`` adjacency matrix in the order of `ids`.
- `edges`: alternatively, a table with one row per edge, whose identifiers all
  appear in `ids`.

# Keywords
- `source`, `target::Symbol`: edge-table column names.
- `weight::Union{Nothing,Symbol}`: edge-weight column; `nothing` gives unit weights.
- `directed::Bool`: whether influence is directional; default `false`.

# Returns
- `NetworkStructure`.

# Examples
```julia
using DrSnow, DataFrames
edges = DataFrame(source=["a", "b"], target=["b", "c"])
g = NetworkStructure(["a", "b", "c", "d"], edges)      # "d" is an isolate
neighbor_matrix(g)
```
"""
struct NetworkStructure{T} <: InterferenceStructure
    ids::Vector{T}
    index::Dict{T,Int}
    A::SparseMatrixCSC{Float64,Int}   # A[i, j]: influence of j on i
    At::SparseMatrixCSC{Float64,Int}  # transpose: column i lists the sources of i
    directed::Bool
    weighted::Bool
end

function NetworkStructure(ids::AbstractVector, A::AbstractMatrix; directed::Bool=false)
    idv, index = _sv_build_index(ids, "NetworkStructure")
    n = length(idv)
    size(A) == (n, n) || throw(DimensionMismatch("NetworkStructure: adjacency matrix " *
                                                 "must be $n × $n"))
    S = SparseMatrixCSC{Float64,Int}(sparse(Float64.(A)))
    dropzeros!(S)
    all(isfinite, nonzeros(S)) || throw(ArgumentError("NetworkStructure: non-finite " *
                                                      "edge weights"))
    any(<(0), nonzeros(S)) && throw(ArgumentError("NetworkStructure: negative edge " *
                                                  "weights are not supported"))
    any(i -> S[i, i] != 0, 1:n) &&
        throw(ArgumentError("NetworkStructure: self-loops (non-zero diagonal) are not " *
                            "allowed"))
    if !directed && !issymmetric(S)
        throw(ArgumentError("NetworkStructure: adjacency matrix is not symmetric; pass " *
                            "`directed=true` for a directed network"))
    end
    weighted = any(!=(1.0), nonzeros(S))
    return NetworkStructure(idv, index, S, SparseMatrixCSC(transpose(S)), directed,
                            weighted)
end

function NetworkStructure(ids::AbstractVector, edges; source::Symbol=:source,
                          target::Symbol=:target, weight::Union{Nothing,Symbol}=nothing,
                          directed::Bool=false)
    Tables.istable(edges) || throw(ArgumentError("NetworkStructure: `edges` must be a " *
                                                 "table or an adjacency matrix"))
    require_columns(edges, [source, target, weight]; context="NetworkStructure")
    idv, index = _sv_build_index(ids, "NetworkStructure")
    e = DataFrame(edges)
    n = length(idv)
    src = Int[]
    dst = Int[]
    for (s, t) in zip(e[!, source], e[!, target])
        (haskey(index, s) && haskey(index, t)) ||
            throw(ArgumentError("NetworkStructure: edge $(repr(s)) → $(repr(t)) " *
                                "refers to a unit not listed in `ids`"))
        s == t && throw(ArgumentError("NetworkStructure: self-loop on unit $(repr(s))"))
        push!(src, index[s])
        push!(dst, index[t])
    end
    w = weight === nothing ? ones(length(src)) : Float64.(collect(e[!, weight]))
    # A[target, source] = w; duplicated edges are an error rather than summed silently.
    seen = Set{Tuple{Int,Int}}()
    for (s, t) in zip(src, dst)
        key = directed ? (t, s) : minmax(s, t)
        key in seen && throw(ArgumentError("NetworkStructure: duplicated edge between " *
                                           "$(repr(idv[s])) and $(repr(idv[t]))"))
        push!(seen, key)
    end
    I = directed ? dst : vcat(dst, src)
    J = directed ? src : vcat(src, dst)
    W = directed ? w : vcat(w, w)
    return NetworkStructure(idv, sparse(I, J, W, n, n); directed=directed)
end

"""
    PartitionStructure(ids, groups)
    PartitionStructure(data, id; group)

Partial interference: units interact only within their group (household, village,
school, market) and never across groups.

Partial interference (Hudgens and Halloran 2008) allows arbitrary spillovers within
groups while ruling them out across groups, which makes
two-stage randomized designs ([`TwoStageRandomization`](@ref)) and group-level
inference possible. A unit's neighbours are the other members of its group.

# Arguments
- `ids::AbstractVector`: unique unit identifiers.
- `groups::AbstractVector`: their group labels, one per unit.
- `data`, `id::Symbol`: alternatively, a table and its unit column.

# Keywords
- `group::Symbol`: group column of the table method; it must be constant within
  unit.

# Returns
- `PartitionStructure`.

# Examples
```julia
using DrSnow
p = PartitionStructure(1:6, [1, 1, 1, 2, 2, 2])
neighbor_matrix(p)
```

# References
- Hudgens, M. G., & Halloran, M. E. (2008). Toward causal inference with
  interference. *Journal of the American Statistical Association*, 103(482),
  832–842.
- Tchetgen Tchetgen, E. J., & VanderWeele, T. J. (2012). On causal inference in the
  presence of interference. *Statistical Methods in Medical Research*, 21(1),
  55–75.
"""
struct PartitionStructure{T,G} <: InterferenceStructure
    ids::Vector{T}
    index::Dict{T,Int}
    groups::Vector{G}
    members::Vector{Vector{Int}}   # member indices per group label
    group_of::Vector{Int}          # group position of each unit
end

function PartitionStructure(ids::AbstractVector, groups::AbstractVector)
    idv, index = _sv_build_index(ids, "PartitionStructure")
    length(groups) == length(idv) ||
        throw(DimensionMismatch("PartitionStructure: one group label per unit required"))
    any(ismissing, groups) && throw(ArgumentError("PartitionStructure: group labels " *
                                                  "contain missing values"))
    labels = unique(groups)
    pos = Dict(l => k for (k, l) in enumerate(labels))
    group_of = [pos[g] for g in groups]
    members = [Int[] for _ in labels]
    for (i, g) in enumerate(group_of)
        push!(members[g], i)
    end
    return PartitionStructure(idv, index, collect(labels), members, group_of)
end

function PartitionStructure(data, id::Symbol; group::Symbol)
    u = _sv_unit_table(data, id, [group], "PartitionStructure")
    return PartitionStructure(u[!, id], u[!, group])
end

"""
    structure_units(s::InterferenceStructure) -> Vector

Unit identifiers of `s`, in the internal order used by the matrices returned by
[`pairwise_distances`](@ref), [`neighbor_matrix`](@ref) and
[`shortest_path_hops`](@ref), and by [`compute_exposure`](@ref) when it is given a
treatment vector.

Assignment mechanisms used with `s` (for example
`CompleteRandomization(n_units(s), k)`) index units in this order, so a design built
for a structure must list units in the order of `structure_units(s)`.

# Arguments
- `s::InterferenceStructure`: the structure.

# Returns
- `Vector` of unit identifiers (the stored vector; do not mutate it).

# Examples
```julia
using DrSnow
g = NetworkStructure(["a", "b"], [0 1; 1 0])
structure_units(g)    # ["a", "b"]
```
"""
structure_units(s::InterferenceStructure) = s.ids

n_units(s::InterferenceStructure) = length(s.ids)

function Base.show(io::IO, s::SpatialStructure)
    kind = s.metric === :haversine ? "haversine, $(s.units)" : "euclidean"
    print(io, "SpatialStructure($(length(s.ids)) units, $kind)")
end
function Base.show(io::IO, s::NetworkStructure)
    print(io, "NetworkStructure($(length(s.ids)) units, $(nnz(s.A)) directed arcs, ",
          s.directed ? "directed" : "undirected", s.weighted ? ", weighted" : "", ")")
end
Base.show(io::IO, s::PartitionStructure) =
    print(io, "PartitionStructure($(length(s.ids)) units, $(length(s.members)) groups)")

# ---------------------------------------------------------------------------------
# Matching data to structures by key
# ---------------------------------------------------------------------------------

# Structure index of every value in `idcol`; errors on identifiers unknown to `s`.
function _sv_rows_to_index(s::InterferenceStructure, idcol::AbstractVector, context)
    out = Vector{Int}(undef, length(idcol))
    for (r, u) in enumerate(idcol)
        ismissing(u) && throw(ArgumentError("$context: unit identifier is missing in " *
                                            "row $r"))
        k = get(s.index, u, 0)
        k == 0 && throw(ArgumentError("$context: unit $(repr(u)) (row $r) is not in the " *
                                      "interference structure"))
        out[r] = k
    end
    return out
end

# Rows → structure index, additionally requiring every structure unit to appear.
function _sv_require_units(s::InterferenceStructure, idcol::AbstractVector, context)
    idx = _sv_rows_to_index(s, idcol, context)
    present = falses(length(s.ids))
    present[idx] .= true
    if !all(present)
        miss = s.ids[.!present]
        throw(ArgumentError("$context: $(length(miss)) unit(s) of the interference " *
                            "structure are absent from the data (e.g. " *
                            "$(repr(first(miss)))); exposures depend on every unit's " *
                            "treatment, so the unit sets must coincide"))
    end
    return idx
end

# ---------------------------------------------------------------------------------
# Distances
# ---------------------------------------------------------------------------------

@inline function _sv_haversine(lat1, lon1, lat2, lon2, R)
    p1 = deg2rad(lat1)
    p2 = deg2rad(lat2)
    dp = p2 - p1
    dl = deg2rad(lon2 - lon1)
    a = sin(dp / 2)^2 + cos(p1) * cos(p2) * sin(dl / 2)^2
    return 2R * asin(sqrt(min(1.0, a)))
end

@inline function _sv_distance(s::SpatialStructure, i::Int, j::Int)
    c = s.coords
    if s.metric === :haversine
        return _sv_haversine(c[i, 1], c[i, 2], c[j, 1], c[j, 2], s.radius)
    end
    return hypot(c[i, 1] - c[j, 1], c[i, 2] - c[j, 2])
end

"""
    pairwise_distances(s::SpatialStructure) -> Matrix{Float64}

Dense ``N × N`` matrix of distances between the units of `s`, in the order of
[`structure_units`](@ref)`(s)`: haversine distances in `s.units` (kilometres or
miles) for latitude/longitude structures, Euclidean distances otherwise.

The dense matrix needs ``N^2`` memory; for large ``N`` prefer
[`neighbor_matrix`](@ref) with a radius, which is sparse.

# Arguments
- `s::SpatialStructure`: the spatial structure.

# Returns
- `Matrix{Float64}`: symmetric, with zero diagonal.

# Examples
```julia
using DrSnow
D = pairwise_distances(SpatialStructure(1:3; x=[0, 3, 0], y=[0, 4, 1]))
D[1, 2]   # 5.0
```
"""
function pairwise_distances(s::SpatialStructure)
    n = length(s.ids)
    D = zeros(n, n)
    for j in 1:n, i in (j + 1):n
        d = _sv_distance(s, i, j)
        D[i, j] = d
        D[j, i] = d
    end
    return D
end

# All ordered pairs (i ≠ j) within distance `r`, as parallel vectors (zero distances
# are kept, unlike in a sparse matrix of distances).
function _sv_pairs_within(s::SpatialStructure, r::Real)
    r >= 0 || throw(ArgumentError("radius must be non-negative"))
    n = length(s.ids)
    I = Int[]
    J = Int[]
    D = Float64[]
    for j in 1:n, i in (j + 1):n
        d = _sv_distance(s, i, j)
        if d <= r
            push!(I, i, j)
            push!(J, j, i)
            push!(D, d, d)
        end
    end
    return I, J, D
end

# ---------------------------------------------------------------------------------
# Neighbourhoods and shortest paths
# ---------------------------------------------------------------------------------

"""
    neighbor_matrix(s; radius=nothing, weighted=false) -> SparseMatrixCSC{Float64,Int}

Sparse ``N × N`` matrix (order of [`structure_units`](@ref)) whose row ``i`` marks
the units whose treatment can affect unit ``i``:

- `NetworkStructure`: the adjacency matrix (its edge weights if `weighted=true`);
- `SpatialStructure`: the other units within distance `radius` (required);
- `PartitionStructure`: the other members of ``i``'s group.

Entries are 1 unless `weighted=true`. The matrix is the building block of neighbour
exposures (``W z`` counts treated neighbours) and the default weight matrix of
[`treatment_moran_test`](@ref).

# Arguments
- `s::InterferenceStructure`: the structure.

# Keywords
- `radius`: neighbourhood radius, in the distance units of the structure (spatial
  structures only, required there).
- `weighted::Bool`: return edge weights instead of indicators (networks only);
  default `false`.

# Returns
- `SparseMatrixCSC{Float64,Int}`.

# Examples
```julia
using DrSnow
s = SpatialStructure(1:4; x=[0.0, 1.0, 2.5, 10.0], y=zeros(4))
neighbor_matrix(s; radius=2.0)          # unit 4 has no neighbour
```
"""
function neighbor_matrix(s::NetworkStructure; radius=nothing, weighted::Bool=false)
    radius === nothing || throw(ArgumentError("`radius` applies to spatial structures"))
    weighted && return copy(s.A)
    B = copy(s.A)
    fill!(nonzeros(B), 1.0)
    return B
end

function neighbor_matrix(s::SpatialStructure; radius=nothing, weighted::Bool=false)
    radius === nothing && throw(ArgumentError("neighbor_matrix: spatial structures " *
                                              "need a `radius`"))
    weighted && throw(ArgumentError("neighbor_matrix: `weighted` applies to networks"))
    I, J, _ = _sv_pairs_within(s, radius)
    n = length(s.ids)
    return sparse(I, J, ones(length(I)), n, n)
end

function neighbor_matrix(s::PartitionStructure; radius=nothing, weighted::Bool=false)
    (radius === nothing && !weighted) ||
        throw(ArgumentError("neighbor_matrix: partitions take no radius/weights"))
    I = Int[]
    J = Int[]
    for m in s.members, i in m, j in m
        i == j && continue
        push!(I, i)
        push!(J, j)
    end
    n = length(s.ids)
    return sparse(I, J, ones(length(I)), n, n)
end

"""
    shortest_path_hops(s::NetworkStructure; max_hops, direction=:influence)
        -> SparseMatrixCSC{Int,Int}

Shortest-path (hop) distances between units of a network, up to `max_hops`, computed
by breadth-first search.

Entry ``(i, j)`` is the length of the shortest path along which the treatment of
``j`` reaches ``i`` (`direction = :influence`), or of the shortest path ignoring
edge directions (`direction = :undirected`); pairs farther apart than `max_hops`, or
unreachable, are structural zeros, and the diagonal is empty. Unlike powers of the
adjacency matrix, which count walks, this assigns every pair to a single distance, so
a unit's second-order neighbours do not include its first-order neighbours. Hop
distances define [`HopExposure`](@ref) and the kernel of
[`NetworkHACVcov`](@ref).

# Arguments
- `s::NetworkStructure`: the network.

# Keywords
- `max_hops::Integer`: largest distance computed (at least 1).
- `direction::Symbol`: `:influence` (default) or `:undirected`.

# Returns
- `SparseMatrixCSC{Int,Int}` of hop distances.

# Examples
```julia
using DrSnow
g = NetworkStructure(1:4, [0 1 0 0; 1 0 1 0; 0 1 0 1; 0 0 1 0])   # a path
H = shortest_path_hops(g; max_hops=2)
H[1, 3]                       # 2
count(==(2), H)               # ordered pairs exactly two hops apart: 4
```
"""
function shortest_path_hops(s::NetworkStructure; max_hops::Integer,
                            direction::Symbol=:influence)
    max_hops >= 1 || throw(ArgumentError("max_hops must be ≥ 1"))
    direction in (:influence, :undirected) ||
        throw(ArgumentError("direction must be :influence or :undirected"))
    G = direction === :undirected && s.directed ? s.At + s.A : s.At
    n = length(s.ids)
    rows = rowvals(G)
    I = Int[]
    J = Int[]
    V = Int[]
    dist = fill(-1, n)
    frontier = Int[]
    next = Int[]
    touched = Int[]
    for i in 1:n
        dist[i] = 0
        push!(touched, i)
        empty!(frontier)
        push!(frontier, i)
        for h in 1:max_hops
            empty!(next)
            for u in frontier, k in nzrange(G, u)
                v = rows[k]           # v is a source of u (can influence u)
                if dist[v] < 0
                    dist[v] = h
                    push!(touched, v)
                    push!(next, v)
                    push!(I, i)
                    push!(J, v)
                    push!(V, h)
                end
            end
            isempty(next) && break
            frontier, next = next, frontier
        end
        for v in touched
            dist[v] = -1
        end
        empty!(touched)
    end
    return sparse(I, J, V, n, n)
end
