# Design-based estimation under interference (Aronow & Samii 2017).
#
# An exposure mapping assigns each unit a discrete exposure condition as a function of
# the whole assignment vector. The known assignment mechanism then determines the
# probability π_i(k) that unit i is in condition k, and the joint probabilities
# π_ij(k, l) needed for variance estimation. Probabilities are computed by exact
# enumeration of the design's support (small designs) or by Monte Carlo; both are
# stored as a matrix of condition codes (units × assignments) plus assignment weights,
# from which marginal and joint probabilities are obtained by (blocked) products.

"""
    ExposureMapping(spec=NeighborExposure(:any); cutpoints=nothing, direct=true)
    ExposureMapping(f)

Discrete exposure mapping: a rule that assigns every unit an *exposure condition*
(a label such as `"treated_exposed"`) as a function of the whole assignment vector,
used by the design-based estimators [`exposure_probabilities`](@ref) and
[`exposure_effects`](@ref).

Following Aronow and Samii (2017), the mapping ``D_i = f_i(z)`` partitions the
assignments into conditions, and the potential outcome of unit ``i`` in condition
``k`` is ``Y_i(k)``; this presumes that the mapping is *correctly specified*, i.e.
that ``Y_i(z)`` depends on ``z`` only through ``f_i(z)``. With the default
specification and `direct = true` the four conditions are the Aronow–Samii
conditions: own treatment crossed with having at least one treated neighbour.

Correct specification is rarely credible in full (spillovers beyond the recorded
ties, or of higher order). Sävje (2024) shows that the mapping can instead be read as
a *definition* of the effect: when it is misspecified, the same estimators target
the expected exposure effect
``τ(a, b) = N^{-1} \\sum_i [\\bar y_i(a) - \\bar y_i(b)]``, with
``\\bar y_i(d) = E\\{Y_i(Z) \\mid D_i = d\\}`` the average potential outcome over the
assignments of the actual design that put unit ``i`` in condition ``d``. This
estimand depends on the design; the Horvitz–Thompson estimator with exact
probabilities remains unbiased for it, and consistency holds when the dependence
between the units' specification errors is limited; Leung (2022) gives related
results under approximate neighbourhood interference. Variance estimation is more
delicate under misspecification: the conservative variance estimator is then not
guaranteed to be conservative (Sävje 2024). Report the mapping as part of the
estimand.

# Arguments
- `spec::ExposureSpec`: exposure specification to discretize; default
  `NeighborExposure(:any)`.
- `f`: alternatively, a function `f(structure, z::BitVector)` returning one label
  (`String`) or `missing` per unit.

# Keywords
- `cutpoints`: `nothing` (default), in which case the exposure columns must be
  binary: a single column gives `"exposed"` / `"unexposed"`, several columns (rings,
  hops) give the exposed columns joined by `+` (e.g. `"hop1+hop2"`) or
  `"unexposed"`. With `cutpoints = [c₁, …, c_m]` a single numeric exposure is binned
  into ``(-∞, c_1], (c_1, c_2], …, (c_m, ∞)``.
- `direct::Bool`: prefix the unit's own treatment (`"treated_…"` / `"control_…"`);
  default `true`.

Units whose exposure is undefined (for example network isolates under
`isolates = :missing`) get a `missing` condition and are excluded from the estimand
population.

# Returns
- `ExposureMapping`.

# Examples
```julia
using DrSnow
g = NetworkStructure(1:5, [0 1 0 0 0; 1 0 1 0 0; 0 1 0 1 0; 0 0 1 0 1; 0 0 0 1 0])
z = [true, false, false, true, false]
exposure_conditions(g, z, ExposureMapping())             # 4 Aronow–Samii conditions
m2 = ExposureMapping(NeighborExposure(:share); cutpoints=[0.0, 0.5])
exposure_conditions(g, z, m2)
```

# References
- Aronow, P. M., & Samii, C. (2017). Estimating average causal effects under general
  interference, with application to a social network experiment. *Annals of Applied
  Statistics*, 11(4), 1912–1947.
- Sävje, F. (2024). Causal inference with misspecified exposure mappings: Separating
  definitions and assumptions. *Biometrika*, 111(1), 1–15.
- Leung, M. P. (2022). Causal inference under approximate neighborhood interference.
  *Econometrica*, 90(1), 267–293.
- Manski, C. F. (2013). Identification of treatment response with social
  interactions. *Econometrics Journal*, 16(1), S1–S23.
"""
struct ExposureMapping{S,F}
    spec::S
    cutpoints::Union{Nothing,Vector{Float64}}
    direct::Bool
    f::F
end

function ExposureMapping(spec::ExposureSpec=NeighborExposure(:any);
                         cutpoints=nothing, direct::Bool=true)
    cp = cutpoints === nothing ? nothing : Float64.(collect(cutpoints))
    if cp !== nothing
        (isempty(cp) || !issorted(cp) || !allunique(cp)) &&
            throw(ArgumentError("ExposureMapping: cutpoints must be strictly increasing"))
        length(exposure_columns(spec)) == 1 ||
            throw(ArgumentError("ExposureMapping: cutpoints require a single-column " *
                                "exposure specification"))
    end
    return ExposureMapping(spec, cp, direct, nothing)
end

ExposureMapping(f::Function) = ExposureMapping(nothing, nothing, false, f)

function _sv_fmt_interval(cp, b)
    b == 1 && return "exposure≤" * _sv_fmt(cp[1])
    b == length(cp) + 1 && return "exposure>" * _sv_fmt(cp[end])
    return _sv_fmt(cp[b - 1]) * "<exposure≤" * _sv_fmt(cp[b])
end

# Prepared mapping: returns a closure z::BitVector -> Vector{Union{Missing,String}}.
function _sv_prepare_mapping(s::InterferenceStructure, m::ExposureMapping)
    if m.f !== nothing
        return function (z)
            lab = m.f(s, z)
            length(lab) == length(z) ||
                throw(DimensionMismatch("ExposureMapping function must return one " *
                                        "label per unit"))
            return Union{Missing,String}[ismissing(x) ? missing : string(x) for x in lab]
        end
    end
    p = _sv_prepare(s, m.spec)
    names = p.names
    return function (z)
        E = _sv_eval_vector(p, z)
        n, K = size(E)
        out = Vector{Union{Missing,String}}(undef, n)
        for i in 1:n
            row = view(E, i, :)
            if any(isnan, row)
                out[i] = missing
                continue
            end
            lab = if m.cutpoints !== nothing
                _sv_fmt_interval(m.cutpoints, searchsortedfirst(m.cutpoints, row[1]))
            else
                all(v -> v == 0 || v == 1, row) ||
                    throw(ArgumentError("ExposureMapping: exposure `$(join(names, ","))`" *
                                        " is not binary; supply `cutpoints` or a " *
                                        "thresholded / :any specification"))
                on = [names[k] for k in 1:K if row[k] == 1]
                isempty(on) ? "unexposed" : K == 1 ? "exposed" : join(on, "+")
            end
            out[i] = m.direct ? (z[i] ? "treated_" : "control_") * lab : lab
        end
        return out
    end
end

"""
    exposure_conditions(s, z, mapping=ExposureMapping()) -> Vector{Union{Missing,String}}

Exposure condition of every unit of `s` under the assignment `z`, according to the
discrete exposure mapping `mapping`.

Tabulating the observed conditions shows how many units fall in each condition, and
hence which contrasts can be estimated with reasonable precision.

# Arguments
- `s::InterferenceStructure`: the structure.
- `z::AbstractVector`: binary assignment vector, in the order of
  [`structure_units`](@ref)`(s)`.
- `mapping::ExposureMapping`: the mapping; default `ExposureMapping()`.

# Returns
- `Vector{Union{Missing,String}}` in the order of `structure_units(s)`; `missing`
  where the condition is undefined.

# Examples
```julia
using DrSnow
g = NetworkStructure(1:4, [0 1 0 0; 1 0 1 0; 0 1 0 0; 0 0 0 0])
exposure_conditions(g, Bool[1, 0, 0, 1])     # unit 4 is an isolate: missing
```
"""
function exposure_conditions(s::InterferenceStructure, z::AbstractVector,
                             mapping::ExposureMapping=ExposureMapping())
    length(z) == n_units(s) || throw(DimensionMismatch("z must have one entry per unit"))
    zb = BitVector([_sv_binary_value(v, "exposure_conditions") == 1 for v in z])
    return _sv_prepare_mapping(s, mapping)(zb)
end

# ---------------------------------------------------------------------------------
# Enumeration of small designs
# ---------------------------------------------------------------------------------

function _sv_ncomb(n, k)
    (k < 0 || k > n) && return 0.0
    return Float64(binomial(big(n), big(k)))
end

# All k-subsets of 1:n, lexicographic, as a vector of index vectors.
function _sv_combinations(n::Int, k::Int)
    out = Vector{Vector{Int}}()
    k == 0 && return [Int[]]
    c = collect(1:k)
    while true
        push!(out, copy(c))
        i = k
        while i >= 1 && c[i] == n - k + i
            i -= 1
        end
        i == 0 && break
        c[i] += 1
        for j in (i + 1):k
            c[j] = c[j - 1] + 1
        end
    end
    return out
end

_sv_support_size(::AssignmentMechanism) = Inf
function _sv_support_size(m::BernoulliAssignment)
    return 2.0^count(p -> 0 < p < 1, m.p)
end
_sv_support_size(m::CompleteRandomization) = _sv_ncomb(m.n, m.n_treated)
_sv_support_size(m::StratifiedRandomization) =
    prod(_sv_ncomb(length(g), k) for (g, k) in zip(m.groups, m.n_treated))
_sv_support_size(m::ClusterRandomization) =
    _sv_ncomb(length(m.groups), m.n_treated_clusters)

# (Z, w): Z is n × R BitMatrix of all assignments in the support, w their probabilities.
function _sv_enumerate(m::BernoulliAssignment)
    n = length(m.p)
    free = findall(p -> 0 < p < 1, m.p)
    base = m.p .== 1
    R = 2^length(free)
    Z = falses(n, R)
    w = ones(R)
    for r in 1:R
        Z[:, r] .= base
        bits = r - 1
        for (b, i) in enumerate(free)
            on = (bits >> (b - 1)) & 1 == 1
            Z[i, r] = on
            w[r] *= on ? m.p[i] : 1 - m.p[i]
        end
    end
    return Z, w
end

function _sv_enumerate(m::CompleteRandomization)
    combs = _sv_combinations(m.n, m.n_treated)
    Z = falses(m.n, length(combs))
    for (r, c) in enumerate(combs)
        Z[c, r] .= true
    end
    return Z, fill(1 / length(combs), length(combs))
end

function _sv_enumerate(m::StratifiedRandomization)
    parts = [_sv_combinations(length(g), k) for (g, k) in zip(m.groups, m.n_treated)]
    R = prod(length.(parts))
    Z = falses(m.n, R)
    for (r, choice) in enumerate(Iterators.product(parts...))
        for (g, c) in zip(m.groups, choice)
            Z[g[c], r] .= true
        end
    end
    return Z, fill(1 / R, R)
end

function _sv_enumerate(m::ClusterRandomization)
    combs = _sv_combinations(length(m.groups), m.n_treated_clusters)
    Z = falses(m.n, length(combs))
    for (r, c) in enumerate(combs), g in c
        Z[m.groups[g], r] .= true
    end
    return Z, fill(1 / length(combs), length(combs))
end

_sv_design_name(m::AssignmentMechanism) = string(nameof(typeof(m)))

# Check that an observed assignment lies in the support of a known design.
_sv_check_support(::AssignmentMechanism, z) = nothing
function _sv_check_support(m::BernoulliAssignment, z)
    all(i -> !(m.p[i] == 0 && z[i]) && !(m.p[i] == 1 && !z[i]), eachindex(z)) ||
        throw(ArgumentError("observed assignment has probability zero under the design"))
end
function _sv_check_support(m::CompleteRandomization, z)
    count(z) == m.n_treated ||
        throw(ArgumentError("observed assignment has $(count(z)) treated units but the " *
                            "design treats $(m.n_treated)"))
end
function _sv_check_support(m::StratifiedRandomization, z)
    for (g, k) in zip(m.groups, m.n_treated)
        count(z[g]) == k || throw(ArgumentError("observed treated count in a stratum " *
                                                "differs from the design"))
    end
end
function _sv_check_support(m::ClusterRandomization, z)
    nt = 0
    for g in m.groups
        all(==(z[g[1]]), z[g]) ||
            throw(ArgumentError("observed assignment varies within a cluster"))
        nt += z[g[1]]
    end
    nt == m.n_treated_clusters ||
        throw(ArgumentError("observed number of treated clusters differs from the design"))
end

# ---------------------------------------------------------------------------------
# Exposure probabilities
# ---------------------------------------------------------------------------------

"""
    ExposureProbabilities

Probabilities of every exposure condition for every unit under an assignment
mechanism, produced by [`exposure_probabilities`](@ref) and consumed by
[`exposure_effects`](@ref).

The marginal probabilities ``π_i(k) = P(D_i = k)`` define the Horvitz–Thompson
weights, and the joint probabilities ``π_{ij}(k, l) = P(D_i = k, D_j = l)`` enter
the variance estimator. Both are computed from a matrix of condition codes (units by
assignments) and assignment weights: the full support of the design when
enumerated exactly, a set of simulated assignments (frequencies) under Monte Carlo,
or user-supplied assignments. Joint probability matrices are computed on demand and
cached (for up to 3000 units), so reusing the object for several outcomes is cheap.

# Fields
- `ids`: unit identifiers (order of [`structure_units`](@ref)).
- `levels::Vector{String}`: exposure conditions that occur under the design.
- `pi::Matrix{Float64}`: ``N × K`` marginal probabilities ``π_i(k)``.
- `codes::Matrix{UInt8}`: condition code of every unit (rows) in every enumerated or
  simulated assignment (columns); `0` means undefined.
- `weights::Vector{Float64}`: probability of each column (sums to one).
- `defined::BitVector`: units whose condition is defined.
- `method::Symbol`: `:exact`, `:monte_carlo` or `:supplied`.
- `design::String`, `mapping`: the design and exposure mapping used.

# Accessors
- [`exposure_positivity`](@ref)`(P)`.
"""
struct ExposureProbabilities{T,M}
    ids::Vector{T}
    levels::Vector{String}
    pi::Matrix{Float64}
    codes::Matrix{UInt8}
    weights::Vector{Float64}
    defined::BitVector
    method::Symbol
    design::String
    mapping::M
    joint_cache::Dict{Tuple{Int,Int},Matrix{Float64}}
end

function Base.show(io::IO, P::ExposureProbabilities)
    print(io, "ExposureProbabilities($(length(P.ids)) units, $(length(P.levels)) " *
              "conditions, $(P.method), $(length(P.weights)) assignments)")
end

function Base.show(io::IO, ::MIME"text/plain", P::ExposureProbabilities)
    println(io, "Exposure probabilities ($(P.method); design: $(P.design); ",
            length(P.weights), " assignments)")
    show(io, exposure_positivity(P))
end

"""
    exposure_probabilities(s, design; mapping=ExposureMapping(), method=:auto,
                           draws=10_000, max_exact=50_000,
                           rng=Random.default_rng()) -> ExposureProbabilities
    exposure_probabilities(s, assignments::AbstractMatrix; mapping=ExposureMapping(),
                           weights=nothing) -> ExposureProbabilities

Marginal and joint probabilities of each exposure condition for every unit under
the assignment mechanism `design`, the design-based ingredients of the Aronow and
Samii (2017) estimators.

Because a unit's exposure depends on the treatments of others, ``π_i(k)`` varies
across units even under simple designs (well-connected units are exposed more
often) and must be computed from the design rather than assumed. With
`method = :exact` the design's support is enumerated (Bernoulli, complete,
stratified and cluster randomization), giving exact ``π_i(k)`` and ``π_{ij}(k, l)``;
`method = :auto` does so when the support has at most `max_exact` assignments.

With `method = :monte_carlo` the probabilities are the raw frequencies over `draws`
simulated assignments. They are then estimates, and the Horvitz–Thompson estimator
that uses them is no longer exactly unbiased: ``1/\\hat π_i(k)`` is a noisy (and, by
Jensen's inequality, upward-biased) estimate of ``1/π_i(k)``, and a unit that never
falls in a condition gets ``\\hat π = 0`` even though its true probability is
positive, which [`exposure_positivity`](@ref) then reports as a positivity failure.
The relative Monte Carlo error of ``\\hat π_i(k)`` is about
``\\sqrt{(1 - π_i(k)) / (R π_i(k))}`` with ``R`` draws, and joint probabilities of
rare pairs are estimated with far less precision. As guidance, choose ``R`` so that
``R`` times the smallest relevant marginal probability is at least several hundred
(e.g. ``R ≥ 10^5`` when some ``π_i(k)`` is near ``0.005``), check that
`exposure_positivity` reports no zero probabilities that are not structural, and
confirm that estimates are stable when `draws` is doubled or the seed changed.

The second method takes a user-supplied set of assignments (one column per
assignment, e.g. the complete support of a bespoke design) with optional
probabilities; it is exact when the columns are the complete support.

# Arguments
- `s::InterferenceStructure`: the structure.
- `design::AssignmentMechanism`: the design over `structure_units(s)`, in that
  order.
- `assignments::AbstractMatrix`: alternatively, an ``N × R`` 0/1 matrix of
  assignments.

# Keywords
- `mapping::ExposureMapping`: default `ExposureMapping()` (the four Aronow–Samii
  conditions with `NeighborExposure(:any)`).
- `method::Symbol`: `:auto` (default), `:exact` or `:monte_carlo`.
- `draws::Integer`: number of simulated assignments for Monte Carlo; default
  10 000.
- `max_exact::Integer`: largest support enumerated under `:auto`; default 50 000.
- `rng::AbstractRNG`: random number generator for Monte Carlo draws.
- `weights`: probabilities of the supplied assignments (default: equally likely);
  must be non-negative and sum to one.

# Returns
- [`ExposureProbabilities`](@ref).

# Examples
```julia
using DrSnow, StableRNGs
rng = StableRNG(1)
n = 40
A = zeros(Int, n, n)
for i in 1:n, j in (i + 1):n
    rand(rng) < 0.08 && (A[i, j] = A[j, i] = 1)
end
g = NetworkStructure(1:n, A; directed=false)
design = CompleteRandomization(n, 15)
P = exposure_probabilities(g, design; method=:monte_carlo, draws=20_000, rng=rng)
exposure_positivity(P)
```

# References
- Aronow, P. M., & Samii, C. (2017). Estimating average causal effects under general
  interference, with application to a social network experiment. *Annals of Applied
  Statistics*, 11(4), 1912–1947.
"""
function exposure_probabilities(s::InterferenceStructure, design::AssignmentMechanism;
                                mapping::ExposureMapping=ExposureMapping(),
                                method::Symbol=:auto, draws::Integer=10_000,
                                max_exact::Integer=50_000,
                                rng::AbstractRNG=Random.default_rng())
    n_units(design) == n_units(s) ||
        throw(DimensionMismatch("design has $(n_units(design)) units, structure has " *
                                "$(n_units(s))"))
    method in (:auto, :exact, :monte_carlo) ||
        throw(ArgumentError("method must be :auto, :exact or :monte_carlo"))
    size_support = _sv_support_size(design)
    exact = method === :exact || (method === :auto && size_support <= max_exact)
    if exact
        isfinite(size_support) ||
            throw(ArgumentError("exact enumeration is not available for " *
                                "$(_sv_design_name(design)); use method=:monte_carlo"))
        size_support <= max(max_exact, 5_000_000) ||
            throw(ArgumentError("design support has $(size_support) assignments; use " *
                                "method=:monte_carlo"))
        Z, w = _sv_enumerate(design)
        return _sv_probabilities(s, mapping, Z, w, :exact, _sv_design_name(design))
    end
    draws >= 1 || throw(ArgumentError("draws must be positive"))
    Z = falses(n_units(s), draws)
    for r in 1:draws
        Z[:, r] .= draw_assignment(rng, design)
    end
    return _sv_probabilities(s, mapping, Z, fill(1 / draws, draws), :monte_carlo,
                             _sv_design_name(design))
end

function exposure_probabilities(s::InterferenceStructure, assignments::AbstractMatrix;
                                mapping::ExposureMapping=ExposureMapping(),
                                weights=nothing)
    size(assignments, 1) == n_units(s) ||
        throw(DimensionMismatch("assignments must have one row per structure unit"))
    R = size(assignments, 2)
    w = weights === nothing ? fill(1 / R, R) : Float64.(collect(weights))
    length(w) == R || throw(DimensionMismatch("one weight per assignment required"))
    (all(>=(0), w) && isapprox(sum(w), 1; atol=1e-8)) ||
        throw(ArgumentError("weights must be non-negative and sum to one"))
    Z = BitMatrix([_sv_binary_value(v, "exposure_probabilities") == 1
                   for v in assignments])
    return _sv_probabilities(s, mapping, Z, w, :supplied, "supplied assignments")
end

function _sv_probabilities(s, mapping, Z::BitMatrix, w, method, dname)
    f = _sv_prepare_mapping(s, mapping)
    n, R = size(Z)
    codes = zeros(UInt8, n, R)
    lookup = Dict{String,UInt8}()
    labels = String[]
    defined = BitVector()
    for r in 1:R
        lab = f(Z[:, r])
        def = .!ismissing.(lab)
        if r == 1
            defined = def
        elseif def != defined
            throw(ArgumentError("the exposure mapping is undefined for a set of units " *
                                "that changes with the assignment; this is not supported"))
        end
        for i in 1:n
            def[i] || continue
            c = get(lookup, lab[i], 0x00)
            if c == 0x00
                length(labels) >= 254 && throw(ArgumentError("more than 254 exposure " *
                                                             "conditions"))
                push!(labels, lab[i])
                c = UInt8(length(labels))
                lookup[lab[i]] = c
            end
            codes[i, r] = c
        end
    end
    # canonical order of levels
    perm = sortperm(labels)
    recode = zeros(UInt8, length(labels))
    for (newc, oldc) in enumerate(perm)
        recode[oldc] = UInt8(newc)
    end
    for k in eachindex(codes)
        codes[k] == 0 || (codes[k] = recode[codes[k]])
    end
    levels = labels[perm]
    pim = zeros(n, length(levels))
    for r in 1:R, i in 1:n
        c = codes[i, r]
        c == 0 || (pim[i, c] += w[r])
    end
    return ExposureProbabilities(copy(s.ids), levels, pim, codes, collect(w), defined,
                                 method, dname, mapping,
                                 Dict{Tuple{Int,Int},Matrix{Float64}}())
end

"""
    exposure_positivity(P::ExposureProbabilities; small=0.01) -> DataFrame

Positivity diagnostics for each exposure condition: how many units can never, or
only rarely, be observed in it under the design.

A unit with ``π_i(k) = 0`` can never be observed in condition ``k``, so contrasts
involving ``k`` are not identified for it without further assumptions (for example,
an isolate can never be "exposed"); units with small positive probabilities receive
large Horvitz–Thompson weights and make the estimates unstable. For each condition
the table reports the number of units with zero probability, the number with
probability below `small`, the smallest positive probability and the expected number
of units in the condition, ``\\sum_i π_i(k)``. Units with an undefined condition are
excluded. With Monte Carlo probabilities, zeros may be artefacts of too few draws
(see [`exposure_probabilities`](@ref)).

# Arguments
- `P::ExposureProbabilities`: exposure probabilities.

# Keywords
- `small::Real`: threshold for "small" probabilities; default 0.01.

# Returns
- `DataFrame` with columns `condition`, `n_zero`, `n_small`, `min_positive` and
  `expected_units`.

# Examples
```julia
using DrSnow
g = NetworkStructure(1:6, [0 1 0 0 0 0; 1 0 1 0 0 0; 0 1 0 0 0 0;
                           0 0 0 0 1 0; 0 0 0 1 0 0; 0 0 0 0 0 0])
exposure_positivity(exposure_probabilities(g, CompleteRandomization(6, 2)))
```
"""
function exposure_positivity(P::ExposureProbabilities; small::Real=0.01)
    d = P.defined
    rows = map(enumerate(P.levels)) do (k, lab)
        p = P.pi[d, k]
        pos = p[p .> 0]
        (condition=lab, n_zero=count(==(0), p), n_small=count(x -> 0 < x < small, p),
         min_positive=isempty(pos) ? NaN : minimum(pos), expected_units=sum(p))
    end
    return DataFrame(rows)
end

const _SV_JOINT_CACHE_MAX = 3000

# Joint probabilities π_ij(k, l) for i ∈ rows, j ∈ cols. Full matrices are cached for
# moderate N; otherwise the requested block is computed from the draws.
function _sv_joint(P::ExposureProbabilities, k::Integer, l::Integer,
                   rows::AbstractVector{Int}, cols::AbstractVector{Int})
    n = size(P.codes, 1)
    if n <= _SV_JOINT_CACHE_MAX
        J = get!(P.joint_cache, (Int(k), Int(l))) do
            haskey(P.joint_cache, (Int(l), Int(k))) ?
                Matrix(transpose(P.joint_cache[(Int(l), Int(k))])) :
                _sv_joint_block(P, k, l, 1:n, 1:n)
        end
        return J[rows, cols]
    end
    return _sv_joint_block(P, k, l, rows, cols)
end

function _sv_joint_block(P::ExposureProbabilities, k::Integer, l::Integer,
                         rows::AbstractVector{Int}, cols::AbstractVector{Int})
    R = size(P.codes, 2)
    out = zeros(length(rows), length(cols))
    bs = 2048
    for r0 in 1:bs:R
        r1 = min(R, r0 + bs - 1)
        A = Float64.(view(P.codes, rows, r0:r1) .== k) .* transpose(view(P.weights, r0:r1))
        B = Float64.(view(P.codes, cols, r0:r1) .== l)
        mul!(out, A, transpose(B), 1.0, 1.0)
    end
    return out
end

# ---------------------------------------------------------------------------------
# Horvitz–Thompson / Hájek estimation
# ---------------------------------------------------------------------------------

# Conservative covariance matrix of the HT totals Ŷ_T(k), k ∈ lv (Aronow & Samii
# 2017, Sec. 5), for values `y` of the units in `pop` observed in condition `c`.
function _sv_total_cov(P::ExposureProbabilities, pop::Vector{Int}, c::Vector{Int},
                       y::Vector{Float64}, lv::Vector{Int})
    K = length(lv)
    V = zeros(K, K)
    npop = length(pop)
    members = [findall(==(k), c) for k in lv]                  # positions within pop
    for (a, k) in enumerate(lv)
        Sk = members[a]
        isempty(Sk) && continue
        J = _sv_joint(P, k, k, pop[Sk], pop)                  # |S_k| × npop
        pk = P.pi[pop, k]
        v = 0.0
        a2 = 0.0
        for (r, i) in enumerate(Sk)
            pii = pk[i]
            v += (1 - pii) * (y[i] / pii)^2
            for j in Sk
                j == i && continue
                Jij = J[r, j]
                Jij > 0 || continue
                v += (Jij - pii * pk[j]) / Jij * y[i] * y[j] / (pii * pk[j])
            end
            m = 0
            for j in 1:npop
                j != i && J[r, j] == 0 && (m += 1)
            end
            a2 += y[i]^2 / pii * m
        end
        V[a, a] = v + a2
    end
    for a in 1:K, b in (a + 1):K
        k, l = lv[a], lv[b]
        Sk, Sl = members[a], members[b]
        pk = P.pi[pop, k]
        pl = P.pi[pop, l]
        c1 = 0.0
        c2 = 0.0
        if !isempty(Sk)
            Jkl = _sv_joint(P, k, l, pop[Sk], pop)             # |S_k| × npop
            for (r, i) in enumerate(Sk)
                for j in Sl
                    Jij = Jkl[r, j]
                    Jij > 0 || continue
                    c1 += (Jij - pk[i] * pl[j]) / Jij * y[i] * y[j] / (pk[i] * pl[j])
                end
                c2 += y[i]^2 / (2pk[i]) * count(==(0), view(Jkl, r, :))
            end
        end
        if !isempty(Sl)
            Jlk = _sv_joint(P, l, k, pop[Sl], pop)
            for (r, j) in enumerate(Sl)
                c2 += y[j]^2 / (2pl[j]) * count(==(0), view(Jlk, r, :))
            end
        end
        V[a, b] = V[b, a] = c1 - c2
    end
    return V
end

"""
    ExposureEffects <: CausalEstimate

Design-based estimates of average contrasts between exposure conditions, returned by
[`exposure_effects`](@ref).

Each coefficient `"a - b"` estimates
``τ(a, b) = N^{-1} \\sum_{i=1}^N [Y_i(a) - Y_i(b)]``, the average over the ``N``
units of the estimand population of the difference between the potential outcomes
in exposure conditions ``a`` and ``b`` (or, under a misspecified mapping, the
expected exposure effect of Sävje 2024; see [`ExposureMapping`](@ref)). `vcov` is
the Aronow–Samii conservative variance estimator (for the Hájek estimator, its
linearized version), which is conservative in expectation under a correctly
specified mapping, so intervals tend to over-cover. Inference uses the normal
reference distribution (`dof_residual` is `Inf`).

# Fields
- `coef::Vector{Float64}`, `vcov::Matrix{Float64}`, `names::Vector{String}`:
  estimates, their covariance and the contrast names.
- `estimator::Symbol`: `:hajek` or `:horvitz_thompson`.
- `level_means::DataFrame`: per condition, the estimated mean potential outcome,
  the number of units observed in it and the expected number of units.
- `n::Int`: size of the estimand population.
- `excluded::Vector{Any}`: identifiers of units outside it (undefined exposure, or
  zero probability under `positivity = :restrict`).
- `probabilities::ExposureProbabilities`: the probabilities used.

# Accessors
- The [`CausalEstimate`](@ref) interface (`coef`, `vcov`, `confint`, [`tidy`](@ref),
  …).
"""
struct ExposureEffects{P} <: CausalEstimate
    coef::Vector{Float64}
    vcov::Matrix{Float64}
    names::Vector{String}
    estimator::Symbol
    level_means::DataFrame
    n::Int
    excluded::Vector{Any}
    probabilities::P
end

StatsAPI.coef(r::ExposureEffects) = r.coef
StatsAPI.vcov(r::ExposureEffects) = r.vcov
StatsAPI.coefnames(r::ExposureEffects) = r.names
StatsAPI.nobs(r::ExposureEffects) = r.n
estimand(::ExposureEffects) = "average contrasts between exposure conditions"
method_name(r::ExposureEffects) =
    (r.estimator === :hajek ? "Hájek" : "Horvitz–Thompson") *
    " exposure-contrast estimator (Aronow–Samii)"

function show_details(io::IO, r::ExposureEffects)
    println(io)
    println(io, "Exposure probabilities: ", r.probabilities.method, " (",
            length(r.probabilities.weights), " assignments, design ",
            r.probabilities.design, ")")
    isempty(r.excluded) ||
        println(io, length(r.excluded), " unit(s) outside the estimand population " *
                    "(undefined exposure or zero probability)")
    println(io, "Variance: Aronow–Samii conservative estimator.")
end

"""
    exposure_effects(data, outcome, treatment, s, design; unit, mapping=ExposureMapping(),
                     contrasts=nothing, reference=nothing, estimator=:hajek,
                     positivity=:error, kwargs...) -> ExposureEffects
    exposure_effects(data, outcome, treatment, s, P::ExposureProbabilities; unit, ...)

Design-based Horvitz–Thompson or Hájek estimates of average causal contrasts between
exposure conditions under interference, with the conservative variance estimator of
Aronow and Samii (2017).

The estimand is the average contrast
``τ(a, b) = N^{-1} \\sum_i [Y_i(a) - Y_i(b)]`` between exposure conditions ``a`` and
``b`` (for example `"treated_unexposed"` versus `"control_unexposed"` for the direct
effect, `"control_exposed"` versus `"control_unexposed"` for the spillover on
untreated units), over the units for which both conditions have positive
probability. Identification rests on two assumptions: the assignment mechanism is
known (it is used to compute ``π_i(k)``), and the exposure mapping is correctly
specified. The data cannot confirm the second one; under misspecification the
estimators target the design-dependent expected exposure effect of Sävje (2024)
instead (see [`ExposureMapping`](@ref)).

With ``D_i`` the observed condition, the Horvitz–Thompson estimator of the mean
potential outcome in condition ``k`` is
```math
\\hat μ_{HT}(k) = \\frac{1}{N} \\sum_{i=1}^N \\frac{1\\{D_i = k\\} Y_i}{π_i(k)},
```
unbiased when the probabilities are exact, and the Hájek estimator divides by
``\\sum_i 1\\{D_i = k\\}/π_i(k)`` instead of ``N``; it has a small finite-sample bias
but is invariant to shifts of the outcome and usually much less variable, which is
why it is the default. The variance of the Horvitz–Thompson totals is estimated with
the Aronow–Samii estimator, which uses the joint probabilities ``π_{ij}(k, l)`` and
adds a correction for pairs of units that can never be jointly observed in the
relevant conditions (``π_{ij} = 0``), making it conservative in expectation; for the
Hájek estimator it is applied to the residuals from the condition means
(linearization). Intervals use a normal approximation, whose justification requires
that dependence among units' exposures be limited as ``N`` grows (Aronow and Samii
2017; Leung 2022). The Horvitz–Thompson estimator is skewed when few units fall in a
condition, so its normal-approximation intervals can undercover in small samples
even though its variance estimator is conservative.

With Monte Carlo probabilities (the default for large designs) the estimates inherit
the Monte Carlo error of ``\\hat π``; see [`exposure_probabilities`](@ref) for
guidance on `draws`. Positivity violations are an error by default, because
dropping the affected units changes the estimand; `positivity = :restrict` makes
that change explicit and records the excluded units. For regression-based
alternatives that do not require the design, see [`exposure_regression`](@ref); for
a test of the null of no spillovers, [`spillover_fisher_test`](@ref).

# Arguments
- `data`: table with one row per unit of `s`, matched by `unit`; every unit's
  outcome is required.
- `outcome::Symbol`, `treatment::Symbol`: outcome and binary treatment columns.
- `s::InterferenceStructure`: the structure.
- `design::AssignmentMechanism`: the design over `structure_units(s)`; or
  `P::ExposureProbabilities`, precomputed with the same mapping (recommended when
  estimating several outcomes).

# Keywords
- `unit::Symbol`: unit-identifier column.
- `mapping::ExposureMapping`: default `ExposureMapping()` (first method only; the
  second uses the mapping stored in `P`).
- `contrasts`: vector of `"a" => "b"` pairs; default: every condition against
  `reference`.
- `reference`: reference condition; default `"control_unexposed"` when present,
  otherwise the first condition.
- `estimator::Symbol`: `:hajek` (default) or `:horvitz_thompson`.
- `positivity::Symbol`: `:error` (default) throws if a unit has zero probability of
  a condition used in the contrasts; `:restrict` removes such units from the
  estimand population.
- `kwargs...`: passed to [`exposure_probabilities`](@ref) (`method`, `draws`,
  `max_exact`, `rng`).

# Returns
- [`ExposureEffects`](@ref).

# Examples
```julia
using DrSnow, DataFrames, StableRNGs
rng = StableRNG(1)
n = 80
A = zeros(Int, n, n)
for i in 1:n, j in (i + 1):n
    rand(rng) < 0.05 && (A[i, j] = A[j, i] = 1)
end
g = NetworkStructure(1:n, A)
design = CompleteRandomization(n, 30)
z = draw_assignment(rng, design)
exposed = coalesce.(compute_exposure(g, z, NeighborExposure(:any)).any, 0.0)
df = DataFrame(id=1:n, z=Int.(z), y=1.0 .* z .+ 0.5 .* exposed .+ randn(rng, n))
r = exposure_effects(df, :y, :z, g, design; unit=:id, method=:monte_carlo,
                     draws=20_000, rng=rng)       # isolates are outside the estimand
confint(r)
```

# References
- Aronow, P. M., & Samii, C. (2017). Estimating average causal effects under general
  interference, with application to a social network experiment. *Annals of Applied
  Statistics*, 11(4), 1912–1947.
- Horvitz, D. G., & Thompson, D. J. (1952). A generalization of sampling without
  replacement from a finite universe. *Journal of the American Statistical
  Association*, 47(260), 663–685.
- Sävje, F. (2024). Causal inference with misspecified exposure mappings: Separating
  definitions and assumptions. *Biometrika*, 111(1), 1–15.
- Leung, M. P. (2022). Causal inference under approximate neighborhood interference.
  *Econometrica*, 90(1), 267–293.
- Hudgens, M. G., & Halloran, M. E. (2008). Toward causal inference with
  interference. *Journal of the American Statistical Association*, 103(482),
  832–842.
"""
function exposure_effects(data, outcome::Symbol, treatment::Symbol,
                          s::InterferenceStructure, design::AssignmentMechanism;
                          unit::Symbol, mapping::ExposureMapping=ExposureMapping(),
                          contrasts=nothing, reference=nothing, estimator::Symbol=:hajek,
                          positivity::Symbol=:error, kwargs...)
    z = _sv_cross_section_treatment(data, treatment, s, unit, "exposure_effects")
    _sv_check_support(design, z)
    P = exposure_probabilities(s, design; mapping=mapping, kwargs...)
    return exposure_effects(data, outcome, treatment, s, P; unit=unit,
                            contrasts=contrasts, reference=reference,
                            estimator=estimator, positivity=positivity)
end

# Observed treatment in structure order, from a one-row-per-unit table.
function _sv_cross_section_treatment(data, treatment, s, unit, context)
    require_columns(data, [treatment, unit]; context=context)
    df = DataFrame(data; copycols=false)
    nrow(df) == n_units(s) ||
        throw(ArgumentError("$context: data must have exactly one row per structure " *
                            "unit ($(n_units(s)) rows expected, got $(nrow(df)))"))
    idx = _sv_require_units(s, df[!, unit], context)
    z = falses(n_units(s))
    for (r, i) in enumerate(idx)
        v = _sv_binary_value(df[r, treatment], context)
        ismissing(v) && throw(ArgumentError("$context: missing treatment in row $r"))
        z[i] = v == 1
    end
    return z
end

function exposure_effects(data, outcome::Symbol, treatment::Symbol,
                          s::InterferenceStructure, P::ExposureProbabilities;
                          unit::Symbol, contrasts=nothing, reference=nothing,
                          estimator::Symbol=:hajek, positivity::Symbol=:error)
    context = "exposure_effects"
    estimator in (:hajek, :horvitz_thompson) ||
        throw(ArgumentError("estimator must be :hajek or :horvitz_thompson"))
    positivity in (:error, :restrict) ||
        throw(ArgumentError("positivity must be :error or :restrict"))
    P.ids == s.ids || throw(ArgumentError("$context: probabilities were computed for a " *
                                          "different structure"))
    require_columns(data, [outcome]; context=context)
    z = _sv_cross_section_treatment(data, treatment, s, unit, context)
    df = DataFrame(data; copycols=false)
    idx = _sv_rows_to_index(s, df[!, unit], context)
    n = n_units(s)
    y = fill(NaN, n)
    for (r, i) in enumerate(idx)
        v = df[r, outcome]
        (ismissing(v) || !isfinite(v)) &&
            throw(ArgumentError("$context: outcome is missing or non-finite for unit " *
                                "$(repr(s.ids[i])); design-based estimation needs every " *
                                "unit's outcome"))
        y[i] = Float64(v)
    end
    lab = _sv_prepare_mapping(s, P.mapping)(z)
    lookup = Dict(l => k for (k, l) in enumerate(P.levels))
    cobs = zeros(Int, n)
    for i in 1:n
        ismissing(lab[i]) && continue
        haskey(lookup, lab[i]) ||
            throw(ArgumentError("$context: observed condition `$(lab[i])` of unit " *
                                "$(repr(s.ids[i])) never occurs in the probability " *
                                "draws; increase `draws` or check the design"))
        cobs[i] = lookup[lab[i]]
    end
    # contrasts
    pairs = if contrasts === nothing
        ref = reference === nothing ?
              ("control_unexposed" in P.levels ? "control_unexposed" : P.levels[1]) :
              string(reference)
        ref in P.levels || throw(ArgumentError("$context: reference `$ref` is not an " *
                                               "exposure condition; conditions: " *
                                               join(P.levels, ", ")))
        [l => ref for l in P.levels if l != ref]
    else
        [string(first(p)) => string(last(p)) for p in contrasts]
    end
    isempty(pairs) && throw(ArgumentError("$context: no contrasts to estimate"))
    for p in pairs, l in (first(p), last(p))
        l in P.levels || throw(ArgumentError("$context: unknown exposure condition " *
                                             "`$l`; conditions: " * join(P.levels, ", ")))
    end
    lvnames = unique(vcat(first.(pairs), last.(pairs)))
    lv = [lookup[l] for l in lvnames]
    # estimand population: defined units with positive probability for every level used
    ok = [cobs[i] != 0 && all(k -> P.pi[i, k] > 0, lv) for i in 1:n]
    bad = [i for i in 1:n if cobs[i] != 0 && !ok[i]]
    if !isempty(bad) && positivity === :error
        throw(ArgumentError("$context: $(length(bad)) unit(s) have zero probability of " *
                            "at least one condition in the contrasts (e.g. " *
                            "$(repr(s.ids[bad[1]]))), so the contrast is not identified " *
                            "for them. Use positivity=:restrict to redefine the " *
                            "estimand " *
                            "on the remaining units (see exposure_positivity)"))
    end
    pop = findall(ok)
    N = length(pop)
    N >= 2 || throw(ArgumentError("$context: fewer than two units in the estimand " *
                                  "population"))
    c = cobs[pop]
    yp = y[pop]
    # totals and means per level
    K = length(lv)
    tot = zeros(K)
    wsum = zeros(K)
    nobs_k = zeros(Int, K)
    for (a, k) in enumerate(lv), (q, i) in enumerate(pop)
        c[q] == k || continue
        tot[a] += yp[q] / P.pi[i, k]
        wsum[a] += 1 / P.pi[i, k]
        nobs_k[a] += 1
    end
    if estimator === :hajek
        # (Horvitz–Thompson remains defined, and unbiased, with an empty condition.)
        for (a, l) in enumerate(lvnames)
            nobs_k[a] > 0 ||
                throw(ArgumentError("$context: no unit is observed in condition `$l`, " *
                                    "so its Hájek mean is undefined"))
        end
    end
    means = estimator === :hajek ? tot ./ wsum : tot ./ N
    L = zeros(length(pairs), K)
    for (q, p) in enumerate(pairs)
        L[q, findfirst(==(first(p)), lvnames)] += 1
        L[q, findfirst(==(last(p)), lvnames)] -= 1
    end
    b = L * means
    # Hájek: linearized variance = HT variance of residuals from the condition means
    resid(q) = c[q] in lv ? yp[q] - means[findfirst(==(c[q]), lv)] : 0.0
    u = estimator === :hajek ? [resid(q) for q in eachindex(yp)] : yp
    Vt = _sv_total_cov(P, pop, c, u, lv)
    V = Symmetric(L * Vt * transpose(L) ./ N^2) |> Matrix
    lm = DataFrame(condition=lvnames, mean=means, n_observed=nobs_k,
                   expected_units=[sum(P.pi[pop, k]) for k in lv])
    excluded = Any[s.ids[i] for i in 1:n if !ok[i]]
    names = [first(p) * " - " * last(p) for p in pairs]
    return ExposureEffects(b, V, names, estimator, lm, N, excluded, P)
end
