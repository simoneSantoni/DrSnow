# Additional assignment mechanisms (matched pairs, blocked cluster randomization,
# rerandomization) and exact enumeration of assignment supports.
#
# These extend the interface defined in designs.jl; the existing mechanisms there
# are not modified.

"""
    MatchedPairsRandomization(pairs)

Matched-pairs design: units are grouped in pairs (typically matched on baseline
covariates) and exactly one unit of each pair is treated, with probability 1/2,
independently across pairs; the support has ``2^P`` equally likely assignments for
``P`` pairs.

Matched pairs are the limiting case of stratification with two units per stratum.
Within-pair comparisons remove the matched covariates from the error, which can
substantially increase power. With one treated and one control unit per pair the
Neyman variance of a within-stratum difference cannot be estimated pair by pair; the
`:studentized` statistic of [`randomization_test`](@ref) therefore uses the
between-pair variance of the pair differences, which is conservative for the
average effect (Imai 2008 discusses this estimator; see also Imbens and Rubin 2015,
ch. 10).

# Arguments
- `pairs::AbstractVector`: one pair label per unit; every label must occur exactly
  twice.

# Returns
- `MatchedPairsRandomization`.

# Examples
```julia
using DrSnow, StableRNGs
m = MatchedPairsRandomization([1, 1, 2, 2, 3, 3])
n_assignments(m)                             # 8
draw_assignment(StableRNG(1), m)
```

# References
- Imbens, G. W., & Rubin, D. B. (2015). *Causal Inference for Statistics, Social,
  and Biomedical Sciences: An Introduction*, ch. 10. Cambridge University Press.
- Imai, K. (2008). Variance identification and efficiency analysis in randomized
  experiments under the matched-pair design. *Statistics in Medicine*, 27,
  4857–4873.
"""
struct MatchedPairsRandomization <: AssignmentMechanism
    n::Int
    pairs::Vector{Tuple{Int,Int}}
end

function MatchedPairsRandomization(pairs::AbstractVector)
    idx = Dict{Any,Vector{Int}}()
    for (i, p) in enumerate(pairs)
        push!(get!(idx, p, Int[]), i)
    end
    for (l, v) in idx
        length(v) == 2 || throw(ArgumentError("pair label $l has $(length(v)) units; " *
                                              "every pair must have exactly 2"))
    end
    labels = sort!(collect(keys(idx)); by=_ri_sortkey)
    return MatchedPairsRandomization(length(pairs),
                                     [(idx[l][1], idx[l][2]) for l in labels])
end

"""
    BlockClusterRandomization(clusters, blocks, n_treated_per_block)
    BlockClusterRandomization(clusters, blocks, z)

Cluster randomization within blocks: in every block a fixed number of whole clusters
is treated, all subsets of clusters of that size being equally likely,
independently across blocks, and every unit of a treated cluster is treated.

This is the design of many field experiments that randomize schools within
districts or villages within regions. Randomization inference permutes cluster
labels within blocks; statistics compare clusters within blocks and use cluster-level
variance estimates. It is the mechanism that [`randomization_test`](@ref) builds
when both `strata` and `cluster` are given.

# Arguments
- `clusters::AbstractVector`: one cluster label per unit.
- `blocks::AbstractVector`: one block label per unit; every cluster must lie in a
  single block.
- `n_treated_per_block::AbstractDict`: number of treated clusters for each block
  label.
- `z::AbstractVector`: alternatively, an observed assignment (constant within
  clusters) from which the per-block counts are taken.

# Returns
- `BlockClusterRandomization`.

# Examples
```julia
using DrSnow, StableRNGs
school = repeat(1:8; inner=5)                 # 8 schools of 5 pupils
district = repeat([1, 2]; inner=20)           # 4 schools per district
m = BlockClusterRandomization(school, district, Dict(1 => 2, 2 => 2))
n_assignments(m)                              # 36
draw_assignment(StableRNG(2), m)
```
"""
struct BlockClusterRandomization <: AssignmentMechanism
    n::Int
    blocks::Vector{Vector{Vector{Int}}}
    n_treated::Vector{Int}
end

function _ri_block_clusters(clusters::AbstractVector, blocks::AbstractVector)
    length(clusters) == length(blocks) ||
        throw(DimensionMismatch("clusters and blocks must have the same length"))
    cl_units = _ri_groups(clusters)
    block_of = Dict{Any,Any}()
    for (c, units) in cl_units
        bs = unique(blocks[units])
        length(bs) == 1 || throw(ArgumentError("cluster $c spans several blocks"))
        block_of[c] = bs[1]
    end
    blabels = sort!(unique(collect(values(block_of))); by=_ri_sortkey)
    out = Vector{Vector{Vector{Int}}}()
    for b in blabels
        push!(out, [units for (c, units) in cl_units if isequal(block_of[c], b)])
    end
    return blabels, out
end

function BlockClusterRandomization(clusters::AbstractVector, blocks::AbstractVector,
                                   counts::AbstractDict)
    blabels, bl = _ri_block_clusters(clusters, blocks)
    nt = [Int(counts[b]) for b in blabels]
    all(i -> 0 <= nt[i] <= length(bl[i]), eachindex(nt)) ||
        throw(ArgumentError("treated-cluster count exceeds the clusters in a block"))
    return BlockClusterRandomization(length(clusters), bl, nt)
end

function BlockClusterRandomization(clusters::AbstractVector, blocks::AbstractVector,
                                   z::AbstractVector{<:Union{Bool,Integer}})
    length(z) == length(clusters) ||
        throw(DimensionMismatch("clusters and z lengths differ"))
    blabels, bl = _ri_block_clusters(clusters, blocks)
    nt = Int[]
    for cls in bl
        k = 0
        for units in cls
            zs = z[units]
            all(==(zs[1]), zs) ||
                throw(ArgumentError("assignment varies within a cluster"))
            k += zs[1] == 1
        end
        push!(nt, k)
    end
    return BlockClusterRandomization(length(clusters), bl, nt)
end

"""
    Rerandomization(base, X; threshold, max_draws=100_000)
    Rerandomization(base, accept; max_draws=100_000)

Rerandomization (Morgan and Rubin 2012): assignments are drawn from the `base`
mechanism and redrawn until a pre-specified balance criterion is met, so the design
is the base design restricted to the acceptable assignments.

With a covariate matrix `X` the criterion is the Mahalanobis balance criterion
[`balance_mahalanobis`](@ref)`(X, z) ≤ threshold`; alternatively any function
`accept(z)::Bool` can be given. Under complete randomization and approximately
normal covariate means, ``M`` is approximately ``χ^2_k`` with ``k`` covariates, so a
threshold equal to the ``p_a`` quantile of ``χ^2_k`` accepts about a share ``p_a``
of assignments. Rerandomization improves the precision of the difference in means
when the covariates predict the outcome.

The randomization distribution must be computed under the same criterion that was
used in the experiment: re-randomizing without it produces a reference distribution
that is too dispersed, and the test is then conservative. Randomization tests that
use this mechanism are exact (Morgan and Rubin 2012). Exact enumeration filters the
base support by the criterion; Monte Carlo inference draws until acceptance, which
can be slow for strict thresholds.

# Arguments
- `base::AssignmentMechanism`: the design before the balance restriction.
- `X::AbstractMatrix`: `n × k` covariates, rows in unit order.
- `accept`: alternatively, a function `z -> Bool` defining acceptable assignments.

# Keywords
- `threshold::Real`: maximum acceptable Mahalanobis distance (required with `X`).
- `max_draws::Integer`: maximum number of base draws per accepted assignment before
  an error is raised; default 100 000.

# Returns
- `Rerandomization`.

# Examples
```julia
using DrSnow, StableRNGs
X = randn(StableRNG(1), 20, 2)
m = Rerandomization(CompleteRandomization(20, 10), X; threshold=0.45)
z = draw_assignment(StableRNG(2), m)
balance_mahalanobis(X, z) <= 0.45             # true
```

# References
- Morgan, K. L., & Rubin, D. B. (2012). Rerandomization to improve covariate balance
  in experiments. *Annals of Statistics*, 40(2), 1263–1282.
"""
struct Rerandomization{M<:AssignmentMechanism,F} <: AssignmentMechanism
    base::M
    accept::F
    max_draws::Int
end

function Rerandomization(base::AssignmentMechanism, X::AbstractMatrix;
                         threshold::Real, max_draws::Integer=100_000)
    size(X, 1) == n_units(base) ||
        throw(DimensionMismatch("X must have one row per unit of the base mechanism"))
    Xf = Matrix{Float64}(X)
    thr = float(threshold)
    thr >= 0 || throw(ArgumentError("threshold must be non-negative"))
    return Rerandomization(base, z -> balance_mahalanobis(Xf, z) <= thr, Int(max_draws))
end

Rerandomization(base::AssignmentMechanism, accept::Function;
                max_draws::Integer=100_000) =
    Rerandomization(base, accept, Int(max_draws))

"""
    balance_mahalanobis(X, z) -> Float64

Mahalanobis distance between the treated and control covariate means,
```math
M = (\\bar x_1 - \\bar x_0)'
    \\left[\\left(\\tfrac{1}{n_1} + \\tfrac{1}{n_0}\\right) S\\right]^{-1}
    (\\bar x_1 - \\bar x_0),
```
with ``S`` the full-sample covariance matrix of the covariates (Morgan and Rubin
2012).

``M`` is the balance criterion of [`Rerandomization`](@ref). Under complete
randomization and approximate normality of the mean differences it is approximately
``χ^2_k``-distributed with ``k`` covariates. Because ``S`` does not depend on the
assignment, ``M`` is invariant to affine transformations of the covariates. A
pseudo-inverse is used when ``S`` is singular (collinear covariates), in which case
the redundant directions are ignored.

# Arguments
- `X::AbstractMatrix`: covariates, one row per unit.
- `z::AbstractVector`: 0/1 treatment indicator in the row order of `X`.

# Returns
- `Float64`; `Inf` when one arm is empty.

# Examples
```julia
using DrSnow, StableRNGs
rng = StableRNG(1)
X = randn(rng, 30, 3)
z = [trues(15); falses(15)]
balance_mahalanobis(X, z)
```

# References
- Morgan, K. L., & Rubin, D. B. (2012). Rerandomization to improve covariate balance
  in experiments. *Annals of Statistics*, 40(2), 1263–1282.
"""
function balance_mahalanobis(X::AbstractMatrix, z::AbstractVector)
    size(X, 1) == length(z) || throw(DimensionMismatch("X rows and z length differ"))
    t = findall(==(1), z)
    c = findall(==(0), z)
    (isempty(t) || isempty(c)) && return Inf
    d = vec(mean(X[t, :]; dims=1) .- mean(X[c, :]; dims=1))
    S = cov(Matrix{Float64}(X)) .* (1 / length(t) + 1 / length(c))
    return float(dot(d, pinv(S) * d))
end

"""
    n_units(m::AssignmentMechanism) -> Int
    n_units(s::InterferenceStructure) -> Int

Number of units assigned by the mechanism `m` (the length of the vectors returned
by [`draw_assignment`](@ref)), or the number of units of an interference structure.

# Arguments
- `m::AssignmentMechanism` or `s::InterferenceStructure`.

# Returns
- `Int`.

# Examples
```julia
using DrSnow
n_units(CompleteRandomization(10, 4))                   # 10
n_units(PartitionStructure(1:6, [1, 1, 1, 2, 2, 2]))     # 6
```
"""
n_units

n_units(m::Union{MatchedPairsRandomization,BlockClusterRandomization}) = m.n
n_units(m::Rerandomization) = n_units(m.base)

function draw_assignment(rng::AbstractRNG, m::MatchedPairsRandomization)
    z = falses(m.n)
    for (a, b) in m.pairs
        z[rand(rng, Bool) ? a : b] = true
    end
    return z
end

function draw_assignment(rng::AbstractRNG, m::BlockClusterRandomization)
    z = falses(m.n)
    for (cls, k) in zip(m.blocks, m.n_treated)
        for c in randperm(rng, length(cls))[1:k]
            z[cls[c]] .= true
        end
    end
    return z
end

function draw_assignment(rng::AbstractRNG, m::Rerandomization)
    for _ in 1:m.max_draws
        z = draw_assignment(rng, m.base)
        m.accept(z) && return z
    end
    error("Rerandomization: no acceptable assignment in $(m.max_draws) draws; " *
          "the criterion may be too strict")
end

treatment_probabilities(m::MatchedPairsRandomization; kwargs...) = fill(0.5, m.n)
function treatment_probabilities(m::BlockClusterRandomization; kwargs...)
    p = zeros(m.n)
    for (cls, k) in zip(m.blocks, m.n_treated)
        for units in cls
            p[units] .= k / length(cls)
        end
    end
    return p
end

# ---------------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------------

# Sort key that orders mixed label types deterministically.
_ri_sortkey(x) = x isa Real ? (0, float(x), "") : (1, 0.0, string(x))

# Ordered (label => unit indices) pairs, labels sorted.
function _ri_groups(labels::AbstractVector)
    d = Dict{Any,Vector{Int}}()
    for (i, l) in enumerate(labels)
        push!(get!(d, l, Int[]), i)
    end
    ks = sort!(collect(keys(d)); by=_ri_sortkey)
    return [k => d[k] for k in ks]
end

# ---------------------------------------------------------------------------------
# Enumeration
# ---------------------------------------------------------------------------------

"""
    n_assignments(m::AssignmentMechanism) -> Union{BigInt,Nothing}

Number of assignments with positive probability under `m`, i.e. the size of the
support of the design, or `nothing` when it is not known without enumeration
([`CustomAssignment`](@ref), [`Rerandomization`](@ref)).

The support size decides whether exact randomization inference is feasible: the
randomization functions enumerate the support when it has at most `nperm`
assignments (`exact = :auto`). For stratified and blocked designs it is the product
of the per-stratum binomial coefficients; for matched pairs ``2^P``; for Bernoulli
designs ``2^m`` with ``m`` the number of units whose probability is strictly
between 0 and 1.

# Arguments
- `m::AssignmentMechanism`: the design.

# Returns
- `BigInt` or `nothing`.

# Examples
```julia
using DrSnow
n_assignments(CompleteRandomization(10, 5))                          # 252
m = StratifiedRandomization([1, 1, 1, 1, 2, 2], Dict(1 => 2, 2 => 1))
n_assignments(m)                                                     # 6 × 2 = 12
```
"""
n_assignments(m::CompleteRandomization) = binomial(big(m.n), m.n_treated)
n_assignments(m::StratifiedRandomization) =
    prod(binomial(big(length(g)), k) for (g, k) in zip(m.groups, m.n_treated); init=big(1))
n_assignments(m::ClusterRandomization) =
    binomial(big(length(m.groups)), m.n_treated_clusters)
n_assignments(m::BlockClusterRandomization) =
    prod(binomial(big(length(c)), k) for (c, k) in zip(m.blocks, m.n_treated); init=big(1))
n_assignments(m::MatchedPairsRandomization) = big(2)^length(m.pairs)
n_assignments(m::BernoulliAssignment) = big(2)^count(p -> 0 < p < 1, m.p)
n_assignments(::AssignmentMechanism) = nothing

# Number of base assignments that enumeration must visit (cost), or `nothing`.
_ri_enumeration_cost(m::AssignmentMechanism) = n_assignments(m)
_ri_enumeration_cost(m::Rerandomization) = _ri_enumeration_cost(m.base)

const _RI_MAX_ENUMERATE = 2_000_000

# All k-subsets of 1:n in lexicographic order.
function _ri_combinations(n::Int, k::Int)
    out = Vector{Vector{Int}}()
    k == 0 && return [Int[]]
    k > n && return out
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

# Cartesian product of per-component choices. `choices[j]` is a vector of unit-index
# vectors to set to true.
function _ri_product_assignments(n::Int, choices::Vector{Vector{Vector{Int}}})
    out = [falses(n)]
    for opts in choices
        next = BitVector[]
        for z in out, o in opts
            z2 = copy(z)
            z2[o] .= true
            push!(next, z2)
        end
        out = next
    end
    return out
end

"""
    enumerate_assignments(m; limit=2_000_000) -> (assignments, probabilities)

List every assignment with positive probability under the design `m`, together
with its probability.

Exact randomization inference evaluates the test statistic on every element of the
support and weights it by its probability; the resulting p-value has no Monte Carlo
error. For uniform designs (complete, stratified, cluster, blocked-cluster, matched
pairs) all probabilities are equal; for Bernoulli designs they are products of the
unit probabilities; for [`Rerandomization`](@ref) the base support is filtered by
the balance criterion and the probabilities renormalized. Enumeration is not
available for [`CustomAssignment`](@ref).

# Arguments
- `m::AssignmentMechanism`: any mechanism except `CustomAssignment`.

# Keywords
- `limit::Integer`: maximum number of (base) assignments to visit; larger supports
  raise an `ArgumentError`. Default 2 000 000.

# Returns
- `Vector{BitVector}` of assignments and `Vector{Float64}` of probabilities summing
  to one.

# Examples
```julia
using DrSnow
zs, w = enumerate_assignments(CompleteRandomization(6, 3))
length(zs), sum(w)                                       # (20, 1.0)
```
"""
function enumerate_assignments(m::AssignmentMechanism; limit::Integer=_RI_MAX_ENUMERATE)
    cost = _ri_enumeration_cost(m)
    cost === nothing &&
        throw(ArgumentError("exact enumeration is not available for $(nameof(typeof(m)))"))
    cost <= limit || throw(ArgumentError("mechanism has $cost assignments, above the " *
                                         "enumeration limit $limit"))
    zs = _ri_enumerate(m)
    w = _ri_enumeration_weights(m, zs)
    return zs, w
end

_ri_enumeration_weights(::AssignmentMechanism, zs) = fill(1 / length(zs), length(zs))

function _ri_enumeration_weights(m::BernoulliAssignment, zs)
    w = [prod(z[i] ? m.p[i] : 1 - m.p[i] for i in eachindex(m.p); init=1.0) for z in zs]
    return w ./ sum(w)
end

_ri_enumerate(m::CompleteRandomization) =
    _ri_product_assignments(m.n, [_ri_combinations(m.n, m.n_treated)])

_ri_enumerate(m::StratifiedRandomization) =
    _ri_product_assignments(m.n, [[g[c] for c in _ri_combinations(length(g), k)]
                                  for (g, k) in zip(m.groups, m.n_treated)])

_ri_enumerate(m::ClusterRandomization) =
    _ri_product_assignments(m.n, [[reduce(vcat, m.groups[c]; init=Int[])
                                   for c in _ri_combinations(length(m.groups),
                                                             m.n_treated_clusters)]])

_ri_enumerate(m::BlockClusterRandomization) =
    _ri_product_assignments(m.n, [[reduce(vcat, cls[c]; init=Int[])
                                   for c in _ri_combinations(length(cls), k)]
                                  for (cls, k) in zip(m.blocks, m.n_treated)])

_ri_enumerate(m::MatchedPairsRandomization) =
    _ri_product_assignments(m.n, [[[a], [b]] for (a, b) in m.pairs])

function _ri_enumerate(m::BernoulliAssignment)
    fixed = findall(==(1.0), m.p)
    free = findall(p -> 0 < p < 1, m.p)
    base = falses(length(m.p))
    base[fixed] .= true
    out = BitVector[]
    for mask in 0:(2^length(free) - 1)
        z = copy(base)
        for (j, i) in enumerate(free)
            z[i] = isodd(mask >> (j - 1))
        end
        push!(out, z)
    end
    return out
end

function _ri_enumerate(m::Rerandomization)
    zs = filter(m.accept, _ri_enumerate(m.base))
    isempty(zs) && error("Rerandomization: no assignment satisfies the criterion")
    return zs
end

function _ri_enumeration_weights(m::Rerandomization, zs)
    w = _ri_enumeration_weights(m.base, zs)
    return w ./ sum(w)
end

_ri_enumerate(m::AssignmentMechanism) =
    throw(ArgumentError("exact enumeration is not available for $(nameof(typeof(m)))"))

# ---------------------------------------------------------------------------------
# Support checks: is the observed assignment possible under the mechanism?
# ---------------------------------------------------------------------------------

_ri_in_support(m::CompleteRandomization, z) = count(z) == m.n_treated
_ri_in_support(m::StratifiedRandomization, z) =
    all(count(view(z, g)) == k for (g, k) in zip(m.groups, m.n_treated))
function _ri_in_support(m::ClusterRandomization, z)
    all(all(==(z[g[1]]), view(z, g)) for g in m.groups) || return false
    return count(z[g[1]] for g in m.groups) == m.n_treated_clusters
end
function _ri_in_support(m::BlockClusterRandomization, z)
    for (cls, k) in zip(m.blocks, m.n_treated)
        all(all(==(z[g[1]]), view(z, g)) for g in cls) || return false
        count(z[g[1]] for g in cls) == k || return false
    end
    return true
end
_ri_in_support(m::MatchedPairsRandomization, z) = all(z[a] != z[b] for (a, b) in m.pairs)
_ri_in_support(m::BernoulliAssignment, z) =
    all((z[i] && m.p[i] > 0) || (!z[i] && m.p[i] < 1) for i in eachindex(z))
_ri_in_support(m::Rerandomization, z) = _ri_in_support(m.base, z) && m.accept(z)
_ri_in_support(::AssignmentMechanism, z) = true

# Structure used by the test statistics: strata (unit groups) and variance clusters.
_ri_strata(m::StratifiedRandomization) = m.groups
_ri_strata(m::MatchedPairsRandomization) = [[a, b] for (a, b) in m.pairs]
_ri_strata(m::BlockClusterRandomization) =
    [reduce(vcat, cls; init=Int[]) for cls in m.blocks]
_ri_strata(m::Rerandomization) = _ri_strata(m.base)
_ri_strata(::AssignmentMechanism) = nothing

_ri_clusters(m::ClusterRandomization) = m.groups
_ri_clusters(m::BlockClusterRandomization) = reduce(vcat, m.blocks; init=Vector{Int}[])
_ri_clusters(m::Rerandomization) = _ri_clusters(m.base)
_ri_clusters(::AssignmentMechanism) = nothing

# Short description for printing.
_ri_describe(m::BernoulliAssignment) = "Bernoulli assignment ($(length(m.p)) units)"
_ri_describe(m::CompleteRandomization) =
    "complete randomization ($(m.n_treated) of $(m.n) units treated)"
_ri_describe(m::StratifiedRandomization) =
    "stratified randomization ($(length(m.groups)) strata, $(m.n) units)"
_ri_describe(m::ClusterRandomization) =
    "cluster randomization ($(m.n_treated_clusters) of $(length(m.groups)) clusters " *
    "treated)"
_ri_describe(m::BlockClusterRandomization) =
    "blocked cluster randomization ($(length(m.blocks)) blocks, " *
    "$(sum(length, m.blocks)) clusters)"
_ri_describe(m::MatchedPairsRandomization) =
    "matched-pairs randomization ($(length(m.pairs)) pairs)"
_ri_describe(m::Rerandomization) = "rerandomization over " * _ri_describe(m.base)
_ri_describe(m::AssignmentMechanism) = string(nameof(typeof(m)))
