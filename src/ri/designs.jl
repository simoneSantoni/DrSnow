# Assignment mechanisms: the known (or assumed) randomization used by design-based
# inference. Shared by randomization inference (src/ri) and interference (src/sutva).
#
# Interface for every `AssignmentMechanism` `m`:
#     draw_assignment(rng, m) :: BitVector   one re-randomized treatment vector
#     n_units(m)              :: Int
#     treatment_probabilities(m) :: Vector{Float64}   marginal P(Z_i = 1)
# The mechanisms operate on unit indices 1:n in the order of the unit vector the
# caller used to build them; callers must map ids ↔ indices by key.

"""
    AssignmentMechanism

Abstract supertype of treatment-assignment mechanisms: the known (or assumed)
probability distribution ``P(Z = z)`` over binary treatment vectors
``z ∈ \\{0, 1\\}^n`` that design-based inference conditions on.

In the design-based (randomization) framework of Fisher (1935) and Neyman (1923),
potential outcomes are fixed and the only source of randomness is the assignment of
treatment. The assignment mechanism is therefore the model: randomization tests
compare an observed statistic with its distribution over assignments drawn from the
mechanism, and design-based estimators weight units by probabilities computed from
it. Getting the mechanism right is essential; a test run under a different design
from the one actually used (for example complete randomization when the experiment
was stratified, or ignoring a rerandomization criterion) has the wrong reference
distribution and no guaranteed size.

Mechanisms index units ``1, …, n`` in the order of the unit vector used to build
them; the estimation functions match data rows to that order by key. Every subtype
implements [`draw_assignment`](@ref), [`n_units`](@ref) and
[`treatment_probabilities`](@ref); most also support exact enumeration of their
support through [`n_assignments`](@ref) and [`enumerate_assignments`](@ref).

# Subtypes
- [`BernoulliAssignment`](@ref), [`CompleteRandomization`](@ref),
  [`StratifiedRandomization`](@ref), [`ClusterRandomization`](@ref),
  [`BlockClusterRandomization`](@ref), [`MatchedPairsRandomization`](@ref),
  [`Rerandomization`](@ref), [`CustomAssignment`](@ref) and, for partial
  interference, [`TwoStageRandomization`](@ref).

# Examples
```julia
using DrSnow, StableRNGs
m = CompleteRandomization(10, 4)
m isa AssignmentMechanism                   # true
draw_assignment(StableRNG(1), m)
```

# References
- Fisher, R. A. (1935). *The Design of Experiments*. Oliver & Boyd.
- Splawa-Neyman, J., Dabrowska, D. M., & Speed, T. P. (1990). On the application of
  probability theory to agricultural experiments. Essay on principles. Section 9.
  *Statistical Science*, 5(4), 465–472. (Translation of the 1923 original.)
- Imbens, G. W., & Rubin, D. B. (2015). *Causal Inference for Statistics, Social,
  and Biomedical Sciences: An Introduction*, ch. 3–5. Cambridge University Press.
"""
abstract type AssignmentMechanism end

"""
    BernoulliAssignment(n, p)

Bernoulli (independent) assignment: each of `n` units is treated independently, unit
``i`` with probability ``p_i``, so that
``P(Z = z) = ∏_i p_i^{z_i} (1 - p_i)^{1 - z_i}``.

The number of treated units is random under this design. Randomization inference
that conditions on the observed number treated (as the column-based defaults of
[`randomization_test`](@ref) do, by building a [`CompleteRandomization`](@ref)) is
still valid under a Bernoulli design with a common ``p``, because conditionally on
the number treated all assignments are equally likely (Imbens and Rubin 2015, ch.
5); passing `BernoulliAssignment` explicitly as `mechanism` instead uses the
unconditional distribution. With unit-specific probabilities the conditional
distribution is no longer uniform, so the mechanism should be passed explicitly.

# Arguments
- `n::Integer`: number of units.
- `p`: common treatment probability (`Real`) or vector of `n` unit-specific
  probabilities in ``[0, 1]``. Units with ``p_i ∈ \\{0, 1\\}`` are never
  re-randomized.

# Returns
- `BernoulliAssignment`.

# Examples
```julia
using DrSnow, StableRNGs
m = BernoulliAssignment(8, 0.5)
draw_assignment(StableRNG(1), m)
n_assignments(m)                               # 256
```
"""
struct BernoulliAssignment <: AssignmentMechanism
    p::Vector{Float64}
    function BernoulliAssignment(n::Integer, p)
        pv = p isa Real ? fill(float(p), n) : float.(collect(p))
        length(pv) == n || throw(DimensionMismatch("length(p) must equal n"))
        all(x -> 0 <= x <= 1, pv) || throw(ArgumentError("probabilities must be in [0,1]"))
        return new(pv)
    end
end

"""
    CompleteRandomization(n, n_treated)

Completely randomized design: exactly `n_treated` of the `n` units are treated and
all ``\\binom{n}{n_1}`` subsets of that size are equally likely.

This is the design of the classical Fisher exact test and of Neyman's (1923)
repeated-sampling analysis of the difference in means, and the default mechanism
that DrSnow builds from data when no strata or clusters are declared. Every unit has
marginal treatment probability ``n_1 / n``.

# Arguments
- `n::Integer`: number of units.
- `n_treated::Integer`: number of treated units, ``0 ≤ n_1 ≤ n``.

# Returns
- `CompleteRandomization`.

# Examples
```julia
using DrSnow, StableRNGs
m = CompleteRandomization(10, 5)
n_assignments(m)                              # 252
draw_assignment(StableRNG(2), m)
```

# References
- Imbens, G. W., & Rubin, D. B. (2015). *Causal Inference for Statistics, Social,
  and Biomedical Sciences: An Introduction*, ch. 4–6. Cambridge University Press.
"""
struct CompleteRandomization <: AssignmentMechanism
    n::Int
    n_treated::Int
    function CompleteRandomization(n::Integer, n_treated::Integer)
        0 <= n_treated <= n || throw(ArgumentError("need 0 ≤ n_treated ≤ n"))
        return new(n, n_treated)
    end
end

"""
    StratifiedRandomization(strata, n_treated_per_stratum)
    StratifiedRandomization(strata, z)

Stratified (block) randomization: complete randomization carried out independently
within each stratum, with a fixed number of treated units per stratum.

Under this design the assignment probability ``n_{1s} / n_s`` may differ across
strata, so pooled comparisons of treated and control units are confounded by
stratum; DrSnow's randomization statistics therefore compare units within strata
(the difference in means is the stratum-size-weighted average of within-stratum
differences), and re-randomization permutes treatment only within strata.

# Arguments
- `strata::AbstractVector`: one stratum label per unit.
- `n_treated_per_stratum::AbstractDict`: number of treated units for each stratum
  label.
- `z::AbstractVector`: alternatively, an observed 0/1 assignment from which the
  per-stratum treated counts are taken.

# Returns
- `StratifiedRandomization`.

# Examples
```julia
using DrSnow, StableRNGs
strata = [1, 1, 1, 1, 2, 2, 2, 2]
m = StratifiedRandomization(strata, Dict(1 => 2, 2 => 1))
treatment_probabilities(m)                   # 0.5 in stratum 1, 0.25 in stratum 2
draw_assignment(StableRNG(3), m)
```

# References
- Imbens, G. W., & Rubin, D. B. (2015). *Causal Inference for Statistics, Social,
  and Biomedical Sciences: An Introduction*, ch. 9. Cambridge University Press.
"""
struct StratifiedRandomization <: AssignmentMechanism
    n::Int
    groups::Vector{Vector{Int}}
    n_treated::Vector{Int}
end

function StratifiedRandomization(strata::AbstractVector, counts::AbstractDict)
    labels = unique(strata)
    groups = [findall(==(l), strata) for l in labels]
    nt = [Int(counts[l]) for l in labels]
    all(i -> 0 <= nt[i] <= length(groups[i]), eachindex(nt)) ||
        throw(ArgumentError("treated count exceeds stratum size"))
    return StratifiedRandomization(length(strata), groups, nt)
end

function StratifiedRandomization(strata::AbstractVector, z::AbstractVector{<:Union{Bool,Integer}})
    length(z) == length(strata) || throw(DimensionMismatch("strata and z lengths differ"))
    labels = unique(strata)
    counts = Dict(l => count(i -> strata[i] == l && z[i] == 1, eachindex(z)) for l in labels)
    return StratifiedRandomization(strata, counts)
end

"""
    ClusterRandomization(clusters, n_treated_clusters)
    ClusterRandomization(clusters, z)

Cluster randomization: whole clusters (classrooms, villages, firms) are assigned by
complete randomization of `n_treated_clusters` of the clusters, and every unit of a
treated cluster is treated.

The effective number of randomized units is the number of clusters, not the number
of individuals. Randomization inference re-randomizes clusters, and the studentized
statistics of [`randomization_test`](@ref) use cluster-level variance estimates, so
inference remains valid with few clusters when it is based on the exact
randomization distribution.

# Arguments
- `clusters::AbstractVector`: one cluster label per unit.
- `n_treated_clusters::Integer`: number of treated clusters.
- `z::AbstractVector{Bool}`: alternatively, an observed assignment, which must be
  constant within clusters; the number of treated clusters is taken from it.

# Returns
- `ClusterRandomization`.

# Examples
```julia
using DrSnow, StableRNGs
clusters = repeat(1:6; inner=3)
m = ClusterRandomization(clusters, 3)
n_assignments(m)                             # 20
draw_assignment(StableRNG(4), m)
```

# References
- Imbens, G. W., & Rubin, D. B. (2015). *Causal Inference for Statistics, Social,
  and Biomedical Sciences: An Introduction*. Cambridge University Press.
"""
struct ClusterRandomization <: AssignmentMechanism
    n::Int
    groups::Vector{Vector{Int}}
    n_treated_clusters::Int
end

function ClusterRandomization(clusters::AbstractVector, n_treated_clusters::Integer)
    labels = unique(clusters)
    groups = [findall(==(l), clusters) for l in labels]
    0 <= n_treated_clusters <= length(groups) ||
        throw(ArgumentError("n_treated_clusters must be between 0 and #clusters"))
    return ClusterRandomization(length(clusters), groups, Int(n_treated_clusters))
end

function ClusterRandomization(clusters::AbstractVector, z::AbstractVector{Bool})
    labels = unique(clusters)
    nt = 0
    for l in labels
        zs = z[clusters .== l]
        all(==(zs[1]), zs) || throw(ArgumentError("assignment varies within cluster $l"))
        nt += zs[1]
    end
    return ClusterRandomization(clusters, nt)
end

"""
    CustomAssignment(n, sampler; probabilities=nothing)

Arbitrary assignment mechanism defined by a sampling function: `sampler(rng)` must
return a length-`n` vector of `Bool`s (or 0/1 values) drawn from the design.

Use it for designs not covered by the built-in types (for example a
covariate-adaptive or a constrained randomization), provided the sampler reproduces
the procedure that was actually used. Exact enumeration is not available, so
randomization inference uses Monte Carlo draws. Marginal treatment probabilities are
taken from `probabilities` when given and are otherwise estimated by simulation in
[`treatment_probabilities`](@ref).

# Arguments
- `n::Integer`: number of units.
- `sampler`: function `rng -> AbstractVector{Bool}`.

# Keywords
- `probabilities`: optional vector of known marginal treatment probabilities.

# Returns
- `CustomAssignment`.

# Examples
```julia
using DrSnow, Random, StableRNGs
# treat 3 of the first 5 and 2 of the last 5 units, but never units 1 and 2 together
function sampler(rng)
    while true
        z = vcat(shuffle(rng, [1, 1, 1, 0, 0]), shuffle(rng, [1, 1, 0, 0, 0])) .== 1
        !(z[1] && z[2]) && return z
    end
end
m = CustomAssignment(10, sampler)
treatment_probabilities(m; rng=StableRNG(5), draws=2_000)
```
"""
struct CustomAssignment{F} <: AssignmentMechanism
    n::Int
    sampler::F
    probabilities::Union{Nothing,Vector{Float64}}
end
CustomAssignment(n::Integer, sampler; probabilities=nothing) =
    CustomAssignment(Int(n), sampler, probabilities === nothing ? nothing :
                     float.(collect(probabilities)))

n_units(m::BernoulliAssignment) = length(m.p)
n_units(m::Union{CompleteRandomization,StratifiedRandomization,ClusterRandomization,
                 CustomAssignment}) = m.n

"""
    draw_assignment(rng, m::AssignmentMechanism) -> BitVector
    draw_assignment(m::AssignmentMechanism) -> BitVector

Draw one treatment vector from the assignment mechanism `m`.

Draws are independent across calls and reproducible given the state of `rng`. The
randomization-inference functions draw per-chunk seeds up front, so their results
do not depend on the number of threads. The second form uses
`Random.default_rng()`.

# Arguments
- `rng::AbstractRNG`: random number generator.
- `m::AssignmentMechanism`: the design.

# Returns
- `BitVector` of length [`n_units`](@ref)`(m)`, `true` for treated units, in the
  unit order of `m`.

# Examples
```julia
using DrSnow, StableRNGs
draw_assignment(StableRNG(1), MatchedPairsRandomization([1, 1, 2, 2, 3, 3]))
```
"""
draw_assignment(rng::AbstractRNG, m::BernoulliAssignment) = rand(rng, length(m.p)) .< m.p

function draw_assignment(rng::AbstractRNG, m::CompleteRandomization)
    z = falses(m.n)
    z[randperm(rng, m.n)[1:m.n_treated]] .= true
    return z
end

function draw_assignment(rng::AbstractRNG, m::StratifiedRandomization)
    z = falses(m.n)
    for (g, k) in zip(m.groups, m.n_treated)
        z[g[randperm(rng, length(g))[1:k]]] .= true
    end
    return z
end

function draw_assignment(rng::AbstractRNG, m::ClusterRandomization)
    z = falses(m.n)
    for c in randperm(rng, length(m.groups))[1:m.n_treated_clusters]
        z[m.groups[c]] .= true
    end
    return z
end

function draw_assignment(rng::AbstractRNG, m::CustomAssignment)
    z = m.sampler(rng)
    length(z) == m.n || throw(DimensionMismatch("sampler returned length $(length(z)), " *
                                                "expected $(m.n)"))
    return BitVector(z .== true)
end

draw_assignment(m::AssignmentMechanism) = draw_assignment(Random.default_rng(), m)

"""
    treatment_probabilities(m; rng=Random.default_rng(), draws=10_000)
        -> Vector{Float64}

Marginal treatment probability ``π_i = P(Z_i = 1)`` of every unit under the design
`m`.

The probabilities are exact (closed form) for Bernoulli, complete, stratified,
cluster, blocked-cluster, matched-pairs and two-stage designs, and for a
[`CustomAssignment`](@ref) with supplied `probabilities`. For other mechanisms
(rerandomization, custom samplers) they are Monte Carlo frequencies from `draws`
simulated assignments, with standard error about ``\\sqrt{π_i(1 - π_i)/draws}``; use
more draws when the probabilities enter inverse-probability weights.

# Arguments
- `m::AssignmentMechanism`: the design.

# Keywords
- `rng::AbstractRNG`: random number generator for the Monte Carlo case.
- `draws::Int`: number of simulated assignments in the Monte Carlo case; default
  10 000.

# Returns
- `Vector{Float64}` of length `n_units(m)`.

# Examples
```julia
using DrSnow, StableRNGs
X = randn(StableRNG(1), 20, 2)
m = Rerandomization(CompleteRandomization(20, 10), X; threshold=0.45)
treatment_probabilities(CompleteRandomization(20, 10))[1]         # 0.5 (exact)
treatment_probabilities(m; rng=StableRNG(2), draws=2_000)         # Monte Carlo
```
"""
treatment_probabilities(m::BernoulliAssignment; kwargs...) = copy(m.p)
treatment_probabilities(m::CompleteRandomization; kwargs...) =
    fill(m.n_treated / m.n, m.n)
function treatment_probabilities(m::StratifiedRandomization; kwargs...)
    p = zeros(m.n)
    for (g, k) in zip(m.groups, m.n_treated)
        p[g] .= k / length(g)
    end
    return p
end
treatment_probabilities(m::ClusterRandomization; kwargs...) =
    fill(m.n_treated_clusters / length(m.groups), m.n)
function treatment_probabilities(m::AssignmentMechanism;
                                 rng::AbstractRNG=Random.default_rng(), draws::Int=10_000)
    m isa CustomAssignment && m.probabilities !== nothing && return copy(m.probabilities)
    acc = zeros(n_units(m))
    for _ in 1:draws
        acc .+= draw_assignment(rng, m)
    end
    return acc ./ draws
end
