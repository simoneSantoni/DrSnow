# Partial interference and two-stage randomized (saturation) designs
# (Hudgens & Halloran 2008; Baird, Bohren, McIntosh & Özler 2018).

"""
    TwoStageRandomization(groups, saturations, n_groups)

Two-stage randomized (randomized saturation) design for partial interference: groups
are first randomized to treatment saturations, then units within each group are
randomized to treatment at the group's saturation.

Stage 1 completely randomizes the groups (one label per unit in `groups`) to the
saturation levels, with `n_groups[k]` groups receiving `saturations[k]`. Stage 2
treats, in a group of size ``m`` assigned saturation ``α``, exactly
``\\text{round}(α m)`` units, all subsets of that size being equally likely. By
varying the share of treated units across groups, the design identifies spillover
effects on untreated units and how direct effects vary with the saturation, under
partial interference (Hudgens and Halloran 2008; Baird, Bohren, McIntosh and Özler
2018). A pure-control saturation ``α = 0`` provides groups with no treated units,
which identify the untreated counterfactual without spillovers. Being an
[`AssignmentMechanism`](@ref), the design can be used with
[`exposure_probabilities`](@ref), [`spillover_fisher_test`](@ref) and the other
design-based functions.

# Arguments
- `groups::AbstractVector`: one group label per unit.
- `saturations::AbstractVector`: distinct saturation levels in ``[0, 1]``.
- `n_groups::AbstractVector{<:Integer}`: number of groups assigned to each level;
  they must sum to the number of groups.

# Returns
- `TwoStageRandomization`.

# Examples
```julia
using DrSnow, StableRNGs
village = repeat(1:40; inner=10)
design = TwoStageRandomization(village, [0.0, 0.3, 0.7], [10, 15, 15])
z = draw_assignment(StableRNG(1), design)
treatment_probabilities(design)[1]           # (10·0 + 15·0.3 + 15·0.7) / 40 = 0.375
```

# References
- Hudgens, M. G., & Halloran, M. E. (2008). Toward causal inference with
  interference. *Journal of the American Statistical Association*, 103(482),
  832–842.
- Baird, S., Bohren, J. A., McIntosh, C., & Özler, B. (2018). Optimal design of
  experiments in the presence of interference. *Review of Economics and Statistics*,
  100(5), 844–860.
"""
struct TwoStageRandomization <: AssignmentMechanism
    n::Int
    groups::Vector{Vector{Int}}
    saturations::Vector{Float64}
    n_groups::Vector{Int}
end

function TwoStageRandomization(groups::AbstractVector, saturations::AbstractVector,
                               n_groups::AbstractVector{<:Integer})
    length(saturations) == length(n_groups) ||
        throw(DimensionMismatch("one group count per saturation level required"))
    all(a -> 0 <= a <= 1, saturations) ||
        throw(ArgumentError("saturations must lie in [0, 1]"))
    allunique(saturations) || throw(ArgumentError("saturation levels must be distinct"))
    labels = unique(groups)
    sum(n_groups) == length(labels) ||
        throw(ArgumentError("n_groups must sum to the number of groups " *
                            "($(length(labels)))"))
    all(>=(0), n_groups) || throw(ArgumentError("n_groups must be non-negative"))
    members = [findall(==(l), groups) for l in labels]
    return TwoStageRandomization(length(groups), members, Float64.(collect(saturations)),
                                 Int.(collect(n_groups)))
end

n_units(m::TwoStageRandomization) = m.n

function draw_assignment(rng::AbstractRNG, m::TwoStageRandomization)
    z = falses(m.n)
    order = randperm(rng, length(m.groups))
    pos = 0
    for (a, k) in zip(m.saturations, m.n_groups), _ in 1:k
        pos += 1
        g = m.groups[order[pos]]
        nt = round(Int, a * length(g))
        z[g[randperm(rng, length(g))[1:nt]]] .= true
    end
    return z
end

function treatment_probabilities(m::TwoStageRandomization; kwargs...)
    G = length(m.groups)
    p = zeros(m.n)
    for g in m.groups
        sz = length(g)
        p[g] .= sum(k / G * round(Int, a * sz) / sz
                    for (a, k) in zip(m.saturations, m.n_groups))
    end
    return p
end

"""
    TwoStageEffects <: CausalEstimate

Result of [`two_stage_effects`](@ref): direct, indirect (spillover), total and
overall effects from a two-stage randomized design.

For saturation levels ``α`` and the reference level ``α_0``, the coefficients are
- `"direct(α)"` ``= \\bar Y(1; α) - \\bar Y(0; α)``;
- `"indirect(α vs α₀)"` ``= \\bar Y(0; α) - \\bar Y(0; α_0)``, the spillover on
  untreated units;
- `"total(α vs α₀)"` ``= \\bar Y(1; α) - \\bar Y(0; α_0)``;
- `"overall(α vs α₀)"` ``= \\bar Y(α) - \\bar Y(α_0)``,
where ``\\bar Y(z; α)`` is the average over groups of the group-mean potential
outcome of units with treatment ``z`` when the group receives saturation ``α``, and
``\\bar Y(α)`` the average group-mean outcome. Signs are "treated minus control" and
"higher minus reference saturation"; Hudgens and Halloran (2008) define the direct
and indirect effects with the opposite sign.

# Fields
- `coef::Vector{Float64}`, `vcov::Matrix{Float64}`, `names::Vector{String}`:
  estimates, covariance and names.
- `means::DataFrame`: per saturation level, the number of groups and the estimated
  treated, control and overall means.
- `n::Int`: number of units; `n_groups::Int`: number of groups.
- `dof::Int`: degrees of freedom of the t reference, ``\\min_α C_α - 1`` (a
  heuristic; see [`two_stage_effects`](@ref)).

# Accessors
- The [`CausalEstimate`](@ref) interface; `dof_residual(r)` returns `dof`.
"""
struct TwoStageEffects <: CausalEstimate
    coef::Vector{Float64}
    vcov::Matrix{Float64}
    names::Vector{String}
    means::DataFrame
    n::Int
    n_groups::Int
    dof::Int
end

StatsAPI.coef(r::TwoStageEffects) = r.coef
StatsAPI.vcov(r::TwoStageEffects) = r.vcov
StatsAPI.coefnames(r::TwoStageEffects) = r.names
StatsAPI.nobs(r::TwoStageEffects) = r.n
StatsAPI.dof_residual(r::TwoStageEffects) = r.dof
estimand(::TwoStageEffects) = "direct, indirect, total and overall effects " *
                              "(group-averaged, two-stage randomization)"
method_name(::TwoStageEffects) = "Two-stage randomization estimator (Hudgens–Halloran)"
function show_details(io::IO, r::TwoStageEffects)
    println(io)
    println(io, r.n_groups, " groups. Variance: between-group estimator (conservative).")
end

"""
    two_stage_effects(data, outcome, treatment; group, saturation, reference=nothing)
        -> TwoStageEffects

Direct, indirect (spillover), total and overall effects under partial interference,
estimated from a two-stage randomized design in which groups are randomized to
treatment saturations and units within groups are randomized to treatment
(Hudgens and Halloran 2008).

Under partial interference a unit's potential outcome depends on its own treatment
and on the treatments in its group. Hudgens and Halloran (2008) define effects as
averages over the within-group assignment distribution implied by a saturation
``α``: ``\\bar Y(z; α)`` is the average, over groups, of the mean potential outcome of
units with treatment ``z`` in a group assigned saturation ``α``. The direct effect
``\\bar Y(1; α) - \\bar Y(0; α)`` compares treated and untreated units at the same
saturation; the indirect effect ``\\bar Y(0; α) - \\bar Y(0; α_0)`` is the spillover
on untreated units of raising the saturation from ``α_0`` to ``α``; the total effect
combines both, and the overall effect compares average outcomes of groups at the two
saturations. These estimands are policy-relevant but depend on the saturations and
on the within-group assignment mechanism of the design; they are not unit-level
effects of a single neighbour's treatment.

Estimates are averages of group-level means: for each group, the mean outcome of its
treated units, of its untreated units and of all units; for each saturation level,
the average over the groups assigned to it, which is unbiased for the averages
defined above under the two-stage design. The variance of each level's mean vector
is estimated by the between-group sample covariance divided by the number of groups
``C_α``, treating levels as independent. This is the classical two-stage-sampling
estimator, which is conservative in the design-based sense: its expectation exceeds
the true variance by the between-group variance of the group-level targets divided
by the total number of groups, the analogue of Neyman's conservative variance for
the difference in means. Intervals use a Student-t distribution with
``\\min_α C_α - 1`` degrees of freedom; this is a heuristic small-sample
adjustment, not a derived reference distribution, and coverage with very few groups
per level is not guaranteed. Liu and Hudgens (2014) give large-sample randomization
inference for these estimands, and Basse and Feller (2018) discuss alternative
estimators and variance estimators for two-stage designs. At least two groups per
saturation level are required.

# Arguments
- `data`: table with one row per unit; missing values are not allowed in the
  columns used.
- `outcome::Symbol`: outcome column.
- `treatment::Symbol`: binary treatment column.

# Keywords
- `group::Symbol`: group (cluster) identifier.
- `saturation::Symbol`: the saturation assigned to each unit's group (constant
  within group).
- `reference`: reference saturation ``α_0`` for the indirect, total and overall
  effects; default the lowest level.

# Returns
- [`TwoStageEffects`](@ref).

# Examples
```julia
using DrSnow, DataFrames, StableRNGs
rng = StableRNG(2)
village = repeat(1:40; inner=10)
design = TwoStageRandomization(village, [0.0, 0.3, 0.7], [10, 15, 15])
z = draw_assignment(rng, design)
sat = [round(count(z[village .== v]) / 10; digits=1) for v in village]
df = DataFrame(village=village, sat=sat, z=Int.(z))
df.y = 1.0 .* df.z .+ 0.5 .* df.sat .* (1 .- df.z) .+ randn(rng, nrow(df))
r = two_stage_effects(df, :y, :z; group=:village, saturation=:sat)
confint(r)
```

# References
- Hudgens, M. G., & Halloran, M. E. (2008). Toward causal inference with
  interference. *Journal of the American Statistical Association*, 103(482),
  832–842.
- Liu, L., & Hudgens, M. G. (2014). Large sample randomization inference of causal
  effects in the presence of interference. *Journal of the American Statistical
  Association*, 109(505), 288–301.
- Basse, G., & Feller, A. (2018). Analyzing two-stage experiments in the presence of
  interference. *Journal of the American Statistical Association*, 113, 41–55.
- Tchetgen Tchetgen, E. J., & VanderWeele, T. J. (2012). On causal inference in the
  presence of interference. *Statistical Methods in Medical Research*, 21(1),
  55–75.
- Baird, S., Bohren, J. A., McIntosh, C., & Özler, B. (2018). Optimal design of
  experiments in the presence of interference. *Review of Economics and Statistics*,
  100(5), 844–860.
"""
function two_stage_effects(data, outcome::Symbol, treatment::Symbol; group::Symbol,
                           saturation::Symbol, reference=nothing)
    context = "two_stage_effects"
    require_columns(data, [outcome, treatment, group, saturation]; context=context)
    df = DataFrame(data; copycols=false)
    for c in (outcome, treatment, group, saturation)
        any(ismissing, df[!, c]) &&
            throw(ArgumentError("$context: `$c` has missing values"))
    end
    z = [_sv_binary_value(v, context) for v in df[!, treatment]]
    y = Float64.(df[!, outcome])
    all(isfinite, y) || throw(ArgumentError("$context: non-finite outcomes"))
    gcol = df[!, group]
    labels = unique(gcol)
    gpos = Dict(l => k for (k, l) in enumerate(labels))
    G = length(labels)
    sat = fill(NaN, G)
    s1 = zeros(G); n1 = zeros(Int, G); s0 = zeros(G); n0 = zeros(Int, G)
    for r in eachindex(y)
        g = gpos[gcol[r]]
        a = Float64(df[r, saturation])
        if isnan(sat[g])
            sat[g] = a
        elseif sat[g] != a
            throw(ArgumentError("$context: saturation varies within group " *
                                "$(repr(gcol[r]))"))
        end
        if z[r] == 1
            s1[g] += y[r]; n1[g] += 1
        else
            s0[g] += y[r]; n0[g] += 1
        end
    end
    levels = sort(unique(sat))
    length(levels) >= 1 || throw(ArgumentError("$context: no saturation levels"))
    ref = reference === nothing ? levels[1] : Float64(reference)
    ref in levels || throw(ArgumentError("$context: reference saturation $ref not found"))
    # group-level quantities: [treated mean, control mean, overall mean]
    q = fill(NaN, G, 3)
    for g in 1:G
        n1[g] > 0 && (q[g, 1] = s1[g] / n1[g])
        n0[g] > 0 && (q[g, 2] = s0[g] / n0[g])
        q[g, 3] = (s1[g] + s0[g]) / (n1[g] + n0[g])
    end
    # per level: mean vector over groups and its covariance (NaN where undefined)
    L = length(levels)
    M = fill(NaN, L, 3)
    Vs = [fill(NaN, 3, 3) for _ in 1:L]
    ng = zeros(Int, L)
    for (k, a) in enumerate(levels)
        gs = findall(==(a), sat)
        ng[k] = length(gs)
        ng[k] >= 2 || throw(ArgumentError("$context: saturation $a has $(ng[k]) " *
                                          "group(s); " *
                                          "at least two are needed for variance " *
                                          "estimation"))
        for c in 1:3
            vals = q[gs, c]
            if all(isfinite, vals)
                M[k, c] = mean(vals)
            elseif any(isfinite, vals)
                throw(ArgumentError("$context: some but not all groups with saturation " *
                                    "$a have units with " *
                                    (c == 1 ? "treatment" : "no treatment") *
                                    "; group-level means are undefined"))
            end
        end
        ok = [c for c in 1:3 if isfinite(M[k, c])]
        Vs[k][ok, ok] = cov(q[gs, ok]) ./ ng[k]
    end
    # coefficients as linear combinations of the stacked level means
    idx(k, c) = 3 * (k - 1) + c
    rows = Tuple{String,Vector{Tuple{Int,Float64}}}[]
    fmt(a) = _sv_fmt(a)
    kref = findfirst(==(ref), levels)
    for (k, a) in enumerate(levels)
        isfinite(M[k, 1]) && isfinite(M[k, 2]) &&
            push!(rows, ("direct($(fmt(a)))", [(idx(k, 1), 1.0), (idx(k, 2), -1.0)]))
    end
    for (k, a) in enumerate(levels)
        k == kref && continue
        tag = "($(fmt(a)) vs $(fmt(ref)))"
        isfinite(M[k, 2]) && isfinite(M[kref, 2]) &&
            push!(rows, ("indirect" * tag, [(idx(k, 2), 1.0), (idx(kref, 2), -1.0)]))
        isfinite(M[k, 1]) && isfinite(M[kref, 2]) &&
            push!(rows, ("total" * tag, [(idx(k, 1), 1.0), (idx(kref, 2), -1.0)]))
        push!(rows, ("overall" * tag, [(idx(k, 3), 1.0), (idx(kref, 3), -1.0)]))
    end
    isempty(rows) && throw(ArgumentError("$context: no estimable effect (need treated " *
                                         "and untreated units, or two saturation levels)"))
    m = vec(permutedims(M))                    # stacked [level1 c1..c3, level2 …]
    Vbig = zeros(3L, 3L)
    for k in 1:L
        r = (3k - 2):(3k)
        Vbig[r, r] .= replace(Vs[k], NaN => 0.0)
    end
    Lmat = zeros(length(rows), 3L)
    for (i, (_, terms)) in enumerate(rows), (j, w) in terms
        Lmat[i, j] += w
    end
    b = [sum(w * m[j] for (j, w) in terms) for (_, terms) in rows]
    V = Matrix(Symmetric(Lmat * Vbig * transpose(Lmat)))
    means = DataFrame(saturation=levels, n_groups=ng,
                      mean_treated=_sv_nan_to_missing.(M[:, 1]),
                      mean_control=_sv_nan_to_missing.(M[:, 2]),
                      mean_overall=M[:, 3])
    return TwoStageEffects(b, V, first.(rows), means, length(y), G, minimum(ng) - 1)
end
