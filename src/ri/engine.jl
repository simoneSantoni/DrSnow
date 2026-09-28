# Shared machinery: build the design (mechanism + observed assignment) from data,
# construct the reference set of assignments (exact enumeration or Monte Carlo),
# evaluate statistics over it (optionally multithreaded, reproducibly), and compute
# randomization p-values.
#
# Reference-set convention. Exact: every assignment in the support with its
# probability (the observed assignment is one of them). Monte Carlo: the observed
# assignment followed by B independent draws, all with weight one. In both cases a
# p-value is the weighted share of the reference set at least as extreme as the
# observed statistic, which gives (1 + #extreme) / (1 + B) under Monte Carlo.

struct _ri_Design{M<:AssignmentMechanism}
    mech::M
    z::BitVector          # observed assignment in unit order
    rows::Vector{Int}     # data row of each unit
    ctx::_ri_Context
end

function _ri_check_complete(data, cols, context)
    for c in cols
        c === nothing && continue
        any(ismissing, data[!, c]) &&
            throw(ArgumentError("$context: column $c has missing values; remove or " *
                                "impute them explicitly (dropping units changes the " *
                                "assignment mechanism)"))
    end
end

function _ri_binary(v::AbstractVector, name, context)
    out = falses(length(v))
    for (i, x) in enumerate(v)
        if x == 1
            out[i] = true
        elseif x != 0
            throw(ArgumentError("$context: treatment column $name must be 0/1 or Bool"))
        end
    end
    return out
end

_ri_numeric_matrix(data, cols::Vector{Symbol}, rows) =
    isempty(cols) ? zeros(length(rows), 0) :
    reduce(hcat, [Vector{Float64}(data[rows, c]) for c in cols])

# Canonical unit order when the design is given by columns: sorting by every column
# the analysis uses makes results (including Monte Carlo draws) invariant to the
# order of the data rows; rows equal on all these columns are interchangeable.
function _ri_canonical_rows(data, cols)
    n = nrow(data)
    isempty(cols) && return collect(1:n)
    keys = [Tuple(_ri_sortkey(data[i, c]) for c in cols) for i in 1:n]
    return sortperm(keys)
end

"""
Internal: build the randomization design for a binary treatment.
`mechanism` (an `AssignmentMechanism` on units = rows, or rows sorted by `id`) and
the column-based `strata` / `cluster` keywords are mutually exclusive.
"""
function _ri_design(data, treatment::Symbol; mechanism=nothing, strata=nothing,
                    cluster=nothing, id=nothing, covariates::Vector{Symbol}=Symbol[],
                    sortcols::Vector{Symbol}=Symbol[], context::String="randomization")
    cols = Any[treatment, strata, cluster, id]
    append!(cols, covariates)
    append!(cols, sortcols)
    require_columns(data, cols; context=context)
    _ri_check_complete(data, cols, context)
    n = nrow(data)
    n >= 2 || throw(ArgumentError("$context: need at least two units"))
    if mechanism !== nothing
        mechanism isa AssignmentMechanism ||
            throw(ArgumentError("$context: mechanism must be an AssignmentMechanism"))
        (strata === nothing && cluster === nothing) ||
            throw(ArgumentError("$context: give either `mechanism` or `strata`/`cluster`" *
                                ", not both"))
        n_units(mechanism) == n ||
            throw(DimensionMismatch("$context: mechanism has $(n_units(mechanism)) " *
                                    "units but data has $n rows"))
    end
    rows = if id !== nothing
        allunique(data[!, id]) || throw(ArgumentError("$context: id column $id " *
                                                      "has duplicates"))
        sortperm(data[!, id]; by=_ri_sortkey)
    elseif mechanism !== nothing
        collect(1:n)
    else
        canon = Symbol[c for c in (cluster, strata, treatment) if c !== nothing]
        append!(canon, sortcols)
        append!(canon, covariates)
        _ri_canonical_rows(data, unique(canon))
    end
    z = _ri_binary(data[rows, treatment], treatment, context)
    (any(z) && !all(z)) ||
        throw(ArgumentError("$context: both treated and control units are required"))
    mech = mechanism !== nothing ? mechanism :
           _ri_mechanism_from_columns(data, rows, z, strata, cluster, context)
    _ri_in_support(mech, z) ||
        throw(ArgumentError("$context: the observed assignment has probability zero " *
                            "under the stated mechanism ($(_ri_describe(mech)))"))
    X = _ri_numeric_matrix(data, covariates, rows)
    return _ri_Design(mech, z, rows, _ri_context(mech, X))
end

function _ri_mechanism_from_columns(data, rows, z, strata, cluster, context)
    n = length(rows)
    if cluster !== nothing
        cl = data[rows, cluster]
        for (l, g) in _ri_groups(cl)
            all(==(z[g[1]]), view(z, g)) ||
                throw(ArgumentError("$context: treatment varies within cluster $l"))
        end
        if strata !== nothing
            return BlockClusterRandomization(cl, data[rows, strata], z)
        end
        groups = [g for (_, g) in _ri_groups(cl)]
        return ClusterRandomization(n, groups, count(z[g[1]] for g in groups))
    elseif strata !== nothing
        groups = [g for (_, g) in _ri_groups(data[rows, strata])]
        return StratifiedRandomization(n, groups, [count(view(z, g)) for g in groups])
    end
    return CompleteRandomization(n, count(z))
end

# ---------------------------------------------------------------------------------
# Reference set
# ---------------------------------------------------------------------------------

const _RI_CHUNK = 64

struct _ri_Plan
    exact::Bool
    zs::Vector{BitVector}     # exact: the support
    w::Vector{Float64}        # exact: probabilities
    seeds::Vector{UInt64}     # Monte Carlo: one seed per chunk of draws
    B::Int                    # Monte Carlo draws (exact: support size)
end

function _ri_plan(mech::AssignmentMechanism, nperm::Integer, exact, rng::AbstractRNG)
    nperm >= 1 || throw(ArgumentError("nperm must be positive"))
    cost = _ri_enumeration_cost(mech)
    use_exact = if exact === :auto
        cost !== nothing && cost <= nperm
    elseif exact === true
        cost === nothing &&
            throw(ArgumentError("exact enumeration is not available for " *
                                "$(nameof(typeof(mech)))"))
        cost <= _RI_MAX_ENUMERATE ||
            throw(ArgumentError("exact enumeration would visit $cost assignments " *
                                "(limit $_RI_MAX_ENUMERATE); use Monte Carlo"))
        true
    elseif exact === false
        false
    else
        throw(ArgumentError("exact must be :auto, true or false"))
    end
    if use_exact
        zs, w = enumerate_assignments(mech)
        return _ri_Plan(true, zs, w, UInt64[], length(zs))
    end
    return _ri_Plan(false, BitVector[], Float64[], task_seeds(rng, cld(nperm, _RI_CHUNK)),
                    Int(nperm))
end

_ri_nrows(p::_ri_Plan) = p.exact ? length(p.zs) : p.B + 1
_ri_weights(p::_ri_Plan) = p.exact ? p.w : ones(p.B + 1)

# Evaluate `f(z) -> Real or vector of length K` on every assignment of the reference
# set. Results are placed by index, and Monte Carlo chunk c always uses
# Xoshiro(seeds[c]), so the output does not depend on the number of threads.
function _ri_map(f, plan::_ri_Plan, mech::AssignmentMechanism, zobs::BitVector, K::Int;
                 threaded::Bool=Threads.nthreads() > 1)
    plan.exact && return _ri_map_list(f, plan.zs, K; threaded)
    return _ri_map_mc(f, r -> draw_assignment(r, mech), zobs, plan.seeds, plan.B, K;
                      threaded)
end

# Monte Carlo evaluation with an arbitrary draw function `draw(rng)`; row 1 holds
# f(obs), rows 2..B+1 the draws.
function _ri_map_mc(f, draw, obs, seeds::Vector{UInt64}, B::Int, K::Int;
                    threaded::Bool)
    out = Matrix{Float64}(undef, B + 1, K)
    out[1, :] .= f(obs)
    body = c -> begin
        r = Xoshiro(seeds[c])
        for b in ((c - 1) * _RI_CHUNK + 1):min(c * _RI_CHUNK, B)
            out[b + 1, :] .= f(draw(r))
        end
    end
    _ri_foreach(body, length(seeds), threaded)
    return out
end

# The assignments of the reference set (materialized; Monte Carlo draws use the same
# seeds as `_ri_map`, so they coincide with the draws it evaluates).
function _ri_assignments(plan::_ri_Plan, mech::AssignmentMechanism, zobs::BitVector)
    plan.exact && return plan.zs
    zs = Vector{BitVector}(undef, plan.B + 1)
    zs[1] = zobs
    for c in eachindex(plan.seeds)
        r = Xoshiro(plan.seeds[c])
        for b in ((c - 1) * _RI_CHUNK + 1):min(c * _RI_CHUNK, plan.B)
            zs[b + 1] = draw_assignment(r, mech)
        end
    end
    return zs
end

# Evaluate f over a materialized list of assignments.
function _ri_map_list(f, zs::Vector{BitVector}, K::Int; threaded::Bool)
    M = length(zs)
    out = Matrix{Float64}(undef, M, K)
    body = c -> begin
        for i in ((c - 1) * _RI_CHUNK + 1):min(c * _RI_CHUNK, M)
            out[i, :] .= f(zs[i])
        end
    end
    _ri_foreach(body, cld(M, _RI_CHUNK), threaded)
    return out
end

function _ri_foreach(body, nch::Int, threaded::Bool)
    if threaded && Threads.nthreads() > 1 && nch > 1
        Threads.@threads for c in 1:nch
            body(c)
        end
    else
        for c in 1:nch
            body(c)
        end
    end
    return nothing
end

# ---------------------------------------------------------------------------------
# p-values
# ---------------------------------------------------------------------------------

_ri_tol(x) = 1e-9 * max(1.0, abs(x))

@inline function _ri_extreme(v, obs, alternative::Symbol, tol)
    alternative === :two_sided && return abs(v) >= abs(obs) - tol
    alternative === :greater && return v >= obs - tol
    return v <= obs + tol
end

function _ri_check_alternative(alternative::Symbol)
    alternative in (:two_sided, :greater, :less) ||
        throw(ArgumentError("alternative must be :two_sided, :greater or :less"))
end

# Weighted share of the reference set at least as extreme as `obs`; rows where the
# statistic is undefined (NaN) are dropped. Returns (p, total weight, #dropped).
function _ri_pvalue(obs::Real, vals::AbstractVector, w::AbstractVector,
                    alternative::Symbol)
    tol = _ri_tol(obs)
    num = 0.0
    den = 0.0
    dropped = 0
    for (v, wi) in zip(vals, w)
        if isnan(v)
            dropped += 1
            continue
        end
        den += wi
        _ri_extreme(v, obs, alternative, tol) && (num += wi)
    end
    den > 0 || error("the test statistic is undefined for every assignment")
    return min(1.0, num / den), den, dropped
end

_ri_mc_se(p, B) = B > 0 ? sqrt(p * (1 - p) / B) : 0.0

function _ri_method_string(plan::_ri_Plan, dropped::Int=0)
    s = plan.exact ?
        "randomization inference, exact enumeration of $(plan.B) assignments" :
        "randomization inference, Monte Carlo with $(plan.B) draws"
    dropped > 0 && (s *= " ($dropped assignments with undefined statistic excluded)")
    return s
end

# Outcome vector (in unit order) adjusted for a sharp null of additive effects.
function _ri_adjusted_outcome(data, outcome::Symbol, design::_ri_Design, tau0)
    y = Vector{Float64}(data[design.rows, outcome])
    if tau0 isa Symbol
        tau = Vector{Float64}(data[design.rows, tau0])
        return y .- tau .* design.z
    end
    return y .- float(tau0) .* design.z
end
