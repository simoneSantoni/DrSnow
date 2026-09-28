# Blocked and matched-pair designs formed on a prognostic score or on covariates.

"""
    BlockingDesign

A blocked (stratified) or matched-pair randomized design, with its assignment
mechanism; the result of [`block_design`](@ref).

The object ties the three stages of a blocked experiment together. It drives the
randomization ([`assign_treatment`](@ref)), design-based inference
(`randomization_test(...; mechanism=bd.mechanism, id=bd.id)`) and estimation
([`experiment_estimate`](@ref)), so that the analysis uses exactly the blocks, and
the prognostic score, that determined the assignment. Units are identified by key
(`ids`), so the design can be joined back to data in any row order.

# Fields
- `ids::Vector`: unit identifiers in the design's unit order (sorted by id when an `id`
  column was given, otherwise the data's row numbers).
- `id::Union{Nothing,Symbol}`: identifier column, or `nothing`.
- `blocks::Vector{Int}`: block label (1, 2, …) of each unit.
- `mechanism::AssignmentMechanism`: a `MatchedPairsRandomization` (pairs with one
  treated unit each) or a `StratifiedRandomization` (complete randomization of
  `n_treated[b]` units within block `b`), on the units in `ids` order.
- `n_treated::Vector{Int}`: number of treated units per block.
- `block_sizes::Vector{Int}`: number of units per block.
- `score::Union{Nothing,Vector{Float64}}`: the prognostic score used, in `ids` order
  (`nothing` for purely covariate-based designs).
- `r2::Float64`: out-of-sample R² of the score (`NaN` if unknown).
- `method::Symbol`: `:pairs` or `:blocks`.
- `algorithm::Symbol`: `:optimal`, `:greedy`, or `:quantile` for consecutive blocks of
  the sorted score.
- `distance::Symbol`: `:score` or `:mahalanobis`.
- `objective::Float64`: total within-block distance, the sum over blocks of the
  pairwise distances between block members.
- `covariates::Vector{Symbol}`: covariates of the Mahalanobis distance.
"""
struct BlockingDesign
    ids::Vector{Any}
    id::Union{Nothing,Symbol}
    blocks::Vector{Int}
    mechanism::AssignmentMechanism
    n_treated::Vector{Int}
    block_sizes::Vector{Int}
    score::Union{Nothing,Vector{Float64}}
    r2::Float64
    method::Symbol
    algorithm::Symbol
    distance::Symbol
    objective::Float64
    covariates::Vector{Symbol}
end

n_units(bd::BlockingDesign) = length(bd.ids)

function Base.show(io::IO, ::MIME"text/plain", bd::BlockingDesign)
    what = bd.method === :pairs ? "Matched-pair design" : "Blocked design"
    println(io, what, " (", bd.algorithm, " ", bd.method === :pairs ? "matching" :
                                                   "blocking", " on ",
            bd.distance === :score ? "the prognostic score" :
            "Mahalanobis distance of " * join(bd.covariates, ", "), ")")
    sz = extrema(bd.block_sizes)
    @printf(io, "Units: %d in %d blocks (sizes %d–%d); treated: %d\n", length(bd.ids),
            length(bd.block_sizes), sz[1], sz[2], sum(bd.n_treated))
    @printf(io, "Total within-block distance: %.6g\n", bd.objective)
    isnan(bd.r2) || @printf(io, "Out-of-sample R² of the score: %.4f\n", bd.r2)
    print(io, "Assignment mechanism: ", _ri_describe(bd.mechanism))
end

Base.show(io::IO, bd::BlockingDesign) =
    print(io, "BlockingDesign($(length(bd.ids)) units, $(length(bd.block_sizes)) blocks)")

# Units in canonical order: sorted by id, or row order without id.
function _des_unit_rows(data, id, ctx)
    if id === nothing
        return collect(1:nrow(data)), Any[1:nrow(data)...]
    end
    require_columns(data, [id]; context=ctx)
    v = data[!, id]
    any(ismissing, v) && throw(ArgumentError("$ctx: id column $id has missing values"))
    allunique(v) || throw(ArgumentError("$ctx: id column $id has duplicates"))
    rows = sortperm(v; by=_ri_sortkey)
    return rows, Any[v[rows]...]
end

# Score aligned with the canonical rows.
function _des_align_score(score, data, rows, ids, id, ctx)
    score === nothing && return nothing, NaN
    if score isa Symbol
        require_columns(data, [score]; context=ctx)
        return _ml_column(data, score; context=ctx)[rows], NaN
    elseif score isa PrognosticScore
        if score.id !== nothing || id !== nothing
            score.id === nothing && throw(ArgumentError(
                "$ctx: the PrognosticScore has no ids; build it with `id=$(id)`"))
            id === nothing && throw(ArgumentError(
                "$ctx: pass `id=$(score.id)` to match the prognostic score by unit"))
            d = Dict(zip(score.ids, score.score))
            length(d) == length(score.ids) ||
                throw(ArgumentError("$ctx: duplicate ids in the prognostic score"))
            s = Float64[]
            for u in ids
                haskey(d, u) || throw(ArgumentError("$ctx: unit $u has no prognostic " *
                                                    "score"))
                push!(s, d[u])
            end
            return s, score.r2
        end
        length(score.score) == nrow(data) ||
            throw(DimensionMismatch("$ctx: score has $(length(score.score)) units, " *
                                    "data has $(nrow(data)) rows"))
        return score.score[rows], score.r2
    elseif score isa AbstractVector{<:Real}
        length(score) == nrow(data) ||
            throw(DimensionMismatch("$ctx: score must have one value per row of data"))
        s = Float64.(score)
        all(isfinite, s) || throw(ArgumentError("$ctx: score has non-finite values"))
        return s[rows], NaN
    end
    throw(ArgumentError("$ctx: score must be a column name, a vector or a " *
                        "PrognosticScore"))
end

function _des_mahalanobis_matrix(X::AbstractMatrix)
    S = cov(X)
    W = pinv(Symmetric(Matrix(S)))
    n = size(X, 1)
    D = zeros(n, n)
    for i in 1:n, j in (i + 1):n
        d = X[i, :] .- X[j, :]
        D[i, j] = D[j, i] = sqrt(max(dot(d, W * d), 0.0))
    end
    return D
end

# Near-equal consecutive groups of the sorted order.
function _des_consecutive_blocks(order::Vector{Int}, nb::Int)
    n = length(order)
    blk = zeros(Int, n)
    base, rem_ = divrem(n, nb)
    pos = 0
    for b in 1:nb
        sz = base + (b <= rem_ ? 1 : 0)
        for k in 1:sz
            blk[order[pos + k]] = b
        end
        pos += sz
    end
    return blk
end

# Attach the leftover unit of an odd-sized pairing to the nearest pair.
function _des_pairs_to_blocks(pairs, left, D, n)
    blk = zeros(Int, n)
    for (b, (i, j)) in enumerate(pairs)
        blk[i] = blk[j] = b
    end
    if left != 0
        best = argmin([D[left, i] + D[left, j] for (i, j) in pairs])
        blk[left] = best
    end
    return blk
end

# Relabel blocks 1..B in order of their first unit (canonical order).
function _des_relabel(blk::Vector{Int})
    map_ = Dict{Int,Int}()
    out = similar(blk)
    for (i, b) in enumerate(blk)
        out[i] = get!(map_, b, length(map_) + 1)
    end
    return out
end

_des_n_treated(k::Int, p::Real) = clamp(floor(Int, p * k + 0.5), 1, k - 1)

"""
    block_design(data, score=nothing; id=nothing, method=:pairs, block_size=2,
                 n_blocks=nothing, algorithm=:optimal, distance=nothing,
                 covariates=Symbol[], p_treat=0.5) -> BlockingDesign

Form matched pairs or blocks of similar units before randomization, on a prognostic
score or on the Mahalanobis distance of baseline covariates.

Blocking restricts randomization to comparisons among similar units: assignment is
completely random within each block, independently across blocks, so every unit
still has a known assignment probability and the estimand, the sample (or
population) average treatment effect, is unchanged. What blocking changes is
precision. The design variance of the blocked difference in means depends only on the
within-block variation of the potential outcomes, so blocks that are homogeneous in
the outcome remove the between-block variance from the comparison (Imai, King & Nall
2009; Athey & Imbens 2017). Forming blocks on a prognostic score
([`prognostic_score`](@ref)) targets exactly that variation. Among stratified designs
that treat each unit with probability one half, a matched-pair design on a suitable
index of the covariates maximizes precision (Bai 2022).

Pairs and blocks are formed as follows. With `method = :pairs` each pair receives one
treated and one control unit. On a scalar score, `algorithm = :optimal` pairs
adjacent units in sorted order, which minimizes the total within-pair distance; on the
Mahalanobis distance it solves the minimum-distance non-bipartite matching exactly
with Edmonds' (1965) blossom algorithm, as in Greevy et al. (2004) and Lu et al.
(2011) (``O(n^3)``; practical up to a few thousand units). `algorithm = :greedy`
repeatedly pairs the two closest remaining units, which reproduces the `optGreedy`
algorithm of `blockTools` (Moore 2012). With an odd number of units the leftover unit
joins its nearest pair, forming one block of three. With `method = :blocks`, blocks of
`block_size` units (or `n_blocks` blocks) are consecutive quantile groups of the
sorted score, with sizes differing by at most one; on the Mahalanobis distance only
greedy blocking is available (each block grows from the closest remaining pair, and
leftover units join the nearest blocks), because optimal multivariate blocking with
more than two units per block is NP-hard.

Within block ``b``, `round(p_treat × size)` units are treated, with at least one
treated and one control unit per block. The returned mechanism is a
`MatchedPairsRandomization` when every block is a pair with one treated unit and a
`StratifiedRandomization` otherwise. For the analysis, pairs leave no within-pair
degrees of freedom for estimating effect heterogeneity: the matched-pair variance
estimator is conservative (Bai, Romano & Shaikh 2022), whereas blocks with at least
two treated and two control units permit the blocked Neyman variance; see
[`experiment_estimate`](@ref). Units should be blocked only on pre-treatment
information. Rerandomization (Morgan & Rubin 2012; `Rerandomization`) is an
alternative when balance on many covariates matters more than on one score.

# Arguments
- `data::AbstractDataFrame`: the experimental units, one row per unit.
- `score = nothing`: `nothing` (use `covariates`), a column name, a numeric vector with
  one value per row of `data`, or a [`PrognosticScore`](@ref) (matched to units by
  `id`).

# Keywords
- `id::Union{Nothing,Symbol} = nothing`: unit identifier column. With an `id` the design
  does not depend on the row order of `data` and can be joined back by key; without
  one, units are the rows of `data` in their current order.
- `method::Symbol = :pairs`: `:pairs` (matched pairs) or `:blocks`.
- `block_size::Integer = 2`: units per block for `method = :blocks`.
- `n_blocks = nothing`: number of blocks instead of `block_size` (scalar score only).
- `algorithm::Symbol = :optimal`: `:optimal` or `:greedy` (see above).
- `distance = nothing`: `:score` (the default when a score is given) or
  `:mahalanobis` (the default otherwise; a given score is then appended to
  `covariates`).
- `covariates::Vector{Symbol} = Symbol[]`: numeric baseline covariates for the
  Mahalanobis distance.
- `p_treat::Real = 0.5`: share treated within each block.

# Returns
- [`BlockingDesign`](@ref).

# Examples
```julia
using DrSnow, DataFrames, StableRNGs
rng = StableRNG(1)
n = 200
df = DataFrame(id=1:n, x1=randn(rng, n), x2=randn(rng, n))
df.y = df.x1 .+ 0.5 .* df.x2 .+ randn(rng, n)          # baseline outcome
ps = prognostic_score(df, :y; covariates=[:x1, :x2], id=:id, rng=StableRNG(2))
bd = block_design(df, ps; id=:id)                                  # optimal pairs
bd4 = block_design(df, ps; id=:id, method=:blocks, block_size=4)   # quantile blocks
bdm = block_design(df; id=:id, covariates=[:x1, :x2])              # Mahalanobis
df2 = assign_treatment(bd, df; rng=StableRNG(3))
variance_reduction(bd)
```

# References
- Athey, S., & Imbens, G. W. (2017). The econometrics of randomized experiments. In
  A. V. Banerjee & E. Duflo (Eds.), *Handbook of Economic Field Experiments* (Vol. 1,
  pp. 73–140). North-Holland.
- Bai, Y. (2022). Optimality of matched-pair designs in randomized controlled trials.
  *American Economic Review*, 112(12), 3911–3940.
- Bai, Y., Romano, J. P., & Shaikh, A. M. (2022). Inference in experiments with
  matched pairs. *Journal of the American Statistical Association*, 117(540),
  1726–1737.
- Edmonds, J. (1965). Paths, trees, and flowers. *Canadian Journal of Mathematics*,
  17, 449–467.
- Greevy, R., Lu, B., Silber, J. H., & Rosenbaum, P. (2004). Optimal multivariate
  matching before randomization. *Biostatistics*, 5(2), 263–275.
- Imai, K., King, G., & Nall, C. (2009). The essential role of pair matching in
  cluster-randomized experiments, with application to the Mexican universal health
  insurance evaluation. *Statistical Science*, 24(1), 29–53.
- Lu, B., Greevy, R., Xu, X., & Beck, C. (2011). Optimal nonbipartite matching and its
  statistical applications. *The American Statistician*, 65(1), 21–30.
- Moore, R. T. (2012). Multivariate continuous blocking to improve political science
  experiments. *Political Analysis*, 20(4), 460–479.
"""
function block_design(data::AbstractDataFrame, score=nothing;
                      id::Union{Nothing,Symbol}=nothing, method::Symbol=:pairs,
                      block_size::Integer=2, n_blocks::Union{Nothing,Integer}=nothing,
                      algorithm::Symbol=:optimal,
                      distance::Union{Nothing,Symbol}=nothing,
                      covariates::Vector{Symbol}=Symbol[], p_treat::Real=0.5)
    ctx = "block_design"
    method in (:pairs, :blocks) ||
        throw(ArgumentError("$ctx: method must be :pairs or :blocks"))
    algorithm in (:optimal, :greedy) ||
        throw(ArgumentError("$ctx: algorithm must be :optimal or :greedy"))
    0 < p_treat < 1 || throw(ArgumentError("$ctx: p_treat must be in (0, 1)"))
    dist = distance === nothing ? (score === nothing ? :mahalanobis : :score) : distance
    dist in (:score, :mahalanobis) ||
        throw(ArgumentError("$ctx: distance must be :score or :mahalanobis"))
    dist === :score && score === nothing &&
        throw(ArgumentError("$ctx: distance = :score needs a score"))
    dist === :mahalanobis && isempty(covariates) && score === nothing &&
        throw(ArgumentError("$ctx: give a score or covariates"))
    n = nrow(data)
    n >= 2 || throw(ArgumentError("$ctx: need at least two units"))
    rows, ids = _des_unit_rows(data, id, ctx)
    s, r2 = _des_align_score(score, data, rows, ids, id, ctx)
    require_columns(data, covariates; context=ctx)

    # distance matrix (pairwise) and sorted order for the scalar case
    if dist === :score
        order = sortperm(collect(1:n); by=i -> (s[i], i))
        D = abs.(s .- s')
    else
        X = isempty(covariates) ? zeros(n, 0) :
            _ml_matrix(data, covariates; context=ctx)[rows, :]
        s === nothing || (X = hcat(X, s))
        size(X, 2) >= 1 || throw(ArgumentError("$ctx: no covariates"))
        D = _des_mahalanobis_matrix(X)
        order = Int[]
    end

    if method === :pairs
        block_size == 2 || n_blocks === nothing ||
            throw(ArgumentError("$ctx: method = :pairs uses pairs; use " *
                                "method = :blocks for block_size / n_blocks"))
        n >= 2 || throw(ArgumentError("$ctx: need at least two units"))
        pairs, left = if dist === :score
            algorithm === :optimal ? _des_pairs_1d(s, order) : _des_greedy_pairs(D)
        else
            algorithm === :optimal ? _des_optimal_pairs(D) : _des_greedy_pairs(D)
        end
        blk = _des_pairs_to_blocks(pairs, left, D, n)
    else
        k = Int(block_size)
        k >= 2 || throw(ArgumentError("$ctx: block_size must be at least 2"))
        if dist === :score
            nb = n_blocks === nothing ? fld(n, k) : Int(n_blocks)
            (1 <= nb && n ÷ nb >= 2) ||
                throw(ArgumentError("$ctx: blocks must have at least two units"))
            blk = _des_consecutive_blocks(order, nb)
        else
            n_blocks === nothing ||
                throw(ArgumentError("$ctx: n_blocks is only available for scalar scores"))
            algorithm === :greedy ||
                throw(ArgumentError("$ctx: optimal blocking with block_size > 2 on a " *
                                    "multivariate distance is NP-hard; use " *
                                    "algorithm = :greedy, or method = :pairs"))
            fld(n, k) >= 1 || throw(ArgumentError("$ctx: block_size exceeds n"))
            blk = _des_greedy_blocks(D, k)
        end
    end
    blk = _des_relabel(blk)
    B = maximum(blk)
    sizes = [count(==(b), blk) for b in 1:B]
    all(>=(2), sizes) || error("internal error: block with fewer than two units")
    nt = [_des_n_treated(sz, p_treat) for sz in sizes]
    obj = 0.0
    for b in 1:B
        mem = findall(==(b), blk)
        for a in eachindex(mem), c in (a + 1):length(mem)
            obj += D[mem[a], mem[c]]
        end
    end
    mech = if all(==(2), sizes) && all(==(1), nt)
        MatchedPairsRandomization(blk)
    else
        StratifiedRandomization(blk, Dict(b => nt[b] for b in 1:B))
    end
    return BlockingDesign(ids, id, blk, mech, nt, sizes, s, r2, method,
                          dist === :score && method === :blocks ? :quantile : algorithm,
                          dist, obj, copy(covariates))
end

"""
    assign_treatment(bd::BlockingDesign, data; rng=Random.default_rng(),
                     treatment=:treated, block=:block) -> DataFrame

Draw one treatment assignment from a blocked or matched-pair design and attach it,
with the block labels, to the experimental data.

The assignment is a single draw from the design's mechanism (`bd.mechanism`):
complete randomization of the planned number of treated units within each block,
independently across blocks, or one treated unit per pair. Because the draw comes
from the stored mechanism, the same object describes the randomization distribution
used later by `randomization_test` and the assignment probabilities used by
[`experiment_estimate`](@ref). For a credible experiment, draw the assignment once
with a recorded seed (pass a `StableRNG` for results that are stable across Julia
versions) and do not redraw it after inspecting balance; if balance on additional
covariates matters, build that into the design (e.g. `Rerandomization`, Morgan &
Rubin 2012) so that inference accounts for it.

# Arguments
- `bd::BlockingDesign`: the design, from [`block_design`](@ref).
- `data::AbstractDataFrame`: the experimental units, in any row order when `bd.id` is
  set (units are matched by key), otherwise in the row order used to build the design.

# Keywords
- `rng::AbstractRNG = Random.default_rng()`: random-number generator; the assignment
  is reproducible given `rng`.
- `treatment::Symbol = :treated`: name of the new 0/1 treatment column.
- `block::Symbol = :block`: name of the new block-label column.

# Returns
- `DataFrame`: a copy of `data` with the columns `block` and `treatment` added.

# Examples
```julia
using DrSnow, DataFrames, StableRNGs
rng = StableRNG(1)
df = DataFrame(id=1:40, x=randn(rng, 40))
bd = block_design(df, :x; id=:id)
df2 = assign_treatment(bd, df; rng=StableRNG(2024))
df2.y = df2.x .+ 0.5 .* df2.treated .+ randn(rng, 40)
randomization_test(df2, :y, :treated; mechanism=bd.mechanism, id=:id,
                   rng=StableRNG(7))
```

# References
- Morgan, K. L., & Rubin, D. B. (2012). Rerandomization to improve covariate balance
  in experiments. *Annals of Statistics*, 40(2), 1263–1282.
"""
function assign_treatment(bd::BlockingDesign, data::AbstractDataFrame;
                          rng::AbstractRNG=Random.default_rng(),
                          treatment::Symbol=:treated, block::Symbol=:block)
    ctx = "assign_treatment"
    idx = _des_design_index(bd, data, ctx)
    z = draw_assignment(rng, bd.mechanism)
    out = DataFrame(data; copycols=true)
    out[!, block] = bd.blocks[idx]
    out[!, treatment] = Int.(z[idx])
    return out
end

# For each row of `data`, the position of that unit in the design.
function _des_design_index(bd::BlockingDesign, data, ctx)
    n = nrow(data)
    n == length(bd.ids) || throw(DimensionMismatch("$ctx: data has $n rows but the " *
                                                   "design has $(length(bd.ids)) units"))
    bd.id === nothing && return collect(1:n)
    require_columns(data, [bd.id]; context=ctx)
    pos = Dict(u => i for (i, u) in enumerate(bd.ids))
    idx = Int[]
    for u in data[!, bd.id]
        haskey(pos, u) || throw(ArgumentError("$ctx: unit $u is not in the design"))
        push!(idx, pos[u])
    end
    allunique(idx) || throw(ArgumentError("$ctx: duplicate ids in data"))
    return idx
end

"""
    variance_reduction(bd::BlockingDesign; score=bd.score, r2=bd.r2) -> NamedTuple

Approximate expected precision gain of a blocked design over complete randomization
with the same number of treated units, computed from the prognostic score.

For the score ``s`` itself the comparison is exact. The design variance of the
block-size-weighted difference in means of ``s`` is

```math
V_B(s) = \\sum_b w_b^2\\, S_b^2 \\left(\\frac{1}{n_{1b}} + \\frac{1}{n_{0b}}\\right),
\\qquad
V_C(s) = S^2 \\left(\\frac{1}{n_1} + \\frac{1}{n_0}\\right),
```

under blocking and complete randomization respectively, where ``w_b`` is the share of
units in block ``b``, ``S_b^2`` the within-block and ``S^2`` the overall variance of
the score, and ``n_{1b}, n_{0b}`` the treated and control counts. Their ratio,
`score_variance_ratio`, says how well the blocks balance the score.

For the outcome the result is an **approximation** under a stylized model: the
outcome is the score plus independent noise, ``Y = s + e`` with ``e`` independent of
the blocks and of ``s``, ``\\text{Var}(s)/\\text{Var}(Y)`` equal to the score's
out-of-sample R², and a constant treatment effect. The noise variance is then
``\\sigma_e^2 = S^2 (1 - R^2)/R^2`` and the expected variance ratio is

```math
\\frac{V_B(s) + \\sigma_e^2\\, c_B}{V_C(s) + \\sigma_e^2\\, c_C},
\\qquad
c_B = \\sum_b w_b^2 \\left(\\frac{1}{n_{1b}} + \\frac{1}{n_{0b}}\\right),
\\quad
c_C = \\frac{1}{n_1} + \\frac{1}{n_0}.
```

For comparison, regression adjustment on the score under complete randomization has
large-sample variance ratio ``1 - R^2``. These figures are expectations for the design
under the stated model, not guarantees for a realized experiment: they are optimistic
when the score's prediction errors are correlated within blocks, when effects are
heterogeneous in ways the score does not capture, or when the R² was estimated on a
population different from the experimental one. The realized precision is what
[`experiment_estimate`](@ref) reports after the experiment.

# Arguments
- `bd::BlockingDesign`: a design from [`block_design`](@ref).

# Keywords
- `score = bd.score`: score values in the design's unit order (default: the score the
  design was built on); required when the design was built without a score.
- `r2::Real = bd.r2`: out-of-sample R² of the score for the outcome, in `(0, 1]`. When
  it is unknown (`NaN`), only the score-level quantities are returned.

# Returns
- `NamedTuple` with `score_variance_ratio` (``V_B(s)/V_C(s)``),
  `outcome_variance_ratio` (the approximation above), `regression_adjustment_ratio`
  (``1 - R^2``), `effective_sample_multiplier` (`1 / outcome_variance_ratio`), where
  these three are `missing` when `r2` is unknown, plus `within_block_sd` (pooled
  within-block standard deviation of the score), `score_sd` and `r2`.

# Examples
```julia
using DrSnow, DataFrames, StableRNGs
rng = StableRNG(1)
df = DataFrame(id=1:200, s=randn(rng, 200))
bd = block_design(df, :s; id=:id, method=:blocks, block_size=4)
variance_reduction(bd; r2=0.5)
```

# References
- Athey, S., & Imbens, G. W. (2017). The econometrics of randomized experiments. In
  A. V. Banerjee & E. Duflo (Eds.), *Handbook of Economic Field Experiments* (Vol. 1,
  pp. 73–140). North-Holland.
- Imai, K., King, G., & Nall, C. (2009). The essential role of pair matching in
  cluster-randomized experiments, with application to the Mexican universal health
  insurance evaluation. *Statistical Science*, 24(1), 29–53.
"""
function variance_reduction(bd::BlockingDesign; score=bd.score, r2::Real=bd.r2)
    score === nothing &&
        throw(ArgumentError("variance_reduction: the design has no score; pass `score`"))
    s = Float64.(collect(score))
    length(s) == length(bd.ids) ||
        throw(DimensionMismatch("variance_reduction: score must have one value per unit"))
    N = length(s)
    B = length(bd.block_sizes)
    vb, cb, ssw = 0.0, 0.0, 0.0
    for b in 1:B
        mem = findall(==(b), bd.blocks)
        Nb = length(mem)
        n1 = bd.n_treated[b]
        n0 = Nb - n1
        w = Nb / N
        S2 = var(s[mem])
        vb += w^2 * S2 * (1 / n1 + 1 / n0)
        cb += w^2 * (1 / n1 + 1 / n0)
        ssw += (Nb - 1) * S2
    end
    n1 = sum(bd.n_treated)
    n0 = N - n1
    S2 = var(s)
    S2 > 0 || throw(ArgumentError("variance_reduction: the score is constant"))
    cc = 1 / n1 + 1 / n0
    vc = S2 * cc
    ratio_s = vb / vc
    ratio_y = missing
    if isfinite(r2)
        0 < r2 <= 1 || throw(ArgumentError("variance_reduction: r2 must be in (0, 1]"))
        σ2e = S2 * (1 - r2) / r2
        ratio_y = (vb + σ2e * cb) / (vc + σ2e * cc)
    end
    return (score_variance_ratio=ratio_s, outcome_variance_ratio=ratio_y,
            regression_adjustment_ratio=isfinite(r2) ? 1 - r2 : missing,
            effective_sample_multiplier=ratio_y === missing ? missing : 1 / ratio_y,
            within_block_sd=sqrt(ssw / (N - B)), score_sd=sqrt(S2), r2=float(r2))
end
