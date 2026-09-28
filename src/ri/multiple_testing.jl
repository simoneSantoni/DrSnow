# Multiple testing: Westfall–Young step-down adjustment using the joint
# randomization distribution, and Holm / Benjamini–Hochberg adjustments of plain
# p-values.

"""
    holm_adjust(p) -> Vector{Float64}

Holm (1979) step-down adjusted p-values, which control the family-wise error rate
(the probability of at least one false rejection) under arbitrary dependence among
the tests.

With the p-values sorted as ``p_{(1)} ≤ … ≤ p_{(m)}``, the adjusted p-value of the
``i``-th smallest is ``\\max_{j ≤ i} \\min\\{1, (m - j + 1)\\, p_{(j)}\\}``; rejecting
the hypotheses whose adjusted p-value is at most ``α`` is Holm's sequentially
rejective procedure. It is uniformly more powerful than the Bonferroni correction
and requires no assumption on the dependence of the tests, but, unlike
[`westfall_young_adjust`](@ref), it cannot exploit that dependence: with strongly
correlated outcomes the Westfall–Young adjustment computed from the joint
randomization distribution is less conservative.

# Arguments
- `p::AbstractVector{<:Real}`: unadjusted p-values in ``[0, 1]``.

# Returns
- `Vector{Float64}` of adjusted p-values, in the input order.

# Examples
```julia
using DrSnow
holm_adjust([0.01, 0.04, 0.03])   # [0.03, 0.06, 0.06]
```

# References
- Holm, S. (1979). A simple sequentially rejective multiple test procedure.
  *Scandinavian Journal of Statistics*, 6(2), 65–70.
"""
function holm_adjust(p::AbstractVector{<:Real})
    _ri_check_pvalues(p)
    m = length(p)
    o = sortperm(p)
    adj = similar(p, Float64)
    run = 0.0
    for (i, k) in enumerate(o)
        run = max(run, min(1.0, (m - i + 1) * p[k]))
        adj[k] = run
    end
    return adj
end

"""
    bh_adjust(p) -> Vector{Float64}

Benjamini–Hochberg (1995) adjusted p-values ("q-values" in the loose sense), which
control the false discovery rate, the expected share of false rejections among all
rejections.

With the p-values sorted as ``p_{(1)} ≤ … ≤ p_{(m)}``, the adjusted p-value of the
``i``-th smallest is ``\\min_{j ≥ i} \\min\\{1, m\\, p_{(j)} / j\\}``; rejecting the
hypotheses whose adjusted p-value is at most ``q`` controls the false discovery rate
at level ``q m_0 / m ≤ q`` (``m_0`` true nulls) when the tests are independent, and
more generally under positive regression dependence on the subset of true nulls
(Benjamini and Yekutieli 2001). The false discovery rate is a weaker criterion than
the family-wise error rate: it is appropriate for exploratory analyses of many
outcomes, not for confirmatory claims about each individual hypothesis.

# Arguments
- `p::AbstractVector{<:Real}`: unadjusted p-values in ``[0, 1]``.

# Returns
- `Vector{Float64}` of adjusted p-values, in the input order.

# Examples
```julia
using DrSnow
bh_adjust([0.01, 0.04, 0.03])   # [0.03, 0.04, 0.04]
```

# References
- Benjamini, Y., & Hochberg, Y. (1995). Controlling the false discovery rate: A
  practical and powerful approach to multiple testing. *Journal of the Royal
  Statistical Society: Series B (Methodological)*, 57(1), 289–300.
- Benjamini, Y., & Yekutieli, D. (2001). The control of the false discovery rate in
  multiple testing under dependency. *Annals of Statistics*, 29(4), 1165–1188.
"""
function bh_adjust(p::AbstractVector{<:Real})
    _ri_check_pvalues(p)
    m = length(p)
    o = sortperm(p; rev=true)
    adj = similar(p, Float64)
    run = 1.0
    for (i, k) in enumerate(o)
        rank = m - i + 1
        run = min(run, m / rank * p[k])
        adj[k] = min(1.0, run)
    end
    return adj
end

function _ri_check_pvalues(p)
    isempty(p) && throw(ArgumentError("need at least one p-value"))
    all(x -> 0 <= x <= 1, p) || throw(ArgumentError("p-values must lie in [0, 1]"))
end

_ri_orient(x, alt::Symbol) = alt === :two_sided ? abs(x) : alt === :greater ? x : -x

"""
    westfall_young_adjust(observed, draws; method=:minp, alternative=:two_sided,
                          weights=nothing) -> NamedTuple

Westfall–Young (1993) step-down adjusted p-values computed from the joint
randomization (or resampling) distribution of ``K`` test statistics; they control
the family-wise error rate while exploiting the dependence among the statistics.

The hypotheses are ordered from the most to the least significant. For the ``j``-th
hypothesis in that order, the adjusted p-value is the share of the reference set in
which the most extreme statistic among the hypotheses not yet rejected (the ``j``-th
and all less significant ones) is at least as extreme as the ``j``-th observed
statistic; monotonicity along the order is then enforced. With `method = :minp` the
comparison is on the per-hypothesis p-values (scale-free; recommended when the
statistics are on different scales); with `method = :maxt` it is on the (oriented)
statistics themselves, which should then be studentized so that they are
comparable. The `:maxt` version with studentized statistics is the randomization
counterpart of the step-down procedure of Romano and Wolf (2005).

Strong control of the family-wise error rate requires subset pivotality: the joint
distribution of the statistics of any subset of true nulls must not depend on
whether the other nulls are true. In randomization inference of sharp nulls this
holds exactly, because under the sharp nulls of a subset the outcomes of those
hypotheses are fixed and their joint distribution is the randomization distribution
of the design, whatever the effects on other outcomes. With Monte Carlo draws the
adjusted p-values include the observed row in the reference set, giving the
``(1 + \\#)/(1 + B)`` form.

# Arguments
- `observed::AbstractVector`: the ``K`` observed statistics.
- `draws::AbstractMatrix`: ``B × K`` statistics under re-randomized assignments,
  with the same assignment in each row. Rows containing `NaN` are dropped.

# Keywords
- `method::Symbol`: `:minp` (default) or `:maxt`.
- `alternative`: `:two_sided` (compares absolute values), `:greater` or `:less`,
  either one `Symbol` or a vector with one entry per hypothesis.
- `weights`: `nothing` (default) when the rows of `draws` are Monte Carlo draws, in
  which case the observed row is added to the reference set; otherwise the
  probabilities of the rows, and `draws` must then be the complete enumerated
  support (including the observed assignment).

# Returns
- `NamedTuple` `(raw, adjusted)` of `Vector{Float64}`s in the input order: the
  unadjusted randomization p-values and the step-down adjusted ones.

# Examples
```julia
using DrSnow, Random, StableRNGs
rng = StableRNG(4)
Y = randn(rng, 30, 3); Y[1:15, 1] .+= 1.0          # effect on the first outcome only
z = [trues(15); falses(15)]
dim(z) = [sum(Y[z, k]) / 15 - sum(Y[.!z, k]) / 15 for k in 1:3]
draws = reduce(vcat, [dim(shuffle(rng, z))' for _ in 1:1999])
westfall_young_adjust(dim(z), draws).adjusted
```

# References
- Westfall, P. H., & Young, S. S. (1993). *Resampling-Based Multiple Testing:
  Examples and Methods for p-Value Adjustment*. Wiley.
- Romano, J. P., & Wolf, M. (2005). Exact and approximate stepdown methods for
  multiple hypothesis testing. *Journal of the American Statistical Association*,
  100(469), 94–108.
- Young, A. (2019). Channeling Fisher: Randomization tests and the statistical
  insignificance of seemingly significant experimental results. *Quarterly Journal
  of Economics*, 134(2), 557–598.
"""
function westfall_young_adjust(observed::AbstractVector, draws::AbstractMatrix;
                               method::Symbol=:minp, alternative=:two_sided,
                               weights=nothing)
    K = length(observed)
    size(draws, 2) == K ||
        throw(DimensionMismatch("draws must have one column per statistic"))
    if weights === nothing
        R = vcat(reshape(Float64.(observed), 1, K), Matrix{Float64}(draws))
        w = ones(size(R, 1))
    else
        length(weights) == size(draws, 1) ||
            throw(DimensionMismatch("weights must have one entry per row of draws"))
        R = Matrix{Float64}(draws)
        w = Float64.(weights)
    end
    return _ri_westfall_young(Float64.(observed), R, w, alternative, method)
end

# R is the full reference set (observed included), w its weights.
function _ri_westfall_young(obs::Vector{Float64}, R::Matrix{Float64}, w::Vector{Float64},
                            alternative, method::Symbol)
    method in (:minp, :maxt) || throw(ArgumentError("method must be :minp or :maxt"))
    K = length(obs)
    alts = alternative isa Symbol ? fill(alternative, K) : collect(alternative)
    length(alts) == K || throw(DimensionMismatch("one alternative per hypothesis"))
    foreach(_ri_check_alternative, alts)
    keep = [all(!isnan, view(R, i, :)) for i in axes(R, 1)]
    R = R[keep, :]
    w = w[keep]
    isempty(w) && error("no reference assignment has all statistics defined")
    W = sum(w)
    E = similar(R)
    for k in 1:K, b in axes(R, 1)
        E[b, k] = _ri_orient(R[b, k], alts[k])
    end
    eo = [_ri_orient(obs[k], alts[k]) for k in 1:K]
    # per-hypothesis p-value of every reference row (and of the observed values)
    P = similar(E)
    raw = zeros(K)
    for k in 1:K
        col = view(E, :, k)
        o = sortperm(col)
        sv = col[o]
        suffix = reverse(cumsum(reverse(w[o])))
        pfun(v) = (j = searchsortedfirst(sv, v - _ri_tol(v));
                   j > length(sv) ? 0.0 : suffix[j] / W)
        for b in axes(E, 1)
            P[b, k] = pfun(E[b, k])
        end
        raw[k] = pfun(eo[k])
    end
    adjusted = zeros(K)
    if method === :minp
        order = sortperm(collect(zip(raw, -eo)))
        q = fill(Inf, size(P, 1))
        for j in K:-1:1
            k = order[j]
            q .= min.(q, view(P, :, k))
            adjusted[k] = sum(w[b] for b in eachindex(q) if q[b] <= raw[k] + 1e-12;
                              init=0.0) / W
        end
    else
        order = sortperm(eo; rev=true)
        u = fill(-Inf, size(E, 1))
        for j in K:-1:1
            k = order[j]
            u .= max.(u, view(E, :, k))
            tol = _ri_tol(eo[k])
            adjusted[k] = sum(w[b] for b in eachindex(u) if u[b] >= eo[k] - tol;
                              init=0.0) / W
        end
    end
    # enforce monotonicity along the step-down order
    run = 0.0
    for k in order
        run = max(run, adjusted[k])
        adjusted[k] = min(1.0, run)
    end
    return (raw=raw, adjusted=adjusted)
end

"""
    MultipleTestingResult

Result of [`ri_multiple_testing`](@ref): randomization tests of the sharp null of no
effect on each of several outcomes, with three multiplicity adjustments computed
from the same reference set.

# Fields
- `hypotheses::Vector{String}`: outcome names, one hypothesis each.
- `observed::Vector{Float64}`: observed statistics.
- `pvalues::Vector{Float64}`: unadjusted randomization p-values.
- `adjusted::Vector{Float64}`: Westfall–Young step-down adjusted p-values
  (family-wise error rate, using the joint randomization distribution).
- `holm::Vector{Float64}`: Holm adjustment of `pvalues` (family-wise error rate,
  any dependence).
- `bh::Vector{Float64}`: Benjamini–Hochberg adjustment of `pvalues` (false
  discovery rate).
- `method::Symbol`: `:minp` or `:maxt` (Westfall–Young variant).
- `statistic_name::String`, `alternative::Symbol`: statistic and alternative.
- `exact::Bool`, `n_draws::Int`, `n_dropped::Int`, `mechanism::String`: reference
  set and design, as in [`RandomizationTestResult`](@ref).
- `distribution::Matrix{Float64}`, `weights::Vector{Float64}`: joint reference set
  (one row per assignment, one column per outcome).

# Accessors
- [`randomization_distribution`](@ref)`(r)`.
"""
struct MultipleTestingResult
    hypotheses::Vector{String}
    observed::Vector{Float64}
    pvalues::Vector{Float64}
    adjusted::Vector{Float64}
    holm::Vector{Float64}
    bh::Vector{Float64}
    method::Symbol
    statistic_name::String
    alternative::Symbol
    exact::Bool
    n_draws::Int
    n_dropped::Int
    distribution::Matrix{Float64}
    weights::Vector{Float64}
    mechanism::String
end

randomization_distribution(r::MultipleTestingResult) = (r.distribution, r.weights)

function Base.show(io::IO, ::MIME"text/plain", r::MultipleTestingResult)
    println(io, "Randomization tests with multiplicity adjustment")
    println(io, "H₀ (sharp, per outcome): no effect of treatment on the outcome for any unit")
    println(io, "Statistic: ", r.statistic_name, "; alternative: ", r.alternative)
    println(io, "Assignment mechanism: ", r.mechanism)
    println(io, "Reference distribution: ", r.exact ? "exact, " : "Monte Carlo, ",
            r.n_draws, r.exact ? " assignments" : " draws")
    println(io, "Westfall–Young step-down (", r.method, ") controls the FWER; Holm: FWER; ",
            "BH: FDR.")
    w = max(8, maximum(length, r.hypotheses))
    @printf(io, "%-*s %12s %9s %9s %9s %9s\n", w, "Outcome", "Statistic", "p", "p (WY)",
            "p (Holm)", "p (BH)")
    for i in eachindex(r.hypotheses)
        @printf(io, "%-*s %12.4g %9.4g %9.4g %9.4g %9.4g\n", w, r.hypotheses[i],
                r.observed[i], r.pvalues[i], r.adjusted[i], r.holm[i], r.bh[i])
    end
end

Base.show(io::IO, r::MultipleTestingResult) =
    print(io, "MultipleTestingResult($(length(r.hypotheses)) hypotheses, $(r.method))")

"""
    ri_multiple_testing(data, outcomes, treatment; statistic=:diff_means,
                        method=:minp, alternative=:two_sided, mechanism=nothing,
                        strata=nothing, cluster=nothing, id=nothing,
                        covariates=Symbol[], nperm=10_000, exact=:auto,
                        rng=Random.default_rng(), threaded=Threads.nthreads() > 1)
        -> MultipleTestingResult

Randomization tests of the sharp null of no treatment effect on each of several
outcomes, with p-values adjusted for multiple testing by the Westfall–Young
step-down procedure (family-wise error rate), by Holm's procedure and by the
Benjamini–Hochberg procedure (false discovery rate).

Experiments usually report effects on many outcomes, and reporting the smallest
unadjusted p-values overstates the evidence. All outcomes are tested on a common set
of re-randomized assignments, so the joint randomization distribution of the ``K``
statistics is available and the Westfall–Young adjustment can use their dependence;
because subset pivotality holds exactly for sharp nulls, the adjustment controls the
family-wise error rate in the strong sense, in finite samples when the reference
distribution is exact (Westfall and Young 1993; Young 2019). Holm's adjustment
controls the same error rate without using the dependence and is therefore more
conservative for correlated outcomes; the Benjamini–Hochberg adjustment controls
the weaker false discovery rate under independence or positive dependence. When the
outcomes form a single family with a common interpretation, a pre-specified index or
an omnibus test can be more powerful than any per-outcome adjustment.

The tests are exact for the sharp nulls; as in [`randomization_test`](@ref), use a
studentized statistic when the hypotheses of interest concern average effects, and
state the family of hypotheses before looking at the results.

# Arguments
- `data`: a `DataFrame` with one row per randomized unit.
- `outcomes::Vector{Symbol}`: outcome columns, one hypothesis each.
- `treatment::Symbol`: 0/1 treatment column.

# Keywords
- `statistic`, `alternative`, design keywords (`mechanism`, `strata`, `cluster`,
  `id`), `covariates`, `nperm`, `exact`, `rng`, `threaded`: as in
  [`randomization_test`](@ref).
- `method::Symbol`: Westfall–Young variant, `:minp` (default) or `:maxt` (see
  [`westfall_young_adjust`](@ref)).

# Returns
- [`MultipleTestingResult`](@ref); `r.adjusted`, `r.holm` and `r.bh` hold the
  adjusted p-values.

# Examples
```julia
using DrSnow, DataFrames, StableRNGs
rng = StableRNG(4)
df = DataFrame(block=repeat(1:3; inner=10), d=repeat([1, 1, 1, 1, 1, 0, 0, 0, 0, 0], 3))
df.y1 = 0.8 .* df.d .+ randn(rng, 30)
df.y2 = randn(rng, 30)
df.y3 = 0.5 .* df.y2 .+ randn(rng, 30)
r = ri_multiple_testing(df, [:y1, :y2, :y3], :d; strata=:block, rng=rng)
r.pvalues, r.adjusted, r.holm
```

# References
- Westfall, P. H., & Young, S. S. (1993). *Resampling-Based Multiple Testing:
  Examples and Methods for p-Value Adjustment*. Wiley.
- Romano, J. P., & Wolf, M. (2005). Exact and approximate stepdown methods for
  multiple hypothesis testing. *Journal of the American Statistical Association*,
  100(469), 94–108.
- Young, A. (2019). Channeling Fisher: Randomization tests and the statistical
  insignificance of seemingly significant experimental results. *Quarterly Journal
  of Economics*, 134(2), 557–598.
- Holm, S. (1979). A simple sequentially rejective multiple test procedure.
  *Scandinavian Journal of Statistics*, 6(2), 65–70.
- Benjamini, Y., & Hochberg, Y. (1995). Controlling the false discovery rate: A
  practical and powerful approach to multiple testing. *Journal of the Royal
  Statistical Society: Series B (Methodological)*, 57(1), 289–300.
"""
function ri_multiple_testing(data, outcomes::Vector{Symbol}, treatment::Symbol;
                             statistic=:diff_means, method::Symbol=:minp,
                             alternative::Symbol=:two_sided, mechanism=nothing,
                             strata::Union{Nothing,Symbol}=nothing,
                             cluster::Union{Nothing,Symbol}=nothing,
                             id::Union{Nothing,Symbol}=nothing,
                             covariates::Vector{Symbol}=Symbol[], nperm::Integer=10_000,
                             exact=:auto, rng::AbstractRNG=Random.default_rng(),
                             threaded::Bool=Threads.nthreads() > 1)
    ctxname = "ri_multiple_testing"
    isempty(outcomes) && throw(ArgumentError("$ctxname: need at least one outcome"))
    allunique(outcomes) || throw(ArgumentError("$ctxname: duplicate outcomes"))
    method in (:minp, :maxt) || throw(ArgumentError("method must be :minp or :maxt"))
    _ri_check_alternative(alternative)
    stat = _ri_parse_statistic(statistic)
    _ri_check_covariates(stat, covariates, ctxname)
    alt = _ri_signed(stat) ? alternative : _ri_unsigned_alternative(alternative, stat)
    design = _ri_design(data, treatment; mechanism, strata, cluster, id, covariates,
                        sortcols=outcomes, context=ctxname)
    ctx = design.ctx
    preps = map(outcomes) do o
        y = Vector{Float64}(data[design.rows, o])
        all(isfinite, y) || throw(ArgumentError("$ctxname: $o has non-finite values"))
        _ri_prepare(stat, y, ctx)
    end
    K = length(outcomes)
    f = z -> [_ri_eval(stat, preps[k], z, ctx) for k in 1:K]
    plan = _ri_plan(design.mech, nperm, exact, rng)
    R = _ri_map(f, plan, design.mech, design.z, K; threaded)
    obs = f(design.z)
    any(isnan, obs) && throw(ArgumentError("$ctxname: statistic undefined for the " *
                                           "observed assignment"))
    w = _ri_weights(plan)
    wy = _ri_westfall_young(obs, R, w, alt, method)
    dropped = count(i -> any(isnan, view(R, i, :)), axes(R, 1))
    return MultipleTestingResult(string.(outcomes), obs, wy.raw, wy.adjusted,
                                 holm_adjust(wy.raw), bh_adjust(wy.raw), method,
                                 _ri_name(stat), alt, plan.exact, plan.B, dropped, R, w,
                                 _ri_describe(design.mech))
end
