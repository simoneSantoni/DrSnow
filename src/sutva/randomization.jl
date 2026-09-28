# Randomization inference under interference.
#
# - spillover_fisher_test: exact test of the sharp null of no spillovers, conditioning
#   on the treatments of a set of focal units (Athey, Eckles & Imbens 2018).
# - exposure_balance_test: covariate balance across exposure levels, calibrated by the
#   actual assignment mechanism (so structural imbalance, e.g. high-degree units being
#   exposed more often, does not produce false rejections).
# - treatment_moran_test: Moran's I of the treatment vector against the design's
#   randomization distribution, with Cliff–Ord permutation moments as secondary output.

# ---------------------------------------------------------------------------------
# Conditional assignment samplers: draw Z ~ design | Z_F = z_F
# ---------------------------------------------------------------------------------

function _sv_conditional_sampler(m::BernoulliAssignment, z::BitVector, focal::BitVector)
    free = findall(.!focal)
    p = m.p[free]
    return function (rng)
        zz = copy(z)
        for (k, i) in enumerate(free)
            zz[i] = rand(rng) < p[k]
        end
        return zz
    end
end

function _sv_conditional_sampler(m::CompleteRandomization, z::BitVector, focal::BitVector)
    free = findall(.!focal)
    vals = z[free]
    return function (rng)
        zz = copy(z)
        zz[free] .= vals[randperm(rng, length(free))]
        return zz
    end
end

function _sv_conditional_sampler(m::StratifiedRandomization, z::BitVector,
                                 focal::BitVector)
    frees = [filter(i -> !focal[i], g) for g in m.groups]
    vals = [z[f] for f in frees]
    return function (rng)
        zz = copy(z)
        for (f, v) in zip(frees, vals)
            zz[f] .= v[randperm(rng, length(f))]
        end
        return zz
    end
end

function _sv_conditional_sampler(m::ClusterRandomization, z::BitVector, focal::BitVector)
    free = [c for (c, g) in enumerate(m.groups) if !any(focal[g])]
    vals = [z[m.groups[c][1]] for c in free]
    return function (rng)
        zz = copy(z)
        perm = randperm(rng, length(free))
        for (c, v) in zip(free, vals[perm])
            zz[m.groups[c]] .= v
        end
        return zz
    end
end

# Any other mechanism: rejection sampling from the unconditional design.
function _sv_conditional_sampler(m::AssignmentMechanism, z::BitVector, focal::BitVector;
                                 max_tries::Int=10_000)
    F = findall(focal)
    zF = z[F]
    return function (rng)
        for _ in 1:max_tries
            zz = draw_assignment(rng, m)
            zz[F] == zF && return BitVector(zz)
        end
        throw(ArgumentError("could not draw an assignment matching the focal units' " *
                            "treatments in $max_tries tries under " *
                            "$(_sv_design_name(m)); use fewer focal units or a design " *
                            "with an exact conditional sampler"))
    end
end

# ---------------------------------------------------------------------------------
# Test statistics on focal units
# ---------------------------------------------------------------------------------

# Average over own-treatment strata of mean(y | exposed) − mean(y | unexposed).
function _sv_stat_difference(y, z, e)
    num = 0.0
    den = 0
    for s in (false, true)
        a = 0.0; na = 0; b = 0.0; nb = 0
        for i in eachindex(y)
            z[i] == s || continue
            if e[i] > 0
                a += y[i]; na += 1
            else
                b += y[i]; nb += 1
            end
        end
        if na > 0 && nb > 0
            num += (na + nb) * (a / na - b / nb)
            den += na + nb
        end
    end
    return den == 0 ? NaN : num / den
end

# Pooled within-own-treatment OLS slope of y on e.
function _sv_stat_slope(y, z, e)
    sxy = 0.0
    sxx = 0.0
    for s in (false, true)
        idx = findall(==(s), z)
        length(idx) < 2 && continue
        me = mean(e[idx])
        my = mean(y[idx])
        for i in idx
            sxy += (e[i] - me) * (y[i] - my)
            sxx += (e[i] - me)^2
        end
    end
    return sxx > 0 ? sxy / sxx : NaN
end

function _sv_statistic_function(statistic)
    statistic === :difference && return _sv_stat_difference
    statistic === :slope && return _sv_stat_slope
    statistic isa Function && return statistic
    throw(ArgumentError("statistic must be :difference, :slope or a function " *
                        "(y, z, e) -> Float64"))
end

"""
    spillover_fisher_test(data, outcome, treatment, s, design; unit,
                          focal=:random, focal_share=0.5,
                          exposure=NeighborExposure(:any), statistic=:difference,
                          alternative=:two_sided, draws=1999,
                          rng=Random.default_rng()) -> DiagnosticTest

Exact randomization test of the sharp null hypothesis of **no spillovers**, i.e.
that every unit's outcome depends only on its own treatment,
``Y_i(z) = Y_i(z_i)`` for all assignments ``z``, following Athey, Eckles and Imbens
(2018).

This null is not sharp in the usual sense: it does not determine the outcomes of
units whose own treatment would change under re-randomization. Athey, Eckles and
Imbens (2018) restore exactness by conditioning. A set of *focal* units is fixed, and
the test re-randomizes only the treatments of the other units, drawing from `design`
**conditionally on the focal units' own treatments**. Under the null the focal
units' outcomes are then fixed across the conditional reference set, while their
exposures (computed from the treatments of non-focal neighbours) vary, so the
conditional randomization distribution of any statistic of the focal outcomes and
exposures is known exactly. The resulting p-value is exact for any statistic, any
exposure definition and any interference structure, provided the design is the one
that assigned treatment; Basse, Feller and Toulis (2019) and Puelz, Basse, Feller
and Toulis (2022) develop the general conditioning approach and more powerful
choices of focal units.

The exposure specification and the statistic only affect power: a test built on the
wrong exposure remains valid but may have little power against the spillovers that
are actually present. Power also requires variation in the focal units' exposures
across the conditional reference set, which is why the focal set should be a
substantial share of the units but leave enough non-focal neighbours. Draws in which
the statistic is undefined are assigned the value 0; this defines a statistic on
every assignment and keeps the test exact. With `focal = :random`, focal units are
drawn with `rng` independently of the assignment. The p-value is the Monte Carlo
``(1 + \\#)/(1 + B)`` p-value of [`permutation_pvalue`](@ref). A non-rejection does
not show that there are no spillovers.

# Arguments
- `data`: table with one row per unit of `s`, matched by `unit`; outcomes are needed
  only for focal units.
- `outcome::Symbol`, `treatment::Symbol`: outcome and binary treatment columns.
- `s::InterferenceStructure`: the structure.
- `design::AssignmentMechanism`: the design over `structure_units(s)`. Bernoulli,
  complete, stratified and cluster designs have exact conditional samplers; other
  designs use rejection sampling, which can fail when many units are focal.

# Keywords
- `unit::Symbol`: unit-identifier column.
- `focal`: vector of focal unit identifiers, or `:random` (default) to draw
  `round(focal_share * N)` focal units; the identifiers used are stored in
  `details.focal`.
- `focal_share::Real`: share of focal units under `focal = :random`; default 0.5.
- `exposure::ExposureSpec`: exposure of the focal units (its first column is used);
  default `NeighborExposure(:any)`.
- `statistic`: `:difference` (default; average over own-treatment strata of the
  difference in mean outcomes between exposed (``e > 0``) and unexposed focal
  units), `:slope` (within-stratum least-squares slope of the outcome on the
  exposure, for continuous exposures) or a function `(y, z, e) -> Float64` of the
  focal outcomes, treatments and exposures.
- `alternative::Symbol`: `:two_sided` (default), `:greater` or `:less`.
- `draws::Integer`: number of conditional re-randomizations; default 1999.
- `rng::AbstractRNG`: random number generator.

# Returns
- [`DiagnosticTest`](@ref) with `details = (focal, n_focal, draws,
  statistic_draws)`.

# Examples
```julia
using DrSnow, DataFrames, StableRNGs
rng = StableRNG(2)
n = 120
A = zeros(Int, n, n)
for i in 1:n, j in (i + 1):n
    rand(rng) < 0.04 && (A[i, j] = A[j, i] = 1)
end
g = NetworkStructure(1:n, A)
design = CompleteRandomization(n, 40)
z = draw_assignment(rng, design)
share = coalesce.(compute_exposure(g, z, NeighborExposure(:share)).share, 0.0)
df = DataFrame(id=1:n, z=Int.(z), y=0.5 .* z .+ 1.0 .* share .+ randn(rng, n))
t = spillover_fisher_test(df, :y, :z, g, design; unit=:id, rng=rng)
pvalue(t)
```

# References
- Athey, S., Eckles, D., & Imbens, G. W. (2018). Exact p-values for network
  interference. *Journal of the American Statistical Association*, 113(521),
  230–240.
- Basse, G. W., Feller, A., & Toulis, P. (2019). Randomization tests of causal
  effects under interference. *Biometrika*, 106(2), 487–494.
- Puelz, D., Basse, G., Feller, A., & Toulis, P. (2022). A graph-theoretic approach
  to randomization tests of causal effects under general interference. *Journal of
  the Royal Statistical Society: Series B (Statistical Methodology)*, 84(1),
  174–204.
"""
function spillover_fisher_test(data, outcome::Symbol, treatment::Symbol,
                               s::InterferenceStructure, design::AssignmentMechanism;
                               unit::Symbol, focal=:random, focal_share::Real=0.5,
                               exposure::ExposureSpec=NeighborExposure(:any),
                               statistic=:difference, alternative::Symbol=:two_sided,
                               draws::Integer=1999,
                               rng::AbstractRNG=Random.default_rng())
    context = "spillover_fisher_test"
    n_units(design) == n_units(s) ||
        throw(DimensionMismatch("$context: design and structure sizes differ"))
    draws >= 1 || throw(ArgumentError("$context: draws must be positive"))
    require_columns(data, [outcome]; context=context)
    z = _sv_cross_section_treatment(data, treatment, s, unit, context)
    _sv_check_support(design, z)
    n = n_units(s)
    fmask = falses(n)
    if focal === :random
        0 < focal_share < 1 || throw(ArgumentError("$context: focal_share must be in " *
                                                   "(0, 1)"))
        nf = clamp(round(Int, focal_share * n), 1, n - 1)
        fmask[randperm(rng, n)[1:nf]] .= true
    else
        for u in focal
            k = get(s.index, u, 0)
            k == 0 && throw(ArgumentError("$context: focal unit $(repr(u)) is not in " *
                                          "the structure"))
            fmask[k] = true
        end
        (any(fmask) && !all(fmask)) ||
            throw(ArgumentError("$context: focal units must be a non-empty proper " *
                                "subset of the units"))
    end
    df = DataFrame(data; copycols=false)
    idx = _sv_rows_to_index(s, df[!, unit], context)
    y = fill(NaN, n)
    for (r, i) in enumerate(idx)
        fmask[i] || continue
        v = df[r, outcome]
        (ismissing(v) || !isfinite(v)) &&
            throw(ArgumentError("$context: missing outcome for focal unit " *
                                "$(repr(s.ids[i]))"))
        y[i] = Float64(v)
    end
    p = _sv_prepare(s, exposure)
    stat = _sv_statistic_function(statistic)
    F = findall(fmask)
    e_obs = _sv_eval_vector(p, z)[:, 1]
    use = [i for i in F if !isnan(e_obs[i])]        # exposure undefined: isolates
    length(use) >= 2 || throw(ArgumentError("$context: fewer than two focal units with " *
                                            "a defined exposure"))
    T(zz, e) = stat(y[use], zz[use], e[use])
    t_obs = T(z, e_obs)
    isfinite(t_obs) || throw(ArgumentError("$context: the statistic is undefined on the " *
                                           "observed assignment (no exposure variation " *
                                           "among focal units within treatment strata)"))
    sampler = _sv_conditional_sampler(design, z, fmask)
    tdraws = zeros(draws)
    for b in 1:draws
        zz = sampler(rng)
        e = _sv_eval_vector(p, zz)[:, 1]
        any(i -> isnan(e[i]), use) &&
            throw(ArgumentError("$context: focal exposure undefined under a " *
                                "re-randomized assignment"))
        v = T(zz, e)
        tdraws[b] = isfinite(v) ? v : 0.0
    end
    pv = permutation_pvalue(t_obs, tdraws; alternative=alternative)
    stname = statistic isa Symbol ? string(statistic) : "user-defined"
    return DiagnosticTest("Fisher randomization test of no spillovers (focal units)",
        "no spillovers: each unit's outcome depends only on its own treatment " *
        "(exposure: $(join(exposure_columns(exposure), ",")))",
        t_obs, pv;
        method="conditional randomization test given $(length(F)) focal units' " *
               "treatments (Athey–Eckles–Imbens 2018); $(draws) draws from " *
               "$(_sv_design_name(design)); statistic: $stname",
        note="Exact for the sharp null; power depends on the focal units, the exposure " *
             "definition and the statistic. A non-rejection is not evidence that the " *
             "no-interference assumption holds.",
        details=(focal=s.ids[F], n_focal=length(use), draws=draws,
                 statistic_draws=tdraws))
end

# ---------------------------------------------------------------------------------
# Balance of covariates across exposure levels
# ---------------------------------------------------------------------------------

"""
    exposure_balance_test(data, treatment, s, design; unit, covariates,
                          exposure=NeighborExposure(:any), among=:all,
                          draws=1999, rng=Random.default_rng()) -> DiagnosticTest

Randomization test of covariate balance across exposure levels, calibrated by the
assignment mechanism.

For each pre-treatment covariate the statistic is the difference in means between
exposed (``e > 0``) and unexposed units (among all, untreated or treated units),
divided by the covariate's standard deviation over all units; the joint statistic
is the largest absolute standardized difference. Both are referred to their
distribution over assignments drawn from `design`. The calibration matters: under
interference, exposure is generally *not* balanced in the naive sense even in a
perfectly randomized experiment, because well-connected or centrally located units
are exposed more often, and comparing the differences with a t distribution would
reject far too often. Referring them to the design's own distribution gives a test
with correct size under the null. Draws in which a difference is undefined (no
exposed or no unexposed units) contribute 0.

The null hypothesis is that treatment was assigned by `design`; this is a check of
the assumed assignment mechanism (for example of an "as-if random" natural
experiment), not of outcomes or of the exposure mapping. A non-rejection does not
show that the design is correct. Imbalance that the design itself creates between
exposed and unexposed units is the reason design-based estimators weight by
exposure probabilities ([`exposure_effects`](@ref)).

# Arguments
- `data`: table with one row per unit of `s`, matched by `unit`.
- `treatment::Symbol`: binary treatment column.
- `s::InterferenceStructure`: the structure.
- `design::AssignmentMechanism`: the design over `structure_units(s)`.

# Keywords
- `unit::Symbol`: unit-identifier column.
- `covariates::Vector{Symbol}`: pre-treatment covariates (no missing values, not
  constant).
- `exposure::ExposureSpec`: exposure definition (first column used); default
  `NeighborExposure(:any)`.
- `among::Symbol`: compare exposed and unexposed units among `:all` (default),
  `:control` or `:treated` units.
- `draws::Integer`: number of re-randomizations; default 1999.
- `rng::AbstractRNG`: random number generator.

# Returns
- [`DiagnosticTest`](@ref); `details.table` holds the per-covariate standardized
  differences and their randomization p-values (unadjusted for multiplicity).

# Examples
```julia
using DrSnow, DataFrames, StableRNGs
rng = StableRNG(3)
n = 100
A = zeros(Int, n, n)
for i in 1:n, j in (i + 1):n
    rand(rng) < 0.05 && (A[i, j] = A[j, i] = 1)
end
g = NetworkStructure(1:n, A)
design = CompleteRandomization(n, 40)
df = DataFrame(id=1:n, z=Int.(draw_assignment(rng, design)),
               age=30 .+ 5 .* randn(rng, n), degree=vec(sum(A; dims=2)))
exposure_balance_test(df, :z, g, design; unit=:id, covariates=[:age, :degree], rng=rng)
```

# References
- Aronow, P. M., & Samii, C. (2017). Estimating average causal effects under general
  interference, with application to a social network experiment. *Annals of Applied
  Statistics*, 11(4), 1912–1947.
- Imbens, G. W., & Rubin, D. B. (2015). *Causal Inference for Statistics, Social,
  and Biomedical Sciences: An Introduction*. Cambridge University Press.
"""
function exposure_balance_test(data, treatment::Symbol, s::InterferenceStructure,
                               design::AssignmentMechanism; unit::Symbol,
                               covariates::Vector{Symbol},
                               exposure::ExposureSpec=NeighborExposure(:any),
                               among::Symbol=:all, draws::Integer=1999,
                               rng::AbstractRNG=Random.default_rng())
    context = "exposure_balance_test"
    isempty(covariates) && throw(ArgumentError("$context: supply `covariates`"))
    among in (:all, :control, :treated) ||
        throw(ArgumentError("$context: among must be :all, :control or :treated"))
    n_units(design) == n_units(s) ||
        throw(DimensionMismatch("$context: design and structure sizes differ"))
    require_columns(data, covariates; context=context)
    z = _sv_cross_section_treatment(data, treatment, s, unit, context)
    _sv_check_support(design, z)
    df = DataFrame(data; copycols=false)
    idx = _sv_rows_to_index(s, df[!, unit], context)
    n = n_units(s)
    X = zeros(n, length(covariates))
    for (c, cov) in enumerate(covariates), (r, i) in enumerate(idx)
        v = df[r, cov]
        (ismissing(v) || !isfinite(v)) &&
            throw(ArgumentError("$context: covariate `$cov` is missing/non-finite"))
        X[i, c] = Float64(v)
    end
    sds = vec(std(X; dims=1))
    any(iszero, sds) && throw(ArgumentError("$context: a covariate is constant"))
    p = _sv_prepare(s, exposure)
    function stats(zz)
        e = _sv_eval_vector(p, zz)[:, 1]
        keep = [!isnan(e[i]) && (among === :all || zz[i] == (among === :treated))
                for i in 1:n]
        ex = keep .& (e .> 0)
        un = keep .& (e .== 0)
        (any(ex) && any(un)) || return fill(NaN, length(covariates))
        return [(mean(X[ex, c]) - mean(X[un, c])) / sds[c] for c in eachindex(sds)]
    end
    d_obs = stats(z)
    all(isfinite, d_obs) || throw(ArgumentError("$context: observed assignment has no " *
                                                "exposed or no unexposed units"))
    D = zeros(draws, length(covariates))
    for b in 1:draws
        d = stats(draw_assignment(rng, design))
        D[b, :] .= map(x -> isfinite(x) ? x : 0.0, d)
    end
    m_obs = maximum(abs, d_obs)
    m_draw = vec(maximum(abs.(D); dims=2))
    pv = permutation_pvalue(m_obs, m_draw; alternative=:greater)
    table = DataFrame(covariate=covariates, std_difference=d_obs,
                      pvalue=[permutation_pvalue(d_obs[c], D[:, c]) for c in
                              eachindex(covariates)])
    return DiagnosticTest("Randomization balance test across exposure levels",
        "treatment was assigned by the stated design ($(_sv_design_name(design)))",
        m_obs, pv;
        method="max |standardized difference| (exposed − unexposed, among $among), " *
               "$(draws) draws from the design",
        note="Calibrated by the assignment mechanism, so structural differences " *
             "between exposed and unexposed units do not cause rejections. A " *
             "non-rejection does not show that the design is correct.",
        details=(table=table, draws=draws))
end

# ---------------------------------------------------------------------------------
# Moran's I of the treatment vector
# ---------------------------------------------------------------------------------

function _sv_moran(x::AbstractVector{<:Real}, W::SparseMatrixCSC, S0::Real)
    d = x .- mean(x)
    den = sum(abs2, d)
    den > 0 || return NaN
    return length(x) / S0 * dot(d, W * d) / den
end

# Cliff–Ord moments of Moran's I under random permutation of `x`.
function _sv_moran_moments(x::AbstractVector{<:Real}, W::SparseMatrixCSC)
    n = length(x)
    n >= 4 || return (NaN, NaN)
    S0 = sum(W)
    Wsym = W + transpose(W)
    S1 = 0.5 * sum(abs2, nonzeros(Wsym))
    S2 = sum(abs2, vec(sum(W; dims=2)) .+ vec(sum(W; dims=1)))
    d = x .- mean(x)
    b2 = n * sum(d .^ 4) / sum(abs2, d)^2
    EI = -1 / (n - 1)
    EI2 = (n * ((n^2 - 3n + 3) * S1 - n * S2 + 3S0^2) -
           b2 * ((n^2 - n) * S1 - 2n * S2 + 6S0^2)) / ((n - 1) * (n - 2) * (n - 3) * S0^2)
    return EI, EI2 - EI^2
end

"""
    treatment_moran_test(data, treatment, s, design; unit, radius=nothing,
                         weights=nothing, row_standardize=true, alternative=:greater,
                         draws=1999, rng=Random.default_rng()) -> DiagnosticTest

Moran's I of the treatment vector — the spatial or network autocorrelation of
treatment — referred to its randomization distribution under `design`.

Moran's (1950) statistic is
``I = (N / S_0) \\, \\tilde z' W \\tilde z / \\tilde z' \\tilde z``, where
``\\tilde z`` is the demeaned treatment vector, ``W`` a weight matrix with zero
diagonal and ``S_0`` the sum of its entries. A large ``I`` means that treated units
are clustered in space or in the network. In a natural experiment claimed to be
"as-if randomly" assigned across units, clustering beyond what the stated mechanism
produces is a warning sign (treatment may follow spatially correlated
determinants of the outcome) and also raises the exposure of treated units'
neighbours, which matters for spillover analyses. In a design that randomizes
clusters, clustering is expected, and the design should be declared (e.g.
[`ClusterRandomization`](@ref)) so that the reference distribution reflects it.

The primary p-value is always the Monte Carlo randomization p-value under `design`.
For complete randomization, and for Bernoulli designs with a common probability
(conditionally on the number treated), the exact permutation mean and variance of
``I`` (Cliff and Ord 1981) and a normal-approximation p-value are reported in
`details` as a secondary output; the normal approximation can be poor for sparse
weight matrices or few treated units. The test concerns the assignment mechanism,
not interference in outcomes, and a non-rejection does not show that treatment was
randomly assigned.

# Arguments
- `data`: table with one row per unit of `s`, matched by `unit`.
- `treatment::Symbol`: binary treatment column.
- `s::InterferenceStructure`: the structure.
- `design::AssignmentMechanism`: the design over `structure_units(s)`.

# Keywords
- `unit::Symbol`: unit-identifier column.
- `radius`: neighbourhood radius for spatial structures, used to build
  `W = neighbor_matrix(s; radius)` when `weights` is not given.
- `weights`: alternatively, an ``N × N`` weight matrix in the order of
  [`structure_units`](@ref) (its diagonal is set to zero).
- `row_standardize::Bool`: divide each row of ``W`` by its sum; default `true`.
- `alternative::Symbol`: `:greater` (clustering; default), `:less` (dispersion) or
  `:two_sided`.
- `draws::Integer`: number of re-randomizations; default 1999.
- `rng::AbstractRNG`: random number generator.

# Returns
- [`DiagnosticTest`](@ref) with `details = (expected, variance, z, pvalue_normal,
  draws)`; the analytic entries are `nothing` for other designs.

# Examples
```julia
using DrSnow, DataFrames, StableRNGs
rng = StableRNG(4)
n = 80
x, y = 10 .* rand(rng, n), 10 .* rand(rng, n)
s = SpatialStructure(1:n; x=x, y=y)
df = DataFrame(id=1:n, z=Int.(x .< 4))                  # treatment clustered in space
treatment_moran_test(df, :z, s, CompleteRandomization(n, count(==(1), df.z));
                     unit=:id, radius=2.0, rng=rng)
```

# References
- Moran, P. A. P. (1950). Notes on continuous stochastic phenomena. *Biometrika*,
  37(1/2), 17–23.
- Cliff, A. D., & Ord, J. K. (1981). *Spatial Processes: Models and Applications*.
  Pion.
"""
function treatment_moran_test(data, treatment::Symbol, s::InterferenceStructure,
                              design::AssignmentMechanism; unit::Symbol, radius=nothing,
                              weights=nothing, row_standardize::Bool=true,
                              alternative::Symbol=:greater, draws::Integer=1999,
                              rng::AbstractRNG=Random.default_rng())
    context = "treatment_moran_test"
    n = n_units(s)
    n_units(design) == n || throw(DimensionMismatch("$context: design and structure " *
                                                    "sizes differ"))
    z = _sv_cross_section_treatment(data, treatment, s, unit, context)
    _sv_check_support(design, z)
    W = if weights === nothing
        neighbor_matrix(s; radius=radius)
    else
        size(weights) == (n, n) || throw(DimensionMismatch("$context: weights must be " *
                                                           "$n × $n"))
        SparseMatrixCSC{Float64,Int}(sparse(Float64.(weights)))
    end
    W = copy(W)
    for i in 1:n
        W[i, i] = 0.0
    end
    dropzeros!(W)
    nnz(W) > 0 || throw(ArgumentError("$context: the weight matrix has no links"))
    if row_standardize
        rs = vec(sum(W; dims=2))
        W = SparseMatrixCSC(Diagonal([r > 0 ? 1 / r : 0.0 for r in rs]) * W)
    end
    S0 = sum(W)
    zf = Float64.(z)
    I_obs = _sv_moran(zf, W, S0)
    isfinite(I_obs) || throw(ArgumentError("$context: treatment is constant"))
    Id = zeros(draws)
    for b in 1:draws
        v = _sv_moran(Float64.(draw_assignment(rng, design)), W, S0)
        Id[b] = isfinite(v) ? v : 0.0
    end
    pv = permutation_pvalue(I_obs, Id; alternative=alternative)
    analytic = design isa CompleteRandomization ||
               (design isa BernoulliAssignment && allequal(design.p))
    EI, VI = analytic ? _sv_moran_moments(zf, W) : (nothing, nothing)
    zstat = analytic && VI > 0 ? (I_obs - EI) / sqrt(VI) : nothing
    pn = zstat === nothing ? nothing :
         alternative === :greater ? ccdf(Normal(), zstat) :
         alternative === :less ? cdf(Normal(), zstat) : two_sided_pvalue(zstat)
    return DiagnosticTest("Moran's I of treatment (randomization test)",
        "treatment was assigned by the stated design ($(_sv_design_name(design)))",
        I_obs, pv;
        method="randomization distribution of Moran's I, $(draws) draws; " *
               (row_standardize ? "row-standardized" : "raw") * " weights",
        note="Tests the assignment mechanism, not interference in outcomes. The " *
             "Cliff–Ord moments in `details` are exact permutation moments under " *
             "complete randomization." *
             (analytic ? "" : " (Not reported for this design.)"),
        details=(expected=EI, variance=VI, z=zstat, pvalue_normal=pn, draws=draws))
end
