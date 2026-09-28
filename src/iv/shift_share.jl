# Shift-share (Bartik) instruments: construction, estimation with exposure-robust
# inference (Adão, Kolesár & Morales 2019; Borusyak, Hull & Jaravel 2022),
# Rotemberg-weight decompositions (Goldsmith-Pinkham, Sorkin & Swift 2020), and
# recentering / randomization inference with counterfactual shocks (Borusyak & Hull
# 2023).
#
# Notation: regions i = 1..n (rows of `data`), sectors k = 1..K with shares
# s_ik (columns `shares`) and shocks g_k; instrument B_i = Σ_k s_ik g_k. All
# computations use sqrt(weight)-transformed data with the controls (covariates,
# fixed effects, intercept) projected out, as in the rest of the IV area.

"""Align shocks with the share columns: a vector in the same order, or a dictionary
keyed by share-column name (`Symbol` or `String`)."""
function _iv_ss_shocks(shares::Vector{Symbol}, shocks)
    if shocks isa AbstractDict
        g = zeros(length(shares))
        for (k, s) in enumerate(shares)
            v = haskey(shocks, s) ? shocks[s] :
                haskey(shocks, string(s)) ? shocks[string(s)] :
                throw(ArgumentError("no shock given for share column `$s`"))
            g[k] = float(v)
        end
        return g
    end
    length(shocks) == length(shares) ||
        throw(ArgumentError("`shocks` must have one entry per share column " *
                            "($(length(shares))), got $(length(shocks))"))
    g = Float64.(collect(shocks))
    all(isfinite, g) || throw(ArgumentError("shocks must be finite"))
    return g
end

function _iv_ss_share_matrix(sub::AbstractDataFrame, shares::Vector{Symbol})
    S = Matrix{Float64}(hcat([Float64.(sub[!, c]) for c in shares]...))
    all(>=(0), S) || @warn "some shares are negative"
    return S
end

"""
    shift_share_instrument(data, shares, shocks) -> Vector{Union{Missing,Float64}}

Construct the shift-share (Bartik) instrument from exposure shares and shocks.

A shift-share instrument combines a set of shocks ``g_k`` to sectors (industries,
products, origin countries) ``k = 1, \\dots, K`` with each unit's (region's) exposure
shares ``s_{ik}`` to those sectors,

```math
B_i = \\sum_{k=1}^K s_{ik}\\, g_k ,
```

typically predicting local labour-demand or trade shocks from national industry
growth and a region's initial industry composition (Bartik 1991). The construction
itself is mechanical; the identifying assumption lies elsewhere. Goldsmith-Pinkham,
Sorkin and Swift (2020) show that 2SLS with ``B_i`` as the instrument is numerically
equivalent to a GMM estimator that uses the shares as instruments, so that
identification can rest on the exogeneity of the shares; Borusyak, Hull and Jaravel
(2022) show that it can instead rest on many as-good-as-randomly assigned shocks,
with the shares allowed to be endogenous. Which view is credible shapes the
diagnostics and the inference to report; see [`shift_share_iv`](@ref) and
[`rotemberg_weights`](@ref).

Shares need not sum to one across sectors (an incomplete share sum is common when a
residual sector is omitted); in that case Borusyak, Hull and Jaravel (2022) recommend
controlling for the sum of shares, which [`shift_share_iv`](@ref) does by default.
Negative shares trigger a warning.

# Arguments
- `data::AbstractDataFrame`: one row per region (unit).
- `shares`: the share columns ``s_{\\cdot k}``, a `Symbol` or vector of `Symbol`s, one
  per sector.
- `shocks`: the shocks ``g_k``, either a vector aligned with `shares` or a dictionary
  keyed by share-column name (`Symbol` or `String`).

# Returns
- A vector with one entry per row of `data`, `missing` where any share is missing.

# Examples
```julia
using DrSnow, DataFrames, StableRNGs
rng = StableRNG(1)
n, K = 200, 10
S = rand(rng, n, K)
S = S ./ sum(S; dims=2) .* 0.9                 # shares sum to 0.9 in every region
df = DataFrame(S, [Symbol("s", k) for k in 1:K])
g = randn(rng, K)
df.bartik = shift_share_instrument(df, [Symbol("s", k) for k in 1:K], g)
first(df.bartik, 3)
```

# References
- Bartik, T. J. (1991). *Who Benefits from State and Local Economic Development
  Policies?* W. E. Upjohn Institute for Employment Research.
- Goldsmith-Pinkham, P., Sorkin, I., & Swift, H. (2020). Bartik instruments: What,
  when, why, and how. *American Economic Review*, 110(8), 2586–2624.
- Borusyak, K., Hull, P., & Jaravel, X. (2022). Quasi-experimental shift-share
  research designs. *Review of Economic Studies*, 89(1), 181–213.
"""
function shift_share_instrument(data::AbstractDataFrame, shares, shocks)
    sh = _as_symbols(shares)
    isempty(sh) && throw(ArgumentError("at least one share column is required"))
    require_columns(data, sh; context="shift_share_instrument")
    _iv_check_numeric(data, sh, "shift_share_instrument")
    g = _iv_ss_shocks(sh, shocks)
    out = Vector{Union{Missing,Float64}}(missing, nrow(data))
    for i in 1:nrow(data)
        acc = 0.0
        ok = true
        for (k, c) in enumerate(sh)
            v = data[i, c]
            if ismissing(v)
                ok = false
                break
            end
            acc += v * g[k]
        end
        ok && (out[i] = acc)
    end
    return out
end

"""Internal shift-share design on transformed, partialled data."""
struct _IVSSDesign
    y::Vector{Float64}        # M(√w y)
    x::Vector{Float64}        # M(√w x)
    B::Vector{Float64}        # M(√w B)
    S::Matrix{Float64}        # √w s (not partialled), n × K
    g::Vector{Float64}
    sw::Vector{Float64}
    ctrl::Vector{AbstractMatrix{Float64}}
    n::Int
end

function _iv_ss_design(sub, outcome, treatment, shares, g, covariates, fe, weights;
                       extra=nothing)
    n = nrow(sub)
    w = weights === nothing ? ones(n) : Float64.(sub[!, weights])
    all(>(0), w) || throw(ArgumentError("weights must be strictly positive"))
    sw = sqrt.(w)
    ctrl, _ = _iv_ctrl_basis(sub, covariates, fe, sw)
    if extra !== nothing && size(extra, 2) > 0
        E = extra .* sw
        Ep = E - _iv_ctrl_proj(ctrl, E)
        Qe, re = _iv_orthobasis(Ep)
        re > 0 && push!(ctrl, Qe)
    end
    S = _iv_ss_share_matrix(sub, shares)
    Braw = S * g
    St = S .* sw
    M(v) = v .- _iv_ctrl_proj(ctrl, v)
    y = M(Float64.(sub[!, outcome]) .* sw)
    x = M(Float64.(sub[!, treatment]) .* sw)
    B = M(Braw .* sw)
    sum(abs2, B) > 1e-12 * max(1.0, sum(abs2, Braw .* sw)) ||
        throw(ArgumentError("the shift-share instrument is collinear with the controls"))
    return _IVSSDesign(y, x, B, St, g, sw, ctrl, n)
end

"""
AKM / AKM0 pieces (Adão, Kolesár & Morales 2019, as implemented in ShiftShareSE):
`ĥ` = coefficients of the partialled instrument on the shares; sector scores
`cR_k = ĥ_k Σᵢ sᵢₖ ε̂ᵢ`, `cW_k = ĥ_k Σᵢ sᵢₖ x̃ᵢ`, summed within sector clusters.
"""
function _iv_ss_akm(des::_IVSSDesign, β::Real, clusters)
    S = des.S
    Qs, r = _iv_orthobasis(S)
    # least-squares coefficients of B on S (drop collinear shares, as ShiftShareSE)
    F = qr(S, ColumnNorm())
    keep = sort(F.p[1:r])
    Sk = S[:, keep]
    h = Sk \ des.B
    e = des.y .- β .* des.x
    cR = h .* (Sk' * e)
    cW = h .* (Sk' * des.x)
    cY = h .* (Sk' * des.y)
    if clusters !== nothing
        cl = _iv_codes(collect(clusters)[keep])
        cR, cW, cY = (vec(_iv_group_sums(reshape(c, :, 1), cl)) for c in (cR, cW, cY))
    end
    return (cR=cR, cW=cW, cY=cY, RX=dot(des.x, des.B), rank=r)
end

"""AKM0 confidence set (null-imposed; quadratic inversion as in ShiftShareSE)."""
function _iv_ss_akm0_set(parts, β̂::Real, level::Real)
    cv = quantile(Normal(), 1 - (1 - level) / 2)
    # accept β0 iff (β̂ − β0)² RX² ≤ cv² Σ_k (cY_k − β0 cW_k)²  (cR0 = cY − β0 cW)
    RX = parts.RX
    a2 = RX^2 - cv^2 * sum(abs2, parts.cW)
    a1 = -2 * β̂ * RX^2 + 2 * cv^2 * dot(parts.cY, parts.cW)
    a0 = β̂^2 * RX^2 - cv^2 * sum(abs2, parts.cY)
    iv = _iv_quadratic_region(a2, a1, a0)
    pf = b0 -> begin
        isinf(b0) && return 2 * ccdf(Normal(), abs(RX) / sqrt(sum(abs2, parts.cW)))
        se0 = sqrt(sum(abs2, parts.cY .- b0 .* parts.cW)) / abs(RX)
        se0 > 0 || return β̂ == b0 ? 1.0 : 0.0
        2 * ccdf(Normal(), abs(β̂ - b0) / se0)
    end
    return WeakIVConfidenceSet("AKM0 (Adão, Kolesár & Morales 2019)", level,
                               _iv_set_kind(iv), iv, cv, β̂, pf)
end

"""Shock-level aggregation and IV (Borusyak, Hull & Jaravel 2022)."""
function _iv_ss_bhj(des::_IVSSDesign, q::Matrix{Float64}, clusters)
    S = des.S
    sk = S' * des.sw                              # Σᵢ wᵢ sᵢₖ (exposure weights)
    any(<=(0), sk) && throw(ArgumentError("every sector needs positive total exposure " *
                                          "for the shock-level regression"))
    ȳ = (S' * des.y) ./ sk                        # Σ wᵢ sᵢₖ y⊥ᵢ / sₖ (transformed)
    x̄ = (S' * des.x) ./ sk
    # weighted shock-level IV of ȳ on x̄ with controls q, instrument g
    rs = sqrt.(sk)
    Qq, _ = _iv_orthobasis(q .* rs)
    P(v) = v .- Qq * (Qq' * v)
    gt, yt, xt = P(des.g .* rs), P(ȳ .* rs), P(x̄ .* rs)
    den = dot(gt, xt)
    β = dot(gt, yt) / den
    ε = yt .- β .* xt
    sc = gt .* ε
    if clusters !== nothing
        sc = vec(_iv_group_sums(reshape(sc, :, 1), _iv_codes(collect(clusters))))
    end
    V = sum(abs2, sc) / den^2
    tab = DataFrame(exposure=sk, shock=des.g, outcome=ȳ, treatment=x̄)
    return β, V, tab
end

"""
    ShiftShareIVEstimate <: CausalEstimate

Result of [`shift_share_iv`](@ref): a shift-share 2SLS estimate with exposure-robust
inference.

The point estimate is the region-level 2SLS coefficient of the outcome on the
treatment with the (optionally recentered) shift-share instrument. Because regions
with similar exposure shares have correlated residuals, conventional region-level
standard errors are generally too small (Adão, Kolesár and Morales 2019); the object
therefore stores all available standard errors and intervals side by side, and the
covariance returned by `vcov(r)` is the one selected with `se`. The AKM0 confidence
set and the randomization-inference set can be unbounded or a union of intervals
and are stored as [`WeakIVConfidenceSet`](@ref)s. The `StatsAPI` accessors `coef`,
`vcov`, `stderror`, `confint` and `coeftable` refer to the selected covariance; with
`se = :akm` or `:bhj` the reference distribution is normal (`dof_residual = Inf`).

# Fields
- `coef`, `vcov`, `coefnames`, `nobs`, `dof_residual`: the treatment coefficient and
  the covariance selected by `se`.
- `se_type::String`: label of the selected covariance.
- `inference::DataFrame`: every available standard error and interval, with columns
  `method`, `se`, `lower`, `upper`: region-level HC1 or cluster-robust, AKM, the
  BHJ shock-level regression, and the AKM0 set (bounds of its convex hull when it is
  bounded, with `se` the implied half-width divided by the critical value).
- `akm0_set::WeakIVConfidenceSet`: the null-imposed AKM0 confidence set (may be
  unbounded).
- `shock_level::DataFrame`: the BHJ shock-level data, with columns `sector`,
  `exposure` (``s_k = \\sum_i w_i s_{ik}``), `shock`, and the exposure-weighted
  residualized `outcome` and `treatment`.
- `n_shocks::Int`, `n_shock_clusters::Int`: number of shocks and of shock clusters
  (0 without clustering).
- `ri`: `nothing`, or a `NamedTuple` `(pvalue, set, draws)` with the Borusyak–Hull
  randomization-inference p-value for ``\\beta = 0``, the inverted confidence set and
  the number of counterfactual draws.
- `recentered::Bool`, `expected_instrument`: whether the instrument was recentered
  and, if so, the expected instrument ``\\mu_i`` (one entry per estimation-sample row).
- `level::Float64`, `estimand::String`, `estimand_note::String`: confidence level and
  estimand description.
- `first_stage::WeakIVDiagnostics`: region-level first-stage diagnostics (not
  exposure-robust).
- `iv::IVEstimate`: the region-level 2SLS fit with the (recentered) instrument and
  the implied controls.
"""
struct ShiftShareIVEstimate <: CausalEstimate
    coef::Vector{Float64}
    vcov::Matrix{Float64}
    coefnames::Vector{String}
    nobs::Int
    dof_residual::Float64
    se_type::String
    inference::DataFrame
    akm0_set::WeakIVConfidenceSet
    shock_level::DataFrame
    n_shocks::Int
    n_shock_clusters::Int
    ri::Union{Nothing,NamedTuple}
    recentered::Bool
    expected_instrument::Union{Nothing,Vector{Float64}}
    level::Float64
    estimand::String
    estimand_note::String
    first_stage::WeakIVDiagnostics
    iv::IVEstimate
end

StatsAPI.coef(r::ShiftShareIVEstimate) = r.coef
StatsAPI.vcov(r::ShiftShareIVEstimate) = r.vcov
StatsAPI.coefnames(r::ShiftShareIVEstimate) = r.coefnames
StatsAPI.nobs(r::ShiftShareIVEstimate) = r.nobs
StatsAPI.dof_residual(r::ShiftShareIVEstimate) = r.dof_residual
StatsAPI.confint(r::ShiftShareIVEstimate; level::Real=r.level) =
    invoke(StatsAPI.confint, Tuple{CausalEstimate}, r; level=level)
estimand(r::ShiftShareIVEstimate) = r.estimand
method_name(r::ShiftShareIVEstimate) =
    r.recentered ? "Shift-share IV (recentered)" : "Shift-share IV"

function show_details(io::IO, r::ShiftShareIVEstimate)
    println(io)
    @printf(io, "Shocks: %d%s; covariance: %s\n", r.n_shocks,
            r.n_shock_clusters > 0 ? " in $(r.n_shock_clusters) clusters" : "",
            r.se_type)
    println(io, "Inference (", round(Int, 100 * r.level), "%):")
    for row in eachrow(r.inference)
        @printf(io, "  %-28s se = %-10.4g [%s, %s]\n", row.method, row.se,
                _iv_fmt(row.lower), _iv_fmt(row.upper))
    end
    print(io, "  AKM0 set: ")
    show(io, r.akm0_set)
    println(io)
    if r.ri !== nothing
        @printf(io, "  Randomization inference (shock draws): p(β = 0) = %.4g\n",
                r.ri.pvalue)
    end
    fs = r.first_stage.first_stage[1]
    @printf(io, "First stage (region level): F = %.2f (%s)\n", fs.F,
            r.first_stage.vcov_type)
    println(io, "Estimand: ", r.estimand)
    println(io, "Note: ", r.estimand_note)
end

"""
    shift_share_iv(data, outcome, treatment, shares, shocks;
                   covariates=Symbol[], fe=Symbol[], weights=nothing,
                   se=:akm, shock_clusters=nothing, shock_covariates=nothing,
                   add_share_sum=true, shock_draws=nothing, cluster=nothing,
                   vcov=nothing, level=0.95) -> ShiftShareIVEstimate

Shift-share instrumental-variables regression with exposure-robust inference.

The function estimates the region-level model ``y_i = \\beta x_i + w_i'\\gamma +
\\varepsilon_i`` by 2SLS with the shift-share instrument ``B_i = \\sum_k s_{ik} g_k``
([`shift_share_instrument`](@ref)). Two identification strategies justify the
instrument. Under the *exogenous-shares* view of Goldsmith-Pinkham, Sorkin and Swift
(2020, GPSS) each share is a valid instrument and ``B_i`` aggregates them with the
shocks as weights; the diagnostics of [`rotemberg_weights`](@ref) are then the
relevant ones. Under the *exogenous-shocks* view of Borusyak, Hull and Jaravel (2022,
BHJ) the shares may be endogenous, but the shocks are as good as randomly assigned
with respect to the exposure-weighted unobservables, ``E[g_k \\mid \\bar\\varepsilon, s]
= q_k'\\mu``, and there are many shocks with no single one dominating the exposure
(the Herfindahl index of exposure weights vanishes). The shock view turns the
design into a quasi-experiment at the shock level: its effective sample size is the
number of shocks, not the number of regions. With heterogeneous effects, the
coefficient is a weighted average of region-level treatment effects whose weights
are non-negative only when the first stage is monotone in the shocks (BHJ); under
the GPSS view it combines just-identified share-IV estimands with Rotemberg weights
that can be negative.

Inference must account for the fact that regions with similar shares are exposed to
the same shocks and hence have correlated residuals, which conventional
heteroskedasticity- or region-cluster-robust standard errors ignore; Adão, Kolesár
and Morales (2019, AKM) document severe over-rejection of such standard errors in
placebo exercises. The available standard errors, all reported in `r.inference`,
are:

1. `:akm` (default): the AKM exposure-robust standard error,
   ``\\widehat{\\text{se}}^2 = \\sum_k (\\hat h_k \\sum_i s_{ik}\\hat\\varepsilon_i)^2 /
   (\\sum_i \\tilde x_i \\tilde B_i)^2``, with ``\\hat h`` the coefficients of the
   partialled instrument ``\\tilde B`` on the shares and ``\\hat\\varepsilon`` the 2SLS
   residual; the sector scores are summed within `shock_clusters` when these are
   given, and critical values are normal. The null-imposed AKM0 confidence set,
   which AKM find to have better size with few or concentrated shocks, is always
   computed (`r.akm0_set`, possibly unbounded); both are validated against the R
   package ShiftShareSE.
2. `:bhj`: the numerically equivalent shock-level IV regression of BHJ, in which the
   exposure-weighted averages of the residualized outcome and treatment,
   ``\\bar y_k`` and ``\\bar x_k``, are regressed on each other with instrument ``g_k``,
   weights ``s_k = \\sum_i w_i s_{ik}`` and shock-level controls (an intercept and
   `shock_covariates`), with heteroskedasticity-robust (HC0) or
   `shock_clusters`-clustered standard errors at the shock level. Its point estimate
   equals the region-level one when the region-level controls include
   ``\\sum_k s_{ik} q_k`` for every shock-level control ``q_k``, which is ensured
   automatically (`add_share_sum = true` adds the sum of shares, the counterpart of
   the shock-level intercept; `shock_covariates` adds ``\\sum_k s_{ik} q_k``).
3. `:standard`: the region-level HC1 or cluster-robust covariance chosen with
   `cluster` or `vcov`, which is not exposure-robust and is reported for comparison.

When exposure to the shocks is not random, for instance because some regions are
systematically more exposed to all shocks, Borusyak and Hull (2023) show that the
instrument must be recentered by its expected value under the shock-assignment
process. Pass `shock_draws`, a ``K \\times R`` matrix of counterfactual shock vectors
drawn from that process (for instance permutations of the shocks within clusters);
the instrument becomes ``\\tilde B_i = B_i - \\mu_i`` with ``\\mu_i = \\sum_k s_{ik}
R^{-1}\\sum_r g_k^{(r)}``, and randomization inference is added: for ``H_0: \\beta =
\\beta_0`` the statistic ``T(\\beta_0) = \\sum_i \\tilde B_i(\\tilde y_i - \\beta_0 \\tilde
x_i)`` is compared with its values under the counterfactual instruments, and the
p-value function is inverted exactly (`r.ri`). The validity of this inference rests
entirely on the user-supplied assignment process. In practice report the AKM (or
BHJ) interval, the AKM0 set, and, under the GPSS view, the Rotemberg decomposition;
check balance of the shocks against shock-level characteristics and of the
instrument against pre-period outcomes.

# Arguments
- `data::AbstractDataFrame`: one row per region; incomplete rows are dropped.
- `outcome::Symbol`, `treatment::Symbol`: the outcome ``y`` and the endogenous
  treatment ``x``.
- `shares`: the share columns, at least two (`Symbol`s).
- `shocks`: the shocks, a vector aligned with `shares` or a dictionary keyed by share
  name.

# Keywords
- `covariates::Vector{Symbol}`, `fe::Vector{Symbol}`: region-level controls and
  fixed effects (default none); singleton fixed-effect groups are not supported.
- `weights::Union{Nothing,Symbol}`: strictly positive regression weights (default
  `nothing`), e.g. population; they also define the exposure weights ``s_k``.
- `se::Symbol`: the covariance stored in `vcov(r)`, `:akm` (default), `:bhj` or
  `:standard`; the others are reported in `r.inference` regardless.
- `shock_clusters`: `nothing` (default) or a vector with one cluster label per shock,
  e.g. a coarser industry classification; used by AKM and BHJ.
- `shock_covariates`: `nothing` (default) or a ``K \\times m`` matrix of shock-level
  controls ``q_k``.
- `add_share_sum::Bool`: add ``\\sum_k s_{ik}`` as a region-level control (default
  `true`; it is added only when it varies or fixed effects are present).
- `shock_draws`: `nothing` (default) or a ``K \\times R`` matrix of counterfactual shock
  vectors with ``R \\ge 19``, which triggers recentering and randomization inference.
- `cluster`, `vcov`: region-level covariance for `se = :standard` and for the
  region-level row of `r.inference` (default HC1).
- `level::Real`: confidence level (default 0.95).

# Returns
- A [`ShiftShareIVEstimate`](@ref); `estimate(r)` is the 2SLS coefficient,
  `r.inference` compares the standard errors, `r.akm0_set` is the AKM0 set and
  `r.ri` holds the randomization inference when `shock_draws` is given.

# Examples
```julia
using DrSnow, DataFrames, StableRNGs
rng = StableRNG(2)
n, K = 400, 30
sh = [Symbol("s", k) for k in 1:K]
S = rand(rng, n, K) .^ 3
S = S ./ sum(S; dims=2)
g = randn(rng, K)
u = S * randn(rng, K) .+ 0.5 .* randn(rng, n)          # exposure-correlated errors
x = S * g .+ 0.5 .* u .+ 0.3 .* randn(rng, n)
y = 1.5 .* x .+ u
df = hcat(DataFrame(y=y, x=x), DataFrame(S, sh))
draws = reduce(hcat, [g[sortperm(rand(rng, K))] for _ in 1:199])  # permutations
r = shift_share_iv(df, :y, :x, sh, g; shock_draws=draws)
r.inference
r.akm0_set
r.ri.pvalue
```

# References
- Adão, R., Kolesár, M., & Morales, E. (2019). Shift-share designs: Theory and
  inference. *Quarterly Journal of Economics*, 134(4), 1949–2010.
- Borusyak, K., Hull, P., & Jaravel, X. (2022). Quasi-experimental shift-share
  research designs. *Review of Economic Studies*, 89(1), 181–213.
- Goldsmith-Pinkham, P., Sorkin, I., & Swift, H. (2020). Bartik instruments: What,
  when, why, and how. *American Economic Review*, 110(8), 2586–2624.
- Borusyak, K., & Hull, P. (2023). Nonrandom exposure to exogenous shocks.
  *Econometrica*, 91(6), 2155–2185.
- Bartik, T. J. (1991). *Who Benefits from State and Local Economic Development
  Policies?* W. E. Upjohn Institute for Employment Research.
"""
function shift_share_iv(data::AbstractDataFrame, outcome::Symbol, treatment::Symbol,
                        shares, shocks; covariates=Symbol[], fe=Symbol[],
                        weights::Union{Nothing,Symbol}=nothing, se::Symbol=:akm,
                        shock_clusters=nothing, shock_covariates=nothing,
                        add_share_sum::Bool=true, shock_draws=nothing,
                        cluster=nothing, vcov=nothing, level::Real=0.95)
    se in (:akm, :bhj, :standard) ||
        throw(ArgumentError("se must be :akm, :bhj or :standard, got :$se"))
    0 < level < 1 || throw(ArgumentError("level must be in (0, 1)"))
    sh = _as_symbols(shares)
    covs, fes = _as_symbols(covariates), _as_symbols(fe)
    K = length(sh)
    K >= 2 || throw(ArgumentError("at least two share columns are required"))
    g = _iv_ss_shocks(sh, shocks)
    if shock_clusters !== nothing
        length(shock_clusters) == K ||
            throw(ArgumentError("shock_clusters must have one entry per shock ($K)"))
    end
    q = shock_covariates === nothing ? zeros(K, 0) : Matrix{Float64}(shock_covariates)
    size(q, 1) == K || throw(ArgumentError("shock_covariates must have $K rows"))
    if shock_draws !== nothing
        size(shock_draws, 1) == K ||
            throw(ArgumentError("shock_draws must be a K × R matrix with K = $K"))
        size(shock_draws, 2) >= 19 ||
            throw(ArgumentError("shock_draws needs at least 19 counterfactual draws"))
    end
    vce = _iv_vcov_estimator(cluster, vcov)
    cols = unique(vcat([outcome, treatment], sh, covs, fes, _iv_cluster_names(vce),
                       weights === nothing ? Symbol[] : [weights]))
    require_columns(data, cols; context="shift_share_iv")
    _iv_check_numeric(data, vcat([outcome, treatment], sh), "shift_share_iv")
    mask = trues(nrow(data))
    for c in cols
        mask .&= .!ismissing.(data[!, c])
    end
    sub = disallowmissing(data[mask, cols])
    S = _iv_ss_share_matrix(sub, sh)
    n = nrow(sub)
    # automatic region-level controls implied by the shock-level controls
    extra = zeros(n, 0)
    if add_share_sum
        ssum = vec(sum(S; dims=2))
        if maximum(ssum) - minimum(ssum) > 1e-10 * max(1.0, maximum(abs, ssum)) ||
           !isempty(fes)
            extra = hcat(extra, ssum)
        end
    end
    size(q, 2) > 0 && (extra = hcat(extra, S * q))
    μ = nothing
    gB = g
    if shock_draws !== nothing
        gbar = vec(mean(Matrix{Float64}(shock_draws); dims=2))
        μ = S * gbar
    end
    tmp = copy(sub)
    tmp[!, :__ss_instrument__] = S * g .- (μ === nothing ? 0.0 : μ)
    extra_names = Symbol[]
    for j in 1:size(extra, 2)
        nm = Symbol("__ss_control_", j, "__")
        tmp[!, nm] = extra[:, j]
        push!(extra_names, nm)
    end
    ivr = iv_regression(tmp, outcome, [treatment], [:__ss_instrument__];
                        covariates=vcat(covs, extra_names), fe=fes, weights=weights,
                        vcov=vce, level=level)
    nobs(ivr) == n || throw(ArgumentError("shift_share_iv: singleton fixed-effect " *
                                          "groups are not supported; drop them first"))
    des = _iv_ss_design(sub, outcome, treatment, sh, g, covs, fes, weights; extra=extra)
    if μ !== nothing
        # recentered instrument: partial out of √w (B − μ)
        Bt = (S * g .- μ) .* des.sw
        des = _IVSSDesign(des.y, des.x, Bt .- _iv_ctrl_proj(des.ctrl, Bt), des.S,
                          des.g .- vec(mean(Matrix{Float64}(shock_draws); dims=2)),
                          des.sw, des.ctrl, des.n)
    end
    β = dot(des.B, des.y) / dot(des.B, des.x)
    isapprox(β, ivr.coef[1]; rtol=1e-6, atol=1e-10) ||
        error("internal: shift-share design does not reproduce 2SLS")
    cv = critical_value(level)
    parts = _iv_ss_akm(des, β, shock_clusters)
    se_akm = sqrt(sum(abs2, parts.cR)) / abs(parts.RX)
    akm0 = _iv_ss_akm0_set(parts, β, level)
    qb = hcat(ones(K), q)
    βb, Vb, shock_tab = _iv_ss_bhj(des, qb, shock_clusters)
    se_bhj = sqrt(Vb)
    shock_tab = hcat(DataFrame(sector=sh), shock_tab)
    se_reg = sqrt(ivr.vcov[1, 1])
    creg = critical_value(level, ivr.dof_residual)
    hull = isempty(akm0.intervals) ? (NaN, NaN) :
           (akm0.intervals[1][1], akm0.intervals[end][2])
    inference = DataFrame(method=["region-level " * ivr.vcov_type, "AKM",
                                  "BHJ shock-level", "AKM0 (hull of set)"],
                          se=[se_reg, se_akm, se_bhj,
                              akm0.kind === :bounded ? (hull[2] - hull[1]) / (2cv) : Inf],
                          lower=[β - creg * se_reg, β - cv * se_akm, βb - cv * se_bhj,
                                 hull[1]],
                          upper=[β + creg * se_reg, β + cv * se_akm, βb + cv * se_bhj,
                                 hull[2]])
    if se === :akm
        V, label, dof = fill(se_akm^2, 1, 1), "AKM exposure-robust", Inf
    elseif se === :bhj
        isapprox(βb, β; rtol=1e-6, atol=1e-10) ||
            throw(ArgumentError("the shock-level estimate differs from the region-level " *
                                "one; include the implied controls (add_share_sum) or " *
                                "use se = :akm"))
        V, label, dof = fill(Vb, 1, 1), "BHJ shock-level robust", Inf
    else
        V, label, dof = ivr.vcov[1:1, 1:1], ivr.vcov_type, ivr.dof_residual
    end
    ri = nothing
    if shock_draws !== nothing
        ri = _iv_ss_randomization(des, sub, S, Matrix{Float64}(shock_draws), μ, level)
    end
    ncl = shock_clusters === nothing ? 0 : length(unique(shock_clusters))
    note = "Consistent under many exogenous shocks (BHJ 2022) or exogenous shares " *
           "(GPSS 2020). With heterogeneous effects it is a weighted average of " *
           "region-level effects with weights that are non-negative only when the " *
           "first stage is monotone in the shocks."
    μ !== nothing && (note *= " The instrument is recentered by its expected value " *
                               "under the supplied shock draws (Borusyak & Hull 2023).")
    return ShiftShareIVEstimate([β], V, [string(treatment)], n, dof, label, inference,
                                akm0, shock_tab, K, ncl, ri, μ !== nothing, μ,
                                float(level),
                                "weighted average of region-level treatment effects",
                                note, ivr.first_stage, ivr)
end

"""Borusyak–Hull randomization inference with counterfactual shock draws: exact
inversion of the p-value function (it only changes at the crossing points)."""
function _iv_ss_randomization(des::_IVSSDesign, sub, S, draws, μ, level)
    R = size(draws, 2)
    ctrlM(v) = v .- _iv_ctrl_proj(des.ctrl, v)
    # T_r(β0) = a_r − β0 b_r with B⁽ʳ⁾ = S g⁽ʳ⁾ − μ (partialled)
    a0, b0 = dot(des.B, des.y), dot(des.B, des.x)
    a = zeros(R)
    b = zeros(R)
    for r in 1:R
        Br = ctrlM((S * draws[:, r] .- μ) .* des.sw)
        a[r] = dot(Br, des.y)
        b[r] = dot(Br, des.x)
    end
    tol(x) = 1e-9 * max(1.0, abs(x))
    pf = β0 -> begin
        if isinf(β0)
            return (1 + count(r -> abs(b[r]) >= abs(b0) - tol(b0), 1:R)) / (1 + R)
        end
        t = abs(a0 - β0 * b0)
        (1 + count(r -> abs(a[r] - β0 * b[r]) >= t - tol(t), 1:R)) / (1 + R)
    end
    # crossing points |a_r − βb_r| = |a0 − βb0|
    cand = Float64[]
    for r in 1:R
        for (num, den) in ((a[r] - a0, b[r] - b0), (a[r] + a0, b[r] + b0))
            abs(den) > 1e-14 * max(1.0, abs(b0)) && push!(cand, num / den)
        end
    end
    sort!(cand)
    α = 1 - level
    pts = isempty(cand) ? [0.0] :
          vcat(cand[1] - 1 - abs(cand[1]), [(cand[i] + cand[i + 1]) / 2
                                             for i in 1:(length(cand) - 1)],
               cand[end] + 1 + abs(cand[end]))
    acc = [pf(p) > α for p in pts]
    iv = Tuple{Float64,Float64}[]
    start = acc[1] ? -Inf : NaN
    for i in 1:(length(pts) - 1)
        if acc[i] != acc[i + 1]
            bd = cand[i]
            acc[i] ? push!(iv, (start, bd)) : (start = bd)
        end
    end
    acc[end] && push!(iv, (start, Inf))
    set = WeakIVConfidenceSet("randomization inference with counterfactual shocks " *
                              "(Borusyak & Hull 2023)", level, _iv_set_kind(iv), iv,
                              NaN, a0 / b0, pf)
    return (pvalue=pf(0.0), set=set, draws=R)
end

# ---------------------------------------------------------------------------
# Rotemberg weights (Goldsmith-Pinkham, Sorkin & Swift 2020)
# ---------------------------------------------------------------------------

"""
    RotembergDecomposition

Result of [`rotemberg_weights`](@ref): the decomposition of a shift-share (or
share-instrument 2SLS) estimate into just-identified share-level estimates.

The decomposition writes the estimate as ``\\hat\\beta = \\sum_k \\hat\\alpha_k
\\hat\\beta_k``, where ``\\hat\\beta_k`` is the just-identified IV estimate that uses
sector ``k``'s share as the only instrument and ``\\hat\\alpha_k`` is its Rotemberg
weight (Goldsmith-Pinkham, Sorkin and Swift 2020). The weights sum to one but can be
negative. Under the exogenous-shares interpretation, the sectors with the largest
``|\\hat\\alpha_k|`` are those whose share exogeneity matters most for the estimate
and deserve the most scrutiny (balance on pre-period characteristics, pre-trends);
heterogeneity of the ``\\hat\\beta_k`` among high-weight sectors signals either
heterogeneous effects or misspecification.

# Fields
- `table::DataFrame`: one row per sector, sorted by `|alpha|` (largest first), with
  columns `sector`, `alpha` (Rotemberg weight), `shock` (the shock; in panels the
  α-weighted average of the sector's period shocks, ``\\sum_t \\alpha_{kt} g_{kt} /
  \\alpha_k``; `missing` for `estimator = :tsls`), `beta` (the just-identified IV
  estimate with that share as the instrument; in panels ``\\sum_t \\alpha_{kt}
  \\beta_{kt} / \\alpha_k``; `missing` when not defined) and `first_stage_F`
  (homoskedastic first-stage F of the sector's instrument ``\\sum_t s_k 1\\{t\\} g_{kt}``,
  or of the share itself for `estimator = :tsls`).
- `by_period::Union{Nothing,DataFrame}`: in panels, one row per sector × period with
  `sector`, `period`, `alpha`, `shock`, `beta` and `first_stage_F` (of the
  sector-period share instrument); `nothing` in a cross-section.
- `estimate::Float64`: the decomposed estimate ``\\sum_k \\hat\\alpha_k \\hat\\beta_k``
  (Bartik 2SLS, or 2SLS with all share instruments for `estimator = :tsls`).
- `estimator::Symbol`: `:bartik` or `:tsls`.
- `negative_weight_sum::Float64`, `positive_weight_sum::Float64`: sums of the
  negative and of the positive weights.
- `top5_share::Float64`: share of ``\\sum_k |\\hat\\alpha_k|`` accounted for by the five
  largest ``|\\hat\\alpha_k|``.

# References
- Goldsmith-Pinkham, P., Sorkin, I., & Swift, H. (2020). Bartik instruments: What,
  when, why, and how. *American Economic Review*, 110(8), 2586–2624.
"""
struct RotembergDecomposition
    table::DataFrame
    by_period::Union{Nothing,DataFrame}
    estimate::Float64
    estimator::Symbol
    negative_weight_sum::Float64
    positive_weight_sum::Float64
    top5_share::Float64
end

function Base.show(io::IO, ::MIME"text/plain", r::RotembergDecomposition)
    println(io, "Rotemberg weights (Goldsmith-Pinkham, Sorkin & Swift 2020)",
            r.estimator === :tsls ? ", 2SLS with all share instruments" : "",
            r.by_period === nothing ? "" :
            ", aggregated over $(length(unique(r.by_period.period))) periods")
    _iv_printf(io, "Estimate: %.4g; sum of negative weights: %.4g; sum of " *
               "positive weights: %.4g\n", r.estimate, r.negative_weight_sum,
               r.positive_weight_sum)
    @printf(io, "Top 5 sectors account for %.1f%% of Σ|α|\n", 100 * r.top5_share)
    show(io, MIME"text/plain"(), first(r.table, min(5, nrow(r.table))); summary=false,
         eltypes=false)
    println(io)
end

"""Period-specific shocks as a K × T matrix aligned with the share columns and the
sorted periods: a dictionary `period => shocks` (vector or dictionary by share
name) or a K × T matrix."""
function _iv_ss_panel_shocks(sh::Vector{Symbol}, shocks, periods::Vector)
    K, T = length(sh), length(periods)
    if shocks isa AbstractMatrix
        size(shocks) == (K, T) ||
            throw(ArgumentError("a shock matrix must be K × T = $K × $T (share columns " *
                                "× sorted periods), got $(size(shocks))"))
        Gm = Float64.(shocks)
    elseif shocks isa AbstractDict
        Gm = zeros(K, T)
        for (t, p) in enumerate(periods)
            haskey(shocks, p) || throw(ArgumentError("no shocks given for period $p"))
            Gm[:, t] = _iv_ss_shocks(sh, shocks[p])
        end
    else
        throw(ArgumentError("with `period`, `shocks` must be a dictionary " *
                            "period => shocks or a K × T matrix"))
    end
    all(isfinite, Gm) || throw(ArgumentError("shocks must be finite"))
    return Gm
end

"""
    rotemberg_weights(data, outcome, treatment, shares, shocks; period=nothing,
                      estimator=:bartik, covariates=Symbol[], fe=Symbol[],
                      weights=nothing) -> RotembergDecomposition

Rotemberg-weight decomposition of a shift-share estimate into share-level
just-identified estimates (Goldsmith-Pinkham, Sorkin and Swift 2020).

Goldsmith-Pinkham, Sorkin and Swift (2020, GPSS) show that the Bartik 2SLS estimator
is numerically identical to a GMM estimator that uses the ``K`` shares as instruments
with weight matrix ``gg'``. If identification rests on the exogeneity of the shares,
the estimate is a weighted combination of the just-identified estimates that use one
share at a time,

```math
\\hat\\beta = \\sum_k \\hat\\alpha_k \\hat\\beta_k , \\qquad
\\hat\\alpha_k = \\frac{a_k\\, s_k' M x}{\\sum_{k'} a_{k'}\\, s_{k'}' M x}, \\qquad
\\hat\\beta_k = \\frac{s_k' M y}{s_k' M x},
```

where ``M`` projects out the controls (with weights) and ``a`` is the weight the
estimator places on each share instrument. With `estimator = :bartik` (default)
``a = g``, the shocks. With `estimator = :tsls`, ``a = \\hat\\pi``, the first-stage
coefficients of the treatment on all share instruments, which decomposes the
overidentified 2SLS estimator that uses the shares as separate instruments (GPSS,
Section IV); `shocks` may then be `nothing`. The weights sum to one. They show which
sectors drive the estimate and, since they can be negative, whether the estimate is a
convex combination of the share-specific estimands; with heterogeneous effects,
negative weights mean that the Bartik estimand need not lie within the range of the
share-level estimands. GPSS recommend reporting the top-weight sectors, their
``\\hat\\beta_k`` and first-stage strength, and examining the plausibility of the
exogeneity of their shares.

In panels (`period` given; one row per region × period) with period-specific shocks
``g_{kt}``, the instrument is ``B_{it} = \\sum_k s_{ikt} g_{kt}``, i.e. the share
instruments are ``s_k \\times 1\\{\\text{period} = t\\}``. The decomposition is computed
over the ``K \\times T`` sector-period instruments (`by_period`) and aggregated by
sector as in GPSS's replication code: ``\\alpha_k = \\sum_t \\alpha_{kt}``,
``\\beta_k = \\sum_t \\alpha_{kt}\\beta_{kt} / \\alpha_k`` and ``g_k = \\sum_t
\\alpha_{kt} g_{kt} / \\alpha_k``. Period fixed effects are usually included in `fe`.
The decomposition is descriptive: it carries no standard errors, and it is
uninformative about validity under the exogenous-shocks view of Borusyak, Hull and
Jaravel (2022), for which [`shift_share_iv`](@ref) provides the relevant inference.
The panel and overidentified decompositions are validated against GPSS's R package
`bartik.weight`.

# Arguments
- `data::AbstractDataFrame`: one row per region (or region × period); incomplete rows
  are dropped.
- `outcome::Symbol`, `treatment::Symbol`: the outcome and the endogenous treatment.
- `shares`: the share columns (at least two).
- `shocks`: in a cross-section, a vector aligned with `shares` or a dictionary by
  share name; in a panel, a dictionary `period => shocks` or a ``K \\times T`` matrix
  whose columns follow the sorted periods; ignored for `estimator = :tsls` (pass
  `nothing`).

# Keywords
- `period::Union{Nothing,Symbol}`: the period column for panels (default `nothing`,
  cross-section).
- `estimator::Symbol`: `:bartik` (default) or `:tsls`.
- `covariates`, `fe`, `weights`: controls, fixed effects and strictly positive
  weights, as in [`shift_share_iv`](@ref) (default none).

# Returns
- A [`RotembergDecomposition`](@ref); `rw.table` lists the sectors by absolute weight
  and `rw.estimate` reproduces the decomposed 2SLS estimate.

# Examples
```julia
using DrSnow, DataFrames, StableRNGs
rng = StableRNG(3)
n, K = 400, 15
sh = [Symbol("s", k) for k in 1:K]
S = rand(rng, n, K) .^ 2
S = S ./ sum(S; dims=2)
g = randn(rng, K)
w = randn(rng, n)
x = S * g .+ 0.3 .* w .+ 0.5 .* randn(rng, n)
y = 2.0 .* x .+ w .+ randn(rng, n)
df = hcat(DataFrame(y=y, x=x, w=w), DataFrame(S, sh))
rw = rotemberg_weights(df, :y, :x, sh, g; covariates=[:w])
first(rw.table, 5)
rotemberg_weights(df, :y, :x, sh, nothing; estimator=:tsls, covariates=[:w])
```

# References
- Goldsmith-Pinkham, P., Sorkin, I., & Swift, H. (2020). Bartik instruments: What,
  when, why, and how. *American Economic Review*, 110(8), 2586–2624.
- Borusyak, K., Hull, P., & Jaravel, X. (2022). Quasi-experimental shift-share
  research designs. *Review of Economic Studies*, 89(1), 181–213.
- Bartik, T. J. (1991). *Who Benefits from State and Local Economic Development
  Policies?* W. E. Upjohn Institute for Employment Research.
"""
function rotemberg_weights(data::AbstractDataFrame, outcome::Symbol, treatment::Symbol,
                           shares, shocks; period::Union{Nothing,Symbol}=nothing,
                           estimator::Symbol=:bartik, covariates=Symbol[], fe=Symbol[],
                           weights::Union{Nothing,Symbol}=nothing)
    estimator in (:bartik, :tsls) ||
        throw(ArgumentError("estimator must be :bartik or :tsls, got :$estimator"))
    sh = _as_symbols(shares)
    covs, fes = _as_symbols(covariates), _as_symbols(fe)
    K = length(sh)
    K >= 2 || throw(ArgumentError("at least two share columns are required"))
    if estimator === :bartik && shocks === nothing
        throw(ArgumentError("the Bartik decomposition needs `shocks`"))
    end
    cols = unique(vcat([outcome, treatment], sh, covs, fes,
                       weights === nothing ? Symbol[] : [weights],
                       period === nothing ? Symbol[] : [period]))
    require_columns(data, cols; context="rotemberg_weights")
    _iv_check_numeric(data, vcat([outcome, treatment], sh), "rotemberg_weights")
    mask = trues(nrow(data))
    for c in cols
        mask .&= .!ismissing.(data[!, c])
    end
    sub = disallowmissing(data[mask, cols])
    n = nrow(sub)
    periods = period === nothing ? Any[nothing] : sort(unique(sub[!, period]))
    T = length(periods)
    pidx = period === nothing ? ones(Int, n) :
           (d -> [d[v] for v in sub[!, period]])(Dict(p => t for (t, p) in
                                                      enumerate(periods)))
    Gm = if shocks === nothing
        zeros(K, T)
    elseif period === nothing
        reshape(_iv_ss_shocks(sh, shocks), K, 1)
    else
        _iv_ss_panel_shocks(sh, shocks, periods)
    end
    w = weights === nothing ? ones(n) : Float64.(sub[!, weights])
    all(>(0), w) || throw(ArgumentError("weights must be strictly positive"))
    sw = sqrt.(w)
    ctrl, _ = _iv_ctrl_basis(sub, covs, fes, sw)
    Mp(v) = v .- _iv_ctrl_proj(ctrl, v)
    S = _iv_ss_share_matrix(sub, sh)
    # sector-period share instruments s_k 1{t}, J = K·T columns (k fastest)
    J = K * T
    Z = zeros(n, J)
    for i in 1:n
        t = pidx[i]
        for k in 1:K
            Z[i, (t - 1) * K + k] = S[i, k] * sw[i]
        end
    end
    Zp = Mp(Z)
    x̃ = Mp(Float64.(sub[!, treatment]) .* sw)
    ỹ = Mp(Float64.(sub[!, outcome]) .* sw)
    sx = Zp' * x̃
    sy = Zp' * ỹ
    a = if estimator === :bartik
        vec(Gm)
    else
        J < n || throw(ArgumentError("2SLS with all $J share instruments needs more " *
                                     "observations than instruments (n = $n)"))
        F = qr(Zp, ColumnNorm())
        dR = abs.(diag(F.R))
        r = count(>(maximum(dR) * max(n, J) * eps() * 1e3), dR)
        r > 0 || throw(ArgumentError("the share instruments are collinear with the " *
                                     "controls"))
        piv = F.p[1:r]
        coefs = zeros(J)
        coefs[piv] = Zp[:, piv] \ x̃
        coefs
    end
    den = dot(a, sx)
    abs(den) > 1e-12 * max(1.0, norm(a) * norm(sx)) ||
        throw(ArgumentError("the first stage of the decomposed estimator is zero"))
    α = a .* sx ./ den
    scale = maximum(abs, sx)
    ok = abs.(sx) .> 1e-12 * scale
    βkt = Union{Missing,Float64}[ok[j] ? sy[j] / sx[j] : missing for j in 1:J]
    kx = sum(size(b, 2) for b in ctrl; init=0)
    Fstat(zc) = begin
        zz = dot(zc, zc)
        zz > 1e-14 * max(1.0, norm(x̃)^2) || return 0.0
        π = dot(zc, x̃) / zz
        res = x̃ .- π .* zc
        π^2 * zz / (sum(abs2, res) / max(1, n - kx - 1))
    end
    Fkt = [Fstat(Zp[:, j]) for j in 1:J]
    estimate = dot(a, sy) / den
    if T == 1
        tab = DataFrame(sector=sh, alpha=α, shock=Gm[:, 1], beta=βkt, first_stage_F=Fkt)
        byp = nothing
    else
        byp = DataFrame(sector=repeat(sh, T), period=repeat(periods; inner=K), alpha=α,
                        shock=vec(Gm), beta=βkt, first_stage_F=Fkt)
        αk = [sum(α[(t - 1) * K + k] for t in 1:T) for k in 1:K]
        agg(v) = map(1:K) do k
            idx = [(t - 1) * K + k for t in 1:T]
            abs(αk[k]) > 1e-14 || return missing
            any(j -> ismissing(v[j]) && α[j] != 0, idx) && return missing
            sum((α[j] == 0 ? 0.0 : α[j] * v[j]) for j in idx) / αk[k]
        end
        # first stage of the sector's Bartik component Σₜ s_k 1{t} g_kt
        Fk = [Fstat(Zp[:, [(t - 1) * K + k for t in 1:T]] *
                    (estimator === :bartik ? Gm[k, :] :
                     a[[(t - 1) * K + k for t in 1:T]])) for k in 1:K]
        tab = DataFrame(sector=sh, alpha=αk,
                        shock=estimator === :bartik ? agg(vec(Gm)) :
                              fill(missing, K),
                        beta=agg(βkt), first_stage_F=Fk)
    end
    if estimator === :tsls && T == 1
        tab.shock = fill(missing, K)
    end
    sort!(tab, :alpha; by=abs, rev=true)
    tot = sum(abs, tab.alpha)
    top5 = sum(abs.(tab.alpha[1:min(5, K)])) / tot
    return RotembergDecomposition(tab, byp, estimate, estimator,
                                  sum(min.(tab.alpha, 0.0)), sum(max.(tab.alpha, 0.0)),
                                  top5)
end
