# Difference-in-differences with a continuous treatment (Callaway, Goodman-Bacon &
# Sant'Anna 2024) for two periods: units receive a dose D ≥ 0 in the second period
# (D = 0: untreated comparison group). The dose-response of the outcome change among
# treated units is estimated with a B-spline sieve (OLS), and compared with the
# untreated group's mean change.

"""
    ContinuousDiDEstimate <: CausalEstimate

Dose-specific difference-in-differences estimates from [`did_continuous`](@ref):
two scalar summaries and two curves evaluated on a grid of doses.

The level curve estimates
``E[\\Delta Y \\mid D = d] - E[\\Delta Y \\mid D = 0]``, which equals
``ATT(d \\mid d)`` under parallel trends and ``ATE(d)`` under strong parallel trends.
The slope curve estimates ``\\partial E[\\Delta Y \\mid D = d] / \\partial d``, which
equals the average causal response ``ACR(d)`` under strong parallel trends but, under
parallel trends alone, ``ACRT(d \\mid d)`` plus a selection-bias term (Callaway,
Goodman-Bacon and Sant'Anna, forthcoming; see [`did_continuous`](@ref)). The two
coefficients average these curves over the distribution of doses among treated units.

# Fields
- `coef::Vector{Float64}`, `vcov::Matrix{Float64}`: the summaries
  `[ATT^o, ACRT^o]` (see [`did_continuous`](@ref) for their interpretation) and
  their influence-function covariance.
- `dose::Vector{Float64}`: evaluation grid of doses.
- `att::Vector{Float64}`, `att_se::Vector{Float64}`: the level curve on the grid and
  its pointwise standard errors.
- `acrt::Vector{Float64}`, `acrt_se::Vector{Float64}`: the slope curve on the grid
  and its pointwise standard errors.
- `att_vcov::Matrix{Float64}`, `acrt_vcov::Matrix{Float64}`: covariance of each curve
  across grid points (for simultaneous bands).
- `nobs::Int`, `n_treated::Int`, `n_control::Int`, `n_clusters::Int`: observations,
  treated units, untreated (``D = 0``) units and clusters.
- `settings::NamedTuple`: `degree`, `knots` (interior knots), `boundary` (the range
  of treated doses), the spline coefficients and the untreated mean change.
- `influence::Matrix{Float64}`: unit-level influence functions of the two
  coefficients.

# Accessors
`coef`, `vcov`, `stderror`, `coefnames`, `coeftable`, `nobs`,
[`confint(::ContinuousDiDEstimate)`](@ref) (also for the curves).

# References
- Callaway, B., Goodman-Bacon, A., & Sant'Anna, P. H. C. (forthcoming).
  Difference-in-differences with a continuous treatment. *American Economic
  Review*. Earlier version: NBER Working Paper 32117 (2024).
"""
struct ContinuousDiDEstimate <: CausalEstimate
    coef::Vector{Float64}
    vcov::Matrix{Float64}
    dose::Vector{Float64}
    att::Vector{Float64}
    att_se::Vector{Float64}
    acrt::Vector{Float64}
    acrt_se::Vector{Float64}
    att_vcov::Matrix{Float64}
    acrt_vcov::Matrix{Float64}
    nobs::Int
    n_treated::Int
    n_control::Int
    n_clusters::Int
    settings::NamedTuple
    influence::Matrix{Float64}
end

StatsAPI.coef(r::ContinuousDiDEstimate) = r.coef
StatsAPI.vcov(r::ContinuousDiDEstimate) = r.vcov
StatsAPI.nobs(r::ContinuousDiDEstimate) = r.nobs
StatsAPI.coefnames(::ContinuousDiDEstimate) = ["ATT^o", "ACRT^o"]
estimand(::ContinuousDiDEstimate) =
    "ATT^o = E[ATT(D|D) | D>0]; ACRT^o = E[ACRT(D|D) | D>0]"
method_name(r::ContinuousDiDEstimate) =
    "Continuous-treatment DiD (B-spline degree $(r.settings.degree), " *
    "$(length(r.settings.knots)) interior knots)"

function show_details(io::IO, r::ContinuousDiDEstimate)
    println(io)
    println(io, "Treated units: ", r.n_treated, ", untreated (D = 0): ", r.n_control)
    r.n_clusters > 0 && println(io, "Clusters: ", r.n_clusters)
    println(io, "ATT(d|d) is identified under parallel trends; ACRT and comparisons ",
            "across doses require strong parallel trends (see docs).")
end

"""
    confint(r::ContinuousDiDEstimate; level=0.95, curve=nothing, uniform=false,
            rng=Random.default_rng(), ndraws=10_000) -> Matrix{Float64}

Confidence intervals for the summary parameters of a continuous-treatment DiD, or
pointwise intervals and simultaneous bands for its dose-response curves.

Without `curve`, the result has one row per coefficient (`ATT^o`, `ACRT^o`) with
normal critical values. With `curve = :att` or `:acrt`, it has one row per grid
dose in `r.dose`; `uniform = true` replaces the pointwise normal critical value by
the `level` quantile of the maximum absolute t-statistic across the grid, simulated
from the estimated covariance of the curve (Montiel Olea and Plagborg-Møller, 2019),
so that the band covers the whole curve on the grid jointly. Statements about the
shape of the dose response, e.g. that it is flat, should rest on the simultaneous
band. The intervals inherit the limitations of the sieve standard errors described
in [`did_continuous`](@ref).

# Arguments
- `r::ContinuousDiDEstimate`: result of [`did_continuous`](@ref).

# Keywords
- `level::Real = 0.95`: confidence level.
- `curve = nothing`: `nothing` for the two coefficients, `:att` for the level curve
  or `:acrt` for the slope curve.
- `uniform::Bool = false`: simultaneous band over the grid (curves only).
- `rng::AbstractRNG = Random.default_rng()`, `ndraws::Integer = 10_000`: generator
  and number of draws for the simulated critical value.

# Returns
- `Matrix{Float64}`: lower and upper bounds in columns 1 and 2.

# References
- Montiel Olea, J. L., & Plagborg-Møller, M. (2019). Simultaneous confidence bands:
  Theory, implementation, and an application to SVARs. *Journal of Applied
  Econometrics*, 34(1), 1–17.
"""
function StatsAPI.confint(r::ContinuousDiDEstimate; level::Real=0.95, curve=nothing,
                          uniform::Bool=false, rng::AbstractRNG=Random.default_rng(),
                          ndraws::Integer=10_000)
    curve === nothing && return _did_pointwise_ci(r, level)
    curve in (:att, :acrt) || throw(ArgumentError("curve must be :att or :acrt"))
    b, s, V = curve === :att ? (r.att, r.att_se, r.att_vcov) :
              (r.acrt, r.acrt_se, r.acrt_vcov)
    c = uniform ? _uniform_critical_value(Float64[], V, Inf, level, rng, ndraws,
                                          eachindex(b)) : critical_value(level)
    return _did_ci_with(b, s, c)
end

# ---------------------------------------------------------------------------
# B-splines (Cox–de Boor)
# ---------------------------------------------------------------------------

function _did_bspline_knots(interior, lo, hi, degree)
    return vcat(fill(lo, degree + 1), sort(collect(interior)), fill(hi, degree + 1))
end

# Basis N_{j,p}(x), j = 1..(length(t) - p - 1); right endpoint included.
function _did_bspline_basis(x::Real, t::Vector{Float64}, p::Int)
    m = length(t) - 1
    N = zeros(m)
    hi = t[end]
    for j in 1:m
        if (t[j] <= x < t[j + 1]) || (x == hi && t[j] < t[j + 1] == hi)
            N[j] = 1.0
        end
    end
    for k in 1:p
        Nn = zeros(m - k)
        for j in 1:(m - k)
            a = t[j + k] - t[j]
            b = t[j + k + 1] - t[j + 1]
            v = 0.0
            a > 0 && (v += (x - t[j]) / a * N[j])
            b > 0 && (v += (t[j + k + 1] - x) / b * N[j + 1])
            Nn[j] = v
        end
        N = Nn
    end
    return N
end

function _did_bspline_deriv(x::Real, t::Vector{Float64}, p::Int)
    p == 0 && return zeros(length(t) - 1)
    Nl = _did_bspline_basis(x, t, p - 1)          # length(t) - p
    K = length(t) - p - 1
    d = zeros(K)
    for j in 1:K
        a = t[j + p] - t[j]
        b = t[j + p + 1] - t[j + 1]
        a > 0 && (d[j] += p / a * Nl[j])
        b > 0 && (d[j] -= p / b * Nl[j + 1])
    end
    return d
end

"""
    did_continuous(data, outcome, dose, unit, time; degree=3, num_knots=0,
                   knots=nothing, dose_grid=nothing, cluster=nothing,
                   weights=nothing) -> ContinuousDiDEstimate

Difference-in-differences with a continuous treatment (dose) in a two-period panel,
following Callaway, Goodman-Bacon and Sant'Anna (forthcoming).

In the second period each unit receives a dose ``D \\ge 0``; units with ``D = 0``
form the comparison group. Let ``Y_t(d)`` be potential outcomes under dose ``d``.
The paper distinguishes the effect of dose ``d`` among units that chose dose
``d'``, ``ATT(d \\mid d') = E[Y_2(d) - Y_2(0) \\mid D = d']``, from the population
effect ``ATE(d) = E[Y_2(d) - Y_2(0)]``, and defines the corresponding causal
responses to a marginal change in the dose,

```math
ACRT(d \\mid d') = \\frac{\\partial ATT(l \\mid d')}{\\partial l}\\Big|_{l = d},
\\qquad ACR(d) = \\frac{\\partial ATE(d)}{\\partial d}.
```

Under **parallel trends**, ``E[\\Delta Y(0) \\mid D = d] = E[\\Delta Y(0) \\mid D =
0]`` for all doses, the level comparison identifies
``ATT(d \\mid d) = E[\\Delta Y \\mid D = d] - E[\\Delta Y \\mid D = 0]``, the effect of
dose ``d`` for the units that received it. Comparisons *across* doses are, however,
not causal under parallel trends: the slope of the level curve is

```math
\\frac{\\partial E[\\Delta Y \\mid D = d]}{\\partial d}
= ACRT(d \\mid d) + \\frac{\\partial ATT(d \\mid l)}{\\partial l}\\Big|_{l = d},
```

the causal response plus a selection-bias term that is non-zero when units that
choose different doses would experience different effects of the same dose. Under
the stronger assumption of **strong parallel trends**, that the average trend of
``Y(d)`` among units choosing dose ``d`` equals its trend in the whole population
for every ``d`` (ruling out selection into doses on treatment effects), the level
curve identifies ``ATE(d)`` and its slope identifies ``ACR(d)``. Strong parallel
trends cannot be assessed with pre-treatment data from two periods and should be
justified on substantive grounds.

The estimator fits ``E[\\Delta Y \\mid D = d]`` among treated units with a B-spline
of degree `degree` (interior knots at dose quantiles, or at `knots`) and subtracts
the mean change of the untreated units. The two summary coefficients average the
curves over the treated dose distribution: the first, labelled `ATT^o`, is
``E[ATT(D \\mid D) \\mid D > 0]`` under parallel trends; the second, labelled
`ACRT^o`, is the average estimated slope ``E[\\partial E[\\Delta Y \\mid D = d] /
\\partial d \\,|_{d = D} \\mid D > 0]``, which equals the average of ``ACR(D)`` over
treated doses under strong parallel trends and otherwise contains the selection
bias above. Standard errors come from influence functions that treat the spline as a
fixed parametric sieve (knots fixed; heteroskedasticity-robust, clustered by unit or
by `cluster`), with normal critical values. They ignore the approximation bias of
the sieve (Chen, 2007), and with a few hundred treated units intervals for the
curves can undercover near the boundaries of the dose support. Fewer knots give
smoother curves with more bias; more knots give less bias with more variance.

# Arguments
- `data`: a long-format panel with exactly two periods (units observed in only one
  period are dropped with a warning).
- `outcome::Symbol`: outcome column.
- `dose::Symbol`: dose received in the second period (constant within unit or
  measured in the second period; `0` = untreated). Doses must be nonnegative.
- `unit::Symbol`: unit identifier.
- `time::Symbol`: time column with two distinct values.

# Keywords
- `degree::Integer = 3`: degree of the B-spline (3 is cubic).
- `num_knots::Integer = 0`: number of interior knots, placed at equally spaced
  quantiles of the treated doses; `0` fits a global polynomial of degree `degree`.
- `knots = nothing`: explicit interior knots (strictly inside the range of treated
  doses); overrides `num_knots`.
- `dose_grid = nothing`: doses at which the curves are reported; by default 50
  points between the 1st and 99th percentiles of the treated doses.
- `cluster::Union{Nothing,Symbol} = nothing`: a unit-invariant cluster column.
- `weights::Union{Nothing,Symbol} = nothing`: unit weights.

# Returns
- `ContinuousDiDEstimate`: `coef = [ATT^o, ACRT^o]`; the level curve `r.att` and the
  slope curve `r.acrt` on `r.dose` with standard errors;
  `confint(r; curve=:att, uniform=true)` gives simultaneous bands.

# Examples
```julia
using DrSnow, DataFrames, StableRNGs
rng = StableRNG(1)
n = 1_000
dose = [rand(rng) < 0.3 ? 0.0 : rand(rng) for _ in 1:n]   # 30% untreated
α = randn(rng, n)
y1 = α .+ randn(rng, n)
y2 = α .+ 0.5 .+ 2 .* dose .- dose .^ 2 .+ randn(rng, n)   # ATE(d) = 2d - d²
df = DataFrame(id=repeat(1:n, 2), year=repeat([1, 2]; inner=n), y=vcat(y1, y2),
               dose=vcat(dose, dose))
r = did_continuous(df, :y, :dose, :id, :year; degree=2)   # quadratic sieve
coeftable(r)
confint(r; curve=:acrt, uniform=true, rng=StableRNG(2))
```

# References
- Callaway, B., Goodman-Bacon, A., & Sant'Anna, P. H. C. (forthcoming).
  Difference-in-differences with a continuous treatment. *American Economic
  Review*. Earlier version: NBER Working Paper 32117 (2024).
- Chen, X. (2007). Large sample sieve estimation of semi-nonparametric models. In
  *Handbook of Econometrics* (pp. 5549–5632). Elsevier.
- de Chaisemartin, C., & D'Haultfœuille, X. (2023). Two-way fixed effects and
  differences-in-differences with heterogeneous treatment effects: A survey. *The
  Econometrics Journal*, 26(3), C1–C30.
- Callaway, B., Goodman-Bacon, A., & Sant'Anna, P. H. C. (2026). contdid:
  Difference-in-differences with a continuous treatment. R package version 0.1.1.
"""
function did_continuous(data, outcome::Symbol, dose::Symbol, unit::Symbol,
                        time::Symbol; degree::Integer=3, num_knots::Integer=0,
                        knots=nothing, dose_grid=nothing,
                        cluster::Union{Nothing,Symbol}=nothing,
                        weights::Union{Nothing,Symbol}=nothing)
    degree >= 1 || throw(ArgumentError("degree must be ≥ 1"))
    num_knots >= 0 || throw(ArgumentError("num_knots must be ≥ 0"))
    df = _did_prepare(data, [outcome, dose, unit, time, cluster, weights];
                      context="did_continuous")
    periods = sort(unique(df[!, time]))
    length(periods) == 2 || throw(ArgumentError(
        "did_continuous needs exactly two periods (found $(length(periods))); " *
        "subset the data to a pre- and a post-treatment period"))
    units = unique(df[!, unit])
    uidx = Dict(u => i for (i, u) in enumerate(units))
    n = length(units)
    y = fill(NaN, n, 2)
    D = fill(NaN, n)
    wv = ones(n)
    clv = cluster === nothing ? nothing : Vector{Any}(undef, n)
    seen = falses(n, 2)
    for r in eachrow(df)
        i = uidx[r[unit]]
        p = r[time] == periods[1] ? 1 : 2
        seen[i, p] && throw(ArgumentError(
            "did_continuous: unit $(r[unit]) observed twice in period $(r[time])"))
        seen[i, p] = true
        y[i, p] = r[outcome]
        d = float(r[dose])
        d >= 0 || throw(ArgumentError("doses must be ≥ 0"))
        if p == 2 || isnan(D[i])
            D[i] = d
        end
        weights === nothing || (wv[i] = float(r[weights]))
        cluster === nothing || (clv[i] = r[cluster])
    end
    bal = vec(all(seen; dims=2))
    if !all(bal)
        @warn "did_continuous: dropping $(count(!, bal)) unit(s) not observed in both " *
              "periods"
        y, D, wv = y[bal, :], D[bal], wv[bal]
        clv === nothing || (clv = clv[bal])
        n = count(bal)
    end
    any(<(0), wv) && throw(ArgumentError("weights must be ≥ 0"))
    dy = y[:, 2] .- y[:, 1]
    tr = D .> 0
    n1, n0 = count(tr), count(!, tr)
    n0 > 0 || throw(ArgumentError(
        "did_continuous: no untreated (D = 0) units; ATT(d|d) is not identified " *
        "without a comparison group"))
    Dt = D[tr]
    lo, hi = extrema(Dt)
    lo < hi || throw(ArgumentError("did_continuous: all treated units have the " *
                                   "same dose; use a binary DiD estimator"))
    interior = if knots !== nothing
        k = sort(Float64.(collect(knots)))
        all(x -> lo < x < hi, k) || throw(ArgumentError(
            "knots must lie strictly inside the range of treated doses"))
        k
    else
        num_knots == 0 ? Float64[] :
        [quantile(Dt, q) for q in range(0, 1; length=num_knots + 2)[2:(end - 1)]]
    end
    t = _did_bspline_knots(interior, lo, hi, Int(degree))
    K = length(t) - degree - 1
    p = Int(degree)
    Ψ = reduce(vcat, (_did_bspline_basis(d, t, p)' for d in Dt))
    dΨ = reduce(vcat, (_did_bspline_deriv(d, t, p)' for d in Dt))
    length(unique(Dt)) > K || throw(ArgumentError(
        "did_continuous: too few distinct treated doses for $K spline coefficients " *
        "(reduce degree or num_knots)"))
    w1 = wv[tr]
    w0 = wv[.!tr]
    Q = Ψ' * (Ψ .* w1) ./ sum(w1)
    rank(Q) == K || throw(ArgumentError(
        "did_continuous: the spline design is singular (knots without doses between " *
        "them?); reduce num_knots"))
    β = Q \ (Ψ' * (w1 .* dy[tr]) ./ sum(w1))
    ȳ0 = sum(w0 .* dy[.!tr]) / sum(w0)
    resid = dy[tr] .- Ψ * β
    Qi = inv(Q)
    # influence functions (scaled so that estimate - truth ≈ mean over all n units)
    sw1 = sum(w1) / n
    sw0 = sum(w0) / n
    IFβ = zeros(n, K)
    IFβ[tr, :] = ((Ψ .* (w1 .* resid)) * Qi') ./ sw1
    IF0 = zeros(n)
    IF0[.!tr] = w0 .* (dy[.!tr] .- ȳ0) ./ sw0
    fitted = Ψ * β
    att_o = sum(w1 .* fitted) / sum(w1) - ȳ0
    slope = dΨ * β
    acrt_o = sum(w1 .* slope) / sum(w1)
    ψbar = vec(sum(Ψ .* w1; dims=1)) ./ sum(w1)
    dψbar = vec(sum(dΨ .* w1; dims=1)) ./ sum(w1)
    IFatt = IFβ * ψbar .- IF0
    IFatt[tr] .+= w1 .* (fitted .- (att_o + ȳ0)) ./ sw1
    IFacrt = IFβ * dψbar
    IFacrt[tr] .+= w1 .* (slope .- acrt_o) ./ sw1
    grid = dose_grid === nothing ?
           collect(range(quantile(Dt, 0.01), quantile(Dt, 0.99); length=50)) :
           Float64.(collect(dose_grid))
    all(g -> lo <= g <= hi, grid) || throw(ArgumentError(
        "dose_grid must lie within the range of treated doses [$lo, $hi]"))
    Bg = reduce(vcat, (_did_bspline_basis(g, t, p)' for g in grid))
    dBg = reduce(vcat, (_did_bspline_deriv(g, t, p)' for g in grid))
    IFcurve = IFβ * Bg' .- IF0
    IFslope = IFβ * dBg'
    cl = clv
    V, G = _if_vcov(hcat(IFatt, IFacrt), cl)
    Vatt, _ = _if_vcov(IFcurve, cl)
    Vacrt, _ = _if_vcov(IFslope, cl)
    return ContinuousDiDEstimate([att_o, acrt_o], V, grid, Bg * β .- ȳ0,
                                 sqrt.(max.(diag(Vatt), 0.0)), dBg * β,
                                 sqrt.(max.(diag(Vacrt), 0.0)), Vatt, Vacrt, 2n, n1,
                                 n0, cluster === nothing ? 0 : G,
                                 (degree=p, knots=interior, boundary=(lo, hi),
                                  spline_coef=β, control_mean_change=ȳ0),
                                 hcat(IFatt, IFacrt))
end
