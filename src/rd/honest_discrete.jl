# Inference with a discrete running variable (Kolesár & Rothe 2018): confidence
# intervals under bounded misspecification error (RDHonest `RDHonestBME`) and lower
# bounds on the smoothness constant M (RDHonest `RDSmoothnessBound`).

"""
    RDBMEEstimate <: CausalEstimate

Result of [`rd_honest_bme`](@ref): a local polynomial RD estimate with a uniform kernel
and a confidence interval that is honest under the bounded misspecification error (BME)
assumption of Kolesár and Rothe (2018).

`coef` and `stderror` are the local polynomial estimate and its
Eicker–Huber–White standard error. `confint(r; level)` is the BME interval. It is
**not** of the form estimate ± z·se: it is the union, over the specification errors at
the support points nearest to the cutoff, of shifted intervals (see
[`rd_honest_bme`](@ref)). `pvalues` returns the p-value obtained by inverting that
interval.

# Fields
- `estimate`, `se`: point estimate and standard error.
- `max_bias`: largest absolute specification error used by the interval at `level`.
- `conf_low`, `conf_high`: two-sided BME interval; `conf_low_onesided`,
  `conf_high_onesided`: one-sided bounds.
- `pvalue`, `level`: p-value from inverting the interval, and the confidence level.
- `bandwidth`, `order`, `cutoff`: window half-width, polynomial order and cutoff.
- `n`: observations within the bandwidth; `support_left`, `support_right`: numbers of
  support points on each side within the bandwidth.
- `leverage`: maximal leverage of the estimation weights.
- `dev::Vector{Float64}`, `se_dev::Vector{Float64}`: for every combination of support
  points and signs, the shift of the estimate and its standard error (used by `confint`
  at other levels).
"""
struct RDBMEEstimate <: CausalEstimate
    estimate::Float64
    se::Float64
    max_bias::Float64
    conf_low::Float64
    conf_high::Float64
    conf_low_onesided::Float64
    conf_high_onesided::Float64
    pvalue::Float64
    level::Float64
    bandwidth::Float64
    order::Int
    cutoff::Float64
    n::Int
    support_left::Int
    support_right::Int
    leverage::Float64
    dev::Vector{Float64}
    se_dev::Vector{Float64}
end

StatsAPI.coef(r::RDBMEEstimate) = [r.estimate]
StatsAPI.vcov(r::RDBMEEstimate) = fill(r.se^2, 1, 1)
StatsAPI.coefnames(r::RDBMEEstimate) = ["RD effect (local polynomial, uniform kernel)"]
StatsAPI.nobs(r::RDBMEEstimate) = r.n
pvalues(r::RDBMEEstimate) = [r.pvalue]
estimand(::RDBMEEstimate) = "ATE at the cutoff (sharp RD, discrete running variable)"
method_name(r::RDBMEEstimate) = "RD with bounded misspecification error (order $(r.order))"

function _rd_bme_ci(r::RDBMEEstimate, level::Real)
    (0 < level < 1) || throw(ArgumentError("level must be in (0, 1), got $level"))
    z = quantile(Normal(), 1 - (1 - level) / 2)
    lo = r.estimate .+ r.dev .- z .* r.se_dev
    hi = r.estimate .+ r.dev .+ z .* r.se_dev
    l = argmin(lo)
    u = argmax(hi)
    return lo[l], hi[u], max(abs(r.dev[u]), abs(r.dev[l]))
end

function StatsAPI.confint(r::RDBMEEstimate; level::Real=r.level)
    lo, hi, _ = _rd_bme_ci(r, level)
    return [lo hi]
end

function show_details(io::IO, r::RDBMEEstimate)
    println(io)
    lv = round(Int, 100 * r.level)
    @printf(io, "BME %d%% CI: [%.4f, %.4f] (largest specification error %.4g)\n", lv,
            r.conf_low, r.conf_high, r.max_bias)
    @printf(io, "p-value (inverting the BME interval): %.4g\n", r.pvalue)
    @printf(io, "One-sided %d%% bounds: (-Inf, %.4f], [%.4f, Inf)\n", lv,
            r.conf_high_onesided, r.conf_low_onesided)
    @printf(io, "Bandwidth %g; support points within it: %d left / %d right\n",
            r.bandwidth, r.support_left, r.support_right)
end

"""
    rd_honest_bme(data, outcome, running; cutoff=0.0, h=Inf, order=0,
                  level=0.95) -> RDBMEEstimate

Confidence interval for a sharp RD design with a **discrete** running variable under the
bounded misspecification error (BME) assumption of Kolesár and Rothe (2018), equivalent
to `RDHonestBME` of the R package `RDHonest`.

When the running variable takes few distinct values near the cutoff (age in years, test
scores on a coarse grid), the continuity-based argument for local polynomial RD
estimators fails. The bandwidth cannot shrink around the cutoff, and a polynomial fitted
to a handful of support points is misspecified to an unknown degree. Lee and Card
(2008) proposed clustering standard errors by value of the running variable to account
for this specification error. Kolesár and Rothe (2018) show that those intervals can
have coverage far below nominal, and even below that of unclustered intervals. They
propose instead two honest alternatives: a bound on curvature, implemented in
[`rd_honest`](@ref), and the BME approach implemented here.

A polynomial of order `order` is fitted with a uniform kernel to the observations within
`h` of the cutoff (by default, all of them). Let ``\\delta(x)`` be the specification
error, the difference between the conditional mean and the polynomial fit, at support
point ``x``. The BME assumption states that, on each side, the specification error at
the cutoff is no larger in absolute value than the largest ``|\\delta(x_g)|`` among the
support points in the window. The bias of the estimate is then bounded by the
specification errors at the observed support points, which can be estimated from cell
means. The interval is the union of the intervals
```math
\\hat\\tau + s_- \\hat\\delta(x_{g_-}) + s_+ \\hat\\delta(x_{g_+})
    \\pm z_{1-\\alpha/2}\\,\\text{se}_{g_-, g_+}
```
over support points ``g_-`` below and ``g_+`` above the cutoff and signs ``s_\\pm``. The
standard errors account for the estimation of ``\\hat\\delta`` through the joint EHW
covariance of cell means and polynomial coefficients.

BME needs no smoothness constant, unlike [`rd_honest`](@ref). In exchange, it rests on
an assumption about the cutoff relative to the observed support points that is itself
untestable. It can be conservative, and it needs more support points on each side than
the polynomial order. Prefer [`rd_honest`](@ref) when a credible `M` can be justified;
use this function as a complement or when no such bound is available. Report the window,
the polynomial order and the number of support points on each side.

# Arguments
- `data::AbstractDataFrame`: one row per unit. Missing values are dropped.
- `outcome::Symbol`: outcome column.
- `running::Symbol`: discrete running variable. Units with `running ≥ cutoff` are
  treated.

# Keywords
- `cutoff::Real=0.0`: the RD cutoff.
- `h::Real=Inf`: half-width of the estimation window (uniform kernel).
- `order::Integer=0`: polynomial order (`0` is a difference in means, `1` a local linear
  fit, and so on).
- `level::Real=0.95`: confidence level.

# Returns
- [`RDBMEEstimate`](@ref), whose `confint` is the BME interval.

# Examples
```julia
using DrSnow, CSV, DataFrames
cghs = CSV.read(pkgdir(DrSnow, "test", "validation", "rd", "cghs_sample.csv"), DataFrame)
r = rd_honest_bme(cghs, :log_earn, :yearat14; cutoff=1947, h=3, order=1)
confint(r), r.support_left, r.support_right
```

# References
- Kolesár, M., & Rothe, C. (2018). Inference in regression discontinuity designs with a
  discrete running variable. *American Economic Review*, 108(8), 2277–2304.
- Lee, D. S., & Card, D. (2008). Regression discontinuity inference with specification
  error. *Journal of Econometrics*, 142(2), 655–674.
- Armstrong, T. B., & Kolesár, M. (2020). Simple and honest confidence intervals in
  nonparametric regression. *Quantitative Economics*, 11(1), 1–39.
"""
function rd_honest_bme(data::AbstractDataFrame, outcome::Symbol, running::Symbol;
                       cutoff::Real=0.0, h::Real=Inf, order::Integer=0,
                       level::Real=0.95)
    ctx = "rd_honest_bme"
    (0 < level < 1) || throw(ArgumentError("$ctx: level must be in (0, 1)"))
    order >= 0 || throw(ArgumentError("$ctx: order must be ≥ 0"))
    h > 0 || throw(ArgumentError("$ctx: h must be positive"))
    E = _rd_extract(data, outcome, running; context=ctx)
    xall = (E.x .- Float64(cutoff)) .+ 0.0   # map -0.0 to 0.0
    ind = (xall .<= h) .& (xall .>= -h)
    x = xall[ind]
    y = E.y[ind]
    n = length(x)
    support = sort(unique(x))
    G = length(support)
    Gm = count(<(0), support)
    (Gm >= 1 && G - Gm >= 1) || throw(ArgumentError(
        "$ctx: need support points on both sides of the cutoff within the bandwidth"))
    design(v) = begin
        P = _rd_vander(v, order)
        hcat(P, (v .>= 0) .* P)
    end
    X1raw = design(x)
    # column order of R's `y ~ (I(x^1) + ... + I(x^p)) * I(x >= 0)`
    k = order + 1
    perm = vcat(1:k, k + 1:2k)
    X1 = X1raw[:, perm]
    all(_rd_h_indep_cols(X1)) || throw(ArgumentError(
        "$ctx: the polynomial of order $order is not identified (need at least " *
        "$(order + 1) support points on each side within the bandwidth)"))
    n > 2k || throw(ArgumentError("$ctx: too few observations within the bandwidth"))
    F1 = qr(X1)
    R1 = UpperTriangular(F1.R)
    b1 = R1 \ (Matrix(F1.Q)' * y)
    e1 = y .- X1 * b1
    Q1inv = inv(Symmetric(Matrix(R1' * R1)))
    gidx = Dict(v => i for (i, v) in enumerate(support))
    g = [gidx[v] for v in x]
    cnt = zeros(G)
    sy = zeros(G)
    for i in 1:n
        cnt[g[i]] += 1
        sy[g[i]] += y[i]
    end
    b2 = sy ./ cnt
    e2 = y .- b2[g]
    delta = b2 .- design(support)[:, perm] * b1
    S = hcat((X1 .* e1) * Q1inv, zeros(n, G))
    for i in 1:n
        S[i, 2k + g[i]] = e2[i] / cnt[g[i]]
    end
    Vm = n .* cov(S)                   # R: length(y) * var(cbind(...))
    aa = zeros(G + 1, 2k + G)
    aa[1:G, 1:2k] = -design(support)[:, perm]
    for j in 1:G
        aa[j, 2k + j] = 1
    end
    aa[G + 1, order + 2] = 1
    vdt = aa * Vm * aa'
    tau = b1[order + 2]
    # all combinations (g_-, g_+, s_-, s_+), in the order of R's expand.grid
    dev = Float64[]
    sedev = Float64[]
    sel = zeros(G + 1)
    for sp in (-1, 1), sm in (-1, 1), gp in (Gm + 1):G, gm in 1:Gm
        fill!(sel, 0.0)
        sel[gm] = sm
        sel[gp] = sp
        sel[G + 1] = 1
        push!(sedev, sqrt(max(dot(sel, vdt * sel), 0.0)))
        push!(dev, sm * delta[gm] + sp * delta[gp])
    end
    se = sqrt(vdt[G + 1, G + 1])
    (isfinite(se) && se > 0) || throw(ArgumentError(
        "$ctx: the standard error is not positive; too few observations per support " *
        "point"))
    zc = quantile(Normal(), 1 - (1 - level) / 2)
    za = quantile(Normal(), level)
    lo = tau .+ dev .- zc .* sedev
    hi = tau .+ dev .+ zc .* sedev
    l = argmin(lo)
    u = argmax(hi)
    wt = (R1 \ Matrix(Matrix(F1.Q)'))[order + 2, :]
    # p-value by inverting the interval: 0 is excluded at level 1 - a iff
    # z_{1-a/2} < max(min_g (τ + dev)/se, min_g -(τ + dev)/se)
    zstar = max(minimum((tau .+ dev) ./ sedev), minimum(-(tau .+ dev) ./ sedev))
    pv = zstar > 0 ? min(1.0, 2 * ccdf(Normal(), zstar)) : 1.0
    return RDBMEEstimate(tau, se, max(abs(dev[u]), abs(dev[l])), lo[l], hi[u],
                         minimum(tau .+ dev .- za .* sedev),
                         maximum(tau .+ dev .+ za .* sedev), pv, Float64(level),
                         Float64(h), Int(order), Float64(cutoff), n, Gm, G - Gm,
                         maximum(wt .^ 2) / sum(wt .^ 2), dev, sedev)
end

"""
    rd_smoothness_bound(data, outcome, running; cutoff=0.0, s=1, separate=false,
                        multiple=true, level=0.95, sclass=:holder,
                        rng=Random.default_rng(), ndraws=10_000) -> DataFrame

Estimate and lower confidence bound for a **lower bound** on the smoothness constant `M`
of a sharp RD design with a discrete running variable (Kolesár & Rothe 2018, online
appendix; `RDSmoothnessBound` of the R package `RDHonest`).

Honest inference with [`rd_honest`](@ref) requires a bound `M` on the second derivative
of the regression function, chosen a priori. The data cannot tell how large `M` must
be, because the curvature between support points and at the cutoff is never observed.
They can, however, show that `M` must be *at least* some value, since observed changes in
slope imply a minimum curvature. On each side of the cutoff the distinct support
points are grouped into consecutive blocks of `s` points. Each triple of adjacent blocks
gives a curvature estimate ``\\hat\\Delta``, a scaled second difference of the block
means, whose expectation bounds `M` from below in the chosen smoothness class. With
`multiple = true` all triples are used, and the critical values account for the maximum
over them. They are simulated with `ndraws` normal draws from `rng`, so results match
`RDSmoothnessBound` only up to simulation noise. With `multiple = false` only the triple
closest to the cutoff is used and the computation is deterministic.

The output is a sanity check, not a way to choose `M`. An `M` passed to
[`rd_honest`](@ref) that lies below the lower confidence bound is inconsistent with the
data at the chosen level. An `M` above it is not thereby justified. A small estimate
does not show that the regression function is smooth, especially near the cutoff, where
the bound is least informative.

# Arguments
- `data::AbstractDataFrame`: one row per unit. Missing values are dropped.
- `outcome::Symbol`: outcome column.
- `running::Symbol`: discrete running variable.

# Keywords
- `cutoff::Real=0.0`: the RD cutoff.
- `s::Integer=1`: number of support points averaged in each block. Larger blocks reduce
  noise but smooth out local curvature.
- `separate::Bool=false`: report bounds separately below and above the cutoff.
- `multiple::Bool=true`: use all triples of blocks (otherwise only the one closest to
  the cutoff).
- `level::Real=0.95`: level of the one-sided lower confidence bound.
- `sclass=:holder`: `:holder` or `:taylor` smoothness class, as in [`rd_honest`](@ref).
- `rng::AbstractRNG=Random.default_rng()`: generator for the simulated critical values
  (used when `multiple = true`).
- `ndraws::Integer=10_000`: number of simulation draws (at least 100).

# Returns
- `DataFrame` with columns `side` (`"pooled"`, or `"below"` and `"above"`), `estimate`
  (median-unbiased estimate of the lower bound on `M`) and `conf_low` (lower confidence
  bound).

# Examples
```julia
using DrSnow, CSV, DataFrames, StableRNGs
cghs = CSV.read(pkgdir(DrSnow, "test", "validation", "rd", "cghs_sample.csv"), DataFrame)
rd_smoothness_bound(cghs, :log_earn, :yearat14; cutoff=1947, s=2, rng=StableRNG(1))
rd_smoothness_bound(cghs, :log_earn, :yearat14; cutoff=1947, multiple=false,
                    separate=true)
```

# References
- Kolesár, M., & Rothe, C. (2018). Inference in regression discontinuity designs with a
  discrete running variable. *American Economic Review*, 108(8), 2277–2304.
- Armstrong, T. B., & Kolesár, M. (2018). Optimal inference in a class of regression
  models. *Econometrica*, 86(2), 655–683.
- Armstrong, T. B., & Kolesár, M. (2020). Simple and honest confidence intervals in
  nonparametric regression. *Quantitative Economics*, 11(1), 1–39.
"""
function rd_smoothness_bound(data::AbstractDataFrame, outcome::Symbol, running::Symbol;
                             cutoff::Real=0.0, s::Integer=1, separate::Bool=false,
                             multiple::Bool=true, level::Real=0.95, sclass=:holder,
                             rng::AbstractRNG=Random.default_rng(),
                             ndraws::Integer=10_000)
    ctx = "rd_smoothness_bound"
    (0 < level < 1) || throw(ArgumentError("$ctx: level must be in (0, 1)"))
    s >= 1 || throw(ArgumentError("$ctx: s must be a positive integer"))
    ndraws >= 100 || throw(ArgumentError("$ctx: ndraws must be at least 100"))
    scl = _rd_h_sclass(sclass)
    alpha = 1 - Float64(level)
    d, _ = _rd_h_build(data, outcome, running; cutoff, treatment=nothing,
                       covariates=Symbol[], cluster=nothing, weights=nothing,
                       class=:srd, context=ctx)
    (any(d.p) && any(d.m)) || throw(ArgumentError(
        "$ctx: no observations on one side of the cutoff"))
    d = _rd_h_prelim_var(d, :ehw)
    s2 = d.sigma2[:, 1]
    Y = d.Y[:, 1]
    function Dk(Ys, Xs, xu, ss, j)
        blk(a, b) = (Xs .>= xu[a]) .& (Xs .<= xu[b])
        I1 = blk(3j * s - 3s + 1, 3j * s - 2s)
        I2 = blk(3j * s - 2s + 1, 3j * s - s)
        I3 = blk(3j * s - s + 1, 3j * s)
        m1, m2, m3 = mean(Xs[I1]), mean(Xs[I2]), mean(Xs[I3])
        lam = (m3 - m2) / (m3 - m1)
        q1, q2, q3 = mean(Xs[I1] .^ 2), mean(Xs[I2] .^ 2), mean(Xs[I3] .^ 2)
        den = scl === :taylor ? (1 - lam) * q3 + lam * q1 + q2 :
              (1 - lam) * q3 + lam * q1 - q2
        Del = 2 * (lam * mean(Ys[I1]) + (1 - lam) * mean(Ys[I3]) - mean(Ys[I2])) / den
        VD = 4 * (lam^2 * mean(ss[I1]) / count(I1) +
                  (1 - lam)^2 * mean(ss[I3]) / count(I3) +
                  mean(ss[I2]) / count(I2)) / den^2
        return (Del, sqrt(VD))
    end
    xp = unique(d.X[d.p])
    xm = sort(unique(abs.(d.X[d.m])))
    Sp = fld(length(xp), 3s)
    Sm = fld(length(xm), 3s)
    min(Sp, Sm) == 0 && throw(ArgumentError(
        "$ctx: s = $s is too large: fewer than 3s support points on a side"))
    if !multiple
        Sp = Sm = 1
    end
    Dp = [Dk(Y[d.p], d.X[d.p], xp, s2[d.p], j) for j in 1:Sp]
    Dm = [Dk(Y[d.m], abs.(d.X[d.m]), xm, s2[d.m], j) for j in 1:Sm]
    function cvfun(Mv, Z, sds, a)
        if length(sds) == 1
            return _rd_cvb(Mv / sds[1], a)
        end
        maxS = [abs(maximum(Z[i, j] + Mv / sds[j] for j in eachindex(sds)))
                for i in axes(Z, 1)]
        return _rd_quantile_type7(maxS, [1 - a])[1]
    end
    function hatM(D)
        dels = first.(D)
        sds = last.(D)
        ts = abs.(dels ./ sds)
        mt = maximum(ts)
        Z = length(D) == 1 ? zeros(0, 1) : randn(rng, ndraws, length(D))
        est = 0.0
        low = 0.0
        if mt > cvfun(0.0, Z, sds, 0.5)
            est = _rd_h_find_zero_pos(Mv -> mt - cvfun(Mv, Z, sds, 0.5))
        end
        if mt >= cvfun(0.0, Z, sds, alpha)
            low = _rd_h_find_zero_pos(Mv -> mt - cvfun(Mv, Z, sds, alpha))
        end
        return est, low
    end
    if separate
        ne = hatM(Dm)
        po = hatM(Dp)
        return DataFrame(side=["below", "above"], estimate=[ne[1], po[1]],
                         conf_low=[ne[2], po[2]])
    end
    pooled = hatM(vcat(Dm, Dp))
    return DataFrame(side=["pooled"], estimate=[pooled[1]], conf_low=[pooled[2]])
end
