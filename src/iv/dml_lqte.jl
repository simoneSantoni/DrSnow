# Distributional effects for compliers with cross-fitted machine-learned nuisances:
# local quantile treatment effects (LQTE) and complier outcome distributions.
#
# With a binary instrument Z, binary treatment D, covariates X, and the LATE
# assumptions (conditional independence of Z, exclusion, monotonicity, first stage,
# overlap), the distribution of the potential outcome Y(d) among compliers is
# identified (Abadie 2002, Frölich & Melly 2013):
#     F_d(y) = s_d E[ψ_d(y)] / E[φ_D],   s_1 = 1, s_0 = -1,
#     ψ_d(y) = g₁(X) − g₀(X) + Z (1{D=d, Y≤y} − g₁)/m − (1−Z)(1{D=d, Y≤y} − g₀)/(1−m),
#     φ_D    = r₁ − r₀ + Z (D − r₁)/m − (1 − Z)(D − r₀)/(1 − m)          (complier share),
# with g_z(X) = P(D = d, Y ≤ y | Z = z, X), r_z(X) = P(D = 1 | Z = z, X) and
# m(X) = P(Z = 1 | X). The local potential quantile q_d(τ) solves F_d(q) = τ and the
# LQTE is q₁(τ) − q₀(τ). The estimating equations follow DoubleML's DoubleMLLPQ
# (Belloni, Chernozhukov, Fernández-Val & Hansen 2017), including the nested
# preliminary IPW quantile at which g is estimated and the normalized IPW weights.
#
# Variance. DoubleML treats the complier share in the denominator as known; the
# default here (`variance = :influence`) also accounts for its estimation, which adds
# −τ (φ_D − E φ_D)/E φ_D to the influence function. `variance = :doubleml` reproduces
# DoubleML.

"""
    LQTEEstimate <: CausalEstimate

Local quantile treatment effects for compliers, estimated by [`dml_lqte`](@ref).

The object holds, at each requested quantile level ``\\tau``, the local quantile
treatment effect ``q_1(\\tau) - q_0(\\tau)``, where ``q_d(\\tau)`` is the
``\\tau``-quantile of the potential outcome ``Y(d)`` among compliers, together with the
two local potential quantiles themselves, the estimated complier share and the influence
functions from which pointwise and uniform inference is computed. The LQTE is a
difference of quantiles of two marginal distributions, not the quantile of individual
treatment effects, which is not identified without further assumptions.

`coef(r)` are the LQTEs at `r.quantiles`, `vcov(r)` their joint covariance from the
influence functions, `stderror(r)` their standard errors, and `confint(r; level,
uniform=false)` gives pointwise intervals or, with `uniform = true`, simultaneous
sup-t bands over the quantiles from a multiplier bootstrap.

# Fields
- `quantiles::Vector{Float64}`: the quantile levels ``\\tau``.
- `coef::Vector{Float64}`, `vcov::Matrix{Float64}`: the LQTEs and their covariance
  (median rule over cross-fitting repetitions).
- `lpq::Matrix{Float64}`, `lpq_se::Matrix{Float64}`: the local potential quantiles
  ``q_d(\\tau)`` of ``Y(0)`` (column 1) and ``Y(1)`` (column 2) for compliers, and their
  standard errors (medians over repetitions).
- `complier_share::Float64`: the estimated share of compliers (median over
  repetitions).
- `influence::Array{Float64,3}`: the ``n \\times Q \\times R`` influence functions of the
  LQTEs (``Q`` quantiles, ``R`` repetitions).
- `supt::Vector{Float64}`: bootstrap draws of the sup-t statistic, pooled over
  repetitions.
- `all_coef::Matrix{Float64}`: the ``Q \\times R`` estimates of each repetition.
- `converged::BitMatrix`: ``2Q \\times R`` indicators of whether the estimating equation
  changed sign; where it did not, the minimizer of the absolute score over the data
  points is reported.
- `folds::Matrix{Int}`: the fold assignments of each repetition; `n::Int`: number of
  observations.
- `variance::Symbol`: `:influence` or `:doubleml`; `learners`: names of the nuisance
  learners; `trim::Float64`, `normalize_ipw::Bool`: the propensity-score settings.

# References
- Frölich, M., & Melly, B. (2013). Unconditional quantile treatment effects under
  endogeneity. *Journal of Business & Economic Statistics*, 31(3), 346–357.
- Belloni, A., Chernozhukov, V., Fernández-Val, I., & Hansen, C. (2017). Program
  evaluation and causal inference with high-dimensional data. *Econometrica*, 85(1),
  233–298.
"""
struct LQTEEstimate <: CausalEstimate
    quantiles::Vector{Float64}
    coef::Vector{Float64}
    vcov::Matrix{Float64}
    lpq::Matrix{Float64}
    lpq_se::Matrix{Float64}
    complier_share::Float64
    influence::Array{Float64,3}
    supt::Vector{Float64}
    all_coef::Matrix{Float64}
    converged::BitMatrix
    folds::Matrix{Int}
    n::Int
    variance::Symbol
    learners::Vector{Pair{Symbol,String}}
    trim::Float64
    normalize_ipw::Bool
end

StatsAPI.coef(r::LQTEEstimate) = r.coef
StatsAPI.vcov(r::LQTEEstimate) = r.vcov
StatsAPI.coefnames(r::LQTEEstimate) = ["LQTE($(q))" for q in r.quantiles]
StatsAPI.nobs(r::LQTEEstimate) = r.n
estimand(::LQTEEstimate) = "local (complier) quantile treatment effects"
method_name(::LQTEEstimate) = "DML local quantile treatment effects"

function StatsAPI.confint(r::LQTEEstimate; level::Real=0.95, uniform::Bool=false)
    0 < level < 1 || throw(ArgumentError("level must be in (0, 1), got $level"))
    c = uniform ? quantile(r.supt, level) : critical_value(level)
    b, s = r.coef, stderror(r)
    return hcat(b .- c .* s, b .+ c .* s)
end

function show_details(io::IO, r::LQTEEstimate)
    println(io)
    println(io, "Local potential quantiles for compliers:")
    for (j, q) in enumerate(r.quantiles)
        @printf(io, "  τ = %.3g: q₀ = %.4g (se %.3g), q₁ = %.4g (se %.3g)\n", q,
                r.lpq[j, 1], r.lpq_se[j, 1], r.lpq[j, 2], r.lpq_se[j, 2])
    end
    @printf(io, "Estimated complier share: %.4g\n", r.complier_share)
    K, R = maximum(r.folds), size(r.folds, 2)
    println(io, "Cross-fitting: $K folds × $R repetition", R == 1 ? "" : "s",
            "; variance: ", r.variance === :influence ?
            "influence function incl. complier-share estimation" : "DoubleML")
    for (nm, l) in r.learners
        println(io, "  ", nm, ": ", l)
    end
    all(r.converged) ||
        println(io, "Warning: the score did not change sign for some quantile; the ",
                "minimizer of |score| over the data points is reported.")
    return nothing
end

"""
    ComplierDistribution

Distribution functions of the potential outcomes of compliers, estimated by
[`dml_complier_cdf`](@ref).

The object tabulates, on a grid of outcome values ``y``, the estimated cumulative
distribution functions ``F_0(y) = P(Y(0) \\le y \\mid \\text{complier})`` and
``F_1(y) = P(Y(1) \\le y \\mid \\text{complier})``, their pointwise standard errors and
uniform (sup-t) confidence bands. Comparing the two functions shows how the
instrument-induced treatment shifts the whole outcome distribution of compliers
(Imbens and Rubin 1997; Abadie 2002), beyond the mean shift measured by the LATE.

# Fields
- `table::DataFrame`: one row per grid point `y`, with `cdf0`, `se0`, `lower0` and
  `upper0` (uniform band) for ``Y(0)`` and the same columns for ``Y(1)`` (`cdf1`, `se1`,
  `lower1`, `upper1`).
- `complier_share::Float64`: the estimated share of compliers.
- `level::Float64`: the confidence level of the bands.
- `critical_values::Tuple{Float64,Float64}`: the sup-t critical values for ``Y(0)`` and
  ``Y(1)``.
- `rearranged::Bool`: whether the estimates and bands were monotonized.
- `n::Int`: number of observations.

# References
- Imbens, G. W., & Rubin, D. B. (1997). Estimating outcome distributions for compliers
  in instrumental variables models. *Review of Economic Studies*, 64(4), 555–574.
- Abadie, A. (2002). Bootstrap tests for distributional treatment effects in
  instrumental variable models. *Journal of the American Statistical Association*,
  97(457), 284–292.
"""
struct ComplierDistribution
    table::DataFrame
    complier_share::Float64
    level::Float64
    critical_values::Tuple{Float64,Float64}
    rearranged::Bool
    n::Int
end

StatsAPI.nobs(r::ComplierDistribution) = r.n

function Base.show(io::IO, ::MIME"text/plain", r::ComplierDistribution)
    println(io, "Complier potential-outcome distributions (DML), ", nrow(r.table),
            " grid points; complier share ", @sprintf("%.4g", r.complier_share))
    @printf(io, "%g%% uniform bands (sup-t critical values %.3g, %.3g)%s\n",
            100 * r.level, r.critical_values..., r.rearranged ? ", rearranged" : "")
    for row in eachrow(r.table)
        @printf(io, "  y = %9.4g   F₀ = %.3f [%.3f, %.3f]   F₁ = %.3f [%.3f, %.3f]\n",
                row.y, row.cdf0, row.lower0, row.upper0, row.cdf1, row.lower1, row.upper1)
    end
end

Base.show(io::IO, r::ComplierDistribution) =
    print(io, "ComplierDistribution(", nrow(r.table), " grid points)")

# ---------------------------------------------------------------------------
# Shared pieces
# ---------------------------------------------------------------------------

"""Normalized IPW propensity (DoubleML `_normalize_ipw`)."""
function _iv_dq_normalize(m::AbstractVector, z::AbstractVector)
    a = mean(z ./ m)
    b = mean((1 .- z) ./ (1 .- m))
    return z .* m .* a .+ (1 .- z) .* (1 .- (1 .- m) .* b)
end

_iv_dq_phiD(m, r0, r1, z, d) = r1 .- r0 .+ z .* (d .- r1) ./ m .-
                               (1 .- z) .* (d .- r0) ./ (1 .- m)

function _iv_dq_proba(learner, X, t, Xnew, seed, what, ctx; constant_ok::Bool=false)
    isempty(t) && throw(ArgumentError("$ctx: no training observations for $what"))
    constant_ok && all(==(t[1]), t) && return fill(float(t[1]), size(Xnew, 1))
    all(==(t[1]), t) &&
        throw(ArgumentError("$ctx: the training sample for $what contains a single " *
                            "class; use fewer folds, a coarser quantile grid, or check " *
                            "for one-sided non-compliance"))
    p = fitpredict_proba(learner, X, t, Xnew; rng=Random.Xoshiro(seed))
    all(isfinite, p) || throw(ArgumentError("$ctx: non-finite predictions for $what"))
    return p
end

"""
Step function `θ ↦ c + Σ_{i: yᵢ ≤ θ} bᵢ` on sorted jump points: returns the jump
locations and the function value on each plateau `[y_(j), y_(j+1))`, plus the value
left of the first jump.
"""
function _iv_dq_steps(yv::AbstractVector, bv::AbstractVector, c::Real)
    o = sortperm(yv)
    ys = yv[o]
    cs = cumsum(bv[o])
    # merge ties: the value at a jump includes all tied points
    keep = [j == length(ys) || ys[j + 1] != ys[j] for j in eachindex(ys)]
    return ys[keep], c .+ cs[keep], float(c)
end

function _iv_dq_eval(steps, θ::Real)
    ys, vals, left = steps
    j = searchsortedlast(ys, θ)
    return j == 0 ? left : vals[j]
end

"""
Preliminary IPW quantile. DoubleML minimizes |score| with Brent's method started from
the bracket of `_get_bracket_guess` around `start` (a local search); here: the data
point starting the plateau that minimizes |score| among the plateaus inside that
bracket (ties: closest to `start`).
"""
function _iv_dq_ipw_root(steps, start, lo, hi)
    ys, vals, left = steps
    isempty(ys) && return start
    f = θ -> _iv_dq_eval(steps, θ)
    L = hi - lo
    δ = 0.1
    a, b = lo, hi
    while δ <= 1.0
        a = max(start - δ * L / 2, lo)
        b = min(start + δ * L / 2, hi)
        sign(f(a)) != sign(f(b)) && break
        δ += 0.1
    end
    inside = [j for j in eachindex(ys) if a <= ys[j] <= b]
    isempty(inside) && (inside = collect(eachindex(ys)))
    best = argmin(j -> (abs(vals[j]), abs(ys[j] - start)), inside)
    return ys[best]
end

"""
Brent's root finder, a transcription of SciPy's `brentq` (scipy/optimize/Zeros/
brentq.c) with its default tolerances, so that roots of step-function scores land on
the same side of a jump as in DoubleML.
"""
function _iv_brentq(f, xa::Float64, xb::Float64; xtol::Float64=2e-12,
                    rtol::Float64=4 * eps(Float64), maxiter::Int=100)
    xpre, xcur = xa, xb
    xblk = 0.0
    fblk = 0.0
    spre = 0.0
    scur = 0.0
    fpre = f(xpre)
    fcur = f(xcur)
    fpre == 0 && return xpre
    fcur == 0 && return xcur
    signbit(fpre) == signbit(fcur) && throw(ArgumentError("brentq: no sign change"))
    for _ in 1:maxiter
        if fpre != 0 && fcur != 0 && signbit(fpre) != signbit(fcur)
            xblk = xpre
            fblk = fpre
            spre = scur = xcur - xpre
        end
        if abs(fblk) < abs(fcur)
            xpre, xcur, xblk = xcur, xblk, xcur
            fpre, fcur, fblk = fcur, fblk, fcur
        end
        δ = (xtol + rtol * abs(xcur)) / 2
        sbis = (xblk - xcur) / 2
        (fcur == 0 || abs(sbis) < δ) && return xcur
        if abs(spre) > δ && abs(fcur) < abs(fpre)
            stry = if xpre == xblk
                -fcur * (xcur - xpre) / (fcur - fpre)
            else
                dpre = (fpre - fcur) / (xpre - xcur)
                dblk = (fblk - fcur) / (xblk - xcur)
                -fcur * (fblk * dblk - fpre * dpre) / (dblk * dpre * (fblk - fpre))
            end
            if 2 * abs(stry) < min(abs(spre), 3 * abs(sbis) - δ)
                spre = scur
                scur = stry
            else
                spre = sbis
                scur = sbis
            end
        else
            spre = sbis
            scur = sbis
        end
        xpre = xcur
        fpre = fcur
        xcur += abs(scur) > δ ? scur : (sbis > 0 ? δ : -δ)
        fcur = f(xcur)
    end
    return xcur
end

"""Root of the final score as DoubleML: bracket as in `_get_bracket_guess`, then
`brentq`. Without a sign change, the data point minimizing |score| is returned with
`converged = false`."""
function _iv_dq_final_root(f, steps, start, lo, hi)
    L = hi - lo
    δ = 0.1
    a, b = lo, hi
    found = false
    while !found && δ <= 1.0
        a = max(start - δ * L / 2, lo)
        b = min(start + δ * L / 2, hi)
        found = sign(f(a)) != sign(f(b))
        δ += 0.1
    end
    found && return (_iv_brentq(f, a, b), true)
    j = argmin(abs.(steps[2]))
    return (steps[1][j], false)
end

"""Weighted Gaussian KDE at 0 with Silverman's bandwidth (statsmodels
`KDEUnivariate(u).fit(kernel="gau", bw="silverman", weights=w, fft=False)`)."""
function _iv_dq_kde0(u::AbstractVector, w::AbstractVector)
    n = length(u)
    A = min(std(u), (quantile(u, 0.75) - quantile(u, 0.25)) / 1.349)
    A > 0 || (A = std(u))
    h = 0.9 * A * n^(-0.2)
    sw = sum(w)
    sw != 0 || throw(ArgumentError("weighted density at the quantile is not " *
                                   "identified (weights sum to zero)"))
    return sum(w .* pdf.(Normal(), u ./ h)) / (h * sw)
end

"""Stratified half split of `T` (by `strata`) and inner fold ids for the first half."""
function _iv_dq_prelim_split(T::Vector{Int}, strata, K::Int, rng::AbstractRNG)
    t1 = Int[]
    for s in sort!(unique(strata[T]))
        idx = T[strata[T] .== s]
        idx = idx[randperm(rng, length(idx))]
        append!(t1, idx[1:fld(length(idx), 2)])
    end
    sort!(t1)
    inner = zeros(Int, length(t1))
    st = strata[t1]
    for s in unique(st)
        pos = findall(==(s), st)
        pos = pos[randperm(rng, length(pos))]
        inner[pos] = (0:(length(pos) - 1)) .% K .+ 1
    end
    return t1, inner
end

function _iv_dq_prepare(data, outcome, treatment, instrument, covariates, folds, n_folds,
                        n_rep, rng, ctx)
    require_columns(data, [outcome, treatment, instrument]; context=ctx)
    y = _ml_column(data, outcome; context=ctx)
    d = _ml_column(data, treatment; context=ctx)
    z = _ml_column(data, instrument; context=ctx)
    _ml_check_binary_col(d, treatment, ctx)
    _ml_check_binary_col(z, instrument, ctx)
    covs, X, _, _, F = _ml_setup(data, [outcome, treatment, instrument], covariates,
                                 nothing, folds, n_folds, n_rep, rng, d .+ 2 .* z;
                                 context=ctx)
    return y, d, z, X, F
end

# ---------------------------------------------------------------------------
# LQTE core (one repetition)
# ---------------------------------------------------------------------------

function _iv_lqte_rep(y, d, z, X, fold, quantiles, ol, tl, il, trim, normalize, variance,
                      seeds, prelim, ipw_override, ctx)
    n = length(y)
    K = maximum(fold)
    Q = length(quantiles)
    strata = d .+ 2 .* z
    ymin, ymax = extrema(y)
    starts = [quantile(y[d .== dv], τ) for τ in quantiles, dv in (0, 1)]
    mz = zeros(n); m0 = zeros(n); m1 = zeros(n)
    g0 = zeros(n, Q, 2); g1 = zeros(n, Q, 2)
    ipw = zeros(K, Q, 2)
    for k in 1:K
        T = findall(!=(k), fold)
        E = findall(==(k), fold)
        s = view(seeds, :, k)
        t1, inner = prelim === nothing ?
                    _iv_dq_prelim_split(T, strata, K, Random.Xoshiro(s[1])) : prelim[k]
        t2 = setdiff(T, t1)
        # preliminary nuisances on the first half
        mzp = zeros(length(t1))
        for j in 1:maximum(inner)
            te = inner .== j
            tr = t1[.!te]
            mzp[te] = _iv_dq_proba(il, X[tr, :], z[tr], X[t1[te], :], s[1 + j],
                                   "the preliminary instrument propensity", ctx)
        end
        mzp = clamp.(mzp, trim, 1 - trim)
        zt1, dt1 = z[t1], d[t1]
        normalize && (mzp = _iv_dq_normalize(mzp, zt1))
        a0 = t1[zt1 .== 0]
        a1 = t1[zt1 .== 1]
        r0p = _iv_dq_proba(tl, X[a0, :], d[a0], X[t1, :], s[K + 2], "P(D=1|Z=0,X)", ctx)
        r1p = _iv_dq_proba(tl, X[a1, :], d[a1], X[t1, :], s[K + 3], "P(D=1|Z=1,X)", ctx)
        compp = mean(_iv_dq_phiD(mzp, r0p, r1p, zt1, dt1))
        wz = (zt1 ./ mzp .- (1 .- zt1) ./ (1 .- mzp)) ./ compp
        b2 = t2[z[t2] .== 0]
        b3 = t2[z[t2] .== 1]
        for (jd, dv) in enumerate((0, 1)), (iq, τ) in enumerate(quantiles)
            sgn = 2dv - 1.0
            sel = dt1 .== dv
            if ipw_override === nothing
                st = _iv_dq_steps(y[t1][sel], sgn .* wz[sel] ./ length(t1), -τ)
                ipw[k, iq, jd] = _iv_dq_ipw_root(st, starts[iq, jd], ymin, ymax)
            else
                ipw[k, iq, jd] = ipw_override[(k, dv, iq)]
            end
            q = ipw[k, iq, jd]
            tgt = Float64.((d .== dv) .& (y .<= q))
            base = K + 3 + 4 * ((jd - 1) * Q + iq - 1)
            # a subsample in which the event never (always) occurs predicts 0 (1)
            g0[E, iq, jd] = _iv_dq_proba(ol, X[b2, :], tgt[b2], X[E, :], s[base + 1],
                                         "P(D=$dv, Y≤q | Z=0, X)", ctx;
                                         constant_ok=true)
            g1[E, iq, jd] = _iv_dq_proba(ol, X[b3, :], tgt[b3], X[E, :], s[base + 2],
                                         "P(D=$dv, Y≤q | Z=1, X)", ctx;
                                         constant_ok=true)
        end
        # nuisances refitted on the whole training fold
        T0 = T[z[T] .== 0]
        T1 = T[z[T] .== 1]
        last = K + 3 + 4 * 2Q
        mz[E] = _iv_dq_proba(il, X[T, :], z[T], X[E, :], s[last + 1],
                             "the instrument propensity", ctx)
        m0[E] = _iv_dq_proba(tl, X[T0, :], d[T0], X[E, :], s[last + 2], "P(D=1|Z=0,X)",
                             ctx)
        m1[E] = _iv_dq_proba(tl, X[T1, :], d[T1], X[E, :], s[last + 3], "P(D=1|Z=1,X)",
                             ctx)
    end
    mz = clamp.(mz, trim, 1 - trim)
    madj = normalize ? _iv_dq_normalize(mz, z) : mz
    φD = _iv_dq_phiD(madj, m0, m1, z, d)
    comp = mean(φD)
    comp > 0 || throw(ArgumentError("$ctx: the estimated complier share is not positive " *
                                    "($(comp)); the first stage is not identified"))
    lpq = zeros(Q, 2); J = zeros(Q, 2); ψ = zeros(n, Q, 2)
    conv = trues(Q, 2)
    wz = z ./ madj .- (1 .- z) ./ (1 .- madj)
    for (jd, dv) in enumerate((0, 1)), (iq, τ) in enumerate(quantiles)
        sgn = 2dv - 1.0
        ind = d .== dv
        G0, G1 = g0[:, iq, jd], g1[:, iq, jd]
        c0 = sgn .* (G1 .- G0 .- z .* G1 ./ madj .+ (1 .- z) .* G0 ./ (1 .- madj))
        b = sgn .* wz .* ind ./ comp
        st = _iv_dq_steps(y[ind], b[ind] ./ n, mean(c0) / comp - τ)
        start = mean(ipw[:, iq, jd])
        f = θ -> mean(c0 .+ sgn .* wz .* ind .* (y .<= θ)) / comp - τ
        θ, ok = _iv_dq_final_root(f, st, start, ymin, ymax)
        conv[iq, jd] = ok
        lpq[iq, jd] = θ
        A = c0 .+ sgn .* wz .* ind .* (y .<= θ)
        J[iq, jd] = _iv_dq_kde0(y .- θ, b)
        ψ[:, iq, jd] = variance === :doubleml ? A ./ comp .- τ :
                       (A .- τ .* φD) ./ comp
    end
    return (lpq=lpq, J=J, psi=ψ, comp=comp, ipw=ipw, conv=conv, g0=g0, g1=g1, mz=mz,
            m0=m0, m1=m1)
end

function _iv_lqte_seeds(rng, K, Q, R)
    S = K + 3 + 4 * 2Q + 3
    return reshape(task_seeds(rng, S * K * R), S, K, R)
end

"""
    dml_lqte(data, outcome, treatment, instrument; quantiles=[0.25, 0.5, 0.75],
             covariates=Symbol[], outcome_learner=PenalizedLogisticLearner(),
             treatment_learner=PenalizedLogisticLearner(),
             instrument_learner=PenalizedLogisticLearner(), trim=0.01,
             normalize_ipw=true, variance=:influence, n_folds=5, n_rep=1, folds=nothing,
             n_boot=1000, rng=Random.default_rng()) -> LQTEEstimate

Local quantile treatment effects for compliers with a binary instrument, estimated by
double/debiased machine learning.

The LATE summarizes the effect of a treatment on the mean outcome of compliers; many
questions concern the whole distribution instead (does a training programme raise low
earnings or high earnings?). With a binary instrument ``Z``, a binary treatment ``D``
and covariates ``X``, the local potential quantile ``q_d(\\tau)`` is the
``\\tau``-quantile of ``Y(d)`` among compliers, and the local quantile treatment effect
(LQTE) is ``q_1(\\tau) - q_0(\\tau)``. Abadie, Angrist and Imbens (2002) and Frölich and
Melly (2013) show that the complier distributions, and hence the LQTE, are identified
under the LATE assumptions holding conditionally on ``X``: conditional independence
of ``Z`` given ``X``, exclusion, monotonicity (no defiers), a non-zero first stage and
overlap, ``0 < P(Z = 1 \\mid X) < 1``. Only overlap and the first stage can be checked
in the data. The LQTE compares quantiles of two marginal distributions; it equals the
quantile of individual effects only under rank invariance, which the data cannot
verify.

For each ``\\tau`` and ``d \\in \\{0, 1\\}`` the estimator solves the Neyman-orthogonal
moment condition of Belloni, Chernozhukov, Fernández-Val and Hansen (2017), as
implemented in DoubleML's `DoubleMLLPQ`,

```math
\\frac{s_d}{p_C} E\\Big[g_1(X) - g_0(X)
  + \\frac{Z\\,(1\\{D = d, Y \\le q\\} - g_1(X))}{m(X)}
  - \\frac{(1 - Z)(1\\{D = d, Y \\le q\\} - g_0(X))}{1 - m(X)}\\Big] = \\tau ,
```

with ``s_1 = 1``, ``s_0 = -1``, ``g_z(X) = P(D = d, Y \\le q \\mid Z = z, X)``,
``m(X) = P(Z = 1 \\mid X)`` clipped to ``[\\text{trim}, 1 - \\text{trim}]`` (and, with
`normalize_ipw`, rescaled so that the weights ``Z/m`` and ``(1 - Z)/(1 - m)`` each
average to one over the sample), and ``p_C`` the orthogonal (DML) estimate of the complier
share built from ``r_z(X) = P(D = 1 \\mid Z = z, X)``. Because ``g_z`` depends on the
unknown quantile, it is fitted at a preliminary inverse-probability-weighted estimate
of the quantile computed on half of each training fold (nested cross-fitting, as in
DoubleML); ``r_z`` and ``m`` are fitted on the whole training fold, and all nuisance
predictions are out of fold. With repeated cross-fitting (`n_rep > 1`) estimates and
covariances are aggregated by the median rule of Chernozhukov et al. (2018).

Standard errors use the influence function ``-\\psi / f``, with ``f`` the density of the
complier potential outcome at the quantile, estimated by a weighted Gaussian kernel with
Silverman's bandwidth (as in DoubleML). With `variance = :influence` (the default) the
influence function also accounts for the estimation of the complier share in the
denominator, adding ``-\\tau(\\phi_D - E\\phi_D)/E\\phi_D`` for the complier share score
``\\phi_D``; `variance = :doubleml` treats the share as known, ignoring that source of
sampling variation, and reproduces DoubleML's standard errors.
`confint(r; uniform=true)` gives simultaneous sup-t bands over the quantiles from a
multiplier bootstrap of the influence functions with Rademacher weights, appropriate
when the effect is read across the whole distribution rather than at one pre-specified
quantile. Quantile estimates in the tails are imprecise when the complier share is
small, since the effective sample is roughly the number of compliers. Use
[`dml_complier_cdf`](@ref) for the full distribution functions and [`late_ipw`](@ref) or
[`dml_iivm`](@ref) for the mean effect.

# Arguments
- `data`: a `DataFrame`; the columns used must not contain missing values.
- `outcome::Symbol`: the continuous outcome ``Y``.
- `treatment::Symbol`: the binary (0/1) treatment ``D``.
- `instrument::Symbol`: the binary (0/1) instrument ``Z``.

# Keywords
- `quantiles`: the quantile levels in ``(0, 1)`` (default `[0.25, 0.5, 0.75]`).
- `covariates::Vector{Symbol}`: the numeric covariates ``X`` (default none).
- `outcome_learner`, `treatment_learner`, `instrument_learner`: classifiers
  implementing [`fitpredict_proba`](@ref) for ``g_z``, ``r_z`` and ``m`` (default
  `PenalizedLogisticLearner()` for all three).
- `trim::Real`: clipping of the instrument propensity score (default 0.01).
- `normalize_ipw::Bool`: rescale the inverse-probability weights to average one
  (default `true`, as in DoubleML).
- `variance::Symbol`: `:influence` (default) or `:doubleml`, as described above.
- `n_folds::Integer`, `n_rep::Integer`, `folds`: number of cross-fitting folds
  (default 5, stratified by ``(D, Z)``), repetitions (default 1), and optional fixed
  fold assignments.
- `n_boot::Integer`: multiplier-bootstrap draws for the uniform bands (default 1000).
- `rng::AbstractRNG`: draws the folds, the nested splits, the learners' seeds and the
  bootstrap weights.

Keywords whose names begin with an underscore are internal (used to replay the
nested sample splits of the validation reference) and are not part of the public
interface.

# Returns
- An [`LQTEEstimate`](@ref); `coef(r)` are the LQTEs, `r.lpq` the local potential
  quantiles, `confint(r; uniform=true)` the simultaneous bands.

# Examples
```julia
using DrSnow, DataFrames, StableRNGs
rng = StableRNG(1)
n = 800
x1, x2 = randn(rng, n), randn(rng, n)
z = Float64.(rand(rng, n) .< 1 ./ (1 .+ exp.(-0.5 .* x1)))
u = rand(rng, n)
d = Float64.((u .< 0.2) .| ((u .< 0.6) .& (z .== 1)))       # 40% compliers
y = x1 .+ d .* (1.0 .+ randn(rng, n)) .+ randn(rng, n)      # effect spreads Y(1)
df = DataFrame(y=y, d=d, z=z, x1=x1, x2=x2)
r = dml_lqte(df, :y, :d, :z; covariates=[:x1, :x2], quantiles=[0.25, 0.5, 0.75],
             n_folds=3, rng=StableRNG(2))
coef(r), r.lpq
confint(r; uniform=true)
```

# References
- Abadie, A., Angrist, J., & Imbens, G. (2002). Instrumental variables estimates of
  the effect of subsidized training on the quantiles of trainee earnings.
  *Econometrica*, 70(1), 91–117.
- Frölich, M., & Melly, B. (2013). Unconditional quantile treatment effects under
  endogeneity. *Journal of Business & Economic Statistics*, 31(3), 346–357.
- Belloni, A., Chernozhukov, V., Fernández-Val, I., & Hansen, C. (2017). Program
  evaluation and causal inference with high-dimensional data. *Econometrica*, 85(1),
  233–298.
- Chernozhukov, V., Chetverikov, D., Demirer, M., Duflo, E., Hansen, C., Newey, W., &
  Robins, J. (2018). Double/debiased machine learning for treatment and structural
  parameters. *The Econometrics Journal*, 21(1), C1–C68.
- Bach, P., Chernozhukov, V., Kurz, M. S., & Spindler, M. (2022). DoubleML – An
  object-oriented implementation of double machine learning in Python. *Journal of
  Machine Learning Research*, 23(53), 1–6. (Reference software used for validation.)
"""
function dml_lqte(data, outcome::Symbol, treatment::Symbol, instrument::Symbol;
                  quantiles=[0.25, 0.5, 0.75], covariates=Symbol[],
                  outcome_learner=PenalizedLogisticLearner(),
                  treatment_learner=PenalizedLogisticLearner(),
                  instrument_learner=PenalizedLogisticLearner(), trim::Real=0.01,
                  normalize_ipw::Bool=true, variance::Symbol=:influence,
                  n_folds::Integer=5, n_rep::Integer=1, folds=nothing,
                  n_boot::Integer=1000, rng::AbstractRNG=Random.default_rng(),
                  _prelim=nothing, _ipw=nothing)
    ctx = "dml_lqte"
    qs = Float64.(collect(quantiles))
    isempty(qs) && throw(ArgumentError("$ctx: at least one quantile is required"))
    all(q -> 0 < q < 1, qs) || throw(ArgumentError("$ctx: quantiles must be in (0, 1)"))
    variance in (:influence, :doubleml) ||
        throw(ArgumentError("$ctx: variance must be :influence or :doubleml"))
    n_boot >= 1 || throw(ArgumentError("$ctx: n_boot must be positive"))
    trim = _ml_check_trim(trim)
    y, d, z, X, F = _iv_dq_prepare(data, outcome, treatment, instrument, covariates,
                                   folds, n_folds, n_rep, rng, ctx)
    n, R = size(F)
    K, Q = maximum(F), length(qs)
    seeds = _iv_lqte_seeds(rng, K, Q, R)
    all_coef = zeros(Q, R)
    all_vcov = Matrix{Float64}[]
    lpqs = zeros(Q, 2, R)
    lpq_ses = zeros(Q, 2, R)
    comps = zeros(R)
    conv = falses(2Q, R)
    Ψ = zeros(n, Q, R)
    supt = Float64[]
    for r in 1:R
        out = _iv_lqte_rep(y, d, z, X, F[:, r], qs, outcome_learner, treatment_learner,
                           instrument_learner, trim, normalize_ipw, variance,
                           view(seeds, :, :, r), _prelim, _ipw, ctx)
        IF0 = -out.psi[:, :, 1] ./ out.J[:, 1]'
        IF1 = -out.psi[:, :, 2] ./ out.J[:, 2]'
        IFd = IF1 .- IF0
        Ψ[:, :, r] = IFd
        all_coef[:, r] = out.lpq[:, 2] .- out.lpq[:, 1]
        push!(all_vcov, Matrix(Symmetric(IFd' * IFd)) ./ n^2)
        lpqs[:, :, r] = out.lpq
        lpq_ses[:, 1, r] = sqrt.(vec(sum(abs2, IF0; dims=1))) ./ n
        lpq_ses[:, 2, r] = sqrt.(vec(sum(abs2, IF1; dims=1))) ./ n
        comps[r] = out.comp
        conv[:, r] = vec(out.conv)
        _, st = _multiplier_bootstrap(rng, IFd, nothing, n_boot)
        append!(supt, st)
    end
    θ, V = _ml_aggregate(all_coef, all_vcov)
    lpq = dropdims(median(lpqs; dims=3); dims=3)
    lse = dropdims(median(lpq_ses; dims=3); dims=3)
    learners = [:ml_g => _ml_learner_name(outcome_learner),
                :ml_m_d => _ml_learner_name(treatment_learner),
                :ml_m_z => _ml_learner_name(instrument_learner)]
    return LQTEEstimate(qs, θ, V, lpq, lse, median(comps), Ψ, supt, all_coef, conv, F,
                        n, variance, learners, trim, normalize_ipw)
end

"""
    dml_complier_cdf(data, outcome, treatment, instrument; grid=nothing,
                     covariates=Symbol[], outcome_learner=PenalizedLogisticLearner(),
                     treatment_learner=PenalizedLogisticLearner(),
                     instrument_learner=PenalizedLogisticLearner(), trim=0.01,
                     normalize_ipw=true, n_folds=5, folds=nothing, level=0.95,
                     n_boot=1000, rearrange=true, rng=Random.default_rng())
        -> ComplierDistribution

Distribution functions of the potential outcomes ``Y(0)`` and ``Y(1)`` among compliers,
estimated with cross-fitted machine-learned nuisances and uniform confidence bands.

Imbens and Rubin (1997) show that, with a binary instrument that is valid and
monotone, the marginal distributions of both potential outcomes are identified for
compliers, and Abadie (2002) uses them to test distributional treatment effects. With
covariates, under the LATE assumptions holding conditionally on ``X``, the complier
distribution function at ``y`` is

```math
F_d(y) = \\frac{s_d}{p_C} E\\Big[g_1(X) - g_0(X)
  + \\frac{Z\\,(1\\{D = d, Y \\le y\\} - g_1(X))}{m(X)}
  - \\frac{(1 - Z)(1\\{D = d, Y \\le y\\} - g_0(X))}{1 - m(X)}\\Big] ,
```

with ``s_1 = 1``, ``s_0 = -1``, ``g_z(X) = P(D = d, Y \\le y \\mid Z = z, X)``,
``m(X) = P(Z = 1 \\mid X)`` and ``p_C`` the orthogonal estimate of the complier share
(Belloni, Chernozhukov, Fernández-Val and Hansen 2017). The expression is the
Neyman-orthogonal score of [`dml_lqte`](@ref) evaluated on a grid of outcome values
rather than solved for a quantile; ``g_z`` is fitted by `outcome_learner` on each
training fold separately for every grid point, ``d`` and ``z``, and all predictions are
out of fold. The difference ``F_1 - F_0`` describes how the instrument-induced
treatment shifts the outcome distribution of compliers; as with quantile effects, it
does not identify the distribution of individual effects.

Influence functions include the estimation of ``p_C``. The bands are sup-t bands over
the grid, computed separately for ``Y(0)`` and ``Y(1)`` from a multiplier bootstrap of
the influence functions with Rademacher weights (`n_boot` draws), so that each band
covers the whole function with asymptotic probability `level`. With `rearrange =
true` the estimates and the band limits are sorted, which makes them monotone
(Chernozhukov, Fernández-Val and Galichon 2010), and clipped to ``[0, 1]``;
rearrangement does not reduce the coverage of the bands and weakly reduces the
estimation error of the curve. Cross-fitting uses a single repetition.

# Arguments
- `data`: a `DataFrame`; the columns used must not contain missing values.
- `outcome::Symbol`: the outcome ``Y``.
- `treatment::Symbol`: the binary (0/1) treatment ``D``.
- `instrument::Symbol`: the binary (0/1) instrument ``Z``.

# Keywords
- `grid`: evaluation points (default `nothing`: the 5%, 10%, …, 95% sample quantiles
  of ``Y``).
- `covariates`, `outcome_learner`, `treatment_learner`, `instrument_learner`, `trim`,
  `normalize_ipw`, `n_folds`, `folds`, `rng`: as in [`dml_lqte`](@ref), with the same
  defaults.
- `level::Real`: confidence level of the uniform bands (default 0.95).
- `n_boot::Integer`: multiplier-bootstrap draws (default 1000).
- `rearrange::Bool`: monotonize and clip the estimates and bands (default `true`).

# Returns
- A [`ComplierDistribution`](@ref); `cd.table` holds the estimated distribution
  functions, standard errors and band limits at each grid point.

# Examples
```julia
using DrSnow, DataFrames, StableRNGs
rng = StableRNG(1)
n = 800
x1, x2 = randn(rng, n), randn(rng, n)
z = Float64.(rand(rng, n) .< 1 ./ (1 .+ exp.(-0.5 .* x1)))
u = rand(rng, n)
d = Float64.((u .< 0.2) .| ((u .< 0.6) .& (z .== 1)))
y = x1 .+ d .* (1.0 .+ randn(rng, n)) .+ randn(rng, n)
df = DataFrame(y=y, d=d, z=z, x1=x1, x2=x2)
cdist = dml_complier_cdf(df, :y, :d, :z; covariates=[:x1, :x2], n_folds=3,
                         rng=StableRNG(2))
cdist.table
```

# References
- Imbens, G. W., & Rubin, D. B. (1997). Estimating outcome distributions for compliers
  in instrumental variables models. *Review of Economic Studies*, 64(4), 555–574.
- Abadie, A. (2002). Bootstrap tests for distributional treatment effects in
  instrumental variable models. *Journal of the American Statistical Association*,
  97(457), 284–292.
- Belloni, A., Chernozhukov, V., Fernández-Val, I., & Hansen, C. (2017). Program
  evaluation and causal inference with high-dimensional data. *Econometrica*, 85(1),
  233–298.
- Chernozhukov, V., Fernández-Val, I., & Galichon, A. (2010). Quantile and probability
  curves without crossing. *Econometrica*, 78(3), 1093–1125.
"""
function dml_complier_cdf(data, outcome::Symbol, treatment::Symbol, instrument::Symbol;
                          grid=nothing, covariates=Symbol[],
                          outcome_learner=PenalizedLogisticLearner(),
                          treatment_learner=PenalizedLogisticLearner(),
                          instrument_learner=PenalizedLogisticLearner(), trim::Real=0.01,
                          normalize_ipw::Bool=true, n_folds::Integer=5, folds=nothing,
                          level::Real=0.95, n_boot::Integer=1000, rearrange::Bool=true,
                          rng::AbstractRNG=Random.default_rng())
    ctx = "dml_complier_cdf"
    0 < level < 1 || throw(ArgumentError("$ctx: level must be in (0, 1)"))
    n_boot >= 1 || throw(ArgumentError("$ctx: n_boot must be positive"))
    trim = _ml_check_trim(trim)
    y, d, z, X, F = _iv_dq_prepare(data, outcome, treatment, instrument, covariates,
                                   folds, n_folds, 1, rng, ctx)
    fold = F[:, 1]
    n = length(y)
    ys = grid === nothing ? unique(quantile(y, 0.05:0.05:0.95)) :
         sort!(unique(Float64.(collect(grid))))
    isempty(ys) && throw(ArgumentError("$ctx: empty grid"))
    J = length(ys)
    K = maximum(fold)
    seeds = _ml_seeds(rng, K, 3 + 4J, 1)
    z0 = BitVector(z .== 0)
    z1 = .!z0
    specs = [_MLNuisance(:ml_m_z, instrument_learner, z, X, true),
             _MLNuisance(:ml_m_d_z0, treatment_learner, d, X, true, z0),
             _MLNuisance(:ml_m_d_z1, treatment_learner, d, X, true, z1)]
    P = hcat(_ml_crossfit(specs, fold, view(seeds, :, 1:3, 1); parallel=true, context=ctx),
             zeros(n, 4J))
    # g_z(y) = P(D = d, Y ≤ y | Z = z, X); a training subsample in which the event
    # never (always) occurs gives the prediction 0 (1)
    col = 3
    for (jd, dv) in enumerate((0, 1)), j in 1:J
        tgt = Float64.((d .== dv) .& (y .<= ys[j]))
        for zmask in (z0, z1)
            col += 1
            for k in 1:K
                te = fold .== k
                tr = .!te .& zmask
                t = tgt[tr]
                P[te, col] = if all(==(t[1]), t)
                    fill(t[1], count(te))
                else
                    fitpredict_proba(outcome_learner, X[tr, :], t, X[te, :];
                                     rng=Random.Xoshiro(seeds[k, col, 1]))
                end
            end
        end
    end
    all(isfinite, P) || throw(ArgumentError("$ctx: learners returned non-finite " *
                                            "predictions"))
    mz = clamp.(P[:, 1], trim, 1 - trim)
    madj = normalize_ipw ? _iv_dq_normalize(mz, z) : mz
    φD = _iv_dq_phiD(madj, P[:, 2], P[:, 3], z, d)
    comp = mean(φD)
    comp > 0 || throw(ArgumentError("$ctx: the estimated complier share is not positive"))
    est = zeros(J, 2)
    IF = zeros(n, J, 2)
    for (jd, dv) in enumerate((0, 1)), j in 1:J
        sgn = 2dv - 1.0
        col = 3 + 2 * ((jd - 1) * J + j - 1)
        G0, G1 = P[:, col + 1], P[:, col + 2]
        ind = Float64.((d .== dv) .& (y .<= ys[j]))
        A = sgn .* (G1 .- G0 .+ z .* (ind .- G1) ./ madj .-
                    (1 .- z) .* (ind .- G0) ./ (1 .- madj))
        Fv = mean(A) / comp
        est[j, jd] = Fv
        IF[:, j, jd] = (A .- Fv .* φD) ./ comp
    end
    se = zeros(J, 2)
    crit = zeros(2)
    lower = zeros(J, 2)
    upper = zeros(J, 2)
    for jd in 1:2
        Ψ = IF[:, :, jd]
        se[:, jd] = sqrt.(vec(sum(abs2, Ψ; dims=1))) ./ n
        _, st = _multiplier_bootstrap(rng, Ψ, nothing, n_boot)
        crit[jd] = isempty(st) ? critical_value(level) : quantile(st, level)
        lower[:, jd] = est[:, jd] .- crit[jd] .* se[:, jd]
        upper[:, jd] = est[:, jd] .+ crit[jd] .* se[:, jd]
        if rearrange
            est[:, jd] = clamp.(sort(est[:, jd]), 0, 1)
            lower[:, jd] = clamp.(sort(lower[:, jd]), 0, 1)
            upper[:, jd] = clamp.(sort(upper[:, jd]), 0, 1)
        end
    end
    tab = DataFrame(y=ys, cdf0=est[:, 1], se0=se[:, 1], lower0=lower[:, 1],
                    upper0=upper[:, 1], cdf1=est[:, 2], se1=se[:, 2], lower1=lower[:, 2],
                    upper1=upper[:, 2])
    return ComplierDistribution(tab, comp, float(level), (crit[1], crit[2]), rearrange, n)
end
