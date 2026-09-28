# Confidence intervals for a constant additive effect by inverting Fisher
# randomization tests, and Hodges–Lehmann-type point estimates.
#
# All values of τ are tested against the same reference set of assignments (the
# exact support, or a fixed set of Monte Carlo draws), so the confidence set is a
# deterministic function of the data and the RNG seed.
#
# Linear statistics (difference in means, Lin's adjusted difference) satisfy
# T(z, Y - τZ) = a(z) - τ c(z), with a(z) = T(z, Y) and c(z) = T(z, Z). Whether draw
# z is at least as extreme as the observed assignment is then a step function of τ
# with a single breakpoint, and the p-value as a function of τ is evaluated exactly
# at every breakpoint by a sorted sweep: the interval is exact, with no grid.
# Other statistics are inverted by bracketing and bisection.

"""
    RandomizationInterval

Confidence interval for a constant additive treatment effect obtained by inverting
Fisher randomization tests, together with a Hodges–Lehmann-type point estimate;
returned by [`ri_confint`](@ref).

The interval is the set of hypothesized effects ``τ_0`` that the randomization tests
do not reject (or its hull, if that set is not an interval). Its coverage guarantee
is the one of the inverted tests: exact under the constant-additive-effect model
(given an exact reference distribution), and asymptotic for the average effect under
heterogeneity only for studentized statistics (see [`ri_confint`](@ref)).

# Fields
- `lower::Float64`, `upper::Float64`: end points (possibly `-Inf` / `Inf` for
  one-sided intervals or when no finite bound is rejected).
- `level::Float64`: confidence level.
- `estimate::Union{Nothing,Float64}`: Hodges–Lehmann-type point estimate (the effect
  at which the observed statistic equals the mean of its randomization
  distribution), or `nothing` when not requested or not defined (unsigned
  statistics).
- `statistic_name::String`, `alternative::Symbol`: statistic and alternative.
- `exact::Bool`, `n_draws::Int`: whether the reference set is the full support, and
  its size (or the number of Monte Carlo draws).
- `method::String`: inversion algorithm (exact sweep or bisection).
- `contiguous::Bool`: `false` if the set of non-rejected effects was not an
  interval; the reported interval is then its hull.

# Accessors
- `confint(r)` returns `(lower, upper)`.
"""
struct RandomizationInterval
    lower::Float64
    upper::Float64
    level::Float64
    estimate::Union{Nothing,Float64}
    statistic_name::String
    alternative::Symbol
    exact::Bool
    n_draws::Int
    method::String
    contiguous::Bool
end

StatsAPI.confint(r::RandomizationInterval) = (r.lower, r.upper)

function Base.show(io::IO, ::MIME"text/plain", r::RandomizationInterval)
    lv = round(r.level * 100; digits=2)
    println(io, "Randomization-based confidence interval for a constant additive effect")
    println(io, "Statistic: ", r.statistic_name, "; alternative: ", r.alternative)
    println(io, "Reference distribution: ", r.exact ? "exact, " : "Monte Carlo, ",
            r.n_draws, r.exact ? " assignments" : " draws")
    println(io, "Inversion: ", r.method)
    r.estimate === nothing ||
        @printf(io, "Hodges–Lehmann-type estimate: %.6g\n", r.estimate)
    @printf(io, "%g%% interval: [%.6g, %.6g]", lv, r.lower, r.upper)
    r.contiguous || print(io, "\nNote: the non-rejected set is not an interval; the " *
                              "interval shown is its hull.")
end

Base.show(io::IO, r::RandomizationInterval) =
    @printf(io, "RandomizationInterval([%.4g, %.4g], level %g)", r.lower, r.upper,
            r.level)

"""
    ri_confint(data, outcome, treatment; level=0.95, statistic=:diff_means,
               alternative=:two_sided, hodges_lehmann=true, mechanism=nothing,
               strata=nothing, cluster=nothing, id=nothing, covariates=Symbol[],
               nperm=2_000, exact=:auto, rng=Random.default_rng(),
               threaded=Threads.nthreads() > 1, tol=1e-6) -> RandomizationInterval

Confidence interval for a constant additive treatment effect ``τ``, obtained by
inverting Fisher randomization tests of the sharp nulls ``H_0: Y_i(1) = Y_i(0) + τ_0``
for every unit, with a Hodges–Lehmann-type point estimate.

Under the constant-effect model ``Y_i(1) = Y_i(0) + τ``, every sharp null
``τ = τ_0`` can be tested exactly with [`randomization_test`](@ref) on the adjusted
outcomes ``Y - τ_0 Z``, and the set of values not rejected at level ``α`` is a
confidence set with coverage at least ``1 - α`` in finite samples (Rosenbaum 2002,
ch. 2; Imbens and Rubin 2015, ch. 5). The two-sided interval is **equal-tailed**: it
contains every ``τ_0`` rejected by neither one-sided test at level ``α/2``. It is
therefore not the inversion of the two-sided test reported by
[`randomization_test`](@ref), which compares ``|T|`` with ``|T_{obs}|``; for
asymmetric randomization distributions the two can disagree near the end points.
One-sided alternatives give one-sided bounds (`:greater` gives ``[L, ∞)``, `:less`
gives ``(-∞, U]``).

When effects are heterogeneous the constant-effect model is misspecified, and the
interval need not cover the average effect for statistics such as the plain
difference in means. Testing ``τ = τ_0`` with a studentized statistic is a test of
the weak null of an average effect ``τ_0`` (the adjusted outcomes have average
effect zero exactly when ``\\bar τ = τ_0``), so inverting the `:studentized` or
`:lin_studentized` statistic gives an interval that is exact under the constant
effect model and asymptotically valid for the average effect in the settings of Wu
and Ding (2021) and Zhao and Ding (2021).

All values of ``τ_0`` are tested against the same reference set of assignments, so
the interval is a deterministic function of the data and of the random seed. For the
statistics that are linear in the outcomes (`:diff_means`, `:lin`) the p-value is a
step function of ``τ_0`` with known break points, and the end points are located
exactly by a sweep over them. For the other statistics the end points are found by
bracketing and bisection to tolerance `tol` (in units of a Neyman standard error);
this presumes that the one-sided p-values are monotone in ``τ_0``, which holds for
the difference in means and for rank statistics and approximately for studentized
ones.

The point estimate is the value of ``τ_0`` at which the observed statistic equals the
mean of its randomization distribution (Hodges and Lehmann 1963). For the rank-sum
statistic without strata this is the classical Hodges–Lehmann estimator, the median
of the ``n_1 n_0`` pairwise differences ``Y_i - Y_j`` between treated units ``i`` and
control units ``j``; for the difference in means it is the (stratum-weighted)
difference in means itself, up to Monte Carlo error when the reference set is
simulated.

# Arguments
- `data`, `outcome`, `treatment`: as in [`randomization_test`](@ref).

# Keywords
- `level::Real`: confidence level; default 0.95.
- `statistic`: as in [`randomization_test`](@ref). `:ks` gives a confidence set
  based on the upper-tail Kolmogorov–Smirnov test, without a point estimate.
- `alternative::Symbol`: `:two_sided` (default; equal-tailed), `:greater` or
  `:less`.
- `hodges_lehmann::Bool`: also compute the Hodges–Lehmann-type point estimate;
  default `true`.
- `mechanism`, `strata`, `cluster`, `id`, `covariates`, `nperm`, `exact`, `rng`,
  `threaded`: as in [`randomization_test`](@ref); the default `nperm` is 2 000.
- `tol::Real`: bisection tolerance for non-linear statistics, relative to a Neyman
  standard error; default ``10^{-6}``.

# Returns
- [`RandomizationInterval`](@ref); `confint(ci)` gives `(lower, upper)` and
  `ci.estimate` the point estimate.

# Examples
```julia
using DrSnow, DataFrames, StableRNGs
rng = StableRNG(3)
df = DataFrame(block=repeat(1:4; inner=10), d=repeat([1, 1, 1, 1, 1, 0, 0, 0, 0, 0], 4))
df.y = 0.3 .* df.block .+ 1.0 .* df.d .+ randn(rng, 40)
ci = ri_confint(df, :y, :d; strata=:block, rng=rng)
confint(ci), ci.estimate
ri_confint(df, :y, :d; statistic=:rank_sum, rng=rng)     # unstratified Hodges–Lehmann
```

# References
- Hodges, J. L., & Lehmann, E. L. (1963). Estimates of location based on rank
  tests. *Annals of Mathematical Statistics*, 34(2), 598–611.
- Rosenbaum, P. R. (2002). *Observational Studies* (2nd ed.), ch. 2. Springer.
- Imbens, G. W., & Rubin, D. B. (2015). *Causal Inference for Statistics, Social,
  and Biomedical Sciences: An Introduction*, ch. 5. Cambridge University Press.
- Wu, J., & Ding, P. (2021). Randomization tests for weak null hypotheses in
  randomized experiments. *Journal of the American Statistical Association*,
  116(536), 1898–1913.
- Zhao, A., & Ding, P. (2021). Covariate-adjusted Fisher randomization tests for the
  average treatment effect. *Journal of Econometrics*, 225(2), 278–294.
"""
function ri_confint(data, outcome::Symbol, treatment::Symbol; level::Real=0.95,
                    statistic=:diff_means, alternative::Symbol=:two_sided,
                    hodges_lehmann::Bool=true, mechanism=nothing,
                    strata::Union{Nothing,Symbol}=nothing,
                    cluster::Union{Nothing,Symbol}=nothing,
                    id::Union{Nothing,Symbol}=nothing,
                    covariates::Vector{Symbol}=Symbol[], nperm::Integer=2_000,
                    exact=:auto, rng::AbstractRNG=Random.default_rng(),
                    threaded::Bool=Threads.nthreads() > 1, tol::Real=1e-6)
    ctxname = "ri_confint"
    0 < level < 1 || throw(ArgumentError("level must be in (0, 1)"))
    _ri_check_alternative(alternative)
    stat = _ri_parse_statistic(statistic)
    _ri_check_covariates(stat, covariates, ctxname)
    signed = _ri_signed(stat)
    signed || alternative === :two_sided ||
        throw(ArgumentError("$ctxname: one-sided intervals need a signed statistic"))
    design = _ri_design(data, treatment; mechanism, strata, cluster, id, covariates,
                        sortcols=[outcome], context=ctxname)
    y = Vector{Float64}(data[design.rows, outcome])
    all(isfinite, y) || throw(ArgumentError("$ctxname: outcome has non-finite values"))
    plan = _ri_plan(design.mech, nperm, exact, rng)
    alpha = 1 - level
    # thresholds for the (greater, less) tails; -1 disables a tail
    thr = alternative === :two_sided ? (signed ? (alpha / 2, alpha / 2) : (alpha, -1.0)) :
          alternative === :greater ? (alpha, -1.0) : (-1.0, alpha)
    if _ri_linear(stat)
        lo, hi, est, contiguous = _ri_ci_linear(stat, y, design, plan, thr, threaded,
                                                hodges_lehmann)
        method = "exact inversion over the reference set (statistic linear in the " *
                 "outcome)"
    else
        lo, hi, est, contiguous = _ri_ci_bisection(stat, y, design, plan, thr, threaded,
                                                   hodges_lehmann && signed, tol)
        method = "bracketing and bisection (tolerance $(tol) × Neyman s.e.)"
    end
    return RandomizationInterval(lo, hi, float(level), est, _ri_name(stat), alternative,
                                 plan.exact, plan.B, method, contiguous)
end

# ---------------------------------------------------------------------------------
# Exact inversion for linear statistics
# ---------------------------------------------------------------------------------

# One tail of the p-value as a function of τ: rows are extreme for τ ≥ t (up), for
# τ ≤ t (down), or for every τ (const). p(τ) = (C + U(≤τ) + D(≥τ)) / W.
struct _ri_StepTail
    tu::Vector{Float64}; cu::Vector{Float64}   # sorted breakpoints, cumulative weight
    td::Vector{Float64}; cd::Vector{Float64}
    C::Float64
    W::Float64
end

function _ri_steptail(up, down, C, W)
    su = sort(up; by=first)
    sd = sort(down; by=first)
    return _ri_StepTail(first.(su), cumsum(last.(su)), first.(sd), cumsum(last.(sd)), C, W)
end

function (t::_ri_StepTail)(tau::Float64)
    ku = searchsortedlast(t.tu, tau)
    u = ku == 0 ? 0.0 : t.cu[ku]
    kd = searchsortedfirst(t.td, tau)
    dtot = isempty(t.cd) ? 0.0 : t.cd[end]
    d = dtot - (kd <= 1 ? 0.0 : t.cd[kd - 1])
    return (t.C + u + d) / t.W
end

function _ri_ci_linear(stat, y, design, plan, thr, threaded, want_est)
    ctx = design.ctx
    zf = Float64.(design.z)
    vals = _ri_map(z -> (_ri_eval(stat, y, z, ctx), _ri_eval(stat, zf, z, ctx)),
                   plan, design.mech, design.z, 2; threaded)
    ao = _ri_eval(stat, y, design.z, ctx)
    co = _ri_eval(stat, zf, design.z, ctx)
    w = _ri_weights(plan)
    ok = [!isnan(vals[i, 1]) && !isnan(vals[i, 2]) for i in axes(vals, 1)]
    a = vals[ok, 1]; c = vals[ok, 2]; w = w[ok]
    W = sum(w)
    epsd = 1e-10 * max(1.0, abs(co))
    upg = Tuple{Float64,Float64}[]; downg = Tuple{Float64,Float64}[]
    upl = Tuple{Float64,Float64}[]; downl = Tuple{Float64,Float64}[]
    Cg = 0.0; Cl = 0.0
    tol = _ri_tol(ao)
    for i in eachindex(a)
        d = co - c[i]                 # extreme (greater) iff τ d ≥ ao - a
        if abs(d) <= epsd
            a[i] >= ao - tol && (Cg += w[i])
            a[i] <= ao + tol && (Cl += w[i])
        else
            t = (ao - a[i]) / d
            if d > 0
                push!(upg, (t, w[i])); push!(downl, (t, w[i]))
            else
                push!(downg, (t, w[i])); push!(upl, (t, w[i]))
            end
        end
    end
    pg = _ri_steptail(upg, downg, Cg, W)
    pl = _ri_steptail(upl, downl, Cl, W)
    accept(tau) = (thr[1] < 0 || pg(tau) > thr[1]) && (thr[2] < 0 || pl(tau) > thr[2])
    bps = sort!(unique!(vcat(pg.tu, pg.td)))
    lo, hi, contiguous = _ri_accepted_hull(accept, bps)
    est = nothing
    if want_est
        abar = sum(w .* a) / W
        cbar = sum(w .* c) / W
        den = co - cbar
        abs(den) > epsd ||
            error("Hodges–Lehmann estimate not identified: the statistic does not " *
                  "respond to a shift in the treated outcomes")
        est = (ao - abar) / den
    end
    return lo, hi, est, contiguous
end

# Hull of the accepted pieces of the real line partitioned by sorted breakpoints.
function _ri_accepted_hull(accept, bps::Vector{Float64})
    if isempty(bps)
        return accept(0.0) ? (-Inf, Inf, true) :
               error("the confidence set is empty at this level")
    end
    m = length(bps)
    # pieces: 1 = (-Inf, b1); 2j = {bj}; 2j+1 = (bj, bj+1) (last: (bm, Inf))
    npieces = 2m + 1
    rep(k) = k == 1 ? bps[1] - (1 + abs(bps[1])) :
             iseven(k) ? bps[k ÷ 2] :
             (k == npieces ? bps[m] + (1 + abs(bps[m])) :
              (bps[(k - 1) ÷ 2] + bps[(k + 1) ÷ 2]) / 2)
    acc = [accept(rep(k)) for k in 1:npieces]
    idx = findall(acc)
    isempty(idx) && error("the confidence set is empty at this level: every effect " *
                          "value is rejected")
    kf, kl = first(idx), last(idx)
    lo = kf == 1 ? -Inf : bps[kf ÷ 2]
    hi = kl == npieces ? Inf : (iseven(kl) ? bps[kl ÷ 2] : bps[(kl + 1) ÷ 2])
    contiguous = all(acc[kf:kl])
    return lo, hi, contiguous
end

# ---------------------------------------------------------------------------------
# Bisection for non-linear statistics
# ---------------------------------------------------------------------------------

function _ri_ci_bisection(stat, y, design, plan, thr, threaded, want_est, tol)
    ctx = design.ctx
    zs = _ri_assignments(plan, design.mech, design.z)
    w = _ri_weights(plan)
    zf = Float64.(design.z)
    cache = Dict{Float64,NTuple{3,Float64}}()
    function evaluate(tau)
        haskey(cache, tau) && return cache[tau]
        yt = y .- tau .* zf
        prep = _ri_prepare(stat, yt, ctx)
        v = vec(_ri_map_list(z -> _ri_eval(stat, prep, z, ctx), zs, 1; threaded))
        o = _ri_eval(stat, prep, design.z, ctx)
        isnan(o) && error("statistic undefined for the observed assignment")
        pg, _, _ = _ri_pvalue(o, v, w, :greater)
        pl, _, _ = _ri_pvalue(o, v, w, :less)
        ok = .!isnan.(v)
        mu = sum(w[ok] .* v[ok]) / sum(w[ok])
        h = o - mu
        # h is exactly zero on flat stretches (e.g. rank statistics between two
        # pairwise differences); do not let rounding noise decide its sign
        abs(h) <= 1e-9 * max(abs(o), abs(mu), 1e-3) && (h = 0.0)
        return cache[tau] = (pg, pl, h)
    end
    center = _ri_stratified_dim(y, design.z, ctx)
    y1 = y[design.z]; y0 = y[.!design.z]
    s = sqrt(var(y1) / length(y1) + var(y0) / length(y0))
    (isfinite(s) && s > 0) || (s = max(std(y), 1.0))
    step = tol * s
    if thr[2] < 0 && thr[1] >= 0 && !_ri_signed(stat)
        # unsigned statistic: accept iff upper-tail p > α
        acc = tau -> evaluate(tau)[1] > thr[1]
        acc(center) || error("the difference in means is rejected by the " *
                             "$(_ri_name(stat)) test at this level; the confidence " *
                             "set cannot be located by bisection")
        lo = _ri_bisect_boundary(acc, center, -s, step)
        hi = _ri_bisect_boundary(acc, center, s, step)
        return lo, hi, nothing, true
    end
    accg = tau -> thr[1] < 0 || evaluate(tau)[1] > thr[1]
    accl = tau -> thr[2] < 0 || evaluate(tau)[2] > thr[2]
    start = center
    if !(accg(start) && accl(start))
        # the point estimate is used as an alternative starting point
        start = _ri_hl_bisection(evaluate, center, s, step)
        (accg(start) && accl(start)) ||
            error("could not find a non-rejected effect value to start the inversion")
    end
    lo = thr[1] < 0 ? -Inf : _ri_bisect_boundary(accg, start, -s, step)
    hi = thr[2] < 0 ? Inf : _ri_bisect_boundary(accl, start, s, step)
    est = want_est ? _ri_hl_bisection(evaluate, center, s, step) : nothing
    return lo, hi, est, true
end

# Starting from an accepted point `x0`, move in direction sign(dir) with doubling
# steps until a rejected point is found, then bisect. Returns ±Inf if none is found.
function _ri_bisect_boundary(accept, x0, dir, step)
    a = x0
    b = x0 + dir
    k = 0
    while accept(b)
        a = b
        dir *= 2
        b = x0 + dir
        k += 1
        k > 60 && return dir > 0 ? Inf : -Inf
    end
    while abs(b - a) > step
        m = (a + b) / 2
        accept(m) ? (a = m) : (b = m)
    end
    return (a + b) / 2
end

# Hodges–Lehmann-type estimate: midpoint of sup{τ: h(τ) > 0} and inf{τ: h(τ) < 0},
# where h(τ) = T_obs(τ) - E₀[T(τ)] is non-increasing in τ.
function _ri_hl_bisection(evaluate, center, s, step)
    pos = tau -> evaluate(tau)[3] > 0
    neg = tau -> evaluate(tau)[3] < 0
    sup_pos = _ri_bisect_sign(pos, center, s, step, true)
    inf_neg = _ri_bisect_sign(neg, center, s, step, false)
    return (sup_pos + inf_neg) / 2
end

# For a predicate that is true on (-∞, t*) (increasing=true: find sup of true set) or
# true on (t*, ∞) (find inf of true set).
function _ri_bisect_sign(pred, center, s, step, lefttrue::Bool)
    # bracket: a satisfies pred, b does not
    dir = lefttrue ? -s : s
    a = center
    k = 0
    while !pred(a)
        a = center + dir
        dir *= 2
        k += 1
        k > 60 && error("Hodges–Lehmann estimate: could not bracket the root")
    end
    b = a
    dir = lefttrue ? s : -s
    k = 0
    while pred(b)
        b = a + dir
        dir *= 2
        k += 1
        k > 60 && error("Hodges–Lehmann estimate: could not bracket the root")
    end
    while abs(b - a) > step
        m = (a + b) / 2
        pred(m) ? (a = m) : (b = m)
    end
    return (a + b) / 2
end
