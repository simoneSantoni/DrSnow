# Analytic power, minimum detectable effects and sample sizes.
#
# Every calculator takes keyword arguments and leaves exactly one of `power`, the
# effect and one design size (sample size, number of clusters, ...) as `nothing`; that
# quantity is solved for. The power of a linear estimator with standard error `se` and
# `dof` degrees of freedom is the exact power of the two-sided (or one-sided) t test
# (noncentral t, both rejection tails, as in R's `pwr` and `PowerUpR`), or of the z
# test with `distribution = :normal`.

"""
    PowerAnalysis

Result of an analytic power, minimum-detectable-effect or sample-size calculation.

A `PowerAnalysis` records one point on the power surface of a design: the power of a
level-`alpha` test of the null of no effect, the effect size at which that power is
attained and the design sizes (units, clusters, blocks, periods), with the one
quantity that was left unspecified solved for. It is returned by
[`power_means`](@ref), [`power_proportions`](@ref), [`power_cluster`](@ref),
[`power_blocked`](@ref), [`power_did`](@ref), [`power_iv`](@ref) and
[`power_rd`](@ref). For every calculator except the arcsine test of proportions the
power is that of a test whose statistic is `effect / se`, with a noncentral t (or
normal) distribution under the alternative, so `se` and `dof` fully describe the
design's precision.

The minimum detectable effect (MDE) is the smallest true effect that the design
detects with the stated power (Bloom 1995). The solved MDE uses the exact power
function (both rejection tails, noncentral t). `mde_multiplier` reports, for
comparison, Bloom's multiplier approximation
``\\text{MDE} \\approx (t_{1-\\alpha/2} + t_{\\text{power}})\\,\\text{se}``, which ignores
the far rejection tail and the noncentrality of the t distribution and is therefore
slightly different from the exact value, most visibly with few degrees of freedom.

# Fields
- `design::String`: description of the design and of the test.
- `solved::Symbol`: the quantity that was solved for: `:power`, the effect keyword
  (`:effect`, or `:p1` for proportions) or a size keyword such as `:n`,
  `:n_clusters`, `:cluster_size`, `:n_blocks` or `:compliance`.
- `power::Float64`: power of the test at the (solved) design.
- `effect::Float64`: effect in outcome units (the MDE when `solved` is the effect
  keyword); for [`power_proportions`](@ref) the treated-arm proportion `p1`.
- `parameters::NamedTuple`: every input with the solved value filled in, plus derived
  quantities (e.g. `design_effect`, `variance_factor`, `itt`, `n_units`). Sizes solved
  for are continuous; round them up (`ceil`) for a feasible design.
- `se::Float64`: standard error of the estimator at the design (`NaN` when power is not
  a function of effect / se, as for the arcsine test of proportions).
- `dof::Float64`: degrees of freedom of the t reference distribution (`Inf` for the
  normal).
- `alpha::Float64`: significance level of the test.
- `alternative::Symbol`: `:two_sided`, `:greater` or `:less`.
- `distribution::Symbol`: `:t` or `:normal`.
- `mde_multiplier::Float64`: Bloom's multiplier approximation of the MDE at the power
  in `power` (`NaN` where `se` is undefined).
- `note::String`: approximations and caveats specific to the calculation.

# References
- Bloom, H. S. (1995). Minimum detectable effects: A simple way to report the
  statistical power of experimental designs. *Evaluation Review*, 19(5), 547–556.
- Cohen, J. (1988). *Statistical Power Analysis for the Behavioral Sciences* (2nd ed.).
  Lawrence Erlbaum Associates.
"""
struct PowerAnalysis
    design::String
    solved::Symbol
    power::Float64
    effect::Float64
    parameters::NamedTuple
    se::Float64
    dof::Float64
    alpha::Float64
    alternative::Symbol
    distribution::Symbol
    mde_multiplier::Float64
    note::String
end

function Base.show(io::IO, ::MIME"text/plain", r::PowerAnalysis)
    println(io, "Power analysis: ", r.design)
    alt = r.alternative === :two_sided ? "two-sided" : "one-sided ($(r.alternative))"
    ref = r.distribution === :t ? @sprintf("t(%.6g)", r.dof) : "normal"
    @printf(io, "Test: %s, alpha = %.4g, %s reference\n", alt, r.alpha, ref)
    println(io, "Solved for: ", r.solved)
    for (k, v) in pairs(r.parameters)
        v isa Real || continue
        if k === r.solved && _des_is_size(k)
            @printf(io, "  %-22s %.6g (round up: %d)\n", string(k), v, ceil(Int, v - 1e-9))
        else
            @printf(io, "  %-22s %.6g\n", string(k), v)
        end
    end
    isnan(r.se) || @printf(io, "Standard error: %.6g\n", r.se)
    isnan(r.mde_multiplier) ||
        @printf(io, "Bloom multiplier MDE at this power: %.6g\n", r.mde_multiplier)
    isempty(r.note) || print(io, "Note: ", r.note)
end

Base.show(io::IO, r::PowerAnalysis) =
    @printf(io, "PowerAnalysis(%s: power = %.4g, effect = %.4g)", r.solved, r.power,
            r.effect)

_des_is_size(k::Symbol) = k in (:n, :n_clusters, :cluster_size, :n_blocks, :block_size,
                                :n_units, :sample_size)

# ------------------------------------------------------------------ power engine

function _des_check_alpha(alpha, alternative, distribution)
    0 < alpha < 1 || throw(ArgumentError("alpha must be in (0, 1), got $alpha"))
    alternative in (:two_sided, :greater, :less) ||
        throw(ArgumentError("alternative must be :two_sided, :greater or :less"))
    distribution in (:t, :normal) ||
        throw(ArgumentError("distribution must be :t or :normal"))
    return nothing
end

"""
Power of the level-`alpha` test of `θ = 0` for an estimator with noncentrality
`ncp = effect / se`: t(`dof`) (noncentral t under the alternative) or normal reference.
"""
function _des_power_ncp(ncp::Real, dof::Real, alpha::Real, alternative::Symbol,
                        distribution::Symbol)
    if distribution === :t
        dof > 0 || throw(ArgumentError("the design has non-positive degrees of " *
                                       "freedom ($dof); increase the sample size"))
        ref = TDist(dof)
        alt = NoncentralT(dof, ncp)
    else
        ref = Normal()
        alt = Normal(ncp, 1.0)
    end
    if alternative === :two_sided
        c = quantile(ref, 1 - alpha / 2)
        return ccdf(alt, c) + cdf(alt, -c)
    elseif alternative === :greater
        return ccdf(alt, quantile(ref, 1 - alpha))
    else
        return cdf(alt, -quantile(ref, 1 - alpha))
    end
end

# Bloom's multiplier MDE `(t_{1-α/2} + t_{power}) se` (PowerUpR's `.mdes.fun`).
function _des_mde_multiplier(power, se, dof, alpha, alternative, distribution)
    (isfinite(se) && 0 < power < 1) || return NaN
    ref = distribution === :t ? TDist(dof) : Normal()
    t1 = abs(quantile(ref, alternative === :two_sided ? alpha / 2 : alpha))
    t2 = abs(quantile(ref, power))
    m = power >= 0.5 ? t1 + t2 : t1 - t2
    return m * se
end

"""
Bisection for the root of `f(x) = target` for a monotone `f` on `[lo, hi]`; `hi = Inf`
is expanded geometrically. `increasing` gives the direction of `f`.
"""
function _des_root(f, target::Real, lo::Real, hi::Real; increasing::Bool=true,
                   what::AbstractString="quantity", maxexpand::Int=200)
    g(x) = increasing ? f(x) - target : target - f(x)
    a = float(lo)
    b = isfinite(hi) ? float(hi) : max(2 * abs(a), 1.0)
    ga = g(a)
    ga > 0 && throw(ArgumentError(increasing ?
        "the target is already exceeded at the lower bound $(a) of $what" :
        "the target cannot be reached for $what in [$(lo), $(hi)]"))
    k = 0
    while g(b) < 0
        isfinite(hi) && throw(ArgumentError("the target cannot be reached for $what " *
                                            "in [$(lo), $(hi)]"))
        a = b
        b *= 2
        k += 1
        k > maxexpand && throw(ArgumentError("the target cannot be reached for $what"))
    end
    for _ in 1:300
        m = (a + b) / 2
        (m == a || m == b) && break
        g(m) < 0 ? (a = m) : (b = m)
        abs(b - a) <= 1e-13 * max(1.0, abs(b)) && break
    end
    return (a + b) / 2
end

"""
Shared driver. `pars` holds every input (one entry is `nothing`: the unknown, unless
`power` is). `powfun(p)` returns the power for a complete parameter set `p`, and
`sefun(p)` returns `(se, dof)` (`se = NaN` when not defined). `ranges[k]` gives the
search interval and monotonicity for each solvable key.
"""
function _des_analysis(design::String, pars::NamedTuple, power, powfun, sefun,
                       effect_key::Symbol, ranges::Dict{Symbol,<:Tuple},
                       alpha, alternative, distribution; note::String="")
    unknown = [k for (k, v) in pairs(pars) if v === nothing]
    if power === nothing
        isempty(unknown) ||
            throw(ArgumentError("leave exactly one of `power` and " *
                                "$(join(sort!(collect(keys(ranges))), ", ")) as " *
                                "`nothing`; got also $(join(unknown, ", "))"))
        p = pars
        pw = powfun(p)
        solved = :power
    else
        (0 < power < 1) || throw(ArgumentError("power must be in (0, 1)"))
        power > alpha || throw(ArgumentError("the target power must exceed alpha " *
                                             "(the power at a zero effect)"))
        length(unknown) == 1 ||
            throw(ArgumentError("give `power` and leave exactly one of " *
                                "$(join(sort!(collect(keys(ranges))), ", ")) as " *
                                "`nothing` to solve for it"))
        u = unknown[1]
        haskey(ranges, u) || throw(ArgumentError("cannot solve for `$u`; solvable: " *
                                                 join(sort!(collect(keys(ranges))), ", ")))
        lo, hi, increasing = ranges[u]
        f = x -> powfun(merge(pars, NamedTuple{(u,)}((x,))))
        x = _des_root(f, power, lo, hi; increasing=increasing, what="`$u`")
        p = merge(pars, NamedTuple{(u,)}((x,)))
        pw = powfun(p)
        solved = u
    end
    se, dof = sefun(p)
    eff = float(p[effect_key])
    mult = _des_mde_multiplier(pw, se, dof, alpha, alternative, distribution)
    params = merge(p, (power=pw,))
    return PowerAnalysis(design, solved, pw, eff, params, float(se), float(dof),
                         float(alpha), alternative, distribution, mult, note)
end

function _des_linear(design, pars, power, effect_key, sefun, ranges, alpha, alternative,
                     distribution; note="")
    _des_check_alpha(alpha, alternative, distribution)
    sgn = alternative === :less ? -1.0 : 1.0
    powfun = p -> begin
        se, dof = sefun(p)
        _des_power_ncp(p[effect_key] / se, dof, alpha, alternative, distribution)
    end
    # the effect is searched on the |effect| scale
    ranges[effect_key] = (0.0, Inf, true)
    if power !== nothing && pars[effect_key] === nothing
        pf2 = p -> powfun(merge(p, NamedTuple{(effect_key,)}((sgn * p[effect_key],))))
        r = _des_analysis(design, pars, power, pf2, sefun, effect_key, ranges, alpha,
                          alternative, distribution; note=note)
        params = merge(r.parameters, NamedTuple{(effect_key,)}((sgn * r.effect,)))
        return PowerAnalysis(r.design, r.solved, r.power, sgn * r.effect, params, r.se,
                             r.dof, r.alpha, r.alternative, r.distribution,
                             r.mde_multiplier, r.note)
    end
    return _des_analysis(design, pars, power, powfun, sefun, effect_key, ranges, alpha,
                         alternative, distribution; note=note)
end

_des_pos(x, name) = (x === nothing || x > 0) ? x :
                    throw(ArgumentError("$name must be positive, got $x"))
_des_unit(x, name; closed=false) =
    (closed ? 0 <= x <= 1 : 0 < x < 1) ? x :
    throw(ArgumentError("$name must be in $(closed ? "[0, 1]" : "(0, 1)"), got $x"))
_des_r2(x, name) = (0 <= x < 1) ? float(x) :
                   throw(ArgumentError("$name must be in [0, 1), got $x"))

# ------------------------------------------------------------------ two-sample means

"""
    power_means(; effect=nothing, sd=1.0, n=nothing, power=nothing, p_treat=0.5,
                alpha=0.05, alternative=:two_sided, r2=0.0, n_covariates=0,
                distribution=:t) -> PowerAnalysis

Power, minimum detectable effect or total sample size for a difference in means in a
completely randomized experiment, optionally with covariate adjustment.

The design assigns `p n` of `n` units to treatment completely at random, and the
estimand is the average treatment effect ``\\tau = E[Y(1) - Y(0)]``, estimated by the
difference in means or, with baseline covariates that explain a share ``R^2`` of the
outcome variance, by regression adjustment (ANCOVA; Lin 2013 for the interacted
version). Treating the outcome standard deviation `sd` as common to both arms, the
standard error is

```math
\\text{se} = \\sigma \\sqrt{\\frac{1 - R^2}{p (1 - p)\\, n}},
```

and the test of ``H_0: \\tau = 0`` uses a t reference with ``n - 2 - k``
degrees of freedom, where `k = n_covariates`. Power is computed exactly from the
noncentral t distribution with noncentrality ``\\tau / \\text{se}``, counting both
rejection tails for a two-sided test, as in R's `pwr::pwr.t.test` and
`pwr.t2n.test`. With `r2 = 0`, `p_treat = 0.5` and `n = 2m` the result reproduces
`pwr.t.test(n = m, d = effect / sd)` exactly. `distribution = :normal` gives the
large-sample z test instead.

The calculation assumes a constant effect (or equal outcome variances in the two
arms), independent units and a correctly specified variance. The `r2` that enters is
the *out-of-sample* share of variance explained, since an in-sample R² overstates the
gain: take it from a pilot or historical data set, e.g. as the cross-fitted R² of a
[`prognostic_score`](@ref). Unequal allocation (`p_treat ≠ 0.5`) always costs precision
when variances are equal. For clustered, blocked or repeated-measures designs use
[`power_cluster`](@ref), [`power_blocked`](@ref) or [`power_did`](@ref); when the
design or estimator is too complex for a closed form, simulate it with
[`declare_design`](@ref) and [`diagnose_design`](@ref).

# Arguments
None positional: the design is described by keywords. Leave exactly one of `effect`,
`n` and `power` as `nothing`; that quantity is solved for.

# Keywords
- `effect`: true difference in means in outcome units; when solved for, the minimum
  detectable effect. Its sign matters only for one-sided alternatives.
- `sd::Real = 1.0`: outcome standard deviation, common to both arms. With `sd = 1`
  the effect is in standard-deviation units (Cohen's d).
- `n`: total number of units in the experiment (continuous when solved for).
- `power`: target power, in `(alpha, 1)`.
- `p_treat::Real = 0.5`: share of units assigned to treatment.
- `alpha::Real = 0.05`: significance level.
- `alternative::Symbol = :two_sided`: `:two_sided`, `:greater` (``\\tau > 0``) or
  `:less`.
- `r2::Real = 0.0`: out-of-sample share of outcome variance explained by the
  adjustment covariates, in `[0, 1)`.
- `n_covariates::Integer = 0`: number of adjustment covariates; each costs one degree
  of freedom of the t reference.
- `distribution::Symbol = :t`: `:t` (exact small-sample power) or `:normal` (z test).

# Returns
- [`PowerAnalysis`](@ref).

# Examples
```julia
using DrSnow
power_means(effect=0.3, n=200)                  # power
power_means(n=200, power=0.8)                   # minimum detectable effect
power_means(effect=0.3, power=0.8, r2=0.4)      # sample size with adjustment
```

# References
- Bloom, H. S. (1995). Minimum detectable effects: A simple way to report the
  statistical power of experimental designs. *Evaluation Review*, 19(5), 547–556.
- Cohen, J. (1988). *Statistical Power Analysis for the Behavioral Sciences* (2nd ed.).
  Lawrence Erlbaum Associates.
- Duflo, E., Glennerster, R., & Kremer, M. (2007). Using randomization in development
  economics research: A toolkit. In T. P. Schultz & J. Strauss (Eds.), *Handbook of
  Development Economics* (Vol. 4, pp. 3895–3962). Elsevier.
- Lin, W. (2013). Agnostic notes on regression adjustments to experimental data:
  Reexamining Freedman's critique. *Annals of Applied Statistics*, 7(1), 295–318.
"""
function power_means(; effect=nothing, sd::Real=1.0, n=nothing, power=nothing,
                     p_treat::Real=0.5, alpha::Real=0.05, alternative::Symbol=:two_sided,
                     r2::Real=0.0, n_covariates::Integer=0, distribution::Symbol=:t)
    _des_pos(sd, "sd"); _des_unit(p_treat, "p_treat"); _des_r2(r2, "r2")
    n_covariates >= 0 || throw(ArgumentError("n_covariates must be non-negative"))
    _des_pos(n, "n")
    k = 2 + n_covariates
    distribution === :t && n !== nothing && n <= k &&
        throw(ArgumentError("n must exceed 2 + n_covariates = $k"))
    sefun = p -> (sd * sqrt((1 - r2) / (p_treat * (1 - p_treat) * p.n)),
                  distribution === :t ? p.n - k : Inf)
    ranges = Dict{Symbol,Tuple}(:n => (distribution === :t ? k + 1e-8 : 1e-8, Inf, true))
    des = "difference in means, completely randomized" *
          (r2 > 0 ? ", covariate-adjusted (R² = $(r2))" : "")
    return _des_linear(des, (effect=effect, n=n), power, :effect, sefun, ranges, alpha,
                       alternative, distribution)
end

# ------------------------------------------------------------------ proportions

"""
    power_proportions(; p1=nothing, p0, n=nothing, power=nothing, p_treat=0.5,
                      alpha=0.05, alternative=:two_sided, method=:arcsine)
        -> PowerAnalysis

Power, detectable treated-arm proportion or total sample size for comparing a binary
outcome between the arms of a completely randomized experiment.

The estimand is the difference in proportions ``p_1 - p_0`` between the treated arm
and the control arm, with `n₁ = p_treat × n` treated and `n₀ = n - n₁` control
units. Because the variance of a proportion depends on its level, power is not a
function of the difference alone, and two standard large-sample approximations are
offered. With `method = :arcsine` (the default) the test is based on Cohen's
variance-stabilizing effect size ``h = 2\\arcsin\\sqrt{p_1} - 2\\arcsin\\sqrt{p_0}``,
with power

```math
\\Phi\\!\\left(h\\sqrt{n_1 n_0 / n} - z_{1-\\alpha/2}\\right)
+ \\Phi\\!\\left(-h\\sqrt{n_1 n_0 / n} - z_{1-\\alpha/2}\\right),
```

which reproduces R's `pwr::pwr.2p2n.test` (`pwr.2p.test` for equal arms). With
`method = :pooled` the test is the two-sample z test with the pooled variance under
the null and the unpooled variance under the alternative (Fleiss, Levin & Paik 2003,
ch. 4); with equal arms it reproduces R's `power.prop.test(strict = TRUE)`.

Both approximations rely on the normal approximation to the binomial and become
unreliable when an arm has few expected events (e.g. `n p₀ < 10`); simulate the
exact test with [`declare_design`](@ref) in that case. When solving for `p1`, the
solution is the detectable proportion above `p0` (below it for
`alternative = :less`). Covariate adjustment is not modelled; for a linear
probability model with adjustment, [`power_means`](@ref) with
`sd = √(p₀(1 - p₀))` and the adjustment `r2` is a reasonable approximation.

# Arguments
None positional: the design is described by keywords. Leave exactly one of `p1`, `n`
and `power` as `nothing`; that quantity is solved for.

# Keywords
- `p1`: outcome proportion in the treated arm, in `(0, 1)`.
- `p0::Real`: outcome proportion in the control arm, in `(0, 1)` (required).
- `n`: total sample size (continuous when solved for).
- `power`: target power, in `(alpha, 1)`.
- `p_treat::Real = 0.5`: share of units assigned to treatment.
- `alpha::Real = 0.05`: significance level.
- `alternative::Symbol = :two_sided`: `:two_sided`, `:greater` or `:less`.
- `method::Symbol = :arcsine`: `:arcsine` (Cohen's h, normal test) or `:pooled`
  (pooled-variance z test).

# Returns
- [`PowerAnalysis`](@ref); `effect` is `p1`, and `parameters` holds `p0` and
  `difference = p1 - p0`. `se` and `mde_multiplier` are `NaN` for `:arcsine`.

# Examples
```julia
using DrSnow
power_proportions(p1=0.35, p0=0.25, n=600)                  # power
power_proportions(p0=0.25, n=600, power=0.8)                # detectable p1
power_proportions(p1=0.35, p0=0.25, power=0.8, method=:pooled)
```

# References
- Cohen, J. (1988). *Statistical Power Analysis for the Behavioral Sciences* (2nd ed.).
  Lawrence Erlbaum Associates.
- Fleiss, J. L., Levin, B., & Paik, M. C. (2003). *Statistical Methods for Rates and
  Proportions* (3rd ed.). Wiley.
"""
function power_proportions(; p1=nothing, p0::Real, n=nothing, power=nothing,
                           p_treat::Real=0.5, alpha::Real=0.05,
                           alternative::Symbol=:two_sided, method::Symbol=:arcsine)
    _des_check_alpha(alpha, alternative, :normal)
    _des_unit(p0, "p0"); _des_unit(p_treat, "p_treat"); _des_pos(n, "n")
    p1 === nothing || _des_unit(p1, "p1")
    method in (:arcsine, :pooled) ||
        throw(ArgumentError("method must be :arcsine or :pooled"))
    q = p_treat
    powfun = function (p)
        n1, n0 = q * p.n, (1 - q) * p.n
        if method === :arcsine
            h = 2 * asin(sqrt(p.p1)) - 2 * asin(sqrt(p0))
            return _des_power_ncp(h * sqrt(n1 * n0 / p.n), Inf, alpha, alternative,
                                  :normal)
        end
        d = p.p1 - p0
        pbar = (n1 * p.p1 + n0 * p0) / p.n
        s0 = sqrt(pbar * (1 - pbar) * (1 / n1 + 1 / n0))
        s1 = sqrt(p.p1 * (1 - p.p1) / n1 + p0 * (1 - p0) / n0)
        if alternative === :two_sided
            z = quantile(Normal(), 1 - alpha / 2)
            return cdf(Normal(), (d - z * s0) / s1) + cdf(Normal(), (-d - z * s0) / s1)
        end
        z = quantile(Normal(), 1 - alpha)
        return alternative === :greater ? cdf(Normal(), (d - z * s0) / s1) :
               cdf(Normal(), (-d - z * s0) / s1)
    end
    sefun = p -> (sqrt(p.p1 * (1 - p.p1) / (q * p.n) + p0 * (1 - p0) / ((1 - q) * p.n)),
                  Inf)
    eps_ = 1e-12
    ranges = Dict{Symbol,Tuple}(:n => (1e-8, Inf, true),
                                :p1 => alternative === :less ? (eps_, p0, false) :
                                       (p0, 1 - eps_, true))
    des = "two proportions (" * (method === :arcsine ? "arcsine h, normal test" :
                                 "pooled z test") * ")"
    r = _des_analysis(des, (p1=p1, n=n), power, powfun, sefun, :p1, ranges, alpha,
                      alternative, :normal)
    params = merge(r.parameters, (p0=float(p0), difference=r.effect - p0))
    se = method === :arcsine ? NaN : r.se
    mult = method === :arcsine ? NaN : r.mde_multiplier
    return PowerAnalysis(r.design, r.solved, r.power, r.effect, params, se, Inf,
                         r.alpha, alternative, :normal, mult, r.note)
end

# ------------------------------------------------------------------ cluster RCTs

function _des_cluster_cv(cv, cluster_sizes)
    cluster_sizes === nothing && return float(cv), nothing
    cv == 0 || throw(ArgumentError("give either `cv` or `cluster_sizes`, not both"))
    s = float.(collect(cluster_sizes))
    (length(s) >= 2 && all(>(0), s)) ||
        throw(ArgumentError("cluster_sizes must hold at least two positive sizes"))
    return std(s; corrected=false) / mean(s), mean(s)
end

"""
    power_cluster(; effect=nothing, sd=1.0, icc, cluster_size=nothing,
                  n_clusters=nothing, power=nothing, p_treat=0.5, cv=0.0,
                  cluster_sizes=nothing, r2_cluster=0.0, r2_individual=0.0,
                  n_cluster_covariates=0, alpha=0.05, alternative=:two_sided,
                  distribution=:t) -> PowerAnalysis

Power, minimum detectable effect, number of clusters or cluster size for a two-level
cluster-randomized trial.

In a cluster-randomized trial whole clusters (schools, villages, clinics) are
assigned to treatment, and the estimand is the individual-level average treatment
effect. Outcomes of individuals in the same cluster are correlated: with intraclass
correlation ``\\rho`` (the share of outcome variance that lies between clusters),
``J`` clusters of mean size ``m``, a share ``p`` of clusters treated and covariates
that explain shares ``R^2_2`` and ``R^2_1`` of the between- and within-cluster
variance, the standard error of the difference in means (equivalently of the
cluster-level or cluster-robust regression estimator) is

```math
\\text{se} = \\sigma \\sqrt{\\frac{\\rho (1 - R^2_2)(1 + \\text{cv}^2)
  + (1 - \\rho)(1 - R^2_1) / m}{p (1 - p)\\, J}},
```

where `cv` is the coefficient of variation of cluster sizes (Raudenbush 1997; Bloom
2006; Eldridge, Ashby & Kerry 2006 for unequal sizes). The corresponding design
effect relative to individual randomization is ``1 + ((1 + \\text{cv}^2) m - 1)\\rho``.
The test uses a t reference with ``J - 2 - k`` degrees of freedom (`k` cluster-level
covariates): the effective sample size is the number of clusters, not individuals.
With `cv = 0` the calculation reproduces `PowerUpR::power.cra2r2` and `mdes.cra2r2`
(Dong & Maynard 2013) exactly.

Two consequences shape cluster designs. First, power is bounded as the cluster size
grows, because the between-cluster term does not shrink with `m`: with a fixed number
of clusters some targets are unattainable at any cluster size (Hemming et al. 2011),
and solving for `cluster_size` then throws. Second, the ICC and the explained
variances are the decisive inputs and are rarely known precisely; take them from
comparable studies or pilot data and report power over a plausible range. With few
clusters, cluster-robust standard errors tend to be too small, and randomization
inference or small-sample corrections are preferable in the analysis.
For designs this formula does not cover (unequal treated shares across strata, three
levels, attrition), simulate with [`declare_design`](@ref).

# Arguments
None positional: the design is described by keywords. Leave exactly one of `effect`,
`n_clusters`, `cluster_size` and `power` as `nothing`; that quantity is solved for.

# Keywords
- `effect`: true average effect in outcome units; the MDE when solved for.
- `sd::Real = 1.0`: total (between plus within) outcome standard deviation.
- `icc::Real`: intraclass correlation in `[0, 1]` (required).
- `cluster_size`: mean number of individuals per cluster.
- `n_clusters`: total number of clusters, treated and control.
- `power`: target power, in `(alpha, 1)`.
- `p_treat::Real = 0.5`: share of clusters assigned to treatment.
- `cv::Real = 0.0`: coefficient of variation of cluster sizes (0 for equal sizes).
- `cluster_sizes = nothing`: the planned cluster sizes themselves, as an alternative to
  `cv` and `cluster_size`; their mean and population coefficient of variation are
  then used.
- `r2_cluster::Real = 0.0`, `r2_individual::Real = 0.0`: shares of the between- and
  within-cluster outcome variance explained by covariates, in `[0, 1)`.
- `n_cluster_covariates::Integer = 0`: number of cluster-level covariates, each
  costing one degree of freedom.
- `alpha::Real = 0.05`, `alternative::Symbol = :two_sided`,
  `distribution::Symbol = :t`: as in [`power_means`](@ref).

# Returns
- [`PowerAnalysis`](@ref); `parameters` also holds `design_effect`, `cv` and
  `n_individuals` (mean cluster size times the number of clusters).

# Examples
```julia
using DrSnow
power_cluster(effect=0.25, icc=0.1, cluster_size=20, n_clusters=40)   # power
power_cluster(effect=0.25, icc=0.1, cluster_size=20, power=0.8)       # clusters
power_cluster(icc=0.05, cluster_sizes=[12, 30, 8, 25, 40, 15], n_clusters=60,
              power=0.8)                                              # MDE
```

# References
- Bloom, H. S. (2006). *The core analytics of randomized experiments for social
  research* (MDRC Working Papers on Research Methodology). MDRC.
- Dong, N., & Maynard, R. (2013). PowerUp!: A tool for calculating minimum detectable
  effect sizes and minimum required sample sizes for experimental and
  quasi-experimental design studies. *Journal of Research on Educational
  Effectiveness*, 6(1), 24–67.
- Eldridge, S. M., Ashby, D., & Kerry, S. (2006). Sample size for cluster randomized
  trials: Effect of coefficient of variation of cluster size and analysis method.
  *International Journal of Epidemiology*, 35(5), 1292–1300.
- Hemming, K., Girling, A. J., Sitch, A. J., Marsh, J., & Lilford, R. J. (2011).
  Sample size calculations for cluster randomised controlled trials with a fixed
  number of clusters. *BMC Medical Research Methodology*, 11, 102.
- Raudenbush, S. W. (1997). Statistical analysis and optimal design for cluster
  randomized trials. *Psychological Methods*, 2(2), 173–185.
"""
function power_cluster(; effect=nothing, sd::Real=1.0, icc::Real, cluster_size=nothing,
                       n_clusters=nothing, power=nothing, p_treat::Real=0.5,
                       cv::Real=0.0, cluster_sizes=nothing, r2_cluster::Real=0.0,
                       r2_individual::Real=0.0, n_cluster_covariates::Integer=0,
                       alpha::Real=0.05, alternative::Symbol=:two_sided,
                       distribution::Symbol=:t)
    _des_pos(sd, "sd"); _des_unit(icc, "icc"; closed=true); _des_unit(p_treat, "p_treat")
    _des_r2(r2_cluster, "r2_cluster"); _des_r2(r2_individual, "r2_individual")
    cv >= 0 || throw(ArgumentError("cv must be non-negative"))
    cvv, msz = _des_cluster_cv(cv, cluster_sizes)
    if msz !== nothing
        cluster_size === nothing ||
            throw(ArgumentError("give either `cluster_size` or `cluster_sizes`"))
        cluster_size = msz
    end
    _des_pos(cluster_size, "cluster_size"); _des_pos(n_clusters, "n_clusters")
    n_cluster_covariates >= 0 ||
        throw(ArgumentError("n_cluster_covariates must be non-negative"))
    k = 2 + n_cluster_covariates
    distribution === :t && n_clusters !== nothing && n_clusters <= k &&
        throw(ArgumentError("n_clusters must exceed 2 + n_cluster_covariates = $k"))
    varfac = p -> icc * (1 - r2_cluster) * (1 + cvv^2) +
                  (1 - icc) * (1 - r2_individual) / p.cluster_size
    sefun = p -> (sd * sqrt(varfac(p) / (p_treat * (1 - p_treat) * p.n_clusters)),
                  distribution === :t ? p.n_clusters - k : Inf)
    ranges = Dict{Symbol,Tuple}(
        :n_clusters => (distribution === :t ? k + 1e-8 : 1e-8, Inf, true),
        :cluster_size => (1e-8, 1e12, true))
    des = "cluster-randomized trial (difference in means, ICC = $(icc))"
    r = _des_linear(des, (effect=effect, n_clusters=n_clusters, cluster_size=cluster_size),
                    power, :effect, sefun, ranges, alpha, alternative, distribution;
                    note=cvv > 0 ? "unequal cluster sizes via the design effect of " *
                                   "Eldridge et al. (2006), cv = $(round(cvv; digits=4))" :
                         "")
    m = r.parameters.cluster_size
    deff = 1 + ((1 + cvv^2) * m - 1) * icc
    params = merge(r.parameters, (icc=float(icc), cv=cvv, design_effect=deff,
                                  n_individuals=m * r.parameters.n_clusters))
    return PowerAnalysis(r.design, r.solved, r.power, r.effect, params, r.se, r.dof,
                         r.alpha, r.alternative, r.distribution, r.mde_multiplier, r.note)
end

# ------------------------------------------------------------------ blocked RCTs

"""
    power_blocked(; effect=nothing, sd=1.0, n_blocks=nothing, block_size, power=nothing,
                  p_treat=0.5, r2=0.0, n_covariates=0, alpha=0.05,
                  alternative=:two_sided, distribution=:t) -> PowerAnalysis

Power, minimum detectable effect or number of blocks for an individually randomized
experiment that is blocked (stratified) or matched in pairs.

Units are grouped into ``J`` blocks of ``m`` similar units, and within each block a
share ``p`` is assigned to treatment completely at random; matched pairs are the case
``m = 2``. The estimand is the average treatment effect, estimated by a regression of
the outcome on the treatment indicator, block fixed effects and (optionally) `k`
covariates. Assuming a constant treatment effect, the standard error is

```math
\\text{se} = \\sigma \\sqrt{\\frac{1 - R^2}{p (1 - p)\\, J m}},
```

with a t reference on ``J(m - 1) - k - 1`` degrees of freedom (``J - 1`` for matched
pairs without covariates), where ``R^2`` is the share of outcome variance explained by
the block indicators and covariates together. This reproduces
`PowerUpR::power.bira2c1` and `mdes.bira2c1` (Dong & Maynard 2013) exactly.

Blocking gains precision exactly to the extent that units within a block have similar
outcomes, so the relevant ``R^2`` is that of the variable the blocks are formed on. For
blocks formed on a prognostic score ([`prognostic_score`](@ref),
[`block_design`](@ref)), the score's out-of-sample R² is a natural planning value
when blocks are tight (small relative to the spread of the score);
[`variance_reduction`](@ref) evaluates the expected gain for a formed design.
Among stratified designs that treat each unit with probability one half, a
matched-pair design that pairs units on a suitable index of the covariates maximizes
the precision of the estimated average effect (Bai 2022). Pairs leave no within-pair
degrees of freedom, however: the usual variance estimators are conservative, and
consistent alternatives use "pairs of pairs" (Bai, Romano & Shaikh 2022). The formula
does not account for effect heterogeneity across blocks or for attrition that breaks
pairs; check such designs by simulation with [`declare_design`](@ref).

# Arguments
None positional: the design is described by keywords. Leave exactly one of `effect`,
`n_blocks` and `power` as `nothing`; that quantity is solved for.

# Keywords
- `effect`: true average effect in outcome units; the MDE when solved for.
- `sd::Real = 1.0`: outcome standard deviation (unconditional).
- `n_blocks`: number of blocks ``J`` (continuous when solved for).
- `block_size::Real`: number of units per block ``m`` (at least 2; required).
- `power`: target power, in `(alpha, 1)`.
- `p_treat::Real = 0.5`: share treated within each block.
- `r2::Real = 0.0`: share of outcome variance explained by the blocks and covariates,
  in `[0, 1)`.
- `n_covariates::Integer = 0`: number of covariates besides the block indicators.
- `alpha::Real = 0.05`, `alternative::Symbol = :two_sided`,
  `distribution::Symbol = :t`: as in [`power_means`](@ref).

# Returns
- [`PowerAnalysis`](@ref); `parameters` also holds `block_size` and `n_units`.

# Examples
```julia
using DrSnow
power_blocked(effect=0.2, n_blocks=100, block_size=2, r2=0.5)   # matched pairs
power_blocked(n_blocks=50, block_size=8, r2=0.3, power=0.8)     # MDE
power_blocked(effect=0.2, block_size=4, r2=0.4, power=0.8)      # blocks needed
```

# References
- Bai, Y. (2022). Optimality of matched-pair designs in randomized controlled trials.
  *American Economic Review*, 112(12), 3911–3940.
- Bai, Y., Romano, J. P., & Shaikh, A. M. (2022). Inference in experiments with
  matched pairs. *Journal of the American Statistical Association*, 117(540),
  1726–1737.
- Bloom, H. S. (2006). *The core analytics of randomized experiments for social
  research* (MDRC Working Papers on Research Methodology). MDRC.
- Dong, N., & Maynard, R. (2013). PowerUp!: A tool for calculating minimum detectable
  effect sizes and minimum required sample sizes for experimental and
  quasi-experimental design studies. *Journal of Research on Educational
  Effectiveness*, 6(1), 24–67.
"""
function power_blocked(; effect=nothing, sd::Real=1.0, n_blocks=nothing,
                       block_size::Real, power=nothing, p_treat::Real=0.5, r2::Real=0.0,
                       n_covariates::Integer=0, alpha::Real=0.05,
                       alternative::Symbol=:two_sided, distribution::Symbol=:t)
    _des_pos(sd, "sd"); _des_unit(p_treat, "p_treat"); _des_r2(r2, "r2")
    block_size >= 2 || throw(ArgumentError("block_size must be at least 2"))
    _des_pos(n_blocks, "n_blocks")
    n_covariates >= 0 || throw(ArgumentError("n_covariates must be non-negative"))
    dfun = J -> J * (block_size - 1) - n_covariates - 1
    distribution === :t && n_blocks !== nothing && dfun(n_blocks) <= 0 &&
        throw(ArgumentError("too few blocks: residual degrees of freedom " *
                            "$(dfun(n_blocks)) ≤ 0"))
    sefun = p -> (sd * sqrt((1 - r2) / (p_treat * (1 - p_treat) * p.n_blocks *
                                        block_size)),
                  distribution === :t ? dfun(p.n_blocks) : Inf)
    jmin = (n_covariates + 1) / (block_size - 1)
    ranges = Dict{Symbol,Tuple}(:n_blocks => (distribution === :t ? jmin + 1e-8 : 1e-8,
                                              Inf, true))
    des = "blocked randomized experiment ($(block_size) units per block, block fixed " *
          "effects" * (r2 > 0 ? ", R² = $(r2)" : "") * ")"
    r = _des_linear(des, (effect=effect, n_blocks=n_blocks), power, :effect, sefun,
                    ranges, alpha, alternative, distribution)
    params = merge(r.parameters, (block_size=float(block_size),
                                  n_units=r.parameters.n_blocks * block_size))
    return PowerAnalysis(r.design, r.solved, r.power, r.effect, params, r.se, r.dof,
                         r.alpha, r.alternative, r.distribution, r.mde_multiplier, r.note)
end

# ------------------------------------------------------------------ DiD / ANCOVA

"""
Variance factor `v` such that Var(estimator) = sd² v / (p (1 - p) n) for the POST,
DiD and ANCOVA estimators with `m` pre and `r` post periods and within-unit
correlation matrix `C` of the `m + r` periods (pre first).
"""
function _des_panel_factor(estimator::Symbol, C::AbstractMatrix, m::Int, r::Int)
    a = vcat(zeros(m), fill(1 / r, r))            # post mean
    b = vcat(fill(1 / m, m), zeros(r))            # pre mean
    vpost = dot(a, C * a)
    estimator === :post && return vpost
    vpre = dot(b, C * b)
    cpp = dot(a, C * b)
    estimator === :did && return vpost + vpre - 2 * cpp
    return vpost - cpp^2 / vpre                    # :ancova
end

function _des_panel_corr(rho, corr, m::Int, r::Int)
    T = m + r
    if corr !== nothing
        rho === nothing || rho == 0 ||
            throw(ArgumentError("give either `rho` or `corr`, not both"))
        C = Matrix{Float64}(corr)
        size(C) == (T, T) ||
            throw(DimensionMismatch("corr must be $(T)×$(T) (pre then post periods)"))
        (issymmetric(C) && all(≈(1.0), diag(C))) ||
            throw(ArgumentError("corr must be a symmetric correlation matrix"))
        isposdef(Symmetric(C) + 1e-12I) ||
            throw(ArgumentError("corr must be positive semi-definite"))
        return C, "user correlation matrix"
    end
    ρ = rho === nothing ? 0.0 : float(rho)
    -1 / (T - 1) <= ρ <= 1 || throw(ArgumentError("rho must be in [-1/(T-1), 1]"))
    C = fill(ρ, T, T)
    C[diagind(C)] .= 1.0
    return C, "equicorrelated periods (rho = $(ρ))"
end

"""
    power_did(; effect=nothing, sd=1.0, n=nothing, power=nothing, pre_periods=1,
              post_periods=1, rho=0.0, corr=nothing, estimator=:ancova, p_treat=0.5,
              alpha=0.05, alternative=:two_sided, distribution=:t) -> PowerAnalysis

Power, minimum detectable effect or number of units for a randomized experiment with
repeated outcome measurements before and after treatment.

Each of `n` units is observed in ``m`` = `pre_periods` baseline rounds and
``r`` = `post_periods` follow-up rounds, and a share ``p`` of units is treated after
the baseline. The estimand is the average effect on the mean of the post-period
outcomes. Three estimators are compared (McKenzie 2012; Frison & Pocock 1992):
`:post`, the difference in post-period means; `:did`, the difference in
post-minus-pre changes (equivalently unit and period fixed effects); and `:ancova`,
the regression of the post-period mean on treatment and the pre-period mean. The same
formulas apply to a difference-in-differences comparison when treatment is as good
as random conditional on unit fixed effects (parallel trends).

With per-period outcome standard deviation ``\\sigma``, each estimator has variance
``\\sigma^2 v / (p (1 - p) n)``. Under McKenzie's assumption of a constant correlation
``\\rho`` between any two periods of the same unit, the variance factor is

```math
v_{\\text{POST}} = \\frac{1 + (r - 1)\\rho}{r}, \\qquad
v_{\\text{DiD}} = \\frac{1 + (m - 1)\\rho}{m} + \\frac{1 + (r - 1)\\rho}{r} - 2\\rho,
\\qquad
v_{\\text{ANCOVA}} = \\frac{1 + (r - 1)\\rho}{r} - \\frac{m \\rho^2}{1 + (m - 1)\\rho}.
```

Any other serial correlation, e.g. AR(1) with ``\\text{corr}(Y_s, Y_t) = \\phi^{|s-t|}``,
is passed as the full correlation matrix `corr` of the ``m + r`` periods (pre-periods
first), and ``v`` is computed from it. The reference distribution is t with
``n - 2`` degrees of freedom (``n - 3`` for ANCOVA).

ANCOVA is never less efficient than POST or DiD, and with one baseline and one
follow-up round DiD beats POST only when the correlation exceeds one half. For noisy,
weakly autocorrelated outcomes (profits, incomes, expenditures), several follow-up
rounds average out noise and can give more power than a single baseline and
follow-up with the same budget (McKenzie 2012). The main practical risk is
the serial-correlation input: Burlig, Preonas & Woerman (2020) extend these formulas to
arbitrary serial correlation and show that ignoring non-constant correlation yields
incorrectly powered experiments. Passing `corr`, e.g. estimated from pre-existing panel
data on the same outcome, follows their approach under a common per-period variance.
The formulas assume a constant effect, balanced panels and independent units;
clustered panels or attrition call for simulation with [`declare_design`](@ref).

# Arguments
None positional: the design is described by keywords. Leave exactly one of `effect`,
`n` and `power` as `nothing`; that quantity is solved for.

# Keywords
- `effect`: true effect on the post-period mean in outcome units; the MDE when solved
  for.
- `sd::Real = 1.0`: per-period outcome standard deviation.
- `n`: number of units (continuous when solved for).
- `power`: target power, in `(alpha, 1)`.
- `pre_periods::Integer = 1`, `post_periods::Integer = 1`: numbers of rounds before
  and after treatment (`pre_periods` may be 0 only for `:post`).
- `rho = 0.0`: common correlation between any two periods of a unit, in
  `[-1/(m+r-1), 1]`.
- `corr = nothing`: full ``(m + r) \\times (m + r)`` correlation matrix of the periods,
  pre-periods first; replaces `rho`.
- `estimator::Symbol = :ancova`: `:ancova`, `:did` or `:post`.
- `p_treat::Real = 0.5`: share of units treated.
- `alpha::Real = 0.05`, `alternative::Symbol = :two_sided`,
  `distribution::Symbol = :t`: as in [`power_means`](@ref).

# Returns
- [`PowerAnalysis`](@ref); `parameters.variance_factor` is ``v``, and `parameters`
  also holds `pre_periods` and `post_periods`.

# Examples
```julia
using DrSnow
power_did(effect=0.2, n=400, pre_periods=1, post_periods=3, rho=0.5)
power_did(effect=0.2, power=0.8, rho=0.3, estimator=:did)      # units needed
φ = 0.7
C = [φ^abs(s - t) for s in 1:6, t in 1:6]                       # AR(1) correlation
power_did(effect=0.2, n=300, pre_periods=3, post_periods=3, corr=C)
```

# References
- Burlig, F., Preonas, L., & Woerman, M. (2020). Panel data and experimental design.
  *Journal of Development Economics*, 144, 102458.
- Frison, L., & Pocock, S. J. (1992). Repeated measures in clinical trials: Analysis
  using mean summary statistics and its implications for design. *Statistics in
  Medicine*, 11(13), 1685–1704.
- McKenzie, D. (2012). Beyond baseline and follow-up: The case for more T in
  experiments. *Journal of Development Economics*, 99(2), 210–221.
"""
function power_did(; effect=nothing, sd::Real=1.0, n=nothing, power=nothing,
                   pre_periods::Integer=1, post_periods::Integer=1, rho=0.0, corr=nothing,
                   estimator::Symbol=:ancova, p_treat::Real=0.5, alpha::Real=0.05,
                   alternative::Symbol=:two_sided, distribution::Symbol=:t)
    _des_pos(sd, "sd"); _des_unit(p_treat, "p_treat"); _des_pos(n, "n")
    estimator in (:ancova, :did, :post) ||
        throw(ArgumentError("estimator must be :ancova, :did or :post"))
    (pre_periods >= 1 || estimator === :post) ||
        throw(ArgumentError("pre_periods must be at least 1 for $estimator"))
    pre_periods >= 0 && post_periods >= 1 ||
        throw(ArgumentError("need pre_periods ≥ 0 and post_periods ≥ 1"))
    m, r = Int(pre_periods), Int(post_periods)
    C, cdesc = _des_panel_corr(rho, corr, m, r)
    v = _des_panel_factor(estimator, C, m, r)
    v > 0 || throw(ArgumentError("the variance factor is not positive ($v); the " *
                                 "correlation structure makes the estimator exact"))
    k = estimator === :ancova ? 3 : 2
    distribution === :t && n !== nothing && n <= k &&
        throw(ArgumentError("n must exceed $k"))
    sefun = p -> (sd * sqrt(v / (p_treat * (1 - p_treat) * p.n)),
                  distribution === :t ? p.n - k : Inf)
    ranges = Dict{Symbol,Tuple}(:n => (distribution === :t ? k + 1e-8 : 1e-8, Inf, true))
    des = "repeated measurements, $(uppercase(string(estimator))) estimator " *
          "($m pre, $r post periods; $cdesc)"
    r_ = _des_linear(des, (effect=effect, n=n), power, :effect, sefun, ranges, alpha,
                     alternative, distribution)
    params = merge(r_.parameters, (variance_factor=v, pre_periods=m, post_periods=r))
    return PowerAnalysis(r_.design, r_.solved, r_.power, r_.effect, params, r_.se,
                         r_.dof, r_.alpha, r_.alternative, r_.distribution,
                         r_.mde_multiplier, r_.note)
end

# ------------------------------------------------------------------ IV / encouragement

"""
    power_iv(; effect=nothing, compliance=nothing, sd=1.0, n=nothing, power=nothing,
             p_treat=0.5, r2=0.0, n_covariates=0, alpha=0.05,
             alternative=:two_sided, distribution=:t) -> PowerAnalysis

Power, minimum detectable local average treatment effect, sample size or required
compliance rate for an encouragement design with a randomized binary instrument.

A share `p_treat` of `n` units is randomly encouraged (``Z = 1``), and take-up of the
treatment ``D`` is imperfect. Under the assumptions of Angrist, Imbens & Rubin (1996),
namely random assignment of ``Z``, the exclusion restriction and monotonicity (no
defiers), the Wald (2SLS) estimand is the local average treatment effect (LATE) of
compliers, and the intention-to-treat (ITT) effect of encouragement on the outcome
equals

```math
\\text{ITT} = \\pi \\cdot \\text{LATE}, \\qquad
\\pi = P(D = 1 \\mid Z = 1) - P(D = 1 \\mid Z = 0),
```

where ``\\pi`` (`compliance`) is the first-stage share of compliers.

Under the null of a zero LATE the Wald test is asymptotically equivalent to the test
of the reduced-form ITT effect, whose standard error is
``\\sigma \\sqrt{(1 - R^2) / (p (1 - p) n)}``. The calculation therefore runs the
ITT calculation of [`power_means`](@ref) at effect ``\\pi \\cdot \\text{LATE}``, so the MDE
of the LATE is the ITT MDE divided by ``\\pi`` and the required sample size grows with
``1 / \\pi^2``: halving compliance quadruples the sample (Duflo, Glennerster & Kremer
2007). The reported `se` is the delta-method standard error of the LATE at
``\\text{LATE} = 0``, namely ``\\text{se}_{\\text{ITT}} / \\pi``.

This is a normal-theory approximation. It ignores sampling noise in the first stage
and effect heterogeneity under the alternative, both of which lower power, and it
says nothing about weak-instrument problems, which arise when the compliance rate is
small relative to its sampling error. With low compliance, check the design by
simulation with [`declare_design`](@ref) and the IV estimator that will be used in the
analysis. The exclusion restriction and monotonicity cannot be checked from the power
calculation; they are design assumptions.

# Arguments
None positional: the design is described by keywords. Leave exactly one of `effect`,
`compliance`, `n` and `power` as `nothing`; that quantity is solved for.

# Keywords
- `effect`: the LATE in outcome units; the MDE of the LATE when solved for.
- `compliance`: first-stage difference in take-up, in `(0, 1]`.
- `sd::Real = 1.0`: outcome standard deviation.
- `n`: total number of units (continuous when solved for).
- `power`: target power, in `(alpha, 1)`.
- `p_treat::Real = 0.5`: share of units encouraged.
- `r2::Real = 0.0`, `n_covariates::Integer = 0`: covariate adjustment of the reduced
  form, as in [`power_means`](@ref).
- `alpha::Real = 0.05`, `alternative::Symbol = :two_sided`,
  `distribution::Symbol = :t`: as in [`power_means`](@ref).

# Returns
- [`PowerAnalysis`](@ref); `parameters.itt` is the implied ITT effect.

# Examples
```julia
using DrSnow
power_iv(effect=0.5, compliance=0.4, n=1000)          # power
power_iv(compliance=0.3, n=2000, power=0.8)           # MDE of the LATE
power_iv(effect=0.5, n=1000, power=0.8)               # compliance needed
```

# References
- Angrist, J. D., Imbens, G. W., & Rubin, D. B. (1996). Identification of causal
  effects using instrumental variables. *Journal of the American Statistical
  Association*, 91(434), 444–455.
- Duflo, E., Glennerster, R., & Kremer, M. (2007). Using randomization in development
  economics research: A toolkit. In T. P. Schultz & J. Strauss (Eds.), *Handbook of
  Development Economics* (Vol. 4, pp. 3895–3962). Elsevier.
"""
function power_iv(; effect=nothing, compliance=nothing, sd::Real=1.0, n=nothing,
                  power=nothing, p_treat::Real=0.5, r2::Real=0.0,
                  n_covariates::Integer=0, alpha::Real=0.05,
                  alternative::Symbol=:two_sided, distribution::Symbol=:t)
    _des_pos(sd, "sd"); _des_unit(p_treat, "p_treat"); _des_r2(r2, "r2")
    _des_pos(n, "n")
    compliance === nothing || 0 < compliance <= 1 ||
        throw(ArgumentError("compliance must be in (0, 1], got $compliance"))
    k = 2 + n_covariates
    distribution === :t && n !== nothing && n <= k &&
        throw(ArgumentError("n must exceed 2 + n_covariates = $k"))
    sefun = p -> (sd * sqrt((1 - r2) / (p_treat * (1 - p_treat) * p.n)) / p.compliance,
                  distribution === :t ? p.n - k : Inf)
    ranges = Dict{Symbol,Tuple}(:n => (distribution === :t ? k + 1e-8 : 1e-8, Inf, true),
                                :compliance => (1e-10, 1.0, true))
    des = "encouragement design / randomized instrument (Wald–2SLS test of the LATE)"
    r = _des_linear(des, (effect=effect, compliance=compliance, n=n), power, :effect,
                    sefun, ranges, alpha, alternative, distribution;
                    note="normal-theory approximation (reduced-form test); ignores " *
                         "first-stage noise and effect heterogeneity under the " *
                         "alternative")
    params = merge(r.parameters, (itt=r.effect * r.parameters.compliance,))
    return PowerAnalysis(r.design, r.solved, r.power, r.effect, params, r.se, r.dof,
                         r.alpha, r.alternative, r.distribution, r.mde_multiplier, r.note)
end
