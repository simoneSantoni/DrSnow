# Honest (bias-aware) inference in RD designs under a bound on the second derivative
# (Armstrong & Kolesár 2018, 2020; Kolesár & Rothe 2018; Noack & Rothe 2024), the
# equivalent of the R package RDHonest.

"""
    RDHonestEstimate <: CausalEstimate

Result of [`rd_honest`](@ref): a local linear RD estimate with an honest (bias-aware)
confidence interval.

`coef` is the local linear estimate ``\\hat\\tau``, without bias correction, and
`stderror` is its standard error. `confint(r; level)` is the **honest** interval
``\\hat\\tau \\pm \\text{cv}_{1-\\alpha}(\\bar b / \\text{se})\\,\\text{se}``. Here
``\\bar b`` is the largest bias of ``\\hat\\tau`` over the smoothness class indexed by `M`,
and ``\\text{cv}_{1-\\alpha}(B)`` is the ``1-\\alpha`` quantile of ``|N(B, 1)|``
(Armstrong & Kolesár 2020). This is not the interval
``\\hat\\tau \\pm z_{1-\\alpha/2}\\,\\text{se}``. It is wider, and its coverage is at
least ``1-\\alpha`` for every regression function in the class, not only at the true
one. That guarantee holds for the class defined by the value of `M` used. When `M` came
from the rule of thumb (`M_rule_of_thumb = true`), the interval is honest only if the
rule of thumb happens to bound the true curvature (see [`rd_honest`](@ref)). `pvalues`
returns the matching honest p-values. The `t` column of `coeftable` is simply
estimate / se.

# Fields
- `design::Symbol`: `:sharp`, `:fuzzy`, or `:point` (value of the regression function at
  a point).
- `estimate`, `se`: point estimate and standard error.
- `max_bias`: worst-case bias ``\\bar b`` over the smoothness class.
- `cv`: critical value at `level`; `conf_low`, `conf_high`: honest two-sided interval;
  `conf_low_onesided`, `conf_high_onesided`: honest one-sided bounds.
- `pvalue`: honest p-value for ``H_0: \\tau = 0``.
- `level::Float64`: confidence level.
- `M`: smoothness bound behind the worst-case bias. For fuzzy designs it is
  ``(M_Y + |\\hat\\theta| M_D) / |\\hat\\tau_D|``. `M_rf` and `M_fs` are the bounds for
  the outcome (reduced form) and treatment (first stage) regressions (`M_fs = 0` in
  sharp designs).
- `M_rule_of_thumb::Bool`: `true` when `M` came from the Armstrong and Kolesár (2020)
  rule of thumb rather than from the user.
- `first_stage`, `reduced_form`: jumps in treatment and outcome (fuzzy designs;
  `nothing` otherwise).
- `V::Vector{Float64}`: variance components: `[Var(τ̂_Y)]` (sharp and point designs) or
  `[Var(τ̂_Y), Cov, Cov, Var(τ̂_D)]` (fuzzy designs).
- `bias_constant`: worst-case bias per unit of `M` (bias = `M * bias_constant`).
- `bandwidth`, `bandwidth_selected::Bool`, `opt_criterion::Symbol`: bandwidth used,
  whether it was selected, and the criterion used to select it.
- `eff_obs`: effective number of observations (relative to a uniform kernel);
  `leverage`: maximal leverage ``\\max_i k_i^2 / \\sum_i k_i^2`` of the estimation
  weights.
- `kernel`, `sclass` (`:holder` or `:taylor`), `vce` (`:nn` or `:ehw`), `J`, `cutoff`,
  `T0`: settings used.
- `n_left`, `n_right`: observations on each side; `n_h_left`, `n_h_right`: those with
  positive kernel weight.
- `covariates::Vector{Symbol}`: covariates used, after dropping collinear ones;
  `n_clusters::Int`: clusters in the estimation sample (0 when unclustered).
"""
struct RDHonestEstimate <: CausalEstimate
    design::Symbol
    estimate::Float64
    se::Float64
    max_bias::Float64
    cv::Float64
    conf_low::Float64
    conf_high::Float64
    conf_low_onesided::Float64
    conf_high_onesided::Float64
    pvalue::Float64
    level::Float64
    M::Float64
    M_rf::Float64
    M_fs::Float64
    M_rule_of_thumb::Bool
    first_stage::Union{Nothing,Float64}
    reduced_form::Union{Nothing,Float64}
    V::Vector{Float64}
    bias_constant::Float64
    bandwidth::Float64
    bandwidth_selected::Bool
    opt_criterion::Symbol
    eff_obs::Float64
    leverage::Float64
    kernel::Symbol
    sclass::Symbol
    vce::Symbol
    J::Int
    cutoff::Float64
    T0::Float64
    n_left::Int
    n_right::Int
    n_h_left::Int
    n_h_right::Int
    covariates::Vector{Symbol}
    n_clusters::Int
end

StatsAPI.coef(r::RDHonestEstimate) = [r.estimate]
StatsAPI.vcov(r::RDHonestEstimate) = fill(r.se^2, 1, 1)
StatsAPI.coefnames(r::RDHonestEstimate) =
    [r.design === :point ? "Conditional mean at x0 (local linear)" :
     "RD effect (local linear)"]
StatsAPI.nobs(r::RDHonestEstimate) = r.n_left + r.n_right
pvalues(r::RDHonestEstimate) = [r.pvalue]

function StatsAPI.confint(r::RDHonestEstimate; level::Real=r.level)
    (0 < level < 1) || throw(ArgumentError("level must be in (0, 1), got $level"))
    cv = _rd_cvb(r.max_bias / r.se, 1 - level)
    return [r.estimate - cv * r.se r.estimate + cv * r.se]
end

function estimand(r::RDHonestEstimate)
    r.design === :sharp && return "ATE at the cutoff (sharp RD)"
    r.design === :fuzzy && return "LATE for compliers at the cutoff (fuzzy RD)"
    return "E[Y | X = x0]"
end

method_name(r::RDHonestEstimate) =
    "Honest local linear RD ($(r.kernel) kernel, $(r.sclass == :holder ? "Hölder" :
                                                   "Taylor") class)"

function show_details(io::IO, r::RDHonestEstimate)
    println(io)
    lv = round(Int, 100 * r.level)
    @printf(io, "Honest %d%% CI: [%.4f, %.4f] (maximum bias %.4g, critical value %.4f)\n",
            lv, r.conf_low, r.conf_high, r.max_bias, r.cv)
    @printf(io, "One-sided %d%% bounds: (-Inf, %.4f], [%.4f, Inf); honest p-value %.4g\n",
            lv, r.conf_high_onesided, r.conf_low_onesided, r.pvalue)
    src = r.bandwidth_selected ? "optimal for $(uppercase(string(r.opt_criterion)))" :
          "supplied"
    @printf(io, "Bandwidth %.4g (%s); ", r.bandwidth, src)
    @printf(io, "effective observations %.1f; maximal leverage %.3g\n", r.eff_obs,
            r.leverage)
    if r.design === :fuzzy
        @printf(io, "First stage %.4f; M (outcome) = %.4g, M (first stage) = %.4g%s\n",
                r.first_stage, r.M_rf, r.M_fs, r.M_rule_of_thumb ? " (rule of thumb)" : "")
    else
        @printf(io, "Smoothness bound M = %.4g%s\n", r.M,
                r.M_rule_of_thumb ? " (rule of thumb)" : "")
    end
    isempty(r.covariates) || println(io, "Covariates: ", join(r.covariates, ", "))
    r.n_clusters > 0 && println(io, "Clusters: ", r.n_clusters)
    println(io, "Note: the interval is bias-aware (not estimate ± z·se); coeftable's ",
            "t column is estimate / se.")
end

function _rd_h_sclass(s)
    v = lowercase(string(s))
    v in ("h", "holder", "hölder") && return :holder
    v in ("t", "taylor") && return :taylor
    throw(ArgumentError("sclass must be :holder or :taylor (got $(repr(s)))"))
end

function _rd_h_criterion(s)
    v = Symbol(lowercase(string(s)))
    v in (:mse, :flci, :oci) && return v
    throw(ArgumentError("opt_criterion must be :mse, :flci or :oci (got $(repr(s)))"))
end

# Build the internal data object from a DataFrame.
function _rd_h_build(data, outcome, running; cutoff, treatment, covariates, cluster,
                     weights, class::Symbol, context::String)
    E = _rd_extract(data, outcome, running; treatment, covariates, cluster, weights,
                    context)
    n = length(E.x)
    n > 0 || throw(ArgumentError("$context: no complete observations"))
    if E.W !== nothing
        all(>(0), E.W) || throw(ArgumentError("$context: weights must be positive"))
    end
    # Cluster codes from the sorted distinct identifiers.
    cl_all = nothing
    if E.C !== nothing
        keys_ = unique(E.C)
        try
            sort!(keys_)
        catch
        end
        idx = Dict(k => i for (i, k) in enumerate(keys_))
        cl_all = [idx[c] for c in E.C]
    end
    # Sort by the running variable, breaking ties by the other columns, so that the
    # floating-point summation order (and hence the result) does not depend on the
    # order of the rows in `data`.
    K = hcat(E.x, E.y)
    E.T === nothing || (K = hcat(K, E.T))
    E.W === nothing || (K = hcat(K, E.W))
    cl_all === nothing || (K = hcat(K, Float64.(cl_all)))
    E.Z === nothing || (K = hcat(K, reshape(E.Z, n, :)))
    ord = sortperm(collect(eachrow(K)); alg=MergeSort)
    X = (E.x[ord] .- Float64(cutoff)) .+ 0.0   # `+ 0.0` maps -0.0 to 0.0 (as R's unique)
    Y = reshape(E.y[ord], :, 1)
    E.T === nothing || (Y = hcat(Y, E.T[ord]))
    covs = E.Z === nothing ? nothing : Matrix{Float64}(reshape(E.Z, n, :)[ord, :])
    cl = cl_all === nothing ? nothing : cl_all[ord]
    w = E.W === nothing ? ones(n) : E.W[ord]
    d = _RDHonestData(class, X, Y, nothing, w, X .>= 0, X .< 0, covs, cl, nothing,
                      nothing)
    return d, E
end

"""
    rd_honest(data, outcome, running; cutoff=0.0, treatment=nothing, M=nothing,
              kernel=:triangular, h=nothing, opt_criterion=:mse, beta=0.8,
              sclass=:holder, vce=nothing, J=3, cluster=nothing, weights=nothing,
              covariates=Symbol[], T0=0.0, point_inference=false,
              level=0.95) -> RDHonestEstimate

Honest (bias-aware) confidence intervals for sharp and fuzzy regression discontinuity
designs, and for the value of a regression function at a point, based on local linear
regression. Equivalent to `RDHonest` from the R package of the same name.

The estimand is the same as for [`rd_estimate`](@ref): the jump
``\\tau = \\lim_{x\\downarrow c} E[Y\\mid X=x] - \\lim_{x\\uparrow c} E[Y\\mid X=x]``,
or the ratio of outcome and take-up jumps in a fuzzy design. The approach to smoothing
bias is different. Instead of estimating the bias and correcting for it, Armstrong and
Kolesár (2018, 2020) assume that the regression function lies in a class ``\\mathcal
F(M)`` with bounded curvature on each side of the cutoff. The Hölder class
(`sclass = :holder`, the default) requires ``|f''(x)| \\le M`` everywhere. The Taylor
class (`:taylor`) only bounds the deviation from a linear approximation at the cutoff by
``M x^2/2``. The local linear estimator ``\\hat\\tau = \\sum_i k_i Y_i`` is linear in
the outcomes, so its largest bias over the class can be computed exactly. In the Hölder
class it is
```math
\\bar b = \\frac{M}{2}\\,\\Big|\\sum_{x_i < c} k_i (x_i - c)^2
    - \\sum_{x_i \\ge c} k_i (x_i - c)^2\\Big|.
```
The interval ``\\hat\\tau \\pm \\text{cv}_{1-\\alpha}(\\bar b/\\text{se})\\,\\text{se}``,
with ``\\text{cv}_{1-\\alpha}(B)`` the ``1-\\alpha`` quantile of ``|N(B, 1)|``, then has
coverage of at least ``1-\\alpha`` uniformly over ``\\mathcal F(M)``, up to the normal
approximation of the estimator. Its validity does not require the bandwidth to shrink
with the sample size. It therefore applies with any bandwidth and with a discrete
running variable, where the continuity-based asymptotics of `rd_estimate` break down
(Kolesár & Rothe 2018). Armstrong and Kolesár (2018, 2020) show that such fixed-length
intervals are highly efficient within the class.

**The smoothness bound `M` is the key input, and the data cannot deliver it.** Armstrong
and Kolesár (2018) show that no confidence interval can adapt to the unknown smoothness
of the regression function. An interval that is shorter when the function happens to be
smooth cannot keep coverage over the whole class. Armstrong and Kolesár (2020)
therefore advise choosing `M` a priori, from knowledge of the application. When `M` is
not given, this function uses their rule of thumb, the largest absolute second
derivative of global quartic fits on each side of the cutoff, as `RDHonest` does. With a
data-driven `M` the interval is **not honest in the formal sense**. It is honest for
the class ``\\mathcal F(\\hat M)``, which may not contain the true function, and a
rule-of-thumb `M` can understate curvature near the cutoff. Report results for a range
of `M` values. For a discrete running variable, [`rd_smoothness_bound`](@ref) gives a
data-based *lower* bound on `M`, a useful check that a chosen `M` is not too small.

When `h` is not given, the bandwidth minimises the worst-case MSE (`:mse`), the length
of the honest two-sided interval (`:flci`), or the `beta` quantile of excess length of
one-sided intervals (`:oci`). It is computed from preliminary homoskedastic variance
estimates on each side, obtained with a local linear fit at the Imbens and Kalyanaraman
(2012) bandwidth. The package checks coverage by Monte Carlo at the least favourable
function of the Hölder class, with `M` known. At 300 replications of ``n = 500`` the
honest 95% interval covered 0.92–0.95 across bandwidth rules and continuous or discrete
running variables, while ``\\hat\\tau \\pm 1.96\\,\\text{se}`` covered 0.90–0.92 (see
the Validation page).

**Fuzzy designs** (`treatment` given): `M = (M_Y, M_D)` bounds the curvature of the
outcome and take-up regressions. The estimate is the ratio of the jumps. The interval
uses the delta-method standard error and the linearised worst-case bias
``(M_Y + |\\hat\\theta| M_D)\\,\\bar b_1 / |\\hat\\tau_D|``, as in `RDHonest` (Armstrong &
Kolesár 2020). This interval requires a strong first stage. For inference that is
robust to a weak first stage, pass the result to [`rd_honest_ar_confidence_set`](@ref)
(Noack & Rothe 2024). The bandwidth is chosen with a preliminary estimate `T0` of the
effect (default 0). Re-running with `T0` set to the estimate, as `RDHonest` recommends,
aligns the bandwidth with the estimand.

**Covariates** enter linearly, as in `RDHonest`. When `M` or `h` is not supplied, a
bandwidth is first selected without covariates, and the outcome is adjusted with the
covariate coefficients from that local fit. `M` and the bandwidth are then computed on
the adjusted outcome, and the final estimate is the local linear regression with the
covariates.

In practice, use `rd_honest` when the running variable is discrete or coarse, when the
local sample is small, or when you want a transparent statement of the smoothness
assumption behind the interval. Report `M` (and whether it came from the rule of
thumb), the bandwidth, the maximum bias and the honest interval. Warnings about large
leverage signal that a few observations dominate the estimate. Imbens and Wager (2019)
propose a related minimax approach that optimises the estimation weights directly
instead of using a local linear fit; it is not implemented here.

# Arguments
- `data::AbstractDataFrame`: one row per unit. Rows with a missing value in any used
  column are dropped.
- `outcome::Symbol`: outcome column.
- `running::Symbol`: running variable. Units with `running ≥ cutoff` are treated.

# Keywords
- `cutoff::Real=0.0`: the RD cutoff, or the point ``x_0`` when
  `point_inference = true`.
- `treatment::Union{Nothing,Symbol}=nothing`: treatment take-up. Supplying it makes the
  design fuzzy.
- `M=nothing`: bound on the second derivative, as a number, or a pair `(M_Y, M_D)` in
  fuzzy designs, in units of the outcome per squared unit of the running variable.
  `nothing` uses the Armstrong and Kolesár (2020) rule of thumb.
- `kernel=:triangular`: `:triangular`, `:epanechnikov` or `:uniform`.
- `h::Union{Nothing,Real}=nothing`: bandwidth. When `nothing` it is optimal for
  `opt_criterion`.
- `opt_criterion=:mse`: `:mse`, `:flci` or `:oci`.
- `beta::Real=0.8`: quantile of excess length used by `:oci`.
- `sclass=:holder`: `:holder` or `:taylor` smoothness class.
- `vce=nothing`: `:nn` (nearest-neighbour variance, the default without clusters) or
  `:ehw` (Eicker–Huber–White with local linear residuals, the default and only option
  with `cluster`).
- `J::Integer=3`: number of nearest neighbours for `vce = :nn`.
- `cluster::Union{Nothing,Symbol}=nothing`: cluster identifier for cluster-robust
  variances. The preliminary variance used to select the bandwidth then includes a
  Moulton-type intra-cluster correlation.
- `weights::Union{Nothing,Symbol}=nothing`: positive observation weights, for example
  cell sizes of data aggregated by value of the running variable.
- `covariates::Vector{Symbol}=Symbol[]`: predetermined covariates (not allowed with
  `point_inference`).
- `T0::Real=0.0`: preliminary estimate of the fuzzy effect, used for bandwidth
  selection.
- `point_inference::Bool=false`: estimate ``E[Y \\mid X = x_0]`` at ``x_0`` = `cutoff`
  instead of the RD jump.
- `level::Real=0.95`: confidence level (`alpha = 1 - level` in `RDHonest`).

# Returns
- [`RDHonestEstimate`](@ref). `confint` gives the honest interval, and `r.max_bias`,
  `r.M` and `r.M_rule_of_thumb` document the assumption behind it.

# Examples
```julia
using DrSnow, CSV, DataFrames
lee = CSV.read(pkgdir(DrSnow, "test", "validation", "rd", "lee08.csv"), DataFrame)
r = rd_honest(lee, :voteshare, :margin; M=0.1, h=10, kernel=:uniform)
confint(r)                                          # honest 95% interval
r = rd_honest(lee, :voteshare, :margin)             # rule-of-thumb M, MSE-optimal h
r.M, r.M_rule_of_thumb
[confint(rd_honest(lee, :voteshare, :margin; M=m, opt_criterion=:flci))
 for m in (0.02, 0.05, 0.1)]                        # sensitivity to M

rcp = CSV.read(pkgdir(DrSnow, "test", "validation", "rd", "rcp_sample.csv"), DataFrame)
rf = rd_honest(rcp, :log_cn, :elig_year; treatment=:retired, M=(0.001, 0.002),
               h=7)
rd_honest_ar_confidence_set(rf)     # weak first stage: an unbounded set
```

# References
- Armstrong, T. B., & Kolesár, M. (2018). Optimal inference in a class of regression
  models. *Econometrica*, 86(2), 655–683.
- Armstrong, T. B., & Kolesár, M. (2020). Simple and honest confidence intervals in
  nonparametric regression. *Quantitative Economics*, 11(1), 1–39.
- Kolesár, M., & Rothe, C. (2018). Inference in regression discontinuity designs with a
  discrete running variable. *American Economic Review*, 108(8), 2277–2304.
- Noack, C., & Rothe, C. (2024). Bias-aware inference in fuzzy regression discontinuity
  designs. *Econometrica*, 92(3), 687–711.
- Imbens, G., & Kalyanaraman, K. (2012). Optimal bandwidth choice for the regression
  discontinuity estimator. *Review of Economic Studies*, 79(3), 933–959.
- Imbens, G., & Wager, S. (2019). Optimized regression discontinuity designs. *Review
  of Economics and Statistics*, 101(2), 264–278.
"""
function rd_honest(data::AbstractDataFrame, outcome::Symbol, running::Symbol;
                   cutoff::Real=0.0, treatment=nothing, M=nothing, kernel=:triangular,
                   h::Union{Nothing,Real}=nothing, opt_criterion=:mse, beta::Real=0.8,
                   sclass=:holder, vce=nothing, J::Integer=3, cluster=nothing,
                   weights=nothing, covariates=Symbol[], T0::Real=0.0,
                   point_inference::Bool=false, level::Real=0.95)
    ctx = "rd_honest"
    (0 < level < 1) || throw(ArgumentError("$ctx: level must be in (0, 1)"))
    (0 < beta < 1) || throw(ArgumentError("$ctx: beta must be in (0, 1)"))
    J >= 1 || throw(ArgumentError("$ctx: J must be a positive integer"))
    kern = _rd_h_kernel(kernel)
    crit = _rd_h_criterion(opt_criterion)
    scl = _rd_h_sclass(sclass)
    alpha = 1 - Float64(level)
    if point_inference
        treatment === nothing || throw(ArgumentError(
            "$ctx: point_inference does not support a treatment variable"))
        isempty(covariates) || throw(ArgumentError(
            "$ctx: covariates are not allowed with point_inference"))
    end
    class = point_inference ? :ip : treatment === nothing ? :srd : :frd
    se_method = if vce === nothing
        cluster === nothing ? :nn : :ehw
    else
        v = Symbol(lowercase(string(vce)))
        v === :hc0 && (v = :ehw)
        v in (:nn, :ehw) || throw(ArgumentError("$ctx: vce must be :nn or :ehw"))
        v
    end
    (cluster !== nothing && se_method === :nn) && throw(ArgumentError(
        "$ctx: vce = :nn is not available with clustered data; use vce = :ehw"))
    d, E = _rd_h_build(data, outcome, running; cutoff, treatment, covariates, cluster,
                       weights, class, context=ctx)
    if class !== :ip
        (any(d.p) && any(d.m)) || throw(ArgumentError(
            "$ctx: no observations on one side of the cutoff"))
    end
    Mvec = nothing
    if M !== nothing
        Mvec = Float64.(collect(M isa Real ? (M,) : M))
        mlen = class === :frd ? 2 : 1
        (length(Mvec) == mlen && all(>=(0), Mvec)) || throw(ArgumentError(
            "$ctx: M must be a non-negative number" *
            (mlen == 2 ? " pair (M_Y, M_D) for fuzzy designs" : "")))
    end
    if h !== nothing
        (isfinite(h) && h > 0) || throw(ArgumentError("$ctx: h must be positive"))
    end
    T0f = Float64(T0)
    opt(dd, MM) = _rd_h_optbw(_rd_h_copy(dd; covs=nothing, Y_unadj=nothing), MM, kern,
                              crit, alpha, Float64(beta), scl, T0f)
    # Covariates: select a bandwidth without them and adjust the outcome.
    if d.covs !== nothing && (Mvec === nothing || h === nothing)
        d0 = _rd_h_copy(d; covs=nothing)
        h0 = h === nothing ? opt(d0, _rd_h_mrot(d0)) : Float64(h)
        r0 = _rd_h_npreg(d, h0, kern; se_method=:ehw)
        r0.eff_obs == 0 && throw(ArgumentError(
            "$ctx: the covariate-adjustment fit is not identified at bandwidth $h0"))
        d = _rd_h_copy(d; Y_unadj=d.Y, Y=Matrix{Float64}(r0.Yadj))
    end
    rot = Mvec === nothing
    rot && (Mvec = _rd_h_mrot(d))
    selected = h === nothing
    hh = selected ? opt(d, Mvec) : Float64(h)
    d.Y_unadj === nothing || (d = _rd_h_copy(d; Y=d.Y_unadj, Y_unadj=nothing))
    r = _rd_h_nprhonest(d, Mvec, kern, hh; se_method, J=Int(J), sclass=scl,
                        warn_collinear=true)
    r.eff_obs == 0 && throw(ArgumentError(
        "$ctx: the local linear fit is not identified at bandwidth $hh (too few " *
        "distinct values of the running variable with positive kernel weight on a " *
        "side); use a larger bandwidth"))
    (isfinite(r.se) && r.se > 0) || throw(ArgumentError(
        "$ctx: the standard error is not positive (se = $(r.se))"))
    if !isfinite(r.leverage) || r.leverage > 0.1
        @warn "$ctx: maximal leverage is large ($(round(r.leverage; digits=2))); " *
              "inference may be inaccurate. Consider a bigger bandwidth."
    end
    B = r.bias / r.se
    cv = _rd_cvb(B, alpha)
    za = quantile(Normal(), 1 - alpha)
    kw = [_rd_h_kern(kern, x / hh) > 0 for x in d.X]
    covnames = isempty(E.covariates) ? Symbol[] : E.covariates[r.covs_kept]
    design = class === :ip ? :point : class === :frd ? :fuzzy : :sharp
    return RDHonestEstimate(design, r.estimate, r.se, r.bias, cv, r.estimate - cv * r.se,
                            r.estimate + cv * r.se, r.estimate - (B + za) * r.se,
                            r.estimate + (B + za) * r.se,
                            _rd_h_pvalue(r.estimate / r.se, B), Float64(level), r.M,
                            r.M_rf, r.M_fs, rot,
                            class === :frd ? r.fs : nothing,
                            class === :frd ? r.rf : nothing, r.V, r.bias_constant, hh,
                            selected, crit, r.eff_obs, r.leverage, kern, scl, se_method,
                            Int(J), Float64(cutoff), T0f, count(d.m), count(d.p),
                            count(kw .& d.m), count(kw .& d.p), covnames,
                            d.cluster === nothing ? 0 : maximum(d.cluster))
end

"""
    rd_honest_ar_confidence_set(r::RDHonestEstimate; level=r.level) -> NamedTuple

Bias-aware Anderson–Rubin confidence set for the fuzzy RD parameter
``\\theta = \\tau_Y / \\tau_D`` (Noack & Rothe 2024), valid whether or not the first
stage is strong.

In a fuzzy RD design the delta-method interval of [`rd_honest`](@ref) relies on the
jump in take-up ``\\tau_D`` being large relative to its sampling error. When the first
stage is weak, that interval can badly under-cover (Feir, Lemieux & Marmer 2016). Noack
and Rothe (2024) combine the Anderson and Rubin (1949) idea with the bias-aware
approach of Armstrong and Kolesár (2020). For each candidate ``\\theta_0`` the jump in
``Y - \\theta_0 D`` is zero under ``H_0: \\theta = \\theta_0``. It is estimated with the
local linear weights of `r`, with standard error ``s(\\theta_0) = (V_{YY} - 2\\theta_0
V_{YD} + \\theta_0^2 V_{DD})^{1/2}`` and worst-case bias
``b(\\theta_0) = (M_Y + |\\theta_0| M_D)\\,c``, where ``c`` is `r.bias_constant`, under
the smoothness bounds of `r`. ``\\theta_0`` is retained when
```math
|\\hat\\tau_Y - \\theta_0\\hat\\tau_D| \\le
    \\text{cv}_{1-\\alpha}\\big(b(\\theta_0)/s(\\theta_0)\\big)\\,s(\\theta_0).
```
Each test is honest over the smoothness class, so the set has coverage of at least
``1-\\alpha`` uniformly over the class, whatever the strength of the first stage.

The set is computed on a fine grid over the whole real line (on an arctangent scale),
refined by bisection. It can be a bounded interval, a union of disjoint intervals or
rays, or the whole real line when the first stage cannot be distinguished from zero
given `M_D`. An unbounded set is informative: the data do not bound the effect. The
bandwidth, kernel, smoothness bounds and variance estimator are those of `r`. Noack and
Rothe (2024) recommend a bandwidth chosen for the Anderson–Rubin statistic, which can be
passed through `h` in [`rd_honest`](@ref). All caveats about the choice of `M` in
[`rd_honest`](@ref) apply, now to both `M_Y` and `M_D`. In the package's Monte Carlo
check with strong and weak first stages, the set covered 0.96 of the time at nominal
0.95, but with only 100 replications. The continuity-based analogue with robust bias
correction is [`rd_weak_iv_confidence_set`](@ref).

# Arguments
- `r::RDHonestEstimate`: a fuzzy design fitted by [`rd_honest`](@ref) with `treatment`.
  Sharp designs raise an `ArgumentError`.

# Keywords
- `level::Real=r.level`: confidence level.

# Returns
- `NamedTuple` with fields `kind` (`:interval`, `:union`, `:real_line` or `:empty`),
  `intervals::Vector{Tuple{Float64,Float64}}` (with `±Inf` for unbounded ends) and
  `level`.

# Examples
```julia
using DrSnow, CSV, DataFrames
rcp = CSV.read(pkgdir(DrSnow, "test", "validation", "rd", "rcp_sample.csv"), DataFrame)
r = rd_honest(rcp, :log_cn, :elig_year; treatment=:retired, M=(0.001, 0.002),
              h=7)
cs = rd_honest_ar_confidence_set(r)
cs.kind, cs.intervals           # weak first stage (≈ 0.08): the whole line
confint(r)                     # delta-method honest interval, for comparison
```

# References
- Noack, C., & Rothe, C. (2024). Bias-aware inference in fuzzy regression discontinuity
  designs. *Econometrica*, 92(3), 687–711.
- Armstrong, T. B., & Kolesár, M. (2020). Simple and honest confidence intervals in
  nonparametric regression. *Quantitative Economics*, 11(1), 1–39.
- Anderson, T. W., & Rubin, H. (1949). Estimation of the parameters of a single
  equation in a complete system of stochastic equations. *The Annals of Mathematical
  Statistics*, 20(1), 46–63.
- Feir, D., Lemieux, T., & Marmer, V. (2016). Weak identification in fuzzy regression
  discontinuity designs. *Journal of Business & Economic Statistics*, 34(2), 185–196.
"""
function rd_honest_ar_confidence_set(r::RDHonestEstimate; level::Real=r.level)
    r.design === :fuzzy || throw(ArgumentError(
        "rd_honest_ar_confidence_set requires a fuzzy design (pass `treatment`)"))
    (0 < level < 1) || throw(ArgumentError("level must be in (0, 1), got $level"))
    alpha = 1 - Float64(level)
    Vyy, Vyd, Vdd = r.V[1], (r.V[2] + r.V[3]) / 2, r.V[4]
    a, b = r.reduced_form, r.first_stage
    c = r.bias_constant
    function g(t)
        s2 = Vyy - 2 * t * Vyd + t^2 * Vdd
        s = sqrt(max(s2, 0.0))
        bias = (r.M_rf + abs(t) * r.M_fs) * c
        if s == 0
            return abs(a - t * b) - bias      # accept iff |a - t b| ≤ bias
        end
        return abs(a - t * b) - _rd_cvb(bias / s, alpha) * s
    end
    # behaviour at ±∞ (per unit |t|): accepted iff |b| ≤ cv(M_D c/√Vdd) √Vdd
    sd_inf = sqrt(max(Vdd, 0.0))
    ginf = sd_inf == 0 ? abs(b) - r.M_fs * c :
           abs(b) - _rd_cvb(r.M_fs * c / sd_inf, alpha) * sd_inf
    acc_inf = ginf <= 0
    # grid on the arctangent scale, plus points around the estimates
    scale = max(abs(a / b), abs(a), 1.0)
    phis = range(-pi / 2, pi / 2; length=4003)[2:(end - 1)]
    ts = sort(unique(vcat(scale .* tan.(phis), a / b, 0.0, -1e12 * scale,
                          1e12 * scale)))
    acc = [g(t) <= 0 for t in ts]
    intervals = Tuple{Float64,Float64}[]
    k = 1
    nt = length(ts)
    while k <= nt
        if acc[k]
            j = k
            while j < nt && acc[j + 1]
                j += 1
            end
            lo = k == 1 ? (acc_inf ? -Inf : ts[1]) : _rd_h_bisect(g, ts[k - 1], ts[k])
            hi = j == nt ? (acc_inf ? Inf : ts[nt]) : _rd_h_bisect(g, ts[j], ts[j + 1])
            push!(intervals, (lo, hi))
            k = j + 1
        else
            k += 1
        end
    end
    kind = if isempty(intervals)
        :empty
    elseif length(intervals) == 1 && intervals[1] == (-Inf, Inf)
        :real_line
    elseif length(intervals) == 1 && all(isfinite, intervals[1])
        :interval
    else
        :union
    end
    return (kind=kind, intervals=intervals, level=Float64(level))
end
