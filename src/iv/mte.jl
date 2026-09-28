# Marginal treatment effects (Heckman & Vytlacil 2005): propensity score, common
# support, local-IV estimation of the MTE (parametric normal / polynomial, Brinch,
# Mogstad & Wiswall 2017; semiparametric partially linear with local polynomials,
# Carneiro, Heckman & Vytlacil 2011), treatment-effect parameters as weighted
# averages of the MTE, and bootstrap inference.
#
# Model: D = 1{U_D ≤ P(Z, X)}, U_D ~ U(0, 1); Yⱼ = X'βⱼ + Uⱼ with
# E[U₁ − U₀ | U_D = u, X] = k(u) (separable MTE). Then
#   E[Y | X, P = p] = X'β₀ + p X'(β₁ − β₀) + K(p),   K(p) = ∫₀ᵖ k(u) du,
#   MTE(x, u) = ∂E[Y | X = x, P = p]/∂p |_{p=u} = x'(β₁ − β₀) + k(u).
# Here X excludes the intercept; the intercept of β₁ − β₀ is absorbed in k.

# ---------------------------------------------------------------------------
# Propensity score and support
# ---------------------------------------------------------------------------

function _iv_mte_prep(data, outcome, treatment, instruments, covariates, context)
    inst, covs = _as_symbols(instruments), _as_symbols(covariates)
    isempty(inst) && throw(ArgumentError("$context: at least one instrument is required"))
    cols = unique(vcat(outcome === nothing ? Symbol[] : [outcome], [treatment], inst,
                       covs))
    require_columns(data, cols; context=context)
    _iv_check_numeric(data, vcat(outcome === nothing ? Symbol[] : [outcome],
                                 [treatment], inst), context)
    mask = trues(nrow(data))
    for c in cols
        mask .&= .!ismissing.(data[!, c])
    end
    sub = disallowmissing(data[mask, cols])
    d = Float64.(sub[!, treatment])
    all(x -> x == 0 || x == 1, d) ||
        throw(ArgumentError("$context: treatment `$treatment` must be binary (0/1)"))
    0 < sum(d) < length(d) || throw(ArgumentError("$context: the treatment does not vary"))
    return sub, d, inst, covs, findall(mask)
end

function _iv_propensity(sub::AbstractDataFrame, d::Vector{Float64}, inst, covs,
                        link::Symbol)
    link in (:probit, :logit, :lpm) ||
        throw(ArgumentError("link must be :probit, :logit or :lpm, got :$link"))
    A = hcat(_iv_exog_matrix(sub, covs, true), Matrix{Float64}(sub[!, inst]))
    F = qr(A, ColumnNorm())
    r = count(>(1e-10 * abs(F.R[1, 1])), abs.(diag(F.R)))
    A = A[:, sort(F.p[1:r])]
    if link === :lpm
        p = A * (A \ d)
    else
        m = GLM.glm(A, d, Binomial(), link === :probit ? ProbitLink() : LogitLink())
        p = GLM.predict(m)
    end
    if any(x -> !(0 < x < 1), p)
        throw(ArgumentError("estimated propensity scores outside (0, 1) " *
                            "($(count(x -> !(0 < x < 1), p)) observations); use a " *
                            "probit / logit link or fewer covariates"))
    end
    return p
end

"""
    mte_propensity(data, treatment, instruments; covariates=Symbol[], link=:probit)
        -> NamedTuple

Propensity score and common-support diagnostics for marginal-treatment-effect
analysis.

In the generalized Roy model that underlies marginal treatment effects, treatment is
chosen according to the latent index ``D = 1\\{U_D \\le P(Z, X)\\}``, where ``U_D`` is
uniformly distributed given the covariates ``X`` and the propensity score ``P(Z, X) =
\\Pr(D = 1 \\mid Z, X)`` summarizes all the instrument variation. Vytlacil (2002) shows
that this latent-index structure is equivalent to the independence and monotonicity
assumptions of Imbens and Angrist (1994), so the MTE framework adds no restriction to
the LATE model beyond them; Heckman and Vytlacil (2005) build on it the
characterization of treatment-effect parameters as weighted averages of the MTE. The
MTE at ``u`` is identified, without functional-form assumptions, only at values of
``u`` that the propensity score actually takes; the support of ``P`` therefore
determines which parameters the data can speak to.

This function fits the propensity by probit (default), logit or a linear probability
model on the covariates and the instruments, and reports the common support
``[\\max(\\min P \\mid D=1, \\min P \\mid D=0), \\min(\\max P \\mid D=1, \\max P \\mid
D=0)]``, the range of propensities observed among both treated and untreated units,
together with the distribution of ``P`` by treatment status. Parameters such as the
ATE require the MTE on all of ``[0, 1]`` and are nonparametrically identified only
when the support of ``P`` is the full unit interval (conditional on ``X``); a narrow
support confines nonparametric estimation to a narrow band of ``u`` (Carneiro,
Heckman and Vytlacil 2011). With covariates, the relevant support is conditional on
``X``, and the unconditional support reported here overstates it.

# Arguments
- `data::AbstractDataFrame`: the data; incomplete rows are dropped.
- `treatment::Symbol`: the binary (0/1) treatment ``D``.
- `instruments`: the excluded instrument(s) ``Z``, a `Symbol` or vector of `Symbol`s.

# Keywords
- `covariates::Vector{Symbol}`: covariates ``X`` entering the propensity score (default
  none).
- `link::Symbol`: `:probit` (default), `:logit` or `:lpm` (linear probability model);
  fitted values outside ``(0, 1)`` raise an error.

# Returns
- A `NamedTuple` with `propensity` (one entry per data row, `missing` for incomplete
  rows), `support = (lower, upper)`, `n_outside` (number of units outside the common
  support), `by_treatment` (`DataFrame` of propensity quantiles by treatment status)
  and `histogram` (`DataFrame` of counts by treatment status over 20 bins of ``P``).

# Examples
```julia
using DrSnow, DataFrames, StableRNGs
rng = StableRNG(4)
n = 3_000
z, x = randn(rng, n), randn(rng, n)
ud = rand(rng, n)                                  # unobserved resistance U_D
d = Float64.(ud .<= 1 ./ (1 .+ exp.(-(1.5 .* z .+ 0.5 .* x))))
df = DataFrame(d=d, z=z, x=x)
ps = mte_propensity(df, :d, :z; covariates=[:x], link=:logit)
ps.support, ps.n_outside
ps.by_treatment
```

# References
- Heckman, J. J., & Vytlacil, E. (2005). Structural equations, treatment effects,
  and econometric policy evaluation. *Econometrica*, 73(3), 669–738.
- Vytlacil, E. (2002). Independence, monotonicity, and latent index models: An
  equivalence result. *Econometrica*, 70(1), 331–341.
- Imbens, G. W., & Angrist, J. D. (1994). Identification and estimation of local
  average treatment effects. *Econometrica*, 62(2), 467–475.
- Carneiro, P., Heckman, J. J., & Vytlacil, E. J. (2011). Estimating marginal returns
  to education. *American Economic Review*, 101(6), 2754–2781.
"""
function mte_propensity(data::AbstractDataFrame, treatment::Symbol, instruments;
                        covariates=Symbol[], link::Symbol=:probit)
    sub, d, inst, covs, rows = _iv_mte_prep(data, nothing, treatment, instruments,
                                            covariates, "mte_propensity")
    p = _iv_propensity(sub, d, inst, covs, link)
    s = _iv_support_summary(p, d)
    out = Vector{Union{Missing,Float64}}(missing, nrow(data))
    out[rows] = p
    return merge((propensity=out,), s)
end

function _iv_support_summary(p, d)
    t = d .== 1
    lo = max(minimum(p[t]), minimum(p[.!t]))
    hi = min(maximum(p[t]), maximum(p[.!t]))
    lo < hi || throw(ArgumentError("no common support: the propensity ranges of " *
                                   "treated and untreated units do not overlap"))
    qs = [0.0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0]
    bt = DataFrame(quantile=qs, treated=quantile(p[t], qs),
                   untreated=quantile(p[.!t], qs))
    edges = range(0, 1; length=21)
    h = DataFrame(lower=edges[1:(end - 1)], upper=edges[2:end],
                  treated=[count(x -> e0 <= x < e1 || (e1 == 1 && x == 1), p[t])
                           for (e0, e1) in zip(edges[1:(end - 1)], edges[2:end])],
                  untreated=[count(x -> e0 <= x < e1 || (e1 == 1 && x == 1), p[.!t])
                             for (e0, e1) in zip(edges[1:(end - 1)], edges[2:end])])
    return (support=(lower=lo, upper=hi), n_outside=count(x -> x < lo || x > hi, p),
            by_treatment=bt, histogram=h)
end

# ---------------------------------------------------------------------------
# Local polynomial regression (Gaussian kernel)
# ---------------------------------------------------------------------------

"""Local linear fit of every column of `M` on `x`, evaluated at `x` itself
(Robinson residualization). Blocked, O(n²)."""
function _iv_loclin_fitted(x::Vector{Float64}, M::Matrix{Float64}, h::Real)
    n = length(x)
    out = similar(M)
    xM = x .* M
    bs = 512
    for i0 in 1:bs:n
        I = i0:min(n, i0 + bs - 1)
        W = exp.(-0.5 .* ((x[I] .- x') ./ h) .^ 2)
        S0 = vec(sum(W; dims=2))
        Sx = W * x
        Sxx = W * (x .^ 2)
        T0 = W * M
        Tx = W * xM
        x0 = x[I]
        S1 = Sx .- x0 .* S0
        S2 = Sxx .- 2 .* x0 .* Sx .+ x0 .^ 2 .* S0
        T1 = Tx .- x0 .* T0
        den = S0 .* S2 .- S1 .^ 2
        out[I, :] = (S2 .* T0 .- S1 .* T1) ./ den
    end
    return out
end

"""Local polynomial (degree `deg`) level and first derivative of E[y | x] at `grid`."""
function _iv_locpoly(x::Vector{Float64}, y::Vector{Float64}, grid::AbstractVector,
                     h::Real; deg::Int=2)
    lev = zeros(length(grid))
    der = zeros(length(grid))
    for (g, x0) in enumerate(grid)
        t = x .- x0
        w = exp.(-0.5 .* (t ./ h) .^ 2)
        A = hcat([t .^ j for j in 0:deg]...)
        Aw = A .* w
        b = Symmetric(A' * Aw) \ (Aw' * y)
        lev[g] = b[1]
        der[g] = b[2]
    end
    return lev, der
end

# ---------------------------------------------------------------------------
# MTE estimation
# ---------------------------------------------------------------------------

"""
Fit the MTE model on (y, d, X (no intercept), p). Returns a NamedTuple with the
covariate part `δ` (coefficients of p·X), `kfun(u)` (k(u)), `Kfun(u)` (∫ k up to a
constant; differences are what matter), and extra coefficients.
"""
function _iv_mte_fit(y, X, p, method, degree, bandwidth, rbandwidth, lo, hi)
    n, q = size(X)
    if method === :polynomial || method === :normal
        extra = method === :polynomial ?
                hcat([p .^ l for l in 2:(degree + 1)]...) :
                reshape(pdf.(Normal(), quantile.(Normal(), p)), :, 1)
        A = hcat(ones(n), X, p, p .* X, extra)
        F = qr(A, ColumnNorm())
        rk = count(>(1e-10 * abs(F.R[1, 1])), abs.(diag(F.R)))
        rk == size(A, 2) || throw(ArgumentError("the MTE regression is collinear (the " *
                                                "propensity score has too little " *
                                                "variation for this specification)"))
        b = A \ y
        c = b[2 + q]                          # coefficient on p (intercept part)
        δ = b[(3 + q):(2 + 2q)]
        φ = b[(3 + 2q):end]
        if method === :polynomial
            kfun = u -> c + sum(l * φ[l - 1] * u^(l - 1) for l in 2:(degree + 1))
            Kfun = u -> c * u + sum(φ[l - 1] * u^l for l in 2:(degree + 1))
        else
            # K(p) = −s φ(Φ⁻¹(p)) ⇒ k(u) = s Φ⁻¹(u); coefficient on φ(Φ⁻¹(p)) is −s
            s = -φ[1]
            kfun = u -> c + s * quantile(Normal(), u)
            Kfun = u -> c * u - s * (0 < u < 1 ? pdf(Normal(), quantile(Normal(), u)) :
                                     0.0)
        end
        return (δ=δ, β0=b[2:(1 + q)], kfun=kfun, Kfun=Kfun, extra=φ, c=c)
    end
    # semiparametric: Robinson double residual, then local quadratic in p
    if q > 0
        M = hcat(y, X, p .* X)
        R = M .- _iv_loclin_fitted(p, M, rbandwidth)
        ry, rX = R[:, 1], R[:, 2:end]
        b = rX \ ry
        β0, δ = b[1:q], b[(q + 1):end]
    else
        β0, δ = Float64[], Float64[]
    end
    ỹ = q > 0 ? y .- X * β0 .- (p .* X) * δ : copy(y)
    grid = collect(range(lo, hi; length=101))
    lev, der = _iv_locpoly(p, ỹ, grid, bandwidth; deg=2)
    interp(v) = u -> begin
        (lo - 1e-12 <= u <= hi + 1e-12) ||
            throw(ArgumentError("u = $u is outside the common support [$lo, $hi]; " *
                                "the semiparametric MTE is not identified there"))
        t = clamp((u - lo) / (hi - lo) * 100, 0, 100)
        j = min(floor(Int, t), 99)
        f = t - j
        (1 - f) * v[j + 1] + f * v[j + 2]
    end
    return (δ=δ, β0=β0, kfun=interp(der), Kfun=interp(lev), extra=Float64[], c=NaN)
end

"""Treatment-effect parameters from a fitted MTE model."""
function _iv_mte_params(fit, X, p, lo, hi, method, late_bounds, policy_p)
    mx = size(X, 2) == 0 ? zeros(length(p)) : X * fit.δ
    K = fit.Kfun
    names = String[]
    vals = Float64[]
    full = method !== :semiparametric
    if full
        K0, K1 = K(0.0), K(1.0)
        push!(names, "ATE"); push!(vals, mean(mx) + K1 - K0)
        push!(names, "ATT")
        push!(vals, sum(p .* mx .+ K.(p) .- K0) / sum(p))
        push!(names, "ATU")
        push!(vals, sum((1 .- p) .* mx .+ K1 .- K.(p)) / sum(1 .- p))
    end
    a, b = late_bounds === nothing ? (lo, hi) : late_bounds
    push!(names, @sprintf("LATE(%.3g, %.3g)", a, b))
    push!(vals, mean(mx) + (K(b) - K(a)) / (b - a))
    if policy_p !== nothing
        Δ = policy_p .- p
        abs(sum(Δ)) > 1e-10 * length(p) ||
            throw(ArgumentError("the policy does not change the average propensity; " *
                                "the PRTE is not defined"))
        push!(names, "PRTE")
        push!(vals, sum(Δ .* mx .+ K.(policy_p) .- K.(p)) / sum(Δ))
    end
    return names, vals
end

"""
    MTEEstimate <: CausalEstimate

Result of [`mte`](@ref): marginal treatment effects estimated by local instrumental
variables and the treatment-effect parameters obtained as weighted averages of them.

The coefficients are treatment-effect parameters named by `coefnames(r)`. Which
parameters are present depends on the method. The parametric methods (`:polynomial`,
`:normal`) report `"ATE"`, `"ATT"` and `"ATU"`, which integrate the MTE over all of
``[0, 1]`` and therefore rely on the functional form wherever the propensity score
has no support; `"LATE(a, b)"`, the average MTE over ``u \\in [a, b]``; and `"PRTE"`
when a policy is given. The semiparametric method estimates the MTE only on the
common support of the propensity score and reports only `"LATE(a, b)"` and `"PRTE"`:
the ATE, ATT and ATU are not estimated because they are not identified without
extrapolation beyond the support. The covariance is the bootstrap covariance of the
parameters, and `confint` uses normal critical values.

# Fields
- `coef::Vector{Float64}`, `vcov::Matrix{Float64}`, `coefnames::Vector{String}`:
  parameter estimates, their bootstrap covariance and names.
- `nobs::Int`: observations used (after trimming to the common support when
  `trim = true`).
- `level::Float64`: confidence level.
- `method::Symbol`: `:normal`, `:polynomial` or `:semiparametric`.
- `curve::DataFrame`: the MTE evaluated at the covariate means on a grid of ``u``,
  with bootstrap standard errors and pointwise normal intervals (columns `u`, `mte`,
  `se`, `lower`, `upper`); the grid is 49 points in ``[0.02, 0.98]`` for the
  parametric methods and 21 points on the common support for the semiparametric one.
- `covariate_coef::Vector{Float64}`: ``\\beta_1 - \\beta_0`` for the covariates (the
  covariate part of the MTE); `covariates::Vector{Symbol}` names them.
- `support::NamedTuple`: `(lower, upper)`, the common support of the propensity
  score; `n_outside::Int`: units outside it; `trimmed::Bool`: whether they were
  dropped.
- `propensity::Vector{Union{Missing,Float64}}`: the fitted propensity, one entry per
  data row.
- `link::Symbol`, `degree::Int`, `bandwidth::Float64`: propensity link, polynomial
  degree and the local-polynomial bandwidth used.
- `n_bootstrap::Int`, `bootstrap::Matrix{Float64}`: number of bootstrap draws and the
  draws of the parameters (one row per draw).
"""
struct MTEEstimate <: CausalEstimate
    coef::Vector{Float64}
    vcov::Matrix{Float64}
    coefnames::Vector{String}
    nobs::Int
    level::Float64
    method::Symbol
    curve::DataFrame
    covariate_coef::Vector{Float64}
    covariates::Vector{Symbol}
    support::NamedTuple
    n_outside::Int
    trimmed::Bool
    propensity::Vector{Union{Missing,Float64}}
    link::Symbol
    degree::Int
    bandwidth::Float64
    n_bootstrap::Int
    bootstrap::Matrix{Float64}
end

StatsAPI.coef(r::MTEEstimate) = r.coef
StatsAPI.vcov(r::MTEEstimate) = r.vcov
StatsAPI.coefnames(r::MTEEstimate) = r.coefnames
StatsAPI.nobs(r::MTEEstimate) = r.nobs
StatsAPI.confint(r::MTEEstimate; level::Real=r.level) =
    invoke(StatsAPI.confint, Tuple{CausalEstimate}, r; level=level)
estimand(::MTEEstimate) = "treatment-effect parameters as weighted averages of the MTE"
function method_name(r::MTEEstimate)
    r.method === :normal && return "MTE (parametric normal, local IV)"
    r.method === :polynomial && return "MTE (polynomial of degree $(r.degree), local IV)"
    return "MTE (semiparametric local IV)"
end

function show_details(io::IO, r::MTEEstimate)
    println(io)
    @printf(io, "Propensity: %s; common support [%.3f, %.3f] (%d units outside%s)\n",
            r.link, r.support.lower, r.support.upper, r.n_outside,
            r.trimmed ? ", dropped" : "")
    @printf(io, "Standard errors: nonparametric bootstrap (%d draws)\n", r.n_bootstrap)
    c = r.curve
    idx = unique(round.(Int, range(1, nrow(c); length=min(nrow(c), 7))))
    println(io, "MTE at covariate means:")
    for i in idx
        @printf(io, "  u = %.3f: %.4g (se %.3g)\n", c.u[i], c.mte[i], c.se[i])
    end
    if r.method === :semiparametric
        println(io, "Note: ATE / ATT / ATU need the MTE on all of [0, 1]; they are not " *
                    "identified without extrapolation beyond the common support.")
    else
        println(io, "Note: parameters outside the common support rely on the " *
                    "functional form of the MTE.")
    end
end

"""
    mte(data, outcome, treatment, instruments; covariates=Symbol[],
        method=:semiparametric, link=:probit, degree=2, bandwidth=nothing,
        residual_bandwidth=nothing, trim=true, late_bounds=nothing, policy=nothing,
        n_bootstrap=200, level=0.95, rng=Random.default_rng()) -> MTEEstimate

Marginal treatment effects (MTE) by local instrumental variables, and the
treatment-effect parameters they imply.

With a continuous instrument, the LATE of a single instrument change is only one of
many parameters that the data can inform. Heckman and Vytlacil (2005) organize them
around the marginal treatment effect in the generalized Roy model: treatment is
``D = 1\\{U_D \\le P(Z, X)\\}``, with ``U_D`` uniform given ``X`` (the "resistance" to
treatment) and ``P`` the propensity score, and potential outcomes are ``Y_j = X'\\beta_j
+ U_j``. Assuming that ``(U_0, U_1, U_D)`` is independent of ``Z`` given ``X`` and that
``E[U_1 - U_0 \\mid U_D = u, X] = k(u)`` does not depend on ``X`` (additive
separability), the MTE and the conditional mean of the outcome satisfy

```math
E[Y \\mid X, P = p] = X'\\beta_0 + p\\,X'(\\beta_1 - \\beta_0) + K(p), \\qquad
\\mathrm{MTE}(x, u) = x'(\\beta_1 - \\beta_0) + k(u) = \\frac{\\partial
E[Y \\mid X = x, P = p]}{\\partial p}\\Big|_{p = u},
```

with ``K(p) = \\int_0^p k(u)\\,du``. The MTE is the average effect for units at the
margin of indifference ``U_D = u``; units with high ``u`` are unlikely to take the
treatment. A declining MTE indicates selection on gains ("essential heterogeneity",
in which case IV estimands depend on the instrument; Heckman, Urzua and Vytlacil
2006): those most likely to be treated benefit most. The independence and
monotonicity of the LATE model are equivalent to the latent-index structure
(Vytlacil 2002); the separability assumption, which lets the MTE be identified
over the unconditional support of ``P`` rather than the support conditional on
``X``, is a substantive restriction (Carneiro, Heckman and Vytlacil 2011; Brinch,
Mogstad and Wiswall 2017).

Three estimators of ``k(u)`` are available. The `:semiparametric` method (default;
Carneiro, Heckman and Vytlacil 2011) first removes the covariate terms by Robinson's
(1988) double-residual regression of ``Y``, ``X`` and ``P X`` on ``P`` (local linear,
Gaussian kernel, `residual_bandwidth`), then fits a local quadratic regression of
``Y - X'\\hat\\beta_0 - P X'\\hat\\delta`` on ``P`` (`bandwidth`), whose derivative
estimates ``k(u)``; it is identified only on the common support of ``P``. The
`:polynomial` method (Brinch, Mogstad and Wiswall 2017) takes ``K(p)`` to be a
polynomial of degree `degree + 1`, so that the MTE is a polynomial of degree
`degree` in ``u``, and estimates it by least squares; with a discrete instrument the
number of distinct propensity values, together with separability, limits how flexible
an MTE can be identified. The
`:normal` method assumes joint normality of the unobservables, so that ``k(u) = c + s
\\Phi^{-1}(u)``, and estimates it by the local-IV regression on
``\\phi(\\Phi^{-1}(p))`` (the control-function version, not the maximum-likelihood
switching regression).

Treatment-effect parameters are computed as sample analogues of weighted averages of
the MTE (Heckman and Vytlacil 2005), for instance ``\\mathrm{ATT} = \\sum_i
\\int_0^{P_i} \\mathrm{MTE}(X_i, u)\\,du / \\sum_i P_i``. The ATE, ATT and ATU are reported
only for the parametric methods (`:polynomial`, `:normal`): they integrate the MTE
over all of ``[0, 1]``, which is nonparametrically identified only if the propensity
score has full support on the unit interval, so in practice they extrapolate the
parametric form beyond the support. For every method the function reports
``\\mathrm{LATE}(a, b)``, the average MTE over ``u \\in [a, b]`` (default: the common
support), and, when `policy` is given, the policy-relevant treatment effect (PRTE)
of a policy that shifts the propensity from ``P_i`` to ``P'_i``, ``\\sum_i
\\int_{P_i}^{P'_i} \\mathrm{MTE}(X_i, u)\\,du / \\sum_i (P'_i - P_i)``. For the
semiparametric method the bounds of the LATE and the policy propensities must lie
within the common support. When point identification of a target parameter would
require extrapolation, [`mte_bounds`](@ref) delivers the sharp bounds implied by the
data and explicit shape restrictions instead (Mogstad, Santos and Torgovitsky 2018).

Standard errors are obtained by the nonparametric bootstrap over units
(`n_bootstrap` draws): in every draw the propensity score is re-estimated and the
MTE and the parameters are recomputed with the original bandwidths, support and
evaluation points; resamples with a degenerate propensity fit are redrawn with a
warning. Intervals use normal critical values. Bandwidth choice matters for the
semiparametric estimates, whose bootstrap intervals do not account for smoothing bias;
report the propensity-score support ([`mte_propensity`](@ref)) and the sensitivity of
the MTE curve to the method and bandwidth.

# Arguments
- `data::AbstractDataFrame`: the data; incomplete rows are dropped.
- `outcome::Symbol`: the outcome ``Y``.
- `treatment::Symbol`: the binary (0/1) treatment ``D``.
- `instruments`: the excluded instrument(s) ``Z``, a `Symbol` or vector of `Symbol`s.

# Keywords
- `covariates::Vector{Symbol}`: covariates ``X`` (default none); they enter both the
  propensity score and the MTE (linearly, through ``\\beta_1 - \\beta_0``).
- `method::Symbol`: `:semiparametric` (default), `:polynomial` or `:normal`.
- `link::Symbol`: propensity model, `:probit` (default), `:logit` or `:lpm`.
- `degree::Int`: degree of the MTE polynomial for `method = :polynomial` (default 2,
  at least 1).
- `bandwidth`, `residual_bandwidth`: kernel bandwidths in propensity units for the
  local quadratic regression and the double-residual step (defaults ``2\\,\\mathrm{sd}(P)
  n^{-1/7}`` and ``1.06\\,\\mathrm{sd}(P) n^{-1/5}``).
- `trim::Bool`: drop units outside the common support before estimation (default
  `true`).
- `late_bounds`: `nothing` (default, the common support) or `(a, b)` with ``0 \\le a <
  b \\le 1``, the ``u``-interval of the reported LATE.
- `policy`: `nothing` (default), a function mapping the vector of propensity scores to
  the propensities under the policy (e.g. `p -> min.(p .+ 0.1, 1)`), or a vector with
  one entry per data row.
- `n_bootstrap::Int`: bootstrap draws (default 200, at least 20).
- `level::Real`: confidence level (default 0.95).
- `rng::AbstractRNG`: random-number generator for the bootstrap.

# Returns
- An [`MTEEstimate`](@ref); `coeftable(r)` lists the parameters, `r.curve` the MTE
  curve at the covariate means with pointwise intervals.

# Examples
```julia
using DrSnow, DataFrames, StableRNGs
rng = StableRNG(4)
n = 3_000
z, x = randn(rng, n), randn(rng, n)
ud = rand(rng, n)                                  # unobserved resistance U_D
d = Float64.(ud .<= 1 ./ (1 .+ exp.(-(1.5 .* z .+ 0.5 .* x))))
y0 = x .+ randn(rng, n)
y = y0 .+ d .* (1.0 .+ 0.5 .* x .- 2.0 .* (ud .- 0.5))  # MTE(x, u) = 2 + 0.5x - 2u
df = DataFrame(y=y, d=d, z=z, x=x)
r = mte(df, :y, :d, :z; covariates=[:x], method=:polynomial, degree=2,
        link=:logit, policy=p -> min.(p .+ 0.1, 1.0), n_bootstrap=100,
        rng=StableRNG(5))
coeftable(r)
first(r.curve, 5)
rs = mte(df, :y, :d, :z; covariates=[:x], link=:logit, n_bootstrap=50,
         rng=StableRNG(5))                         # LATE over the support only
coefnames(rs)
```

# References
- Heckman, J. J., & Vytlacil, E. (2005). Structural equations, treatment effects,
  and econometric policy evaluation. *Econometrica*, 73(3), 669–738.
- Heckman, J. J., Urzua, S., & Vytlacil, E. (2006). Understanding instrumental
  variables in models with essential heterogeneity. *Review of Economics and
  Statistics*, 88(3), 389–432.
- Vytlacil, E. (2002). Independence, monotonicity, and latent index models: An
  equivalence result. *Econometrica*, 70(1), 331–341.
- Carneiro, P., Heckman, J. J., & Vytlacil, E. J. (2011). Estimating marginal returns
  to education. *American Economic Review*, 101(6), 2754–2781.
- Brinch, C. N., Mogstad, M., & Wiswall, M. (2017). Beyond LATE with a discrete
  instrument. *Journal of Political Economy*, 125(4), 985–1039.
- Robinson, P. M. (1988). Root-N-consistent semiparametric regression.
  *Econometrica*, 56(4), 931–954.
- Mogstad, M., Santos, A., & Torgovitsky, A. (2018). Using instrumental variables for
  inference about policy relevant treatment parameters. *Econometrica*, 86(5),
  1589–1619.
"""
function mte(data::AbstractDataFrame, outcome::Symbol, treatment::Symbol, instruments;
             covariates=Symbol[], method::Symbol=:semiparametric, link::Symbol=:probit,
             degree::Int=2, bandwidth=nothing, residual_bandwidth=nothing,
             trim::Bool=true, late_bounds=nothing, policy=nothing,
             n_bootstrap::Int=200, level::Real=0.95,
             rng::AbstractRNG=Random.default_rng())
    method in (:semiparametric, :polynomial, :normal) ||
        throw(ArgumentError("method must be :semiparametric, :polynomial or :normal"))
    degree >= 1 || throw(ArgumentError("degree must be at least 1"))
    n_bootstrap >= 20 || throw(ArgumentError("n_bootstrap must be at least 20"))
    0 < level < 1 || throw(ArgumentError("level must be in (0, 1)"))
    if late_bounds !== nothing
        (length(late_bounds) == 2 && 0 <= late_bounds[1] < late_bounds[2] <= 1) ||
            throw(ArgumentError("late_bounds must be (a, b) with 0 ≤ a < b ≤ 1"))
    end
    sub, d, inst, covs, rows = _iv_mte_prep(data, outcome, treatment, instruments,
                                            covariates, "mte")
    y = Float64.(sub[!, outcome])
    X = _iv_exog_matrix(sub, covs, false)
    p = _iv_propensity(sub, d, inst, covs, link)
    sup = _iv_support_summary(p, d)
    lo, hi = sup.support
    keep = trim ? (p .>= lo) .& (p .<= hi) : trues(length(p))
    n = count(keep)
    n >= 20 || throw(ArgumentError("mte: too few observations on the common support"))
    if method === :semiparametric && late_bounds !== nothing
        (lo <= late_bounds[1] && late_bounds[2] <= hi) ||
            throw(ArgumentError("late_bounds must lie in the common support " *
                                "[$(round(lo; digits=4)), $(round(hi; digits=4))] for " *
                                "the semiparametric method"))
    end
    sdp = std(p[keep])
    h = bandwidth === nothing ? 2 * sdp * n^(-1 / 7) : float(bandwidth)
    hr = residual_bandwidth === nothing ? 1.06 * sdp * n^(-1 / 5) :
         float(residual_bandwidth)
    (h > 0 && hr > 0) || throw(ArgumentError("bandwidths must be positive"))
    policy_of = pp -> begin
        policy === nothing && return nothing
        v = policy isa Function ? policy(pp) : Float64.(collect(policy))
        length(v) == length(pp) ||
            throw(ArgumentError("the policy must give one propensity per unit"))
        all(x -> 0 <= x <= 1, v) ||
            throw(ArgumentError("policy propensities must lie in [0, 1]"))
        if method === :semiparametric && any(x -> x < lo || x > hi, v)
            throw(ArgumentError("policy propensities outside the common support: the " *
                                "semiparametric PRTE is not identified"))
        end
        Float64.(v)
    end
    if policy !== nothing && !(policy isa Function)
        length(policy) == nrow(data) ||
            throw(ArgumentError("a policy vector must have one entry per data row"))
    end
    pol_full = policy === nothing || policy isa Function ? nothing :
               Float64.(collect(policy))[rows]
    polp(pp, idx) = pol_full === nothing ? policy_of(pp) : policy_of(pol_full[idx])
    ugrid = method === :semiparametric ? collect(range(lo, hi; length=21)) :
            collect(range(0.02, 0.98; length=49))
    xbar = size(X, 2) == 0 ? Float64[] : vec(mean(X[keep, :]; dims=1))
    function estimate_once(yv, Xv, pv, idxv)
        local fit, nms, vls, crv
        fit = _iv_mte_fit(yv, Xv, pv, method, degree, h, hr, lo, hi)
        nms, vls = _iv_mte_params(fit, Xv, pv, lo, hi, method, late_bounds,
                                  polp(pv, idxv))
        crv = [dot(xbar, fit.δ) + fit.kfun(u) for u in ugrid]
        return nms, vls, crv, fit
    end
    idx0 = findall(keep)
    names, vals, curve, fit0 = estimate_once(y[keep], X[keep, :], p[keep], idx0)
    # bootstrap: resample units, re-estimate the propensity score
    seeds = task_seeds(rng, n_bootstrap)
    Bv = zeros(n_bootstrap, length(vals))
    Bc = zeros(n_bootstrap, length(ugrid))
    nall = length(y)
    nfail = 0
    for b in 1:n_bootstrap
        brng = Xoshiro(seeds[b])
        ok = false
        for _ in 1:10
            ib = rand(brng, 1:nall, nall)
            try
                pb = _iv_propensity(sub[ib, :], d[ib], inst, covs, link)
                kb = trim ? (pb .>= lo) .& (pb .<= hi) : trues(nall)
                sel = ib[kb]
                _, vb, cb, _ = estimate_once(y[sel], X[sel, :], pb[kb], sel)
                Bv[b, :] = vb
                Bc[b, :] = cb
                ok = true
                break
            catch err
                err isa ArgumentError || err isa LinearAlgebra.SingularException ||
                    err isa LinearAlgebra.PosDefException || rethrow()
                nfail += 1
            end
        end
        ok || error("mte: bootstrap draw $b failed repeatedly")
    end
    nfail > 0 && @warn "mte: $nfail bootstrap resample(s) were redrawn (degenerate " *
                       "propensity fit or support)"
    V = cov(Bv)
    se_c = vec(std(Bc; dims=1))
    cv = critical_value(level)
    cdf_ = DataFrame(u=ugrid, mte=curve, se=se_c, lower=curve .- cv .* se_c,
                     upper=curve .+ cv .* se_c)
    pfull = Vector{Union{Missing,Float64}}(missing, nrow(data))
    pfull[rows] = p
    return MTEEstimate(vals, Matrix(Symmetric(V)), names, n, float(level), method, cdf_,
                       fit0.δ, covs, sup.support, sup.n_outside, trim, pfull, link,
                       degree, h, n_bootstrap, Bv)
end
