# k-class estimators (LIML, Fuller, general κ) and their heteroskedasticity-robust
# jackknife versions (HLIM, HFUL), with conventional, robust / cluster, Bekker (1994)
# many-instrument and Hausman et al. (2012) many-instrument heteroskedasticity-robust
# standard errors.
#
# Everything is computed on the partialled `_IVDesign` of a 2SLS fit (covariates,
# fixed effects and weights already absorbed). For a k-class estimator with
# included exogenous regressors W and full instrument set [Z, W], the coefficient on
# the endogenous regressors and its covariance equal those of the same estimator on
# the partialled data (Frisch–Waugh–Lovell), because the k-class instrument
# (I − κ M_[Z,W]) X leaves W unchanged.

"""
    KClassEstimate <: CausalEstimate

Result of [`kclass_iv`](@ref): a k-class (LIML, Fuller, user-supplied κ) or jackknife
k-class (HLIM, HFUL) estimate of the coefficients on the endogenous regressors.

The object stores the point estimates and covariance of the endogenous coefficients
only: covariates, fixed effects and weights are partialled out before estimation, so
their coefficients are not reported. It also keeps the 2SLS fit of the same
specification (`tsls`), which is useful for comparison (a large gap between LIML-type
estimates and 2SLS is a symptom of many-instrument bias) and whose `estimand` label
describes the causal interpretation available under heterogeneous effects. The
`StatsAPI` accessors `coef`, `vcov`, `stderror`, `confint`, `coeftable`, `nobs` and
`dof_residual` work on it, and `estimate(r)` returns the first endogenous coefficient.

# Fields
- `coef::Vector{Float64}`, `vcov::Matrix{Float64}`, `coefnames::Vector{String}`:
  endogenous coefficients, their covariance and names.
- `nobs::Int`, `dof_residual::Float64`: sample size and the degrees of freedom of the
  reference distribution.
- `method::Symbol`: `:liml`, `:fuller`, `:kclass`, `:hlim` or `:hful`.
- `kappa::Float64`: the κ of the k-class estimator; for HLIM and HFUL the jackknife
  eigenvalue ``\\hat\\alpha`` (see [`kclass_iv`](@ref)).
- `fuller_alpha::Float64`: the Fuller constant α (Fuller) or C (HFUL); `NaN` for the
  other methods.
- `se_type::String`: the covariance estimator used.
- `outcome`, `endogenous`, `instruments`, `covariates`, `fe`, `weights`: the
  specification.
- `level::Float64`: default confidence level.
- `estimand::String`, `estimand_note::String`: the target parameter and its caveats.
- `first_stage::WeakIVDiagnostics`: first-stage diagnostics of the specification.
- `tsls::IVEstimate`: the 2SLS fit of the same specification.
"""
struct KClassEstimate <: CausalEstimate
    coef::Vector{Float64}
    vcov::Matrix{Float64}
    coefnames::Vector{String}
    nobs::Int
    dof_residual::Float64
    method::Symbol
    kappa::Float64
    fuller_alpha::Float64
    se_type::String
    outcome::Symbol
    endogenous::Vector{Symbol}
    instruments::Vector{Symbol}
    covariates::Vector{Symbol}
    fe::Vector{Symbol}
    weights::Union{Nothing,Symbol}
    level::Float64
    estimand::String
    estimand_note::String
    first_stage::WeakIVDiagnostics
    tsls::IVEstimate
end

StatsAPI.coef(r::KClassEstimate) = r.coef
StatsAPI.vcov(r::KClassEstimate) = r.vcov
StatsAPI.coefnames(r::KClassEstimate) = r.coefnames
StatsAPI.nobs(r::KClassEstimate) = r.nobs
StatsAPI.dof_residual(r::KClassEstimate) = r.dof_residual
StatsAPI.confint(r::KClassEstimate; level::Real=r.level) =
    invoke(StatsAPI.confint, Tuple{CausalEstimate}, r; level=level)
estimand(r::KClassEstimate) = r.estimand

function method_name(r::KClassEstimate)
    r.method === :liml && return "LIML"
    r.method === :fuller && return @sprintf("Fuller (α = %g)", r.fuller_alpha)
    r.method === :hlim && return "HLIM (jackknife LIML)"
    r.method === :hful && return @sprintf("HFUL (jackknife Fuller, C = %g)",
                                          r.fuller_alpha)
    return @sprintf("k-class (κ = %.6g)", r.kappa)
end

function show_details(io::IO, r::KClassEstimate)
    println(io)
    @printf(io, "κ = %.6g; covariance: %s\n", r.kappa, r.se_type)
    for s in r.first_stage.first_stage
        @printf(io, "First stage (%s): F = %.2f (%s)\n", s.endogenous, s.F,
                r.first_stage.vcov_type)
    end
    @printf(io, "Instruments: %d; 2SLS estimate for comparison: %.4g (se %.3g)\n",
            length(r.instruments), r.tsls.coef[1], sqrt(r.tsls.vcov[1, 1]))
    println(io, "Estimand: ", r.estimand)
    println(io, "Note: ", r.estimand_note)
end

const _IV_KCLASS_METHODS = (:liml, :fuller, :kclass, :hlim, :hful)

"""
    kclass_iv(data, outcome, endogenous, instruments;
              method=:liml, kappa=nothing, fuller_alpha=1.0, se=:default,
              covariates=Symbol[], fe=Symbol[], weights=nothing,
              cluster=nothing, vcov=nothing, level=0.95,
              drop_singletons=true) -> KClassEstimate

k-class instrumental-variables estimators (LIML, Fuller, arbitrary κ) and their
jackknife versions (HLIM, HFUL), which are less biased than 2SLS with many or weak
instruments.

With many instruments relative to the sample size, or with weak instruments, 2SLS is
biased towards OLS: the first-stage fitted value ``P D`` contains the part of the
first-stage error that correlates with the structural error, and the bias grows
with the number of instruments. Under Bekker's (1994) asymptotics, in which the number
of instruments ``K`` grows proportionally with ``n``, 2SLS is inconsistent while
limited-information maximum likelihood (LIML) remains consistent under
homoskedasticity. The k-class family nests these estimators. With ``\\bar Y = [y, D]``,
the projection ``P`` on the excluded instruments and ``M = I - P``, all after
partialling out covariates, fixed effects and weights, the k-class estimator is

```math
\\hat\\beta(\\kappa) = [D'(I - \\kappa M) D]^{-1} D'(I - \\kappa M) y ,
```

so that ``\\kappa = 0`` is OLS and ``\\kappa = 1`` is 2SLS. LIML (Anderson and Rubin
1949) sets κ to the smallest root of
``\\det(\\bar Y'\\bar Y - \\kappa \\bar Y'M\\bar Y) = 0`` (so ``\\kappa \\ge 1``); it
is approximately median-unbiased but has no finite moments, which shows as occasional
extreme estimates when identification is weak. Fuller
(1977) uses ``\\kappa = \\kappa_{\\text{LIML}} - \\alpha/(n - L)``, where ``L`` counts the
instruments plus the included exogenous regressors and absorbed fixed-effect levels;
the estimator has finite moments, α = 1 makes it approximately unbiased and α = 4
approximately minimizes the mean squared error.

LIML and Fuller are not consistent with many instruments under heteroskedasticity.
Hausman, Newey, Woutersen, Chao and Swanson (2012) remove the own-observation terms
``P_{ii}`` that cause the problem. HLIM uses ``\\tilde\\alpha``, the smallest
eigenvalue of ``(\\bar Y'\\bar Y)^{-1}\\bar Y'(P - \\operatorname{diag}P)\\bar Y``; HFUL
uses ``\\hat\\alpha = [\\tilde\\alpha - (1 - \\tilde\\alpha)C/n] / [1 - (1 -
\\tilde\\alpha)C/n]`` with ``C`` = `fuller_alpha`; both then compute
``\\hat\\beta = [D'(P - \\operatorname{diag}P)D - \\hat\\alpha D'D]^{-1}
[D'(P - \\operatorname{diag}P)y - \\hat\\alpha D'y]``. The covariates and fixed effects
are partialled out before the jackknife step, and the many-instrument theory assumes
that such controls are few; with many controls (fixed effects with many levels) use
[`jive`](@ref) with `method = :ujive`, whose leave-one-out step accounts for the
controls.

Standard errors are selected with `se`:

1. `:standard` uses the covariance type chosen with `cluster` or `vcov` (HC1 by
   default) and treats κ as fixed: the homoskedastic
   ``\\hat\\sigma^2[D'(I - \\kappa M)D]^{-1}`` or the sandwich with scores
   ``[(I - \\kappa M)D]_i \\hat e_i``. These are conventional fixed-``K`` asymptotics.
2. `:bekker` (LIML and Fuller) gives the Bekker (1994) standard errors, valid under
   homoskedasticity when ``K`` grows with ``n``. With the residual ``\\hat u``,
   ``\\hat\\alpha = \\hat u'P\\hat u/\\hat u'\\hat u``,
   ``\\tilde X = D - \\hat u\\hat u'D/\\hat u'\\hat u`` and
   ``H = D'PD - \\hat\\alpha D'D``, the variance is ``V = H^{-1}\\Sigma_B H^{-1}`` with
   ``\\Sigma_B = \\hat\\sigma^2[(1 - \\hat\\alpha)^2\\tilde X'P\\tilde X +
   \\hat\\alpha^2\\tilde X'M\\tilde X]``.
3. `:hhn` (LIML and Fuller) adds the corrections of Hansen, Hausman and Newey (2008)
   for non-normal errors (third and fourth moments). With the full regressor matrix
   ``X = [D, W]``, the projection ``P`` on all instruments ``[Z, W]`` (diagonal
   ``p_{ii}``, rank ``K``), ``\\tau = K/n``, ``\\kappa_p = \\sum_i p_{ii}^2/K``,
   ``\\hat\\Upsilon = PX`` and ``\\hat V = (I - P)\\tilde X``,
   ```math
   \\Sigma = \\Sigma_B + \\hat A + \\hat A' + \\hat B, \\quad
   \\hat A = \\sum_i (p_{ii} - \\tau)\\hat\\Upsilon_i
             \\Big(\\sum_j \\hat u_j^2 \\hat V_j / n\\Big)', \\quad
   \\hat B = \\frac{K(\\kappa_p - \\tau)}{n(1 - 2\\tau + \\kappa_p\\tau)}
             \\sum_i (\\hat u_i^2 - \\hat\\sigma^2)\\hat V_i\\hat V_i' ,
   ```
   and ``V = H^{-1}\\Sigma H^{-1}`` is reported for the endogenous block. It is valid
   under homoskedasticity with many, possibly weak, instruments and non-Gaussian
   errors; the corrections are small when ``K/n`` is small. The controls enter
   ``W`` as a dense basis, so the number of fixed-effect levels must be moderate.
4. `:many_robust` (HLIM and HFUL) is the many-instrument, heteroskedasticity-robust
   variance of Hausman et al. (2012). With the residual ``\\hat\\varepsilon``,
   ``\\hat X = D - \\hat\\varepsilon\\hat\\varepsilon'D
   /\\hat\\varepsilon'\\hat\\varepsilon``,
   ``\\dot X = P\\hat X`` and ``H`` as in the estimator,
   ```math
   \\Sigma = \\sum_i \\hat\\varepsilon_i^2 (\\dot X_i - P_{ii}\\hat X_i)
             (\\dot X_i - P_{ii}\\hat X_i)'
           + \\sum_{i \\ne j} P_{ij}^2 \\hat X_i \\hat\\varepsilon_i
             \\hat\\varepsilon_j \\hat X_j' ,
   \\qquad V = H^{-1}\\Sigma H^{-1} .
   ```
   Clustering is not supported by this variance.

For inference that is robust to weak
identification as well, use the jackknife Anderson–Rubin test of Mikusheva and Sun
(2022), `weak_iv_test(r.tsls; method = :jackknife_ar)` ([`weak_iv_test`](@ref)).

The estimand deserves care. These estimators are consistent for the coefficient of a
linear model with constant effects. With heterogeneous treatment effects, Kolesár
(2013) shows that the probability limit of LIML (and of the related estimators) is
not in general a non-negatively weighted average of the instrument-specific LATEs and
can lie outside their range, whereas 2SLS and UJIVE estimate the same convex
combination of LATEs; `estimand(r)` records this and `estimand(r.tsls)` the 2SLS
estimand. When effects are plausibly heterogeneous and a LATE-type interpretation is
wanted, [`jive`](@ref) with `method = :ujive` is the natural many-instrument
alternative.

# Arguments
- `data::AbstractDataFrame`: the data; rows with missing values in used columns are
  dropped.
- `outcome::Symbol`: the outcome ``y``.
- `endogenous`: the endogenous regressor(s) ``D``, a `Symbol` or vector.
- `instruments`: the excluded instruments ``Z``, a `Symbol` or vector, at least as
  many as endogenous regressors.

# Keywords
- `method::Symbol`: `:liml` (default), `:fuller`, `:kclass`, `:hlim` or `:hful`.
- `kappa::Union{Nothing,Real}`: the κ used with `method = :kclass` (required there,
  and an error with any other method); default `nothing`.
- `fuller_alpha::Real`: the Fuller constant α for `:fuller`, or C for `:hful`
  (default 1); must be positive.
- `se::Symbol`: covariance estimator. The default `:default` means `:standard` for
  LIML, Fuller and k-class and `:many_robust` for HLIM and HFUL. `:bekker` and `:hhn`
  are available for LIML and Fuller only; `:many_robust` is available (and is the
  only option) for HLIM and HFUL.
- `covariates`, `fe`, `weights`, `cluster`, `vcov`, `level`, `drop_singletons`: as in
  [`iv_regression`](@ref) (default no controls, unweighted, HC1, `level = 0.95`).
  `cluster`/`vcov` affect only the `:standard` standard errors; the many-instrument
  variances assume independent observations.

# Returns
- A [`KClassEstimate`](@ref); `estimate(r)`, `confint(r)` and `coeftable(r)` refer to
  the endogenous coefficients, `r.kappa` is the κ used and `r.tsls` the 2SLS fit.

# Examples
```julia
using DrSnow, DataFrames, StableRNGs
rng = StableRNG(11)
n, K = 1_000, 30
Z = randn(rng, n, K)
v = randn(rng, n)
d = Z * fill(0.08, K) .+ v                           # many weak instruments
y = 1.0 .* d .+ 0.7 .* v .+ 0.7 .* randn(rng, n)     # true effect 1
zs = [Symbol("z", j) for j in 1:K]
df = DataFrame(Z, zs)
df.y, df.d = y, d
r = kclass_iv(df, :y, :d, zs; method=:liml, vcov=Vcov.simple())
kclass_iv(df, :y, :d, zs; method=:fuller, se=:bekker)
kclass_iv(df, :y, :d, zs; method=:hful)        # many instruments, heteroskedastic
estimate(r), estimate(r.tsls)
```

# References
- Anderson, T. W., & Rubin, H. (1949). Estimation of the parameters of a single
  equation in a complete system of stochastic equations. *Annals of Mathematical
  Statistics*, 20(1), 46–63.
- Fuller, W. A. (1977). Some properties of a modification of the limited
  information estimator. *Econometrica*, 45(4), 939–953.
- Bekker, P. A. (1994). Alternative approximations to the distributions of
  instrumental variable estimators. *Econometrica*, 62(3), 657–681.
- Hansen, C., Hausman, J., & Newey, W. (2008). Estimation with many instrumental
  variables. *Journal of Business & Economic Statistics*, 26(4), 398–422.
- Hausman, J. A., Newey, W. K., Woutersen, T., Chao, J. C., & Swanson, N. R. (2012).
  Instrumental variable estimation with heteroskedasticity and many instruments.
  *Quantitative Economics*, 3(2), 211–255.
- Kolesár, M. (2013). Estimation in an instrumental variables model with treatment
  effect heterogeneity (Working Paper No. 2013-2). Princeton University, Department
  of Economics.
- Mikusheva, A., & Sun, L. (2022). Inference with many weak instruments. *Review of
  Economic Studies*, 89(5), 2663–2686.
"""
function kclass_iv(data::AbstractDataFrame, outcome::Symbol, endogenous, instruments;
                   method::Symbol=:liml, kappa::Union{Nothing,Real}=nothing,
                   fuller_alpha::Real=1.0, se::Symbol=:default, level::Real=0.95,
                   kwargs...)
    method in _IV_KCLASS_METHODS ||
        throw(ArgumentError("method must be one of $(_IV_KCLASS_METHODS), got :$method"))
    se in (:default, :standard, :bekker, :hhn, :many_robust) ||
        throw(ArgumentError("se must be :default, :standard, :bekker, :hhn or " *
                            ":many_robust"))
    if method === :kclass
        kappa === nothing && throw(ArgumentError("method = :kclass requires `kappa`"))
        isfinite(kappa) || throw(ArgumentError("kappa must be finite"))
    elseif kappa !== nothing
        throw(ArgumentError("`kappa` is only used with method = :kclass"))
    end
    fuller_alpha > 0 || throw(ArgumentError("fuller_alpha must be positive"))
    jack = method in (:hlim, :hful)
    se === :default && (se = jack ? :many_robust : :standard)
    if se in (:bekker, :hhn) && !(method in (:liml, :fuller))
        throw(ArgumentError("Bekker and Hansen–Hausman–Newey standard errors are " *
                            "available for LIML and Fuller"))
    end
    if se === :many_robust && !jack
        throw(ArgumentError("se = :many_robust is available for HLIM and HFUL (LIML " *
                            "and Fuller are inconsistent with many instruments under " *
                            "heteroskedasticity)"))
    end
    if jack && se === :standard
        throw(ArgumentError("HLIM / HFUL support se = :many_robust only"))
    end
    tsls = iv_regression(data, outcome, endogenous, instruments; level=level, kwargs...)
    des = tsls.design
    if se === :many_robust && des.vcov_kind === :cluster
        throw(ArgumentError("the many-instrument robust variance does not support " *
                            "clustering"))
    end
    y, D, Z = des.y, des.D, des.Z
    n, p = size(D)
    k = size(Z, 2)
    Q, _ = _iv_orthobasis(Z)
    Ybar = hcat(y, D)
    MYbar = Ybar - Q * (Q' * Ybar)
    L = k + des.k_exog + des.dof_fes
    K = p + des.k_exog + des.dof_fes          # regressors incl. controls
    falpha = NaN
    if jack
        h = vec(sum(abs2, Q; dims=2))
        PYbar = Q * (Q' * Ybar)
        A = Symmetric(Ybar' * PYbar - Ybar' * (h .* Ybar))
        α̃ = minimum(eigvals(A, Symmetric(Ybar' * Ybar)))
        κ = α̃
        if method === :hful
            falpha = float(fuller_alpha)
            c = (1 - α̃) * falpha / n
            κ = (α̃ - c) / (1 - c)
        end
        PD = PYbar[:, 2:end]
        Py = PYbar[:, 1]
        H = D' * PD - D' * (h .* D) - κ .* (D' * D)
        g = D' * Py - D' * (h .* y) - κ .* (D' * y)
        β = Symmetric(H) \ g
        e = y - D * β
        V = _iv_hful_vcov(Q, h, D, e, H)
        se_label = "many-instrument heteroskedasticity-robust (Hausman et al. 2012)"
        dof = float(max(1, n - K))
    else
        κ = if method === :kclass
            float(kappa)
        else
            κl = minimum(eigvals(Symmetric(Ybar' * Ybar), Symmetric(Ybar' * MYbar)))
            if method === :fuller
                falpha = float(fuller_alpha)
                n - L > 0 || throw(ArgumentError("not enough observations for Fuller"))
                κl - falpha / (n - L)
            else
                κl
            end
        end
        MD = MYbar[:, 2:end]
        My = MYbar[:, 1]
        Dk = D - κ .* MD                         # k-class instrument (I − κM)D
        H = Symmetric(Dk' * D)
        β = H \ (Dk' * y)
        e = y - D * β
        if se === :bekker
            V = _iv_bekker_vcov(Q, D, e, n - K)
            se_label = "Bekker (1994) many-instrument, homoskedastic"
            dof = float(max(1, n - K))
        elseif se === :hhn
            Qc = _iv_kclass_control_basis(data, tsls, kwargs)
            V = _iv_hhn_vcov(Qc, Q, D, e)
            se_label = "Hansen, Hausman & Newey (2008) many-instrument, " *
                       "homoskedastic, non-normal errors"
            dof = float(max(1, n - K))
        else
            Hinv = inv(H)
            if des.vcov_kind === :simple
                V = Matrix(Hinv) .* (sum(abs2, e) / _iv_resid_dof(des, p))
            else
                V = _iv_psd(Hinv * _iv_meat(des, Dk .* e, p) * Hinv)
            end
            se_label = _iv_vcov_label(des)
            dof = _iv_ref_dof(des, p)
        end
    end
    all(isfinite, β) || error("k-class estimate is not finite (singular system)")
    est, note = _iv_kclass_estimand(method, tsls)
    return KClassEstimate(β, Matrix(Symmetric(V)), string.(tsls.endogenous), n, dof,
                          method, κ, falpha, se_label, outcome, tsls.endogenous,
                          tsls.instruments, tsls.covariates, tsls.fe, tsls.weights,
                          float(level), est, note, tsls.first_stage, tsls)
end

function _iv_kclass_estimand(method::Symbol, tsls::IVEstimate)
    if method === :kclass
        return ("k-class coefficient (constant-effects model)",
                "The k-class estimator targets the coefficient of a linear model with " *
                "constant effects; with heterogeneous effects only κ = 1 (2SLS) has " *
                "the weighted-average-of-LATEs interpretation (see `estimand(r.tsls)`).")
    end
    return ("structural coefficient (constant-effects model)",
            "Consistent for the coefficient of a linear model with constant effects. " *
            "With heterogeneous effects the probability limit of LIML-type " *
            "estimators is not in general a non-negatively weighted average of " *
            "LATEs (Kolesár 2013); 2SLS / UJIVE target " * tsls.estimand * ".")
end

"""Bekker (1994) / Hansen–Hausman–Newey (2008) Σ_B variance on the partialled data."""
function _iv_bekker_vcov(Q, D, u, dof)
    dof > 0 || throw(ArgumentError("not enough observations for Bekker standard errors"))
    σ2 = sum(abs2, u) / dof
    Pu = Q * (Q' * u)
    α = dot(u, Pu) / sum(abs2, u)
    Xt = D - u * ((u' * D) ./ sum(abs2, u))
    PXt = Q * (Q' * Xt)
    PD = Q * (Q' * D)
    H = Symmetric(D' * PD - α .* (D' * D))
    Σ = σ2 .* ((1 - α)^2 .* (Xt' * PXt) .+ α^2 .* (Xt' * (Xt - PXt)))
    Hinv = inv(H)
    return _iv_psd(Hinv * Σ * Hinv)
end

"""
`u' ((Qa Qa') ∘ (Qb Qb')) v` without forming n × n matrices:
`Σ_{a,b} (Σᵢ uᵢ Qa_ia Qb_ib)(Σⱼ vⱼ Qa_ja Qb_jb)`.
"""
function _iv_hadamard_quad(Qa::AbstractMatrix, Qb::AbstractMatrix, u::AbstractVector,
                           v::AbstractVector)
    Mu = Qa' * (u .* Qb)
    Mv = u === v ? Mu : Qa' * (v .* Qb)
    return sum(Mu .* Mv)
end

"""
Many-instrument heteroskedasticity-robust variance (Hausman et al. 2012) for
jackknife k-class estimators with `P = QQ'`, `h = diag P`.
"""
function _iv_hful_vcov(Q, h, D, e, H)
    p = size(D, 2)
    Xh = D - e * ((e' * D) ./ sum(abs2, e))
    Xd = Q * (Q' * Xh) .- h .* Xh              # Σ_{j≠i} P_ij X̂_j
    Σ = Xd' * (Xd .* (e .^ 2))
    A = Xh .* e
    for a in 1:p, b in a:p
        # Σ_{i≠j} P_ij² A_ia A_jb = A_a'(P∘P)A_b − Σ_i P_ii² A_ia A_ib
        s = _iv_hadamard_quad(Q, Q, A[:, a], A[:, b]) - sum(h .^ 2 .* A[:, a] .* A[:, b])
        Σ[a, b] += s
        a != b && (Σ[b, a] += s)
    end
    Hinv = inv(Symmetric(Matrix(H)))
    return _iv_psd(Hinv * Σ * Hinv)
end

"""
Orthonormal (weighted) basis of the control space (intercept / covariates / fixed
effects) of a fitted specification, as a dense matrix (for the HHN variance).
"""
function _iv_kclass_control_basis(data, tsls::IVEstimate, kwargs)
    kw = Dict{Symbol,Any}(kwargs)
    vce = _iv_vcov_estimator(get(kw, :cluster, nothing), get(kw, :vcov, nothing))
    used = unique(vcat(tsls.outcome, tsls.endogenous, tsls.instruments, tsls.covariates,
                       tsls.fe, _iv_cluster_names(vce),
                       tsls.weights === nothing ? Symbol[] : [tsls.weights]))
    sub = disallowmissing(data[tsls.design.esample, used])
    n = nrow(sub)
    sw = tsls.weights === nothing ? ones(n) : sqrt.(Float64.(sub[!, tsls.weights]))
    ctrl, _ = _iv_ctrl_basis(sub, tsls.covariates, tsls.fe, sw)
    l = sum(size(b, 2) for b in ctrl; init=0)
    n * l <= _IV_MAX_DENSE ||
        throw(ArgumentError("too many control / fixed-effect columns ($l) for the " *
                            "Hansen–Hausman–Newey variance (dense n × $l basis)"))
    return l == 0 ? zeros(n, 0) : hcat([Matrix{Float64}(b) for b in ctrl]...)
end

"""
Hansen, Hausman & Newey (2008) variance `Λ̂ = Ĥ⁻¹Σ̂Ĥ⁻¹`, `Σ̂ = Σ̂_B + Â + Â' + B̂`,
for a k-class estimator with residual `u` on the partialled data (`D` partialled,
`Qz` orthonormal basis of the partialled instruments), computed with the full
regressor matrix `X = [D, Qc]` and instrument projection `P = QcQc' + QzQz'`
(`Qc` an orthonormal basis of the controls). Returns the endogenous block.
"""
function _iv_hhn_vcov(Qc::AbstractMatrix, Qz::AbstractMatrix, D::AbstractMatrix,
                      u::AbstractVector)
    n, p = size(D)
    l = size(Qc, 2)
    G = p + l
    Kin = size(Qz, 2) + l                                # rank of [Z, W]
    n - G > 0 || throw(ArgumentError("not enough observations for HHN standard errors"))
    X = hcat(D, Qc)
    proj(v) = Qz * (Qz' * v) .+ (l > 0 ? Qc * (Qc' * v) : zero(v))
    uu = sum(abs2, u)
    σ2 = uu / (n - G)
    α = dot(u, proj(u)) / uu
    Xt = X .- u * ((u' * X) ./ uu)
    PX = proj(X)
    PXt = proj(Xt)
    Vh = Xt .- PXt                                       # (I − P) X̃
    H = Symmetric(X' * PX .- α .* (X' * X))
    ΣB = σ2 .* ((1 - α)^2 .* (Xt' * PXt) .+ α^2 .* (Xt' * Vh))
    ptt = vec(sum(abs2, Qz; dims=2)) .+ (l > 0 ? vec(sum(abs2, Qc; dims=2)) : 0.0)
    τ = Kin / n
    κ = sum(abs2, ptt) / Kin
    A = (PX' * (ptt .- τ)) * ((Vh' * (u .^ 2)) ./ n)'
    B = Kin * (κ - τ) .* (Vh' * (Vh .* (u .^ 2 .- σ2))) ./ (n * (1 - 2τ + κ * τ))
    Σ = ΣB .+ A .+ A' .+ B
    Hi = inv(H)
    Λ = Hi * Σ * Hi
    return _iv_psd(Λ[1:p, 1:p])
end
