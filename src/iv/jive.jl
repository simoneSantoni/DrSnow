# Jackknife IV estimators: JIVE1 / JIVE2 (Angrist, Imbens & Krueger 1999) and UJIVE
# (Kolesár 2013), with standard (fixed-instrument) and many-instrument robust
# standard errors.
#
# Unlike the other IV estimators, the leave-one-out steps need the *un-partialled*
# first stage (the leverage of the full instrument set [Z, W] and of the controls W
# separately), so this file builds its own design: data are multiplied by sqrt(w)
# and the control space (fixed effects + covariates) is represented by an
# orthonormal basis stored in blocks — a sparse normalized dummy matrix for one
# fixed effect, a dense basis for several — so that projections, leverages and
# Hadamard-product quadratic forms of projection matrices can be computed without
# n × n matrices.

"""Internal: design for leave-one-out estimators (see `src/iv/jive.jl`)."""
struct _IVJiveDesign
    yt::Vector{Float64}              # sqrt(w) y
    Dt::Matrix{Float64}              # sqrt(w) D
    yp::Vector{Float64}              # yt with the control space projected out
    Dp::Matrix{Float64}
    ctrl::Vector{AbstractMatrix{Float64}}   # orthonormal blocks spanning controls
    Qz::Matrix{Float64}              # orthonormal basis of the partialled instruments
    hW::Vector{Float64}              # leverage of the controls
    hZ::Vector{Float64}              # leverage of instruments + controls
    base::_IVDesign                  # partialled design (covariance helpers)
end

_iv_ctrl_proj(ctrl, v) = isempty(ctrl) ? zero(v) : sum(b * (b' * v) for b in ctrl)

function _iv_ctrl_lev(ctrl, n)
    h = zeros(n)
    for b in ctrl
        h .+= Vector{Float64}(vec(sum(abs2, b; dims=2)))
    end
    return h
end

const _IV_MAX_DENSE = 60_000_000     # max entries of a dense dummy matrix

"""
Orthonormal blocks spanning the (weighted) fixed-effect and covariate space, plus
the covariate rank `k_exog` (after absorbing fixed effects).
"""
function _iv_ctrl_basis(sub, covariates::Vector{Symbol}, fe::Vector{Symbol},
                        sw::Vector{Float64})
    n = nrow(sub)
    ctrl = AbstractMatrix{Float64}[]
    if length(fe) == 1
        g = _iv_codes(sub[!, fe[1]])
        G = maximum(g)
        Wg = zeros(G)
        for i in 1:n
            Wg[g[i]] += sw[i]^2
        end
        push!(ctrl, sparse(1:n, g, sw ./ sqrt.(Wg[g]), n, G))
    elseif length(fe) > 1
        codes = [_iv_codes(sub[!, f]) for f in fe]
        L = sum(maximum.(codes))
        n * L <= _IV_MAX_DENSE ||
            throw(ArgumentError("too many fixed-effect levels ($L) for the leave-one-" *
                                "out estimators with several fixed effects (dense " *
                                "basis of $n × $L); use one fixed effect or fewer levels"))
        M = zeros(n, L)
        off = 0
        for c in codes
            for i in 1:n
                M[i, off + c[i]] = sw[i]
            end
            off += maximum(c)
        end
        Qf, r = _iv_orthobasis(M)
        r > 0 && push!(ctrl, Qf)
    end
    Wc = _iv_exog_matrix(sub, covariates, isempty(fe)) .* sw
    k_exog = 0
    if size(Wc, 2) > 0
        Wp = Wc - _iv_ctrl_proj(ctrl, Wc)
        Qw, k_exog = _iv_orthobasis(Wp)
        k_exog > 0 && push!(ctrl, Qw)
    end
    return ctrl, k_exog
end

"""
    _iv_jive_design(sub, outcome, endo, Zraw, covariates, fe, weights, vce, esample)

`sub` is the complete-case estimation sample (fixed-effect singletons removed).
`Zraw` is the n × k instrument matrix (need not have full column rank).
"""
function _iv_jive_design(sub::AbstractDataFrame, outcome::Symbol, endo::Vector{Symbol},
                         Zraw::AbstractMatrix, covariates::Vector{Symbol},
                         fe::Vector{Symbol}, weights, vce, esample::BitVector)
    n = nrow(sub)
    w = weights === nothing ? ones(n) : Float64.(sub[!, weights])
    all(>(0), w) || throw(ArgumentError("weights must be strictly positive"))
    sw = sqrt.(w)
    yt = Float64.(sub[!, outcome]) .* sw
    Dt = Matrix{Float64}(hcat([Float64.(sub[!, c]) for c in endo]...)) .* sw
    Zt = Matrix{Float64}(Zraw) .* sw
    ctrl, k_exog = _iv_ctrl_basis(sub, covariates, fe, sw)
    Zp = Zt - _iv_ctrl_proj(ctrl, Zt)
    Qz, kz = _iv_orthobasis(Zp)
    kz > 0 || throw(ArgumentError("the instruments are collinear with the controls"))
    yp = yt - _iv_ctrl_proj(ctrl, yt)
    Dp = Dt - _iv_ctrl_proj(ctrl, Dt)
    _iv_check_rank(Dp, "endogenous regressors", "after partialling out the controls")
    hW = _iv_ctrl_lev(ctrl, n)
    hZ = hW .+ vec(sum(abs2, Qz; dims=2))
    if maximum(hZ) > 1 - 1e-9
        throw(ArgumentError("some observations have first-stage leverage 1 (e.g. an " *
                            "instrument category or fixed-effect group with a single " *
                            "observation); leave-one-out estimates are not defined " *
                            "for them — drop such observations"))
    end
    kind = _iv_vcov_kind(vce)
    groups = Vector{Int}[_iv_codes(sub[!, c]) for c in _iv_cluster_names(vce)]
    if kind === :cluster
        minimum(maximum.(groups)) >= 2 ||
            throw(ArgumentError("cluster-robust inference needs at least 2 clusters"))
    end
    dof_fes = _iv_dof_fes(sub, fe, groups)
    base = _IVDesign(yp, Dp, Qz, n, k_exog, dof_fes, kind, groups, sw, esample)
    return _IVJiveDesign(yt, Dt, yp, Dp, ctrl, Qz, hW, hZ, base)
end

"""`P_[Z,W] v` for the jackknife design."""
_iv_jive_projZ(jd::_IVJiveDesign, v) =
    _iv_ctrl_proj(jd.ctrl, v) .+ jd.Qz * (jd.Qz' * v)

"""Jackknife first-stage fitted values (the constructed instrument)."""
function _iv_jive_instrument(jd::_IVJiveDesign, method::Symbol)
    Dt, hZ, hW = jd.Dt, jd.hZ, jd.hW
    PZD = _iv_jive_projZ(jd, Dt)
    if method === :jive1
        return (PZD .- hZ .* Dt) ./ (1 .- hZ)
    elseif method === :jive2
        return PZD .- hZ .* Dt
    elseif method === :ujive
        PWD = _iv_ctrl_proj(jd.ctrl, Dt)
        return (PZD .- hZ .* Dt) ./ (1 .- hZ) .- (PWD .- hW .* Dt) ./ (1 .- hW)
    end
    throw(ArgumentError("method must be :jive1, :jive2 or :ujive, got :$method"))
end

"""
Point estimate, covariance, residuals and constructed instrument of a jackknife IV
estimator. JIVE1 / JIVE2 are IV regressions of y on [D, W] with instruments
[D̂, W] (AIK 1999); UJIVE is `(X̂'D)⁻¹X̂'y` with `X̂ = G D` (Kolesár 2013).
"""
function _iv_jive_fit(jd::_IVJiveDesign, method::Symbol, se::Symbol)
    des = jd.base
    p = size(jd.Dt, 2)
    X̂ = _iv_jive_instrument(jd, method)
    if method === :ujive
        Zi = X̂
        A = X̂' * jd.Dt
        β = A \ (X̂' * jd.yt)
    else
        Zi = X̂ .- _iv_ctrl_proj(jd.ctrl, X̂)       # M_W X̂
        A = Zi' * jd.Dp
        β = A \ (Zi' * jd.yp)
    end
    all(isfinite, β) || error("jackknife IV estimate is not finite (zero first stage)")
    e = jd.yp .- jd.Dp * β
    Ainv = inv(A)
    if se === :many_robust
        p == 1 || throw(ArgumentError("many-instrument robust standard errors require " *
                                      "one endogenous regressor"))
        method === :ujive ||
            throw(ArgumentError("se = :many_robust is available for UJIVE only"))
        des.vcov_kind === :cluster &&
            throw(ArgumentError("se = :many_robust does not support clustering"))
        V = _iv_ujive_many_vcov(jd, X̂[:, 1], e, A[1, 1])
        label = "many-instrument heteroskedasticity-robust"
        dof = Inf
    elseif des.vcov_kind === :simple
        σ2 = sum(abs2, e) / _iv_resid_dof(des, p)
        V = σ2 .* (Ainv * (Zi' * Zi) * Ainv')
        label = "homoskedastic"
        dof = _iv_ref_dof(des, p)
    else
        V = _iv_psd(Ainv * _iv_meat(des, Zi .* e, p) * Ainv')
        label = _iv_vcov_label(des)
        dof = _iv_ref_dof(des, p)
    end
    return β, Matrix(Symmetric(V)), label, dof, e, X̂
end

"""
Many-instrument robust variance for UJIVE with one endogenous regressor:
`[Σᵢ ε̂ᵢ² X̂ᵢ² + Σ_{i≠j} Gᵢⱼ Gⱼᵢ (Dᵢ ε̂ᵢ)(Dⱼ ε̂ⱼ)] / (X̂'D)²`, where
`G = (I − D_Z)⁻¹(P_Z − D_Z) − (I − D_W)⁻¹(P_W − D_W)` has a zero diagonal; `P_Z` is
the projection on the instruments and controls `[Z, W]`, `P_W` on the controls `W`,
and `D_Z`, `D_W` are their diagonals.
"""
function _iv_ujive_many_vcov(jd::_IVJiveDesign, xh::Vector{Float64},
                             e::Vector{Float64}, A::Real)
    a = jd.Dt[:, 1] .* e
    s1 = 1 ./ (1 .- jd.hZ)
    s0 = 1 ./ (1 .- jd.hW)
    u = s1 .* a
    v = s0 .* a
    d = u .- v
    T = 0.0
    blocks = jd.ctrl
    Qz = jd.Qz
    # (u−v)'(P_W∘P_W)(u−v) + 2 u'(P_W∘P_z)(u−v) + u'(P_z∘P_z)u, P_W = Σ blocks
    for b1 in blocks, b2 in blocks
        T += _iv_hadamard_quad(b1, b2, d, d)
    end
    for b in blocks
        T += 2 * _iv_hadamard_quad(b, Qz, u, d)
    end
    T += _iv_hadamard_quad(Qz, Qz, u, u)
    T -= sum(a .^ 2 .* (s1 .* jd.hZ .- s0 .* jd.hW) .^ 2)   # remove i = j terms
    Σ = sum(abs2, xh .* e) + T
    Σ > 0 || error("many-instrument variance estimate is not positive")
    return fill(Σ / A^2, 1, 1)
end

"""
    JIVEEstimate <: CausalEstimate

Result of [`jive`](@ref): a jackknife IV estimate (JIVE1, JIVE2, UJIVE or CJIVE) of the
coefficients on the endogenous regressors.

Covariates and fixed effects are partialled out or enter the leave-one-out step as
controls; their coefficients are not reported. The 2SLS fit of the same specification
is kept in `tsls` for comparison, and the estimand label is taken from it, since under
heterogeneous effects UJIVE targets the same weighted average of LATEs as 2SLS. The
`StatsAPI` accessors `coef`, `vcov`, `stderror`, `confint`, `coeftable`, `nobs` and
`dof_residual` work on it, and `estimate(r)` returns the first endogenous coefficient.

# Fields
- `coef::Vector{Float64}`, `vcov::Matrix{Float64}`, `coefnames::Vector{String}`:
  endogenous coefficients, their covariance and names.
- `nobs::Int`, `dof_residual::Float64`: sample size and the degrees of freedom of the
  reference distribution (`Inf` for the many-instrument variance, i.e. a normal
  reference; ``G - 1`` under clustering).
- `method::Symbol`: `:jive1`, `:jive2`, `:ujive` or `:cjive`.
- `se_type::String`: the covariance estimator used.
- `outcome`, `endogenous`, `instruments`, `covariates`, `fe`, `weights`: the
  specification.
- `level::Float64`: default confidence level.
- `estimand::String`, `estimand_note::String`: the target parameter and its caveats.
- `first_stage::WeakIVDiagnostics`: first-stage diagnostics of the specification.
- `tsls::IVEstimate`: 2SLS on the same sample.
"""
struct JIVEEstimate <: CausalEstimate
    coef::Vector{Float64}
    vcov::Matrix{Float64}
    coefnames::Vector{String}
    nobs::Int
    dof_residual::Float64
    method::Symbol
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

StatsAPI.coef(r::JIVEEstimate) = r.coef
StatsAPI.vcov(r::JIVEEstimate) = r.vcov
StatsAPI.coefnames(r::JIVEEstimate) = r.coefnames
StatsAPI.nobs(r::JIVEEstimate) = r.nobs
StatsAPI.dof_residual(r::JIVEEstimate) = r.dof_residual
StatsAPI.confint(r::JIVEEstimate; level::Real=r.level) =
    invoke(StatsAPI.confint, Tuple{CausalEstimate}, r; level=level)
estimand(r::JIVEEstimate) = r.estimand
method_name(r::JIVEEstimate) = uppercase(string(r.method))

function show_details(io::IO, r::JIVEEstimate)
    println(io)
    println(io, "Covariance: ", r.se_type)
    @printf(io, "Instruments: %d; 2SLS estimate for comparison: %.4g (se %.3g)\n",
            length(r.instruments), r.tsls.coef[1], sqrt(r.tsls.vcov[1, 1]))
    println(io, "Estimand: ", r.estimand)
    println(io, "Note: ", r.estimand_note)
end

"""
    jive(data, outcome, endogenous, instruments; method=:ujive, se=:standard,
         covariates=Symbol[], fe=Symbol[], weights=nothing, cluster=nothing,
         vcov=nothing, level=0.95, drop_singletons=true) -> JIVEEstimate

Jackknife instrumental-variables estimators, which remove the own-observation bias of
2SLS with many instruments by predicting each unit's treatment from the other units'
first stage.

The many-instrument bias of 2SLS arises because the first-stage fitted value
``(PD)_i`` of unit ``i`` contains ``P_{ii}D_i`` and hence the unit's own first-stage
error, which is correlated with its structural error. Jackknife IV estimators (JIVE)
replace the fitted value by a leave-one-out prediction that excludes unit ``i`` from
the first stage, which removes the correlation and makes the estimator consistent
when the number of instruments grows with the sample size (Angrist, Imbens and
Krueger 1999; Chao, Swanson, Hausman, Newey and Woutersen 2012). They are the
standard estimators for designs with many group-indicator instruments, such as
examiner or judge designs ([`judge_iv`](@ref)).

Let ``P_Z`` be the projection on the instruments *and* the controls ``[Z, W]``, with
diagonal ``D_Z`` and leverages ``h^Z_i``, and ``P_W`` the projection on the controls
``W`` alone, with diagonal ``D_W`` and leverages ``h^W_i``. The methods are:

1. `:jive1` (Angrist, Imbens and Krueger 1999): the leave-one-out prediction
   ``\\hat D_i = [(P_Z D)_i - h^Z_i D_i]/(1 - h^Z_i)``, followed by IV of ``y`` on
   ``[D, W]`` with instruments ``[\\hat D, W]``.
2. `:jive2` (Angrist, Imbens and Krueger 1999): ``\\hat D_i = (P_Z D)_i - h^Z_i D_i``,
   then the same IV regression.
3. `:ujive` (Kolesár 2013; default): the constructed instrument is ``\\hat X = G D``
   with
   ```math
   G = (I - D_Z)^{-1}(P_Z - D_Z) - (I - D_W)^{-1}(P_W - D_W),
   ```
   the leave-one-out prediction from the full first stage minus the leave-one-out
   prediction from the controls alone, and ``\\hat\\beta = (\\hat X'D)^{-1}\\hat X'y``.
   Because the controls are also handled by leave-one-out, UJIVE remains consistent
   when the number of covariates grows with the sample (for instance, many fixed
   effects), whereas JIVE1 and JIVE2 with covariates do not.
4. `:cjive` (Frandsen, Leslie and McIntyre 2025): cluster jackknife IV. When errors
   are correlated within clusters, the leave-one-out prediction still uses the other
   observations of the unit's cluster, whose first-stage errors are correlated with
   its structural error, so the many-instrument bias is not removed. CJIVE predicts
   the treatment of every observation in cluster ``g`` from the first stage fitted
   without cluster ``g``, ``\\hat D_g = \\hat X_g - H_g(I - H_g)^{-1}\\hat e_g``, where
   ``\\hat X`` and ``\\hat e`` are the first-stage fitted values and residuals and
   ``H_g`` is the cluster block of the instrument projection, and sets
   ``\\hat\\beta = (\\hat D'D)^{-1}\\hat D'y``. Covariates, fixed effects and weights are
   partialled out on the full sample first, as in the authors' R package
   `clusterIV`. It requires one clustering variable (`cluster`), and every instrument
   must vary outside each cluster.

Covariates, one or several fixed effects (entered as dummies in ``W``) and weights are
supported. Observations with first-stage leverage one (for example, an instrument
category with a single observation) have no leave-one-out prediction and are not
allowed.

Standard errors are selected with `se`. The default `:standard` uses the covariance
type chosen with `cluster` or `vcov` (HC1 by default) and treats the jackknife
instrument as fixed: the scores are ``\\hat Z_i\\hat e_i`` with ``\\hat Z = M_W\\hat D``
(JIVE1, JIVE2) or ``\\hat X`` (UJIVE) and ``\\hat e = M_W(y - D\\hat\\beta)``, with the
small-sample factors of [`iv_regression`](@ref). For CJIVE the cluster-robust
covariance is ``A^{-1}(\\sum_g s_g s_g')A^{-\\top} \\cdot G/(G - 1)`` with
``A = \\hat D'D`` and ``s_g = \\sum_{i \\in g}\\hat D_i\\hat\\varepsilon_i``, and the
reference distribution is ``t(G - 1)``, as in `clusterIV`. The `:many_robust` option
(UJIVE with one endogenous regressor and independent observations) adds the
many-instrument term ``\\sum_{i \\ne j} G_{ij}G_{ji}(D_i\\hat e_i)(D_j\\hat e_j)`` to the
robust variance, following Chao et al. (2012), which keeps the standard errors valid
when the number of instruments grows with ``n``; the reference distribution is then
normal. None of these standard errors is robust to weak identification; with many
weak instruments use the jackknife Anderson–Rubin test of Mikusheva and Sun (2022),
`weak_iv_test(r.tsls; method = :jackknife_ar)` ([`weak_iv_test`](@ref)).

With a binary treatment, heterogeneous effects and valid instruments that satisfy
monotonicity, Kolesár (2013) shows that UJIVE estimates the same non-negatively
weighted average of LATEs as 2SLS, while LIML-type estimators ([`kclass_iv`](@ref))
need not. The estimand label is therefore taken from the 2SLS fit
(`estimand(r.tsls)`), including its caveats about covariates and several
instruments (see [`late_2sls`](@ref)); for JIVE1 and JIVE2 with covariates the note
adds that they are inconsistent when the number of covariates grows. Prefer UJIVE in
applications with many instruments and many controls; use CJIVE when the data are
clustered and instruments or errors are correlated within clusters.

# Arguments
- `data::AbstractDataFrame`: the data; rows with missing values in used columns are
  dropped.
- `outcome::Symbol`: the outcome ``y``.
- `endogenous`: the endogenous regressor(s) ``D``, a `Symbol` or vector.
- `instruments`: the excluded instruments ``Z``, a `Symbol` or vector (for example,
  group indicators); they need not have full column rank.

# Keywords
- `method::Symbol`: `:ujive` (default), `:jive1`, `:jive2` or `:cjive`.
- `se::Symbol`: `:standard` (default) or `:many_robust` (UJIVE, one endogenous
  regressor, no clustering). CJIVE supports `:standard` only, which for it is the
  cluster-robust variance described above.
- `covariates`, `fe`, `weights`, `cluster`, `vcov`, `level`, `drop_singletons`: as in
  [`iv_regression`](@ref) (default no controls, unweighted, HC1, `level = 0.95`,
  singletons dropped); `cluster` is required for `:cjive`.

# Returns
- A [`JIVEEstimate`](@ref); `estimate(r)`, `confint(r)` and `coeftable(r)` refer to
  the endogenous coefficients, and `r.tsls` is the 2SLS fit on the same sample.

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
df.y, df.d, df.x, df.g = y, d, randn(rng, n), rand(rng, 1:50, n)
r = jive(df, :y, :d, zs; covariates=[:x])                  # UJIVE
jive(df, :y, :d, zs; se=:many_robust)
jive(df, :y, :d, zs; method=:cjive, cluster=:g)
weak_iv_test(r.tsls; beta0=1.0, method=:jackknife_ar)
```

# References
- Angrist, J. D., Imbens, G. W., & Krueger, A. B. (1999). Jackknife instrumental
  variables estimation. *Journal of Applied Econometrics*, 14(1), 57–67.
- Chao, J. C., Swanson, N. R., Hausman, J. A., Newey, W. K., & Woutersen, T. (2012).
  Asymptotic distribution of JIVE in a heteroskedastic IV regression with many
  instruments. *Econometric Theory*, 28(1), 42–86.
- Kolesár, M. (2013). Estimation in an instrumental variables model with treatment
  effect heterogeneity (Working Paper No. 2013-2). Princeton University, Department
  of Economics.
- Mikusheva, A., & Sun, L. (2022). Inference with many weak instruments. *Review of
  Economic Studies*, 89(5), 2663–2686.
- Frandsen, B., Leslie, E., & McIntyre, S. (2025). Cluster jackknife instrumental
  variables estimation. *Review of Economics and Statistics*, advance online
  publication. https://doi.org/10.1162/rest.a.263
"""
function jive(data::AbstractDataFrame, outcome::Symbol, endogenous, instruments;
              method::Symbol=:ujive, se::Symbol=:standard, level::Real=0.95,
              covariates=Symbol[], fe=Symbol[], weights::Union{Nothing,Symbol}=nothing,
              cluster=nothing, vcov=nothing, drop_singletons::Bool=true)
    method in (:jive1, :jive2, :ujive, :cjive) ||
        throw(ArgumentError("method must be :jive1, :jive2, :ujive or :cjive, got " *
                            ":$method"))
    se in (:standard, :many_robust) ||
        throw(ArgumentError("se must be :standard or :many_robust"))
    tsls = iv_regression(data, outcome, endogenous, instruments; covariates=covariates,
                         fe=fe, weights=weights, cluster=cluster, vcov=vcov,
                         level=level, drop_singletons=drop_singletons)
    endo, inst = tsls.endogenous, tsls.instruments
    covs, fes = tsls.covariates, tsls.fe
    if method === :cjive
        se === :standard ||
            throw(ArgumentError("CJIVE uses cluster-robust standard errors (se = " *
                                ":standard)"))
        β, V, label, dof = _iv_cjive_fit(tsls.design)
        return JIVEEstimate(β, V, string.(endo), tsls.design.n, dof, method, label,
                            outcome, endo, inst, covs, fes, weights, float(level),
                            tsls.estimand,
                            tsls.estimand_note * " CJIVE predicts each observation's " *
                            "treatment from the first stage fitted without its " *
                            "cluster.", tsls.first_stage, tsls)
    end
    vce = _iv_vcov_estimator(cluster, vcov)
    esample = tsls.design.esample
    used = unique(vcat(outcome, endo, inst, covs, fes, _iv_cluster_names(vce),
                       weights === nothing ? Symbol[] : [weights]))
    sub = disallowmissing(data[esample, used])
    Zraw = hcat([Float64.(sub[!, c]) for c in inst]...)
    jd = _iv_jive_design(sub, outcome, endo, Zraw, covs, fes, weights, vce, esample)
    β, V, label, dof, _, _ = _iv_jive_fit(jd, method, se)
    note = tsls.estimand_note
    if method !== :ujive && !(isempty(covs) && isempty(fes))
        note *= " JIVE1/JIVE2 with covariates are inconsistent when the number of " *
                "covariates grows with the sample; UJIVE is not."
    end
    return JIVEEstimate(β, V, string.(endo), nrow(sub), dof, method, label, outcome,
                        endo, inst, covs, fes, weights, float(level), tsls.estimand,
                        note, tsls.first_stage, tsls)
end

"""
Cluster jackknife IV (Frandsen, Leslie & McIntyre 2025) on the partialled design:
the instrument of cluster `g` is the first-stage prediction fitted without cluster
`g`, `D̂_g = Z_g π̂_(−g) = X̂_g − H_g (I − H_g)⁻¹ ê_g` (`H_g` the within-cluster block of
the instrument projection), `β̂ = (D̂'D)⁻¹ D̂'y`, and the cluster-robust covariance
`A⁻¹ (Σ_g s_g s_g') A⁻ᵀ G/(G − 1)` with `A = D̂'D`, `s_g = Σ_{i∈g} D̂ᵢ ε̂ᵢ`.
"""
function _iv_cjive_fit(des::_IVDesign)
    des.vcov_kind === :cluster && length(des.groups) == 1 ||
        throw(ArgumentError("CJIVE requires one clustering variable (`cluster = :g`)"))
    g = des.groups[1]
    G = maximum(g)
    Q, _ = _iv_orthobasis(des.Z)
    D = des.D
    X̂ = Q * (Q' * D)
    ê = D .- X̂
    D̂ = copy(X̂)
    members = [Int[] for _ in 1:G]
    for (i, c) in enumerate(g)
        push!(members[c], i)
    end
    for rows in members
        Qg = Q[rows, :]
        Hg = Qg * Qg'
        Mg = Symmetric(Matrix(1.0I, length(rows), length(rows)) .- Hg)
        if minimum(eigvals(Mg)) < 1e-10
            throw(ArgumentError("the leave-cluster-out first stage is not defined for a " *
                                "cluster with $(length(rows)) observation(s): an " *
                                "instrument has no variation outside that cluster"))
        end
        D̂[rows, :] .-= Hg * (Mg \ ê[rows, :])
    end
    A = D̂' * D
    β = A \ (D̂' * des.y)
    all(isfinite, β) || error("CJIVE estimate is not finite (zero first stage)")
    ε̂ = des.y .- D * β
    Sg = _iv_group_sums(D̂ .* ε̂, g)
    Ai = inv(A)
    V = _iv_psd(Ai * (Sg' * Sg) * Ai' .* (G / (G - 1)))
    return β, Matrix(Symmetric(V)),
           "cluster-robust, leave-cluster-out instrument (CJIVE; $G clusters)",
           float(G - 1)
end

# ---------------------------------------------------------------------------
# Mikusheva & Sun (2022) jackknife Anderson–Rubin test
# ---------------------------------------------------------------------------

"""
Precompute the pieces of the Mikusheva–Sun jackknife AR statistic as polynomials in
β: numerator `N(β) = Σ_{i≠j} Pᵢⱼ eᵢ eⱼ` (quadratic) and the cross-fit variance
`Φ(β) = (2/K) Σ_{i≠j} P̃ᵢⱼ² [eᵢ(Me)ᵢ][eⱼ(Me)ⱼ]` (quartic), `e = y − βD`, with
`P̃ᵢⱼ² = Pᵢⱼ² / (MᵢᵢMⱼⱼ + Mᵢⱼ²)`. The n² pair sums are accumulated in blocks.
"""
function _iv_jar_parts(des::_IVDesign)
    size(des.D, 2) == 1 || throw(ArgumentError("the jackknife AR test requires one " *
                                               "endogenous regressor"))
    des.vcov_kind === :cluster &&
        throw(ArgumentError("the jackknife AR test assumes independent observations; " *
                            "it is not available with clustering"))
    Q, K = _iv_orthobasis(des.Z)
    y, d = des.y, des.D[:, 1]
    n = length(y)
    h = vec(sum(abs2, Q; dims=2))
    Py, Pd = Q * (Q' * y), Q * (Q' * d)
    My, Md = y .- Py, d .- Pd
    # numerator: e'Pe − Σ hᵢ eᵢ², e = y − βd  → coefficients of 1, β, β²
    N0 = dot(y, Py) - sum(h .* y .^ 2)
    N1 = -2 * (dot(d, Py) - sum(h .* y .* d))
    N2 = dot(d, Pd) - sum(h .* d .^ 2)
    # eᵢ (Me)ᵢ = fa + fb β + fc β²
    F = hcat(y .* My, -(y .* Md .+ d .* My), d .* Md)
    S = zeros(3, 3)
    m = 1 .- h
    bs = 1024
    for i0 in 1:bs:n
        I = i0:min(n, i0 + bs - 1)
        for j0 in 1:bs:n
            J = j0:min(n, j0 + bs - 1)
            Pb = Q[I, :] * Q[J, :]'
            W = Pb .^ 2 ./ (m[I] .* m[J]' .+ Pb .^ 2)
            if i0 == j0
                for t in eachindex(I)
                    W[t, t] = 0.0
                end
            end
            S .+= F[I, :]' * W * F[J, :]
        end
    end
    S .*= 2 / K
    return (N=(N0, N1, N2), S=Matrix(Symmetric(S)), K=K)
end

function _iv_jar_stat(parts, β::Real)
    if isinf(β)
        N = parts.N[3]
        Φ = parts.S[3, 3]
    else
        N = parts.N[1] + parts.N[2] * β + parts.N[3] * β^2
        v = [1.0, β, β^2]
        Φ = dot(v, parts.S * v)
    end
    Φ > 0 || return N > 0 ? Inf : 0.0
    return N / sqrt(parts.K * Φ)
end

# Weak-IV-robust tests are properties of the specification: forward to the 2SLS fit.
weak_iv_test(r::Union{KClassEstimate,JIVEEstimate}; kwargs...) =
    weak_iv_test(r.tsls; kwargs...)
weak_iv_confidence_set(r::Union{KClassEstimate,JIVEEstimate}; kwargs...) =
    weak_iv_confidence_set(r.tsls; kwargs...)
