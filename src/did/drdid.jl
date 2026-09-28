# Sant'Anna & Zhao (2020) doubly robust DiD, with IPW and outcome-regression
# variants, for panel data and repeated cross-sections.
#
# The kernels follow the estimators and influence functions of the R package DRDID
# (v1.3.0) line by line; see test/validation/did for the numerical comparison.
# They return (att, ψ) with ψ scaled so that att - θ ≈ mean(ψ).

const _DID_KERNEL_METHODS = (:dr_improved, :dr, :ipw, :ipw_unnormalized, :reg)

function _did_check_method(method)
    method in _DID_KERNEL_METHODS || throw(ArgumentError(
        "method must be one of $(join(repr.(_DID_KERNEL_METHODS), ", ")); " *
        "got $(repr(method))"))
    return method
end

# ---------------------------------------------------------------------------
# Nuisance models
# ---------------------------------------------------------------------------

function _did_rank_check(X, what)
    F = qr(X, ColumnNorm())
    r = count(abs.(diag(F.R)) .> 1e-10 * max(1.0, abs(F.R[1, 1])))
    r < size(X, 2) && throw(ArgumentError(
        "$what: covariate matrix is rank deficient (collinear covariates or no " *
        "variation within the subsample); remove some covariates"))
    return nothing
end

# Weighted least squares coefficients.
function _did_wls(X::AbstractMatrix, y::AbstractVector, w::AbstractVector, what)
    size(X, 1) >= size(X, 2) || throw(ArgumentError(
        "$what: fewer observations ($(size(X, 1))) than covariates ($(size(X, 2)))"))
    sw = sqrt.(w)
    Xw = X .* sw
    _did_rank_check(Xw, what)
    return Xw \ (y .* sw)
end

# Damped Newton minimization of a smooth strictly convex function. `fgh(γ)` returns
# (value, gradient, Hessian). Stops on the Newton decrement g'H⁻¹g (scale invariant):
# far from the optimum steps are damped by backtracking; near it full Newton steps are
# taken (quadratic convergence) and stagnation at the floating-point floor counts as
# convergence. Returns (γ, converged).
function _did_newton_min(fgh, γ; maxiter=200)
    f, g, H = fgh(γ)
    scale = max(1.0, abs(f))
    prevdec = Inf
    for _ in 1:maxiter
        step = try
            Symmetric(H) \ g
        catch
            return γ, false
        end
        dec = dot(g, step)
        (isfinite(dec) && dec > -sqrt(eps()) * scale) || return γ, false
        dec <= 1e-24 * scale && return γ, true
        if dec < 1e-8 * scale
            dec >= prevdec / 10 && return γ, true     # at the rounding floor
            γ = γ .- step
        else
            t = 1.0
            fnew = first(fgh(γ .- step))
            while !(fnew <= f - 1e-4 * t * dec) && t > 1e-12
                t /= 2
                fnew = first(fgh(γ .- t .* step))
            end
            t <= 1e-12 && return γ, false
            γ = γ .- t .* step
        end
        prevdec = dec
        f, g, H = fgh(γ)
        all(isfinite, γ) || return γ, false
    end
    return γ, false
end

# Weighted logistic regression by maximum likelihood (Newton–Raphson). A small
# dedicated solver avoids depending on keyword changes across GLM.jl versions; it is
# validated against R's glm-based DRDID to ~1e-12.
function _did_logit(X::AbstractMatrix, D::AbstractVector, w::AbstractVector)
    _did_rank_check(X .* sqrt.(w), "propensity score")
    d = float.(D)
    p̄ = sum(w .* d) / sum(w)
    (0 < p̄ < 1) || throw(ArgumentError("propensity score: treatment does not vary"))
    γ0 = zeros(size(X, 2))
    γ0[1] = log(p̄ / (1 - p̄))    # the first column is the intercept
    function fgh(γ)
        η = X * γ
        p = _did_logistic.(η)
        nll = -sum(w .* (d .* η .- log1p.(exp.(-abs.(η))) .- max.(η, 0)))
        return nll, -(X' * (w .* (d .- p))), X' * (X .* (w .* p .* (1 .- p)))
    end
    γ, ok = _did_newton_min(fgh, γ0)
    ok || throw(ArgumentError(
        "propensity score logit did not converge (perfect separation or lack of " *
        "overlap between groups is a likely reason)"))
    return γ, _did_logistic.(X * γ)
end

_did_logistic(x) = x >= 0 ? 1 / (1 + exp(-x)) : exp(x) / (1 + exp(x))

# Inverse probability tilting / calibrated propensity score (Graham, Pinto and Egel
# 2012; the "improved" DR estimator of Sant'Anna & Zhao 2020): minimizes the convex
#   L(γ) = mean(w * ((1 - D) exp(Xγ) - D Xγ)),
# whose first-order condition balances the ps/(1-ps)-weighted control covariate
# means with the treated means exactly. Started from the logit estimate (as in R).
function _did_ipt_pscore(X::AbstractMatrix, D::AbstractVector, w::AbstractVector)
    γ0, ps_logit = _did_logit(X, D, w)
    n = size(X, 1)
    d = float.(D)
    function fgh(γ)
        η = X * γ
        e = exp.(η)
        L = sum(w .* ((1 .- d) .* e .- d .* η)) / n
        return L, X' * (w .* ((1 .- d) .* e .- d)) ./ n,
               X' * (X .* (w .* (1 .- d) .* e)) ./ n
    end
    γ, ok = _did_newton_min(fgh, γ0)
    if !ok
        @warn "calibrated (IPT) propensity score did not converge; using the logit " *
              "propensity score instead (as R's DRDID does)"
        return ps_logit, false
    end
    return _did_logistic.(X * γ), true
end

function _did_pscore_parts(X, D, w, trim_level)
    _, ps = _did_logit(X, D, w)
    ps = min.(ps, 1 - 1e-6)
    trim = [D[i] == 1 ? ps[i] < 1.01 : ps[i] < trim_level for i in eachindex(ps)]
    n = size(X, 1)
    W = ps .* (1 .- ps) .* w
    XtWX = X' * (X .* W)
    Hinv = inv(Symmetric(XtWX)) .* n
    asy_ps = (w .* (D .- ps) .* X) * Hinv
    return ps, float.(trim), asy_ps
end

function _did_asy_ols(X, resid, wsub)
    n = size(X, 1)
    XpX = (X .* wsub)' * X ./ n
    return ((wsub .* resid) .* X) / Symmetric(XpX)
end

# ---------------------------------------------------------------------------
# Panel kernels: dy = Y_post - Y_pre, D = treatment-group indicator.
# ---------------------------------------------------------------------------

function _did_kernel_panel(method::Symbol, dy::AbstractVector, D::AbstractVector,
                           X::AbstractMatrix, w0::AbstractVector; trim_level=0.995)
    n = length(dy)
    D = float.(D)
    any(==(1), D) || throw(ArgumentError("no treated units in the 2×2 comparison"))
    any(==(0), D) || throw(ArgumentError("no comparison units in the 2×2 comparison"))
    w = w0 ./ mean(w0)
    ctrl = D .== 0
    if method === :reg
        β = _did_wls(X[ctrl, :], dy[ctrl], w[ctrl], "outcome regression")
        out = X * β
        w_treat = w .* D
        w_cont = w .* D
        att_treat = w_treat .* dy
        att_cont = w_cont .* out
        eta_treat = mean(att_treat) / mean(w_treat)
        eta_cont = mean(att_cont) / mean(w_cont)
        att = eta_treat - eta_cont
        asy = _did_asy_ols(X, dy .- out, w .* (1 .- D))
        inf_treat = (att_treat .- w_treat .* eta_treat) ./ mean(w_treat)
        M1 = X' * w_cont ./ n
        inf_control = (att_cont .- w_cont .* eta_cont .+ asy * M1) ./ mean(w_cont)
        return att, inf_treat .- inf_control
    elseif method === :ipw || method === :ipw_unnormalized
        ps, trim, asy_ps = _did_pscore_parts(X, D, w, trim_level)
        w_treat = trim .* w .* D
        w_cont = trim .* w .* ps .* (1 .- D) ./ (1 .- ps)
        att_treat = w_treat .* dy
        att_cont = w_cont .* dy
        if method === :ipw_unnormalized
            pD = mean(w .* D)
            att = mean(att_treat) / pD - mean(att_cont) / pD
            mom = X' * att_cont ./ n
            ψ = (att_treat .- att_cont .- asy_ps * mom .- w .* D .* att) ./ pD
            return att, ψ
        end
        mwt, mwc = mean(w_treat), mean(w_cont)
        eta_treat = mean(att_treat) / mwt
        eta_cont = mean(att_cont) / mwc
        att = eta_treat - eta_cont
        inf_treat = (att_treat .- w_treat .* eta_treat) ./ mwt
        M2 = X' * (w_cont .* (dy .- eta_cont)) ./ n
        inf_control = (att_cont .- w_cont .* eta_cont .+ asy_ps * M2) ./ mwc
        return att, inf_treat .- inf_control
    elseif method === :dr
        ps, trim, asy_ps = _did_pscore_parts(X, D, w, trim_level)
        β = _did_wls(X[ctrl, :], dy[ctrl], w[ctrl], "outcome regression")
        out = X * β
        w_treat = trim .* w .* D
        w_cont = trim .* w .* ps .* (1 .- D) ./ (1 .- ps)
        mwt, mwc = mean(w_treat), mean(w_cont)
        dr_treat = w_treat .* (dy .- out)
        dr_cont = w_cont .* (dy .- out)
        eta_treat = mean(dr_treat) / mwt
        eta_cont = mean(dr_cont) / mwc
        att = eta_treat - eta_cont
        asy_ols = _did_asy_ols(X, dy .- out, w .* (1 .- D))
        M1 = X' * w_treat ./ n
        M2 = X' * (w_cont .* (dy .- out .- eta_cont)) ./ n
        M3 = X' * w_cont ./ n
        inf_treat = (dr_treat .- w_treat .* eta_treat .- asy_ols * M1) ./ mwt
        inf_control = (dr_cont .- w_cont .* eta_cont .+ asy_ps * M2 .- asy_ols * M3) ./ mwc
        return att, inf_treat .- inf_control
    elseif method === :dr_improved
        ps, _ = _did_ipt_pscore(X, D, w)
        ps = min.(ps, 1 - 1e-6)
        trim = float.([D[i] == 1 ? ps[i] < 1.01 : ps[i] < trim_level for i in 1:n])
        wr = w .* ps ./ (1 .- ps)
        β = _did_wls(X[ctrl, :], dy[ctrl], wr[ctrl], "outcome regression")
        out = X * β
        summand = trim .* (1 .- (1 .- D) ./ (1 .- ps)) .* (dy .- out)
        pD = mean(D .* w)
        att = mean(w .* summand) / pD
        ψ = w .* trim .* (summand .- D .* att) ./ pD
        return att, ψ
    end
    _did_check_method(method)
end

# ---------------------------------------------------------------------------
# Repeated cross-section kernels.
# ---------------------------------------------------------------------------

function _did_kernel_rc(method::Symbol, y::AbstractVector, post::AbstractVector,
                        D::AbstractVector, X::AbstractMatrix, w0::AbstractVector;
                        trim_level=0.995)
    n = length(y)
    D = float.(D)
    post = float.(post)
    for (d, p, lbl) in ((1, 0, "treated/pre"), (1, 1, "treated/post"),
                        (0, 0, "comparison/pre"), (0, 1, "comparison/post"))
        any(i -> D[i] == d && post[i] == p, 1:n) ||
            throw(ArgumentError("no observations in the $lbl cell"))
    end
    w = w0 ./ mean(w0)
    cell(d, p) = (D .== d) .& (post .== p)
    fit_cell(d, p, wts, what) = X * _did_wls(X[cell(d, p), :], y[cell(d, p)],
                                             wts[cell(d, p)], what)
    asy_cell(d, p, fitted) = _did_asy_ols(X, y .- fitted,
                                          w .* (D .== d) .* (post .== p))
    if method === :reg
        out_pre = fit_cell(0, 0, w, "outcome regression (comparison, pre)")
        out_post = fit_cell(0, 1, w, "outcome regression (comparison, post)")
        w_tpre = w .* D .* (1 .- post)
        w_tpost = w .* D .* post
        w_cont = w .* D
        a_tpre = w_tpre .* y
        a_tpost = w_tpost .* y
        a_cont = w_cont .* (out_post .- out_pre)
        eta_tpre = mean(a_tpre) / mean(w_tpre)
        eta_tpost = mean(a_tpost) / mean(w_tpost)
        eta_cont = mean(a_cont) / mean(w_cont)
        att = (eta_tpost - eta_tpre) - eta_cont
        asy_pre = asy_cell(0, 0, out_pre)
        asy_post = asy_cell(0, 1, out_post)
        inf_treat = (a_tpost .- w_tpost .* eta_tpost) ./ mean(w_tpost) .-
                    (a_tpre .- w_tpre .* eta_tpre) ./ mean(w_tpre)
        M1 = X' * w_cont ./ n
        inf_control = (a_cont .- w_cont .* eta_cont .+ asy_post * M1 .- asy_pre * M1) ./
                      mean(w_cont)
        return att, inf_treat .- inf_control
    elseif method === :ipw || method === :ipw_unnormalized
        ps, trim, asy_ps = _did_pscore_parts(X, D, w, trim_level)
        if method === :ipw_unnormalized
            w_tpre = w .* D .* (1 .- post)
            w_tpost = w .* D .* post
            w_cpre = trim .* w .* ps .* (1 .- D) .* (1 .- post) ./ (1 .- ps)
            w_cpost = trim .* w .* ps .* (1 .- D) .* post ./ (1 .- ps)
            Π = mean(w .* D)
            λ = mean(w .* post)
            λ0 = mean(w .* (1 .- post))
            e_tpre = w_tpre .* y ./ (Π * λ0)
            e_tpost = w_tpost .* y ./ (Π * λ)
            e_cpre = w_cpre .* y ./ (Π * λ0)
            e_cpost = w_cpost .* y ./ (Π * λ)
            a_tpre, a_tpost, a_cpre, a_cpost = mean(e_tpre), mean(e_tpost),
                                               mean(e_cpre), mean(e_cpost)
            att = (a_tpost - a_tpre) - (a_cpost - a_cpre)
            infl(e, a, lam, s) = e .- a .- (w .* D .- Π) .* a ./ Π .-
                                 (w .* s .- lam) .* a ./ lam
            i_tpost = infl(e_tpost, a_tpost, λ, post)
            i_tpre = infl(e_tpre, a_tpre, λ0, 1 .- post)
            i_cpost = infl(e_cpost, a_cpost, λ, post)
            i_cpre = infl(e_cpre, a_cpre, λ0, 1 .- post)
            mom = X' * (-e_cpost) ./ n .- X' * (-e_cpre) ./ n
            ψ = (i_tpost .- i_tpre) .- (i_cpost .- i_cpre) .+ asy_ps * mom
            return att, ψ
        end
        w_tpre = trim .* w .* D .* (1 .- post)
        w_tpost = trim .* w .* D .* post
        w_cpre = trim .* w .* ps .* (1 .- D) .* (1 .- post) ./ (1 .- ps)
        w_cpost = trim .* w .* ps .* (1 .- D) .* post ./ (1 .- ps)
        m = mean.((w_tpre, w_tpost, w_cpre, w_cpost))
        e = (w_tpre .* y ./ m[1], w_tpost .* y ./ m[2], w_cpre .* y ./ m[3],
             w_cpost .* y ./ m[4])
        a = mean.(e)
        att = (a[2] - a[1]) - (a[4] - a[3])
        i_tpre = e[1] .- w_tpre .* a[1] ./ m[1]
        i_tpost = e[2] .- w_tpost .* a[2] ./ m[2]
        i_cpre = e[3] .- w_cpre .* a[3] ./ m[3]
        i_cpost = e[4] .- w_cpost .* a[4] ./ m[4]
        M2pre = X' * (w_cpre .* (y .- a[3])) ./ n ./ m[3]
        M2post = X' * (w_cpost .* (y .- a[4])) ./ n ./ m[4]
        inf_cont = (i_cpost .- i_cpre) .+ asy_ps * (M2post .- M2pre)
        return att, (i_tpost .- i_tpre) .- inf_cont
    elseif method === :dr || method === :dr_improved
        improved = method === :dr_improved
        if improved
            ps, _ = _did_ipt_pscore(X, D, w)
            ps = min.(ps, 1 - 1e-6)
            trim = float.([D[i] == 1 ? ps[i] < 1.01 : ps[i] < trim_level for i in 1:n])
            asy_ps = nothing
            wr = w .* ps ./ (1 .- ps)
            oc_pre = fit_cell(0, 0, wr, "outcome regression (comparison, pre)")
            oc_post = fit_cell(0, 1, wr, "outcome regression (comparison, post)")
        else
            ps, trim, asy_ps = _did_pscore_parts(X, D, w, trim_level)
            oc_pre = fit_cell(0, 0, w, "outcome regression (comparison, pre)")
            oc_post = fit_cell(0, 1, w, "outcome regression (comparison, post)")
        end
        oc = post .* oc_post .+ (1 .- post) .* oc_pre
        ot_pre = fit_cell(1, 0, w, "outcome regression (treated, pre)")
        ot_post = fit_cell(1, 1, w, "outcome regression (treated, post)")
        tt = improved ? ones(n) : trim     # R trims treated weights only in :dr
        w_tpre = tt .* w .* D .* (1 .- post)
        w_tpost = tt .* w .* D .* post
        w_cpre = trim .* w .* ps .* (1 .- D) .* (1 .- post) ./ (1 .- ps)
        w_cpost = trim .* w .* ps .* (1 .- D) .* post ./ (1 .- ps)
        w_d = tt .* w .* D
        w_dt1 = tt .* w .* D .* post
        w_dt0 = tt .* w .* D .* (1 .- post)
        mw = mean.((w_tpre, w_tpost, w_cpre, w_cpost, w_d, w_dt1, w_dt0))
        e_tpre = w_tpre .* (y .- oc) ./ mw[1]
        e_tpost = w_tpost .* (y .- oc) ./ mw[2]
        e_cpre = w_cpre .* (y .- oc) ./ mw[3]
        e_cpost = w_cpost .* (y .- oc) ./ mw[4]
        e_dpost = w_d .* (ot_post .- oc_post) ./ mw[5]
        e_dt1post = w_dt1 .* (ot_post .- oc_post) ./ mw[6]
        e_dpre = w_d .* (ot_pre .- oc_pre) ./ mw[5]
        e_dt0pre = w_dt0 .* (ot_pre .- oc_pre) ./ mw[7]
        a = mean.((e_tpre, e_tpost, e_cpre, e_cpost, e_dpost, e_dt1post, e_dpre,
                   e_dt0pre))
        att = (a[2] - a[1]) - (a[4] - a[3]) + (a[5] - a[6]) - (a[7] - a[8])
        i_tpre = e_tpre .- w_tpre .* a[1] ./ mw[1]
        i_tpost = e_tpost .- w_tpost .* a[2] ./ mw[2]
        i_cpre = e_cpre .- w_cpre .* a[3] ./ mw[3]
        i_cpost = e_cpost .- w_cpost .* a[4] ./ mw[4]
        inf_eff = (e_dpost .- w_d .* a[5] ./ mw[5]) .-
                  (e_dt1post .- w_dt1 .* a[6] ./ mw[6]) .-
                  ((e_dpre .- w_d .* a[7] ./ mw[5]) .- (e_dt0pre .- w_dt0 .* a[8] ./ mw[7]))
        if improved
            return att, (i_tpost .- i_tpre) .- (i_cpost .- i_cpre) .+ inf_eff
        end
        asy_cpre = asy_cell(0, 0, oc_pre)
        asy_cpost = asy_cell(0, 1, oc_post)
        asy_tpre = asy_cell(1, 0, ot_pre)
        asy_tpost = asy_cell(1, 1, ot_post)
        M1post = -(X' * w_tpost) ./ n ./ mw[2]
        M1pre = -(X' * w_tpre) ./ n ./ mw[1]
        inf_treat_or = asy_cpost * M1post .+ asy_cpre * M1pre
        M2pre = X' * (w_cpre .* (y .- oc .- a[3])) ./ n ./ mw[3]
        M2post = X' * (w_cpost .* (y .- oc .- a[4])) ./ n ./ mw[4]
        inf_cont_ps = asy_ps * (M2post .- M2pre)
        M3post = -(X' * w_cpost) ./ n ./ mw[4]
        M3pre = -(X' * w_cpre) ./ n ./ mw[3]
        inf_cont_or = asy_cpost * M3post .+ asy_cpre * M3pre
        mom_post = X' * (w_d ./ mw[5] .- w_dt1 ./ mw[6]) ./ n
        mom_pre = X' * (w_d ./ mw[5] .- w_dt0 ./ mw[7]) ./ n
        inf_or = (asy_tpost .- asy_cpost) * mom_post .- (asy_tpre .- asy_cpre) * mom_pre
        inf_treat = i_tpost .- i_tpre .+ inf_treat_or
        inf_cont = i_cpost .- i_cpre .+ inf_cont_ps .+ inf_cont_or
        return att, inf_treat .- inf_cont .+ inf_eff .+ inf_or
    end
    _did_check_method(method)
end

# ---------------------------------------------------------------------------
# User-facing two-period estimator
# ---------------------------------------------------------------------------

function _did_design_matrix(df, covariates)
    n = nrow(df)
    X = ones(n, 1 + length(covariates))
    for (j, c) in enumerate(covariates)
        col = df[!, c]
        eltype(col) <: Union{Missing,Real} || throw(ArgumentError(
            "covariate `$c` must be numeric (create dummy columns for categorical " *
            "covariates)"))
        X[:, j + 1] .= float.(col)
    end
    return X
end

"""
    did_drdid(data, outcome, treatment, unit, time;
              method=:dr_improved, covariates=Symbol[], weights=nothing,
              cluster=nothing, trim_level=0.995) -> DiDEstimate
    did_drdid(panel::TreatmentPanel; kwargs...) -> DiDEstimate

Two-period difference-in-differences estimators of the average treatment effect on
the treated under conditional parallel trends (Sant'Anna and Zhao, 2020), for panel
data or repeated cross-sections.

The estimand is ``ATT = E[Y_{1}(1) - Y_{1}(0) \\mid D = 1]`` in the post-treatment
period. Identification rests on *conditional* parallel trends,
``E[Y_1(0) - Y_0(0) \\mid X, D = 1] = E[Y_1(0) - Y_0(0) \\mid X, D = 0]``, on no
anticipation, and on overlap, ``\\Pr(D = 1 \\mid X) < 1``, for pre-treatment
covariates ``X`` (Heckman, Ichimura and Todd, 1997; Abadie, 2005). Conditioning on
covariates allows the untreated trend to depend on observed characteristics whose
distribution differs between groups; the covariates must not be affected by
treatment, which is why their pre-treatment values are used. Three families of
estimators are available. Outcome regression (`:reg`) models
``E[\\Delta Y \\mid X, D = 0]`` linearly and averages the predicted counterfactual
trend over the treated. Inverse probability weighting (`:ipw`, normalized or Hájek
weights, and `:ipw_unnormalized`, Horvitz–Thompson weights) reweights the comparison
group by the odds of the propensity score (Abadie, 2005). The doubly robust
estimators combine both and are consistent if either the propensity-score model or
the outcome-regression model is correctly specified, in the spirit of Robins,
Rotnitzky and Zhao (1994); they are locally efficient when both are.

The default `:dr_improved` is the *improved* doubly robust estimator of Sant'Anna and
Zhao (2020): the propensity score is fitted by inverse probability tilting (Graham,
Pinto and Egel, 2012) and the outcome regression by weighted least squares, which
makes the estimator doubly robust for inference as well as for the point estimate
(the first-step estimation error does not enter the influence function when either
model is correct). `:dr` is the *traditional* locally efficient doubly robust
estimator with a logit propensity score and an OLS outcome regression. For repeated
cross-sections the kernels of Sant'Anna and Zhao (2020, Section 3) model the four
group × period cells. Without covariates all methods reduce to the simple 2×2
difference in mean changes.

Standard errors come from the estimated influence function, including the
estimation of the nuisance models, with normal critical values (`dof_residual =
Inf`); with `cluster` the influence function is summed within clusters. Comparison
units with an estimated propensity score of at least `trim_level` receive zero weight,
as in R's `DRDID`, against which the implementation is validated; limited overlap
shows up as extreme weights and unstable estimates, so inspect the propensity score
distribution. For more than two periods or staggered adoption use
[`did_callaway_santanna`](@ref), which applies these estimators to every
cohort-period comparison.

# Arguments
- `data`: a `DataFrame` with exactly two time periods (long format).
- `outcome::Symbol`: outcome column.
- `treatment::Symbol`: treatment-**group** indicator (1 for units treated in the
  second period). For panel data a time-varying ``D_{it}`` (0 in the first period, 1
  in the second for treated units) is also accepted.
- `unit`: unit identifier for panel data, or `nothing` for repeated
  cross-sections. Panel units not observed in both periods are dropped with a
  warning.
- `time::Symbol`: time column with two distinct values; the later one is the
  post-treatment period.

# Keywords
- `method::Symbol = :dr_improved`: `:dr_improved`, `:dr`, `:ipw`,
  `:ipw_unnormalized` or `:reg`, as described above.
- `covariates::Vector{Symbol} = Symbol[]`: numeric pre-treatment covariates; for
  panel data the first-period values are used. An intercept is always included.
- `weights::Union{Nothing,Symbol} = nothing`: sampling-weight column (normalized to
  mean one; for panels the first-period weight is used).
- `cluster::Union{Nothing,Symbol} = nothing`: column for cluster-robust inference;
  by default units (panel) or observations (cross-sections) are independent.
- `trim_level::Real = 0.995`: comparison units with propensity score at or above
  this value get zero weight.

# Returns
- `DiDEstimate`: the ATT with `dof_residual = Inf`; `details` holds the influence
  function (`influence`), the method, and whether the data are a panel.

# Examples
```julia
using DrSnow, CSV, DataFrames
file = joinpath(pkgdir(DrSnow), "test", "validation", "did", "lalonde_panel.csv")
nsw = CSV.read(file, DataFrame)
r = did_drdid(nsw, :re, :experimental, :id, :year;
              covariates=[:age, :educ, :black, :married, :nodegree, :hisp, :re74])
coef(r), stderror(r), confint(r; level=0.90)
did_drdid(nsw, :re, :experimental, :id, :year; covariates=[:age, :educ],
          method=:reg)
```

# References
- Sant'Anna, P. H. C., & Zhao, J. (2020). Doubly robust difference-in-differences
  estimators. *Journal of Econometrics*, 219(1), 101–122.
- Abadie, A. (2005). Semiparametric difference-in-differences estimators. *Review of
  Economic Studies*, 72(1), 1–19.
- Heckman, J. J., Ichimura, H., & Todd, P. E. (1997). Matching as an econometric
  evaluation estimator: Evidence from evaluating a job training programme. *Review
  of Economic Studies*, 64(4), 605–654.
- Graham, B. S., Pinto, C. C. de X., & Egel, D. (2012). Inverse probability tilting
  for moment condition models with missing data. *Review of Economic Studies*,
  79(3), 1053–1079.
- Robins, J. M., Rotnitzky, A., & Zhao, L. P. (1994). Estimation of regression
  coefficients when some regressors are not always observed. *Journal of the
  American Statistical Association*, 89(427), 846–866.
- Sant'Anna, P. H. C., & Zhao, J. (2026). DRDID: Doubly robust
  difference-in-differences estimators. R package version 1.3.0.
"""
function did_drdid(data, outcome::Symbol, treatment::Symbol, unit, time::Symbol;
                   method::Symbol=:dr_improved, covariates::Vector{Symbol}=Symbol[],
                   weights::Union{Nothing,Symbol}=nothing,
                   cluster::Union{Nothing,Symbol}=nothing, trim_level::Real=0.995)
    _did_check_method(method)
    df = _did_prepare(data, [outcome, treatment, unit, time, covariates..., weights,
                             cluster]; context="did_drdid")
    periods = sort!(unique(df[!, time]))
    length(periods) == 2 || throw(ArgumentError(
        "did_drdid needs exactly two time periods; found $(length(periods)). For more " *
        "periods use did_callaway_santanna."))
    post = df[!, time] .== periods[2]
    Draw = _did_binary(df[!, treatment], treatment)
    wraw = weights === nothing ? ones(nrow(df)) : float.(df[!, weights])
    any(<(0), wraw) && throw(ArgumentError("weights must be non-negative"))
    label = "Sant'Anna–Zhao DiD (" * string(method) * ")"
    if unit === nothing
        X = _did_design_matrix(df, covariates)
        att, ψ = _did_kernel_rc(method, float.(df[!, outcome]), post, Draw, X, wraw;
                                trim_level=trim_level)
        cl = cluster === nothing ? nothing : df[!, cluster]
        V, G = _if_vcov(reshape(ψ, :, 1), cl)
        return DiDEstimate([att], V, ["ATT"], nrow(df), Inf, cluster === nothing ? 0 : G,
                           count(Draw), count(!, Draw), 2,
                           label * ", repeated cross-sections",
                           "ATT", (influence=ψ, method=method, panel=false,
                                   n_treated_obs=count(Draw)))
    end
    # Panel: pair observations by unit id (never by row order).
    ids = df[!, unit]
    idx_pre = Dict{Any,Int}()
    idx_post = Dict{Any,Int}()
    for i in 1:nrow(df)
        d = post[i] ? idx_post : idx_pre
        haskey(d, ids[i]) && throw(ArgumentError(
            "unit $(ids[i]) is observed more than once in a period"))
        d[ids[i]] = i
    end
    units = [u for u in keys(idx_pre) if haskey(idx_post, u)]
    try
        sort!(units)
    catch
    end
    dropped = length(idx_pre) + length(idx_post) - 2 * length(units)
    dropped > 0 && @warn "did_drdid: dropped $dropped observations of units not " *
                         "observed in both periods (balanced panel required)"
    i0 = [idx_pre[u] for u in units]
    i1 = [idx_post[u] for u in units]
    y = float.(df[!, outcome])
    dy = y[i1] .- y[i0]
    # Group indicator: accepts a time-invariant group dummy or D_it (0 pre, 1 post).
    D = Draw[i1]
    if !(Draw[i0] == Draw[i1] || !any(Draw[i0]))
        throw(ArgumentError("treatment must be a time-invariant group indicator, or a " *
                            "D_it that is 0 for every unit in the first period"))
    end
    X = _did_design_matrix(df[i0, :], covariates)
    w = wraw[i0]
    att, ψ = _did_kernel_panel(method, dy, D, X, w; trim_level=trim_level)
    cl = cluster === nothing ? nothing : df[i0, cluster]
    V, G = _if_vcov(reshape(ψ, :, 1), cl)
    return DiDEstimate([att], V, ["ATT"], 2 * length(units), Inf,
                       cluster === nothing ? 0 : G, count(D), count(!, D), 2,
                       label * ", panel", "ATT",
                       (influence=ψ, method=method, panel=true, units=units))
end

did_drdid(panel::TreatmentPanel; kwargs...) =
    did_drdid(panel.data, panel.outcome, panel.treatment, panel.unit_id, panel.time;
              covariates=panel.covariates, kwargs...)
