# Local polynomial RD point estimation and robust bias-corrected inference
# (rdrobust equivalent): sharp, fuzzy, sharp kink and fuzzy kink designs.

"""
    RDEstimate <: CausalEstimate

Result of [`rd_estimate`](@ref): a local polynomial regression discontinuity estimate
with conventional, bias-corrected and robust bias-corrected inference.

The object stores the three inferential variants reported by `rdrobust` (Calonico,
Cattaneo & Titiunik 2014). The *conventional* estimate ``\\hat\\tau_p`` is the difference
between the two local polynomial fits of order `p` at the cutoff (in levels or in the
derivative of order `deriv`), with standard error `se_conventional`. The
*bias-corrected* estimate ``\\hat\\tau_p^{\\text{bc}} = \\hat\\tau_p - \\hat B``
subtracts an estimate ``\\hat B`` of the leading smoothing bias, obtained from local
polynomials of order `q` with pilot bandwidth `b`. The *robust* standard error
`se_robust` also accounts for the sampling variability of ``\\hat B``, which the
conventional standard error ignores; this is what makes the bias-corrected interval
valid at MSE-optimal bandwidths.

**Which estimate is the headline.** DrSnow follows the reporting convention of
`rdrobust` (Calonico, Cattaneo & Titiunik 2014; Cattaneo, Idrobo & Titiunik 2020):
`coef(r)` is the *conventional* point estimate ``\\hat\\tau_p`` (the field
`tau_conventional`), while `stderror`, `tstats`, `pvalues` and `confint` are the robust
bias-corrected ones. The robust interval
``\\hat\\tau_p^{\\text{bc}} \\pm z_{1-\\alpha/2}\\,\\text{se}_{\\text{rb}}`` is centred on
the bias-corrected estimate (the field `tau_bias_corrected`), so it is generally not
symmetric around `coef(r)`, and the t statistic is
``\\hat\\tau_p^{\\text{bc}} / \\text{se}_{\\text{rb}}`` rather than `coef / stderror`.
`coeftable`, `tidy` and `regtable` therefore show the conventional estimate next to the
robust standard error, p-value and interval, which is the layout recommended for
reporting. The two point estimates differ by the estimated bias ``\\hat B`` (on the U.S.
Senate data of Cattaneo, Frandsen & Titiunik 2015, 7.414 conventional versus 7.507
bias-corrected). [`rd_inference_table`](@ref) lists all three rows. The conventional
interval ``\\hat\\tau_p \\pm z_{1-\\alpha/2}\\,\\text{se}_{\\text{conv}}`` under-covers at
MSE-optimal bandwidths, where bias and standard deviation are of the same order, and
should not be used for inference with them.

# Fields
- `design::Symbol`: `:sharp`, `:fuzzy`, `:sharp_kink`, `:fuzzy_kink` (`deriv = 1`), or
  `:sharp_deriv`, `:fuzzy_deriv` (`deriv ≥ 2`).
- `tau_conventional::Float64`: conventional estimate ``\\hat\\tau_p`` (returned by
  `coef`, as in `rdrobust`).
- `tau_bias_corrected::Float64`: bias-corrected estimate (the centre of `confint`).
- `se_conventional::Float64`, `se_robust::Float64`: conventional and robust standard
  errors (`stderror` returns `se_robust`).
- `first_stage`: `nothing` (sharp designs) or a `NamedTuple` with the jump in treatment
  take-up (`tau_conventional`, `tau_bias_corrected`, `se_conventional`, `se_robust`).
- `reduced_form`: `nothing` (sharp designs) or a `NamedTuple` with the outcome jump
  (`tau_conventional`, `tau_bias_corrected`).
- `vcov_robust_yt::Matrix{Float64}`: in fuzzy designs, the 2×2 robust covariance of the
  bias-corrected (reduced form, first stage) jumps, used by
  [`rd_weak_iv_confidence_set`](@ref); empty for sharp designs.
- `h_left`, `h_right`, `b_left`, `b_right`: main and pilot (bias) bandwidths.
- `n_left`, `n_right`: observations on each side of the cutoff; `n_h_left`,
  `n_h_right`: observations within `h`; `n_b_left`, `n_b_right`: within `b`;
  `m_left`, `m_right`: distinct values of the running variable on each side.
- `cutoff`, `p`, `q`, `deriv`, `kernel`, `vce`, `bwselect`, `nnmatch`: settings used
  (`bwselect = :manual` when bandwidths were supplied).
- `beta_left`, `beta_right`: local polynomial coefficients of order `p` of the
  (covariate-adjusted) outcome on each side, in powers of `x - cutoff`.
- `bias_left`, `bias_right`: estimated leading bias on each side.
- `covariates::Vector{Symbol}`, `gamma::Matrix{Float64}`: covariates kept for
  adjustment and their coefficients (rows: covariates; columns: outcome and, in fuzzy
  designs, treatment).
- `n_clusters::Tuple{Int,Int}`: clusters on each side (zeros when unclustered).
- `level::Float64`: confidence level used when printing.

`estimand(r)` and `method_name(r)` describe the target parameter and the estimator;
`nobs(r)` is `n_left + n_right`.
"""
struct RDEstimate <: CausalEstimate
    design::Symbol
    tau_conventional::Float64
    tau_bias_corrected::Float64
    se_conventional::Float64
    se_robust::Float64
    first_stage::Union{Nothing,NamedTuple}
    reduced_form::Union{Nothing,NamedTuple}
    vcov_robust_yt::Matrix{Float64}
    h_left::Float64
    h_right::Float64
    b_left::Float64
    b_right::Float64
    n_left::Int
    n_right::Int
    n_h_left::Int
    n_h_right::Int
    n_b_left::Int
    n_b_right::Int
    m_left::Int
    m_right::Int
    cutoff::Float64
    p::Int
    q::Int
    deriv::Int
    kernel::Symbol
    vce::Symbol
    bwselect::Symbol
    nnmatch::Int
    beta_left::Vector{Float64}
    beta_right::Vector{Float64}
    bias_left::Float64
    bias_right::Float64
    covariates::Vector{Symbol}
    gamma::Matrix{Float64}
    n_clusters::Tuple{Int,Int}
    level::Float64
end

# Reporting convention of rdrobust (Calonico, Cattaneo & Titiunik 2014; Cattaneo,
# Idrobo & Titiunik 2020): the headline point estimate is the conventional local
# polynomial estimate, while the standard error, t statistic, p-value and confidence
# interval are the robust bias-corrected ones. The robust interval is centred on the
# bias-corrected estimate, so `confint` and `tstats` are defined explicitly rather than
# derived from `coef`.
StatsAPI.coef(r::RDEstimate) = [r.tau_conventional]
StatsAPI.vcov(r::RDEstimate) = fill(r.se_robust^2, 1, 1)
StatsAPI.coefnames(r::RDEstimate) = ["RD effect"]
StatsAPI.nobs(r::RDEstimate) = r.n_left + r.n_right
tstats(r::RDEstimate) = [r.tau_bias_corrected / r.se_robust]
function StatsAPI.confint(r::RDEstimate; level::Real=0.95)
    c = critical_value(level, Inf)
    return [r.tau_bias_corrected - c * r.se_robust r.tau_bias_corrected + c * r.se_robust]
end

function estimand(r::RDEstimate)
    d = r.deriv
    if r.design === :sharp
        return "ATE at the cutoff (sharp RD)"
    elseif r.design === :fuzzy
        return "LATE for compliers at the cutoff (fuzzy RD)"
    elseif r.design === :sharp_kink
        return "change in the slope of E[Y|X] at the cutoff (sharp kink RD)"
    elseif r.design === :fuzzy_kink
        return "ratio of slope changes of E[Y|X] and E[D|X] at the cutoff (fuzzy kink RD)"
    elseif r.design === :sharp_deriv
        return "jump in the derivative of order $d of E[Y|X] at the cutoff"
    else
        return "ratio of jumps in derivatives of order $d of E[Y|X] and E[D|X] at " *
               "the cutoff"
    end
end

method_name(r::RDEstimate) = "Local polynomial RD (p = $(r.p), q = $(r.q), " *
                             "$(r.kernel) kernel, $(uppercase(string(r.vce))) variance)"

"""
    rd_inference_table(r::RDEstimate; level=r.level) -> DataFrame
    rd_inference_table(r::RDFlexEstimate; level=r.level) -> DataFrame

Conventional, bias-corrected and robust bias-corrected rows of a local polynomial RD
estimate, in the layout of the `rdrobust` summary.

The three rows combine two point estimates with two standard errors (Calonico, Cattaneo
& Titiunik 2014). *Conventional*: ``\\hat\\tau_p`` with the conventional standard error;
its interval ignores the smoothing bias and under-covers at MSE-optimal bandwidths.
*Bias-corrected*: ``\\hat\\tau_p^{\\text{bc}}`` with the conventional standard error; this
row ignores the variability of the bias estimate and is shown only because `rdrobust`
shows it. *Robust*: ``\\hat\\tau_p^{\\text{bc}}`` with the robust standard error; this is
the row with valid inference at MSE- and CER-optimal bandwidths, and the one returned
by `stderror`, `pvalues` and `confint`; `coef` returns the conventional estimate, as
`rdrobust` does.

`rdrobust` prints the conventional point estimate as its coefficient together with the
robust interval of the third row; DrSnow's `coef`, `confint` and `coeftable` follow the
same convention (see [`RDEstimate`](@ref)).

# Arguments
- `r`: a result of [`rd_estimate`](@ref) or [`rd_flex`](@ref).

# Keywords
- `level::Real=r.level`: confidence level of the intervals (default: the level stored in
  `r`, 0.95 unless changed in the call that produced it).

# Returns
- `DataFrame` with one row per method (`"Conventional"`, `"Bias-corrected"`,
  `"Robust"`) and columns `method`, `estimate`, `se`, `z`, `pvalue` (two-sided,
  normal), `ci_lower`, `ci_upper`.

# Examples
```julia
using DrSnow, CSV, DataFrames
senate = CSV.read(pkgdir(DrSnow, "test", "validation", "rd", "senate.csv"), DataFrame)
r = rd_estimate(senate, :vote, :margin)
tab = rd_inference_table(r; level=0.90)
(estimate = tab.estimate[1], ci = (tab.ci_lower[3], tab.ci_upper[3]))  # rdrobust layout
```

# References
- Calonico, S., Cattaneo, M. D., & Titiunik, R. (2014). Robust nonparametric confidence
  intervals for regression-discontinuity designs. *Econometrica*, 82(6), 2295–2326.
- Calonico, S., Cattaneo, M. D., Farrell, M. H., & Titiunik, R. (2017). rdrobust:
  Software for regression-discontinuity designs. *The Stata Journal*, 17(2), 372–404.
"""
function rd_inference_table(r::RDEstimate; level::Real=r.level)
    z = critical_value(level)
    est = [r.tau_conventional, r.tau_bias_corrected, r.tau_bias_corrected]
    se = [r.se_conventional, r.se_conventional, r.se_robust]
    t = est ./ se
    return DataFrame(method=["Conventional", "Bias-corrected", "Robust"], estimate=est,
                     se=se, z=t, pvalue=two_sided_pvalue.(t), ci_lower=est .- z .* se,
                     ci_upper=est .+ z .* se)
end

function show_details(io::IO, r::RDEstimate)
    println(io)
    @printf(io, "Cutoff = %g; bandwidths h = (%.4g, %.4g), b = (%.4g, %.4g) [%s]\n",
            r.cutoff, r.h_left, r.h_right, r.b_left, r.b_right, r.bwselect)
    @printf(io, "Observations left/right: %d / %d; within h: %d / %d\n", r.n_left,
            r.n_right, r.n_h_left, r.n_h_right)
    if r.n_clusters != (0, 0)
        @printf(io, "Clusters left/right: %d / %d\n", r.n_clusters...)
    end
    isempty(r.covariates) || println(io, "Covariates: ", join(r.covariates, ", "))
    tab = rd_inference_table(r)
    lv = round(Int, 100 * r.level)
    println(io, "Inference ($lv% CI):")
    for row in eachrow(tab)
        @printf(io, "  %-15s %10.4f  se %8.4f  p %8.4g  [%.4f, %.4f]\n", row.method,
                row.estimate, row.se, row.pvalue, row.ci_lower, row.ci_upper)
    end
    if r.first_stage !== nothing
        fs = r.first_stage
        tfs = fs.tau_bias_corrected / fs.se_robust
        @printf(io, "First stage (jump in treatment): %.4f (se %.4f); robust z = %.3f\n",
                fs.tau_conventional, fs.se_conventional, tfs)
    end
end

# ---------------------------------------------------------------------------------------
# Core computation on sorted, split data
# ---------------------------------------------------------------------------------------

function _rd_side_fit(sd, c, h, b, p, q, kernel, T_on::Bool)
    w_h = _rd_kweight(sd.X, c, h, kernel)
    w_b = _rd_kweight(sd.X, c, b, kernel)
    if sd.W !== nothing
        w_h .*= sd.W
        w_b .*= sd.W
    end
    ind_h = w_h .> 0
    ind_b = w_b .> 0
    ind = h > b ? ind_h : ind_b
    eX = sd.X[ind]
    W_h = w_h[ind]
    W_b = w_b[ind]
    # The local fits are identified only with enough distinct running-variable values.
    nuh = length(unique(eX[W_h .> 0]))
    nub = length(unique(eX[W_b .> 0]))
    if nuh < p + 1 || nub < q + 1
        side = c > sd.X[end] ? "left" : "right"
        throw(ArgumentError(
            "rd_estimate: too few distinct values of the running variable within the " *
            "bandwidth on the $side side of the cutoff ($nuh within h, $nub within b; " *
            "need at least p + 1 = $(p + 1) and q + 1 = $(q + 1)); use a larger " *
            "bandwidth or lower polynomial orders"))
    end
    u = (eX .- c) ./ h
    R_q = _rd_vander(eX .- c, q)
    R_p = R_q[:, 1:(p + 1)]
    RpW = R_p .* W_h
    L = RpW' * (u .^ (p + 1))
    invG_q = _rd_xxinv(sqrt.(W_b) .* R_q)
    invG_p = _rd_xxinv(sqrt.(W_h) .* R_p)
    e_p1 = zeros(q + 1)
    e_p1[p + 2] = 1
    Qt = RpW' .- h^(p + 1) .* (L * e_p1') * ((R_q * invG_q) .* W_b)'
    Q = Matrix(Qt')
    D = _rd_design_D(sd.Y[ind], T_on ? sd.T[ind] : nothing,
                     sd.Z === nothing ? nothing : sd.Z[ind, :])
    beta_p = invG_p * (RpW' * D)
    beta_q = invG_q * ((R_q .* W_b)' * D)
    beta_bc = invG_p * (Q' * D)
    return (; ind, eX, W_h, W_b, R_q, R_p, RpW, invG_q, invG_p, Q, D, beta_p, beta_q,
            beta_bc, n_h=count(ind_h), n_b=count(ind_b),
            eC=sd.C === nothing ? nothing : sd.C[ind],
            dups=isempty(sd.dups) ? Int[] : sd.dups[ind],
            dupsid=isempty(sd.dupsid) ? Int[] : sd.dupsid[ind])
end

function _rd_core(S; h_l, h_r, b_l, b_r, scalepar::Float64, level::Float64,
                  bwselect::Symbol, covariates::Vector{Symbol})
    c, p, q, deriv = S.c, S.p, S.q, S.deriv
    fuzzy = S.T !== nothing
    dT = fuzzy ? 1 : 0
    dZ = S.Z === nothing ? 0 : size(S.Z, 2)
    colsZ = (2 + dT):(1 + dT + dZ)
    fl = _rd_side_fit(S.L, c, h_l, b_l, p, q, S.kernel, fuzzy)
    fr = _rd_side_fit(S.R, c, h_r, b_r, p, q, S.kernel, fuzzy)
    fd = factorial(deriv)
    i = deriv + 1
    beta_p = fr.beta_p .- fl.beta_p
    beta_bc = fr.beta_bc .- fl.beta_bc

    gamma = zeros(0, 1 + dT)
    reduced_form = nothing
    first_stage_pt = nothing
    sY0 = sT0 = nothing   # selectors of the Y and T components (for fuzzy covariance)
    if dZ == 0
        tau_Y_cl = scalepar * fd * beta_p[i, 1]
        tau_Y_bc = scalepar * fd * beta_bc[i, 1]
        tau_cl, tau_bc = tau_Y_cl, tau_Y_bc
        s_Y = [1.0]
        bias_l = scalepar * fd * (fl.beta_p[i, 1] - fl.beta_bc[i, 1])
        bias_r = scalepar * fd * (fr.beta_p[i, 1] - fr.beta_bc[i, 1])
        beta_Y_l = scalepar * fd .* fl.beta_p[:, 1]
        beta_Y_r = scalepar * fd .* fr.beta_p[:, 1]
        if fuzzy
            tau_T_cl = fd * beta_p[i, 2]
            tau_T_bc = fd * beta_bc[i, 2]
            tau_cl = tau_Y_cl / tau_T_cl
            s_Y = [1 / tau_T_cl, -(tau_Y_cl / tau_T_cl^2)]
            B_F = [tau_Y_cl - tau_Y_bc, tau_T_cl - tau_T_bc]
            tau_bc = tau_cl - dot(s_Y, B_F)
            sV_T = [0.0, 1.0]
            B_F_l = [scalepar * fd * (fl.beta_p[i, 1] - fl.beta_bc[i, 1]),
                     fd * (fl.beta_p[i, 2] - fl.beta_bc[i, 2])]
            B_F_r = [scalepar * fd * (fr.beta_p[i, 1] - fr.beta_bc[i, 1]),
                     fd * (fr.beta_p[i, 2] - fr.beta_bc[i, 2])]
            bias_l = dot(s_Y, B_F_l)
            bias_r = dot(s_Y, B_F_r)
            sY0, sT0 = [1.0, 0.0], sV_T
            first_stage_pt = (tau_T_cl, tau_T_bc)
            reduced_form = (tau_conventional=tau_Y_cl, tau_bias_corrected=tau_Y_bc)
        end
    else
        # Covariate adjustment (Calonico, Cattaneo, Farrell & Titiunik 2019).
        function zparts(f)
            U = f.RpW' * f.D
            ZWD = (f.D[:, colsZ] .* f.W_h)' * f.D
            UiGU = U[:, colsZ]' * (f.invG_p * U)
            return ZWD[:, colsZ] .- UiGU[:, colsZ],
                   ZWD[:, 1:(1 + dT)] .- UiGU[:, 1:(1 + dT)]
        end
        ZWZ_l, ZWY_l = zparts(fl)
        ZWZ_r, ZWY_r = zparts(fr)
        gamma = _rd_ginv(ZWZ_l .+ ZWZ_r) * (ZWY_l .+ ZWY_r)
        s_Y = vcat(1.0, -gamma[:, 1])
        if !fuzzy
            tau_cl = scalepar * fd * dot(s_Y, beta_p[i, :])
            tau_bc = scalepar * fd * dot(s_Y, beta_bc[i, :])
            bias_l = scalepar * fd *
                     (dot(s_Y, fl.beta_p[i, :]) - dot(s_Y, fl.beta_bc[i, :]))
            bias_r = scalepar * fd *
                     (dot(s_Y, fr.beta_p[i, :]) - dot(s_Y, fr.beta_bc[i, :]))
            beta_Y_l = scalepar * fd .* (fl.beta_p * s_Y)
            beta_Y_r = scalepar * fd .* (fr.beta_p * s_Y)
        else
            s_T = vcat(1.0, -gamma[:, 2])
            sV_T = vcat(0.0, 1.0, -gamma[:, 2])
            yz(Bm) = vcat(Bm[i, 1], Bm[i, colsZ])
            tz(Bm) = vcat(Bm[i, 2], Bm[i, colsZ])
            tau_Y_cl = scalepar * fd * dot(s_Y, yz(beta_p))
            tau_Y_bc = scalepar * fd * dot(s_Y, yz(beta_bc))
            tau_T_cl = fd * dot(s_T, tz(beta_p))
            tau_T_bc = fd * dot(s_T, tz(beta_bc))
            B_F_l = [scalepar * fd * (dot(s_Y, yz(fl.beta_p)) - dot(s_Y, yz(fl.beta_bc))),
                     fd * (dot(s_T, tz(fl.beta_p)) - dot(s_T, tz(fl.beta_bc)))]
            B_F_r = [scalepar * fd * (dot(s_Y, yz(fr.beta_p)) - dot(s_Y, yz(fr.beta_bc))),
                     fd * (dot(s_T, tz(fr.beta_p)) - dot(s_T, tz(fr.beta_bc)))]
            beta_Y_l = scalepar * fd .* (hcat(fl.beta_p[:, 1], fl.beta_p[:, colsZ]) * s_Y)
            beta_Y_r = scalepar * fd .* (hcat(fr.beta_p[:, 1], fr.beta_p[:, colsZ]) * s_Y)
            tau_cl = tau_Y_cl / tau_T_cl
            B_F = [tau_Y_cl - tau_Y_bc, tau_T_cl - tau_T_bc]
            sd2 = [1 / tau_T_cl, -(tau_Y_cl / tau_T_cl^2)]
            tau_bc = tau_cl - dot(sd2, B_F)
            bias_l = dot(sd2, B_F_l)
            bias_r = dot(sd2, B_F_r)
            s_Y = vcat(1 / tau_T_cl, -(tau_Y_cl / tau_T_cl^2),
                       -(1 / tau_T_cl) .* gamma[:, 1] .+
                       (tau_Y_cl / tau_T_cl^2) .* gamma[:, 2])
            sY0 = vcat(1.0, 0.0, -gamma[:, 1])
            sT0 = sV_T
            first_stage_pt = (tau_T_cl, tau_T_bc)
            reduced_form = (tau_conventional=tau_Y_cl, tau_bias_corrected=tau_Y_bc)
        end
    end

    # Residuals
    vce = S.vce
    has_cl = S.has_cluster
    function resids(f, which)
        if which === :p
            return _rd_residuals(f.eX, f.D, f.R_p, f.beta_p, f.invG_p, f.W_h, vce,
                                 S.nnmatch, f.dups, f.dupsid, p + 1, has_cl)
        else
            return _rd_residuals(f.eX, f.D, f.R_q, f.beta_q, f.invG_q, f.W_b, vce,
                                 S.nnmatch, f.dups, f.dupsid, q + 1, has_cl)
        end
    end
    res_h_l, res_h_r = resids(fl, :p), resids(fr, :p)
    res_b_l, res_b_r = vce === :nn ? (res_h_l, res_h_r) : (resids(fl, :q), resids(fr, :q))
    grp_l = has_cl ? _rd_cluster_groups(fl.eC) : nothing
    grp_r = has_cl ? _rd_cluster_groups(fr.eC) : nothing
    hb_match = h_l == b_l && h_r == b_r

    # Variance of the combination a'(outcome, treatment, covariates) (and cross terms).
    crv = has_cl && vce in (:cr2, :cr3)
    function V_cl(f, res, grp, a, b)
        return f.invG_p * _rd_vmeat(vce, f.R_p, f.W_h, f.invG_p, res * a, res * b, grp,
                                    p + 1) * f.invG_p
    end
    function V_rb(f, res, grp, a, b)
        if hb_match && has_cl
            return f.invG_q * _rd_vmeat(vce, f.R_q, f.W_h, f.invG_q, res * a, res * b,
                                        grp, q + 1) * f.invG_q
        elseif crv
            return f.invG_p * _rd_meat_qq(f.Q, f.R_q, f.W_b, f.invG_q, res * a, res * b,
                                          grp, vce === :cr2) * f.invG_p
        else
            return f.invG_p * _rd_meat(f.Q, res * a, res * b, grp, q + 1) * f.invG_p
        end
    end
    sc = scalepar^2 * fd^2
    V_tau_cl = sc * (V_cl(fl, res_h_l, grp_l, s_Y, s_Y) .+
                     V_cl(fr, res_h_r, grp_r, s_Y, s_Y))[i, i]
    V_tau_rb = sc * (V_rb(fl, res_b_l, grp_l, s_Y, s_Y) .+
                     V_rb(fr, res_b_r, grp_r, s_Y, s_Y))[i, i]
    se_cl = sqrt(V_tau_cl)
    se_rb = sqrt(V_tau_rb)

    first_stage = nothing
    Vyt = zeros(0, 0)
    if fuzzy
        V_T_cl = fd^2 * (V_cl(fl, res_h_l, grp_l, sT0, sT0) .+
                         V_cl(fr, res_h_r, grp_r, sT0, sT0))[i, i]
        V_T_rb = fd^2 * (V_rb(fl, res_b_l, grp_l, sT0, sT0) .+
                         V_rb(fr, res_b_r, grp_r, sT0, sT0))[i, i]
        first_stage = (tau_conventional=first_stage_pt[1],
                       tau_bias_corrected=first_stage_pt[2],
                       se_conventional=sqrt(V_T_cl), se_robust=sqrt(V_T_rb))
        vyy = scalepar^2 * fd^2 * (V_rb(fl, res_b_l, grp_l, sY0, sY0) .+
                                   V_rb(fr, res_b_r, grp_r, sY0, sY0))[i, i]
        vyt_m = V_rb(fl, res_b_l, grp_l, sY0, sT0) .+ V_rb(fr, res_b_r, grp_r, sY0, sT0)
        vyt = scalepar * fd^2 * vyt_m[i, i]
        Vyt = [vyy vyt; vyt V_T_rb]
    end

    design = if deriv == 0
        fuzzy ? :fuzzy : :sharp
    elseif deriv == 1
        fuzzy ? :fuzzy_kink : :sharp_kink
    else
        fuzzy ? :fuzzy_deriv : :sharp_deriv
    end
    return RDEstimate(design, tau_cl, tau_bc, se_cl, se_rb, first_stage, reduced_form, Vyt,
                      h_l, h_r, b_l, b_r, S.N_l, S.N_r, fl.n_h, fr.n_h, fl.n_b, fr.n_b,
                      S.M_l, S.M_r, c, p, q, deriv, S.kernel, vce, bwselect, S.nnmatch,
                      vec(beta_Y_l), vec(beta_Y_r), bias_l, bias_r, covariates,
                      Matrix{Float64}(gamma), (S.g_l, S.g_r), level)
end

_rd_pair(v::Real) = (Float64(v), Float64(v))
function _rd_pair(v)
    length(v) == 2 || throw(ArgumentError("bandwidths must be a scalar or a 2-element " *
                                          "(left, right) collection"))
    return (Float64(v[1]), Float64(v[2]))
end

"""
    rd_estimate(data, outcome, running; cutoff=0.0, treatment=nothing, deriv=0,
                p=nothing, q=nothing, h=nothing, b=nothing, rho=nothing,
                bwselect=:mserd, kernel=:triangular, vce=:nn, nnmatch=3,
                cluster=nothing, weights=nothing, covariates=Symbol[], level=0.95,
                scalepar=1, scaleregul=1, masspoints=:adjust, bwcheck=nothing,
                bwrestrict=true, sharpbw=false, covs_drop=true,
                stdvars=false) -> RDEstimate

Local polynomial estimation of sharp, fuzzy and kink regression discontinuity (RD)
designs with robust bias-corrected inference, equivalent to `rdrobust` (R/Stata).

In an RD design, introduced by Thistlethwaite and Campbell (1960), treatment is
assigned, wholly or in part, by whether a score (running variable) ``X`` crosses a
known cutoff ``c``. In the *sharp* design ``D = 1\\{X \\ge c\\}`` and the estimand is the
average treatment effect at the cutoff,
```math
\\tau = E[Y(1) - Y(0) \\mid X = c]
     = \\lim_{x \\downarrow c} E[Y \\mid X = x] - \\lim_{x \\uparrow c} E[Y \\mid X = x],
```
identified when ``E[Y(0) \\mid X = x]`` and ``E[Y(1) \\mid X = x]`` are continuous at
``c`` (Hahn, Todd & van der Klaauw 2001). In the *fuzzy* design (`treatment` given) the
probability of take-up jumps at ``c`` without going from 0 to 1; the estimand is the
ratio of the outcome jump to the take-up jump, which under continuity, monotonicity (no
defiers at the cutoff) and a non-zero first stage is the average effect for compliers
at the cutoff. With `deriv = 1` (*kink* designs) jumps in levels are replaced by jumps in
first derivatives. Card, Lee, Pei and Weber (2015) give the kink estimand its causal
meaning: when the policy rule has a kink at ``c`` and the density of unobserved
heterogeneity is smooth there, the ratio of the slope change in ``E[Y \\mid X]`` to the
slope change in treatment identifies a weighted average of marginal effects of a
continuous treatment. In a sharp kink design the slope change of the known policy rule
``\\kappa`` is not estimated: pass `scalepar = 1/κ` to report the effect per unit of
treatment. Continuity is an assumption about potential outcomes and cannot be tested.
The data can speak only to its implications: a continuous density of the running
variable ([`rd_density_test`](@ref); McCrary 2008) and no jumps in predetermined
covariates ([`rd_covariate_balance`](@ref); Lee 2008).

The estimator fits weighted polynomials of order `p` on each side of the cutoff, with
kernel weights ``K((X_i - c)/h)/h`` that vanish outside the bandwidth ``h``, and takes the
difference of the fitted values (or `deriv`-th derivatives) at ``c``. By default
``p = 1`` (local linear) and, for derivatives, ``p = \\text{deriv} + 1``, so a kink design
uses a local quadratic fit. Global high-order polynomials are not offered, for the
reasons in Gelman and Imbens (2019). At the MSE-optimal bandwidth
([`rd_bandwidth`](@ref)) the smoothing bias is of the same order as the standard
deviation, so conventional intervals under-cover. Robust bias correction (Calonico,
Cattaneo & Titiunik 2014) subtracts the leading bias, estimated with a polynomial of
order ``q = p + 1`` and pilot bandwidth ``b``, and inflates the standard error for the
variability of that estimate; Calonico, Cattaneo and Farrell (2020) derive bandwidths
that minimise the coverage error of the resulting interval. Predetermined covariates enter
linearly with common coefficients on both sides (Calonico, Cattaneo, Farrell & Titiunik
2019). If the covariates are continuous at the cutoff, the estimand is unchanged and
precision improves. Adjustment does not substitute for continuity.

Inference is based on the normal approximation. Following `rdrobust`, `coef` returns the
conventional estimate while `stderror`, `pvalues` and `confint` give robust
bias-corrected inference, with the interval centred on the bias-corrected estimate (see
[`RDEstimate`](@ref)). The default variance estimator is the nearest-neighbour estimator of
`rdrobust`, which is robust to heteroskedasticity. With `cluster` the estimator is
cluster-robust (CR1, CR2 or CR3), but normal critical values are kept as in `rdrobust`,
so few clusters within the bandwidth call for caution. The package's Monte Carlo check
uses model 1 of Calonico, Cattaneo and Titiunik (2014) at ``n = 500``. There the robust
95% interval covered 0.909 and 0.924 of the time in runs of 3,000 and 2,000
replications, against 0.902 for `rdrobust` in 1,500 R replications (see the
Validation page).

In practice, report the estimate with its robust interval, the bandwidths and the
effective numbers of observations on each side, an RD plot ([`rd_plot_data`](@ref)), and
the falsification analyses of Cattaneo, Idrobo and Titiunik (2020): density and
covariate tests, placebo cutoffs ([`rd_placebo_cutoffs`](@ref)), donut estimates
([`rd_donut`](@ref)) and bandwidth sensitivity ([`rd_bandwidth_sensitivity`](@ref)).
Other functions cover other situations:
- few distinct values of the running variable near the cutoff: [`rd_honest`](@ref) or
  [`rd_honest_bme`](@ref);
- bias-aware intervals under an explicit smoothness bound: [`rd_honest`](@ref);
- a weak first stage: [`rd_weak_iv_confidence_set`](@ref);
- inference in a small window where assignment is as good as random:
  [`rd_randomization_test`](@ref);
- adjustment for many or non-linear covariates: [`rd_flex`](@ref).

# Arguments
- `data::AbstractDataFrame`: one row per unit. Rows with a missing value in any column
  used by the call are dropped.
- `outcome::Symbol`: outcome column.
- `running::Symbol`: running variable (score). Units with `running ≥ cutoff` are on the
  treated side.

# Keywords
- `cutoff::Real=0.0`: the RD threshold ``c``; it must lie inside the range of `running`.
- `treatment::Union{Nothing,Symbol}=nothing`: treatment take-up. Supplying it makes the
  design fuzzy, and the estimate becomes the ratio of the outcome and take-up jumps.
- `deriv::Integer=0`: order of the derivative whose jump is estimated. Set it to `1` for
  sharp or fuzzy kink designs.
- `p::Union{Nothing,Integer}=nothing`: order of the local polynomial. The default is
  `1` when `deriv = 0` and `deriv + 1` otherwise (so `p = 2` for kink designs).
- `q::Union{Nothing,Integer}=nothing`: order of the bias-correction polynomial, by
  default `p + 1`.
- `h`, `b`: main and pilot bandwidths, as scalars or `(left, right)` pairs. If `h` is
  `nothing` (default), both are selected with `bwselect`. If only `h` is given, `b = h`
  (or `h / rho`), as in `rdrobust`.
- `rho::Union{Nothing,Real}=nothing`: if set, `b = h / rho`. With `rho = 1` the pilot
  and main bandwidths coincide.
- `bwselect=:mserd`: bandwidth selector (see [`rd_bandwidth`](@ref)). `:cerrd` gives
  smaller, coverage-oriented bandwidths.
- `kernel=:triangular`: `:triangular` (the `rdrobust` default), `:epanechnikov` or
  `:uniform`. The uniform kernel reproduces unweighted regressions within the
  bandwidth.
- `vce=:nn`: variance estimator: `:nn` (nearest neighbours) or `:hc0`–`:hc3`. With
  `cluster`, the options are `:cr1` (default), `:cr2` (Bell–McCaffrey type) or `:cr3`
  (jackknife type), as in `rdrobust` 4.0. `:hc*` options combined with `cluster` are
  switched to the corresponding CR option with a warning.
- `nnmatch::Integer=3`: number of neighbours for `vce = :nn`.
- `cluster::Union{Nothing,Symbol}=nothing`: cluster identifier for cluster-robust
  variances. It also changes the default `vce` to `:cr1`.
- `weights::Union{Nothing,Symbol}=nothing`: non-negative observation weights, which
  multiply the kernel weights.
- `covariates::Vector{Symbol}=Symbol[]`: predetermined covariates for linear adjustment
  (Calonico, Cattaneo, Farrell & Titiunik 2019).
- `level::Real=0.95`: confidence level stored in the result and used when printing;
  `confint(r; level)` accepts any other level.
- `scalepar::Real=1`: factor that multiplies the outcome estimand, for example `1/κ` in
  a sharp kink design with known policy slope change ``\\kappa``.
- `scaleregul::Real=1`: scale of the regularisation term in bandwidth selection. `0`
  disables it.
- `masspoints=:adjust`: handling of repeated values of the running variable: `:adjust`
  (count distinct values and adjust bandwidths), `:check` (only report them) or `:off`.
- `bwcheck::Union{Nothing,Integer}=nothing`: minimum number of distinct values of the
  running variable within the bandwidth on each side. It is set automatically when
  mass points are detected.
- `bwrestrict::Bool=true`: cap the selected bandwidths at the range of the running
  variable.
- `sharpbw::Bool=false`: in fuzzy designs, select bandwidths for the outcome equation
  only, as for a sharp design.
- `covs_drop::Bool=true`: drop covariates that are collinear with earlier ones.
- `stdvars::Bool=false`: when bandwidths are selected from the data, select them on the
  outcome and running variable divided by their standard deviations (as
  `rdrobust(..., stdvars = TRUE)`). Estimation always uses the original data.

# Returns
- [`RDEstimate`](@ref). `coef` gives the conventional estimate and `confint` the robust
  bias-corrected interval, as in `rdrobust`; the field `tau_bias_corrected` holds the
  bias-corrected estimate.
  [`rd_inference_table`](@ref) gives all three inference rows.

# Examples
```julia
using DrSnow, CSV, DataFrames, StableRNGs
senate = CSV.read(pkgdir(DrSnow, "test", "validation", "rd", "senate.csv"), DataFrame)
r = rd_estimate(senate, :vote, :margin)       # sharp RD, MSE-optimal bandwidth
coef(r), confint(r)                           # conventional estimate, robust CI
r.tau_bias_corrected                          # bias-corrected estimate (CI centre)
rd_inference_table(r)
rd_estimate(senate, :vote, :margin; covariates=[:presdemvoteshlag1], cluster=:state)

rng = StableRNG(1)
n = 5_000
x = 2 .* rand(rng, n) .- 1
d = Float64.(rand(rng, n) .< 0.2 .+ 0.6 .* (x .>= 0))   # take-up jumps by 0.6
y = 1 .+ x .+ 2 .* d .+ randn(rng, n)
rd_estimate(DataFrame(; y, x, d), :y, :x; treatment=:d)  # fuzzy RD, effect 2
yk = 1 .+ 0.5 .* x .+ 1.5 .* max.(x, 0) .+ 0.2 .* randn(rng, n)
rd_estimate(DataFrame(; y=yk, x), :y, :x; deriv=1)       # sharp kink, slope change 1.5
```

# References
- Thistlethwaite, D. L., & Campbell, D. T. (1960). Regression-discontinuity analysis:
  An alternative to the ex post facto experiment. *Journal of Educational Psychology*,
  51(6), 309–317.
- Hahn, J., Todd, P., & van der Klaauw, W. (2001). Identification and estimation of
  treatment effects with a regression-discontinuity design. *Econometrica*, 69(1),
  201–209.
- Calonico, S., Cattaneo, M. D., & Titiunik, R. (2014). Robust nonparametric confidence
  intervals for regression-discontinuity designs. *Econometrica*, 82(6), 2295–2326.
- Calonico, S., Cattaneo, M. D., & Farrell, M. H. (2020). Optimal bandwidth choice for
  robust bias-corrected inference in regression discontinuity designs. *The
  Econometrics Journal*, 23(2), 192–210.
- Calonico, S., Cattaneo, M. D., Farrell, M. H., & Titiunik, R. (2019). Regression
  discontinuity designs using covariates. *Review of Economics and Statistics*, 101(3),
  442–451.
- Lee, D. S. (2008). Randomized experiments from non-random selection in U.S. House
  elections. *Journal of Econometrics*, 142(2), 675–697.
- Gelman, A., & Imbens, G. (2019). Why high-order polynomials should not be used in
  regression discontinuity designs. *Journal of Business & Economic Statistics*, 37(3),
  447–456.
- Card, D., Lee, D. S., Pei, Z., & Weber, A. (2015). Inference on causal effects in a
  generalized regression kink design. *Econometrica*, 83(6), 2453–2483.
- Cattaneo, M. D., Idrobo, N., & Titiunik, R. (2020). *A Practical Introduction to
  Regression Discontinuity Designs: Foundations*. Cambridge University Press.
- Calonico, S., Cattaneo, M. D., Farrell, M. H., & Titiunik, R. (2017). rdrobust:
  Software for regression-discontinuity designs. *The Stata Journal*, 17(2), 372–404.
"""
function rd_estimate(data::AbstractDataFrame, outcome::Symbol, running::Symbol;
                     cutoff::Real=0.0, treatment=nothing, deriv::Integer=0,
                     p::Union{Nothing,Integer}=nothing, q::Union{Nothing,Integer}=nothing,
                     h=nothing, b=nothing, rho::Union{Nothing,Real}=nothing,
                     bwselect=:mserd, kernel=:triangular, vce=:nn, nnmatch::Integer=3,
                     cluster=nothing, weights=nothing, covariates=Symbol[],
                     level::Real=0.95, scalepar::Real=1, scaleregul::Real=1,
                     masspoints=:adjust, bwcheck::Union{Nothing,Integer}=nothing,
                     bwrestrict::Bool=true, sharpbw::Bool=false, covs_drop::Bool=true,
                     stdvars::Bool=false)
    ctx = "rd_estimate"
    0 < level < 1 || throw(ArgumentError("$ctx: level must be in (0, 1)"))
    E = _rd_extract(data, outcome, running; treatment, covariates, cluster, weights,
                    context=ctx)
    Z, znames = _rd_prepare_covariates(E.Z, E.covariates, covs_drop)
    p = p === nothing ? (deriv == 0 ? 1 : deriv + 1) : Int(p)
    q = q === nothing ? p + 1 : Int(q)
    method = _rd_bwselect(bwselect)
    S = _rd_setup(E.y, E.x, E.T, Z, E.C, E.W; c=Float64(cutoff), p, q,
                  deriv=Int(deriv), kernel=_rd_kernel(kernel), vce=_rd_vce(vce),
                  nnmatch=Int(nnmatch), masspoints=_rd_masspoints(masspoints),
                  bwcheck, bwrestrict, sharpbw, context=ctx)
    if rho !== nothing
        rho > 0 || throw(ArgumentError("$ctx: rho must be positive"))
    end
    if h !== nothing
        h_l, h_r = _rd_pair(h)
        (h_l > 0 && h_r > 0) || throw(ArgumentError("$ctx: h must be positive"))
        if b !== nothing
            b_l, b_r = _rd_pair(b)
            (b_l > 0 && b_r > 0) || throw(ArgumentError("$ctx: b must be positive"))
        elseif rho !== nothing
            b_l, b_r = h_l / rho, h_r / rho
        else
            b_l, b_r = h_l, h_r
        end
        # rdrobust: with h given and rho given, b = h / rho even if b is given.
        if rho !== nothing
            b_l, b_r = h_l / rho, h_r / rho
        end
        method_used = :manual
    elseif S.N < 20
        @warn "$ctx: fewer than 20 observations; using the whole sample as bandwidth"
        h_l = h_r = b_l = b_r = max(S.range_l, S.range_r)
        method_used = :manual
    else
        bws = if stdvars
            _rd_std_bandwidths(E.y, E.x, E.T, Z, E.C, E.W, [method]; c=Float64(cutoff),
                               p, q, deriv=Int(deriv), kernel=S.kernel, vce=_rd_vce(vce),
                               nnmatch=Int(nnmatch), masspoints=S.masspoints, bwcheck,
                               bwrestrict, sharpbw, scaleregul=Float64(scaleregul),
                               context=ctx, bwselect_style=false)
        else
            first(_rd_select_bandwidths(S, [method]; scaleregul=Float64(scaleregul)))
        end
        h_l, h_r, b_l, b_r = bws[method]
        if rho !== nothing
            b_l, b_r = h_l / rho, h_r / rho
        end
        method_used = method
    end
    return _rd_core(S; h_l, h_r, b_l, b_r, scalepar=Float64(scalepar),
                    level=Float64(level), bwselect=method_used, covariates=znames)
end

"""
    rd_weak_iv_confidence_set(r::RDEstimate; level=0.95) -> NamedTuple

Confidence set for the fuzzy RD effect that remains valid when the jump in treatment
take-up is small, obtained by inverting Anderson–Rubin-type tests built from robust
bias-corrected local polynomial jumps.

A fuzzy RD design is an instrumental-variables problem at the cutoff: the estimand
``\\theta = \\tau_Y / \\tau_D`` is the ratio of the jump in the outcome regression to the
jump in the take-up regression (Hahn, Todd & van der Klaauw 2001). When ``\\tau_D`` is
small relative to its sampling error, the delta-method interval around the ratio
estimate has poor coverage, as with weak instruments. Feir, Lemieux and Marmer (2016)
show this for fuzzy RD designs and propose to invert a test that does not divide by the
estimated first stage. For each candidate ``\\theta_0`` the null
``H_0: \\theta = \\theta_0`` implies that the jump in ``Y - \\theta_0 D`` is zero. The
statistic is
```math
t(\\theta_0) = \\frac{\\hat\\tau_Y - \\theta_0 \\hat\\tau_D}
    {(\\hat V_{YY} - 2\\theta_0 \\hat V_{YD} + \\theta_0^2 \\hat V_{DD})^{1/2}},
```
in the spirit of Anderson and Rubin (1949). The set collects the ``\\theta_0`` with
``|t(\\theta_0)| \\le z_{1-\\alpha/2}``. The inequality is quadratic in ``\\theta_0``,
so the set is computed in closed form. It can be a bounded interval, the union of two
unbounded rays, the whole real line (when the first stage is not significantly
different from zero), or empty.

**This is an adaptation, not the original procedure.** Feir, Lemieux and Marmer (2016)
build the statistic from conventional local polynomial estimates. Here
``\\hat\\tau_Y``, ``\\hat\\tau_D`` and ``\\hat V`` are the *robust bias-corrected* jumps
and their robust covariance from `r` (Calonico, Cattaneo & Titiunik 2014), so the test
inherits robust bias correction's treatment of smoothing bias at MSE-optimal
bandwidths. The bandwidths are those of `r`, chosen for the fuzzy ratio rather than for
the Anderson–Rubin statistic. The package claims no formal guarantee for the
combination beyond those of its two components; its coverage with strong and weak first
stages is checked by Monte Carlo (see the Validation page). For a bias-aware
alternative with guaranteed coverage under an explicit bound on the curvature of both
regressions, use [`rd_honest`](@ref) with
[`rd_honest_ar_confidence_set`](@ref) (Noack & Rothe 2024). Report the set as it is:
an unbounded set means the data do not determine the sign or size of the effect.

# Arguments
- `r::RDEstimate`: a fuzzy (or fuzzy kink) design estimated by [`rd_estimate`](@ref)
  with `treatment`. Sharp designs raise an `ArgumentError`.

# Keywords
- `level::Real=0.95`: confidence level; the test uses the normal critical value
  ``z_{1-\\alpha/2}`` with ``\\alpha = 1 - \\text{level}``.

# Returns
- `NamedTuple` with fields
  - `kind`: `:interval`, `:two_rays`, `:real_line` or `:empty` (`:two_rays` also covers
    the degenerate case of a single ray, when the quadratic term vanishes);
  - `intervals::Vector{Tuple{Float64,Float64}}`: the pieces of the set, with `±Inf` for
    unbounded ends;
  - `level`;
  - `first_stage_z`: robust ``z`` statistic of the first-stage jump.

# Examples
```julia
using DrSnow, DataFrames, StableRNGs
rng = StableRNG(1)
n = 5_000
x = 2 .* rand(rng, n) .- 1
d = Float64.(rand(rng, n) .< 0.4 .+ 0.1 .* (x .>= 0))   # weak first stage (0.1)
y = 1 .+ x .+ 2 .* d .+ randn(rng, n)
r = rd_estimate(DataFrame(; y, x, d), :y, :x; treatment=:d)
cs = rd_weak_iv_confidence_set(r)
cs.kind, cs.intervals, cs.first_stage_z
```

# References
- Feir, D., Lemieux, T., & Marmer, V. (2016). Weak identification in fuzzy regression
  discontinuity designs. *Journal of Business & Economic Statistics*, 34(2), 185–196.
- Anderson, T. W., & Rubin, H. (1949). Estimation of the parameters of a single
  equation in a complete system of stochastic equations. *The Annals of Mathematical
  Statistics*, 20(1), 46–63.
- Calonico, S., Cattaneo, M. D., & Titiunik, R. (2014). Robust nonparametric confidence
  intervals for regression-discontinuity designs. *Econometrica*, 82(6), 2295–2326.
- Hahn, J., Todd, P., & van der Klaauw, W. (2001). Identification and estimation of
  treatment effects with a regression-discontinuity design. *Econometrica*, 69(1),
  201–209.
- Noack, C., & Rothe, C. (2024). Bias-aware inference in fuzzy regression discontinuity
  designs. *Econometrica*, 92(3), 687–711.
"""
function rd_weak_iv_confidence_set(r::RDEstimate; level::Real=0.95)
    r.first_stage === nothing && throw(ArgumentError(
        "rd_weak_iv_confidence_set requires a fuzzy RD estimate (pass `treatment`)"))
    z = critical_value(level)
    a = r.reduced_form.tau_bias_corrected
    bT = r.first_stage.tau_bias_corrected
    Vyy, Vyt, Vtt = r.vcov_robust_yt[1, 1], r.vcov_robust_yt[1, 2], r.vcov_robust_yt[2, 2]
    # Non-rejection: (a - t b)^2 <= z^2 (Vyy - 2 t Vyt + t^2 Vtt)
    # <=> A t^2 + B t + C <= 0
    A = bT^2 - z^2 * Vtt
    B = -2 * a * bT + 2 * z^2 * Vyt
    C = a^2 - z^2 * Vyy
    disc = B^2 - 4 * A * C
    fs_z = bT / sqrt(Vtt)
    if abs(A) < eps(Float64) * max(bT^2, z^2 * Vtt)
        # Linear case
        if B == 0
            return (kind=C <= 0 ? :real_line : :empty,
                    intervals=C <= 0 ? [(-Inf, Inf)] : Tuple{Float64,Float64}[],
                    level=Float64(level), first_stage_z=fs_z)
        end
        t0 = -C / B
        iv = B > 0 ? (-Inf, t0) : (t0, Inf)
        return (kind=:two_rays, intervals=[iv], level=Float64(level), first_stage_z=fs_z)
    end
    if A > 0
        disc < 0 && return (kind=:empty, intervals=Tuple{Float64,Float64}[],
                            level=Float64(level), first_stage_z=fs_z)
        r1 = (-B - sqrt(disc)) / (2A)
        r2 = (-B + sqrt(disc)) / (2A)
        return (kind=:interval, intervals=[(min(r1, r2), max(r1, r2))],
                level=Float64(level), first_stage_z=fs_z)
    else
        disc < 0 && return (kind=:real_line, intervals=[(-Inf, Inf)],
                            level=Float64(level), first_stage_z=fs_z)
        r1 = (-B - sqrt(disc)) / (2A)
        r2 = (-B + sqrt(disc)) / (2A)
        lo, hi = min(r1, r2), max(r1, r2)
        return (kind=:two_rays, intervals=[(-Inf, lo), (hi, Inf)], level=Float64(level),
                first_stage_z=fs_z)
    end
end
