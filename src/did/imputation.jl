# Borusyak, Jaravel & Spiess (2024) imputation estimator.
#
# Notation: Ω₀ = untreated observations (never treated, or before G_i - anticipation),
# Ω₁ = treated observations. With first-stage design Z = [unit dummies, time dummies
# (one dropped), covariates] and Ω = diag(regression weights):
#   γ̂ = (Z₀'ΩZ₀)⁻¹ Z₀'Ω Y₀,   τ̂_it = Y_it - Z_it γ̂  (it ∈ Ω₁),
#   θ̂_w = Σ_{Ω₁} w_it τ̂_it = Σ_all v_it Y_it,
#   v = w on Ω₁,  v₀ = -Ω Z₀ (Z₀'ΩZ₀)⁻¹ Z₁' w  on Ω₀.
# Conservative variance (BJS Theorem 3): Σ_clusters (Σ v_it ε̃_it)², with ε̃ the
# first-stage residual on Ω₀ and τ̂_it - τ̄_{g,t} on Ω₁, where τ̄ averages τ̂ within
# cohort × period cells with weights v².

"""
    did_imputation(data, outcome, treatment, unit, time;
                   horizons=nothing, pretrends=0, covariates=Symbol[],
                   weights=nothing, cluster=unit, anticipation=0)
        -> Union{DiDEstimate, EventStudyEstimate}
    did_imputation(panel::TreatmentPanel; kwargs...)

Imputation estimator of Borusyak, Jaravel and Spiess (2024) for staggered adoption:
untreated potential outcomes of treated observations are imputed from a two-way
fixed effects model fitted on untreated observations only.

The estimands are weighted averages of the individual effects
``\\tau_{it} = Y_{it}(1) - Y_{it}(0)`` over treated observations: the overall ATT
(all treated observations, weighted by `weights` if given) and, for each horizon
``h``, the average over observations ``h`` periods after first treatment. The model
for untreated potential outcomes is ``Y_{it}(0) = \\alpha_i + \\lambda_t +
X_{it}'\\beta + \\varepsilon_{it}`` with ``E[\\varepsilon_{it}] = 0`` for all
observations, which is parallel trends (in all periods, conditional on the linear
covariate term) combined with no anticipation beyond `anticipation` periods; treatment
effects are left unrestricted. The estimator has three steps:

1. Fit the fixed effects model on untreated observations (never-treated units, and
   observations before period `G_i - anticipation`).
2. Impute ``\\hat Y_{it}(0)`` for every treated observation and form
   ``\\hat\\tau_{it} = Y_{it} - \\hat Y_{it}(0)``.
3. Average ``\\hat\\tau_{it}`` over the treated observations of each estimand.

Borusyak, Jaravel and Spiess (2024, Theorem 2) show that among linear unbiased
estimators of such averages the imputation estimator is efficient when the errors
``\\varepsilon_{it}`` are homoskedastic and serially uncorrelated (spherical); with
serially correlated or heteroskedastic errors that optimality no longer holds,
although the estimator remains unbiased and consistent. It uses not-yet-treated
observations as controls and so exploits all pre-treatment periods, which makes it
more precise than [`did_callaway_santanna`](@ref) or [`did_sun_abraham`](@ref)
under parallel trends but also more reliant on parallel trends holding over the
whole pre-period; the extended TWFE estimator of Wooldridge (2025),
[`did_etwfe`](@ref) with its default not-yet-treated controls, gives the same point
estimates in balanced panels without covariates, and the two-stage estimator of
Gardner (2022) is equivalent as well.

Standard errors use the conservative clustered variance of Borusyak, Jaravel and
Spiess (2024, Theorem 3), in which ``\\hat\\tau_{it}`` is demeaned within cohort ×
period cells; the variance is conservative because treatment-effect heterogeneity
within these cells cannot be separated from noise. Inference uses normal critical
values (`dof_residual = Inf`). The pre-trend check (their Test 1) regresses the
outcome on unit and period fixed effects, covariates and indicators for the
`pretrends` periods before treatment, on untreated observations only; the lead
coefficients are zero under parallel trends and no anticipation. They are estimated
separately from the post-treatment effects and are not measured against a reference
period, so they are not comparable to TWFE-style event-study coefficients and cannot
be used by [`honest_did`](@ref) (Roth, 2026). Treated observations whose unit or
period has no untreated observation cannot be imputed and are dropped with a
warning; the untreated observations must connect all units and periods. Results
match the R package `didimputation`.

# Arguments
- `data`: a long-format panel.
- `outcome::Symbol`: outcome column.
- `treatment`: an absorbing 0/1 indicator or [`FirstTreated`](@ref)`(column)`.
- `unit::Symbol`, `time::Symbol`: unit and time columns.

# Keywords
- `horizons = nothing`: `nothing` for the overall ATT only, `:all` for every observed
  event time `h ≥ -anticipation`, or a collection of event times.
- `pretrends::Integer = 0`: number of pre-treatment lead coefficients
  (`e = -1, …, -pretrends`, shifted by `anticipation`).
- `covariates::Vector{Symbol} = Symbol[]`: time-varying controls in the model for
  ``Y_{it}(0)``.
- `weights::Union{Nothing,Symbol} = nothing`: observation weights, used both in the
  first-step regression and in the averages.
- `cluster::Union{Nothing,Symbol} = unit`: clustering column for the variance.
- `anticipation::Integer = 0`: number of anticipation periods.

# Returns
- `DiDEstimate` (overall ATT) when `horizons === nothing` and `pretrends == 0`.
- Otherwise an `EventStudyEstimate` with the lead coefficients (negative relative
  periods) followed by the horizon effects and their joint covariance, so that
  [`pre_trend_test`](@ref) gives the Borusyak–Jaravel–Spiess pre-trend test; its
  `details.att` is the overall ATT and `reference` is empty.

# Examples
```julia
using DrSnow, CSV, DataFrames
file = joinpath(pkgdir(DrSnow), "test", "validation", "did", "mpdta.csv")
mpdta = CSV.read(file, DataFrame)
att = did_imputation(mpdta, :lemp, FirstTreated(:first_treat), :countyreal, :year)
es = did_imputation(mpdta, :lemp, FirstTreated(:first_treat), :countyreal, :year;
                    horizons=:all, pretrends=3)
pre_trend_test(es)
```

# References
- Borusyak, K., Jaravel, X., & Spiess, J. (2024). Revisiting event-study designs:
  Robust and efficient estimation. *Review of Economic Studies*, 91(6), 3253–3285.
- Gardner, J. (2022). Two-stage differences in differences. Working paper,
  arXiv:2207.05943.
- Wooldridge, J. M. (2025). Two-way fixed effects, the two-way Mundlak regression,
  and difference-in-differences estimators. *Empirical Economics*, 69(5),
  2545–2587.
- Liu, L., Wang, Y., & Xu, Y. (2024). A practical guide to counterfactual estimators
  for causal inference with time-series cross-sectional data. *American Journal of
  Political Science*, 68(1), 160–176.
- Roth, J. (2026). Interpreting event-studies from recent difference-in-differences
  methods. *The Japanese Economic Review*, 77(2), 275–288.
- Butts, K. (2026). didimputation: Imputation estimator from Borusyak, Jaravel, and
  Spiess (2021). R package version 0.5.1.
"""
function did_imputation(data, outcome::Symbol, treatment, unit::Symbol, time::Symbol;
                        horizons=nothing, pretrends::Integer=0,
                        covariates::Vector{Symbol}=Symbol[],
                        weights::Union{Nothing,Symbol}=nothing,
                        cluster::Union{Nothing,Symbol}=unit, anticipation::Integer=0)
    pretrends >= 0 || throw(ArgumentError("pretrends must be ≥ 0"))
    tcol = _did_treatment_column(treatment)
    clcol = cluster === nothing ? unit : cluster
    df = _did_prepare(data, [outcome, tcol, unit, time, covariates..., weights, clcol];
                      context="did_imputation", treatment=treatment)
    tm = treatment_timing(df, treatment, unit, time; anticipation=anticipation)
    tm.absorbing || throw(ArgumentError(
        "did_imputation requires an absorbing (staggered-adoption) treatment"))
    δ = Int(anticipation)
    n = nrow(df)
    G = tm.row_cohort
    P = tm.row_period
    et = _did_event_time(tm)
    treated = [G[i] > 0 && P[i] >= G[i] - δ for i in 1:n]
    any(treated) || throw(ArgumentError("did_imputation: no treated observations"))
    untreated = .!treated
    # Imputability: unit and period must appear among untreated observations.
    u_ok = falses(length(tm.units))
    p_ok = falses(length(tm.periods))
    for i in findall(untreated)
        u_ok[tm.row_unit[i]] = true
        p_ok[P[i]] = true
    end
    imputable = [u_ok[tm.row_unit[i]] && p_ok[P[i]] for i in 1:n]
    bad = treated .& .!imputable
    if any(bad)
        @warn "did_imputation: dropping $(count(bad)) treated observation(s) whose " *
              "unit or period has no untreated observation (effects not identified)"
        keep = .!bad
        df = df[keep, :]
        tm = treatment_timing(df, treatment, unit, time; anticipation=anticipation)
        n = nrow(df)
        G, P, et = tm.row_cohort, tm.row_period, _did_event_time(tm)
        treated = treated[keep]
        untreated = .!treated
        any(treated) || throw(ArgumentError(
            "did_imputation: no treated observation can be imputed"))
    end
    y = float.(df[!, outcome])
    ω = weights === nothing ? ones(n) : float.(df[!, weights])
    any(<(0), ω) && throw(ArgumentError("weights must be non-negative"))
    _did_fe_connected(tm, untreated) || throw(ArgumentError(
        "did_imputation: the untreated observations do not connect all units and " *
        "periods, so the unit and time effects are not identified"))
    Z, _ = _did_fe_design(tm, df, covariates, untreated)
    i0 = findall(untreated)
    i1 = findall(treated)
    Z0 = Z[i0, :]
    Z1 = Z[i1, :]
    A = Symmetric(sparse(Z0' * (ω[i0] .* Z0)))
    F = try
        cholesky(A)
    catch err
        err isa Union{PosDefException,LinearAlgebra.SingularException,
                      LinearAlgebra.ZeroPivotException} || rethrow()
        throw(ArgumentError("did_imputation: the first stage is not identified on the " *
                            "untreated observations (disconnected units/periods or " *
                            "collinear covariates)"))
    end
    γ = F \ (Z0' * (ω[i0] .* y[i0]))
    ε0 = y[i0] .- Z0 * γ
    τ = y[i1] .- Z1 * γ
    # Targets: overall ATT (+ horizons).
    targets = Tuple{Symbol,Int}[(:att, 0)]
    if horizons !== nothing
        hs = horizons === :all ? sort!(unique(et[i1])) : sort!(unique(collect(horizons)))
        for h in hs
            any(==(h), et[i1]) || throw(ArgumentError(
                "did_imputation: no treated observations at horizon $h"))
            push!(targets, (:h, h))
        end
    end
    K = length(targets)
    W1 = zeros(length(i1), K)
    for (k, (kind, h)) in enumerate(targets)
        sel = kind === :att ? trues(length(i1)) : et[i1] .== h
        W1[sel, k] = ω[i1][sel]
        W1[:, k] ./= sum(W1[:, k])
    end
    θ = W1' * τ
    V0 = -(ω[i0] .* (Z0 * (F \ Matrix(Z1' * W1))))     # weights on untreated Y
    # Residuals for the variance: τ̂ demeaned within cohort × period using v² weights.
    cellkey = [(G[i], P[i]) for i in i1]
    E1 = zeros(length(i1), K)
    for k in 1:K
        num = Dict{Tuple{Int,Int},Float64}()
        den = Dict{Tuple{Int,Int},Float64}()
        for (r, c) in enumerate(cellkey)
            v2 = W1[r, k]^2
            num[c] = get(num, c, 0.0) + v2 * τ[r]
            den[c] = get(den, c, 0.0) + v2
        end
        for (r, c) in enumerate(cellkey)
            τbar = den[c] > 0 ? num[c] / den[c] : 0.0
            E1[r, k] = τ[r] - τbar
        end
    end
    cl = df[!, clcol]
    # Score matrix (rows = observations) for the imputation targets.
    S = zeros(n, K)
    S[i0, :] = V0 .* ε0
    S[i1, :] = W1 .* E1
    lead_names = Int[]
    Xpre = nothing
    if pretrends > 0
        leads = [-δ - j for j in pretrends:-1:1]
        Xl = zeros(length(i0), length(leads))
        for (j, e) in enumerate(leads)
            Xl[:, j] = Float64.(et[i0] .== e)
            any(Xl[:, j] .> 0) || throw(ArgumentError(
                "did_imputation: no untreated observations at relative period $e; " *
                "reduce pretrends"))
        end
        # FWL: residualize the leads on the first-stage design (within Ω₀).
        Xt = Xl .- Z0 * (F \ Matrix(Z0' * (ω[i0] .* Xl)))
        XtX = Symmetric(Xt' * (ω[i0] .* Xt))
        isposdef(XtX) || throw(ArgumentError(
            "did_imputation: pre-trend coefficients not identified; reduce pretrends"))
        βpre = XtX \ (Xt' * (ω[i0] .* ε0))
        epre = ε0 .- Xt * βpre
        Spre = zeros(n, length(leads))
        Spre[i0, :] = (ω[i0] .* epre .* Xt) / XtX
        # Reference small-sample scaling from FixedEffectModels (as fixest does).
        c = _did_imputation_pre_scale(df[i0, :], outcome, leads, et[i0], covariates,
                                      unit, time, weights, clcol, βpre, Spre[i0, :])
        S = hcat(Spre .* sqrt(c), S)
        θ = vcat(βpre, θ)
        lead_names = leads
    end
    Ssum = _cluster_sums(S, cl)
    Vall = Matrix(Symmetric(Ssum' * Ssum))
    ncl = size(Ssum, 1)
    npre = length(lead_names)
    iatt = npre + 1
    att = DiDEstimate([θ[iatt]], Vall[iatt:iatt, iatt:iatt], ["ATT"], n, Inf, ncl,
                      _did_n_ever(tm), _did_n_never(tm), length(tm.periods),
                      "Borusyak–Jaravel–Spiess imputation",
                      "ATT (average of imputed effects over treated observations)",
                      (n_treated_obs=length(i1), n_untreated_obs=length(i0),
                       effects=DataFrame(unit=tm.units[tm.row_unit[i1]],
                                         time=tm.periods[P[i1]], event_time=et[i1],
                                         effect=τ)))
    (horizons === nothing && npre == 0) && return att
    keep = vcat(1:npre, (npre + 2):length(θ))
    rel = vcat(lead_names, [h for (kind, h) in targets[2:end]])
    return EventStudyEstimate(rel, θ[keep], Vall[keep, keep], Int[], n, Inf, ncl,
                              "Borusyak–Jaravel–Spiess imputation event study",
                              "average imputed effect at horizon e (leads: pre-trend " *
                              "coefficients on untreated observations)", Float64[],
                              (att=att, timing=tm, binned=(false, false),
                               n_pretrend=npre, n_treated=_did_n_ever(tm),
                               n_control=_did_n_never(tm), n_periods=length(tm.periods),
                               note=npre > 0 ? _DID_BJS_PRETREND_NOTE : ""))
end

const _DID_BJS_PRETREND_NOTE = "Negative relative periods are BJS pre-trend " *
                                "coefficients (Test 1), not imputed effects."

did_imputation(panel::TreatmentPanel; kwargs...) =
    did_imputation(panel.data, panel.outcome, panel.treatment, panel.unit_id, panel.time;
                   covariates=panel.covariates, kwargs...)

# Sparse first-stage design: dummies for units and periods present in the untreated
# sample (the first such period dropped), plus covariates.
function _did_fe_design(tm, df, covariates, untreated)
    n = nrow(df)
    units0 = sort!(unique(tm.row_unit[untreated]))
    periods0 = sort!(unique(tm.row_period[untreated]))
    ucol = Dict(u => j for (j, u) in enumerate(units0))
    pcol = Dict(p => length(units0) + j - 1 for (j, p) in enumerate(periods0) if j > 1)
    ncov = length(covariates)
    ncol = length(units0) + length(periods0) - 1 + ncov
    I = Int[]
    J = Int[]
    Vv = Float64[]
    for i in 1:n
        u = get(ucol, tm.row_unit[i], 0)
        u > 0 && (push!(I, i); push!(J, u); push!(Vv, 1.0))
        p = get(pcol, tm.row_period[i], 0)
        p > 0 && (push!(I, i); push!(J, p); push!(Vv, 1.0))
    end
    base = length(units0) + length(periods0) - 1
    for (k, c) in enumerate(covariates)
        col = df[!, c]
        eltype(col) <: Union{Missing,Real} ||
            throw(ArgumentError("covariate `$c` must be numeric"))
        for i in 1:n
            push!(I, i)
            push!(J, base + k)
            push!(Vv, float(col[i]))
        end
    end
    return sparse(I, J, Vv, n, ncol), (units0, periods0)
end

# Ratio between FixedEffectModels' clustered variance of the lead coefficients
# (with its small-sample corrections, as in fixest) and the raw cluster-sum variance.
function _did_imputation_pre_scale(d0, outcome, leads, et0, covariates, unit, time,
                                   weights, clcol, βpre, Spre0)
    d0 = copy(d0)
    cols = Symbol[]
    for e in leads
        c = _did_fresh_name(d0, "lead=$e")
        d0[!, c] = Float64.(et0 .== e)
        push!(cols, c)
    end
    f = make_formula(outcome, vcat(cols, covariates); fe=[unit, time])
    vc = Vcov.cluster(clcol)
    m = weights === nothing ? reg(d0, f, vc; drop_singletons=false) :
        reg(d0, f, vc; weights=weights, drop_singletons=false)
    idx = [coef_index(m, c) for c in cols]
    isapprox(coef(m)[idx], βpre; rtol=1e-6, atol=1e-8) ||
        @warn "did_imputation: pre-trend coefficients differ from FixedEffectModels"
    Vf = StatsAPI.vcov(m)[idx, idx]
    Ss = _cluster_sums(Spre0, d0[!, clcol])
    Vraw = Ss' * Ss
    return mean(diag(Vf) ./ diag(Vraw))
end

# Are all units and periods with untreated observations in one connected component of
# the bipartite unit–period graph formed by the untreated observations?
function _did_fe_connected(tm, untreated)
    N = length(tm.units)
    parent = collect(1:(N + length(tm.periods)))
    findroot(x) = (while parent[x] != x
                       parent[x] = parent[parent[x]]
                       x = parent[x]
                   end; x)
    used = falses(length(parent))
    for i in findall(untreated)
        a, b = tm.row_unit[i], N + tm.row_period[i]
        used[a] = used[b] = true
        ra, rb = findroot(a), findroot(b)
        ra != rb && (parent[ra] = rb)
    end
    roots = Set(findroot(x) for x in findall(used))
    return length(roots) <= 1
end
