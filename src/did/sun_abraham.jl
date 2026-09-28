# Sun & Abraham (2021) interaction-weighted event-study estimator.

"""
    did_sun_abraham(data, outcome, treatment, unit, time;
                    max_pre=nothing, max_post=nothing, omit_period=nothing,
                    covariates=Symbol[], weights=nothing, cluster=unit, vcov=nothing,
                    anticipation=0) -> EventStudyEstimate
    did_sun_abraham(panel::TreatmentPanel; kwargs...) -> EventStudyEstimate

Interaction-weighted (IW) event-study estimator of Sun and Abraham (2021), robust to
treatment effects that differ across cohorts.

The building blocks are the cohort-specific average treatment effects on the treated
at event time ``e``,
``CATT_{g,e} = E[Y_{i,g+e}(g) - Y_{i,g+e}(\\infty) \\mid G_i = g]``, and the target is
their average over cohorts at each horizon,
``\\nu_e = \\sum_{g \\in C_e} \\Pr(G_i = g \\mid G_i \\in C_e)\\, CATT_{g,e}``, where
``C_e`` is the set of treated cohorts observed ``e`` periods after treatment.
Identification requires parallel trends between every treated cohort and the
comparison group, no anticipation (beyond `anticipation` periods) and an absorbing
treatment; unlike the dynamic TWFE specification, it does not require homogeneous
effects across cohorts. Sun and Abraham (2021) show that the TWFE coefficients mix
``CATT_{g,e'}`` from other relative periods with weights that can be negative, so
that heterogeneity alone can produce spurious pre-trends; the IW estimator avoids
this by estimating each ``CATT_{g,e}`` separately.

The estimator proceeds in two steps:

1. Estimate a regression with unit and period fixed effects and a full set of
   cohort × relative-period indicators ``1\\{G_i = g\\}\\, 1\\{t - g = e\\}`` for every
   treated cohort ``g`` and every observed ``e`` other than `omit_period`, using the
   never-treated units as the comparison group. When there are none, the last-treated
   cohort serves as the comparison group and the periods from its (anticipation-
   adjusted) treatment onwards are dropped. The coefficients estimate
   ``CATT_{g,e}``.
2. Average the ``CATT_{g,e}`` across cohorts at each ``e`` with weights equal to each
   cohort's share of the (weighted) observations at that relative period.

Because the regression is fully saturated in cohort × event time, reporting only a
window never lets periods outside it contaminate the reported coefficients. The
covariance of the aggregated effects is ``W V W'``, the delta method with the cohort
shares ``W`` treated as **fixed** (as in `fixest::sunab`). It therefore omits the
sampling variability of the estimated shares, which the asymptotic variance of Sun
and Abraham (2021) accounts for; the omission is usually small but can matter with
small cohorts. Standard errors are
cluster-robust (units by default) with ``G - 1`` degrees of freedom. Units treated
in the first period, which have no pre-treatment observation, are dropped. The
results reproduce `fixest::sunab`, against which the implementation is validated.
With never-treated units the IW estimator uses only them as controls; the imputation
estimator [`did_imputation`](@ref) and [`did_callaway_santanna`](@ref) with
`control_group = :not_yet_treated` also use not-yet-treated observations.

# Arguments
- `data`: a long-format panel.
- `outcome::Symbol`: outcome column.
- `treatment`: an absorbing 0/1 indicator or [`FirstTreated`](@ref)`(column)`.
- `unit::Symbol`, `time::Symbol`: unit and time columns (fixed effects).

# Keywords
- `max_pre = nothing`, `max_post = nothing`: report only relative periods from
  `-max_pre` to `max_post` (estimation is always fully saturated).
- `omit_period = nothing`: reference relative period (default
  `-1 - anticipation`); every cohort must be observed at it.
- `covariates::Vector{Symbol} = Symbol[]`: time-varying controls.
- `weights::Union{Nothing,Symbol} = nothing`: regression weights, also used for the
  cohort shares.
- `cluster = unit`: clustering column(s); `nothing` for heteroskedasticity-robust
  standard errors.
- `vcov = nothing`: a `FixedEffectModels` covariance estimator; takes precedence
  over `cluster`.
- `anticipation::Integer = 0`: number of anticipation periods.

# Returns
- `EventStudyEstimate`: the IW estimates ``\\hat\\nu_e`` with their covariance.
  `details` contains `cohort_effects` (a `DataFrame` of ``\\widehat{CATT}_{g,e}``,
  standard errors and aggregation weights), `aggregation_weights`, the saturated
  `model`, the `comparison` group used, and `att`, a [`DiDEstimate`](@ref) of the
  average post-treatment effect weighting every treated cohort–period cell by its
  number of observations (`fixest`'s `agg = "ATT"`).

# Examples
```julia
using DrSnow, CSV, DataFrames
file = joinpath(pkgdir(DrSnow), "test", "validation", "did", "mpdta.csv")
mpdta = CSV.read(file, DataFrame)
es = did_sun_abraham(mpdta, :lemp, FirstTreated(:first_treat), :countyreal, :year)
coeftable(es)
es.details.att                   # cell-size weighted post-treatment ATT
first(es.details.cohort_effects, 5)
```

# References
- Sun, L., & Abraham, S. (2021). Estimating dynamic treatment effects in event
  studies with heterogeneous treatment effects. *Journal of Econometrics*, 225(2),
  175–199.
- Goodman-Bacon, A. (2021). Difference-in-differences with variation in treatment
  timing. *Journal of Econometrics*, 225(2), 254–277.
- Roth, J., Sant'Anna, P. H. C., Bilinski, A., & Poe, J. (2023). What's trending in
  difference-in-differences? A synthesis of the recent econometrics literature.
  *Journal of Econometrics*, 235(2), 2218–2244.
- Bergé, L. (2026). fixest: Fast fixed-effects estimations. R package version
  0.14.2.
"""
function did_sun_abraham(data, outcome::Symbol, treatment, unit::Symbol, time::Symbol;
                         max_pre::Union{Nothing,Integer}=nothing,
                         max_post::Union{Nothing,Integer}=nothing,
                         omit_period::Union{Nothing,Integer}=nothing,
                         covariates::Vector{Symbol}=Symbol[],
                         weights::Union{Nothing,Symbol}=nothing,
                         cluster::Union{Nothing,Symbol,Vector{Symbol}}=unit,
                         vcov=nothing, anticipation::Integer=0)
    tcol = _did_treatment_column(treatment)
    df = _did_prepare(data, [outcome, tcol, unit, time, covariates..., weights,
                             _did_cluster_symbols(cluster)...];
                      context="did_sun_abraham", treatment=treatment)
    tm = treatment_timing(df, treatment, unit, time; anticipation=anticipation)
    tm.absorbing || throw(ArgumentError(
        "did_sun_abraham requires an absorbing (staggered-adoption) treatment"))
    ref = omit_period === nothing ? -1 - Int(anticipation) : Int(omit_period)
    # Drop units treated from the first (anticipation-adjusted) period.
    early = (tm.row_cohort .> 0) .& (tm.row_cohort .<= 1 + anticipation)
    if any(early)
        @warn "did_sun_abraham: dropping $(length(unique(tm.row_unit[early]))) " *
              "unit(s) treated in the first period (no pre-treatment period)"
        df = df[.!early, :]
        tm = treatment_timing(df, treatment, unit, time; anticipation=anticipation)
    end
    cohorts = _did_cohorts(tm)
    isempty(cohorts) && throw(ArgumentError("did_sun_abraham: no treated units"))
    control_note = "never-treated units"
    control_cohort = 0
    if _did_n_never(tm) == 0
        length(cohorts) >= 2 || throw(ArgumentError(
            "did_sun_abraham: no never-treated units and a single cohort; the " *
            "cohort-specific effects are not identified"))
        control_cohort = last(cohorts)
        control_note = "last-treated cohort ($(tm.periods[control_cohort])), periods " *
                       "before it is treated"
        cut = control_cohort - anticipation
        df = df[tm.row_period .< cut, :]
        # Re-deriving the timing on the shortened sample codes that cohort as 0.
        tm = treatment_timing(df, treatment, unit, time; anticipation=anticipation)
        cohorts = _did_cohorts(tm)
        isempty(cohorts) && throw(ArgumentError(
            "did_sun_abraham: no treated cohort left before the last cohort is treated"))
        @info "did_sun_abraham: no never-treated units; using the $control_note as " *
              "the comparison group"
    end
    et = _did_event_time(tm)
    ω = weights === nothing ? ones(nrow(df)) : float.(df[!, weights])
    cells = Tuple{Int,Int}[]
    cols = Symbol[]
    cellw = Float64[]
    for g in cohorts
        rows_g = tm.row_cohort .== g
        es_g = sort!(unique(et[rows_g]))
        ref in es_g || throw(ArgumentError(
            "did_sun_abraham: cohort $(tm.periods[g]) has no observations at the " *
            "reference period e = $ref"))
        for e in es_g
            e == ref && continue
            mask = rows_g .& (et .== e)
            c = _did_fresh_name(df, "cohort=$(tm.periods[g]):e=$e")
            df[!, c] = Float64.(mask)
            push!(cells, (g, e))
            push!(cols, c)
            push!(cellw, sum(ω[mask]))
        end
    end
    isempty(cols) && throw(ArgumentError("did_sun_abraham: no cohort × period cells"))
    f = make_formula(outcome, vcat(cols, covariates); fe=[unit, time])
    vc = _did_vcov_estimator(cluster, vcov)
    m = weights === nothing ? reg(df, f, vc) : reg(df, f, vc; weights=weights)
    idx = Int[]
    for ((g, e), c) in zip(cells, cols)
        j = try
            coef_index(m, c)
        catch err
            err isa ErrorException || rethrow()
            throw(ArgumentError("did_sun_abraham: CATT for cohort $(tm.periods[g]) at " *
                                "e = $e is not identified (collinear with the fixed " *
                                "effects)"))
        end
        push!(idx, j)
    end
    β = coef(m)[idx]
    Vβ = Matrix(StatsAPI.vcov(m)[idx, idx])
    rel_all = sort!(unique(last.(cells)))
    lo = max_pre === nothing ? typemin(Int) : -Int(max_pre)
    hi = max_post === nothing ? typemax(Int) : Int(max_post)
    rel = filter(e -> lo <= e <= hi, rel_all)
    isempty(rel) && throw(ArgumentError("did_sun_abraham: no relative periods in window"))
    Wm = zeros(length(rel), length(cells))
    for (r, e) in enumerate(rel)
        sel = findall(c -> c[2] == e, cells)
        Wm[r, sel] = cellw[sel] ./ sum(cellw[sel])
    end
    θ = Wm * β
    V = Wm * Vβ * Wm'
    # fixest's agg = "ATT": every post-treatment cell weighted by its observations.
    post = findall(c -> c[2] >= 0, cells)
    att = nothing
    if !isempty(post)
        wa = zeros(length(cells))
        wa[post] = cellw[post] ./ sum(cellw[post])
        att = DiDEstimate([dot(wa, β)], fill(dot(wa, Vβ * wa), 1, 1), ["ATT"], nobs(m),
                          dof_residual(m), _did_nclusters(m), _did_n_ever(tm),
                          _did_n_never(tm), length(tm.periods),
                          "Sun–Abraham (post-period cells weighted by size)",
                          "ATT (average over treated cohort-period cells)",
                          (weights=wa,))
    end
    se_β = sqrt.(max.(diag(Vβ), 0.0))
    catt = DataFrame(cohort=[tm.periods[c[1]] for c in cells], e=last.(cells),
                     estimate=β, std_error=se_β,
                     weight=[sum(Wm[:, k]) for k in eachindex(cells)])
    return EventStudyEstimate(rel, θ, Matrix(Symmetric(V)), [ref], nobs(m),
                              dof_residual(m), _did_nclusters(m),
                              "Sun–Abraham interaction-weighted event study",
                              "cohort-share weighted CATT at event time e (relative to " *
                              "e = $ref)", Float64[],
                              (model=m, timing=tm, binned=(false, false),
                               cohort_effects=catt, aggregation_weights=Wm, att=att,
                               comparison=control_note, n_treated=_did_n_ever(tm),
                               n_control=_did_n_never(tm), n_periods=length(tm.periods),
                               note=""))
end

did_sun_abraham(panel::TreatmentPanel; kwargs...) =
    did_sun_abraham(panel.data, panel.outcome, panel.treatment, panel.unit_id,
                    panel.time; covariates=panel.covariates, kwargs...)
