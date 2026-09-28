# Event-study estimation: a single entry point (`event_study`) that dispatches to the
# TWFE dynamic specification, Sun–Abraham, the imputation estimator or
# Callaway–Sant'Anna, all returning `EventStudyEstimate`.

const _DID_ES_ESTIMATORS = (:auto, :twfe, :sun_abraham, :imputation, :callaway_santanna)

"""
    event_study(data, outcome, treatment, unit, time;
                estimator=:auto, max_pre=nothing, max_post=nothing, omit_period=nothing,
                endpoints=:bin, covariates=Symbol[], weights=nothing, cluster=unit,
                vcov=nothing, anticipation=0, kwargs...) -> EventStudyEstimate
    event_study(panel::TreatmentPanel; kwargs...) -> EventStudyEstimate

Dynamic treatment effects by event time under staggered adoption, through a single
interface to four event-study estimators.

The target parameters are the average effects ``e`` periods after first treatment,
``\\beta_e = E[Y_{i,G_i+e}(G_i) - Y_{i,G_i+e}(\\infty) \\mid \\text{treated at } G_i]``,
for ``e \\ge 0``, averaged over the cohorts observed at that horizon, together with
placebo coefficients for ``e < 0``. Event time ``e = t - G_i`` is counted on the
ordered period index, so gaps and `Date`s are handled. Identification rests on
parallel trends in untreated potential outcomes and on no anticipation beyond
`anticipation` periods; treatment must be absorbing (staggered adoption). The
estimators differ in how they use the data and in what their coefficients mean when
treatment effects differ across cohorts, which is the typical case (Roth, Sant'Anna,
Bilinski and Poe, 2023; de Chaisemartin and D'Haultfœuille, 2023):

- `:twfe` estimates the dynamic two-way fixed effects specification
  ``Y_{it} = \\alpha_i + \\lambda_t + \\sum_{e \\ne e_0} \\beta_e 1\\{t - G_i = e\\} +
  X_{it}'\\gamma + \\varepsilon_{it}`` with reference period ``e_0`` (`omit_period`).
  It is valid with a single treatment cohort (or homogeneous effects across cohorts).
  With several cohorts and heterogeneous effects, each coefficient is a weighted
  combination of effects from several relative periods, including periods excluded
  from the specification, so even pre-treatment coefficients can be non-zero when
  parallel trends holds (Sun and Abraham, 2021); a warning is issued.
- `:sun_abraham` is the interaction-weighted estimator [`did_sun_abraham`](@ref),
  robust to heterogeneous effects across cohorts.
- `:imputation` is the imputation estimator of Borusyak, Jaravel and Spiess (2024),
  [`did_imputation`](@ref); its pre-treatment coefficients come from a separate
  regression on untreated observations and are not measured against a reference
  period.
- `:callaway_santanna` aggregates the group-time effects of
  [`did_callaway_santanna`](@ref) by event time with [`aggregate_att`](@ref).
- `:auto` (default) selects `:twfe` when all treated units start treatment in the
  same period and `:sun_abraham` otherwise.

For `:twfe`, an event window from `-max_pre` to `max_post` that is narrower than the
data raises the question of what to do with relative periods outside it. With
`endpoints = :bin` (default) they are pooled into the endpoint coefficients
(`e ≤ -max_pre` and `e ≥ max_post`). Binning is not innocuous: it imposes
that the effect is constant beyond each endpoint, and the endpoint coefficients are
averages over the binned periods; if effects keep changing beyond the window, the
restriction is violated and all coefficients can be biased (Schmidheiny and
Siegloch, 2023). With `endpoints = :trim` observations of treated units outside the
window are dropped instead, which avoids that restriction at the cost of sample
size, and without never-treated units the fully dynamic TWFE specification is not
identified (Borusyak, Jaravel and Spiess, 2024). Periods outside the window are
never pooled into the reference period. [`honest_did`](@ref) requires unbinned
coefficients, so use `endpoints = :trim` (or a window that covers all periods)
before a sensitivity analysis.

Inference for the regression-based estimators (`:twfe`, `:sun_abraham`) is
cluster-robust at the unit level by default with ``G - 1`` degrees of freedom; the
imputation and Callaway–Sant'Anna estimators use their own influence-function or
conservative variances with normal critical values. Report the whole path with
simultaneous confidence bands (`confint(es; uniform=true)`), interpret the
pre-treatment coefficients with the caveats of [`pre_trend_test`](@ref), and note
that the cohort composition behind ``\\beta_e`` usually changes with ``e``.

# Arguments
- `data`: a long-format panel (`DataFrame` or Tables.jl source).
- `outcome::Symbol`: outcome column.
- `treatment`: an absorbing 0/1 indicator ``D_{it}`` or
  [`FirstTreated`](@ref)`(column)`.
- `unit::Symbol`: unit identifier (unit fixed effects).
- `time::Symbol`: time column (period fixed effects).

# Keywords
- `estimator::Symbol = :auto`: `:auto`, `:twfe`, `:sun_abraham`, `:imputation` or
  `:callaway_santanna`, as described above.
- `max_pre = nothing`, `max_post = nothing`: the event window from `-max_pre` to
  `max_post`; `nothing` uses every observed relative period (fully dynamic). For
  `:sun_abraham` they only restrict the reported coefficients; for `:imputation`,
  `max_pre` is the number of pre-trend coefficients (default 0) and `max_post` the
  last horizon; for `:callaway_santanna` they set `min_e` and `max_e` of the
  aggregation.
- `omit_period = nothing`: the reference relative period normalized to zero
  (default `-1 - anticipation`); it must lie inside the window, have
  observations, and not be a binned endpoint.
- `endpoints::Symbol = :bin`: `:bin` pools relative periods beyond the window into
  the endpoint coefficients; `:trim` drops treated observations outside the window
  (`:twfe` only).
- `covariates::Vector{Symbol} = Symbol[]`: time-varying controls.
- `weights::Union{Nothing,Symbol} = nothing`: regression or sampling weights.
- `cluster = unit`: clustering column(s); `nothing` for heteroskedasticity-robust
  standard errors (regression-based estimators).
- `vcov = nothing`: a `FixedEffectModels` covariance estimator for `:twfe` and
  `:sun_abraham`; takes precedence over `cluster`.
- `anticipation::Integer = 0`: number of periods before treatment in which units may
  respond; it moves the default reference period, and for the robust estimators the
  last clean pre-treatment period, earlier.
- `kwargs...`: further keywords for the selected estimator (e.g. `control_group`,
  `method`, `rng` for `:callaway_santanna`); `:twfe` accepts none.

# Returns
- `EventStudyEstimate`: coefficients by relative period with their full covariance
  (`dof_residual = G - 1` under clustering for the regression-based estimators).
  For `:twfe`, `details` holds the fitted `model`, the [`TreatmentTiming`](@ref),
  the `binned` flags and the `endpoints` choice. Coefficients that are not
  identified (collinear with the fixed effects) and empty relative periods inside
  the window raise an error.

# Examples
```julia
using DrSnow, CSV, DataFrames, StableRNGs
file = joinpath(pkgdir(DrSnow), "test", "validation", "did", "mpdta.csv")
mpdta = CSV.read(file, DataFrame)
es = event_study(mpdta, :lemp, :d, :countyreal, :year)       # Sun–Abraham
relative_periods(es), coef(es)
confint(es; uniform=true, rng=StableRNG(1))
pre_trend_test(es)
# single cohort, TWFE with a trimmed window
sub = filter(r -> r.first_treat in (0, 2006), mpdta)
event_study(sub, :lemp, :d, :countyreal, :year; estimator=:twfe, max_pre=2,
            max_post=1, endpoints=:trim)
```

# References
- Sun, L., & Abraham, S. (2021). Estimating dynamic treatment effects in event
  studies with heterogeneous treatment effects. *Journal of Econometrics*, 225(2),
  175–199.
- Borusyak, K., Jaravel, X., & Spiess, J. (2024). Revisiting event-study designs:
  Robust and efficient estimation. *Review of Economic Studies*, 91(6), 3253–3285.
- Callaway, B., & Sant'Anna, P. H. C. (2021). Difference-in-differences with multiple
  time periods. *Journal of Econometrics*, 225(2), 200–230.
- Schmidheiny, K., & Siegloch, S. (2023). On event studies and distributed-lags in
  two-way fixed effects models: Identification, equivalence, and generalization.
  *Journal of Applied Econometrics*, 38(5), 695–713.
- Freyaldenhoven, S., Hansen, C., & Shapiro, J. M. (2019). Pre-event trends in the
  panel event-study design. *American Economic Review*, 109(9), 3307–3338.
- Roth, J., Sant'Anna, P. H. C., Bilinski, A., & Poe, J. (2023). What's trending in
  difference-in-differences? A synthesis of the recent econometrics literature.
  *Journal of Econometrics*, 235(2), 2218–2244.
- de Chaisemartin, C., & D'Haultfœuille, X. (2023). Two-way fixed effects and
  differences-in-differences with heterogeneous treatment effects: A survey. *The
  Econometrics Journal*, 26(3), C1–C30.
"""
function event_study(data, outcome::Symbol, treatment, unit::Symbol, time::Symbol;
                     estimator::Symbol=:auto, max_pre::Union{Nothing,Integer}=nothing,
                     max_post::Union{Nothing,Integer}=nothing,
                     omit_period::Union{Nothing,Integer}=nothing,
                     endpoints::Symbol=:bin, covariates::Vector{Symbol}=Symbol[],
                     weights::Union{Nothing,Symbol}=nothing,
                     cluster::Union{Nothing,Symbol,Vector{Symbol}}=unit, vcov=nothing,
                     anticipation::Integer=0, kwargs...)
    estimator in _DID_ES_ESTIMATORS || throw(ArgumentError(
        "estimator must be one of $(join(repr.(_DID_ES_ESTIMATORS), ", "))"))
    endpoints in (:bin, :trim) || throw(ArgumentError("endpoints must be :bin or :trim"))
    max_pre === nothing || max_pre >= 0 || throw(ArgumentError("max_pre must be ≥ 0"))
    max_post === nothing || max_post >= 0 || throw(ArgumentError("max_post must be ≥ 0"))
    if estimator === :auto
        tcol = _did_treatment_column(treatment)
        df = _did_prepare(data, [tcol, unit, time]; context="event_study",
                          treatment=treatment)
        tm = treatment_timing(df, treatment, unit, time)
        estimator = length(_did_cohorts(tm)) > 1 ? :sun_abraham : :twfe
    end
    if estimator === :twfe
        isempty(kwargs) || throw(ArgumentError(
            "unsupported keyword(s) for estimator=:twfe: $(join(keys(kwargs), ", "))"))
        return _did_event_study_twfe(data, outcome, treatment, unit, time; max_pre,
                                     max_post, omit_period, endpoints, covariates,
                                     weights, cluster, vcov, anticipation)
    elseif estimator === :sun_abraham
        return did_sun_abraham(data, outcome, treatment, unit, time; max_pre, max_post,
                               omit_period, covariates, weights, cluster, vcov,
                               anticipation, kwargs...)
    elseif estimator === :imputation
        vcov === nothing || throw(ArgumentError(
            "estimator=:imputation uses the Borusyak–Jaravel–Spiess variance; use " *
            "`cluster` instead of `vcov`"))
        horizons = max_post === nothing ? :all : 0:max_post
        return did_imputation(data, outcome, treatment, unit, time; horizons=horizons,
                              pretrends=max_pre === nothing ? 0 : max_pre, covariates,
                              weights, cluster=cluster isa Vector ? only(cluster) : cluster,
                              anticipation, kwargs...)
    else # :callaway_santanna
        vcov === nothing || throw(ArgumentError(
            "estimator=:callaway_santanna uses influence-function inference"))
        cs = did_callaway_santanna(data, outcome, treatment, unit, time; covariates,
                                   weights, anticipation,
                                   cluster=cluster == unit ? nothing :
                                           (cluster isa Vector ? only(cluster) : cluster),
                                   kwargs...)
        return aggregate_att(cs, :dynamic;
                             min_e=max_pre === nothing ? nothing : -max_pre,
                             max_e=max_post)
    end
end

event_study(panel::TreatmentPanel; kwargs...) =
    event_study(panel.data, panel.outcome, panel.treatment, panel.unit_id, panel.time;
                covariates=panel.covariates, kwargs...)

function _did_event_study_twfe(data, outcome, treatment, unit, time; max_pre, max_post,
                               omit_period, endpoints, covariates, weights, cluster,
                               vcov, anticipation)
    tcol = _did_treatment_column(treatment)
    df = _did_prepare(data, [outcome, tcol, unit, time, covariates..., weights,
                             _did_cluster_symbols(cluster)...];
                      context="event_study", treatment=treatment)
    tm = treatment_timing(df, treatment, unit, time; anticipation=anticipation)
    omit_period = omit_period === nothing ? -1 - anticipation : Int(omit_period)
    tm.absorbing || throw(ArgumentError(
        "event_study requires an absorbing treatment (no unit leaves treatment); " *
        "event time is not defined for treatments that switch on and off"))
    cohorts = _did_cohorts(tm)
    isempty(cohorts) && throw(ArgumentError("event_study: no treated units"))
    if length(cohorts) > 1
        @warn "event_study(estimator=:twfe) with $(length(cohorts)) treatment cohorts: " *
              "TWFE event-study coefficients are contaminated by effects from other " *
              "relative periods when effects differ across cohorts (Sun & Abraham " *
              "2021). Use estimator=:sun_abraham, :imputation or :callaway_santanna."
    end
    et = _did_event_time(tm)
    trows = tm.row_cohort .> 0
    emin, emax = extrema(et[trows])
    lo = max_pre === nothing ? emin : -Int(max_pre)
    hi = max_post === nothing ? emax : Int(max_post)
    lo <= omit_period <= hi || throw(ArgumentError(
        "omit_period = $omit_period lies outside the event window [$lo, $hi]"))
    lo < emin && throw(ArgumentError(
        "no treated observations at relative period $lo (earliest observed: $emin); " *
        "reduce max_pre"))
    hi > emax && throw(ArgumentError(
        "no treated observations at relative period $hi (latest observed: $emax); " *
        "reduce max_post"))
    bin_lo = endpoints === :bin && emin < lo
    bin_hi = endpoints === :bin && emax > hi
    if (bin_lo && omit_period == lo) || (bin_hi && omit_period == hi)
        throw(ArgumentError("omit_period = $omit_period is a binned endpoint, which " *
                            "would pool distant periods into the reference; widen the " *
                            "window or use endpoints=:trim"))
    end
    keep = endpoints === :trim ? .!(trows .& ((et .< lo) .| (et .> hi))) :
           trues(nrow(df))
    if !all(keep)
        df = df[keep, :]
        et = et[keep]
        trows = trows[keep]
    end
    eb = [trows[i] ? clamp(et[i], lo, hi) : typemin(Int) for i in eachindex(et)]
    count(==(omit_period), eb) > 0 || throw(ArgumentError(
        "no observations at the reference period $omit_period"))
    rel = [k for k in lo:hi if k != omit_period]
    cols = Symbol[]
    for k in rel
        c = _did_fresh_name(df, "event_time=$k")
        df[!, c] = Float64.(eb .== k)
        any(eb .== k) || throw(ArgumentError(
            "no observations at relative period $k inside the event window"))
        push!(cols, c)
    end
    f = make_formula(outcome, vcat(cols, covariates); fe=[unit, time])
    vc = _did_vcov_estimator(cluster, vcov)
    m = weights === nothing ? reg(df, f, vc) : reg(df, f, vc; weights=weights)
    idx = Int[]
    for (k, c) in zip(rel, cols)
        j = try
            coef_index(m, c)
        catch err
            err isa ErrorException || rethrow()
            throw(ArgumentError(
                "event-study coefficient for relative period $k is not identified " *
                "(collinear with the unit and time fixed effects). A fully dynamic " *
                "TWFE specification needs never-treated units; otherwise restrict the " *
                "window (max_pre/max_post) or use estimator=:imputation"))
        end
        push!(idx, j)
    end
    V = Matrix(StatsAPI.vcov(m)[idx, idx])
    single = length(cohorts) == 1
    method = "TWFE event study" * (bin_lo || bin_hi ? " (binned endpoints)" :
                                   endpoints === :trim ? " (trimmed window)" : "")
    return EventStudyEstimate(rel, coef(m)[idx], V, [omit_period], nobs(m),
                              dof_residual(m), _did_nclusters(m), method,
                              single ? "ATT at event time e relative to e = $omit_period" :
                              "TWFE event-study coefficients (mix cohorts)",
                              Float64[], (model=m, timing=tm, binned=(bin_lo, bin_hi),
                                          endpoints=endpoints, n_treated=_did_n_ever(tm),
                                          n_control=_did_n_never(tm),
                                          n_periods=length(tm.periods), note=""))
end
