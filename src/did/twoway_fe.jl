# Two-way fixed effects difference-in-differences.

"""
    did_twfe(data, outcome, treatment, unit, time;
             covariates=Symbol[], weights=nothing, cluster=unit, vcov=nothing,
             warn_heterogeneity=true) -> DiDEstimate
    did_twfe(panel::TreatmentPanel; kwargs...) -> DiDEstimate

Two-way fixed effects (TWFE) difference-in-differences regression of the outcome on a
treatment indicator, unit fixed effects and period fixed effects.

The estimated model is

```math
Y_{it} = \\alpha_i + \\lambda_t + \\tau D_{it} + X_{it}'\\beta + \\varepsilon_{it},
```

fitted by least squares with `FixedEffectModels.reg`. In the canonical design, in
which one group of units is treated from a common date and the others are never
treated, ``\\tau`` identifies the average treatment effect on the treated,
``ATT = E[Y_{it}(1) - Y_{it}(0) \\mid D_{it} = 1]``, under two assumptions: *parallel
trends*, that in the absence of treatment the average untreated potential outcome
``Y_{it}(0)`` would have evolved in the same way in the treated and comparison
groups, and *no anticipation*, that treatment has no effect before it starts
(Ashenfelter and Card, 1985; Card and Krueger, 1994; Angrist and Pischke, 2009;
Roth, Sant'Anna, Bilinski and Poe, 2023). Parallel trends restricts post-treatment
counterfactuals and cannot be tested; pre-treatment data speak only to parallel
*pre*-trends (see [`pre_trend_test`](@ref)). It is also specific to the scale of the
outcome: except under strong conditions, it cannot hold for both ``Y`` and
``\\log Y`` (Roth and Sant'Anna, 2023). Time-varying covariates enter linearly; if
they are affected by treatment they are "bad controls" and bias ``\\hat\\tau``.

With **staggered adoption** (several treatment cohorts) or a **non-absorbing**
treatment, ``\\tau`` is no longer an ATT even under parallel trends: it is a weighted
average of the unit–period treatment effects in which some weights can be negative,
because already-treated units serve as controls for later-treated ones
(de Chaisemartin and D'Haultfœuille, 2020; Goodman-Bacon, 2021; Borusyak, Jaravel
and Spiess, 2024). When effects vary over time or across cohorts, ``\\hat\\tau`` can
have the opposite sign of every underlying effect. `did_twfe` detects these designs
and, with `warn_heterogeneity = true`, warns and reports the number and total of
the negative weights (see [`twfe_weights`](@ref) and [`bacon_decomposition`](@ref)).
In such designs prefer a heterogeneity-robust estimator:
[`did_callaway_santanna`](@ref), [`did_sun_abraham`](@ref),
[`did_imputation`](@ref) or [`did_etwfe`](@ref) (see the surveys by de Chaisemartin
and D'Haultfœuille, 2023, and Roth et al., 2023).

Inference is by default cluster-robust at the unit level, which allows arbitrary
serial correlation of ``\\varepsilon_{it}`` within units; ignoring that correlation
grossly overstates precision in DiD applications (Bertrand, Duflo and Mullainathan,
2004). Cluster at the level at which treatment is assigned (e.g. the state for a
state policy; Abadie, Athey, Imbens and Wooldridge, 2023). Confidence intervals use
the t distribution with ``G - 1`` degrees of freedom for ``G`` clusters. Cluster-robust
inference is unreliable with few clusters and, in particular, with few *treated*
clusters, where it can over-reject severely (Cameron, Gelbach and Miller, 2008;
Conley and Taber, 2011; MacKinnon, Nielsen and Webb, 2023); consider
randomization inference or wild-bootstrap methods there.

# Arguments
- `data`: a long-format panel (one row per unit and period; `DataFrame` or Tables.jl
  source). Rows with missing values in the used columns are dropped with a warning.
- `outcome::Symbol`: outcome column.
- `treatment`: a 0/1 treatment-indicator column ``D_{it}`` (absorbing or not), or
  [`FirstTreated`](@ref)`(column)` giving each unit's first treated period.
- `unit::Symbol`: unit identifier; absorbed as unit fixed effects.
- `time::Symbol`: time-period column; absorbed as period fixed effects.

# Keywords
- `covariates::Vector{Symbol} = Symbol[]`: time-varying controls ``X_{it}``; they
  should not be affected by treatment.
- `weights::Union{Nothing,Symbol} = nothing`: regression-weight column (e.g.
  population weights for aggregate units).
- `cluster = unit`: clustering column(s) for the variance; `nothing` gives
  heteroskedasticity-robust standard errors, which ignore serial correlation and are
  rarely appropriate in panels.
- `vcov = nothing`: a `FixedEffectModels` covariance estimator (e.g.
  `Vcov.cluster(:state)`); takes precedence over `cluster`.
- `warn_heterogeneity::Bool = true`: warn, with the negative-weight diagnostic, when
  the design is staggered or treatment is non-absorbing.

# Returns
- `DiDEstimate`: `coef` is ``\\hat\\tau``; `dof_residual` is ``G - 1`` under
  clustering; `n_treated` and `n_control` count ever-treated and never-treated
  units. `details` holds the fitted model (`model`, with the covariance of all
  regressors), the [`TreatmentTiming`](@ref) (`timing`), the flags `staggered` and
  `absorbing`, and the [`TWFEWeights`](@ref) diagnostic (`twfe_weights`) for
  staggered or non-absorbing designs.

# Examples
```julia
using DrSnow, CSV, DataFrames
file = joinpath(pkgdir(DrSnow), "test", "validation", "did", "mpdta.csv")
mpdta = CSV.read(file, DataFrame)
# a single-cohort design: the 2004 cohort against never-treated counties
sub = filter(r -> r.first_treat in (0, 2004), mpdta)
r = did_twfe(sub, :lemp, :d, :countyreal, :year)
coef(r), stderror(r), confint(r; level=0.90)
did_twfe(sub, :lemp, :d, :countyreal, :year; covariates=[:lpop])
```

# References
- Angrist, J. D., & Pischke, J.-S. (2009). *Mostly Harmless Econometrics: An
  Empiricist's Companion*. Princeton University Press.
- Roth, J., Sant'Anna, P. H. C., Bilinski, A., & Poe, J. (2023). What's trending in
  difference-in-differences? A synthesis of the recent econometrics literature.
  *Journal of Econometrics*, 235(2), 2218–2244.
- de Chaisemartin, C., & D'Haultfœuille, X. (2020). Two-way fixed effects estimators
  with heterogeneous treatment effects. *American Economic Review*, 110(9),
  2964–2996.
- Goodman-Bacon, A. (2021). Difference-in-differences with variation in treatment
  timing. *Journal of Econometrics*, 225(2), 254–277.
- de Chaisemartin, C., & D'Haultfœuille, X. (2023). Two-way fixed effects and
  differences-in-differences with heterogeneous treatment effects: A survey. *The
  Econometrics Journal*, 26(3), C1–C30.
- Bertrand, M., Duflo, E., & Mullainathan, S. (2004). How much should we trust
  differences-in-differences estimates? *Quarterly Journal of Economics*, 119(1),
  249–275.
- Cameron, A. C., Gelbach, J. B., & Miller, D. L. (2008). Bootstrap-based
  improvements for inference with clustered errors. *Review of Economics and
  Statistics*, 90(3), 414–427.
- Abadie, A., Athey, S., Imbens, G. W., & Wooldridge, J. M. (2023). When should you
  adjust standard errors for clustering? *Quarterly Journal of Economics*, 138(1),
  1–35.
- Roth, J., & Sant'Anna, P. H. C. (2023). When is parallel trends sensitive to
  functional form? *Econometrica*, 91(2), 737–747.
"""
function did_twfe(data, outcome::Symbol, treatment, unit::Symbol, time::Symbol;
                  covariates::Vector{Symbol}=Symbol[],
                  weights::Union{Nothing,Symbol}=nothing,
                  cluster::Union{Nothing,Symbol,Vector{Symbol}}=unit,
                  vcov=nothing, warn_heterogeneity::Bool=true)
    tcol = _did_treatment_column(treatment)
    df = _did_prepare(data, [outcome, tcol, unit, time, covariates..., weights,
                             _did_cluster_symbols(cluster)...];
                      context="did_twfe", treatment=treatment)
    tm = treatment_timing(df, treatment, unit, time)
    dcol = tcol
    if treatment isa FirstTreated
        dcol = _did_fresh_name(df, "treated")
        df[!, dcol] = Float64.(tm.row_treated)
    end
    any(tm.row_treated) || throw(ArgumentError(
        "did_twfe: no treated observations (treatment is zero everywhere)"))
    all(tm.row_treated) && throw(ArgumentError(
        "did_twfe: every observation is treated; there is no comparison"))
    f = make_formula(outcome, vcat(dcol, covariates); fe=[unit, time])
    vc = _did_vcov_estimator(cluster, vcov)
    m = weights === nothing ? reg(df, f, vc) : reg(df, f, vc; weights=weights)
    j = try
        coef_index(m, dcol)
    catch err
        err isa ErrorException || rethrow()
        throw(ArgumentError("did_twfe: the treatment coefficient is not identified: " *
                            "treatment is collinear with the unit and time fixed " *
                            "effects (e.g. every treated unit is treated in every " *
                            "period, or all units switch at the same time with no " *
                            "comparison group)"))
    end
    b = coef(m)[j]
    V = StatsAPI.vcov(m)[j:j, j:j]
    staggered = _did_staggered(tm)
    diag_w = nothing
    note = ""
    if staggered || !tm.absorbing
        diag_w = try
            twfe_weights(df, outcome, dcol, unit, time; weights=weights)
        catch
            nothing
        end
        design = !tm.absorbing ? "non-absorbing treatment (units switch in and out)" :
                 "staggered adoption ($(length(_did_cohorts(tm))) treatment cohorts)"
        neg = diag_w === nothing ? "" :
              " $(diag_w.n_negative) of $(diag_w.n_treated_cells) treated unit-period " *
              "cells receive negative weight (sum of negative weights " *
              "$(round(diag_w.sum_negative; digits=3)))."
        note = "TWFE with $design: the coefficient is a weighted average of " *
               "heterogeneous effects with possibly negative weights.$neg"
        warn_heterogeneity && @warn "did_twfe: $note Consider " *
            "did_callaway_santanna, did_sun_abraham or did_imputation; see " *
            "bacon_decomposition and twfe_weights for diagnostics."
    end
    ncl = _did_nclusters(m)
    method = "Two-way fixed effects"
    return DiDEstimate([b], Matrix(V), [string(dcol)], nobs(m), dof_residual(m), ncl,
                       _did_n_ever(tm), _did_n_never(tm), length(tm.periods), method,
                       staggered || !tm.absorbing ?
                       "TWFE coefficient: weighted average of cell effects (see " *
                       "details.twfe_weights)" : "ATT",
                       (model=m, timing=tm, staggered=staggered, absorbing=tm.absorbing,
                        twfe_weights=diag_w, note=note))
end

did_twfe(panel::TreatmentPanel; kwargs...) =
    did_twfe(panel.data, panel.outcome, panel.treatment, panel.unit_id, panel.time;
             covariates=panel.covariates, kwargs...)

"""
    parallel_trends_test(data, outcome, treatment, unit, time; kwargs...)
        -> DiagnosticTest
    parallel_trends_test(panel::TreatmentPanel; kwargs...) -> DiagnosticTest

Joint test that all pre-treatment event-time effects are zero, computed from a fully
dynamic event study.

The function estimates an event study with [`event_study`](@ref), passing every
keyword through, and applies [`pre_trend_test`](@ref) to it: a Wald test of
``H_0: \\beta_e = 0`` for all estimated ``e < 0``, using the full covariance of the
placebo coefficients (an ``F(q, G - 1)`` reference under clustering with ``G``
clusters for regression-based estimators, ``\\chi^2(q)`` otherwise). Despite its
name, this is a test of **parallel pre-trends** (and of no anticipation), not of
parallel trends: the identifying assumption concerns untreated potential outcomes
after treatment and cannot be tested. A non-rejection is not evidence that the
assumption holds. Pre-trend tests often have low power against violations large
enough to matter, and conditioning the analysis on passing them biases the
post-treatment estimates and distorts coverage (Roth, 2022). Report the
pre-treatment coefficients themselves with simultaneous bands, and complement or
replace the test by a sensitivity analysis ([`honest_did`](@ref); Rambachan and
Roth, 2023) or an equivalence-type approach that asks whether the data can rule out
economically relevant pre-trends (Bilinski and Hatfield, 2026).

The choice of estimator matters. With several treatment cohorts the default
`estimator = :auto` uses [`did_sun_abraham`](@ref), whose placebo coefficients are
not contaminated by treatment-effect heterogeneity, unlike those of the TWFE
dynamic specification (Sun and Abraham, 2021). The placebo coefficients of
`estimator = :callaway_santanna` with the default varying base period are short
differences between consecutive periods rather than differences from a common
reference period (Roth, 2026).

# Arguments
- `data`: a long-format panel.
- `outcome::Symbol`: outcome column.
- `treatment`: an absorbing 0/1 indicator or [`FirstTreated`](@ref)`(column)`.
- `unit::Symbol`, `time::Symbol`: unit and time columns.

# Keywords
All keywords of [`event_study`](@ref) are accepted and passed through, e.g.
`estimator`, `max_pre`, `max_post`, `endpoints`, `covariates`, `cluster`,
`anticipation`. The `TreatmentPanel` method passes the panel's covariates.

# Returns
- `DiagnosticTest`: the Wald statistic, p-value and degrees of freedom;
  `details.event_study` holds the fitted [`EventStudyEstimate`](@ref), and
  `details.coefficients` and `details.periods` the tested coefficients.

# Examples
```julia
using DrSnow, CSV, DataFrames
file = joinpath(pkgdir(DrSnow), "test", "validation", "did", "mpdta.csv")
mpdta = CSV.read(file, DataFrame)
t = parallel_trends_test(mpdta, :lemp, :d, :countyreal, :year)
t.pvalue
t.details.event_study        # the underlying Sun–Abraham event study
```

# References
- Roth, J. (2022). Pretest with caution: Event-study estimates after testing for
  parallel trends. *American Economic Review: Insights*, 4(3), 305–322.
- Rambachan, A., & Roth, J. (2023). A more credible approach to parallel trends.
  *Review of Economic Studies*, 90(5), 2555–2591.
- Bilinski, A., & Hatfield, L. A. (2026). Nothing to see here? A non-inferiority
  approach to parallel trends. *Statistics in Medicine*, 45(3–5), e70296.
- Sun, L., & Abraham, S. (2021). Estimating dynamic treatment effects in event
  studies with heterogeneous treatment effects. *Journal of Econometrics*, 225(2),
  175–199.
- Roth, J. (2026). Interpreting event-studies from recent difference-in-differences
  methods. *The Japanese Economic Review*, 77(2), 275–288.
"""
function parallel_trends_test(data, outcome::Symbol, treatment, unit::Symbol,
                              time::Symbol; kwargs...)
    es = event_study(data, outcome, treatment, unit, time; kwargs...)
    t = pre_trend_test(es)
    return DiagnosticTest(t.name, t.null, t.statistic, t.pvalue; dof=t.dof,
                          method=t.method * "; " * es.method, note=t.note,
                          details=merge(t.details, (event_study=es,)))
end

parallel_trends_test(panel::TreatmentPanel; kwargs...) =
    parallel_trends_test(panel.data, panel.outcome, panel.treatment, panel.unit_id,
                         panel.time; covariates=panel.covariates, kwargs...)
