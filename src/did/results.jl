# Result types shared by the DiD estimators.
#
# - `DiDEstimate`: a scalar ATT-type estimate (TWFE, Sant'Anna–Zhao, imputation, …).
# - `EventStudyEstimate`: coefficients indexed by relative period (event time), with
#   the full covariance matrix. Every event-study estimator returns this type so that
#   downstream tools (pre-trend tests, sensitivity analysis, plotting) need a single
#   interface: `relative_periods(es)`, `coef(es)`, `vcov(es)`, `es.reference`.
# - `AggregatedATT`: Callaway–Sant'Anna simple / group / calendar aggregations.

"""
    DiDEstimate <: CausalEstimate

Scalar difference-in-differences estimate: an average treatment effect on the treated
(ATT), a weighted average of cell or event-time effects, or a regression coefficient
whose causal interpretation depends on the estimator that produced it.

The object is returned by the two-by-two and summary estimators of the area
([`did_twfe`](@ref), [`did_drdid`](@ref), [`did_imputation`](@ref) without horizons)
and by summaries of richer results ([`event_study_average`](@ref), the `overall`
entries of aggregated event studies). The `estimand` field states in words what the
coefficient estimates; for the two-way fixed effects coefficient under staggered
adoption it deliberately does *not* say "ATT", because that coefficient is a
weighted average of cell effects with possibly negative weights (Goodman-Bacon,
2021; de Chaisemartin and D'Haultfœuille, 2020). Inference uses a t reference with
`dof_residual(r)` degrees of freedom when that is finite (typically ``G - 1`` with
``G`` clusters) and the standard normal otherwise (influence-function estimators).

# Fields
- `coef::Vector{Float64}`: the estimate (a vector of length one).
- `vcov::Matrix{Float64}`: its ``1 \\times 1`` variance estimate.
- `coefnames::Vector{String}`: the coefficient label.
- `nobs::Int`: number of observations (rows) used.
- `dof::Float64`: degrees of freedom of the t reference distribution (`G - 1` under
  clustering; `Inf` for influence-function or normal inference).
- `n_clusters::Int`: number of clusters (0 when the variance is not clustered).
- `n_treated::Int`: number of ever-treated units (treated observations for repeated
  cross-sections); `-1` when not applicable.
- `n_control::Int`: number of never-treated units (comparison observations for
  repeated cross-sections); `-1` when not applicable.
- `n_periods::Int`: number of time periods.
- `method::String`: the estimator.
- `estimand::String`: the target parameter in words.
- `details::NamedTuple`: estimator-specific output, such as the fitted model, the
  influence function, the [`TreatmentTiming`](@ref) and negative-weight diagnostics.

# Accessors
`coef`, `vcov`, `stderror`, `confint(r; level=0.95)`, `coeftable`, `coefnames`,
`nobs`, `dof_residual`, [`estimate`](@ref), [`estimand`](@ref) and
[`method_name`](@ref).

# References
- Goodman-Bacon, A. (2021). Difference-in-differences with variation in treatment
  timing. *Journal of Econometrics*, 225(2), 254–277.
- de Chaisemartin, C., & D'Haultfœuille, X. (2020). Two-way fixed effects estimators
  with heterogeneous treatment effects. *American Economic Review*, 110(9),
  2964–2996.
"""
struct DiDEstimate <: CausalEstimate
    coef::Vector{Float64}
    vcov::Matrix{Float64}
    coefnames::Vector{String}
    nobs::Int
    dof::Float64
    n_clusters::Int
    n_treated::Int
    n_control::Int
    n_periods::Int
    method::String
    estimand::String
    details::NamedTuple
end

StatsAPI.coef(r::DiDEstimate) = r.coef
StatsAPI.vcov(r::DiDEstimate) = r.vcov
StatsAPI.coefnames(r::DiDEstimate) = r.coefnames
StatsAPI.nobs(r::DiDEstimate) = r.nobs
StatsAPI.dof_residual(r::DiDEstimate) = r.dof
estimand(r::DiDEstimate) = r.estimand
method_name(r::DiDEstimate) = r.method

function show_details(io::IO, r::DiDEstimate)
    println(io)
    r.n_treated >= 0 && print(io, "Treated units: ", r.n_treated)
    r.n_control >= 0 && print(io, ", never-treated/comparison units: ", r.n_control)
    r.n_periods > 0 && print(io, ", periods: ", r.n_periods)
    println(io)
    if r.n_clusters > 0
        df_note = isfinite(r.dof) ? " (t reference with $(Int(r.dof)) df)" : ""
        println(io, "Clusters: ", r.n_clusters, df_note)
    end
    note = get(r.details, :note, "")
    isempty(note) || println(io, "Note: ", note)
    return nothing
end

"""
    EventStudyEstimate <: CausalEstimate

Dynamic (event-time) treatment effects: one coefficient per relative period
``e = t - G_i`` together with their full covariance matrix.

Every event-study estimator of the package returns this type: the two-way fixed
effects dynamic specification of [`event_study`](@ref), the interaction-weighted
estimator [`did_sun_abraham`](@ref), the imputation estimator
[`did_imputation`](@ref), the dynamic aggregations of
[`did_callaway_santanna`](@ref) and [`did_etwfe`](@ref) (via
[`aggregate_att`](@ref)), and [`did_multiplegt_dyn`](@ref). Coefficient ``k``
estimates an effect at event time `relative_periods(es)[k]`, measured in periods of
the ordered time index, so that gaps in the time variable or `Date` values do not
create gaps in event time. Pre-treatment coefficients (``e < 0``) are placebo
estimates whose population value is zero under parallel trends and no anticipation;
how they are constructed (relative to a common reference period, or as short
differences) depends on the estimator and matters for their interpretation (Roth,
2026). Holding the full covariance matrix lets downstream tools, namely
[`pre_trend_test`](@ref), [`event_study_average`](@ref), simultaneous confidence
bands and the sensitivity analysis [`honest_did`](@ref), use the joint sampling
distribution of all coefficients.

# Fields
- `rel_periods::Vector{Int}`: relative periods of the coefficients.
- `coef::Vector{Float64}`, `vcov::Matrix{Float64}`: estimates and their full
  covariance matrix.
- `reference::Vector{Int}`: relative periods normalized to zero (e.g. `[-1]`); empty
  for estimators without a normalization (imputation, varying-base-period
  Callaway–Sant'Anna).
- `nobs::Int`, `dof::Float64`, `n_clusters::Int`: as in [`DiDEstimate`](@ref).
- `method::String`, `estimand::String`: the estimator and the target in words.
- `supt_draws::Vector{Float64}`: bootstrap draws of the sup-t statistic for
  simultaneous (uniform) confidence bands; empty when none were computed, in which
  case `confint(es; uniform=true)` simulates them from the estimated covariance.
- `details::NamedTuple`: estimator-specific output. It always contains `n_treated`,
  `n_control`, `n_periods` and `binned`, a pair of `Bool`s recording whether the
  first and last coefficients pool all relative periods beyond them.

# Accessors
`coef`, `vcov`, `stderror`, `coefnames` (labels `"e=k"`, or `"e<=k"`/`"e>=k"` for
binned endpoints), `coeftable`, `nobs`, `dof_residual`,
[`relative_periods`](@ref), `confint(es; level, uniform)`, [`estimate`](@ref),
[`pre_trend_test`](@ref) and [`event_study_average`](@ref).

# References
- Roth, J. (2026). Interpreting event-studies from recent difference-in-differences
  methods. *The Japanese Economic Review*, 77(2), 275–288.
- Sun, L., & Abraham, S. (2021). Estimating dynamic treatment effects in event
  studies with heterogeneous treatment effects. *Journal of Econometrics*, 225(2),
  175–199.
"""
struct EventStudyEstimate <: CausalEstimate
    rel_periods::Vector{Int}
    coef::Vector{Float64}
    vcov::Matrix{Float64}
    reference::Vector{Int}
    nobs::Int
    dof::Float64
    n_clusters::Int
    method::String
    estimand::String
    supt_draws::Vector{Float64}
    details::NamedTuple
end

StatsAPI.coef(r::EventStudyEstimate) = r.coef
StatsAPI.vcov(r::EventStudyEstimate) = r.vcov
StatsAPI.nobs(r::EventStudyEstimate) = r.nobs
StatsAPI.dof_residual(r::EventStudyEstimate) = r.dof
estimand(r::EventStudyEstimate) = r.estimand
method_name(r::EventStudyEstimate) = r.method

function StatsAPI.coefnames(r::EventStudyEstimate)
    names = ["e=$(k)" for k in r.rel_periods]
    binned = get(r.details, :binned, (false, false))
    if !isempty(names)
        binned[1] && (names[1] = "e<=$(r.rel_periods[1])")
        binned[2] && (names[end] = "e>=$(r.rel_periods[end])")
    end
    return names
end

"""
    relative_periods(es::EventStudyEstimate) -> Vector{Int}

Relative periods (event times) of the coefficients of an event-study result, in the
order of `coef(es)`.

Event time is ``e = t - G_i``, the number of periods since a unit's first treated
period ``G_i``, counted on the ordered index of observed periods. Negative values
are pre-treatment (placebo) periods, `0` is the first treated period, and the
reference period(s) normalized to zero are not included (see `es.reference`).
Binned endpoint coefficients are labelled by their endpoint.

# Arguments
- `es::EventStudyEstimate`: an event-study result.

# Returns
- `Vector{Int}`: one relative period per coefficient.

# Examples
```julia
using DrSnow, CSV, DataFrames
file = joinpath(pkgdir(DrSnow), "test", "validation", "did", "mpdta.csv")
mpdta = CSV.read(file, DataFrame)
es = did_sun_abraham(mpdta, :lemp, FirstTreated(:first_treat), :countyreal, :year)
relative_periods(es)          # [-4, -3, -2, 0, 1, 2, 3]
```
"""
relative_periods(r::EventStudyEstimate) = r.rel_periods

"""
    estimate(es::EventStudyEstimate) -> Float64

Equal-weighted average of the post-treatment (``e \\ge 0``) event-time coefficients.

This is a convenient scalar summary, not a canonical estimand: with unbalanced
event-time composition, different cohorts contribute to different horizons, so the
average mixes horizon and composition effects. Use
[`event_study_average`](@ref) for its standard error and for other weightings, and
[`aggregate_att`](@ref) for the Callaway–Sant'Anna summary parameters.

# Arguments
- `es::EventStudyEstimate`: an event-study result with at least one ``e \\ge 0``
  coefficient.

# Returns
- `Float64`: the average; equal to `coef(event_study_average(es))[1]`.
"""
estimate(r::EventStudyEstimate) = estimate(event_study_average(r))

function show_details(io::IO, r::EventStudyEstimate)
    println(io)
    isempty(r.reference) ||
        println(io, "Reference (normalized to 0) relative period(s): ",
                join(r.reference, ", "))
    binned = get(r.details, :binned, (false, false))
    if any(binned)
        println(io, "Endpoint coefficients pool all relative periods beyond them (binned).")
    end
    r.n_clusters > 0 && println(io, "Clusters: ", r.n_clusters)
    note = get(r.details, :note, "")
    isempty(note) || println(io, "Note: ", note)
    return nothing
end

"""
    AggregatedATT <: CausalEstimate

Aggregation of group-time average treatment effects ``ATT(g,t)`` into a summary
parameter and its components, returned by [`aggregate_att`](@ref) for the
`:simple`, `:group` and `:calendar` aggregations of
[`did_callaway_santanna`](@ref) and [`did_etwfe`](@ref).

The first coefficient is the overall summary parameter; the remaining ones are the
components: cohort-specific averages ``\\theta(g)`` for `:group`, calendar-period
averages ``\\theta(t)`` for `:calendar`, none for `:simple` (Callaway and Sant'Anna,
2021, Section 4). For Callaway–Sant'Anna the covariance comes from influence
functions that include the estimation of the aggregation weights (cohort shares);
for the extended TWFE estimator it comes from the delta method applied to the
regression covariance, with cell sizes treated as fixed.

# Fields
- `kind::Symbol`: `:simple`, `:group` or `:calendar`.
- `labels::Vector`: cohort or calendar-period labels of the components, in the units
  of the time variable.
- `coef::Vector{Float64}`, `vcov::Matrix{Float64}`: `[overall; components]` and their
  full covariance.
- `nobs::Int`, `n_clusters::Int`: observations and clusters.
- `supt_draws::Vector{Float64}`: bootstrap sup-t draws for the components (uniform
  bands); empty when not computed.
- `influence::Matrix{Float64}`: influence functions (units × coefficients), scaled so
  that `estimate - truth ≈ mean(influence[:, k])`; empty for regression-based
  aggregations.
- `details::NamedTuple`: estimator-specific output (degrees of freedom, estimand and
  method labels for regression-based aggregations).

# Accessors
`coef`, `vcov`, `stderror`, `coefnames` (`"ATT"`, then `"g=…"` or `"t=…"`),
`coeftable`, `confint(r; level, uniform)`, `nobs`, `dof_residual`,
[`estimate`](@ref).

# References
- Callaway, B., & Sant'Anna, P. H. C. (2021). Difference-in-differences with multiple
  time periods. *Journal of Econometrics*, 225(2), 200–230.
"""
struct AggregatedATT <: CausalEstimate
    kind::Symbol
    labels::Vector
    coef::Vector{Float64}
    vcov::Matrix{Float64}
    nobs::Int
    n_clusters::Int
    supt_draws::Vector{Float64}
    influence::Matrix{Float64}
    details::NamedTuple
end

StatsAPI.coef(r::AggregatedATT) = r.coef
StatsAPI.vcov(r::AggregatedATT) = r.vcov
StatsAPI.nobs(r::AggregatedATT) = r.nobs
StatsAPI.dof_residual(r::AggregatedATT) = get(r.details, :dof, Inf)
function estimand(r::AggregatedATT)
    haskey(r.details, :estimand) && return r.details.estimand
    r.kind === :simple && return "ATT (simple weighted average)"
    r.kind === :group && return "ATT by cohort; overall = cohort-size weighted"
    return "ATT by calendar period; overall = average over periods"
end
method_name(r::AggregatedATT) =
    get(r.details, :method, "Callaway–Sant'Anna aggregation ($(r.kind))")

function StatsAPI.coefnames(r::AggregatedATT)
    prefix = r.kind === :group ? "g=" : "t="
    return vcat(["ATT"], [prefix * string(l) for l in r.labels])
end

function show_details(io::IO, r::AggregatedATT)
    println(io)
    r.n_clusters > 0 && println(io, "Clusters: ", r.n_clusters)
    return nothing
end

# ---------------------------------------------------------------------------
# Confidence intervals with optional simultaneous (sup-t) bands
# ---------------------------------------------------------------------------

"""
    confint(es::EventStudyEstimate; level=0.95, uniform=false,
            rng=Random.default_rng(), ndraws=10_000) -> Matrix{Float64}

Pointwise confidence intervals, or simultaneous (sup-t) confidence bands, for the
coefficients of an event study.

Pointwise intervals ``\\hat\\beta_e \\pm c_{1-\\alpha/2}\\, \\widehat{se}_e`` cover
each coefficient with probability `level` individually; they do not cover the whole
event-time path jointly, and a path that stays inside a pointwise band can still be
rejected. With `uniform = true` the intervals form a simultaneous band that covers
all coefficients jointly with asymptotic probability `level`: the half-width is
``c^{*}\\, \\widehat{se}_e``, where ``c^{*}`` is the `level` quantile of
``\\max_e |\\hat\\beta_e - \\beta_e| / \\widehat{se}_e``. The quantile is taken from
the multiplier-bootstrap draws stored in `es` when available (Callaway and
Sant'Anna, 2021) and is otherwise simulated from `ndraws` draws of a Gaussian vector
(Student-t when `dof_residual(es)` is finite) with the estimated correlation matrix,
the plug-in sup-t band of Montiel Olea and Plagborg-Møller (2019). Coefficients with
zero variance are excluded from the maximum. Simultaneous bands are the appropriate
object for statements about the shape of the whole path, e.g. that all
pre-treatment coefficients are compatible with zero (Freyaldenhoven, Hansen and
Shapiro, 2019).

# Arguments
- `es::EventStudyEstimate`: an event-study result.

# Keywords
- `level::Real = 0.95`: confidence level.
- `uniform::Bool = false`: return simultaneous sup-t bands instead of pointwise
  intervals.
- `rng::AbstractRNG = Random.default_rng()`: random-number generator for the
  simulated critical value (only used when `uniform = true` and no bootstrap draws
  are stored).
- `ndraws::Integer = 10_000`: number of simulated draws for that critical value.

# Returns
- `Matrix{Float64}`: lower and upper bounds in columns 1 and 2, one row per
  coefficient in the order of `relative_periods(es)`.

# Examples
```julia
using DrSnow, CSV, DataFrames, StableRNGs
file = joinpath(pkgdir(DrSnow), "test", "validation", "did", "mpdta.csv")
mpdta = CSV.read(file, DataFrame)
es = did_sun_abraham(mpdta, :lemp, FirstTreated(:first_treat), :countyreal, :year)
confint(es; level=0.90)
confint(es; uniform=true, rng=StableRNG(1))
```

# References
- Montiel Olea, J. L., & Plagborg-Møller, M. (2019). Simultaneous confidence bands:
  Theory, implementation, and an application to SVARs. *Journal of Applied
  Econometrics*, 34(1), 1–17.
- Freyaldenhoven, S., Hansen, C., & Shapiro, J. M. (2019). Pre-event trends in the
  panel event-study design. *American Economic Review*, 109(9), 3307–3338.
- Callaway, B., & Sant'Anna, P. H. C. (2021). Difference-in-differences with multiple
  time periods. *Journal of Econometrics*, 225(2), 200–230.
"""
function StatsAPI.confint(r::EventStudyEstimate; level::Real=0.95, uniform::Bool=false,
                          rng::AbstractRNG=Random.default_rng(), ndraws::Integer=10_000)
    uniform || return _did_pointwise_ci(r, level)
    c = _uniform_critical_value(r.supt_draws, StatsAPI.vcov(r), r.dof, level, rng,
                                    ndraws, eachindex(r.coef))
    return _did_ci_with(StatsAPI.coef(r), StatsAPI.stderror(r), c)
end

"""
    confint(r::AggregatedATT; level=0.95, uniform=false,
            rng=Random.default_rng(), ndraws=10_000) -> Matrix{Float64}

Pointwise confidence intervals for an aggregated ATT and its components, or a
simultaneous band for the components.

With `uniform = true` the component intervals (all rows but the first, overall
coefficient) form a sup-t band that covers every cohort or calendar-period effect
jointly with probability `level`; the critical value is computed as for
[`confint(::EventStudyEstimate)`](@ref) (stored multiplier-bootstrap draws, else a
Gaussian simulation from the estimated covariance). The overall coefficient keeps
its pointwise interval.

# Arguments
- `r::AggregatedATT`: an aggregation from [`aggregate_att`](@ref).

# Keywords
- `level::Real = 0.95`: confidence level.
- `uniform::Bool = false`: simultaneous band for the components.
- `rng::AbstractRNG = Random.default_rng()`, `ndraws::Integer = 10_000`: generator
  and number of draws for a simulated critical value.

# Returns
- `Matrix{Float64}`: lower and upper bounds in columns 1 and 2 (row 1: overall).

# References
- Montiel Olea, J. L., & Plagborg-Møller, M. (2019). Simultaneous confidence bands:
  Theory, implementation, and an application to SVARs. *Journal of Applied
  Econometrics*, 34(1), 1–17.
"""
function StatsAPI.confint(r::AggregatedATT; level::Real=0.95, uniform::Bool=false,
                          rng::AbstractRNG=Random.default_rng(), ndraws::Integer=10_000)
    ci = _did_pointwise_ci(r, level)
    (uniform && length(r.coef) > 1) || return ci
    idx = 2:length(r.coef)
    c = _uniform_critical_value(r.supt_draws, StatsAPI.vcov(r),
                                StatsAPI.dof_residual(r), level, rng, ndraws, idx)
    ci[idx, :] = _did_ci_with(r.coef[idx], StatsAPI.stderror(r)[idx], c)
    return ci
end

function _did_pointwise_ci(r::CausalEstimate, level)
    c = critical_value(level, StatsAPI.dof_residual(r))
    return _did_ci_with(StatsAPI.coef(r), StatsAPI.stderror(r), c)
end

_did_ci_with(b, s, c) = hcat(b .- c .* s, b .+ c .* s)


# ---------------------------------------------------------------------------
# Averages of event-time effects and pre-trend tests
# ---------------------------------------------------------------------------

"""
    event_study_average(es::EventStudyEstimate; periods=:post, weights=nothing)
        -> DiDEstimate

Weighted average of event-time coefficients with its delta-method standard error.

The estimand is ``\\sum_{e \\in S} w_e \\beta_e`` for a set ``S`` of relative periods
and weights summing to one; the variance is ``w' \\hat V_{SS} w`` with the full
covariance of the selected coefficients, so correlation between horizons is taken
into account, and the weights are treated as fixed. Averages over post-treatment
horizons summarize the effect over an exposure window; averages over pre-treatment
horizons summarize placebo estimates. Because the composition of cohorts usually
differs across event times, an equal-weighted average of ``\\beta_e`` is not in
general the ATT over treated cells; for Callaway–Sant'Anna use the overall
parameters of [`aggregate_att`](@ref), whose weights are estimated and whose
standard errors account for that. With a finite `dof_residual(es)` the result keeps
the t reference of the event study.

# Arguments
- `es::EventStudyEstimate`: an event-study result.

# Keywords
- `periods = :post`: `:post` (all ``e \\ge 0``), `:pre` (all ``e < 0``), or a
  collection of estimated relative periods.
- `weights = nothing`: `nothing` for equal weights, or one weight per selected
  period (normalized to sum to one).

# Returns
- `DiDEstimate`: the average; `details` holds the event study, the normalized weights
  and the periods averaged.

# Examples
```julia
using DrSnow, CSV, DataFrames
file = joinpath(pkgdir(DrSnow), "test", "validation", "did", "mpdta.csv")
mpdta = CSV.read(file, DataFrame)
es = did_sun_abraham(mpdta, :lemp, FirstTreated(:first_treat), :countyreal, :year)
avg = event_study_average(es; periods=0:3)
confint(avg)
event_study_average(es; periods=:pre)       # average placebo coefficient
```
"""
function event_study_average(r::EventStudyEstimate; periods=:post, weights=nothing)
    rp = r.rel_periods
    idx = if periods === :post
        findall(>=(0), rp)
    elseif periods === :pre
        findall(<(0), rp)
    else
        _did_period_positions(rp, periods)
    end
    isempty(idx) && throw(ArgumentError("no event-time coefficients selected"))
    w = weights === nothing ? fill(1 / length(idx), length(idx)) :
        (length(weights) == length(idx) ? float.(collect(weights)) ./ sum(weights) :
         throw(DimensionMismatch("need one weight per selected period")))
    b = dot(w, r.coef[idx])
    v = dot(w, r.vcov[idx, idx] * w)
    lbl = "average of e ∈ {" * join(rp[idx], ", ") * "}"
    return DiDEstimate([b], fill(v, 1, 1), [lbl], r.nobs, r.dof, r.n_clusters,
                       get(r.details, :n_treated, -1), get(r.details, :n_control, -1),
                       get(r.details, :n_periods, 0), r.method * " (event-time average)",
                       "average of event-time effects", (event_study=r, weights=w,
                                                          periods=rp[idx]))
end

function _did_period_positions(rp, periods)
    want = collect(periods)
    missing_p = setdiff(want, rp)
    isempty(missing_p) || throw(ArgumentError(
        "relative period(s) $(join(missing_p, ", ")) not estimated in this event study"))
    return [findfirst(==(p), rp) for p in want]
end

const _DID_PRETREND_NOTE =
    "Pre-trend tests have low power against many relevant violations of parallel " *
    "trends, and conditioning on passing them can bias the post-treatment estimates " *
    "(Roth 2022). A non-rejection does not show that parallel trends holds; report " *
    "the pre-period estimates and consider sensitivity analysis (Rambachan & Roth 2023)."

"""
    pre_trend_test(es::EventStudyEstimate; periods=nothing) -> DiagnosticTest

Joint Wald test that the pre-treatment event-time coefficients of an event study are
all zero.

Under parallel trends and no anticipation, the population values of the
pre-treatment (placebo) coefficients ``\\beta_e``, ``e < 0``, are zero. The test
statistic is ``W = \\hat\\beta_S' \\hat V_{SS}^{-1} \\hat\\beta_S`` for the tested set
``S`` with ``q = |S|`` coefficients, using their full covariance matrix. With a
finite `dof_residual(es)` (e.g. ``G - 1`` under clustering with ``G`` clusters) the
reference distribution is ``F(q, G - 1)`` for ``W/q``; otherwise it is
``\\chi^2(q)``.

A rejection is evidence against parallel pre-trends or of anticipation effects. A
non-rejection is **not** evidence that parallel trends holds: parallel trends is an
assumption about post-treatment counterfactuals, pre-trend tests often have low
power against economically relevant violations, and reporting estimates only
conditional on passing such a test distorts their sampling distribution (Roth,
2022; Freyaldenhoven, Hansen and Shapiro, 2019; Kahn-Lang and Lang, 2020). Report
the pre-period estimates with simultaneous bands, and assess how conclusions change
under bounded violations with [`honest_did`](@ref) (Rambachan and Roth, 2023). The
meaning of the coefficients tested depends on the estimator: short differences
between consecutive periods for the varying base period of Callaway–Sant'Anna,
differences from a common reference period for TWFE and Sun–Abraham, leads
estimated on untreated observations for the imputation estimator (Roth, 2026).

# Arguments
- `es::EventStudyEstimate`: an event-study result with pre-treatment coefficients.

# Keywords
- `periods = nothing`: relative periods to test; by default all estimated ``e < 0``.

# Returns
- `DiagnosticTest`: statistic, p-value and degrees of freedom; `details` holds the
  tested coefficients, their periods, covariance and the `WaldTest`.

# Examples
```julia
using DrSnow, CSV, DataFrames
file = joinpath(pkgdir(DrSnow), "test", "validation", "did", "mpdta.csv")
mpdta = CSV.read(file, DataFrame)
es = did_sun_abraham(mpdta, :lemp, FirstTreated(:first_treat), :countyreal, :year)
t = pre_trend_test(es)
t.pvalue
pre_trend_test(es; periods=[-3, -2])
```

# References
- Roth, J. (2022). Pretest with caution: Event-study estimates after testing for
  parallel trends. *American Economic Review: Insights*, 4(3), 305–322.
- Freyaldenhoven, S., Hansen, C., & Shapiro, J. M. (2019). Pre-event trends in the
  panel event-study design. *American Economic Review*, 109(9), 3307–3338.
- Kahn-Lang, A., & Lang, K. (2020). The promise and pitfalls of
  differences-in-differences: Reflections on 16 and Pregnant and other applications.
  *Journal of Business & Economic Statistics*, 38(3), 613–620.
- Rambachan, A., & Roth, J. (2023). A more credible approach to parallel trends.
  *Review of Economic Studies*, 90(5), 2555–2591.
- Roth, J. (2026). Interpreting event-studies from recent difference-in-differences
  methods. *The Japanese Economic Review*, 77(2), 275–288.
"""
function pre_trend_test(r::EventStudyEstimate; periods=nothing)
    rp = r.rel_periods
    idx = periods === nothing ? findall(<(0), rp) : _did_period_positions(rp, periods)
    isempty(idx) && throw(ArgumentError(
        "the event study has no pre-treatment coefficients to test"))
    return _did_joint_zero_test(r.coef[idx], r.vcov[idx, idx], r.dof, rp[idx],
                                "Joint pre-trend test",
                                "all pre-treatment event-time effects are zero")
end

function _did_joint_zero_test(b, V, dof, labels, name, null)
    w = wald_test(b, V; dof=dof)
    dofs = isfinite(dof) ? (w.dof1, dof) : (w.dof1,)
    method = isfinite(dof) ? "Wald test (F, full covariance)" :
             "Wald test (χ², full covariance)"
    return DiagnosticTest(name, null, w.statistic, w.pvalue; dof=dofs, method=method,
                          note=_DID_PRETREND_NOTE,
                          details=(coefficients=b, periods=labels, vcov=V, wald=w))
end
