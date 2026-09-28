# Viz: plotting API.
#
# The plotting functions exported here are generic stubs. Their methods live in the
# Makie package extension (`ext/DrSnowMakieExt/`), which Julia loads automatically
# once a Makie backend is loaded (`using CairoMakie` for files, `using GLMakie` or
# `using WGLMakie` for interactive use). Without a backend each stub throws an error
# explaining how to enable plotting.
#
# The viz area has two layers:
#   1. `src/viz/data.jl` (no Makie dependency): `_viz_*_data` functions turning result
#      objects into plain DataFrames / NamedTuples (event-time coefficients with
#      pointwise and uniform intervals, synthetic-control trajectories, ...). They are
#      unit-tested without a backend and are the only place that reads result fields.
#   2. `ext/DrSnowMakieExt/` draws those tables with Makie.
#
# Adding plotting support for a new result type
# ---------------------------------------------
# - Results that are (or return) existing types need nothing: every `CausalEstimate`
#   works with `plot_coefficients` (via `tidy`), and anything returning an
#   `EventStudyEstimate` works with `plot_event_study`.
# - A new type with the *same shape* as an existing plot (e.g. event-time
#   coefficients, a randomization distribution): add a method of the matching
#   `_viz_*_data` hook in `src/viz/data.jl` (e.g.
#   `_viz_event_study_data(r::MyType; level, uniform, rng)` returning the documented
#   columns). The Makie code then plots it unchanged. Add a test in `test/viz/`.
# - A genuinely new kind of plot (e.g. a sensitivity curve): add
#     (a) a stub pair `function plot_foo end` / `function plot_foo! end` below, with a
#         docstring and a `_viz_no_backend` fallback, and export both here;
#     (b) a `_viz_foo_data` extractor in `src/viz/data.jl`;
#     (c) the Makie methods in a new file `ext/DrSnowMakieExt/foo.jl`, included from
#         `ext/DrSnowMakieExt/DrSnowMakieExt.jl`, using the shared helpers there
#         (`_mk_figure`, `_mk_axis`, `_mk_color`, `_mk_ci!`, ...);
#     (d) tests in `test/viz/` and an entry in the `@docs` block of
#         `docs/src/results.md`.
# The viz area is included last, so `data.jl` may reference every area's types.

include("data.jl")
include("data_diagnostics.jl")

export plot_event_study, plot_event_study!
export plot_coefficients, plot_coefficients!
export plot_rd, plot_rd!
export plot_synth, plot_synth!
export plot_randomization_distribution, plot_randomization_distribution!
export plot_bacon, plot_bacon!
export plot_gates, plot_gates!
export plot_cate, plot_cate!
export plot_spillover_rings, plot_spillover_rings!
export plot_confidence_set, plot_confidence_set!
export plot_trends, plot_trends!, plot_balance, plot_balance!
export plot_rd_placebos, plot_rd_placebos!, plot_rd_sensitivity, plot_rd_sensitivity!
export plot_rd_density, plot_rd_density!, plot_honest_did, plot_honest_did!
export plot_judge_first_stage, plot_judge_first_stage!, plot_rotemberg, plot_rotemberg!
export plot_mte, plot_mte!, plot_synth_in_time, plot_synth_in_time!, drsnow_theme
export plot_variable_importance, plot_variable_importance!
export plot_rate, plot_rate!
export plot_assignment_probabilities, plot_assignment_probabilities!
export plot_excursion_effect, plot_excursion_effect!
export plot_confidence_sequence, plot_confidence_sequence!
export plot_gs_boundaries, plot_gs_boundaries!
export plot_power_curve, plot_power_curve!, plot_blocks, plot_blocks!

const _VIZ_BACKEND_HINT =
    "requires a Makie backend. Load one to enable DrSnow's plotting extension: " *
    "`using CairoMakie` (static PNG/SVG/PDF output) or `using GLMakie` (interactive)."

function _viz_no_backend(f, args)
    if Base.get_extension(@__MODULE__, :DrSnowMakieExt) === nothing
        throw(ErrorException("$(nameof(f)) " * _VIZ_BACKEND_HINT))
    end
    throw(MethodError(f, args))
end

# Common keyword arguments of the plotting functions, referenced by the docstrings.
const _VIZ_COMMON_KW = """
- `figure::NamedTuple = (;)`: keyword arguments for `Makie.Figure`, e.g.
  `figure = (size = (800, 500),)`. The default size depends on the plot.
- `axis::NamedTuple = (;)`: keyword arguments for `Makie.Axis` (e.g. `title`,
  `xlabel`, `limits`); they override the plot's own axis settings.
- `colors = nothing`: a vector of colours, one per series. The default is DrSnow's
  fixed-order categorical palette (see [`drsnow_theme`](@ref)), which supports at
  most eight series; pass `colors` explicitly for more. Series are always also
  distinguished by marker shape, line style or a label, never by colour alone.
"""

"""
    plot_event_study(es, more...; level=0.95, uniform=false, labels=nothing,
                     connect=false, figure=(;), axis=(;), colors=nothing,
                     rng=Random.default_rng()) -> Makie.Figure

Event-study plot: estimated dynamic treatment effects by period relative to treatment
onset, with pointwise confidence intervals and, optionally, simultaneous (sup-t)
confidence bands.

The plot displays the event-time coefficients ``\\hat\\theta_k`` of one or more
event-study estimates against the relative period ``k`` (periods since adoption, with
``k = 0`` the first treated period). Each coefficient is a contrast with the
*reference period(s)*, which are normalized to zero and drawn as hollow markers
without intervals; the path is therefore identified only up to that normalization,
and moving the reference period shifts every coefficient. A dashed vertical line
between ``k = -1`` and ``k = 0`` marks the onset of treatment and a horizontal line
marks zero. Binned endpoints (all periods at or beyond a window limit) are labelled
`≤k` / `≥k`.

Whiskers are pointwise `level` confidence intervals: each covers its own coefficient
with probability `level`, so with many horizons some intervals will exclude zero by
chance alone. Statements about the whole path, such as "all pre-period coefficients
are zero" or "the effect is positive at every horizon", require simultaneous bands.
With `uniform = true` the plot adds sup-t bands (Montiel Olea and Plagborg-Møller
2019) as wide translucent bars behind the whiskers; they cover all displayed
coefficients jointly with probability `level`. The bands use the stored bootstrap
draws when the estimator provides them (e.g. the multiplier bootstrap of
[`did_callaway_santanna`](@ref)) and are otherwise simulated from the estimated
covariance matrix, a Gaussian approximation; pass `rng` for reproducible bands.
Freyaldenhoven, Hansen, Pérez Pérez and Shapiro (2026) recommend this kind of
display: both types of interval, a visible normalization and a marked onset.

Pre-period coefficients close to zero are consistent with parallel trends but do not
establish them: parallel trends concern untreated potential outcomes in the
post-treatment periods, pre-trend tests often have low power, and conditioning an
analysis on passing such a test distorts subsequent inference (Roth 2022). Use
[`pre_trend_test`](@ref) for a joint test and [`honest_did`](@ref) /
[`plot_honest_did`](@ref) for inference that allows bounded violations (Rambachan
and Roth 2023). Under staggered adoption with heterogeneous effects the coefficients
of a two-way fixed-effects event study are contaminated by effects from other
periods (Sun and Abraham 2021); overlaying heterogeneity-robust estimators is a
direct visual check (Roth, Sant'Anna, Bilinski and Poe 2023).

Several results are overlaid with small horizontal offsets (purely cosmetic),
distinct colours and marker shapes, and a legend. Estimators construct pre-period
coefficients differently (for example, [`did_callaway_santanna`](@ref) reports short
differences under its default varying base period, whereas a TWFE event study
reports long differences relative to one reference period), so overlaid pre-period
paths need not be comparable even when the post-period paths are.

# Arguments
- `es`: an [`EventStudyEstimate`](@ref) (from [`event_study`](@ref),
  [`did_sun_abraham`](@ref), [`did_imputation`](@ref),
  [`aggregate_att`](@ref)`(cs, :dynamic)`, ...) or a [`SpilloverEventStudy`](@ref),
  which is drawn with one series per exposure group (see also
  [`plot_spillover_rings`](@ref)).
- `more...`: further results to overlay on the same axis.

# Keywords
- `level::Real = 0.95`: confidence level of the pointwise intervals and of the
  simultaneous bands.
- `uniform::Bool = false`: also draw simultaneous sup-t bands (not available for
  spillover event studies).
- `labels = nothing`: legend labels, one per result; the default is the
  [`method_name`](@ref) of each result.
- `connect::Bool = false`: join each series' point estimates (including the
  reference period) with a line. Lines suggest interpolation between periods; the
  default shows the coefficients as separate estimates.
- `rng::AbstractRNG = Random.default_rng()`: random-number generator for simulated
  simultaneous bands.
$(_VIZ_COMMON_KW)

# Returns
- `Makie.Figure`; the axis is `fig.content[1]`. The subtitle states the interval
  types and the meaning of hollow markers.

# Examples
```julia
using DrSnow, CairoMakie, DataFrames, StableRNGs
rng = StableRNG(1)
g = rand(rng, [0, 4, 6], 150)                        # adoption period (0 = never)
df = DataFrame(unit=repeat(1:150; inner=8), year=repeat(1:8, 150))
df.d = Int.((g[df.unit] .> 0) .& (df.year .>= g[df.unit]))
df.y = randn(rng, 150)[df.unit] .+ 0.2 .* df.year .+
       df.d .* (1 .+ 0.3 .* (df.year .- g[df.unit])) .+ randn(rng, nrow(df))
es_sa = did_sun_abraham(df, :y, :d, :unit, :year)
es_cs = aggregate_att(did_callaway_santanna(df, :y, :d, :unit, :year;
                                            rng=StableRNG(2)), :dynamic)
fig = plot_event_study(es_sa, es_cs; labels=["Sun–Abraham", "Callaway–Sant'Anna"],
                       uniform=true, rng=StableRNG(3))
```

# References
- Freyaldenhoven, S., Hansen, C., Pérez Pérez, J., & Shapiro, J. M. (2026).
  Visualization, identification, and estimation in the linear panel event-study
  design. In V. Chernozhukov, J. Hörner, E. La Ferrara, & I. Werning (Eds.),
  *Advances in Economics and Econometrics: Twelfth World Congress* (Vol. 2,
  pp. 225–268). Cambridge University Press.
- Montiel Olea, J. L., & Plagborg-Møller, M. (2019). Simultaneous confidence bands:
  Theory, implementation, and an application to SVARs. *Journal of Applied
  Econometrics*, 34(1), 1–17.
- Rambachan, A., & Roth, J. (2023). A more credible approach to parallel trends.
  *Review of Economic Studies*, 90(5), 2555–2591.
- Roth, J. (2022). Pretest with caution: Event-study estimates after testing for
  parallel trends. *American Economic Review: Insights*, 4(3), 305–322.
- Roth, J., Sant'Anna, P. H. C., Bilinski, A., & Poe, J. (2023). What's trending in
  difference-in-differences? A synthesis of the recent econometrics literature.
  *Journal of Econometrics*, 235(2), 2218–2244.
- Sun, L., & Abraham, S. (2021). Estimating dynamic treatment effects in event
  studies with heterogeneous treatment effects. *Journal of Econometrics*, 225(2),
  175–199.
"""
function plot_event_study end
"""
    plot_event_study!(ax, es, more...; level=0.95, uniform=false, labels=nothing,
                      connect=false, colors=nothing, rng=Random.default_rng()) -> ax

Draw an event-study plot into an existing Makie axis, for example one panel of a
multi-panel figure. This is the mutating counterpart of [`plot_event_study`](@ref),
which describes the display and how to read it; no legend or subtitle is added
(use `Makie.axislegend(ax)` for a legend).

# Arguments
- `ax::Makie.AbstractAxis`: the axis to draw into.
- `es`, `more...`: event-study results, as in [`plot_event_study`](@ref).
- Keywords: those of [`plot_event_study`](@ref) except `figure` and `axis`.

# Returns
- `ax`, the axis drawn into.
"""
function plot_event_study! end
plot_event_study(args...; kwargs...) = _viz_no_backend(plot_event_study, args)
plot_event_study!(args...; kwargs...) = _viz_no_backend(plot_event_study!, args)

"""
    plot_coefficients(r; level=0.95, terms=nothing, labels=nothing, figure=(;),
                      axis=(;), colors=nothing) -> Makie.Figure
    plot_coefficients(rs::AbstractVector; kwargs...) -> Makie.Figure

Coefficient ("forest" or dot-and-whisker) plot of point estimates with confidence
intervals for one or several [`CausalEstimate`](@ref)s.

Each row shows a point estimate with its `level` confidence interval, computed by
[`tidy`](@ref) from `confint(r; level)` and hence with the estimator's own reference
distribution (a t distribution with the result's residual degrees of freedom when
these are finite, the normal otherwise; the robust bias-corrected interval for RD).
A dashed line marks zero. When every result contributes a single coefficient (for
example the ATT from several estimators of the same design) each result gets one
row, labelled by `labels`; otherwise rows are coefficients and results are
distinguished by colour, marker shape and a legend, with small vertical offsets.
Results without a variance (e.g. a synthetic control fitted without placebos) are
drawn as points without whiskers.

Graphs of estimates and intervals communicate regression results more accurately
than tables (Kastellec and Leoni 2007), because position along a common scale is the
visual encoding that readers decode most precisely (Cleveland and McGill 1984). Two
cautions apply when comparing rows. Overlap of two intervals is not a test of the
difference between the estimates: independent estimates whose intervals overlap
moderately can still differ significantly (Cumming and Finch 2005), and estimates
computed from the same data are correlated, so their difference needs a joint test
that uses the full covariance matrix (see [`wald_test`](@ref)). And different
estimators generally target different estimands (the ATT, a LATE, a particular
weighted average of cohort effects, ...): aligned rows invite a comparison that is
meaningful only when the estimands coincide, which [`estimand`](@ref) reports.

# Arguments
- `r` / `rs`: a result, or a vector of results, each a [`CausalEstimate`](@ref).

# Keywords
- `level::Real = 0.95`: confidence level of the intervals. They are pointwise; no
  multiplicity adjustment is made across rows.
- `terms = nothing`: the coefficients to show: `nothing` (all), a vector of
  coefficient names, a `Regex`, or a predicate on the name.
- `labels = nothing`: one label per result; the default is the
  [`method_name`](@ref) of each result.
$(_VIZ_COMMON_KW)

# Returns
- `Makie.Figure`; rows run from top to bottom in input order.

# Examples
```julia
using DrSnow, CairoMakie, DataFrames, StableRNGs
rng = StableRNG(1)
g = rand(rng, [0, 4, 6], 150)
df = DataFrame(unit=repeat(1:150; inner=8), year=repeat(1:8, 150))
df.d = Int.((g[df.unit] .> 0) .& (df.year .>= g[df.unit]))
df.y = randn(rng, 150)[df.unit] .+ 0.2 .* df.year .+ df.d .+ randn(rng, nrow(df))
r_twfe = did_twfe(df, :y, :d, :unit, :year; warn_heterogeneity=false)
r_cs = aggregate_att(did_callaway_santanna(df, :y, :d, :unit, :year;
                                           rng=StableRNG(2)), :simple)
plot_coefficients([r_twfe, r_cs]; labels=["TWFE", "Callaway–Sant'Anna"])
```

# References
- Cleveland, W. S., & McGill, R. (1984). Graphical perception: Theory,
  experimentation, and application to the development of graphical methods.
  *Journal of the American Statistical Association*, 79(387), 531–554.
- Cumming, G., & Finch, S. (2005). Inference by eye: Confidence intervals and how to
  read pictures of data. *American Psychologist*, 60(2), 170–180.
- Kastellec, J. P., & Leoni, E. L. (2007). Using graphs instead of tables in
  political science. *Perspectives on Politics*, 5(4), 755–771.
"""
function plot_coefficients end
"""
    plot_coefficients!(ax, r_or_rs; level=0.95, terms=nothing, labels=nothing,
                       colors=nothing) -> ax

Draw a coefficient (forest) plot into an existing Makie axis. This is the mutating
counterpart of [`plot_coefficients`](@ref), which describes the display; no legend
or subtitle is added.

# Arguments
- `ax::Makie.AbstractAxis`: the axis to draw into.
- `r_or_rs`: a [`CausalEstimate`](@ref) or a vector of them.
- Keywords: those of [`plot_coefficients`](@ref) except `figure` and `axis`.

# Returns
- `ax`, the axis drawn into.
"""
function plot_coefficients! end
plot_coefficients(args...; kwargs...) = _viz_no_backend(plot_coefficients, args)
plot_coefficients!(args...; kwargs...) = _viz_no_backend(plot_coefficients!, args)

"""
    plot_rd(pd::RDPlotData; estimate=nothing, ci=false, figure=(;), axis=(;),
            colors=nothing) -> Makie.Figure

Regression discontinuity plot in the layout of `rdplot`: binned means of the outcome,
a global polynomial fit on each side of the cutoff and, optionally, the local
polynomial fits on which the RD estimate is based.

The points are sample means of the outcome within bins of the running variable, with
the number and placement of bins chosen by the data-driven methods of Calonico,
Cattaneo and Titiunik (2015) stored in `pd` (by default evenly spaced bins whose
number mimics the variability of the raw data, which shows the dispersion of the
outcome; IMSE-optimal bins instead trace the underlying regression function). The
solid lines are global polynomial fits (order `pd.p`, 4 by default) on each side,
and a dashed vertical line marks the cutoff. Both are *descriptive*: global
high-order polynomials are a poor basis for estimating the jump (Gelman and Imbens
2019), and the visual size of a discontinuity depends on the binning, so the plot
supports but never replaces formal estimation (Imbens and Lemieux 2008; Cattaneo,
Idrobo and Titiunik 2020).

Passing the [`RDEstimate`](@ref) from [`rd_estimate`](@ref) on the same data overlays
the local polynomial fits within the main bandwidth (dashed) and shades the
bandwidth window: these are the observations and fits the estimate actually uses.
The distance between the two local fits at the cutoff is the conventional local
polynomial estimate. The subtitle reports the robust bias-corrected estimate and its
confidence interval (Calonico, Cattaneo and Titiunik 2014), which is centred on the
bias-corrected estimate and therefore need not be centred on the plotted jump.
Covariate-adjusted estimates cannot be overlaid, because their fitted intercepts are
not outcome levels.

With `ci = true` each bin mean gets its pointwise confidence interval (level
`pd.level`). These intervals describe the sampling variability of each bin mean; they
are neither a test of the discontinuity nor a band for the regression function.

# Arguments
- `pd::RDPlotData`: from [`rd_plot_data`](@ref).

# Keywords
- `estimate::Union{Nothing,RDEstimate} = nothing`: an estimate from
  [`rd_estimate`](@ref) on the same data and cutoff, whose local fits and bandwidth
  are overlaid.
- `ci::Bool = false`: draw the confidence intervals of the bin means.
$(_VIZ_COMMON_KW)

# Returns
- `Makie.Figure`. The subtitle reports the number of bins, the binning method and,
  with `estimate`, the robust bias-corrected estimate and interval.

# Examples
```julia
using DrSnow, CairoMakie, DataFrames, StableRNGs
rng = StableRNG(1)
x = 2 .* rand(rng, 1500) .- 1
df = DataFrame(x=x, y=0.4 .+ 0.8 .* x .- 0.5 .* x .^ 2 .+ 0.6 .* (x .>= 0) .+
                     0.3 .* randn(rng, 1500))
pd = rd_plot_data(df, :y, :x; cutoff=0.0)
fig = plot_rd(pd; estimate=rd_estimate(df, :y, :x), ci=true)
```

# References
- Calonico, S., Cattaneo, M. D., & Titiunik, R. (2014). Robust nonparametric
  confidence intervals for regression-discontinuity designs. *Econometrica*, 82(6),
  2295–2326.
- Calonico, S., Cattaneo, M. D., & Titiunik, R. (2015). Optimal data-driven
  regression discontinuity plots. *Journal of the American Statistical Association*,
  110(512), 1753–1769.
- Cattaneo, M. D., Idrobo, N., & Titiunik, R. (2020). *A Practical Introduction to
  Regression Discontinuity Designs: Foundations*. Cambridge University Press.
- Gelman, A., & Imbens, G. (2019). Why high-order polynomials should not be used in
  regression discontinuity designs. *Journal of Business & Economic Statistics*,
  37(3), 447–456.
- Imbens, G. W., & Lemieux, T. (2008). Regression discontinuity designs: A guide to
  practice. *Journal of Econometrics*, 142(2), 615–635.
"""
function plot_rd end
"""
    plot_rd!(ax, pd::RDPlotData; estimate=nothing, ci=false, colors=nothing) -> ax

Draw an RD plot (binned means, global and optional local polynomial fits) into an
existing Makie axis. This is the mutating counterpart of [`plot_rd`](@ref), which
describes the display; no legend or subtitle is added.

# Arguments
- `ax::Makie.AbstractAxis`: the axis to draw into.
- `pd::RDPlotData`: from [`rd_plot_data`](@ref).
- Keywords: those of [`plot_rd`](@ref) except `figure` and `axis`.

# Returns
- `ax`, the axis drawn into.
"""
function plot_rd! end
plot_rd(args...; kwargs...) = _viz_no_backend(plot_rd, args)
plot_rd!(args...; kwargs...) = _viz_no_backend(plot_rd!, args)

"""
    plot_synth(r; kind=:both, cohort=nothing, placebos=true, placebo_cutoff=Inf,
               time_weights=true, figure=(;), axis=(;), colors=nothing) -> Makie.Figure

Synthetic-control plots: the treated unit's outcome path against its estimated
counterfactual, and the gap between them, optionally against the in-space placebo
gaps of the donor units.

Supported results are [`SyntheticControlEstimate`](@ref),
[`SyntheticDiDEstimate`](@ref), [`AugmentedSCEstimate`](@ref) and
[`MatrixCompletionEstimate`](@ref), all read through [`synth_gaps`](@ref). Two
panels are available, stacked on a shared time axis with `kind = :both` (default):

- `kind = :trajectories`: the treated outcome (solid) and the estimated
  counterfactual (dashed): the donor-weighted average for synthetic control, its
  bias-corrected version for augmented synthetic control, the completed matrix entry
  for matrix completion, and for synthetic DiD the unit-weighted control path *plus*
  the time-weighted pre-treatment level difference (Arkhangelsky et al. 2021). A
  synthetic-DiD counterfactual therefore matches the treated unit in trends rather
  than levels, and its post-period gaps are the synthetic-DiD effects. For synthetic
  DiD the pre-treatment time weights ``\\lambda`` are drawn as scaled bars along the
  bottom of the panel (`time_weights`): the periods they emphasise are those whose
  levels anchor the counterfactual.
- `kind = :gaps`: treated minus counterfactual over time, with a line at zero. For a
  classic synthetic control fitted with `placebo = true` the placebo gaps of all
  donors (each donor treated as if it had been treated, with the remaining units as
  its donor pool) are drawn as a grey "spaghetti" behind the treated gap
  (`placebos`), the display introduced by Abadie, Diamond and Hainmueller (2010).

Reading the panels. Post-treatment gaps are the per-period effect estimates;
pre-treatment gaps show the quality of fit, and a counterfactual that fails to track
the treated unit before treatment gives little reason to trust it afterwards
(Abadie, Diamond and Hainmueller 2010; Abadie 2021). In the spaghetti plot, the
treated gap is informative when it is unusual in the placebo distribution after
treatment while being typical before. Placebos with poor pre-treatment fit carry
little information about the post-period, so Abadie, Diamond and Hainmueller (2010)
also show the plot after discarding donors whose pre-treatment MSPE exceeds 20, 5
or 2 times the treated unit's; here `placebo_cutoff` is on the RMSPE scale, so an
MSPE multiple ``m`` corresponds to `placebo_cutoff = sqrt(m)`. The formal version of
this comparison is the permutation test of [`synth_in_space_placebo`](@ref)
(ratio of post- to pre-treatment RMSPE), whose reference distribution
[`plot_randomization_distribution`](@ref) draws; with ``J`` donors its smallest
attainable p-value is ``1/(J+1)``. No pointwise intervals are drawn around the gaps;
the subtitle reports the average effect and, when available, its standard error.

A dashed vertical line marks the first treated period. With staggered adoption
(synthetic DiD with cohorts) the gap panel shows every cohort and the trajectory
panel shows the cohort selected by `cohort` (default: the earliest).

# Arguments
- `r`: a synthetic-control-type result (see above).

# Keywords
- `kind::Symbol = :both`: `:trajectories`, `:gaps` or `:both`.
- `cohort = nothing`: the adoption cohort shown in the trajectory panel under
  staggered adoption (default: the earliest).
- `placebos::Bool = true`: draw the donors' placebo gaps when `r` stores them.
- `placebo_cutoff::Real = Inf`: omit placebos whose pre-treatment RMSPE exceeds
  `placebo_cutoff` times the treated unit's (`Inf` keeps all).
- `time_weights::Bool = true`: draw the synthetic-DiD time weights.
$(_VIZ_COMMON_KW)

# Returns
- `Makie.Figure` with one panel, or two stacked panels for `kind = :both`.

# Examples
```julia
using DrSnow, CairoMakie, DataFrames, StableRNGs
rng = StableRNG(1)
N, T, T0 = 16, 18, 12
f = cumsum(randn(rng, T)) .+ range(0, 4; length=T)          # common factor
load, α = 0.5 .+ rand(rng, N), 10 .+ 2 .* randn(rng, N)
df = DataFrame(unit=repeat(1:N; inner=T), year=repeat(1991:(1990 + T), N))
df.d = Int.((df.unit .== N) .& (df.year .> 1990 + T0))
df.y = α[df.unit] .+ load[df.unit] .* f[df.year .- 1990] .- 3.0 .* df.d .+
       0.3 .* randn(rng, nrow(df))
r = synthetic_control(df, :y, :d, :unit, :year; placebo=true, rng=StableRNG(2))
fig = plot_synth(r; placebo_cutoff=sqrt(5))
plot_synth(synthetic_did(df, :y, :d, :unit, :year; se_method=:none);
           kind=:trajectories)
```

# References
- Abadie, A. (2021). Using synthetic controls: Feasibility, data requirements, and
  methodological aspects. *Journal of Economic Literature*, 59(2), 391–425.
- Abadie, A., Diamond, A., & Hainmueller, J. (2010). Synthetic control methods for
  comparative case studies: Estimating the effect of California's tobacco control
  program. *Journal of the American Statistical Association*, 105(490), 493–505.
- Arkhangelsky, D., Athey, S., Hirshberg, D. A., Imbens, G. W., & Wager, S. (2021).
  Synthetic difference-in-differences. *American Economic Review*, 111(12),
  4088–4118.
- Athey, S., Bayati, M., Doudchenko, N., Imbens, G., & Khosravi, K. (2021). Matrix
  completion methods for causal panel data models. *Journal of the American
  Statistical Association*, 116(536), 1716–1730.
- Ben-Michael, E., Feller, A., & Rothstein, J. (2021). The augmented synthetic
  control method. *Journal of the American Statistical Association*, 116(536),
  1789–1803.
"""
function plot_synth end
"""
    plot_synth!(ax, r; kind=:trajectories, cohort=nothing, placebos=true,
                placebo_cutoff=Inf, time_weights=true, colors=nothing) -> ax

Draw one synthetic-control panel into an existing Makie axis. This is the mutating
counterpart of [`plot_synth`](@ref), which describes both panels; `kind` must be
`:trajectories` or `:gaps` (`:both` needs a figure), and no legend or subtitle is
added.

# Arguments
- `ax::Makie.AbstractAxis`: the axis to draw into.
- `r`: a synthetic-control-type result, as in [`plot_synth`](@ref).
- Keywords: those of [`plot_synth`](@ref) except `figure` and `axis`.

# Returns
- `ax`, the axis drawn into.
"""
function plot_synth! end
plot_synth(args...; kwargs...) = _viz_no_backend(plot_synth, args)
plot_synth!(args...; kwargs...) = _viz_no_backend(plot_synth!, args)

"""
    plot_randomization_distribution(r; hypothesis=1, bins=nothing, figure=(;),
                                    axis=(;), colors=nothing) -> Makie.Figure

Histogram of a randomization (permutation or placebo) reference distribution, with the
observed test statistic marked.

Randomization inference holds the potential outcomes fixed and treats the assignment
mechanism as the only source of randomness (Fisher 1935; Imbens and Rubin 2015).
Under a *sharp* null hypothesis, which specifies every unit's missing potential
outcome (e.g. no effect for any unit), the test statistic can be recomputed for every
assignment the design could have produced; the histogram is that reference
distribution, with bar heights giving the share of assignments (weighted by their
assignment probabilities when the distribution is enumerated exactly). The solid
vertical line is the observed statistic; for a two-sided test, which compares
``|T|``, a dashed line marks its mirror image ``-T``, and the p-value is the share of
the distribution at least as extreme as the observed value. The legend and subtitle
report the observed value and the p-value.

The p-value is exact in finite samples for the sharp null, whatever the outcome
distribution, but it is discrete: with ``M`` assignments it moves in steps of
roughly ``1/M``, and a design with few distinct assignments cannot produce small
p-values. A non-rejection does not show that the effect is zero for every unit, and
a rejection of the sharp null says nothing about the size of an average effect
without further assumptions (Young 2019 discusses how randomization and
conventional p-values can diverge in practice).

Supported results:
- [`RandomizationTestResult`](@ref) from [`randomization_test`](@ref).
- [`MultipleTestingResult`](@ref) from [`ri_multiple_testing`](@ref): the marginal
  reference distribution of hypothesis number `hypothesis`, with its *unadjusted*
  p-value; the Westfall–Young adjusted p-values (Westfall and Young 1993), which
  account for the other hypotheses, are stored in the result.
- The [`DiagnosticTest`](@ref) returned by [`synth_in_space_placebo`](@ref): the
  statistics of the donor placebos (by default the post/pre-treatment RMSPE ratio,
  a one-sided test), as in Abadie, Diamond and Hainmueller (2010). Here the reference
  set is the donors rather than a set of random assignments, so the p-value has a
  permutation interpretation only under the assumption that the treated unit was
  as good as randomly chosen among the units.

# Arguments
- `r`: the test result (see above).

# Keywords
- `hypothesis::Integer = 1`: the hypothesis (column) shown for a
  [`MultipleTestingResult`](@ref).
- `bins = nothing`: number of histogram bins; the default is about the square root of
  the number of reference values, between 10 and 60.
$(_VIZ_COMMON_KW)

# Returns
- `Makie.Figure`.

# Examples
```julia
using DrSnow, CairoMakie, DataFrames, StableRNGs, Random
rng = StableRNG(1)
df = DataFrame(d=shuffle(rng, repeat([0, 1], 30)), x=randn(rng, 60))
df.y = df.x .+ 0.5 .* df.d .+ randn(rng, 60)
rt = randomization_test(df, :y, :d; nperm=5000, rng=StableRNG(2))
plot_randomization_distribution(rt)
```

# References
- Abadie, A., Diamond, A., & Hainmueller, J. (2010). Synthetic control methods for
  comparative case studies: Estimating the effect of California's tobacco control
  program. *Journal of the American Statistical Association*, 105(490), 493–505.
- Fisher, R. A. (1935). *The Design of Experiments*. Oliver & Boyd.
- Imbens, G. W., & Rubin, D. B. (2015). *Causal Inference for Statistics, Social,
  and Biomedical Sciences: An Introduction*. Cambridge University Press.
- Westfall, P. H., & Young, S. S. (1993). *Resampling-Based Multiple Testing:
  Examples and Methods for p-Value Adjustment*. Wiley.
- Young, A. (2019). Channeling Fisher: Randomization tests and the statistical
  insignificance of seemingly significant experimental results. *Quarterly Journal
  of Economics*, 134(2), 557–598.
"""
function plot_randomization_distribution end
"""
    plot_randomization_distribution!(ax, r; hypothesis=1, bins=nothing,
                                     colors=nothing) -> ax

Draw a randomization reference distribution with the observed statistic into an
existing Makie axis. This is the mutating counterpart of
[`plot_randomization_distribution`](@ref), which describes the display; no legend or
subtitle is added.

# Arguments
- `ax::Makie.AbstractAxis`: the axis to draw into.
- `r`: a supported test result, as in [`plot_randomization_distribution`](@ref).
- Keywords: those of [`plot_randomization_distribution`](@ref) except `figure` and
  `axis`.

# Returns
- `ax`, the axis drawn into.
"""
function plot_randomization_distribution! end
plot_randomization_distribution(args...; kwargs...) =
    _viz_no_backend(plot_randomization_distribution, args)
plot_randomization_distribution!(args...; kwargs...) =
    _viz_no_backend(plot_randomization_distribution!, args)

"""
    plot_bacon(b::BaconDecomposition; figure=(;), axis=(;),
               colors=nothing) -> Makie.Figure

Goodman-Bacon decomposition plot: every two-group, two-period DiD comparison hidden in
a two-way fixed-effects (TWFE) DiD coefficient, plotted as its estimate against its
weight.

Goodman-Bacon (2021) shows that, in a balanced panel with staggered adoption of an
absorbing binary treatment and no covariates, the TWFE coefficient equals a weighted
average ``\\hat\\beta^{TWFE} = \\sum_k s_k \\hat\\beta_k`` of all 2×2 DiD estimates
that compare a timing group with a group whose treatment status does not change in
the window, with non-negative weights ``s_k`` that sum to one and grow with group
sizes and with the variance of treatment in the comparison (groups treated near the
middle of the panel get more weight). The plot shows each ``\\hat\\beta_k`` against
``s_k``, with colour and marker shape by comparison type and a dashed line at the
TWFE estimate, the weighted mean of the points. Points far from the line with large
weight are the comparisons that drive the TWFE estimate.

The two comparison types that use already-treated units as controls ("later vs
earlier treated", "later vs always treated") are the problematic ones: when
treatment effects evolve over time, the change in the controls' outcomes includes
their own treatment-effect dynamics, which can bias TWFE and even reverse its sign.
A plot in which such comparisons carry little weight, or agree with the clean
comparisons, is reassuring about the TWFE summary; it does not validate parallel
trends, which every 2×2 comparison assumes. Heterogeneity-robust estimators
([`did_callaway_santanna`](@ref), [`did_sun_abraham`](@ref),
[`did_imputation`](@ref)) avoid these comparisons by construction.

# Arguments
- `b::BaconDecomposition`: from [`bacon_decomposition`](@ref).

# Keywords
$(_VIZ_COMMON_KW)

# Returns
- `Makie.Figure`.

# Examples
```julia
using DrSnow, CairoMakie, DataFrames, StableRNGs
rng = StableRNG(1)
g = rand(rng, [0, 4, 6], 150)
df = DataFrame(unit=repeat(1:150; inner=8), year=repeat(1:8, 150))
df.d = Int.((g[df.unit] .> 0) .& (df.year .>= g[df.unit]))
df.y = randn(rng, 150)[df.unit] .+ 0.2 .* df.year .+
       df.d .* (1 .+ 0.5 .* (df.year .- g[df.unit])) .+ randn(rng, nrow(df))
plot_bacon(bacon_decomposition(df, :y, :d, :unit, :year))
```

# References
- Goodman-Bacon, A. (2021). Difference-in-differences with variation in treatment
  timing. *Journal of Econometrics*, 225(2), 254–277.
- Roth, J., Sant'Anna, P. H. C., Bilinski, A., & Poe, J. (2023). What's trending in
  difference-in-differences? A synthesis of the recent econometrics literature.
  *Journal of Econometrics*, 235(2), 2218–2244.
"""
function plot_bacon end
"""
    plot_bacon!(ax, b::BaconDecomposition; colors=nothing) -> ax

Draw a Goodman-Bacon decomposition plot into an existing Makie axis. This is the
mutating counterpart of [`plot_bacon`](@ref), which describes the display; no legend
or subtitle is added.

# Arguments
- `ax::Makie.AbstractAxis`: the axis to draw into.
- `b::BaconDecomposition`: from [`bacon_decomposition`](@ref).
- `colors`: as in [`plot_bacon`](@ref).

# Returns
- `ax`, the axis drawn into.
"""
function plot_bacon! end
plot_bacon(args...; kwargs...) = _viz_no_backend(plot_bacon, args)
plot_bacon!(args...; kwargs...) = _viz_no_backend(plot_bacon!, args)

"""
    plot_gates(g::GenericMLInference; figure=(;), axis=(;),
               colors=nothing) -> Makie.Figure

Sorted group average treatment effects (GATES) of the generic machine-learning
inference of Chernozhukov, Demirer, Duflo and Fernández-Val (2025), with the average
treatment effect for reference.

[`generic_ml`](@ref) repeatedly splits a randomized experiment into an auxiliary
half, on which a machine-learning proxy ``S(Z)`` of the conditional average treatment
effect is trained, and a main half, on which units are sorted into ``K`` groups by
quantiles of the proxy. The GATES are ``\\gamma_k = E[Y(1) - Y(0) \\mid G_k]``, the
average effects within those groups. The bars show ``\\hat\\gamma_1, …,
\\hat\\gamma_K`` from the least to the most affected group (as predicted) with their
confidence intervals, and a dashed line with a shaded band shows the average
treatment effect (the BLP coefficient ``\\beta_1``) and its interval. Estimates and
interval bounds are medians over the sample splits; to account for the splitting
uncertainty, each split's intervals are computed at level ``1 - \\alpha/2`` so that
the reported intervals have nominal level ``1 - \\alpha`` (the result's `level`), a
conservative construction. The subtitle reports the most-minus-least-affected
difference ``\\gamma_K - \\gamma_1`` and its p-value.

The GATES are valid inference on features of the treatment-effect heterogeneity
*along the proxy*, whatever the quality of the proxy: a flat profile means that the
proxy did not detect heterogeneity, not that there is none, and an increasing
profile shows that the proxy ranks units by their effects. The GATES are not
estimates of individual or conditional effects; use [`clan`](@ref) to describe who
the most and least affected units are. Intervals are pointwise across groups.

# Arguments
- `g::GenericMLInference`: from [`generic_ml`](@ref).

# Keywords
$(_VIZ_COMMON_KW)

# Returns
- `Makie.Figure`.

# Examples
```julia
using DrSnow, CairoMakie, DataFrames, StableRNGs
rng = StableRNG(1)
df = DataFrame(x1=randn(rng, 600), x2=randn(rng, 600), d=rand(rng, [0, 1], 600))
df.y = 0.5 .* df.x1 .+ df.d .* (0.4 .+ 0.8 .* df.x1) .+ randn(rng, 600)
g = generic_ml(df, :y, :d; covariates=[:x1, :x2], proxy_learner=OLSLearner(),
               propensity=0.5, n_splits=20, n_groups=4, rng=StableRNG(2))
plot_gates(g)
```

# References
- Chernozhukov, V., Demirer, M., Duflo, E., & Fernández-Val, I. (2025).
  Fisher–Schultz lecture: Generic machine learning inference on heterogeneous
  treatment effects in randomized experiments. *Econometrica*, 93(4), 1121–1164.
"""
function plot_gates end
"""
    plot_gates!(ax, g::GenericMLInference; colors=nothing) -> ax

Draw a GATES plot into an existing Makie axis. This is the mutating counterpart of
[`plot_gates`](@ref), which describes the display; no legend or subtitle is added.

# Arguments
- `ax::Makie.AbstractAxis`: the axis to draw into.
- `g::GenericMLInference`: from [`generic_ml`](@ref).
- `colors`: as in [`plot_gates`](@ref).

# Returns
- `ax`, the axis drawn into.
"""
function plot_gates! end
plot_gates(args...; kwargs...) = _viz_no_backend(plot_gates, args)
plot_gates!(args...; kwargs...) = _viz_no_backend(plot_gates!, args)

"""
    plot_cate(c::CATEPredictor; modifier=nothing, level=0.95, bins=30, figure=(;),
              axis=(;), colors=nothing) -> Makie.Figure
    plot_cate(f::Union{CausalForest,InstrumentalForest}; modifier=nothing, level=0.95,
              bins=30, figure=(;), axis=(;), colors=nothing) -> Makie.Figure

Distribution of estimated conditional average treatment effects (CATEs), overall or
against one effect modifier, with the average effect for reference.

For a DR-learner ([`cate_dr_learner`](@ref), Kennedy 2023) the plot uses the
cross-fitted (out-of-fold) predictions ``\\hat\\tau(x_i)``; for a causal forest
([`causal_forest`](@ref)) or an instrumental forest ([`instrumental_forest`](@ref),
whose target is a conditional LATE) the out-of-bag estimates (Wager and Athey 2018;
Athey, Tibshirani and Wager 2019). Without `modifier` the plot is a histogram of the
estimates, with the AIPW average effect as a vertical line and its `level`
confidence interval as a shaded band. With `modifier = :x` the estimates are plotted
against covariate `x`, with the average effect as a horizontal line and band; for a
forest every estimate also carries its pointwise `level` confidence interval from
the little-bags variance estimator ([`predict_interval`](@ref)).

This plot is descriptive and easy to over-read. The spread of the histogram mixes
true heterogeneity with estimation noise (which widens it) and regularization (which
narrows it), so it does not estimate the distribution of individual effects. Forest
intervals are pointwise: with hundreds of units, many intervals will exclude the
average effect by chance even without heterogeneity. A pattern against `modifier`
is a marginal description that can reflect other, correlated covariates. For
inference on heterogeneity use the best linear projection
([`cate_projection`](@ref)), sorted group effects ([`gates`](@ref),
[`plot_gates`](@ref)) or the rank-weighted average treatment effect
([`rank_average_treatment_effect`](@ref), [`plot_rate`](@ref)).

# Arguments
- `c::CATEPredictor`: from [`cate_dr_learner`](@ref); or
- `f`: a [`CausalForest`](@ref) or [`InstrumentalForest`](@ref).

# Keywords
- `modifier::Union{Nothing,Symbol} = nothing`: an effect modifier of the DR-learner
  or a covariate of the forest to put on the horizontal axis; `nothing` draws a
  histogram.
- `level::Real = 0.95`: confidence level of the average-effect band and of the
  forest's pointwise intervals.
- `bins::Integer = 30`: number of histogram bins.
$(_VIZ_COMMON_KW)

# Returns
- `Makie.Figure`.

# Examples
```julia
using DrSnow, CairoMakie, DataFrames, StableRNGs
rng = StableRNG(1)
df = DataFrame(x1=rand(rng, 400), x2=rand(rng, 400), d=Float64.(rand(rng, 400) .< 0.5))
df.y = df.x2 .+ (1 .+ 2 .* df.x1) .* df.d .+ randn(rng, 400)
cf = causal_forest(df, :y, :d; covariates=[:x1, :x2], num_trees=200, w_hat=0.5,
                   rng=StableRNG(2))
plot_cate(cf; modifier=:x1)
c = cate_dr_learner(df, :y, :d; covariates=[:x1, :x2], outcome_learner=OLSLearner(),
                    cate_learner=OLSLearner(), rng=StableRNG(3))
plot_cate(c)
```

# References
- Athey, S., Tibshirani, J., & Wager, S. (2019). Generalized random forests. *Annals
  of Statistics*, 47(2), 1148–1178.
- Kennedy, E. H. (2023). Towards optimal doubly robust estimation of heterogeneous
  causal effects. *Electronic Journal of Statistics*, 17(2), 3008–3049.
- Wager, S., & Athey, S. (2018). Estimation and inference of heterogeneous treatment
  effects using random forests. *Journal of the American Statistical Association*,
  113(523), 1228–1242.
"""
function plot_cate end
"""
    plot_cate!(ax, c; modifier=nothing, level=0.95, bins=30, colors=nothing) -> ax

Draw estimated CATEs (histogram, or against an effect modifier) into an existing
Makie axis. This is the mutating counterpart of [`plot_cate`](@ref), which describes
the display and its limits; no legend or subtitle is added.

# Arguments
- `ax::Makie.AbstractAxis`: the axis to draw into.
- `c`: a [`CATEPredictor`](@ref), [`CausalForest`](@ref) or
  [`InstrumentalForest`](@ref).
- Keywords: those of [`plot_cate`](@ref) except `figure` and `axis`.

# Returns
- `ax`, the axis drawn into.
"""
function plot_cate! end
plot_cate(args...; kwargs...) = _viz_no_backend(plot_cate, args)
plot_cate!(args...; kwargs...) = _viz_no_backend(plot_cate!, args)

"""
    plot_variable_importance(f::GeneralizedRandomForest; top=nothing,
                             decay_exponent=2, max_depth=4, figure=(;), axis=(;),
                             colors=nothing) -> Makie.Figure

Horizontal bar chart of a generalized random forest's split-frequency variable
importance, sorted from the most to the least important covariate.

The importance of covariate ``j`` ([`variable_importance`](@ref), as in grf) is the
share of the forest's splits made on ``j`` at each depth ``d \\le`` `max_depth`,
averaged over depths with weights proportional to ``d^{-\\text{decay\\_exponent}}``,
so that splits near the root count most. It measures how much the forest *uses* a
covariate, which is a property of the fitted algorithm rather than of the causal
effect: it is not a test of effect modification, carries no uncertainty
quantification, splits credit between correlated covariates, and split-based
importance measures tend to favour covariates with many possible split points
(Strobl, Boulesteix, Zeileis and Hothorn 2007). For a causal forest the splits
target heterogeneity in the treatment effect, so importance is a useful screening
device for which covariates to examine with [`cate_projection`](@ref) or
[`plot_cate`](@ref) (Athey and Wager 2019), not a substitute for that analysis.

# Arguments
- `f::GeneralizedRandomForest`: a fitted forest (e.g. from [`causal_forest`](@ref)).

# Keywords
- `top::Union{Nothing,Integer} = nothing`: show only the `top` most important
  covariates (default: all).
- `decay_exponent::Real = 2`: exponent of the depth weights; larger values
  concentrate the measure on splits near the root.
- `max_depth::Integer = 4`: deepest level of the trees that is counted.
$(_VIZ_COMMON_KW)

# Returns
- `Makie.Figure`.

# Examples
```julia
using DrSnow, CairoMakie, DataFrames, StableRNGs
rng = StableRNG(1)
df = DataFrame(x1=rand(rng, 400), x2=rand(rng, 400), x3=rand(rng, 400),
               d=Float64.(rand(rng, 400) .< 0.5))
df.y = df.x2 .+ (1 .+ 2 .* df.x1) .* df.d .+ randn(rng, 400)
cf = causal_forest(df, :y, :d; covariates=[:x1, :x2, :x3], num_trees=200,
                   w_hat=0.5, rng=StableRNG(2))
plot_variable_importance(cf)
```

# References
- Athey, S., & Wager, S. (2019). Estimating treatment effects with causal forests:
  An application. *Observational Studies*, 5(2), 37–51.
- Athey, S., Tibshirani, J., & Wager, S. (2019). Generalized random forests. *Annals
  of Statistics*, 47(2), 1148–1178.
- Strobl, C., Boulesteix, A.-L., Zeileis, A., & Hothorn, T. (2007). Bias in random
  forest variable importance measures: Illustrations, sources and a solution. *BMC
  Bioinformatics*, 8, 25.
"""
function plot_variable_importance end
"""
    plot_variable_importance!(ax, f::GeneralizedRandomForest; top=nothing,
                              decay_exponent=2, max_depth=4, colors=nothing) -> ax

Draw a variable-importance bar chart into an existing Makie axis. This is the
mutating counterpart of [`plot_variable_importance`](@ref), which defines the measure
and its limits; no title or subtitle is added.

# Arguments
- `ax::Makie.AbstractAxis`: the axis to draw into.
- `f::GeneralizedRandomForest`: a fitted forest.
- Keywords: those of [`plot_variable_importance`](@ref) except `figure` and `axis`.

# Returns
- `ax`, the axis drawn into.
"""
function plot_variable_importance! end
plot_variable_importance(args...; kwargs...) =
    _viz_no_backend(plot_variable_importance, args)
plot_variable_importance!(args...; kwargs...) =
    _viz_no_backend(plot_variable_importance!, args)

"""
    plot_rate(r::RATEEstimate; level=0.95, figure=(;), axis=(;),
              colors=nothing) -> Makie.Figure

Targeting operator characteristic (TOC) curve of a rank-weighted average treatment
effect (RATE) estimate: how much larger the average effect is among the units that a
prioritization rule ranks first.

For a priority score ``S(X)`` (for example a CATE prediction or a risk score),
Yadlowsky, Fleming, Shah, Brunskill and Wager (2025) define
``\\mathrm{TOC}(q) = E[Y(1) - Y(0) \\mid F(S(X)) \\ge 1 - q] - E[Y(1) - Y(0)]``, the
average effect among the top ``q`` share of units by priority minus the overall
average effect. The plot draws the estimated ``\\mathrm{TOC}(q)`` against ``q`` for
each priority rule in `r` (and their difference when two rules are compared), with
pointwise `level` bands from the half-sample bootstrap and a line at zero. The RATE
summarizes the curve as a weighted area under it: the AUTOC weights all ``q``
equally and emphasises the top of the ranking, and the Qini weights by ``q`` and
emphasises broad targeting; the subtitle reports the estimate(s) with standard
errors.

A curve well above zero at small ``q`` means the rule identifies units with larger
effects; a curve near zero means no detectable targeting value; ``\\mathrm{TOC}(1) =
0`` by construction. The bands are pointwise in ``q`` and are not a test of the
whole curve; the test of "no targeting value" is the RATE itself. Valid evaluation
requires that the priorities were not fitted on the evaluation data (e.g. estimated
on a training split), since in-sample priorities overstate targeting value.

# Arguments
- `r::RATEEstimate`: from [`rank_average_treatment_effect`](@ref).

# Keywords
- `level::Real = 0.95`: confidence level of the pointwise bands.
$(_VIZ_COMMON_KW)

# Returns
- `Makie.Figure`.

# Examples
```julia
using DrSnow, CairoMakie, DataFrames, StableRNGs
rng = StableRNG(1)
df = DataFrame(x1=rand(rng, 800), x2=rand(rng, 800), d=Float64.(rand(rng, 800) .< 0.5))
df.y = df.x2 .+ (1 .+ 2 .* df.x1) .* df.d .+ randn(rng, 800)
train, test = df[1:400, :], df[401:800, :]
cf_train = causal_forest(train, :y, :d; covariates=[:x1, :x2], w_hat=0.5,
                         num_trees=200, rng=StableRNG(2))
cf_test = causal_forest(test, :y, :d; covariates=[:x1, :x2], w_hat=0.5,
                        num_trees=200, rng=StableRNG(3))
priority = predict(cf_train, Matrix(test[:, [:x1, :x2]]))   # held-out priorities
plot_rate(rank_average_treatment_effect(cf_test, priority; R=100, rng=StableRNG(4)))
```

# References
- Yadlowsky, S., Fleming, S., Shah, N., Brunskill, E., & Wager, S. (2025).
  Evaluating treatment prioritization rules via rank-weighted average treatment
  effects. *Journal of the American Statistical Association*, 120(549), 38–51.
"""
function plot_rate end
"""
    plot_rate!(ax, r::RATEEstimate; level=0.95, colors=nothing) -> ax

Draw TOC curves into an existing Makie axis. This is the mutating counterpart of
[`plot_rate`](@ref), which defines the curve and its reading; no legend or subtitle
is added.

# Arguments
- `ax::Makie.AbstractAxis`: the axis to draw into.
- `r::RATEEstimate`: from [`rank_average_treatment_effect`](@ref).
- Keywords: those of [`plot_rate`](@ref) except `figure` and `axis`.

# Returns
- `ax`, the axis drawn into.
"""
function plot_rate! end
plot_rate(args...; kwargs...) = _viz_no_backend(plot_rate, args)
plot_rate!(args...; kwargs...) = _viz_no_backend(plot_rate!, args)

"""
    plot_spillover_rings(r; level=0.95, figure=(;), axis=(;),
                         colors=nothing) -> Makie.Figure

Direct and spillover effects by exposure group, from spatial or network
difference-in-differences and exposure-mapping estimators.

Under interference a unit's outcome depends on other units' treatments, and the
estimands are defined through an *exposure mapping* that summarizes the treatment
of a unit's neighbourhood, for instance rings of distance to the nearest treated
unit (Aronow and Samii 2017; Butts 2021). Two displays are produced:

- For a [`SpilloverEventStudy`](@ref) ([`spillover_event_study`](@ref)): one
  event-study series per group, `treated` (event time from own adoption) and
  `exposed:<ring>` (never-treated units, event time from first exposure), each
  normalized to zero at the omitted reference period ``-1`` (hollow markers), with
  binned endpoints labelled `≤` / `≥` and pointwise `level` intervals from the
  model's reference distribution. It is read like [`plot_event_study`](@ref):
  exposed-group leads near zero are consistent with, but do not establish, parallel
  trends for that group.
- For a [`SpilloverRegression`](@ref) ([`spillover_did`](@ref),
  [`exposure_regression`](@ref)), an [`ExposureEffects`](@ref) or a
  `TwoStageEffects` result: point estimates with pointwise `level` intervals by
  term, direct effects first and then the spillover exposures in order (e.g. rings
  from nearest to farthest), distinguished by colour and marker shape.

Every coefficient is a contrast with the clean controls, the units beyond the
outermost ring (or with no treated neighbour), which are assumed to be unaffected.
That assumption cannot be checked from these estimates: if spillovers reach the
controls, every effect is biased. Spillover estimates that decline towards zero
with distance are consistent with a correctly chosen outermost ring; estimates that
remain large at the last ring suggest that the rings are too narrow. Cells with few
units (or clusters) give imprecise and possibly under-covering intervals.

# Arguments
- `r`: a [`SpilloverEventStudy`](@ref), [`SpilloverRegression`](@ref),
  [`ExposureEffects`](@ref) or `TwoStageEffects` result.

# Keywords
- `level::Real = 0.95`: confidence level of the pointwise intervals.
$(_VIZ_COMMON_KW)

# Returns
- `Makie.Figure`.

# Examples
```julia
using DrSnow, CairoMakie, DataFrames, StableRNGs
rng = StableRNG(1)
N, T = 500, 7
s = SpatialStructure(["c\$(i)" for i in 1:N]; x=100 .* rand(rng, N),
                     y=100 .* rand(rng, N))
tr = rand(rng, N) .< 0.12
panel = DataFrame(id=repeat(s.ids, T), t=repeat(1:T; inner=N))
panel.d = Int.(tr[repeat(1:N, T)] .& (panel.t .>= 4))
panel.y = randn(rng, nrow(panel)) .+ 2.0 .* panel.d
es = spillover_event_study(panel, :y, :d, s; unit=:id, time=:t,
                           exposure=RingExposure([5.0, 10.0]), leads=3, lags=2)
plot_spillover_rings(es)
```

# References
- Aronow, P. M., & Samii, C. (2017). Estimating average causal effects under general
  interference, with application to a social network experiment. *Annals of
  Applied Statistics*, 11(4), 1912–1947.
- Butts, K. (2021). Difference-in-differences estimation with spatial spillovers.
  arXiv:2105.03737.
"""
function plot_spillover_rings end
"""
    plot_spillover_rings!(ax, r; level=0.95, colors=nothing) -> ax

Draw direct and spillover effects into an existing Makie axis. This is the mutating
counterpart of [`plot_spillover_rings`](@ref), which describes the display and its
identifying assumption; no legend or subtitle is added.

# Arguments
- `ax::Makie.AbstractAxis`: the axis to draw into.
- `r`: a spillover result, as in [`plot_spillover_rings`](@ref).
- Keywords: those of [`plot_spillover_rings`](@ref) except `figure` and `axis`.

# Returns
- `ax`, the axis drawn into.
"""
function plot_spillover_rings! end
plot_spillover_rings(args...; kwargs...) = _viz_no_backend(plot_spillover_rings, args)
plot_spillover_rings!(args...; kwargs...) = _viz_no_backend(plot_spillover_rings!, args)

"""
    plot_confidence_set(s::WeakIVConfidenceSet; limits=nothing, npoints=801,
                        wald=nothing, figure=(;), axis=(;),
                        colors=nothing) -> Makie.Figure

Test-inversion view of a weak-instrument-robust confidence set: the p-value of
``H_0: \\beta = b`` as a function of the hypothesized effect ``b``, with the accepted
region shaded.

A weak-instrument-robust confidence set collects every value ``b`` that a test with
correct size under weak instruments does not reject: the Anderson–Rubin test
(Anderson and Rubin 1949), the conditional likelihood-ratio test (Moreira 2003), the
K (Lagrange multiplier) test (Kleibergen 2002) or a jackknife Anderson–Rubin test,
as selected in [`weak_iv_confidence_set`](@ref). The curve is the test's p-value
function evaluated on a grid of `npoints` values, the dashed horizontal line is the
significance level `1 - level`, and the shaded region, where the curve lies above
the line, is the confidence set. Because the set is obtained by inverting a test
rather than as "estimate ± critical value × standard error", it can be a bounded
interval, a union of two disjoint half-lines, or the whole real line. The latter two
shapes are the signature of weak identification and mean that the data cannot rule
out arbitrarily large effects, not that the computation failed. The subtitle writes
out the set.

Passing the [`IVEstimate`](@ref) as `wald` adds the conventional Wald interval as a
bar below the axis. When it is much shorter than the robust set, or excludes values
the robust test accepts, the precision of the conventional interval is illusory
(Andrews, Stock and Sun 2019). A vertical line marks the point estimate, which the
robust set need not contain symmetrically.

# Arguments
- `s::WeakIVConfidenceSet`: from [`weak_iv_confidence_set`](@ref).

# Keywords
- `limits = nothing`: `(lo, hi)` range of ``b`` to plot; the default covers the
  point estimate, the finite end points of the set and, with `wald`, the Wald
  interval, with extra room when the set is unbounded.
- `npoints::Integer = 801`: number of grid points at which the p-value function is
  evaluated (at least 10).
- `wald::Union{Nothing,IVEstimate} = nothing`: the IV estimate whose conventional
  Wald interval is drawn for comparison.
$(_VIZ_COMMON_KW)

# Returns
- `Makie.Figure`.

# Examples
```julia
using DrSnow, CairoMakie, DataFrames, StableRNGs
rng = StableRNG(1)
n = 500
z = Float64.(rand(rng, n) .< 0.5)
u = randn(rng, n)
df = DataFrame(z=z, d=0.1 .* z .+ 0.6 .* u .+ randn(rng, n))      # weak first stage
df.y = df.d .+ u .+ randn(rng, n)
r = iv_regression(df, :y, :d, :z)
plot_confidence_set(weak_iv_confidence_set(r; method=:ar); wald=r)
```

# References
- Anderson, T. W., & Rubin, H. (1949). Estimation of the parameters of a single
  equation in a complete system of stochastic equations. *Annals of Mathematical
  Statistics*, 20(1), 46–63.
- Andrews, I., Stock, J. H., & Sun, L. (2019). Weak instruments in instrumental
  variables regression: Theory and practice. *Annual Review of Economics*, 11,
  727–753.
- Kleibergen, F. (2002). Pivotal statistics for testing structural parameters in
  instrumental variables regression. *Econometrica*, 70(5), 1781–1803.
- Moreira, M. J. (2003). A conditional likelihood ratio test for structural models.
  *Econometrica*, 71(4), 1027–1048.
"""
function plot_confidence_set end
"""
    plot_confidence_set!(ax, s::WeakIVConfidenceSet; limits=nothing, npoints=801,
                         wald=nothing, colors=nothing) -> ax

Draw the p-value function and accepted region of a weak-instrument-robust confidence
set into an existing Makie axis. This is the mutating counterpart of
[`plot_confidence_set`](@ref), which describes the display; no legend or subtitle is
added.

# Arguments
- `ax::Makie.AbstractAxis`: the axis to draw into.
- `s::WeakIVConfidenceSet`: from [`weak_iv_confidence_set`](@ref).
- Keywords: those of [`plot_confidence_set`](@ref) except `figure` and `axis`.

# Returns
- `ax`, the axis drawn into.
"""
function plot_confidence_set! end
plot_confidence_set(args...; kwargs...) = _viz_no_backend(plot_confidence_set, args)
plot_confidence_set!(args...; kwargs...) = _viz_no_backend(plot_confidence_set!, args)

include("diagnostics.jl")
include("data_adaptive.jl")
include("data_sequential.jl")      # after _VIZ_COMMON_KW (used in its docstrings)

include("data_design.jl")   # design-area plots (power curves, blocks)
