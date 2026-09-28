# Viz: design-diagnostic plots (stubs and docstrings) and the DrSnow Makie theme.
#
# Methods live in the Makie extension (ext/DrSnowMakieExt/trends.jl, balance.jl,
# rd_falsification.jl, honest_did.jl, iv_design.jl, synth_in_time.jl, theme.jl);
# the data they draw come from src/viz/data_diagnostics.jl. Exports are in viz.jl.

"""
    plot_trends(data, outcome, treatment, unit, time; by=:cohort, covariates=Symbol[],
                cohorts=nothing, level=0.95, ci=true, figure=(;), axis=(;),
                colors=nothing) -> Makie.Figure
    plot_trends(data, outcome, tm::TreatmentTiming; kwargs...) -> Makie.Figure

Raw outcome trends of a panel with (possibly staggered) treatment adoption: the mean
outcome in every period for each adoption cohort and for the never-treated units, or
for ever- versus never-treated units.

With `by = :cohort` (default) there is one line per adoption cohort (units first
treated in the same period) plus the never-treated units, and dashed vertical lines
in each cohort's colour mark its adoption date; with `by = :treated` there are two
lines, ever treated and never treated, and a dashed line for every adoption date. The
shaded bands are pointwise `level` confidence intervals for each period's mean
(t intervals from the cross-section of units in that group and period). They
describe the precision of each mean separately; they are not intervals for
differences between groups or for changes over time, which are correlated within
units. With `covariates` the outcome is first adjusted for covariate composition,
``y - (x - \\bar x)'\\hat\\beta``, with ``\\hat\\beta`` estimated within group ×
period cells so that the adjustment does not remove differences in trends; the
subtitle reports the adjustment.

This is a descriptive view of the data behind a difference-in-differences design,
and it is worth looking at before any estimation (Roth, Sant'Anna, Bilinski and Poe
2023). Two limits apply. Parallel pre-treatment paths are neither necessary nor
sufficient for parallel trends, which concern untreated potential outcomes in the
post-treatment periods: level differences are allowed, and diverging paths can
reflect anticipation or composition changes as well as a violation. And with
staggered adoption the cohorts' pre-periods are different calendar periods, so a
cohort-by-cohort visual comparison is the relevant one (Callaway and Sant'Anna
2021). Use an event study with [`pre_trend_test`](@ref), and
[`honest_did`](@ref) for inference that allows bounded violations, for formal
assessments; conditioning the analysis on a visual or statistical pre-test has
consequences for inference (Roth 2022).

# Arguments
- `data`: a table with one row per unit and period.
- `outcome::Symbol`: the outcome column.
- `treatment`: a 0/1 treatment column (absorbing) or a [`FirstTreated`](@ref)
  specification, as in the DiD estimators.
- `unit::Symbol`, `time::Symbol`: unit identifier and time period columns.
- `tm::TreatmentTiming`: alternatively, the adoption timing built from the same
  `data` rows by [`treatment_timing`](@ref).

# Keywords
- `by::Symbol = :cohort`: `:cohort` (one line per adoption cohort) or `:treated`
  (ever versus never treated).
- `covariates::Vector{Symbol} = Symbol[]`: covariates to adjust for before averaging.
- `cohorts = nothing`: a subset of cohorts (adoption dates, in time units) to show;
  the never-treated group is always shown.
- `level::Real = 0.95`: confidence level of the pointwise bands.
- `ci::Bool = true`: draw the bands.
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
df.x = randn(rng, nrow(df))
df.y = randn(rng, 150)[df.unit] .+ 0.2 .* df.year .+ 0.5 .* df.x .+ df.d .+
       randn(rng, nrow(df))
plot_trends(df, :y, :d, :unit, :year)                              # by cohort
plot_trends(df, :y, :d, :unit, :year; by=:treated, covariates=[:x])
```

# References
- Callaway, B., & Sant'Anna, P. H. C. (2021). Difference-in-differences with
  multiple time periods. *Journal of Econometrics*, 225(2), 200–230.
- Roth, J. (2022). Pretest with caution: Event-study estimates after testing for
  parallel trends. *American Economic Review: Insights*, 4(3), 305–322.
- Roth, J., Sant'Anna, P. H. C., Bilinski, A., & Poe, J. (2023). What's trending in
  difference-in-differences? A synthesis of the recent econometrics literature.
  *Journal of Econometrics*, 235(2), 2218–2244.
"""
function plot_trends end
"""
    plot_trends!(ax, data, outcome, treatment, unit, time; kwargs...) -> ax
    plot_trends!(ax, data, outcome, tm::TreatmentTiming; kwargs...) -> ax

Draw raw outcome trends into an existing Makie axis. This is the mutating
counterpart of [`plot_trends`](@ref), which describes the display and its limits; no
legend or subtitle is added.

# Arguments
- `ax::Makie.AbstractAxis`: the axis to draw into.
- `data`, `outcome`, `treatment`, `unit`, `time` (or `tm`): as in
  [`plot_trends`](@ref).
- Keywords: those of [`plot_trends`](@ref) except `figure` and `axis`.

# Returns
- `ax`, the axis drawn into.
"""
function plot_trends! end
plot_trends(args...; kwargs...) = _viz_no_backend(plot_trends, args)
plot_trends!(args...; kwargs...) = _viz_no_backend(plot_trends!, args)

"""
    plot_balance(x; data=nothing, threshold=nothing, annotate=true, figure=(;),
                 axis=(;), colors=nothing) -> Makie.Figure

Covariate balance ("love") plot: one row per covariate, with a measure of imbalance
between the groups being compared on the horizontal axis and a dashed line at zero.

Balance of predetermined covariates is the standard observable check of a design:
covariates cannot be affected by treatment, so systematic differences between the
compared groups signal that the groups differ in other ways too. Three inputs are
supported, each with its own measure:

- A [`pretreatment_balance`](@ref) table (difference-in-differences): standardized
  differences of each adoption cohort against its comparison group, the difference
  in means divided by the square root of the average of the two group variances
  (Austin 2009; Imbens and Rubin 2015), one marker per cohort. Dotted lines at
  `±threshold` (default 0.1) mark a common rule of thumb for a "small" standardized
  difference; it is a descriptive convention, not a test, and it is scale-free
  precisely because it does not depend on the sample size.
- An [`ri_balance_test`](@ref) result (experiments), or its `details.per_covariate`
  table: standardized differences between treated and control units, with the
  randomization p-value of each covariate written next to its row, unadjusted and
  with the Westfall–Young step-down adjustment for testing many covariates
  (Westfall and Young 1993); the caption reports the omnibus test. In a randomized
  experiment some imbalance is expected by chance, and a small p-value is a signal
  worth investigating, not proof of a failed randomization.
- An [`rd_covariate_balance`](@ref) table (regression discontinuity): the
  bias-corrected discontinuity in each covariate at the cutoff with its robust
  confidence interval, and unadjusted and Holm-adjusted p-values (Holm 1979).
  Covariates are on their own scales; pass the estimation `data` to divide each
  estimate and interval by the covariate's standard deviation, so that all
  covariates share one axis. A discontinuity in a predetermined covariate
  undermines the continuity assumptions of the design (Cattaneo, Idrobo and
  Titiunik 2020).

In all three cases the absence of imbalance in observed covariates is consistent
with, but does not establish, balance in unobserved determinants of the outcome.
Rows are ordered as in the input, first covariate at the top.

# Arguments
- `x`: a [`pretreatment_balance`](@ref) table, an [`ri_balance_test`](@ref) result
  or its per-covariate table, or an [`rd_covariate_balance`](@ref) table.

# Keywords
- `data = nothing`: the estimation data, used only for RD tables, to express
  discontinuities in standard-deviation units.
- `threshold = nothing`: position of the dotted reference lines at `±threshold`;
  `nothing` uses the default for the input (0.1 for standardized differences, none
  for RD), and `false` draws none.
- `annotate::Bool = true`: write the p-values next to each row when the input has
  them.
$(_VIZ_COMMON_KW)

# Returns
- `Makie.Figure`.

# Examples
```julia
using DrSnow, CairoMakie, DataFrames, StableRNGs, Random
rng = StableRNG(1)
expt = DataFrame(d=shuffle(rng, repeat([0, 1], 40)), age=randn(rng, 80),
                 income=randn(rng, 80))
plot_balance(ri_balance_test(expt, :d, [:age, :income]; nperm=500,
                             rng=StableRNG(2)))
x = 2 .* rand(rng, 2000) .- 1
rdd = DataFrame(x=x, z1=1 .+ 0.5 .* x .+ randn(rng, 2000),
                z2=10 .+ 5 .* x .+ 3 .* randn(rng, 2000))
plot_balance(rd_covariate_balance(rdd, [:z1, :z2], :x); data=rdd)
```

# References
- Austin, P. C. (2009). Balance diagnostics for comparing the distribution of
  baseline covariates between treatment groups in propensity-score matched samples.
  *Statistics in Medicine*, 28(25), 3083–3107.
- Cattaneo, M. D., Idrobo, N., & Titiunik, R. (2020). *A Practical Introduction to
  Regression Discontinuity Designs: Foundations*. Cambridge University Press.
- Holm, S. (1979). A simple sequentially rejective multiple test procedure.
  *Scandinavian Journal of Statistics*, 6(2), 65–70.
- Imbens, G. W., & Rubin, D. B. (2015). *Causal Inference for Statistics, Social,
  and Biomedical Sciences: An Introduction*. Cambridge University Press.
- Westfall, P. H., & Young, S. S. (1993). *Resampling-Based Multiple Testing:
  Examples and Methods for p-Value Adjustment*. Wiley.
"""
function plot_balance end
"""
    plot_balance!(ax, x; data=nothing, threshold=nothing, annotate=true,
                  colors=nothing) -> ax

Draw a covariate balance plot into an existing Makie axis. This is the mutating
counterpart of [`plot_balance`](@ref), which describes the supported inputs and
measures; no legend or caption is added.

# Arguments
- `ax::Makie.AbstractAxis`: the axis to draw into.
- `x`: a balance table or test, as in [`plot_balance`](@ref).
- Keywords: those of [`plot_balance`](@ref) except `figure` and `axis`.

# Returns
- `ax`, the axis drawn into.
"""
function plot_balance! end
plot_balance(args...; kwargs...) = _viz_no_backend(plot_balance, args)
plot_balance!(args...; kwargs...) = _viz_no_backend(plot_balance!, args)

"""
    plot_rd_placebos(tab; estimate=nothing, figure=(;), axis=(;),
                     colors=nothing) -> Makie.Figure

Placebo-cutoff plot for a regression discontinuity design: RD estimates at artificial
cutoffs, where no treatment changes, against the cutoff location.

Each point is the bias-corrected RD estimate with its robust confidence interval
(Calonico, Cattaneo and Titiunik 2014) at a placebo cutoff from
[`rd_placebo_cutoffs`](@ref). Placebo cutoffs below the true cutoff use only
observations below it (controls) and those above only observations above it
(treated), so that the true discontinuity cannot contaminate them. The true cutoff
is marked by a dashed line and, when `estimate` is given, the actual estimate is
highlighted there for comparison. The outcome's regression function should be
continuous at the placebo cutoffs, so estimates there should be close to zero
(Imbens and Lemieux 2008; Cattaneo, Idrobo and Titiunik 2020).

Discontinuities at placebo cutoffs weaken the case that the jump at the true cutoff
is caused by treatment, since they show that the regression function has jumps or
kinks the local polynomial does not absorb. Their absence does not validate the
design: the test is informative only about the placebo locations, and with many
placebo cutoffs some intervals will exclude zero by chance (the intervals are
pointwise). Placebo estimates use fewer observations than the actual estimate and
are correspondingly less precise.

# Arguments
- `tab`: the table returned by [`rd_placebo_cutoffs`](@ref).

# Keywords
- `estimate::Union{Nothing,RDEstimate} = nothing`: the estimate at the true cutoff,
  from [`rd_estimate`](@ref) on the same data.
$(_VIZ_COMMON_KW)

# Returns
- `Makie.Figure`. The caption notes placebo cutoffs that could not be estimated.

# Examples
```julia
using DrSnow, CairoMakie, DataFrames, StableRNGs
rng = StableRNG(1)
x = 2 .* rand(rng, 2000) .- 1
df = DataFrame(x=x, y=0.4 .+ 0.8 .* x .+ 0.6 .* (x .>= 0) .+ 0.3 .* randn(rng, 2000))
tab = rd_placebo_cutoffs(df, :y, :x; placebo_cutoffs=[-0.6, -0.3, 0.3, 0.6])
plot_rd_placebos(tab; estimate=rd_estimate(df, :y, :x))
```

# References
- Calonico, S., Cattaneo, M. D., & Titiunik, R. (2014). Robust nonparametric
  confidence intervals for regression-discontinuity designs. *Econometrica*, 82(6),
  2295–2326.
- Cattaneo, M. D., Idrobo, N., & Titiunik, R. (2020). *A Practical Introduction to
  Regression Discontinuity Designs: Foundations*. Cambridge University Press.
- Imbens, G. W., & Lemieux, T. (2008). Regression discontinuity designs: A guide to
  practice. *Journal of Econometrics*, 142(2), 615–635.
"""
function plot_rd_placebos end
"""
    plot_rd_placebos!(ax, tab; estimate=nothing, colors=nothing) -> ax

Draw an RD placebo-cutoff plot into an existing Makie axis. This is the mutating
counterpart of [`plot_rd_placebos`](@ref), which describes the display; no legend or
caption is added.

# Arguments
- `ax::Makie.AbstractAxis`: the axis to draw into.
- `tab`: the table returned by [`rd_placebo_cutoffs`](@ref).
- Keywords: those of [`plot_rd_placebos`](@ref) except `figure` and `axis`.

# Returns
- `ax`, the axis drawn into.
"""
function plot_rd_placebos! end
plot_rd_placebos(args...; kwargs...) = _viz_no_backend(plot_rd_placebos, args)
plot_rd_placebos!(args...; kwargs...) = _viz_no_backend(plot_rd_placebos!, args)

"""
    plot_rd_sensitivity(tab; figure=(;), axis=(;), colors=nothing) -> Makie.Figure

Sensitivity plot for a regression discontinuity estimate: bias-corrected estimates
with robust confidence intervals across bandwidths, or across donut radii.

Two tables are supported. For [`rd_bandwidth_sensitivity`](@ref) the horizontal
axis is the main bandwidth ``h`` (the average of the left and right bandwidths when
they differ), labelled also with its multiple of the data-driven bandwidth, which is
highlighted (multiplier 1). For [`rd_donut`](@ref) it is the donut radius: the
estimate excludes observations with ``|x - c|`` below the radius, as in Barreca,
Guldi, Lindo and Waddell (2011), to check whether the result is driven by units
closest to the cutoff, where sorting or heaping is most likely; radius 0 is
highlighted. A dashed line marks zero and a dotted line the baseline estimate.

Movement across the axis is expected and mixes bias and noise: larger bandwidths
reduce variance but increase smoothing bias, and donut estimates extrapolate further
to the cutoff with fewer observations (Calonico, Cattaneo and Titiunik 2014;
Cattaneo, Idrobo and Titiunik 2020). The intervals are pointwise and the estimates
share observations, so they are strongly correlated; overlapping intervals do not
test stability, and a single interval excluding zero at one bandwidth is not
evidence of an effect. The informative pattern is a systematic drift of the point
estimates, for example with bandwidth, that is large relative to the baseline
interval.

# Arguments
- `tab`: the table returned by [`rd_bandwidth_sensitivity`](@ref) or
  [`rd_donut`](@ref).

# Keywords
$(_VIZ_COMMON_KW)

# Returns
- `Makie.Figure`.

# Examples
```julia
using DrSnow, CairoMakie, DataFrames, StableRNGs
rng = StableRNG(1)
x = 2 .* rand(rng, 2000) .- 1
df = DataFrame(x=x, y=0.4 .+ 0.8 .* x .+ 0.6 .* (x .>= 0) .+ 0.3 .* randn(rng, 2000))
plot_rd_sensitivity(rd_bandwidth_sensitivity(df, :y, :x))
plot_rd_sensitivity(rd_donut(df, :y, :x; radii=[0, 0.02, 0.05, 0.1]))
```

# References
- Barreca, A. I., Guldi, M., Lindo, J. M., & Waddell, G. R. (2011). Saving babies?
  Revisiting the effect of very low birth weight classification. *Quarterly Journal
  of Economics*, 126(4), 2117–2123.
- Calonico, S., Cattaneo, M. D., & Titiunik, R. (2014). Robust nonparametric
  confidence intervals for regression-discontinuity designs. *Econometrica*, 82(6),
  2295–2326.
- Cattaneo, M. D., Idrobo, N., & Titiunik, R. (2020). *A Practical Introduction to
  Regression Discontinuity Designs: Foundations*. Cambridge University Press.
"""
function plot_rd_sensitivity end
"""
    plot_rd_sensitivity!(ax, tab; colors=nothing) -> ax

Draw an RD bandwidth or donut sensitivity plot into an existing Makie axis. This is
the mutating counterpart of [`plot_rd_sensitivity`](@ref), which describes the
display; no legend or caption is added.

# Arguments
- `ax::Makie.AbstractAxis`: the axis to draw into.
- `tab`: the table returned by [`rd_bandwidth_sensitivity`](@ref) or
  [`rd_donut`](@ref).
- `colors`: as in [`plot_rd_sensitivity`](@ref).

# Returns
- `ax`, the axis drawn into.
"""
function plot_rd_sensitivity! end
plot_rd_sensitivity(args...; kwargs...) = _viz_no_backend(plot_rd_sensitivity, args)
plot_rd_sensitivity!(args...; kwargs...) = _viz_no_backend(plot_rd_sensitivity!, args)

"""
    plot_rd_density(data, running, t::DiagnosticTest; npoints=40, bins=nothing,
                    limits=nothing, level=0.95, ci=true, figure=(;), axis=(;),
                    colors=nothing) -> Makie.Figure
    plot_rd_density(x::AbstractVector, t::DiagnosticTest; kwargs...) -> Makie.Figure

Manipulation-test plot for the running variable of a regression discontinuity design,
in the layout of `rdplotdensity`: a histogram of the running variable with local
polynomial density estimates on each side of the cutoff.

If units can precisely manipulate the running variable to fall on the preferred side
of the cutoff, its density is typically discontinuous there, and the continuity
assumptions behind the design become doubtful (McCrary 2008). The plot shows a
histogram of the running variable, normalized to a density and with the cutoff as a
bin edge (grey bars), and the local polynomial density estimators of Cattaneo,
Jansson and Ma (2020) on each side of the cutoff, with pointwise `level` confidence
bands. The estimation settings (kernel, polynomial order, bandwidths) are taken from
the [`rd_density_test`](@ref) result `t`. Each side's curve uses only that side's
observations, so the gap between the two curves at the cutoff is the estimated
density discontinuity; the caption reports the robust test statistic and p-value.
The test result does not store the data, so pass the same running variable that
produced `t`.

The bands are pointwise, and visual gaps away from the cutoff are not part of the
test. A non-rejection does not show that there was no manipulation: the test has
limited power in small samples, and manipulation in both directions can leave the
density continuous. Heaping at round values of the running variable shows up as
spikes in the histogram and is a separate concern.

# Arguments
- `data`, `running::Symbol`: a table and the running-variable column used in `t`;
  or
- `x::AbstractVector`: the running variable itself.
- `t::DiagnosticTest`: from [`rd_density_test`](@ref).

# Keywords
- `npoints::Integer = 40`: number of grid points per side for the density curves.
- `bins = nothing`: number of histogram bins (default: about the square root of the
  number of observations in the plotted range, between 10 and 40).
- `limits = nothing`: `(lo, hi)` range of the running variable to plot (default:
  two bandwidths on each side of the cutoff).
- `level::Real = 0.95`: confidence level of the pointwise bands.
- `ci::Bool = true`: draw the bands.
$(_VIZ_COMMON_KW)

# Returns
- `Makie.Figure`.

# Examples
```julia
using DrSnow, CairoMakie, DataFrames, StableRNGs
rng = StableRNG(1)
df = DataFrame(x=vcat(2 .* rand(rng, 2000) .- 1, 0.08 .* rand(rng, 150)))  # bunching
t = rd_density_test(df, :x; cutoff=0.0)
plot_rd_density(df, :x, t)
```

# References
- Cattaneo, M. D., Jansson, M., & Ma, X. (2020). Simple local polynomial density
  estimators. *Journal of the American Statistical Association*, 115(531),
  1449–1455.
- McCrary, J. (2008). Manipulation of the running variable in the regression
  discontinuity design: A density test. *Journal of Econometrics*, 142(2), 698–714.
"""
function plot_rd_density end
"""
    plot_rd_density!(ax, data, running, t; npoints=40, bins=nothing, limits=nothing,
                     level=0.95, ci=true, colors=nothing) -> ax
    plot_rd_density!(ax, x, t; kwargs...) -> ax

Draw an RD density (manipulation-test) plot into an existing Makie axis. This is the
mutating counterpart of [`plot_rd_density`](@ref), which describes the display; no
legend or caption is added.

# Arguments
- `ax::Makie.AbstractAxis`: the axis to draw into.
- `data`, `running` (or `x`), `t`: as in [`plot_rd_density`](@ref).
- Keywords: those of [`plot_rd_density`](@ref) except `figure` and `axis`.

# Returns
- `ax`, the axis drawn into.
"""
function plot_rd_density! end
plot_rd_density(args...; kwargs...) = _viz_no_backend(plot_rd_density, args)
plot_rd_density!(args...; kwargs...) = _viz_no_backend(plot_rd_density!, args)

"""
    plot_honest_did(r::HonestDiDResult; breakdown=nothing, figure=(;), axis=(;),
                    colors=nothing) -> Makie.Figure

Sensitivity plot of a Rambachan–Roth ("Honest DiD") analysis, in the layout of the
HonestDiD package's `createSensitivityPlot`: robust confidence sets for a
post-treatment effect as the allowed violation of parallel trends grows.

Rambachan and Roth (2023) replace exact parallel trends by the restriction that the
post-treatment difference in trends ``\\delta_{post}`` lies in a set ``\\Delta(M)``
disciplined by the pre-treatment coefficients, and report confidence sets for the
target ``\\theta`` (a post-treatment effect or an average of several) that are valid
under that restriction. Under *relative magnitudes*, ``\\Delta^{RM}(\\bar M)``,
each post-treatment change in the trend difference is at most ``\\bar M`` times the
largest pre-treatment change; under *smoothness*, ``\\Delta^{SD}(M)``, the slope of
the trend difference may change by at most ``M`` per period (``M = 0`` allows only
linear trends). The plot shows, from left to right, the original confidence interval
(labelled "Orig.", valid only under exact parallel trends) and the robust confidence
set at each value of ``M`` or ``\\bar M`` in `r` as vertical intervals, with a
dashed line at zero.

The quantity of interest is the *breakdown value*: the largest ``M`` at which the
robust set still excludes zero, equivalently where the sets start to include it,
marked by a dotted vertical line. It is `breakdown` when given (for example the
continuous value from [`honest_breakdown`](@ref)) and otherwise the smallest grid
value in `r` whose set contains zero, which is coarser. A result that survives a
breakdown of ``\\bar M = 1`` remains significant if post-treatment violations are as
large as the worst pre-treatment one; whether that is plausible is a substantive
judgement about the application, not a statistical one. Values of ``M`` at which the
robust set is empty, meaning that the data reject the restriction itself (the
pre-treatment coefficients are inconsistent with ``\\Delta(M)``), are marked with a
cross on the axis. The sets are pointwise in ``M``.

# Arguments
- `r::HonestDiDResult`: from [`honest_did`](@ref).

# Keywords
- `breakdown::Union{Nothing,Real} = nothing`: a breakdown value to mark instead of
  the grid value stored in `r`.
$(_VIZ_COMMON_KW)

# Returns
- `Makie.Figure`. The subtitle states the restriction, the inference method and how
  the breakdown value was obtained.

# Examples
```julia
using DrSnow, CairoMakie, DataFrames, StableRNGs
rng = StableRNG(1)
g = rand(rng, [0, 5, 7], 240)
df = DataFrame(unit=repeat(1:240; inner=10), year=repeat(1:10, 240))
df.d = Int.((g[df.unit] .> 0) .& (df.year .>= g[df.unit]))
df.y = randn(rng, 240)[df.unit] .+ 0.2 .* df.year .+ df.d .+ randn(rng, nrow(df))
es = event_study(df, :y, :d, :unit, :year; estimator=:twfe, max_pre=3, max_post=2,
                 endpoints=:trim)
r = honest_did(es; restriction=:relative_magnitudes, M=0:0.25:1.5, rng=StableRNG(2))
plot_honest_did(r; breakdown=honest_breakdown(es; restriction=:relative_magnitudes,
                                              rng=StableRNG(3)))
```

# References
- Rambachan, A., & Roth, J. (2023). A more credible approach to parallel trends.
  *Review of Economic Studies*, 90(5), 2555–2591.
- Roth, J. (2022). Pretest with caution: Event-study estimates after testing for
  parallel trends. *American Economic Review: Insights*, 4(3), 305–322.
"""
function plot_honest_did end
"""
    plot_honest_did!(ax, r::HonestDiDResult; breakdown=nothing, colors=nothing) -> ax

Draw an Honest DiD sensitivity plot into an existing Makie axis. This is the mutating
counterpart of [`plot_honest_did`](@ref), which describes the display and the
breakdown value; no legend or subtitle is added.

# Arguments
- `ax::Makie.AbstractAxis`: the axis to draw into.
- `r::HonestDiDResult`: from [`honest_did`](@ref).
- Keywords: those of [`plot_honest_did`](@ref) except `figure` and `axis`.

# Returns
- `ax`, the axis drawn into.
"""
function plot_honest_did! end
plot_honest_did(args...; kwargs...) = _viz_no_backend(plot_honest_did, args)
plot_honest_did!(args...; kwargs...) = _viz_no_backend(plot_honest_did!, args)

"""
    plot_judge_first_stage(data, r::JudgeIVEstimate; nbins=20, trim=0.01,
                           level=0.95, histogram=true, figure=(;), axis=(;),
                           colors=nothing) -> Makie.Figure
    plot_judge_first_stage(data, treatment, leniency; kwargs...) -> Makie.Figure

First-stage plot of a judge (examiner) design, in the layout of Dobbie, Goldin and
Yang (2018, Figure 1): the treatment rate against the leave-one-out leniency of the
assigned decision-maker.

In a judge design, cases are (conditionally) randomly assigned to decision-makers who
differ in their propensity to treat, and each case's instrument is the leniency of its
judge computed without that case (Kling 2006; Dobbie, Goldin and Yang 2018). The top
panel shows binned treatment rates (equal-count bins of leniency, with pointwise
`level` intervals for each bin's rate) and the linear fit over all cases; its slope,
with a heteroskedasticity-robust standard error, is reported in the caption. With
`histogram = true` a lower panel, sharing the leniency axis, shows the distribution
of leniency as the share of cases. The bins and histogram drop the `trim` tails of
leniency to keep extreme judges from dominating the display; the fit and the slope
use every case.

A steep, roughly linear relation shows a strong first stage and that leniency
shifts treatment over its whole support; the histogram shows how much variation
there is and where the instrument has little support. The plot does not test the
exclusion restriction or monotonicity, the assumptions that give the IV estimand its
LATE interpretation; average monotonicity can be probed by the first stage within
subgroups of cases, and Frandsen, Lefgren and Leslie (2023) propose a joint test of
exclusion and monotonicity.

# Arguments
- `data`: the case-level data used to build the instrument (rows aligned with it).
- `r::JudgeIVEstimate`: from [`judge_iv`](@ref); its treatment column and stored
  leniency are used. Alternatively:
- `treatment::Symbol` and `leniency`: the treatment column and the leniency, either
  a vector with one entry per row of `data` (e.g. from [`judge_leniency`](@ref)) or
  a column name.

# Keywords
- `nbins::Integer = 20`: number of equal-count leniency bins.
- `trim::Real = 0.01`: share of cases trimmed from each tail of leniency for the bins
  and histogram.
- `level::Real = 0.95`: confidence level of the bin-rate intervals.
- `histogram::Bool = true`: add the leniency histogram panel.
$(_VIZ_COMMON_KW)

# Returns
- `Makie.Figure` with one or two panels.

# Examples
```julia
using DrSnow, CairoMakie, DataFrames, StableRNGs
rng = StableRNG(1)
λ = 0.2 .+ 0.5 .* rand(rng, 40)                      # judge leniency
df = DataFrame(judge=repeat(1:40; inner=40))
df.d = Float64.(rand(rng, nrow(df)) .< λ[df.judge])
df.y = df.d .+ randn(rng, nrow(df))
r = judge_iv(df, :y, :d, :judge)
plot_judge_first_stage(df, r)
```

# References
- Dobbie, W., Goldin, J., & Yang, C. S. (2018). The effects of pretrial detention on
  conviction, future crime, and employment: Evidence from randomly assigned judges.
  *American Economic Review*, 108(2), 201–240.
- Frandsen, B., Lefgren, L., & Leslie, E. (2023). Judging judge fixed effects.
  *American Economic Review*, 113(1), 253–277.
- Kling, J. R. (2006). Incarceration length, employment, and earnings. *American
  Economic Review*, 96(3), 863–876.
"""
function plot_judge_first_stage end
"""
    plot_judge_first_stage!(ax, data, r_or_treatment, [leniency]; nbins=20, trim=0.01,
                            level=0.95, colors=nothing) -> ax

Draw the binned first stage of a judge design into an existing Makie axis, without
the leniency histogram panel (which needs a figure). This is the mutating
counterpart of [`plot_judge_first_stage`](@ref), which describes the display; no
legend or caption is added.

# Arguments
- `ax::Makie.AbstractAxis`: the axis to draw into.
- `data`, `r_or_treatment`, `leniency`: as in [`plot_judge_first_stage`](@ref).
- Keywords: those of [`plot_judge_first_stage`](@ref) except `histogram`, `figure`
  and `axis`.

# Returns
- `ax`, the axis drawn into.
"""
function plot_judge_first_stage! end
plot_judge_first_stage(args...; kwargs...) = _viz_no_backend(plot_judge_first_stage, args)
plot_judge_first_stage!(args...; kwargs...) =
    _viz_no_backend(plot_judge_first_stage!, args)

"""
    plot_rotemberg(r::RotembergDecomposition; x=:first_stage_F, label=5, figure=(;),
                   axis=(;), colors=nothing) -> Makie.Figure

Rotemberg-weight plot for a shift-share (Bartik) instrument, in the layout of
Goldsmith-Pinkham, Sorkin and Swift (2020, Figure 2): which industry shares drive the
Bartik estimate, and whether their just-identified estimates agree.

Goldsmith-Pinkham, Sorkin and Swift (2020) show that the Bartik IV estimate equals
``\\sum_k \\hat\\alpha_k \\hat\\beta_k``, a weighted sum of the just-identified IV
estimates ``\\hat\\beta_k`` that use each share ``k`` as the sole instrument, with
Rotemberg weights ``\\hat\\alpha_k`` that sum to one but can be negative. Each point is
one share: its ``\\hat\\beta_k`` against its first-stage F statistic
(`x = :first_stage_F`, log scale) or its shock (`x = :shock`). Marker area is
proportional to ``|\\hat\\alpha_k|``, filled circles mark positive and hollow
triangles negative weights, the `label` largest weights are labelled, and a dashed
line marks the Bartik estimate. Shares with a zero first stage (no ``\\hat\\beta_k``)
are omitted.

The plot shows where identification comes from. When a few shares carry most of the
weight, the design is effectively a comparison of places by those shares, and their
exogeneity deserves scrutiny; widely dispersed ``\\hat\\beta_k`` among high-weight
shares indicate heterogeneous effects or violations of exogeneity for some shares;
and large negative weights mean the estimate need not be a convex average of
share-specific effects. High-weight shares with weak first stages yield noisy
``\\hat\\beta_k``. When identification is argued from the shocks rather than the
shares (Borusyak, Hull and Jaravel 2022), the shock-level view (`x = :shock`) is the
relevant one.

# Arguments
- `r::RotembergDecomposition`: from [`rotemberg_weights`](@ref).

# Keywords
- `x::Symbol = :first_stage_F`: horizontal axis, `:first_stage_F` (log scale) or
  `:shock`.
- `label::Integer = 5`: number of largest-weight shares to label.
$(_VIZ_COMMON_KW)

# Returns
- `Makie.Figure`.

# Examples
```julia
using DrSnow, CairoMakie, DataFrames, StableRNGs
rng = StableRNG(1)
n, K = 300, 12
S = rand(rng, n, K) .^ 3
S ./= sum(S; dims=2)                                   # industry shares
g = randn(rng, K)                                      # national shocks
df = DataFrame(S, [Symbol("s", k) for k in 1:K])
df.d = 0.8 .* (S * g) .+ randn(rng, n)
df.y = df.d .+ randn(rng, n)
plot_rotemberg(rotemberg_weights(df, :y, :d, [Symbol("s", k) for k in 1:K], g))
```

# References
- Borusyak, K., Hull, P., & Jaravel, X. (2022). Quasi-experimental shift-share
  research designs. *Review of Economic Studies*, 89(1), 181–213.
- Goldsmith-Pinkham, P., Sorkin, I., & Swift, H. (2020). Bartik instruments: What,
  when, why, and how. *American Economic Review*, 110(8), 2586–2624.
"""
function plot_rotemberg end
"""
    plot_rotemberg!(ax, r::RotembergDecomposition; x=:first_stage_F, label=5,
                    colors=nothing) -> ax

Draw a Rotemberg-weight plot into an existing Makie axis. This is the mutating
counterpart of [`plot_rotemberg`](@ref), which describes the display; no legend or
caption is added.

# Arguments
- `ax::Makie.AbstractAxis`: the axis to draw into.
- `r::RotembergDecomposition`: from [`rotemberg_weights`](@ref).
- Keywords: those of [`plot_rotemberg`](@ref) except `figure` and `axis`.

# Returns
- `ax`, the axis drawn into.
"""
function plot_rotemberg! end
plot_rotemberg(args...; kwargs...) = _viz_no_backend(plot_rotemberg, args)
plot_rotemberg!(args...; kwargs...) = _viz_no_backend(plot_rotemberg!, args)

"""
    plot_mte(r::MTEEstimate; parameters=true, propensity=true, figure=(;), axis=(;),
             colors=nothing) -> Makie.Figure

Marginal treatment effect (MTE) curve with its pointwise confidence band, the common
support of the propensity score, and the treatment-effect parameters it implies.

In the generalized Roy model the MTE is the average effect for units at a given
margin of indifference, ``\\mathrm{MTE}(x, u) = E[Y(1) - Y(0) \\mid X = x, U_D = u]``,
where ``U_D`` is the unobserved resistance to treatment normalized to be uniform
(Heckman and Vytlacil 2005). Units with low ``u`` take treatment even at low
propensity scores. The curve plots the MTE from [`mte`](@ref) against ``u`` at the
covariate means ``\\bar x``, with a pointwise bootstrap confidence band at the level
of the estimate. A curve that declines in ``u`` indicates selection on gains: those
most eager to be treated benefit most (Carneiro, Heckman and Vytlacil 2011).

The MTE is identified from variation in the propensity score, so only on its common
support, which is marked; the region outside it is shaded. For the parametric
methods (normal, polynomial) the curve there is an extrapolation from the functional
form, and for the semiparametric method it is not drawn. With
`parameters = true` the conventional parameters, which are weighted averages of the
MTE (ATE, ATT, ATU, LATE and, when requested, a policy-relevant treatment effect),
are drawn as labelled horizontal lines; parameters that need the MTE off the support
(e.g. the ATE when the propensity score does not reach 0 and 1) inherit the
extrapolation. `propensity = true` adds a rug of the estimated propensity scores.
The band is pointwise in ``u``; this plot shows a point-identified curve and does not
draw partial-identification bounds.

# Arguments
- `r::MTEEstimate`: from [`mte`](@ref).

# Keywords
- `parameters::Bool = true`: draw the treatment-effect parameters.
- `propensity::Bool = true`: draw a rug of the estimated propensity scores.
$(_VIZ_COMMON_KW)

# Returns
- `Makie.Figure`.

# Examples
```julia
using DrSnow, CairoMakie, DataFrames, StableRNGs
rng = StableRNG(1)
n = 2000
df = DataFrame(z=randn(rng, n), x=randn(rng, n))
v = randn(rng, n)
df.d = Float64.(0.9 .* df.z .+ 0.3 .* df.x .- v .> 0)
ud = 0.5 .* (1 .+ tanh.(0.8 .* v))                    # resistance, increasing in v
df.y = 0.5 .* df.x .+ df.d .* (1.5 .- 2.0 .* ud) .+ randn(rng, n)
r = mte(df, :y, :d, [:z]; covariates=[:x], method=:polynomial, degree=2,
        n_bootstrap=50, rng=StableRNG(2))
plot_mte(r)
```

# References
- Carneiro, P., Heckman, J. J., & Vytlacil, E. J. (2011). Estimating marginal
  returns to education. *American Economic Review*, 101(6), 2754–2781.
- Heckman, J. J., & Vytlacil, E. (2005). Structural equations, treatment effects,
  and econometric policy evaluation. *Econometrica*, 73(3), 669–738.
"""
function plot_mte end
"""
    plot_mte!(ax, r::MTEEstimate; parameters=true, propensity=true,
              colors=nothing) -> ax

Draw an MTE curve into an existing Makie axis. This is the mutating counterpart of
[`plot_mte`](@ref), which describes the display; no legend or caption is added.

# Arguments
- `ax::Makie.AbstractAxis`: the axis to draw into.
- `r::MTEEstimate`: from [`mte`](@ref).
- Keywords: those of [`plot_mte`](@ref) except `figure` and `axis`.

# Returns
- `ax`, the axis drawn into.
"""
function plot_mte! end
plot_mte(args...; kwargs...) = _viz_no_backend(plot_mte, args)
plot_mte!(args...; kwargs...) = _viz_no_backend(plot_mte!, args)

"""
    plot_synth_in_time(r::SyntheticControlEstimate, backdated; figure=(;), axis=(;),
                       colors=nothing) -> Makie.Figure

In-time placebo ("backdating") plot for a synthetic control, in the layout of
Abadie, Diamond and Hainmueller (2015, Figure 4).

The synthetic control is refitted as if treatment had started at an earlier, placebo
date ``t_0`` ([`synth_in_time_placebo`](@ref)`(r, t₀)`), using only data before
``t_0``; its donor weights are then applied to the donors' outcomes in every period.
The plot shows the treated unit's outcome, the synthetic control of the actual
design, and the backdated synthetic control, with dashed lines at the placebo and
the actual treatment dates. Between the two dates the backdated counterfactual is an
out-of-sample prediction of an untreated period.

A backdated counterfactual that tracks the treated unit between the two dates and
diverges only after the actual date is consistent with the original estimate: the
method predicts untreated outcomes well out of sample, and the effect does not
appear before treatment (Abadie, Diamond and Hainmueller 2015; Abadie 2021). A gap
opening before the actual date points to anticipation, a poor counterfactual, or an
earlier shock; agreement does not prove the original estimate, since the placebo
period may differ from the post-treatment period.

# Arguments
- `r::SyntheticControlEstimate`: the synthetic control of the actual design.
- `backdated::SyntheticControlEstimate`: from [`synth_in_time_placebo`](@ref)`(r, t₀)`.

# Keywords
$(_VIZ_COMMON_KW)

# Returns
- `Makie.Figure`.

# Examples
```julia
using DrSnow, CairoMakie, DataFrames, StableRNGs
rng = StableRNG(1)
N, T, T0 = 16, 18, 12
f = cumsum(randn(rng, T)) .+ range(0, 4; length=T)
load, α = 0.5 .+ rand(rng, N), 10 .+ 2 .* randn(rng, N)
df = DataFrame(unit=repeat(1:N; inner=T), year=repeat(1991:(1990 + T), N))
df.d = Int.((df.unit .== N) .& (df.year .> 1990 + T0))
df.y = α[df.unit] .+ load[df.unit] .* f[df.year .- 1990] .- 3.0 .* df.d .+
       0.3 .* randn(rng, nrow(df))
r = synthetic_control(df, :y, :d, :unit, :year)
plot_synth_in_time(r, synth_in_time_placebo(r, 1998))
```

# References
- Abadie, A. (2021). Using synthetic controls: Feasibility, data requirements, and
  methodological aspects. *Journal of Economic Literature*, 59(2), 391–425.
- Abadie, A., Diamond, A., & Hainmueller, J. (2015). Comparative politics and the
  synthetic control method. *American Journal of Political Science*, 59(2), 495–510.
"""
function plot_synth_in_time end
"""
    plot_synth_in_time!(ax, r, backdated; colors=nothing) -> ax

Draw an in-time placebo (backdating) plot into an existing Makie axis. This is the
mutating counterpart of [`plot_synth_in_time`](@ref), which describes the display;
no legend or caption is added.

# Arguments
- `ax::Makie.AbstractAxis`: the axis to draw into.
- `r`, `backdated`: as in [`plot_synth_in_time`](@ref).
- `colors`: as in [`plot_synth_in_time`](@ref).

# Returns
- `ax`, the axis drawn into.
"""
function plot_synth_in_time! end
plot_synth_in_time(args...; kwargs...) = _viz_no_backend(plot_synth_in_time, args)
plot_synth_in_time!(args...; kwargs...) = _viz_no_backend(plot_synth_in_time!, args)

"""
    drsnow_theme(; fontsize=14, font=:sans, colors=nothing) -> Makie.Theme

A Makie theme reproducing the style of DrSnow's plots, for users' own figures and
for publication settings.

The style follows established guidance on statistical graphics. Estimates are shown
as positions along common scales, the most accurately decoded visual encoding
(Cleveland and McGill 1984), with light horizontal grid lines only, no top or right
spines, left-aligned titles and frameless legends, so that ink is spent on data
rather than decoration. The categorical palette has eight colours in a fixed order
(never cycled), chosen to remain distinguishable for readers with common forms of
colour-vision deficiency, and series identity is always carried redundantly by
marker shape, line style or a direct label, so figures also survive greyscale
printing (Wong 2011). In the theme the palette is also used for the line and scatter
cycles, with matching marker shapes.

DrSnow's plotting functions always apply their own axis styling, but take fonts,
font size and figure-level attributes from the active Makie theme, so

```julia
with_theme(drsnow_theme(; font=:serif, fontsize=10)) do
    fig = plot_event_study(es; figure=(size=(500, 320),))
    save("event_study.pdf", fig)
end
```

produces a serif figure sized for a single journal column, and
`set_theme!(drsnow_theme())` applies the theme for the whole session. For print,
save vector formats (PDF, SVG) or rasters with `px_per_unit ≥ 3`.

# Keywords
- `fontsize::Real = 14`: base font size in points; titles, labels and tick labels are
  scaled from it.
- `font::Symbol = :sans`: `:sans` (TeX Gyre Heros, Makie's default) or `:serif`
  (New Computer Modern, matching LaTeX documents).
- `colors = nothing`: a vector of colours replacing the default palette (for the
  theme's cycles only; DrSnow's plotting functions take their own `colors`
  keyword).

# Returns
- `Makie.Theme` (requires a Makie backend to be loaded).

# Examples
```julia
using DrSnow, CairoMakie, DataFrames, StableRNGs
rng = StableRNG(1)
g = rand(rng, [0, 4, 6], 100)
df = DataFrame(unit=repeat(1:100; inner=8), year=repeat(1:8, 100))
df.d = Int.((g[df.unit] .> 0) .& (df.year .>= g[df.unit]))
df.y = 0.2 .* df.year .+ df.d .+ randn(rng, nrow(df))
fig = with_theme(drsnow_theme(; font=:serif, fontsize=10)) do
    plot_trends(df, :y, :d, :unit, :year; figure=(size=(500, 320),))
end
# save("trends.pdf", fig)                     # vector output
# save("trends.png", fig; px_per_unit=3)      # high-resolution raster
```

# References
- Cleveland, W. S., & McGill, R. (1984). Graphical perception: Theory,
  experimentation, and application to the development of graphical methods.
  *Journal of the American Statistical Association*, 79(387), 531–554.
- Wong, B. (2011). Points of view: Color blindness. *Nature Methods*, 8(6), 441.
"""
function drsnow_theme end
drsnow_theme(args...; kwargs...) = _viz_no_backend(drsnow_theme, args)
