# Results, tables and plots

```@meta
CurrentModule = DrSnow
```

Every DrSnow estimator returns a [`CausalEstimate`](@ref) that supports the
StatsAPI accessors (`coef`, `vcov`, `stderror`, `confint(r; level)`, `coeftable`,
`nobs`, `dof_residual`) together with [`estimand`](@ref) and [`method_name`](@ref).
This page shows how to turn results into data frames, regression tables and
publication-quality figures.

| Task | Function | Requires |
|---|---|---|
| One row per coefficient (broom-style) | [`tidy`](@ref) | — |
| One-row model summary | [`glance`](@ref) | — |
| Any Tables.jl sink (`DataFrame(r)`, `CSV.write`) | Tables.jl interface | — |
| Multi-column regression tables (text, LaTeX, HTML) | `regtable` | `using RegressionTables` |
| Event-study plots, several estimators overlaid | [`plot_event_study`](@ref) | a Makie backend |
| Forest plot of any estimates | [`plot_coefficients`](@ref) | a Makie backend |
| RD plot (binned means, fits, bandwidth) | [`plot_rd`](@ref) | a Makie backend |
| Synthetic control trajectories, gaps, placebo spaghetti | [`plot_synth`](@ref) | a Makie backend |
| Randomization / placebo distributions | [`plot_randomization_distribution`](@ref) | a Makie backend |
| Goodman-Bacon decomposition | [`plot_bacon`](@ref) | a Makie backend |
| GATES and CATE predictions | [`plot_gates`](@ref), [`plot_cate`](@ref) | a Makie backend |
| Forest variable importance, RATE / TOC curves | [`plot_variable_importance`](@ref), [`plot_rate`](@ref) | a Makie backend |
| Ring / exposure spillover effects | [`plot_spillover_rings`](@ref) | a Makie backend |
| Weak-IV-robust confidence sets (AR/CLR/LM) | [`plot_confidence_set`](@ref) | a Makie backend |
| Raw outcome trends by cohort / treatment status | [`plot_trends`](@ref) | a Makie backend |
| Covariate balance ("love plots") | [`plot_balance`](@ref) | a Makie backend |
| RD placebo cutoffs, bandwidth / donut sensitivity | [`plot_rd_placebos`](@ref), [`plot_rd_sensitivity`](@ref) | a Makie backend |
| RD manipulation (density) test | [`plot_rd_density`](@ref) | a Makie backend |
| Honest DiD sensitivity (robust CIs vs `M`) | [`plot_honest_did`](@ref) | a Makie backend |
| Judge first stage, Rotemberg weights, MTE curve | [`plot_judge_first_stage`](@ref), [`plot_rotemberg`](@ref), [`plot_mte`](@ref) | a Makie backend |
| Synthetic control backdating (in-time placebo) | [`plot_synth_in_time`](@ref) | a Makie backend |
| Publication theme for DrSnow and your own figures | [`drsnow_theme`](@ref) | a Makie backend |

## Tidy results

[`tidy`](@ref) returns a `DataFrame` with columns `term`, `estimate`, `std_error`,
`statistic`, `p_value`, `conf_low` and `conf_high`. The p-values and intervals are
the same as those of `coeftable`: a t reference with `dof_residual(r)` degrees of
freedom when it is finite (e.g. `G - 1` clusters), the normal distribution
otherwise. Keyword arguments other than `level` go to `confint`, so estimator-specific
intervals are available:

```julia
using DrSnow
es = aggregate_att(did_callaway_santanna(df, :y, :d, :unit, :year), :dynamic)
tidy(es; level=0.90)                 # pointwise 90% intervals
tidy(es; uniform=true)               # simultaneous (sup-t) bands
```

A vector of results is stacked with a `model` column, which is convenient for
comparing estimators or for custom plotting:

```julia
results = [did_twfe(df, :y, :d, :unit, :year),
           event_study_average(did_imputation(df, :y, :d, :unit, :year;
                                              horizons=:all)),
           synthetic_did(df, :y, :d, :unit, :year)]
tidy(results; names=["TWFE", "Imputation", "Synthetic DiD"])
```

Results that carry no variance (for example a synthetic control fitted without
placebos, or `synthetic_did(...; se_method=:none)`) get `missing` inference columns
rather than an error. [`glance`](@ref) gives a one-row summary with `method`,
`estimand`, `nobs`, `n_coef`, `dof_residual` and `n_clusters`. `tidy` also accepts a
[`DiagnosticTest`](@ref) (one row with the statistic, degrees of freedom, p-value and
method).

Every `CausalEstimate` and `DiagnosticTest` is a Tables.jl table with the `tidy`
columns, so `DataFrame(r)`, `CSV.write("estimates.csv", r)` or
`Tables.columntable(r)` work directly.

## Regression tables

Loading [RegressionTables.jl](https://github.com/jmboehm/RegressionTables.jl)
activates the `DrSnowRegressionTablesExt` extension, and `regtable` accepts DrSnow
results, alone or mixed with FixedEffectModels / GLM fits:

```julia
using DrSnow, RegressionTables
r_did  = did_twfe(panel, :y, :d, :unit, :year)
r_iv   = iv_regression(df, :y, :d, :z; covariates=[:x])
r_rd   = rd_estimate(rdd, :y, :x)
r_dml  = dml_plr(df, :y, :d; covariates=[:x1, :x2])
r_sdid = synthetic_did(panel, :y, :d, :unit, :year)

regtable(r_did, r_iv, r_rd, r_dml, r_sdid)                         # text
regtable(r_did, r_iv, r_rd, r_dml, r_sdid; render=LatexTable())    # LaTeX
regtable(r_did, r_sdid; below_statistic=ConfInt)                   # intervals
```

Each column shows the coefficients with standard errors, significance stars based on
DrSnow's own p-values (so t(`G - 1`) references and normal-based estimators are
respected), the estimator (`method_name`) and the number of observations; IV columns
also report the Kleibergen–Paap first-stage F. The dependent-variable header is the
outcome column when the result records it, and a short form of the estimand
otherwise. A DrSnow result must appear among the first three columns (put it first
when mixing with many foreign models). Results without a variance cannot be
tabulated.

## Plotting

The plotting functions are provided by the `DrSnowMakieExt` package extension, which
Julia loads automatically together with any
[Makie](https://docs.makie.org) backend: `using CairoMakie` for files (PNG, SVG,
PDF), `using GLMakie` or `using WGLMakie` for interactive windows. Without a backend
the functions throw an error explaining this.

Each plot comes in two forms:

- `plot_x(result; kwargs...)` builds and returns a styled `Makie.Figure` with axis
  labels, a caption describing the intervals and, when there are several series, a
  legend. The axis is `fig.content[1]`; customize it through the `axis` keyword
  (a `NamedTuple` of `Makie.Axis` attributes, e.g. `axis = (title = "Effect on
  earnings", ylabel = "log points")`) and the figure through `figure`
  (e.g. `figure = (size = (900, 500),)`).
- `plot_x!(ax, result; kwargs...)` draws into an existing `Makie.Axis` and returns
  it, for multi-panel figures.

The default colors are a fixed-order, colorblind-safe categorical palette; series are
also distinguished by marker shape or line style and labelled in a legend, so no
information is carried by color alone. Pass `colors` to override them.

```julia
using DrSnow, CairoMakie

# Event studies: pointwise CIs (whiskers) and sup-t bands (shaded), several
# estimators of the same design overlaid
es_twfe = event_study(df, :y, :d, :unit, :year; estimator=:twfe)
es_sa   = did_sun_abraham(df, :y, :d, :unit, :year)
es_cs   = aggregate_att(did_callaway_santanna(df, :y, :d, :unit, :year), :dynamic)
es_bjs  = did_imputation(df, :y, :d, :unit, :year; horizons=:all, pretrends=3)
fig = plot_event_study(es_twfe, es_sa, es_cs, es_bjs;
                       labels=["TWFE", "Sun–Abraham", "Callaway–Sant'Anna",
                               "Imputation"], uniform=true)
save("event_study.png", fig)

# Compare headline estimates across designs
plot_coefficients([r_did, r_sdid, r_dml]; labels=["TWFE", "SDID", "DML"])

# RD plot with the local polynomial fits used for estimation
plot_rd(rd_plot_data(rdd, :y, :x); estimate=rd_estimate(rdd, :y, :x), ci=true)

# Synthetic control: trajectories on top, gaps with in-space placebos below
sc = synthetic_control(panel, :y, :d, :unit, :year; placebo=true)
plot_synth(sc)
plot_synth(synthetic_did(panel, :y, :d, :unit, :year))   # with time weights λ

# Randomization inference and placebo distributions
plot_randomization_distribution(randomization_test(expt, :y, :d))
plot_randomization_distribution(synth_in_space_placebo(sc))

# Staggered-adoption diagnostics, heterogeneity, spillovers, weak IV
plot_bacon(bacon_decomposition(df, :y, :d, :unit, :year))
plot_gates(generic_ml(expt, :y, :d; covariates=[:x1, :x2]))
plot_cate(cate_dr_learner(expt, :y, :d; covariates=[:x1, :x2]); modifier=:x1)
plot_spillover_rings(spillover_event_study(panel, :y, :d, s; unit=:id, time=:t,
                                           exposure=RingExposure([5.0, 10.0])))
plot_confidence_set(weak_iv_confidence_set(r_iv; method=:ar); wald=r_iv)

# Multi-panel figure with the mutating forms
fig = Figure(size=(1000, 420))
plot_event_study!(Axis(fig[1, 1]; title="Callaway–Sant'Anna"), es_cs)
plot_event_study!(Axis(fig[1, 2]; title="Imputation"), es_bjs)
```

A complete script producing figures for several areas is
`examples/plotting_demo.jl`.

### Design diagnostics

The diagnostic plots take the outputs of DrSnow's falsification and sensitivity
functions (or the data, for raw trends) and draw them in the same style:

```julia
# DiD: raw means by adoption cohort, adjusted for covariate composition
plot_trends(panel, :y, :d, :unit, :year; covariates=[:income])
plot_trends(panel, :y, treatment_timing(panel, :d, :unit, :year); by=:treated)

# Covariate balance (love plots) for panels, experiments and RD designs
plot_balance(pretreatment_balance(panel, :d, :unit, :year; covariates=[:x1, :x2]))
plot_balance(ri_balance_test(expt, :d, [:age, :income]))
plot_balance(rd_covariate_balance(rdd, [:z1, :z2], :x); data=rdd)  # SD units

# RD falsification
plot_rd_placebos(rd_placebo_cutoffs(rdd, :y, :x); estimate=rd_estimate(rdd, :y, :x))
plot_rd_sensitivity(rd_bandwidth_sensitivity(rdd, :y, :x))
plot_rd_sensitivity(rd_donut(rdd, :y, :x; radii=[0, 0.02, 0.05]))
plot_rd_density(rdd, :x, rd_density_test(rdd, :x))

# Honest DiD: robust confidence sets as parallel trends is relaxed
es = event_study(panel, :y, :d, :unit, :year; estimator=:twfe, endpoints=:trim)
plot_honest_did(honest_did(es; M=0:0.25:2); breakdown=honest_breakdown(es))

# IV designs
plot_judge_first_stage(cases, judge_iv(cases, :y, :detained, :judge))
plot_rotemberg(rotemberg_weights(df, :y, :d, shares, shocks))
plot_mte(mte(df, :y, :d, [:z]; covariates=[:x], method=:polynomial))

# Synthetic control: backdated counterfactual
sc = synthetic_control(panel, :y, :d, :state, :year)
plot_synth_in_time(sc, synth_in_time_placebo(sc, 1980))
```

- **Trends.** Means per period with confidence bands of the means; dashed lines at
  each cohort's adoption date. Covariate adjustment uses the within group × period
  slope, so it does not remove differences in trends. Similar pre-period paths are
  neither necessary nor sufficient for parallel trends.
- **Balance.** Standardized differences for `pretreatment_balance` (descriptive;
  dotted lines at ±0.1 are a convention, not a test) and `ri_balance_test` (with
  randomization and Westfall–Young p-values); bias-corrected discontinuities with
  robust intervals and Holm-adjusted p-values for `rd_covariate_balance`.
- **RD density.** The curves are the local polynomial density estimators of the test
  (same kernel, order and bandwidths), fitted on each side separately; at the
  cutoff they equal the test's side estimates. `rd_density_test` does not store the
  data, so pass the running variable again.
- **Honest DiD.** The breakdown line uses the value passed as `breakdown` (the
  continuous search of `honest_breakdown`) or else the grid value in the result.
- **Judge first stage.** The binned rates and histogram trim the tails of leniency;
  the linear fit and its slope use all cases. The histogram is a separate panel
  sharing the leniency axis (no second y-axis).
- **MTE.** The point-identified curve from `mte` with its bootstrap band; the
  shaded region lies outside the common support of the propensity score.

### Themes and export

[`drsnow_theme`](@ref) returns a Makie theme with DrSnow's styling (colorblind-safe
fixed-order palette with distinct markers, light horizontal grid, no top/right
spines). DrSnow's plots keep their own axis styling but take fonts, font size and
figure-level settings from the active theme, so a theme also sets the typography of
DrSnow figures:

```julia
using DrSnow, CairoMakie

# One figure, serif fonts at 10 pt, sized for a journal column (units: points)
fig = with_theme(drsnow_theme(; font=:serif, fontsize=10)) do
    plot_trends(panel, :y, :d, :unit, :year; figure=(size=(500, 320),),
                axis=(title="Employment by adoption cohort",))
end
save("trends.pdf", fig)                    # vector (PDF/SVG): sizes in points
save("trends.png", fig; px_per_unit=3)     # raster at 3 pixels per point

# Whole session, including your own Makie plots
set_theme!(drsnow_theme())
```

Every non-mutating plot function accepts `figure` (attributes of `Makie.Figure`,
e.g. `size`) and `axis` (attributes of the main `Makie.Axis`, e.g. `title`,
`xlabel`, `limits`), and every `plot_x!` form draws into an axis you created, for
multi-panel figures. Use `colors` to override the palette of a single plot.

## Benchmarks

`benchmark/benchmarks.jl` defines a BenchmarkTools `SUITE` (usable with
PkgBenchmark) timing representative estimators of every area on small and medium
simulated problems; `julia --project=benchmark benchmark/run.jl` prints a table.
`benchmark/README.md` has instructions and reference timings, and the optional
`Benchmarks` GitHub workflow runs the suite on demand or on pull requests labelled
`run benchmarks`.

### What the plots show

- **Event studies.** Reference (normalized) periods are hollow markers at zero; a
  dashed line separates pre- from post-treatment periods. Binned endpoint
  coefficients are labelled `≤k` / `≥k`. Uniform bands use the stored multiplier
  bootstrap draws when the estimator provides them (Callaway–Sant'Anna) and are
  simulated from the estimated covariance otherwise (pass `rng` for reproducibility).
  Visual inspection of pre-period coefficients is not a test of parallel trends; see
  [`pre_trend_test`](@ref) and its caveats.
- **RD.** The global polynomial is a visual summary of the data; the estimate is
  based on the local polynomial fits within the bandwidth (dashed, shaded window).
- **Synthetic control.** For synthetic DiD the synthetic path includes the
  time-weighted pre-period level adjustment, and the pre-period weights `λ` are drawn
  as bars along the bottom. The placebo spaghetti can be restricted to donors with a
  good pre-treatment fit with `placebo_cutoff` (a multiple of the treated unit's
  pre-treatment RMSPE).
- **Randomization distributions.** Two-sided tests compare `|T|`, so both `T_obs` and
  `-T_obs` are marked.
- **CATE.** Individual DR-learner predictions are noisy; inference on heterogeneity
  should use GATES ([`plot_gates`](@ref)) or [`cate_projection`](@ref).

## Adding plotting support for a new result type

The viz area has a backend-independent data layer (`src/viz/data.jl`, internal
functions `_viz_*_data` returning DataFrames / NamedTuples, unit-tested without
Makie) and the Makie drawing code (`ext/DrSnowMakieExt/`).

1. **Nothing to do** if the new estimator returns an existing type: every
   `CausalEstimate` works with [`plot_coefficients`](@ref), [`tidy`](@ref),
   [`glance`](@ref) and `regtable`, and any estimator returning an
   [`EventStudyEstimate`](@ref) (e.g. a new staggered-DiD estimator) works with
   [`plot_event_study`](@ref), including overlays with the existing estimators.
2. **Same shape as an existing plot**: add a method of the matching hook in
   `src/viz/data.jl`, e.g. `_viz_event_study_data(r::MyType; level, uniform, rng)`
   returning the documented columns (`rel_period`, `label`, `estimate`, `conf_low`,
   `conf_high`, `uniform_low`, `uniform_high`, `reference`, `group`), or
   `_viz_randomization_data(r::MyType; hypothesis)`. The plotting function then
   accepts the new type. If the type stores its cluster count outside an
   `n_clusters` field, add a `_glance_nclusters(r::MyType)` method there too.
3. **A new kind of plot**: add a documented stub pair (`plot_foo`, `plot_foo!`) with a
   `_viz_no_backend` fallback and its export to `src/viz/viz.jl`, a `_viz_foo_data`
   extractor to `src/viz/data.jl`, the drawing code in a new
   `ext/DrSnowMakieExt/foo.jl` (included from `DrSnowMakieExt.jl`, using the shared
   helpers `_mk_figure`, `_mk_axis`, `_mk_colors`, `_mk_legend!`, `_mk_caption`),
   tests in `test/viz/`, and the new functions in the `@docs` block below.

The functions and types described on this page are documented in the [API reference](reference/results.md).
