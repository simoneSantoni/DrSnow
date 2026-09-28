# DrSnow.jl

```@meta
CurrentModule = DrSnow
```

DrSnow is a Julia package for design-based causal inference with natural
experiments: difference-in-differences, instrumental variables, regression
discontinuity, synthetic control, randomization inference, interference
(spillovers) and double/debiased machine learning. The name recalls John Snow's
1854 study of cholera in London, an early natural experiment.

The package aims to make the identifying assumptions explicit and the inference
correct:

- Every estimator returns a [`CausalEstimate`](@ref) with its full covariance matrix,
  so `coef`, `vcov`, `stderror`, `confint(r; level)`, `coeftable`, [`tidy`](@ref) and
  [`glance`](@ref) work the same way everywhere, and every result states its estimand.
- Diagnostics return a [`DiagnosticTest`](@ref). A non-rejection is never reported as
  evidence that an assumption holds, and untestable assumptions are not described as
  tested.
- Estimators are validated against reference implementations (mostly R packages) on
  published data, and most inferential procedures have a Monte Carlo size or
  coverage check in the test suite (see [Validation](validation.md)).

DrSnow is pre-1.0 and not registered in the General registry. Version 0.2 is a
rewrite; see the changelog in the repository for breaking changes from 0.1.

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/simoneSantoni/DrSnow_alpha")
```

DrSnow requires Julia 1.10 or later. Optional features load as package extensions
when the corresponding package is loaded next to DrSnow:

| Feature | Load | Documentation |
|:--|:--|:--|
| Plots of event studies, RD, synthetic control, … | a Makie backend, e.g. `using CairoMakie` | [Results, tables and plots](@ref) |
| Regression tables (text, LaTeX, HTML) | `using RegressionTables` | [Results, tables and plots](@ref) |
| Any MLJ model as a nuisance learner | `using MLJ` (regressors need only `MLJModelInterface`) | [Causal machine learning](ml.md) |
| Web interface | `using HTTP, JSON3, CSV` | [Web interface](gui.md) |
| `TreatmentPanels.jl` input | `using TreatmentPanels` | [Synthetic control](synth.md) |

## Which design, which function?

| Design | Main functions | Guide |
|:--|:--|:--|
| Two groups, one adoption date | [`did_twfe`](@ref), [`event_study`](@ref) | [DiD](did.md) |
| Staggered adoption | [`did_callaway_santanna`](@ref), [`did_sun_abraham`](@ref), [`did_imputation`](@ref), [`did_etwfe`](@ref) | [DiD](did.md) |
| Diagnosing a TWFE estimate | [`bacon_decomposition`](@ref), [`twfe_weights`](@ref) | [DiD](did.md) |
| Treatment switching on and off | [`did_multiplegt_dyn`](@ref) | [DiD](did.md) |
| Sensitivity to non-parallel trends | [`pre_trend_test`](@ref), [`honest_did`](@ref) | [DiD](did.md) |
| Instrument, continuous or multi-valued treatment | [`iv_regression`](@ref), [`weak_iv_confidence_set`](@ref) | [IV](iv.md) |
| Binary instrument and treatment (LATE) | [`late_2sls`](@ref), [`estimate_compliance`](@ref), [`complier_characteristics`](@ref) | [IV](iv.md) |
| Many instruments, judge designs, shift-share | [`kclass_iv`](@ref), [`jive`](@ref), [`judge_iv`](@ref), [`shift_share_iv`](@ref) | [IV](iv.md) |
| Cutoff in a running variable | [`rd_estimate`](@ref), [`rd_density_test`](@ref) | [RD](rd.md) |
| Few treated units, long pre-period | [`synthetic_did`](@ref), [`synthetic_control`](@ref), [`augmented_synthetic_control`](@ref) | [Synthetic control](synth.md) |
| Known assignment mechanism | [`randomization_test`](@ref), [`ri_confint`](@ref), [`ri_regression`](@ref) | [Randomization inference](ri.md) |
| Spillovers between units | [`exposure_effects`](@ref), [`spillover_fisher_test`](@ref), [`spillover_did`](@ref), [`conley_vcov`](@ref) | [Interference](sutva.md) |
| High-dimensional controls, heterogeneous effects | [`dml_plr`](@ref), [`dml_irm`](@ref), [`cate_dr_learner`](@ref), [`generic_ml`](@ref) | [Causal ML](ml.md) |

## Reading guide

- New to the package: start with the [Tutorial](tutorial.md), which runs a staggered
  DiD, an IV analysis, a sharp RD and a randomization test end to end.
- Choosing or configuring an estimator: each methods guide explains the estimands,
  assumptions, options and references for its area, and ends with the reference
  documentation of its functions.
- Reporting: [Results, tables and plots](@ref) covers data frames, regression tables
  and figures.
- Point-and-click exploration: the [Web interface](gui.md).
- Checking the numbers: [Validation](validation.md) lists what each area is compared
  against and how to rerun the comparisons.
- Shared types and helpers: the [API Reference](api.md), with an index of every
  documented name.
