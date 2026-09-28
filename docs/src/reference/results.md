# Results and Plotting: API

```@meta
CurrentModule = DrSnow
```

Reference documentation for the functions and types described in [Results, tables and plots](../results.md).

Every plotting function below requires a Makie backend (`using CairoMakie` for files,
`using GLMakie` or `using WGLMakie` for interactive use) and has a mutating `!` variant
that draws into an existing axis. Unless a docstring says otherwise, intervals are
pointwise: each covers its own quantity at the stated level, not all displayed
quantities jointly. The plots for adaptive experiments, sequential inference and
experimental design are documented with their areas: [Adaptive
Experiments](adaptive.md), [Sequential Inference](sequential.md) and [Experimental
Design and Power](design.md).

## Tidy results

```@docs
tidy
glance
```

## Estimates and event studies

```@docs
plot_coefficients
plot_coefficients!
plot_event_study
plot_event_study!
```

## Difference-in-differences diagnostics

```@docs
plot_trends
plot_trends!
plot_bacon
plot_bacon!
plot_honest_did
plot_honest_did!
```

## Regression discontinuity

```@docs
plot_rd
plot_rd!
plot_rd_placebos
plot_rd_placebos!
plot_rd_sensitivity
plot_rd_sensitivity!
plot_rd_density
plot_rd_density!
```

## Synthetic control

```@docs
plot_synth
plot_synth!
plot_synth_in_time
plot_synth_in_time!
```

## Randomization inference and covariate balance

```@docs
plot_randomization_distribution
plot_randomization_distribution!
plot_balance
plot_balance!
```

## Instrumental variables

```@docs
plot_confidence_set
plot_confidence_set!
plot_judge_first_stage
plot_judge_first_stage!
plot_rotemberg
plot_rotemberg!
plot_mte
plot_mte!
```

## Heterogeneous treatment effects

```@docs
plot_gates
plot_gates!
plot_cate
plot_cate!
plot_variable_importance
plot_variable_importance!
plot_rate
plot_rate!
```

## Interference

```@docs
plot_spillover_rings
plot_spillover_rings!
```

## Plot style

```@docs
drsnow_theme
```
