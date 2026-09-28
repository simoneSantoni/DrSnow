# Experimental Design and Power: API

```@meta
CurrentModule = DrSnow
```

Reference documentation for the functions and types described in [Experimental design and power](../design.md).

## Analytic power and minimum detectable effects

```@docs
PowerAnalysis
power_means
power_proportions
power_cluster
power_blocked
power_did
power_iv
power_rd
```

## Prognostic scores, blocking and matched pairs

```@docs
PrognosticScore
prognostic_score
BlockingDesign
block_design
assign_treatment
variance_reduction
```

## Analysis of randomized experiments

```@docs
ExperimentEstimate
experiment_estimate
```

## Simulation-based design diagnosis and optimization

```@docs
DeclaredDesign
declare_design
DesignDiagnosis
diagnose_design
diagnose_grid
DesignOptimization
optimize_design
```

## Plots

```@docs
plot_power_curve
plot_power_curve!
plot_blocks
plot_blocks!
```
