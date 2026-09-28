# Regression Discontinuity: API

```@meta
CurrentModule = DrSnow
```

Reference documentation for the functions and types described in [Regression discontinuity designs](../rd.md).

## Continuity-based estimation and inference

Local polynomial estimation with robust bias-corrected inference (`rdrobust`
equivalents), bandwidth selection, weak-first-stage inference for fuzzy designs, and
flexible covariate adjustment.

```@docs
rd_estimate
RDEstimate
rd_inference_table
rd_bandwidth
RDBandwidth
rd_weak_iv_confidence_set
rd_flex
RDFlexEstimate
```

## Honest (bias-aware) inference

Confidence intervals that are valid uniformly over a smoothness class, including with a
discrete running variable (`RDHonest` equivalents).

```@docs
rd_honest
RDHonestEstimate
rd_honest_ar_confidence_set
rd_honest_bme
RDBMEEstimate
rd_smoothness_bound
```

## Local randomization

```@docs
rd_randomization_test
rd_window_selection
```

## Manipulation and density tests

```@docs
rd_density_test
rd_density_bandwidth
rd_mccrary_test
```

## Falsification and sensitivity analyses

```@docs
rd_covariate_balance
rd_placebo_cutoffs
rd_donut
rd_bandwidth_sensitivity
```

## RD plots

```@docs
rd_plot_data
RDPlotData
```
