# Synthetic Control: API

```@meta
CurrentModule = DrSnow
```

Reference documentation for the functions and types described in [Synthetic control methods](../synth.md).

## Panel preparation

```@docs
synth_panel
SynthPanel
```

## Classic synthetic control

The Abadie–Diamond–Hainmueller estimator for one treated unit, with design-based
inference and robustness checks.

```@docs
synthetic_control
SyntheticControlEstimate
synth_in_space_placebo
synth_leave_one_out
synth_in_time_placebo
```

## Synthetic difference-in-differences

```@docs
synthetic_did
SyntheticDiDEstimate
synth_cohorts
synth_time_weights
```

## Augmented synthetic control and conformal inference

```@docs
augmented_synthetic_control
AugmentedSCEstimate
synth_conformal_inference
```

## Matrix completion

```@docs
matrix_completion
MatrixCompletionEstimate
```

## Accessors shared by all estimators

```@docs
synth_weights
synth_gaps
```
