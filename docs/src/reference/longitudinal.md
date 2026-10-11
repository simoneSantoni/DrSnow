# Longitudinal g-methods: API

```@meta
CurrentModule = DrSnow
```

Reference documentation for the functions and types described in [Longitudinal treatment regimes](../longitudinal.md).

## Treatment regimes

```@docs
TreatmentRegime
StaticRegime
DynamicRegime
always_treat
never_treat
```

## Estimation

```@docs
longitudinal_regime
LongitudinalEstimate
```

## Contrasts and marginal structural models

```@docs
regime_contrast
regime_msm
RegimeContrast
```

## Diagnostics and plots

```@docs
regime_weights
plot_regime_weights
plot_regime_weights!
```
