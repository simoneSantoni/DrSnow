# Interference (SUTVA): API

```@meta
CurrentModule = DrSnow
```

Reference documentation for the functions and types described in
[Interference and spillovers](../sutva.md).

## Interference structures (API)

Unit-keyed descriptions of who can affect whom: spatial locations, networks and
partitions into groups.

```@docs
InterferenceStructure
SpatialStructure
NetworkStructure
PartitionStructure
structure_units
pairwise_distances
neighbor_matrix
shortest_path_hops
```

## Exposure specifications (API)

Rules that summarize the treatments of a unit's neighbours into its exposure.

```@docs
ExposureSpec
NeighborExposure
RingExposure
HopExposure
CustomExposure
compute_exposure
exposure_columns
```

## Design-based estimation (API)

Discrete exposure mappings, exposure probabilities under the design, and
Horvitz–Thompson and Hájek estimators of exposure contrasts (Aronow and Samii 2017).

```@docs
ExposureMapping
exposure_conditions
ExposureProbabilities
exposure_probabilities
exposure_positivity
ExposureEffects
exposure_effects
```

## Randomization tests under interference (API)

Conditional randomization tests of the null of no spillovers, and tests of the
assignment mechanism calibrated by the design.

```@docs
spillover_fisher_test
exposure_balance_test
treatment_moran_test
```

## Dependence-robust variance estimators (API)

Spatial (Conley) and network HAC covariance estimators for regressions.

```@docs
ConleyVcov
NetworkHACVcov
conley_vcov
network_hac_vcov
```

## Spillover regressions and event studies (API)

Difference in differences and cross-sectional regressions with exposure terms.

```@docs
SpilloverRegression
spillover_did
exposure_regression
SpilloverEventStudy
spillover_event_study
spillover_pretrend_test
```

## Partial interference and two-stage designs (API)

```@docs
TwoStageRandomization
TwoStageEffects
two_stage_effects
```
