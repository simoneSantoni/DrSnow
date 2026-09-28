# Randomization Inference: API

```@meta
CurrentModule = DrSnow
```

Reference documentation for the functions and types described in
[Randomization inference](../ri.md). Randomization inference treats potential outcomes
as fixed and the assignment of treatment as the only source of randomness, so every
procedure below starts from an explicit assignment mechanism.

## Assignment mechanisms (API)

The known (or assumed) designs that generate the reference distribution, and the
functions that draw, enumerate and describe them.

```@docs
AssignmentMechanism
BernoulliAssignment
CompleteRandomization
StratifiedRandomization
ClusterRandomization
BlockClusterRandomization
MatchedPairsRandomization
Rerandomization
CustomAssignment
draw_assignment
treatment_probabilities
n_units
n_assignments
enumerate_assignments
balance_mahalanobis
```

## Fisher randomization tests (API)

Exact tests of sharp null hypotheses, and confidence intervals and Hodges–Lehmann
estimates obtained by inverting them.

```@docs
randomization_test
RandomizationTestResult
randomization_distribution
ri_confint
RandomizationInterval
```

## Regression coefficients and balance (API)

Randomization-c and randomization-t inference for regression coefficients, and
randomization tests of covariate balance.

```@docs
ri_regression
RIRegressionResult
ri_balance_test
```

## Multiple testing (API)

Step-down adjustments from the joint randomization distribution, and Holm and
Benjamini–Hochberg adjustments of unadjusted p-values.

```@docs
ri_multiple_testing
MultipleTestingResult
westfall_young_adjust
holm_adjust
bh_adjust
```
