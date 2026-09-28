# Core interface: API

```@meta
CurrentModule = DrSnow
```

Reference documentation for the types and helpers shared by every area of the
package: the result interface, diagnostic tests, inference helpers, formula
construction and the panel container. The tabulation functions [`tidy`](@ref) and
[`glance`](@ref) are documented with the other output tools in
[Results and Plotting](results.md).

## Module (API)

```@docs
DrSnow
```

## Result interface (API)

Every estimator returns a subtype of [`CausalEstimate`](@ref), which stores the
estimates and their full covariance matrix and derives standard errors, Wald
intervals and p-values from one reference distribution.

```@docs
CausalEstimate
estimate(::CausalEstimate)
estimand
method_name
tstats
pvalues
```

## Diagnostic tests (API)

Falsification and specification tests return a [`DiagnosticTest`](@ref), which
reports non-rejections as non-rejections and never as evidence that an assumption
holds.

```@docs
DiagnosticTest
rejects
```

## Inference helpers (API)

Critical values and p-values for normal and Student-t references, joint Wald tests
with the full covariance matrix, and valid Monte Carlo randomization p-values.

```@docs
critical_value
two_sided_pvalue
WaldTest
wald_test
permutation_pvalue
```

## Formulas (API)

```@docs
make_formula
```

## Panel container (API)

```@docs
TreatmentPanel
validate_panel
preprocess_panel
```
