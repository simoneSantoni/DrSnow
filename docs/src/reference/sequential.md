# Sequential Inference: API

```@meta
CurrentModule = DrSnow
```

Reference documentation for the functions and types described in [Sequential inference](../sequential.md).

## Confidence sequences (API)

Anytime-valid confidence sequences for a mean and for the average treatment effect,
computed over data in arrival order, and the result type they return.

```@docs
confseq_mean
confseq_ate
ConfidenceSequence
confint(::ConfidenceSequence)
pvalues(::ConfidenceSequence)
```

## Sequential tests and e-processes (API)

Always-valid tests: the mixture SPRT for two-arm experiments, the test dual to a
confidence sequence, and summaries of the first crossing.

```@docs
msprt_test
SequentialTest
sequential_test
stopping
sequence_path
DiagnosticTest(::SequentialTest)
```

## Streaming monitors (API)

Online versions of the procedures above, updated observation by observation.

```@docs
SequentialMonitor
MeanMonitor
ATEMonitor
MSPRTMonitor
fit!(::SequentialMonitor, ::Any...)
snapshot
confidence_sequence
```

## Group-sequential designs: spending functions (API)

```@docs
SpendingFunction
OBFSpending
PocockSpending
PowerSpending
HSDSpending
spending
```

## Group-sequential designs: design and analysis (API)

```@docs
gs_design
GroupSequentialDesign
gs_analysis
GroupSequentialAnalysis
```

## Plots (API)

```@docs
plot_confidence_sequence
plot_confidence_sequence!
plot_gs_boundaries
plot_gs_boundaries!
```
