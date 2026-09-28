"""
    DrSnow

Design-based causal inference for natural experiments in the social sciences.

Each methodological area lives in its own directory under `src/` and declares its
own exports in an aggregator file (`src/<area>/<area>.jl`). All areas share the
result interface in `src/core/estimate.jl` (every estimate is a `CausalEstimate`
supporting `coef`, `vcov`, `stderror`, `confint`, `coeftable`, `nobs`) and the
diagnostic-test type `DiagnosticTest`.

See the documentation for the list of implemented estimators.
"""
module DrSnow

using DataFrames
using Distributions
using FixedEffectModels
using FixedEffectModels: Vcov
using GLM
using LinearAlgebra
using Printf
using Random
using SparseArrays
using Statistics
using StatsAPI
using StatsBase
using StatsModels
using Tables

import StatsAPI: coef, vcov, stderror, confint, coeftable, coefnames, nobs,
                 dof_residual, pvalue, fit, predict

# Re-export the StatsAPI accessors so `using DrSnow` is enough to query results.
export coef, vcov, stderror, confint, coeftable, coefnames, nobs, dof_residual, pvalue
export Vcov

# Order matters: later areas build on core; did before sutva/ml (spillover DiD).
include("core/core.jl")
include("ri/ri.jl")
include("did/did.jl")
include("iv/iv.jl")
include("rd/rd.jl")
include("sutva/sutva.jl")
include("synth/synth.jl")
include("ml/ml.jl")
include("sequential/sequential.jl")
include("adaptive/adaptive.jl")
include("design/design.jl")
include("viz/viz.jl")
include("gui/gui.jl")

end # module DrSnow
