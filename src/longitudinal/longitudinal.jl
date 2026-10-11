# Longitudinal g-methods: mean counterfactual outcomes under static and dynamic
# treatment regimes with time-varying treatment and confounding (iterated-regression
# g-computation, IPW, longitudinal TMLE, sequentially doubly robust estimation),
# contrasts and marginal structural model projections, positivity diagnostics.
# Non-exported helpers are prefixed `_lg_`.

include("regimes.jl")
include("data.jl")
include("estimate.jl")
include("engine.jl")
include("contrasts.jl")
include("diagnostics.jl")

export TreatmentRegime, StaticRegime, DynamicRegime, always_treat, never_treat
export LongitudinalEstimate, longitudinal_regime
export RegimeContrast, regime_contrast, regime_msm, regime_weights
