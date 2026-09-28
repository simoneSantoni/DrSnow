# Synthetic control methods: panel preparation, synthetic DiD (synthdid), classic
# Abadie–Diamond–Hainmueller synthetic control with inference and robustness checks,
# augmented synthetic control with conformal inference, and matrix completion.
#
# SynthControl.jl / TreatmentPanels.jl interoperability lives in the package
# extension `ext/DrSnowTreatmentPanelsExt.jl` (loaded with `using TreatmentPanels`).

include("panel.jl")
include("solvers.jl")
include("common.jl")
include("sdid.jl")
include("adh.jl")
include("ascm.jl")
include("mcnnm.jl")

export SynthPanel, synth_panel
export SyntheticDiDEstimate, synthetic_did, synth_cohorts
export SyntheticControlEstimate, synthetic_control
export synth_in_space_placebo, synth_leave_one_out, synth_in_time_placebo
export AugmentedSCEstimate, augmented_synthetic_control, synth_conformal_inference
export MatrixCompletionEstimate, matrix_completion
export synth_weights, synth_time_weights, synth_gaps
