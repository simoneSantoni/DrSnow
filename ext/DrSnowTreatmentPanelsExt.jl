# Interoperability with TreatmentPanels.jl (the panel type used by SynthControl.jl).
#
# - `synth_panel(bp::TreatmentPanels.BalancedPanel)` converts a TreatmentPanels panel to a
#   DrSnow `SynthPanel`, so DrSnow's estimators can be run on it directly.
# - `TreatmentPanels.BalancedPanel(p::SynthPanel)` converts back, so SynthControl.jl's
#   estimators can be run on data prepared with DrSnow.
module DrSnowTreatmentPanelsExt

using DrSnow
using DrSnow: DataFrames
import TreatmentPanels

"""
    synth_panel(bp::TreatmentPanels.BalancedPanel) -> SynthPanel

Convert a TreatmentPanels.jl `BalancedPanel` (as used by SynthControl.jl) into a
DrSnow [`SynthPanel`](@ref). Only absorbing (continuous) treatments are supported.
"""
function DrSnow.synth_panel(bp::TreatmentPanels.BalancedPanel)
    any(ismissing, bp.W) &&
        throw(ArgumentError("synth_panel: the treatment matrix has missing entries"))
    W = Int.(Bool.(bp.W))
    return DrSnow._sc_panel_from_matrices(Matrix{Float64}(bp.Y), W, collect(bp.is),
                                          collect(bp.ts);
                                          outcome=Symbol(bp.outcome_var),
                                          treatment=:treated, unit=Symbol(bp.id_var),
                                          time=Symbol(bp.t_var))
end

DrSnow.synthetic_did(bp::TreatmentPanels.BalancedPanel; kwargs...) =
    DrSnow.synthetic_did(DrSnow.synth_panel(bp); kwargs...)
DrSnow.synthetic_control(bp::TreatmentPanels.BalancedPanel; kwargs...) =
    DrSnow.synthetic_control(DrSnow.synth_panel(bp); kwargs...)
DrSnow.augmented_synthetic_control(bp::TreatmentPanels.BalancedPanel; kwargs...) =
    DrSnow.augmented_synthetic_control(DrSnow.synth_panel(bp); kwargs...)
DrSnow.matrix_completion(bp::TreatmentPanels.BalancedPanel; kwargs...) =
    DrSnow.matrix_completion(DrSnow.synth_panel(bp); kwargs...)

"""
    TreatmentPanels.BalancedPanel(p::SynthPanel)

Convert a DrSnow [`SynthPanel`](@ref) into a TreatmentPanels.jl `BalancedPanel`
(treatment starting at each treated unit's adoption period and lasting to the end).
"""
function TreatmentPanels.BalancedPanel(p::DrSnow.SynthPanel)
    N, T = size(p.Y)
    df = DataFrames.DataFrame(p.unit => repeat(p.units; inner=T),
                              p.time => repeat(p.times; outer=N),
                              p.outcome => vec(permutedims(p.Y)))
    rows = (p.n_control + 1):N
    assignment = [p.units[i] => p.times[p.adoption[i]] for i in rows]
    ta = length(assignment) == 1 ? only(assignment) : assignment
    return TreatmentPanels.BalancedPanel(df, ta; id_var=p.unit, t_var=p.time,
                                         outcome_var=p.outcome)
end

end # module
