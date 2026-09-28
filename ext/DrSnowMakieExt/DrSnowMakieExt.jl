# Makie plotting for DrSnow results. Loaded automatically with any Makie backend
# (`using CairoMakie`, `using GLMakie`, ...).
#
# Every plot has a mutating form `plot_x!(ax, result; ...)` that draws into an
# existing `Makie.Axis` and returns it, and a non-mutating form `plot_x(result; ...)`
# that builds a styled `Figure` (with legend and caption) and returns it. The data
# drawn come from the backend-independent `DrSnow._viz_*_data` functions
# (src/viz/data.jl); this module only draws. See src/viz/viz.jl for how to add a
# plot for a new result type.
module DrSnowMakieExt

using DrSnow
using DrSnow: DataFrames, Statistics, StatsAPI, Random
using DrSnow: CausalEstimate, DiagnosticTest, critical_value, tidy
using DataFrames: DataFrame, nrow
using Makie

import DrSnow: plot_event_study, plot_event_study!, plot_coefficients, plot_coefficients!,
               plot_rd, plot_rd!, plot_synth, plot_synth!,
               plot_randomization_distribution, plot_randomization_distribution!,
               plot_bacon, plot_bacon!, plot_gates, plot_gates!, plot_cate, plot_cate!,
               plot_spillover_rings, plot_spillover_rings!,
               plot_confidence_set, plot_confidence_set!, plot_variable_importance,
               plot_variable_importance!, plot_rate, plot_rate!
import DrSnow: plot_confidence_sequence, plot_confidence_sequence!, plot_gs_boundaries,
               plot_gs_boundaries!

include("common.jl")
include("event_study.jl")
include("coefficients.jl")
include("rd.jl")
include("synth.jl")
include("randomization.jl")
include("did.jl")
include("ml.jl")
include("sutva.jl")
include("iv.jl")
include("trends.jl")
include("balance.jl")
include("rd_falsification.jl")
include("honest_did.jl")
include("iv_design.jl")
include("synth_in_time.jl")
include("theme.jl")
include("grf.jl")
include("adaptive.jl")
include("sequential.jl")
include("design.jl")

end # module DrSnowMakieExt
