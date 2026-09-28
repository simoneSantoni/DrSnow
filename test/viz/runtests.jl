# Viz area: tidy results, RegressionTables compatibility, plot data and Makie plots.
#
# The Makie tests load CairoMakie (a test dependency) and render every plot to PNG.
# Set DRSNOW_TEST_VIZ=false to skip them (the Makie-free tests still run).

using Tables
import Aqua
import RegressionTables
using Printf: @sprintf

include("fixtures.jl")

const VIZ_FIX = viz_fixtures()
include("fixtures_diagnostics.jl")
const VIZ_DIAG = viz_diag_fixtures()
const VIZ_MAKIE = lowercase(get(ENV, "DRSNOW_TEST_VIZ", "true")) != "false"

include("test_tidy.jl")
include("test_data.jl")
include("test_diagnostics_data.jl")
include("test_regtables.jl")

if VIZ_MAKIE
    include("test_stubs.jl")        # before loading a backend
    import CairoMakie
    include("test_makie.jl")
    include("test_diagnostics_makie.jl")
else
    @info "DRSNOW_TEST_VIZ=false: skipping Makie plotting tests"
end

include("test_grf_plots.jl")
include("test_adaptive_plots.jl")
include("test_sequential_plots.jl")
include("test_design_plots.jl")
