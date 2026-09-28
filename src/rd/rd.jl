# Regression discontinuity designs (sharp, fuzzy, kink).

include("utils.jl")
include("bandwidth.jl")
include("estimate.jl")
include("density.jl")
include("mccrary.jl")
include("plot.jl")
include("falsification.jl")
include("local_randomization.jl")
include("honest_utils.jl")
include("honest.jl")
include("honest_discrete.jl")

export RDBandwidth, RDEstimate, RDPlotData, RDHonestEstimate, RDBMEEstimate
export rd_bandwidth, rd_estimate, rd_inference_table, rd_weak_iv_confidence_set
export rd_density_test, rd_density_bandwidth, rd_mccrary_test
export rd_plot_data
export rd_covariate_balance, rd_placebo_cutoffs, rd_donut, rd_bandwidth_sensitivity
export rd_randomization_test, rd_window_selection
export rd_honest, rd_honest_ar_confidence_set, rd_honest_bme, rd_smoothness_bound
