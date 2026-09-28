# Always-valid and sequential inference: confidence sequences, anytime-valid p-values,
# e-processes, and group-sequential designs with alpha spending.

using StatsAPI: fit!

include("boundaries.jl")
include("results.jl")
include("monitors.jl")
include("ate.jl")
include("msprt.jl")
include("gs_spending.jl")
include("gs_engine.jl")
include("gs_design.jl")
include("gs_analysis.jl")

export ConfidenceSequence, SequentialTest, SequentialMonitor
export MeanMonitor, ATEMonitor, MSPRTMonitor
export fit!, snapshot, confidence_sequence, sequential_test, sequence_path, stopping
export confseq_mean, confseq_ate, msprt_test
export SpendingFunction, OBFSpending, PocockSpending, PowerSpending, HSDSpending
export spending, gs_design, GroupSequentialDesign, gs_analysis, GroupSequentialAnalysis
