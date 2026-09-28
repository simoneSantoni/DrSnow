# Interference (SUTVA violations): unit-keyed interference structures, exposure
# mappings, design-based estimation, randomization tests, spillover DiD and
# dependence-robust (spatial / network HAC) variance estimators.

include("structures.jl")
include("exposure.jl")
include("design_based.jl")
include("randomization.jl")
include("hac.jl")
include("spillover_did.jl")
include("partial_interference.jl")

export InterferenceStructure, SpatialStructure, NetworkStructure, PartitionStructure
export structure_units, pairwise_distances, neighbor_matrix, shortest_path_hops
export ExposureSpec, NeighborExposure, RingExposure, HopExposure, CustomExposure
export compute_exposure, exposure_columns
export ExposureMapping, exposure_conditions, ExposureProbabilities
export exposure_probabilities, exposure_positivity, ExposureEffects, exposure_effects
export spillover_fisher_test, exposure_balance_test, treatment_moran_test
export ConleyVcov, NetworkHACVcov, conley_vcov, network_hac_vcov
export SpilloverRegression, spillover_did, exposure_regression
export SpilloverEventStudy, spillover_event_study, spillover_pretrend_test
export TwoStageRandomization, TwoStageEffects, two_stage_effects
