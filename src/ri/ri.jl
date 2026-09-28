# Randomization inference (Fisher exact tests) under known assignment mechanisms.

include("designs.jl")
include("mechanisms.jl")
include("statistics.jl")
include("engine.jl")
include("fisher_test.jl")
include("confint.jl")
include("multiple_testing.jl")
include("regression.jl")
include("balance.jl")

export AssignmentMechanism, BernoulliAssignment, CompleteRandomization
export StratifiedRandomization, ClusterRandomization, CustomAssignment
export draw_assignment, treatment_probabilities, n_units
export MatchedPairsRandomization, BlockClusterRandomization, Rerandomization
export balance_mahalanobis, n_assignments, enumerate_assignments
export randomization_test, RandomizationTestResult, randomization_distribution
export ri_confint, RandomizationInterval
export ri_multiple_testing, MultipleTestingResult, westfall_young_adjust
export holm_adjust, bh_adjust
export ri_regression, RIRegressionResult
export ri_balance_test
