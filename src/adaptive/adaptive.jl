# Adaptive experiments: response-adaptive assignment policies (bandits) with recorded
# assignment probabilities, inference after adaptive data collection (adaptively
# weighted AIPW for arm and policy values), off-policy evaluation from logged bandit
# data, and causal excursion effects in micro-randomized trials (WCLS, EMEE).
# Non-exported helpers are prefixed `_ad_`.

include("policies.jl")
include("experiment.jl")
include("weighting.jl")
include("ope.jl")
include("mrt.jl")

export AdaptivePolicy, BetaBernoulliThompson, GaussianThompson, TopTwoThompson
export EpsilonGreedy, SoftmaxPolicy, UCBPolicy, LinearThompson
export assignment_probabilities, update_policy!, n_arms
export BanditEnvironment, BernoulliBandit, GaussianBandit, ContextualBandit
export AdaptiveExperiment, assign!, observe!, experiment_log, AdaptiveLog
export run_adaptive_experiment, cumulative_regret
export AdaptiveEstimate, adaptive_arm_values, adaptive_policy_value, naive_arm_means
export off_policy_value, bandit_dr_scores
export ExcursionEffectEstimate, wcls, emee
