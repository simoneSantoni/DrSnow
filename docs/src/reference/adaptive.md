# Adaptive Experiments: API

```@meta
CurrentModule = DrSnow
```

Reference documentation for the functions and types described in [Adaptive experiments](../adaptive.md).

## Assignment policies

```@docs
AdaptivePolicy
BetaBernoulliThompson
GaussianThompson
TopTwoThompson
EpsilonGreedy
SoftmaxPolicy
UCBPolicy
LinearThompson
assignment_probabilities
update_policy!
n_arms
```

## Running and simulating experiments

```@docs
AdaptiveExperiment
assign!
observe!
experiment_log
AdaptiveLog
run_adaptive_experiment
BanditEnvironment
BernoulliBandit
GaussianBandit
ContextualBandit
cumulative_regret
```

## Inference after adaptive data collection

```@docs
adaptive_arm_values
adaptive_policy_value
naive_arm_means
AdaptiveEstimate
```

## Off-policy evaluation and policy learning

```@docs
off_policy_value
bandit_dr_scores
```

## Micro-randomized trials

```@docs
wcls
emee
ExcursionEffectEstimate
```

## Plots

```@docs
plot_assignment_probabilities
plot_assignment_probabilities!
plot_excursion_effect
plot_excursion_effect!
```
