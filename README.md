# DrSnow

<p align="center">
  <img src="imgs/icon.png" alt="DrSnow logo" width="160"/>
</p>

<p align="center">
  <a href="https://github.com/simoneSantoni/DrSnow_alpha/actions/workflows/CI.yml"><img src="https://github.com/simoneSantoni/DrSnow_alpha/actions/workflows/CI.yml/badge.svg" alt="CI"/></a>
  <a href="https://github.com/simoneSantoni/DrSnow_alpha/actions/workflows/documenter.yml"><img src="https://github.com/simoneSantoni/DrSnow_alpha/actions/workflows/documenter.yml/badge.svg" alt="Documentation build"/></a>
  <a href="https://simonesantoni.github.io/DrSnow_alpha/"><img src="https://img.shields.io/badge/docs-dev-blue.svg" alt="Documentation"/></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="License: MIT"/></a>
</p>

DrSnow is a Julia package for design-based causal inference with natural experiments
in the social sciences. It covers difference-in-differences, instrumental variables,
regression discontinuity, synthetic control, randomization inference, interference
(spillovers) and double/debiased machine learning, with one result interface across
methods.

Every estimator returns a `CausalEstimate` that carries its full covariance matrix and
states its estimand, so `coef`, `vcov`, `stderror`, `confint(r; level)`, `coeftable`,
`tidy` and `glance` work the same way everywhere. Diagnostics return a
`DiagnosticTest`; a non-rejection is never reported as evidence that an assumption
holds. Estimators are validated against reference implementations, mostly R
packages, and most inferential procedures have a Monte Carlo size or coverage
check in the test suite.

DrSnow is pre-1.0 and under active development. Version 0.2 is a rewrite of 0.1 with
many breaking changes; see [CHANGELOG.md](CHANGELOG.md).

## Methods

| Area | Key functions | Guide |
|:--|:--|:--|
| Difference-in-differences | `did_twfe`, `event_study`, `did_callaway_santanna` + `aggregate_att`, `did_sun_abraham`, `did_imputation`, `did_drdid`, `did_etwfe`, `did_multiplegt_dyn`, `did_continuous`; diagnostics `bacon_decomposition`, `twfe_weights`, `pre_trend_test`; sensitivity `honest_did`, `honest_breakdown` | [did.md](docs/src/did.md) |
| Instrumental variables | `iv_regression`, `late_2sls`, `weak_iv_test`, `weak_iv_confidence_set` (AR, CLR, K), `tf_confint`, `complier_characteristics`, `late_ipw`, `plausibly_exogenous`, `kclass_iv`, `jive`, `judge_iv`, `shift_share_iv`, `mte`, `mte_bounds`, `residual_prediction_test`, `ml_first_stage`, `dml_lqte` | [iv.md](docs/src/iv.md) |
| Regression discontinuity | `rd_estimate` (sharp, fuzzy, kink), `rd_bandwidth`, `rd_density_test`, `rd_plot_data`, `rd_covariate_balance`, `rd_randomization_test`; honest inference `rd_honest`, `rd_honest_bme`; `rd_mccrary_test` | [rd.md](docs/src/rd.md) |
| Synthetic control | `synthetic_did`, `synthetic_control`, `augmented_synthetic_control`, `matrix_completion`, placebo and conformal inference | [synth.md](docs/src/synth.md) |
| Randomization inference | `randomization_test`, `ri_confint`, `ri_regression`, `ri_balance_test`, `ri_multiple_testing`; assignment mechanisms | [ri.md](docs/src/ri.md) |
| Sequential inference | `confseq_mean`, `confseq_ate`, `msprt_test`, streaming monitors, `gs_design`, `gs_analysis` | [sequential.md](docs/src/sequential.md) |
| Adaptive experiments | Thompson sampling and other policies with floors, `run_adaptive_experiment`, `adaptive_arm_values`, `off_policy_value`, `wcls`, `emee` | [adaptive.md](docs/src/adaptive.md) |
| Design and power | `power_means`, `power_cluster`, `power_did`, `power_rd`, `block_design`, `experiment_estimate`, `diagnose_design`, `optimize_design` | [design.md](docs/src/design.md) |
| Interference (SUTVA) | spatial, network and partition structures, `compute_exposure`, `exposure_effects` (Aronow–Samii), `spillover_fisher_test`, `conley_vcov`, `spillover_did`, `two_stage_effects` | [sutva.md](docs/src/sutva.md) |
| Causal machine learning | `dml_plr`, `dml_irm`, `dml_pliv`, `dml_iivm`, `dml_did`, `cate_dr_learner`, `generic_ml` (BLP/GATES/CLAN), `policy_tree`, `ppi_mean`/`ppi_ols`; generalized random forests `causal_forest`, `instrumental_forest`, `rank_average_treatment_effect`; meta-learners `s_learner`, `t_learner`, `x_learner`, `r_learner`; `subgroup_effects`; `dml_did_multi`, `rd_flex`; `dsl_regression`, `ppi_ate` | [ml.md](docs/src/ml.md) |
| Results and plots | `tidy`, `glance`, Tables.jl interface, `regtable` (extension), `plot_event_study`, `plot_rd`, `plot_synth`, … (Makie extension) | [results.md](docs/src/results.md) |

## Installation

DrSnow is not registered in the General registry. Install it from GitHub (Julia 1.10
or later):

```julia
using Pkg
Pkg.add(url="https://github.com/simoneSantoni/DrSnow_alpha")
```

For development:

```bash
git clone https://github.com/simoneSantoni/DrSnow_alpha.git
cd DrSnow_alpha
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

## Quick start

```julia
using DrSnow, DataFrames, Random

rng = Xoshiro(1)

# Staggered adoption: 300 units observed 2010-2019; cohorts start in 2014, 2016 or never
cohort = rand(rng, [0, 2014, 2016], 300)                  # 0 = never treated
df = DataFrame(unit = repeat(1:300, inner = 10), year = repeat(2010:2019, 300))
df.first_treat = cohort[df.unit]
df.d = Int.((df.first_treat .> 0) .& (df.year .>= df.first_treat))
effect = df.d .* (1.0 .+ 0.5 .* (df.year .- df.first_treat))  # grows with exposure
df.y = 0.01 .* df.unit .+ 0.2 .* (df.year .- 2010) .+ effect .+ randn(rng, nrow(df))

cs = did_callaway_santanna(df, :y, FirstTreated(:first_treat), :unit, :year; rng = rng)
aggregate_att(cs, :simple)             # overall ATT
es = aggregate_att(cs, :dynamic)       # event-study aggregation
pre_trend_test(cs)                     # joint Wald test of pre-treatment ATT(g,t)

# Sharp regression discontinuity: jump of 0.5 at x = 0
x = 2 .* rand(rng, 2_000) .- 1
rdd = DataFrame(x = x, y = 1 .+ x .+ 0.5 .* (x .>= 0) .+ 0.3 .* randn(rng, 2_000))
rd = rd_estimate(rdd, :y, :x)          # robust bias-corrected inference
rd_density_test(rdd, :x)               # manipulation (density) test
```

With a Makie backend installed, the event study can be plotted:

```julia
using CairoMakie
plot_event_study(es)
```

The [tutorial](docs/src/tutorial.md) works through a staggered DiD with HonestDiD
sensitivity analysis, an IV analysis with weak-instrument-robust inference, a sharp
RD and a randomization test on real and simulated data.

## Optional extensions

Heavy dependencies are package extensions that load only when you load the
corresponding package next to DrSnow:

| Load | Enables |
|:--|:--|
| `CairoMakie` (or another Makie backend) | `plot_event_study`, `plot_rd`, `plot_synth`, `plot_bacon`, `plot_gates`, `plot_confidence_set`, … |
| `RegressionTables` | `regtable` with DrSnow results (text, LaTeX, HTML) |
| `MLJ` (regressors need only `MLJModelInterface`) | any MLJ model as a nuisance learner via `MLJLearner` |
| `HTTP`, `JSON3`, `CSV` (all three) | `launch_gui()`: a local web interface for DiD, event studies, RD, IV and synthetic DiD |
| `TreatmentPanels` | conversion to and from `TreatmentPanels.BalancedPanel` (SynthControl.jl) |

The web interface is documented in [gui.md](docs/src/gui.md); a Dockerfile for it is
included.

## Validation

The test suite compares DrSnow with reference implementations on published data,
using reference values committed under [`test/validation/`](test/validation/)
together with the R scripts that generated them:

- DiD: did, DRDID, fixest, didimputation, bacondecomp, TwoWayFEWeights, HonestDiD,
  DIDmultiplegtDYN and etwfe (on `mpdta`, LaLonde and other data);
- IV: AER, sandwich, ivmodel, fixest, ivDiag and ShiftShareSE (Card 1995, Mroz 1987);
- RD: rdrobust 4.0.0 and rddensity 3.0 (U.S. Senate data), to about 1e-9;
- synthetic control: synthdid, augsynth, Synth and MCPanel (Proposition 99, Basque
  Country);
- randomization inference: ri2 and exact enumeration;
- interference: interference (Aronow–Samii) and `fixest::vcov_conley`;
- causal ML: DoubleML and DRDID.

Monte Carlo checks of size and coverage run with reduced replication counts by
default and with full counts when `DRSNOW_SLOW_TESTS=true`. The
[validation page](docs/src/validation.md) lists packages, versions and tolerances.

```bash
julia --project=. -e 'using Pkg; Pkg.test()'                                # all areas
DRSNOW_TEST_GROUP=did julia --project=. -e 'using Pkg; Pkg.test()'          # one area
DRSNOW_SLOW_TESTS=true julia --project=. -e 'using Pkg; Pkg.test()'         # full Monte Carlo
```

## Documentation

- Online manual: <https://simonesantoni.github.io/DrSnow_alpha/>
- Sources of the manual, readable on GitHub: [`docs/src/`](docs/src/) —
  [overview](docs/src/index.md), [tutorial](docs/src/tutorial.md), methods guides
  ([DiD](docs/src/did.md), [IV](docs/src/iv.md), [RD](docs/src/rd.md),
  [synthetic control](docs/src/synth.md), [randomization inference](docs/src/ri.md),
  [interference](docs/src/sutva.md), [causal ML](docs/src/ml.md)),
  [results and plots](docs/src/results.md), [web interface](docs/src/gui.md),
  [validation](docs/src/validation.md), [API](docs/src/api.md).
- Example scripts: [`examples/`](examples/).
- Literature reviews that informed the package: [`literature/`](literature/).

## FAQ

**Does passing a pre-trend test mean parallel trends holds?**
No. Pre-trend tests often have low power, and conditioning on passing them distorts
inference (Roth 2022). Report the pre-period estimates and use `honest_did` to see how
large a violation would change the conclusion.

**Should I use `did_twfe` with staggered adoption?**
Only with care. With several adoption dates and effects that vary over time, the TWFE
coefficient can put negative weight on some effects; `did_twfe` warns in that case.
Use `bacon_decomposition` and `twfe_weights` to diagnose it, and a
heterogeneity-robust estimator (`did_callaway_santanna`, `did_imputation`,
`did_sun_abraham`, `did_etwfe`) for estimation.

**How are standard errors clustered?**
Panel estimators cluster by unit by default; pass `cluster` (one or more columns) or a
`vcov` estimator such as `Vcov.cluster(:state)`. IV defaults to HC1. Each guide
documents its defaults.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) and the development conventions in
[docs/CONVENTIONS.md](docs/CONVENTIONS.md).

## Citation

```bibtex
@software{drsnow,
  author = {Santoni, Simone and contributors},
  title  = {DrSnow: Design-Based Causal Inference for Natural Experiments in Julia},
  year   = {2026},
  url    = {https://github.com/simoneSantoni/DrSnow_alpha},
  note   = {Version 0.2}
}
```

Please also cite the papers behind the methods you use; each guide lists them.

## License

MIT; see [LICENSE](LICENSE).
