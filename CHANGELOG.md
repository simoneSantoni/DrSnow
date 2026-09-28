# Changelog

All notable changes to DrSnow will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased] — 0.2.0

This release rebuilds DrSnow following the September 2026 panel review
(`docs/panel_review_2026-09-27.md`). Most v0.1 inferential outputs were incorrect;
every area has been rewritten, validated against reference implementations (mostly R
packages) and checked by Monte Carlo size and coverage tests. Pre-1.0, the API breaks
freely; the list below is complete for exported names.

### Security

- Model formulas are built programmatically. The v0.1 `eval(Meta.parse(...))` formula
  construction allowed arbitrary code execution through column names, including from
  CSV uploads to the GUI. A test now forbids `eval`, `Meta.parse` and runtime `include`
  in `src/` and `ext/`.
- The GUI was rewritten (see below) with CSRF protection, Host/Origin checks, streaming
  upload limits, a strict Content-Security-Policy and no `innerHTML` with user data.

### Added

- **Shared foundation.** `CausalEstimate <: StatsAPI.StatisticalModel` for every result
  (`coef`, `vcov`, `stderror`, `confint(; level)`, `coeftable`, `pvalues`, `nobs`,
  `dof_residual`, `estimate`, `estimand`, `method_name`); `DiagnosticTest` whose printout
  never presents non-rejection as evidence for an assumption; `wald_test` with the full
  covariance; `critical_value`; `permutation_pvalue`; `make_formula`; `tidy`, `glance`
  and a Tables.jl interface for results.
- **Difference-in-Differences.** Cohort layer (`treatment_timing`); Callaway–Sant'Anna
  (`did_callaway_santanna`, `aggregate_att`, panel and repeated cross-sections, bootstrap
  uniform bands); Sant'Anna–Zhao doubly robust DiD (`did_drdid`); Sun–Abraham
  (`did_sun_abraham`); Borusyak–Jaravel–Spiess imputation (`did_imputation`);
  Goodman-Bacon decomposition (`bacon_decomposition`); de Chaisemartin–D'Haultfœuille
  weights (`twfe_weights`) and dynamic estimator (`did_multiplegt_dyn`); Wooldridge
  extended TWFE including Poisson/logit (`did_etwfe`); continuous-treatment DiD
  (`did_continuous`); HonestDiD sensitivity analysis (`honest_did`, `honest_breakdown`);
  `pretreatment_balance`.
- **Instrumental variables.** `iv_regression` on FixedEffectModels (multiple endogenous
  regressors and instruments, fixed effects, weights, HC1 / one- and two-way cluster);
  effective F, Kleibergen–Paap, Stock–Yogo, tF (`weak_iv_test`, `tf_confint`); analytic
  Anderson–Rubin, CLR and K confidence sets (`weak_iv_confidence_set`); κ-weighted
  complier profiles and IPW LATE (`complier_characteristics`, `late_ipw`,
  `complier_outcome_distribution`); Kitagawa and Huber–Mellace validity tests; Hansen J
  and Durbin–Wu–Hausman; plausibly exogenous sensitivity (`plausibly_exogenous`);
  LATE extrapolation (`late_extrapolation`); LIML, Fuller, HLIM, HFUL (`kclass_iv`);
  JIVE/UJIVE (`jive`); judge designs (`judge_iv` and diagnostics); shift-share IV with
  AKM/BHJ inference and Rotemberg weights; marginal treatment effects (`mte`).
- **Regression discontinuity** (new). `rd_estimate` (sharp, fuzzy, kink; robust
  bias-corrected inference; all rdrobust variance options), `rd_bandwidth` (ten
  selectors), `rd_density_test`, `rd_plot_data`, falsification helpers, local
  randomization tests and a weak-IV-robust fuzzy RD confidence set. Matches
  rdrobust/rddensity/rdplot to about 1e-9.
- **Synthetic control** (new). `synthetic_did` (SDID, SC and DiD, placebo / bootstrap /
  jackknife, staggered adoption), classic `synthetic_control` with placebo inference,
  `augmented_synthetic_control` with conformal inference, `matrix_completion`
  (MC-NNM). Matches synthdid, augsynth, MCPanel and Synth.
- **Randomization inference** (new). Assignment mechanisms, `randomization_test` with
  exact enumeration, `ri_confint`, `ri_regression` (Young 2019), `ri_balance_test`,
  Westfall–Young / Holm / BH multiple-testing adjustments. Matches ri2.
- **Interference** (rewritten). Unit-keyed spatial, network and partition structures;
  exposure mappings; Aronow–Samii estimators; Fisher tests for spillovers; Conley and
  network HAC covariance; spillover DiD and ring event studies; two-stage randomized
  designs.
- **Causal machine learning** (new). Pluggable learners (built-in OLS, ridge, lasso,
  logistic, k-NN; any MLJ model via the `DrSnowMLJExt` extension); cross-fitting; DML
  for PLR, IRM, PLIV, IIVM and DiD; DR-learner, BLP/GATES/CLAN, CATE projection;
  policy trees; prediction-powered inference. Matches DoubleML.
- **Heterogeneous treatment effects.** Pure-Julia generalized random forests (port of
  grf 2.6.1's core: honesty, subsampling, local centering, little-bags variance,
  clusters, weights, missing covariates): `causal_forest`, `regression_forest`,
  `instrumental_forest`, `ForestLearner`, with `average_treatment_effect`,
  `best_linear_projection`, `test_calibration`, `variable_importance`,
  `rank_average_treatment_effect` (AUTOC/Qini), `policy_value`; S-, T-, X- and
  R-learners (`s_learner`, `t_learner`, `x_learner`, `r_learner`,
  `metalearner_bootstrap`); pre-specified `subgroup_effects`, `interaction_effects`,
  `heterogeneity_test`.
- **Honest RD inference.** `rd_honest` (Armstrong–Kolesár bias-aware intervals, sharp,
  fuzzy and at a point; RDHonest port), `rd_honest_ar_confidence_set` (Noack–Rothe
  2024), `rd_honest_bme` (Kolesár–Rothe 2018, discrete running variable),
  `rd_smoothness_bound`, `rd_mccrary_test`, and the `stdvars` option.
- **IV extensions.** Heteroskedasticity- and cluster-robust CLR and K; Kitagawa test
  with covariates; parametric LATE extrapolation; Hansen–Hausman–Newey variance
  (`kclass_iv(...; se=:hhn)`) and CJIVE with clustered SEs; MTE bounds
  (`mte_bounds`, Mogstad–Santos–Torgovitsky 2018); panel Rotemberg weights; the
  Frandsen–Lefgren–Leslie test as the default `judge_validity_test`.
- **Diagnostic plots and theme.** `plot_trends`, `plot_balance`, `plot_rd_placebos`,
  `plot_rd_sensitivity`, `plot_rd_density`, `plot_honest_did`,
  `plot_judge_first_stage`, `plot_rotemberg`, `plot_mte`, `plot_synth_in_time`,
  `plot_variable_importance`, `plot_rate`, and `drsnow_theme`.
- **Machine learning for natural experiments and experimentation** (roadmap from the
  review of machine learning for social science experiments):
  - staggered DML difference-in-differences (`dml_did_multi`) and flexible ML
    covariate adjustment in RD (`rd_flex`, Noack, Olma & Rothe 2024);
  - a new `sequential` area: confidence sequences (`confseq_mean`, `confseq_ate`),
    streaming monitors, mixture-SPRT always-valid p-values (`msprt_test`), and
    group-sequential designs and analyses (`gs_design`, `gs_analysis`);
  - a new `adaptive` area: response-adaptive policies (Thompson sampling, UCB,
    ε-greedy, contextual) with floors and burn-in, adaptively weighted AIPW
    (`adaptive_arm_values`, `adaptive_policy_value`), off-policy evaluation, and
    micro-randomized trials (`wcls`, `emee`);
  - inference with ML-measured variables: design-based supervised learning
    (`dsl_regression`), PPI for causal targets (`ppi_ate`, `cross_ppi`), DiD and RD
    with predicted outcomes, regression calibration and a differential-error test;
  - ML-era IV diagnostics: `residual_prediction_test`, cross-fitted first-stage
    strength (`ml_first_stage`), DML Anderson–Rubin, `multiple_iv_weights`
    (Mogstad–Torgovitsky–Walters), and local quantile treatment effects
    (`dml_lqte`, `dml_complier_cdf`);
  - a new `design` area: analytic power and MDE (`power_means`, `power_cluster`,
    `power_did`, `power_rd`, …), blocking and matched pairs on prognostic scores
    (`block_design`), shared-score analysis (`experiment_estimate`), and
    simulation-based diagnosis and surrogate-model design optimization.
- **Benchmarks.** `benchmark/` suite (BenchmarkTools, PkgBenchmark-compatible) with an
  optional GitHub workflow.
- **Results and plotting.** Makie extension (`plot_event_study`, `plot_rd`, `plot_synth`,
  `plot_randomization_distribution`, `plot_bacon`, `plot_gates`, `plot_cate`,
  `plot_coefficients`, `plot_spillover_rings`, `plot_confidence_set`) and a
  RegressionTables.jl extension.
- **Tooling.** CI test workflow (Julia 1.10 and current, Linux/macOS/Windows), Aqua
  checks, grouped test runner (`DRSNOW_TEST_GROUP`, `DRSNOW_SLOW_TESTS`), validation
  data and generating R scripts under `test/validation/`.

### Changed (breaking)

- RD results (`rd_estimate`, `rd_flex`) follow the reporting convention of `rdrobust`:
  `coef` is now the conventional point estimate, while `stderror`, `pvalues`, `tstats`
  and `confint` remain robust bias-corrected, with the interval centred on the
  bias-corrected estimate (`tau_bias_corrected`). `tidy` and `regtable` therefore show
  the conventional estimate next to robust inference. Previously `coef` returned the
  bias-corrected estimate, contradicting the documented reporting advice.
- Minimum Julia is 1.10. GUI packages are no longer hard dependencies; the GUI, MLJ,
  Makie, RegressionTables and TreatmentPanels integrations are package extensions.
- `DiDEstimate` and `EventStudyEstimate` have new fields (`coef`, `vcov`, …); the old
  `att`/`se`/`ci_lower`/`ci_upper`/`model` fields are gone (model in `details.model`).
  Event-study coefficient names are `"e=k"` (binned endpoints `"e<=k"`, `"e>=k"`).
- `did_twfe`: `n_treated`/`n_control` count ever-treated / never-treated units; CIs use
  t(G−1); `vcov` keyword; warnings for staggered or non-absorbing designs; errors for
  non-identified treatment effects.
- `event_study`: fully dynamic by default; out-of-window periods are binned
  (`endpoints=:bin`) or trimmed, never pooled into the reference period; multiple
  cohorts default to Sun–Abraham (`estimator=:auto`); errors on empty or
  non-identified periods.
- `pre_trend_test` and `parallel_trends_test` return `DiagnosticTest` and use a
  full-covariance Wald test; `parallel_trends_test` wraps `event_study`
  (`n_pre_periods` removed).
- `IVEstimate` subtypes `CausalEstimate`; default covariance is HC1; `cluster_var` is
  replaced by `cluster`/`vcov`; `first_stage_diagnostics` returns
  `WeakIVDiagnostics` with keyword covariates; `estimate_compliance` requires binary D
  and Z; `complier_characteristics(data, treatment, instrument, variables)` returns
  actual complier means.
- Convenience DiD methods no longer call `preprocess_panel`; results do not depend on
  row order anywhere in the package.
- The GUI is `launch_gui`/`stop_gui` after `using DrSnow, HTTP, JSON3, CSV`.

### Removed

- IV: `weak_iv_inference` (use `weak_iv_confidence_set`), `test_monotonicity` (use
  `first_stage_sign_test`, `instrument_validity_test`), `test_exclusion_restriction`
  (use `zero_first_stage_test`, `instrument_balance`), `sensitivity_analysis` (use
  `plausibly_exogenous`), `external_validity_test` (use `late_extrapolation`), and the
  types `MonotonicityTest`, `ExclusionTest`, `SensitivityResult`,
  `ExternalValidityTest`.
- Interference: `SpatialPanel`, `NetworkPanel`, `compute_distance_matrix`,
  `network_exposure`, `test_network_interference`, `balance_test_neighbors`,
  `neighbor_treatment_correlation`, `did_with_spillover_controls`,
  `spatial_spillover_gradient`, `placebo_spillover_test`,
  `estimate_spillover_by_degree`, `SUTVATestResult`, `SpilloverEstimate`,
  `SpilloverGradient`. See the interference guide for replacements.
- Genie-based GUI, `test_gui_load.jl`, `docs/src/algorithms.md` (content moved to the
  per-method guides).
- Third-party PDFs are no longer tracked in the repository.

### Fixed

- The estimand note printed by `iv_regression` / `late_2sls` described a binary
  instrument with a multi-valued treatment and covariates (e.g. Card 1995) as a
  multi-valued-instrument estimand. It now reports a weighted combination of conditional
  average causal responses (or within-cell ACRs with saturated controls), with the
  appropriate weighting caveats.

Every defect listed in the panel review, including: wrong 2SLS standard errors and
ignored clustering; a parallel-trends test that rejected valid designs; event-study
windows pooling out-of-window periods into the reference; placeholder diagnostics that
always passed; row-order-dependent spillover results; a Moran's I variance that was
always negative; NaN p-values reported as "no spillover".

## [0.1.0] - 2025-12-15

### Added

- **Core DiD Methods (Phase 1)**
  - Two-way fixed effects (TWFE) estimation
  - Event study designs with pre-trend testing
  - Parallel trends testing
  - Core data structures (`TreatmentPanel`, `DiDEstimate`)
  - Data validation and preprocessing utilities

- **SUTVA Diagnostics (Phase 3)**
  - Spatial spillover detection
  - Network spillover analysis
  - Distance matrix computation
  - Neighbor treatment correlation tests
  - Spillover gradient estimation

- **IV/LATE Methods (Phase 2)**
  - Two-stage least squares estimation
  - First-stage diagnostics (F-statistics, partial R²)
  - Compliance analysis and complier characterization
  - Monotonicity testing
  - Exclusion restriction testing
  - Sensitivity analysis
  - Weak-IV robust inference (Anderson-Rubin)
  - External validity testing

- **Web-based GUI**
  - Interactive data upload and preview
  - Point-and-click DiD analysis
  - Real-time visualizations with Plotly
  - Session management
  - Result export functionality
  - Parallel trends and SUTVA diagnostics interface

- **Documentation**
  - Comprehensive tutorial
  - Methods documentation with mathematical formulations
  - GUI user guide
  - API reference
  - Working examples for all major features

- **Examples**
  - Basic DiD demonstration
  - SUTVA diagnostics workflow
  - LATE estimation workflow

### Documentation

- Complete Documenter.jl setup
- GitHub Pages deployment
- Auto-generated API documentation

## [0.0.1] - 2025-12-13

### Added

- Initial project structure
- Basic package scaffolding
- Literature review organization

---

## Version History Notes

### Versioning Strategy

- **Major version (1.0.0)**: Breaking changes to public API
- **Minor version (0.1.0)**: New features, backward compatible
- **Patch version (0.0.1)**: Bug fixes, documentation

### Breaking Changes Policy

We will avoid breaking changes in minor versions (0.x.0) whenever possible. When breaking changes are necessary, they will be:

1. Clearly documented in the changelog
2. Announced in advance
3. Accompanied by migration guides
4. Deprecated gradually when feasible

### Deprecation Process

Deprecated features will:

1. Trigger warnings for one minor version
2. Be removed in the following minor version
3. Be documented in changelog and docs

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for how to contribute to DrSnow.

## Questions?

- **Bug reports**: [GitHub Issues](https://github.com/simoneSantoni/DrSnow_alpha/issues)
- **Feature requests**: [GitHub Issues](https://github.com/simoneSantoni/DrSnow_alpha/issues)
- **General questions**: [GitHub Discussions](https://github.com/simoneSantoni/DrSnow_alpha/discussions)
