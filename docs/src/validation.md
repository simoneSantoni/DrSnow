# Validation

```@meta
CurrentModule = DrSnow
```

DrSnow's estimators are checked in two ways, both part of the test suite:

1. **Reference comparisons.** Estimates, standard errors, test statistics and
   intervals are compared with established implementations, mostly R packages, on
   published or simulated data. Reference values and the scripts that generated them
   are committed under `test/validation/<area>/`, so the comparisons run without R.
2. **Monte Carlo checks.** Most inferential procedures (confidence intervals,
   bands, tests) have a size or coverage check on a simulated design with known truth.

The "Observed agreement" column reports the differences observed when the comparisons
were run for validation, relative unless stated otherwise; they are not the test
tolerances. The tests accept looser tolerances, which are given in the test files. The
methods guides give more detail and list the known convention differences from the
reference implementations.

## Reference implementations

| Area | DrSnow | Reference (version) | Data | Observed agreement |
|:--|:--|:--|:--|:--|
| DiD | `did_callaway_santanna`, `aggregate_att` (panel and repeated cross-sections) | did 2.5.1 | `mpdta`, simulated RC | ATT ≤ 5e-11, SE ≤ 2e-10 |
| DiD | `did_drdid` (5 methods, panel and RC, weights) | DRDID 1.3.0 | LaLonde/CPS, DRDID simulated | estimates ≤ 4e-9, SE ≤ 1e-9 |
| DiD | `did_twfe`, binned TWFE event study, `did_sun_abraham` | fixest 0.14.2 | `mpdta` | estimates ≤ 1e-14, SE within 0.02% |
| DiD | `did_imputation` | didimputation 0.5.1 | `mpdta` | estimates ≤ 5e-9, SE ≤ 1e-10 |
| DiD | `bacon_decomposition`, `twfe_weights` | bacondecomp 0.1.1, TwoWayFEWeights 2.1.0 | `mpdta` | exact |
| DiD | `honest_did` (ΔSD, ΔRM, ΔSDRM, sign/shape restrictions, FLCI) | HonestDiD 0.2.8 | `mpdta`, Benzarti–Carloni | identical accepted grid points (conditional); hybrid within 2 grid steps; FLCI half-length within 0.2% |
| DiD | `did_multiplegt_dyn` (17 settings) | DIDmultiplegtDYN 2.4.0 | simulated, Favara–Imbs | ≤ 1e-12 |
| DiD | `did_etwfe` + `aggregate_att` (linear, Poisson, logit) | etwfe 0.6.2, fixest 0.14.2, marginaleffects 1.0.0 | `mpdta` | linear ≤ 1e-13; Poisson/logit ≤ 1e-9 |
| IV | `iv_regression`: coefficients; iid, HC1, one- and two-way cluster SEs; FE and weights | AER, sandwich, fixest | Card (1995), Mroz (1987) | coefficients ≤ 1e-9, SE ≤ 1e-8 |
| IV | weak-IV F, Wu–Hausman, Sargan; AR and CLR tests and sets; tF | AER, ivmodel, ivDiag 1.0.6 (`tF`) | Card, Mroz | numerical precision |
| IV | LIML, Fuller, HLIM/HFUL, JIVE/UJIVE, judge designs, shift-share (AKM, AKM0), MTE | ivmodel, AER, fixest, ShiftShareSE, KernSmooth | simulated | numerical precision |
| IV | robust / cluster CLR and K tests and sets | independent R implementation (Kleibergen 2005) | Card | ≤ 1e-13 |
| IV | parametric `late_extrapolation` | GMM sandwich in R (numDeriv) | simulated | SE ≤ 1e-9 |
| IV | HHN variance for LIML/Fuller; CJIVE | explicit-matrix HHN formula; ManyIV (Kolesár); clusterIV `cjive` | simulated | HHN ≤ 1e-14 (ManyIV MD SE within 1.4%); CJIVE ≤ 1e-15 |
| IV | panel / overidentified `rotemberg_weights` | bartik.weight (GPSS) | ADH subset | ≤ 1e-11 |
| IV | `mte_bounds` (MST 2018) | ivmte 1.4.0 (lpSolveAPI) | AE, ivmteSimData | ≤ 1.5e-8 |
| RD | `rd_estimate` (75 cases), `rd_bandwidth` (10 selectors), `rd_plot_data` | rdrobust 4.0.0 | Senate, simulated | ≤ 1e-9 |
| RD | `rd_density_test`, `rd_density_bandwidth` | rddensity 3.0, lpdensity 3.0.1 | Senate, simulated | ≤ 1e-9 |
| RD | `rd_honest` (sharp, fuzzy, point; MSE/FLCI/OCI bandwidths; clusters, weights, covariates), critical values | RDHonest 1.0.2.9000 | Lee (2008), Head Start, simulated | ≤ 1e-9 at a given bandwidth; optimised bandwidth ~1e-8 |
| RD | `rd_honest_bme`, `rd_smoothness_bound` | RDHonest | CPS/CGHS samples | ≤ 1e-8 |
| RD | `rd_mccrary_test` | rdd 0.57 (`DCdensity`) | simulated | ≤ 1e-13 |
| RD | `stdvars` option | rdrobust 4.0.0 | simulated | ≤ 1e-12 |
| Synthetic control | `synthetic_did` (SDID, SC, DiD; placebo, bootstrap, jackknife) | synthdid 0.0.9 | Prop 99, simulated | ≤ 1e-7 |
| Synthetic control | `augmented_synthetic_control`, conformal inference | augsynth 0.2.0 | Prop 99 | numerical precision |
| Synthetic control | `synthetic_control` | Synth 1.1.10 | Basque Country | identical weights given Synth's `V` |
| Synthetic control | `matrix_completion` | MCPanel | Prop 99 | fits at fixed `λ` |
| Randomization inference | exact p-values, sharp nulls, designs | ri2 0.5.0, randomizr 2.0.1; `wilcox.test`, `p.adjust` | simulated | exact |
| Interference | Aronow–Samii Horvitz–Thompson and Hájek estimators and variances | interference 0.1.0 | simulated network | machine precision |
| Interference | `conley_vcov` | fixest (`vcov_conley`) | simulated spatial panel | machine precision |
| Causal ML | `dml_plr`, `dml_irm`, `dml_pliv`, `dml_iivm` with fixed folds | DoubleML 1.0.2 (mlr3) | simulated | ≤ 1e-10 |
| Causal ML | GRF post-estimation (`average_treatment_effect`, `best_linear_projection`, `test_calibration`, `rank_average_treatment_effect`, `double_robust_scores`) given identical nuisances | grf 2.6.1 | simulated (plain, clustered, weighted, continuous W, IV) | ≤ 1e-8 (141 checks) |
| Causal ML | `causal_forest`, `regression_forest`, `instrumental_forest` full fits | grf 2.6.1, 20 seeds each | simulated | all 36 summaries within \|z\| < 2.7 of grf's seed distribution; CATE seed-mean correlation 0.99997 |
| Causal ML | `dml_did` scores | DRDID | simulated panel and RC | exact point estimates |
| Causal ML | `dml_did_multi` with fixed folds (panel and RC; never / not-yet treated; anticipation) | DoubleML 0.11.4 (Python, `DoubleMLDIDMulti`) | simulated | ATT(g,t) and aggregates ≤ 1e-7; SE ≤ 1e-6 (Hájek influence function) |
| RD | `rd_flex` with fixed folds (sharp, fuzzy) | DoubleML 0.11.4 (Python, `RDFlex`), rdrobust 2.1.0 | simulated | ≤ 1e-9 (fuzzy ≤ 1e-7) |
| IV | `residual_prediction_test` (strong, weak at β₀; het/hom/cluster) | RPIV 1.1.1 (Scheidegger et al.) | simulated | ≤ 1e-15 |
| IV | `ml_first_stage` (cross-fitted F, partial R², effective F) | FixedEffectModels on out-of-fold residuals; `dml_irm` | simulated | ≤ 1e-8 |
| IV | `multiple_iv_weights` (Mogstad–Torgovitsky–Walters) | exact population decompositions (MTW Propositions 5–7) | constructed | ≤ 1e-10 |
| IV | `dml_lqte` (local potential quantiles, LQTE) | DoubleML 0.11.4 (Python, `DoubleMLLPQ`) | simulated | estimates ≤ 1e-12, SE ≤ 1e-9 |
| Causal ML | `dsl_regression`, `dsl_proportions` (linear, logit, fixed effects, clusters, unequal label probabilities) | dsl 0.1.0 (R) | simulated | ≤ 2e-5 (logit ≤ 1e-4; two-way FE matches the exact within estimator to 1e-9) |
| Sequential | Hoeffding, empirical-Bernstein, betting and normal-mixture confidence sequences | confseq 0.0.11 (Python) | simulated | ≤ 1e-15 |
| Sequential | group-sequential designs (spending families, classical bounds, futility) and analyses | gsDesign 3.11.0, rpact 4.4.0 | 72 designs, 4 analyses | bounds ≤ 1e-6 |
| Adaptive | adaptively weighted AIPW (two-point, constant allocation), contextual weights | authors' code (Hadad et al. 2021; Zhan et al. 2021) | Thompson-sampling experiment | ≤ 1e-10 |
| Adaptive | `wcls`, `emee` (micro-randomized trials) | MRTAnalysis 0.4.1 | HeartSteps mimic, simulated | estimates ≤ 5e-11 |
| Design | `power_means`, `power_proportions`, `power_cluster`, `power_blocked`, `power_rd` | pwr 1.3.0, PowerUpR 1.1.0, rdpower 3.0 | Senate (RD), analytic | ≤ 1e-8 |
| Design | `experiment_estimate`; greedy Mahalanobis pairs; `diagnose_design` | estimatr 2.0.0; blockTools 0.6.6; DeclareDesign 1.1.1 | simulated | ≤ 1e-10; identical pairs; diagnosands within 2.2 Monte Carlo SE |

Where no R version is listed, the generating script does not record one; the R
version used for the recorded references is R 4.6.1.

## Monte Carlo checks

Each area's test group includes Monte Carlo checks of the size of its tests and the
coverage of its intervals: for example, coverage of Callaway–Sant'Anna intervals and
uniform bands, size of pre-trend tests, coverage of the Anderson–Rubin set with weak
first stages, coverage of the robust RD interval, size of the density and
randomization tests, coverage of SDID placebo, bootstrap and jackknife intervals,
size of the studentized randomization test under a heterogeneous weak null, the
family-wise error rate of the Westfall–Young adjustment, and coverage of the DML
estimators. Replication counts use `mc_reps(full, fast)`: continuous integration runs
the reduced count, and the full count runs with

```bash
DRSNOW_SLOW_TESTS=true julia --project=. -e 'using Pkg; Pkg.test()'
```

A single area can be selected with `DRSNOW_TEST_GROUP` (comma-separated, e.g.
`DRSNOW_TEST_GROUP=did,rd`). The full Monte Carlo suite takes considerably longer
than the default run.

## Regenerating the references

The generating scripts live next to the data they produce:

| Area | Scripts |
|:--|:--|
| DiD | `test/validation/did/generate_did_references.R`, `generate_honestdid_references.R`, `generate_dcdh_references.R`, `generate_etwfe_references.R` |
| IV | `test/validation/iv/make_reference.R`, `make_reference_manyiv.R`, `make_reference_judge.R`, `make_reference_shiftshare.R`, `make_reference_mte.R`, `make_reference_robust_extrap.R`, `make_reference_hhn_cjive.R`, `make_reference_rotemberg.R`, `make_reference_mtebounds.R` |
| RD | `test/validation/rd/generate_reference.R`, `generate_reference_honest.R`; `make_flex_data.jl`, then `doubleml_rdflex_reference.py` (Python) |
| Synthetic control | `test/validation/synth/generate_references.R` |
| Randomization inference | `test/validation/ri/make_ri2_reference.R` (writes `ri2_reference.jl`) |
| Interference | `test/validation/sutva/generate_aronow_samii.jl` and `generate_conley.jl` (data), then `aronow_samii_reference.R` and `conley_reference.R` |
| Causal ML | `test/validation/ml/make_data.jl` (data and folds), then `doubleml_reference.R` and `drdid_reference.R`; `grf_reference.R` and `grf_forest_reference.R`; `make_did_multi_data.jl`, then `doubleml_did_multi_reference.py` (Python); `dsl_reference.R` |
| IV (ML diagnostics) | `test/validation/iv/make_reference_rpiv.R`, `make_reference_lqte.py` (Python) |
| Sequential | `test/validation/sequential/confseq_reference.py` (Python), `gs_reference.R` |
| Adaptive | `test/validation/adaptive/hadad_reference.py` (Python), `generate_mrt_references.R` |
| Design | `test/validation/design/make_reference_power.R`, `make_reference_experiments.R` |

Run them from the repository root. If the R packages are installed in a separate
library, point R to it with `R_LIBS_USER`:

```bash
R_LIBS_USER=/path/to/rlib Rscript test/validation/rd/generate_reference.R
```

Most scripts record the package versions they used (`*_versions.txt`,
`reference_versions.csv`, or a header comment). Datasets are written with 17
significant digits so that R and Julia work on identical inputs. After
regenerating, run the corresponding test group and review any change in the
committed CSV files.
