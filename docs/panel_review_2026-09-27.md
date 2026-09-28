# DrSnow: Panel Assessment (2026-09-27)

Six independent reviewers examined DrSnow v0.1.0 at commit `722867c`:

| Reviewer | Lens |
|---|---|
| Software engineering / Julia | Packaging, idioms, tests, CI, GUI, security |
| DiD econometrics | TWFE, event studies, pre-trend tests, staggered adoption |
| IV / LATE econometrics | 2SLS, weak-IV inference, compliance, LATE diagnostics |
| Interference / SUTVA | Spatial and network spillover tools |
| Causal ML | Readiness for ML estimators, ecosystem positioning |
| Natural-experiments methodology | Scientific framing, design coverage, docs, usability |

Every reviewer worked read-only on a scratch copy and ran Monte Carlo simulations
where a statistical claim could be checked. Findings marked **[V]** were verified
by running code. Findings marked **[R]** were inferred from reading. Findings
reported independently by two or more reviewers are marked **(×n)**.

---

## 1. Verdict

DrSnow has a clean layout, good docstring habits, a thoughtful literature base,
and a correct point estimator for the textbook cases (2×2 TWFE and just-identified
2SLS both match FixedEffectModels to machine precision). It is **not yet safe for
applied research.** Most inferential outputs are wrong, several diagnostics are
placeholders that report "pass", and the shipped examples print conclusions that
contradict their own data-generating processes.

The problems are concentrated and fixable. Most have a short, known fix that
delegates to FixedEffectModels, which is already a dependency. The strategic
opportunity is real: there is no Julia implementation of Callaway–Sant'Anna,
HonestDiD, regression discontinuity (rdrobust-style), doubly-robust DiD, or
design-based interference estimators.

### Current state at a glance

| Area | Status |
|---|---|
| Test suite | 100 tests: 90 pass, 2 fail, 8 error on Julia 1.10 and 1.13 [V]. Seven exported functions never execute in the suite. |
| CI | Docs only. No test workflow, so the red suite is invisible. |
| Examples | DiD and LATE demos run but print wrong conclusions. SUTVA demo crashes. [V] |
| GUI | Never loaded by `using DrSnow`. Cannot serve requests on Julia ≥1.11. Every analysis endpoint errors on 1.10. [V] |
| Security | Arbitrary code execution through column names, reachable from CSV uploads. [V] |
| Advertised but absent | Synthetic control, synthetic DiD, HTE/ML, plotting, `did_callaway_santanna`, `plot_event_study`, `DrSnow.SUTVA`, `detect_spillovers`. |

---

## 2. Critical problems (fix before anyone uses the package)

### 2.1 Arbitrary code execution through column names (×4) [V]

Seven call sites build model formulas as strings and run them through
`eval(Meta.parse("@formula(" * ... * ")"))`:

- `src/did/twoway_fe.jl:66`, `:175`
- `src/did/event_study.jl:99`
- `src/sutva/spatial_analysis.jl:268`
- `src/sutva/spillover_tests.jl:220`
- `src/sutva/network_diagnostics.jl:152`, `:263`

A column named `outcome ~ treatment) + (@eval Main (println("INJECTED"))) + (0`
executed code inside `did_twfe`. The GUI passes uploaded CSV headers straight into
these functions. The same mechanism breaks ordinary names such as `log wage`,
prevents precompilation, and makes `formula::Any`.

**Fix:** build formulas programmatically with `term(y) ~ term(d) + fe(unit) + fe(time)`
and read coefficients by name. `src/iv/late.jl` already uses `term()`.

### 2.2 `late_2sls` standard errors are wrong and clustering is ignored (×5) [V]

`_compute_2sls_se` (`src/iv/late.jl:455-506`) is an invented formula. It computes
residuals from the fitted treatment rather than the actual treatment, then divides
by a function of the first-stage F. The error depends on first-stage strength, so
SEs can be too large or too small.

| Design | Reported SE vs correct | 95% CI coverage |
|---|---|---|
| Moderate first stage | 2.4× to 6.5× too large | 1.00 |
| Very strong first stage (R² ≈ 0.96) | too small | **0.30** |

`cluster_var` is accepted and documented but only used to drop missing rows. SEs
are bit-identical with and without it. The LATE demo reports SE 2.85 and "not
significant" where the correct SE is 0.31 with t ≈ 11.

**Fix:** delegate to `FixedEffectModels.reg(df, @formula(y ~ x + (d ~ z)), Vcov.cluster(:g))`.

### 2.3 `parallel_trends_test` rejects valid designs whenever there is an effect (×4) [V]

The regression includes leads only. All post-treatment observations fall into the
omitted baseline, so the treatment effect loads onto the leads.

| Scenario (parallel trends true) | Rejection rate |
|---|---|
| No treatment effect | 0.06 |
| Treatment effect of 2 or 5 | **1.00** |

The shipped `basic_did_demo.jl` prints "Parallel trends assumption rejected"
(F = 178) on data where parallel trends holds by construction. The function is
also exposed in the GUI.

### 2.4 Event-study window silently pools out-of-window periods into the reference (×4) [V]

Event times outside `[-max_pre, max_post]` get all-zero dummies and join the
omitted period. With a true dynamic effect of 10 + k, the full window recovers it
exactly, but a (−2, 2) window returns −4.5 at k = −2 and understates every post
coefficient. Pre-trend tests with a trimmed window reject 33–40% of the time under
the null. This is also why `test/did/test_event_study.jl:44` fails.

Related [V]: an `omit_period` outside the window, or all units treated at once,
yields coefficients of 0.0 with SE NaN and no error.

### 2.5 Placeholder diagnostics that report a pass (×4) [V]

| Function | What happens |
|---|---|
| `external_validity_test` | Calls `include("late.jl")` at runtime (`late_diagnostics.jl:444`) and crashes. If it ran, it would return `externally_valid = true`, `p = 1.0` regardless of input. |
| `test_exclusion_restriction` | Main test is hard-coded to statistic 0, p = 1.0. |
| `complier_characteristics` | Regresses X on Z, which is an instrument balance test. It has only nominal power to detect complier differences (rejects 5.5% when compliers differ by 0.7 SD). The demo prints "Compliers appear similar to population". |
| `test_monotonicity` | Checks the sign of the pooled first stage. It missed 20% defiers and flagged a valid decreasing instrument as "evidence of defiers". |

### 2.6 SUTVA module: row misalignment, unidentified regressions, crashes (×3) [V]

- **Unit ordering.** Matrix rows are matched to units under three conventions:
  order of first appearance, `groupby` order, and implicit coordinate order.
  Shuffling data rows changed a spillover estimate from 1.00 to 0.03 and a network
  estimate from about 1 to −0.01. Nothing documents or validates which unit a
  coordinate or adjacency row belongs to.
- **Unidentified by construction.** `placebo_spillover_test` uses a unit-invariant
  regressor alongside unit fixed effects, so it is absorbed in 100% of runs. All
  test and demo DGPs use time-invariant treatment, which makes every FE spillover
  regression unidentified.
- **NaN reported as reassurance.** Interpretation blocks branch on `p < 0.05`, so a
  NaN p-value prints "No significant spillover detected".
- **Moran's I always crashes.** The variance formula in `neighbor_treatment_correlation`
  mixes two textbook formulas, goes negative, and raises `DomainError` in every run.
- **Treated status from `mode(D)`.** In a standard 4-pre / 4-post design, treated
  units are classified as controls.
- **`network_exposure` counts walks, not hops.** On a clustered graph with only
  1-hop spillovers, the 2-hop test rejects 39% of the time.

### 2.7 GUI does not work (×2) [V]

- `using DrSnow` never loads it: the package-extension branch in `src/DrSnow.jl` is
  empty and there is no `ext/`.
- The `Genie = "3.0.0"` compat pins HTTP 0.9, which cannot serve any request on
  Julia ≥1.11.
- On Julia 1.10: every error path calls a nonexistent `json(data, status)` method;
  upload reads a JSON payload from a multipart request; the results table reads
  fields that do not exist; the SUTVA route calls methods that do not exist.
- Frontend inserts CSV headers and values into `innerHTML` unescaped (XSS).

---

## 3. High-severity problems

**DiD**

- `n_control` counts treated units' pre-periods, so it roughly doubles. This makes
  `test/did/test_twoway_fe.jl:51` fail (×4) [V].
- **Staggered adoption goes undetected.** With all-positive growing effects, TWFE
  returned −0.69 for a true ATT of 4.36. dCDH weights had 80 of 440 treated cells
  negative. The event study showed leads of about 2.4 where the truth is 0. No
  warning is issued [V].
- Pre-trend Wald statistics ignore the covariance of the leads (×4) [V].
- CIs are hard-coded to 1.96. With 8 clusters, coverage is 0.83 versus 0.88 using
  t(G−1). There is no `level` argument [V].
- `balance_check` always returns NaN for the treated mean. It is unused and
  unexported [V].

**IV / LATE**

- `weak_iv_inference` uses a fixed 100-point grid on [−10, 10] and reports the hull.
  Coverage is 0.23 with a strong instrument and 0.46 when the effect is 15. It never
  reports unbounded or two-ray sets correctly (×4) [V].
- `first_stage_diagnostics` uses a homoskedastic F and mislabels the rule of thumb
  as Stock–Yogo. One heteroskedastic design gave F = 87 where the effective F is
  10 [V].
- `estimate_compliance` accepts continuous treatments and decreasing instruments,
  producing a rate of −0.51 and "1157 compliers" for years of education [V].
- `sensitivity_analysis` applies symmetric bounds to one-signed violations and
  ignores sampling error, so the demo prints "Robust" for an estimate with
  p = 0.24 [V].

**Interference**

- `balance_test_neighbors` over-rejects at 15–20% under random assignment, and the
  rate grows with N. It never uses covariates despite its docstring [V].
- `spatial_spillover_gradient` reports control outcome levels by distance, not an
  effect. It produces a tight "gradient" when there is no treatment at all [V].
- Spillover SEs ignore spatial and network dependence. Size is 10–15% under
  correlated shocks [V].
- `estimate_spillover_by_degree` confuses count-versus-share exposure
  misspecification with hub heterogeneity. It rejects 97% of the time under a
  count DGP [V].

**Engineering**

- Genie, CSV, JSON3, UUIDs and Dates are hard dependencies the core never uses.
  They pull roughly 30 packages, including HTTP 0.9, into every install (×3) [V].
- No compat bounds for Distributions or the stdlibs. Aqua reports 3 failures [V].
- `SpatialPanel` and `NetworkPanel` docstrings claim `<: TreatmentPanel`, but they
  are unrelated mutable structs that copy its fields. `did_twfe(::SpatialPanel)` is
  a MethodError [V].
- `EventStudyEstimate` is used by tests but not exported [V].
- The Dockerfile copies a `Manifest.toml` that is gitignored and dockerignored, so
  a clean build fails [R].

---

## 4. Documentation and communication

- **Overclaiming.** The module docstring and README list five capability areas. Two
  of them, synthetic control and HTE/ML, have no code. "Sophisticated plotting" does
  not exist. "10–100× faster" has no benchmark. "Numerically identical to reghdfe"
  is not true of DrSnow's own CIs or pre-trend tests.
- **Code that does not run.** README and tutorial snippets call nonexistent
  functions and submodules, keyword constructors that do not exist, and scalar
  arithmetic on vectors. The `index.md` quick start builds an all-zero treatment
  and reports ATT 0.0 without error.
- **Dead links.** The README points to a Docker guide, a GUI guide, and six website
  pages that do not exist. Its CI badge points to a nonexistent workflow.
- **Language that turns non-rejection into evidence.** Examples include "Treatment
  appears randomly assigned", "Balance appears adequate", "excludable", "Robust to
  modest violations", and "Parallel trends assumption appears satisfied".
- **`algorithms.md`** describes several functions differently from what the code
  does (Moran's I, spatial gradient, complier characteristics, parallel trends).
- **Weak tests.** They check types, non-NaN values and `atol = 1.5` on point
  estimates. No SE, coverage, or reference-implementation comparison exists, which
  is why none of the above was caught.
- **Repository hygiene.** `didMethodPapers/` tracks 27 publisher PDFs, a
  redistribution risk. `literature/` has placeholder citations (`arXiv:2212.xxxxx`)
  and at least one likely misattributed journal.

---

## 5. Architecture assessment

The present abstractions will not carry the roadmap:

- **No shared result interface.** Point estimates are called `att`, `estimate`,
  `direct_effect` or `coefficients`. Only DiD results subtype `CausalEstimate`.
  Results store a scalar SE, not a vcov. Nothing implements StatsAPI, so
  `coef(result)` fails and RegressionTables.jl cannot consume results.
- **No shared panel interface.** `TreatmentPanel` is concrete, binary-only and
  panel-only. The spatial and network panels duplicate it. IV ignores it.
- **No cohort or timing representation**, which every modern DiD estimator needs.
- **No RNG plumbing.** Adding `rng` keywords now costs nothing because nothing
  stochastic exists yet.
- **Inconsistent API.** `test_exclusion_restriction` swaps the order of the
  instrument and treatment arguments. Keywords are named both `cluster` and
  `cluster_var`. Covariates are sometimes positional.

**Recommended target design**

```julia
abstract type AbstractDesign end          # roles: y, d, z, x, cluster, weights
struct PanelDesign <: AbstractDesign      # + unit, time, cohort (G_i)
struct InterferenceStructure              # unit-keyed coordinates or sparse adjacency

abstract type CausalEstimate <: StatsAPI.StatisticalModel end
# every result: coef, vcov, stderror, confint(; level), nobs, coeftable, pvalue

fit(Estimator, design; vcov = Vcov.cluster(:unit), rng = default_rng())
```

Every stochastic function should take an `rng` keyword and store its seed. Tests
should use StableRNGs.

---

## 6. Roadmap

### P0: stop misleading users (target v0.1.1)

1. Replace `eval(Meta.parse)` with programmatic `term`/`fe` formulas.
2. Rebuild `late_2sls` on FixedEffectModels IV syntax with robust and clustered vcov.
   Report effective F.
3. Remove `parallel_trends_test`, or re-implement it as a joint Wald test on a fully
   dynamic event study. Bin event-study endpoints. Use the full-vcov Wald statistic.
   Use `confint` with t(G−1).
4. Un-export or clearly mark as experimental: `external_validity_test`,
   `test_exclusion_restriction`, `test_monotonicity`, `placebo_spillover_test`,
   `neighbor_treatment_correlation`. Rename `complier_characteristics` to
   `instrument_balance`, or implement Abadie κ.
5. Fix `n_control`, `mode(D)` classification, unit-keyed matrix alignment, and
   silent 0.0/NaN returns. Throw an error when the key regressor is dropped.
6. Export `EventStudyEstimate`. Fix test imports. Add a CI test workflow.
7. Trim the README and module docstring to what exists. Fix or remove dead links
   and broken snippets. Rewrite non-rejection language.
8. Move the GUI into a package extension, or disable it. Drop unused hard
   dependencies. Add compat bounds. Raise the minimum Julia to 1.10.
9. Remove the tracked PDFs from the repository.

### P1: credible core (v0.2)

- StatsAPI result interface with `level` and `vcov` keywords, and RegressionTables
  compatibility.
- Cohort and timing layer. Goodman-Bacon decomposition and a dCDH negative-weight
  warning in `did_twfe`.
- Callaway–Sant'Anna with doubly-robust 2×2 kernels (Sant'Anna–Zhao), four
  aggregations, and multiplier bootstrap. Sun–Abraham as the default multi-cohort
  event study.
- Analytic Anderson–Rubin sets, effective F and the tF procedure.
- Event-study plotting via a Makie package extension.
- Validation suite (section 7).

### P2: fulfil the natural-experiments mission (v0.3)

- **Regression discontinuity:** sharp, fuzzy and kink designs; MSE- and CER-optimal
  bandwidths; robust bias-corrected CIs; density manipulation tests; RD plots.
  All reviewers who addressed scope named this the largest gap.
- **Randomization inference:** Fisher exact tests under the actual assignment
  mechanism. This also gives valid replacements for the spillover balance and
  Moran tests.
- **Design-based interference:** unit-keyed exposure mappings, Aronow–Samii
  Horvitz–Thompson estimators, Butts ring DiD, and Conley spatial HAC SEs.
- HonestDiD (Rambachan–Roth) sensitivity, using JuMP and HiGHS as weak dependencies.
- Imputation DiD (Borusyak–Jaravel–Spiess), and dCDH for switching treatments.

### P3: extensions (v0.4+)

- **Causal ML:** DML for the partially linear, interactive and IV models, including
  LATE with ML nuisances; DML-DiD; DR-learner with BLP/GATES/CLAN; policy learning;
  prediction-powered inference for ML-coded outcomes. Learners plug in through an
  MLJ package extension.
- **More IV designs:** judge/examiner designs (leave-one-out leniency, UJIVE,
  Frandsen–Lefgren–Leslie test); shift-share IV (Rotemberg weights, AKM and BHJ
  inference); LIML and Fuller for many instruments; marginal treatment effects;
  Kitagawa and Mourifié–Wan instrument-validity tests.
- **Synthetic control and synthetic DiD:** interoperate with SynthControl.jl and
  TreatmentPanels.jl rather than duplicating them. Resolve the `TreatmentPanel`
  name clash.
- Defer causal forests. No Julia GRF exists, and a DecisionTree.jl approximation
  would give invalid CIs.

### Positioning

Existing Julia packages cover part of this space: DiffinDiffs.jl and
EventStudyInteracts.jl (Sun–Abraham), SynthControl.jl, DiDInt.jl, and CausalELM.jl
(meta-learners on extreme learning machines). DrSnow's defensible niche is
**design-based natural-experiment inference with honest diagnostics**:
Callaway–Sant'Anna, HonestDiD, RD, interference-aware DiD, and ML-nuisance
estimators for DiD and LATE. None of these currently exists in Julia.

---

## 7. Proposed validation suite

1. Card–Krueger (1994) 2×2: match `fixest` / `reghdfe` coefficients and SEs.
2. `mpdta` from R `did`: TWFE and binned event study now; `att_gt`/`aggte` and
   HonestDiD later.
3. Angrist–Krueger (1991) quarter of birth: match `ivreg` / `ivreghdfe` on 2SLS β,
   SEs (iid, HC1, cluster), effective F and the AR set.
4. Angrist–Evans or JTPA: complier shares and κ-weighted complier means.
5. Lee (2008) Senate data for RD, once added: match `rdrobust` and `rddensity`.
6. Monte Carlo size and coverage checks in a slow test group, with reduced reps in
   CI: pre-trend test size, 2SLS coverage, AR coverage under weak instruments,
   spillover recovery.
7. Property tests: results must not change when data rows are shuffled.
8. Aqua.jl and JET.jl in the test suite.

---

## 8. Points of disagreement or uncertainty

- **Pre-trend degrees of freedom.** Several reviewers flagged `dof_residual` instead
  of G−1. The DiD reviewer verified that FixedEffectModels 1.13 already returns G−1
  under clustering, so the df is correct by accident of the dependency. The
  covariance problem stands regardless.
- **Size of the 2SLS SE error.** Reported inflation ranged from 2.4× to 6.5×. The IV
  reviewer showed the error depends on first-stage R², and flips sign above 0.5.
  All reviewers agree the formula is wrong.
- **Dockerfile healthcheck.** Whether the base image ships `curl` was not verified.
- **Effort estimates.** Items P0 through the start of P1 were estimated at 6–10
  focused weeks for one developer. That is a reviewer judgment, not a measurement.
