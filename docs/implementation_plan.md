> **Historical planning note (v0.1, 2025).** Superseded by the 0.2 rewrite; every method listed here is now implemented or deliberately out of scope. See `CHANGELOG.md` and the method guides in `docs/src/` for the current state.

# DrSnow Implementation Plan

A Julia library for harnessing natural experiments in social science research, providing cutting-edge causal inference methods with a focus on efficiency, visualization, and accessibility.

## User Review Required

> [!IMPORTANT]
> **Phased Development Approach**: This is a substantial project that will be developed in multiple phases. I recommend starting with Phase 1 (foundational infrastructure) to establish the package structure and a minimal working implementation before expanding to additional features.

> [!IMPORTANT]
> **GUI Framework Decision**: For the GUI application requirement, I need your input on the preferred approach:
>
> - **Web-based GUI** (Dash.jl/Genie.jl): More portable, runs in browser, easier deployment
> - **Desktop GUI** (GTK.jl/Qt.jl via QML): Native desktop application, potentially better performance
> - **Jupyter-based** (Pluto.jl notebooks): Interactive notebooks with reactive cells
>
> Please indicate which approach aligns best with your target users and deployment scenarios.

> [!IMPORTANT]
> **ML Backend**: For ML-powered HTE, we'll integrate with MLJ.jl. However, some advanced methods (like Generalized Random Forests) may require interfacing with R or implementing from scratch. Should I:
>
> - Prioritize pure Julia implementations (more work, better performance)
> - Allow R/Python interop for specialized methods (faster to implement, adds dependencies)

## Proposed Changes

This implementation will be organized in **5 phases**, starting with foundational infrastructure and progressively adding advanced features.

---

### Phase 1: Foundation & Basic DiD

#### [NEW] [Project.toml](file:///home/simon/githubRepos/DrSnow/Project.toml)

Initialize Julia package with:

- Package metadata (name, UUID, version, authors)
- Core dependencies: DataFrames.jl, Statistics.jl, LinearAlgebra.jl, StatsBase.jl, GLM.jl, FixedEffectModels.jl
- Testing dependencies: Test.jl, Random.jl

#### [NEW] [src/DrSnow.jl](file:///home/simon/githubRepos/DrSnow/src/DrSnow.jl)

Main module file that:

- Defines the DrSnow module
- Includes all submodules
- Exports public API functions
- Defines core data structures: `TreatmentPanel`, `DiDEstimate`, `DiDConfig`

#### [NEW] [src/core/data_structures.jl](file:///home/simon/githubRepos/DrSnow/src/core/data_structures.jl)

Common data structures:

- `TreatmentPanel`: Struct for panel data with treatment indicators
- `CausalEstimate`: Abstract type for all estimation results
- Data validation functions
- Preprocessing utilities

#### [NEW] [src/did/twoway_fe.jl](file:///home/simon/githubRepos/DrSnow/src/did/twoway_fe.jl)

Standard two-way fixed effects DiD:

- `did_twfe()`: Main estimation function
- Parallel trends testing
- Standard errors (clustered, robust)
- Coefficient extraction and formatting

#### [NEW] [src/did/event_study.jl](file:///home/simon/githubRepos/DrSnow/src/did/event_study.jl)

Event study designs:

- `event_study()`: Dynamic treatment effects
- Relative time period construction
- Pre-trend testing

#### [NEW] [test/runtests.jl](file:///home/simon/githubRepos/DrSnow/test/runtests.jl)

Test suite setup with test sets for each module

#### [NEW] [test/did/test_twoway_fe.jl](file:///home/simon/githubRepos/DrSnow/test/did/test_twoway_fe.jl)

Unit tests for two-way FE DiD:

- Synthetic data generation
- Numerical accuracy tests against known results
- Edge case handling

---

### Phase 2: Advanced DiD & Synthetic Control

#### [NEW] [src/did/callaway_santanna.jl](file:///home/simon/githubRepos/DrSnow/src/did/callaway_santanna.jl)

Callaway & Sant'Anna (2021) staggered DiD:

- Group-time average treatment effects
- Aggregation schemes (simple, dynamic, calendar)
- Never-treated and not-yet-treated comparison groups

#### [NEW] [src/synth/synthetic_control.jl](file:///home/simon/githubRepos/DrSnow/src/synth/synthetic_control.jl)

Synthetic control method (Abadie et al.):

- Optimization for synthetic control weights
- Inference via placebo tests
- In-space placebo tests

#### [NEW] [src/synth/synthetic_did.jl](file:///home/simon/githubRepos/DrSnow/src/synth/synthetic_did.jl)

Synthetic DiD (Arkhangelsky et al.):

- Combined synthetic control and DiD approach
- Regularized weights
- Robust variance estimation

#### [NEW] [test/did/test_callaway_santanna.jl](file:///home/simon/githubRepos/DrSnow/test/did/test_callaway_santanna.jl)

Tests for staggered DiD implementation

#### [NEW] [test/synth/test_synthetic_control.jl](file:///home/simon/githubRepos/DrSnow/test/synth/test_synthetic_control.jl)

Tests for synthetic control methods

---

### Phase 3: Heterogeneous Treatment Effects

#### [NEW] [src/hte/basic_hte.jl](file:///home/simon/githubRepos/DrSnow/src/hte/basic_hte.jl)

Standard HTE estimation:

- Subgroup analysis
- Conditional average treatment effects (CATE)
- Interaction effects

#### [NEW] [src/hte/ml_learners.jl](file:///home/simon/githubRepos/DrSnow/src/hte/ml_learners.jl)

ML-powered meta-learners:

- S-Learner: Single model approach
- T-Learner: Two model approach  
- X-Learner: Cross-fitted learner
- Integration with MLJ.jl for flexible model choice

#### [NEW] [src/hte/causal_forest.jl](file:///home/simon/githubRepos/DrSnow/src/hte/causal_forest.jl)

Forest-based methods:

- Causal forest implementation or R interop
- Honest splitting
- Variable importance measures

#### [NEW] [src/hte/dml.jl](file:///home/simon/githubRepos/DrSnow/src/hte/dml.jl)

Double/Debiased Machine Learning:

- Cross-fitting infrastructure
- Nuisance parameter estimation
- Debiased effect estimation
- Neyman orthogonality

#### [NEW] [test/hte/test_ml_learners.jl](file:///home/simon/githubRepos/DrSnow/test/hte/test_ml_learners.jl)

Tests for ML-based HTE estimation

---

### Phase 4: SUTVA & LATE Assessment

#### [NEW] [src/diagnostics/sutva.jl](file:///home/simon/githubRepos/DrSnow/src/diagnostics/sutva.jl)

SUTVA violation assessment:

- Spillover effect detection
- Spatial correlation tests (Moran's I, Geary's C)
- Network interference models
- Randomization inference for spillovers

#### [NEW] [src/iv/late.jl](file:///home/simon/githubRepos/DrSnow/src/iv/late.jl)

LATE and IV estimation:

- Two-stage least squares (2SLS)
- First-stage diagnostics (F-statistics)
- Compliance rate estimation
- Complier characteristics

#### [NEW] [src/iv/late_diagnostics.jl](file:///home/simon/githubRepos/DrSnow/src/iv/late_diagnostics.jl)

LATE boundary conditions:

- Monotonicity testing
- Exclusion restriction tests
- Sensitivity analysis for violations
- External validity assessment

#### [NEW] [test/diagnostics/test_sutva.jl](file:///home/simon/githubRepos/DrSnow/test/diagnostics/test_sutva.jl)

Tests for SUTVA diagnostics

#### [NEW] [test/iv/test_late.jl](file:///home/simon/githubRepos/DrSnow/test/iv/test_late.jl)

Tests for IV/LATE estimation

---

### Phase 5: Visualization & GUI

#### [NEW] [src/viz/plots.jl](file:///home/simon/githubRepos/DrSnow/src/viz/plots.jl)

Visualization infrastructure using Makie.jl:

- Event study plots with confidence intervals
- Parallel trends diagnostics
- Effect heterogeneity visualization
- Placebo test plots
- Balance plots
- Custom themes for publication-quality figures

#### [NEW] [src/viz/recipes.jl](file:///home/simon/githubRepos/DrSnow/src/viz/recipes.jl)

Plot recipes for common visualizations:

- `@recipe` macros for Plots.jl integration
- Consistent styling across plot types

#### [NEW] [src/gui/app.jl](file:///home/simon/githubRepos/DrSnow/src/gui/app.jl)

GUI application (framework TBD based on user preference):

- Data import interface
- Method selection panel
- Parameter configuration
- Real-time result visualization
- Export functionality (plots, tables, LaTeX)

#### [NEW] [src/gui/components.jl](file:///home/simon/githubRepos/DrSnow/src/gui/components.jl)

Reusable GUI components:

- Data browser
- Result tables
- Interactive plots

---

### Documentation

#### [NEW] [docs/make.jl](file:///home/simon/githubRepos/DrSnow/docs/make.jl)

Documenter.jl build script

#### [NEW] [docs/src/index.md](file:///home/simon/githubRepos/DrSnow/docs/src/index.md)

Main documentation landing page with overview and quick start

#### [NEW] [docs/src/tutorials/](file:///home/simon/githubRepos/DrSnow/docs/src/tutorials/)

Tutorial directory with examples:

- `basic_did.md`: Simple DiD example
- `event_study.md`: Event study walkthrough
- `synthetic_control.md`: Synthetic control example
- `hte_analysis.md`: HTE with ML example
- Each tutorial uses both synthetic and real-world datasets

#### [NEW] [docs/src/api/](file:///home/simon/githubRepos/DrSnow/docs/src/api/)

API reference documentation:

- Auto-generated from docstrings
- Organized by module

#### [NEW] [docs/src/methods/](file:///home/simon/githubRepos/DrSnow/docs/src/methods/)

Methodological background:

- Mathematical foundations
- Assumptions and diagnostics
- Literature references
- When to use each method

#### [MODIFY] [README.md](file:///home/simon/githubRepos/DrSnow/README.md)

Expand with:

- Installation instructions
- Quick start example
- Feature overview with links to docs
- Citation information
- Contributing guidelines
- Badges (CI status, coverage, docs)

---

### Infrastructure

#### [NEW] [.github/workflows/CI.yml](file:///home/simon/githubRepos/DrSnow/.github/workflows/CI.yml)

GitHub Actions for:

- Running tests on multiple Julia versions (1.9, 1.10, nightly)
- Code coverage reporting
- Documentation deployment

#### [NEW] [.github/workflows/Documentation.yml](file:///home/simon/githubRepos/DrSnow/.github/workflows/Documentation.yml)

Automatic documentation build and deployment to GitHub Pages

#### [MODIFY] [.gitignore](file:///home/simon/githubRepos/DrSnow/.gitignore)

Update for Julia-specific patterns:

- Manifest.toml (optional, good practice for libraries)
- docs/build/
- *.jl.cov, *.jl.*.cov,*.jl.mem

#### [NEW] [LICENSE](file:///home/simon/githubRepos/DrSnow/LICENSE)

Open source license (suggest MIT or Apache 2.0)

## Verification Plan

### Phase 1 Verification

**Automated Tests:**

```bash
# From repository root
julia --project=. -e 'using Pkg; Pkg.test()'
```

This will run all tests in `test/runtests.jl` including:

- Data structure validation tests
- Two-way FE DiD numerical accuracy (compare against simulated data with known ATT)
- Event study coefficient extraction

**Manual Verification:**

1. Create a simple test script `examples/basic_did_demo.jl`:

   ```julia
   using DrSnow, DataFrames
   
   # Generate simple 2x2 DiD data
   data = generate_test_panel(n_units=100, n_periods=10, treatment_period=6)
   
   # Estimate DiD
   result = did_twfe(data, :outcome, :treated, :unit_id, :time_period)
   
   # Display results
   println(result)
   ```

2. Run: `julia --project=. examples/basic_did_demo.jl`
3. Verify output shows reasonable ATT estimate and standard errors

### Phase 2-4 Verification

**Automated Tests:**
Same command as Phase 1, tests will accumulate in test suite

**Numerical Accuracy:**

- Compare DiD estimates against Stata/R implementations on same synthetic dataset
- Verify synthetic control weights sum to 1 and are non-negative
- Check LATE estimates match 2SLS manual calculation

### Phase 5 Verification

**Visualization Tests:**

```bash
julia --project=. test/viz/visual_tests.jl
```

This will generate sample plots and save to `test/viz/outputs/` for manual inspection

**GUI Testing:**

1. Launch GUI: `julia --project=. -e 'using DrSnow; launch_gui()'`
2. Load sample dataset from `examples/data/`
3. Run DiD analysis via GUI
4. Verify results match command-line version
5. Export plot and verify file creation

**Documentation Build:**

```bash
julia --project=docs docs/make.jl
```

Check that documentation builds without errors and contains all expected pages

### Performance Benchmarking

After Phase 1-3 completion:

```bash
julia --project=. benchmark/benchmarks.jl
```

Benchmarks will compare DrSnow performance against:

- R packages (fixest, did, gsynth)
- Python packages (linearmodels, CausalML)

Target: Within 2x performance of specialized packages, faster than Python equivalents

### Integration Testing

**End-to-end workflow test:**

```julia
# Test complete workflow from data → analysis → visualization
using DrSnow, CSV

data = CSV.read("examples/data/labor_market.csv", DataFrame)
result = did_twfe(data, ...)
plot_event_study(result)
```

All integration tests will be in `test/integration/` and run as part of CI.
