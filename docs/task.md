> **Historical planning note (v0.1, 2025).** Superseded by the 0.2 rewrite; every method listed here is now implemented or deliberately out of scope. See `CHANGELOG.md` and the method guides in `docs/src/` for the current state.

# DrSnow Implementation Task List

## Project Setup

- [x] Initialize Julia package structure with Project.toml and Manifest.toml
- [x] Set up src/ directory with main module file
- [x] Configure .gitignore for Julia projects
- [x] Create LICENSE file (open source)
- [x] Set up test/ directory structure

## Core Library Modules

### Difference-in-Differences (DiD)

- [x] Implement standard two-way fixed effects DiD
- [ ] Implement staggered adoption DiD (Callaway & Sant'Anna)
- [x] Implement event study designs
- [x] Add parallel trends testing functionality

### Synthetic DiD

- [ ] Implement synthetic control method (Abadie et al.)
- [ ] Implement synthetic DiD (Arkhangelsky et al.)
- [ ] Add optimization routines for weight calculations
- [ ] Implement placebo tests and inference

### Heterogeneous Treatment Effects (HTE)

- [ ] Implement standard HTE estimation (subgroup analysis)
- [ ] Implement conditional average treatment effects (CATE)
- [ ] Add forest-based methods (causal forests)
- [ ] Implement double/debiased machine learning for HTE

### ML-Powered HTE

- [ ] Integrate with MLJ.jl for model flexibility
- [ ] Implement X-learner, S-learner, T-learner
- [ ] Add cross-fitting for double machine learning
- [ ] Implement generic ML estimator (GRF, neural networks)

### SUTVA Violation Assessment

- [ ] Implement spillover effect detection
- [ ] Add network interference models
- [ ] Implement spatial correlation tests
- [ ] Create diagnostic plots for SUTVA violations

### LATE Boundary Conditions

- [ ] Implement instrumental variable (IV) estimation
- [ ] Add compliance testing and complier identification
- [ ] Implement monotonicity tests
- [ ] Add sensitivity analysis for LATE assumptions

## Visualization & Communication

- [ ] Create plotting recipes for Plots.jl/Makie.jl
- [ ] Implement event study plots
- [ ] Add parallel trends visualization
- [ ] Create effect heterogeneity plots
- [ ] Implement diagnostic plots (balance, placebo tests)
- [ ] Add customizable themes and export options

## GUI Application

- [ ] Research Julia GUI frameworks (Blink.jl, Dash.jl, or web-based)
- [ ] Design GUI layout and user workflow
- [ ] Implement data loading interface
- [ ] Create parameter configuration panels
- [ ] Add result visualization panels
- [ ] Implement export functionality

## Documentation

- [ ] Set up Documenter.jl
- [ ] Write comprehensive README
- [ ] Create getting started guide
- [ ] Write API reference documentation
- [ ] Develop concrete examples for each method
- [ ] Create tutorials with synthetic and real data
- [ ] Add mathematical background for each method

## Testing & Quality

- [ ] Write unit tests for core functions
- [ ] Create integration tests for workflows
- [ ] Add numerical accuracy tests against R/Python implementations
- [ ] Implement continuous integration (GitHub Actions)
- [ ] Set up code coverage tracking
- [ ] Performance benchmarking suite

## Additional Infrastructure

- [ ] Define common data structures (treatment indicators, outcomes, covariates)
- [ ] Implement data validation and preprocessing utilities
- [ ] Create bootstrapping and inference utilities
- [ ] Add parallel computation support
- [ ] Implement caching for expensive computations
