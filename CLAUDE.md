# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

DrSnow is a Julia package (pre-1.0, unregistered, Julia >= 1.10) for design-based causal
inference with natural experiments: DiD, IV, RD, synthetic control, randomization
inference, interference (SUTVA) and causal ML. Version 0.2 is a full rewrite after the
review in `docs/panel_review_2026-09-27.md`; `CHANGELOG.md` [Unreleased] lists what
exists and every breaking change from 0.1.

## Common Commands

```bash
# Install dependencies (Manifest.toml files are gitignored)
julia --project=. -e 'using Pkg; Pkg.instantiate()'

# Full test suite (reduced Monte Carlo counts, as in CI)
julia --project=. -e 'using Pkg; Pkg.test()'

# One or more test groups: quality, core, ri, did, iv, rd, sutva, synth, ml, viz, gui
DRSNOW_TEST_GROUP=did,iv julia --project=. -e 'using Pkg; Pkg.test()'

# Full Monte Carlo size/coverage replication counts (slow)
DRSNOW_SLOW_TESTS=true julia --project=. -e 'using Pkg; Pkg.test()'

# Build docs (strict: missing docstrings, bad @ref, failing @example blocks all error)
julia --project=docs -e 'using Pkg; Pkg.develop(PackageSpec(path=pwd())); Pkg.instantiate()'
julia --project=docs docs/make.jl

# Web GUI: from a clone (uses the app/ environment), or from any session
julia launch_gui.jl
julia -e 'using DrSnow, HTTP, JSON3, CSV; launch_gui()'
```

Test files rely on the `using` block and the `mc_reps` helper in `test/runtests.jl`;
run them through `DRSNOW_TEST_GROUP` rather than `include`-ing one file on its own.

CI: `.github/workflows/CI.yml` runs the tests (Julia 1.10 and current; Linux, macOS,
Windows); `.github/workflows/documenter.yml` builds the docs on Julia 1.10 and deploys
them to GitHub Pages from `master`.

## Architecture

- `src/DrSnow.jl` loads dependencies, re-exports the StatsAPI accessors (`coef`, `vcov`,
  `stderror`, `confint`, `coeftable`, `nobs`, ...) and `Vcov`, then includes one
  aggregator per area in a fixed order: `core`, `ri`, `did`, `iv`, `rd`, `sutva`,
  `synth`, `ml`, `viz`, `gui`. An area may use anything from earlier areas.
- `src/<area>/<area>.jl` includes the area's files and holds **all** of its `export`s.
- `src/core/`: `CausalEstimate` (every result type subtypes it and stores the full
  `vcov`), `DiagnosticTest`, `wald_test`, `critical_value`, `make_formula` (formulas are
  never built from strings), `require_columns`, `coef_index`, `tidy`/`glance`,
  `TreatmentPanel`.
- Regressions go through `FixedEffectModels.reg` (IV via `(d ~ z)` terms) or GLM.
  Units are matched by key, never by row position; results must not depend on row
  order.
- Package extensions in `ext/` (weak deps in `Project.toml`): `DrSnowMakieExt` (plot
  methods; stubs and plot-data extractors in `src/viz/`), `DrSnowRegressionTablesExt`,
  `DrSnowMLJExt` (`MLJLearner`), `DrSnowGUIExt` (HTTP + JSON3 + CSV; stubs in
  `src/gui/gui.jl`, static frontend in `ext/DrSnowGUIExt/static/`),
  `DrSnowTreatmentPanelsExt`. `using DrSnow` alone loads none of them.
- `app/Project.toml` is the GUI deployment environment used by `launch_gui.jl` and the
  `Dockerfile`.

### Tests, validation, docs

- `test/runtests.jl` runs `test/<area>/runtests.jl` inside a testset per area, plus
  `quality` (Aqua; bans `eval`, `Meta.parse` and runtime `include` in `src/` and `ext/`).
- `test/validation/<area>/` holds reference datasets, R (and some Julia) generating
  scripts and reference values; the tests read the committed CSVs, so R is not needed.
  `docs/src/validation.md` summarizes packages, versions and tolerances.
- Docs: `docs/src/<area>.md` guides each end with an `@docs` block for that area's
  exports; shared core names are on `docs/src/api.md`. `checkdocs = :exports`, no
  `warnonly`: every export must be documented exactly once. `docs/src/tutorial.md` is
  made of `@example` blocks that run on the validation datasets.
- `docs/task.md`, `docs/implementation_plan.md`, `docs/walkthrough.md` are historical
  planning notes, not Documenter pages.

## Development Conventions

Read `docs/CONVENTIONS.md` before changing `src/` or `test/`. It defines the module
layout (one aggregator with exports per area), the shared result interface
(`CausalEstimate`, `DiagnosticTest`), banned patterns (string-built formulas,
positional unit matching, silent NaN/0.0 results), keyword naming (`cluster`, `vcov`,
`level`, `weights`, `covariates`, `rng`), wording rules for diagnostics, and the testing
bar (reference comparisons plus Monte Carlo size/coverage checks via `mc_reps` with
`StableRNG` seeds).

## Code Style

- Julia style guide: `snake_case` functions/variables, `CamelCase` types, 4-space indent,
  lines under 92 characters.
- Exported functions need docstrings with `# Arguments`, `# Returns`, `# Examples` (and
  `# References` for methods from the literature); see `CONTRIBUTING.md`.
- Private helpers are prefixed with `_` and not exported.
- Printed output states estimates, intervals and test results without editorializing.
