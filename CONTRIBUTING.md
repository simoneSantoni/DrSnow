# Contributing to DrSnow

Thank you for your interest in contributing. Please be respectful and constructive in
all interactions.

## Reporting bugs and suggesting features

Open an [issue](https://github.com/simoneSantoni/DrSnow_alpha/issues) with:

- a clear title;
- a minimal, self-contained example that reproduces the problem (simulated data with
  a fixed seed is ideal);
- what you expected and what happened, including the full error message;
- your Julia version, OS and DrSnow version or commit.

Incorrect statistical output (a wrong standard error, an interval that does not
match a reference implementation) is a bug. If possible, include the reference value
and how you obtained it (package, version and call).

For a new method, describe the use case and the proposed API, and cite the paper(s)
and, if one exists, a reference implementation that it can be validated against.

## Development workflow

```bash
git clone https://github.com/simoneSantoni/DrSnow_alpha.git
cd DrSnow_alpha
julia --project=. -e 'using Pkg; Pkg.instantiate()'
git checkout -b my-feature
```

Before changing `src/` or `test/`, read [docs/CONVENTIONS.md](docs/CONVENTIONS.md). It
is short and binding: the module layout, the shared result interface, banned patterns,
keyword names and the testing bar.

### Tests

```bash
julia --project=. -e 'using Pkg; Pkg.test()'                          # everything
DRSNOW_TEST_GROUP=did,iv julia --project=. -e 'using Pkg; Pkg.test()' # some areas
DRSNOW_SLOW_TESTS=true julia --project=. -e 'using Pkg; Pkg.test()'   # full Monte Carlo
```

Test groups are `quality` (Aqua, and a check that `src/` and `ext/` never use `eval`,
`Meta.parse` or runtime `include`), `core`, `ri`, `did`, `iv`, `rd`, `sutva`,
`synth`, `ml`, `viz` and `gui`. Continuous integration (`.github/workflows/CI.yml`)
runs the full suite with reduced Monte Carlo counts on Julia 1.10 and the current
release.

New estimators need, at a minimum:

- a check against known truth on a simulated design;
- a check against a reference (FixedEffectModels, a closed form, or an R/Stata
  package); store reference values and the script that produced them under
  `test/validation/<area>/`, recording package versions;
- a Monte Carlo size or coverage check for every interval or test, with
  `mc_reps(full, fast)` replication counts and `StableRNG` seeds;
- `@test_throws` checks for invalid input and a row-shuffling invariance check.

### Documentation

Each area has a guide, `docs/src/<area>.md`, that ends with an `@docs` block listing
every export of the area. Shared foundation types and helpers are documented in
`docs/src/api.md`. The build uses `checkdocs = :exports` without `warnonly`, so an
undocumented export, a docstring included twice, a broken `@ref` or a failing
`@example` block fails the build. The tutorial is executed on every build.

```bash
julia --project=docs -e 'using Pkg; Pkg.develop(PackageSpec(path=pwd())); Pkg.instantiate()'
julia --project=docs docs/make.jl       # output in docs/build/
```

Exported functions need docstrings with `# Arguments`, `# Returns`, `# Examples` and,
for methods from the literature, `# References`:

```julia
"""
    my_estimator(data, outcome, treatment; covariates=Symbol[], cluster=nothing,
                 level=0.95) -> MyEstimate

One-sentence summary, then the estimand and the identifying assumptions.

# Arguments
- `data`: a table with one row per ...
- `outcome::Symbol`, `treatment::Symbol`: ...
- `covariates::Vector{Symbol}`: ...

# Returns
- `MyEstimate` (a `CausalEstimate`).

# Examples
```julia
r = my_estimator(df, :y, :d; cluster=:state)
confint(r; level=0.9)
```

# References
- Author, A. (2024). Title. *Journal*, 1(1), 1–10.
"""
```

Update `CHANGELOG.md` under `[Unreleased]` for user-visible changes, and list every
breaking change.

### Pull requests

Describe what changed and why, reference related issues, and state how the change
was validated (tests added, references compared). Keep commits focused.

## Project structure

```
DrSnow/
├── src/
│   ├── DrSnow.jl          # loads packages and one aggregator per area, in order
│   ├── core/              # CausalEstimate, DiagnosticTest, inference helpers,
│   │                      # make_formula, TreatmentPanel
│   ├── ri/                # randomization inference, assignment mechanisms
│   ├── did/               # difference-in-differences
│   ├── iv/                # instrumental variables
│   ├── rd/                # regression discontinuity
│   ├── sutva/             # interference and spillovers
│   ├── synth/             # synthetic control, SDID, matrix completion
│   ├── ml/                # learners, DML, CATE, policy learning, PPI
│   ├── viz/               # plotting stubs and plot data (methods in ext/)
│   └── gui/               # launch_gui / stop_gui stubs (methods in ext/)
├── ext/                   # package extensions: Makie, RegressionTables, MLJ,
│                          # GUI (HTTP + JSON3 + CSV), TreatmentPanels
├── test/
│   ├── runtests.jl        # grouped runner (DRSNOW_TEST_GROUP, DRSNOW_SLOW_TESTS)
│   ├── <area>/runtests.jl # one directory per area, plus quality/ and viz/
│   └── validation/<area>/ # reference data, R scripts and reference values
├── docs/
│   ├── make.jl, Project.toml
│   ├── src/               # index, tutorial, one guide per area, results, gui,
│   │                      # validation, api
│   └── CONVENTIONS.md     # development conventions
├── app/                   # deployment environment for the GUI (launch_gui.jl, Docker)
├── examples/              # runnable scripts, one or more per area
└── literature/            # literature reviews (not used by the code)
```

Each `src/<area>/<area>.jl` aggregator includes the area's files and holds all of its
`export` statements. An area may use anything from areas included before it in
`src/DrSnow.jl`.

### Adding a method

1. Implement it in `src/<area>/`, returning a `CausalEstimate` subtype (or a
   `DiagnosticTest` for a test), and include the file from the aggregator.
2. Export it from `src/<area>/<area>.jl`.
3. Add tests (see above) under `test/<area>/` and include them from
   `test/<area>/runtests.jl`.
4. Describe it in `docs/src/<area>.md` and add it to that page's `@docs` block.
5. Optionally, add or extend a script in `examples/`.

Optional heavy dependencies go in `[weakdeps]` and `[extensions]` with code under
`ext/`; the generic function with a helpful error when the extension is not loaded
lives in the owning area.

## Release process (maintainers)

1. Update the version in `Project.toml` and move the `[Unreleased]` changelog entry
   under the new version.
2. Tag the release (`git tag v0.x.y`), push the tag and create a GitHub release.

## License

By contributing, you agree that your contributions will be licensed under the MIT
License.
