# Repository Guidelines

## Project Structure & Module Organization

DrSnow is a Julia 1.10+ causal-inference package. Read `CONTRIBUTING.md` and the
binding `docs/CONVENTIONS.md` before changing source or tests.

- `src/DrSnow.jl` loads area aggregators; `src/<area>/` contains estimators,
  inference, experimental design, and shared `core/` utilities. Each
  `<area>.jl` owns its area's includes and exports; respect include order.
- `ext/` houses optional integrations; GUI assets live in
  `ext/DrSnowGUIExt/static/`.
- `test/<area>/` mirrors source areas; `test/validation/<area>/` stores reference
  fixtures and generation scripts.
- `docs/src/` contains guides and API references; `examples/` contains runnable
  demos, `benchmark/` performance tooling, and `literature/` research notes.

## Build, Test, and Development Commands

Run from the repository root:

```bash
# Install package dependencies
julia --project=. -e 'using Pkg; Pkg.instantiate()'
# Run all tests, including Aqua quality checks
julia --project=. -e 'using Pkg; Pkg.test()'
# Run selected test areas
DRSNOW_TEST_GROUP=did,iv julia --project=. -e 'using Pkg; Pkg.test()'
# Run full Monte Carlo replication counts
DRSNOW_SLOW_TESTS=true julia --project=. -e 'using Pkg; Pkg.test()'
# Run a local example
julia --project=. examples/basic_did_demo.jl
# Prepare and build documentation into docs/build/
julia --project=docs -e 'using Pkg; Pkg.develop(PackageSpec(path=pwd())); Pkg.instantiate()'
julia --project=docs docs/make.jl
```

## Coding Style & Naming Conventions

Use four-space indentation, lines under 92 characters, `snake_case` functions,
and `CamelCase` types. Follow existing formatting; quality checks use Aqua.
Build formulas with `make_formula` or term APIs; never use `eval`, `Meta.parse`,
`@eval`, or runtime includes in source or extensions. Validate inputs explicitly
and match observations by key. Estimates subtype `CausalEstimate` and retain full
covariance matrices. Document exports using the structure in
`docs/CONVENTIONS.md`; Documenter checks export coverage.

## Testing Guidelines

Use Julia `Test`, naming files `test_<feature>.jl` and including them from the
area's `runtests.jl`. Check simulated truth, reference estimates and standard
errors, invalid inputs, and row-shuffling invariance. Inferential procedures need
Monte Carlo size or coverage checks using `mc_reps(full, fast)` and `StableRNG`
seeds. Store reference scripts and package versions alongside fixtures.

## Commit & Pull Request Guidelines

Git history contains only initial commits, so no message convention is established.
Keep commits focused with descriptive subjects. PRs should explain what changed
and why, link related issues, and report validation. Update relevant guides and API
references, and record user-visible and breaking changes under `[Unreleased]` in
`CHANGELOG.md`.
