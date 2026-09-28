# DrSnow development conventions

These rules apply to every area of the package. They exist because the v0.1 review
(`docs/panel_review_2026-09-27.md`) found that most defects came from violating them.

## Layout and ownership

- `src/DrSnow.jl` only loads packages and includes one aggregator per area:
  `src/<area>/<area>.jl`. The aggregator includes the area's files and holds the
  area's `export` statements. Do not add exports anywhere else.
- Areas: `core`, `ri` (randomization inference), `did`, `iv`, `rd`, `sutva`
  (interference), `synth`, `ml`, `viz` (plot stubs). Include order is fixed in
  `src/DrSnow.jl`; an area may use anything from areas included before it.
- Tests: `test/<area>/runtests.jl` is included by `test/runtests.jl` inside a
  testset named after the area. Run one area with
  `DRSNOW_TEST_GROUP=did julia --project=. -e 'using Pkg; Pkg.test()'`.
- Docs: each area has `docs/src/<area>.md` (methods, assumptions, references,
  and an `@docs` block listing every export of the area). `docs/make.jl` uses
  `checkdocs = :exports`, so an undocumented export breaks the docs build.
- Optional heavy dependencies (Makie, MLJ, JuMP/HiGHS, Genie, …) go in
  `[weakdeps]` + `[extensions]` with code under `ext/`. The core stub (a generic
  function with a helpful error when the extension is not loaded) lives in the
  owning area.

## Estimation code

- Never build formulas from strings. Use `make_formula` (`src/core/formulas.jl`)
  or `StatsModels.term` / `FixedEffectModels.fe`. `eval`, `Meta.parse`, `@eval`
  and runtime `include` are banned in `src/` (enforced by `test/quality`).
- Validate inputs up front with `require_columns` and clear `ArgumentError`s.
  Never assume row order: match units by key (`Dict(id => index)`), never by
  position. Results must be invariant to shuffling data rows.
- Read coefficients from fitted models by name (`coef_index`), which also errors
  when the coefficient was dropped as collinear. Never return `0.0` / `NaN` for a
  non-identified quantity: throw an informative error instead.
- Delegate regressions to `FixedEffectModels.reg` (IV via `(d ~ z)` terms) or GLM
  whenever possible instead of hand-rolled linear algebra.

## Results and inference

- Every estimate type subtypes `CausalEstimate` and implements `coef`, `vcov`,
  `coefnames`, `nobs`; set `dof_residual` when a t reference is appropriate
  (e.g. `G - 1` clusters), and `estimand` / `method_name`. Store the full `vcov`,
  never only standard errors. Do not hard-code 1.96: use `confint(r; level)` /
  `critical_value(level, dof)`.
- Joint tests use `wald_test` with the full covariance matrix.
- Diagnostic and falsification tests return `DiagnosticTest`. Wording rules: a
  non-rejection is never reported as evidence that an assumption holds; untestable
  assumptions are never described as "tested"; placeholders are never exported.
- Keyword names are uniform across the package:
  - `cluster::Union{Nothing,Symbol,Vector{Symbol}}` and/or
    `vcov::FixedEffectModels.CovarianceEstimator` (`vcov` wins if both given);
  - `level::Real = 0.95`;
  - `weights::Union{Nothing,Symbol}`;
  - `covariates::Vector{Symbol} = Symbol[]` (always a keyword);
  - `rng::AbstractRNG = Random.default_rng()` for anything stochastic; draw
    per-task seeds up front with `task_seeds` so results do not depend on
    threading.
  - Positional argument order: `(data, outcome, treatment, [instrument|running
    variable], ...)`.

## Tests

- Test estimators against known truth from simulated DGPs *and* against a
  reference (FixedEffectModels, closed-form formulas, or published / R reference
  values stored in `test/validation/`). Point-estimate `atol=1.5` checks are not
  acceptable; compare SEs too.
- Every inferential procedure gets a Monte Carlo size or coverage check using
  `mc_reps(full, fast)` so CI runs a reduced count. Use `StableRNG` seeds, never
  `Random.seed!` on the global RNG.
- Test error paths (`@test_throws`) and row-shuffling invariance.

## Style

- Julia style: `snake_case` functions, `CamelCase` types, 4-space indent, lines
  under 92 characters. Exported functions have docstrings with `# Arguments`,
  `# Returns`, `# Examples`, and `# References` for methods from the literature.
- Printed output states estimates, intervals and test results; it does not
  editorialize ("✓ robust", "appears random", "SUTVA holds").

## API docstrings

Docstrings are the package's scientific reference: they are rendered on the API
reference pages (`docs/src/reference/<area>.md`) and at the REPL. They are written in
full, precise prose for a reader trained in quantitative social science, and they
anchor every method in the literature.

**Structure of an estimator or test docstring** (in this order):

1. Signature line(s) indented four spaces, with keywords and defaults, then
   `-> ReturnType`.
2. A one-sentence summary.
3. A scientific description, two to four paragraphs of prose (not bullet lists):
   - the research question and the **estimand**, defined formally (potential
     outcomes; use ``` ``…`` ``` for inline math and a ```` ```math ```` block for
     displayed equations);
   - the **identifying assumptions**, stated precisely, and which of them the data can
     and cannot speak to;
   - the **estimator** and its key properties (e.g. double robustness, Neyman
     orthogonality, efficiency, rates), with the construction in words or as a short
     numbered algorithm when that is clearer;
   - **inference**: the variance estimator or reference distribution, degrees of
     freedom, finite-sample caveats, and known Monte Carlo behaviour where the package
     documents it;
   - **practical guidance**: when to prefer this method over its alternatives (link
     them with ``[`name`](@ref)``), common pitfalls, and what to report.
   Cite in prose as "Callaway and Sant'Anna (2021)".
4. `# Arguments`: one or two full sentences per positional argument (coding,
   units, consequences).
5. `# Keywords`: one or two full sentences per keyword, with its default and what
   changing it does.
6. `# Returns`: the type, and the fields and accessors a user needs.
7. `# Examples`: runnable, self-contained code where feasible (simulate data with
   `StableRNG`), never using `log` or `exp` as variable names.
8. `# References`: usually three to eight entries: the originating paper(s), the
   paper that defines the implemented variant, important extensions or critiques, a
   review or textbook treatment, and the reference software used for validation.
   Format: `- Author, A., Author, B., & Author, C. (Year). Title. *Journal*,
   Volume(Issue), pages.` Working papers give the series and number; books give the
   publisher. Add a DOI only when verified.

Types get a summary, a paragraph on what the object represents, `# Fields`
(every public field) and, where relevant, the accessors that work on it. Short helper
functions and plotting `!` variants may be brief but still have `# Arguments`,
`# Returns` and a sentence pointing to the non-mutating variant.

**Citation integrity.** Cite only works whose existence and bibliographic details are
certain. Verify every new reference (web search, the publisher's page, or an entry
already checked in the method guides); never invent volume, page or DOI details, and
omit a reference rather than guess. Prefer the published version over working papers.

**Tone.** Neutral and precise. State what a test's non-rejection does not show, what
an estimator does not identify, and approximations as approximations.
