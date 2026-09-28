# DrSnow Documentation Assessment (2026-09-28)

Six reviewers examined the documentation of branch `overhaul/panel-roadmap` at commit
`cbf84f4`, read-only:

| Reviewer | Lens |
|---|---|
| API accuracy | Docstrings vs method signatures; docstring, guide and example code executed |
| Econometrics | DiD, IV, RD, synthetic control guides |
| Statistics and causal ML | RI, interference, ML, sequential, adaptive, design guides |
| Applied user | Onboarding, navigation, pedagogy, publication workflow |
| Docs engineering | Build, CI and deployment, links, citations, repository docs |
| Claims audit | Every quantitative claim vs enforcing tests; provenance and licensing |

**[V]** marks findings verified by running code; the rest were checked by reading code
or literature. Items marked *please verify* depend on reviewer recall.

---

## 1. Verdict

The documentation is accurate at the API level and unusually honest about limitations.
Every exported name is documented exactly once, no docstring names a keyword the code
lacks, the strict Documenter build passes with zero warnings on Julia 1.10 and 1.13,
all 29 executed tutorial blocks and all 19 example scripts run, and 25 re-run
validation test files (~8,800 tests) pass.

The problems fall into four groups:

1. **Decisions only the maintainer can make**: licensing of ported GPL code, the
   repository name and public site, and PDFs in git history.
2. **A handful of statements that are wrong** and would mislead users, two of them
   caused by code behaviour.
3. **Validation claims stated more tightly than the tests enforce.**
4. **Usability**: the guides are reference-style, show no output or figures, and lack
   a decision guide and an R/Stata crosswalk.

---

## 2. Decisions for the maintainer

| # | Issue | Evidence | Options |
|---|---|---|---|
| D1 | **GPL code ported into an MIT package.** grf's C++ core is ported line by line (`src/ml/grf_core.jl`); DRDID, rdrobust, RDHonest, HonestDiD, DIDmultiplegtDYN, did and dsl are followed step by step; R's `Brent_fmin` and SciPy's `brentq` are transcribed. No NOTICE file exists. | file headers cited in the audit | License review; relicense (e.g. GPL-3), obtain permission, or clean-room rewrite; add `THIRD_PARTY_NOTICES`. Origin licenses must be checked against each package's DESCRIPTION/LICENSE. |
| D2 | **Every package URL points to a repository and site that do not exist.** The remote is the private `simoneSantoni/DrSnow_alpha`; docs, README, badges, CONTRIBUTING and CHANGELOG use `DrSnow` (404). [V] | `gh repo view`; `docs/make.jl:9,14` | Rename the repository, or change every URL. |
| D3 | **The public docs site serves the v0.1 manual**, whose inference the CHANGELOG calls incorrect. `origin/master` is 154 commits behind. [V] | live Pages site | Merge and redeploy, or take the site down until release. |
| D4 | **~58 third-party and unpublished PDFs remain in git history**; the CHANGELOG implies they were removed. | `97a70c7` | `git filter-repo` before publishing, or reword the CHANGELOG. |
| D5 | Copyright holder is "Simon Santoni" in `LICENSE` but "Simone Santoni" elsewhere. | `LICENSE:3` | Pick one. |

---

## 3. Statements that are wrong or misleading

| Sev | Where | Problem | Fix |
|---|---|---|---|
| High | `src/iv/estimator.jl:283–309`; `tutorial.md:158–163`; `iv.md:58–65` | IV estimand printout for binary Z, multi-valued D and covariates says "multi-valued instrument … adjacent instrument values". The tutorial (Card) shows it. [V] | Add the case: ACR (Angrist & Imbens 1995) with the covariate caveat (Blandhol et al. 2022; Słoczyński 2022); add a table row; regenerate tutorial. |
| High | `rd.md:62–67`; `tutorial.md:230–233`; `src/rd/estimate.jl:79` | Docs say report the conventional estimate with the robust CI, but `coef`, `tidy` and `regtable` return the bias-corrected estimate (Senate: 7.507 vs 7.414). [V] | Follow rdrobust (conventional headline, robust SE/CI), or document and provide an accessor. |
| High | `adaptive.md:180–183, 275–290` | Recommends confidence sequences on AIPW scores to monitor adaptive experiments and claims finite-sample validity for bounded outcomes. Scores are unbounded with decaying floors; Monte Carlo coverage 0.81 / 0.88 / 0.97 in a high-signal design. The "practical split" is invalid when stopping depends on the sequence. [V] | Drop the claim; state conditions; say the combination is not implemented or validated; or add a helper with a Monte Carlo check. |
| Med | `did.md:300–301` | ACRT defined as ∂ATT(d\|d)/∂d; in Callaway, Goodman-Bacon & Sant'Anna it is ∂ATT(l\|d)/∂l at l=d. | Call the curve the dose-response slope; relate it to ACRT/ACR. |
| Med | `did.md:228–229` | ETWFE (not-yet-treated) requires parallel trends in all pre-periods; not stated. | State it; add a paragraph contrasting parallel-trends variants across CS/BJS/ETWFE/dCDH. |
| Med | `tutorial.md:123–136` | Says Δ^RM bounds deviation levels; it bounds consecutive changes. | Reword (did.md is correct). |
| Med | `synth.md:129–130` | Permutation test is exact only under random assignment across units. | State the assumption; cite Abadie (2021), Firpo & Possebom (2018). |
| Med | `sequential.md:139–142` | Says `gs_analysis` recomputes spending at observed information; it uses planned timing unless `information_fraction` is passed. | Document `information_fraction`, I_max and over/under-running. |
| Med | `sequential.md:64–65` | Batch functions tune `t_opt`/`n_opt` to the realized n by default, which is data-dependent when n came from stopping. | Document; recommend pre-registered values. |
| Med | `ml.md:84–88` | Attributes DrSnow's repeated-split variance rule to Chernozhukov et al. §3.4; DrSnow's is more conservative (*please verify*). `dml_did_multi` uses the mean rule. | Reword. |
| Med | `adaptive.md:19–21`; `src/adaptive/policies.jl:28` | Policies default to no floor; inference then runs silently without positivity. | Document; warn in code. Check Zhan et al. decay condition (*please verify* α < 1/2). |
| Med | `design.md:138–139` | Matched-pair variance citation should be Bai, Romano & Shaikh (2022, JASA). | Fix; mention pairs-of-pairs. |
| Med | `sutva.md:31–34, 106–111` | Misspecified exposure mappings are described loosely; Monte Carlo exposure probabilities are raw frequencies. | Cite Sävje (2024); give draw-count guidance. |
| Med | several | Rosier-than-data Monte Carlo wording: `rd_flex` forest 0.935 with SE/SD 0.91; honest RD 0.92 at the least-favourable function; SC placebo-variance coverage 0.925 undisclosed; AR-set figure from 100 reps. [V] | Report realized numbers and replication counts. |

Lower-severity accuracy items (≈30) are listed in the reviewers' notes: BJS spherical
errors, binning constant-effect assumption, Sun–Abraham share variance, RD kink default
`p`, CER wording, rule-of-thumb M and honesty, UJIVE projection, `se` defaults, MTE
parametric-only parameters, conformal rejection convention, local-randomization
window selection by non-rejection, RI equal-tailed vs two-sided duality, conditioning on
the observed number treated, `variance_reduction` approximation, ML measurement caveats.

---

## 4. API and example defects [V]

| Sev | Where | Problem |
|---|---|---|
| High | `src/design/analysis.jl:157` | `experiment_estimate` docstring example throws (default `method=:difference` rejects covariates). |
| High | `src/did/honest_did.jl:801–804` | `honest_did`/`honest_breakdown` example fails with default `endpoints=:bin`; needs `:trim`, and the requirement is undocumented there. |
| Med | `src/ml/grf.jl:31,67` | `num_trees` listed as a field of `CausalForest`/`RegressionForest`; it lives in `params`. |
| Med | adaptive/design docs and docstrings | Variables named `log`/`exp` fail on Julia 1.10 after Base's use and shadow Base. |
| Med | several | Undocumented keywords (`honest_breakdown`'s 10, `predict_interval(parallel)`, `rd_plot_data(covs_drop)`, `instrumental_forest(weights, cluster)`); private `_prelim`/`_ipw` exposed in `dml_lqte`. |
| Med | `ml.md:54` | Custom learners also need `fitpredict_proba` for propensity roles. |
| Med | README, tutorial, 46 docstrings | `StableRNGs`, `CSV`, `DataFrames` are used but not dependencies; first run fails. |
| Low | 49 / 45 / 39 exports | Missing `# Arguments` / `# Returns` / `# Examples` (31 are `plot_*!` stubs); 60 structs lack `# Fields`; 6 lack `# References`; six docstrings contain math as code blocks. |
| Low | `src/rd` | `rd_donut` raises a raw `BoundsError` when a radius empties the sample. |

Statistics: 448 exports; 0 documented keywords missing from code; 96 inferable return
types all match; ~55% of 306 docstring examples verified runnable (80% need
user data); 0 static errors in 342 code blocks.

---

## 5. Validation and reproducibility claims

- **Tolerances are mislabelled (High).** `validation.md:12` says the numbers are the
  largest differences the tests accept; for ~30 of 46 rows they are observed differences,
  while tests enforce tolerances 10²–10⁷ times looser. rdrobust agreement observed at
  1.5e-8 exceeds the documented 1e-9. Guide pages and validation.md disagree on the same
  comparisons (RPIV, LQTE, ManyIV, confseq, rd_honest bandwidth). **Fix:** two columns
  ("observed at commit X", "test tolerance"), or tighten tests; make pages agree.
- **Rendering bugs**: `validation.md:51` (unescaped `|z|` splits the row) and `:102`
  (cell text outside the table). [V]
- **Reference versions** not recorded in committed files for ~15 packages (gsDesign,
  rpact, grf, DoubleML, ManyIV, clusterIV, ivmte, RPIV, AER, sandwich, ivmodel, fixest,
  ShiftShareSE, interference); GitHub sources unpinned (MCPanel, RDHonest, bartik.weight,
  Hadad code, dsl).
- **Regeneration table** misses six scripts.
- **Test hygiene**: the `validation` test group has no `runtests.jl` and runs nothing;
  some test files depend on globals from other files, so single-file runs fail.
- **Data provenance/licensing** of committed datasets and verbatim-copied reference
  code is stated only in script comments.
- **Monte Carlo claims.** Re-runs reproduce `did_multi_montecarlo.csv` byte for byte and
  the Hadad et al. study at T=1000 to 5e-17; the `rd_flex` table matches its CSV [V].
  But:
  - **(High)** "every inferential procedure has a Monte Carlo size or coverage check"
    (validation.md, index.md, README, iv.md, sutva.md, ml.md) is false. Examples without
    one: `honest_did`, `honest_breakdown`, `dsl_proportions`, `ppi_regression`,
    `policy_value`, `mte_bounds`, `rd_honest_bme`, `rd_smoothness_bound`, `rd_donut`,
    `rd_placebo_cutoffs`, `synth_in_time_placebo`, `wald_test`.
  - **(High)** The ml_measurement.md Monte Carlo table has no committed output; its
    "800 units" row has no code, and the stated replication counts need an unstated
    `MC_REPS=2000`.
  - **(Medium)** About ten figures quoted in prose appear in no script, CSV or log
    (robust CLR/K at 200/500 clusters, RPIV/DML-AR/LQTE rates, FLL rates, forest coverage
    incl. the bracketed grf figures, honest RD and AR coverage, sequential coverage,
    ASCM jackknife "two thirds", WCLS/EMEE coverage); the rdrobust 0.902 target
    hard-coded in a test has no generating R script.
  - **(Low)** No test reads the committed Monte Carlo CSVs; they carry no SHA, Julia
    version or replication count; two scripts seed with `hash(...)`, which is not stable
    across Julia versions; some ranges are rounded outward.
- **Benchmarks** are honestly framed but from an uncommitted tree 45 commits ago;
  allocations reproduce, times differ 2–4×.
- **CHANGELOG** overclaims completeness ("complete for exported names"), contradicts its
  own deprecation policy, and is not Keep-a-Changelog compliant.

---

## 6. Usability

Works well: the executed tutorial on real data with interpretation; results that print
their estimand and honest non-rejection language; `regtable` and `plot_event_study`
produce publication output [V]; rd.md's R/Stata analogue table and falsification
workflow; per-area "which estimator" tables.

Gaps, in priority order:

1. **No executed examples or shown output in any method guide** (0 `@example` blocks in
   16 guides; three guides have no code).
2. **No figures anywhere** (CairoMakie not in the docs environment).
3. **Onboarding fails as written**: install instructions omit DataFrames/CSV/StableRNGs
   and environment basics. [V]
4. **Example data hidden** under `test/validation`; no datasets page or "bring your own
   data" guidance outside DiD.
5. **Decision support scattered**: the index "Which design" table omits sequential,
   adaptive, design, PPI/DSL, kink, continuous DiD; five different headings for choice
   tables.
6. **No R/Stata crosswalk** for an audience coming from fixest, did, rdrobust, grf.
7. **Publication tables half documented** (row alignment, dropping controls, label
   length).
8. Guides read as implementation notes; assumptions are not checklists; results.md mixes
   developer material; heading/spelling inconsistency; no FAQ, citation file, glossary or
   performance page.

---

## 7. Documentation engineering

- **Deployment**: Actions-Pages without `deploydocs` means no versioned docs, no
  `stable`, no PR previews; version selector 404s. [V]
- **Doctests effectively off**: zero `jldoctest` blocks.
- **`@meta CurrentModule` missing** on 11 of 20 pages.
- **Citations** hand-written in two styles across 10 pages (215 entries, duplicates,
  missing pages/titles, no DOIs); sub-pages cite works listed only on parent pages.
  Canonical references missing (e.g. Roth et al. 2023; Hahn, Todd & van der Klaauw 2001;
  Lee & Lemieux 2010; Firpo & Possebom 2018; Crump et al. 2009; Chung & Romano 2013).
- **Assets**: 4.85 MB unused `icon.png`; oversized logo; no favicon or dark logo.
- **Search index** at 97.8% of its warning threshold.
- **Contributor docs** (CLAUDE.md, CONTRIBUTING, CONVENTIONS) omit the three newest
  areas and still mention Genie; `master`'s CLAUDE.md is the v0.1 text.
- **Housekeeping**: examples not indexed or linked; `docs/walkthrough.md` lacks a
  historical banner; Python `.gitignore`; 22 leftover agent worktrees (367 MB).

---

## 8. Recommended plan

**Phase A: blockers before any public release** (maintainer decisions D1–D5).

**Phase B: correctness (S–M).**
1. Fix the IV estimand printout and the RD headline/reporting mismatch in code; update
   guides and tutorial.
2. Correct or withdraw the adaptive-plus-sequential recommendation.
3. Fix the medium accuracy items in section 3 and the API defects in section 4.
4. Relabel validation tolerances, fix table rendering, record reference versions, add
   `test/validation/runtests.jl`, make test files self-contained.
5. Reword the "every procedure has a Monte Carlo check" claim or add the missing checks;
   commit outputs (with SHA and replication count) for every Monte Carlo figure quoted
   in the docs, or remove the figure.

**Phase C: infrastructure (M).**
6. `deploydocs` with versions and previews; `CurrentModule` everywhere; quality checks
   for both; doctests for ~20 deterministic examples; CI job running `examples/`.
7. DocumenterCitations with a package `.bib` (seed from `~/slimer/bibliography.bib`
   where entries match).
8. Update contributor docs; CHANGELOG to Keep a Changelog; assets; housekeeping.

**Phase D: usability (M–L).**
9. Page template per guide: problem, executed worked example with output and
   interpretation, assumptions checklist, options, theory, validation, references, API.
10. Figures via CairoMakie in the docs build.
11. New pages: Choosing a method; Coming from R or Stata; Datasets and your own data;
    FAQ, performance, reproducibility and citing; Examples rendered with Literate.
12. Onboarding: dependencies, environments, expected output.
