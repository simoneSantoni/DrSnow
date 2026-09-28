# DrSnow benchmarks

`benchmarks.jl` defines a [BenchmarkTools](https://github.com/JuliaCI/BenchmarkTools.jl)
`SUITE` (the layout [PkgBenchmark](https://github.com/JuliaCI/PkgBenchmark.jl)
expects) timing representative estimators of every area on simulated data, each at a
**small** and a **medium** size:

| Area | Benchmarks | Small | Medium |
|---|---|---|---|
| `did` | `did_twfe`, `did_callaway_santanna` (with the default 999-draw multiplier bootstrap), `event_study` (TWFE) | 200 units × 10 periods | 2,000 × 12 |
| `iv` | `iv_regression`, `weak_iv_confidence_set` (AR, CLR) | n = 1,000 | 20,000 |
| `rd` | `rd_estimate`, `rd_bandwidth` | n = 1,000 | 20,000 |
| `synth` | `synthetic_did` (point estimate, `se_method=:none`) | 30 units × 20 periods | 100 × 40 |
| `ri` | `randomization_test` (1,000 permutations, single-threaded) | n = 200 | 2,000 |
| `ml` | `dml_irm` (OLS / logistic learners, 5 folds) | n = 1,000 | 10,000 |
| `sutva` | `exposure_probabilities` (2,000 draws), `exposure_effects` (Hájek) | 200-node network | 1,000 |

The data are drawn once with fixed `StableRNG` seeds, and stochastic estimators get a
fresh `StableRNG` in every sample, so each run times exactly the same computation.
The suite is tuned for short runs (at most about 2 s per benchmark); it is meant to
catch regressions of the order of 2×, not a few percent.

## Running

The benchmark environment uses DrSnow from the parent directory (`[sources]`,
Julia ≥ 1.11; on Julia 1.10 run
`julia --project=benchmark -e 'using Pkg; Pkg.develop(path=".")'` once).

```bash
julia --project=benchmark -e 'using Pkg; Pkg.instantiate()'

# Whole suite, Markdown table on stdout (about 3–4 minutes)
julia --project=benchmark benchmark/run.jl

# One area or estimator, saving raw results
julia --project=benchmark benchmark/run.jl rd --json rd.json

# Compare two saved runs (ratio of medians; ±20% counts as invariant)
julia --project=benchmark benchmark/compare.jl new.json old.json

# With PkgBenchmark (e.g. comparing git revisions)
julia --project=benchmark -e 'using PkgBenchmark, DrSnow;
    export_markdown(stdout, judge(DrSnow, "HEAD", "master"))'
```

Continuous integration: the optional `Benchmarks` workflow
(`.github/workflows/benchmark.yml`) runs on demand (`workflow_dispatch`) and on pull
requests labelled `run benchmarks`, where it also runs the same suite on the base
commit and writes the comparison to the job summary. It never fails on a slowdown;
shared runners are too noisy for hard thresholds.

## Reference results

One run of `benchmark/run.jl` on 2026-09-27 at commit `479b9dd`+ (this suite's first
version): Julia 1.13.0, Linux (Fedora 44), Intel Core i9-14900HX (32 threads, only
one used by the benchmarks: `JULIA_NUM_THREADS` unset), 32 GB RAM, on a laptop with
other work running, so treat differences below ~20% as noise. Medians over the
samples collected in ~2 s per benchmark.

| Area | Benchmark | Size | Median time | Memory | Allocs |
|---|---|---|---:|---:|---:|
| did | `did_callaway_santanna` | small | 6.983 ms | 4.85 MiB | 35703 |
| did | `did_callaway_santanna` | medium | 73.653 ms | 43.55 MiB | 515268 |
| did | `did_twfe` | small | 8.449 ms | 2.19 MiB | 41891 |
| did | `did_twfe` | medium | 74.444 ms | 28.19 MiB | 704734 |
| did | `event_study` | small | 5.528 ms | 1.47 MiB | 26379 |
| did | `event_study` | medium | 51.372 ms | 18.83 MiB | 395755 |
| iv | `iv_regression` | small | 4.493 ms | 829.33 KiB | 1453 |
| iv | `iv_regression` | medium | 9.043 ms | 12.56 MiB | 1458 |
| iv | `weak_iv_confidence_set_ar` | small | 49.236 μs | 109.76 KiB | 122 |
| iv | `weak_iv_confidence_set_ar` | medium | 889.208 μs | 1.99 MiB | 122 |
| iv | `weak_iv_confidence_set_clr` | small | 4.620 ms | 2.59 MiB | 48635 |
| iv | `weak_iv_confidence_set_clr` | medium | 5.856 ms | 4.04 MiB | 48657 |
| ml | `dml_irm` | small | 2.277 ms | 1.73 MiB | 1544 |
| ml | `dml_irm` | medium | 25.408 ms | 13.79 MiB | 1624 |
| rd | `rd_bandwidth` | small | 1.201 ms | 1.02 MiB | 9188 |
| rd | `rd_bandwidth` | medium | 25.302 ms | 19.49 MiB | 199304 |
| rd | `rd_estimate` | small | 1.513 ms | 1.26 MiB | 9509 |
| rd | `rd_estimate` | medium | 28.323 ms | 25.00 MiB | 199727 |
| ri | `randomization_test` | small | 6.399 ms | 2.77 MiB | 10098 |
| ri | `randomization_test` | medium | 49.441 ms | 24.86 MiB | 44422 |
| sutva | `exposure_effects` | small | 961.164 μs | 1.35 MiB | 2669 |
| sutva | `exposure_effects` | medium | 40.246 ms | 30.07 MiB | 16039 |
| sutva | `exposure_probabilities` | small | 118.573 ms | 79.78 MiB | 1533188 |
| sutva | `exposure_probabilities` | medium | 586.922 ms | 389.37 MiB | 7416605 |
| synth | `synthetic_did` | small | 4.177 ms | 250.77 KiB | 3754 |
| synth | `synthetic_did` | medium | 76.283 ms | 1.87 MiB | 34378 |

`exposure_probabilities` (Monte Carlo over 2,000 simulated assignments)
dominates the design-based interference workflow; the
other estimators run in well under a second at these sizes.

