# Compare two saved benchmark runs (from `run.jl --json`) and print a Markdown table.
#
#     julia --project=benchmark benchmark/compare.jl target.json baseline.json
#
# Ratios are median target time / median baseline time; BenchmarkTools.judge flags
# regressions and improvements beyond a 20% time tolerance (shared CI runners are
# noisy, so smaller changes are reported as invariant).

using BenchmarkTools

function main(target_path, baseline_path)
    target = only(BenchmarkTools.load(target_path))
    baseline = only(BenchmarkTools.load(baseline_path))
    println("| Area | Benchmark | Size | Baseline | Target | Ratio | Verdict |")
    println("|---|---|---|---:|---:|---:|---|")
    rows = []
    for (area, grp) in target, (k, t) in grp
        haskey(baseline, area) && haskey(baseline[area], k) || continue
        b = baseline[area][k]
        tm, bm = median(t), median(b)
        j = judge(tm, bm; time_tolerance=0.20)
        push!(rows, (area, k[1], k[2], bm, tm, j))
    end
    sort!(rows; by=r -> (r[1], r[2], r[3] == "small" ? 0 : 1))
    for (area, name, size, bm, tm, j) in rows
        println("| $area | `$name` | $size | $(BenchmarkTools.prettytime(time(bm))) | " *
                "$(BenchmarkTools.prettytime(time(tm))) | " *
                "$(round(time(tm) / time(bm); digits=2)) | $(time(j)) |")
    end
end

main(ARGS[1], ARGS[2])
