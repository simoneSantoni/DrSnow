# Run the DrSnow benchmark suite and print a Markdown table of the results.
#
#     julia --project=benchmark benchmark/run.jl [filter] [--json results.json]
#
# `filter` (optional) keeps benchmarks whose area or name contains it, e.g. `did`.
# With `--json path` the raw BenchmarkTools results are also saved (compare two
# saved runs with `BenchmarkTools.judge`).

using BenchmarkTools
using Printf

include(joinpath(@__DIR__, "benchmarks.jl"))

function main(args)
    json = nothing
    filt = nothing
    i = 1
    while i <= length(args)
        if args[i] == "--json"
            json = args[i + 1]
            i += 2
        else
            filt = args[i]
            i += 1
        end
    end
    suite = SUITE
    if filt !== nothing
        suite = BenchmarkGroup()
        for (area, grp) in SUITE, (k, b) in grp
            if occursin(filt, area) || occursin(filt, string(k[1]))
                haskey(suite, area) || (suite[area] = BenchmarkGroup([area]))
                suite[area][k] = b
            end
        end
    end
    println("Tuning and running $(length(BenchmarkTools.leaves(suite))) benchmarks...")
    tune!(suite)
    res = run(suite; verbose=false)
    json === nothing || BenchmarkTools.save(json, res)
    println()
    println("| Area | Benchmark | Size | Median time | Memory | Allocs |")
    println("|---|---|---|---:|---:|---:|")
    rows = sort!([(a, k[1], k[2], median(t)) for (a, grp) in res for (k, t) in grp];
                 by=r -> (r[1], r[2], r[3] == "small" ? 0 : 1))
    for (a, name, size, t) in rows
        println("| $a | `$name` | $size | $(BenchmarkTools.prettytime(time(t))) | " *
                "$(BenchmarkTools.prettymemory(memory(t))) | $(allocs(t)) |")
    end
    return res
end

main(ARGS)
