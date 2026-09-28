# Generates the inputs of the Aronow–Samii reference comparison.
#
#   julia --project=<env with DrSnow + StableRNGs> \
#       test/validation/sutva/generate_aronow_samii.jl
#   Rscript test/validation/sutva/aronow_samii_reference.R <path-to-R-library>
#
# Outputs (plain text, parsed by test/sutva/test_validation.jl):
#   as_edges.csv        undirected edge list (source,target) on units 1..N
#   as_units.csv        unit,z,y   (observed assignment and outcome)
#   as_draws.txt        one line per assignment draw, N characters of 0/1
# The R script then writes as_reference.csv with the `interference` package results.
# The draws are stored (not regenerated) so the comparison never depends on RNG streams.

using DrSnow, StableRNGs, Random

const DIR = @__DIR__
rng = StableRNG(20260927)
N = 40
edges = Tuple{Int,Int}[]
for i in 1:N, j in (i + 1):N
    rand(rng) < 0.08 && push!(edges, (i, j))
end
# every unit gets at least one neighbour (isolates are handled elsewhere)
deg = zeros(Int, N)
for (a, b) in edges
    deg[a] += 1
    deg[b] += 1
end
for i in 1:N
    if deg[i] == 0
        j = i == N ? 1 : i + 1
        push!(edges, (min(i, j), max(i, j)))
        deg[i] += 1
        deg[j] += 1
    end
end
A = zeros(N, N)
for (a, b) in edges
    A[a, b] = A[b, a] = 1
end
design = CompleteRandomization(N, 12)
z = draw_assignment(rng, design)
expo = A * z
y = [3.0 + 0.3 * deg[i] + 2.0 * z[i] + 1.0 * (expo[i] > 0) + randn(rng) for i in 1:N]
R = 600
draws = reduce(hcat, [draw_assignment(rng, design) for _ in 1:R])

open(joinpath(DIR, "as_edges.csv"), "w") do io
    println(io, "source,target")
    for (a, b) in edges
        println(io, a, ",", b)
    end
end
open(joinpath(DIR, "as_units.csv"), "w") do io
    println(io, "unit,z,y")
    for i in 1:N
        println(io, i, ",", Int(z[i]), ",", repr(y[i]))
    end
end
open(joinpath(DIR, "as_draws.txt"), "w") do io
    for r in 1:R
        println(io, join(Int.(draws[:, r])))
    end
end
