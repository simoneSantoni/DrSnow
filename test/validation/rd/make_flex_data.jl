# Generates the simulated RD datasets and fold assignments used to validate `rd_flex`
# against DoubleML's RDFlex (Python; see doubleml_rdflex_reference.py).
#
#   julia test/validation/rd/make_flex_data.jl
#
# The CSV files are committed; this script documents how they were produced.

using Random

const DIR = @__DIR__

function write_csv(path, header, cols)
    open(path, "w") do io
        println(io, join(header, ","))
        for i in eachindex(cols[1])
            println(io, join((c[i] isa Integer ? string(c[i]) : repr(Float64(c[i]))
                              for c in cols), ","))
        end
    end
end

# Fold ids 1:K balanced within strata.
function strat_folds(rng, strata, K)
    f = zeros(Int, length(strata))
    off = 0
    for s in sort(unique(strata))
        g = findall(==(s), strata)
        p = g[randperm(rng, length(g))]
        for (i, idx) in enumerate(p)
            f[idx] = mod1(i + off, K)
        end
        off += length(g)
    end
    return f
end

rng = Xoshiro(20260929)
n = 1500
x = 2 .* rand(rng, n) .- 1
Z = randn(rng, n, 4)
gz = sin.(2 .* Z[:, 1]) .+ 0.5 .* Z[:, 2] .^ 2 .+ 0.5 .* Z[:, 3] .+
     0.3 .* Z[:, 1] .* Z[:, 4]
side = Int.(x .>= 0)
v = randn(rng, n)
d = Int.(0.2 .+ 0.5 .* side .+ 0.3 .* Z[:, 1] .+ 0.6 .* v .> 0.5)
y_sharp = 0.5 .* side .+ 0.8 .* x .- 0.3 .* x .^ 2 .+ gz .+ 0.5 .* randn(rng, n)
y_fuzzy = 0.8 .* d .+ 0.8 .* x .- 0.3 .* x .^ 2 .+ gz .+ 0.3 .* v .+ 0.5 .* randn(rng, n)
fold_sharp = strat_folds(rng, side, 5)
fold_fuzzy = strat_folds(rng, 2 .* side .+ d, 5)
write_csv(joinpath(DIR, "flex_data.csv"),
          ["x", "z1", "z2", "z3", "z4", "d", "y_sharp", "y_fuzzy", "fold_sharp",
           "fold_fuzzy"],
          Any[x, Z[:, 1], Z[:, 2], Z[:, 3], Z[:, 4], d, y_sharp, y_fuzzy, fold_sharp,
              fold_fuzzy])
