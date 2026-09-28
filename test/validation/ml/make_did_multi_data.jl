# Generates the simulated staggered-adoption datasets and fold assignments used to
# validate `dml_did_multi` against DoubleML's `DoubleMLDIDMulti` (Python; see
# doubleml_did_multi_reference.py).
#
#   julia test/validation/ml/make_did_multi_data.jl
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

logistic(x) = 1 / (1 + exp(-x))

# Cohort (first treatment period; 0 = never) from a multinomial logit in x.
function draw_cohort(rng, x1, x2)
    s = [0.0, 0.4 * x1 - 0.2 * x2, 0.2 + 0.3 * x2, -0.1 + 0.5 * x1]
    p = exp.(s) ./ sum(exp.(s))
    u = rand(rng)
    k = findfirst(>=(u), cumsum(p))
    return (0, 3, 4, 5)[k === nothing ? 4 : k]
end

att(g, t) = t >= g ? 1.0 + 0.5 * (t - g) + 0.1 * (g - 3) : 0.0

rng = Xoshiro(20260928)
T = 5

# ---- balanced panel -------------------------------------------------------------
N = 800
x1 = randn(rng, N)
x2 = randn(rng, N)
G = [draw_cohort(rng, x1[i], x2[i]) for i in 1:N]
α = 0.5 .* x1 .+ randn(rng, N)
ids, ts, gs, ys, c1, c2, c3 = Int[], Int[], Int[], Float64[], Float64[], Float64[],
                              Float64[]
for i in 1:N, t in 1:T
    x3 = 0.5 * x2[i] + 0.3 * t + randn(rng)              # time-varying covariate
    trend = t * (0.4 * x1[i] + 0.3 * x2[i]^2 - 0.2)       # covariate-specific trends
    y = α[i] + 0.2 * t + trend + 0.3 * x3 + (G[i] > 0 ? att(G[i], t) : 0.0) +
        randn(rng)
    push!(ids, i); push!(ts, t); push!(gs, G[i]); push!(ys, y)
    push!(c1, x1[i]); push!(c2, x2[i]); push!(c3, x3)
end
write_csv(joinpath(DIR, "did_multi_panel.csv"), ["id", "t", "g", "y", "x1", "x2", "x3"],
          Any[ids, ts, gs, ys, c1, c2, c3])
fold_u = strat_folds(rng, G, 5)
write_csv(joinpath(DIR, "did_multi_panel_folds.csv"), ["id", "fold"],
          Any[collect(1:N), fold_u])

# ---- repeated cross-sections -----------------------------------------------------
n = 4000
x1 = randn(rng, n)
x2 = randn(rng, n)
G = [draw_cohort(rng, x1[i], x2[i]) for i in 1:n]
tt = rand(rng, 1:T, n)
y = [0.5 * x1[i] + 0.2 * tt[i] + tt[i] * (0.4 * x1[i] + 0.3 * x2[i]^2 - 0.2) +
     0.3 * (G[i] > 0) + (G[i] > 0 ? att(G[i], tt[i]) : 0.0) + randn(rng) for i in 1:n]
write_csv(joinpath(DIR, "did_multi_rcs.csv"), ["id", "t", "g", "y", "x1", "x2"],
          Any[collect(1:n), tt, G, y, x1, x2])
fold_o = strat_folds(rng, G .* 10 .+ tt, 5)
write_csv(joinpath(DIR, "did_multi_rcs_folds.csv"), ["id", "fold"],
          Any[collect(1:n), fold_o])
