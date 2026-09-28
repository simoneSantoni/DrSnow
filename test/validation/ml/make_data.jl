# Generates the simulated datasets and fold assignments used to validate the DML
# estimators of DrSnow against the R package DoubleML (see doubleml_reference.R).
#
#   julia --project=. test/validation/ml/make_data.jl
#
# The CSV files are committed; this script documents how they were produced.

using Random
using LinearAlgebra

const DIR = @__DIR__

function write_csv(path, header, cols)
    open(path, "w") do io
        println(io, join(header, ","))
        for i in eachindex(cols[1])
            println(io, join((repr(Float64(c[i])) for c in cols), ","))
        end
    end
end

function folds_matrix(rng, n, K, R; strata=nothing)
    F = zeros(Int, n, R)
    for r in 1:R
        groups = strata === nothing ? [collect(1:n)] :
                 [findall(==(s), strata) for s in sort(unique(strata))]
        off = 0
        for g in groups
            p = g[randperm(rng, length(g))]
            for (i, idx) in enumerate(p)
                F[idx, r] = mod1(i + off, K)
            end
            off += length(g)
        end
    end
    return F
end

logistic(x) = 1 / (1 + exp(-x))

rng = Xoshiro(20260927)
n = 500
p = 5

# Common covariates with correlation 0.5^|j-k|
Σ = [0.5^abs(j - k) for j in 1:p, k in 1:p]
L = cholesky(Σ).L
X = randn(rng, n, p) * L'
xs = [X[:, j] for j in 1:p]

# PLR (two continuous treatments)
d1 = 0.8 .* X[:, 1] .+ 0.4 .* X[:, 3] .+ randn(rng, n)
d2 = 0.5 .* X[:, 2] .- 0.3 .* X[:, 4] .+ 0.3 .* d1 .+ randn(rng, n)
y_plr = 0.5 .* d1 .- 0.3 .* d2 .+ X[:, 1] .+ 0.5 .* X[:, 2] .^ 2 .+ randn(rng, n)

# IRM (binary treatment)
m_irm = logistic.(0.5 .* X[:, 1] .- 0.5 .* X[:, 2])
d_irm = Float64.(rand(rng, n) .< m_irm)
y_irm = 1.0 .* d_irm .+ X[:, 1] .+ 0.5 .* X[:, 3] .+ 0.5 .* d_irm .* X[:, 2] .+
        randn(rng, n)

# PLIV (continuous treatment, two instruments)
z1 = 0.5 .* X[:, 1] .+ randn(rng, n)
z2 = -0.3 .* X[:, 2] .+ randn(rng, n)
u = randn(rng, n)
d_pliv = 0.7 .* z1 .+ 0.4 .* z2 .+ 0.5 .* X[:, 1] .+ u .+ 0.5 .* randn(rng, n)
y_pliv = 0.6 .* d_pliv .+ X[:, 1] .- 0.5 .* X[:, 4] .+ 0.8 .* u .+ randn(rng, n)

# IIVM (binary instrument and treatment; two-sided and one-sided non-compliance)
z_iv = Float64.(rand(rng, n) .< logistic.(0.4 .* X[:, 1]))
v = randn(rng, n)
d_iv = Float64.((0.2 .+ 1.2 .* z_iv .+ 0.4 .* X[:, 2] .+ v) .> 0.8)
d_os = z_iv .* Float64.((0.5 .+ 0.4 .* X[:, 2] .+ v) .> 0.0)
y_iv = 0.8 .* d_iv .+ X[:, 1] .+ 0.5 .* X[:, 3] .+ 0.6 .* v .+ randn(rng, n)
y_os = 0.8 .* d_os .+ X[:, 1] .+ 0.5 .* X[:, 3] .+ 0.6 .* v .+ randn(rng, n)

header = vcat(["x$j" for j in 1:p],
              ["d1", "d2", "y_plr", "d_irm", "y_irm", "z1", "z2", "d_pliv", "y_pliv",
               "z_iv", "d_iv", "y_iv", "d_os", "y_os"])
cols = vcat(xs, [d1, d2, y_plr, d_irm, y_irm, z1, z2, d_pliv, y_pliv, z_iv, d_iv, y_iv,
                 d_os, y_os])
write_csv(joinpath(DIR, "dml_data.csv"), header, cols)

# Folds: 5 folds × 2 repetitions, stratified by (z_iv, d_iv) so every training fold
# contains all instrument/treatment cells (DoubleML accepts any partition).
F = folds_matrix(rng, n, 5, 2; strata=2 .* z_iv .+ d_iv .+ 4 .* d_irm)
write_csv(joinpath(DIR, "dml_folds.csv"), ["fold_rep1", "fold_rep2"],
          [F[:, 1], F[:, 2]])
println("wrote dml_data.csv and dml_folds.csv")

# ------------------------------------------------------------------ 2×2 DiD
# Panel (wide): baseline covariates, group, outcomes in both periods; the trend
# depends on covariates so an unconditional DiD is biased.
nd = 1000
Xd = randn(rng, nd, 3)
dg = Float64.(rand(rng, nd) .< logistic.(-0.3 .+ 0.6 .* Xd[:, 1] .- 0.4 .* Xd[:, 2]))
α = 0.5 .* Xd[:, 1] .+ randn(rng, nd)
y0 = α .+ Xd[:, 2] .+ randn(rng, nd)
y1 = α .+ Xd[:, 2] .+ 1.0 .+ 0.8 .* Xd[:, 1] .+ 0.4 .* Xd[:, 3] .+ 1.5 .* dg .+
     randn(rng, nd)
write_csv(joinpath(DIR, "did_panel.csv"), ["id", "x1", "x2", "x3", "d", "y0", "y1"],
          [collect(1.0:nd), Xd[:, 1], Xd[:, 2], Xd[:, 3], dg, y0, y1])

# Repeated cross-sections
nc = 1500
Xc = randn(rng, nc, 3)
dc = Float64.(rand(rng, nc) .< logistic.(-0.3 .+ 0.6 .* Xc[:, 1] .- 0.4 .* Xc[:, 2]))
tc = Float64.(rand(rng, nc) .< 0.5)
yc = 0.5 .* Xc[:, 1] .+ Xc[:, 2] .+ 0.3 .* dc .+
     tc .* (1.0 .+ 0.8 .* Xc[:, 1] .+ 0.4 .* Xc[:, 3] .+ 1.5 .* dc) .+ randn(rng, nc)
write_csv(joinpath(DIR, "did_rcs.csv"), ["x1", "x2", "x3", "d", "post", "y"],
          [Xc[:, 1], Xc[:, 2], Xc[:, 3], dc, tc, yc])
println("wrote did_panel.csv and did_rcs.csv")
