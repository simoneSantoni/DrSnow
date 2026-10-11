# Generates the simulated longitudinal dataset used to validate the g-methods of
# DrSnow against the R packages ltmle and lmtp (see ltmle_reference.R).
#
#   julia --project=. test/validation/longitudinal/make_data.jl
#
# Three periods with treatment–confounder feedback: the continuous covariate l1 and
# the binary covariate l2 of period t depend on the previous treatment, and the
# treatment of period t depends on (l1, l2) of the period, the baseline covariates
# (w1, w2) and the previous treatment. The final outcomes y_cont (continuous) and
# y_bin (binary) depend on the whole treatment and covariate history. The CSV is in
# long format (one row per unit and period) with the outcomes repeated in every row.

using Random

const DIR = @__DIR__

logistic(x) = 1 / (1 + exp(-x))

rng = Xoshiro(20261011)
n, K = 1000, 3

id = Int[]
t = Int[]
w1 = Float64[]
w2 = Float64[]
l1 = Float64[]
l2 = Float64[]
a = Float64[]
y_cont = Float64[]
y_bin = Float64[]
for i in 1:n
    wi1 = randn(rng)
    wi2 = Float64(rand(rng) < 0.4)
    a_prev, l1_prev = 0.0, 0.0
    suma, suml1 = 0.0, 0.0
    L1 = zeros(K)
    L2 = zeros(K)
    A = zeros(K)
    for s in 1:K
        L1[s] = 0.4 * a_prev + 0.3 * wi1 + 0.2 * l1_prev + randn(rng)
        L2[s] = Float64(rand(rng) < logistic(-0.5 + 0.6 * a_prev + 0.3 * L1[s]))
        p = logistic(-0.3 + 0.5 * L1[s] + 0.4 * L2[s] - 0.3 * wi1 + 0.4 * wi2 + a_prev)
        A[s] = Float64(rand(rng) < p)
        suma += A[s]
        suml1 += L1[s]
        a_prev, l1_prev = A[s], L1[s]
    end
    yc = 1 + 0.5 * suma + 0.3 * suml1 + 0.2 * wi1 + 0.4 * L2[K] + randn(rng)
    yb = Float64(rand(rng) < logistic(-1 + 0.5 * suma + 0.3 * L1[K] + 0.2 * wi1 +
                                      0.3 * L2[K]))
    for s in 1:K
        push!(id, i); push!(t, s); push!(w1, wi1); push!(w2, wi2)
        push!(l1, L1[s]); push!(l2, L2[s]); push!(a, A[s])
        push!(y_cont, yc); push!(y_bin, yb)
    end
end

open(joinpath(DIR, "longitudinal_data.csv"), "w") do io
    println(io, "id,t,w1,w2,l1,l2,a,y_cont,y_bin")
    for r in eachindex(id)
        println(io, join([string(id[r]), string(t[r]), repr(w1[r]), repr(w2[r]), repr(l1[r]),
                          repr(l2[r]), repr(a[r]), repr(y_cont[r]), repr(y_bin[r])], ","))
    end
end
println("wrote longitudinal_data.csv ($(n) units × $(K) periods)")
