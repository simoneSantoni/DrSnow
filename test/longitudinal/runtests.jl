# Tests of the longitudinal g-methods area (src/longitudinal): treatment regimes,
# data handling, the four estimators, contrasts and MSM projections, parity with
# ltmle / lmtp, and Monte Carlo coverage and double-robustness checks.

const LG_VALIDATION = joinpath(@__DIR__, "..", "validation", "longitudinal")

"""Minimal CSV reader for the committed validation files."""
function lg_read_csv(file)
    lines = readlines(joinpath(LG_VALIDATION, file))
    header = [strip(h, '"') for h in split(lines[1], ',')]
    cols = [String[] for _ in header]
    for l in lines[2:end]
        isempty(strip(l)) && continue
        for (j, c) in enumerate(split(l, ','))
            push!(cols[j], strip(c, '"'))
        end
    end
    df = DataFrame()
    for (h, c) in zip(header, cols)
        v = tryparse.(Float64, c)
        df[!, Symbol(h)] = any(isnothing, v) ? c : Float64.(v)
    end
    return df
end

"""Binomial tolerance for Monte Carlo coverage checks with `reps` replications."""
lg_cover_tol(reps; level=0.95) = 3.5 * sqrt(level * (1 - level) / reps) + 0.01

lg_logistic(x) = 1 / (1 + exp(-x))

"""
Simulated long panel with treatment–confounder feedback. `outcome` and `treatment`
select the functional forms of the outcome mean (`:linear`, `:exp`, `:interaction`)
and of the treatment mechanism (`:linear`, `:quadratic`, `:step`); `y_binary` makes the outcome Bernoulli. Returns the
DataFrame and a function `truth(regime)` that simulates E[Y^d] with `n_truth`
counterfactual draws.
"""
function lg_sim(rng, n; K=2, outcome=:linear, treatment=:linear, y_binary=false)
    mean_y(a, l, w) = outcome === :linear ?
        1 + 0.5 * sum(a) + 0.3 * sum(l) + 0.2 * w :
        outcome === :exp ? 0.5 * sum(a) + sum(exp, l) + 0.2 * w :
        1 + 0.5 * sum(a) + 0.3 * sum(l) + 3.0 * a[end] * l[end] + 0.2 * w
    prob_a(l, w, a_prev) = treatment === :linear ?
        lg_logistic(0.5 * l - 0.4 * w + 0.8 * a_prev) :
        treatment === :quadratic ? lg_logistic(-0.8 + 1.5 * l^2 - 0.4 * w + 0.8 * a_prev) :
        lg_logistic(-1.5 + 3.0 * (l > 0) - 0.4 * w + 0.8 * a_prev)
    draw_y(rng, a, l, w) = y_binary ?
        Float64(rand(rng) < lg_logistic(mean_y(a, l, w) - 2)) :
        mean_y(a, l, w) + randn(rng)
    id = repeat(1:n; inner=K)
    t = repeat(1:K, n)
    w = repeat(randn(rng, n); inner=K)
    l = zeros(n * K)
    a = zeros(n * K)
    y = Vector{Union{Missing,Float64}}(missing, n * K)
    for i in 1:n
        a_prev = 0.0
        L = zeros(K)
        A = zeros(K)
        for s in 1:K
            r = (i - 1) * K + s
            L[s] = 0.5 * a_prev + 0.3 * w[r] + randn(rng)
            A[s] = Float64(rand(rng) < prob_a(L[s], w[r], a_prev))
            l[r], a[r] = L[s], A[s]
            a_prev = A[s]
        end
        y[(i - 1) * K + K] = draw_y(rng, A, L, w[(i - 1) * K + 1])
    end
    df = DataFrame(; id, t, w, l, a, y)
    truth = function (regime; n_truth=400_000, seed=777)
        r2 = StableRNG(seed)
        acc = 0.0
        for i in 1:n_truth
            wi = randn(r2)
            a_prev = 0.0
            L = zeros(K)
            A = zeros(K)
            for s in 1:K
                L[s] = 0.5 * a_prev + 0.3 * wi + randn(r2)
                A[s] = regime isa StaticRegime ? regime.actions[s] :
                       Float64(regime.rule((t=s, baseline=(w=wi,),
                                            covariates=[(l=L[k],) for k in 1:s],
                                            treatment=A[1:(s - 1)])))
                a_prev = A[s]
            end
            acc += y_binary ? lg_logistic(mean_y(A, L, wi) - 2) : mean_y(A, L, wi)
        end
        acc / n_truth
    end
    return df, truth
end

include("test_regimes.jl")
include("test_estimators.jl")
include("test_reference.jl")
include("test_montecarlo.jl")
