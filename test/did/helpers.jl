# Shared helpers for the DiD tests: simulated designs, R reference data, MC checks.

const DID_VALIDATION_DIR = joinpath(@__DIR__, "..", "validation", "did")

"""Read a numeric CSV written by generate_did_references.R (NA → NaN)."""
function did_read_csv(file)
    lines = readlines(joinpath(DID_VALIDATION_DIR, file))
    hdr = Symbol.(split(lines[1], ","))
    rows = [split(l, ",") for l in lines[2:end]]
    df = DataFrame()
    for (j, h) in enumerate(hdr)
        col = [String(r[j]) for r in rows]
        v = [c == "NA" ? NaN : tryparse(Float64, c) for c in col]
        df[!, h] = any(isnothing, v) ? col : Float64.(v)
    end
    return df
end

"""
Monte Carlo acceptance band for a rejection/coverage rate `rate` from `R` draws with
nominal probability `p`: within `z` binomial standard errors plus a small slack.
"""
mc_close(rate, p, R; z=3.5, slack=0.01) = abs(rate - p) <= z * sqrt(p * (1 - p) / R) + slack

"""
Staggered-adoption panel. `cohorts` are first-treatment periods (0 = never treated)
assigned with probabilities `shares`. Outcome
    y = α_i + λ_t + γ x_i t + τ(g, e) 1{treated} + ε,
with a time-invariant covariate x_i whose distribution differs by cohort when
`confounded = true` (then parallel trends holds only conditional on x).
"""
function sim_staggered(rng; N=200, T=6, cohorts=[0, 3, 5], shares=nothing,
                       effect=(g, e) -> 1.0 + 0.5e, sigma=1.0, confounded=false,
                       gamma=0.0, ar=0.0, time0=2000, assign=nothing,
                       anticipation_effects=false)
    shares = shares === nothing ? fill(1 / length(cohorts), length(cohorts)) : shares
    cum = cumsum(shares)
    g = assign !== nothing ? collect(assign) :
        [cohorts[searchsortedfirst(cum, rand(rng) * cum[end])] for _ in 1:N]
    x = randn(rng, N) .+ (confounded ? 0.8 .* (g .> 0) : 0.0)
    α = randn(rng, N) .+ 0.5 .* x
    λ = cumsum(randn(rng, T)) .* 0.5
    rows = N * T
    unit = repeat(1:N; inner=T)
    per = repeat(1:T; outer=N)
    y = zeros(rows); d = zeros(Int, rows)
    for i in 1:N
        u = 0.0
        for t in 1:T
            r = (i - 1) * T + t
            u = ar * u + sigma * randn(rng)
            treated = g[i] > 0 && t >= g[i]
            d[r] = treated
            # effect(g, e) is applied to treated rows, or to every row of an
            # eventually-treated unit when `anticipation_effects` (effects before g).
            hit = treated || (anticipation_effects && g[i] > 0)
            y[r] = α[i] + λ[t] + gamma * x[i] * t + (hit ? effect(g[i], t - g[i]) : 0.0) + u
        end
    end
    return DataFrame(unit=unit, time=per .+ (time0 - 1), y=y, d=d,
                     g=[g[i] == 0 ? 0 : g[i] + time0 - 1 for i in unit],
                     x=x[unit], xt=x[unit] .+ 0.1 .* randn(rng, rows))
end

"""True ATT(g,t) aggregated as the cohort-share weighted average of post cells."""
function true_simple_att(df, effect)
    tr = df[df.d .== 1, :]
    return mean(effect(r.g - 1999, r.time - r.g) for r in eachrow(tr))
end

shuffle_rows(rng, df) = df[randperm(rng, nrow(df)), :]
