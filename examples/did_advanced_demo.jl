# Advanced difference-in-differences with DrSnow:
#   1. extended TWFE (Wooldridge) for staggered adoption, linear and Poisson;
#   2. sensitivity of an event study to non-parallel trends (Rambachan & Roth);
#   3. non-absorbing, non-binary treatments (de Chaisemartin & D'Haultfœuille);
#   4. a continuous treatment dose (Callaway, Goodman-Bacon & Sant'Anna).
#
# Run with:  julia --project=. examples/did_advanced_demo.jl

using DrSnow
using DataFrames
using Random
using Statistics

rng = MersenneTwister(2026)

# ---------------------------------------------------------------------------
# Simulated staggered adoption: cohorts first treated in periods 4, 6, 8 or never;
# effects grow with exposure and differ across cohorts.
# ---------------------------------------------------------------------------
N, T = 400, 10
cohort = rand(rng, [0, 4, 6, 8], N)
α = randn(rng, N)
λ = cumsum(0.2 .* randn(rng, T))
rows = DataFrame(unit=Int[], year=Int[], first_treat=Int[], d=Int[], y=Float64[],
                 visits=Int[])
for i in 1:N, t in 1:T
    treated = cohort[i] > 0 && t >= cohort[i]
    e = t - cohort[i]
    τ = treated ? 0.5 + 0.2e + 0.1 * (cohort[i] == 4) : 0.0
    y = α[i] + λ[t] + τ + randn(rng)
    μ = exp(0.8 + 0.3α[i] + 0.05t + (treated ? 0.25 : 0.0))
    # Poisson draw by inversion (keeps the example free of extra packages)
    k, p, u = 0, exp(-μ), rand(rng)
    c = p
    while u > c
        k += 1
        p *= μ / k
        c += p
    end
    push!(rows, (i, 2000 + t, cohort[i] == 0 ? 0 : 2000 + cohort[i], Int(treated), y,
                 k))
end

println("=== 1. Extended TWFE (Wooldridge 2021, 2023) ===")
et = did_etwfe(rows, :y, FirstTreated(:first_treat), :unit, :year)
display(aggregate_att(et, :simple))
display(aggregate_att(et, :dynamic))
# Poisson QMLE: parallel trends in logs; effects on the count scale
etp = did_etwfe(rows, :visits, FirstTreated(:first_treat), :unit, :year;
                family=:poisson)
display(aggregate_att(etp, :group))

println("\n=== 2. Sensitivity analysis (Rambachan & Roth 2023) ===")
cs = did_callaway_santanna(rows, :y, FirstTreated(:first_treat), :unit, :year;
                           base_period=:universal, bootstrap=false)
es = aggregate_att(cs, :dynamic; min_e=-4, max_e=3)
display(es)
rm = honest_did(es; restriction=:relative_magnitudes, M=0:0.5:2, rng=rng)
display(rm)
sd = honest_did(es; restriction=:smoothness, M=[0.0, 0.05, 0.1], target=0:3)
display(sd)
println("Breakdown M̄ (relative magnitudes, e = 0): ",
        round(honest_breakdown(es; restriction=:relative_magnitudes, rng=rng);
              digits=3))

println("\n=== 3. Treatments that switch on and off (dCDH 2024) ===")
G, P = 300, 8
dyn = DataFrame(g=Int[], t=Int[], d=Float64[], y=Float64[])
for g in 1:G
    d0 = rand(rng, 0:1)                  # period-one treatment (status quo)
    F = rand(rng, [3, 4, 5, 6, 99])      # first change
    up = d0 == 0 ? true : rand(rng, Bool)
    a = randn(rng)
    for t in 1:P
        d = float(d0)
        if t >= F
            d = d0 + (up ? 1 : -1) * (t >= F + 2 ? 2 : 1)
        end
        push!(dyn, (g, t, d, a + 0.1t + 0.4 * (d - d0) + randn(rng)))
    end
end
es_dyn = did_multiplegt_dyn(dyn, :y, :d, :g, :t; effects=4, placebo=2)
display(es_dyn)
display(pre_trend_test(es_dyn))
println("Average total effect per unit of treatment: ",
        es_dyn.details.average_total_effect)
es_norm = did_multiplegt_dyn(dyn, :y, :d, :g, :t; effects=4, normalized=true)
println("Normalized effects (per unit of cumulative treatment): ",
        round.(coef(es_norm); digits=3))

println("\n=== 4. Continuous dose (Callaway, Goodman-Bacon & Sant'Anna 2024) ===")
n = 800
dose = [rand(rng) < 0.3 ? 0.0 : rand(rng) for _ in 1:n]
a = randn(rng, n)
cont = DataFrame(id=vcat(1:n, 1:n), year=vcat(fill(1, n), fill(2, n)),
                 dose=vcat(dose, dose),
                 y=vcat(a .+ randn(rng, n),
                        a .+ 0.3 .+ (1.5 .* dose .- 0.5 .* dose .^ 2) .+
                        randn(rng, n)))
cd = did_continuous(cont, :y, :dose, :id, :year; degree=2)
display(cd)
band = confint(cd; curve=:acrt, uniform=true, rng=rng)
println("ACRT(d) at d = ", round(cd.dose[25]; digits=2), ": ",
        round(cd.acrt[25]; digits=3), " (uniform band ",
        round.(band[25, :]; digits=3), "); causal only under strong parallel trends")
