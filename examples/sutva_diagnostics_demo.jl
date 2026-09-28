"""
Interference (SUTVA) tools in DrSnow: three simulated studies with known spillovers.

1. Spatial natural experiment: a policy adopted by some counties in year 5 raises the
   outcome of treated counties and spills over onto untreated counties within 40 km.
   Ring DiD (Butts 2021) with clustered and Conley standard errors, a ring event
   study, and a randomization check of the "as-if random" assignment claim.
2. Network experiment: a completely randomized treatment on a friendship network with
   first-order spillovers. Design-based (Aronow–Samii) estimates, an exact Fisher test
   of no spillovers (Athey–Eckles–Imbens), and an exposure regression with network
   HAC standard errors.
3. Two-stage randomized (saturation) design with partial interference within
   villages (Hudgens–Halloran).

Run with `julia --project=. examples/sutva_diagnostics_demo.jl`.
"""

using DrSnow
using DataFrames
using FixedEffectModels: reg, @formula, fe
using Random
using Statistics

rng = Xoshiro(20260927)

function report(r; truth=nothing)
    println(method_name(r))
    ct = coeftable(r)
    show(stdout, MIME"text/plain"(), ct)
    println()
    if truth !== nothing
        println("True values: ", join(["$(k) = $(v)" for (k, v) in truth], ", "))
    end
    println()
end

println("=" ^ 78)
println("1. Spatial natural experiment with ring spillovers")
println("=" ^ 78)

# 1,200 counties in a 4° × 6° box; 8% adopt a policy in year 5 of 8.
n_counties = 1200
n_years = 8
lat = 38 .+ 4 .* rand(rng, n_counties)
lon = -100 .+ 6 .* rand(rng, n_counties)
ids = ["county_$(i)" for i in 1:n_counties]
adopter = rand(rng, n_counties) .< 0.08
counties = DataFrame(county=ids, lat=lat, lon=lon, adopter=adopter)

panel = DataFrame(county=repeat(ids, n_years), year=repeat(2011:2018; inner=n_counties))
panel = leftjoin(panel, counties; on=:county)
panel.policy = Int.(panel.adopter .& (panel.year .>= 2015))

s = SpatialStructure(counties, :county; lat=:lat, lon=:lon)       # haversine, km
rings = RingExposure([20.0, 40.0])                               # (0, 20], (20, 40] km
ex = compute_exposure(panel, :policy, s, rings; unit=:county, time=:year)

# Outcome: county and year effects, direct effect 2.0, spillovers onto untreated
# counties of 1.0 (nearest treated county within 20 km) and 0.4 (20–40 km).
# Shocks are spatially correlated within each year.
cfe = Dict(zip(ids, randn(rng, n_counties)))
yfe = Dict(zip(2011:2018, randn(rng, n_years)))
shock = Dict{Tuple{String,Int},Float64}()
W = neighbor_matrix(s; radius=30.0)
for yr in 2011:2018
    e = randn(rng, n_counties)
    v = 0.5 .* (e .+ (W * e) ./ max.(1, vec(sum(W; dims=2))))
    for (i, c) in enumerate(ids)
        shock[(c, yr)] = v[i]
    end
end
panel.y = [cfe[c] for c in panel.county] .+ [yfe[t] for t in panel.year] .+
          2.0 .* panel.policy .+
          (1 .- panel.policy) .* (1.0 .* ex.ring_0_20 .+ 0.4 .* ex.ring_20_40) .+
          [shock[(c, t)] for (c, t) in zip(panel.county, panel.year)]

post = panel.year .>= 2015
n_ring1 = count(post .& (panel.policy .== 0) .& (ex.ring_0_20 .== 1)) ÷ 4
n_ring2 = count(post .& (panel.policy .== 0) .& (ex.ring_20_40 .== 1)) ÷ 4
println("Treated counties: $(count(adopter)). Untreated counties within 20 km of a " *
        "treated county after adoption: $(n_ring1); 20–40 km: $(n_ring2).\n")

println("Naive TWFE DiD (all untreated counties as controls):")
naive = reg(panel, @formula(y ~ policy + fe(county) + fe(year)), Vcov.cluster(:county))
println("  policy = ", round(coef(naive)[1]; digits=3), " (se ",
        round(stderror(naive)[1]; digits=3), "); true direct effect 2.0. The control " *
        "group\n  is contaminated by spillovers, so the naive contrast is biased " *
        "downwards.\n")

r = spillover_did(panel, :y, :policy, s; unit=:county, time=:year, exposure=rings)
report(r; truth=["direct" => 2.0, "spill_control:ring_0_20" => 1.0,
                 "spill_control:ring_20_40" => 0.4, "spill_treated:*" => 0.0])

println("Same regression with Conley standard errors (uniform kernel, 60 km,")
println("within-year spatial correlation plus serial correlation within county):")
rc = spillover_did(panel, :y, :policy, s; unit=:county, time=:year, exposure=rings,
                   vcov=ConleyVcov(s; unit=:county, cutoff=60.0, time=:year))
report(rc)

println("Ring event study (reference year: one year before adoption / exposure):")
es = spillover_event_study(panel, :y, :policy, s; unit=:county, time=:year,
                           exposure=rings, leads=3, lags=3)
show(stdout, MIME"text/plain"(), es.table)
println("\n")
show(stdout, MIME"text/plain"(), spillover_pretrend_test(es))
println()

println("Is the policy's geography consistent with complete randomization?")
cross = counties
cross.d = Int.(cross.adopter)
mt = treatment_moran_test(cross, :d, s, CompleteRandomization(n_counties, count(adopter));
                          unit=:county, radius=40.0, draws=999, rng=rng)
show(stdout, MIME"text/plain"(), mt)
println()

println("=" ^ 78)
println("2. Randomized experiment on a friendship network")
println("=" ^ 78)

n = 500
A = zeros(n, n)
for i in 1:n, j in (i + 1):n
    rand(rng) < 4 / n && (A[i, j] = A[j, i] = 1)
end
people = ["p$(i)" for i in 1:n]
g = NetworkStructure(people, A)
design = CompleteRandomization(n, 150)
z = draw_assignment(rng, design)
exposed = A * z .> 0
# direct effect 1.0; spillover 0.5 onto untreated people with a treated friend
y0 = 2 .+ randn(rng, n)
df = DataFrame(id=people, z=Int.(z),
               y=y0 .+ 1.0 .* z .+ 0.5 .* (1 .- z) .* exposed .+ 0.3 .* randn(rng, n))
iso = count(==(0), vec(sum(A; dims=2)))
println("$(n) people, $(Int(sum(A) / 2)) friendships, $(iso) isolates (outside the " *
        "estimand: their exposure is undefined).\n")

P = exposure_probabilities(g, design; draws=5000, rng=rng)
println("Exposure-condition positivity:")
show(stdout, MIME"text/plain"(), exposure_positivity(P))
println("\n")
ae = exposure_effects(df, :y, :z, g, P; unit=:id, positivity=:restrict)
report(ae; truth=["treated_exposed - control_unexposed" => 1.0,
                  "treated_unexposed - control_unexposed" => 1.0,
                  "control_exposed - control_unexposed" => 0.5])

ft = spillover_fisher_test(df, :y, :z, g, design; unit=:id, draws=999, rng=rng)
show(stdout, MIME"text/plain"(), ft)
println()

println("Exposure regression (share of treated friends) with network HAC SEs:")
er = exposure_regression(df, :y, :z, g; unit=:id,
                         vcov=NetworkHACVcov(g; unit=:id, bandwidth=2))
report(er)
println("(The DGP depends on any treated friend, not on the share, so these slopes")
println(" are linear-projection coefficients, not the design-based contrasts above.)\n")

println("=" ^ 78)
println("3. Two-stage randomized design (partial interference within villages)")
println("=" ^ 78)

n_villages = 60
size_v = 12
village = repeat(["v$(k)" for k in 1:n_villages]; inner=size_v)
d2 = TwoStageRandomization(village, [0.25, 0.75], [30, 30])
z2 = draw_assignment(rng, d2)
share = Dict(v => mean(z2[village .== v]) for v in unique(village))
sat = [share[v] for v in village]
# direct effect 1.0; every villager gains 0.8 × village saturation
base = randn(rng, length(village)) .+ repeat(randn(rng, n_villages); inner=size_v)
vil = DataFrame(village=village, z=Int.(z2), saturation=sat,
                y=base .+ 1.0 .* z2 .+ 0.8 .* sat)
te = two_stage_effects(vil, :y, :z; group=:village, saturation=:saturation)
# indirect = 0.8 × (0.75 − 0.25); total = 1.0 + 0.4; overall = 0.5 × 1.0 + 0.4
report(te; truth=["direct(α)" => 1.0, "indirect(0.75 vs 0.25)" => 0.4,
                  "total(0.75 vs 0.25)" => 1.4, "overall(0.75 vs 0.25)" => 0.9])
