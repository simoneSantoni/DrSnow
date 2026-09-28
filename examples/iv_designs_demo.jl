# IV research designs with DrSnow: many instruments (incl. clustered data), judge
# designs, shift-share instruments (incl. panels), marginal treatment effects and
# their bounds, robust weak-IV inference, and extrapolation beyond compliers.
#
# Run from the repository root:
#     julia --project=. examples/iv_designs_demo.jl
#
# Every data set below is simulated, so the true parameters are known and printed
# next to the estimates. Printed test results state statistics and p-values only.

using DrSnow, DataFrames, Random, Statistics, Printf

rng = Xoshiro(20260927)
section(title) = (println(); println("="^78); println(title); println("="^78))
show_plain(x) = (show(stdout, MIME"text/plain"(), x); println())
normcdf(x) = DrSnow.cdf(DrSnow.Normal(), x)

# ----------------------------------------------------------------------------
# 1. Many instruments: 2SLS bias, LIML / Fuller / HFUL / UJIVE, jackknife AR
# ----------------------------------------------------------------------------
section("1. Many weak instruments (true effect = 1.0)")
n, K = 1000, 40
Z = randn(rng, n, K)
v = randn(rng, n)
u = 0.7 .* v .+ sqrt(1 - 0.7^2) .* randn(rng, n)
u .*= 0.5 .+ abs.(Z[:, 1])                          # heteroskedastic errors
d = Z * fill(0.08, K) .+ v
y = 1.0 .* d .+ u
many = DataFrame(y=y, d=d)
zs = [Symbol("z", j) for j in 1:K]
for j in 1:K
    many[!, zs[j]] = Z[:, j]
end

tsls = late_2sls(many, :y, :d, zs)
@printf("First-stage effective F: %.2f with %d instruments\n",
        tsls.first_stage.effective_F, K)
fits = [("2SLS", tsls),
        ("LIML (Bekker SE)", kclass_iv(many, :y, :d, zs; se=:bekker)),
        ("Fuller(1) (Bekker SE)", kclass_iv(many, :y, :d, zs; method=:fuller,
                                            se=:bekker)),
        ("HFUL (many-IV robust SE)", kclass_iv(many, :y, :d, zs; method=:hful)),
        ("UJIVE (many-IV robust SE)", jive(many, :y, :d, zs; se=:many_robust))]
for (name, r) in fits
    ci = confint(r)
    @printf("  %-26s %7.3f  (se %.3f)  95%% CI [%.3f, %.3f]\n", name, coef(r)[1],
            stderror(r)[1], ci[1, 1], ci[1, 2])
end
jar = weak_iv_confidence_set(tsls; method=:jackknife_ar)
println("Jackknife AR (Mikusheva & Sun 2022) 95% set: ", jar.intervals)
show_plain(weak_iv_test(tsls; method=:jackknife_ar, beta0=0.0))
# heteroskedasticity-robust CLR (Kleibergen 2005) with a handful of instruments
few = late_2sls(many, :y, :d, zs[1:4])
println("Robust CLR 95% set (4 instruments): ",
        weak_iv_confidence_set(few; method=:clr).intervals)

# LIML with Hansen–Hausman–Newey standard errors: group-indicator instruments of
# very different sizes and skewed errors
ng, Kg = 1200, 80
grp = vcat(repeat(1:Kg; inner=3), rand(rng, 1:Kg, ng - 3Kg))
e1 = (randn(rng, ng) .^ 2 .+ randn(rng, ng) .^ 2 .- 2) ./ 2
dg = 0.35 .* randn(rng, Kg)[grp] .+ 0.6 .* e1 .+ 0.8 .* (-log.(rand(rng, ng)) .- 1)
groups = DataFrame(y=0.5 .* dg .+ e1, d=dg)
gz = [Symbol("g", j) for j in 2:Kg]
for j in 2:Kg
    groups[!, gz[j - 1]] = Float64.(grp .== j)
end
for se in (:standard, :bekker, :hhn)
    lf = kclass_iv(groups, :y, :d, gz; method=:liml, se=se, vcov=Vcov.simple())
    @printf("  LIML (true 0.5) %.3f, se (%s) %.4f\n", coef(lf)[1], se, stderror(lf)[1])
end

# Clustered data: cluster shocks in both equations and instruments concentrated
# within clusters. UJIVE keeps the own-cluster bias; CJIVE removes it.
nc, Gc, Kc = 3000, 150, 60
cl = rand(rng, 1:Gc, nc)
gc = [mod(c - 1 + rand(rng, 0:4), Kc) + 1 for c in cl]
shock = randn(rng, Gc)
dc = 0.6 .* randn(rng, Kc)[gc] .+ shock[cl] .+ randn(rng, nc)
clustered = DataFrame(y=0.5 .* dc .+ 0.8 .* shock[cl] .+ randn(rng, nc), d=dc, cl=cl)
cz = [Symbol("c", j) for j in 2:Kc]
for j in 2:Kc
    clustered[!, cz[j - 1]] = Float64.(gc .== j)
end
for m in (:ujive, :cjive)
    jf = jive(clustered, :y, :d, cz; method=m, cluster=:cl)
    @printf("  %-5s (true 0.5) %.3f (cluster se %.3f)\n", uppercase(string(m)),
            coef(jf)[1], stderror(jf)[1])
end

# ----------------------------------------------------------------------------
# 2. Judge design
# ----------------------------------------------------------------------------
section("2. Judge design: randomly assigned judges within courts")
n_courts, per_court, cases = 12, 8, 50
J = n_courts * per_court
court_of = repeat(1:n_courts; inner=per_court)
leniency = 0.2 .+ 0.5 .* rand(rng, J)
N = J * cases
court = rand(rng, 1:n_courts, N)
judge = [(c - 1) * per_court + rand(rng, 1:per_court) for c in court]
prior = randn(rng, N)                                 # case characteristic
resist = normcdf.(0.7 .* prior .+ randn(rng, N))       # latent resistance to treatment
detained = Float64.(resist .< leniency[judge])
effect = 1.0 .+ 0.8 .* (resist .- 0.5)                # heterogeneous effects
outcome = 0.5 .* prior .+ detained .* effect .+ randn(rng, N)
cases_df = DataFrame(y=outcome, d=detained, judge=judge, court=court, prior=prior,
                     group=Float64.(rand(rng, N) .< 0.5))
# marginal cases: resistance between the least and most lenient judges
marg = (resist .> minimum(leniency)) .& (resist .< maximum(leniency))
@printf("Average effect among cases whose decision depends on the judge: %.3f\n",
        mean(effect[marg]))
r = judge_iv(cases_df, :y, :d, :judge; strata=[:court])
show_plain(r)
ru = judge_iv(cases_df, :y, :d, :judge; strata=[:court], method=:ujive)
@printf("UJIVE with judge indicators: %.3f (se %.3f)\n", coef(ru)[1], stderror(ru)[1])
show_plain(judge_balance_test(cases_df, :d, :judge, [:prior, :group]; strata=[:court]))
fll = judge_validity_test(cases_df, :y, :d, :judge; strata=[:court], rng=Xoshiro(1))
show_plain(fll)
@printf("FLL fit component: χ²(%d) = %.2f (p = %.3f); slope component p = %.3f\n",
        fll.details.fit_dof, fll.details.fit_statistic, fll.details.fit_pvalue,
        fll.details.slope_pvalue)
m = judge_subsample_monotonicity(cases_df, :d, :judge, [:group]; strata=[:court])
show_plain(m)
show_plain(m.details.table)

# ----------------------------------------------------------------------------
# 3. Shift-share instrument
# ----------------------------------------------------------------------------
section("3. Shift-share IV (true effect = 1.5)")
nr, ns = 300, 40
raw = rand(rng, nr, ns) .^ 3
S = raw ./ sum(raw; dims=2)
g = randn(rng, ns)
sector_err = randn(rng, ns)
vr = randn(rng, nr)
treat = 2.0 .* (S * g) .+ vr
resp = 1.5 .* treat .+ 2 .* (S * sector_err) .+ 0.5 .* vr .+ randn(rng, nr)
regions = DataFrame(y=resp, x=treat, pop=exp.(0.5 .* randn(rng, nr)))
sh = [Symbol("s", k) for k in 1:ns]
for k in 1:ns
    regions[!, sh[k]] = S[:, k]
end
ss = shift_share_iv(regions, :y, :x, sh, g; weights=:pop)
show_plain(ss)
show_plain(ss.inference)
rw = rotemberg_weights(regions, :y, :x, sh, g; weights=:pop)
show_plain(rw)
# counterfactual shocks: permutations of the observed shocks
draws = hcat([g[randperm(rng, ns)] for _ in 1:499]...)
ssr = shift_share_iv(regions, :y, :x, sh, g; weights=:pop, shock_draws=draws)
@printf("Recentered estimate %.3f; randomization-inference p-value for β = 0: %.4f\n",
        coef(ssr)[1], ssr.ri.pvalue)
println("Randomization-inference 95% set: ", ssr.ri.set.intervals)
# the overidentified 2SLS with the shares as separate instruments
show_plain(rotemberg_weights(regions, :y, :x, sh, nothing; weights=:pop,
                             estimator=:tsls))
# two periods with period-specific shocks: weights aggregated by sector (GPSS)
panel = vcat(regions, regions)
panel.year = repeat([2000, 2010]; inner=nr)
g2 = randn(rng, ns)
panel.x[(nr + 1):end] = 2.0 .* (S * g2) .+ vr
panel.y[(nr + 1):end] = 1.5 .* panel.x[(nr + 1):end] .+ randn(rng, nr)
rp = rotemberg_weights(panel, :y, :x, sh, Dict(2000 => g, 2010 => g2); period=:year,
                       fe=[:year], weights=:pop)
show_plain(rp)

# ----------------------------------------------------------------------------
# 4. Marginal treatment effects
# ----------------------------------------------------------------------------
section("4. Marginal treatment effects (normal selection model)")
nm = 4000
zc = randn(rng, nm)
xc = randn(rng, nm)
V = randn(rng, nm)
Dm = Float64.(V .<= 0.8 .* zc .+ 0.3 .* xc)
Y0 = xc .+ 0.3 .* V .+ randn(rng, nm)
Y1 = 1.0 .+ 1.5 .* xc .- 0.5 .* V .+ randn(rng, nm)
mdf = DataFrame(y=ifelse.(Dm .== 1, Y1, Y0), d=Dm, z=zc, x=xc)
@printf("Sample ATE %.3f, ATT %.3f, ATU %.3f; MTE(x̄, u) = 1 + 0.5x̄ − 0.8Φ⁻¹(u)\n",
        mean(Y1 .- Y0), mean((Y1 .- Y0)[Dm .== 1]), mean((Y1 .- Y0)[Dm .== 0]))
ps = mte_propensity(mdf, :d, [:z]; covariates=[:x])
@printf("Common support of the propensity score: [%.3f, %.3f]\n", ps.support.lower,
        ps.support.upper)
# policy: raise every propensity by 0.05, capped at the upper end of the support
# (the semiparametric PRTE is identified only for policies within the support)
hi = ps.support.upper
for method in (:normal, :polynomial, :semiparametric)
    mr = mte(mdf, :y, :d, [:z]; covariates=[:x], method=method,
             policy=p -> min.(p .+ 0.05, hi), n_bootstrap=100, rng=Xoshiro(2))
    show_plain(mr)
end

# ----------------------------------------------------------------------------
# 5. Bounds on the ATE and ATT with a binary instrument (Mogstad, Santos &
#    Torgovitsky 2018)
# ----------------------------------------------------------------------------
section("5. MTE bounds with a binary instrument")
nb = 5000
zb = Float64.(rand(rng, nb) .< 0.5)
ub = rand(rng, nb)
db = Float64.(ub .< ifelse.(zb .== 1, 0.65, 0.35))
yb = Float64.(rand(rng, nb) .< 0.2 .+ 0.3 .* (1 .- ub) .+ 0.25 .* db .* (1 .- ub))
bdf = DataFrame(y=yb, d=db, z=zb)
@printf("True ATE %.3f; LATE for compliers %.3f\n", mean(0.25 .* (1 .- ub)),
        mean((0.25 .* (1 .- ub))[(ub .>= 0.35) .& (ub .< 0.65)]))
sat = (regressors=[:d, :z, (:d, :z)],)           # saturated IV-like estimands
for (label, kw) in (("nonparametric (constant splines)",
                     (basis=:spline, degree=0, knots=[0.35, 0.65])),
                    ("Bernstein, degree 3", (basis=:bernstein, degree=3)),
                    ("Bernstein 3, decreasing MTRs",
                     (basis=:bernstein, degree=3, m0_monotone=:decreasing,
                      m1_monotone=:decreasing)))
    for tg in (:ate, :att)
        b = mte_bounds(bdf, :y, :d, :z; target=tg, ivlike=sat, kw...)
        @printf("  %-34s %s bounds [%.3f, %.3f]\n", label, uppercase(string(tg)),
                b.lower, b.upper)
    end
end
bl = mte_bounds(bdf, :y, :d, :z; target=:late, late_from=(z=0,), late_to=(z=1,),
                ivlike=sat)
@printf("  LATE for the instrument change: [%.3f, %.3f] (point identified: %s)\n",
        bl.lower, bl.upper, bl.point_identified)

# ----------------------------------------------------------------------------
# 6. Beyond compliers with a continuous covariate; Kitagawa's test given cells
# ----------------------------------------------------------------------------
section("6. Parametric LATE extrapolation and the conditional Kitagawa test")
ne = 4000
xe = 4 .* (rand(rng, ne) .- 0.5)
cell = rand(rng, 0:2, ne)
ze = Float64.(rand(rng, ne) .< ifelse.(cell .== 1, 0.7, 0.35))   # Z random given cell
ue = rand(rng, ne)
at = ue .< 0.15
co = .!at .& (ue .< 0.15 .+ 0.85 .* (0.45 .+ 0.1 .* xe))
de = Float64.(at .| (co .& (ze .== 1)))
ye = xe .+ cell .+ randn(rng, ne) .+ de .* (2 .+ xe)             # effect 2 + x
edf = DataFrame(y=ye, d=de, z=ze, x=xe, cell=string.(cell))
ex = late_extrapolation(edf, :y, :d, :z; covariates=[:x, :cell],
                        targets=[:compliers, :population, :treated])
show_plain(ex)
@printf("True ATE %.3f; true effect on the treated %.3f\n", mean(2 .+ xe),
        mean((2 .+ xe)[de .== 1]))
show_plain(instrument_validity_test(edf, :y, :d, :z; covariates=[:cell],
                                    n_bootstrap=499, rng=Xoshiro(3)))
