# Tables and figures for DrSnow results: tidy data frames, RegressionTables and the
# Makie plotting extension, on real data where the repository ships it.
#
# Requires CairoMakie (and optionally RegressionTables) in the active environment:
#     julia --project=. -e 'using Pkg; Pkg.add(["CairoMakie", "RegressionTables"])'
# or run from any environment where DrSnow, CairoMakie and DataFrames are available.
# Then, from the repository root:
#     julia --project=. examples/plotting_demo.jl [output directory]
#
# Figures are written as PNG files to `plots/` (or the directory given).

if Base.find_package("CairoMakie") === nothing
    error("this demo needs CairoMakie: `using Pkg; Pkg.add(\"CairoMakie\")`")
end

using DrSnow
using CairoMakie
using DataFrames
using Random
using Statistics

outdir = isempty(ARGS) ? joinpath(pwd(), "plots") : ARGS[1]
mkpath(outdir)
savefig(name, fig) = (save(joinpath(outdir, name * ".png"), fig; px_per_unit=2);
                      println("  wrote ", joinpath(outdir, name * ".png")))
valdir = joinpath(@__DIR__, "..", "test", "validation")

# Minimal CSV reader for the numeric/quoted files shipped with the tests.
function read_csv(path)
    lines = readlines(path)
    header = Symbol.(strip.(split(lines[1], ','), '"'))
    rows = [strip.(split(l, ','), '"') for l in lines[2:end]]
    df = DataFrame()
    for (j, h) in enumerate(header)
        raw = [String(r[j]) for r in rows]
        num = tryparse.(Float64, raw)
        df[!, h] = any(isnothing, num) ? raw : Float64.(num)
    end
    return df
end

# ---------------------------------------------------------------------------------
# 1. Staggered DiD: minimum wage and teen employment (mpdta, Callaway & Sant'Anna)
# ---------------------------------------------------------------------------------
println("Staggered DiD (mpdta)")
mp = read_csv(joinpath(valdir, "did", "mpdta.csv"))
mp.first_treat = Int.(mp.first_treat)
mp.year = Int.(mp.year)
mp.countyreal = Int.(mp.countyreal)
mp.d = Int.((mp.first_treat .> 0) .& (mp.year .>= mp.first_treat))
treat = FirstTreated(:first_treat)

es_twfe = event_study(mp, :lemp, treat, :countyreal, :year; estimator=:twfe)
es_sa = did_sun_abraham(mp, :lemp, treat, :countyreal, :year)
es_cs = aggregate_att(did_callaway_santanna(mp, :lemp, treat, :countyreal, :year;
                                            rng=Xoshiro(1)), :dynamic)
es_bjs = did_imputation(mp, :lemp, treat, :countyreal, :year; horizons=:all,
                        pretrends=3)
savefig("event_study_overlay",
        plot_event_study(es_twfe, es_sa, es_cs, es_bjs;
                         labels=["TWFE", "Sun–Abraham", "Callaway–Sant'Anna",
                                 "Imputation (BJS)"], uniform=true, rng=Xoshiro(2),
                         axis=(title="Minimum wage and log teen employment",
                               ylabel="Effect on log employment")))
savefig("bacon", plot_bacon(bacon_decomposition(mp, :lemp, :d, :countyreal, :year)))

# Tidy output and a regression table of the headline estimates
twfe = did_twfe(mp, :lemp, :d, :countyreal, :year; warn_heterogeneity=false)
att_cs = aggregate_att(did_callaway_santanna(mp, :lemp, treat, :countyreal, :year;
                                             rng=Xoshiro(1)), :simple)
att_bjs = did_imputation(mp, :lemp, treat, :countyreal, :year)
println(tidy([twfe, att_cs, att_bjs]; names=["TWFE", "Callaway–Sant'Anna", "BJS"]))
println(vcat(glance.([twfe, att_cs, att_bjs])...))
savefig("did_estimates",
        plot_coefficients([twfe, att_cs, att_bjs];
                          labels=["TWFE", "Callaway–Sant'Anna", "Imputation (BJS)"],
                          axis=(title="Average effect on log teen employment",)))
if Base.find_package("RegressionTables") !== nothing
    @eval using RegressionTables
    println(regtable(twfe, att_cs, att_bjs))
end

# ---------------------------------------------------------------------------------
# 2. Synthetic control: California's Proposition 99
# ---------------------------------------------------------------------------------
println("Synthetic control (Proposition 99)")
p99 = read_csv(joinpath(valdir, "synth", "prop99.csv"))
p99.Year = Int.(p99.Year)
p99.treated = Int.(p99.treated)
sc = synthetic_control(p99, :PacksPerCapita, :treated, :State, :Year; placebo=true,
                       rng=Xoshiro(3))
savefig("synth_control", plot_synth(sc; placebo_cutoff=sqrt(5)))
savefig("synth_placebo_test", plot_randomization_distribution(synth_in_space_placebo(sc)))
sdid = synthetic_did(p99, :PacksPerCapita, :treated, :State, :Year; rng=Xoshiro(4))
savefig("synth_did", plot_synth(sdid; kind=:trajectories))

# ---------------------------------------------------------------------------------
# 3. Instrumental variables: returns to schooling (Card 1995), weak-IV-robust set
# ---------------------------------------------------------------------------------
println("IV (Card 1995)")
card = read_csv(joinpath(valdir, "iv", "card.csv"))
iv = iv_regression(card, :lwage, :educ, :nearc4;
                   covariates=[:exper, :expersq, :black, :smsa, :south])
savefig("iv_ar_set", plot_confidence_set(weak_iv_confidence_set(iv; method=:ar);
                                         wald=iv))

# ---------------------------------------------------------------------------------
# 4. Simulated designs: RD, randomization inference, heterogeneity
# ---------------------------------------------------------------------------------
println("Simulated RD, experiment and heterogeneity")
rng = Xoshiro(5)
x = 2 .* rand(rng, 2000) .- 1
rdd = DataFrame(x=x, y=0.5 .+ 0.8 .* x .- 0.4 .* x .^ 2 .+ 0.5 .* (x .>= 0) .+
                     0.3 .* randn(rng, 2000))
savefig("rd", plot_rd(rd_plot_data(rdd, :y, :x); estimate=rd_estimate(rdd, :y, :x),
                      ci=true))

n = 800
expt = DataFrame(x1=randn(rng, n), x2=randn(rng, n),
                 d=shuffle(rng, repeat([0, 1], n ÷ 2)))
expt.y = 0.5 .* expt.x1 .+ expt.d .* (0.3 .+ 0.6 .* expt.x1) .+ randn(rng, n)
savefig("randomization", plot_randomization_distribution(
    randomization_test(expt, :y, :d; nperm=5000, rng=Xoshiro(6))))
gml = generic_ml(expt, :y, :d; covariates=[:x1, :x2], propensity=0.5, n_splits=50,
                 n_groups=5, rng=Xoshiro(7))
savefig("gates", plot_gates(gml))
cate = cate_dr_learner(expt, :y, :d; covariates=[:x1, :x2], rng=Xoshiro(8))
savefig("cate_by_x1", plot_cate(cate; modifier=:x1))

# A combined multi-panel figure with the mutating forms
fig = Figure(size=(1100, 420))
plot_event_study!(Axis(fig[1, 1]; title="Callaway–Sant'Anna",
                       xlabel="Years since treatment"), es_cs)
plot_rd!(Axis(fig[1, 2]; title="Regression discontinuity", xlabel="Running variable"),
         rd_plot_data(rdd, :y, :x))
plot_gates!(Axis(fig[1, 3]; title="GATES"), gml)
savefig("panel", fig)

# ---------------------------------------------------------------------------------
# 5. Design diagnostics: trends, balance, falsification and sensitivity
# ---------------------------------------------------------------------------------
println("Design diagnostics")
# DiD (mpdta): raw trends by cohort, pre-treatment balance, Honest DiD
savefig("trends", plot_trends(mp, :lemp, treat, :countyreal, :year;
                              axis=(title="Log teen employment by adoption cohort",)))
mp.lpop_t = mp.lpop .+ 0.0            # time-invariant covariate, as a Float64 column
savefig("balance_did", plot_balance(pretreatment_balance(mp, treat, :countyreal, :year;
                                                         covariates=[:lpop_t])))
hd = honest_did(es_sa; restriction=:relative_magnitudes, M=0:0.25:1.5,
                rng=Xoshiro(9))
savefig("honest_did", plot_honest_did(hd; breakdown=honest_breakdown(
    es_sa; restriction=:relative_magnitudes, rng=Xoshiro(10))))

# Experiment: randomization balance test
expt.x3 = randn(rng, n)
savefig("balance_ri", plot_balance(ri_balance_test(expt, :d, [:x1, :x2, :x3];
                                                   nperm=2000, rng=Xoshiro(11))))

# RD: covariate balance, placebo cutoffs, bandwidth sensitivity, density test
rdd.z = 0.3 .* rdd.x .+ randn(rng, nrow(rdd))
rdd.w = 10 .+ 4 .* rdd.x .+ 3 .* randn(rng, nrow(rdd))
savefig("balance_rd", plot_balance(rd_covariate_balance(rdd, [:z, :w], :x); data=rdd))
savefig("rd_placebos", plot_rd_placebos(rd_placebo_cutoffs(rdd, :y, :x);
                                        estimate=rd_estimate(rdd, :y, :x)))
savefig("rd_bandwidths", plot_rd_sensitivity(rd_bandwidth_sensitivity(rdd, :y, :x)))
savefig("rd_density", plot_rd_density(rdd, :x, rd_density_test(rdd, :x)))

# Synthetic control (Proposition 99): backdating to 1980
savefig("synth_in_time", plot_synth_in_time(sc, synth_in_time_placebo(sc, 1980)))

# IV designs: judge first stage (simulated judge data shipped with the tests),
# Rotemberg weights and an MTE curve on simulated data
jd = read_csv(joinpath(valdir, "iv", "judge.csv"))
jd.judge = Int.(jd.judge)
jd.court = Int.(jd.court)
savefig("judge_first_stage",
        plot_judge_first_stage(jd, judge_iv(jd, :y, :d, :judge; strata=[:court])))
K = 15
S = rand(rng, 400, K) .^ 3
S ./= sum(S; dims=2)
shocks = randn(rng, K)
ss = DataFrame(S, [Symbol("s", k) for k in 1:K])
ss.d = 0.8 .* (S * shocks) .+ randn(rng, 400)
ss.y = ss.d .+ randn(rng, 400)
savefig("rotemberg", plot_rotemberg(rotemberg_weights(ss, :y, :d,
                                                      [Symbol("s", k) for k in 1:K],
                                                      shocks)))
m = 3000
md = DataFrame(z=randn(rng, m), x=randn(rng, m), v=randn(rng, m))
md.d = Float64.(0.9 .* md.z .+ 0.3 .* md.x .> md.v)
ud = [count(<(v), md.v) / m for v in md.v]            # rank of the unobservable
md.y = 0.5 .* md.x .+ md.d .* (1.5 .- 2 .* ud) .+ randn(rng, m)
savefig("mte", plot_mte(mte(md, :y, :d, [:z]; covariates=[:x], method=:polynomial,
                            n_bootstrap=100, rng=Xoshiro(12))))

# Publication settings: serif theme, single-column size, vector output
with_theme(drsnow_theme(; font=:serif, fontsize=10)) do
    fig = plot_event_study(es_cs; figure=(size=(500, 320),))
    save(joinpath(outdir, "event_study_serif.pdf"), fig)
    savefig("event_study_serif", fig)
end
println("Done.")
