# Small simulated designs and fitted results used by the viz tests (and handy for
# rendering the plots by hand: `include("test/viz/fixtures.jl"); f = viz_fixtures()`).

using DrSnow, DataFrames, Random, StableRNGs, Statistics

"""Staggered-adoption panel with dynamic effects (cohorts 0 = never treated)."""
function viz_staggered(rng; N=150, T=8, cohorts=[0, 4, 6])
    g = rand(rng, cohorts, N)
    α = randn(rng, N)
    λ = cumsum(randn(rng, T)) .* 0.3
    rows = [(unit=i, time=2000 + t, g=g[i], d=Int(g[i] > 0 && t >= g[i]),
             y=α[i] + λ[t] + (g[i] > 0 && t >= g[i] ? 1.0 + 0.4 * (t - g[i]) : 0.0) +
               0.8 * randn(rng))
            for i in 1:N for t in 1:T]
    return DataFrame(rows)
end

"""Factor-model panel with one treated unit (last) from period `T0 + 1`."""
function viz_synth_panel(rng; N=16, T=18, T0=12, effect=-3.0)
    f = cumsum(randn(rng, T)) .+ range(0, 4; length=T)
    load = 0.5 .+ rand(rng, N)
    α = 10 .+ 2 .* randn(rng, N)
    rows = [(unit="u$(i)", time=1990 + t,
             d=Int(i == N && t > T0),
             y=α[i] + load[i] * f[t] + (i == N && t > T0 ? effect : 0.0) +
               0.3 * randn(rng))
            for i in 1:N for t in 1:T]
    return DataFrame(rows)
end

function viz_iv_data(rng; n=500, strength=0.25)
    z = rand(rng, n) .< 0.5
    u = randn(rng, n)
    d = strength .* z .+ 0.6 .* u .+ randn(rng, n)
    y = 1.0 .* d .+ u .+ randn(rng, n)
    return DataFrame(y=y, d=d, z=Float64.(z), x=randn(rng, n))
end

function viz_rd_data(rng; n=1500)
    x = 2 .* rand(rng, n) .- 1
    y = 0.4 .+ 0.8 .* x .- 0.5 .* x .^ 2 .+ 0.6 .* (x .>= 0) .+ 0.3 .* randn(rng, n)
    return DataFrame(y=y, x=x)
end

function viz_experiment(rng; n=60)
    d = shuffle(rng, vcat(ones(Int, n ÷ 2), zeros(Int, n - n ÷ 2)))
    x1 = randn(rng, n)
    x2 = randn(rng, n)
    y = 0.5 .* x1 .+ d .* (0.4 .+ 0.8 .* x1) .+ randn(rng, n)
    return DataFrame(y=y, d=d, x1=x1, x2=x2)
end

function viz_ring_panel(rng; N=500, T=7, adopt=4, share=0.12, L=100.0)
    xs = L .* rand(rng, N)
    ys = L .* rand(rng, N)
    s = SpatialStructure(["c$(i)" for i in 1:N]; x=xs, y=ys)
    tr = rand(rng, N) .< share
    panel = DataFrame(id=repeat(s.ids, T), t=repeat(1:T; inner=N))
    ui = repeat(1:N, T)
    panel.d = Int.(tr[ui] .& (panel.t .>= adopt))
    ex = compute_exposure(panel, :d, s, RingExposure([5.0, 10.0]); unit=:id, time=:t)
    panel.y = randn(rng, N)[ui] .+ randn(rng, T)[panel.t] .+ 2.0 .* panel.d .+
              (1 .- panel.d) .* (1.0 .* ex.ring_0_5 .+ 0.4 .* ex.ring_5_10) .+
              0.5 .* randn(rng, nrow(panel))
    return panel, s
end

"""Fit one result of every plotted type (a few seconds)."""
function viz_fixtures(; seed=20260927)
    rng = StableRNG(seed)
    stag = viz_staggered(rng)
    # (silence the expected warning about TWFE event studies under staggering)
    es_twfe = Base.CoreLogging.with_logger(Base.CoreLogging.NullLogger()) do
        event_study(stag, :y, :d, :unit, :time; estimator=:twfe)
    end
    es_sa = did_sun_abraham(stag, :y, :d, :unit, :time)
    cs = did_callaway_santanna(stag, :y, :d, :unit, :time; rng=StableRNG(1))
    es_cs = aggregate_att(cs, :dynamic)
    es_imp = did_imputation(stag, :y, :d, :unit, :time; horizons=:all,
                            pretrends=3)
    did = did_twfe(stag, :y, :d, :unit, :time; warn_heterogeneity=false)
    bacon = bacon_decomposition(stag, :y, :d, :unit, :time)

    sp = viz_synth_panel(rng)
    sc = synthetic_control(sp, :y, :d, :unit, :time; placebo=true, rng=StableRNG(2))
    sdid = synthetic_did(sp, :y, :d, :unit, :time; replications=50, rng=StableRNG(3))
    sdid_nose = synthetic_did(sp, :y, :d, :unit, :time; se_method=:none)
    ascm = augmented_synthetic_control(sp, :y, :d, :unit, :time; se_method=:none)
    mc = matrix_completion(sp, :y, :d, :unit, :time; se_method=:none)

    ivd = viz_iv_data(rng)
    iv = iv_regression(ivd, :y, :d, :z; covariates=[:x])
    ar = weak_iv_confidence_set(iv; method=:ar)
    ivweak = iv_regression(viz_iv_data(rng; strength=0.03), :y, :d, :z)
    ar_weak = weak_iv_confidence_set(ivweak; method=:ar)

    rdd = viz_rd_data(rng)
    rd = rd_estimate(rdd, :y, :x)
    rdp = rd_plot_data(rdd, :y, :x)

    ex = viz_experiment(rng)
    ri = randomization_test(ex, :y, :d; nperm=2000, rng=StableRNG(4))
    dml = dml_plr(ex, :y, :d; covariates=[:x1, :x2], outcome_learner=OLSLearner(),
                  treatment_learner=OLSLearner(), rng=StableRNG(5))
    big = viz_experiment(rng; n=600)
    gml = generic_ml(big, :y, :d; covariates=[:x1, :x2], proxy_learner=OLSLearner(),
                     propensity=0.5, n_splits=10, n_groups=4, rng=StableRNG(6),
                     parallel=false)
    cate = cate_dr_learner(big, :y, :d; covariates=[:x1, :x2],
                           outcome_learner=OLSLearner(), cate_learner=OLSLearner(),
                           rng=StableRNG(7))

    rp, s = viz_ring_panel(rng)
    rings = RingExposure([5.0, 10.0])
    ring_es = spillover_event_study(rp, :y, :d, s; unit=:id, time=:t, exposure=rings,
                                    leads=3, lags=2)
    ring_did = spillover_did(rp, :y, :d, s; unit=:id, time=:t, exposure=rings)

    return (; stag, es_twfe, es_sa, cs, es_cs, es_imp, did, bacon, sc, sdid, sdid_nose,
            ascm, mc, iv, ar, ivweak, ar_weak, rd, rdp, ri, dml, gml, cate, ring_es,
            ring_did)
end
