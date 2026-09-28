# Fitted results for the design-diagnostic plots (trends, balance, RD falsification,
# Honest DiD, IV design plots, synthetic-control backdating). Uses the helpers of
# fixtures.jl. Handy for rendering by hand:
# `include("test/viz/fixtures.jl"); include("test/viz/fixtures_diagnostics.jl");
#  d = viz_diag_fixtures()`.

"""Staggered panel with a covariate that shifts the outcome and differs by cohort."""
function viz_diag_panel(rng; N=240, T=10, cohorts=[0, 5, 7])
    df = viz_staggered(rng; N=N, T=T, cohorts=cohorts)
    df.x = 0.3 .* df.g ./ maximum(cohorts) .+ randn(rng, nrow(df))
    df.w = randn(rng, nrow(df))
    df.y .+= 0.8 .* df.x
    return df
end

"""RD data with predetermined covariates (one smooth, one mildly discontinuous)."""
function viz_rd_cov_data(rng; n=2000)
    x = 2 .* rand(rng, n) .- 1
    z1 = 1.0 .+ 0.5 .* x .+ randn(rng, n)
    z2 = 10 .+ 5 .* x .+ 3 .* randn(rng, n)
    z3 = 0.2 .* x .+ 0.25 .* (x .>= 0) .+ 0.5 .* randn(rng, n)
    y = 0.4 .+ 0.8 .* x .- 0.5 .* x .^ 2 .+ 0.6 .* (x .>= 0) .+ 0.3 .* randn(rng, n)
    return DataFrame(y=y, x=x, z1=z1, z2=z2, z3=z3)
end

"""Judge design: `J` judges with leniency U(0.2, 0.7), random assignment."""
function viz_judge_data(rng; J=40, cases=40)
    λ = 0.2 .+ 0.5 .* rand(rng, J)
    judge = repeat(1:J; inner=cases)
    n = length(judge)
    u = rand(rng, n)
    d = Float64.(u .< λ[judge])
    y = 1.0 .* d .+ randn(rng, n)
    return DataFrame(y=y, d=d, judge=judge)
end

"""Shift-share cross-section with K sectors."""
function viz_shift_share(rng; n=300, K=12)
    S = rand(rng, n, K) .^ 3
    S ./= sum(S; dims=2)
    g = randn(rng, K)
    df = DataFrame(d=0.8 .* (S * g) .+ randn(rng, n))
    df.y = 1.0 .* df.d .+ randn(rng, n)
    sh = [Symbol("s", k) for k in 1:K]
    for k in 1:K
        df[!, sh[k]] = S[:, k]
    end
    return df, sh, g
end

"""Binary-treatment data with a continuous instrument and heterogeneous effects."""
function viz_mte_data(rng; n=2000)
    z = randn(rng, n)
    x = randn(rng, n)
    v = randn(rng, n)
    d = Float64.(0.9 .* z .+ 0.3 .* x .- v .> 0)
    ud = DrSnow.cdf.(DrSnow.Normal(), v)
    y = 0.5 .* x .+ d .* (1.5 .- 2.0 .* ud) .+ randn(rng, n)
    return DataFrame(y=y, d=d, z=z, x=x)
end

function viz_diag_fixtures(; seed=20260928)
    rng = StableRNG(seed)
    panel = viz_diag_panel(rng)
    tm = treatment_timing(panel, :d, :unit, :time)
    pb = pretreatment_balance(panel, :d, :unit, :time; covariates=[:x, :w])
    es = Base.CoreLogging.with_logger(Base.CoreLogging.NullLogger()) do
        event_study(panel, :y, :d, :unit, :time; estimator=:twfe, max_pre=3,
                    max_post=2, endpoints=:trim)
    end
    hd_rm = honest_did(es; restriction=:relative_magnitudes, M=0:0.25:1.5,
                       rng=StableRNG(11))
    hd_sd = honest_did(es; restriction=:smoothness, M=[0.0, 0.05, 0.1, 0.2, 0.4])
    bd_rm = honest_breakdown(es; restriction=:relative_magnitudes, rng=StableRNG(12))

    ex = viz_experiment(rng; n=80)
    ex.x3 = randn(rng, nrow(ex))
    rib = ri_balance_test(ex, :d, [:x1, :x2, :x3]; nperm=500, rng=StableRNG(13),
                          threaded=false)

    rdd = viz_rd_cov_data(rng)
    rdcb = rd_covariate_balance(rdd, [:z1, :z2, :z3], :x)
    rd = rd_estimate(rdd, :y, :x)
    rdpl = rd_placebo_cutoffs(rdd, :y, :x; placebo_cutoffs=[-0.6, -0.4, -0.2, 0.2, 0.4,
                                                              0.6])
    rdbw = rd_bandwidth_sensitivity(rdd, :y, :x)
    rddo = rd_donut(rdd, :y, :x; radii=[0, 0.02, 0.05, 0.1])
    # running variable with bunching just above the cutoff
    xr = vcat(rdd.x, 0.08 .* rand(rng, 150))
    rdden = rd_density_test(xr)
    rdden_ok = rd_density_test(rdd, :x)

    jd = viz_judge_data(rng)
    jiv = Base.CoreLogging.with_logger(Base.CoreLogging.NullLogger()) do
        judge_iv(jd, :y, :d, :judge)
    end
    ssd, sh, g = viz_shift_share(rng)
    rw = rotemberg_weights(ssd, :y, :d, sh, g)

    md = viz_mte_data(rng)
    mte_poly = mte(md, :y, :d, [:z]; covariates=[:x], method=:polynomial, degree=2,
                   n_bootstrap=40, rng=StableRNG(14))
    mte_semi = mte(md, :y, :d, [:z]; covariates=[:x], n_bootstrap=30,
                   rng=StableRNG(15))

    sp = viz_synth_panel(rng)
    sc = synthetic_control(sp, :y, :d, :unit, :time)
    sc_back = synth_in_time_placebo(sc, 1998)

    return (; panel, tm, pb, es, hd_rm, hd_sd, bd_rm, ex, rib, rdd, rdcb, rd, rdpl,
            rdbw, rddo, xr, rdden, rdden_ok, jd, jiv, ssd, rw, md, mte_poly, mte_semi,
            sp, sc, sc_back)
end
