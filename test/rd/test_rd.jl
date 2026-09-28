# Unit tests for the RD area: input validation, invariances, internal consistency.

quiet(f) = Base.CoreLogging.with_logger(f, Base.CoreLogging.NullLogger())

function rd_sim_sharp(rng, n; tau=0.5)
    x = 2 .* rand(rng, n) .- 1
    y = 1 .+ x .- 0.5 .* x .^ 2 .+ tau .* (x .>= 0) .+ 0.3 .* randn(rng, n)
    return DataFrame(y=y, x=x)
end

@testset "input validation" begin
    df = RD_SENATE
    @test_throws ArgumentError rd_estimate(df, :nope, :margin)
    @test_throws ArgumentError rd_estimate(df, :vote, :margin; cutoff=200)
    @test_throws ArgumentError rd_estimate(df, :vote, :margin; p=2, q=2)
    @test_throws ArgumentError rd_estimate(df, :vote, :margin; deriv=2, p=1)
    @test_throws ArgumentError rd_estimate(df, :vote, :margin; kernel=:gaussian)
    @test_throws ArgumentError rd_estimate(df, :vote, :margin; vce=:hc9)
    @test_throws ArgumentError rd_estimate(df, :vote, :margin; vce=:cr2)
    @test_throws ArgumentError rd_estimate(df, :vote, :margin; vce=:cr1)
    # HC options with a cluster variable switch to the CR analogue (with a warning)
    r_hc2 = @test_logs (:warn,) rd_estimate(df, :vote, :margin; vce=:hc2, cluster=:state)
    @test r_hc2.vce === :cr2
    @test rd_estimate(df, :vote, :margin; cluster=:state).vce === :cr1
    @test_throws ArgumentError rd_estimate(df, :vote, :margin; bwselect=:ik)
    @test_throws ArgumentError rd_estimate(df, :vote, :margin; level=95)
    @test_throws ArgumentError rd_estimate(df, :vote, :margin; h=-1)
    @test_throws ArgumentError rd_estimate(df, :vote, :margin; h=(1, 2, 3))
    # local fit not identified: too few distinct running values within the bandwidth
    @test_throws ArgumentError rd_estimate(df, :vote, :margin; h=0.05)
    @test_throws ArgumentError rd_estimate(df, :vote, :margin; cluster=[:state, :year])
    @test_throws ArgumentError rd_estimate(df, :state, :margin)
    bad = DataFrame(y=randn(StableRNG(1), 100), x=randn(StableRNG(2), 100),
                    w=-ones(100), d=zeros(100))
    @test_throws ArgumentError rd_estimate(bad, :y, :x; weights=:w)
    @test_throws ArgumentError rd_estimate(bad, :y, :x; treatment=:d)
    @test_throws ArgumentError rd_bandwidth(bad[1:10, :], :y, :x)
    r = rd_estimate(df, :vote, :margin)
    @test_throws ArgumentError rd_weak_iv_confidence_set(r)
    @test_throws ArgumentError rd_density_test(df, :margin; cutoff=150)
    @test_throws ArgumentError rd_density_test(df, :margin; p=9)
    @test_throws ArgumentError rd_density_test(df, :margin; fitselect=:restricted,
                                               bwselect=:each)
    @test_throws ArgumentError rd_density_test(df, :margin; vce=:bootstrap)
    @test_throws ArgumentError rd_plot_data(df, :vote, :margin; binselect=:foo)
    @test_throws ArgumentError rd_covariate_balance(df, Symbol[], :margin)
    @test_throws ArgumentError rd_donut(df, :vote, :margin; radii=[-1])
    @test_throws ArgumentError rd_randomization_test(df, :vote, :margin; window=0.01)
    @test_throws ArgumentError rd_randomization_test(df, :vote, :margin; window=1,
                                                     statistic=:median)
end

@testset "row order and missing values do not matter" begin
    rng = StableRNG(11)
    for kw in ((;), (; vce=:hc2), (; cluster=:state), (; covariates=[:presdemvoteshlag1]),
               (; kernel=:uniform, p=2))
        a = rd_estimate(RD_SENATE, :vote, :margin; kw...)
        shuffled = RD_SENATE[randperm(rng, nrow(RD_SENATE)), :]
        b = rd_estimate(shuffled, :vote, :margin; kw...)
        @test a.tau_bias_corrected ≈ b.tau_bias_corrected rtol = 1e-10
        @test a.se_robust ≈ b.se_robust rtol = 1e-10
        @test a.h_left ≈ b.h_left rtol = 1e-10
    end
    # rows with missing outcome are dropped (Senate has 93 missing votes)
    complete = dropmissing(RD_SENATE, [:vote, :margin])
    a = rd_estimate(RD_SENATE, :vote, :margin)
    b = rd_estimate(complete, :vote, :margin)
    @test a.tau_bias_corrected == b.tau_bias_corrected
    @test nobs(a) == nrow(complete)
    d1 = rd_density_test(RD_SENATE, :margin)
    d2 = rd_density_test(RD_SENATE[randperm(rng, nrow(RD_SENATE)), :], :margin)
    @test d1.statistic ≈ d2.statistic rtol = 1e-12
    p1 = rd_plot_data(RD_SENATE, :vote, :margin)
    p2 = rd_plot_data(RD_SENATE[randperm(rng, nrow(RD_SENATE)), :], :vote, :margin)
    @test p1.bins.mean_y ≈ p2.bins.mean_y
end

@testset "result interface" begin
    r = rd_estimate(RD_SENATE, :vote, :margin)
    @test r isa CausalEstimate
    # rdrobust convention: conventional point estimate, robust bias-corrected inference
    @test coef(r) == [r.tau_conventional]
    @test tstats(r) ≈ [r.tau_bias_corrected / r.se_robust]
    @test pvalues(r)[1] ≈ two_sided_pvalue(r.tau_bias_corrected / r.se_robust)
    @test sum(confint(r)) / 2 ≈ r.tau_bias_corrected
    ct = coeftable(r)
    @test ct.cols[1][1] == r.tau_conventional && ct.cols[2][1] ≈ r.se_robust
    @test stderror(r) ≈ [r.se_robust]
    @test vcov(r) ≈ fill(r.se_robust^2, 1, 1)
    @test length(coefnames(r)) == 1
    @test dof_residual(r) == Inf
    @test occursin("ATE at the cutoff", estimand(r))
    ci90 = confint(r; level=0.90)
    tab = rd_inference_table(r; level=0.90)
    @test tab.ci_lower[3] ≈ ci90[1, 1]
    @test tab.method == ["Conventional", "Bias-corrected", "Robust"]
    @test tab.se[1] == tab.se[2] == r.se_conventional
    s = sprint(show, MIME"text/plain"(), r)
    @test occursin("Robust", s) && occursin("bandwidths", s)
    @test occursin("RD", sprint(show, r))
    fz = rd_estimate(RD_SIM, :y_fuzzy, :x; treatment=:d)
    @test occursin("LATE", estimand(fz))
    @test occursin("First stage", sprint(show, MIME"text/plain"(), fz))
    @test occursin("kink", estimand(rd_estimate(RD_SIM, :y_kink, :x; deriv=1)))
    @test occursin("kink", estimand(rd_estimate(RD_SIM, :y_fkink, :x;
                                                treatment=:d_kink, deriv=1)))
    bw = rd_bandwidth(RD_SENATE, :vote, :margin)
    @test bw.h_left ≈ r.h_left
    @test occursin("mserd", sprint(show, MIME"text/plain"(), bw))
    @test occursin("RDBandwidth", sprint(show, bw))
    pd = rd_plot_data(RD_SENATE, :vote, :margin)
    @test sum(pd.bins.n) == nrow(dropmissing(RD_SENATE, [:vote, :margin]))
    @test nrow(pd.poly) == 1000
    @test occursin("RD plot data", sprint(show, MIME"text/plain"(), pd))
end

@testset "estimation identities" begin
    # With h = b the bias-corrected estimate equals the order-q local polynomial fit.
    r = rd_estimate(RD_SENATE, :vote, :margin; h=15)
    r2 = rd_estimate(RD_SENATE, :vote, :margin; h=15, p=2, q=3)
    @test r.tau_bias_corrected ≈ r2.tau_conventional rtol = 1e-8
    # scalepar scales the estimand and standard errors
    r3 = rd_estimate(RD_SENATE, :vote, :margin; h=15, scalepar=-2)
    @test r3.tau_bias_corrected ≈ -2 * r.tau_bias_corrected
    @test r3.se_robust ≈ 2 * r.se_robust
    # rho sets b = h / rho
    r4 = rd_estimate(RD_SENATE, :vote, :margin; h=10, rho=0.5)
    @test r4.b_left == 20.0
    # fuzzy with a deterministic treatment equals the sharp design
    df = copy(RD_SIM)
    df.dsharp = Float64.(df.x .>= 0)
    s = rd_estimate(df, :y_sharp, :x; h=0.4)
    f = quiet(() -> rd_estimate(df, :y_sharp, :x; h=0.4, treatment=:dsharp))
    @test f.tau_conventional ≈ s.tau_conventional rtol = 1e-10
    @test f.first_stage.tau_conventional ≈ 1.0
    # large-sample recovery of the true jump
    big = rd_sim_sharp(StableRNG(5), 20_000; tau=0.5)
    rb = rd_estimate(big, :y, :x)
    @test abs(rb.tau_bias_corrected - 0.5) < 4 * rb.se_robust
    @test rb.se_robust < 0.05
end

@testset "weak-IV robust confidence set" begin
    r = rd_estimate(RD_SIM, :y_fuzzy, :x; treatment=:d)
    cs = rd_weak_iv_confidence_set(r)
    @test cs.kind === :interval
    lo, hi = only(cs.intervals)
    @test lo < r.tau_bias_corrected < hi
    # The Anderson–Rubin statistic at τ0 equals the robust t statistic of a sharp RD on
    # Y − τ0 D with the same bandwidths.
    for τ0 in (0.3, lo, hi)
        df = copy(RD_SIM)
        df.ytil = df.y_fuzzy .- τ0 .* df.d
        s = rd_estimate(df, :ytil, :x; h=(r.h_left, r.h_right), b=(r.b_left, r.b_right))
        a = r.reduced_form.tau_bias_corrected
        b = r.first_stage.tau_bias_corrected
        V = r.vcov_robust_yt
        t_ar = (a - τ0 * b) / sqrt(V[1, 1] - 2τ0 * V[1, 2] + τ0^2 * V[2, 2])
        @test t_ar ≈ s.tau_bias_corrected / s.se_robust rtol = 1e-8
        τ0 == 0.3 || @test abs(t_ar) ≈ critical_value(0.95) rtol = 1e-8
    end
    @test cs.first_stage_z ≈ r.first_stage.tau_bias_corrected / r.first_stage.se_robust
    # With covariates and clustering the same identity holds.
    rc = rd_estimate(RD_SIM, :y_fuzzy, :x; treatment=:d, covariates=[:z1], cluster=:g)
    csc = rd_weak_iv_confidence_set(rc; level=0.9)
    @test csc.level == 0.9
    @test !isempty(csc.intervals)
    for v in (:cr1, :cr3)
        rv = rd_estimate(RD_SIM, :y_fuzzy, :x; treatment=:d, cluster=:g, vce=v)
        τ0 = 0.4
        df = copy(RD_SIM)
        df.ytil = df.y_fuzzy .- τ0 .* df.d
        s = rd_estimate(df, :ytil, :x; h=(rv.h_left, rv.h_right),
                        b=(rv.b_left, rv.b_right), cluster=:g, vce=v)
        a = rv.reduced_form.tau_bias_corrected
        b = rv.first_stage.tau_bias_corrected
        V = rv.vcov_robust_yt
        t_ar = (a - τ0 * b) / sqrt(V[1, 1] - 2τ0 * V[1, 2] + τ0^2 * V[2, 2])
        @test t_ar ≈ s.tau_bias_corrected / s.se_robust rtol = 1e-8
    end
    # A first stage indistinguishable from zero gives an unbounded set.
    rng = StableRNG(3)
    n = 800
    x = 2 .* rand(rng, n) .- 1
    d = Float64.(rand(rng, n) .< 0.5)
    y = x .+ d .+ randn(rng, n)
    rw = rd_estimate(DataFrame(y=y, x=x, d=d), :y, :x; treatment=:d)
    csw = rd_weak_iv_confidence_set(rw)
    if abs(csw.first_stage_z) < critical_value(0.95)
        @test csw.kind in (:two_rays, :real_line)
        @test any(iv -> isinf(iv[1]) || isinf(iv[2]), csw.intervals)
    end
end

@testset "density test" begin
    t = rd_density_test(RD_SENATE, :margin)
    @test t isa DiagnosticTest
    @test t.pvalue ≈ t.details.p_jackknife
    @test t.statistic == t.details.t_jackknife
    s = sprint(show, MIME"text/plain"(), t)
    @test occursin("not evidence", s)
    @test !occursin("no manipulation", lowercase(s))
    tv = rd_density_test(collect(skipmissing(RD_SENATE.margin)))
    @test tv.statistic == t.statistic
    tn = rd_density_test(RD_SENATE, :margin; binomial=false)
    @test tn.details.binomial === nothing
    @test rd_density_bandwidth(RD_SENATE, :margin).side == ["left", "right", "diff", "sum"]
    # a clear density discontinuity is detected
    rng = StableRNG(4)
    x = vcat(-rand(rng, 1000), rand(rng, 3000))
    @test rd_density_test(x).pvalue < 1e-4
end

@testset "falsification helpers" begin
    cb = rd_covariate_balance(RD_SENATE, [:presdemvoteshlag1, :demvoteshlag1, :dopen],
                              :margin)
    @test nrow(cb) == 3
    @test all(cb.pvalue_holm .>= cb.pvalue)
    @test haskey(DataFrames.metadata(cb), "note")
    r1 = rd_estimate(RD_SENATE, :presdemvoteshlag1, :margin)
    @test cb.estimate[1] ≈ r1.tau_bias_corrected
    @test_throws ArgumentError rd_covariate_balance(RD_SENATE, [:dopen], :margin;
                                                    covariates=[:year])
    pc = rd_placebo_cutoffs(RD_SENATE, :vote, :margin; placebo_cutoffs=[-10, 10, 100])
    @test pc.side == [:control, :treated, :treated]
    sub = RD_SENATE[coalesce.(RD_SENATE.margin .< 0, false), :]
    @test pc.estimate[1] ≈ rd_estimate(sub, :vote, :margin; cutoff=-10).tau_bias_corrected
    @test ismissing(pc.estimate[3]) && !isempty(pc.note[3])   # too close to the boundary
    @test_throws ArgumentError rd_placebo_cutoffs(RD_SENATE, :vote, :margin;
                                                  placebo_cutoffs=[0])
    @test nrow(rd_placebo_cutoffs(RD_SENATE, :vote, :margin)) == 6
    dn = rd_donut(RD_SENATE, :vote, :margin; radii=[0, 1])
    base = rd_estimate(RD_SENATE, :vote, :margin)
    @test dn.estimate[1] ≈ base.tau_bias_corrected
    @test dn.n_excluded_left[1] == dn.n_excluded_right[1] == 0
    @test dn.n_excluded_left[2] + dn.n_excluded_right[2] ==
          count(v -> !ismissing(v) && abs(v) < 1, RD_SENATE.margin)
    bs = rd_bandwidth_sensitivity(RD_SENATE, :vote, :margin; multipliers=[0.5, 1])
    @test bs.estimate[2] ≈ base.tau_bias_corrected
    @test bs.h_left[1] ≈ 0.5 * base.h_left
    be = rd_bandwidth_sensitivity(RD_SENATE, :vote, :margin; bandwidths=[10, 20])
    @test be.b_left == [10.0, 20.0]
    @test all(ismissing, be.multiplier)
    @test_throws ArgumentError rd_bandwidth_sensitivity(RD_SENATE, :vote, :margin; h=3)
end

@testset "local randomization" begin
    t1 = rd_randomization_test(RD_SENATE, :vote, :margin; window=2.5, rng=StableRNG(1))
    t2 = rd_randomization_test(RD_SENATE, :vote, :margin; window=2.5, rng=StableRNG(1))
    @test t1.pvalue == t2.pvalue
    @test 0 < t1.pvalue <= 1
    w = coalesce.(abs.(RD_SENATE.margin) .<= 2.5, false) .& .!ismissing.(RD_SENATE.vote)
    sub = RD_SENATE[w, :]
    dm = mean(sub.vote[sub.margin .>= 0]) - mean(sub.vote[sub.margin .< 0])
    @test t1.statistic ≈ dm
    @test t1.details.n_left + t1.details.n_right == nrow(sub)
    for st in (:ks, :ranksum)
        t = rd_randomization_test(RD_SENATE, :vote, :margin; window=2.5, statistic=st,
                                  reps=199, rng=StableRNG(2))
        @test 0 < t.pvalue <= 1
    end
    sel = rd_window_selection(RD_SENATE, [:presdemvoteshlag1], :margin; nwindows=4,
                              reps=199, rng=StableRNG(3))
    @test nrow(sel.table) == 4
    @test issorted(sel.table.window)
    @test sel.window === nothing || sel.window in sel.table.window
end
