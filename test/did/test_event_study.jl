@testset "Event studies (TWFE) and pre-trend tests" begin
    dyn(g, e) = e >= 0 ? 2.0 + e : 0.0
    function sim_single(rng; N=200, T=8, g=5, share=0.5, sigma=1.0, effect=dyn)
        nt = round(Int, N * share)
        return sim_staggered(rng; N=N, T=T, effect=effect, sigma=sigma,
                             assign=vcat(fill(g, nt), zeros(Int, N - nt)))
    end

    @testset "fully dynamic TWFE: truth and FixedEffectModels reference" begin
        rng = StableRNG(201)
        df = sim_single(rng; N=600)
        es = event_study(df, :y, :d, :unit, :time)      # :auto → :twfe (one cohort)
        @test es isa EventStudyEstimate
        @test occursin("TWFE", es.method)
        @test relative_periods(es) == [-4, -3, -2, 0, 1, 2, 3]
        @test es.reference == [-1]
        truth = dyn.(5, relative_periods(es))
        @test all(abs.(coef(es) .- truth) .< 4 .* stderror(es))
        # reference: hand-built dummies in FixedEffectModels
        d2 = copy(df)
        rel = [g > 0 ? t - g : -1000 for (t, g) in zip(d2.time, d2.g)]
        cols = Symbol[]
        for k in relative_periods(es)
            c = Symbol("k", k < 0 ? "m$(-k)" : "$k")
            d2[!, c] = Float64.(rel .== k)
            push!(cols, c)
        end
        m = DrSnow.FixedEffectModels.reg(d2, make_formula(:y, cols; fe=[:unit, :time]),
                                         Vcov.cluster(:unit))
        @test coef(es) ≈ coef(m)
        @test vcov(es) ≈ vcov(m)
        @test dof_residual(es) == 599
        @test es.details.binned == (false, false)
    end

    @testset "endpoints: binning and trimming never pool into the reference" begin
        rng = StableRNG(202)
        df = sim_single(rng; N=2000, sigma=0.5)
        eb = event_study(df, :y, :d, :unit, :time; max_pre=2, max_post=1)
        @test relative_periods(eb) == [-2, 0, 1]
        @test eb.details.binned == (true, true)
        @test coefnames(eb) == ["e<=-2", "e=0", "e>=1"]
        # pre-period coefficient is unbiased (the v0.1 bug returned about -4.5 here)
        @test abs(coef(eb)[1]) < 4 * stderror(eb)[1]
        @test abs(coef(eb)[2] - 2.0) < 4 * stderror(eb)[2]
        et = event_study(df, :y, :d, :unit, :time; max_pre=2, max_post=1, endpoints=:trim)
        @test et.details.binned == (false, false)
        @test abs(coef(et)[1]) < 4 * stderror(et)[1]
        @test abs(coef(et)[3] - 3.0) < 4 * stderror(et)[3]
        @test nobs(et) < nobs(eb)
    end

    @testset "validation errors" begin
        rng = StableRNG(203)
        df = sim_single(rng; N=60)
        @test_throws ArgumentError event_study(df, :y, :d, :unit, :time; omit_period=-9)
        @test_throws ArgumentError event_study(df, :y, :d, :unit, :time; max_pre=9)
        @test_throws ArgumentError event_study(df, :y, :d, :unit, :time; max_post=9)
        @test_throws ArgumentError event_study(df, :y, :d, :unit, :time; max_pre=1)
        @test_throws ArgumentError event_study(df, :y, :d, :unit, :time; endpoints=:x)
        @test_throws ArgumentError event_study(df, :y, :d, :unit, :time; estimator=:x)
        # all units treated at once: dynamic coefficients not identified → error, not 0/NaN
        all_t = sim_staggered(rng; N=40, T=6, assign=fill(4, 40))
        @test_throws ArgumentError event_study(all_t, :y, :d, :unit, :time)
        # non-absorbing treatment
        sw = copy(df)
        sw.d[findfirst(sw.d .== 1)] = 0
        sw.d[findfirst(sw.d .== 1) + 1] = 0
        sw.d[findlast(sw.d .== 1)] = 0
        @test_throws ArgumentError event_study(sw, :y, :d, :unit, :time; estimator=:twfe)
    end

    @testset "gaps and Date time variables use the period index" begin
        rng = StableRNG(204)
        df = sim_single(rng; N=100)
        dg = copy(df)
        dg.time = 2 .* dg.time               # biennial data
        e1 = event_study(df, :y, :d, :unit, :time)
        e2 = event_study(dg, :y, :d, :unit, :time)
        @test relative_periods(e1) == relative_periods(e2)
        @test coef(e1) ≈ coef(e2)
        dd = copy(df)
        dd.time = Date.(dd.time)
        @test coef(event_study(dd, :y, :d, :unit, :time)) ≈ coef(e1)
    end

    @testset "estimator dispatch" begin
        rng = StableRNG(205)
        st = sim_staggered(rng; N=200, T=7, cohorts=[0, 3, 5])
        @test occursin("Sun–Abraham", event_study(st, :y, :d, :unit, :time).method)
        @test occursin("imputation",
                       event_study(st, :y, :d, :unit, :time; estimator=:imputation,
                                   max_pre=2).method)
        ecs = event_study(st, :y, :d, :unit, :time; estimator=:callaway_santanna,
                          rng=StableRNG(1))
        @test occursin("Callaway", ecs.method)
        @test_logs (:warn, r"cohorts") event_study(st, :y, :d, :unit, :time;
                                                   estimator=:twfe)
        @test_throws ArgumentError event_study(st, :y, :d, :unit, :time;
                                               estimator=:twfe, rng=StableRNG(1))
        p = TreatmentPanel(st, :y, :d, :unit, :time)
        @test coef(event_study(p)) ≈ coef(event_study(st, :y, :d, :unit, :time))
    end

    @testset "pre_trend_test: full covariance Wald, DiagnosticTest" begin
        rng = StableRNG(206)
        df = sim_single(rng; N=300)
        es = event_study(df, :y, :d, :unit, :time)
        t = pre_trend_test(es)
        @test t isa DiagnosticTest
        pre = findall(<(0), relative_periods(es))
        b = coef(es)[pre]; V = vcov(es)[pre, pre]
        @test t.statistic ≈ dot(b, V \ b) / length(pre)
        @test t.dof == (length(pre), 299)
        @test occursin("Roth", t.note)
        s = sprint(show, MIME"text/plain"(), t)
        @test !occursin("satisfied", s)
        t2 = pre_trend_test(es; periods=[-2])
        @test t2.dof[1] == 1
        @test_throws ArgumentError pre_trend_test(es; periods=[-7])
        pt = parallel_trends_test(df, :y, :d, :unit, :time)
        @test pt.statistic ≈ t.statistic
        @test pt.details.event_study isa EventStudyEstimate
        post_only = event_study(df, :y, :d, :unit, :time; max_pre=1, endpoints=:trim)
        @test_throws ArgumentError pre_trend_test(post_only)
    end

    @testset "averages and uniform bands" begin
        rng = StableRNG(207)
        df = sim_single(rng; N=300)
        es = event_study(df, :y, :d, :unit, :time)
        avg = event_study_average(es)
        post = findall(>=(0), relative_periods(es))
        w = fill(1 / length(post), length(post))
        @test coef(avg)[1] ≈ mean(coef(es)[post])
        @test stderror(avg)[1] ≈ sqrt(w' * vcov(es)[post, post] * w)
        @test estimate(es) ≈ coef(avg)[1]
        @test coef(event_study_average(es; periods=[0, 1], weights=[3, 1]))[1] ≈
              0.75 * coef(es)[post[1]] + 0.25 * coef(es)[post[2]]
        @test_throws ArgumentError event_study_average(es; periods=[42])
        cu = confint(es; uniform=true, rng=StableRNG(3))
        cp = confint(es)
        @test all(cu[:, 2] .- cu[:, 1] .> cp[:, 2] .- cp[:, 1])
        @test cu == confint(es; uniform=true, rng=StableRNG(3))
    end

    @testset "row-shuffling invariance" begin
        rng = StableRNG(208)
        df = sim_single(rng; N=80)
        e1 = event_study(df, :y, :d, :unit, :time; max_pre=2, max_post=2)
        e2 = event_study(shuffle_rows(rng, df), :y, :d, :unit, :time; max_pre=2, max_post=2)
        @test coef(e1) ≈ coef(e2)
        @test vcov(e1) ≈ vcov(e2)
    end

    @testset "Monte Carlo: pre-trend test size with dynamic effects" begin
        # Parallel trends holds; effects are large and growing. The v0.1 test rejected
        # in 100% of samples here.
        R = mc_reps(1000, 200)
        rng = StableRNG(209)
        rej_full = 0; rej_bin = 0; cover_u = 0
        for _ in 1:R
            df = sim_single(rng; N=60, T=8, sigma=1.0)
            es = event_study(df, :y, :d, :unit, :time)
            rej_full += pre_trend_test(es).pvalue < 0.05
            eb = event_study(df, :y, :d, :unit, :time; max_pre=2, max_post=2)
            rej_bin += pre_trend_test(eb).pvalue < 0.05
            cu = confint(es; uniform=true, rng=rng, ndraws=2000)
            truth = dyn.(5, relative_periods(es))
            cover_u += all(cu[:, 1] .<= truth .<= cu[:, 2])
        end
        @test mc_close(rej_full / R, 0.05, R; slack=0.015)
        @test mc_close(rej_bin / R, 0.05, R; slack=0.015)
        @test mc_close(cover_u / R, 0.95, R; slack=0.02)
    end
end
