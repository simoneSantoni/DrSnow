@testset "Callaway–Sant'Anna" begin
    eff(g, e) = e >= 0 ? 1.0 + 0.5 * e + 0.3 * (g - 3) : 0.0   # heterogeneous & dynamic

    function true_theta_e(df, e)
        # cohort-size weighted average of τ(g, e) across cohorts observed at e
        u = unique(df[df.g .> 0, [:unit, :g]])
        gs = u.g .- 1999
        T = maximum(df.time) - 1999
        ok = [g + e in 1:T for g in gs]
        return mean(eff(g, e) for g in gs[ok])
    end

    @testset "truth recovery: ATT(g,t) and aggregations" begin
        rng = StableRNG(501)
        df = sim_staggered(rng; N=3000, T=6, cohorts=[0, 3, 4, 5], effect=eff)
        cs = did_callaway_santanna(df, :y, :d, :unit, :time; rng=StableRNG(1))
        @test cs isa CallawaySantAnnaEstimate
        se = stderror(cs)
        truth = [eff(g, t - g) for (g, t) in zip(cs.groups, cs.times)]
        @test all(abs.(cs.coef .- truth) .< 4.5 .* se)
        @test length(cs.coef) == 3 * 5
        @test coefnames(cs)[1] == "ATT(g=2002, t=2001)"
        simple = aggregate_att(cs, :simple)
        @test simple isa AggregatedATT
        @test abs(coef(simple)[1] - true_simple_att(df, eff)) < 4 * stderror(simple)[1]
        @test estimate(cs) ≈ coef(simple)[1]
        dyn = aggregate_att(cs, :dynamic)
        @test dyn isa EventStudyEstimate
        @test relative_periods(dyn) == collect(-3:3)
        for (k, e) in enumerate(relative_periods(dyn))
            @test abs(coef(dyn)[k] - true_theta_e(df, e)) < 4.5 * stderror(dyn)[k]
        end
        @test coef(dyn.details.overall)[1] ≈ mean(coef(dyn)[4:7])
        @test coef(event_study_average(dyn))[1] ≈ coef(dyn.details.overall)[1]
        @test stderror(event_study_average(dyn))[1] ≈ stderror(dyn.details.overall)[1]
        grp = aggregate_att(cs, :group)
        @test grp.labels == [2002, 2003, 2004]
        @test coefnames(grp) == ["ATT", "g=2002", "g=2003", "g=2004"]
        cal = aggregate_att(cs, :calendar)
        @test cal.labels == [2002, 2003, 2004, 2005]
        @test_throws ArgumentError aggregate_att(cs, :foo)
        @test_throws ArgumentError aggregate_att(cs, :dynamic; min_e=10)
        bal = aggregate_att(cs, :dynamic; balance_e=1)
        @test maximum(relative_periods(bal)) == 1
        t = pre_trend_test(cs)
        @test t isa DiagnosticTest && t.dof == (count(cs.times .< cs.groups),)
        @test occursin("Callaway", sprint(show, MIME"text/plain"(), cs))
    end

    @testset "covariates: conditional parallel trends" begin
        rng = StableRNG(502)
        df = sim_staggered(rng; N=4000, T=5, cohorts=[0, 3, 4], effect=(g, e) -> 1.0,
                           confounded=true, gamma=0.8)
        naive = aggregate_att(did_callaway_santanna(df, :y, :d, :unit, :time;
                                                    bootstrap=false), :simple)
        @test abs(coef(naive)[1] - 1.0) > 4 * stderror(naive)[1]
        for m in (:dr, :dr_improved, :ipw, :reg)
            cs = did_callaway_santanna(df, :y, :d, :unit, :time; covariates=[:x],
                                       method=m, bootstrap=false)
            s = aggregate_att(cs, :simple)
            @test abs(coef(s)[1] - 1.0) < 4 * stderror(s)[1]
        end
        cny = did_callaway_santanna(df, :y, :d, :unit, :time; covariates=[:x],
                                    control_group=:not_yet_treated, bootstrap=false)
        s = aggregate_att(cny, :simple)
        @test abs(coef(s)[1] - 1.0) < 4 * stderror(s)[1]
    end

    @testset "anticipation, base periods, no never-treated" begin
        rng = StableRNG(503)
        ant(g, e) = e >= -1 ? 1.0 : 0.0              # effects start one period early
        df = sim_staggered(rng; N=3000, T=7, cohorts=[0, 4, 6], effect=ant,
                           anticipation_effects=true)
        cs0 = did_callaway_santanna(df, :y, :d, :unit, :time; bootstrap=false)
        cs1 = did_callaway_santanna(df, :y, :d, :unit, :time; anticipation=1,
                                    bootstrap=false)
        d1 = aggregate_att(cs1, :dynamic)
        @test abs(coef(d1)[findfirst(==(0), relative_periods(d1))] - 1.0) <
              4 * stderror(d1)[findfirst(==(0), relative_periods(d1))]
        d0 = aggregate_att(cs0, :dynamic)
        k0 = findfirst(==(0), relative_periods(d0))
        @test abs(coef(d0)[k0] - 1.0) > 4 * stderror(d0)[k0]    # ignores anticipation
        u = aggregate_att(did_callaway_santanna(df, :y, :d, :unit, :time;
                                                base_period=:universal,
                                                bootstrap=false), :dynamic)
        @test u.reference == [-1]
        @test !(-1 in relative_periods(u))
        nn = sim_staggered(rng; N=600, T=6, cohorts=[3, 5], effect=eff)
        csn = @test_logs (:warn, r"no never-treated") match_mode = :any begin
            did_callaway_santanna(nn, :y, :d, :unit, :time; bootstrap=false)
        end
        @test all(csn.times .< 5)                     # periods from 5 on dropped
        @test Set(csn.groups) == Set([3])
        csy = did_callaway_santanna(nn, :y, :d, :unit, :time;
                                    control_group=:not_yet_treated, bootstrap=false)
        @test Set(csy.groups) == Set([3]) && all(csy.times .< 5)
        # units treated in the first period are dropped with a warning
        early = sim_staggered(rng; N=200, T=5, cohorts=[0, 1, 3])
        @test_logs (:warn, r"already treated") match_mode = :any did_callaway_santanna(
            early, :y, :d, :unit, :time; bootstrap=false)
    end

    @testset "inference plumbing: clusters, bootstrap, invariance" begin
        rng = StableRNG(504)
        df = sim_staggered(rng; N=300, T=5, cohorts=[0, 3, 4], effect=eff)
        cs = did_callaway_santanna(df, :y, :d, :unit, :time; rng=StableRNG(7), biters=499)
        @test length(cs.supt_draws) == 499
        cs2 = did_callaway_santanna(df, :y, :d, :unit, :time; rng=StableRNG(7), biters=499)
        @test cs.supt_draws == cs2.supt_draws           # reproducible with rng
        cu = confint(cs; uniform=true)
        cp = confint(cs)
        @test all(cu[:, 2] .- cu[:, 1] .>= cp[:, 2] .- cp[:, 1])
        dyn = aggregate_att(cs, :dynamic; rng=StableRNG(2))
        @test !isempty(dyn.supt_draws)
        @test all(diff(confint(dyn; uniform=true); dims=2) .>= diff(confint(dyn); dims=2))
        df.state = (df.unit .- 1) .÷ 6
        csc = did_callaway_santanna(df, :y, :d, :unit, :time; cluster=:state,
                                    bootstrap=false)
        @test csc.n_clusters == 50
        @test coef(csc) ≈ coef(cs)
        @test !(stderror(csc) ≈ stderror(cs))
        bad = copy(df); bad.state[1] = 999
        @test_throws ArgumentError did_callaway_santanna(bad, :y, :d, :unit, :time;
                                                         cluster=:state)
        sh = did_callaway_santanna(shuffle_rows(rng, df), :y, :d, :unit, :time;
                                   bootstrap=false)
        @test coef(sh) ≈ coef(cs)
        @test vcov(sh) ≈ vcov(cs)
        fcs = did_callaway_santanna(df, :y, FirstTreated(:g), :unit, :time; bootstrap=false)
        @test coef(fcs) ≈ coef(cs)
        # unbalanced panels: incomplete units dropped with a warning
        @test_logs (:warn, r"not observed in every period") match_mode = :any begin
            did_callaway_santanna(df[2:end, :], :y, :d, :unit, :time; bootstrap=false)
        end
        @test_throws ArgumentError did_callaway_santanna(df, :y, :d, :unit, :time;
                                                         control_group=:all)
        @test_throws ArgumentError did_callaway_santanna(df, :y, :d, :unit, :time;
                                                         base_period=:x)
    end

    @testset "repeated cross-sections" begin
        rng = StableRNG(506)
        pan = sim_staggered(rng; N=6000, T=5, cohorts=[0, 3, 4], effect=eff)
        # keep one random period per unit: a repeated cross-section
        pick = rand(rng, 1:5, 6000)
        rc = pan[[(u - 1) * 5 + pick[u] for u in 1:6000], :]
        cs = did_callaway_santanna(rc, :y, FirstTreated(:g), nothing, :time;
                                   bootstrap=false)
        @test nobs(cs) == 6000
        truth = [eff(g, t - g) for (g, t) in zip(cs.groups, cs.times)]
        @test all(abs.(cs.coef .- truth) .< 4.5 .* stderror(cs))
        s = aggregate_att(cs, :simple)
        @test isfinite(coef(s)[1]) && stderror(s)[1] > 0
        @test coef(did_callaway_santanna(shuffle_rows(rng, rc), :y, FirstTreated(:g),
                                         nothing, :time; bootstrap=false)) ≈ coef(cs)
        @test_throws ArgumentError did_callaway_santanna(rc, :y, :d, nothing, :time)
        @test occursin("Observations", sprint(show, MIME"text/plain"(), cs))
    end

    @testset "Monte Carlo: analytic SEs and uniform bands" begin
        R = mc_reps(1000, 100)
        rng = StableRNG(505)
        hit_s = 0; hit_0 = 0; hit_u = 0
        for _ in 1:R
            df = sim_staggered(rng; N=300, T=5, cohorts=[0, 3, 4], effect=eff)
            cs = did_callaway_santanna(df, :y, :d, :unit, :time; rng=rng, biters=199)
            s = aggregate_att(cs, :simple; bootstrap=false)
            ci = confint(s)
            hit_s += ci[1, 1] <= true_simple_att(df, eff) <= ci[1, 2]
            dyn = aggregate_att(cs, :dynamic; rng=rng, biters=199)
            truth = [true_theta_e(df, e) for e in relative_periods(dyn)]
            k = findfirst(==(0), relative_periods(dyn))
            cd = confint(dyn)
            hit_0 += cd[k, 1] <= truth[k] <= cd[k, 2]
            cu = confint(dyn; uniform=true)
            hit_u += all(cu[:, 1] .<= truth .<= cu[:, 2])
        end
        @test mc_close(hit_s / R, 0.95, R; slack=0.02)
        @test mc_close(hit_0 / R, 0.95, R; slack=0.02)
        @test mc_close(hit_u / R, 0.95, R; slack=0.03)
    end
end
