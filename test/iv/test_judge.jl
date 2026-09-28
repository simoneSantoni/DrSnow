# Judge / examiner designs.

const JUDGE_REF = let r = iv_read_csv(joinpath(IV_VALDIR, "reference_judge.csv"))
    Dict(row.quantity => row.value for row in eachrow(r))
end
const JUDGE_DF = let df = iv_read_csv(joinpath(IV_VALDIR, "judge.csv"))
    df.judge = Int.(df.judge)
    df.court = Int.(df.court)
    df
end

@testset "judge designs" begin
    @testset "validation against R (fixest, explicit UJIVE)" begin
        df = JUDGE_DF
        z = @test_logs (:warn, r"single case") match_mode = :any judge_leniency(
            df, :d, :judge; strata=[:court])
        @test ismissing(z[end])                        # the single-case judge
        @test z[1] ≈ JUDGE_REF["leniency_1"] rtol = 1e-9
        @test z[2] ≈ JUDGE_REF["leniency_2"] rtol = 1e-9
        @test std(skipmissing(z)) ≈ JUDGE_REF["leniency_sd"] rtol = 1e-9
        r = judge_iv(df, :y, :d, :judge; strata=[:court])
        @test coef(r)[1] ≈ JUDGE_REF["tsls_coef"] rtol = 1e-8
        @test stderror(r)[1] ≈ JUDGE_REF["tsls_se"] rtol = 1e-6
        @test r.first_stage.coef[1] ≈ JUDGE_REF["fs_coef"] rtol = 1e-8
        @test sqrt(r.first_stage.vcov[1, 1]) ≈ JUDGE_REF["fs_se"] rtol = 1e-6
        @test dof_residual(r) == r.n_judges - 1
        @test r.n_judges == 48
        rr = judge_iv(df, :y, :d, :judge; strata=[:court], residualize=false)
        @test coef(rr)[1] ≈ JUDGE_REF["tsls_raw_coef"] rtol = 1e-8
        ru = judge_iv(df, :y, :d, :judge; strata=[:court], method=:ujive)
        @test coef(ru)[1] ≈ JUDGE_REF["ujive_coef"] rtol = 1e-8
        @test stderror(ru)[1] ≈ JUDGE_REF["ujive_se"] rtol = 1e-6
        b = judge_balance_test(df, :d, :judge, [:x, :g]; strata=[:court])
        @test b.statistic ≈ JUDGE_REF["balance_F"] rtol = 1e-6
        @test b.dof == (2, 47.0)
    end

    @testset "interface, printing, errors" begin
        df = iv_judge_dgp(StableRNG(201); n_courts=4, judges=5, cases=30)
        r = judge_iv(df, :y, :d, :judge; strata=[:court], covariates=[:x])
        @test r isa JudgeIVEstimate && r isa CausalEstimate
        @test coefnames(r) == ["d"]
        @test occursin("marginal", estimand(r))
        s = sprint(show, MIME"text/plain"(), r)
        @test occursin("leniency", s) && occursin("Judges: 20", s)
        @test length(r.leniency) == nrow(df)
        @test_throws ArgumentError judge_iv(df, :y, :d, :judge; method=:foo)
        @test_throws ArgumentError judge_iv(df, :y, :d, :nojudge)
        df1 = DataFrame(y=randn(4), d=[0.0, 1, 0, 1], judge=[1, 1, 2, 3])
        @test_throws ArgumentError judge_iv(df1, :y, :d, :judge)
        @test_throws ArgumentError judge_balance_test(df, :d, :judge, Symbol[])
        @test_throws ArgumentError judge_validity_test(df, :y, :d, :judge; n_segments=0,
                                                       method=:minimum_distance)
        @test_throws ArgumentError judge_validity_test(df, :y, :d, :judge;
                                                       outcome_bounds=(1, 0))
        @test_throws ArgumentError judge_subsample_monotonicity(df, :d, :judge, Symbol[])
        # clustering can be changed
        r2 = judge_iv(df, :y, :d, :judge; strata=[:court], covariates=[:x],
                      vcov=Vcov.robust())
        @test coef(r2) ≈ coef(r) rtol = 1e-10
        @test occursin("HC1", r2.se_type)
    end

    @testset "row-shuffling invariance" begin
        df = iv_judge_dgp(StableRNG(202); n_courts=4, judges=5, cases=30, slope=0.5)
        sh = df[shuffle(StableRNG(203), 1:nrow(df)), :]
        for m in (:leniency, :ujive)
            a = judge_iv(df, :y, :d, :judge; strata=[:court], method=m)
            b = judge_iv(sh, :y, :d, :judge; strata=[:court], method=m)
            @test coef(a) ≈ coef(b) rtol = 1e-9
            @test stderror(a) ≈ stderror(b) rtol = 1e-7
        end
        ta = judge_validity_test(df, :y, :d, :judge; strata=[:court], rng=StableRNG(1),
                                 n_knots=3)
        tb = judge_validity_test(sh, :y, :d, :judge; strata=[:court], rng=StableRNG(1),
                                 n_knots=3)
        @test ta.statistic ≈ tb.statistic rtol = 1e-8
        ma = judge_validity_test(df, :y, :d, :judge; strata=[:court], rng=StableRNG(1),
                                 n_bootstrap=99, method=:minimum_distance)
        mb = judge_validity_test(sh, :y, :d, :judge; strata=[:court], rng=StableRNG(1),
                                 n_bootstrap=99, method=:minimum_distance)
        @test ma.statistic ≈ mb.statistic rtol = 1e-8
    end

    @testset "truth recovery and coverage (constant effect)" begin
        R = mc_reps(400, 60)
        rng = StableRNG(204)
        cov = zeros(Bool, R, 2)
        est = zeros(R)
        for rep in 1:R
            df = iv_judge_dgp(rng; n_courts=6, judges=6, cases=50, tau=1.0)
            a = judge_iv(df, :y, :d, :judge; strata=[:court])
            u = judge_iv(df, :y, :d, :judge; strata=[:court], method=:ujive)
            est[rep] = coef(u)[1]
            for (j, r) in enumerate((a, u))
                ci = confint(r)
                cov[rep, j] = ci[1, 1] <= 1.0 <= ci[1, 2]
            end
        end
        @test abs(median(est) - 1.0) < 0.1
        for j in 1:2
            @test abs(mean(cov[:, j]) - 0.95) < mc_tol(0.95, R; slack=0.03)
        end
    end

    @testset "validity test and monotonicity checks: size and power" begin
        R = mc_reps(200, 30)
        rng = StableRNG(205)
        rej = zeros(Bool, R, 5)
        for rep in 1:R
            ok = iv_judge_dgp(rng; n_courts=6, judges=8, cases=60, slope=0.5)
            rej[rep, 1] = rejects(judge_validity_test(ok, :y, :d, :judge; strata=[:court],
                                                      n_simulations=999, rng=rng))
            rej[rep, 2] = rejects(judge_subsample_monotonicity(ok, :d, :judge, [:g];
                                                               strata=[:court]))
            rej[rep, 3] = rejects(judge_balance_test(ok, :d, :judge, [:x, :g];
                                                     strata=[:court]))
            bad = iv_judge_dgp(rng; n_courts=6, judges=8, cases=60, direct=0.5)
            rej[rep, 4] = rejects(judge_validity_test(bad, :y, :d, :judge;
                                                      strata=[:court], n_simulations=999,
                                                      rng=rng))
            mono = iv_judge_dgp(rng; n_courts=6, judges=8, cases=60, defy=1.0)
            rej[rep, 5] = rejects(judge_subsample_monotonicity(mono, :d, :judge, [:g];
                                                               strata=[:court]))
        end
        @test mean(rej[:, 1]) < 0.05 + mc_tol(0.05, R; slack=0.03)
        @test mean(rej[:, 2]) < 0.05 + mc_tol(0.05, R; slack=0.02)
        @test mean(rej[:, 3]) < 0.05 + mc_tol(0.05, R; slack=0.03)
        @test mean(rej[:, 4]) > 0.8
        @test mean(rej[:, 5]) > 0.8
    end

    @testset "validity test details" begin
        df = iv_judge_dgp(StableRNG(206); n_courts=5, judges=8, cases=50)
        t = judge_validity_test(df, :y, :d, :judge; strata=[:court], n_segments=2,
                                n_bootstrap=199, rng=StableRNG(1),
                                method=:minimum_distance)
        @test t.details.judge_means isa DataFrame
        @test nrow(t.details.judge_means) == 40
        @test all(abs.(t.details.slopes) .<= t.details.slope_bound + 1e-9)
        @test 0 <= t.details.pvalue_chisq <= 1
        # binary outcome bound
        tb = judge_validity_test(df, :d, :d, :judge; n_bootstrap=99, rng=StableRNG(2),
                                 outcome_bounds=(0, 1), method=:minimum_distance)
        @test tb.details.slope_bound == 1.0
        m = judge_subsample_monotonicity(df, :d, :judge, [:g]; strata=[:court])
        @test Set(m.details.table.sample) == Set(["standard", "reverse"])
        @test all(m.details.table.coef .> 0)
    end
end
