# Heteroskedasticity- / cluster-robust CLR and K tests (Kleibergen 2005; Stata weakiv).
# References: test/validation/iv/make_reference_robust_extrap.R (formulas coded
# independently in R, AMS conditional p-value by numerical integration).

const IV_REF_RX = let r = iv_read_csv(joinpath(IV_VALDIR, "reference_robust_extrap.csv"))
    Dict((row.case, row.quantity) => row.value for row in eachrow(r))
end

@testset "Robust CLR and K" begin
    @testset "validation against R (Card, 2 instruments, HC1 and cluster)" begin
        for (tp, kw) in (("hc1", (;)), ("cluster", (cluster=:region,)))
            r = late_2sls(CARD, :lwage, :educ, [:nearc4, :nearc2]; covariates=CARD_CTRL,
                          kw...)
            for b0 in (0, 0.1, 0.3)
                ref(q) = IV_REF_RX[("rclr_$tp", "$(q)_$(b0)")]
                t = weak_iv_test(r; beta0=b0, method=:clr)
                k = weak_iv_test(r; beta0=b0, method=:k)
                @test t.details.AR ≈ ref("AR") rtol = 1e-9
                @test t.details.rk ≈ ref("rk") rtol = 1e-9
                @test t.statistic ≈ ref("LR") rtol = 1e-8
                @test t.pvalue ≈ ref("p_clr") atol = 1e-8
                @test k.statistic ≈ ref("K") rtol = 1e-8
                @test k.pvalue ≈ ref("p_k") atol = 1e-10
                @test occursin("robust", t.name)
            end
        end
    end

    @testset "reduces to Moreira's statistics under homoskedasticity" begin
        df = iv_linear_dgp(StableRNG(401); n=300, k=3, pi=[0.2, 0.1, 0.0], rho=0.8,
                           hetero=true)
        r = late_2sls(df, :y, :d, [:z1, :z2, :z3]; covariates=[:x], vcov=Vcov.simple())
        rp = DrSnow._iv_robust_moreira_parts(r.design)
        mp = DrSnow._iv_moreira_parts(r.design)
        for β in (-2.0, 0.0, 0.7, 1.0, 3.0, Inf, -Inf)
            AR, K, J, rk = DrSnow._iv_robust_moreira_stats(rp, β)
            QS, QT, QST = DrSnow._iv_moreira_stats(mp, β)
            @test AR ≈ QS rtol = 1e-9
            @test K ≈ QST^2 / QT rtol = 1e-8 atol = 1e-12
            @test rk ≈ QT rtol = 1e-9
            lr = DrSnow._iv_lr_stat(QS, QT, QST)
            @test DrSnow._iv_robust_lr(AR, J, rk) ≈ lr rtol = 1e-8
        end
    end

    @testset "just identified: CLR = K = AR" begin
        df = iv_linear_dgp(StableRNG(402); n=300, pi=0.15, hetero=true)
        r = late_2sls(df, :y, :d, :z1; covariates=[:x])
        for b0 in (-1.0, 0.5, 1.0, 2.0)
            ar = weak_iv_test(r; beta0=b0)
            clr = weak_iv_test(r; beta0=b0, method=:clr)
            @test clr.pvalue ≈ ar.pvalue rtol = 1e-8
            @test weak_iv_test(r; beta0=b0, method=:k).statistic ≈ ar.statistic rtol = 1e-8
        end
        sa = weak_iv_confidence_set(r)
        sc = weak_iv_confidence_set(r; method=:clr)
        @test sc.kind === sa.kind
        for (a, b) in zip(sa.intervals, sc.intervals)
            @test a[1] ≈ b[1] rtol = 1e-6
            @test a[2] ≈ b[2] rtol = 1e-6
        end
    end

    @testset "confidence sets, interface, invariance" begin
        df = iv_linear_dgp(StableRNG(403); n=400, k=3, pi=[0.3, 0.2, 0.1], hetero=true,
                           G=40)
        r = late_2sls(df, :y, :d, [:z1, :z2, :z3]; covariates=[:x], cluster=:g)
        for m in (:clr, :k)
            cs = weak_iv_confidence_set(r; method=m)
            @test occursin("cluster", cs.method)
            @test !isempty(cs.intervals)
            for (a, b) in cs.intervals
                isfinite(a) && @test DrSnow.pvalue(cs, a) ≈ 0.05 atol = 1e-6
                isfinite(b) && @test DrSnow.pvalue(cs, b) ≈ 0.05 atol = 1e-6
            end
            mid = first(cs.intervals)
            if all(isfinite, mid)
                @test DrSnow.pvalue(cs, (mid[1] + mid[2]) / 2) >= 0.05 - 1e-8
            end
        end
        sh = df[shuffle(StableRNG(404), 1:nrow(df)), :]
        rs = late_2sls(sh, :y, :d, [:z1, :z2, :z3]; covariates=[:x], cluster=:g)
        @test weak_iv_test(rs; beta0=0.8, method=:clr).statistic ≈
              weak_iv_test(r; beta0=0.8, method=:clr).statistic rtol = 1e-8
        r2 = iv_regression(df, :y, [:d, :x], [:z1, :z2, :z3])
        @test_throws ArgumentError weak_iv_test(r2; beta0=[1.0, 1.0], method=:clr)
    end

    @testset "Monte Carlo: size under heteroskedasticity and clustering" begin
        R = mc_reps(2000, 300)
        rng = StableRNG(405)
        rej = zeros(4)
        for _ in 1:R
            df = iv_linear_dgp(rng; n=300, k=4, pi=[0.1, 0.05, 0.0, 0.0], rho=0.9,
                               hetero=true)
            r = late_2sls(df, :y, :d, [:z1, :z2, :z3, :z4]; covariates=[:x])
            rej[1] += weak_iv_test(r; beta0=1.0, method=:clr).pvalue < 0.05
            rej[2] += weak_iv_test(r; beta0=1.0, method=:k).pvalue < 0.05
            dc = iv_linear_dgp(rng; n=600, k=3, pi=[0.08, 0.04, 0.0], rho=0.8, G=60)
            rc = late_2sls(dc, :y, :d, [:z1, :z2, :z3]; covariates=[:x], cluster=:g)
            rej[3] += weak_iv_test(rc; beta0=1.0, method=:clr).pvalue < 0.05
            rej[4] += weak_iv_test(rc; beta0=1.0, method=:k).pvalue < 0.05
        end
        rej ./= R
        @test abs(rej[1] - 0.05) < mc_tol(0.05, R; slack=0.02)
        @test abs(rej[2] - 0.05) < mc_tol(0.05, R; slack=0.02)
        @test abs(rej[3] - 0.05) < mc_tol(0.05, R; slack=0.03)
        @test abs(rej[4] - 0.05) < mc_tol(0.05, R; slack=0.03)
    end
end
