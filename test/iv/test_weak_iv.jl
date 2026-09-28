using DrSnow: pvalue   # StatsAPI.pvalue (not exported by core)

@testset "weak-IV-robust inference" begin
    @testset "AR set: all four shapes, endpoints solve AR(β) = c" begin
        # bounded: strong instrument
        df = iv_linear_dgp(StableRNG(31); n=500, pi=0.8)
        for kw in ((vcov=Vcov.simple(),), NamedTuple())
            r = late_2sls(df, :y, :d, :z1; covariates=[:x], kw...)
            s = weak_iv_confidence_set(r)
            @test s.kind === :bounded
            a, b = s.intervals[1]
            @test a < estimate(r) < b
            @test pvalue(s, a) ≈ 0.05 atol = 1e-8
            @test pvalue(s, b) ≈ 0.05 atol = 1e-8
            @test estimate(r) in s
            @test !(b + 1 in s)
            @test pvalue(s, a) ≈ weak_iv_test(r; beta0=a).pvalue rtol = 1e-8
        end
        # union of two rays / whole line: irrelevant instrument
        found = Set{Symbol}()
        for seed in 1:60
            d = iv_linear_dgp(StableRNG(100 + seed); n=200, pi=0.0, rho=0.9)
            s = weak_iv_confidence_set(late_2sls(d, :y, :d, :z1; covariates=[:x]))
            push!(found, s.kind)
            if s.kind === :union_of_rays
                (_, a), (b, _) = s.intervals
                @test a < b
                @test pvalue(s, a) ≈ 0.05 atol = 1e-8
                @test pvalue(s, (a + b) / 2) < 0.05
                @test 1e6 in s && -1e6 in s
            elseif s.kind === :real_line
                @test pvalue(s, Inf) >= 0.05
            end
        end
        @test :union_of_rays in found
        @test :real_line in found
        # empty: overidentified, homoskedastic, instruments disagree strongly
        rng = StableRNG(32)
        n = 400
        z1, z2 = randn(rng, n), randn(rng, n)
        d = z1 .+ z2 .+ 0.5 .* randn(rng, n)
        y = d .+ 3 .* z1 .+ 0.1 .* randn(rng, n)          # z1 violates exclusion
        dfe = DataFrame(y=y, d=d, z1=z1, z2=z2)
        re = late_2sls(dfe, :y, :d, [:z1, :z2]; vcov=Vcov.simple())
        se_ = weak_iv_confidence_set(re)
        @test se_.kind === :empty
        @test isempty(se_.intervals)
        @test occursin("empty", sprint(show, MIME"text/plain"(), se_))
    end

    @testset "robust overidentified AR set: polynomial roots vs brute force" begin
        for seed in 1:6
            df = iv_linear_dgp(StableRNG(40 + seed); n=300, k=3, pi=[0.15, 0.1, 0.05],
                               hetero=true, G=25)
            for kw in (NamedTuple(), (cluster=:g,))
                r = late_2sls(df, :y, :d, [:z1, :z2, :z3]; covariates=[:x], kw...)
                s = weak_iv_confidence_set(r)
                # brute force: dense scan of the exact p-value function
                grid = estimate(r) .+ 60 .* sinh.(range(-8, 8; length=20_001)) ./ sinh(8)
                acc = [pvalue(s, b) >= 0.05 for b in grid]
                mem = [b in s for b in grid]
                mismatch = findall(acc .!= mem)
                # disagreements only within numerical distance of a boundary
                bds = filter(isfinite, vcat([collect(iv) for iv in s.intervals]...))
                @test all(any(abs(grid[i] - bd) < 1e-6 * (1 + abs(bd)) for bd in bds)
                          for i in mismatch)
                for bd in bds
                    @test pvalue(s, bd) ≈ 0.05 atol = 1e-7
                end
                left = !isempty(s.intervals) && isinf(s.intervals[1][1])
                @test (pvalue(s, -Inf) >= 0.05) == left
            end
        end
    end

    @testset "several endogenous regressors: joint AR test" begin
        rng = StableRNG(33)
        n = 2000
        Z = randn(rng, n, 3)
        v = randn(rng, n, 2)
        d1 = Z * [1.0, 0.5, 0.0] .+ v[:, 1]
        d2 = Z * [0.0, 0.5, 1.0] .+ v[:, 2]
        y = d1 .- d2 .+ v[:, 1] .+ randn(rng, n)
        df = DataFrame(y=y, d1=d1, d2=d2, z1=Z[:, 1], z2=Z[:, 2], z3=Z[:, 3])
        r = iv_regression(df, :y, [:d1, :d2], [:z1, :z2, :z3])
        @test weak_iv_test(r; beta0=[1.0, -1.0]).pvalue > 0.001
        @test weak_iv_test(r; beta0=[0.0, 0.0]).pvalue < 1e-6
        @test_throws ArgumentError weak_iv_test(r; beta0=[1.0])
        @test_throws ArgumentError weak_iv_confidence_set(r)
        @test_throws ArgumentError weak_iv_test(r; method=:clr)
    end

    @testset "CLR and K" begin
        df = iv_linear_dgp(StableRNG(34); n=500, k=3, pi=0.2)
        r = late_2sls(df, :y, :d, [:z1, :z2, :z3]; covariates=[:x], vcov=Vcov.simple())
        tc = weak_iv_test(r; beta0=1.0, method=:clr)
        tk = weak_iv_test(r; beta0=1.0, method=:k)
        @test 0 <= tc.pvalue <= 1 && 0 <= tk.pvalue <= 1
        @test tk.statistic ≈ tk.details.QST^2 / tk.details.QT
        # CLR = AR (F form) when k = 1
        r1 = late_2sls(df, :y, :d, :z1; covariates=[:x], vcov=Vcov.simple())
        @test weak_iv_test(r1; beta0=0.3, method=:clr).pvalue ≈
              weak_iv_test(r1; beta0=0.3).pvalue rtol = 1e-8
        s1 = weak_iv_confidence_set(r1; method=:clr)
        sa = weak_iv_confidence_set(r1)
        @test s1.intervals[1][1] ≈ sa.intervals[1][1] rtol = 1e-6
        @test s1.intervals[1][2] ≈ sa.intervals[1][2] rtol = 1e-6
        sk = weak_iv_confidence_set(r; method=:k)
        @test estimate(r) in sk
        # robust covariance uses the Kleibergen (2005) robust versions
        # (test_weak_iv_robust.jl)
        rr = late_2sls(df, :y, :d, [:z1, :z2, :z3]; covariates=[:x])
        @test occursin("robust", weak_iv_test(rr; method=:clr).name)
        @test occursin("HC1", weak_iv_confidence_set(rr; method=:k).method)
        @test_throws ArgumentError weak_iv_test(rr; method=:wald)
    end

    @testset "CLR conditional p-value matches simulation" begin
        rng = StableRNG(35)
        for (k, qT, m) in ((2, 3.0, 4.0), (3, 10.0, 5.0), (5, 1.0, 9.0))
            B = 200_000
            q1 = randn(rng, B) .^ 2
            qk = [sum(abs2, randn(rng, k - 1)) for _ in 1:B]
            lr = 0.5 .* (q1 .+ qk .- qT .+ sqrt.((q1 .+ qk .+ qT) .^ 2 .- 4 .* qT .* qk))
            @test DrSnow._iv_clr_pvalue(m, qT, k, Inf) ≈ mean(lr .> m) atol = 0.004
        end
        # qT = 0: LR = Q_S ~ χ²_k exactly
        @test DrSnow._iv_clr_pvalue(7.0, 0.0, 4, Inf) ≈
              DrSnow.ccdf(DrSnow.Chisq(4), 7.0) atol = 1e-10
    end

    @testset "Monte Carlo: AR coverage with weak instruments" begin
        R = mc_reps(2000, 300)
        rng = StableRNG(36)
        cov = zeros(4)
        for _ in 1:R
            df = iv_linear_dgp(rng; n=200, pi=0.08, beta=1.0, rho=0.9, hetero=true)
            r = late_2sls(df, :y, :d, :z1; covariates=[:x])
            ci = confint(r)
            cov[1] += ci[1, 1] <= 1.0 <= ci[1, 2]
            cov[2] += 1.0 in weak_iv_confidence_set(r)
            dfc = iv_linear_dgp(rng; n=400, k=2, pi=0.05, beta=1.0, rho=0.8, G=40)
            rc = late_2sls(dfc, :y, :d, [:z1, :z2]; covariates=[:x], cluster=:g)
            cov[3] += 1.0 in weak_iv_confidence_set(rc)
            rh = late_2sls(dfc, :y, :d, [:z1, :z2]; covariates=[:x], vcov=Vcov.simple())
            cov[4] += 1.0 in weak_iv_confidence_set(rh)
        end
        cov ./= R
        @test cov[1] < 0.85                         # 2SLS t-interval fails
        @test abs(cov[2] - 0.95) < mc_tol(0.95, R)  # robust AR, just identified
        @test abs(cov[3] - 0.95) < mc_tol(0.95, R; slack=0.03)  # cluster AR, k = 2
        @test cov[4] < cov[3]                       # iid AR ignores clustering
    end

    @testset "Monte Carlo: CLR and K size under weak instruments (homoskedastic)" begin
        R = mc_reps(2000, 300)
        rng = StableRNG(37)
        rej = zeros(3)
        for _ in 1:R
            df = iv_linear_dgp(rng; n=250, k=4, pi=[0.1, 0.05, 0.0, 0.0], rho=0.9)
            r = late_2sls(df, :y, :d, [:z1, :z2, :z3, :z4]; covariates=[:x],
                          vcov=Vcov.simple())
            rej[1] += weak_iv_test(r; beta0=1.0, method=:clr).pvalue < 0.05
            rej[2] += weak_iv_test(r; beta0=1.0, method=:k).pvalue < 0.05
            rej[3] += weak_iv_test(r; beta0=1.0).pvalue < 0.05
        end
        rej ./= R
        for j in 1:3
            @test abs(rej[j] - 0.05) < mc_tol(0.05, R)
        end
    end
end
