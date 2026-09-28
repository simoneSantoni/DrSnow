# Huber & Mellace (2015) instrument-validity test.

@testset "huber_mellace_test" begin
    @testset "statistic, details and errors" begin
        df, _ = iv_binary_dgp(StableRNG(501); n=3000)
        t = huber_mellace_test(df, :y, :d, :z; n_bootstrap=199, rng=StableRNG(1))
        @test t isa DiagnosticTest
        @test length(t.details.theta) == 4
        @test 0 < t.details.q < 1 && 0 < t.details.r < 1
        # q and r from the arm-specific take-up rates
        p1, p0 = mean(df.d[df.z .== 1]), mean(df.d[df.z .== 0])
        @test t.details.q ≈ p0 / p1
        @test t.details.r ≈ (1 - p1) / (1 - p0)
        # θ₁ from its definition
        g11 = sort(df.y[(df.d .== 1) .& (df.z .== 1)])
        k = round(Int, t.details.q * length(g11))
        @test t.details.theta[1] ≈ mean(g11[1:k]) - mean(df.y[(df.d .== 1) .& (df.z .== 0)])
        # reproducible with the same rng; invariant to recoding Z as 1 − Z
        t2 = huber_mellace_test(df, :y, :d, :z; n_bootstrap=199, rng=StableRNG(1))
        @test t2.pvalue == t.pvalue
        df.zr = 1 .- df.z
        t3 = huber_mellace_test(df, :y, :d, :zr; n_bootstrap=199, rng=StableRNG(1))
        @test t3.statistic ≈ t.statistic
        @test_throws ArgumentError huber_mellace_test(df, :y, :d, :z; n_bootstrap=10)
        @test_throws ArgumentError huber_mellace_test(df, :y, :x, :z)
    end

    @testset "Monte Carlo size and power" begin
        R = mc_reps(300, 40)
        rng = StableRNG(502)
        rej = zeros(Bool, R, 2)
        for rep in 1:R
            ok, _ = iv_binary_dgp(rng; n=1500, pa=0.25)
            rej[rep, 1] = rejects(huber_mellace_test(ok, :y, :d, :z; n_bootstrap=199,
                                                     rng=rng))
            bad, _ = iv_binary_dgp(rng; n=1500, pa=0.25, direct=2.5)
            rej[rep, 2] = rejects(huber_mellace_test(bad, :y, :d, :z; n_bootstrap=199,
                                                     rng=rng))
        end
        @test mean(rej[:, 1]) < 0.05 + mc_tol(0.05, R; slack=0.02)
        @test mean(rej[:, 2]) > 0.7
    end
end
