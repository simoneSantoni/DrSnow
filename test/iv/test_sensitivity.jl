@testset "plausibly exogenous (Conley, Hansen & Rossi 2012)" begin
    @testset "degenerate supports / priors reproduce 2SLS" begin
        df = iv_linear_dgp(StableRNG(71); n=500, k=2, hetero=true, G=30)
        r = late_2sls(df, :y, :d, [:z1, :z2]; covariates=[:x], cluster=:g)
        ci = confint(r)
        u = plausibly_exogenous(r; gamma=[(0.0, 0.0), (0.0, 0.0)])
        @test u.lower ≈ ci[1, 1] rtol = 1e-8
        @test u.upper ≈ ci[1, 2] rtol = 1e-8
        l = plausibly_exogenous(r; method=:ltz, gamma_mean=[0.0, 0.0],
                                gamma_vcov=zeros(2, 2))
        @test l.lower ≈ ci[1, 1] rtol = 1e-8
        @test l.estimate ≈ estimate(r)
        @test occursin("union", sprint(show, MIME"text/plain"(), u))
        @test occursin("local-to-zero", sprint(show, MIME"text/plain"(), l))
    end

    @testset "sign and units: a positive direct effect biases 2SLS upward" begin
        df = iv_linear_dgp(StableRNG(72); n=50_000, pi=0.5, beta=1.0, gamma=0.2)
        r = late_2sls(df, :y, :d, :z1; covariates=[:x])
        A = 1 / 0.5
        @test estimate(r) - 1.0 ≈ A * 0.2 atol = 0.05
        l = plausibly_exogenous(r; method=:ltz, gamma_mean=0.2, gamma_vcov=0.0)
        @test l.adjustment[1] ≈ A rtol = 0.05
        @test abs(l.estimate - 1.0) < 4 * l.se
        u = plausibly_exogenous(r; gamma=(0.15, 0.25))
        @test u.lower < 1.0 < u.upper
        @test u.estimate_range[1] ≈ estimate(r) - l.adjustment[1] * 0.25
        @test u.estimate_range[2] ≈ estimate(r) - l.adjustment[1] * 0.15
        # a wider support gives a wider interval
        w = plausibly_exogenous(r; gamma=(0.0, 0.4))
        @test w.lower < u.lower && w.upper > u.upper
    end

    @testset "input validation" begin
        df = iv_linear_dgp(StableRNG(73); n=300, k=2)
        r = late_2sls(df, :y, :d, [:z1, :z2])
        @test_throws ArgumentError plausibly_exogenous(r)                  # no gamma
        @test_throws ArgumentError plausibly_exogenous(r; gamma=(0.0, 1.0))  # k = 2
        @test_throws ArgumentError plausibly_exogenous(r; gamma=[(1.0, 0.0), (0.0, 0.0)])
        @test_throws ArgumentError plausibly_exogenous(r; method=:ltz, gamma_mean=0.0,
                                                       gamma_vcov=1.0)
        @test_throws ArgumentError plausibly_exogenous(r; method=:ltz,
                                                       gamma_mean=[0.0, 0.0],
                                                       gamma_vcov=[1.0 0.0; 0.0 -1.0])
        @test_throws ArgumentError plausibly_exogenous(r; method=:bayes)
        df.d2 = df.z2 .+ randn(StableRNG(74), 300)
        r2 = iv_regression(df, :y, [:d, :d2], [:z1, :z2])
        @test_throws ArgumentError plausibly_exogenous(r2; gamma=[(0.0, 0.1), (0.0, 0.1)])
    end

    @testset "Monte Carlo: UCI and LTZ coverage with a direct effect" begin
        R = mc_reps(2000, 300)
        rng = StableRNG(75)
        cov = zeros(3)
        for _ in 1:R
            γ = 0.2 + 0.1 * randn(rng)                  # drawn from the LTZ prior
            df = iv_linear_dgp(rng; n=400, pi=0.5, beta=1.0, rho=0.5, hetero=true,
                               gamma=γ)
            r = late_2sls(df, :y, :d, :z1; covariates=[:x])
            ci = confint(r)
            cov[1] += ci[1, 1] <= 1.0 <= ci[1, 2]
            l = plausibly_exogenous(r; method=:ltz, gamma_mean=0.2, gamma_vcov=0.01)
            cov[2] += l.lower <= 1.0 <= l.upper
            # UCI with a support that contains the true γ
            u = plausibly_exogenous(r; gamma=(γ - 0.05, γ + 0.05))
            cov[3] += u.lower <= 1.0 <= u.upper
        end
        cov ./= R
        @test cov[1] < 0.8                                   # ignoring γ fails
        @test abs(cov[2] - 0.95) < mc_tol(0.95, R; slack=0.015)
        @test cov[3] >= 0.95 - mc_tol(0.95, R; slack=0.0)   # conservative
    end
end
