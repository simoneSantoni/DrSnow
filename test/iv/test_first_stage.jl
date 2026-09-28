@testset "first-stage and weak-IV diagnostics" begin
    @testset "one instrument: effective F = Wald F, OP critical values" begin
        df = iv_linear_dgp(StableRNG(21); n=600, hetero=true, G=30)
        for kw in ((vcov=Vcov.simple(),), NamedTuple(), (cluster=:g,))
            r = late_2sls(df, :y, :d, :z1; covariates=[:x], kw...)
            fs = r.first_stage
            @test fs.effective_F ≈ fs.first_stage[1].F rtol = 1e-10
            # K_eff = 1 whenever k = 1: the familiar 23.109 (τ = 10%) etc.
            @test fs.op_critical_values.K_eff ≈ 1.0
            @test fs.op_critical_values.tau_10 ≈ 23.1085 atol = 1e-3
            @test fs.op_critical_values.tau_5 ≈ 37.418 atol = 1e-2
            @test fs.op_critical_values.tau_20 ≈ 15.062 atol = 1e-2
            @test fs.op_critical_values.tau_30 ≈ 12.039 atol = 1e-2
        end
        r = late_2sls(df, :y, :d, :z1; covariates=[:x], vcov=Vcov.simple())
        fs = r.first_stage
        @test fs.cragg_donald_F ≈ fs.first_stage[1].F_homoskedastic rtol = 1e-10
        @test fs.first_stage[1].F ≈ fs.first_stage[1].F_homoskedastic rtol = 1e-10
        # KP rk Wald F = robust first-stage Wald F with one endogenous regressor and
        # one instrument (FixedEffectModels' homoskedastic KP uses n, not the dof)
        rr = late_2sls(df, :y, :d, :z1; covariates=[:x])
        @test rr.first_stage.kleibergen_paap_F ≈ rr.first_stage.first_stage[1].F rtol = 1e-8
        @test fs.stock_yogo.size.size_10 == 16.38
        @test fs.stock_yogo.bias === nothing
        # partial R² by hand
        des = r.design
        e = des.D[:, 1] - des.Z * (des.Z \ des.D[:, 1])
        @test fs.first_stage[1].partial_r2 ≈ 1 - sum(abs2, e) / sum(abs2, des.D[:, 1])
    end

    @testset "effective F is invariant to linear recombination of instruments" begin
        df = iv_linear_dgp(StableRNG(22); n=800, k=3, pi=[0.3, 0.1, 0.2], hetero=true)
        r1 = late_2sls(df, :y, :d, [:z1, :z2, :z3]; covariates=[:x])
        df2 = copy(df)
        df2.z1 = 3 .* df.z1 .+ df.z2
        df2.z2 = df.z2 .- 0.5 .* df.z3
        df2.z3 = 10 .* df.z3
        r2 = late_2sls(df2, :y, :d, [:z1, :z2, :z3]; covariates=[:x])
        @test r1.first_stage.effective_F ≈ r2.first_stage.effective_F rtol = 1e-8
        @test r1.first_stage.op_critical_values.K_eff ≈
              r2.first_stage.op_critical_values.K_eff rtol = 1e-6
        @test coef(r1)[1] ≈ coef(r2)[1] rtol = 1e-9
        # homoskedastic covariance: K_eff = k and the effective F is the usual F
        r3 = late_2sls(df, :y, :d, [:z1, :z2, :z3]; covariates=[:x], vcov=Vcov.simple())
        @test r3.first_stage.op_critical_values.K_eff ≈ 3 rtol = 1e-8
        @test r3.first_stage.effective_F ≈ r3.first_stage.first_stage[1].F rtol = 1e-8
        @test r3.first_stage.stock_yogo.bias.bias_10 == 9.08
    end

    @testset "several endogenous regressors" begin
        rng = StableRNG(23)
        n = 3000
        Z = randn(rng, n, 3)
        v = randn(rng, n, 2)
        d1 = Z * [1.0, 0.5, 0.0] .+ v[:, 1]
        d2 = Z * [0.0, 0.0, 0.0] .+ v[:, 2]             # not identified
        y = d1 .+ d2 .+ v[:, 1] .+ randn(rng, n)
        df = DataFrame(y=y, d1=d1, d2=d2, z1=Z[:, 1], z2=Z[:, 2], z3=Z[:, 3])
        r = iv_regression(df, :y, [:d1, :d2], [:z1, :z2, :z3]; vcov=Vcov.simple())
        fs = first_stage_diagnostics(r)
        @test length(fs.first_stage) == 2
        @test fs.effective_F === nothing
        @test fs.stock_yogo === nothing
        @test fs.first_stage[1].sanderson_windmeijer_F > 50
        @test fs.first_stage[2].sanderson_windmeijer_F < 10
        @test fs.cragg_donald_F < 10
        @test occursin("Sanderson", sprint(show, MIME"text/plain"(), fs))
    end

    @testset "data method" begin
        df = iv_linear_dgp(StableRNG(24); n=300)
        fs = first_stage_diagnostics(df, :d, :z1; covariates=[:x])
        r = late_2sls(df, :y, :d, :z1; covariates=[:x])
        @test fs.first_stage[1].F ≈ r.first_stage.first_stage[1].F
        @test fs.effective_F ≈ r.first_stage.effective_F
        @test occursin("effective F", sprint(show, MIME"text/plain"(), fs))
    end

    @testset "tF critical values (LMMP 2022, Table 3)" begin
        cv = DrSnow._iv_tf_critical_value
        @test cv(4.0) == 18.66
        @test cv(3.9) == Inf
        @test cv(2.0) == Inf
        @test cv(10.0) ≈ 3.4356 atol = 1e-3       # LMMP: F = 10 gives 3.43
        @test cv(25.0) == 2.46
        @test cv(104.7) ≈ 1.967 atol = 0.001      # table step at sqrt(F) = 10.2-10.3
        @test cv(106.1) == 1.96
        @test cv(1000.0) == 1.96
        Fs = range(4.0, 110.0; length=400)
        @test all(diff(cv.(Fs)) .<= 1e-12)          # non-increasing
        df = iv_linear_dgp(StableRNG(25); n=500, pi=0.2)
        r = late_2sls(df, :y, :d, :z1; covariates=[:x])
        t = tf_confint(r)
        @test t.critical_value == cv(r.first_stage.first_stage[1].F)
        @test t.upper - t.lower ≈ 2 * t.critical_value * stderror(r)[1]
        @test_throws ArgumentError tf_confint(r; level=0.9)
        df2 = iv_linear_dgp(StableRNG(26); n=300, k=2)
        @test_throws ArgumentError tf_confint(late_2sls(df2, :y, :d, [:z1, :z2]))
    end

    @testset "Monte Carlo: tF coverage with a moderately weak instrument" begin
        R = mc_reps(3000, 400)
        rng = StableRNG(27)
        cov_t, cov_tf, Fs = 0, 0, Float64[]
        for _ in 1:R
            df = iv_linear_dgp(rng; n=250, pi=0.18, beta=1.0, rho=0.9, hetero=true)
            r = late_2sls(df, :y, :d, :z1; covariates=[:x])
            ci = confint(r)
            cov_t += ci[1, 1] <= 1.0 <= ci[1, 2]
            t = tf_confint(r)
            cov_tf += t.lower <= 1.0 <= t.upper
            push!(Fs, r.first_stage.first_stage[1].F)
        end
        @test 3 < median(Fs) < 20                       # the design is weak-ish
        @test cov_tf / R >= 0.95 - mc_tol(0.95, R; slack=0.0)
        @test cov_tf / R > cov_t / R                     # t-ratio CI under-covers
    end
end
