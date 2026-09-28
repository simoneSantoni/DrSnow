using DrSnow: FixedEffectModels

@testset "iv_regression / late_2sls" begin
    @testset "truth recovery with endogeneity; OLS is biased" begin
        df = iv_linear_dgp(StableRNG(1); n=20_000, pi=0.8, beta=1.5, rho=0.7)
        r = late_2sls(df, :y, :d, :z1; covariates=[:x])
        @test r isa IVEstimate
        @test r isa CausalEstimate
        @test coefnames(r)[1] == "d"
        @test abs(estimate(r) - 1.5) < 4 * stderror(r)[1]
        @test stderror(r)[1] < 0.03
        ols = FixedEffectModels.reg(df, make_formula(:y, [:d, :x]))
        @test coef(ols)[2] - 1.5 > 0.3              # OLS bias from corr(u, v) > 0
        @test nobs(r) == 20_000
        @test size(vcov(r)) == (3, 3)
        @test issymmetric(vcov(r))
    end

    @testset "matches FixedEffectModels for every covariance type" begin
        df = iv_linear_dgp(StableRNG(2); n=800, k=2, hetero=true, G=30)
        df.h = rand(StableRNG(3), 1:6, 800)
        df.w = 0.5 .+ rand(StableRNG(4), 800)
        for (kw, vce) in (((vcov=Vcov.simple(),), Vcov.simple()),
                          (NamedTuple(), Vcov.robust()),
                          ((cluster=:g,), Vcov.cluster(:g)),
                          ((cluster=[:g, :h],), Vcov.cluster(:g, :h)))
            r = iv_regression(df, :y, :d, [:z1, :z2]; covariates=[:x], fe=[:h],
                              weights=:w, kw...)
            m = FixedEffectModels.reg(df, make_formula(:y, [:x]; fe=[:h],
                                                       endogenous=[:d],
                                                       instruments=[:z1, :z2]),
                                      vce; weights=:w)
            i = findfirst(==("d"), coefnames(m))
            @test coef(r)[1] ≈ coef(m)[i] rtol = 1e-8
            @test stderror(r)[1] ≈ stderror(m)[i] rtol = 1e-6
            @test dof_residual(r) == dof_residual(m)
            # the internal partialled design reproduces the same covariance
            des = r.design
            β, V, _, _ = DrSnow._iv_tsls(des, des.y, des.D, des.Z)
            @test β[1] ≈ coef(r)[1] rtol = 1e-7
            @test sqrt(V[1, 1]) ≈ stderror(r)[1] rtol = 1e-6
        end
    end

    @testset "covariate absorbed by the fixed effects (small sample)" begin
        df = iv_linear_dgp(StableRNG(16); n=60, G=6)
        df.gx = sqrt.(Float64.(df.g))                 # constant within g
        r = late_2sls(df, :y, :d, :z1; covariates=[:x, :gx], fe=[:g], vcov=Vcov.simple())
        m = FixedEffectModels.reg(df, make_formula(:y, [:x, :gx]; fe=[:g],
                                                   endogenous=[:d], instruments=[:z1]),
                                  Vcov.simple())
        i = findfirst(==("d"), coefnames(m))
        @test stderror(r)[1] ≈ stderror(m)[i] rtol = 1e-8
        @test r.design.k_exog == 1                    # only x survives
        @test dof_residual(r) == dof_residual(m)
        @test_throws ArgumentError late_2sls(df, :y, :d, :gx; fe=[:g])
    end

    @testset "row-shuffling invariance" begin
        df = iv_linear_dgp(StableRNG(5); n=600, k=2, G=25)
        r1 = iv_regression(df, :y, :d, [:z1, :z2]; covariates=[:x], cluster=:g)
        perm = randperm(StableRNG(6), nrow(df))
        r2 = iv_regression(df[perm, :], :y, :d, [:z1, :z2]; covariates=[:x], cluster=:g)
        @test coef(r1) ≈ coef(r2) rtol = 1e-10
        @test vcov(r1) ≈ vcov(r2) rtol = 1e-8
        @test r1.first_stage.effective_F ≈ r2.first_stage.effective_F rtol = 1e-8
        s1 = weak_iv_confidence_set(r1)
        s2 = weak_iv_confidence_set(r2)
        @test s1.intervals[1][1] ≈ s2.intervals[1][1] rtol = 1e-7
    end

    @testset "missing values and categorical covariates" begin
        df = iv_linear_dgp(StableRNG(7); n=500)
        df.cat = rand(StableRNG(8), ["a", "b", "c"], 500)
        dfm = allowmissing(df)
        dfm.y[3] = missing
        dfm.x[10] = missing
        r = late_2sls(dfm, :y, :d, :z1; covariates=[:x, :cat])
        @test nobs(r) == 498
        @test length(coef(r)) == 5          # d, intercept, x, cat: b, c
    end

    @testset "estimand labels" begin
        df, _ = iv_binary_dgp(StableRNG(9); n=1000)
        df.cell = rand(StableRNG(10), 1:5, 1000)
        @test estimand(late_2sls(df, :y, :d, :z)) == "LATE"
        @test occursin("cell LATEs", estimand(late_2sls(df, :y, :d, :z; fe=[:cell])))
        @test occursin("covariate-adjusted",
                       estimand(late_2sls(df, :y, :d, :z; covariates=[:x])))
        @test occursin("negative", late_2sls(df, :y, :d, :z; covariates=[:x]).estimand_note)
        @test occursin("ACR", estimand(late_2sls(df, :y, :x, :z)))
        # binary instrument, multi-valued treatment, with controls (Card-type design):
        # an ACR, never described as a multi-valued-instrument estimand
        df.dm = df.d .+ rand(StableRNG(13), 0:3, 1000)
        rs = late_2sls(df, :y, :dm, :z; fe=[:cell])
        @test occursin("cell ACRs", estimand(rs))
        @test occursin("same sign", rs.estimand_note)
        rc = late_2sls(df, :y, :dm, :z; covariates=[:x])
        @test occursin("conditional ACRs", estimand(rc))
        @test occursin("negative", rc.estimand_note)
        @test !occursin("multi-valued instrument", rc.estimand_note)
        @test occursin("weighted average of LATEs", estimand(late_2sls(df, :y, :d, :pre)))
        df2 = iv_linear_dgp(StableRNG(11); n=400, k=3)
        df2.d2 = df2.z2 .+ randn(StableRNG(12), 400)
        r2 = iv_regression(df2, :y, [:d, :d2], [:z1, :z2, :z3])
        @test occursin("structural", estimand(r2))
    end

    @testset "confint, level and display" begin
        df = iv_linear_dgp(StableRNG(13); n=300)
        r = late_2sls(df, :y, :d, :z1; level=0.9)
        ci = confint(r)
        c = critical_value(0.9, dof_residual(r))
        @test ci[1, 2] - ci[1, 1] ≈ 2c * stderror(r)[1]
        @test confint(r; level=0.95)[1, 2] > ci[1, 2]
        s = sprint(show, MIME"text/plain"(), r)
        @test occursin("2SLS", s) && occursin("effective F", s)
        @test !occursin("CoefTable(", s)
        @test occursin("2SLS", sprint(show, r))
    end

    @testset "input validation" begin
        df = iv_linear_dgp(StableRNG(14); n=200, k=1)
        @test_throws ArgumentError late_2sls(df, :y, :d, :nope)
        @test_throws ArgumentError iv_regression(df, :y, [:d, :x], [:z1])
        @test_throws ArgumentError late_2sls(df, :y, :d, :d)
        @test_throws ArgumentError late_2sls(df, :y, :d, :z1; covariates=[:z1])
        @test_throws ArgumentError late_2sls(df, :y, :d, :z1; level=1.5)
        @test_throws ArgumentError late_2sls(df, :y, :d, :z1; vcov="robust")
        df.s = string.(df.z1)
        @test_throws ArgumentError late_2sls(df, :y, :d, :s)
        # an instrument collinear with the controls
        df.zc = 2 .* df.x
        @test_throws ArgumentError late_2sls(df, :y, :d, :zc; covariates=[:x])
        @test_throws ArgumentError late_2sls(df, :y, :d, :z1; cluster=:nope)
        df.w = -ones(200)
        @test_throws ArgumentError late_2sls(df, :y, :d, :z1; weights=:w)
    end

    @testset "Monte Carlo: 2SLS CI coverage (strong instrument)" begin
        R = mc_reps(2000, 300)
        rng = StableRNG(15)
        cover = zeros(3)
        for _ in 1:R
            df = iv_linear_dgp(rng; n=400, pi=0.6, beta=1.0, rho=0.5, hetero=true,
                               G=40)
            r_h = late_2sls(df, :y, :d, :z1; covariates=[:x], vcov=Vcov.simple())
            r_r = late_2sls(df, :y, :d, :z1; covariates=[:x])
            r_c = late_2sls(df, :y, :d, :z1; covariates=[:x], cluster=:g)
            for (j, r) in enumerate((r_h, r_r, r_c))
                ci = confint(r)
                cover[j] += ci[1, 1] <= 1.0 <= ci[1, 2]
            end
        end
        cover ./= R
        # cluster-robust CI covers; the iid and HC1 CIs ignore the cluster shocks
        @test abs(cover[3] - 0.95) < mc_tol(0.95, R; slack=0.025)
        @test cover[1] < cover[3]
    end
end
