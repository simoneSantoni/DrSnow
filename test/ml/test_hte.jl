function ml_rct_data(rng, n; hetero=true, p=0.5)
    X = randn(rng, n, 3)
    d = Float64.(rand(rng, n) .< p)
    τ = hetero ? 1.0 .+ X[:, 1] : fill(1.0, n)
    y = d .* τ .+ X[:, 2] .+ 0.5 .* X[:, 3] .+ randn(rng, n)
    df = DataFrame(X, [:x1, :x2, :x3])
    df.d = d
    df.y = y
    return df
end

@testset "Heterogeneous effects" begin
    @testset "DR-learner" begin
        df = ml_irm_data(StableRNG(31), 1500)
        kw = (covariates=[:x1, :x2, :x3], outcome_learner=OLSLearner(),
              propensity_learner=LogisticLearner())
        c = cate_dr_learner(df, :y, :d; kw..., effect_modifiers=[:x2],
                            cate_learner=OLSLearner(), rng=StableRNG(1))
        @test c isa CATEPredictor
        @test length(c.pseudo_outcomes) == 1500 && length(c.cate_oof) == 1500
        # the ATE is the AIPW (DML-IRM) estimate with the same folds and learners
        r = dml_irm(df, :y, :d; kw..., folds=c.folds, trim=0.01)
        @test c.ate ≈ coef(r)[1] rtol = 1e-10
        @test c.ate_se ≈ stderror(r)[1] rtol = 1e-10
        # τ(x) = 1 + 0.5 x2: projection on x2 recovers the slope
        pr = cate_projection(c)
        @test coefnames(pr) == ["(Intercept)", "x2"]
        @test abs(coef(pr)[2] - 0.5) < 4 * stderror(pr)[2]
        @test dof_residual(pr) == 1498
        p0 = cate_projection(c; basis=Symbol[])
        @test coef(p0)[1] ≈ c.ate
        @test stderror(p0)[1] ≈ c.ate_se rtol = 1e-3
        # OLS second stage: the predictor is the linear projection
        @test predict(c, DataFrame(x2=[0.0, 1.0])) ≈ [coef(pr)[1], sum(coef(pr))]
        @test predict(c, [0.0; 1.0;;]) ≈ predict(c, DataFrame(x2=[0.0, 1.0]))
        @test_throws ArgumentError predict(c, DataFrame(x1=[0.0]))
        @test_throws DimensionMismatch predict(c, zeros(2, 2))
        @test_throws ArgumentError cate_projection(c; basis=[:x1])
        c2 = cate_dr_learner(df, :y, :d; kw..., effect_modifiers=[:x2],
                             cate_learner=OLSLearner(), rng=StableRNG(1))
        @test c2.cate_oof == c.cate_oof
        @test occursin("ATE (AIPW)", sprint(show, MIME"text/plain"(), c))
        # nonparametric second stage runs and gives sensible predictions
        ck = cate_dr_learner(df, :y, :d; kw..., effect_modifiers=[:x2],
                             cate_learner=KNNLearner(k=100), rng=StableRNG(2))
        pk = predict(ck, DataFrame(x2=[-1.0, 1.0]))
        @test pk[2] > pk[1]
        @test_throws ArgumentError cate_dr_learner(df, :y, :x1; covariates=[:x2])

        # Monte Carlo: coverage of the projection coefficients
        reps = mc_reps(400, 60)
        cover = zeros(2)
        for rep in 1:reps
            rg = StableRNG(3000 + rep)
            dr = ml_irm_data(rg, 800)
            cr = cate_dr_learner(dr, :y, :d; kw..., effect_modifiers=[:x2],
                                 cate_learner=OLSLearner(), rng=rg)
            ci = confint(cate_projection(cr))
            cover .+= (ci[:, 1] .<= [1.0, 0.5] .<= ci[:, 2])
        end
        @info "Monte Carlo CATE projection coverage" cover ./ reps
        @test all(abs.(cover ./ reps .- 0.95) .<= ml_cover_tol(reps))
    end

    @testset "generic ML (BLP / GATES / CLAN)" begin
        # the weighted least-squares sandwich equals FixedEffectModels
        rng = StableRNG(41)
        n = 200
        dfr = DataFrame(x=randn(rng, n), z=randn(rng, n), w=rand(rng, n) .+ 0.5,
                        g=rand(rng, 1:15, n))
        dfr.y = 1 .+ dfr.x .+ randn(rng, n)
        Z = hcat(ones(n), dfr.x, dfr.z)
        f = make_formula(:y, [:x, :z])
        b, V = DrSnow._ml_wls_sandwich(Z, dfr.y, dfr.w, nothing, 0)
        m = DrSnow.FixedEffectModels.reg(dfr, f, Vcov.robust(); weights=:w)
        @test b ≈ coef(m) && V ≈ vcov(m)
        gi, G = DrSnow._ml_group_index(dfr.g)
        _, Vc = DrSnow._ml_wls_sandwich(Z, dfr.y, dfr.w, gi, G)
        mc = DrSnow.FixedEffectModels.reg(dfr, f, Vcov.cluster(:g); weights=:w)
        @test Vc ≈ vcov(mc)

        df = ml_rct_data(StableRNG(42), 1000)
        g = generic_ml(df, :y, :d; covariates=[:x1, :x2, :x3], proxy_learner=OLSLearner(),
                       n_splits=20, n_groups=4, rng=StableRNG(1))
        @test g isa GenericMLInference
        @test size(blp(g), 1) == 2 && size(gates(g), 1) == 5 && size(clan(g), 1) == 3
        t = blp_test(g)
        @test t isa DiagnosticTest && rejects(t)
        @test gates(g).estimate[end] > 0 && gates(g).pvalue[end] < 0.01
        @test clan(g).covariate[1] == "x1" && clan(g).difference[1] > 0
        @test g.lambda > 0
        @test occursin("Sorted group", sprint(show, MIME"text/plain"(), g))
        g2 = generic_ml(df, :y, :d; covariates=[:x1, :x2, :x3],
                        proxy_learner=OLSLearner(), n_splits=20, n_groups=4,
                        rng=StableRNG(1), parallel=false)
        @test blp(g2) == blp(g) && gates(g2) == gates(g)
        # known propensity column / constant, and clusters
        df.p = fill(0.5, nrow(df))
        gp = generic_ml(df, :y, :d; covariates=[:x1, :x2, :x3], propensity=:p,
                        proxy_learner=OLSLearner(), n_splits=5, rng=StableRNG(2))
        @test isfinite(blp(gp).estimate[1])
        df.cl = repeat(1:100, inner=10)
        gc = generic_ml(df, :y, :d; covariates=[:x1, :x2, :x3], cluster=:cl,
                        proxy_learner=OLSLearner(), n_splits=5, rng=StableRNG(3))
        @test isfinite(blp(gc).estimate[2])
        @test_throws ArgumentError generic_ml(df, :y, :d; covariates=[:x1], n_groups=1)
        @test_throws ArgumentError generic_ml(df, :y, :d; covariates=[:x1], propensity=1.0)
        @test_throws ArgumentError generic_ml(df, :y, :d; covariates=[:x1], level=1.0)
        # a constant proxy is jittered and reported
        gm = generic_ml(df, :y, :d; covariates=[:x1], proxy_learner=MeanLearner(),
                        n_splits=3, rng=StableRNG(4))
        @test gm.n_degenerate == 3

        # Monte Carlo: size of the heterogeneity test, ATE coverage, power
        reps = mc_reps(200, 40)
        rej0 = 0
        rej1 = 0
        cov_ate = 0
        for rep in 1:reps
            rg = StableRNG(5000 + rep)
            d0 = ml_rct_data(rg, 600; hetero=false)
            g0 = generic_ml(d0, :y, :d; covariates=[:x1, :x2, :x3],
                            proxy_learner=OLSLearner(), n_splits=11, rng=rg)
            rej0 += blp_test(g0).pvalue < 0.05
            b0 = blp(g0)
            cov_ate += b0.lower[1] <= 1.0 <= b0.upper[1]
            d1 = ml_rct_data(rg, 600)
            g1 = generic_ml(d1, :y, :d; covariates=[:x1, :x2, :x3],
                            proxy_learner=OLSLearner(), n_splits=11, rng=rg)
            rej1 += blp_test(g1).pvalue < 0.05
        end
        @info "Monte Carlo generic ML" size = rej0 / reps power = rej1 / reps ate_coverage =
            cov_ate / reps
        @test rej0 / reps <= 0.05 + ml_cover_tol(reps)
        @test cov_ate / reps >= 0.95 - ml_cover_tol(reps)
        @test rej1 / reps >= 0.8
    end
end
