# Meta-learners (S, T, X, R) and classical subgroup / interaction analyses.

function meta_data(rng, n; confounded=true)
    X = randn(rng, n, 3)
    e = confounded ? 1 ./ (1 .+ exp.(-0.5 .* X[:, 2])) : fill(0.5, n)
    d = Float64.(rand(rng, n) .< e)
    τ = 1 .+ X[:, 1]
    y = X[:, 2] .+ 0.5 .* X[:, 3] .+ τ .* d .+ randn(rng, n)
    df = DataFrame(X, [:x1, :x2, :x3])
    df.d = d
    df.y = y
    df.tau = τ
    return df
end

@testset "Meta-learners" begin
    xs = [:x1, :x2, :x3]
    df = meta_data(StableRNG(1), 1500)

    @testset "closed forms with linear learners" begin
        # T-learner with OLS: difference of the two arm regressions
        t = t_learner(df, :y, :d; covariates=xs, outcome_learner=OLSLearner(),
                      rng=StableRNG(2))
        X = Matrix(df[:, xs])
        tr = df.d .== 1
        b1 = hcat(ones(count(tr)), X[tr, :]) \ df.y[tr]
        b0 = hcat(ones(count(.!tr)), X[.!tr, :]) \ df.y[.!tr]
        @test t.cate ≈ hcat(ones(nrow(df)), X) * (b1 .- b0)
        @test t.ate ≈ mean(t.cate)
        @test predict(t, df[1:4, :]) ≈ t.cate[1:4]
        # S-learner with an additive OLS model: constant effect equal to the coefficient
        s = s_learner(df, :y, :d; covariates=xs, outcome_learner=OLSLearner(),
                      rng=StableRNG(3))
        bs = hcat(ones(nrow(df)), X, df.d) \ df.y
        @test s.cate ≈ fill(bs[end], nrow(df))
        # X-learner with OLS everywhere and a constant propensity: convex combination
        x = x_learner(df, :y, :d; covariates=xs, outcome_learner=OLSLearner(),
                      propensity_learner=MeanLearner(), rng=StableRNG(4))
        g = mean(df.d)
        D1 = df.y[tr] .- hcat(ones(count(tr)), X[tr, :]) * b0
        D0 = hcat(ones(count(.!tr)), X[.!tr, :]) * b1 .- df.y[.!tr]
        τ1 = hcat(ones(nrow(df)), X) * (hcat(ones(count(tr)), X[tr, :]) \ D1)
        τ0 = hcat(ones(nrow(df)), X) * (hcat(ones(count(.!tr)), X[.!tr, :]) \ D0)
        @test x.cate ≈ g .* τ0 .+ (1 - g) .* τ1
        # R-learner with OLS final stage: weighted least squares of the pseudo-outcome
        r = r_learner(df, :y, :d; covariates=xs, effect_modifiers=[:x1],
                      outcome_learner=OLSLearner(), propensity_learner=LogisticLearner(),
                      final_learner=OLSLearner(), rng=StableRNG(5))
        res_w = df.d .- r.nuisance.e
        res_y = df.y .- r.nuisance.m
        Z = hcat(ones(nrow(df)), df.x1)
        β = (Z .* res_w) \ res_y
        @test r.cate ≈ Z * β
        @test abs(β[2] - 1) < 0.15 && abs(β[1] - 1) < 0.15
        @test predict(r, DataFrame(x1=[0.0, 1.0])) ≈ [β[1], β[1] + β[2]]
        @test predict(r, [0.0; 1.0;;]) ≈ [β[1], β[1] + β[2]]
    end

    @testset "flexible learners, cross-fitting and reproducibility" begin
        for f in (s_learner, t_learner, x_learner, r_learner)
            m = f(df, :y, :d; covariates=xs, rng=StableRNG(6))
            @test m isa MetaLearner
            @test cor(m.cate_oof, df.tau) > 0.8
            @test abs(m.ate - mean(df.tau)) < 0.3
            m2 = f(df, :y, :d; covariates=xs, rng=StableRNG(6), parallel=false)
            @test m2.cate_oof == m.cate_oof && m2.cate == m.cate
            @test occursin("No standard errors", sprint(show, MIME"text/plain"(), m))
        end
        m = t_learner(df, :y, :d; covariates=xs, rng=StableRNG(7))
        @test predict(m, df[1:5, :]) == predict(m, df[1:5, :])
        @test_throws ArgumentError predict(m, DataFrame(x1=[0.0]))
        @test_throws DimensionMismatch predict(m, zeros(2, 2))
        @test_throws ArgumentError t_learner(df, :y, :x1; covariates=xs)
        @test_throws ArgumentError r_learner(df, :y, :d; covariates=xs,
                                             effect_modifiers=Symbol[])
        @test_throws ArgumentError s_learner(df, :y, :d; covariates=Symbol[])
        # clusters: folds keep clusters together
        df.cl = repeat(1:150, inner=10)
        mc = t_learner(df, :y, :d; covariates=xs, outcome_learner=OLSLearner(),
                       cluster=:cl, rng=StableRNG(8))
        @test all(length(unique(mc.folds[df.cl .== g])) == 1 for g in 1:150)
    end

    @testset "bootstrap" begin
        m = t_learner(df, :y, :d; covariates=xs, outcome_learner=OLSLearner(),
                      rng=StableRNG(9))
        nd = DataFrame(x1=[-1.0, 1.0], x2=0.0, x3=0.0)
        b = metalearner_bootstrap(m; newdata=nd, B=60, rng=StableRNG(10))
        @test b isa HTEEstimate && coefnames(b) == ["ATE (plug-in)", "CATE[1]", "CATE[2]"]
        @test coef(b)[2:3] ≈ predict(m, nd)
        @test size(b.details.percentile) == (3, 2)
        b2 = metalearner_bootstrap(m; newdata=nd, B=60, rng=StableRNG(10), parallel=false)
        @test vcov(b2) == vcov(b)
        br = metalearner_bootstrap(r_learner(df, :y, :d; covariates=xs,
                                             effect_modifiers=[:x1],
                                             outcome_learner=OLSLearner(),
                                             propensity_learner=LogisticLearner(),
                                             final_learner=OLSLearner(),
                                             rng=StableRNG(11));
                                   newdata=DataFrame(x1=[0.0]), B=20, rng=StableRNG(12))
        @test length(coef(br)) == 2
        @test_throws ArgumentError metalearner_bootstrap(m; B=1)
        # Monte Carlo: bootstrap coverage of the ATE with OLS arms (valid here)
        reps = mc_reps(200, 15)
        cover = 0
        for rep in 1:reps
            rg = StableRNG(4000 + rep)
            dr = meta_data(rg, 400; confounded=false)
            mr = t_learner(dr, :y, :d; covariates=xs, outcome_learner=OLSLearner(),
                           rng=rg)
            ci = confint(metalearner_bootstrap(mr; B=mc_reps(200, 60), rng=rg))
            cover += ci[1, 1] <= 1.0 <= ci[1, 2]
        end
        @info "Monte Carlo T-learner bootstrap ATE coverage" cover / reps
        @test abs(cover / reps - 0.95) <= ml_cover_tol(reps)
    end
end

@testset "Subgroup and interaction effects" begin
    df = meta_data(StableRNG(20), 1200; confounded=false)
    df.g = ifelse.(df.x1 .> 0, "pos", "neg")
    df.cl = repeat(1:120, inner=10)
    xs = [:x1, :x2, :x3]

    @testset "regression methods equal FixedEffectModels" begin
        r = subgroup_effects(df, :y, :d, :g; cluster=:cl)
        @test coefnames(r) == ["g = neg", "g = pos"]
        dd = copy(df)
        dd.dneg = dd.d .* (dd.g .== "neg")
        dd.dpos = dd.d .* (dd.g .== "pos")
        m = DrSnow.FixedEffectModels.reg(dd, make_formula(:y, [:dneg, :dpos]; fe=[:g]),
                                         Vcov.cluster(:cl))
        @test coef(r) ≈ coef(m) && vcov(r) ≈ vcov(m)
        @test dof_residual(r) == 119
        h = heterogeneity_test(r)
        @test h isa DiagnosticTest && h.dof == (1, 119.0)
        @test h.statistic ≈ (coef(r)[1] - coef(r)[2])^2 /
                            (vcov(r)[1, 1] + vcov(r)[2, 2] - 2vcov(r)[1, 2])
        @test coef(r)[2] > coef(r)[1]
        @test r.details.p_adjusted ≈ holm_adjust(pvalues(r))
        rb = subgroup_effects(df, :y, :d, :g; adjust=:bh, covariates=[:x2])
        @test rb.details.p_adjusted ≈ bh_adjust(pvalues(rb))
        @test_throws ArgumentError subgroup_effects(df, :y, :d, :g; adjust=:nope)
        @test_throws ArgumentError subgroup_effects(df, :y, :d, :g; method=:nope)
        df.one = fill("a", nrow(df))
        @test_throws ArgumentError subgroup_effects(df, :y, :d, :one)
        ie = interaction_effects(df, :y, :d, [:x1]; covariates=[:x2], cluster=:cl)
        dd.x1c = dd.x1 .- mean(dd.x1)
        dd.dx1 = dd.d .* dd.x1c
        m2 = DrSnow.FixedEffectModels.reg(dd, make_formula(:y, [:d, :dx1, :x1c, :x2]),
                                          Vcov.cluster(:cl))
        @test coef(ie) ≈ coef(m2)[2:3] && vcov(ie) ≈ vcov(m2)[2:3, 2:3]
        @test coefnames(ie) == ["d", "d × x1"]
        @test abs(coef(ie)[2] - 1) < 4 * stderror(ie)[2]
        @test heterogeneity_test(ie).pvalue < 1e-6
        @test_throws ArgumentError heterogeneity_test(average_treatment_effect(
            causal_forest(df, :y, :d; covariates=xs, num_trees=20, rng=StableRNG(1))))
        @test_throws ArgumentError interaction_effects(df, :y, :d, Symbol[])
    end

    @testset "AIPW scores and causal forests" begin
        kw = (covariates=xs, outcome_learner=OLSLearner(),
              propensity_learner=LogisticLearner())
        r = subgroup_effects(df, :y, :d, :g; method=:aipw, kw..., rng=StableRNG(21))
        # each coefficient is the AIPW estimate on the subgroup (same folds and scores)
        c = cate_dr_learner(df, :y, :d; kw..., rng=StableRNG(21))
        φ = c.pseudo_outcomes
        @test coef(r) ≈ [mean(φ[df.g .== "neg"]), mean(φ[df.g .== "pos"])]
        ia = interaction_effects(df, :y, :d, [:x1]; method=:aipw, kw...,
                                 rng=StableRNG(21), center=false)
        pr = cate_projection(c; basis=[:x1])
        @test coef(ia) ≈ coef(pr)
        @test sqrt.(diag(vcov(ia))) ≈ stderror(pr) rtol = 1e-8
        cf = causal_forest(df, :y, :d; covariates=xs, num_trees=300, rng=StableRNG(22))
        rf = subgroup_effects(cf, df.g)
        Γ = get_scores(cf)
        @test coef(rf) ≈ [mean(Γ[df.g .== "neg"]), mean(Γ[df.g .== "pos"])]
        @test_throws DimensionMismatch subgroup_effects(cf, df.g[1:10])

        # Monte Carlo: coverage of the subgroup effects and size of the equality test
        reps = mc_reps(300, 30)
        cover = zeros(2)
        rej = 0
        for rep in 1:reps
            rg = StableRNG(6000 + rep)
            d0 = meta_data(rg, 600; confounded=true)
            d0.tau .= 1.0
            d0.y = d0.x2 .+ 0.5 .* d0.x3 .+ d0.d .+ randn(rg, 600)
            d0.g = ifelse.(d0.x3 .> 0, "a", "b")
            s = subgroup_effects(d0, :y, :d, :g; method=:aipw, kw..., rng=rg)
            ci = confint(s)
            cover .+= ci[:, 1] .<= 1.0 .<= ci[:, 2]
            rej += heterogeneity_test(s).pvalue < 0.05
        end
        @info "Monte Carlo AIPW subgroup effects" cover ./ reps size = rej / reps
        @test all(abs.(cover ./ reps .- 0.95) .<= ml_cover_tol(reps))
        @test rej / reps <= 0.05 + ml_cover_tol(reps)
    end
end
