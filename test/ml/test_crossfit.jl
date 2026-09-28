@testset "Cross-fitting infrastructure" begin
    @testset "crossfit_folds" begin
        F = crossfit_folds(103, 5, 3; rng=StableRNG(1))
        @test size(F) == (103, 3)
        for r in 1:3
            c = [count(==(k), F[:, r]) for k in 1:5]
            @test sort(unique(F[:, r])) == 1:5 && maximum(c) - minimum(c) <= 1
        end
        @test F == crossfit_folds(103, 5, 3; rng=StableRNG(1))
        @test F[:, 1] != F[:, 2]
        # stratification balances each stratum across folds
        s = vcat(ones(Int, 40), zeros(Int, 60))
        Fs = crossfit_folds(100, 4, 1; rng=StableRNG(2), strata=s)
        @test all(count((Fs[:, 1] .== k) .& (s .== 1)) == 10 for k in 1:4)
        # cluster grouping: one fold per cluster, invariant to row order
        g = repeat(["c$(i)" for i in 1:20], inner=3)
        Fg = crossfit_folds(60, 5, 2; rng=StableRNG(3), groups=g)
        for r in 1:2, c in unique(g)
            @test length(unique(Fg[g .== c, r])) == 1
        end
        perm = randperm(StableRNG(4), 60)
        Fp = crossfit_folds(60, 5, 2; rng=StableRNG(3), groups=g[perm])
        @test Fp == Fg[perm, :]
        @test_throws ArgumentError crossfit_folds(10, 1)
        @test_throws ArgumentError crossfit_folds(10, 5, 0)
        @test_throws ArgumentError crossfit_folds(3, 5)
        @test_throws ArgumentError crossfit_folds(12, 5; groups=repeat(1:3, 4))
        @test_throws DimensionMismatch crossfit_folds(10, 2; strata=1:3)
    end

    @testset "fold validation" begin
        df = DataFrame(y=randn(StableRNG(5), 20), d=randn(StableRNG(6), 20),
                       x=randn(StableRNG(7), 20), g=repeat(1:5, inner=4),
                       f=repeat(1:2, 10))
        kw = (covariates=[:x], outcome_learner=OLSLearner(),
              treatment_learner=OLSLearner())
        @test_throws ArgumentError dml_plr(df, :y, :d; kw..., folds=fill(1, 20))
        @test_throws ArgumentError dml_plr(df, :y, :d; kw..., folds=repeat([1, 3], 10))
        @test_throws DimensionMismatch dml_plr(df, :y, :d; kw..., folds=[1, 2])
        # folds that split a cluster are rejected
        @test_throws ArgumentError dml_plr(df, :y, :d; kw..., folds=:f, cluster=:g)
        @test dml_plr(df, :y, :d; kw..., folds=:f) isa DMLEstimate
        @test_throws ArgumentError dml_plr(df, :y, :d; kw..., cluster=[:g, :f])
    end

    @testset "linear-score algebra" begin
        rng = StableRNG(8)
        pa = -1 .- rand(rng, 50)
        pb = randn(rng, 50)
        θ, ψ, v = DrSnow._ml_solve_score(pa, pb, nothing, 0)
        @test θ ≈ -sum(pb) / sum(pa)
        @test sum(ψ) ≈ 0 atol = 1e-12
        @test v ≈ mean(ψ .^ 2) / mean(pa)^2 / 50
        cl = repeat(1:10, inner=5)
        _, _, vc = DrSnow._ml_solve_score(pa, pb, cl, 10)
        S = [sum(ψ[cl .== g]) for g in 1:10]
        @test vc ≈ 10 / 9 * sum(S .^ 2) / sum(pa)^2
        # median aggregation over repetitions
        ac = [1.0 2.0 4.0]
        V = [fill(0.01, 1, 1), fill(0.04, 1, 1), fill(0.09, 1, 1)]
        θa, Va = DrSnow._ml_aggregate(ac, V)
        @test θa == [2.0]
        @test Va[1, 1] ≈ median([0.01 + 1.0, 0.04, 0.09 + 4.0])
    end

    @testset "propensity clipping" begin
        m = [0.001, 0.5, 0.999, 0.02]
        @test DrSnow._ml_clip!(m, 0.01) == 2
        @test m == [0.01, 0.5, 0.99, 0.02]
        @test DrSnow._ml_clip!([0.0, 1.0], 0) == 0
    end
end
