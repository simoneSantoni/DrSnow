struct MLTestHalfLearner <: NuisanceLearner end
DrSnow.fitpredict(::MLTestHalfLearner, X, y, Xnew; kwargs...) = fill(0.5, size(Xnew, 1))

@testset "Nuisance learners" begin
    rng = StableRNG(101)
    n, p = 300, 4
    X = randn(rng, n, p)
    β = [1.0, -0.5, 0.0, 0.25]
    y = X * β .+ 0.3 .+ randn(rng, n)
    Xnew = randn(rng, 7, p)

    @testset "OLS" begin
        Z = hcat(ones(n), X)
        b = Z \ y
        @test fitpredict(OLSLearner(), X, y, Xnew) ≈ hcat(ones(7), Xnew) * b
        @test fitpredict(OLSLearner(intercept=false), X, y, Xnew) ≈ Xnew * (X \ y)
        # integer weights equal duplicated rows
        w = rand(StableRNG(2), 1:3, n)
        idx = reduce(vcat, [fill(i, w[i]) for i in 1:n])
        @test fitpredict(OLSLearner(), X, y, Xnew; weights=w) ≈
              fitpredict(OLSLearner(), X[idx, :], y[idx], Xnew)
        # rank-deficient design still predicts the fitted values
        Xc = hcat(X, X[:, 1] .+ X[:, 2])
        @test fitpredict(OLSLearner(), Xc, y, Xc) ≈ fitpredict(OLSLearner(), X, y, X)
        # no covariates: the mean
        @test fitpredict(OLSLearner(), zeros(n, 0), y, zeros(3, 0)) ≈ fill(mean(y), 3)
        @test_throws DimensionMismatch fitpredict(OLSLearner(), X, y[1:10], Xnew)
        @test_throws DimensionMismatch fitpredict(OLSLearner(), X, y, Xnew[:, 1:2])
        @test_throws ArgumentError fitpredict_proba(OLSLearner(), X, y, Xnew)
    end

    @testset "Ridge" begin
        λ = 0.3
        μ = vec(mean(X; dims=1))
        σ = vec(std(X; corrected=false, dims=1))
        Zs = (X .- μ') ./ σ'
        βs = (Zs' * Zs ./ n + λ * I) \ (Zs' * (y .- mean(y)) ./ n)
        pred = mean(y) .+ ((Xnew .- μ') ./ σ') * βs
        @test fitpredict(RidgeLearner(lambda=λ), X, y, Xnew) ≈ pred
        @test fitpredict(RidgeLearner(lambda=0.0), X, y, Xnew) ≈
              fitpredict(OLSLearner(), X, y, Xnew)
        # LOOCV choice: exact leave-one-out on the standardized design
        lams = [1e-3, 0.1, 10.0]
        f = fitpredict(RidgeLearner(lambda=:loocv, lambdas=lams), X, y, Xnew)
        cvs = map(lams) do l
            H = Zs * ((Zs' * Zs ./ n + l * I) \ Zs') ./ n .+ 1 / n
            e = (y .- mean(y)) .- H * (y .- mean(y))
            mean((e ./ (1 .- diag(H))) .^ 2)
        end
        @test f ≈ fitpredict(RidgeLearner(lambda=lams[argmin(cvs)]), X, y, Xnew)
        @test_throws ArgumentError fitpredict(RidgeLearner(lambda=-1.0), X, y, Xnew)
    end

    @testset "Lasso / elastic net (KKT conditions)" begin
        w = fill(1 / n, n)
        μ = vec(mean(X; dims=1))
        σ = vec(std(X; corrected=false, dims=1))
        Zs = (X .- μ') ./ σ'
        for (α, λ) in ((1.0, 0.1), (0.5, 0.2), (1.0, 0.02))
            fit = DrSnow._ml_fit(LassoLearner(lambda=λ, alpha=α), X, y)
            βs = fit.β .* σ
            r = (y .- mean(y)) .- Zs * βs
            g = Zs' * (w .* r) .- λ * (1 - α) .* βs
            for j in 1:p
                if βs[j] != 0
                    @test g[j] ≈ λ * α * sign(βs[j]) atol = 1e-6
                else
                    @test abs(g[j]) <= λ * α + 1e-8
                end
            end
            @test fit.b0 ≈ mean(y) - dot(μ, fit.β)
        end
        # large λ: all coefficients zero → predicts the mean
        @test fitpredict(LassoLearner(lambda=100.0), X, y, Xnew) ≈ fill(mean(y), 7)
        # λ = 0 reproduces OLS
        @test fitpredict(LassoLearner(lambda=0.0), X, y, Xnew) ≈
              fitpredict(OLSLearner(), X, y, Xnew) rtol = 1e-5
        # cross-validated λ is reproducible given the task RNG
        a = fitpredict(LassoLearner(), X, y, Xnew; rng=StableRNG(5))
        b = fitpredict(LassoLearner(), X, y, Xnew; rng=StableRNG(5))
        @test a == b
        @test fitpredict(LassoLearner(rule=:one_se), X, y, Xnew; rng=StableRNG(5)) isa
              Vector{Float64}
        # sparse truth is recovered
        fit = DrSnow._ml_fit(LassoLearner(), X, y; rng=StableRNG(3))
        @test abs(fit.β[1] - 1.0) < 0.15 && abs(fit.β[3]) < 0.1
        @test_throws ArgumentError fitpredict(LassoLearner(alpha=2.0), X, y, Xnew)
        @test_throws ArgumentError fitpredict(LassoLearner(lambda=:bic), X, y, Xnew)
        # constant column is ignored
        Xk = hcat(X, ones(n))
        @test fitpredict(LassoLearner(lambda=0.05), Xk, y, hcat(Xnew, ones(7))) ≈
              fitpredict(LassoLearner(lambda=0.05), X, y, Xnew)
    end

    @testset "Logistic learners" begin
        prob = 1 ./ (1 .+ exp.(-(0.3 .+ X * [1.0, -1.0, 0.0, 0.5])))
        d = Float64.(rand(StableRNG(7), n) .< prob)
        m = DrSnow.GLM.glm(hcat(ones(n), X), d, DrSnow.Binomial())
        ref = DrSnow.GLM.predict(m, hcat(ones(7), Xnew))
        @test fitpredict_proba(LogisticLearner(), X, d, Xnew) ≈ ref rtol = 1e-6
        @test fitpredict(LogisticLearner(), X, d, Xnew) ≈ ref rtol = 1e-6
        # weighted fit (Newton) equals the unweighted fit on duplicated rows (GLM)
        w = rand(StableRNG(8), 1:3, n)
        idx = reduce(vcat, [fill(i, w[i]) for i in 1:n])
        @test fitpredict_proba(LogisticLearner(), X, d, Xnew; weights=w) ≈
              fitpredict_proba(LogisticLearner(), X[idx, :], d[idx], Xnew) rtol = 1e-7
        @test_throws ArgumentError fitpredict_proba(LogisticLearner(), X, y, Xnew)
        @test_throws ArgumentError fitpredict_proba(LogisticLearner(), X, zeros(n), Xnew)
        # penalized logistic: tiny λ ≈ MLE; KKT at a fixed λ
        @test fitpredict_proba(PenalizedLogisticLearner(lambda=1e-7), X, d, Xnew) ≈ ref rtol =
            1e-3
        μ = vec(mean(X; dims=1))
        σ = vec(std(X; corrected=false, dims=1))
        Zs = (X .- μ') ./ σ'
        for (α, λ) in ((1.0, 0.02), (0.0, 0.05))
            fit = DrSnow._ml_fit(PenalizedLogisticLearner(lambda=λ, alpha=α), X, d)
            pr = DrSnow._ml_sigmoid.(X * fit.β .+ fit.b0)
            βs = fit.β .* σ
            g = Zs' * ((d .- pr) ./ n) .- λ * (1 - α) .* βs
            @test abs(mean(d .- pr)) < 1e-6
            for j in 1:p
                if βs[j] != 0
                    @test g[j] ≈ λ * α * sign(βs[j]) atol = 1e-5
                else
                    @test abs(g[j]) <= λ * α + 1e-6
                end
            end
        end
        pcv = fitpredict_proba(PenalizedLogisticLearner(), X, d, Xnew; rng=StableRNG(1))
        @test all(0 .< pcv .< 1)
        @test pcv == fitpredict_proba(PenalizedLogisticLearner(), X, d, Xnew;
                                      rng=StableRNG(1))
    end

    @testset "kNN and mean" begin
        @test fitpredict(KNNLearner(k=1), X, y, X) ≈ y
        @test fitpredict(KNNLearner(k=n), X, y, Xnew) ≈ fill(mean(y), 7)
        d = Float64.(y .> 0)
        pk = fitpredict_proba(KNNLearner(k=15), X, d, Xnew)
        @test all(0 .<= pk .<= 1)
        @test_throws ArgumentError fitpredict(KNNLearner(k=0), X, y, Xnew)
        @test fitpredict(MeanLearner(), X, y, Xnew) ≈ fill(mean(y), 7)
        @test fitpredict(MeanLearner(), X, y, Xnew; weights=1:n) ≈
              fill(sum((1:n) .* y) / sum(1:n), 7)
        @test fitpredict_proba(MeanLearner(), X, d, Xnew) ≈ fill(mean(d), 7)
    end

    @testset "Custom and MLJ stub" begin
        @test fitpredict(MLTestHalfLearner(), X, y, Xnew) == fill(0.5, 7)
        @test_throws ArgumentError fitpredict_proba(MLTestHalfLearner(), X, y, Xnew)
        # an object that is not an MLJ model always reaches the stub
        @test_throws ArgumentError fitpredict(MLJLearner(:not_a_model), X, y, Xnew)
        @test_throws ArgumentError fitpredict_proba(MLJLearner(:not_a_model), X, y, Xnew)
    end
end
