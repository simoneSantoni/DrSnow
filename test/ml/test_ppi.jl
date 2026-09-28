function ml_ppi_data(rng, n, N; bias=0.3, noise=0.5)
    X = randn(rng, n + N, 2)
    y = 1 .+ 0.5 .* X[:, 1] .- 0.25 .* X[:, 2] .+ randn(rng, n + N)
    f = y .+ bias .+ 0.2 .* X[:, 1] .+ noise .* randn(rng, n + N)   # biased predictor
    yb = Float64.(y .> 1)
    fb = clamp.(0.7 .* yb .+ 0.15 .+ 0.1 .* X[:, 2] .+ 0.1 .* randn(rng, n + N), 0, 1)
    lab = DataFrame(x1=X[1:n, 1], x2=X[1:n, 2], y=y[1:n], f=f[1:n], yb=yb[1:n],
                    fb=fb[1:n])
    un = DataFrame(x1=X[(n + 1):end, 1], x2=X[(n + 1):end, 2], f=f[(n + 1):end],
                   fb=fb[(n + 1):end])
    return lab, un
end

@testset "Prediction-powered inference" begin
    lab, un = ml_ppi_data(StableRNG(81), 300, 3000)
    n, N = 300, 3000
    @testset "closed forms" begin
        # λ = 0: labeled-only estimates; λ = 1: original PPI
        r0 = ppi_mean(lab, un, :y, :f; lambda=0)
        @test coef(r0)[1] ≈ mean(lab.y)
        @test stderror(r0)[1] ≈ std(lab.y; corrected=false) / sqrt(n)
        r1 = ppi_mean(lab, un, :y, :f; lambda=1)
        @test coef(r1)[1] ≈ mean(lab.y) - mean(lab.f) + mean(un.f)
        @test vcov(r1)[1, 1] ≈ var(lab.y .- lab.f; corrected=false) / n +
                               var(un.f; corrected=false) / N
        r = ppi_mean(lab, un, :y, :f)
        fall = vcat(lab.f, un.f)
        λ = cov(lab.y, lab.f; corrected=false) / ((1 + n / N) * var(fall; corrected=false))
        @test r.lambda ≈ λ
        @test coef(r)[1] ≈ mean(lab.y) + λ * (mean(un.f) - mean(lab.f))
        @test stderror(r)[1] < stderror(r0)[1]
        @test ppi_mean(lab.y, lab.f, un.f).coef ≈ coef(r)
        # OLS with fixed λ: normal equations of the PPI++ objective
        rl = ppi_ols(lab, un, :y, :f; covariates=[:x1, :x2], lambda=0.6)
        X = hcat(ones(n), lab.x1, lab.x2)
        Xu = hcat(ones(N), un.x1, un.x2)
        H = 0.4 .* X' * X ./ n .+ 0.6 .* Xu' * Xu ./ N
        b = X' * (lab.y .- 0.6 .* lab.f) ./ n .+ 0.6 .* Xu' * un.f ./ N
        @test coef(rl) ≈ H \ b
        @test coefnames(rl) == ["(Intercept)", "x1", "x2"]
        r0l = ppi_ols(lab, un, :y, :f; covariates=[:x1, :x2], lambda=0)
        @test coef(r0l) ≈ X \ lab.y
        @test r0l.classical_coef ≈ X \ lab.y
        ro = ppi_ols(lab, un, :y, :f; covariates=[:x1, :x2])
        @test 0 <= ro.lambda <= 1
        @test all(stderror(ro) .< sqrt.(diag(ro.classical_vcov)))
        # logistic: λ = 0 is the labeled logistic MLE
        g0 = ppi_logistic(lab, un, :yb, :fb; covariates=[:x1, :x2], lambda=0)
        m = DrSnow.GLM.glm(X, lab.yb, DrSnow.Binomial())
        @test coef(g0) ≈ coef(m) rtol = 1e-6
        go = ppi_logistic(lab, un, :yb, :fb; covariates=[:x1, :x2])
        @test 0 <= go.lambda <= 1
        @test occursin("unlabeled N = 3000", sprint(show, MIME"text/plain"(), go))
        @test_throws ArgumentError ppi_logistic(lab, un, :y, :fb; covariates=[:x1])
        @test_throws ArgumentError ppi_logistic(lab, un, :yb, :f; covariates=[:x1])
        @test_throws ArgumentError ppi_logistic(lab, un, :yb, :fb; lambda=2.0)
        @test_throws ArgumentError ppi_ols(lab, un, :y, :f; covariates=[:nope])
        @test_throws ArgumentError ppi_mean(lab, un, :y, :f; lambda=-1)
        @test_throws ArgumentError ppi_mean(lab, un, :y, :f; lambda=:best)
    end

    @testset "Monte Carlo coverage" begin
        reps = mc_reps(1000, 150)
        # population targets from a large sample of the same DGP
        big, _ = ml_ppi_data(StableRNG(82), 400_000, 1)
        Xb = hcat(ones(nrow(big)), big.x1, big.x2)
        θ_ols = Xb \ big.y
        θ_mean = 1.0
        mb = DrSnow.GLM.glm(Xb, big.yb, DrSnow.Binomial())
        θ_log = coef(mb)
        cm = cl = cg = 0
        for rep in 1:reps
            l, u = ml_ppi_data(StableRNG(9000 + rep), 200, 2000)
            ci = confint(ppi_mean(l, u, :y, :f))
            cm += ci[1, 1] <= θ_mean <= ci[1, 2]
            ci = confint(ppi_ols(l, u, :y, :f; covariates=[:x1, :x2]))
            cl += ci[2, 1] <= θ_ols[2] <= ci[2, 2]
            ci = confint(ppi_logistic(l, u, :yb, :fb; covariates=[:x1, :x2]))
            cg += ci[2, 1] <= θ_log[2] <= ci[2, 2]
        end
        @info "Monte Carlo PPI coverage" mean = cm / reps ols = cl / reps logistic =
            cg / reps
        for c in (cm, cl, cg)
            @test abs(c / reps - 0.95) <= ml_cover_tol(reps)
        end
    end
end
