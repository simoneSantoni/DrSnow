@testset "simplex least squares satisfies KKT conditions" begin
    rng = StableRNG(11)
    for (m, n) in [(5, 3), (10, 30), (40, 12), (3, 50)]
        A = randn(rng, m, n)
        b = randn(rng, m) .+ 0.3
        w = DrSnow._sc_simplex_ls(A, b)
        @test all(>=(0), w)
        @test sum(w) ≈ 1 atol = 1e-12
        g = A' * (A * w .- b)              # gradient / 2
        act = w .> 1e-10
        mu = mean(g[act])
        @test maximum(abs.(g[act] .- mu)) < 1e-8 * max(1, norm(A)^2)
        @test all(g[.!act] .>= mu - 1e-8 * max(1, norm(A)^2))
        # agrees with a long Frank–Wolfe run on the same problem
        x = fill(1 / n, n)
        for _ in 1:20_000
            x = DrSnow._sc_fw_step(A, x, b, 0.0)
        end
        @test sum(abs2, A * w .- b) <= sum(abs2, A * x .- b) + 1e-8
    end
    @test DrSnow._sc_simplex_ls(randn(StableRNG(1), 4, 1), randn(StableRNG(2), 4)) == [1.0]
end

@testset "NNLS" begin
    A = [1.0 0.0; 0.0 1.0; 1.0 1.0]
    @test DrSnow._sc_nnls(A, [1.0, 2.0, 3.0]) ≈ [1.0, 2.0]
    @test DrSnow._sc_nnls(A, [-1.0, 2.0, 1.0]) ≈ [0.0, 1.5]
end

@testset "Frank–Wolfe weights decrease the objective" begin
    rng = StableRNG(12)
    Y = randn(rng, 20, 11)
    lam, vals = DrSnow._sc_weight_fw(Y, 0.1; max_iter=500, min_decrease=1e-8)
    @test all(>=(0), lam) && sum(lam) ≈ 1
    @test all(diff(vals) .<= 1e-12)
    @test DrSnow._sc_sparsify([0.5, 0.1, 0.4]) ≈ [0.5, 0.0, 0.4] ./ 0.9
    @test DrSnow._sc_sum_normalize([0.0, 0.0]) == [0.5, 0.5]
end

@testset "Nelder–Mead" begin
    rosen(x) = (1 - x[1])^2 + 100 * (x[2] - x[1]^2)^2
    x, f = DrSnow._sc_nelder_mead(rosen, [-1.2, 1.0]; max_iter=10_000, ftol=1e-14)
    @test x ≈ [1.0, 1.0] atol = 1e-4
    @test f < 1e-8
end
