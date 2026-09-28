@testset "Assignment mechanisms" begin
    @testset "matched pairs" begin
        pairs = [3, 1, 2, 1, 3, 2]
        m = MatchedPairsRandomization(pairs)
        @test n_units(m) == 6
        @test n_assignments(m) == 8
        @test treatment_probabilities(m) == fill(0.5, 6)
        for s in 1:20
            z = draw_assignment(StableRNG(s), m)
            @test all(count(z[pairs .== p]) == 1 for p in 1:3)
        end
        zs, w = enumerate_assignments(m)
        @test length(unique(zs)) == 8 && sum(w) ≈ 1
        @test_throws ArgumentError MatchedPairsRandomization([1, 1, 1, 2])
    end

    @testset "blocked cluster" begin
        cl = [1, 1, 2, 2, 3, 4, 4, 5, 6, 6]
        bl = [1, 1, 1, 1, 1, 2, 2, 2, 2, 2]
        z = [1, 1, 0, 0, 0, 1, 1, 0, 0, 0]
        m = BlockClusterRandomization(cl, bl, z)
        @test n_assignments(m) == 9
        zs, w = enumerate_assignments(m)
        bf = bf_block_cluster(cl, bl, Bool.(z))
        @test Set(collect.(zs)) == Set(bf)
        @test treatment_probabilities(m) ≈ fill(1 / 3, 10)
        for s in 1:20
            d = draw_assignment(StableRNG(s), m)
            @test collect(d) in bf
        end
        m2 = BlockClusterRandomization(cl, bl, Dict(1 => 1, 2 => 1))
        @test n_assignments(m2) == 9
        @test_throws ArgumentError BlockClusterRandomization(cl, [1; bl[2:end-1]; 1], z)
        @test_throws ArgumentError BlockClusterRandomization(cl, bl, [1; zeros(Int, 9)])
    end

    @testset "rerandomization" begin
        rng = StableRNG(7)
        X = randn(rng, 10, 2)
        base = CompleteRandomization(10, 5)
        thr = 1.0
        m = Rerandomization(base, X; threshold=thr)
        for s in 1:10
            @test balance_mahalanobis(X, draw_assignment(StableRNG(s), m)) <= thr
        end
        zs, w = enumerate_assignments(m)
        all_z, _ = enumerate_assignments(base)
        @test length(zs) == count(z -> balance_mahalanobis(X, z) <= thr, all_z)
        @test sum(w) ≈ 1
        @test n_assignments(m) === nothing
        # Mahalanobis distance matches the textbook formula
        z = Bool[1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
        d = vec(mean(X[z, :]; dims=1) - mean(X[.!z, :]; dims=1))
        @test balance_mahalanobis(X, z) ≈ d' * inv(cov(X) * (1 / 5 + 1 / 5)) * d
        @test balance_mahalanobis(X, falses(10)) == Inf
        strict = Rerandomization(base, z -> false; max_draws=5)
        @test_throws ErrorException draw_assignment(StableRNG(1), strict)
    end

    @testset "enumeration" begin
        @test n_assignments(CompleteRandomization(10, 5)) == 252
        @test n_assignments(StratifiedRandomization([1, 1, 2, 2, 2], [1, 0, 1, 1, 0])) == 6
        @test n_assignments(ClusterRandomization([1, 1, 2, 3, 3], 1)) == 3
        @test n_assignments(BernoulliAssignment(4, [0.5, 1.0, 0.0, 0.2])) == 4
        @test n_assignments(CustomAssignment(3, r -> [true, false, true])) === nothing
        zs, w = enumerate_assignments(CompleteRandomization(6, 3))
        @test length(zs) == 20 && allunique(zs) && all(count.(zs) .== 3)
        mb = BernoulliAssignment(3, [0.2, 0.5, 0.9])
        zs, w = enumerate_assignments(mb)
        @test length(zs) == 8 && sum(w) ≈ 1
        i = findfirst(z -> z == BitVector([1, 0, 1]), zs)
        @test w[i] ≈ 0.2 * 0.5 * 0.9
        @test_throws ArgumentError enumerate_assignments(CompleteRandomization(40, 20))
        @test_throws ArgumentError enumerate_assignments(CustomAssignment(3,
                                                         r -> [true, false, true]))
        st = [1, 1, 2, 2, 2, 3, 3]
        zobs = Bool[1, 0, 1, 1, 0, 0, 1]
        zs, _ = enumerate_assignments(StratifiedRandomization(st, zobs))
        @test Set(collect.(zs)) == Set(bf_stratified(st, zobs))
        zs, _ = enumerate_assignments(ClusterRandomization([1, 1, 2, 3, 3, 4], 2))
        @test Set(collect.(zs)) == Set(bf_cluster([1, 1, 2, 3, 3, 4], 2))
    end

    @testset "support checks" begin
        @test DrSnow._ri_in_support(CompleteRandomization(4, 2), BitVector([1, 1, 0, 0]))
        @test !DrSnow._ri_in_support(CompleteRandomization(4, 2), BitVector([1, 0, 0, 0]))
        @test !DrSnow._ri_in_support(ClusterRandomization([1, 1, 2, 2], 1),
                                     BitVector([1, 0, 0, 0]))
        @test !DrSnow._ri_in_support(BernoulliAssignment(2, [0.0, 0.5]), BitVector([1, 0]))
        @test !DrSnow._ri_in_support(MatchedPairsRandomization([1, 1]), BitVector([1, 1]))
    end
end
