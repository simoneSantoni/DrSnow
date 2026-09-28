# Non-bipartite matching: blossom algorithm vs exhaustive search, 1-D pairing,
# greedy pairs vs blockTools.

# Minimum total distance over maximum-cardinality matchings, by dynamic programming
# over subsets (n ≤ 14).
function brute_matching_cost(D)
    n = size(D, 1)
    memo = Dict{Int,Float64}()
    function f(mask)
        mask == 0 && return 0.0
        haskey(memo, mask) && return memo[mask]
        i = trailing_zeros(mask) + 1
        rest = mask & ~(1 << (i - 1))
        best = Inf
        if isodd(count_ones(mask))       # the lowest unit may stay unmatched
            best = f(rest)
        end
        for j in (i + 1):n
            if rest & (1 << (j - 1)) != 0
                best = min(best, D[i, j] + f(rest & ~(1 << (j - 1))))
            end
        end
        memo[mask] = best
        return best
    end
    # for odd n exactly one unit is unmatched; the recursion above allows the lowest
    # unit of an odd set to be skipped, which covers every choice of the unmatched unit
    return f((1 << n) - 1)
end

pair_cost(D, pairs) = sum(D[i, j] for (i, j) in pairs; init=0.0)

@testset "blossom matching = exhaustive optimum" begin
    rng = StableRNG(21)
    for trial in 1:150
        n = rand(rng, 2:12)
        D = if trial % 3 == 0
            A = rand(rng, 0:4, n, n)            # many ties: exercises blossoms
            Float64.(A + A')
        else
            P = randn(rng, n, 2)
            [sqrt(sum(abs2, P[i, :] .- P[j, :])) for i in 1:n, j in 1:n]
        end
        D[diagind(D)] .= 0
        pairs, left = DrSnow._des_optimal_pairs(D)
        @test length(pairs) == n ÷ 2
        @test (left == 0) == iseven(n)
        used = vcat(collect.(pairs)...)
        @test allunique(used)
        @test pair_cost(D, pairs) ≈ brute_matching_cost(D) atol = 1e-8 * max(1, maximum(D))
    end
    # a larger instance is at least as good as greedy
    P = randn(rng, 60, 3)
    D = [sqrt(sum(abs2, P[i, :] .- P[j, :])) for i in 1:60, j in 1:60]
    po, _ = DrSnow._des_optimal_pairs(D)
    pg, _ = DrSnow._des_greedy_pairs(D)
    @test pair_cost(D, po) <= pair_cost(D, pg) + 1e-10
end

@testset "1-D optimal pairing" begin
    rng = StableRNG(22)
    for n in 2:11
        s = randn(rng, n)
        order = sortperm(s)
        pairs, left = DrSnow._des_pairs_1d(s, order)
        D = abs.(s .- s')
        @test pair_cost(D, pairs) ≈ brute_matching_cost(D) atol = 1e-10
        @test (left == 0) == iseven(n)
    end
end

@testset "greedy pairs reproduce blockTools optGreedy" begin
    d = CSV.read(joinpath(DES_VALDIR, "experiment_data.csv"), DataFrame)
    bt = CSV.read(joinpath(DES_VALDIR, "reference_blocktools.csv"), DataFrame)
    bd = block_design(d; id=:id, covariates=[:x1, :x2], distance=:mahalanobis,
                      algorithm=:greedy)
    mine = Set([Set(String.(bd.ids[bd.blocks .== b])) for b in 1:maximum(bd.blocks)])
    theirs = Set([Set(String.(bt.id[bt.pair .== b])) for b in unique(bt.pair)])
    @test mine == theirs
    bo = block_design(d; id=:id, covariates=[:x1, :x2], distance=:mahalanobis)
    @test bo.objective <= bd.objective
end
