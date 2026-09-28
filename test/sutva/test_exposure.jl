using SparseArrays

# Brute-force all-pairs shortest paths (Floyd–Warshall) for checking the BFS.
function _floyd(A)
    n = size(A, 1)
    D = fill(typemax(Int) ÷ 2, n, n)
    for i in 1:n
        D[i, i] = 0
    end
    for i in 1:n, j in 1:n
        A[i, j] != 0 && (D[i, j] = 1)
    end
    for k in 1:n, i in 1:n, j in 1:n
        D[i, j] = min(D[i, j], D[i, k] + D[k, j])
    end
    return D
end

function _random_graph(rng, n, p)
    A = zeros(n, n)
    for i in 1:n, j in (i + 1):n
        rand(rng) < p && (A[i, j] = A[j, i] = 1)
    end
    return A
end

@testset "Exposure mappings" begin
    # star 1-2, 1-3, 1-4 plus edge 4-5 and isolate 6
    e = DataFrame(source=[1, 1, 1, 4], target=[2, 3, 4, 5])
    g = NetworkStructure(1:6, e)
    z = [0, 1, 1, 0, 1, 0]

    @testset "NeighborExposure statistics on a network" begin
        sh = compute_exposure(g, z, NeighborExposure(:share))
        @test sh.unit == 1:6
        @test isequal(sh.share, [2 / 3, 0.0, 0.0, 0.5, 0.0, missing])
        ct = compute_exposure(g, z, NeighborExposure(:count))
        @test isequal(ct.count, [2.0, 0.0, 0.0, 1.0, 0.0, missing])
        an = compute_exposure(g, z, NeighborExposure(:any))
        @test isequal(an.any, [1.0, 0.0, 0.0, 1.0, 0.0, missing])
        th = compute_exposure(g, z, NeighborExposure(:share; threshold=0.6))
        @test isequal(th[!, 2], [1.0, 0.0, 0.0, 0.0, 0.0, missing])
        @test names(th)[2] == "share_ge_0.6"
        iz = compute_exposure(g, z, NeighborExposure(:share; isolates=:zero))
        @test iz.share[6] == 0.0
        @test_throws ArgumentError NeighborExposure(:mean)
        @test_throws ArgumentError NeighborExposure(:share; isolates=:drop)
        @test_throws ArgumentError compute_exposure(g, z, NeighborExposure(:share;
                                                                          radius=1.0))
        @test_throws DimensionMismatch compute_exposure(g, z[1:5], NeighborExposure())
        @test_throws ArgumentError compute_exposure(g, [0, 2, 0, 0, 0, 0],
                                                    NeighborExposure())
    end

    @testset "weighted and directed networks" begin
        ew = DataFrame(source=[1, 2], target=[3, 3], w=[1.0, 3.0])
        gw = NetworkStructure(1:3, ew; weight=:w, directed=true)
        r = compute_exposure(gw, [1, 0, 0], NeighborExposure(:weighted_sum))
        @test isequal(r[!, 2], [missing, missing, 1.0])   # units 1, 2 have no sources
        r2 = compute_exposure(gw, [0, 1, 0], NeighborExposure(:weighted_share))
        @test r2[3, 2] ≈ 0.75
        r3 = compute_exposure(gw, [0, 1, 0], NeighborExposure(:share))
        @test r3[3, 2] ≈ 0.5
    end

    @testset "spatial neighbours and rings" begin
        # units on a line at x = 0, 1, 2.5, 6
        s = SpatialStructure(["a", "b", "c", "d"]; x=[0.0, 1.0, 2.5, 6.0],
                             y=zeros(4))
        n1 = compute_exposure(s, [1, 0, 0, 0], NeighborExposure(:count; radius=1.5))
        @test isequal(n1.count, [0.0, 1.0, 0.0, missing])
        dec = compute_exposure(s, [1, 0, 1, 0],
                               NeighborExposure(:weighted_sum; radius=3.0,
                                                decay=d -> 1 / d))
        @test dec[2, 2] ≈ 1 / 1 + 1 / 1.5
        @test_throws ArgumentError compute_exposure(s, [1, 0, 0, 0], NeighborExposure())
        rings = RingExposure([1.0, 2.0, 4.0])
        @test exposure_columns(rings) == ["ring_0_1", "ring_1_2", "ring_2_4"]
        r = compute_exposure(s, [1, 0, 0, 0], rings)
        @test Matrix(r[:, 2:4]) == [0 0 0; 1 0 0; 0 0 1; 0 0 0]
        # nearest treated unit decides the ring when several are treated
        rn = compute_exposure(s, [1, 0, 1, 0], rings)
        @test Matrix(rn[:, 2:4]) == [0 0 1; 1 0 0; 0 0 1; 0 0 1]
        ra = compute_exposure(s, [1, 0, 1, 0], RingExposure([1.0, 2.0, 4.0]; stat=:any))
        @test Matrix(ra[:, 2:4]) == [0 0 1; 1 1 0; 0 0 1; 0 0 1]
        rc = compute_exposure(s, [1, 1, 1, 0], RingExposure([1.0, 2.0, 4.0]; stat=:count))
        @test Matrix(rc[:, 2:4]) == [1 0 1; 1 1 0; 0 1 1; 0 0 1]
        rs = compute_exposure(s, [1, 0, 1, 0], RingExposure([1.0, 2.0]; stat=:share))
        @test isequal(rs[1, 2], 0.0) && ismissing(rs[1, 3])  # empty band → missing
        @test_throws ArgumentError RingExposure([2.0, 1.0])
        @test_throws ArgumentError RingExposure(Float64[])
        @test_throws ArgumentError RingExposure([1.0]; stat=:mean)
        @test_throws ArgumentError compute_exposure(g, z, rings)
    end

    @testset "HopExposure uses shortest paths (checked against Floyd–Warshall)" begin
        rng = StableRNG(11)
        for rep in 1:3
            n = 25
            A = _random_graph(rng, n, 0.12)
            gg = NetworkStructure(1:n, A)
            zz = rand(rng, 0:1, n)
            D = _floyd(A)
            deg = vec(sum(A; dims=2))
            h = compute_exposure(gg, zz, HopExposure(3; stat=:count))
            hn = compute_exposure(gg, zz, HopExposure(3; stat=:nearest))
            for i in 1:n
                if deg[i] == 0
                    @test all(ismissing, h[i, 2:4])
                    continue
                end
                for k in 1:3
                    @test h[i, k + 1] == count(j -> j != i && D[i, j] == k && zz[j] == 1,
                                               1:n)
                end
                dmin = minimum([D[i, j] for j in 1:n if j != i && zz[j] == 1];
                               init=typemax(Int))
                @test [hn[i, k + 1] for k in 1:3] == [Float64(dmin == k) for k in 1:3]
            end
        end
        @test_throws ArgumentError HopExposure(0)
        s = SpatialStructure(1:2; x=[0.0, 1.0], y=[0.0, 0.0])
        @test_throws ArgumentError compute_exposure(s, [1, 0], HopExposure(1))
    end

    @testset "partition exposure is leave-one-out saturation" begin
        p = PartitionStructure(1:6, [1, 1, 1, 2, 2, 2])
        r = compute_exposure(p, [1, 1, 0, 0, 0, 1], NeighborExposure(:share))
        @test r.share == [0.5, 0.5, 1.0, 0.5, 0.5, 0.0]
    end

    @testset "CustomExposure" begin
        f(s, z) = [coalesce(sum(skipmissing(z)), 0.0) - coalesce(z[i], 0.0)
                   for i in eachindex(z)]
        r = compute_exposure(g, z, CustomExposure(f, "others_treated"))
        @test r.others_treated == [3.0, 2.0, 2.0, 3.0, 2.0, 3.0]
        bad = CustomExposure((s, z) -> zeros(2), ["x"])
        @test_throws DimensionMismatch compute_exposure(g, z, bad)
    end

    @testset "panel exposures: time-varying, keyed, shuffling-invariant" begin
        rng = StableRNG(5)
        n = 30
        T = 4
        A = _random_graph(rng, n, 0.15)
        ids = ["u$i" for i in 1:n]
        gg = NetworkStructure(ids, A)
        adopt = rand(rng, [2, 3, 4, 99], n)
        panel = DataFrame(id=repeat(ids, T), t=repeat(1:T; inner=n))
        panel.d = Int.(panel.t .>= adopt[parse.(Int, chop.(panel.id; head=1, tail=0))])
        spec = NeighborExposure(:share)
        ex = compute_exposure(panel, :d, gg, spec; unit=:id, time=:t)
        @test ex.id == panel.id && ex.t == panel.t
        for t in 1:T
            rows = panel.t .== t
            zt = panel.d[rows][sortperm(panel.id[rows], by=x -> parse(Int, x[2:end]))]
            ref = compute_exposure(gg, zt, spec)
            got = ex[rows, :]
            got = got[sortperm(got.id, by=x -> parse(Int, x[2:end])), :]
            @test isequal(got.share, ref.share)
        end
        perm = randperm(rng, nrow(panel))
        ex2 = compute_exposure(panel[perm, :], :d, gg, spec; unit=:id, time=:t)
        @test isequal(ex2.share, ex.share[perm])
        # a missing treatment makes neighbours' exposure in that period missing
        p2 = copy(panel)
        p2.d = allowmissing(p2.d)
        k = findfirst(r -> r.t == 2, eachrow(p2))
        p2.d[k] = missing
        ex3 = compute_exposure(p2, :d, gg, spec; unit=:id, time=:t)
        i = parse(Int, p2.id[k][2:end])
        nb = findall(!iszero, A[:, i])
        for j in nb
            r = findfirst(x -> x.id == ids[j] && x.t == 2, eachrow(p2))
            @test ismissing(ex3.share[r])
        end
        # unit sets must coincide; duplicates are errors
        @test_throws ArgumentError compute_exposure(panel[panel.id .!= "u1", :], :d, gg,
                                                    spec; unit=:id, time=:t)
        @test_throws ArgumentError compute_exposure(vcat(panel, panel[1:1, :]), :d, gg,
                                                    spec; unit=:id, time=:t)
        bad = copy(panel)
        bad.id[1] = "zz"
        @test_throws ArgumentError compute_exposure(bad, :d, gg, spec; unit=:id, time=:t)
        @test_throws ArgumentError compute_exposure(panel, :nope, gg, spec; unit=:id)
    end

    @testset "panel exposure scales (sparse products)" begin
        rng = StableRNG(9)
        n = 3000
        T = 25
        s = SpatialStructure(1:n; x=100 .* rand(rng, n), y=100 .* rand(rng, n))
        panel = DataFrame(id=repeat(1:n, T), t=repeat(1:T; inner=n))
        panel.d = Int.(rand(rng, nrow(panel)) .< 0.2)
        compute_exposure(panel[1:n, :], :d, s, RingExposure([2.0, 4.0]); unit=:id)
        el = @elapsed compute_exposure(panel, :d, s, RingExposure([2.0, 4.0]);
                                       unit=:id, time=:t)
        @test el < 20
    end
end
