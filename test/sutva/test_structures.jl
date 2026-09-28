using SparseArrays

@testset "Interference structures" begin
    @testset "SpatialStructure: coordinates by name, distances" begin
        s = SpatialStructure(["a", "b", "c"]; x=[0.0, 3.0, 0.0], y=[0.0, 4.0, 1.0])
        D = pairwise_distances(s)
        @test D[1, 2] == 5.0 && D[2, 1] == 5.0 && D[1, 3] == 1.0
        @test structure_units(s) == ["a", "b", "c"]
        @test n_units(s) == 3
        # haversine: one degree of latitude on the mean-radius sphere
        g = SpatialStructure(1:2; lat=[45.0, 46.0], lon=[9.0, 9.0])
        @test pairwise_distances(g)[1, 2] ≈ 6371.0 * π / 180
        gm = SpatialStructure(1:2; lat=[45.0, 46.0], lon=[9.0, 9.0], units=:mi)
        @test pairwise_distances(gm)[1, 2] ≈ 3958.7613 * π / 180
        gr = SpatialStructure(1:2; lat=[0.0, 0.0], lon=[0.0, 90.0], earth_radius=1.0)
        @test pairwise_distances(gr)[1, 2] ≈ π / 2
        # antipodal-safe and dateline-crossing
        dl = SpatialStructure(1:2; lat=[0.0, 0.0], lon=[179.5, -179.5])
        @test pairwise_distances(dl)[1, 2] ≈ 6371.0 * π / 180
        # table constructor, panel rows with constant coordinates, any row order
        df = DataFrame(id=["b", "a", "b", "a"], lat=[46.0, 45.0, 46.0, 45.0],
                       lon=[9.0, 9.0, 9.0, 9.0])
        st = SpatialStructure(df, :id; lat=:lat, lon=:lon)
        @test Set(structure_units(st)) == Set(["a", "b"])
        @test pairwise_distances(st)[1, 2] ≈ 6371.0 * π / 180
    end

    @testset "SpatialStructure errors" begin
        @test_throws ArgumentError SpatialStructure(1:2; lat=[95.0, 0.0], lon=[0.0, 0.0])
        @test_throws ArgumentError SpatialStructure(1:2; lat=[0.0, 0.0], lon=[0.0, 400.0])
        @test_throws ArgumentError SpatialStructure(1:2; x=[0.0, 1.0])
        @test_throws ArgumentError SpatialStructure(1:2)
        @test_throws ArgumentError SpatialStructure(1:2; lat=[0.0, 1.0], lon=[0.0, 1.0],
                                                    x=[0.0, 1.0], y=[0.0, 1.0])
        @test_throws ArgumentError SpatialStructure([1, 1]; x=[0.0, 1.0], y=[0.0, 1.0])
        @test_throws DimensionMismatch SpatialStructure(1:3; x=[0.0, 1.0], y=[0.0, 1.0])
        @test_throws ArgumentError SpatialStructure([1, missing]; x=[0.0, 1.0],
                                                    y=[0.0, 1.0])
        @test_throws ArgumentError SpatialStructure(1:2; x=[0.0, NaN], y=[0.0, 1.0])
        df = DataFrame(id=[1, 1], x=[0.0, 1.0], y=[0.0, 0.0])
        @test_throws ArgumentError SpatialStructure(df, :id; x=:x, y=:y)
        @test_throws ArgumentError SpatialStructure(df, :nope; x=:x, y=:y)
        @test_throws ArgumentError SpatialStructure(1:2; x=[0.0, 1.0], y=[0.0, 1.0],
                                                    earth_radius=10.0)
    end

    @testset "NetworkStructure from matrix and edge list" begin
        A = [0 1 0; 1 0 1; 0 1 0]
        g = NetworkStructure(["x", "y", "z"], A)
        @test !g.directed && !g.weighted
        @test neighbor_matrix(g) == sparse(Float64.(A))
        e = DataFrame(source=["z", "x"], target=["y", "y"])   # any order, by id
        g2 = NetworkStructure(["x", "y", "z", "iso"], e)
        B = Matrix(neighbor_matrix(g2))
        @test B[1:3, 1:3] == A && all(B[4, :] .== 0)
        # directed: source → target means A[target, source]
        gd = NetworkStructure(["x", "y", "z"], DataFrame(source=["x"], target=["y"]);
                              directed=true)
        @test Matrix(gd.A) == [0 0 0; 1 0 0; 0 0 0]
        gw = NetworkStructure(1:2, DataFrame(source=[1], target=[2], w=[2.5]);
                              weight=:w)
        @test gw.weighted && neighbor_matrix(gw; weighted=true)[1, 2] == 2.5
        @test neighbor_matrix(gw)[1, 2] == 1.0
    end

    @testset "NetworkStructure errors" begin
        @test_throws ArgumentError NetworkStructure(1:2, [0 1; 0 0])       # asymmetric
        @test_throws ArgumentError NetworkStructure(1:2, [1 1; 1 0])       # self-loop
        @test_throws ArgumentError NetworkStructure(1:2, [0 -1; -1 0])     # negative
        @test_throws DimensionMismatch NetworkStructure(1:3, [0 1; 1 0])
        @test_throws ArgumentError NetworkStructure(1:2, DataFrame(source=[1], target=[3]))
        @test_throws ArgumentError NetworkStructure(1:2, DataFrame(source=[1], target=[1]))
        dup = DataFrame(source=[1, 2], target=[2, 1])
        @test_throws ArgumentError NetworkStructure(1:2, dup)
        @test_throws ArgumentError NetworkStructure(1:2, DataFrame(a=[1], b=[2]))
    end

    @testset "shortest_path_hops is BFS, not walk counts" begin
        # triangle 1-2-3 plus tail 3-4-5: walks of length 2 reach 1-hop neighbours,
        # shortest paths do not.
        e = DataFrame(source=[1, 2, 1, 3, 4], target=[2, 3, 3, 4, 5])
        g = NetworkStructure(1:5, e)
        H = shortest_path_hops(g; max_hops=3)
        @test H[1, 2] == 1 && H[1, 3] == 1 && H[1, 4] == 2 && H[1, 5] == 3
        @test H[5, 1] == 3 && H[4, 2] == 2
        @test iszero(H[1, 1])
        H2 = shortest_path_hops(g; max_hops=2)
        @test iszero(H2[1, 5])
        # directed chain a → b → c: influence reaches c from a in 2 hops, not a from c
        gd = NetworkStructure(["a", "b", "c"], DataFrame(source=["a", "b"],
                                                         target=["b", "c"]); directed=true)
        Hd = shortest_path_hops(gd; max_hops=2)
        @test Hd[3, 1] == 2 && iszero(Hd[1, 3])
        Hu = shortest_path_hops(gd; max_hops=2, direction=:undirected)
        @test Hu[1, 3] == 2
        @test_throws ArgumentError shortest_path_hops(g; max_hops=0)
        @test_throws ArgumentError shortest_path_hops(g; max_hops=1, direction=:up)
    end

    @testset "PartitionStructure" begin
        p = PartitionStructure(["a", "b", "c", "d"], [1, 2, 1, 2])
        W = Matrix(neighbor_matrix(p))
        @test W == [0 0 1 0; 0 0 0 1; 1 0 0 0; 0 1 0 0]
        df = DataFrame(id=["a", "b", "a"], g=["x", "y", "x"])
        @test n_units(PartitionStructure(df, :id; group=:g)) == 2
        @test_throws DimensionMismatch PartitionStructure(1:3, [1, 1])
        @test_throws ArgumentError PartitionStructure(1:2, [1, missing])
        @test_throws ArgumentError neighbor_matrix(p; radius=1.0)
    end

    @testset "neighbor_matrix for spatial structures" begin
        s = SpatialStructure(1:4; x=[0.0, 1.0, 0.0, 5.0], y=[0.0, 0.0, 0.0, 5.0])
        W = Matrix(neighbor_matrix(s; radius=1.0))
        @test W == [0 1 1 0; 1 0 1 0; 1 1 0 0; 0 0 0 0]   # co-located units included
        @test_throws ArgumentError neighbor_matrix(s)
        @test_throws ArgumentError neighbor_matrix(s; radius=-1.0)
    end

    @testset "show methods" begin
        s = SpatialStructure(1:2; lat=[0.0, 1.0], lon=[0.0, 1.0])
        @test occursin("haversine", sprint(show, s))
        @test occursin("undirected", sprint(show, NetworkStructure(1:2, [0 1; 1 0])))
        @test occursin("2 groups", sprint(show, PartitionStructure(1:2, [1, 2])))
    end
end
