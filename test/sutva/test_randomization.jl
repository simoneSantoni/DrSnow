using SparseArrays

function _sv_ri_graph(rng, n, p)
    A = zeros(n, n)
    for i in 1:n, j in (i + 1):n
        rand(rng) < p && (A[i, j] = A[j, i] = 1)
    end
    for i in 1:n
        if sum(A[i, :]) == 0
            j = i == n ? 1 : i + 1
            A[i, j] = A[j, i] = 1
        end
    end
    return A
end

@testset "Randomization inference under interference" begin
    @testset "conditional samplers hold focal units fixed and respect the design" begin
        rng = StableRNG(1)
        n = 12
        focal = falses(n)
        focal[[1, 4, 7, 10]] .= true
        designs = [BernoulliAssignment(n, 0.4), CompleteRandomization(n, 5),
                   StratifiedRandomization(repeat([1, 2], 6), Dict(1 => 3, 2 => 2)),
                   ClusterRandomization(repeat(1:6, inner=2), 3),
                   CustomAssignment(n, r -> rand(r, n) .< 0.5)]
        for d in designs
            z = draw_assignment(rng, d)
            smp = DrSnow._sv_conditional_sampler(d, z, focal)
            for _ in 1:50
                zz = smp(rng)
                @test zz[focal] == z[focal]
                d isa CompleteRandomization && @test count(zz) == 5
                if d isa ClusterRandomization
                    @test all(zz[2k - 1] == zz[2k] for k in 1:6)
                    @test count(zz) == 6
                end
                if d isa StratifiedRandomization
                    @test count(zz[1:2:end]) == 3 && count(zz[2:2:end]) == 2
                end
            end
        end
        # rejection sampling gives up with an informative error
        impossible = CustomAssignment(n, r -> falses(n))
        z = trues(n)
        smp = DrSnow._sv_conditional_sampler(impossible, z, focal; max_tries=10)
        @test_throws ArgumentError smp(rng)
    end

    @testset "Fisher test p-value equals exact conditional enumeration" begin
        rng = StableRNG(2)
        n = 10
        A = _sv_ri_graph(rng, n, 0.3)
        g = NetworkStructure(1:n, A)
        d = CompleteRandomization(n, 4)
        z = draw_assignment(rng, d)
        y = randn(rng, n) .+ 0.8 .* (A * z .> 0)
        df = DataFrame(id=1:n, z=Int.(z), y=y)
        focal = [1, 2, 3, 4, 5]
        t = spillover_fisher_test(df, :y, :z, g, d; unit=:id, focal=focal, draws=40_000,
                                  rng=StableRNG(3))
        # exact: all permutations of the non-focal treatments
        nf = 6:10
        k = count(z[nf])
        tobs = t.statistic
        stats = Float64[]
        for c in DrSnow._sv_combinations(length(nf), k)
            zz = copy(z)
            zz[nf] .= false
            zz[nf[c]] .= true
            e = A * zz .> 0
            v = DrSnow._sv_stat_difference(y[focal], zz[focal], Float64.(e[focal]))
            push!(stats, isfinite(v) ? v : 0.0)
        end
        pexact = mean(abs.(stats) .>= abs(tobs) - 1e-12)
        @test abs(t.pvalue - pexact) < 0.01
        @test t.details.focal == focal
        @test occursin("not evidence", t.note)
    end

    @testset "Fisher test: size under the null and power with spillovers" begin
        rng = StableRNG(4)
        n = 120
        A = _sv_ri_graph(rng, n, 0.03)
        g = NetworkStructure(1:n, A)
        d = CompleteRandomization(n, 40)
        base = randn(rng, n)
        reps = mc_reps(1000, 150)
        rej0 = 0
        rej1 = 0
        for rep in 1:reps
            z = draw_assignment(rng, d)
            expo = A * z .> 0
            df = DataFrame(id=1:n, z=Int.(z), y0=base .+ 1.0 .* z,
                           y1=base .+ 1.0 .* z .+ 1.0 .* expo)
            t0 = spillover_fisher_test(df, :y0, :z, g, d; unit=:id, draws=199, rng=rng)
            t1 = spillover_fisher_test(df, :y1, :z, g, d; unit=:id, draws=199, rng=rng)
            rej0 += t0.pvalue <= 0.05
            rej1 += t1.pvalue <= 0.05
        end
        size_ = rej0 / reps
        power = rej1 / reps
        @info "spillover_fisher_test: size and power at 5%" size_ power reps
        @test size_ <= 0.05 + 3 * sqrt(0.05 * 0.95 / reps)
        @test power >= 0.7
    end

    @testset "Fisher test: slope statistic, invariance and errors" begin
        rng = StableRNG(5)
        n = 40
        A = _sv_ri_graph(rng, n, 0.1)
        ids = ["i$(k)" for k in 1:n]
        g = NetworkStructure(ids, A)
        d = BernoulliAssignment(n, 0.3)
        z = draw_assignment(rng, d)
        df = DataFrame(id=ids, z=Int.(z), y=randn(rng, n) .+ 2 .* (A * z) ./ vec(sum(A;
                                                                               dims=2)))
        focal = ids[1:20]
        t1 = spillover_fisher_test(df, :y, :z, g, d; unit=:id, focal=focal,
                                   exposure=NeighborExposure(:share), statistic=:slope,
                                   draws=299, rng=StableRNG(9))
        t2 = spillover_fisher_test(df[randperm(rng, n), :], :y, :z, g, d; unit=:id,
                                   focal=focal, exposure=NeighborExposure(:share),
                                   statistic=:slope, draws=299, rng=StableRNG(9))
        @test t1.pvalue == t2.pvalue && t1.statistic ≈ t2.statistic
        custom = (y, z, e) -> cor(y, e)
        t3 = spillover_fisher_test(df, :y, :z, g, d; unit=:id, focal=focal,
                                   exposure=NeighborExposure(:share), statistic=custom,
                                   draws=99, rng=StableRNG(9))
        @test 0 < t3.pvalue <= 1
        @test_throws ArgumentError spillover_fisher_test(df, :y, :z, g, d; unit=:id,
                                                         focal=["nope"])
        @test_throws ArgumentError spillover_fisher_test(df, :y, :z, g, d; unit=:id,
                                                         focal=ids)
        @test_throws ArgumentError spillover_fisher_test(df, :y, :z, g, d; unit=:id,
                                                         statistic=:mean)
        @test_throws ArgumentError spillover_fisher_test(df, :y, :z, g, d; unit=:id,
                                                         focal_share=1.5)
        @test_throws DimensionMismatch spillover_fisher_test(df, :y, :z, g,
                                                             BernoulliAssignment(5, 0.5);
                                                             unit=:id)
    end

    @testset "exposure_balance_test: correct size despite structural imbalance" begin
        # Degree varies a lot and the covariate is degree itself: exposed units have
        # higher degree by construction, so a naive two-sample test over-rejects.
        rng = StableRNG(6)
        n = 150
        A = zeros(n, n)
        hubs = 1:10
        for i in 11:n
            for h in hubs
                rand(rng) < 0.3 && (A[i, h] = A[h, i] = 1)
            end
            j = rand(rng, 11:n)
            j != i && (A[i, j] = A[j, i] = 1)
        end
        g = NetworkStructure(1:n, A)
        deg = vec(sum(A; dims=2))
        d = CompleteRandomization(n, 30)
        reps = mc_reps(400, 80)
        rej = 0
        naive = 0
        for rep in 1:reps
            z = draw_assignment(rng, d)
            df = DataFrame(id=1:n, z=Int.(z), degree=deg, x=randn(rng, n))
            t = exposure_balance_test(df, :z, g, d; unit=:id, covariates=[:degree, :x],
                                      draws=199, rng=rng)
            rej += t.pvalue <= 0.05
            ex = (A * z .> 0)
            a, b = deg[ex], deg[.!ex]
            tt = (mean(a) - mean(b)) / sqrt(var(a) / length(a) + var(b) / length(b))
            naive += abs(tt) > 1.96
        end
        @info "exposure_balance_test size vs naive t-test" rej / reps naive / reps
        @test rej / reps <= 0.05 + 3 * sqrt(0.05 * 0.95 / reps)
        @test naive / reps > 0.5
        # power: a spatially smooth covariate and treatment concentrated in one region
        # (not what complete randomization produces)
        rs = StableRNG(12)
        sp = SpatialStructure(1:n; x=10 .* rand(rs, n), y=10 .* rand(rs, n))
        xc = sp.coords[:, 1]
        z = zeros(Int, n)
        z[sortperm(xc; rev=true)[1:30]] .= 1
        df = DataFrame(id=1:n, z=z, xc=xc)
        t = exposure_balance_test(df, :z, sp, d; unit=:id, covariates=[:xc],
                                  exposure=NeighborExposure(:any; radius=1.5), draws=499,
                                  rng=StableRNG(1))
        @test t.pvalue < 0.01
        @test nrow(t.details.table) == 1
        @test_throws ArgumentError exposure_balance_test(df, :z, sp, d; unit=:id,
                                                         covariates=Symbol[])
        @test_throws ArgumentError exposure_balance_test(df, :z, sp, d; unit=:id,
                                                         covariates=[:xc], among=:some)
    end

    @testset "Moran's I: Cliff–Ord moments equal exact permutation moments" begin
        rng = StableRNG(7)
        n = 10
        s = SpatialStructure(1:n; x=10 .* rand(rng, n), y=10 .* rand(rng, n))
        W = neighbor_matrix(s; radius=4.0)
        k = 4
        for rs in (true, false)
            Wm = rs ? sparse(Diagonal([r > 0 ? 1 / r : 0.0 for r in vec(sum(W; dims=2))]) *
                             W) : W
            S0 = sum(Wm)
            vals = Float64[]
            for c in DrSnow._sv_combinations(n, k)
                z = zeros(n)
                z[c] .= 1
                push!(vals, DrSnow._sv_moran(z, Wm, S0))
            end
            z = zeros(n)
            z[1:k] .= 1
            EI, VI = DrSnow._sv_moran_moments(z, Wm)
            @test EI ≈ mean(vals)
            @test VI ≈ mean((vals .- mean(vals)) .^ 2)
        end
        df = DataFrame(id=1:n, z=[ones(Int, k); zeros(Int, n - k)])
        t = treatment_moran_test(df, :z, s, CompleteRandomization(n, k); unit=:id,
                                 radius=4.0, draws=999, rng=StableRNG(2))
        @test t.details.expected ≈ -1 / (n - 1)
        @test t.details.variance > 0 && 0 < t.pvalue <= 1
        tc = treatment_moran_test(df, :z, s, ClusterRandomization(repeat(1:5, inner=2), 2);
                                  unit=:id, radius=4.0, draws=99, rng=StableRNG(2))
        @test tc.details.expected === nothing
        @test_throws ArgumentError treatment_moran_test(df, :z, s,
                                                        CompleteRandomization(n, k);
                                                        unit=:id)
        @test_throws DimensionMismatch treatment_moran_test(df, :z, s,
                                                            CompleteRandomization(n, k);
                                                            unit=:id, weights=ones(3, 3))
    end

    @testset "Moran's I: size under the design, power under clustered assignment" begin
        rng = StableRNG(8)
        n = 100
        s = SpatialStructure(1:n; x=10 .* rand(rng, n), y=10 .* rand(rng, n))
        d = CompleteRandomization(n, 30)
        reps = mc_reps(400, 80)
        rej = 0
        for rep in 1:reps
            z = draw_assignment(rng, d)
            df = DataFrame(id=1:n, z=Int.(z))
            t = treatment_moran_test(df, :z, s, d; unit=:id, radius=2.0, draws=199,
                                     rng=rng)
            rej += t.pvalue <= 0.05
        end
        @info "treatment_moran_test size at 5%" rej / reps
        @test rej / reps <= 0.05 + 3 * sqrt(0.05 * 0.95 / reps)
        # spatially clustered treatment: the 30 units closest to a corner
        dist = [hypot(s.coords[i, 1], s.coords[i, 2]) for i in 1:n]
        z = zeros(Int, n)
        z[sortperm(dist)[1:30]] .= 1
        t = treatment_moran_test(DataFrame(id=1:n, z=z), :z, s, d; unit=:id, radius=2.0,
                                 draws=499, rng=StableRNG(1))
        @test t.pvalue < 0.01 && t.details.z > 3
    end
end
