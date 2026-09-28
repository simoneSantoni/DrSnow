function _sv_test_graph(rng, n, p)
    A = zeros(n, n)
    for i in 1:n, j in (i + 1):n
        rand(rng) < p && (A[i, j] = A[j, i] = 1)
    end
    for i in 1:n                              # no isolates
        if sum(A[i, :]) == 0
            j = i == n ? 1 : i + 1
            A[i, j] = A[j, i] = 1
        end
    end
    return A
end

# Potential outcomes under the four Aronow–Samii conditions.
_sv_po(base, c) = base + (startswith(c, "treated") ? 2.0 : 0.0) +
                  (endswith(c, "_exposed") ? 1.0 : 0.0)

@testset "Design-based estimation (Aronow–Samii)" begin
    @testset "exact enumeration: HT unbiased, variance conservative" begin
        rng = StableRNG(21)
        n = 12
        A = _sv_test_graph(rng, n, 0.25)
        g = NetworkStructure(1:n, A)
        design = CompleteRandomization(n, 4)
        P = exposure_probabilities(g, design; method=:exact)
        @test P.method === :exact
        @test length(P.weights) == binomial(n, 4)
        @test sum(P.pi; dims=2) ≈ ones(n)
        base = 5 .+ randn(rng, n)
        truth = Dict(l => mean(_sv_po(base[i], l) for i in 1:n) for l in P.levels)
        cons = ["treated_exposed" => "control_unexposed",
                "control_exposed" => "control_unexposed",
                "treated_exposed" => "treated_unexposed"]
        tau = [truth[a] - truth[b] for (a, b) in cons]
        Zs = DrSnow._sv_enumerate(design)[1]
        R = size(Zs, 2)
        est = zeros(R, 3)
        vhat = zeros(R, 3)
        for r in 1:R
            lab = exposure_conditions(g, Zs[:, r])
            df = DataFrame(id=1:n, z=Int.(Zs[:, r]),
                           y=[_sv_po(base[i], lab[i]) for i in 1:n])
            e = exposure_effects(df, :y, :z, g, P; unit=:id, contrasts=cons,
                                 estimator=:horvitz_thompson)
            est[r, :] = coef(e)
            vhat[r, :] = diag(vcov(e))
        end
        m = vec(mean(est; dims=1))
        @test m ≈ tau atol = 1e-10                 # exactly unbiased
        truevar = vec(mean((est .- m') .^ 2; dims=1))
        @test all(vec(mean(vhat; dims=1)) .>= truevar .- 1e-10)   # conservative
    end

    @testset "Bernoulli: exact probabilities match closed form" begin
        e = DataFrame(source=[1, 1, 2, 4], target=[2, 3, 3, 5])
        g = NetworkStructure(1:5, e)
        p = 0.3
        P = exposure_probabilities(g, BernoulliAssignment(5, p))     # :auto → exact
        @test P.method === :exact
        deg = [2, 2, 2, 1, 1]
        k = findfirst(==("control_unexposed"), P.levels)
        @test P.pi[:, k] ≈ (1 - p) .* (1 - p) .^ deg
        k2 = findfirst(==("treated_exposed"), P.levels)
        @test P.pi[:, k2] ≈ p .* (1 .- (1 - p) .^ deg)
        # joint probability of two non-adjacent, neighbour-disjoint units factorizes
        J = DrSnow._sv_joint(P, k, k, [4], [1])
        @test J[1, 1] ≈ P.pi[4, k] * P.pi[1, k]
        # Monte Carlo approximates the exact probabilities
        Pm = exposure_probabilities(g, BernoulliAssignment(5, p); method=:monte_carlo,
                                    draws=40_000, rng=StableRNG(3))
        @test Pm.levels == P.levels
        @test maximum(abs.(Pm.pi .- P.pi)) < 0.015
    end

    @testset "supplied assignments reproduce exact enumeration" begin
        rng = StableRNG(4)
        A = _sv_test_graph(rng, 8, 0.3)
        g = NetworkStructure(1:8, A)
        d = CompleteRandomization(8, 3)
        Pe = exposure_probabilities(g, d; method=:exact)
        Z, w = DrSnow._sv_enumerate(d)
        Ps = exposure_probabilities(g, Matrix{Int}(Z); weights=w)
        @test Ps.method === :supplied && Ps.pi ≈ Pe.pi
        df = DataFrame(id=1:8, z=Int.(Z[:, 7]), y=randn(rng, 8))
        @test coef(exposure_effects(df, :y, :z, g, Pe; unit=:id, positivity=:restrict,
                                    estimator=:horvitz_thompson)) ≈
              coef(exposure_effects(df, :y, :z, g, Ps; unit=:id, positivity=:restrict,
                                    estimator=:horvitz_thompson))
        @test_throws DimensionMismatch exposure_probabilities(g, Matrix{Int}(Z[1:7, :]))
        @test_throws ArgumentError exposure_probabilities(g, Matrix{Int}(Z);
                                                          weights=fill(1.0, size(Z, 2)))
    end

    @testset "enumeration of stratified and cluster designs" begin
        p = PartitionStructure(1:6, [1, 1, 1, 2, 2, 2])
        ds = StratifiedRandomization([1, 1, 1, 2, 2, 2], Dict(1 => 1, 2 => 2))
        m = ExposureMapping(NeighborExposure(:share); cutpoints=[0.25, 0.75])
        P = exposure_probabilities(p, ds; mapping=m)
        @test P.method === :exact && length(P.weights) == 9
        @test sum(P.weights) ≈ 1
        # in stratum 1 (1 of 3 treated) a control's saturation is 1/2 w.p. 1
        k = findfirst(==("control_0.25<exposure≤0.75"), P.levels)
        @test P.pi[1, k] ≈ 2 / 3
        dc = ClusterRandomization([1, 1, 2, 2, 3, 3], 1)
        Pc = exposure_probabilities(p, dc; mapping=ExposureMapping(NeighborExposure(:any)))
        @test length(Pc.weights) == 3
    end

    @testset "mapping labels" begin
        g = NetworkStructure(1:4, DataFrame(source=[1, 2, 3], target=[2, 3, 4]))
        z = [1, 0, 0, 0]
        @test exposure_conditions(g, z) ==
              ["treated_unexposed", "control_exposed", "control_unexposed",
               "control_unexposed"]
        m2 = ExposureMapping(HopExposure(2; stat=:any))
        @test exposure_conditions(g, z, m2)[3] == "control_hop2"
        m3 = ExposureMapping(HopExposure(2; stat=:any); direct=false)
        @test exposure_conditions(g, [0, 1, 0, 0], m3)[1] == "hop1"
        @test exposure_conditions(g, [1, 0, 1, 0], m2)[2] == "control_hop1+hop2" ||
              exposure_conditions(g, [1, 0, 1, 0], m2)[2] == "control_hop1"
        mc = ExposureMapping((s, z) -> [z[i] ? "a" : "b" for i in eachindex(z)])
        @test exposure_conditions(g, z, mc) == ["a", "b", "b", "b"]
        mshare = ExposureMapping(NeighborExposure(:share))
        @test_throws ArgumentError exposure_conditions(g, z, mshare)
        @test_throws ArgumentError ExposureMapping(HopExposure(2); cutpoints=[0.5])
        @test_throws ArgumentError ExposureMapping(NeighborExposure(:share);
                                                   cutpoints=[0.5, 0.2])
    end

    @testset "positivity: error by default, :restrict redefines the estimand" begin
        # units 1–2 form a treatment cluster and are neighbours: unit 1 treated implies
        # unit 1 exposed, so "treated_unexposed" has probability 0 for units 1 and 2.
        g = NetworkStructure(1:6, DataFrame(source=[1, 3, 5], target=[2, 4, 6]))
        d = ClusterRandomization([1, 1, 2, 3, 4, 5], 2)
        P = exposure_probabilities(g, d)
        pos = exposure_positivity(P)
        @test pos.n_zero[pos.condition .== "treated_unexposed"][1] == 2
        z = Int.([1, 1, 1, 0, 0, 0])
        df = DataFrame(id=1:6, z=z, y=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        @test_throws ArgumentError exposure_effects(df, :y, :z, g, P; unit=:id)
        r = exposure_effects(df, :y, :z, g, P; unit=:id, positivity=:restrict,
                             estimator=:horvitz_thompson)
        @test nobs(r) == 4 && sort(r.excluded) == [1, 2]
    end

    @testset "isolates are outside the estimand population" begin
        g = NetworkStructure(1:5, DataFrame(source=[1, 2], target=[2, 3]))
        d = CompleteRandomization(5, 2)
        P = exposure_probabilities(g, d)
        @test P.defined == BitVector([1, 1, 1, 0, 0])
        df = DataFrame(id=1:5, z=[1, 0, 0, 1, 0], y=collect(1.0:5.0))
        r = exposure_effects(df, :y, :z, g, P; unit=:id, estimator=:horvitz_thompson)
        @test nobs(r) == 3 && Set(r.excluded) == Set([4, 5])
    end

    @testset "row-shuffling invariance and errors" begin
        rng = StableRNG(8)
        n = 30
        A = _sv_test_graph(rng, n, 0.1)
        ids = ["n$(i)" for i in 1:n]
        g = NetworkStructure(ids, A)
        d = CompleteRandomization(n, 10)
        P = exposure_probabilities(g, d; draws=2000, rng=StableRNG(1))
        z = draw_assignment(StableRNG(2), d)
        df = DataFrame(id=ids, z=Int.(z), y=randn(rng, n))
        r1 = exposure_effects(df, :y, :z, g, P; unit=:id, positivity=:restrict)
        r2 = exposure_effects(df[randperm(rng, n), :], :y, :z, g, P; unit=:id,
                              positivity=:restrict)
        @test coef(r1) ≈ coef(r2) && vcov(r1) ≈ vcov(r2)
        @test_throws ArgumentError exposure_effects(df, :y, :z, g, P; unit=:id,
                                                    contrasts=["foo" => "control_exposed"])
        @test_throws ArgumentError exposure_effects(df, :y, :z, g, P; unit=:id,
                                                    reference="foo")
        @test_throws ArgumentError exposure_effects(df, :y, :z, g, P; unit=:id,
                                                    estimator=:ols)
        @test_throws ArgumentError exposure_effects(df[1:(n - 1), :], :y, :z, g, P;
                                                    unit=:id)
        dm = allowmissing(df)
        dm.y[3] = missing
        @test_throws ArgumentError exposure_effects(dm, :y, :z, g, P; unit=:id,
                                                    positivity=:restrict)
        wrong = copy(df)
        wrong.z .= 0
        wrong.z[1:11] .= 1                        # 11 treated, design treats 10
        @test_throws ArgumentError exposure_effects(wrong, :y, :z, g, d; unit=:id,
                                                    draws=100)
        @test_throws DimensionMismatch exposure_probabilities(g,
                                                              CompleteRandomization(5, 2))
        @test_throws ArgumentError exposure_probabilities(g, d; method=:exact)
        @test_throws ArgumentError exposure_probabilities(g, CustomAssignment(n,
                                                          r -> rand(r, n) .< 0.5);
                                                          method=:exact)
        s = sprint(show, MIME"text/plain"(), r1)
        @test occursin("Aronow", s)
        @test occursin("conditions", sprint(show, P))
    end

    @testset "Monte Carlo: conservative variance (HT), coverage (Hájek)" begin
        rng = StableRNG(2024)
        n = 150
        A = _sv_test_graph(rng, n, 0.02)
        g = NetworkStructure(1:n, A)
        d = CompleteRandomization(n, 50)
        P = exposure_probabilities(g, d; draws=3000, rng=StableRNG(7))
        base = 3 .+ randn(rng, n)
        levels = ["treated_exposed", "control_exposed", "treated_unexposed"]
        cons = [l => "control_unexposed" for l in levels]
        ok = [all(P.pi[i, findfirst(==(l), P.levels)] > 0 for l in
                  vcat(levels, "control_unexposed")) for i in 1:n]
        tau = [mean(_sv_po(base[i], a) - _sv_po(base[i], b) for i in 1:n if ok[i])
               for (a, b) in cons]
        reps = mc_reps(600, 100)
        est = zeros(reps, 3)
        vhat = zeros(reps, 3)
        cover = zeros(3)
        for rep in 1:reps
            z = draw_assignment(rng, d)
            lab = exposure_conditions(g, z)
            df = DataFrame(id=1:n, z=Int.(z), y=[_sv_po(base[i], lab[i]) + 0.5randn(rng)
                                                 for i in 1:n])
            ht = exposure_effects(df, :y, :z, g, P; unit=:id, contrasts=cons,
                                  estimator=:horvitz_thompson, positivity=:restrict)
            est[rep, :] = coef(ht)
            vhat[rep, :] = diag(vcov(ht))
            hj = exposure_effects(df, :y, :z, g, P; unit=:id, contrasts=cons,
                                  positivity=:restrict)
            ci = confint(hj)
            cover .+= (ci[:, 1] .<= tau .<= ci[:, 2])
        end
        cover ./= reps
        ratio = vec(mean(vhat; dims=1)) ./ vec(var(est; dims=1))
        @info "Aronow–Samii: E[V̂]/Var (HT) and Hájek 95% coverage" ratio cover reps
        @test all(ratio .>= 0.85)                 # conservative up to MC noise
        mcse = sqrt.(vec(var(est; dims=1)) ./ reps)
        @test all(abs.(vec(mean(est; dims=1)) .- tau) .<= 4 .* mcse)
        @test all(cover .>= 0.9)
    end
end
