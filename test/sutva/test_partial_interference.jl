@testset "Partial interference: two-stage randomization" begin
    G = 40
    m = 10
    groups = repeat(["g$(k)" for k in 1:G]; inner=m)
    sats = [0.2, 0.6]

    @testset "TwoStageRandomization mechanism" begin
        d = TwoStageRandomization(groups, sats, [20, 20])
        @test n_units(d) == G * m
        @test treatment_probabilities(d) ≈ fill(0.4, G * m)
        rng = StableRNG(1)
        for _ in 1:20
            z = draw_assignment(rng, d)
            counts = [count(z[((k - 1) * m + 1):(k * m)]) for k in 1:G]
            @test sort(counts) == sort(vcat(fill(2, 20), fill(6, 20)))
        end
        # uneven group sizes: probabilities use round(α m_g) / m_g
        d2 = TwoStageRandomization([1, 1, 1, 2, 2, 2, 2], [0.5, 1.0], [1, 1])
        @test treatment_probabilities(d2) ≈ [fill((2 / 3 + 1) / 2, 3); fill(0.75, 4)]
        @test_throws ArgumentError TwoStageRandomization(groups, sats, [10, 20])
        @test_throws ArgumentError TwoStageRandomization(groups, [0.2, 1.2], [20, 20])
        @test_throws ArgumentError TwoStageRandomization(groups, [0.2, 0.2], [20, 20])
        @test_throws DimensionMismatch TwoStageRandomization(groups, sats, [40])
        # usable by design-based tools
        p = PartitionStructure(1:(G * m), groups)
        P = exposure_probabilities(p, d; mapping=ExposureMapping(NeighborExposure(:share);
                                                                 cutpoints=[0.4]),
                                   draws=500, rng=StableRNG(2))
        @test P.method === :monte_carlo
    end

    # Heterogeneous potential outcomes under stratified interference.
    rng0 = StableRNG(10)
    base = randn(rng0, G * m) .+ repeat(randn(rng0, G); inner=m)
    po(i, z, a) = base[i] + 1.5z + 2a * (1 - z) + 0.5a * z + 0.3 * z * base[i]
    avg(f) = mean(f(i) for i in 1:(G * m))
    ov(a) = avg(i -> a * po(i, 1, a) + (1 - a) * po(i, 0, a))
    truth = Dict("direct(0.2)" => avg(i -> po(i, 1, 0.2) - po(i, 0, 0.2)),
                 "direct(0.6)" => avg(i -> po(i, 1, 0.6) - po(i, 0, 0.6)),
                 "indirect(0.6 vs 0.2)" => avg(i -> po(i, 0, 0.6) - po(i, 0, 0.2)),
                 "total(0.6 vs 0.2)" => avg(i -> po(i, 1, 0.6) - po(i, 0, 0.2)),
                 "overall(0.6 vs 0.2)" => ov(0.6) - ov(0.2))
    function draw_df(rng)
        d = TwoStageRandomization(groups, sats, [20, 20])
        z = draw_assignment(rng, d)
        # the saturation of each group is recoverable from its treated share
        satg = Dict(gk => count(z[groups .== gk]) == 2 ? 0.2 : 0.6
                    for gk in unique(groups))
        s = [satg[gk] for gk in groups]
        return DataFrame(g=groups, z=Int.(z), s=s,
                         y=[po(i, z[i], s[i]) for i in 1:(G * m)])
    end

    @testset "estimates, hand computation and invariance" begin
        df = draw_df(StableRNG(3))
        r = two_stage_effects(df, :y, :z; group=:g, saturation=:s)
        @test coefnames(r) == ["direct(0.2)", "direct(0.6)", "indirect(0.6 vs 0.2)",
                               "total(0.6 vs 0.2)", "overall(0.6 vs 0.2)"]
        @test dof_residual(r) == 19
        gm = combine(groupby(df, :g), [:y, :z] => ((y, z) -> mean(y[z .== 1])) => :y1,
                     [:y, :z] => ((y, z) -> mean(y[z .== 0])) => :y0, :y => mean => :ya,
                     :s => first => :s)
        lo = gm[gm.s .== 0.2, :]
        hi = gm[gm.s .== 0.6, :]
        @test coef(r)[1] ≈ mean(lo.y1) - mean(lo.y0)
        @test coef(r)[3] ≈ mean(hi.y0) - mean(lo.y0)
        @test coef(r)[4] ≈ mean(hi.y1) - mean(lo.y0)
        @test coef(r)[5] ≈ mean(hi.ya) - mean(lo.ya)
        @test vcov(r)[1, 1] ≈ var(lo.y1 .- lo.y0) / 20
        @test vcov(r)[3, 3] ≈ var(hi.y0) / 20 + var(lo.y0) / 20
        # covariance between direct(0.6) and indirect shares the high-saturation groups
        @test vcov(r)[2, 3] ≈ (cov(hi.y1, hi.y0) - var(hi.y0)) / 20
        r2 = two_stage_effects(df[randperm(StableRNG(4), nrow(df)), :], :y, :z; group=:g,
                               saturation=:s)
        @test coef(r2) ≈ coef(r) && vcov(r2) ≈ vcov(r)
        rr = two_stage_effects(df, :y, :z; group=:g, saturation=:s, reference=0.6)
        @test "indirect(0.2 vs 0.6)" in coefnames(rr)
        @test nrow(r.means) == 2
        @test occursin("Hudgens", sprint(show, MIME"text/plain"(), r))
    end

    @testset "Monte Carlo: unbiased, conservative variance, coverage" begin
        reps = mc_reps(1500, 200)
        names_ = collect(keys(truth))
        est = Dict(k => Float64[] for k in names_)
        cov_ = Dict(k => 0 for k in names_)
        rng = StableRNG(11)
        for rep in 1:reps
            r = two_stage_effects(draw_df(rng), :y, :z; group=:g, saturation=:s)
            ci = confint(r)
            for (j, nm) in enumerate(coefnames(r))
                push!(est[nm], coef(r)[j])
                cov_[nm] += ci[j, 1] <= truth[nm] <= ci[j, 2]
            end
        end
        for nm in names_
            @test abs(mean(est[nm]) - truth[nm]) <= 4 * std(est[nm]) / sqrt(reps)
            @test cov_[nm] / reps >= 0.9
        end
        @info "two_stage_effects coverage" Dict(k => v / reps for (k, v) in cov_) reps
    end

    @testset "errors and undefined quantities" begin
        df = draw_df(StableRNG(5))
        bad = copy(df)
        bad.s[1] = 0.9
        @test_throws ArgumentError two_stage_effects(bad, :y, :z; group=:g, saturation=:s)
        @test_throws ArgumentError two_stage_effects(df, :y, :z; group=:g, saturation=:s,
                                                     reference=0.5)
        one = df[in.(df.g, Ref(["g1", "g2", "g3"])), :]
        one.s .= 0.2
        one = vcat(one, DataFrame(g="g4", z=0, s=0.6, y=1.0))
        @test_throws ArgumentError two_stage_effects(one, :y, :z; group=:g, saturation=:s)
        # saturation 0: no treated units → no direct(0) but indirect/overall exist
        d0 = copy(df)
        d0.z[d0.s .== 0.2] .= 0
        d0.s[d0.s .== 0.2] .= 0.0
        r0 = two_stage_effects(d0, :y, :z; group=:g, saturation=:s)
        @test !("direct(0)" in coefnames(r0)) && "indirect(0.6 vs 0)" in coefnames(r0)
        # some but not all groups of a level lack treated units
        d1 = copy(df)
        d1.z[d1.g .== "g1"] .= 0
        @test_throws ArgumentError two_stage_effects(d1, :y, :z; group=:g, saturation=:s)
        dm = allowmissing(df)
        dm.y[1] = missing
        @test_throws ArgumentError two_stage_effects(dm, :y, :z; group=:g, saturation=:s)
    end
end
