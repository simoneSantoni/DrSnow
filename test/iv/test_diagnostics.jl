"""Naive Kitagawa statistic by explicit enumeration of intervals (test oracle)."""
function iv_kitagawa_naive(y, d, z, grid, ξ)
    y1, d1 = y[z .== 1], d[z .== 1]
    y0, d0 = y[z .== 0], d[z .== 0]
    m, n = length(y1), length(y0)
    λ = m / (m + n)
    best = -Inf
    for a in eachindex(grid), b in a:length(grid)
        lo, hi = grid[a], grid[b]
        for (dd, sgn) in ((1, 1), (0, -1))
            P = count(i -> lo <= y1[i] <= hi && d1[i] == dd, eachindex(y1)) / m
            Q = count(i -> lo <= y0[i] <= hi && d0[i] == dd, eachindex(y0)) / n
            σ = sqrt(λ * P * (1 - P) + (1 - λ) * Q * (1 - Q))
            best = max(best, sgn * (Q - P) / max(ξ, σ))
        end
    end
    return sqrt(m * n / (m + n)) * best
end

@testset "IV assumption diagnostics" begin
    @testset "instrument_balance" begin
        df, _ = iv_binary_dgp(StableRNG(61); n=2000, confounded=true)
        t = instrument_balance(df, :z, [:x, :pre])
        @test t isa DiagnosticTest
        @test rejects(t)                                  # Z depends on x
        t2 = instrument_balance(df, :z, [:pre]; covariates=[:x])
        @test t2.dof == (1, t2.dof[2])
        @test nrow(t2.details.table) == 1
        @test_throws ArgumentError instrument_balance(df, :z, Symbol[])
        @test occursin("Non-rejection is not evidence",
                       sprint(show, MIME"text/plain"(), t2))
    end

    @testset "first_stage_sign_test" begin
        rng = StableRNG(62)
        n = 4000
        grp = rand(rng, ["a", "b", "c", "d"], n)
        z = Float64.(rand(rng, n) .< 0.5)
        # group d: defiers dominate (instrument lowers take-up)
        slope = ifelse.(grp .== "d", -0.3, 0.4)
        d = Float64.(rand(rng, n) .< 0.3 .+ slope .* z .+ 0.3 .* (grp .== "d"))
        df = DataFrame(d=d, z=z, grp=grp, tiny=vcat(fill("x", 10), fill("y", n - 10)))
        t = first_stage_sign_test(df, :d, :z, [:grp, :tiny])
        @test rejects(t)
        @test "tiny = x" in t.details.skipped
        @test nrow(t.details.table) == 5
        @test all(t.details.table.pvalue_holm .>= t.details.table.pvalue_one_sided)
        @test_throws ArgumentError first_stage_sign_test(df, :d, :z, Symbol[])
    end

    @testset "zero_first_stage_test" begin
        rng = StableRNG(63)
        n = 3000
        elig = rand(rng, n) .< 0.6
        z = Float64.(rand(rng, n) .< 0.5)
        d = Float64.(elig .& (z .== 1) .& (rand(rng, n) .< 0.7))
        y0 = randn(rng, n)
        df = DataFrame(y=y0 .+ 2 .* d, ydirect=y0 .+ 2 .* d .+ 0.3 .* z, d=d, z=z,
                       inel=.!elig)
        t = zero_first_stage_test(df, :y, :d, :z; subset=:inel)
        @test t.details.first_stage_F < 1e-8                # no compliers
        @test !rejects(t; alpha=0.001)
        t2 = zero_first_stage_test(df, :ydirect, :d, :z; subset=df.inel)
        @test rejects(t2)
        @test t2.details.reduced_form[1] ≈ 0.3 atol = 0.15
        @test_throws ArgumentError zero_first_stage_test(df, :y, :d, :z;
                                                         subset=falses(n))
    end

    @testset "overidentification and endogeneity tests: errors and labels" begin
        df = iv_linear_dgp(StableRNG(64); n=500, k=2)
        r1 = late_2sls(df, :y, :d, :z1)
        @test_throws ArgumentError overidentification_test(r1)
        @test occursin("Hansen", overidentification_test(late_2sls(df, :y, :d,
                                                                     [:z1, :z2])).name)
        rs = late_2sls(df, :y, :d, [:z1, :z2]; vcov=Vcov.simple())
        @test occursin("Sargan", overidentification_test(rs).name)
        @test_throws ArgumentError endogeneity_test(r1; endogenous=[:x])
        @test rejects(endogeneity_test(late_2sls(iv_linear_dgp(StableRNG(65); n=3000,
                                                               rho=0.6), :y, :d, :z1)))
    end

    @testset "Monte Carlo: size (and power) of regression-based tests" begin
        R = mc_reps(2000, 300)
        rng = StableRNG(66)
        rej = zeros(6)
        for _ in 1:R
            # valid instruments, heteroskedastic, clustered
            df = iv_linear_dgp(rng; n=400, k=3, pi=0.4, rho=0.5, hetero=true, G=40)
            df.pre1 = randn(rng, 400)
            df.pre2 = df.pre1 .+ randn(rng, 400)
            r = late_2sls(df, :y, :d, [:z1, :z2, :z3]; covariates=[:x], cluster=:g)
            rej[1] += rejects(overidentification_test(r))
            rs = late_2sls(df, :y, :d, [:z1, :z2, :z3]; covariates=[:x],
                           vcov=Vcov.simple())
            dfh = iv_linear_dgp(rng; n=400, k=3, pi=0.4, rho=0.5)
            rh = late_2sls(dfh, :y, :d, [:z1, :z2, :z3]; covariates=[:x],
                           vcov=Vcov.simple())
            rej[2] += rejects(overidentification_test(rh))
            rej[3] += rejects(instrument_balance(df, [:z1, :z2], [:pre1, :pre2];
                                                 cluster=:g))
            # exogenous regressor: DWH size
            dfe = iv_linear_dgp(rng; n=400, pi=0.5, rho=0.0, hetero=true)
            rej[4] += rejects(endogeneity_test(late_2sls(dfe, :y, :d, :z1;
                                                         covariates=[:x])))
            # first-stage sign test under monotonicity (all subgroups positive)
            n = 800
            g = rand(rng, 1:4, n)
            z = Float64.(rand(rng, n) .< 0.5)
            dd = Float64.(rand(rng, n) .< 0.3 .+ 0.02 .* z)
            rej[5] += rejects(first_stage_sign_test(DataFrame(d=dd, z=z, g=g), :d, :z,
                                                    [:g]))
            # power of the overidentification test: one invalid instrument
            dfi = iv_linear_dgp(rng; n=400, k=2, pi=0.5, gamma=0.4)
            rej[6] += rejects(overidentification_test(late_2sls(dfi, :y, :d, [:z1, :z2])))
        end
        rej ./= R
        @test abs(rej[1] - 0.05) < mc_tol(0.05, R; slack=0.03)   # Hansen J, 40 clusters
        @test abs(rej[2] - 0.05) < mc_tol(0.05, R)               # Sargan
        @test abs(rej[3] - 0.05) < mc_tol(0.05, R; slack=0.03)   # balance, clustered
        @test abs(rej[4] - 0.05) < mc_tol(0.05, R)               # DWH, robust
        @test rej[5] <= 0.05 + mc_tol(0.05, R)                   # sign test (conservative)
        @test rej[6] > 0.8                                       # J power
    end

    @testset "Kitagawa instrument-validity test" begin
        rng = StableRNG(67)
        df, _ = iv_binary_dgp(rng; n=300)
        y, d, z = df.y, df.d .== 1, df.z
        grid = sort(unique(y))
        t = instrument_validity_test(df, :y, :d, :z; n_bootstrap=99, max_points=10_000,
                                     rng=StableRNG(1))
        @test t.statistic ≈ iv_kitagawa_naive(y, d, z, grid, 0.07) rtol = 1e-10
        t1 = instrument_validity_test(df, :y, :d, :z; trimming=1.0, n_bootstrap=99,
                                      max_points=10_000, rng=StableRNG(1))
        @test t1.statistic ≈ iv_kitagawa_naive(y, d, z, grid, 1.0) rtol = 1e-10
        # reproducible given the rng; invariant to orientation of Z
        t2 = instrument_validity_test(df, :y, :d, :z; n_bootstrap=99, rng=StableRNG(2))
        t3 = instrument_validity_test(df, :y, :d, :z; n_bootstrap=99, rng=StableRNG(2))
        @test t2.pvalue == t3.pvalue
        df.zr = 1 .- df.z
        t4 = instrument_validity_test(df, :y, :d, :zr; n_bootstrap=99, rng=StableRNG(2))
        @test t4.statistic ≈ t2.statistic
        @test_throws ArgumentError instrument_validity_test(df, :y, :x, :z)
        @test_throws ArgumentError instrument_validity_test(df, :y, :d, :z; n_bootstrap=10)

        R = mc_reps(400, 60)
        rej_valid, rej_invalid = 0, 0
        for _ in 1:R
            dv, _ = iv_binary_dgp(rng; n=600, pc=0.3, slope=0.0)
            rej_valid += rejects(instrument_validity_test(dv, :y, :d, :z;
                                                          n_bootstrap=199,
                                                          max_points=40, rng=rng))
            # exclusion violation: Z shifts the outcome of never-takers by 2 SD
            di, _ = iv_binary_dgp(rng; n=600, pc=0.3, slope=0.0)
            nt = (di.d .== 0)
            di.y .+= 2.0 .* di.z .* nt
            rej_invalid += rejects(instrument_validity_test(di, :y, :d, :z;
                                                            n_bootstrap=199,
                                                            max_points=40, rng=rng))
        end
        @test rej_valid / R <= 0.05 + mc_tol(0.05, R)
        @test rej_invalid / R > 0.7
    end
end
