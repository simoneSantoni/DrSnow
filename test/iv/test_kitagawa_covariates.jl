# Kitagawa (2015) instrument-validity test with discrete conditioning covariates.

"""Instrument confounded by a discrete covariate `x` that also shifts outcomes and
take-up; `direct` shifts never-takers' outcomes when Z = 1 (exclusion violation);
`nocomp = true` removes compliers (all inequalities binding)."""
function iv_kitagawa_cov_dgp(rng; n=1000, direct=0.0, nocomp=false)
    x = rand(rng, 0:2, n)
    z = Float64.(rand(rng, n) .< ifelse.(x .== 1, 0.75, 0.3))
    u = rand(rng, n)
    at = u .< 0.1 .+ 0.1 .* x
    co = nocomp ? falses(n) : .!at .& (u .< 0.6 .+ 0.1 .* x)
    d = Float64.(at .| (co .& (z .== 1)))
    y = 2.0 .* x .+ randn(rng, n) .+ d .+ direct .* z .* (1 .- d)
    return DataFrame(y=y, d=d, z=z, x=x)
end

@testset "Kitagawa test with covariates" begin
    @testset "moments and interface" begin
        df = iv_kitagawa_cov_dgp(StableRNG(601); n=3000)
        t = instrument_validity_test(df, :y, :d, :z; covariates=[:x], n_bootstrap=199,
                                     rng=StableRNG(1))
        @test t.details.n_cells == 3
        @test occursin("covariates", t.name)
        @test 0 < t.pvalue <= 1
        @test t.details.critical_value_95 > 0
        # the κ moments equal the conditional density differences (brute force)
        cell = df.x .== 1
        s = df[cell, :]
        px = mean(s.z)
        a, b = quantile(df.y, 0.3), quantile(df.y, 0.6)
        g = (a .<= df.y .<= b) .& cell
        κ1 = df.d .* (px .- df.z) ./ (px * (1 - px))
        lhs = mean(κ1 .* g) * nrow(df) / nrow(s)
        inB(v) = (a .<= v .<= b)
        P1 = mean(inB(s.y[s.z .== 1]) .& (s.d[s.z .== 1] .== 1))
        P0 = mean(inB(s.y[s.z .== 0]) .& (s.d[s.z .== 0] .== 1))
        @test lhs ≈ P0 - P1 rtol = 1e-10
        # interval sums helper against brute force
        grid = sort(unique(quantile(df.y, range(0, 1; length=12))))
        V = hcat(κ1, df.d)
        S = DrSnow._iv_kit_interval_sums(df.y, V, grid)
        idx = 0
        for i in 1:length(grid), j in i:length(grid)
            idx += 1
            m = grid[i] .<= df.y .<= grid[j]
            @test S[idx, 1] ≈ sum(κ1[m]) atol = 1e-9
            @test S[idx, 2] ≈ sum(df.d[m]) atol = 1e-9
        end
        # determinism and row-order invariance
        t2 = instrument_validity_test(df[shuffle(StableRNG(2), 1:nrow(df)), :], :y, :d,
                                      :z; covariates=[:x], n_bootstrap=199,
                                      rng=StableRNG(1))
        @test t2.statistic ≈ t.statistic rtol = 1e-10
        # errors: a cell without instrument variation
        bad = copy(df)
        bad.x[bad.z .== 1] .= 5
        @test_throws ArgumentError instrument_validity_test(bad, :y, :d, :z;
                                                            covariates=[:x],
                                                            n_bootstrap=99)
        @test_throws ArgumentError instrument_validity_test(df, :y, :d, :z;
                                                            covariates=[:x],
                                                            max_points=1)
    end

    @testset "Monte Carlo: size (least favourable null), conservativeness, power" begin
        R = mc_reps(500, 100)
        rng = StableRNG(602)
        rej = zeros(3)
        for r in 1:R
            lf = iv_kitagawa_cov_dgp(rng; nocomp=true)
            rej[1] += pvalue(instrument_validity_test(lf, :y, :d, :z; covariates=[:x],
                                                      n_bootstrap=199, max_points=30,
                                                      rng=StableRNG(r))) < 0.05
            ok = iv_kitagawa_cov_dgp(rng)
            rej[2] += pvalue(instrument_validity_test(ok, :y, :d, :z; covariates=[:x],
                                                      n_bootstrap=199, max_points=30,
                                                      rng=StableRNG(r))) < 0.05
            bad = iv_kitagawa_cov_dgp(rng; direct=2.0)
            rej[3] += pvalue(instrument_validity_test(bad, :y, :d, :z; covariates=[:x],
                                                      n_bootstrap=199, max_points=30,
                                                      rng=StableRNG(r))) < 0.05
        end
        rej ./= R
        @test abs(rej[1] - 0.05) < mc_tol(0.05, R; slack=0.02)   # binding null
        @test rej[2] < 0.05 + mc_tol(0.05, R)                     # slack null
        @test rej[3] > 0.7                                        # exclusion violated
    end
end
