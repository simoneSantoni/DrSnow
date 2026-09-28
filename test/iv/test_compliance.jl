# Population values for iv_binary_dgp(pc = 0.5, pa = 0.2, late = 2, slope = 1):
# complier share varies with sign(x): P(C | x) = 0.8 (0.5 + 0.15 sign x), so
# E[x | C] = 0.15 E|x| / 0.5 and E[x | N] = −0.8 · 0.15 E|x| / 0.4.
const IV_EABSX = sqrt(2 / π)
const IV_POP = (share_c=0.4, share_a=0.2, share_n=0.4,
                xbar_c=0.15 * IV_EABSX / 0.5, xbar_a=0.0,
                xbar_n=-0.8 * 0.15 * IV_EABSX / 0.4,
                late=2.0 + 0.15 * IV_EABSX / 0.5)

@testset "compliance, complier profiles, IPW LATE" begin
    @testset "truth recovery (unconfounded instrument)" begin
        df, truth = iv_binary_dgp(StableRNG(51); n=40_000)
        ca = estimate_compliance(df, :d, :z)
        @test ca isa CausalEstimate
        @test coefnames(ca) == ["compliers", "always_takers", "never_takers"]
        @test sum(coef(ca)) ≈ 1
        pop = [IV_POP.share_c, IV_POP.share_a, IV_POP.share_n]
        @test all(abs.(coef(ca) .- pop) .< 4 .* stderror(ca))
        @test !ca.instrument_reversed
        prof = complier_characteristics(df, :d, :z, [:x, :pre])
        row = prof.table[1, :]
        @test abs(row.complier_mean - IV_POP.xbar_c) < 4 * row.complier_se
        @test abs(row.always_taker_mean - IV_POP.xbar_a) < 4 * row.always_taker_se
        @test abs(row.never_taker_mean - IV_POP.xbar_n) < 4 * row.never_taker_se
        @test row.difference_pvalue < 0.001            # compliers differ in x
        # sample identity: population mean = share-weighted type means
        s = coef(prof.shares)
        @test row.population_mean ≈ s[1] * row.complier_mean + s[2] *
              row.always_taker_mean + s[3] * row.never_taker_mean atol = 0.02
        lw = late_ipw(df, :y, :d, :z)
        r2 = late_2sls(df, :y, :d, :z)
        @test estimate(lw) ≈ estimate(r2) rtol = 1e-10       # Wald = 2SLS
        @test stderror(lw)[1] ≈ stderror(r2)[1] rtol = 1e-3  # IF vs HC1
        @test abs(estimate(lw) - IV_POP.late) < 4 * stderror(lw)[1]
        # E[Y(1)|C] − E[Y(0)|C] = LATE
        @test coef(lw)[2] - coef(lw)[3] ≈ coef(lw)[1] rtol = 1e-10
    end

    @testset "confounded instrument: IPW recovers the unconditional LATE" begin
        df, truth = iv_binary_dgp(StableRNG(52); n=40_000, confounded=true)
        naive = late_2sls(df, :y, :d, :z)
        lw = late_ipw(df, :y, :d, :z; covariates=[:x])
        @test abs(estimate(lw) - IV_POP.late) < 4 * stderror(lw)[1]
        @test abs(estimate(naive) - IV_POP.late) > 8 * stderror(naive)[1]
        ca = estimate_compliance(df, :d, :z; covariates=[:x])
        @test abs(coef(ca)[1] - IV_POP.share_c) < 4 * stderror(ca)[1]
        prof = complier_characteristics(df, :d, :z, [:x]; covariates=[:x])
        @test abs(prof.table.complier_mean[1] - IV_POP.xbar_c) <
              4 * prof.table.complier_se[1]
        @test occursin("logit", prof.method)
        @test occursin("Complier share", sprint(show, MIME"text/plain"(), prof))
        @test occursin("IPW", sprint(show, MIME"text/plain"(), lw))
    end

    @testset "orientation, invariance, clustering" begin
        df, _ = iv_binary_dgp(StableRNG(53); n=3000)
        df.zr = 1 .- df.z
        a = estimate_compliance(df, :d, :z)
        b = estimate_compliance(df, :d, :zr)
        @test b.instrument_reversed
        @test coef(a) ≈ coef(b)
        @test estimate(late_ipw(df, :y, :d, :zr)) ≈ estimate(late_ipw(df, :y, :d, :z))
        perm = randperm(StableRNG(54), nrow(df))
        @test coef(late_ipw(df[perm, :], :y, :d, :z; covariates=[:x])) ≈
              coef(late_ipw(df, :y, :d, :z; covariates=[:x])) rtol = 1e-8
        df.g = rand(StableRNG(55), 1:50, nrow(df))
        lc = late_ipw(df, :y, :d, :z; cluster=:g)
        @test dof_residual(lc) == 49
        @test all(isfinite, stderror(lc))
        cdfs = complier_outcome_distribution(df, :y, :d, :z; points=[0.0, 2.0])
        @test nrow(cdfs) == 2 && all(cdfs.se_treated .> 0)
    end

    @testset "complier outcome distributions" begin
        rng = StableRNG(56)
        df, _ = iv_binary_dgp(rng; n=30_000, slope=0.0, late=1.0)
        pts = [-1.0, 0.0, 1.0, 2.0]
        cdfs = complier_outcome_distribution(df, :y, :d, :z; points=pts)
        # Y(0) | complier = x + e with x | C having density ∝ (0.5 + 0.15 sign x) φ(x)
        # compute the population CDF by simulation
        xs = randn(rng, 2_000_000)
        keep = rand(rng, length(xs)) .< (0.5 .+ 0.15 .* sign.(xs)) ./ 0.65
        y0 = xs[keep] .+ randn(rng, count(keep))
        for (i, p) in enumerate(pts)
            F0 = mean(y0 .<= p)
            F1 = mean(y0 .+ 1.0 .<= p)
            @test abs(cdfs.cdf_untreated[i] - F0) < 4 * cdfs.se_untreated[i]
            @test abs(cdfs.cdf_treated[i] - F1) < 4 * cdfs.se_treated[i]
        end
    end

    @testset "input validation" begin
        df, _ = iv_binary_dgp(StableRNG(57); n=500)
        @test_throws ArgumentError estimate_compliance(df, :x, :z)      # non-binary D
        @test_throws ArgumentError estimate_compliance(df, :d, :x)      # non-binary Z
        df.one = ones(nrow(df))
        @test_throws ArgumentError estimate_compliance(df, :d, :one)
        @test_throws ArgumentError late_ipw(df, :y, :d, :nope)
        @test_throws ArgumentError complier_characteristics(df, :d, :z, Symbol[])
        # zero first stage
        n = 400
        dz = DataFrame(d=repeat([1.0, 0.0], n ÷ 2), z=repeat([1.0, 1.0, 0.0, 0.0], n ÷ 4),
                       y=randn(StableRNG(58), n))
        @test_throws ArgumentError estimate_compliance(dz, :d, :z)
        # perfect prediction of Z by a covariate
        df.zz = df.z
        @test_throws ArgumentError late_ipw(df, :y, :d, :z; covariates=[:zz])
    end

    @testset "Monte Carlo: coverage of IPW LATE, shares and complier means" begin
        R = mc_reps(1500, 250)
        rng = StableRNG(59)
        cover = zeros(4)
        for _ in 1:R
            df, _ = iv_binary_dgp(rng; n=1500, confounded=true)
            lw = late_ipw(df, :y, :d, :z; covariates=[:x])
            ci = confint(lw)
            cover[1] += ci[1, 1] <= IV_POP.late <= ci[1, 2]
            ca = estimate_compliance(df, :d, :z; covariates=[:x])
            cc = confint(ca)
            cover[2] += cc[1, 1] <= IV_POP.share_c <= cc[1, 2]
            prof = complier_characteristics(df, :d, :z, [:x]; covariates=[:x])
            t = prof.table
            cover[3] += abs(t.complier_mean[1] - IV_POP.xbar_c) <= 1.96 * t.complier_se[1]
            cover[4] += abs(t.never_taker_mean[1] - IV_POP.xbar_n) <=
                        1.96 * t.never_taker_se[1]
        end
        cover ./= R
        for j in 1:4
            @test abs(cover[j] - 0.95) < mc_tol(0.95, R; slack=0.015)
        end
    end
end

@testset "complier_characteristics warns on impossible subgroup means" begin
    rng = StableRNG(77)
    n = 4000
    x = Float64.(rand(rng, n) .< 0.5)
    # instrument correlated with x (not as good as random unconditionally)
    z = Float64.(rand(rng, n) .< ifelse.(x .== 1, 0.9, 0.1))
    d = Float64.(rand(rng, n) .< 0.3 .+ 0.1 .* z)
    df = DataFrame(d=d, z=z, x=x)
    @test_logs (:warn, r"outside the observed range") match_mode=:any complier_characteristics(df, :d, :z, [:x])
end
