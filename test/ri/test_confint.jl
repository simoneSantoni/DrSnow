@testset "Confidence intervals by test inversion" begin
    rng = StableRNG(202)
    n = 12
    z = Bool[1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0]
    x = randn(rng, n)
    y = x .+ 1.0 .* z .+ 0.5 .* randn(rng, n)
    df = DataFrame(y=y, d=Int.(z), x=x, s=repeat([1, 2], 6))

    # duality: end points separate rejected and non-rejected sharp nulls
    function check_duality(ci, kw; eps=1e-6)
        a = (1 - ci.level) / 2
        pg(t) = randomization_test(df, :y, :d; tau0=t, alternative=:greater, kw...).pvalue
        pl(t) = randomization_test(df, :y, :d; tau0=t, alternative=:less, kw...).pvalue
        @test pg(ci.lower - eps) <= a
        @test pg(ci.lower + eps) > a && pl(ci.lower + eps) > a
        @test pl(ci.upper + eps) <= a
        @test pg(ci.upper - eps) > a && pl(ci.upper - eps) > a
    end

    @testset "exact inversion for linear statistics" begin
        for (s, cov) in ((:diff_means, Symbol[]), (:lin, [:x]))
            ci = ri_confint(df, :y, :d; statistic=s, covariates=cov, level=0.9)
            @test ci.exact && ci.n_draws == 924
            @test ci.contiguous
            @test occursin("exact inversion", ci.method)
            check_duality(ci, (statistic=s, covariates=cov))
        end
        # stratified design, exact
        ci = ri_confint(df, :y, :d; strata=:s, level=0.8)
        check_duality(ci, (strata=:s,))
        # Hodges–Lehmann-type estimate of the difference in means under complete
        # randomization is the difference in means itself
        ci = ri_confint(df, :y, :d)
        @test ci.estimate ≈ mean(y[z]) - mean(y[.!z])
    end

    @testset "bisection for non-linear statistics" begin
        for s in (:studentized, :rank_sum)
            ci = ri_confint(df, :y, :d; statistic=s, level=0.9, tol=1e-9)
            @test occursin("bisection", ci.method)
            check_duality(ci, (statistic=s,); eps=1e-5)
        end
        ks = ri_confint(df, :y, :d; statistic=:ks, level=0.8, tol=1e-9)
        @test ks.estimate === nothing
        @test ks.lower < mean(y[z]) - mean(y[.!z]) < ks.upper
    end

    @testset "Wilcoxon interval and Hodges–Lehmann estimate match R" begin
        R = RI2_REF[:wilcoxon]
        d = DataFrame(y=R.y, d=Int.(R.z))
        ci = ri_confint(d, :y, :d; statistic=:rank_sum, level=R.ci_level, tol=1e-10)
        @test ci.estimate ≈ R.hl atol = 1e-7
        @test ci.lower ≈ R.ci_lower atol = 1e-7
        @test ci.upper ≈ R.ci_upper atol = 1e-7
        # HL = median of all treated-minus-control differences
        yt = R.y[R.z .== 1]; yc = R.y[R.z .== 0]
        @test ci.estimate ≈ median([a - b for a in yt for b in yc]) atol = 1e-7
    end

    @testset "one-sided bounds" begin
        lo = ri_confint(df, :y, :d; alternative=:greater, level=0.95)
        @test lo.upper == Inf && isfinite(lo.lower)
        @test randomization_test(df, :y, :d; tau0=lo.lower - 1e-6,
                                 alternative=:greater).pvalue <= 0.05
        @test randomization_test(df, :y, :d; tau0=lo.lower + 1e-6,
                                 alternative=:greater).pvalue > 0.05
        hi = ri_confint(df, :y, :d; alternative=:less, statistic=:rank_sum, tol=1e-9)
        @test hi.lower == -Inf && isfinite(hi.upper)
        @test_throws ArgumentError ri_confint(df, :y, :d; statistic=:ks,
                                              alternative=:greater)
    end

    @testset "Monte Carlo reference set and printing" begin
        rng2 = StableRNG(8)
        N = 80
        d = DataFrame(y=randn(rng2, N), d=Int.(1:N .<= 40))
        d.y .+= 2.0 .* d.d
        a = ri_confint(d, :y, :d; nperm=999, rng=StableRNG(1))
        b = ri_confint(d, :y, :d; nperm=999, rng=StableRNG(1))
        @test !a.exact && a.lower == b.lower && a.upper == b.upper
        @test a.lower < 2 < a.upper
        s = sprint(show, MIME"text/plain"(), a)
        @test occursin("95% interval", s)
        @test confint(a) == (a.lower, a.upper)
        # tiny reference set: the test can never reject at 5%, so the set is ℝ
        tiny = DataFrame(y=[1.0, 2.0, 3.0, 5.0], d=[1, 0, 1, 0])
        t = ri_confint(tiny, :y, :d)
        @test t.lower == -Inf && t.upper == Inf
        @test_throws ArgumentError ri_confint(d, :y, :d; level=1.2)
    end

    @testset "coverage under a constant effect (Monte Carlo)" begin
        reps = mc_reps(400, 60)
        cover = 0
        cover_rank = 0
        rng3 = StableRNG(303)
        for r in 1:reps
            N = 30
            y0 = randn(rng3, N) .^ 2
            zz = shuffle(rng3, [trues(12); falses(18)])
            dd = DataFrame(y=y0 .+ 1.5 .* zz, d=Int.(zz))
            c = ri_confint(dd, :y, :d; nperm=199, rng=StableRNG(r), hodges_lehmann=false)
            cover += c.lower <= 1.5 <= c.upper
            cr = ri_confint(dd, :y, :d; statistic=:rank_sum, nperm=199,
                            rng=StableRNG(r), hodges_lehmann=false, tol=1e-4)
            cover_rank += cr.lower <= 1.5 <= cr.upper
        end
        se = sqrt(0.95 * 0.05 / reps)
        @test cover / reps >= 0.95 - 3se
        @test cover_rank / reps >= 0.95 - 3se
        @info "RI CI coverage (constant effect, nominal 0.95)" reps diff_means =
            cover / reps rank_sum = cover_rank / reps
    end
end
