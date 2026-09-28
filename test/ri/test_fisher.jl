@testset "Fisher randomization test" begin
    rng = StableRNG(101)
    n = 10
    z = Bool[1, 0, 1, 1, 0, 0, 1, 0, 1, 0]
    y = randn(rng, n) .+ 0.8 .* z
    x = randn(rng, n, 2)
    df = DataFrame(y=y, d=Int.(z), x1=x[:, 1], x2=x[:, 2], s=[1, 1, 1, 1, 2, 2, 2, 2, 2, 2])

    @testset "exact p-values equal brute-force enumeration" begin
        zs = bf_complete(n, 5)
        stats = [(:diff_means, bf_dim), (:studentized, bf_neyman_t), (:rank_sum, bf_rank),
                 (:lin, bf_lin(x))]
        for (s, f) in stats, alt in (:two_sided, :greater, :less)
            cov = s === :lin ? [:x1, :x2] : Symbol[]
            r = randomization_test(df, :y, :d; statistic=s, alternative=alt,
                                   covariates=cov)
            @test r.exact
            @test r.n_draws == 252
            @test r.pvalue ≈ bf_pvalue(f, y, z, zs; alternative=alt) atol = 1e-12
            @test r.observed ≈ f(y, z)
        end
        rk = randomization_test(df, :y, :d; statistic=:ks)
        @test rk.alternative === :greater
        @test rk.pvalue ≈ bf_pvalue(bf_ks, y, z, zs; alternative=:greater) atol = 1e-12
        # user-supplied statistic
        med = (yy, zz) -> median(yy[zz]) - median(yy[.!zz])
        ru = randomization_test(df, :y, :d; statistic=med)
        @test ru.pvalue ≈ bf_pvalue(med, y, z, zs) atol = 1e-12
        # constant additive null via adjusted outcomes
        for tau in (-0.5, 0.3, 1.7)
            r = randomization_test(df, :y, :d; tau0=tau, statistic=:rank_sum)
            @test r.pvalue ≈ bf_pvalue(bf_rank, y .- tau .* z, z, zs) atol = 1e-12
        end
        df.tau = fill(0.3, n)
        @test randomization_test(df, :y, :d; tau0=:tau).pvalue ≈
              randomization_test(df, :y, :d; tau0=0.3).pvalue
        # the reference set is the support; weights are probabilities
        v, w = randomization_distribution(randomization_test(df, :y, :d))
        @test length(v) == 252 && sum(w) ≈ 1
    end

    @testset "stratified, cluster, pairs and Bernoulli designs" begin
        st = df.s
        r = randomization_test(df, :y, :d; strata=:s, statistic=:diff_means)
        zs = bf_stratified(st, z)
        @test r.n_draws == length(zs)
        @test r.pvalue ≈ bf_pvalue(bf_strat_dim(st), y, z, zs) atol = 1e-12
        # cluster randomization
        cl = [1, 1, 2, 2, 3, 3, 4, 4, 5, 5]
        zc = Bool[1, 1, 0, 0, 1, 1, 0, 0, 0, 0]
        dc = DataFrame(y=y, d=Int.(zc), c=cl)
        rc = randomization_test(dc, :y, :d; cluster=:c)
        @test rc.n_draws == 10
        @test rc.pvalue ≈ bf_pvalue(bf_dim, y, zc, bf_cluster(cl, 2)) atol = 1e-12
        rcs = randomization_test(dc, :y, :d; cluster=:c, statistic=:studentized)
        @test 0 < rcs.pvalue <= 1
        # matched pairs: stratified with one treated per pair (paired variance)
        pr = [1, 1, 2, 2, 3, 3, 4, 4, 5, 5]
        zp = Bool[1, 0, 0, 1, 1, 0, 0, 1, 1, 0]
        dp = DataFrame(y=y, d=Int.(zp), p=pr)
        m = MatchedPairsRandomization(pr)
        r1 = randomization_test(dp, :y, :d; mechanism=m, statistic=:studentized)
        r2 = randomization_test(dp, :y, :d; strata=:p, statistic=:studentized)
        @test r1.pvalue ≈ r2.pvalue
        dpair = [y[2p - 1] - y[2p] for p in 1:5] .* [zp[2p - 1] ? 1 : -1 for p in 1:5]
        tpair = mean(dpair) / (std(dpair) / sqrt(5))
        @test r1.observed ≈ tpair
        @test r1.n_draws == 32
        # Bernoulli with unequal probabilities: probability-weighted enumeration
        p = [0.3, 0.6, 0.5, 0.5, 0.4, 0.7, 0.5, 0.2]
        yb = y[1:8]; zb = Bool[1, 1, 0, 1, 0, 1, 0, 0]
        mb = BernoulliAssignment(8, p)
        rb = randomization_test(DataFrame(y=yb, d=Int.(zb)), :y, :d; mechanism=mb)
        allz = bf_assignments(8, [[i] for i in 1:8], _ -> true)
        wz = [prod(zz[i] ? p[i] : 1 - p[i] for i in 1:8) for zz in allz]
        @test rb.exact
        @test rb.pvalue ≈ bf_pvalue(bf_dim, yb, zb, allz, wz) atol = 1e-12
        @test rb.n_dropped == 2        # all-treated and all-control assignments
    end

    @testset "agreement with ri2 (exact enumeration)" begin
        R = RI2_REF[:complete]
        d = DataFrame(y=R.y, d=Int.(R.z))
        r = randomization_test(d, :y, :d)
        @test r.n_draws == R.n_perm
        @test r.observed ≈ R.estimate
        @test r.pvalue ≈ R.p_two
        @test randomization_test(d, :y, :d; alternative=:greater).pvalue ≈ R.p_upper
        @test randomization_test(d, :y, :d; alternative=:less).pvalue ≈ R.p_lower
        @test randomization_test(d, :y, :d; tau0=0.7).pvalue ≈ R.p_two_h07
        @test randomization_test(d, :y, :d; tau0=0.7, alternative=:greater).pvalue ≈
              R.p_upper_h07
        rs = randomization_test(d, :y, :d; statistic=:studentized)
        @test rs.observed ≈ R.t_stud          # Neyman t = HC2 t
        @test rs.pvalue ≈ R.p_two_stud

        R = RI2_REF[:blocked]
        d = DataFrame(y=R.y, d=Int.(R.z), b=Int.(R.block))
        r = randomization_test(d, :y, :d; strata=:b)
        @test r.n_draws == R.n_perm
        @test r.observed ≈ R.estimate
        @test r.pvalue ≈ R.p_two

        R = RI2_REF[:blocked_unequal]
        d = DataFrame(y=R.y, d=Int.(R.z), b=Int.(R.block))
        r = randomization_test(d, :y, :d; strata=:b, nperm=10_000)
        @test r.exact && r.n_draws == R.n_perm
        @test r.observed ≈ R.estimate
        @test r.pvalue ≈ R.p_two
        @test randomization_test(d, :y, :d; strata=:b, nperm=10_000,
                                 alternative=:greater).pvalue ≈ R.p_upper
        @test randomization_test(d, :y, :d; strata=:b, nperm=10_000,
                                 tau0=-0.3).pvalue ≈ R.p_two_h
        @test randomization_test(d, :y, :d; strata=:b, nperm=10_000, tau0=-0.3,
                                 alternative=:greater).pvalue ≈ R.p_upper_h

        R = RI2_REF[:cluster]
        d = DataFrame(y=R.y, d=Int.(R.z), c=Int.(R.cluster))
        r = randomization_test(d, :y, :d; cluster=:c)
        @test r.n_draws == R.n_perm
        @test r.observed ≈ R.estimate
        @test r.pvalue ≈ R.p_two
        @test randomization_test(d, :y, :d; cluster=:c, alternative=:less).pvalue ≈
              R.p_lower

        R = RI2_REF[:block_cluster]
        d = DataFrame(y=R.y, d=Int.(R.z), b=Int.(R.block), c=Int.(R.cluster))
        r = randomization_test(d, :y, :d; cluster=:c, strata=:b)
        @test r.n_draws == R.n_perm
        @test r.observed ≈ R.estimate
        @test r.pvalue ≈ R.p_two
    end

    @testset "rank-sum equals the exact Wilcoxon test (R wilcox.test)" begin
        R = RI2_REF[:wilcoxon]
        d = DataFrame(y=R.y, d=Int.(R.z))
        @test randomization_test(d, :y, :d; statistic=:rank_sum).pvalue ≈ R.p_two
        @test randomization_test(d, :y, :d; statistic=:rank_sum,
                                 alternative=:greater).pvalue ≈ R.p_greater
    end

    @testset "Monte Carlo: convention, reproducibility, invariance" begin
        rng2 = StableRNG(5)
        N = 60
        big = DataFrame(y=randn(rng2, N), d=Int.(1:N .<= 30), id=1:N,
                        s=repeat(1:3, 20))
        r = randomization_test(big, :y, :d; nperm=999, rng=StableRNG(1))
        @test !r.exact && r.n_draws == 999
        v, w = randomization_distribution(r)
        @test length(v) == 1000 && v[1] == r.observed && all(==(1.0), w)
        k = count(abs.(v[2:end]) .>= abs(r.observed) - 1e-9)
        @test r.pvalue ≈ (1 + k) / 1000
        @test r.mc_se ≈ sqrt(r.pvalue * (1 - r.pvalue) / 999)
        r2 = randomization_test(big, :y, :d; nperm=999, rng=StableRNG(1))
        @test r2.pvalue == r.pvalue && r2.distribution == r.distribution
        r3 = randomization_test(big, :y, :d; nperm=999, rng=StableRNG(2))
        @test r3.distribution != r.distribution
        @test randomization_test(big, :y, :d; nperm=999, rng=StableRNG(1),
                                 threaded=true).distribution == r.distribution
        # row order does not matter (design built from columns, or fixed by id)
        sh = big[randperm(StableRNG(9), N), :]
        for kw in ((;), (strata=:s,), (id=:id,))
            a = randomization_test(big, :y, :d; nperm=500, rng=StableRNG(3), kw...)
            b = randomization_test(sh, :y, :d; nperm=500, rng=StableRNG(3), kw...)
            @test a.pvalue == b.pvalue
            @test sort(a.distribution) == sort(b.distribution)
        end
        # exact results are invariant to row order for any statistic
        sh10 = df[randperm(StableRNG(4), n), :]
        for s in (:studentized, :rank_sum, :ks)
            @test randomization_test(sh10, :y, :d; statistic=s).pvalue ≈
                  randomization_test(df, :y, :d; statistic=s).pvalue
        end
        # exact = true / false are honored
        @test !randomization_test(df, :y, :d; exact=false, nperm=50,
                                  rng=StableRNG(1)).exact
        @test randomization_test(big, :y, :d; strata=:s, exact=false, nperm=10,
                                 rng=StableRNG(1)).n_draws == 10
        @test randomization_test(df, :y, :d; exact=true, nperm=10).exact
    end

    @testset "rerandomization inference uses the accepted set" begin
        X = hcat(df.x1, df.x2)
        m = Rerandomization(CompleteRandomization(n, 5), X;
                            threshold=balance_mahalanobis(X, z) + 1e-9)
        r = randomization_test(df, :y, :d; mechanism=m)
        acc = [zz for zz in bf_complete(n, 5) if balance_mahalanobis(X, zz) <=
                                                  balance_mahalanobis(X, z) + 1e-9]
        @test r.n_draws == length(acc)
        @test r.pvalue ≈ bf_pvalue(bf_dim, y, z, acc)
        tight = Rerandomization(CompleteRandomization(n, 5), X;
                                threshold=balance_mahalanobis(X, z) / 2)
        @test_throws ArgumentError randomization_test(df, :y, :d; mechanism=tight)
    end

    @testset "results and wording" begin
        r = randomization_test(df, :y, :d)
        s = sprint(show, MIME"text/plain"(), r)
        @test occursin("exact p-value", s)
        @test !occursin(r"robust|holds|appears random"i, s)
        t = DiagnosticTest(r)
        @test t isa DiagnosticTest && t.pvalue == r.pvalue
        @test DrSnow.pvalue(r) == r.pvalue
        @test rejects(r; alpha=1.0)
        @test occursin("RandomizationTestResult", sprint(show, r))
    end

    @testset "errors" begin
        @test_throws ArgumentError randomization_test(df, :nope, :d)
        @test_throws ArgumentError randomization_test(df, :y, :d; alternative=:both)
        @test_throws ArgumentError randomization_test(df, :y, :d; statistic=:median)
        @test_throws ArgumentError randomization_test(df, :y, :d; covariates=[:x1])
        @test_throws ArgumentError randomization_test(df, :y, :d; statistic=:ks,
                                                      alternative=:less)
        bad = copy(df); bad.d = fill(1, n)
        @test_throws ArgumentError randomization_test(bad, :y, :d)
        bad.d = [2; zeros(Int, n - 1)]
        @test_throws ArgumentError randomization_test(bad, :y, :d)
        mis = allowmissing(copy(df)); mis.y[1] = missing
        @test_throws ArgumentError randomization_test(mis, :y, :d)
        @test_throws DimensionMismatch randomization_test(df, :y, :d;
                                                          mechanism=CompleteRandomization(8, 4))
        @test_throws ArgumentError randomization_test(df, :y, :d;
                                                      mechanism=CompleteRandomization(n, 4))
        @test_throws ArgumentError randomization_test(df, :y, :d; strata=:s,
                                                      mechanism=CompleteRandomization(n, 5))
        cm = CustomAssignment(n, r -> shuffle(r, z))
        @test_throws ArgumentError randomization_test(df, :y, :d; mechanism=cm, exact=true)
        @test !randomization_test(df, :y, :d; mechanism=cm, nperm=50,
                                  rng=StableRNG(1)).exact
        @test_throws ArgumentError randomization_test(df, :y, :d; exact=:yes)
        @test_throws ArgumentError randomization_test(df, :y, :d; nperm=0)
        dc = DataFrame(y=y, d=Int.(z), c=[1, 1, 2, 2, 3, 3, 4, 4, 5, 5])
        @test_throws ArgumentError randomization_test(dc, :y, :d; cluster=:c)
        dupid = copy(df); dupid.id = [1; 1:(n - 1)]
        @test_throws ArgumentError randomization_test(dupid, :y, :d; id=:id)
    end
end
