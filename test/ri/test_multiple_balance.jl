# Naive Westfall–Young step-down straight from the definition (reference set R
# includes the observed row; larger oriented values are more extreme).
function naive_wy(eo, E, w; method)
    K = length(eo); M = size(E, 1); W = sum(w)
    pv(k, v) = sum(w[b] for b in 1:M if E[b, k] >= v - 1e-9 * max(1, abs(v)); init=0.0) / W
    raw = [pv(k, eo[k]) for k in 1:K]
    order = method === :minp ? sortperm(collect(zip(raw, -eo))) : sortperm(eo; rev=true)
    adj = zeros(K)
    for (j, k) in enumerate(order)
        rest = order[j:end]
        if method === :minp
            q = [minimum(pv(kk, E[b, kk]) for kk in rest) for b in 1:M]
            adj[k] = sum(w[b] for b in 1:M if q[b] <= raw[k] + 1e-12; init=0.0) / W
        else
            u = [maximum(E[b, kk] for kk in rest) for b in 1:M]
            adj[k] = sum(w[b] for b in 1:M if u[b] >= eo[k] - 1e-9 * max(1, eo[k]);
                         init=0.0) / W
        end
    end
    run = 0.0
    for k in order
        run = max(run, adj[k]); adj[k] = min(run, 1.0)
    end
    return raw, adj
end

@testset "Multiple testing" begin
    @testset "Holm and Benjamini–Hochberg match R p.adjust" begin
        R = RI2_REF[:padjust]
        @test holm_adjust(R.p) ≈ R.holm
        @test bh_adjust(R.p) ≈ R.bh
        @test holm_adjust([0.01, 0.04, 0.03]) ≈ [0.03, 0.06, 0.06]
        @test_throws ArgumentError holm_adjust(Float64[])
        @test_throws ArgumentError bh_adjust([0.5, 1.2])
    end

    @testset "Westfall–Young equals the naive definition" begin
        rng = StableRNG(11)
        B, K = 300, 4
        C = [1 0.6 0.3 0; 0.6 1 0.3 0; 0.3 0.3 1 0; 0 0 0 1]
        L = cholesky(C).L
        draws = (L * randn(rng, K, B))'
        obs = [2.4, -1.9, 0.5, 2.2]
        for method in (:minp, :maxt)
            r = westfall_young_adjust(obs, draws; method=method)
            Rfull = vcat(obs', draws)
            raw, adj = naive_wy(abs.(obs), abs.(Rfull), ones(B + 1); method=method)
            @test r.raw ≈ raw
            @test r.adjusted ≈ adj
            @test all(r.adjusted .>= r.raw .- 1e-12)
            # weighted (exact-enumeration) interface with unit weights and the
            # observed row included gives the same answer
            r2 = westfall_young_adjust(obs, Rfull; method=method, weights=ones(B + 1))
            @test r2.adjusted ≈ r.adjusted
        end
        # one-sided alternatives
        r = westfall_young_adjust(obs, draws; alternative=:greater, method=:maxt)
        _, adj = naive_wy(obs, vcat(obs', draws), ones(B + 1); method=:maxt)
        @test r.adjusted ≈ adj
        # identical hypotheses carry no multiplicity penalty
        dup = hcat(draws[:, 1], draws[:, 1])
        r = westfall_young_adjust([obs[1], obs[1]], dup)
        @test r.adjusted ≈ r.raw
        @test_throws DimensionMismatch westfall_young_adjust(obs, draws[:, 1:2])
        @test_throws ArgumentError westfall_young_adjust(obs, draws; method=:bonf)
    end

    @testset "ri_multiple_testing: exact and Monte Carlo" begin
        rng = StableRNG(12)
        n = 10
        z = Bool[1, 1, 0, 1, 0, 0, 1, 0, 1, 0]
        df = DataFrame(d=Int.(z), y1=randn(rng, n) .+ 1.5 .* z, y2=randn(rng, n),
                       y3=randn(rng, n))
        df.y4 = df.y1 .+ 0.1 .* randn(rng, n)
        r = ri_multiple_testing(df, [:y1, :y2, :y3, :y4], :d)
        @test r.exact && r.n_draws == 252
        zs = bf_complete(n, 5)
        for (k, o) in enumerate([:y1, :y2, :y3, :y4])
            @test r.pvalues[k] ≈ bf_pvalue(bf_dim, df[!, o], z, zs)
            @test r.pvalues[k] ≈ randomization_test(df, o, :d).pvalue
        end
        E = hcat([[abs(bf_dim(df[!, o], zz)) for zz in zs] for o in [:y1, :y2, :y3, :y4]]...)
        eo = [abs(bf_dim(df[!, o], z)) for o in [:y1, :y2, :y3, :y4]]
        _, adj = naive_wy(eo, E, fill(1 / 252, 252); method=:minp)
        @test r.adjusted ≈ adj
        @test r.holm ≈ holm_adjust(r.pvalues) && r.bh ≈ bh_adjust(r.pvalues)
        s = sprint(show, MIME"text/plain"(), r)
        @test occursin("p (WY)", s)
        rm = ri_multiple_testing(df, [:y1, :y2], :d; method=:maxt, statistic=:studentized)
        @test all(rm.adjusted .>= rm.pvalues .- 1e-12)
        @test_throws ArgumentError ri_multiple_testing(df, Symbol[], :d)
        @test_throws ArgumentError ri_multiple_testing(df, [:y1, :y1], :d)
        @test_throws ArgumentError ri_multiple_testing(df, [:y1], :d; method=:holm)
        v, w = randomization_distribution(r)
        @test size(v) == (252, 4)
    end

    @testset "family-wise error rate under the global sharp null (Monte Carlo)" begin
        reps = mc_reps(400, 60)
        fw_wy = 0; fw_none = 0
        rng = StableRNG(13)
        for rep in 1:reps
            N = 40
            f = randn(rng, N)
            dd = DataFrame(d=Int.(shuffle(rng, [trues(20); falses(20)])))
            for k in 1:5
                dd[!, Symbol("y$k")] = 0.7 .* f .+ randn(rng, N)
            end
            r = ri_multiple_testing(dd, [Symbol("y$k") for k in 1:5], :d; nperm=199,
                                    rng=StableRNG(rep))
            fw_wy += any(r.adjusted .<= 0.05)
            fw_none += any(r.pvalues .<= 0.05)
        end
        se = sqrt(0.05 * 0.95 / reps)
        @test fw_wy / reps <= 0.05 + 3se
        @info "FWER, 5 correlated outcomes (nominal 0.05)" reps westfall_young =
            fw_wy / reps unadjusted = fw_none / reps
    end
end

@testset "Randomization balance test" begin
    rng = StableRNG(21)
    n = 10
    z = Bool[1, 0, 1, 0, 1, 1, 0, 0, 1, 0]
    df = DataFrame(d=Int.(z), a=randn(rng, n), b=randn(rng, n), c=rand(rng, 0:1, n),
                   s=[1, 1, 1, 1, 2, 2, 2, 2, 2, 2])
    covs = [:a, :b, :c]

    @testset "exact omnibus and per-covariate p-values" begin
        t = ri_balance_test(df, :d, covs)
        @test t isa DiagnosticTest
        zs = bf_complete(n, 5)
        X = Matrix{Float64}(df[:, covs])
        D = [vec(mean(X[zz, :]; dims=1) - mean(X[.!zz, :]; dims=1)) for zz in zs]
        Dm = reduce(hcat, D)'
        mu = vec(mean(Dm; dims=1))
        S = cov(Dm; corrected=false)
        maha(v) = (v - mu)' * pinv(S) * (v - mu)
        dobs = vec(mean(X[z, :]; dims=1) - mean(X[.!z, :]; dims=1))
        mo = maha(dobs)
        @test t.statistic ≈ mo
        @test t.pvalue ≈ mean(maha.(D) .>= mo - 1e-9 * max(1, mo))
        per = t.details.per_covariate
        @test per.difference ≈ dobs
        for (j, c) in enumerate(covs)
            @test per.pvalue[j] ≈ randomization_test(df, c, :d).pvalue
        end
        @test all(per.pvalue_westfall_young .>= per.pvalue .- 1e-12)
        @test t.details.exact && t.details.n_draws == 252
        tz = ri_balance_test(df, :d, covs; statistic=:max_abs_z)
        @test tz.pvalue ≈ mean([maximum(abs.(d - mu) ./ sqrt.(diag(S))) for d in D] .>=
                               maximum(abs.(dobs - mu) ./ sqrt.(diag(S))) - 1e-9)
        ts = ri_balance_test(df, :d, covs; strata=:s)
        @test occursin("stratum", ts.note)
    end

    @testset "honest wording and errors" begin
        t = ri_balance_test(df, :d, covs)
        s = sprint(show, MIME"text/plain"(), t)
        @test occursin("does not show", s)
        @test !occursin(r"balanced\.|appears random|holds"i, s)
        @test_throws ArgumentError ri_balance_test(df, :d, Symbol[])
        @test_throws ArgumentError ri_balance_test(df, :d, covs; statistic=:ml)
        dfs = copy(df); dfs.g = string.(df.c)
        @test_throws ArgumentError ri_balance_test(dfs, :d, [:g])
    end

    @testset "size under complete randomization (Monte Carlo)" begin
        reps = mc_reps(400, 80)
        rej = 0
        rng2 = StableRNG(22)
        for rep in 1:reps
            N = 50
            dd = DataFrame(d=Int.(shuffle(rng2, [trues(25); falses(25)])),
                           a=randn(rng2, N), b=randexp(rng2, N), c=rand(rng2, 0:1, N))
            t = ri_balance_test(dd, :d, [:a, :b, :c]; nperm=199, rng=StableRNG(rep))
            rej += t.pvalue <= 0.05
        end
        se = sqrt(0.05 * 0.95 / reps)
        @test rej / reps <= 0.05 + 3se
        @info "RI balance test size (nominal 0.05)" reps rate = rej / reps
    end
end
