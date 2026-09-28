@testset "Randomization inference for regression coefficients" begin
    FEM = DrSnow.FixedEffectModels
    rng = StableRNG(404)
    G = 8
    sizes = [3, 5, 4, 6, 3, 4, 5, 4]
    cl = reduce(vcat, [fill(g, s) for (g, s) in enumerate(sizes)])
    N = length(cl)
    zc = Bool[1, 0, 1, 0, 0, 1, 0, 0]
    d = Int.(zc[cl])
    x = randn(rng, N)
    y = 0.4 .* d .+ x .+ randn(rng, G)[cl] .+ randn(rng, N)
    y2 = randn(rng, N)
    df = DataFrame(y=y, y2=y2, d=d, x=x, c=cl, b=Int.(cl .> 4))

    @testset "exact enumeration equals brute-force refitting" begin
        r = ri_regression(df, [:y, :y2], :d; covariates=[:x], cluster=:c)
        @test r.exact && r.n_draws == binomial(G, 3)
        f = make_formula(:y, [:d, :x])
        m = FEM.reg(df, f, Vcov.cluster(:c))
        i = findfirst(==("d"), coefnames(m))
        @test r.table.estimate[1] ≈ coef(m)[i]
        @test r.table.std_error[1] ≈ stderror(m)[i]
        @test r.table.t[1] ≈ coef(m)[i] / stderror(m)[i]
        # brute force over all assignments of 3 of 8 clusters
        bs = Float64[]; ts = Float64[]
        for zz in bf_cluster(cl, 3)
            dd = copy(df); dd.d = Int.(zz)
            mm = FEM.reg(dd, f, Vcov.cluster(:c))
            push!(bs, coef(mm)[i]); push!(ts, coef(mm)[i] / stderror(mm)[i])
        end
        b0 = coef(m)[i]; t0 = b0 / stderror(m)[i]
        @test r.table.p_randomization_c[1] ≈ mean(abs.(bs) .>= abs(b0) - 1e-9)
        @test r.table.p_randomization_t[1] ≈ mean(abs.(ts) .>= abs(t0) - 1e-9)
        # a single binary treatment through an explicit mechanism gives the same
        mech = ClusterRandomization(df.c, 3)
        r2 = ri_regression(df, :y, :d; covariates=[:x], mechanism=mech,
                           vcov=Vcov.cluster(:c))
        @test r2.table.p_randomization_t[1] ≈ r.table.p_randomization_t[1]
        @test r.omnibus !== nothing && isempty(r.joint)
        @test r.table.p_westfall_young[1] >= r.table.p_randomization_t[1] - 1e-12
        @test 0 <= r.omnibus.pvalue <= 1
        @test pvalues(r) == r.table.p_randomization_t
        @test coefnames(r) == ["y: d", "y2: d"]
        @test nobs(r) == N
    end

    @testset "blocked clusters, fixed effects, row-order invariance" begin
        r = ri_regression(df, :y, :d; cluster=:c, strata=:b, fe=[:b], nperm=5_000)
        @test r.exact
        @test occursin("blocked cluster", r.mechanism)
        sh = df[randperm(StableRNG(2), N), :]
        rs = ri_regression(sh, :y, :d; cluster=:c, strata=:b, fe=[:b], nperm=5_000)
        @test rs.table.p_randomization_t ≈ r.table.p_randomization_t
        # Monte Carlo is also invariant (canonical unit order)
        a = ri_regression(df, :y, :d; cluster=:c, nperm=60, exact=false, rng=StableRNG(4))
        b = ri_regression(sh, :y, :d; cluster=:c, nperm=60, exact=false, rng=StableRNG(4))
        @test a.table.p_randomization_t == b.table.p_randomization_t
        @test a.distribution == b.distribution
    end

    @testset "multiple treatment arms (label permutation) and joint tests" begin
        rng2 = StableRNG(6)
        n = 90
        arm = shuffle(rng2, repeat(0:2, 30))
        dm = DataFrame(a1=Int.(arm .== 1), a2=Int.(arm .== 2), x=randn(rng2, n))
        dm.y = 0.8 .* dm.a1 .+ dm.x .+ randn(rng2, n)
        r = ri_regression(dm, :y, [:a1, :a2]; covariates=[:x], nperm=300, rng=StableRNG(1))
        @test !r.exact && r.n_draws == 300
        @test nrow(r.table) == 2 && length(r.joint) == 1 && r.omnibus === nothing
        @test r.joint[1].dof == (2,)
        m = FEM.reg(dm, make_formula(:y, [:a1, :a2, :x]), Vcov.robust())
        idx = [findfirst(==(s), coefnames(m)) for s in ("a1", "a2")]
        @test r.joint[1].statistic ≈ wald_test(coef(m)[idx], vcov(m)[idx, idx]).chi2
        # permuted draws keep arm sizes fixed: the Wald distribution has no NaN rows
        @test r.n_dropped == 0
        r2 = ri_regression(dm, :y, [:a1, :a2]; covariates=[:x], nperm=300,
                           rng=StableRNG(1))
        @test r2.distribution == r.distribution
        s = sprint(show, MIME"text/plain"(), r)
        @test occursin("p RI-t", s) && occursin("Joint", s)
        @test_throws ArgumentError ri_regression(dm, :y, [:a1, :a2]; exact=true)
        @test_throws ArgumentError ri_regression(dm, :y, [:a1, :a2];
                                                 mechanism=CompleteRandomization(n, 30))
    end

    @testset "errors" begin
        @test_throws ArgumentError ri_regression(df, :y, :nope)
        @test_throws ArgumentError ri_regression(df, :y, :d; covariates=[:d])
        @test_throws ArgumentError ri_regression(df, :y, :d; alternative=:bad)
        bad = copy(df); bad.d[1] = 1 - bad.d[1]
        @test_throws ArgumentError ri_regression(bad, :y, :d; cluster=:c)
        # treatment absorbed by cluster fixed effects is not identified
        @test_throws ErrorException ri_regression(df, :y, :d; fe=[:c], cluster=:c)
    end

    @testset "size under cluster randomization with few clusters (Monte Carlo)" begin
        reps = mc_reps(300, 40)
        rej_ri = 0; rej_conv = 0
        rng3 = StableRNG(505)
        for rep in 1:reps
            Gs = 10
            nsz = [2, 2, 2, 3, 3, 3, 4, 5, 20, 30]      # unequal cluster sizes
            cc = reduce(vcat, [fill(g, s) for (g, s) in enumerate(nsz)])
            zz = shuffle(rng3, [trues(3); falses(7)])
            u = randn(rng3, Gs)[cc] .* 2 .+ randn(rng3, length(cc))
            dd = DataFrame(y=u, d=Int.(zz[cc]), c=cc)
            r = ri_regression(dd, :y, :d; cluster=:c, nperm=200)   # exact: 120
            rej_ri += r.table.p_randomization_t[1] <= 0.05
            rej_conv += r.table.p_conventional[1] <= 0.05
        end
        se = sqrt(0.05 * 0.95 / reps)
        @test rej_ri / reps <= 0.05 + 3se
        @info "ri_regression size, 10 clusters (nominal 0.05)" reps randomization_t =
            rej_ri / reps conventional_cluster_t = rej_conv / reps
    end
end
