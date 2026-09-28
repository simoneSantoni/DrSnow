# k-class (LIML, Fuller, HLIM, HFUL) and jackknife IV (JIVE1, JIVE2, UJIVE), and the
# Mikusheva–Sun jackknife AR test.

using DrSnow: FixedEffectModels

@testset "k-class and jackknife IV" begin
    zs(k) = [Symbol("z", j) for j in 1:k]

    @testset "special cases: κ = 1 is 2SLS, κ = 0 is OLS" begin
        df = iv_linear_dgp(StableRNG(101); n=700, k=3, G=20)
        df.f = rand(StableRNG(102), 1:5, 700)
        df.w = 0.5 .+ rand(StableRNG(103), 700)
        for kw in ((vcov=Vcov.simple(),), NamedTuple(), (cluster=:g,))
            r1 = kclass_iv(df, :y, :d, zs(3); method=:kclass, kappa=1.0,
                           covariates=[:x], fe=[:f], weights=:w, kw...)
            ts = iv_regression(df, :y, :d, zs(3); covariates=[:x], fe=[:f], weights=:w,
                               kw...)
            @test coef(r1)[1] ≈ coef(ts)[1] rtol = 1e-9
            @test stderror(r1)[1] ≈ stderror(ts)[1] rtol = 1e-7
            @test dof_residual(r1) == dof_residual(ts)
        end
        r0 = kclass_iv(df, :y, :d, zs(3); method=:kclass, kappa=0.0, covariates=[:x],
                       vcov=Vcov.simple())
        ols = FixedEffectModels.reg(df, make_formula(:y, [:d, :x]))
        @test coef(r0)[1] ≈ coef(ols)[2] rtol = 1e-9
        # just identified: LIML = 2SLS (κ_LIML = 1)
        rj = kclass_iv(df, :y, :d, :z1; covariates=[:x])
        @test rj.kappa ≈ 1.0 atol = 1e-10
        @test coef(rj)[1] ≈ coef(late_2sls(df, :y, :d, :z1; covariates=[:x]))[1] rtol = 1e-8
    end

    @testset "interface, printing and errors" begin
        df = iv_linear_dgp(StableRNG(104); n=400, k=4)
        r = kclass_iv(df, :y, :d, zs(4); method=:fuller)
        @test r isa KClassEstimate && r isa CausalEstimate
        @test coefnames(r) == ["d"]
        @test nobs(r) == 400
        @test occursin("Fuller", sprint(show, MIME"text/plain"(), r))
        @test occursin("constant-effects", estimand(r))
        @test occursin("HFUL", method_name(kclass_iv(df, :y, :d, zs(4); method=:hful)))
        @test_throws ArgumentError kclass_iv(df, :y, :d, zs(4); method=:foo)
        @test_throws ArgumentError kclass_iv(df, :y, :d, zs(4); method=:kclass)
        @test_throws ArgumentError kclass_iv(df, :y, :d, zs(4); kappa=0.5)
        @test_throws ArgumentError kclass_iv(df, :y, :d, zs(4); se=:bekker, method=:hlim)
        @test_throws ArgumentError kclass_iv(df, :y, :d, zs(4); se=:many_robust)
        @test_throws ArgumentError kclass_iv(df, :y, :d, zs(4); method=:hful, cluster=:g)
        @test_throws ArgumentError kclass_iv(df, :y, :d, zs(4); method=:hful,
                                             se=:standard)
        @test_throws ArgumentError kclass_iv(df, :y, :d, zs(4); fuller_alpha=0)
        rj = jive(df, :y, :d, zs(4))
        @test rj isa JIVEEstimate
        @test method_name(rj) == "UJIVE"
        @test occursin("UJIVE", sprint(show, MIME"text/plain"(), rj))
        @test estimand(rj) == estimand(rj.tsls)
        @test_throws ArgumentError jive(df, :y, :d, zs(4); method=:jive3)
        @test_throws ArgumentError jive(df, :y, :d, zs(4); se=:bekker)
        @test_throws ArgumentError jive(df, :y, :d, zs(4); method=:jive1, se=:many_robust)
        @test_throws ArgumentError jive(df, :y, :d, zs(4); se=:many_robust, cluster=:g)
        @test_throws ArgumentError weak_iv_test(rj; method=:jackknife_ar, beta0=[1, 2])
        @test_throws ArgumentError weak_iv_test(late_2sls(df, :y, :d, zs(4); cluster=:g);
                                                method=:jackknife_ar)
        # a single-observation instrument category has leverage one
        df.cat = [i == 1 ? 1.0 : 0.0 for i in 1:400]
        @test_throws ArgumentError jive(df, :y, :d, [:z1, :cat])
    end

    @testset "row-shuffling invariance" begin
        df = iv_linear_dgp(StableRNG(105); n=500, k=6, G=15, hetero=true)
        df.f = rand(StableRNG(106), 1:4, 500)
        sh = df[shuffle(StableRNG(107), 1:500), :]
        for f in (d -> kclass_iv(d, :y, :d, zs(6); covariates=[:x], fe=[:f]),
                  d -> kclass_iv(d, :y, :d, zs(6); method=:hful, fe=[:f]),
                  d -> jive(d, :y, :d, zs(6); covariates=[:x], fe=[:f, :g]),
                  d -> jive(d, :y, :d, zs(6); se=:many_robust, fe=[:f]))
            a, b = f(df), f(sh)
            @test coef(a) ≈ coef(b) rtol = 1e-9
            @test stderror(a) ≈ stderror(b) rtol = 1e-7
        end
    end

    @testset "JIVE1 with category dummies = leave-one-out mean instrument" begin
        rng = StableRNG(108)
        n, J = 900, 30
        j = rand(rng, 1:J, n)
        lev = randn(rng, J)
        v = randn(rng, n)
        d = 0.8 .* lev[j] .+ v
        y = 1.0 .* d .+ 0.5 .* v .+ randn(rng, n)
        df = DataFrame(y=y, d=d, j=j)
        for q in 2:J
            df[!, Symbol("j", q)] = Float64.(j .== q)
        end
        dums = [Symbol("j", q) for q in 2:J]
        sums = Dict(q => sum(d[j .== q]) for q in 1:J)
        cnts = Dict(q => count(==(q), j) for q in 1:J)
        df.loo = [(sums[j[i]] - d[i]) / (cnts[j[i]] - 1) for i in 1:n]
        r = jive(df, :y, :d, dums; method=:jive1)
        t = late_2sls(df, :y, :d, :loo)
        @test coef(r)[1] ≈ coef(t)[1] rtol = 1e-9
        @test stderror(r)[1] ≈ stderror(t)[1] rtol = 1e-6
        # with a fixed effect, UJIVE with dummies is invariant to including the FE
        # levels among the controls or letting them be absorbed
        df.s = rand(StableRNG(109), 1:3, n)
        a = jive(df, :y, :d, dums; fe=[:s])
        for q in 2:3
            df[!, Symbol("s", q)] = Float64.(df.s .== q)
        end
        b = jive(df, :y, :d, dums; covariates=[:s2, :s3])
        @test coef(a)[1] ≈ coef(b)[1] rtol = 1e-9
    end

    @testset "many weak instruments: bias of 2SLS vs LIML / UJIVE" begin
        R = mc_reps(400, 60)
        rng = StableRNG(110)
        est = zeros(R, 5)
        for rep in 1:R
            df = iv_linear_dgp(rng; n=400, k=30, pi=0.08, beta=1.0, rho=0.8)
            est[rep, 1] = coef(late_2sls(df, :y, :d, zs(30)))[1]
            est[rep, 2] = coef(kclass_iv(df, :y, :d, zs(30)))[1]
            est[rep, 3] = coef(kclass_iv(df, :y, :d, zs(30); method=:fuller))[1]
            est[rep, 4] = coef(jive(df, :y, :d, zs(30)))[1]
            est[rep, 5] = coef(kclass_iv(df, :y, :d, zs(30); method=:hful))[1]
        end
        med = [median(est[:, j]) for j in 1:5]
        @test med[1] - 1 > 0.15                   # 2SLS biased toward OLS
        @test all(abs.(med[2:5] .- 1) .< 0.12)
    end

    @testset "Monte Carlo coverage with many instruments" begin
        R = mc_reps(1000, 150)
        rng = StableRNG(111)
        cov = zeros(Bool, R, 5)
        size_jar = zeros(Bool, R)
        for rep in 1:R
            # homoskedastic many instruments: Bekker SEs for LIML / Fuller
            df = iv_linear_dgp(rng; n=500, k=40, pi=0.1, beta=1.0, rho=0.5)
            c(r) = (ci = confint(r); ci[1, 1] <= 1.0 <= ci[1, 2])
            cov[rep, 1] = c(kclass_iv(df, :y, :d, zs(40); se=:bekker))
            cov[rep, 2] = c(kclass_iv(df, :y, :d, zs(40); method=:fuller, se=:bekker))
            # heteroskedastic many instruments: HFUL / UJIVE robust SEs, jackknife AR
            dh = iv_linear_dgp(rng; n=500, k=40, pi=0.1, beta=1.0, rho=0.5, hetero=true)
            cov[rep, 3] = c(kclass_iv(dh, :y, :d, zs(40); method=:hful))
            cov[rep, 4] = c(kclass_iv(dh, :y, :d, zs(40); method=:hlim))
            cov[rep, 5] = c(jive(dh, :y, :d, zs(40); se=:many_robust))
            size_jar[rep] = rejects(weak_iv_test(late_2sls(dh, :y, :d, zs(40));
                                                 method=:jackknife_ar, beta0=1.0))
        end
        for j in 1:5
            @test abs(mean(cov[:, j]) - 0.95) < mc_tol(0.95, R; slack=0.03)
        end
        @test mean(size_jar) < 0.05 + mc_tol(0.05, R; slack=0.02)
    end

    @testset "jackknife AR: power and confidence set" begin
        df = iv_linear_dgp(StableRNG(112); n=800, k=30, pi=0.15, beta=1.0, hetero=true)
        r = late_2sls(df, :y, :d, zs(30))
        t0 = weak_iv_test(r; method=:jackknife_ar, beta0=0.0)
        @test rejects(t0)
        @test occursin("Mikusheva", t0.name)
        cs = weak_iv_confidence_set(r; method=:jackknife_ar)
        @test 1.0 in cs
        @test !(0.0 in cs)
        @test DrSnow.pvalue(cs, 1.0) > 0.05
    end
end
