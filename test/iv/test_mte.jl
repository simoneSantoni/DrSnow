# Marginal treatment effects.

const MTE_REF = let r = iv_read_csv(joinpath(IV_VALDIR, "reference_mte.csv"))
    Dict((row.case, row.quantity) => row.value for row in eachrow(r))
end
const MTE_DF = iv_read_csv(joinpath(IV_VALDIR, "mte.csv"))

@testset "marginal treatment effects" begin
    curve_at(r, u) = r.curve.mte[argmin(abs.(r.curve.u .- u))]

    @testset "validation against R (formulas coded from the papers)" begin
        df = MTE_DF
        ps = mte_propensity(df, :d, [:z]; covariates=[:x])
        @test ps.propensity[1] ≈ MTE_REF[("mte", "p1")] rtol = 1e-7
        @test ps.support.lower ≈ MTE_REF[("mte", "lower")] rtol = 1e-7
        @test ps.support.upper ≈ MTE_REF[("mte", "upper")] rtol = 1e-7
        pol = p -> min.(p .+ 0.1, 1.0)
        for (case, m) in (("poly", :polynomial), ("normal", :normal))
            r = mte(df, :y, :d, [:z]; covariates=[:x], method=m, policy=pol,
                    n_bootstrap=20, rng=StableRNG(1))
            @test nobs(r) == MTE_REF[("mte", "n_keep")]
            for (i, nm) in enumerate(("ATE", "ATT", "ATU", "LATE", "PRTE"))
                @test coef(r)[i] ≈ MTE_REF[(case, nm)] rtol = 1e-6
            end
            @test curve_at(r, 0.5) ≈ MTE_REF[(case, "mte_050")] rtol = 1e-6
            @test curve_at(r, 0.2) ≈ MTE_REF[(case, "mte_020")] rtol = 1e-6
        end
        r = mte(df, :y, :d, [:z]; covariates=[:x], bandwidth=0.15,
                residual_bandwidth=0.05, n_bootstrap=20, rng=StableRNG(1))
        @test r.covariate_coef[1] ≈ MTE_REF[("semi", "delta")] rtol = 1e-6
        @test coef(r)[1] ≈ MTE_REF[("semi", "late")] rtol = 1e-6
        @test startswith(only(coefnames(r)), "LATE(")
        for j in (5, 10, 15)
            @test r.curve.mte[j + 1] ≈ MTE_REF[("semi", "mte_grid$j")] rtol = 1e-6
        end
        r0 = mte(df, :y, :d, [:z, :x]; bandwidth=0.15, n_bootstrap=20, rng=StableRNG(1))
        # the propensity is the same (x enters as an instrument instead)
        for j in (5, 10, 15)
            @test r0.curve.mte[j + 1] ≈ MTE_REF[("semi0", "mte_grid$j")] rtol = 1e-6
            @test r0.curve.mte[j + 1] ≈ MTE_REF[("semi0", "locpoly_grid$j")] rtol = 1e-3
        end
    end

    @testset "interface, printing, errors" begin
        df, _ = iv_mte_dgp(StableRNG(401); n=800)
        r = mte(df, :y, :d, [:z]; covariates=[:x], method=:normal, n_bootstrap=20,
                rng=StableRNG(2))
        @test r isa MTEEstimate && r isa CausalEstimate
        @test coefnames(r)[1:3] == ["ATE", "ATT", "ATU"]
        @test size(vcov(r)) == (4, 4)
        @test size(r.bootstrap) == (20, 4)
        s = sprint(show, MIME"text/plain"(), r)
        @test occursin("common support", s) && occursin("bootstrap", s)
        @test length(r.propensity) == nrow(df)
        # same seed, same answer
        r2 = mte(df, :y, :d, [:z]; covariates=[:x], method=:normal, n_bootstrap=20,
                 rng=StableRNG(2))
        @test vcov(r2) == vcov(r)
        rl = mte(df, :y, :d, [:z]; covariates=[:x], method=:polynomial, link=:logit,
                 late_bounds=(0.2, 0.8), n_bootstrap=20, rng=StableRNG(2))
        @test coefnames(rl)[4] == "LATE(0.2, 0.8)"
        ps = mte_propensity(df, :d, [:z]; covariates=[:x])
        @test ps.support.lower < ps.support.upper
        @test nrow(ps.histogram) == 20
        @test sum(ps.histogram.treated) == sum(df.d)
        # a linear probability model predicting outside (0, 1) is an error
        @test_throws ArgumentError mte_propensity(df, :d, [:z]; covariates=[:x],
                                                  link=:lpm)
        dfd, _ = iv_mte_dgp(StableRNG(404); n=800, γ=(0.0, 0.3, 0.0), zdist=:discrete)
        pl = mte_propensity(dfd, :d, [:z]; link=:lpm)
        @test length(unique(skipmissing(pl.propensity))) == 3
        @test_throws ArgumentError mte(df, :y, :d, [:z]; method=:foo)
        @test_throws ArgumentError mte(df, :y, :d, Symbol[])
        @test_throws ArgumentError mte(df, :y, :z, [:x])            # non-binary D
        @test_throws ArgumentError mte(df, :y, :d, [:z]; n_bootstrap=5)
        @test_throws ArgumentError mte(df, :y, :d, [:z]; late_bounds=(0.5, 0.2))
        @test_throws ArgumentError mte(df, :y, :d, [:z]; late_bounds=(0.0, 1.0))
        @test_throws ArgumentError mte(df, :y, :d, [:z]; link=:cauchit)
        @test_throws ArgumentError mte(df, :y, :d, [:z]; method=:polynomial,
                                       policy=p -> p)          # no change in P
        @test_throws ArgumentError mte(df, :y, :d, [:z]; policy=p -> p .+ 2)
        @test_throws ArgumentError mte(df, :y, :d, [:z]; policy=ones(3))
        # row order does not matter
        sh = df[shuffle(StableRNG(3), 1:nrow(df)), :]
        a = mte(df, :y, :d, [:z]; covariates=[:x], method=:polynomial, n_bootstrap=20,
                rng=StableRNG(4))
        b = mte(sh, :y, :d, [:z]; covariates=[:x], method=:polynomial, n_bootstrap=20,
                rng=StableRNG(4))
        @test coef(a) ≈ coef(b) rtol = 1e-8
    end

    @testset "truth recovery (normal selection DGP)" begin
        df, tr = iv_mte_dgp(StableRNG(402); n=6000)
        rn = mte(df, :y, :d, [:z]; covariates=[:x], method=:normal, n_bootstrap=50,
                 rng=StableRNG(5))
        for (i, nm) in enumerate((:ATE, :ATT, :ATU))
            @test abs(coef(rn)[i] - getproperty(tr, nm)) < 4 * stderror(rn)[i]
        end
        for u in (0.2, 0.5, 0.8)
            i = argmin(abs.(rn.curve.u .- u))
            @test abs(rn.curve.mte[i] - tr.mte(mean(df.x), u)) < 4 * rn.curve.se[i]
        end
        rs = mte(df, :y, :d, [:z]; covariates=[:x], n_bootstrap=30, rng=StableRNG(6))
        mid = findall(u -> 0.25 <= u <= 0.75, rs.curve.u)
        err = [rs.curve.mte[i] - tr.mte(mean(df.x), rs.curve.u[i]) for i in mid]
        @test maximum(abs, err) < 0.35
        @test rs.covariate_coef[1] ≈ 0.5 atol = 0.2
    end

    @testset "Monte Carlo: bootstrap coverage (polynomial and normal)" begin
        R = mc_reps(300, 30)
        rng = StableRNG(403)
        cov = zeros(Bool, R, 3)
        for rep in 1:R
            df, tr = iv_mte_dgp(rng; n=1500)
            r = mte(df, :y, :d, [:z]; covariates=[:x], method=:normal, n_bootstrap=99,
                    rng=rng)
            ci = confint(r)
            cov[rep, 1] = ci[1, 1] <= tr.ATE <= ci[1, 2]
            cov[rep, 2] = ci[2, 1] <= tr.ATT <= ci[2, 2]
            cov[rep, 3] = ci[3, 1] <= tr.ATU <= ci[3, 2]
        end
        for j in 1:3
            @test abs(mean(cov[:, j]) - 0.95) < mc_tol(0.95, R; slack=0.05)
        end
    end
end
