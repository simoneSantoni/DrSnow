# MTE bounds (Mogstad, Santos & Torgovitsky 2018) and the shared LP solver.
# Reference: test/validation/iv/make_reference_mtebounds.R (ivmte with lpSolveAPI).

const IV_REF_MTEB = let r = iv_read_csv(joinpath(IV_VALDIR, "reference_mtebounds.csv"))
    Dict((row.case, row.quantity) => row.value for row in eachrow(r))
end
const AE_COLL = iv_read_csv(joinpath(IV_VALDIR, "ae_collapsed.csv"))
const IVMTE_SIM = iv_read_csv(joinpath(IV_VALDIR, "ivmte_sim.csv"))

"""ivmte's audit grid in u: 0, 1 and the first 25 points of the base-2 Halton
sequence (rounded to 8 digits as in ivmte)."""
function ivmte_audit_grid(n=25)
    h = map(1:n) do j
        f, r, i = 1.0, 0.0, j
        while i > 0
            f /= 2
            r += f * (i % 2)
            i ÷= 2
        end
        round(r; digits=8)
    end
    return sort(vcat(0.0, 1.0, h))
end

@testset "MTE bounds" begin
    @testset "LP solver (core): primal and dual forms agree" begin
        rng = StableRNG(901)
        for _ in 1:100
            n = rand(rng, 2:6)
            m = rand(rng, n:30)
            me = rand(rng, 0:2)
            A = randn(rng, m, n)
            x0 = randn(rng, n)
            b = A * x0 .+ rand(rng, m)
            Ae = randn(rng, me, n)
            be = Ae * x0
            A = vcat(A, Matrix(1.0I, n, n), -Matrix(1.0I, n, n))
            b = vcat(b, fill(10.0, 2n))
            c = randn(rng, n)
            for mx in (true, false)
                s1, o1, _ = DrSnow._lp_free(c, A, b, Ae, be; maximize=mx)
                s2, o2, x2 = DrSnow._lp_free_dual(c, A, b, Ae, be; maximize=mx)
                @test s1 === s2 === :optimal
                @test o2 ≈ o1 rtol = 1e-8 atol = 1e-9
                @test maximum(A * x2 .- b) < 1e-8
                me > 0 && @test maximum(abs.(Ae * x2 .- be)) < 1e-8
            end
        end
        @test DrSnow._lp_free_dual([1.0], reshape([-1.0], 1, 1), [0.0], zeros(0, 1),
                                   Float64[]; maximize=true)[1] === :unbounded
        @test DrSnow._lp_free_dual([1.0], reshape([1.0, -1.0], 2, 1), [0.0, -1.0],
                                   zeros(0, 1), Float64[]; maximize=true)[1] === :infeasible
        # the historical DiD aliases point to the shared solver
        @test DrSnow._did_simplex === DrSnow._lp_simplex
    end

    @testset "validation against ivmte" begin
        grid = ivmte_audit_grid()
        ref(case, q) = IV_REF_MTEB[(case, q)]
        function check(case, b; rtol=1e-6)
            @test b.lower ≈ ref(case, "lower") rtol = rtol atol = 1e-7
            @test b.upper ≈ ref(case, "upper") rtol = rtol atol = 1e-7
            @test b.min_criterion ≈ ref(case, "criterion") rtol = 1e-4 atol = 1e-9
        end
        aeiv = (regressors=[:morekids, :samesex, (:morekids, :samesex)],)
        for tg in (:att, :ate)
            b = mte_bounds(AE_COLL, :worked, :morekids, :samesex; target=tg,
                           covariates=[:yob], basis=:bernstein, degree=1, ivlike=aeiv,
                           weights=:count, u_grid=grid)
            check("ae_linear_$(tg)", b)
            @test b.nobs == nrow(AE_COLL)
        end
        b = mte_bounds(AE_COLL, :worked, :morekids, :samesex; target=:att,
                       basis=:spline, degree=2, knots=[1 / 3, 2 / 3],
                       m0_monotone=:increasing, m1_monotone=:increasing,
                       mte_monotone=:decreasing, ivlike=aeiv, weights=:count,
                       u_grid=grid)
        check("ae_spline_mono", b)
        simiv = (regressors=[:d, :z, (:d, :z)],)
        b = mte_bounds(IVMTE_SIM, :y, :d, :z; target=:late, late_from=(z=1,),
                       late_to=(z=3,), covariates=[:x], basis=:bernstein, degree=3,
                       ivlike=simiv, u_grid=grid)
        check("sim_late", b)
        b = mte_bounds(IVMTE_SIM, :y, :d, :z; target=:ate, covariates=[:x],
                       basis=:spline, degree=1, knots=[0.25, 0.5, 0.75],
                       ivlike=[(regressors=[:z1, :z2, :z3, :x],),
                               (regressors=[:d, :x],),
                               (regressors=[:d], instruments=[:z])], u_grid=grid)
        check("sim_multi", b)
        @test nrow(b.moments) == 5 + 3 + 2
        b = mte_bounds(IVMTE_SIM, :y, :d, :z; target=:genlate, u_interval=(0.2, 0.4),
                       covariates=[:x], basis=:bernstein, degree=3, ivlike=simiv,
                       u_grid=grid)
        check("sim_genlate", b)
        b = mte_bounds(IVMTE_SIM, :y, :d, [:z1, :z2, :z3]; target=:ate, basis=:spline,
                       degree=0, knots=[0.2, 0.4, 0.6, 0.8],
                       m0_monotone=:decreasing, m1_monotone=:decreasing,
                       ivlike=(regressors=[:d, :z1, :z2, :z3, (:d, :z1), (:d, :z2),
                                           (:d, :z3)],), u_grid=grid)
        check("sim_const_dec", b; rtol=1e-5)        # lp_solve is less precise here
        @test b.min_criterion > 0.1
        b = mte_bounds(IVMTE_SIM, :y, :d, :z; target=:atu, link=:probit,
                       covariates=[:x], interact=[:x], interact_degree=1,
                       basis=:bernstein, degree=2,
                       ivlike=(regressors=[:d, :z1, :z2, :z3, :x],), u_grid=grid)
        check("sim_atu_probit", b)
        @test occursin("MTE bounds", sprint(show, MIME"text/plain"(), b))
    end

    @testset "closed forms: Manski-type bounds and point identification" begin
        rng = StableRNG(902)
        n = 4000
        z = Float64.(rand(rng, n) .< 0.5)
        u = rand(rng, n)
        p0, p1 = 0.3, 0.7
        d = Float64.(u .< ifelse.(z .== 1, p1, p0))
        y = Float64.(rand(rng, n) .< 0.3 .+ 0.4 .* u .+ 0.2 .* d)
        df = DataFrame(y=y, d=d, z=z)
        ph0, ph1 = mean(d[z .== 0]), mean(d[z .== 1])
        sat = (regressors=[:d, :z, (:d, :z)],)
        # piecewise-constant MTRs with knots at the propensities: the nonparametric
        # (Heckman–Vytlacil) bounds on the ATE for a binary outcome
        b = mte_bounds(df, :y, :d, :z; target=:ate, basis=:spline, degree=0,
                       knots=[ph0, ph1], ivlike=sat)
        EY1 = mean((y .* d)[z .== 1])            # ∫₀^{p1} m₁
        EY0 = mean((y .* (1 .- d))[z .== 0])     # ∫_{p0}^1 m₀
        @test b.min_criterion < 1e-9
        @test b.lower ≈ EY1 + (1 - ph1) * 0 - (EY0 + ph0 * 1) atol = 1e-8
        @test b.upper ≈ EY1 + (1 - ph1) * 1 - (EY0 + ph0 * 0) atol = 1e-8
        # the LATE for the instrument change is point identified by the Wald ratio
        wald = (mean(y[z .== 1]) - mean(y[z .== 0])) / (ph1 - ph0)
        for (bs, dg, kn) in ((:bernstein, 3, Float64[]), (:spline, 0, [ph0, ph1]))
            bl = mte_bounds(df, :y, :d, :z; target=:late, late_from=(z=0,),
                            late_to=(z=1,), basis=bs, degree=dg, knots=kn,
                            ivlike=(regressors=[:d], instruments=[:z]))
            @test bl.point_identified
            @test bl.lower ≈ wald rtol = 1e-7
            @test bl.upper ≈ wald rtol = 1e-7
        end
        # linear MTRs (the true ones: m₀ = 0.3 + 0.4u, m₁ = 0.5 + 0.4u) are point
        # identified by the four saturated estimands: ATE = 0.2
        bl1 = mte_bounds(df, :y, :d, :z; target=:ate, basis=:bernstein, degree=1,
                         ivlike=sat)
        @test bl1.point_identified
        @test abs(bl1.lower - 0.2) < 0.08
        ml = bl1.mtr_coef.lower
        @test length(ml.m0) == 2 && length(bl1.mtr_terms) == 2
        @test ml.m1[1] - ml.m0[1] ≈ bl1.lower atol = 0.1   # MTE at u = 0 ≈ ATE here
        # shape restrictions and more IV-like estimands tighten the bounds
        b2 = mte_bounds(df, :y, :d, :z; target=:ate, basis=:bernstein, degree=2,
                        ivlike=sat)
        b3 = mte_bounds(df, :y, :d, :z; target=:ate, basis=:bernstein, degree=2,
                        ivlike=sat, m0_monotone=:increasing, m1_monotone=:increasing)
        @test b3.lower >= b2.lower - 1e-9 && b3.upper <= b2.upper + 1e-9
        b4 = mte_bounds(df, :y, :d, :z; target=:ate, basis=:bernstein, degree=2)
        @test b4.lower <= b2.lower + 1e-9 && b4.upper >= b2.upper - 1e-9
        # PRTE of a uniform propensity increase equals a genlate-type average
        pr = mte_bounds(df, :y, :d, :z; target=:prte, basis=:spline, degree=0,
                        knots=[ph0, ph1], ivlike=sat,
                        policy_propensity=p -> min.(p .+ 0.1, 1.0))
        @test pr.lower <= pr.upper
        @test b.lower - 1e-8 <= b.upper
        # the implied IV-like estimands match the estimates when the criterion is 0
        @test b.moments.implied_lower ≈ b.moments.estimate atol = 1e-8
        # the bounds do not depend on the row order
        sh = df[shuffle(StableRNG(903), 1:n), :]
        bs_ = mte_bounds(sh, :y, :d, :z; target=:ate, basis=:bernstein, degree=2,
                         ivlike=sat)
        @test bs_.lower ≈ b2.lower rtol = 1e-8
        @test bs_.upper ≈ b2.upper rtol = 1e-8
    end

    @testset "errors" begin
        df = IVMTE_SIM
        @test_throws ArgumentError mte_bounds(df, :y, :d, :z; target=:bogus)
        @test_throws ArgumentError mte_bounds(df, :y, :d, :z; target=:late)
        @test_throws ArgumentError mte_bounds(df, :y, :d, :z; target=:genlate)
        @test_throws ArgumentError mte_bounds(df, :y, :d, :z; target=:genlate,
                                              u_interval=(0.5, 0.2))
        @test_throws ArgumentError mte_bounds(df, :y, :d, :z; target=:prte)
        @test_throws ArgumentError mte_bounds(df, :y, :d, :z; basis=:spline,
                                              knots=[1.5])
        @test_throws ArgumentError mte_bounds(df, :y, :d, :z; basis=:bogus)
        @test_throws ArgumentError mte_bounds(df, :y, :d, :z; m0_monotone=:up)
        @test_throws ArgumentError mte_bounds(df, :y, :d, :z; interact=[:x])
        @test_throws ArgumentError mte_bounds(df, :y, :d, :z;
                                              ivlike=(instruments=[:z],))
        @test_throws ArgumentError mte_bounds(df, :y, :d, :z;
                                              ivlike=(regressors=[:d, :x],
                                                      instruments=[:z]))
        @test_throws ArgumentError mte_bounds(df, :y, :d, :z;
                                              ivlike=(regressors=[:d],
                                                      components=[:bogus]))
        @test_throws ArgumentError mte_bounds(df, :y, :d, :z; mtr_bounds=(1, 0))
        # infeasible shape restrictions
        @test_throws ArgumentError mte_bounds(df, :y, :d, :z; mte_range=(5, 6))
        @test_throws ArgumentError mte_bounds(df, :y, :x, :z)      # non-binary D
    end
end
