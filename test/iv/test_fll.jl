# Frandsen, Lefgren & Leslie (2023) test of exclusion and monotonicity in judge
# designs: fit component (spline + judge-indicator Wald test), slope component
# (Andrews & Soares GMS), weighted Bonferroni.

"""Monotonicity violation: half of the judges treat the cases with the *highest*
resistance `U` (`D = 1{U > 1 − λ}`) instead of the lowest, and effects vary with `U`,
so judges with similar propensities have different mean outcomes."""
function fll_mono_dgp(rng; J=40, cases=60)
    λ = 0.2 .+ 0.5 .* rand(rng, J)
    rev = isodd.(1:J)
    judge = repeat(1:J; inner=cases)
    U = rand(rng, J * cases)
    D = Float64.(ifelse.(rev[judge], U .> 1 .- λ[judge], U .< λ[judge]))
    Y = D .* (1 .+ 3 .* (U .- 0.5)) .+ randn(rng, J * cases)
    return DataFrame(y=Y, d=D, judge=judge)
end

@testset "FLL judge-design test" begin
    @testset "quadratic B-spline basis" begin
        kn = [0.1, 0.3, 0.35, 0.6, 0.9]
        x = collect(range(0.1, 0.9; length=201))
        S, dS = DrSnow._iv_bspline(x, kn, 2)
        @test size(S, 2) == length(kn) + 1
        @test maximum(abs.(sum(S; dims=2) .- 1)) < 1e-12       # partition of unity
        @test maximum(abs.(sum(dS; dims=2))) < 1e-10
        @test all(>=(-1e-14), S)
        h = 1e-5
        Sp, _ = DrSnow._iv_bspline(x[2:(end - 1)] .+ h, kn, 2)
        Sm, _ = DrSnow._iv_bspline(x[2:(end - 1)] .- h, kn, 2)
        @test maximum(abs.((Sp .- Sm) ./ (2h) .- dS[2:(end - 1), :])) < 1e-3
        # FLL's slope-at-knot formula equals the spline derivative at the knots
        δ = randn(StableRNG(801), length(kn) + 1)
        _, dk = DrSnow._iv_bspline(kn, kn, 2)
        tt = vcat(kn[1], kn, kn[end])
        sl = [2 * (δ[l + 1] - δ[l]) / (tt[l + 2] - tt[l]) for l in 1:length(kn)]
        @test dk * δ ≈ sl rtol = 1e-10
        # reproduces a quadratic exactly
        f(v) = 1 + 2v - 3v^2
        c = S \ f.(x)
        @test S * c ≈ f.(x) rtol = 1e-10
    end

    @testset "statistic, details and combination" begin
        df = iv_judge_dgp(StableRNG(802); n_courts=6, judges=8, cases=60, slope=0.5)
        t = judge_validity_test(df, :y, :d, :judge; strata=[:court], rng=StableRNG(1))
        dt = t.details
        @test t.statistic == dt.fit_statistic
        @test dt.fit_dof == 48 - 6 - 5          # J − (m + 1) − (courts − 1)
        @test dt.fit_pvalue ≈ DrSnow.ccdf(DrSnow.Chisq(dt.fit_dof), dt.fit_statistic)
        @test t.pvalue ≈ min(1.0, dt.fit_pvalue / 0.9, dt.slope_pvalue / 0.1)
        @test length(dt.slopes) == 5 && length(dt.knots) == 5
        @test nrow(dt.judge_means) == 48
        # γ̂ is orthogonal to the spline and strata directions (identically)
        jm = dt.judge_means
        @test abs(sum(jm.n .* jm.gamma)) < 1e-8 * sum(jm.n)
        t1 = judge_validity_test(df, :y, :d, :judge; strata=[:court], rng=StableRNG(1),
                                 omega=1.0)
        @test t1.pvalue ≈ dt.fit_pvalue
        t0 = judge_validity_test(df, :y, :d, :judge; strata=[:court], rng=StableRNG(1),
                                 omega=0.0)
        @test t0.pvalue ≈ dt.slope_pvalue
        # an implausibly tight outcome range makes the slope component reject
        dfb = iv_judge_dgp(StableRNG(805); n_courts=6, judges=8, cases=150, tau=3.0)
        tb = judge_validity_test(dfb, :y, :d, :judge; strata=[:court], rng=StableRNG(1),
                                 omega=0.0, n_knots=2, outcome_bounds=(0.0, 0.05))
        @test tb.details.slope_statistic > 0
        @test tb.pvalue < 0.05
        @test occursin("Frandsen", t.name)
        # errors
        @test_throws ArgumentError judge_validity_test(df, :y, :d, :judge; n_knots=1)
        @test_throws ArgumentError judge_validity_test(df, :y, :d, :judge; omega=1.5)
        @test_throws ArgumentError judge_validity_test(df, :y, :d, :judge; n_knots=60)
        @test_throws ArgumentError judge_validity_test(df, :y, :d, :judge; method=:x)
        @test_throws ArgumentError judge_validity_test(df, :y, :d, :judge;
                                                       n_simulations=10)
    end

    @testset "Monte Carlo: size and power" begin
        R = mc_reps(500, 80)
        rng = StableRNG(803)
        rej = zeros(R, 4)
        for rep in 1:R
            ok = iv_judge_dgp(rng; n_courts=6, judges=8, cases=60, slope=0.5)
            t = judge_validity_test(ok, :y, :d, :judge; strata=[:court], rng=rng,
                                    n_simulations=999)
            rej[rep, 1] = t.details.fit_pvalue < 0.05
            rej[rep, 2] = t.pvalue < 0.05
            ex = iv_judge_dgp(rng; n_courts=6, judges=8, cases=60, direct=0.3)
            rej[rep, 3] = pvalue(judge_validity_test(ex, :y, :d, :judge;
                                                     strata=[:court], rng=rng,
                                                     n_simulations=999)) < 0.05
            mono = fll_mono_dgp(rng)
            rej[rep, 4] = pvalue(judge_validity_test(mono, :y, :d, :judge; rng=rng,
                                                     n_simulations=999)) < 0.05
        end
        m = vec(mean(rej; dims=1))
        @test m[1] < 0.05 + mc_tol(0.05, R; slack=0.02)
        @test m[2] < 0.05 + mc_tol(0.05, R; slack=0.02)
        @test m[3] > 0.6          # exclusion violated
        @test m[4] > 0.6          # monotonicity violated with heterogeneous effects
    end
end
