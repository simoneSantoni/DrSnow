function seq_rule(r)
    r.rule == "obf" && return OBFSpending()
    r.rule == "pocock" && return PocockSpending()
    r.rule == "power" && return PowerSpending(r.par)
    r.rule == "hsd" && return HSDSpending(r.par)
    r.rule == "obrien_fleming" && return :obrien_fleming
    r.rule == "pocock_classic" && return :pocock
    return :haybittle_peto
end
seq_futility(r) = r.futility == "none" ? nothing :
                  r.futility == "hsd" ? HSDSpending(r.fpar) : PowerSpending(r.fpar)

@testset "group-sequential designs" begin
    @testset "spending functions" begin
        α = 0.025
        @test spending(OBFSpending(), 1.0, α) ≈ α
        @test spending(OBFSpending(), 0.5, α) ≈ 2 * (1 - cdf(Normal(),
                                                    quantile(Normal(), 1 - α / 2) /
                                                    sqrt(0.5)))
        @test spending(PocockSpending(), 1.0, α) ≈ α
        @test spending(PowerSpending(3), 0.5, α) ≈ α / 8
        @test spending(HSDSpending(0), 0.3, α) ≈ 0.3α
        @test spending(HSDSpending(-4), 1.0, α) ≈ α
        @test spending(OBFSpending(), 0.0, α) == 0
        @test issorted(spending(HSDSpending(1), 0:0.1:1, α))
        @test_throws ArgumentError spending(OBFSpending(), 1.5, α)
        @test_throws ArgumentError PowerSpending(-1)
    end

    ref = CSV.read(joinpath(SEQ_VALDIR, "gs_designs_reference.csv"), DataFrame)
    for g in groupby(ref, :id)
        r = g[1, :]
        @testset "design $(r.id) = $(r.source) ($(r.rule), k=$(r.k), sided=$(r.sided))" begin
        t = parse.(Float64, split(r.timing, ";"))
        d = gs_design(; k=r.k, alpha=r.alpha, beta=r.beta, sided=r.sided, timing=t,
                      efficacy=seq_rule(r), futility=seq_futility(r),
                      binding=r.binding)
        # gsDesign/rpact solve boundaries to ~1e-6 (their tolerances)
        @test d.efficacy_z ≈ g.upper atol = 2e-6
        fin = isfinite.(g.lower)
        @test d.futility_z[fin] ≈ g.lower[fin] atol = 2e-6
        @test d.alpha_spent ≈ g.alpha_spent atol = 1e-7
        @test d.inflation ≈ r.inflation atol = 1e-6
        @test d.expected_information_h0 ≈ r.en0 atol = 1e-6
        @test d.expected_information_h1 ≈ r.en1 atol = 1e-6
        @test d.power ≈ 1 - r.beta atol = 1e-7
        end
    end

    @testset "design properties" begin
        d = gs_design(; k=4, n_fixed=400)
        @test d.n ≈ 400 .* d.inflation .* d.timing
        @test d.alpha_spent[end] ≈ 0.025 atol = 1e-9
        @test d.nominal_p ≈ 1 .- cdf.(Normal(), d.efficacy_z)
        @test d.efficacy_z[1] > d.efficacy_z[end]
        # one look = fixed design
        d1 = gs_design(; k=1)
        @test d1.efficacy_z[1] ≈ quantile(Normal(), 0.975) atol = 1e-9
        @test d1.inflation ≈ 1 atol = 1e-8
        @test occursin("Inflation factor", sprint(show, MIME"text/plain"(), d))
        # binding futility lowers the efficacy bounds; non-binding does not
        nb = gs_design(; k=3, futility=HSDSpending(-2))
        bd = gs_design(; k=3, futility=HSDSpending(-2), binding=true)
        @test nb.efficacy_z ≈ gs_design(; k=3).efficacy_z
        @test all(bd.efficacy_z .<= nb.efficacy_z .+ 1e-12)
        @test nb.futility_z[end] == nb.efficacy_z[end]
        @test_throws ArgumentError gs_design(; k=0)
        @test_throws ArgumentError gs_design(; timing=[0.5, 0.4, 1.0])
        @test_throws ArgumentError gs_design(; efficacy=:foo)
        @test_throws ArgumentError gs_design(; sided=2, futility=HSDSpending(-2))
        @test_throws ArgumentError gs_design(; binding=true)
        @test_throws ArgumentError gs_design(; k=3, efficacy=:haybittle_peto,
                                             hp_bound=1.0)
    end
end

@testset "group-sequential analysis" begin
    aref = CSV.read(joinpath(SEQ_VALDIR, "gs_analysis_reference.csv"), DataFrame)
    for g in groupby(aref, :case)
        r = g[1, :]
        @testset "rpact case $(r.case) ($(r.design))" begin
        des = r.design == "asOF" ? OBFSpending() :
              r.design == "asP" ? PocockSpending() : PowerSpending(3)
        d = gs_design(; k=r.k, alpha=0.025, beta=0.2, efficacy=des)
        fs = r.final_stage
        # rpact uses the planned information rates: standard errors proportional to
        # 1/√t_k anchored at the stopping look reproduce them exactly
        seK = g.effect[fs] / g.z[fs]
        se = seK .* sqrt.(g.info_rate[fs] ./ g.info_rate)
        a = gs_analysis(d, g.z .* se, se; information_fraction=g.info_rate)
        @test a.efficacy_z ≈ g.critical atol = 2e-6
        @test a.decision === :efficacy && a.stop_look == fs
        ok = g.repeated_p .< 0.49
        @test a.repeated_pvalue[ok] ≈ accumulate(min, g.repeated_p)[ok] atol = 2e-6
        @test a.pvalue_adjusted ≈ r.final_p rtol = 1e-5
        @test a.estimate_median_unbiased ≈ r.median_unbiased atol = 1e-5
        @test collect(a.ci_adjusted) ≈ [r.final_lower, r.final_upper] atol = 1e-5
        # rpact's RCIs: estimate ± (critical value at the planned rates) × se
        sea = g.effect ./ g.z
        @test g.rci_lower ≈ g.effect .- a.efficacy_z .* sea atol = 1e-6
        @test g.rci_upper ≈ g.effect .+ a.efficacy_z .* sea atol = 1e-6
        # with the actual standard errors the bounds use the observed information
        rs = gs_analysis(d, g.effect, sea; information_fraction=g.info_rate)
        @test rs.rci_lower ≈ g.effect .- rs.efficacy_z .* sea
        @test rs.efficacy_z ≈ g.critical atol = 5e-3
        @test coef(a) == [a.estimate_median_unbiased]
        @test confint(a) == [a.ci_adjusted[1] a.ci_adjusted[2]]
        @test pvalues(a) == [a.pvalue_adjusted]
        end
    end

    @testset "interim looks, flexibility and decisions" begin
        d = gs_design(; k=3, alpha=0.025, beta=0.1)
        a = gs_analysis(d, [0.1], [0.2])
        @test a.decision === :continue && !a.stopped
        @test a.pvalue_adjusted === nothing
        @test confint(a) ≈ [0.1 - a.efficacy_z[1] * 0.2 0.1 + a.efficacy_z[1] * 0.2]
        @test pvalues(a) == [a.repeated_pvalue[1]]
        @test_throws ArgumentError confint(a; level=0.9)
        # unplanned timing: the Lan–DeMets bound spends f(t) at the observed t
        b = gs_analysis(d, [0.1], [0.2]; information_fraction=[0.5])
        @test b.efficacy_z[1] ≈ quantile(Normal(), 1 - spending(OBFSpending(), 0.5,
                                                                 0.025))
        # final look without crossing
        c = gs_analysis(d, [0.1, 0.1, 0.1], [0.3, 0.2, 0.15])
        @test c.decision === :final_no_rejection && c.stopped
        @test c.pvalue_adjusted > 0.025
        @test c.ci_adjusted[1] < 0 < c.ci_adjusted[2]
        # adjusted CI at another level
        @test confint(c; level=0.9)[1] > c.ci_adjusted[1]
        # futility (non-binding)
        f = gs_design(; k=3, futility=HSDSpending(-2))
        ff = gs_analysis(f, [-0.2], [0.2])
        @test ff.decision === :futility
        # two-sided design: rejection in either direction
        t2 = gs_design(; k=3, sided=2, alpha=0.05)
        neg = gs_analysis(t2, [-0.9], [0.2])
        @test neg.decision === :efficacy && neg.pvalue_adjusted < 0.05
        @test occursin("Decision", sprint(show, MIME"text/plain"(), neg))
        # classical and Haybittle–Peto designs
        hp = gs_design(; k=3, efficacy=:haybittle_peto)
        h = gs_analysis(hp, [0.5, 0.4], [0.15, 0.1])
        @test h.decision === :efficacy && h.stop_look == 1
        @test h.repeated_pvalue[1] == 1.0
        wt = gs_design(; k=3, efficacy=:pocock)
        w = gs_analysis(wt, [0.3], [0.2])
        @test w.efficacy_z[1] ≈ wt.efficacy_z[1] atol = 1e-8
        # errors
        @test_throws ArgumentError gs_analysis(d, [0.1, 0.2, 0.3, 0.4], fill(0.1, 4))
        @test_throws DimensionMismatch gs_analysis(d, [0.1, 0.2], [0.1])
        @test_throws ArgumentError gs_analysis(d, [0.1, 0.2], [0.1, 0.2])
        @test_throws ArgumentError gs_analysis(d, [0.1], [-0.1])
    end
end
