# de Chaisemartin–D'Haultfœuille DID_ℓ (did_multiplegt_dyn) against DIDmultiplegtDYN
# (test/validation/did/generate_dcdh_references.R), against Callaway–Sant'Anna in
# the binary staggered case, invariance, error paths and Monte Carlo coverage.

const DCDH_CASES = Dict(
    "fi_cl" => (:fi, (effects=5, placebo=3, cluster=:state_n)),
    "fi_nocl" => (:fi, (effects=4, placebo=2)),
    "fi_norm" => (:fi, (effects=5, placebo=2, normalized=true, cluster=:state_n)),
    "fi_w" => (:fi, (effects=3, placebo=2, weights=:w1, cluster=:state_n)),
    "fi_never" => (:fi, (effects=3, placebo=1, only_never_switchers=true)),
    "fi_same" => (:fi, (effects=3, placebo=2, same_switchers=true)),
    "sim_both" => (:sim, (effects=5, placebo=3)),
    "sim_cl" => (:sim, (effects=5, placebo=2, cluster=:region)),
    "sim_in" => (:sim, (effects=4, placebo=2, switchers=:in)),
    "sim_out" => (:sim, (effects=4, placebo=2, switchers=:out)),
    "sim_norm" => (:sim, (effects=5, placebo=2, normalized=true, weights=:w)),
    "sim_never" => (:sim, (effects=3, placebo=2, only_never_switchers=true)),
    "sim_tnp" => (:sim, (effects=4, placebo=2, trends_nonparam=[:big], cluster=:region)),
    "sim_tnp2" => (:sim, (effects=3, placebo=1, trends_nonparam=[:big], normalized=true,
                          switchers=:in)),
    "fi_ctrl" => (:fi, (effects=3, placebo=2, controls=[:Dl_hpi], cluster=:state_n)),
    "sim_ctrl" => (:sim, (effects=4, placebo=2, controls=[:x])),
    "sim_ctrl_n" => (:sim, (effects=3, placebo=1, controls=[:x], normalized=true,
                            weights=:w, cluster=:region)))

function dcdh_run(data, which, kw)
    which === :fi && return did_multiplegt_dyn(data.fi, :Dl_vloans_b, :inter_bra,
                                               :county, :year; kw...)
    return did_multiplegt_dyn(data.sim, :y, :d, :g, :t; kw...)
end

# Binary, non-absorbing panel with homogeneous effects `tau` (per period of exposure
# to the current treatment) for coverage checks.
function dcdh_sim_panel(rng; G=200, T=6, tau=1.0)
    rows = DataFrame(g=Int[], t=Int[], d=Int[], y=Float64[])
    for g in 1:G
        F = rand(rng, [3, 4, 5, 99])
        back = rand(rng) < 0.3
        a = randn(rng)
        for t in 1:T
            d = t >= F ? 1 : 0
            back && t >= F + 2 && (d = 0)
            push!(rows, (g, t, d, a + 0.3t + tau * d + randn(rng)))
        end
    end
    return rows
end

@testset "did_multiplegt_dyn" begin
    data = (fi=did_read_csv("favara_imbs.csv"), sim=did_read_csv("dcdh_sim.csv"))
    data.sim.big = Int.(data.sim.region .> 12)
    ref = did_read_csv("r_dcdh.csv")
    tests = did_read_csv("r_dcdh_tests.csv")

    @testset "vs DIDmultiplegtDYN: $c" for c in sort(collect(keys(DCDH_CASES)))
        which, kw = DCDH_CASES[c]
        es = @test_logs min_level = Base.CoreLogging.Error dcdh_run(data, which, kw)
        r = ref[ref.case .== c, :]
        rp = relative_periods(es)
        b, se = coef(es), stderror(es)
        for row in eachrow(r)
            if row.kind == "ate"
                ate = es.details.average_total_effect
                @test coef(ate)[1] ≈ row.estimate atol = 1e-10
                @test stderror(ate)[1] ≈ row.se rtol = 1e-9
                continue
            end
            e = row.kind == "effect" ? Int(row.index) - 1 : -1 - Int(row.index)
            k = findfirst(==(e), rp)
            @test k !== nothing
            @test b[k] ≈ row.estimate atol = 1e-10
            @test se[k] ≈ row.se rtol = 1e-9
            @test es.details.n_switchers_unweighted[k] == row.switchers
            # R counts placebo cells from switchers in only (coalesce of the two
            # count columns); effect cell counts are compared.
            row.kind == "effect" && @test es.details.n_obs[k] == row.n
        end
        t = tests[tests.case .== c, :][1, :]
        eff = findall(>=(0), rp)
        if length(eff) > 1
            # R computes 1 - pchisq(), accurate only to ~1e-15 absolute
            @test isapprox(es.details.joint_effects_test.pvalue, t.p_effects;
                           rtol=1e-6, atol=1e-10)
        end
        if !isnan(t.p_placebos)
            @test isapprox(pre_trend_test(es).pvalue, t.p_placebos; rtol=1e-6,
                           atol=1e-10)
        end
        @test es.reference == [-1]
        @test dof_residual(es) == Inf
    end

    @testset "equals Callaway–Sant'Anna (not yet treated), binary staggered" begin
        rng = StableRNG(314)
        df = sim_staggered(rng; N=240, T=7, cohorts=[0, 3, 4, 6],
                           effect=(g, e) -> 1.0 + 0.3e + 0.2g)
        es = did_multiplegt_dyn(df, :y, :d, :unit, :time; effects=4, placebo=2)
        cs = did_callaway_santanna(df, :y, :d, :unit, :time;
                                   control_group=:not_yet_treated, method=:reg,
                                   base_period=:universal, bootstrap=false)
        csd = aggregate_att(cs, :dynamic)
        for e in 0:3
            @test coef(es)[findfirst(==(e), relative_periods(es))] ≈
                  coef(csd)[findfirst(==(e), relative_periods(csd))] atol = 1e-10
        end
        # placebo ℓ is the long difference Y_{F-1-ℓ} - Y_{F-1}: the universal-base
        # CS pre-period effect at e = -1 - ℓ restricted to the same switchers
        @test all(isfinite, stderror(es))
    end

    @testset "invariance and weights" begin
        sim = data.sim
        es = did_multiplegt_dyn(sim, :y, :d, :g, :t; effects=3, placebo=1)
        es2 = did_multiplegt_dyn(shuffle_rows(StableRNG(1), sim), :y, :d, :g, :t;
                                 effects=3, placebo=1)
        @test coef(es) ≈ coef(es2) atol = 1e-12
        @test vcov(es) ≈ vcov(es2) atol = 1e-12
        # unit weights equal no weights; duplicated rows aggregate to the same cells
        sim1 = copy(sim)
        sim1.one = ones(nrow(sim1))
        esw = did_multiplegt_dyn(sim1, :y, :d, :g, :t; effects=3, placebo=1,
                                 weights=:one)
        @test coef(esw) ≈ coef(es) atol = 1e-12
        dup = vcat(sim, sim)
        esd = did_multiplegt_dyn(dup, :y, :d, :g, :t; effects=3, placebo=1)
        @test coef(esd) ≈ coef(es) atol = 1e-10
        # cluster equal to the group is the default
        esg = did_multiplegt_dyn(sim, :y, :d, :g, :t; effects=3, placebo=1, cluster=:g)
        @test stderror(esg) ≈ stderror(es)
        # the event-study interface works on the result
        avg = event_study_average(es)
        @test coef(avg)[1] ≈ mean(coef(es)[relative_periods(es) .>= 0])
        @test size(confint(es; uniform=true, rng=StableRNG(2))) == (4, 2)
    end

    @testset "errors" begin
        sim = data.sim
        @test_throws ArgumentError did_multiplegt_dyn(sim, :y, :d, :g, :nope)
        @test_throws ArgumentError did_multiplegt_dyn(sim, :y, :d, :g, :t; effects=0)
        @test_throws ArgumentError did_multiplegt_dyn(sim, :y, :d, :g, :t; placebo=-1)
        @test_throws ArgumentError did_multiplegt_dyn(sim, :y, :d, :g, :t;
                                                      switchers=:up)
        # cluster not nested
        bad = copy(sim)
        bad.cl = rand(StableRNG(3), 1:5, nrow(bad))
        @test_throws ArgumentError did_multiplegt_dyn(bad, :y, :d, :g, :t; cluster=:cl)
        # everybody switches at the same date: Design Restriction 1 fails
        same = DataFrame(g=repeat(1:20; inner=5), t=repeat(1:5; outer=20))
        same.d = Int.(same.t .>= 3)
        same.y = randn(StableRNG(4), nrow(same))
        @test_throws ArgumentError did_multiplegt_dyn(same, :y, :d, :g, :t)
        # non-parametric trends by state: every county of a state switches at the
        # same date, so no comparison group exists (R errors too)
        @test_throws ArgumentError did_multiplegt_dyn(data.fi, :Dl_vloans_b, :inter_bra,
                                                      :county, :year;
                                                      trends_nonparam=[:state_n])
        tv = copy(sim)
        tv.cls = rand(StableRNG(8), 1:2, nrow(tv))
        @test_throws ArgumentError did_multiplegt_dyn(tv, :y, :d, :g, :t;
                                                      trends_nonparam=[:cls])
        # more effects than estimable: warns and truncates
        @test_logs (:warn, r"only") match_mode = :any did_multiplegt_dyn(
            sim, :y, :d, :g, :t; effects=20)
    end

    @testset "Monte Carlo coverage" begin
        R = mc_reps(400, 80)
        rng = StableRNG(2718)
        cover = zeros(Int, 2)
        for _ in 1:R
            df = dcdh_sim_panel(rng)
            es = did_multiplegt_dyn(df, :y, :d, :g, :t; effects=2)
            ci = confint(es)
            # effect of switching on for ℓ periods = τ in every period (homogeneous,
            # no dynamics): DID_ℓ = τ for ℓ = 1, 2
            for k in 1:2
                cover[k] += ci[k, 1] <= 1.0 <= ci[k, 2]
            end
        end
        # the variance estimator is conservative (cohort demeaning): coverage at
        # least nominal up to Monte Carlo error
        for k in 1:2
            @test cover[k] / R >= 0.95 - 3.5 * sqrt(0.95 * 0.05 / R) - 0.01
        end
    end
end
