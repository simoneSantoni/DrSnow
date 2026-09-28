# Simulation-based diagnosis: DeclareDesign reference, analytic power, reproducibility,
# natural-experiment designs, failures.

des_pop(rng, p) = (U = randn(rng, p.n); DataFrame(Y0=U, Y1=U .+ p.effect))

function des_simple_design()
    return declare_design(des_pop; params=(n=100, effect=0.3),
                          assignment=(data, p) -> CompleteRandomization(p.n, p.n ÷ 2),
                          estimand=(data, p) -> mean(data.Y1 .- data.Y0),
                          estimators="DiM" =>
                              (data, p) -> experiment_estimate(data, :Y, :Z),
                          name="two-arm trial")
end

@testset "diagnosands vs DeclareDesign" begin
    ref = CSV.read(joinpath(DES_VALDIR, "reference_declaredesign.csv"), DataFrame)
    d = des_simple_design()
    S = mc_reps(20_000, 4_000)
    for (n, eff) in ((100, 0.3), (40, 0.2))
        dx = diagnose_design(d; sims=S, params=(n=n, effect=eff), bootstrap=100,
                             rng=StableRNG(n))
        dg = dx.diagnosands
        for r in eachrow(ref[(ref.n .== n) .& (ref.effect .== eff), :])
            nm = Symbol(r.diagnosand)
            row = dg[dg.diagnosand .== nm, :]
            @test nrow(row) == 1
            v, se = row.value[1], row.mc_se[1]
            tol = 4.5 * sqrt(se^2 + r.mc_se^2) + 1e-12
            @test abs(v - r.value) <= tol
        end
    end
end

@testset "simulated power matches analytic power" begin
    d = des_simple_design()
    S = mc_reps(4000, 1000)
    for n in (60, 200)
        dx = diagnose_design(d; sims=S, params=(n=n,), rng=StableRNG(7 + n))
        p = only(dx.diagnosands.value[dx.diagnosands.diagnosand .== :power])
        pa = power_means(effect=0.3, n=n).power
        @test abs(p - pa) < 4 * sqrt(pa * (1 - pa) / S)
    end
end

@testset "reproducibility and threading" begin
    d = des_simple_design()
    a = diagnose_design(d; sims=200, rng=StableRNG(1), threaded=true)
    b = diagnose_design(d; sims=200, rng=StableRNG(1), threaded=false)
    @test isequal(a.simulations, b.simulations)
    @test isequal(a.diagnosands, b.diagnosands)
    c = diagnose_design(d; sims=200, rng=StableRNG(2))
    @test !isequal(a.simulations.estimate, c.simulations.estimate)
    @test nrow(a.simulations) == 200 && all(.!a.simulations.failed)
    @test occursin("power", sprint(show, MIME"text/plain"(), a))
    # grid: one row per (grid point, diagnosand)
    g = diagnose_grid(d, (n=[40, 80], effect=[0.0, 0.5]); sims=100, rng=StableRNG(3))
    @test g.parameters == [:n, :effect]
    pw = g.diagnosands[g.diagnosands.diagnosand .== :power, :]
    @test nrow(pw) == 4
    @test pw.value[(pw.n .== 80) .& (pw.effect .== 0.5)][1] >
          pw.value[(pw.n .== 40) .& (pw.effect .== 0.0)][1]
    g2 = diagnose_grid(d, DataFrame(n=[40, 80], effect=[0.0, 0.5]); sims=100,
                       rng=StableRNG(3), threaded=false)
    @test nrow(g2.diagnosands[g2.diagnosands.diagnosand .== :power, :]) == 2
    # type-S / exaggeration missing when nothing is significant
    tiny = diagnose_design(d; sims=20, params=(n=10, effect=0.0), rng=StableRNG(4),
                           alpha=1e-9)
    ts = tiny.diagnosands[tiny.diagnosands.diagnosand .== :type_s_rate, :value][1]
    @test ts === missing
end

@testset "blocked design declared with block_design" begin
    # pairs formed on a covariate in the population step (ids 1:n are in row order,
    # so the design's unit order is the row order)
    function popb(rng, p)
        x = randn(rng, p.n)
        U = x .+ 0.5 .* randn(rng, p.n)
        df = DataFrame(id=1:p.n, x=x, Y0=U, Y1=U .+ p.effect)
        df.pair = block_design(df, :x; id=:id).blocks
        return df
    end
    ests = ["pairs" => (data, p) -> experiment_estimate(data, :Y, :Z; blocks=:pair),
            "complete" => (data, p) -> experiment_estimate(data, :Y, :Z)]
    d = declare_design(popb; params=(n=60, effect=0.4), estimand=0.4, estimators=ests,
                       assignment=(data, p) -> MatchedPairsRandomization(data.pair))
    dx = diagnose_design(d; sims=mc_reps(1000, 200), rng=StableRNG(5))
    dg = dx.diagnosands
    sd(e) = only(dg.value[(dg.estimator .== e) .& (dg.diagnosand .== :sd_estimate)])
    se(e) = only(dg.value[(dg.estimator .== e) .& (dg.diagnosand .== :mean_se)])
    @test sd("pairs") ≈ sd("complete")          # same estimate, different variance
    @test se("pairs") < se("complete")          # the pair variance uses the matching
    @test only(dg.value[(dg.estimator .== "pairs") .& (dg.diagnosand .== :coverage)]) >= 0.9
end

@testset "natural experiment: DiD panel vs power_did" begin
    function panel(rng, p)
        n, T = p.n, 2
        u = randn(rng, n)
        d = zeros(Int, n); d[randperm(rng, n)[1:(n ÷ 2)]] .= 1
        rows = [(unit=i, year=t, treat=d[i] * (t == 2),
                 y=sqrt(p.rho) * u[i] + sqrt(1 - p.rho) * randn(rng) +
                   p.effect * d[i] * (t == 2)) for i in 1:n for t in 1:T]
        return DataFrame(rows)
    end
    d = declare_design(panel; params=(n=200, effect=0.25, rho=0.5), estimand=0.25,
                       estimators="TWFE" => (data, p) -> did_twfe(data, :y, :treat, :unit,
                                                                 :year;
                                                                 warn_heterogeneity=false))
    S = mc_reps(2000, 300)
    dx = diagnose_design(d; sims=S, rng=StableRNG(6))
    dg = dx.diagnosands
    a = power_did(effect=0.25, n=200, rho=0.5, estimator=:did)
    sdv = only(dg.value[dg.diagnosand .== :sd_estimate])
    @test abs(sdv / a.se - 1) < 4 / sqrt(2 * S)
    pw = only(dg.value[dg.diagnosand .== :power])
    @test abs(pw - a.power) < 4 * sqrt(a.power * (1 - a.power) / S) + 0.02
end

@testset "failures and input errors" begin
    flaky(rng, p) = DataFrame(Y0=randn(rng, 20), Y1=randn(rng, 20))
    est(data, p) = rand(Random.Xoshiro(hash(data.Y0[1])), Bool) ? error("boom") :
                   experiment_estimate(data, :Y, :Z)
    d = declare_design(flaky; assignment=CompleteRandomization(20, 10), estimators=est)
    @test_throws ErrorException diagnose_design(d; sims=50, rng=StableRNG(1))
    dx = diagnose_design(d; sims=50, rng=StableRNG(1), on_error=:record)
    @test any(dx.simulations.failed) && hasproperty(dx.simulations, :error)
    @test dx.diagnosands.n_failed[1] == count(dx.simulations.failed)
    @test !(:bias in dx.diagnosands.diagnosand)          # no estimand declared
    bad = declare_design(flaky; assignment=CompleteRandomization(10, 5),
                         estimators=(x, p) -> experiment_estimate(x, :Y, :Z))
    @test_throws DimensionMismatch diagnose_design(bad; sims=2, rng=StableRNG(1))
    notest = declare_design(flaky; assignment=CompleteRandomization(20, 10),
                            estimators=(x, p) -> 1.0)
    @test_throws ArgumentError diagnose_design(notest; sims=2, rng=StableRNG(1))
    @test_throws ArgumentError declare_design(1; estimators=est)
    @test_throws ArgumentError declare_design(flaky; estimators=["a" => est, "a" => est])
    @test_throws ArgumentError declare_design(flaky; estimators=est, assignment=3)
    @test_throws ArgumentError diagnose_design(d; sims=10, alpha=2.0)
    @test_throws ArgumentError diagnose_design(d; sims=10, on_error=:ignore)
    @test_throws ArgumentError diagnose_grid(d, (;); sims=10)
end
