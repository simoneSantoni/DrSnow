# experiment_estimate: estimatr references, design-score reuse, invariance, coverage.

@testset "experiment_estimate vs estimatr" begin
    d = CSV.read(joinpath(DES_VALDIR, "experiment_data.csv"), DataFrame)
    ref = CSV.read(joinpath(DES_VALDIR, "reference_estimatr.csv"), DataFrame)
    R = Dict(String(r.case) => r for r in eachrow(ref))
    chk(r, case) = (@test coef(r)[1] ≈ R[case].estimate atol = 1e-10;
                    @test stderror(r)[1] ≈ R[case].std_error atol = 1e-10;
                    @test dof_residual(r) ≈ R[case].df atol = 1e-8)
    chk(experiment_estimate(d, :y, :z), "dim")
    chk(experiment_estimate(d, :yb, :zb; blocks=:block), "dim_blocked")
    chk(experiment_estimate(d, :yp, :zp; blocks=:pair), "dim_pairs")
    chk(experiment_estimate(d, :y, :z; method=:lin, covariates=[:x1, :x2]), "lin")
    chk(experiment_estimate(d, :yb, :zb; method=:block_fe, blocks=:block,
                            covariates=[:x1]), "block_fe")
    chk(experiment_estimate(d, :yb, :zb; method=:lin, covariates=[:x1]),
        "lin_hc1_noblock_zb")
    # confidence intervals use the t reference with the reported dof
    r = experiment_estimate(d, :yp, :zp; blocks=:pair)
    @test confint(r)[1, 2] - coef(r)[1] ≈ quantile(TDist(29), 0.975) * stderror(r)[1]
    @test length(r.block_estimates) == 30
    @test occursin("Imai", sprint(show, MIME"text/plain"(), r))
    # row-order invariance
    sh = d[randperm(StableRNG(3), nrow(d)), :]
    @test coef(experiment_estimate(sh, :yb, :zb; blocks=:block))[1] ≈
          R["dim_blocked"].estimate atol = 1e-10
    @test stderror(experiment_estimate(sh, :y, :z; method=:lin,
                                       covariates=[:x1, :x2]))[1] ≈
          R["lin"].std_error atol = 1e-10
    a = experiment_estimate(d, :y, :z; method=:dml, covariates=[:x1, :x2], id=:id,
                            learner=OLSLearner(), rng=StableRNG(4))
    b = experiment_estimate(sh, :y, :z; method=:dml, covariates=[:x1, :x2], id=:id,
                            learner=OLSLearner(), rng=StableRNG(4))
    @test coef(a) == coef(b) && stderror(a) == stderror(b)
    @test isinf(dof_residual(a))
end

@testset "experiment_estimate errors" begin
    d = CSV.read(joinpath(DES_VALDIR, "experiment_data.csv"), DataFrame)
    @test_throws ArgumentError experiment_estimate(d, :y, :z; covariates=[:x1])
    @test_throws ArgumentError experiment_estimate(d, :y, :x1)
    @test_throws ArgumentError experiment_estimate(d, :y, :z; method=:ols)
    @test_throws ArgumentError experiment_estimate(d, :y, :z; method=:dml)
    @test_throws ArgumentError experiment_estimate(d, :y, :nope)
    # blocked Neyman variance needs two units per arm unless all blocks are pairs
    d2 = copy(d); d2.b = vcat(repeat(1:10, inner=4), repeat(11:20, inner=2))
    d2.zz = vcat(repeat([1, 1, 0, 0], 10), repeat([1, 0], 10))
    @test_throws ArgumentError experiment_estimate(d2, :y, :zz; blocks=:b)
    @test experiment_estimate(d2, :y, :zz; blocks=:b, method=:block_fe) isa
          ExperimentEstimate
end

@testset "design and analysis share blocks and scores" begin
    rng = StableRNG(41)
    n = 100
    pilot = DataFrame(x1=randn(rng, 300), x2=randn(rng, 300))
    pilot.y = 2 .* pilot.x1 .+ pilot.x2 .^ 2 .+ 0.5 .* randn(rng, 300)
    df = DataFrame(id=1:n, x1=randn(rng, n), x2=randn(rng, n))
    ps = prognostic_score(pilot, :y; covariates=[:x1, :x2], target=df, id=:id,
                          learner=KNNLearner(), rng=StableRNG(1))
    bd = block_design(df, ps; id=:id)
    a = assign_treatment(bd, df; rng=StableRNG(2))
    a.y = 2 .* a.x1 .+ a.x2 .^ 2 .+ 0.5 .* randn(rng, n) .+ 1.0 .* a.treated
    for m in (:difference, :block_fe, :lin, :dml)
        r = experiment_estimate(a, :y, :treated, bd; method=m, rng=StableRNG(3))
        @test r.n_blocks == 50
        @test (m === :difference) == !(:prognostic_score in r.covariates)
        @test abs(coef(r)[1] - 1.0) < 4 * stderror(r)[1]
    end
    # the score as covariate equals passing it by hand with the design's blocks
    r1 = experiment_estimate(a, :y, :treated, bd; method=:lin)
    a.prognostic_score = bd.score[[findfirst(==(u), bd.ids) for u in a.id]]
    r2 = experiment_estimate(a, :y, :treated; method=:lin, blocks=:block,
                             covariates=[:prognostic_score])
    @test coef(r1) ≈ coef(r2) && stderror(r1) ≈ stderror(r2)
    # assignment inconsistent with the design
    bad = copy(a); bad.treated = zeros(Int, n); bad.treated[1:50] .= 1
    select!(bad, Not(:prognostic_score))
    @test_throws ArgumentError experiment_estimate(bad, :y, :treated, bd)
    @test_throws DimensionMismatch experiment_estimate(a[1:10, :], :y, :treated, bd)
end

@testset "experiment_estimate Monte Carlo coverage" begin
    rng = StableRNG(42)
    S = mc_reps(1000, 200)
    n = 80
    x = randn(rng, n)
    df = DataFrame(id=1:n, x=x)
    bd = block_design(df, :x; id=:id)
    cover = Dict(m => 0 for m in (:difference, :lin, :dml))
    for _ in 1:S
        a = assign_treatment(bd, df; rng=rng)
        a.y = x .+ 0.5 .* x .^ 2 .+ randn(rng, n) .+ 0.3 .* a.treated
        for m in keys(cover)
            r = m === :difference ?
                experiment_estimate(a, :y, :treated, bd; method=m) :
                experiment_estimate(a, :y, :treated, bd; method=m, covariates=[:x],
                                    use_score=false, learner=OLSLearner(),
                                    rng=Random.Xoshiro(rand(rng, UInt64)))
            ci = confint(r)
            cover[m] += ci[1, 1] <= 0.3 <= ci[1, 2]
        end
    end
    for (m, c) in cover
        @test c / S >= 0.95 - 3 * sqrt(0.95 * 0.05 / S) - 0.01
    end
end
