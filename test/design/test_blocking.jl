# Prognostic scores, block formation, assignment and variance-reduction diagnostics.

function des_blocking_data(rng; n=120, npilot=400)
    mk(m) = (X = randn(rng, m, 3);
             DataFrame(id=["u$(i)" for i in 1:m], x1=X[:, 1], x2=X[:, 2], x3=X[:, 3]))
    pilot = mk(npilot)
    pilot.y = 1.5 .* pilot.x1 .- pilot.x2 .+ randn(rng, npilot)
    sample = mk(n)
    return pilot, sample
end

@testset "prognostic_score" begin
    pilot, sample = des_blocking_data(StableRNG(31))
    ps = prognostic_score(pilot, :y; covariates=[:x1, :x2, :x3], learner=OLSLearner(),
                          target=sample, id=:id, rng=StableRNG(1))
    @test length(ps.score) == nrow(sample) && ps.ids == sample.id
    @test !ps.crossfit
    @test 0.65 < ps.r2 < 0.85               # true R² = 3.25 / 4.25 ≈ 0.76
    # a model fitted on all of pilot: OLS predictions are exact linear functions
    X = hcat(ones(nrow(pilot)), Matrix(pilot[:, [:x1, :x2, :x3]]))
    β = X \ pilot.y
    @test ps.score ≈ hcat(ones(nrow(sample)), Matrix(sample[:, [:x1, :x2, :x3]])) * β
    # reproducible and invariant to the row order of both data sets
    ps2 = prognostic_score(pilot[randperm(StableRNG(2), nrow(pilot)), :], :y;
                           covariates=[:x1, :x2, :x3], learner=OLSLearner(),
                           target=sample[end:-1:1, :], id=:id, rng=StableRNG(1))
    d2 = Dict(zip(ps2.ids, ps2.score))
    @test [d2[u] for u in ps.ids] ≈ ps.score
    @test ps2.r2 ≈ ps.r2
    # cross-fitted scores on the experimental sample itself
    pc = prognostic_score(pilot, :y; covariates=[:x1, :x2], learner=RidgeLearner(),
                          id=:id, rng=StableRNG(3))
    @test pc.crossfit && length(pc.score) == nrow(pilot)
    pc2 = prognostic_score(pilot[end:-1:1, :], :y; covariates=[:x1, :x2],
                           learner=RidgeLearner(), id=:id, rng=StableRNG(3))
    @test Dict(zip(pc2.ids, pc2.score)) == Dict(zip(pc.ids, pc.score))
    @test occursin("out-of-sample R²", sprint(show, MIME"text/plain"(), ps))
    @test_throws ArgumentError prognostic_score(pilot, :y; covariates=[:y, :x1])
    @test_throws ArgumentError prognostic_score(pilot, :y; covariates=[:nope])
    @test_throws ArgumentError prognostic_score(pilot, :y; covariates=Symbol[])
    @test_throws ArgumentError prognostic_score(vcat(pilot, pilot[1:1, :]), :y;
                                                covariates=[:x1], id=:id)
end

@testset "block_design on a score" begin
    pilot, sample = des_blocking_data(StableRNG(32))
    ps = prognostic_score(pilot, :y; covariates=[:x1, :x2, :x3], target=sample, id=:id,
                          rng=StableRNG(1))
    bd = block_design(sample, ps; id=:id)
    @test bd.mechanism isa MatchedPairsRandomization
    @test all(==(2), bd.block_sizes) && sum(bd.n_treated) == 60
    s = bd.score
    o = sortperm(s)
    # optimal 1-D pairs are adjacent in sorted order
    @test all(bd.blocks[o[2k - 1]] == bd.blocks[o[2k]] for k in 1:60)
    @test bd.objective ≈ sum(s[o[2k]] - s[o[2k - 1]] for k in 1:60)
    @test bd.r2 == ps.r2
    # invariance to the row order of the data
    sh = sample[randperm(StableRNG(5), nrow(sample)), :]
    bs = block_design(sh, ps; id=:id)
    @test bs.ids == bd.ids && bs.blocks == bd.blocks
    # a score column gives the same design
    sample.ps = [Dict(zip(ps.ids, ps.score))[u] for u in sample.id]
    @test block_design(sample, :ps; id=:id).blocks == bd.blocks
    # quantile blocks: consecutive groups with sizes differing by at most one
    bq = block_design(sample, ps; id=:id, method=:blocks, block_size=7)
    @test bq.mechanism isa StratifiedRandomization
    @test length(bq.block_sizes) == 17 && extrema(bq.block_sizes) == (7, 8)
    @test count(bq.blocks[o][2:end] .!= bq.blocks[o][1:end-1]) == 16   # contiguous
    @test bq.n_treated == [floor(Int, k / 2 + 0.5) for k in bq.block_sizes]
    b10 = block_design(sample, ps; id=:id, method=:blocks, n_blocks=10)
    @test length(b10.block_sizes) == 10
    # greedy pairing on the score is never better than the optimum
    bgr = block_design(sample, ps; id=:id, algorithm=:greedy)
    @test bgr.objective >= bd.objective - 1e-12
    # odd number of units: one block of three
    b3 = block_design(sample[1:11, :], ps; id=:id)
    @test sort(b3.block_sizes) == [2, 2, 2, 2, 3]
    @test b3.mechanism isa StratifiedRandomization
    @test occursin("Matched-pair design", sprint(show, MIME"text/plain"(), bd))
end

@testset "block_design on covariates" begin
    _, sample = des_blocking_data(StableRNG(33); n=40)
    bm = block_design(sample; id=:id, covariates=[:x1, :x2])
    @test bm.distance === :mahalanobis && bm.mechanism isa MatchedPairsRandomization
    bg = block_design(sample; id=:id, covariates=[:x1, :x2], method=:blocks,
                      block_size=4, algorithm=:greedy)
    @test all(==(4), bg.block_sizes)
    bgo = block_design(sample[1:42 - 3, :]; id=:id, covariates=[:x1, :x2],
                       method=:blocks, block_size=4, algorithm=:greedy)
    @test sort(unique(bgo.block_sizes)) == [4, 5]
    sh = sample[end:-1:1, :]
    @test block_design(sh; id=:id, covariates=[:x1, :x2]).blocks == bm.blocks
    @test_throws ArgumentError block_design(sample; id=:id, covariates=[:x1, :x2],
                                            method=:blocks, block_size=4)
    @test_throws ArgumentError block_design(sample; id=:id)
    @test_throws ArgumentError block_design(sample, :x1; id=:id, method=:strata)
    @test_throws ArgumentError block_design(sample, :x1; id=:id, p_treat=1.0)
    @test_throws ArgumentError block_design(vcat(sample, sample[1:1, :]), :x1; id=:id)
    @test_throws DimensionMismatch block_design(sample, [1.0, 2.0])
end

@testset "assign_treatment and randomization inference" begin
    pilot, sample = des_blocking_data(StableRNG(34); n=60)
    ps = prognostic_score(pilot, :y; covariates=[:x1, :x2], target=sample, id=:id,
                          rng=StableRNG(1))
    bd = block_design(sample, ps; id=:id, method=:blocks, block_size=4)
    a = assign_treatment(bd, sample; rng=StableRNG(7))
    @test all(sum(a.treated[a.block .== b]) == bd.n_treated[b] for b in 1:15)
    a2 = assign_treatment(bd, sample[end:-1:1, :]; rng=StableRNG(7))
    @test Dict(zip(a2.id, a2.treated)) == Dict(zip(a.id, a.treated))
    a.y = 1.5 .* a.x1 .- a.x2 .+ randn(StableRNG(8), 60)
    rt = randomization_test(a, :y, :treated; mechanism=bd.mechanism, id=:id,
                            nperm=500, rng=StableRNG(9))
    @test 0 < rt.pvalue <= 1
    @test_throws DimensionMismatch assign_treatment(bd, sample[1:10, :])
end

@testset "variance_reduction" begin
    pilot, sample = des_blocking_data(StableRNG(35); n=80)
    ps = prognostic_score(pilot, :y; covariates=[:x1, :x2, :x3], target=sample, id=:id,
                          rng=StableRNG(1))
    bd = block_design(sample, ps; id=:id, method=:blocks, block_size=4)
    vr = variance_reduction(bd; r2=0.6)
    s = bd.score
    vb = sum((4 / 80)^2 * var(s[bd.blocks .== b]) * (1 / 2 + 1 / 2) for b in 1:20)
    @test vr.score_variance_ratio ≈ vb / (var(s) * (1 / 40 + 1 / 40))
    @test vr.regression_adjustment_ratio ≈ 0.4
    @test 0 < vr.outcome_variance_ratio < 1
    @test vr.effective_sample_multiplier ≈ 1 / vr.outcome_variance_ratio
    # Monte Carlo: Y = s + e with Var(e) chosen so that R² = 0.6
    rng = StableRNG(36)
    σe = sqrt(var(s) * 0.4 / 0.6)
    S = mc_reps(4000, 1500)
    cr = CompleteRandomization(80, 40)
    dims(z, y) = mean(y[z]) - mean(y[.!z])
    eb, ec = Float64[], Float64[]
    for _ in 1:S
        y = s .+ σe .* randn(rng, 80)
        push!(eb, dims(draw_assignment(rng, bd.mechanism), y))
        push!(ec, dims(draw_assignment(rng, cr), y))
    end
    ratio = var(eb) / var(ec)
    @test abs(ratio / vr.outcome_variance_ratio - 1) < 5 * sqrt(4 / S)
    vx = variance_reduction(block_design(sample, :x1; id=:id))
    @test vx.outcome_variance_ratio === missing && vx.score_variance_ratio > 0
    @test_throws ArgumentError variance_reduction(block_design(sample; id=:id,
                                                               covariates=[:x1]))
end
