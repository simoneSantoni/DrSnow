# Plot data and Makie plots for the longitudinal area (regime weights).

const LG_VIZ = let
    rng = StableRNG(21)
    n, K = 200, 3
    id = repeat(1:n; inner=K)
    t = repeat(1:K, n)
    w = repeat(randn(rng, n); inner=K)
    l = zeros(n * K)
    a = zeros(n * K)
    y = Vector{Union{Missing,Float64}}(missing, n * K)
    for i in 1:n
        a_prev, total = 0.0, 0.0
        for s in 1:K
            r = (i - 1) * K + s
            l[r] = 0.5 * a_prev + 0.3 * w[r] + randn(rng)
            a[r] = rand(rng) < 1 / (1 + exp(-(0.5 * l[r] + 0.8 * a_prev)))
            total += 0.5 * a[r] + 0.3 * l[r]
            s == K && (y[r] = total + randn(rng))
            a_prev = a[r]
        end
    end
    df = DataFrame(; id, t, w, l, a, y)
    longitudinal_regime(df, :y, :a, :id, :t; regime=[always_treat(K), never_treat(K)],
                        covariates=[:l], baseline=[:w], estimator=:tmle,
                        outcome_learner=OLSLearner(), treatment_learner=LogisticLearner(),
                        crossfit=false)
end

@testset "longitudinal plot data" begin
    r = LG_VIZ
    d = DrSnow._viz_regime_weights_data(r)
    @test d.labels == ["always treat", "never treat"] && d.periods == [1, 2, 3]
    @test d.bound == 100.0 && d.n == 200
    @test nrow(d.table) == sum(count, r.followers)
    @test nrow(d.counts) == 6
    s = d.table[(d.table.regime .== 1) .& (d.table.period .== 2), :]
    @test s.weight ≈ r.weights[1][r.followers[1][:, 2], 2]
    @test d.counts.n_following[2] == count(r.followers[1][:, 2])
    d2 = DrSnow._viz_regime_weights_data(r; regimes=["never treat"])
    @test d2.labels == ["never treat"] && all(d2.table.regime .== 1)
    @test DrSnow._viz_regime_weights_data(r; regimes=[2]).labels == ["never treat"]
    @test_throws ArgumentError DrSnow._viz_regime_weights_data(r; regimes=["nope"])
    @test_throws ArgumentError DrSnow._viz_regime_weights_data(r; regimes=Int[])
end

if VIZ_MAKIE
    @testset "longitudinal Makie plots" begin
        CM = CairoMakie
        outdir = mktempdir()
        fig = plot_regime_weights(LG_VIZ)
        @test fig isa CM.Figure
        ax = only(filter(c -> c isa CM.Axis, fig.content))
        @test count(p -> p isa CM.BoxPlot, ax.scene.plots) == 2
        p = joinpath(outdir, "weights.png")
        CM.save(p, fig; px_per_unit=1)
        @test filesize(p) > 5_000
        fig2 = CM.Figure()
        @test plot_regime_weights!(CM.Axis(fig2[1, 1]), LG_VIZ; regimes=[1]) isa CM.Axis
    end
end
