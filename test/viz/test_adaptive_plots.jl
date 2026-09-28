# Plot data and Makie plots for adaptive experiments (assignment probabilities,
# excursion effects).

const AD_VIZ = let
    log = run_adaptive_experiment(GaussianThompson(3; floor=0.2, floor_decay=0.5,
                                                   burnin=30),
                                  GaussianBandit([0.0, 0.3, 0.6]), 300;
                                  rng=StableRNG(1))
    env = ContextualBandit(x -> [0.0, x[1]], rng -> randn(rng, 1), 2)
    clog = run_adaptive_experiment(LinearThompson(2, 1; floor=0.1), env, 200;
                                   batch_size=50, rng=StableRNG(2))
    rng = StableRNG(3)
    n, T = 25, 20
    df = DataFrame(id=repeat(1:n; inner=T), dp=repeat(1:T, n) ./ T, x=rand(rng, n * T))
    df.a = Float64.(rand(rng, n * T) .< 0.5)
    df.y = df.x .+ df.a .* (0.5 .- 0.5 .* df.dp) .+ randn(rng, n * T)
    df.yb = Float64.(rand(rng, n * T) .< 0.3 .* exp.(0.3 .* df.a))
    w = wcls(df, :y, :a, :id; rand_prob=0.5, moderators=[:dp, :x], controls=[:dp, :x])
    w0 = wcls(df, :y, :a, :id; rand_prob=0.5, controls=[:x])
    e = emee(df, :yb, :a, :id; rand_prob=0.5, controls=[:x])
    (; log, clog, w, w0, e)
end

@testset "adaptive plot data" begin
    f = AD_VIZ
    d = DrSnow._viz_assignment_data(f.log)
    @test nrow(d.table) == 300 * 3
    @test d.burnin == 30 && d.K == 3 && !d.contextual
    @test d.floor.floor ≈ 0.2 .* (1:300) .^ -0.5
    s = d.table[d.table.t .== 100, :]
    @test sum(s.probability) ≈ 1
    dc = DrSnow._viz_assignment_data(f.clog)
    @test dc.contextual && nrow(dc.table) == 4 * 2
    @test dc.table.probability[1:2] ≈ vec(mean(f.clog.probabilities[1:50, :]; dims=1))

    x = DrSnow._viz_excursion_data(f.w; moderator=:dp, values=[0.0, 1.0], at=(x=0.5,))
    @test x.table.estimate ≈ [coef(f.w)' * [1, 0, 0.5], coef(f.w)' * [1, 1, 0.5]]
    @test all(x.table.conf_low .< x.table.estimate .< x.table.conf_high)
    @test x.held.x == 0.5 && x.scale === :difference
    g = DrSnow._viz_excursion_data(f.w)
    @test g.moderator === :dp && nrow(g.table) == 50
    @test g.held.x ≈ mean(f.w.moderator_means[2])
    z = DrSnow._viz_excursion_data(f.w0; level=0.9)
    @test nrow(z.table) == 1 && z.table.conf_high[1] ≈ confint(f.w0; level=0.9)[1, 2]
    rr = DrSnow._viz_excursion_data(f.e; relative_risk=true)
    @test rr.table.estimate[1] ≈ exp(coef(f.e)[1]) && rr.scale === :rr
    @test_throws ArgumentError DrSnow._viz_excursion_data(f.w; relative_risk=true)
    @test_throws ArgumentError DrSnow._viz_excursion_data(f.w; moderator=:nope)
    @test_throws ArgumentError DrSnow._viz_excursion_data(f.w0; moderator=:x)
    @test_throws ArgumentError DrSnow._viz_excursion_data(f.w; at=(nope=1,))
end

if VIZ_MAKIE
    @testset "adaptive Makie plots" begin
        f = AD_VIZ
        CM = CairoMakie
        outdir = mktempdir()
        save_ok(fig, name) = (p = joinpath(outdir, name * ".png");
                              CM.save(p, fig; px_per_unit=1); filesize(p) > 5_000)
        axes_of(fig) = filter(c -> c isa CM.Axis, fig.content)
        fig = plot_assignment_probabilities(f.log)
        @test fig isa CM.Figure
        ax = only(axes_of(fig))
        @test count(p -> p isa CM.Stairs, ax.scene.plots) == 3
        @test save_ok(fig, "assign")
        @test save_ok(plot_assignment_probabilities(f.clog; floor=false), "assign_ctx")
        fig = plot_excursion_effect(f.w; moderator=:dp)
        ax = only(axes_of(fig))
        @test count(p -> p isa CM.Band, ax.scene.plots) == 1
        @test save_ok(fig, "excursion")
        @test save_ok(plot_excursion_effect(f.w0), "excursion_marginal")
        @test save_ok(plot_excursion_effect(f.e; relative_risk=true), "emee_rr")
        fig2 = CM.Figure()
        @test plot_assignment_probabilities!(CM.Axis(fig2[1, 1]), f.log) isa CM.Axis
        @test plot_excursion_effect!(CM.Axis(fig2[1, 2]), f.w) isa CM.Axis
    end
end
