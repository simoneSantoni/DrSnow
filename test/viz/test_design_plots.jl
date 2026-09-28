# Plot data and Makie plots for the design area (power curves, blocks).

const DES_VIZ_FIX = let
    pop(rng, p) = (U = randn(rng, p.n); DataFrame(Y0=U, Y1=U .+ p.effect))
    d = declare_design(pop; params=(n=60, effect=0.4),
                       assignment=(data, p) -> CompleteRandomization(p.n, p.n ÷ 2),
                       estimand=(data, p) -> p.effect,
                       estimators="DiM" => (data, p) -> experiment_estimate(data, :Y, :Z))
    g1 = diagnose_grid(d, (n=[40, 80, 160],); sims=60, rng=StableRNG(1))
    g2 = diagnose_grid(d, (n=[40, 80], effect=[0.2, 0.5]); sims=40, rng=StableRNG(2))
    opt = optimize_design(d; space=(n=20:10:300,), max_sims=1200, sims_per_point=100,
                          rng=StableRNG(3))
    an = [power_means(effect=0.4, n=n) for n in 20:20:200]
    df = DataFrame(id=1:30, x=randn(StableRNG(4), 30))
    bd = block_design(df, :x; id=:id, method=:blocks, block_size=3)
    (; d, g1, g2, opt, an, bd, bdc=block_design(df; id=:id, covariates=[:x]))
end

@testset "design plot data" begin
    f = DES_VIZ_FIX
    p1 = DrSnow._viz_power_curve_data(f.g1)
    @test p1.parameter == "n" && nrow(p1.points) == 3
    @test issorted(p1.points.x)
    @test all(p1.points.conf_low .<= p1.points.power .<= p1.points.conf_high)
    @test_throws ArgumentError DrSnow._viz_power_curve_data(f.g2)
    p2 = DrSnow._viz_power_curve_data(f.g2; parameter=:n)
    @test length(unique(p2.points.series)) == 2
    @test_throws ArgumentError DrSnow._viz_power_curve_data(f.g1; estimator="nope")
    po = DrSnow._viz_power_curve_data(f.opt)
    @test po.target == 0.8 && po.best == f.opt.best.n
    @test nrow(po.curve) == length(20:10:300)
    @test sum(po.points.x .> 0) == length(unique(f.opt.evaluations.n))
    pa = DrSnow._viz_power_curve_data(f.an)
    @test pa.parameter == "n" && pa.points.power ≈ [r.power for r in f.an]
    @test_throws ArgumentError DrSnow._viz_power_curve_data(f.an; parameter=:icc)
    @test_throws ArgumentError DrSnow._viz_power_curve_data(1.0)
    b = DrSnow._viz_blocks_data(f.bd)
    @test nrow(b) == 30 && sort(unique(b.position)) == 1:10
    m = [mean(b.score[b.position .== k]) for k in 1:10]
    @test issorted(m)
    @test_throws ArgumentError DrSnow._viz_blocks_data(f.bdc)
end

if VIZ_MAKIE
    @testset "design Makie plots" begin
        f = DES_VIZ_FIX
        CM = CairoMakie
        outdir = mktempdir()
        save_ok(fig, name) = (p = joinpath(outdir, name * ".png");
                              CM.save(p, fig; px_per_unit=1); filesize(p) > 5_000)
        for (x, kw, name) in ((f.g1, (;), "grid"), (f.g2, (parameter=:n,), "grid2"),
                              (f.opt, (;), "opt"), (f.an, (target=0.8,), "analytic"))
            fig = plot_power_curve(x; kw...)
            @test fig isa CM.Figure
            @test save_ok(fig, name)
        end
        fig = CM.Figure()
        ax = CM.Axis(fig[1, 1])
        @test plot_power_curve!(ax, f.g1) === ax
        fig = plot_blocks(f.bd)
        @test fig isa CM.Figure && save_ok(fig, "blocks")
        @test_throws ArgumentError plot_blocks(f.bdc)
    end
end
