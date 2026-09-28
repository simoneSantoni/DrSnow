# Plot data and Makie plots for the sequential area (confidence-sequence paths,
# group-sequential boundaries).

const SEQ_VIZ = let rng = StableRNG(55)
    n = 300
    df = DataFrame(d=rand(rng, n) .< 0.5)
    df.y = randn(rng, n) .+ 0.4 .* df.d
    cs = confseq_ate(df, :y, :d; propensity=0.5)
    mt = msprt_test(df, :y, :d)
    d = gs_design(; k=3, futility=HSDSpending(-2))
    an = gs_analysis(d, [0.2, 0.3], [0.2, 0.14])
    (; df, cs, mt, d, an)
end

@testset "sequential plot data" begin
    f = SEQ_VIZ
    cd = DrSnow._viz_confseq_data(f.cs)
    @test nrow(cd.path) == 300 && cd.null == 0 && cd.level == 0.95
    @test all(isnan, cd.path.lower[1:19]) && all(!isnan, cd.path.lower[20:end])
    ok = 20:300
    # without running intersection the CS contains the fixed-n CI at every n
    cn = DrSnow._viz_confseq_data(confseq_ate(f.df, :y, :d; propensity=0.5,
                                              running_intersection=false))
    @test all(cn.path.lower[ok] .<= cn.path.fixed_low[ok])
    @test all(cn.path.fixed_high[ok] .<= cn.path.upper[ok])
    md = DrSnow._viz_confseq_data(f.mt)
    @test md.null == 0 && all(isnan, md.path.fixed_low)
    m = MeanMonitor(; method=:betting, bounds=(0, 1))
    fit!(m, rand(StableRNG(1), 50))
    @test nrow(DrSnow._viz_confseq_data(m).path) == 50
    @test_throws ArgumentError DrSnow._viz_confseq_data(f.d)
    gd = DrSnow._viz_gs_data(f.d)
    @test gd.bounds.efficacy == f.d.efficacy_z && gd.observed === nothing
    ga = DrSnow._viz_gs_data(f.an)
    @test ga.observed.z == f.an.z
    @test gd.fixed ≈ 1.959964 atol = 1e-6
    @test_throws ArgumentError DrSnow._viz_gs_data(f.cs)
    @test DrSnow._viz_nominal_p(1.959964) ≈ 0.025 atol = 1e-7
end

if VIZ_MAKIE
    @testset "sequential Makie plots" begin
        f = SEQ_VIZ
        CM = CairoMakie
        outdir = mktempdir()
        save_ok(fig, name) = (p = joinpath(outdir, name * ".png");
                              CM.save(p, fig; px_per_unit=1); filesize(p) > 5_000)
        axes_of(fig) = filter(c -> c isa CM.Axis, fig.content)
        fig = plot_confidence_sequence(f.cs)
        ax = only(axes_of(fig))
        @test count(p -> p isa CM.Band, ax.scene.plots) == 1
        @test save_ok(fig, "cs")
        @test save_ok(plot_confidence_sequence(f.mt; fixed=false), "msprt")
        fig = plot_gs_boundaries(f.an)
        @test count(p -> p isa CM.ScatterLines, only(axes_of(fig)).scene.plots) == 3
        @test save_ok(fig, "gs")
        @test save_ok(plot_gs_boundaries(f.d; scale=:p), "gs_p")
        fig2 = CM.Figure()
        @test plot_confidence_sequence!(CM.Axis(fig2[1, 1]), f.cs) isa CM.Axis
        @test plot_gs_boundaries!(CM.Axis(fig2[1, 2]), f.d) isa CM.Axis
        @test_throws ArgumentError plot_gs_boundaries(f.d; scale=:foo)
    end
end
