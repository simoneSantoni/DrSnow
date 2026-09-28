# Plot data and Makie plots for generalized random forests (variable importance,
# CATE by covariate, RATE / TOC curves).

const GRF_FIX = let rng = StableRNG(321)
    n = 400
    X = rand(rng, n, 3)
    d = Float64.(rand(rng, n) .< 0.5)
    y = X[:, 2] .+ (1 .+ 2 .* X[:, 1]) .* d .+ randn(rng, n)
    df = DataFrame(X, [:x1, :x2, :x3])
    df.d = d
    df.y = y
    cf = causal_forest(df, :y, :d; covariates=[:x1, :x2, :x3], num_trees=200,
                       w_hat=0.5, rng=StableRNG(1))
    rate = rank_average_treatment_effect(cf, hcat(predict(cf), df.x3); R=50,
                                         rng=StableRNG(2))
    (; df, cf, rate)
end

@testset "forest plot data" begin
    f = GRF_FIX
    vi = DrSnow._viz_variable_importance_data(f.cf)
    @test issorted(vi.importance; rev=true)
    @test sort(vi.variable) == ["x1", "x2", "x3"]
    @test nrow(DrSnow._viz_variable_importance_data(f.cf; top=2)) == 2
    @test_throws ArgumentError DrSnow._viz_variable_importance_data(f.cf; top=0)
    @test_throws ArgumentError DrSnow._viz_variable_importance_data(f.rate)
    cd = DrSnow._viz_forest_cate_data(f.cf; modifier=:x1, level=0.9)
    @test issorted(cd.cate.x)
    @test all(cd.cate.conf_low .<= cd.cate.estimate .<= cd.cate.conf_high)
    @test cd.ate.estimate ≈ coef(average_treatment_effect(f.cf))[1]
    @test cd.label == "CATE" && cd.level == 0.9
    @test all(ismissing, DrSnow._viz_forest_cate_data(f.cf).cate.x)
    @test_throws ArgumentError DrSnow._viz_forest_cate_data(f.cf; modifier=:nope)
    rd = DrSnow._viz_rate_data(f.rate; level=0.9)
    @test nrow(rd.toc) == 3 * 10                     # two rules + difference
    @test all(rd.toc.conf_low .<= rd.toc.estimate .<= rd.toc.conf_high)
    @test rd.rate.estimate == coef(f.rate)
    @test_throws ArgumentError DrSnow._viz_rate_data(f.cf)
end

if VIZ_MAKIE
    @testset "forest Makie plots" begin
        f = GRF_FIX
        CM = CairoMakie
        outdir = mktempdir()
        save_ok(fig, name) = (p = joinpath(outdir, name * ".png");
                              CM.save(p, fig; px_per_unit=1); filesize(p) > 5_000)
        axes_of(fig) = filter(c -> c isa CM.Axis, fig.content)
        fig = plot_variable_importance(f.cf)
        @test fig isa CM.Figure
        # the most important covariate is drawn at the top (largest y position)
        yt = only(axes_of(fig)).yticks[]
        @test yt[2][argmax(yt[1])] ==
              DrSnow._viz_variable_importance_data(f.cf).variable[1]
        @test save_ok(fig, "varimp")
        @test save_ok(plot_variable_importance(f.cf; top=2), "varimp_top")
        @test save_ok(plot_cate(f.cf), "cf_hist")
        fig = plot_cate(f.cf; modifier=:x1, level=0.9)
        ax = only(axes_of(fig))
        @test count(p -> p isa CM.Rangebars, ax.scene.plots) == 1
        @test save_ok(fig, "cf_x1")
        @test_throws ArgumentError plot_cate(f.cf; modifier=:nope)
        fig = plot_rate(f.rate)
        ax = only(axes_of(fig))
        @test count(p -> p isa CM.Lines, ax.scene.plots) == 3
        @test save_ok(fig, "rate")
        fig2 = CM.Figure()
        ax2 = CM.Axis(fig2[1, 1])
        @test plot_rate!(ax2, f.rate) === ax2
        @test plot_variable_importance!(CM.Axis(fig2[1, 2]), f.cf) isa CM.Axis
        @test_throws MethodError plot_rate(f.cf)
    end
end
