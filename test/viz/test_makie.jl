@testset "Makie plots" begin
    f = VIZ_FIX
    CM = CairoMakie
    ext = Base.get_extension(DrSnow, :DrSnowMakieExt)
    @test ext !== nothing
    outdir = mktempdir()
    first_axis(fig) = only(filter(c -> c isa CM.Axis, fig.content))
    axes_of(fig) = filter(c -> c isa CM.Axis, fig.content)
    legends(fig) = filter(c -> c isa CM.Legend, fig.content)
    nplots(ax, T) = count(p -> p isa T, ax.scene.plots)

    function check_save(fig, name)
        path = joinpath(outdir, name * ".png")
        CM.save(path, fig; px_per_unit=1)
        return isfile(path) && filesize(path) > 5_000
    end

    @testset "event studies" begin
        fig = plot_event_study(f.es_cs; uniform=true, rng=StableRNG(1))
        @test fig isa CM.Figure
        ax = first_axis(fig)
        @test isempty(legends(fig))                       # single series
        @test nplots(ax, CM.Rangebars) == 2               # pointwise + uniform
        @test occursin("uniform", ax.subtitle[])
        @test check_save(fig, "es_single")
        fig = plot_event_study(f.es_twfe, f.es_sa, f.es_cs, f.es_imp;
                               labels=["TWFE", "SA", "CS", "BJS"], connect=true)
        @test length(legends(fig)) == 1
        ax = first_axis(fig)
        @test nplots(ax, CM.Lines) == 4
        ticks = ax.xticks[]
        @test first(ticks) == sort(unique(vcat(relative_periods.(
            [f.es_twfe, f.es_sa, f.es_cs, f.es_imp])..., f.es_twfe.reference)))
        @test check_save(fig, "es_overlay")
        # draw into an existing axis
        fig2 = CM.Figure()
        ax2 = CM.Axis(fig2[1, 1])
        @test plot_event_study!(ax2, f.es_sa) === ax2
        @test_throws ArgumentError plot_event_study(f.es_cs, f.es_sa; labels=["one"])
        @test_throws ArgumentError plot_event_study(f.did)
        # more series than palette colours need explicit colours
        many = fill(f.es_sa, 9)
        @test_throws ArgumentError plot_event_study(many...)
        @test plot_event_study(many...; colors=fill(:black, 9),
                               labels=string.(1:9)) isa CM.Figure
        @test_throws ArgumentError plot_event_study(f.es_sa, f.es_cs; colors=[:red])
    end

    @testset "coefficients" begin
        fig = plot_coefficients([f.did, f.sdid, f.sc, f.dml, f.rd];
                                labels=["TWFE", "SDID", "SC", "DML", "RD"])
        ax = first_axis(fig)
        @test ax.yticks[][2] == ["TWFE", "SDID", "SC", "DML", "RD"]
        @test isempty(legends(fig))
        @test check_save(fig, "coef_single")
        fig = plot_coefficients([f.iv, f.ring_did])
        @test length(legends(fig)) == 1
        @test check_save(fig, "coef_multi")
        @test plot_coefficients(f.iv; terms=["d"]) isa CM.Figure
        # a result without variance is drawn without an interval (with a warning)
        @test_logs (:warn,) match_mode = :any plot_coefficients([f.sdid_nose, f.did])
        ax = CM.Axis(CM.Figure()[1, 1])
        @test plot_coefficients!(ax, f.iv) === ax
        @test_throws ArgumentError plot_coefficients("not a result")
    end

    @testset "regression discontinuity" begin
        fig = plot_rd(f.rdp; estimate=f.rd, ci=true)
        ax = first_axis(fig)
        @test nplots(ax, CM.Lines) == 4                   # 2 global + 2 local fits
        @test nplots(ax, CM.VSpan) == 1
        @test occursin("RD estimate", ax.subtitle[])
        @test check_save(fig, "rd")
        fig = plot_rd(f.rdp)
        @test nplots(first_axis(fig), CM.Lines) == 2
        ax = CM.Axis(CM.Figure()[1, 1])
        @test plot_rd!(ax, f.rdp) === ax
    end

    @testset "synthetic control" begin
        fig = plot_synth(f.sc)
        @test length(axes_of(fig)) == 2 && length(legends(fig)) == 1
        @test occursin("Placebo", join(legends(fig)[1].entrygroups[][1][2] .|>
                                       e -> e.label[], " "))
        @test check_save(fig, "synth_sc")
        fig = plot_synth(f.sdid; kind=:trajectories)
        ax = first_axis(fig)
        @test nplots(ax, CM.BarPlot) == 1                 # time weights
        @test check_save(fig, "synth_sdid")
        @test nplots(first_axis(plot_synth(f.sdid; kind=:trajectories,
                                           time_weights=false)), CM.BarPlot) == 0
        @test plot_synth(f.ascm; kind=:gaps) isa CM.Figure
        @test plot_synth(f.mc) isa CM.Figure
        @test plot_synth(f.sc; placebo_cutoff=2.0) isa CM.Figure
        @test_throws ArgumentError plot_synth(f.sc; kind=:nope)
        ax = CM.Axis(CM.Figure()[1, 1])
        @test plot_synth!(ax, f.sc; kind=:gaps) === ax
        @test_throws ArgumentError plot_synth!(ax, f.sc; kind=:both)
    end

    @testset "randomization distributions" begin
        fig = plot_randomization_distribution(f.ri)
        ax = first_axis(fig)
        @test nplots(ax, CM.Hist) == 1
        @test nplots(ax, CM.VLines) == 2                  # observed and mirror image
        @test occursin("p = ", ax.subtitle[])
        @test check_save(fig, "ri")
        fig = plot_randomization_distribution(synth_in_space_placebo(f.sc); bins=8)
        @test nplots(first_axis(fig), CM.VLines) == 1     # one-sided
        @test first_axis(fig).ylabel[] == "Share of placebo units"
        @test_throws ArgumentError plot_randomization_distribution(f.did)
    end

    @testset "Bacon, GATES, CATE" begin
        fig = plot_bacon(f.bacon)
        ntypes = length(unique(f.bacon.comparisons.type))
        @test nplots(first_axis(fig), CM.Scatter) == ntypes
        @test check_save(fig, "bacon")
        fig = plot_gates(f.gml)
        ax = first_axis(fig)
        @test length(ax.xticks[][1]) == f.gml.n_groups
        @test check_save(fig, "gates")
        @test check_save(plot_cate(f.cate), "cate")
        @test check_save(plot_cate(f.cate; modifier=:x1), "cate_x1")
        @test_throws ArgumentError plot_cate(f.cate; modifier=:nope)
    end

    @testset "spillovers and confidence sets" begin
        fig = plot_spillover_rings(f.ring_es)
        @test length(legends(fig)) == 1
        @test check_save(fig, "rings_es")
        fig = plot_spillover_rings(f.ring_did)
        @test first_axis(fig).xticks[][2] == coefnames(f.ring_did)
        @test check_save(fig, "rings_did")
        fig = plot_confidence_set(f.ar; wald=f.iv)
        ax = first_axis(fig)
        @test nplots(ax, CM.VSpan) == length(f.ar.intervals)
        @test check_save(fig, "ar")
        @test plot_confidence_set(f.ar_weak; wald=f.ivweak) isa CM.Figure
        @test plot_confidence_set(f.ar; limits=(-1, 1), npoints=50) isa CM.Figure
    end

    @testset "wrong argument types with a backend loaded" begin
        @test_throws MethodError plot_rd(f.did)
        @test_throws MethodError plot_bacon(f.did)
    end

    @testset "Aqua on the extensions" begin
        rt = Base.get_extension(DrSnow, :DrSnowRegressionTablesExt)
        @test rt !== nothing
        # the extensions add methods to DrSnow's own functions / for DrSnow's types,
        # which Aqua attributes to the parent package
        own = Any[DrSnow.CausalEstimate;
                  [getfield(DrSnow, n) for n in names(DrSnow)
                   if startswith(string(n), "plot_") || n === :drsnow_theme]]
        Aqua.test_piracies(ext; treat_as_own=own)
        Aqua.test_piracies(rt; treat_as_own=own)
        @test isempty(Test.detect_ambiguities(ext, rt; recursive=false))
    end
end
