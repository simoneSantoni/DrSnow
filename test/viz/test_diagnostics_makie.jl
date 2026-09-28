@testset "Makie diagnostic plots" begin
    f = VIZ_DIAG
    g = VIZ_FIX
    CM = CairoMakie
    outdir = mktempdir()
    axes_of(fig) = filter(c -> c isa CM.Axis, fig.content)
    first_axis(fig) = first(axes_of(fig))
    legends(fig) = filter(c -> c isa CM.Legend, fig.content)
    nplots(ax, T) = count(p -> p isa T, ax.scene.plots)
    function check_save(fig, name)
        path = joinpath(outdir, name * ".png")
        CM.save(path, fig; px_per_unit=1)
        return isfile(path) && filesize(path) > 5_000
    end
    newax() = CM.Axis(CM.Figure()[1, 1])

    @testset "trends" begin
        fig = plot_trends(f.panel, :y, :d, :unit, :time)
        ax = first_axis(fig)
        @test nplots(ax, CM.Band) == 3
        @test nplots(ax, CM.VLines) == 2                  # two adoption dates
        @test length(legends(fig)) == 1
        @test check_save(fig, "trends")
        fig = plot_trends(f.panel, :y, f.tm; by=:treated, covariates=[:x], ci=false)
        ax = first_axis(fig)
        @test nplots(ax, CM.Band) == 0
        @test occursin("adjusted", ax.ylabel[])
        ax = newax()
        @test plot_trends!(ax, f.panel, :y, :d, :unit, :time) === ax
        @test plot_trends!(ax, f.panel, :y, f.tm; by=:treated) === ax
    end

    @testset "balance" begin
        fig = plot_balance(f.pb)
        ax = first_axis(fig)
        @test length(legends(fig)) == 1                   # one series per cohort
        @test nplots(ax, CM.Scatter) == 2
        @test check_save(fig, "balance_pre")
        fig = plot_balance(f.rib)
        @test length(axes_of(fig)) == 2                   # p-value column
        @test occursin("Omnibus", first_axis(fig).subtitle[])
        @test check_save(fig, "balance_ri")
        fig = plot_balance(f.rdcb; data=f.rdd)
        @test nplots(first_axis(fig), CM.Rangebars) == 1
        @test occursin("SD units", first_axis(fig).xlabel[])
        @test check_save(fig, "balance_rd")
        @test length(axes_of(plot_balance(f.rdcb; annotate=false))) == 1
        @test nplots(first_axis(plot_balance(f.pb; threshold=false)), CM.VLines) == 1
        ax = newax()
        @test plot_balance!(ax, f.rib) === ax
        @test occursin("p = ", ax.yticks[][2][1])        # labels carry p-values
        @test_throws ArgumentError plot_balance(DataFrame(a=[1]))
    end

    @testset "RD falsification" begin
        fig = plot_rd_placebos(f.rdpl; estimate=f.rd)
        @test nplots(first_axis(fig), CM.Rangebars) == 2
        @test check_save(fig, "rd_placebos")
        @test plot_rd_placebos(f.rdpl) isa CM.Figure
        fig = plot_rd_sensitivity(f.rdbw)
        @test occursin("×1", join(first_axis(fig).xticks[][2]))
        @test check_save(fig, "rd_bw")
        @test check_save(plot_rd_sensitivity(f.rddo), "rd_donut")
        fig = plot_rd_density(f.xr, f.rdden)
        ax = first_axis(fig)
        @test nplots(ax, CM.BarPlot) == 1 && nplots(ax, CM.Band) == 2
        @test occursin("p < 0.001", ax.subtitle[])
        @test check_save(fig, "rd_density")
        @test nplots(first_axis(plot_rd_density(f.rdd, :x, f.rdden_ok; ci=false)),
                     CM.Band) == 0
        for (fn, args) in ((plot_rd_placebos!, (f.rdpl,)),
                           (plot_rd_sensitivity!, (f.rddo,)),
                           (plot_rd_density!, (f.rdd, :x, f.rdden_ok)))
            ax = newax()
            @test fn(ax, args...) === ax
        end
    end

    @testset "Honest DiD" begin
        fig = plot_honest_did(f.hd_rm; breakdown=f.bd_rm)
        ax = first_axis(fig)
        @test nplots(ax, CM.Rangebars) == 2               # original + robust
        @test ax.xticks[][2][1] == "Orig."
        @test any(p -> p isa CM.VLines, ax.scene.plots)   # breakdown line
        @test check_save(fig, "honest_rm")
        fig = plot_honest_did(f.hd_sd)
        @test occursin("no breakdown", first_axis(fig).subtitle[])
        ax = newax()
        @test plot_honest_did!(ax, f.hd_sd) === ax
    end

    @testset "IV design plots" begin
        fig = plot_judge_first_stage(f.jd, f.jiv)
        @test length(axes_of(fig)) == 2                   # first stage + histogram
        @test check_save(fig, "judge")
        fig = plot_judge_first_stage(f.jd, :d, f.jiv.leniency; histogram=false,
                                     nbins=10)
        @test length(axes_of(fig)) == 1
        jd = copy(f.jd)
        jd.len = f.jiv.leniency
        @test plot_judge_first_stage(jd, :d, :len) isa CM.Figure
        @test_throws ArgumentError plot_judge_first_stage(f.jd)
        ax = newax()
        @test plot_judge_first_stage!(ax, f.jd, f.jiv) === ax

        fig = plot_rotemberg(f.rw; label=3)
        ax = first_axis(fig)
        @test nplots(ax, CM.Text) == 3
        @test check_save(fig, "rotemberg")
        @test plot_rotemberg(f.rw; x=:shock) isa CM.Figure
        @test_throws ArgumentError plot_rotemberg(f.rw; x=:alpha)

        fig = plot_mte(f.mte_poly)
        ax = first_axis(fig)
        @test nplots(ax, CM.Band) == 1
        @test nplots(ax, CM.HLines) == 1 + length(coef(f.mte_poly))  # zero + params
        @test check_save(fig, "mte_poly")
        fig = plot_mte(f.mte_semi; parameters=false, propensity=false)
        @test nplots(first_axis(fig), CM.LineSegments) == 0
        @test check_save(fig, "mte_semi")
        ax = newax()
        @test plot_mte!(ax, f.mte_poly) === ax
    end

    @testset "synthetic control backdating" begin
        fig = plot_synth_in_time(f.sc, f.sc_back)
        @test nplots(first_axis(fig), CM.Lines) == 3
        @test check_save(fig, "synth_in_time")
        ax = newax()
        @test plot_synth_in_time!(ax, f.sc, f.sc_back) === ax
    end

    @testset "theme" begin
        th = drsnow_theme()
        @test th isa CM.Theme
        @test th.fontsize[] == 14
        @test drsnow_theme(; font=:serif, fontsize=9).fontsize[] == 9
        @test_throws ArgumentError drsnow_theme(; font=:mono)
        @test_throws ArgumentError drsnow_theme(; colors=[])
        # plots honor the theme's figure-level settings
        fig = CM.with_theme(drsnow_theme(; fontsize=9)) do
            plot_trends(f.panel, :y, :d, :unit, :time)
        end
        @test fig.scene.theme.fontsize[] == 9
        @test check_save(fig, "theme_small")
    end

    @testset "figure / axis keywords on every plot" begin
        FIG = (size=(640, 420),)
        AX = (title="Custom title",)
        calls = Any[
            () -> plot_event_study(g.es_sa; figure=FIG, axis=AX),
            () -> plot_coefficients(g.did; figure=FIG, axis=AX),
            () -> plot_rd(g.rdp; figure=FIG, axis=AX),
            () -> plot_synth(g.sc; kind=:gaps, figure=FIG, axis=AX),
            () -> plot_randomization_distribution(g.ri; figure=FIG, axis=AX),
            () -> plot_bacon(g.bacon; figure=FIG, axis=AX),
            () -> plot_gates(g.gml; figure=FIG, axis=AX),
            () -> plot_cate(g.cate; figure=FIG, axis=AX),
            () -> plot_spillover_rings(g.ring_did; figure=FIG, axis=AX),
            () -> plot_confidence_set(g.ar; figure=FIG, axis=AX),
            () -> plot_trends(f.panel, :y, :d, :unit, :time; figure=FIG, axis=AX),
            () -> plot_balance(f.rib; figure=FIG, axis=AX),
            () -> plot_rd_placebos(f.rdpl; figure=FIG, axis=AX),
            () -> plot_rd_sensitivity(f.rdbw; figure=FIG, axis=AX),
            () -> plot_rd_density(f.xr, f.rdden; figure=FIG, axis=AX),
            () -> plot_honest_did(f.hd_rm; figure=FIG, axis=AX),
            () -> plot_judge_first_stage(f.jd, f.jiv; figure=FIG, axis=AX),
            () -> plot_rotemberg(f.rw; figure=FIG, axis=AX),
            () -> plot_mte(f.mte_poly; figure=FIG, axis=AX),
            () -> plot_synth_in_time(f.sc, f.sc_back; figure=FIG, axis=AX),
        ]
        for c in calls
            fig = c()
            @test fig isa CM.Figure
            @test Tuple(fig.scene.viewport[].widths) == (640, 420)
            @test first_axis(fig).title[] == "Custom title"
        end
    end
end
