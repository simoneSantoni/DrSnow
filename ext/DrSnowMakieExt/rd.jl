# Regression discontinuity plots (rdplot layout).

function plot_rd!(ax::Makie.AbstractAxis, pd::DrSnow.RDPlotData;
                  estimate::Union{Nothing,DrSnow.RDEstimate}=nothing, ci::Bool=false,
                  colors=nothing)
    cols = _mk_colors(colors, 2)
    b = pd.bins
    if estimate !== nothing
        vspan!(ax, estimate.cutoff - estimate.h_left, estimate.cutoff + estimate.h_right;
               color=(cols[2], 0.08))
    end
    vlines!(ax, pd.cutoff; color=_MK_REF, linestyle=:dash, linewidth=1)
    if ci
        ok = _mk_has_ci(b.ci_lower, b.ci_upper)
        any(ok) && rangebars!(ax, b.mean_x[ok], Float64.(b.ci_lower[ok]),
                              Float64.(b.ci_upper[ok]); color=(_MK_INK2, 0.6),
                              linewidth=1, whiskerwidth=4)
    end
    scatter!(ax, b.mean_x, b.mean_y; color=_MK_INK2, markersize=7, strokecolor=:white,
             strokewidth=0.8, label="Bin means")
    for side in (:left, :right)
        p = pd.poly[pd.poly.side .== side, :]
        lines!(ax, p.x, p.y; color=cols[1], linewidth=2,
               label="Global polynomial (p = $(pd.p))")
    end
    if estimate !== nothing
        lf = DrSnow._viz_rd_local_fit(estimate)
        for side in (:left, :right)
            p = lf[lf.side .== side, :]
            lines!(ax, p.x, p.y; color=cols[2], linewidth=2.5, linestyle=:dash,
                   label="Local polynomial (p = $(estimate.p)) within bandwidth")
        end
    end
    return ax
end

function plot_rd(pd::DrSnow.RDPlotData; estimate::Union{Nothing,DrSnow.RDEstimate}=nothing,
                 ci::Bool=false, colors=nothing, figure::NamedTuple=(;),
                 axis::NamedTuple=(;))
    parts = ["$(sum(pd.nbins)) bins ($(pd.binselect) selection)"]
    ci && push!(parts, "whiskers: $(_mk_pct(pd.level)) CIs of bin means")
    if estimate !== nothing
        lo, hi = StatsAPI.confint(estimate; level=estimate.level)[1, :]
        push!(parts, "RD estimate $(_mk_fmt(DrSnow.estimate(estimate))), robust " *
                     "$(_mk_pct(estimate.level)) CI [$(_mk_fmt(lo)), $(_mk_fmt(hi))]")
        push!(parts, "shaded: bandwidth")
    end
    sub = _mk_caption(parts)
    fig = _mk_figure(figure)
    ax = _mk_axis(fig[1, 1], axis; xlabel="Running variable", ylabel="Outcome",
                  subtitle=sub)
    plot_rd!(ax, pd; estimate=estimate, ci=ci, colors=colors)
    _mk_legend!(fig, ax)
    return fig
end
