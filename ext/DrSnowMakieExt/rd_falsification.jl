# RD falsification plots: placebo cutoffs, bandwidth / donut sensitivity, density test.

function _mk_draw_rd_placebos!(ax, d; colors=nothing)
    t = d.table
    cols = _mk_colors(colors, 2)
    hlines!(ax, 0.0; color=_MK_REF, linewidth=1)
    act = t[t.true_cutoff, :]
    pl = t[.!t.true_cutoff, :]
    if nrow(act) > 0
        vlines!(ax, act.cutoff; color=_MK_REF, linestyle=:dash, linewidth=1)
    end
    nrow(pl) > 0 && rangebars!(ax, pl.cutoff, pl.conf_low, pl.conf_high; color=cols[1],
                               linewidth=1.5, whiskerwidth=7)
    nrow(pl) > 0 && scatter!(ax, pl.cutoff, pl.estimate; color=cols[1], marker=:circle,
                             markersize=10, strokecolor=:white, strokewidth=1,
                             label="Placebo cutoffs")
    if nrow(act) > 0
        rangebars!(ax, act.cutoff, act.conf_low, act.conf_high; color=cols[2],
                   linewidth=2, whiskerwidth=9)
        scatter!(ax, act.cutoff, act.estimate; color=cols[2], marker=:diamond,
                 markersize=13, strokecolor=:white, strokewidth=1,
                 label="Estimate at the true cutoff")
    end
    return ax
end

function DrSnow.plot_rd_placebos!(ax::Makie.AbstractAxis, tab;
                                  estimate::Union{Nothing,DrSnow.RDEstimate}=nothing,
                                  colors=nothing)
    d = DrSnow._viz_rd_placebo_data(tab; estimate=estimate)
    return _mk_draw_rd_placebos!(ax, d; colors=colors)
end

function DrSnow.plot_rd_placebos(tab; estimate::Union{Nothing,DrSnow.RDEstimate}=nothing,
                                 colors=nothing, figure::NamedTuple=(;),
                                 axis::NamedTuple=(;))
    d = DrSnow._viz_rd_placebo_data(tab; estimate=estimate)
    parts = ["Whiskers: robust bias-corrected CIs",
             "placebos left of the true cutoff use control observations only, right " *
             "of it treated observations only"]
    estimate === nothing || push!(parts, "dashed: true cutoff")
    d.n_failed > 0 && push!(parts, "$(d.n_failed) placebo cutoff(s) not estimable")
    fig = _mk_figure(figure)
    ax = _mk_axis(fig[1, 1], axis; xlabel="Cutoff (running variable)",
                  ylabel="RD estimate", subtitle=_mk_caption(parts))
    _mk_draw_rd_placebos!(ax, d; colors=colors)
    _mk_legend!(fig, ax)
    return fig
end

function _mk_draw_rd_sensitivity!(ax, d; colors=nothing)
    t = d.table
    cols = _mk_colors(colors, 2)
    hlines!(ax, 0.0; color=_MK_REF, linewidth=1)
    b = t[t.baseline, :]
    nb = t[.!t.baseline, :]
    nrow(b) > 0 && hlines!(ax, b.estimate[1]; color=_MK_REF, linestyle=:dot,
                           linewidth=1)
    base_lbl = d.kind === :bandwidth ? "Data-driven bandwidth" : "No donut"
    other_lbl = d.kind === :bandwidth ? "Scaled bandwidths" : "Donut radii"
    if nrow(nb) > 0
        rangebars!(ax, nb.x, nb.conf_low, nb.conf_high; color=cols[1], linewidth=1.5,
                   whiskerwidth=7)
        scatter!(ax, nb.x, nb.estimate; color=cols[1], markersize=10,
                 strokecolor=:white, strokewidth=1, label=other_lbl)
    end
    if nrow(b) > 0
        rangebars!(ax, b.x, b.conf_low, b.conf_high; color=cols[2], linewidth=2,
                   whiskerwidth=9)
        scatter!(ax, b.x, b.estimate; color=cols[2], marker=:diamond, markersize=13,
                 strokecolor=:white, strokewidth=1, label=base_lbl)
    end
    return ax
end

function DrSnow.plot_rd_sensitivity!(ax::Makie.AbstractAxis, tab; colors=nothing)
    d = DrSnow._viz_rd_sensitivity_data(tab)
    return _mk_draw_rd_sensitivity!(ax, d; colors=colors)
end

function DrSnow.plot_rd_sensitivity(tab; colors=nothing, figure::NamedTuple=(;),
                                    axis::NamedTuple=(;))
    d = DrSnow._viz_rd_sensitivity_data(tab)
    parts = ["Whiskers: robust bias-corrected CIs"]
    if d.kind === :bandwidth
        push!(parts, "dotted: estimate at the data-driven bandwidth")
        d.asymmetric && push!(parts, "x: average of left and right bandwidths")
        xl = "Main bandwidth h"
    else
        push!(parts, "dotted: estimate without a donut")
        xl = "Donut radius (observations with |x − c| < radius excluded)"
    end
    d.n_failed > 0 && push!(parts, "$(d.n_failed) row(s) not estimable")
    fig = _mk_figure(figure)
    ax = _mk_axis(fig[1, 1], axis; xlabel=xl, ylabel="RD estimate",
                  subtitle=_mk_caption(parts))
    _mk_draw_rd_sensitivity!(ax, d; colors=colors)
    if d.kind === :bandwidth
        # secondary labels: multipliers under the bandwidths
        ax.xticks = (d.table.x, [_mk_fmt(x) * "\n" * l for (x, l) in
                                 zip(d.table.x, d.table.label)])
    end
    _mk_legend!(fig, ax)
    return fig
end

function _mk_draw_rd_density!(ax, d; ci::Bool=true, colors=nothing)
    cols = _mk_colors(colors, 2)
    e = d.hist.edges
    mids = (e[1:end-1] .+ e[2:end]) ./ 2
    barplot!(ax, mids, d.hist.density; width=diff(e), gap=0.08, color=(_MK_MUTED, 0.7),
             strokewidth=0, label="Histogram")
    vlines!(ax, d.cutoff; color=_MK_REF, linestyle=:dash, linewidth=1)
    for (i, side, lbl) in ((1, :left, "Density estimate, left"),
                           (2, :right, "Density estimate, right"))
        s = d.density[d.density.side .== side, :]
        nrow(s) == 0 && continue
        if ci
            ok = _mk_has_ci(s.conf_low, s.conf_high)
            any(ok) && band!(ax, s.x[ok], max.(0.0, Float64.(s.conf_low[ok])),
                             Float64.(s.conf_high[ok]); color=(cols[i], 0.2))
        end
        lines!(ax, s.x, s.f; color=cols[i], linewidth=2.2,
               linestyle=side === :left ? :solid : :dash, label=lbl)
    end
    xlims!(ax, d.limits...)
    ylims!(ax, 0, nothing)
    return ax
end

function DrSnow.plot_rd_density!(ax::Makie.AbstractAxis, args...; npoints::Integer=40,
                                 bins=nothing, limits=nothing, level::Real=0.95,
                                 ci::Bool=true, colors=nothing)
    d = DrSnow._viz_rd_density_data(args...; npoints=npoints, bins=bins, limits=limits,
                                    level=level)
    return _mk_draw_rd_density!(ax, d; ci=ci, colors=colors)
end

DrSnow.plot_rd_density(data::DataFrames.AbstractDataFrame, running::Symbol,
                       t::DiagnosticTest; kwargs...) =
    DrSnow.plot_rd_density(data[!, running], t; kwargs...)

function DrSnow.plot_rd_density(x::AbstractVector, t::DiagnosticTest; npoints::Integer=40,
                                bins=nothing, limits=nothing, level::Real=0.95,
                                ci::Bool=true, colors=nothing, figure::NamedTuple=(;),
                                axis::NamedTuple=(;))
    d = DrSnow._viz_rd_density_data(x, t; npoints=npoints, bins=bins, limits=limits,
                                    level=level)
    parts = ["Density test (Cattaneo, Jansson & Ma): robust t = " *
             "$(_mk_fmt(d.statistic)), " *
             "$(_mk_peq(d.pvalue))",
             "local polynomial (p = $(d.p)) fits by side, bandwidths " *
             "($(_mk_fmt(d.h[1])), $(_mk_fmt(d.h[2])))"]
    ci && push!(parts, "bands: $(_mk_pct(d.level)) pointwise")
    fig = _mk_figure(figure)
    ax = _mk_axis(fig[1, 1], axis; xlabel="Running variable", ylabel="Density",
                  subtitle=_mk_caption(parts))
    _mk_draw_rd_density!(ax, d; ci=ci, colors=colors)
    _mk_legend!(fig, ax)
    return fig
end
