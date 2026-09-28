# Confidence-sequence paths and group-sequential boundaries.

function _mk_draw_confseq!(ax, cd; fixed::Bool, null::Bool, colors)
    cols = _mk_colors(colors, 2)
    p = cd.path
    x = Float64.(p.n)
    ok = .!isnan.(p.lower) .& .!isnan.(p.upper)
    if any(ok)
        band!(ax, x[ok], p.lower[ok], p.upper[ok]; color=(cols[1], 0.25),
              label="$(round(100 * cd.level; digits=1))% confidence sequence")
    end
    lines!(ax, x, p.estimate; color=cols[1], linewidth=2, label="Estimate")
    if fixed && any(!isnan, p.fixed_low)
        lines!(ax, x, p.fixed_low; color=cols[2], linestyle=:dash, linewidth=1.2,
               label="Fixed-n CI (valid at one n only)")
        lines!(ax, x, p.fixed_high; color=cols[2], linestyle=:dash, linewidth=1.2)
    end
    null && isfinite(cd.null) &&
        hlines!(ax, cd.null; color=_MK_REF, linestyle=:dot, linewidth=1.2,
                label="Null value")
    return ax
end

function plot_confidence_sequence!(ax::Makie.AbstractAxis, r; fixed::Bool=true,
                                   null::Bool=true, colors=nothing)
    return _mk_draw_confseq!(ax, DrSnow._viz_confseq_data(r); fixed=fixed, null=null,
                             colors=colors)
end

function plot_confidence_sequence(r; fixed::Bool=true, null::Bool=true, colors=nothing,
                                  figure::NamedTuple=(;), axis::NamedTuple=(;))
    cd = DrSnow._viz_confseq_data(r)
    p = cd.path
    sub = "n = $(p.n[end]): [$(_mk_fmt(p.lower[end])), $(_mk_fmt(p.upper[end]))]"
    fig = _mk_figure(figure)
    ax = _mk_axis(fig[1, 1], axis; xlabel="Sample size", ylabel=cd.label,
                  title=cd.title, subtitle=sub)
    _mk_draw_confseq!(ax, cd; fixed=fixed, null=null, colors=colors)
    _mk_legend!(fig, ax)
    return fig
end

function _mk_draw_gs!(ax, gd; scale::Symbol, colors)
    scale in (:z, :p) || throw(ArgumentError("scale must be :z or :p"))
    cols = _mk_colors(colors, 3)
    tr(z) = scale === :z ? z : DrSnow._viz_nominal_p.(z)
    b = gd.bounds
    scatterlines!(ax, b.timing, tr(b.efficacy); color=cols[1], marker=:utriangle,
                  markersize=11, linewidth=2, label="Efficacy bound")
    if any(!isnan, b.lower)
        ok = .!isnan.(b.lower)
        scatterlines!(ax, b.timing[ok], tr(b.lower[ok]); color=cols[2],
                      marker=:dtriangle, markersize=11, linewidth=2,
                      label=gd.sided == 2 ? "Lower bound" : "Futility bound")
    end
    hlines!(ax, tr([gd.fixed])[1]; color=_MK_REF, linestyle=:dot, linewidth=1.2,
            label="Fixed-sample critical value")
    if gd.observed !== nothing
        o = gd.observed
        scatterlines!(ax, o.timing, tr(o.z); color=cols[3], marker=:circle,
                      markersize=10, linewidth=1.5, linestyle=:dash,
                      label="Observed statistic")
    end
    return ax
end

function plot_gs_boundaries!(ax::Makie.AbstractAxis, d; analysis=nothing,
                             scale::Symbol=:z, colors=nothing)
    return _mk_draw_gs!(ax, DrSnow._viz_gs_data(d; analysis=analysis); scale=scale,
                        colors=colors)
end

function plot_gs_boundaries(d; analysis=nothing, scale::Symbol=:z, colors=nothing,
                            figure::NamedTuple=(;), axis::NamedTuple=(;))
    gd = DrSnow._viz_gs_data(d; analysis=analysis)
    fig = _mk_figure(figure)
    defaults = scale === :p ? (yscale=log10, ylabel="Nominal one-sided p-value") :
               (ylabel="Z statistic",)
    ax = _mk_axis(fig[1, 1], axis; xlabel="Information fraction", title=gd.title,
                  defaults...)
    _mk_draw_gs!(ax, gd; scale=scale, colors=colors)
    _mk_legend!(fig, ax)
    return fig
end
