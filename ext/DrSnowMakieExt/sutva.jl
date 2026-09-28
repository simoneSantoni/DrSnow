# Direct and spillover effects by exposure group (rings, hops, saturations).

const _MK_EXPOSURE = Union{DrSnow.SpilloverRegression,DrSnow.ExposureEffects,
                           DrSnow.TwoStageEffects}

function plot_spillover_rings!(ax::Makie.AbstractAxis, es::DrSnow.SpilloverEventStudy;
                               level::Real=0.95, colors=nothing)
    return plot_event_study!(ax, es; level=level, colors=colors)
end

function plot_spillover_rings!(ax::Makie.AbstractAxis, r::_MK_EXPOSURE; level::Real=0.95,
                               colors=nothing)
    cols = _mk_colors(colors, 2)
    t = DrSnow._viz_exposure_effect_data(r; level=level)
    x = 1:nrow(t)
    hlines!(ax, 0.0; color=_MK_REF, linewidth=1)
    for (k, kind) in enumerate(("direct", "spillover"))
        s = findall(==(kind), t.kind)
        isempty(s) && continue
        ok = s[_mk_has_ci(t.conf_low[s], t.conf_high[s])]
        isempty(ok) || rangebars!(ax, x[ok], Float64.(t.conf_low[ok]),
                                  Float64.(t.conf_high[ok]); color=cols[k],
                                  linewidth=1.5, whiskerwidth=8)
        scatter!(ax, x[s], t.estimate[s]; color=cols[k], marker=_mk_marker(k),
                 markersize=11, strokecolor=:white, strokewidth=1,
                 label=kind == "direct" ? "Direct / other effects" : "Spillover effects")
    end
    ax.xticks = (collect(x), t.term)
    return ax
end

function plot_spillover_rings(r::Union{DrSnow.SpilloverEventStudy,_MK_EXPOSURE};
                              level::Real=0.95, colors=nothing, figure::NamedTuple=(;),
                              axis::NamedTuple=(;))
    r isa DrSnow.SpilloverEventStudy &&
        return plot_event_study(r; level=level, colors=colors, figure=figure,
                                axis=merge((title="Direct and spillover effects by " *
                                                  "event time",), axis))
    fig = _mk_figure(figure)
    ax = _mk_axis(fig[1, 1], axis; ylabel="Estimated effect",
                  subtitle="$(DrSnow.method_name(r)); whiskers: $(_mk_pct(level)) CIs",
                  xticklabelrotation=length(StatsAPI.coefnames(r)) > 4 ? π / 6 : 0.0)
    plot_spillover_rings!(ax, r; level=level, colors=colors)
    _mk_legend!(fig, ax)
    return fig
end
