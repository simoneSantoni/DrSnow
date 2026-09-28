# Randomization (reference) distributions with the observed statistic.

function _mk_draw_randomization!(ax, rd; bins, colors)
    cols = _mk_colors(colors, 2)
    v, w = rd.values, rd.weights
    nb = bins === nothing ? clamp(round(Int, sqrt(length(v))), 10, 60) : bins
    lo, hi = extrema(vcat(v, rd.observed))
    if rd.alternative === :two_sided
        lo, hi = min(lo, -abs(rd.observed)), max(hi, abs(rd.observed))
    end
    edges = hi > lo ? collect(range(lo, hi; length=nb + 1)) : [lo - 0.5, lo + 0.5]
    hist!(ax, v; bins=edges, weights=w, normalization=:probability,
          color=(cols[1], 0.55), strokecolor=:white, strokewidth=0.5,
          label="Randomization distribution")
    vlines!(ax, rd.observed; color=cols[2], linewidth=2.5,
            label="Observed = $(_mk_fmt(rd.observed; digits=4))")
    if rd.alternative === :two_sided && rd.observed != 0
        vlines!(ax, -rd.observed; color=cols[2], linewidth=1.5, linestyle=:dash,
                label="−Observed (two-sided)")
    end
    return ax
end

function plot_randomization_distribution!(ax::Makie.AbstractAxis, r;
                                          hypothesis::Integer=1, bins=nothing,
                                          colors=nothing)
    rd = DrSnow._viz_randomization_data(r; hypothesis=hypothesis)
    return _mk_draw_randomization!(ax, rd; bins=bins, colors=colors)
end

function plot_randomization_distribution(r; hypothesis::Integer=1, bins=nothing,
                                         colors=nothing, figure::NamedTuple=(;),
                                         axis::NamedTuple=(;))
    rd = DrSnow._viz_randomization_data(r; hypothesis=hypothesis)
    alt = replace(string(rd.alternative), "_" => "-")
    sub = "Randomization p = $(_mk_fmt(rd.pvalue; digits=3)) ($alt); " *
          "$(length(rd.values)) reference values"
    fig = _mk_figure(figure)
    ax = _mk_axis(fig[1, 1], axis; xlabel=rd.name, ylabel="Share of $(rd.draws)",
                  subtitle=sub)
    _mk_draw_randomization!(ax, rd; bins=bins, colors=colors)
    _mk_legend!(fig, ax)
    return fig
end
