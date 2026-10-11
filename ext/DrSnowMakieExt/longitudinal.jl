# Longitudinal g-methods: cumulative inverse probability weights by regime and period.

function DrSnow.plot_regime_weights!(ax::Makie.AbstractAxis,
                                     r::DrSnow.LongitudinalEstimate; regimes=nothing,
                                     colors=nothing)
    d = DrSnow._viz_regime_weights_data(r; regimes=regimes)
    t = d.table
    R = length(d.labels)
    cols = _mk_colors(colors, R)
    off = _mk_dodge(R; width=0.6)
    w = min(0.25, 0.6 / R)
    for k in 1:R
        s = t[t.regime .== k, :]
        nrow(s) == 0 && continue
        boxplot!(ax, s.period .+ off[k], s.weight; width=w, color=(cols[k], 0.6),
                 strokecolor=cols[k], strokewidth=1, whiskercolor=cols[k],
                 mediancolor=_MK_INK, outliercolor=cols[k], markersize=5,
                 label=d.labels[k])
    end
    isfinite(d.bound) && hlines!(ax, d.bound; color=_MK_REF, linestyle=:dash, linewidth=1,
                                 label="bound 1/g_bound")
    c = d.counts
    ymax = isempty(t.weight) ? 1.0 : maximum(t.weight)
    for row in eachrow(c)
        text!(ax, row.period + off[row.regime], ymax; text=string(row.n_following),
              align=(:center, :bottom), fontsize=10, color=_MK_INK2, offset=(0, 4))
    end
    ax.xticks = (1:r.K, string.(d.periods))
    return ax
end

function DrSnow.plot_regime_weights(r::DrSnow.LongitudinalEstimate; regimes=nothing,
                                    colors=nothing, figure::NamedTuple=(;),
                                    axis::NamedTuple=(;))
    d = DrSnow._viz_regime_weights_data(r; regimes=regimes)
    fig = _mk_figure(figure)
    sub = _mk_caption([d.method, "$(d.n) units",
                       "numbers above the boxes: followers through the period",
                       "cumulative probability bounded below at $(_mk_fmt(1 / d.bound))"])
    ax = _mk_axis(fig[1, 1], axis; xlabel="Period",
                  ylabel="Cumulative inverse probability weight", subtitle=sub)
    DrSnow.plot_regime_weights!(ax, r; regimes=regimes, colors=colors)
    _mk_legend!(fig, ax)
    return fig
end
