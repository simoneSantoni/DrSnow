# Goodman-Bacon decomposition plot.

const _MK_BACON_TYPES = [
    :treated_vs_never => "Treated vs never treated",
    :earlier_vs_later => "Earlier vs later treated",
    :later_vs_earlier => "Later vs earlier treated (already-treated controls)",
    :later_vs_always => "Later vs always treated (already-treated controls)",
]

function plot_bacon!(ax::Makie.AbstractAxis, b::DrSnow.BaconDecomposition; colors=nothing)
    c = b.comparisons
    present = [p for p in _MK_BACON_TYPES if any(==(p.first), c.type)]
    other = setdiff(unique(c.type), first.(present))
    append!(present, [t => string(t) for t in other])
    cols = _mk_colors(colors, length(_MK_BACON_TYPES) + length(other))
    hlines!(ax, b.twfe_estimate; color=_MK_INK2, linestyle=:dash, linewidth=1.2,
            label="TWFE estimate = $(_mk_fmt(b.twfe_estimate; digits=4))")
    for (i, (t, lbl)) in enumerate(present)
        s = c[c.type .== t, :]
        k = something(findfirst(p -> p.first == t, _MK_BACON_TYPES), i)
        scatter!(ax, s.weight, s.estimate; color=cols[k], marker=_mk_marker(k),
                 markersize=12, strokecolor=:white, strokewidth=1, label=lbl)
    end
    return ax
end

function plot_bacon(b::DrSnow.BaconDecomposition; colors=nothing, figure::NamedTuple=(;),
                    axis::NamedTuple=(;))
    fig = _mk_figure(figure; size=(760, 500))
    ax = _mk_axis(fig[1, 1], axis; xlabel="Weight in the TWFE coefficient",
                  ylabel="2×2 DiD estimate",
                  subtitle="$(nrow(b.comparisons)) comparisons; TWFE = Σ weight × " *
                           "estimate", xgridvisible=true, xgridcolor=(:black, 0.07))
    plot_bacon!(ax, b; colors=colors)
    _mk_legend!(fig, ax; nbanks=2)
    return fig
end
