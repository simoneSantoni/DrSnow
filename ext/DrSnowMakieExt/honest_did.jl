# Rambachan–Roth (Honest DiD) sensitivity plot.

function _mk_honest_xpos(t)
    Ms = t.M
    step = length(Ms) > 1 ? minimum(diff(Ms)) : max(abs(Ms[1]), 1.0)
    step > 0 || (step = 1.0)
    return Ms[1] - 1.5 * step, step
end

function _mk_draw_honest!(ax, d; colors=nothing)
    t = d.table
    cols = _mk_colors(colors, 2)
    x0, step = _mk_honest_xpos(t)
    hlines!(ax, 0.0; color=_MK_REF, linestyle=:dash, linewidth=1)
    lo, hi = d.original
    olbl = "Original CI (exact parallel trends)"
    rangebars!(ax, [x0], [lo], [hi]; color=cols[2], linewidth=2.5, whiskerwidth=10,
               label=olbl)
    scatter!(ax, [x0], [d.estimate]; color=cols[2], marker=:diamond, markersize=11,
             strokecolor=:white, strokewidth=1, label=olbl)
    ok = .!t.rejected
    any(ok) && rangebars!(ax, t.M[ok], t.lb[ok], t.ub[ok]; color=cols[1],
                          linewidth=2.5, whiskerwidth=10,
                          label="Robust CI, Δ($(d.mname))")
    if any(t.rejected)
        scatter!(ax, t.M[t.rejected], fill(0.0, count(t.rejected)); color=_MK_INK2,
                 marker=:xcross, markersize=10,
                 label="Δ($(d.mname)) rejected (empty set)")
    end
    if isfinite(d.breakdown)
        vlines!(ax, d.breakdown; color=_MK_INK2, linestyle=:dot, linewidth=1.5,
                label="Breakdown $(d.mname) = $(_mk_fmt(d.breakdown))")
    end
    ticks = vcat(x0, t.M)
    ax.xticks = (ticks, vcat("Orig.", [_mk_fmt(m) for m in t.M]))
    xmax = max(t.M[end], isfinite(d.breakdown) ? d.breakdown : -Inf)
    xlims!(ax, x0 - 0.7 * step, xmax + 0.7 * step)
    return ax
end

function DrSnow.plot_honest_did!(ax::Makie.AbstractAxis, r::DrSnow.HonestDiDResult;
                                 breakdown=nothing, colors=nothing)
    d = DrSnow._viz_honest_data(r; breakdown=breakdown)
    return _mk_draw_honest!(ax, d; colors=colors)
end

function DrSnow.plot_honest_did(r::DrSnow.HonestDiDResult; breakdown=nothing,
                                colors=nothing, figure::NamedTuple=(;),
                                axis::NamedTuple=(;))
    d = DrSnow._viz_honest_data(r; breakdown=breakdown)
    rname = d.restriction === :smoothness ? "smoothness" :
            d.restriction === :relative_magnitudes ? "relative magnitudes" :
            "relative magnitudes of deviations from a linear trend"
    bd = isfinite(d.breakdown) ?
         (d.breakdown_source === :grid ? "breakdown: smallest grid value whose set " *
                                          "contains 0" :
          "breakdown value as given") :
         "no breakdown within the grid"
    parts = ["$(_mk_pct(d.level)) confidence sets; restriction $(d.delta) ($rname)",
             "method $(d.method)", bd]
    fig = _mk_figure(figure)
    ax = _mk_axis(fig[1, 1], axis; xlabel=d.mname, ylabel="Target effect θ",
                  subtitle=_mk_caption(parts))
    _mk_draw_honest!(ax, d; colors=colors)
    _mk_legend!(fig, ax; nbanks=2)
    return fig
end
