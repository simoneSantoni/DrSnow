# Weak-instrument-robust confidence sets: p-value function of the inverted test.

function _mk_set_string(intervals)
    isempty(intervals) && return "∅"
    piece((a, b)) = (isfinite(a) ? "[" : "(") * _mk_fmt(a) * ", " * _mk_fmt(b) *
                    (isfinite(b) ? "]" : ")")
    return join(piece.(intervals), " ∪ ")
end

function _mk_cs_limits(s, wald, level)
    wald === nothing && return nothing
    lo, hi = DrSnow._viz_cs_limits(s)
    ci = StatsAPI.confint(wald; level=level)[1, :]
    span = ci[2] - ci[1]
    return (min(lo, ci[1] - 0.5 * span), max(hi, ci[2] + 0.5 * span))
end

function plot_confidence_set!(ax::Makie.AbstractAxis, s::DrSnow.WeakIVConfidenceSet;
                              limits=nothing, npoints::Integer=801,
                              wald::Union{Nothing,DrSnow.IVEstimate}=nothing,
                              colors=nothing)
    cols = _mk_colors(colors, 2)
    lims = limits === nothing ? _mk_cs_limits(s, wald, s.level) : limits
    d = DrSnow._viz_confidence_set_data(s; limits=lims, npoints=npoints)
    lo, hi = d.limits
    lv = _mk_pct(d.level)
    for (a, b) in d.intervals
        a2, b2 = max(a, lo), min(b, hi)
        a2 < b2 && vspan!(ax, a2, b2; color=(cols[1], 0.15),
                          label="$lv robust confidence set")
    end
    hlines!(ax, 1 - d.level; color=_MK_REF, linestyle=:dash, linewidth=1)
    lines!(ax, d.grid, d.pvalue; color=cols[1], linewidth=2.2,
           label="p-value of H₀: β = b")
    isfinite(d.estimate) && vlines!(ax, d.estimate; color=_MK_INK2, linewidth=1.2,
                                    label="Point estimate")
    if wald !== nothing
        ci = StatsAPI.confint(wald; level=d.level)[1, :]
        linesegments!(ax, [Point2f(max(ci[1], lo), -0.04), Point2f(min(ci[2], hi), -0.04)];
                      color=cols[2], linewidth=4, label="Wald $lv CI (not weak-IV robust)")
    end
    xlims!(ax, lo, hi)
    ylims!(ax, wald === nothing ? -0.02 : -0.08, 1.02)
    return ax
end

function plot_confidence_set(s::DrSnow.WeakIVConfidenceSet; limits=nothing,
                             npoints::Integer=801,
                             wald::Union{Nothing,DrSnow.IVEstimate}=nothing,
                             colors=nothing, figure::NamedTuple=(;), axis::NamedTuple=(;))
    sub = _mk_caption(["Test: $(s.method)",
                       "$(_mk_pct(s.level)) set: $(_mk_set_string(s.intervals))",
                       "dashed: significance level $(_mk_fmt(1 - s.level))"])
    fig = _mk_figure(figure)
    ax = _mk_axis(fig[1, 1], axis; xlabel="Hypothesized effect b", ylabel="p-value",
                  subtitle=sub)
    plot_confidence_set!(ax, s; limits=limits, npoints=npoints, wald=wald, colors=colors)
    _mk_legend!(fig, ax; nbanks=2)
    return fig
end
