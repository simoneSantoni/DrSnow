# Synthetic control in-time placebo (backdating) plot.

function _mk_draw_synth_in_time!(ax, d; colors=nothing)
    t = d.table
    cols = _mk_colors(colors, 3)
    vlines!(ax, d.x_placebo; color=cols[3], linestyle=:dash, linewidth=1.2)
    vlines!(ax, d.x_treatment; color=_MK_REF, linestyle=:dash, linewidth=1.2)
    lines!(ax, t.x, t.treated; color=_MK_INK, linewidth=2.2, label="Treated unit")
    lines!(ax, t.x, t.synthetic; color=cols[1], linewidth=2, linestyle=:dash,
           label="Synthetic control (actual date)")
    lines!(ax, t.x, t.backdated; color=cols[3], linewidth=2, linestyle=:dot,
           label="Synthetic control (backdated to $(d.placebo_time))")
    d.ticks === nothing || (ax.xticks = d.ticks)
    return ax
end

function DrSnow.plot_synth_in_time!(ax::Makie.AbstractAxis,
                                    r::DrSnow.SyntheticControlEstimate,
                                    b::DrSnow.SyntheticControlEstimate; colors=nothing)
    d = DrSnow._viz_synth_in_time_data(r, b)
    return _mk_draw_synth_in_time!(ax, d; colors=colors)
end

function DrSnow.plot_synth_in_time(r::DrSnow.SyntheticControlEstimate,
                                   b::DrSnow.SyntheticControlEstimate; colors=nothing,
                                   figure::NamedTuple=(;), axis::NamedTuple=(;))
    d = DrSnow._viz_synth_in_time_data(r, b)
    parts = ["Dashed verticals: placebo date $(d.placebo_time) and actual treatment " *
             "$(d.treatment_time)",
             "backdated weights fitted on periods before $(d.placebo_time) only"]
    fig = _mk_figure(figure)
    ax = _mk_axis(fig[1, 1], axis; xlabel="Time", ylabel="Outcome",
                  subtitle=_mk_caption(parts))
    _mk_draw_synth_in_time!(ax, d; colors=colors)
    _mk_legend!(fig, ax; nbanks=2)
    return fig
end
