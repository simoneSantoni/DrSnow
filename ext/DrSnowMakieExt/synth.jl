# Synthetic control, synthetic DiD, augmented SC and matrix completion plots.

const _MK_SYNTH = Union{DrSnow.SyntheticControlEstimate,DrSnow.SyntheticDiDEstimate,
                        DrSnow.AugmentedSCEstimate,DrSnow.MatrixCompletionEstimate}

_mk_synth_label(r::DrSnow.SyntheticDiDEstimate) =
    r.method === :sdid ? "Synthetic DiD (incl. level adjustment)" :
    r.method === :sc ? "Synthetic control" : "DiD counterfactual"
_mk_synth_label(::DrSnow.MatrixCompletionEstimate) = "Matrix completion counterfactual"
_mk_synth_label(::DrSnow.AugmentedSCEstimate) = "Augmented synthetic control"
_mk_synth_label(::DrSnow.SyntheticControlEstimate) = "Synthetic control"

_mk_time(x) = x isa Real ? Float64(x) : x

function _mk_onset_lines!(ax, sd, sel)
    for (c, o) in zip(sd.cohorts, sd.onset)
        (o === nothing || (sel !== :all && !isequal(c, sel))) && continue
        vlines!(ax, _mk_time(o); color=_MK_REF, linestyle=:dash, linewidth=1)
    end
end

function _mk_synth_trajectories!(ax, r, sd; colors, time_weights::Bool)
    cols = _mk_colors(colors, 2)
    g = sd.gaps
    g = ismissing(sd.cohort) ? g : g[isequal.(g.cohort, sd.cohort), :]
    t = _mk_time.(g.time)
    _mk_onset_lines!(ax, sd, sd.cohort)
    if time_weights && sd.time_weights !== nothing && nrow(sd.time_weights) > 0
        w = sd.time_weights
        lo, hi = extrema(vcat(g.treated, g.synthetic))
        span = hi - lo > 0 ? hi - lo : 1.0
        base = lo - 0.2 * span
        h = w.weight ./ max(maximum(w.weight), eps())
        barplot!(ax, _mk_time.(w.time), base .+ 0.15 * span .* h; fillto=base,
                 color=(cols[2], 0.35), strokewidth=0, gap=0.25,
                 label="Pre-period weights λ (scaled)")
    end
    lines!(ax, t, g.treated; color=cols[1], linewidth=2.2, label="Treated")
    lines!(ax, t, g.synthetic; color=cols[2], linewidth=2.2, linestyle=:dash,
           label=_mk_synth_label(r))
    return ax
end

function _mk_synth_gaps!(ax, r, sd; colors, placebos::Bool)
    hlines!(ax, 0.0; color=_MK_REF, linewidth=1)
    _mk_onset_lines!(ax, sd, :all)
    if placebos && sd.placebo_gaps !== nothing && nrow(sd.placebo_gaps) > 0
        p = sd.placebo_gaps
        xs = Float64[]
        ys = Float64[]
        for u in unique(p.unit)
            s = p[p.unit .== u, :]
            append!(xs, Float64.(_mk_time.(s.time)))
            append!(ys, s.gap)
            push!(xs, NaN)
            push!(ys, NaN)
        end
        lines!(ax, xs, ys; color=(_MK_MUTED, 0.8), linewidth=1,
               label="Placebo gaps (donors)")
    end
    g = sd.gaps
    cohorts = sd.cohorts
    cols = _mk_colors(colors, length(cohorts))
    for (i, c) in enumerate(cohorts)
        s = ismissing(c) ? g : g[isequal.(g.cohort, c), :]
        lbl = ismissing(c) ? "Treated − synthetic" : "Cohort $(c)"
        lines!(ax, _mk_time.(s.time), s.gap; color=cols[i], linewidth=2.2, label=lbl)
    end
    return ax
end

function plot_synth!(ax::Makie.AbstractAxis, r::_MK_SYNTH; kind::Symbol=:trajectories,
                     cohort=nothing, placebos::Bool=true, placebo_cutoff::Real=Inf,
                     time_weights::Bool=true, colors=nothing)
    sd = DrSnow._viz_synth_data(r; cohort=cohort, placebo_cutoff=placebo_cutoff)
    if kind === :trajectories
        return _mk_synth_trajectories!(ax, r, sd; colors=colors, time_weights=time_weights)
    elseif kind === :gaps
        return _mk_synth_gaps!(ax, r, sd; colors=colors, placebos=placebos)
    end
    throw(ArgumentError("plot_synth!: kind must be :trajectories or :gaps (use " *
                        "plot_synth for :both)"))
end

function _mk_synth_subtitle(r, sd)
    s = "ATT $(_mk_fmt(DrSnow.estimate(r)))"
    se = DrSnow._tidy_vcov(r)
    se === nothing || (s *= " (se $(_mk_fmt(sqrt(se[1, 1]))))")
    ismissing(sd.cohort) || length(sd.cohorts) == 1 ||
        (s *= "; trajectories: cohort $(sd.cohort)")
    return s * "; dashed line: first treated period"
end

function plot_synth(r::_MK_SYNTH; kind::Symbol=:both, cohort=nothing,
                    placebos::Bool=true, placebo_cutoff::Real=Inf,
                    time_weights::Bool=true, colors=nothing, figure::NamedTuple=(;),
                    axis::NamedTuple=(;))
    kind in (:both, :trajectories, :gaps) ||
        throw(ArgumentError("plot_synth: kind must be :both, :trajectories or :gaps"))
    sd = DrSnow._viz_synth_data(r; cohort=cohort, placebo_cutoff=placebo_cutoff)
    ylab = string(r.panel.outcome)
    sub = _mk_synth_subtitle(r, sd)
    if kind === :both
        fig = _mk_figure(figure; size=(760, 680))
        ax1 = _mk_axis(fig[1, 1], axis; ylabel=ylab, subtitle=sub)
        ax2 = _mk_axis(fig[2, 1], axis; xlabel="Time", ylabel="Gap (treated − synthetic)",
                       title="", subtitle="")
        _mk_synth_trajectories!(ax1, r, sd; colors=colors, time_weights=time_weights)
        gap_colors = colors === nothing && length(sd.cohorts) == 1 ?
                     [_MK_PALETTE[1]] : colors
        _mk_synth_gaps!(ax2, r, sd; colors=gap_colors, placebos=placebos)
        linkxaxes!(ax1, ax2)
        hidexdecorations!(ax1; grid=false, ticks=false)
        _mk_legend!(fig, [ax1, ax2]; row=3, nbanks=2)
        rowsize!(fig.layout, 1, Auto(1.4))
        return fig
    end
    fig = _mk_figure(figure)
    ax = _mk_axis(fig[1, 1], axis; xlabel="Time",
                  ylabel=kind === :gaps ? "Gap (treated − synthetic)" : ylab, subtitle=sub)
    if kind === :gaps
        _mk_synth_gaps!(ax, r, sd; colors=colors, placebos=placebos)
    else
        _mk_synth_trajectories!(ax, r, sd; colors=colors, time_weights=time_weights)
    end
    _mk_legend!(fig, ax)
    return fig
end
