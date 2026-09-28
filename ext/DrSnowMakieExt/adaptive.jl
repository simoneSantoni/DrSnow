# Adaptive experiments: assignment probabilities over time, excursion-effect curves.

function DrSnow.plot_assignment_probabilities!(ax::Makie.AbstractAxis,
                                               l::DrSnow.AdaptiveLog; floor::Bool=true,
                                               colors=nothing)
    d = DrSnow._viz_assignment_data(l)
    t = d.table
    cols = _mk_colors(colors, d.K)
    for k in 1:d.K
        s = t[t.arm .== k, :]
        if nrow(s) > 1
            stairs!(ax, s.t, s.probability; step=:post, color=cols[k], linewidth=1.8,
                    linestyle=(:solid, :dash, :dot, :dashdot)[mod1(k, 4)],
                    label="arm $k")
        else
            scatter!(ax, s.t, s.probability; color=cols[k], marker=_mk_marker(k),
                     label="arm $k")
        end
    end
    if floor && nrow(d.floor) > 0
        lines!(ax, d.floor.t, d.floor.floor; color=_MK_REF, linestyle=:dash,
               linewidth=1.2, label="floor")
    end
    d.burnin > 0 && vlines!(ax, d.burnin + 0.5; color=_MK_INK2, linestyle=:dot,
                            linewidth=1.2, label="end of burn-in")
    ylims!(ax, -0.02, 1.02)
    return ax
end

function DrSnow.plot_assignment_probabilities(l::DrSnow.AdaptiveLog; floor::Bool=true,
                                              colors=nothing, figure::NamedTuple=(;),
                                              axis::NamedTuple=(;))
    d = DrSnow._viz_assignment_data(l)
    fig = _mk_figure(figure)
    sub = _mk_caption([d.policy, "$(DrSnow.nobs(l)) units in " *
                                 "$(length(l.batch_start)) batch(es)",
                       d.contextual ? "batch means over units' covariates" : ""])
    ax = _mk_axis(fig[1, 1], axis; xlabel="Unit (order of assignment)",
                  ylabel="Assignment probability", subtitle=sub)
    DrSnow.plot_assignment_probabilities!(ax, l; floor=floor, colors=colors)
    _mk_legend!(fig, ax)
    return fig
end

function DrSnow.plot_excursion_effect!(ax::Makie.AbstractAxis,
                                       r::DrSnow.ExcursionEffectEstimate;
                                       moderator=nothing, values=nothing, at=nothing,
                                       level::Real=0.95, relative_risk::Bool=false,
                                       colors=nothing)
    d = DrSnow._viz_excursion_data(r; moderator=moderator, values=values, at=at,
                                   level=level, relative_risk=relative_risk)
    t = d.table
    cols = _mk_colors(colors, 1)
    hlines!(ax, d.scale === :rr ? 1.0 : 0.0; color=_MK_REF, linestyle=:dash,
            linewidth=1)
    ci = "$(_mk_pct(level)) CI"
    if d.moderator === nothing || nrow(t) == 1
        rangebars!(ax, t.x, t.conf_low, t.conf_high; color=cols[1], linewidth=2.5,
                   whiskerwidth=12, label=ci)
        scatter!(ax, t.x, t.estimate; color=cols[1], markersize=12,
                 label="Excursion effect")
    else
        band!(ax, t.x, t.conf_low, t.conf_high; color=(cols[1], 0.2),
              label="Pointwise $ci")
        lines!(ax, t.x, t.estimate; color=cols[1], linewidth=2.5,
               label="Excursion effect")
    end
    return ax
end

function DrSnow.plot_excursion_effect(r::DrSnow.ExcursionEffectEstimate;
                                      moderator=nothing, values=nothing, at=nothing,
                                      level::Real=0.95, relative_risk::Bool=false,
                                      colors=nothing, figure::NamedTuple=(;),
                                      axis::NamedTuple=(;))
    d = DrSnow._viz_excursion_data(r; moderator=moderator, values=values, at=at,
                                   level=level, relative_risk=relative_risk)
    fig = _mk_figure(figure)
    ylab = d.scale === :difference ? "Effect on the proximal outcome" :
           (d.scale === :rr ? "Relative risk" : "Log relative risk")
    held = isempty(d.held) ? "" :
           "other moderators at " * join(["$k = $(_mk_fmt(v))" for (k, v) in
                                          pairs(d.held)], ", ")
    sub = _mk_caption([d.method, "$(r.n_ids) participants, " *
                                 "t($(Int(DrSnow.dof_residual(r)))) reference", held])
    xl = d.moderator === nothing ? "" : string(d.moderator)
    ax = _mk_axis(fig[1, 1], axis; xlabel=xl, ylabel=ylab, subtitle=sub)
    d.moderator === nothing && hidexdecorations!(ax)
    DrSnow.plot_excursion_effect!(ax, r; moderator=moderator, values=values, at=at,
                                  level=level, relative_risk=relative_risk,
                                  colors=colors)
    _mk_legend!(fig, ax)
    return fig
end
