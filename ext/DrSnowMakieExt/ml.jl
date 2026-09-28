# Heterogeneous effects: GATES and DR-learner CATE predictions.

function plot_gates!(ax::Makie.AbstractAxis, g::DrSnow.GenericMLInference; colors=nothing)
    cols = _mk_colors(colors, 2)
    d = DrSnow._viz_gates_data(g)
    t = d.gates
    K = nrow(t)
    lv = _mk_pct(d.level)
    hspan!(ax, d.ate.lower, d.ate.upper; color=(cols[2], 0.12))
    hlines!(ax, d.ate.estimate; color=cols[2], linewidth=2, linestyle=:dash,
            label="ATE (BLP β₁) with $lv CI")
    hlines!(ax, 0.0; color=_MK_REF, linewidth=1)
    barplot!(ax, 1:K, t.estimate; color=(cols[1], 0.8), strokewidth=0, gap=0.35,
             label="GATES with $lv CI")
    rangebars!(ax, 1:K, t.lower, t.upper; color=_MK_INK, linewidth=1.5, whiskerwidth=8)
    labels = copy(t.group)
    labels[1] *= "\nleast affected"
    K > 1 && (labels[end] *= "\nmost affected")
    ax.xticks = (1:K, labels)
    return ax
end

function plot_gates(g::DrSnow.GenericMLInference; colors=nothing, figure::NamedTuple=(;),
                    axis::NamedTuple=(;))
    diff = DrSnow.gates(g)
    row = findfirst(s -> occursin(" - ", s), diff.group)
    sub = "Groups by predicted effect ($(g.learner)); median over $(g.n_splits) splits"
    row === nothing ||
        (sub *= "; $(diff.group[row]) = $(_mk_fmt(diff.estimate[row])), " *
                "p = $(_mk_fmt(diff.pvalue[row]))")
    fig = _mk_figure(figure)
    ax = _mk_axis(fig[1, 1], axis; xlabel="Group", ylabel="Average treatment effect",
                  subtitle=sub)
    plot_gates!(ax, g; colors=colors)
    _mk_legend!(fig, ax)
    return fig
end

function plot_cate!(ax::Makie.AbstractAxis, c::DrSnow.CATEPredictor;
                    modifier::Union{Nothing,Symbol}=nothing, level::Real=0.95,
                    bins::Integer=30, colors=nothing)
    cols = _mk_colors(colors, 2)
    crit = critical_value(level, c.n_clusters > 0 ? c.n_clusters - 1 : Inf)
    lo, hi = c.ate - crit * c.ate_se, c.ate + crit * c.ate_se
    lbl = "ATE (AIPW) with $(_mk_pct(level)) CI"
    if modifier === nothing
        vspan!(ax, lo, hi; color=(cols[2], 0.15))
        hist!(ax, c.cate_oof; bins=bins, color=(cols[1], 0.6), strokecolor=:white,
              strokewidth=0.5, label="Cross-fitted CATE predictions")
        vlines!(ax, c.ate; color=cols[2], linewidth=2.5, label=lbl)
    else
        j = findfirst(==(modifier), c.effect_modifiers)
        j === nothing && throw(ArgumentError("plot_cate: $modifier is not an effect " *
                                             "modifier; available: " *
                                             join(c.effect_modifiers, ", ")))
        hspan!(ax, lo, hi; color=(cols[2], 0.15))
        scatter!(ax, c.V[:, j], c.cate_oof; color=(cols[1], 0.45), markersize=5,
                 label="Cross-fitted CATE predictions")
        hlines!(ax, c.ate; color=cols[2], linewidth=2.5, label=lbl)
    end
    return ax
end

function plot_cate(c::DrSnow.CATEPredictor; modifier::Union{Nothing,Symbol}=nothing,
                   level::Real=0.95, bins::Integer=30, colors=nothing,
                   figure::NamedTuple=(;), axis::NamedTuple=(;))
    fig = _mk_figure(figure)
    sub = "Out-of-fold DR-learner predictions (noisy individually; use GATES or " *
          "cate_projection for inference)"
    ax = modifier === nothing ?
         _mk_axis(fig[1, 1], axis; xlabel="Predicted CATE", ylabel="Count", subtitle=sub) :
         _mk_axis(fig[1, 1], axis; xlabel=string(modifier), ylabel="Predicted CATE",
                  subtitle=sub)
    plot_cate!(ax, c; modifier=modifier, level=level, bins=bins, colors=colors)
    _mk_legend!(fig, ax)
    return fig
end
