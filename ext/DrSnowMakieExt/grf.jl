# Generalized random forests: variable importance, CATE by covariate, TOC curves.

function plot_variable_importance!(ax::Makie.AbstractAxis,
                                   f::DrSnow.GeneralizedRandomForest;
                                   top=nothing, decay_exponent::Real=2,
                                   max_depth::Integer=4, colors=nothing)
    cols = _mk_colors(colors, 1)
    vi = DrSnow._viz_variable_importance_data(f; decay_exponent=decay_exponent,
                                              max_depth=max_depth, top=top)
    k = nrow(vi)
    pos = collect(k:-1:1)                       # most important on top
    barplot!(ax, pos, vi.importance; direction=:x, color=(cols[1], 0.85),
             strokewidth=0, gap=0.3)
    ax.yticks = (pos, vi.variable)
    return ax
end

function plot_variable_importance(f::DrSnow.GeneralizedRandomForest; top=nothing,
                                  decay_exponent::Real=2, max_depth::Integer=4,
                                  colors=nothing, figure::NamedTuple=(;),
                                  axis::NamedTuple=(;))
    k = top === nothing ? length(f.covariates) : min(Int(top), length(f.covariates))
    fig = _mk_figure(figure; size=(640, max(260, 60 + 26 * k)))
    sub = "Share of splits by covariate, depths 1–$(max_depth) weighted by " *
          "depth^-$(decay_exponent) ($(length(f.forest.trees)) trees)"
    ax = _mk_axis(fig[1, 1], axis; xlabel="Importance", ylabel="",
                  title=DrSnow._ml_grf_label(f), subtitle=sub, xgridvisible=true,
                  ygridvisible=false)
    plot_variable_importance!(ax, f; top=top, decay_exponent=decay_exponent,
                              max_depth=max_depth, colors=colors)
    xlims!(ax, 0, nothing)
    return fig
end

function plot_cate!(ax::Makie.AbstractAxis,
                    f::Union{DrSnow.CausalForest,DrSnow.InstrumentalForest};
                    modifier::Union{Nothing,Symbol}=nothing, level::Real=0.95,
                    bins::Integer=30, colors=nothing)
    cols = _mk_colors(colors, 2)
    d = DrSnow._viz_forest_cate_data(f; modifier=modifier, level=level)
    t = d.cate
    a = d.ate
    lbl = "Average effect (AIPW) with $(_mk_pct(level)) CI"
    if modifier === nothing
        vspan!(ax, a.conf_low, a.conf_high; color=(cols[2], 0.15))
        hist!(ax, t.estimate; bins=bins, color=(cols[1], 0.6), strokecolor=:white,
              strokewidth=0.5, label="Out-of-bag $(d.label) estimates")
        vlines!(ax, a.estimate; color=cols[2], linewidth=2.5, label=lbl)
    else
        hspan!(ax, a.conf_low, a.conf_high; color=(cols[2], 0.15))
        x = Float64.(t.x)
        rangebars!(ax, x, t.conf_low, t.conf_high; color=(cols[1], 0.25), linewidth=1,
                   label="Pointwise $(_mk_pct(level)) CI")
        scatter!(ax, x, t.estimate; color=cols[1], markersize=4,
                 label="Out-of-bag $(d.label) estimates")
        hlines!(ax, a.estimate; color=cols[2], linewidth=2.5, label=lbl)
    end
    return ax
end

function plot_cate(f::Union{DrSnow.CausalForest,DrSnow.InstrumentalForest};
                   modifier::Union{Nothing,Symbol}=nothing, level::Real=0.95,
                   bins::Integer=30, colors=nothing, figure::NamedTuple=(;),
                   axis::NamedTuple=(;))
    fig = _mk_figure(figure)
    tgt = f isa DrSnow.CausalForest ? "CATE" : "conditional LATE"
    sub = "$(DrSnow._ml_grf_label(f)), out-of-bag estimates " *
          "($(length(f.forest.trees)) trees); intervals are pointwise"
    ax = modifier === nothing ?
         _mk_axis(fig[1, 1], axis; xlabel="Estimated $(tgt)", ylabel="Count",
                  subtitle=sub) :
         _mk_axis(fig[1, 1], axis; xlabel=string(modifier), ylabel="Estimated $(tgt)",
                  subtitle=sub)
    plot_cate!(ax, f; modifier=modifier, level=level, bins=bins, colors=colors)
    _mk_legend!(fig, ax)
    return fig
end

function plot_rate!(ax::Makie.AbstractAxis, r::DrSnow.RATEEstimate; level::Real=0.95,
                    colors=nothing)
    d = DrSnow._viz_rate_data(r; level=level)
    t = d.toc
    prios = unique(t.priority)
    cols = _mk_colors(colors, length(prios))
    hlines!(ax, 0.0; color=_MK_REF, linewidth=1)
    for (i, p) in enumerate(prios)
        s = t[t.priority .== p, :]
        band!(ax, s.q, s.conf_low, s.conf_high; color=(cols[i], 0.18))
        lines!(ax, s.q, s.estimate; color=cols[i], linewidth=2,
               linestyle=i == 1 ? :solid : (i == 2 ? :dash : :dot), label=p)
        scatter!(ax, s.q, s.estimate; color=cols[i], marker=_mk_marker(i), markersize=7)
    end
    return ax
end

function plot_rate(r::DrSnow.RATEEstimate; level::Real=0.95, colors=nothing,
                   figure::NamedTuple=(;), axis::NamedTuple=(;))
    fig = _mk_figure(figure)
    tb = tidy(r; level=level)
    parts = ["$(tb.term[i]) = $(_mk_fmt(tb.estimate[i])) " *
             "(se $(_mk_fmt(tb.std_error[i])))" for i in 1:nrow(tb)]
    sub = _mk_caption(vcat(parts, "$(r.R) half-sample bootstrap draws, " *
                                  "$(_mk_pct(level)) pointwise bands"))
    ax = _mk_axis(fig[1, 1], axis; xlabel="Share treated by priority (q)",
                  ylabel="TOC(q): effect among top q minus ATE", subtitle=sub)
    plot_rate!(ax, r; level=level, colors=colors)
    _mk_legend!(fig, ax)
    return fig
end
