# Raw outcome trends by adoption cohort / ever-treated status.

function _mk_draw_trends!(ax, d; ci::Bool, colors)
    t = d.table
    groups = unique(t[:, [:group, :group_order]])
    sort!(groups, :group_order)
    ng = nrow(groups)
    cols = _mk_colors(colors, ng)
    colof = Dict(g => cols[i] for (i, g) in enumerate(groups.group))
    # adoption dates (first treated period) behind the data
    for o in eachrow(d.onsets)
        c = d.by === :cohort ? colof[o.group] : _MK_REF
        vlines!(ax, o.x; color=(c, 0.9), linestyle=:dash,
                linewidth=1.2)
    end
    for (i, g) in enumerate(groups.group)
        s = t[t.group .== g, :]
        c = colof[g]
        if ci
            ok = _mk_has_ci(s.conf_low, s.conf_high)
            any(ok) && band!(ax, s.x[ok], Float64.(s.conf_low[ok]),
                             Float64.(s.conf_high[ok]); color=(c, 0.15))
        end
        ls = g == "Never treated" ? :dash : :solid
        lines!(ax, s.x, s.mean; color=c, linewidth=2, linestyle=ls)
        scatter!(ax, s.x, s.mean; color=c, marker=_mk_marker(i), markersize=9,
                 strokecolor=:white, strokewidth=1)
        # legend entry combining line and marker
        lines!(ax, [NaN], [NaN]; color=c, linewidth=2, linestyle=ls, label=g)
        scatter!(ax, [NaN], [NaN]; color=c, marker=_mk_marker(i), markersize=9,
                 strokecolor=:white, strokewidth=1, label=g)
    end
    if d.ticks !== nothing
        ax.xticks = d.ticks
    elseif length(unique(t.x)) <= 20 && all(isinteger, t.x)
        ax.xticks = sort!(unique(t.x))
        ax.xtickformat = xs -> string.(round.(Int, xs))
    end
    return ax
end

function _mk_trends_caption(d, ci)
    parts = String[]
    ci && push!(parts, "Bands: $(_mk_pct(d.level)) CIs of the period means")
    push!(parts, d.by === :cohort ? "dashed verticals: adoption date of each cohort" :
                 "dashed verticals: adoption dates")
    d.adjusted && push!(parts, "adjusted for " * join(string.(d.covariates), ", ") *
                               " (within group × period)")
    d.absorbing || push!(parts, "treatment is not absorbing: groups by first " *
                                "treatment")
    return _mk_caption(parts)
end

function DrSnow.plot_trends!(ax::Makie.AbstractAxis, data, outcome::Symbol,
                             args...; by::Symbol=:cohort, covariates=Symbol[],
                             cohorts=nothing, level::Real=0.95, ci::Bool=true,
                             colors=nothing)
    d = DrSnow._viz_trends_data(data, outcome, args...; by=by,
                                covariates=Vector{Symbol}(covariates), level=level,
                                cohorts=cohorts)
    return _mk_draw_trends!(ax, d; ci=ci, colors=colors)
end

function DrSnow.plot_trends(data, outcome::Symbol, args...; by::Symbol=:cohort,
                            covariates=Symbol[], cohorts=nothing, level::Real=0.95,
                            ci::Bool=true, colors=nothing, figure::NamedTuple=(;),
                            axis::NamedTuple=(;))
    d = DrSnow._viz_trends_data(data, outcome, args...; by=by,
                                covariates=Vector{Symbol}(covariates), level=level,
                                cohorts=cohorts)
    fig = _mk_figure(figure)
    ylab = (d.adjusted ? "Mean of $(outcome) (covariate-adjusted)" :
            "Mean of $(outcome)")
    ax = _mk_axis(fig[1, 1], axis; xlabel="Time", ylabel=ylab,
                  subtitle=_mk_trends_caption(d, ci))
    _mk_draw_trends!(ax, d; ci=ci, colors=colors)
    _mk_legend!(fig, ax; nbanks=nrow(unique(d.table[:, [:group]])) > 5 ? 2 : 1)
    return fig
end
