# Coefficient (forest) plots for any CausalEstimate or vector of them.

_mk_as_vector(r::CausalEstimate) = [r]
_mk_as_vector(rs::AbstractVector) = collect(rs)
_mk_as_vector(r) = DrSnow._viz_unsupported("plot_coefficients", r)

function _mk_draw_coefficients!(ax, cd; colors)
    t = cd.table
    vlines!(ax, 0.0; color=_MK_REF, linestyle=:dash, linewidth=1)
    if cd.single
        # one row per model; identity is carried by the tick label
        c = _mk_colors(colors, 1)[1]
        y = [findfirst(==(m), cd.models) for m in t.model]
        ok = _mk_has_ci(t.conf_low, t.conf_high)
        any(ok) && rangebars!(ax, y[ok], Float64.(t.conf_low[ok]),
                              Float64.(t.conf_high[ok]); direction=:x, color=c,
                              linewidth=1.5, whiskerwidth=7)
        scatter!(ax, t.estimate, y; color=c, markersize=10, strokecolor=:white,
                 strokewidth=1)
        ax.yticks = (1:length(cd.models), cd.models)
    else
        terms = unique(t.term)
        n = length(cd.models)
        cols = _mk_colors(colors, n)
        offs = _mk_dodge(n; width=0.5)
        for (i, m) in enumerate(cd.models)
            s = t[t.model .== m, :]
            y = [findfirst(==(term), terms) + offs[i] for term in s.term]
            ok = _mk_has_ci(s.conf_low, s.conf_high)
            any(ok) && rangebars!(ax, y[ok], Float64.(s.conf_low[ok]),
                                  Float64.(s.conf_high[ok]); direction=:x,
                                  color=cols[i], linewidth=1.5, whiskerwidth=6)
            scatter!(ax, s.estimate, y; color=cols[i], marker=_mk_marker(i),
                     markersize=10, strokecolor=:white, strokewidth=1, label=m)
        end
        ax.yticks = (1:length(terms), terms)
    end
    ax.yreversed = true
    return ax
end

function plot_coefficients!(ax::Makie.AbstractAxis, r; level::Real=0.95, terms=nothing,
                            labels=nothing, colors=nothing)
    cd = DrSnow._viz_coef_data(_mk_as_vector(r); level=level, terms=terms,
                               labels=labels)
    return _mk_draw_coefficients!(ax, cd; colors=colors)
end

function plot_coefficients(r; level::Real=0.95, terms=nothing, labels=nothing,
                           colors=nothing, figure::NamedTuple=(;),
                           axis::NamedTuple=(;))
    cd = DrSnow._viz_coef_data(_mk_as_vector(r); level=level, terms=terms,
                               labels=labels)
    nrows = cd.single ? length(cd.models) : length(unique(cd.table.term))
    fig = _mk_figure(figure; size=(720, clamp(140 + 42 * nrows, 260, 900)))
    ax = _mk_axis(fig[1, 1], axis; xlabel="Estimate",
                  subtitle="Points: estimates; whiskers: $(_mk_pct(level)) CIs",
                  ygridvisible=false, xgridvisible=true, xgridcolor=(:black, 0.07))
    _mk_draw_coefficients!(ax, cd; colors=colors)
    cd.single || _mk_legend!(fig, ax)
    return fig
end
