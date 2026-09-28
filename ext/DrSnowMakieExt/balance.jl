# Covariate balance ("love") plots.

_mk_pfmt(p) = ismissing(p) ? "–" : p < 0.001 ? "<0.001" : string(round(p; digits=3))

_mk_peq(p) = ismissing(p) ? "p = –" : p < 0.001 ? "p < 0.001" : "p = " * _mk_pfmt(p)

_mk_ptext(row) = _mk_peq(row.pvalue) *
                 (ismissing(row.pvalue_adj) ? "" :
                  " (adj. " * _mk_pfmt(row.pvalue_adj) * ")")

"""Column of p-values right of the balance axis, aligned with its rows."""
function _mk_balance_pcolumn!(fig, ax, d)
    t = d.table
    covs = unique(t.covariate)
    pax = Axis(fig[1, 2]; width=150, yticksvisible=false, yticklabelsvisible=false,
               xticksvisible=false, xticklabelsvisible=false, xgridvisible=false,
               ygridvisible=false, leftspinevisible=false, rightspinevisible=false,
               topspinevisible=false, bottomspinevisible=false,
               xlabel=" ", xlabelcolor=:transparent)
    for (i, c) in enumerate(covs)
        row = t[findfirst(==(c), t.covariate), :]
        text!(pax, 0.0, length(covs) - i + 1; text=_mk_ptext(row),
              align=(:left, :center), fontsize=12, color=_MK_INK2)
    end
    text!(pax, 0.0, length(covs) + 0.45; text="p-value", align=(:left, :center),
          fontsize=12, color=_MK_INK)
    xlims!(pax, 0, 1)
    linkyaxes!(ax, pax)
    ylims!(ax, 0.4, length(covs) + 0.6)
    ylims!(pax, 0.4, length(covs) + 0.6)
    colgap!(fig.layout, 1, 10)
    return pax
end

function _mk_draw_balance!(ax, d; threshold=nothing, annotate::Bool=true, colors=nothing)
    t = d.table
    covs = unique(t.covariate)
    groups = unique(t.group)
    ng = length(groups)
    cols = _mk_colors(colors, ng)
    ypos = Dict(c => length(covs) - i + 1 for (i, c) in enumerate(covs))  # first on top
    offs = ng == 1 ? [0.0] : _mk_dodge(ng; width=0.4)
    thr = threshold === nothing ? d.threshold : threshold === false ? nothing :
          Float64(threshold)
    vlines!(ax, 0.0; color=_MK_REF, linestyle=:dash, linewidth=1)
    if thr !== nothing
        vlines!(ax, [-thr, thr]; color=_MK_REF, linestyle=:dot, linewidth=1)
    end
    for (i, g) in enumerate(groups)
        s = t[t.group .== g, :]
        y = [ypos[c] for c in s.covariate] .+ offs[i]
        ok = _mk_has_ci(s.conf_low, s.conf_high)
        any(ok) && rangebars!(ax, y[ok], Float64.(s.conf_low[ok]),
                              Float64.(s.conf_high[ok]); direction=:x, color=cols[i],
                              linewidth=1.5, whiskerwidth=7)
        scatter!(ax, s.estimate, y; color=cols[i], marker=_mk_marker(i), markersize=11,
                 strokecolor=:white, strokewidth=1, label=isempty(g) ? nothing : g)
    end
    ax.yticks = ([ypos[c] for c in covs], covs)
    ylims!(ax, 0.4, length(covs) + 0.6)
    # x range: data, intervals and thresholds, symmetric around zero
    vals = Float64[]
    append!(vals, t.estimate)
    append!(vals, Float64.(collect(skipmissing(t.conf_low))))
    append!(vals, Float64.(collect(skipmissing(t.conf_high))))
    thr === nothing || append!(vals, [-thr, thr])
    m = maximum(abs, filter(isfinite, vals); init=0.0)
    m = m > 0 ? 1.1 * m : 1.0
    xlims!(ax, -m, m)
    if annotate && any(!ismissing, t.pvalue) && ng == 1
        # p-values appended to the covariate labels (the figure form uses a column)
        lbl = Dict(r.covariate => r.covariate * "  " * _mk_ptext(r) for r in eachrow(t))
        ax.yticks = ([ypos[c] for c in covs], [lbl[c] for c in covs])
    end
    return ax
end

function _mk_balance_caption(d, threshold)
    parts = String[]
    thr = threshold === nothing ? d.threshold : threshold === false ? nothing : threshold
    if d.kind === :pretreatment
        push!(parts, "Cohort vs comparison group in the cohort's pre-treatment periods")
        push!(parts, "descriptive, not a test")
    elseif d.kind === :ri
        d.test === nothing ||
            push!(parts, "Omnibus randomization test: statistic " *
                         "$(_mk_fmt(d.test.statistic)), $(_mk_peq(d.test.pvalue))")
        push!(parts, "p-values: randomization, adj. = Westfall–Young")
    else
        push!(parts, "Whiskers: robust bias-corrected CIs")
        push!(parts, "adj. = Holm-adjusted p-values")
    end
    thr === nothing || push!(parts, "dotted: ±$(_mk_fmt(thr))")
    return _mk_caption(parts)
end

function DrSnow.plot_balance!(ax::Makie.AbstractAxis, x; data=nothing, threshold=nothing,
                              annotate::Bool=true, colors=nothing)
    d = DrSnow._viz_balance_data(x; data=data)
    return _mk_draw_balance!(ax, d; threshold=threshold, annotate=annotate,
                             colors=colors)
end

function DrSnow.plot_balance(x; data=nothing, threshold=nothing, annotate::Bool=true,
                             colors=nothing, figure::NamedTuple=(;), axis::NamedTuple=(;))
    d = DrSnow._viz_balance_data(x; data=data)
    ncov = length(unique(d.table.covariate))
    fig = _mk_figure(figure; size=(720, max(300, 150 + 38 * ncov)))
    ax = _mk_axis(fig[1, 1], axis; xlabel=d.xlabel,
                  subtitle=_mk_balance_caption(d, threshold), ygridvisible=false,
                  xgridvisible=true, xgridcolor=(:black, 0.07))
    single = length(unique(d.table.group)) == 1
    pcol = annotate && single && any(!ismissing, d.table.pvalue)
    _mk_draw_balance!(ax, d; threshold=threshold, annotate=false, colors=colors)
    pcol && _mk_balance_pcolumn!(fig, ax, d)
    single || _mk_legend!(fig, ax)
    return fig
end
