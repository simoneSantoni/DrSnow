# Event-study plots (also used for ring / spillover event studies).

# Split results into series: one per result, or one per exposure group.
function _mk_es_series(rs, labels; level, uniform, rng)
    series = Tuple{String,DataFrame}[]
    for (r, lbl) in zip(rs, labels)
        d = DrSnow._viz_event_study_data(r; level=level, uniform=uniform, rng=rng)
        groups = unique(d.group)
        if length(groups) == 1
            push!(series, (lbl, d))
        else
            for g in groups
                name = length(rs) == 1 ? g : lbl * ": " * g
                push!(series, (name, d[d.group .== g, :]))
            end
        end
    end
    return series
end

function _mk_draw_event_study!(ax, series; colors, connect::Bool, uniform::Bool)
    n = length(series)
    cols = _mk_colors(colors, n)
    offs = _mk_dodge(n)
    hlines!(ax, 0.0; color=_MK_REF, linewidth=1)
    vlines!(ax, -0.5; color=_MK_REF, linestyle=:dash, linewidth=1)
    ticks = Dict{Int,String}()
    for (i, (lbl, d)) in enumerate(series)
        c, m = cols[i], _mk_marker(i)
        for row in eachrow(d)
            get!(ticks, row.rel_period, row.label)
        end
        est = d[.!d.reference, :]
        x = est.rel_period .+ offs[i]
        if uniform
            ok = _mk_has_ci(est.uniform_low, est.uniform_high)
            any(ok) && rangebars!(ax, x[ok], Float64.(est.uniform_low[ok]),
                                  Float64.(est.uniform_high[ok]); color=(c, 0.28),
                                  linewidth=7, whiskerwidth=0)
        end
        ok = _mk_has_ci(est.conf_low, est.conf_high)
        any(ok) && rangebars!(ax, x[ok], Float64.(est.conf_low[ok]),
                              Float64.(est.conf_high[ok]); color=c, linewidth=1.5,
                              whiskerwidth=7)
        if connect
            dd = sort(d, :rel_period)
            lines!(ax, dd.rel_period .+ offs[i], dd.estimate; color=(c, 0.7),
                   linewidth=1.2)
        end
        scatter!(ax, x, est.estimate; color=c, marker=m, markersize=10,
                 strokecolor=:white, strokewidth=1, label=lbl)
        ref = d[d.reference, :]
        nrow(ref) > 0 && scatter!(ax, ref.rel_period .+ offs[i], zeros(nrow(ref));
                                  color=:white, marker=m, markersize=10, strokecolor=c,
                                  strokewidth=1.5)
    end
    ks = sort!(collect(keys(ticks)))
    ax.xticks = (ks, [ticks[k] for k in ks])
    return ax
end

function _mk_es_caption(level, uniform, has_ref)
    lv = _mk_pct(level)
    return _mk_caption(["Whiskers: $lv pointwise CIs",
                        uniform ? "shaded bars: $lv uniform (sup-t) bands" : "",
                        has_ref ? "hollow: reference period (normalized to 0)" : ""])
end

function plot_event_study!(ax::Makie.AbstractAxis, r, rs...; level::Real=0.95,
                           uniform::Bool=false, labels=nothing, connect::Bool=false,
                           colors=nothing, rng::Random.AbstractRNG=Random.default_rng())
    all_r = (r, rs...)
    lbls = _mk_check_labels(_mk_labels(labels, all_r, DrSnow.method_name), length(all_r))
    series = _mk_es_series(all_r, lbls; level=level, uniform=uniform, rng=rng)
    return _mk_draw_event_study!(ax, series; colors=colors, connect=connect,
                                 uniform=uniform)
end

function plot_event_study(r, rs...; level::Real=0.95, uniform::Bool=false,
                          labels=nothing, connect::Bool=false, colors=nothing,
                          rng::Random.AbstractRNG=Random.default_rng(),
                          figure::NamedTuple=(;), axis::NamedTuple=(;))
    all_r = (r, rs...)
    lbls = _mk_check_labels(_mk_labels(labels, all_r, DrSnow.method_name), length(all_r))
    series = _mk_es_series(all_r, lbls; level=level, uniform=uniform, rng=rng)
    has_ref = any(s -> any(s[2].reference), series)
    fig = _mk_figure(figure)
    ax = _mk_axis(fig[1, 1], axis; xlabel="Periods relative to treatment",
                  ylabel="Estimated effect",
                  subtitle=_mk_es_caption(level, uniform, has_ref))
    _mk_draw_event_study!(ax, series; colors=colors, connect=connect, uniform=uniform)
    length(series) > 1 && _mk_legend!(fig, ax)
    return fig
end
