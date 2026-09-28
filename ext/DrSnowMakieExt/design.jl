# Design area: power curves (grids, surrogate optimization, analytic) and blocks.

const _MK_POWER_TYPES = Union{DrSnow.DesignDiagnosis,DrSnow.DesignOptimization,
                              AbstractVector{DrSnow.PowerAnalysis}}

function _mk_draw_power_curve!(ax, d; target=nothing, colors=nothing)
    pts = d.points
    series = unique(pts.series)
    ncol = length(series) + (d.curve === nothing ? 0 : 1)
    cols = _mk_colors(colors, ncol)
    tgt = target === nothing ? d.target : target
    tgt === nothing || hlines!(ax, Float64(tgt); color=_MK_REF, linestyle=:dash,
                               linewidth=1, label="Target power")
    if d.curve !== nothing
        lines!(ax, d.curve.x, d.curve.power; color=cols[end], linewidth=2.5,
               label=first(d.curve.series))
    end
    d.best === nothing || vlines!(ax, d.best; color=_MK_REF, linestyle=:dot,
                                  linewidth=1.5, label="Chosen design")
    for (i, s) in enumerate(series)
        t = pts[pts.series .== s, :]
        any(t.conf_high .> t.conf_low) &&
            rangebars!(ax, t.x, t.conf_low, t.conf_high; color=(cols[i], 0.6),
                       linewidth=1.2, whiskerwidth=5)
        d.curve === nothing && lines!(ax, t.x, t.power; color=cols[i], linewidth=1.5)
        scatter!(ax, t.x, t.power; color=cols[i], marker=_mk_marker(i), markersize=9,
                 strokecolor=:white, strokewidth=1, label=s)
    end
    return ax
end

function DrSnow.plot_power_curve!(ax::Makie.AbstractAxis, x::_MK_POWER_TYPES;
                                  parameter=nothing, estimator=nothing, level::Real=0.95,
                                  target=nothing, colors=nothing)
    d = DrSnow._viz_power_curve_data(x; parameter=parameter, estimator=estimator,
                                     level=level)
    return _mk_draw_power_curve!(ax, d; target=target, colors=colors)
end

function DrSnow.plot_power_curve(x::_MK_POWER_TYPES; parameter=nothing, estimator=nothing,
                                 level::Real=0.95, target=nothing, colors=nothing,
                                 figure::NamedTuple=(;), axis::NamedTuple=(;))
    d = DrSnow._viz_power_curve_data(x; parameter=parameter, estimator=estimator,
                                     level=level)
    parts = String[]
    if x isa DrSnow.DesignDiagnosis
        push!(parts, "Simulated power ($(x.sims) simulations per design, alpha = " *
                     "$(x.alpha)); whiskers: $(_mk_pct(level)) Monte Carlo intervals")
    elseif x isa DrSnow.DesignOptimization
        push!(parts, "Points: simulated power with $(_mk_pct(level)) binomial " *
                     "intervals; line: fitted surrogate; $(x.total_sims) simulations")
    else
        push!(parts, "Analytic power: " * first(x).design)
    end
    fig = _mk_figure(figure)
    ax = _mk_axis(fig[1, 1], axis; xlabel=d.parameter, ylabel="Power",
                  subtitle=_mk_caption(parts))
    _mk_draw_power_curve!(ax, d; target=target, colors=colors)
    ylims!(ax, -0.02, 1.02)
    _mk_legend!(fig, ax; nbanks=1)
    return fig
end

function DrSnow.plot_blocks!(ax::Makie.AbstractAxis, bd::DrSnow.BlockingDesign;
                             colors=nothing)
    cols = _mk_colors(colors, 1)
    d = DrSnow._viz_blocks_data(bd)
    g = DataFrames.combine(DataFrames.groupby(d, :position), :score => minimum => :lo,
                           :score => maximum => :hi)
    rangebars!(ax, g.position, g.lo, g.hi; color=(_MK_INK2, 0.6), linewidth=1.5,
               whiskerwidth=0, label="Range within block")
    scatter!(ax, d.position, d.score; color=(cols[1], 0.85), markersize=6,
             label="Units")
    return ax
end

function DrSnow.plot_blocks(bd::DrSnow.BlockingDesign; colors=nothing,
                            figure::NamedTuple=(;), axis::NamedTuple=(;))
    B = length(bd.block_sizes)
    sub = "$(length(bd.ids)) units in $B blocks ($(bd.algorithm) " *
          "$(bd.method === :pairs ? "pairs" : "blocks"))" *
          (isnan(bd.r2) ? "" : "; out-of-sample R² of the score = " *
                               string(round(bd.r2; digits=3)))
    fig = _mk_figure(figure)
    ax = _mk_axis(fig[1, 1], axis; xlabel="Block (ordered by mean score)",
                  ylabel="Prognostic score", subtitle=sub)
    DrSnow.plot_blocks!(ax, bd; colors=colors)
    _mk_legend!(fig, ax)
    return fig
end
