# IV design plots: judge first stage, Rotemberg weights, MTE curve.

# --- judge first stage -------------------------------------------------------------

function _mk_judge_data(data, args; kw...)
    if length(args) == 1
        return DrSnow._viz_judge_data(data, args[1]; kw...)
    elseif length(args) == 2
        tr, len = args
        lv = len isa Symbol ? data[!, len] : len
        return DrSnow._viz_judge_data(data, tr, lv; kw...)
    end
    throw(ArgumentError("plot_judge_first_stage: pass (data, result) or " *
                        "(data, treatment, leniency)"))
end

function _mk_draw_judge!(ax, d; colors=nothing)
    cols = _mk_colors(colors, 2)
    b = d.bins
    zr = extrema(d.leniency)
    f = d.fit
    lines!(ax, [zr...], f.intercept .+ f.slope .* [zr...]; color=cols[2], linewidth=2,
           label="Linear fit (all cases)")
    rangebars!(ax, b.leniency, b.conf_low, b.conf_high; color=(cols[1], 0.8),
               linewidth=1.3, whiskerwidth=5)
    scatter!(ax, b.leniency, b.rate; color=cols[1], markersize=9, strokecolor=:white,
             strokewidth=1, label="Binned treatment rate")
    return ax
end

function DrSnow.plot_judge_first_stage!(ax::Makie.AbstractAxis, data, args...;
                                        nbins::Integer=20, trim::Real=0.01,
                                        level::Real=0.95, colors=nothing)
    d = _mk_judge_data(data, args; nbins=nbins, trim=trim, level=level)
    return _mk_draw_judge!(ax, d; colors=colors)
end

function DrSnow.plot_judge_first_stage(data, args...; nbins::Integer=20,
                                       trim::Real=0.01, level::Real=0.95,
                                       histogram::Bool=true, colors=nothing,
                                       figure::NamedTuple=(;), axis::NamedTuple=(;))
    d = _mk_judge_data(data, args; nbins=nbins, trim=trim, level=level)
    f = d.fit
    parts = ["Slope $(_mk_fmt(f.slope)) (robust s.e. $(_mk_fmt(f.se_slope))), " *
             "$(d.n) cases",
             "$(nbins) equal-count bins with $(_mk_pct(d.level)) CIs",
             d.trim > 0 ? "bins and histogram exclude the " *
                          "$(DrSnow._viz_short(100 * d.trim))% " *
                          "tails of leniency" : ""]
    fig = _mk_figure(figure; size=(720, histogram ? 560 : 460))
    ax = _mk_axis(fig[1, 1], axis; xlabel=histogram ? "" : "Judge leniency",
                  ylabel="Treatment rate", subtitle=_mk_caption(parts))
    _mk_draw_judge!(ax, d; colors=colors)
    if histogram
        hax = _mk_axis(fig[2, 1], (;); xlabel="Judge leniency (leave-one-out)",
                       ylabel="Share of cases")
        w = fill(1 / length(d.leniency), length(d.leniency))
        hist!(hax, d.leniency; bins=30, weights=w, color=(_MK_MUTED, 0.9),
              strokewidth=0)
        ylims!(hax, 0, nothing)
        linkxaxes!(ax, hax)
        hidexdecorations!(ax; grid=false, ticks=false)
        rowsize!(fig.layout, 2, Relative(0.25))
        rowgap!(fig.layout, 1, 6)
        _mk_legend!(fig, ax; row=3)
    else
        _mk_legend!(fig, ax)
    end
    return fig
end

# --- Rotemberg weights ---------------------------------------------------------------

function _mk_draw_rotemberg!(ax, d; x::Symbol=:first_stage_F, colors=nothing)
    x in (:first_stage_F, :shock) ||
        throw(ArgumentError("plot_rotemberg: x must be :first_stage_F or :shock"))
    t = d.table
    nrow(t) > 0 || throw(ArgumentError("plot_rotemberg: no share has an estimate"))
    cols = _mk_colors(colors, 2)
    xs = x === :first_stage_F ? max.(t.first_stage_F, 1e-3) : t.shock
    amax = maximum(t.abs_alpha)
    ms = 6 .+ 30 .* sqrt.(t.abs_alpha ./ amax)       # area ∝ |α|
    hlines!(ax, d.estimate; color=_MK_INK2, linestyle=:dash, linewidth=1.2,
            label="Bartik estimate = $(_mk_fmt(d.estimate))")
    pos = .!t.negative
    any(pos) && scatter!(ax, xs[pos], t.beta[pos]; color=(cols[1], 0.75),
                         markersize=ms[pos], strokecolor=:white, strokewidth=1,
                         label="Positive weight")
    neg = t.negative
    any(neg) && scatter!(ax, xs[neg], t.beta[neg]; color=(:white, 0.0),
                         marker=:utriangle, markersize=ms[neg], strokecolor=cols[2],
                         strokewidth=1.8, label="Negative weight")
    for i in findall(t.labelled)
        text!(ax, xs[i], t.beta[i]; text=" " * t.sector[i], align=(:left, :bottom),
              fontsize=11, color=_MK_INK, offset=(ms[i] / 2.5, ms[i] / 2.5))
    end
    return ax
end

function DrSnow.plot_rotemberg!(ax::Makie.AbstractAxis, r::DrSnow.RotembergDecomposition;
                                x::Symbol=:first_stage_F, label::Integer=5,
                                colors=nothing)
    d = DrSnow._viz_rotemberg_data(r; label=label)
    return _mk_draw_rotemberg!(ax, d; x=x, colors=colors)
end

function DrSnow.plot_rotemberg(r::DrSnow.RotembergDecomposition; x::Symbol=:first_stage_F,
                               label::Integer=5, colors=nothing, figure::NamedTuple=(;),
                               axis::NamedTuple=(;))
    d = DrSnow._viz_rotemberg_data(r; label=label)
    parts = ["Marker area ∝ |Rotemberg weight α̂ₖ|",
             "Σ positive α = $(_mk_fmt(d.positive_weight_sum)), Σ negative α = " *
             "$(_mk_fmt(d.negative_weight_sum))",
             "top 5 shares: $(_mk_fmt(100 * d.top5_share))% of Σ|α|",
             d.n_dropped > 0 ? "$(d.n_dropped) share(s) without an estimate omitted" :
             ""]
    xl = x === :first_stage_F ? "First-stage F of the share instrument (log scale)" :
         "Shock gₖ"
    defaults = x === :first_stage_F ? (xscale=log10, xautolimitmargin=(0.05, 0.12)) :
               (xautolimitmargin=(0.05, 0.12),)
    fig = _mk_figure(figure)
    ax = _mk_axis(fig[1, 1], axis; xlabel=xl, ylabel="Just-identified estimate β̂ₖ",
                  subtitle=_mk_caption(parts), defaults...)
    _mk_draw_rotemberg!(ax, d; x=x, colors=colors)
    _mk_legend!(fig, ax)
    return fig
end

# --- MTE curve -------------------------------------------------------------------------

const _MK_PARAM_STYLES = [:dash, :dot, :dashdot, :dashdotdot, (:dot, :loose),
                          (:dash, :loose)]

function _mk_draw_mte!(ax, d; parameters::Bool=true, propensity::Bool=true,
                       colors=nothing)
    c = d.curve
    npar = parameters ? nrow(d.parameters) : 0
    cols = _mk_colors(colors, 1 + npar)
    lo, hi = d.support
    lo > 0 && vspan!(ax, 0, lo; color=(:black, 0.05))
    hi < 1 && vspan!(ax, hi, 1; color=(:black, 0.05))
    vlines!(ax, [lo, hi]; color=_MK_REF, linestyle=:dash, linewidth=1)
    hlines!(ax, 0.0; color=_MK_REF, linewidth=1)
    band!(ax, c.u, c.lower, c.upper; color=(cols[1], 0.18))
    lines!(ax, c.u, c.mte; color=cols[1], linewidth=2.4, label="MTE(x̄, u)")
    if npar > 0
        p = d.parameters
        for i in 1:npar
            hlines!(ax, p.estimate[i]; color=cols[1 + i], linewidth=1.5,
                    linestyle=_MK_PARAM_STYLES[mod1(i, length(_MK_PARAM_STYLES))],
                    label="$(p.term[i]) = $(_mk_fmt(p.estimate[i]))")
        end
    end
    if propensity && !isempty(d.propensity)
        # rug of propensity scores along the bottom
        yl = extrema(vcat(c.lower, c.upper, 0.0))
        span = yl[2] - yl[1]
        y0 = yl[1] - 0.06 * span
        linesegments!(ax, vec([Point2f(p, y) for y in (y0, y0 + 0.04 * span),
                               p in d.propensity]); color=(_MK_INK2, 0.25),
                      linewidth=0.6)
    end
    xlims!(ax, 0, 1)
    return ax
end

function DrSnow.plot_mte!(ax::Makie.AbstractAxis, r::DrSnow.MTEEstimate;
                          parameters::Bool=true, propensity::Bool=true, colors=nothing)
    d = DrSnow._viz_mte_data(r; parameters=parameters)
    return _mk_draw_mte!(ax, d; parameters=parameters, propensity=propensity,
                         colors=colors)
end

function DrSnow.plot_mte(r::DrSnow.MTEEstimate; parameters::Bool=true,
                         propensity::Bool=true, colors=nothing, figure::NamedTuple=(;),
                         axis::NamedTuple=(;))
    d = DrSnow._viz_mte_data(r; parameters=parameters)
    extrap = d.method === :semiparametric ? "curve estimated on the support only" :
             "shaded: outside the support (functional-form extrapolation)"
    parts = [DrSnow.method_name(r),
             "band: $(_mk_pct(d.level)) pointwise bootstrap CIs",
             "dashed verticals: common support of P [$(_mk_fmt(d.support.lower)), " *
             "$(_mk_fmt(d.support.upper))]", extrap,
             propensity ? "rug: propensity scores" : ""]
    fig = _mk_figure(figure)
    ax = _mk_axis(fig[1, 1], axis; xlabel="Unobserved resistance to treatment u",
                  ylabel="Marginal treatment effect", subtitle=_mk_caption(parts))
    _mk_draw_mte!(ax, d; parameters=parameters, propensity=propensity, colors=colors)
    _mk_legend!(fig, ax; nbanks=nrow(d.parameters) > 2 ? 2 : 1)
    return fig
end
