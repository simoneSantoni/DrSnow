# Shared styling and helpers for all DrSnow plots.

# Categorical palette in a fixed order (never cycled), validated for colour-vision
# deficiencies on adjacent pairs; identity is always also carried by marker shape,
# line style or a label, never by colour alone.
const _MK_PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300",
                     "#4a3aa7", "#e34948"]
const _MK_MARKERS = [:circle, :utriangle, :rect, :diamond, :dtriangle, :pentagon,
                     :hexagon, :star5]
const _MK_INK = "#0b0b0b"         # primary text / data ink
const _MK_INK2 = "#52514e"        # secondary text, neutral data (bin means)
const _MK_REF = "#8a8984"         # reference lines (zero, onset, cutoff)
const _MK_MUTED = "#b9b8b3"       # background data (placebo spaghetti)

function _mk_colors(colors, n::Integer)
    if colors === nothing
        n <= length(_MK_PALETTE) ||
            throw(ArgumentError("more than $(length(_MK_PALETTE)) series: pass `colors` " *
                                "explicitly or plot fewer series"))
        return _MK_PALETTE[1:n]
    end
    cs = colors isa AbstractVector ? collect(colors) : [colors]
    length(cs) >= n || throw(ArgumentError("need at least $n colors, got $(length(cs))"))
    return cs[1:n]
end

_mk_marker(i) = _MK_MARKERS[mod1(i, length(_MK_MARKERS))]

# Fonts, font size and background come from the active Makie theme (default 14 pt on
# white), so `with_theme(drsnow_theme(font=:serif, fontsize=10))` applies to DrSnow plots.
_mk_figure(figure::NamedTuple; size=(720, 460)) = Figure(; merge((size=size,), figure)...)

const _MK_AXIS_STYLE = (xgridvisible=false, ygridcolor=(:black, 0.07),
                        topspinevisible=false, rightspinevisible=false,
                        leftspinecolor=_MK_INK2, bottomspinecolor=_MK_INK2,
                        xtickcolor=_MK_INK2, ytickcolor=_MK_INK2,
                        titlealign=:left, subtitlecolor=_MK_INK2,
                        xlabelcolor=_MK_INK, ylabelcolor=_MK_INK)

"""Create a styled axis at `pos`; `defaults` are plot-specific, `axis` (user) wins."""
_mk_axis(pos, axis::NamedTuple; defaults...) =
    Axis(pos; merge(_MK_AXIS_STYLE, NamedTuple(defaults), axis)...)

"""Horizontal legend below the plot area, if the axis has labelled plots."""
function _mk_legend!(fig, ax; row=2, col=1, nbanks=1, kwargs...)
    axes = ax isa AbstractVector ? ax : [ax]
    plots, labels = Any[], String[]
    for a in axes
        p, l = Makie.get_labeled_plots(a; merge=true, unique=true)
        for (pi, li) in zip(p, l)
            li in labels && continue
            push!(plots, pi)
            push!(labels, li)
        end
    end
    isempty(plots) && return nothing
    return Legend(fig[row, col], plots, labels; orientation=:horizontal,
                  framevisible=false, tellheight=true, tellwidth=false, nbanks=nbanks,
                  labelcolor=_MK_INK, kwargs...)
end

_mk_pct(level) = string(round(100 * level; digits=1)) |> s -> endswith(s, ".0") ?
                 s[1:end-2] * "%" : s * "%"

_mk_fmt(x::Real; digits=3) = isfinite(x) ? string(round(x; sigdigits=digits)) :
                             (x > 0 ? "∞" : "-∞")

"""Rows of `lo`/`hi` that are both present and finite."""
_mk_has_ci(lo, hi) = [!ismissing(a) && !ismissing(b) && isfinite(a) && isfinite(b)
                      for (a, b) in zip(lo, hi)]

"""Offsets that dodge `n` series around each position (total spread ≤ `width`)."""
function _mk_dodge(n::Integer; width=0.5)
    n == 1 && return [0.0]
    δ = min(0.2, width / (n - 1))
    return [(i - (n + 1) / 2) * δ for i in 1:n]
end

"""Join caption `parts` with "; ", breaking lines at about `width` characters."""
function _mk_caption(parts; width::Integer=88)
    lines = String[]
    cur = ""
    for p in parts
        isempty(p) && continue
        cand = isempty(cur) ? p : cur * "; " * p
        if length(cand) > width && !isempty(cur)
            push!(lines, cur * ";")
            cur = p
        else
            cur = cand
        end
    end
    isempty(cur) || push!(lines, cur)
    return join(lines, "\n")
end

_mk_labels(labels, rs, default) =
    labels === nothing ? [default(r) for r in rs] : string.(collect(labels))

function _mk_check_labels(labels, n)
    length(labels) == n || throw(ArgumentError("need one label per result ($n), got " *
                                               "$(length(labels))"))
    return labels
end
