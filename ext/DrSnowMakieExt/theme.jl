# A Makie theme reproducing DrSnow's plot style, for users' own figures and for
# publication settings (fonts / font size are honored by DrSnow's plots too).

function DrSnow.drsnow_theme(; fontsize::Real=14, font::Symbol=:sans, colors=nothing)
    font in (:sans, :serif) || throw(ArgumentError("font must be :sans or :serif"))
    fontsize > 0 || throw(ArgumentError("fontsize must be positive"))
    pal = colors === nothing ? _MK_PALETTE : collect(colors)
    isempty(pal) && throw(ArgumentError("colors must not be empty"))
    ax = (; _MK_AXIS_STYLE..., titlesize=fontsize + 2, subtitlesize=fontsize - 2,
          xlabelsize=fontsize, ylabelsize=fontsize, xticklabelsize=fontsize - 1,
          yticklabelsize=fontsize - 1)
    base = Theme(; fontsize=fontsize, backgroundcolor=:white, textcolor=_MK_INK,
                 palette=(color=pal, marker=_MK_MARKERS,
                          patchcolor=pal),
                 Axis=ax,
                 Legend=(framevisible=false, labelcolor=_MK_INK),
                 Colorbar=(spinewidth=0, ticklabelcolor=_MK_INK2),
                 Lines=(linewidth=2, cycle=[:color]),
                 Scatter=(markersize=10, strokecolor=:white, strokewidth=1,
                          cycle=[:color, :marker]),
                 BarPlot=(cycle=[:color],))
    return font === :serif ? merge(base, Makie.theme_latexfonts()) : base
end
