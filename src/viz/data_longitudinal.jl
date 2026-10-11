# Plot stubs and backend-independent plot data for the longitudinal area: the
# distribution of the cumulative inverse probability weights of each regime by period.

"""
    plot_regime_weights(r::LongitudinalEstimate; regimes=nothing, figure=(;), axis=(;),
                        colors=nothing) -> Makie.Figure

Distribution of the cumulative inverse probability weights of the followers of each
regime, period by period, as box plots.

For a regime ``d`` the followers through period ``t`` are the units whose observed
treatments equal the regime's actions up to ``t``, and their weights are
``1 / \\max(\\hat g_{1:t}, g_{bound})`` with ``\\hat g_{1:t}`` the estimated cumulative
probability of those actions. The plot draws one box per regime and period (the box
spans the quartiles, the whiskers the 1.5 interquartile range, points beyond are
drawn individually), with the number of followers written above each box and a
dashed line at the bound `1 / g_bound`. Weights that grow quickly with the period,
boxes reaching the bound and shrinking follower counts are the signatures of weak
support for a sustained regime (Cole and Hernán 2008): the inverse-probability and
targeted estimators then rest on few, heavily weighted units, and every estimator
extrapolates. The plot shows the fitted nuisances of the first repetition of the
sample split; see [`regime_weights`](@ref) for the same information as a table.

# Arguments
- `r::LongitudinalEstimate`: from [`longitudinal_regime`](@ref).

# Keywords
- `regimes = nothing`: the regimes to draw, as indices or labels (default: all).
$(_VIZ_COMMON_KW)

# Returns
- `Makie.Figure`. The subtitle states the estimator, the number of units and the
  bound.

# Examples
```julia
using DrSnow, CairoMakie
# with r from longitudinal_regime(...; regime=[always_treat(K), never_treat(K)]):
# plot_regime_weights(r)
```

# References
- Cole, S. R., & Hernán, M. A. (2008). Constructing inverse probability weights for
  marginal structural models. *American Journal of Epidemiology*, 168(6), 656–664.
"""
function plot_regime_weights end
"""
    plot_regime_weights!(ax, r::LongitudinalEstimate; regimes=nothing, colors=nothing)
        -> ax

Draw the regime weight box plots into an existing Makie axis. This is the mutating
counterpart of [`plot_regime_weights`](@ref), which describes the display; no legend
or subtitle is added.

# Arguments
- `ax::Makie.AbstractAxis`: the axis to draw into.
- `r::LongitudinalEstimate`: from [`longitudinal_regime`](@ref).
- Keywords: those of [`plot_regime_weights`](@ref) except `figure` and `axis`.

# Returns
- `ax`, the axis drawn into.
"""
function plot_regime_weights! end
plot_regime_weights(args...; kwargs...) = _viz_no_backend(plot_regime_weights, args)
plot_regime_weights!(args...; kwargs...) = _viz_no_backend(plot_regime_weights!, args)

"""
    _viz_regime_weights_data(r::LongitudinalEstimate; regimes=nothing)

NamedTuple with `table` (`regime::Int`, `label`, `period::Int`, `weight`; one row per
follower and period), `counts` (`regime`, `label`, `period`, `n_following`),
`labels`, `periods`, `bound` (`1 / g_bound`), `method`, `n`.
"""
function _viz_regime_weights_data(r::LongitudinalEstimate; regimes=nothing)
    R = length(r.names)
    sel = if regimes === nothing
        collect(1:R)
    else
        map(collect(regimes)) do x
            j = x isa Integer ? Int(x) : findfirst(==(string(x)), r.names)
            (j === nothing || !(1 <= j <= R)) &&
                throw(ArgumentError("regime $(repr(x)) not found"))
            j
        end
    end
    isempty(sel) && throw(ArgumentError("no regimes selected"))
    reg = Int[]
    lab = String[]
    per = Int[]
    wt = Float64[]
    creg = Int[]
    clab = String[]
    cper = Int[]
    cnt = Int[]
    for (k, j) in enumerate(sel)
        for t in 1:r.K
            f = findall(r.followers[j][:, t])
            append!(reg, fill(k, length(f)))
            append!(lab, fill(r.names[j], length(f)))
            append!(per, fill(t, length(f)))
            append!(wt, r.weights[j][f, t])
            push!(creg, k)
            push!(clab, r.names[j])
            push!(cper, t)
            push!(cnt, length(f))
        end
    end
    return (table=DataFrame(regime=reg, label=lab, period=per, weight=wt),
            counts=DataFrame(regime=creg, label=clab, period=cper, n_following=cnt),
            labels=r.names[sel], periods=r.times, bound=1 / r.g_bound,
            method=method_name(r), n=r.n)
end
