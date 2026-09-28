# Plot stubs and backend-independent plot data for the design area (power curves,
# blocks formed on a prognostic score). The Makie methods live in
# ext/DrSnowMakieExt/design.jl.

"""
    plot_power_curve(x; parameter=nothing, estimator=nothing, level=0.95,
                     target=nothing, figure=(;), axis=(;),
                     colors=nothing) -> Makie.Figure

Statistical power as a function of a design parameter (sample size, number of
clusters, effect size, ...), from simulation-based design diagnosis or analytic
power formulas.

Power is the probability that the design's test rejects the null when the specified
effect is present. Reading power off a curve rather than a single number shows how
sensitive it is to the design parameter and to the assumed effect, which is the
point of diagnosing a design before running it (Blair, Cooper, Coppock and Humphreys
2019). Three inputs are supported:

- A [`DesignDiagnosis`](@ref) from [`diagnose_grid`](@ref): simulated power at each
  grid point, with `level` Monte Carlo intervals from the bootstrap Monte Carlo
  standard errors, one series per estimator and per combination of the other varied
  parameters. The intervals quantify simulation error only (they shrink with the
  number of simulations), not uncertainty about the assumed data-generating process.
- A [`DesignOptimization`](@ref) from [`optimize_design`](@ref): simulated power at
  the evaluated designs (points with binomial `level` intervals), the fitted
  surrogate curve used to search the design space, the target power (dashed) and
  the chosen design (dotted). With several parameters the curve is drawn along
  `parameter`, with the other parameters fixed at the chosen design.
- A vector of [`PowerAnalysis`](@ref) results (an analytic power curve), for example
  `[power_means(effect=0.3, n=n) for n in 50:10:400]`; analytic power has no
  simulation error, so no intervals are drawn.

Power depends on the assumed effect size and variance components, which are usually
uncertain; a curve computed at an optimistic effect overstates power everywhere. A
target of 0.8 is a convention, not a requirement, and designs just reaching it are
fragile to those assumptions.

# Arguments
- `x`: a [`DesignDiagnosis`](@ref), a [`DesignOptimization`](@ref), or a vector of
  [`PowerAnalysis`](@ref) results.

# Keywords
- `parameter::Union{Nothing,Symbol} = nothing`: the design parameter on the
  horizontal axis (default: the only varied parameter).
- `estimator = nothing`: estimator label(s) to show for a diagnosis (default: all).
- `level::Real = 0.95`: level of the Monte Carlo or binomial intervals.
- `target::Union{Nothing,Real} = nothing`: draw a target-power line (default: the
  optimization's target, if any).
$(_VIZ_COMMON_KW)

# Returns
- `Makie.Figure`. The subtitle states the kind of power (simulated or analytic) and
  the number of simulations.

# Examples
```julia
using DrSnow, CairoMakie
plot_power_curve([power_means(effect=0.3, n=n) for n in 50:25:400]; target=0.8)
plot_power_curve([power_cluster(effect=0.3, icc=0.1, cluster_size=20, n_clusters=J)
                  for J in 10:4:80]; parameter=:n_clusters, target=0.8)
```

# References
- Blair, G., Cooper, J., Coppock, A., & Humphreys, M. (2019). Declaring and
  diagnosing research designs. *American Political Science Review*, 113(3),
  838–859.
"""
function plot_power_curve end
"""
    plot_power_curve!(ax, x; parameter=nothing, estimator=nothing, level=0.95,
                      target=nothing, colors=nothing) -> ax

Draw a power curve into an existing Makie axis. This is the mutating counterpart of
[`plot_power_curve`](@ref), which describes the supported inputs; no legend or
subtitle is added.

# Arguments
- `ax::Makie.AbstractAxis`: the axis to draw into.
- `x`: a design diagnosis, design optimization or vector of power analyses, as in
  [`plot_power_curve`](@ref).
- Keywords: those of [`plot_power_curve`](@ref) except `figure` and `axis`.

# Returns
- `ax`, the axis drawn into.
"""
function plot_power_curve! end
plot_power_curve(args...; kwargs...) = _viz_no_backend(plot_power_curve, args)
plot_power_curve!(args...; kwargs...) = _viz_no_backend(plot_power_curve!, args)

"""
    plot_blocks(bd::BlockingDesign; figure=(;), axis=(;),
                colors=nothing) -> Makie.Figure

The blocks of a blocked (stratified) randomization built on a prognostic score: the
score of every unit by block, with the range within each block.

Blocking randomizes treatment within groups of similar units, so that the treatment
and control groups are balanced on the blocking variables by design and the
variance of the effect estimate falls to the extent that the blocks predict the
outcome (Imai, King and Nall 2009). When the blocks are formed on a *prognostic
score*, a prediction of the untreated outcome from baseline covariates fitted on
separate (e.g. pilot or historical) data (Hansen 2008), the relevant question is how
homogeneous each block is in that prediction. The plot shows one column per block,
ordered by the block's mean score, with every unit's score as a point and the
within-block range as a vertical segment. Tight blocks mean that the design balances
the predicted outcome closely; wide blocks (typically at the tails of the score
distribution, where units are sparse) contribute less precision.

Homogeneity in the score says nothing about balance on outcome determinants the
score does not capture, and a score fitted on the experimental sample's own
outcomes would make the analysis depend on the outcomes. The plot requires a design
built on a score; designs matched on several covariates by Mahalanobis distance
have no single score to display.

# Arguments
- `bd::BlockingDesign`: from [`block_design`](@ref) with a score.

# Keywords
$(_VIZ_COMMON_KW)

# Returns
- `Makie.Figure`.

# Examples
```julia
using DrSnow, CairoMakie, DataFrames, StableRNGs
df = DataFrame(id=1:30, x=randn(StableRNG(1), 30))
plot_blocks(block_design(df, :x; id=:id, method=:blocks, block_size=3))
```

# References
- Hansen, B. B. (2008). The prognostic analogue of the propensity score.
  *Biometrika*, 95(2), 481–488.
- Imai, K., King, G., & Nall, C. (2009). The essential role of pair matching in
  cluster-randomized experiments, with application to the Mexican universal health
  insurance evaluation. *Statistical Science*, 24(1), 29–53.
"""
function plot_blocks end
"""
    plot_blocks!(ax, bd::BlockingDesign; colors=nothing) -> ax

Draw the blocks of a score-based blocking design into an existing Makie axis. This
is the mutating counterpart of [`plot_blocks`](@ref), which describes the display;
no legend or subtitle is added.

# Arguments
- `ax::Makie.AbstractAxis`: the axis to draw into.
- `bd::BlockingDesign`: from [`block_design`](@ref) with a score.
- `colors`: as in [`plot_blocks`](@ref).

# Returns
- `ax`, the axis drawn into.
"""
function plot_blocks! end
plot_blocks(args...; kwargs...) = _viz_no_backend(plot_blocks, args)
plot_blocks!(args...; kwargs...) = _viz_no_backend(plot_blocks!, args)

# ------------------------------------------------------------------ plot data

function _viz_power_param(params::Vector{Symbol}, parameter, what)
    isempty(params) &&
        throw(ArgumentError("plot_power_curve: the $what varies no design parameter"))
    if parameter === nothing
        length(params) == 1 ||
            throw(ArgumentError("plot_power_curve: several parameters vary " *
                                "($(join(params, ", "))); choose one with `parameter`"))
        return params[1]
    end
    Symbol(parameter) in params ||
        throw(ArgumentError("plot_power_curve: `$(parameter)` is not a varied parameter"))
    return Symbol(parameter)
end

"""
    _viz_power_curve_data(x; parameter=nothing, estimator=nothing, level=0.95)

Power-curve data: `(points, curve, parameter, target, best)` where `points` has
columns `x`, `power`, `conf_low`, `conf_high`, `series`; `curve` (or `nothing`) has
`x`, `power`, `series`; `target` and `best` are `nothing` or numbers.
"""
function _viz_power_curve_data(r::DesignDiagnosis; parameter=nothing, estimator=nothing,
                               level::Real=0.95)
    p = _viz_power_param(r.parameters, parameter, "diagnosis")
    d = r.diagnosands[r.diagnosands.diagnosand .== :power, :]
    if estimator !== nothing
        keep = string.(estimator isa AbstractVector ? estimator : [estimator])
        d = d[in.(d.estimator, Ref(Set(keep))), :]
        nrow(d) > 0 || throw(ArgumentError("plot_power_curve: unknown estimator"))
    end
    others = setdiff(r.parameters, [p])
    z = critical_value(level)
    series = [join(vcat(["$(o) = $(row[o])" for o in others], [row.estimator]), ", ")
              for row in eachrow(d)]
    se = coalesce.(d.mc_se, 0.0)
    pts = DataFrame(x=Float64.(d[!, p]), power=Float64.(d.value),
                    conf_low=clamp.(d.value .- z .* se, 0, 1),
                    conf_high=clamp.(d.value .+ z .* se, 0, 1), series=series)
    sort!(pts, [:series, :x])
    return (points=pts, curve=nothing, parameter=string(p), target=nothing, best=nothing)
end

function _viz_power_curve_data(r::DesignOptimization; parameter=nothing, estimator=nothing,
                               level::Real=0.95)
    params = collect(keys(r.best))
    p = parameter === nothing ?
        (length(params) == 1 ? params[1] :
         throw(ArgumentError("plot_power_curve: several parameters; choose `parameter`"))) :
        Symbol(parameter)
    p in params || throw(ArgumentError("plot_power_curve: unknown parameter $(parameter)"))
    others = setdiff(params, [p])
    ev = r.evaluations
    mask = trues(nrow(ev))
    for o in others
        mask .&= ev[!, o] .== r.best[o]
    end
    g = combine(groupby(ev[mask, :], p), :rejections => sum => :k, :sims => sum => :m)
    z = critical_value(level)
    pw = g.k ./ g.m
    se = sqrt.(pw .* (1 .- pw) ./ g.m)
    pts = DataFrame(x=Float64.(g[!, p]), power=pw, conf_low=clamp.(pw .- z .* se, 0, 1),
                    conf_high=clamp.(pw .+ z .* se, 0, 1),
                    series=fill("Simulated ($(r.estimator))", nrow(g)))
    sort!(pts, :x)
    pr = r.predictions
    pm = trues(nrow(pr))
    for o in others
        pm .&= pr[!, o] .== r.best[o]
    end
    cur = DataFrame(x=Float64.(pr[pm, p]), power=Float64.(pr.power[pm]),
                    series=fill("Surrogate ($(r.surrogate))", count(pm)))
    sort!(cur, :x)
    return (points=pts, curve=cur, parameter=string(p), target=r.target,
            best=float(r.best[p]))
end

function _viz_power_curve_data(rs::AbstractVector{PowerAnalysis}; parameter=nothing,
                               estimator=nothing, level::Real=0.95)
    isempty(rs) && throw(ArgumentError("plot_power_curve: no power analyses"))
    p = if parameter === nothing
        ks = [k for k in keys(rs[1].parameters)
              if k !== :power && all(r -> haskey(r.parameters, k), rs) &&
                 length(unique(r.parameters[k] for r in rs)) > 1]
        length(ks) == 1 ||
            throw(ArgumentError("plot_power_curve: choose the varying parameter with " *
                                "`parameter` (candidates: $(join(ks, ", ")))"))
        ks[1]
    else
        Symbol(parameter)
    end
    all(r -> haskey(r.parameters, p), rs) ||
        throw(ArgumentError("plot_power_curve: `$(p)` is not a parameter of every " *
                            "analysis"))
    x = Float64[r.parameters[p] for r in rs]
    pw = Float64[r.power for r in rs]
    o = sortperm(x)
    pts = DataFrame(x=x[o], power=pw[o], conf_low=pw[o], conf_high=pw[o],
                    series=fill("Analytic", length(rs)))
    return (points=pts, curve=nothing, parameter=string(p), target=nothing, best=nothing)
end

_viz_power_curve_data(r; kwargs...) = _viz_unsupported("plot_power_curve", r)

"""
    _viz_blocks_data(bd::BlockingDesign) -> DataFrame

Columns `unit`, `block`, `position` (rank of the block by mean score) and `score`.
"""
function _viz_blocks_data(bd::BlockingDesign)
    bd.score === nothing &&
        throw(ArgumentError("plot_blocks: the design was not built on a score"))
    B = length(bd.block_sizes)
    means = [mean(bd.score[bd.blocks .== b]) for b in 1:B]
    rank = invperm(sortperm(means))
    return DataFrame(unit=string.(bd.ids), block=bd.blocks, position=rank[bd.blocks],
                     score=bd.score)
end

_viz_blocks_data(r) = _viz_unsupported("plot_blocks", r)
