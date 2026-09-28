# Plot stubs and plot data for the sequential area: confidence-sequence paths and
# group-sequential boundaries. Makie methods: ext/DrSnowMakieExt/sequential.jl.

"""
    plot_confidence_sequence(r; fixed=true, null=true, figure=(;), axis=(;),
                             colors=nothing) -> Makie.Figure

Path of an anytime-valid confidence sequence as observations accrue: the running
estimate inside the confidence-sequence band, compared with the fixed-sample
confidence interval at the same level.

A ``(1 - \\alpha)`` confidence sequence is a sequence of intervals ``C_n`` whose
coverage holds *uniformly over time*, ``P(\\theta \\in C_n \\text{ for all } n) \\ge
1 - \\alpha`` (Howard, Ramdas, McAuliffe and Sekhon 2021). The experiment can
therefore be monitored continuously and stopped at any data-dependent time, and the
interval at the stopping time is still valid. The plot draws the running estimate
(line) inside the shaded band ``C_n`` against the sample size ``n``. With
`fixed = true` the conventional fixed-sample interval ``\\hat\\theta_n \\pm
z_{1-\\alpha/2}\\, \\hat\\sigma_n / \\sqrt{n}`` is added as dashed lines: it is valid
only at a single sample size fixed in advance, and using it at every look, as naive
"peeking" does, inflates the type I error well beyond ``\\alpha`` (Armitage,
McPherson and Rowe 1969). The gap between the two is the price of anytime validity.
With `null = true` the null value of the always-valid test is drawn as a dotted line;
the sequential test rejects at the first ``n`` at which the band excludes it
(Johari, Koomen, Pekelis and Walsh 2022).

The band is wider than the fixed-sample interval at every ``n`` and shrinks at a
slightly slower rate; asymptotic confidence sequences (Waudby-Smith, Arbour, Sinha,
Kennedy and Ramdas 2024) are valid only as the sample grows, like a CLT interval.
Infinite bounds (at very small ``n``) are not drawn.

# Arguments
- `r`: a [`ConfidenceSequence`](@ref), a [`SequentialTest`](@ref) that records an
  interval path (e.g. from [`msprt_test`](@ref)), or a sequential monitor
  ([`MeanMonitor`](@ref), [`ATEMonitor`](@ref), [`MSPRTMonitor`](@ref)) created with
  `record = true`.

# Keywords
- `fixed::Bool = true`: draw the fixed-sample interval for comparison (when the
  result provides it).
- `null::Bool = true`: draw the null value of the always-valid test.
$(_VIZ_COMMON_KW)

# Returns
- `Makie.Figure`. The subtitle reports the interval at the last observation.

# Examples
```julia
using DrSnow, CairoMakie, DataFrames, StableRNGs
rng = StableRNG(1)
df = DataFrame(d=rand(rng, 300) .< 0.5)
df.y = randn(rng, 300) .+ 0.4 .* df.d
plot_confidence_sequence(confseq_ate(df, :y, :d; propensity=0.5))
```

# References
- Armitage, P., McPherson, C. K., & Rowe, B. C. (1969). Repeated significance tests
  on accumulating data. *Journal of the Royal Statistical Society: Series A*,
  132(2), 235–244.
- Howard, S. R., Ramdas, A., McAuliffe, J., & Sekhon, J. (2021). Time-uniform,
  nonparametric, nonasymptotic confidence sequences. *Annals of Statistics*, 49(2),
  1055–1080.
- Johari, R., Koomen, P., Pekelis, L., & Walsh, D. (2022). Always valid inference:
  Continuous monitoring of A/B tests. *Operations Research*, 70(3), 1806–1821.
- Waudby-Smith, I., Arbour, D., Sinha, R., Kennedy, E. H., & Ramdas, A. (2024).
  Time-uniform central limit theory and asymptotic confidence sequences. *Annals of
  Statistics*, 52(6), 2613–2640.
"""
function plot_confidence_sequence end
"""
    plot_confidence_sequence!(ax, r; fixed=true, null=true, colors=nothing) -> ax

Draw a confidence-sequence path into an existing Makie axis. This is the mutating
counterpart of [`plot_confidence_sequence`](@ref), which describes the display; no
legend, title or subtitle is added.

# Arguments
- `ax::Makie.AbstractAxis`: the axis to draw into.
- `r`: a confidence sequence, sequential test or recording monitor, as in
  [`plot_confidence_sequence`](@ref).
- Keywords: those of [`plot_confidence_sequence`](@ref) except `figure` and `axis`.

# Returns
- `ax`, the axis drawn into.
"""
function plot_confidence_sequence! end
plot_confidence_sequence(args...; kwargs...) =
    _viz_no_backend(plot_confidence_sequence, args)
plot_confidence_sequence!(args...; kwargs...) =
    _viz_no_backend(plot_confidence_sequence!, args)

"""
    plot_gs_boundaries(d; analysis=nothing, scale=:z, figure=(;), axis=(;),
                       colors=nothing) -> Makie.Figure

Stopping boundaries of a group-sequential design against the information fraction,
optionally with the test statistics observed at the interim analyses.

A group-sequential design analyses the data at ``K`` planned looks and stops early
for efficacy when the standardized statistic ``Z_k`` crosses the efficacy bound
(and, if the design has one, for futility when it falls below the futility bound).
The bounds come from an error-spending function that allocates the overall type I
error ``\\alpha`` across looks as a function of the information fraction (Lan and
DeMets 1983), so that the probability of ever crossing an efficacy bound under the
null is ``\\alpha``. The plot draws the efficacy bound (and the futility bound, or
the lower bound of a two-sided design) at each look against its information
fraction, with the fixed-sample critical value as a dotted reference line.
O'Brien–Fleming-type spending gives very conservative early bounds that approach
the fixed-sample value at the final look, and Pocock-type spending gives roughly
constant bounds (Jennison and Turnbull 2000).

Passing a [`GroupSequentialAnalysis`](@ref) as `analysis` (or as `d`) overlays the
observed statistics at the looks performed so far, together with the bounds
recomputed at the observed information fractions. A statistic above the efficacy
bound at a look is a rejection at that look. Crossing the fixed-sample line alone,
when the group-sequential bound is higher, is not a rejection. After early stopping,
naive point estimates are biased away from the null and fixed-sample intervals do
not have their nominal coverage; use the repeated confidence intervals and, once the
trial has stopped, the stagewise-ordering (median-unbiased) inference stored in the
analysis (Jennison and Turnbull 2000). With `scale = :p` the bounds are shown as nominal
one-sided p-values on a log scale, the form in which they are often reported.

# Arguments
- `d`: a [`GroupSequentialDesign`](@ref) (from [`gs_design`](@ref)), or a
  [`GroupSequentialAnalysis`](@ref) (from [`gs_analysis`](@ref)), in which case its
  design and observed path are drawn.

# Keywords
- `analysis = nothing`: a [`GroupSequentialAnalysis`](@ref) of `d` to overlay.
- `scale::Symbol = :z`: `:z` (standardized statistic) or `:p` (nominal one-sided
  p-value, log scale).
$(_VIZ_COMMON_KW)

# Returns
- `Makie.Figure`.

# Examples
```julia
using DrSnow, CairoMakie
d = gs_design(; k=4, futility=HSDSpending(-2))
plot_gs_boundaries(d)
plot_gs_boundaries(gs_analysis(d, [0.1, 0.25], [0.2, 0.14]); scale=:p)
```

# References
- Jennison, C., & Turnbull, B. W. (2000). *Group Sequential Methods with
  Applications to Clinical Trials*. Chapman & Hall/CRC.
- Lan, K. K. G., & DeMets, D. L. (1983). Discrete sequential boundaries for clinical
  trials. *Biometrika*, 70(3), 659–663.
- O'Brien, P. C., & Fleming, T. R. (1979). A multiple testing procedure for clinical
  trials. *Biometrics*, 35(3), 549–556.
- Pocock, S. J. (1977). Group sequential methods in the design and analysis of
  clinical trials. *Biometrika*, 64(2), 191–199.
"""
function plot_gs_boundaries end
"""
    plot_gs_boundaries!(ax, d; analysis=nothing, scale=:z, colors=nothing) -> ax

Draw the boundaries of a group-sequential design (and optionally the observed
statistics) into an existing Makie axis. This is the mutating counterpart of
[`plot_gs_boundaries`](@ref), which describes the display; no legend or title is
added, and with `scale = :p` the axis should be given a log scale by the caller.

# Arguments
- `ax::Makie.AbstractAxis`: the axis to draw into.
- `d`: a [`GroupSequentialDesign`](@ref) or [`GroupSequentialAnalysis`](@ref).
- Keywords: those of [`plot_gs_boundaries`](@ref) except `figure` and `axis`.

# Returns
- `ax`, the axis drawn into.
"""
function plot_gs_boundaries! end
plot_gs_boundaries(args...; kwargs...) = _viz_no_backend(plot_gs_boundaries, args)
plot_gs_boundaries!(args...; kwargs...) = _viz_no_backend(plot_gs_boundaries!, args)

# ---------------------------------------------------------------------------
# Plot data
# ---------------------------------------------------------------------------

"""
    _viz_confseq_data(r) -> NamedTuple

`path` (DataFrame with `n`, `estimate`, `lower`, `upper`, `fixed_low`, `fixed_high`;
infinite bounds replaced by `NaN`), `null`, `level`, `label`, `title`.
"""
function _viz_confseq_data(r::ConfidenceSequence)
    isempty(r.n) && throw(ArgumentError("the confidence sequence has no observations"))
    z = critical_value(r.level)
    h = z .* r.sigma ./ sqrt.(r.n)
    fin(v) = [isfinite(x) ? x : NaN for x in v]
    path = DataFrame(n=copy(r.n), estimate=copy(r.estimate), lower=fin(r.lower),
                     upper=fin(r.upper), fixed_low=fin(r.estimate .- h),
                     fixed_high=fin(r.estimate .+ h))
    return (path=path, null=r.null, level=r.level, label=r.estimand,
            title=r.method)
end

function _viz_confseq_data(t::SequentialTest)
    isempty(t.n) && throw(ArgumentError("the sequential test has no observations"))
    all(isnan, t.lower) &&
        throw(ArgumentError("this sequential test has no interval path to plot"))
    fin(v) = [isfinite(x) ? x : NaN for x in v]
    nullv = tryparse(Float64, strip(last(split(t.null, "="))))
    path = DataFrame(n=copy(t.n), estimate=copy(t.estimate), lower=fin(t.lower),
                     upper=fin(t.upper), fixed_low=fill(NaN, length(t.n)),
                     fixed_high=fill(NaN, length(t.n)))
    return (path=path, null=nullv === nothing ? NaN : nullv, level=1 - t.alpha,
            label=strip(first(split(t.null, "="))), title=t.name)
end

_viz_confseq_data(m::MSPRTMonitor) = _viz_confseq_data(sequential_test(m))
_viz_confseq_data(m::SequentialMonitor) = _viz_confseq_data(confidence_sequence(m))
_viz_confseq_data(x) =
    throw(ArgumentError("plot_confidence_sequence needs a ConfidenceSequence, a " *
                        "SequentialTest or a sequential monitor, got $(typeof(x))"))

"""
    _viz_gs_data(d; analysis=nothing) -> NamedTuple

`bounds` (DataFrame: `look`, `timing`, `efficacy`, `lower` — `NaN` when absent),
`observed` (DataFrame: `look`, `timing`, `z`, `efficacy`, or `nothing`), `fixed`
(the fixed-sample critical value), `sided`, `title`.
"""
function _viz_gs_data(d::GroupSequentialDesign; analysis=nothing)
    lower = d.sided == 2 ? -d.efficacy_z :
            [isfinite(a) ? a : NaN for a in d.futility_z]
    bounds = DataFrame(look=1:d.k, timing=copy(d.timing), efficacy=copy(d.efficacy_z),
                       lower=lower)
    obs = nothing
    if analysis !== nothing
        analysis isa GroupSequentialAnalysis ||
            throw(ArgumentError("analysis must be a GroupSequentialAnalysis"))
        obs = DataFrame(look=collect(eachindex(analysis.z)),
                        timing=copy(analysis.timing), z=copy(analysis.z),
                        efficacy=copy(analysis.efficacy_z))
    end
    fixed = quantile(Normal(), 1 - (d.sided == 2 ? d.alpha / 2 : d.alpha))
    return (bounds=bounds, observed=obs, fixed=fixed, sided=d.sided,
            title=_seq_sf_name(d.efficacy))
end

_viz_gs_data(a::GroupSequentialAnalysis; analysis=nothing) =
    _viz_gs_data(a.design; analysis=a)
_viz_gs_data(x; kwargs...) =
    throw(ArgumentError("plot_gs_boundaries needs a GroupSequentialDesign or " *
                        "GroupSequentialAnalysis, got $(typeof(x))"))

"""Nominal one-sided p-value of a Z bound (for the `:p` scale), floored at 1e-12."""
_viz_nominal_p(z::Real) = isnan(z) ? NaN : max(ccdf(Normal(), z), 1e-12)
