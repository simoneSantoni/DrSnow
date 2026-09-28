# Plot data: backend-independent extraction of what each plot draws.
#
# Every `_viz_*_data` function returns plain DataFrames / NamedTuples, is tested
# without Makie, and is the extension point for new result types (add a method here;
# see the header of `viz.jl`). The Makie extension only draws these tables.

_viz_unsupported(plot, r) =
    throw(ArgumentError("$plot: no plotting method for $(typeof(r)); see the \"Adding " *
                        "plotting support\" section of the results documentation"))

# ---------------------------------------------------------------------------------
# glance hooks for result types that store the number of clusters elsewhere
# ---------------------------------------------------------------------------------

_glance_nclusters(r::IVEstimate) =
    (n = _iv_nclusters(r.design); n > 0 ? n : missing)

# ---------------------------------------------------------------------------------
# Event studies
# ---------------------------------------------------------------------------------

"""
    _viz_event_study_data(r; level=0.95, uniform=false, rng=default_rng())

Event-time coefficients as a DataFrame with columns `rel_period::Int`, `label`
(tick label, `≤k` / `≥k` for binned endpoints), `estimate`, `conf_low`,
`conf_high`, `uniform_low`, `uniform_high` (`missing` unless `uniform`), `reference`
(`true` for normalized periods, drawn at zero without intervals) and `group`
(`""` for single-series results; the exposure group for spillover event studies).
Rows are sorted by `group` and `rel_period`.
"""
function _viz_event_study_data(r::EventStudyEstimate; level::Real=0.95,
                               uniform::Bool=false,
                               rng::AbstractRNG=Random.default_rng())
    rp = relative_periods(r)
    ci = StatsAPI.confint(r; level=level)
    uci = uniform ? StatsAPI.confint(r; level=level, uniform=true, rng=rng) : nothing
    labels = replace.(StatsAPI.coefnames(r), "e<=" => "≤", "e>=" => "≥", "e=" => "")
    n = length(rp)
    opt(v) = Vector{Union{Missing,Float64}}(v)
    df = DataFrame(rel_period=collect(Int, rp), label=String.(labels),
                   estimate=Vector{Float64}(StatsAPI.coef(r)),
                   conf_low=opt(ci[:, 1]), conf_high=opt(ci[:, 2]),
                   uniform_low=opt(uci === nothing ? fill(missing, n) : uci[:, 1]),
                   uniform_high=opt(uci === nothing ? fill(missing, n) : uci[:, 2]),
                   reference=falses(n), group=fill("", n))
    for k in r.reference
        k in rp && continue
        push!(df, (k, string(k), 0.0, missing, missing, missing, missing, true, ""))
    end
    return sort!(df, :rel_period)
end

function _viz_event_study_data(r::SpilloverEventStudy; level::Real=0.95,
                               uniform::Bool=false,
                               rng::AbstractRNG=Random.default_rng())
    uniform && throw(ArgumentError("uniform bands are not available for spillover " *
                                   "event studies"))
    t = r.table
    crit = critical_value(level, StatsAPI.dof_residual(r))
    lbl(e) = e == -r.leads ? "≤$(e)" : e == r.lags ? "≥$(e)" : string(e)
    groups = unique(String.(t.group))
    opt(v) = Vector{Union{Missing,Float64}}(v)
    none = fill(missing, nrow(t))
    df = DataFrame(rel_period=Vector{Int}(t.event_time),
                   label=[lbl(e) for e in t.event_time],
                   estimate=Vector{Float64}(t.estimate),
                   conf_low=opt(t.estimate .- crit .* t.std_error),
                   conf_high=opt(t.estimate .+ crit .* t.std_error),
                   uniform_low=opt(none), uniform_high=opt(none),
                   reference=falses(nrow(t)), group=String.(t.group))
    for g in groups
        push!(df, (-1, "-1", 0.0, missing, missing, missing, missing, true, g))
    end
    order = Dict(g => i for (i, g) in enumerate(groups))
    df.group_order = [order[g] for g in df.group]
    sort!(df, [:group_order, :rel_period])
    return select!(df, Not(:group_order))
end

_viz_event_study_data(r; kwargs...) = _viz_unsupported("plot_event_study", r)

# ---------------------------------------------------------------------------------
# Coefficients (forest plots)
# ---------------------------------------------------------------------------------

_viz_select_terms(names, ::Nothing) = trues(length(names))
_viz_select_terms(names, re::Regex) = occursin.(re, names)
_viz_select_terms(names, f::Function) = BitVector(f.(names))
function _viz_select_terms(names, terms::AbstractVector)
    want = string.(terms)
    missing_t = setdiff(want, names)
    isempty(missing_t) || throw(ArgumentError("terms not found: $(join(missing_t, ", "))"))
    return in.(names, Ref(Set(want)))
end

"""
    _viz_coef_data(rs; level=0.95, terms=nothing, labels=nothing)

`tidy` output of one or several results (`model` column holds `labels`), restricted
to `terms`, with rows lacking a standard error dropped. Also returns whether every
model contributes exactly one row (then the plot uses one row per model).
"""
function _viz_coef_data(rs::AbstractVector; level::Real=0.95, terms=nothing,
                        labels=nothing)
    isempty(rs) && throw(ArgumentError("plot_coefficients: no results to plot"))
    t = tidy(rs; level=level, names=labels)
    t = t[_viz_select_terms(t.term, terms), :]
    dropped = count(ismissing, t.std_error)
    dropped > 0 && @warn "plot_coefficients: $dropped coefficient(s) without a " *
                         "standard error are drawn without an interval"
    nrow(t) == 0 && throw(ArgumentError("plot_coefficients: no coefficients selected"))
    models = unique(t.model)
    length(models) == length(rs) ||
        throw(ArgumentError("plot_coefficients: labels must be unique"))
    single = all(m -> count(==(m), t.model) == 1, models)
    return (table=t, models=models, single=single)
end

_viz_coef_data(r::CausalEstimate; kwargs...) = _viz_coef_data([r]; kwargs...)
_viz_coef_data(r; kwargs...) = _viz_unsupported("plot_coefficients", r)

# ---------------------------------------------------------------------------------
# Regression discontinuity
# ---------------------------------------------------------------------------------

"""
    _viz_rd_local_fit(r::RDEstimate; npoints=100)

Local polynomial fits of the outcome within the main bandwidth on each side,
evaluated on a grid (`side`, `x`, `y`), from the stored coefficients (powers of
`x - cutoff`). Not available for covariate-adjusted fits, whose intercepts are not
outcome levels.
"""
function _viz_rd_local_fit(r::RDEstimate; npoints::Integer=100)
    isempty(r.covariates) ||
        throw(ArgumentError("plot_rd: local fits are not drawn for covariate-adjusted " *
                            "estimates"))
    poly(b, u) = sum(b[k] * u^(k - 1) for k in eachindex(b))
    xl = range(r.cutoff - r.h_left, r.cutoff; length=npoints)
    xr = range(r.cutoff, r.cutoff + r.h_right; length=npoints)
    return DataFrame(side=vcat(fill(:left, npoints), fill(:right, npoints)),
                     x=vcat(collect(xl), collect(xr)),
                     y=vcat([poly(r.beta_left, x - r.cutoff) for x in xl],
                            [poly(r.beta_right, x - r.cutoff) for x in xr]))
end

# ---------------------------------------------------------------------------------
# Synthetic control
# ---------------------------------------------------------------------------------

const _VIZ_SYNTH = Union{SyntheticControlEstimate,SyntheticDiDEstimate,
                         AugmentedSCEstimate,MatrixCompletionEstimate}

"""
    _viz_synth_data(r; cohort=nothing, placebo_cutoff=Inf)

NamedTuple with
- `gaps`: `synth_gaps(r)` with a `cohort` column (`missing` without staggering),
- `cohorts`: cohort labels (adoption periods), `cohort`: the one selected for the
  trajectory panel,
- `onset`: first treated period of each cohort (vector aligned with `cohorts`),
- `time_weights`: `synth_time_weights(r)` of the selected cohort for synthetic DiD
  (`nothing` otherwise),
- `placebo_gaps`: long DataFrame (`unit`, `time`, `gap`) of in-space placebo gaps for
  classic synthetic control fitted with placebos (else `nothing`).
"""
function _viz_synth_data(r::_VIZ_SYNTH; cohort=nothing, placebo_cutoff::Real=Inf)
    g = synth_gaps(r)
    if !hasproperty(g, :cohort)
        first_post = findfirst(g.post)
        onset_t = first_post === nothing ? nothing : g.time[first_post]
        g.cohort = Vector{Union{Missing,eltype(g.time)}}(fill(missing, nrow(g)))
        cohorts = Any[missing]
        onsets = Any[onset_t]
    else
        cohorts = Any[c for c in unique(g.cohort)]
        onsets = Any[c for c in cohorts]
    end
    sel = if cohort === nothing
        first(cohorts)
    else
        cohort in skipmissing(cohorts) ||
            throw(ArgumentError("plot_synth: cohort $cohort not found; cohorts are " *
                                join(string.(cohorts), ", ")))
        cohort
    end
    tw = nothing
    if r isa SyntheticDiDEstimate && r.method !== :did
        w = synth_time_weights(r)
        tw = hasproperty(w, :cohort) ? w[w.cohort .== sel, [:time, :weight]] : w
    end
    pl = nothing
    if r isa SyntheticControlEstimate && r.placebo !== nothing
        p = r.placebo
        keep = findall(p.pre_rmspe .<= placebo_cutoff * r.pre_rmspe)
        T = size(p.gaps, 2)
        times = r.panel.times[1:T]
        pl = DataFrame(unit=repeat(p.units[keep]; inner=T),
                       time=repeat(times; outer=length(keep)),
                       gap=vec(permutedims(p.gaps[keep, :])))
    end
    return (gaps=g, cohorts=cohorts, cohort=sel, onset=onsets, time_weights=tw,
            placebo_gaps=pl)
end

_viz_synth_data(r; kwargs...) = _viz_unsupported("plot_synth", r)

# ---------------------------------------------------------------------------------
# Randomization distributions
# ---------------------------------------------------------------------------------

"""
    _viz_randomization_data(r; hypothesis=1)

NamedTuple `(values, weights, observed, alternative, pvalue, name, draws)` of a
randomization reference distribution.
"""
function _viz_randomization_data(r::RandomizationTestResult; hypothesis::Integer=1)
    v, w = randomization_distribution(r)
    return (values=v, weights=w, observed=r.observed, alternative=r.alternative,
            pvalue=r.pvalue, name=r.statistic_name, draws="assignments")
end

function _viz_randomization_data(r::MultipleTestingResult; hypothesis::Integer=1)
    1 <= hypothesis <= length(r.observed) ||
        throw(ArgumentError("hypothesis must be between 1 and $(length(r.observed))"))
    v, w = randomization_distribution(r)
    return (values=v[:, hypothesis], weights=w, observed=r.observed[hypothesis],
            alternative=r.alternative, pvalue=r.pvalues[hypothesis],
            name=r.statistic_name * " (" * r.hypotheses[hypothesis] * ")",
            draws="assignments")
end

function _viz_randomization_data(t::DiagnosticTest; hypothesis::Integer=1)
    haskey(t.details, :placebo_statistics) ||
        throw(ArgumentError("plot_randomization_distribution: this DiagnosticTest does " *
                            "not store a placebo / randomization distribution"))
    v = Vector{Float64}(t.details.placebo_statistics)
    alt = occursin("RMSPE", t.name) ? :greater : :two_sided
    return (values=v, weights=ones(length(v)), observed=t.statistic, alternative=alt,
            pvalue=t.pvalue, name=t.name, draws="placebo units")
end

_viz_randomization_data(r; kwargs...) =
    _viz_unsupported("plot_randomization_distribution", r)

# ---------------------------------------------------------------------------------
# Heterogeneity (ML)
# ---------------------------------------------------------------------------------

"""
    _viz_gates_data(g::GenericMLInference)

NamedTuple `(gates, ate)`: the GATES rows G1…GK (without the `GK - G1` contrast)
and the BLP ATE row (`estimate`, `lower`, `upper`).
"""
function _viz_gates_data(g::GenericMLInference)
    t = gates(g)
    t = t[.!occursin.(" - ", t.group), :]
    b = blp(g)
    return (gates=t, ate=b[1, :], level=g.level)
end

_viz_gates_data(r) = _viz_unsupported("plot_gates", r)

# ---------------------------------------------------------------------------------
# Spillovers (non-event-study results)
# ---------------------------------------------------------------------------------

"""
    _viz_exposure_effect_data(r; level=0.95)

`tidy(r)` with a `kind` column (`"direct"` or `"spillover"`), terms in their
original order (direct first).
"""
function _viz_exposure_effect_data(r::Union{SpilloverRegression,ExposureEffects,
                                            TwoStageEffects};
                                   level::Real=0.95)
    t = tidy(r; level=level)
    t.kind = [occursin(r"spill|indirect|exposed|ring|hop"i, s) ? "spillover" : "direct"
              for s in t.term]
    return t
end

_viz_exposure_effect_data(r; kwargs...) = _viz_unsupported("plot_spillover_rings", r)

# ---------------------------------------------------------------------------------
# Weak-IV confidence sets
# ---------------------------------------------------------------------------------

"""
    _viz_confidence_set_data(s::WeakIVConfidenceSet; limits=nothing, npoints=801)

NamedTuple `(grid, pvalue, level, intervals, estimate, limits)`: the p-value
function of the inverted test evaluated on a grid.
"""
function _viz_confidence_set_data(s::WeakIVConfidenceSet; limits=nothing,
                                  npoints::Integer=801)
    npoints >= 10 || throw(ArgumentError("npoints must be at least 10"))
    lo, hi = limits === nothing ? _viz_cs_limits(s) : Float64.(limits)
    lo < hi || throw(ArgumentError("limits must be increasing"))
    grid = collect(range(lo, hi; length=npoints))
    p = [s.pvalue_function(b) for b in grid]
    return (grid=grid, pvalue=p, level=s.level, intervals=s.intervals,
            estimate=s.estimate, limits=(lo, hi), method=s.method)
end

function _viz_cs_limits(s::WeakIVConfidenceSet)
    ends = Float64[e for iv in s.intervals for e in iv if isfinite(e)]
    isfinite(s.estimate) && push!(ends, s.estimate)
    isempty(ends) && return (-1.0, 1.0)
    lo, hi = extrema(ends)
    span = max(hi - lo, 0.5 * abs(s.estimate), 1.0)
    unbounded = any(iv -> !isfinite(iv[1]) || !isfinite(iv[2]), s.intervals)
    pad = unbounded ? 1.0 * span : 0.35 * span
    return (lo - pad, hi + pad)
end

_viz_confidence_set_data(r; kwargs...) = _viz_unsupported("plot_confidence_set", r)

# ---------------------------------------------------------------------------------
# Generalized random forests (variable importance, CATE by covariate, RATE / TOC)
# ---------------------------------------------------------------------------------

"""
    _viz_variable_importance_data(f::GeneralizedRandomForest; decay_exponent=2,
                                  max_depth=4, top=nothing)

`variable_importance(f)` sorted by decreasing importance (ties by covariate
order), restricted to the `top` most important covariates when given.
"""
function _viz_variable_importance_data(f::GeneralizedRandomForest;
                                       decay_exponent::Real=2, max_depth::Integer=4,
                                       top::Union{Nothing,Integer}=nothing)
    vi = variable_importance(f; decay_exponent=decay_exponent, max_depth=max_depth)
    vi = vi[sortperm(vi.importance; rev=true, alg=MergeSort), :]
    top === nothing && return vi
    top >= 1 || throw(ArgumentError("plot_variable_importance: top must be ≥ 1"))
    return vi[1:min(Int(top), nrow(vi)), :]
end

_viz_variable_importance_data(r; kwargs...) =
    _viz_unsupported("plot_variable_importance", r)

"""
    _viz_forest_cate_data(f::Union{CausalForest,InstrumentalForest};
                          modifier=nothing, level=0.95, ate=nothing)

Out-of-bag CATE estimates with pointwise `level` intervals (little-bags variance)
as a DataFrame with columns `x` (the `modifier` covariate, `missing` without one),
`estimate`, `conf_low`, `conf_high`, sorted by `x`; plus the AIPW average effect
(`ate` may pass a precomputed [`HTEEstimate`](@ref)) and its interval.
"""
function _viz_forest_cate_data(f::Union{CausalForest,InstrumentalForest};
                               modifier::Union{Nothing,Symbol}=nothing,
                               level::Real=0.95, ate=nothing)
    pi = predict_interval(f; level=level)
    x = if modifier === nothing
        Vector{Union{Missing,Float64}}(missing, nobs(f))
    else
        j = findfirst(==(modifier), f.covariates)
        j === nothing && throw(ArgumentError("plot_cate: $modifier is not a covariate " *
                                             "of the forest; available: " *
                                             join(f.covariates, ", ")))
        Vector{Union{Missing,Float64}}(f.X[:, j])
    end
    t = DataFrame(x=x, estimate=pi.estimate, conf_low=pi.conf_low,
                  conf_high=pi.conf_high)
    t = t[.!isnan.(t.estimate), :]
    modifier === nothing || sort!(t, :x)
    a = ate === nothing ? average_treatment_effect(f) : ate
    ci = StatsAPI.confint(a; level=level)
    return (cate=t, ate=(estimate=coef(a)[1], conf_low=ci[1, 1], conf_high=ci[1, 2]),
            level=Float64(level), modifier=modifier,
            label=f isa CausalForest ? "CATE" : "conditional LATE")
end

"""
    _viz_rate_data(r::RATEEstimate; level=0.95)

The TOC curve of a RATE estimate: `r.toc` with pointwise `conf_low` / `conf_high`
(normal intervals from the bootstrap standard errors), one block per priority
rule (and the difference with two rules).
"""
function _viz_rate_data(r::RATEEstimate; level::Real=0.95)
    c = critical_value(level)
    t = copy(r.toc)
    t.conf_low = t.estimate .- c .* t.std_error
    t.conf_high = t.estimate .+ c .* t.std_error
    return (toc=t, rate=tidy(r; level=level), target=r.target, level=Float64(level))
end

_viz_rate_data(r; kwargs...) = _viz_unsupported("plot_rate", r)
