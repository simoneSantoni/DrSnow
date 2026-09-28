# Classic synthetic control of Abadie & Gardeazabal (2003) and Abadie, Diamond &
# Hainmueller (2010, 2015): donor weights W minimise the V-weighted distance between
# the treated unit's predictors and the synthetic unit's, with V chosen to minimise the
# pre-treatment outcome MSPE (nested optimisation, as in the R package Synth).

struct _sc_AdhSpec
    predictors::Vector{Symbol}
    predictor_periods::Union{Nothing,Vector{Any}}
    special_predictors::Vector{Pair{Symbol,Vector{Any}}}
    fit_periods::Union{Nothing,Vector{Any}}
    v::Union{Nothing,Vector{Float64}}
    v_method::Symbol
    standardize::Bool
    v_starts::Int
end

"""
    SyntheticControlEstimate <: CausalEstimate

Result of [`synthetic_control`](@ref): a classic Abadie–Diamond–Hainmueller synthetic
control for one treated unit.

The headline coefficient `"ATT"` is the average post-treatment gap between the treated
unit and its synthetic control, ``\\frac{1}{T - T_0}\\sum_{t > T_0} (Y_{1t} -
\\sum_j \\hat w_j Y_{jt})``. Inference for this estimator is design-based. Use
[`synth_in_space_placebo`](@ref) for the permutation p-value, which is exact only if
treatment is exchangeable across units. `vcov` returns the variance of the placebo ATTs
obtained by treating each donor in turn (the placebo variance of Arkhangelsky et al.
2021, Algorithm 4). That variance assumes that treated and donor units have the same
error variance, and `vcov` throws if placebos were not computed (`placebo = false`). The
properties `donors`, `times` and `treated_unit` give the donor ids, the time values and
the id of the treated unit.

# Fields
- `att::Float64`: average post-treatment gap.
- `weights::Vector{Float64}`: donor weights, in the order of the `donors` property.
- `v::Vector{Float64}`, `predictor_names::Vector{String}`: predictor weights (diagonal
  of ``V``, summing to one) and predictor labels.
- `predictor_balance::DataFrame`: treated, synthetic and donor-average predictor
  values.
- `treated_path`, `synthetic_path`: outcome paths of the treated unit and its synthetic
  control; `n_pre::Int`: number of pre-treatment periods.
- `loss_v::Float64`: MSPE over the fit periods; `loss_w::Float64`: ``V``-weighted
  predictor discrepancy, on the scale used in the optimisation.
- `pre_rmspe`, `post_rmspe`: root mean squared prediction errors (gaps) before and after
  treatment.
- `placebo`: `nothing`, or a `NamedTuple` with in-space placebo results for each donor
  (`units`, `gaps`, `pre_rmspe`, `post_rmspe`, `att`).
- `panel::SynthPanel`, `treated_row::Int`, `donor_rows::Vector{Int}`: the data and the
  rows used.
"""
struct SyntheticControlEstimate{P<:SynthPanel} <: CausalEstimate
    att::Float64
    panel::P
    treated_row::Int
    donor_rows::Vector{Int}
    weights::Vector{Float64}
    v::Vector{Float64}
    predictor_names::Vector{String}
    predictor_balance::DataFrame
    treated_path::Vector{Float64}
    synthetic_path::Vector{Float64}
    n_pre::Int
    loss_v::Float64
    loss_w::Float64
    pre_rmspe::Float64
    post_rmspe::Float64
    placebo::Union{Nothing,NamedTuple}
    spec::_sc_AdhSpec
end

StatsAPI.coef(r::SyntheticControlEstimate) = [r.att]
StatsAPI.coefnames(::SyntheticControlEstimate) = ["ATT"]
StatsAPI.nobs(r::SyntheticControlEstimate) =
    (1 + length(r.donor_rows)) * length(r.treated_path)
function StatsAPI.vcov(r::SyntheticControlEstimate)
    r.placebo === nothing &&
        throw(ArgumentError("no placebo variance available; re-run synthetic_control " *
                            "with placebo = true (or use synth_in_space_placebo)"))
    a = r.placebo.att
    return fill(sum(abs2, a .- mean(a)) / length(a), 1, 1)
end
estimand(::SyntheticControlEstimate) = "ATT of the treated unit (average post-period gap)"
method_name(::SyntheticControlEstimate) = "Synthetic control (Abadie et al.)"

function Base.getproperty(r::SyntheticControlEstimate, s::Symbol)
    s === :donors && return getfield(r, :panel).units[getfield(r, :donor_rows)]
    s === :times && return getfield(r, :panel).times[1:length(getfield(r, :treated_path))]
    s === :treated_unit && return getfield(r, :panel).units[getfield(r, :treated_row)]
    return getfield(r, s)
end
Base.propertynames(r::SyntheticControlEstimate) =
    (fieldnames(typeof(r))..., :donors, :times, :treated_unit)

# ---------------------------------------------------------------------------------------
# Predictors
# ---------------------------------------------------------------------------------------

function _sc_period_indices(p::SynthPanel, periods, what::AbstractString, n_pre::Integer)
    tidx = Dict(t => j for (j, t) in enumerate(p.times))
    idx = Int[]
    for t in periods
        haskey(tidx, t) || throw(ArgumentError("synthetic_control: $what period " *
                                               "$(repr(t)) is not in the data"))
        j = tidx[t]
        j <= n_pre || throw(ArgumentError("synthetic_control: $what period $(repr(t)) " *
                                          "is not before treatment"))
        push!(idx, j)
    end
    isempty(idx) && throw(ArgumentError("synthetic_control: empty $what period set"))
    return idx
end

function _sc_variable_matrix(p::SynthPanel, var::Symbol)
    var === p.outcome && return p.Y
    k = findfirst(==(var), p.covariates)
    k === nothing && throw(ArgumentError("synthetic_control: predictor `$var` is not a " *
                                         "covariate of the panel"))
    return view(p.X, :, :, k)
end

function _sc_mean_over(M, row, idx, var, unit)
    vals = [M[row, j] for j in idx if !ismissing(M[row, j])]
    isempty(vals) && throw(ArgumentError("synthetic_control: predictor `$var` is " *
                                         "missing in all selected periods for unit " *
                                         "$(repr(unit))"))
    return mean(vals)
end

# k × length(rows) matrix of predictor values and their labels.
function _sc_adh_predictors(p::SynthPanel, spec::_sc_AdhSpec, rows, n_pre)
    if isempty(spec.predictors) && isempty(spec.special_predictors)
        return Matrix{Float64}(transpose(p.Y[rows, 1:n_pre])),
               ["$(p.outcome)[$(t)]" for t in p.times[1:n_pre]]
    end
    cols = Vector{Vector{Float64}}()
    names = String[]
    pidx = spec.predictor_periods === nothing ? collect(1:n_pre) :
           _sc_period_indices(p, spec.predictor_periods, "predictor", n_pre)
    for var in spec.predictors
        M = _sc_variable_matrix(p, var)
        push!(cols, [_sc_mean_over(M, r, pidx, var, p.units[r]) for r in rows])
        push!(names, string(var))
    end
    for (var, periods) in spec.special_predictors
        M = _sc_variable_matrix(p, var)
        idx = _sc_period_indices(p, periods, "special predictor", n_pre)
        push!(cols, [_sc_mean_over(M, r, idx, var, p.units[r]) for r in rows])
        tp = p.times[idx]
        push!(names, length(tp) == 1 ? "$(var)[$(tp[1])]" :
                     "$(var)[mean $(tp[1])–$(tp[end])]")
    end
    return Matrix{Float64}(transpose(reduce(hcat, cols))), names
end

# ---------------------------------------------------------------------------------------
# Nested optimisation
# ---------------------------------------------------------------------------------------

function _sc_normalize_v(p)
    s = sum(abs, p)
    return s > 0 ? abs.(p) ./ s : fill(1 / length(p), length(p))
end

function _sc_adh_w(X0s, x1s, v; support=nothing)
    sv = sqrt.(v)
    return _sc_simplex_ls(sv .* X0s, sv .* x1s; support=support)
end

function _sc_adh_optimize_v(X0s, x1s, Z0, z1, v_starts, rng)
    k = size(X0s, 1)
    last_support = Ref(trues(size(X0s, 2)))
    loss(p) = begin
        s = sum(abs, p)
        s > 0 || return Inf
        w = _sc_adh_w(X0s, x1s, abs.(p) ./ s; support=last_support[])
        last_support[] = w .> 0
        mean(abs2, z1 .- Z0 * w)
    end
    starts = Vector{Vector{Float64}}([fill(1 / k, k)])
    # Synth's regression-based start: diag(B B') with B from regressing outcomes on
    # standardised predictors.
    Xall = hcat(ones(size(X0s, 2) + 1), transpose(hcat(x1s, X0s)))
    Zall = hcat(z1, Z0)
    if size(Xall, 1) > size(Xall, 2) && rank(Xall) == size(Xall, 2)
        B = (Xall' * Xall) \ (Xall' * Zall')
        sv2 = vec(sum(abs2, B[2:end, :]; dims=2))
        sum(sv2) > 0 && push!(starts, sv2 ./ sum(sv2))
    end
    for _ in 1:v_starts
        push!(starts, _sc_normalize_v(rand(rng, k)))
    end
    best_p, best_f = starts[1], Inf
    nm(x) = _sc_nelder_mead(loss, x; max_iter=5000, ftol=1e-10, xtol=1e-10)
    for s in starts
        p, f = nm(s)
        for _ in 1:5   # restart from the optimum while it still improves
            p2, f2 = nm(p)
            improved = f2 < f - 1e-10 * abs(f)
            if f2 <= f
                p, f = p2, f2
            end
            improved || break
        end
        if f < best_f
            best_p, best_f = p, f
        end
    end
    return _sc_normalize_v(best_p)
end

function _sc_adh_fit(p::SynthPanel, spec::_sc_AdhSpec, treated_row::Int,
                     donor_rows::Vector{Int}, n_pre::Int, rng::AbstractRNG)
    length(donor_rows) >= 2 ||
        throw(ArgumentError("synthetic_control: at least two donor units are needed"))
    rows = vcat(treated_row, donor_rows)
    Xall, names = _sc_adh_predictors(p, spec, rows, n_pre)
    k = size(Xall, 1)
    X1raw = Xall[:, 1]
    X0raw = Xall[:, 2:end]
    for j in 1:k
        if std(X0raw[j, :]) == 0
            throw(ArgumentError("synthetic_control: predictor $(names[j]) has no " *
                                "variation across donor units; remove it"))
        end
    end
    Xs = spec.standardize ? Xall ./ std(Xall; dims=2) : Xall
    x1s, X0s = Xs[:, 1], Xs[:, 2:end]
    fit_idx = spec.fit_periods === nothing ? collect(1:n_pre) :
              _sc_period_indices(p, spec.fit_periods, "fit", n_pre)
    z1 = p.Y[treated_row, fit_idx]
    Z0 = Matrix(transpose(p.Y[donor_rows, fit_idx]))
    v = if spec.v !== nothing
        length(spec.v) == k ||
            throw(ArgumentError("synthetic_control: v has length $(length(spec.v)), " *
                                "expected one weight per predictor ($k)"))
        all(>=(0), spec.v) && sum(spec.v) > 0 ||
            throw(ArgumentError("synthetic_control: v must be non-negative and not all 0"))
        spec.v ./ sum(spec.v)
    elseif k == 1 || spec.v_method === :equal
        fill(1 / k, k)
    else
        _sc_adh_optimize_v(X0s, x1s, Z0, z1, spec.v_starts, rng)
    end
    w = _sc_adh_w(X0s, x1s, v)
    loss_v = mean(abs2, z1 .- Z0 * w)
    d = x1s .- X0s * w
    loss_w = dot(d, v .* d)
    treated = p.Y[treated_row, :]
    synthetic = vec(transpose(p.Y[donor_rows, :]) * w)
    balance = DataFrame(predictor=names, treated=X1raw, synthetic=X0raw * w,
                        donor_mean=vec(mean(X0raw; dims=2)), v=v)
    return (; w, v, names, balance, loss_v, loss_w, treated, synthetic)
end

function _sc_rmspe(gap, n_pre)
    T = length(gap)
    pre = sqrt(mean(abs2, gap[1:n_pre]))
    post = sqrt(mean(abs2, gap[(n_pre + 1):T]))
    return pre, post
end

function _sc_adh_placebos(p, spec, treated_row, donor_rows, n_pre, rng,
                          pool::Symbol)
    J = length(donor_rows)
    T = size(p.Y, 2)
    gaps = zeros(J, T)
    pre = zeros(J)
    post = zeros(J)
    att = zeros(J)
    seeds = task_seeds(rng, J)
    Threads.@threads for j in 1:J
        others = [d for d in donor_rows if d != donor_rows[j]]
        pool === :all && push!(others, treated_row)
        f = _sc_adh_fit(p, spec, donor_rows[j], others, n_pre, Random.Xoshiro(seeds[j]))
        g = f.treated .- f.synthetic
        gaps[j, :] = g
        pre[j], post[j] = _sc_rmspe(g, n_pre)
        att[j] = mean(g[(n_pre + 1):T])
    end
    return (units=p.units[donor_rows], gaps=gaps, pre_rmspe=pre, post_rmspe=post,
            att=att, pool=pool)
end

function _sc_adh_estimate(p::SynthPanel, spec::_sc_AdhSpec, placebo::Bool,
                          rng::AbstractRNG, pool::Symbol=:donors)
    _sc_n_treated(p) == 1 ||
        throw(ArgumentError("synthetic_control: the classic synthetic control handles " *
                            "exactly one treated unit (found $(_sc_n_treated(p))); use " *
                            "synthetic_did or augmented_synthetic_control for several"))
    treated_row = size(p.Y, 1)
    donor_rows = collect(1:p.n_control)
    n_pre = p.adoption[treated_row] - 1
    f = _sc_adh_fit(p, spec, treated_row, donor_rows, n_pre, rng)
    gap = f.treated .- f.synthetic
    pre, post = _sc_rmspe(gap, n_pre)
    att = mean(gap[(n_pre + 1):end])
    pl = placebo ? _sc_adh_placebos(p, spec, treated_row, donor_rows, n_pre, rng, pool) :
         nothing
    return SyntheticControlEstimate(att, p, treated_row, donor_rows, f.w, f.v, f.names,
                                    f.balance, f.treated, f.synthetic, n_pre, f.loss_v,
                                    f.loss_w, pre, post, pl, spec)
end

"""
    synthetic_control(data, outcome, treatment, unit, time; predictors=Symbol[],
                      predictor_periods=nothing, special_predictors=Pair[],
                      fit_periods=nothing, v=nothing, v_method=nothing,
                      standardize=nothing, placebo=true, placebo_pool=:donors,
                      v_starts=0, rng=Random.default_rng()) -> SyntheticControlEstimate
    synthetic_control(panel::SynthPanel; kwargs...) -> SyntheticControlEstimate

Classic synthetic control estimator (Abadie & Gardeazabal 2003; Abadie, Diamond &
Hainmueller 2010, 2015) for a single treated unit, reproducing the R package `Synth`.

In a comparative case study, one aggregate unit (a region, a country) is exposed to an
intervention from period ``T_0 + 1`` onwards. The target is the effect on that unit,
``\\tau_{1t} = Y_{1t}(1) - Y_{1t}(0)`` for ``t > T_0``, and its average over the
post-treatment periods, which is the headline `"ATT"`. The counterfactual
``Y_{1t}(0)`` is estimated by a *synthetic control*, a weighted average
``\\sum_j w_j Y_{jt}`` of untreated donor units. The weights are non-negative and sum
to one, and they are chosen so that the synthetic control reproduces the treated unit's
pre-treatment characteristics and outcomes. Abadie, Diamond and Hainmueller (2010)
justify the method under a linear factor model
``Y_{it}(0) = \\delta_t + \\theta_t^\\top Z_i + \\lambda_t^\\top \\mu_i +
\\varepsilon_{it}``. If the weights reproduce the treated unit's pre-treatment outcomes
and covariates, the bias of the estimator is bounded by a term that shrinks as the
number of pre-treatment periods grows relative to the scale of the transitory shocks.
**That bound assumes a (near-)perfect pre-treatment fit.** With an imperfect fit the
estimator can be biased even with many pre-treatment periods (Ferman & Pinto 2021), and
the method should not be used when the treated unit lies far outside the convex hull of
the donors (Abadie 2021). [`augmented_synthetic_control`](@ref) corrects for imperfect
fit. The assumptions of no anticipation and no spillovers to donors are untestable and
must be argued from the application (Abadie 2021).

The weights ``W`` minimise ``(X_1 - X_0 W)^\\top V (X_1 - X_0 W)`` over the simplex,
where ``X_1`` and ``X_0`` hold the predictors of the treated unit and the donors. The
diagonal predictor-weight matrix ``V`` is chosen by a nested search that minimises the
mean squared prediction error (MSPE) of the outcome over the fit periods, as in `Synth`.
Predictors are standardised by their standard deviation across units, and Nelder–Mead
is started from equal weights and from `Synth`'s regression-based weights, plus
`v_starts` random points if requested. The inner problem is solved exactly through a
non-negative least-squares reformulation. When no predictors are given, the outcome in
every pre-treatment period is used, with equal weights and no standardisation
(outcome-only synthetic control). The nested problem is not convex and often has several
local optima. Klößner, Kaul, Pfeifer and Schieler (2018) show that the predictor
weights, and hence the donor weights, can be far from unique. The default deterministic
starts reproduce `Synth`'s solutions (for example the Basque weights of Abadie and
Gardeazabal 2003). Random starts can find a ``V`` with lower pre-treatment MSPE and
different donor weights. When conclusions depend on the weights, report their
sensitivity to ``V``, or fix ``V`` with `v`. Kaul, Klößner, Pfeifer and Schieler (2022)
show that when all pre-treatment outcomes are used as separate predictors, the other
covariates become irrelevant to the fit.

Inference is design-based. The in-space placebo test ([`synth_in_space_placebo`](@ref))
re-estimates the synthetic control with each donor in the role of the treated unit and
ranks the treated unit's post/pre RMSPE ratio among the placebos. That test is exact
only if treatment is exchangeable across units, that is, if the treated unit could
equally have been any of the donors (Abadie 2021). `vcov` returns the variance of the
placebo ATTs (Arkhangelsky et al. 2021, Algorithm 4). It is a rough scale of the
estimator's noise under homoskedasticity across units, not a sampling variance with
known coverage. Complement the estimate with a leave-one-out analysis
([`synth_leave_one_out`](@ref)) and a backdating check
([`synth_in_time_placebo`](@ref)). Report the donor weights, the predictor balance and
the gap plot ([`synth_gaps`](@ref)). With several treated units, or when a level
difference between the treated unit and the donors is plausible, prefer
[`synthetic_did`](@ref); with a poor pre-treatment fit, prefer
[`augmented_synthetic_control`](@ref).

# Arguments
- `data`: long panel (see [`synth_panel`](@ref)), with columns `outcome`, `treatment`,
  `unit` and `time`; exactly one unit may be treated. Or
- `panel::SynthPanel`: a prepared panel.

# Keywords
- `predictors::Vector{Symbol}=Symbol[]`: variables averaged, ignoring missing values,
  over `predictor_periods`. The outcome itself may be listed.
- `predictor_periods=nothing`: time values over which `predictors` are averaged
  (default: all pre-treatment periods).
- `special_predictors=Pair[]`: pairs `variable => periods`, each averaged over its own
  periods, for example `[:gdpcap => 1960:1969, :popdens => [1969]]`.
- `fit_periods=nothing`: periods whose outcome MSPE is minimised when choosing ``V``
  (default: all pre-treatment periods).
- `v=nothing`: user-supplied predictor weights (diagonal of ``V``); skips the nested
  search.
- `v_method=nothing`: `:optimize` (the default with predictors) or `:equal` (the default
  without predictors).
- `standardize=nothing`: divide predictors by their standard deviation across units
  (default `true` with predictors, `false` for outcome-only fits).
- `placebo::Bool=true`: run in-space placebos (each donor treated in turn, with ``V``
  re-optimised). They are needed for [`synth_in_space_placebo`](@ref) and `vcov`, and
  they multiply the computing time by about the number of donors.
- `placebo_pool::Symbol=:donors`: donor pool of each placebo fit. With `:donors` (the
  default, as in Abadie et al. 2010) placebos use the other donors only, so each placebo
  unit has one donor fewer than the treated unit and the permutation is not exactly
  symmetric. With `:all` the treated unit is also a potential donor. This makes the
  procedure a symmetric permutation test of the sharp null of no effect on any unit,
  but the placebo gaps are contaminated when that null is false.
- `v_starts::Integer=0`: additional random starting points for the ``V`` search.
- `rng::AbstractRNG=Random.default_rng()`: generator for the random starting points.

# Returns
- [`SyntheticControlEstimate`](@ref).

# Examples
```julia
using DrSnow, CSV, DataFrames
prop99 = CSV.read(pkgdir(DrSnow, "test", "validation", "synth", "prop99.csv"), DataFrame)
r = synthetic_control(prop99, :PacksPerCapita, :treated, :State, :Year)
coef(r), r.pre_rmspe
synth_in_space_placebo(r)

basque = CSV.read(pkgdir(DrSnow, "test", "validation", "synth", "basque.csv"), DataFrame;
                  missingstring="NA")
basque = basque[basque.regionno .!= 1, :]                    # drop Spain as a whole
basque.treated = Int.((basque.regionno .== 17) .& (basque.year .>= 1970))
rb = synthetic_control(basque, :gdpcap, :treated, :regionname, :year;
                       predictors=[:invest], predictor_periods=1964:1969,
                       special_predictors=[:gdpcap => 1960:1969, :popdens => [1969]],
                       placebo=false)
synth_weights(rb)
rb.predictor_balance
```

# References
- Abadie, A., & Gardeazabal, J. (2003). The economic costs of conflict: A case study
  of the Basque Country. *American Economic Review*, 93(1), 113–132.
- Abadie, A., Diamond, A., & Hainmueller, J. (2010). Synthetic control methods for
  comparative case studies: Estimating the effect of California's tobacco control
  program. *Journal of the American Statistical Association*, 105(490), 493–505.
- Abadie, A., Diamond, A., & Hainmueller, J. (2015). Comparative politics and the
  synthetic control method. *American Journal of Political Science*, 59(2), 495–510.
- Abadie, A. (2021). Using synthetic controls: Feasibility, data requirements, and
  methodological aspects. *Journal of Economic Literature*, 59(2), 391–425.
- Ferman, B., & Pinto, C. (2021). Synthetic controls with imperfect pretreatment fit.
  *Quantitative Economics*, 12(4), 1197–1221.
- Klößner, S., Kaul, A., Pfeifer, G., & Schieler, M. (2018). Comparative politics and
  the synthetic control method revisited: A note on Abadie et al. (2015). *Swiss
  Journal of Economics and Statistics*, 154(1), 11.
- Kaul, A., Klößner, S., Pfeifer, G., & Schieler, M. (2022). Standard synthetic control
  methods: The case of using all preintervention outcomes together with covariates.
  *Journal of Business & Economic Statistics*, 40(3), 1362–1376.
- Abadie, A., Diamond, A., & Hainmueller, J. (2011). Synth: An R package for synthetic
  control methods in comparative case studies. *Journal of Statistical Software*,
  42(13), 1–17.
"""
function synthetic_control(data, outcome::Symbol, treatment::Symbol, unit::Symbol,
                           time::Symbol; predictors::Vector{Symbol}=Symbol[],
                           special_predictors=Pair{Symbol,Any}[], kwargs...)
    vars = unique(vcat(predictors, Symbol[first(sp) for sp in special_predictors]))
    covs = [v for v in vars if v !== outcome]
    panel = synth_panel(data, outcome, treatment, unit, time; covariates=covs)
    return synthetic_control(panel; predictors=predictors,
                             special_predictors=special_predictors, kwargs...)
end

function synthetic_control(panel::SynthPanel; predictors::Vector{Symbol}=Symbol[],
                           predictor_periods=nothing,
                           special_predictors=Pair{Symbol,Any}[], fit_periods=nothing,
                           v=nothing, v_method::Union{Nothing,Symbol}=nothing,
                           standardize::Union{Nothing,Bool}=nothing,
                           placebo::Bool=true, placebo_pool::Symbol=:donors,
                           v_starts::Integer=0, rng::AbstractRNG=Random.default_rng())
    has_pred = !isempty(predictors) || !isempty(special_predictors)
    vm = v_method === nothing ? (has_pred ? :optimize : :equal) : v_method
    vm in (:optimize, :equal) ||
        throw(ArgumentError("synthetic_control: v_method must be :optimize or :equal"))
    sp = Pair{Symbol,Vector{Any}}[Symbol(first(s)) => collect(Any, last(s))
                                  for s in special_predictors]
    spec = _sc_AdhSpec(collect(Symbol, predictors),
                       predictor_periods === nothing ? nothing :
                       collect(Any, predictor_periods), sp,
                       fit_periods === nothing ? nothing : collect(Any, fit_periods),
                       v === nothing ? nothing : collect(Float64, v), vm,
                       standardize === nothing ? has_pred : standardize, v_starts)
    placebo_pool in (:donors, :all) ||
        throw(ArgumentError("synthetic_control: placebo_pool must be :donors or :all"))
    return _sc_adh_estimate(panel, spec, placebo, rng, placebo_pool)
end

# ---------------------------------------------------------------------------------------
# Inference and robustness
# ---------------------------------------------------------------------------------------

"""
    synth_in_space_placebo(r::SyntheticControlEstimate; statistic=:rmspe_ratio,
                           pre_rmspe_cutoff=Inf) -> DiagnosticTest

In-space placebo (permutation) test of Abadie, Diamond and Hainmueller (2010): the
treated unit's post-treatment deviation is ranked among those obtained by applying the
synthetic control method to each donor in turn.

With a single treated unit and a few dozen donors, sampling-based inference is not
available. Abadie, Diamond and Hainmueller (2010) instead ask whether the effect
estimated for the treated unit is large relative to the "effects" obtained when the
intervention is reassigned to each untreated unit, for which the true effect is zero.
The default statistic is the ratio of post- to pre-treatment root mean squared
prediction error (RMSPE). It scales each unit's post-treatment gap by the quality of its
own pre-treatment fit, so that poorly fitted placebos do not dominate (Abadie, Diamond
& Hainmueller 2015). The p-value is
``(1 + \\#\\{\\text{placebos at least as extreme}\\}) / (1 + J)``, so with ``J`` placebos
the smallest attainable value is ``1/(J+1)``.

**When the test is exact.** The p-value is a Fisher randomization p-value for the sharp
null of no effect on any unit only if the intervention was assigned at random, or more
generally if treatment is exchangeable across units: before the intervention, every
unit in the pool was equally likely to be the treated one (Abadie 2021; Firpo &
Possebom 2018). In most applications the treated unit was not chosen at random. The
test then evaluates how unusual the treated unit's estimate is relative to a uniform
benchmark assignment distribution, which is a descriptive calibration rather than a
guarantee of size. Firpo and Possebom (2018) show how to assess sensitivity to
non-uniform assignment probabilities. The permutation is also not exactly symmetric
under the default `placebo_pool = :donors` of [`synthetic_control`](@ref), because
placebo units have one donor fewer than the treated unit.

The test concerns the sharp null of no effect for any unit. A rejection indicates an
unusually large deviation for the treated unit. A non-rejection with 20 to 40 donors is
weak evidence, because the attainable p-values are coarse. Excluding placebos with a
poor pre-treatment fit (`pre_rmspe_cutoff`) follows Abadie, Diamond and Hainmueller
(2010), but it changes the reference set and hence the p-value, so report the cutoff
used. For inference over time rather than across units, see
[`synth_conformal_inference`](@ref). Cattaneo, Feng and Titiunik (2021) develop
prediction intervals for synthetic control estimates, which are not implemented here.

# Arguments
- `r::SyntheticControlEstimate`: a result of [`synthetic_control`](@ref) fitted with
  `placebo = true`.

# Keywords
- `statistic::Symbol=:rmspe_ratio`: `:rmspe_ratio` (post/pre RMSPE ratio, one-sided) or
  `:att` (average post-treatment gap, two-sided).
- `pre_rmspe_cutoff::Real=Inf`: drop placebos whose pre-treatment RMSPE exceeds
  `pre_rmspe_cutoff` times the treated unit's. Abadie, Diamond and Hainmueller (2010)
  used cutoffs of 20, 5 and 2 on the MSPE scale; pass their square roots here.

# Returns
- `DiagnosticTest` whose statistic is the treated unit's statistic and whose p-value is
  the permutation p-value. `details` holds `units` (placebos used),
  `placebo_statistics`, `n_placebos` and `rank` (one plus the number of placebo
  statistics strictly greater than the treated unit's; for the one-sided RMSPE ratio,
  1 means the treated unit is the most extreme).

# Examples
```julia
using DrSnow, CSV, DataFrames
prop99 = CSV.read(pkgdir(DrSnow, "test", "validation", "synth", "prop99.csv"), DataFrame)
r = synthetic_control(prop99, :PacksPerCapita, :treated, :State, :Year)
t = synth_in_space_placebo(r)
t.pvalue, t.details.rank, t.details.n_placebos
synth_in_space_placebo(r; statistic=:att, pre_rmspe_cutoff=sqrt(5)).pvalue
```

# References
- Abadie, A., Diamond, A., & Hainmueller, J. (2010). Synthetic control methods for
  comparative case studies: Estimating the effect of California's tobacco control
  program. *Journal of the American Statistical Association*, 105(490), 493–505.
- Abadie, A., Diamond, A., & Hainmueller, J. (2015). Comparative politics and the
  synthetic control method. *American Journal of Political Science*, 59(2), 495–510.
- Abadie, A. (2021). Using synthetic controls: Feasibility, data requirements, and
  methodological aspects. *Journal of Economic Literature*, 59(2), 391–425.
- Firpo, S., & Possebom, V. (2018). Synthetic control method: Inference, sensitivity
  analysis and confidence sets. *Journal of Causal Inference*, 6(2), 20160026.
- Cattaneo, M. D., Feng, Y., & Titiunik, R. (2021). Prediction intervals for synthetic
  control methods. *Journal of the American Statistical Association*, 116(536),
  1865–1880.
"""
function synth_in_space_placebo(r::SyntheticControlEstimate; statistic::Symbol=:rmspe_ratio,
                                pre_rmspe_cutoff::Real=Inf)
    r.placebo === nothing &&
        throw(ArgumentError("synth_in_space_placebo: re-run synthetic_control with " *
                            "placebo = true"))
    pl = r.placebo
    keep = pl.pre_rmspe .<= pre_rmspe_cutoff * r.pre_rmspe
    any(keep) || throw(ArgumentError("synth_in_space_placebo: no placebo unit passes " *
                                     "the pre-RMSPE cutoff"))
    J = count(keep)
    if statistic === :rmspe_ratio
        r.pre_rmspe > 0 || throw(ArgumentError("synth_in_space_placebo: the treated " *
                                               "unit is fitted exactly before treatment; " *
                                               "the RMSPE ratio is undefined"))
        obs = r.post_rmspe / r.pre_rmspe
        draws = (pl.post_rmspe ./ pl.pre_rmspe)[keep]
        p = permutation_pvalue(obs, draws; alternative=:greater)
        name = "In-space placebo test (post/pre RMSPE ratio)"
    elseif statistic === :att
        obs = r.att
        draws = pl.att[keep]
        p = permutation_pvalue(obs, draws; alternative=:two_sided)
        name = "In-space placebo test (average post-treatment gap)"
    else
        throw(ArgumentError("statistic must be :rmspe_ratio or :att"))
    end
    note = "Permutation over $J donor units; the smallest attainable p-value is " *
           @sprintf("%.3g", 1 / (J + 1)) * ". The test treats the treated unit as " *
           "exchangeable with the donors."
    return DiagnosticTest(name, "no effect of the intervention on any unit",
                          obs, p; method="in-space permutation (Abadie et al. 2010)",
                          note=note,
                          details=(units=pl.units[keep], placebo_statistics=draws,
                                   n_placebos=J,
                                   rank=1 + count(>(obs), draws)))
end

"""
    synth_leave_one_out(r::SyntheticControlEstimate; tol=1e-6,
                        rng=Random.default_rng()) -> NamedTuple

Leave-one-out robustness check of a synthetic control estimate (Abadie, Diamond &
Hainmueller 2015): the synthetic control is re-estimated with each donor that has a
positive weight removed in turn.

Synthetic control weights are often sparse, so the counterfactual may rest on a handful
of donors. If one of them experienced an idiosyncratic shock after the intervention, or
was itself affected by it, the estimated effect would reflect that donor rather than the
intervention. Abadie, Diamond and Hainmueller (2015) therefore re-estimate the model
leaving out, one at a time, each donor with positive weight. The ``V`` search is re-run
unless ``V`` was supplied, and the rest of the specification is unchanged. Abadie
(2021) recommends reporting this analysis as a standard part of a synthetic control
study.

Read the results as a sensitivity analysis, not a test. Estimates that stay close to the
original, with a comparable pre-treatment fit, show that no single donor drives the
result. A large change points to a fragile counterfactual, and the post-treatment gaps
of that donor deserve scrutiny. Because the nested ``V`` optimisation can have several
local optima, part of the variation across leave-one-out fits may come from the
optimisation rather than from the donors themselves (Klößner, Kaul, Pfeifer & Schieler
2018). Compare the pre-treatment RMSPE of each fit with the original.

# Arguments
- `r::SyntheticControlEstimate`: result of [`synthetic_control`](@ref) with at least
  three donors.

# Keywords
- `tol::Real=1e-6`: donors with weight at most `tol` are not dropped.
- `rng::AbstractRNG=Random.default_rng()`: generator for the random starting points of
  the ``V`` search, used only when the original fit requested `v_starts > 0`.

# Returns
- `NamedTuple` with fields
  - `summary::DataFrame`: one row per dropped donor, with `dropped`, `att`, `pre_rmspe`
    and `post_rmspe`;
  - `gaps::DataFrame`: the gap paths, with `dropped`, `time` and `gap`.

# Examples
```julia
using DrSnow, CSV, DataFrames
prop99 = CSV.read(pkgdir(DrSnow, "test", "validation", "synth", "prop99.csv"), DataFrame)
r = synthetic_control(prop99, :PacksPerCapita, :treated, :State, :Year; placebo=false)
loo = synth_leave_one_out(r)
loo.summary
extrema(loo.summary.att), coef(r)
```

# References
- Abadie, A., Diamond, A., & Hainmueller, J. (2015). Comparative politics and the
  synthetic control method. *American Journal of Political Science*, 59(2), 495–510.
- Abadie, A. (2021). Using synthetic controls: Feasibility, data requirements, and
  methodological aspects. *Journal of Economic Literature*, 59(2), 391–425.
- Klößner, S., Kaul, A., Pfeifer, G., & Schieler, M. (2018). Comparative politics and
  the synthetic control method revisited: A note on Abadie et al. (2015). *Swiss
  Journal of Economics and Statistics*, 154(1), 11.
"""
function synth_leave_one_out(r::SyntheticControlEstimate; tol::Real=1e-6,
                             rng::AbstractRNG=Random.default_rng())
    p = r.panel
    drop = r.donor_rows[r.weights .> tol]
    length(r.donor_rows) >= 3 ||
        throw(ArgumentError("synth_leave_one_out: needs at least three donors"))
    summ = DataFrame(dropped=eltype(p.units)[], att=Float64[], pre_rmspe=Float64[],
                     post_rmspe=Float64[])
    gaps = DataFrame(dropped=eltype(p.units)[], time=eltype(p.times)[], gap=Float64[])
    T = size(p.Y, 2)
    for d in drop
        donors = [j for j in r.donor_rows if j != d]
        f = _sc_adh_fit(p, r.spec, r.treated_row, donors, r.n_pre, rng)
        g = f.treated .- f.synthetic
        pre, post = _sc_rmspe(g, r.n_pre)
        push!(summ, (p.units[d], mean(g[(r.n_pre + 1):T]), pre, post))
        append!(gaps, DataFrame(dropped=fill(p.units[d], T), time=p.times, gap=g))
    end
    return (summary=summ, gaps=gaps)
end

"""
    synth_in_time_placebo(r::SyntheticControlEstimate, placebo_time; placebo=false,
                          rng=Random.default_rng()) -> SyntheticControlEstimate

In-time placebo (backdating) check of a synthetic control estimate (Abadie, Diamond &
Hainmueller 2015): the intervention is reassigned to an earlier date, and the method is
re-applied using pre-treatment data only.

If the synthetic control method captures the treated unit's untreated trajectory, then
applying it with a fictitious intervention date ``t_p`` before the true one should give
post-``t_p`` gaps close to zero until the actual intervention. This function
re-estimates the synthetic control on the actual pre-treatment periods, treating
`placebo_time` as the start of treatment. Predictor, special-predictor and fit windows
are restricted to periods before `placebo_time`, and an error is raised if a window
becomes empty. The ``V`` search is re-run unless ``V`` was supplied. Abadie (2021)
recommends backdating as a way to check the predictive power of the synthetic control
and to detect anticipation effects, since the gap between the backdated synthetic
control and the treated unit should open only at the actual intervention date.

A large placebo "effect" signals a poor fit, anticipation of the intervention, or a
shock specific to the treated unit before treatment, and weakens the case for the
original estimate. A small one is consistent with a valid design, but it does not prove
validity: the backdated fit uses fewer periods and tests only the pre-treatment window.
The result is a full [`SyntheticControlEstimate`](@ref), so its gaps, weights and (with
`placebo = true`) permutation test can be examined like those of the original fit. Keep
enough periods before `placebo_time` for a meaningful fit.

# Arguments
- `r::SyntheticControlEstimate`: result of [`synthetic_control`](@ref).
- `placebo_time`: a time value strictly inside the pre-treatment period, with at least
  one earlier period.

# Keywords
- `placebo::Bool=false`: also run in-space placebos on the backdated design.
- `rng::AbstractRNG=Random.default_rng()`: generator for random starting points of the
  ``V`` search (used when the original fit requested `v_starts > 0`).

# Returns
- [`SyntheticControlEstimate`](@ref) for the backdated design, estimated on the periods
  before the actual intervention. Its `att` is the average placebo gap between
  `placebo_time` and the last pre-treatment period.

# Examples
```julia
using DrSnow, CSV, DataFrames
prop99 = CSV.read(pkgdir(DrSnow, "test", "validation", "synth", "prop99.csv"), DataFrame)
r = synthetic_control(prop99, :PacksPerCapita, :treated, :State, :Year; placebo=false)
backdated = synth_in_time_placebo(r, 1980)
coef(backdated), backdated.pre_rmspe
synth_gaps(backdated)
```

# References
- Abadie, A., Diamond, A., & Hainmueller, J. (2015). Comparative politics and the
  synthetic control method. *American Journal of Political Science*, 59(2), 495–510.
- Abadie, A. (2021). Using synthetic controls: Feasibility, data requirements, and
  methodological aspects. *Journal of Economic Literature*, 59(2), 391–425.
- Abadie, A., Diamond, A., & Hainmueller, J. (2010). Synthetic control methods for
  comparative case studies: Estimating the effect of California's tobacco control
  program. *Journal of the American Statistical Association*, 105(490), 493–505.
"""
function synth_in_time_placebo(r::SyntheticControlEstimate, placebo_time;
                               placebo::Bool=false, rng::AbstractRNG=Random.default_rng())
    p = r.panel
    tp = findfirst(==(placebo_time), p.times)
    (tp === nothing || tp < 2 || tp > r.n_pre) &&
        throw(ArgumentError("synth_in_time_placebo: placebo_time must be a pre-treatment " *
                            "period with at least one earlier period"))
    before(ts) = ts === nothing ? nothing :
                 Any[t for t in ts if findfirst(==(t), p.times) < tp]
    spec = r.spec
    sp = Pair{Symbol,Vector{Any}}[first(s) => before(last(s))
                                  for s in spec.special_predictors]
    for (i, s) in enumerate(sp)
        isempty(last(s)) &&
            throw(ArgumentError("synth_in_time_placebo: special predictor " *
                                "$(first(s)) has no periods before $placebo_time"))
    end
    pp = before(spec.predictor_periods)
    pp !== nothing && isempty(pp) &&
        throw(ArgumentError("synth_in_time_placebo: no predictor periods before " *
                            "$placebo_time"))
    fp = before(spec.fit_periods)
    fp !== nothing && isempty(fp) &&
        throw(ArgumentError("synth_in_time_placebo: no fit periods before $placebo_time"))
    newspec = _sc_AdhSpec(spec.predictors, pp, sp, fp, spec.v, spec.v_method,
                          spec.standardize, spec.v_starts)
    rows = vcat(r.donor_rows, r.treated_row)
    adoption = vcat(zeros(Int, length(r.donor_rows)), tp)
    sub = _sc_subpanel(p, rows, adoption; periods=1:r.n_pre)
    return _sc_adh_estimate(sub, newspec, placebo, rng)
end

function synth_weights(r::SyntheticControlEstimate)
    return DataFrame(unit=r.panel.units[r.donor_rows], weight=r.weights)
end

function synth_gaps(r::SyntheticControlEstimate)
    T = length(r.treated_path)
    return DataFrame(time=r.panel.times[1:T], treated=r.treated_path,
                     synthetic=r.synthetic_path,
                     gap=r.treated_path .- r.synthetic_path, post=(1:T) .> r.n_pre)
end

function Base.show(io::IO, ::MIME"text/plain", r::SyntheticControlEstimate)
    println(io, method_name(r), " — estimand: ", estimand(r))
    println(io, "  Treated unit: ", r.treated_unit, "; donors: ", length(r.donor_rows),
            "; treatment starts ", r.panel.times[r.n_pre + 1])
    @printf(io, "  ATT (average post-period gap): %.4f\n", r.att)
    @printf(io, "  RMSPE pre: %.4f, post: %.4f (ratio %.3f)\n", r.pre_rmspe,
            r.post_rmspe, r.pre_rmspe > 0 ? r.post_rmspe / r.pre_rmspe : Inf)
    if r.placebo !== nothing && r.pre_rmspe > 0
        t = synth_in_space_placebo(r)
        @printf(io, "  In-space placebo p-value (RMSPE ratio rank): %.4g (%d placebos)\n",
                t.pvalue, t.details.n_placebos)
        @printf(io, "  Placebo standard deviation of the ATT: %.4f\n", sqrt(vcov(r)[1]))
    end
    nz = count(>(1e-6), r.weights)
    println(io, "  Donors with positive weight: ", nz)
    ord = sortperm(r.weights; rev=true)
    for j in ord[1:min(nz, 5)]
        @printf(io, "    %-30s %.4f\n", string(r.panel.units[r.donor_rows[j]]),
                r.weights[j])
    end
end

Base.show(io::IO, r::SyntheticControlEstimate) =
    print(io, method_name(r), "(", @sprintf("%.4g", r.att), ")")
