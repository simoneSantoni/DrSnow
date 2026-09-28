# Regression estimators with spillover terms: two-way fixed-effects DiD with
# exposure (ring / hop / neighbour) terms split by own treatment (Butts 2021), a ring
# event study, and cross-sectional exposure regressions. All regressions are run by
# FixedEffectModels with formulas built programmatically; coefficients are read by
# name and non-identified terms raise errors.

"""
    SpilloverRegression <: CausalEstimate

Result of [`spillover_did`](@ref) or [`exposure_regression`](@ref): regression
estimates of direct and spillover effects under an exposure specification.

The coefficients, read by name from the fitted model, are:
- `"direct"`: effect of own treatment on treated units with no exposure (no other
  treated unit within the exposure neighbourhood), relative to unexposed untreated
  units;
- `"spill_control:<col>"`: spillover onto untreated units in exposure column
  `<col>` (e.g. a distance ring), relative to unexposed untreated units;
- `"spill_treated:<col>"`: *additional* effect on treated units that are also
  exposed, so that their total effect is `direct + spill_treated:<col>`;
- `"spill:<col>"` when spillover terms are pooled over own treatment
  (`split_by_treatment = false`);
- followed by the coefficients of any covariates.

# Fields
- `model`: the fitted `FixedEffectModel`.
- `coef::Vector{Float64}`, `vcov::Matrix{Float64}`, `names::Vector{String}`:
  coefficients, their full covariance under the chosen variance estimator, and
  names.
- `kind::Symbol`: `:did` or `:cross_section`.
- `details::NamedTuple`: exposure columns, variance estimator, rows dropped
  (undefined exposure or missing values), numbers of units (and periods).

# Accessors
- The [`CausalEstimate`](@ref) interface; `dof_residual` is that of the fitted model
  (``G - 1`` under one-way clustering).
"""
struct SpilloverRegression{M} <: CausalEstimate
    model::M
    coef::Vector{Float64}
    vcov::Matrix{Float64}
    names::Vector{String}
    kind::Symbol
    details::NamedTuple
end

StatsAPI.coef(r::SpilloverRegression) = r.coef
StatsAPI.vcov(r::SpilloverRegression) = r.vcov
StatsAPI.coefnames(r::SpilloverRegression) = r.names
StatsAPI.nobs(r::SpilloverRegression) = StatsAPI.nobs(r.model)
StatsAPI.dof_residual(r::SpilloverRegression) = StatsAPI.dof_residual(r.model)
estimand(r::SpilloverRegression) = r.kind === :did ?
    "direct and spillover effects under the exposure mapping (TWFE DiD)" :
    "direct and spillover effects under the exposure mapping (cross-section)"
method_name(r::SpilloverRegression) = r.kind === :did ?
    "Spillover DiD (two-way fixed effects)" : "Exposure regression"

function show_details(io::IO, r::SpilloverRegression)
    d = r.details
    println(io)
    println(io, "Exposure: ", d.exposure, "; variance: ", d.vcov)
    d.dropped > 0 && println(io, d.dropped, " row(s) dropped (undefined exposure or " *
                                            "missing values)")
end

_sv_default_vcov(unit, cluster, vcov) =
    vcov !== nothing ? vcov :
    cluster !== nothing ? Vcov.cluster(_as_symbols(cluster)...) :
    unit === nothing ? Vcov.robust() : Vcov.cluster(unit)

function _sv_check_new_columns(df, names)
    for n in names
        n in propertynames(df) &&
            throw(ArgumentError("data already has a column named `$n`, which the " *
                                "estimator needs to create; rename it"))
    end
end

# Add own-treatment and exposure regressors; returns their names.
function _sv_add_spill_terms!(work, D, E, cols, split)
    work[!, :direct] = D
    names = Symbol[:direct]
    for (k, c) in enumerate(cols)
        e = E[k]
        if split
            nc = Symbol("spill_control:" * c)
            nt = Symbol("spill_treated:" * c)
            work[!, nc] = [ismissing(x) ? missing : x * (1 - d) for (x, d) in zip(e, D)]
            work[!, nt] = [ismissing(x) ? missing : x * d for (x, d) in zip(e, D)]
            push!(names, nc, nt)
        else
            n = Symbol("spill:" * c)
            work[!, n] = e
            push!(names, n)
        end
    end
    return names
end

function _sv_fit_and_extract(work, outcome, regs, covariates, fes, vc, weights, context)
    f = make_formula(outcome, vcat(regs, covariates); fe=fes)
    m = weights === nothing ? FixedEffectModels.reg(work, f, vc) :
        FixedEffectModels.reg(work, f, vc; weights=weights)
    names = string.(vcat(regs, covariates))
    idx = Int[]
    for nm in names
        push!(idx, try
            coef_index(m, nm)
        catch err
            err isa ErrorException || rethrow()
            throw(ArgumentError("$context: coefficient `$nm` is not identified " *
                                "(collinear " *
                                "with the fixed effects or other regressors). " *
                                (startswith(nm, "spill") ?
                                 "No variation in this exposure term remains; use fewer " *
                                 "or wider rings/hops." :
                                 "Treatment must vary over time within units " *
                                 "(staggered or pre/post timing).")))
        end)
    end
    b = StatsAPI.coef(m)[idx]
    V = Matrix(StatsAPI.vcov(m)[idx, idx])
    return m, b, V, names
end

"""
    spillover_did(data, outcome, treatment, s; unit, time, exposure,
                  split_by_treatment=true, covariates=Symbol[], weights=nothing,
                  cluster=nothing, vcov=nothing) -> SpilloverRegression

Difference-in-differences with spillovers (Butts 2021): a two-way fixed-effects
regression that separates the direct effect of treatment from spillovers onto nearby
untreated and treated units.

When treatment effects cross unit boundaries, the conventional difference in
differences is biased in two ways: nearby control units are affected, so they no
longer identify the counterfactual trend, and treated units' outcomes reflect the
treatment of their neighbours as well as their own (Clarke 2017; Butts 2021). The
estimated model is
```math
Y_{it} = α_i + λ_t + τ D_{it} + \\sum_k δ^C_k E_{itk} (1 - D_{it})
         + \\sum_k δ^T_k E_{itk} D_{it} + X_{it}'β + ε_{it},
```
where ``D_{it}`` is the binary, time-varying treatment and ``E_{itk}`` are the
exposure columns computed from the period-``t`` treatments of the other units (by
default the nearest-treated-unit distance rings of [`RingExposure`](@ref)).
Untreated units beyond every ring (or with no treated neighbour) are the *clean
controls* that identify the counterfactual trend; ``τ`` is the direct effect on
treated units with no treated neighbour, ``δ^C_k`` the spillover onto untreated
units in ring ``k``, and ``δ^T_k`` the additional effect on treated units that are
also exposed.

Identification requires (i) parallel trends in untreated potential outcomes for all
exposure groups, (ii) that spillovers vanish beyond the outermost ring or exposure
neighbourhood, which is a maintained assumption that the data can only partly probe
(e.g. by adding an outer ring and checking that its coefficient is small), and (iii)
variation in treatment timing (pre/post or staggered). With staggered adoption and
heterogeneous or dynamic effects, TWFE coefficients are weighted averages of
group-time effects that can be biased, as in standard TWFE difference in differences
(de Chaisemartin and D'Haultfœuille 2020; Goodman-Bacon 2021); prefer a single
adoption date or inspect [`spillover_event_study`](@ref). Each coefficient is
identified by the units in its exposure cell; when a cell contains few units or
clusters, cluster-robust intervals undercover, so tabulate the cells with
[`compute_exposure`](@ref) before relying on them. If outcomes of nearby units
share shocks beyond the unit clusters, use [`ConleyVcov`](@ref) for the variance.

# Arguments
- `data`: panel with one row per unit and period; units are matched to `s` by the
  `unit` column, and the unit sets must coincide.
- `outcome::Symbol`, `treatment::Symbol`: outcome and binary treatment columns.
- `s::InterferenceStructure`: the structure.

# Keywords
- `unit::Symbol`, `time::Symbol`: unit and period columns.
- `exposure::ExposureSpec` (required): e.g. `RingExposure([10, 20, 30])` (spatial;
  the outermost radius is the maximal spillover distance) or
  `NeighborExposure(:any)` / `HopExposure(2)` (networks).
- `split_by_treatment::Bool`: separate spillover terms for untreated and treated
  units (default `true`); otherwise one pooled term per exposure column.
- `covariates::Vector{Symbol}`: time-varying controls.
- `weights::Union{Nothing,Symbol}`: regression weights.
- `cluster`: clustering column(s); default `unit`.
- `vcov`: any `FixedEffectModels` covariance estimator, e.g.
  `Vcov.cluster(:state)`, [`ConleyVcov`](@ref) or [`NetworkHACVcov`](@ref); takes
  precedence over `cluster`.

# Returns
- [`SpilloverRegression`](@ref).

# Examples
```julia
using DrSnow, DataFrames, StableRNGs
rng = StableRNG(5)
n, T = 300, 6
s = SpatialStructure(1:n; x=100 .* rand(rng, n), y=100 .* rand(rng, n))
ever = rand(rng, n) .< 0.1
panel = DataFrame(id=repeat(1:n; inner=T), year=repeat(1:T; outer=n))
panel.d = Int.(ever[panel.id] .& (panel.year .>= 4))
rings = RingExposure([5.0, 10.0])
e = compute_exposure(panel, :d, s, rings; unit=:id, time=:year)
panel.y = 0.1 .* panel.year .+ 1.0 .* panel.d .+ 0.5 .* e.ring_0_5 .+
          randn(rng, nrow(panel))
r = spillover_did(panel, :y, :d, s; unit=:id, time=:year, exposure=rings)
confint(r)
```

# References
- Butts, K. (2021). Difference-in-differences estimation with spatial spillovers.
  arXiv:2105.03737.
- Clarke, D. (2017). Estimating difference-in-differences in the presence of
  spillovers. MPRA Paper No. 81604.
- de Chaisemartin, C., & D'Haultfœuille, X. (2020). Two-way fixed effects estimators
  with heterogeneous treatment effects. *American Economic Review*, 110(9),
  2964–2996.
- Goodman-Bacon, A. (2021). Difference-in-differences with variation in treatment
  timing. *Journal of Econometrics*, 225(2), 254–277.
"""
function spillover_did(data, outcome::Symbol, treatment::Symbol, s::InterferenceStructure;
                       unit::Symbol, time::Symbol, exposure::ExposureSpec,
                       split_by_treatment::Bool=true, covariates::Vector{Symbol}=Symbol[],
                       weights::Union{Nothing,Symbol}=nothing, cluster=nothing,
                       vcov=nothing)
    context = "spillover_did"
    require_columns(data, [outcome, treatment, unit, time, weights, covariates...];
                    context=context)
    work = DataFrame(data; copycols=false)
    p = _sv_prepare(s, exposure)
    rows, Em = _sv_panel_exposure(work, treatment, s, p; unit=unit, time=time,
                                  context=context)
    any(ismissing, work[!, treatment]) &&
        throw(ArgumentError("$context: treatment has missing values"))
    D = Float64[_sv_binary_value(v, context) for v in work[!, treatment]]
    E = [[_sv_nan_to_missing(e[u, t]) for (u, t) in rows] for e in Em]
    _sv_check_did_support(D, E, rows, context)
    newcols = vcat(:direct, split_by_treatment ?
                   vcat([[Symbol("spill_control:" * c), Symbol("spill_treated:" * c)]
                         for c in p.names]...) :
                   [Symbol("spill:" * c) for c in p.names])
    _sv_check_new_columns(work, newcols)
    regs = _sv_add_spill_terms!(work, D, E, p.names, split_by_treatment)
    vc = _sv_default_vcov(unit, cluster, vcov)
    m, b, V, names = _sv_fit_and_extract(work, outcome, regs, covariates, [unit, time],
                                         vc, weights, context)
    details = (exposure=join(p.names, ", "), vcov=string(vc),
               dropped=nrow(work) - StatsAPI.nobs(m),
               n_units=length(unique(first.(rows))), n_periods=maximum(last.(rows)),
               split_by_treatment=split_by_treatment)
    return SpilloverRegression(m, b, V, names, :did, details)
end

function _sv_check_did_support(D, E, rows, context)
    clean = [D[r] == 0 && all(e -> !ismissing(e[r]) && e[r] == 0, E)
             for r in eachindex(D)]
    any(clean) || throw(ArgumentError("$context: no clean control observations " *
                                      "(untreated and unexposed); spillovers cannot be " *
                                      "separated from the counterfactual trend"))
    byunit = Dict{Int,Vector{Float64}}()
    for (r, (u, _)) in enumerate(rows)
        push!(get!(byunit, u, Float64[]), D[r])
    end
    any(v -> length(unique(v)) > 1, values(byunit)) ||
        throw(ArgumentError("$context: treatment never changes within a unit, so the " *
                            "direct effect is absorbed by the unit fixed effects; a " *
                            "pre/post or staggered design is required"))
    return nothing
end

"""
    exposure_regression(data, outcome, treatment, s; unit,
                        exposure=NeighborExposure(:share), split_by_treatment=true,
                        covariates=Symbol[], weights=nothing, fe=Symbol[],
                        vcov=Vcov.robust()) -> SpilloverRegression

Cross-sectional regression of the outcome on own treatment and exposure terms,
```math
Y_i = α + τ Z_i + \\sum_k γ^C_k E_{ik} (1 - Z_i) + \\sum_k γ^T_k E_{ik} Z_i
      + X_i'β + ε_i,
```
with the exposures computed from the structure and the observed treatments (one
row per unit of `s`).

This is the regression counterpart of the design-based estimator
[`exposure_effects`](@ref). Its coefficients are linear-projection contrasts; they
have a causal interpretation as average direct and spillover effects only if the
exposure mapping is correctly specified, effects are homogeneous or the projection
is saturated, and exposure is as good as randomly assigned *given the regressors*.
The last condition is not implied by randomization of ``Z``: even under complete
randomization, the probability of being exposed depends on the number of neighbours
(degree, or the number of units within the radius), and if that number is related to
the outcome, an unweighted regression confounds the spillover with it (Aronow and
Samii 2017). Controlling for the determinants of exposure probabilities (for example
degree, or fixed effects for them) or using the inverse-probability-weighted
[`exposure_effects`](@ref) addresses this.

Outcomes of connected or nearby units are typically correlated, so use a
dependence-robust variance such as [`NetworkHACVcov`](@ref) or
[`ConleyVcov`](@ref) rather than the default heteroskedasticity-robust one when
dependence is plausible.

# Arguments
- `data`: table with one row per unit of `s`, matched by `unit`.
- `outcome::Symbol`, `treatment::Symbol`: outcome and binary treatment columns.
- `s::InterferenceStructure`: the structure.

# Keywords
- `unit::Symbol`: unit-identifier column.
- `exposure::ExposureSpec`: default `NeighborExposure(:share)`.
- `split_by_treatment::Bool`: separate exposure terms for untreated and treated
  units (default `true`).
- `covariates::Vector{Symbol}`: controls, e.g. degree.
- `weights::Union{Nothing,Symbol}`: regression weights.
- `fe::Vector{Symbol}`: absorbed fixed effects (e.g. strata or villages).
- `vcov`: covariance estimator; default `Vcov.robust()`.

# Returns
- [`SpilloverRegression`](@ref).

# Examples
```julia
using DrSnow, DataFrames, StableRNGs
rng = StableRNG(6)
n = 150
A = zeros(Int, n, n)
for i in 1:n, j in (i + 1):n
    rand(rng) < 0.03 && (A[i, j] = A[j, i] = 1)
end
g = NetworkStructure(1:n, A)
z = draw_assignment(rng, CompleteRandomization(n, 50))
spec = NeighborExposure(:share; isolates=:zero)
sh = compute_exposure(g, z, spec).share
df = DataFrame(id=1:n, z=Int.(z), degree=vec(sum(A; dims=2)))
df.y = 1.0 .* z .+ 0.8 .* sh .+ randn(rng, n)
r = exposure_regression(df, :y, :z, g; unit=:id, exposure=spec, covariates=[:degree],
                        vcov=NetworkHACVcov(g; unit=:id, bandwidth=2))
coeftable(r)
```

# References
- Aronow, P. M., & Samii, C. (2017). Estimating average causal effects under general
  interference, with application to a social network experiment. *Annals of Applied
  Statistics*, 11(4), 1912–1947.
- Leung, M. P. (2022). Causal inference under approximate neighborhood interference.
  *Econometrica*, 90(1), 267–293.
- Kojevnikov, D., Marmer, V., & Song, K. (2021). Limit theorems for network
  dependent random variables. *Journal of Econometrics*, 222(2), 882–908.
"""
function exposure_regression(data, outcome::Symbol, treatment::Symbol,
                             s::InterferenceStructure; unit::Symbol,
                             exposure::ExposureSpec=NeighborExposure(:share),
                             split_by_treatment::Bool=true,
                             covariates::Vector{Symbol}=Symbol[],
                             weights::Union{Nothing,Symbol}=nothing,
                             fe::Vector{Symbol}=Symbol[], vcov=Vcov.robust())
    context = "exposure_regression"
    require_columns(data, [outcome, treatment, unit, weights, covariates..., fe...];
                    context=context)
    work = DataFrame(data; copycols=false)
    p = _sv_prepare(s, exposure)
    rows, Em = _sv_panel_exposure(work, treatment, s, p; unit=unit, time=nothing,
                                  context=context)
    any(ismissing, work[!, treatment]) &&
        throw(ArgumentError("$context: treatment has missing values"))
    D = Float64[_sv_binary_value(v, context) for v in work[!, treatment]]
    E = [[_sv_nan_to_missing(e[u, 1]) for (u, _) in rows] for e in Em]
    newcols = vcat(:direct, split_by_treatment ?
                   vcat([[Symbol("spill_control:" * c), Symbol("spill_treated:" * c)]
                         for c in p.names]...) :
                   [Symbol("spill:" * c) for c in p.names])
    _sv_check_new_columns(work, newcols)
    regs = _sv_add_spill_terms!(work, D, E, p.names, split_by_treatment)
    m, b, V, names = _sv_fit_and_extract(work, outcome, regs, covariates, fe, vcov,
                                         weights, context)
    details = (exposure=join(p.names, ", "), vcov=string(vcov),
               dropped=nrow(work) - StatsAPI.nobs(m), n_units=nrow(work),
               split_by_treatment=split_by_treatment)
    return SpilloverRegression(m, b, V, names, :cross_section, details)
end

# ---------------------------------------------------------------------------------
# Ring / exposure event study
# ---------------------------------------------------------------------------------

"""
    SpilloverEventStudy <: CausalEstimate

Result of [`spillover_event_study`](@ref): dynamic direct and spillover effects by
event time.

Coefficients are named `"<group>:<event time>"`, where `group` is `treated` or
`exposed:<col>`; the binned end points are `"<group>:≤-L"` and `"<group>:≥K"`, and
event time ``-1`` is the omitted reference period of every group. Group-event-time
cells that do not occur in the sample are omitted and listed in
`details.omitted_cells`.

# Fields
- `model`: the fitted `FixedEffectModel`.
- `coef::Vector{Float64}`, `vcov::Matrix{Float64}`, `names::Vector{String}`:
  coefficients, full covariance and names.
- `groups::Vector{String}`, `event_times::Vector{Int}`: group and event time of
  each event-time coefficient.
- `table::DataFrame`: group, event time, whether the event time is a binned end
  point, estimate, standard error and pointwise confidence interval.
- `leads::Int`, `lags::Int`: event window.
- `details::NamedTuple`: units per group, number of clean controls, units dropped
  for undefined exposure, variance estimator and omitted cells.

# Accessors
- The [`CausalEstimate`](@ref) interface; [`spillover_pretrend_test`](@ref).
"""
struct SpilloverEventStudy{M} <: CausalEstimate
    model::M
    coef::Vector{Float64}
    vcov::Matrix{Float64}
    names::Vector{String}
    groups::Vector{String}
    event_times::Vector{Int}
    table::DataFrame
    leads::Int
    lags::Int
    details::NamedTuple
end

StatsAPI.coef(r::SpilloverEventStudy) = r.coef
StatsAPI.vcov(r::SpilloverEventStudy) = r.vcov
StatsAPI.coefnames(r::SpilloverEventStudy) = r.names
StatsAPI.nobs(r::SpilloverEventStudy) = StatsAPI.nobs(r.model)
StatsAPI.dof_residual(r::SpilloverEventStudy) = StatsAPI.dof_residual(r.model)
estimand(::SpilloverEventStudy) = "dynamic direct and spillover effects by event time"
method_name(::SpilloverEventStudy) = "Spillover event study (two-way fixed effects)"

function show_details(io::IO, r::SpilloverEventStudy)
    println(io)
    for (g, n) in pairs(r.details.group_units)
        println(io, "  ", g, ": ", n, " unit(s)")
    end
    println(io, "  clean controls: ", r.details.clean_controls, " unit(s)")
end

"""
    spillover_event_study(data, outcome, treatment, s; unit, time, exposure,
                          leads=3, lags=3, covariates=Symbol[], weights=nothing,
                          cluster=nothing, vcov=nothing, level=0.95)
        -> SpilloverEventStudy

Event study with spillover groups, the "ring event study" of Butts (2021): dynamic
effects of treatment on treated units and of exposure on never-treated units near
treated ones, relative to clean controls.

Treatment must be absorbing (once treated, always treated). Units are classified as
*treated* (ever treated; event time measured from own adoption), *exposed:<col>*
(never treated but exposed at some period; event time measured from the first
period of exposure, and `<col>` is the first exposure column active then, e.g. the
ring the unit falls in when first exposed), or *clean controls* (never treated and
never exposed). Event-time indicators for ``-L, …, K`` (reference ``-1``) are
interacted with the group, with the end points binned (``≤ -L``, ``≥ K``) so that no
out-of-window period is pooled into the reference, and unit and period fixed effects
are included. Units whose exposure is undefined in some period are dropped.

The coefficients are TWFE event-study estimates. They identify the dynamic direct
and spillover effects under parallel trends for every group relative to the clean
controls and no spillovers beyond the exposure neighbourhood, and they share the
known problems of TWFE event studies under staggered adoption with heterogeneous
effects, where coefficients can be contaminated by effects from other periods (Sun
and Abraham 2021). Leads (event times ``≤ -2``) can be tested jointly with
[`spillover_pretrend_test`](@ref); such pre-tests have limited power, and
conditioning the analysis on passing them distorts inference (Roth 2022).

# Arguments
- `data`, `outcome`, `treatment`, `s`: as in [`spillover_did`](@ref); the treatment
  must be absorbing.

# Keywords
- `unit`, `time`, `exposure`, `covariates`, `weights`, `cluster`, `vcov`: as in
  [`spillover_did`](@ref).
- `leads::Integer`: number of pre-periods ``L ≥ 2`` (``-L`` is binned); default 3.
- `lags::Integer`: number of post-periods ``K ≥ 0`` (``K`` is binned); default 3.
- `level::Real`: confidence level of the pointwise intervals in `table`; default
  0.95.

# Returns
- [`SpilloverEventStudy`](@ref).

# Examples
```julia
using DrSnow, DataFrames, StableRNGs
rng = StableRNG(7)
n, T = 150, 8
s = SpatialStructure(1:n; x=100 .* rand(rng, n), y=100 .* rand(rng, n))
ever = rand(rng, n) .< 0.2
panel = DataFrame(id=repeat(1:n; inner=T), year=repeat(1:T; outer=n))
panel.d = Int.(ever[panel.id] .& (panel.year .>= 5))
rings = RingExposure([15.0])
e = compute_exposure(panel, :d, s, rings; unit=:id, time=:year)
panel.y = 1.0 .* panel.d .+ 0.5 .* e.ring_0_15 .+ randn(rng, nrow(panel))
es = spillover_event_study(panel, :y, :d, s; unit=:id, time=:year, exposure=rings,
                           leads=3, lags=2)
es.table
```

# References
- Butts, K. (2021). Difference-in-differences estimation with spatial spillovers.
  arXiv:2105.03737.
- Sun, L., & Abraham, S. (2021). Estimating dynamic treatment effects in event
  studies with heterogeneous treatment effects. *Journal of Econometrics*, 225(2),
  175–199.
- Roth, J. (2022). Pretest with caution: Event-study estimates after testing for
  parallel trends. *American Economic Review: Insights*, 4(3), 305–322.
"""
function spillover_event_study(data, outcome::Symbol, treatment::Symbol,
                               s::InterferenceStructure; unit::Symbol, time::Symbol,
                               exposure::ExposureSpec, leads::Integer=3, lags::Integer=3,
                               covariates::Vector{Symbol}=Symbol[],
                               weights::Union{Nothing,Symbol}=nothing, cluster=nothing,
                               vcov=nothing, level::Real=0.95)
    context = "spillover_event_study"
    leads >= 2 || throw(ArgumentError("$context: leads must be ≥ 2 (event time -1 is " *
                                      "the reference)"))
    lags >= 0 || throw(ArgumentError("$context: lags must be ≥ 0"))
    require_columns(data, [outcome, treatment, unit, time, weights, covariates...];
                    context=context)
    work = DataFrame(data; copycols=false)
    p = _sv_prepare(s, exposure)
    rows, Em = _sv_panel_exposure(work, treatment, s, p; unit=unit, time=time,
                                  context=context)
    any(ismissing, work[!, treatment]) &&
        throw(ArgumentError("$context: treatment has missing values"))
    D = Float64[_sv_binary_value(v, context) for v in work[!, treatment]]
    N = n_units(s)
    T = maximum(last.(rows))
    Dm = fill(NaN, N, T)
    for (r, (u, t)) in enumerate(rows)
        Dm[u, t] = D[r]
    end
    G = fill(typemax(Int), N)                 # adoption period (index)
    for u in 1:N
        prev = 0.0
        for t in 1:T
            isnan(Dm[u, t]) && continue
            Dm[u, t] < prev && throw(ArgumentError("$context: treatment of unit " *
                                                   "$(repr(s.ids[u])) switches off; the " *
                                                   "event study requires absorbing " *
                                                   "treatment"))
            prev = Dm[u, t]
            (Dm[u, t] == 1 && G[u] == typemax(Int)) && (G[u] = t)
        end
    end
    any(<(typemax(Int)), G) || throw(ArgumentError("$context: no unit is ever treated"))
    K = length(p.names)
    F = fill(typemax(Int), N)                 # first exposure period (never treated)
    ring = zeros(Int, N)
    undefined = falses(N)
    observed = [falses(T) for _ in 1:N]
    for (u, t) in rows
        observed[u][t] = true
    end
    for u in 1:N
        G[u] == typemax(Int) || continue
        for t in 1:T
            observed[u][t] || continue
            vals = [Em[k][u, t] for k in 1:K]
            if any(isnan, vals)
                undefined[u] = true
                break
            end
            k = findfirst(>(0), vals)
            if k !== nothing && F[u] == typemax(Int)
                F[u] = t
                ring[u] = k
            end
        end
    end
    groups = vcat("treated", ["exposed:" * c for c in p.names])
    event_times = [e for e in -leads:lags if e != -1]
    label(e) = e == -leads ? "≤" * string(-leads) : e == lags ? "≥" * string(lags) :
               string(e)
    names = [Symbol(g * ":" * label(e)) for g in groups for e in event_times]
    _sv_check_new_columns(work, names)
    keep = [!undefined[u] for (u, _) in rows]
    for nm in names
        work[!, nm] = zeros(nrow(work))
    end
    for (r, (u, t)) in enumerate(rows)
        keep[r] || continue
        gi, start = G[u] < typemax(Int) ? (1, G[u]) :
                    F[u] < typemax(Int) ? (1 + ring[u], F[u]) : (0, 0)
        gi == 0 && continue
        e = clamp(t - start, -leads, lags)
        e == -1 && continue
        work[r, Symbol(groups[gi] * ":" * label(e))] = 1.0
    end
    sub = work[keep, :]
    # keep (group, event time) cells that occur in the sample
    cells = [(g, e, Symbol(g * ":" * label(e))) for g in groups for e in event_times]
    cells = [c for c in cells if any(!=(0), sub[!, c[3]])]
    regs = [c[3] for c in cells]
    count(u -> G[u] == typemax(Int) && F[u] == typemax(Int) && !undefined[u], 1:N) > 0 ||
        throw(ArgumentError("$context: no clean control units (never treated and never " *
                            "exposed)"))
    vc = _sv_default_vcov(unit, cluster, vcov)
    m, b, V, cn = _sv_fit_and_extract(sub, outcome, regs, covariates, [unit, time], vc,
                                      weights, context)
    nreg = length(regs)
    se = sqrt.(max.(diag(V)[1:nreg], 0.0))
    crit = critical_value(level, StatsAPI.dof_residual(m))
    gcol = [c[1] for c in cells]
    ecol = [c[2] for c in cells]
    table = DataFrame(group=gcol, event_time=ecol, binned=[e in (-leads, lags) for e in
                                                          ecol],
                      estimate=b[1:nreg], std_error=se, lower=b[1:nreg] .- crit .* se,
                      upper=b[1:nreg] .+ crit .* se)
    group_units = Dict{String,Int}("treated" => count(<(typemax(Int)), G))
    for (k, c) in enumerate(p.names)
        n = count(u -> G[u] == typemax(Int) && ring[u] == k && !undefined[u], 1:N)
        n > 0 && (group_units["exposed:" * c] = n)
    end
    details = (group_units=group_units,
               clean_controls=count(u -> G[u] == typemax(Int) &&
                                         F[u] == typemax(Int) && !undefined[u], 1:N),
               dropped_units=count(undefined), vcov=string(vc),
               omitted_cells=setdiff(names, regs))
    return SpilloverEventStudy(m, b, V, cn, gcol, ecol, table, Int(leads), Int(lags),
                               details)
end

"""
    spillover_pretrend_test(es::SpilloverEventStudy; group=nothing) -> DiagnosticTest

Joint Wald test that all lead coefficients (event times ``≤ -2``) of a spillover
event study are zero, for one group or for all groups together.

The test uses [`wald_test`](@ref) with the full covariance matrix of the leads and,
when the model has finite residual degrees of freedom (e.g. ``G - 1`` under
clustering), its F form. Zero leads are an implication of parallel trends (and of no
anticipation) for the group; the test cannot establish parallel trends, has low
power against smooth or post-treatment violations, and conditioning estimation on
passing it distorts subsequent inference (Roth 2022).

# Arguments
- `es::SpilloverEventStudy`: an event-study result.

# Keywords
- `group`: group name (e.g. `"treated"` or `"exposed:ring_0_10"`), or `nothing`
  (default) to test the leads of all groups jointly.

# Returns
- [`DiagnosticTest`](@ref) (F test when the model has finite residual degrees of
  freedom, χ² otherwise); `details.coefficients` lists the leads tested.

# Examples
```julia
using DrSnow, DataFrames, StableRNGs
rng = StableRNG(7)
n, T = 150, 8
s = SpatialStructure(1:n; x=100 .* rand(rng, n), y=100 .* rand(rng, n))
ever = rand(rng, n) .< 0.2
panel = DataFrame(id=repeat(1:n; inner=T), year=repeat(1:T; outer=n))
panel.d = Int.(ever[panel.id] .& (panel.year .>= 5))
panel.y = 1.0 .* panel.d .+ randn(rng, nrow(panel))
es = spillover_event_study(panel, :y, :d, s; unit=:id, time=:year,
                           exposure=RingExposure([15.0]), leads=3, lags=2)
spillover_pretrend_test(es; group="treated")
```

# References
- Wald, A. (1943). Tests of statistical hypotheses concerning several parameters
  when the number of observations is large. *Transactions of the American
  Mathematical Society*, 54(3), 426–482.
- Roth, J. (2022). Pretest with caution: Event-study estimates after testing for
  parallel trends. *American Economic Review: Insights*, 4(3), 305–322.
"""
function spillover_pretrend_test(es::SpilloverEventStudy; group=nothing)
    idx = [k for k in eachindex(es.groups)
           if es.event_times[k] <= -2 && (group === nothing || es.groups[k] == group)]
    isempty(idx) && throw(ArgumentError("no lead coefficients for group $(repr(group))"))
    w = wald_test(es.coef[idx], es.vcov[idx, idx]; dof=StatsAPI.dof_residual(es))
    return DiagnosticTest("Pre-trend test (spillover event study)",
        "all lead coefficients are zero" *
        (group === nothing ? "" : " for group $(group)"),
        w.statistic, w.pvalue; dof=(w.dof1, w.dof2),
        method=isfinite(w.dof2) ? "Wald F test, full covariance" :
               "Wald χ² test, full covariance",
        note="Low power against smooth violations of parallel trends; a " *
             "non-rejection does not establish parallel trends.",
        details=(coefficients=es.names[idx],))
end
