# Callaway & Sant'Anna (2021) group-time average treatment effects and their
# aggregations. The implementation mirrors R's `did` (v2.5) for balanced panels:
# the 2×2 kernels are the Sant'Anna–Zhao estimators in drdid.jl, influence functions
# are scaled so that estimate - truth ≈ mean(ψ) over units, and analytic standard
# errors are sqrt(Σ_c S_c² ) / n (unit-level clusters by default).

"""
    CallawaySantAnnaEstimate <: CausalEstimate

Group-time average treatment effects ``ATT(g,t)`` of Callaway and Sant'Anna (2021),
returned by [`did_callaway_santanna`](@ref).

``ATT(g,t) = E[Y_t(g) - Y_t(\\infty) \\mid G = g]`` is the average effect in period
``t`` for the cohort first treated in period ``g``. The object stores every estimated
cell with its base (comparison) period, the full covariance matrix, and the
unit-level influence functions from which [`aggregate_att`](@ref) builds summary
parameters and their standard errors, and from which multiplier-bootstrap sup-t
bands are computed. Cells with ``t < g`` are placebo (pre-treatment) estimates.

# Fields
- `groups::Vector{Int}`, `times::Vector{Int}`, `bases::Vector{Int}`: cohort, period
  and base period of each ``ATT(g,t)``, as **period indices** into `periods`.
- `coef::Vector{Float64}`, `vcov::Matrix{Float64}`: estimates and full covariance.
- `influence::Matrix{Float64}`: influence functions (units × cells), scaled so that
  `estimate - truth ≈ mean(influence[:, k])`.
- `periods::Vector`: time labels.
- `unit_cohort::Vector{Int}`, `unit_weight::Vector{Float64}`,
  `unit_cluster::Union{Nothing,Vector}`: per-unit cohort (period index, `0` = never
  treated), sampling weight and cluster, used by the aggregations (per observation
  for repeated cross-sections).
- `n_clusters::Int`, `nobs::Int`: clusters and observations.
- `supt_draws::Vector{Float64}`: multiplier-bootstrap sup-t draws over all cells
  (empty when `bootstrap = false`).
- `settings::NamedTuple`: `control_group`, `anticipation`, `base_period`, `method`,
  `covariates`, `bootstrap`, `biters`, and sample information.

# Accessors
`coef`, `vcov`, `stderror`, `coefnames` (`"ATT(g,t)"` labels in time units),
`coeftable`, `confint(cs; level, uniform)`, `nobs`, [`estimate`](@ref) (the simple
aggregated ATT), [`pre_trend_test`](@ref) and [`aggregate_att`](@ref).

# References
- Callaway, B., & Sant'Anna, P. H. C. (2021). Difference-in-differences with multiple
  time periods. *Journal of Econometrics*, 225(2), 200–230.
"""
struct CallawaySantAnnaEstimate <: CausalEstimate
    groups::Vector{Int}
    times::Vector{Int}
    bases::Vector{Int}
    coef::Vector{Float64}
    vcov::Matrix{Float64}
    influence::Matrix{Float64}
    periods::Vector
    unit_cohort::Vector{Int}
    unit_weight::Vector{Float64}
    unit_cluster::Union{Nothing,Vector}
    n_clusters::Int
    nobs::Int
    supt_draws::Vector{Float64}
    settings::NamedTuple
end

StatsAPI.coef(r::CallawaySantAnnaEstimate) = r.coef
StatsAPI.vcov(r::CallawaySantAnnaEstimate) = r.vcov
StatsAPI.nobs(r::CallawaySantAnnaEstimate) = r.nobs
StatsAPI.coefnames(r::CallawaySantAnnaEstimate) =
    ["ATT(g=$(r.periods[g]), t=$(r.periods[t]))" for (g, t) in zip(r.groups, r.times)]
estimand(::CallawaySantAnnaEstimate) = "ATT(g,t): effect in period t for cohort g"
method_name(r::CallawaySantAnnaEstimate) =
    "Callaway–Sant'Anna ($(r.settings.method), $(r.settings.control_group) controls, " *
    "$(r.settings.base_period) base period)"

"""
    estimate(cs::CallawaySantAnnaEstimate) -> Float64

The simple aggregated ATT of a Callaway–Sant'Anna estimate, the average of all
post-treatment ``ATT(g,t)`` weighted by cohort size.

# Arguments
- `cs::CallawaySantAnnaEstimate`: group-time effects from
  [`did_callaway_santanna`](@ref).

# Returns
- `Float64`: `estimate(aggregate_att(cs, :simple))`; use
  [`aggregate_att`](@ref) for its standard error.
"""
estimate(r::CallawaySantAnnaEstimate) = estimate(aggregate_att(r, :simple;
                                                               bootstrap=false))

function show_details(io::IO, r::CallawaySantAnnaEstimate)
    println(io)
    what = get(r.settings, :panel, true) ? "Units" : "Observations"
    println(io, what, ": ", length(r.unit_cohort), "; never treated: ",
            count(==(0), r.unit_cohort), "; anticipation: ", r.settings.anticipation)
    isempty(r.settings.covariates) ||
        println(io, "Covariates: ", join(r.settings.covariates, ", "))
    r.n_clusters > 0 && println(io, "Clusters: ", r.n_clusters)
    println(io, "Use aggregate_att(cs, :dynamic / :group / :calendar / :simple) to ",
            "summarize; pre_trend_test(cs) tests pre-treatment ATT(g,t).")
end

"""
    confint(cs::CallawaySantAnnaEstimate; level=0.95, uniform=false,
            rng=Random.default_rng(), ndraws=10_000) -> Matrix{Float64}

Pointwise confidence intervals for all group-time effects, or simultaneous bands
that cover every ``ATT(g,t)`` jointly.

Pointwise intervals use normal critical values and the influence-function standard
errors. With `uniform = true` the critical value is the `level` quantile of the
multiplier-bootstrap sup-t statistic stored in `cs` (Callaway and Sant'Anna, 2021,
Section 4.1), or, when the estimate was computed with `bootstrap = false`, of
`ndraws` simulated draws from the estimated covariance (Montiel Olea and
Plagborg-Møller, 2019). Simultaneous bands account for the many cells being
examined at once and are the appropriate basis for statements about the whole set
of effects.

# Arguments
- `cs::CallawaySantAnnaEstimate`: group-time effects.

# Keywords
- `level::Real = 0.95`: confidence level.
- `uniform::Bool = false`: return simultaneous sup-t bands.
- `rng::AbstractRNG = Random.default_rng()`, `ndraws::Integer = 10_000`: generator
  and number of draws for a simulated critical value.

# Returns
- `Matrix{Float64}`: lower and upper bounds in columns 1 and 2, one row per cell.
"""
function StatsAPI.confint(r::CallawaySantAnnaEstimate; level::Real=0.95,
                          uniform::Bool=false, rng::AbstractRNG=Random.default_rng(),
                          ndraws::Integer=10_000)
    uniform || return _did_pointwise_ci(r, level)
    c = _uniform_critical_value(r.supt_draws, r.vcov, Inf, level, rng, ndraws,
                                    eachindex(r.coef))
    return _did_ci_with(r.coef, StatsAPI.stderror(r), c)
end

"""
    pre_trend_test(cs::CallawaySantAnnaEstimate) -> DiagnosticTest

Joint Wald (``\\chi^2``) test that all pre-treatment group-time effects
``ATT(g,t)``, ``t < g``, are zero, with the full influence-function covariance
matrix (the pre-test reported by R's `did`).

With the default varying base period, each pre-treatment cell compares period ``t``
with period ``t - 1``, so the test asks whether consecutive pre-treatment changes of
each cohort match those of its comparison group; with `base_period = :universal` all
cells are measured against period ``g - 1 -`` anticipation. The caveats of
[`pre_trend_test`](@ref) apply: a non-rejection does not establish parallel trends,
such tests may have low power, and conditioning on passing them distorts inference
(Roth, 2022).

# Arguments
- `cs::CallawaySantAnnaEstimate`: group-time effects with at least one
  pre-treatment cell.

# Returns
- `DiagnosticTest`: statistic, p-value and degrees of freedom.

# References
- Callaway, B., & Sant'Anna, P. H. C. (2021). Difference-in-differences with multiple
  time periods. *Journal of Econometrics*, 225(2), 200–230.
- Roth, J. (2022). Pretest with caution: Event-study estimates after testing for
  parallel trends. *American Economic Review: Insights*, 4(3), 305–322.
"""
function pre_trend_test(r::CallawaySantAnnaEstimate)
    idx = findall(k -> r.times[k] < r.groups[k], eachindex(r.coef))
    isempty(idx) && throw(ArgumentError("no pre-treatment ATT(g,t) to test"))
    labels = StatsAPI.coefnames(r)[idx]
    return _did_joint_zero_test(r.coef[idx], r.vcov[idx, idx], Inf, labels,
                                "Callaway–Sant'Anna pre-trend test",
                                "all pre-treatment ATT(g,t) are zero")
end

"""
    did_callaway_santanna(data, outcome, treatment, unit, time;
                          covariates=Symbol[], control_group=:never_treated,
                          anticipation=0, base_period=:varying, method=:dr,
                          weights=nothing, cluster=nothing, bootstrap=true,
                          biters=999, rng=Random.default_rng(), trim_level=0.995)
        -> CallawaySantAnnaEstimate
    did_callaway_santanna(panel::TreatmentPanel; kwargs...)

Group-time average treatment effects ``ATT(g,t)`` for staggered adoption
(Callaway and Sant'Anna, 2021), each estimated from a clean 2×2 comparison with
the doubly robust, inverse probability weighting or outcome regression estimators of
Sant'Anna and Zhao (2020).

The building block is ``ATT(g,t) = E[Y_t(g) - Y_t(\\infty) \\mid G = g]``, the
average effect in period ``t`` for the cohort first treated in period ``g``. It is
identified under (i) an absorbing treatment; (ii) limited anticipation, no effect
more than `anticipation` periods before ``g``; (iii) parallel trends conditional on
pre-treatment covariates between cohort ``g`` and the comparison group, which is
either the never-treated units (`control_group = :never_treated`) or all units not
yet treated by period ``t`` (`:not_yet_treated`); and (iv) overlap, that the
generalized propensity score is bounded away from one. The never-treated version
requires parallel trends only relative to the never-treated group and only from the
last pre-treatment period onwards; the not-yet-treated version uses more comparison
units, and is therefore more precise, but also imposes parallel trends between
treated cohorts in the periods before the later cohort is treated (Callaway and
Sant'Anna, 2021, Assumptions 4 and 5). Each ``ATT(g,t)`` is a 2×2
DiD between cohort ``g`` and the comparison group, from a base period to period
``t``: for post-treatment cells the base is ``g - 1 -`` anticipation, the last clean
period. Because already-treated units are never used as controls, the estimates do
not suffer from the negative weighting of two-way fixed effects (Goodman-Bacon,
2021), whatever the pattern of effect heterogeneity.

Pre-treatment cells (``t < g``) are placebo estimates whose meaning depends on
`base_period`. With `:varying` (the default, as in R's `did`) each compares period
``t`` with ``t - 1``, so they are short differences that describe period-to-period
deviations; with `:universal` every cell uses ``g - 1 -`` anticipation, which
yields a conventional event-study path relative to a common reference period and is
required by [`honest_did`](@ref). The two choices give identical post-treatment
estimates but different pre-treatment coefficients and plots, so state which was
used (Roth, 2026).

Inference is based on the influence functions of the 2×2 estimators, which account
for the estimation of the propensity score and outcome regressions, with normal
critical values; the variance is clustered at the unit level (or at `cluster`).
With `bootstrap = true` a multiplier bootstrap with Rademacher weights at the
cluster level produces the sup-t draws used by simultaneous confidence bands;
standard errors are always analytic. Doubly robust estimation with many covariates
and small cohorts can be unstable; check the size of each cohort and comparison
group. Summarize the ``ATT(g,t)`` with [`aggregate_att`](@ref) rather than
reporting the raw cells. The implementation reproduces R's `did` package for panels
and repeated cross-sections.

# Arguments
- `data`: a long, balanced panel (units missing some periods are dropped with a
  warning), or repeated cross-sections.
- `outcome::Symbol`: outcome column.
- `treatment`: an absorbing 0/1 indicator, or [`FirstTreated`](@ref)`(column)`
  (never treated coded 0, as in R's `did`).
- `unit`: unit identifier, or `nothing` for **repeated cross-sections** (then
  `treatment` must be `FirstTreated`, every observation is its own unit and the
  repeated cross-section kernels are used).
- `time::Symbol`: time column.

# Keywords
- `covariates::Vector{Symbol} = Symbol[]`: numeric pre-treatment covariates; each
  comparison uses their values in the earlier of its two periods, as in R.
- `control_group::Symbol = :never_treated`: `:never_treated` or `:not_yet_treated`
  (units not yet treated by ``\\max(t, \\text{base}) +`` anticipation).
- `anticipation::Integer = 0`: number of anticipation periods.
- `base_period::Symbol = :varying`: `:varying` or `:universal`, as described above;
  with `:universal` the reference cell is omitted.
- `method::Symbol = :dr`: `:dr` (traditional doubly robust, the default in R's
  `did`), `:dr_improved`, `:ipw`, `:ipw_unnormalized` or `:reg`; see
  [`did_drdid`](@ref).
- `weights::Union{Nothing,Symbol} = nothing`: sampling weights (the earlier period's
  value in each comparison).
- `cluster::Union{Nothing,Symbol} = nothing`: a unit-invariant column within which
  influence functions are summed; by default units.
- `bootstrap::Bool = true`: compute multiplier-bootstrap sup-t draws.
- `biters::Integer = 999`: number of bootstrap draws.
- `rng::AbstractRNG = Random.default_rng()`: generator for the bootstrap.
- `trim_level::Real = 0.995`: propensity-score trimming level for comparison units.

Units treated in the first period (after anticipation) are dropped, since they have
no pre-treatment period. Without never-treated units, the periods from the last
cohort's (anticipation-adjusted) treatment onwards are dropped, and with
`:never_treated` that cohort then serves as the comparison group (as in R's `did`).

# Returns
- `CallawaySantAnnaEstimate`: the ``ATT(g,t)`` with covariance and influence
  functions.

# Examples
```julia
using DrSnow, CSV, DataFrames, StableRNGs
file = joinpath(pkgdir(DrSnow), "test", "validation", "did", "mpdta.csv")
mpdta = CSV.read(file, DataFrame)
cs = did_callaway_santanna(mpdta, :lemp, FirstTreated(:first_treat), :countyreal,
                           :year; covariates=[:lpop], rng=StableRNG(1))
coeftable(cs)
es = aggregate_att(cs, :dynamic; rng=StableRNG(2))
confint(es; uniform=true)
```

# References
- Callaway, B., & Sant'Anna, P. H. C. (2021). Difference-in-differences with multiple
  time periods. *Journal of Econometrics*, 225(2), 200–230.
- Sant'Anna, P. H. C., & Zhao, J. (2020). Doubly robust difference-in-differences
  estimators. *Journal of Econometrics*, 219(1), 101–122.
- Goodman-Bacon, A. (2021). Difference-in-differences with variation in treatment
  timing. *Journal of Econometrics*, 225(2), 254–277.
- Roth, J. (2026). Interpreting event-studies from recent difference-in-differences
  methods. *The Japanese Economic Review*, 77(2), 275–288.
- Roth, J., Sant'Anna, P. H. C., Bilinski, A., & Poe, J. (2023). What's trending in
  difference-in-differences? A synthesis of the recent econometrics literature.
  *Journal of Econometrics*, 235(2), 2218–2244.
- Callaway, B., & Sant'Anna, P. H. C. (2026). did: Treatment effects with multiple
  periods and groups. R package version 2.5.1.
"""
function did_callaway_santanna(data, outcome::Symbol, treatment, unit, time::Symbol;
                               covariates::Vector{Symbol}=Symbol[],
                               control_group::Symbol=:never_treated,
                               anticipation::Integer=0, base_period::Symbol=:varying,
                               method::Symbol=:dr,
                               weights::Union{Nothing,Symbol}=nothing,
                               cluster::Union{Nothing,Symbol}=nothing,
                               bootstrap::Bool=true, biters::Integer=999,
                               rng::AbstractRNG=Random.default_rng(),
                               trim_level::Real=0.995)
    _did_check_method(method)
    _did_check_control_group(control_group)
    base_period in (:varying, :universal) ||
        throw(ArgumentError("base_period must be :varying or :universal"))
    anticipation >= 0 || throw(ArgumentError("anticipation must be ≥ 0"))
    δ = Int(anticipation)
    tcol = _did_treatment_column(treatment)
    if unit === nothing
        return _did_cs_repeated_cross_sections(data, outcome, treatment, time; covariates,
            control_group, δ, base_period, method, weights, cluster, bootstrap, biters,
            rng, trim_level)
    end
    df, tm, G, Tuse, exclude_last = _did_cs_panel_sample(
        data, [outcome, tcol, unit, time, covariates..., weights, cluster], treatment,
        unit, time, control_group, δ; context="did_callaway_santanna")
    N, T = length(tm.units), length(tm.periods)
    # Wide arrays (units × periods), matched by key.
    Y = fill(NaN, N, T)
    Wt = ones(N, T)
    k = length(covariates)
    Xt = ones(N, 1 + k, T)
    y = float.(df[!, outcome])
    wcol = weights === nothing ? nothing : float.(df[!, weights])
    wcol !== nothing && any(<(0), wcol) && throw(ArgumentError("weights must be ≥ 0"))
    Xrows = _did_design_matrix(df, covariates)
    for i in 1:nrow(df)
        u, p = tm.row_unit[i], tm.row_period[i]
        Y[u, p] = y[i]
        wcol === nothing || (Wt[u, p] = wcol[i])
        Xt[u, :, p] = Xrows[i, :]
    end
    ucl = nothing
    if cluster !== nothing
        cv = df[!, cluster]
        ucl = Vector{eltype(cv)}(undef, N)
        seen = falses(N)
        for i in 1:nrow(df)
            u = tm.row_unit[i]
            if seen[u] && !isequal(ucl[u], cv[i])
                throw(ArgumentError("cluster variable `$cluster` varies within unit " *
                                    "$(tm.units[u]); it must be unit-invariant"))
            end
            ucl[u] = cv[i]
            seen[u] = true
        end
    end
    glist = sort!(unique(filter(g -> g > 0 && g != exclude_last, G)))
    isempty(glist) && throw(ArgumentError("did_callaway_santanna: no treated cohorts"))
    groups, times, bases, att = Int[], Int[], Int[], Float64[]
    cols = Vector{Vector{Float64}}()
    skipped = String[]
    for g in glist
        pret_g = g - δ - 1
        tseq = base_period === :varying ? (2:Tuse) : (1:Tuse)
        for t in tseq
            base = (base_period === :universal || g <= t) ? pret_g : t - 1
            base == t && continue          # universal reference cell (ATT ≡ 0)
            treated = G .== g
            ctrl = _did_control_units(G, g, t, base, control_group, δ)
            s = treated .| ctrl
            if !any(ctrl)
                push!(skipped, "(g=$(tm.periods[g]), t=$(tm.periods[t]))")
                continue
            end
            early_p = min(t, base)
            dy = Y[s, t] .- Y[s, base]
            X = Xt[s, :, early_p]
            w = Wt[s, early_p]
            a, ψ = try
                _did_kernel_panel(method, dy, treated[s], X, w; trim_level=trim_level)
            catch err
                err isa ArgumentError || rethrow()
                throw(ArgumentError("ATT(g=$(tm.periods[g]), t=$(tm.periods[t])): " *
                                    err.msg))
            end
            full = zeros(N)
            full[s] = (N / count(s)) .* ψ
            push!(groups, g); push!(times, t); push!(bases, base); push!(att, a)
            push!(cols, full)
        end
    end
    isempty(skipped) || @warn "did_callaway_santanna: no comparison units for " *
                              join(skipped, ", ") * "; these ATT(g,t) are not reported"
    isempty(att) && throw(ArgumentError("did_callaway_santanna: no estimable ATT(g,t)"))
    Ψ = reduce(hcat, cols)
    V, ncl = _if_vcov(Ψ, ucl)
    supt = Float64[]
    if bootstrap
        _, supt = _multiplier_bootstrap(rng, Ψ, ucl, biters)
    end
    uw = Wt[:, 1]
    settings = (control_group=control_group, anticipation=δ, base_period=base_period,
                method=method, covariates=covariates, bootstrap=bootstrap,
                biters=Int(biters), units=tm.units, n_periods_used=Tuse, panel=true)
    return CallawaySantAnnaEstimate(groups, times, bases, att, V, Ψ, tm.periods, G, uw,
                                    ucl, cluster === nothing ? 0 : ncl, N * Tuse, supt,
                                    settings)
end

# Repeated cross-sections (R's `did` with panel = FALSE): every observation is its own
# unit; each ATT(g,t) uses the observations of cohort g and of the comparison group in
# periods t and base, with the repeated cross-section kernels.
function _did_cs_repeated_cross_sections(data, outcome, treatment, time; covariates,
                                         control_group, δ, base_period, method, weights,
                                         cluster, bootstrap, biters, rng, trim_level)
    treatment isa FirstTreated || throw(ArgumentError(
        "repeated cross-sections need the cohort of every observation: pass " *
        "FirstTreated(column) as the treatment"))
    tcol = treatment.column
    df, tm, G, Tuse, exclude_last = _did_cs_rc_sample(
        data, [outcome, tcol, time, covariates..., weights, cluster], treatment, time,
        control_group, δ; context="did_callaway_santanna")
    n = nrow(df)
    P = tm.row_period
    y = float.(df[!, outcome])
    w = weights === nothing ? ones(n) : float.(df[!, weights])
    any(<(0), w) && throw(ArgumentError("weights must be ≥ 0"))
    X = _did_design_matrix(df, covariates)
    cl = cluster === nothing ? nothing : df[!, cluster]
    glist = sort!(unique(filter(g -> g > 0 && g != exclude_last, G)))
    isempty(glist) && throw(ArgumentError("did_callaway_santanna: no treated cohorts"))
    groups, times, bases, att = Int[], Int[], Int[], Float64[]
    cols = Vector{Vector{Float64}}()
    for g in glist
        pret_g = g - δ - 1
        for t in (base_period === :varying ? (2:Tuse) : (1:Tuse))
            base = (base_period === :universal || g <= t) ? pret_g : t - 1
            base == t && continue
            ctrl = _did_control_units(G, g, t, base, control_group, δ)
            s = ((G .== g) .| ctrl) .& ((P .== t) .| (P .== base))
            a, ψ = try
                _did_kernel_rc(method, y[s], P[s] .== t, G[s] .== g, X[s, :], w[s];
                               trim_level=trim_level)
            catch err
                err isa ArgumentError || rethrow()
                throw(ArgumentError("ATT(g=$(tm.periods[g]), t=$(tm.periods[t])): " *
                                    err.msg))
            end
            full = zeros(n)
            full[s] = (n / count(s)) .* ψ
            push!(groups, g); push!(times, t); push!(bases, base); push!(att, a)
            push!(cols, full)
        end
    end
    Ψ = reduce(hcat, cols)
    V, ncl = _if_vcov(Ψ, cl)
    supt = bootstrap ? last(_multiplier_bootstrap(rng, Ψ, cl, biters)) : Float64[]
    settings = (control_group=control_group, anticipation=δ, base_period=base_period,
                method=method, covariates=covariates, bootstrap=bootstrap,
                biters=Int(biters), units=Int[], n_periods_used=Tuse, panel=false)
    return CallawaySantAnnaEstimate(groups, times, bases, att, V, Ψ, tm.periods, G, w, cl,
                                    cluster === nothing ? 0 : ncl, n, supt, settings)
end

# Sample preparation shared by the Callaway–Sant'Anna panel estimators
# (`did_callaway_santanna`, `dml_did_multi`): absorbing treatment, balanced panel
# (incomplete units dropped), units treated in the first period dropped, and the
# last-treated cohort handling when there are no never-treated units. Returns
# `(df, tm, G, Tuse, exclude_last)`.
function _did_cs_panel_sample(data, cols, treatment, unit, time, control_group, δ;
                              context)
    df = _did_prepare(data, cols; context=context, treatment=treatment)
    tm = treatment_timing(df, treatment, unit, time; anticipation=δ)
    tm.absorbing || throw(ArgumentError(
        "$(context) requires an absorbing (staggered-adoption) treatment"))
    # Balanced panel: drop units not observed in every period.
    if !tm.balanced
        cnt = zeros(Int, length(tm.units))
        foreach(u -> cnt[u] += 1, tm.row_unit)
        full = cnt[tm.row_unit] .== length(tm.periods)
        @warn "$(context): dropping $(count(<(length(tm.periods)), cnt)) " *
              "unit(s) not observed in every period (balanced panel required)"
        df = df[full, :]
        nrow(df) > 0 || throw(ArgumentError("no unit is observed in every period"))
        tm = treatment_timing(df, treatment, unit, time; anticipation=δ)
    end
    # Drop units treated in the first period (no pre-treatment period).
    early = (tm.row_cohort .> 0) .& (tm.row_cohort .<= 1 + δ)
    if any(early)
        @warn "$(context): dropped $(length(unique(tm.row_unit[early]))) " *
              "unit(s) already treated in the first period" *
              (δ > 0 ? " (accounting for anticipation = $δ)" : "")
        df = df[.!early, :]
        tm = treatment_timing(df, treatment, unit, time; anticipation=δ)
    end
    N, T = length(tm.units), length(tm.periods)
    G = copy(tm.unit_cohort)
    Tuse = T
    exclude_last = 0
    if !any(==(0), G)
        latest = maximum(G)
        Tuse = latest - δ - 1
        Tuse >= 2 || throw(ArgumentError(
            "$(context): no never-treated units and too few periods before " *
            "the last cohort is treated"))
        if control_group === :never_treated
            @warn "$(context): no never-treated units; the last-treated " *
                  "cohort ($(tm.periods[latest])) is used as the comparison group and " *
                  "periods from $(tm.periods[latest - δ]) on are dropped"
            G[G .== latest] .= 0
        else
            exclude_last = latest
        end
    end
    return df, tm, G, Tuse, exclude_last
end

# Repeated cross-section counterpart of `_did_cs_panel_sample`.
function _did_cs_rc_sample(data, cols, treatment, time, control_group, δ; context)
    treatment isa FirstTreated || throw(ArgumentError(
        "repeated cross-sections need the cohort of every observation: pass " *
        "FirstTreated(column) as the treatment"))
    df = _did_prepare(data, cols; context=context, treatment=treatment)
    tm = treatment_timing(df, treatment, nothing, time; anticipation=δ)
    early = (tm.row_cohort .> 0) .& (tm.row_cohort .<= 1 + δ)
    if any(early)
        @warn "$(context): dropped $(count(early)) observation(s) of cohorts " *
              "already treated in the first period"
        df = df[.!early, :]
        tm = treatment_timing(df, treatment, nothing, time; anticipation=δ)
    end
    G = copy(tm.row_cohort)
    Tuse = length(tm.periods)
    exclude_last = 0
    if !any(==(0), G)
        latest = maximum(G)
        Tuse = latest - δ - 1
        Tuse >= 2 || throw(ArgumentError(
            "$(context): no never-treated observations and too few periods " *
            "before the last cohort is treated"))
        if control_group === :never_treated
            @warn "$(context): no never-treated observations; the " *
                  "last-treated cohort is used as the comparison group"
            G[G .== latest] .= 0
        else
            exclude_last = latest
        end
    end
    return df, tm, G, Tuse, exclude_last
end

did_callaway_santanna(panel::TreatmentPanel; kwargs...) =
    did_callaway_santanna(panel.data, panel.outcome, panel.treatment, panel.unit_id,
                          panel.time; covariates=panel.covariates, kwargs...)

# ---------------------------------------------------------------------------
# Aggregations (R did::aggte)
# ---------------------------------------------------------------------------

# Influence function of the estimated aggregation weights (R's `wif`).
function _did_wif(keepers, pg, wind, Gunit, group)
    isempty(keepers) && return zeros(length(wind), 0)
    Spg = sum(pg[keepers])
    centered = reduce(hcat, [wind .* (Gunit .== group[k]) .- pg[k] for k in keepers])
    if1 = centered ./ Spg
    if2 = vec(sum(centered; dims=2)) * (pg[keepers] ./ Spg^2)'
    return if1 .- if2
end

function _did_agg_if(att, Ψ, which, wagg, wif)
    ψ = Ψ[:, which] * wagg
    wif === nothing || (ψ .+= wif * att[which])
    return ψ
end

"""
    aggregate_att(cs::CallawaySantAnnaEstimate, type=:simple;
                  balance_e=nothing, min_e=nothing, max_e=nothing,
                  bootstrap=cs.settings.bootstrap, biters=cs.settings.biters,
                  rng=Random.default_rng()) -> Union{AggregatedATT, EventStudyEstimate}

Aggregate Callaway–Sant'Anna group-time effects into summary parameters (Callaway and
Sant'Anna, 2021, Section 4), with standard errors that account for the estimation
of the aggregation weights.

Group-time effects are numerous and individually imprecise; the aggregations answer
specific questions with weighted averages
``\\theta = \\sum_{g,t} w(g,t)\\, ATT(g,t)`` over post-treatment cells:

- `:simple`: the average of all post-treatment ``ATT(g,t)`` weighted by cohort size,
  an overall ATT across treated cohort-periods.
- `:group`: ``\\theta(g)``, the average effect for cohort ``g`` over its
  post-treatment periods, and as overall parameter their cohort-size weighted
  average, which weights every treated unit equally irrespective of how long it is
  observed as treated.
- `:dynamic`: ``\\theta(e)``, the average effect ``e`` periods after treatment across
  the cohorts observed at ``e``, weighted by cohort size; the overall parameter is
  the average of ``\\theta(e)`` over ``e \\ge 0``. Because different cohorts
  contribute at different ``e``, changes in ``\\theta(e)`` mix dynamics with
  compositional changes; `balance_e` keeps only cohorts observed for at least
  `balance_e` post-treatment periods, which fixes the composition at the cost of
  fewer cohorts.
- `:calendar`: ``\\theta(t)``, the average effect in period ``t`` across the cohorts
  treated by ``t``; the overall parameter is the average over periods.

The weights depend on estimated cohort shares, so the influence function of each
aggregate adds the estimation effect of the weights to the weighted influence
functions of the cells; standard errors use normal critical values and the
clustering of the underlying estimate. With `bootstrap = true`, multiplier-bootstrap
sup-t draws are computed for simultaneous bands over the components (event times,
cohorts or periods). For `:dynamic` the pre-treatment ``\\theta(e)``, ``e < 0``, are
averages of the placebo cells and inherit their interpretation from `base_period`
(see [`did_callaway_santanna`](@ref)).

# Arguments
- `cs::CallawaySantAnnaEstimate`: group-time effects from
  [`did_callaway_santanna`](@ref).
- `type::Symbol = :simple`: `:simple`, `:group`, `:dynamic` or `:calendar`.

# Keywords
- `balance_e::Union{Nothing,Integer} = nothing`: for `:dynamic`, keep only cohorts
  observed for at least `balance_e` post-treatment periods.
- `min_e`, `max_e = nothing`: for `:dynamic`, the range of event times reported;
  `max_e` also caps the post-treatment periods used by `:simple` and `:group`.
- `bootstrap::Bool = cs.settings.bootstrap`: compute sup-t draws for uniform bands.
- `biters::Integer = cs.settings.biters`: number of bootstrap draws.
- `rng::AbstractRNG = Random.default_rng()`: generator for the bootstrap.

# Returns
- `EventStudyEstimate` for `:dynamic`, with `details.overall` (a
  [`DiDEstimate`](@ref) of the average post-treatment ``\\theta(e)``) and bootstrap
  sup-t draws;
- `AggregatedATT` otherwise, with the overall parameter as first coefficient.

# Examples
```julia
using DrSnow, CSV, DataFrames, StableRNGs
file = joinpath(pkgdir(DrSnow), "test", "validation", "did", "mpdta.csv")
mpdta = CSV.read(file, DataFrame)
cs = did_callaway_santanna(mpdta, :lemp, FirstTreated(:first_treat), :countyreal,
                           :year; rng=StableRNG(1))
aggregate_att(cs, :simple)
aggregate_att(cs, :group)
es = aggregate_att(cs, :dynamic; min_e=-3, max_e=3, rng=StableRNG(2))
confint(es; uniform=true)
es.details.overall
```

# References
- Callaway, B., & Sant'Anna, P. H. C. (2021). Difference-in-differences with multiple
  time periods. *Journal of Econometrics*, 225(2), 200–230.
- Roth, J., Sant'Anna, P. H. C., Bilinski, A., & Poe, J. (2023). What's trending in
  difference-in-differences? A synthesis of the recent econometrics literature.
  *Journal of Econometrics*, 235(2), 2218–2244.
"""
function aggregate_att(cs::CallawaySantAnnaEstimate, type::Symbol=:simple;
                       balance_e::Union{Nothing,Integer}=nothing,
                       min_e::Union{Nothing,Integer}=nothing,
                       max_e::Union{Nothing,Integer}=nothing,
                       bootstrap::Bool=cs.settings.bootstrap,
                       biters::Integer=cs.settings.biters,
                       rng::AbstractRNG=Random.default_rng())
    type in (:simple, :group, :dynamic, :calendar) || throw(ArgumentError(
        "type must be :simple, :group, :dynamic or :calendar"))
    att, Ψ = cs.coef, cs.influence
    group, t = cs.groups, cs.times
    Gunit, wind = cs.unit_cohort, cs.unit_weight
    n = size(Ψ, 1)
    glist = sort!(unique(group))
    pgg = [mean(wind .* (Gunit .== g)) for g in glist]
    pg = pgg[[searchsortedfirst(glist, g) for g in group]]
    maxe = max_e === nothing ? typemax(Int) ÷ 2 : Int(max_e)
    mine = min_e === nothing ? typemin(Int) ÷ 2 : Int(min_e)
    keepers = findall(k -> group[k] <= t[k] <= group[k] + maxe, eachindex(att))
    cl = cs.unit_cluster
    finish(IFs) = _if_vcov(IFs, cl)
    boot(IFs) = bootstrap ? last(_multiplier_bootstrap(rng, IFs, cl, biters)) :
                Float64[]
    ncl = cs.n_clusters
    if type === :simple
        isempty(keepers) && throw(ArgumentError("no post-treatment ATT(g,t)"))
        wk = pg[keepers] ./ sum(pg[keepers])
        θ = dot(wk, att[keepers])
        ψ = _did_agg_if(att, Ψ, keepers, wk, _did_wif(keepers, pg, wind, Gunit, group))
        V, _ = finish(reshape(ψ, :, 1))
        return AggregatedATT(:simple, Any[], [θ], V, cs.nobs, ncl, Float64[],
                             reshape(ψ, :, 1), (source=cs,))
    elseif type === :group
        θg = Float64[]
        IFg = Vector{Vector{Float64}}()
        for g in glist
            wh = findall(k -> group[k] == g && g <= t[k] <= g + maxe, eachindex(att))
            isempty(wh) && throw(ArgumentError("cohort $(cs.periods[g]) has no " *
                                               "post-treatment ATT(g,t)"))
            wg = pg[wh] ./ sum(pg[wh])
            push!(θg, dot(wg, att[wh]))
            push!(IFg, _did_agg_if(att, Ψ, wh, wg, nothing))
        end
        IFgm = reduce(hcat, IFg)
        wo = pgg ./ sum(pgg)
        θ = dot(wo, θg)
        wif = _did_wif(collect(eachindex(glist)), pgg, wind, Gunit, glist)
        ψ = IFgm * wo .+ wif * θg
        IFs = hcat(ψ, IFgm)
        V, _ = finish(IFs)
        return AggregatedATT(:group, collect(cs.periods[glist]), vcat(θ, θg), V, cs.nobs,
                             ncl, boot(IFgm), IFs, (source=cs,))
    elseif type === :calendar
        tl = sort!(unique(t[t .>= minimum(group)]))
        tl = filter(t1 -> any(k -> t[k] == t1 && group[k] <= t1, eachindex(att)), tl)
        isempty(tl) && throw(ArgumentError("no post-treatment ATT(g,t)"))
        θt = Float64[]
        IFt = Vector{Vector{Float64}}()
        for t1 in tl
            wh = findall(k -> t[k] == t1 && group[k] <= t1, eachindex(att))
            wt = pg[wh] ./ sum(pg[wh])
            push!(θt, dot(wt, att[wh]))
            push!(IFt, _did_agg_if(att, Ψ, wh, wt, _did_wif(wh, pg, wind, Gunit, group)))
        end
        IFtm = reduce(hcat, IFt)
        θ = mean(θt)
        ψ = vec(mean(IFtm; dims=2))
        IFs = hcat(ψ, IFtm)
        V, _ = finish(IFs)
        return AggregatedATT(:calendar, collect(cs.periods[tl]), vcat(θ, θt), V, cs.nobs,
                             ncl, boot(IFtm), IFs, (source=cs,))
    end
    # :dynamic
    e_all = t .- group
    Tmax = maximum(t)
    incl = balance_e === nothing ? trues(length(att)) : (Tmax .- group .>= balance_e)
    eseq = sort!(unique(e_all[incl]))
    if balance_e !== nothing
        eseq = filter(e -> balance_e - (Tmax - 1) <= e <= balance_e, eseq)
    end
    eseq = filter(e -> mine <= e <= maxe, eseq)
    isempty(eseq) && throw(ArgumentError(
        "no event times left after applying balance_e/min_e/max_e"))
    θe = Float64[]
    IFe = Vector{Vector{Float64}}()
    for e in eseq
        wh = findall(k -> e_all[k] == e && incl[k], eachindex(att))
        we = pg[wh] ./ sum(pg[wh])
        push!(θe, dot(we, att[wh]))
        push!(IFe, _did_agg_if(att, Ψ, wh, we, _did_wif(wh, pg, wind, Gunit, group)))
    end
    IFem = reduce(hcat, IFe)
    V, _ = finish(IFem)
    post = findall(>=(0), eseq)
    overall = nothing
    if !isempty(post)
        ψo = vec(mean(IFem[:, post]; dims=2))
        Vo, _ = finish(reshape(ψo, :, 1))
        overall = DiDEstimate([mean(θe[post])], Vo, ["ATT"], cs.nobs, Inf, ncl,
                              count(>(0), Gunit), count(==(0), Gunit),
                              length(cs.periods),
                              "Callaway–Sant'Anna dynamic aggregation",
                              "average of θ(e) over e ≥ 0", (influence=ψo,))
    end
    δ = cs.settings.anticipation
    ref = cs.settings.base_period === :universal ? [-1 - δ] : Int[]
    return EventStudyEstimate(eseq, θe, V, ref, cs.nobs, Inf, ncl,
                              "Callaway–Sant'Anna event study (" *
                              "$(cs.settings.method), $(cs.settings.control_group))",
                              "θ(e): cohort-size weighted average of ATT(g, g+e)",
                              boot(IFem),
                              (overall=overall, influence=IFem, binned=(false, false),
                               balance_e=balance_e, source=cs,
                               n_treated=count(>(0), Gunit),
                               n_control=count(==(0), Gunit),
                               n_periods=length(cs.periods),
                               note=cs.settings.base_period === :varying ?
                                    "Pre-treatment θ(e) compare consecutive periods " *
                                    "(varying base period)." : ""))
end
