# Plot data for design diagnostics: raw trends, covariate balance, RD falsification,
# Honest DiD sensitivity, IV design plots (judge first stage, Rotemberg weights, MTE)
# and synthetic-control backdating. Backend-independent, like `data.jl`: every
# `_viz_*_data` function returns plain DataFrames / NamedTuples and is tested
# without Makie.

# ---------------------------------------------------------------------------------
# Raw (or covariate-adjusted) outcome trends by cohort / ever-treated status
# ---------------------------------------------------------------------------------

_viz_opt(v) = Vector{Union{Missing,Float64}}(v)

"""x positions for time values: the values themselves when real, else the index."""
_viz_time_x(periods) = eltype(periods) <: Real ? Float64.(periods) :
                       Float64.(collect(1:length(periods)))

"""
    _viz_trends_data(data, outcome, tm::TreatmentTiming; by=:cohort, covariates=[],
                     level=0.95, cohorts=nothing)
    _viz_trends_data(data, outcome, treatment, unit, time; kwargs...)

Mean outcome by group and period with `level` confidence intervals of the mean
(t with `n - 1` degrees of freedom over the observations of the cell; no
clustering across periods is needed because each interval refers to one period).
Groups are the adoption cohorts and the never-treated units (`by = :cohort`) or
ever- vs never-treated units (`by = :treated`).

With `covariates`, the outcome is first adjusted for covariate composition:
`y - (x - x̄)'β̂`, where `β̂` is the within group × period OLS coefficient of the
covariates (so the adjustment does not absorb differences in group trends) and
`x̄` the overall covariate mean.

Returns a NamedTuple with
- `table`: `group::String`, `group_order::Int`, `cohort` (cohort period value,
  `missing` for never-treated / ever-treated groups), `period::Int`, `time`,
  `x::Float64`, `mean`, `conf_low`, `conf_high`, `n`, sorted by group and period;
- `onsets`: `group`, `group_order`, `time`, `x` of every adoption date (the first
  treated period of each cohort);
- `ticks`: `(positions, labels)` for the time axis when time is not numeric,
  otherwise `nothing`;
- `adjusted::Bool`, `coefficients` (the covariate coefficients, or `nothing`),
  `level`, `by`, `absorbing`.
"""
function _viz_trends_data(data, outcome::Symbol, tm::TreatmentTiming;
                          by::Symbol=:cohort, covariates::Vector{Symbol}=Symbol[],
                          level::Real=0.95, cohorts=nothing)
    by in (:cohort, :treated) ||
        throw(ArgumentError("plot_trends: `by` must be :cohort or :treated"))
    0 < level < 1 || throw(ArgumentError("level must be in (0, 1)"))
    require_columns(data, vcat(outcome, covariates); context="plot_trends")
    nrow(data) == length(tm.row_period) || throw(ArgumentError(
        "plot_trends: the TreatmentTiming was built from a dataset with " *
        "$(length(tm.row_period)) rows, but `data` has $(nrow(data))"))
    y0 = data[!, outcome]
    keep = .!ismissing.(y0)
    for c in covariates
        keep .&= .!ismissing.(data[!, c])
    end
    any(keep) || throw(ArgumentError("plot_trends: no non-missing outcome values"))
    rows = findall(keep)
    y = Float64.(y0[rows])
    P = tm.row_period[rows]
    G = tm.row_cohort[rows]
    all_cohorts = sort!(unique(G[G .> 0]))
    if cohorts !== nothing
        want = collect(cohorts)
        idx = Dict(p => i for (i, p) in enumerate(tm.periods))
        sel = Int[]
        for c in want
            haskey(idx, c) && idx[c] in all_cohorts || throw(ArgumentError(
                "plot_trends: cohort $c not found; cohorts are " *
                join(string.(tm.periods[all_cohorts]), ", ")))
            push!(sel, idx[c])
        end
        all_cohorts = sort!(unique(sel))
    end
    gkey = by === :cohort ? G : Int.(G .> 0)          # 0 = never, else cohort / 1
    coefs = nothing
    if !isempty(covariates)
        X = Matrix{Float64}(undef, length(rows), length(covariates))
        for (j, c) in enumerate(covariates)
            X[:, j] = Float64.(data[rows, c])
        end
        cell = Dict{Tuple{Int,Int},Vector{Int}}()
        for i in eachindex(y)
            push!(get!(cell, (gkey[i], P[i]), Int[]), i)
        end
        yd, Xd = copy(y), copy(X)
        for ix in values(cell)
            yd[ix] .-= mean(y[ix])
            Xd[ix, :] .-= mean(X[ix, :]; dims=1)
        end
        rank(Xd) == size(Xd, 2) || throw(ArgumentError(
            "plot_trends: covariates are collinear or constant within group × period " *
            "cells; cannot adjust"))
        coefs = Xd \ yd
        y = y .- (X .- mean(X; dims=1)) * coefs
    end
    groups = Tuple{Int,String,Any}[]                 # (key, label, cohort value)
    any(==(0), G) && push!(groups, (0, "Never treated", missing))
    if by === :cohort
        for g in all_cohorts
            push!(groups, (g, "Cohort " * string(tm.periods[g]), tm.periods[g]))
        end
    else
        any(>(0), G) && push!(groups, (1, "Ever treated", missing))
    end
    length(groups) >= 1 || throw(ArgumentError("plot_trends: no groups to plot"))
    xs = _viz_time_x(tm.periods)
    out = DataFrame(group=String[], group_order=Int[], cohort=Any[], period=Int[],
                    time=Any[], x=Float64[], mean=Float64[],
                    conf_low=Union{Missing,Float64}[], conf_high=Union{Missing,Float64}[],
                    n=Int[])
    incl = by === :treated && cohorts !== nothing ?
           (i -> G[i] == 0 || G[i] in all_cohorts) : (i -> true)
    for (o, (k, lab, cv)) in enumerate(groups)
        ix = [i for i in eachindex(y) if gkey[i] == k && incl(i)]
        isempty(ix) && continue
        for p in sort!(unique(P[ix]))
            v = y[ix[P[ix] .== p]]
            m = mean(v)
            n = length(v)
            lo, hi = if n > 1
                h = critical_value(level, n - 1) * std(v) / sqrt(n)
                (m - h, m + h)
            else
                (missing, missing)
            end
            push!(out, (lab, o, cv, p, tm.periods[p], xs[p], m, lo, hi, n))
        end
    end
    onsets = DataFrame(group=String[], group_order=Int[], time=Any[], x=Float64[])
    for g in all_cohorts
        if by === :cohort
            o = findfirst(t -> t[1] == g, groups)
            push!(onsets, (groups[o][2], o, tm.periods[g], xs[g]))
        else
            o = findfirst(t -> t[1] == 1, groups)
            push!(onsets, ("Ever treated", o, tm.periods[g], xs[g]))
        end
    end
    ticks = eltype(tm.periods) <: Real ? nothing : (xs, string.(tm.periods))
    return (table=out, onsets=onsets, ticks=ticks, adjusted=!isempty(covariates),
            coefficients=coefs, covariates=covariates, level=float(level), by=by,
            absorbing=tm.absorbing)
end

function _viz_trends_data(data, outcome::Symbol, treatment, unit, time; kwargs...)
    tm = treatment_timing(data, treatment, unit, time)
    return _viz_trends_data(data, outcome, tm; kwargs...)
end

# ---------------------------------------------------------------------------------
# Covariate balance ("love plots")
# ---------------------------------------------------------------------------------

"""
    _viz_balance_data(x; data=nothing, level=0.95)

Rows of a balance plot from
- a [`pretreatment_balance`](@ref) table (standardized differences by cohort; no
  intervals, the table is descriptive),
- a [`ri_balance_test`](@ref) result or its `per_covariate` table (standardized
  differences with unadjusted and Westfall–Young adjusted randomization p-values),
- a [`rd_covariate_balance`](@ref) table (bias-corrected discontinuities with robust
  intervals and Holm-adjusted p-values). With `data`, estimates and intervals are
  divided by each covariate's standard deviation in `data`, so covariates on
  different scales share one axis.

Returns a NamedTuple `(table, kind, xlabel, threshold, adjustment, test)`; `table`
has columns `covariate::String`, `group::String` (cohort label or `""`),
`estimate`, `conf_low`, `conf_high`, `pvalue`, `pvalue_adj` (the last four
`missing` where not available). `kind` is `:pretreatment`, `:ri` or `:rd`.
"""
function _viz_balance_data(t::AbstractDataFrame; data=nothing, level::Real=0.95)
    cols = Set(propertynames(t))
    none(n) = Vector{Union{Missing,Float64}}(missing, n)
    if :std_diff in cols && :cohort in cols
        data === nothing || throw(ArgumentError(
            "plot_balance: `data` is only used for RD balance tables"))
        ok = isfinite.(t.std_diff)
        bad = count(!, ok)
        bad > 0 && @warn "plot_balance: $bad standardized difference(s) undefined " *
                         "(zero variance) are not drawn"
        s = t[ok, :]
        n = nrow(s)
        tab = DataFrame(covariate=string.(s.covariate),
                        group="Cohort " .* string.(s.cohort),
                        estimate=Float64.(s.std_diff), conf_low=none(n),
                        conf_high=none(n), pvalue=none(n), pvalue_adj=none(n))
        length(unique(tab.group)) == 1 && (tab.group .= "")
        return (table=tab, kind=:pretreatment,
                xlabel="Standardized difference (cohort − comparison)", threshold=0.1,
                adjustment="", test=nothing)
    elseif :std_difference in cols && :pvalue_westfall_young in cols
        data === nothing || throw(ArgumentError(
            "plot_balance: `data` is only used for RD balance tables"))
        n = nrow(t)
        tab = DataFrame(covariate=string.(t.covariate), group=fill("", n),
                        estimate=Float64.(t.std_difference), conf_low=none(n),
                        conf_high=none(n),
                        pvalue=Vector{Union{Missing,Float64}}(t.pvalue),
                        pvalue_adj=Vector{Union{Missing,Float64}}(t.pvalue_westfall_young))
        return (table=tab, kind=:ri,
                xlabel="Standardized difference (treated − control)", threshold=0.1,
                adjustment="Westfall–Young", test=nothing)
    elseif :pvalue_holm in cols && :ci_lower in cols
        ok = .!ismissing.(t.estimate)
        bad = count(!, ok)
        bad > 0 && @warn "plot_balance: $bad covariate(s) without an estimate are not " *
                         "drawn (see the `note` column)"
        s = t[ok, :]
        sd = ones(nrow(s))
        if data !== nothing
            require_columns(data, Symbol.(s.covariate); context="plot_balance")
            for (i, c) in enumerate(Symbol.(s.covariate))
                v = collect(skipmissing(data[!, c]))
                sd[i] = length(v) > 1 ? std(Float64.(v)) : 0.0
                sd[i] > 0 || throw(ArgumentError(
                    "plot_balance: covariate $c has zero standard deviation in `data`"))
            end
        end
        tab = DataFrame(covariate=string.(s.covariate), group=fill("", nrow(s)),
                        estimate=Float64.(s.estimate) ./ sd,
                        conf_low=_viz_opt(Float64.(s.ci_lower) ./ sd),
                        conf_high=_viz_opt(Float64.(s.ci_upper) ./ sd),
                        pvalue=Vector{Union{Missing,Float64}}(s.pvalue),
                        pvalue_adj=Vector{Union{Missing,Float64}}(s.pvalue_holm))
        xl = data === nothing ? "Discontinuity at the cutoff (bias-corrected)" :
             "Discontinuity at the cutoff (SD units, bias-corrected)"
        return (table=tab, kind=:rd, xlabel=xl, threshold=nothing, adjustment="Holm",
                test=nothing)
    end
    throw(ArgumentError("plot_balance: expected the output of pretreatment_balance, " *
                        "rd_covariate_balance or ri_balance_test"))
end

function _viz_balance_data(t::DiagnosticTest; data=nothing, level::Real=0.95)
    haskey(t.details, :per_covariate) || throw(ArgumentError(
        "plot_balance: this DiagnosticTest has no per-covariate balance table"))
    d = _viz_balance_data(t.details.per_covariate; data=data, level=level)
    return merge(d, (test=(statistic=t.statistic, pvalue=t.pvalue, name=t.name),))
end

_viz_balance_data(x; kwargs...) = _viz_unsupported("plot_balance", x)

# ---------------------------------------------------------------------------------
# RD placebo cutoffs, bandwidth sensitivity and donut estimates
# ---------------------------------------------------------------------------------

"""
    _viz_rd_placebo_data(tab; estimate=nothing)

Rows of an [`rd_placebo_cutoffs`](@ref) table that have an estimate (`cutoff`,
`side`, `estimate`, `conf_low`, `conf_high`, `pvalue`, `true_cutoff::Bool`), plus
the estimate at the true cutoff when an `RDEstimate` is given (`side = :actual`).
Also returns the number of placebo cutoffs that could not be estimated.
"""
function _viz_rd_placebo_data(tab::AbstractDataFrame;
                              estimate::Union{Nothing,RDEstimate}=nothing)
    require_columns(tab, [:cutoff, :side, :estimate, :ci_lower, :ci_upper];
                    context="plot_rd_placebos")
    ok = .!ismissing.(tab.estimate)
    s = tab[ok, :]
    out = DataFrame(cutoff=Float64.(s.cutoff), side=Symbol.(s.side),
                    estimate=Float64.(s.estimate), conf_low=Float64.(s.ci_lower),
                    conf_high=Float64.(s.ci_upper),
                    pvalue=Vector{Union{Missing,Float64}}(s.pvalue),
                    true_cutoff=falses(nrow(s)))
    if estimate !== nothing
        ci = StatsAPI.confint(estimate; level=estimate.level)
        push!(out, (estimate.cutoff, :actual, estimate.tau_bias_corrected, ci[1, 1],
                    ci[1, 2], pvalues(estimate)[1], true))
    end
    sort!(out, :cutoff)
    return (table=out, n_failed=count(!, ok),
            level=estimate === nothing ? nothing : estimate.level)
end

_viz_rd_placebo_data(x; kwargs...) = _viz_unsupported("plot_rd_placebos", x)

"""
    _viz_rd_sensitivity_data(tab)

Estimates against the bandwidth (from [`rd_bandwidth_sensitivity`](@ref); `x` is the
main bandwidth, the mean of `h_left` and `h_right` when they differ) or against the
donut radius (from [`rd_donut`](@ref)). Columns `x`, `estimate`, `conf_low`,
`conf_high`, `baseline::Bool` (multiplier 1 / radius 0), `n` (effective
observations), `label` (tick label); `kind` is `:bandwidth` or `:donut`.
"""
function _viz_rd_sensitivity_data(tab::AbstractDataFrame)
    cols = Set(propertynames(tab))
    kind = :h_left in cols && :multiplier in cols ? :bandwidth :
           :radius in cols ? :donut :
           throw(ArgumentError("plot_rd_sensitivity: expected the output of " *
                               "rd_bandwidth_sensitivity or rd_donut"))
    ok = .!ismissing.(tab.estimate)
    s = tab[ok, :]
    nrow(s) > 0 || throw(ArgumentError("plot_rd_sensitivity: no estimates to plot"))
    if kind === :bandwidth
        x = (Float64.(s.h_left) .+ Float64.(s.h_right)) ./ 2
        base = [!ismissing(m) && m == 1 for m in s.multiplier]
        lab = [ismissing(m) ? _viz_short(h) : "×" * _viz_short(m)
               for (m, h) in zip(s.multiplier, x)]
    else
        x = Float64.(s.radius)
        base = x .== 0
        lab = _viz_short.(x)
    end
    out = DataFrame(x=x, estimate=Float64.(s.estimate), conf_low=Float64.(s.ci_lower),
                    conf_high=Float64.(s.ci_upper), baseline=base,
                    n=Int.(s.n_h_left) .+ Int.(s.n_h_right), label=lab)
    sort!(out, :x)
    return (table=out, kind=kind, n_failed=count(!, ok),
            asymmetric=kind === :bandwidth && any(s.h_left .!= s.h_right))
end

_viz_rd_sensitivity_data(x) = _viz_unsupported("plot_rd_sensitivity", x)

_viz_short(x::Real) = (v = round(Float64(x); sigdigits=3);
                       isinteger(v) && abs(v) < 1e6 ? string(Int(v)) : string(v))

# ---------------------------------------------------------------------------------
# RD density (manipulation) test
# ---------------------------------------------------------------------------------

"""
    _viz_rd_density_data(x, t::DiagnosticTest; npoints=40, bins=nothing,
                         limits=nothing, level=0.95)

Histogram and local polynomial density estimates on each side of the cutoff for the
running variable `x` and its [`rd_density_test`](@ref) result `t`.

The density curves use the local polynomial CDF estimator of the test (Cattaneo,
Jansson & Ma 2020) with the test's kernel, order `p` and bandwidths: at each grid
point `x₀` the empirical CDF of the full sample is regressed on a polynomial in
`x - x₀` with kernel weights, using only the observations on the same side of the
cutoff (so the estimate never smooths across it); the density is the slope. At
the cutoff this reproduces the test's order-`p` side estimates. Pointwise `level`
intervals use the jackknife-type variance of the regression of the empirical CDF
(influence of each observation on the CDF values in the window). Grid points
whose window has fewer than `p + 2` observations are skipped.

Returns a NamedTuple `(hist, density, cutoff, h, f_left, f_right, statistic,
pvalue, n, limits)`: `hist` has `edges` and `density` (histogram normalized to
integrate to one over the full sample), `density` columns `side`, `x`, `f`,
`conf_low`, `conf_high`.
"""
function _viz_rd_density_data(x::AbstractVector, t::DiagnosticTest; npoints::Integer=40,
                              bins=nothing, limits=nothing, level::Real=0.95)
    d = t.details
    (haskey(d, :f_left) && haskey(d, :h_left) && haskey(d, :cutoff)) ||
        throw(ArgumentError("plot_rd_density: expected the result of rd_density_test"))
    npoints >= 5 || throw(ArgumentError("npoints must be at least 5"))
    c = Float64(d.cutoff)
    xs = sort!(Float64.(collect(skipmissing(x))))
    N = length(xs)
    N >= 10 || throw(ArgumentError("plot_rd_density: too few observations"))
    nl = count(<(c), xs)
    (nl == d.n_left && N - nl == d.n_right) || throw(ArgumentError(
        "plot_rd_density: `x` does not match the data of the test (expected " *
        "$(d.n_left) observations left and $(d.n_right) right of the cutoff)"))
    hl, hr = Float64(d.h_left), Float64(d.h_right)
    lo, hi = limits === nothing ? (max(xs[1], c - 2hl), min(xs[end], c + 2hr)) :
             Float64.(limits)
    lo < c < hi || throw(ArgumentError("plot_rd_density: limits must contain the cutoff"))
    F = collect(0:(N - 1)) ./ (N - 1)                 # empirical CDF (as the test)
    # ties share the CDF value of their last occurrence (mass points)
    for i in (N - 1):-1:1
        xs[i] == xs[i + 1] && (F[i] = F[i + 1])
    end
    p = Int(d.p)
    crit = critical_value(level)
    dens = DataFrame(side=Symbol[], x=Float64[], f=Float64[],
                     conf_low=Union{Missing,Float64}[], conf_high=Union{Missing,Float64}[])
    for (side, a, b, h) in ((:left, lo, c, hl), (:right, c, hi, hr))
        onside = side === :left ? (xs .< c) : (xs .>= c)
        for x0 in range(a, b; length=npoints)
            f, se = _viz_lpdensity(xs, F, onside, x0, h, p, d.kernel)
            isnan(f) && continue
            ok = isfinite(se)
            push!(dens, (side, x0, f, ok ? f - crit * se : missing,
                         ok ? f + crit * se : missing))
        end
    end
    nin = count(v -> lo <= v <= hi, xs)
    nb = bins === nothing ? clamp(round(Int, sqrt(nin)), 10, 40) : Int(bins)
    nb >= 2 || throw(ArgumentError("bins must be at least 2"))
    # bins aligned so that the cutoff is an edge
    wl = (c - lo) / max(1, round(Int, nb * (c - lo) / (hi - lo)))
    wr = (hi - c) / max(1, round(Int, nb * (hi - c) / (hi - lo)))
    edges = vcat(collect(range(lo, c; step=wl)), collect(range(c + wr, hi; step=wr)))
    edges[end] < hi && push!(edges, hi)
    counts = zeros(length(edges) - 1)
    for v in xs
        (v < lo || v > hi) && continue
        k = searchsortedlast(edges, v)
        k = clamp(k, 1, length(counts))
        counts[k] += 1
    end
    hd = counts ./ (N .* diff(edges))
    return (hist=(edges=edges, density=hd), density=dens, cutoff=c, h=(hl, hr),
            f_left=d.conventional.f_left, f_right=d.conventional.f_right,
            statistic=t.statistic, pvalue=t.pvalue, n=N, limits=(lo, hi),
            level=float(level), p=p)
end

function _viz_rd_density_data(data::AbstractDataFrame, running::Symbol, t::DiagnosticTest;
                              kwargs...)
    require_columns(data, [running]; context="plot_rd_density")
    return _viz_rd_density_data(data[!, running], t; kwargs...)
end

_viz_rd_density_data(x, t; kwargs...) = _viz_unsupported("plot_rd_density", t)

"""Local polynomial density at `x0` (slope of the CDF regression) and its s.e."""
function _viz_lpdensity(xs, F, onside, x0, h, p, kernel)
    N = length(xs)
    win = [i for i in eachindex(xs) if onside[i] && abs(xs[i] - x0) <= h]
    length(win) >= p + 2 || return (NaN, NaN)
    u = (xs[win] .- x0) ./ h
    w = if kernel === :uniform
        fill(0.5 / h, length(u))
    elseif kernel === :triangular
        (1 .- abs.(u)) ./ h
    else
        0.75 .* (1 .- u .^ 2) ./ h
    end
    X = [u[i]^k for i in eachindex(u), k in 0:p]
    XW = X .* w
    S = XW' * X
    Sinv = try
        inv(S)
    catch err
        err isa SingularException || rethrow()
        return (NaN, NaN)
    end
    all(isfinite, Sinv) || return (NaN, NaN)
    β = Sinv * (XW' * F[win])
    f = β[2] / h
    # influence of observation i on Σ_j XW_j F_j: XW_j / (N-1) for every window
    # observation j at or above x_i (ties share the CDF value)
    L = zeros(N, p + 1)
    acc = zeros(p + 1)
    wpos = length(win)
    for i in N:-1:1
        while wpos >= 1 && xs[win[wpos]] >= xs[i]
            acc .+= view(XW, wpos, :)
            wpos -= 1
        end
        L[i, :] = acc ./ (N - 1)
    end
    L .-= mean(L; dims=1)
    V = Sinv * (L' * L) * Sinv
    v = V[2, 2]
    return (f, v >= 0 ? sqrt(v) / h : NaN)
end

# ---------------------------------------------------------------------------------
# Honest DiD sensitivity
# ---------------------------------------------------------------------------------

"""
    _viz_honest_data(r::HonestDiDResult; breakdown=nothing)

NamedTuple `(table, original, estimate, breakdown, mname, level, method, delta)`:
`table` has columns `M`, `lb`, `ub`, `rejected::Bool` (the data reject `Δ(M)`, empty
set). `breakdown` is the given value (e.g. from [`honest_breakdown`](@ref)) or the
grid-based `r.breakdown`; `breakdown_source` says which.
"""
function _viz_honest_data(r::HonestDiDResult; breakdown=nothing)
    o = sortperm(r.M)
    rej = isnan.(r.lb[o]) .| isnan.(r.ub[o])
    tab = DataFrame(M=r.M[o], lb=r.lb[o], ub=r.ub[o], rejected=rej)
    bd, src = breakdown === nothing ? (r.breakdown, :grid) : (Float64(breakdown), :given)
    mname = r.restriction === :smoothness ? "M" : "M̄"
    return (table=tab, original=r.original, estimate=r.estimate, breakdown=bd,
            breakdown_source=src, mname=mname, level=r.level, method=r.method,
            delta=r.delta, restriction=r.restriction)
end

_viz_honest_data(x; kwargs...) = _viz_unsupported("plot_honest_did", x)

# ---------------------------------------------------------------------------------
# Judge designs: first stage against leniency
# ---------------------------------------------------------------------------------

"""
    _viz_judge_data(data, treatment, leniency; nbins=20, level=0.95, trim=0.01)

Binned first stage of a judge design: cases are grouped into `nbins` equal-count
bins of the leniency instrument (after trimming the `trim` tails, as in Dobbie,
Goldin & Yang 2018) and the treatment rate with its `level` interval is computed per
bin. Also returns the OLS line of the treatment on leniency over all cases (`fit`:
`intercept`, `slope`, `se_slope`, heteroskedasticity-robust) and the trimmed
leniency values for the histogram.
"""
function _viz_judge_data(data::AbstractDataFrame, treatment::Symbol,
                         leniency::AbstractVector; nbins::Integer=20, level::Real=0.95,
                         trim::Real=0.01)
    require_columns(data, [treatment]; context="plot_judge_first_stage")
    length(leniency) == nrow(data) || throw(ArgumentError(
        "plot_judge_first_stage: leniency has $(length(leniency)) entries, data has " *
        "$(nrow(data)) rows"))
    nbins >= 2 || throw(ArgumentError("nbins must be at least 2"))
    0 <= trim < 0.5 || throw(ArgumentError("trim must be in [0, 0.5)"))
    ok = [!ismissing(leniency[i]) && !ismissing(data[i, treatment]) &&
          isfinite(leniency[i]) for i in 1:nrow(data)]
    z = Float64.(leniency[ok])
    dd = Float64.(data[ok, treatment])
    length(z) >= 2 * nbins || throw(ArgumentError(
        "plot_judge_first_stage: too few cases for $nbins bins"))
    # OLS fit on all cases (robust s.e.)
    zc = z .- mean(z)
    slope = dot(zc, dd) / dot(zc, zc)
    icpt = mean(dd) - slope * mean(z)
    e = dd .- icpt .- slope .* z
    se = sqrt(sum(abs2, zc .* e)) / dot(zc, zc)
    qlo, qhi = quantile(z, [trim, 1 - trim])
    keep = (z .>= qlo) .& (z .<= qhi)
    zt, dt = z[keep], dd[keep]
    o = sortperm(zt)
    edges = round.(Int, range(0, length(zt); length=nbins + 1))
    bins = DataFrame(leniency=Float64[], rate=Float64[], conf_low=Float64[],
                     conf_high=Float64[], n=Int[])
    for b in 1:nbins
        ix = o[(edges[b] + 1):edges[b + 1]]
        isempty(ix) && continue
        m = mean(dt[ix])
        n = length(ix)
        h = n > 1 ? critical_value(level, n - 1) * std(dt[ix]) / sqrt(n) : 0.0
        push!(bins, (mean(zt[ix]), m, m - h, m + h, n))
    end
    return (bins=bins, fit=(intercept=icpt, slope=slope, se_slope=se),
            leniency=zt, n=length(z), level=float(level), trim=float(trim))
end

function _viz_judge_data(data::AbstractDataFrame, r::JudgeIVEstimate; kwargs...)
    return _viz_judge_data(data, r.treatment, r.leniency; kwargs...)
end

_viz_judge_data(data, x; kwargs...) = _viz_unsupported("plot_judge_first_stage", x)

# ---------------------------------------------------------------------------------
# Rotemberg weights
# ---------------------------------------------------------------------------------

"""
    _viz_rotemberg_data(r::RotembergDecomposition; label=5)

Rows with a share-specific estimate (`sector`, `alpha`, `abs_alpha`, `beta`,
`first_stage_F`, `shock`, `negative::Bool`, `labelled::Bool` for the `label` largest
`|α|`), the Bartik estimate and the weight summaries.
"""
function _viz_rotemberg_data(r::RotembergDecomposition; label::Integer=5)
    t = r.table
    ok = .!ismissing.(t.beta)
    s = t[ok, :]
    ord = sortperm(abs.(s.alpha); rev=true)
    lab = falses(nrow(s))
    lab[ord[1:min(label, nrow(s))]] .= true
    tab = DataFrame(sector=string.(s.sector), alpha=Float64.(s.alpha),
                    abs_alpha=abs.(Float64.(s.alpha)), beta=Float64.(s.beta),
                    first_stage_F=Float64.(s.first_stage_F), shock=Float64.(s.shock),
                    negative=s.alpha .< 0, labelled=lab)
    return (table=tab, estimate=r.estimate, negative_weight_sum=r.negative_weight_sum,
            positive_weight_sum=r.positive_weight_sum, top5_share=r.top5_share,
            n_dropped=count(!, ok))
end

_viz_rotemberg_data(x; kwargs...) = _viz_unsupported("plot_rotemberg", x)

# ---------------------------------------------------------------------------------
# Marginal treatment effects
# ---------------------------------------------------------------------------------

"""
    _viz_mte_data(r::MTEEstimate; parameters=true)

NamedTuple `(curve, support, parameters, propensity, method, level)`: the stored MTE
curve (`u`, `mte`, `lower`, `upper`), the common support of the propensity score,
the summary parameters (`tidy` rows: `term`, `estimate`, `conf_low`, `conf_high`;
empty when `parameters = false`) and the non-missing propensity scores.
"""
function _viz_mte_data(r::MTEEstimate; parameters::Bool=true)
    c = r.curve
    curve = DataFrame(u=Float64.(c.u), mte=Float64.(c.mte), lower=Float64.(c.lower),
                      upper=Float64.(c.upper))
    pars = parameters ? tidy(r; level=r.level)[:, [:term, :estimate, :conf_low,
                                                   :conf_high]] :
           DataFrame(term=String[], estimate=Float64[], conf_low=Float64[],
                     conf_high=Float64[])
    return (curve=curve, support=(lower=Float64(r.support.lower),
                                  upper=Float64(r.support.upper)),
            parameters=pars, propensity=Float64.(collect(skipmissing(r.propensity))),
            method=r.method, level=r.level)
end

_viz_mte_data(x; kwargs...) = _viz_unsupported("plot_mte", x)

# ---------------------------------------------------------------------------------
# Synthetic control: in-time placebo (backdating)
# ---------------------------------------------------------------------------------

"""
    _viz_synth_in_time_data(r, backdated)

Treated path over the full sample, the original synthetic control, and the
backdated synthetic control (weights of `backdated` =
[`synth_in_time_placebo`](@ref)`(r, t₀)` applied to the donors' outcomes in every
period, so the backdated counterfactual is extended past the placebo date and the
actual treatment date). Returns `(table, placebo_time, treatment_time, x_placebo,
x_treatment, ticks)` with `table` columns `time`, `x`, `treated`, `synthetic`,
`backdated`.
"""
function _viz_synth_in_time_data(r::SyntheticControlEstimate,
                                 b::SyntheticControlEstimate)
    p = r.panel
    row = Dict(u => i for (i, u) in enumerate(p.units))
    string(b.treated_unit) == string(r.treated_unit) || throw(ArgumentError(
        "plot_synth_in_time: the two results have different treated units"))
    w = synth_weights(b)
    T = length(r.treated_path)
    syn_b = zeros(T)
    for (u, wt) in zip(w.unit, w.weight)
        haskey(row, u) || throw(ArgumentError(
            "plot_synth_in_time: donor $u of the backdated fit is not in the original " *
            "panel"))
        syn_b .+= wt .* p.Y[row[u], 1:T]
    end
    times = p.times[1:T]
    xs = _viz_time_x(times)
    tb = b.panel.times[b.n_pre + 1]
    tr = times[r.n_pre + 1]
    kb = findfirst(==(tb), times)
    tab = DataFrame(time=times, x=xs, treated=r.treated_path, synthetic=r.synthetic_path,
                    backdated=syn_b)
    ticks = eltype(times) <: Real ? nothing : (xs, string.(times))
    return (table=tab, placebo_time=tb, treatment_time=tr, x_placebo=xs[kb],
            x_treatment=xs[r.n_pre + 1], ticks=ticks)
end

_viz_synth_in_time_data(r, b) = _viz_unsupported("plot_synth_in_time", r)
