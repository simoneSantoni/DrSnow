# Falsification and sensitivity analyses for RD designs (Cattaneo, Idrobo & Titiunik
# 2020, Section 5): covariate balance at the cutoff, placebo cutoffs, donut-hole
# estimates, and bandwidth sensitivity.

function _rd_result_row(r::RDEstimate)
    ci = confint(r)
    return (estimate=r.tau_bias_corrected, se_robust=r.se_robust,
            pvalue=pvalues(r)[1], ci_lower=ci[1, 1], ci_upper=ci[1, 2],
            estimate_conventional=r.tau_conventional, h_left=r.h_left,
            h_right=r.h_right, n_h_left=r.n_h_left, n_h_right=r.n_h_right)
end

const _RD_EMPTY_ROW = (estimate=missing, se_robust=missing, pvalue=missing,
                       ci_lower=missing, ci_upper=missing, estimate_conventional=missing,
                       h_left=missing, h_right=missing, n_h_left=missing,
                       n_h_right=missing)

"""Holm (1979) step-down adjusted p-values (valid under arbitrary dependence)."""
function _rd_holm(p::AbstractVector)
    m = length(p)
    m == 0 && return Float64[]
    o = sortperm(p)
    adj = similar(p, Float64)
    running = 0.0
    for (k, i) in enumerate(o)
        running = max(running, min(1.0, (m - k + 1) * p[i]))
        adj[i] = running
    end
    return adj
end

"""
    rd_covariate_balance(data, covariates, running; cutoff=0.0, level=0.95,
                         kwargs...) -> DataFrame

Falsification test of an RD design: the discontinuity at the cutoff in each
predetermined covariate, estimated as if the covariate were the outcome.

Continuity-based RD designs rest on the assumption that units just below and just above
the cutoff are comparable. Covariates that are fixed before treatment is assigned
(pre-treatment characteristics, lagged outcomes) cannot be affected by treatment, so
their conditional expectation should not jump at the cutoff. A jump is evidence of
sorting or of another policy that changes at the same threshold (Lee 2008; Lee &
Lemieux 2010). For each covariate ``Z`` this function estimates
``\\lim_{x \\downarrow c} E[Z \\mid X = x] - \\lim_{x \\uparrow c} E[Z \\mid X = x]``
with [`rd_estimate`](@ref). Each covariate gets its own data-driven bandwidth,
because the covariate regressions differ in curvature from the outcome regression. The
robust bias-corrected inference of Calonico, Cattaneo and Titiunik (2014) is reported,
as recommended by Cattaneo, Idrobo and Titiunik (2020, Section 5).

With several covariates, some nominal rejections are expected by chance. The table
therefore adds Holm (1979) step-down adjusted p-values, which control the family-wise
error rate under arbitrary dependence between the tests. Interpret the results
asymmetrically. A clear discontinuity undermines the design. The absence of detected
discontinuities is not evidence that the design is valid: the tests may lack power
near the cutoff, and balance in observed covariates says nothing about unobserved
ones. Report point estimates and intervals, not only p-values, alongside the density
test ([`rd_density_test`](@ref)). Covariates that pass the test can then be used for
precision in [`rd_estimate`](@ref) or [`rd_flex`](@ref). Under local randomization, the
analogous check is [`rd_window_selection`](@ref).

# Arguments
- `data::AbstractDataFrame`: one row per unit.
- `covariates::AbstractVector`: names of the predetermined covariates to test (each is
  used as an outcome; missing values are dropped covariate by covariate).
- `running::Symbol`: running variable.

# Keywords
- `cutoff::Real=0.0`: the RD threshold.
- `level::Real=0.95`: confidence level of the reported intervals.
- `kwargs...`: passed to [`rd_estimate`](@ref), for example `p`, `kernel`, `bwselect`,
  `vce`, `cluster` or `h`. The keyword `covariates` is not accepted here, because
  covariate adjustment has no role in a balance test.

# Returns
- `DataFrame` with one row per covariate and columns `covariate`, `estimate`
  (bias-corrected jump), `se_robust`, `pvalue`, `pvalue_holm`, `ci_lower`, `ci_upper`
  (robust interval), `estimate_conventional`, `h_left`, `h_right`, `n_h_left`,
  `n_h_right`, and `note` (the reason when a row could not be computed, with `missing`
  estimates). The table metadata key `"note"` repeats the interpretation caveats.

# Examples
```julia
using DrSnow, CSV, DataFrames
senate = CSV.read(pkgdir(DrSnow, "test", "validation", "rd", "senate.csv"), DataFrame)
bal = rd_covariate_balance(senate, [:presdemvoteshlag1, :demvoteshlag1, :population],
                           :margin)
bal[:, [:covariate, :estimate, :ci_lower, :ci_upper, :pvalue_holm]]
```

# References
- Lee, D. S. (2008). Randomized experiments from non-random selection in U.S. House
  elections. *Journal of Econometrics*, 142(2), 675–697.
- Lee, D. S., & Lemieux, T. (2010). Regression discontinuity designs in economics.
  *Journal of Economic Literature*, 48(2), 281–355.
- Calonico, S., Cattaneo, M. D., & Titiunik, R. (2014). Robust nonparametric confidence
  intervals for regression-discontinuity designs. *Econometrica*, 82(6), 2295–2326.
- Cattaneo, M. D., Idrobo, N., & Titiunik, R. (2020). *A Practical Introduction to
  Regression Discontinuity Designs: Foundations*. Cambridge University Press.
- Holm, S. (1979). A simple sequentially rejective multiple test procedure.
  *Scandinavian Journal of Statistics*, 6(2), 65–70.
"""
function rd_covariate_balance(data::AbstractDataFrame, covariates::AbstractVector,
                              running::Symbol; cutoff::Real=0.0, level::Real=0.95,
                              kwargs...)
    isempty(covariates) &&
        throw(ArgumentError("rd_covariate_balance: no covariates given"))
    haskey(kwargs, :covariates) && throw(ArgumentError(
        "rd_covariate_balance: `covariates` are the outcomes here; covariate " *
        "adjustment is not used for balance tests"))
    covs = Symbol.(covariates)
    require_columns(data, vcat(covs, running); context="rd_covariate_balance")
    rows = []
    for z in covs
        row = try
            r = rd_estimate(data, z, running; cutoff=cutoff, level=level, kwargs...)
            merge(_rd_result_row_level(r, level), (note="",))
        catch err
            err isa ArgumentError || rethrow()
            merge(_RD_EMPTY_ROW, (note=sprint(showerror, err),))
        end
        push!(rows, merge((covariate=z,), row))
    end
    df = DataFrame(rows)
    ok = .!ismissing.(df.pvalue)
    holm = Vector{Union{Missing,Float64}}(missing, nrow(df))
    holm[ok] = _rd_holm(Float64.(df.pvalue[ok]))
    insertcols!(df, :pvalue, :pvalue_holm => holm; after=true)
    metadata!(df, "note",
              "Discontinuities in predetermined covariates are evidence against the " *
              "design. Holm-adjusted p-values control the family-wise error rate " *
              "across covariates. Non-rejection does not establish validity (limited " *
              "power; unobserved covariates are untested).")
    return df
end

function _rd_result_row_level(r::RDEstimate, level)
    row = _rd_result_row(r)
    ci = confint(r; level=level)
    return merge(row, (ci_lower=ci[1, 1], ci_upper=ci[1, 2]))
end

"""
    rd_placebo_cutoffs(data, outcome, running; cutoff=0.0, placebo_cutoffs=nothing,
                       level=0.95, kwargs...) -> DataFrame

Placebo (artificial) cutoff analysis: RD estimates at thresholds where treatment does
not change.

The causal reading of a jump at the true cutoff presumes that the regression functions
would otherwise be continuous there. That assumption cannot be tested at ``c`` itself,
but a design in which ``E[Y \\mid X = x]`` jumps at many points where nothing happens
would make a jump at ``c`` less convincing. This function estimates the RD effect at
each placebo cutoff ``\\tilde c \\ne c`` with [`rd_estimate`](@ref), where the true
effect is zero by construction. To keep the true treatment effect from contaminating
the placebo estimates, a placebo cutoff below ``c`` uses only control units
(``X < c``) and one above ``c`` uses only treated units (``X \\ge c``), as recommended
by Cattaneo, Idrobo and Titiunik (2020, Section 5). Each estimate has its own
data-driven bandwidth and robust bias-corrected interval. Imbens and Lemieux (2008)
suggest the medians of the running variable on each side as natural placebo cutoffs,
and the default adds the quartiles.

Several placebo discontinuities, especially near ``c``, suggest that the regression
functions are not smooth at the scale of the bandwidth, which weakens the case for the
estimate at the true cutoff. As with any falsification test, finding no placebo effects
does not validate the design. The tests may have little power, and continuity at other
points does not imply continuity at ``c``. With several placebo cutoffs, about
``\\alpha`` of them are expected to reject by chance. Placebo cutoffs close to ``c``
leave few observations on the side that borders the true cutoff; such rows can fail
and are reported with a `note`.

# Arguments
- `data::AbstractDataFrame`: one row per unit.
- `outcome::Symbol`: outcome column.
- `running::Symbol`: running variable.

# Keywords
- `cutoff::Real=0.0`: the true cutoff.
- `placebo_cutoffs=nothing`: vector of placebo cutoffs (none may equal `cutoff`). By
  default, the 25th, 50th and 75th percentiles of the running variable on each side of
  the true cutoff.
- `level::Real=0.95`: confidence level of the reported intervals.
- `kwargs...`: passed to [`rd_estimate`](@ref) (for example `p`, `kernel`, `bwselect`,
  `cluster`).

# Returns
- `DataFrame` with one row per placebo cutoff and columns `cutoff`, `side`
  (`:control` or `:treated`), `estimate` (bias-corrected), `se_robust`, `pvalue`,
  `ci_lower`, `ci_upper`, `estimate_conventional`, `h_left`, `h_right`, `n_h_left`,
  `n_h_right`, and `note` (the reason when an estimate could not be computed).

# Examples
```julia
using DrSnow, CSV, DataFrames
senate = CSV.read(pkgdir(DrSnow, "test", "validation", "rd", "senate.csv"), DataFrame)
rd_placebo_cutoffs(senate, :vote, :margin; placebo_cutoffs=[-20, -10, 10, 20])
```

# References
- Imbens, G. W., & Lemieux, T. (2008). Regression discontinuity designs: A guide to
  practice. *Journal of Econometrics*, 142(2), 615–635.
- Cattaneo, M. D., Idrobo, N., & Titiunik, R. (2020). *A Practical Introduction to
  Regression Discontinuity Designs: Foundations*. Cambridge University Press.
- Calonico, S., Cattaneo, M. D., & Titiunik, R. (2014). Robust nonparametric confidence
  intervals for regression-discontinuity designs. *Econometrica*, 82(6), 2295–2326.
"""
function rd_placebo_cutoffs(data::AbstractDataFrame, outcome::Symbol, running::Symbol;
                            cutoff::Real=0.0, placebo_cutoffs=nothing, level::Real=0.95,
                            kwargs...)
    require_columns(data, [outcome, running]; context="rd_placebo_cutoffs")
    xr = collect(skipmissing(data[!, running]))
    c = Float64(cutoff)
    if placebo_cutoffs === nothing
        xl = filter(<(c), xr); xg = filter(>=(c), xr)
        (isempty(xl) || isempty(xg)) && throw(ArgumentError(
            "rd_placebo_cutoffs: no observations on one side of the cutoff"))
        placebo_cutoffs = vcat(quantile(xl, [0.25, 0.5, 0.75]),
                               quantile(xg, [0.25, 0.5, 0.75]))
    end
    rows = []
    for pc in placebo_cutoffs
        pc == c && throw(ArgumentError("rd_placebo_cutoffs: placebo cutoff equals the " *
                                       "true cutoff"))
        side = pc < c ? :control : :treated
        keep = [!ismissing(v) && (side === :control ? v < c : v >= c)
                for v in data[!, running]]
        sub = data[keep, :]
        row = try
            r = rd_estimate(sub, outcome, running; cutoff=pc, level=level, kwargs...)
            merge(_rd_result_row_level(r, level), (note="",))
        catch err
            err isa ArgumentError || rethrow()
            merge(_RD_EMPTY_ROW, (note=sprint(showerror, err),))
        end
        push!(rows, merge((cutoff=Float64(pc), side=side), row))
    end
    return DataFrame(rows)
end

"""
    rd_donut(data, outcome, running; radii, cutoff=0.0, level=0.95,
             kwargs...) -> DataFrame

Donut-hole RD: re-estimates the effect after excluding the observations closest to the
cutoff, for a sequence of hole radii.

Units whose score lies closest to the cutoff are the most likely to have manipulated it,
and are also where heaping of the running variable at round numbers can bias local
polynomial estimates (Barreca, Guldi, Lindo & Waddell 2011; Barreca, Lindo & Waddell
2016). Yet those same observations carry the most weight in a local polynomial
estimate. For each radius ``r`` this function drops the units with ``|X - c| < r`` and
re-estimates the effect with [`rd_estimate`](@ref), re-selecting the bandwidths
unless `h` is given. The table shows how much the estimate depends on the units nearest
the threshold.

Stable estimates as the hole widens are reassuring about sorting at the cutoff, but the
comparison is not a formal test. Estimates with a hole extrapolate the regression
functions over the excluded interval, so their bias grows with ``r`` and their intervals
widen as observations are lost (Cattaneo, Idrobo & Titiunik 2020, Section 5). Changes
across radii mix possible manipulation, extrapolation bias and sampling noise. Use
small radii relative to the bandwidth and include `0` as the baseline. A radius that
leaves too few observations near the cutoff produces a row with a `note` instead of an
estimate, or, if the sample on one side is exhausted, an error.

# Arguments
- `data::AbstractDataFrame`: one row per unit.
- `outcome::Symbol`: outcome column.
- `running::Symbol`: running variable.

# Keywords
- `radii`: vector of non-negative radii of the hole, in units of the running variable
  (required). Include `0` for the baseline estimate.
- `cutoff::Real=0.0`: the RD threshold.
- `level::Real=0.95`: confidence level of the reported intervals.
- `kwargs...`: passed to [`rd_estimate`](@ref).

# Returns
- `DataFrame` with one row per radius and columns `radius`, `n_excluded_left`,
  `n_excluded_right`, `estimate` (bias-corrected), `se_robust`, `pvalue`, `ci_lower`,
  `ci_upper`, `estimate_conventional`, `h_left`, `h_right`, `n_h_left`, `n_h_right`
  and `note`.

# Examples
```julia
using DrSnow, CSV, DataFrames
senate = CSV.read(pkgdir(DrSnow, "test", "validation", "rd", "senate.csv"), DataFrame)
rd_donut(senate, :vote, :margin; radii=[0, 0.5, 1, 2])
```

# References
- Barreca, A. I., Guldi, M., Lindo, J. M., & Waddell, G. R. (2011). Saving babies?
  Revisiting the effect of very low birth weight classification. *Quarterly Journal of
  Economics*, 126(4), 2117–2123.
- Barreca, A. I., Lindo, J. M., & Waddell, G. R. (2016). Heaping-induced bias in
  regression-discontinuity designs. *Economic Inquiry*, 54(1), 268–293.
- Cattaneo, M. D., Idrobo, N., & Titiunik, R. (2020). *A Practical Introduction to
  Regression Discontinuity Designs: Foundations*. Cambridge University Press.
"""
function rd_donut(data::AbstractDataFrame, outcome::Symbol, running::Symbol; radii,
                  cutoff::Real=0.0, level::Real=0.95, kwargs...)
    require_columns(data, [outcome, running]; context="rd_donut")
    all(>=(0), radii) || throw(ArgumentError("rd_donut: radii must be non-negative"))
    c = Float64(cutoff)
    xs = data[!, running]
    rows = []
    for r in radii
        excl = [!ismissing(v) && abs(v - c) < r for v in xs]
        nl = count(i -> excl[i] && xs[i] < c, eachindex(xs))
        nr = count(excl) - nl
        row = try
            est = rd_estimate(data[.!excl, :], outcome, running; cutoff=c, level=level,
                              kwargs...)
            merge(_rd_result_row_level(est, level), (note="",))
        catch err
            err isa ArgumentError || rethrow()
            merge(_RD_EMPTY_ROW, (note=sprint(showerror, err),))
        end
        push!(rows, merge((radius=Float64(r), n_excluded_left=nl, n_excluded_right=nr),
                          row))
    end
    return DataFrame(rows)
end

"""
    rd_bandwidth_sensitivity(data, outcome, running; cutoff=0.0,
                             multipliers=[0.5, 0.75, 1.0, 1.25, 1.5, 2.0],
                             bandwidths=nothing, level=0.95, kwargs...) -> DataFrame

Sensitivity of a local polynomial RD estimate to the choice of bandwidth.

The data-driven bandwidth of [`rd_bandwidth`](@ref) balances bias and variance
asymptotically, but in a given sample the estimate may depend noticeably on it. This
function re-estimates the effect over a grid of bandwidths. By default it first fits
[`rd_estimate`](@ref) with its data-driven bandwidths, then scales both the main
bandwidth ``h`` and the pilot bandwidth ``b`` by each multiplier. Alternatively, it
uses the main bandwidths in `bandwidths`, with ``b = h`` (the `rdrobust` default for
manual bandwidths) unless `rho` is passed.

Reading the table requires care (Cattaneo, Idrobo & Titiunik 2020, Section 5). Larger
bandwidths reduce variance but increase smoothing bias, so differences between rows mix
bias and sampling noise. The estimates are strongly correlated because they share
observations, so the rows are not independent replications. Robust bias-corrected
intervals remain valid for bandwidths that shrink at the MSE-optimal rate. They are not
valid for bandwidths much larger than the MSE-optimal one, because the bias then
dominates. Smaller bandwidths give intervals closer to nominal coverage but noisier
estimates. A pattern that is stable across moderate multipliers is reassuring. A trend
in the estimate as ``h`` grows usually reflects curvature of the regression functions,
not a treatment effect that varies with the bandwidth.

# Arguments
- `data::AbstractDataFrame`: one row per unit.
- `outcome::Symbol`: outcome column.
- `running::Symbol`: running variable.

# Keywords
- `cutoff::Real=0.0`: the RD threshold.
- `multipliers=[0.5, 0.75, 1.0, 1.25, 1.5, 2.0]`: positive factors applied to the
  data-driven `(h, b)`.
- `bandwidths=nothing`: explicit main bandwidths (scalars or `(left, right)` pairs). If
  given, they are used instead of `multipliers`.
- `level::Real=0.95`: confidence level of the reported intervals.
- `kwargs...`: passed to [`rd_estimate`](@ref), both for the baseline fit (for example
  `bwselect`) and for each row. `h` and `b` are not accepted, since the grid sets them.

# Returns
- `DataFrame` with one row per bandwidth and columns `multiplier` (`missing` for
  explicit bandwidths), `h_left`, `h_right`, `b_left`, `b_right`, `estimate`
  (bias-corrected), `se_robust`, `pvalue`, `ci_lower`, `ci_upper`,
  `estimate_conventional`, `se_conventional`, `n_h_left` and `n_h_right`.

# Examples
```julia
using DrSnow, CSV, DataFrames
senate = CSV.read(pkgdir(DrSnow, "test", "validation", "rd", "senate.csv"), DataFrame)
rd_bandwidth_sensitivity(senate, :vote, :margin)
rd_bandwidth_sensitivity(senate, :vote, :margin; bandwidths=[5, 10, 20, 40])
```

# References
- Cattaneo, M. D., Idrobo, N., & Titiunik, R. (2020). *A Practical Introduction to
  Regression Discontinuity Designs: Foundations*. Cambridge University Press.
- Calonico, S., Cattaneo, M. D., & Titiunik, R. (2014). Robust nonparametric confidence
  intervals for regression-discontinuity designs. *Econometrica*, 82(6), 2295–2326.
- Imbens, G. W., & Lemieux, T. (2008). Regression discontinuity designs: A guide to
  practice. *Journal of Econometrics*, 142(2), 615–635.
"""
function rd_bandwidth_sensitivity(data::AbstractDataFrame, outcome::Symbol,
                                  running::Symbol; cutoff::Real=0.0,
                                  multipliers=[0.5, 0.75, 1.0, 1.25, 1.5, 2.0],
                                  bandwidths=nothing, level::Real=0.95, kwargs...)
    for k in (:h, :b)
        haskey(kwargs, k) && throw(ArgumentError(
            "rd_bandwidth_sensitivity: use `bandwidths` or `multipliers`, not `$k`"))
    end
    settings = Tuple{Union{Missing,Float64},Any,Any}[]
    if bandwidths === nothing
        base = rd_estimate(data, outcome, running; cutoff=cutoff, level=level, kwargs...)
        for m in multipliers
            m > 0 || throw(ArgumentError("multipliers must be positive"))
            push!(settings, (Float64(m), (m * base.h_left, m * base.h_right),
                             (m * base.b_left, m * base.b_right)))
        end
    else
        for hh in bandwidths
            push!(settings, (missing, hh, nothing))
        end
    end
    kw = Dict{Symbol,Any}(kwargs)
    delete!(kw, :bwselect)
    rows = []
    for (m, hh, bb) in settings
        r = bb === nothing ?
            rd_estimate(data, outcome, running; cutoff=cutoff, level=level, h=hh, kw...) :
            rd_estimate(data, outcome, running; cutoff=cutoff, level=level, h=hh, b=bb,
                        kw...)
        ci = confint(r; level=level)
        push!(rows, (multiplier=m, h_left=r.h_left, h_right=r.h_right, b_left=r.b_left,
                     b_right=r.b_right, estimate=r.tau_bias_corrected,
                     se_robust=r.se_robust, pvalue=pvalues(r)[1], ci_lower=ci[1, 1],
                     ci_upper=ci[1, 2], estimate_conventional=r.tau_conventional,
                     se_conventional=r.se_conventional, n_h_left=r.n_h_left,
                     n_h_right=r.n_h_right))
    end
    return DataFrame(rows)
end
