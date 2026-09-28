# Local randomization inference for RD designs (Cattaneo, Frandsen & Titiunik 2015;
# Cattaneo, Titiunik & Vazquez-Bare 2016, 2017): Fisher randomization tests in a small
# window around the cutoff, and covariate-balance-based window selection.

function _rd_lr_stat(y::AbstractVector, z::AbstractVector{Bool}, statistic::Symbol)
    yt = y[z]
    yc = y[.!z]
    if statistic === :diffmeans
        return mean(yt) - mean(yc)
    elseif statistic === :ks
        pts = sort(unique(y))
        ft = [count(<=(v), yt) / length(yt) for v in pts]
        fc = [count(<=(v), yc) / length(yc) for v in pts]
        return maximum(abs.(ft .- fc))
    else  # :ranksum, standardized Wilcoxon rank-sum statistic
        r = tiedrank(y)
        nt, n = length(yt), length(y)
        W = sum(r[z])
        EW = nt * (n + 1) / 2
        VW = nt * (n - nt) / n * var(r; corrected=true)
        return (W - EW) / sqrt(VW)
    end
end

function _rd_lr_window(x, c, window)
    wl, wr = _rd_pair(window)
    (wl > 0 && wr > 0) || throw(ArgumentError("window half-widths must be positive"))
    return wl, wr, (x .>= c - wl) .& (x .<= c + wr)
end

function _rd_lr_draws(y, z, statistic, reps, rng)
    m = CompleteRandomization(length(z), count(z))
    seeds = task_seeds(rng, reps)
    draws = Vector{Float64}(undef, reps)
    Threads.@threads for b in 1:reps
        zb = draw_assignment(Xoshiro(seeds[b]), m)
        draws[b] = _rd_lr_stat(y, Vector{Bool}(zb), statistic)
    end
    return draws
end

"""
    rd_randomization_test(data, outcome, running; window, cutoff=0.0,
                          statistic=:diffmeans, reps=999, level=0.95,
                          rng=Random.default_rng()) -> DiagnosticTest

Local randomization inference for an RD design (as in `rdrandinf`): a Fisher
randomization test of the sharp null of no effect for any unit in a small window around
the cutoff.

The local randomization framework of Cattaneo, Frandsen and Titiunik (2015) takes
literally the idea that, very close to the cutoff, which side a unit lands on is as good
as random (Lee 2008). It rests on two assumptions about a window
``W = [c - w_\\ell, c + w_r]`` (Cattaneo, Titiunik & Vazquez-Bare 2017; Cattaneo,
Idrobo & Titiunik 2024). First, the treatment assignment ``Z_i = 1\\{X_i \\ge c\\}`` of
the units in ``W`` follows a known mechanism, here complete randomization with the
observed numbers of treated and control units. Second, an *exclusion* condition: within
``W`` the potential outcomes do not depend on the running variable, so ``Y_i(0)`` and
``Y_i(1)`` are unaffected by where in the window ``X_i`` falls. The second condition is
stronger than continuity and fails if the outcome trends with the score inside the
window. It is plausible only in narrow windows, or after a model has adjusted outcomes
for the score. Neither condition can be tested directly.

Under the sharp null ``H_0: Y_i(1) = Y_i(0)`` for all ``i \\in W``, the outcomes are
fixed, and the distribution of any statistic follows from re-drawing assignments. The
test compares the observed difference in means, Kolmogorov–Smirnov statistic or
standardised Wilcoxon rank-sum statistic with `reps` re-randomisations and reports
``p = (1 + \\#\\{b: |T_b| \\ge |T_{\\text{obs}}|\\})/(1 + \\text{reps})``. Given the
window, this p-value is valid in finite samples. `details` also reports a large-sample
Neyman analysis of the average effect in the window (difference in means, conservative
standard error, normal interval), which tests a weaker null than the sharp one.

The inference is conditional on the window, and the window is usually chosen from the
same data ([`rd_window_selection`](@ref)). The reported p-value does not account for
that selection, so report results for several windows. A rejection is evidence of an
effect for some units in the window, under the assumptions. A non-rejection of the
sharp null is not evidence that treatment has no effect: the windows are small and the
test may have little power. The estimand, an effect for units in ``W``, differs from
the effect at the cutoff targeted by the continuity-based [`rd_estimate`](@ref). The
two approaches complement each other, and local randomization is most useful when the
running variable is discrete or the sample near the cutoff is small (Cattaneo, Idrobo &
Titiunik 2024).

# Arguments
- `data::AbstractDataFrame`: one row per unit. Rows with a missing outcome or running
  variable are dropped.
- `outcome::Symbol`: outcome column.
- `running::Symbol`: running variable. Units in the window with `running ≥ cutoff` are
  treated.

# Keywords
- `window`: half-width of the window around the cutoff, as a scalar or a
  `(left, right)` pair (required). The window must contain at least two units on each
  side.
- `cutoff::Real=0.0`: the RD threshold.
- `statistic::Symbol=:diffmeans`: `:diffmeans` (difference in means), `:ks`
  (Kolmogorov–Smirnov, sensitive to differences anywhere in the distribution) or
  `:ranksum` (standardised Wilcoxon rank sum, robust to outliers).
- `reps::Integer=999`: number of re-randomisations. The smallest attainable p-value is
  `1/(reps + 1)`.
- `level::Real=0.95`: level of the Neyman interval in `details`.
- `rng::AbstractRNG=Random.default_rng()`: random number generator. Per-draw seeds are
  drawn up front, so results do not depend on the number of threads.

# Returns
- `DiagnosticTest` whose statistic is the observed test statistic and whose p-value is
  the randomization p-value. `details` holds `estimate` (difference in means),
  `n_left`, `n_right` (units in the window below and above the cutoff), `window_left`,
  `window_right`, `neyman_se`, `neyman_pvalue`, `neyman_ci`, `statistic` and `reps`.

# Examples
```julia
using DrSnow, CSV, DataFrames, StableRNGs
senate = CSV.read(pkgdir(DrSnow, "test", "validation", "rd", "senate.csv"), DataFrame)
t = rd_randomization_test(senate, :vote, :margin; window=2.5, rng=StableRNG(1))
t.pvalue, t.details.estimate, t.details.neyman_ci
rd_randomization_test(senate, :vote, :margin; window=1.0, statistic=:ranksum,
                      rng=StableRNG(1))
```

# References
- Cattaneo, M. D., Frandsen, B. R., & Titiunik, R. (2015). Randomization inference in
  the regression discontinuity design: An application to party advantages in the U.S.
  Senate. *Journal of Causal Inference*, 3(1), 1–24.
- Cattaneo, M. D., Titiunik, R., & Vazquez-Bare, G. (2017). Comparing inference
  approaches for RD designs: A reexamination of the effect of Head Start on child
  mortality. *Journal of Policy Analysis and Management*, 36(3), 643–681.
- Cattaneo, M. D., Titiunik, R., & Vazquez-Bare, G. (2016). Inference in regression
  discontinuity designs under local randomization. *The Stata Journal*, 16(2), 331–367.
- Cattaneo, M. D., Idrobo, N., & Titiunik, R. (2024). *A Practical Introduction to
  Regression Discontinuity Designs: Extensions*. Cambridge University Press.
- Lee, D. S. (2008). Randomized experiments from non-random selection in U.S. House
  elections. *Journal of Econometrics*, 142(2), 675–697.
"""
function rd_randomization_test(data::AbstractDataFrame, outcome::Symbol, running::Symbol;
                               window, cutoff::Real=0.0, statistic::Symbol=:diffmeans,
                               reps::Integer=999, level::Real=0.95,
                               rng::AbstractRNG=Random.default_rng())
    ctx = "rd_randomization_test"
    statistic in (:diffmeans, :ks, :ranksum) || throw(ArgumentError(
        "$ctx: statistic must be :diffmeans, :ks or :ranksum"))
    reps >= 1 || throw(ArgumentError("$ctx: reps must be positive"))
    E = _rd_extract(data, outcome, running; context=ctx)
    c = Float64(cutoff)
    wl, wr, inw = _rd_lr_window(E.x, c, window)
    y = E.y[inw]
    z = E.x[inw] .>= c
    nt, nc = count(z), count(.!z)
    (nt >= 2 && nc >= 2) || throw(ArgumentError(
        "$ctx: the window must contain at least two observations on each side " *
        "(found $nc below and $nt above the cutoff)"))
    obs = _rd_lr_stat(y, z, statistic)
    draws = _rd_lr_draws(y, z, statistic, Int(reps), rng)
    pv = permutation_pvalue(obs, draws)
    est = mean(y[z]) - mean(y[.!z])
    se = sqrt(var(y[z]) / nt + var(y[.!z]) / nc)
    q = critical_value(level)
    details = (estimate=est, n_left=nc, n_right=nt, window_left=wl, window_right=wr,
               neyman_se=se, neyman_pvalue=two_sided_pvalue(est / se),
               neyman_ci=(est - q * se, est + q * se), statistic=statistic,
               reps=Int(reps))
    return DiagnosticTest("Local randomization RD test (Fisher)",
                          "no unit in the window is affected by treatment (sharp null)",
                          obs, pv;
                          method="randomization inference ($reps draws, complete " *
                                 "randomization within [$(c - wl), $(c + wr)], " *
                                 "statistic = $statistic)",
                          note="Valid only if treatment is as-if randomly assigned " *
                               "within the window; the conclusion depends on the " *
                               "window. Non-rejection of the sharp null does not show " *
                               "that the effect is zero.",
                          details=details)
end

"""
    rd_window_selection(data, covariates, running; cutoff=0.0, windows=nothing,
                        nwindows=10, obs_min=10, obs_step=5, level=0.15,
                        statistic=:diffmeans, reps=999,
                        rng=Random.default_rng()) -> NamedTuple

Covariate-balance-based window selection for local randomization RD inference (as in
`rdwinselect`).

Local randomization inference ([`rd_randomization_test`](@ref)) needs a window around the
cutoff in which assignment is as good as random. Cattaneo, Frandsen and Titiunik (2015)
propose to choose it from predetermined covariates. If assignment is as good as random
in a window, covariates should be balanced there. Balance typically deteriorates as the
window grows and the running variable, which is correlated with the covariates, starts
to differ between the two groups. The procedure has four steps.
1. Form a sequence of nested symmetric windows. By default, the ``j``-th window is the
   smallest one containing `obs_min + (j − 1)·obs_step` units on each side.
2. In each window, test balance of every covariate with a randomization test.
3. Record the smallest p-value across covariates in each window.
4. Select the largest window such that the minimum p-value is at least `level` in it and
   in every smaller window.

**Selection by non-rejection.** The rule treats a non-rejection as a license to use the
window. It therefore uses a high threshold, `level = 0.15` by default, following
Cattaneo, Frandsen and Titiunik (2015) and Cattaneo, Titiunik and Vazquez-Bare (2017).
Here the costly error is failing to detect imbalance, and a larger level makes the
procedure more conservative, selecting smaller windows. Non-rejection is weak evidence,
especially in the smallest windows, where few units give the tests little power. Balance
in observed covariates is necessary but not sufficient for local randomization and says
nothing about the exclusion condition that outcomes do not depend on the score within
the window. Inference in the selected window is **not adjusted for the selection**:
p-values from [`rd_randomization_test`](@ref) treat the window as fixed. Treat the
selected window as a starting point, and report results for it and for nearby windows
(Cattaneo, Idrobo & Titiunik 2024).

# Arguments
- `data::AbstractDataFrame`: one row per unit.
- `covariates::AbstractVector`: predetermined covariates to test (at least one; missing
  values are dropped covariate by covariate).
- `running::Symbol`: running variable.

# Keywords
- `cutoff::Real=0.0`: the RD threshold.
- `windows=nothing`: explicit vector of half-widths. If given, `nwindows`, `obs_min` and
  `obs_step` are ignored.
- `nwindows::Integer=10`: number of windows in the default sequence (fewer if the data
  run out).
- `obs_min::Integer=10`: units on each side in the smallest window.
- `obs_step::Integer=5`: increase in units on each side from one window to the next.
- `level::Real=0.15`: threshold that the minimum balance p-value must reach.
- `statistic::Symbol=:diffmeans`, `reps::Integer=999`: test statistic and number of
  re-randomisations of each balance test, as in [`rd_randomization_test`](@ref).
- `rng::AbstractRNG=Random.default_rng()`: random number generator. Seeds for all tests
  are drawn up front.

# Returns
- `NamedTuple` with fields
  - `table::DataFrame`: one row per window, with `window`, `n_left`, `n_right`,
    `min_pvalue` and `min_pvalue_covariate`;
  - `window`: the selected half-width, or `nothing` if even the smallest window shows
    imbalance at `level`.

# Examples
```julia
using DrSnow, CSV, DataFrames, StableRNGs
senate = CSV.read(pkgdir(DrSnow, "test", "validation", "rd", "senate.csv"), DataFrame)
sel = rd_window_selection(senate, [:presdemvoteshlag1, :demvoteshlag1], :margin;
                          reps=499, rng=StableRNG(1))
sel.table
sel.window
```

# References
- Cattaneo, M. D., Frandsen, B. R., & Titiunik, R. (2015). Randomization inference in
  the regression discontinuity design: An application to party advantages in the U.S.
  Senate. *Journal of Causal Inference*, 3(1), 1–24.
- Cattaneo, M. D., Titiunik, R., & Vazquez-Bare, G. (2017). Comparing inference
  approaches for RD designs: A reexamination of the effect of Head Start on child
  mortality. *Journal of Policy Analysis and Management*, 36(3), 643–681.
- Cattaneo, M. D., Titiunik, R., & Vazquez-Bare, G. (2016). Inference in regression
  discontinuity designs under local randomization. *The Stata Journal*, 16(2), 331–367.
- Cattaneo, M. D., Idrobo, N., & Titiunik, R. (2024). *A Practical Introduction to
  Regression Discontinuity Designs: Extensions*. Cambridge University Press.
"""
function rd_window_selection(data::AbstractDataFrame, covariates::AbstractVector,
                             running::Symbol; cutoff::Real=0.0, windows=nothing,
                             nwindows::Integer=10, obs_min::Integer=10,
                             obs_step::Integer=5, level::Real=0.15,
                             statistic::Symbol=:diffmeans, reps::Integer=999,
                             rng::AbstractRNG=Random.default_rng())
    ctx = "rd_window_selection"
    isempty(covariates) && throw(ArgumentError("$ctx: no covariates given"))
    covs = Symbol.(covariates)
    require_columns(data, vcat(covs, running); context=ctx)
    c = Float64(cutoff)
    xall = collect(skipmissing(data[!, running]))
    if windows === nothing
        dl = sort(c .- filter(<(c), xall))
        dr = sort(filter(>=(c), xall) .- c)
        windows = Float64[]
        for j in 1:nwindows
            k = obs_min + (j - 1) * obs_step
            (k > length(dl) || k > length(dr)) && break
            push!(windows, max(dl[k], dr[k]))
        end
        isempty(windows) && throw(ArgumentError(
            "$ctx: fewer than obs_min observations on one side of the cutoff"))
    end
    windows = sort(unique(Float64.(windows)))
    seeds = task_seeds(rng, length(windows) * length(covs))
    rows = []
    k = 0
    for w in windows
        minp, argmin_cov = Inf, :none
        nl = nr = 0
        for z in covs
            k += 1
            sub = dropmissing(data[:, [z, running]])
            t = rd_randomization_test(sub, z, running; window=w, cutoff=c,
                                      statistic=statistic, reps=reps,
                                      rng=Xoshiro(seeds[k]))
            if t.pvalue < minp
                minp, argmin_cov = t.pvalue, z
            end
            nl, nr = t.details.n_left, t.details.n_right
        end
        push!(rows, (window=w, n_left=nl, n_right=nr, min_pvalue=minp,
                     min_pvalue_covariate=argmin_cov))
    end
    table = DataFrame(rows)
    chosen = nothing
    for row in eachrow(table)
        row.min_pvalue >= level || break
        chosen = row.window
    end
    return (table=table, window=chosen)
end
