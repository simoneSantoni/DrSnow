# Accessors shared by all synthetic-control-type results.

"""
    synth_weights(r) -> DataFrame

Donor (unit) weights of a synthetic-control-type estimate.

Each estimator of the area imputes the treated units' untreated outcomes as a weighted
combination of never-treated units. The weights show which donors drive the
counterfactual, and they are a central part of what should be reported (Abadie 2021).
Classic synthetic control weights ([`synthetic_control`](@ref)) are non-negative and sum
to one, so the counterfactual interpolates within the convex hull of the donors, and
they are often sparse. Synthetic DiD unit weights ([`synthetic_did`](@ref)) are also on
the simplex but are regularised, so they spread over more donors. Augmented synthetic
control weights ([`augmented_synthetic_control`](@ref)) still sum to one but can be
negative, since the ridge correction extrapolates beyond the convex hull; the returned
table then also shows the underlying SCM weights. Weights that are highly concentrated,
or that depend strongly on specification choices, call for the robustness checks in
[`synth_leave_one_out`](@ref).

# Arguments
- `r`: a [`SyntheticDiDEstimate`](@ref), [`SyntheticControlEstimate`](@ref) or
  [`AugmentedSCEstimate`](@ref).

# Returns
- `DataFrame` with columns `unit` and `weight`. Staggered-adoption SDID results add a
  `cohort` column (the adoption period each set of weights belongs to), and augmented
  synthetic control results add `scm_weight`.

# Examples
```julia
using DrSnow, CSV, DataFrames
prop99 = CSV.read(pkgdir(DrSnow, "test", "validation", "synth", "prop99.csv"), DataFrame)
r = synthetic_control(prop99, :PacksPerCapita, :treated, :State, :Year; placebo=false)
w = synth_weights(r)
sort(w[w.weight .> 1e-6, :], :weight; rev=true)
```

# References
- Abadie, A. (2021). Using synthetic controls: Feasibility, data requirements, and
  methodological aspects. *Journal of Economic Literature*, 59(2), 391–425.
- Abadie, A., Diamond, A., & Hainmueller, J. (2010). Synthetic control methods for
  comparative case studies: Estimating the effect of California's tobacco control
  program. *Journal of the American Statistical Association*, 105(490), 493–505.
"""
function synth_weights end

"""
    synth_time_weights(r::SyntheticDiDEstimate) -> DataFrame

Pre-treatment period weights ``\\hat\\lambda`` of a synthetic difference-in-differences
estimate.

Synthetic DiD compares post-treatment outcomes with a weighted average of pre-treatment
periods rather than with the plain pre-treatment mean. The weights ``\\hat\\lambda_t``
are chosen so that, for control units, the weighted pre-treatment outcomes predict the
average post-treatment outcome up to a constant (Arkhangelsky et al. 2021). They
often concentrate on the periods just before adoption. Weights on distant periods
indicate that those periods resemble the post-treatment period better. `method = :did`
uses equal weights, and `method = :sc` sets every time weight to zero (no time
weighting).

# Arguments
- `r::SyntheticDiDEstimate`: result of [`synthetic_did`](@ref).

# Returns
- `DataFrame` with columns `time` (pre-treatment periods) and `weight`, plus `cohort`
  under staggered adoption.

# Examples
```julia
using DrSnow, CSV, DataFrames
prop99 = CSV.read(pkgdir(DrSnow, "test", "validation", "synth", "prop99.csv"), DataFrame)
r = synthetic_did(prop99, :PacksPerCapita, :treated, :State, :Year; se_method=:none)
tw = synth_time_weights(r)
tw[tw.weight .> 0, :]
```

# References
- Arkhangelsky, D., Athey, S., Hirshberg, D. A., Imbens, G. W., & Wager, S. (2021).
  Synthetic difference-in-differences. *American Economic Review*, 111(12), 4088–4118.
"""
function synth_time_weights end

"""
    synth_gaps(r) -> DataFrame

Period-by-period comparison of the treated unit(s) with their synthetic counterfactual.

The gap ``\\hat\\tau_t = Y_{1t} - \\hat Y_{1t}(0)`` between the treated outcome and its
synthetic counterpart is the basic output of every synthetic-control-type analysis.
Before treatment it measures the quality of the fit, and after treatment it estimates
the period-specific effect. A good pre-treatment fit over a long period is what makes
the post-treatment gaps credible (Abadie, Diamond & Hainmueller 2010; Abadie 2021).
Pre-treatment gaps that are large or trending relative to the post-treatment gaps
undermine the estimate. This function returns the series that the gap plots of
[`plot_synth`](@ref) draw.

What the synthetic path contains depends on the estimator. For
[`SyntheticControlEstimate`](@ref) and [`AugmentedSCEstimate`](@ref) it is the weighted
donor outcome, with the treated units averaged in the latter. For
[`SyntheticDiDEstimate`](@ref) it includes the time-weighted pre-period level
adjustment, so post-treatment gaps equal the `synthdid` effect curve, and staggered
designs add a `cohort` column. For [`MatrixCompletionEstimate`](@ref) the rows average
the treated units' observed and imputed untreated outcomes by period, and `post` marks
periods from the first adoption onwards.

# Arguments
- `r`: any synthetic-control-type estimate of this package.

# Returns
- `DataFrame` with columns `time`, `treated` (average outcome of the treated units),
  `synthetic` (estimated counterfactual), `gap = treated - synthetic` and `post` (period
  at or after adoption), plus `cohort` for staggered SDID results.

# Examples
```julia
using DrSnow, CSV, DataFrames
prop99 = CSV.read(pkgdir(DrSnow, "test", "validation", "synth", "prop99.csv"), DataFrame)
r = synthetic_control(prop99, :PacksPerCapita, :treated, :State, :Year; placebo=false)
g = synth_gaps(r)
sqrt(sum(abs2, g.gap[.!g.post]) / count(!, g.post))   # pre-treatment RMSPE
g[g.post, :]
```

# References
- Abadie, A., Diamond, A., & Hainmueller, J. (2010). Synthetic control methods for
  comparative case studies: Estimating the effect of California's tobacco control
  program. *Journal of the American Statistical Association*, 105(490), 493–505.
- Abadie, A. (2021). Using synthetic controls: Feasibility, data requirements, and
  methodological aspects. *Journal of Economic Literature*, 59(2), 391–425.
"""
function synth_gaps end

_sc_check_level(level) = (0 < level < 1) ||
    throw(ArgumentError("level must be in (0, 1), got $level"))

function _sc_show_se_line(io, est, se, se_label, level=0.95)
    @printf(io, "  Estimate:   %.4f\n", est)
    if se === nothing
        println(io, "  Std. error: not computed (se_method = :none)")
    else
        c = critical_value(level)
        @printf(io, "  Std. error: %.4f  (%s)\n", se, se_label)
        @printf(io, "  %d%% CI:     [%.4f, %.4f]  (normal approximation)\n",
                round(Int, 100 * level), est - c * se, est + c * se)
    end
end
