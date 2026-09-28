# Descriptive pre-treatment covariate balance by treatment cohort.

"""
    pretreatment_balance(data, treatment, unit, time; covariates,
                         control_group=:never_treated, anticipation=0) -> DataFrame
    pretreatment_balance(panel::TreatmentPanel; kwargs...) -> DataFrame

Descriptive comparison of covariate means between each treatment cohort and its
comparison group in the cohort's own pre-treatment periods.

Unconditional parallel trends is more plausible when treated and comparison units
are similar in characteristics that affect outcome trends. For every cohort ``g``
the function compares covariate means of the cohort with those of the comparison
observations in the periods before treatment, ``t < g -`` anticipation, so that the
comparison is not contaminated by effects of treatment on the covariates. The
comparison observations are those of never-treated units (`control_group =
:never_treated`) or of all units not yet treated in that period
(`:not_yet_treated`), matching the comparison groups of
[`did_callaway_santanna`](@ref). The standardized difference, the difference in
means divided by the square root of the average of the two variances, is a
scale-free measure of imbalance that, unlike a t-statistic, does not grow with the
sample size (Imbens and Rubin, 2015; Austin, 2009).

This is a descriptive diagnostic, not a test. Large standardized differences (a
common rule of thumb flags absolute values above 0.1 to 0.25) indicate that the
groups differ in observed characteristics, in which case parallel trends may only
be plausible conditional on them; consider passing the covariates to
[`did_callaway_santanna`](@ref) or [`did_drdid`](@ref). Balance in levels is neither
necessary nor sufficient for parallel trends, which concerns trends, and small
differences do not establish comparability on unobservables.

# Arguments
- `data`: a long-format panel.
- `treatment`: an absorbing 0/1 indicator or [`FirstTreated`](@ref)`(column)`.
- `unit::Symbol`, `time::Symbol`: unit and time columns.

# Keywords
- `covariates::Vector{Symbol}` (required): numeric covariates to compare; the
  `TreatmentPanel` method uses the panel's covariates.
- `control_group::Symbol = :never_treated`: `:never_treated` or `:not_yet_treated`.
- `anticipation::Integer = 0`: number of anticipation periods; pre-treatment
  periods of cohort ``g`` end at ``g - 1 -`` anticipation.

# Returns
- `DataFrame`: one row per cohort and covariate with columns `cohort`, `covariate`,
  `n_treated` and `n_control` (observations), `treated_mean`, `control_mean`,
  `difference` and `std_diff` (the standardized difference).

# Examples
```julia
using DrSnow, CSV, DataFrames
file = joinpath(pkgdir(DrSnow), "test", "validation", "did", "mpdta.csv")
mpdta = CSV.read(file, DataFrame)
pretreatment_balance(mpdta, FirstTreated(:first_treat), :countyreal, :year;
                     covariates=[:lpop])
```

# References
- Imbens, G. W., & Rubin, D. B. (2015). *Causal Inference for Statistics, Social,
  and Biomedical Sciences: An Introduction*. Cambridge University Press.
- Austin, P. C. (2009). Balance diagnostics for comparing the distribution of
  baseline covariates between treatment groups in propensity-score matched samples.
  *Statistics in Medicine*, 28(25), 3083–3107.
- Abadie, A. (2005). Semiparametric difference-in-differences estimators. *Review of
  Economic Studies*, 72(1), 1–19.
"""
function pretreatment_balance(data, treatment, unit::Symbol, time::Symbol;
                              covariates::Vector{Symbol},
                              control_group::Symbol=:never_treated,
                              anticipation::Integer=0)
    isempty(covariates) && throw(ArgumentError("pretreatment_balance: no covariates"))
    _did_check_control_group(control_group)
    tcol = _did_treatment_column(treatment)
    df = _did_prepare(data, [tcol, unit, time, covariates...];
                      context="pretreatment_balance", treatment=treatment)
    tm = treatment_timing(df, treatment, unit, time; anticipation=anticipation)
    δ = Int(anticipation)
    G, P = tm.row_cohort, tm.row_period
    out = DataFrame(cohort=Any[], covariate=Symbol[], n_treated=Int[], n_control=Int[],
                    treated_mean=Float64[], control_mean=Float64[], difference=Float64[],
                    std_diff=Float64[])
    for g in _did_cohorts(tm)
        pre = P .< g - δ
        tr = (G .== g) .& pre
        ctrl = if control_group === :never_treated
            (G .== 0) .& pre
        else
            ((G .== 0) .| (P .< G .- δ)) .& (G .!= g) .& pre
        end
        (any(tr) && any(ctrl)) || continue
        for c in covariates
            x = float.(df[!, c])
            a, b = x[tr], x[ctrl]
            d = mean(a) - mean(b)
            s = sqrt((_did_var0(a) + _did_var0(b)) / 2)
            push!(out, (tm.periods[g], c, length(a), length(b), mean(a), mean(b), d,
                        s > 0 ? d / s : NaN))
        end
    end
    nrow(out) > 0 || throw(ArgumentError(
        "pretreatment_balance: no cohort has both pre-treatment and comparison " *
        "observations"))
    return out
end

_did_var0(x) = length(x) > 1 ? var(x) : 0.0

function pretreatment_balance(panel::TreatmentPanel; kwargs...)
    return pretreatment_balance(panel.data, panel.treatment, panel.unit_id, panel.time;
                                covariates=panel.covariates, kwargs...)
end
