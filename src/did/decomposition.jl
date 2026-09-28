# Diagnostics of the TWFE estimator under staggered adoption:
# Goodman-Bacon (2021) decomposition and de Chaisemartin & D'Haultfœuille (2020)
# weights.

"""
    TWFEWeights

Decomposition of the two-way fixed effects coefficient into weights on the
treatment effects of the treated unit–period cells (de Chaisemartin and
D'Haultfœuille, 2020), returned by [`twfe_weights`](@ref).

Under parallel trends, the TWFE coefficient satisfies

```math
\\beta_{fe} = E\\Big[\\sum_{(i,t):\\, D_{it} = 1} \\frac{\\omega_{it}}{N_1}\\,
W_{it}\\, \\Delta_{it}\\Big], \\qquad N_1 = \\sum_{(i,t):\\, D_{it} = 1} \\omega_{it},
```

where ``\\Delta_{it}`` is the treatment effect in cell ``(i,t)``, ``\\omega_{it}``
its regression weight, and ``W_{it}`` is proportional to the residual of ``D_{it}``
from a regression on unit and period fixed effects, normalized so that the weights
average to one over treated cells. Negative ``W_{it}`` arise for cells treated early
and observed late, whose outcomes are used as controls. The summary ``\\sigma_{fe}``
is the smallest standard deviation of the cell effects that is compatible with the
observed ``\\beta_{fe}`` and an ATT of zero (de Chaisemartin and D'Haultfœuille,
2020, Corollary 1); a small value relative to
plausible effect heterogeneity signals that ``\\beta_{fe}`` is not a reliable
summary.

# Fields
- `weights::DataFrame`: one row per treated unit–period cell with columns `unit`,
  `time`, `weight` (``W_{it}``, averaging to one) and `share`
  (``\\omega_{it} W_{it} / N_1``, summing to one).
- `beta_fe::Float64`: the TWFE coefficient.
- `n_treated_cells::Int`, `n_negative::Int`: number of treated cells, and of those
  with a negative weight.
- `sum_negative::Float64`, `sum_positive::Float64`: sums of the negative and of the
  positive `share`s (they add up to one).
- `sigma_fe::Float64`: ``|\\beta_{fe}| / \\sigma(W)``.

# References
- de Chaisemartin, C., & D'Haultfœuille, X. (2020). Two-way fixed effects estimators
  with heterogeneous treatment effects. *American Economic Review*, 110(9),
  2964–2996.
"""
struct TWFEWeights
    weights::DataFrame
    beta_fe::Float64
    n_treated_cells::Int
    n_negative::Int
    sum_negative::Float64
    sum_positive::Float64
    sigma_fe::Float64
end

function Base.show(io::IO, ::MIME"text/plain", w::TWFEWeights)
    println(io, "TWFE weights (de Chaisemartin & D'Haultfœuille 2020)")
    @printf(io, "TWFE coefficient β_fe: %.6g\n", w.beta_fe)
    println(io, "Treated cells: ", w.n_treated_cells, "; with negative weight: ",
            w.n_negative)
    @printf(io, "Sum of positive weights: %.4f; sum of negative weights: %.4f\n",
            w.sum_positive, w.sum_negative)
    @printf(io, "σ_fe (min. SD of effects compatible with ATT = 0): %.4g\n", w.sigma_fe)
end

Base.show(io::IO, w::TWFEWeights) =
    print(io, "TWFEWeights(", w.n_negative, "/", w.n_treated_cells, " negative)")

"""
    twfe_weights(data, outcome, treatment, unit, time; weights=nothing) -> TWFEWeights

Weights that the two-way fixed effects coefficient places on the treatment effect of
each treated unit–period cell (de Chaisemartin and D'Haultfœuille, 2020,
Theorem 1).

The diagnostic answers the question of which average of effects a TWFE regression
estimates. Under parallel trends and no anticipation, ``\\beta_{fe}`` equals a
weighted sum of cell effects ``\\Delta_{it}`` with weights proportional to the
residuals of ``D_{it}`` from a regression of the treatment on unit and period fixed
effects (see [`TWFEWeights`](@ref)). The weights depend only on the treatment
design, not on the outcome, so they can be computed before any outcome is examined.
When all weights are positive, ``\\beta_{fe}`` is a convex (if not policy-relevant)
average of effects; negative weights, which arise under staggered adoption because
already-treated observations serve as controls, mean that it can have the opposite
sign of every cell effect if effects are heterogeneous (see also Goodman-Bacon,
2021, and Borusyak, Jaravel and Spiess, 2024). The diagnostic describes the
estimand; it does not measure bias, which depends on the unknown pattern of effect
heterogeneity. It works for staggered and for non-absorbing binary treatments and
for unbalanced panels.

The implementation is validated against the R package `TwoWayFEWeights`.

# Arguments
- `data`: a long-format panel.
- `outcome::Symbol`: outcome column (used only for ``\\beta_{fe}``).
- `treatment`: a 0/1 indicator ``D_{it}`` or [`FirstTreated`](@ref)`(column)`.
- `unit::Symbol`, `time::Symbol`: unit and time columns.

# Keywords
- `weights::Union{Nothing,Symbol} = nothing`: regression-weight column, as in
  [`did_twfe`](@ref).

# Returns
- `TWFEWeights`: the cell weights and the summaries `n_negative`, `sum_negative`
  and `sigma_fe`.

# Examples
```julia
using DrSnow, CSV, DataFrames
file = joinpath(pkgdir(DrSnow), "test", "validation", "did", "mpdta.csv")
mpdta = CSV.read(file, DataFrame)
w = twfe_weights(mpdta, :lemp, FirstTreated(:first_treat), :countyreal, :year)
w.n_negative, w.sum_negative, w.sigma_fe
```

# References
- de Chaisemartin, C., & D'Haultfœuille, X. (2020). Two-way fixed effects estimators
  with heterogeneous treatment effects. *American Economic Review*, 110(9),
  2964–2996.
- Goodman-Bacon, A. (2021). Difference-in-differences with variation in treatment
  timing. *Journal of Econometrics*, 225(2), 254–277.
- Borusyak, K., Jaravel, X., & Spiess, J. (2024). Revisiting event-study designs:
  Robust and efficient estimation. *Review of Economic Studies*, 91(6), 3253–3285.
- de Chaisemartin, C., & D'Haultfœuille, X. (2023). Two-way fixed effects and
  differences-in-differences with heterogeneous treatment effects: A survey. *The
  Econometrics Journal*, 26(3), C1–C30.
- Quispe, A., Ciccia, D., Knau, F., Malezieux, M., Sow, D., Zhang, S., &
  de Chaisemartin, C. (2026). TwoWayFEWeights: Estimation of the weights attached to
  the two-way fixed effects regressions. R package version 2.1.0.
"""
function twfe_weights(data, outcome::Symbol, treatment, unit::Symbol, time::Symbol;
                      weights::Union{Nothing,Symbol}=nothing)
    tcol = _did_treatment_column(treatment)
    df = _did_prepare(data, [outcome, tcol, unit, time, weights];
                      context="twfe_weights", treatment=treatment)
    tm = treatment_timing(df, treatment, unit, time)
    dcol = _did_fresh_name(df, "treated")
    df[!, dcol] = Float64.(tm.row_treated)
    any(tm.row_treated) || throw(ArgumentError("twfe_weights: no treated observations"))
    fD = make_formula(dcol, Symbol[]; fe=[unit, time])
    kw = weights === nothing ? (;) : (; weights=weights)
    mD = reg(df, fD; save=:residuals, drop_singletons=false, kw...)
    ε = Vector{Float64}(coalesce.(residuals(mD), NaN))
    ω = weights === nothing ? ones(nrow(df)) : float.(df[!, weights])
    tr = findall(tm.row_treated)
    all(isfinite, ε[tr]) || throw(ArgumentError(
        "twfe_weights: could not residualize the treatment on the fixed effects"))
    N1 = sum(ω[tr])
    denom = sum(ω[tr] .* ε[tr]) / N1
    abs(denom) > 1e-12 || throw(ArgumentError(
        "twfe_weights: the treatment is collinear with the fixed effects"))
    W = ε[tr] ./ denom
    share = ω[tr] .* W ./ N1
    fY = make_formula(outcome, [dcol]; fe=[unit, time])
    mY = reg(df, fY; drop_singletons=false, kw...)
    β = coef(mY)[coef_index(mY, dcol)]
    σw = sqrt(sum(ω[tr] ./ N1 .* (W .- 1) .^ 2))
    tab = DataFrame(unit=df[tr, unit], time=df[tr, time], weight=W, share=share)
    return TWFEWeights(tab, β, length(tr), count(<(0), W),
                       sum(share[W .< 0]; init=0.0), sum(share[W .>= 0]; init=0.0),
                       σw > 0 ? abs(β) / σw : Inf)
end

"""
    BaconDecomposition

Goodman-Bacon (2021) decomposition of a two-way fixed effects coefficient into all
two-group, two-period (2×2) difference-in-differences comparisons, returned by
[`bacon_decomposition`](@ref).

The TWFE coefficient equals ``\\sum_k s_k \\hat\\beta_k``, a weighted average of 2×2
DiD estimates ``\\hat\\beta_k`` with weights ``s_k \\ge 0`` that sum to one and depend
on group sizes and on the variance of treatment within each comparison (larger for
groups treated near the middle of the panel). Comparisons of later-treated units with
earlier-treated units (and with units treated throughout) use already-treated
observations as controls; their estimates subtract changes in the treatment effects
of the earlier group, so with effects that evolve over time they are biased, and this
is how TWFE can fail under staggered adoption even though every weight is positive.

# Fields
- `comparisons::DataFrame`: one row per 2×2 comparison with columns `treated` and
  `control` (cohort labels in time units, `"never treated"` or `"always treated"`),
  `type` (`:treated_vs_never`, `:earlier_vs_later`, `:later_vs_earlier`,
  `:later_vs_always`), `estimate` (the 2×2 DiD) and `weight`.
- `by_type::DataFrame`: total weight and weighted average estimate per type.
- `twfe_estimate::Float64`: the TWFE coefficient, equal to
  `sum(comparisons.weight .* comparisons.estimate)`.

# References
- Goodman-Bacon, A. (2021). Difference-in-differences with variation in treatment
  timing. *Journal of Econometrics*, 225(2), 254–277.
"""
struct BaconDecomposition
    comparisons::DataFrame
    by_type::DataFrame
    twfe_estimate::Float64
end

function Base.show(io::IO, ::MIME"text/plain", b::BaconDecomposition)
    println(io, "Goodman-Bacon decomposition of the TWFE coefficient")
    @printf(io, "TWFE estimate: %.6g\n", b.twfe_estimate)
    show(io, MIME"text/plain"(), b.by_type; summary=false, eltypes=false)
    println(io)
end

Base.show(io::IO, b::BaconDecomposition) =
    print(io, "BaconDecomposition(", nrow(b.comparisons), " comparisons)")

"""
    bacon_decomposition(data, outcome, treatment, unit, time) -> BaconDecomposition

Goodman-Bacon (2021) decomposition of the two-way fixed effects DiD coefficient into
all 2×2 comparisons between treatment-timing groups and never-treated units, with
their weights.

In a balanced panel with an absorbing binary treatment and no covariates, the TWFE
coefficient is an exact weighted average of four kinds of 2×2 DiD estimates:
treated cohorts against never-treated units, earlier-treated against
not-yet-treated cohorts (before the later cohort is treated), later-treated against
earlier-treated cohorts (after the earlier cohort is treated), and later-treated
cohorts against units treated in every period. Under parallel trends, the first two
kinds identify averages of treatment effects; the last two use already-treated
units as controls and are biased when effects change over time. The decomposition
shows how much of the TWFE estimate comes from these problematic comparisons and
whether they differ from the clean ones. It is a descriptive decomposition of the
estimate, not a test: the weights reflect sample sizes and treatment variance
rather than policy relevance, and even clean comparisons weight effects in ways
that may not match the target parameter (see [`twfe_weights`](@ref) for the implied
weights on cell effects). For estimation under staggered adoption use
[`did_callaway_santanna`](@ref), [`did_sun_abraham`](@ref),
[`did_imputation`](@ref) or [`did_etwfe`](@ref).

The requirements of the exact decomposition, a balanced panel, an absorbing binary
treatment, and no covariates or weights, are enforced. The results match the R
package `bacondecomp`.

# Arguments
- `data`: a balanced long-format panel.
- `outcome::Symbol`: outcome column.
- `treatment`: an absorbing 0/1 indicator or [`FirstTreated`](@ref)`(column)`.
- `unit::Symbol`, `time::Symbol`: unit and time columns.

# Returns
- `BaconDecomposition`: `sum(b.comparisons.weight .* b.comparisons.estimate)` equals
  `b.twfe_estimate`, which equals `coef(did_twfe(...))[1]` without covariates.

# Examples
```julia
using DrSnow, CSV, DataFrames
file = joinpath(pkgdir(DrSnow), "test", "validation", "did", "mpdta.csv")
mpdta = CSV.read(file, DataFrame)
b = bacon_decomposition(mpdta, :lemp, FirstTreated(:first_treat), :countyreal, :year)
b.by_type
```

# References
- Goodman-Bacon, A. (2021). Difference-in-differences with variation in treatment
  timing. *Journal of Econometrics*, 225(2), 254–277.
- Baker, A. C., Larcker, D. F., & Wang, C. C. Y. (2022). How much should we trust
  staggered difference-in-differences estimates? *Journal of Financial Economics*,
  144(2), 370–395.
- Roth, J., Sant'Anna, P. H. C., Bilinski, A., & Poe, J. (2023). What's trending in
  difference-in-differences? A synthesis of the recent econometrics literature.
  *Journal of Econometrics*, 235(2), 2218–2244.
- Flack, E., & Jee, E. (2020). bacondecomp: Goodman-Bacon decomposition. R package
  version 0.1.1.
"""
function bacon_decomposition(data, outcome::Symbol, treatment, unit::Symbol,
                             time::Symbol)
    tcol = _did_treatment_column(treatment)
    df = _did_prepare(data, [outcome, tcol, unit, time];
                      context="bacon_decomposition", treatment=treatment)
    tm = treatment_timing(df, treatment, unit, time)
    tm.balanced || throw(ArgumentError(
        "bacon_decomposition requires a balanced panel"))
    tm.absorbing || throw(ArgumentError(
        "bacon_decomposition requires an absorbing (staggered-adoption) treatment"))
    N, T = length(tm.units), length(tm.periods)
    Y = zeros(N, T)
    y = float.(df[!, outcome])
    for i in eachindex(y)
        Y[tm.row_unit[i], tm.row_period[i]] = y[i]
    end
    G = tm.unit_cohort
    D = [G[i] > 0 && t >= G[i] ? 1.0 : 0.0 for i in 1:N, t in 1:T]
    Dt = _did_twoway_demean(D)
    VD = mean(abs2, Dt)
    VD > 0 || throw(ArgumentError("bacon_decomposition: treatment has no variation " *
                                  "within units and periods"))
    β_twfe = sum(Dt .* Y) / sum(abs2, Dt)
    label(g) = g == 0 ? "never treated" : g == 1 ? "always treated" :
               tm.periods[g]
    rows = NamedTuple[]
    function add!(treated_g, control_g, type, units_mask, prange)
        Ds = D[units_mask, prange]
        Dd = _did_twoway_demean(Ds)
        Vs = mean(abs2, Dd)
        Vs > 1e-14 || return nothing
        β = sum(Dd .* Y[units_mask, prange]) / sum(abs2, Dd)
        share = count(units_mask) * length(prange) / (N * T)
        push!(rows, (treated=label(treated_g), control=label(control_g), type=type,
                     estimate=β, weight=share^2 * Vs / VD))
        return nothing
    end
    groups = sort!(unique(filter(>(0), G)))
    has_never = any(==(0), G)
    for k in groups
        has_never && add!(k, 0, :treated_vs_never, (G .== k) .| (G .== 0), 1:T)
    end
    for (a, k) in enumerate(groups), l in groups[(a + 1):end]
        mask = (G .== k) .| (G .== l)
        k > 1 && add!(k, l, :earlier_vs_later, mask, 1:(l - 1))
        add!(l, k, k == 1 ? :later_vs_always : :later_vs_earlier, mask, k:T)
    end
    comps = DataFrame(rows)
    isempty(rows) && throw(ArgumentError("bacon_decomposition: no 2×2 comparisons"))
    total = sum(comps.weight .* comps.estimate)
    if abs(sum(comps.weight) - 1) > 1e-8 || abs(total - β_twfe) > 1e-8 * max(1, abs(β_twfe))
        @warn "bacon_decomposition: weights do not reproduce the TWFE coefficient " *
              "(sum of weights $(sum(comps.weight)))"
    end
    by = combine(groupby(comps, :type), :weight => sum => :weight,
                 [:weight, :estimate] => ((w, e) -> sum(w .* e) / sum(w)) => :estimate)
    return BaconDecomposition(comps, by, β_twfe)
end

function _did_twoway_demean(A::AbstractMatrix)
    return A .- mean(A; dims=2) .- mean(A; dims=1) .+ mean(A)
end
