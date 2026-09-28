# Measurement-error diagnostics and corrections for ML-measured variables:
# regression calibration of a covariate measured with error when a validation
# subsample has the true value, and a test of differential prediction error across
# treatment arms, periods or sides of a cutoff.

"""
    regression_calibration(data, outcome, mismeasured, gold; covariates=Symbol[],
                           labeled=nothing, cluster=nothing) -> MeasurementEstimate

Regression-calibration estimate of a linear regression in which one covariate is
observed on every row only through an error-prone measurement (for example an ML
prediction) and its true value is observed on a validation subsample.

The estimand is the coefficient vector of the population least-squares
regression of `outcome` on ``(1, X, Z)``, where ``X`` is the true covariate
(`gold`) and ``Z`` the error-free `covariates`. On every row only the
measurement ``X^\\ast`` (`mismeasured`) is available. Under classical
(additive, independent) error, the naive slope on ``X^\\ast`` is attenuated
towards zero by the reliability ratio ``\\operatorname{Var}(X)/
\\operatorname{Var}(X^\\ast)`` (in the simple regression), and the coefficients
on correlated covariates are biased as well (Carroll et al. 2006). Fong
and Tyler (2021) discuss the same problem for ML predictions used as regression
covariates.

Regression calibration (Carroll et al. 2006, ch. 4) fits the calibration model
``E[X \\mid X^\\ast, Z]``, linear in ``X^\\ast`` and ``Z``, by least squares on
the validation rows, and replaces ``X`` by its fitted value
``\\hat X = \\hat E[X \\mid X^\\ast, Z]`` in the least-squares regression of the
outcome on ``(1, \\hat X, Z)`` over all rows. The estimator is consistent under
three assumptions that the data can only partly assess: (i) non-differential
error (surrogacy), meaning that ``X^\\ast`` carries no information about the
outcome given ``(X, Z)``; (ii) validation rows that are a random sample, or
selected on ``(X^\\ast, Z)`` only, with gold values measured without error;
and (iii) a linear calibration function, which holds exactly under joint
normality and is otherwise an approximation that can fail for strongly
non-linear calibration curves (e.g. a classifier's scores). The covariance is
obtained from the stacked estimating equations of both steps, so it accounts
for the estimation of the calibration model; it is clustered with `cluster`,
has no small-sample factor, and inference uses normal critical values.

When the error may be differential, for example an ML measure whose error
depends on the outcome or on treatment, regression calibration is biased; use
[`dsl_regression`](@ref) with `predicted_vars = [gold]`, which requires a known
labelling design instead of assumptions on the error. Report the calibration
slope (`details.calibration_slope`, the reliability ratio under classical error
without covariates) and the naive estimate alongside the corrected one.

# Arguments
- `data::AbstractDataFrame`: one row per unit.
- `outcome::Symbol`: the dependent variable, observed on every row.
- `mismeasured::Symbol`: the error-prone measurement ``X^\\ast``, observed on
  every row.
- `gold::Symbol`: the true covariate ``X``, observed on validation rows
  (`missing` elsewhere unless `labeled` is given).

# Keywords
- `covariates::Vector{Symbol} = Symbol[]`: error-free regressors, included in
  both the calibration and the outcome model.
- `labeled::Union{Nothing,Symbol} = nothing`: 0/1 validation indicator (default:
  rows with a non-missing `gold`).
- `cluster::Union{Nothing,Symbol} = nothing`: cluster column for the covariance.

# Returns
- [`MeasurementEstimate`](@ref) with coefficients `"(Intercept)"`, `gold` and
  the `covariates`; `naive_coef` is the regression on ``X^\\ast`` (HC0 or CR0
  sandwich), and `details` has the calibration coefficients
  (`calibration_coef`, `calibration_vcov`, `calibration_names`),
  `calibration_slope` and `labeled`.

# Examples
```julia
using DrSnow, DataFrames, StableRNGs
rng = StableRNG(14)
n = 2000
age = randn(rng, n)
x_true = 0.5 .* age .+ randn(rng, n)
x_pred = x_true .+ 0.8 .* randn(rng, n)                 # classical error
y = 1 .+ 0.8 .* x_true .+ 0.3 .* age .+ randn(rng, n)
val = rand(rng, n) .< 0.2
df = DataFrame(; y, x_pred, age,
               x_true=[v ? x : missing for (v, x) in zip(val, x_true)])
r = regression_calibration(df, :y, :x_pred, :x_true; covariates=[:age])
coef(r)[2], r.naive_coef[2], r.details.calibration_slope
```

# References
- Carroll, R. J., Ruppert, D., Stefanski, L. A., & Crainiceanu, C. M. (2006).
  *Measurement Error in Nonlinear Models: A Modern Perspective* (2nd ed.).
  Chapman & Hall/CRC.
- Fong, C., & Tyler, M. (2021). Machine learning predictions as regression
  covariates. *Political Analysis*, 29(4), 467–484.
- Wang, S., McCormick, T. H., & Leek, J. T. (2020). Methods for correcting
  inference based on outcomes predicted by machine learning. *Proceedings of
  the National Academy of Sciences*, 117(48), 30266–30275.
"""
function regression_calibration(data::AbstractDataFrame, outcome::Symbol,
                                mismeasured::Symbol, gold::Symbol;
                                covariates::Vector{Symbol}=Symbol[],
                                labeled::Union{Nothing,Symbol}=nothing,
                                cluster::Union{Nothing,Symbol}=nothing)
    ctx = "regression_calibration"
    require_columns(data, vcat(outcome, mismeasured, gold, covariates, labeled, cluster);
                    context=ctx)
    _ml_meas_complete(data, vcat(outcome, mismeasured, covariates, cluster); context=ctx)
    R, _ = _ml_meas_labels(data, [gold], labeled, nothing, false; context=ctx)
    n = nrow(data)
    x = _ml_meas_gold(data, gold, R; context=ctx)
    y = _ml_column(data, outcome; context=ctx)
    xs = _ml_column(data, mismeasured; context=ctx)
    Z = _ml_matrix(data, covariates; context=ctx)
    cl, G = _ml_meas_clusters(data, cluster; context=ctx)
    W = hcat(ones(n), xs, Z)
    p1 = size(W, 2)
    count(R) > p1 || throw(ArgumentError("$(ctx): too few validation rows"))
    Wl = W[R, :]
    rank(Wl) == p1 || throw(ArgumentError("$(ctx): collinear calibration regressors"))
    γ = Wl \ x[R]
    xhat = W * γ
    Vd = hcat(ones(n), xhat, Z)
    p2 = size(Vd, 2)
    rank(Vd) == p2 || throw(ArgumentError("$(ctx): collinear outcome regressors"))
    β = Vd \ y
    r1 = R .* (x .- W * γ)
    r2 = y .- Vd * β
    M = hcat(W .* r1, Vd .* r2)
    J = zeros(p1 + p2, p1 + p2)
    J[1:p1, 1:p1] = (W' * (W .* R)) ./ n
    J[(p1 + 1):end, (p1 + 1):end] = (Vd' * Vd) ./ n
    # ∂/∂γ of the outcome moment: V depends on γ through x̂ = Wγ (column 2)
    J21 = β[2] .* (Vd' * W) ./ n
    J21[2, :] .-= vec(sum(W .* r2; dims=1)) ./ n
    J[(p1 + 1):end, 1:p1] = J21
    Vall = _ml_meas_sandwich(M, J, cl)
    V = Vall[(p1 + 1):end, (p1 + 1):end]
    names = vcat("(Intercept)", string(gold), string.(covariates))
    # naive: regression on X* (HC0 / CR0 sandwich)
    Xn = W
    βn = Xn \ y
    Vn = _ml_meas_sandwich(Xn .* (y .- Xn * βn), (Xn' * Xn) ./ n, cl)
    return MeasurementEstimate(names, β, Matrix(Symmetric(V)),
                               "Regression calibration (linear model)",
                               "coefficients of the regression on the true covariate", n,
                               count(R), G, βn, Vn,
                               (calibration_coef=γ,
                                calibration_vcov=Matrix(Symmetric(Vall[1:p1, 1:p1])),
                                calibration_names=vcat("(Intercept)", string(mismeasured),
                                                       string.(covariates)),
                                calibration_slope=γ[2], labeled=R))
end

"""
    differential_error_test(data, outcome, prediction; by, labeled=nothing,
                            label_prob=nothing, normalize_prob=true, cluster=nothing)
        -> DiagnosticTest

Wald test of whether the mean prediction error of an ML measurement differs
across the cells defined by the columns `by` (treatment arms, periods,
treatment-group × period cells, or sides of an RD cutoff), using the
gold-labelled rows.

The null hypothesis is that the mean error ``E[Y - f \\mid \\text{cell}]`` is the
same in every cell, where ``Y`` is the gold-standard value and ``f`` the
prediction. Differences in the mean error across the cells that a design
compares are what bias estimates that use the ML measurement as the outcome: a
difference between arms for an experiment, the change over time of the
treated-minus-comparison error gap for difference-in-differences, a jump at the
cutoff for regression discontinuity. The test is deliberately broad: it also
rejects when errors differ in ways a design removes (for example a
group-specific error that is constant over time, which DiD differences out).
Gold labels are assumed error-free.

The cell means are the coefficients of a weighted least-squares regression of
the error on cell indicators over the labelled rows, with weights ``1/\\pi_i``
so that each cell mean refers to all rows of the cell when labelling
probabilities vary (`FixedEffectModels`, HC1 or cluster-robust covariance). The
statistic is the Wald F test that all cell means are equal, with denominator
degrees of freedom ``G - 1`` with clusters and ``n_{\\text{labeled}} - K``
otherwise (``K`` cells).

A non-rejection is not evidence that the error is non-differential: power is
low with few labels, and equal mean errors can coexist with errors that depend
on the true outcome in ways that bias other estimands. A rejection indicates
that the naive estimate is likely to be biased, not by how much. The corrected
estimators ([`dsl_regression`](@ref), [`ppi_ate`](@ref),
[`did_with_predicted_outcome`](@ref), [`rd_with_predicted_outcome`](@ref)) do
not rely on this test, and their naive-versus-corrected contrast
(`bias_test`) targets the bias of the specific estimand directly.

# Arguments
- `data::AbstractDataFrame`: the analysis data.
- `outcome::Symbol`: gold-standard outcome, `missing` when not labelled (unless
  `labeled` is given).
- `prediction::Symbol`: the ML prediction, observed on every row.

# Keywords
- `by::Vector{Symbol}`: columns whose combinations define the cells (required;
  at least two cells, each with at least two labelled rows).
- `labeled = nothing`, `label_prob = nothing`, `normalize_prob = true`:
  labelling design, as in [`dsl_pseudo_outcome`](@ref).
- `cluster::Union{Nothing,Symbol} = nothing`: cluster column for the covariance.

# Returns
- [`DiagnosticTest`](@ref) with the F statistic, p-value and degrees of freedom;
  `details` has `cells`, `mean_error`, `se` and `n_labeled` per cell.

# Examples
```julia
using DrSnow, DataFrames, StableRNGs
rng = StableRNG(15)
n = 2000
treated = Float64.(rand(rng, n) .< 0.5)
y = 1 .+ treated .+ randn(rng, n)
y_llm = 0.9 .* y .+ 0.4 .* treated .+ 0.5 .* randn(rng, n)   # differential error
lab = rand(rng, n) .< 0.2
df = DataFrame(y_expert=[l ? v : missing for (l, v) in zip(lab, y)], y_llm=y_llm,
               treated=treated)
t = differential_error_test(df, :y_expert, :y_llm; by=[:treated])
t.pvalue, t.details.mean_error
```

# References
- Egami, N., Hinck, M., Stewart, B. M., & Wei, H. (2023). Using imperfect
  surrogates for downstream inference: Design-based supervised learning for
  social science applications of large language models. *Advances in Neural
  Information Processing Systems*, 36, 68589–68601.
- Carroll, R. J., Ruppert, D., Stefanski, L. A., & Crainiceanu, C. M. (2006).
  *Measurement Error in Nonlinear Models: A Modern Perspective* (2nd ed.).
  Chapman & Hall/CRC.
"""
function differential_error_test(data::AbstractDataFrame, outcome::Symbol,
                                 prediction::Symbol; by::Vector{Symbol},
                                 labeled::Union{Nothing,Symbol}=nothing,
                                 label_prob::Union{Nothing,Symbol}=nothing,
                                 normalize_prob::Bool=true,
                                 cluster::Union{Nothing,Symbol}=nothing)
    ctx = "differential_error_test"
    isempty(by) && throw(ArgumentError("$(ctx): `by` must name at least one column"))
    require_columns(data, vcat(outcome, prediction, by, labeled, label_prob, cluster);
                    context=ctx)
    _ml_meas_complete(data, vcat(prediction, by, cluster); context=ctx)
    R, π = _ml_meas_labels(data, [outcome], labeled, label_prob, normalize_prob;
                           context=ctx)
    y = _ml_meas_gold(data, outcome, R; context=ctx)
    f = _ml_column(data, prediction; context=ctx)
    idx = findall(R)
    keys_ = [Tuple(data[i, c] for c in by) for i in idx]
    cells = unique(keys_)
    try
        sort!(cells)
    catch
    end
    K = length(cells)
    K >= 2 || throw(ArgumentError("$(ctx): the labelled rows fall in fewer than two " *
                                  "cells"))
    cid = Dict(c => k for (k, c) in enumerate(cells))
    nk = zeros(Int, K)
    for key in keys_
        nk[cid[key]] += 1
    end
    all(>=(2), nk) || throw(ArgumentError("$(ctx): every cell needs at least two " *
                                          "labelled rows (counts: $(nk))"))
    df = DataFrame(:__e => y[idx] .- f[idx], :__w => 1 ./ π[idx])
    dnames = [Symbol("__cell", k) for k in 1:K]
    for k in 1:K
        df[!, dnames[k]] = Float64[cid[key] == k for key in keys_]
    end
    vc = if cluster === nothing
        Vcov.robust()
    else
        df[!, :__cl] = data[idx, cluster]
        Vcov.cluster(:__cl)
    end
    m = FixedEffectModels.reg(df, make_formula(:__e, dnames; intercept=false), vc;
                              weights=:__w)
    b = StatsAPI.coef(m)
    V = StatsAPI.vcov(m)
    Rm = hcat(ones(K - 1), -Matrix{Float64}(I, K - 1, K - 1))
    Gc = cluster === nothing ? 0 : length(unique(df.__cl))
    dof = cluster === nothing ? length(idx) - K : Gc - 1
    dof >= 1 || throw(ArgumentError("$(ctx): not enough labelled rows or clusters"))
    w = wald_test(b, V; R=Rm, dof=dof)
    labels = [join(("$(c)=$(v)" for (c, v) in zip(by, cell)), ", ") for cell in cells]
    return DiagnosticTest("Differential prediction-error test",
                          "the mean prediction error E[Y − prediction] is equal across " *
                          "the cells defined by $(join(string.(by), " × "))",
                          w.statistic, w.pvalue; dof=(w.dof1, dof),
                          method="weighted (1/π) regression of the error on cell " *
                                 "indicators, " *
                                 (cluster === nothing ? "HC1" : "cluster-robust") *
                                 " Wald F",
                          note="Non-rejection is not evidence that the error is " *
                               "non-differential; the test has low power with few " *
                               "labels and only compares mean errors.",
                          details=(cells=labels, mean_error=b,
                                   se=sqrt.(max.(diag(V), 0.0)), n_labeled=nk))
end
