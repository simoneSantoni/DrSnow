# Design-based supervised learning (Egami, Hinck, Stewart & Wei 2023): regression
# and category proportions when variables are measured by ML / LLM predictions and a
# subsample selected with known probabilities carries expert labels. Moments,
# variance and fixed-effect handling follow the R package dsl (naoki-egami/dsl).
# Repeated cross-fitting is aggregated with `_ml_aggregate` (median rule).

const _ML_DSL_REFS = """
- Egami, N., Hinck, M., Stewart, B. M., & Wei, H. (2023). Using imperfect
  surrogates for downstream inference: Design-based supervised learning for
  social science applications of large language models. *Advances in Neural
  Information Processing Systems*, 36, 68589–68601.
- Egami, N., Hinck, M., Stewart, B. M., & Wei, H. (2024). Using large language
  model annotations for the social sciences: A general framework of using
  predicted variables in downstream analyses. Working paper.
- Robins, J. M., Rotnitzky, A., & Zhao, L. P. (1994). Estimation of regression
  coefficients when some regressors are not always observed. *Journal of the
  American Statistical Association*, 89(427), 846–866.
- Chernozhukov, V., Chetverikov, D., Demirer, M., Duflo, E., Hansen, C., Newey,
  W., & Robins, J. (2018). Double/debiased machine learning for treatment and
  structural parameters. *The Econometrics Journal*, 21(1), C1–C68."""

"""
    dsl_regression(data, outcome; covariates=Symbol[], prediction=Symbol[],
                   predicted_vars=[outcome], features=Symbol[], family=:gaussian,
                   fe=Symbol[], intercept=true, labeled=nothing, label_prob=nothing,
                   normalize_prob=true, cluster=nothing, learner=OLSLearner(),
                   n_folds=5, n_rep=1, folds=nothing, rng=Random.default_rng(),
                   parallel=false) -> MeasurementEstimate

Design-based supervised learning (DSL) estimate of a linear, logistic or
fixed-effects regression in which the outcome and/or some regressors are
measured by an ML model or LLM, using a subsample with expert (gold-standard)
labels drawn with known probabilities.

The estimand is the vector of coefficients ``\\beta`` of the regression that
would be run if the gold-standard values of all variables were observed on
every row: the least-squares projection (`family = :gaussian`, with or without
absorbed fixed effects) or the logistic-regression coefficients
(`family = :binomial`) of the true outcome on the true regressors. The
ML-measured variables ``V`` (`predicted_vars`) are observed in gold-standard
form only on rows with ``R_i = 1``, selected with probabilities ``\\pi_i`` that
are known by design and may depend on observed variables. Identification rests
on three conditions: the gold labels are error-free; the labelling
probabilities are known and bounded away from zero; and every other variable is
observed on every row. No assumption is made about the ML predictions, whose
errors may be correlated with the outcome, the regressors or treatment (Egami et
al. 2023).

For each ML-measured variable a model ``\\hat g`` of ``V`` given the ML
prediction(s) and `features` is trained on labelled rows and cross-fitted, so
each row's prediction is made by a model that never saw its fold. The regression
is estimated as the root of the design-based moment

```math
\\frac{1}{n}\\sum_{i=1}^{n} \\Big[\\Big(1 - \\frac{R_i}{\\pi_i}\\Big)
    m(\\hat D_i; \\beta) + \\frac{R_i}{\\pi_i} m(D_i; \\beta)\\Big] = 0,
```

where ``m`` is the least-squares or logistic score, ``D_i`` the row with the
gold-standard values and ``\\hat D_i`` the row with ``\\hat g`` in place of the
ML-measured variables. The moment has the augmented inverse-probability-weighted
form of Robins, Rotnitzky and Zhao (1994) with a known selection probability,
so its expectation is the full-data score whatever the quality of ``\\hat g``:
the estimator is consistent and asymptotically normal for ``\\beta``, and
better predictions only reduce its variance. With only the outcome ML-measured
and a linear model, the estimator is least squares on the pseudo-outcome of
[`dsl_pseudo_outcome`](@ref); with `fe` it is the within (fixed-effects)
regression of that pseudo-outcome, estimated with `FixedEffectModels`.

The covariance is the sandwich ``J^{-1}\\Omega J^{-1\\prime}/n`` of the same
moment (cluster sums of the moment contributions with `cluster`), without
small-sample factors and with normal critical values, as in the R package
`dsl`. With `n_rep > 1` the cross-fitting is repeated on independent fold
splits; the point estimate is the median of the repetition estimates
``\\hat\\beta_r`` and each variance is the median of ``\\widehat{\\operatorname{
Var}}_r + (\\hat\\beta_r - \\tilde\\beta)^2``, which adds the dispersion across
splits on the scale of ``\\operatorname{Var}(\\hat\\beta)`` and is therefore
more conservative than the rule of Chernozhukov et al. (2018, §3.4), in which
the dispersion enters the variance of ``\\sqrt n(\\hat\\beta - \\beta)``.
Correlations between coefficients are averaged across repetitions. Very small
``\\pi_i`` produce large weights and unstable estimates; oversampling rows
where the ML measurement is least reliable, while keeping every probability
away from zero, is usually efficient. Report the number of labelled rows, the
labelling design and the out-of-fold RMSE of ``\\hat g`` (`details.rmse`); the
naive fit that plugs in the raw predictions (`naive_coef`) is shown for
comparison only. Use [`ppi_regression`](@ref) when only the outcome is
ML-measured and a single fixed prediction is available (it tunes the weight on
the prediction), and [`regression_calibration`](@ref) when labelling
probabilities are unknown and non-differential error is credible.

# Arguments
- `data::AbstractDataFrame`: one row per unit of analysis.
- `outcome::Symbol`: dependent variable (observed on every row unless it is
  ML-measured).

# Keywords
- `covariates::Vector{Symbol} = Symbol[]`: regressors; they may include
  ML-measured ones listed in `predicted_vars`.
- `prediction = Symbol[]`: ML prediction column(s), observed on every row; all
  of them are inputs of every ``\\hat g`` (as in R's dsl).
- `predicted_vars::Vector{Symbol} = [outcome]`: columns with gold-standard values
  on labelled rows; values on unlabelled rows are ignored and may be `missing`.
- `features::Vector{Symbol} = Symbol[]`: further inputs of ``\\hat g``.
- `family::Symbol = :gaussian`: `:gaussian` (linear) or `:binomial` (logistic,
  0/1 outcome).
- `fe::Vector{Symbol} = Symbol[]`: fixed effects absorbed with
  `FixedEffectModels`; allowed for the linear model when only the outcome is
  ML-measured.
- `intercept::Bool = true`: include an intercept (ignored with `fe`).
- `labeled = nothing`, `label_prob = nothing`, `normalize_prob = true`: labelling
  design, as in [`dsl_pseudo_outcome`](@ref); probabilities may depend on
  covariates. Without `label_prob` the labelled rows are treated as a simple
  random sample.
- `cluster = nothing`: cluster column for the variance and the cross-fitting
  groups; use at least the level at which labels were sampled.
- `learner = OLSLearner()`: [`NuisanceLearner`](@ref) for ``\\hat g`` (binary
  targets use `fitpredict_proba` when available), or `nothing` to use the raw
  prediction columns (one per ML-measured variable, in order) without
  recalibration.
- `n_folds = 5`, `n_rep = 1`, `folds = nothing`, `rng = Random.default_rng()`,
  `parallel = false`: cross-fitting options; `n_rep > 1` repeats the fold split
  and aggregates as described above.

# Returns
- [`MeasurementEstimate`](@ref). `naive_coef` holds the regression with the raw
  predictions plugged in (when each ML-measured variable has one prediction
  column); `details` has `rep_coef` and `rep_vcov` (per repetition), `fitted`
  (``\\hat g`` of the first repetition), `prob`, `labeled`, `folds`, `rmse` (the
  out-of-fold RMSE of ``\\hat g`` on labelled rows), `family`, `fe`,
  `predicted_vars` and `learner`.

# Examples
```julia
using DrSnow, DataFrames, StableRNGs
rng = StableRNG(4)
n = 2000
x1, x2 = randn(rng, n), randn(rng, n)
y = 0.5 .+ 1.0 .* x1 .- 0.5 .* x2 .+ randn(rng, n)
pred_y = 0.8 .* y .+ 0.3 .* x1 .+ 0.6 .* randn(rng, n)  # error correlated with x1
p = clamp.(0.15 .+ 0.2 .* (x2 .> 0), 0.05, 1.0)          # known design
lab = rand(rng, n) .< p
df = DataFrame(y=[l ? v : missing for (l, v) in zip(lab, y)], pred_y=pred_y,
               x1=x1, x2=x2, p=p)
r = dsl_regression(df, :y; covariates=[:x1, :x2], prediction=:pred_y,
                   label_prob=:p, normalize_prob=false, rng=StableRNG(5))
coeftable(r)
r.naive_coef          # biased slope on x1
```

# References
$(_ML_DSL_REFS)
- Wang, S., McCormick, T. H., & Leek, J. T. (2020). Methods for correcting
  inference based on outcomes predicted by machine learning. *Proceedings of
  the National Academy of Sciences*, 117(48), 30266–30275.
- Fong, C., & Tyler, M. (2021). Machine learning predictions as regression
  covariates. *Political Analysis*, 29(4), 467–484.
"""
function dsl_regression(data::AbstractDataFrame, outcome::Symbol;
                        covariates::Vector{Symbol}=Symbol[], prediction=Symbol[],
                        predicted_vars::Vector{Symbol}=[outcome],
                        features::Vector{Symbol}=Symbol[], family::Symbol=:gaussian,
                        fe::Vector{Symbol}=Symbol[], intercept::Bool=true,
                        labeled::Union{Nothing,Symbol}=nothing,
                        label_prob::Union{Nothing,Symbol}=nothing,
                        normalize_prob::Bool=true,
                        cluster::Union{Nothing,Symbol}=nothing, learner=OLSLearner(),
                        n_folds::Integer=5, n_rep::Integer=1, folds=nothing,
                        rng::AbstractRNG=Random.default_rng(), parallel::Bool=false)
    ctx = "dsl_regression"
    family in (:gaussian, :binomial) ||
        throw(ArgumentError("$(ctx): family must be :gaussian or :binomial"))
    preds = _as_symbols(prediction)
    isempty(predicted_vars) &&
        throw(ArgumentError("$(ctx): predicted_vars must name at least one column"))
    allunique(predicted_vars) ||
        throw(ArgumentError("$(ctx): predicted_vars has duplicates"))
    for v in predicted_vars
        (v == outcome || v in covariates) ||
            throw(ArgumentError("$(ctx): ML-measured variable $(v) is neither the " *
                                "outcome nor a covariate"))
    end
    isempty(intersect(preds, predicted_vars)) ||
        throw(ArgumentError("$(ctx): a prediction column cannot be an ML-measured " *
                            "variable"))
    (isempty(preds) && isempty(features) && learner !== nothing) &&
        throw(ArgumentError("$(ctx): pass `prediction` and/or `features`"))
    outcome in covariates && throw(ArgumentError("$(ctx): outcome is also a covariate"))
    if !isempty(fe)
        family === :gaussian ||
            throw(ArgumentError("$(ctx): fixed effects are supported for the linear " *
                                "model only"))
        predicted_vars == [outcome] ||
            throw(ArgumentError("$(ctx): with fixed effects only the outcome may be " *
                                "ML-measured (include ML-measured covariates without " *
                                "`fe`, e.g. with dummy columns)"))
        isempty(covariates) && throw(ArgumentError("$(ctx): no covariates to estimate"))
    end
    observed = setdiff(vcat(outcome, covariates), predicted_vars)
    require_columns(data, vcat(outcome, covariates, preds, predicted_vars, features, fe,
                               labeled, label_prob, cluster); context=ctx)
    _ml_meas_complete(data, vcat(observed, preds, features, fe, cluster); context=ctx)
    R, π = _ml_meas_labels(data, predicted_vars, labeled, label_prob, normalize_prob;
                           context=ctx)
    n = nrow(data)
    gold = [_ml_meas_gold(data, v, R; context=ctx) for v in predicted_vars]
    if family === :binomial && outcome in predicted_vars
        _ml_meas_is_binary(gold[1][R]) ||
            throw(ArgumentError("$(ctx): the outcome must be 0/1 for family = :binomial"))
    end
    cl, G = _ml_meas_clusters(data, cluster; context=ctx)
    groups = cluster === nothing ? nothing : data[!, cluster]
    F = _ml_meas_folds(data, folds, n, n_folds, n_rep, rng, R, groups; context=ctx)
    Gp = _ml_meas_g(data, gold, preds, features, R, learner, F, rng; parallel=parallel,
                    context=ctx)
    w = R ./ π
    # observed parts of the design
    obs = Dict{Symbol,Vector{Float64}}(c => _ml_column(data, c; context=ctx)
                                       for c in observed)
    gold_of = Dict(v => gold[k] for (k, v) in enumerate(predicted_vars))
    col_o(c) = haskey(gold_of, c) ? gold_of[c] : obs[c]
    use_int = intercept && isempty(fe)
    names = vcat(use_int ? ["(Intercept)"] : String[], string.(covariates))
    isempty(names) && throw(ArgumentError("$(ctx): no regressors"))
    build(colf) = hcat(use_int ? ones(n, 1) : zeros(n, 0),
                       isempty(covariates) ? zeros(n, 0) :
                       reduce(hcat, [colf(c) for c in covariates]))
    yo = col_o(outcome)
    Xo = build(col_o)
    nrep = size(F, 2)
    all_coef = Matrix{Float64}(undef, length(names), nrep)
    all_vcov = Vector{Matrix{Float64}}(undef, nrep)
    for r in 1:nrep
        pred_of = Dict(v => Gp[:, k, r] for (k, v) in enumerate(predicted_vars))
        col_p(c) = haskey(pred_of, c) ? pred_of[c] : obs[c]
        β, V = _ml_dsl_fit(family, fe, data, yo, col_p(outcome), Xo, build(col_p), w,
                           covariates, cl; context=ctx)
        all_coef[:, r] = β
        all_vcov[r] = V
    end
    θ, V = _ml_aggregate(all_coef, all_vcov)
    # naive plug-in of the raw predictions
    nb, nV = if length(preds) == length(predicted_vars)
        raw = Dict(v => _ml_column(data, preds[k]; context=ctx)
                   for (k, v) in enumerate(predicted_vars))
        col_r(c) = haskey(raw, c) ? raw[c] : obs[c]
        yr = col_r(outcome)
        if family === :binomial && !all(v -> 0 <= v <= 1, yr)
            Float64[], zeros(0, 0)
        else
            _ml_dsl_fit(family, fe, data, yr, yr, build(col_r), build(col_r), zeros(n),
                        covariates, cl; context=ctx)
        end
    else
        Float64[], zeros(0, 0)
    end
    g1 = Gp[:, :, 1]
    rmse = [sqrt(mean(abs2, gold[k][R] .- g1[R, k])) for k in eachindex(gold)]
    model = !isempty(fe) ? "fixed-effects regression" :
            family === :gaussian ? "linear regression" : "logistic regression"
    details = (family=family, fe=fe, predicted_vars=predicted_vars, prob=π,
               labeled=R, fitted=g1, rmse=rmse, folds=F, rep_coef=all_coef,
               rep_vcov=all_vcov, learner=learner === nothing ? "none (raw prediction)" :
                                          _ml_learner_name(learner))
    return MeasurementEstimate(names, θ, V, "DSL " * model,
                               "coefficients of the $(model) on the gold-standard " *
                               "variables", n, count(R), G, nb, nV, details)
end

"""One DSL fit for given gold (`o`) and predicted (`p`) data. Returns (β, V)."""
function _ml_dsl_fit(family, fe, data, yo, yp, Xo, Xp, w, covariates, cl; context)
    if isempty(fe)
        β, M, J = family === :gaussian ?
                  _ml_dsl_linear(yo, yp, Xo, Xp, w; context=context) :
                  _ml_dsl_logit(yo, yp, Xo, Xp, w; context=context)
        return β, _ml_meas_sandwich(M, J, cl)
    end
    # Fixed effects: within regression of the pseudo-outcome (only the outcome is
    # ML-measured, so Xo == Xp).
    ỹ = yp .+ w .* (yo .- yp)
    return _ml_meas_fe_fit(data, ỹ, covariates, fe, cl; context=context)
end

"""Within regression of `y` on `covariates` with fixed effects `fe`; sandwich (CR0)."""
function _ml_meas_fe_fit(data, y::AbstractVector, covariates, fe, cl; context)
    ycol = :__ml_meas_outcome
    ycol in propertynames(data) &&
        throw(ArgumentError("$(context): reserved column name $(ycol) is in use"))
    df = DataFrame([c => data[!, c] for c in unique(vcat(covariates, fe))])
    df[!, ycol] = y
    m = FixedEffectModels.reg(df, make_formula(ycol, covariates; fe=fe); tol=1e-12,
                              maxiter=100_000)
    β = [StatsAPI.coef(m)[coef_index(m, c)] for c in covariates]
    P = _ml_meas_partial_out(df, vcat(ycol, covariates), fe)
    ỹ = P[:, 1]
    X̃ = P[:, 2:end]
    e = ỹ .- X̃ * β
    n = length(e)
    J = (X̃' * X̃) ./ n
    return β, _ml_meas_sandwich(X̃ .* e, J, cl)
end

"""
    dsl_proportions(data, category; prediction=nothing, features=Symbol[],
                    levels=nothing, by=nothing, labeled=nothing, label_prob=nothing,
                    normalize_prob=true, cluster=nothing, learner=OLSLearner(),
                    n_folds=5, n_rep=1, folds=nothing, rng=Random.default_rng(),
                    parallel=false) -> MeasurementEstimate

Design-based supervised learning estimate of category proportions, optionally
within groups such as years, when documents are classified by an ML model or
LLM and a subsample selected with known probabilities is coded by experts.

The estimands are the population shares ``p_k = P(Y = k)`` (or
``P(Y = k \\mid B = b)`` within the groups ``b`` of `by`) of the gold-standard
category ``Y``. The predicted category is typically misclassified at rates
that differ across categories and over time, so the shares of predicted
categories are biased estimates of these proportions and trends in them can be
artefacts of classifier drift. As in [`dsl_regression`](@ref), identification
requires error-free expert codes on the labelled rows and labelling
probabilities ``\\pi_i`` that are known by design and bounded away from zero;
no assumption is made about the classifier.

For each category ``k``, a cross-fitted model ``\\hat g_k`` of the indicator
``1\\{Y = k\\}`` is trained on the labelled rows from the indicators of the
predicted category (and `features`), and the proportion is the mean (within
group) of the pseudo-outcome

```math
\\hat g_{k,i} + \\frac{R_i}{\\pi_i}\\big(1\\{Y_i = k\\} - \\hat g_{k,i}\\big),
```

an augmented inverse-probability-weighted estimator (Robins, Rotnitzky & Zhao
1994) that is consistent for ``p_k`` whatever the classifier's accuracy. The
covariance of all proportions (all categories and groups) is estimated jointly
from their influence functions, with cluster sums under `cluster`, and
inference uses normal critical values. With `n_rep > 1` repetitions are
aggregated by the median rule described in [`dsl_regression`](@ref). The
naive estimate stored in `naive_coef` is the share of each predicted category.
The corrected proportions are not constrained to ``[0, 1]``: with a rare
category and few labelled rows an estimate can fall slightly outside it, which
signals that the labelled subsample is too small for that category.

# Arguments
- `data::AbstractDataFrame`: one row per document or unit.
- `category::Symbol`: the gold-standard category; `missing` on unlabelled rows
  unless `labeled` is given.

# Keywords
- `prediction::Union{Nothing,Symbol} = nothing`: the predicted category, coded
  like `category` and observed on every row.
- `features::Vector{Symbol} = Symbol[]`: further numeric inputs of
  ``\\hat g_k``.
- `levels = nothing`: categories to report (default: sorted union of the
  observed gold and predicted categories).
- `by::Union{Nothing,Symbol} = nothing`: grouping column (e.g. period);
  proportions are estimated within each group.
- `labeled`, `label_prob`, `normalize_prob`, `cluster`, `n_folds`, `n_rep`,
  `folds`, `rng`, `parallel`: as in [`dsl_regression`](@ref).
- `learner = OLSLearner()`: [`NuisanceLearner`](@ref) for ``\\hat g_k``, or
  `nothing` to use the predicted-category indicators directly (no
  recalibration).

# Returns
- [`MeasurementEstimate`](@ref) with coefficient names `"k"` or
  `"k | by = g"`; `details` has `levels`, `by`, `prob`, `labeled`, `fitted`,
  `folds`, `rep_coef` and `rep_vcov`.

# Examples
```julia
using DrSnow, DataFrames, StableRNGs
rng = StableRNG(4)
n = 3000
year = rand(rng, [2020, 2021], n)
topic = [rand(rng) < (y == 2020 ? 0.3 : 0.5) ? "econ" : "other" for y in year]
wrong = rand(rng, n) .< ifelse.(year .== 2021, 0.25, 0.10)   # classifier drift
topic_llm = ifelse.(wrong, ifelse.(topic .== "econ", "other", "econ"), topic)
lab = rand(rng, n) .< 0.2
df = DataFrame(topic_expert=[l ? t : missing for (l, t) in zip(lab, topic)],
               topic_llm=topic_llm, year=year)
r = dsl_proportions(df, :topic_expert; prediction=:topic_llm, by=:year,
                    rng=StableRNG(5))
coeftable(r)
```

# References
$(_ML_DSL_REFS)
"""
function dsl_proportions(data::AbstractDataFrame, category::Symbol;
                         prediction::Union{Nothing,Symbol}=nothing,
                         features::Vector{Symbol}=Symbol[], levels=nothing,
                         by::Union{Nothing,Symbol}=nothing,
                         labeled::Union{Nothing,Symbol}=nothing,
                         label_prob::Union{Nothing,Symbol}=nothing,
                         normalize_prob::Bool=true,
                         cluster::Union{Nothing,Symbol}=nothing, learner=OLSLearner(),
                         n_folds::Integer=5, n_rep::Integer=1, folds=nothing,
                         rng::AbstractRNG=Random.default_rng(), parallel::Bool=false)
    ctx = "dsl_proportions"
    (prediction === nothing && (isempty(features) || learner === nothing)) &&
        throw(ArgumentError("$(ctx): pass `prediction` (and/or `features` with a " *
                            "learner)"))
    require_columns(data, [category, prediction, by, labeled, label_prob, cluster,
                           features...]; context=ctx)
    _ml_meas_complete(data, [prediction, by, cluster, features...]; context=ctx)
    R, π = _ml_meas_labels(data, [category], labeled, label_prob, normalize_prob;
                           context=ctx)
    n = nrow(data)
    gv = data[!, category]
    pv = prediction === nothing ? nothing : data[!, prediction]
    levs = if levels === nothing
        u = unique(vcat(collect(skipmissing(gv[R])),
                        pv === nothing ? Any[] : collect(pv)))
        try
            sort!(u)
        catch
        end
        u
    else
        collect(levels)
    end
    length(levs) >= 2 || throw(ArgumentError("$(ctx): need at least two categories"))
    K = length(levs)
    for i in 1:n
        R[i] && !(gv[i] in levs) &&
            throw(ArgumentError("$(ctx): gold category $(gv[i]) is not in `levels`"))
    end
    targets = [Float64[R[i] && isequal(gv[i], k) ? 1.0 : 0.0 for i in 1:n] for k in levs]
    P = pv === nothing ? zeros(n, 0) :
        reduce(hcat, [Float64.(isequal.(pv, k)) for k in levs])
    feat = hcat(P, _ml_matrix(data, features; context=ctx))
    # inputs of ĝ: predicted-category dummies (dropping one: the dummies sum to one)
    # and the features
    Xg = size(P, 2) > 0 ? hcat(P[:, 2:end], feat[:, (K + 1):end]) : feat
    cl, G = _ml_meas_clusters(data, cluster; context=ctx)
    groups = cluster === nothing ? nothing : data[!, cluster]
    F = _ml_meas_folds(data, folds, n, n_folds, n_rep, rng, R, groups; context=ctx)
    Gp = if learner === nothing
        pv === nothing && throw(ArgumentError("$(ctx): learner = nothing needs " *
                                              "`prediction`"))
        repeat(reshape(P, n, K, 1), 1, 1, size(F, 2))
    else
        size(Xg, 2) >= 1 || throw(ArgumentError("$(ctx): no inputs for the prediction " *
                                                "model"))
        _ml_meas_crossfit(targets, Xg, BitVector(R), learner, F, rng; parallel=parallel,
                          context=ctx)
    end
    bylev, D = _ml_meas_group_dummies(data, by)
    names = [by === nothing ? string(k) : "$(k) | $(by) = $(b)" for b in bylev
             for k in levs]
    nrep = size(F, 2)
    all_coef = Matrix{Float64}(undef, length(names), nrep)
    all_vcov = Vector{Matrix{Float64}}(undef, nrep)
    for r in 1:nrep
        Ỹ = reduce(hcat, [_ml_meas_pseudo(Gp[:, k, r], targets[k], R, π) for k in 1:K])
        all_coef[:, r], all_vcov[r] = _ml_meas_group_means(Ỹ, D, cl)
    end
    θ, V = _ml_aggregate(all_coef, all_vcov)
    nb, nV = pv === nothing ? (Float64[], zeros(0, 0)) : _ml_meas_group_means(P, D, cl)
    details = (levels=levs, by=bylev, prob=π, labeled=R, fitted=Gp[:, :, 1], folds=F,
               rep_coef=all_coef, rep_vcov=all_vcov)
    return MeasurementEstimate(names, θ, V, "DSL category proportions",
                               "category proportions of the gold-standard coding", n,
                               count(R), G, nb, nV, details)
end

"""Levels of `by` (or `[nothing]`) and the n × B indicator matrix."""
function _ml_meas_group_dummies(data, by)
    n = nrow(data)
    by === nothing && return Any["all"], ones(n, 1)
    v = data[!, by]
    u = unique(v)
    try
        sort!(u)
    catch
    end
    return u, reduce(hcat, [Float64.(isequal.(v, b)) for b in u])
end

"""
Means of the columns of `Y` within the groups of `D` (n × B indicators), ordered
group-major, with their joint influence-function covariance (cluster sums).
"""
function _ml_meas_group_means(Y::AbstractMatrix, D::AbstractMatrix, cl)
    n, K = size(Y)
    B = size(D, 2)
    θ = Vector{Float64}(undef, K * B)
    Ψ = zeros(n, K * B)
    for b in 1:B
        idx = D[:, b] .== 1
        pb = count(idx) / n
        pb > 0 || throw(ArgumentError("empty group"))
        for k in 1:K
            j = (b - 1) * K + k
            θ[j] = mean(view(Y, idx, k))
            Ψ[idx, j] .= (Y[idx, k] .- θ[j]) ./ pb
        end
    end
    V, _ = _if_vcov(Ψ, cl)
    return θ, V
end
