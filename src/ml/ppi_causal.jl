# Prediction-powered inference for causal targets: average treatment effects and
# treatment-effect regressions (with fixed effects) when the outcome is an ML / LLM
# prediction on every unit and the true outcome is observed on a labelled subsample
# drawn with known probabilities; and cross-prediction-powered inference when the
# predictor is trained on the labelled data themselves (Zrnic & Candès 2024).

const _ML_PPI_CAUSAL_REFS = """
- Angelopoulos, A. N., Bates, S., Fannjiang, C., Jordan, M. I., & Zrnic, T.
  (2023). Prediction-powered inference. *Science*, 382(6671), 669–674.
- Angelopoulos, A. N., Duchi, J. C., & Zrnic, T. (2023). PPI++: Efficient
  prediction-powered inference. arXiv:2311.01453.
- Egami, N., Hinck, M., Stewart, B. M., & Wei, H. (2023). Using imperfect
  surrogates for downstream inference: Design-based supervised learning for
  social science applications of large language models. *Advances in Neural
  Information Processing Systems*, 36, 68589–68601.
- Robins, J. M., Rotnitzky, A., & Zhao, L. P. (1994). Estimation of regression
  coefficients when some regressors are not always observed. *Journal of the
  American Statistical Association*, 89(427), 846–866."""

"""
Power-tuned pseudo-outcome regression. `X` (n × p) is the design (already
residualized on fixed effects when `fe` is non-empty, via the `resid` closure).
Returns (β, V, λ, β_naive, V_naive, β_λ0, V_λ0).
"""
function _ml_ppi_pseudo_fit(y, f, R, π, X::Matrix{Float64}, resid, cl, lambda, target;
                            context)
    n, p = size(X)
    w = R ./ π
    H = (X' * X) ./ n
    Fh = lu(H; check=false)
    (issuccess(Fh) && rank(H) == p) ||
        throw(ArgumentError("$(context): collinear regressors"))
    Hinv = inv(Fh)
    wy = resid(w .* y)          # (R/π) Y
    bf = resid(f .- w .* f)     # (1 − R/π) f
    fr = resid(f)
    λ = if lambda === :optimal
        # residuals are exactly linear in λ: e(λ) = (wy − Xβ₀) + λ (bf − Xδ)
        β0 = Hinv * (X' * wy) ./ n
        δ = Hinv * (X' * bf) ./ n
        A = _cluster_sums(X .* (wy .- X * β0), cl)
        B = _cluster_sums(X .* (bf .- X * δ), cl)
        C = target === nothing ? Matrix{Float64}(I, p, p) : reshape(target, p, 1)
        U = A * Hinv * C
        W = B * Hinv * C
        den = sum(abs2, W)
        den > 0 ? max(-sum(U .* W) / den, 0.0) : 0.0
    else
        (lambda isa Real && isfinite(lambda) && lambda >= 0) ||
            throw(ArgumentError("$(context): lambda must be :optimal or a non-negative " *
                                "number"))
        Float64(lambda)
    end
    fit(ỹ) = _ml_ppi_ols_sandwich(X, ỹ, Hinv, H, cl)
    β, V = fit(wy .+ λ .* bf)
    βn, Vn = fit(fr)
    β0, V0 = fit(wy)
    return β, V, λ, βn, Vn, β0, V0
end

"""Least squares of `ỹ` on `X` (given `H⁻¹`) and its sandwich covariance."""
function _ml_ppi_ols_sandwich(X, ỹ, Hinv, H, cl)
    b = Hinv * (X' * ỹ) ./ size(X, 1)
    return b, _ml_meas_sandwich(X .* (ỹ .- X * b), H, cl)
end

"""Common input handling for the PPI causal estimators."""
function _ml_ppi_causal_inputs(data, outcome, prediction, cols, labeled, label_prob,
                               normalize_prob, cluster; context)
    require_columns(data, vcat(outcome, prediction, cols, labeled, label_prob, cluster);
                    context=context)
    _ml_meas_complete(data, vcat(prediction, cols, cluster); context=context)
    R, π = _ml_meas_labels(data, [outcome], labeled, label_prob, normalize_prob;
                           context=context)
    y = _ml_meas_gold(data, outcome, R; context=context)
    f = _ml_column(data, prediction; context=context)
    cl, G = _ml_meas_clusters(data, cluster; context=context)
    return R, π, y, f, cl, G
end

"""Residualizer on fixed effects (identity without `fe`)."""
function _ml_ppi_resid(data, fe::Vector{Symbol})
    isempty(fe) && return identity
    df = DataFrame([c => data[!, c] for c in fe])
    return function (v)
        df[!, :__ml_ppi_v] = v
        return _ml_meas_partial_out(df, [:__ml_ppi_v], fe)[:, 1]
    end
end

"""
    ppi_regression(data, outcome, prediction; covariates=Symbol[], fe=Symbol[],
                   intercept=true, target=nothing, lambda=:optimal, labeled=nothing,
                   label_prob=nothing, normalize_prob=true, cluster=nothing)
        -> MeasurementEstimate

Prediction-powered least-squares (optionally fixed-effects) regression of an
outcome that is observed on a labelled subsample and predicted by an ML model or
LLM on every row, with PPI++ power tuning, in the single-sample design where the
labelled rows are drawn from the analysis sample with known probabilities.

The estimand is the vector of coefficients of the least-squares (or, with `fe`,
within) regression of the *true* outcome ``Y`` on the regressors, which must be
observed on every row; only the outcome is ML-measured. Row ``i`` is labelled
(``R_i = 1``) with probability ``\\pi_i``, known by design and possibly
dependent on covariates or treatment. Identification requires error-free gold
labels, labelling probabilities bounded away from zero, and a prediction
``f`` that was not trained on the labels of the analysis sample (otherwise use
[`dsl_regression`](@ref) with a cross-fitted learner, or [`cross_ppi`](@ref));
the prediction may be biased in any way, including differentially by treatment.

The estimator is least squares (within regression with `fe`) of the
pseudo-outcome

```math
\\tilde Y_i(\\lambda) = \\lambda f_i + \\frac{R_i}{\\pi_i}(Y_i - \\lambda f_i),
```

whose conditional mean given the regressors equals that of ``Y_i`` for every
fixed ``\\lambda``, so the regression targets the estimand defined with the true
outcome. ``\\lambda = 1`` is prediction-powered inference in the single-sample
form (Angelopoulos et al. 2023), ``\\lambda = 0`` inverse-probability weighting
of the labelled rows, and the construction is the augmented
inverse-probability-weighted estimator of Robins, Rotnitzky and Zhao (1994)
with the working prediction ``\\lambda f``. Because the residuals are linear in
``\\lambda``, `lambda = :optimal` computes in closed form the ``\\lambda \\ge 0``
that minimizes the estimated variance of the `target` coefficient (the trace of
the covariance when `target = nothing`), as in PPI++ (Angelopoulos, Duchi &
Zrnic 2023); unlike [`ppi_ols`](@ref), it is not capped at one.

The variance is the heteroskedasticity- or cluster-robust sandwich of the
pseudo-outcome regression, without small-sample factors, with normal critical
values; the estimation of ``\\lambda`` is ignored, which is justified
asymptotically. The labelled-only (``\\lambda = 0``) fit and the naive fit that
uses the prediction as outcome are kept for comparison. Cluster at the level at
which treatment was assigned or labels were sampled, whichever is coarser. For
a randomized treatment use [`ppi_ate`](@ref), which adds Lin's (2013)
covariate adjustment; with ML-measured regressors use [`dsl_regression`](@ref).

# Arguments
- `data::AbstractDataFrame`: the analysis sample.
- `outcome::Symbol`: the gold-standard outcome, `missing` when unlabelled
  (unless `labeled` is given).
- `prediction::Symbol`: the ML prediction of the outcome, observed on every row.

# Keywords
- `covariates::Vector{Symbol} = Symbol[]`: regressors, observed on every row.
- `fe::Vector{Symbol} = Symbol[]`: fixed effects absorbed with
  `FixedEffectModels`.
- `intercept::Bool = true`: include an intercept (ignored with `fe`).
- `target::Union{Nothing,Symbol} = nothing`: the coefficient whose variance the
  power tuning minimizes (default: all coefficients' total variance).
- `lambda = :optimal`: `:optimal` or a fixed non-negative number.
- `labeled = nothing`, `label_prob = nothing`, `normalize_prob = true`: labelling
  design, as in [`dsl_pseudo_outcome`](@ref).
- `cluster::Union{Nothing,Symbol} = nothing`: cluster column for the variance.

# Returns
- [`MeasurementEstimate`](@ref); `naive_coef` is the regression of the
  prediction, and `details` holds `lambda`, the ``\\lambda = 0`` fit
  (`ipw_coef`, `ipw_vcov`), `prob` and `labeled`.

# Examples
```julia
using DrSnow, DataFrames, StableRNGs
rng = StableRNG(6)
n = 3000
school = rand(rng, 1:60, n)
block = div.(school .- 1, 10)                       # 6 blocks of 10 schools
treated = Float64.(isodd.(school))                  # school-level assignment
y = 1 .+ 0.5 .* treated .+ 0.2 .* block .+ randn(rng, n)
y_llm = 0.2 .+ 0.8 .* y .+ 0.3 .* treated .+ 0.5 .* randn(rng, n)
lab = rand(rng, n) .< 0.3
df = DataFrame(y_expert=[l ? v : missing for (l, v) in zip(lab, y)], y_llm=y_llm,
               treated=treated, block=block, school=school)
r = ppi_regression(df, :y_expert, :y_llm; covariates=[:treated], fe=[:block],
                   target=:treated, cluster=:school)
coef(r), r.naive_coef, r.details.lambda
```

# References
$(_ML_PPI_CAUSAL_REFS)
- Lin, W. (2013). Agnostic notes on regression adjustments to experimental data:
  Reexamining Freedman's critique. *The Annals of Applied Statistics*, 7(1),
  295–318.
"""
function ppi_regression(data::AbstractDataFrame, outcome::Symbol, prediction::Symbol;
                        covariates::Vector{Symbol}=Symbol[], fe::Vector{Symbol}=Symbol[],
                        intercept::Bool=true, target::Union{Nothing,Symbol}=nothing,
                        lambda=:optimal, labeled::Union{Nothing,Symbol}=nothing,
                        label_prob::Union{Nothing,Symbol}=nothing,
                        normalize_prob::Bool=true,
                        cluster::Union{Nothing,Symbol}=nothing)
    ctx = "ppi_regression"
    R, π, y, f, cl, G = _ml_ppi_causal_inputs(data, outcome, prediction,
                                              vcat(covariates, fe), labeled, label_prob,
                                              normalize_prob, cluster; context=ctx)
    n = nrow(data)
    use_int = intercept && isempty(fe)
    names = vcat(use_int ? ["(Intercept)"] : String[], string.(covariates))
    isempty(names) && throw(ArgumentError("$(ctx): no regressors"))
    resid = _ml_ppi_resid(data, fe)
    X = hcat(use_int ? ones(n, 1) : zeros(n, 0),
             _ml_matrix(data, covariates; context=ctx))
    if !isempty(fe)
        X = reduce(hcat, [resid(X[:, j]) for j in axes(X, 2)])
        X = reshape(X, n, :)
    end
    tvec = nothing
    if target !== nothing
        j = findfirst(==(string(target)), names)
        j === nothing && throw(ArgumentError("$(ctx): target $(target) is not a " *
                                             "regressor"))
        tvec = zeros(length(names))
        tvec[j] = 1.0
    end
    β, V, λ, βn, Vn, β0, V0 = _ml_ppi_pseudo_fit(y, f, R, π, X, resid, cl, lambda, tvec;
                                                 context=ctx)
    model = isempty(fe) ? "linear regression" : "fixed-effects regression"
    return MeasurementEstimate(names, β, V, "PPI++ $(model) (single-sample design)",
                               "coefficients of the $(model) on the true outcome", n,
                               count(R), G, βn, Vn,
                               (lambda=λ, ipw_coef=β0, ipw_vcov=V0, prob=π, labeled=R))
end

"""
    ppi_ate(data, outcome, treatment, prediction; covariates=Symbol[], fe=Symbol[],
            lambda=:optimal, labeled=nothing, label_prob=nothing, normalize_prob=true,
            cluster=nothing) -> MeasurementEstimate

Average treatment effect in a randomized experiment whose outcome is predicted by
an ML model or LLM for every unit and measured by experts on a labelled
subsample, by prediction-powered inference with PPI++ power tuning.

The estimand is the average treatment effect ``\\tau = E[Y(1) - Y(0)]`` of a
binary, randomized treatment on the *true* outcome (with `fe`, the
regression-weighted average of block effects described below). The prediction
``f`` is observed for every unit, the gold standard ``Y`` for the labelled units,
selected with probabilities ``\\pi_i`` that are known by design and may depend on
treatment and covariates. Beyond randomization (and SUTVA), identification
requires error-free gold labels, labelling probabilities bounded away from zero
in both arms, and a prediction not trained on the labels of the experiment. The
prediction may be arbitrarily biased, including differently in the two arms,
for example when an LLM coder reacts to treatment-induced wording, which is the
case in which the naive difference in predicted outcomes is biased: the
correction uses the labelled units of both arms.

The estimator applies the pseudo-outcome regression of [`ppi_regression`](@ref),
``\\tilde Y_i(\\lambda) = \\lambda f_i + (R_i/\\pi_i)(Y_i - \\lambda f_i)``, to
one of three designs. Without covariates it is the difference in means of the
pseudo-outcome between arms. With `covariates` it is the fully interacted
regression adjustment of Lin (2013), with arm-specific slopes on covariates
centred at their full-sample means, and the ATE is the treatment coefficient.
With `fe` it is a treatment-effect regression with absorbed fixed effects (for
example the blocks of a block-randomized design); the coefficient is then a
conditional-variance-weighted average of block effects, which equals the ATE
only when effects are homogeneous or the treated share is constant across
blocks. `lambda = :optimal` chooses the ``\\lambda \\ge 0`` that minimizes the
estimated variance of the ATE coefficient (Angelopoulos, Duchi & Zrnic 2023).

The standard error is the heteroskedasticity- or cluster-robust sandwich of the
pseudo-outcome regression without small-sample factors, with normal critical
values. In a cluster-randomized experiment, or when labels were sampled by
cluster, pass `cluster`. Report the labelled share in each arm, the naive
estimate (`naive_coef`) and the labelled-only inverse-probability-weighted
estimate (`details.ipw_coef`) alongside the prediction-powered one. When the
labelling probabilities are unknown, none of these estimators applies; a
differential-error diagnostic such as [`differential_error_test`](@ref) can
indicate a problem with the naive estimate but cannot validate it.

# Arguments
- `data::AbstractDataFrame`: one row per experimental unit.
- `outcome::Symbol`: the gold-standard outcome, `missing` when unlabelled (unless
  `labeled` is given).
- `treatment::Symbol`: the randomized treatment, coded 0/1.
- `prediction::Symbol`: the ML prediction of the outcome, observed for every unit.

# Keywords
- `covariates::Vector{Symbol} = Symbol[]`: pre-treatment covariates for Lin's
  adjustment (or additional regressors with `fe`).
- `fe::Vector{Symbol} = Symbol[]`: fixed effects (e.g. randomization blocks).
- `lambda = :optimal`: `:optimal` or a fixed non-negative number.
- `labeled = nothing`, `label_prob = nothing`, `normalize_prob = true`: labelling
  design, as in [`dsl_pseudo_outcome`](@ref).
- `cluster::Union{Nothing,Symbol} = nothing`: cluster column for the variance.

# Returns
- [`MeasurementEstimate`](@ref) with the single coefficient `"ATE"`; `naive_coef`
  is the same contrast computed with the prediction as outcome, and `details`
  holds `lambda`, the full regression (`full_names`, `full_coef`, `full_vcov`),
  the labelled-only estimate (`ipw_coef`, `ipw_se`), `prob` and `labeled`.

# Examples
```julia
using DrSnow, DataFrames, StableRNGs
rng = StableRNG(7)
n = 2000
age = randn(rng, n)
treated = Float64.(rand(rng, n) .< 0.5)
y = 1 .+ 1.0 .* treated .+ 0.5 .* age .+ randn(rng, n)
y_llm = 0.3 .+ 0.8 .* y .+ 0.5 .* treated .+ 0.6 .* randn(rng, n)
p_label = ifelse.(treated .== 1, 0.2, 0.3)                 # known design
lab = rand(rng, n) .< p_label
df = DataFrame(y_expert=[l ? v : missing for (l, v) in zip(lab, y)], y_llm=y_llm,
               treated=treated, age=age, p_label=p_label)
r = ppi_ate(df, :y_expert, :treated, :y_llm; covariates=[:age],
            label_prob=:p_label, normalize_prob=false)
coef(r), confint(r), r.naive_coef
```

# References
$(_ML_PPI_CAUSAL_REFS)
- Lin, W. (2013). Agnostic notes on regression adjustments to experimental data:
  Reexamining Freedman's critique. *The Annals of Applied Statistics*, 7(1),
  295–318.
- Imbens, G. W., & Rubin, D. B. (2015). *Causal Inference for Statistics,
  Social, and Biomedical Sciences: An Introduction*. Cambridge University Press.
"""
function ppi_ate(data::AbstractDataFrame, outcome::Symbol, treatment::Symbol,
                 prediction::Symbol; covariates::Vector{Symbol}=Symbol[],
                 fe::Vector{Symbol}=Symbol[], lambda=:optimal,
                 labeled::Union{Nothing,Symbol}=nothing,
                 label_prob::Union{Nothing,Symbol}=nothing, normalize_prob::Bool=true,
                 cluster::Union{Nothing,Symbol}=nothing)
    ctx = "ppi_ate"
    R, π, y, f, cl, G = _ml_ppi_causal_inputs(data, outcome, prediction,
                                              vcat(treatment, covariates, fe), labeled,
                                              label_prob, normalize_prob, cluster;
                                              context=ctx)
    n = nrow(data)
    d = _ml_column(data, treatment; context=ctx)
    _ml_check_binary_col(d, treatment, ctx)
    (any(R .& (d .== 1)) && any(R .& (d .== 0))) ||
        throw(ArgumentError("$(ctx): need labelled units in both treatment arms"))
    Z = _ml_matrix(data, covariates; context=ctx)
    names = ["(Intercept)", "ATE"]
    X, resid = if isempty(fe)
        Zc = Z .- mean(Z; dims=1)
        append!(names, string.(covariates), ["ATE × " * string(c) for c in covariates])
        hcat(ones(n), d, Zc, d .* Zc), identity
    else
        names = vcat(["ATE"], string.(covariates))
        rs = _ml_ppi_resid(data, fe)
        reshape(reduce(hcat, [rs(v) for v in eachcol(hcat(d, Z))]), n, :), rs
    end
    j = findfirst(==("ATE"), names)
    tvec = zeros(length(names))
    tvec[j] = 1.0
    β, V, λ, βn, Vn, β0, V0 = _ml_ppi_pseudo_fit(y, f, R, π, X, resid, cl, lambda, tvec;
                                                 context=ctx)
    return MeasurementEstimate(["ATE"], [β[j]], fill(V[j, j], 1, 1),
                               "PPI++ average treatment effect", "ATE", n, count(R), G,
                               [βn[j]], fill(Vn[j, j], 1, 1),
                               (lambda=λ, full_names=names, full_coef=β, full_vcov=V,
                                ipw_coef=β0[j], ipw_se=sqrt(V0[j, j]), prob=π,
                                labeled=R))
end

"""
    cross_ppi(labeled, unlabeled, outcome; features, covariates=Symbol[],
              intercept=true, family=:gaussian, learner=OLSLearner(), n_folds=10,
              lambda=:optimal, rng=Random.default_rng()) -> MeasurementEstimate

Cross-prediction-powered inference (Zrnic & Candès 2024): prediction-powered
estimates of a mean or of linear or logistic regression coefficients when no
pre-trained predictor exists and the predictor must be trained on the labelled
data themselves.

The estimands are those of [`ppi_mean`](@ref), [`ppi_ols`](@ref) and
[`ppi_logistic`](@ref): the population mean of the outcome (no covariates), or
the population least-squares or logistic coefficients of the outcome on
`covariates`. The labelled units must be a random sample from the population of
the unlabelled units, with error-free outcomes, and `features` (e.g. text
embeddings) and `covariates` must be observed in both samples. Training a
predictor on the labelled sample and then using its in-sample predictions in
[`ppi_mean`](@ref) would bias the correction term, because the predictions
overfit the very labels used to debias them.

Cross-prediction avoids this. The labelled sample is split into `n_folds`
folds; a model of the outcome on `features` is trained on all folds but one;
each labelled unit receives the prediction of the model that did not see it,
and each unlabelled unit the average of the `n_folds` models' predictions. The
PPI++ estimator is then applied to these predictions, so that every label is
used both for training and for debiasing. The covariance is the PPI++ sandwich
computed with the cross-fitted predictions and power tuning as in
[`ppi_ols`](@ref) (for the mean, ``\\hat\\lambda`` is truncated at zero only);
Zrnic and Candès (2024) show that this is asymptotically valid under a
stability condition on the learning algorithm, so very unstable learners may
give intervals that are too short. Inference uses normal critical values.

Compared with a predictor trained on external data, cross-prediction spends no
labels on a separate training split; compared with the labelled-only estimate
(`details.classical_coef`), it gains precision to the extent that the features
predict the outcome. When the labelled units are a subsample of the analysis
sample selected with known, unequal probabilities, use
[`dsl_regression`](@ref) instead, which cross-fits the prediction model in the
same way and handles the labelling design.

# Arguments
- `labeled::AbstractDataFrame`: labelled sample with `outcome`, `features` and
  `covariates`.
- `unlabeled::AbstractDataFrame`: unlabelled sample with `features` and
  `covariates`.
- `outcome::Symbol`: the true outcome (0/1 for `family = :binomial`).

# Keywords
- `features::Vector{Symbol}`: inputs of the predictor (required, non-empty).
- `covariates::Vector{Symbol} = Symbol[]`: regressors of the model of interest;
  with no covariates and an intercept the estimand is the mean.
- `intercept::Bool = true`: include an intercept.
- `family::Symbol = :gaussian`: `:gaussian` (mean or linear regression) or
  `:binomial` (logistic regression; predictions are clamped to ``[0, 1]``).
- `learner = OLSLearner()`: the [`NuisanceLearner`](@ref) trained on each set of
  folds (binary outcomes use `fitpredict_proba` when available).
- `n_folds::Integer = 10`: number of cross-prediction folds.
- `lambda = :optimal`: power tuning, `:optimal` or a fixed number (in ``[0, 1]``
  for the logistic model).
- `rng = Random.default_rng()`: random-number generator for the folds and the
  learners.

# Returns
- [`MeasurementEstimate`](@ref); `nobs` counts both samples and `n_labeled` the
  labelled one. `details` has `lambda`, the labelled-only estimate
  (`classical_coef`, `classical_vcov`), `n_unlabeled`, `n_folds`, the
  out-of-fold predictions of the labelled units (`oof_prediction`) and the
  averaged predictions of the unlabelled units (`unlabeled_prediction`).

# Examples
```julia
using DrSnow, DataFrames, StableRNGs
rng = StableRNG(14)
m = 4000
e1, e2 = randn(rng, m), randn(rng, m)
treated = Float64.(rand(rng, m) .< 0.5)
y = 0.5 .* treated .+ e1 .- 0.5 .* e2 .+ 0.5 .* randn(rng, m)
df = DataFrame(; y, e1, e2, treated)
lab, unlab = df[1:300, :], df[301:end, :]
r = cross_ppi(lab, unlab, :y; features=[:e1, :e2], rng=StableRNG(15))
confint(r), r.details.classical_coef
cross_ppi(lab, unlab, :y; features=[:e1, :e2], covariates=[:treated],
          rng=StableRNG(15))
```

# References
- Zrnic, T., & Candès, E. J. (2024). Cross-prediction-powered inference.
  *Proceedings of the National Academy of Sciences*, 121(15), e2322083121.
- Angelopoulos, A. N., Bates, S., Fannjiang, C., Jordan, M. I., & Zrnic, T.
  (2023). Prediction-powered inference. *Science*, 382(6671), 669–674.
- Angelopoulos, A. N., Duchi, J. C., & Zrnic, T. (2023). PPI++: Efficient
  prediction-powered inference. arXiv:2311.01453.
- Chernozhukov, V., Chetverikov, D., Demirer, M., Duflo, E., Hansen, C., Newey,
  W., & Robins, J. (2018). Double/debiased machine learning for treatment and
  structural parameters. *The Econometrics Journal*, 21(1), C1–C68.
"""
function cross_ppi(labeled::AbstractDataFrame, unlabeled::AbstractDataFrame,
                   outcome::Symbol; features::Vector{Symbol},
                   covariates::Vector{Symbol}=Symbol[], intercept::Bool=true,
                   family::Symbol=:gaussian, learner=OLSLearner(), n_folds::Integer=10,
                   lambda=:optimal, rng::AbstractRNG=Random.default_rng())
    ctx = "cross_ppi"
    family in (:gaussian, :binomial) ||
        throw(ArgumentError("$(ctx): family must be :gaussian or :binomial"))
    isempty(features) && throw(ArgumentError("$(ctx): `features` must be non-empty"))
    learner isa NuisanceLearner ||
        throw(ArgumentError("$(ctx): learner must be a NuisanceLearner"))
    require_columns(labeled, vcat(outcome, features, covariates);
                    context=ctx * " (labeled)")
    require_columns(unlabeled, vcat(features, covariates); context=ctx * " (unlabeled)")
    Y = _ml_column(labeled, outcome; context=ctx)
    Z = _ml_matrix(labeled, features; context=ctx)
    Zu = _ml_matrix(unlabeled, features; context=ctx)
    n, N = length(Y), size(Zu, 1)
    N >= 1 || throw(ArgumentError("$(ctx): the unlabeled sample is empty"))
    family === :binomial && !_ml_meas_is_binary(Y) &&
        throw(ArgumentError("$(ctx): the outcome must be 0/1 for family = :binomial"))
    F = crossfit_folds(n, n_folds, 1; rng=rng)[:, 1]
    seeds = task_seeds(rng, n_folds)
    proba = _ml_meas_is_binary(Y) && _ml_meas_has_proba(learner)
    f = zeros(n)
    fu = zeros(N)
    for k in 1:n_folds
        test = F .== k
        train = .!test
        Xnew = vcat(Z[test, :], Zu)
        trng = Random.Xoshiro(seeds[k])
        p = proba ? fitpredict_proba(learner, Z[train, :], Y[train], Xnew; rng=trng) :
            fitpredict(learner, Z[train, :], Y[train], Xnew; rng=trng)
        all(isfinite, p) || throw(ArgumentError("$(ctx): non-finite predictions"))
        nt = count(test)
        f[test] = p[1:nt]
        fu .+= p[(nt + 1):end] ./ n_folds
    end
    if family === :binomial
        clamp!(f, 0.0, 1.0)
        clamp!(fu, 0.0, 1.0)
    end
    X = _ml_matrix(labeled, covariates; context=ctx)
    Xu = _ml_matrix(unlabeled, covariates; context=ctx)
    if intercept
        X = hcat(ones(n), X)
        Xu = hcat(ones(N), Xu)
    end
    size(X, 2) >= 1 || throw(ArgumentError("$(ctx): no regressors"))
    ismean = isempty(covariates) && intercept && family === :gaussian
    names = ismean ? [string(outcome)] : _ml_ppi_names(covariates, intercept)
    pf = _ml_ppi_fit(family, X, Y, f, Xu, fu, lambda, !ismean, names, "cross-PPI++")
    model = ismean ? "mean" : family === :gaussian ? "linear regression" :
            "logistic regression"
    return MeasurementEstimate(names, pf.coef, pf.vcov, "Cross-PPI++ " * model,
                               ismean ? "population mean" :
                               "population regression coefficients", n + N, n, 0,
                               Float64[], zeros(0, 0),
                               (lambda=pf.lambda, classical_coef=pf.classical_coef,
                                classical_vcov=pf.classical_vcov, n_unlabeled=N,
                                n_folds=n_folds, oof_prediction=f,
                                unlabeled_prediction=fu))
end
