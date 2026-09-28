# Prediction-powered inference (Angelopoulos, Bates, Fannjiang, Jordan & Zrnic 2023)
# with power tuning (PPI++; Angelopoulos, Duchi & Zrnic 2023): valid inference on
# means and linear / logistic regression coefficients when a small labeled sample
# has the true outcome and a machine-learning (or LLM) prediction, and a large
# unlabeled sample has only the prediction.
#
# For a convex loss ℓ, the PPI++ estimator minimizes
#   L_λ(θ) = (1/n) Σ ℓ(Xᵢ, Yᵢ; θ) + λ [(1/N) Σ ℓ(X̃ⱼ, f̃ⱼ; θ) - (1/n) Σ ℓ(Xᵢ, fᵢ; θ)]
# and √n(θ̂ - θ) → N(0, H⁻¹ [Cov(∇ℓ - λ∇ℓᶠ) + λ² (n/N) Cov(∇ℓᶠ)] H⁻¹).

"""
    PPIEstimate <: CausalEstimate

Result of the two-sample prediction-powered estimators [`ppi_mean`](@ref),
[`ppi_ols`](@ref) and [`ppi_logistic`](@ref).

The object holds a PPI++ estimate of a population mean or of population
regression coefficients computed from a small labelled sample, in which both the
true outcome and a machine-learning prediction are observed, and a large
unlabelled sample, in which only the prediction is observed (Angelopoulos et al.
2023; Angelopoulos, Duchi & Zrnic 2023). The covariance is the plug-in estimate
of the asymptotic sandwich covariance at the chosen power-tuning parameter; the
labelled-only ("classical") estimate is stored alongside so that the efficiency
gain from the predictions can be reported. Inference uses normal critical values
(`dof_residual` is `Inf`).

`coef`, `vcov`, `stderror`, `confint(r; level)`, `coeftable`, `coefnames`,
`estimand` and `method_name` work as for every [`CausalEstimate`](@ref); `nobs`
returns the number of labelled observations.

# Fields
- `names::Vector{String}`: coefficient names (the outcome name for a mean,
  `"(Intercept)"` and the covariate names for a regression).
- `coef::Vector{Float64}`: prediction-powered point estimates.
- `vcov::Matrix{Float64}`: their estimated covariance, already divided by the
  labelled sample size.
- `lambda::Float64`: the power-tuning parameter used (`0` ignores the
  predictions and reproduces the labelled-only estimator; `1` is the original
  prediction-powered estimator of Angelopoulos et al. 2023).
- `classical_coef::Vector{Float64}`, `classical_vcov::Matrix{Float64}`: the
  labelled-only estimate and its heteroskedasticity-robust (HC0) sandwich
  covariance, for comparison.
- `n_labeled::Int`, `n_unlabeled::Int`: sizes of the labelled and unlabelled
  samples.
- `method::String`: description of the estimator (e.g. `"PPI++ mean"`).
"""
struct PPIEstimate <: CausalEstimate
    names::Vector{String}
    coef::Vector{Float64}
    vcov::Matrix{Float64}
    lambda::Float64
    classical_coef::Vector{Float64}
    classical_vcov::Matrix{Float64}
    n_labeled::Int
    n_unlabeled::Int
    method::String
end

StatsAPI.coef(r::PPIEstimate) = r.coef
StatsAPI.vcov(r::PPIEstimate) = r.vcov
StatsAPI.coefnames(r::PPIEstimate) = r.names
StatsAPI.nobs(r::PPIEstimate) = r.n_labeled
estimand(r::PPIEstimate) = r.method == "PPI++ mean" ? "population mean" :
                           "population regression coefficients"
method_name(r::PPIEstimate) = r.method

function show_details(io::IO, r::PPIEstimate)
    println(io)
    @printf(io, "Labeled n = %d, unlabeled N = %d, λ = %.4g\n", r.n_labeled,
            r.n_unlabeled, r.lambda)
    ratio = diag(r.classical_vcov) ./ diag(r.vcov)
    println(io, "Variance ratio (labeled-only / PPI) by coefficient: ",
            join((@sprintf("%.3g", x) for x in ratio), ", "))
    return nothing
end

_ml_ppi_names(covs, intercept) = vcat(intercept ? ["(Intercept)"] : String[], string.(covs))

function _ml_ppi_check_lambda(lambda)
    lambda === :optimal && return nothing
    lambda isa Real || throw(ArgumentError("lambda must be :optimal or a number"))
    lambda >= 0 || throw(ArgumentError("lambda must be non-negative"))
    return nothing
end

# Gradients (rows) of the loss at θ for labels `t`.
function _ml_ppi_grads(family, X, t, θ)
    η = X * θ
    r = family === :gaussian ? η .- t : _ml_sigmoid.(η) .- t
    return X .* r
end

function _ml_ppi_hessian(family, Xall, θ)
    family === :gaussian && return (Xall' * Xall) ./ size(Xall, 1)
    p = _ml_sigmoid.(Xall * θ)
    return (Xall' * (Xall .* (p .* (1 .- p)))) ./ size(Xall, 1)
end

_ml_cov(G) = (C = G .- mean(G; dims=1); (C' * C) ./ size(G, 1))

# Point estimate of the PPI++ objective at a given λ.
function _ml_ppi_solve(family, X, Y, f, Xu, fu, λ)
    n, N = size(X, 1), size(Xu, 1)
    if family === :gaussian
        H = (1 - λ) .* (X' * X) ./ n .+ λ .* (Xu' * Xu) ./ N
        b = X' * (Y .- λ .* f) ./ n .+ λ .* (Xu' * fu) ./ N
        return Symmetric(H) \ b
    end
    # logistic: Newton with step halving on the (convex for λ ≤ 1) objective
    obj(θ) = begin
        η = X * θ
        ηu = Xu * θ
        sp(x) = x > 0 ? x + log1p(exp(-x)) : log1p(exp(x))
        (sum(-(Y .- λ .* f) .* η .+ (1 - λ) .* sp.(η)) / n +
         λ * sum(-fu .* ηu .+ sp.(ηu)) / N)
    end
    θ = zeros(size(X, 2))
    fθ = obj(θ)
    for _ in 1:100
        p = _ml_sigmoid.(X * θ)
        pu = _ml_sigmoid.(Xu * θ)
        g = X' * ((1 - λ) .* p .- Y .+ λ .* f) ./ n .+ λ .* (Xu' * (pu .- fu)) ./ N
        H = (1 - λ) .* (X' * (X .* (p .* (1 .- p)))) ./ n .+
            λ .* (Xu' * (Xu .* (pu .* (1 .- pu)))) ./ N
        step = (Symmetric(H) + 1e-12I) \ g
        t = 1.0
        new = obj(θ .- step)
        while new > fθ && t > 1e-8
            t /= 2
            new = obj(θ .- t .* step)
        end
        θ = θ .- t .* step
        done = abs(fθ - new) <= 1e-14 * (abs(new) + 1e-14)
        fθ = new
        done && break
    end
    return θ
end

# Asymptotic covariance of θ̂_λ (already divided by n) and the power-tuning λ̂.
function _ml_ppi_cov(family, X, Y, f, Xu, fu, θ, λ)
    n, N = size(X, 1), size(Xu, 1)
    Hinv = inv(Symmetric(_ml_ppi_hessian(family, vcat(X, Xu), θ)))
    gY = _ml_ppi_grads(family, X, Y, θ)
    gf = _ml_ppi_grads(family, X, f, θ)
    gu = _ml_ppi_grads(family, Xu, fu, θ)
    V = Hinv * (_ml_cov(gY .- λ .* gf) ./ n .+ λ^2 .* _ml_cov(gu) ./ N) * Hinv
    return Matrix(Symmetric(V))
end

function _ml_ppi_lambda(family, X, Y, f, Xu, fu, θ)
    n, N = size(X, 1), size(Xu, 1)
    Hinv = inv(Symmetric(_ml_ppi_hessian(family, vcat(X, Xu), θ)))
    gY = _ml_ppi_grads(family, X, Y, θ)
    gf = _ml_ppi_grads(family, X, f, θ)
    gu = _ml_ppi_grads(family, Xu, fu, θ)
    CY = gY .- mean(gY; dims=1)
    Cf = gf .- mean(gf; dims=1)
    Cyf = (CY' * Cf) ./ n
    Vf = _ml_cov(vcat(gf, gu))
    num = tr(Hinv * (Cyf .+ Cyf') * Hinv)
    den = 2 * (1 + n / N) * tr(Hinv * Vf * Hinv)
    return den > 0 ? num / den : 0.0
end

function _ml_ppi_fit(family, X, Y, f, Xu, fu, lambda, clip, names, method)
    n, N = size(X, 1), size(Xu, 1)
    n > size(X, 2) || throw(ArgumentError("too few labeled observations"))
    N >= 1 || throw(ArgumentError("need at least one unlabeled observation"))
    _ml_ppi_check_lambda(lambda)
    λ = if lambda === :optimal
        θ1 = _ml_ppi_solve(family, X, Y, f, Xu, fu, 1.0)
        l = _ml_ppi_lambda(family, X, Y, f, Xu, fu, θ1)
        clip ? clamp(l, 0.0, 1.0) : max(l, 0.0)
    else
        Float64(lambda)
    end
    family === :binomial && λ > 1 &&
        throw(ArgumentError("lambda must be in [0, 1] for the logistic model"))
    θ = _ml_ppi_solve(family, X, Y, f, Xu, fu, λ)
    V = _ml_ppi_cov(family, X, Y, f, Xu, fu, θ, λ)
    θc = _ml_ppi_solve(family, X, Y, f, X, f, 0.0)
    Hc = inv(Symmetric(_ml_ppi_hessian(family, X, θc)))
    Vc = Matrix(Symmetric(Hc * _ml_cov(_ml_ppi_grads(family, X, Y, θc)) * Hc ./ n))
    return PPIEstimate(names, θ, V, λ, θc, Vc, n, N, method)
end

function _ml_ppi_inputs(labeled, unlabeled, outcome, prediction, covs, intercept, ctx)
    require_columns(labeled, vcat(outcome, prediction, covs); context=ctx * " (labeled)")
    require_columns(unlabeled, vcat(prediction, covs); context=ctx * " (unlabeled)")
    Y = _ml_column(labeled, outcome; context=ctx)
    f = _ml_column(labeled, prediction; context=ctx)
    fu = _ml_column(unlabeled, prediction; context=ctx)
    X = _ml_matrix(labeled, covs; context=ctx)
    Xu = _ml_matrix(unlabeled, covs; context=ctx)
    if intercept
        X = hcat(ones(size(X, 1)), X)
        Xu = hcat(ones(size(Xu, 1)), Xu)
    end
    size(X, 2) >= 1 || throw(ArgumentError("$ctx: no regressors"))
    return X, Y, f, Xu, fu
end

"""
    ppi_mean(labeled, unlabeled, outcome, prediction; lambda=:optimal) -> PPIEstimate
    ppi_mean(y, yhat, yhat_unlabeled; lambda=:optimal) -> PPIEstimate

Prediction-powered estimate of a population mean from a small labelled sample and
a large sample in which only a machine-learning prediction of the outcome is
available, with PPI++ power tuning.

The estimand is the population mean ``\\theta = E[Y]`` of an outcome that is
costly to measure (for example an expert coding of a document) and that a fixed
predictor ``f`` (a trained classifier, an LLM annotation) approximates cheaply.
The data are ``n`` labelled units with ``(Y_i, f_i)`` and ``N`` unlabelled units
with ``\\tilde f_j`` only. Identification requires that the labelled units be a
random sample from the same population as the unlabelled ones (equivalently,
that units were selected for labelling completely at random), that the labels
be error-free measurements of ``Y``, and that the predictor was not trained on
the labelled sample; the predictions themselves may be arbitrarily biased or
noisy. The estimator of Angelopoulos et al. (2023), generalized by the PPI++
weight ``\\lambda`` of Angelopoulos, Duchi and Zrnic (2023), is

```math
\\hat\\theta_\\lambda = \\frac{1}{N} \\sum_{j=1}^{N} \\lambda \\tilde f_j
    + \\frac{1}{n} \\sum_{i=1}^{n} (Y_i - \\lambda f_i),
```

the labelled-sample mean plus a prediction-based correction whose expectation
is zero: ``\\hat\\theta_\\lambda`` is unbiased for every fixed ``\\lambda``, and
the predictions affect only its precision. The construction is the classical
augmented estimator for data missing completely at random (Robins, Rotnitzky &
Zhao 1994) with a fixed working model ``\\lambda f``.

The variance ``\\operatorname{Var}(Y - \\lambda f)/n + \\lambda^2
\\operatorname{Var}(\\tilde f)/N`` is estimated by sample moments. With
`lambda = :optimal` the estimate ``\\hat\\lambda = \\widehat{\\operatorname{Cov}}
(Y, f) / \\{(1 + n/N) \\widehat{\\operatorname{Var}}(f)\\}`` (with the variance
of ``f`` pooled over both samples) minimizes the estimated variance, so the
prediction-powered interval is asymptotically never wider than the
labelled-only interval, and it is much narrower when the predictions are
strongly correlated with the outcome. For the mean ``\\hat\\lambda`` is
truncated at zero but not at one. Intervals use normal critical values;
estimating ``\\lambda`` does not change the first-order asymptotics, but with a
few dozen labels the normal approximation and the variance estimate can be
poor.

Use [`ppi_ols`](@ref) or [`ppi_logistic`](@ref) for regression coefficients,
[`cross_ppi`](@ref) when the predictor has to be trained on the labelled data
themselves, and [`dsl_regression`](@ref) or [`ppi_regression`](@ref) when the
labelled units are a subsample of the analysis sample drawn with known,
possibly unequal, probabilities. Report the two sample sizes, the power-tuning
parameter and the labelled-only estimate (`classical_coef`), which the printed
output compares with the prediction-powered one.

# Arguments
- `labeled::AbstractDataFrame`: the labelled sample; it must contain `outcome`
  and `prediction` without missing values.
- `unlabeled::AbstractDataFrame`: the unlabelled sample; it must contain
  `prediction`.
- `outcome::Symbol`: the true (gold-standard) outcome column.
- `prediction::Symbol`: the prediction column, computed by the same fixed
  predictor in both samples.
- `y`, `yhat`, `yhat_unlabeled`: the vector form of the same inputs (outcome and
  prediction on the labelled sample, prediction on the unlabelled sample).

# Keywords
- `lambda = :optimal`: the power-tuning parameter. `:optimal` estimates the
  variance-minimizing non-negative value; a fixed non-negative number is used as
  given (`1` reproduces the original prediction-powered estimator, `0` the
  labelled-sample mean).

# Returns
- [`PPIEstimate`](@ref) with a single coefficient named after `outcome` (or
  `"mean"` in the vector form).

# Examples
```julia
using DrSnow, DataFrames, StableRNGs
rng = StableRNG(1)
n, N = 300, 5000
y = 1 .+ randn(rng, n + N)
pred = 0.3 .+ 0.8 .* y .+ 0.5 .* randn(rng, n + N)   # biased, noisy prediction
df = DataFrame(y=y, y_pred=pred)
r = ppi_mean(df[1:n, :], df[(n + 1):end, :], :y, :y_pred)
confint(r)
r.lambda, r.classical_coef
```

# References
- Angelopoulos, A. N., Bates, S., Fannjiang, C., Jordan, M. I., & Zrnic, T.
  (2023). Prediction-powered inference. *Science*, 382(6671), 669–674.
- Angelopoulos, A. N., Duchi, J. C., & Zrnic, T. (2023). PPI++: Efficient
  prediction-powered inference. arXiv:2311.01453.
- Robins, J. M., Rotnitzky, A., & Zhao, L. P. (1994). Estimation of regression
  coefficients when some regressors are not always observed. *Journal of the
  American Statistical Association*, 89(427), 846–866.
- Zrnic, T., & Candès, E. J. (2024). Cross-prediction-powered inference.
  *Proceedings of the National Academy of Sciences*, 121(15), e2322083121.
"""
function ppi_mean(labeled::AbstractDataFrame, unlabeled::AbstractDataFrame,
                  outcome::Symbol, prediction::Symbol; lambda=:optimal)
    X, Y, f, Xu, fu = _ml_ppi_inputs(labeled, unlabeled, outcome, prediction, Symbol[],
                                     true, "ppi_mean")
    return _ml_ppi_fit(:gaussian, X, Y, f, Xu, fu, lambda, false, [string(outcome)],
                       "PPI++ mean")
end

function ppi_mean(y::AbstractVector{<:Real}, yhat::AbstractVector{<:Real},
                  yhat_unlabeled::AbstractVector{<:Real}; lambda=:optimal)
    length(y) == length(yhat) ||
        throw(DimensionMismatch("y and yhat must have the same length"))
    X = ones(length(y), 1)
    Xu = ones(length(yhat_unlabeled), 1)
    return _ml_ppi_fit(:gaussian, X, Float64.(y), Float64.(yhat), Xu,
                       Float64.(yhat_unlabeled), lambda, false, ["mean"], "PPI++ mean")
end

"""
    ppi_ols(labeled, unlabeled, outcome, prediction; covariates=Symbol[],
            intercept=true, lambda=:optimal) -> PPIEstimate

Prediction-powered estimate of population least-squares coefficients when the
outcome is observed only in a small labelled sample and a machine-learning or
LLM prediction of it is available in both the labelled and a large unlabelled
sample, with PPI++ power tuning.

The estimand is the population projection coefficient ``\\theta^\\ast =
\\arg\\min_\\theta E[(Y - X'\\theta)^2]`` of the true outcome on the regressors
``X`` (the covariates and, by default, an intercept). It has a causal
interpretation only through the design, for example when ``X`` contains a
randomized treatment. The regressors must be observed without error in both
samples; only the outcome is replaced by its prediction ``f``. As for
[`ppi_mean`](@ref), the labelled units must be a random sample from the
population of the unlabelled units, the labels must be error-free, and the
predictor must not have been trained on the labelled sample, but the
predictions may be biased in any way, including in a way that depends on ``X``.

For the squared loss ``\\ell``, the PPI++ estimator minimizes the
prediction-powered empirical risk

```math
L_\\lambda(\\theta) = \\frac{1}{n}\\sum_{i=1}^{n} \\ell(X_i, Y_i; \\theta)
    + \\lambda \\Big[ \\frac{1}{N}\\sum_{j=1}^{N} \\ell(\\tilde X_j, \\tilde f_j;
    \\theta) - \\frac{1}{n}\\sum_{i=1}^{n} \\ell(X_i, f_i; \\theta) \\Big],
```

whose expectation equals the population risk for every fixed ``\\lambda``, so
``\\hat\\theta_\\lambda`` is consistent for ``\\theta^\\ast`` (Angelopoulos et al.
2023; Angelopoulos, Duchi & Zrnic 2023). The asymptotic covariance is the
sandwich ``H^{-1}\\{\\operatorname{Cov}(\\nabla\\ell - \\lambda\\nabla\\ell^f)/n +
\\lambda^2 \\operatorname{Cov}(\\nabla\\ell^{\\tilde f})/N\\}H^{-1}``, with the
Hessian ``H`` estimated on the pooled regressors of both samples and the
gradient covariances by their sample analogues. With `lambda = :optimal`,
``\\lambda`` is chosen to minimize the trace of the estimated covariance
(estimated at the ``\\lambda = 1`` solution) and clipped to ``[0, 1]``, so the
estimator is asymptotically at least as efficient as labelled-only least
squares. Inference uses normal critical values and no small-sample
correction.

Compare `coef(r)` and `stderror(r)` with the labelled-only fit stored in
`classical_coef` and `classical_vcov` (HC0 sandwich); the printed output reports
the variance ratio for each coefficient. For a binary outcome and a logistic
model use [`ppi_logistic`](@ref); when the labelled units are a subsample of the
analysis sample selected with known, possibly covariate-dependent,
probabilities, use [`ppi_regression`](@ref) or [`dsl_regression`](@ref), which
also allow fixed effects and ML-measured regressors respectively.

# Arguments
- `labeled::AbstractDataFrame`: labelled sample with `outcome`, `prediction` and
  `covariates`.
- `unlabeled::AbstractDataFrame`: unlabelled sample with `prediction` and
  `covariates`.
- `outcome::Symbol`: the true (gold-standard) outcome.
- `prediction::Symbol`: the prediction of the outcome from a fixed predictor.

# Keywords
- `covariates::Vector{Symbol} = Symbol[]`: regressors, observed in both samples.
- `intercept::Bool = true`: whether to include an intercept.
- `lambda = :optimal`: the power-tuning parameter; `:optimal` estimates it and
  clips it to ``[0, 1]``, a fixed non-negative number is used as given (`1` is
  the original prediction-powered estimator, `0` labelled-only least squares).

# Returns
- [`PPIEstimate`](@ref) with coefficients `"(Intercept)"` and the covariate
  names.

# Examples
```julia
using DrSnow, DataFrames, StableRNGs
rng = StableRNG(2)
m = 6000
treated = Float64.(rand(rng, m) .< 0.5)
age = randn(rng, m)
y = 1 .+ 0.5 .* treated .+ 0.3 .* age .+ randn(rng, m)
y_pred = 0.2 .+ 0.8 .* y .+ 0.2 .* treated .+ 0.5 .* randn(rng, m)
df = DataFrame(; y, y_pred, treated, age)
lab, unlab = df[1:400, :], df[401:end, :]
r = ppi_ols(lab, unlab, :y, :y_pred; covariates=[:treated, :age])
coeftable(r)
```

# References
- Angelopoulos, A. N., Bates, S., Fannjiang, C., Jordan, M. I., & Zrnic, T.
  (2023). Prediction-powered inference. *Science*, 382(6671), 669–674.
- Angelopoulos, A. N., Duchi, J. C., & Zrnic, T. (2023). PPI++: Efficient
  prediction-powered inference. arXiv:2311.01453.
- Wang, S., McCormick, T. H., & Leek, J. T. (2020). Methods for correcting
  inference based on outcomes predicted by machine learning. *Proceedings of
  the National Academy of Sciences*, 117(48), 30266–30275.
"""
function ppi_ols(labeled::AbstractDataFrame, unlabeled::AbstractDataFrame,
                 outcome::Symbol, prediction::Symbol; covariates=Symbol[],
                 intercept::Bool=true, lambda=:optimal)
    covs = Symbol.(collect(covariates))
    X, Y, f, Xu, fu = _ml_ppi_inputs(labeled, unlabeled, outcome, prediction, covs,
                                     intercept, "ppi_ols")
    return _ml_ppi_fit(:gaussian, X, Y, f, Xu, fu, lambda, true,
                       _ml_ppi_names(covs, intercept), "PPI++ linear regression")
end

"""
    ppi_logistic(labeled, unlabeled, outcome, prediction; covariates=Symbol[],
                 intercept=true, lambda=:optimal) -> PPIEstimate

Prediction-powered estimate of population logistic-regression coefficients of a
binary outcome that is observed only in a small labelled sample, using
predicted probabilities or predicted labels available in both samples, with
PPI++ power tuning.

The estimand is the minimizer ``\\theta^\\ast`` of the population logistic
log-loss of the true binary outcome on the regressors ``X``, that is, the
coefficients of the best logistic approximation to ``P(Y = 1 \\mid X)`` (the
true conditional probability when the logistic model is correctly specified).
The assumptions are those of [`ppi_ols`](@ref): labelled units drawn at random
from the population of the unlabelled units, error-free labels, a predictor not
trained on the labelled sample, and regressors observed in both samples. The
predictions must lie in ``[0, 1]``; they may be calibrated probabilities or hard
0/1 classifications and may be biased.

The estimator minimizes the PPI++ objective of [`ppi_ols`](@ref) with the
logistic log-loss, which is convex for ``\\lambda \\in [0, 1]``, by Newton's
method with step halving (Angelopoulos, Duchi & Zrnic 2023). The covariance is
the corresponding sandwich with the logistic Hessian estimated on the pooled
regressors. `lambda = :optimal` minimizes the trace of the estimated covariance
over ``[0, 1]``; fixed values above one are rejected because the objective can
then be non-convex. Inference uses normal critical values; with rare outcomes
or few labels the labelled-only fit (`classical_coef`, HC0 sandwich) may be
unstable, and so is the power-tuning estimate.

# Arguments
- `labeled::AbstractDataFrame`: labelled sample with the 0/1 `outcome`, the
  `prediction` and the `covariates`.
- `unlabeled::AbstractDataFrame`: unlabelled sample with `prediction` and
  `covariates`.
- `outcome::Symbol`: the true binary outcome, coded 0/1.
- `prediction::Symbol`: predicted probabilities or predicted labels in
  ``[0, 1]``.

# Keywords
- `covariates::Vector{Symbol} = Symbol[]`: regressors, observed in both samples.
- `intercept::Bool = true`: whether to include an intercept.
- `lambda = :optimal`: `:optimal` (estimated, clipped to ``[0, 1]``) or a fixed
  number in ``[0, 1]``.

# Returns
- [`PPIEstimate`](@ref) with coefficients on the logit scale.

# Examples
```julia
using DrSnow, DataFrames, StableRNGs
rng = StableRNG(3)
m = 6000
treated = Float64.(rand(rng, m) .< 0.5)
employed = Float64.(rand(rng, m) .< 1 ./ (1 .+ exp.(-(-0.5 .+ 0.8 .* treated))))
flip = rand(rng, m) .< 0.15                       # classifier errs on 15% of units
employed_pred = ifelse.(flip, 1 .- employed, employed)
df = DataFrame(; employed, employed_pred, treated)
r = ppi_logistic(df[1:500, :], df[501:end, :], :employed, :employed_pred;
                 covariates=[:treated])
coeftable(r)
```

# References
- Angelopoulos, A. N., Bates, S., Fannjiang, C., Jordan, M. I., & Zrnic, T.
  (2023). Prediction-powered inference. *Science*, 382(6671), 669–674.
- Angelopoulos, A. N., Duchi, J. C., & Zrnic, T. (2023). PPI++: Efficient
  prediction-powered inference. arXiv:2311.01453.
"""
function ppi_logistic(labeled::AbstractDataFrame, unlabeled::AbstractDataFrame,
                      outcome::Symbol, prediction::Symbol; covariates=Symbol[],
                      intercept::Bool=true, lambda=:optimal)
    covs = Symbol.(collect(covariates))
    X, Y, f, Xu, fu = _ml_ppi_inputs(labeled, unlabeled, outcome, prediction, covs,
                                     intercept, "ppi_logistic")
    all(v -> v == 0 || v == 1, Y) ||
        throw(ArgumentError("ppi_logistic: the outcome must be binary (0/1)"))
    (all(v -> 0 <= v <= 1, f) && all(v -> 0 <= v <= 1, fu)) ||
        throw(ArgumentError("ppi_logistic: predictions must lie in [0, 1]"))
    return _ml_ppi_fit(:binomial, X, Y, f, Xu, fu, lambda, true,
                       _ml_ppi_names(covs, intercept), "PPI++ logistic regression")
end
