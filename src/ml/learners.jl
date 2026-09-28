# Nuisance learners: the prediction interface used by every cross-fitted estimator.
#
# A learner is any subtype of `NuisanceLearner` implementing
#     fitpredict(learner, X, y, Xnew; rng, weights)        -> Vector{Float64}  (E[y | x])
# and, when it can estimate class probabilities for a binary 0/1 target,
#     fitpredict_proba(learner, X, y, Xnew; rng, weights)  -> Vector{Float64}  (P(y=1 | x))
# `X` and `Xnew` are `Matrix{Float64}` (possibly with zero columns), `rng` is a
# task-specific RNG created by the caller from a pre-drawn seed, and `weights` is
# `nothing` or a vector of non-negative observation weights.

"""
    NuisanceLearner

Abstract supertype of the prediction methods that estimate nuisance functions
(conditional means and propensity scores) inside every cross-fitted estimator of the
package.

Double/debiased machine learning, the DR-learner, generalized random forests with
local centering and design-based supervised learning all have the same structure: a
low-dimensional target parameter is identified by a moment condition that also depends
on unknown functions of the covariates, such as the outcome regression
``g_0(x) = E[Y \\mid X = x]`` or the propensity score ``m_0(x) = P(D = 1 \\mid X = x)``.
Because the moment conditions are Neyman orthogonal, first-order errors in these
nuisance estimates do not bias the target, and root-``n`` inference remains valid when
the learners converge at rates as slow as ``o(n^{-1/4})`` in mean square and are
evaluated out of fold (Chernozhukov et al., 2018; Chernozhukov et al., 2022). A
`NuisanceLearner` is the object that produces these out-of-fold predictions. The
theory places no restriction on the learner beyond its rate, so the choice among
linear, penalized, nearest-neighbour, forest or MLJ learners is an empirical question
about prediction quality; [`nuisance_loss`](@ref) reports the out-of-fold loss of each
nuisance for comparison.

A learner is a subtype of `NuisanceLearner` implementing

```julia
fitpredict(learner, X, y, Xnew; rng, weights) -> Vector{Float64}
```

which fits the regression of `y` on the columns of `X` and returns estimates of
``E[y \\mid x]`` at the rows of `Xnew`. A learner that is to be used in a propensity
or other classification role (for example `propensity_learner` in [`dml_irm`](@ref) or
[`dml_iivm`](@ref)) must **also** implement

```julia
fitpredict_proba(learner, X, y, Xnew; rng, weights) -> Vector{Float64}
```

returning ``P(y = 1 \\mid x)`` for a binary 0/1 target. There is no automatic fallback
from one method to the other: the generic `fitpredict_proba` method throws an
`ArgumentError`, so a custom learner that defines only `fitpredict` can be used for
conditional means but is rejected in a probability role. `X` and `Xnew` are
`Matrix{Float64}` and may have zero columns; `rng` is a task-specific random-number
generator that the estimator creates from seeds drawn up front, so results do not
depend on the number of threads; `weights` is `nothing` or a vector of non-negative
observation weights, which a learner should honour or reject with an error.

Built-in learners: [`OLSLearner`](@ref), [`RidgeLearner`](@ref),
[`LassoLearner`](@ref), [`LogisticLearner`](@ref),
[`PenalizedLogisticLearner`](@ref), [`KNNLearner`](@ref), [`MeanLearner`](@ref) and
the honest regression forest [`ForestLearner`](@ref). Any MLJ model can be used through
[`MLJLearner`](@ref).

# Examples
```julia
using DrSnow, StableRNGs, Statistics, Random

# A custom learner: the conditional mean within the sign of the first covariate.
struct SignMeanLearner <: NuisanceLearner end

function DrSnow.fitpredict(::SignMeanLearner, X, y, Xnew;
                           rng=Random.default_rng(), weights=nothing)
    pos = X[:, 1] .> 0
    mp, mn = mean(y[pos]), mean(y[.!pos])
    return [x > 0 ? mp : mn for x in Xnew[:, 1]]
end

# Needed for propensity roles: probabilities of a binary target.
DrSnow.fitpredict_proba(l::SignMeanLearner, X, y, Xnew; kwargs...) =
    clamp.(fitpredict(l, X, y, Xnew; kwargs...), 0.0, 1.0)

rng = StableRNG(1)
X = randn(rng, 200, 2)
d = Float64.(rand(rng, 200) .< ifelse.(X[:, 1] .> 0, 0.7, 0.3))
fitpredict_proba(SignMeanLearner(), X, d, X[1:3, :])
```

# References
- Chernozhukov, V., Chetverikov, D., Demirer, M., Duflo, E., Hansen, C., Newey, W.,
  & Robins, J. (2018). Double/debiased machine learning for treatment and structural
  parameters. *The Econometrics Journal*, 21(1), C1–C68.
- Chernozhukov, V., Escanciano, J. C., Ichimura, H., Newey, W. K., & Robins, J. M.
  (2022). Locally robust semiparametric estimation. *Econometrica*, 90(4),
  1501–1535.
- Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical
  Learning: Data Mining, Inference, and Prediction* (2nd ed.). Springer.
- Bach, P., Kurz, M. S., Chernozhukov, V., Spindler, M., & Klaassen, S. (2024).
  DoubleML: An object-oriented implementation of double machine learning in R.
  *Journal of Statistical Software*, 108(3), 1–56.
"""
abstract type NuisanceLearner end

"""
    fitpredict(learner, X, y, Xnew; rng=Random.default_rng(), weights=nothing)
        -> Vector{Float64}

Fit `learner` to the regression of `y` on the columns of `X` and return estimates of
the conditional mean ``E[y \\mid x]`` at the rows of `Xnew`.

This is the single method through which every cross-fitted estimator of the package
obtains its conditional-mean nuisances: the estimator calls it once per fold and
nuisance with the training folds as `(X, y)` and the held-out fold as `Xnew`, so the
predictions it returns are out of fold (Chernozhukov et al., 2018). Calling it
directly is useful for checking a learner or a custom implementation outside an
estimator. The generic method for an arbitrary [`NuisanceLearner`](@ref) throws an
`ArgumentError`; every built-in learner has its own method, and a custom learner must
define one.

# Arguments
- `learner::NuisanceLearner`: the prediction method, with its tuning parameters.
- `X::AbstractMatrix`: training covariates, one row per observation; it may have zero
  columns, in which case learners return the (weighted) training mean.
- `y::AbstractVector`: training target, of length `size(X, 1)`.
- `Xnew::AbstractMatrix`: prediction points, with the same columns as `X`.

# Keywords
- `rng::AbstractRNG = Random.default_rng()`: random-number generator for learners
  with internal randomness (cross-validation folds, bootstrap subsamples); inside
  estimators it is a task-specific generator seeded up front.
- `weights = nothing`: `nothing` for equal weights, or a vector of non-negative
  observation weights of length `size(X, 1)`.

# Returns
- `Vector{Float64}` of length `size(Xnew, 1)` with the predicted conditional means.

# Examples
```julia
using DrSnow, StableRNGs
rng = StableRNG(1)
X = randn(rng, 100, 3)
y = X * [1.0, 0.5, 0.0] .+ randn(rng, 100)
fitpredict(OLSLearner(), X, y, X[1:5, :])
```

# References
- Chernozhukov, V., Chetverikov, D., Demirer, M., Duflo, E., Hansen, C., Newey, W.,
  & Robins, J. (2018). Double/debiased machine learning for treatment and structural
  parameters. *The Econometrics Journal*, 21(1), C1–C68.
"""
function fitpredict(l::NuisanceLearner, X, y, Xnew; kwargs...)
    throw(ArgumentError("fitpredict is not implemented for $(typeof(l))"))
end

"""
    fitpredict_proba(learner, X, y, Xnew; rng=Random.default_rng(), weights=nothing)
        -> Vector{Float64}

Fit `learner` to a binary 0/1 target `y` and return estimated probabilities
``P(y = 1 \\mid x)`` at the rows of `Xnew`.

Estimators call this method for every nuisance that is a probability: the propensity
score ``P(D = 1 \\mid X)`` in [`dml_irm`](@ref) and [`dml_did`](@ref), the instrument
propensity and compliance probabilities in [`dml_iivm`](@ref), and similar roles
elsewhere. Inverse-probability weights divide by these predictions, so calibrated
probabilities matter more here than for conditional means; the estimators clip them
to `[trim, 1 - trim]`, which keeps the weights finite but does not remedy limited
overlap (Crump et al., 2009). Probability-capable built-in learners are
[`LogisticLearner`](@ref), [`PenalizedLogisticLearner`](@ref), [`KNNLearner`](@ref),
[`MeanLearner`](@ref), [`ForestLearner`](@ref) and suitable [`MLJLearner`](@ref)s.
The generic method throws an `ArgumentError`, so a learner without its own
`fitpredict_proba` (for example [`OLSLearner`](@ref) or [`LassoLearner`](@ref), or a
custom learner that defines only [`fitpredict`](@ref)) is rejected in a probability
role rather than silently producing values outside `[0, 1]`.

# Arguments
- `learner::NuisanceLearner`: a learner with a `fitpredict_proba` method.
- `X::AbstractMatrix`: training covariates (may have zero columns).
- `y::AbstractVector`: binary 0/1 training target; other values throw an
  `ArgumentError`.
- `Xnew::AbstractMatrix`: prediction points, with the same columns as `X`.

# Keywords
- `rng::AbstractRNG = Random.default_rng()`: generator for learners with internal
  randomness.
- `weights = nothing`: `nothing` or non-negative observation weights.

# Returns
- `Vector{Float64}` of length `size(Xnew, 1)` with probabilities in `[0, 1]`.

# Examples
```julia
using DrSnow, StableRNGs
rng = StableRNG(2)
X = randn(rng, 200, 2)
d = Float64.(rand(rng, 200) .< 1 ./ (1 .+ exp.(-X[:, 1])))
fitpredict_proba(LogisticLearner(), X, d, X[1:5, :])
```

# References
- Crump, R. K., Hotz, V. J., Imbens, G. W., & Mitnik, O. A. (2009). Dealing with
  limited overlap in estimation of average treatment effects. *Biometrika*, 96(1),
  187–199.
- Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical
  Learning: Data Mining, Inference, and Prediction* (2nd ed.). Springer.
"""
function fitpredict_proba(l::NuisanceLearner, X, y, Xnew; kwargs...)
    throw(ArgumentError("$(typeof(l)) does not estimate class probabilities; use a " *
                        "classifier such as LogisticLearner(), " *
                        "PenalizedLogisticLearner(), KNNLearner() or an MLJ " *
                        "probabilistic classifier via MLJLearner"))
end

"""Short human-readable learner description used in printed results."""
_ml_learner_name(l) = string(nameof(typeof(l)))

function _ml_check_xy(X, y, Xnew)
    size(X, 1) == length(y) ||
        throw(DimensionMismatch("X has $(size(X, 1)) rows but y has $(length(y))"))
    size(X, 2) == size(Xnew, 2) ||
        throw(DimensionMismatch("X and Xnew must have the same number of columns"))
    size(X, 1) > 0 || throw(ArgumentError("cannot fit a learner on zero observations"))
    return nothing
end

function _ml_check_binary(y)
    all(v -> v == 0 || v == 1, y) ||
        throw(ArgumentError("fitpredict_proba requires a binary 0/1 target"))
    return nothing
end

# Linear predictor fitted object shared by the linear learners.
struct _MLLinearFit
    b0::Float64
    β::Vector{Float64}
    logistic::Bool
end

function _ml_predict(f::_MLLinearFit, Xnew::AbstractMatrix)
    η = Xnew * f.β .+ f.b0
    return f.logistic ? _ml_sigmoid.(η) : η
end

# ---------------------------------------------------------------------------- OLS

"""
    OLSLearner(; intercept=true)

Ordinary (optionally weighted) least-squares regression as a
[`NuisanceLearner`](@ref) for conditional means.

The learner estimates ``E[y \\mid x]`` by the linear projection
``\\hat\\beta_0 + x'\\hat\\beta``, minimizing ``\\sum_i w_i (y_i - \\beta_0 -
x_i'\\beta)^2``. Rank-deficient designs are handled by a column-pivoted QR
decomposition, so collinear covariates do not cause an error. The fit is
deterministic and ignores `rng`. With a fixed low-dimensional covariate set, DML with
`OLSLearner` nuisances is Robinson's (1988) partialling-out estimator with sample
splitting; given the same sample splits it reproduces DoubleML with the `mlr3` learner
`regr.lm`, which the package uses as a validation benchmark (Bach et al., 2024). OLS
is consistent for the nuisance only when the conditional mean is linear in the
supplied covariates, so rich specifications (interactions, splines) or a flexible
learner are needed when that is doubtful. It has no `fitpredict_proba` method: for
propensity scores use [`LogisticLearner`](@ref) or
[`PenalizedLogisticLearner`](@ref).

# Keywords
- `intercept::Bool = true`: include an unpenalized intercept; set to `false` only if
  the covariates already contain a constant or the target is centred.

# Fields
- `intercept::Bool`: whether an intercept is fitted.

# Examples
```julia
using DrSnow, DataFrames, StableRNGs
rng = StableRNG(3)
n = 500
df = DataFrame(x1=randn(rng, n), x2=randn(rng, n))
df.d = 0.5 .* df.x1 .+ randn(rng, n)
df.y = 1.0 .* df.d .+ df.x1 .- 0.5 .* df.x2 .+ randn(rng, n)
dml_plr(df, :y, :d; covariates=[:x1, :x2], outcome_learner=OLSLearner(),
        treatment_learner=OLSLearner(), rng=StableRNG(4))
```

# References
- Robinson, P. M. (1988). Root-N-consistent semiparametric regression.
  *Econometrica*, 56(4), 931–954.
- Bach, P., Kurz, M. S., Chernozhukov, V., Spindler, M., & Klaassen, S. (2024).
  DoubleML: An object-oriented implementation of double machine learning in R.
  *Journal of Statistical Software*, 108(3), 1–56.
"""
Base.@kwdef struct OLSLearner <: NuisanceLearner
    intercept::Bool = true
end

function _ml_fit(l::OLSLearner, X::AbstractMatrix, y::AbstractVector; weights=nothing,
                 rng=Random.default_rng())
    n, p = size(X)
    Z = l.intercept ? hcat(ones(n), X) : Matrix{Float64}(X)
    size(Z, 2) == 0 && return _MLLinearFit(0.0, zeros(p), false)
    yv = Vector{Float64}(y)
    if weights !== nothing
        sw = sqrt.(_ml_normweights(weights, n))
        Z = Z .* sw
        yv = yv .* sw
    end
    b = qr(Z, ColumnNorm()) \ yv
    return l.intercept ? _MLLinearFit(b[1], b[2:end], false) : _MLLinearFit(0.0, b, false)
end

function fitpredict(l::OLSLearner, X::AbstractMatrix, y::AbstractVector,
                    Xnew::AbstractMatrix; rng=Random.default_rng(), weights=nothing)
    _ml_check_xy(X, y, Xnew)
    return _ml_predict(_ml_fit(l, X, y; weights=weights), Xnew)
end

# -------------------------------------------------------------------------- Ridge

"""
    RidgeLearner(; lambda=:loocv, lambdas=exp10.(range(-4, 2; length=40)))

Ridge regression with a data-driven penalty as a [`NuisanceLearner`](@ref) for
conditional means.

Ridge regression (Hoerl & Kennard, 1970) shrinks the coefficients of a linear
predictor towards zero with a squared ``\\ell_2`` penalty. The learner standardizes the
covariates with weighted moments (``\\tilde x``), leaves the intercept unpenalized, and
minimizes

```math
\\frac{\\sum_i w_i (y_i - \\beta_0 - \\tilde x_i'\\beta)^2}{\\sum_i w_i}
+ \\lambda \\lVert \\beta \\rVert_2^2 ,
```

returning coefficients on the original scale; constant covariates are dropped. With
`lambda = :loocv` the penalty is chosen from the grid `lambdas` by exact weighted
leave-one-out cross-validation, computed in closed form from the singular value
decomposition and the hat-matrix leverages (Hastie et al., 2009, Sections 3.4 and
7.10), so the choice is deterministic and does not use `rng`. Ridge keeps every
covariate, which suits dense signals spread over many correlated covariates (for
example, many dummies or basis expansions), whereas [`LassoLearner`](@ref) suits
approximately sparse signals. It estimates conditional means only; it has no
`fitpredict_proba` method.

# Keywords
- `lambda::Union{Real,Symbol} = :loocv`: `:loocv` selects the penalty by
  leave-one-out cross-validation over `lambdas`; a non-negative number fixes it
  (`0` gives least squares on the standardized covariates).
- `lambdas::Vector{Float64} = exp10.(range(-4, 2; length=40))`: candidate penalties
  for `:loocv`, on the scale of the objective above.

# Fields
- `lambda::Union{Real,Symbol}`: fixed penalty or `:loocv`.
- `lambdas::Vector{Float64}`: candidate grid for leave-one-out selection.

# Examples
```julia
using DrSnow, StableRNGs
rng = StableRNG(5)
X = randn(rng, 200, 20)
y = X * fill(0.2, 20) .+ randn(rng, 200)
fitpredict(RidgeLearner(), X, y, X[1:5, :])
fitpredict(RidgeLearner(lambda=0.1), X, y, X[1:5, :])
```

# References
- Hoerl, A. E., & Kennard, R. W. (1970). Ridge regression: Biased estimation for
  nonorthogonal problems. *Technometrics*, 12(1), 55–67.
- Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical
  Learning: Data Mining, Inference, and Prediction* (2nd ed.). Springer.
"""
Base.@kwdef struct RidgeLearner <: NuisanceLearner
    lambda::Union{Real,Symbol} = :loocv
    lambdas::Vector{Float64} = exp10.(range(-4, 2; length=40))
end

function _ml_fit(l::RidgeLearner, X::AbstractMatrix, y::AbstractVector; weights=nothing,
                 rng=Random.default_rng())
    n, p = size(X)
    w = _ml_normweights(weights, n)
    yv = Vector{Float64}(y)
    ybar = dot(w, yv)
    μ, σ, ok = _ml_moments(X, w)
    if p == 0 || !any(ok)
        return _MLLinearFit(ybar, zeros(p), false)
    end
    Z = _ml_standardized(X, μ, σ, ok)[:, ok]
    sw = sqrt.(w)
    A = Z .* sw
    F = svd(A)
    uy = F.U' * ((yv .- ybar) .* sw)
    d2 = F.S .^ 2
    λ = if l.lambda isa Real
        l.lambda >= 0 || throw(ArgumentError("lambda must be non-negative"))
        Float64(l.lambda)
    elseif l.lambda === :loocv
        best, bestλ = Inf, first(l.lambdas)
        U2 = F.U .^ 2
        for λc in l.lambdas
            shrink = d2 ./ (d2 .+ λc)
            fit_s = F.U * (shrink .* uy)            # √w ⊙ fitted (centred)
            h = U2 * shrink .+ w                    # leverages incl. intercept
            cv = 0.0
            for i in 1:n
                w[i] > 0 || continue
                e = ((yv[i] - ybar) - fit_s[i] / sw[i]) / (1 - min(h[i], 1 - 1e-12))
                cv += w[i] * e^2
            end
            if cv < best
                best, bestλ = cv, λc
            end
        end
        bestλ
    else
        throw(ArgumentError("lambda must be a non-negative number or :loocv"))
    end
    βs_ok = F.V * ((F.S ./ (d2 .+ λ)) .* uy)
    βs = zeros(p)
    βs[ok] .= βs_ok
    b0, β = _ml_unstandardize(ybar, βs, μ, σ, ok)
    return _MLLinearFit(b0, β, false)
end

function fitpredict(l::RidgeLearner, X::AbstractMatrix, y::AbstractVector,
                    Xnew::AbstractMatrix; rng=Random.default_rng(), weights=nothing)
    _ml_check_xy(X, y, Xnew)
    return _ml_predict(_ml_fit(l, X, y; weights=weights), Xnew)
end

# -------------------------------------------------------------------------- Lasso

"""
    LassoLearner(; lambda=:cv, alpha=1.0, nfolds=5, nlambda=50,
                 lambda_min_ratio=nothing, rule=:min)

Lasso or elastic-net regression with a cross-validated penalty as a
[`NuisanceLearner`](@ref) for conditional means; the default learner of the DML
estimators.

The lasso (Tibshirani, 1996) adds an ``\\ell_1`` penalty that sets many coefficients
exactly to zero, and the elastic net mixes it with a ridge penalty. The learner uses
glmnet's parameterization (Friedman et al., 2010): with observation weights
normalized to sum to one and covariates standardized with weighted moments
(``\\tilde x``), it minimizes

```math
\\tfrac12 \\sum_i w_i (y_i - \\beta_0 - \\tilde x_i'\\beta)^2
+ \\lambda \\bigl[\\alpha \\lVert \\beta \\rVert_1
+ \\tfrac{1-\\alpha}{2} \\lVert \\beta \\rVert_2^2\\bigr]
```

by cyclic coordinate descent with warm starts along a path, with an unpenalized
intercept and coefficients returned on the original scale. With `lambda = :cv` the
path has `nlambda` log-spaced values from the smallest penalty that sets every
coefficient to zero down to `lambda_min_ratio` times that value, and the penalty is
chosen by `nfolds`-fold cross-validation of the weighted mean squared error, with
folds drawn from `rng`. `rule = :min` takes the minimizer of the cross-validated
error and `rule = :one_se` the largest penalty within one standard error of the
minimum, a sparser and more conservative choice (Hastie et al., 2009, Section 7.10).

In DML the lasso is the natural learner under approximate sparsity, where the
conditional mean is well approximated by a few of many (possibly constructed)
covariates; Belloni et al. (2014) show that post-selection inference on a treatment
effect is valid when the lasso is applied to both the outcome and the treatment
equations, which is what [`dml_plr`](@ref) does with its default learners.
Cross-validated penalties are a practical choice rather than the theoretically
motivated plug-in penalty of Belloni et al., and the ``o(n^{-1/4})`` rate the DML
theory requires is an assumption about the sparsity of the true function. It
estimates conditional means only; for probabilities use
[`PenalizedLogisticLearner`](@ref).

# Keywords
- `lambda::Union{Real,Symbol} = :cv`: `:cv` for cross-validated selection, or a
  fixed non-negative penalty on the scale of the objective above.
- `alpha::Float64 = 1.0`: elastic-net mixing parameter in `[0, 1]`; `1` is the
  lasso, `0` ridge, values in between the elastic net.
- `nfolds::Int = 5`: number of cross-validation folds (capped at the sample size).
- `nlambda::Int = 50`: length of the penalty path.
- `lambda_min_ratio::Union{Nothing,Float64} = nothing`: ratio of the smallest to the
  largest penalty on the path; `nothing` uses `1e-4` when there are more observations
  than covariates and `1e-2` otherwise, as glmnet does.
- `rule::Symbol = :min`: `:min` or `:one_se` selection rule.

# Fields
- `lambda::Union{Real,Symbol}`, `alpha::Float64`, `nfolds::Int`, `nlambda::Int`,
  `lambda_min_ratio::Union{Nothing,Float64}`, `rule::Symbol`: the keywords above.

# Examples
```julia
using DrSnow, StableRNGs
rng = StableRNG(6)
X = randn(rng, 200, 30)
y = 2 .* X[:, 1] .- X[:, 2] .+ randn(rng, 200)
fitpredict(LassoLearner(), X, y, X[1:5, :]; rng=StableRNG(1))
fitpredict(LassoLearner(lambda=0.05, alpha=0.5), X, y, X[1:5, :])
```

# References
- Tibshirani, R. (1996). Regression shrinkage and selection via the lasso. *Journal
  of the Royal Statistical Society: Series B (Methodological)*, 58(1), 267–288.
- Friedman, J., Hastie, T., & Tibshirani, R. (2010). Regularization paths for
  generalized linear models via coordinate descent. *Journal of Statistical
  Software*, 33(1), 1–22.
- Belloni, A., Chernozhukov, V., & Hansen, C. (2014). Inference on treatment effects
  after selection among high-dimensional controls. *The Review of Economic Studies*,
  81(2), 608–650.
- Chernozhukov, V., Chetverikov, D., Demirer, M., Duflo, E., Hansen, C., Newey, W.,
  & Robins, J. (2018). Double/debiased machine learning for treatment and structural
  parameters. *The Econometrics Journal*, 21(1), C1–C68.
- Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical
  Learning: Data Mining, Inference, and Prediction* (2nd ed.). Springer.
"""
Base.@kwdef struct LassoLearner <: NuisanceLearner
    lambda::Union{Real,Symbol} = :cv
    alpha::Float64 = 1.0
    nfolds::Int = 5
    nlambda::Int = 50
    lambda_min_ratio::Union{Nothing,Float64} = nothing
    rule::Symbol = :min
end

function _ml_fit(l::LassoLearner, X::AbstractMatrix, y::AbstractVector; weights=nothing,
                 rng=Random.default_rng())
    0 <= l.alpha <= 1 || throw(ArgumentError("alpha must be in [0, 1]"))
    w = _ml_normweights(weights, size(X, 1))
    b0, β, _ = _ml_enet_fit(X, Vector{Float64}(y), w; family=:gaussian, alpha=l.alpha,
                            lambda=l.lambda, nlambda=l.nlambda,
                            lambda_min_ratio=l.lambda_min_ratio, nfolds=l.nfolds,
                            rule=l.rule, rng=rng)
    return _MLLinearFit(b0, β, false)
end

function fitpredict(l::LassoLearner, X::AbstractMatrix, y::AbstractVector,
                    Xnew::AbstractMatrix; rng=Random.default_rng(), weights=nothing)
    _ml_check_xy(X, y, Xnew)
    return _ml_predict(_ml_fit(l, X, y; weights=weights, rng=rng), Xnew)
end

# ----------------------------------------------------------------------- Logistic

"""
    LogisticLearner(; intercept=true)

Unpenalized logistic regression as a [`NuisanceLearner`](@ref) for propensity scores
and other binary-target probabilities.

The learner models ``P(y = 1 \\mid x) = \\Lambda(\\beta_0 + x'\\beta)`` with the
logistic function ``\\Lambda`` and estimates the coefficients by maximum likelihood:
unweighted fits use GLM.jl with a tight convergence tolerance, weighted fits a
Newton–Raphson iteration with step halving. Both [`fitpredict`](@ref) and
[`fitpredict_proba`](@ref) return the fitted probabilities. The fit is deterministic
and ignores `rng`; given the same sample splits, DML with this propensity learner
reproduces DoubleML with the `mlr3` learner `classif.log_reg` (Bach et al., 2024).
The logistic model is the classical parametric propensity score. Under (quasi-)
separation the maximum-likelihood estimates diverge and fitted probabilities approach
0 or 1; non-finite coefficients are set to zero, and the estimators' propensity
clipping keeps weights finite, but separation signals limited overlap that clipping
does not repair (Crump et al., 2009). With many covariates prefer
[`PenalizedLogisticLearner`](@ref). A training target with a single class throws an
`ArgumentError`, since probabilities are then not identified.

# Keywords
- `intercept::Bool = true`: include an intercept.

# Fields
- `intercept::Bool`: whether an intercept is fitted.

# Examples
```julia
using DrSnow, StableRNGs
rng = StableRNG(7)
X = randn(rng, 300, 2)
d = Float64.(rand(rng, 300) .< 1 ./ (1 .+ exp.(-(0.5 .* X[:, 1] .- X[:, 2]))))
fitpredict_proba(LogisticLearner(), X, d, X[1:5, :])
```

# References
- Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical
  Learning: Data Mining, Inference, and Prediction* (2nd ed.). Springer.
- Crump, R. K., Hotz, V. J., Imbens, G. W., & Mitnik, O. A. (2009). Dealing with
  limited overlap in estimation of average treatment effects. *Biometrika*, 96(1),
  187–199.
- Bach, P., Kurz, M. S., Chernozhukov, V., Spindler, M., & Klaassen, S. (2024).
  DoubleML: An object-oriented implementation of double machine learning in R.
  *Journal of Statistical Software*, 108(3), 1–56.
"""
Base.@kwdef struct LogisticLearner <: NuisanceLearner
    intercept::Bool = true
end

function _ml_fit(l::LogisticLearner, X::AbstractMatrix, y::AbstractVector;
                 weights=nothing, rng=Random.default_rng())
    _ml_check_binary(y)
    n, p = size(X)
    Z = l.intercept ? hcat(ones(n), X) : Matrix{Float64}(X)
    size(Z, 2) == 0 && return _MLLinearFit(0.0, zeros(p), true)
    yv = Vector{Float64}(y)
    if all(==(yv[1]), yv)
        throw(ArgumentError("LogisticLearner: the training target has a single class " *
                            "($(yv[1])); probabilities are not identified"))
    end
    b = if weights === nothing
        m = GLM.glm(Z, yv, Binomial(), LogitLink(); atol=1e-12, rtol=1e-12, maxiter=100)
        map(x -> isfinite(x) ? x : 0.0, GLM.coef(m))
    else
        _ml_logistic_newton(Z, yv, _ml_normweights(weights, n))
    end
    return l.intercept ? _MLLinearFit(b[1], b[2:end], true) : _MLLinearFit(0.0, b, true)
end

# Weighted logistic MLE by Newton–Raphson with step halving (used for weighted fits;
# unweighted fits are delegated to GLM.jl).
function _ml_logistic_nll(Z, y, w, b)
    η = Z * b
    s = 0.0
    for i in eachindex(y)
        s -= w[i] * (y[i] * η[i] - (η[i] > 0 ? η[i] + log1p(exp(-η[i])) : log1p(exp(η[i]))))
    end
    return s
end

function _ml_logistic_newton(Z, y, w; maxit::Int=100, tol::Float64=1e-13)
    b = zeros(size(Z, 2))
    f = _ml_logistic_nll(Z, y, w, b)
    for _ in 1:maxit
        pr = _ml_sigmoid.(Z * b)
        g = Z' * (w .* (y .- pr))
        H = Z' * (Z .* (w .* pr .* (1 .- pr)))
        step = (Symmetric(H) + 1e-12 * I) \ g
        t = 1.0
        fnew = _ml_logistic_nll(Z, y, w, b .+ step)
        while fnew > f && t > 1e-8
            t /= 2
            fnew = _ml_logistic_nll(Z, y, w, b .+ t .* step)
        end
        b .+= t .* step
        done = abs(f - fnew) <= tol * (abs(fnew) + tol)
        f = fnew
        done && break
    end
    return b
end

function fitpredict(l::LogisticLearner, X::AbstractMatrix, y::AbstractVector,
                    Xnew::AbstractMatrix; rng=Random.default_rng(), weights=nothing)
    _ml_check_xy(X, y, Xnew)
    return _ml_predict(_ml_fit(l, X, y; weights=weights), Xnew)
end

fitpredict_proba(l::LogisticLearner, X::AbstractMatrix, y::AbstractVector,
                 Xnew::AbstractMatrix; rng=Random.default_rng(), weights=nothing) =
    fitpredict(l, X, y, Xnew; rng=rng, weights=weights)

# ---------------------------------------------------------- Penalized logistic

"""
    PenalizedLogisticLearner(; lambda=:cv, alpha=1.0, nfolds=5, nlambda=50,
                             lambda_min_ratio=nothing, rule=:min)

Lasso, elastic-net or ridge penalized logistic regression with a cross-validated
penalty as a [`NuisanceLearner`](@ref) for probabilities; the default propensity
learner of [`dml_irm`](@ref) and [`dml_iivm`](@ref).

The learner fits ``P(y = 1 \\mid x) = \\Lambda(\\beta_0 + \\tilde x'\\beta)`` on
covariates standardized with weighted moments, minimizing the weighted negative
log-likelihood plus the elastic-net penalty
``\\lambda[\\alpha \\lVert\\beta\\rVert_1
+ \\tfrac{1-\\alpha}{2}\\lVert\\beta\\rVert_2^2]`` with
glmnet's parameterization (Friedman et al., 2010): observation weights normalized to
sum to one, an unpenalized intercept, and iteratively reweighted least squares with
an inner coordinate-descent solver. `alpha = 1` is the ``\\ell_1``-penalized (lasso)
logistic regression (Tibshirani, 1996), `alpha = 0` ridge logistic regression. With
`lambda = :cv` the penalty is chosen on a log-spaced path of `nlambda` values by
`nfolds`-fold cross-validation of the binomial deviance, with folds drawn from `rng`
and stratified by class; `rule` chooses between the minimizing penalty and the
one-standard-error rule, as in [`LassoLearner`](@ref).

Penalization stabilizes propensity estimates with many covariates and under
near-separation, where [`LogisticLearner`](@ref) breaks down, but shrinkage pulls
fitted probabilities towards the marginal treatment share, and the resulting
inverse-probability weights should still be inspected; clipping in the estimators
keeps them finite without addressing limited overlap (Crump et al., 2009). Both
[`fitpredict`](@ref) and [`fitpredict_proba`](@ref) return probabilities. A training
target with a single class throws an `ArgumentError`.

# Keywords
- `lambda::Union{Real,Symbol} = :cv`: `:cv` for cross-validated selection, or a
  fixed non-negative penalty.
- `alpha::Float64 = 1.0`: elastic-net mixing parameter in `[0, 1]`.
- `nfolds::Int = 5`: number of cross-validation folds.
- `nlambda::Int = 50`: length of the penalty path.
- `lambda_min_ratio::Union{Nothing,Float64} = nothing`: smallest-to-largest penalty
  ratio; `nothing` uses `1e-4` when there are more observations than covariates and
  `1e-2` otherwise.
- `rule::Symbol = :min`: `:min` or `:one_se` selection rule.

# Fields
- `lambda::Union{Real,Symbol}`, `alpha::Float64`, `nfolds::Int`, `nlambda::Int`,
  `lambda_min_ratio::Union{Nothing,Float64}`, `rule::Symbol`: the keywords above.

# Examples
```julia
using DrSnow, StableRNGs
rng = StableRNG(8)
X = randn(rng, 300, 20)
d = Float64.(rand(rng, 300) .< 1 ./ (1 .+ exp.(-X[:, 1])))
fitpredict_proba(PenalizedLogisticLearner(), X, d, X[1:5, :]; rng=StableRNG(1))
```

# References
- Friedman, J., Hastie, T., & Tibshirani, R. (2010). Regularization paths for
  generalized linear models via coordinate descent. *Journal of Statistical
  Software*, 33(1), 1–22.
- Tibshirani, R. (1996). Regression shrinkage and selection via the lasso. *Journal
  of the Royal Statistical Society: Series B (Methodological)*, 58(1), 267–288.
- Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical
  Learning: Data Mining, Inference, and Prediction* (2nd ed.). Springer.
- Crump, R. K., Hotz, V. J., Imbens, G. W., & Mitnik, O. A. (2009). Dealing with
  limited overlap in estimation of average treatment effects. *Biometrika*, 96(1),
  187–199.
"""
Base.@kwdef struct PenalizedLogisticLearner <: NuisanceLearner
    lambda::Union{Real,Symbol} = :cv
    alpha::Float64 = 1.0
    nfolds::Int = 5
    nlambda::Int = 50
    lambda_min_ratio::Union{Nothing,Float64} = nothing
    rule::Symbol = :min
end

function _ml_fit(l::PenalizedLogisticLearner, X::AbstractMatrix, y::AbstractVector;
                 weights=nothing, rng=Random.default_rng())
    _ml_check_binary(y)
    0 <= l.alpha <= 1 || throw(ArgumentError("alpha must be in [0, 1]"))
    yv = Vector{Float64}(y)
    all(==(yv[1]), yv) &&
        throw(ArgumentError("PenalizedLogisticLearner: the training target has a " *
                            "single class; probabilities are not identified"))
    w = _ml_normweights(weights, size(X, 1))
    b0, β, _ = _ml_enet_fit(X, yv, w; family=:binomial, alpha=l.alpha, lambda=l.lambda,
                            nlambda=l.nlambda, lambda_min_ratio=l.lambda_min_ratio,
                            nfolds=l.nfolds, rule=l.rule, rng=rng)
    return _MLLinearFit(b0, β, true)
end

function fitpredict(l::PenalizedLogisticLearner, X::AbstractMatrix, y::AbstractVector,
                    Xnew::AbstractMatrix; rng=Random.default_rng(), weights=nothing)
    _ml_check_xy(X, y, Xnew)
    return _ml_predict(_ml_fit(l, X, y; weights=weights, rng=rng), Xnew)
end

fitpredict_proba(l::PenalizedLogisticLearner, X::AbstractMatrix, y::AbstractVector,
                 Xnew::AbstractMatrix; rng=Random.default_rng(), weights=nothing) =
    fitpredict(l, X, y, Xnew; rng=rng, weights=weights)

# ---------------------------------------------------------------------------- kNN

"""
    KNNLearner(; k=10, standardize=true)

``k``-nearest-neighbour regression and classification as a [`NuisanceLearner`](@ref).

The learner estimates ``E[y \\mid x]`` (and, for a binary target, ``P(y = 1 \\mid x)``)
by the average of the targets of the `k` training observations closest to ``x`` in
Euclidean distance (Cover & Hart, 1967; Hastie et al., 2009, Section 2.3). With
`standardize = true` each covariate is centred and scaled by its training-sample mean
and standard deviation, so the distance does not depend on units, and covariates that
are constant in the training sample are ignored. With observation weights the prediction
is the weighted average over the same `k` neighbours. Ties in distance are broken by
training-row order, and `k` is capped at the training sample size. The fit is
deterministic and ignores `rng`.

Nearest neighbours are a simple, assumption-light nonparametric benchmark, but their
convergence rate deteriorates quickly with the number of covariates, and with few
neighbours the estimated probabilities are coarse (multiples of ``1/k``) and can equal
0 or 1, producing extreme inverse-probability weights; prefer smoother learners such as
[`PenalizedLogisticLearner`](@ref) or [`ForestLearner`](@ref) for propensity scores.

# Keywords
- `k::Int = 10`: number of neighbours; must be at least 1. Larger `k` lowers
  variance and raises bias.
- `standardize::Bool = true`: standardize covariates before computing distances.

# Fields
- `k::Int`: number of neighbours.
- `standardize::Bool`: whether covariates are standardized.

# Examples
```julia
using DrSnow, StableRNGs
rng = StableRNG(9)
X = randn(rng, 300, 2)
y = sin.(2 .* X[:, 1]) .+ 0.5 .* randn(rng, 300)
fitpredict(KNNLearner(k=15), X, y, X[1:5, :])
```

# References
- Cover, T., & Hart, P. (1967). Nearest neighbor pattern classification. *IEEE
  Transactions on Information Theory*, 13(1), 21–27.
- Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical
  Learning: Data Mining, Inference, and Prediction* (2nd ed.). Springer.
"""
Base.@kwdef struct KNNLearner <: NuisanceLearner
    k::Int = 10
    standardize::Bool = true
end

function fitpredict(l::KNNLearner, X::AbstractMatrix, y::AbstractVector,
                    Xnew::AbstractMatrix; rng=Random.default_rng(), weights=nothing)
    _ml_check_xy(X, y, Xnew)
    l.k >= 1 || throw(ArgumentError("k must be at least 1"))
    n, p = size(X)
    w = weights === nothing ? ones(n) : _ml_normweights(weights, n)
    k = min(l.k, n)
    if p == 0
        return fill(dot(w, y) / sum(w), size(Xnew, 1))
    end
    μ, σ, ok = l.standardize ? _ml_moments(X, fill(1 / n, n)) :
               (zeros(p), ones(p), trues(p))
    A = _ml_standardized(X, μ, σ, ok)
    B = _ml_standardized(Xnew, μ, σ, ok)
    out = zeros(size(B, 1))
    d = zeros(n)
    for i in axes(B, 1)
        @inbounds for t in 1:n
            s = 0.0
            for j in 1:p
                s += (A[t, j] - B[i, j])^2
            end
            d[t] = s
        end
        nn = partialsortperm(d, 1:k)
        sw = sum(w[nn])
        out[i] = sw > 0 ? dot(w[nn], y[nn]) / sw : mean(y[nn])
    end
    return out
end

function fitpredict_proba(l::KNNLearner, X::AbstractMatrix, y::AbstractVector,
                          Xnew::AbstractMatrix; rng=Random.default_rng(), weights=nothing)
    _ml_check_binary(y)
    return fitpredict(l, X, y, Xnew; rng=rng, weights=weights)
end

# --------------------------------------------------------------------------- Mean

"""
    MeanLearner()

A covariate-free [`NuisanceLearner`](@ref) that predicts the (weighted) training mean
of the target at every point.

For a conditional mean the learner returns ``\\bar y``, the (weighted) training
average; for a binary target [`fitpredict_proba`](@ref) returns the training share of
ones. As a propensity learner it therefore encodes a constant propensity score, which
is correct in a completely randomized experiment (or a Bernoulli design with a common
assignment probability) and misspecified otherwise: with covariate-dependent
assignment the doubly robust scores are then consistent only if the outcome
regressions are (Robins et al., 1994). As an outcome learner it reduces
augmented-IPW scores to pure inverse-probability weighting. It is useful as the
baseline against which [`nuisance_loss`](@ref) judges other learners, and for design-
based analyses of randomized experiments. It has no fields or tuning parameters and
ignores `rng`.

# Examples
```julia
using DrSnow, StableRNGs
rng = StableRNG(10)
X = randn(rng, 100, 2)
d = Float64.(rand(rng, 100) .< 0.5)
fitpredict_proba(MeanLearner(), X, d, X[1:3, :])
```

# References
- Robins, J. M., Rotnitzky, A., & Zhao, L. P. (1994). Estimation of regression
  coefficients when some regressors are not always observed. *Journal of the American
  Statistical Association*, 89(427), 846–866.
- Imbens, G. W., & Rubin, D. B. (2015). *Causal Inference for Statistics, Social, and
  Biomedical Sciences: An Introduction*. Cambridge University Press.
"""
struct MeanLearner <: NuisanceLearner end

function fitpredict(::MeanLearner, X::AbstractMatrix, y::AbstractVector,
                    Xnew::AbstractMatrix; rng=Random.default_rng(), weights=nothing)
    _ml_check_xy(X, y, Xnew)
    w = _ml_normweights(weights, length(y))
    return fill(dot(w, y), size(Xnew, 1))
end

function fitpredict_proba(l::MeanLearner, X::AbstractMatrix, y::AbstractVector,
                          Xnew::AbstractMatrix; rng=Random.default_rng(), weights=nothing)
    _ml_check_binary(y)
    return fitpredict(l, X, y, Xnew; rng=rng, weights=weights)
end

# ------------------------------------------------------------------------ MLJ stub

"""
    MLJLearner(model)

Wrap any supervised MLJ model as a [`NuisanceLearner`](@ref), giving the cross-fitted
estimators access to the machine-learning methods of the MLJ ecosystem (gradient
boosting, random forests, neural networks, support vector machines, stacked
ensembles, ...).

The DML theory requires only that nuisance estimates converge fast enough, at rate
``o(n^{-1/4})`` in mean square, and are used out of fold (Chernozhukov et al., 2018);
it does not favour a learner family, so flexible learners from MLJ are appropriate
when the conditional means or propensity scores are nonlinear or involve
interactions that linear and penalized learners miss. Their tuning is the user's
responsibility: hyperparameters are taken as given (wrap the model in MLJ's
`TunedModel` for internal cross-validation), and [`nuisance_loss`](@ref) reports the
out-of-fold loss for comparison with the built-in learners.

The method is provided by the `DrSnowMLJExt` package extension, which loads
automatically once `MLJModelInterface` is loaded (for example through `using MLJ`);
without it, calls throw an `ArgumentError` explaining how to load it. The wrapper
behaves as follows.

- In [`fitpredict`](@ref), deterministic regressors return their point predictions and
  probabilistic regressors the mean of their predictive distribution; classifiers are
  rejected.
- In [`fitpredict_proba`](@ref), probabilistic classifiers return ``P(y = 1 \\mid x)``;
  regressors (for example regression trees) are fitted to the 0/1 target and their
  predictions clipped to `[0, 1]`; deterministic classifiers are rejected, because
  hard labels are not probabilities. Classification requires MLJ's full data
  interface (load `MLJ` or `MLJBase`).
- At least one covariate is required, and observation weights are passed on only to
  models that support them; otherwise weights throw an `ArgumentError`.
- A fresh copy of the model is fitted in every cross-fitting task; if the model has an
  `rng` field it is set from the task seed, so results are reproducible and do not
  depend on threading.

# Arguments
- `model`: an MLJ model instance (a subtype of `MLJModelInterface.Supervised`) with
  its hyperparameters set.

# Fields
- `model`: the wrapped MLJ model; it is copied, never mutated, during fitting.

# Examples
```julia
using DrSnow, DataFrames, StableRNGs, MLJDecisionTreeInterface
rng = StableRNG(11)
n = 400
df = DataFrame(x1=randn(rng, n), x2=randn(rng, n))
df.d = sin.(df.x1) .+ randn(rng, n)
df.y = 0.5 .* df.d .+ abs.(df.x2) .+ randn(rng, n)
tree = MLJDecisionTreeInterface.RandomForestRegressor(n_trees=50)
dml_plr(df, :y, :d; covariates=[:x1, :x2], outcome_learner=MLJLearner(tree),
        treatment_learner=MLJLearner(tree), rng=StableRNG(12))
```

# References
- Chernozhukov, V., Chetverikov, D., Demirer, M., Duflo, E., Hansen, C., Newey, W.,
  & Robins, J. (2018). Double/debiased machine learning for treatment and structural
  parameters. *The Econometrics Journal*, 21(1), C1–C68.
- Bach, P., Kurz, M. S., Chernozhukov, V., Spindler, M., & Klaassen, S. (2024).
  DoubleML: An object-oriented implementation of double machine learning in R.
  *Journal of Statistical Software*, 108(3), 1–56.
- Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical
  Learning: Data Mining, Inference, and Prediction* (2nd ed.). Springer.
"""
struct MLJLearner{M} <: NuisanceLearner
    model::M
end

_ml_learner_name(l::MLJLearner) = "MLJLearner(" * string(nameof(typeof(l.model))) * ")"

const _ML_MLJ_HINT = "MLJLearner requires the DrSnowMLJExt extension: load " *
                     "MLJModelInterface (e.g. `using MLJ` or `using MLJModelInterface`) " *
                     "and pass an MLJ model"

fitpredict(l::MLJLearner, X, y, Xnew; kwargs...) = throw(ArgumentError(_ML_MLJ_HINT))
fitpredict_proba(l::MLJLearner, X, y, Xnew; kwargs...) = throw(ArgumentError(_ML_MLJ_HINT))
