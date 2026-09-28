# MLJ integration: `MLJLearner(model)` wraps any MLJ model as a DrSnow NuisanceLearner.
#
# Uses only the MLJModelInterface API (`reformat`, `fit`, `predict`), so any model
# package works. Classification targets need MLJ's full data interface (categorical
# vectors), which is available once MLJ or MLJBase is loaded.

module DrSnowMLJExt

using DrSnow
using DrSnow: MLJLearner, _ml_check_xy, _ml_check_binary
using MLJModelInterface
using Random
using Statistics: mean

const MMI = MLJModelInterface

function _table(X::AbstractMatrix)
    p = size(X, 2)
    names = Tuple(Symbol("x", j) for j in 1:p)
    return NamedTuple{names}(Tuple(Vector{Float64}(X[:, j]) for j in 1:p))
end

# Copy the model and seed its RNG (if it has one) from the task RNG.
function _seeded(model, rng::AbstractRNG)
    m = deepcopy(model)
    if hasproperty(m, :rng)
        seed = rand(rng, UInt64)
        try
            setproperty!(m, :rng, Random.Xoshiro(seed))
        catch
            try
                setproperty!(m, :rng, Int(seed >> 1))
            catch
            end
        end
    end
    return m
end

_is_classifier(model) = MMI.target_scitype(model) <: AbstractVector{<:MMI.Finite}

function _fit_predict(model, X, y, Xnew, weights)
    size(X, 2) >= 1 ||
        throw(ArgumentError("MLJLearner: at least one covariate is required"))
    Xt = _table(X)
    args = if weights === nothing
        (Xt, y)
    else
        MMI.supports_weights(model) ||
            throw(ArgumentError("MLJLearner: $(nameof(typeof(model))) does not " *
                                "support observation weights"))
        (Xt, y, Vector{Float64}(weights))
    end
    fitresult, _, _ = MMI.fit(model, 0, MMI.reformat(model, args...)...)
    return MMI.predict(model, fitresult, MMI.reformat(model, _table(Xnew))...)
end

_point(yhat) = eltype(yhat) <: Real ? Vector{Float64}(yhat) : Float64.(mean.(yhat))

function DrSnow.fitpredict(l::MLJLearner{<:MMI.Supervised}, X::AbstractMatrix,
                           y::AbstractVector, Xnew::AbstractMatrix;
                           rng::AbstractRNG=Random.default_rng(), weights=nothing)
    _ml_check_xy(X, y, Xnew)
    model = _seeded(l.model, rng)
    _is_classifier(model) &&
        throw(ArgumentError("MLJLearner: $(nameof(typeof(model))) is a classifier; " *
                            "use it through fitpredict_proba (e.g. as a propensity " *
                            "learner)"))
    return _point(_fit_predict(model, X, Vector{Float64}(y), Xnew, weights))
end

function DrSnow.fitpredict_proba(l::MLJLearner{<:MMI.Supervised}, X::AbstractMatrix,
                                 y::AbstractVector, Xnew::AbstractMatrix;
                                 rng::AbstractRNG=Random.default_rng(), weights=nothing)
    _ml_check_xy(X, y, Xnew)
    _ml_check_binary(y)
    model = _seeded(l.model, rng)
    if !_is_classifier(model)
        # regressor on the 0/1 target: an estimate of P(y = 1 | x), clipped to [0, 1]
        pr = _point(_fit_predict(model, X, Vector{Float64}(y), Xnew, weights))
        return clamp.(pr, 0.0, 1.0)
    end
    model isa MMI.Probabilistic ||
        throw(ArgumentError("MLJLearner: $(nameof(typeof(model))) is a deterministic " *
                            "classifier; probabilities are required"))
    MMI.get_interface_mode() isa MMI.FullInterface ||
        throw(ArgumentError("MLJLearner: classification requires MLJ's full data " *
                            "interface; load MLJ or MLJBase (`using MLJBase`)"))
    yc = MMI.categorical(Int.(y))
    one = yc[findfirst(==(1), y)]
    yhat = _fit_predict(model, X, yc, Xnew, weights)
    return Float64[DrSnow.Distributions.pdf(d, one) for d in yhat]
end

end # module DrSnowMLJExt
