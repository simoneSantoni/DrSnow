# Prognostic scores: predicted outcomes from baseline covariates, used to form blocks
# and matched pairs at the design stage and as a covariate at the analysis stage.

"""
    PrognosticScore

Predicted outcomes of the experimental units from baseline covariates, with
out-of-sample measures of predictive accuracy; the result of
[`prognostic_score`](@ref).

A prognostic score (Hansen 2008) is a prediction of the outcome a unit would have
without treatment, ``\\hat\\mu(X) \\approx E[Y(0) \\mid X]``, built only from pre-treatment
covariates. At the design stage it is the one-dimensional variable on which units are
blocked or paired ([`block_design`](@ref)); at the analysis stage it is an adjustment
covariate ([`experiment_estimate`](@ref)); and its out-of-sample R² is the planning
input for power calculations ([`power_means`](@ref), [`power_blocked`](@ref)) and for
[`variance_reduction`](@ref). Because the score is a function of baseline covariates
only, any error in it costs precision but cannot bias a randomized comparison.

# Fields
- `score::Vector{Float64}`: predicted outcome for each experimental unit.
- `ids::Vector`: unit identifiers aligned with `score` (values of the `id` column, or
  row numbers of the experimental data when no `id` was given).
- `id::Union{Nothing,Symbol}`: name of the identifier column.
- `r2::Float64`: cross-fitted out-of-sample R² in the training data,
  ``1 - \\sum (y - \\hat y)^2 / \\sum (y - \\bar y)^2``; negative for a model that
  predicts worse than the mean.
- `rmse::Float64`: cross-fitted root mean squared prediction error.
- `outcome_sd::Float64`: standard deviation of the training outcome.
- `learner::String`: description of the learner.
- `covariates::Vector{Symbol}`: the baseline covariates used.
- `n_train::Int`: number of training observations.
- `n_folds::Int`: number of cross-fitting folds.
- `crossfit::Bool`: `true` when the experimental units' scores are themselves
  cross-fitted (training data = experimental sample); `false` when they come from a
  model fitted on separate pilot or historical data.

# References
- Hansen, B. B. (2008). The prognostic analogue of the propensity score.
  *Biometrika*, 95(2), 481–488.
"""
struct PrognosticScore
    score::Vector{Float64}
    ids::Vector{Any}
    id::Union{Nothing,Symbol}
    r2::Float64
    rmse::Float64
    outcome_sd::Float64
    learner::String
    covariates::Vector{Symbol}
    n_train::Int
    n_folds::Int
    crossfit::Bool
end

function Base.show(io::IO, ::MIME"text/plain", p::PrognosticScore)
    println(io, "Prognostic score (", p.learner, " on ", length(p.covariates),
            " covariates)")
    @printf(io, "Training observations: %d; experimental units: %d\n", p.n_train,
            length(p.score))
    @printf(io, "Cross-fitted (%d folds) out-of-sample R² = %.4f, ", p.n_folds, p.r2)
    @printf(io, "RMSE = %.4g (outcome sd %.4g)\n", p.rmse, p.outcome_sd)
    print(io, p.crossfit ? "Scores of the experimental units are cross-fitted." :
              "Scores of the experimental units come from a model fitted on the " *
              "training data.")
end

Base.show(io::IO, p::PrognosticScore) =
    @printf(io, "PrognosticScore(%d units, R² = %.3f)", length(p.score), p.r2)

"""
    prognostic_score(train, outcome; covariates, learner=RidgeLearner(), target=nothing,
                     id=nothing, n_folds=5, rng=Random.default_rng(),
                     parallel=true) -> PrognosticScore

Predict the outcomes of the experimental units from baseline covariates with any
[`NuisanceLearner`](@ref), for use in blocking, power calculations and covariate
adjustment.

The prognostic score of Hansen (2008) is a prediction of the untreated outcome from
pre-treatment covariates, ``\\hat\\mu(X) \\approx E[Y(0) \\mid X]``. In experimental design
it plays the role that the baseline outcome plays in classical blocking: units with
similar scores have similar expected outcomes, so blocks or pairs formed on the score
([`block_design`](@ref)) remove the predictable part of the outcome variance from the
comparison of arms, and the same score can then enter the analysis as a covariate
([`experiment_estimate`](@ref)). Bai (2022) shows that, among stratified designs
treating each unit with probability one half, the precision-maximizing design pairs
units on an index of the covariates, which is the baseline outcome in an important
special case; a well-predicting score is a feasible stand-in. Aufenanger (2017) and
Gui & Kim (2025) study machine-learning and language-model predictions of the outcome
for stratification.

Two sampling situations are handled. With `target` (the experimental sample, disjoint
from `train`: pilot, historical or control-only data), the learner is fitted on all of
`train` and predicts `target`. Without `target`, `train` is the experimental sample
itself, the outcome is a pre-treatment measurement (e.g. a baseline survey), and every
unit's score is cross-fitted, i.e. predicted by a model that did not see that unit, so
that no unit's score is fitted to its own noise. In both cases the reported R² and
RMSE are cross-fitted in `train`, which is the relevant measure of how much outcome
variance the score will remove; an in-sample R² would overstate it.

Only baseline covariates may be used: a score built from post-treatment variables can
absorb part of the treatment effect and invalidates the design-based analysis. A poor
score costs precision but not validity, because assignment remains random given the
blocks. The out-of-sample R² transfers to the experiment only if the training and
experimental populations are similar; with historical or pilot data from a different
population, treat it as optimistic.

# Arguments
- `train::AbstractDataFrame`: training data containing the outcome and covariates.
- `outcome::Symbol`: outcome to predict (the experiment's outcome, or its
  pre-treatment measurement when `target` is omitted).

# Keywords
- `covariates::Vector{Symbol}`: numeric baseline covariates, present in `target` too
  (required).
- `learner = RidgeLearner()`: any [`NuisanceLearner`](@ref), e.g. `OLSLearner()`,
  `LassoLearner()`, `ForestLearner()` or `MLJLearner(model)`.
- `target = nothing`: the experimental sample to score; when omitted, `train` is
  scored by cross-fitting.
- `id = nothing`: unit identifier column of the experimental sample (`target`, or
  `train` without `target`). Strongly recommended, since the score is matched to units
  by key downstream and the folds then do not depend on row order.
- `n_folds::Integer = 5`: number of cross-fitting folds.
- `rng::AbstractRNG = Random.default_rng()`: random-number generator for the folds and
  the learner; results are reproducible given `rng`.
- `parallel::Bool = true`: fit folds on threads; results are identical with and
  without threads.

# Returns
- [`PrognosticScore`](@ref).

# Examples
```julia
using DrSnow, DataFrames, StableRNGs
rng = StableRNG(1)
mk(n) = (X = randn(rng, n, 3);
         DataFrame(x1=X[:, 1], x2=X[:, 2], x3=X[:, 3],
                   y=X * [1.0, 0.5, 0.0] .+ randn(rng, n)))
pilot = mk(500)                                   # historical data with outcomes
sample = mk(200); sample.id = 1:200               # experimental units
ps = prognostic_score(pilot, :y; covariates=[:x1, :x2, :x3], target=sample,
                      id=:id, rng=StableRNG(2))
ps.r2
```

# References
- Aufenanger, T. (2017). *Machine learning to improve experimental design* (FAU
  Discussion Papers in Economics No. 16/2017). Friedrich-Alexander University
  Erlangen-Nuremberg.
- Bai, Y. (2022). Optimality of matched-pair designs in randomized controlled trials.
  *American Economic Review*, 112(12), 3911–3940.
- Gui, G., & Kim, S. (2025). *Leveraging LLMs to improve experimental design: A
  generative stratification approach* (arXiv:2509.25709). arXiv.
- Hansen, B. B. (2008). The prognostic analogue of the propensity score.
  *Biometrika*, 95(2), 481–488.
"""
function prognostic_score(train::AbstractDataFrame, outcome::Symbol;
                          covariates::Vector{Symbol}, learner=RidgeLearner(),
                          target::Union{Nothing,AbstractDataFrame}=nothing,
                          id::Union{Nothing,Symbol}=nothing, n_folds::Integer=5,
                          rng::AbstractRNG=Random.default_rng(), parallel::Bool=true)
    ctx = "prognostic_score"
    learner isa NuisanceLearner ||
        throw(ArgumentError("$ctx: learner must be a NuisanceLearner"))
    isempty(covariates) && throw(ArgumentError("$ctx: need at least one covariate"))
    outcome in covariates &&
        throw(ArgumentError("$ctx: the outcome cannot be one of the covariates"))
    require_columns(train, vcat(outcome, covariates); context=ctx)
    exp_data = target === nothing ? train : target
    require_columns(exp_data, vcat(covariates, id === nothing ? Symbol[] : [id]);
                    context=ctx)
    # canonical row order of the training data: by id when it is the experimental
    # sample, otherwise by (outcome, covariates) so folds do not depend on row order
    trows = if target === nothing && id !== nothing
        allunique(train[!, id]) ||
            throw(ArgumentError("$ctx: id column $id has duplicates"))
        sortperm(train[!, id]; by=_ri_sortkey)
    else
        _ri_canonical_rows(train, vcat(outcome, covariates))
    end
    tr = train[trows, :]
    y = _ml_column(tr, outcome; context=ctx)
    X = _ml_matrix(tr, covariates; context=ctx)
    n = length(y)
    n >= n_folds || throw(ArgumentError("$ctx: need at least n_folds = $n_folds " *
                                        "training observations"))
    F = crossfit_folds(n, n_folds, 1; rng=rng)
    seeds = _ml_seeds(rng, n_folds, 1, 1)
    oof = _ml_crossfit([_MLNuisance(:prognostic, learner, y, X, false)], F[:, 1],
                       view(seeds, :, :, 1); parallel=parallel, context=ctx)[:, 1]
    sst = sum(abs2, y .- mean(y))
    sst > 0 || throw(ArgumentError("$ctx: the training outcome is constant"))
    sse = sum(abs2, y .- oof)
    r2 = 1 - sse / sst
    rmse = sqrt(sse / n)
    if target === nothing
        score = oof
        ids = id === nothing ? Any[trows...] : Any[tr[i, id] for i in 1:n]
    else
        Xt = _ml_matrix(target, covariates; context=ctx)
        seed = task_seeds(rng, 1)[1]
        score = fitpredict(learner, X, y, Xt; rng=Random.Xoshiro(seed))
        length(score) == size(Xt, 1) && all(isfinite, score) ||
            throw(ArgumentError("$ctx: the learner returned invalid predictions"))
        score = Vector{Float64}(score)
        if id === nothing
            ids = Any[1:nrow(target)...]
        else
            allunique(target[!, id]) ||
                throw(ArgumentError("$ctx: id column $id has duplicates"))
            ids = Any[target[!, id]...]
        end
    end
    return PrognosticScore(score, ids, id, r2, rmse, std(y), _ml_learner_name(learner),
                           copy(covariates), n, Int(n_folds), target === nothing)
end
