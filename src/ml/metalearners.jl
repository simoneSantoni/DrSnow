# Meta-learners for the CATE of a binary treatment: S-, T- and X-learners (Künzel,
# Sekhon, Bickel & Yu 2019) and the R-learner (Nie & Wager 2021), with any
# NuisanceLearner. They deliver point predictions; a pairs bootstrap of summaries is
# provided by `metalearner_bootstrap`.

"""
    MetaLearner

Fitted meta-learner of the conditional average treatment effect, returned by
[`s_learner`](@ref), [`t_learner`](@ref), [`x_learner`](@ref) or
[`r_learner`](@ref).

Meta-learners (Künzel et al. 2019; Nie & Wager 2021) combine generic regression
learners into an estimator of ``\\tau(x) = E[Y(1) - Y(0) \\mid X = x]``. The object
holds the training data, in-sample and cross-fitted CATE predictions, the plug-in
average treatment effect and the fitted learners. It provides point predictions only:
`predict(m, newdata)` refits the learner on all observations with the stored seeds and
predicts at new points, and [`metalearner_bootstrap`](@ref) gives heuristic bootstrap
standard errors. For formal inference use doubly robust scores (see
[`cate_projection`](@ref), [`generic_ml`](@ref)) or [`causal_forest`](@ref).

# Fields
- `kind::Symbol`: `:S`, `:T`, `:X` or `:R`.
- `covariates::Vector{Symbol}`: covariates ``X`` of the outcome (and propensity)
  learners.
- `effect_modifiers::Vector{Symbol}`: variables ``V`` of the final CATE model
  (R-learner; equal to `covariates` for the other learners).
- `outcome::Symbol`, `treatment::Symbol`: the outcome and treatment columns.
- `X::Matrix{Float64}`, `V::Matrix{Float64}`, `Y::Vector{Float64}`,
  `W::Vector{Float64}`: training covariates, effect modifiers, outcomes and treatment
  indicators.
- `cate::Vector{Float64}`: in-sample CATE predictions from the all-data fit.
- `cate_oof::Vector{Float64}`: cross-fitted CATE predictions (each fold predicted by
  a learner fitted on the other folds); use these for honest evaluation, e.g. with
  [`rank_average_treatment_effect`](@ref) or calibration against doubly robust
  scores.
- `ate::Float64`: plug-in average treatment effect, `mean(cate)`.
- `nuisance::NamedTuple`: cross-fitted ``\\hat m(X) = \\hat E[Y \\mid X]`` and
  ``\\hat e(X)`` (R-learner, fields `m`, `e`), the propensity used for weighting
  (X-learner, field `e`), or empty.
- `folds::Vector{Int}`: fold id of each observation.
- `cluster::Union{Nothing,Vector{Int}}`, `n_clusters::Int`: cluster index (or
  `nothing`) and number of clusters, used by the folds and the bootstrap.
- `learner_objects::NamedTuple`: the learner objects of each stage
  (`outcome_learner`, and where applicable `cate_learner`, `propensity_learner`,
  `final_learner`).
- `learners::Vector{Pair{Symbol,String}}`: descriptions of those learners.
- `seeds::Vector{UInt64}`: per-stage seeds drawn from `rng`, reused by `predict`.
"""
struct MetaLearner
    kind::Symbol
    covariates::Vector{Symbol}
    effect_modifiers::Vector{Symbol}
    outcome::Symbol
    treatment::Symbol
    X::Matrix{Float64}
    V::Matrix{Float64}
    Y::Vector{Float64}
    W::Vector{Float64}
    cate::Vector{Float64}
    cate_oof::Vector{Float64}
    ate::Float64
    nuisance::NamedTuple
    folds::Vector{Int}
    cluster::Union{Nothing,Vector{Int}}
    n_clusters::Int
    learner_objects::NamedTuple
    learners::Vector{Pair{Symbol,String}}
    seeds::Vector{UInt64}
end

const _ML_META_NAMES = Dict(:S => "S-learner", :T => "T-learner", :X => "X-learner",
                            :R => "R-learner")

function Base.show(io::IO, ::MIME"text/plain", m::MetaLearner)
    println(io, _ML_META_NAMES[m.kind], " for the CATE (point predictions)")
    println(io, "Observations: ", length(m.Y), ", covariates: ", length(m.covariates))
    for (nm, l) in m.learners
        println(io, "  ", nm, ": ", l)
    end
    @printf(io, "Plug-in ATE (mean in-sample CATE): %.4g\n", m.ate)
    q = quantile(m.cate_oof, [0.1, 0.5, 0.9])
    @printf(io, "Cross-fitted CATE predictions: 10%% %.4g, median %.4g, 90%% %.4g\n", q...)
    println(io, "No standard errors: use metalearner_bootstrap, a causal forest, or " *
                "DR scores with best_linear_projection / GATES for inference.")
    return nothing
end

Base.show(io::IO, m::MetaLearner) =
    print(io, "MetaLearner(", m.kind, ", n = ", length(m.Y), ")")

# --------------------------------------------------------------- algorithms

_ml_meta_proba(l) = !(l isa _ML_GRF_LPM)

function _ml_meta_propensity(l, X, W, Xnew, rng)
    return _ml_meta_proba(l) ? fitpredict_proba(l, X, W, Xnew; rng=rng) :
           clamp.(fitpredict(l, X, W, Xnew; rng=rng), 0.0, 1.0)
end

"""
CATE predictions at `Xnew` (and `Vnew` for the R-learner) from a learner trained on
`(X, V, Y, W)`; `nuis` carries the cross-fitted R-learner nuisances of the training
rows. `seeds` has one seed per stage.
"""
function _ml_meta_fit(kind, L, X, V, Y, W, Xnew, Vnew, seeds, nuis, ctx)
    t = W .== 1
    c = .!t
    (count(t) >= 2 && count(c) >= 2) ||
        throw(ArgumentError("$(ctx): need at least two treated and two control units " *
                            "in every training sample"))
    rngs = [Random.Xoshiro(s) for s in seeds]
    if kind === :S
        XW = hcat(X, W)
        m = size(Xnew, 1)
        Xq = vcat(hcat(Xnew, ones(m)), hcat(Xnew, zeros(m)))
        μ = fitpredict(L.outcome_learner, XW, Y, Xq; rng=rngs[1])
        return μ[1:m] .- μ[(m + 1):end]
    elseif kind === :T
        μ1 = fitpredict(L.outcome_learner, X[t, :], Y[t], Xnew; rng=rngs[1])
        μ0 = fitpredict(L.outcome_learner, X[c, :], Y[c], Xnew; rng=rngs[2])
        return μ1 .- μ0
    elseif kind === :X
        μ0_t = fitpredict(L.outcome_learner, X[c, :], Y[c], X[t, :]; rng=rngs[1])
        μ1_c = fitpredict(L.outcome_learner, X[t, :], Y[t], X[c, :]; rng=rngs[2])
        D1 = Y[t] .- μ0_t
        D0 = μ1_c .- Y[c]
        τ1 = fitpredict(L.cate_learner, X[t, :], D1, Xnew; rng=rngs[3])
        τ0 = fitpredict(L.cate_learner, X[c, :], D0, Xnew; rng=rngs[4])
        g = _ml_meta_propensity(L.propensity_learner, X, W, Xnew, rngs[5])
        return g .* τ0 .+ (1 .- g) .* τ1
    else  # :R, with cross-fitted nuisances of the training rows
        r = W .- nuis.e
        keep = abs.(r) .> 1e-8
        count(keep) >= 2 || throw(ArgumentError("$(ctx): treatment residuals are all zero"))
        pseudo = (Y[keep] .- nuis.m[keep]) ./ r[keep]
        return fitpredict(L.final_learner, V[keep, :], pseudo, Vnew; rng=rngs[1],
                          weights=r[keep] .^ 2)
    end
end

const _ML_META_STAGES = 5

function _ml_metalearner(kind, data, outcome, treatment; covariates, effect_modifiers,
                         learners::NamedTuple, n_folds, folds, cluster, rng, parallel,
                         ctx)
    vs = Symbol.(collect(effect_modifiers))
    require_columns(data, vcat(treatment, vs); context=ctx)
    w = _ml_column(data, treatment; context=ctx)
    _ml_check_binary_col(w, treatment, ctx)
    covs, X, cid, G, F = _ml_setup(data, [outcome, treatment], covariates, cluster, folds,
                                   n_folds, 1, rng, w; context=ctx)
    isempty(covs) && throw(ArgumentError("$(ctx): covariates must not be empty"))
    size(F, 2) == 1 || throw(ArgumentError("$(ctx): supply a single column of fold ids"))
    y = _ml_column(data, outcome; context=ctx)
    V = kind === :R ? _ml_matrix(data, vs; context=ctx) : X
    fold = F[:, 1]
    K = maximum(fold)
    seeds = task_seeds(rng, (K + 1) * _ML_META_STAGES + 2)
    stage_seeds(k) = seeds[(k * _ML_META_STAGES + 1):((k + 1) * _ML_META_STAGES)]
    nuis = (;)
    if kind === :R
        ns = _ml_seeds(Random.Xoshiro(seeds[end]), K, 2, 1)
        specs = [_MLNuisance(:m, learners.outcome_learner, y, X, false),
                 _MLNuisance(:e, learners.propensity_learner, w, X,
                             _ml_meta_proba(learners.propensity_learner))]
        P = _ml_crossfit(specs, fold, view(ns, :, :, 1); parallel=parallel, context=ctx)
        nuis = (m=P[:, 1], e=clamp.(P[:, 2], 0.0, 1.0))
    end
    cate = _ml_meta_fit(kind, learners, X, V, y, w, X, V, stage_seeds(0), nuis, ctx)
    oof = zeros(length(y))
    run = function (k)
        tr = fold .!= k
        te = .!tr
        nk = kind === :R ? (m=nuis.m[tr], e=nuis.e[tr]) : nuis
        oof[te] .= _ml_meta_fit(kind, learners, X[tr, :], V[tr, :], y[tr], w[tr],
                                X[te, :], V[te, :], stage_seeds(k), nk, ctx)
        return nothing
    end
    if parallel && Threads.nthreads() > 1
        tasks = [Threads.@spawn run(k) for k in 1:K]
        for t in tasks
            try
                fetch(t)
            catch e
                e isa TaskFailedException ? throw(e.task.exception) : rethrow()
            end
        end
    else
        foreach(run, 1:K)
    end
    if kind === :X
        e = _ml_meta_propensity(learners.propensity_learner, X, w, X,
                                Random.Xoshiro(stage_seeds(0)[5]))
        nuis = (e=e,)
    end
    lnames = [k => _ml_learner_name(v) for (k, v) in pairs(learners)]
    return MetaLearner(kind, covs, kind === :R ? vs : covs, outcome, treatment, X, V, y, w,
                       cate, oof, mean(cate), nuis, fold, cid, G, learners, lnames, seeds)
end

"""
    s_learner(data, outcome, treatment; covariates, outcome_learner=ForestLearner(),
              n_folds=5, folds=nothing, cluster=nothing, rng=Random.default_rng(),
              parallel=true) -> MetaLearner

Estimate the conditional average treatment effect of a binary treatment with the
S-learner ("single" learner) of Künzel, Sekhon, Bickel and Yu (2019).

The estimand is ``\\tau(x) = E[Y(1) - Y(0) \\mid X = x]``, identified under
unconfoundedness, ``\\{Y(0), Y(1)\\} \\perp W \\mid X``, overlap,
``0 < P(W = 1 \\mid X) < 1``, and SUTVA; the first cannot be tested with the data,
and in a randomized experiment all hold by design. Under these assumptions
``\\tau(x) = \\mu(x, 1) - \\mu(x, 0)`` with ``\\mu(x, w) = E[Y \\mid X = x, W = w]``.

The S-learner fits one regression of ``Y`` on ``(X, W)``, treating the treatment
indicator as just another feature, and predicts
``\\hat\\tau(x) = \\hat\\mu(x, 1) - \\hat\\mu(x, 0)``. Pooling all observations makes
it stable when the effect is small or zero, but a regularized or tree learner may
shrink the treatment towards irrelevance and bias ``\\hat\\tau`` towards zero, and an
additive learner (e.g. [`OLSLearner`](@ref) without interactions) forces a constant
effect. Künzel et al. (2019) find it competitive when the CATE is often zero and
weak otherwise. The in-sample predictions `cate` come from the all-data fit; the
cross-fitted predictions `cate_oof` come from fits that exclude each fold.

Meta-learners provide point predictions only and do not model the propensity score,
so in observational data their bias depends on how well the outcome model adjusts
for confounding. For inference use a [`causal_forest`](@ref) (pointwise intervals),
doubly robust scores with [`cate_projection`](@ref), [`best_linear_projection`](@ref)
or [`generic_ml`](@ref) (GATES), or the heuristic bootstrap of
[`metalearner_bootstrap`](@ref). The doubly robust [`cate_dr_learner`](@ref) and the
[`r_learner`](@ref) are usually preferable with confounding.

# Arguments
- `data`: a `DataFrame` with one row per unit.
- `outcome::Symbol`: numeric outcome column ``Y``.
- `treatment::Symbol`: treatment column ``W``, coded `0`/`1`.

# Keywords
- `covariates::Vector{Symbol}`: covariates ``X`` (required, non-empty).
- `outcome_learner = ForestLearner()`: any [`NuisanceLearner`](@ref) (including
  [`MLJLearner`](@ref)) for ``\\mu(x, w)``.
- `n_folds::Integer = 5`: folds for the cross-fitted predictions `cate_oof`
  (stratified by treatment).
- `folds = nothing`: user-supplied fold ids (column name or vector) instead of random
  folds.
- `cluster = nothing`: cluster column; folds (and the bootstrap) keep clusters
  together.
- `rng::AbstractRNG = Random.default_rng()`: random-number generator; per-stage
  seeds are drawn up front.
- `parallel::Bool = true`: fit folds on several threads when available.

# Returns
- [`MetaLearner`](@ref) with `kind = :S`.

# Examples
```julia
using DrSnow, DataFrames, StableRNGs
rng = StableRNG(1)
n = 800
df = DataFrame(x1=randn(rng, n), x2=randn(rng, n))
df.d = Float64.(rand(rng, n) .< 0.5)
df.y = df.x2 .+ (1 .+ df.x1) .* df.d .+ randn(rng, n)
m = s_learner(df, :y, :d; covariates=[:x1, :x2],
              outcome_learner=ForestLearner(num_trees=100), rng=StableRNG(2))
predict(m, DataFrame(x1=[0.0, 1.0], x2=[0.0, 0.0]))
```

# References
- Künzel, S. R., Sekhon, J. S., Bickel, P. J., & Yu, B. (2019). Metalearners for
  estimating heterogeneous treatment effects using machine learning. *Proceedings of
  the National Academy of Sciences*, 116(10), 4156–4165.
- Kennedy, E. H. (2023). Towards optimal doubly robust estimation of heterogeneous
  causal effects. *Electronic Journal of Statistics*, 17(2), 3008–3049.
- Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical
  Learning: Data Mining, Inference, and Prediction* (2nd ed.). Springer.
"""
function s_learner(data, outcome::Symbol, treatment::Symbol; covariates=Symbol[],
                   outcome_learner=ForestLearner(), n_folds::Integer=5, folds=nothing,
                   cluster=nothing, rng::AbstractRNG=Random.default_rng(),
                   parallel::Bool=true)
    return _ml_metalearner(:S, data, outcome, treatment; covariates=covariates,
                           effect_modifiers=covariates,
                           learners=(outcome_learner=outcome_learner,), n_folds=n_folds,
                           folds=folds, cluster=cluster, rng=rng, parallel=parallel,
                           ctx="s_learner")
end

"""
    t_learner(data, outcome, treatment; covariates, outcome_learner=ForestLearner(),
              n_folds=5, folds=nothing, cluster=nothing, rng=Random.default_rng(),
              parallel=true) -> MetaLearner

Estimate the conditional average treatment effect of a binary treatment with the
T-learner ("two" learners) of Künzel, Sekhon, Bickel and Yu (2019).

The estimand ``\\tau(x) = E[Y(1) - Y(0) \\mid X = x]`` is identified, under
unconfoundedness, overlap and SUTVA (see [`s_learner`](@ref)), as
``\\mu_1(x) - \\mu_0(x)`` with ``\\mu_w(x) = E[Y \\mid X = x, W = w]``. The T-learner
fits ``\\mu_1`` on treated units and ``\\mu_0`` on control units with the same
learner and predicts ``\\hat\\tau(x) = \\hat\\mu_1(x) - \\hat\\mu_0(x)``.

Fitting the arms separately lets the learner capture any form of effect
heterogeneity, but the difference of two separately regularized fits can create
spurious heterogeneity: each fit smooths or selects variables differently, and the
smaller arm is estimated less precisely, so ``\\hat\\tau`` inherits the error of the
worse of the two regressions. Künzel et al. (2019) show that it performs well when
the CATE is complex and both arms are large, and poorly with unbalanced arms or a
simple CATE; the [`x_learner`](@ref) was designed for the unbalanced case. As with
all meta-learners, the output is point predictions only; see [`s_learner`](@ref)
for inference options and [`cate_dr_learner`](@ref) for a doubly robust
alternative.

# Arguments
- `data`: a `DataFrame` with one row per unit.
- `outcome::Symbol`: numeric outcome column ``Y``.
- `treatment::Symbol`: treatment column ``W``, coded `0`/`1`.

# Keywords
- `covariates::Vector{Symbol}`: covariates ``X`` (required, non-empty).
- `outcome_learner = ForestLearner()`: [`NuisanceLearner`](@ref) used for both
  ``\\mu_0`` and ``\\mu_1``.
- `n_folds::Integer = 5`, `folds = nothing`, `cluster = nothing`: folds for the
  cross-fitted predictions `cate_oof`, as in [`s_learner`](@ref).
- `rng::AbstractRNG = Random.default_rng()`, `parallel::Bool = true`: seeds (drawn up
  front) and threading.

# Returns
- [`MetaLearner`](@ref) with `kind = :T`.

# Examples
```julia
using DrSnow, DataFrames, StableRNGs
rng = StableRNG(1)
n = 800
df = DataFrame(x1=randn(rng, n), x2=randn(rng, n))
df.d = Float64.(rand(rng, n) .< 0.5)
df.y = df.x2 .+ (1 .+ df.x1) .* df.d .+ randn(rng, n)
m = t_learner(df, :y, :d; covariates=[:x1, :x2],
              outcome_learner=ForestLearner(num_trees=100), rng=StableRNG(2))
m.ate
```

# References
- Künzel, S. R., Sekhon, J. S., Bickel, P. J., & Yu, B. (2019). Metalearners for
  estimating heterogeneous treatment effects using machine learning. *Proceedings of
  the National Academy of Sciences*, 116(10), 4156–4165.
- Kennedy, E. H. (2023). Towards optimal doubly robust estimation of heterogeneous
  causal effects. *Electronic Journal of Statistics*, 17(2), 3008–3049.
"""
function t_learner(data, outcome::Symbol, treatment::Symbol; covariates=Symbol[],
                   outcome_learner=ForestLearner(), n_folds::Integer=5, folds=nothing,
                   cluster=nothing, rng::AbstractRNG=Random.default_rng(),
                   parallel::Bool=true)
    return _ml_metalearner(:T, data, outcome, treatment; covariates=covariates,
                           effect_modifiers=covariates,
                           learners=(outcome_learner=outcome_learner,), n_folds=n_folds,
                           folds=folds, cluster=cluster, rng=rng, parallel=parallel,
                           ctx="t_learner")
end

"""
    x_learner(data, outcome, treatment; covariates, outcome_learner=ForestLearner(),
              cate_learner=outcome_learner, propensity_learner=ForestLearner(),
              n_folds=5, folds=nothing, cluster=nothing, rng=Random.default_rng(),
              parallel=true) -> MetaLearner

Estimate the conditional average treatment effect of a binary treatment with the
X-learner of Künzel, Sekhon, Bickel and Yu (2019).

The estimand is ``\\tau(x) = E[Y(1) - Y(0) \\mid X = x]``, identified under
unconfoundedness, overlap and SUTVA (see [`s_learner`](@ref)). The X-learner imputes
individual treatment effects with the outcome model of the opposite arm and then
smooths them, in three stages:

1. Fit ``\\mu_0`` on control units and ``\\mu_1`` on treated units.
2. Impute effects ``D^1_i = Y_i - \\hat\\mu_0(X_i)`` for treated and
   ``D^0_i = \\hat\\mu_1(X_i) - Y_i`` for control units, and regress them on ``X``
   within each group with `cate_learner`, giving ``\\hat\\tau_1`` and
   ``\\hat\\tau_0``.
3. Combine them with the propensity score ``\\hat e(x)``:

```math
\\hat\\tau(x) = \\hat e(x)\\,\\hat\\tau_0(x) + \\{1 - \\hat e(x)\\}\\,\\hat\\tau_1(x).
```

The weighting leans on the estimate built from the larger group's outcome model (if
few units are treated, ``\\hat e`` is small and ``\\hat\\tau_1``, which uses the
well-estimated ``\\hat\\mu_0``, dominates). Künzel et al. (2019) show that this makes
the X-learner efficient in unbalanced designs and when the CATE is smoother than the
outcome functions, a common situation in practice. The propensity enters only as a
weight, so the X-learner is not doubly robust: in observational data its bias depends
on the outcome models. Point predictions only; see [`s_learner`](@ref) for inference
options and [`cate_dr_learner`](@ref) or [`r_learner`](@ref) for orthogonalized
alternatives. When `propensity_learner` is a linear-probability learner
([`OLSLearner`](@ref), [`RidgeLearner`](@ref), [`LassoLearner`](@ref)) its
predictions are clipped to ``[0, 1]``; otherwise [`fitpredict_proba`](@ref) is used.

# Arguments
- `data`: a `DataFrame` with one row per unit.
- `outcome::Symbol`: numeric outcome column ``Y``.
- `treatment::Symbol`: treatment column ``W``, coded `0`/`1`.

# Keywords
- `covariates::Vector{Symbol}`: covariates ``X`` (required, non-empty).
- `outcome_learner = ForestLearner()`: learner for ``\\mu_0`` and ``\\mu_1``.
- `cate_learner = outcome_learner`: second-stage learner for the imputed effects.
- `propensity_learner = ForestLearner()`: learner for the weights ``\\hat e(x)``.
- `n_folds::Integer = 5`, `folds = nothing`, `cluster = nothing`: folds for the
  cross-fitted predictions `cate_oof`, as in [`s_learner`](@ref).
- `rng::AbstractRNG = Random.default_rng()`, `parallel::Bool = true`: seeds (drawn up
  front) and threading.

# Returns
- [`MetaLearner`](@ref) with `kind = :X` (the propensity used for weighting is in
  `nuisance.e`).

# Examples
```julia
using DrSnow, DataFrames, StableRNGs
rng = StableRNG(1)
n = 800
df = DataFrame(x1=randn(rng, n), x2=randn(rng, n))
df.d = Float64.(rand(rng, n) .< 0.2)
df.y = df.x2 .+ (1 .+ df.x1) .* df.d .+ randn(rng, n)
m = x_learner(df, :y, :d; covariates=[:x1, :x2],
              outcome_learner=ForestLearner(num_trees=100),
              propensity_learner=LogisticLearner(), rng=StableRNG(2))
predict(m, DataFrame(x1=[0.0, 1.0], x2=[0.0, 0.0]))
```

# References
- Künzel, S. R., Sekhon, J. S., Bickel, P. J., & Yu, B. (2019). Metalearners for
  estimating heterogeneous treatment effects using machine learning. *Proceedings of
  the National Academy of Sciences*, 116(10), 4156–4165.
- Kennedy, E. H. (2023). Towards optimal doubly robust estimation of heterogeneous
  causal effects. *Electronic Journal of Statistics*, 17(2), 3008–3049.
"""
function x_learner(data, outcome::Symbol, treatment::Symbol; covariates=Symbol[],
                   outcome_learner=ForestLearner(), cate_learner=outcome_learner,
                   propensity_learner=ForestLearner(), n_folds::Integer=5, folds=nothing,
                   cluster=nothing, rng::AbstractRNG=Random.default_rng(),
                   parallel::Bool=true)
    L = (outcome_learner=outcome_learner, cate_learner=cate_learner,
         propensity_learner=propensity_learner)
    return _ml_metalearner(:X, data, outcome, treatment; covariates=covariates,
                           effect_modifiers=covariates, learners=L, n_folds=n_folds,
                           folds=folds, cluster=cluster, rng=rng, parallel=parallel,
                           ctx="x_learner")
end

"""
    r_learner(data, outcome, treatment; covariates, effect_modifiers=covariates,
              outcome_learner=ForestLearner(), propensity_learner=ForestLearner(),
              final_learner=LassoLearner(), n_folds=5, folds=nothing, cluster=nothing,
              rng=Random.default_rng(), parallel=true) -> MetaLearner

Estimate the conditional average treatment effect of a binary treatment with the
R-learner of Nie and Wager (2021), which minimizes a Neyman-orthogonal loss based on
Robinson's (1988) residual-on-residual transformation.

The estimand is ``\\tau(v) = E[Y(1) - Y(0) \\mid V = v]`` for effect modifiers ``V``
(a subset of the covariates ``X``), identified under unconfoundedness given ``X``,
overlap and SUTVA (see [`s_learner`](@ref)). Writing ``m(x) = E[Y \\mid X = x]`` and
``e(x) = E[W \\mid X = x]``, Robinson's decomposition gives
``Y - m(X) = \\tau(X)\\{W - e(X)\\} + \\varepsilon`` with
``E[\\varepsilon \\mid X, W] = 0``. The R-learner cross-fits ``\\hat m`` and ``\\hat e``
and minimizes the empirical R-loss over the final learner's class,

```math
\\hat\\tau = \\arg\\min_\\tau \\sum_{i=1}^n
  \\Big[\\{Y_i - \\hat m(X_i)\\} - \\tau(V_i)\\{W_i - \\hat e(X_i)\\}\\Big]^2
  + \\Lambda(\\tau).
```

Because the loss is orthogonal to the nuisances, errors in ``\\hat m`` and ``\\hat e``
affect ``\\hat\\tau`` only through their products; Nie and Wager (2021) show a
quasi-oracle property: under rate conditions, ``\\hat\\tau`` attains the error bound
it would have with known nuisances. The R-loss equals a weighted regression of the
pseudo-outcome ``\\{Y - \\hat m(X)\\}/\\{W - \\hat e(X)\\}`` on ``V`` with weights
``\\{W - \\hat e(X)\\}^2``, so any learner accepting observation weights can be the
final stage (units with ``|W - \\hat e| \\le 10^{-8}`` are dropped). Built-in choices are
[`LassoLearner`](@ref) and [`RidgeLearner`](@ref) (a penalized linear CATE in ``V``;
add transformations of ``V`` to `effect_modifiers` for nonlinear effects) and
[`ForestLearner`](@ref). Propensity predictions are clipped to ``[0, 1]`` only, so
units with ``\\hat e`` near 0 or 1 receive small weights rather than large ones.

Causal forests ([`causal_forest`](@ref)) rely on the same residualization (local
centering) of ``Y`` and ``W``; the R-learner is a natural choice under confounding.
Point predictions only; see [`s_learner`](@ref) for inference options and
[`cate_dr_learner`](@ref) for the doubly robust alternative.

# Arguments
- `data`: a `DataFrame` with one row per unit.
- `outcome::Symbol`: numeric outcome column ``Y``.
- `treatment::Symbol`: treatment column ``W``, coded `0`/`1`.

# Keywords
- `covariates::Vector{Symbol}`: confounders ``X`` of the nuisance models (required).
- `effect_modifiers::Vector{Symbol} = covariates`: variables ``V`` of the final stage
  (non-empty).
- `outcome_learner = ForestLearner()`: learner for ``m(x) = E[Y \\mid X = x]``.
- `propensity_learner = ForestLearner()`: learner for ``e(x)``; linear-probability
  learners are clipped to ``[0, 1]``, other learners use [`fitpredict_proba`](@ref).
- `final_learner = LassoLearner()`: weighted final-stage learner for ``\\tau(v)``.
- `n_folds::Integer = 5`, `folds = nothing`, `cluster = nothing`: folds for the
  nuisance cross-fitting and the cross-fitted predictions `cate_oof` (stratified by
  treatment, grouped by cluster).
- `rng::AbstractRNG = Random.default_rng()`, `parallel::Bool = true`: seeds (drawn up
  front) and threading.

# Returns
- [`MetaLearner`](@ref) with `kind = :R`; the cross-fitted nuisances are in
  `nuisance.m` and `nuisance.e`.

# Examples
```julia
using DrSnow, DataFrames, StableRNGs
rng = StableRNG(1)
n = 800
df = DataFrame(x1=randn(rng, n), x2=randn(rng, n), x3=randn(rng, n))
df.d = Float64.(rand(rng, n) .< 1 ./ (1 .+ exp.(-0.5 .* df.x2)))
df.y = df.x2 .+ (1 .+ df.x1) .* df.d .+ randn(rng, n)
m = r_learner(df, :y, :d; covariates=[:x1, :x2, :x3], effect_modifiers=[:x1],
              outcome_learner=ForestLearner(num_trees=100),
              propensity_learner=ForestLearner(num_trees=100),
              final_learner=RidgeLearner(), rng=StableRNG(2))
predict(m, DataFrame(x1=[-1.0, 0.0, 1.0]))
```

# References
- Nie, X., & Wager, S. (2021). Quasi-oracle estimation of heterogeneous treatment
  effects. *Biometrika*, 108(2), 299–319.
- Robinson, P. M. (1988). Root-N-consistent semiparametric regression.
  *Econometrica*, 56(4), 931–954.
- Athey, S., Tibshirani, J., & Wager, S. (2019). Generalized random forests. *The
  Annals of Statistics*, 47(2), 1148–1178.
- Kennedy, E. H. (2023). Towards optimal doubly robust estimation of heterogeneous
  causal effects. *Electronic Journal of Statistics*, 17(2), 3008–3049.
- Chernozhukov, V., Chetverikov, D., Demirer, M., Duflo, E., Hansen, C., Newey, W.,
  & Robins, J. (2018). Double/debiased machine learning for treatment and structural
  parameters. *The Econometrics Journal*, 21(1), C1–C68.
"""
function r_learner(data, outcome::Symbol, treatment::Symbol; covariates=Symbol[],
                   effect_modifiers=covariates, outcome_learner=ForestLearner(),
                   propensity_learner=ForestLearner(), final_learner=LassoLearner(),
                   n_folds::Integer=5, folds=nothing, cluster=nothing,
                   rng::AbstractRNG=Random.default_rng(), parallel::Bool=true)
    isempty(effect_modifiers) &&
        throw(ArgumentError("r_learner: effect_modifiers must not be empty"))
    L = (outcome_learner=outcome_learner, propensity_learner=propensity_learner,
         final_learner=final_learner)
    return _ml_metalearner(:R, data, outcome, treatment; covariates=covariates,
                           effect_modifiers=effect_modifiers, learners=L, n_folds=n_folds,
                           folds=folds, cluster=cluster, rng=rng, parallel=parallel,
                           ctx="r_learner")
end

"""
    predict(m::MetaLearner, newdata) -> Vector{Float64}

CATE predictions of a fitted meta-learner at new points.

The learner is refitted on all training observations with the stored per-stage seeds
(so repeated calls give identical predictions; the R-learner reuses its stored
cross-fitted nuisances) and evaluated at the rows of `newdata`. The predictions are
point estimates of ``\\tau(x)`` without standard errors; see
[`metalearner_bootstrap`](@ref) for heuristic bootstrap uncertainty and
[`s_learner`](@ref) for formal inference options.

# Arguments
- `m::MetaLearner`: a fitted S-, T-, X- or R-learner.
- `newdata::Union{AbstractDataFrame,AbstractMatrix}`: a `DataFrame` containing the
  covariate columns (for the R-learner, the effect-modifier columns), or a numeric
  matrix with those columns in the same order.

# Returns
- `Vector{Float64}`: one predicted CATE per row of `newdata`.

# Examples
```julia
using DrSnow, DataFrames, StableRNGs
rng = StableRNG(1)
n = 800
df = DataFrame(x1=randn(rng, n))
df.d = Float64.(rand(rng, n) .< 0.5)
df.y = (1 .+ df.x1) .* df.d .+ randn(rng, n)
m = t_learner(df, :y, :d; covariates=[:x1],
              outcome_learner=ForestLearner(num_trees=100), rng=StableRNG(2))
predict(m, DataFrame(x1=[-1.0, 1.0]))
```
"""
function StatsAPI.predict(m::MetaLearner, newdata::AbstractDataFrame)
    ctx = "predict(MetaLearner)"
    cols = m.kind === :R ? m.effect_modifiers : m.covariates
    require_columns(newdata, cols; context=ctx)
    return StatsAPI.predict(m, _ml_matrix(newdata, cols; context=ctx))
end

function StatsAPI.predict(m::MetaLearner, Xnew::AbstractMatrix)
    ctx = "predict(MetaLearner)"
    ref = m.kind === :R ? m.V : m.X
    size(Xnew, 2) == size(ref, 2) ||
        throw(DimensionMismatch("$(ctx): expected $(size(ref, 2)) columns"))
    Xn = Matrix{Float64}(Xnew)
    return _ml_meta_fit(m.kind, m.learner_objects, m.X, m.V, m.Y, m.W,
                        m.kind === :R ? zeros(size(Xn, 1), size(m.X, 2)) : Xn, Xn,
                        m.seeds[1:_ML_META_STAGES], m.nuisance, ctx)
end

"""
    metalearner_bootstrap(m::MetaLearner; newdata=nothing, B=200,
                          rng=Random.default_rng(), parallel=true) -> HTEEstimate

Nonparametric pairs bootstrap of a meta-learner's plug-in average treatment effect
and of its CATE predictions at chosen points.

The whole learner, including the R-learner's cross-fitted nuisances, is refitted on
`B` resamples drawn with replacement from the units (or from the clusters, when the
learner was fitted with `cluster`); resamples with fewer than two treated or two
control units are discarded. The returned estimates are the original plug-in ATE,
``n^{-1}\\sum_i \\hat\\tau(X_i)``, and, with `newdata`, the CATE predictions at each of
its rows; their covariance is the sample covariance of the bootstrap replicates
(Efron & Tibshirani 1993), and 95% percentile intervals are stored in
`details.percentile`.

The bootstrap is a heuristic here. Its validity requires the estimator to be a smooth
functional of the empirical distribution, which regularized, tuned or tree-based
learners are not; duplicated observations in resamples also distort
nearest-neighbour and tree learners, and the plug-in estimates may carry
regularization bias that the bootstrap does not remove. Treat the standard errors as
a rough indication of sampling variability. For formal inference prefer
[`causal_forest`](@ref) with [`predict_interval`](@ref), or doubly robust scores with
[`cate_projection`](@ref), [`best_linear_projection`](@ref) or
[`generic_ml`](@ref).

# Arguments
- `m::MetaLearner`: a fitted S-, T-, X- or R-learner.

# Keywords
- `newdata = nothing`: optional `DataFrame` (with the covariate or, for the
  R-learner, effect-modifier columns) or matrix of points at which to bootstrap the
  CATE.
- `B::Integer = 200`: number of bootstrap resamples (at least 2).
- `rng::AbstractRNG = Random.default_rng()`: random-number generator; per-resample
  seeds are drawn up front so results do not depend on threading.
- `parallel::Bool = true`: run resamples on several threads when available.

# Returns
- [`HTEEstimate`](@ref) with coefficients `"ATE (plug-in)"` and `"CATE[i]"`,
  `dof_residual = Inf`, and `details` holding `B` (valid resamples), `percentile`
  (95% percentile bounds), `draws` (the replicates) and a note.

# Examples
```julia
using DrSnow, DataFrames, StableRNGs
rng = StableRNG(1)
n = 500
df = DataFrame(x1=randn(rng, n), x2=randn(rng, n))
df.d = Float64.(rand(rng, n) .< 0.5)
df.y = df.x2 .+ (1 .+ df.x1) .* df.d .+ randn(rng, n)
m = t_learner(df, :y, :d; covariates=[:x1, :x2], outcome_learner=OLSLearner(),
              rng=StableRNG(2))
b = metalearner_bootstrap(m; newdata=DataFrame(x1=[0.0, 1.0], x2=[0.0, 0.0]),
                          B=50, rng=StableRNG(3))
coeftable(b)
```

# References
- Efron, B., & Tibshirani, R. J. (1993). *An Introduction to the Bootstrap*. Chapman &
  Hall.
- Künzel, S. R., Sekhon, J. S., Bickel, P. J., & Yu, B. (2019). Metalearners for
  estimating heterogeneous treatment effects using machine learning. *Proceedings of
  the National Academy of Sciences*, 116(10), 4156–4165.
"""
function metalearner_bootstrap(m::MetaLearner; newdata=nothing, B::Integer=200,
                               rng::AbstractRNG=Random.default_rng(), parallel::Bool=true)
    ctx = "metalearner_bootstrap"
    B >= 2 || throw(ArgumentError("$(ctx): B must be at least 2"))
    Xq, Vq = if newdata === nothing
        zeros(0, size(m.X, 2)), zeros(0, size(m.V, 2))
    elseif newdata isa AbstractDataFrame
        cols = m.kind === :R ? m.effect_modifiers : m.covariates
        require_columns(newdata, cols; context=ctx)
        M = _ml_matrix(newdata, cols; context=ctx)
        m.kind === :R ? (zeros(size(M, 1), size(m.X, 2)), M) : (M, M)
    else
        M = Matrix{Float64}(newdata)
        m.kind === :R ? (zeros(size(M, 1), size(m.X, 2)), M) : (M, M)
    end
    nq = size(Xq, 1)
    n = length(m.Y)
    units = m.cluster === nothing ? collect(1:n) : m.cluster
    G = m.cluster === nothing ? n : m.n_clusters
    members = [Int[] for _ in 1:G]
    for i in 1:n
        push!(members[units[i]], i)
    end
    seeds = task_seeds(rng, B)
    T = fill(NaN, B, 1 + nq)
    K = maximum(m.folds)
    run = function (b)
        brng = Random.Xoshiro(seeds[b])
        idx = reduce(vcat, members[rand(brng, 1:G, G)])
        X, V, Y, W = m.X[idx, :], m.V[idx, :], m.Y[idx], m.W[idx]
        (count(==(1), W) >= 2 && count(==(0), W) >= 2) || return nothing
        st = rand(brng, UInt64, _ML_META_STAGES)
        nuis = m.nuisance
        if m.kind === :R
            F = crossfit_folds(length(Y), K, 1; rng=brng, strata=W)[:, 1]
            ns = _ml_seeds(brng, K, 2, 1)
            L = m.learner_objects
            specs = [_MLNuisance(:m, L.outcome_learner, Y, X, false),
                     _MLNuisance(:e, L.propensity_learner, W, X,
                                 _ml_meta_proba(L.propensity_learner))]
            P = _ml_crossfit(specs, F, view(ns, :, :, 1); parallel=false, context=ctx)
            nuis = (m=P[:, 1], e=clamp.(P[:, 2], 0.0, 1.0))
        end
        Xall = vcat(X, Xq)
        Vall = vcat(V, Vq)
        τ = _ml_meta_fit(m.kind, m.learner_objects, X, V, Y, W, Xall, Vall, st, nuis, ctx)
        T[b, 1] = mean(τ[1:length(Y)])
        nq > 0 && (T[b, 2:end] .= τ[(length(Y) + 1):end])
        return nothing
    end
    if parallel && Threads.nthreads() > 1
        tasks = [Threads.@spawn run(b) for b in 1:B]
        for t in tasks
            try
                fetch(t)
            catch e
                e isa TaskFailedException ? throw(e.task.exception) : rethrow()
            end
        end
    else
        foreach(run, 1:B)
    end
    ok = findall(b -> !isnan(T[b, 1]), 1:B)
    length(ok) >= 2 || throw(ArgumentError("$(ctx): too few valid bootstrap samples"))
    T = T[ok, :]
    est = vcat(m.ate, nq > 0 ? predict(m, m.kind === :R ? Vq : Xq) : Float64[])
    names = vcat("ATE (plug-in)", ["CATE[$i]" for i in 1:nq])
    V = size(T, 2) == 1 ? fill(var(T[:, 1]), 1, 1) : cov(T)
    pct = hcat([quantile(T[:, j], 0.025) for j in axes(T, 2)],
               [quantile(T[:, j], 0.975) for j in axes(T, 2)])
    return HTEEstimate(names, est, Matrix(Symmetric(V)), n, Inf,
                       "plug-in ATE and CATEs of the " * _ML_META_NAMES[m.kind],
                       _ML_META_NAMES[m.kind] * " with pairs bootstrap",
                       (B=length(ok), percentile=pct, draws=T,
                        note="Bootstrap standard errors for machine-learning fits " *
                             "are heuristic."))
end
