# Heterogeneous treatment effects with the DR-learner (Kennedy 2023): cross-fitted
# doubly-robust pseudo-outcomes regressed on effect modifiers with a user-chosen
# second-stage learner, and the best linear projection of the CATE on a basis with
# valid inference (Semenova & Chernozhukov 2021).

"""
    CATEPredictor

Fitted DR-learner of the conditional average treatment effect, returned by
[`cate_dr_learner`](@ref).

The object stores everything needed to predict, summarize and project the
conditional average treatment effect (CATE)
``\\tau(v) = E[Y(1) - Y(0) \\mid V = v]`` of a binary treatment: the cross-fitted
nuisance predictions, the doubly robust pseudo-outcomes built from them, the
second-stage learner and its out-of-fold predictions, and the augmented
inverse-probability-weighted (AIPW) estimate of the average treatment effect implied by
the pseudo-outcomes. `predict(c, newdata)` refits the second stage on all
pseudo-outcomes and evaluates it at new effect-modifier values;
[`cate_projection`](@ref) gives inference on the best linear projection of
``\\tau(V)`` on a basis. Pointwise confidence intervals for ``\\tau(v)`` are not
available because the second stage is a generic learner.

# Fields
- `cate_learner`: the second-stage [`NuisanceLearner`](@ref) that regresses the
  pseudo-outcomes on the effect modifiers.
- `effect_modifiers::Vector{Symbol}`: names of the effect-modifier columns ``V``.
- `V::Matrix{Float64}`: the effect-modifier matrix of the estimation sample
  (`n × length(effect_modifiers)`).
- `pseudo_outcomes::Vector{Float64}`: cross-fitted doubly robust pseudo-outcomes
  ``\\hat\\varphi_i = \\hat\\mu_1 - \\hat\\mu_0 + D(Y - \\hat\\mu_1)/\\hat\\pi
  - (1 - D)(Y - \\hat\\mu_0)/(1 - \\hat\\pi)``, evaluated at ``X_i``.
- `cate_oof::Vector{Float64}`: out-of-fold CATE predictions (the second stage is
  also cross-fitted on the same folds), suitable for in-sample summaries, ranking
  and calibration checks.
- `mu0::Vector{Float64}`, `mu1::Vector{Float64}`: out-of-fold predictions of
  ``\\mu_d(X) = E[Y \\mid D = d, X]``.
- `propensity::Vector{Float64}`: out-of-fold propensity scores
  ``\\hat\\pi(X) = \\hat P(D = 1 \\mid X)`` after clipping to `[trim, 1 - trim]`.
- `trim::Float64`: the clipping threshold; `n_trimmed::Int`: the number of clipped
  propensity predictions.
- `ate::Float64`, `ate_se::Float64`: AIPW estimate of the average treatment effect
  (the mean of the pseudo-outcomes) and its (cluster-robust, when `cluster` was
  given) standard error.
- `folds::Vector{Int}`: fold id of each observation.
- `cluster::Union{Nothing,Vector{Int}}`, `n_clusters::Int`: cluster index of each
  observation (or `nothing`) and the number of clusters.
- `learners::Vector{Pair{Symbol,String}}`: description of the learner used for each
  nuisance (`:ml_g0`, `:ml_g1`, `:ml_m`) and for the second stage (`:cate`).
- `seed::UInt64`: seed used when the second stage is refitted by `predict`.
"""
struct CATEPredictor
    cate_learner::Any
    effect_modifiers::Vector{Symbol}
    V::Matrix{Float64}
    pseudo_outcomes::Vector{Float64}
    cate_oof::Vector{Float64}
    mu0::Vector{Float64}
    mu1::Vector{Float64}
    propensity::Vector{Float64}
    trim::Float64
    n_trimmed::Int
    ate::Float64
    ate_se::Float64
    folds::Vector{Int}
    cluster::Union{Nothing,Vector{Int}}
    n_clusters::Int
    learners::Vector{Pair{Symbol,String}}
    seed::UInt64
end

function Base.show(io::IO, ::MIME"text/plain", c::CATEPredictor)
    println(io, "DR-learner CATE predictor (Kennedy 2023)")
    println(io, "Observations: ", length(c.pseudo_outcomes), ", effect modifiers: ",
            join(c.effect_modifiers, ", "))
    for (nm, l) in c.learners
        println(io, "  ", nm, ": ", l)
    end
    z = critical_value(0.95)
    @printf(io, "ATE (AIPW): %.4g (se %.3g, 95%% CI [%.4g, %.4g])\n", c.ate, c.ate_se,
            c.ate - z * c.ate_se, c.ate + z * c.ate_se)
    q = quantile(c.cate_oof, [0.1, 0.5, 0.9])
    @printf(io, "Out-of-fold CATE predictions: 10%% %.4g, median %.4g, 90%% %.4g\n", q...)
    c.n_trimmed > 0 && @printf(io, "Propensities clipped to [%.3g, %.3g]: %d\n", c.trim,
                               1 - c.trim, c.n_trimmed)
    return nothing
end

Base.show(io::IO, c::CATEPredictor) =
    print(io, "CATEPredictor(n = ", length(c.pseudo_outcomes), ")")

"""
    predict(c::CATEPredictor, newdata) -> Vector{Float64}

Predict the conditional average treatment effect at new effect-modifier values from
a fitted DR-learner.

The second-stage learner stored in `c` is refitted on all cross-fitted doubly robust
pseudo-outcomes (with the stored seed, so repeated calls return identical values) and
evaluated at the rows of `newdata`, giving ``\\hat\\tau(v)`` for each row. The
predictions inherit the statistical properties of the DR-learner described in
[`cate_dr_learner`](@ref): they are point predictions without standard errors. For
in-sample summaries prefer the out-of-fold predictions `c.cate_oof`, which are never
evaluated on the observations used to fit them.

# Arguments
- `c::CATEPredictor`: a DR-learner fitted by [`cate_dr_learner`](@ref).
- `newdata::Union{AbstractDataFrame,AbstractMatrix}`: a `DataFrame` that contains the
  effect-modifier columns `c.effect_modifiers`, or a numeric matrix with exactly
  those columns in the same order.

# Returns
- `Vector{Float64}`: one predicted CATE per row of `newdata`.

# Examples
```julia
using DrSnow, DataFrames, StableRNGs
rng = StableRNG(1)
n = 1000
df = DataFrame(x1=randn(rng, n), x2=randn(rng, n))
df.d = Float64.(rand(rng, n) .< 1 ./ (1 .+ exp.(-0.5 .* df.x2)))
df.y = df.x2 .+ (1 .+ df.x1) .* df.d .+ randn(rng, n)
c = cate_dr_learner(df, :y, :d; covariates=[:x1, :x2], effect_modifiers=[:x1],
                    cate_learner=OLSLearner(), rng=StableRNG(2))
predict(c, DataFrame(x1=[-1.0, 0.0, 1.0]))
```
"""
function StatsAPI.predict(c::CATEPredictor, newdata::AbstractDataFrame)
    require_columns(newdata, c.effect_modifiers; context="predict(CATEPredictor)")
    return StatsAPI.predict(c, _ml_matrix(newdata, c.effect_modifiers;
                                          context="predict(CATEPredictor)"))
end

function StatsAPI.predict(c::CATEPredictor, Vnew::AbstractMatrix)
    size(Vnew, 2) == size(c.V, 2) ||
        throw(DimensionMismatch("expected $(size(c.V, 2)) effect-modifier columns"))
    return fitpredict(c.cate_learner, c.V, c.pseudo_outcomes, Matrix{Float64}(Vnew);
                      rng=Random.Xoshiro(c.seed))
end

"""Cross-fitted AIPW nuisances and pseudo-outcomes for a binary treatment."""
function _ml_aipw_nuisances(y, d, X, F, seeds, outcome_learner, propensity_learner,
                            trim, parallel, ctx)
    treated = BitVector(d .== 1)
    specs = [_MLNuisance(:ml_g0, outcome_learner, y, X, false, .!treated),
             _MLNuisance(:ml_g1, outcome_learner, y, X, false, treated),
             _MLNuisance(:ml_m, propensity_learner, d, X, true)]
    P = _ml_crossfit(specs, F, seeds; parallel=parallel, context=ctx)
    μ0, μ1, π = P[:, 1], P[:, 2], P[:, 3]
    nt = _ml_clip!(π, trim)
    φ = μ1 .- μ0 .+ d .* (y .- μ1) ./ π .- (1 .- d) .* (y .- μ0) ./ (1 .- π)
    return μ0, μ1, π, nt, φ
end

"""Standard error of a sample mean, cluster-robust when `cluster` is given."""
function _ml_mean_se(x, cluster, G)
    n = length(x)
    e = x .- mean(x)
    if cluster === nothing
        return sqrt(sum(abs2, e) / n) / sqrt(n)
    end
    S = zeros(G)
    for i in 1:n
        S[cluster[i]] += e[i]
    end
    return sqrt(G / (G - 1) * sum(abs2, S)) / n
end

"""
    cate_dr_learner(data, outcome, treatment; covariates=Symbol[],
                    effect_modifiers=covariates, outcome_learner=LassoLearner(),
                    propensity_learner=PenalizedLogisticLearner(),
                    cate_learner=LassoLearner(), trim=0.01, stratify=true, n_folds=5,
                    folds=nothing, cluster=nothing, rng=Random.default_rng(),
                    parallel=true) -> CATEPredictor

Estimate the conditional average treatment effect of a binary treatment with the
doubly robust DR-learner of Kennedy (2023).

The target is the conditional average treatment effect (CATE) given a set of effect
modifiers ``V``,

```math
\\tau(v) = E[Y(1) - Y(0) \\mid V = v],
```

where ``Y(1)`` and ``Y(0)`` are the potential outcomes under treatment and control
and ``V`` is a subset of (or equal to) the confounders ``X``. Identification rests on
unconfoundedness, ``\\{Y(0), Y(1)\\} \\perp D \\mid X``, and overlap,
``0 < \\pi(X) = P(D = 1 \\mid X) < 1``, together with the stable unit treatment value
assumption. Unconfoundedness cannot be checked with the data; overlap can be
inspected through the stored propensity scores. In a randomized experiment both hold
by design.

The estimator follows Kennedy (2023). The outcome regressions
``\\mu_d(X) = E[Y \\mid D = d, X]`` and the propensity score ``\\pi(X)`` are
cross-fitted over `n_folds` folds, and each observation receives the doubly robust
(AIPW) pseudo-outcome of Robins, Rotnitzky and Zhao (1994),

```math
\\hat\\varphi = \\hat\\mu_1(X) - \\hat\\mu_0(X)
  + \\frac{D\\,\\{Y - \\hat\\mu_1(X)\\}}{\\hat\\pi(X)}
  - \\frac{(1 - D)\\{Y - \\hat\\mu_0(X)\\}}{1 - \\hat\\pi(X)},
```

whose conditional mean given ``X`` equals ``\\tau(X)`` when either nuisance is
correct. The second-stage learner `cate_learner` then regresses ``\\hat\\varphi`` on
``V``. Kennedy (2023, Theorem 2) shows that the error of this two-stage procedure
equals that of an infeasible oracle regression of the true pseudo-outcome on ``V``
plus a term driven by the product of the propensity and outcome-regression errors, so
the DR-learner attains the oracle rate whenever that product is of smaller order.
Propensity predictions are clipped to `[trim, 1 - trim]`; clipping keeps the
pseudo-outcomes finite but does not repair poor overlap, and heavy clipping signals
that the CATE is weakly identified in parts of the covariate space.

The second stage is a generic learner, so no pointwise confidence intervals for
``\\tau(v)`` are reported. Valid inference is available for summaries of the CATE:
[`cate_projection`](@ref) gives the best linear projection of ``\\tau(V)`` on a basis
with robust standard errors (Semenova & Chernozhukov 2021), and the stored mean of
the pseudo-outcomes is the AIPW estimate of the average treatment effect with its
standard error. For pointwise intervals use [`causal_forest`](@ref) with
[`predict_interval`](@ref); for tests of heterogeneity in randomized experiments use
[`generic_ml`](@ref); for alternatives with the same inputs see
[`r_learner`](@ref) and [`x_learner`](@ref). Report the nuisance learners, the number
of clipped propensities and the projection coefficients rather than individual
predictions.

# Arguments
- `data`: a `DataFrame` (or Tables.jl table) with one row per unit.
- `outcome::Symbol`: the numeric outcome column ``Y``.
- `treatment::Symbol`: the treatment column ``D``, coded `0`/`1`.

# Keywords
- `covariates::Vector{Symbol} = Symbol[]`: confounders ``X`` used by the outcome and
  propensity learners.
- `effect_modifiers::Vector{Symbol} = covariates`: variables ``V`` on which the CATE
  is modelled in the second stage; a smaller set gives a coarser but more precisely
  estimated CATE.
- `outcome_learner = LassoLearner()`: [`NuisanceLearner`](@ref) for
  ``\\mu_0`` and ``\\mu_1`` (fitted separately on controls and treated units).
- `propensity_learner = PenalizedLogisticLearner()`: learner for ``\\pi(X)``; it
  must implement [`fitpredict_proba`](@ref).
- `cate_learner = LassoLearner()`: second-stage learner that regresses the
  pseudo-outcomes on ``V``.
- `trim::Real = 0.01`: propensity clipping threshold (`0` disables clipping).
- `stratify::Bool = true`: stratify the folds by treatment status.
- `n_folds::Integer = 5`: number of cross-fitting folds.
- `folds = nothing`: user-supplied fold ids (a column name or a vector) instead of
  random folds; a single column only.
- `cluster = nothing`: cluster column; folds keep clusters together and the ATE
  standard error and projections become cluster-robust.
- `rng::AbstractRNG = Random.default_rng()`: random-number generator for folds and
  learner seeds (per-task seeds are drawn up front).
- `parallel::Bool = true`: fit folds on several threads when available; results do
  not depend on the number of threads.

# Returns
- [`CATEPredictor`](@ref): use `predict(c, newdata)`, [`cate_projection`](@ref), and
  the fields `cate_oof`, `pseudo_outcomes`, `ate`, `ate_se`, `propensity`.

# Examples
```julia
using DrSnow, DataFrames, StableRNGs
rng = StableRNG(1)
n = 1000
df = DataFrame(x1=randn(rng, n), x2=randn(rng, n), x3=randn(rng, n))
df.d = Float64.(rand(rng, n) .< 1 ./ (1 .+ exp.(-0.5 .* df.x2)))
df.y = df.x2 .+ (1 .+ df.x1) .* df.d .+ randn(rng, n)
c = cate_dr_learner(df, :y, :d; covariates=[:x1, :x2, :x3],
                    effect_modifiers=[:x1], cate_learner=OLSLearner(),
                    rng=StableRNG(2))
predict(c, DataFrame(x1=[-1.0, 0.0, 1.0]))
coeftable(cate_projection(c))
```

# References
- Kennedy, E. H. (2023). Towards optimal doubly robust estimation of heterogeneous
  causal effects. *Electronic Journal of Statistics*, 17(2), 3008–3049.
- Robins, J. M., Rotnitzky, A., & Zhao, L. P. (1994). Estimation of regression
  coefficients when some regressors are not always observed. *Journal of the
  American Statistical Association*, 89(427), 846–866.
- Semenova, V., & Chernozhukov, V. (2021). Debiased machine learning of conditional
  average treatment effects and other causal functions. *The Econometrics Journal*,
  24(2), 264–289.
- Chernozhukov, V., Chetverikov, D., Demirer, M., Duflo, E., Hansen, C., Newey, W.,
  & Robins, J. (2018). Double/debiased machine learning for treatment and structural
  parameters. *The Econometrics Journal*, 21(1), C1–C68.
- Künzel, S. R., Sekhon, J. S., Bickel, P. J., & Yu, B. (2019). Metalearners for
  estimating heterogeneous treatment effects using machine learning. *Proceedings of
  the National Academy of Sciences*, 116(10), 4156–4165.
"""
function cate_dr_learner(data, outcome::Symbol, treatment::Symbol; covariates=Symbol[],
                         effect_modifiers=covariates, outcome_learner=LassoLearner(),
                         propensity_learner=PenalizedLogisticLearner(),
                         cate_learner=LassoLearner(), trim::Real=0.01,
                         stratify::Bool=true, n_folds::Integer=5, folds=nothing,
                         cluster=nothing, rng::AbstractRNG=Random.default_rng(),
                         parallel::Bool=true)
    ctx = "cate_dr_learner"
    trim = _ml_check_trim(trim)
    vs = Symbol.(collect(effect_modifiers))
    require_columns(data, vcat(treatment, vs); context=ctx)
    d = _ml_column(data, treatment; context=ctx)
    _ml_check_binary_col(d, treatment, ctx)
    covs, X, cid, G, F = _ml_setup(data, [outcome, treatment], covariates, cluster, folds,
                                   n_folds, 1, rng, stratify ? d : nothing; context=ctx)
    size(F, 2) == 1 || throw(ArgumentError("$ctx: supply a single column of fold ids"))
    y = _ml_column(data, outcome; context=ctx)
    V = _ml_matrix(data, vs; context=ctx)
    K = maximum(F)
    seeds = _ml_seeds(rng, K, 4, 1)
    final_seed = task_seeds(rng, 1)[1]
    fold = F[:, 1]
    μ0, μ1, π, nt, φ = _ml_aipw_nuisances(y, d, X, fold, view(seeds, :, 1:3, 1),
                                          outcome_learner, propensity_learner, trim,
                                          parallel, ctx)
    oof = _ml_crossfit([_MLNuisance(:cate, cate_learner, φ, V, false)], fold,
                       view(seeds, :, 4:4, 1); parallel=parallel, context=ctx)[:, 1]
    learners = [:ml_g0 => _ml_learner_name(outcome_learner),
                :ml_g1 => _ml_learner_name(outcome_learner),
                :ml_m => _ml_learner_name(propensity_learner),
                :cate => _ml_learner_name(cate_learner)]
    return CATEPredictor(cate_learner, vs, V, φ, oof, μ0, μ1, π, trim, nt, mean(φ),
                         _ml_mean_se(φ, cid, G), fold, cid, G, learners, final_seed)
end

"""
    CATEProjection <: CausalEstimate

Best linear projection of the conditional average treatment effect on a basis of
effect modifiers, returned by [`cate_projection`](@ref) (DR-learner) and
[`best_linear_projection`](@ref) (causal and instrumental forests).

The coefficients ``\\beta`` solve
``\\min_b E[\\{\\tau(A) - b'A\\}^2]`` for the basis ``A`` (including an intercept when
requested), estimated by least squares of doubly robust scores on ``A``. They are a
well-defined summary of how the CATE co-varies with the basis whether or not the CATE
is linear (Semenova & Chernozhukov 2021). The object supports the standard
[`CausalEstimate`](@ref) interface: `coef`, `vcov`, `stderror`, `confint(r; level)`,
`coeftable`, `nobs`, `dof_residual`, `estimand` and `method_name`.

# Fields
- `names::Vector{String}`: coefficient names (`"(Intercept)"` and the basis columns).
- `coef::Vector{Float64}`: estimated projection coefficients.
- `vcov::Matrix{Float64}`: their heteroskedasticity- or cluster-robust covariance
  (HC1 / CR1 for the DR-learner; the requested `vcov_type` for forests).
- `n::Int`: number of observations used.
- `dof::Float64`: residual degrees of freedom for `t` inference: `G - 1` with `G`
  clusters, otherwise `n - k` for `k` coefficients.
- `method::String`: description of the estimator that produced the projection.
"""
struct CATEProjection <: CausalEstimate
    names::Vector{String}
    coef::Vector{Float64}
    vcov::Matrix{Float64}
    n::Int
    dof::Float64
    method::String
end

CATEProjection(names, coef, vcov, n, dof) =
    CATEProjection(names, coef, vcov, n, dof,
                   "DR-learner CATE projection (Semenova–Chernozhukov)")

StatsAPI.coef(r::CATEProjection) = r.coef
StatsAPI.vcov(r::CATEProjection) = r.vcov
StatsAPI.coefnames(r::CATEProjection) = r.names
StatsAPI.nobs(r::CATEProjection) = r.n
StatsAPI.dof_residual(r::CATEProjection) = r.dof
estimand(::CATEProjection) = "best linear projection of the CATE"
method_name(r::CATEProjection) = r.method

"""
    cate_projection(c::CATEPredictor; basis=c.effect_modifiers,
                    intercept=true) -> CATEProjection

Estimate the best linear projection of the conditional average treatment effect on a
basis of effect modifiers, with valid inference, from a fitted DR-learner.

For a basis ``A`` (a subset of the effect modifiers ``V``, preceded by a constant when
`intercept = true`), the estimand is the coefficient vector

```math
\\beta = \\arg\\min_b E\\big[\\{\\tau(V) - b'A\\}^2\\big]
       = E[AA']^{-1} E[A\\,\\tau(V)],
\\qquad \\tau(v) = E[Y(1) - Y(0) \\mid V = v].
```

It is defined without assuming that the CATE is linear: ``\\beta`` summarizes the
linear trend of ``\\tau`` along ``A``, and with only an intercept it equals the
average treatment effect. Identification requires the same unconfoundedness and
overlap assumptions as [`cate_dr_learner`](@ref).

Because the cross-fitted doubly robust pseudo-outcomes ``\\hat\\varphi`` satisfy
``E[\\varphi \\mid X] = \\tau(X)``, ``\\beta`` is estimated by ordinary least squares of
``\\hat\\varphi`` on ``A``. Semenova and Chernozhukov (2021) show that this estimator is
``\\sqrt{n}``-consistent and asymptotically normal under the product-rate conditions of
double/debiased machine learning, because the pseudo-outcome is Neyman-orthogonal to
the nuisance functions. Standard errors are heteroskedasticity-robust (HC1) or, when
the DR-learner was fitted with `cluster`, cluster-robust (CR1) with `G - 1`
degrees of freedom; otherwise `t` inference uses `n - k` degrees of freedom. The
precision of the projection is limited by the variance of the pseudo-outcomes, which
grows as propensities approach 0 or 1.

Use the projection to report whether and how effects vary along pre-specified
dimensions; a slope that is not significantly different from zero does not show that
the CATE is constant along that variable (it may vary nonlinearly). For projections
from forest-based scores see [`best_linear_projection`](@ref); for categorical
subgroups see [`subgroup_effects`](@ref).

# Arguments
- `c::CATEPredictor`: a DR-learner fitted by [`cate_dr_learner`](@ref).

# Keywords
- `basis::Vector{Symbol} = c.effect_modifiers`: effect-modifier columns on which to
  project; must be a subset of `c.effect_modifiers` and may be empty when
  `intercept = true`.
- `intercept::Bool = true`: include a constant; set `false` only when the basis spans
  the constant (e.g. a full set of group dummies).

# Returns
- [`CATEProjection`](@ref): coefficients named `"(Intercept)"` and the basis columns,
  with robust covariance; use `coeftable`, `confint`, `stderror`.

# Examples
```julia
using DrSnow, DataFrames, StableRNGs
rng = StableRNG(1)
n = 1000
df = DataFrame(x1=randn(rng, n), x2=randn(rng, n))
df.d = Float64.(rand(rng, n) .< 1 ./ (1 .+ exp.(-0.5 .* df.x2)))
df.y = df.x2 .+ (1 .+ df.x1) .* df.d .+ randn(rng, n)
c = cate_dr_learner(df, :y, :d; covariates=[:x1, :x2], rng=StableRNG(2))
coeftable(cate_projection(c; basis=[:x1]))
```

# References
- Semenova, V., & Chernozhukov, V. (2021). Debiased machine learning of conditional
  average treatment effects and other causal functions. *The Econometrics Journal*,
  24(2), 264–289.
- Kennedy, E. H. (2023). Towards optimal doubly robust estimation of heterogeneous
  causal effects. *Electronic Journal of Statistics*, 17(2), 3008–3049.
- Chernozhukov, V., Chetverikov, D., Demirer, M., Duflo, E., Hansen, C., Newey, W.,
  & Robins, J. (2018). Double/debiased machine learning for treatment and structural
  parameters. *The Econometrics Journal*, 21(1), C1–C68.
"""
function cate_projection(c::CATEPredictor; basis=c.effect_modifiers,
                         intercept::Bool=true)
    bs = Symbol.(collect(basis))
    idx = map(bs) do b
        j = findfirst(==(b), c.effect_modifiers)
        j === nothing && throw(ArgumentError("cate_projection: $(b) is not an effect " *
                                             "modifier of the fitted CATEPredictor"))
        j
    end
    n = length(c.pseudo_outcomes)
    Z = intercept ? hcat(ones(n), c.V[:, idx]) : c.V[:, idx]
    size(Z, 2) >= 1 || throw(ArgumentError("cate_projection: empty basis"))
    names = vcat(intercept ? ["(Intercept)"] : String[], string.(bs))
    β, V = _ml_wls_sandwich(Z, c.pseudo_outcomes, ones(n), c.cluster, c.n_clusters)
    dof = c.cluster === nothing ? n - size(Z, 2) : c.n_clusters - 1.0
    return CATEProjection(names, β, V, n, dof)
end
