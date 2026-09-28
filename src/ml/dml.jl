# Double/debiased machine learning with Neyman-orthogonal linear scores
# (Chernozhukov et al. 2018), mirroring the DoubleML package: partially linear
# regression (PLR), interactive regression (IRM: ATE / ATTE), partially linear IV
# (PLIV) and interactive IV (IIVM: LATE).
#
# Every score is linear in the target parameter, ψ = θ ψ_a + ψ_b. For each
# repetition of cross-fitting, θ solves mean(ψ) = 0 over the whole sample ("DML2")
# and the variance is J⁻² mean(ψ²)/n with J = mean(ψ_a) (cluster-robust with
# `cluster`). Repetitions are aggregated with the median rule: θ̃ = median_r θ_r and
# Var(θ̃) = median_r(se_r² + (θ_r - θ̃)²), with the dispersion term on the Var(θ̂)
# scale (more conservative than DoubleML, which adds it to the √n-scaled variance).

"""
    DMLEstimate <: CausalEstimate

Result of a cross-fitted double/debiased machine-learning (DML) estimator:
[`dml_plr`](@ref), [`dml_irm`](@ref), [`dml_pliv`](@ref), [`dml_iivm`](@ref) and
[`dml_did`](@ref).

Every estimator in this family solves an estimating equation that is linear in the
target parameter, ``E[ψ(W; θ_0, η_0)] = 0`` with ``ψ = θ ψ_a + ψ_b``, where the
score ``ψ`` is Neyman orthogonal with respect to the nuisance functions ``η`` and the
nuisances are predicted for each observation by learners trained on the folds that do
not contain it (Chernozhukov et al. 2018). The object stores, for every repetition of
the sample split, the point estimate, the per-observation score components, the
out-of-fold nuisance predictions and their prediction losses, so that an estimate can
be audited (see [`nuisance_loss`](@ref)) and re-aggregated.

With ``R`` repetitions of cross-fitting, the reported coefficient is the median
``\\tilde θ_j = \\operatorname{median}_r \\hat θ_{r,j}`` and its reported variance is

```math
\\widehat{\\operatorname{Var}}(\\tilde θ_j) = \\operatorname{median}_r \\big\\{
\\hat s_{r,j}^2 + (\\hat θ_{r,j} - \\tilde θ_j)^2 \\big\\},
```

where ``\\hat s_{r,j}`` is the standard error of repetition ``r``. The split-to-split
dispersion term is added on the scale of ``\\operatorname{Var}(\\hat θ)``. In the rule
of Chernozhukov et al. (2018, Section 3.4), as implemented in DoubleML (Bach et al.
2022, 2024), the same term is added to the ``\\sqrt{n}``-scaled asymptotic variance,
so that it contributes ``(\\hat θ_{r,j} - \\tilde θ_j)^2 / n`` to the variance of
``\\tilde θ_j``. For the same splits DrSnow's standard errors are therefore at least
as large as DoubleML's, and more conservative whenever the estimates vary
materially across splits. Covariances between coefficients use the average over
repetitions of the per-repetition correlation matrices, rescaled by the aggregated
standard errors, which keeps the matrix positive semi-definite. With ``R = 1`` the
covariance of the single split is returned.

Supports `coef`, `vcov`, `stderror`, `confint(r; level)`, `coeftable`, `nobs`,
`dof_residual` (``G - 1`` with clustering, so that intervals use a ``t_{G-1}``
reference; otherwise `Inf`), `estimand`, `method_name`,
[`simultaneous_confint`](@ref) and [`nuisance_loss`](@ref).

# Fields
- `model::Symbol`: model, one of `:plr`, `:irm`, `:pliv`, `:iivm`, `:did` (panel)
  or `:did_cs` (repeated cross-sections).
- `score::Symbol`: score (`:partialling_out`, `:iv_type`, `:ATE`, `:ATTE`, `:LATE`,
  `:ATT`).
- `names::Vector{String}`: coefficient names (the treatment columns).
- `coef::Vector{Float64}`, `vcov::Matrix{Float64}`: aggregated estimates and their
  covariance (see above).
- `all_coef::Matrix{Float64}`, `all_se::Matrix{Float64}`: `T × R` per-repetition
  estimates and standard errors (`T` coefficients, `R` repetitions).
- `all_vcov::Vector{Matrix{Float64}}`: the `T × T` covariance of each repetition.
- `psi_a::Array{Float64,3}`, `psi_b::Array{Float64,3}`, `psi::Array{Float64,3}`:
  `n × R × T` score components and the score evaluated at each repetition's
  estimate (``ψ = θ ψ_a + ψ_b``).
- `predictions::Dict{Symbol,Array{Float64,3}}`: out-of-fold nuisance predictions
  (`n × R × T`; propensities after clipping), keyed by nuisance name (`:ml_l`,
  `:ml_m`, `:ml_g0`, ...).
- `losses::Dict{Symbol,Matrix{Float64}}`: out-of-fold RMSE (conditional means) or
  log loss (probabilities, before clipping), `R × T` per nuisance.
- `folds::Matrix{Int}`: fold ids, `n × R`.
- `cluster::Union{Nothing,Vector{Int}}`, `n_clusters::Int`: cluster index of every
  observation (or `nothing`) and the number of clusters ``G`` (0 without
  clustering).
- `n::Int`: number of observations (units for panel [`dml_did`](@ref)).
- `learners::Vector{Pair{Symbol,String}}`: the learner used for each nuisance.
- `trim::Float64`, `n_trimmed::Matrix{Int}`: propensity clipping threshold and the
  number of clipped predictions, `R × T`.
- `seeds::Vector{UInt64}`: per-task seeds drawn from `rng`.
- `estimand::String`, `method::String`: descriptions returned by `estimand` and
  `method_name`.

# References
- Bach, P., Chernozhukov, V., Kurz, M. S., & Spindler, M. (2022). DoubleML – An
  object-oriented implementation of double machine learning in Python. *Journal of
  Machine Learning Research*, 23(53), 1–6.
- Bach, P., Kurz, M. S., Chernozhukov, V., Spindler, M., & Klaassen, S. (2024).
  DoubleML: An object-oriented implementation of double machine learning in R.
  *Journal of Statistical Software*, 108(3), 1–56.
- Chernozhukov, V., Chetverikov, D., Demirer, M., Duflo, E., Hansen, C., Newey, W., &
  Robins, J. (2018). Double/debiased machine learning for treatment and structural
  parameters. *The Econometrics Journal*, 21(1), C1–C68.
"""
struct DMLEstimate <: CausalEstimate
    model::Symbol
    score::Symbol
    names::Vector{String}
    coef::Vector{Float64}
    vcov::Matrix{Float64}
    all_coef::Matrix{Float64}
    all_se::Matrix{Float64}
    all_vcov::Vector{Matrix{Float64}}
    psi_a::Array{Float64,3}
    psi_b::Array{Float64,3}
    psi::Array{Float64,3}
    predictions::Dict{Symbol,Array{Float64,3}}
    losses::Dict{Symbol,Matrix{Float64}}
    folds::Matrix{Int}
    cluster::Union{Nothing,Vector{Int}}
    n_clusters::Int
    n::Int
    learners::Vector{Pair{Symbol,String}}
    trim::Float64
    n_trimmed::Matrix{Int}
    seeds::Vector{UInt64}
    estimand::String
    method::String
end

StatsAPI.coef(r::DMLEstimate) = r.coef
StatsAPI.vcov(r::DMLEstimate) = r.vcov
StatsAPI.coefnames(r::DMLEstimate) = r.names
StatsAPI.nobs(r::DMLEstimate) = r.n
StatsAPI.dof_residual(r::DMLEstimate) = r.cluster === nothing ? Inf : r.n_clusters - 1.0
estimand(r::DMLEstimate) = r.estimand
method_name(r::DMLEstimate) = r.method

function show_details(io::IO, r::DMLEstimate)
    K = maximum(r.folds)
    R = size(r.folds, 2)
    println(io)
    print(io, "Cross-fitting: $K folds × $R repetition", R == 1 ? "" : "s")
    R > 1 && print(io, " (median aggregation)")
    println(io)
    r.cluster === nothing ||
        println(io, "Clusters: $(r.n_clusters) (folds grouped by cluster; ",
                "t($(r.n_clusters - 1)))")
    for (nm, l) in r.learners
        loss = get(r.losses, nm, nothing)
        if loss === nothing
            println(io, "  ", nm, ": ", l)
        else
            measure = haskey(r.predictions, nm) && _ml_is_proba_nuisance(r, nm) ?
                      "log loss" : "RMSE"
            @printf(io, "  %s: %s (out-of-fold %s %.4g)\n", nm, l, measure, mean(loss))
        end
    end
    if r.trim > 0 && any(>(0), r.n_trimmed)
        @printf(io, "Propensity clipping at [%.3g, %.3g]: %d prediction(s) clipped",
                r.trim, 1 - r.trim, sum(r.n_trimmed))
        println(io, " (total over repetitions)")
    end
    return nothing
end

const _ML_PROBA_NUISANCES = (:ml_m_prop, :ml_m, :ml_r0, :ml_r1)

function _ml_is_proba_nuisance(r::DMLEstimate, nm)
    r.model in (:irm, :iivm, :did, :did_cs) && nm in _ML_PROBA_NUISANCES && return true
    return false
end

"""
    nuisance_loss(r::DMLEstimate) -> DataFrame

Out-of-fold prediction quality of every nuisance function of a DML fit.

The validity of DML inference rests on the nuisance functions being estimated well
enough: the bias of the orthogonal score is of the order of the product of the
nuisance estimation errors, which must vanish faster than ``n^{-1/2}``
(Chernozhukov et al. 2018). Out-of-fold losses are the honest measure of that
accuracy, because each prediction comes from a learner that did not see the
observation. The table reports the root mean squared error for conditional means,
computed on the subsample on which the nuisance is defined (for example, the controls
for `ml_g0`), and the log loss for probabilities (propensity scores and compliance
probabilities, evaluated before clipping). Values are averaged over repetitions.

Use the table to compare candidate learners on the same folds (pass `folds`
explicitly or reuse `rng` seeds) and report it alongside the estimate. A loss close
to that of [`MeanLearner`](@ref) indicates that the covariates carry little
predictive information for that nuisance, which is not in itself a problem; a loss
that changes markedly across learners warns that the estimate may be sensitive to
the choice of learner.

# Arguments
- `r::DMLEstimate`: a fitted DML estimate.

# Returns
- `DataFrame` with columns `nuisance`, `treatment`, `learner`, `measure`
  (`"RMSE"` or `"log loss"`) and `value`, one row per nuisance and treatment.

# Examples
```julia
using DrSnow, DataFrames, StableRNGs
rng = StableRNG(11)
n = 400
x1, x2 = randn(rng, n), randn(rng, n)
d = Float64.(rand(rng, n) .< 1 ./ (1 .+ exp.(-0.5 .* x1)))
y = d .+ x1 .+ 0.5 .* x2 .+ randn(rng, n)
df = DataFrame(; y, d, x1, x2)
r = dml_irm(df, :y, :d; covariates=[:x1, :x2], n_folds=3, rng=StableRNG(12))
nuisance_loss(r)
```

# References
- Chernozhukov, V., Chetverikov, D., Demirer, M., Duflo, E., Hansen, C., Newey, W., &
  Robins, J. (2018). Double/debiased machine learning for treatment and structural
  parameters. *The Econometrics Journal*, 21(1), C1–C68.
- Bach, P., Kurz, M. S., Chernozhukov, V., Spindler, M., & Klaassen, S. (2024).
  DoubleML: An object-oriented implementation of double machine learning in R.
  *Journal of Statistical Software*, 108(3), 1–56.
"""
function nuisance_loss(r::DMLEstimate)
    lnames = Dict(r.learners)
    rows = NamedTuple[]
    for (nm, L) in sort!(collect(r.losses); by=first)
        for t in axes(L, 2)
            push!(rows, (nuisance=string(nm), treatment=r.names[t],
                         learner=get(lnames, nm, ""),
                         measure=_ml_is_proba_nuisance(r, nm) ? "log loss" : "RMSE",
                         value=mean(L[:, t])))
        end
    end
    return DataFrame(rows)
end

# ----------------------------------------------------------------- generic driver

_ml_rmse(y, f) = sqrt(mean((y .- f) .^ 2))

function _ml_logloss(y, p)
    s = 0.0
    for i in eachindex(y)
        q = clamp(p[i], 1e-15, 1 - 1e-15)
        s -= y[i] * log(q) + (1 - y[i]) * log(1 - q)
    end
    return s / length(y)
end

"""
Run DML over repetitions and treatments. `rep_fn(r, t)` returns a NamedTuple with
`psi_a`, `psi_b`, `preds` (Vector of Symbol => Vector), `losses` (Vector of
Symbol => Float64) and `ntrim`.
"""
function _ml_run_dml(rep_fn, T::Int, F::Matrix{Int}, cluster, G)
    n, R = size(F)
    psi_a = zeros(n, R, T)
    psi_b = zeros(n, R, T)
    psi = zeros(n, R, T)
    preds = Dict{Symbol,Array{Float64,3}}()
    losses = Dict{Symbol,Matrix{Float64}}()
    ntrim = zeros(Int, R, T)
    all_coef = zeros(T, R)
    all_se = zeros(T, R)
    all_vcov = Vector{Matrix{Float64}}(undef, R)
    for r in 1:R
        Ψ = zeros(n, T)
        J = zeros(T)
        for t in 1:T
            res = rep_fn(r, t)
            θ, ψ, v = _ml_solve_score(res.psi_a, res.psi_b, cluster, G)
            psi_a[:, r, t] .= res.psi_a
            psi_b[:, r, t] .= res.psi_b
            psi[:, r, t] .= ψ
            Ψ[:, t] .= ψ
            J[t] = mean(res.psi_a)
            all_coef[t, r] = θ
            all_se[t, r] = sqrt(v)
            ntrim[r, t] = res.ntrim
            for (nm, p) in res.preds
                A = get!(preds, nm) do
                    fill(NaN, n, R, T)
                end
                A[:, r, t] .= p
            end
            for (nm, l) in res.losses
                L = get!(losses, nm) do
                    fill(NaN, R, T)
                end
                L[r, t] = l
            end
        end
        all_vcov[r] = _ml_score_cov(Ψ, J, cluster, G)
    end
    θ, V = _ml_aggregate(all_coef, all_vcov)
    return (; θ, V, all_coef, all_se, all_vcov, psi_a, psi_b, psi, preds, losses, ntrim)
end

function _ml_make_estimate(out, model, score, names, F, cluster, G, n, learners, trim,
                           seeds, est, method)
    return DMLEstimate(model, score, names, out.θ, out.V, out.all_coef, out.all_se,
                       out.all_vcov, out.psi_a, out.psi_b, out.psi, out.preds,
                       out.losses, F, cluster, G, n, learners, trim, out.ntrim,
                       vec(seeds), est, method)
end

"""Resolve covariates / cluster / folds shared by all DML front ends."""
function _ml_setup(data, cols, covariates, cluster, folds, n_folds, n_rep, rng, strata;
                   context)
    covs = Symbol.(collect(covariates))
    cl = cluster === nothing ? Symbol[] :
         (cluster isa AbstractVector ? Symbol.(cluster) : [Symbol(cluster)])
    fcol = folds isa Symbol ? [folds] : Symbol[]
    require_columns(data, vcat(cols, covs, cl, fcol); context=context)
    _ml_check_common(n_folds, n_rep)
    n = nrow(data)
    n >= 2 || throw(ArgumentError("$(context): need at least two observations"))
    X = _ml_matrix(data, covs; context=context)
    cid, G = _ml_cluster_ids(data, cluster; context=context)
    F = _ml_resolve_folds(data, folds, n, n_folds, n_rep, rng, strata, cid;
                          context=context)
    return covs, X, cid, G, F
end

# ------------------------------------------------------------------------- PLR

"""
    dml_plr(data, outcome, treatment; covariates=Symbol[],
            outcome_learner=LassoLearner(), treatment_learner=LassoLearner(),
            score=:partialling_out, g_learner=outcome_learner, n_folds=5, n_rep=1,
            folds=nothing, cluster=nothing, rng=Random.default_rng(),
            parallel=true) -> DMLEstimate

Double/debiased machine-learning estimator of the coefficient of a treatment in the
partially linear regression model.

The model is Robinson's (1988) partially linear regression

```math
Y = θ_0 D + g_0(X) + ε, \\quad E[ε \\mid D, X] = 0, \\qquad
D = m_0(X) + V, \\quad E[V \\mid X] = 0,
```

in which the controls ``X`` enter through an unknown, possibly high-dimensional
function ``g_0``. Under conditional exogeneity of ``D`` given ``X`` (selection on
observables) and a constant effect, ``θ_0`` is the causal effect of a unit change in
``D``. With heterogeneous effects of a binary treatment it is a weighted average of
the conditional effects ``τ(x)`` with weights proportional to
``\\operatorname{Var}(D \\mid X = x)``, not the ATE; use [`dml_irm`](@ref) for the
ATE or ATT. Conditional exogeneity cannot be tested from the data; the linearity of
the effect in ``D`` can be probed only by comparing specifications.

With `score = :partialling_out` the estimator is the residual-on-residual regression
of Robinson (1988) with machine-learned conditional means
``ℓ_0(X) = E[Y \\mid X]`` and ``m_0(X) = E[D \\mid X]``, based on the Neyman-orthogonal
score

```math
ψ(W; θ, η) = \\{Y - ℓ(X) - θ (D - m(X))\\}\\{D - m(X)\\}.
```

With `score = :iv_type` the score is ``ψ = \\{Y - θ D - g(X)\\}\\{D - m(X)\\}``, where
``g`` is learned from ``Y - \\hat θ_0 D`` after a preliminary partialling-out estimate
``\\hat θ_0`` (as in DoubleML). Orthogonality makes the estimating equation
insensitive to first-order errors in the nuisances, so the remaining bias is of the
order of ``\\lVert \\hat m - m_0 \\rVert \\cdot \\lVert \\hat ℓ - ℓ_0 \\rVert``;
``\\sqrt{n}`` inference holds when this product is ``o(n^{-1/2})``, for instance when both
learners converge faster than ``n^{-1/4}`` (Chernozhukov et al. 2018). The nuisances
are cross-fitted over `n_folds` folds and ``θ`` solves the pooled empirical moment
(the DML2 estimator). With a lasso for both nuisances, the estimator is closely
related to the post-double-selection estimator of Belloni, Chernozhukov and Hansen
(2014).

The variance is the sandwich ``\\hat J^{-2} n^{-2} \\sum_i \\hat ψ_i^2`` with
``\\hat J = n^{-1} \\sum_i ψ_{a,i}``; with `cluster`, scores are summed within
clusters, multiplied by ``G/(G-1)``, the folds are formed from whole clusters, and
intervals use a ``t_{G-1}`` reference. Several treatments may be given as a vector:
each coefficient is estimated separately with the other treatments added to the
controls (DoubleML's multiple-treatment design), and the joint covariance is formed
from the scores, which [`simultaneous_confint`](@ref) uses for simultaneous
inference. Results depend on the random sample split; setting `n_rep > 1` aggregates
over splits (see [`DMLEstimate`](@ref) for the aggregation rule). Report the
learners, the number of folds and repetitions, and [`nuisance_loss`](@ref).

# Arguments
- `data`: a `DataFrame` (or table convertible to one) with one row per
  observation.
- `outcome::Symbol`: the numeric outcome column ``Y``.
- `treatment::Union{Symbol,Vector{Symbol}}`: the numeric treatment column(s)
  ``D``; binary or continuous. With several treatments, one coefficient is
  estimated for each.

# Keywords
- `covariates::Vector{Symbol} = Symbol[]`: the controls ``X``; they must be numeric
  (dummy-encode categorical variables beforehand).
- `outcome_learner = LassoLearner()`: the [`NuisanceLearner`](@ref) for
  ``ℓ_0(X) = E[Y \\mid X]``.
- `treatment_learner = LassoLearner()`: the learner for ``m_0(X) = E[D \\mid X]``.
- `score::Symbol = :partialling_out`: `:partialling_out` or `:iv_type`; the two
  are asymptotically equivalent under correct specification.
- `g_learner = outcome_learner`: the learner for ``g_0`` in the IV-type score
  (ignored with `:partialling_out`).
- `n_folds::Integer = 5`: number of cross-fitting folds (at least 2). More folds
  train the learners on more data at a higher computational cost.
- `n_rep::Integer = 1`: number of independent repetitions of the sample split.
- `folds = nothing`: user-supplied fold ids (a vector, an `n × n_rep` matrix or a
  column name); overrides `n_folds` and `n_rep`. See [`crossfit_folds`](@ref).
- `cluster::Union{Nothing,Symbol} = nothing`: cluster column; folds are drawn at the
  cluster level and the variance is cluster-robust.
- `rng::AbstractRNG = Random.default_rng()`: draws the folds and then one seed per
  (repetition, nuisance, fold) task, so results do not depend on `parallel` or the
  number of threads.
- `parallel::Bool = true`: fit the fold × nuisance tasks on threads.

# Returns
- [`DMLEstimate`](@ref) with `model = :plr`.

# Examples
```julia
using DrSnow, DataFrames, StableRNGs
rng = StableRNG(1)
n = 500
x1, x2, x3 = randn(rng, n), randn(rng, n), randn(rng, n)
d = 0.5 .* x1 .- 0.3 .* x2 .+ randn(rng, n)
y = 1.0 .* d .+ sin.(x1) .+ 0.5 .* x3 .+ randn(rng, n)
df = DataFrame(; y, d, x1, x2, x3)
r = dml_plr(df, :y, :d; covariates=[:x1, :x2, :x3], n_folds=5, n_rep=3,
            rng=StableRNG(2))
coeftable(r)
```

# References
- Robinson, P. M. (1988). Root-N-consistent semiparametric regression.
  *Econometrica*, 56(4), 931–954.
- Chernozhukov, V., Chetverikov, D., Demirer, M., Duflo, E., Hansen, C., Newey, W., &
  Robins, J. (2018). Double/debiased machine learning for treatment and structural
  parameters. *The Econometrics Journal*, 21(1), C1–C68.
- Chernozhukov, V., Escanciano, J. C., Ichimura, H., Newey, W. K., & Robins, J. M.
  (2022). Locally robust semiparametric estimation. *Econometrica*, 90(4),
  1501–1535.
- Belloni, A., Chernozhukov, V., & Hansen, C. (2014). Inference on treatment effects
  after selection among high-dimensional controls. *The Review of Economic Studies*,
  81(2), 608–650.
- Bach, P., Kurz, M. S., Chernozhukov, V., Spindler, M., & Klaassen, S. (2024).
  DoubleML: An object-oriented implementation of double machine learning in R.
  *Journal of Statistical Software*, 108(3), 1–56.
"""
function dml_plr(data, outcome::Symbol, treatment;
                 covariates=Symbol[], outcome_learner=LassoLearner(),
                 treatment_learner=LassoLearner(), score::Symbol=:partialling_out,
                 g_learner=outcome_learner, n_folds::Integer=5, n_rep::Integer=1,
                 folds=nothing, cluster=nothing, rng::AbstractRNG=Random.default_rng(),
                 parallel::Bool=true)
    ctx = "dml_plr"
    score in (:partialling_out, :iv_type) ||
        throw(ArgumentError("$ctx: score must be :partialling_out or :iv_type"))
    ds = treatment isa Symbol ? [treatment] : Symbol.(collect(treatment))
    isempty(ds) && throw(ArgumentError("$ctx: at least one treatment is required"))
    covs, X, cid, G, F = _ml_setup(data, vcat(outcome, ds), covariates, cluster, folds,
                                   n_folds, n_rep, rng, nothing; context=ctx)
    y = _ml_column(data, outcome; context=ctx)
    D = _ml_matrix(data, ds; context=ctx)
    T = length(ds)
    K = maximum(F)
    R = size(F, 2)
    seeds = _ml_seeds(rng, K, 3 * T, R)
    rep_fn = function (r, t)
        Xt = hcat(X, D[:, setdiff(1:T, t)])
        d = D[:, t]
        specs = [_MLNuisance(:ml_l, outcome_learner, y, Xt, false),
                 _MLNuisance(:ml_m, treatment_learner, d, Xt, false)]
        P = _ml_crossfit(specs, F[:, r], view(seeds, :, (3t - 2):(3t - 1), r);
                         parallel=parallel, context=ctx)
        l̂, m̂ = P[:, 1], P[:, 2]
        v = d .- m̂
        preds = [:ml_l => l̂, :ml_m => m̂]
        losses = [:ml_l => _ml_rmse(y, l̂), :ml_m => _ml_rmse(d, m̂)]
        if score === :partialling_out
            return (psi_a=-(v .* v), psi_b=v .* (y .- l̂), preds=preds, losses=losses,
                    ntrim=0)
        end
        θ0 = sum(v .* (y .- l̂)) / sum(v .* v)
        spec_g = [_MLNuisance(:ml_g, g_learner, y .- θ0 .* d, Xt, false)]
        ĝ = _ml_crossfit(spec_g, F[:, r], view(seeds, :, (3t):(3t), r);
                         parallel=parallel, context=ctx)[:, 1]
        push!(preds, :ml_g => ĝ)
        push!(losses, :ml_g => _ml_rmse(y .- θ0 .* d, ĝ))
        return (psi_a=-(v .* d), psi_b=v .* (y .- ĝ), preds=preds, losses=losses, ntrim=0)
    end
    out = _ml_run_dml(rep_fn, T, F, cid, G)
    learners = [:ml_l => _ml_learner_name(outcome_learner),
                :ml_m => _ml_learner_name(treatment_learner)]
    score === :iv_type && push!(learners, :ml_g => _ml_learner_name(g_learner))
    sname = score === :partialling_out ? "partialling out" : "IV-type"
    return _ml_make_estimate(out, :plr, score, string.(ds), F, cid, G, nrow(data),
                             learners, 0.0, seeds,
                             "θ in Y = θD + g(X) + ε (partially linear)",
                             "DML partially linear regression ($sname score)")
end

# ------------------------------------------------------------------------- IRM

"""
    dml_irm(data, outcome, treatment; covariates=Symbol[],
            outcome_learner=LassoLearner(), propensity_learner=PenalizedLogisticLearner(),
            score=:ATE, trim=0.01, stratify=true, n_folds=5, n_rep=1, folds=nothing,
            cluster=nothing, rng=Random.default_rng(), parallel=true) -> DMLEstimate

Double/debiased machine-learning (augmented inverse-probability-weighting) estimator
of the average treatment effect, or of the average treatment effect on the treated,
of a binary treatment.

The interactive regression model leaves the outcome equation fully nonparametric,
``Y = g_0(D, X) + U`` with ``E[U \\mid D, X] = 0``, and ``D = m_0(X) + V`` with
propensity score ``m_0(X) = P(D = 1 \\mid X)``. In potential-outcome notation the
targets are

```math
θ_{\\mathrm{ATE}} = E[Y(1) - Y(0)], \\qquad
θ_{\\mathrm{ATT}} = E[Y(1) - Y(0) \\mid D = 1].
```

They are identified under unconfoundedness, ``\\{Y(0), Y(1)\\} ⊥ D \\mid X``, and
overlap, ``0 < m_0(X) < 1`` (for the ATT only ``m_0(X) < 1`` is needed), together
with no interference between units. Unconfoundedness cannot be tested; overlap can
be inspected through the estimated propensities (`r.predictions[:ml_m]`).

With ``g_d(X) = E[Y \\mid D = d, X]``, fitted on the treated and the control
observations of the training folds separately, the ATE score is the augmented
inverse-probability-weighting score of Robins, Rotnitzky and Zhao (1994), which is
the efficient influence function of the ATE (Hahn 1998):

```math
ψ(W; θ, η) = g_1(X) - g_0(X) + \\frac{D\\{Y - g_1(X)\\}}{m(X)}
- \\frac{(1 - D)\\{Y - g_0(X)\\}}{1 - m(X)} - θ .
```

For the ATT (`score = :ATTE`) the score is
``ψ = p^{-1}[D\\{Y - g_0(X)\\} - m(X)(1 - D)\\{Y - g_0(X)\\}/\\{1 - m(X)\\} - D θ]``, where
``p`` is the treated share of the fold that contains the observation. Both scores
are doubly robust (consistent if either the outcome regressions or the propensity
score are consistently estimated) and Neyman orthogonal, so ``\\sqrt{n}`` inference
holds when the product of the outcome-regression and propensity errors is
``o(n^{-1/2})`` (Chernozhukov et al. 2018, Section 5.1). Inference and aggregation
over repetitions are as in [`dml_plr`](@ref) and [`DMLEstimate`](@ref).

Estimated propensities are clipped to ``[\\mathrm{trim}, 1 - \\mathrm{trim}]`` and
the number of clipped predictions is stored in `n_trimmed` and printed. Clipping
keeps the inverse weights finite, but it does not repair limited overlap: the
estimand is still the full-population ATE (or ATT), the clipped weights introduce a
bias of unknown sign, and the normal approximation can be poor when a few
observations carry very large weights. When many predictions are clipped, consider
changing the estimand explicitly: trimming the sample to units with propensities
inside ``[0.1, 0.9]`` targets the effect in that subpopulation (Crump, Hotz, Imbens
and Mitnik 2009), and overlap weights ``m(X)\\{1 - m(X)\\}`` target the
overlap-weighted effect (Li, Morgan and Zaslavsky 2018), available as
`target = :overlap` in [`average_treatment_effect`](@ref) for a
[`causal_forest`](@ref). Report the propensity distribution with the estimate.

# Arguments
- `data`: a `DataFrame` with one row per observation.
- `outcome::Symbol`: the numeric outcome column.
- `treatment::Symbol`: the binary treatment column, coded 0/1.

# Keywords
- `covariates::Vector{Symbol} = Symbol[]`: numeric confounders ``X``.
- `outcome_learner = LassoLearner()`: the [`NuisanceLearner`](@ref) used for both
  ``g_0`` and ``g_1``.
- `propensity_learner = PenalizedLogisticLearner()`: the learner for ``m_0``; it must
  implement [`fitpredict_proba`](@ref).
- `score::Symbol = :ATE`: `:ATE` or `:ATTE` (the ATT).
- `trim::Real = 0.01`: propensity clipping threshold in ``[0, 0.5)``; `0` disables
  clipping.
- `stratify::Bool = true`: draw folds stratified by treatment, so every fold
  contains treated and control units (ignored when `folds` is given).
- `n_folds::Integer = 5`, `n_rep::Integer = 1`, `folds = nothing`,
  `cluster = nothing`, `rng = Random.default_rng()`, `parallel::Bool = true`: as in
  [`dml_plr`](@ref).

# Returns
- [`DMLEstimate`](@ref) with `model = :irm`; the nuisance predictions are stored
  under `:ml_g0`, `:ml_g1` and `:ml_m`.

# Examples
```julia
using DrSnow, DataFrames, StableRNGs
rng = StableRNG(3)
n = 800
x1, x2 = randn(rng, n), randn(rng, n)
d = Float64.(rand(rng, n) .< 1 ./ (1 .+ exp.(-(0.5 .* x1 .- 0.5 .* x2))))
y = (1 .+ 0.5 .* x1) .* d .+ x1 .+ x2 .+ randn(rng, n)   # ATE = 1
df = DataFrame(; y, d, x1, x2)
r = dml_irm(df, :y, :d; covariates=[:x1, :x2], n_folds=5, n_rep=3,
            rng=StableRNG(4))
r_att = dml_irm(df, :y, :d; covariates=[:x1, :x2], score=:ATTE, rng=StableRNG(4))
```

# References
- Robins, J. M., Rotnitzky, A., & Zhao, L. P. (1994). Estimation of regression
  coefficients when some regressors are not always observed. *Journal of the
  American Statistical Association*, 89(427), 846–866.
- Hahn, J. (1998). On the role of the propensity score in efficient semiparametric
  estimation of average treatment effects. *Econometrica*, 66(2), 315–331.
- Chernozhukov, V., Chetverikov, D., Demirer, M., Duflo, E., Hansen, C., Newey, W., &
  Robins, J. (2018). Double/debiased machine learning for treatment and structural
  parameters. *The Econometrics Journal*, 21(1), C1–C68.
- Crump, R. K., Hotz, V. J., Imbens, G. W., & Mitnik, O. A. (2009). Dealing with
  limited overlap in estimation of average treatment effects. *Biometrika*, 96(1),
  187–199.
- Li, F., Morgan, K. L., & Zaslavsky, A. M. (2018). Balancing covariates via
  propensity score weighting. *Journal of the American Statistical Association*,
  113(521), 390–400.
- Bach, P., Kurz, M. S., Chernozhukov, V., Spindler, M., & Klaassen, S. (2024).
  DoubleML: An object-oriented implementation of double machine learning in R.
  *Journal of Statistical Software*, 108(3), 1–56.
"""
function dml_irm(data, outcome::Symbol, treatment::Symbol; covariates=Symbol[],
                 outcome_learner=LassoLearner(),
                 propensity_learner=PenalizedLogisticLearner(), score::Symbol=:ATE,
                 trim::Real=0.01, stratify::Bool=true, n_folds::Integer=5,
                 n_rep::Integer=1, folds=nothing, cluster=nothing,
                 rng::AbstractRNG=Random.default_rng(), parallel::Bool=true)
    ctx = "dml_irm"
    score in (:ATE, :ATTE) || throw(ArgumentError("$ctx: score must be :ATE or :ATTE"))
    trim = _ml_check_trim(trim)
    require_columns(data, [treatment]; context=ctx)
    d = _ml_column(data, treatment; context=ctx)
    _ml_check_binary_col(d, treatment, ctx)
    covs, X, cid, G, F = _ml_setup(data, [outcome, treatment], covariates, cluster,
                                   folds, n_folds, n_rep, rng, stratify ? d : nothing;
                                   context=ctx)
    y = _ml_column(data, outcome; context=ctx)
    K = maximum(F)
    R = size(F, 2)
    seeds = _ml_seeds(rng, K, 3, R)
    treated = d .== 1
    rep_fn = function (r, _)
        specs = [_MLNuisance(:ml_g0, outcome_learner, y, X, false, .!treated),
                 _MLNuisance(:ml_g1, outcome_learner, y, X, false, BitVector(treated)),
                 _MLNuisance(:ml_m, propensity_learner, d, X, true)]
        P = _ml_crossfit(specs, F[:, r], view(seeds, :, :, r); parallel=parallel,
                         context=ctx)
        g0, g1, m = P[:, 1], P[:, 2], P[:, 3]
        losses = [:ml_g0 => _ml_rmse(y[.!treated], g0[.!treated]),
                  :ml_g1 => _ml_rmse(y[treated], g1[treated]),
                  :ml_m => _ml_logloss(d, m)]
        nt = _ml_clip!(m, trim)
        preds = [:ml_g0 => g0, :ml_g1 => g1, :ml_m => m]
        if score === :ATE
            psi_b = g1 .- g0 .+ d .* (y .- g1) ./ m .- (1 .- d) .* (y .- g0) ./ (1 .- m)
            psi_a = fill(-1.0, length(y))
        else
            p = zeros(length(y))
            for k in 1:K
                idx = F[:, r] .== k
                p[idx] .= mean(d[idx])
            end
            any(==(0), p) && throw(ArgumentError("$ctx: a fold contains no treated " *
                                                 "units; ATTE is not identified there"))
            u0 = y .- g0
            psi_b = d .* u0 ./ p .- m .* (1 .- d) .* u0 ./ (p .* (1 .- m))
            psi_a = -d ./ p
        end
        return (psi_a=psi_a, psi_b=psi_b, preds=preds, losses=losses, ntrim=nt)
    end
    out = _ml_run_dml(rep_fn, 1, F, cid, G)
    learners = [:ml_g0 => _ml_learner_name(outcome_learner),
                :ml_g1 => _ml_learner_name(outcome_learner),
                :ml_m => _ml_learner_name(propensity_learner)]
    est = score === :ATE ? "ATE" : "ATT (ATTE)"
    return _ml_make_estimate(out, :irm, score, [string(treatment)], F, cid, G,
                             nrow(data), learners, trim, seeds, est,
                             "DML interactive regression model (AIPW, $(score))")
end

# ------------------------------------------------------------------------ PLIV

"""
    dml_pliv(data, outcome, treatment, instrument; covariates=Symbol[],
             outcome_learner=LassoLearner(), treatment_learner=LassoLearner(),
             instrument_learner=LassoLearner(), score=:partialling_out,
             g_learner=outcome_learner, n_folds=5, n_rep=1, folds=nothing,
             cluster=nothing, rng=Random.default_rng(), parallel=true) -> DMLEstimate

Double/debiased machine-learning estimator of the coefficient of an endogenous
treatment in the partially linear instrumental-variables model.

The model is

```math
Y = θ_0 D + g_0(X) + ε, \\quad E[ε \\mid Z, X] = 0, \\qquad
Z = m_0(X) + V, \\quad E[V \\mid X] = 0,
```

where the treatment ``D`` may be correlated with ``ε`` and the instrument(s) ``Z``
are valid only conditionally on the controls ``X``, which enter through unknown
functions. Identification requires conditional exogeneity and exclusion
(``E[ε \\mid Z, X] = 0``) and relevance (``Z`` predicts ``D`` given ``X``). Exclusion
is not testable; relevance is, through the residualized first stage. With a constant
effect ``θ_0`` is the causal effect of ``D``; with heterogeneous effects it is a
weighted average whose weights depend on the first stage. For a binary instrument and
a binary treatment, [`dml_iivm`](@ref) targets the local average treatment effect
directly.

With `score = :partialling_out` and one instrument, the Neyman-orthogonal score is

```math
ψ(W; θ, η) = \\{Y - ℓ(X) - θ (D - r(X))\\}\\{Z - m(X)\\},
```

with ``ℓ_0 = E[Y \\mid X]``, ``r_0 = E[D \\mid X]`` and ``m_0 = E[Z \\mid X]`` learned
with cross-fitting (Chernozhukov et al. 2018, Section 4.2). With several instruments,
the residualized treatment is projected by OLS (with intercept, in sample) on the
residualized instruments and the fitted value is used as the single optimal
instrument, as in DoubleML. `score = :iv_type` (one instrument only) uses
``ψ = \\{Y - θ D - g(X)\\}\\{Z - m(X)\\}``, with ``g`` learned from ``Y - \\hat θ_0 D``
after a preliminary partialling-out estimate.

Inference uses the sandwich variance of the linear score, as in [`dml_plr`](@ref),
and is reliable only when the instrument is strong. Cross-fitting and orthogonality
do not protect against weak instruments: when the residualized first stage is weak,
the normal approximation fails and the estimate is biased towards the confounded
association. Inspect the first stage (for example, regress ``D - \\hat r(X)`` on
``Z - \\hat m(X)``) and report it. See [`DMLEstimate`](@ref) for aggregation over
repetitions.

# Arguments
- `data`: a `DataFrame` with one row per observation.
- `outcome::Symbol`: the numeric outcome column.
- `treatment::Symbol`: the numeric endogenous treatment column.
- `instrument::Union{Symbol,Vector{Symbol}}`: one or more numeric instrument
  columns.

# Keywords
- `covariates::Vector{Symbol} = Symbol[]`: numeric controls ``X``.
- `outcome_learner = LassoLearner()`: the learner for ``ℓ_0 = E[Y \\mid X]``.
- `treatment_learner = LassoLearner()`: the learner for ``r_0 = E[D \\mid X]``.
- `instrument_learner = LassoLearner()`: the learner for ``m_0 = E[Z \\mid X]``
  (fitted separately for each instrument).
- `score::Symbol = :partialling_out`: `:partialling_out` or `:iv_type` (the latter
  requires exactly one instrument).
- `g_learner = outcome_learner`: the learner for ``g_0`` in the IV-type score.
- `n_folds::Integer = 5`, `n_rep::Integer = 1`, `folds = nothing`,
  `cluster = nothing`, `rng = Random.default_rng()`, `parallel::Bool = true`: as in
  [`dml_plr`](@ref).

# Returns
- [`DMLEstimate`](@ref) with `model = :pliv`.

# Examples
```julia
using DrSnow, DataFrames, StableRNGs
rng = StableRNG(5)
n = 800
x1, x2, u = randn(rng, n), randn(rng, n), randn(rng, n)
z = 0.5 .* x1 .+ randn(rng, n)
d = 0.8 .* z .+ 0.5 .* x2 .+ u .+ randn(rng, n)
y = 1.0 .* d .+ x1 .+ x2 .- u .+ randn(rng, n)     # θ₀ = 1, D endogenous
df = DataFrame(; y, d, z, x1, x2)
r = dml_pliv(df, :y, :d, :z; covariates=[:x1, :x2], rng=StableRNG(6))
```

# References
- Chernozhukov, V., Chetverikov, D., Demirer, M., Duflo, E., Hansen, C., Newey, W., &
  Robins, J. (2018). Double/debiased machine learning for treatment and structural
  parameters. *The Econometrics Journal*, 21(1), C1–C68.
- Robinson, P. M. (1988). Root-N-consistent semiparametric regression.
  *Econometrica*, 56(4), 931–954.
- Chernozhukov, V., Escanciano, J. C., Ichimura, H., Newey, W. K., & Robins, J. M.
  (2022). Locally robust semiparametric estimation. *Econometrica*, 90(4),
  1501–1535.
- Bach, P., Kurz, M. S., Chernozhukov, V., Spindler, M., & Klaassen, S. (2024).
  DoubleML: An object-oriented implementation of double machine learning in R.
  *Journal of Statistical Software*, 108(3), 1–56.
"""
function dml_pliv(data, outcome::Symbol, treatment::Symbol, instrument;
                  covariates=Symbol[], outcome_learner=LassoLearner(),
                  treatment_learner=LassoLearner(), instrument_learner=LassoLearner(),
                  score::Symbol=:partialling_out, g_learner=outcome_learner,
                  n_folds::Integer=5, n_rep::Integer=1, folds=nothing, cluster=nothing,
                  rng::AbstractRNG=Random.default_rng(), parallel::Bool=true)
    ctx = "dml_pliv"
    score in (:partialling_out, :iv_type) ||
        throw(ArgumentError("$ctx: score must be :partialling_out or :iv_type"))
    zs = instrument isa Symbol ? [instrument] : Symbol.(collect(instrument))
    isempty(zs) && throw(ArgumentError("$ctx: at least one instrument is required"))
    (score === :iv_type && length(zs) > 1) &&
        throw(ArgumentError("$ctx: score = :iv_type requires exactly one instrument"))
    covs, X, cid, G, F = _ml_setup(data, vcat(outcome, treatment, zs), covariates,
                                   cluster, folds, n_folds, n_rep, rng, nothing;
                                   context=ctx)
    y = _ml_column(data, outcome; context=ctx)
    d = _ml_column(data, treatment; context=ctx)
    Z = _ml_matrix(data, zs; context=ctx)
    L = length(zs)
    K = maximum(F)
    R = size(F, 2)
    seeds = _ml_seeds(rng, K, 3 + L, R)
    rep_fn = function (r, _)
        specs = [_MLNuisance(:ml_l, outcome_learner, y, X, false),
                 _MLNuisance(:ml_r, treatment_learner, d, X, false)]
        for (j, z) in enumerate(zs)
            push!(specs, _MLNuisance(L == 1 ? :ml_m : Symbol("ml_m_", z),
                                     instrument_learner, Z[:, j], X, false))
        end
        P = _ml_crossfit(specs, F[:, r], view(seeds, :, 1:(2 + L), r);
                         parallel=parallel, context=ctx)
        l̂, r̂ = P[:, 1], P[:, 2]
        M̂ = P[:, 3:end]
        preds = Pair{Symbol,Vector{Float64}}[:ml_l => l̂, :ml_r => r̂]
        losses = Pair{Symbol,Float64}[:ml_l => _ml_rmse(y, l̂), :ml_r => _ml_rmse(d, r̂)]
        for j in 1:L
            push!(preds, specs[2 + j].name => M̂[:, j])
            push!(losses, specs[2 + j].name => _ml_rmse(Z[:, j], M̂[:, j]))
        end
        u = y .- l̂
        w = d .- r̂
        V = Z .- M̂
        if L == 1
            v = V[:, 1]
            if score === :partialling_out
                return (psi_a=-(w .* v), psi_b=v .* u, preds=preds, losses=losses,
                        ntrim=0)
            end
            θ0 = sum(v .* u) / sum(w .* v)
            spec_g = [_MLNuisance(:ml_g, g_learner, y .- θ0 .* d, X, false)]
            ĝ = _ml_crossfit(spec_g, F[:, r], view(seeds, :, (3 + L):(3 + L), r);
                             parallel=parallel, context=ctx)[:, 1]
            push!(preds, :ml_g => ĝ)
            push!(losses, :ml_g => _ml_rmse(y .- θ0 .* d, ĝ))
            return (psi_a=-(d .* v), psi_b=v .* (y .- ĝ), preds=preds, losses=losses,
                    ntrim=0)
        end
        A = hcat(ones(length(w)), V)
        ṽ = A * (qr(A, ColumnNorm()) \ w)
        return (psi_a=-(w .* ṽ), psi_b=ṽ .* u, preds=preds, losses=losses, ntrim=0)
    end
    out = _ml_run_dml(rep_fn, 1, F, cid, G)
    learners = [:ml_l => _ml_learner_name(outcome_learner),
                :ml_r => _ml_learner_name(treatment_learner)]
    for z in zs
        push!(learners, (L == 1 ? :ml_m : Symbol("ml_m_", z)) =>
                        _ml_learner_name(instrument_learner))
    end
    score === :iv_type && push!(learners, :ml_g => _ml_learner_name(g_learner))
    sname = score === :partialling_out ? "partialling out" : "IV-type"
    return _ml_make_estimate(out, :pliv, score, [string(treatment)], F, cid, G,
                             nrow(data), learners, 0.0, seeds,
                             "θ in Y = θD + g(X) + ε with E[ε | Z, X] = 0",
                             "DML partially linear IV ($sname score)")
end

# ------------------------------------------------------------------------ IIVM

"""
    dml_iivm(data, outcome, treatment, instrument; covariates=Symbol[],
             outcome_learner=LassoLearner(),
             instrument_learner=PenalizedLogisticLearner(),
             treatment_learner=PenalizedLogisticLearner(), always_takers=true,
             never_takers=true, trim=0.01, stratify=true, n_folds=5, n_rep=1,
             folds=nothing, cluster=nothing, rng=Random.default_rng(),
             parallel=true) -> DMLEstimate

Double/debiased machine-learning estimator of the local average treatment effect
(LATE) with a binary instrument and a binary treatment, in the interactive IV model.

With potential treatments ``D(z)`` and potential outcomes ``Y(d)``, the target is the
average effect for compliers,

```math
θ_0 = E[Y(1) - Y(0) \\mid D(1) > D(0)]
= \\frac{E[g_1(X) - g_0(X)]}{E[r_1(X) - r_0(X)]},
```

where ``g_z(X) = E[Y \\mid Z = z, X]`` and ``r_z(X) = P(D = 1 \\mid Z = z, X)``. The
second equality holds under the conditional LATE assumptions of Abadie (2003) and
Frölich (2007): the instrument is as good as randomly assigned given ``X``, it
affects the outcome only through the treatment (exclusion), no unit is a defier
(monotonicity, ``D(1) ≥ D(0)``), the first stage ``E[r_1(X) - r_0(X)]`` is non-zero,
and the instrument propensity ``m_0(X) = P(Z = 1 \\mid X)`` lies strictly between 0 and
1. Only the first stage and overlap can be checked in the data; exclusion and
monotonicity are maintained assumptions.

The estimator solves the linear orthogonal score ``ψ = θ ψ_a + ψ_b`` with

```math
ψ_b = g_1 - g_0 + \\frac{Z(Y - g_1)}{m} - \\frac{(1 - Z)(Y - g_0)}{1 - m}, \\qquad
ψ_a = -\\Big\\{ r_1 - r_0 + \\frac{Z(D - r_1)}{m} - \\frac{(1 - Z)(D - r_0)}{1 - m}
\\Big\\},
```

the ratio of the augmented inverse-probability-weighted estimators of the reduced
form and of the first stage (Frölich 2007), with all nuisances cross-fitted
(Chernozhukov et al. 2018, Section 5.2; DoubleML). With one-sided non-compliance set
`always_takers = false` (then ``r_0 \\equiv 0``) or `never_takers = false`
(``r_1 \\equiv 1``); the function checks that the data are compatible with the
restriction.

Inference uses the sandwich variance of the linear score, as in [`dml_plr`](@ref).
The LATE is a ratio, and the normal approximation deteriorates when the first stage
is weak. Instrument propensities are clipped to
``[\\mathrm{trim}, 1 - \\mathrm{trim}]``; as for [`dml_irm`](@ref), clipping keeps the
weights finite but does not repair limited overlap in the instrument. The
non-machine-learning counterpart is [`late_ipw`](@ref).

# Arguments
- `data`: a `DataFrame` with one row per observation.
- `outcome::Symbol`: the numeric outcome column.
- `treatment::Symbol`: the binary treatment column, coded 0/1.
- `instrument::Symbol`: the binary instrument column, coded 0/1.

# Keywords
- `covariates::Vector{Symbol} = Symbol[]`: numeric covariates ``X`` that make the
  instrument conditionally valid.
- `outcome_learner = LassoLearner()`: the learner for ``g_0`` and ``g_1``.
- `instrument_learner = PenalizedLogisticLearner()`: the learner for ``m_0``; it
  must implement [`fitpredict_proba`](@ref).
- `treatment_learner = PenalizedLogisticLearner()`: the learner for ``r_0`` and
  ``r_1``; it must implement [`fitpredict_proba`](@ref).
- `always_takers::Bool = true`: `false` imposes ``r_0 \\equiv 0`` (no unit is
  treated when ``Z = 0``); an error is thrown if some unit has ``Z = 0, D = 1``.
- `never_takers::Bool = true`: `false` imposes ``r_1 \\equiv 1``; an error is thrown
  if some unit has ``Z = 1, D = 0``.
- `trim::Real = 0.01`: clipping threshold for the instrument propensity, in
  ``[0, 0.5)``.
- `stratify::Bool = true`: stratify the folds by the four ``(Z, D)`` cells (ignored
  when `folds` is given).
- `n_folds::Integer = 5`, `n_rep::Integer = 1`, `folds = nothing`,
  `cluster = nothing`, `rng = Random.default_rng()`, `parallel::Bool = true`: as in
  [`dml_plr`](@ref).

# Returns
- [`DMLEstimate`](@ref) with `model = :iivm` and `score = :LATE`.

# Examples
```julia
using DrSnow, DataFrames, StableRNGs
rng = StableRNG(7)
n = 1000
x1, x2 = randn(rng, n), randn(rng, n)
z = Float64.(rand(rng, n) .< 1 ./ (1 .+ exp.(-0.5 .* x1)))
complier = rand(rng, n) .< 0.6
always = .!complier .& (rand(rng, n) .< 0.5)
d = Float64.(always .| (complier .& (z .== 1)))
y = 2.0 .* d .+ x1 .+ 0.5 .* x2 .+ 0.5 .* always .+ randn(rng, n)   # LATE = 2
df = DataFrame(; y, d, z, x1, x2)
r = dml_iivm(df, :y, :d, :z; covariates=[:x1, :x2], rng=StableRNG(8))
```

# References
- Abadie, A. (2003). Semiparametric instrumental variable estimation of treatment
  response models. *Journal of Econometrics*, 113(2), 231–263.
- Frölich, M. (2007). Nonparametric IV estimation of local average treatment effects
  with covariates. *Journal of Econometrics*, 139(1), 35–75.
- Chernozhukov, V., Chetverikov, D., Demirer, M., Duflo, E., Hansen, C., Newey, W., &
  Robins, J. (2018). Double/debiased machine learning for treatment and structural
  parameters. *The Econometrics Journal*, 21(1), C1–C68.
- Bach, P., Kurz, M. S., Chernozhukov, V., Spindler, M., & Klaassen, S. (2024).
  DoubleML: An object-oriented implementation of double machine learning in R.
  *Journal of Statistical Software*, 108(3), 1–56.
"""
function dml_iivm(data, outcome::Symbol, treatment::Symbol, instrument::Symbol;
                  covariates=Symbol[], outcome_learner=LassoLearner(),
                  instrument_learner=PenalizedLogisticLearner(),
                  treatment_learner=PenalizedLogisticLearner(),
                  always_takers::Bool=true, never_takers::Bool=true, trim::Real=0.01,
                  stratify::Bool=true, n_folds::Integer=5, n_rep::Integer=1,
                  folds=nothing, cluster=nothing, rng::AbstractRNG=Random.default_rng(),
                  parallel::Bool=true)
    ctx = "dml_iivm"
    trim = _ml_check_trim(trim)
    require_columns(data, [treatment, instrument]; context=ctx)
    d = _ml_column(data, treatment; context=ctx)
    z = _ml_column(data, instrument; context=ctx)
    _ml_check_binary_col(d, treatment, ctx)
    _ml_check_binary_col(z, instrument, ctx)
    if !always_takers && any((z .== 0) .& (d .== 1))
        throw(ArgumentError("$ctx: always_takers = false but some units have Z = 0 " *
                            "and D = 1"))
    end
    if !never_takers && any((z .== 1) .& (d .== 0))
        throw(ArgumentError("$ctx: never_takers = false but some units have Z = 1 " *
                            "and D = 0"))
    end
    covs, X, cid, G, F = _ml_setup(data, [outcome, treatment, instrument], covariates,
                                   cluster, folds, n_folds, n_rep, rng,
                                   stratify ? 2 .* z .+ d : nothing; context=ctx)
    y = _ml_column(data, outcome; context=ctx)
    K = maximum(F)
    R = size(F, 2)
    seeds = _ml_seeds(rng, K, 5, R)
    z1 = BitVector(z .== 1)
    z0 = .!z1
    rep_fn = function (r, _)
        specs = [_MLNuisance(:ml_g0, outcome_learner, y, X, false, z0),
                 _MLNuisance(:ml_g1, outcome_learner, y, X, false, z1),
                 _MLNuisance(:ml_m, instrument_learner, z, X, true)]
        slots = [1, 2, 3]
        if always_takers
            push!(specs, _MLNuisance(:ml_r0, treatment_learner, d, X, true, z0))
            push!(slots, 4)
        end
        if never_takers
            push!(specs, _MLNuisance(:ml_r1, treatment_learner, d, X, true, z1))
            push!(slots, 5)
        end
        P = _ml_crossfit(specs, F[:, r], view(seeds, :, slots, r); parallel=parallel,
                         context=ctx)
        g0, g1, m = P[:, 1], P[:, 2], P[:, 3]
        r0 = always_takers ? P[:, 4] : zeros(length(y))
        r1 = never_takers ? P[:, end] : ones(length(y))
        losses = Pair{Symbol,Float64}[:ml_g0 => _ml_rmse(y[z0], g0[z0]),
                                      :ml_g1 => _ml_rmse(y[z1], g1[z1]),
                                      :ml_m => _ml_logloss(z, m)]
        always_takers && push!(losses, :ml_r0 => _ml_logloss(d[z0], r0[z0]))
        never_takers && push!(losses, :ml_r1 => _ml_logloss(d[z1], r1[z1]))
        nt = _ml_clip!(m, trim)
        preds = [:ml_g0 => g0, :ml_g1 => g1, :ml_m => m, :ml_r0 => r0, :ml_r1 => r1]
        psi_b = g1 .- g0 .+ z .* (y .- g1) ./ m .- (1 .- z) .* (y .- g0) ./ (1 .- m)
        psi_a = -(r1 .- r0 .+ z .* (d .- r1) ./ m .- (1 .- z) .* (d .- r0) ./ (1 .- m))
        return (psi_a=psi_a, psi_b=psi_b, preds=preds, losses=losses, ntrim=nt)
    end
    out = _ml_run_dml(rep_fn, 1, F, cid, G)
    learners = [:ml_g0 => _ml_learner_name(outcome_learner),
                :ml_g1 => _ml_learner_name(outcome_learner),
                :ml_m => _ml_learner_name(instrument_learner)]
    always_takers && push!(learners, :ml_r0 => _ml_learner_name(treatment_learner))
    never_takers && push!(learners, :ml_r1 => _ml_learner_name(treatment_learner))
    return _ml_make_estimate(out, :iivm, :LATE, [string(treatment)], F, cid, G,
                             nrow(data), learners, trim, seeds, "LATE",
                             "DML interactive IV model (LATE)")
end

# ------------------------------------------------------- simultaneous inference

"""
    simultaneous_confint(r::DMLEstimate; level=0.95, n_boot=1000, method=:normal,
                         rng=Random.default_rng()) -> NamedTuple

Simultaneous (sup-t) confidence intervals and single-step max-t adjusted p-values for
all coefficients of a DML estimate, by the Gaussian multiplier bootstrap of the
estimated scores.

When a DML fit reports several coefficients (several treatments in
[`dml_plr`](@ref)), marginal 95% intervals do not cover all coefficients jointly with
probability 95%. The sup-t band ``\\tilde θ_j \\pm c_{1-α} \\hat s_j`` uses the critical
value ``c_{1-α}``, the ``(1-α)`` quantile of ``\\max_j |t^*_j|``, where for multiplier
draws ``ξ_1, \\dots, ξ_n`` with mean zero and unit variance

```math
t^*_j = \\frac{\\sum_{i=1}^n ξ_i \\hat ψ_{ij}}{n \\hat J_j \\hat s_{r,j}} ,
```

``\\hat ψ_{ij}`` is the estimated score, ``\\hat J_j`` the mean of ``ψ_a`` and
``\\hat s_{r,j}`` the standard error of the repetition. The procedure follows
Chernozhukov, Chetverikov and Kato (2013) and reproduces DoubleML's `bootstrap()`
followed by `confint(joint = TRUE)`. The adjusted p-value of coefficient ``j`` is
the share of bootstrap maxima that exceed ``|\\tilde θ_j / \\hat s_j|`` (single-step
max-t), which controls the familywise error rate asymptotically.

With clustering the multipliers are drawn per cluster and applied to cluster sums
of the scores, with the factor ``\\sqrt{G/(G-1)}``. With repeated cross-fitting the
bootstrap maxima of all repetitions are pooled, and the bands are centred at the
aggregated estimates with the aggregated standard errors of [`DMLEstimate`](@ref).
The critical value is a normal-approximation quantity; with few clusters it does not
incorporate the ``t_{G-1}`` correction used by `confint`. Report `n_boot` and the
multiplier distribution; results vary slightly with `rng`.

# Arguments
- `r::DMLEstimate`: a fitted DML estimate, typically with several coefficients.

# Keywords
- `level::Real = 0.95`: joint (familywise) coverage level.
- `n_boot::Integer = 1000`: bootstrap draws per repetition of cross-fitting.
- `method::Symbol = :normal`: multiplier distribution: `:normal` (standard normal),
  `:wild` (the two-point distribution of Mammen 1993) or `:bayes` (standard
  exponential minus one).
- `rng::AbstractRNG = Random.default_rng()`: random-number generator for the
  multipliers.

# Returns
- `NamedTuple` with fields `names`, `estimate`, `lower`, `upper`, `pvalues`
  (max-t adjusted), `critical_value` and `level`.

# Examples
```julia
using DrSnow, DataFrames, StableRNGs
rng = StableRNG(9)
n = 500
x1, x2 = randn(rng, n), randn(rng, n)
d1 = x1 .+ randn(rng, n)
d2 = 0.5 .* x2 .+ randn(rng, n)
d3 = randn(rng, n)
y = 1.0 .* d1 .+ 0.5 .* d2 .+ x1 .+ x2 .+ randn(rng, n)
df = DataFrame(; y, d1, d2, d3, x1, x2)
r = dml_plr(df, :y, [:d1, :d2, :d3]; covariates=[:x1, :x2], n_folds=3,
            rng=StableRNG(10))
simultaneous_confint(r; level=0.95, n_boot=2000, rng=StableRNG(5))
```

# References
- Chernozhukov, V., Chetverikov, D., & Kato, K. (2013). Gaussian approximations and
  multiplier bootstrap for maxima of sums of high-dimensional random vectors. *The
  Annals of Statistics*, 41(6), 2786–2819.
- Chernozhukov, V., Chetverikov, D., Demirer, M., Duflo, E., Hansen, C., Newey, W., &
  Robins, J. (2018). Double/debiased machine learning for treatment and structural
  parameters. *The Econometrics Journal*, 21(1), C1–C68.
- Mammen, E. (1993). Bootstrap and wild bootstrap for high dimensional linear
  models. *The Annals of Statistics*, 21(1), 255–285.
- Bach, P., Kurz, M. S., Chernozhukov, V., Spindler, M., & Klaassen, S. (2024).
  DoubleML: An object-oriented implementation of double machine learning in R.
  *Journal of Statistical Software*, 108(3), 1–56.
"""
function simultaneous_confint(r::DMLEstimate; level::Real=0.95, n_boot::Integer=1000,
                              method::Symbol=:normal,
                              rng::AbstractRNG=Random.default_rng())
    0 < level < 1 || throw(ArgumentError("level must be in (0, 1)"))
    n_boot >= 1 || throw(ArgumentError("n_boot must be positive"))
    method in (:normal, :wild, :bayes) ||
        throw(ArgumentError("method must be :normal, :wild or :bayes"))
    n, R, T = size(r.psi)
    maxt = zeros(n_boot * R)
    for rep in 1:R
        Ψ = r.psi[:, rep, :]
        J = vec(mean(r.psi_a[:, rep, :]; dims=1))
        se = r.all_se[:, rep]
        if r.cluster === nothing
            ξ = _ml_multipliers(rng, method, n_boot, n)
            Tb = (ξ * Ψ) ./ (n .* (J .* se)')
        else
            G = r.n_clusters
            S = zeros(G, T)
            for i in 1:n
                @views S[r.cluster[i], :] .+= Ψ[i, :]
            end
            ξ = _ml_multipliers(rng, method, n_boot, G)
            Tb = (ξ * S) ./ (n .* (J .* se)') .* sqrt(G / (G - 1))
        end
        for b in 1:n_boot
            maxt[(rep - 1) * n_boot + b] = maximum(abs, view(Tb, b, :))
        end
    end
    crit = quantile(maxt, level)
    b = StatsAPI.coef(r)
    s = StatsAPI.stderror(r)
    tabs = abs.(b ./ s)
    padj = [mean(maxt .>= t) for t in tabs]
    return (names=copy(r.names), estimate=copy(b), lower=b .- crit .* s,
            upper=b .+ crit .* s, pvalues=padj, critical_value=crit, level=Float64(level))
end

function _ml_multipliers(rng, method, B, n)
    if method === :normal
        return randn(rng, B, n)
    elseif method === :bayes
        return randexp(rng, B, n) .- 1.0
    else
        # Mammen (1993) two-point distribution
        a = (1 - sqrt(5)) / 2
        b = (1 + sqrt(5)) / 2
        pa = (sqrt(5) + 1) / (2 * sqrt(5))
        return map(u -> u < pa ? a : b, rand(rng, B, n))
    end
end
