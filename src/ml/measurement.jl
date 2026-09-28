# Inference with machine-learned measurements: result types and the design-based
# pseudo-outcome engine shared by design-based supervised learning (DSL),
# prediction-powered inference for causal targets, the natural-experiment wrappers
# and the measurement-error tools.
#
# Setting. Every row i has a machine-learned measurement (an ML / LLM prediction
# f_i, and/or text features) of a variable V_i; the gold-standard V_i is observed
# only for rows selected for expert labelling (R_i = 1) with a known probability
# π_i > 0 that may depend on observed variables. With a cross-fitted prediction
# ĝ_i of V_i (trained on labelled rows of other folds), the design-based
# pseudo-outcome
#     Ṽ_i = ĝ_i + (R_i / π_i) (V_i − ĝ_i)
# satisfies E[Ṽ_i | data used by ĝ, covariates] = E[V_i | ...] whatever the
# quality of ĝ, because E[R_i / π_i | V_i, f_i, X_i] = 1 (Egami, Hinck, Stewart
# & Wei 2023). Any estimator that is linear in the outcome is therefore unbiased
# when applied to Ṽ, and its usual robust / cluster-robust variance applied to Ṽ
# accounts for the labelling noise.

# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

"""
    MeasurementEstimate <: CausalEstimate

Result of the estimators for variables measured by machine learning:
[`dsl_regression`](@ref), [`dsl_proportions`](@ref), [`ppi_ate`](@ref),
[`ppi_regression`](@ref), [`cross_ppi`](@ref) and
[`regression_calibration`](@ref).

The object holds estimates of parameters defined with the *true* (gold-standard)
variables, obtained by combining an ML or LLM measurement available on every row
with gold-standard values on a labelled or validation subsample, together with
their estimated covariance. For the design-based estimators the covariance is a
sandwich (cluster-robust with `cluster`) of the corrected moment conditions,
without small-sample factors, and inference uses normal critical values
(`dof_residual` is `Inf`). The "naive" fit that treats the ML measurement as if
it were the true variable is stored for comparison only: its interval is not
valid when prediction errors are correlated with the regressors, with treatment
or with the outcome.

`coef`, `vcov`, `stderror`, `confint(r; level)`, `coeftable`, `coefnames`,
`nobs`, `estimand` and `method_name` work as for every
[`CausalEstimate`](@ref).

# Fields
- `names::Vector{String}`: coefficient names.
- `coef::Vector{Float64}`: corrected point estimates.
- `vcov::Matrix{Float64}`: their estimated covariance.
- `method::String`: description of the estimator.
- `estimand::String`: description of the target parameter.
- `nobs::Int`: rows used (labelled plus unlabelled; for [`cross_ppi`](@ref) the
  sum of both samples).
- `n_labeled::Int`: rows with the gold-standard measure (validation rows for
  [`regression_calibration`](@ref)).
- `n_clusters::Int`: number of clusters used by the variance (0 without
  clustering).
- `naive_coef::Vector{Float64}`, `naive_vcov::Matrix{Float64}`: the same model
  fitted with the ML measurement in place of the true variable (empty when not
  defined).
- `details::NamedTuple`: method-specific output, for example the power-tuning
  parameter `lambda`, the labelling probabilities `prob` and indicator `labeled`,
  cross-fitted predictions `fitted`, fold assignments `folds`, per-repetition
  estimates `rep_coef` and covariances `rep_vcov`, the labelled-only fit, or the
  calibration coefficients; see the docstring of each estimator.
"""
struct MeasurementEstimate <: CausalEstimate
    names::Vector{String}
    coef::Vector{Float64}
    vcov::Matrix{Float64}
    method::String
    estimand::String
    nobs::Int
    n_labeled::Int
    n_clusters::Int
    naive_coef::Vector{Float64}
    naive_vcov::Matrix{Float64}
    details::NamedTuple
end

StatsAPI.coef(r::MeasurementEstimate) = r.coef
StatsAPI.vcov(r::MeasurementEstimate) = r.vcov
StatsAPI.coefnames(r::MeasurementEstimate) = r.names
StatsAPI.nobs(r::MeasurementEstimate) = r.nobs
estimand(r::MeasurementEstimate) = r.estimand
method_name(r::MeasurementEstimate) = r.method

function show_details(io::IO, r::MeasurementEstimate)
    println(io)
    @printf(io, "Rows: %d, with gold-standard measure: %d", r.nobs, r.n_labeled)
    r.n_clusters > 0 && @printf(io, ", clusters: %d", r.n_clusters)
    println(io)
    if hasproperty(r.details, :lambda)
        @printf(io, "Power-tuning λ = %.4g\n", r.details.lambda)
    end
    if !isempty(r.naive_coef)
        se = sqrt.(max.(diag(r.naive_vcov), 0.0))
        println(io, "Naive plug-in of the ML measurement (not valid under differential ",
                "prediction error):")
        for (j, nm) in enumerate(r.names)
            @printf(io, "  %-24s %10.4g (se %.3g)\n", nm, r.naive_coef[j], se[j])
        end
    end
    return nothing
end

"""
    MeasuredOutcomeEstimate <: CausalEstimate

Result of [`did_with_predicted_outcome`](@ref) and
[`rd_with_predicted_outcome`](@ref): a natural-experiment estimate whose outcome
is an ML or LLM measurement, corrected for prediction error with a gold-labelled
subsample.

The object wraps two fits of the same design estimator (a difference-in-
differences or regression-discontinuity estimate): `corrected`, applied to the
corrected outcome, and `naive`, applied to the raw ML measurement. `coef`,
`vcov`, `coefnames`, `confint` and `estimate` refer to the corrected estimate.
In the design-based mode (`mode = :design`) they are those of `corrected`, so
estimator-specific options such as `confint(r; uniform=true)` are forwarded and
`dof_residual` is that of the design estimator; in the stable-error mode
(`mode = :stable_error`) the covariance is a cluster bootstrap that re-estimates
the calibration model, intervals use normal critical values and `confint`
accepts only `level`. `bias_test` contrasts the corrected and naive estimates:
a rejection indicates that the prediction error is differential with respect to
the design (so the naive estimate is biased); a non-rejection is not evidence
that the naive estimate is unbiased, because the test has low power when few
units are labelled.

`stderror`, `coeftable`, `nobs`, `estimand` and `method_name` also work as for
every [`CausalEstimate`](@ref).

# Fields
- `corrected`: the design estimator (e.g. `DiDEstimate`,
  `CallawaySantAnnaEstimate`, `RDEstimate`) applied to the corrected outcome.
- `naive`: the same estimator applied to the raw ML measurement.
- `names::Vector{String}`: coefficient names of the corrected estimate.
- `coef::Vector{Float64}`, `vcov::Matrix{Float64}`: corrected estimates and
  their covariance (a bootstrap covariance in the stable-error mode).
- `mode::Symbol`: `:design` (known labelling probabilities, DSL pseudo-outcome)
  or `:stable_error` (regression calibration under a stable-error assumption).
- `bias_test::DiagnosticTest`: test that the naive and corrected estimands
  coincide, i.e. that the design estimator applied to the prediction error has
  mean zero.
- `nobs::Int`: rows used in the analysis.
- `n_labeled::Int`: gold-labelled rows among them.
- `n_dropped_training::Int`: units (or clusters) excluded because their rows were
  used to train the measurement (`measure_training`).
- `details::NamedTuple`: in the design-based mode the pseudo-outcome `pseudo`,
  the cross-fitted prediction `fitted`, `labeled`, the labelling probabilities
  `prob`, `folds`, the contrast fit `contrast` and the list of design cells
  without labels `unlabeled_cells`; in the stable-error mode the calibrated
  outcome `calibrated`, `labeled`, `bootstrap_draws`, `bootstrap_failures` and
  `unlabeled_cells`.
"""
struct MeasuredOutcomeEstimate{E<:CausalEstimate} <: CausalEstimate
    corrected::E
    naive::E
    names::Vector{String}
    coef::Vector{Float64}
    vcov::Matrix{Float64}
    mode::Symbol
    bias_test::DiagnosticTest
    nobs::Int
    n_labeled::Int
    n_dropped_training::Int
    details::NamedTuple
end

StatsAPI.coef(r::MeasuredOutcomeEstimate) = r.coef
StatsAPI.vcov(r::MeasuredOutcomeEstimate) = r.vcov
StatsAPI.coefnames(r::MeasuredOutcomeEstimate) = r.names
StatsAPI.nobs(r::MeasuredOutcomeEstimate) = r.nobs
StatsAPI.dof_residual(r::MeasuredOutcomeEstimate) =
    r.mode === :design ? StatsAPI.dof_residual(r.corrected) : Inf
estimand(r::MeasuredOutcomeEstimate) = estimand(r.corrected)
method_name(r::MeasuredOutcomeEstimate) =
    (r.mode === :design ? "Prediction-corrected (design-based) " :
     "Prediction-corrected (stable-error calibration) ") * method_name(r.corrected)
estimate(r::MeasuredOutcomeEstimate) =
    r.mode === :design ? estimate(r.corrected) : first(r.coef)

tstats(r::MeasuredOutcomeEstimate) =
    r.mode === :design ? tstats(r.corrected) : r.coef ./ StatsAPI.stderror(r)

function StatsAPI.confint(r::MeasuredOutcomeEstimate; level::Real=0.95, kwargs...)
    r.mode === :design && return StatsAPI.confint(r.corrected; level=level, kwargs...)
    isempty(kwargs) || throw(ArgumentError("confint: options $(keys(kwargs)) are only " *
                                           "available in the design-based mode"))
    c = critical_value(level, Inf)
    s = StatsAPI.stderror(r)
    return hcat(r.coef .- c .* s, r.coef .+ c .* s)
end

function show_details(io::IO, r::MeasuredOutcomeEstimate)
    println(io)
    @printf(io, "Rows: %d, gold-labelled: %d", r.nobs, r.n_labeled)
    r.n_dropped_training > 0 &&
        @printf(io, ", units excluded (used to train the measure): %d",
                r.n_dropped_training)
    println(io)
    b = StatsAPI.coef(r.naive)
    s = StatsAPI.stderror(r.naive)
    @printf(io, "Naive estimate with the raw ML measurement: %.4g (se %.3g)\n", b[1], s[1])
    t = r.bias_test
    @printf(io, "%s: estimate %.4g, p-value = %.4g\n", t.name, t.details.difference,
            t.pvalue)
    return nothing
end

# ---------------------------------------------------------------------------
# Labelling design
# ---------------------------------------------------------------------------

"""
Labelled indicator `R` and labelling probabilities `π` for `data`.

Without `labeled`, a row is labelled when all `gold` columns are non-missing. With
`label_prob`, probabilities must lie in (0, 1]; with `normalize`, they are rescaled
to average the realized labelled share (as in R's dsl). Without `label_prob`, the
labelled rows are taken to be a simple random sample: `π = n_labeled / n`.
"""
function _ml_meas_labels(data, gold::AbstractVector{Symbol}, labeled, label_prob,
                         normalize::Bool; context::AbstractString)
    n = nrow(data)
    n > 0 || throw(ArgumentError("$(context): data has no rows"))
    R = if labeled === nothing
        r = trues(n)
        for c in gold
            r .&= .!ismissing.(data[!, c])
        end
        r
    else
        v = data[!, labeled]
        any(ismissing, v) &&
            throw(ArgumentError("$(context): column $(labeled) has missing values"))
        all(x -> x == 0 || x == 1, v) ||
            throw(ArgumentError("$(context): column $(labeled) must be 0/1 or Bool"))
        BitVector(v .== 1)
    end
    nl = count(R)
    nl >= 2 || throw(ArgumentError("$(context): need at least two gold-labelled rows, " *
                                   "found $(nl)"))
    π = if label_prob === nothing
        fill(nl / n, n)
    else
        p = _ml_column(data, label_prob; context=context)
        all(x -> 0 < x <= 1, p) ||
            throw(ArgumentError("$(context): labelling probabilities in $(label_prob) " *
                                "must lie in (0, 1]"))
        normalize ? p .* ((nl / n) / mean(p)) : p
    end
    return R, π
end

"""Gold-standard values: numeric on labelled rows, 0.0 placeholders elsewhere."""
function _ml_meas_gold(data, col::Symbol, R::AbstractVector{Bool};
                       context::AbstractString)
    v = data[!, col]
    out = zeros(length(v))
    T = nonmissingtype(eltype(v))
    T <: Real || throw(ArgumentError("$(context): column $(col) must be numeric " *
                                     "(got $(T))"))
    for i in eachindex(v)
        R[i] || continue
        x = v[i]
        ismissing(x) && throw(ArgumentError("$(context): labelled row $i has a missing " *
                                            "value in $(col)"))
        isfinite(x) || throw(ArgumentError("$(context): column $(col) has non-finite " *
                                           "values in labelled rows"))
        out[i] = Float64(x)
    end
    return out
end

"""Check that none of `cols` has missing values (these columns are needed on all rows)."""
function _ml_meas_complete(data, cols; context::AbstractString)
    for c in cols
        c === nothing && continue
        any(ismissing, data[!, c]) &&
            throw(ArgumentError("$(context): column $(c) has missing values; it must be " *
                                "observed on every row (drop incomplete rows first)"))
    end
    return nothing
end

# ---------------------------------------------------------------------------
# Cross-fitted prediction of the gold-standard variables
# ---------------------------------------------------------------------------

const _ML_MEAS_GENERIC_PROBA = which(fitpredict_proba,
                                     Tuple{NuisanceLearner,Any,Any,Any})

"""`true` when `learner` implements `fitpredict_proba` (not only the generic error)."""
function _ml_meas_has_proba(learner)
    m = which(fitpredict_proba, Tuple{typeof(learner),Matrix{Float64},Vector{Float64},
                                      Matrix{Float64}})
    return m !== _ML_MEAS_GENERIC_PROBA
end

_ml_meas_is_binary(y) = all(v -> v == 0 || v == 1, y)

"""
Cross-fitted predictions of each target (columns of the returned n × V × n_rep
array): learners are trained on labelled rows outside the fold and predict every
row of the fold. Binary targets use `fitpredict_proba` when the learner has it.
"""
function _ml_meas_crossfit(targets::Vector{Vector{Float64}}, F::Matrix{Float64},
                           R::BitVector, learner, folds::Matrix{Int}, rng;
                           parallel::Bool, context::AbstractString)
    n = size(F, 1)
    V = length(targets)
    K = maximum(folds)
    nrep = size(folds, 2)
    size(F, 2) >= 1 ||
        throw(ArgumentError("$(context): no inputs for the prediction model; pass " *
                            "`prediction` and/or `features`"))
    seeds = _ml_seeds(rng, K, V, nrep)
    out = Array{Float64}(undef, n, V, nrep)
    specs = map(1:V) do v
        yl = targets[v][R]
        proba = _ml_meas_is_binary(yl) && _ml_meas_has_proba(learner)
        _MLNuisance(Symbol("gold_", v), learner, targets[v], F, proba, R)
    end
    for r in 1:nrep
        out[:, :, r] = _ml_crossfit(specs, view(folds, :, r), view(seeds, :, :, r);
                                    parallel=parallel, context=context)
    end
    return out
end

"""Fold matrix for the measurement estimators (grouped by `groups`, stratified by R)."""
function _ml_meas_folds(data, folds, n, n_folds, n_rep, rng, R, groups; context)
    _ml_check_common(n_folds, n_rep)
    return _ml_resolve_folds(data, folds, n, n_folds, n_rep, rng, collect(R), groups;
                             context=context)
end

"""Design-based pseudo-outcome `ĝ + (R/π)(V − ĝ)`."""
_ml_meas_pseudo(g, v, R, π) = g .+ (R ./ π) .* (v .- g)

# ---------------------------------------------------------------------------
# Moment solvers and sandwich variance
# ---------------------------------------------------------------------------
#
# The DSL moment for a model with moment function m(D; β) is
#   m̃_i(β) = (1 − w_i) m(D̂_i; β) + w_i m(D_i; β),   w_i = R_i / π_i,
# where D̂ replaces the gold-standard variables by their cross-fitted predictions.

"""Linear model: closed-form root of the DSL moment. Returns (β, moments, J)."""
function _ml_dsl_linear(yo, yp, Xo, Xp, w; context)
    n = length(yo)
    a = 1 .- w
    J = (Xp' * (Xp .* a) .+ Xo' * (Xo .* w)) ./ n
    b = (Xp' * (a .* yp) .+ Xo' * (w .* yo)) ./ n
    F = lu(J; check=false)
    (issuccess(F) && rank(J) == size(J, 1)) ||
        throw(ArgumentError("$(context): the regressors are collinear; the coefficients " *
                            "are not identified"))
    β = F \ b
    M = Xp .* (a .* (yp .- Xp * β)) .+ Xo .* (w .* (yo .- Xo * β))
    return β, M, J
end

"""Jacobian (mean derivative, sign-flipped) of the logistic DSL moment."""
function _ml_dsl_logit_jac(Xo, Xp, a, w, β)
    qp = _ml_sigmoid.(Xp * β)
    qo = _ml_sigmoid.(Xo * β)
    return (Xp' * (Xp .* (a .* qp .* (1 .- qp))) .+
            Xo' * (Xo .* (w .* qo .* (1 .- qo)))) ./ size(Xp, 1)
end

"""Logistic model: Newton root of the DSL moment with step halving."""
function _ml_dsl_logit(yo, yp, Xo, Xp, w; context, maxit::Int=200)
    n, p = size(Xp)
    a = 1 .- w
    mom(β) = (Xp' * (a .* (yp .- _ml_sigmoid.(Xp * β))) .+
              Xo' * (w .* (yo .- _ml_sigmoid.(Xo * β)))) ./ n
    jac(β) = _ml_dsl_logit_jac(Xo, Xp, a, w, β)
    β = zeros(p)
    g = mom(β)
    converged = false
    for _ in 1:maxit
        if norm(g) < 1e-11
            converged = true
            break
        end
        J = jac(β)
        F = lu(J; check=false)
        issuccess(F) || throw(ArgumentError("$(context): singular Jacobian in the " *
                                            "logistic DSL moment (collinear regressors " *
                                            "or separation)"))
        step = F \ g
        t = 1.0
        βn = β .+ step
        gn = mom(βn)
        while norm(gn) >= norm(g) && t > 1e-10
            t /= 2
            βn = β .+ t .* step
            gn = mom(βn)
        end
        t <= 1e-10 && break
        β, g = βn, gn
    end
    converged = converged || norm(g) < 1e-8
    converged || throw(ArgumentError("$(context): the logistic DSL moment equations " *
                                     "have no root (check for separation)"))
    sp = _ml_sigmoid.(Xp * β)
    so = _ml_sigmoid.(Xo * β)
    M = Xp .* (a .* (yp .- sp)) .+ Xo .* (w .* (yo .- so))
    return β, M, jac(β)
end

"""
Sandwich covariance `J⁻¹ Ω J⁻¹' / n` with `Ω = Σ_c S_c S_c' / n` (cluster sums of the
moment rows; each row its own cluster when `clusters === nothing`). No small-sample
factor, as in R's dsl.
"""
function _ml_meas_sandwich(M::AbstractMatrix, J::AbstractMatrix, clusters)
    n = size(M, 1)
    S = _cluster_sums(M, clusters)
    Jinv = inv(J)
    V = Jinv * ((S' * S) ./ n) * Jinv' ./ n
    return Matrix(Symmetric((V .+ V') ./ 2))
end

"""Cluster vector (or `nothing`) and number of clusters."""
function _ml_meas_clusters(data, cluster; context)
    cluster === nothing && return nothing, 0
    g, G = _ml_cluster_ids(data, cluster; context=context)
    return g, G
end

"""Residualize the columns `cols` of `df` on the fixed effects `fe` (FixedEffectModels)."""
function _ml_meas_partial_out(df, cols::Vector{Symbol}, fe::Vector{Symbol})
    lhs = Tuple(StatsModels.term(c) for c in cols)
    rhs = _sumterms([FixedEffectModels.fe(f) for f in fe])
    res = FixedEffectModels.partial_out(df, StatsModels.FormulaTerm(lhs, rhs);
                                        tol=1e-12, maxiter=100_000)
    R = res[1]
    nrow(R) == nrow(df) ||
        throw(ArgumentError("partial_out dropped rows; check the fixed-effect columns"))
    return Matrix{Float64}(R)
end

# ---------------------------------------------------------------------------
# Exported building block
# ---------------------------------------------------------------------------

"""
    dsl_pseudo_outcome(data, outcome; prediction=Symbol[], features=Symbol[],
                       labeled=nothing, label_prob=nothing, normalize_prob=true,
                       learner=OLSLearner(), cluster=nothing, n_folds=5,
                       folds=nothing, rng=Random.default_rng(), parallel=false)
        -> NamedTuple

Design-based pseudo-outcome of design-based supervised learning (DSL) for an
outcome measured by machine learning, to be analysed with any estimator that is
linear in the outcome.

The setting is that of Egami, Hinck, Stewart and Wei (2023): every row carries an
ML or LLM measurement of the outcome, and an expert (gold-standard) value
``Y_i`` is observed only on rows selected for labelling (``R_i = 1``), with a
selection probability ``\\pi_i`` that is known by design and may depend on
observed variables. With ``\\hat g_i`` a cross-fitted prediction of ``Y_i``
from the ML prediction(s) and `features`, trained on the labelled rows of the
other folds, the pseudo-outcome is

```math
\\tilde Y_i = \\hat g_i + \\frac{R_i}{\\pi_i} (Y_i - \\hat g_i).
```

Because ``E[R_i/\\pi_i \\mid Y_i, \\hat g_i, X_i] = 1``, ``E[\\tilde Y_i \\mid
X_i] = E[Y_i \\mid X_i]`` however poor ``\\hat g`` is: the construction is the
augmented inverse-probability-weighted score for data missing by design
(Robins, Rotnitzky & Zhao 1994), with the labelling probability known rather
than estimated. Consequently any estimator that is linear in the outcome given
the design (a difference in means, least squares, fixed-effects and
difference-in-differences regressions, local-polynomial regression
discontinuity) applied to ``\\tilde Y`` targets the estimand defined with the
true outcome, and its heteroskedasticity- or cluster-robust standard errors
computed on ``\\tilde Y`` account for the labelling noise. A more accurate
``\\hat g`` only reduces the variance.

The guarantee rests on three conditions that the data cannot check: the gold
labels are error-free measurements of the outcome of interest; the labelling
probabilities are known (from the sampling design, not estimated from the
labels) and bounded away from zero, since rows with tiny ``\\pi_i`` receive
weights ``1/\\pi_i`` that make the pseudo-outcome, and therefore every estimate,
very noisy; and ``\\hat g_i`` does not use the label of row ``i``, which
cross-fitting ensures. Cluster the downstream variance at least at the level at
which labels were sampled; with `cluster`, folds are formed by cluster so that
``\\hat g`` never uses the labels of the cluster it predicts. For regression
models, prefer [`dsl_regression`](@ref), which also handles ML-measured
covariates and logistic models; for difference-in-differences and regression
discontinuity use [`did_with_predicted_outcome`](@ref) and
[`rd_with_predicted_outcome`](@ref).

# Arguments
- `data::AbstractDataFrame`: one row per unit of analysis.
- `outcome::Symbol`: the gold-standard outcome; `missing` on unlabelled rows
  unless `labeled` is given (values on unlabelled rows are then ignored).

# Keywords
- `prediction = Symbol[]`: ML prediction column(s) of the outcome, observed on
  every row.
- `features::Vector{Symbol} = Symbol[]`: further inputs of the model for
  ``\\hat g`` (e.g. text features). With `prediction = Symbol[]`, the measurement
  itself is learned from `features` on the labelled rows by cross-fitting, which
  is the cross-prediction setting of Zrnic and Candès (2024).
- `labeled = nothing`: 0/1 column marking labelled rows (default: rows with a
  non-missing `outcome`).
- `label_prob = nothing`: column of labelling probabilities in ``(0, 1]``. The
  default treats the labelled rows as a simple random sample,
  ``\\pi_i = n_{\\text{labeled}}/n``.
- `normalize_prob = true`: rescale `label_prob` to average the realized labelled
  share, as in the R package dsl; set `false` to use the design probabilities as
  given (e.g. Bernoulli sampling).
- `learner = OLSLearner()`: a [`NuisanceLearner`](@ref) for ``\\hat g`` (binary
  outcomes use `fitpredict_proba` when the learner implements it), or `nothing`
  to use the single `prediction` column as ``\\hat g`` without recalibration.
- `cluster = nothing`: column defining the cross-fitting groups.
- `n_folds = 5`, `folds = nothing`, `rng = Random.default_rng()`,
  `parallel = false`: cross-fitting options (see [`crossfit_folds`](@ref)).

# Returns
- `NamedTuple` `(pseudo, fitted, labeled, prob, folds)`: the pseudo-outcome
  ``\\tilde Y``, the cross-fitted ``\\hat g``, the labelled indicator, the
  labelling probabilities used and the fold assignment.

# Examples
```julia
using DrSnow, DataFrames, StableRNGs, Statistics
rng = StableRNG(1)
n = 2000
treated = Float64.(rand(rng, n) .< 0.5)
y = 1 .+ 1.0 .* treated .+ randn(rng, n)
y_llm = 0.3 .+ 0.8 .* y .+ 0.4 .* treated .+ 0.6 .* randn(rng, n)  # differential
lab = rand(rng, n) .< 0.25
df = DataFrame(y_expert=[l ? v : missing for (l, v) in zip(lab, y)], y_llm=y_llm,
               treated=treated)
p = dsl_pseudo_outcome(df, :y_expert; prediction=:y_llm, rng=StableRNG(2))
mean(p.pseudo[treated .== 1]) - mean(p.pseudo[treated .== 0])   # close to 1
mean(y_llm[treated .== 1]) - mean(y_llm[treated .== 0])         # naive, biased
```

# References
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
  structural parameters. *The Econometrics Journal*, 21(1), C1–C68.
- Zrnic, T., & Candès, E. J. (2024). Cross-prediction-powered inference.
  *Proceedings of the National Academy of Sciences*, 121(15), e2322083121.
"""
function dsl_pseudo_outcome(data::AbstractDataFrame, outcome::Symbol;
                            prediction=Symbol[], features::Vector{Symbol}=Symbol[],
                            labeled::Union{Nothing,Symbol}=nothing,
                            label_prob::Union{Nothing,Symbol}=nothing,
                            normalize_prob::Bool=true, learner=OLSLearner(),
                            cluster::Union{Nothing,Symbol}=nothing,
                            n_folds::Integer=5, folds=nothing,
                            rng::AbstractRNG=Random.default_rng(), parallel::Bool=false)
    ctx = "dsl_pseudo_outcome"
    preds = _as_symbols(prediction)
    require_columns(data, vcat(outcome, preds, features, labeled, label_prob, cluster);
                    context=ctx)
    _ml_meas_complete(data, vcat(preds, features, cluster); context=ctx)
    R, π = _ml_meas_labels(data, [outcome], labeled, label_prob, normalize_prob;
                           context=ctx)
    y = _ml_meas_gold(data, outcome, R; context=ctx)
    n = nrow(data)
    groups = cluster === nothing ? nothing : data[!, cluster]
    F = _ml_meas_folds(data, folds, n, n_folds, 1, rng, R, groups; context=ctx)
    g = _ml_meas_g(data, [y], preds, features, R, learner, F, rng; parallel=parallel,
                   context=ctx)[:, 1, 1]
    return (pseudo=_ml_meas_pseudo(g, y, R, π), fitted=g, labeled=R, prob=π,
            folds=F[:, 1])
end

"""
Predictions `ĝ` (n × V × n_rep) of the gold targets: cross-fitted with `learner`, or
the raw prediction columns (paired with the targets) when `learner === nothing`.
"""
function _ml_meas_g(data, targets, preds, features, R, learner, F, rng; parallel,
                    context, extra::Matrix{Float64}=zeros(nrow(data), 0))
    n = nrow(data)
    if learner === nothing
        length(preds) == length(targets) ||
            throw(ArgumentError("$(context): with `learner = nothing` give exactly one " *
                                "prediction column per ML-measured variable"))
        G = Array{Float64}(undef, n, length(targets), size(F, 2))
        for (v, p) in enumerate(preds), r in axes(F, 2)
            G[:, v, r] = _ml_column(data, p; context=context)
        end
        return G
    end
    learner isa NuisanceLearner ||
        throw(ArgumentError("$(context): learner must be a NuisanceLearner or nothing"))
    X = hcat(_ml_matrix(data, vcat(preds, features); context=context), extra)
    return _ml_meas_crossfit(targets, X, BitVector(R), learner, F, rng;
                             parallel=parallel, context=context)
end
