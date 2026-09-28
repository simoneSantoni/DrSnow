# Machine-learning-powered specification tests for linear IV models: residual
# prediction tests of Scheidegger, Londschien & Bühlmann (2025).
#
# The sample is split into an auxiliary sample (a fraction `aux_fraction`) and a main
# sample. On the auxiliary sample a learner is trained to predict IV residuals from the
# instruments (and exogenous covariates); the learned, clipped weight function ŵ with
# |ŵ| ≤ 1 is then correlated with the residuals of the main sample. Under the null the
# residuals are mean-independent of the instruments, so the normalized correlation is
# asymptotically N(0, 1) whatever the learner did (a floor γ on the variance guards
# against degenerate weights). The test is one-sided because ŵ is built to correlate
# positively with the residuals.
#
# - strong-identification version (Procedure 2): two-stage least squares residuals, with
#   a variance correction for estimating β on the main sample;
# - weak-IV-robust version (Procedure 3): residuals at a hypothesized β₀ after
#   partialling out the exogenous covariates (Anderson–Rubin-type); inverting it gives a
#   confidence set that is also a joint specification test (an empty set rejects the
#   model).
#
# The learners come from the ml area (`fitpredict`), which is loaded after this area;
# they are only called at run time.

# ---------------------------------------------------------------------------
# Data preparation and sample split
# ---------------------------------------------------------------------------

"""Codes `1:G` numbered in sorted order of the labels (row-order invariant)."""
function _iv_sorted_codes(v::AbstractVector)
    u = unique(v)
    try
        sort!(u)
    catch
    end
    idx = Dict(x => i for (i, x) in enumerate(u))
    return [idx[x] for x in v], length(u)
end

function _iv_rp_prepare(data::AbstractDataFrame, outcome::Symbol, endogenous, instruments,
                        covariates, cluster, aux_sample, ctx::AbstractString)
    endo = _as_symbols(endogenous)
    inst = _as_symbols(instruments)
    covs = _as_symbols(covariates)
    isempty(endo) && throw(ArgumentError("$ctx: at least one endogenous regressor is " *
                                         "required"))
    isempty(inst) && throw(ArgumentError("$ctx: at least one instrument is required"))
    length(inst) >= length(endo) ||
        throw(ArgumentError("$ctx: need at least as many instruments as endogenous " *
                            "regressors ($(length(inst)) < $(length(endo)))"))
    cl = cluster === nothing ? Symbol[] : _as_symbols(cluster)
    length(cl) <= 1 || throw(ArgumentError("$ctx: only one-way clustering is supported"))
    auxcol = aux_sample isa Symbol ? [aux_sample] : Symbol[]
    used = unique(vcat(outcome, endo, inst, covs, cl, auxcol))
    require_columns(data, used; context=ctx)
    _iv_check_numeric(data, vcat(outcome, endo, inst), ctx)
    keep = BitVector([all(c -> !ismissing(data[i, c]), used) for i in 1:nrow(data)])
    sub = disallowmissing(data[keep, used])
    n = nrow(sub)
    y = Float64.(sub[!, outcome])
    X = Matrix{Float64}(hcat([Float64.(sub[!, c]) for c in endo]...))
    Z = Matrix{Float64}(hcat([Float64.(sub[!, c]) for c in inst]...))
    all(isfinite, y) && all(isfinite, X) && all(isfinite, Z) ||
        throw(ArgumentError("$ctx: outcome, endogenous regressors and instruments must " *
                            "be finite"))
    Cb = _iv_exog_matrix(sub, covs, true)            # intercept first
    Cp = Cb[:, 2:end]                                # covariates for the learner
    cid, G = cl == Symbol[] ? (nothing, 0) : _iv_sorted_codes(sub[!, cl[1]])
    aux = if aux_sample === nothing
        nothing
    elseif aux_sample isa Symbol
        v = sub[!, aux_sample]
        all(x -> x == 0 || x == 1, v) ||
            throw(ArgumentError("$ctx: `aux_sample` column must be Bool or 0/1"))
        BitVector(v .== 1)
    else
        length(aux_sample) == nrow(data) ||
            throw(DimensionMismatch("$ctx: `aux_sample` must have one entry per row of " *
                                    "`data` ($(nrow(data)))"))
        BitVector(Bool.(collect(aux_sample))[keep])
    end
    return (; y, X, Z, Cb, Cp, cid, G, n, aux, endo, inst, covs)
end

"""Auxiliary-sample indicator: a random `frac` of observations (or clusters)."""
function _iv_rp_split(prep, frac, rng::AbstractRNG, ctx)
    n = prep.n
    if prep.aux !== nothing
        aux = prep.aux
    else
        f = frac === nothing ? min(0.5, exp(1) / log(n)) : float(frac)
        0 < f < 1 || throw(ArgumentError("$ctx: aux_fraction must be in (0, 1)"))
        if prep.cid === nothing
            m = round(Int, n * f)
            aux = falses(n)
            aux[randperm(rng, n)[1:m]] .= true
        else
            G = prep.G
            m = round(Int, G * f)
            chosen = falses(G)
            chosen[randperm(rng, G)[1:m]] .= true
            aux = BitVector(chosen[prep.cid])
        end
    end
    na, nm = count(aux), n - count(aux)
    kx = size(prep.X, 2) + size(prep.Cb, 2)
    kz = size(prep.Z, 2) + size(prep.Cb, 2)
    (na > kz && nm > kz) ||
        throw(ArgumentError("$ctx: the auxiliary ($na) and main ($nm) samples must " *
                            "each have more than $kz observations"))
    if prep.cid !== nothing
        length(unique(prep.cid[.!aux])) >= 2 ||
            throw(ArgumentError("$ctx: the main sample needs at least two clusters"))
    end
    kx <= kz || throw(ArgumentError("$ctx: under-identified model"))
    return aux
end

# ---------------------------------------------------------------------------
# Weight function (Procedures 1 and 3, step 3)
# ---------------------------------------------------------------------------

"""Clip predictions: `sign(w₀) min(|w₀|, K) / K`, K the `q` quantile of |train preds|."""
function _iv_rp_clip(pred_train::AbstractVector, pred_test::AbstractVector, q::Real)
    if q == 0
        return sign.(pred_test)
    end
    K = quantile(abs.(pred_train), q)
    K > 0 || return sign.(pred_test)
    return sign.(pred_test) .* min.(abs.(pred_test), K) ./ K
end

function _iv_rp_learn(learner, P_train, target, P_test, seed)
    Pn = vcat(P_train, P_test)
    pred = fitpredict(learner, P_train, target, Pn; rng=Random.Xoshiro(seed))
    length(pred) == size(Pn, 1) ||
        throw(DimensionMismatch("learner returned $(length(pred)) predictions for " *
                                "$(size(Pn, 1)) rows"))
    all(isfinite, pred) || throw(ArgumentError("learner returned non-finite predictions"))
    na = size(P_train, 1)
    return pred[1:na], pred[(na + 1):end]
end

"""Residuals of the columns of `A` on the columns of `B` (least squares)."""
_iv_rp_resid(A, B) = A - B * (qr(B, ColumnNorm()) \ A)

"""Two-stage least squares of `y` on `Xb` with instruments `Zb` (all columns
included): coefficients, residuals and first-stage fitted values."""
function _iv_rp_tsls(y, Xb, Zb)
    F = qr(Zb, ColumnNorm())
    Xh = Zb * (F \ Xb)
    b = (Xh' * Xh) \ (Xh' * y)
    return b, y - Xb * b, Xh
end

# ---------------------------------------------------------------------------
# Test statistics
# ---------------------------------------------------------------------------

function _iv_rp_variance(kind::Symbol, a::AbstractVector, r::AbstractVector,
                         wr_mean::Real, cl)
    n0 = length(r)
    if kind === :heteroskedastic
        return mean(a .^ 2 .* r .^ 2) - wr_mean^2
    elseif kind === :homoskedastic
        return mean(a .^ 2) * mean(r .^ 2)
    else
        codes, G = _iv_sorted_codes(cl)
        S = zeros(G)
        for i in 1:n0
            S[codes[i]] += a[i] * r[i]
        end
        return sum(abs2, S) / n0 - n0 / G * wr_mean^2
    end
end

"""Strong-identification statistic on the main sample given weights `w`."""
function _iv_rp_strong_stat(y, Xb, Zb, w, kind, gamma, cl)
    _, r, Xh = _iv_rp_tsls(y, Xb, Zb)
    ZAw = -Xh * ((Xh' * Xh) \ (Xb' * w))
    n0 = length(r)
    wr = mean(w .* r)
    s2 = _iv_rp_variance(kind, w .+ ZAw, r, wr, cl)
    return _iv_rp_finish(sum(w .* r), s2, mean(r .^ 2), n0, gamma)
end

"""Weak-IV-robust statistic at β₀ on the main sample given weights `w`."""
function _iv_rp_weak_stat(y, X, Cb, w, β0, kind, gamma, cl)
    MR = _iv_rp_resid(y - X * β0, Cb)
    Mw = _iv_rp_resid(w, Cb)
    n0 = length(MR)
    wr = mean(MR .* Mw)
    s2 = if kind === :heteroskedastic
        mean(MR .^ 2 .* Mw .^ 2) - wr^2
    elseif kind === :homoskedastic
        mean(MR .^ 2) * mean(Mw .^ 2)
    else
        _iv_rp_variance(:cluster, Mw, MR, wr, cl)
    end
    return _iv_rp_finish(sum(MR .* Mw), s2, mean(MR .^ 2), n0, gamma)
end

function _iv_rp_finish(num, s2, noise, n0, gamma)
    frac = noise > 0 ? s2 / noise : 0.0
    floored = !(frac >= gamma)
    s2u = floored ? gamma * noise : s2
    s2u > 0 || return (T=0.0, T_raw=0.0, var_fraction=frac, floored=true)
    T = num / sqrt(n0 * s2u)
    T_raw = s2 > 0 ? num / sqrt(n0 * s2) : (num == 0 ? 0.0 : copysign(Inf, num))
    return (T=T, T_raw=T_raw, var_fraction=frac, floored=floored)
end

function _iv_rp_variance_kind(variance, cluster, ctx)
    v = variance === nothing ? (cluster === nothing ? :heteroskedastic : :cluster) :
        variance
    v in (:heteroskedastic, :homoskedastic, :cluster) ||
        throw(ArgumentError("$ctx: variance must be :heteroskedastic, :homoskedastic " *
                            "or :cluster"))
    v === :cluster && cluster === nothing &&
        throw(ArgumentError("$ctx: variance = :cluster requires `cluster`"))
    return v
end

# ---------------------------------------------------------------------------
# Weak-IV-robust engine: p-value function of β₀ for a fixed split
# ---------------------------------------------------------------------------

function _iv_rp_weak_engine(prep, aux, learner, weight_update, use_covariates, q, kind,
                            gamma, seed, ctx)
    weight_update in (:refit, :linear, :fixed) ||
        throw(ArgumentError("$ctx: weight_update must be :refit, :linear or :fixed"))
    A, M = aux, .!aux
    Cb_A = prep.Cb[A, :]
    MY = vec(_iv_rp_resid(prep.y[A], Cb_A))
    MX = _iv_rp_resid(prep.X[A, :], Cb_A)
    P = use_covariates ? hcat(prep.Cp, prep.Z) : prep.Z
    P_A, P_M = P[A, :], P[M, :]
    yM, XM, CbM = prep.y[M], prep.X[M, :], prep.Cb[M, :]
    clM = prep.cid === nothing ? nothing : prep.cid[M]
    p = size(prep.X, 2)
    # 2SLS on the auxiliary sample (partialled), used for :fixed and as a reference
    ZA = prep.Z[A, :]
    Xh = ZA * (qr(ZA, ColumnNorm()) \ MX)
    b_aux = (Xh' * Xh) \ (Xh' * MY)
    lin = nothing
    fixed = nothing
    if weight_update === :linear
        fY = _iv_rp_learn(learner, P_A, MY, P_M, seed)
        fX = [_iv_rp_learn(learner, P_A, MX[:, j], P_M, seed) for j in 1:p]
        lin = (fY, fX)
    elseif weight_update === :fixed
        fixed = _iv_rp_learn(learner, P_A, MY - MX * b_aux, P_M, seed)
    end
    weights_at = function (β0::AbstractVector)
        tr, te = if weight_update === :refit
            _iv_rp_learn(learner, P_A, MY - MX * β0, P_M, seed)
        elseif weight_update === :linear
            (lin[1][1] - sum(β0[j] .* lin[2][j][1] for j in 1:p),
             lin[1][2] - sum(β0[j] .* lin[2][j][2] for j in 1:p))
        else
            fixed
        end
        return _iv_rp_clip(tr, te, q)
    end
    stat_at = function (β0::AbstractVector)
        w = weights_at(β0)
        return _iv_rp_weak_stat(yM, XM, CbM, w, β0, kind, gamma, clM)
    end
    return (stat_at=stat_at, b_aux=b_aux, n_aux=count(A), n_main=count(M))
end

function _iv_rp_learner_label(learner)
    return try
        _ml_learner_name(learner)
    catch
        string(nameof(typeof(learner)))
    end
end

const _IV_RP_REF = "Scheidegger, Londschien & Bühlmann (2025)"

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

"""
    residual_prediction_test(data, outcome, endogenous, instruments;
                             covariates=Symbol[], learner=ForestLearner(),
                             beta0=nothing, aux_fraction=nothing, aux_sample=nothing,
                             clip_quantile=0.8, gamma=0.05, variance=nothing,
                             cluster=nothing, weight_update=:refit,
                             use_covariates=true, rng=Random.default_rng())
        -> DiagnosticTest

Machine-learning-powered specification test of the linear IV model: the residual
prediction test of Scheidegger, Londschien and Bühlmann (2025).

The linear IV model ``Y = X'\\beta + C'\\theta + \\varepsilon`` with endogenous
regressors ``X``, exogenous covariates ``C`` (including an intercept) and excluded
instruments ``Z`` is taken to be well specified when the conditional moment
restriction

```math
H_0:\\; E[Y - X'\\beta - C'\\theta \\mid Z, C] = 0
\\quad \\text{for some } (\\beta, \\theta)
```

holds. It is implied by a linear structural equation together with an instrument that
is mean-independent of the structural error, and it fails, for instance, when an
instrument has a direct effect on ``Y`` (a violation of exclusion), when the
instrument is correlated with the error, or when the structural function is
nonlinear in ``X``. Classical overidentification tests
([`overidentification_test`](@ref)) examine only the unconditional moments
``E[Z\\varepsilon] = 0``; they have no power in the just-identified case and little
power against violations that are nonlinear in ``Z``. The residual prediction test
instead asks whether a flexible learner can predict the IV residuals from
``(Z, C)``, which targets the conditional restriction directly.

The procedure splits the sample into an auxiliary sample (a fraction `aux_fraction`, by
default ``\\min(0.5, e / \\log n)``, drawn at the cluster level when `cluster` is given)
and a main sample. On the auxiliary sample, `learner` is trained to predict the IV
residuals from ``(Z, C)``; its predictions are clipped at the `clip_quantile` quantile
``K`` of their absolute in-sample values and rescaled, giving a weight function
``\\hat w = \\operatorname{sign}(\\hat w_0) \\min(|\\hat w_0|, K)/K`` with
``|\\hat w| \\le 1``. On the main sample of size ``n_0`` the statistic
``T = \\sum_i \\hat w(Z_i, C_i) r_i / \\sqrt{n_0 \\hat\\sigma^2}`` is compared with the
standard normal distribution, one-sided (p-value ``1 - \\Phi(T)``), because ``\\hat w``
is constructed to correlate positively with the residuals. Because ``\\hat w`` is a
fixed function on the main sample, ``T`` is asymptotically normal under ``H_0`` whatever
the learner did, so the learner affects power but not size; to guard against weight
functions that are nearly orthogonal to the residuals, ``\\hat\\sigma^2`` is replaced by
the floor `gamma` times the mean squared residual when it falls below it
(`details.floored`). Two versions are implemented. With `beta0 = nothing` (the
strong-identification version, Procedures 1–2 of the paper), ``r`` are 2SLS residuals
re-estimated on the main sample and ``\\hat\\sigma^2`` accounts for the estimation of
``\\beta``; it requires strong instruments. With `beta0 = β₀` (the weak-IV-robust
version, Procedure 3), the test is of the joint hypothesis
``E[Y - X'\\beta_0 - C'\\theta \\mid Z, C] = 0`` for some ``\\theta``, with
``r = Y - X'\\beta_0`` after partialling out ``C``; like the Anderson and Rubin (1949)
test it is valid whatever the strength of the instruments, and inverting it over
``\\beta_0`` gives [`residual_prediction_confidence_set`](@ref).

Some violations cannot be detected by any learner. In the just-identified case the
2SLS residuals are linearly uncorrelated with ``Z`` by construction, so only nonlinear
violations are detectable; more generally, a violation that lies in the span of
``E[X \\mid Z]`` is absorbed by the estimate of ``\\beta`` (Lemma 1 of the paper). A
non-rejection is therefore not evidence of a correctly specified model, and a
rejection does not say which assumption fails. The result depends on the random
sample split and on the learner's randomness; report the seed or pass an explicit
`aux_sample`. The heteroskedasticity-robust variance is the default; the
homoskedastic variance adds homoskedasticity to the null hypothesis.

# Arguments
- `data::AbstractDataFrame`: the data; rows with missing values in used columns are
  dropped.
- `outcome::Symbol`: the outcome ``Y`` (numeric).
- `endogenous`: the endogenous regressor(s) ``X``, a `Symbol` or a vector (numeric).
- `instruments`: the excluded instrument(s) ``Z``, a `Symbol` or a vector (numeric);
  at least as many as endogenous regressors.

# Keywords
- `covariates::Vector{Symbol}`: exogenous controls ``C`` (default none); categorical
  columns are dummy-coded and an intercept is always included.
- `learner`: the regression learner that predicts the residuals, any
  [`NuisanceLearner`](@ref) (default `ForestLearner()`, an honest regression forest;
  Lasso, kNN or MLJ models via [`MLJLearner`](@ref) are alternatives). The choice
  affects power only.
- `beta0`: `nothing` (default, strong-identification version) or the hypothesized
  coefficient(s) of `endogenous`, a number or a vector with one entry per endogenous
  regressor (weak-IV-robust version).
- `aux_fraction::Real`: fraction of observations (or clusters) in the auxiliary
  sample (default `nothing`, meaning ``\\min(0.5, e / \\log n)``).
- `aux_sample`: `nothing` (default, random split drawn with `rng`), or an explicit
  auxiliary-sample indicator, a Bool vector with one entry per row of `data` or the
  name of a 0/1 column; it makes the split independent of `rng` and of row order.
- `clip_quantile::Real`: the quantile in ``[0, 1]`` at which the learned weights are
  clipped (default 0.8); `0` uses only the sign of the predictions.
- `gamma::Real`: floor on the variance relative to the mean squared residual (default
  0.05); larger values make the test more conservative against degenerate weights.
- `variance`: `:heteroskedastic`, `:homoskedastic` or `:cluster`; the default
  `nothing` means `:heteroskedastic`, or `:cluster` when `cluster` is given.
- `cluster`: a column for one-way cluster-robust variance and a cluster-level split
  (default `nothing`).
- `weight_update::Symbol` (weak-IV-robust version only): `:refit` (default) refits the
  learner on the residuals at ``\\beta_0`` (Procedure 3); `:linear` fits the learner once
  to the partialled outcome and to each partialled endogenous regressor and combines
  the predictions linearly in ``\\beta_0`` (exact for linear smoothers and much faster
  for confidence sets); `:fixed` uses the weight function learned from the
  auxiliary-sample 2SLS residuals. All three give valid tests.
- `use_covariates::Bool` (weak-IV-robust version only): include ``C`` among the
  learner's predictors (default `true`).
- `rng::AbstractRNG`: draws the sample split and the learner's seed.

# Returns
- A [`DiagnosticTest`](@ref) with the one-sided p-value; `details` holds `T_raw` (the
  statistic before flooring), `var_fraction`, `floored`, `n_aux`, `n_main`, `beta`
  (the main-sample 2SLS estimate, or `beta0`), `learner`, `variance` and `aux_sample`
  (the auxiliary-sample indicator over the rows used).

# Examples
```julia
using DrSnow, DataFrames, StableRNGs
rng = StableRNG(1)
n = 1_000
x, z1, z2, v = randn(rng, n), randn(rng, n), randn(rng, n), randn(rng, n)
d = 0.6 .* z1 .+ 0.4 .* z2 .+ 0.5 .* x .+ v
# z1 has a nonlinear direct effect on y: the model is misspecified
y = d .+ x .+ 0.8 .* z1 .^ 2 .+ 0.6 .* v .+ randn(rng, n)
df = DataFrame(y=y, d=d, z1=z1, z2=z2, x=x)
residual_prediction_test(df, :y, :d, [:z1, :z2]; covariates=[:x],
                         learner=ForestLearner(num_trees=100), rng=StableRNG(2))
# weak-IV-robust version at β₀ = 1, using only the valid instrument
residual_prediction_test(df, :y, :d, :z2; covariates=[:x], beta0=1.0,
                         learner=ForestLearner(num_trees=100), rng=StableRNG(3))
```

# References
- Scheidegger, C., Londschien, M., & Bühlmann, P. (2025). Machine-learning-powered
  specification testing in linear instrumental variable models. arXiv:2506.12771.
  (R package `RPIV`, used for validation.)
- Anderson, T. W., & Rubin, H. (1949). Estimation of the parameters of a single
  equation in a complete system of stochastic equations. *Annals of Mathematical
  Statistics*, 20(1), 46–63.
- Sargan, J. D. (1958). The estimation of economic relationships using instrumental
  variables. *Econometrica*, 26(3), 393–415.
- Hansen, L. P. (1982). Large sample properties of generalized method of moments
  estimators. *Econometrica*, 50(4), 1029–1054.
"""
function residual_prediction_test(data::AbstractDataFrame, outcome::Symbol, endogenous,
                                  instruments; covariates=Symbol[],
                                  learner=ForestLearner(), beta0=nothing,
                                  aux_fraction=nothing, aux_sample=nothing,
                                  clip_quantile::Real=0.8, gamma::Real=0.05,
                                  variance=nothing, cluster=nothing,
                                  weight_update::Symbol=:refit, use_covariates::Bool=true,
                                  rng::AbstractRNG=Random.default_rng())
    ctx = "residual_prediction_test"
    0 <= clip_quantile <= 1 || throw(ArgumentError("$ctx: clip_quantile must be in [0, 1]"))
    gamma >= 0 || throw(ArgumentError("$ctx: gamma must be non-negative"))
    kind = _iv_rp_variance_kind(variance, cluster, ctx)
    prep = _iv_rp_prepare(data, outcome, endogenous, instruments, covariates, cluster,
                          aux_sample, ctx)
    aux = _iv_rp_split(prep, aux_fraction, rng, ctx)
    seed = task_seeds(rng, 1)[1]
    lname = _iv_rp_learner_label(learner)
    p = size(prep.X, 2)
    clM = prep.cid === nothing ? nothing : prep.cid[.!aux]
    if beta0 === nothing
        A, M = aux, .!aux
        Xb = hcat(prep.Cb[:, 1:1], prep.X, prep.Cb[:, 2:end])
        Zb = hcat(prep.Cb[:, 1:1], prep.Z, prep.Cb[:, 2:end])
        _, rA, _ = _iv_rp_tsls(prep.y[A], Xb[A, :], Zb[A, :])
        P = hcat(prep.Z, prep.Cp)
        tr, te = _iv_rp_learn(learner, P[A, :], rA, P[M, :], seed)
        w = _iv_rp_clip(tr, te, clip_quantile)
        st = _iv_rp_strong_stat(prep.y[M], Xb[M, :], Zb[M, :], w, kind, gamma, clM)
        bM, _, _ = _iv_rp_tsls(prep.y[M], Xb[M, :], Zb[M, :])
        beta = bM[2:(1 + p)]
        null = "the linear IV model is well specified: E[Y − X'β − C'θ | Z, C] = 0 for " *
               "some (β, θ)"
        name = "Residual prediction specification test"
        method = "sample split ($(count(A)) auxiliary / $(count(M)) main), $lname " *
                 "weights, 2SLS residuals, $kind variance, one-sided N(0, 1)"
        note = "Valid under strong identification; the learner affects power, not size. " *
               "Non-rejection is not evidence of correct specification: violations in " *
               "the span of E[X | Z] (in particular linear violations when just " *
               "identified) are undetectable, and the learner may miss others. The " *
               "result depends on the random sample split."
    else
        β0 = beta0 isa Number ? fill(float(beta0), 1) : Float64.(collect(beta0))
        length(β0) == p ||
            throw(DimensionMismatch("$ctx: beta0 must have $p entries (one per " *
                                    "endogenous regressor)"))
        eng = _iv_rp_weak_engine(prep, aux, learner, weight_update, use_covariates,
                                 clip_quantile, kind, gamma, seed, ctx)
        st = eng.stat_at(β0)
        beta = β0
        null = "E[Y − X'β₀ − C'θ | Z, C] = 0 for some θ, with β₀ = " *
               join(string.(round.(β0; sigdigits=6)), ", ")
        name = "Residual prediction test (weak-IV robust)"
        method = "sample split ($(eng.n_aux) auxiliary / $(eng.n_main) main), $lname " *
                 "weights ($weight_update), residuals at β₀ partialled on covariates, " *
                 "$kind variance, one-sided N(0, 1)"
        note = "Valid whatever the instrument strength; tests the joint hypothesis of " *
               "a well-specified model and β = β₀. Non-rejection is not evidence that " *
               "either holds. The result depends on the random sample split."
    end
    pval = ccdf(Normal(), st.T)
    return DiagnosticTest(name * " (" * _IV_RP_REF * ")", null, st.T, pval;
                          method=method, note=note,
                          details=(T_raw=st.T_raw, var_fraction=st.var_fraction,
                                   floored=st.floored, n_aux=count(aux),
                                   n_main=count(.!aux), beta=beta, learner=lname,
                                   variance=kind, aux_sample=aux))
end

"""
    residual_prediction_confidence_set(data, outcome, endogenous, instruments;
                                       level=0.95, covariates=Symbol[],
                                       learner=ForestLearner(), grid=nothing,
                                       n_grid=101, weight_update=:refit,
                                       aux_fraction=nothing, aux_sample=nothing,
                                       clip_quantile=0.8, gamma=0.05, variance=nothing,
                                       cluster=nothing, use_covariates=true,
                                       rng=Random.default_rng())
        -> WeakIVConfidenceSet

Weak-IV-robust confidence set for the coefficient of one endogenous regressor that is
at the same time a specification test of the linear IV model (Scheidegger,
Londschien and Bühlmann 2025, eq. 30).

The set collects the values ``\\beta_0`` that the weak-IV-robust residual prediction
test of [`residual_prediction_test`](@ref) does not reject,

```math
\\mathcal C = \\{\\beta_0 : p(\\beta_0) \\ge 1 - \\text{level}\\} ,
```

where ``p(\\beta_0)`` is the p-value of the test of
``E[Y - X\\beta_0 - C'\\theta \\mid Z, C] = 0`` for some ``\\theta``. It is the analogue
of the Anderson and Rubin (1949) confidence set, with the learned weight function in
place of the instruments themselves: under a well-specified model it covers the true
coefficient with asymptotic probability at least `level` whatever the strength of the
instruments, so it may be unbounded when identification is weak. Because every
``\\beta_0`` is tested jointly with the specification, **an empty set rejects the linear
IV model** at level ``1 - \\text{level}``. A non-empty set is not evidence of correct
specification, and a set that is narrower than the Wald interval may reflect partial
rejection of the model rather than precision.

One sample split and one learner seed are used for all ``\\beta_0``. The p-value function
is evaluated on `grid` (by default `n_grid` points on the full-sample 2SLS estimate
± 20 standard errors); boundaries are refined by bisection, and the search is extended
beyond the grid until rejection, unbounded sides being reported as `±Inf`. With
`weight_update = :refit` every evaluation refits the learner (the procedure of the
paper, with hyperparameters held fixed as in its Remark 3); `:linear` fits the
learner only ``1 + p`` times and is much faster, exactly so for linear smoothers. The
set depends on the random split; report the seed or pass `aux_sample`.

# Arguments
- `data`, `outcome::Symbol`, `endogenous`, `instruments`: as in
  [`residual_prediction_test`](@ref); exactly one endogenous regressor is allowed.

# Keywords
- `level::Real`: confidence level (default 0.95).
- `grid`: `nothing` (default) or a vector of candidate values of ``\\beta_0``.
- `n_grid::Integer`: number of points of the default grid (default 101, at least 3).
- `weight_update::Symbol`: `:refit` (default), `:linear` or `:fixed`, as in
  [`residual_prediction_test`](@ref).
- `covariates`, `learner`, `aux_fraction`, `aux_sample`, `clip_quantile`, `gamma`,
  `variance`, `cluster`, `use_covariates`, `rng`: as in
  [`residual_prediction_test`](@ref), with the same defaults.

# Returns
- A [`WeakIVConfidenceSet`](@ref): `intervals` (empty when the model is rejected),
  `kind`, `estimate` (the full-sample 2SLS estimate), `critical_value` (the one-sided
  normal critical value); `DrSnow.pvalue(set, β₀)` evaluates the p-value function.

# Examples
```julia
using DrSnow, DataFrames, StableRNGs
rng = StableRNG(1)
n = 1_000
x, z1, z2, v = randn(rng, n), randn(rng, n), randn(rng, n), randn(rng, n)
d = 0.6 .* z1 .+ 0.4 .* z2 .+ 0.5 .* x .+ v
y = d .+ x .+ 0.8 .* z1 .^ 2 .+ 0.6 .* v .+ randn(rng, n)   # misspecified
df = DataFrame(y=y, d=d, z1=z1, z2=z2, x=x)
cs = residual_prediction_confidence_set(df, :y, :d, [:z1, :z2]; covariates=[:x],
                                        learner=ForestLearner(num_trees=100),
                                        weight_update=:linear, rng=StableRNG(4))
isempty(cs.intervals)      # true: the linear IV model is rejected
```

# References
- Scheidegger, C., Londschien, M., & Bühlmann, P. (2025). Machine-learning-powered
  specification testing in linear instrumental variable models. arXiv:2506.12771.
- Anderson, T. W., & Rubin, H. (1949). Estimation of the parameters of a single
  equation in a complete system of stochastic equations. *Annals of Mathematical
  Statistics*, 20(1), 46–63.
- Andrews, I., Stock, J. H., & Sun, L. (2019). Weak instruments in instrumental
  variables regression: Theory and practice. *Annual Review of Economics*, 11,
  727–753.
"""
function residual_prediction_confidence_set(data::AbstractDataFrame, outcome::Symbol,
                                            endogenous, instruments; level::Real=0.95,
                                            covariates=Symbol[], learner=ForestLearner(),
                                            grid=nothing, n_grid::Integer=101,
                                            weight_update::Symbol=:refit,
                                            aux_fraction=nothing, aux_sample=nothing,
                                            clip_quantile::Real=0.8, gamma::Real=0.05,
                                            variance=nothing, cluster=nothing,
                                            use_covariates::Bool=true,
                                            rng::AbstractRNG=Random.default_rng())
    ctx = "residual_prediction_confidence_set"
    0 < level < 1 || throw(ArgumentError("$ctx: level must be in (0, 1)"))
    0 <= clip_quantile <= 1 || throw(ArgumentError("$ctx: clip_quantile must be in [0, 1]"))
    gamma >= 0 || throw(ArgumentError("$ctx: gamma must be non-negative"))
    n_grid >= 3 || throw(ArgumentError("$ctx: n_grid must be at least 3"))
    kind = _iv_rp_variance_kind(variance, cluster, ctx)
    prep = _iv_rp_prepare(data, outcome, endogenous, instruments, covariates, cluster,
                          aux_sample, ctx)
    size(prep.X, 2) == 1 ||
        throw(ArgumentError("$ctx: confidence sets require exactly one endogenous " *
                            "regressor"))
    aux = _iv_rp_split(prep, aux_fraction, rng, ctx)
    seed = task_seeds(rng, 1)[1]
    eng = _iv_rp_weak_engine(prep, aux, learner, weight_update, use_covariates,
                             clip_quantile, kind, gamma, seed, ctx)
    # full-sample 2SLS (partialled) for centring the grid
    MY = vec(_iv_rp_resid(prep.y, prep.Cb))
    MX = vec(_iv_rp_resid(prep.X, prep.Cb))
    MZ = _iv_rp_resid(prep.Z, prep.Cb)
    xh = MZ * (qr(MZ, ColumnNorm()) \ MX)
    b = dot(xh, MY) / dot(xh, MX)
    e = MY .- MX .* b
    se = sqrt(sum(abs2, xh .* e)) / abs(dot(xh, MX))
    scale = isfinite(se) && se > 0 ? se : max(abs(b), 1.0)
    α = 1 - level
    zc = quantile(Normal(), level)
    pfun = β -> ccdf(Normal(), eng.stat_at([float(β)]).T)
    acc = β -> pfun(β) >= α
    g = grid === nothing ? collect(range(b - 20scale, b + 20scale; length=n_grid)) :
        sort!(Float64.(collect(grid)))
    length(g) >= 2 || throw(ArgumentError("$ctx: grid needs at least two points"))
    a = acc.(g)
    bisect = (lo, hi, alo) -> begin    # acceptance differs between lo and hi
        for _ in 1:60
            mid = (lo + hi) / 2
            (mid == lo || mid == hi) && break
            if acc(mid) == alo
                lo = mid
            else
                hi = mid
            end
        end
        (lo + hi) / 2
    end
    # extend beyond the grid edges while accepted
    extend = (edge, dir) -> begin
        step = max(g[end] - g[1], scale)
        prev = edge
        while step < 1e10 * scale
            nxt = edge + dir * step
            if !acc(nxt)
                return dir > 0 ? bisect(prev, nxt, true) : bisect(nxt, prev, false)
            end
            prev = nxt
            step *= 4
        end
        return dir * Inf
    end
    intervals = Tuple{Float64,Float64}[]
    i = 1
    while i <= length(g)
        if !a[i]
            i += 1
            continue
        end
        j = i
        while j < length(g) && a[j + 1]
            j += 1
        end
        lo = i == 1 ? extend(g[1], -1) : bisect(g[i - 1], g[i], false)
        hi = j == length(g) ? extend(g[end], 1) : bisect(g[j], g[j + 1], true)
        push!(intervals, (lo, hi))
        i = j + 1
    end
    lname = _iv_rp_learner_label(learner)
    method = "residual prediction (weak-IV robust, $lname, $weight_update, $kind)"
    return WeakIVConfidenceSet(method, float(level), _iv_set_kind(intervals), intervals,
                               zc, b, pfun)
end
