# Estimation engine of the longitudinal g-methods: treatment mechanism, sequential
# (iterated) outcome regressions with optional targeting (TMLE) or sequentially
# doubly robust pseudo-outcomes (SDR), inverse probability weights, efficient influence
# functions, cross-fitting over units, repetitions and the unit bootstrap.

const _LG_ESTIMATORS = (:sdr, :tmle, :gcomp, :ipw)

struct _LGOptions
    estimator::Symbol
    outcome_learner::Any
    treatment_learner::Any
    g_bound::Float64
    q_bound::Float64
    normalize::Bool
    offset::Float64        # lo / (hi - lo) of the outcome scaling (Horvitz–Thompson)
    parallel::Bool
    context::String
end

_lg_logit(p) = log(p / (1 - p))

"""
Solve the score equation of the intercept-only logistic fluctuation
`Σᵢ wᵢ (yᵢ - expit(logit(qᵢ) + ε)) = 0` for `ε` (observations with `wᵢ = 0` drop
out). The score is strictly decreasing in `ε`, so a bracketed Newton iteration
converges to the unique root; the root is confined to [-30, 30] for the degenerate
cases in which every weighted target lies at one bound.
"""
function _lg_fluctuate(y::AbstractVector, q::AbstractVector, w::AbstractVector)
    idx = findall(>(0), w)
    isempty(idx) && return 0.0
    o = _lg_logit.(q[idx])
    yy = y[idx]
    ww = w[idx]
    score(ε) = sum(ww[j] * (yy[j] - _ml_sigmoid(o[j] + ε)) for j in eachindex(ww))
    lo, hi = -30.0, 30.0
    score(lo) <= 0 && return lo
    score(hi) >= 0 && return hi
    ε = 0.0
    for _ in 1:200
        s = score(ε)
        s > 0 ? (lo = ε) : (hi = ε)
        dd = -sum(ww[j] * (p = _ml_sigmoid(o[j] + ε); p * (1 - p)) for j in eachindex(ww))
        εn = dd < 0 ? ε - s / dd : (lo + hi) / 2
        (lo < εn < hi) || (εn = (lo + hi) / 2)
        abs(εn - ε) <= 1e-13 * (1 + abs(ε)) && return εn
        ε = εn
    end
    return ε
end

"""
Fit the nuisances of one fold: the treatment mechanism and, per regime, the
sequential regressions and cumulative weights. Learners are trained on the units with
`train = true` and predictions are returned for every unit (in sample for the
training units, out of fold for the others).
"""
function _lg_fold(d::_LGData, ys::Vector{Float64}, D::Vector{Matrix{Float64}},
                  XG, XQ, XQD, train::BitVector, seeds::AbstractVector{UInt64},
                  o::_LGOptions)
    n, K = d.n, d.K
    R = length(D)
    ctx = o.context
    phat = Matrix{Float64}(undef, n, K)
    for t in 1:K
        a = view(d.A, :, t)
        atr = a[train]
        (all(==(0), atr) || all(==(1), atr)) &&
            throw(ArgumentError("$ctx: the treatment does not vary in period " *
                                "$(d.times[t]) within a training fold; use fewer folds"))
        pr = fitpredict_proba(o.treatment_learner, XG[t][train, :], atr, XG[t];
                              rng=Random.Xoshiro(seeds[t]))
        length(pr) == n ||
            throw(DimensionMismatch("$ctx: the treatment learner returned $(length(pr)) " *
                                    "predictions for $n units"))
        all(isfinite, pr) ||
            throw(ArgumentError("$ctx: the treatment learner returned non-finite " *
                                "probabilities in period $(d.times[t])"))
        phat[:, t] .= clamp.(pr, 0.0, 1.0)
    end
    out = Vector{NamedTuple}(undef, R)
    for r in 1:R
        Dr = D[r]
        cum = Matrix{Float64}(undef, n, K)
        follow = falses(n, K)
        W = zeros(n, K)
        for i in 1:n
            c = 1.0
            f = true
            for t in 1:K
                c *= Dr[i, t] == 1 ? phat[i, t] : 1 - phat[i, t]
                cum[i, t] = c
                f = f && d.A[i, t] == Dr[i, t]
                follow[i, t] = f
                W[i, t] = f ? 1 / max(c, o.g_bound) : 0.0
            end
        end
        Q = Matrix{Float64}(undef, n, o.estimator === :ipw ? 0 : K + 1)
        if o.estimator !== :ipw
            Q[:, K + 1] .= ys
            tgt = copy(ys)
            for t in K:-1:1
                s = seeds[K + (r - 1) * K + t]
                pred = fitpredict(o.outcome_learner, XQ[t][train, :], tgt[train],
                                  XQD[r][t]; rng=Random.Xoshiro(s))
                length(pred) == n ||
                    throw(DimensionMismatch("$ctx: the outcome learner returned " *
                                            "$(length(pred)) predictions for $n units"))
                all(isfinite, pred) ||
                    throw(ArgumentError("$ctx: the outcome learner returned non-finite " *
                                        "predictions in period $(d.times[t])"))
                if o.estimator !== :sdr
                    pred = clamp.(pred, o.q_bound, 1 - o.q_bound)
                end
                if o.estimator === :tmle
                    ε = _lg_fluctuate(tgt[train], pred[train], W[train, t])
                    pred = _ml_sigmoid.(_lg_logit.(pred) .+ ε)
                end
                Q[:, t] .= pred
                if o.estimator === :sdr
                    @inbounds for i in 1:n
                        acc = Q[i, t]
                        base = t == 1 ? 1.0 : max(cum[i, t - 1], o.g_bound)
                        for s2 in t:K
                            follow[i, s2] || break
                            acc += base / max(cum[i, s2], o.g_bound) *
                                   (Q[i, s2 + 1] - Q[i, s2])
                        end
                        tgt[i] = acc
                    end
                else
                    tgt .= pred
                end
            end
        end
        out[r] = (; Q, W, cum, follow)
    end
    return phat, out
end

"""
One repetition of the estimator on the scaled outcome `ys`: cross-fit over the folds
in `fold` (a single fold with `crossfit = false`), assemble the out-of-fold
quantities and compute the estimates and efficient influence functions.
"""
function _lg_estimate_scaled(d::_LGData, ys::Vector{Float64}, D::Vector{Matrix{Float64}},
                             fold::AbstractVector{Int}, seeds::AbstractMatrix{UInt64},
                             o::_LGOptions, crossfit::Bool)
    n, K = d.n, d.K
    R = length(D)
    nf = crossfit ? maximum(fold) : 1
    XG = [_lg_design(d, t, nothing; current=false) for t in 1:K]
    XQ = [_lg_design(d, t, nothing; current=true) for t in 1:K]
    XQD = [[_lg_design(d, t, Dr; current=true) for t in 1:K] for Dr in D]
    phat = zeros(n, K)
    Q = [zeros(n, o.estimator === :ipw ? 0 : K + 1) for _ in 1:R]
    W = [zeros(n, K) for _ in 1:R]
    cum = [zeros(n, K) for _ in 1:R]
    follow = [falses(n, K) for _ in 1:R]
    run = function (k)
        test = crossfit ? BitVector(fold .== k) : trues(n)
        train = crossfit ? .!test : trues(n)
        ph, outs = _lg_fold(d, ys, D, XG, XQ, XQD, train, view(seeds, :, k), o)
        rows = findall(test)
        phat[rows, :] .= ph[rows, :]
        for r in 1:R
            Q[r][rows, :] .= outs[r].Q[rows, :]
            W[r][rows, :] .= outs[r].W[rows, :]
            cum[r][rows, :] .= outs[r].cum[rows, :]
            follow[r][rows, :] .= outs[r].follow[rows, :]
        end
        return nothing
    end
    if o.parallel && Threads.nthreads() > 1 && nf > 1
        tasks = [Threads.@spawn run(k) for k in 1:nf]
        for t in tasks
            try
                fetch(t)
            catch e
                e isa TaskFailedException ? throw(e.task.exception) : rethrow()
            end
        end
    else
        foreach(run, 1:nf)
    end
    ψ = zeros(R)
    eif = zeros(n, R)
    nb = zeros(Int, R, K)
    for r in 1:R
        Wr, Qr = W[r], Q[r]
        any(follow[r][:, K]) ||
            throw(ArgumentError("$(o.context): no unit follows regime $r through the " *
                                "last period, so E[Y^d] has no support in the data"))
        if o.estimator === :ipw
            wK = Wr[:, K]
            if o.normalize
                sw = sum(wK)
                ψ[r] = dot(wK, ys) / sw
                eif[:, r] .= wK .* (ys .- ψ[r]) ./ (sw / n)
            else
                # Horvitz–Thompson is not affine equivariant: evaluate it on the
                # original scale (ys + offset is y / range), then remove the offset.
                c = o.offset
                ψ[r] = mean(wK .* (ys .+ c)) - c
                eif[:, r] .= wK .* (ys .+ c) .- c .- ψ[r]
            end
        else
            raw = copy(Qr[:, 1])
            for t in 1:K
                raw .+= Wr[:, t] .* (Qr[:, t + 1] .- Qr[:, t])
            end
            ψ[r] = o.estimator === :sdr ? mean(raw) : mean(Qr[:, 1])
            eif[:, r] .= raw .- ψ[r]
        end
        for t in 1:K
            nb[r, t] = count(follow[r][:, t] .& (cum[r][:, t] .< o.g_bound))
        end
    end
    logloss = [_ml_logloss(d.A[:, t], phat[:, t]) for t in 1:K]
    return (; ψ, eif, Q, W, cum, follow, phat, nb, logloss)
end

"""Range used to scale the outcome to [0, 1]: `(lo, hi, scaled)`."""
function _lg_outcome_range(y, bounds, context)
    if bounds !== nothing
        length(bounds) == 2 || throw(ArgumentError("$context: outcome_bounds must have " *
                                                   "two elements"))
        lo, hi = float(bounds[1]), float(bounds[2])
        lo < hi || throw(ArgumentError("$context: outcome_bounds must be increasing"))
        all(v -> lo <= v <= hi, y) ||
            throw(ArgumentError("$context: the outcome lies outside outcome_bounds"))
        return lo, hi, !(lo == 0 && hi == 1)
    end
    lo, hi = extrema(y)
    hi > lo || throw(ArgumentError("$context: the outcome is constant"))
    (lo >= 0 && hi <= 1) && return 0.0, 1.0, false
    return lo, hi, true
end

"""Fold ids per unit: drawn, or taken from a vector/matrix over units or a column."""
function _lg_resolve_folds(data, folds, d::_LGData, n_folds, n_rep, rng, context)
    folds === nothing &&
        return crossfit_folds(d.n, n_folds, n_rep; rng=rng, groups=d.cluster)
    f = folds
    if folds isa Symbol
        df = data isa DataFrame ? data : DataFrame(data)
        require_columns(df, [folds]; context=context)
        f = df[d.rows[:, 1], folds]
        any(ismissing, f) && throw(ArgumentError("$context: missing fold ids"))
        for t in 2:d.K
            df[d.rows[:, t], folds] == f ||
                throw(ArgumentError("$context: the fold column must be constant within " *
                                    "units"))
        end
        f = Int.(f)
    end
    return _ml_resolve_folds(nothing, f, d.n, n_folds, n_rep, rng, nothing, d.cluster;
                             context=context)
end

"""Unit indices of one bootstrap resample (clusters resampled when present)."""
function _lg_resample(d::_LGData, rng::AbstractRNG)
    d.cluster === nothing && return rand(rng, 1:d.n, d.n)
    members = [Int[] for _ in 1:d.n_clusters]
    for (i, c) in enumerate(d.cluster)
        push!(members[c], i)
    end
    idx = Int[]
    for c in rand(rng, 1:d.n_clusters, d.n_clusters)
        append!(idx, members[c])
    end
    return idx
end

"""
    longitudinal_regime(data, outcome, treatment, unit, time; regime,
                        covariates=Symbol[], baseline=Symbol[], estimator=:sdr,
                        outcome_learner=LassoLearner(),
                        treatment_learner=PenalizedLogisticLearner(), crossfit=true,
                        n_folds=5, n_rep=1, folds=nothing, cluster=nothing,
                        g_bound=0.01, q_bound=1e-4, outcome_bounds=nothing,
                        normalize=true, bootstrap=0, rng=Random.default_rng(),
                        parallel=true) -> LongitudinalEstimate
    longitudinal_regime(panel::TreatmentPanel; kwargs...)

Mean counterfactual outcome under one or several treatment regimes in a longitudinal
study with time-varying treatment and time-varying confounding, estimated by the
g-methods of Robins: iterated-regression g-computation, inverse probability of
treatment weighting, longitudinal targeted minimum loss-based estimation, or the
sequentially doubly robust estimator, with optional cross-fitting of machine-learned
nuisance functions.

**Setting and estimand.** Each unit is observed over ``K`` periods. In period ``t``
the time-varying covariates ``L_t`` are measured, then the binary treatment ``A_t``
is chosen, and the outcome ``Y`` is measured after period ``K``; baseline covariates
``W`` are measured before period 1. Write ``H_t = (W, \\bar L_t, \\bar A_{t-1})`` for the
history when ``A_t`` is decided, and let ``d = (d_1, …, d_K)`` be a treatment regime
([`StaticRegime`](@ref) with ``d_t = a_t`` fixed, or [`DynamicRegime`](@ref) with
``d_t = d_t(H_t)``). The estimand for each regime is the mean of the counterfactual
outcome that would be observed if every unit's treatment were set to ``d`` at every
period,

```math
ψ_d = E[Y^{d}] ,
```

and contrasts such as ``ψ_{\\bar 1} - ψ_{\\bar 0}`` are the effects of sustained
strategies ([`regime_contrast`](@ref), [`regime_msm`](@ref)). This is not a
single-period adjustment problem: a covariate such as ``L_2`` is both a confounder
of ``A_2`` and an effect of ``A_1``, so conditioning on it in a regression of ``Y``
on the whole treatment history blocks part of the effect of ``A_1`` and opens a
collider path, while ignoring it leaves ``A_2`` confounded. Robins' (1986) g-formula
resolves this by standardizing sequentially over the covariate process.

**Identification.** Under consistency (``Y = Y^{\\bar A}`` and
``L_t = L_t^{\\bar A_{t-1}}``), sequential exchangeability
(``Y^d ⊥ A_t \\mid H_t = h_t, \\bar A_{t-1} = \\bar d_{t-1}(h_t)`` for every ``t``,
i.e. no unmeasured confounding of each treatment given the observed past) and
positivity (``P(A_t = d_t(h_t) \\mid H_t = h_t) > 0`` whenever ``h_t`` is compatible
with ``d``), ``ψ_d`` equals the g-formula

```math
ψ_d = \\sum_{\\bar l} E[Y \\mid \\bar L = \\bar l, \\bar A = \\bar d(\\bar l)]
\\prod_{t=1}^{K} P\\bigl(L_t = l_t \\mid \\bar L_{t-1} = \\bar l_{t-1},
\\bar A_{t-1} = \\bar d_{t-1}(\\bar l_{t-1})\\bigr),
```

equivalently the iterated conditional expectation ``ψ_d = E[Q_1(H_1)]`` with
``Q_{K+1} = Y`` and ``Q_t(h_t) = E[Q_{t+1}(H_{t+1}) \\mid H_t = h_t, A_t = d_t(h_t)]``
(Bang and Robins 2005), and equivalently the weighted mean
``E[Y \\prod_t 1\\{A_t = d_t(H_t)\\} / g_t(d_t(H_t) \\mid H_t)]`` with
``g_t(a \\mid h_t) = P(A_t = a \\mid H_t = h_t)`` (Robins, Hernán and Brumback 2000).
Sequential exchangeability is untestable; positivity is partly checkable, and
[`regime_weights`](@ref) reports the empirical support of each regime.

**Estimators.** All four work with the outcome scaled to ``[0, 1]`` by its observed
range (or `outcome_bounds`), the treatment mechanism ``\\hat g_t`` estimated by
`treatment_learner` from ``H_t``, the cumulative probability
``\\hat g_{1:t} = \\prod_{s ≤ t} \\hat g_s(d_s \\mid H_s)`` bounded below by `g_bound` (as
in ltmle), and the cumulative weights
``\\hat W_t = 1\\{\\bar A_t = \\bar d_t\\} / \\hat g_{1:t}``.

- `:gcomp` (iterated-regression g-computation, Bang and Robins 2005): for
  ``t = K, …, 1`` regress the previous pseudo-outcome ``\\hat Q_{t+1}`` on
  ``(H_t, A_t)`` over all units with `outcome_learner`, and set ``\\hat Q_t`` to the
  prediction with the treatment history replaced by the regime's actions (bounded to
  ``[q_{bound}, 1 - q_{bound}]``); ``\\hat ψ_d`` is the mean of ``\\hat Q_1``. It is
  consistent when every outcome regression is correctly specified and has no
  valid influence-function variance; use `bootstrap`.
- `:ipw` (inverse probability of treatment weighting): the Hájek estimator
  ``\\sum_i \\hat W_{K,i} Y_i / \\sum_i \\hat W_{K,i}`` (`normalize = true`) or the
  Horvitz–Thompson mean ``n^{-1} \\sum_i \\hat W_{K,i} Y_i``. It is consistent when the
  treatment mechanism is correctly specified; its variance treats the weights as
  known, which is conservative for a correctly specified parametric mechanism
  (Robins, Hernán and Brumback 2000) and is the convention of ltmle.
- `:tmle` (van der Laan and Gruber 2012): the g-computation recursion with, at each
  ``t``, a logistic fluctuation of ``\\hat Q_t`` with intercept ``ε_t`` fitted among the
  followers through ``t`` with weights ``\\hat W_t``, so that the targeted
  ``\\hat Q^*_t`` solves the ``t``-th component of the efficient score; ``\\hat ψ_d`` is
  the mean of ``\\hat Q^*_1``. It is doubly robust (consistent if, at every ``t``,
  either the outcome regression or the treatment mechanism is correct) and
  asymptotically efficient when both are, and remains a substitution estimator
  within the outcome's range. The fluctuation and the regressions follow ltmle
  (Lendle et al. 2017), and with [`FractionalLogitLearner`](@ref),
  [`LogisticLearner`](@ref) and `crossfit = false` the estimates and standard errors
  reproduce ltmle's.
- `:sdr` (sequentially doubly robust, Díaz, Williams, Hoffman and Schenck 2023; the
  one-step estimator of the sequential-regression representation): the regression at
  ``t`` targets the pseudo-outcome
  ``\\sum_{s ≥ t} (\\hat W_s / \\hat W_{t-1}) (\\hat Q_{s+1} - \\hat Q_s) + \\hat Q_t``,
  and
  ``\\hat ψ_d`` is the mean of the resulting efficient influence function at
  ``t = 1``. It is sequentially doubly robust: consistent if, for every ``t``, either
  ``\\hat Q_s`` for all ``s ≥ t`` or ``\\hat g_s`` for all ``s ≤ t`` is consistent, a
  strictly weaker condition than that of `:tmle` (Luedtke et al. 2017; Rotnitzky,
  Robins and Babino 2017), at the price of not being a substitution estimator and of
  regression targets outside ``[0, 1]`` (so `outcome_learner` must accept
  unbounded targets: not [`FractionalLogitLearner`](@ref)). It mirrors `lmtp_sdr`
  (Williams and Díaz 2023) for deterministic regimes.

With `crossfit = true` the units are split into `n_folds` folds (whole clusters when
`cluster` is given); for each fold the treatment mechanism and the whole backward
recursion are fitted on the other folds and evaluated on the held-out units, so that
every unit's nuisance values are out of fold (Chernozhukov et al. 2018). This
removes the Donsker-class conditions on the learners that in-sample fitting would
require, which matters with flexible learners; with parametric learners
`crossfit = false` reproduces the classical estimators. Several `n_rep` repetitions
of the split are aggregated by the median rule of [`DMLEstimate`](@ref).

**Inference.** For `:ipw`, `:tmle` and `:sdr` the covariance of the estimates across
regimes is the sample covariance of the estimated efficient influence functions
``\\hat D_{d,i} = \\sum_t \\hat W_{t,i} (\\hat Q_{t+1,i} - \\hat Q_{t,i}) + \\hat Q_{1,i}
- \\hat ψ_d`` divided by ``n``, with a normal reference; with `cluster`, influence
functions are summed within clusters and a ``t_{G-1}`` reference is used. This is
valid when both nuisances converge at rate ``o(n^{-1/4})`` (for `:tmle` and `:sdr`;
`:ipw` needs a correct treatment mechanism), and it does not account for the bounding
of the cumulative probability, which trades bias for variance under near-violations of
positivity. `bootstrap = B` replaces it by the unit (or cluster) bootstrap, which
refits everything, including the sample splits, in each replicate; it is the only
inference offered for `:gcomp`.

**Practical guidance.** Prefer `:sdr` or `:tmle` with cross-fitting; compare the
four estimators, which agree when all nuisances are well estimated. Inspect
[`regime_weights`](@ref) and [`plot_regime_weights`](@ref): few followers, very
large weights or many bounded probabilities signal a regime with little support, in
which case every estimator extrapolates and the inverse-probability estimators are
unstable. Baseline covariates should include everything that predicts both
treatment and outcome; time-varying covariates must be measured *before* the
period's treatment. The number of regressions grows with ``K``, and so does the
reliance on positivity, since the cumulative probability of following a sustained
regime shrinks with ``K``. For micro-randomized trials, where the randomization
probabilities are known and the target is a proximal excursion effect rather than a
sustained regime, see [`wcls`](@ref); for staggered adoption under parallel trends
rather than sequential exchangeability, see [`did_callaway_santanna`](@ref).

# Arguments
- `data`: a `DataFrame` (or table convertible to one) in long format, one row per
  unit and period, with every unit observed in every period; rows may be in any
  order.
- `outcome::Symbol`: the outcome column, read from each unit's last period (it may
  be missing in earlier rows). Binary 0/1 or numeric.
- `treatment::Symbol`: the binary 0/1 treatment column ``A_t``.
- `unit::Symbol`, `time::Symbol`: the unit identifier and the (sortable) period
  column.

# Keywords
- `regime`: a [`TreatmentRegime`](@ref) or a vector of them; one coefficient is
  estimated per regime, with the joint covariance.
- `covariates::Vector{Symbol} = Symbol[]`: time-varying covariates ``L_t``, measured
  in each period before that period's treatment.
- `baseline::Vector{Symbol} = Symbol[]`: baseline covariates ``W``, constant within
  units.
- `estimator::Symbol = :sdr`: `:sdr`, `:tmle`, `:gcomp` or `:ipw`.
- `outcome_learner = LassoLearner()`: the [`NuisanceLearner`](@ref) for the
  sequential regressions (fitted to targets in ``[0, 1]``, unbounded for `:sdr`).
- `treatment_learner = PenalizedLogisticLearner()`: the learner for
  ``P(A_t = 1 \\mid H_t)``; it must implement [`fitpredict_proba`](@ref).
- `crossfit::Bool = true`: cross-fit the nuisances over units.
- `n_folds::Integer = 5`, `n_rep::Integer = 1`, `folds = nothing`: as in
  [`dml_plr`](@ref), over units; `folds` may be a vector or matrix indexed by the
  units in sorted order of their identifiers, or a column name constant within
  units.
- `cluster::Union{Nothing,Symbol} = nothing`: a cluster column constant within
  units; folds are grouped by cluster and the variance is cluster-robust.
- `g_bound::Real = 0.01`: lower bound on the cumulative treatment probability
  ``\\hat g_{1:t}`` (ltmle's `gbounds[1]`); the weights are at most `1 / g_bound`.
- `q_bound::Real = 1e-4`: the predictions of the outcome regressions are kept within
  ``[q_{bound}, 1 - q_{bound}]`` on the scaled scale (`:gcomp`, `:tmle`), so that
  logits are finite.
- `outcome_bounds = nothing`: known bounds ``(lo, hi)`` of the outcome used for
  scaling instead of its observed range.
- `normalize::Bool = true`: Hájek (`true`) or Horvitz–Thompson (`false`) form of
  the `:ipw` estimator.
- `bootstrap::Integer = 0`: number of unit (cluster) bootstrap replicates; when
  positive, the reported covariance is the bootstrap one.
- `rng::AbstractRNG = Random.default_rng()`: draws the folds, one seed per task and
  the bootstrap resamples, so results do not depend on `parallel` or the number of
  threads.
- `parallel::Bool = true`: fit the folds on separate threads.

# Returns
- [`LongitudinalEstimate`](@ref).

# Examples
```julia
using DrSnow, DataFrames, StableRNGs
rng = StableRNG(1)
n, K = 600, 3
id = repeat(1:n; inner=K)
t = repeat(1:K, n)
w = repeat(randn(rng, n); inner=K)                 # baseline covariate
l = zeros(n * K)
a = zeros(n * K)
y = Vector{Union{Missing,Float64}}(missing, n * K)
for i in 1:n
    a_prev, total = 0.0, 0.0
    for s in 1:K
        r = (i - 1) * K + s
        l[r] = 0.5 * a_prev + 0.3 * w[r] + randn(rng)     # affected by past treatment
        p = 1 / (1 + exp(-(0.5 * l[r] - 0.4 * w[r] + 0.8 * a_prev)))
        a[r] = rand(rng) < p
        total += 0.5 * a[r] + 0.3 * l[r]                  # cumulative effect
        s == K && (y[r] = total + randn(rng))
        a_prev = a[r]
    end
end
df = DataFrame(; id, t, w, l, a, y)
r = longitudinal_regime(df, :y, :a, :id, :t; regime=[always_treat(K), never_treat(K)],
                        covariates=[:l], baseline=[:w], estimator=:sdr,
                        outcome_learner=OLSLearner(),
                        treatment_learner=LogisticLearner(), rng=StableRNG(2))
regime_contrast(r)                      # E[Y^(1,1,1)] - E[Y^(0,0,0)]
regime_weights(r)                       # support of each regime by period
```

# References
- Robins, J. M. (1986). A new approach to causal inference in mortality studies with a
  sustained exposure period—application to control of the healthy worker survivor
  effect. *Mathematical Modelling*, 7(9–12), 1393–1512.
- Robins, J. M., Hernán, M. A., & Brumback, B. (2000). Marginal structural models and
  causal inference in epidemiology. *Epidemiology*, 11(5), 550–560.
- Bang, H., & Robins, J. M. (2005). Doubly robust estimation in missing data and
  causal inference models. *Biometrics*, 61(4), 962–973.
- van der Laan, M. J., & Gruber, S. (2012). Targeted minimum loss based estimation
  of causal effects of multiple time point interventions. *The International Journal
  of Biostatistics*, 8(1), Article 9.
- Díaz, I., Williams, N., Hoffman, K. L., & Schenck, E. J. (2023). Nonparametric
  causal effects based on longitudinal modified treatment policies. *Journal of the
  American Statistical Association*, 118(542), 846–857.
- Luedtke, A. R., Sofrygin, O., van der Laan, M. J., & Carone, M. (2017). Sequential
  double robustness in right-censored longitudinal models. arXiv:1705.02459.
- Rotnitzky, A., Robins, J., & Babino, L. (2017). On the multiply robust estimation
  of the mean of a longitudinal outcome. arXiv:1705.08582.
- Chernozhukov, V., Chetverikov, D., Demirer, M., Duflo, E., Hansen, C., Newey, W., &
  Robins, J. (2018). Double/debiased machine learning for treatment and structural
  parameters. *The Econometrics Journal*, 21(1), C1–C68.
- Hernán, M. A., & Robins, J. M. (2020). *Causal Inference: What If*. Chapman &
  Hall/CRC.
- Lendle, S. D., Schwab, J., Petersen, M. L., & van der Laan, M. J. (2017). ltmle: An
  R package implementing targeted minimum loss-based estimation for longitudinal
  data. *Journal of Statistical Software*, 81(1), 1–21.
- Williams, N. T., & Díaz, I. (2023). lmtp: An R package for estimating the causal
  effects of modified treatment policies. *Observational Studies*, 9(2), 103–122.
"""
function longitudinal_regime(data, outcome::Symbol, treatment::Symbol, unit::Symbol,
                             time::Symbol; regime, covariates=Symbol[], baseline=Symbol[],
                             estimator::Symbol=:sdr, outcome_learner=LassoLearner(),
                             treatment_learner=PenalizedLogisticLearner(),
                             crossfit::Bool=true, n_folds::Integer=5, n_rep::Integer=1,
                             folds=nothing, cluster=nothing, g_bound::Real=0.01,
                             q_bound::Real=1e-4, outcome_bounds=nothing,
                             normalize::Bool=true, bootstrap::Integer=0,
                             rng::AbstractRNG=Random.default_rng(), parallel::Bool=true)
    ctx = "longitudinal_regime"
    estimator in _LG_ESTIMATORS ||
        throw(ArgumentError("$ctx: estimator must be one of $(_LG_ESTIMATORS)"))
    0 < g_bound < 1 || throw(ArgumentError("$ctx: g_bound must be in (0, 1)"))
    0 <= q_bound < 0.5 || throw(ArgumentError("$ctx: q_bound must be in [0, 0.5)"))
    bootstrap >= 0 || throw(ArgumentError("$ctx: bootstrap must be non-negative"))
    bootstrap == 1 && throw(ArgumentError("$ctx: bootstrap needs at least 2 replicates"))
    regimes = regime isa TreatmentRegime ? TreatmentRegime[regime] :
              TreatmentRegime[r for r in regime]
    isempty(regimes) && throw(ArgumentError("$ctx: at least one regime is required"))
    all(r -> r isa TreatmentRegime, regimes) ||
        throw(ArgumentError("$ctx: regime must be a TreatmentRegime or a vector of them"))
    labels = _lg_label.(regimes)
    allunique(labels) ||
        throw(ArgumentError("$ctx: regime labels must be unique; got $(labels)"))
    crossfit || n_rep == 1 ||
        throw(ArgumentError("$ctx: n_rep > 1 requires crossfit = true"))
    d = _lg_data(data, outcome, treatment, unit, time; covariates=covariates,
                 baseline=baseline, cluster=cluster, context=ctx)
    D = [_lg_actions(r, d, ctx) for r in regimes]
    lo, hi, scaled = _lg_outcome_range(d.y, outcome_bounds, ctx)
    ys = scaled ? (d.y .- lo) ./ (hi - lo) : copy(d.y)
    F = if crossfit
        _ml_check_common(n_folds, n_rep)
        _lg_resolve_folds(data, folds, d, n_folds, n_rep, rng, ctx)
    else
        ones(Int, d.n, 1)
    end
    R, K = length(regimes), d.K
    nf = crossfit ? maximum(F) : 1
    nseeds = K + R * K
    seeds = reshape(task_seeds(rng, nseeds * nf * size(F, 2)), nseeds, nf, size(F, 2))
    o = _LGOptions(estimator, outcome_learner, treatment_learner, Float64(g_bound),
                   Float64(q_bound), normalize, scaled ? lo / (hi - lo) : 0.0, parallel,
                   ctx)
    reps = [_lg_estimate_scaled(d, ys, D, F[:, r], seeds[:, :, r], o, crossfit)
            for r in 1:size(F, 2)]
    range = hi - lo
    all_coef = reduce(hcat, [rep.ψ for rep in reps]) .* range .+ lo
    if estimator === :gcomp
        θ = [median(all_coef[j, :]) for j in 1:R]
        V = nothing
        eif = nothing
    else
        vcovs = [_if_vcov(rep.eif, d.cluster)[1] for rep in reps]
        θs, Vs = _ml_aggregate(reduce(hcat, [rep.ψ for rep in reps]), vcovs)
        θ = θs .* range .+ lo
        V = Vs .* range^2
        eif = reps[1].eif .* range
    end
    boot = zeros(bootstrap, R)
    for b in 1:bootstrap
        idx = _lg_resample(d, rng)
        db = _lg_subset(d, idx)
        groups = d.cluster === nothing ? idx : d.cluster[idx]
        Fb = crossfit ? crossfit_folds(length(idx), nf, 1; rng=rng, groups=groups) :
             ones(Int, length(idx), 1)
        sb = reshape(task_seeds(rng, nseeds * nf), nseeds, nf)
        rb = _lg_estimate_scaled(db, ys[idx], [Dr[idx, :] for Dr in D], Fb[:, 1], sb, o,
                                 crossfit)
        boot[b, :] .= rb.ψ .* range .+ lo
    end
    Vrep = bootstrap > 0 ? Matrix(Symmetric(cov(boot; dims=1))) : V
    first = reps[1]
    Qs = [estimator === :ipw ? zeros(d.n, 0) : Q .* range .+ lo for Q in first.Q]
    learners = [:outcome => _ml_learner_name(outcome_learner),
                :treatment => _ml_learner_name(treatment_learner)]
    return LongitudinalEstimate(estimator, regimes, labels, θ, Vrep, V, eif, boot,
                                all_coef, Qs, first.W, first.cum, first.follow,
                                first.phat, first.nb, first.logloss, F, d.cluster,
                                d.n_clusters, d.n, K, d.ids, d.times, learners,
                                Float64(g_bound), Float64(q_bound), (lo, hi), normalize,
                                vec(seeds))
end

function longitudinal_regime(panel::TreatmentPanel; covariates=panel.covariates,
                             kwargs...)
    return longitudinal_regime(panel.data, panel.outcome, panel.treatment, panel.unit_id,
                               panel.time; covariates=covariates, kwargs...)
end
