# Instrument-strength diagnostics for machine-learned (cross-fitted) first stages, and
# weak-instrument-robust (Anderson–Rubin-type) inference for DML IV estimators.
#
# Neyman orthogonality makes the DML IV scores insensitive to first-order errors in
# the nuisance functions, but it does nothing for weak identification: the DML IV
# estimator is still a ratio whose denominator is the orthogonalized first stage, and
# its normal approximation fails when that first stage is small relative to its
# sampling error. The diagnostics below measure the first stage on the out-of-fold
# residualized data; the DML-AR test and set are valid whatever its strength.

"""
    MLFirstStage

Cross-fitted first-stage strength diagnostics of a double/debiased machine-learning IV
model (see [`ml_first_stage`](@ref)).

The object measures the strength of the instrument after the covariates have been
partialled out by machine-learned, cross-fitted nuisance functions, which is the
first stage that actually identifies the DML IV estimators [`dml_pliv`](@ref) and
[`dml_iivm`](@ref). For the partially linear IV model it holds the coefficients of the
out-of-fold treatment residual on the out-of-fold instrument residuals and the usual
family of F statistics; for the interactive IV model it holds the orthogonalized
first-stage effect of the binary instrument, which equals the complier share under
monotonicity. With repeated cross-fitting the statistics are aggregated over
repetitions by the median. As for linear 2SLS, a large statistic speaks only to
relevance, not to the validity of the instrument.

# Fields
- `model::Symbol`: `:pliv` (partially linear IV) or `:iivm` (interactive IV / LATE).
- `treatment::String`, `instruments::Vector{String}`: variable names.
- `estimate::Vector{Float64}`, `vcov::Matrix{Float64}`: for `:pliv`, the coefficients
  of the out-of-fold treatment residual ``D - \\hat r(X)`` on the instrument residuals
  ``Z - \\hat m(X)`` (no intercept) and their robust covariance; for `:iivm`, the
  orthogonal-score (DML) estimate of the first-stage effect ``E[r_1(X) - r_0(X)]``,
  with ``r_z(X) = P(D = 1 \\mid Z = z, X)``, and its variance.
  Aggregated over repetitions by the median rule of Chernozhukov et al. (2018).
- `F::Float64`: the robust (HC1 or cluster) Wald F of the first stage, median over
  repetitions; for `:iivm`, the squared t-statistic of the first-stage effect.
- `F_homoskedastic`, `partial_r2`: the conventional F and the partial ``R^2`` of the
  instrument residuals (`:pliv` only, otherwise `nothing`).
- `effective_F::Float64`: the Montiel Olea–Pflueger effective F (equal to `F` with one
  instrument), median over repetitions.
- `op_critical_values`: its simplified 5% critical values `(tau_5, tau_10, tau_20,
  tau_30, K_eff)`, taken from the repetition with the median effective F.
- `optimal_instrument_F`: for `:pliv` with several instruments, the robust F of the
  treatment residual on a cross-fitted linear optimal instrument (projection
  coefficients estimated on the other folds); otherwise `nothing`.
- `all_F::Vector{Float64}`: `F` in each repetition.
- `n::Int`: number of observations; `n_clusters::Int`: number of clusters (0 without
  clustering); `vcov_type::String`: covariance estimator.

# References
- Chernozhukov, V., Chetverikov, D., Demirer, M., Duflo, E., Hansen, C., Newey, W., &
  Robins, J. (2018). Double/debiased machine learning for treatment and structural
  parameters. *The Econometrics Journal*, 21(1), C1–C68.
- Montiel Olea, J. L., & Pflueger, C. (2013). A robust test for weak instruments.
  *Journal of Business & Economic Statistics*, 31(3), 358–369.
"""
struct MLFirstStage
    model::Symbol
    treatment::String
    instruments::Vector{String}
    estimate::Vector{Float64}
    vcov::Matrix{Float64}
    F::Float64
    F_homoskedastic::Union{Nothing,Float64}
    partial_r2::Union{Nothing,Float64}
    effective_F::Float64
    op_critical_values::NamedTuple
    optimal_instrument_F::Union{Nothing,Float64}
    all_F::Vector{Float64}
    n::Int
    n_clusters::Int
    vcov_type::String
end

StatsAPI.nobs(r::MLFirstStage) = r.n

function Base.show(io::IO, ::MIME"text/plain", r::MLFirstStage)
    println(io, "Cross-fitted first stage (", r.model === :pliv ?
                "partially linear IV" : "interactive IV", "), treatment ", r.treatment,
            ", instrument", length(r.instruments) == 1 ? " " : "s ",
            join(r.instruments, ", "))
    println(io, "Observations: ", r.n, r.n_clusters > 0 ?
                " ($(r.n_clusters) clusters)" : "", "; covariance: ", r.vcov_type)
    se = sqrt.(max.(diag(r.vcov), 0.0))
    for (j, nm) in enumerate(r.instruments)
        lbl = r.model === :pliv ? "coefficient on residualized $nm" :
              "first-stage effect of $nm (complier share)"
        @printf(io, "  %s: %.4g (se %.3g)\n", lbl, r.estimate[j], se[j])
    end
    @printf(io, "Robust F = %.4g", r.F)
    length(r.all_F) > 1 && @printf(io, " (median of %d repetitions; range %.4g–%.4g)",
                                   length(r.all_F), minimum(r.all_F), maximum(r.all_F))
    println(io)
    r.F_homoskedastic === nothing ||
        @printf(io, "Homoskedastic F = %.4g, partial R² = %.4g\n", r.F_homoskedastic,
                r.partial_r2)
    @printf(io, "Effective F = %.4g (Olea–Pflueger critical value, τ = 10%%: %.4g)\n",
            r.effective_F, r.op_critical_values.tau_10)
    r.optimal_instrument_F === nothing ||
        @printf(io, "Cross-fitted optimal-instrument F = %.4g\n", r.optimal_instrument_F)
    println(io, "Reference thresholds: F ≥ 10 (Staiger–Stock rule of thumb); effective ",
            "F above the Olea–Pflueger value; F ≥ 104.7 for a 5% t-test with ",
            "unadjusted critical values (Lee et al. 2022).")
    println(io, "Note: orthogonal (DML) scores do not protect against weak instruments; ",
            "use dml_weak_iv_confidence_set for identification-robust inference.")
end

Base.show(io::IO, r::MLFirstStage) =
    @printf(io, "MLFirstStage(%s, F = %.4g)", r.model, r.F)

# ---------------------------------------------------------------------------
# Per-repetition statistics
# ---------------------------------------------------------------------------

"""HC1 or CR1 covariance of OLS without intercept (FixedEffectModels corrections)."""
function _iv_mlfs_ols_vcov(X::AbstractMatrix, e::AbstractVector, cluster, G)
    n, k = size(X)
    bread = inv(Symmetric(X' * X))
    S = X .* e
    meat = if cluster === nothing
        (S' * S) .* (n / (n - k))
    else
        C = zeros(G, k)
        for i in 1:n
            @views C[cluster[i], :] .+= S[i, :]
        end
        (C' * C) .* ((n - 1) / (n - k) * G / (G - 1))
    end
    return Matrix(Symmetric(bread * meat * bread))
end

function _iv_mlfs_pliv_rep(w::AbstractVector, V::AbstractMatrix, fold::AbstractVector,
                           cluster, G)
    n, L = size(V)
    π = V \ w
    e = w - V * π
    Vπ = _iv_mlfs_ols_vcov(V, e, cluster, G)
    F = _iv_wald_F(π, Vπ)
    Fh = (dot(V * π, V * π) / L) / (sum(abs2, e) / (n - L))
    r2 = 1 - sum(abs2, e) / sum(abs2, w)
    Q = Matrix(Symmetric(V' * V))
    effF, opcv = _iv_effective_F(π, Vπ, Q)
    Fopt = nothing
    if L > 1
        ṽ = zeros(n)
        for k in unique(fold)
            te = fold .== k
            tr = .!te
            ṽ[te] = V[te, :] * (V[tr, :] \ w[tr])
        end
        a = dot(ṽ, w) / dot(ṽ, ṽ)
        ea = w .- a .* ṽ
        va = _iv_mlfs_ols_vcov(reshape(ṽ, :, 1), ea, cluster, G)[1, 1]
        Fopt = a^2 / va
    end
    return (est=π, V=Vπ, F=F, Fh=Fh, r2=r2, effF=effF, opcv=opcv, Fopt=Fopt)
end

function _iv_mlfs_iivm_rep(φ::AbstractVector, cluster, G)
    n = length(φ)
    est = mean(φ)
    c = φ .- est
    v = if cluster === nothing
        sum(abs2, c) / n^2
    else
        S = zeros(G)
        for i in 1:n
            S[cluster[i]] += c[i]
        end
        sum(abs2, S) / n^2 * G / (G - 1)
    end
    F = est^2 / v
    Vm = fill(v, 1, 1)
    _, opcv = _iv_effective_F([est], Vm, fill(1.0, 1, 1))
    return (est=[est], V=Vm, F=F, Fh=nothing, r2=nothing, effF=F, opcv=opcv, Fopt=nothing)
end

function _iv_mlfs_assemble(model, treatment, instruments, reps, n, cluster, G)
    R = length(reps)
    all_coef = reduce(hcat, [r.est for r in reps])
    θ, V = _ml_aggregate(all_coef, [r.V for r in reps])
    med(f) = median([f(r) for r in reps])
    Fh = reps[1].Fh === nothing ? nothing : med(r -> r.Fh)
    r2 = reps[1].r2 === nothing ? nothing : med(r -> r.r2)
    Fopt = reps[1].Fopt === nothing ? nothing : med(r -> r.Fopt)
    allF = [r.F for r in reps]
    # critical values of the repetition whose effective F is the median one
    effs = [r.effF for r in reps]
    im = sortperm(effs)[cld(R, 2)]
    vt = cluster === nothing ? "heteroskedasticity-robust (HC1)" :
         "cluster-robust ($G clusters)"
    return MLFirstStage(model, string(treatment), string.(instruments), θ, V,
                        median(allF), Fh, r2, median(effs), reps[im].opcv, Fopt, allF, n,
                        cluster === nothing ? 0 : G, vt)
end

# `DMLEstimate` is defined in the ml area (loaded after this one), so methods taking a
# fitted DML estimate are untyped and check the type at run time.
function _iv_require_dml(r, ctx)
    r isa DMLEstimate ||
        throw(ArgumentError("$ctx: expected a DMLEstimate (from dml_pliv or dml_iivm), " *
                            "got $(typeof(r))"))
    return nothing
end

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

"""
    ml_first_stage(data, treatment, instrument; covariates=Symbol[], model=:pliv,
                   treatment_learner=nothing, instrument_learner=nothing, trim=0.01,
                   n_folds=5, n_rep=1, folds=nothing, cluster=nothing, stratify=true,
                   rng=Random.default_rng(), parallel=true) -> MLFirstStage
    ml_first_stage(r::DMLEstimate, data=nothing; instrument=nothing) -> MLFirstStage

Instrument-strength diagnostics for IV models whose nuisance functions are learned by
machine learning with cross-fitting.

Double/debiased machine learning (Chernozhukov et al. 2018) estimates IV parameters
from Neyman-orthogonal scores, which makes the estimator insensitive to first-order
errors in the learned nuisance functions. Orthogonality does nothing, however, for
weak identification: the DML IV estimator of [`dml_pliv`](@ref) or [`dml_iivm`](@ref)
is still a ratio whose denominator is the orthogonalized first stage, and its normal
approximation fails when that first stage is small relative to its sampling error,
exactly as for 2SLS (Staiger and Stock 1997; Andrews, Stock and Sun 2019). This
function measures the strength of the instrument on the same residualized scale that
the DML estimator uses.

With `model = :pliv` (the partially linear IV model ``Y = D\\theta + g(X) + \\varepsilon``,
``E[\\varepsilon \\mid Z, X] = 0``), the out-of-fold treatment residual ``D - \\hat r(X)``,
``r(X) = E[D \\mid X]``, is regressed without intercept on the out-of-fold instrument
residuals ``Z - \\hat m(X)``, ``m(X) = E[Z \\mid X]``. The function reports the robust
(HC1 or cluster) Wald F, the homoskedastic F, the partial ``R^2`` and the Montiel
Olea and Pflueger (2013) effective F with its simplified critical values; with several
instruments it also reports the F of a *cross-fitted* linear optimal instrument, whose
projection coefficients are estimated on the other folds, because the in-sample
projection used by [`dml_pliv`](@ref) with several instruments would overstate
strength. With `model = :iivm` (binary ``Z`` and ``D``, the LATE setting), the
statistic is based on the orthogonalized first stage

```math
\\theta_{FS} = E\\Big[r_1(X) - r_0(X) + \\frac{Z (D - r_1(X))}{m(X)}
             - \\frac{(1 - Z)(D - r_0(X))}{1 - m(X)}\\Big] ,
```

with ``r_z(X) = P(D = 1 \\mid Z = z, X)`` and ``m(X) = P(Z = 1 \\mid X)``, the DML
estimate of the average effect of ``Z`` on ``D``, which is the complier share under
monotonicity; `F` is ``(\\hat\\theta_{FS}/\\widehat{\\text{se}})^2``.

The printed reference thresholds are not applied: ``F \\ge 10`` (Staiger and Stock
1997), an effective F above the Montiel Olea–Pflueger critical value for a worst-case
bias of 10% of the benchmark, and ``F \\ge 104.7`` for a 5% t-test with conventional
critical values (Lee, McCrary, Moreira and Porter 2022). These thresholds were derived
for linear IV with a fixed number of instruments; for DML they are heuristics. When
the first stage is weak, use [`dml_weak_iv_test`](@ref) and
[`dml_weak_iv_confidence_set`](@ref), whose size does not depend on instrument
strength, instead of the DML Wald interval. The second method reuses the nuisance
predictions, folds and clusters stored in a fitted [`DMLEstimate`](@ref), so that the
diagnostics refer to exactly the nuisance functions behind the reported estimate.

# Arguments
- `data`: a `DataFrame` with the treatment, instrument(s) and covariates.
- `treatment::Symbol`: the endogenous treatment ``D`` (binary for `:iivm`).
- `instrument`: the instrument ``Z``, a `Symbol`, or a vector of `Symbol`s for `:pliv`;
  one binary instrument for `:iivm`.
- `r::DMLEstimate`: alternatively, a fit of [`dml_pliv`](@ref) or [`dml_iivm`](@ref).
  For `:iivm` no data are needed; for `:pliv`, `data` must contain the treatment and
  instrument columns, with the same rows as the fit.

# Keywords
- `covariates::Vector{Symbol}`: the controls ``X`` (default none).
- `model::Symbol`: `:pliv` (default) or `:iivm`.
- `treatment_learner`, `instrument_learner`: [`NuisanceLearner`](@ref)s for ``r``
  (``r_z``) and ``m``. The default `nothing` means `LassoLearner()` for `:pliv` and
  `PenalizedLogisticLearner()` for `:iivm`.
- `trim::Real`: clipping of the instrument propensity ``m`` to ``[t, 1 - t]`` for
  `:iivm` (default 0.01).
- `n_folds::Integer`, `n_rep::Integer`, `folds`: number of cross-fitting folds (default
  5), repetitions (default 1) and optional fixed fold assignments, as in
  [`dml_pliv`](@ref). With the same `folds` and deterministic learners the nuisances
  coincide with those of the DML estimator.
- `cluster`: a column for cluster-robust statistics and cluster-level folds (default
  `nothing`).
- `stratify::Bool`: for `:iivm`, stratify the folds by ``(Z, D)`` (default `true`).
- `rng::AbstractRNG`: draws the folds and the learners' seeds; `parallel::Bool`:
  cross-fit the nuisances in parallel (default `true`).
- `instrument` (second method): the instrument column name for a `dml_pliv` fit with a
  single instrument, or a label for a `dml_iivm` fit.

# Returns
- An [`MLFirstStage`](@ref) with `F`, `effective_F`, `op_critical_values` and the
  first-stage estimate.

# Examples
```julia
using DrSnow, DataFrames, StableRNGs
rng = StableRNG(1)
n = 1_000
x1, x2 = randn(rng, n), randn(rng, n)
z = Float64.(rand(rng, n) .< 1 ./ (1 .+ exp.(-0.5 .* x1)))   # Z depends on X
u = rand(rng, n)
d = Float64.((u .< 0.2) .| ((u .< 0.5) .& (z .== 1)))       # 30% compliers
y = d .+ x1 .+ 0.5 .* x2 .+ 0.5 .* (u .< 0.2) .+ randn(rng, n)
df = DataFrame(y=y, d=d, z=z, x1=x1, x2=x2)
ml_first_stage(df, :d, :z; covariates=[:x1, :x2], rng=StableRNG(2))
r = dml_iivm(df, :y, :d, :z; covariates=[:x1, :x2], rng=StableRNG(3))
ml_first_stage(r; instrument=:z)          # complier share and its F
```

# References
- Chernozhukov, V., Chetverikov, D., Demirer, M., Duflo, E., Hansen, C., Newey, W., &
  Robins, J. (2018). Double/debiased machine learning for treatment and structural
  parameters. *The Econometrics Journal*, 21(1), C1–C68.
- Staiger, D., & Stock, J. H. (1997). Instrumental variables regression with weak
  instruments. *Econometrica*, 65(3), 557–586.
- Montiel Olea, J. L., & Pflueger, C. (2013). A robust test for weak instruments.
  *Journal of Business & Economic Statistics*, 31(3), 358–369.
- Andrews, I., Stock, J. H., & Sun, L. (2019). Weak instruments in instrumental
  variables regression: Theory and practice. *Annual Review of Economics*, 11,
  727–753.
- Lee, D. S., McCrary, J., Moreira, M. J., & Porter, J. (2022). Valid t-ratio
  inference for IV. *American Economic Review*, 112(10), 3260–3290.
"""
function ml_first_stage(data, treatment::Symbol, instrument; covariates=Symbol[],
                        model::Symbol=:pliv, treatment_learner=nothing,
                        instrument_learner=nothing, trim::Real=0.01,
                        n_folds::Integer=5, n_rep::Integer=1, folds=nothing,
                        cluster=nothing, stratify::Bool=true,
                        rng::AbstractRNG=Random.default_rng(), parallel::Bool=true)
    ctx = "ml_first_stage"
    model in (:pliv, :iivm) || throw(ArgumentError("$ctx: model must be :pliv or :iivm"))
    zs = _as_symbols(instrument)
    isempty(zs) && throw(ArgumentError("$ctx: at least one instrument is required"))
    if model === :pliv
        tl = treatment_learner === nothing ? LassoLearner() : treatment_learner
        il = instrument_learner === nothing ? LassoLearner() : instrument_learner
        covs, X, cid, G, F = _ml_setup(data, vcat(treatment, zs), covariates, cluster,
                                       folds, n_folds, n_rep, rng, nothing; context=ctx)
        d = _ml_column(data, treatment; context=ctx)
        Z = _ml_matrix(data, zs; context=ctx)
        L = length(zs)
        K, R = maximum(F), size(F, 2)
        seeds = _ml_seeds(rng, K, 1 + L, R)
        reps = map(1:R) do r
            specs = [_MLNuisance(:ml_r, tl, d, X, false)]
            for j in 1:L
                push!(specs, _MLNuisance(Symbol("ml_m_", zs[j]), il, Z[:, j], X, false))
            end
            P = _ml_crossfit(specs, F[:, r], view(seeds, :, :, r); parallel=parallel,
                             context=ctx)
            _iv_mlfs_pliv_rep(d .- P[:, 1], Z .- P[:, 2:end], F[:, r], cid, G)
        end
        return _iv_mlfs_assemble(:pliv, treatment, zs, reps, nrow(data), cid, G)
    end
    length(zs) == 1 || throw(ArgumentError("$ctx: model = :iivm needs one binary " *
                                           "instrument"))
    z_sym = zs[1]
    tl = treatment_learner === nothing ? PenalizedLogisticLearner() : treatment_learner
    il = instrument_learner === nothing ? PenalizedLogisticLearner() : instrument_learner
    trim = _ml_check_trim(trim)
    require_columns(data, [treatment, z_sym]; context=ctx)
    d = _ml_column(data, treatment; context=ctx)
    z = _ml_column(data, z_sym; context=ctx)
    _ml_check_binary_col(d, treatment, ctx)
    _ml_check_binary_col(z, z_sym, ctx)
    covs, X, cid, G, F = _ml_setup(data, [treatment, z_sym], covariates, cluster, folds,
                                   n_folds, n_rep, rng, stratify ? 2 .* z .+ d : nothing;
                                   context=ctx)
    K, R = maximum(F), size(F, 2)
    seeds = _ml_seeds(rng, K, 3, R)
    z1 = BitVector(z .== 1)
    reps = map(1:R) do r
        specs = [_MLNuisance(:ml_m, il, z, X, true),
                 _MLNuisance(:ml_r0, tl, d, X, true, .!z1),
                 _MLNuisance(:ml_r1, tl, d, X, true, z1)]
        P = _ml_crossfit(specs, F[:, r], view(seeds, :, :, r); parallel=parallel,
                         context=ctx)
        m, r0, r1 = P[:, 1], P[:, 2], P[:, 3]
        _ml_clip!(m, trim)
        φ = r1 .- r0 .+ z .* (d .- r1) ./ m .- (1 .- z) .* (d .- r0) ./ (1 .- m)
        _iv_mlfs_iivm_rep(φ, cid, G)
    end
    return _iv_mlfs_assemble(:iivm, treatment, zs, reps, nrow(data), cid, G)
end

function ml_first_stage(r, data=nothing; instrument=nothing)
    ctx = "ml_first_stage"
    _iv_require_dml(r, ctx)
    n, R, _ = size(r.psi)
    G = r.n_clusters
    if r.model === :iivm
        reps = [_iv_mlfs_iivm_rep(-r.psi_a[:, k, 1], r.cluster, G) for k in 1:R]
        zname = instrument === nothing ? "Z" : string(instrument)
        return _iv_mlfs_assemble(:iivm, r.names[1], [zname], reps, n, r.cluster, G)
    end
    r.model === :pliv ||
        throw(ArgumentError("$ctx: needs a dml_pliv or dml_iivm estimate " *
                            "(got :$(r.model))"))
    data === nothing &&
        throw(ArgumentError("$ctx: `data` is required for a dml_pliv estimate"))
    nrow(data) == n || throw(DimensionMismatch("$ctx: data has $(nrow(data)) rows, the " *
                                               "estimate $n"))
    mkeys = [nm for (nm, _) in r.learners if startswith(string(nm), "ml_m")]
    zs = if mkeys == [:ml_m]
        instrument === nothing &&
            throw(ArgumentError("$ctx: pass `instrument` (the column used by dml_pliv)"))
        _as_symbols(instrument)
    else
        [Symbol(string(k)[6:end]) for k in mkeys]
    end
    length(zs) == length(mkeys) ||
        throw(ArgumentError("$ctx: `instrument` must list $(length(mkeys)) column(s)"))
    d = _ml_column(data, Symbol(r.names[1]); context=ctx)
    Z = _ml_matrix(data, zs; context=ctx)
    reps = map(1:R) do k
        w = d .- r.predictions[:ml_r][:, k, 1]
        M = reduce(hcat, [r.predictions[m][:, k, 1] for m in mkeys])
        _iv_mlfs_pliv_rep(w, Z .- M, r.folds[:, k], r.cluster, G)
    end
    return _iv_mlfs_assemble(:pliv, r.names[1], zs, reps, n, r.cluster, G)
end

# ---------------------------------------------------------------------------
# DML Anderson–Rubin test and confidence set
# ---------------------------------------------------------------------------

"""Sums defining the AR statistic of repetition `k` (score ψ(θ) = ψ_b + θ ψ_a)."""
function _iv_dmlar_parts(r, k::Int)
    a = r.psi_a[:, k, 1]
    b = r.psi_b[:, k, 1]
    if r.cluster === nothing
        Sa, Sb, c = a, b, 1.0
    else
        G = r.n_clusters
        Sa, Sb = zeros(G), zeros(G)
        for i in eachindex(a)
            Sa[r.cluster[i]] += a[i]
            Sb[r.cluster[i]] += b[i]
        end
        c = G / (G - 1)
    end
    return (A=sum(a), B=sum(b), Qaa=c * dot(Sa, Sa), Qab=c * dot(Sa, Sb),
            Qbb=c * dot(Sb, Sb))
end

function _iv_dmlar_stat(p, θ::Real)
    if isinf(θ)
        return p.Qaa > 0 ? p.A^2 / p.Qaa : 0.0
    end
    S = p.B + θ * p.A
    V = p.Qbb + 2θ * p.Qab + θ^2 * p.Qaa
    V > 0 || return S == 0 ? 0.0 : Inf
    return S^2 / V
end

_iv_dmlar_dist(r) = r.cluster === nothing ? Chisq(1) :
                                 FDist(1, r.n_clusters - 1)

function _iv_dmlar_check(r, ctx)
    _iv_require_dml(r, ctx)
    r.model in (:pliv, :iivm) ||
        throw(ArgumentError("$ctx: needs a dml_pliv or dml_iivm estimate " *
                            "(got :$(r.model))"))
    if r.model === :pliv
        r.score === :partialling_out ||
            throw(ArgumentError("$ctx: requires score = :partialling_out (the IV-type " *
                                "score's nuisance g depends on a preliminary estimate " *
                                "of θ and is not valid under the null)"))
        any(p -> first(p) === :ml_m, r.learners) ||
            throw(ArgumentError("$ctx: dml_pliv with several instruments uses an " *
                                "in-sample optimal-instrument projection that depends " *
                                "on the treatment; fit one instrument (or an index of " *
                                "the instruments) for weak-IV-robust inference"))
    end
    return nothing
end

"""Upper median: the ⌊R/2⌋ + 1-th smallest value."""
_iv_upper_median(v) = sort(v)[fld(length(v), 2) + 1]

"""
    dml_weak_iv_test(r::DMLEstimate; beta0=0.0) -> DiagnosticTest

Weak-instrument-robust (Anderson–Rubin-type) test of a hypothesized value of the
parameter of a DML IV model.

The DML estimators of the partially linear IV model ([`dml_pliv`](@ref)) and of the LATE
in the interactive IV model ([`dml_iivm`](@ref)) solve an orthogonal moment condition
whose score is linear in the parameter, ``\\psi(\\theta) = \\psi_b + \\theta \\psi_a``,
where ``\\psi_a`` is (minus) the orthogonalized first stage. The Wald t-test divides by
the estimated first stage and inherits the weak-instrument problem of 2SLS. The score
test with the null imposed does not: because none of the cross-fitted nuisance functions
depends on ``\\theta``, ``E[\\psi(\\theta_0)] = 0`` at the true value whatever the
strength of the instrument, and the statistic

```math
AR(\\beta_0) = \\frac{\\big(\\sum_i \\psi_i(\\beta_0)\\big)^2}{\\sum_i \\psi_i(\\beta_0)^2}
```

(cluster sums and a factor ``G/(G - 1)`` with clustering) is asymptotically
``\\chi^2(1)`` under ``H_0: \\theta = \\beta_0``, or referred to ``F(1, G - 1)`` with
``G`` clusters. This is the logic of the Anderson and Rubin (1949) test transposed to
orthogonal scores, as in the post-regularization instrumental-variable inference of
Chernozhukov, Hansen and Spindler (2015) and in the identification-robust inference for
the LATE with high-dimensional covariates of Ma (2026). With repeated cross-fitting the
reported p-value is the upper median of the per-repetition p-values and `statistic` is
the median statistic.

The test is robust to weak identification but not to invalid instruments: it
maintains conditional independence and exclusion (and monotonicity for the LATE),
and its non-rejection is not evidence for ``\\beta_0``. With the IV-type score of
`dml_pliv` the nuisance ``g`` depends on a preliminary estimate of ``\\theta`` and is not
valid under the null, so the partialling-out score is required; `dml_pliv` with
several instruments is not supported because its in-sample optimal-instrument
projection depends on the treatment (use one instrument or an index of the
instruments).

# Arguments
- `r::DMLEstimate`: a fit of [`dml_pliv`](@ref) with one instrument and
  `score = :partialling_out`, or of [`dml_iivm`](@ref).

# Keywords
- `beta0::Real`: the hypothesized value of the parameter (default 0).

# Returns
- A [`DiagnosticTest`](@ref); `details` holds the per-repetition `statistics` and
  `pvalues`.

# Examples
```julia
using DrSnow, DataFrames, StableRNGs
rng = StableRNG(1)
n = 1_000
x1, x2 = randn(rng, n), randn(rng, n)
z = Float64.(rand(rng, n) .< 1 ./ (1 .+ exp.(-0.5 .* x1)))
u = rand(rng, n)
d = Float64.((u .< 0.2) .| ((u .< 0.5) .& (z .== 1)))
y = d .+ x1 .+ 0.5 .* x2 .+ 0.5 .* (u .< 0.2) .+ randn(rng, n)
df = DataFrame(y=y, d=d, z=z, x1=x1, x2=x2)
r = dml_pliv(df, :y, :d, :z; covariates=[:x1, :x2], rng=StableRNG(4))
dml_weak_iv_test(r; beta0=0.0)
```

# References
- Anderson, T. W., & Rubin, H. (1949). Estimation of the parameters of a single
  equation in a complete system of stochastic equations. *Annals of Mathematical
  Statistics*, 20(1), 46–63.
- Chernozhukov, V., Hansen, C., & Spindler, M. (2015). Post-selection and
  post-regularization inference in linear models with many controls and instruments.
  *American Economic Review*, 105(5), 486–490.
- Chernozhukov, V., Chetverikov, D., Demirer, M., Duflo, E., Hansen, C., Newey, W., &
  Robins, J. (2018). Double/debiased machine learning for treatment and structural
  parameters. *The Econometrics Journal*, 21(1), C1–C68.
- Ma, Y. (2026). Identification-robust inference for the LATE with high-dimensional
  covariates. *Journal of Econometrics*, 257, 106302.
"""
function dml_weak_iv_test(r; beta0::Real=0.0)
    ctx = "dml_weak_iv_test"
    _iv_dmlar_check(r, ctx)
    R = size(r.psi, 2)
    dist = _iv_dmlar_dist(r)
    stats = [_iv_dmlar_stat(_iv_dmlar_parts(r, k), float(beta0)) for k in 1:R]
    ps = ccdf.(dist, stats)
    method = "DML Anderson–Rubin score test, " *
             (r.cluster === nothing ? "χ²(1)" :
              "cluster-robust, F(1, $(r.n_clusters - 1))") *
             (R > 1 ? ", upper median over $R repetitions" : "")
    dof = r.cluster === nothing ? (1,) : (1, r.n_clusters - 1)
    return DiagnosticTest("DML weak-IV-robust (Anderson–Rubin) test",
                          "θ = $(beta0) (" * r.estimand * ")", median(stats),
                          _iv_upper_median(ps); dof=dof, method=method,
                          note="Size does not depend on instrument strength; " *
                               "validity still requires instrument exogeneity, " *
                               "exclusion (and monotonicity for the LATE).",
                          details=(statistics=stats, pvalues=ps))
end

"""
    dml_weak_iv_confidence_set(r::DMLEstimate; level=0.95) -> WeakIVConfidenceSet

Weak-instrument-robust confidence set for the parameter of a DML IV model, obtained by
inverting the Anderson–Rubin-type score test of [`dml_weak_iv_test`](@ref).

The set is ``\\{\\beta_0 : AR(\\beta_0) \\le c\\}``, with ``c`` the `level` quantile of
``\\chi^2(1)`` (or of ``F(1, G - 1)`` with clustering). Because the orthogonal score is
linear in the parameter, the acceptance region of each cross-fitting repetition is a
quadratic inequality in ``\\beta_0`` and is solved in closed form: it is a bounded
interval when the orthogonalized first stage is significant at the chosen level, the
union of two rays when it is weak, the whole real line when the parameter is not
identified by the data, and possibly empty. Its coverage does not depend on the
strength of the instrument (Chernozhukov, Hansen and Spindler 2015; Ma 2026), whereas
the DML Wald interval can undercover badly when the first stage is weak. With
repeated cross-fitting a value belongs to the set when at least half of the
repetitions accept it, which is the set of values whose upper-median p-value (the rule
of [`dml_weak_iv_test`](@ref)) is at least ``1 - \\text{level}``.

Report the set alongside the first-stage diagnostics of [`ml_first_stage`](@ref); an
unbounded set is informative about weak identification rather than a failure of the
method. The requirements on the fit are those of [`dml_weak_iv_test`](@ref).

# Arguments
- `r::DMLEstimate`: a fit of [`dml_pliv`](@ref) (one instrument, partialling-out
  score) or of [`dml_iivm`](@ref).

# Keywords
- `level::Real`: confidence level (default 0.95).

# Returns
- A [`WeakIVConfidenceSet`](@ref): `intervals`, `kind`, `critical_value`, and
  `estimate` (the DML point estimate); `DrSnow.pvalue(set, β₀)` evaluates the
  p-value function.

# Examples
```julia
using DrSnow, DataFrames, StableRNGs
rng = StableRNG(1)
n = 1_000
x1, x2 = randn(rng, n), randn(rng, n)
z = Float64.(rand(rng, n) .< 1 ./ (1 .+ exp.(-0.5 .* x1)))
u = rand(rng, n)
d = Float64.((u .< 0.2) .| ((u .< 0.5) .& (z .== 1)))
y = d .+ x1 .+ 0.5 .* x2 .+ 0.5 .* (u .< 0.2) .+ randn(rng, n)
df = DataFrame(y=y, d=d, z=z, x1=x1, x2=x2)
r = dml_iivm(df, :y, :d, :z; covariates=[:x1, :x2], rng=StableRNG(3))
dml_weak_iv_confidence_set(r; level=0.9)
```

# References
- Anderson, T. W., & Rubin, H. (1949). Estimation of the parameters of a single
  equation in a complete system of stochastic equations. *Annals of Mathematical
  Statistics*, 20(1), 46–63.
- Chernozhukov, V., Hansen, C., & Spindler, M. (2015). Post-selection and
  post-regularization inference in linear models with many controls and instruments.
  *American Economic Review*, 105(5), 486–490.
- Ma, Y. (2026). Identification-robust inference for the LATE with high-dimensional
  covariates. *Journal of Econometrics*, 257, 106302.
- Andrews, I., Stock, J. H., & Sun, L. (2019). Weak instruments in instrumental
  variables regression: Theory and practice. *Annual Review of Economics*, 11,
  727–753.
"""
function dml_weak_iv_confidence_set(r; level::Real=0.95)
    ctx = "dml_weak_iv_confidence_set"
    0 < level < 1 || throw(ArgumentError("$ctx: level must be in (0, 1)"))
    _iv_dmlar_check(r, ctx)
    R = size(r.psi, 2)
    dist = _iv_dmlar_dist(r)
    c = quantile(dist, level)
    parts = [_iv_dmlar_parts(r, k) for k in 1:R]
    sets = map(parts) do p
        _iv_quadratic_region(p.A^2 - c * p.Qaa, 2 * (p.A * p.B - c * p.Qab),
                             p.B^2 - c * p.Qbb)
    end
    intervals = if R == 1
        sets[1]
    else
        need = cld(R, 2)
        bps = sort!(unique([x for s in sets for iv in s for x in iv if isfinite(x)]))
        inside(x) = count(s -> any(iv -> iv[1] <= x <= iv[2], s), sets) >= need
        if isempty(bps)
            inside(0.0) ? [(-Inf, Inf)] : Tuple{Float64,Float64}[]
        else
            # segments: (-∞, b1), (b1, b2), ..., (bK, ∞); test midpoints
            inner = [(bps[i] + bps[i + 1]) / 2 for i in 1:(length(bps) - 1)]
            mids = vcat(bps[1] - 1.0, inner, bps[end] + 1.0)
            ok = inside.(mids)
            lo = vcat(-Inf, bps)
            hi = vcat(bps, Inf)
            out = Tuple{Float64,Float64}[]
            i = 1
            while i <= length(ok)
                if !ok[i]
                    i += 1
                    continue
                end
                j = i
                while j < length(ok) && ok[j + 1]
                    j += 1
                end
                push!(out, (lo[i], hi[j]))
                i = j + 1
            end
            out
        end
    end
    pfun = β -> _iv_upper_median([ccdf(dist, _iv_dmlar_stat(p, β)) for p in parts])
    method = "DML Anderson–Rubin" * (r.cluster === nothing ? "" : " (cluster-robust)") *
             (R > 1 ? ", $R repetitions" : "")
    return WeakIVConfidenceSet(method, float(level), _iv_set_kind(intervals), intervals,
                               c, r.coef[1], pfun)
end
