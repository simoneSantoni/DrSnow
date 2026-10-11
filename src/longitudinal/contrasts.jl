# Contrasts between regimes and marginal structural models projected from the
# regime-specific estimates, with delta-method covariances from the joint efficient
# influence functions (or from the bootstrap replicates).

"""
    RegimeContrast <: CausalEstimate

Contrasts between treatment regimes or coefficients of a marginal structural model,
returned by [`regime_contrast`](@ref) and [`regime_msm`](@ref).

The coefficients are differences ``ψ_d - ψ_{d_0}``, log ratios
``\\log(ψ_d / ψ_{d_0})``, log odds ratios
``\\operatorname{logit}(ψ_d) - \\operatorname{logit}(ψ_{d_0})`` or the coefficients
``β`` of a working marginal structural model ``ψ_d ≈ x_d'β``, each a smooth function
of the regime-specific estimates of a [`LongitudinalEstimate`](@ref). Their
covariance is obtained by the delta method from the joint influence functions of
the regime estimates (so the dependence between regimes estimated on the same units
is accounted for), or from the bootstrap replicates when the estimate carries them.
For ratios and odds ratios the coefficients are on the log scale: `exp.(coef(c))`
and `exp.(confint(c))` give the ratios and their intervals.

Supports `coef`, `vcov`, `stderror`, `confint(c; level)`, `coeftable`, `nobs`,
`dof_residual` (``G - 1`` with clusters), `estimand`, `method_name`, [`tidy`](@ref)
and [`glance`](@ref).

# Fields
- `kind::Symbol`: `:difference`, `:ratio`, `:odds_ratio` or `:msm`.
- `names::Vector{String}`, `coef::Vector{Float64}`, `vcov::Matrix{Float64}`.
- `eif::Union{Nothing,Matrix{Float64}}`: `n × p` influence functions of the
  coefficients (`nothing` when the covariance comes from the bootstrap).
- `bootstrap_coef::Matrix{Float64}`: `B × p` bootstrap replicates (`0 × p` when
  none).
- `reference::String`: the reference regime (empty for `:msm`).
- `source::String`: the estimator of the regime means.
- `n::Int`, `cluster`, `n_clusters::Int`: as in [`LongitudinalEstimate`](@ref).
"""
struct RegimeContrast <: CausalEstimate
    kind::Symbol
    names::Vector{String}
    coef::Vector{Float64}
    vcov::Matrix{Float64}
    eif::Union{Nothing,Matrix{Float64}}
    bootstrap_coef::Matrix{Float64}
    reference::String
    source::String
    n::Int
    cluster::Union{Nothing,Vector{Int}}
    n_clusters::Int
end

StatsAPI.coef(r::RegimeContrast) = r.coef
StatsAPI.vcov(r::RegimeContrast) = r.vcov
StatsAPI.coefnames(r::RegimeContrast) = r.names
StatsAPI.nobs(r::RegimeContrast) = r.n
StatsAPI.dof_residual(r::RegimeContrast) = r.cluster === nothing ? Inf : r.n_clusters - 1.0
estimand(r::RegimeContrast) = r.kind === :difference ?
    "difference in mean counterfactual outcomes between regimes" :
    r.kind === :ratio ? "log ratio of mean counterfactual outcomes between regimes" :
    r.kind === :odds_ratio ? "log odds ratio of mean counterfactual outcomes" :
    "coefficients of a working marginal structural model over regimes"
method_name(r::RegimeContrast) = "Regime " * (r.kind === :msm ? "MSM projection" :
                                              "contrast") * " (" * r.source * ")"

function show_details(io::IO, r::RegimeContrast)
    println(io)
    r.kind === :msm || println(io, "Reference regime: ", r.reference)
    r.kind in (:ratio, :odds_ratio) &&
        println(io, "Coefficients are on the log scale; exponentiate for ratios")
    if size(r.bootstrap_coef, 1) > 0
        println(io, "Covariance: bootstrap, ", size(r.bootstrap_coef, 1), " replicates")
    else
        println(io, "Covariance: delta method from the joint influence functions")
    end
    return nothing
end

# Delta method for a smooth map `f(ψ)` of the regime means with Jacobian `J(ψ)`.
function _lg_transform(r::LongitudinalEstimate, f, J, names, kind, reference, context)
    ψ = r.coef
    θ = f(ψ)
    B = size(r.bootstrap_coef, 1)
    if B > 0
        reps = reduce(vcat, [permutedims(f(r.bootstrap_coef[b, :])) for b in 1:B])
        V = Matrix(Symmetric(cov(reps; dims=1)))
        eif = nothing
    else
        r.eif === nothing && throw(ArgumentError(
            "$context: the g-computation estimator has no influence functions; refit " *
            "with `bootstrap = B` or use estimator = :tmle or :sdr"))
        Jm = J(ψ)
        eif = r.eif * Jm'
        V = _if_vcov(eif, r.cluster)[1]
        reps = zeros(0, length(θ))
    end
    return RegimeContrast(kind, names, θ, V, eif, reps, reference, method_name(r), r.n,
                          r.cluster, r.n_clusters)
end

"""
    regime_contrast(r::LongitudinalEstimate; reference=1, type=:difference)
        -> RegimeContrast

Contrasts of every regime of `r` against a reference regime: differences, log
ratios or log odds ratios of the mean counterfactual outcomes.

With regime means ``ψ_d = E[Y^d]`` from [`longitudinal_regime`](@ref) and a
reference regime ``d_0``, the contrasts are ``ψ_d - ψ_{d_0}`` (`:difference`, the
average effect of following ``d`` rather than ``d_0``), ``\\log(ψ_d / ψ_{d_0})``
(`:ratio`; for a binary outcome the log risk ratio) and
``\\operatorname{logit} ψ_d - \\operatorname{logit} ψ_{d_0}`` (`:odds_ratio`). With
``d = \\bar 1`` and ``d_0 = \\bar 0`` the difference is the effect of sustained
treatment versus never treating, the usual summary of a marginal structural
analysis (Robins, Hernán and Brumback 2000); with several regimes the contrasts
are the parameters of a saturated marginal structural model with the chosen
reference level.

The covariance comes from the delta method applied to the joint influence functions
of the regime means stored in `r`, so that estimates sharing units, folds and the
treatment mechanism are correctly correlated; with `bootstrap > 0` in `r` it is the
covariance of the transformed bootstrap replicates. Ratios require all means to be
positive and odds ratios all means strictly inside ``(0, 1)``; otherwise an
`ArgumentError` is thrown. Intervals for ratios should be formed on the log scale
and exponentiated, which is what `exp.(confint(c))` does; the reported standard
errors are those of the log ratios.

# Arguments
- `r::LongitudinalEstimate`: regime means fitted jointly (one call with a vector
  of regimes), with influence functions or bootstrap replicates.

# Keywords
- `reference = 1`: the reference regime, as an index or a label.
- `type::Symbol = :difference`: `:difference`, `:ratio` or `:odds_ratio`.

# Returns
- [`RegimeContrast`](@ref) with one coefficient per non-reference regime, named
  `"<label> vs <reference label>"`.

# Examples
```julia
using DrSnow, DataFrames, StableRNGs
rng = StableRNG(5)
n, K = 500, 2
id = repeat(1:n; inner=K)
t = repeat(1:K, n)
l = randn(rng, n * K)
a = zeros(n * K)
y = Vector{Union{Missing,Float64}}(missing, n * K)
for i in 1:n
    a1 = Float64(rand(rng) < 1 / (1 + exp(-l[2i - 1])))
    l[2i] += 0.5 * a1
    a2 = Float64(rand(rng) < 1 / (1 + exp(-(l[2i] + a1))))
    a[2i - 1], a[2i] = a1, a2
    y[2i] = Float64(rand(rng) < 1 / (1 + exp(-(-1 + 0.7 * (a1 + a2) + 0.5 * l[2i]))))
end
df = DataFrame(; id, t, l, a, y)
r = longitudinal_regime(df, :y, :a, :id, :t; regime=[never_treat(K), always_treat(K)],
                        covariates=[:l], estimator=:tmle, outcome_learner=OLSLearner(),
                        treatment_learner=LogisticLearner(), rng=StableRNG(6))
regime_contrast(r)                               # risk difference
exp.(coef(regime_contrast(r; type=:ratio)))      # risk ratio
```

# References
- Robins, J. M., Hernán, M. A., & Brumback, B. (2000). Marginal structural models and
  causal inference in epidemiology. *Epidemiology*, 11(5), 550–560.
- Hernán, M. A., & Robins, J. M. (2020). *Causal Inference: What If*. Chapman &
  Hall/CRC.
"""
function regime_contrast(r::LongitudinalEstimate; reference=1, type::Symbol=:difference)
    ctx = "regime_contrast"
    type in (:difference, :ratio, :odds_ratio) ||
        throw(ArgumentError("$ctx: type must be :difference, :ratio or :odds_ratio"))
    R = length(r.coef)
    R >= 2 || throw(ArgumentError("$ctx: at least two regimes are needed"))
    ref = reference isa Integer ? Int(reference) :
          findfirst(==(string(reference)), r.names)
    (ref === nothing || !(1 <= ref <= R)) &&
        throw(ArgumentError("$ctx: reference regime $(repr(reference)) not found"))
    others = [j for j in 1:R if j != ref]
    names = [r.names[j] * " vs " * r.names[ref] for j in others]
    if type === :ratio
        all(>(0), r.coef) ||
            throw(ArgumentError("$ctx: ratios need strictly positive regime means"))
    elseif type === :odds_ratio
        all(v -> 0 < v < 1, r.coef) ||
            throw(ArgumentError("$ctx: odds ratios need regime means strictly inside " *
                                "(0, 1)"))
    end
    g = type === :difference ? identity :
        type === :ratio ? (x -> log(x)) : (x -> log(x / (1 - x)))
    dg = type === :difference ? (x -> 1.0) :
         type === :ratio ? (x -> 1 / x) : (x -> 1 / (x * (1 - x)))
    f = ψ -> [g(ψ[j]) - g(ψ[ref]) for j in others]
    J = function (ψ)
        Jm = zeros(length(others), R)
        for (i, j) in enumerate(others)
            Jm[i, j] = dg(ψ[j])
            Jm[i, ref] = -dg(ψ[ref])
        end
        Jm
    end
    return _lg_transform(r, f, J, names, type, r.names[ref], ctx)
end

"""
    regime_msm(r::LongitudinalEstimate, X; names=nothing, weights=nothing)
        -> RegimeContrast

Coefficients of a working marginal structural model fitted to the regime means of
`r`: ``ψ_d ≈ x_d'β`` for the rows ``x_d`` of the design matrix `X`.

A marginal structural model (Robins, Hernán and Brumback 2000) summarizes how the
mean counterfactual outcome varies over a family of regimes, for example
``E[Y^{\\bar a}] = β_0 + β_1 \\sum_t a_t`` for the static regimes with cumulative
treatment ``\\sum_t a_t``. Here the model is fitted by (weighted) least squares to the
regime-specific estimates of [`longitudinal_regime`](@ref):
``\\hat β = (X'ΩX)^{-1} X'Ω \\hat ψ``, with ``Ω`` the diagonal matrix of `weights`
(identity by default). The influence function of ``\\hat β`` is the same linear map
of the influence functions of ``\\hat ψ``, which gives its covariance (Petersen et al.
2014 describe this projection approach for targeted estimators; ltmle's `ltmleMSM`
fits the pooled model directly). When the model is saturated (as many free
coefficients as regimes) the fit is exact and the coefficients are linear
contrasts; otherwise ``β`` is the projection of the regime means onto the model, a
well-defined summary even when the model is misspecified, and its interpretation is
that of the best approximation in the chosen metric.

# Arguments
- `r::LongitudinalEstimate`: jointly estimated regime means.
- `X::AbstractMatrix`: the ``R × p`` design matrix, one row per regime of `r` in
  order (include a column of ones for an intercept).

# Keywords
- `names = nothing`: coefficient names (default `"β1", …`).
- `weights = nothing`: non-negative weights of the regimes in the least-squares
  fit (for example their empirical support).

# Returns
- [`RegimeContrast`](@ref) with `kind = :msm`.

# Examples
```julia
using DrSnow
# with r from longitudinal_regime(...; regime=[StaticRegime([0,0]), StaticRegime([1,0]),
#                                              StaticRegime([1,1])])
# the model E[Y^a] = β0 + β1 * (number of treated periods):
# regime_msm(r, [1 0; 1 1; 1 2]; names=["intercept", "per treated period"])
```

# References
- Robins, J. M., Hernán, M. A., & Brumback, B. (2000). Marginal structural models and
  causal inference in epidemiology. *Epidemiology*, 11(5), 550–560.
- Petersen, M., Schwab, J., Gruber, S., Blaser, N., Schomaker, M., & van der Laan,
  M. (2014). Targeted maximum likelihood estimation for dynamic and static
  longitudinal marginal structural working models. *Journal of Causal Inference*,
  2(2), 147–185.
"""
function regime_msm(r::LongitudinalEstimate, X::AbstractMatrix; names=nothing,
                    weights=nothing)
    ctx = "regime_msm"
    R = length(r.coef)
    Xm = Matrix{Float64}(X)
    size(Xm, 1) == R ||
        throw(ArgumentError("$ctx: X must have one row per regime ($R), got " *
                            "$(size(Xm, 1))"))
    p = size(Xm, 2)
    p >= 1 || throw(ArgumentError("$ctx: X needs at least one column"))
    w = weights === nothing ? ones(R) : Float64.(collect(weights))
    length(w) == R || throw(DimensionMismatch("$ctx: need one weight per regime"))
    all(>=(0), w) || throw(ArgumentError("$ctx: weights must be non-negative"))
    XtW = Xm' .* w'
    M = XtW * Xm
    rank(M) == p || throw(ArgumentError("$ctx: the design matrix is rank deficient"))
    P = M \ XtW                                   # p × R
    nm = names === nothing ? ["β$j" for j in 1:p] : string.(collect(names))
    length(nm) == p || throw(DimensionMismatch("$ctx: need $p names"))
    return _lg_transform(r, ψ -> P * ψ, ψ -> P, nm, :msm, "", ctx)
end
