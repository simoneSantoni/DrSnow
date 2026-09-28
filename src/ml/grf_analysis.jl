# Post-estimation for generalized random forests, mirroring grf: doubly robust
# scores, average treatment effects (AIPW; ATE, ATT, ATC, overlap-weighted), the best
# linear projection of the CATE, the calibration test, variable importance, the rank
# average treatment effect (RATE: AUTOC / Qini with half-sample bootstrap) and
# policy values.

"""
    HTEEstimate <: CausalEstimate

Generic result type of the heterogeneous-effects tools: [`average_treatment_effect`](@ref),
[`policy_value`](@ref), [`subgroup_effects`](@ref), [`interaction_effects`](@ref) and
[`metalearner_bootstrap`](@ref).

An `HTEEstimate` holds one or more estimated effects (an average effect, subgroup effects,
interaction coefficients, policy values, …) with their full covariance matrix and the
reference distribution for inference. It supports the full `CausalEstimate` interface:
`coef`, `vcov`, `stderror`, `confint(r; level)`, `coeftable`, `tidy`, `nobs`,
`dof_residual`, `estimand` and `method_name`; [`heterogeneity_test`](@ref) extracts the
joint test stored by [`subgroup_effects`](@ref) and [`interaction_effects`](@ref).

# Fields
- `names::Vector{String}`: coefficient names.
- `coef::Vector{Float64}`, `vcov::Matrix{Float64}`: estimates and their covariance.
- `n::Int`: number of observations used.
- `dof::Float64`: residual degrees of freedom of the t reference (`Inf` for normal
  inference, `G - 1` with `G` clusters).
- `estimand::String`, `method::String`: descriptions used when printing.
- `details::NamedTuple`: method-specific output, for example the target and subset of
  [`average_treatment_effect`](@ref), the share treated by the rule in
  [`policy_value`](@ref), or the heterogeneity test and multiplicity-adjusted p-values of
  [`subgroup_effects`](@ref).
"""
struct HTEEstimate <: CausalEstimate
    names::Vector{String}
    coef::Vector{Float64}
    vcov::Matrix{Float64}
    n::Int
    dof::Float64
    estimand::String
    method::String
    details::NamedTuple
end

StatsAPI.coef(r::HTEEstimate) = r.coef
StatsAPI.vcov(r::HTEEstimate) = r.vcov
StatsAPI.coefnames(r::HTEEstimate) = r.names
StatsAPI.nobs(r::HTEEstimate) = r.n
StatsAPI.dof_residual(r::HTEEstimate) = r.dof
estimand(r::HTEEstimate) = r.estimand
method_name(r::HTEEstimate) = r.method

function show_details(io::IO, r::HTEEstimate)
    d = r.details
    any(k -> haskey(d, k), (:heterogeneity, :p_adjusted, :note)) && println(io)
    if haskey(d, :heterogeneity)
        t = d.heterogeneity
        lab = length(t.dof) == 2 ? @sprintf("F(%d, %g)", t.dof[1], t.dof[2]) :
              @sprintf("χ²(%d)", t.dof[1])
        @printf(io, "%s: %s = %.4g, p = %.4g\n", t.name, lab, t.statistic, t.pvalue)
    end
    if haskey(d, :p_adjusted)
        println(io, "Multiplicity-adjusted p-values (", d.adjust, "): ",
                join([@sprintf("%s %.4g", nm, p) for (nm, p) in zip(r.names, d.p_adjusted)],
                     ", "))
    end
    haskey(d, :note) && !isempty(d.note) && println(io, "Note: ", d.note)
    return nothing
end

"""
    heterogeneity_test(r::HTEEstimate) -> DiagnosticTest

Return the joint Wald test of effect homogeneity stored in the result of
[`subgroup_effects`](@ref) or [`interaction_effects`](@ref).

For [`subgroup_effects`](@ref) the null hypothesis is that all subgroup effects are equal;
for [`interaction_effects`](@ref) it is that all treatment-by-moderator interaction
coefficients are zero. The statistic is a Wald statistic built from the full covariance
matrix of the estimates (an F statistic when a finite-sample reference is used, otherwise
``\\chi^2``). Testing the interaction jointly, rather than comparing subgroup-specific
significance, is the practice recommended for subgroup analyses by Wang, Lagakos, Ware,
Hunter and Drazen (2007). A non-rejection does not show that effects are homogeneous: the
test has power only against differences along the pre-specified subgroups or moderators,
and it is often underpowered. For data-driven heterogeneity see [`test_calibration`](@ref),
[`generic_ml`](@ref) and [`rank_average_treatment_effect`](@ref).

# Arguments
- `r::HTEEstimate`: a result of [`subgroup_effects`](@ref) or
  [`interaction_effects`](@ref); other results carry no heterogeneity test and raise an
  `ArgumentError`.

# Returns
- `DiagnosticTest` with the statistic, degrees of freedom and p-value.

# Examples
```julia
using DrSnow, DataFrames, StableRNGs
rng = StableRNG(1)
n = 400
df = DataFrame(region=rand(rng, ["north", "south"], n), d=rand(rng, 0:1, n))
df.y = df.d .* (1 .+ (df.region .== "south")) .+ randn(rng, n)
heterogeneity_test(subgroup_effects(df, :y, :d, :region))
```

# References
- Wang, R., Lagakos, S. W., Ware, J. H., Hunter, D. J., & Drazen, J. M. (2007). Statistics
  in medicine — reporting of subgroup analyses in clinical trials. *New England Journal of
  Medicine*, 357(21), 2189–2194.
"""
function heterogeneity_test(r::HTEEstimate)
    haskey(r.details, :heterogeneity) ||
        throw(ArgumentError("heterogeneity_test: this result has no heterogeneity test"))
    return r.details.heterogeneity
end

# ------------------------------------------------------------------- helpers

"""grf's observation_weights: sample weights or 1/cluster size, normalized."""
function _ml_grf_obs_weights(f::GeneralizedRandomForest)
    n = nobs(f)
    raw = if f.sample_weights !== nothing
        copy(f.sample_weights)
    elseif f.cluster !== nothing && f.equalize_cluster_weights
        cnt = zeros(f.n_clusters)
        for g in f.cluster
            cnt[g] += 1
        end
        [1 / cnt[g] for g in f.cluster]
    else
        ones(n)
    end
    return raw ./ sum(raw)
end

function _ml_grf_subset(f::GeneralizedRandomForest, subset, context)
    n = nobs(f)
    subset === nothing && return collect(1:n)
    if eltype(subset) == Bool
        length(subset) == n ||
            throw(DimensionMismatch("$(context): a logical subset must have length $n"))
        idx = findall(subset)
    else
        idx = sort(unique(Int.(collect(subset))))
        all(i -> 1 <= i <= n, idx) ||
            throw(ArgumentError("$(context): subset indices must be in 1:$n"))
    end
    isempty(idx) && throw(ArgumentError("$(context): the subset is empty"))
    return idx
end

_ml_grf_clusters_or_units(f) = f.cluster === nothing ? collect(1:nobs(f)) : f.cluster

_ml_is_binary(v) = all(x -> x == 0 || x == 1, v)

"""
Weighted means of the columns of `Ψ` and their covariance
`Σ_c (Σ_{i∈c} wᵢ(Ψᵢ - θ))(…)' / (Σw)² · G/(G-1)` with `G` the clusters of positive
weight (grf's `.sigma2.hat`).
"""
function _ml_grf_wmean_vcov(Ψ::AbstractMatrix, w::AbstractVector, clusters)
    sw = sum(w)
    θ = vec(sum(Ψ .* w; dims=1)) ./ sw
    g, _ = _ml_group_index_any(clusters)
    G = maximum(g)
    S = zeros(G, size(Ψ, 2))
    wsum = zeros(G)
    for i in axes(Ψ, 1)
        @views S[g[i], :] .+= w[i] .* (Ψ[i, :] .- θ)
        wsum[g[i]] += w[i]
    end
    nadj = count(>(0), wsum)
    nadj >= 2 || throw(ArgumentError("need units from at least two clusters"))
    V = (S' * S) ./ sw^2 .* (nadj / (nadj - 1))
    return θ, Matrix(Symmetric(V)), nadj
end

function _ml_group_index_any(v)
    u = sort(unique(v))
    idx = Dict(x => i for (i, x) in enumerate(u))
    return [idx[x] for x in v], length(u)
end

"""
Weighted least squares with sandwich covariance as R's `sandwich::vcovCL`
(`clusters` a vector; unit clusters when `nothing`): `type = :HC3` (cluster-level
leverage correction, no small-sample factor; with unit clusters it equals
`vcovHC(type = "HC3")`) or `:HC1` (factor `G/(G-1)·(n-1)/(n-k)`; `hc = true` gives
`vcovHC`'s `n/(n-k)`).
"""
function _ml_grf_lm(Z::AbstractMatrix, y::AbstractVector, w::AbstractVector, clusters;
                    type::Symbol=:HC3, hc::Bool=false)
    n, k = size(Z)
    sw = sqrt.(w)
    Xt = Z .* sw
    F = qr(Xt, ColumnNorm())
    rank(F.R) == k || throw(ArgumentError("collinear regressors in the projection"))
    β = F \ (y .* sw)
    et = (y .- Z * β) .* sw
    A = inv(Symmetric(Xt' * Xt))
    g, G = clusters === nothing ? (collect(1:n), n) : _ml_group_index_any(clusters)
    M = zeros(k, k)
    if type === :HC1
        S = zeros(G, k)
        for i in 1:n
            @views S[g[i], :] .+= Xt[i, :] .* et[i]
        end
        M = S' * S
        adj = hc ? n / (n - k) : G / (G - 1) * (n - 1) / (n - k)
        M .*= adj
    elseif type === :HC3
        members = [Int[] for _ in 1:G]
        for i in 1:n
            push!(members[g[i]], i)
        end
        # sandwich::meatCL: ψ_g = Z_g' (I - Z_g A Z_g' W_g)⁻¹ W_g e_g, with no
        # G/(G-1) factor for HC3 (sandwich cancels it)
        e = y .- Z * β
        for m in members
            isempty(m) && continue
            Zg = Z[m, :]
            H = Zg * A * Zg' * Diagonal(w[m])
            ψ = Zg' * ((I - H) \ (w[m] .* e[m]))
            M .+= ψ * ψ'
        end
    else
        throw(ArgumentError("vcov_type must be :HC3 or :HC1"))
    end
    return β, Matrix(Symmetric(A * M * A)), G
end

# ------------------------------------------------------------------- scores

"""
    get_scores(f::CausalForest; subset=nothing, debiasing_weights=nothing,
               num_trees_for_weights=500, rng=Random.Xoshiro(f.seed))
        -> Vector{Float64}
    get_scores(f::InstrumentalForest; subset=nothing, debiasing_weights=nothing,
               compliance_score=nothing, num_trees_for_weights=500,
               rng=Random.Xoshiro(f.seed)) -> Vector{Float64}

Doubly robust (AIPW) scores of the conditional effect estimated by a forest (grf's
`get_scores`), the building block of all average-effect summaries.

The score of unit ``i`` augments the forest's out-of-bag CATE with a weighted residual,
```math
\\Gamma_i = \\hat\\tau(X_i)
    + \\gamma_i \\{Y_i - \\hat Y(X_i) - \\hat\\tau(X_i)(W_i - \\hat W(X_i))\\},
```
so that its mean identifies an average effect even if ``\\hat\\tau`` is biased, provided
the debiasing weights ``\\gamma_i`` are consistent (Robins, Rotnitzky & Zhao 1994; Athey &
Wager 2019). For a binary treatment
``\\gamma_i = (W_i - \\hat W_i) / \\{\\hat W_i(1 - \\hat W_i)\\}``, the inverse-propensity
weight; for a continuous treatment ``\\gamma_i = (W_i - \\hat W_i) / \\hat V(X_i)``, with
``\\hat V`` an out-of-bag regression forest for ``E[(W - \\hat W)^2 \\mid X]``, which
yields the average partial effect. For an instrumental forest with a binary instrument
``\\gamma_i = (Z_i - \\hat Z_i) / \\{\\hat Z_i(1 - \\hat Z_i)\\} / \\hat\\Delta(X_i)``,
where the compliance score ``\\hat\\Delta(x)`` estimates the conditional effect of ``Z`` on
``W`` (from an auxiliary causal forest unless supplied); the mean score then estimates the
average conditional LATE.

The scores are Neyman-orthogonal: their mean is insensitive to first-order errors in
``\\hat Y``, ``\\hat W`` and ``\\hat\\tau``, which is what allows valid root-n inference on
averages of flexibly estimated nuisances, as in double/debiased machine learning
(Chernozhukov et al. 2018). Propensities or instrument propensities close to 0 or 1 make
``\\gamma_i`` large and the scores heavy-tailed, and exactly 0 or 1 raises an error; the
scores stay well defined in the interior but limited overlap inflates their variance (see
[`average_treatment_effect`](@ref) for remedies). The scores are used by
[`average_treatment_effect`](@ref), [`best_linear_projection`](@ref) and
[`rank_average_treatment_effect`](@ref); for per-arm scores for policy learning see
[`double_robust_scores`](@ref).

# Arguments
- `f`: a [`CausalForest`](@ref) or an [`InstrumentalForest`](@ref) (the latter needs a
  binary instrument unless `debiasing_weights` are supplied).

# Keywords
- `subset = nothing`: row indices or a logical vector selecting the rows to score (default:
  all rows).
- `debiasing_weights = nothing`: user-supplied ``\\gamma`` (length `n` or the subset
  length), which replaces the default weights.
- `compliance_score = nothing` (instrumental forests): user-supplied ``\\hat\\Delta``.
- `num_trees_for_weights::Integer = 500`: trees of the auxiliary forest (continuous
  treatment, or compliance score).
- `rng::AbstractRNG = Random.Xoshiro(f.seed)`: generator of the auxiliary forest; the
  default makes the scores reproducible from the fitted forest.

# Returns
- `Vector{Float64}`: one score per row of the subset.

# Examples
```julia
using DrSnow, DataFrames, StableRNGs, Statistics
rng = StableRNG(1)
df = DataFrame(x1=rand(rng, 400), x2=rand(rng, 400))
df.d = Int.(rand(rng, 400) .< 0.5)
df.y = (1 .+ df.x1) .* df.d .+ randn(rng, 400)
cf = causal_forest(df, :y, :d; covariates=[:x1, :x2], num_trees=200,
                   rng=StableRNG(2))
Γ = get_scores(cf)
mean(Γ)       # the AIPW estimate of the ATE (truth 1.5)
```

# References
- Athey, S., & Wager, S. (2019). Estimating treatment effects with causal forests: An
  application. *Observational Studies*, 5(2), 37–51.
- Chernozhukov, V., Chetverikov, D., Demirer, M., Duflo, E., Hansen, C., Newey, W., &
  Robins, J. (2018). Double/debiased machine learning for treatment and structural
  parameters. *The Econometrics Journal*, 21(1), C1–C68.
- Robins, J. M., Rotnitzky, A., & Zhao, L. P. (1994). Estimation of regression coefficients
  when some regressors are not always observed. *Journal of the American Statistical
  Association*, 89(427), 846–866.
"""
function get_scores(f::CausalForest; subset=nothing, debiasing_weights=nothing,
                    num_trees_for_weights::Integer=500,
                    rng::AbstractRNG=Random.Xoshiro(f.seed))
    ctx = "get_scores"
    idx = _ml_grf_subset(f, subset, ctx)
    τ = f.predictions
    any(isnan, view(τ, idx)) &&
        throw(ArgumentError("$(ctx): some out-of-bag predictions are undefined; " *
                            "increase num_trees"))
    γ = if debiasing_weights !== nothing
        _ml_grf_len_subset(debiasing_weights, nobs(f), idx, "debiasing_weights", ctx)
    elseif _ml_is_binary(f.W)
        what = f.W_hat[idx]
        any(p -> p <= 0 || p >= 1, what) &&
            throw(ArgumentError("$(ctx): estimated propensities of exactly 0 or 1"))
        (f.W[idx] .- what) ./ (what .* (1 .- what))
    else
        v = _ml_regression_forest(f.X, (f.W .- f.W_hat) .^ 2, f.sample_weights,
                                  f.cluster, f.covariates, :V;
                                  num_trees=num_trees_for_weights, ci_group_size=1,
                                  rng=rng).predictions
        ((f.W .- f.W_hat) ./ v)[idx]
    end
    res = f.Y[idx] .- (f.Y_hat[idx] .+ τ[idx] .* (f.W[idx] .- f.W_hat[idx]))
    return τ[idx] .+ γ .* res
end

function get_scores(f::InstrumentalForest; subset=nothing, debiasing_weights=nothing,
                    compliance_score=nothing, num_trees_for_weights::Integer=500,
                    rng::AbstractRNG=Random.Xoshiro(f.seed))
    ctx = "get_scores"
    idx = _ml_grf_subset(f, subset, ctx)
    τ = f.predictions
    any(isnan, view(τ, idx)) &&
        throw(ArgumentError("$(ctx): some out-of-bag predictions are undefined"))
    γ = if debiasing_weights !== nothing
        _ml_grf_len_subset(debiasing_weights, nobs(f), idx, "debiasing_weights", ctx)
    else
        _ml_is_binary(f.Z) ||
            throw(ArgumentError("$(ctx): average conditional LATEs require a binary " *
                                "instrument (or debiasing_weights)"))
        Δ = if compliance_score === nothing
            cf = _ml_causal_forest(f.X, f.W, f.Z, f.sample_weights,
                                   f.cluster, f.covariates,
                                   :W, :Z, nothing; y_hat=f.W_hat, w_hat=f.Z_hat,
                                   num_trees=num_trees_for_weights, rng=rng)
            cf.predictions[idx]
        else
            _ml_grf_len_subset(compliance_score, nobs(f), idx, "compliance_score", ctx)
        end
        zh = f.Z_hat[idx]
        (minimum(zh) <= 0.01 || maximum(zh) >= 0.99) &&
            @warn "Estimated instrument propensities are close to 0 or 1; poor " *
                  "overlap may hurt the average conditional LATE estimate."
        minimum(abs, Δ) <= 0.01 * std(f.W[idx]) &&
            @warn "The instrument appears weak for some units: compliance scores as " *
                  "low as $(round(minimum(Δ); sigdigits=3))."
        (f.Z[idx] .- zh) ./ (zh .* (1 .- zh)) ./ Δ
    end
    res = f.Y[idx] .- (f.Y_hat[idx] .+ τ[idx] .* (f.W[idx] .- f.W_hat[idx]))
    return τ[idx] .+ γ .* res
end

function _ml_grf_len_subset(v, n, idx, name, ctx)
    x = Float64.(collect(v))
    length(x) == n && return x[idx]
    length(x) == length(idx) && return x
    throw(DimensionMismatch("$(ctx): $(name) must have length n or the subset length"))
end

function _ml_grf_overlap_warning(what, target)
    lo, hi = extrema(what)
    if target !== :overlap
        if lo <= 0.05 && hi >= 0.95
            @warn "Estimated treatment propensities range from $(round(lo; digits=3)) " *
                  "to $(round(hi; digits=3)); consider target = :overlap or trimming " *
                  "the sample (Crump, Hotz, Imbens & Mitnik 2009)."
        elseif lo <= 0.05 && target !== :treated
            @warn "Estimated treatment propensities go as low as " *
                  "$(round(lo; digits=3)); effects for some controls are poorly " *
                  "identified (consider target = :treated)."
        elseif hi >= 0.95 && target !== :control
            @warn "Estimated treatment propensities go as high as " *
                  "$(round(hi; digits=3)); effects for some treated units are poorly " *
                  "identified (consider target = :control)."
        end
    end
    return nothing
end

# ---------------------------------------------------------- average effects

"""
    average_treatment_effect(f::CausalForest; target=:all, subset=nothing,
                             debiasing_weights=nothing, num_trees_for_weights=500,
                             rng=Random.Xoshiro(f.seed)) -> HTEEstimate
    average_treatment_effect(f::InstrumentalForest; target=:all, subset=nothing,
                             debiasing_weights=nothing, compliance_score=nothing,
                             num_trees_for_weights=500, rng=Random.Xoshiro(f.seed))
        -> HTEEstimate

Doubly robust (AIPW) estimates of average treatment effects from a causal or instrumental
forest (grf's `average_treatment_effect`, `method = "AIPW"`).

The available estimands, for a binary treatment with propensity score
``e(x) = P(W = 1 \\mid X = x)``, are
```math
\\mathrm{ATE} = E[\\tau(X)], \\quad
\\mathrm{ATT} = E[\\tau(X) \\mid W = 1], \\quad
\\mathrm{ATC} = E[\\tau(X) \\mid W = 0], \\quad
\\mathrm{ATO} = \\frac{E[e(X)\\{1 - e(X)\\}\\tau(X)]}{E[e(X)\\{1 - e(X)\\}]},
```
selected by `target = :all`, `:treated`, `:control` and `:overlap`. With `target = :all`
the estimate is the (observation-weighted) mean of the doubly robust scores of
[`get_scores`](@ref); it is also available for a continuous treatment (average partial
effect) and, for an [`InstrumentalForest`](@ref) with a binary instrument, for the average
conditional LATE. The ATT and ATC average ``\\hat\\tau(X_i)`` over the treated (controls)
and add a normalized AIPW correction. The overlap-weighted effect of Li, Morgan and
Zaslavsky (2018) is estimated by the residual-on-residual regression of ``Y - \\hat Y`` on
``W - \\hat W``, whose slope converges to the ATO under unconfoundedness. Identification of
the ATE, ATT and ATC requires unconfoundedness and overlap (strict overlap for the ATE;
``e(x) < 1`` suffices for the ATT and ``e(x) > 0`` for the ATC).

For `target = :all` inference uses the influence-function variance of the mean score
```math
\\hat V = \\frac{G}{G - 1}
    \\frac{\\sum_c \\{\\sum_{i \\in c} w_i(\\Gamma_i - \\hat\\theta)\\}^2}{(\\sum_i w_i)^2}
```
over clusters ``c`` (or units), with a t reference with `G - 1` degrees of freedom when the
forest was clustered. The ATT and ATC variances add the sampling variance of the averaged
CATEs to that of the AIPW correction, as in grf, and the overlap-weighted effect uses HC3
standard errors (cluster-robust HC1 with clusters). These estimators are root-n consistent
and asymptotically normal when the nuisances converge fast enough (the product of the
errors of ``\\hat Y``/``\\hat\\tau`` and ``\\hat W`` is ``o(n^{-1/2})``), which is the
usual double robustness of AIPW (Robins, Rotnitzky & Zhao 1994; Athey & Wager 2019).

Estimated propensities near 0 or 1 signal limited overlap, and a warning is issued. Scores
built from them remain finite as long as the propensities lie strictly inside ``(0, 1)``,
but they are dominated by a few large inverse weights, so the variance is inflated and the
normal approximation can be poor; finiteness does not repair the lack of overlap. Two
principled remedies change the estimand and should be reported as such: trimming the sample
to units with estimated propensities inside, say, ``[0.1, 0.9]`` via `subset` (Crump, Hotz,
Imbens & Mitnik 2009), which targets the effect in the trimmed population, and
`target = :overlap`, the overlap-weighted effect (Li, Morgan & Zaslavsky 2018), which
down-weights units with extreme propensities and is well defined under limited overlap.
When propensities approach only 0, the ATT may still be well identified, and when they
approach only 1, the ATC.

# Arguments
- `f`: a [`CausalForest`](@ref) or an [`InstrumentalForest`](@ref) (binary instrument; only
  `target = :all`).

# Keywords
- `target::Symbol = :all`: `:all` (ATE, average partial effect or average conditional
  LATE), `:treated` (ATT), `:control` (ATC) or `:overlap` (ATO); the last three are for
  causal forests, and `:treated` / `:control` need a binary treatment.
- `subset = nothing`: rows to average over (indices or a logical vector), for conditional
  averages in a subgroup or for trimming.
- `debiasing_weights = nothing`, `compliance_score = nothing`,
  `num_trees_for_weights::Integer = 500`, `rng = Random.Xoshiro(f.seed)`: passed to
  [`get_scores`](@ref).

# Returns
- [`HTEEstimate`](@ref) with one coefficient (`"ATE"`, `"ATT"`, `"ATC"`, `"ATO"` or
  `"LATE"`); `dof_residual` is `G - 1` with clusters and `Inf` otherwise, and `details`
  holds the target and the subset.

# Examples
```julia
using DrSnow, DataFrames, StableRNGs
rng = StableRNG(1)
n = 500
df = DataFrame(x1=rand(rng, n), x2=rand(rng, n))
df.d = Int.(rand(rng, n) .< 0.2 .+ 0.6 .* df.x2)
df.y = df.x2 .+ (1 .+ df.x1) .* df.d .+ randn(rng, n)
cf = causal_forest(df, :y, :d; covariates=[:x1, :x2], num_trees=200,
                   rng=StableRNG(2))
average_treatment_effect(cf)                         # ATE (truth 1.5)
average_treatment_effect(cf; target=:treated)        # ATT
average_treatment_effect(cf; target=:overlap)        # overlap-weighted
average_treatment_effect(cf; subset=df.x1 .> 0.5)    # conditional on a subgroup
keep = 0.1 .<= cf.W_hat .<= 0.9                      # trimming (changes the estimand)
average_treatment_effect(cf; subset=keep)
```

# References
- Athey, S., & Wager, S. (2019). Estimating treatment effects with causal forests: An
  application. *Observational Studies*, 5(2), 37–51.
- Crump, R. K., Hotz, V. J., Imbens, G. W., & Mitnik, O. A. (2009). Dealing with limited
  overlap in estimation of average treatment effects. *Biometrika*, 96(1), 187–199.
- Hahn, J. (1998). On the role of the propensity score in efficient semiparametric
  estimation of average treatment effects. *Econometrica*, 66(2), 315–331.
- Li, F., Morgan, K. L., & Zaslavsky, A. M. (2018). Balancing covariates via propensity
  score weighting. *Journal of the American Statistical Association*, 113(521), 390–400.
- Robins, J. M., Rotnitzky, A., & Zhao, L. P. (1994). Estimation of regression coefficients
  when some regressors are not always observed. *Journal of the American Statistical
  Association*, 89(427), 846–866.
"""
function average_treatment_effect(f::CausalForest; target::Symbol=:all, subset=nothing,
                                  debiasing_weights=nothing,
                                  num_trees_for_weights::Integer=500,
                                  rng::AbstractRNG=Random.Xoshiro(f.seed))
    ctx = "average_treatment_effect"
    target in (:all, :treated, :control, :overlap) ||
        throw(ArgumentError("$(ctx): target must be :all, :treated, :control or :overlap"))
    idx = _ml_grf_subset(f, subset, ctx)
    wobs = _ml_grf_obs_weights(f)[idx]
    cl = _ml_grf_clusters_or_units(f)[idx]
    length(unique(cl)) >= 2 ||
        throw(ArgumentError("$(ctx): the subset must contain more than one cluster"))
    binary = _ml_is_binary(f.W)
    binary && _ml_grf_overlap_warning(f.W_hat[idx], target)
    dof = f.cluster === nothing ? Inf : length(unique(cl)) - 1.0
    names = Dict(:all => "ATE", :treated => "ATT", :control => "ATC",
                 :overlap => "ATO")
    labels = Dict(:all => binary ? "average treatment effect" :
                          "average partial effect",
                  :treated => "average treatment effect on the treated",
                  :control => "average treatment effect on the controls",
                  :overlap => "overlap-weighted average treatment effect")
    mk(θ, v) = HTEEstimate([names[target]], [θ], fill(v, 1, 1), length(idx), dof,
                           labels[target], "Causal forest AIPW (grf)",
                           (target=target, subset=idx))
    if target === :all
        Γ = get_scores(f; subset=idx, debiasing_weights=debiasing_weights,
                       num_trees_for_weights=num_trees_for_weights, rng=rng)
        θ, V, _ = _ml_grf_wmean_vcov(reshape(Γ, :, 1), wobs, cl)
        return mk(θ[1], V[1, 1])
    end
    W, What, Y, Yhat = f.W[idx], f.W_hat[idx], f.Y[idx], f.Y_hat[idx]
    τ = f.predictions[idx]
    any(isnan, τ) && throw(ArgumentError("$(ctx): some out-of-bag predictions are " *
                                         "undefined; increase num_trees"))
    if target === :overlap
        any(w -> w <= 0, wobs) &&
            throw(ArgumentError("$(ctx): target = :overlap requires positive weights"))
        Z = hcat(ones(length(idx)), W .- What)
        clu = f.cluster === nothing ? nothing : cl
        β, V, _ = _ml_grf_lm(Z, Y .- Yhat, wobs, clu;
                             type=f.cluster === nothing ? :HC3 : :HC1,
                             hc=f.cluster === nothing)
        return mk(β[2], V[2, 2])
    end
    binary || throw(ArgumentError("$(ctx): target = :$(target) requires a binary " *
                                  "treatment; use :all or :overlap"))
    Y0 = Yhat .- What .* τ
    Y1 = Yhat .+ (1 .- What) .* τ
    tr = findall(==(1), W)
    co = findall(==(0), W)
    (isempty(tr) || isempty(co)) &&
        throw(ArgumentError("$(ctx): the subset needs treated and control units"))
    grp = target === :treated ? tr : co
    raw = sum(wobs[grp] .* τ[grp]) / sum(wobs[grp])
    rawvar = sum(wobs[grp] .^ 2 .* (τ[grp] .- raw) .^ 2) / sum(wobs[grp])^2
    gc = target === :treated ? What[co] ./ (1 .- What[co]) : ones(length(co))
    gt = target === :treated ? ones(length(tr)) : (1 .- What[tr]) ./ What[tr]
    γ = zeros(length(idx))
    γ[co] .= gc ./ sum(wobs[co] .* gc) .* sum(wobs)
    γ[tr] .= gt ./ sum(wobs[tr] .* gt) .* sum(wobs)
    dr = W .* γ .* (Y .- Y1) .- (1 .- W) .* γ .* (Y .- Y0)
    corr = sum(dr .* wobs) / sum(wobs)
    σ2 = if f.cluster !== nothing
        g, G = _ml_group_index_any(cl)
        S = zeros(G)
        wsum = zeros(G)
        for i in eachindex(dr)
            S[g[i]] += dr[i] * wobs[i]
            wsum[g[i]] += wobs[i]
        end
        nadj = count(>(0), wsum)
        sum(abs2, S) / sum(wobs)^2 * nadj / (nadj - 1)
    else
        m = count(>(0), wobs)
        sum(wobs .^ 2 .* dr .^ 2) / sum(wobs)^2 * m / (m - 1)
    end
    return mk(raw + corr, rawvar + σ2)
end

function average_treatment_effect(f::InstrumentalForest; target::Symbol=:all,
                                  subset=nothing, debiasing_weights=nothing,
                                  compliance_score=nothing,
                                  num_trees_for_weights::Integer=500,
                                  rng::AbstractRNG=Random.Xoshiro(f.seed))
    ctx = "average_treatment_effect"
    target === :all ||
        throw(ArgumentError("$(ctx): only target = :all is available for instrumental " *
                            "forests"))
    idx = _ml_grf_subset(f, subset, ctx)
    wobs = _ml_grf_obs_weights(f)[idx]
    cl = _ml_grf_clusters_or_units(f)[idx]
    Γ = get_scores(f; subset=idx, debiasing_weights=debiasing_weights,
                   compliance_score=compliance_score,
                   num_trees_for_weights=num_trees_for_weights, rng=rng)
    θ, V, G = _ml_grf_wmean_vcov(reshape(Γ, :, 1), wobs, cl)
    dof = f.cluster === nothing ? Inf : G - 1.0
    return HTEEstimate(["LATE"], θ, V, length(idx), dof,
                       "average conditional local average treatment effect",
                       "Instrumental forest AIPW (grf)", (target=:all, subset=idx))
end

# ------------------------------------------------------ best linear projection

"""
    best_linear_projection(f, A=nothing; subset=nothing, target=:all, vcov_type=:HC3,
                           debiasing_weights=nothing, compliance_score=nothing,
                           num_trees_for_weights=500, rng=Random.Xoshiro(f.seed))
        -> CATEProjection

Best linear projection of the conditional effect on a set of covariates, with doubly robust
inference (grf's `best_linear_projection`; Semenova & Chernozhukov 2021).

The estimand is the coefficient vector of the population least-squares projection of the
CATE on ``(1, A)``,
```math
\\beta = \\arg\\min_{b} E\\big[\\{\\tau(X_i) - (b_0 + A_i'b_1)\\}^2\\big],
```
which summarizes how the effect varies with ``A`` without assuming that ``\\tau`` is
linear. Because the doubly robust scores ``\\Gamma_i`` of [`get_scores`](@ref) satisfy
``E[\\Gamma_i \\mid X_i] \\approx \\tau(X_i)``, ``\\beta`` is estimated by the weighted OLS
regression of ``\\Gamma_i`` on ``(1, A_i)``, and the orthogonality of the scores makes the
usual sandwich covariance valid despite the machine-learned nuisances (Semenova &
Chernozhukov 2021). With `A = nothing` the intercept is the AIPW ATE. `target = :overlap`
weights units by ``\\hat W(1 - \\hat W)`` and projects the CATE in the overlap population
(Li, Morgan & Zaslavsky 2018); otherwise a warning is issued when estimated propensities
come within 0.01 of 0 or 1.

Standard errors are heteroskedasticity- and cluster-robust (`vcov_type = :HC3`, R's
`sandwich::vcovCL`, as in grf; or `:HC1`), clustered on the forest's clusters or on units.
The coefficients describe the projection, not causal effects of ``A``: ``A`` is a
descriptor of who benefits, and a non-zero slope can reflect variables correlated with
``A``. A projection on many covariates is noisy, and a zero slope does not show that
effects are homogeneous. For a nonparametric alternative based on sample splitting see
[`generic_ml`](@ref) (BLP and GATES).

# Arguments
- `f`: a [`CausalForest`](@ref) or an [`InstrumentalForest`](@ref).
- `A`: `nothing` (intercept only), a vector of covariate names of the forest, a numeric
  vector (one covariate) or a matrix / `DataFrame` with `n` (or subset-length) rows.

# Keywords
- `subset = nothing`: rows used (indices or logical vector).
- `target::Symbol = :all`: `:all` or `:overlap` (causal forests only).
- `vcov_type::Symbol = :HC3`: `:HC3` or `:HC1`.
- `debiasing_weights = nothing`, `compliance_score = nothing`,
  `num_trees_for_weights::Integer = 500`, `rng = Random.Xoshiro(f.seed)`: passed to
  [`get_scores`](@ref).

# Returns
- [`CATEProjection`](@ref) (a `CausalEstimate` with `coef`, `vcov`, `coeftable`);
  `dof_residual` is the residual degrees of freedom, or `G - 1` with clusters.

# Examples
```julia
using DrSnow, DataFrames, StableRNGs
rng = StableRNG(1)
df = DataFrame(x1=rand(rng, 500), x2=rand(rng, 500))
df.d = Int.(rand(rng, 500) .< 0.5)
df.y = (1 .+ 2 .* df.x1) .* df.d .+ randn(rng, 500)
cf = causal_forest(df, :y, :d; covariates=[:x1, :x2], num_trees=200,
                   rng=StableRNG(2))
coeftable(best_linear_projection(cf, [:x1, :x2]))   # slope on x1 near 2
```

# References
- Semenova, V., & Chernozhukov, V. (2021). Debiased machine learning of conditional average
  treatment effects and other causal functions. *The Econometrics Journal*, 24(2), 264–289.
- Athey, S., & Wager, S. (2019). Estimating treatment effects with causal forests: An
  application. *Observational Studies*, 5(2), 37–51.
- Li, F., Morgan, K. L., & Zaslavsky, A. M. (2018). Balancing covariates via propensity
  score weighting. *Journal of the American Statistical Association*, 113(521), 390–400.
"""
function best_linear_projection(f::Union{CausalForest,InstrumentalForest}, A=nothing;
                                subset=nothing, target::Symbol=:all,
                                vcov_type::Symbol=:HC3, debiasing_weights=nothing,
                                compliance_score=nothing,
                                num_trees_for_weights::Integer=500,
                                rng::AbstractRNG=Random.Xoshiro(f.seed))
    ctx = "best_linear_projection"
    target in (:all, :overlap) ||
        throw(ArgumentError("$(ctx): target must be :all or :overlap"))
    idx = _ml_grf_subset(f, subset, ctx)
    wobs = _ml_grf_obs_weights(f)
    w = wobs[idx]
    if target === :overlap
        f isa CausalForest ||
            throw(ArgumentError("$(ctx): target = :overlap requires a causal forest"))
        ow = f.W_hat .* (1 .- f.W_hat)
        idx = filter(i -> ow[i] > eps(), idx)
        w = wobs[idx] .* ow[idx]
    elseif f isa CausalForest && _ml_is_binary(f.W)
        lo, hi = extrema(f.W_hat[idx])
        (lo <= 0.01 || hi >= 0.99) &&
            @warn "Estimated treatment propensities range from $(round(lo; digits=3)) " *
                  "to $(round(hi; digits=3)); consider target = :overlap or a subset."
    end
    cl = _ml_grf_clusters_or_units(f)[idx]
    length(unique(cl)) >= 2 ||
        throw(ArgumentError("$(ctx): the subset must contain more than one cluster"))
    Γ = f isa CausalForest ?
        get_scores(f; subset=idx, debiasing_weights=debiasing_weights,
                   num_trees_for_weights=num_trees_for_weights, rng=rng) :
        get_scores(f; subset=idx, debiasing_weights=debiasing_weights,
                   compliance_score=compliance_score,
                   num_trees_for_weights=num_trees_for_weights, rng=rng)
    Amat, anames = _ml_grf_blp_design(f, A, idx, ctx)
    Z = hcat(ones(length(idx)), Amat)
    β, V, G = _ml_grf_lm(Z, Γ, w, cl; type=vcov_type)
    dof = f.cluster === nothing ? length(idx) - size(Z, 2) : G - 1.0
    return CATEProjection(vcat("(Intercept)", anames), β, V, length(idx), dof,
                          "Best linear projection of the CATE (" *
                          _ml_grf_label(f) * ", $(vcov_type))")
end

function _ml_grf_blp_design(f, A, idx, ctx)
    n = nobs(f)
    A === nothing && return zeros(length(idx), 0), String[]
    if A isa AbstractVector{Symbol}
        js = map(A) do a
            j = findfirst(==(a), f.covariates)
            j === nothing && throw(ArgumentError("$(ctx): $(a) is not a covariate of " *
                                                 "the forest; pass a matrix instead"))
            j
        end
        M = f.X[idx, js]
        any(isnan, M) && throw(ArgumentError("$(ctx): A has missing values"))
        return M, string.(A)
    end
    M, nm = if A isa AbstractDataFrame
        Matrix{Float64}(A), names(A)
    elseif A isa AbstractVector
        reshape(Float64.(A), :, 1), ["A1"]
    else
        Matrix{Float64}(A), ["A$j" for j in axes(A, 2)]
    end
    size(M, 1) == n && (M = M[idx, :])
    size(M, 1) == length(idx) ||
        throw(DimensionMismatch("$(ctx): A must have n or subset-length rows"))
    all(isfinite, M) || throw(ArgumentError("$(ctx): A must be finite"))
    return M, String.(nm)
end

"""
    cate_projection(f::Union{CausalForest,InstrumentalForest}; basis=f.covariates,
                    kwargs...) -> CATEProjection

Best linear projection of a causal (or instrumental) forest's conditional effect on the
covariates `basis`, with doubly robust inference.

This is a convenience method with the interface of
[`cate_projection`](@ref)`(::CATEPredictor)`, so that forests and DR-learners can be
summarized in the same way; it is identical to
[`best_linear_projection`](@ref)`(f, basis; kwargs...)`, whose documentation describes the
estimand, the estimator and its caveats (Semenova & Chernozhukov 2021).

# Arguments
- `f`: a [`CausalForest`](@ref) or an [`InstrumentalForest`](@ref).

# Keywords
- `basis::Vector{Symbol} = f.covariates`: covariates of the forest on which the effect is
  projected.
- `kwargs...`: passed to [`best_linear_projection`](@ref) (`subset`, `target`, `vcov_type`,
  …).

# Returns
- [`CATEProjection`](@ref)

# Examples
```julia
using DrSnow, DataFrames, StableRNGs
rng = StableRNG(1)
df = DataFrame(x1=rand(rng, 400), x2=rand(rng, 400))
df.d = Int.(rand(rng, 400) .< 0.5)
df.y = (1 .+ df.x1) .* df.d .+ randn(rng, 400)
cf = causal_forest(df, :y, :d; covariates=[:x1, :x2], num_trees=200,
                   rng=StableRNG(2))
cate_projection(cf; basis=[:x1])
```

# References
- Semenova, V., & Chernozhukov, V. (2021). Debiased machine learning of conditional average
  treatment effects and other causal functions. *The Econometrics Journal*, 24(2), 264–289.
"""
cate_projection(f::Union{CausalForest,InstrumentalForest}; basis=f.covariates, kwargs...) =
    best_linear_projection(f, Symbol.(collect(basis)); kwargs...)

# ------------------------------------------------------------ calibration

"""
    test_calibration(f::Union{CausalForest,RegressionForest}; vcov_type=:HC3)
        -> DiagnosticTest

Omnibus test of whether a forest's predictions are calibrated and, for a causal forest,
whether it detects treatment-effect heterogeneity (grf's `test_calibration`).

For a causal forest the test is the best-linear-predictor regression of Chernozhukov,
Demirer, Duflo and Fernández-Val (2025) applied to the out-of-bag forest predictions: with
``\\bar \\tau`` the weighted mean of ``\\hat\\tau(X_i)``, the centred outcome is regressed,
with observation weights and without intercept, on the mean and the differential forest
prediction,
```math
Y_i - \\hat Y_i = \\alpha \\, \\bar \\tau (W_i - \\hat W_i)
    + \\beta \\, \\{\\hat\\tau(X_i) - \\bar \\tau\\}(W_i - \\hat W_i) + \\varepsilon_i .
```
A coefficient ``\\alpha = 1`` indicates that the average prediction is correct, and
``\\beta = 1`` that the predicted heterogeneity is well calibrated; ``\\beta > 0``
indicates that the forest's CATE estimates are positively associated with the true effects,
that is, that it detects heterogeneity. For a regression forest the target is ``Y`` and the
regressors are ``\\bar \\mu`` and ``\\hat\\mu(X_i) - \\bar \\mu``.

The returned test is the one-sided t test of ``H_0: \\beta \\le 0`` with cluster- (or
unit-) robust `vcov_type` standard errors and ``n - 2`` degrees of freedom, as in grf; both
coefficients, their standard errors and one-sided p-values are in `details.table`. Because
out-of-bag predictions are used, the test is an approximation: out-of-bag predictions are
not fully independent of the outcome they are evaluated on. A rejection is evidence that
the forest captures some heterogeneity; a non-rejection does not show that effects are
homogeneous, only that this forest did not detect heterogeneity with this sample. For
inference on heterogeneity with explicit sample splitting see [`generic_ml`](@ref) and
[`rank_average_treatment_effect`](@ref).

# Arguments
- `f`: a [`CausalForest`](@ref) or a [`RegressionForest`](@ref) with positive observation
  weights and defined out-of-bag predictions.

# Keywords
- `vcov_type::Symbol = :HC3`: `:HC3` (default, as grf) or `:HC1`.

# Returns
- `DiagnosticTest` whose statistic is the t statistic of the differential coefficient;
  `details.table` holds both coefficients and `details.vcov` their covariance.

# Examples
```julia
using DrSnow, DataFrames, StableRNGs
rng = StableRNG(1)
df = DataFrame(x1=rand(rng, 500), x2=rand(rng, 500))
df.d = Int.(rand(rng, 500) .< 0.5)
df.y = (4 .* df.x1 .- 2) .* df.d .+ randn(rng, 500)
cf = causal_forest(df, :y, :d; covariates=[:x1, :x2], num_trees=200,
                   rng=StableRNG(2))
t = test_calibration(cf)
t.details.table
```

# References
- Chernozhukov, V., Demirer, M., Duflo, E., & Fernández-Val, I. (2025). Fisher– Schultz
  lecture: Generic machine learning inference on heterogeneous treatment effects in
  randomized experiments, with an application to immunization in India. *Econometrica*,
  93(4), 1121–1164.
- Athey, S., & Wager, S. (2019). Estimating treatment effects with causal forests: An
  application. *Observational Studies*, 5(2), 37–51.
"""
function test_calibration(f::Union{CausalForest,RegressionForest};
                          vcov_type::Symbol=:HC3)
    w = _ml_grf_obs_weights(f)
    any(<=(0), w) && throw(ArgumentError("test_calibration: weights must be positive"))
    preds = f.predictions
    any(isnan, preds) && throw(ArgumentError("test_calibration: undefined out-of-bag " *
                                             "predictions; increase num_trees"))
    m = sum(w .* preds) / sum(w)
    if f isa CausalForest
        wc = f.W .- f.W_hat
        yv = f.Y .- f.Y_hat
        Z = hcat(wc .* m, wc .* (preds .- m))
    else
        yv = copy(f.Y)
        Z = hcat(fill(m, length(preds)), preds .- m)
    end
    β, V, G = _ml_grf_lm(Z, yv, w, _ml_grf_clusters_or_units(f); type=vcov_type)
    se = sqrt.(max.(diag(V), 0.0))
    t = β ./ se
    dof = length(yv) - 2
    p1 = [t[j] < 0 ? 1 - two_sided_pvalue(t[j], dof) / 2 : two_sided_pvalue(t[j], dof) / 2
          for j in 1:2]
    tab = DataFrame(term=["mean.forest.prediction", "differential.forest.prediction"],
                    estimate=β, std_error=se, t=t, p_value_one_sided=p1)
    null = f isa CausalForest ?
           "the coefficient on the differential forest prediction is ≤ 0 (the forest " *
           "detects no treatment-effect heterogeneity)" :
           "the coefficient on the differential forest prediction is ≤ 0 (the forest " *
           "predictions do not track the outcome)"
    return DiagnosticTest("Calibration test (" * _ml_grf_label(f) * ")", null,
                          t[2], p1[2]; dof=(dof,),
                          method="weighted OLS on out-of-bag predictions, " *
                                 "$(vcov_type) cluster-robust SEs, one-sided t test",
                          note="A mean-prediction coefficient near 1 indicates a " *
                               "correct average prediction. Non-rejection does not " *
                               "show that effects are homogeneous.",
                          details=(table=tab, vcov=V))
end

# ------------------------------------------------------ variable importance

"""
    split_frequencies(f::GeneralizedRandomForest; max_depth=4) -> Matrix{Int}

Count the splits on each covariate at each depth of a forest's trees (grf's
`split_frequencies`).

Entry `[d, j]` is the number of splits on covariate `j` at depth `d` summed over all trees,
with the root at depth 1. The counts are a descriptive summary of the fitted trees and the
input of [`variable_importance`](@ref); they say which variables the splitting rule used,
not which variables causally modify the effect.

# Arguments
- `f::GeneralizedRandomForest`: a fitted forest.

# Keywords
- `max_depth::Integer = 4`: deepest level counted (at least 1).

# Returns
- `Matrix{Int}` of size `max_depth × p`, with columns in the order of `f.covariates`.

# Examples
```julia
using DrSnow, DataFrames, StableRNGs
rng = StableRNG(1)
df = DataFrame(x1=rand(rng, 300), x2=rand(rng, 300))
df.y = 3 .* df.x1 .+ randn(rng, 300)
rf = regression_forest(df, :y; covariates=[:x1, :x2], num_trees=100,
                       rng=StableRNG(2))
split_frequencies(rf; max_depth=2)
```
"""
function split_frequencies(f::GeneralizedRandomForest; max_depth::Integer=4)
    max_depth >= 1 || throw(ArgumentError("split_frequencies: max_depth must be ≥ 1"))
    return _grf_split_frequencies(f.forest, Int(max_depth))
end

"""
    variable_importance(f::GeneralizedRandomForest; decay_exponent=2, max_depth=4)
        -> DataFrame

Split-frequency variable importance of a forest (grf's `variable_importance`).

At each depth ``d \\le`` `max_depth` the share of splits made on each covariate is computed
from [`split_frequencies`](@ref), and the shares are averaged over depths with weights
proportional to ``d^{-\\text{decay\\_exponent}}``, so that splits near the root count more.
For a causal forest this measures how often the forest uses a covariate to split on
treatment-effect heterogeneity, and it is a useful description of which variables drive the
estimated heterogeneity (Athey & Wager 2019).

The measure is descriptive, not a test, and it has known biases: it favours covariates with
many distinct values, it can split importance arbitrarily among correlated covariates, and
a variable can be important for the fit without modifying the effect causally. To test
heterogeneity along a covariate use [`best_linear_projection`](@ref); for omnibus tests use
[`test_calibration`](@ref) or [`rank_average_treatment_effect`](@ref).

# Arguments
- `f::GeneralizedRandomForest`: a fitted forest.

# Keywords
- `decay_exponent::Real = 2`: exponent of the depth weights; larger values concentrate on
  the top of the trees.
- `max_depth::Integer = 4`: deepest level used.

# Returns
- `DataFrame` with columns `variable` and `importance` (non-negative, summing to at most
  1), in covariate order.

# Examples
```julia
using DrSnow, DataFrames, StableRNGs
rng = StableRNG(1)
df = DataFrame(x1=rand(rng, 400), x2=rand(rng, 400), x3=rand(rng, 400))
df.d = Int.(rand(rng, 400) .< 0.5)
df.y = (1 .+ 2 .* df.x1) .* df.d .+ randn(rng, 400)
cf = causal_forest(df, :y, :d; covariates=[:x1, :x2, :x3], num_trees=200,
                   rng=StableRNG(2))
sort(variable_importance(cf), :importance; rev=true)
```

# References
- Athey, S., & Wager, S. (2019). Estimating treatment effects with causal forests: An
  application. *Observational Studies*, 5(2), 37–51.
- Athey, S., Tibshirani, J., & Wager, S. (2019). Generalized random forests. *The Annals of
  Statistics*, 47(2), 1148–1178.
"""
function variable_importance(f::GeneralizedRandomForest; decay_exponent::Real=2,
                             max_depth::Integer=4)
    S = Float64.(split_frequencies(f; max_depth=max_depth))
    S ./= max.(1.0, sum(S; dims=2))
    wts = (1:size(S, 1)) .^ (-float(decay_exponent))
    imp = vec(S' * wts) ./ sum(wts)
    return DataFrame(variable=string.(f.covariates), importance=imp)
end

# ------------------------------------------------------------------- RATE

"""
    RATEEstimate <: CausalEstimate

Rank-weighted average treatment effect(s) returned by
[`rank_average_treatment_effect`](@ref).

The coefficients are the RATE (AUTOC or Qini) of each prioritization rule and, with two
rules, their difference; the covariance comes from the half-sample bootstrap, and inference
uses the normal reference. The Targeting Operator Characteristic curve on the requested
grid is stored alongside. Supports `coef`, `vcov`, `stderror`, `confint`, `coeftable`,
`nobs`, `estimand` and `method_name`.

# Fields
- `names::Vector{String}`: coefficient names (`"priority1 | AUTOC"`, …, and
  `"priority1 - priority2 | AUTOC"` with two rules).
- `coef::Vector{Float64}`, `vcov::Matrix{Float64}`: RATE estimates and their bootstrap
  covariance.
- `n::Int`: number of scored observations.
- `target::Symbol`: `:AUTOC` or `:QINI`.
- `toc::DataFrame`: the Targeting Operator Characteristic ``\\mathrm{TOC}(q)`` on the grid
  `q`, with columns `q`, `priority`, `estimate` and `std_error`.
- `R::Int`: number of bootstrap replications.
- `draws::Matrix{Float64}`: bootstrap draws of the coefficients (`R ×` number of
  coefficients).
"""
struct RATEEstimate <: CausalEstimate
    names::Vector{String}
    coef::Vector{Float64}
    vcov::Matrix{Float64}
    n::Int
    target::Symbol
    toc::DataFrame
    R::Int
    draws::Matrix{Float64}
end

StatsAPI.coef(r::RATEEstimate) = r.coef
StatsAPI.vcov(r::RATEEstimate) = r.vcov
StatsAPI.coefnames(r::RATEEstimate) = r.names
StatsAPI.nobs(r::RATEEstimate) = r.n
estimand(r::RATEEstimate) = "rank average treatment effect ($(r.target))"
method_name(::RATEEstimate) = "RATE (Yadlowsky et al.), half-sample bootstrap"

function show_details(io::IO, r::RATEEstimate)
    println(io)
    println(io, "Bootstrap: $(r.R) half-samples. TOC(q) = mean effect among the top " *
                "q share by priority minus the ATE (see `r.toc`).")
    return nothing
end

"""
    rank_average_treatment_effect(f, priorities; target=:AUTOC, q=0.1:0.1:1.0, R=200,
                                  subset=nothing, debiasing_weights=nothing,
                                  compliance_score=nothing, num_trees_for_weights=500,
                                  rng=Random.default_rng()) -> RATEEstimate
    rank_average_treatment_effect(scores::AbstractVector, priorities; target=:AUTOC,
                                  q=0.1:0.1:1.0, R=200, weights=nothing,
                                  cluster=nothing, rng=Random.default_rng())
        -> RATEEstimate

Evaluate a treatment prioritization rule by its rank-weighted average treatment effect
(Yadlowsky, Fleming, Shah, Brunskill & Wager 2025; grf's `rank_average_treatment_effect`).

A prioritization rule ``S(X)`` ranks units for treatment (for example a CATE estimate, a
risk score or a baseline covariate). Its Targeting Operator Characteristic compares the
average effect among the top-``q`` share of units under ``S`` with the overall average
effect,
```math
\\mathrm{TOC}(q) = E[Y_i(1) - Y_i(0) \\mid F_S(S(X_i)) \\ge 1 - q] - E[Y_i(1) - Y_i(0)],
```
and the RATE integrates it: ``\\mathrm{AUTOC} = \\int_0^1 \\mathrm{TOC}(q)\\,dq``, which
emphasizes the top of the ranking, or
``\\mathrm{QINI} = \\int_0^1 q\\,\\mathrm{TOC}(q)\\,dq``, which weighs all shares more
evenly and is more powerful when benefits are spread out. A RATE is zero when the rule is
unrelated to the effects and positive when it ranks units with larger effects first, so it
serves both to evaluate rules and to test for heterogeneity along ``S``.

The TOC is estimated by sorting the doubly robust scores ``\\Gamma_i`` of
[`get_scores`](@ref) by decreasing priority (tied priorities are averaged) and taking
weighted running means; with a vector of scores the scores are used directly. Standard
errors come from the half-sample bootstrap over units (or clusters) with `R` replications,
as in grf; with two rules the bootstrap is paired and their difference is reported, which
gives a test of whether one rule prioritizes better than the other. Validity requires that
the priorities were not fitted on the observations whose scores are evaluated: use a forest
trained on a separate sample or cross-fitted predictions. Out-of-bag predictions
`predict(cf)` are accepted, as in grf, but reuse the same data and can make the test
slightly anticonservative. The scores inherit the overlap requirements of
[`get_scores`](@ref); propensities within 0.05 of 0 or 1 trigger a warning.

# Arguments
- `f`: a [`CausalForest`](@ref) or [`InstrumentalForest`](@ref) with a binary treatment; or
  `scores`, a vector of doubly robust scores computed elsewhere.
- `priorities`: a vector (length `n` or the subset length; larger values mean higher
  priority), a two-column matrix or `DataFrame`, or a tuple of two vectors.

# Keywords
- `target::Symbol = :AUTOC`: `:AUTOC` or `:QINI`.
- `q = 0.1:0.1:1.0`: increasing grid in `(0, 1]` ending at 1 on which the TOC is reported.
- `R::Integer = 200`: bootstrap replications (`0` or `1` gives no standard errors).
- `rng::AbstractRNG = Random.default_rng()`: generator of the bootstrap draws.
- `subset = nothing`, `debiasing_weights = nothing`, `compliance_score = nothing`,
  `num_trees_for_weights::Integer = 500`: as in [`get_scores`](@ref) (forest method).
- `weights = nothing`, `cluster = nothing`: positive observation weights and cluster
  identifiers (scores method).

# Returns
- [`RATEEstimate`](@ref); `coeftable(r)` gives the RATE with bootstrap standard errors and
  `r.toc` the TOC curve.

# Examples
```julia
using DrSnow, DataFrames, StableRNGs
rng = StableRNG(1)
n = 500
df = DataFrame(x1=rand(rng, n), x2=rand(rng, n))
df.d = Int.(rand(rng, n) .< 0.5)
df.y = 2 .* df.x1 .* df.d .+ randn(rng, n)
tr, ho = df[1:250, :], df[251:end, :]
cf_train = causal_forest(tr, :y, :d; covariates=[:x1, :x2], num_trees=200,
                         rng=StableRNG(2))
cf_eval = causal_forest(ho, :y, :d; covariates=[:x1, :x2], num_trees=200,
                        rng=StableRNG(3))
rate = rank_average_treatment_effect(cf_eval, predict(cf_train, ho);
                                     R=100, rng=StableRNG(4))
coeftable(rate)
```

# References
- Yadlowsky, S., Fleming, S., Shah, N., Brunskill, E., & Wager, S. (2025). Evaluating
  treatment prioritization rules via rank-weighted average treatment effects. *Journal of
  the American Statistical Association*, 120(549), 38–51.
- Athey, S., & Wager, S. (2019). Estimating treatment effects with causal forests: An
  application. *Observational Studies*, 5(2), 37–51.
"""
function rank_average_treatment_effect(f::Union{CausalForest,InstrumentalForest},
                                       priorities; target::Symbol=:AUTOC,
                                       q=0.1:0.1:1.0, R::Integer=200, subset=nothing,
                                       debiasing_weights=nothing, compliance_score=nothing,
                                       num_trees_for_weights::Integer=500,
                                       rng::AbstractRNG=Random.default_rng())
    ctx = "rank_average_treatment_effect"
    _ml_is_binary(f.W) || throw(ArgumentError("$(ctx): requires a binary treatment"))
    idx0 = _ml_grf_subset(f, subset, ctx)
    wobs = _ml_grf_obs_weights(f)
    idx = filter(i -> wobs[i] > 0, idx0)
    any(p -> p == 0 || p == 1, f.W_hat[idx]) &&
        throw(ArgumentError("$(ctx): some estimated propensities are exactly 0 or 1"))
    lo, hi = extrema(f.W_hat[idx])
    (lo <= 0.05 || hi >= 0.95) &&
        @warn "Estimated treatment propensities range from $(round(lo; digits=3)) to " *
              "$(round(hi; digits=3)); consider restricting the sample with `subset`."
    keep = [i in Set(idx) for i in idx0]
    P, pnames = _ml_rate_priorities(priorities, nobs(f), idx0, ctx)
    P = P[keep, :]
    pos(v) = v === nothing ? nothing :
             (length(v) == nobs(f) ? v[idx] : collect(v)[keep])
    Γ = f isa CausalForest ?
        get_scores(f; subset=idx, debiasing_weights=pos(debiasing_weights),
                   num_trees_for_weights=num_trees_for_weights) :
        get_scores(f; subset=idx, debiasing_weights=pos(debiasing_weights),
                   compliance_score=pos(compliance_score),
                   num_trees_for_weights=num_trees_for_weights)
    cl = _ml_grf_clusters_or_units(f)[idx]
    return _ml_rate(Γ, P, pnames, wobs[idx], cl, target, q, R, rng, ctx)
end

function rank_average_treatment_effect(scores::AbstractVector, priorities;
                                       target::Symbol=:AUTOC, q=0.1:0.1:1.0,
                                       R::Integer=200, weights=nothing, cluster=nothing,
                                       rng::AbstractRNG=Random.default_rng())
    ctx = "rank_average_treatment_effect"
    Γ = Float64.(collect(scores))
    all(isfinite, Γ) || throw(ArgumentError("$(ctx): scores must be finite"))
    n = length(Γ)
    P, pnames = _ml_rate_priorities(priorities, n, collect(1:n), ctx)
    w = if weights === nothing
        ones(n)
    else
        length(weights) == n ||
            throw(DimensionMismatch("$(ctx): weights must have the length of scores"))
        all(>(0), weights) || throw(ArgumentError("$(ctx): weights must be positive"))
        Float64.(weights) ./ sum(weights)
    end
    cl = cluster === nothing ? collect(1:n) : collect(cluster)
    length(cl) == n || throw(DimensionMismatch("$(ctx): cluster has the wrong length"))
    return _ml_rate(Γ, P, pnames, w, cl, target, q, R, rng, ctx)
end

function _ml_rate_priorities(priorities, n, idx, ctx)
    cols, names = if priorities isa Tuple
        length(priorities) == 2 ||
            throw(ArgumentError("$(ctx): give one or two priority rules"))
        [collect(priorities[1]), collect(priorities[2])], ["priority1", "priority2"]
    elseif priorities isa AbstractDataFrame
        [collect(c) for c in eachcol(priorities)], names(priorities)
    elseif priorities isa AbstractMatrix
        [priorities[:, j] for j in axes(priorities, 2)],
        ["priority$j" for j in axes(priorities, 2)]
    else
        [collect(priorities)], ["priority1"]
    end
    1 <= length(cols) <= 2 || throw(ArgumentError("$(ctx): give one or two priority rules"))
    P = zeros(Int, length(idx), length(cols))
    for (j, c) in enumerate(cols)
        any(ismissing, c) && throw(ArgumentError("$(ctx): priorities have missing values"))
        v = if length(c) == n
            c[idx]
        elseif length(c) == length(idx)
            c
        else
            throw(DimensionMismatch("$(ctx): priorities must have length n or the " *
                                    "subset length"))
        end
        u = sort(unique(v))
        rk = Dict(x => i for (i, x) in enumerate(u))
        P[:, j] = [rk[x] for x in v]
    end
    return P, String.(names)
end

function _ml_rate(Γ, P, pnames, w, cl, target, q, R, rng, ctx)
    target in (:AUTOC, :QINI) || throw(ArgumentError("$(ctx): target must be :AUTOC " *
                                                     "or :QINI"))
    qs = Float64.(collect(q))
    (issorted(qs; lt=<=) && first(qs) > 0 && last(qs) == 1 && allunique(qs)) ||
        throw(ArgumentError("$(ctx): q must be an increasing grid in (0, 1] ending at 1"))
    R >= 0 || throw(ArgumentError("$(ctx): R must be non-negative"))
    g, G = _ml_group_index_any(cl)
    G >= 2 || throw(ArgumentError("$(ctx): need units from at least two clusters"))
    dw = Γ .* w
    np = size(P, 2)
    stat(ix) = reduce(vcat, [_ml_rate_stat(dw, w, view(P, :, j), ix, qs, target)
                             for j in 1:np])
    t0 = stat(collect(eachindex(Γ)))
    members = [Int[] for _ in 1:G]
    for i in eachindex(g)
        push!(members[g[i]], i)
    end
    nbs = G ÷ 2
    T = zeros(R, length(t0))
    for r in 1:R
        pick = randperm(rng, G)[1:nbs]
        ix = reduce(vcat, members[pick])
        T[r, :] = stat(ix)
    end
    k = length(qs) + 1
    est = reshape(t0, k, np)
    names = copy(pnames)
    if np == 2
        est = hcat(est, est[:, 1] .- est[:, 2])
        push!(names, pnames[1] * " - " * pnames[2])
        T = hcat(T, T[:, 1:k] .- T[:, (k + 1):(2k)])
    end
    nc = size(est, 2)
    D = T[:, [(j - 1) * k + 1 for j in 1:nc]]
    V = R >= 2 ? cov(D) : zeros(nc, nc)
    sds = R >= 2 ? vec(std(T; dims=1)) : zeros(size(T, 2))
    toc = DataFrame(q=repeat(qs, nc), priority=repeat(names; inner=length(qs)),
                    estimate=vec(est[2:end, :]),
                    std_error=vcat([sds[((j - 1) * k + 2):(j * k)] for j in 1:nc]...))
    coefv = est[1, :]
    coefv[abs.(coefv) .< 1e-15] .= 0.0
    return RATEEstimate(names .* " | $(target)", coefv, Matrix(Symmetric(V)),
                        length(Γ), target, toc, Int(R), D)
end

# grf's estimate_rate: RATE and TOC on the grid for the observations `ix`.
function _ml_rate_stat(dw, w, prio, ix, qs, target)
    p = prio[ix]
    sidx = sortperm(p; rev=true, alg=MergeSort)
    sw = w[ix][sidx]
    ps = p[sidx]
    dws = dw[ix][sidx]
    m = length(ix)
    DR = similar(sw)
    if allunique(p)
        DR .= dws ./ sw
    else
        a = 1
        while a <= m
            b = a
            while b < m && ps[b + 1] == ps[a]
                b += 1
            end
            v = sum(view(dws, a:b)) / sum(view(sw, a:b))
            DR[a:b] .= v
            a = b + 1
        end
    end
    cw = cumsum(sw)
    W = cw[end]
    C1 = cumsum(DR .* sw)
    ate = C1[end] / W
    toc = C1 ./ cw .- ate
    rate = target === :AUTOC ? sum(toc .* sw) / sum(sw) :
           sum(cw ./ W .* sw .* toc) / sum(sw)
    nw = qs .* W
    idx = [searchsortedlast(cw, x + 1e-15) for x in nw]
    mx = maximum(idx)
    den_adj = nw .- cw[max.(idx, 1)]
    num_adj = den_adj .* DR[min.(idx .+ 1, mx)]
    idx = max.(idx, 1)
    tocg = (C1[idx] .+ num_adj) ./ (cw[idx] .+ den_adj) .- ate
    return vcat(rate, tocg)
end

# ------------------------------------------------------------- policy values

"""
    double_robust_scores(f::CausalForest) -> Matrix{Float64}

Arm-specific doubly robust scores of the potential-outcome means from a causal forest with
a binary treatment (policytree's `double_robust_scores`).

For arm ``w \\in \\{0, 1\\}`` the score of unit ``i`` is the AIPW score
```math
\\Gamma_i(w) = \\hat\\mu_w(X_i)
    + \\frac{1\\{W_i = w\\}\\{Y_i - \\hat\\mu_w(X_i)\\}}{\\hat P(W_i = w \\mid X_i)},
```
with ``\\hat\\mu_0 = \\hat Y - \\hat W \\hat\\tau`` and
``\\hat\\mu_1 = \\hat Y + (1 - \\hat W) \\hat\\tau`` built from the forest's out-of-bag
centering estimates and CATEs, and ``\\hat P(W = 1 \\mid X) = \\hat W``. Under
unconfoundedness and overlap ``E[\\Gamma_i(w)] = E[Y_i(w)]`` whenever either the outcome
regressions or the propensity score are correctly estimated (Robins, Rotnitzky & Zhao
1994), so the mean of ``\\Gamma_i(\\pi(X_i))`` estimates the value ``E[Y_i(\\pi(X_i))]`` of
a treatment rule ``\\pi``. These scores are the objective of doubly robust empirical
welfare maximization (Athey & Wager 2021; Zhou, Athey & Wager 2023): pass them to
[`policy_tree`](@ref) to learn a rule, or use [`policy_value`](@ref) to evaluate one. They
require estimated propensities strictly inside ``(0, 1)``; values near the boundary produce
large scores and noisy welfare estimates.

# Arguments
- `f::CausalForest`: a causal forest with a binary treatment.

# Returns
- `Matrix{Float64}` of size `n × 2` with columns `[control, treated]`, in the row order of
  the training data.

# Examples
```julia
using DrSnow, DataFrames, StableRNGs
rng = StableRNG(1)
df = DataFrame(x1=rand(rng, 400), x2=rand(rng, 400))
df.d = Int.(rand(rng, 400) .< 0.5)
df.y = (df.x1 .- 0.5) .* df.d .+ randn(rng, 400)
cf = causal_forest(df, :y, :d; covariates=[:x1, :x2], num_trees=200,
                   rng=StableRNG(2))
Γ = double_robust_scores(cf)
tree = policy_tree(Γ, cf.X; depth=1, covariates=cf.covariates)
```

# References
- Athey, S., & Wager, S. (2021). Policy learning with observational data. *Econometrica*,
  89(1), 133–161.
- Robins, J. M., Rotnitzky, A., & Zhao, L. P. (1994). Estimation of regression coefficients
  when some regressors are not always observed. *Journal of the American Statistical
  Association*, 89(427), 846–866.
- Zhou, Z., Athey, S., & Wager, S. (2023). Offline multi-action policy learning:
  Generalization and optimization. *Operations Research*, 71(1), 148–183.
- Sverdrup, E., Kanodia, A., Zhou, Z., Athey, S., & Wager, S. (2020). policytree: Policy
  learning via doubly robust empirical welfare maximization over trees. *Journal of Open
  Source Software*, 5(50), 2232.
"""
function double_robust_scores(f::CausalForest)
    _ml_is_binary(f.W) ||
        throw(ArgumentError("double_robust_scores: requires a binary treatment"))
    τ = f.predictions
    any(isnan, τ) && throw(ArgumentError("double_robust_scores: undefined out-of-bag " *
                                         "predictions"))
    μ0 = f.Y_hat .- f.W_hat .* τ
    μ1 = f.Y_hat .+ (1 .- f.W_hat) .* τ
    Γ0 = μ0 .+ (1 .- f.W) .* (f.Y .- μ0) ./ (1 .- f.W_hat)
    Γ1 = μ1 .+ f.W .* (f.Y .- μ1) ./ f.W_hat
    return hcat(Γ0, Γ1)
end

"""
    policy_value(f::CausalForest, policy; subset=nothing) -> HTEEstimate

Doubly robust estimate of the mean outcome under a treatment rule and of its gains over
treating everyone and treating no one.

For a treatment rule ``\\pi: \\mathcal{X} \\to \\{0, 1\\}`` the estimand is the policy
value ``V(\\pi) = E[Y_i(\\pi(X_i))]``, together with the contrasts ``V(\\pi) - E[Y_i(1)]``
and ``V(\\pi) - E[Y_i(0)]``. It is identified under unconfoundedness and overlap and
estimated by the (observation-weighted) mean of ``\\Gamma_i(\\pi(X_i))``, with the
arm-specific AIPW scores of [`double_robust_scores`](@ref) (Robins, Rotnitzky & Zhao 1994);
the contrasts use the differences of scores, so their standard errors account for the
common noise. This is the policy evaluation step of the doubly robust empirical welfare
maximization framework of Athey and Wager (2021) and Zhou, Athey and Wager (2023), cf.
Kitagawa and Tetenov (2018) for the experimental case.

Inference uses the cluster-robust (or unit-level) variance of the weighted mean of the
scores, with a t reference with `G - 1` degrees of freedom when the forest was clustered.
The estimate is valid only for a rule chosen independently of the evaluation data: a rule
learned on the same observations (for example a [`policy_tree`](@ref) fitted to the same
scores) has an optimistic in-sample value, so evaluate rules learned on a separate sample
or out of fold. The value refers to the population the forest was trained on; it does not
account for spillovers or general-equilibrium effects of scaling up the rule.

# Arguments
- `f::CausalForest`: a causal forest with a binary treatment.
- `policy`: a 0/1 (or `Bool`) vector of treatment assignments of length `n`, in the row
  order of the training data, or a [`PolicyTree`](@ref) whose covariates are covariates of
  the forest.

# Keywords
- `subset = nothing`: rows on which the rule is evaluated (indices or logical vector).

# Returns
- [`HTEEstimate`](@ref) with coefficients `value(policy)`,
  `value(policy) - value(treat all)` and `value(policy) - value(treat none)`;
  `details.share_treated` is the share assigned to treatment.

# Examples
```julia
using DrSnow, DataFrames, StableRNGs
rng = StableRNG(1)
n = 500
df = DataFrame(x1=rand(rng, n), x2=rand(rng, n))
df.d = Int.(rand(rng, n) .< 0.5)
df.y = (df.x1 .- 0.5) .* df.d .+ randn(rng, n)
tr, ho = df[1:250, :], df[251:end, :]
cf_train = causal_forest(tr, :y, :d; covariates=[:x1, :x2], num_trees=200,
                         rng=StableRNG(2))
cf_eval = causal_forest(ho, :y, :d; covariates=[:x1, :x2], num_trees=200,
                        rng=StableRNG(3))
policy_value(cf_eval, predict(cf_train, ho) .> 0)
```

# References
- Athey, S., & Wager, S. (2021). Policy learning with observational data. *Econometrica*,
  89(1), 133–161.
- Zhou, Z., Athey, S., & Wager, S. (2023). Offline multi-action policy learning:
  Generalization and optimization. *Operations Research*, 71(1), 148–183.
- Robins, J. M., Rotnitzky, A., & Zhao, L. P. (1994). Estimation of regression coefficients
  when some regressors are not always observed. *Journal of the American Statistical
  Association*, 89(427), 846–866.
- Kitagawa, T., & Tetenov, A. (2018). Who should be treated? Empirical welfare maximization
  methods for treatment choice. *Econometrica*, 86(2), 591–616.
"""
function policy_value(f::CausalForest, policy; subset=nothing)
    ctx = "policy_value"
    idx = _ml_grf_subset(f, subset, ctx)
    n = nobs(f)
    π = if policy isa PolicyTree
        js = map(policy.covariates) do c
            j = findfirst(==(c), f.covariates)
            j === nothing && throw(ArgumentError("$(ctx): policy covariate $(c) is not " *
                                                 "a covariate of the forest"))
            j
        end
        StatsAPI.predict(policy, f.X[:, js])
    else
        length(policy) == n ||
            throw(DimensionMismatch("$(ctx): policy must have length $n"))
        collect(policy)
    end
    all(a -> a == 0 || a == 1, π) ||
        throw(ArgumentError("$(ctx): policy must assign 0 (control) or 1 (treat)"))
    Γ = double_robust_scores(f)[idx, :]
    a = Int.(π[idx])
    v = [Γ[i, a[i] + 1] for i in eachindex(a)]
    Ψ = hcat(v, v .- Γ[:, 2], v .- Γ[:, 1])
    w = _ml_grf_obs_weights(f)[idx]
    cl = _ml_grf_clusters_or_units(f)[idx]
    θ, V, G = _ml_grf_wmean_vcov(Ψ, w, cl)
    dof = f.cluster === nothing ? Inf : G - 1.0
    return HTEEstimate(["value(policy)", "value(policy) - value(treat all)",
                        "value(policy) - value(treat none)"], θ, V, length(idx), dof,
                       "value of the treatment rule", "Doubly robust policy evaluation",
                       (share_treated=mean(a),))
end
