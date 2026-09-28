# Matrix completion with nuclear-norm penalisation (MC-NNM; Athey, Bayati, Doudchenko,
# Imbens & Khosravi 2021). The untreated potential outcomes are modelled as
# L + u 1' + 1 v' with unpenalised unit (u) and time (v) effects, fitted on untreated
# cells by the coordinate-descent / soft-impute algorithm of the authors' R package
# MCPanel (`mcnnm_fit`, `mcnnm_cv`).

"""
    MatrixCompletionEstimate <: CausalEstimate

Result of [`matrix_completion`](@ref): a nuclear-norm matrix completion (MC-NNM) estimate
of the average effect on treated unit-periods.

The coefficient `"ATT"` is the average of ``Y_{it} - \\hat Y_{it}(0)`` over treated
cells, where the untreated potential outcomes are imputed by the fitted low-rank matrix
plus unit and time effects. `vcov` is the placebo or bootstrap variance. It is
approximate, because the penalty ``\\lambda`` is held fixed across replications (see
[`matrix_completion`](@ref)), and it throws when `se_method = :none`.

# Fields
- `att::Float64`: average effect over treated cells.
- `se`, `se_method`, `replications`, `replicate_estimates`: standard error, its method,
  and the replicate estimates behind it.
- `lambda::Float64`: nuclear-norm penalty; `cv::Union{Nothing,DataFrame}`: validation
  RMSE by candidate ``\\lambda`` (`nothing` when ``\\lambda`` was supplied).
- `L::Matrix{Float64}`: fitted low-rank component; `unit_effects::Vector{Float64}`,
  `time_effects::Vector{Float64}`: fitted unpenalised effects.
- `Y0hat::Matrix{Float64}`: imputed untreated outcomes (the sum of the components).
- `rank::Int`: numerical rank of `L`.
- `panel::SynthPanel`: the data used.
"""
struct MatrixCompletionEstimate{P<:SynthPanel} <: CausalEstimate
    att::Float64
    se::Union{Nothing,Float64}
    se_method::Symbol
    replications::Int
    replicate_estimates::Vector{Float64}
    lambda::Float64
    cv::Union{Nothing,DataFrame}
    L::Matrix{Float64}
    unit_effects::Vector{Float64}
    time_effects::Vector{Float64}
    Y0hat::Matrix{Float64}
    rank::Int
    panel::P
end

StatsAPI.coef(r::MatrixCompletionEstimate) = [r.att]
StatsAPI.coefnames(::MatrixCompletionEstimate) = ["ATT"]
StatsAPI.nobs(r::MatrixCompletionEstimate) = length(r.panel.Y)
function StatsAPI.vcov(r::MatrixCompletionEstimate)
    r.se === nothing && throw(ArgumentError("no standard error was computed; re-run " *
                                            "with se_method = :placebo or :bootstrap"))
    return fill(r.se^2, 1, 1)
end
estimand(::MatrixCompletionEstimate) = "ATT (average over treated unit-periods)"
method_name(::MatrixCompletionEstimate) = "Matrix completion (MC-NNM)"

# ---------------------------------------------------------------------------------------
# MCPanel algorithm
# ---------------------------------------------------------------------------------------

function _sc_mc_objective(M, mask, L, u, v, sum_sing, lam)
    E = (L .+ u .+ v' .- M) .* mask
    return sum(abs2, E) / sum(mask) + lam * sum_sing
end

function _sc_mc_update_u(M, mask, L, v)
    u = zeros(size(M, 1))
    for i in axes(M, 1)
        l = 0
        s = 0.0
        for j in axes(M, 2)
            if mask[i, j] > 0
                l += 1
                s += L[i, j] + v[j] - M[i, j]
            end
        end
        u[i] = l > 0 ? -s / l : 0.0
    end
    return u
end

function _sc_mc_update_v(M, mask, L, u)
    v = zeros(size(M, 2))
    for j in axes(M, 2)
        l = 0
        s = 0.0
        for i in axes(M, 1)
            if mask[i, j] > 0
                l += 1
                s += L[i, j] + u[i] - M[i, j]
            end
        end
        v[j] = l > 0 ? -s / l : 0.0
    end
    return v
end

function _sc_mc_update_L(M, mask, L, u, v, lam)
    P = (M .- (L .+ u .+ v')) .* mask .+ L
    F = svd(P)
    s = max.(F.S .- lam * sum(mask) / 2, 0.0)
    return F.U * Diagonal(s) * F.Vt, s
end

function _sc_mc_initialize(M, mask, est_u, est_v; niter=1000, tol=1e-5)
    N, T = size(M)
    u, v, L = zeros(N), zeros(T), zeros(N, T)
    obj = _sc_mc_objective(M, mask, L, u, v, 0.0, 0.0)
    for _ in 1:niter
        u = est_u ? _sc_mc_update_u(M, mask, L, v) : zeros(N)
        v = est_v ? _sc_mc_update_v(M, mask, L, u) : zeros(T)
        new = _sc_mc_objective(M, mask, L, u, v, 0.0, 0.0)
        rel = (new - obj) / obj
        (rel < tol && rel >= 0) && break
        obj = new
    end
    P = (M .- (u .+ v')) .* mask
    lam_max = 2 * maximum(svdvals(P)) / sum(mask)
    return u, v, lam_max
end

function _sc_mc_fit(M, mask, L, u, v, est_u, est_v, lam; niter=1000, tol=1e-5)
    obj = _sc_mc_objective(M, mask, L, u, v, sum(svdvals(L)), lam)
    for _ in 1:niter
        u = est_u ? _sc_mc_update_u(M, mask, L, v) : zeros(size(M, 1))
        v = est_v ? _sc_mc_update_v(M, mask, L, u) : zeros(size(M, 2))
        L, s = _sc_mc_update_L(M, mask, L, u, v, lam)
        new = _sc_mc_objective(M, mask, L, u, v, sum(s), lam)
        rel = (obj - new) / obj
        new < 1e-8 && break
        (rel < tol && rel >= 0) && break
        obj = new
    end
    return L, u, v
end

# Warm-started path over decreasing lambdas (MCPanel NNM_with_uv_init).
function _sc_mc_path(M, mask, u0, v0, est_u, est_v, lambdas; niter=1000, tol=1e-5)
    L = zeros(size(M))
    u, v = u0, v0
    out = Vector{Tuple{Matrix{Float64},Vector{Float64},Vector{Float64}}}()
    for lam in lambdas
        L, u, v = _sc_mc_fit(M, mask, L, u, v, est_u, est_v, lam; niter=niter, tol=tol)
        push!(out, (L, u, v))
    end
    return out
end

function _sc_mc_grid(lam_max, n)
    n >= 2 || throw(ArgumentError("matrix_completion: n_lambda must be at least 2"))
    ex = range(log10(lam_max), log10(lam_max) - 3; length=n - 1)
    return vcat(10.0 .^ ex, 0.0)
end

# MCPanel `mcnnm_fit`: fit at a given lambda along the default path.
function _sc_mc_fit_lambda(M, mask, lam, est_u, est_v; niter=1000, tol=1e-5,
                           path_length=100)
    u0, v0, lam_max = _sc_mc_initialize(M, mask, est_u, est_v; niter=niter, tol=tol)
    lams = lam >= lam_max ? [lam] :
           vcat(filter(>=(lam), _sc_mc_grid(lam_max, path_length)), lam)
    return last(_sc_mc_path(M, mask, u0, v0, est_u, est_v, lams; niter=niter, tol=tol))
end

function _sc_mc_rmse(M, mask, L, u, v)
    E = (L .+ u .+ v' .- M) .* mask
    return sqrt(sum(abs2, E) / sum(mask))
end

# MCPanel `mcnnm_cv` with folds drawn from `rng`.
function _sc_mc_cv(M, mask, est_u, est_v, n_lambda, n_folds, cv_ratio, niter, tol, rng)
    folds = map(1:n_folds) do _
        fm = mask .* (rand(rng, size(M)...) .< cv_ratio)
        u, v, lm = _sc_mc_initialize(M .* fm, fm, est_u, est_v; niter=1000, tol=1e-5)
        (mask=fm, u=u, v=v, lam_max=lm)
    end
    lam_max = maximum(f.lam_max for f in folds)
    lambdas = _sc_mc_grid(lam_max, n_lambda)
    mse = zeros(length(lambdas), n_folds)
    for (k, f) in enumerate(folds)
        val = mask .* (1 .- f.mask)
        sum(val) > 0 || throw(ArgumentError("matrix_completion: empty validation fold; " *
                                            "lower cv_ratio"))
        path = _sc_mc_path(M .* f.mask, f.mask, f.u, f.v, est_u, est_v, lambdas;
                           niter=niter, tol=tol)
        for (i, (L, u, v)) in enumerate(path)
            mse[i, k] = _sc_mc_rmse(M, val, L, u, v)^2
        end
    end
    rmse = sqrt.(vec(mean(mse; dims=2)))
    best = argmin(rmse)
    return lambdas[best], lambdas, DataFrame(lambda=lambdas, rmse=rmse)
end

function _sc_mc_att(Y, W, L, u, v)
    Yhat = L .+ u .+ v'
    return mean((Y .- Yhat)[W .== 1]), Yhat
end

function _sc_mc_replicate(Y, adoption_rows, lam, est_u, est_v, niter, tol)
    N, T = size(Y)
    W = zeros(Int, N, T)
    for (i, a) in enumerate(adoption_rows)
        a > 0 && (W[i, a:T] .= 1)
    end
    mask = Float64.(1 .- W)
    L, u, v = _sc_mc_fit_lambda(Y, mask, lam, est_u, est_v; niter=niter, tol=tol)
    return first(_sc_mc_att(Y, W, L, u, v))
end

"""
    matrix_completion(data, outcome, treatment, unit, time; kwargs...)
    matrix_completion(panel::SynthPanel; lambda=nothing, n_lambda=30, n_folds=5,
                      cv_ratio=0.8, unit_effects=true, time_effects=true,
                      max_iter=1000, tol=1e-5, se_method=:placebo, replications=200,
                      rng=Random.default_rng()) -> MatrixCompletionEstimate

Matrix completion with nuclear-norm penalisation (MC-NNM) for causal panel data models
(Athey, Bayati, Doudchenko, Imbens & Khosravi 2021), following the authors' `MCPanel`
package.

Athey et al. (2021) view the estimation of treatment effects in panel data as a missing
data problem. The ``N \\times T`` matrix of untreated potential outcomes ``Y(0)`` is
observed only in untreated cells, and the treated cells must be imputed. The estimand is
the average effect over treated cells,
``\\tau = |\\mathcal T|^{-1}\\sum_{(i,t) \\in \\mathcal T} (Y_{it}(1) - Y_{it}(0))``.
The model is
``Y_{it}(0) = L_{it} + u_i + v_t + \\varepsilon_{it}``, with ``L`` approximately
low-rank, which covers interactive fixed-effects models (Bai 2009). Athey et al. (2021)
show that regressions on a unit's own past (unconfoundedness) and on other units in the
same period (synthetic control) can both be read as matrix completion methods that
exploit different patterns in the data. The key assumption is that
the pattern of missing (treated) cells is unrelated to the idiosyncratic errors given
the low-rank structure, together with enough untreated cells in every row and column to
recover ``L``. Treated cells must not help predict the untreated outcomes, which rules
out anticipation and spillovers.

The estimator solves
```math
\\min_{L, u, v}\\; \\frac{1}{|\\mathcal O|}\\sum_{(i,t) \\in \\mathcal O}
    (Y_{it} - L_{it} - u_i - v_t)^2 + \\lambda \\lVert L \\rVert_*,
```
over the untreated cells ``\\mathcal O``, with unpenalised unit and time effects.
The nuclear norm ``\\lVert L \\rVert_*`` (the sum of singular values) is the convex
relaxation of the rank. It is computed by coordinate descent with singular-value
soft-thresholding and warm starts over a decreasing grid of penalties, in the spirit of
the soft-impute algorithm of Mazumder, Hastie and Tibshirani (2010). Treated cells are
imputed by ``\\hat Y_{it}(0) = \\hat L_{it} + \\hat u_i + \\hat v_t``. The method works for
block and staggered adoption, and for general missingness patterns. ``\\lambda`` is
chosen by cross-validation over held-out untreated cells. Each of `n_folds` random
training masks keeps each untreated cell with probability `cv_ratio`. The grid has
`n_lambda` values, spanning three decades below the smallest ``\\lambda`` that gives
``L = 0``, plus 0, and the value with the smallest validation RMSE is selected.

Standard errors are approximate. The `:placebo` method re-assigns the treated units'
adoption patterns to randomly chosen never-treated units and re-estimates on
never-treated units only. The `:bootstrap` method resamples units with replacement. In
both, ``\\lambda`` is held at its full-sample value, so neither accounts for the
selection of ``\\lambda``. The placebo variance also assumes homoskedasticity across
units, so report the intervals as indicative. MC-NNM is attractive with many
units and periods and complex adoption patterns. With a single treated unit and few
donors, [`synthetic_control`](@ref), [`synthetic_did`](@ref) or
[`augmented_synthetic_control`](@ref) are more transparent, because their weights can be
inspected. Report ``\\lambda``, the rank of ``\\hat L`` and the pre-treatment fit of the
imputed outcomes ([`synth_gaps`](@ref)).

# Arguments
- `data`: long panel (see [`synth_panel`](@ref)), with columns `outcome`, `treatment`,
  `unit` and `time`; or
- `panel::SynthPanel`: a prepared panel.

# Keywords
- `lambda=nothing`: nuclear-norm penalty. By default it is cross-validated.
- `n_lambda::Integer=30`: number of candidate penalties.
- `n_folds::Integer=5`: number of random training masks for cross-validation.
- `cv_ratio::Real=0.8`: share of untreated cells kept for training in each fold.
- `unit_effects::Bool=true`, `time_effects::Bool=true`: include the unpenalised unit
  and time effects.
- `max_iter::Integer=1000`, `tol::Real=1e-5`: coordinate-descent iterations and
  relative objective tolerance.
- `se_method::Symbol=:placebo`: `:placebo` (needs more never-treated than treated
  units), `:bootstrap` or `:none`.
- `replications::Integer=200`: number of placebo or bootstrap replications.
- `rng::AbstractRNG=Random.default_rng()`: generator for the cross-validation masks and
  the replications.

# Returns
- [`MatrixCompletionEstimate`](@ref).

# Examples
```julia
using DrSnow, CSV, DataFrames, StableRNGs
prop99 = CSV.read(pkgdir(DrSnow, "test", "validation", "synth", "prop99.csv"), DataFrame)
r = matrix_completion(prop99, :PacksPerCapita, :treated, :State, :Year;
                      replications=50, rng=StableRNG(1))
coef(r), r.lambda, r.rank
last(synth_gaps(r), 3)
```

# References
- Athey, S., Bayati, M., Doudchenko, N., Imbens, G., & Khosravi, K. (2021). Matrix
  completion methods for causal panel data models. *Journal of the American
  Statistical Association*, 116(536), 1716–1730.
- Mazumder, R., Hastie, T., & Tibshirani, R. (2010). Spectral regularization algorithms
  for learning large incomplete matrices. *Journal of Machine Learning Research*, 11,
  2287–2322.
- Bai, J. (2009). Panel data models with interactive fixed effects. *Econometrica*,
  77(4), 1229–1279.
- Doudchenko, N., & Imbens, G. W. (2016). Balancing, regression,
  difference-in-differences and synthetic control methods: A synthesis. NBER Working
  Paper 22791.
"""
function matrix_completion(data, outcome::Symbol, treatment::Symbol, unit::Symbol,
                           time::Symbol; kwargs...)
    return matrix_completion(synth_panel(data, outcome, treatment, unit, time); kwargs...)
end

function matrix_completion(panel::SynthPanel; lambda=nothing, n_lambda::Integer=30,
                           n_folds::Integer=5, cv_ratio::Real=0.8,
                           unit_effects::Bool=true, time_effects::Bool=true,
                           max_iter::Integer=1000, tol::Real=1e-5,
                           se_method::Symbol=:placebo, replications::Integer=200,
                           rng::AbstractRNG=Random.default_rng())
    se_method in (:placebo, :bootstrap, :none) ||
        throw(ArgumentError("matrix_completion: se_method must be :placebo, " *
                            ":bootstrap or :none"))
    0 < cv_ratio < 1 || throw(ArgumentError("matrix_completion: cv_ratio must be in " *
                                            "(0, 1)"))
    lambda === nothing || lambda >= 0 ||
        throw(ArgumentError("matrix_completion: lambda must be non-negative"))
    Y = panel.Y
    W = _sc_treatment_matrix(panel)
    mask = Float64.(1 .- W)
    cv = nothing
    lam = if lambda === nothing
        n_folds >= 1 || throw(ArgumentError("matrix_completion: n_folds must be ≥ 1"))
        best, _, cv = _sc_mc_cv(Y, mask, unit_effects, time_effects, n_lambda, n_folds,
                                cv_ratio, max_iter, tol, rng)
        best
    else
        Float64(lambda)
    end
    L, u, v = _sc_mc_fit_lambda(Y, mask, lam, unit_effects, time_effects;
                                niter=max_iter, tol=tol)
    att, Yhat = _sc_mc_att(Y, W, L, u, v)
    rk = count(>(1e-6 * max(1.0, opnorm(Y))), svdvals(L))
    draws = Float64[]
    se = nothing
    reps = 0
    nc, N = panel.n_control, size(Y, 1)
    if se_method !== :none
        replications >= 2 || throw(ArgumentError("matrix_completion: replications " *
                                                 "must be at least 2"))
        seeds = task_seeds(rng, replications)
        draws = zeros(replications)
        if se_method === :placebo
            nt = N - nc
            nc > nt || throw(ArgumentError("matrix_completion: the placebo standard " *
                                           "error needs more never-treated than treated " *
                                           "units"))
            trt_adopt = panel.adoption[(nc + 1):N]
            Threads.@threads for b in 1:replications
                perm = randperm(Random.Xoshiro(seeds[b]), nc)
                rows = vcat(perm[1:(nc - nt)], perm[(nc - nt + 1):nc])
                ad = vcat(zeros(Int, nc - nt), trt_adopt)
                draws[b] = _sc_mc_replicate(Y[rows, :], ad, lam, unit_effects,
                                            time_effects, max_iter, tol)
            end
        else
            Threads.@threads for b in 1:replications
                r = Random.Xoshiro(seeds[b])
                while true
                    ind = sort(rand(r, 1:N, N))
                    (all(<=(nc), ind) || all(>(nc), ind)) && continue
                    draws[b] = _sc_mc_replicate(Y[ind, :], panel.adoption[ind], lam,
                                                unit_effects, time_effects, max_iter, tol)
                    break
                end
            end
        end
        se = _sc_replicate_se(draws)
        reps = replications
    end
    return MatrixCompletionEstimate(att, se, se_method, reps, draws, lam, cv, L, u, v,
                                    Yhat, rk, panel)
end

function synth_gaps(r::MatrixCompletionEstimate)
    p = r.panel
    rows = _sc_treated_rows(p)
    first_adopt = minimum(p.adoption[rows])
    T = size(p.Y, 2)
    tr = vec(mean(p.Y[rows, :]; dims=1))
    sy = vec(mean(r.Y0hat[rows, :]; dims=1))
    return DataFrame(time=p.times, treated=tr, synthetic=sy, gap=tr .- sy,
                     post=(1:T) .>= first_adopt)
end

function Base.show(io::IO, ::MIME"text/plain", r::MatrixCompletionEstimate)
    p = r.panel
    println(io, method_name(r), " — estimand: ", estimand(r))
    label = "$(r.se_method), $(r.replications) replications, λ fixed; approximate"
    _sc_show_se_line(io, r.att, r.se, label)
    @printf(io, "  λ = %.4g%s; rank(L) = %d\n", r.lambda,
            r.cv === nothing ? " (user-supplied)" : " (cross-validated)", r.rank)
    @printf(io, "  Units: %d never-treated, %d treated; periods: %d; treated cells: %d\n",
            p.n_control, size(p.Y, 1) - p.n_control, size(p.Y, 2),
            count(==(1), _sc_treatment_matrix(p)))
end

Base.show(io::IO, r::MatrixCompletionEstimate) =
    print(io, method_name(r), "(", @sprintf("%.4g", r.att),
          r.se === nothing ? ")" : @sprintf(" (se %.3g))", r.se))
