# Pure-Julia elastic-net solvers (Gaussian and binomial) by cyclic coordinate descent,
# following Friedman, Hastie & Tibshirani (2010, J. Stat. Softw.). Used by the
# built-in `LassoLearner`, `RidgeLearner` (via closed form) and
# `PenalizedLogisticLearner`.
#
# Parameterization (identical to glmnet with `standardize = TRUE`): observation
# weights are normalized to sum to one, predictors are standardized with weighted
# moments (1/Σw convention), and the objective is
#   Gaussian:  ½ Σ wᵢ (yᵢ - β₀ - xᵢ'β)² + λ [α‖β‖₁ + (1-α)/2 ‖β‖²]
#   binomial: -Σ wᵢ ℓᵢ(β₀, β)          + λ [α‖β‖₁ + (1-α)/2 ‖β‖²]
# on the standardized scale; coefficients are returned on the original scale.

_ml_soft(z, g) = z > g ? z - g : (z < -g ? z + g : 0.0)

"""Normalized observation weights (sum 1); `nothing` means equal weights."""
function _ml_normweights(weights, n::Integer)
    if weights === nothing
        return fill(1.0 / n, n)
    end
    length(weights) == n ||
        throw(DimensionMismatch("weights must have length $n, got $(length(weights))"))
    w = Float64.(weights)
    all(x -> isfinite(x) && x >= 0, w) ||
        throw(ArgumentError("weights must be finite and non-negative"))
    s = sum(w)
    s > 0 || throw(ArgumentError("weights must not all be zero"))
    return w ./ s
end

"""
Weighted standardization. Returns `(μ, σ, ok)`; `ok[j]` is false for (numerically)
constant columns, which are excluded from penalized fits (coefficient 0).
"""
function _ml_moments(X::AbstractMatrix, w::AbstractVector)
    p = size(X, 2)
    μ = zeros(p)
    σ = ones(p)
    ok = trues(p)
    @inbounds for j in 1:p
        m = 0.0
        for i in axes(X, 1)
            m += w[i] * X[i, j]
        end
        v = 0.0
        for i in axes(X, 1)
            v += w[i] * (X[i, j] - m)^2
        end
        μ[j] = m
        s = sqrt(max(v, 0.0))
        if s <= 1e-10 * max(1.0, abs(m))
            ok[j] = false
        else
            σ[j] = s
        end
    end
    return μ, σ, ok
end

function _ml_standardized(X::AbstractMatrix, μ, σ, ok)
    Z = (X .- μ') ./ σ'
    for j in axes(Z, 2)
        ok[j] || (Z[:, j] .= 0.0)
    end
    return Z
end

"""Largest λ at which all penalized coefficients are zero."""
function _ml_lambda_max(Z, r, w, α)
    g = 0.0
    for j in axes(Z, 2)
        g = max(g, abs(dot(view(Z, :, j), w .* r)))
    end
    return g / max(α, 1e-3)
end

function _ml_lambda_path(lmax, nlambda, ratio)
    lmax <= 0 && return [0.0]
    nlambda == 1 && return [lmax]
    return exp.(range(log(lmax), log(lmax * ratio); length=nlambda))
end

# One coordinate-descent solve at a fixed λ for weighted least squares with an
# (optional) unpenalized intercept. `r` holds the current residual z - β₀ - Zβ and is
# updated in place. `v[j] = Σ wᵢ Zᵢⱼ²`.
function _ml_cd_wls!(β, b0::Base.RefValue{Float64}, r, Z, w, v, λ, α, ok;
                     intercept::Bool, tol::Float64, maxit::Int)
    p = length(β)
    sw = sum(w)
    l1 = λ * α
    l2 = λ * (1 - α)
    active = falses(p)
    iter = 0
    while iter < maxit
        # full sweep over all coordinates
        iter += 1
        dmax = _ml_cd_sweep!(β, b0, r, Z, w, v, l1, l2, ok, trues(p), intercept, sw)
        for j in 1:p
            active[j] = β[j] != 0.0
        end
        dmax < tol && break
        # iterate on the active set until convergence
        while iter < maxit
            iter += 1
            d = _ml_cd_sweep!(β, b0, r, Z, w, v, l1, l2, ok, active, intercept, sw)
            d < tol && break
        end
    end
    return iter
end

function _ml_cd_sweep!(β, b0, r, Z, w, v, l1, l2, ok, set, intercept, sw)
    dmax = 0.0
    n = length(r)
    @inbounds for j in eachindex(β)
        (ok[j] && set[j]) || continue
        bj = β[j]
        g = 0.0
        for i in 1:n
            g += w[i] * Z[i, j] * r[i]
        end
        z = g + v[j] * bj
        nb = _ml_soft(z, l1) / (v[j] + l2)
        if nb != bj
            d = nb - bj
            for i in 1:n
                r[i] -= d * Z[i, j]
            end
            β[j] = nb
            dmax = max(dmax, v[j] * d^2)
        end
    end
    if intercept
        d = 0.0
        @inbounds for i in 1:n
            d += w[i] * r[i]
        end
        d /= sw
        if d != 0.0
            r .-= d
            b0[] += d
            dmax = max(dmax, sw * d^2)
        end
    end
    return dmax
end

"""
Gaussian elastic-net path on standardized `Z` (weighted-centred) and centred `yc`.
Returns a p × L coefficient matrix (standardized scale).
"""
function _ml_enet_gaussian_path(Z, yc, w, lambdas, α, ok; tol=1e-10, maxit=100_000)
    p = size(Z, 2)
    B = zeros(p, length(lambdas))
    β = zeros(p)
    r = copy(yc)
    v = [dot(w, view(Z, :, j) .^ 2) for j in 1:p]
    scale = max(dot(w, yc .^ 2), eps())
    b0 = Ref(0.0)
    for (l, λ) in enumerate(lambdas)
        _ml_cd_wls!(β, b0, r, Z, w, v, λ, α, ok; intercept=false, tol=tol * scale,
                    maxit=maxit)
        B[:, l] .= β
    end
    return B
end

_ml_sigmoid(x) = x >= 0 ? 1 / (1 + exp(-x)) : exp(x) / (1 + exp(x))

"""
Binomial elastic-net path by IRLS with an inner coordinate-descent solver.
Returns `(b0s, B)` on the standardized scale.
"""
function _ml_enet_logistic_path(Z, y, w, lambdas, α, ok; tol=1e-8, maxit=100_000,
                                maxouter=100)
    n, p = size(Z)
    B = zeros(p, length(lambdas))
    b0s = zeros(length(lambdas))
    β = zeros(p)
    ybar = clamp(dot(w, y), 1e-6, 1 - 1e-6)
    b0 = Ref(log(ybar / (1 - ybar)))
    η = fill(b0[], n)
    ww = similar(y, Float64)
    r = similar(ww)
    v = zeros(p)
    for (l, λ) in enumerate(lambdas)
        devold = Inf
        for _ in 1:maxouter
            dev = 0.0
            @inbounds for i in 1:n
                pr = clamp(_ml_sigmoid(η[i]), 1e-5, 1 - 1e-5)
                ww[i] = w[i] * pr * (1 - pr)
                r[i] = (y[i] - pr) / (pr * (1 - pr))   # z - η
                dev -= 2 * w[i] * (y[i] * log(pr) + (1 - y[i]) * log(1 - pr))
            end
            for j in 1:p
                v[j] = ok[j] ? dot(ww, view(Z, :, j) .^ 2) : 1.0
            end
            _ml_cd_wls!(β, b0, r, Z, ww, v, λ, α, ok; intercept=true, tol=tol * 0.25,
                        maxit=maxit)
            mul!(η, Z, β)
            η .+= b0[]
            abs(dev - devold) <= tol * (abs(dev) + 0.1) && break
            devold = dev
        end
        B[:, l] .= β
        b0s[l] = b0[]
    end
    return b0s, B
end

# Back-transform standardized coefficients to the original scale.
function _ml_unstandardize(b0_std, βs, μ, σ, ok)
    β = zeros(length(βs))
    for j in eachindex(βs)
        ok[j] && (β[j] = βs[j] / σ[j])
    end
    return b0_std - dot(μ, β), β
end

"""
    _ml_enet_fit(X, y, w; family, alpha, lambda, nlambda, lambda_min_ratio, nfolds,
                 rule, rng) -> (b0, β, λ)

Fit an elastic net with λ either fixed (`lambda::Real`) or chosen by K-fold
cross-validation (`lambda = :cv`, rule `:min` or `:one_se`). `w` must sum to one.
"""
function _ml_enet_fit(X::AbstractMatrix, y::AbstractVector, w::AbstractVector;
                      family::Symbol, alpha::Real, lambda, nlambda::Int,
                      lambda_min_ratio, nfolds::Int, rule::Symbol, rng::AbstractRNG)
    n, p = size(X)
    μ, σ, ok = _ml_moments(X, w)
    Z = _ml_standardized(X, μ, σ, ok)
    ybar = dot(w, y)
    ratio = lambda_min_ratio === nothing ? (n > p ? 1e-4 : 1e-2) : Float64(lambda_min_ratio)
    lmax = _ml_lambda_max(Z, y .- ybar, w, alpha)
    if lambda isa Real
        lambda >= 0 || throw(ArgumentError("lambda must be non-negative"))
        path = filter(>(lambda), _ml_lambda_path(lmax, nlambda, ratio))
        push!(path, Float64(lambda))
        b0, β = _ml_enet_path_fit(Z, y, w, path, alpha, ok, family, ybar)
        b, bb = _ml_unstandardize(b0[end], β[:, end], μ, σ, ok)
        return b, bb, Float64(lambda)
    end
    lambda === :cv || throw(ArgumentError("lambda must be a non-negative number or :cv"))
    rule in (:min, :one_se) || throw(ArgumentError("rule must be :min or :one_se"))
    path = _ml_lambda_path(lmax, nlambda, ratio)
    if p == 0 || !any(ok) || lmax <= 0
        b0, β = _ml_enet_path_fit(Z, y, w, [0.0], alpha, ok, family, ybar)
        b, bb = _ml_unstandardize(b0[end], β[:, end], μ, σ, ok)
        return b, bb, 0.0
    end
    K = min(nfolds, n)
    K >= 2 || throw(ArgumentError("need at least 2 observations for cross-validation"))
    fold = _ml_simple_folds(rng, n, K, family === :binomial ? y : nothing)
    L = length(path)
    loss = zeros(K, L)
    wk = zeros(K)
    for k in 1:K
        tr = fold .!= k
        te = .!tr
        wtr = w[tr] ./ sum(w[tr])
        Xtr = X[tr, :]
        μk, σk, okk = _ml_moments(Xtr, wtr)
        Zk = _ml_standardized(Xtr, μk, σk, okk)
        ytr = y[tr]
        b0k, Bk = _ml_enet_path_fit(Zk, ytr, wtr, path, alpha, okk, family, dot(wtr, ytr))
        wte = w[te]
        wk[k] = sum(wte)
        Xte = X[te, :]
        for l in 1:L
            b, bb = _ml_unstandardize(b0k[l], Bk[:, l], μk, σk, okk)
            η = Xte * bb .+ b
            loss[k, l] = _ml_loss(family, y[te], η, wte)
        end
    end
    wk ./= sum(wk)
    cvm = vec(sum(loss .* wk; dims=1))
    cvsd = vec(sqrt.(sum(((loss .- cvm') .^ 2) .* wk; dims=1) ./ max(K - 1, 1)))
    imin = argmin(cvm)
    idx = if rule === :min
        imin
    else
        thr = cvm[imin] + cvsd[imin]
        findfirst(l -> cvm[l] <= thr, 1:L)
    end
    b0, B = _ml_enet_path_fit(Z, y, w, path[1:idx], alpha, ok, family, ybar)
    b, bb = _ml_unstandardize(b0[end], B[:, end], μ, σ, ok)
    return b, bb, path[idx]
end

function _ml_enet_path_fit(Z, y, w, path, alpha, ok, family, ybar)
    if family === :gaussian
        B = _ml_enet_gaussian_path(Z, y .- ybar, w, path, alpha, ok)
        return fill(ybar, length(path)), B
    else
        return _ml_enet_logistic_path(Z, y, w, path, alpha, ok)
    end
end

function _ml_loss(family, y, η, w)
    sw = sum(w)
    if family === :gaussian
        return sum(w .* (y .- η) .^ 2) / sw
    else
        s = 0.0
        for i in eachindex(y)
            pr = clamp(_ml_sigmoid(η[i]), 1e-12, 1 - 1e-12)
            s -= w[i] * (y[i] * log(pr) + (1 - y[i]) * log(1 - pr))
        end
        return 2 * s / sw
    end
end

"""Random K-fold assignment (stratified by a binary `strata` vector when given)."""
function _ml_simple_folds(rng::AbstractRNG, n::Integer, K::Integer, strata=nothing)
    fold = zeros(Int, n)
    groups = strata === nothing ? [collect(1:n)] :
             [findall(==(s), strata) for s in sort(unique(strata))]
    offset = 0
    for g in groups
        perm = g[randperm(rng, length(g))]
        for (i, idx) in enumerate(perm)
            fold[idx] = mod1(i + offset, K)
        end
        offset += length(g)
    end
    return fold
end
