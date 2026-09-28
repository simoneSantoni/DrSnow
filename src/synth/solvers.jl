# Numerical building blocks shared by the synthetic-control estimators.
#
# - Frank–Wolfe weights with ridge penalty, exactly as in the R package synthdid
#   (Arkhangelsky et al. 2021, reference implementation), including sparsification and
#   the covariate-adjusted variant.
# - Simplex-constrained least squares solved exactly through a Lawson–Hanson NNLS
#   reformulation (no external QP solver).
# - A Nelder–Mead minimiser for the nested V-matrix search of classic synthetic control.

# ---------------------------------------------------------------------------------------
# synthdid Frank–Wolfe
# ---------------------------------------------------------------------------------------

_sc_sum_normalize(x) = sum(x) != 0 ? x ./ sum(x) : fill(1 / length(x), length(x))

function _sc_sparsify(v::AbstractVector)
    w = copy(v)
    w[w .<= maximum(w) / 4] .= 0
    return w ./ sum(w)
end

# One Frank–Wolfe step for min_x ||A x - b||² + eta ||x||² over the unit simplex, with
# exact line search (synthdid `fw.step`).
function _sc_fw_step(A::AbstractMatrix, x::AbstractVector, b::AbstractVector, eta::Real)
    Ax = A * x
    half_grad = A' * (Ax .- b) .+ eta .* x
    i = argmin(half_grad)
    d_x = -x
    d_x[i] = 1 - x[i]
    all(iszero, d_x) && return x
    d_err = A[:, i] .- Ax
    step = -dot(half_grad, d_x) / (sum(abs2, d_err) + eta * sum(abs2, d_x))
    return x .+ clamp(step, 0.0, 1.0) .* d_x
end

# synthdid `sc.weight.fw`: weights on the first T0 columns of Y reproducing its last
# column, with optional intercept (column centering) and ridge penalty zeta.
function _sc_weight_fw(Y::AbstractMatrix, zeta::Real; intercept::Bool=true,
                       init=nothing, min_decrease::Real=1e-3, max_iter::Integer=1000)
    N0, T0 = size(Y, 1), size(Y, 2) - 1
    lambda = init === nothing ? fill(1 / T0, T0) : collect(Float64, init)
    Yc = intercept ? Y .- mean(Y; dims=1) : Matrix{Float64}(Y)
    A = Yc[:, 1:T0]
    b = Yc[:, T0 + 1]
    eta = N0 * zeta^2
    vals = Float64[]
    # In-place version of `_sc_fw_step` (same arithmetic, no per-iteration allocation)
    Ax = A * lambda
    res = similar(b)
    hg = similar(lambda)
    t = 0
    while t < max_iter && (t < 2 || vals[t - 1] - vals[t] > min_decrease^2)
        t += 1
        res .= Ax .- b
        mul!(hg, transpose(A), res)
        hg .+= eta .* lambda
        i = argmin(hg)
        xi = lambda[i]
        if !(xi == 1 && count(!iszero, lambda) == 1)
            derr2 = 0.0
            @inbounds for r in eachindex(Ax)
                derr2 += (A[r, i] - Ax[r])^2
            end
            dx2 = sum(abs2, lambda) - xi^2 + (1 - xi)^2
            step = -(hg[i] - dot(hg, lambda)) / (derr2 + eta * dx2)
            s = clamp(step, 0.0, 1.0)
            lambda .*= (1 - s)
            lambda[i] += s
            mul!(Ax, A, lambda)
        end
        err2 = 0.0
        @inbounds for r in eachindex(Ax)
            err2 += (Ax[r] - b[r])^2
        end
        push!(vals, zeta^2 * sum(abs2, lambda) + err2 / N0)
    end
    return lambda, vals
end

# synthdid `collapsed.form`: controls' pre-periods plus their post-period average,
# and the treated units' average pre-period path and post average.
function _sc_collapsed_form(Y::AbstractMatrix, N0::Integer, T0::Integer)
    N, T = size(Y)
    top = hcat(Y[1:N0, 1:T0], mean(Y[1:N0, (T0 + 1):T]; dims=2))
    bottom = hcat(mean(Y[(N0 + 1):N, 1:T0]; dims=1),
                  mean(Y[(N0 + 1):N, (T0 + 1):T]))
    return vcat(top, bottom)
end

# Σ_k X[:, :, k] β_k
function _sc_contract3(X::AbstractArray{<:Real,3}, beta::AbstractVector)
    out = zeros(size(X, 1), size(X, 2))
    for k in eachindex(beta)
        out .+= beta[k] .* view(X, :, :, k)
    end
    return out
end

# synthdid `sc.weight.fw.covariates`, operating on the collapsed form (Yc, Xc).
function _sc_weight_fw_covariates(Yc::AbstractMatrix, Xc::AbstractArray{<:Real,3};
                                  zeta_lambda::Real=0.0, zeta_omega::Real=0.0,
                                  lambda_intercept::Bool=true,
                                  omega_intercept::Bool=true, min_decrease::Real=1e-3,
                                  max_iter::Integer=1000, lambda=nothing, omega=nothing,
                                  update_lambda::Bool=true, update_omega::Bool=true)
    T0 = size(Yc, 2) - 1
    N0 = size(Yc, 1) - 1
    K = size(Xc, 3)
    lambda = lambda === nothing ? fill(1 / T0, T0) : collect(Float64, lambda)
    omega = omega === nothing ? fill(1 / N0, N0) : collect(Float64, omega)
    beta = zeros(K)
    # Centring is linear, so the centred (Y - Xβ) blocks are Y0 - Σ β_k X_k with the
    # centred pieces precomputed. Residuals are mean-zero, so the β-gradient computed
    # with centred covariates equals synthdid's (which uses uncentred ones).
    cen(M, c) = c ? M .- mean(M; dims=1) : Matrix{Float64}(M)
    Yl0 = cen(Yc[1:N0, :], lambda_intercept)
    Yo0 = cen(transpose(Yc[:, 1:T0]), omega_intercept)
    Xl = [cen(Xc[1:N0, :, k], lambda_intercept) for k in 1:K]
    Xo = [cen(transpose(Xc[:, 1:T0, k]), omega_intercept) for k in 1:K]
    Yl = copy(Yl0)
    Yo = copy(Yo0)
    eta_l = N0 * zeta_lambda^2
    eta_o = T0 * zeta_omega^2

    function update_weights(lambda, omega)
        if update_lambda
            lambda = _sc_fw_step(view(Yl, :, 1:T0), lambda, view(Yl, :, T0 + 1), eta_l)
        end
        err_lambda = Yl * vcat(lambda, -1.0)
        if update_omega
            omega = _sc_fw_step(view(Yo, :, 1:N0), omega, view(Yo, :, N0 + 1), eta_o)
        end
        err_omega = Yo * vcat(omega, -1.0)
        val = zeta_omega^2 * sum(abs2, omega) + zeta_lambda^2 * sum(abs2, lambda) +
              sum(abs2, err_omega) / T0 + sum(abs2, err_lambda) / N0
        return (; val, lambda, omega, err_lambda, err_omega)
    end

    vals = Float64[]
    t = 0
    w = update_weights(lambda, omega)
    grad = zeros(K)
    while t < max_iter && (t < 2 || abs(vals[t - 1] - vals[t]) > min_decrease^2)
        t += 1
        lam1 = vcat(w.lambda, -1.0)
        om1 = vcat(w.omega, -1.0)
        for k in 1:K
            grad[k] = -(dot(w.err_lambda, Xl[k] * lam1) / N0 +
                        dot(w.err_omega, Xo[k] * om1) / T0)
        end
        beta .-= (1 / t) .* grad
        copyto!(Yl, Yl0)
        copyto!(Yo, Yo0)
        for k in 1:K
            Yl .-= beta[k] .* Xl[k]
            Yo .-= beta[k] .* Xo[k]
        end
        w = update_weights(w.lambda, w.omega)
        push!(vals, w.val)
    end
    return (lambda=w.lambda, omega=w.omega, beta=beta, vals=vals)
end

# ---------------------------------------------------------------------------------------
# Non-negative least squares (Lawson & Hanson 1974, ch. 23) and simplex least squares
# ---------------------------------------------------------------------------------------

function _sc_nnls(A::AbstractMatrix, b::AbstractVector; max_iter::Integer=0,
                  init_passive=nothing)
    m, n = size(A)
    max_iter = max_iter > 0 ? max_iter : 30 * n + 100
    x = zeros(n)
    passive = falses(n)
    blocked = falses(n)
    tol = 10 * eps(Float64) * opnorm(A, 1) * max(m, n)
    if init_passive !== nothing
        # Warm start: least squares on a guessed support, shrunk until it is feasible.
        # This is a valid Lawson–Hanson state (x_P > 0 solves the problem restricted to
        # P), so the main loop below still certifies optimality.
        P = findall(init_passive)
        while !isempty(P)
            z = A[:, P] \ b
            if all(>(tol), z)
                x[P] = z
                passive[P] .= true
                break
            end
            P = P[z .> tol]
        end
    end
    iter = 0
    while true
        w = A' * (b .- A * x)
        cand = [j for j in 1:n if !passive[j] && !blocked[j] && w[j] > tol]
        isempty(cand) && break
        iter += 1
        iter > max_iter && error("_sc_nnls: no convergence after $max_iter iterations")
        j = cand[argmax(w[cand])]
        passive[j] = true
        first = true
        while true
            P = findall(passive)
            z = zeros(n)
            z[P] = A[:, P] \ b
            if first && z[j] <= tol
                # numerically the entering variable cannot move: skip it this round
                passive[j] = false
                blocked[j] = true
                break
            end
            first = false
            if all(k -> z[k] > tol, P)
                x = z
                fill!(blocked, false)
                break
            end
            Q = [k for k in P if z[k] <= tol]
            alpha = minimum(x[k] / (x[k] - z[k]) for k in Q)
            x = x .+ alpha .* (z .- x)
            for k in P
                if x[k] <= tol
                    passive[k] = false
                    x[k] = 0.0
                end
            end
            iter += 1
            iter > max_iter && error("_sc_nnls: no convergence after $max_iter iterations")
            any(passive) || break
        end
    end
    return x
end

"""
    _sc_simplex_ls(A, b) -> w

Exact solution of `min ||A w - b||²` subject to `w ≥ 0`, `sum(w) = 1`.

Writing `C = A - b 1'`, the problem is `min ||C w||²` over the simplex. For `κ > 0` the
NNLS problem `min_{v≥0} ||[C; κ1'] v - [0; κ]||²` has solution `v = s w*` with
`s = κ²/(κ² + d²) > 0`, where `w*` is the simplex solution and `d² = ||C w*||²`, so
`w* = v / sum(v)`.
"""
function _sc_simplex_ls(A::AbstractMatrix, b::AbstractVector; support=nothing)
    n = size(A, 2)
    n == 1 && return [1.0]
    C = A .- b
    kappa = sqrt(maximum(sum(abs2, C; dims=1)))
    kappa = kappa > 0 ? kappa : 1.0
    E = vcat(C, fill(kappa, 1, n))
    f = vcat(zeros(size(A, 1)), kappa)
    v = _sc_nnls(E, f; init_passive=support)
    s = sum(v)
    s > 0 || error("_sc_simplex_ls: degenerate solution")
    w = v ./ s
    return w
end

# ---------------------------------------------------------------------------------------
# Nelder–Mead (adaptive parameters of Gao & Han 2012)
# ---------------------------------------------------------------------------------------

function _sc_nelder_mead(f, x0::AbstractVector; max_iter::Integer=5000,
                         ftol::Real=1e-10, xtol::Real=1e-10, initial_step::Real=0.1)
    n = length(x0)
    alpha, gamma = 1.0, 1.0 + 2 / n
    rho, sigma = 0.75 - 1 / (2n), 1.0 - 1 / n
    step = initial_step * max(maximum(abs, x0), 1e-8 + 0.0)
    step = step > 0 ? step : initial_step
    simplex = [collect(Float64, x0)]
    for i in 1:n
        x = collect(Float64, x0)
        x[i] += step
        push!(simplex, x)
    end
    fvals = [f(x) for x in simplex]
    iter = 0
    while iter < max_iter
        iter += 1
        ord = sortperm(fvals)
        simplex = simplex[ord]
        fvals = fvals[ord]
        if abs(fvals[end] - fvals[1]) <= ftol * (abs(fvals[1]) + ftol)
            break
        end
        scale = maximum(abs, simplex[1]) + xtol
        spread = maximum(maximum(abs, simplex[i] .- simplex[1]) for i in 2:(n + 1))
        if spread <= xtol * scale
            break
        end
        centroid = sum(simplex[1:n]) ./ n
        xr = centroid .+ alpha .* (centroid .- simplex[end])
        fr = f(xr)
        if fr < fvals[1]
            xe = centroid .+ gamma .* (xr .- centroid)
            fe = f(xe)
            if fe < fr
                simplex[end], fvals[end] = xe, fe
            else
                simplex[end], fvals[end] = xr, fr
            end
        elseif fr < fvals[n]
            simplex[end], fvals[end] = xr, fr
        else
            if fr < fvals[end]
                xc = centroid .+ rho .* (xr .- centroid)
                fc = f(xc)
                accept = fc <= fr
            else
                xc = centroid .+ rho .* (simplex[end] .- centroid)
                fc = f(xc)
                accept = fc < fvals[end]
            end
            if accept
                simplex[end], fvals[end] = xc, fc
            else
                for i in 2:(n + 1)
                    simplex[i] = simplex[1] .+ sigma .* (simplex[i] .- simplex[1])
                    fvals[i] = f(simplex[i])
                end
            end
        end
    end
    i = argmin(fvals)
    return simplex[i], fvals[i]
end
