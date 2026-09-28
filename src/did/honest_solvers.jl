# Small dense optimisation kernels used by the Rambachan–Roth sensitivity analysis
# (honest_did.jl). The problems are tiny (tens of variables), so plain dense
# revised-simplex and active-set/homotopy methods are fast and deterministic, and
# no external solver is needed.
#
# - `_did_simplex`, `_did_lp_free`: aliases of the shared simplex LP solver
#   `_lp_simplex` / `_lp_free` in src/core/lp.jl (moved there unchanged).
# - `_did_lasso_path`: exact solution path of min ½ z'Qz + r'z + μ‖z‖₁ (Q ≻ 0),
#   used to trace the bias/variance frontier of fixed-length confidence intervals.
# - `_did_truncnorm_quantile`, `_did_folded_normal_quantile`.

# The simplex LP kernels live in src/core/lp.jl (shared with the IV area); these
# aliases keep the historical names.
const _did_simplex = _lp_simplex
const _did_lp_free = _lp_free

# ---------------------------------------------------------------------------
# Lasso homotopy (exact piecewise-linear solution path)
# ---------------------------------------------------------------------------

"""
    _did_lasso_path(Q, r) -> (μs, Z)

Breakpoints `μ_0 > μ_1 > … > μ_m = 0` and solutions `Z[:, k]` of
`min_z ½ z'Qz + r'z + μ‖z‖₁` (`Q` positive definite). Between breakpoints the
solution is linear in `μ`; for `μ ≥ μ_0` it is zero.
"""
function _did_lasso_path(Q::AbstractMatrix, r::AbstractVector)
    p = length(r)
    μ = p == 0 ? 0.0 : maximum(abs, r)
    z = zeros(p)
    μs = [μ]
    Z = [copy(z)]
    μ <= 0 && return (μs, reduce(hcat, Z; init=zeros(p, 0)))
    j0 = argmax(abs.(r))
    active = [j0]
    signs = [-sign(r[j0])]
    for _ in 1:(50 * p + 50)
        QAA = Symmetric(Matrix(Q[active, active]))
        a = -(QAA \ r[active])
        b = -(QAA \ signs)
        inactive = setdiff(1:p, active)
        α = Q[inactive, active] * a .+ r[inactive]
        β = Q[inactive, active] * b
        best = 0.0
        event = (:none, 0, 0.0)
        lim = μ * (1 - 1e-10)
        for (k, j) in enumerate(inactive), sg in (1.0, -1.0)
            den = sg - β[k]
            abs(den) > 1e-14 || continue
            m = α[k] / den
            if m < lim && m > best
                best = m
                event = (:enter, j, -sg)
            end
        end
        for (k, j) in enumerate(active)
            abs(b[k]) > 1e-14 || continue
            m = -a[k] / b[k]
            if m < lim && m > best
                best = m
                event = (:leave, k, 0.0)
            end
        end
        μ = best
        z = zeros(p)
        z[active] = a .+ μ .* b
        push!(μs, μ)
        push!(Z, copy(z))
        μ <= 0 && break
        if event[1] === :enter
            push!(active, event[2])
            push!(signs, event[3])
        else
            z[active[event[2]]] = 0.0
            Z[end] = copy(z)
            deleteat!(active, event[2])
            deleteat!(signs, event[2])
            if isempty(active)   # restart from zero (cannot normally happen)
                j0 = argmax(abs.(r))
                active = [j0]
                signs = [-sign(r[j0])]
            end
        end
    end
    μs[end] > 0 && error("Lasso homotopy did not converge")
    return (μs, reduce(hcat, Z))
end

# ---------------------------------------------------------------------------
# Normal-distribution helpers
# ---------------------------------------------------------------------------

"""
    _did_truncnorm_quantile(p, l, u)

`p`-quantile of a standard normal truncated to `[l, u]` (as TruncatedNormal::norminvp),
computed in the tail on the log scale so that it is accurate far from zero.
"""
function _did_truncnorm_quantile(p::Real, l::Real, u::Real)
    l <= u || throw(ArgumentError("truncation interval is empty"))
    l == u && return float(l)
    N = Normal()
    if l >= 0
        la, lb = logccdf(N, l), logccdf(N, u)
        # log P(Z > q) = log(P(Z>l) - p (P(Z>l) - P(Z>u)))
        lq = la + log1p(-p * (1 - exp(lb - la)))
        return clamp(invlogccdf(N, lq), float(l), float(u))
    elseif u <= 0
        return -_did_truncnorm_quantile(1 - p, -u, -l)
    else
        a, b = cdf(N, l), cdf(N, u)
        return clamp(quantile(N, a + p * (b - a)), float(l), float(u))
    end
end

"""
    _did_folded_normal_quantile(p, t)

`p`-quantile of `|Z + t|`, `Z ~ N(0, 1)`: the critical value `cv_α(t)` of a
fixed-length confidence interval with worst-case bias `t` standard errors.
"""
function _did_folded_normal_quantile(p::Real, t::Real)
    t = abs(float(t))
    N = Normal()
    f(c) = cdf(N, c - t) - cdf(N, -c - t) - p
    lo = max(0.0, t + quantile(N, p) - 1e-8)
    hi = t + quantile(N, (1 + p) / 2) + 1e-8
    while f(lo) > 0
        lo /= 2
        lo < 1e-300 && return 0.0
    end
    while f(hi) < 0
        hi *= 2
    end
    for _ in 1:200
        mid = (lo + hi) / 2
        f(mid) < 0 ? (lo = mid) : (hi = mid)
        hi - lo <= 1e-14 * max(1.0, hi) && break
    end
    return (lo + hi) / 2
end
