# Small dense linear-programming kernels shared across areas (moved from
# src/did/honest_solvers.jl; the Rambachan–Roth sensitivity analysis and the IV
# marginal-treatment-effect bounds use them). No external solver is needed.
#
# - `_lp_simplex`: max c'x s.t. A x = b, x ≥ 0 (two-phase revised simplex, Dantzig
#   pricing with a switch to Bland's rule under degeneracy). Returns a basic optimal
#   solution and the simplex multipliers (dual solution).
# - `_lp_free`: max/min c'x over free x s.t. A_ub x ≤ b_ub, A_eq x = b_eq, solved in
#   standard form (one slack per inequality; basis size = number of constraints).
# - `_lp_free_dual`: the same problem solved through its dual, whose basis size is the
#   number of free variables — much faster when there are many more inequality
#   constraints than variables (e.g. shape constraints imposed on a grid).

struct _LPResult
    status::Symbol              # :optimal, :infeasible, :unbounded, :failed
    x::Vector{Float64}
    objective::Float64
    duals::Vector{Float64}      # multipliers y with A'y ≥ c at a maximum
end

function _lp_simplex_core!(basis::Vector{Int}, c::Vector{Float64}, A::Matrix{Float64},
                            b::Vector{Float64}; maxiter::Int=10_000)
    m, n = size(A)
    isbasic = falses(n)
    isbasic[basis] .= true
    stall = 0
    bland = false
    cscale = max(norm(c, Inf), floatmin())
    for _ in 1:maxiter
        F = lu(A[:, basis]; check=false)
        issuccess(F) || return :failed
        xB = F \ b
        y = F' \ c[basis]
        tol_d = 1e-9 * max(cscale, norm(y, Inf) * norm(A, Inf))
        q = 0
        best = tol_d
        for j in 1:n
            isbasic[j] && continue
            dj = c[j] - dot(view(A, :, j), y)
            if dj > best
                q = j
                bland && break
                best = dj
            end
        end
        q == 0 && return :optimal
        u = F \ A[:, q]
        ptol = 1e-9 * max(norm(u, Inf), floatmin())
        r = 0
        θ = Inf
        for i in 1:m
            u[i] > ptol || continue
            t = max(xB[i], 0.0) / u[i]
            if t < θ * (1 - 1e-12) - 1e-300 ||
               (r > 0 && t <= θ * (1 + 1e-12) && basis[i] < basis[r])
                θ = t
                r = i
            end
        end
        r == 0 && return :unbounded
        stall = θ <= 1e-13 * max(1.0, norm(xB, Inf)) ? stall + 1 : 0
        stall > 25 && (bland = true)
        isbasic[basis[r]] = false
        basis[r] = q
        isbasic[q] = true
    end
    return :failed
end

"""
    _lp_simplex(c, A, b) -> _LPResult

Maximize `c'x` subject to `A x = b`, `x ≥ 0` with a two-phase revised simplex.
`duals` are the simplex multipliers (a solution of the dual `min b'y s.t. A'y ≥ c`).
"""
function _lp_simplex(c::AbstractVector, A::AbstractMatrix, b::AbstractVector)
    m, n = size(A)
    A0 = Matrix{Float64}(A)
    b0 = Vector{Float64}(b)
    sgn = ones(m)
    for i in 1:m
        if b0[i] < 0
            A0[i, :] .*= -1
            b0[i] = -b0[i]
            sgn[i] = -1.0
        end
    end
    # Phase 1: artificial variables n+1..n+m.
    A1 = hcat(A0, Matrix{Float64}(I, m, m))
    c1 = vcat(zeros(n), fill(-1.0, m))
    basis = collect((n + 1):(n + m))
    st = _lp_simplex_core!(basis, c1, A1, b0)
    st === :optimal || return _LPResult(:failed, Float64[], NaN, Float64[])
    xB = A1[:, basis] \ b0
    infeas = sum((xB[i] for i in 1:m if basis[i] > n); init=0.0)
    if infeas > 1e-9 * max(1.0, norm(b0, Inf))
        return _LPResult(:infeasible, Float64[], NaN, Float64[])
    end
    # Drive remaining (zero-valued) artificials out of the basis; drop redundant rows.
    keep = trues(m)
    for i in 1:m
        basis[i] > n || continue
        F = lu(A1[:, basis]; check=false)
        issuccess(F) || return _LPResult(:failed, Float64[], NaN, Float64[])
        ei = zeros(m)
        ei[i] = 1.0
        row = F' \ ei
        jbest, vbest = 0, 1e-9 * max(1.0, norm(row, Inf))
        for j in 1:n
            j in basis && continue
            v = abs(dot(row, view(A0, :, j)))
            if v > vbest
                jbest, vbest = j, v
            end
        end
        if jbest > 0
            basis[i] = jbest
        else
            keep[i] = false
        end
    end
    A2 = A0[keep, :]
    b2 = b0[keep]
    basis2 = basis[keep]
    any(>(n), basis2) && return _LPResult(:failed, Float64[], NaN, Float64[])
    c0 = Vector{Float64}(c)
    st = _lp_simplex_core!(basis2, c0, A2, b2)
    st === :unbounded && return _LPResult(:unbounded, Float64[], Inf, Float64[])
    st === :optimal || return _LPResult(:failed, Float64[], NaN, Float64[])
    F = lu(A2[:, basis2])
    x = zeros(n)
    x[basis2] = max.(F \ b2, 0.0)
    y2 = F' \ c0[basis2]
    y = zeros(m)
    y[keep] = y2
    return _LPResult(:optimal, x, dot(c0, x), y .* sgn)
end

"""
    _lp_free(c, Aub, bub, Aeq, beq; maximize) -> (status, objective, x)

Optimize `c'x` over free `x` subject to `Aub x ≤ bub` and `Aeq x = beq` (either block
may have zero rows).
"""
function _lp_free(c, Aub, bub, Aeq, beq; maximize::Bool)
    n = length(c)
    mu, me = size(Aub, 1), size(Aeq, 1)
    # x = x⁺ - x⁻, slacks s ≥ 0 for the inequalities.
    A = vcat(hcat(Aub, -Aub, Matrix{Float64}(I, mu, mu)),
             hcat(Aeq, -Aeq, zeros(me, mu)))
    b = vcat(bub, beq)
    cc = vcat(c, -c, zeros(mu)) .* (maximize ? 1.0 : -1.0)
    lp = _lp_simplex(cc, A, b)
    lp.status === :optimal || return (lp.status, NaN, Float64[])
    x = lp.x[1:n] .- lp.x[(n + 1):(2n)]
    return (:optimal, dot(c, x), x)
end

"""
    _lp_free_dual(c, Aub, bub, Aeq, beq; maximize) -> (status, objective, x)

Same problem as [`_lp_free`](@ref) (optimize `c'x` over free `x` subject to
`Aub x ≤ bub`, `Aeq x = beq`), solved through the dual

`min bub'y + beq'w  s.t.  Aub'y + Aeq'w = c,  y ≥ 0,  w free`

with the simplex method; `x` is recovered from the simplex multipliers of the dual.
The basis has as many rows as there are primal variables, so this is the method of
choice when the number of inequality constraints is large. Status `:infeasible`
(primal infeasible) or `:unbounded` (primal unbounded) is reported instead of a
solution when appropriate.
"""
function _lp_free_dual(c, Aub, bub, Aeq, beq; maximize::Bool)
    n = length(c)
    size(Aub, 2) == n && size(Aeq, 2) == n ||
        throw(DimensionMismatch("constraint matrices must have length(c) columns"))
    cs = maximize ? Vector{Float64}(c) : -Vector{Float64}(c)
    A = hcat(Matrix{Float64}(Aub'), Matrix{Float64}(Aeq'), -Matrix{Float64}(Aeq'))
    cc = -vcat(Vector{Float64}(bub), Vector{Float64}(beq), -Vector{Float64}(beq))
    lp = _lp_simplex(cc, A, cs)
    lp.status === :infeasible && return (:unbounded, maximize ? Inf : -Inf, Float64[])
    lp.status === :unbounded && return (:infeasible, NaN, Float64[])
    lp.status === :optimal || return (lp.status, NaN, Float64[])
    x = -lp.duals
    return (:optimal, dot(c, x), x)
end
