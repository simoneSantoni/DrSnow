# Weak-instrument-robust inference: Anderson–Rubin (homoskedastic / robust /
# cluster), Moreira's CLR and Kleibergen's K (homoskedastic), their
# heteroskedasticity- / cluster-robust versions (Kleibergen 2005, as in Stata's
# `weakiv`), and confidence sets.

"""
    WeakIVConfidenceSet

Confidence set for the coefficient of a single endogenous regressor obtained by
inverting a test whose size does not depend on the strength of the instruments.

A confidence set with correct coverage whatever the strength of identification must be
unbounded with positive probability (Dufour 1997), so sets obtained by test inversion
need not be intervals: depending on the data they can be a bounded interval, the
complement of an interval (a union of two rays), the whole real line, a finite union of
intervals, or, in overidentified models, empty. An unbounded set signals that the data
are uninformative about the coefficient; an empty Anderson–Rubin set signals that the
overidentifying restrictions are rejected for every value of the coefficient. The set
is the collection ``\\{\\beta_0 : p(\\beta_0) \\ge 1 - \\text{level}\\}`` of the inverted
test's p-value function, which is stored so that p-values can be computed for any
hypothesized value.

# Fields
- `method::String`: the test that was inverted, with its covariance type.
- `level::Float64`: the confidence level.
- `kind::Symbol`: `:bounded` (one finite interval), `:union_of_rays`
  (``(-\\infty, a] \\cup [b, \\infty)``), `:real_line`, `:empty`, `:ray` (one half-line)
  or `:union_of_intervals` (any other finite union).
- `intervals::Vector{Tuple{Float64,Float64}}`: disjoint closed intervals, with
  endpoints possibly `±Inf`, whose union is the set.
- `critical_value::Float64`: critical value of the inverted statistic.
- `estimate::Float64`: the point estimate, for reference (the 2SLS estimate; the DML
  estimate for [`dml_weak_iv_confidence_set`](@ref)).
- `pvalue_function`: the function ``\\beta_0 \\mapsto p(\\beta_0)`` of the inverted test,
  also available as `DrSnow.pvalue(set, β₀)` (a method of the StatsAPI `pvalue`
  generic).

Membership is tested with `β₀ in set`.

# References
- Dufour, J.-M. (1997). Some impossibility theorems in econometrics with applications
  to structural and dynamic models. *Econometrica*, 65(6), 1365–1388.
- Mikusheva, A. (2010). Robust confidence sets in the presence of weak instruments.
  *Journal of Econometrics*, 157(2), 236–247.
"""
struct WeakIVConfidenceSet
    method::String
    level::Float64
    kind::Symbol
    intervals::Vector{Tuple{Float64,Float64}}
    critical_value::Float64
    estimate::Float64
    pvalue_function::Function
end

Base.in(b::Real, s::WeakIVConfidenceSet) = any(iv -> iv[1] <= b <= iv[2], s.intervals)
StatsAPI.pvalue(s::WeakIVConfidenceSet, b0::Real) = s.pvalue_function(float(b0))

function _iv_set_kind(iv::Vector{Tuple{Float64,Float64}})
    isempty(iv) && return :empty
    if length(iv) == 1
        a, b = iv[1]
        isinf(a) && isinf(b) && return :real_line
        (isinf(a) || isinf(b)) && return :ray
        return :bounded
    end
    if length(iv) == 2 && isinf(iv[1][1]) && isinf(iv[2][2])
        return :union_of_rays
    end
    return :union_of_intervals
end

function Base.show(io::IO, ::MIME"text/plain", s::WeakIVConfidenceSet)
    println(io, @sprintf("%g%%", 100 * s.level), " ", s.method, " confidence set (",
            s.kind, ")")
    if isempty(s.intervals)
        println(io, "  empty set (the test rejects every value: the model's ",
                "overidentifying or specification restrictions are rejected)")
    else
        parts = [@sprintf("%s%s, %s%s", isinf(a) ? "(" : "[", _iv_fmt(a), _iv_fmt(b),
                          isinf(b) ? ")" : "]") for (a, b) in s.intervals]
        println(io, "  ", join(parts, " ∪ "))
    end
    @printf(io, "  point estimate: %.4g; critical value: %.4g\n", s.estimate,
            s.critical_value)
end

Base.show(io::IO, s::WeakIVConfidenceSet) =
    print(io, "WeakIVConfidenceSet(", s.kind, ", ", s.intervals, ")")

_iv_fmt(x) = isinf(x) ? (x > 0 ? "∞" : "-∞") : @sprintf("%.6g", x)

# ---------------------------------------------------------------------------
# Anderson–Rubin
# ---------------------------------------------------------------------------

"""AR F statistic (any number of endogenous regressors) at `β0` on the design."""
function _iv_ar_stat(des::_IVDesign, β0::AbstractVector)
    u = des.y - des.D * β0
    B, V, _ = _iv_ols(des, u, des.Z)
    return _iv_wald_F(vec(B), V)
end

"""Precomputed reduced-form quantities for the scalar-β AR statistic."""
function _iv_ar_parts(des::_IVDesign)
    size(des.D, 2) == 1 || throw(ArgumentError("AR confidence sets require exactly " *
                                               "one endogenous regressor"))
    Z = des.Z
    k = size(Z, 2)
    B, V, E = _iv_ols(des, hcat(des.y, des.D[:, 1]), Z)
    γ, π = B[:, 1], B[:, 2]
    V11 = V[1:k, 1:k]
    V12 = V[1:k, (k + 1):(2k)]
    V22 = V[(k + 1):(2k), (k + 1):(2k)]
    Σ = (E' * E) ./ _iv_resid_dof(des, k)
    return (γ=γ, π=π, V11=V11, V12=V12, V22=V22, Σ=Σ, Q=Matrix(Symmetric(Z' * Z)), k=k)
end

function _iv_ar_scalar(parts, β::Real)
    k = parts.k
    if isinf(β)
        return _iv_wald_F(parts.π, parts.V22)
    end
    g = parts.γ .- β .* parts.π
    Vb = parts.V11 .- β .* (parts.V12 .+ parts.V12') .+ β^2 .* parts.V22
    return _iv_wald_F(g, Vb)
end

"""Region `{β : a2 β² + a1 β + a0 ≤ 0}` as a list of intervals."""
function _iv_quadratic_region(a2::Real, a1::Real, a0::Real)
    scale = max(abs(a2), abs(a1), abs(a0), floatmin(Float64))
    if abs(a2) <= 1e-14 * scale
        if abs(a1) <= 1e-14 * scale
            return a0 <= 0 ? [(-Inf, Inf)] : Tuple{Float64,Float64}[]
        end
        r = -a0 / a1
        return a1 > 0 ? [(-Inf, r)] : [(r, Inf)]
    end
    disc = a1^2 - 4 * a2 * a0
    if a2 > 0
        disc < 0 && return Tuple{Float64,Float64}[]
        q = -0.5 * (a1 + copysign(sqrt(disc), a1))
        r1, r2 = q / a2, (q == 0 ? 0.0 : a0 / q)
        return [(min(r1, r2), max(r1, r2))]
    else
        disc <= 0 && return [(-Inf, Inf)]
        q = -0.5 * (a1 + copysign(sqrt(disc), a1))
        r1, r2 = q / a2, (q == 0 ? 0.0 : a0 / q)
        return [(-Inf, min(r1, r2)), (max(r1, r2), Inf)]
    end
end

"""
Analytic AR confidence region for one endogenous regressor.

Homoskedastic (any k) and robust/cluster with k = 1: the acceptance region is a
quadratic inequality, solved in closed form. Robust/cluster with k > 1: the
statistic is a ratio g(β)'V(β)⁻¹g(β) with g affine and V quadratic in β, so
`det V(β) (AR(β) − c)` is a polynomial of degree ≤ 2k; its real roots are found as
eigenvalues of the Chebyshev colleague matrix and polished by bisection on the exact
statistic (no grid search).
"""
function _iv_ar_region(des::_IVDesign, parts, c::Real, center::Real, scale::Real)
    k = parts.k
    ck = c * k
    if des.vcov_kind === :simple
        Q, Σ, γ, π = parts.Q, parts.Σ, parts.γ, parts.π
        a2 = dot(π, Q * π) - ck * Σ[2, 2]
        a1 = -2 * dot(γ, Q * π) + 2 * ck * Σ[1, 2]
        a0 = dot(γ, Q * γ) - ck * Σ[1, 1]
        return _iv_quadratic_region(a2, a1, a0)
    elseif k == 1
        γ, π = parts.γ[1], parts.π[1]
        a2 = π^2 - ck * parts.V22[1, 1]
        a1 = -2 * γ * π + 2 * ck * parts.V12[1, 1]
        a0 = γ^2 - ck * parts.V11[1, 1]
        return _iv_quadratic_region(a2, a1, a0)
    end
    f = β -> _iv_ar_scalar(parts, β) - c
    P = β -> begin
        g = parts.γ .- β .* parts.π
        Vb = Symmetric(parts.V11 .- β .* (parts.V12 .+ parts.V12') .+ β^2 .* parts.V22)
        F = cholesky(Vb; check=false)
        issuccess(F) || return -ck * det(Vb)
        return det(F) * (dot(g, F \ g) - ck)
    end
    roots = _iv_poly_real_roots(P, 2k, center, scale)
    return _iv_region_from_roots(f, roots, f(-Inf), f(Inf))
end

"""Real roots of a polynomial (degree ≤ `deg`) given as a function, via Chebyshev
interpolation on `center ± scale` and the colleague-matrix eigenvalues."""
function _iv_poly_real_roots(P, deg::Int, center::Real, scale::Real)
    N = deg + 1
    t = [cos(π * (j + 0.5) / N) for j in 0:(N - 1)]
    v = [P(center + scale * tj) for tj in t]
    c = [2 / N * sum(v[j + 1] * cos(m * π * (j + 0.5) / N) for j in 0:(N - 1))
         for m in 0:deg]
    c[1] /= 2
    cmax = maximum(abs, c)
    cmax == 0 && return Float64[]
    d = deg
    while d > 0 && abs(c[d + 1]) <= 1e-12 * cmax
        d -= 1
    end
    d == 0 && return Float64[]
    if d == 1
        return [center + scale * (-c[1] / c[2])]
    end
    A = zeros(d, d)
    A[1, 2] = 1.0
    for i in 2:(d - 1)
        A[i, i - 1] = 0.5
        A[i, i + 1] = 0.5
    end
    A[d, :] .= -c[1:d] ./ (2 * c[d + 1])
    A[d, d - 1] += 0.5
    ev = eigvals(A)
    rts = [real(z) for z in ev if abs(imag(z)) <= 1e-6 * (1 + abs(real(z)))]
    return sort!(center .+ scale .* rts)
end

"""Bisection for a sign change of `f` on `[a, b]` (finite endpoints)."""
function _iv_bisect(f, a::Float64, b::Float64; iters::Int=200)
    fa = f(a)
    for _ in 1:iters
        m = (a + b) / 2
        (m == a || m == b) && break
        fm = f(m)
        if (fm <= 0) == (fa <= 0)
            a, fa = m, fm
        else
            b = m
        end
    end
    return (a + b) / 2
end

"""
Build `{β : f(β) ≤ 0}` from candidate roots of f, polishing boundaries by bisection
on `f` itself and searching outward if the tails disagree with the limits at ±∞.
"""
function _iv_region_from_roots(f, roots::Vector{Float64}, f_neg_inf::Real,
                               f_pos_inf::Real)
    rts = unique(r -> round(r; sigdigits=12), filter(isfinite, roots))
    sort!(rts)
    pts = Float64[]
    if isempty(rts)
        push!(pts, 0.0)
    else
        push!(pts, rts[1] - (1 + abs(rts[1])))
        for i in 1:(length(rts) - 1)
            push!(pts, (rts[i] + rts[i + 1]) / 2)
        end
        push!(pts, rts[end] + (1 + abs(rts[end])))
    end
    # extend outward until the sign matches the limit at ±∞
    lo = pts[1]
    while ((f(lo) <= 0) != (f_neg_inf <= 0)) && lo > -1e15
        lo = lo - 4 * (1 + abs(lo))
    end
    lo != pts[1] && pushfirst!(pts, lo)
    hi = pts[end]
    while ((f(hi) <= 0) != (f_pos_inf <= 0)) && hi < 1e15
        hi = hi + 4 * (1 + abs(hi))
    end
    hi != pts[end] && push!(pts, hi)
    acc = [f(x) <= 0 for x in pts]
    # boundaries between consecutive sample points
    bounds = Float64[]
    for i in 1:(length(pts) - 1)
        if acc[i] != acc[i + 1]
            push!(bounds, _iv_bisect(f, pts[i], pts[i + 1]))
        end
    end
    # assemble intervals
    out = Tuple{Float64,Float64}[]
    cur_start = acc[1] ? -Inf : NaN
    bi = 1
    state = acc[1]
    for i in 1:(length(pts) - 1)
        if acc[i] != acc[i + 1]
            b = bounds[bi]
            bi += 1
            if state
                push!(out, (cur_start, b))
            else
                cur_start = b
            end
            state = !state
        end
    end
    state && push!(out, (cur_start, Inf))
    return out
end

# ---------------------------------------------------------------------------
# Moreira (2003) CLR and Kleibergen (2002) K statistics (homoskedastic)
# ---------------------------------------------------------------------------

function _iv_moreira_parts(des::_IVDesign)
    size(des.D, 2) == 1 || throw(ArgumentError("CLR and K tests require exactly one " *
                                               "endogenous regressor"))
    Z = des.Z
    k = size(Z, 2)
    Y = hcat(des.y, des.D[:, 1])
    ZtY = Z' * Y
    M = Symmetric(ZtY' * (Symmetric(Z' * Z) \ ZtY))       # Y'P_Z Y
    E = Y - Z * (Z \ Y)
    Ω = Symmetric((E' * E) ./ _iv_resid_dof(des, k))
    return (M=Matrix(M), Ω=Matrix(Ω), Ωi=inv(Ω), k=k, dof=_iv_resid_dof(des, k))
end

function _iv_moreira_stats(parts, β::Real)
    b0, a0 = isinf(β) ? ([0.0, 1.0], [1.0, 0.0]) : ([1.0, -β], [β, 1.0])
    M, Ω, Ωi = parts.M, parts.Ω, parts.Ωi
    sb = dot(b0, Ω * b0)
    ta = Ωi * a0
    st = dot(a0, ta)
    QS = dot(b0, M * b0) / sb
    QT = dot(ta, M * ta) / st
    QST = dot(b0, M * ta) / sqrt(sb * st)
    return QS, QT, QST
end

function _iv_lr_stat(QS, QT, QST)
    disc = (QS + QT)^2 - 4 * (QS * QT - QST^2)
    return 0.5 * (QS - QT + sqrt(max(disc, 0.0)))
end

"""Gauss–Legendre nodes and weights on [-1, 1] (Golub–Welsch)."""
function _iv_gauss_legendre(n::Int)
    β = [i / sqrt(4i^2 - 1) for i in 1:(n - 1)]
    E = eigen(SymTridiagonal(zeros(n), β))
    return E.values, 2 .* E.vectors[1, :] .^ 2
end

const _IV_GL_NODES, _IV_GL_WEIGHTS = _iv_gauss_legendre(96)

"""
Conditional p-value of the LR statistic given `Q_T = qT` (Andrews, Moreira & Stock
2006, 2007): `1 − 2K ∫₀^{π/2} F_{χ²_k}((qT + m)/(1 + qT sin²u / m)) cos^{k−2}u du`
with `K = Γ(k/2)/(√π Γ((k−1)/2))` (the substitution s = sin u of their formula).
For k = 1, LR equals the AR statistic and F(1, dof) is used (as in R `ivmodel`).
"""
function _iv_clr_pvalue(m::Real, qT::Real, k::Int, dof::Real)
    m <= 0 && return 1.0
    k == 1 && return ccdf(FDist(1, dof), m)    # LR = AR when k = 1
    # 2K = 1 / ∫₀^{π/2} cos^{k−2}u du; using the same quadrature for the normalizing
    # constant makes the formula exact at qT = 0 (LR = Q_S ~ χ²_k).
    χ = Chisq(k)
    s = 0.0
    norm = 0.0
    for (x, w) in zip(_IV_GL_NODES, _IV_GL_WEIGHTS)
        u = (x + 1) * π / 4                         # map [-1, 1] -> [0, π/2]
        c = cos(u)^(k - 2)
        s += w * cdf(χ, (qT + m) / (1 + qT * sin(u)^2 / m)) * c
        norm += w * c
    end
    return clamp(1 - s / norm, 0.0, 1.0)
end

function _iv_clr_p(parts, β)
    QS, QT, QST = _iv_moreira_stats(parts, β)
    return _iv_clr_pvalue(_iv_lr_stat(QS, QT, QST), QT, parts.k, parts.dof)
end

function _iv_k_p(parts, β)
    QS, QT, QST = _iv_moreira_stats(parts, β)
    return ccdf(Chisq(1), QST^2 / QT)
end

"""
Numerical inversion of a p-value function on the compactified line
`β = center + scale·tan θ`, `θ ∈ (−π/2, π/2)`, with bisection refinement of every
acceptance boundary; the limits at ±∞ are evaluated exactly.
"""
function _iv_invert_pvalue(pfun, α::Real, center::Real, scale::Real; ngrid::Int=2001)
    θs = [-π / 2 + π * j / (ngrid + 1) for j in 1:ngrid]
    βof(θ) = center + scale * tan(θ)
    g = θ -> (abs(θ) >= π / 2 ? pfun(θ < 0 ? -Inf : Inf) : pfun(βof(θ))) - α
    pts = vcat(-π / 2, θs, π / 2)
    vals = [g(θ) for θ in pts]
    acc = vals .>= 0
    out = Tuple{Float64,Float64}[]
    state = acc[1]
    start = -Inf
    for i in 1:(length(pts) - 1)
        if acc[i] != acc[i + 1]
            a, b = pts[i], pts[i + 1]
            # bisection in θ on g (accept ⇔ g ≥ 0)
            ga_acc = acc[i]
            for _ in 1:100
                mθ = (a + b) / 2
                (mθ == a || mθ == b) && break
                if (g(mθ) >= 0) == ga_acc
                    a = mθ
                else
                    b = mθ
                end
            end
            bd = βof((a + b) / 2)
            if state
                push!(out, (start, bd))
            else
                start = bd
            end
            state = !state
        end
    end
    state && push!(out, (start, Inf))
    return out
end

# ---------------------------------------------------------------------------
# Heteroskedasticity- / cluster-robust K and CLR (Kleibergen 2005)
# ---------------------------------------------------------------------------
#
# With reduced-form estimates R = [γ̂ π̂] (k × 2) and the robust / cluster covariance
# V of vec(R) (blocks V_rs = Cov(R e_r, R e_s)), write for directions b and a in R²
# ĝ = R b (the AR moment, b = (1, −β)) and t = R a (a = (β, 1), the direction of the
# structural Jacobian). Then with Ω = Var(ĝ), Δ = Cov(t, ĝ), D̃ = t − ΔΩ⁻¹ĝ and
# Ψ = Var(t) − ΔΩ⁻¹Δ':
#   AR = ĝ'Ω⁻¹ĝ,  K = (D̃'Ω⁻¹ĝ)² / D̃'Ω⁻¹D̃,  J = AR − K,  rk = D̃'Ψ⁻¹D̃,
#   LR = ½ [AR − rk + √((AR + rk)² − 4 J rk)],
# and the CLR p-value is Moreira's conditional one with Q_T replaced by rk. Every
# statistic is invariant to rescaling b and a, so β = ±∞ is handled with b = (0, 1),
# a = (1, 0). With the homoskedastic covariance V = Σ ⊗ (Z'Z)⁻¹ the statistics equal
# Moreira's Q_S, Q_ST²/Q_T and Q_T exactly.

function _iv_robust_moreira_parts(des::_IVDesign)
    size(des.D, 2) == 1 || throw(ArgumentError("CLR and K tests require exactly one " *
                                               "endogenous regressor"))
    parts = _iv_ar_parts(des)
    k = parts.k
    return (R=hcat(parts.γ, parts.π), V=(parts.V11, parts.V12, parts.V12', parts.V22),
            k=k, dof=_iv_ref_dof(des, k))
end

"""Covariance block `Σ_rs u_r v_s V_rs` of `R u` and `R v`."""
_iv_rm_cov(V, u, v) = u[1] * v[1] .* V[1] .+ u[1] * v[2] .* V[2] .+
                      u[2] * v[1] .* V[3] .+ u[2] * v[2] .* V[4]

function _iv_robust_moreira_stats(parts, β::Real)
    b, a = isinf(β) ? ([0.0, 1.0], [1.0, 0.0]) :
           ([1.0, -β] ./ sqrt(1 + β^2), [β, 1.0] ./ sqrt(1 + β^2))
    R, V = parts.R, parts.V
    g = R * b
    t = R * a
    Ω = Symmetric(_iv_rm_cov(V, b, b))
    Δ = _iv_rm_cov(V, a, b)
    Vt = _iv_rm_cov(V, a, a)
    Ωi = pinv(Matrix(Ω))
    D̃ = t .- Δ * (Ωi * g)
    Ψ = Symmetric(Vt .- Δ * Ωi * Δ')
    AR = dot(g, Ωi * g)
    den = dot(D̃, Ωi * D̃)
    K = den > 0 ? dot(D̃, Ωi * g)^2 / den : 0.0
    K = min(max(K, 0.0), AR)
    rk = max(dot(D̃, pinv(Matrix(Ψ)) * D̃), 0.0)
    return AR, K, AR - K, rk
end

_iv_robust_lr(AR, J, rk) = 0.5 * (AR - rk + sqrt(max((AR + rk)^2 - 4 * J * rk, 0.0)))

function _iv_robust_clr_p(parts, β)
    AR, _, J, rk = _iv_robust_moreira_stats(parts, β)
    return _iv_clr_pvalue(_iv_robust_lr(AR, J, rk), rk, parts.k, parts.dof)
end

function _iv_robust_k_p(parts, β)
    _, K, _, _ = _iv_robust_moreira_stats(parts, β)
    return ccdf(Chisq(1), K)
end

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

function _iv_check_method(method::Symbol)
    method in (:ar, :clr, :k, :jackknife_ar) ||
        throw(ArgumentError("method must be :ar, :clr, :k or :jackknife_ar, got " *
                            ":$method"))
    return method
end

"""
    weak_iv_test(r::IVEstimate; beta0=0.0, method=:ar) -> DiagnosticTest

Test of ``H_0: \\beta = \\beta_0`` for the coefficients on the endogenous regressors whose
size does not depend on the strength of the instruments.

Conventional Wald tests of IV coefficients are unreliable when the instruments are
weak: their null distribution depends on the unknown concentration parameter, and no
Wald-type procedure can control size uniformly over it (Dufour 1997; Staiger and Stock
1997). Weak-identification-robust tests instead exploit the fact that, under the null,
the structural error ``Y - D\\beta_0`` is uncorrelated with the instruments, so a
statistic built from it has a known distribution whatever the first stage. The tests
below differ in power and in the models they cover; they are the standard toolkit
reviewed by Andrews, Stock and Sun (2019). All tests work on the model's estimation
sample after partialling out covariates, fixed effects and weights, and they use the
model's covariance type (homoskedastic, HC1 or cluster-robust).

The Anderson–Rubin test (`method = :ar`; Anderson and Rubin 1949) regresses
``Y - D\\beta_0`` on the instruments and tests that all ``k`` instrument coefficients are
zero with a Wald F statistic and reference distribution ``F(k, \\text{dof})``. It is
exact under homoskedastic normal errors, robust to weak instruments in general, covers
several endogenous regressors (with `beta0` a vector), and is efficient in the
just-identified model. With more instruments than endogenous regressors it also has
power against violations of the overidentifying restrictions, so a rejection need not
indicate ``\\beta \\neq \\beta_0``, and its power is diluted when ``k`` is large. The
conditional likelihood-ratio test (`method = :clr`, one endogenous regressor) of Moreira
(2003) concentrates power in the direction of the endogenous regressor: under
homoskedasticity its p-value is computed conditionally on the statistic ``Q_T`` that is
sufficient for instrument strength, following Andrews, Moreira and Stock (2006), whose
results establish near-optimality among invariant similar tests. With robust or
cluster covariance the function computes the heteroskedasticity- and cluster-robust CLR
built on Kleibergen (2005), as implemented in Stata's `weakiv` (Finlay, Magnusson and
Schaffer 2013). With reduced-form estimates ``\\hat\\gamma`` and ``\\hat\\pi`` and their
robust covariance, let ``\\hat g = \\hat\\gamma - \\beta_0\\hat\\pi``,
``\\Omega = \\operatorname{Var}(\\hat g)``,
``\\tilde D = \\hat\\pi - \\operatorname{Cov}(\\hat\\pi, \\hat g)\\Omega^{-1}\\hat g`` and
``\\Psi = \\operatorname{Var}(\\tilde D)``. The statistic is

```math
LR = \\tfrac12\\Bigl[AR - rk + \\sqrt{(AR + rk)^2 - 4\\,J\\,rk}\\Bigr],
```

with ``AR = \\hat g'\\Omega^{-1}\\hat g``,
``K = (\\tilde D'\\Omega^{-1}\\hat g)^2 / \\tilde D'\\Omega^{-1}\\tilde D``, ``J = AR - K``
and ``rk = \\tilde D'\\Psi^{-1}\\tilde D``,
and the p-value is Moreira's conditional p-value with ``Q_T`` replaced by ``rk``. Under
homoskedasticity these reduce exactly to Moreira's statistics, and with ``k = 1`` the
CLR test coincides with the AR test. The Kleibergen (2002, 2005) K, or Lagrange
multiplier, test (`method = :k`) uses the statistic ``K`` above with a ``\\chi^2(1)``
reference; it is not diluted by many instruments but its power can be non-monotone, with
power losses at alternatives where the Jacobian estimate is uninformative. Under
heteroskedasticity the robust CLR test is no longer efficient; the conditional linear
combination tests of Andrews (2016), which address this, are not implemented.

With many instruments, whose number grows with the sample size, the AR test with a
fixed-``k`` reference distribution is no longer valid. The jackknife Anderson–Rubin
test of Mikusheva and Sun (2022) (`method = :jackknife_ar`) handles many, possibly weak
instruments under heteroskedasticity: with ``e = Y - D\\beta_0`` after partialling out
the controls, ``P`` the instrument projection and ``M = I - P``,

```math
AR(\\beta_0) = \\frac{K^{-1/2}\\sum_{i \\ne j} P_{ij} e_i e_j}{\\sqrt{\\hat\\Phi}}, \\qquad
\\hat\\Phi = \\frac{2}{K}\\sum_{i \\ne j}
  \\frac{P_{ij}^2}{M_{ii}M_{jj} + M_{ij}^2}\\, e_i (Me)_i\\, e_j (Me)_j ,
```

compared one-sidedly with ``N(0, 1)``. Its justification is asymptotic in the number of
instruments, so it is not designed for a handful of them; it requires one endogenous
regressor and independent observations (no clustering), and it costs ``O(n^2K)``
operations. As with all tests, non-rejection of ``\\beta = \\beta_0`` is not evidence
that the coefficient equals ``\\beta_0``, and none of these tests addresses the validity
of the instruments.

# Arguments
- `r::IVEstimate`: a fitted IV model; the test uses its sample, controls, weights and
  covariance type.

# Keywords
- `beta0`: the hypothesized value (default `0.0`); a vector with one entry per
  endogenous regressor for the AR test with several endogenous regressors.
- `method::Symbol`: `:ar` (default), `:clr`, `:k` or `:jackknife_ar`; all but `:ar`
  require one endogenous regressor.

# Returns
- A [`DiagnosticTest`](@ref) with the statistic, degrees of freedom and p-value;
  `details` holds `beta0` and the components of the statistic (for CLR and K: `AR`,
  `K`, `J`, `rk`, or `QS`, `QT`, `QST` under homoskedasticity; for the jackknife AR
  test the number of instruments `K`).

# Examples
```julia
using DrSnow, DataFrames, StableRNGs
rng = StableRNG(4)
n = 1_000
Z = randn(rng, n, 3)
v = randn(rng, n)
d = Z * [0.15, 0.10, 0.05] .+ v
y = 0.5 .* d .+ 0.7 .* v .+ randn(rng, n)
df = DataFrame(y=y, d=d, z1=Z[:, 1], z2=Z[:, 2], z3=Z[:, 3])
r = iv_regression(df, :y, :d, [:z1, :z2, :z3])
weak_iv_test(r; beta0=0.0)                      # Anderson–Rubin
weak_iv_test(r; beta0=0.0, method=:clr)         # robust CLR (HC1)
```

# References
- Anderson, T. W., & Rubin, H. (1949). Estimation of the parameters of a single
  equation in a complete system of stochastic equations. *Annals of Mathematical
  Statistics*, 20(1), 46–63.
- Dufour, J.-M. (1997). Some impossibility theorems in econometrics with applications
  to structural and dynamic models. *Econometrica*, 65(6), 1365–1388.
- Kleibergen, F. (2002). Pivotal statistics for testing structural parameters in
  instrumental variables regression. *Econometrica*, 70(5), 1781–1803.
- Moreira, M. J. (2003). A conditional likelihood ratio test for structural models.
  *Econometrica*, 71(4), 1027–1048.
- Kleibergen, F. (2005). Testing parameters in GMM without assuming that they are
  identified. *Econometrica*, 73(4), 1103–1123.
- Andrews, D. W. K., Moreira, M. J., & Stock, J. H. (2006). Optimal two-sided
  invariant similar tests for instrumental variables regression. *Econometrica*,
  74(3), 715–752.
- Finlay, K., Magnusson, L. M., & Schaffer, M. E. (2013). WEAKIV: Stata module to
  perform weak-instrument-robust tests and confidence intervals for
  instrumental-variable (IV) estimation of linear, probit and tobit models.
  Statistical Software Components S457684, Boston College Department of Economics.
- Andrews, I. (2016). Conditional linear combination tests for weakly identified
  models. *Econometrica*, 84(6), 2155–2182.
- Andrews, I., Stock, J. H., & Sun, L. (2019). Weak instruments in instrumental
  variables regression: Theory and practice. *Annual Review of Economics*, 11,
  727–753.
- Mikusheva, A., & Sun, L. (2022). Inference with many weak instruments. *Review of
  Economic Studies*, 89(5), 2663–2686.
"""
function weak_iv_test(r::IVEstimate; beta0=0.0, method::Symbol=:ar)
    _iv_check_method(method)
    des = r.design
    p, k = size(des.D, 2), size(des.Z, 2)
    b0 = beta0 isa Real ? fill(float(beta0), p) : float.(collect(beta0))
    length(b0) == p ||
        throw(ArgumentError("beta0 must have one entry per endogenous regressor ($p)"))
    null = p == 1 ? "coefficient on $(r.endogenous[1]) equals $(b0[1])" :
           "coefficients on $(join(r.endogenous, ", ")) equal $(b0)"
    if method === :ar
        F = _iv_ar_stat(des, b0)
        dof = _iv_ref_dof(des, k)
        return DiagnosticTest("Anderson–Rubin test", null, F, ccdf(FDist(k, dof), F);
                              dof=(k, dof), method="Anderson–Rubin, " * r.vcov_type,
                              note="Size is robust to weak instruments. With more " *
                                   "instruments than endogenous regressors the test " *
                                   "also rejects when overidentifying restrictions fail.",
                              details=(beta0=b0,))
    end
    if method === :jackknife_ar
        parts = _iv_jar_parts(des)
        s = _iv_jar_stat(parts, b0[1])
        return DiagnosticTest("Jackknife Anderson–Rubin test (Mikusheva & Sun 2022)",
                              null, s, ccdf(Normal(), s);
                              method="jackknife AR with cross-fit variance, one-sided " *
                                     "N(0, 1); K = $(parts.K) instruments",
                              note="Size is robust to weak instruments as the number " *
                                   "of instruments grows, under heteroskedasticity.",
                              details=(beta0=b0, K=parts.K))
    end
    p == 1 || throw(ArgumentError("CLR and K tests require one endogenous regressor"))
    if des.vcov_kind !== :simple
        rp = _iv_robust_moreira_parts(des)
        AR, K, J, rk = _iv_robust_moreira_stats(rp, b0[1])
        lab = r.vcov_type
        if method === :clr
            lr = _iv_robust_lr(AR, J, rk)
            return DiagnosticTest("Conditional likelihood-ratio test, robust " *
                                  "(Kleibergen 2005)", null, lr,
                                  _iv_clr_pvalue(lr, rk, k, rp.dof); dof=(k,),
                                  method="CLR with " * lab * " covariance; p-value " *
                                         "conditional on the rk statistic",
                                  note="Size is robust to weak instruments. Under " *
                                       "heteroskedasticity this CLR is not the " *
                                       "efficient test (Andrews 2016).",
                                  details=(AR=AR, K=K, J=J, rk=rk, beta0=b0))
        else
            return DiagnosticTest("Kleibergen K (LM) test, robust", null, K,
                                  ccdf(Chisq(1), K); dof=(1,),
                                  method="K statistic with " * lab * " covariance, " *
                                         "χ²(1)",
                                  note="Size is robust to weak instruments; power can " *
                                       "be non-monotone (spurious non-rejections far " *
                                       "from the 2SLS estimate).",
                                  details=(AR=AR, K=K, J=J, rk=rk, beta0=b0))
        end
    end
    parts = _iv_moreira_parts(des)
    QS, QT, QST = _iv_moreira_stats(parts, b0[1])
    if method === :clr
        lr = _iv_lr_stat(QS, QT, QST)
        return DiagnosticTest("Conditional likelihood-ratio test (Moreira 2003)", null,
                              lr, _iv_clr_pvalue(lr, QT, k, parts.dof); dof=(k,),
                              method="CLR, homoskedastic; p-value conditional on Q_T",
                              note="Size is robust to weak instruments.",
                              details=(QS=QS, QT=QT, QST=QST, beta0=b0))
    else
        K = QST^2 / QT
        return DiagnosticTest("Kleibergen K (LM) test", null, K, ccdf(Chisq(1), K);
                              dof=(1,), method="K statistic, homoskedastic, χ²(1)",
                              note="Size is robust to weak instruments; power can be " *
                                   "non-monotone (spurious non-rejections far from the " *
                                   "2SLS estimate).",
                              details=(QS=QS, QT=QT, QST=QST, beta0=b0))
    end
end

"""
    weak_iv_confidence_set(r::IVEstimate; method=:ar, level=r.level)
        -> WeakIVConfidenceSet

Confidence set for the coefficient of a single endogenous regressor obtained by
inverting a weak-instrument-robust test.

The set collects every value ``\\beta_0`` that the chosen test of
[`weak_iv_test`](@ref) does not reject at level ``1 - \\text{level}``. Its coverage is
correct whatever the strength of the instruments, which is the property Wald intervals
lack (Dufour 1997), and its shape is informative: a bounded interval when the
instruments are informative, an unbounded set (two rays or the whole line) when they
are not, and possibly an empty set in overidentified models when the AR test rejects
the overidentifying restrictions at every ``\\beta_0``. Reporting the Anderson–Rubin
set next to the 2SLS estimate is the practice recommended by Andrews, Stock and Sun
(2019) for just-identified models, where AR is efficient; with several instruments the
CLR set is generally more informative (Mikusheva 2010).

The Anderson–Rubin set (`method = :ar`, the default) is computed analytically. Under
homoskedasticity with any number of instruments, and with a robust or cluster
covariance and one instrument, the acceptance region solves a quadratic inequality in
``\\beta_0``, so the set is a bounded interval, the union of two rays, the whole real
line or (overidentified case) empty; it is bounded if and only if the first-stage Wald
statistic exceeds the critical value. With a robust or cluster covariance and several
instruments the AR statistic is a ratio of quadratic forms, and the boundaries are
the real roots of a polynomial of degree at most ``2k``, found as eigenvalues of the
Chebyshev colleague matrix and polished by bisection on the exact statistic, so no
grid search is involved. The sets for `:clr`, `:k` and `:jackknife_ar` are obtained by
numerical inversion of the p-value function on the compactified real line: 2,001 points
in ``\\theta = \\arctan((\\beta - \\hat\\beta)/s)``, plus the exact limits at
``\\pm\\infty``, with bisection refinement of every boundary. Because the K test's
power can be non-monotone, its set may include regions far from the truth that the
CLR set, recommended by Mikusheva (2010), tends to exclude.

# Arguments
- `r::IVEstimate`: a fitted IV model with one endogenous regressor.

# Keywords
- `method::Symbol`: the inverted test, `:ar` (default), `:clr`, `:k` or `:jackknife_ar`
  (see [`weak_iv_test`](@ref) for their properties and requirements).
- `level::Real`: the confidence level (default `r.level`, usually 0.95).

# Returns
- A [`WeakIVConfidenceSet`](@ref); `cs.kind` and `cs.intervals` describe the set,
  `β₀ in cs` tests membership and `DrSnow.pvalue(cs, β₀)` evaluates the p-value
  function.

# Examples
```julia
using DrSnow, DataFrames, StableRNGs
rng = StableRNG(5)
n = 500
z = randn(rng, n)
v = randn(rng, n)
d = 0.08 .* z .+ v                              # weak first stage
y = 1.0 .* d .+ 0.8 .* v .+ 0.6 .* randn(rng, n)
df = DataFrame(y=y, d=d, z=z)
r = late_2sls(df, :y, :d, :z)
cs = weak_iv_confidence_set(r)
cs.kind, cs.intervals                           # may be unbounded
DrSnow.pvalue(cs, 0.0)
```

# References
- Anderson, T. W., & Rubin, H. (1949). Estimation of the parameters of a single
  equation in a complete system of stochastic equations. *Annals of Mathematical
  Statistics*, 20(1), 46–63.
- Dufour, J.-M. (1997). Some impossibility theorems in econometrics with applications
  to structural and dynamic models. *Econometrica*, 65(6), 1365–1388.
- Moreira, M. J. (2003). A conditional likelihood ratio test for structural models.
  *Econometrica*, 71(4), 1027–1048.
- Mikusheva, A. (2010). Robust confidence sets in the presence of weak instruments.
  *Journal of Econometrics*, 157(2), 236–247.
- Andrews, I., Stock, J. H., & Sun, L. (2019). Weak instruments in instrumental
  variables regression: Theory and practice. *Annual Review of Economics*, 11,
  727–753.
- Mikusheva, A., & Sun, L. (2022). Inference with many weak instruments. *Review of
  Economic Studies*, 89(5), 2663–2686.
"""
function weak_iv_confidence_set(r::IVEstimate; method::Symbol=:ar, level::Real=r.level)
    _iv_check_method(method)
    0 < level < 1 || throw(ArgumentError("level must be in (0, 1)"))
    des = r.design
    size(des.D, 2) == 1 || throw(ArgumentError("confidence sets require exactly one " *
                                               "endogenous regressor"))
    k = size(des.Z, 2)
    b = r.coef[1]
    se = sqrt(r.vcov[1, 1])
    scale = max(10 * se, 1e-8 * max(1.0, abs(b)))
    isfinite(scale) || (scale = max(1.0, abs(b)))
    α = 1 - level
    if method === :ar
        parts = _iv_ar_parts(des)
        dof = _iv_ref_dof(des, k)
        c = quantile(FDist(k, dof), level)
        iv = _iv_ar_region(des, parts, c, b, scale)
        pf = β -> ccdf(FDist(k, dof), _iv_ar_scalar(parts, β))
        return WeakIVConfidenceSet("Anderson–Rubin (" * r.vcov_type * ")", level,
                                   _iv_set_kind(iv), iv, c, b, pf)
    end
    if method === :jackknife_ar
        parts = _iv_jar_parts(des)
        pf = β -> ccdf(Normal(), _iv_jar_stat(parts, β))
        iv = _iv_invert_pvalue(pf, α, b, scale)
        return WeakIVConfidenceSet("jackknife Anderson–Rubin (Mikusheva & Sun 2022)",
                                   level, _iv_set_kind(iv), iv,
                                   quantile(Normal(), level), b, pf)
    end
    if des.vcov_kind !== :simple
        rp = _iv_robust_moreira_parts(des)
        if method === :clr
            pf = β -> _iv_robust_clr_p(rp, β)
            iv = _iv_invert_pvalue(pf, α, b, scale)
            return WeakIVConfidenceSet("conditional likelihood ratio (Kleibergen 2005, " *
                                       r.vcov_type * ")", level, _iv_set_kind(iv), iv,
                                       NaN, b, pf)
        else
            pf = β -> _iv_robust_k_p(rp, β)
            iv = _iv_invert_pvalue(pf, α, b, scale)
            return WeakIVConfidenceSet("Kleibergen K (" * r.vcov_type * ")", level,
                                       _iv_set_kind(iv), iv, quantile(Chisq(1), level),
                                       b, pf)
        end
    end
    parts = _iv_moreira_parts(des)
    if method === :clr
        pf = β -> _iv_clr_p(parts, β)
        iv = _iv_invert_pvalue(pf, α, b, scale)
        return WeakIVConfidenceSet("conditional likelihood ratio (homoskedastic)", level,
                                   _iv_set_kind(iv), iv, NaN, b, pf)
    else
        pf = β -> _iv_k_p(parts, β)
        iv = _iv_invert_pvalue(pf, α, b, scale)
        return WeakIVConfidenceSet("Kleibergen K (homoskedastic)", level,
                                   _iv_set_kind(iv), iv, quantile(Chisq(1), level), b,
                                   pf)
    end
end
