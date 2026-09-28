# Sensitivity analysis for violations of parallel trends (Rambachan & Roth 2023).
#
# Notation (as in the paper and the HonestDiD R package): the event-study vector
# β̂ = (β̂_pre, β̂_post) with covariance Σ estimates β = τ + δ, where τ_pre = 0 and δ
# is the difference in trends. The target is θ = l'τ_post. Restrictions δ ∈ Δ are
# polyhedra {δ : A δ ≤ d} (or finite unions of them):
#   Δ^SD(M)   |δ_{t+1} - 2δ_t + δ_{t-1}| ≤ M                     (smoothness)
#   Δ^RM(M̄)   |δ_{t+1} - δ_t| ≤ M̄ max_{s<0} |δ_{s+1} - δ_s|     (relative magnitudes)
#   Δ^SDRM(M̄) second differences post ≤ M̄ × max pre second difference (linear trend)
# optionally intersected with a sign restriction on δ_post (bias_sign) or a
# monotonicity restriction on δ (monotonicity).
#
# Confidence sets:
#   - FLCI (Δ^SD only): optimal fixed-length affine confidence interval
#     (Armstrong & Kolesár 2018; Rambachan & Roth 2023, Section 4.1).
#   - conditional: Andrews, Roth & Pakes (2023) conditional moment-inequality test,
#     inverted over a grid of θ values.
#   - hybrid_flci / hybrid_lf: conditional test with a first-stage FLCI or
#     least-favorable test of size κ = α/10.
# The algorithms follow the HonestDiD R package (v0.2.x) step by step, so results
# agree with it up to its Monte Carlo approximations (folded-normal quantiles with
# 10⁶ draws; least-favorable critical values with 1000 draws) and grid resolution.

"""
    HonestDiDResult

Robust confidence sets for a post-treatment target ``\\theta = l'\\tau_{post}`` under a
family of restrictions ``\\Delta(M)`` on violations of parallel trends, one set per
value of ``M`` (Rambachan and Roth, 2023). Returned by [`honest_did`](@ref).

The event-study coefficients are modelled as ``\\beta = \\tau + \\delta``, where
``\\tau`` collects the causal effects (zero before treatment) and ``\\delta`` the
difference in trends between treated and comparison units. Each confidence set is
valid, uniformly over ``\\delta \\in \\Delta(M)``, for the partially identified
``\\theta``; the conventional interval in `original` is valid only when
``\\delta_{post} = 0``. The breakdown value is the smallest ``M`` in the grid at
which the robust set includes zero: how large a violation of parallel trends,
measured in the units of the restriction, is needed before the sign of the effect
is no longer determined.

# Fields
- `restriction::Symbol`: `:smoothness` (``\\Delta^{SD}(M)``),
  `:relative_magnitudes` (``\\Delta^{RM}(\\bar M)``) or `:linear_trend`
  (``\\Delta^{SDRM}(\\bar M)``).
- `delta::String`: the restriction set in the notation of the R package HonestDiD
  (e.g. `"DeltaRM"`, or `"DeltaSDPB"` for smoothness with positive bias).
- `method::Symbol`: `:flci`, `:conditional`, `:hybrid_flci` or `:hybrid_lf`.
- `M::Vector{Float64}`: the ``M`` (or ``\\bar M``) values.
- `lb::Vector{Float64}`, `ub::Vector{Float64}`: bounds of the robust confidence set
  for each ``M`` (its convex hull; `NaN` when every value of ``\\theta`` is rejected,
  i.e. the data reject ``\\Delta(M)`` itself).
- `estimate::Float64`: ``l'\\hat\\beta_{post}``.
- `original::Tuple{Float64,Float64}`: the conventional confidence interval for
  ``l'\\hat\\beta_{post}``, valid only under exact parallel trends.
- `breakdown::Float64`: the smallest ``M`` in the grid whose robust confidence set
  contains zero (`NaN` if none does); see [`honest_breakdown`](@ref) for a
  continuous search.
- `l_vec::Vector{Float64}`, `post_periods::Vector{Int}`, `pre_periods::Vector{Int}`:
  target weights and the relative periods they refer to.
- `level::Float64`: confidence level.
- `details::NamedTuple`: `betahat`, `sigma`, grid settings, FLCI weights and, for
  results from an event study, the `event_study` itself.

# Accessors
[`confint(::HonestDiDResult)`](@ref) returns the robust sets as a matrix.

# References
- Rambachan, A., & Roth, J. (2023). A more credible approach to parallel trends.
  *Review of Economic Studies*, 90(5), 2555–2591.
"""
struct HonestDiDResult
    restriction::Symbol
    delta::String
    method::Symbol
    M::Vector{Float64}
    lb::Vector{Float64}
    ub::Vector{Float64}
    estimate::Float64
    original::Tuple{Float64,Float64}
    breakdown::Float64
    l_vec::Vector{Float64}
    post_periods::Vector{Int}
    pre_periods::Vector{Int}
    level::Float64
    details::NamedTuple
end

"""
    confint(r::HonestDiDResult) -> Matrix{Float64}

Robust confidence sets of a Rambachan–Roth sensitivity analysis, one row per value of
``M``.

The sets are those computed by [`honest_did`](@ref) at its `level`; the level cannot
be changed after the fact. Rows with `NaN` bounds correspond to values of ``M`` for
which every ``\\theta`` is rejected, i.e. the restriction ``\\Delta(M)`` is rejected
by the pre-treatment coefficients.

# Arguments
- `r::HonestDiDResult`: a sensitivity analysis.

# Returns
- `Matrix{Float64}`: a `length(r.M) × 2` matrix of lower and upper bounds.
"""
StatsAPI.confint(r::HonestDiDResult) = hcat(r.lb, r.ub)

function Base.show(io::IO, ::MIME"text/plain", r::HonestDiDResult)
    lv = round(Int, 100 * r.level)
    mname = r.restriction === :smoothness ? "M" : "M̄"
    println(io, "Honest DiD sensitivity analysis (Rambachan & Roth 2023)")
    println(io, "Restriction: ", r.delta, "; method: ", r.method, "; ", lv,
            "% confidence sets")
    nz = findall(!iszero, r.l_vec)
    tgt = length(nz) == 1 && r.l_vec[only(nz)] == 1 ?
          "θ = τ(e = $(r.post_periods[only(nz)]))" :
          "θ = Σ l_e τ(e) over e ∈ {" * join(r.post_periods, ", ") * "}, l = " *
          string(round.(r.l_vec; digits=4))
    println(io, "Target: ", tgt)
    @printf(io, "Estimate l'β̂_post = %.6g; original CI (δ_post = 0): [%.6g, %.6g]\n",
            r.estimate, r.original[1], r.original[2])
    println(io, rpad(mname, 12), rpad("lower", 14), "upper")
    for k in eachindex(r.M)
        @printf(io, "%-12.4g%-14.6g%.6g\n", r.M[k], r.lb[k], r.ub[k])
    end
    if isnan(r.breakdown)
        println(io, "No ", mname, " in the grid gives a confidence set containing 0.")
    else
        @printf(io, "Smallest %s in the grid whose confidence set contains 0: %.4g\n",
                mname, r.breakdown)
    end
    any(isnan, r.lb) && println(io, "NaN: every θ rejected (the data reject Δ(",
                                mname, ") at this level).")
end

Base.show(io::IO, r::HonestDiDResult) =
    print(io, "HonestDiDResult(", r.delta, ", ", r.method, ", ", length(r.M), " values)")

# ---------------------------------------------------------------------------
# Restriction matrices (HonestDiD's .create_A_* functions; columns are
# (δ_pre, δ_post) with the reference period dropped)
# ---------------------------------------------------------------------------

function _did_hd_drop_zero_rows(A)
    keep = [sum(abs2, view(A, i, :)) > 1e-10 for i in axes(A, 1)]
    return A[keep, :]
end

function _did_hd_A_SD(npre, npost; post_only=false)
    T = npre + npost
    At = zeros(T - 1, T + 1)
    for r in 1:(T - 1)
        At[r, r:(r + 2)] = [1.0, -2.0, 1.0]
    end
    At = At[:, setdiff(1:(T + 1), npre + 1)]
    if post_only
        post = (npre + 1):T
        At = At[[any(!iszero, At[i, post]) for i in axes(At, 1)], :]
    end
    return vcat(At, -At)
end

function _did_hd_A_B(npre, npost, dir)
    A = -Matrix{Float64}(I, npre + npost, npre + npost)[(npre + 1):end, :]
    dir === :negative && return -A
    dir === :positive || throw(ArgumentError("bias_sign must be :positive or :negative"))
    return A
end

function _did_hd_A_M(npre, npost, dir; post_only=false)
    T = npre + npost
    A = zeros(T, T)
    for r in 1:(npre - 1)
        A[r, r:(r + 1)] = [1.0, -1.0]
    end
    npre >= 1 && (A[npre, npre] = 1.0)
    if npost > 0
        A[npre + 1, npre + 1] = -1.0
        for r in (npre + 2):T
            A[r, (r - 1):r] = [1.0, -1.0]
        end
    end
    if post_only
        post = (npre + 1):T
        A = A[[any(!iszero, A[i, post]) for i in axes(A, 1)], :]
    end
    dir === :decreasing && return -A
    dir === :increasing ||
        throw(ArgumentError("monotonicity must be :increasing or :decreasing"))
    return A
end

function _did_hd_A_RM(npre, npost, Mbar, s, maxpos)
    T = npre + npost
    At = zeros(T, T + 1)
    for r in 1:T
        At[r, r:(r + 1)] = [-1.0, 1.0]
    end
    v = zeros(1, T + 1)
    v[1, (npre + s):(npre + 1 + s)] = [-1.0, 1.0]
    maxpos || (v .*= -1)
    AUB = vcat(repeat(v, npre, 1), repeat(Mbar .* v, npost, 1))
    A = _did_hd_drop_zero_rows(vcat(At .- AUB, -At .- AUB))
    return A[:, setdiff(1:(T + 1), npre + 1)]
end

function _did_hd_A_SDRM(npre, npost, Mbar, s, maxpos)
    T = npre + npost
    At = zeros(T - 1, T + 1)
    for r in 1:(T - 1)
        At[r, r:(r + 2)] = [1.0, -2.0, 1.0]
    end
    v = zeros(1, T + 1)
    v[1, (npre + 1 + s - 2):(npre + 1 + s)] = [1.0, -2.0, 1.0]
    maxpos || (v .*= -1)
    AUB = vcat(repeat(v, npre - 1, 1), repeat(Mbar .* v, npost, 1))
    A = _did_hd_drop_zero_rows(vcat(At .- AUB, -At .- AUB))
    return A[:, setdiff(1:(T + 1), npre + 1)]
end

# A polyhedral piece {δ : A δ ≤ d} of Δ, with the rows used by the ARP test when
# there are nuisance parameters (several post periods).
struct _DidHDPiece
    A::Matrix{Float64}
    d::Vector{Float64}
    rows::Vector{Int}
end

_did_hd_post_rows(A, npre) =
    [i for i in axes(A, 1) if any(!iszero, view(A, i, (npre + 1):size(A, 2)))]

function _did_hd_extra(npre, npost, bias_sign, monotonicity)
    bias_sign !== nothing && return (_did_hd_A_B(npre, npost, bias_sign), zeros(npost))
    if monotonicity !== nothing
        return (_did_hd_A_M(npre, npost, monotonicity), zeros(npre + npost))
    end
    return (zeros(0, npre + npost), Float64[])
end

"""
Polyhedral pieces of Δ(M) whose union is the restriction set (`for_idset = true`:
without the post-period-moment subsetting, as used for identified sets).
"""
function _did_hd_pieces(restriction, M, npre, npost, bias_sign, monotonicity,
                        post_only; for_idset=false)
    Ae, de = _did_hd_extra(npre, npost, bias_sign, monotonicity)
    pieces = _DidHDPiece[]
    if restriction === :smoothness
        A = vcat(_did_hd_A_SD(npre, npost), Ae)
        d = vcat(fill(float(M), size(A, 1) - size(Ae, 1)), de)
        rows = (post_only && npost > 1 && !for_idset) ? _did_hd_post_rows(A, npre) :
               collect(axes(A, 1))
        push!(pieces, _DidHDPiece(A, d, rows))
        return pieces
    end
    svals = restriction === :relative_magnitudes ? ((-(npre - 1)):0) : ((-(npre - 2)):0)
    for maxpos in (true, false), s in svals
        A0 = restriction === :relative_magnitudes ?
             _did_hd_A_RM(npre, npost, M, s, maxpos) :
             _did_hd_A_SDRM(npre, npost, M, s, maxpos)
        A = vcat(A0, Ae)
        d = vcat(zeros(size(A0, 1)), de)
        if for_idset || !post_only
            push!(pieces, _DidHDPiece(A, d, collect(axes(A, 1))))
        elseif npost > 1
            push!(pieces, _DidHDPiece(A, d, _did_hd_post_rows(A, npre)))
        else
            keep = [i for i in axes(A, 1) if A[i, end] != 0]
            push!(pieces, _DidHDPiece(A[keep, :], d[keep], collect(eachindex(keep))))
        end
    end
    return pieces
end

# ---------------------------------------------------------------------------
# Identified set (used for default grids)
# ---------------------------------------------------------------------------

function _did_hd_idset(pieces, β, l, npre, npost)
    T = npre + npost
    f = vcat(zeros(npre), l)
    Aeq = hcat(Matrix{Float64}(I, npre, npre), zeros(npre, npost))
    θ = dot(l, β[(npre + 1):T])
    lo, hi = Inf, -Inf
    for p in pieces
        smax, vmax, _ = _lp_free(f, p.A, p.d, Aeq, β[1:npre]; maximize=true)
        smin, vmin, _ = _lp_free(f, p.A, p.d, Aeq, β[1:npre]; maximize=false)
        (smax === :optimal && smin === :optimal) || continue
        lo = min(lo, θ - vmax)
        hi = max(hi, θ - vmin)
    end
    return lo, hi
end

# ---------------------------------------------------------------------------
# Fixed-length confidence interval for Δ^SD(M)
# ---------------------------------------------------------------------------

"""
    _did_hd_flci(Σ, M, npre, npost, l, α) -> NamedTuple

Optimal fixed-length confidence interval `ℓ'β̂ ± χ` for `θ = l'τ_post` under
`Δ^SD(M)`: over affine estimators with pre-period weights `ℓ` (unbiased under linear
trends), minimize `χ = cv_α(M b(ℓ)/sd(ℓ)) sd(ℓ)` where `b(ℓ)` is the worst-case bias
per unit of `M`. The bias/standard-deviation frontier is traced exactly by the
solution path of an ℓ₁-penalized quadratic problem (the Lagrangian of HonestDiD's
`.findWorstCaseBiasGivenH`), and `χ` is minimized along it.
"""
function _did_hd_flci(Σ, M, npre, npost, l, α)
    K = npre
    sbar = dot(1:npost, l)
    C0 = sum(abs(dot(1:s, l[(npost - s + 1):npost])) for s in 1:npost) - sbar
    D = Matrix{Float64}(I, K, K)
    for j in 2:K
        D[j, j - 1] = -1.0
    end
    D2 = D * D                       # ℓ = D² z, z = cumsum of the weights w
    Σpre = Σ[1:K, 1:K]
    Σpp = Σ[1:K, (K + 1):end]
    Σpost = dot(l, Σ[(K + 1):end, (K + 1):end] * l)
    G = D2[:, 1:(K - 1)]
    g0 = D2[:, K] .* sbar
    Q = G' * Σpre * G
    r = G' * (Σpre * g0 .+ Σpp * l)
    varf(z) = (ℓ = G * z .+ g0; dot(ℓ, Σpre * ℓ) + 2 * dot(ℓ, Σpp * l) + Σpost)
    biasf(z) = C0 + sum(abs, z; init=0.0) + abs(sbar)
    if K > 1 && !isposdef(Symmetric(Q))
        throw(ArgumentError("the covariance matrix of the pre-period coefficients is " *
                            "singular; the FLCI is not defined"))
    end
    μs, Z = _did_hd_lasso_frontier(Q, r, K)
    hl(z) = (h = sqrt(max(varf(z), 0.0));
             h > 0 || throw(ArgumentError("the target has zero variance"));
             _did_folded_normal_quantile(1 - α, M * biasf(z) / h) * h)
    best_z = Z[:, 1]
    best = hl(best_z)
    for k in 1:(size(Z, 2) - 1)
        za, zb = Z[:, k], Z[:, k + 1]
        φ(t) = hl(za .+ t .* (zb .- za))
        t, v = _did_golden_min(φ, 0.0, 1.0)
        for (tt, vv) in ((t, v), (1.0, φ(1.0)))
            if vv < best
                best = vv
                best_z = za .+ tt .* (zb .- za)
            end
        end
    end
    ℓ = G * best_z .+ g0
    h = sqrt(varf(best_z))
    return (optimal_vec=vcat(ℓ, l), halflength=best, sd=h, bias=biasf(best_z))
end

function _did_hd_lasso_frontier(Q, r, K)
    K == 1 && return ([0.0], zeros(0, 1))
    μs, Z = _did_lasso_path(Q, r)
    return μs, Z
end

function _did_golden_min(f, a, b; tol=1e-10, maxiter=200)
    g = (sqrt(5) - 1) / 2
    c = b - g * (b - a)
    d = a + g * (b - a)
    fc, fd = f(c), f(d)
    for _ in 1:maxiter
        abs(b - a) <= tol && break
        if fc <= fd
            b, d, fd = d, c, fc
            c = b - g * (b - a)
            fc = f(c)
        else
            a, c, fc = c, d, fd
            d = a + g * (b - a)
            fd = f(d)
        end
    end
    x = (a + b) / 2
    return x, f(x)
end

# ---------------------------------------------------------------------------
# Andrews–Roth–Pakes conditional / hybrid tests
# ---------------------------------------------------------------------------

_did_argmax_first(v) = findfirst(==(maximum(v)), v)

# HonestDiD .testInIdentifiedSet (+ first-stage rows): no nuisance parameters.
function _did_hd_test_nonuis(y, Σ, A, d, α; Aadd=nothing, dadd=nothing, lf_cv=nothing,
                             truncate_zero=true)
    st = sqrt.(max.(diag(A * Σ * A'), 0.0))
    At = A ./ st
    dt = d ./ st
    nm = At * y .- dt
    j = _did_argmax_first(nm)
    maxm = nm[j]
    lf_cv !== nothing && maxm > lf_cv && return true
    γ = At[j, :]
    Ab = At .- At[j:j, :]
    db = dt .- dt[j]
    if Aadd !== nothing
        Ab = vcat(Ab, Aadd)
        db = vcat(db, dadd)
    end
    Σγ = Σ * γ
    s2 = dot(γ, Σγ)
    c = Σγ ./ s2
    z = y .- c .* dot(γ, y)
    Ac = Ab * c
    obj = (db .- Ab * z) ./ Ac
    neg = findall(<(0), Ac)
    pos = findall(>(0), Ac)
    vlo = isempty(neg) ? -Inf : maximum(obj[neg])
    vup = isempty(pos) ? Inf : minimum(obj[pos])
    sd = sqrt(s2)
    mu = dt[j]
    q = _did_hd_truncnorm_gen(1 - α, vlo, vup, mu, sd)
    cv = truncate_zero ? max(0.0, q) : q
    return maxm + dt[j] > cv
end

function _did_hd_truncnorm_gen(p, l, u, mu, sd)
    ln, un = (l - mu) / sd, (u - mu) / sd
    ln > un && return NaN
    return mu + sd * _did_truncnorm_quantile(p, ln, un)
end

# HonestDiD .max_program / .check_if_solution_helper
function _did_hd_check(c, tol, sT, γ, Σ, W)
    Σγ = Σ * γ
    f = sT .+ (Σγ ./ dot(γ, Σγ)) .* c
    e1 = zeros(size(W, 2))
    e1[1] = 1.0
    lp = _lp_simplex(f, Matrix(W'), e1)
    lp.status === :optimal || return (false, zeros(size(W, 1)))
    return (abs(c - lp.objective) <= tol, lp.x)
end

_did_roundeps(x) = abs(x) < eps()^(3 / 4) ? 0.0 : x

# HonestDiD .vlo_vup_dual_fn: truncation points by bisection over the value of η at
# which the optimal dual vertex changes.
function _did_hd_vlo_vup_dual(η, sT, γ, Σ, W)
    tol_c, tol_eq = 1e-6, 1e-6
    σB = sqrt(dot(γ, Σ * γ))
    low0 = min(-100.0, η - 20σB)
    high0 = max(100.0, η + 20σB)
    maxiters, switchiters = 10_000, 10
    ok, _ = _did_hd_check(η, tol_eq, sT, γ, Σ, W)
    ok || return (η, Inf)
    bvec = (Σ * γ) ./ dot(γ, Σ * γ)
    function search(start, other, upper)
        ok0, sol = _did_hd_check(start, tol_eq, sT, γ, Σ, W)
        ok0 && return upper ? Inf : -Inf
        dif = 0.0
        iters = 1
        mid = _did_roundeps(dot(sol, sT)) / (1 - dot(sol, bvec))
        while true
            okm, sol = _did_hd_check(mid, tol_eq, sT, γ, Σ, W)
            (okm || iters >= maxiters) && break
            iters += 1
            if iters >= switchiters
                dif = tol_c + 1
                break
            end
            mid = _did_roundeps(dot(sol, sT)) / (1 - dot(sol, bvec))
        end
        lo, hi = upper ? (other, mid) : (mid, other)
        while dif > tol_c && iters < maxiters
            iters += 1
            mid = (lo + hi) / 2
            okm, _ = _did_hd_check(mid, tol_eq, sT, γ, Σ, W)
            if upper
                okm ? (lo = mid) : (hi = mid)
            else
                okm ? (hi = mid) : (lo = mid)
            end
            dif = hi - lo
        end
        return mid
    end
    vup = search(high0, η, true)
    vlo = search(low0, η, false)
    return (vlo, vup)
end

function _did_hd_flci_vlovup(vbar, dbar, S, c)
    V = vcat(vbar', -vbar')
    Vc = V * c
    mm = (dbar .- V * S) ./ Vc
    neg = findall(<(0), Vc)
    pos = findall(>(0), Vc)
    return (isempty(neg) ? -Inf : maximum(mm[neg]), isempty(pos) ? Inf : minimum(mm[pos]))
end

# Hybrid settings: `kind` ∈ (:arp, :lf, :flci).
struct _DidHDHybrid
    kind::Symbol
    κ::Float64
    lf_cv::Float64
    vbar::Vector{Float64}
    halflength::Float64
    flci_vec::Vector{Float64}
end

# HonestDiD .lp_conditional_test_fn; returns `true` when θ is rejected.
function _did_hd_lp_test(yT, XT, Σ, α, hy::_DidHDHybrid, rows, dbar)
    yA = yT[rows]
    XA = XT[rows, :]
    ΣA = Σ[rows, rows]
    k = size(XA, 2)
    sd = sqrt.(max.(diag(ΣA), 0.0))
    W = hcat(sd, XA)
    e1 = zeros(k + 1)
    e1[1] = 1.0
    lp = _lp_simplex(yA, Matrix(W'), e1)
    lp.status === :optimal || return false      # η = -∞ or solver failure: accept
    η = lp.objective
    λ = lp.x
    modsize = α
    if hy.kind === :lf
        modsize = (α - hy.κ) / (1 - hy.κ)
        η > hy.lf_cv && return true
    elseif hy.kind === :flci
        modsize = (α - hy.κ) / (1 - hy.κ)
        V = vcat(hy.vbar', -hy.vbar')
        maximum(V * yT .- dbar) > 0 && return true
    end
    B = λ .> 1e-6
    degenerate = count(B) != k + 1
    XB = XA[B, :]
    fullrank = min(size(XB)...) == 0 ? false : rank(XB) == min(size(XB)...)
    if !fullrank || degenerate
        γ = λ
        Σγ = ΣA * γ
        s2γ = dot(γ, Σγ)
        sT = yA .- Σγ .* (dot(γ, yA) / s2γ)
        vlo, vup = _did_hd_vlo_vup_dual(η, sT, γ, ΣA, W)
        σB2 = s2γ
        abs(σB2) < eps() && return η > 0
        σB = sqrt(σB2)
        maxstat = η / σB
        zlo, zup = _did_hd_zbounds(hy, vlo, vup, σB, γ, rows, yT, Σ, dbar)
    else
        Bc = .!B
        H = hcat(sd[B], XB)
        Hinv = inv(H)
        SB = Matrix{Float64}(I, length(yA), length(yA))[B, :]
        SBc = Matrix{Float64}(I, length(yA), length(yA))[Bc, :]
        ΓB = hcat(sd[Bc], XA[Bc, :]) * Hinv * SB .- SBc
        vB = vec(Hinv[1:1, :] * SB)
        σ2B = dot(vB, ΣA * vB)
        σB = sqrt(σ2B)
        ρ = (ΓB * (ΣA * vB)) ./ σ2B
        mm = (-(ΓB * yA)) ./ ρ .+ dot(vB, yA)
        pos = findall(>(0), ρ)
        neg = findall(<(0), ρ)
        vlo = isempty(pos) ? -Inf : maximum(mm[pos])
        vup = isempty(neg) ? Inf : minimum(mm[neg])
        maxstat = η / σB
        zlo, zup = _did_hd_zbounds(hy, vlo, vup, σB, vB, rows, yT, Σ, dbar)
    end
    (zlo <= maxstat <= zup) || return false
    cval = max(0.0, _did_hd_truncnorm_gen(1 - modsize, zlo, zup, 0.0, 1.0))
    return maxstat > cval
end

function _did_hd_zbounds(hy, vlo, vup, σB, γ, rows, yT, Σ, dbar)
    if hy.kind === :lf
        return vlo / σB, min(vup, hy.lf_cv) / σB
    elseif hy.kind === :flci
        gf = zeros(length(yT))
        gf[rows] = γ
        sg = (Σ * gf) ./ dot(gf, Σ * gf)
        S = yT .- sg .* dot(gf, yT)
        flo, fup = _did_hd_flci_vlovup(hy.vbar, dbar, S, sg)
        return max(vlo, flo) / σB, min(vup, fup) / σB
    end
    return vlo / σB, vup / σB
end

# HonestDiD .compute_least_favorable_cv
function _did_hd_lf_cv(XT, Σ, κ, sims, rng)
    n = size(Σ, 1)
    E = eigen(Symmetric(Matrix(Σ)))
    L = E.vectors * Diagonal(sqrt.(max.(E.values, 0.0)))
    sd = sqrt.(max.(diag(Σ), 0.0))
    vals = Float64[]
    if XT === nothing
        for _ in 1:sims
            ξ = L * randn(rng, n)
            push!(vals, maximum(ξ ./ sd))
        end
    else
        W = hcat(sd, XT)
        e1 = zeros(size(W, 2))
        e1[1] = 1.0
        Wt = Matrix(W')
        for _ in 1:sims
            ξ = L * randn(rng, n)
            lp = _lp_simplex(ξ, Wt, e1)
            lp.status === :optimal && push!(vals, lp.objective)
        end
    end
    isempty(vals) && error("least-favorable critical value: no simulation succeeded")
    return quantile(vals, 1 - κ)
end

# HonestDiD .construct_Gamma: invertible matrix with first row l.
function _did_hd_gamma(l)
    T = length(l)
    jlast = findlast(!iszero, l)
    jlast === nothing && throw(ArgumentError("l_vec must not be all zeros"))
    rows = Matrix{Float64}[reshape(l, 1, :)]
    for j in 1:T
        j == jlast && continue
        e = zeros(1, T)
        e[j] = 1.0
        push!(rows, e)
    end
    return reduce(vcat, rows)
end

"""
Tester for one polyhedral piece: a function θ -> accept::Bool with all θ-invariant
quantities (least-favorable critical values, FLCI first stage) precomputed.
"""
function _did_hd_piece_tester(p::_DidHDPiece, β, Σ, npre, npost, l, α, kind, κ, flci,
                              lf_sims, rng)
    A, d = p.A, p.d
    if npost == 1
        e = zeros(npre + npost)
        e[npre + 1] = 1.0 / l[1]
        if kind === :arp
            return θ -> !_did_hd_test_nonuis(β .- e .* θ, Σ, A, d, α)
        elseif kind === :lf
            cv = _did_hd_lf_cv(nothing, A * Σ * A', κ, lf_sims, rng)
            αt = (α - κ) / (1 - κ)
            return θ -> !_did_hd_test_nonuis(β .- e .* θ, Σ, A, d, αt; lf_cv=cv)
        else
            fl = flci.optimal_vec
            Afs = vcat(fl', -fl')
            dfs = [flci.halflength, flci.halflength]
            αt = (α - κ) / (1 - κ)
            return function (θ)
                y = β .- e .* θ
                maximum(Afs * y .- dfs) > 0 && return false
                return !_did_hd_test_nonuis(y, Σ, A, d, αt; Aadd=Afs, dadd=dfs)
            end
        end
    end
    Γ = _did_hd_gamma(l)
    AG = A[:, (npre + 1):end] / Γ
    a1 = AG[:, 1]
    X = AG[:, 2:end]
    Y = A * β .- d
    ΣY = A * Σ * A'
    rows = p.rows
    if kind === :lf
        cv = _did_hd_lf_cv(X[rows, :], ΣY[rows, rows], κ, lf_sims, rng)
        hy = _DidHDHybrid(:lf, κ, cv, Float64[], NaN, Float64[])
        return θ -> !_did_hd_lp_test(Y .- a1 .* θ, X, ΣY, α, hy, rows, nothing)
    elseif kind === :flci
        fl = flci.optimal_vec
        vbar = pinv(A * A') * (A * fl)
        hlen = flci.halflength
        hy = _DidHDHybrid(:flci, κ, NaN, vbar, hlen, fl)
        vd = dot(vbar, d)
        va = dot(vbar, a1)
        return function (θ)
            dbar = [hlen - vd + (1 - va) * θ, hlen + vd - (1 - va) * θ]
            return !_did_hd_lp_test(Y .- a1 .* θ, X, ΣY, α, hy, rows, dbar)
        end
    end
    hy = _DidHDHybrid(:arp, κ, NaN, Float64[], NaN, Float64[])
    return θ -> !_did_hd_lp_test(Y .- a1 .* θ, X, ΣY, α, hy, rows, nothing)
end

# θ -> accept for the union of the pieces of Δ(M), and the first-stage FLCI.
function _did_hd_acceptor(β, Σ, npre, npost, l, α, restriction, M, kind, bias_sign,
                          monotonicity, post_only, lf_sims, κ, rng)
    flci = kind === :flci ? _did_hd_flci(Σ, M, npre, npost, l, κ) : nothing
    pieces = _did_hd_pieces(restriction, M, npre, npost, bias_sign, monotonicity,
                            post_only)
    testers = [_did_hd_piece_tester(p, β, Σ, npre, npost, l, α, kind, κ, flci,
                                    lf_sims, rng) for p in pieces]
    return (θ -> any(t -> t(θ), testers)), flci
end

# Robust confidence set for one value of M: grid inversion (+ optional refinement).
function _did_hd_conditional_ci(β, Σ, npre, npost, l, α, restriction, M, method,
                                bias_sign, monotonicity, post_only, grid_points,
                                grid_lb, grid_ub, refine, lf_sims, κ, rng)
    kind = method === :conditional ? :arp : method === :hybrid_lf ? :lf : :flci
    accept, flci = _did_hd_acceptor(β, Σ, npre, npost, l, α, restriction, M, kind,
                                    bias_sign, monotonicity, post_only, lf_sims, κ, rng)
    sdθ = sqrt(dot(l, Σ[(npre + 1):end, (npre + 1):end] * l))
    θhat = dot(l, β[(npre + 1):end])
    lo, hi = grid_lb, grid_ub
    if lo === nothing || hi === nothing
        if kind === :flci
            c = dot(flci.optimal_vec, β)
            glo, ghi = c - flci.halflength, c + flci.halflength
        else
            idp = _did_hd_pieces(restriction, M, npre, npost, bias_sign, monotonicity,
                                 post_only; for_idset=true)
            ilo, ihi = _did_hd_idset(idp, β, l, npre, npost)
            if !isfinite(ilo)
                βz = vcat(zeros(npre), β[(npre + 1):end])
                ilo, ihi = _did_hd_idset(idp, βz, l, npre, npost)
                isfinite(ilo) || ((ilo, ihi) = (θhat, θhat))
            end
            glo, ghi = ilo - 20sdθ, ihi + 20sdθ
        end
        lo === nothing && (lo = glo)
        hi === nothing && (hi = ghi)
    end
    grid = collect(range(float(lo), float(hi); length=grid_points))
    acc = map(accept, grid)
    idx = findall(acc)
    isempty(idx) && return (NaN, NaN, false)
    i1, i2 = first(idx), last(idx)
    # (the FLCI first stage bounds the hybrid set by its grid, as in HonestDiD)
    open_end = kind !== :flci && (i1 == 1 || i2 == length(grid))
    lbv, ubv = grid[i1], grid[i2]
    if refine
        i1 > 1 && (lbv = _did_hd_bisect(accept, grid[i1 - 1], grid[i1]))
        i2 < length(grid) && (ubv = _did_hd_bisect(accept, grid[i2 + 1], grid[i2]))
    end
    return (lbv, ubv, open_end)
end

# Boundary between a rejected point `out` and an accepted point `inn`.
function _did_hd_bisect(accept, out, inn; iters=40)
    for _ in 1:iters
        mid = (out + inn) / 2
        accept(mid) ? (inn = mid) : (out = mid)
        abs(out - inn) <= 1e-10 * max(1.0, abs(inn)) && break
    end
    return inn
end

# ---------------------------------------------------------------------------
# User interface
# ---------------------------------------------------------------------------

const _DID_HD_RESTRICTIONS = Dict(:smoothness => :smoothness, :sd => :smoothness,
                                  :relative_magnitudes => :relative_magnitudes,
                                  :rm => :relative_magnitudes)

function _did_hd_delta_name(restriction, bias_sign, monotonicity)
    base = restriction === :smoothness ? "DeltaSD" :
           restriction === :relative_magnitudes ? "DeltaRM" : "DeltaSDRM"
    bias_sign === :positive && return base * "PB"
    bias_sign === :negative && return base * "NB"
    monotonicity === :increasing && return base * "I"
    monotonicity === :decreasing && return base * "D"
    return base
end

"""
    honest_did(es::EventStudyEstimate; restriction=:relative_magnitudes, M=nothing,
               bound=:parallel_trends, method=:auto, target=0, l_vec=nothing,
               bias_sign=nothing, monotonicity=nothing, level=0.95,
               grid_points=1000, grid_lb=nothing, grid_ub=nothing, refine=true,
               post_moments_only=true, lf_sims=1000, rng=Random.default_rng(),
               reference=nothing) -> HonestDiDResult
    honest_did(betahat, sigma, num_pre, num_post; kwargs...) -> HonestDiDResult

Sensitivity analysis for violations of parallel trends (Rambachan and Roth, 2023):
confidence sets for a post-treatment effect that remain valid when the
post-treatment difference in trends is bounded by, or relative to, the
pre-treatment one.

Write the event-study coefficients as ``\\beta = \\tau + \\delta`` with pre- and
post-treatment blocks, where ``\\tau_{pre} = 0`` (no anticipation), ``\\tau_{post}``
are the causal effects and ``\\delta`` is the difference in trends of untreated
potential outcomes between treated and comparison units. Parallel trends is the
assumption ``\\delta_{post} = 0``, which the data cannot test. Instead of imposing it,
the analysis assumes only that ``\\delta \\in \\Delta(M)``, a set that restricts how
much the post-treatment violation can differ from what the pre-treatment
coefficients ``\\hat\\beta_{pre}`` reveal about ``\\delta_{pre}``. The target
``\\theta = l'\\tau_{post}`` is then partially identified, and the function reports,
for each ``M`` in a grid, a confidence set that covers ``\\theta`` with probability
at least `level` uniformly over ``\\Delta(M)``. ``M = 0`` is the most optimistic
restriction; larger values allow larger violations and give wider sets, and the
breakdown value, the smallest ``M`` at which the set contains zero, summarizes the
robustness of a significant finding (see also [`honest_breakdown`](@ref)). The
approach builds on bounded-variation assumptions of Manski and Pepper (2018).

**Restrictions** (`restriction`, `bound`, `bias_sign`, `monotonicity`):

- `:relative_magnitudes` (``\\Delta^{RM}(\\bar M)``, default): each post-treatment
  change in the trend difference between *consecutive* periods is at most ``\\bar M``
  times the largest such change before treatment,
  ``|\\delta_{t+1} - \\delta_t| \\le \\bar M \\max_{s < 0} |\\delta_{s+1} - \\delta_s|``
  for ``t \\ge 0``. The restriction bounds period-to-period changes, not the levels of
  ``\\delta``: violations can accumulate over post-treatment periods, so sets for
  later horizons widen quickly. ``\\bar M = 1`` allows post-treatment shocks as large
  as the worst pre-treatment one. With `bound = :linear_trend` the bound applies to
  deviations from a linear extrapolation of the pre-trend (second differences,
  ``\\Delta^{SDRM}``).
- `:smoothness` (``\\Delta^{SD}(M)``): the slope of the trend difference changes by
  at most ``M`` per period,
  ``|(\\delta_{t+1} - \\delta_t) - (\\delta_t - \\delta_{t-1})| \\le M``; ``M = 0``
  allows exactly linear differential trends extrapolated from the pre-period, and
  ``M`` is measured in units of the outcome per period squared.
- `bias_sign = :positive` or `:negative` adds ``\\delta_{post} \\ge 0`` or ``\\le 0``;
  `monotonicity = :increasing` or `:decreasing` adds a monotone ``\\delta``. At most
  one of the two can be given.

**Inference** (`method`). The confidence sets are based on the normal approximation
``\\hat\\beta \\sim N(\\beta, \\Sigma)`` with ``\\Sigma`` the estimated covariance.
`:flci` (default for `:smoothness` without sign or shape restrictions) is the
optimal fixed-length confidence interval of Armstrong and Kolesár (2018), which has
near-optimal length under ``\\Delta^{SD}``. `:conditional` inverts the conditional
moment-inequality test of Andrews, Roth and Pakes (2023) on a grid of ``\\theta``
values. `:hybrid_lf` (default for relative magnitudes) combines it with a
least-favorable first stage of size ``\\alpha/10`` whose critical value is simulated
with `lf_sims` draws from `rng`, and `:hybrid_flci` (default for `:smoothness` with
sign or shape restrictions) with an FLCI first stage of size ``\\alpha/10``. Robust
sets from grid inversion are reported as the smallest interval containing the
accepted ``\\theta``; with `refine = true` its endpoints are located by bisection
between grid points. Results are validated against the R package HonestDiD.

**Requirements and practice.** The event study must have *consecutive* relative
periods, a single reference period normalized to zero, and at least one pre- and one
post-treatment coefficient. Coefficients must not pool several periods: an
[`event_study`](@ref) with the default `endpoints = :bin` and a window narrower than
the data has binned endpoint coefficients and is rejected; re-estimate with
`endpoints = :trim` or a window covering all relative periods. Suitable inputs are
[`event_study`](@ref) (TWFE with a single cohort, or Sun–Abraham),
[`did_sun_abraham`](@ref), [`aggregate_att`](@ref)`(cs, :dynamic)` from
[`did_callaway_santanna`](@ref) with `base_period = :universal`,
[`did_etwfe`](@ref) with `control_group = :never_treated`, and
[`did_multiplegt_dyn`](@ref). The imputation estimator is not supported, because its
pre-treatment coefficients are not measured relative to a period adjacent to
treatment (Roth, 2026). Choose the restriction and the range of ``M`` on economic
grounds before looking at the results, report the whole sensitivity curve rather
than a single ``M``, and remember that the pre-period coefficients enter both the
restriction and the estimate: with few or noisy pre-periods, relative-magnitude
bounds are imprecise.

# Arguments
- `es::EventStudyEstimate`: an event study as described above. Coefficients before
  the reference period are "pre", those after it "post".
- `betahat::AbstractVector`, `sigma::AbstractMatrix`, `num_pre::Integer`,
  `num_post::Integer`: the raw interface, with `betahat` ordered as the pre periods
  (oldest first) followed by the post periods, the reference period excluded, and
  `sigma` its covariance matrix.

# Keywords
- `restriction::Symbol = :relative_magnitudes`: `:relative_magnitudes` or
  `:smoothness`.
- `M = nothing`: vector of ``M`` (or ``\\bar M``) values; by default ten values from
  0 to 2 for relative magnitudes, and for smoothness from 0 to an upper bound based
  on the pre-period second differences (as in HonestDiD).
- `bound::Symbol = :parallel_trends`: `:parallel_trends` or `:linear_trend` (relative
  magnitudes of deviations from a linear pre-trend; needs two pre-periods).
- `method::Symbol = :auto`: `:flci`, `:conditional`, `:hybrid_flci`, `:hybrid_lf`,
  or `:auto` for the defaults above.
- `target = 0`: event time of the target effect, or a collection of event times
  whose effects are averaged with equal weights.
- `l_vec = nothing`: one weight per post-treatment period, in order; overrides
  `target`.
- `bias_sign = nothing`, `monotonicity = nothing`: optional sign or shape
  restriction, as above.
- `level::Real = 0.95`: confidence level (``\\alpha = 1 -`` `level`).
- `grid_points::Integer = 1000`, `grid_lb = nothing`, `grid_ub = nothing`: grid of
  ``\\theta`` values for the conditional and hybrid methods. The default grid spans
  the identified set at ``\\hat\\beta`` widened by 20 standard errors of
  ``l'\\hat\\beta_{post}`` (the FLCI of level ``1 - \\alpha/10`` for `:hybrid_flci`);
  a warning is issued if an accepted set reaches its end.
- `refine::Bool = true`: locate the endpoints by bisection between grid points
  (`false` returns grid points, as HonestDiD does).
- `post_moments_only::Bool = true`: as in HonestDiD, with several post periods the
  conditional test uses only the moment inequalities that involve post-period
  coefficients.
- `lf_sims::Integer = 1000`: simulation draws for the least-favorable critical
  value.
- `rng::AbstractRNG = Random.default_rng()`: generator for those draws.
- `reference = nothing`: overrides the normalized reference period stored in `es`.

# Returns
- [`HonestDiDResult`](@ref): robust confidence sets per ``M``, the conventional
  interval, and the smallest grid value whose set contains zero.

# Examples
```julia
using DrSnow, CSV, DataFrames, StableRNGs
file = joinpath(pkgdir(DrSnow), "test", "validation", "did", "mpdta.csv")
mpdta = CSV.read(file, DataFrame)
# single cohort; endpoints=:trim keeps every coefficient a single relative period
sub = filter(r -> r.first_treat in (0, 2006), mpdta)
es = event_study(sub, :lemp, :d, :countyreal, :year; estimator=:twfe, max_pre=3,
                 max_post=1, endpoints=:trim)
rm = honest_did(es; restriction=:relative_magnitudes, M=0:0.5:2, rng=StableRNG(1))
confint(rm)
sd = honest_did(es; restriction=:smoothness, M=[0, 0.01, 0.02], target=0:1)
# staggered adoption: Sun–Abraham event study
sa = did_sun_abraham(mpdta, :lemp, FirstTreated(:first_treat), :countyreal, :year)
honest_did(sa; M=0:0.5:1, rng=StableRNG(2))
```

# References
- Rambachan, A., & Roth, J. (2023). A more credible approach to parallel trends.
  *Review of Economic Studies*, 90(5), 2555–2591.
- Andrews, I., Roth, J., & Pakes, A. (2023). Inference for linear conditional moment
  inequalities. *Review of Economic Studies*, 90(6), 2763–2791.
- Armstrong, T. B., & Kolesár, M. (2018). Optimal inference in a class of regression
  models. *Econometrica*, 86(2), 655–683.
- Manski, C. F., & Pepper, J. V. (2018). How do right-to-carry laws affect crime
  rates? Coping with ambiguity using bounded-variation assumptions. *Review of
  Economics and Statistics*, 100(2), 232–244.
- Roth, J. (2026). Interpreting event-studies from recent difference-in-differences
  methods. *The Japanese Economic Review*, 77(2), 275–288.
- Rambachan, A., & Roth, J. (2026). HonestDiD: Robust inference in
  difference-in-differences and event study designs. R package version 0.2.8.
"""
function honest_did(es::EventStudyEstimate; target=0, l_vec=nothing, reference=nothing,
                    kwargs...)
    β, Σ, npre, npost, pre_p, post_p = _did_hd_from_es(es, reference)
    l = _did_hd_lvec(target, l_vec, post_p)
    r = honest_did(β, Σ, npre, npost; l_vec=l, kwargs...)
    return HonestDiDResult(r.restriction, r.delta, r.method, r.M, r.lb, r.ub, r.estimate,
                           r.original, r.breakdown, r.l_vec, post_p, pre_p, r.level,
                           merge(r.details, (event_study=es,)))
end

function honest_did(betahat::AbstractVector, sigma::AbstractMatrix, num_pre::Integer,
                    num_post::Integer; restriction::Symbol=:relative_magnitudes,
                    M=nothing, bound::Symbol=:parallel_trends, method::Symbol=:auto,
                    l_vec=nothing, bias_sign::Union{Nothing,Symbol}=nothing,
                    monotonicity::Union{Nothing,Symbol}=nothing, level::Real=0.95,
                    grid_points::Integer=1000, grid_lb=nothing, grid_ub=nothing,
                    refine::Bool=true, post_moments_only::Bool=true,
                    lf_sims::Integer=1000, rng::AbstractRNG=Random.default_rng())
    β, Σ, l = _did_hd_check_inputs(betahat, sigma, num_pre, num_post, l_vec)
    npre, npost = Int(num_pre), Int(num_post)
    set = _did_hd_settings(restriction, bound, method, bias_sign, monotonicity, level,
                           npre, npost)
    restriction, method = set.restriction, set.method
    grid_points >= 2 || throw(ArgumentError("grid_points must be at least 2"))
    α = 1 - level
    Mvec = M === nothing ? _did_hd_default_M(β, Σ, npre, restriction) :
           Float64.(collect(M))
    isempty(Mvec) && throw(ArgumentError("M must not be empty"))
    any(<(0), Mvec) && throw(ArgumentError("M values must be ≥ 0"))
    κ = α / 10
    lb = similar(Mvec)
    ub = similar(Mvec)
    open_any = false
    flcis = Any[]
    for (k, m) in enumerate(Mvec)
        if method === :flci
            f = _did_hd_flci(Σ, m, npre, npost, l, α)
            c = dot(f.optimal_vec, β)
            lb[k], ub[k] = c - f.halflength, c + f.halflength
            push!(flcis, f)
        else
            lb[k], ub[k], op = _did_hd_conditional_ci(β, Σ, npre, npost, l, α,
                restriction, m, method, bias_sign, monotonicity, post_moments_only,
                Int(grid_points), grid_lb, grid_ub, refine, Int(lf_sims), κ, rng)
            open_any |= op
        end
    end
    open_any && @warn "honest_did: the robust confidence set reaches the end of the " *
                      "θ grid for some M; widen grid_lb/grid_ub"
    θhat = dot(l, β[(npre + 1):end])
    se = sqrt(dot(l, Σ[(npre + 1):end, (npre + 1):end] * l))
    cv = critical_value(level)
    brk = NaN
    for k in sortperm(Mvec)
        if !isnan(lb[k]) && lb[k] <= 0 <= ub[k]
            brk = Mvec[k]
            break
        end
    end
    return HonestDiDResult(restriction, _did_hd_delta_name(restriction, bias_sign,
                                                           monotonicity),
                           method, Mvec, lb, ub, θhat, (θhat - cv * se, θhat + cv * se),
                           brk, l, collect(0:(npost - 1)), collect((-npre - 1):-2),
                           float(level),
                           (betahat=β, sigma=Σ, num_pre=npre, num_post=npost,
                            bias_sign=bias_sign, monotonicity=monotonicity,
                            grid_points=Int(grid_points), refine=refine,
                            post_moments_only=post_moments_only, flci=flcis))
end

"""
    honest_breakdown(es::EventStudyEstimate; restriction=:relative_magnitudes,
                     bound=:parallel_trends, method=:auto, target=0, l_vec=nothing,
                     reference=nothing, bias_sign=nothing, monotonicity=nothing,
                     level=0.95, post_moments_only=true, lf_sims=1000,
                     rng=Random.default_rng(), tol=1e-3, upper=nothing) -> Float64
    honest_breakdown(betahat, sigma, num_pre, num_post; kwargs...) -> Float64

Breakdown value of a Rambachan–Roth sensitivity analysis: the smallest ``M`` (or
``\\bar M``) at which the robust confidence set for the target includes zero.

The breakdown value answers the question of how large a violation of parallel
trends, in the units of the chosen restriction, would have to be before a
significant effect could no longer be distinguished from zero at the given
`level`. For relative magnitudes, ``\\bar M = 1.5`` means that the conclusion
survives post-treatment changes in the trend difference up to one and a half times
the largest pre-treatment change; for smoothness, ``M`` is the permitted change in
the slope of the differential trend per period, in outcome units (Rambachan and
Roth, 2023). Unlike the `breakdown` field of [`HonestDiDResult`](@ref), which is
restricted to the grid of ``M`` values, the value is found by bisection, to relative
tolerance `tol`, on the test of ``H_0: \\theta = 0``, assuming that rejection is
monotone in ``M`` (the restriction sets are nested). Its interpretation is the same
as that of the sensitivity curve: a large breakdown value is reassuring only if
violations of that size are implausible on substantive grounds, and a small one
does not show that the effect is zero. The requirements on `es` are those of
[`honest_did`](@ref): consecutive relative periods, a single reference period, and
no binned endpoint coefficients (use `endpoints = :trim` in
[`event_study`](@ref)).

# Arguments
- `es::EventStudyEstimate`: an event study, as for [`honest_did`](@ref).
- `betahat`, `sigma`, `num_pre`, `num_post`: the raw interface of
  [`honest_did`](@ref).

# Keywords
- `restriction::Symbol = :relative_magnitudes`: `:relative_magnitudes` or
  `:smoothness`.
- `bound::Symbol = :parallel_trends`: `:parallel_trends`, or `:linear_trend` for
  relative magnitudes of deviations from a linear pre-trend.
- `method::Symbol = :auto`: inference method, as in [`honest_did`](@ref).
- `target = 0`: event time(s) of the target effect (equal-weighted average of
  several).
- `l_vec = nothing`: explicit weights on the post-treatment periods; overrides
  `target`.
- `reference = nothing`: overrides the reference period stored in `es`.
- `bias_sign = nothing`, `monotonicity = nothing`: sign or shape restriction on
  ``\\delta``, as in [`honest_did`](@ref).
- `level::Real = 0.95`: confidence level of the robust sets.
- `post_moments_only::Bool = true`: use only moments involving post-period
  coefficients in the conditional test.
- `lf_sims::Integer = 1000`, `rng::AbstractRNG = Random.default_rng()`: simulation
  draws and generator for the least-favorable critical value (hybrid method).
- `tol::Real = 1e-3`: relative tolerance of the bisection.
- `upper = nothing`: largest ``M`` searched; by default the search starts at the
  largest default grid value and doubles it up to twenty times.

# Returns
- `Float64`: the breakdown value; `0.0` when zero is not rejected even at ``M = 0``,
  and `Inf` when zero is rejected for every ``M`` up to `upper` (or the largest value
  searched).

# Examples
```julia
using DrSnow, CSV, DataFrames, StableRNGs
file = joinpath(pkgdir(DrSnow), "test", "validation", "did", "mpdta.csv")
mpdta = CSV.read(file, DataFrame)
sub = filter(r -> r.first_treat in (0, 2006), mpdta)
es = event_study(sub, :lemp, :d, :countyreal, :year; estimator=:twfe, max_pre=3,
                 max_post=1, endpoints=:trim)
honest_breakdown(es; restriction=:relative_magnitudes, rng=StableRNG(1))  # M̄
honest_breakdown(es; restriction=:smoothness, method=:flci)               # M
```

# References
- Rambachan, A., & Roth, J. (2023). A more credible approach to parallel trends.
  *Review of Economic Studies*, 90(5), 2555–2591.
- Rambachan, A., & Roth, J. (2026). HonestDiD: Robust inference in
  difference-in-differences and event study designs. R package version 0.2.8.
"""
function honest_breakdown(es::EventStudyEstimate; target=0, l_vec=nothing,
                          reference=nothing, kwargs...)
    β, Σ, npre, npost, _, post_p = _did_hd_from_es(es, reference)
    l = _did_hd_lvec(target, l_vec, post_p)
    return honest_breakdown(β, Σ, npre, npost; l_vec=l, kwargs...)
end

function honest_breakdown(betahat::AbstractVector, sigma::AbstractMatrix,
                          num_pre::Integer, num_post::Integer;
                          restriction::Symbol=:relative_magnitudes,
                          bound::Symbol=:parallel_trends, method::Symbol=:auto,
                          l_vec=nothing, bias_sign::Union{Nothing,Symbol}=nothing,
                          monotonicity::Union{Nothing,Symbol}=nothing, level::Real=0.95,
                          post_moments_only::Bool=true, lf_sims::Integer=1000,
                          rng::AbstractRNG=Random.default_rng(), tol::Real=1e-3,
                          upper=nothing)
    β, Σ, l = _did_hd_check_inputs(betahat, sigma, num_pre, num_post, l_vec)
    npre, npost = Int(num_pre), Int(num_post)
    set = _did_hd_settings(restriction, bound, method, bias_sign, monotonicity, level,
                           npre, npost)
    α = 1 - level
    κ = α / 10
    function inset(m)
        if set.method === :flci
            f = _did_hd_flci(Σ, m, npre, npost, l, α)
            return abs(dot(f.optimal_vec, β)) <= f.halflength
        end
        kind = set.method === :conditional ? :arp :
               set.method === :hybrid_lf ? :lf : :flci
        acc, _ = _did_hd_acceptor(β, Σ, npre, npost, l, α, set.restriction, m, kind,
                                  bias_sign, monotonicity, post_moments_only,
                                  Int(lf_sims), κ, rng)
        return acc(0.0)
    end
    inset(0.0) && return 0.0
    hi = upper === nothing ?
         maximum(_did_hd_default_M(β, Σ, npre, set.restriction)) : float(upper)
    hi > 0 || (hi = 1.0)
    if upper === nothing
        k = 0
        while !inset(hi)
            hi *= 2
            k += 1
            k > 20 && return Inf
        end
    else
        inset(hi) || return Inf
    end
    lo = 0.0
    while hi - lo > tol * max(hi, 1e-8)
        mid = (lo + hi) / 2
        inset(mid) ? (hi = mid) : (lo = mid)
    end
    return hi
end

function _did_hd_settings(restriction, bound, method, bias_sign, monotonicity, level,
                          npre, npost)
    haskey(_DID_HD_RESTRICTIONS, restriction) || throw(ArgumentError(
        "restriction must be :relative_magnitudes or :smoothness"))
    restriction = _DID_HD_RESTRICTIONS[restriction]
    bound in (:parallel_trends, :linear_trend) || throw(ArgumentError(
        "bound must be :parallel_trends or :linear_trend"))
    if bound === :linear_trend
        restriction === :relative_magnitudes || throw(ArgumentError(
            "bound = :linear_trend applies to restriction = :relative_magnitudes"))
        restriction = :linear_trend
    end
    (bias_sign !== nothing && monotonicity !== nothing) && throw(ArgumentError(
        "specify either a sign restriction (bias_sign) or a shape restriction " *
        "(monotonicity), not both"))
    bias_sign in (nothing, :positive, :negative) ||
        throw(ArgumentError("bias_sign must be :positive or :negative"))
    monotonicity in (nothing, :increasing, :decreasing) ||
        throw(ArgumentError("monotonicity must be :increasing or :decreasing"))
    0 < level < 1 || throw(ArgumentError("level must be in (0, 1)"))
    npre >= 1 || throw(ArgumentError("need at least one pre-treatment coefficient"))
    npost >= 1 || throw(ArgumentError("need at least one post-treatment coefficient"))
    restriction === :linear_trend && npre < 2 && throw(ArgumentError(
        "bound = :linear_trend needs at least 2 pre-treatment coefficients"))
    restricted = bias_sign !== nothing || monotonicity !== nothing
    if method === :auto
        method = restriction === :smoothness ? (restricted ? :hybrid_flci : :flci) :
                 :hybrid_lf
    end
    method in (:flci, :conditional, :hybrid_flci, :hybrid_lf) || throw(ArgumentError(
        "method must be :flci, :conditional, :hybrid_flci or :hybrid_lf"))
    if method in (:flci, :hybrid_flci) && restriction !== :smoothness
        throw(ArgumentError("method = $(repr(method)) is only available for " *
                            "restriction = :smoothness"))
    end
    method === :flci && restricted && @warn "honest_did: method = :flci ignores the " *
        "sign/shape restriction; use :hybrid_flci or :conditional to exploit it"
    return (restriction=restriction, method=method)
end

function _did_hd_check_inputs(betahat, sigma, npre, npost, l_vec)
    β = Float64.(collect(betahat))
    Σ = Matrix{Float64}(sigma)
    size(Σ) == (length(β), length(β)) || throw(DimensionMismatch(
        "sigma must be $(length(β)) × $(length(β))"))
    npre + npost == length(β) || throw(DimensionMismatch(
        "num_pre + num_post must equal length(betahat)"))
    all(isfinite, β) && all(isfinite, Σ) ||
        throw(ArgumentError("betahat and sigma must be finite"))
    maximum(abs, Σ .- Σ') <= 1e-8 * max(1.0, maximum(abs, Σ)) ||
        throw(ArgumentError("sigma must be symmetric"))
    Σ = (Σ .+ Σ') ./ 2
    minimum(eigvals(Symmetric(Σ))) >= -1e-10 * max(1.0, maximum(abs, Σ)) ||
        throw(ArgumentError("sigma must be positive semi-definite"))
    l = l_vec === nothing ? [1.0; zeros(npost - 1)] : Float64.(collect(l_vec))
    length(l) == npost || throw(DimensionMismatch("l_vec must have num_post elements"))
    any(!iszero, l) || throw(ArgumentError("l_vec must not be all zeros"))
    return β, Σ, l
end

function _did_hd_default_M(β, Σ, npre, restriction)
    restriction === :smoothness || return collect(range(0, 2; length=10))
    npre == 1 && return collect(range(0, sqrt(Σ[1, 1]); length=10))
    A = _did_hd_A_SD(npre, 0)
    diffs = A * β[1:npre]
    se = sqrt.(max.(diag(A * Σ[1:npre, 1:npre] * A'), 0.0))
    ubound = maximum(diffs .+ quantile(Normal(), 0.95) .* se)
    return collect(range(0, max(ubound, 0.0); length=10))
end

function _did_hd_from_es(es::EventStudyEstimate, reference)
    rp = es.rel_periods
    binned = get(es.details, :binned, (false, false))
    any(binned) && throw(ArgumentError(
        "honest_did: the event study has binned endpoint coefficients, which pool " *
        "several relative periods; re-estimate with endpoints=:trim or a window " *
        "covering all periods"))
    ref = if reference !== nothing
        Int(reference)
    elseif length(es.reference) == 1
        only(es.reference)
    elseif isempty(es.reference)
        note = occursin("Callaway", es.method) ?
               " For Callaway–Sant'Anna use base_period = :universal (the varying " *
               "base period makes pre-treatment estimates short differences)." :
               " Sensitivity analysis needs pre- and post-treatment coefficients " *
               "measured relative to a common reference period (the imputation " *
               "estimator does not provide them)."
        throw(ArgumentError("honest_did: the event study has no reference period " *
                            "normalized to zero." * note))
    else
        throw(ArgumentError("honest_did: the event study normalizes several periods " *
                            "($(join(es.reference, ", "))); pass `reference`"))
    end
    ref in rp && throw(ArgumentError(
        "reference period $ref has an estimated coefficient in the event study"))
    allp = sort(vcat(rp, ref))
    all(diff(allp) .== 1) || throw(ArgumentError(
        "honest_did needs consecutive relative periods (with the reference period " *
        "$ref); found $(join(rp, ", "))"))
    ord = sortperm(rp)
    rps = rp[ord]
    pre = findall(<(ref), rps)
    post = findall(>(ref), rps)
    isempty(pre) && throw(ArgumentError("honest_did: no pre-treatment coefficients"))
    isempty(post) && throw(ArgumentError("honest_did: no post-treatment coefficients"))
    β = es.coef[ord]
    Σ = es.vcov[ord, ord]
    return β, Σ, length(pre), length(post), rps[pre], rps[post]
end

function _did_hd_lvec(target, l_vec, post_p)
    if l_vec !== nothing
        length(l_vec) == length(post_p) || throw(DimensionMismatch(
            "l_vec must have one weight per post period ($(join(post_p, ", ")))"))
        return Float64.(collect(l_vec))
    end
    want = target isa Integer ? [Int(target)] : Int.(collect(target))
    isempty(want) && throw(ArgumentError("target must not be empty"))
    l = zeros(length(post_p))
    for e in want
        k = findfirst(==(e), post_p)
        k === nothing && throw(ArgumentError(
            "target event time $e is not a post-reference period of the event study " *
            "($(join(post_p, ", ")))"))
        l[k] += 1 / length(want)
    end
    return l
end
