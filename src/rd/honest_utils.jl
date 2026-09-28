# Internal building blocks for honest (bias-aware) RD inference.
#
# The routines mirror the R package `RDHonest` (Kolesár, version 1.0.2.9000, functions
# `NPReg`, `sigmaNN`, `PrelimVar`, `IKBW`, `ROTBW`, `MROT`, `OptBW`, `CVb`) step by step
# so that DrSnow reproduces its estimates, bandwidths and confidence intervals.

# ---------------------------------------------------------------------------------------
# Data container (running variable centred at the cutoff and sorted)
# ---------------------------------------------------------------------------------------

mutable struct _RDHonestData
    class::Symbol                         # :srd, :frd or :ip (inference at a point)
    X::Vector{Float64}                    # running variable minus cutoff, sorted
    Y::Matrix{Float64}                    # outcome (and treatment for :frd)
    Y_unadj::Union{Nothing,Matrix{Float64}}
    w::Vector{Float64}                    # observation weights
    p::BitVector                          # X >= 0
    m::BitVector                          # X < 0
    covs::Union{Nothing,Matrix{Float64}}
    cluster::Union{Nothing,Vector{Int}}   # cluster codes
    sigma2::Union{Nothing,Matrix{Float64}}  # supplied / preliminary variances
    rho::Union{Nothing,Vector{Float64}}   # Moulton intra-cluster correlation
end

function _rd_h_copy(d::_RDHonestData; covs=d.covs, Y=d.Y, Y_unadj=d.Y_unadj,
                    sigma2=d.sigma2, rho=d.rho, class=d.class)
    return _RDHonestData(class, d.X, Y, Y_unadj, d.w, d.p, d.m, covs, d.cluster, sigma2,
                         rho)
end

_rd_h_ny(d::_RDHonestData) = size(d.Y, 2)

# ---------------------------------------------------------------------------------------
# Kernels, critical values, one-dimensional optimisation
# ---------------------------------------------------------------------------------------

function _rd_h_kernel(k)
    s = lowercase(string(k))
    s in ("tri", "triangular") && return :triangular
    s in ("epa", "epanechnikov") && return :epanechnikov
    s in ("uni", "uniform") && return :uniform
    throw(ArgumentError("kernel must be :triangular, :epanechnikov or :uniform " *
                        "(got $(repr(k)))"))
end

# Interior kernels of RDHonest (`EqKern(kernel, boundary = FALSE, order = 0)`).
function _rd_h_kern(kernel::Symbol, u::Float64)
    inside = (u <= 1) & (u >= -1)
    kernel === :triangular && return (1 - abs(u)) * inside
    kernel === :uniform && return inside / 2
    return (3 / 4) * (1 - u^2) * inside
end

"""
    _rd_cvb(B, alpha) -> Float64

Critical value `cv` such that `P(|Z + B| ≤ cv) = 1 - alpha` for `Z ~ N(0, 1)`: the
`1 - alpha` quantile of the folded non-central normal `|N(B, 1)|` (RDHonest `CVb`,
which uses `sqrt(qchisq(1 - alpha, 1, ncp = B^2))` for `B < 10` and `B + z_{1-alpha}`
otherwise). Solved here to machine precision.
"""
function _rd_cvb(B::Real, alpha::Real)
    (0 < alpha < 1) || throw(ArgumentError("alpha must be in (0, 1)"))
    B = Float64(B)
    (isfinite(B) && B >= 0) || throw(ArgumentError("maximum bias ratio must be a " *
                                                   "non-negative number (got $B)"))
    B >= 10 && return B + quantile(Normal(), 1 - alpha)
    target = 1 - alpha
    cover(c) = cdf(Normal(), c - B) - cdf(Normal(), -c - B)
    lo = B + quantile(Normal(), 1 - alpha)
    hi = B + quantile(Normal(), 1 - alpha / 2)
    lo = max(lo - 1e-8, 0.0)
    for _ in 1:200
        mid = (lo + hi) / 2
        (mid == lo || mid == hi) && break
        cover(mid) < target ? (lo = mid) : (hi = mid)
    end
    return (lo + hi) / 2
end

# Two-sided p-value of H0: θ = 0 for an estimator with maximum bias `B` standard errors.
_rd_h_pvalue(t::Real, B::Real) = cdf(Normal(), B - abs(t)) + cdf(Normal(), -B - abs(t))

"""
Brent's one-dimensional minimiser, a transcription of `Brent_fmin` used by R's
`optimize` (so that the selected bandwidths agree with R to its tolerance).
"""
function _rd_brent_fmin(f, ax::Float64, bx::Float64, tol::Float64)
    c = (3 - sqrt(5)) * 0.5
    eps = sqrt(Base.eps(Float64))
    a, b = ax, bx
    v = a + c * (b - a)
    w = v
    x = v
    d = 0.0
    e = 0.0
    fcheck(t) = (y = f(t); isfinite(y) ? y : floatmax(Float64))
    fx = fcheck(x)
    fv = fx
    fw = fx
    tol3 = tol / 3
    while true
        xm = (a + b) * 0.5
        tol1 = eps * abs(x) + tol3
        t2 = tol1 * 2
        abs(x - xm) <= t2 - (b - a) * 0.5 && break
        p = 0.0
        q = 0.0
        r = 0.0
        if abs(e) > tol1
            r = (x - w) * (fx - fv)
            q = (x - v) * (fx - fw)
            p = (x - v) * q - (x - w) * r
            q = (q - r) * 2
            if q > 0
                p = -p
            else
                q = -q
            end
            r = e
            e = d
        end
        if abs(p) >= abs(q * 0.5 * r) || p <= q * (a - x) || p >= q * (b - x)
            e = x < xm ? b - x : a - x
            d = c * e
        else
            d = p / q
            u = x + d
            if u - a < t2 || b - u < t2
                d = tol1
                x >= xm && (d = -d)
            end
        end
        u = abs(d) >= tol1 ? x + d : (d > 0 ? x + tol1 : x - tol1)
        fu = fcheck(u)
        if fu <= fx
            u < x ? (b = x) : (a = x)
            v = w; w = x; x = u
            fv = fw; fw = fx; fx = fu
        else
            u < x ? (a = u) : (b = u)
            if fu <= fw || w == x
                v = w; fv = fw
                w = u; fw = fu
            elseif fu <= fv || v == x || v == w
                v = u; fv = fu
            end
        end
    end
    return x
end

# Modified golden-section search over a sorted grid (RDHonest `gss`), for criteria that
# are piecewise constant in the bandwidth (uniform kernel).
function _rd_h_gss(f, xs::AbstractVector{Float64})
    gr = (sqrt(5) + 1) / 2
    a = 1
    b = length(xs)
    c = round(Int, b - (b - a) / gr)
    d = round(Int, a + (b - a) / gr)
    while b - a > 100
        if f(xs[c]) < f(xs[d])
            b = d
        else
            a = c
        end
        c = round(Int, b - (b - a) / gr)
        d = round(Int, a + (b - a) / gr)
    end
    supp = xs[a:b]
    vals = [f(s) for s in supp]
    return supp[argmin(vals)]
end

# Root of a continuous function with a sign change on [lo, hi] (bisection to machine
# precision).
function _rd_h_bisect(f, lo::Float64, hi::Float64)
    flo = f(lo)
    fhi = f(hi)
    sign(flo) == sign(fhi) && flo != 0 && fhi != 0 &&
        error("internal error: no sign change in bisection")
    flo == 0 && return lo
    fhi == 0 && return hi
    for _ in 1:300
        mid = (lo + hi) / 2
        (mid == lo || mid == hi) && break
        fm = f(mid)
        fm == 0 && return mid
        if sign(fm) == sign(flo)
            lo, flo = mid, fm
        else
            hi = mid
        end
    end
    return (lo + hi) / 2
end

# Root of an increasing-in-magnitude function on (0, ∞) (RDHonest `FindZero` with
# `negative = FALSE`): expand the bracket, then bisect.
function _rd_h_find_zero_pos(f; ival::Float64=1.1)
    lo = min(1 / ival, 1e-3)
    it = 0
    while sign(f(ival)) == sign(f(lo))
        ival *= 2
        lo = min(1 / ival, 1e-3)
        it += 1
        it > 2000 && error("internal error: FindZero did not bracket a root")
    end
    return _rd_h_bisect(f, lo, ival)
end

# ---------------------------------------------------------------------------------------
# Least squares helpers
# ---------------------------------------------------------------------------------------

# Columns of `A` that are linearly independent of the preceding accepted columns, with
# the relative tolerance 1e-7 used by R's `lm` (LINPACK dqrdc2 with limited pivoting).
function _rd_h_indep_cols(A::AbstractMatrix; tol::Float64=1e-7)
    k = size(A, 2)
    keep = falses(k)
    Qs = Vector{Vector{Float64}}()
    for j in 1:k
        z = Vector{Float64}(A[:, j])
        nz = norm(z)
        nz > 0 || continue
        r = copy(z)
        for _ in 1:2
            for q in Qs
                r .-= dot(q, r) .* q
            end
        end
        nr = norm(r)
        if nr >= tol * nz
            keep[j] = true
            push!(Qs, r ./ nr)
        end
    end
    return keep
end

# OLS / WLS coefficients (no rank check beyond `_rd_h_indep_cols`); `nothing` when the
# design is rank deficient.
function _rd_h_ols(X::AbstractMatrix, y::AbstractVecOrMat;
                   w::Union{Nothing,AbstractVector}=nothing)
    if w === nothing
        Xs, ys = X, y
    else
        ok = w .> 0
        sw = sqrt.(w[ok])
        Xs = X[ok, :] .* sw
        ys = y isa AbstractVector ? y[ok] .* sw : y[ok, :] .* sw
    end
    size(Xs, 1) >= size(Xs, 2) || return nothing
    all(_rd_h_indep_cols(Xs)) || return nothing
    F = qr(Xs)
    return F.R \ (Matrix(F.Q)' * ys)
end

# ---------------------------------------------------------------------------------------
# Nearest-neighbour variance (RDHonest `sigmaNN`), X sorted
# ---------------------------------------------------------------------------------------

function _rd_h_sigma_nn(X::AbstractVector{Float64}, Y::AbstractMatrix{Float64}, J::Int,
                        w::AbstractVector{Float64})
    n = length(X)
    ny = size(Y, 2)
    out = zeros(n, ny^2)
    n >= 2 || throw(ArgumentError("the nearest-neighbour variance estimator needs at " *
                                  "least two observations on each side of the cutoff " *
                                  "within the bandwidth"))
    J = min(J, n - 1)
    dists = Float64[]
    u = zeros(ny)
    for k in 1:n
        empty!(dists)
        for i in max(k - J, 1):(k - 1)
            push!(dists, abs(X[i] - X[k]))
        end
        for i in (k + 1):min(k + J, n)
            push!(dists, abs(X[i] - X[k]))
        end
        sort!(dists)
        dJ = dists[J]
        lo = k
        while lo > 1 && abs(X[lo - 1] - X[k]) <= dJ
            lo -= 1
        end
        hi = k
        while hi < n && abs(X[hi + 1] - X[k]) <= dJ
            hi += 1
        end
        Jk = 0.0
        fill!(u, 0.0)
        for i in lo:hi
            i == k && continue
            Jk += w[i]
            for c in 1:ny
                u[c] += w[i] * Y[i, c]
            end
        end
        f = Jk / (Jk + w[k])
        for c in 1:ny
            u[c] = Y[k, c] - u[c] / Jk
        end
        # column-major vec of outer(u, u), as `as.vector(outer(u, u))` in R
        idx = 0
        for b in 1:ny, a in 1:ny
            idx += 1
            out[k, idx] = f * u[a] * u[b]
        end
    end
    return out
end

# ---------------------------------------------------------------------------------------
# Local polynomial regression (RDHonest `NPReg`)
# ---------------------------------------------------------------------------------------

"""
Local polynomial fit of order `order` with bandwidth `h`: RD jump (classes :srd, :frd,
with covariates if present) or intercept (:ip). Returns the estimate, its standard
error, the linear estimation weights `est_w` (estimate = Σ est_w Y), the variance
components `V` (length `ny^2`), effective observations and, for fuzzy designs, the first
stage and reduced form.
"""
function _rd_h_npreg(d::_RDHonestData, h::Float64, kernel::Symbol; order::Int=1,
                     se_method::Symbol=:nn, J::Int=3, warn_collinear::Bool=false)
    X = d.X
    n = length(X)
    ny = _rd_h_ny(d)
    W = h <= 0 ? zeros(n) : [_rd_h_kern(kernel, X[i] / h) * d.w[i] for i in 1:n]
    P = _rd_vander(X, order)
    if d.class === :ip
        Z = P
        Lz = order + 1
    else
        Z = hcat((X .>= 0) .* P, P)
        Lz = 2 * (order + 1)
        d.covs === nothing || (Z = hcat(Z, d.covs))
    end
    ok = W .!= 0
    sw = sqrt.(W[ok])
    na_result = (estimate=0.0, se=NaN, est_w=zeros(n), sigma2=fill(NaN, n, ny^2),
                 eff_obs=0.0, fs=NaN, rf=NaN, V=fill(NaN, ny^2), Yadj=d.Y,
                 beta=zeros(0, ny), covs_kept=trues(size(Z, 2) - Lz), ok=ok,
                 res=fill(NaN, n, ny))
    count(ok) == 0 && return na_result
    Zs = Z[ok, :] .* sw
    keep = _rd_h_indep_cols(Zs)
    all(keep[1:Lz]) || return na_result
    covs_kept = keep[(Lz + 1):end]
    if !all(keep)
        if warn_collinear
            @warn "Covariates collinear with the local linear design were dropped " *
                  "(columns $(findall(!, keep[(Lz + 1):end])))"
        end
        Z = Z[:, keep]
        Zs = Zs[:, keep]
    end
    F = qr(Zs)
    Q = Matrix(F.Q)
    R = UpperTriangular(F.R)
    beta = R \ (Q' * (d.Y[ok, :] .* sw))
    wgt = zeros(n)
    wgt[ok] = (R \ Matrix(Q'))[1, :] .* sw
    res = d.Y .- Z * beta
    Yadj = d.Y
    if size(Z, 2) > Lz
        Yadj = d.Y .- Z[:, (Lz + 1):end] * beta[(Lz + 1):end, :]
    end
    # Effective observations: rescale against the uniform-kernel estimator.
    Wu = d.w .* (abs.(X) .<= h)
    oku = Wu .> 0
    swu = sqrt.(Wu[oku])
    Fu = qr(Z[oku, :] .* swu)
    Ru = UpperTriangular(Fu.R)
    wgt_u = (Ru \ Matrix(Matrix(Fu.Q)'))[1, :] .* swu
    eff_obs = sum(Wu) * sum(wgt_u .^ 2 ./ d.w[oku]) / sum(wgt .^ 2 ./ d.w)

    hsigma2 = if se_method === :nn
        s2 = zeros(n, ny^2)
        if d.class === :ip
            ii = findall(ok)
            s2[ii, :] = _rd_h_sigma_nn(X[ii], d.Y[ii, :], J, d.w[ii])
        else
            for side in (d.m, d.p)
                ii = findall(side .& ok)
                s2[ii, :] = _rd_h_sigma_nn(X[ii], Yadj[ii, :], J, d.w[ii])
            end
        end
        s2
    elseif se_method === :ehw
        s2 = zeros(n, ny^2)
        idx = 0
        for b in 1:ny, a in 1:ny
            idx += 1
            s2[:, idx] = res[:, b] .* res[:, a]
        end
        s2
    else
        d.sigma2
    end
    w2 = wgt .^ 2
    V = if d.cluster === nothing
        vec(sum(w2 .* hsigma2; dims=1))
    elseif se_method === :supplied
        G = maximum(d.cluster)
        sg = zeros(G)
        for i in 1:n
            sg[d.cluster[i]] += wgt[i]
        end
        vec(sum(w2 .* hsigma2; dims=1)) .+ d.rho .* (sum(abs2, sg) - sum(w2))
    else
        G = maximum(d.cluster)
        us = zeros(G, ny)
        for i in findall(ok), c in 1:ny
            us[d.cluster[i], c] += wgt[i] * res[i, c]
        end
        vec(us' * us)
    end
    estimate = beta[1, 1]
    se = sqrt(V[1])
    fs = NaN
    rf = beta[1, 1]
    if d.class === :frd
        fs = beta[1, 2]
        estimate = beta[1, 1] / fs
        se = sqrt(sum([1, -estimate, -estimate, estimate^2] .* V) / fs^2)
    end
    return (estimate=estimate, se=se, est_w=wgt, sigma2=hsigma2, eff_obs=eff_obs, fs=fs,
            rf=rf, V=V, Yadj=Yadj, beta=beta, covs_kept=covs_kept, ok=ok, res=res)
end

# ---------------------------------------------------------------------------------------
# Worst-case bias
# ---------------------------------------------------------------------------------------

# ∫ |a + b s| ds over [lo, hi].
function _rd_h_int_abs_linear(a, b, lo, hi)
    hi <= lo && return 0.0
    f(s) = a + b * s
    F(s) = a * s + b * s^2 / 2
    if b != 0
        r = -a / b
        if lo < r < hi
            return abs(F(r) - F(lo)) + abs(F(hi) - F(r))
        end
    end
    return abs(F(hi) - F(lo))
end

"""
Bias constant `c` with worst-case bias `M c` of the linear estimator with weights `wt`
at points `xx` (nonzero weights only), for the Taylor (`:taylor`) or Hölder (`:holder`)
class with second derivative bounded by `M`. For the Hölder class away from a boundary
(inference at an interior point) the integral representation of Armstrong & Kolesár
(2020) is evaluated exactly (RDHonest integrates numerically).
"""
function _rd_h_bias_constant(wt::Vector{Float64}, xx::Vector{Float64}, h::Float64,
                             sclass::Symbol, boundary::Bool)
    if sclass === :taylor
        return sum(abs.(wt .* xx .^ 2)) / 2
    elseif boundary
        sm = 0.0
        sp = 0.0
        for i in eachindex(xx)
            if xx[i] < 0
                sm += wt[i] * xx[i]^2
            else
                sp += wt[i] * xx[i]^2
            end
        end
        return abs(sm - sp) / 2
    end
    # Interior point: M ∫_0^h |Σ_{x ≥ s} w (x - s)| ds + M ∫_{-h}^0 |Σ_{x ≤ s} w (s - x)| ds
    total = 0.0
    br = sort(unique(vcat(0.0, h, filter(v -> 0 < v < h, xx))))
    for k in 1:(length(br) - 1)
        lo, hi = br[k], br[k + 1]
        mid = (lo + hi) / 2
        a = 0.0
        b = 0.0
        for i in eachindex(xx)
            if xx[i] >= mid
                a += wt[i] * xx[i]
                b -= wt[i]
            end
        end
        total += _rd_h_int_abs_linear(a, b, lo, hi)
    end
    br = sort(unique(vcat(-h, 0.0, filter(v -> -h < v < 0, xx))))
    for k in 1:(length(br) - 1)
        lo, hi = br[k], br[k + 1]
        mid = (lo + hi) / 2
        a = 0.0
        b = 0.0
        for i in eachindex(xx)
            if xx[i] <= mid
                a -= wt[i] * xx[i]
                b += wt[i]
            end
        end
        total += _rd_h_int_abs_linear(a, b, lo, hi)
    end
    return total
end

"""
Estimate, standard error and worst-case bias at bandwidth `h` (RDHonest `NPRHonest`).
With `T0bias = true` (fuzzy designs, used for bandwidth selection) the bias and standard
error are scaled by the first stage and the bias uses the preliminary estimate `T0`.
"""
function _rd_h_nprhonest(d::_RDHonestData, M::Vector{Float64}, kernel::Symbol,
                         h::Float64; se_method::Symbol, J::Int, sclass::Symbol,
                         T0::Float64=0.0, T0bias::Bool=false,
                         warn_collinear::Bool=false)
    r1 = _rd_h_npreg(d, h, kernel; order=1, se_method, J, warn_collinear)
    nz = r1.est_w .!= 0
    wt = r1.est_w[nz]
    xx = d.X[nz]
    bd = d.class === :ip ? length(unique(xx .>= 0)) == 1 : true
    se = r1.se
    if d.class === :frd && T0bias
        se = se * abs(r1.fs)
        Mu = M[1] + M[2] * abs(T0)
        M_rf, M_fs = M[1], M[2]
    elseif d.class === :frd
        Mu = (M[1] + M[2] * abs(r1.estimate)) / abs(r1.fs)
        M_rf, M_fs = M[1], M[2]
    else
        Mu = M[1]
        M_rf, M_fs = M[1], 0.0
    end
    bconst = NaN
    if r1.eff_obs == 0
        bias = se = sqrt(floatmax(Float64) / 10)
    else
        bconst = _rd_h_bias_constant(wt, xx, h, sclass, bd)
        bias = Mu * bconst
    end
    wn = d.w[nz]
    leverage = isempty(wt) ? Inf : maximum(vcat(0.0, wt .^ 2 ./ wn .^ 2)) /
                                   sum(wt .^ 2 ./ wn)
    return (estimate=r1.estimate, se=se, bias=bias, h=h, eff_obs=r1.eff_obs,
            leverage=leverage, M=Mu, M_rf=M_rf, M_fs=M_fs, fs=r1.fs, rf=r1.rf, V=r1.V,
            bias_constant=bconst, est_w=r1.est_w, covs_kept=r1.covs_kept,
            boundary=bd)
end

# ---------------------------------------------------------------------------------------
# Rule-of-thumb M, preliminary variance and bandwidths
# ---------------------------------------------------------------------------------------

# Global quartic rule of thumb for a bound on |f''| (Armstrong & Kolesár 2020).
function _rd_h_mrot_side(Y::AbstractVector, X::AbstractVector, w::AbstractVector)
    length(unique(X)) >= 5 || throw(ArgumentError(
        "insufficient distinct values of the running variable to compute the " *
        "rule of thumb for M (need 5 on each side); supply `M`"))
    r1 = _rd_h_ols(_rd_vander(X, 4), Y; w=w)
    r1 === nothing && throw(ArgumentError(
        "the global quartic fit for the rule of thumb for M is not identified; " *
        "supply `M`"))
    f2(x) = abs(2 * r1[3] + 6 * x * r1[4] + 12 * x^2 * r1[5])
    f2e = abs(r1[5]) <= 1e-10 ? Inf : -r1[4] / (4 * r1[5])
    xmin, xmax = extrema(X)
    M = max(f2(xmin), f2(xmax))
    (xmin < f2e && xmax > f2e) && (M = max(f2(f2e), M))
    return M
end

function _rd_h_mrot(d::_RDHonestData)
    if d.class === :ip
        return [_rd_h_mrot_side(d.Y[:, 1], d.X, d.w)]
    end
    out = Float64[]
    for c in 1:_rd_h_ny(d)
        push!(out, max(_rd_h_mrot_side(d.Y[d.p, c], d.X[d.p], d.w[d.p]),
                       _rd_h_mrot_side(d.Y[d.m, c], d.X[d.m], d.w[d.m])))
    end
    return out
end

_rd_h_nth(v, k) = length(v) >= k ? v[k] : throw(ArgumentError(
    "too few distinct values of the running variable near the cutoff (need at least " *
    "$k on each side)"))

# Imbens & Kalyanaraman (2012) bandwidth for the outcome equation (RDHonest `IKBW`).
function _rd_h_ikbw(d::_RDHonestData)
    X = d.X
    Y = d.Y[:, 1]
    Nm, Np = count(d.m), count(d.p)
    N = Nm + Np
    nu0, mu2 = 4.79999999999999982, -0.10000000000000001   # triangular, boundary, p=1
    cnst = (nu0 / mu2^2)^(1 / 5)
    ds = _rd_h_prelim_var(_rd_h_copy(d; Y=d.Y[:, 1:1], class=:srd), :silverman)
    h1 = 1.84 * std(X) / N^(1 / 5)
    f0 = count(abs.(X) .<= h1) / (2 * N * h1)
    varm = ds.sigma2[findfirst(d.m), 1]
    varp = ds.sigma2[findfirst(d.p), 1]
    b3 = _rd_h_ols(hcat(ones(N), X .>= 0, X, X .^ 2, X .^ 3), Y)
    b3 === nothing && return NaN
    m3 = 6 * b3[5]
    h2m = 7200^(1 / 7) * (varm / (f0 * m3^2))^(1 / 7) * Nm^(-1 / 7)
    h2p = 7200^(1 / 7) * (varp / (f0 * m3^2))^(1 / 7) * Np^(-1 / 7)
    im = (X .>= -h2m) .& (X .< 0)
    ip = (X .<= h2p) .& (X .>= 0)
    bm = _rd_h_ols(_rd_vander(X[im], 2), Y[im])
    bp = _rd_h_ols(_rd_vander(X[ip], 2), Y[ip])
    (bm === nothing || bp === nothing) && return NaN
    m2m = 2 * bm[3]
    m2p = 2 * bp[3]
    rm = 2160 * varm / (count(im) * h2m^4)
    rp = 2160 * varp / (count(ip) * h2p^4)
    return cnst * ((varp + varm) / (f0 * N * ((m2p - m2m)^2 + rm + rp)))^(1 / 5)
end

# Fan & Gijbels (1996) rule-of-thumb bandwidth for inference at a point (RDHonest
# `ROTBW`), used only for the preliminary variance.
function _rd_h_rotbw(d::_RDHonestData)
    X = d.X
    boundary = minimum(X) >= 0 || maximum(X) <= 0
    N = length(X)
    q = _rd_quantile_type7(X, [0.25, 0.75])
    h1 = 1.843 * min(std(X), (q[2] - q[1]) / 1.349) / N^(1 / 5)
    f0 = count(abs.(X) .<= h1) / (2 * N * h1)
    Zq = _rd_vander(X, 4)
    r1 = _rd_h_ols(Zq, d.Y[:, 1])
    r1 === nothing && return NaN
    deriv = r1[3]
    sigma2 = sum(abs2, d.Y[:, 1] .- Zq * r1) / (N - 5)
    nu0, mup = boundary ? (4.79999999999999982, -0.10000000000000001) :
                          (0.66666666666666663, 0.16666666666666666)
    B = deriv * mup
    V = sigma2 * nu0 / f0
    return (V / (B^2 * 2 * 2 * N))^(1 / 5)
end

# Moulton estimate of the intra-cluster correlation of the residuals.
function _rd_h_moulton(u::AbstractMatrix, cl::AbstractVector{Int})
    counts = Dict{Int,Int}()
    for g in cl
        counts[g] = get(counts, g, 0) + 1
    end
    den = sum(abs2, values(counts)) - size(u, 1)
    den > 0 || return zeros(size(u, 2)^2)
    G = maximum(cl)
    us = zeros(G, size(u, 2))
    for i in axes(u, 1), c in axes(u, 2)
        us[cl[i], c] += u[i, c]
    end
    return vec(us' * us .- u' * u) ./ den
end

"""
Preliminary variance estimates (RDHonest `PrelimVar`): homoskedastic on either side of
the cutoff, from a local linear fit with the Imbens–Kalyanaraman (RD) or Fan–Gijbels
(point) bandwidth (`:ehw`), or from a uniform-kernel local constant fit with a Silverman
bandwidth (`:silverman`, used inside the IK selector).
"""
function _rd_h_prelim_var(d::_RDHonestData, se_initial::Symbol)
    X = d.X
    if d.class === :ip
        hmin = max(_rd_h_nth(sort(unique(abs.(X))), 2), _rd_h_nth(sort(abs.(X)), 4))
    else
        hmin = max(_rd_h_nth(sort(unique(X[d.p])), 3),
                   _rd_h_nth(sort(abs.(unique(X[d.m]))), 3),
                   _rd_h_nth(sort(X[d.p]), 4), _rd_h_nth(sort(abs.(X[d.m])), 4))
    end
    ny = _rd_h_ny(d)
    local r1, resid
    if se_initial === :ehw
        drf = d.class === :frd ? _rd_h_copy(d; Y=d.Y[:, 1:1], class=:srd) : d
        h1 = d.class === :ip ? _rd_h_rotbw(drf) : _rd_h_ikbw(drf)
        if isnan(h1)
            @warn "Preliminary bandwidth is NaN, setting it to Inf"
            h1 = Inf
        end
        r1 = _rd_h_npreg(d, max(h1, hmin), :triangular; se_method=:ehw)
        sigma2 = r1.sigma2
    else
        h1 = max(1.84 * std(X) / length(X)^(1 / 5), hmin)
        r1 = _rd_h_npreg(d, h1, :uniform; order=0, se_method=:ehw)
        sigma2 = copy(r1.sigma2)
        nz = r1.est_w .!= 0
        lp = count(d.p .& nz)
        lm = count(d.m .& nz)
        sigma2[d.p, :] .*= lp / (lp - 1)
        sigma2[d.m, :] .*= lm / (lm - 1)
    end
    nz = r1.est_w .!= 0
    any(nz) || throw(ArgumentError("preliminary variance estimate is not identified: " *
                                   "too few observations near the cutoff"))
    rho = nothing
    if d.cluster !== nothing
        rho = _rd_h_moulton(r1.res[nz, :], d.cluster[nz])
    end
    n = length(X)
    s2 = zeros(n, ny^2)
    if d.class === :ip
        s2 .= mean(sigma2[nz, :]; dims=1)
    else
        mp = vec(mean(sigma2[d.p .& nz, :]; dims=1))
        mm = vec(mean(sigma2[d.m .& nz, :]; dims=1))
        for i in 1:n
            s2[i, :] = d.p[i] ? mp : mm
        end
    end
    return _rd_h_copy(d; sigma2=s2, rho=rho)
end

# Optimal bandwidth for the MSE, FLCI or OCI criterion (RDHonest `OptBW`).
function _rd_h_optbw(d::_RDHonestData, M::Vector{Float64}, kernel::Symbol,
                     criterion::Symbol, alpha::Float64, beta::Float64, sclass::Symbol,
                     T0::Float64)
    d.sigma2 === nothing && (d = _rd_h_prelim_var(d, :ehw))
    za = quantile(Normal(), 1 - alpha)
    zb = quantile(Normal(), beta)
    function obj(h)
        r = _rd_h_nprhonest(d, M, kernel, h; se_method=:supplied, J=3, sclass, T0,
                            T0bias=true)
        if criterion === :oci
            return 2 * r.bias + r.se * (za + zb)
        elseif criterion === :mse
            return r.bias^2 + r.se^2
        else
            return 2 * _rd_cvb(r.bias / r.se, alpha) * r.se
        end
    end
    if d.class === :ip
        hmin = _rd_h_nth(sort(unique(abs.(d.X))), 2)
    else
        hmin = max(_rd_h_nth(unique(d.X[d.p]), 2),
                   _rd_h_nth(sort(unique(abs.(d.X[d.m]))), 2))
    end
    hmax = maximum(abs.(d.X))
    if kernel === :uniform
        supp = sort(unique(abs.(d.X)))
        return _rd_h_gss(obj, supp[supp .>= hmin])
    end
    return abs(_rd_brent_fmin(obj, hmin, hmax, Base.eps(Float64)^0.75))
end
