# Numerical integration for group-sequential boundaries (Armitage, McPherson & Rowe
# 1969; Jennison & Turnbull 2000, ch. 19). The standardized statistics Z_k with
# information I_k have the canonical joint distribution
#     Z_k ~ N(θ √I_k, 1),   Cov(Z_j, Z_k) = √(I_j / I_k)  (j ≤ k),
# so S_k = Z_k √I_k has independent increments N(θ ΔI, ΔI). The sub-density of Z_k on
# the continuation region is propagated on the grid of Jennison & Turnbull (p. 349)
# with Simpson weights, exactly as in gsDesign's C code (grid parameter r = 18).

const _SEQ_GS_R = 18

"""Grid points and Simpson weights on `[a, b]` around mean `mu` (gsDesign `gridpts`)."""
function _seq_gridpts(r::Int, mu::Float64, a::Float64, b::Float64)
    rd = float(r)
    r2 = 2 * rd
    r5 = 5r
    r6 = 6r
    z = zeros(12r - 3)
    w = zeros(12r - 3)
    j = 1                       # 1-based index of the last grid point written
    done = false
    ztem = mu - 3 - 4 * log(rd)
    if ztem <= a
        z[1] = a
    elseif ztem >= b
        z[1] = b
        done = true
    else
        z[1] = ztem
    end
    i = 2
    while i < r6 && !done
        ztem = i < r ? mu - 3 - 4 * log(rd / i) :
               i <= r5 ? mu + 3 * (-1 + (i - r) / r2) :
               mu + 3 + 4 * log(rd / (r6 - i))
        if ztem > a
            j += 2
            z[j] = ztem
            if ztem >= b
                z[j] = b
                done = true
            end
            z[j - 1] = (z[j] + z[j - 2]) / 2
        end
        i += 1
    end
    if j > 1
        w[1] = (z[3] - z[1]) / 6
        w[j] = (z[j] - z[j - 2]) / 6
        w[j - 1] = 2 * (z[j] - z[j - 2]) / 3
    end
    k = 2
    while k < j - 1
        w[k] = 2 * (z[k + 1] - z[k - 1]) / 3
        w[k + 1] = (z[k + 3] - z[k - 1]) / 6
        k += 2
    end
    return z[1:j], w[1:j]
end

const _SEQ_INV_SQRT_2PI = 1 / sqrt(2π)

# State of the recursion after a look: weighted sub-density `h` of Z_k at grid `z`
# restricted to the continuation region, and the information of that look.
struct _SeqGSState
    z::Vector{Float64}
    h::Vector{Float64}
    info::Float64
    first::Bool                  # no look yet
end

_seq_gs_start() = _SeqGSState(Float64[], Float64[], 0.0, true)

"""P(Z_k ≥ b, continued before) under drift `theta` at information `ik`."""
function _seq_gs_up(s::_SeqGSState, theta::Float64, ik::Float64, b::Float64)
    isinf(b) && return b > 0 ? 0.0 : _seq_gs_mass(s)
    s.first && return ccdf(Normal(), b - theta * sqrt(ik))
    dl = ik - s.info
    rd = sqrt(dl); rik = sqrt(ik); rim = sqrt(s.info)
    p = 0.0
    @inbounds for i in eachindex(s.z)
        p += s.h[i] * ccdf(Normal(), (b * rik - s.z[i] * rim - theta * dl) / rd)
    end
    return p
end

"""P(Z_k ≤ a, continued before)."""
function _seq_gs_lo(s::_SeqGSState, theta::Float64, ik::Float64, a::Float64)
    isinf(a) && return a < 0 ? 0.0 : _seq_gs_mass(s)
    s.first && return cdf(Normal(), a - theta * sqrt(ik))
    dl = ik - s.info
    rd = sqrt(dl); rik = sqrt(ik); rim = sqrt(s.info)
    p = 0.0
    @inbounds for i in eachindex(s.z)
        p += s.h[i] * cdf(Normal(), (a * rik - s.z[i] * rim - theta * dl) / rd)
    end
    return p
end

_seq_gs_mass(s::_SeqGSState) = s.first ? 1.0 : sum(s.h)

"""Propagate the sub-density to look `k` with continuation region `(a, b)`."""
function _seq_gs_advance(s::_SeqGSState, theta::Float64, ik::Float64, a::Float64,
                         b::Float64; r::Int=_SEQ_GS_R)
    mu = theta * sqrt(ik)
    z, w = _seq_gridpts(r, mu, a, b)
    h = similar(z)
    if s.first
        @inbounds for i in eachindex(z)
            x = z[i] - mu
            h[i] = w[i] * exp(-x^2 / 2) * _SEQ_INV_SQRT_2PI
        end
    else
        dl = ik - s.info
        rd = sqrt(dl); rik = sqrt(ik); rim = sqrt(s.info)
        c = _SEQ_INV_SQRT_2PI * rik / rd
        @inbounds for i in eachindex(z)
            acc = 0.0
            zi = z[i] * rik - theta * dl
            for ii in eachindex(s.z)
                x = (zi - s.z[ii] * rim) / rd
                acc += s.h[ii] * exp(-x^2 / 2)
            end
            h[i] = w[i] * acc * c
        end
    end
    return _SeqGSState(z, h, ik, false)
end

"""
    _seq_gs_probs(a, b, info, theta) -> (upper, lower)

Probabilities of first crossing the upper bound `b[k]` / lower bound `a[k]` at each
look under drift `theta` (gsDesign `gsProbability`). `±Inf` bounds are allowed.
"""
function _seq_gs_probs(a::AbstractVector, b::AbstractVector, info::AbstractVector,
                       theta::Real; r::Int=_SEQ_GS_R)
    K = length(info)
    up = zeros(K); lo = zeros(K)
    s = _seq_gs_start()
    th = float(theta)
    for k in 1:K
        up[k] = _seq_gs_up(s, th, float(info[k]), float(b[k]))
        lo[k] = _seq_gs_lo(s, th, float(info[k]), float(a[k]))
        k < K && (s = _seq_gs_advance(s, th, float(info[k]), float(a[k]), float(b[k]);
                                     r=r))
    end
    return up, lo
end

# Root of a monotone function on a bracket: Illinois (modified regula falsi) steps,
# with a bisection step whenever the bracket fails to halve, until it is `tol` wide.
function _seq_root(f, lo::Float64, hi::Float64; tol::Float64=1e-12, maxit::Int=300)
    a, b = lo, hi
    fa, fb = f(a), f(b)
    fa == 0 && return a
    fb == 0 && return b
    sign(fa) == sign(fb) &&
        throw(ArgumentError("root not bracketed in [$lo, $hi] (f = $fa, $fb)"))
    side = 0
    width = abs(b - a)
    for it in 1:maxit
        c = (a * fb - b * fa) / (fb - fa)
        if !(min(a, b) < c < max(a, b)) || it % 4 == 0 && abs(b - a) > width / 2
            c = (a + b) / 2
            it % 4 == 0 && (width = abs(b - a))
        end
        fc = f(c)
        fc == 0 && return c
        if sign(fc) == sign(fb)
            b, fb = c, fc
            side == -1 && (fa /= 2)
            side = -1
        else
            a, fa = c, fc
            side == 1 && (fb /= 2)
            side = 1
        end
        abs(b - a) < tol && break
    end
    return (a + b) / 2
end

const _SEQ_GS_ZMAX = 40.0

"""Upper bound `b` with `P(Z_k ≥ b, continued) = target` (Inf when `target ≤ 0`)."""
function _seq_gs_solve_upper(s::_SeqGSState, theta::Float64, ik::Float64,
                             target::Float64; lower::Float64=-Inf)
    target <= 1e-300 && return Inf
    f(b) = _seq_gs_up(s, theta, ik, b) - target
    lo = -_SEQ_GS_ZMAX
    f(lo) < 0 && return lo                    # target exceeds the remaining mass
    return _seq_root(f, lo, _SEQ_GS_ZMAX)
end

"""Lower bound `a` with `P(Z_k ≤ a, continued) = target` (-Inf when `target ≤ 0`)."""
function _seq_gs_solve_lower(s::_SeqGSState, theta::Float64, ik::Float64,
                             target::Float64)
    target <= 1e-300 && return -Inf
    f(a) = _seq_gs_lo(s, theta, ik, a) - target
    f(_SEQ_GS_ZMAX) < 0 && return _SEQ_GS_ZMAX
    return _seq_root(f, -_SEQ_GS_ZMAX, _SEQ_GS_ZMAX)
end
