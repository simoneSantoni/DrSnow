# Density discontinuity (manipulation) test based on the local polynomial density
# estimator of Cattaneo, Jansson & Ma (2020): rddensity / rdbwdensity equivalents.

# ---------------------------------------------------------------------------------------
# Kernel moment matrices (exact rational arithmetic; rddensity integrates numerically)
# ---------------------------------------------------------------------------------------

const _RD_Q = Rational{BigInt}

function _rd_poly_mul(a::Vector{_RD_Q}, b::Vector{_RD_Q})
    out = zeros(_RD_Q, length(a) + length(b) - 1)
    for i in eachindex(a), j in eachindex(b)
        out[i + j - 1] += a[i] * b[j]
    end
    return out
end

_rd_monomial(k::Int) = (v = zeros(_RD_Q, k + 1); v[end] = 1; v)
_rd_poly_antider(a::Vector{_RD_Q}) = vcat(zero(_RD_Q), [a[i] / i for i in eachindex(a)])
_rd_poly_eval(a::Vector{_RD_Q}, x) = sum(a[i] * _RD_Q(x)^(i - 1) for i in eachindex(a))

function _rd_poly_add(a::Vector{_RD_Q}, b::Vector{_RD_Q})
    n = max(length(a), length(b))
    out = zeros(_RD_Q, n)
    out[1:length(a)] .+= a
    out[1:length(b)] .+= b
    return out
end

# Kernel restricted to [low, up] ⊂ [-1, 0] or [0, 1], as a polynomial.
function _rd_kernel_poly(kernel::Symbol, low::Int, up::Int)
    kernel === :uniform && return _RD_Q[1 // 2]
    kernel === :epanechnikov && return _RD_Q[3 // 4, 0, -3 // 4]
    return low >= 0 ? _RD_Q[1, -1] : _RD_Q[1, 1]      # 1 - |x|
end

_rd_definite(a, low, up) = (F = _rd_poly_antider(a);
                           _rd_poly_eval(F, up) - _rd_poly_eval(F, low))

const _RD_MOMENT_CACHE = Dict{Tuple,Matrix{Float64}}()
const _RD_MOMENT_LOCK = ReentrantLock()

function _rd_cached(f, key)
    lock(_RD_MOMENT_LOCK) do
        get!(f, _RD_MOMENT_CACHE, key)
    end
end

"""`S[i,j] = ∫ x^{i+j-2} K(x) dx` over `[low, up]` (rddensity `Sgenerate`)."""
function _rd_Sgen(p::Int, low::Int, up::Int, kernel::Symbol)
    _rd_cached((:S, p, low, up, kernel)) do
        k = _rd_kernel_poly(kernel, low, up)
        [Float64(_rd_definite(_rd_poly_mul(_rd_monomial(i + j - 2), k), low, up))
         for i in 1:(p + 1), j in 1:(p + 1)]
    end
end

"""`C[i] = ∫ x^{i+k-1} K(x) dx` over `[low, up]` (rddensity `Cgenerate`)."""
function _rd_Cgen(k::Int, p::Int, low::Int, up::Int, kernel::Symbol)
    _rd_cached((:C, k, p, low, up, kernel)) do
        kp = _rd_kernel_poly(kernel, low, up)
        reshape([Float64(_rd_definite(_rd_poly_mul(_rd_monomial(i + k - 1), kp), low, up))
                 for i in 1:(p + 1)], :, 1)
    end
end

"""
`G[i,j] = ∫∫_{x<y} x^i y^{j-1} K(x)K(y) + ∫∫_{x>y} x^{i-1} y^j K(x)K(y)` over
`[low, up]²` (rddensity `Ggenerate`).
"""
function _rd_Ggen(p::Int, low::Int, up::Int, kernel::Symbol)
    _rd_cached((:G, p, low, up, kernel)) do
        k = _rd_kernel_poly(kernel, low, up)
        G = zeros(p + 1, p + 1)
        for i in 1:(p + 1), j in 1:(p + 1)
            # inner1(y) = ∫_low^y x^i K(x) dx ; inner2(y) = ∫_y^up x^{i-1} K(x) dx
            F1 = _rd_poly_antider(_rd_poly_mul(_rd_monomial(i), k))
            inner1 = _rd_poly_add(F1, _RD_Q[-_rd_poly_eval(F1, low)])
            F2 = _rd_poly_antider(_rd_poly_mul(_rd_monomial(i - 1), k))
            inner2 = _rd_poly_add(-F2, _RD_Q[_rd_poly_eval(F2, up)])
            t1 = _rd_poly_mul(_rd_poly_mul(_rd_monomial(j - 1), k), inner1)
            t2 = _rd_poly_mul(_rd_poly_mul(_rd_monomial(j), k), inner2)
            G[i, j] = Float64(_rd_definite(t1, low, up) + _rd_definite(t2, low, up))
        end
        G
    end
end

function _rd_plus(M::Matrix{Float64}, p::Int)
    T = zeros(p + 2, p + 2)
    T[1, 1] = M[1, 1]
    T[1, 3:(p + 2)] = M[1, 2:(p + 1)]
    T[3:(p + 2), 1] = M[2:(p + 1), 1]
    T[3:(p + 2), 3:(p + 2)] = M[2:(p + 1), 2:(p + 1)]
    return T
end

function _rd_minus(M::Matrix{Float64}, p::Int)
    T = zeros(p + 2, p + 2)
    T[1:2, 1:2] = M[1:2, 1:2]
    if p > 1
        T[1:2, 4:(p + 2)] = M[1:2, 3:(p + 1)]
        T[4:(p + 2), 1:2] = M[3:(p + 1), 1:2]
        T[4:(p + 2), 4:(p + 2)] = M[3:(p + 1), 3:(p + 1)]
    end
    return T
end

_rd_Splus(p, kernel) = _rd_plus(_rd_Sgen(p, 0, 1, kernel), p)
_rd_Gplus(p, kernel) = _rd_plus(_rd_Ggen(p, 0, 1, kernel), p)
function _rd_Cplus(k, p, kernel)
    C = _rd_Cgen(k, p, 0, 1, kernel)
    T = zeros(p + 2, 1)
    T[1] = C[1]
    T[3:(p + 2)] = C[2:(p + 1)]
    return T
end
function _rd_Psi(p)
    d = p > 1 ? vcat([1.0, 0.0, 0.0], [(-1.0)^k for k in 2:p]) : [1.0, 0.0, 0.0]
    P = Matrix(Diagonal(d))
    P[2, 3] = P[3, 2] = -1
    return P
end

# ---------------------------------------------------------------------------------------
# Estimator (rddensity_fV)
# ---------------------------------------------------------------------------------------

"""Tie structure of a sorted vector: unique values, frequencies, first/last indices."""
function _rd_unique_runs(x::AbstractVector)
    n = length(x)
    firsts = Int[]
    lasts = Int[]
    i = 1
    while i <= n
        j = i
        while j < n && x[j + 1] == x[i]
            j += 1
        end
        push!(firsts, i)
        push!(lasts, j)
        i = j + 1
    end
    return (unique=x[firsts], freq=lasts .- firsts .+ 1, first=firsts, last=lasts)
end

"""
Local polynomial density estimates at the cutoff and their variances. Returns a 4×4
matrix: rows (left, right, diff, sum); columns (estimate, jackknife variance,
plug-in variance, derivative of order `s`). `NaN` marks unavailable entries.
"""
function _rd_density_fV(Y, X, N, Nlh, Nrh, hl, hr, p, s, kernel, fitselect, vce,
                        masspoints)
    Nh = Nlh + Nrh
    out = fill(NaN, 4, 4)
    W = Vector{Float64}(undef, Nh)
    for k in 1:Nh
        left = k <= Nlh
        h = left ? hl : hr
        u = X[k] / h
        W[k] = if kernel === :uniform
            1 / (2 * h)
        elseif kernel === :triangular
            left ? (1 + u) / h : (1 - u) / h
        else
            0.75 * (1 - u^2) / h
        end
    end
    if fitselect === :restricted
        Xp = zeros(Nh, p + 2)
        Xp[:, 1] .= 1
        for k in 1:Nh
            left = k <= Nlh
            u = X[k] / (left ? hl : hr)
            left ? (Xp[k, 2] = u) : (Xp[k, 3] = u)
            for j in 4:(p + 2)
                Xp[k, j] = u^(j - 2)
            end
        end
        v = p > 1 ? vcat([0, 1, 1], collect(2:p)) : [0, 1, 1]
        Hp = hl .^ v
    else
        Xp = zeros(Nh, 2 * p + 2)
        Hp = zeros(2 * p + 2)
        for j in 1:(2 * p + 2)
            if isodd(j)
                e = (j - 1) ÷ 2
                for k in 1:Nlh
                    Xp[k, j] = (X[k] / hl)^e
                end
                Hp[j] = hl^e
            else
                e = (j - 2) ÷ 2
                for k in (Nlh + 1):Nh
                    Xp[k, j] = (X[k] / hr)^e
                end
                Hp[j] = hr^e
            end
        end
    end
    XpW = Xp .* W
    Sinv = try
        inv(XpW' * Xp)
    catch err
        err isa SingularException || rethrow()
        return out
    end
    all(isfinite, Sinv) || return out
    HpInv = Diagonal(1 ./ Hp)
    bvec = HpInv * Sinv * (XpW' * Y)
    if fitselect === :restricted
        out[1, 1] = bvec[2]; out[2, 1] = bvec[3]
        out[3, 1] = bvec[3] - bvec[2]; out[4, 1] = bvec[3] + bvec[2]
        out[1, 4] = out[2, 4] = bvec[s + 2]
        out[3, 4] = 0.0; out[4, 4] = 2 * out[1, 4]
    else
        out[1, 1] = bvec[3]; out[2, 1] = bvec[4]
        out[3, 1] = bvec[4] - bvec[3]; out[4, 1] = bvec[4] + bvec[3]
        out[1, 4] = bvec[2 * s + 1]; out[2, 4] = bvec[2 * s + 2]
        out[3, 4] = out[2, 4] - out[1, 4]; out[4, 4] = out[2, 4] + out[1, 4]
    end
    if vce === :jackknife
        ncol = size(Xp, 2)
        L = zeros(Nh, ncol)
        for jj in 1:ncol
            # tail sums: Σ_{j>i} XpW[j, jj] / (N - 1)
            acc = 0.0
            tail = zeros(Nh)
            for i in Nh:-1:1
                tail[i] = acc / (N - 1)
                acc += XpW[i, jj]
            end
            if masspoints
                runs = _rd_unique_runs(X)
                for (f, l) in zip(runs.first, runs.last)
                    L[f:l, jj] .= tail[f]
                end
            else
                L[:, jj] = tail
            end
        end
        V = HpInv * Sinv * (L' * L) * Sinv * HpInv
        a, b = fitselect === :restricted ? (2, 3) : (3, 4)
        out[1, 2] = V[a, a]; out[2, 2] = V[b, b]
        out[3, 2] = V[a, a] + V[b, b] - 2 * V[a, b]
        out[4, 2] = V[a, a] + V[b, b] + 2 * V[a, b]
    else
        if fitselect === :unrestricted
            S = _rd_Sgen(p, 0, 1, kernel)
            G = _rd_Ggen(p, 0, 1, kernel)
            V = inv(S) * G * inv(S)
            out[1, 3] = out[1, 1] * V[2, 2] / (N * hl)
            out[2, 3] = out[2, 1] * V[2, 2] / (N * hr)
            out[3, 3] = out[4, 3] = out[1, 3] + out[2, 3]
        else
            S = _rd_Splus(p, kernel)
            G = _rd_Gplus(p, kernel)
            Psi = _rd_Psi(p)
            Sm = Psi * S * Psi
            Gm = Psi * G * Psi
            A = inv(out[1, 1] .* Sm .+ out[2, 1] .* S)
            V = A * (out[1, 1]^3 .* Gm .+ out[2, 1]^3 .* G) * A
            out[1, 3] = V[2, 2] / (N * hl)
            out[2, 3] = V[3, 3] / (N * hl)
            out[3, 3] = (V[2, 2] + V[3, 3] - 2 * V[2, 3]) / (N * hl)
            out[4, 3] = (V[2, 2] + V[3, 3] + 2 * V[2, 3]) / (N * hl)
        end
    end
    for i in 1:4, j in 2:3
        if !isnan(out[i, j]) && out[i, j] < 0
            out[i, j] = NaN
        end
    end
    return out
end

# ---------------------------------------------------------------------------------------
# Bandwidth selection (rdbwdensity)
# ---------------------------------------------------------------------------------------

function _rd_hermite(x, p)
    p == 0 && return 1.0
    p == 1 && return x
    p == 2 && return x^2 - 1
    p == 3 && return x^3 - 3x
    p == 4 && return x^4 - 6x^2 + 3
    p == 5 && return x^5 - 10x^3 + 15x
    p == 6 && return x^6 - 15x^4 + 45x^2 - 15
    p == 7 && return x^7 - 21x^5 + 105x^3 - 105x
    p == 8 && return x^8 - 28x^6 + 210x^4 - 420x^2 + 105
    p == 9 && return x^9 - 36x^7 + 378x^5 - 1260x^3 + 945x
    return x^10 - 45x^8 + 630x^6 - 3150x^4 + 4725x^2 - 945
end

const _RD_CB = (25884.4444444942, 3430865.45512362, 845007948.042626, 330631733667.038,
                187774809656037.0, 145729502641999264.0, 1.4601350297445e20)
const _RD_CC = (4.80000000000002, 548.571428571555, 100800.000000204, 29558225.4581006,
                12896196859.6126, 7890871468221.61, 6467911284037581.0)

# X: sorted running variable already centred at the cutoff.
function _rd_density_setup(X::Vector{Float64}, masspoints_opt::Bool)
    N = length(X)
    runs = _rd_unique_runs(X)
    hasrep = length(runs.unique) < N
    mflag = hasrep && masspoints_opt
    Y = collect(0:(N - 1)) ./ (N - 1)
    if mflag
        Y = reduce(vcat, [fill(Y[l], f) for (l, f) in zip(runs.last, runs.freq)])
    end
    return (; N, Y, XU=runs.unique, mflag)
end

_rd_kth(v, k) = isempty(v) ? -Inf : v[k]

function _rd_density_bw_table(X::Vector{Float64}, p::Int, kernel::Symbol,
                              fitselect::Symbol, vce::Symbol, regularize::Bool,
                              nlocalmin::Int, nuniquemin::Int, masspoints::Bool)
    D = _rd_density_setup(X, masspoints)
    N, Y, XU = D.N, D.Y, D.XU
    Nl = count(<(0), X); Nr = N - Nl
    NlU = count(<(0), XU); NrU = length(XU) - NlU
    Xmu = mean(X); Xsd = std(X)
    zz = Xmu / Xsd
    fhatb = 1 / (_rd_hermite(zz, p + 2)^2 * pdf(Normal(), zz))
    fhatc = 1 / (_rd_hermite(zz, p)^2 * pdf(Normal(), zz))
    bn = ((2p + 1) / 4 * fhatb * _RD_CB[p] / N)^(1 / (2p + 5)) * Xsd
    cn = (1 / (2p) * fhatc * _RD_CC[p] / N)^(1 / (2p + 1)) * Xsd
    absL = sort(abs.(X[X .< 0]))
    posR = X[X .>= 0]
    absLU = sort(abs.(XU[XU .< 0]))
    posRU = XU[XU .>= 0]
    if regularize
        bn = min(bn, maximum(abs.(XU)))
        cn = min(cn, maximum(abs.(XU)))
        if nlocalmin > 0
            bn = max(bn, _rd_kth(absL, min(20 + p + 2 + 1, Nl)),
                     _rd_kth(posR, min(20 + p + 2 + 1, Nr)))
            cn = max(cn, _rd_kth(absL, min(20 + p + 1, Nl)),
                     _rd_kth(posR, min(20 + p + 1, Nr)))
        end
        if nuniquemin > 0
            bn = max(bn, _rd_kth(absLU, min(20 + p + 2 + 1, NlU)),
                     _rd_kth(posRU, min(20 + p + 2 + 1, NrU)))
            cn = max(cn, _rd_kth(absLU, min(20 + p + 1, NlU)),
                     _rd_kth(posRU, min(20 + p + 1, NrU)))
        end
    end
    ib = abs.(X) .<= bn
    ic = abs.(X) .<= cn
    Xb, Yb = X[ib], Y[ib]
    Xc, Yc = X[ic], Y[ic]
    fV_b = _rd_density_fV(Yb, Xb, N, count(<(0), Xb), count(>=(0), Xb), bn, bn, p + 2,
                          p + 1, kernel, fitselect, vce, D.mflag)
    fV_c = _rd_density_fV(Yc, Xc, N, count(<(0), Xc), count(>=(0), Xc), cn, cn, p, 1,
                          kernel, fitselect, vce, D.mflag)
    hn = fill(NaN, 4, 3)
    hn[:, 2] = N * cn .* (vce === :plugin ? fV_c[:, 3] : fV_c[:, 2])
    if fitselect === :unrestricted
        S = _rd_Sgen(p, 0, 1, kernel)
        C = _rd_Cgen(p + 1, p, 0, 1, kernel)
        sc = (inv(S) * C)[2]
        hn[1, 3] = fV_b[1, 4] * sc * (-1)^p
        hn[2, 3] = fV_b[2, 4] * sc
    else
        Splus = _rd_Splus(p, kernel)
        Cplus = _rd_Cplus(p + 1, p, kernel)
        Psi = _rd_Psi(p)
        Sinv = inv(fV_c[2, 1] .* Splus .+ fV_c[1, 1] .* (Psi * Splus * Psi))
        C = fV_b[1, 4] .*
            (fV_c[2, 1] .* Cplus .+ (-1)^(p + 1) * fV_c[1, 1] .* (Psi * Cplus))
        tmp = Sinv * C
        hn[1, 3] = tmp[2]
        hn[2, 3] = tmp[3]
    end
    hn[3, 3] = hn[2, 3] - hn[1, 3]
    hn[4, 3] = hn[2, 3] + hn[1, 3]
    hn[:, 3] .= hn[:, 3] .^ 2
    hn[:, 1] .= (1 / (2p) .* hn[:, 2] ./ hn[:, 3] ./ N) .^ (1 / (2p + 1))
    for i in 1:4
        if !isnan(hn[i, 2]) && hn[i, 2] < 0
            hn[i, 1] = 0.0
            hn[i, 2] = NaN
        end
        isnan(hn[i, 1]) && (hn[i, 1] = 0.0)
    end
    if regularize
        maxabs = max(abs(XU[1]), XU[end])
        hn[1, 1] = min(hn[1, 1], abs(XU[1]))
        hn[2, 1] = min(hn[2, 1], XU[end])
        hn[3, 1] = min(hn[3, 1], maxabs)
        hn[4, 1] = min(hn[4, 1], maxabs)
        for (on, lv, rv, nl, nr, m) in ((nlocalmin > 0, absL, posR, Nl, Nr, nlocalmin),
                                        (nuniquemin > 0, absLU, posRU, NlU, NrU,
                                         nuniquemin))
            on || continue
            hlmin = _rd_kth(lv, min(nl, m))
            hrmin = _rd_kth(rv, min(nr, m))
            hn[1, 1] = max(hn[1, 1], hlmin)
            hn[2, 1] = max(hn[2, 1], hrmin)
            hn[3, 1] = max(hn[3, 1], hlmin, hrmin)
            hn[4, 1] = max(hn[4, 1], hlmin, hrmin)
        end
    end
    return hn
end

# ---------------------------------------------------------------------------------------
# Binomial tests in shrinking windows around the cutoff
# ---------------------------------------------------------------------------------------

"""Two-sided exact binomial test p-value with the conventions of R's `binom.test`."""
function _rd_binom_pvalue(x::Int, n::Int, p::Float64)
    n == 0 && return 1.0
    p == 0 && return Float64(x == 0)
    p == 1 && return Float64(x == n)
    B = Binomial(n, p)
    relErr = 1 + 1e-7
    d = pdf(B, x)
    m = n * p
    if x == m
        return 1.0
    elseif x < m
        y = count(i -> pdf(B, i) <= d * relErr, ceil(Int, m):n)
        pv = cdf(B, x) + ccdf(B, n - y)
    else
        y = count(i -> pdf(B, i) <= d * relErr, 0:floor(Int, m))
        pv = cdf(B, y - 1) + ccdf(B, x - 1)
    end
    return min(1.0, pv)
end

function _rd_window_steps(w1::Float64, h::Float64, nw::Int)
    k = collect(1:(nw - 1))
    step = w1 * nw > h ? (k .* (h - w1)) : (k .* w1)
    w1 * nw > h && (step = step ./ (nw - 1))
    return w1 .+ step
end

function _rd_binomial_windows(X::Vector{Float64}, hl, hr; binoW=nothing, binoN=nothing,
                              binoWStep=nothing, binoNStep=nothing, binoNW::Int=10,
                              binoP::Real=0.5)
    binoNW > 0 || throw(ArgumentError("binomial_windows must be positive"))
    0 <= binoP <= 1 || throw(ArgumentError("binomial_p must be in [0, 1]"))
    XL = sort(abs.(X[X .< 0]))
    XR = sort(X[X .>= 0])
    Nl, Nr = length(XL), length(XR)
    LW = fill(NaN, binoNW)
    RW = fill(NaN, binoNW)
    if binoW === nothing
        nn = binoN === nothing ? 20 : ceil(Int, binoN)
        nn > 0 || throw(ArgumentError("binomial_n must be positive"))
        LW[1] = RW[1] = max(XL[min(nn, Nl)], XR[min(nn, Nr)])
    else
        wl, wr = _rd_pair(binoW)
        (wl > 0 && wr > 0) || throw(ArgumentError("binomial window must be positive"))
        LW[1], RW[1] = wl, wr
    end
    if binoNW > 1
        if binoWStep === nothing && binoNStep === nothing
            if LW[1] >= hl || RW[1] >= hr
                LW, RW = LW[1:1], RW[1:1]
                binoNW = 1
            else
                # Separate passes (as R's vector arithmetic) avoid FMA contraction,
                # which would move window edges by one ulp relative to rddensity.
                LW[2:end] = _rd_window_steps(LW[1], hl, binoNW)
                RW[2:end] = _rd_window_steps(RW[1], hr, binoNW)
            end
        elseif binoWStep === nothing
            st = ceil(Int, binoNStep)
            for jj in 2:binoNW
                step = max(XL[min(count(<=(LW[jj - 1]), XL) + st, Nl)] - LW[jj - 1],
                           XR[min(count(<=(RW[jj - 1]), XR) + st, Nr)] - RW[jj - 1])
                LW[jj] = LW[jj - 1] + step
                RW[jj] = RW[jj - 1] + step
            end
        else
            sl, sr = _rd_pair(binoWStep)
            stl = collect(1:(binoNW - 1)) .* sl
            str = collect(1:(binoNW - 1)) .* sr
            LW[2:end] = LW[1] .+ stl
            RW[2:end] = RW[1] .+ str
        end
    end
    nL = [count(<=(w), XL) for w in LW]
    nR = [count(<=(w), XR) for w in RW]
    pv = [_rd_binom_pvalue(nL[j], nL[j] + nR[j], Float64(binoP)) for j in eachindex(LW)]
    return DataFrame(window_left=LW, window_right=RW, n_left=nL, n_right=nR, pvalue=pv)
end

# ---------------------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------------------

function _rd_density_opts(p, q, kernel, fitselect, vce, bwselect)
    1 <= p <= 7 || throw(ArgumentError("p must be an integer between 1 and 7"))
    q = q === nothing ? p + 1 : Int(q)
    q >= p || throw(ArgumentError("q cannot be smaller than p"))
    k = _rd_kernel(kernel)
    fs = Symbol(lowercase(string(fitselect)))
    fs in (:unrestricted, :restricted) || throw(ArgumentError(
        "fitselect must be :unrestricted or :restricted"))
    v = Symbol(lowercase(string(vce)))
    v in (:jackknife, :plugin) || throw(ArgumentError("vce must be :jackknife or :plugin"))
    bs = Symbol(lowercase(string(bwselect)))
    bs in (:comb, :each, :diff, :sum) || throw(ArgumentError(
        "bwselect must be :comb, :each, :diff or :sum"))
    (fs === :restricted && bs === :each) && throw(ArgumentError(
        "bwselect = :each is not available with fitselect = :restricted"))
    return q, k, fs, v, bs
end

function _rd_density_x(x, cutoff)
    xs = Float64[v for v in x if !ismissing(v) && !isnan(v)]
    sort!(xs)
    xs .+= 0.0
    isempty(xs) && throw(ArgumentError("running variable has no non-missing values"))
    (cutoff <= xs[1] || cutoff >= xs[end]) && throw(ArgumentError(
        "the cutoff $cutoff must lie strictly inside the range of the running variable"))
    return xs
end

"""
    rd_density_bandwidth(data, running; cutoff=0.0, p=2, kernel=:triangular,
                         fitselect=:unrestricted, vce=:jackknife, masspoints=true,
                         regularize=true, nlocalmin=20+p+1,
                         nuniquemin=20+p+1) -> DataFrame
    rd_density_bandwidth(x::AbstractVector; kwargs...) -> DataFrame

MSE-optimal bandwidths for the local polynomial density estimator of Cattaneo, Jansson
and Ma (2020) at the cutoff, equivalent to `rdbwdensity` from the R/Stata package
`rddensity`.

The density test [`rd_density_test`](@ref) estimates the density of the running variable
on each side of the cutoff as the derivative of a local polynomial fit to the empirical
distribution function. Its bandwidths trade off the smoothing bias (driven by the
``(p+1)``-th derivative of the distribution function at the cutoff) against the variance.
This function reports, for each of four targets, the estimated MSE-optimal bandwidth
together with the variance and squared-bias constants behind it: the left density, the
right density, their difference and their sum. Pilot estimates of these constants use
preliminary bandwidths as in `rddensity`. Regularisation keeps a minimum number of
observations and of distinct values within the bandwidth. The `:comb` choice of
[`rd_density_test`](@ref) combines these rows. Calling this function directly is
useful to report the bandwidths or to fix them across specifications.

# Arguments
- `data::AbstractDataFrame` and `running::Symbol`, or a vector `x`: the running
  variable. Missing values are dropped.

# Keywords
- `cutoff::Real=0.0`: the RD threshold. It must lie strictly inside the range of the
  data.
- `p::Integer=2`: order of the local polynomial for the distribution function (1–7).
- `kernel=:triangular`: `:triangular`, `:epanechnikov` or `:uniform`.
- `fitselect=:unrestricted`: `:unrestricted` (separate fits on each side) or
  `:restricted` (density constrained to be continuous at the cutoff, derivatives may
  differ).
- `vce=:jackknife`: `:jackknife` or `:plugin` variance estimator used in the constants.
- `masspoints::Bool=true`: account for repeated values of the running variable.
- `regularize::Bool=true`: enforce the minimum counts below.
- `nlocalmin::Integer=20+p+1`: minimum number of observations within the bandwidth on
  each side.
- `nuniquemin::Integer=20+p+1`: minimum number of distinct values within the bandwidth
  on each side.

# Returns
- `DataFrame` with columns `side` (`"left"`, `"right"`, `"diff"`, `"sum"`), `bandwidth`,
  `variance` and `bias_squared`.

# Examples
```julia
using DrSnow, CSV, DataFrames
senate = CSV.read(pkgdir(DrSnow, "test", "validation", "rd", "senate.csv"), DataFrame)
bw = rd_density_bandwidth(senate, :margin)
rd_density_test(senate, :margin; h=(bw.bandwidth[1], bw.bandwidth[2]))
```

# References
- Cattaneo, M. D., Jansson, M., & Ma, X. (2020). Simple local polynomial density
  estimators. *Journal of the American Statistical Association*, 115(531), 1449–1455.
- Cattaneo, M. D., Jansson, M., & Ma, X. (2018). Manipulation testing based on density
  discontinuity. *The Stata Journal*, 18(1), 234–261.
"""
function rd_density_bandwidth(x::AbstractVector; cutoff::Real=0.0, p::Integer=2,
                              kernel=:triangular, fitselect=:unrestricted, vce=:jackknife,
                              masspoints::Bool=true, regularize::Bool=true,
                              nlocalmin::Integer=20 + p + 1,
                              nuniquemin::Integer=20 + p + 1)
    _, k, fs, v, _ = _rd_density_opts(Int(p), nothing, kernel, fitselect, vce, :comb)
    xs = _rd_density_x(x, Float64(cutoff)) .- cutoff
    hn = _rd_density_bw_table(xs, Int(p), k, fs, v, regularize, Int(nlocalmin),
                              Int(nuniquemin), masspoints)
    return DataFrame(side=["left", "right", "diff", "sum"], bandwidth=hn[:, 1],
                     variance=hn[:, 2], bias_squared=hn[:, 3])
end

function rd_density_bandwidth(data::AbstractDataFrame, running::Symbol; kwargs...)
    require_columns(data, [running]; context="rd_density_bandwidth")
    return rd_density_bandwidth(data[!, running]; kwargs...)
end

"""
    rd_density_test(data, running; cutoff=0.0, p=2, q=p+1, kernel=:triangular,
                    fitselect=:unrestricted, vce=:jackknife, h=nothing, bwselect=:comb,
                    masspoints=true, regularize=true, nlocalmin=20+p+1,
                    nuniquemin=20+p+1, binomial=true, binomial_windows=10,
                    binomial_n=nothing, binomial_width=nothing,
                    binomial_width_step=nothing, binomial_n_step=nothing,
                    binomial_p=0.5) -> DiagnosticTest
    rd_density_test(x::AbstractVector; kwargs...) -> DiagnosticTest

Manipulation test of Cattaneo, Jansson and Ma (2020): a robust bias-corrected test of
continuity of the density of the running variable at the cutoff, equivalent to
`rddensity` (R/Stata).

If units can manipulate their score precisely, for example to qualify for a benefit,
those just below the cutoff move above it. The density of the running variable then
jumps at the cutoff, and units on the two sides are no longer comparable. McCrary (2008)
proposed testing ``H_0: f(c^+) = f(c^-)``, where ``f`` is the density of ``X``, as an
implication of the absence of such sorting. The test here uses the local polynomial
density estimator of Cattaneo, Jansson and Ma (2020). A polynomial of order `p` is fitted
locally to the empirical distribution function on each side of the cutoff, and its
first derivative at ``c`` estimates the one-sided density. No pre-binning is needed,
and the estimator adapts automatically to the boundary. `fitselect = :restricted`
imposes a common density (but not common higher derivatives) on both sides, that is,
it imposes the null hypothesis in estimation.

The reported statistic is the robust bias-corrected ``t`` statistic
``(\\hat f_q(c^+) - \\hat f_q(c^-))/\\widehat{\\text{se}}``: bandwidths are selected to be
MSE-optimal for order `p`, and the statistic is computed with order `q = p + 1`, with the
jackknife (default) or plug-in standard error and a standard normal reference
distribution. The conventional order-`p` statistic is also stored. As a complement,
`details.binomial` holds exact binomial tests of the share of observations above the
cutoff in a sequence of small windows around it. If assignment is as good as random
near the cutoff, that share should be close to `binomial_p` (Cattaneo, Titiunik &
Vazquez-Bare 2017).

A rejection is evidence consistent with sorting around the cutoff. A non-rejection does
**not** establish that the running variable was not manipulated: power is limited near
the cutoff, and manipulation that leaves the density continuous (for example, sorting
in both directions that offsets in the aggregate) cannot be detected. The test checks an
implication of the design, not the continuity of the potential-outcome regressions,
which is untestable. Report it with covariate balance tests
([`rd_covariate_balance`](@ref)) and an RD plot. The original McCrary test is
available as [`rd_mccrary_test`](@ref).

# Arguments
- `data::AbstractDataFrame` and `running::Symbol`, or a vector `x`: the running
  variable. Missing and `NaN` values are dropped.

# Keywords
- `cutoff::Real=0.0`: the RD threshold. It must lie strictly inside the range of the
  data.
- `p::Integer=2`: order of the local polynomial for the distribution function (1–7);
  `p = 2` gives a locally linear density estimate.
- `q::Union{Nothing,Integer}=nothing`: order used for the bias-corrected statistic, by
  default `p + 1`.
- `kernel=:triangular`: `:triangular`, `:epanechnikov` or `:uniform`.
- `fitselect=:unrestricted`: `:unrestricted` (separate fits on each side) or
  `:restricted` (density constrained to be continuous at the cutoff).
- `vce=:jackknife`: `:jackknife` or `:plugin` standard error.
- `h=nothing`: bandwidth, as a scalar or a `(left, right)` pair; it is selected with
  `bwselect` when `nothing`.
- `bwselect=:comb`: `:each` (side-specific), `:diff` (optimal for the difference),
  `:sum`, or `:comb` (per side, the median of `each`, `diff` and `sum` in the
  unrestricted model; the minimum of `diff` and `sum` in the restricted model).
- `masspoints::Bool=true`, `regularize::Bool=true`, `nlocalmin`, `nuniquemin`: as in
  [`rd_density_bandwidth`](@ref).
- `binomial::Bool=true`: compute the window binomial tests.
- `binomial_windows::Integer=10`: number of windows.
- `binomial_n=nothing`: the first window is the smallest symmetric window with at
  least this many observations on each side (default 20).
- `binomial_width=nothing`: half-width of the first window (scalar or `(left, right)`),
  used instead of `binomial_n` when given.
- `binomial_width_step=nothing`, `binomial_n_step=nothing`: increments between windows,
  in width or in observations. By default the windows grow evenly up to the selected
  bandwidths.
- `binomial_p::Real=0.5`: probability of being above the cutoff under the null.

# Returns
- `DiagnosticTest` whose statistic and p-value are the robust bias-corrected ``t``
  statistic and its two-sided normal p-value for the chosen `vce`. `details` holds
  `f_left`, `f_right`, `f_diff` (density estimates of order `q`), `se_left`,
  `se_right`, `se_diff`, `t_jackknife`, `p_jackknife`, `t_plugin`, `p_plugin`,
  `h_left`, `h_right`, `bwselect`, `n_left`, `n_right`, `n_eff_left`, `n_eff_right`
  (observations within the bandwidths), `p`, `q`, `kernel`, `fitselect`, `vce`, `cutoff`,
  `conventional` (the order-`p` statistic) and `binomial` (a `DataFrame` of window
  tests, or `nothing`).

# Examples
```julia
using DrSnow, CSV, DataFrames
senate = CSV.read(pkgdir(DrSnow, "test", "validation", "rd", "senate.csv"), DataFrame)
t = rd_density_test(senate, :margin)
t.statistic, t.pvalue
first(t.details.binomial, 3)
```

# References
- McCrary, J. (2008). Manipulation of the running variable in the regression
  discontinuity design: A density test. *Journal of Econometrics*, 142(2), 698–714.
- Cattaneo, M. D., Jansson, M., & Ma, X. (2020). Simple local polynomial density
  estimators. *Journal of the American Statistical Association*, 115(531), 1449–1455.
- Cattaneo, M. D., Jansson, M., & Ma, X. (2018). Manipulation testing based on density
  discontinuity. *The Stata Journal*, 18(1), 234–261.
- Cattaneo, M. D., Titiunik, R., & Vazquez-Bare, G. (2017). Comparing inference
  approaches for RD designs: A reexamination of the effect of Head Start on child
  mortality. *Journal of Policy Analysis and Management*, 36(3), 643–681.
- Cattaneo, M. D., Idrobo, N., & Titiunik, R. (2020). *A Practical Introduction to
  Regression Discontinuity Designs: Foundations*. Cambridge University Press.
"""
function rd_density_test(x::AbstractVector; cutoff::Real=0.0, p::Integer=2,
                         q::Union{Nothing,Integer}=nothing, kernel=:triangular,
                         fitselect=:unrestricted, vce=:jackknife, h=nothing,
                         bwselect=:comb, masspoints::Bool=true, regularize::Bool=true,
                         nlocalmin::Integer=20 + p + 1, nuniquemin::Integer=20 + p + 1,
                         binomial::Bool=true, binomial_windows::Integer=10,
                         binomial_n=nothing, binomial_width=nothing,
                         binomial_width_step=nothing, binomial_n_step=nothing,
                         binomial_p::Real=0.5)
    p = Int(p)
    q, k, fs, v, bs = _rd_density_opts(p, q, kernel, fitselect, vce, bwselect)
    c = Float64(cutoff)
    X = _rd_density_x(x, c) .- c
    N = length(X)
    Nl = count(<(0), X); Nr = N - Nl
    if h === nothing
        hn = _rd_density_bw_table(X, p, k, fs, v, regularize, Int(nlocalmin),
                                  Int(nuniquemin), masspoints)
        bwl = hn[:, 1]
        if fs === :unrestricted
            hl, hr = bs === :each ? (bwl[1], bwl[2]) :
                     bs === :diff ? (bwl[3], bwl[3]) :
                     bs === :sum ? (bwl[4], bwl[4]) :
                     (median([bwl[1], bwl[3], bwl[4]]), median([bwl[2], bwl[3], bwl[4]]))
        else
            hh = bs === :diff ? bwl[3] : bs === :sum ? bwl[4] : min(bwl[3], bwl[4])
            hl = hr = hh
        end
        bw_method = bs
    else
        hl, hr = _rd_pair(h)
        (hl > 0 && hr > 0) || throw(ArgumentError("bandwidth must be positive"))
        (fs === :restricted && hl != hr) && throw(ArgumentError(
            "bandwidths must be equal in the restricted model"))
        bw_method = :manual
    end
    D = _rd_density_setup(X, masspoints)
    inw = (X .>= -hl) .& (X .<= hr)
    Xh, Yh = X[inw], D.Y[inw]
    Nlh = count(<(0), Xh); Nrh = length(Xh) - Nlh
    fVq = _rd_density_fV(Yh, Xh, N, Nlh, Nrh, hl, hr, q, 1, k, fs, v, D.mflag)
    t_jk = fVq[3, 1] / sqrt(fVq[3, 2])
    t_asy = fVq[3, 1] / sqrt(fVq[3, 3])
    p_jk = two_sided_pvalue(t_jk)
    p_asy = two_sided_pvalue(t_asy)
    fVp = _rd_density_fV(Yh, Xh, N, Nlh, Nrh, hl, hr, p, 1, k, fs, v, D.mflag)
    col = v === :jackknife ? 2 : 3
    t_conv = fVp[3, 1] / sqrt(fVp[3, col])
    conventional = (f_left=fVp[1, 1], f_right=fVp[2, 1], f_diff=fVp[3, 1],
                    se_diff=sqrt(fVp[3, col]), t=t_conv, pvalue=two_sided_pvalue(t_conv))
    bino = binomial ? _rd_binomial_windows(X, hl, hr; binoW=binomial_width,
                                            binoN=binomial_n,
                                            binoWStep=binomial_width_step,
                                            binoNStep=binomial_n_step,
                                            binoNW=Int(binomial_windows),
                                            binoP=binomial_p) : nothing
    stat, pv = v === :jackknife ? (t_jk, p_jk) : (t_asy, p_asy)
    details = (f_left=fVq[1, 1], f_right=fVq[2, 1], f_diff=fVq[3, 1],
               se_left=sqrt(fVq[1, col]), se_right=sqrt(fVq[2, col]),
               se_diff=sqrt(fVq[3, col]), t_jackknife=t_jk, p_jackknife=p_jk,
               t_plugin=t_asy, p_plugin=p_asy, h_left=hl, h_right=hr,
               bwselect=bw_method, n_left=Nl, n_right=Nr, n_eff_left=Nlh,
               n_eff_right=Nrh, p=p, q=q, kernel=k, fitselect=fs, vce=v, cutoff=c,
               conventional=conventional, binomial=bino)
    note = "A density discontinuity is consistent with sorting of units around the " *
           "cutoff. Non-rejection does not show that the running variable was not " *
           "manipulated: power is limited near the cutoff and manipulation that " *
           "leaves the density continuous is not detectable. This checks an " *
           "implication of the design; it does not test the continuity assumption " *
           "for potential outcomes."
    if isnan(pv)
        note = "The variance estimate was unavailable (too few observations within " *
               "the bandwidth), so the test could not be computed. " * note
    end
    method = "Local polynomial density test (Cattaneo, Jansson & Ma 2020), robust " *
             "bias-corrected, p = $p, q = $q, $(v) standard error, h = " *
             @sprintf("(%.4g, %.4g)", hl, hr)
    return DiagnosticTest("RD density discontinuity test",
                          "the density of the running variable is continuous at the " *
                          "cutoff", stat, pv; method=method, note=note, details=details)
end

function rd_density_test(data::AbstractDataFrame, running::Symbol; kwargs...)
    require_columns(data, [running]; context="rd_density_test")
    return rd_density_test(data[!, running]; kwargs...)
end
