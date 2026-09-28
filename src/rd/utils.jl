# Internal helpers for the regression discontinuity area.
#
# The numerical routines mirror the R package `rdrobust` (Calonico, Cattaneo, Farrell &
# Titiunik, version 4.0.0) step by step so that DrSnow reproduces its point estimates,
# standard errors and bandwidths to floating-point accuracy. Every non-exported name in
# this area carries the `_rd_` prefix because DrSnow is a flat module.

# ---------------------------------------------------------------------------------------
# Option parsing
# ---------------------------------------------------------------------------------------

function _rd_kernel(k)
    s = lowercase(string(k))
    s in ("tri", "triangular") && return :triangular
    s in ("epa", "epanechnikov") && return :epanechnikov
    s in ("uni", "uniform") && return :uniform
    throw(ArgumentError("kernel must be :triangular, :epanechnikov or :uniform " *
                        "(got $(repr(k)))"))
end

const _RD_BWSELECT = (:mserd, :msetwo, :msesum, :msecomb1, :msecomb2,
                      :cerrd, :certwo, :cersum, :cercomb1, :cercomb2)

function _rd_bwselect(b)
    s = Symbol(lowercase(string(b)))
    s in _RD_BWSELECT && return s
    throw(ArgumentError("bwselect must be one of $(join(_RD_BWSELECT, ", ")) " *
                        "(got $(repr(b)))"))
end

function _rd_vce(v)
    s = Symbol(lowercase(string(v)))
    s in (:nn, :hc0, :hc1, :hc2, :hc3, :cr1, :cr2, :cr3) && return s
    throw(ArgumentError("vce must be :nn, :hc0, :hc1, :hc2, :hc3, :cr1, :cr2 or :cr3 " *
                        "(got $(repr(v)))"))
end

function _rd_masspoints(m)
    m === false && return :off
    s = Symbol(lowercase(string(m)))
    s in (:adjust, :check, :off) && return s
    throw(ArgumentError("masspoints must be :adjust, :check or :off (got $(repr(m)))"))
end

# Resolve the variance estimator actually used, following rdrobust 4.0: with a cluster
# variable NN/HC0/HC1 become CR1 and HC2/HC3 become CR2/CR3 (with a warning for the HC
# options); CR options require a cluster variable.
function _rd_resolve_vce(vce::Symbol, has_cluster::Bool)
    if has_cluster
        vce in (:cr1, :cr2, :cr3) && return vce
        vce === :nn && return :cr1
        to = vce in (:hc0, :hc1) ? :cr1 : vce === :hc2 ? :cr2 : :cr3
        @warn "vce = :$vce is not a cluster-robust option; using vce = :$to"
        return to
    else
        vce in (:cr1, :cr2, :cr3) && throw(ArgumentError(
            "vce = :$vce requires a `cluster` variable"))
        return vce
    end
end

# ---------------------------------------------------------------------------------------
# Kernels and design matrices
# ---------------------------------------------------------------------------------------

"""Kernel weights `K((x - c)/h)/h` exactly as `rdrobust_kweight`."""
function _rd_kweight(x::AbstractVector{<:Real}, c::Real, h::Real, kernel::Symbol)
    w = Vector{Float64}(undef, length(x))
    @inbounds for i in eachindex(x)
        u = (x[i] - c) / h
        inside = abs(u) <= 1
        w[i] = if kernel === :epanechnikov
            (0.75 * (1 - u^2) * inside) / h
        elseif kernel === :uniform
            (0.5 * inside) / h
        else
            ((1 - abs(u)) * inside) / h
        end
    end
    return w
end

"""Polynomial basis `[1 u u^2 … u^p]` built by repeated multiplication (as rdrobust)."""
function _rd_vander(u::AbstractVector{<:Real}, p::Integer)
    n = length(u)
    out = ones(Float64, n, p + 1)
    for j in 2:(p + 1), i in 1:n
        out[i, j] = out[i, j - 1] * u[i]
    end
    return out
end

"""`(X'X)⁻¹` via Cholesky, falling back to the Moore–Penrose inverse (as `qrXXinv`)."""
function _rd_xxinv(X::AbstractMatrix)
    G = Symmetric(X' * X)
    F = cholesky(G; check=false)
    issuccess(F) && return Matrix(inv(F))
    return _rd_ginv(Matrix(G))
end

"""Moore–Penrose inverse with the relative tolerance rule of `MASS::ginv`."""
function _rd_ginv(A::AbstractMatrix; tol::Real=1e-20)
    F = svd(A)
    isempty(F.S) && return zeros(size(A, 2), size(A, 1))
    keep = F.S .> max(tol * F.S[1], 0.0)
    return F.V[:, keep] * Diagonal(1 ./ F.S[keep]) * F.U[:, keep]'
end

# ---------------------------------------------------------------------------------------
# Sample utilities
# ---------------------------------------------------------------------------------------

"""Run-length duplicates of a sorted vector: count of the tie group and position in it."""
function _rd_dups(x::AbstractVector{<:Real})
    n = length(x)
    dups = zeros(Int, n)
    dupsid = zeros(Int, n)
    i = 1
    while i <= n
        j = i
        while j < n && x[j + 1] == x[i]
            j += 1
        end
        len = j - i + 1
        for k in i:j
            dups[k] = len
            dupsid[k] = k - i + 1
        end
        i = j + 1
    end
    return dups, dupsid
end

"""Index sets of observations sharing a cluster identifier."""
function _rd_cluster_groups(C::AbstractVector)
    groups = Dict{eltype(C),Vector{Int}}()
    for (i, g) in enumerate(C)
        push!(get!(groups, g, Int[]), i)
    end
    return collect(values(groups))
end

"""R's `quantile(x, prob, type = 2)`."""
function _rd_quantile_type2(x::AbstractVector{<:Real}, prob::Real)
    xs = sort(x)
    n = length(xs)
    np = n * prob
    j = floor(Int, np)
    g = np - j
    if g > 0
        return float(xs[min(j + 1, n)])
    else
        return (xs[max(j, 1)] + xs[min(j + 1, n)]) / 2
    end
end

"""R's `quantile(x, probs, type = 7)` (the default), using R's interpolation formula."""
function _rd_quantile_type7(x::AbstractVector{<:Real}, probs::AbstractVector{<:Real})
    xs = sort(x)
    n = length(xs)
    out = similar(probs, Float64)
    for (k, pr) in enumerate(probs)
        index = 1 + max(n - 1, 0) * pr
        lo = floor(Int, index)
        hi = ceil(Int, index)
        qs = float(xs[lo])
        if index > lo && xs[hi] != qs
            h = index - lo
            qs = (1 - h) * qs + h * xs[hi]
        end
        out[k] = qs
    end
    return out
end

"""R's `seq(from, to, by)` for positive `by`."""
function _rd_seq_by(from::Real, to::Real, by::Real)
    del = to - from
    (del == 0 && to == 0) && return [float(to)]
    n = del / by
    dd = abs(del) / max(abs(to), abs(from))
    dd < 100 * eps(Float64) && return [float(from)]
    m = floor(Int, n + 1e-10)
    return [min(from + k * by, to) for k in 0:m]
end

"""R's `findInterval(x, vec, rightmost.closed = TRUE)`."""
function _rd_find_interval(x::Real, vec::AbstractVector{<:Real})
    N = length(vec)
    i = searchsortedlast(vec, x)
    if i == N && x == vec[N]
        i = N - 1
    end
    return i
end

"""Sequentially drop covariates that are (numerically) collinear with earlier ones."""
function _rd_drop_collinear(Z::Matrix{Float64}, names::Vector{String}; tol=1e-7)
    k = size(Z, 2)
    # rdrobust orders covariates by the length of their names before checking.
    order = sortperm(length.(names); alg=MergeSort)
    kept = Int[]
    for j in order
        z = Z[:, j]
        nz = norm(z)
        if isempty(kept)
            r = z
        else
            A = Z[:, kept]
            r = z - A * (A \ z)
        end
        if nz > 0 && norm(r) > tol * nz
            push!(kept, j)
        end
    end
    sort!(kept)
    return kept
end

# ---------------------------------------------------------------------------------------
# Residuals and variance components
# ---------------------------------------------------------------------------------------

"""
Nearest-neighbour residuals (Calonico, Cattaneo & Titiunik 2014, eq. for σ̂²) for each
column of `D`, for a running variable `X` sorted ascending. Mirrors `rdrobust_res`
including its treatment of ties (`dups`, `dupsid`).
"""
function _rd_nn_residuals(X::AbstractVector{Float64}, D::AbstractMatrix{Float64},
                          matches::Int, dups::AbstractVector{Int},
                          dupsid::AbstractVector{Int})
    n = length(X)
    k = size(D, 2)
    res = Matrix{Float64}(undef, n, k)
    target = min(matches, n - 1)
    @inbounds for pos in 1:n
        rpos = dups[pos] - dupsid[pos]
        lpos = dupsid[pos] - 1
        while lpos + rpos < target
            if pos - lpos - 1 <= 0
                rpos += dups[pos + rpos + 1]
            elseif pos + rpos + 1 > n
                lpos += dups[pos - lpos - 1]
            else
                dl = X[pos] - X[pos - lpos - 1]
                dr = X[pos + rpos + 1] - X[pos]
                if dl > dr
                    rpos += dups[pos + rpos + 1]
                elseif dl < dr
                    lpos += dups[pos - lpos - 1]
                else
                    rpos += dups[pos + rpos + 1]
                    lpos += dups[pos - lpos - 1]
                end
            end
        end
        lo = max(1, pos - lpos)
        hi = min(n, pos + rpos)
        Ji = hi - lo
        f = sqrt(Ji / (Ji + 1))
        for col in 1:k
            s = 0.0
            for j in lo:hi
                s += D[j, col]
            end
            s -= D[pos, col]
            res[pos, col] = f * (D[pos, col] - s / Ji)
        end
    end
    return res
end

"""
Residuals used in the sandwich variance: NN residuals, or (HC0–HC3 / CR1) weighted
regression residuals `D - R β` with the HC scaling of `rdrobust_res`.
"""
function _rd_residuals(X, D, R, beta, invG, w, vce::Symbol, nnmatch::Int, dups,
                       dupsid, d::Int, has_cluster::Bool)
    if vce === :nn
        return _rd_nn_residuals(X, D, nnmatch, dups, dupsid)
    end
    n = size(D, 1)
    e = D .- R * beta
    if vce in (:hc0, :cr1, :cr2, :cr3)
        return e
    elseif vce === :hc1
        return has_cluster ? e : sqrt(n / (n - d)) .* e
    else
        hii = vec(sum((R * invG) .* (R .* w); dims=2))
        s = vce === :hc2 ? sqrt.(1 ./ max.(1 .- hii, 1e-8)) : 1 ./ max.(1 .- hii, 1e-8)
        return s .* e
    end
end

"""
Meat matrix `Σ (a_i RX_i)(b_i RX_i)'` (heteroskedasticity-robust) or its cluster analogue
with the CR1 small-sample factor `(n-1)/(n-k) · G/(G-1)` of `rdrobust_vce`.
`a` and `b` are combined residuals `res * s`.
"""
function _rd_meat(RX::AbstractMatrix, a::AbstractVector, b::AbstractVector,
                  groups::Union{Nothing,Vector{Vector{Int}}}, k_df::Int)
    if groups === nothing
        return RX' * (RX .* (a .* b))
    end
    k = size(RX, 2)
    n = length(a)
    g = length(groups)
    g > 1 || throw(ArgumentError("cluster-robust variance needs at least two clusters " *
                                 "on each side of the cutoff within the bandwidth"))
    wgt = ((n - 1) / (n - k_df)) * (g / (g - 1))
    M = zeros(k, k)
    for idx in groups
        sa = RX[idx, :]' * a[idx]
        sb = RX[idx, :]' * b[idx]
        M .+= sa * sb'
    end
    return wgt .* M
end

_rd_meat(RX, a, groups, k_df) = _rd_meat(RX, a, a, groups, k_df)

# ---------------------------------------------------------------------------------------
# CR2 / CR3 cluster-robust meat matrices (rdrobust 4.0 `rdrobust_vce` crv modes and
# `.rdrobust_vce_qq_cluster`). Cluster scores are linear in the residual, so the
# combined residuals `a = res * s_a`, `b = res * s_b` give Σ_g score_g(a) score_g(b)'.
# ---------------------------------------------------------------------------------------

_rd_try_inv(A; fallback) = try
    B = inv(A)
    all(isfinite, B) ? B : fallback(A)
catch err
    err isa Union{SingularException,LinearAlgebra.LAPACKException} || rethrow()
    fallback(A)
end

"""
Meat for a local polynomial fit with design `R`, kernel weights `w` and
`invG = (R'WR)⁻¹`: CR3 (jackknife-type, score `G (G - L_g)⁻¹ u_g`) or CR2
(Bell–McCaffrey-type, `(I - H_gg)^{-1/2}` adjustment) as in rdrobust.
"""
function _rd_meat_crv(R::AbstractMatrix, w::AbstractVector, invG::AbstractMatrix,
                      a::AbstractVector, b::AbstractVector,
                      groups::Vector{Vector{Int}}, crv2::Bool)
    k = size(R, 2)
    M = zeros(k, k)
    if crv2
        F = cholesky(Symmetric(Matrix(invG)); check=false)
        ok = issuccess(F)
        Ch = ok ? Matrix(F.U) : zeros(k, k)
    else
        G = _rd_try_inv(invG; fallback=A -> _rd_ginv(A; tol=sqrt(eps(Float64))))
    end
    for idx in groups
        Rg = R[idx, :]
        wg = w[idx]
        Lg = Rg' * (Rg .* wg)
        ua = Rg' * (wg .* a[idx])
        ub = Rg' * (wg .* b[idx])
        if crv2
            if ok
                E = eigen(Symmetric(Ch * Lg * Ch'))
                s2 = max.(E.values, 0.0)
                cc = [v < 1e-14 ? 0.0 : (1 / sqrt(max(1 - v, 1e-8)) - 1) / v for v in s2]
                T = I + Lg * Ch' * E.vectors * Diagonal(cc) * E.vectors' * Ch
            else
                T = Matrix(1.0I, k, k)
            end
        else
            Mg = _rd_try_inv(G .- Lg; fallback=A -> _rd_ginv(A; tol=sqrt(eps(Float64))))
            T = I + Lg * Mg
        end
        M .+= (T * ua) * (T * ub)'
    end
    return M
end

"""
Meat of the robust bias-corrected variance, `Σ_g s_g(a) s_g(b)'`, with CR2/CR3 cluster
scores built from the bias-correction design `Q` (rdrobust `.rdrobust_vce_qq_cluster`).
"""
function _rd_meat_qq(Q::AbstractMatrix, R_q::AbstractMatrix, W_b::AbstractVector,
                     invG_q::AbstractMatrix, a::AbstractVector, b::AbstractVector,
                     groups::Vector{Vector{Int}}, crv2::Bool)
    k = size(Q, 2)
    kR = size(R_q, 2)
    M = zeros(k, k)
    G_q = crv2 ? zeros(0, 0) :
          _rd_try_inv(invG_q; fallback=A -> _rd_ginv(A; tol=sqrt(eps(Float64))))
    for idx in groups
        Qg = Q[idx, :]
        Rg = R_q[idx, :]
        wg = W_b[idx]
        Lg = Rg' * (Rg .* wg)
        Pg = Qg' * Rg
        simple = false
        local adjM
        if crv2
            E = eigen(Symmetric(Lg))
            sig = sqrt.(max.(E.values, 0.0))
            keep = sig .> maximum(sig) * 1e-10
            if !any(keep)
                simple = true
            else
                Vr = E.vectors[:, keep]
                sg = sig[keep]
                Tm = Diagonal(sg) * (Vr' * invG_q * Vr) * Diagonal(sg)
                Et = eigen(Symmetric((Tm .+ Tm') ./ 2))
                tau = min.(max.(Et.values, 0.0), 1 - 1e-10)
                gam = 1 ./ sqrt.(1 .- tau) .- 1
                A = Vr * Diagonal(1 ./ sg) * Et.vectors
                adjM = A * Diagonal(gam) * A'
            end
        else
            adjM = _rd_try_inv(G_q .- Lg; fallback=A -> zeros(kR, kR))
        end
        score(r) = simple ? Qg' * r[idx] :
                   Qg' * r[idx] .+ Pg * (adjM * (Rg' * (wg .* r[idx])))
        M .+= score(a) * score(b)'
    end
    return M
end

"""Meat dispatch: CR2/CR3 when clustered with those options, otherwise HC/CR1."""
function _rd_vmeat(vce::Symbol, R::AbstractMatrix, w::AbstractVector,
                   invG::AbstractMatrix, a::AbstractVector, b::AbstractVector,
                   groups, k_df::Int)
    if groups !== nothing && vce in (:cr2, :cr3)
        return _rd_meat_crv(R, w, invG, a, b, groups, vce === :cr2)
    end
    return _rd_meat(R .* w, a, b, groups, k_df)
end
