# Test statistics for randomization inference.
#
# A statistic is evaluated as `_ri_eval(stat, prep, z, ctx)` where
# `prep = _ri_prepare(stat, y, ctx)` holds the (adjusted) outcome and anything that
# does not depend on the assignment (ranks, sort orders), and `ctx` describes the
# design (strata, variance clusters, covariates). Every statistic is invariant to
# the order of units.

abstract type _ri_Statistic end

struct _ri_DiffMeans <: _ri_Statistic end
struct _ri_Studentized <: _ri_Statistic end
struct _ri_RankSum <: _ri_Statistic end
struct _ri_KS <: _ri_Statistic end
struct _ri_Lin <: _ri_Statistic
    studentized::Bool
end
struct _ri_UserStat{F} <: _ri_Statistic
    f::F
end

_ri_name(::_ri_DiffMeans) = "difference in means"
_ri_name(::_ri_Studentized) = "studentized difference in means"
_ri_name(::_ri_RankSum) = "rank-sum (difference in mean normalized ranks)"
_ri_name(::_ri_KS) = "Kolmogorov–Smirnov"
_ri_name(s::_ri_Lin) = s.studentized ? "regression-adjusted t (Lin, robust SE)" :
                       "regression-adjusted difference (Lin)"
_ri_name(::_ri_UserStat) = "user-supplied statistic"

# Signed statistics are centred near zero under the null and support one- and
# two-sided alternatives; unsigned statistics (KS) are upper-tail only.
_ri_signed(::_ri_Statistic) = true
_ri_signed(::_ri_KS) = false

# Linear in the outcome vector for a fixed assignment: enables exact CI inversion.
_ri_linear(::_ri_Statistic) = false
_ri_linear(::_ri_DiffMeans) = true
_ri_linear(s::_ri_Lin) = !s.studentized

_ri_uses_covariates(::_ri_Statistic) = false
_ri_uses_covariates(::_ri_Lin) = true

function _ri_parse_statistic(s)
    s isa _ri_Statistic && return s
    s isa Function && return _ri_UserStat(s)
    s isa Symbol || throw(ArgumentError("statistic must be a Symbol or a function"))
    s in (:diff_means, :difference_in_means) && return _ri_DiffMeans()
    s === :studentized && return _ri_Studentized()
    s in (:rank_sum, :wilcoxon) && return _ri_RankSum()
    s in (:ks, :kolmogorov_smirnov) && return _ri_KS()
    s === :lin && return _ri_Lin(false)
    s === :lin_studentized && return _ri_Lin(true)
    throw(ArgumentError("unknown statistic :$s; use :diff_means, :studentized, " *
                        ":rank_sum, :ks, :lin, :lin_studentized or a function"))
end

# ---------------------------------------------------------------------------------
# Design context
# ---------------------------------------------------------------------------------

struct _ri_Context
    n::Int
    strata::Vector{Vector{Int}}          # unit groups (a single group if unstratified)
    cluster_of::Vector{Int}              # variance unit of each unit
    strata_clusters::Vector{Vector{Int}} # variance units present in each stratum
    nclusters::Int
    cluster_rep::Vector{Int}             # one representative unit per variance unit
    clustered::Bool
    X::Matrix{Float64}                   # covariates, centred at the sample mean
end

function _ri_context(m::AssignmentMechanism, X::AbstractMatrix)
    n = n_units(m)
    st = _ri_strata(m)
    strata = st === nothing ? [collect(1:n)] : [collect(g) for g in st]
    cl = _ri_clusters(m)
    cluster_of = collect(1:n)
    if cl !== nothing
        for (c, g) in enumerate(cl)
            cluster_of[g] .= c
        end
    end
    nclusters = cl === nothing ? n : length(cl)
    sc = [unique(cluster_of[g]) for g in strata]
    rep = zeros(Int, nclusters)
    for i in n:-1:1
        rep[cluster_of[i]] = i
    end
    Xc = Matrix{Float64}(X)
    if size(Xc, 2) > 0
        Xc = Xc .- mean(Xc; dims=1)
    end
    return _ri_Context(n, strata, cluster_of, sc, nclusters, rep, cl !== nothing, Xc)
end

# ---------------------------------------------------------------------------------
# Preparation (assignment-independent work)
# ---------------------------------------------------------------------------------

_ri_prepare(::_ri_Statistic, y::AbstractVector, ctx) = Vector{Float64}(y)

function _ri_prepare(::_ri_RankSum, y::AbstractVector, ctx)
    r = zeros(length(y))
    for g in ctx.strata
        r[g] .= tiedrank(view(y, g)) ./ (length(g) + 1)
    end
    return r
end

function _ri_prepare(::_ri_KS, y::AbstractVector, ctx)
    p = sortperm(y)
    return (y=Vector{Float64}(y), order=p)
end

# ---------------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------------

# Stratum-size-weighted difference in means (plain difference with one stratum).
function _ri_stratified_dim(y::AbstractVector, z::AbstractVector{Bool}, ctx)
    tot = 0.0
    wsum = 0
    for g in ctx.strata
        s1 = 0.0; s0 = 0.0; n1 = 0; n0 = 0
        @inbounds for i in g
            if z[i]
                s1 += y[i]; n1 += 1
            else
                s0 += y[i]; n0 += 1
            end
        end
        if n1 > 0 && n0 > 0
            tot += length(g) * (s1 / n1 - s0 / n0)
            wsum += length(g)
        end
    end
    return wsum == 0 ? NaN : tot / wsum
end

_ri_eval(::_ri_DiffMeans, y, z, ctx) = _ri_stratified_dim(y, z, ctx)
_ri_eval(::_ri_RankSum, r, z, ctx) = _ri_stratified_dim(r, z, ctx)
_ri_eval(s::_ri_UserStat, y, z, ctx) = float(s.f(y, z))

function _ri_eval(::_ri_KS, p, z, ctx)
    n1 = count(z)
    n0 = length(z) - n1
    (n1 == 0 || n0 == 0) && return NaN
    y = p.y
    o = p.order
    f1 = 0.0; f0 = 0.0; d = 0.0
    k = 1
    n = length(o)
    @inbounds while k <= n
        v = y[o[k]]
        while k <= n && y[o[k]] == v
            if z[o[k]]
                f1 += 1 / n1
            else
                f0 += 1 / n0
            end
            k += 1
        end
        d = max(d, abs(f1 - f0))
    end
    return d
end

# Studentized stratified difference in means (Neyman-type variance on variance
# units; matched-pair variance when every stratum has one unit per arm).
function _ri_eval(::_ri_Studentized, y, z, ctx)
    ntot = 0
    for g in ctx.strata
        n1 = count(i -> z[i], g)
        (n1 > 0 && n1 < length(g)) && (ntot += length(g))
    end
    ntot == 0 && return NaN
    acc = zeros(ctx.nclusters)
    beta = 0.0
    V = 0.0
    pairs_ok = true
    ds = Float64[]
    ws = Float64[]
    all_cells_ok = true
    for (s, g) in enumerate(ctx.strata)
        s1 = 0.0; s0 = 0.0; n1 = 0; n0 = 0
        @inbounds for i in g
            if z[i]
                s1 += y[i]; n1 += 1
            else
                s0 += y[i]; n0 += 1
            end
        end
        (n1 == 0 || n0 == 0) && continue
        m1 = s1 / n1; m0 = s0 / n0
        w = length(g) / ntot
        beta += w * (m1 - m0)
        push!(ds, m1 - m0); push!(ws, w)
        @inbounds for i in g
            c = ctx.cluster_of[i]
            acc[c] += z[i] ? (y[i] - m1) / n1 : (y[i] - m0) / n0
        end
        # per-arm sums of squared cluster contributions and cluster counts
        g1 = 0; g0 = 0; q1 = 0.0; q0 = 0.0
        @inbounds for c in ctx.strata_clusters[s]
            if z[ctx.cluster_rep[c]]
                g1 += 1; q1 += acc[c]^2
            else
                g0 += 1; q0 += acc[c]^2
            end
            acc[c] = 0.0
        end
        (g1 == 1 && g0 == 1) || (pairs_ok = false)
        if g1 >= 2 && g0 >= 2
            V += w^2 * (g1 / (g1 - 1) * q1 + g0 / (g0 - 1) * q0)
        else
            all_cells_ok = false
        end
    end
    if !all_cells_ok
        (pairs_ok && length(ds) >= 2) || return NaN
        P = length(ds)
        V = P / (P - 1) * sum((ws[p] * ds[p] - beta / P)^2 for p in 1:P)
    end
    return V > 0 ? beta / sqrt(V) : NaN
end

# Lin (2013) regression adjustment: coefficient on z in y ~ strata + z + Xc + z·Xc,
# optionally divided by its HC2 (or cluster-robust CR1) standard error.
function _ri_eval(s::_ri_Lin, y, z, ctx)
    n = ctx.n
    n1 = count(z)
    (n1 == 0 || n1 == n) && return NaN
    p = size(ctx.X, 2)
    S = length(ctx.strata)
    k = S + 1 + 2p
    D = zeros(n, k)
    for (j, g) in enumerate(ctx.strata)
        D[g, j] .= 1.0
    end
    jz = S + 1
    @inbounds for i in 1:n
        zi = z[i] ? 1.0 : 0.0
        D[i, jz] = zi
        for l in 1:p
            D[i, jz + l] = ctx.X[i, l]
            D[i, jz + p + l] = zi * ctx.X[i, l]
        end
    end
    F = qr(D, ColumnNorm())
    Rd = abs.(diag(F.R))
    (isempty(Rd) || minimum(Rd) <= 1e-10 * maximum(Rd)) && return NaN
    beta = F \ y
    s.studentized || return beta[jz]
    XtXi = inv(Symmetric(D' * D))
    a = XtXi[jz, :]
    e = y .- D * beta
    u = D * a                      # influence weights of beta[jz]
    if ctx.clustered
        G = ctx.nclusters
        acc = zeros(G)
        @inbounds for i in 1:n
            acc[ctx.cluster_of[i]] += u[i] * e[i]
        end
        V = G / (G - 1) * (n - 1) / (n - k) * sum(abs2, acc)
    else
        V = 0.0
        @inbounds for i in 1:n
            h = dot(view(D, i, :), XtXi * view(D, i, :))
            h >= 1 - 1e-10 && return NaN
            V += u[i]^2 * e[i]^2 / (1 - h)
        end
    end
    return V > 0 ? beta[jz] / sqrt(V) : NaN
end
