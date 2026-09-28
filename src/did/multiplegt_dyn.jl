# de Chaisemartin & D'Haultfœuille (2026) event-study estimators DID_ℓ for designs
# where the treatment may be non-binary and non-absorbing (units can switch in and
# out, and the treatment can take many values), as implemented by the
# DIDmultiplegtDYN R package (did_multiplegt_dyn). The implementation follows the R
# package step by step (sample construction, influence functions U_g, cohort
# demeaning and degrees-of-freedom corrections in the variance), so estimates and
# standard errors agree with it.
#
# Notation: groups g, periods t (ranks 1..T of the time variable). d_sq = D_{g,1}
# (status quo: first observed treatment); F_g = first period with D_{g,t} ≠ d_sq
# (T+1 if never); T_g = last period at which some group with the same d_sq has not
# switched yet (so a comparison exists); S_g = +1 (switcher in: post-switch average
# treatment above d_sq) or -1 (switcher out).
#
#   DID_ℓ = (1/N_ℓ) Σ_{g: F_g-1+ℓ ≤ T_g} N_{g,F_g-1+ℓ} S_g [ (Y_{g,F_g-1+ℓ} - Y_{g,F_g-1})
#           - mean over not-yet-switchers g' with d_sq(g') = d_sq(g) of that difference ]
#
# computed separately for switchers in and out and combined with weights N1/N0.

"""
    _did_dcdh_panel(...)

Dense `groups × periods` representation after DIDmultiplegtDYN's sample
construction (missing-treatment imputation, removal of groups whose treatment moves
both above and below the status quo, of status-quo classes without variation in the
switching date, and of periods without not-yet-switched comparison groups).
"""
struct _DidDCDHPanel
    Y::Matrix{Float64}          # outcome (NaN = missing)
    D::Matrix{Float64}          # treatment (NaN = missing)
    N::Matrix{Float64}          # N_gt weights (0 when Y or D is missing)
    times::Vector{Int}          # time ranks of the grid columns
    dsq::Vector{Float64}
    tnp::Vector{Int}            # trends_nonparam class code (0 when not used)
    F::Vector{Float64}          # F_g (T_max + 1 for never switchers)
    Tg::Vector{Float64}
    S::Vector{Float64}          # 1 / 0 / NaN (never switchers)
    dfg::Vector{Float64}
    L::Vector{Float64}
    Lpl::Vector{Float64}
    cluster::Vector{Int}        # cluster code per group (group index if unclustered)
    G::Int                      # number of groups used in the scaling (G_XX)
    t_min::Int
    T_max::Int
    group_labels::Vector
    period_labels::Vector
    X::Array{Float64,3}         # controls (groups × periods × K; NaN = missing)
    W::Matrix{Float64}          # raw weights (NaN for cells absent from the data)
    evc::Matrix{Float64}        # original ever-changed indicator (NaN if absent)
end

_dcdh_nanmean(v) = (s = 0.0; c = 0; for x in v; isnan(x) || (s += x; c += 1); end;
                    c == 0 ? NaN : s / c)

function _did_dcdh_build(gkey, tval, y, d, w, cl, tnp=nothing, xc=nothing)
    # --- ranks (dense) of groups and periods --------------------------------------
    n = length(y)
    ug = unique(gkey)
    try
        sort!(ug)
    catch
    end
    gidx = Dict(g => i for (i, g) in enumerate(ug))
    ut = sort(unique(tval))
    tidx = Dict(t => i for (i, t) in enumerate(ut))
    G0, T0 = length(ug), length(ut)
    Y = fill(NaN, G0, T0)
    D = fill(NaN, G0, T0)
    W = zeros(G0, T0)
    P = falses(G0, T0)
    CLg = zeros(Int, G0)
    clseen = falses(G0)
    TNg = zeros(Int, G0)
    tnseen = falses(G0)
    K = xc === nothing ? 0 : size(xc, 2)
    Xf = fill(NaN, G0, T0, K)
    for i in 1:n
        g, t = gidx[gkey[i]], tidx[tval[i]]
        P[g, t] && throw(ArgumentError(
            "did_multiplegt_dyn: group $(gkey[i]) is observed more than once in period " *
            "$(tval[i]); aggregate the data to group × period cells first (use " *
            "`weights` for cell sizes)"))
        P[g, t] = true
        for k in 1:K
            Xf[g, t, k] = xc[i, k]
        end
        Y[g, t] = y[i]
        D[g, t] = d[i]
        W[g, t] = isnan(w[i]) ? 0.0 : w[i]
        if cl !== nothing
            if clseen[g] && CLg[g] != cl[i]
                throw(ArgumentError("did_multiplegt_dyn: the group variable must be " *
                                    "nested within the cluster variable"))
            end
            CLg[g] = cl[i]
            clseen[g] = true
        end
        if tnp !== nothing
            if tnseen[g] && TNg[g] != tnp[i]
                throw(ArgumentError("did_multiplegt_dyn: the trends_nonparam " *
                                    "variables must be constant within groups"))
            end
            TNg[g] = tnp[i]
            tnseen[g] = true
        end
    end
    # groups with no observed outcome or no observed treatment are dropped
    for g in 1:G0
        row = findall(P[g, :])
        (all(t -> isnan(Y[g, t]), row) || all(t -> isnan(D[g, t]), row)) &&
            (P[g, :] .= false)
    end
    # first/last period with observed treatment, status quo d_sq
    mind = fill(typemax(Int), G0)
    maxd = fill(0, G0)
    for g in 1:G0, t in 1:T0
        if P[g, t] && !isnan(D[g, t])
            mind[g] = min(mind[g], t)
            maxd[g] = max(maxd[g], t)
        end
    end
    dsq = [mind[g] == typemax(Int) ? NaN : D[g, mind[g]] for g in 1:G0]
    # drop periods after a group has moved both above and below its status quo
    for g in 1:G0
        up = dn = false
        for t in 1:T0
            P[g, t] || continue
            if !isnan(D[g, t])
                D[g, t] > dsq[g] && (up = true)
                D[g, t] < dsq[g] && (dn = true)
            end
            (up && dn) && (P[g, t] = false)
        end
    end
    # first switching period F_g (0 = never)
    F = zeros(Int, G0)
    for g in 1:G0
        ever = 0
        prev = -1
        for t in 1:T0
            P[g, t] || continue
            e = (!isnan(D[g, t]) && abs(D[g, t] - dsq[g]) > 0) ? 1 : 0
            ever = max(ever, e)
            if ever == 1 && prev == 0 && F[g] == 0
                F[g] = t
            end
            prev = ever
        end
    end
    # status-quo classes need variation in F_g (sd over rows > 0)
    alive = [any(P[g, :]) for g in 1:G0]
    cls = [(dsq[g], TNg[g]) for g in 1:G0]
    for v in unique(cls[alive])
        rows = [(g, t) for g in 1:G0 for t in 1:T0 if P[g, t] && isequal(cls[g], v)]
        fs = [F[g] for (g, _) in rows]
        if length(fs) < 2 || all(==(fs[1]), fs)
            for (g, t) in rows
                P[g, t] = false
            end
        end
    end
    Gxx = count(g -> any(P[g, :]), 1:G0)
    Gxx > 0 || throw(ArgumentError(_DID_DCDH_NOEFFECT))
    # rows need a not-yet-switched group with the same status quo in that period
    notyet(g, t) = F[g] == 0 || t < F[g]     # 1 - ever_change_d at (g, t)
    ctrl = Dict{Tuple{Int,Float64,Int},Bool}()
    for g in 1:G0, t in 1:T0
        P[g, t] || continue
        key = (t, dsq[g], TNg[g])
        ctrl[key] = get(ctrl, key, false) | notyet(g, t)
    end
    for g in 1:G0, t in 1:T0
        P[g, t] && !ctrl[(t, dsq[g], TNg[g])] && (P[g, t] = false)
    end
    any(P) || throw(ArgumentError(_DID_DCDH_NOEFFECT))
    tpresent = findall(t -> any(P[:, t]), 1:T0)
    t_min, T_max = first(tpresent), last(tpresent)
    Ff = [F[g] == 0 ? T_max + 1.0 : float(F[g]) for g in 1:G0]
    # last period with observed treatment before the switch
    lastobs = fill(NaN, G0)
    for g in 1:G0, t in 1:T0
        if P[g, t] && t < Ff[g] && !isnan(D[g, t])
            lastobs[g] = isnan(lastobs[g]) ? t : max(lastobs[g], t)
        end
    end
    trunc = fill(NaN, G0)
    for g in 1:G0
        for t in 1:T0
            P[g, t] || continue
            t < mind[g] && (Y[g, t] = NaN)
            if Ff[g] < T_max + 1 && isnan(D[g, t]) && t < lastobs[g] && t > mind[g]
                D[g, t] = dsq[g]
            end
            if Ff[g] < T_max + 1 && t > lastobs[g] && lastobs[g] < Ff[g] - 1
                Y[g, t] = NaN
            end
        end
        if Ff[g] < T_max + 1 && lastobs[g] < Ff[g] - 1
            trunc[g] = lastobs[g] + 1
            Ff[g] = T_max + 1
        end
    end
    for g in 1:G0
        dF = 0 < Ff[g] <= T0 && P[g, Int(Ff[g])] ? D[g, Int(Ff[g])] : NaN
        for t in 1:T0
            P[g, t] || continue
            if Ff[g] < T_max + 1 && isnan(D[g, t]) && t > Ff[g] && lastobs[g] == Ff[g] - 1
                D[g, t] = dF
            end
            if Ff[g] == T_max + 1 && isnan(D[g, t]) && t > mind[g] && t < maxd[g]
                D[g, t] = dsq[g]
            end
            Ff[g] == T_max + 1 && t > maxd[g] && (Y[g, t] = NaN)
        end
        Ff[g] == T_max + 1 && (trunc[g] = maxd[g] + 1)
    end
    # balanced grid over the remaining groups and periods
    gkeep = findall(g -> any(P[g, :]), 1:G0)
    Gk, Tk = length(gkeep), length(tpresent)
    Yg = fill(NaN, Gk, Tk)
    Dg = fill(NaN, Gk, Tk)
    Ng = zeros(Gk, Tk)
    Wg = fill(NaN, Gk, Tk)
    Eg = fill(NaN, Gk, Tk)
    Xg = fill(NaN, Gk, Tk, K)
    for (a, g) in enumerate(gkeep), (b, t) in enumerate(tpresent)
        P[g, t] || continue
        Yg[a, b] = Y[g, t]
        Dg[a, b] = D[g, t]
        Ng[a, b] = (isnan(Y[g, t]) || isnan(D[g, t])) ? 0.0 : W[g, t]
        Wg[a, b] = W[g, t]
        Eg[a, b] = (F[g] != 0 && t >= F[g]) ? 1.0 : 0.0
        for k in 1:K
            Xg[a, b, k] = Xf[g, t, k]
        end
    end
    dsq_k = dsq[gkeep]
    tn_k = TNg[gkeep]
    F_k = Ff[gkeep]
    Ftr = [isnan(trunc[g]) ? Ff[g] : min(Ff[g], trunc[g]) for g in gkeep]
    Tg = similar(F_k)
    for a in 1:Gk
        Tg[a] = maximum(Ftr[b] for b in 1:Gk
                        if isequal(dsq_k[b], dsq_k[a]) && tn_k[b] == tn_k[a]) - 1
    end
    # average post-switch treatment and switcher type
    keep = trues(Gk)
    S = fill(NaN, Gk)
    for a in 1:Gk
        vals = [Dg[a, b] for (b, t) in enumerate(tpresent)
                if F_k[a] <= t <= Tg[a] && !isnan(Dg[a, b])]
        avg = isempty(vals) ? NaN : mean(vals)
        if !isnan(avg) && avg == dsq_k[a] && F_k[a] != Tg[a] + 1
            keep[a] = false
            continue
        end
        F_k[a] != T_max + 1 && !isnan(avg) && (S[a] = avg > dsq_k[a] ? 1.0 : 0.0)
    end
    sel = findall(keep)
    Yg, Dg, Ng = Yg[sel, :], Dg[sel, :], Ng[sel, :]
    Wg, Eg, Xg = Wg[sel, :], Eg[sel, :], Xg[sel, :, :]
    dsq_k, tn_k, F_k, Tg, S = dsq_k[sel], tn_k[sel], F_k[sel], Tg[sel], S[sel]
    gk = gkeep[sel]
    dfg = similar(F_k)
    for a in eachindex(F_k)
        b = findfirst(==(Int(F_k[a])), tpresent)
        dfg[a] = b === nothing ? NaN : Dg[a, b]
        isnan(dfg[a]) && F_k[a] == T_max + 1 && (dfg[a] = dsq_k[a])
    end
    L = Tg .- F_k .+ 1
    Lpl = [F_k[a] >= 3 ? min(L[a], F_k[a] - 2) : NaN for a in eachindex(F_k)]
    clus = cl === nothing ? collect(eachindex(gk)) : CLg[gk]
    return _DidDCDHPanel(Yg, Dg, Ng, tpresent, dsq_k, tn_k, F_k, Tg, S, dfg, L, Lpl,
                         clus, Gxx, t_min, T_max, ug[gk], ut, Xg, Wg, Eg)
end

const _DID_DCDH_NOEFFECT =
    "did_multiplegt_dyn: no treatment effect can be estimated: Design Restriction 1 " *
    "of de Chaisemartin & D'Haultfœuille (2026) fails. For every group, another " *
    "group with the same period-one treatment must switch at a different date (or " *
    "never); this fails e.g. when all groups switch at the same date, or when the " *
    "period-one treatment is continuous."

# ---------------------------------------------------------------------------
# Controls (DIDmultiplegtDYN's `controls` option): outcome changes are
# residualized on first-differenced controls, with coefficients estimated among
# not-yet-switched groups separately for each period-one treatment level, and the
# variance accounts for the estimation of those coefficients.
# ---------------------------------------------------------------------------

# Inverse of a symmetric PSD matrix; collinear directions get zero rows/columns
# (Stata's invsym, as DIDmultiplegtDYN's invsym_r).
function _dcdh_invsym(M)
    n = size(M, 1)
    C = cholesky(Symmetric(Matrix(M)), RowMaximum(); check=false)
    r = C.rank
    r == n && return inv(C)
    out = zeros(n, n)
    if r > 0
        idx = C.p[1:r]
        out[idx, idx] = inv(Symmetric(M[idx, idx]))
    end
    return out
end

function _did_dcdh_controls_setup(p::_DidDCDHPanel)
    Gk, Tk = size(p.Y)
    K = size(p.X, 3)
    tt = p.times
    dy = p.Y .- _dcdh_lag(p.Y, 1)
    dX = [p.X[:, :, k] .- _dcdh_lag(p.X[:, :, k], 1) for k in 1:K]
    fdok = [all(k -> !isnan(dX[k][a, b]), 1:K) for a in 1:Gk, b in 1:Tk]
    mask = [p.evc[a, b] == 0 && !isnan(dy[a, b]) && fdok[a, b] for a in 1:Gk, b in 1:Tk]
    # residuals of ΔX on (period × status quo × trends class) means among controls
    cls = collect(zip(p.dsq, p.tnp))
    ucls = unique(cls)
    cid = [findfirst(==(c), ucls) for c in cls]
    resid = [zeros(Gk, Tk) for _ in 1:K]
    prod = [zeros(Gk, Tk) for _ in 1:K]
    for k in 1:K
        sw = zeros(length(ucls), Tk)
        sx = zeros(length(ucls), Tk)
        for a in 1:Gk, b in 1:Tk
            mask[a, b] || continue
            sw[cid[a], b] += p.N[a, b]
            sx[cid[a], b] += p.N[a, b] * dX[k][a, b]
        end
        for a in 1:Gk, b in 1:Tk
            mask[a, b] || continue
            r = sqrt(p.N[a, b]) * (dX[k][a, b] - sx[cid[a], b] / sw[cid[a], b])
            isnan(r) && (r = 0.0)
            resid[k][a, b] = r
            prod[k][a, b] = sqrt(p.N[a, b]) * r
        end
    end
    levels = sort(unique(p.dsq))
    lvl = [findfirst(==(v), levels) for v in p.dsq]
    nlev = length(levels)
    useful = zeros(Int, nlev)
    theta = [zeros(K) for _ in 1:nlev]
    invden = [zeros(K, K) for _ in 1:nlev]
    drop = falses(nlev)
    singular = falses(nlev)
    for l in 1:nlev
        ing = findall(==(l), lvl)
        useful[l] = length(unique(p.F[ing]))
        useful[l] > 1 || continue
        rows = [(a, b) for a in ing for b in 1:Tk if mask[a, b]]
        if isempty(rows)
            drop[l] = true
            singular[l] = true
            useful[l] = 1
            continue
        end
        M = zeros(K, K)
        v = zeros(K)
        for (a, b) in rows
            yw = sqrt(p.N[a, b]) * dy[a, b]
            for j in 1:K
                v[j] += resid[j][a, b] * yw
                for k in 1:K
                    M[j, k] += resid[j][a, b] * resid[k][a, b]
                end
            end
        end
        iM = _dcdh_invsym(M)
        theta[l] = iM * v
        abs(det(M)) <= 1e-16 && (singular[l] = true)
        rmax = maximum(p.F[a] for (a, _) in rows)
        rsum = sum(p.N[a, b] for (a, b) in rows
                   if 2 <= tt[b] <= rmax - 1 && tt[b] < p.F[a]; init=0.0)
        invden[l] = iM .* (rsum * p.G)
    end
    any(singular) && @warn "did_multiplegt_dyn: some controls are not taken into " *
        "account for groups with period-one treatment " *
        join(levels[singular], ", ") * " (too few control observations or " *
        "controls without variation)"
    # E[ΔY | ΔX, period] among not-yet-switched groups of each level (period FE)
    insum = [zeros(Gk, K) for _ in 1:nlev]
    for l in 1:nlev
        (useful[l] > 1 && !drop[l]) || continue
        ing = findall(==(l), lvl)
        yhat = _did_dcdh_fe_predict(p, dy, dX, ing)
        den = zeros(Tk)
        for a in ing, b in 1:Tk
            (p.F[a] > tt[b] && !isnan(dy[a, b])) && (den[b] += 1)
        end
        Td = maximum(p.F[ing]) - 1
        Nc = sum(p.N[a, b] for a in ing for b in 1:Tk
                 if 2 <= tt[b] <= Td && tt[b] < p.F[a] && !isnan(dy[a, b]); init=0.0)
        for k in 1:K, a in ing
            acc = 0.0
            for b in 1:Tk
                (2 <= tt[b] <= p.F[a] - 1) || continue
                e = isnan(yhat[a, b]) ? NaN : (den[b] >= 2 ? yhat[a, b] : 0.0)
                x = prod[k][a, b] * (den[b] >= 2 ? sqrt(den[b] / (den[b] - 1)) : 1.0) *
                    (dy[a, b] - e) / Nc
                isnan(x) || (acc += x)
            end
            insum[l][a, k] = acc
        end
    end
    return (K=K, levels=levels, lvl=lvl, useful=useful, theta=theta, invden=invden,
            insum=insum, drop=drop)
end

# Weighted within-period regression of ΔY on ΔX (no intercept) among the
# not-yet-switched observations of the groups `ing`, and its predictions there.
function _did_dcdh_fe_predict(p, dy, dX, ing)
    Gk, Tk = size(dy)
    K = length(dX)
    tt = p.times
    est = [(a, b) for a in ing for b in 1:Tk
           if p.F[a] > tt[b] && !isnan(dy[a, b]) && !isnan(p.W[a, b]) &&
              all(k -> !isnan(dX[k][a, b]), 1:K)]
    out = fill(NaN, Gk, Tk)
    isempty(est) && return out
    bs = unique(last.(est))
    sw = Dict(b => 0.0 for b in bs)
    sy = Dict(b => 0.0 for b in bs)
    sx = Dict(b => zeros(K) for b in bs)
    for (a, b) in est
        w = p.W[a, b]
        sw[b] += w
        sy[b] += w * dy[a, b]
        for k in 1:K
            sx[b][k] += w * dX[k][a, b]
        end
    end
    XtX = zeros(K, K)
    Xty = zeros(K)
    for (a, b) in est
        w = p.W[a, b]
        xd = [dX[k][a, b] - sx[b][k] / sw[b] for k in 1:K]
        yd = dy[a, b] - sy[b] / sw[b]
        XtX .+= w .* (xd * xd')
        Xty .+= w .* xd .* yd
    end
    β = pinv(XtX) * Xty
    α = Dict(b => (sy[b] - dot(β, sx[b])) / sw[b] for b in bs)
    for a in ing, b in 1:Tk
        p.F[a] > tt[b] || continue
        haskey(α, b) || continue
        x = [dX[k][a, b] for k in 1:K]
        any(isnan, x) && continue
        out[a, b] = dot(β, x) + α[b]
    end
    return out
end

# M_{l,k} terms of the variance correction and the adjusted outcome changes.
function _did_dcdh_controls_adjust!(dy, p, ctrl, lagX, i, wt, Ni)
    Gk, Tk = size(dy)
    tt = p.times
    nlev = length(ctrl.levels)
    Mterm = zeros(nlev, ctrl.K)
    for k in 1:ctrl.K
        dX = lagX[k]
        if Ni != 0
            for a in 1:Gk
                i <= p.Tg[a] - 2 || continue
                l = ctrl.lvl[a]
                for b in 1:Tk
                    (i + 1 <= tt[b] <= p.Tg[a]) || continue
                    x = (p.G / Ni) * wt[a, b] * p.N[a, b] * dX[a, b]
                    isnan(x) || (Mterm[l, k] += x / p.G)
                end
            end
        end
        for l in 1:nlev
            ctrl.useful[l] > 1 || continue
            θ = ctrl.theta[l][k]
            for a in 1:Gk
                ctrl.lvl[a] == l || continue
                for b in 1:Tk
                    dy[a, b] -= θ * dX[a, b]
                end
            end
        end
    end
    return Mterm
end

# Variance correction for the estimated control coefficients (R's part2).
function _did_dcdh_part2(p, ctrl, Mterm)
    Gk = size(p.Y, 1)
    part2 = zeros(Gk)
    for l in eachindex(ctrl.levels)
        ctrl.useful[l] > 1 || continue
        for a in 1:Gk
            inl = ctrl.lvl[a] == l && p.F[a] >= 3
            for j in 1:ctrl.K
                br = -ctrl.theta[l][j]
                if inl
                    for k in 1:ctrl.K
                        br += ctrl.invden[l][j, k] * ctrl.insum[l][a, k]
                    end
                end
                part2[a] += Mterm[l, j] * br
            end
        end
    end
    return part2
end

function _dcdh_subset(p::_DidDCDHPanel, keep)
    k = findall(keep)
    return _DidDCDHPanel(p.Y[k, :], p.D[k, :], p.N[k, :], p.times, p.dsq[k], p.tnp[k],
                         p.F[k], p.Tg[k], p.S[k], p.dfg[k], p.L[k], p.Lpl[k],
                         p.cluster[k], p.G, p.t_min, p.T_max, p.group_labels[k],
                         p.period_labels, p.X[k, :, :], p.W[k, :], p.evc[k, :])
end

# Shift within the grid (row-based lag by k columns, as polars' shift over group).
_dcdh_lag(M, k) = (R = fill(NaN, size(M)); k < size(M, 2) &&
                   (R[:, (k + 1):end] = M[:, 1:(end - k)]); R)

# Kleene helpers: NaN encodes null.
_dcdh_nn(x) = !isnan(x)

"""
Per-effect / per-placebo influence-function columns for one switcher type (the R
function did_multiplegt_dyn_core). Returns a NamedTuple of group-level vectors.
"""
function _did_dcdh_core(p::_DidDCDHPanel, increase, nl, npl, only_never, same_sw,
                        effects_req, clustered, ctrl=nothing)
    Gk, Tk = size(p.Y)
    tt = p.times
    tmat = repeat(reshape(Float64.(tt), 1, :), Gk, 1)
    F, Tg, S = p.F, p.Tg, p.S
    Tmax = p.T_max
    Gxx = p.G
    # status-quo (× trends_nonparam) classes: cell sums by (class, period)
    clskey = collect(zip(p.dsq, p.tnp))
    dsq_levels = unique(clskey)
    dsq_id = [findfirst(==(v), dsq_levels) for v in clskey]
    nd = length(dsq_levels)
    cellsum(M) = (acc = zeros(nd, Tk);
                  for a in 1:Gk, b in 1:Tk
                      isnan(M[a, b]) || (acc[dsq_id[a], b] += M[a, b])
                  end; acc)
    # same-switchers restriction
    still = trues(Gk)
    if same_sw
        chk = zeros(Gk)
        for q in 1:effects_req
            dq = p.Y .- _dcdh_lag(p.Y, q)
            nv = [(_dcdh_nn(dq[a, b]) && F[a] > tt[b]) ?
                  ((only_never && F[a] < Tmax + 1) ? 0.0 : 1.0) : NaN
                  for a in 1:Gk, b in 1:Tk]
            Nc = cellsum(nv .* p.N)
            for a in 1:Gk
                b = findfirst(==(F[a] - 1 + q), tt)
                b === nothing && continue
                (Nc[dsq_id[a], b] > 0 && _dcdh_nn(dq[a, b])) && (chk[a] += 1)
            end
        end
        still = (F .- 1 .+ effects_req .<= Tg) .& (chk .== effects_req)
    end
    out = (U=Vector{Vector{Float64}}(), Uvar=Vector{Vector{Float64}}(),
           N=Float64[], Ndw=Float64[], count=Vector{Matrix{Float64}}(),
           delta_norm=Float64[], delta=Float64[],
           Upl=Vector{Vector{Float64}}(), Uvarpl=Vector{Vector{Float64}}(),
           Npl=Float64[], Ndwpl=Float64[], countpl=Vector{Matrix{Float64}}(),
           delta_norm_pl=Float64[])
    dist_store = Matrix{Float64}[]
    never_store = Matrix{Float64}[]
    for i in 1:nl
        dy = p.Y .- _dcdh_lag(p.Y, i)
        never = [_dcdh_nn(dy[a, b]) ?
                 ((F[a] > tt[b]) ? ((only_never && F[a] < Tmax + 1) ? 0.0 : 1.0) : 0.0) :
                 NaN for a in 1:Gk, b in 1:Tk]
        Nctrl = cellsum(never .* p.N)
        dist = fill(NaN, Gk, Tk)
        for a in 1:Gk, b in 1:Tk
            _dcdh_nn(dy[a, b]) || continue
            dist[a, b] = (still[a] && tt[b] == F[a] - 1 + i && i <= p.L[a] &&
                          S[a] == increase && Nctrl[dsq_id[a], b] > 0) ? 1.0 : 0.0
        end
        push!(dist_store, dist)
        push!(never_store, never)
        distw = dist .* p.N
        Nt = [sum(x for x in view(distw, :, b) if !isnan(x); init=0.0) for b in 1:Tk]
        Ntdw = [sum(x for x in view(dist, :, b) if !isnan(x); init=0.0) for b in 1:Tk]
        inwin = [p.t_min <= tt[b] <= Tmax for b in 1:Tk]
        Ni = sum(Nt[inwin])
        push!(out.N, Ni)
        push!(out.Ndw, sum(Ntdw[inwin]))
        Ntg = cellsum(distw)
        Mterm = nothing
        if ctrl !== nothing
            wt = [dist[a, b] - (Nctrl[dsq_id[a], b] == 0 ? 0.0 :
                                Ntg[dsq_id[a], b] / Nctrl[dsq_id[a], b]) * never[a, b]
                  for a in 1:Gk, b in 1:Tk]
            lagX = [p.X[:, :, k] .- _dcdh_lag(p.X[:, :, k], i) for k in 1:ctrl.K]
            dy = copy(dy)
            Mterm = _did_dcdh_controls_adjust!(dy, p, ctrl, lagX, i, wt, Ni)
        end
        U, Uvar, cnt = _did_dcdh_U(p, i, dy, dist, never, Nctrl, Ntg, Nt, Ni, dsq_id,
                                   clustered, tmat)
        if ctrl !== nothing && Ni != 0
            Uvar = Uvar .- _did_dcdh_part2(p, ctrl, Mterm)
        end
        push!(out.U, U)
        push!(out.Uvar, Uvar)
        push!(out.count, cnt)
        # normalization (cumulative treatment change) and ATE denominators
        dn = 0.0
        dl = 0.0
        if Ni != 0
            for a in 1:Gk
                cum = 0.0
                if S[a] == increase
                    for (b, t) in enumerate(tt)
                        if F[a] <= t <= F[a] - 1 + i && !isnan(p.D[a, b])
                            cum += p.D[a, b] - p.dsq[a]
                        end
                    end
                end
                for b in 1:Tk
                    dist[a, b] == 1 || continue
                    sgn = S[a] == 1 ? 1.0 : -1.0
                    dn += (p.N[a, b] / Ni) * sgn * cum
                    dl += (p.N[a, b] / Ni) * sgn * (p.D[a, b] - p.dsq[a])
                end
            end
        end
        push!(out.delta_norm, dn)
        push!(out.delta, dl)
    end
    for i in 1:npl
        dist, never = dist_store[i], never_store[i]
        Ylag2 = _dcdh_lag(p.Y, 2i)
        Ylag1 = _dcdh_lag(p.Y, i)
        dpl = Ylag2 .- Ylag1
        lagXpl = nothing
        if ctrl !== nothing
            lagXpl = [_dcdh_lag(p.X[:, :, k], 2i) .- _dcdh_lag(p.X[:, :, k], i)
                      for k in 1:ctrl.K]
            for k in 1:ctrl.K, l in eachindex(ctrl.levels)
                ctrl.useful[l] > 1 || continue
                θ = ctrl.theta[l][k]
                for a in 1:Gk
                    ctrl.lvl[a] == l || continue
                    dpl[a, :] .-= θ .* lagXpl[k][a, :]
                end
            end
        end
        neverpl = never .* [_dcdh_nn(x) ? 1.0 : 0.0 for x in dpl]
        Nctrl = cellsum(neverpl .* p.N)
        distpl = fill(NaN, Gk, Tk)
        for a in 1:Gk, b in 1:Tk
            _dcdh_nn(dist[a, b]) || continue
            distpl[a, b] = dist[a, b] * (_dcdh_nn(dpl[a, b]) ? 1.0 : 0.0) *
                           (Nctrl[dsq_id[a], b] > 0 ? 1.0 : 0.0)
        end
        distw = distpl .* p.N
        Nt = [sum(x for x in view(distw, :, b) if !isnan(x); init=0.0) for b in 1:Tk]
        Ntdw = [sum(x for x in view(distpl, :, b) if !isnan(x); init=0.0) for b in 1:Tk]
        inwin = [p.t_min <= tt[b] <= Tmax for b in 1:Tk]
        Ni = sum(Nt[inwin])
        push!(out.Npl, Ni)
        push!(out.Ndwpl, sum(Ntdw[inwin]))
        Ntg = cellsum(distw)
        U, Uvar, cnt = _did_dcdh_U(p, i, dpl, distpl, neverpl, Nctrl, Ntg, Nt, Ni,
                                   dsq_id, clustered, tmat)
        if ctrl !== nothing && Ni != 0
            wt = [distpl[a, b] - (Nctrl[dsq_id[a], b] == 0 ? 0.0 :
                                  Ntg[dsq_id[a], b] / Nctrl[dsq_id[a], b]) * neverpl[a, b]
                  for a in 1:Gk, b in 1:Tk]
            Mpl = zeros(length(ctrl.levels), ctrl.K)
            for k in 1:ctrl.K, a in 1:Gk
                i <= p.Tg[a] - 2 || continue
                for b in 1:Tk
                    (i + 1 <= tt[b] <= p.Tg[a]) || continue
                    x = (p.G / Ni) * wt[a, b] * p.N[a, b] * lagXpl[k][a, b]
                    isnan(x) || (Mpl[ctrl.lvl[a], k] += x / p.G)
                end
            end
            Uvar = Uvar .- _did_dcdh_part2(p, ctrl, Mpl)
        end
        push!(out.Upl, U)
        push!(out.Uvarpl, Uvar)
        push!(out.countpl, cnt)
        dn = 0.0
        if Ni != 0
            for a in 1:Gk
                S[a] == increase || continue
                cum = 0.0
                for (b, t) in enumerate(tt)
                    if F[a] <= t <= F[a] - 1 + i && !isnan(p.D[a, b])
                        cum += p.D[a, b] - p.dsq[a]
                    end
                end
                for b in 1:Tk
                    distpl[a, b] == 1 || continue
                    dn += (p.N[a, b] / Ni) * (S[a] == 1 ? 1.0 : -1.0) * cum
                end
            end
        end
        push!(out.delta_norm_pl, dn)
    end
    return out
end

# Influence function U_g, its cohort-demeaned version for the variance, and the
# per-cell observation counts, for one effect or placebo.
function _did_dcdh_U(p, i, dy, dist, never, Nctrl, Ntg, Nt, Ni, dsq_id, clustered, tmat)
    Gk, Tk = size(dy)
    tt = p.times
    F, Tg = p.F, p.Tg
    U = zeros(Gk)
    Uvar = zeros(Gk)
    cnt = zeros(Gk, Tk)
    Ni == 0 && return U, Uvar, cnt
    ratio(a, b) = Nctrl[dsq_id[a], b] == 0 ? 0.0 : Ntg[dsq_id[a], b] / Nctrl[dsq_id[a], b]
    # --- cohort means and degrees of freedom (variance only) ----------------------
    # controls: cells (d_sq, t); switchers: cohorts (d_sq, F_g, d_fg); union: (d_sq, t)
    Ntcol(b) = Nt[b]
    dofns = [(p.N[a, b] != 0 && _dcdh_nn(dy[a, b]) && never[a, b] == 1 && Ntcol(b) > 0)
             for a in 1:Gk, b in 1:Tk]
    dofs = [(p.N[a, b] != 0 && dist[a, b] == 1) for a in 1:Gk, b in 1:Tk]
    dofu = dofns .| dofs
    function cellstats(mask, keyf)
        cntd = Dict{Any,Float64}()
        tot = Dict{Any,Float64}()
        dofd = Dict{Any,Any}()
        for a in 1:Gk, b in 1:Tk
            mask[a, b] || continue
            k = keyf(a, b)
            cntd[k] = get(cntd, k, 0.0) + p.N[a, b]
            tot[k] = get(tot, k, 0.0) + p.N[a, b] * dy[a, b]
            if clustered
                s = get!(dofd, k, Set{Int}())
                push!(s, p.cluster[a])
            else
                dofd[k] = get(dofd, k, 0) + 1
            end
        end
        dofn = Dict(k => (clustered ? float(length(v)) : float(v)) for (k, v) in dofd)
        return cntd, tot, dofn
    end
    kns(a, b) = (dsq_id[a], b)
    ksw(a, b) = (dsq_id[a], F[a], p.dfg[a])
    cns, tns, dns = cellstats(dofns, kns)
    csw, tsw, dsw = cellstats(dofs, ksw)
    cun, tun, dun = cellstats(dofu, kns)
    for a in 1:Gk
        Tg[a] - 1 >= i || continue
        su = 0.0
        sv = 0.0
        for b in 1:Tk
            t = tt[b]
            (i + 1 <= t <= Tg[a]) || continue
            w = dist[a, b] - ratio(a, b) * never[a, b]
            isnan(w) && continue
            base = (p.G / Ni) * p.N[a, b] * w
            x = dy[a, b]
            if !isnan(x)
                su += base * x
            end
            # cohort-demeaned term
            isctrl = t < F[a]
            issw = t == F[a] - 1 + i
            (isctrl || issw) || continue
            E = 0.0
            dof = 1.0
            dn = dofns[a, b] ? dns[kns(a, b)] : NaN
            ds = dofs[a, b] ? dsw[ksw(a, b)] : NaN
            du = dofu[a, b] ? dun[kns(a, b)] : NaN
            if isctrl && !isnan(dn) && dn >= 2
                E = tns[kns(a, b)] / cns[kns(a, b)]
            end
            if issw && !isnan(ds) && ds >= 2
                E = tsw[ksw(a, b)] / csw[ksw(a, b)]
            end
            if !isnan(du) && du >= 2 && ((issw && ds == 1) || (isctrl && dn == 1))
                E = tun[kns(a, b)] / cun[kns(a, b)]
            end
            issw && ds > 1 && (dof = sqrt(ds / (ds - 1)))
            isctrl && dn > 1 && (dof = sqrt(dn / (dn - 1)))
            if !isnan(du) && du >= 2 && ((issw && ds == 1) || (isctrl && dn == 1))
                dof = sqrt(du / (du - 1))
            end
            isnan(x) || (sv += base * dof * (x - E))
        end
        U[a] = su
        Uvar[a] = sv
    end
    # observation counts (R's count<i>_core_XX)
    for a in 1:Gk, b in 1:Tk
        t = tt[b]
        w = dist[a, b] - ratio(a, b) * never[a, b]
        inwin = (Tg[a] - 1 >= i) && (i + 1 <= t <= Tg[a])
        ut = inwin ? (p.G / Ni) * p.N[a, b] * w * dy[a, b] : 0.0
        cond1 = !isnan(ut) && ut != 0
        cond2 = !isnan(ut) && ut == 0 && dy[a, b] == 0 &&
                (dist[a, b] != 0 || (Ntg[dsq_id[a], b] != 0 && never[a, b] != 0))
        # NaN comparisons are false, matching polars' null semantics here
        cnt[a, b] = (cond1 || cond2) ? p.N[a, b] : 0.0
    end
    return U, Uvar, cnt
end

"""
    did_multiplegt_dyn(data, outcome, treatment, group, time; effects=1, placebo=0,
                       normalized=false, switchers=:both, only_never_switchers=false,
                       same_switchers=false, trends_nonparam=Symbol[],
                       controls=Symbol[], weights=nothing, cluster=nothing)
        -> EventStudyEstimate

Event-study difference-in-differences estimators of de Chaisemartin and
D'Haultfœuille (2026) for designs with a treatment that may be **non-binary** and
**non-absorbing**, and whose lagged values may affect the outcome.

Let ``D_{g,1}`` be group ``g``'s treatment in the first period (its status quo) and
``F_g`` the first period in which its treatment changes. The estimand
``\\delta_\\ell`` is the average effect, among groups that have changed treatment,
of having been on their actual treatment path rather than on their status-quo
treatment for ``\\ell`` periods, measured ``\\ell - 1`` periods after the first
change. The estimator ``DID_\\ell`` compares the outcome change of switchers from
``F_g - 1`` to ``F_g - 1 + \\ell`` with that of groups with the same period-one
treatment that have not switched yet (or never switch), pooling switchers in and
switchers out (the latter with their sign flipped). Identification requires no
anticipation and parallel trends in the absence of treatment changes among groups
with the same period-one treatment; effects may be heterogeneous across groups and
over time and may depend on past treatments, and no restriction is placed on how
switchers' later treatments evolve. Because the effect is that of each switcher's
realized path, ``\\delta_\\ell`` is not an effect per unit of treatment;
`normalized = true` reports ``DID^n_\\ell = DID_\\ell / \\delta^D_\\ell``, the
estimate divided by the average cumulative treatment change, which estimates a
weighted average of the effects of a one-unit change in current and past
treatments. The average total effect per unit of treatment, reported in
`details.average_total_effect`, combines the effects over horizons into a
cost-benefit-type summary. With a binary, absorbing treatment the estimator reduces
to an event-study estimator in the spirit of Callaway and Sant'Anna (2021) with
not-yet-treated controls, and ``DID_1`` generalizes the ``DID_M`` estimator of
de Chaisemartin and D'Haultfœuille (2020).

Placebo estimators compare the outcome changes of switchers and their comparison
groups over the ``\\ell`` periods before ``F_g - 1``; they test parallel pre-trends
and no anticipation, with the usual caveat that a non-rejection does not establish
parallel trends (Roth, 2022). Inference uses the group-level influence functions of
the estimators, clustered at the group level or at the level of `cluster`, with
normal critical values; the full covariance of effects and placebos is estimated,
so joint tests and averages can be computed. Estimates at long horizons rely on few
switchers, since only groups observed long enough after switching contribute;
`same_switchers = true` holds the switcher sample fixed across horizons. With
`controls`, parallel trends is assumed conditional on the evolution of the
covariates, and the variance accounts for the estimation of their coefficients.
The implementation reproduces the R package `DIDmultiplegtDYN`.

# Arguments
- `data`: one row per group × period (several rows per cell are averaged with the
  weights); a group may be observed in a subset of periods. Missing treatments are
  imputed and outcomes discarded as in the R package.
- `outcome::Symbol`: outcome column.
- `treatment::Symbol`: numeric treatment (binary, discrete or continuous).
- `group::Symbol`: group identifier.
- `time::Symbol`: time column.

# Keywords
- `effects::Integer = 1`: number of dynamic effects ``\\ell = 1, \\dots,``
  `effects` (reduced with a warning when fewer can be estimated).
- `placebo::Integer = 0`: number of placebo estimators, at most `effects`. Placebo
  ``\\ell`` compares ``Y_{F_g-1-\\ell} - Y_{F_g-1}`` for the switchers of effect
  ``\\ell`` with the same change for their comparison groups.
- `normalized::Bool = false`: report ``DID^n_\\ell`` instead of ``DID_\\ell``.
- `switchers::Symbol = :both`: `:both`, `:in` (only switchers whose treatment
  increases) or `:out` (only decreases).
- `only_never_switchers::Bool = false`: use only never-switchers as comparison
  groups.
- `same_switchers::Bool = false`: use the same switchers (those observed for all
  `effects` periods) for every effect.
- `trends_nonparam::Vector{Symbol} = Symbol[]`: group-level variables (e.g. a
  region); switchers are compared only with groups in the same class, which allows
  class-specific non-parametric trends.
- `controls::Vector{Symbol} = Symbol[]`: time-varying covariates. Outcome changes are
  adjusted by ``\\theta_d'\\Delta X``, where ``\\theta_d`` is estimated by regressing
  first-differenced outcomes on first-differenced controls with period fixed effects
  among not-yet-switched groups with period-one treatment ``d``.
- `weights::Union{Nothing,Symbol} = nothing`: cell weights ``N_{g,t}``.
- `cluster::Union{Nothing,Symbol} = nothing`: a variable within which groups are
  nested; by default groups.

# Returns
- `EventStudyEstimate`: relative period ``e = \\ell - 1`` for effect ``\\ell``
  (``e = 0`` is the first period after the switch) and ``e = -1 - \\ell`` for placebo
  ``\\ell``; the reference period is ``-1`` (period ``F_g - 1``). `pre_trend_test(es)`
  is the joint placebo test and [`event_study_average`](@ref) gives averages.
  `details` contains `average_total_effect` (a [`DiDEstimate`](@ref) of the average
  total effect per unit of treatment), `n_switchers` (weighted) and
  `n_switchers_unweighted`, `n_obs` (group × period cells used) and
  `n_obs_weighted`, `delta` (the normalizing constants), `joint_effects_test` and
  `influence` (group-level influence functions, groups × coefficients).

# Examples
```julia
using DrSnow, CSV, DataFrames
file = joinpath(pkgdir(DrSnow), "test", "validation", "did", "dcdh_sim.csv")
sim = CSV.read(file, DataFrame; missingstring="NA")
es = did_multiplegt_dyn(sim, :y, :d, :g, :t; effects=3, placebo=2)
coeftable(es)
pre_trend_test(es)                  # joint placebo test
es.details.average_total_effect     # average total effect per unit of treatment
did_multiplegt_dyn(sim, :y, :d, :g, :t; effects=3, normalized=true,
                   cluster=:region)
```

# References
- de Chaisemartin, C., & D'Haultfœuille, X. (2026). Difference-in-differences
  estimators of intertemporal treatment effects. *Review of Economics and
  Statistics*, 108(4), 863–880.
- de Chaisemartin, C., & D'Haultfœuille, X. (2020). Two-way fixed effects estimators
  with heterogeneous treatment effects. *American Economic Review*, 110(9),
  2964–2996.
- de Chaisemartin, C., & D'Haultfœuille, X. (2023). Two-way fixed effects and
  differences-in-differences with heterogeneous treatment effects: A survey. *The
  Econometrics Journal*, 26(3), C1–C30.
- Roth, J. (2022). Pretest with caution: Event-study estimates after testing for
  parallel trends. *American Economic Review: Insights*, 4(3), 305–322.
- Quispe, A., Ciccia, D., Knau, F., Malezieux, M., Sow, D., & de Chaisemartin, C.
  (2026). DIDmultiplegtDYN: Estimation in staggered first switch designs, where
  groups experience their first treatment change at different points in time. R
  package version 2.4.0.
"""
function did_multiplegt_dyn(data, outcome::Symbol, treatment::Symbol, group::Symbol,
                            time::Symbol; effects::Integer=1, placebo::Integer=0,
                            normalized::Bool=false, switchers::Symbol=:both,
                            only_never_switchers::Bool=false,
                            same_switchers::Bool=false,
                            trends_nonparam::Vector{Symbol}=Symbol[],
                            controls::Vector{Symbol}=Symbol[],
                            weights::Union{Nothing,Symbol}=nothing,
                            cluster::Union{Nothing,Symbol}=nothing)
    effects >= 1 || throw(ArgumentError("effects must be a positive integer"))
    placebo >= 0 || throw(ArgumentError("placebo must be ≥ 0"))
    switchers in (:both, :in, :out) ||
        throw(ArgumentError("switchers must be :both, :in or :out"))
    cluster == group && (cluster = nothing)
    cols = [outcome, treatment, group, time, weights, cluster, trends_nonparam...,
            controls...]
    require_columns(data, [c for c in cols if c !== nothing];
                    context="did_multiplegt_dyn")
    df = DataFrame(data; copycols=false)
    keep = .!ismissing.(df[!, group]) .& .!ismissing.(df[!, time])
    cluster === nothing || (keep .&= .!ismissing.(df[!, cluster]))
    for c in controls
        keep .&= .!ismissing.(df[!, c])
        eltype(df[!, c]) <: Union{Missing,Real} ||
            throw(ArgumentError("control `$c` must be numeric"))
    end
    df = df[keep, :]
    nrow(df) > 0 || throw(ArgumentError("did_multiplegt_dyn: no observations"))
    tofloat(v) = [ismissing(x) ? NaN : Float64(x) for x in v]
    y = tofloat(df[!, outcome])
    d = tofloat(df[!, treatment])
    w = weights === nothing ? ones(nrow(df)) : tofloat(df[!, weights])
    any(x -> !isnan(x) && x < 0, w) && throw(ArgumentError("weights must be ≥ 0"))
    xc = isempty(controls) ? nothing :
         reduce(hcat, [tofloat(df[!, c]) for c in controls])
    tn = nothing
    if !isempty(trends_nonparam)
        for c in trends_nonparam
            any(ismissing, df[!, c]) && throw(ArgumentError(
                "did_multiplegt_dyn: trends_nonparam variable `$c` has missing values"))
        end
        keys_tn = [Tuple(df[i, c] for c in trends_nonparam) for i in 1:nrow(df)]
        utn = unique(keys_tn)
        tidx = Dict(k => i for (i, k) in enumerate(utn))
        tn = [tidx[k] for k in keys_tn]
    end
    cl = nothing
    if cluster !== nothing
        cv = df[!, cluster]
        uc = unique(cv)
        cidx = Dict(c => i for (i, c) in enumerate(uc))
        cl = [cidx[c] for c in cv]
    end
    gk = df[!, group]
    tv = df[!, time]
    # groups without any observed outcome or treatment are dropped before ranking
    hasy = Dict{Any,Bool}()
    hasd = Dict{Any,Bool}()
    for i in eachindex(gk)
        hasy[gk[i]] = get(hasy, gk[i], false) | !isnan(y[i])
        hasd[gk[i]] = get(hasd, gk[i], false) | !isnan(d[i])
    end
    ok = [hasy[g] && hasd[g] for g in gk]
    if !all(ok)
        gk, tv, y, d, w = gk[ok], tv[ok], y[ok], d[ok], w[ok]
        cl === nothing || (cl = cl[ok])
        tn === nothing || (tn = tn[ok])
        xc === nothing || (xc = xc[ok, :])
    end
    isempty(y) && throw(ArgumentError("did_multiplegt_dyn: no group has both an " *
                                      "observed outcome and treatment"))
    # aggregate several observations per group × period cell (weighted means)
    if length(unique(zip(gk, tv))) < nrow(df)
        gk, tv, y, d, w, cl, tn, xc = _did_dcdh_aggregate(gk, tv, y, d, w, cl, tn, xc)
    end
    p = _did_dcdh_build(gk, tv, y, d, w, cl, tn, xc)
    ctrl = nothing
    if xc !== nothing
        ctrl = _did_dcdh_controls_setup(p)
        if any(ctrl.drop)
            keepg = [!ctrl.drop[l] for l in ctrl.lvl]
            p = _dcdh_subset(p, keepg)
            ctrl = merge(ctrl, (lvl=ctrl.lvl[keepg],
                                insum=[m[keepg, :] for m in ctrl.insum]))
        end
    end
    return _did_dcdh_estimate(p, Int(effects), Int(placebo), normalized, switchers,
                              only_never_switchers, same_switchers, cl !== nothing,
                              nrow(df), ctrl)
end

function _did_dcdh_aggregate(gk, tv, y, d, w, cl, tn, xc)
    keys_ = collect(zip(gk, tv))
    u = unique(keys_)
    idx = Dict(k => i for (i, k) in enumerate(u))
    m = length(u)
    sw = zeros(m)
    sy = zeros(m)
    sd = zeros(m)
    cy = zeros(m)
    cd = zeros(m)
    clo = cl === nothing ? nothing : zeros(Int, m)
    tno = tn === nothing ? nothing : zeros(Int, m)
    xo = xc === nothing ? nothing : zeros(m, size(xc, 2))
    for i in eachindex(keys_)
        k = idx[keys_[i]]
        wi = isnan(d[i]) ? 0.0 : (isnan(w[i]) ? 0.0 : w[i])
        sw[k] += wi
        isnan(y[i]) || (sy[k] += wi * y[i]; cy[k] += wi)
        isnan(d[i]) || (sd[k] += wi * d[i]; cd[k] += wi)
        clo === nothing || (clo[k] = cl[i])
        tno === nothing || (tno[k] = tn[i])
        xo === nothing || (xo[k, :] .+= wi .* xc[i, :])
    end
    ya = [cy[k] > 0 ? sy[k] / sw[k] : NaN for k in 1:m]
    da = [cd[k] > 0 ? sd[k] / sw[k] : NaN for k in 1:m]
    xo === nothing || (xo ./= sw)
    return first.(u), last.(u), ya, da, sw, clo, tno, xo
end

function _did_dcdh_estimate(p, effects, placebo, normalized, switchers, only_never,
                            same_sw, clustered, nobs_in, ctrl=nothing)
    Gk = size(p.Y, 1)
    Lu = any(==(1.0), p.S) ? maximum(p.L[p.S .== 1.0]) : 0.0
    La = any(==(0.0), p.S) ? maximum(p.L[p.S .== 0.0]) : 0.0
    Lplu = any(==(1.0), p.S) ? _dcdh_nanmax(p.Lpl[p.S .== 1.0]) : 0.0
    Lpla = any(==(0.0), p.S) ? _dcdh_nanmax(p.Lpl[p.S .== 0.0]) : 0.0
    use_in = switchers in (:both, :in)
    use_out = switchers in (:both, :out)
    Lmax = max(use_in ? Lu : 0.0, use_out ? La : 0.0)
    Lmax >= 1 || throw(ArgumentError(_DID_DCDH_NOEFFECT))
    l = Int(min(Lmax, effects))
    l < effects && @warn "did_multiplegt_dyn: only $l effect(s) can be estimated " *
                         "(requested $effects)"
    lpl = 0
    if placebo > 0
        lpl = Int(min(max(use_in ? Lplu : 0.0, use_out ? Lpla : 0.0), placebo, effects))
        lpl < placebo && @warn "did_multiplegt_dyn: only $lpl placebo(s) can be " *
                               "estimated (requested $placebo; placebos cannot exceed " *
                               "the number of effects)"
    end
    cin = use_in && Lu >= 1 ?
          _did_dcdh_core(p, 1.0, Int(min(Lu, l)), Int(min(lpl, Lplu)), only_never,
                         same_sw, l, clustered, ctrl) : nothing
    cout = use_out && La >= 1 ?
           _did_dcdh_core(p, 0.0, Int(min(La, l)), Int(min(lpl, Lpla)), only_never,
                          same_sw, l, clustered, ctrl) : nothing
    getv(c, f, i, default) = (c === nothing || i > length(getfield(c, f))) ? default :
                             getfield(c, f)[i]
    zerog = zeros(Gk)
    G = p.G
    coefs = Float64[]
    Vcols = Vector{Vector{Float64}}()     # group-level variance IFs (scaled)
    Ucols = Vector{Vector{Float64}}()
    nsw = Float64[]
    nsw_dw = Float64[]
    nobs_e = Float64[]
    nobs_dw = Float64[]
    deltas = Float64[]
    labels = Int[]
    function combine(kind, i)
        pl = kind === :placebo
        fU, fV, fN, fNdw, fc, fdn = pl ? (:Upl, :Uvarpl, :Npl, :Ndwpl, :countpl,
                                         :delta_norm_pl) :
                                    (:U, :Uvar, :N, :Ndw, :count, :delta_norm)
        N1 = getv(cin, fN, i, 0.0)
        N0 = getv(cout, fN, i, 0.0)
        tot = N1 + N0
        tot > 0 || throw(ArgumentError(
            "did_multiplegt_dyn: $(pl ? "placebo" : "effect") $i cannot be estimated " *
            "(no switcher or no comparison group)"))
        Up = getv(cin, fU, i, zerog)
        Um = -getv(cout, fU, i, zerog)
        Vin = getv(cin, fV, i, zerog)
        Vout = -getv(cout, fV, i, zerog)
        est = (N1 * sum(Up) / G + N0 * sum(Um) / G) / tot
        Vg = (N1 / tot) .* Vin .+ (N0 / tot) .* Vout
        Ug = (N1 / tot) .* Up .+ (N0 / tot) .* Um
        δ = 1.0
        if normalized
            δ = (N1 / tot) * getv(cin, fdn, i, 0.0) + (N0 / tot) * getv(cout, fdn, i, 0.0)
            δ != 0 || throw(ArgumentError(
                "did_multiplegt_dyn: normalization constant of $(pl ? "placebo" :
                "effect") $i is zero"))
        end
        push!(coefs, est / δ)
        push!(Vcols, Vg ./ δ)
        push!(Ucols, Ug ./ δ)
        push!(nsw, tot)
        push!(nsw_dw, getv(cin, fNdw, i, 0.0) + getv(cout, fNdw, i, 0.0))
        cp = getv(cin, fc, i, zeros(size(p.Y)))
        cm = getv(cout, fc, i, zeros(size(p.Y)))
        cg = max.(cp, cm)
        push!(nobs_e, sum(cg))
        push!(nobs_dw, count(>(0), cg))
        push!(deltas, δ)
    end
    for i in 1:l
        combine(:effect, i)
        push!(labels, i - 1)
    end
    for i in 1:lpl
        combine(:placebo, i)
        push!(labels, -1 - i)
    end
    # covariance from cluster sums of the variance influence functions
    Vmat = reduce(hcat, Vcols)
    S = _cluster_sums(Vmat, clustered ? p.cluster : nothing)
    V = Matrix(Symmetric((S' * S) ./ G^2))
    ncl = clustered ? size(S, 1) : 0
    # average total effect per unit of treatment
    ate = _did_dcdh_ate(p, cin, cout, l, use_in, use_out, clustered, G, nobs_in)
    # order by relative period
    ord = sortperm(labels)
    rel = labels[ord]
    b = coefs[ord]
    Vo = V[ord, ord]
    eidx = findall(>=(0), rel)
    jt = nothing
    if length(eidx) > 1
        jt = _did_joint_zero_test(b[eidx], Vo[eidx, eidx], Inf, rel[eidx],
                                  "Joint test of the effects",
                                  "all dynamic effects DID_ℓ are zero")
    end
    meth = "de Chaisemartin–D'Haultfœuille DID_ℓ (did_multiplegt_dyn" *
           (normalized ? ", normalized" : "") * ")"
    est = normalized ? "DID^n_ℓ: effect of a one-unit change in current and past " *
                       "treatments, ℓ = e + 1 periods after the first switch" :
          "DID_ℓ: effect of the actual treatment path, ℓ = e + 1 periods after the " *
          "first switch (placebos at e ≤ -2)"
    return EventStudyEstimate(rel, b, Vo, [-1], nobs_in, Inf, ncl, meth, est, Float64[],
        (average_total_effect=ate, n_switchers=nsw[ord],
         n_switchers_unweighted=nsw_dw[ord], n_obs=nobs_dw[ord],
         n_obs_weighted=nobs_e[ord], delta=deltas[ord],
         joint_effects_test=jt, influence=Matrix(reduce(hcat, Ucols)[:, ord]),
         variance_influence=Vmat[:, ord], groups=p.group_labels,
         binned=(false, false), n_treated=count(!isnan, p.S),
         n_control=count(isnan, p.S), n_periods=length(p.times),
         switchers=switchers, normalized=normalized,
         note="Effect ℓ is reported at e = ℓ - 1 and placebo ℓ at e = -1 - ℓ; the " *
              "reference period e = -1 is the last period before the switch."))
end

_dcdh_nanmax(v) = (f = filter(!isnan, v); isempty(f) ? 0.0 : maximum(f))

function _did_dcdh_ate(p, cin, cout, l, use_in, use_out, clustered, G, nobs_in)
    function part(c, sgn)
        c === nothing && return nothing
        k = length(c.N)
        sumN = sum(c.N)
        sumN == 0 && return nothing
        num = zeros(size(p.Y, 1))
        numv = zeros(size(p.Y, 1))
        den = 0.0
        for i in 1:k
            c.N[i] == 0 && continue
            wi = c.N[i] / sumN
            num .+= wi .* c.U[i]
            numv .+= wi .* c.Uvar[i]
            den += wi * c.delta[i]
        end
        return (U=sgn .* num ./ den, V=sgn .* numv ./ den, den=den, sumN=sumN)
    end
    pin = use_in ? part(cin, 1.0) : nothing
    pout = use_out ? part(cout, -1.0) : nothing
    dp = pin === nothing ? 0.0 : pin.den
    dm = pout === nothing ? 0.0 : pout.den
    n1 = pin === nothing ? 0.0 : pin.sumN
    n0 = pout === nothing ? 0.0 : pout.sumN
    denom = dp * n1 + dm * n0
    wp = denom > 0 ? dp * n1 / denom : 0.5
    zero_ = zeros(size(p.Y, 1))
    U = wp .* (pin === nothing ? zero_ : pin.U) .+
        (1 - wp) .* (pout === nothing ? zero_ : pout.U)
    Vg = wp .* (pin === nothing ? zero_ : pin.V) .+
         (1 - wp) .* (pout === nothing ? zero_ : pout.V)
    est = sum(U) / G
    S = _cluster_sums(reshape(Vg, :, 1), clustered ? p.cluster : nothing)
    v = sum(abs2, S) / G^2
    return DiDEstimate([est], fill(v, 1, 1), ["average total effect"], nobs_in, Inf,
                       clustered ? size(S, 1) : 0, count(!isnan, p.S), count(isnan, p.S),
                       length(p.times),
                       "de Chaisemartin–D'Haultfœuille average total effect",
                       "average total effect per unit of treatment (δ̂)",
                       (influence=U, note=""))
end
