# Data preparation and MSE/CER-optimal bandwidth selection (rdbwselect equivalent).

# ---------------------------------------------------------------------------------------
# Data preparation shared by rd_estimate and rd_bandwidth
# ---------------------------------------------------------------------------------------

function _rd_side(X, Y, T, Z, C, W, vce)
    dups, dupsid = vce === :nn ? _rd_dups(X) : (Int[], Int[])
    return (X=X, Y=Y, T=T, Z=Z, C=C, W=W, dups=dups, dupsid=dupsid)
end

_rd_rows(::Nothing, idx) = nothing
_rd_rows(v::AbstractVector, idx) = v[idx]
_rd_rows(M::AbstractMatrix, idx) = M[idx, :]

"""
Sort by the running variable, split at the cutoff, and compute everything the bandwidth
selectors and estimators need. Inputs are already free of missing values.
"""
function _rd_setup(y::Vector{Float64}, x::Vector{Float64}, T, Z, C, W; c::Float64,
                   p::Int, q::Int, deriv::Int, kernel::Symbol, vce::Symbol,
                   nnmatch::Int, masspoints::Symbol, bwcheck, bwrestrict::Bool,
                   sharpbw::Bool, context::String)
    n = length(x)
    length(y) == n ||
        throw(DimensionMismatch("outcome and running variable lengths differ"))
    p >= 0 || throw(ArgumentError("$context: p must be ≥ 0"))
    q > p || throw(ArgumentError("$context: q must be greater than p (got p=$p, q=$q)"))
    0 <= deriv <= p || throw(ArgumentError("$context: deriv must satisfy 0 ≤ deriv ≤ p"))
    nnmatch >= 1 || throw(ArgumentError("$context: nnmatch must be a positive integer"))
    if bwcheck !== nothing && bwcheck < 1
        throw(ArgumentError("$context: bwcheck must be a positive integer"))
    end
    if W !== nothing
        any(<(0), W) && throw(ArgumentError("$context: weights must be non-negative"))
        sum(W) > 0 ||
            throw(ArgumentError("$context: weights must include a positive value"))
    end
    ord = sortperm(x; alg=MergeSort)
    x = x[ord] .+ 0.0   # `+ 0.0` maps -0.0 to 0.0 so that `unique` treats them as equal
    y = y[ord]
    T = _rd_rows(T, ord); Z = _rd_rows(Z, ord); C = _rd_rows(C, ord); W = _rd_rows(W, ord)

    x_min, x_max = x[1], x[end]
    (c <= x_min || c >= x_max) && throw(ArgumentError(
        "$context: the cutoff $c must lie strictly inside the range of the running " *
        "variable [$x_min, $x_max]"))
    il = findall(<(c), x)
    ir = findall(>=(c), x)
    N_l, N_r = length(il), length(ir)
    N = N_l + N_r
    range_l = abs(c - x_min)
    range_r = abs(c - x_max)

    # Fuzzy design: perfect one-sided compliance => bandwidths computed as sharp.
    perf_comp = false
    if T !== nothing
        T_l, T_r = T[il], T[ir]
        vl = N_l > 1 ? var(T_l) : 0.0
        vr = N_r > 1 ? var(T_r) : 0.0
        if vl == 0 && vr == 0 && abs(mean(T_l) - mean(T_r)) < sqrt(eps(Float64))
            throw(ArgumentError("$context: the treatment variable has no variation and " *
                                "no jump at the cutoff; the fuzzy RD is not identified"))
        end
        perf_comp = vl == 0 || vr == 0
    end

    has_cluster = C !== nothing
    vce_used = _rd_resolve_vce(vce, has_cluster)
    L = _rd_side(x[il], y[il], _rd_rows(T, il), _rd_rows(Z, il), _rd_rows(C, il),
                 _rd_rows(W, il), vce_used)
    R = _rd_side(x[ir], y[ir], _rd_rows(T, ir), _rd_rows(Z, ir), _rd_rows(C, ir),
                 _rd_rows(W, ir), vce_used)
    g_l = has_cluster ? length(unique(L.C)) : 0
    g_r = has_cluster ? length(unique(R.C)) : 0

    # Mass points
    X_uniq_l = sort(unique(L.X); rev=true)
    X_uniq_r = sort(unique(R.X))
    M_l, M_r = length(X_uniq_l), length(X_uniq_r)
    mass_detected = false
    if masspoints !== :off
        mass_l = 1 - M_l / N_l
        mass_r = 1 - M_r / N_r
        if mass_l >= 0.2 || mass_r >= 0.2
            mass_detected = true
            if masspoints === :check
                @warn "$context: mass points detected in the running variable; " *
                      "consider masspoints = :adjust"
            elseif bwcheck === nothing
                bwcheck = 10
            end
        end
    end
    if masspoints === :off
        M_l, M_r = N_l, N_r
    end

    return (; x, y, T, Z, C, W, c, L, R, N_l, N_r, N, x_min, x_max, range_l, range_r,
            perf_comp, sharpbw, vce=vce_used, has_cluster, g_l, g_r, X_uniq_l, X_uniq_r,
            M_l, M_r, mass_detected, masspoints, bwcheck, bwrestrict, p, q, deriv,
            kernel, nnmatch)
end

# ---------------------------------------------------------------------------------------
# One stage of the plug-in bandwidth computation (rdrobust_bw)
# ---------------------------------------------------------------------------------------

function _rd_design_D(Y, T, Z)
    D = reshape(Y, :, 1)
    T === nothing || (D = hcat(D, T))
    Z === nothing || (D = hcat(D, Z))
    return D
end

"""
Variance, bias and regularisation constants for a local polynomial of order `o`
estimating derivative `nu`, with pilot bandwidth `h_V` for the variance and an
order-`o_B` fit with bandwidth `h_B` for the leading bias. Mirrors `rdrobust_bw`.
"""
function _rd_bw_stage(S, sd, T_use::Bool, o::Int, nu::Int, o_B::Int, h_V::Float64,
                      h_B::Float64, scale::Float64, cache::Dict)
    c, kernel, vce, nnmatch = S.c, S.kernel, S.vce, S.nnmatch
    Tside = T_use ? sd.T : nothing
    dT = Tside === nothing ? 0 : 1
    dZ = sd.Z === nothing ? 0 : size(sd.Z, 2)
    colsZ = (2 + dT):(1 + dT + dZ)
    key = (o, nu)
    if haskey(cache, key)
        V_V, BConst, s = cache[key]
    else
        w = _rd_kweight(sd.X, c, h_V, kernel)
        sd.W === nothing || (w .*= sd.W)
        ind = w .> 0
        eX = sd.X[ind]
        eW = w[ind]
        D_V = _rd_design_D(sd.Y[ind], _rd_rows(Tside, ind), _rd_rows(sd.Z, ind))
        R_V = _rd_vander(eX .- c, o)
        invG_V = _rd_xxinv(R_V .* sqrt.(eW))
        RW = R_V .* eW
        s = [1.0]
        gamma = nothing
        if dZ > 0
            eZ = sd.Z[ind, :]
            U = RW' * D_V
            ZWD = (eZ .* eW)' * D_V
            UiGU = U[:, colsZ]' * (invG_V * U)
            ZWZ = ZWD[:, colsZ] .- UiGU[:, colsZ]
            ZWY = ZWD[:, 1:(1 + dT)] .- UiGU[:, 1:(1 + dT)]
            gamma = _rd_ginv(ZWZ) * ZWY
            s = vcat(1.0, -gamma[:, 1])
        end
        beta_V = invG_V * (RW' * D_V)
        if dZ == 0 && dT == 1
            tau_Y = factorial(nu) * beta_V[nu + 1, 1]
            tau_T = factorial(nu) * beta_V[nu + 1, 2]
            s = [1 / tau_T, -(tau_Y / tau_T^2)]
        elseif dZ > 0 && dT == 1
            s_T = vcat(1.0, -gamma[:, 2])
            tau_Y = factorial(nu) * dot(s, vcat(beta_V[nu + 1, 1], beta_V[nu + 1, colsZ]))
            tau_T = factorial(nu) *
                    dot(s_T, vcat(beta_V[nu + 1, 2], beta_V[nu + 1, colsZ]))
            s = vcat(1 / tau_T, -(tau_Y / tau_T^2),
                     -(1 / tau_T) .* gamma[:, 1] .+ (tau_Y / tau_T^2) .* gamma[:, 2])
        end
        dups = vce === :nn ? sd.dups[ind] : Int[]
        dupsid = vce === :nn ? sd.dupsid[ind] : Int[]
        res_V = _rd_residuals(eX, D_V, R_V, beta_V, invG_V, eW, vce, nnmatch, dups,
                              dupsid, o + 1, S.has_cluster)
        groups = S.has_cluster ? _rd_cluster_groups(sd.C[ind]) : nothing
        rs = res_V * s
        aux = _rd_vmeat(vce, R_V, eW, invG_V, rs, rs, groups, o + 1)
        V_V = (invG_V * aux * invG_V)[nu + 1, nu + 1]
        v = RW' * (((eX .- c) ./ h_V) .^ (o + 1))
        Hp = [h_V^(j - 1) for j in 1:(o + 1)]
        BConst = (Hp .* (invG_V * v))[nu + 1]
        cache[key] = (V_V, BConst, s)
    end

    w = _rd_kweight(sd.X, c, h_B, kernel)
    sd.W === nothing || (w .*= sd.W)
    ind = w .> 0
    eX = sd.X[ind]
    eW = w[ind]
    D_B = _rd_design_D(sd.Y[ind], _rd_rows(Tside, ind), _rd_rows(sd.Z, ind))
    R_B = _rd_vander(eX .- c, o_B)
    invG_B = _rd_xxinv(R_B .* sqrt.(eW))
    RW = R_B .* eW
    beta_B = invG_B * (RW' * D_B)
    BWreg = 0.0
    if scale > 0
        dups = vce === :nn ? sd.dups[ind] : Int[]
        dupsid = vce === :nn ? sd.dupsid[ind] : Int[]
        res_B = _rd_residuals(eX, D_B, R_B, beta_B, invG_B, eW, vce, nnmatch, dups,
                              dupsid, o_B + 1, S.has_cluster)
        groups = S.has_cluster ? _rd_cluster_groups(sd.C[ind]) : nothing
        rs = res_B * s
        V_B = (invG_B * _rd_vmeat(vce, R_B, eW, invG_B, rs, rs, groups, o_B + 1) *
               invG_B)[o + 2, o + 2]
        BWreg = 3 * BConst^2 * V_B
    end
    B = sqrt(2 * (o + 1 - nu)) * BConst * dot(s, beta_B[o + 2, :])
    V = (2 * nu + 1) * h_V^(2 * nu + 1) * V_V
    Rg = scale * (2 * (o + 1 - nu)) * BWreg
    return (V=V, B=B, R=Rg, rate=1 / (2 * o + 3))
end

# ---------------------------------------------------------------------------------------
# Bandwidth selectors
# ---------------------------------------------------------------------------------------

"""
Compute the bandwidths requested in `methods` (a collection of selector symbols) and
return a `Dict{Symbol, NTuple{4,Float64}}` with `(h_left, h_right, b_left, b_right)`.
"""
function _rd_select_bandwidths(S, methods; scaleregul::Float64=1.0,
                               bwp::Union{Nothing,Float64}=nothing)
    p, q, deriv, kernel = S.p, S.q, S.deriv, S.kernel
    x = S.x
    N = S.N
    T_use = S.T !== nothing && !(S.perf_comp || S.sharpbw)
    x_iq = _rd_quantile_type2(x, 0.75) - _rd_quantile_type2(x, 0.25)
    BWp = bwp === nothing ? min(std(x), x_iq / 1.349) : bwp
    C_c = kernel === :epanechnikov ? 2.34 : kernel === :uniform ? 1.843 : 2.576
    c_bw = C_c * BWp * N^(-1 / 5)
    if S.masspoints === :adjust
        c_bw = C_c * BWp * (S.M_l + S.M_r)^(-1 / 5)
    end
    bw_max_l, bw_max_r = S.range_l, S.range_r
    bw_max = max(bw_max_l, bw_max_r)
    S.bwrestrict && (c_bw = min(c_bw, bw_max))
    bwcheck = S.bwcheck
    bw_min_l = bw_min_r = 0.0
    if bwcheck !== nothing
        bwcheck_l = min(bwcheck, length(S.X_uniq_l))
        bwcheck_r = min(bwcheck, length(S.X_uniq_r))
        bw_min_l = abs(S.X_uniq_l[bwcheck_l] - S.c) + 1e-8
        bw_min_r = abs(S.X_uniq_r[bwcheck_r] - S.c) + 1e-8
        c_bw = max(c_bw, bw_min_l, bw_min_r)
    end
    L, R = S.L, S.R
    cl, cr = Dict(), Dict()
    reg = Float64(scaleregul)
    stage(sd, cache, o, nu, oB, hB, sc) = _rd_bw_stage(S, sd, T_use, o, nu, oB, c_bw,
                                                       Float64(hB), sc, cache)
    C_d_l = stage(L, cl, q + 1, q + 1, q + 2, S.range_l, 0.0)
    C_d_r = stage(R, cr, q + 1, q + 1, q + 2, S.range_r, 0.0)
    ms = Set(methods)
    need_two = !isempty(intersect(ms, (:msetwo, :certwo, :msecomb2, :cercomb2)))
    need_sum = !isempty(intersect(ms, (:msesum, :cersum, :msecomb1, :msecomb2,
                                       :cercomb1, :cercomb2)))
    need_rd = !isempty(intersect(ms, (:mserd, :cerrd, :msecomb1, :msecomb2, :cercomb1,
                                      :cercomb2)))
    out = Dict{Symbol,NTuple{4,Float64}}()
    local h_rd, b_rd, h_sum, b_sum, h_two_l, h_two_r, b_two_l, b_two_r
    if need_two
        d_l = (C_d_l.V / C_d_l.B^2)^C_d_l.rate
        d_r = (C_d_r.V / C_d_r.B^2)^C_d_r.rate
        if S.bwrestrict
            d_l = min(d_l, bw_max_l); d_r = min(d_r, bw_max_r)
        end
        if bwcheck !== nothing
            d_l = max(d_l, bw_min_l); d_r = max(d_r, bw_min_r)
        end
        Cb_l = stage(L, cl, q, p + 1, q + 1, d_l, reg)
        Cb_r = stage(R, cr, q, p + 1, q + 1, d_r, reg)
        b_two_l = (Cb_l.V / (Cb_l.B^2 + reg * Cb_l.R))^Cb_l.rate
        b_two_r = (Cb_r.V / (Cb_r.B^2 + reg * Cb_r.R))^Cb_r.rate
        if S.bwrestrict
            b_two_l = min(b_two_l, bw_max_l); b_two_r = min(b_two_r, bw_max_r)
        end
        Ch_l = stage(L, cl, p, deriv, q, b_two_l, reg)
        Ch_r = stage(R, cr, p, deriv, q, b_two_r, reg)
        h_two_l = (Ch_l.V / (Ch_l.B^2 + reg * Ch_l.R))^Ch_l.rate
        h_two_r = (Ch_r.V / (Ch_r.B^2 + reg * Ch_r.R))^Ch_r.rate
        if S.bwrestrict
            h_two_l = min(h_two_l, bw_max_l); h_two_r = min(h_two_r, bw_max_r)
        end
    end
    # :sum uses B_r + B_l, :rd uses B_r - B_l in the bias term.
    function common(sgn)
        d = ((C_d_l.V + C_d_r.V) / (C_d_r.B + sgn * C_d_l.B)^2)^C_d_l.rate
        S.bwrestrict && (d = min(d, bw_max))
        bwcheck !== nothing && (d = max(d, bw_min_l, bw_min_r))
        Cb_l = stage(L, cl, q, p + 1, q + 1, d, reg)
        Cb_r = stage(R, cr, q, p + 1, q + 1, d, reg)
        bb = ((Cb_l.V + Cb_r.V) / ((Cb_r.B + sgn * Cb_l.B)^2 +
                                   reg * (Cb_r.R + Cb_l.R)))^Cb_l.rate
        S.bwrestrict && (bb = min(bb, bw_max))
        Ch_l = stage(L, cl, p, deriv, q, bb, reg)
        Ch_r = stage(R, cr, p, deriv, q, bb, reg)
        hh = ((Ch_l.V + Ch_r.V) / ((Ch_r.B + sgn * Ch_l.B)^2 +
                                   reg * (Ch_r.R + Ch_l.R)))^Ch_l.rate
        S.bwrestrict && (hh = min(hh, bw_max))
        return hh, bb
    end
    if need_sum
        h_sum, b_sum = common(1.0)
    end
    if need_rd
        h_rd, b_rd = common(-1.0)
    end
    cer_n = S.has_cluster ? S.g_l + S.g_r : N
    cer_h = cer_n^(-(p / ((3 + p) * (3 + 2 * p))))
    :mserd in ms && (out[:mserd] = (h_rd, h_rd, b_rd, b_rd))
    :msetwo in ms && (out[:msetwo] = (h_two_l, h_two_r, b_two_l, b_two_r))
    :msesum in ms && (out[:msesum] = (h_sum, h_sum, b_sum, b_sum))
    if :msecomb1 in ms || :cercomb1 in ms
        h1, b1 = min(h_rd, h_sum), min(b_rd, b_sum)
        :msecomb1 in ms && (out[:msecomb1] = (h1, h1, b1, b1))
        :cercomb1 in ms && (out[:cercomb1] = (h1 * cer_h, h1 * cer_h, b1, b1))
    end
    if :msecomb2 in ms || :cercomb2 in ms
        hl = median([h_rd, h_sum, h_two_l]); hr = median([h_rd, h_sum, h_two_r])
        bl = median([b_rd, b_sum, b_two_l]); br = median([b_rd, b_sum, b_two_r])
        :msecomb2 in ms && (out[:msecomb2] = (hl, hr, bl, br))
        :cercomb2 in ms && (out[:cercomb2] = (hl * cer_h, hr * cer_h, bl, br))
    end
    :cerrd in ms && (out[:cerrd] = (h_rd * cer_h, h_rd * cer_h, b_rd, b_rd))
    :cersum in ms && (out[:cersum] = (h_sum * cer_h, h_sum * cer_h, b_sum, b_sum))
    :certwo in ms && (out[:certwo] = (h_two_l * cer_h, h_two_r * cer_h, b_two_l, b_two_r))
    return out, c_bw
end

# Bandwidths selected on standardized data (rdrobust's `stdvars = TRUE`): the outcome and
# the running variable (and the cutoff) are divided by their standard deviations, the
# selector runs on the rescaled data, and the bandwidths are multiplied back by the
# standard deviation of the running variable. `bwselect_style` reproduces the pilot
# scale of `rdbwselect` (computed from the unscaled interquartile range) instead of
# that of `rdrobust` (computed from the rescaled data); the two agree up to rounding.
# Standard deviation computed as R's `sd` (mean rounded to double, sum of squares in
# extended precision), so that standardized data are bit-identical to rdrobust's: with
# a discrete running variable, one-ulp differences change tie-breaking in the
# nearest-neighbour variance.
function _rd_r_sd(x::AbstractVector{<:Real})
    n = length(x)
    n >= 2 || return NaN
    m = setprecision(BigFloat, 128) do
        s = sum(BigFloat.(x))
        tmp = s / n
        tmp += sum(BigFloat.(x) .- tmp) / n
        Float64(tmp)
    end
    v = setprecision(BigFloat, 128) do
        Float64(sum((BigFloat.(x) .- m) .^ 2) / (n - 1))
    end
    return sqrt(v)
end

function _rd_std_bandwidths(y, x, T, Z, C, W, methods; c, p, q, deriv, kernel, vce,
                            nnmatch, masspoints, bwcheck, bwrestrict, sharpbw,
                            scaleregul, context, bwselect_style::Bool)
    x_sd = _rd_r_sd(x)
    y_sd = _rd_r_sd(y)
    (x_sd > 0 && y_sd > 0) || throw(ArgumentError(
        "$context: stdvars = true needs non-constant outcome and running variable"))
    S = _rd_setup(y ./ y_sd, x ./ x_sd, T, Z, C, W; c=c / x_sd, p, q, deriv, kernel,
                  vce, nnmatch, masspoints, bwcheck, bwrestrict, sharpbw, context)
    bwp = nothing
    if bwselect_style
        x_iq = _rd_quantile_type2(x, 0.75) - _rd_quantile_type2(x, 0.25)
        bwp = min(1.0, (x_iq / x_sd) / 1.349)
    end
    bws, _ = _rd_select_bandwidths(S, methods; scaleregul, bwp)
    return Dict(m => x_sd .* v for (m, v) in bws)
end

# ---------------------------------------------------------------------------------------
# Input extraction from a DataFrame
# ---------------------------------------------------------------------------------------

function _rd_float(v, name)
    try
        return Vector{Float64}(v)
    catch
        throw(ArgumentError("column `$name` must be numeric"))
    end
end

"""
Extract complete cases of the columns used by an RD estimator. Returns vectors (and a
covariate matrix) free of missing values, plus the covariate names kept.
"""
function _rd_extract(data, outcome, running; treatment=nothing, covariates=Symbol[],
                     cluster=nothing, weights=nothing, context="rd_estimate")
    if cluster isa AbstractVector
        length(cluster) == 1 || throw(ArgumentError(
            "$context: multi-way clustering is not supported; pass a single column"))
        cluster = only(cluster)
    end
    covariates = Symbol[Symbol(z) for z in covariates]
    cols = Any[outcome, running, treatment, cluster, weights]
    append!(cols, covariates)
    require_columns(data, cols; context=context)
    used = Symbol[Symbol(c) for c in cols if c !== nothing]
    keep = trues(nrow(data))
    for col in used
        keep .&= .!ismissing.(data[!, col])
    end
    for col in used
        col == cluster && continue
        v = data[!, col]
        for i in eachindex(v)
            if keep[i] && v[i] isa AbstractFloat && isnan(v[i])
                keep[i] = false
            end
        end
    end
    sub = data[keep, :]
    y = _rd_float(sub[!, outcome], outcome)
    x = _rd_float(sub[!, running], running)
    T = treatment === nothing ? nothing : _rd_float(sub[!, treatment], treatment)
    Z = isempty(covariates) ? nothing :
        reduce(hcat, [_rd_float(sub[!, z], z) for z in covariates])
    C = cluster === nothing ? nothing : collect(sub[!, cluster])
    W = weights === nothing ? nothing : _rd_float(sub[!, weights], weights)
    return (; y, x, T, Z, C, W, covariates, n_dropped=nrow(data) - nrow(sub))
end

function _rd_prepare_covariates(Z, names::Vector{Symbol}, covs_drop::Bool)
    Z === nothing && return nothing, Symbol[]
    Z = Matrix{Float64}(reshape(Z, size(Z, 1), :))
    if covs_drop
        kept = _rd_drop_collinear(Z, string.(names))
        if length(kept) < size(Z, 2)
            dropped = names[setdiff(1:length(names), kept)]
            @warn "Covariates collinear with others were dropped: $(join(dropped, ", "))"
        end
        return Z[:, kept], names[kept]
    end
    return Z, names
end

# ---------------------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------------------

"""
    RDBandwidth

Result of [`rd_bandwidth`](@ref): data-driven main and pilot bandwidths for local
polynomial RD estimation and inference.

The main bandwidth ``h`` defines the local sample used for the point estimate; the pilot
bandwidth ``b`` defines the sample used to estimate the leading bias for robust bias
correction. Left and right values differ only for the two-sided selectors (`:msetwo`,
`:certwo`, `:msecomb2`, `:cercomb2`). Pass them to [`rd_estimate`](@ref) as
`h = (h_left, h_right)` and `b = (b_left, b_right)`, or let `rd_estimate` select them
with the same `bwselect`.

# Fields
- `method::Symbol`: selector requested (`:all` when every selector was computed).
- `h_left`, `h_right`: main bandwidths of the requested selector (the first row of
  `table` when `method = :all`).
- `b_left`, `b_right`: pilot (bias) bandwidths of that selector.
- `table::DataFrame`: one row per computed selector, with columns `method`, `h_left`,
  `h_right`, `b_left`, `b_right`.
- `cutoff`, `p`, `q`, `deriv`, `kernel`, `vce`: settings used.
- `n_left`, `n_right`: sample sizes on each side of the cutoff.
- `n_h_left`, `n_h_right`: observations with positive kernel weight at `h`.
- `m_left`, `m_right`: numbers of distinct running-variable values on each side.
"""
struct RDBandwidth
    method::Symbol
    h_left::Float64
    h_right::Float64
    b_left::Float64
    b_right::Float64
    table::DataFrame
    cutoff::Float64
    p::Int
    q::Int
    deriv::Int
    kernel::Symbol
    vce::Symbol
    n_left::Int
    n_right::Int
    n_h_left::Int
    n_h_right::Int
    m_left::Int
    m_right::Int
end

function Base.show(io::IO, ::MIME"text/plain", r::RDBandwidth)
    println(io, "RD bandwidth selection ($(r.method)), cutoff = $(r.cutoff)")
    println(io, "p = $(r.p), q = $(r.q), deriv = $(r.deriv), kernel = $(r.kernel), " *
                "vce = $(r.vce)")
    println(io, "Observations: left = $(r.n_left), right = $(r.n_right); within h: " *
                "left = $(r.n_h_left), right = $(r.n_h_right)")
    show(io, MIME"text/plain"(), r.table; summary=false, eltypes=false)
end

Base.show(io::IO, r::RDBandwidth) =
    @printf(io, "RDBandwidth(%s: h = (%.4g, %.4g), b = (%.4g, %.4g))", r.method,
            r.h_left, r.h_right, r.b_left, r.b_right)

"""
    rd_bandwidth(data, outcome, running; cutoff=0.0, bwselect=:mserd,
                 treatment=nothing, deriv=0, p=nothing, q=nothing,
                 kernel=:triangular, vce=:nn, nnmatch=3, cluster=nothing,
                 weights=nothing, covariates=Symbol[], scaleregul=1,
                 masspoints=:adjust, bwcheck=nothing, bwrestrict=true, sharpbw=false,
                 covs_drop=true, stdvars=false) -> RDBandwidth

Data-driven bandwidth selection for local polynomial regression discontinuity designs,
equivalent to `rdbwselect` from the R/Stata package `rdrobust`.

The bandwidth governs the bias–variance trade-off of local polynomial RD estimators. A
larger ``h`` uses more observations but lets curvature of the regression functions bias
the estimate. For a polynomial of order ``p`` estimating the jump in the ``\\nu``-th
derivative (``\\nu`` = `deriv`), the asymptotic bias is of order ``h^{p+1-\\nu}`` and
the variance of order ``1/(n h^{1+2\\nu})``, so the bandwidth that minimises the
asymptotic mean squared error is
```math
h_{\\text{MSE}} = \\left(\\frac{(1 + 2\\nu)\\,\\mathsf{V}}
    {2(p + 1 - \\nu)\\,\\mathsf{B}^2}\\right)^{1/(2p+3)} n^{-1/(2p+3)},
```
where ``\\mathsf{V}`` and ``\\mathsf{B}`` are variance and bias constants that depend on
the kernel, the density of the running variable, the conditional variances and the
``(p+1)``-th derivatives of the regression functions at the cutoff. These constants are
estimated with pilot local polynomial fits, following Calonico, Cattaneo and Titiunik
(2014) and Calonico, Cattaneo and Farrell (2020). A regularisation term in the
denominator, in the spirit of Imbens and Kalyanaraman (2012), keeps the bandwidth finite
when the estimated bias is close to zero. The same machinery gives the pilot bandwidth
``b`` for the bias estimate.

MSE-optimal selectors are `:mserd` (one common bandwidth for the RD difference, the
default), `:msetwo` (different bandwidths on each side), `:msesum` (a common bandwidth
optimal for the sum of the two regression functions), `:msecomb1` (``\\min`` of `mserd`
and `msesum`) and `:msecomb2` (on each side, the median of `mserd`, `msetwo` and
`msesum`). An MSE-optimal bandwidth is optimal for point estimation. It is too large
for conventional confidence intervals, and robust bias correction is what makes
inference valid there ([`rd_estimate`](@ref)). Calonico, Cattaneo and Farrell (2018,
2020) show that the bandwidth minimising the coverage error of robust bias-corrected
intervals shrinks at the faster rate ``n^{-1/(p+3)}`` (for ``\\nu = 0``). The
CER selectors (`:cerrd`, `:certwo`, `:cersum`, `:cercomb1`, `:cercomb2`) are
**rule-of-thumb rescalings**, as in `rdrobust`: they multiply the corresponding
MSE-optimal ``h`` by ``n^{-p/((3+p)(3+2p))}``, with ``n`` the sample size, or the number
of clusters when `cluster` is given, and keep ``b`` unchanged. The rescaled bandwidth
attains the CER-optimal rate, but its constant is not the CER-optimal constant. The
same factor is used for derivatives.

In practice, `:mserd` with robust bias-corrected inference is the standard choice for
reporting a point estimate and interval together; `:cerrd` gives intervals with smaller
coverage error at the cost of a noisier point estimate. With few distinct values of the
running variable (mass points), the selectors can return bandwidths containing very few
support points: `masspoints = :adjust` guards against this, but
[`rd_honest`](@ref) or [`rd_honest_bme`](@ref) are more appropriate for discrete
running variables. The selected bandwidth is a tuning parameter, not a window in which
the design is valid. Report the estimate's sensitivity to it with
[`rd_bandwidth_sensitivity`](@ref).

# Arguments
- `data::AbstractDataFrame`: one row per unit. Rows with a missing value in any used
  column are dropped.
- `outcome::Symbol`: outcome column.
- `running::Symbol`: running variable. Units with `running ≥ cutoff` are on the treated
  side.

# Keywords
- `cutoff::Real=0.0`: the RD threshold.
- `bwselect=:mserd`: selector listed above, or `:all` to compute every selector (see
  the `table` field of the result).
- `treatment::Union{Nothing,Symbol}=nothing`: treatment take-up in fuzzy designs. The
  bandwidth then targets the fuzzy ratio, unless `sharpbw = true`.
- `deriv::Integer=0`: order of the derivative of interest (`1` for kink designs).
- `p::Union{Nothing,Integer}=nothing`: order of the local polynomial, by default `1`
  (or `deriv + 1` when `deriv > 0`).
- `q::Union{Nothing,Integer}=nothing`: order of the bias-correction polynomial, by
  default `p + 1`.
- `kernel=:triangular`: `:triangular`, `:epanechnikov` or `:uniform`.
- `vce=:nn`: variance estimator used in the constants: `:nn` (nearest neighbours) or
  `:hc0`–`:hc3`. With `cluster`, the options are `:cr1` (default), `:cr2` or `:cr3`.
- `nnmatch::Integer=3`: number of neighbours for `vce = :nn`.
- `cluster::Union{Nothing,Symbol}=nothing`: cluster identifier. It switches the
  variance to CR1 and uses the number of clusters in the CER rescaling.
- `weights::Union{Nothing,Symbol}=nothing`: non-negative observation weights.
- `covariates::Vector{Symbol}=Symbol[]`: predetermined covariates, as in
  [`rd_estimate`](@ref). The bandwidth then targets the covariate-adjusted estimator.
- `scaleregul::Real=1`: scale of the regularisation term. `0` disables it.
- `masspoints=:adjust`: `:adjust`, `:check` or `:off` handling of repeated values of the
  running variable.
- `bwcheck::Union{Nothing,Integer}=nothing`: minimum number of distinct values on each
  side within the bandwidth. It is set to 10 automatically when mass points are
  detected.
- `bwrestrict::Bool=true`: cap bandwidths at the range of the running variable.
- `sharpbw::Bool=false`: in fuzzy designs, select bandwidths for the outcome equation
  only.
- `covs_drop::Bool=true`: drop collinear covariates.
- `stdvars::Bool=false`: select bandwidths after dividing the outcome and the running
  variable by their standard deviations (as `rdbwselect(..., stdvars = TRUE)`). The
  bandwidths are reported on the original scale of the running variable, and the pilot
  computations become invariant to the units of the data.

# Returns
- [`RDBandwidth`](@ref) with the bandwidths of the requested selector and, in `table`,
  of every selector computed.

# Examples
```julia
using DrSnow, CSV, DataFrames
senate = CSV.read(pkgdir(DrSnow, "test", "validation", "rd", "senate.csv"), DataFrame)
bw = rd_bandwidth(senate, :vote, :margin)
bw.h_left, bw.b_left
rd_bandwidth(senate, :vote, :margin; bwselect=:all).table   # MSE and CER selectors
```

# References
- Imbens, G., & Kalyanaraman, K. (2012). Optimal bandwidth choice for the regression
  discontinuity estimator. *Review of Economic Studies*, 79(3), 933–959.
- Calonico, S., Cattaneo, M. D., & Titiunik, R. (2014). Robust nonparametric confidence
  intervals for regression-discontinuity designs. *Econometrica*, 82(6), 2295–2326.
- Calonico, S., Cattaneo, M. D., & Farrell, M. H. (2018). On the effect of bias
  estimation on coverage accuracy in nonparametric inference. *Journal of the American
  Statistical Association*, 113(522), 767–779.
- Calonico, S., Cattaneo, M. D., & Farrell, M. H. (2020). Optimal bandwidth choice for
  robust bias-corrected inference in regression discontinuity designs. *The
  Econometrics Journal*, 23(2), 192–210.
- Calonico, S., Cattaneo, M. D., Farrell, M. H., & Titiunik, R. (2017). rdrobust:
  Software for regression-discontinuity designs. *The Stata Journal*, 17(2), 372–404.
"""
function rd_bandwidth(data::AbstractDataFrame, outcome::Symbol, running::Symbol;
                      cutoff::Real=0.0, bwselect=:mserd, treatment=nothing,
                      deriv::Integer=0, p::Union{Nothing,Integer}=nothing,
                      q::Union{Nothing,Integer}=nothing, kernel=:triangular, vce=:nn,
                      nnmatch::Integer=3, cluster=nothing, weights=nothing,
                      covariates=Symbol[], scaleregul::Real=1, masspoints=:adjust,
                      bwcheck::Union{Nothing,Integer}=nothing, bwrestrict::Bool=true,
                      sharpbw::Bool=false, covs_drop::Bool=true, stdvars::Bool=false)
    ctx = "rd_bandwidth"
    E = _rd_extract(data, outcome, running; treatment, covariates, cluster, weights,
                    context=ctx)
    Z, _ = _rd_prepare_covariates(E.Z, E.covariates, covs_drop)
    p = p === nothing ? (deriv == 0 ? 1 : deriv + 1) : Int(p)
    q = q === nothing ? p + 1 : Int(q)
    all_methods = Symbol(lowercase(string(bwselect))) === :all
    method = all_methods ? :all : _rd_bwselect(bwselect)
    length(E.x) >= 20 || throw(ArgumentError(
        "$ctx: at least 20 observations are needed for bandwidth selection"))
    S = _rd_setup(E.y, E.x, E.T, Z, E.C, E.W; c=Float64(cutoff), p, q,
                  deriv=Int(deriv), kernel=_rd_kernel(kernel), vce=_rd_vce(vce),
                  nnmatch=Int(nnmatch), masspoints=_rd_masspoints(masspoints),
                  bwcheck, bwrestrict, sharpbw, context=ctx)
    methods = all_methods ? collect(_RD_BWSELECT) : [method]
    bws = if stdvars
        _rd_std_bandwidths(E.y, E.x, E.T, Z, E.C, E.W, methods; c=Float64(cutoff), p, q,
                           deriv=Int(deriv), kernel=S.kernel, vce=_rd_vce(vce),
                           nnmatch=Int(nnmatch), masspoints=S.masspoints, bwcheck,
                           bwrestrict, sharpbw, scaleregul=Float64(scaleregul),
                           context=ctx, bwselect_style=true)
    else
        first(_rd_select_bandwidths(S, methods; scaleregul=Float64(scaleregul)))
    end
    tab = DataFrame(method=Symbol[], h_left=Float64[], h_right=Float64[],
                    b_left=Float64[], b_right=Float64[])
    for m in methods
        push!(tab, (m, bws[m]...))
    end
    h_l, h_r, b_l, b_r = bws[first(methods)]
    n_h_l = count(>(0), _rd_kweight(S.L.X, S.c, h_l, S.kernel))
    n_h_r = count(>(0), _rd_kweight(S.R.X, S.c, h_r, S.kernel))
    return RDBandwidth(method, h_l, h_r, b_l, b_r, tab, S.c, p, q, Int(deriv), S.kernel,
                       S.vce, S.N_l, S.N_r, n_h_l, n_h_r, S.M_l, S.M_r)
end
