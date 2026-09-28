# Internal IV engine: estimation sample, partialling-out, covariance construction.
#
# Every IV statistic in this area (first-stage diagnostics, weak-IV-robust tests,
# overidentification, sensitivity) is computed from an `_IVDesign`: the outcome,
# endogenous regressors and excluded instruments after
#   1. restricting to the estimation sample of the `FixedEffectModels.reg` fit,
#   2. absorbing fixed effects (weighted, via `FixedEffectModels.partial_out`),
#   3. multiplying by sqrt(weights), and
#   4. projecting out the included exogenous regressors (covariates + intercept).
# After these steps every regression is unweighted OLS without an intercept, and by
# Frisch–Waugh–Lovell the coefficients of interest (and their HC1 / cluster
# covariance matrices, with the degrees-of-freedom corrections below) coincide with
# those of the full regression. The corrections replicate FixedEffectModels/Vcov.jl:
#   HC1:      n / (n - K)
#   cluster:  (n - 1) / (n - K) * G / (G - 1),  G = min over clustering dimensions
#   simple:   σ² = e'e / (n - K)
# where K counts all coefficients of the regression (incl. partialled covariates and
# the intercept) plus the fixed-effect dof (a fixed effect nested in a cluster
# dimension counts as 1). Later IV estimators (LIML, JIVE, shift-share, ...) are
# meant to be built on the same object.

const _IV_FE_TOL = 1e-10

"""
    _IVDesign

Internal: partialled-out, weight-transformed IV design (see `src/iv/design.jl`).
"""
struct _IVDesign
    y::Vector{Float64}          # outcome
    D::Matrix{Float64}          # endogenous regressors, n × p
    Z::Matrix{Float64}          # excluded instruments, n × k
    n::Int
    k_exog::Int                 # rank of included exogenous block (incl. intercept)
    dof_fes::Int                # dof absorbed by fixed effects (nesting-adjusted)
    vcov_kind::Symbol           # :simple, :robust or :cluster
    groups::Vector{Vector{Int}} # cluster codes per clustering dimension (1:G)
    sqrtw::Vector{Float64}      # sqrt of weights (ones if unweighted)
    esample::BitVector          # estimation sample, relative to the input data
end

_iv_nclusters(des::_IVDesign) =
    des.vcov_kind === :cluster ? minimum(maximum.(des.groups)) : 0

"""Residual dof used in small-sample corrections for a regression with `kreg`
regressors of interest on the partialled data."""
_iv_resid_dof(des::_IVDesign, kreg::Integer) = des.n - kreg - des.k_exog - des.dof_fes

"""Reference dof for t / F statistics: G - 1 under clustering, else residual dof."""
function _iv_ref_dof(des::_IVDesign, kreg::Integer)
    des.vcov_kind === :cluster && return float(_iv_nclusters(des) - 1)
    return float(max(1, _iv_resid_dof(des, kreg)))
end

function _iv_vcov_label(des::_IVDesign)
    des.vcov_kind === :simple && return "homoskedastic"
    des.vcov_kind === :robust && return "heteroskedasticity-robust (HC1)"
    return "cluster-robust (" * join(string.(length.(unique.(des.groups))), " × ") *
           " clusters)"
end

# ---------------------------------------------------------------------------
# Keyword handling
# ---------------------------------------------------------------------------

"""
Normalize the `cluster` / `vcov` keywords into a `Vcov.CovarianceEstimator`.
`vcov` wins when both are given. The IV default is HC1 (`Vcov.robust()`).
"""
function _iv_vcov_estimator(cluster, vcov)
    if vcov !== nothing
        vcov isa FixedEffectModels.CovarianceEstimator ||
            throw(ArgumentError("`vcov` must be Vcov.simple(), Vcov.robust() or " *
                                "Vcov.cluster(...), got $(typeof(vcov))"))
        if vcov isa Vcov.RobustCovariance && hasproperty(vcov, :correction) &&
           getproperty(vcov, :correction) !== :hc1
            throw(ArgumentError("only HC1 (Vcov.robust()) is supported for IV models"))
        end
        return vcov
    end
    cluster === nothing && return Vcov.robust()
    cl = _as_symbols(cluster)
    isempty(cl) && return Vcov.robust()
    return Vcov.cluster(cl...)
end

function _iv_vcov_kind(v)
    v isa Vcov.SimpleCovariance && return :simple
    v isa Vcov.RobustCovariance && return :robust
    v isa Vcov.ClusterCovariance && return :cluster
    throw(ArgumentError("unsupported covariance estimator $(typeof(v))"))
end

_iv_cluster_names(v) = v isa Vcov.ClusterCovariance ? collect(Vcov.names(v)) : Symbol[]

function _iv_check_numeric(data, cols, context)
    for c in cols
        T = nonmissingtype(eltype(data[!, c]))
        T <: Real || throw(ArgumentError("$(context): column `$c` must be numeric " *
                                         "(element type $T)"))
    end
    return nothing
end

function _iv_codes(v::AbstractVector)
    d = Dict{Any,Int}()
    out = Vector{Int}(undef, length(v))
    for (i, x) in enumerate(v)
        out[i] = get!(d, x, length(d) + 1)
    end
    return out
end

# ---------------------------------------------------------------------------
# Building the design
# ---------------------------------------------------------------------------

"""
    _iv_build_design(sub, outcome, endo, inst, covariates, fe, weights, vce) -> _IVDesign

`sub` must already be the estimation sample (no missing values in used columns).
"""
function _iv_build_design(sub::AbstractDataFrame, outcome::Symbol, endo::Vector{Symbol},
                          inst::Vector{Symbol}, covariates::Vector{Symbol},
                          fe::Vector{Symbol}, weights, vce, esample::BitVector;
                          check_endogenous::Bool=true)
    n = nrow(sub)
    y = Float64.(sub[!, outcome])
    D = Matrix{Float64}(hcat([Float64.(sub[!, c]) for c in endo]...))
    Z = Matrix{Float64}(hcat([Float64.(sub[!, c]) for c in inst]...))
    W = _iv_exog_matrix(sub, covariates, isempty(fe))
    w = weights === nothing ? ones(n) : Float64.(sub[!, weights])
    all(>(0), w) || throw(ArgumentError("weights must be strictly positive"))
    sqrtw = sqrt.(w)

    # 1. absorb fixed effects (weighted within transformation)
    if !isempty(fe)
        A0 = hcat(y, D, Z, W)
        A = _iv_absorb_fe(A0, sub, fe, weights)
        # as in FixedEffectModels: a column whose squared norm falls below 1e-10 of
        # its original value is collinear with the fixed effects
        absorbed = [sum(abs2, view(A, :, j)) <= 1e-10 * sum(abs2, view(A0, :, j))
                    for j in 1:size(A, 2)]
        y = A[:, 1]
        p, k = size(D, 2), size(Z, 2)
        any(absorbed[(2 + p):(1 + p + k)]) &&
            throw(ArgumentError("an instrument is collinear with the fixed effects"))
        check_endogenous && any(absorbed[2:(1 + p)]) &&
            throw(ArgumentError("an endogenous regressor is collinear with the fixed " *
                                "effects"))
        D = A[:, 2:(1 + p)]
        Z = A[:, (2 + p):(1 + p + k)]
        keepW = .!absorbed[(2 + p + k):end]
        W = A[:, (2 + p + k):end][:, keepW]
    end
    # 2. weights
    y = y .* sqrtw
    D = D .* sqrtw
    Z = Z .* sqrtw
    W = W .* sqrtw
    # 3. project out included exogenous regressors
    k_exog = 0
    if size(W, 2) > 0
        Q, k_exog = _iv_orthobasis(W)
        if k_exog > 0
            y = y - Q * (Q' * y)
            D = D - Q * (Q' * D)
            Z = Z - Q * (Q' * Z)
        end
    end
    _iv_check_rank(Z, "instruments", "after partialling out covariates and fixed " *
                   "effects; an instrument is collinear with the controls")
    check_endogenous && _iv_check_rank(D, "endogenous regressors", "after partialling " *
                                       "out covariates and fixed effects")

    kind = _iv_vcov_kind(vce)
    groups = Vector{Int}[_iv_codes(sub[!, c]) for c in _iv_cluster_names(vce)]
    if kind === :cluster
        G = minimum(maximum.(groups))
        G >= 2 || throw(ArgumentError("cluster-robust inference needs at least 2 " *
                                      "clusters"))
    end
    dof_fes = _iv_dof_fes(sub, fe, groups)
    return _IVDesign(y, D, Z, n, k_exog, dof_fes, kind, groups, sqrtw, esample)
end

function _iv_exog_matrix(sub, covariates::Vector{Symbol}, intercept::Bool)
    n = nrow(sub)
    if isempty(covariates)
        return intercept ? ones(n, 1) : zeros(n, 0)
    end
    f = make_formula(:__iv_lhs__, covariates; intercept=intercept)
    # only the right-hand side is needed (the lhs placeholder is never materialized)
    ts = f.rhs isa Tuple ? f.rhs : (f.rhs,)
    sch = StatsModels.schema(ts, sub)
    rhs = StatsModels.MatrixTerm(StatsModels.apply_schema(ts, sch,
                                                          StatsModels.StatisticalModel))
    X = StatsModels.modelcols(rhs, sub)
    return X isa AbstractVector ? reshape(Float64.(X), :, 1) : Matrix{Float64}(X)
end

"""Orthonormal basis of the column space of `W` (pivoted QR, relative tolerance)."""
function _iv_orthobasis(W::AbstractMatrix)
    size(W, 2) == 0 && return zeros(size(W, 1), 0), 0
    F = qr(W, ColumnNorm())
    R = F.R
    d = abs.(diag(R))
    tol = maximum(d; init=0.0) * max(size(W)...) * eps(Float64) * 1e3
    r = count(>(tol), d)
    r == 0 && return zeros(size(W, 1), 0), 0
    Q = Matrix(F.Q)[:, 1:r]
    return Q, r
end

function _iv_check_rank(M::AbstractMatrix, what, ctx)
    size(M, 2) == 0 && return nothing
    _, r = _iv_orthobasis(M)
    r == size(M, 2) || throw(ArgumentError("the $(what) are not of full column rank " *
                                           ctx))
    return nothing
end

function _iv_absorb_fe(A::Matrix{Float64}, sub, fe::Vector{Symbol}, weights)
    names = [Symbol("__iv_col_", j) for j in 1:size(A, 2)]
    tmp = DataFrame(A, names)
    for f in fe
        tmp[!, f] = sub[!, f]
    end
    weights === nothing || (tmp[!, :__iv_weights__] = sub[!, weights])
    lhs = reduce(+, StatsModels.term.(names))
    rhs = reduce(+, [FixedEffectModels.fe(f) for f in fe])
    res = FixedEffectModels.partial_out(tmp, lhs ~ rhs;
                                        weights=weights === nothing ? nothing :
                                                :__iv_weights__,
                                        tol=_IV_FE_TOL, maxiter=100_000)
    out = Matrix{Float64}(first(res))
    size(out) == size(A) || error("internal: partial_out dropped observations")
    return out
end

function _iv_dof_fes(sub, fe::Vector{Symbol}, groups::Vector{Vector{Int}})
    total = 0
    for f in fe
        codes = _iv_codes(sub[!, f])
        ng = maximum(codes; init=0)
        nested = any(g -> _iv_is_nested(codes, g), groups)
        total += nested ? 1 : ng
    end
    return total
end

"""`true` when every level of `inner` lies within a single level of `outer`."""
function _iv_is_nested(inner::Vector{Int}, outer::Vector{Int})
    m = Dict{Int,Int}()
    for (a, b) in zip(inner, outer)
        c = get!(m, a, b)
        c == b || return false
    end
    return true
end

# ---------------------------------------------------------------------------
# Covariance construction
# ---------------------------------------------------------------------------

"""Sum of scores within each group: returns G × m matrix."""
function _iv_group_sums(S::AbstractMatrix, g::Vector{Int})
    G = maximum(g)
    out = zeros(G, size(S, 2))
    @inbounds for j in 1:size(S, 2), i in 1:size(S, 1)
        out[g[i], j] += S[i, j]
    end
    return out
end

"""
    _iv_meat(des, S, kreg) -> Matrix

Robust (HC1) or (multiway) cluster-robust "meat" `Σ s_i s_i'` for the n × m score
matrix `S` of a regression with `kreg` regressors of interest, including the
Vcov.jl small-sample corrections.
"""
function _iv_meat(des::_IVDesign, S::AbstractMatrix, kreg::Integer)
    n = des.n
    dof = _iv_resid_dof(des, kreg)
    dof > 0 || throw(ArgumentError("not enough observations: residual dof = $dof"))
    if des.vcov_kind === :robust
        return Symmetric(S' * S) .* (n / dof)
    elseif des.vcov_kind === :cluster
        M = _iv_cluster_sum(S, des.groups)
        G = _iv_nclusters(des)
        return Symmetric(M) .* ((n - 1) / dof * G / (G - 1))
    else
        throw(ArgumentError("internal: meat requested for homoskedastic covariance"))
    end
end

"""
Inclusion–exclusion sum `Σ_g (Σ_{i∈g} sᵢ)(Σ_{i∈g} sᵢ)'` over all non-empty
intersections of the clustering dimensions (Cameron, Gelbach & Miller 2011), without
small-sample factor. May be indefinite for multiway clustering.
"""
function _iv_cluster_sum(S::AbstractMatrix, groups::Vector{Vector{Int}})
    m = size(S, 2)
    M = zeros(m, m)
    L = length(groups)
    for mask in 1:(2^L - 1)
        dims = [j for j in 1:L if (mask >> (j - 1)) & 1 == 1]
        g = length(dims) == 1 ? groups[dims[1]] :
            _iv_codes(collect(zip((groups[j] for j in dims)...)))
        C = _iv_group_sums(S, g)
        M .+= (-1)^(length(dims) - 1) .* (C' * C)
    end
    return M
end

"""Make a (possibly indefinite multiway-cluster) covariance matrix PSD."""
function _iv_psd(V::AbstractMatrix)
    S = Symmetric(Matrix(V))
    E = eigen(S)
    all(>=(0), E.values) && return Matrix(S)
    return E.vectors * Diagonal(max.(E.values, 0.0)) * E.vectors'
end

"""
    _iv_ols(des, Y, X) -> (B, V, E)

Multivariate OLS of the columns of `Y` (n × q) on `X` (n × m) on the partialled data,
returning coefficients `B` (m × q), the joint covariance `V` of `vec(B)` (equation
blocks stacked, using the design's covariance type) and residuals `E`.
"""
function _iv_ols(des::_IVDesign, Y::AbstractVecOrMat, X::AbstractMatrix)
    Ym = Y isa AbstractVector ? reshape(Y, :, 1) : Y
    m, q = size(X, 2), size(Ym, 2)
    XtX = Symmetric(X' * X)
    bread = inv(XtX)
    B = bread * (X' * Ym)
    E = Ym - X * B
    if des.vcov_kind === :simple
        Σ = (E' * E) ./ _iv_resid_dof(des, m)
        V = kron(Σ, Matrix(bread))
    else
        S = hcat([X .* E[:, j] for j in 1:q]...)
        Bk = kron(Matrix(1.0I, q, q), Matrix(bread))
        V = _iv_psd(Bk * _iv_meat(des, S, m) * Bk)
    end
    return B, Matrix(Symmetric(V)), E
end

"""
    _iv_tsls(des, y, D, Z) -> (β, V, e, Dhat)

2SLS on the partialled data with the design's covariance type. Reproduces the
coefficients and covariance of the endogenous regressors in the full model.
"""
function _iv_tsls(des::_IVDesign, y::AbstractVector, D::AbstractMatrix, Z::AbstractMatrix)
    Π = Z \ D
    Dhat = Z * Π
    H = Symmetric(Dhat' * Dhat)
    Hinv = inv(H)
    β = Hinv * (Dhat' * y)
    e = y - D * β
    p = size(D, 2)
    if des.vcov_kind === :simple
        V = Matrix(Hinv) .* (sum(abs2, e) / _iv_resid_dof(des, p))
    else
        V = _iv_psd(Hinv * _iv_meat(des, Dhat .* e, p) * Hinv)
    end
    return β, Matrix(Symmetric(V)), e, Dhat
end

"""F-type Wald statistic `b'V⁻¹b / q` (NaN-safe pseudo-inverse)."""
function _iv_wald_F(b::AbstractVector, V::AbstractMatrix)
    return dot(b, pinv(Matrix(Symmetric(V))) * b) / length(b)
end

"""`@printf` with a runtime (e.g. concatenated) format string."""
_iv_printf(io::IO, fmt::AbstractString, args...) =
    Printf.format(io, Printf.Format(fmt), args...)
