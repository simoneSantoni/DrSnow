# Ingestion of long panel data into the wide arrays used by the longitudinal
# estimators. Units are matched by key and periods by sorted value, never by row
# position; every unit must be observed in every period.

struct _LGData
    n::Int
    K::Int
    ids::Vector{Any}                   # unit labels in sorted order
    times::Vector{Any}                 # period labels in sorted order
    A::Matrix{Float64}                 # n × K treatments
    L::Vector{Matrix{Float64}}         # per period, n × p time-varying covariates
    W::Matrix{Float64}                 # n × p_w baseline covariates
    y::Vector{Float64}                 # final outcome (original scale)
    covariate_names::Vector{Symbol}
    baseline_names::Vector{Symbol}
    cluster::Union{Nothing,Vector{Int}}
    n_clusters::Int
    rows::Matrix{Int}                  # n × K row of each unit–period in the data
end

"""Float64 values of `col` at `rows`; errors on missing or non-numeric values."""
function _lg_values(df, col::Symbol, rows, context::AbstractString, what::AbstractString)
    v = df[rows, col]
    any(ismissing, v) &&
        throw(ArgumentError("$context: missing values in $what column $col"))
    T = nonmissingtype(eltype(v))
    T <: Real || throw(ArgumentError("$context: $what column $col must be numeric " *
                                     "(got $T); encode categorical variables as " *
                                     "dummy columns"))
    out = Float64.(v)
    all(isfinite, out) ||
        throw(ArgumentError("$context: non-finite values in $what column $col"))
    return out
end

function _lg_sorted_unique(v, what, context)
    any(ismissing, v) &&
        throw(ArgumentError("$context: missing values in the $what column"))
    u = unique(v)
    try
        sort!(u)
    catch
        what == "time" &&
            throw(ArgumentError("$context: the time column must be sortable"))
    end
    return u
end

function _lg_data(data, outcome::Symbol, treatment::Symbol, unit::Symbol, time::Symbol;
                  covariates, baseline, cluster, context::AbstractString)
    df = data isa DataFrame ? data : DataFrame(data)
    covs = Symbol[Symbol(c) for c in covariates]
    base = Symbol[Symbol(c) for c in baseline]
    cl = cluster === nothing ? Symbol[] : [Symbol(cluster)]
    require_columns(df, vcat([outcome, treatment, unit, time], covs, base, cl);
                    context=context)
    both = intersect(covs, base)
    isempty(both) ||
        throw(ArgumentError("$context: $(join(both, ", ")) listed as both time-varying " *
                            "covariates and baseline covariates"))
    (treatment in covs || treatment in base || outcome in covs || outcome in base) &&
        throw(ArgumentError("$context: the treatment and outcome columns cannot be " *
                            "covariates"))
    ids = _lg_sorted_unique(df[!, unit], "unit", context)
    times = _lg_sorted_unique(df[!, time], "time", context)
    n, K = length(ids), length(times)
    n >= 2 || throw(ArgumentError("$context: need at least two units"))
    uidx = Dict(ids[i] => i for i in 1:n)
    tidx = Dict(times[t] => t for t in 1:K)
    rowat = zeros(Int, n, K)
    u = df[!, unit]
    tm = df[!, time]
    for r in 1:nrow(df)
        i = uidx[u[r]]
        t = tidx[tm[r]]
        rowat[i, t] == 0 ||
            throw(ArgumentError("$context: duplicated unit–period row for unit " *
                                "$(ids[i]) at period $(times[t])"))
        rowat[i, t] = r
    end
    if any(==(0), rowat)
        miss = count(==(0), rowat)
        i = findfirst(==(0), rowat)
        throw(ArgumentError("$context: every unit must be observed in every period; " *
                            "$miss unit–period cell(s) are missing (e.g. unit " *
                            "$(ids[i[1]]) at period $(times[i[2]])). Balance the panel " *
                            "before estimation"))
    end
    A = Matrix{Float64}(undef, n, K)
    L = Vector{Matrix{Float64}}(undef, K)
    for t in 1:K
        rows = rowat[:, t]
        a = _lg_values(df, treatment, rows, context, "treatment")
        all(v -> v == 0 || v == 1, a) ||
            throw(ArgumentError("$context: the treatment must be binary (0/1)"))
        (any(==(1), a) && any(==(0), a)) ||
            throw(ArgumentError("$context: the treatment does not vary in period " *
                                "$(times[t]); the treatment mechanism is not identified"))
        A[:, t] .= a
        Lt = Matrix{Float64}(undef, n, length(covs))
        for (j, c) in enumerate(covs)
            Lt[:, j] .= _lg_values(df, c, rows, context, "covariate")
        end
        L[t] = Lt
    end
    first_rows = rowat[:, 1]
    W = Matrix{Float64}(undef, n, length(base))
    for (j, c) in enumerate(base)
        w1 = _lg_values(df, c, first_rows, context, "baseline")
        for t in 2:K
            wt = _lg_values(df, c, rowat[:, t], context, "baseline")
            wt == w1 || throw(ArgumentError("$context: baseline covariate $c varies " *
                                            "within units; list it in `covariates` " *
                                            "if it is time-varying"))
        end
        W[:, j] .= w1
    end
    y = _lg_values(df, outcome, rowat[:, K], context, "outcome")
    cid, G = nothing, 0
    if cluster !== nothing
        c = Symbol(cluster)
        v1 = df[first_rows, c]
        any(ismissing, v1) && throw(ArgumentError("$context: missing cluster values"))
        for t in 2:K
            df[rowat[:, t], c] == v1 ||
                throw(ArgumentError("$context: the cluster column must be constant " *
                                    "within units"))
        end
        cid, G = _ml_group_index(v1)
    end
    return _LGData(n, K, Vector{Any}(ids), Vector{Any}(times), A, L, W, y, covs, base,
                   cid, G, rowat)
end

"""Rows `idx` of the data (with repetition, for the bootstrap)."""
function _lg_subset(d::_LGData, idx::AbstractVector{Int})
    cl = d.cluster === nothing ? nothing : d.cluster[idx]
    return _LGData(length(idx), d.K, d.ids[idx], d.times, d.A[idx, :],
                   [Lt[idx, :] for Lt in d.L], d.W[idx, :], d.y[idx], d.covariate_names,
                   d.baseline_names, cl, d.n_clusters, zeros(Int, 0, 0))
end

"""
Design matrix of the history at period `t`: baseline covariates, the time-varying
covariates of periods `1:t` and the treatments of periods `1:t-1` (`1:t` when
`current = true`), with the treatments taken from `A` (the observed ones when
`nothing`).
"""
function _lg_design(d::_LGData, t::Int, A::Union{Nothing,AbstractMatrix}; current::Bool)
    T = A === nothing ? d.A : A
    tA = current ? t : t - 1
    cols = Matrix{Float64}[d.W]
    for s in 1:t
        push!(cols, d.L[s])
    end
    tA >= 1 && push!(cols, Matrix{Float64}(T[:, 1:tA]))
    return reduce(hcat, cols)
end
