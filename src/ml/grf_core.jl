# Generalized random forests (Athey, Tibshirani & Wager 2019): the tree-growing and
# prediction engine, a line-by-line port of the C++ core of the R package grf 2.6.1
# (ForestTrainer, TreeTrainer, RandomSampler, the regression / instrumental
# relabeling and splitting rules, honesty with leaf pruning, and the optimized
# prediction strategies with bootstrap-of-little-bags variance and the objective-Bayes
# debiaser). Only the random-number streams differ: every CI group of trees draws
# from its own `Xoshiro` seeded up front, so fits are reproducible and identical with
# any number of threads.
#
# Indices are 1-based; a node id of 0 means "no child". Missing covariate values are
# NaN and are handled as in grf (missingness incorporated in attributes: at each
# split NaNs are sent to the side that maximizes the criterion).

# ------------------------------------------------------------------ data, options

struct _GRFData
    X::Matrix{Float64}          # covariates (NaN = missing)
    y::Vector{Float64}          # outcome (centred for causal / instrumental forests)
    w::Vector{Float64}          # treatment (empty for regression forests)
    z::Vector{Float64}          # instrument (empty for regression forests)
    wt::Vector{Float64}         # sample weights (ones when none)
    has_missing::Bool
end

function _GRFData(X, y; w=Float64[], z=Float64[], wt=nothing)
    n = size(X, 1)
    weights = wt === nothing ? ones(n) : Vector{Float64}(wt)
    return _GRFData(Matrix{Float64}(X), Vector{Float64}(y), Vector{Float64}(w),
                    Vector{Float64}(z), weights, any(isnan, X))
end

struct _GRFOptions
    kind::Symbol                     # :regression or :instrumental
    num_trees::Int
    ci_group_size::Int
    sample_fraction::Float64
    mtry::Int
    min_node_size::Int
    honesty::Bool
    honesty_fraction::Float64
    honesty_prune_leaves::Bool
    alpha::Float64
    imbalance_penalty::Float64
    stabilize_splits::Bool
    reduced_form_weight::Float64
    clusters::Vector{Vector{Int}}    # samples of each cluster; empty = no clustering
    samples_per_cluster::Int
end

"""Copy of `o` with some fields replaced (keyword names are field names)."""
function _grf_options(o::_GRFOptions; kwargs...)
    vals = [haskey(kwargs, f) ? kwargs[f] : getfield(o, f) for f in fieldnames(_GRFOptions)]
    return _GRFOptions(vals...)
end

_grf_nvalues(kind::Symbol) = kind === :regression ? 2 : 7

# ----------------------------------------------------------------------- trees

struct _GRFTree
    root::Int
    left::Vector{Int32}
    right::Vector{Int32}
    split_var::Vector{Int32}
    split_value::Vector{Float64}
    send_missing_left::BitVector
    leaf_samples::Vector{Vector{Int32}}
    drawn::Vector{Int32}             # samples used by the tree (in-bag, both halves)
    values::Matrix{Float64}          # precomputed leaf statistics (types × nodes)
    nonempty::BitVector              # leaf has prediction values
end

_grf_is_leaf(t::_GRFTree, node) = t.left[node] == 0 && t.right[node] == 0

function _grf_find_leaf(t::_GRFTree, X::AbstractMatrix, i::Integer)
    node = t.root
    @inbounds while !(t.left[node] == 0 && t.right[node] == 0)
        v = X[i, t.split_var[node]]
        sv = t.split_value[node]
        if v <= sv || (t.send_missing_left[node] && isnan(v)) || (isnan(sv) && isnan(v))
            node = Int(t.left[node])
        else
            node = Int(t.right[node])
        end
    end
    return node
end

struct _GRFForest
    trees::Vector{_GRFTree}
    opts::_GRFOptions
    num_variables::Int
end

# --------------------------------------------------------------------- sampler

# RandomSampler::sample: floor(N · fraction) of 1:N after a shuffle.
function _grf_sample(rng::AbstractRNG, N::Int, fraction::Float64)
    k = floor(Int, N * fraction)
    return randperm(rng, N)[1:k]
end

_grf_sample_clusters(rng, opts::_GRFOptions, n::Int, fraction::Float64) =
    _grf_sample(rng, isempty(opts.clusters) ? n : length(opts.clusters), fraction)

# RandomSampler::subsample: the first ceil(|s| · fraction) of a shuffled copy
# (and the rest, for the honesty split).
function _grf_subsample(rng::AbstractRNG, s::Vector{Int}, fraction::Float64)
    sh = shuffle(rng, s)
    k = ceil(Int, length(s) * fraction)
    return sh[1:k], sh[(k + 1):end]
end

function _grf_sample_from_clusters(rng::AbstractRNG, opts::_GRFOptions, cl::Vector{Int})
    isempty(opts.clusters) && return copy(cl)
    out = Int[]
    spc = opts.samples_per_cluster
    for c in cl
        s = opts.clusters[c]
        if length(s) <= spc
            append!(out, s)
        else
            append!(out, shuffle(rng, s)[1:spc])
        end
    end
    return out
end

function _grf_samples_in_clusters(opts::_GRFOptions, cl::Vector{Int})
    isempty(opts.clusters) && return copy(cl)
    return reduce(vcat, (opts.clusters[c] for c in cl); init=Int[])
end

# --------------------------------------------------------------- tree training

# Per-task scratch space for split search.
struct _GRFWorkspace
    vals::Vector{Float64}
    perm::Vector{Int}
    sv::Vector{Float64}
    ss::Vector{Int}
    uniq::Vector{Float64}
    counter::Vector{Int}
    wsums::Vector{Float64}
    sums::Vector{Float64}
    nsmall::Vector{Int}
    sumsz::Vector{Float64}
    sumsz2::Vector{Float64}
    resp::Vector{Float64}
end

function _GRFWorkspace(n::Int)
    return _GRFWorkspace(zeros(n), zeros(Int, n), zeros(n), zeros(Int, n), zeros(n),
                         zeros(Int, n), zeros(n), zeros(n), zeros(Int, n), zeros(n),
                         zeros(n), zeros(n))
end

mutable struct _GRFBest
    decrease::Float64
    var::Int
    value::Float64
    send_left::Bool
end

_grf_nan_lt(a::Float64, b::Float64) = a < b || (isnan(a) && !isnan(b))

# Sort the node samples by covariate `var` (NaN first, stable) and collect the
# distinct values (Data::get_all_values). Returns the number of distinct values.
function _grf_sorted_values!(ws::_GRFWorkspace, X, S::Vector{Int}, var::Int,
                             has_missing::Bool)
    m = length(S)
    vals = ws.vals
    @inbounds for k in 1:m
        vals[k] = X[S[k], var]
    end
    perm = view(ws.perm, 1:m)
    if has_missing
        sortperm!(perm, view(vals, 1:m); lt=_grf_nan_lt)
    else
        sortperm!(perm, view(vals, 1:m))
    end
    nu = 0
    @inbounds for k in 1:m
        s = S[perm[k]]
        v = vals[perm[k]]
        ws.ss[k] = s
        ws.sv[k] = v
        if nu == 0 || !(v == ws.uniq[nu] || (isnan(v) && isnan(ws.uniq[nu])))
            nu += 1
            ws.uniq[nu] = v
        end
    end
    return nu
end

# RegressionSplittingRule::find_best_split_value
function _grf_split_value_regression!(best::_GRFBest, ws::_GRFWorkspace, data::_GRFData,
                                      S::Vector{Int}, var::Int, wsum_node::Float64,
                                      sum_node::Float64, min_child::Int,
                                      penalty::Float64)
    m = length(S)
    nu = _grf_sorted_values!(ws, data.X, S, var, data.has_missing)
    nu < 2 && return nothing
    nsplits = nu - 1
    counter, wsums, sums = ws.counter, ws.wsums, ws.sums
    @inbounds for i in 1:nsplits
        counter[i] = 0
        wsums[i] = 0.0
        sums[i] = 0.0
    end
    n_missing = 0
    w_missing = 0.0
    s_missing = 0.0
    idx = 1
    wt, resp, sv, ss = data.wt, ws.resp, ws.sv, ws.ss
    @inbounds for k in 1:(m - 1)
        s = ss[k]
        v = sv[k]
        sw = wt[s]
        if isnan(v)
            w_missing += sw
            s_missing += sw * resp[s]
            n_missing += 1
        else
            wsums[idx] += sw
            sums[idx] += sw * resp[s]
            counter[idx] += 1
        end
        nv = sv[k + 1]
        if v != nv && !isnan(nv)
            idx += 1
        end
    end
    for send_left in (true, false)
        if send_left
            n_left = n_missing
            w_left = w_missing
            s_left = s_missing
        else
            n_missing == 0 && break
            n_left = 0
            w_left = 0.0
            s_left = 0.0
        end
        @inbounds for i in 1:nsplits
            (i == 1 && !send_left) && continue
            n_left += counter[i]
            w_left += wsums[i]
            s_left += sums[i]
            n_left < min_child && continue
            n_right = m - n_left
            n_right < min_child && break
            w_right = wsum_node - w_left
            s_right = sum_node - s_left
            decrease = s_left * s_left / w_left + s_right * s_right / w_right
            decrease -= penalty * (1.0 / n_left + 1.0 / n_right)
            if decrease > best.decrease
                best.decrease = decrease
                best.var = var
                best.value = ws.uniq[i]
                best.send_left = send_left
            end
        end
    end
    return nothing
end

# InstrumentalSplittingRule::find_best_split_value
function _grf_split_value_instrumental!(best::_GRFBest, ws::_GRFWorkspace,
                                        data::_GRFData, S::Vector{Int}, var::Int,
                                        wsum_node, sum_node, mean_z, n_small_node,
                                        sum_z_node, sum_z2_node, min_child_size,
                                        min_node_size::Int, penalty::Float64)
    m = length(S)
    nu = _grf_sorted_values!(ws, data.X, S, var, data.has_missing)
    nu < 2 && return nothing
    nsplits = nu - 1
    counter, wsums, sums = ws.counter, ws.wsums, ws.sums
    nsmall, sumsz, sumsz2 = ws.nsmall, ws.sumsz, ws.sumsz2
    @inbounds for i in 1:nsplits
        counter[i] = 0
        wsums[i] = 0.0
        sums[i] = 0.0
        nsmall[i] = 0
        sumsz[i] = 0.0
        sumsz2[i] = 0.0
    end
    n_missing = 0
    w_missing = s_missing = z_missing = z2_missing = 0.0
    small_missing = 0
    idx = 1
    wt, resp, sv, ss, zz = data.wt, ws.resp, ws.sv, ws.ss, data.z
    @inbounds for k in 1:(m - 1)
        s = ss[k]
        v = sv[k]
        z = zz[s]
        sw = wt[s]
        if isnan(v)
            w_missing += sw
            s_missing += sw * resp[s]
            n_missing += 1
            z_missing += sw * z
            z2_missing += sw * z * z
            z < mean_z && (small_missing += 1)
        else
            wsums[idx] += sw
            sums[idx] += sw * resp[s]
            counter[idx] += 1
            sumsz[idx] += sw * z
            sumsz2[idx] += sw * z * z
            z < mean_z && (nsmall[idx] += 1)
        end
        nv = sv[k + 1]
        if v != nv && !isnan(nv)
            idx += 1
        end
    end
    for send_left in (true, false)
        if send_left
            n_left, w_left, s_left = n_missing, w_missing, s_missing
            z_left, z2_left, small_left = z_missing, z2_missing, small_missing
        else
            n_missing == 0 && break
            n_left, w_left, s_left = 0, 0.0, 0.0
            z_left, z2_left, small_left = 0.0, 0.0, 0
        end
        @inbounds for i in 1:nsplits
            (i == 1 && !send_left) && continue
            n_left += counter[i]
            small_left += nsmall[i]
            w_left += wsums[i]
            s_left += sums[i]
            z_left += sumsz[i]
            z2_left += sumsz2[i]
            large_left = n_left - small_left
            (small_left < min_node_size || large_left < min_node_size) && continue
            n_right = m - n_left
            small_right = n_small_node - small_left
            large_right = n_right - small_right
            (small_right < min_node_size || large_right < min_node_size) && break
            size_left = z2_left - z_left * z_left / w_left
            (size_left < min_child_size || (penalty > 0.0 && size_left == 0)) && continue
            w_right = wsum_node - w_left
            s_right = sum_node - s_left
            z2_right = sum_z2_node - z2_left
            z_right = sum_z_node - z_left
            size_right = z2_right - z_right * z_right / w_right
            (size_right < min_child_size || (penalty > 0.0 && size_right == 0)) && continue
            decrease = s_left * s_left / w_left + s_right * s_right / w_right
            decrease -= penalty * (1.0 / size_left + 1.0 / size_right)
            if decrease > best.decrease
                best.decrease = decrease
                best.var = var
                best.value = ws.uniq[i]
                best.send_left = send_left
            end
        end
    end
    return nothing
end

# InstrumentalRelabelingStrategy::relabel. Returns true when the node cannot be split.
function _grf_relabel_instrumental!(ws::_GRFWorkspace, data::_GRFData, S::Vector{Int},
                                    rfw::Float64)
    sw = 0.0
    ty = tw = tz = 0.0
    @inbounds for s in S
        g = data.wt[s]
        ty += g * data.y[s]
        tw += g * data.w[s]
        tz += g * data.z[s]
        sw += g
    end
    abs(sw) <= 1e-16 && return true
    ay, aw, az = ty / sw, tw / sw, tz / sw
    arz = (1 - rfw) * az + rfw * aw
    num = 0.0
    den = 0.0
    @inbounds for s in S
        g = data.wt[s]
        rz = (1 - rfw) * data.z[s] + rfw * data.w[s]
        num += g * (rz - arz) * (data.y[s] - ay)
        den += g * (rz - arz) * (data.w[s] - aw)
    end
    abs(den) < 1.0e-10 && return true
    τ = num / den
    @inbounds for s in S
        rz = (1 - rfw) * data.z[s] + rfw * data.w[s]
        res = (data.y[s] - ay) - τ * (data.w[s] - aw)
        ws.resp[s] = (rz - arz) * res
    end
    return false
end

# Split variable candidates: mtry drawn from Poisson(mtry), clipped to [1, p], then
# that many distinct variables.
function _grf_split_candidates(rng::AbstractRNG, p::Int, mtry::Int)
    k = mtry > 0 ? rand(rng, Poisson(mtry)) : 0
    k = max(min(k, p), 1)
    return randperm(rng, p)[1:k]
end

# Try to split node samples S; returns (var, value, send_left) or nothing (leaf).
function _grf_split_node(rng, ws::_GRFWorkspace, data::_GRFData, opts::_GRFOptions,
                         S::Vector{Int})
    vars = _grf_split_candidates(rng, size(data.X, 2), opts.mtry)
    length(S) <= opts.min_node_size && return nothing
    best = _GRFBest(0.0, 0, 0.0, true)
    if opts.kind === :regression
        @inbounds for s in S
            ws.resp[s] = data.y[s]
        end
    else
        _grf_relabel_instrumental!(ws, data, S, opts.reduced_form_weight) && return nothing
    end
    wsum = 0.0
    ssum = 0.0
    @inbounds for s in S
        g = data.wt[s]
        wsum += g
        ssum += g * ws.resp[s]
    end
    if opts.kind === :regression || !opts.stabilize_splits
        min_child = max(ceil(Int, length(S) * opts.alpha), 1)
        for v in vars
            _grf_split_value_regression!(best, ws, data, S, v, wsum, ssum, min_child,
                                         opts.imbalance_penalty)
        end
    else
        sz = sz2 = 0.0
        @inbounds for s in S
            g = data.wt[s]
            z = data.z[s]
            sz += g * z
            sz2 += g * z * z
        end
        size_node = sz2 - sz * sz / wsum
        min_child_size = size_node * opts.alpha
        mean_z = sz / wsum
        nsmall = 0
        @inbounds for s in S
            data.z[s] < mean_z && (nsmall += 1)
        end
        for v in vars
            _grf_split_value_instrumental!(best, ws, data, S, v, wsum, ssum, mean_z, nsmall,
                                           sz, sz2, min_child_size, opts.min_node_size,
                                           opts.imbalance_penalty)
        end
    end
    best.decrease <= 0.0 && return nothing
    return (best.var, best.value, best.send_left)
end

# TreeTrainer::train on the clusters `cl` (indices into the cluster list or samples).
function _grf_train_tree(rng::AbstractRNG, ws::_GRFWorkspace, data::_GRFData,
                         opts::_GRFOptions, cl::Vector{Int})
    if opts.honesty
        grow_cl, leaf_cl = _grf_subsample(rng, cl, opts.honesty_fraction)
        root = _grf_sample_from_clusters(rng, opts, grow_cl)
        new_leaf = _grf_sample_from_clusters(rng, opts, leaf_cl)
    else
        root = _grf_sample_from_clusters(rng, opts, cl)
        new_leaf = Int[]
    end
    samples = [root]
    left = Int32[0]
    right = Int32[0]
    svar = Int32[0]
    sval = [0.0]
    nal = BitVector([true])
    i = 1
    while i <= length(samples)
        S = samples[i]
        res = _grf_split_node(rng, ws, data, opts, S)
        if res === nothing
            sval[i] = -1.0
        else
            v, val, sl = res
            svar[i] = v
            sval[i] = val
            nal[i] = sl
            l = length(samples) + 1
            left[i] = l
            right[i] = l + 1
            Sl = Int[]
            Sr = Int[]
            X = data.X
            @inbounds for s in S
                x = X[s, v]
                if x <= val || (sl && isnan(x)) || (isnan(val) && isnan(x))
                    push!(Sl, s)
                else
                    push!(Sr, s)
                end
            end
            push!(samples, Sl, Sr)
            append!(left, (0, 0))
            append!(right, (0, 0))
            append!(svar, (0, 0))
            append!(sval, (0.0, 0.0))
            append!(nal, (true, true))
            samples[i] = Int[]
        end
        i += 1
    end
    drawn = Int32.(_grf_samples_in_clusters(opts, cl))
    leaves = [Int32.(s) for s in samples]
    rootnode = 1
    t = _GRFTree(rootnode, left, right, svar, sval, nal, leaves, drawn,
                 zeros(0, 0), falses(0))
    if !isempty(new_leaf)
        newleaves = [Int32[] for _ in 1:length(samples)]
        for s in new_leaf
            push!(newleaves[_grf_find_leaf(t, data.X, s)], Int32(s))
        end
        leaves = newleaves
        t = _GRFTree(rootnode, left, right, svar, sval, nal, leaves, drawn, zeros(0, 0),
                     falses(0))
        opts.honesty_prune_leaves && (rootnode = _grf_honesty_prune!(t))
    end
    vals, ne = _grf_leaf_values(opts.kind, leaves, data)
    return _GRFTree(rootnode, left, right, svar, sval, nal, leaves, drawn, vals, ne)
end

# Tree::honesty_prune_leaves: bottom-up, a split with an empty child leaf is replaced
# by its other child. Mutates the child arrays; returns the (possibly new) root.
function _grf_honesty_prune!(t::_GRFTree)
    nn = length(t.left)
    is_empty_leaf(nd) = t.left[nd] == 0 && t.right[nd] == 0 && isempty(t.leaf_samples[nd])
    function prune(nd)
        l = Int(t.left[nd])
        r = Int(t.right[nd])
        if is_empty_leaf(l) || is_empty_leaf(r)
            t.left[nd] = 0
            t.right[nd] = 0
            if !is_empty_leaf(l)
                return l
            elseif !is_empty_leaf(r)
                return r
            end
        end
        return nd
    end
    for nd in nn:-1:1
        (t.left[nd] == 0 && t.right[nd] == 0) && continue
        l = Int(t.left[nd])
        if !(t.left[l] == 0 && t.right[l] == 0)
            t.left[nd] = prune(l)
        end
        r = Int(t.right[nd])
        if !(t.left[r] == 0 && t.right[r] == 0)
            t.right[nd] = prune(r)
        end
    end
    return (t.left[1] == 0 && t.right[1] == 0) ? 1 : prune(1)
end

# precompute_prediction_values of the regression / instrumental strategies.
function _grf_leaf_values(kind::Symbol, leaves::Vector{Vector{Int32}}, data::_GRFData)
    K = _grf_nvalues(kind)
    V = zeros(K, length(leaves))
    ne = falses(length(leaves))
    for (j, L) in enumerate(leaves)
        m = length(L)
        m == 0 && continue
        if kind === :regression
            s = 0.0
            g = 0.0
            @inbounds for i in L
                s += data.wt[i] * data.y[i]
                g += data.wt[i]
            end
            abs(g) <= 1e-16 && continue
            V[1, j] = s / m
            V[2, j] = g / m
        else
            sy = sw = sz = syz = swz = szz = sg = 0.0
            @inbounds for i in L
                g = data.wt[i]
                y, w, z = data.y[i], data.w[i], data.z[i]
                sy += g * y
                sw += g * w
                sz += g * z
                syz += g * y * z
                swz += g * w * z
                szz += g * z * z
                sg += g
            end
            abs(sg) <= 1e-16 && continue
            V[:, j] .= (sy / m, sw / m, sz / m, syz / m, swz / m, szz / m, sg / m)
        end
        ne[j] = true
    end
    return V, ne
end

# ------------------------------------------------------------------ forest training

"""Run `f(range)` over chunks of `1:N`, on threads when `parallel`."""
function _ml_grf_foreach_chunk(f, N::Int, parallel::Bool)
    N <= 0 && return nothing
    nt = Threads.nthreads()
    if !parallel || nt == 1 || N == 1
        f(1:N)
        return nothing
    end
    nchunks = min(N, 4 * nt)
    tasks = [Threads.@spawn f(((k - 1) * N ÷ nchunks + 1):(k * N ÷ nchunks))
             for k in 1:nchunks]
    for t in tasks
        try
            fetch(t)
        catch e
            e isa TaskFailedException ? throw(e.task.exception) : rethrow()
        end
    end
    return nothing
end

"""ForestTrainer::train. `seeds` holds one seed per CI group."""
function _grf_train(data::_GRFData, opts::_GRFOptions, seeds::Vector{UInt64};
                    parallel::Bool=true)
    n = size(data.X, 1)
    ci = opts.ci_group_size
    ngroups = length(seeds)
    trees = Vector{_GRFTree}(undef, ngroups * ci)
    _ml_grf_foreach_chunk(ngroups, parallel) do range
        ws = _GRFWorkspace(n)
        for g in range
            rng = Random.Xoshiro(seeds[g])
            if ci == 1
                cl = _grf_sample_clusters(rng, opts, n, opts.sample_fraction)
                trees[g] = _grf_train_tree(rng, ws, data, opts, cl)
            else
                half = _grf_sample_clusters(rng, opts, n, 0.5)
                for j in 1:ci
                    sub, _ = _grf_subsample(rng, half, 2 * opts.sample_fraction)
                    trees[(g - 1) * ci + j] = _grf_train_tree(rng, ws, data, opts, sub)
                end
            end
        end
    end
    return _GRFForest(trees, opts, size(data.X, 2))
end

# ------------------------------------------------------------------- prediction

"""
Leaf of every (tree, point): `L[t, i]` (0 when tree `t` is not used for point `i`,
i.e. the point was drawn by the tree and `oob`).
"""
function _grf_leaf_matrix(f::_GRFForest, X::AbstractMatrix, oob::Bool, parallel::Bool)
    B = length(f.trees)
    m = size(X, 1)
    L = zeros(Int32, B, m)
    _ml_grf_foreach_chunk(B, parallel) do range
        valid = trues(m)
        for b in range
            t = f.trees[b]
            fill!(valid, true)
            if oob
                for s in t.drawn
                    valid[s] = false
                end
            end
            @inbounds for i in 1:m
                valid[i] && (L[b, i] = _grf_find_leaf(t, X, i))
            end
        end
    end
    return L
end

# ObjectiveBayesDebiaser::debias
function _grf_bayes_debias(var_between::Float64, group_noise::Float64, ngroups::Float64)
    initial = var_between - group_noise
    se = max(var_between, group_noise) * sqrt(2.0 / ngroups)
    abs(se) < 1.0e-10 && return 0.0
    ratio = initial / se
    # φ(r) / Φ(r), computed on the log scale (grf: exp(-r²/2)/√(2π) / (erfc(-r/√2)/2),
    # which underflows to NaN for r below about -38)
    return initial + se * exp(logpdf(Normal(), ratio) - logcdf(Normal(), ratio))
end

_grf_point(kind::Symbol, a) = kind === :regression ? a[1] / a[2] :
    (a[4] * a[7] - a[1] * a[3]) / (a[5] * a[7] - a[2] * a[3])

"""
Point predictions (and optionally variances and debiased errors) for the points
whose leaves are in `L` (see `_grf_leaf_matrix`). `yerr`, `zerr` are the outcome
and instrument of the points, needed for `estimate_error` (out-of-bag only).
"""
function _grf_collect(f::_GRFForest, L::Matrix{Int32}; estimate_variance::Bool=false,
                      estimate_error::Bool=false, yerr=nothing, zerr=nothing,
                      parallel::Bool=true)
    kind = f.opts.kind
    K = _grf_nvalues(kind)
    B, m = size(L)
    ci = f.opts.ci_group_size
    pred = fill(NaN, m)
    vars = estimate_variance ? fill(NaN, m) : Float64[]
    errs = estimate_error ? fill(NaN, m) : Float64[]
    _ml_grf_foreach_chunk(m, parallel) do range
        avg = zeros(K)
        for i in range
            fill!(avg, 0.0)
            nl = 0
            @inbounds for b in 1:B
                nd = L[b, i]
                nd == 0 && continue
                t = f.trees[b]
                t.nonempty[nd] || continue
                nl += 1
                for k in 1:K
                    avg[k] += t.values[k, nd]
                end
            end
            nl == 0 && continue
            avg ./= nl
            pred[i] = _grf_point(kind, avg)
            if estimate_variance
                vars[i] = _grf_variance(f, L, i, avg, ci)
            end
            if estimate_error
                errs[i] = _grf_error(f, L, i, avg, yerr[i],
                                     zerr === nothing ? 0.0 : zerr[i])
            end
        end
    end
    return pred, vars, errs
end

# compute_variance of the regression / instrumental strategies (little bags).
function _grf_variance(f::_GRFForest, L, i, avg, ci)
    kind = f.opts.kind
    ci < 2 && return NaN
    B = size(L, 1)
    if kind === :regression
        aw = avg[2]
        ay = avg[1] / aw
    else
        ie = avg[4] * avg[7] - avg[1] * avg[3]
        fs = avg[5] * avg[7] - avg[2] * avg[3]
        τ = ie / fs
        μ = (avg[1] - avg[2] * τ) / avg[7]
    end
    ngood = 0.0
    rho2 = 0.0
    rhog2 = 0.0
    @inbounds for g in 1:(B ÷ ci)
        good = true
        for j in 1:ci
            b = (g - 1) * ci + j
            nd = L[b, i]
            if nd == 0 || !f.trees[b].nonempty[nd]
                good = false
                break
            end
        end
        good || continue
        ngood += 1
        grho = 0.0
        for j in 1:ci
            b = (g - 1) * ci + j
            v = view(f.trees[b].values, :, Int(L[b, i]))
            rho = if kind === :regression
                (v[1] - ay * v[2]) / aw
            else
                psi1 = v[4] - v[5] * τ - v[3] * μ
                psi2 = v[1] - v[2] * τ - v[7] * μ
                (avg[7] * psi1 - avg[3] * psi2) / fs
            end
            rho2 += rho * rho
            grho += rho
        end
        grho /= ci
        rhog2 += grho * grho
    end
    var_between = rhog2 / ngood
    var_total = rho2 / (ngood * ci)
    group_noise = (var_total - var_between) / (ci - 1)
    return _grf_bayes_debias(var_between, group_noise, ngood)
end

# compute_error of the regression / instrumental strategies (debiased OOB error).
function _grf_error(f::_GRFForest, L, i, avg, y, z)
    B = size(L, 1)
    if f.opts.kind === :regression
        aw = avg[2]
        ay = avg[1] / aw
        mse = (ay - y)^2
        bias = 0.0
        nt = 0
        @inbounds for b in 1:B
            nd = L[b, i]
            (nd == 0 || !f.trees[b].nonempty[nd]) && continue
            v = f.trees[b].values
            tv = (v[1, nd] - ay * v[2, nd]) / aw
            bias += tv * tv
            nt += 1
        end
        nt <= 1 && return NaN
        return mse - bias / (nt * (nt - 1))
    end
    rfn = avg[4] * avg[7] - avg[1] * avg[3]
    rfd = avg[6] * avg[7] - avg[3] * avg[3]
    rf = rfn / rfd
    res = y - (z - avg[3] / avg[7]) * rf - avg[1] / avg[7]
    err = res * res
    nt = 0
    @inbounds for b in 1:B
        nd = L[b, i]
        (nd == 0 || !f.trees[b].nonempty[nd]) && continue
        nt += 1
    end
    nt <= 5 && return NaN
    bias = 0.0
    @inbounds for b in 1:B
        nd = L[b, i]
        (nd == 0 || !f.trees[b].nonempty[nd]) && continue
        v = f.trees[b].values
        wl = (nt * avg[7] - v[7, nd]) / (nt - 1)
        yl = (nt * avg[1] - v[1, nd]) / (nt - 1)
        zl = (nt * avg[3] - v[3, nd]) / (nt - 1)
        yzl = (nt * avg[4] - v[4, nd]) / (nt - 1)
        zzl = (nt * avg[6] - v[6, nd]) / (nt - 1)
        rfl = (yzl * wl - yl * zl) / (zzl * wl - zl * zl)
        rl = y - (z - zl / wl) * rfl - yl / wl
        bias += (rl - res)^2
    end
    bias *= (nt - 1) / nt
    return err - bias
end

"""
Forest weights `α_i(x)` (training points × prediction points), as grf's
`get_forest_weights`: for each tree whose leaf containing `x` is non-empty, each
sample of that leaf gets `1 / |leaf|`; the result is averaged over those trees.
"""
function _grf_forest_weights(f::_GRFForest, L::Matrix{Int32}, n::Int)
    B, m = size(L)
    A = zeros(n, m)
    for i in 1:m
        nl = 0
        for b in 1:B
            nd = L[b, i]
            nd == 0 && continue
            t = f.trees[b]
            t.nonempty[nd] || continue
            nl += 1
            leaf = t.leaf_samples[nd]
            for s in leaf
                A[s, i] += 1 / length(leaf)
            end
        end
        nl > 0 && (A[:, i] ./= nl)
    end
    return A
end

"""SplitFrequencyComputer::compute: splits on each variable by depth (depth × p)."""
function _grf_split_frequencies(f::_GRFForest, max_depth::Int)
    R = zeros(Int, max_depth, f.num_variables)
    for t in f.trees
        level = [t.root]
        depth = 0
        while !isempty(level) && depth < max_depth
            next = Int[]
            for nd in level
                _grf_is_leaf(t, nd) && continue
                R[depth + 1, t.split_var[nd]] += 1
                push!(next, t.left[nd], t.right[nd])
            end
            level = next
            depth += 1
        end
    end
    return R
end
