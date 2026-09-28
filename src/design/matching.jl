# Non-bipartite matching for matched-pair and block designs.
#
# - `_des_max_weight_matching`: Edmonds' weighted blossom algorithm (O(n³)), a port of
#   J. van Rantwijk's reference implementation (mwmatching.py, public domain), using
#   integer weights so every dual update is exact. With `maxcardinality = true` it
#   returns a maximum-cardinality matching of maximum total weight; with weights
#   `W - d_ij` this is a minimum-distance perfect matching (Derigs 1988; Lu, Greevy,
#   Xu & Beck 2011 use the same construction for non-bipartite matching designs).
# - `_des_pairs_1d`: exact minimum-distance pairing on a scalar score (adjacent units
#   in sorted order are optimal; one unit is left over when n is odd).
# - `_des_greedy_pairs`, `_des_greedy_blocks`: greedy matching / blocking.
#
# Internally the blossom code uses 0-based vertex, edge and endpoint indices as in the
# reference implementation; arrays are accessed with `+ 1`.

"""
Maximum-weight matching on the general graph with edges `(i, j, w)` (0-based
vertices, integer weights). Returns `mate` (0-based partner or -1 per vertex).
"""
function _des_max_weight_matching(edges::Vector{Tuple{Int,Int,Int}}, nvertex::Int;
                                  maxcardinality::Bool=true)
    nedge = length(edges)
    mate = fill(-1, nvertex)
    nedge == 0 && return mate
    maxweight = max(0, maximum(e[3] for e in edges))
    endpoint = [isodd(p) ? edges[p ÷ 2 + 1][2] : edges[p ÷ 2 + 1][1]
                for p in 0:(2 * nedge - 1)]
    neighbend = [Int[] for _ in 1:nvertex]
    for (k0, (i, j, _)) in enumerate(edges)
        k = k0 - 1
        push!(neighbend[i + 1], 2k + 1)
        push!(neighbend[j + 1], 2k)
    end
    nv2 = 2 * nvertex
    label = zeros(Int, nv2)
    labelend = fill(-1, nv2)
    inblossom = collect(0:(nvertex - 1))
    blossomparent = fill(-1, nv2)
    blossomchilds = Vector{Union{Nothing,Vector{Int}}}(nothing, nv2)
    blossombase = vcat(collect(0:(nvertex - 1)), fill(-1, nvertex))
    blossomendps = Vector{Union{Nothing,Vector{Int}}}(nothing, nv2)
    bestedge = fill(-1, nv2)
    blossombestedges = Vector{Union{Nothing,Vector{Int}}}(nothing, nv2)
    unusedblossoms = collect(nvertex:(nv2 - 1))
    dualvar = vcat(fill(maxweight, nvertex), zeros(Int, nvertex))
    allowedge = falses(nedge)
    queue = Int[]

    slack(k) = (e = edges[k + 1]; dualvar[e[1] + 1] + dualvar[e[2] + 1] - 2 * e[3])

    function leaves!(out::Vector{Int}, b::Int)
        if b < nvertex
            push!(out, b)
        else
            for t in blossomchilds[b + 1]
                t < nvertex ? push!(out, t) : leaves!(out, t)
            end
        end
        return out
    end
    leaves(b) = leaves!(Int[], b)

    function assign_label(w::Int, t::Int, p::Int)
        b = inblossom[w + 1]
        label[w + 1] = t
        label[b + 1] = t
        labelend[w + 1] = p
        labelend[b + 1] = p
        bestedge[w + 1] = -1
        bestedge[b + 1] = -1
        if t == 1
            append!(queue, leaves(b))
        elseif t == 2
            base = blossombase[b + 1]
            mb = mate[base + 1]
            assign_label(endpoint[mb + 1], 1, xor(mb, 1))
        end
        return nothing
    end

    function scan_blossom(v::Int, w::Int)
        path = Int[]
        base = -1
        while v != -1 || w != -1
            b = inblossom[v + 1]
            if (label[b + 1] & 4) != 0
                base = blossombase[b + 1]
                break
            end
            push!(path, b)
            label[b + 1] = 5
            if labelend[b + 1] == -1
                v = -1
            else
                v = endpoint[labelend[b + 1] + 1]
                b = inblossom[v + 1]
                v = endpoint[labelend[b + 1] + 1]
            end
            if w != -1
                v, w = w, v
            end
        end
        for b in path
            label[b + 1] = 1
        end
        return base
    end

    function add_blossom(base::Int, k::Int)
        v, w, _ = edges[k + 1]
        bb = inblossom[base + 1]
        bv = inblossom[v + 1]
        bw = inblossom[w + 1]
        b = pop!(unusedblossoms)
        blossombase[b + 1] = base
        blossomparent[b + 1] = -1
        blossomparent[bb + 1] = b
        path = Int[]
        endps = Int[]
        while bv != bb
            blossomparent[bv + 1] = b
            push!(path, bv)
            push!(endps, labelend[bv + 1])
            v = endpoint[labelend[bv + 1] + 1]
            bv = inblossom[v + 1]
        end
        push!(path, bb)
        reverse!(path)
        reverse!(endps)
        push!(endps, 2k)
        while bw != bb
            blossomparent[bw + 1] = b
            push!(path, bw)
            push!(endps, xor(labelend[bw + 1], 1))
            w = endpoint[labelend[bw + 1] + 1]
            bw = inblossom[w + 1]
        end
        blossomchilds[b + 1] = path
        blossomendps[b + 1] = endps
        label[b + 1] = 1
        labelend[b + 1] = labelend[bb + 1]
        dualvar[b + 1] = 0
        for u in leaves(b)
            label[inblossom[u + 1] + 1] == 2 && push!(queue, u)
            inblossom[u + 1] = b
        end
        bestedgeto = fill(-1, nv2)
        for bv2 in path
            nblists = if blossombestedges[bv2 + 1] === nothing
                [[p ÷ 2 for p in neighbend[u + 1]] for u in leaves(bv2)]
            else
                [blossombestedges[bv2 + 1]]
            end
            for nblist in nblists, kk in nblist
                i, j, _ = edges[kk + 1]
                if inblossom[j + 1] == b
                    i, j = j, i
                end
                bj = inblossom[j + 1]
                if bj != b && label[bj + 1] == 1 &&
                   (bestedgeto[bj + 1] == -1 || slack(kk) < slack(bestedgeto[bj + 1]))
                    bestedgeto[bj + 1] = kk
                end
            end
            blossombestedges[bv2 + 1] = nothing
            bestedge[bv2 + 1] = -1
        end
        blossombestedges[b + 1] = [kk for kk in bestedgeto if kk != -1]
        bestedge[b + 1] = -1
        for kk in blossombestedges[b + 1]
            if bestedge[b + 1] == -1 || slack(kk) < slack(bestedge[b + 1])
                bestedge[b + 1] = kk
            end
        end
        return nothing
    end

    # Python-style (possibly negative) index into a child/endpoint list.
    at(v::Vector{Int}, j::Int) = v[mod(j, length(v)) + 1]

    function expand_blossom(b::Int, endstage::Bool)
        for s in blossomchilds[b + 1]
            blossomparent[s + 1] = -1
            if s < nvertex
                inblossom[s + 1] = s
            elseif endstage && dualvar[s + 1] == 0
                expand_blossom(s, endstage)
            else
                for u in leaves(s)
                    inblossom[u + 1] = s
                end
            end
        end
        if !endstage && label[b + 1] == 2
            childs = blossomchilds[b + 1]
            endps = blossomendps[b + 1]
            entrychild = inblossom[endpoint[xor(labelend[b + 1], 1) + 1] + 1]
            j = findfirst(==(entrychild), childs) - 1
            if isodd(j)
                j -= length(childs)
                jstep = 1
                endptrick = 0
            else
                jstep = -1
                endptrick = 1
            end
            p = labelend[b + 1]
            while j != 0
                label[endpoint[xor(p, 1) + 1] + 1] = 0
                q = xor(xor(at(endps, j - endptrick), endptrick), 1)
                label[endpoint[q + 1] + 1] = 0
                assign_label(endpoint[xor(p, 1) + 1], 2, p)
                allowedge[at(endps, j - endptrick) ÷ 2 + 1] = true
                j += jstep
                p = xor(at(endps, j - endptrick), endptrick)
                allowedge[p ÷ 2 + 1] = true
                j += jstep
            end
            bv = at(childs, j)
            label[endpoint[xor(p, 1) + 1] + 1] = 2
            label[bv + 1] = 2
            labelend[endpoint[xor(p, 1) + 1] + 1] = p
            labelend[bv + 1] = p
            bestedge[bv + 1] = -1
            j += jstep
            while at(childs, j) != entrychild
                bv = at(childs, j)
                if label[bv + 1] == 1
                    j += jstep
                    continue
                end
                found = -1
                for u in leaves(bv)
                    if label[u + 1] != 0
                        found = u
                        break
                    end
                end
                if found != -1
                    u = found
                    label[u + 1] = 0
                    label[endpoint[mate[blossombase[bv + 1] + 1] + 1] + 1] = 0
                    assign_label(u, 2, labelend[u + 1])
                end
                j += jstep
            end
        end
        label[b + 1] = -1
        labelend[b + 1] = -1
        blossomchilds[b + 1] = nothing
        blossomendps[b + 1] = nothing
        blossombase[b + 1] = -1
        blossombestedges[b + 1] = nothing
        bestedge[b + 1] = -1
        push!(unusedblossoms, b)
        return nothing
    end

    function augment_blossom(b::Int, v::Int)
        t = v
        while blossomparent[t + 1] != b
            t = blossomparent[t + 1]
        end
        t >= nvertex && augment_blossom(t, v)
        childs = blossomchilds[b + 1]
        endps = blossomendps[b + 1]
        i = findfirst(==(t), childs) - 1
        j = i
        if isodd(i)
            j -= length(childs)
            jstep = 1
            endptrick = 0
        else
            jstep = -1
            endptrick = 1
        end
        while j != 0
            j += jstep
            t = at(childs, j)
            p = xor(at(endps, j - endptrick), endptrick)
            t >= nvertex && augment_blossom(t, endpoint[p + 1])
            j += jstep
            t = at(childs, j)
            t >= nvertex && augment_blossom(t, endpoint[xor(p, 1) + 1])
            mate[endpoint[p + 1] + 1] = xor(p, 1)
            mate[endpoint[xor(p, 1) + 1] + 1] = p
        end
        blossomchilds[b + 1] = vcat(childs[(i + 1):end], childs[1:i])
        blossomendps[b + 1] = vcat(endps[(i + 1):end], endps[1:i])
        blossombase[b + 1] = blossombase[blossomchilds[b + 1][1] + 1]
        return nothing
    end

    function augment_matching(k::Int)
        v, w, _ = edges[k + 1]
        for (s0, p0) in ((v, 2k + 1), (w, 2k))
            s, p = s0, p0
            while true
                bs = inblossom[s + 1]
                bs >= nvertex && augment_blossom(bs, s)
                mate[s + 1] = p
                labelend[bs + 1] == -1 && break
                t = endpoint[labelend[bs + 1] + 1]
                bt = inblossom[t + 1]
                s = endpoint[labelend[bt + 1] + 1]
                j = endpoint[xor(labelend[bt + 1], 1) + 1]
                bt >= nvertex && augment_blossom(bt, j)
                mate[j + 1] = labelend[bt + 1]
                p = xor(labelend[bt + 1], 1)
            end
        end
        return nothing
    end

    for _ in 1:nvertex
        fill!(label, 0)
        fill!(bestedge, -1)
        for b in (nvertex + 1):nv2
            blossombestedges[b] = nothing
        end
        fill!(allowedge, false)
        empty!(queue)
        for v in 0:(nvertex - 1)
            if mate[v + 1] == -1 && label[inblossom[v + 1] + 1] == 0
                assign_label(v, 1, -1)
            end
        end
        augmented = false
        while true
            while !isempty(queue) && !augmented
                v = pop!(queue)
                for p in neighbend[v + 1]
                    k = p ÷ 2
                    w = endpoint[p + 1]
                    inblossom[v + 1] == inblossom[w + 1] && continue
                    kslack = 0
                    if !allowedge[k + 1]
                        kslack = slack(k)
                        kslack <= 0 && (allowedge[k + 1] = true)
                    end
                    if allowedge[k + 1]
                        if label[inblossom[w + 1] + 1] == 0
                            assign_label(w, 2, xor(p, 1))
                        elseif label[inblossom[w + 1] + 1] == 1
                            base = scan_blossom(v, w)
                            if base >= 0
                                add_blossom(base, k)
                            else
                                augment_matching(k)
                                augmented = true
                                break
                            end
                        elseif label[w + 1] == 0
                            label[w + 1] = 2
                            labelend[w + 1] = xor(p, 1)
                        end
                    elseif label[inblossom[w + 1] + 1] == 1
                        b = inblossom[v + 1]
                        if bestedge[b + 1] == -1 || kslack < slack(bestedge[b + 1])
                            bestedge[b + 1] = k
                        end
                    elseif label[w + 1] == 0
                        if bestedge[w + 1] == -1 || kslack < slack(bestedge[w + 1])
                            bestedge[w + 1] = k
                        end
                    end
                end
            end
            augmented && break
            deltatype = -1
            delta = 0
            deltaedge = -1
            deltablossom = -1
            if !maxcardinality
                deltatype = 1
                delta = minimum(@view dualvar[1:nvertex])
            end
            for v in 0:(nvertex - 1)
                if label[inblossom[v + 1] + 1] == 0 && bestedge[v + 1] != -1
                    d = slack(bestedge[v + 1])
                    if deltatype == -1 || d < delta
                        delta = d
                        deltatype = 2
                        deltaedge = bestedge[v + 1]
                    end
                end
            end
            for b in 0:(nv2 - 1)
                if blossomparent[b + 1] == -1 && label[b + 1] == 1 && bestedge[b + 1] != -1
                    d = slack(bestedge[b + 1]) ÷ 2
                    if deltatype == -1 || d < delta
                        delta = d
                        deltatype = 3
                        deltaedge = bestedge[b + 1]
                    end
                end
            end
            for b in nvertex:(nv2 - 1)
                if blossombase[b + 1] >= 0 && blossomparent[b + 1] == -1 &&
                   label[b + 1] == 2 && (deltatype == -1 || dualvar[b + 1] < delta)
                    delta = dualvar[b + 1]
                    deltatype = 4
                    deltablossom = b
                end
            end
            if deltatype == -1
                deltatype = 1
                delta = max(0, minimum(@view dualvar[1:nvertex]))
            end
            for v in 0:(nvertex - 1)
                lb = label[inblossom[v + 1] + 1]
                if lb == 1
                    dualvar[v + 1] -= delta
                elseif lb == 2
                    dualvar[v + 1] += delta
                end
            end
            for b in nvertex:(nv2 - 1)
                if blossombase[b + 1] >= 0 && blossomparent[b + 1] == -1
                    if label[b + 1] == 1
                        dualvar[b + 1] += delta
                    elseif label[b + 1] == 2
                        dualvar[b + 1] -= delta
                    end
                end
            end
            if deltatype == 1
                break
            elseif deltatype == 2
                allowedge[deltaedge + 1] = true
                i, j, _ = edges[deltaedge + 1]
                label[inblossom[i + 1] + 1] == 0 && ((i, j) = (j, i))
                push!(queue, i)
            elseif deltatype == 3
                allowedge[deltaedge + 1] = true
                i, _, _ = edges[deltaedge + 1]
                push!(queue, i)
            else
                expand_blossom(deltablossom, false)
            end
        end
        augmented || break
        for b in nvertex:(nv2 - 1)
            if blossomparent[b + 1] == -1 && blossombase[b + 1] >= 0 &&
               label[b + 1] == 1 && dualvar[b + 1] == 0
                expand_blossom(b, true)
            end
        end
    end
    for v in 1:nvertex
        mate[v] >= 0 && (mate[v] = endpoint[mate[v] + 1])
    end
    return mate
end

# Integer weights for the blossom algorithm: distances scaled to at most 2^40 so that
# dual variables (sums of a few weights) never overflow Int64.
function _des_int_weights(D::AbstractMatrix)
    dmax = maximum(D)
    dmax > 0 || return zeros(Int, size(D)), 1.0
    scale = 2.0^40 / dmax
    return round.(Int, D .* scale), scale
end

"""
Minimum total distance matching of maximum cardinality on the complete graph with
distance matrix `D` (symmetric, n × n). Returns a vector of pairs `(i, j)` (1-based,
`i < j`) and the unmatched unit (0 when n is even).
"""
function _des_optimal_pairs(D::AbstractMatrix)
    n = size(D, 1)
    n >= 2 || throw(ArgumentError("need at least two units to form pairs"))
    Wd, _ = _des_int_weights(D)
    big = maximum(Wd) + 1
    edges = Tuple{Int,Int,Int}[]
    sizehint!(edges, n * (n - 1) ÷ 2)
    for i in 1:n, j in (i + 1):n
        push!(edges, (i - 1, j - 1, big - Wd[i, j]))
    end
    mate = _des_max_weight_matching(edges, n; maxcardinality=true)
    pairs = Tuple{Int,Int}[]
    left = 0
    for i in 1:n
        m = mate[i]
        if m < 0
            left = i
        elseif i < m + 1
            push!(pairs, (i, m + 1))
        end
    end
    length(pairs) == n ÷ 2 ||
        error("internal error: matching is not of maximum cardinality")
    return pairs, left
end

"""
Exact minimum-distance pairing on a scalar score: adjacent units in sorted order when
n is even; for odd n a dynamic program chooses the unit left over. `order` is the
sorted order of the units (ties already broken deterministically).
"""
function _des_pairs_1d(s::AbstractVector{<:Real}, order::AbstractVector{Int})
    n = length(order)
    x = s[order]
    if iseven(n)
        return [(order[2k - 1], order[2k]) for k in 1:(n ÷ 2)], 0
    end
    # f0[i]: best cost pairing x[1:i] (i even) without a skip;
    # f1[i]: best cost for x[1:i] (i odd) with exactly one unit skipped.
    f0 = fill(Inf, n + 1)
    f1 = fill(Inf, n + 1)
    choice = zeros(Int, n + 1)       # for f1: 1 = skip unit i, 2 = pair (i-1, i)
    f0[1] = 0.0
    for i in 1:n
        if iseven(i)
            f0[i + 1] = f0[i - 1] + (x[i] - x[i - 1])
        else
            a = f0[i]                                   # skip unit i
            b = i >= 3 ? f1[i - 1] + (x[i] - x[i - 1]) : Inf
            if a <= b
                f1[i + 1] = a
                choice[i + 1] = 1
            else
                f1[i + 1] = b
                choice[i + 1] = 2
            end
        end
    end
    pairs = Tuple{Int,Int}[]
    left = 0
    i = n
    while i >= 1
        if left == 0 && isodd(i)
            if choice[i + 1] == 1
                left = order[i]
                i -= 1
            else
                push!(pairs, (order[i - 1], order[i]))
                i -= 2
            end
        else
            push!(pairs, (order[i - 1], order[i]))
            i -= 2
        end
    end
    return reverse!(pairs), left
end

"""
Greedy pairing: repeatedly pair the two closest unmatched units (ties broken by unit
index). Returns pairs and the unit left over (0 when n is even).
"""
function _des_greedy_pairs(D::AbstractMatrix)
    n = size(D, 1)
    cand = [(D[i, j], i, j) for i in 1:n for j in (i + 1):n]
    sort!(cand)
    used = falses(n)
    pairs = Tuple{Int,Int}[]
    for (_, i, j) in cand
        (used[i] || used[j]) && continue
        push!(pairs, (i, j))
        used[i] = used[j] = true
        length(pairs) == n ÷ 2 && break
    end
    left = isodd(n) ? findfirst(!, used) : 0
    return pairs, left
end

"""
Greedy blocking into `nb = fld(n, k)` blocks: each block starts from the closest
remaining pair and grows by adding the unit with the smallest total distance to its
members; the `n mod k` remaining units are then added, one per block, to the block with
the smallest mean distance. Returns block labels 1..nb per unit.
"""
function _des_greedy_blocks(D::AbstractMatrix, k::Int)
    n = size(D, 1)
    nb = fld(n, k)
    nb >= 1 || throw(ArgumentError("block_size $k exceeds the number of units $n"))
    blk = zeros(Int, n)
    remaining = collect(1:n)
    for b in 1:nb
        if length(remaining) < k
            break
        end
        best = (Inf, 0, 0)
        for a in eachindex(remaining), c in (a + 1):length(remaining)
            d = D[remaining[a], remaining[c]]
            if d < best[1]
                best = (d, remaining[a], remaining[c])
            end
        end
        members = [best[2], best[3]]
        while length(members) < k
            bu, bd = 0, Inf
            for u in remaining
                u in members && continue
                d = sum(D[u, m] for m in members)
                if d < bd
                    bu, bd = u, d
                end
            end
            push!(members, bu)
        end
        for u in members
            blk[u] = b
        end
        filter!(u -> blk[u] == 0, remaining)
    end
    extra = zeros(Int, nb)
    for u in remaining
        bestb, bestd = 0, Inf
        for b in 1:nb
            extra[b] >= 1 && any(==(0), extra) && continue
            mem = findall(==(b), blk)
            d = mean(D[u, m] for m in mem)
            if d < bestd
                bestb, bestd = b, d
            end
        end
        blk[u] = bestb
        extra[bestb] += 1
    end
    return blk
end
