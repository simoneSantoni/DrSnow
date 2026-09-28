# Independent reference implementations for the RI tests: brute-force enumeration
# over bit masks and textbook statistics, sharing no code with src/ri.

# All 0/1 vectors over `units` (vector of unit-index groups that move together)
# accepted by `ok(zunits)`; returns unit-level Bool vectors.
function bf_assignments(n, groups, ok)
    G = length(groups)
    out = Vector{Vector{Bool}}()
    for mask in 0:(2^G - 1)
        zg = [isodd(mask >> (g - 1)) for g in 1:G]
        ok(zg) || continue
        z = falses(n)
        for g in 1:G
            zg[g] && (z[groups[g]] .= true)
        end
        push!(out, collect(z))
    end
    return out
end

bf_complete(n, k) = bf_assignments(n, [[i] for i in 1:n], zg -> count(zg) == k)

function bf_stratified(strata::Vector, z::Vector{Bool})
    labs = unique(strata)
    counts = Dict(l => count(z[strata .== l]) for l in labs)
    n = length(z)
    return bf_assignments(n, [[i] for i in 1:n],
                          zg -> all(count(zg[strata .== l]) == counts[l] for l in labs))
end

function bf_cluster(clusters::Vector, k)
    labs = unique(clusters)
    return bf_assignments(length(clusters), [findall(==(l), clusters) for l in labs],
                          zg -> count(zg) == k)
end

function bf_block_cluster(clusters, blocks, z)
    labs = unique(clusters)
    groups = [findall(==(l), clusters) for l in labs]
    gb = [blocks[g[1]] for g in groups]
    counts = Dict(b => count(z[groups[i][1]] for i in eachindex(groups) if gb[i] == b)
                  for b in unique(gb))
    return bf_assignments(length(clusters), groups,
                          zg -> all(count(zg[gb .== b]) == counts[b] for b in keys(counts)))
end

# p-value over an enumerated support with probabilities `w`
function bf_pvalue(stat, y, zobs, zs, w=fill(1 / length(zs), length(zs));
                   alternative=:two_sided)
    t0 = stat(y, zobs)
    num = 0.0; den = 0.0
    for (z, wi) in zip(zs, w)
        t = stat(y, z)
        isnan(t) && continue
        den += wi
        ext = alternative === :two_sided ? abs(t) >= abs(t0) - 1e-9 * max(1, abs(t0)) :
              alternative === :greater ? t >= t0 - 1e-9 * max(1, abs(t0)) :
              t <= t0 + 1e-9 * max(1, abs(t0))
        ext && (num += wi)
    end
    return num / den
end

bf_dim(y, z) = (any(z) && !all(z)) ? mean(y[z]) - mean(y[.!z]) : NaN

function bf_strat_dim(strata)
    labs = unique(strata)
    return (y, z) -> begin
        tot = 0.0
        for l in labs
            s = strata .== l
            tot += count(s) * (mean(y[s .& z]) - mean(y[s .& .!z]))
        end
        tot / length(y)
    end
end

bf_neyman_t(y, z) = bf_dim(y, z) / sqrt(var(y[z]) / count(z) + var(y[.!z]) / count(.!z))

function bf_rank(y, z)
    r = DrSnow.StatsBase.tiedrank(y) ./ (length(y) + 1)
    return mean(r[z]) - mean(r[.!z])
end

function bf_ks(y, z)
    a = y[z]; b = y[.!z]
    return maximum(abs(mean(a .<= v) - mean(b .<= v)) for v in y)
end

# Lin (2013) estimator via GLM on a DataFrame (independent of the package code)
function bf_lin(X)
    Xc = X .- mean(X; dims=1)
    return (y, z) -> begin
        df = DataFrame(y=y, z=Float64.(z))
        rhs = Any[DrSnow.StatsModels.ConstantTerm(1), DrSnow.StatsModels.term(:z)]
        for j in axes(Xc, 2)
            df[!, Symbol("x$j")] = Xc[:, j]
            df[!, Symbol("zx$j")] = Xc[:, j] .* z
            push!(rhs, DrSnow.StatsModels.term(Symbol("x$j")))
            push!(rhs, DrSnow.StatsModels.term(Symbol("zx$j")))
        end
        f = DrSnow.StatsModels.term(:y) ~ reduce(+, rhs)
        m = DrSnow.GLM.lm(f, df)
        coef(m)[2]
    end
end
