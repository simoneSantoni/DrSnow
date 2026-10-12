# Generates the data for the stratified sandwich covariance reference comparison with
# the R package `survey` (svyglm on a stratified design with and without clusters).
#
#   julia --project=<env with StableRNGs> test/validation/core/generate_stratified_vcov.jl
#   Rscript test/validation/core/stratified_vcov_reference.R <path-to-R-library>
#
# Output: stratified_vcov_data.csv (id,stratum,cluster,z,x1,x2,y): 12 strata of
# unequal size, clusters of 2-4 observations nested in strata, a treatment randomized
# within strata with stratum-specific effects, and errors correlated within clusters.
# The R script writes stratified_vcov_reference.csv.

using StableRNGs, Random

rng = StableRNG(2026)
sizes = [8, 12, 16, 20, 24, 10, 14, 18, 22, 26, 30, 12]
open(joinpath(@__DIR__, "stratified_vcov_data.csv"), "w") do io
    println(io, "id,stratum,cluster,z,x1,x2,y")
    id = 0
    cl = 0
    for (h, n_h) in enumerate(sizes)
        # clusters of size 2–4, nested in the stratum
        remaining = n_h
        cluster = Int[]
        while remaining > 0
            k = min(remaining, rand(rng, 2:4))
            remaining -= k
            cl += 1
            append!(cluster, fill(cl, k))
        end
        ucl = Dict(c => randn(rng) for c in unique(cluster))
        n1 = n_h ÷ 2
        z = shuffle(rng, vcat(ones(Int, n1), zeros(Int, n_h - n1)))
        tau = 0.4 * (h - 6.5)
        for i in 1:n_h
            id += 1
            x1 = randn(rng)
            x2 = rand(rng) < 0.5 ? 1 : 0
            y = 0.25 * h + tau * z[i] + 0.5 * x1 - 0.3 * x2 + 0.6 * ucl[cluster[i]] +
                randn(rng)
            println(io, join((id, h, cluster[i], z[i], repr(x1), x2, repr(y)), ","))
        end
    end
end
