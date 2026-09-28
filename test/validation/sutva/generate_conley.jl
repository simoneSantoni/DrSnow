# Generates the data for the Conley (1999) reference comparison with fixest.
#
#   julia --project=<env with DrSnow + StableRNGs> test/validation/sutva/generate_conley.jl
#   Rscript test/validation/sutva/conley_reference.R <path-to-R-library>
#
# Output: conley_data.csv (unit,period,lat,lon,g,x1,x2,y), a panel of 150 locations x 3
# periods with spatially correlated regressors and errors. The R script writes
# conley_reference.csv with fixest coefficients and Conley covariance entries.

using DrSnow, StableRNGs, Random, LinearAlgebra

rng = StableRNG(1999)
N = 150
T = 3
lat = 40 .+ 4 .* rand(rng, N)
lon = -100 .+ 6 .* rand(rng, N)
g = rand(rng, 1:12, N)                         # region fixed effect
s = SpatialStructure(1:N; lat=lat, lon=lon)
D = pairwise_distances(s)
C = exp.(-D ./ 150)                            # spatial correlation of shocks
L = cholesky(Symmetric(C + 1e-8I)).L
open(joinpath(@__DIR__, "conley_data.csv"), "w") do io
    println(io, "unit,period,lat,lon,g,x1,x2,y")
    for t in 1:T
        x1 = L * randn(rng, N)
        x2 = randn(rng, N)
        u = L * randn(rng, N)
        y = 1.0 .+ 0.5 .* x1 .- 0.25 .* x2 .+ 0.3 .* g .+ u
        for i in 1:N
            println(io, join((i, t, repr(lat[i]), repr(lon[i]), g[i], repr(x1[i]),
                              repr(x2[i]), repr(y[i])), ","))
        end
    end
end
