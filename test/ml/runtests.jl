# Tests of the causal machine-learning area (src/ml).

"""Minimal CSV reader for the committed validation files (numeric or string cells)."""
function ml_read_csv(path)
    lines = readlines(path)
    header = [strip(h, '"') for h in split(lines[1], ',')]
    cols = [String[] for _ in header]
    for l in lines[2:end]
        isempty(strip(l)) && continue
        for (j, c) in enumerate(split(l, ','))
            push!(cols[j], strip(c, '"'))
        end
    end
    df = DataFrame()
    for (h, c) in zip(header, cols)
        v = tryparse.(Float64, c)
        df[!, Symbol(h)] = any(isnothing, v) ? c : Float64.(v)
    end
    return df
end

"""Binomial tolerance for Monte Carlo coverage checks with `reps` replications."""
ml_cover_tol(reps; level=0.95) = 3.5 * sqrt(level * (1 - level) / reps) + 0.01

include("test_learners.jl")
include("test_crossfit.jl")
include("test_dml.jl")
include("test_doubleml_parity.jl")
include("test_dml_did_multi.jl")
include("test_hte.jl")
include("test_policy.jl")
include("test_ppi.jl")
include("test_mlj_ext.jl")
include("test_grf.jl")
include("test_grf_parity.jl")
include("test_grf_forest_parity.jl")
include("test_grf_montecarlo.jl")
include("test_metalearners.jl")
include("test_measurement.jl")
include("test_measurement_montecarlo.jl")
