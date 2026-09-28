# Shared helpers for the RD tests: a minimal CSV reader (no extra test dependency) and
# access to the rdrobust / rddensity reference values in test/validation/rd.

const RD_VALIDATION_DIR = joinpath(@__DIR__, "..", "validation", "rd")

function _rd_split_csv_line(line::AbstractString)
    fields = String[]
    buf = IOBuffer()
    inq = false
    for ch in line
        if ch == '"'
            inq = !inq
        elseif ch == ',' && !inq
            push!(fields, String(take!(buf)))
        else
            write(buf, ch)
        end
    end
    push!(fields, String(take!(buf)))
    return fields
end

"""Read a simple CSV (as written by R) into a DataFrame; numeric columns become
`Union{Missing,Float64}`, others stay strings."""
function rd_read_csv(path)
    lines = readlines(path)
    header = _rd_split_csv_line(lines[1])
    rows = [_rd_split_csv_line(l) for l in lines[2:end] if !isempty(l)]
    df = DataFrame()
    for (j, name) in enumerate(header)
        raw = [r[j] for r in rows]
        parsed = [isempty(v) || v == "NA" ? missing : tryparse(Float64, v) for v in raw]
        if all(v -> v !== nothing, parsed)
            df[!, Symbol(name)] = Vector{Union{Missing,Float64}}(parsed)
        else
            df[!, Symbol(name)] = raw
        end
    end
    return df
end

const RD_SENATE = rd_read_csv(joinpath(RD_VALIDATION_DIR, "senate.csv"))
const RD_SIM = let d = rd_read_csv(joinpath(RD_VALIDATION_DIR, "simulated.csv"))
    for c in names(d)
        d[!, c] = Float64.(d[!, c])
    end
    d
end

const RD_REF = let d = rd_read_csv(joinpath(RD_VALIDATION_DIR, "reference.csv"))
    Dict{Tuple{String,String},Float64}((r.case, r.key) => coalesce(r.value, NaN)
                                      for r in eachrow(d))
end

"""Reference value(s): scalar key or vector key (`key[1]`, `key[2]`, …)."""
function rdref(case, key)
    haskey(RD_REF, (case, key)) && return RD_REF[(case, key)]
    out = Float64[]
    i = 1
    while haskey(RD_REF, (case, "$key[$i]"))
        push!(out, RD_REF[(case, "$key[$i]")])
        i += 1
    end
    isempty(out) && error("no reference value for ($case, $key)")
    return out
end
hasref(case, key) = haskey(RD_REF, (case, key)) || haskey(RD_REF, (case, "$key[1]"))

"""Relative-or-absolute closeness used for reference comparisons."""
rdclose(a, b; rtol=1e-7, atol=1e-9) = all(isapprox.(a, b; rtol=rtol, atol=atol))
