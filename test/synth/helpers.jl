# Helpers for the synth test group (no CSV.jl dependency in the test environment).

const SYNTH_REF_DIR = joinpath(@__DIR__, "..", "validation", "synth")

function _split_csv_line(line::AbstractString)
    fields = String[]
    buf = IOBuffer()
    inq = false
    i = firstindex(line)
    while i <= lastindex(line)
        c = line[i]
        if inq
            if c == '"'
                nxt = nextind(line, i)
                if nxt <= lastindex(line) && line[nxt] == '"'
                    write(buf, '"')
                    i = nxt
                else
                    inq = false
                end
            else
                write(buf, c)
            end
        elseif c == '"'
            inq = true
        elseif c == ','
            push!(fields, String(take!(buf)))
        else
            write(buf, c)
        end
        i = nextind(line, i)
    end
    push!(fields, String(take!(buf)))
    return fields
end

function _parse_column(vals::Vector{String})
    nonmissing = filter(v -> v != "NA" && !isempty(v), vals)
    if !isempty(nonmissing) && all(v -> tryparse(Int, v) !== nothing, nonmissing)
        return [v == "NA" || isempty(v) ? missing : parse(Int, v) for v in vals]
    elseif !isempty(nonmissing) && all(v -> tryparse(Float64, v) !== nothing, nonmissing)
        return [v == "NA" || isempty(v) ? missing : parse(Float64, v) for v in vals]
    end
    return [v == "NA" ? missing : v for v in vals]
end

"""Read a CSV written by R's `write.csv(row.names = FALSE)`."""
function read_ref_csv(name::AbstractString)
    lines = readlines(joinpath(SYNTH_REF_DIR, name))
    header = _split_csv_line(lines[1])
    rows = [_split_csv_line(l) for l in lines[2:end] if !isempty(l)]
    df = DataFrame()
    for (j, h) in enumerate(header)
        col = _parse_column([r[j] for r in rows])
        df[!, Symbol(h)] = identity.(col)
    end
    return df
end

load_prop99() = read_ref_csv("prop99.csv")
