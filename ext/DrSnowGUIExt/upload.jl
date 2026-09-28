# CSV/TSV parsing with limits, and table summaries for the frontend.

const _GUI_DELIMS = Dict("comma" => ',', "tab" => '\t', "semicolon" => ';')
const _GUI_PREVIEW_ROWS = 20
const _GUI_MAX_CELL_CHARS = 200
const _GUI_MAX_NAME_CHARS = 256

"""Bytes of the first line (header), stopping at the first newline outside quotes."""
function _gui_header_line(bytes::AbstractVector{UInt8})
    inq = false
    for (i, b) in enumerate(bytes)
        if b == UInt8('"')
            inq = !inq
        elseif !inq && (b == UInt8('\n') || b == UInt8('\r'))
            return view(bytes, 1:(i - 1))
        end
    end
    return view(bytes, 1:length(bytes))
end

"""Number of `delim` characters outside double quotes."""
function _gui_count_delims(line, delim::Char)
    d = UInt8(delim)
    inq = false
    n = 0
    for b in line
        if b == UInt8('"')
            inq = !inq
        elseif !inq && b == d
            n += 1
        end
    end
    return n
end

function _gui_pick_delim(line, choice::AbstractString)
    choice == "auto" || return _GUI_DELIMS[choice]
    counts = [(c, _gui_count_delims(line, c)) for c in (',', '\t', ';')]
    best = counts[argmax(last.(counts))]
    return last(best) == 0 ? ',' : first(best)
end

_gui_is_numeric_col(v::AbstractVector) = nonmissingtype(eltype(v)) <: Real

"""Parse an uploaded table. Every limit is checked before or during parsing."""
function _gui_parse_table(bytes::Vector{UInt8}, cfg::_GuiConfig,
                          delim_choice::AbstractString)
    haskey(_GUI_DELIMS, delim_choice) || delim_choice == "auto" ||
        throw(_GuiUserError(400, "Unknown delimiter option."))
    isempty(bytes) && throw(_GuiUserError(422, "The uploaded file is empty."))
    isvalid(String, bytes) ||
        throw(_GuiUserError(422, "The file must be UTF-8 encoded text (CSV or TSV)."))
    start = length(bytes) >= 3 && bytes[1:3] == UInt8[0xef, 0xbb, 0xbf] ? 4 : 1
    header = _gui_header_line(view(bytes, start:length(bytes)))
    delim = _gui_pick_delim(header, delim_choice)
    ncols_header = _gui_count_delims(header, delim) + 1
    ncols_header > cfg.max_cols && throw(_GuiUserError(422,
        "The file has more than $(cfg.max_cols) columns."))
    f = try
        CSV.File(bytes; delim=delim, header=1, limit=cfg.max_rows + 1, ntasks=1,
                 normalizenames=false, pool=false, stringtype=String,
                 silencewarnings=true, strict=false)
    catch err
        @warn "DrSnow GUI: CSV parsing failed" exception = err
        throw(_GuiUserError(422, "The file could not be parsed as a delimited table."))
    end
    df = DataFrame(f; copycols=true)
    ncol(df) == 0 && throw(_GuiUserError(422, "The file has no columns."))
    ncol(df) > cfg.max_cols && throw(_GuiUserError(422,
        "The file has more than $(cfg.max_cols) columns."))
    nrow(df) == 0 && throw(_GuiUserError(422, "The file has a header but no data rows."))
    nrow(df) > cfg.max_rows && throw(_GuiUserError(422,
        "The file has more than $(cfg.max_rows) data rows."))
    for n in names(df)
        length(n) > _GUI_MAX_NAME_CHARS && throw(_GuiUserError(422,
            "Column names must be at most $(_GUI_MAX_NAME_CHARS) characters long."))
    end
    return df
end

function _gui_cell(x)
    x === missing && return ""
    s = x isa AbstractString ? String(x) : string(x)
    return length(s) > _GUI_MAX_CELL_CHARS ? first(s, _GUI_MAX_CELL_CHARS) * "…" : s
end

function _gui_column_kind(v::AbstractVector)
    T = nonmissingtype(eltype(v))
    T <: Bool && return "binary"
    if T <: Real
        vals = Set(skipmissing(v))
        return issubset(vals, (0, 1)) ? "binary" : "numeric"
    end
    return T <: AbstractString ? "text" : "other"
end

function _gui_table_summary(df::DataFrame, filename::AbstractString)
    cols = [Dict("name" => n, "kind" => _gui_column_kind(df[!, n]),
                 "n_missing" => count(ismissing, df[!, n]),
                 "n_unique" => length(unique(df[!, n])))
            for n in names(df)]
    k = min(nrow(df), _GUI_PREVIEW_ROWS)
    rows = [[_gui_cell(df[i, j]) for j in 1:ncol(df)] for i in 1:k]
    return Dict("filename" => filename, "n_rows" => nrow(df), "n_cols" => ncol(df),
                "columns" => cols, "preview" => rows)
end

"""Keep a short, printable file name for display (it is never used as a path)."""
function _gui_clean_filename(name::AbstractString)
    s = replace(String(name), r"[^\w .\-()]" => "_")
    s = strip(s)
    isempty(s) && return "upload.csv"
    return length(s) > 120 ? first(s, 120) : s
end
