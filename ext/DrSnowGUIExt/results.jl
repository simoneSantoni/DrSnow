# Conversion of DrSnow results into plain JSON-ready dictionaries, and CSV export.

"""JSON-safe number: non-finite values become the strings "Inf"/"-Inf" or `nothing`."""
function _gui_num(x)
    x === nothing && return nothing
    x === missing && return nothing
    x isa Bool && return x
    x isa Integer && return Int(x)
    x isa Real || return _gui_cell(x)
    y = Float64(x)
    isnan(y) && return nothing
    isinf(y) && return y > 0 ? "Inf" : "-Inf"
    return y
end

_gui_label(x) = x isa Real && !(x isa Bool) ? _gui_num(x) : _gui_cell(x)

"""A table: `columns` (header strings) and `rows` (vectors of strings/numbers)."""
_gui_table(title, columns, rows; note="") =
    Dict{String,Any}("title" => String(title), "columns" => collect(String, columns),
                     "rows" => rows, "note" => String(note))

"""Coefficient table from the core `coeftable(r; level)`."""
function _gui_coef_table(r, level::Real; title::AbstractString="Estimates")
    ct = coeftable(r; level=level)
    cols = ["Term"; ct.colnms]
    rows = [Any[ct.rownms[i]; [_gui_num(c[i]) for c in ct.cols]]
            for i in eachindex(ct.rownms)]
    return _gui_table(title, cols, rows)
end

function _gui_df_table(df::AbstractDataFrame, title::AbstractString; note="",
                       max_rows::Int=500)
    k = min(nrow(df), max_rows)
    rows = [Any[_gui_label(df[i, j]) for j in 1:ncol(df)] for i in 1:k]
    nrow(df) > k && (note = strip(note * " Showing the first $k of $(nrow(df)) rows."))
    return _gui_table(title, names(df), rows; note=note)
end

"""Verdict line with the same wording as `show(::DiagnosticTest)`."""
function _gui_verdict(p::Real)
    isnan(p) && return "Not computable."
    p < 0.05 && return "H₀ rejected at the 5% level."
    return "H₀ not rejected at the 5% level. Non-rejection is not evidence that H₀ " *
           "is true."
end

function _gui_diagnostic(t::DiagnosticTest)
    return Dict{String,Any}(
        "name" => t.name, "null" => t.null, "method" => t.method, "note" => t.note,
        "statistic" => _gui_num(t.statistic), "dof" => [_gui_num(d) for d in t.dof],
        "pvalue" => _gui_num(t.pvalue), "verdict" => _gui_verdict(t.pvalue),
        "text" => _gui_text(t))
end

"""A free-form diagnostic block (text produced by the package's own `show`)."""
_gui_text_block(title::AbstractString, obj) =
    Dict{String,Any}("title" => String(title), "text" => _gui_text(obj))

_gui_text(obj) = sprint(show, MIME"text/plain"(), obj; context=:displaysize => (60, 120))

function _gui_new_result(design::AbstractString, r)
    return Dict{String,Any}(
        "design" => String(design),
        "title" => r isa CausalEstimate ? method_name(r) : String(design),
        "estimand" => r isa CausalEstimate ? estimand(r) : "",
        "nobs" => r isa CausalEstimate ? nobs(r) : nothing,
        "tables" => Any[], "diagnostics" => Any[], "text_blocks" => Any[],
        "warnings" => String[], "notes" => String[], "plot" => nothing,
        "summary" => _gui_text(r))
end

# --- Plot payloads (drawn with Plotly in the browser) ---------------------------------

"""Event-study coefficients with pointwise CIs, including the reference period(s)."""
function _gui_event_plot(es::EventStudyEstimate, level::Real; title="Event study")
    rp = relative_periods(es)
    ci = confint(es; level=level)
    x = collect(Int, rp)
    y = Any[_gui_num(v) for v in coef(es)]
    lo = Any[_gui_num(v) for v in ci[:, 1]]
    hi = Any[_gui_num(v) for v in ci[:, 2]]
    return Dict{String,Any}("kind" => "event_study", "title" => title, "x" => x, "y" => y,
                            "lower" => lo, "upper" => hi,
                            "reference" => collect(Int, es.reference), "level" => level)
end

function _gui_rd_plot(pd, outcome::Symbol, running::Symbol)
    bins = pd.bins
    side(s) = [i for i in 1:nrow(bins) if bins.side[i] === s]
    polyside(s) = [i for i in 1:nrow(pd.poly) if pd.poly.side[i] === s]
    part(ix, df, xcol, ycol) = Dict("x" => [_gui_num(df[i, xcol]) for i in ix],
                                    "y" => [_gui_num(df[i, ycol]) for i in ix])
    return Dict{String,Any}(
        "kind" => "rd", "title" => "RD plot (binned means and global polynomial fit)",
        "cutoff" => _gui_num(pd.cutoff), "xlabel" => String(running),
        "ylabel" => String(outcome),
        "bins_left" => part(side(:left), bins, :mean_x, :mean_y),
        "bins_right" => part(side(:right), bins, :mean_x, :mean_y),
        "poly_left" => part(polyside(:left), pd.poly, :x, :y),
        "poly_right" => part(polyside(:right), pd.poly, :x, :y),
        "note" => pd.binselect_description)
end

function _gui_sdid_plot(gaps::DataFrame, outcome::Symbol)
    hascohort = "cohort" in names(gaps)
    cohorts = hascohort ? unique(gaps.cohort) : [nothing]
    series = map(cohorts) do c
        ix = hascohort ? findall(==(c), gaps.cohort) : collect(1:nrow(gaps))
        post = gaps.post[ix]
        k = findfirst(post)
        Dict("cohort" => c === nothing ? "" : _gui_cell(c),
             "time" => [_gui_label(gaps.time[i]) for i in ix],
             "treated" => [_gui_num(gaps.treated[i]) for i in ix],
             "synthetic" => [_gui_num(gaps.synthetic[i]) for i in ix],
             "adoption" => k === nothing ? nothing : _gui_label(gaps.time[ix[k]]))
    end
    return Dict{String,Any}("kind" => "sdid", "title" => "Treated and synthetic control " *
                            "trajectories", "ylabel" => String(outcome), "series" => series)
end

# --- CSV export ------------------------------------------------------------------------

"""Neutralize spreadsheet formula injection in text cells."""
function _gui_csv_cell(x)
    x === nothing && return ""
    x isa Real && return x
    s = string(x)
    if !isempty(s) && first(s) in ('=', '+', '-', '@', '\t', '\r') &&
       tryparse(Float64, s) === nothing
        return "'" * s
    end
    return s
end

function _gui_table_csv(tbl::Dict{String,Any})
    cols = tbl["columns"]
    df = DataFrame([Symbol("c$j") => Any[_gui_csv_cell(row[j]) for row in tbl["rows"]]
                    for j in eachindex(cols)])
    io = IOBuffer()
    CSV.write(io, df; header=[_gui_csv_cell(c) for c in cols])
    return take!(io)
end
