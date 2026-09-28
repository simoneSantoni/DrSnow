# Validation of analysis options sent by the frontend. Every value is checked against
# the uploaded table or a fixed allow-list; nothing the user sends is ever evaluated.

const _GUI_MAX_LIST = 50

function _gui_param(p, key::AbstractString)
    return get(p, Symbol(key), nothing)
end

_gui_bad(msg) = throw(_GuiUserError(400, msg))
_gui_invalid(msg) = throw(_GuiUserError(422, msg))

"""A column chosen for `role`: must be a string naming a column of `df`; returned as
a `Symbol` (never parsed)."""
function _gui_column(df::DataFrame, p, key::AbstractString, role::AbstractString;
                     required::Bool=true, numeric::Bool=false)
    v = _gui_param(p, key)
    if v === nothing || (v isa AbstractString && isempty(v))
        required && _gui_invalid("Choose a column for the $role.")
        return nothing
    end
    v isa AbstractString || _gui_bad("The $role must be given as a column name.")
    name = String(v)
    name in names(df) || _gui_invalid("The $role is not a column of the uploaded table.")
    numeric && !_gui_is_numeric_col(df[!, name]) &&
        _gui_invalid("The $role must be a numeric column.")
    return Symbol(name)
end

function _gui_columns(df::DataFrame, p, key::AbstractString, role::AbstractString;
                      numeric::Bool=false, min::Integer=0)
    v = _gui_param(p, key)
    v === nothing && (v = String[])
    v isa AbstractVector || _gui_bad("The $role must be a list of column names.")
    length(v) <= _GUI_MAX_LIST || _gui_invalid("Too many columns selected as $role.")
    out = Symbol[]
    for x in v
        x isa AbstractString || _gui_bad("The $role must be a list of column names.")
        name = String(x)
        name in names(df) ||
            _gui_invalid("A column selected as $role is not in the uploaded table.")
        numeric && !_gui_is_numeric_col(df[!, name]) &&
            _gui_invalid("Every column selected as $role must be numeric.")
        Symbol(name) in out || push!(out, Symbol(name))
    end
    length(out) >= min || _gui_invalid("Select at least $min column(s) as $role.")
    return out
end

"""One of a fixed set of options, returned as a `Symbol`."""
function _gui_choice(p, key::AbstractString, allowed, default::Symbol)
    v = _gui_param(p, key)
    v === nothing && return default
    v isa AbstractString || _gui_bad("Invalid value for option '$key'.")
    for a in allowed
        v == string(a) && return a
    end
    _gui_bad("Invalid value for option '$key'.")
end

function _gui_bool(p, key::AbstractString, default::Bool)
    v = _gui_param(p, key)
    v === nothing && return default
    v isa Bool || _gui_bad("Option '$key' must be true or false.")
    return v
end

function _gui_int(p, key::AbstractString, default, lo::Integer, hi::Integer)
    v = _gui_param(p, key)
    (v === nothing || (v isa AbstractString && isempty(v))) && return default
    x = v isa Integer ? v :
        v isa AbstractFloat && isinteger(v) ? Int(v) :
        v isa AbstractString ? tryparse(Int, strip(v)) : nothing
    x === nothing && _gui_bad("Option '$key' must be a whole number.")
    lo <= x <= hi || _gui_invalid("Option '$key' must be between $lo and $hi.")
    return Int(x)
end

function _gui_float(p, key::AbstractString, default, lo::Real, hi::Real)
    v = _gui_param(p, key)
    (v === nothing || (v isa AbstractString && isempty(v))) && return default
    x = v isa Real && !(v isa Bool) ? Float64(v) :
        v isa AbstractString ? tryparse(Float64, strip(v)) : nothing
    (x === nothing || !isfinite(x)) && _gui_bad("Option '$key' must be a number.")
    lo <= x <= hi || _gui_invalid("Option '$key' must be between $lo and $hi.")
    return x
end

_gui_level(p) = _gui_float(p, "level", 0.95, 0.5, 0.999)

"""Cluster option: `"none"` or a column name (defaults to `default`)."""
function _gui_cluster(df::DataFrame, p, default)
    v = _gui_param(p, "cluster")
    (v === nothing || v == "") && return default
    v == "none" && return nothing
    return _gui_column(df, p, "cluster", "cluster variable")
end

"""Treatment argument for DiD: a 0/1 indicator column or `FirstTreated(column)`."""
function _gui_did_treatment(df::DataFrame, p)
    col = _gui_column(df, p, "treatment", "treatment variable"; numeric=true)
    kind = _gui_choice(p, "treatment_type", (:indicator, :first_treated), :indicator)
    return kind === :first_treated ? FirstTreated(col) : col, col, kind
end

"""RNG for stochastic procedures: seeded from the request (reproducible) or fresh."""
_gui_rng(p) = Random.Xoshiro(_gui_int(p, "seed", 20260927, 0, typemax(Int32)))

function _gui_distinct!(cols...)
    syms = [c for c in cols if c !== nothing]
    length(unique(syms)) == length(syms) ||
        _gui_invalid("Each role must use a different column.")
    return nothing
end
