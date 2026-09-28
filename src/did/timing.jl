# Treatment-timing (cohort) layer shared by every DiD estimator.
#
# Conventions (documented in `TreatmentTiming`):
# - Time is mapped to an ordered period index 1..T (sorted unique values of the time
#   column), so Dates, non-integer or gapped time variables are handled uniformly and
#   event time is measured in periods.
# - A unit's cohort `G_i` is the period index of its first treated period; `0` codes
#   never treated (within the sample), i.e. `G_i = ∞`.

"""
    FirstTreated(column; never=0)

Marker that a column holds each unit's **first-treatment period** ``G_i`` rather than
a 0/1 treatment indicator ``D_{it}``; pass it wherever a DiD function expects the
treatment argument.

Staggered-adoption designs are fully described by the cohort ``G_i``, the first
period in which unit ``i`` is treated, with ``G_i = \\infty`` for units never treated
in the sample; under absorbing treatment ``D_{it} = 1\\{t \\ge G_i\\}`` (Callaway and
Sant'Anna, 2021; Sun and Abraham, 2021). Supplying ``G_i`` directly is the convention
of R's `did` package and is required for **repeated cross-sections**, where a 0/1
indicator cannot reveal the cohort of an observation drawn before its group was
treated. Values are in the units of the time variable. Never-treated units are coded
by `never` (default `0`, as in R's `did`), `missing`, `nothing` or `Inf`. A cohort
after the last observed period is treated as never treated within the sample, and a
cohort that falls between two observed periods starts at the next observed period.

# Arguments
- `column`: name (`Symbol` or string) of the first-treatment-period column.

# Keywords
- `never = 0`: the value that codes never-treated units, in addition to `missing`,
  `nothing` and `Inf`.

# Returns
- `FirstTreated`: a marker with fields `column::Symbol` and `never`.

# Examples
```julia
using DrSnow, CSV, DataFrames
file = joinpath(pkgdir(DrSnow), "test", "validation", "did", "mpdta.csv")
mpdta = CSV.read(file, DataFrame)
cs = did_callaway_santanna(mpdta, :lemp, FirstTreated(:first_treat), :countyreal,
                           :year; bootstrap=false)
```

# References
- Callaway, B., & Sant'Anna, P. H. C. (2021). Difference-in-differences with multiple
  time periods. *Journal of Econometrics*, 225(2), 200–230.
"""
struct FirstTreated
    column::Symbol
    never::Any
end
FirstTreated(column; never=0) = FirstTreated(Symbol(column), never)

_did_treatment_column(t::Symbol) = t
_did_treatment_column(t::FirstTreated) = t.column
_did_treatment_column(t::AbstractString) = Symbol(t)

"""
    TreatmentTiming

Treatment timing of a panel or of repeated cross-sections: the ordered period index,
each unit's cohort, treatment status, and the design checks every DiD estimator of
the package relies on. Build it with [`treatment_timing`](@ref).

Time is mapped to an ordered index ``1, \\dots, T`` of the sorted distinct values of
the time variable, so `Date`s, non-integer and gapped time variables are handled
uniformly and event time ``e = t - G_i`` is counted in observed periods. A unit's
cohort ``G_i`` is the index of its first treated period, coded `0` for units never
treated in the sample (``G_i = \\infty``). The flags record whether treatment is
absorbing (staggered adoption, required by the cohort-based estimators) and whether
the panel is balanced; `anticipation` is the number of periods before ``G_i`` in
which units may already respond, which the estimators use to shift the last clean
pre-treatment period to ``G_i - 1 - \\text{anticipation}``.

# Fields
- `periods::Vector`: sorted distinct values of the time variable; period `p` is
  `periods[p]`.
- `units::Vector`: unit identifiers (sorted when sortable); empty for repeated
  cross-sections.
- `row_period::Vector{Int}`: period index of every data row.
- `row_unit::Vector{Int}`: unit index of every data row (0 for repeated
  cross-sections).
- `unit_cohort::Vector{Int}`: first treated period index ``G_i`` of every unit;
  **`0` = never treated**.
- `row_cohort::Vector{Int}`: cohort of every data row.
- `row_treated::BitVector`: treatment status ``D_{it}`` of every row (the observed
  indicator, or ``t \\ge G_i`` when built from [`FirstTreated`](@ref)).
- `absorbing::Bool`: `true` when no unit leaves treatment (staggered adoption).
- `balanced::Bool`: every unit is observed in every period exactly once.
- `panel::Bool`: `false` for repeated cross-sections.
- `anticipation::Int`: number of anticipation periods.

The relative (event) time of a row of a treated cohort is
`row_period - row_cohort`.
"""
struct TreatmentTiming{T,U}
    periods::Vector{T}
    units::Vector{U}
    row_period::Vector{Int}
    row_unit::Vector{Int}
    unit_cohort::Vector{Int}
    row_cohort::Vector{Int}
    row_treated::BitVector
    absorbing::Bool
    balanced::Bool
    panel::Bool
    anticipation::Int
end

"""
    treatment_timing(data, treatment, unit, time; anticipation=0) -> TreatmentTiming

Construct the [`TreatmentTiming`](@ref) of a dataset: the ordered period index,
cohorts (first treated period, `0` = never treated), treatment status, and the
absorbing-treatment and balanced-panel checks.

Every staggered-adoption estimator starts from this object, so it is also the
quickest way to inspect a design before estimation: how many cohorts there are and
how large they are, whether never-treated units exist (they are needed as the
comparison group by some estimators and for a fully dynamic TWFE event study), and
whether any unit switches out of treatment. Designs in which treatment turns on and
off, or is not binary, fall outside the cohort framework; see
[`did_multiplegt_dyn`](@ref). The function throws an error for duplicated
unit–period rows and, when `treatment` is a 0/1 indicator, for values other than 0
and 1.

# Arguments
- `data`: a `DataFrame` without missing values in the used columns (a
  [`FirstTreated`](@ref) column may contain `missing` for never-treated units).
- `treatment`: a 0/1 (or `Bool`) treatment-indicator column ``D_{it}``, or
  [`FirstTreated`](@ref)`(column)` giving the first treated period.
- `unit`: unit identifier column, or `nothing` for repeated cross-sections (then
  `treatment` must be `FirstTreated`).
- `time`: time column of any sortable type (integers, `Date`s, …).

# Keywords
- `anticipation::Integer = 0`: number of periods before ``G_i`` in which units may
  respond to treatment; stored for the estimators.

# Returns
- `TreatmentTiming`.

# Examples
```julia
using DrSnow, CSV, DataFrames
file = joinpath(pkgdir(DrSnow), "test", "validation", "did", "mpdta.csv")
mpdta = CSV.read(file, DataFrame)
tm = treatment_timing(mpdta, :d, :countyreal, :year)
tm.periods[filter(>(0), unique(tm.unit_cohort))]   # cohorts in years
tm.absorbing, tm.balanced
```
"""
function treatment_timing(data, treatment, unit, time; anticipation::Integer=0)
    anticipation >= 0 || throw(ArgumentError("anticipation must be ≥ 0"))
    tcol = _did_treatment_column(treatment)
    require_columns(data, [tcol, unit, time]; context="treatment_timing")
    for c in (tcol, unit, time)
        (c === nothing || (treatment isa FirstTreated && c === tcol)) && continue
        any(ismissing, data[!, c]) && throw(ArgumentError(
            "treatment_timing: column `$c` contains missing values"))
    end
    tvals = data[!, time]
    periods = sort!(unique(tvals))
    pindex = Dict(p => i for (i, p) in enumerate(periods))
    row_period = [pindex[v] for v in tvals]
    T = length(periods)
    n = nrow(data)
    if unit === nothing
        treatment isa FirstTreated || throw(ArgumentError(
            "repeated cross-sections need the cohort of every observation: pass " *
            "`FirstTreated(column)` as the treatment argument"))
        row_cohort = [_did_cohort_index(v, periods, treatment.never)
                      for v in data[!, tcol]]
        row_treated = BitVector(row_cohort[i] > 0 && row_period[i] >= row_cohort[i]
                                for i in 1:n)
        return TreatmentTiming(periods, Int[], row_period, zeros(Int, n), Int[],
                               row_cohort, row_treated, true, false, false,
                               Int(anticipation))
    end
    uvals = data[!, unit]
    units = unique(uvals)
    try
        sort!(units)
    catch
    end
    uindex = Dict(u => i for (i, u) in enumerate(units))
    row_unit = [uindex[v] for v in uvals]
    N = length(units)
    # Duplicates (unit, period)
    seen = falses(N, T)
    for i in 1:n
        seen[row_unit[i], row_period[i]] && throw(ArgumentError(
            "treatment_timing: unit $(units[row_unit[i]]) is observed more than once " *
            "in period $(periods[row_period[i]])"))
        seen[row_unit[i], row_period[i]] = true
    end
    balanced = all(seen)
    order = sortperm(collect(zip(row_unit, row_period)))
    unit_cohort = zeros(Int, N)
    absorbing = true
    if treatment isa FirstTreated
        gvals = data[!, tcol]
        assigned = falses(N)
        for i in 1:n
            g = _did_cohort_index(gvals[i], periods, treatment.never)
            u = row_unit[i]
            if assigned[u] && unit_cohort[u] != g
                throw(ArgumentError("treatment_timing: first-treatment period of unit " *
                                    "$(units[u]) varies over time"))
            end
            unit_cohort[u] = g
            assigned[u] = true
        end
        row_cohort = unit_cohort[row_unit]
        row_treated = BitVector(row_cohort[i] > 0 && row_period[i] >= row_cohort[i]
                                for i in 1:n)
    else
        D = _did_binary(data[!, tcol], tcol)
        last_state = zeros(Int8, N)   # 0 = not yet treated, 1 = treated
        for i in order
            u = row_unit[i]
            if D[i]
                unit_cohort[u] == 0 && (unit_cohort[u] = row_period[i])
                last_state[u] = 1
            elseif last_state[u] == 1
                absorbing = false
            end
        end
        row_cohort = unit_cohort[row_unit]
        row_treated = BitVector(D)
    end
    return TreatmentTiming(periods, units, row_period, row_unit, unit_cohort, row_cohort,
                           row_treated, absorbing, balanced, true, Int(anticipation))
end

function _did_binary(x, name)
    out = BitVector(undef, length(x))
    for (i, v) in enumerate(x)
        if v == 1
            out[i] = true
        elseif v == 0
            out[i] = false
        else
            throw(ArgumentError("treatment column `$name` must be 0/1 (or Bool); found " *
                                "value $(repr(v)). For a first-treatment-period column " *
                                "use FirstTreated(:$name)."))
        end
    end
    return out
end

function _did_is_never(v, never)
    v === missing && return true
    v === nothing && return true
    v isa AbstractFloat && isinf(v) && return true
    return isequal(v, never)
end

function _did_cohort_index(v, periods, never)
    _did_is_never(v, never) && return 0
    idx = searchsortedfirst(periods, v)
    return idx > length(periods) ? 0 : idx
end

# Treated cohorts (period indices, sorted) and their unit counts.
function _did_cohorts(tm::TreatmentTiming)
    g = tm.panel ? tm.unit_cohort : tm.row_cohort
    return sort!(unique(filter(>(0), g)))
end

_did_n_never(tm::TreatmentTiming) =
    count(==(0), tm.panel ? tm.unit_cohort : tm.row_cohort)
_did_n_ever(tm::TreatmentTiming) =
    count(>(0), tm.panel ? tm.unit_cohort : tm.row_cohort)

_did_staggered(tm::TreatmentTiming) = length(_did_cohorts(tm)) > 1

# Event time of each row (typemin(Int) for never-treated rows).
_did_event_time(tm::TreatmentTiming) =
    [g > 0 ? p - g : typemin(Int) for (p, g) in zip(tm.row_period, tm.row_cohort)]

"""
    _did_control_units(cohort, g, t, base, control_group, anticipation)

Mask of valid comparison units for cohort `g` at period `t` with base period `base`
(all period indices): never-treated units, plus (for `:not_yet_treated`) units whose
cohort starts after `max(t, base) + anticipation` and is not `g`.
"""
function _did_control_units(cohort::AbstractVector{Int}, g::Int, t::Int, base::Int,
                            control_group::Symbol, anticipation::Int)
    if control_group === :never_treated
        return cohort .== 0
    elseif control_group === :not_yet_treated
        thr = max(t, base) + anticipation
        return (cohort .== 0) .| ((cohort .> thr) .& (cohort .!= g))
    end
    throw(ArgumentError("control_group must be :never_treated or :not_yet_treated"))
end

function _did_check_control_group(cg)
    cg in (:never_treated, :not_yet_treated) ||
        throw(ArgumentError("control_group must be :never_treated or :not_yet_treated"))
    return cg
end

function Base.show(io::IO, ::MIME"text/plain", tm::TreatmentTiming)
    println(io, "TreatmentTiming (", tm.panel ? "panel" : "repeated cross-sections", ")")
    println(io, "Periods: ", length(tm.periods), " (", first(tm.periods), " … ",
            last(tm.periods), ")")
    gs = tm.panel ? tm.unit_cohort : tm.row_cohort
    what = tm.panel ? "units" : "observations"
    println(io, "Never treated ", what, ": ", count(==(0), gs))
    for g in _did_cohorts(tm)
        println(io, "Cohort first treated in ", tm.periods[g], ": ", count(==(g), gs),
                " ", what)
    end
    tm.panel && println(io, "Balanced: ", tm.balanced, "; absorbing treatment: ",
                        tm.absorbing)
    tm.anticipation > 0 && println(io, "Anticipation periods: ", tm.anticipation)
end

Base.show(io::IO, tm::TreatmentTiming) =
    print(io, "TreatmentTiming(", length(tm.periods), " periods, ",
          length(_did_cohorts(tm)), " cohorts)")

# ---------------------------------------------------------------------------
# Data preparation helpers
# ---------------------------------------------------------------------------

# Keep only the needed columns and drop rows with missing values in them. A
# `FirstTreated` column may contain `missing` (= never treated) and is not filtered.
function _did_prepare(data, cols; context, treatment=nothing)
    cols = unique(Symbol[c for c in cols if c !== nothing])
    require_columns(data, cols; context=context)
    df = DataFrame(data; copycols=false)[:, cols]     # copies only the used columns
    n0 = nrow(df)
    check = treatment isa FirstTreated ? setdiff(cols, [treatment.column]) : cols
    df = dropmissing(df, check)
    nrow(df) < n0 && @warn "$(context): dropped $(n0 - nrow(df)) rows with missing values"
    nrow(df) > 0 || throw(ArgumentError("$(context): no complete observations"))
    return df
end

# A column name not present in `df`.
function _did_fresh_name(df, base::AbstractString)
    names_ = Set(propertynames(df))
    s = Symbol(base)
    k = 0
    while s in names_
        k += 1
        s = Symbol(base, "_", k)
    end
    return s
end

_did_cluster_symbols(c::Nothing) = Symbol[]
_did_cluster_symbols(c::Symbol) = [c]
_did_cluster_symbols(c::AbstractVector) = Symbol[Symbol(x) for x in c]

function _did_vcov_estimator(cluster, vcov)
    vcov === nothing || return vcov
    cs = _did_cluster_symbols(cluster)
    isempty(cs) && return Vcov.robust()
    return Vcov.cluster(cs...)
end

_did_nclusters(m) = m.nclusters === nothing ? 0 : minimum(values(m.nclusters))
