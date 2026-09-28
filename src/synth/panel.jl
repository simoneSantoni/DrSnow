# Panel preparation for synthetic-control-type estimators.
#
# Long data are reshaped into a units × periods outcome matrix by key (unit id and time
# value), never by row position, so every estimator in this area is invariant to the
# order of the input rows.

"""
    SynthPanel{U,T}

Balanced panel in matrix form, the common input of the synthetic control, synthetic
difference-in-differences, augmented synthetic control and matrix completion
estimators.

Synthetic-control-type methods compare treated units with weighted combinations of
untreated units over time, so they work with an ``N \\times T`` outcome matrix
``Y = (Y_{it})`` and an adoption pattern rather than with long data. `SynthPanel` stores
that matrix with a deterministic row order. Never-treated (control, or *donor*) units
come first, sorted by unit id, followed by treated units sorted by adoption period and
unit id. Columns are the sorted time periods. Treatment must be binary and absorbing
(once treated, a unit stays treated). Every treated unit needs at least one untreated
period, and at least one unit must never be treated, since only never-treated units
serve as controls. Because cells are matched by `(unit, time)` key, every estimator in
the area is invariant to the order of the input rows.

# Fields
- `Y::Matrix{Float64}`: `N × T` outcome matrix.
- `units::Vector{U}`: unit ids in row order.
- `times::Vector{T}`: sorted time values (column order).
- `adoption::Vector{Int}`: for each row, the column index of the first treated period,
  or `0` for never-treated units.
- `n_control::Int`: number of never-treated units (rows `1:n_control`).
- `X::Array{Union{Missing,Float64},3}`: `N × T × K` covariates, which may contain
  `missing`; estimators that need complete covariates check this themselves.
- `covariates::Vector{Symbol}`: covariate names (third dimension of `X`).
- `outcome`, `treatment`, `unit`, `time`: source column names.

Build it with [`synth_panel`](@ref). All estimators of the area accept either a
`SynthPanel` or the long data with column names.
"""
struct SynthPanel{U,T}
    Y::Matrix{Float64}
    units::Vector{U}
    times::Vector{T}
    adoption::Vector{Int}
    n_control::Int
    X::Array{Union{Missing,Float64},3}
    covariates::Vector{Symbol}
    outcome::Symbol
    treatment::Symbol
    unit::Symbol
    time::Symbol
end

"""
    synth_panel(data, outcome, treatment, unit, time;
                covariates=Symbol[]) -> SynthPanel

Reshape long panel data into a [`SynthPanel`](@ref), validating the balance and
adoption structure that synthetic-control-type estimators require.

Each row of `data` is one unit-period, and cells are matched by the `(unit, time)` key,
never by row position. The panel must be balanced: every unit is observed exactly once
in every period, with a non-missing, finite outcome and a binary treatment. An
unbalanced panel raises an `ArgumentError` that reports the number of missing cells and
an example. No imputation is attempted, because silently filling outcomes would change
the estimand; drop incomplete units or restrict the period range instead, and report
which units were removed. Treatment must be absorbing, with block adoption (all treated
units start together) or staggered adoption (different start dates). Designs in which
treatment switches off, units treated in the first period, and panels without
never-treated units are rejected. Covariates may contain `missing` values; estimators
that need complete covariates check this themselves.

# Arguments
- `data`: a `DataFrame` or any Tables.jl table in long format.
- `outcome::Symbol`: outcome column.
- `treatment::Symbol`: treatment indicator (0/1 or `Bool`), absorbing within unit.
- `unit::Symbol`: unit identifier column.
- `time::Symbol`: time period column. Values must be sortable (integers, dates, …).

# Keywords
- `covariates::Vector{Symbol}=Symbol[]`: time-varying or time-invariant covariate
  columns stored in the `X` array, used as predictors by [`synthetic_control`](@ref)
  or as regressors by [`synthetic_did`](@ref).

# Returns
- [`SynthPanel`](@ref).

# Examples
```julia
using DrSnow, CSV, DataFrames
prop99 = CSV.read(pkgdir(DrSnow, "test", "validation", "synth", "prop99.csv"), DataFrame)
p = synth_panel(prop99, :PacksPerCapita, :treated, :State, :Year)
size(p.Y)                           # (units, periods)
p.units[(p.n_control + 1):end]      # treated units
p.times[p.adoption[end]]            # first treated period
```
"""
function synth_panel(data, outcome::Symbol, treatment::Symbol, unit::Symbol,
                     time::Symbol; covariates::Vector{Symbol}=Symbol[])
    df = data isa AbstractDataFrame ? data : DataFrame(data)
    require_columns(df, [outcome, treatment, unit, time, covariates...];
                    context="synth_panel")
    length(unique([outcome, treatment, unit, time])) == 4 ||
        throw(ArgumentError("synth_panel: outcome, treatment, unit and time must be " *
                            "distinct columns"))
    nrow(df) > 0 || throw(ArgumentError("synth_panel: data has no rows"))
    uid = df[!, unit]
    tid = df[!, time]
    any(ismissing, uid) && throw(ArgumentError("synth_panel: missing values in `$unit`"))
    any(ismissing, tid) && throw(ArgumentError("synth_panel: missing values in `$time`"))
    units = _sc_sorted_unique(uid, unit)
    times = _sc_sorted_unique(tid, time)
    N, T = length(units), length(times)
    uidx = Dict(u => i for (i, u) in enumerate(units))
    tidx = Dict(t => j for (j, t) in enumerate(times))
    K = length(covariates)
    Y = fill(NaN, N, T)
    W = fill(-1, N, T)
    X = Array{Union{Missing,Float64},3}(missing, N, T, K)
    seen = falses(N, T)
    ycol = df[!, outcome]
    dcol = df[!, treatment]
    xcols = [df[!, c] for c in covariates]
    for r in 1:nrow(df)
        i = uidx[uid[r]]
        j = tidx[tid[r]]
        if seen[i, j]
            throw(ArgumentError("synth_panel: duplicate observation for unit " *
                                "$(repr(units[i])) in period $(repr(times[j]))"))
        end
        seen[i, j] = true
        y = ycol[r]
        if ismissing(y) || !(y isa Real) || !isfinite(y)
            throw(ArgumentError("synth_panel: outcome `$outcome` is missing or not " *
                                "finite for unit $(repr(units[i])) in period " *
                                "$(repr(times[j]))"))
        end
        Y[i, j] = y
        W[i, j] = _sc_treatment_value(dcol[r], treatment)
        for k in 1:K
            v = xcols[k][r]
            X[i, j, k] = ismissing(v) ? missing : Float64(v)
        end
    end
    n_missing = count(!, seen)
    if n_missing > 0
        idx = findfirst(!, seen)
        throw(ArgumentError("synth_panel: unbalanced panel, $n_missing unit-period " *
                            "cell(s) missing (e.g. unit $(repr(units[idx[1]])) in " *
                            "period $(repr(times[idx[2]]))). Synthetic control " *
                            "estimators need a balanced panel: drop incomplete units " *
                            "or restrict the period range."))
    end
    return _sc_panel_from_matrices(Y, W, units, times, X, covariates;
                                   outcome=outcome, treatment=treatment, unit=unit,
                                   time=time)
end

function _sc_sorted_unique(v, name)
    u = unique(v)
    try
        return sort(u)
    catch err
        err isa MethodError || rethrow()
        throw(ArgumentError("synth_panel: values of `$name` cannot be sorted"))
    end
end

function _sc_treatment_value(d, name)
    ismissing(d) && throw(ArgumentError("synth_panel: missing values in `$name`"))
    if d isa Bool
        return Int(d)
    elseif d isa Real && (d == 0 || d == 1)
        return Int(d)
    end
    throw(ArgumentError("synth_panel: treatment `$name` must be binary (0/1), got $d"))
end

# Build a SynthPanel from matrices whose rows/columns are already keyed by
# `units`/`times` (in any row order). Validates absorbing treatment and orders rows.
function _sc_panel_from_matrices(Y::AbstractMatrix, W::AbstractMatrix, units, times,
                                 X=Array{Union{Missing,Float64},3}(missing,
                                                                   size(Y)..., 0),
                                 covariates=Symbol[]; outcome::Symbol=:outcome,
                                 treatment::Symbol=:treatment, unit::Symbol=:unit,
                                 time::Symbol=:time)
    N, T = size(Y)
    size(W) == (N, T) || throw(DimensionMismatch("Y and W must have the same size"))
    adoption = zeros(Int, N)
    for i in 1:N
        first_t = findfirst(==(1), view(W, i, :))
        first_t === nothing && continue
        if any(!=(1), view(W, i, first_t:T))
            throw(ArgumentError("synth_panel: treatment is not absorbing for unit " *
                                "$(repr(units[i])) (it switches back to 0). Only " *
                                "absorbing (staggered or block) adoption is supported."))
        end
        if first_t == 1
            throw(ArgumentError("synth_panel: unit $(repr(units[i])) is treated in " *
                                "the first period $(repr(times[1])); treated units " *
                                "need at least one pre-treatment period."))
        end
        adoption[i] = first_t
    end
    controls = findall(==(0), adoption)
    treated = findall(>(0), adoption)
    isempty(treated) && throw(ArgumentError("synth_panel: no treated units (treatment " *
                                            "is 0 everywhere)"))
    isempty(controls) && throw(ArgumentError("synth_panel: no never-treated units; " *
                                             "synthetic control estimators need a " *
                                             "donor pool of never-treated units"))
    order_c = controls[sortperm(units[controls])]
    order_t = treated[sortperm(collect(zip(adoption[treated], units[treated])))]
    ord = vcat(order_c, order_t)
    return SynthPanel(Matrix{Float64}(Y[ord, :]), collect(units[ord]), collect(times),
                      adoption[ord], length(controls),
                      Array{Union{Missing,Float64},3}(X[ord, :, :]),
                      collect(Symbol, covariates), outcome, treatment, unit, time)
end

_sc_n_units(p::SynthPanel) = size(p.Y, 1)
_sc_n_periods(p::SynthPanel) = size(p.Y, 2)
_sc_n_treated(p::SynthPanel) = size(p.Y, 1) - p.n_control
_sc_treated_rows(p::SynthPanel) = (p.n_control + 1):size(p.Y, 1)
_sc_adoption_indices(p::SynthPanel) = sort(unique(p.adoption[_sc_treated_rows(p)]))
_sc_is_block(p::SynthPanel) = length(_sc_adoption_indices(p)) == 1

function _sc_treatment_matrix(p::SynthPanel)
    N, T = size(p.Y)
    W = zeros(Int, N, T)
    for i in 1:N
        a = p.adoption[i]
        a > 0 && (W[i, a:T] .= 1)
    end
    return W
end

# Complete covariate array (Float64) or an informative error.
function _sc_complete_covariates(p::SynthPanel, context::AbstractString)
    if any(ismissing, p.X)
        k = findfirst(k -> any(ismissing, view(p.X, :, :, k)), 1:length(p.covariates))
        throw(ArgumentError("$context: covariate `$(p.covariates[k])` has missing " *
                            "values; this estimator needs complete time-varying " *
                            "covariates"))
    end
    return Array{Float64,3}(p.X)
end

# Sub-panel with a subset of rows and adoption pattern (used by placebo analyses).
function _sc_subpanel(p::SynthPanel, rows::AbstractVector{<:Integer},
                      adoption::AbstractVector{<:Integer}=p.adoption[rows];
                      periods::AbstractVector{<:Integer}=1:size(p.Y, 2))
    W = zeros(Int, length(rows), length(periods))
    for (r, a) in enumerate(adoption)
        if a > 0
            # adoption is a column index of the full panel; map to the sub-period grid
            pos = findfirst(>=(a), collect(periods))
            pos === nothing || (W[r, pos:end] .= 1)
        end
    end
    return _sc_panel_from_matrices(p.Y[rows, periods], W, p.units[rows],
                                   p.times[periods], p.X[rows, periods, :],
                                   p.covariates; outcome=p.outcome,
                                   treatment=p.treatment, unit=p.unit, time=p.time)
end

function Base.show(io::IO, ::MIME"text/plain", p::SynthPanel)
    N, T = size(p.Y)
    println(io, "SynthPanel: $N units × $T periods ($(p.times[1]) – $(p.times[end]))")
    println(io, "  outcome: ", p.outcome, ", treatment: ", p.treatment)
    println(io, "  never-treated units: ", p.n_control, ", treated units: ",
            N - p.n_control)
    ad = _sc_adoption_indices(p)
    println(io, "  adoption periods: ", join(string.(p.times[ad]), ", "),
            length(ad) > 1 ? " (staggered)" : " (block)")
    isempty(p.covariates) || println(io, "  covariates: ", join(p.covariates, ", "))
end

Base.show(io::IO, p::SynthPanel) =
    print(io, "SynthPanel(", size(p.Y, 1), " units × ", size(p.Y, 2), " periods)")
