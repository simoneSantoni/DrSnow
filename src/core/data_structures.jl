# Core panel data structure shared by the panel estimators.
#
# Result types live with their estimators (e.g. `DiDEstimate` in `src/did/`).

"""
    TreatmentPanel(data, outcome, treatment, unit_id, time, covariates=Symbol[])
    TreatmentPanel(data; outcome, treatment, unit_id, time, covariates=Symbol[])

Panel data with a binary treatment indicator, bundling the roles of the columns so
that panel estimators can be called as `did_twfe(panel)` or `event_study(panel)`.

The object stores a long-format panel (one row per unit and period) of outcomes
``Y_{it}``, a treatment indicator ``D_{it} \\in \\{0, 1\\}`` and optional covariates
``X_{it}``. It makes no assumption about the design: the constructor only checks that
the named columns exist. Content checks (binary and time-varying treatment,
duplicates, balance, absorbing treatment) are done by [`validate_panel`](@ref), and
[`preprocess_panel`](@ref) drops incomplete rows, sorts and validates in one step.
Methods that accept a `TreatmentPanel` pass `covariates` on as their `covariates`
keyword.

# Fields
- `data::DataFrame`: the panel in long format.
- `outcome::Symbol`: outcome column ``Y_{it}``.
- `treatment::Symbol`: 0/1 treatment indicator ``D_{it}``.
- `unit_id::Symbol`: unit identifier column.
- `time::Symbol`: time-period column (any sortable type).
- `covariates::Vector{Symbol}`: covariate columns (may be empty).

# Examples
```julia
using DrSnow, CSV, DataFrames
file = joinpath(pkgdir(DrSnow), "test", "validation", "did", "mpdta.csv")
mpdta = CSV.read(file, DataFrame)
panel = TreatmentPanel(mpdta, :lemp, :d, :countyreal, :year, [:lpop])
panel = TreatmentPanel(mpdta; outcome=:lemp, treatment=:d, unit_id=:countyreal,
                       time=:year)
did_twfe(panel; warn_heterogeneity=false)
```
"""
struct TreatmentPanel
    data::DataFrame
    outcome::Symbol
    treatment::Symbol
    unit_id::Symbol
    time::Symbol
    covariates::Vector{Symbol}

    function TreatmentPanel(data, outcome, treatment, unit_id, time,
                            covariates=Symbol[])
        df = data isa DataFrame ? data : DataFrame(data)
        covs = Symbol[Symbol(c) for c in covariates]
        require_columns(df, [outcome, treatment, unit_id, time]; context="TreatmentPanel")
        require_columns(df, covs; context="TreatmentPanel covariates")
        return new(df, Symbol(outcome), Symbol(treatment), Symbol(unit_id), Symbol(time),
                   covs)
    end
end

TreatmentPanel(data; outcome, treatment, unit_id, time, covariates=Symbol[]) =
    TreatmentPanel(data, outcome, treatment, unit_id, time, covariates)

function Base.show(io::IO, p::TreatmentPanel)
    print(io, "TreatmentPanel(", nrow(p.data), " rows; outcome=", p.outcome,
          ", treatment=", p.treatment, ", unit=", p.unit_id, ", time=", p.time)
    isempty(p.covariates) || print(io, ", covariates=", p.covariates)
    print(io, ")")
end

"""
    validate_panel(panel::TreatmentPanel) -> Vector{String}

Check a treatment panel for problems that invalidate or change the interpretation of
panel treatment-effect estimators, and describe each problem found.

The checks are: missing values in the used columns; a treatment that is not binary;
a treatment that does not vary (all zero or all one), so that no effect is
identified; duplicated unit–period rows; an unbalanced panel; and units that leave
treatment after being treated. The last condition (non-absorbing treatment) is
outside the scope of the staggered-adoption estimators such as
[`did_callaway_santanna`](@ref) and [`did_sun_abraham`](@ref), which define cohorts
by the first treated period; [`did_multiplegt_dyn`](@ref) handles such designs.
An empty result means that none of these mechanical problems is present; it says
nothing about the identifying assumptions (parallel trends, no anticipation).

# Arguments
- `panel::TreatmentPanel`: the panel to check.

# Returns
- `Vector{String}`: one message per problem found; empty when none.

# Examples
```julia
using DrSnow, CSV, DataFrames
file = joinpath(pkgdir(DrSnow), "test", "validation", "did", "mpdta.csv")
panel = TreatmentPanel(CSV.read(file, DataFrame), :lemp, :d, :countyreal, :year)
issues = validate_panel(panel)
isempty(issues) || foreach(println, issues)
```
"""
function validate_panel(panel::TreatmentPanel)
    issues = String[]
    df = panel.data
    cols = unique([panel.outcome, panel.treatment, panel.unit_id, panel.time,
                   panel.covariates...])
    for col in cols
        nmiss = count(ismissing, df[!, col])
        nmiss > 0 && push!(issues, "Missing values found in $col ($nmiss rows)")
    end
    dvals = unique(skipmissing(df[!, panel.treatment]))
    if !all(v -> v == 0 || v == 1, dvals)
        push!(issues, "Treatment variable should be binary (0/1), found: " *
                      join(repr.(sort(dvals; by=string)), ", "))
    elseif !isempty(dvals) && length(dvals) == 1
        push!(issues, "Treatment does not vary (all observations have treatment = " *
                      "$(first(dvals))); treatment effects are not identified")
    end
    keycols = [panel.unit_id, panel.time]
    ok = completecases(df, keycols)
    keys_ = df[ok, keycols]
    ndup = nrow(keys_) - nrow(unique(keys_))
    ndup > 0 && push!(issues, "Duplicated unit–period rows: $ndup")
    counts = combine(groupby(keys_, panel.unit_id), nrow => :n)
    nper = length(unique(keys_[!, panel.time]))
    if any(!=(nper), counts.n)
        push!(issues, "Unbalanced panel detected (units have different numbers of " *
                      "observations)")
    end
    if ndup == 0 && all(v -> v == 0 || v == 1, dvals)
        sub = df[ok .& .!ismissing.(df[!, panel.treatment]),
                 [panel.unit_id, panel.time, panel.treatment]]
        sort!(sub, keycols)
        leavers = 0
        for g in groupby(sub, panel.unit_id)
            d = g[!, panel.treatment]
            f = findfirst(==(1), d)
            f !== nothing && any(==(0), d[f:end]) && (leavers += 1)
        end
        leavers > 0 && push!(issues, "Non-absorbing treatment: $leavers unit(s) leave " *
                                     "treatment after being treated")
    end
    return issues
end

"""
    preprocess_panel(data, outcome, treatment, unit_id, time;
                     covariates=Symbol[], drop_missing=true) -> TreatmentPanel

Prepare raw panel data for estimation: copy it, optionally drop rows with missing
values in the used columns, sort by unit and time, and return a validated
[`TreatmentPanel`](@ref).

Problems found by [`validate_panel`](@ref) are reported with a single `@warn`, not
thrown, so that the caller can decide whether they matter for the intended
estimator. Dropping incomplete rows can unbalance a panel; estimators that require
balance (e.g. [`bacon_decomposition`](@ref), [`did_callaway_santanna`](@ref) for
panels) then drop or reject the affected units themselves.

# Arguments
- `data`: raw panel data (a `DataFrame` or any Tables.jl source); it is copied, not
  modified.
- `outcome::Symbol`, `treatment::Symbol`, `unit_id::Symbol`, `time::Symbol`: column
  names of the outcome, the 0/1 treatment indicator, the unit identifier and the
  time period.

# Keywords
- `covariates::Vector{Symbol} = Symbol[]`: covariate columns to keep in the panel.
- `drop_missing::Bool = true`: drop rows with a missing value in any used column.

# Returns
- `TreatmentPanel`: the cleaned, sorted panel.

# Examples
```julia
using DrSnow, CSV, DataFrames
file = joinpath(pkgdir(DrSnow), "test", "validation", "did", "mpdta.csv")
panel = preprocess_panel(CSV.read(file, DataFrame), :lemp, :d, :countyreal, :year;
                         covariates=[:lpop])
```
"""
function preprocess_panel(data, outcome::Symbol, treatment::Symbol, unit_id::Symbol,
                          time::Symbol; covariates::Vector{Symbol}=Symbol[],
                          drop_missing::Bool=true)
    processed = DataFrame(data; copycols=true)
    cols = unique([outcome, treatment, unit_id, time, covariates...])
    require_columns(processed, cols; context="preprocess_panel")
    drop_missing && (processed = dropmissing(processed, cols))
    sort!(processed, [unit_id, time])
    panel = TreatmentPanel(processed, outcome, treatment, unit_id, time, covariates)
    issues = validate_panel(panel)
    isempty(issues) || @warn "Panel validation issues:\n" * join(issues, "\n")
    return panel
end
