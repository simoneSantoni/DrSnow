# Programmatic model-formula construction.
#
# DrSnow never builds formulas from strings. Column names reach StatsModels only as
# `Symbol`s wrapped in `term`, so arbitrary names (including spaces or punctuation)
# are safe and no user-controlled text is ever parsed or evaluated.

"""
    require_columns(data, cols; context="data")

Throw an `ArgumentError` unless every column in `cols` exists in `data`.

# Arguments
- `data`: a `DataFrame` (or any object supporting `propertynames`).
- `cols`: iterable of `Symbol`s; `nothing` entries are ignored.
- `context::String`: label used in the error message.

# Returns
- `nothing`

# Examples
```julia
require_columns(df, [:y, :d, :unit]; context="did_twfe")
```
"""
function require_columns(data, cols; context::AbstractString="data")
    available = Set(Symbol.(propertynames(data)))
    missing_cols = [c for c in cols if c !== nothing && !(Symbol(c) in available)]
    if !isempty(missing_cols)
        throw(ArgumentError("$(context): column(s) $(join(missing_cols, ", ")) " *
                            "not found in data"))
    end
    return nothing
end

_as_symbols(x::Nothing) = Symbol[]
_as_symbols(x::Symbol) = [x]
_as_symbols(x::AbstractString) = [Symbol(x)]
_as_symbols(x) = Symbol[Symbol(s) for s in x]

_sumterms(ts) = isempty(ts) ? nothing : reduce(+, ts)

"""
    make_formula(y, rhs=Symbol[]; fe=Symbol[], endogenous=Symbol[],
                 instruments=Symbol[], intercept=true) -> StatsModels.FormulaTerm

Build a model formula for `FixedEffectModels.reg` (or GLM) from column names,
without parsing strings.

The formula has the form `y ~ rhs + (endogenous ~ instruments) + fe(f₁) + …`: the
exogenous regressors in `rhs`, an instrumental-variables block when `endogenous`
and `instruments` are given, and one absorbed fixed effect per entry of `fe`. Column
names reach StatsModels only as `Symbol`s wrapped in `term`, so names containing
spaces or punctuation are safe and no user-supplied text is ever evaluated; this is
how every DrSnow estimator builds its regressions. Without fixed effects an
intercept is included unless `intercept=false`; with fixed effects the intercept is
absorbed. An empty right-hand side gives the intercept-only model.

The function only builds the formula. Identification of the coefficients of
interest (for example a treatment that is collinear with the fixed effects) is
checked after fitting, when coefficients are read by name.

# Arguments
- `y`: outcome column (`Symbol` or string).
- `rhs`: exogenous regressors: a `Symbol`, a vector of `Symbol`s, or `nothing`.

# Keywords
- `fe`: columns absorbed as fixed effects through `FixedEffectModels.fe`; default
  none.
- `endogenous`, `instruments`: endogenous regressors and their excluded
  instruments; the block `(endogenous ~ instruments)` is added to the right-hand
  side. Both must be empty or both non-empty, otherwise an `ArgumentError` is
  thrown.
- `intercept::Bool`: include an intercept when there are no fixed effects; default
  `true`.

# Returns
- `StatsModels.FormulaTerm`.

# Examples
```julia
using DrSnow
f = make_formula(:y, [:d, :x]; fe=[:unit, :time])      # y ~ d + x + fe(unit) + fe(time)
f_iv = make_formula(:y, [:x]; endogenous=[:d], instruments=[:z])
f_odd = make_formula(Symbol("log wage"), [Symbol("union member")])
```
"""
function make_formula(y, rhs=Symbol[]; fe=Symbol[], endogenous=Symbol[],
                      instruments=Symbol[], intercept::Bool=true)
    xs = _as_symbols(rhs)
    fes = _as_symbols(fe)
    endo = _as_symbols(endogenous)
    inst = _as_symbols(instruments)
    if isempty(endo) != isempty(inst)
        throw(ArgumentError("make_formula: `endogenous` and `instruments` must both " *
                            "be empty or both be non-empty"))
    end
    ts = Any[StatsModels.term(x) for x in xs]
    if !isempty(endo)
        push!(ts, _sumterms(StatsModels.term.(endo)) ~ _sumterms(StatsModels.term.(inst)))
    end
    for f in fes
        push!(ts, FixedEffectModels.fe(f))
    end
    if isempty(fes)
        pushfirst!(ts, StatsModels.ConstantTerm(intercept ? 1 : 0))
    end
    if isempty(ts)
        ts = Any[StatsModels.ConstantTerm(1)]
    end
    return StatsModels.term(Symbol(y)) ~ _sumterms(ts)
end

"""
    coef_index(model, name) -> Int

Position of coefficient `name` in a fitted `StatsAPI.RegressionModel`. Throws an
`ArgumentError` when the coefficient is absent, and an `ErrorException` when it was
dropped as collinear (zero estimate with non-finite or zero variance).
"""
function coef_index(model, name)
    nm = string(name)
    idx = findfirst(==(nm), StatsAPI.coefnames(model))
    idx === nothing && throw(ArgumentError("coefficient `$nm` not found in model"))
    v = StatsAPI.vcov(model)[idx, idx]
    if !isfinite(v) || (iszero(v) && iszero(StatsAPI.coef(model)[idx]))
        error("coefficient `$nm` is not identified (collinear with other regressors " *
              "or fixed effects)")
    end
    return idx
end
