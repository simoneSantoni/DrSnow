# RegressionTables.jl support: `regtable(r1, r2, ...)` accepts any DrSnow
# `CausalEstimate` (alone or mixed with ordinary `RegressionModel`s such as
# FixedEffectModels / GLM fits).
#
# RegressionTables dispatches on `StatsAPI.RegressionModel`, whereas DrSnow results
# subtype `StatsAPI.StatisticalModel` (many are not regressions: synthetic control,
# RD, DML, ...). Each result is therefore wrapped in `_RTResult <: RegressionModel`,
# which forwards the StatsAPI accessors and uses DrSnow's own p-values and intervals
# (t(`dof_residual`) when finite, normal otherwise).
module DrSnowRegressionTablesExt

using DrSnow
using DrSnow: CausalEstimate, StatsAPI
import RegressionTables

struct _RTResult{T<:CausalEstimate} <: StatsAPI.RegressionModel
    r::T
end

_rt_wrap(x::StatsAPI.RegressionModel) = x
_rt_wrap(x::CausalEstimate) = _RTResult(x)

const _RTArg = Union{CausalEstimate,StatsAPI.RegressionModel}

# One method per position of the first DrSnow result (among the first three columns),
# so that every method has an argument of a DrSnow type (no type piracy); tables whose
# first three columns are all foreign models should wrap DrSnow results explicitly.
RegressionTables.regtable(r::CausalEstimate, rrs::_RTArg...; kwargs...) =
    RegressionTables.regtable(map(_rt_wrap, (r, rrs...))...; kwargs...)
RegressionTables.regtable(m1::StatsAPI.RegressionModel, r::CausalEstimate, rrs::_RTArg...;
                          kwargs...) =
    RegressionTables.regtable(map(_rt_wrap, (m1, r, rrs...))...; kwargs...)
RegressionTables.regtable(m1::StatsAPI.RegressionModel, m2::StatsAPI.RegressionModel,
                          r::CausalEstimate, rrs::_RTArg...; kwargs...) =
    RegressionTables.regtable(map(_rt_wrap, (m1, m2, r, rrs...))...; kwargs...)

StatsAPI.coef(x::_RTResult) = StatsAPI.coef(x.r)
StatsAPI.coefnames(x::_RTResult) = String.(StatsAPI.coefnames(x.r))
StatsAPI.vcov(x::_RTResult) = StatsAPI.vcov(x.r)
StatsAPI.stderror(x::_RTResult) = StatsAPI.stderror(x.r)
StatsAPI.nobs(x::_RTResult) = StatsAPI.nobs(x.r)
StatsAPI.dof_residual(x::_RTResult) = StatsAPI.dof_residual(x.r)
StatsAPI.confint(x::_RTResult; level::Real=0.95) = StatsAPI.confint(x.r; level=level)
StatsAPI.islinear(::_RTResult) = true
StatsAPI.responsename(x::_RTResult) = _rt_responsename(x.r)

# Dependent-variable row: the outcome column when the result records it (directly,
# in its panel, or in a fitted regression stored in `details.model`), otherwise a
# short form of the estimand.
function _rt_responsename(r)
    if hasproperty(r, :outcome) && getproperty(r, :outcome) isa Symbol
        return string(getproperty(r, :outcome))
    end
    if hasproperty(r, :panel) && hasproperty(getproperty(r, :panel), :outcome)
        return string(getproperty(r, :panel).outcome)
    end
    if hasproperty(r, :details) && getproperty(r, :details) isa NamedTuple &&
       haskey(r.details, :model) && r.details.model isa StatsAPI.RegressionModel
        try
            return string(StatsAPI.responsename(r.details.model))
        catch
        end
    end
    e = DrSnow.estimand(r)
    isempty(e) && return DrSnow.method_name(r)
    short = strip(first(split(e, r" \(|: "; limit=2)))
    return length(short) <= 40 ? String(short) : first(short, 39) * "…"
end

RegressionTables._pvalue(x::_RTResult) = DrSnow.pvalues(x.r)
RegressionTables.RegressionType(x::_RTResult) =
    RegressionTables.RegressionType(DrSnow.method_name(x.r))
RegressionTables.default_regression_statistics(::_RTResult) = [RegressionTables.Nobs]
RegressionTables.can_standardize(::_RTResult) = false

# Kleibergen–Paap first-stage F for IV results (as RegressionTables prints for
# FixedEffectModels IV fits).
RegressionTables.FStatIV(x::_RTResult{<:DrSnow.IVEstimate}) =
    RegressionTables.FStatIV(x.r.first_stage.kleibergen_paap_F)
RegressionTables.default_regression_statistics(::_RTResult{<:DrSnow.IVEstimate}) =
    [RegressionTables.Nobs, RegressionTables.FStatIV]

end # module DrSnowRegressionTablesExt
