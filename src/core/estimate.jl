# Common result interface for every DrSnow estimator.
#
# Every estimate type subtypes `CausalEstimate` and implements, at minimum:
#     StatsAPI.coef(r)        :: Vector{Float64}
#     StatsAPI.vcov(r)        :: Matrix{Float64}
#     StatsAPI.coefnames(r)   :: Vector{String}
#     StatsAPI.nobs(r)        :: Int
# and optionally:
#     StatsAPI.dof_residual(r)  (default `Inf` → normal critical values)
#     estimand(r)::String       (default "")
#     method_name(r)::String    (default: type name)
# Everything else (stderror, confint, pvalues, coeftable, show) is derived here.

"""
    CausalEstimate <: StatsAPI.StatisticalModel

Abstract supertype of every DrSnow estimation result: difference-in-differences,
instrumental-variable, regression-discontinuity, synthetic-control, design-based
interference and machine-learning estimators all return a subtype.

A `CausalEstimate` represents a vector of estimated causal parameters together with
the full estimated covariance matrix of the estimator and the reference distribution
used for inference. Concrete subtypes implement `coef`, `vcov`, `coefnames` and
`nobs`; they set `dof_residual` when a Student-t reference is appropriate (for
example ``G - 1`` with ``G`` clusters) and describe themselves through
[`estimand`](@ref) and [`method_name`](@ref). Everything else is derived once, in
the same way for every estimator: standard errors are the square roots of the
diagonal of `vcov`; `confint(r; level)` gives Wald intervals ``\\hat θ ± q ⋅ se(\\hat θ)``,
with ``q`` equal to [`critical_value`](@ref)`(level, dof_residual(r))`;
[`pvalues`](@ref) and [`tstats`](@ref) refer to the same distribution; and
`coeftable`, [`tidy`](@ref) and [`glance`](@ref) tabulate the result. Subtypes may
override `confint` (for example with simultaneous bands for event studies), in which
case `tidy` inherits the override. Because the full covariance is stored, joint
hypotheses are tested with [`wald_test`](@ref)`(coef(r), vcov(r); R)`.

The interface is the StatsAPI one, so results also work with packages that consume
`StatisticalModel`s, such as RegressionTables.jl (through a package extension), and
every result is a Tables.jl table of its coefficients.

# Accessors
- `coef`, `vcov`, `stderror`, `confint`, `coeftable`, `coefnames`, `nobs`,
  `dof_residual` (StatsAPI, re-exported);
- [`estimate`](@ref), [`estimand`](@ref), [`method_name`](@ref),
  [`pvalues`](@ref), [`tstats`](@ref), [`tidy`](@ref), [`glance`](@ref).

# Examples
```julia
using DrSnow, DataFrames, StableRNGs
rng = StableRNG(1)
df = DataFrame(unit=repeat(1:40; inner=6), year=repeat(1:6; outer=40))
df.d = Int.((df.unit .<= 20) .& (df.year .>= 4))
df.y = 0.2 .* df.year .+ 1.0 .* df.d .+ randn(rng, nrow(df))
r = did_twfe(df, :y, :d, :unit, :year)
r isa CausalEstimate          # true
coef(r), stderror(r), confint(r; level=0.9)
```
"""
abstract type CausalEstimate <: StatsAPI.StatisticalModel end

"""
    estimand(r::CausalEstimate) -> String

Short verbal description of the causal parameter targeted by `r`, such as `"ATT"`,
`"LATE"` or `"average contrasts between exposure conditions"`.

The estimand is printed in the header of every result and returned by
[`glance`](@ref). It names the target of the estimator under its identifying
assumptions; it does not certify that those assumptions hold. Estimators that do not
set it return the empty string.

# Arguments
- `r::CausalEstimate`: any DrSnow estimation result.

# Returns
- `String` (possibly empty).

# Examples
```julia
using DrSnow, DataFrames, Random, StableRNGs
rng = StableRNG(2)
df = DataFrame(v=repeat(1:8; inner=10), sat=repeat([0.2, 0.8]; inner=40))
df.z = reduce(vcat, [shuffle(rng, (1:10) .<= round(Int, 10s)) for s in df.sat[1:10:end]])
df.y = 0.5 .* df.z .+ 0.3 .* df.sat .+ randn(rng, nrow(df))
estimand(two_stage_effects(df, :y, :z; group=:v, saturation=:sat))
```
"""
estimand(::CausalEstimate) = ""

"""
    method_name(r::CausalEstimate) -> String

Name of the estimation method that produced `r` (for example
`"Hájek exposure-contrast estimator (Aronow–Samii)"`), used in printed output, as
the default model label of [`tidy`](@ref) for vectors of results, and in
[`glance`](@ref). Types that do not override it return their type name.

# Arguments
- `r::CausalEstimate`: any DrSnow estimation result.

# Returns
- `String`.

# Examples
```julia
using DrSnow, DataFrames, Random, StableRNGs
rng = StableRNG(2)
df = DataFrame(v=repeat(1:8; inner=10), sat=repeat([0.2, 0.8]; inner=40))
df.z = reduce(vcat, [shuffle(rng, (1:10) .<= round(Int, 10s)) for s in df.sat[1:10:end]])
df.y = 0.5 .* df.z .+ randn(rng, nrow(df))
method_name(two_stage_effects(df, :y, :z; group=:v, saturation=:sat))
```
"""
method_name(r::CausalEstimate) = string(nameof(typeof(r)))

StatsAPI.dof_residual(::CausalEstimate) = Inf
StatsAPI.stderror(r::CausalEstimate) = sqrt.(max.(diag(StatsAPI.vcov(r)), 0.0))

"""
    tstats(r::CausalEstimate) -> Vector{Float64}

Ratios ``\\hat θ_j / se(\\hat θ_j)`` of every coefficient to its standard error, the
statistics of the individual tests of ``H_0: θ_j = 0``.

They are referred to a t distribution with `dof_residual(r)` degrees of freedom when
that is finite and to the standard normal otherwise (see [`pvalues`](@ref)). A zero
standard error yields `±Inf` or `NaN`, which [`two_sided_pvalue`](@ref) maps to
`NaN`.

# Arguments
- `r::CausalEstimate`: any DrSnow estimation result.

# Returns
- `Vector{Float64}`, one entry per coefficient, in the order of `coefnames(r)`.

# Examples
```julia
using DrSnow, DataFrames, StableRNGs
rng = StableRNG(1)
df = DataFrame(unit=repeat(1:40; inner=6), year=repeat(1:6; outer=40))
df.d = Int.((df.unit .<= 20) .& (df.year .>= 4))
df.y = 0.2 .* df.year .+ 1.0 .* df.d .+ randn(rng, nrow(df))
tstats(did_twfe(df, :y, :d, :unit, :year))
```
"""
tstats(r::CausalEstimate) = StatsAPI.coef(r) ./ StatsAPI.stderror(r)

"""
    pvalues(r::CausalEstimate) -> Vector{Float64}

Two-sided p-values of the individual Wald tests ``H_0: θ_j = 0``, one per
coefficient, computed from [`tstats`](@ref) with a Student-t reference with
`dof_residual(r)` degrees of freedom when that is finite and the standard normal
otherwise.

These are marginal p-values: they do not account for multiplicity. For joint
hypotheses use [`wald_test`](@ref) with the full covariance matrix; for families of
separate hypotheses adjust with [`holm_adjust`](@ref), [`bh_adjust`](@ref) or, in
randomized experiments, [`ri_multiple_testing`](@ref). Some result types override
this method (for example, [`RIRegressionResult`](@ref) returns randomization-t
p-values).

# Arguments
- `r::CausalEstimate`: any DrSnow estimation result.

# Returns
- `Vector{Float64}` in the order of `coefnames(r)`.

# Examples
```julia
using DrSnow, DataFrames, StableRNGs
rng = StableRNG(1)
df = DataFrame(unit=repeat(1:40; inner=6), year=repeat(1:6; outer=40))
df.d = Int.((df.unit .<= 20) .& (df.year .>= 4))
df.y = 0.2 .* df.year .+ 1.0 .* df.d .+ randn(rng, nrow(df))
pvalues(did_twfe(df, :y, :d, :unit, :year))
```
"""
pvalues(r::CausalEstimate) = two_sided_pvalue.(tstats(r), StatsAPI.dof_residual(r))

function StatsAPI.confint(r::CausalEstimate; level::Real=0.95)
    c = critical_value(level, StatsAPI.dof_residual(r))
    b = StatsAPI.coef(r)
    s = StatsAPI.stderror(r)
    return hcat(b .- c .* s, b .+ c .* s)
end

function StatsAPI.coeftable(r::CausalEstimate; level::Real=0.95)
    ci = StatsAPI.confint(r; level=level)
    lv = round(Int, 100 * level)
    return StatsBase.CoefTable(
        hcat(StatsAPI.coef(r), StatsAPI.stderror(r), tstats(r), pvalues(r), ci),
        ["Estimate", "Std. Error", "t", "Pr(>|t|)", "Lower $lv%", "Upper $lv%"],
        StatsAPI.coefnames(r), 4, 3)
end

"""
    estimate(r::CausalEstimate) -> Float64

Headline scalar estimate of a result: by default its first coefficient. Result types
whose headline parameter is not the first coefficient (for example an aggregated
effect stored after group-time effects) override this method.

# Arguments
- `r::CausalEstimate`: any DrSnow estimation result.

# Returns
- `Float64`.

# Examples
```julia
using DrSnow, DataFrames, StableRNGs
rng = StableRNG(1)
df = DataFrame(unit=repeat(1:40; inner=6), year=repeat(1:6; outer=40))
df.d = Int.((df.unit .<= 20) .& (df.year .>= 4))
df.y = 0.2 .* df.year .+ 1.0 .* df.d .+ randn(rng, nrow(df))
estimate(did_twfe(df, :y, :d, :unit, :year))
```
"""
estimate(r::CausalEstimate) = first(StatsAPI.coef(r))

"""
    show_details(io, r::CausalEstimate)

Hook for estimator-specific lines printed after the coefficient table.
"""
show_details(::IO, ::CausalEstimate) = nothing

function Base.show(io::IO, ::MIME"text/plain", r::CausalEstimate)
    header = method_name(r)
    e = estimand(r)
    isempty(e) || (header *= " — estimand: " * e)
    println(io, header)
    println(io, "Observations: ", StatsAPI.nobs(r))
    show(io, MIME"text/plain"(), StatsAPI.coeftable(r))
    show_details(io, r)
end

function Base.show(io::IO, r::CausalEstimate)
    b = StatsAPI.coef(r)
    s = StatsAPI.stderror(r)
    print(io, method_name(r), "(", length(b) == 1 ?
          @sprintf("%.4g (se %.3g)", b[1], s[1]) : "$(length(b)) coefficients", ")")
end

"""
    DiagnosticTest

Package-wide result type of diagnostic, falsification and specification tests
(pre-trend tests, balance tests, randomization tests of design assumptions, tests of
no spillovers, and so on).

A `DiagnosticTest` records the null hypothesis in words, the statistic, its degrees
of freedom, the p-value, the method used to obtain it, and any caveat on
interpretation. The wording rules of the package are built into the type: printing
reports a rejection or a non-rejection at the 5% level and never states that an
identifying assumption "holds", because a non-rejection is not evidence that the
null is true (it may reflect low power), and because many identifying assumptions
are untestable, so that a diagnostic can at most test an implication of them. An
undefined test is reported with `pvalue = NaN`, and [`rejects`](@ref) then throws
rather than returning `false`.

The convenience constructor `DiagnosticTest(name, null, statistic, pvalue; dof=(),
method="", note="", details=NamedTuple())` converts its arguments to the field
types.

# Fields
- `name::String`: name of the test.
- `null::String`: the null hypothesis, in words.
- `statistic::Float64`: value of the test statistic.
- `dof::Tuple`: its degrees of freedom (empty when not applicable, e.g. for
  randomization tests).
- `pvalue::Float64`: the p-value; `NaN` when the test could not be computed.
- `method::String`: how the p-value was obtained, e.g. `"cluster-robust Wald"` or
  `"randomization inference, Monte Carlo with 1999 draws"`.
- `note::String`: caveats (power, what a non-rejection does not imply, the
  assignment mechanism assumed).
- `details::NamedTuple`: test-specific output (per-covariate tables, reference
  distributions, draws).

# Accessors
- `pvalue(t)`, [`rejects`](@ref)`(t; alpha)`, [`tidy`](@ref)`(t)`; the type is also
  a one-row Tables.jl table.

# Examples
```julia
using DrSnow
t = DiagnosticTest("Example test", "the coefficient is zero", 2.1, 0.036;
                   dof=(1, 40), method="Wald F test")
pvalue(t), rejects(t; alpha=0.05)
```
"""
struct DiagnosticTest
    name::String
    null::String
    statistic::Float64
    dof::Tuple
    pvalue::Float64
    method::String
    note::String
    details::NamedTuple
end

function DiagnosticTest(name, null, statistic, pvalue; dof=(), method="", note="",
                        details=NamedTuple())
    return DiagnosticTest(string(name), string(null), float(statistic), Tuple(dof),
                          float(pvalue), string(method), string(note), details)
end

StatsAPI.pvalue(t::DiagnosticTest) = t.pvalue

"""
    rejects(t::DiagnosticTest; alpha=0.05) -> Bool

Whether the null hypothesis of `t` is rejected at level `alpha`, i.e. whether
`pvalue(t) < alpha`.

If the test could not be computed (`pvalue` is `NaN`) the function throws an error
instead of returning `false`, so that an undefined test is never mistaken for a
passed one. A `false` result means only that the data are compatible with the null
at this level; it is not evidence that the null (or an identifying assumption it is
meant to probe) is true.

# Arguments
- `t::DiagnosticTest`: a diagnostic test result.

# Keywords
- `alpha::Real`: significance level; default 0.05.

# Returns
- `Bool`.

# Examples
```julia
using DrSnow
t = DiagnosticTest("Example test", "no effect", 1.2, 0.23)
rejects(t)               # false: not rejected at 5%
rejects(t; alpha=0.25)   # true
```
"""
function rejects(t::DiagnosticTest; alpha::Real=0.05)
    isnan(t.pvalue) && error("$(t.name): p-value is undefined; the test could not be " *
                             "computed (see `note`)")
    return t.pvalue < alpha
end

function Base.show(io::IO, ::MIME"text/plain", t::DiagnosticTest)
    println(io, t.name)
    println(io, "H₀: ", t.null)
    isempty(t.method) || println(io, "Method: ", t.method)
    dofs = isempty(t.dof) ? "" : "(" * join(t.dof, ", ") * ")"
    @printf(io, "Statistic%s = %.4f, p-value = %.4g\n", dofs, t.statistic, t.pvalue)
    if isnan(t.pvalue)
        println(io, "Result: not computable.")
    elseif t.pvalue < 0.05
        println(io, "Result: H₀ rejected at the 5% level.")
    else
        println(io, "Result: H₀ not rejected at the 5% level. Non-rejection is not " *
                    "evidence that H₀ is true.")
    end
    isempty(t.note) || println(io, "Note: ", t.note)
end

Base.show(io::IO, t::DiagnosticTest) =
    @printf(io, "%s: stat = %.4g, p = %.4g", t.name, t.statistic, t.pvalue)

# ---------------------------------------------------------------------------
# Tidy results: `tidy`, `glance` and the Tables.jl interface
# ---------------------------------------------------------------------------

"""
    tidy(r::CausalEstimate; level=0.95, kwargs...) -> DataFrame
    tidy(rs::AbstractVector; level=0.95, names=nothing, kwargs...) -> DataFrame
    tidy(t::DiagnosticTest) -> DataFrame

Coefficient table of an estimation result as a data frame with one row per
coefficient, in the layout of R's `broom::tidy`.

The table follows the "tidy data" convention of one observation (here, one
coefficient) per row and one variable per column (Wickham 2014), so that results from
different estimators can be stacked, filtered, plotted and exported with ordinary
table operations. The columns are `term` (the coefficient name), `estimate`,
`std_error`, `statistic` (the ratio `estimate / std_error`), `p_value`, and the
confidence limits `conf_low` and `conf_high`.

The inference columns are the estimator's own, not recomputed. `p_value` is
[`pvalues`](@ref)`(r)` for ``H_0: \\theta = 0``: by default two-sided with a
t(`dof_residual`) reference distribution when the residual degrees of freedom are
finite (e.g. ``G - 1`` with ``G`` clusters) and the normal otherwise, but results
with a different natural p-value report that one (for example always-valid
p-values for confidence sequences, repeated or stagewise-ordering p-values for
group-sequential analyses, and bias-aware p-values for honest RD estimates), in
which case `statistic` is only a descriptive ratio. The
limits come from `confint(r; level, kwargs...)`, so estimator-specific interval
options are honoured: `tidy(es; uniform=true)` gives simultaneous sup-t bands for an
[`EventStudyEstimate`](@ref), while `p_value` stays pointwise. Results that carry no
variance (e.g. a synthetic control fitted without placebos, whose `vcov` throws) get
`missing` in every inference column rather than a spurious zero.

With a vector of results the tables are stacked with a leading `model` column
(`names`, or [`method_name`](@ref) of each result by default); columns absent from
some results are filled with `missing`. Stacked rows share a layout, not an
estimand: check [`estimand`](@ref) before comparing them. For a
[`DiagnosticTest`](@ref) the single row holds `test`, `null`, `statistic`, `dof`
(comma-separated when there are several), `p_value` and `method`.

Every `CausalEstimate` and `DiagnosticTest` is also a Tables.jl table with these
columns (at the default level 0.95), so `DataFrame(r)` and `CSV.write(file, r)`
work directly. See [`glance`](@ref) for the one-row model summary and
[`plot_coefficients`](@ref) for the corresponding plot.

# Arguments
- `r::CausalEstimate`: any DrSnow estimation result.
- `rs::AbstractVector`: a vector of `CausalEstimate`s to stack.
- `t::DiagnosticTest`: a diagnostic or falsification test.

# Keywords
- `level::Real = 0.95`: confidence level of `conf_low` / `conf_high`, in ``(0, 1)``.
- `names = nothing`: model labels, one per result in `rs` (default: the
  [`method_name`](@ref) of each result).
- `kwargs...`: further keywords forwarded to `confint` (e.g. `uniform = true` and
  `rng` for event studies).

# Returns
- `DataFrame` with one row per coefficient (and a leading `model` column for a
  vector of results), or one row for a `DiagnosticTest`.

# Examples
```julia
using DrSnow, DataFrames, StableRNGs
rng = StableRNG(1)
g = rand(rng, [0, 4, 6], 150)
df = DataFrame(unit=repeat(1:150; inner=8), year=repeat(1:8, 150))
df.d = Int.((g[df.unit] .> 0) .& (df.year .>= g[df.unit]))
df.y = randn(rng, 150)[df.unit] .+ 0.2 .* df.year .+ df.d .+ randn(rng, nrow(df))
r1 = did_twfe(df, :y, :d, :unit, :year; warn_heterogeneity=false)
r2 = did_imputation(df, :y, :d, :unit, :year)
tidy(r1)
tidy([r1, r2]; names=["TWFE", "Imputation"])
es = did_sun_abraham(df, :y, :d, :unit, :year)
tidy(es; level=0.9, uniform=true, rng=StableRNG(2))    # simultaneous bands
```

# References
- Wickham, H. (2014). Tidy data. *Journal of Statistical Software*, 59(10), 1–23.
"""
function tidy(r::CausalEstimate; level::Real=0.95, kwargs...)
    (0 < level < 1) || throw(ArgumentError("level must be in (0, 1), got $level"))
    b = Vector{Float64}(StatsAPI.coef(r))
    terms = String.(StatsAPI.coefnames(r))
    V = _tidy_vcov(r)
    k = length(b)
    if V === nothing
        m = fill(missing, k)
        return DataFrame(term=terms, estimate=b,
                         std_error=Vector{Union{Missing,Float64}}(m),
                         statistic=Vector{Union{Missing,Float64}}(m),
                         p_value=Vector{Union{Missing,Float64}}(m),
                         conf_low=Vector{Union{Missing,Float64}}(m),
                         conf_high=Vector{Union{Missing,Float64}}(m))
    end
    se = StatsAPI.stderror(r)
    ci = StatsAPI.confint(r; level=level, kwargs...)
    ci = ci isa Tuple ? reshape(collect(Float64, ci), 1, 2) : ci
    return DataFrame(term=terms, estimate=b, std_error=se, statistic=b ./ se,
                     p_value=pvalues(r), conf_low=ci[:, 1], conf_high=ci[:, 2])
end

# `vcov` of results without a variance (synthetic-control types fitted without
# placebos / replications) throws an ArgumentError; tidy reports missing inference.
function _tidy_vcov(r::CausalEstimate)
    try
        return StatsAPI.vcov(r)
    catch err
        err isa ArgumentError && return nothing
        rethrow()
    end
end

function tidy(rs::AbstractVector; level::Real=0.95, names=nothing, kwargs...)
    all(r -> r isa CausalEstimate, rs) ||
        throw(ArgumentError("tidy: every element must be a CausalEstimate"))
    labels = names === nothing ? [method_name(r) for r in rs] : string.(collect(names))
    length(labels) == length(rs) ||
        throw(DimensionMismatch("tidy: need one name per result"))
    frames = map(zip(rs, labels)) do (r, lbl)
        t = tidy(r; level=level, kwargs...)
        insertcols!(t, 1, :model => fill(lbl, nrow(t)))
    end
    return isempty(frames) ? DataFrame() : vcat(frames...; cols=:union)
end

function tidy(t::DiagnosticTest)
    return DataFrame(test=[t.name], null=[t.null], statistic=[t.statistic],
                     dof=[isempty(t.dof) ? "" : join(string.(t.dof), ", ")],
                     p_value=[t.pvalue], method=[t.method])
end

"""
    glance(r::CausalEstimate) -> DataFrame

One-row summary of an estimation result, in the layout of R's `broom::glance`: the
method, its estimand and the sample information that governs inference.

The columns are `method` ([`method_name`](@ref)), `estimand` ([`estimand`](@ref)),
`nobs` (observations used), `n_coef` (number of reported coefficients),
`dof_residual` (the degrees of freedom of the t reference distribution used for
p-values and intervals; `Inf` when inference uses the normal distribution) and
`n_clusters` (the number of clusters used by the variance estimator, summed over the
two sides of the cutoff for RD; `missing` when the result does not record
clustering). With clustered standard errors the effective sample size for inference
is the number of clusters rather than `nobs`, and cluster-robust inference with few
clusters is unreliable, which is why `n_clusters` is reported next to `nobs`.

Rows from several results can be stacked with `vcat` to document a set of
specifications, for instance next to the coefficients from [`tidy`](@ref).

# Arguments
- `r::CausalEstimate`: any DrSnow estimation result.

# Returns
- `DataFrame` with one row; stack several with `vcat(glance.(rs)...)`.

# Examples
```julia
using DrSnow, DataFrames, StableRNGs
rng = StableRNG(1)
g = rand(rng, [0, 4, 6], 150)
df = DataFrame(unit=repeat(1:150; inner=8), year=repeat(1:8, 150))
df.d = Int.((g[df.unit] .> 0) .& (df.year .>= g[df.unit]))
df.y = randn(rng, 150)[df.unit] .+ 0.2 .* df.year .+ df.d .+ randn(rng, nrow(df))
r1 = did_twfe(df, :y, :d, :unit, :year; warn_heterogeneity=false)
r2 = did_imputation(df, :y, :d, :unit, :year)
vcat(glance(r1), glance(r2))
```

# References
- Cameron, A. C., & Miller, D. L. (2015). A practitioner's guide to cluster-robust
  inference. *Journal of Human Resources*, 50(2), 317–372.
"""
function glance(r::CausalEstimate)
    return DataFrame(method=[method_name(r)], estimand=[estimand(r)],
                     nobs=[StatsAPI.nobs(r)], n_coef=[length(StatsAPI.coef(r))],
                     dof_residual=[float(StatsAPI.dof_residual(r))],
                     n_clusters=Union{Missing,Int}[_glance_nclusters(r)])
end

# Number of clusters: results store it in an `n_clusters` field (an `Int`, or a
# `(left, right)` tuple for RD); areas whose result types store it elsewhere add a
# method (see src/viz/data.jl).
function _glance_nclusters(r)
    hasproperty(r, :n_clusters) || return missing
    n = getproperty(r, :n_clusters)
    n isa Tuple && (n = sum(n))
    return n isa Integer && n > 0 ? Int(n) : missing
end

# Tables.jl: every result is a table of its coefficients (the `tidy` columns).
Tables.istable(::Type{<:CausalEstimate}) = true
Tables.columnaccess(::Type{<:CausalEstimate}) = true
Tables.columns(r::CausalEstimate) = Tables.columns(tidy(r))
Tables.istable(::Type{DiagnosticTest}) = true
Tables.columnaccess(::Type{DiagnosticTest}) = true
Tables.columns(t::DiagnosticTest) = Tables.columns(tidy(t))
