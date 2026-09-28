# Judge / examiner designs: leave-one-out leniency instruments, 2SLS / UJIVE
# estimation, balance (randomization) checks, the Frandsen–Lefgren–Leslie (2023)
# joint test of exclusion and average monotonicity, and subsample monotonicity
# checks (Dobbie, Goldin & Yang 2018; Bhuller, Dahl, Løken & Mogstad 2020).

"""Complete-case sample for the judge functions; drops judges with a single case."""
function _iv_judge_sample(data::AbstractDataFrame, cols::Vector{Symbol}, judge::Symbol,
                          context::String; rows=nothing)
    require_columns(data, cols; context=context)
    mask = trues(nrow(data))
    rows === nothing || (mask .&= rows)
    for c in cols
        mask .&= .!ismissing.(data[!, c])
    end
    idx = findall(mask)
    cnt = Dict{Any,Int}()
    for i in idx
        cnt[data[i, judge]] = get(cnt, data[i, judge], 0) + 1
    end
    single = count(==(1), values(cnt))
    if single > 0
        @warn "$context: dropping $single judge(s) with a single case (leave-one-out " *
              "leniency is undefined)"
        idx = [i for i in idx if cnt[data[i, judge]] > 1]
    end
    length(unique(data[idx, judge])) >= 2 ||
        throw(ArgumentError("$context: need at least two judges with two or more cases"))
    return idx
end

"""Treatment residualized on strata fixed effects and covariates (case level)."""
function _iv_residualize(sub::AbstractDataFrame, v::Vector{Float64},
                         strata::Vector{Symbol}, covariates::Vector{Symbol})
    isempty(strata) && isempty(covariates) && return v .- mean(v)
    ctrl, _ = _iv_ctrl_basis(sub, covariates, strata, ones(nrow(sub)))
    return v .- _iv_ctrl_proj(ctrl, v)
end

"""Leave-one-out mean of `v` within judge codes `j`; `source` marks the cases that
may contribute (reverse-sample leniency). Returns NaN when no other case contributes."""
function _iv_loo_mean(v::Vector{Float64}, j::Vector{Int}, source::AbstractVector{Bool})
    J = maximum(j)
    s = zeros(J)
    c = zeros(Int, J)
    for i in eachindex(v)
        source[i] || continue
        s[j[i]] += v[i]
        c[j[i]] += 1
    end
    out = similar(v)
    for i in eachindex(v)
        si, ci = s[j[i]], c[j[i]]
        if source[i]
            si -= v[i]
            ci -= 1
        end
        out[i] = ci > 0 ? si / ci : NaN
    end
    return out
end

"""
    judge_leniency(data, treatment, judge; strata=Symbol[], covariates=Symbol[],
                   residualize=true) -> Vector{Union{Missing,Float64}}

Leave-one-out leniency of the assigned judge, the instrument of a judge (examiner)
design.

In judge designs, cases are assigned as good as randomly to decision-makers (judges,
patent examiners, caseworkers, physicians) who differ in their propensity to take a
decision such as detention (Kling 2006; Dobbie, Goldin and Yang 2018). The
decision-maker's propensity is an instrument for the decision. It is estimated from
the decisions of the same judge on other cases: for case ``i`` assigned to judge
``j`` with ``n_j`` cases,

```math
Z_i = \\frac{1}{n_j - 1} \\sum_{k \\in j,\\, k \\ne i} \\tilde D_k ,
```

where ``\\tilde D`` is the decision residualized on the `strata` fixed effects (the
cells, such as court × period, within which assignment is random) and the
`covariates` when `residualize = true` (Dobbie, Goldin and Yang 2018); without strata
or covariates it is the demeaned decision, and with `residualize = false` the raw
decision. Leaving the own case out removes the mechanical correlation between the
case's own decision (and hence its outcome) and the measured leniency of its judge,
which would otherwise produce the own-observation bias that jackknife IV estimators
remove ([`jive`](@ref)).

The leave-one-out mean is a noisy estimate of the judge's propensity when judges
handle few cases, which weakens the first stage but does not bias 2SLS, since the
noise is independent of the case's own errors under random assignment. Residualizing
on strata is essential when judges' caseloads differ in composition across strata;
residualizing on covariates is optional under random assignment and mainly improves
precision. The instrument is used by [`judge_iv`](@ref) and the diagnostics
[`judge_balance_test`](@ref) and [`judge_subsample_monotonicity`](@ref).

# Arguments
- `data::AbstractDataFrame`: case-level data.
- `treatment::Symbol`: the numeric decision (for example 0/1 detention).
- `judge::Symbol`: the judge identifier.

# Keywords
- `strata::Vector{Symbol}`: fixed effects within which judges are randomly assigned
  (default none).
- `covariates::Vector{Symbol}`: case characteristics to residualize on (default
  none).
- `residualize::Bool`: residualize the decision on strata and covariates before
  averaging (default `true`); with `false` the raw decision is averaged.

# Returns
- A `Vector{Union{Missing,Float64}}` with one entry per row of `data`: `missing` for
  rows with missing values in the used columns and for judges with a single case
  (which are dropped with a warning).

# Examples
```julia
using DrSnow, DataFrames, StableRNGs
rng = StableRNG(7)
n, J = 6_000, 60
judge = rand(rng, 1:J, n)                        # random assignment
court = (judge .- 1) .÷ 15 .+ 1                  # 4 courts of 15 judges
stringency = 0.25 .+ 0.5 .* rand(rng, J)         # judge-specific detention rate
d = Float64.(rand(rng, n) .< stringency[judge])
df = DataFrame(d=d, judge=judge, court=court)
df.leniency = judge_leniency(df, :d, :judge; strata=[:court])
```

# References
- Kling, J. R. (2006). Incarceration length, employment, and earnings. *American
  Economic Review*, 96(3), 863–876.
- Dobbie, W., Goldin, J., & Yang, C. S. (2018). The effects of pretrial detention on
  conviction, future crime, and employment: Evidence from randomly assigned judges.
  *American Economic Review*, 108(2), 201–240.
- Frandsen, B., Lefgren, L., & Leslie, E. (2023). Judging judge fixed effects.
  *American Economic Review*, 113(1), 253–277.
"""
function judge_leniency(data::AbstractDataFrame, treatment::Symbol, judge::Symbol;
                        strata=Symbol[], covariates=Symbol[], residualize::Bool=true)
    st, covs = _as_symbols(strata), _as_symbols(covariates)
    cols = unique(vcat([treatment, judge], st, covs))
    _iv_check_numeric(data, [treatment], "judge_leniency")
    idx = _iv_judge_sample(data, cols, judge, "judge_leniency")
    return _iv_leniency_rows(data, idx, treatment, judge, st, covs, residualize)
end

function _iv_leniency_rows(data, idx, treatment, judge, st, covs, residualize;
                           source=nothing)
    sub = disallowmissing(data[idx, unique(vcat([treatment, judge], st, covs))])
    d = Float64.(sub[!, treatment])
    v = residualize ? _iv_residualize(sub, d, st, covs) : d
    j = _iv_codes(sub[!, judge])
    src = source === nothing ? trues(length(idx)) : source[idx]
    z = _iv_loo_mean(v, j, src)
    out = Vector{Union{Missing,Float64}}(missing, nrow(data))
    for (t, i) in enumerate(idx)
        out[i] = isnan(z[t]) ? missing : z[t]
    end
    return out
end

"""
    JudgeIVEstimate <: CausalEstimate

Result of [`judge_iv`](@ref): the IV estimate of the effect of a decision in a judge
(examiner) design, with the design's first-stage summary.

The coefficient is the 2SLS estimate with the leave-one-out leniency instrument
(`method = :leniency`) or the UJIVE estimate with judge indicators as instruments
(`method = :ujive`). The object also records the number of judges, the distribution
of cases per judge (few cases per judge make the leniency measure noisy) and the
first stage of the leniency instrument, and it keeps the leniency 2SLS fit and the
instrument itself for further diagnostics. The `StatsAPI` accessors `coef`, `vcov`,
`stderror`, `confint`, `coeftable`, `nobs` and `dof_residual` work on it.

# Fields
- `coef::Vector{Float64}`, `vcov::Matrix{Float64}`, `coefnames::Vector{String}`,
  `nobs::Int`, `dof_residual::Float64`: the treatment-effect estimate and its
  covariance.
- `method::Symbol`: `:leniency` (2SLS with the leave-one-out leniency instrument) or
  `:ujive` (UJIVE with judge indicators as instruments).
- `se_type::String`: the covariance estimator.
- `outcome::Symbol`, `treatment::Symbol`, `judge::Symbol`, `strata::Vector{Symbol}`,
  `covariates::Vector{Symbol}`: the specification.
- `n_judges::Int`: number of judges in the estimation sample.
- `cases_per_judge::NamedTuple`: `(min, median, max)` of the cases per judge.
- `first_stage::FirstStageResult`: first stage of the leniency instrument (also for
  `method = :ujive`).
- `leniency_sd::Float64`: standard deviation of the leniency instrument.
- `level::Float64`, `estimand::String`, `estimand_note::String`: confidence level,
  target parameter and its caveats.
- `iv::IVEstimate`: the leniency 2SLS fit (also for `method = :ujive`).
- `leniency::Vector{Union{Missing,Float64}}`: the instrument, one entry per row of the
  input data.
"""
struct JudgeIVEstimate <: CausalEstimate
    coef::Vector{Float64}
    vcov::Matrix{Float64}
    coefnames::Vector{String}
    nobs::Int
    dof_residual::Float64
    method::Symbol
    se_type::String
    outcome::Symbol
    treatment::Symbol
    judge::Symbol
    strata::Vector{Symbol}
    covariates::Vector{Symbol}
    n_judges::Int
    cases_per_judge::NamedTuple
    first_stage::FirstStageResult
    leniency_sd::Float64
    level::Float64
    estimand::String
    estimand_note::String
    iv::IVEstimate
    leniency::Vector{Union{Missing,Float64}}
end

StatsAPI.coef(r::JudgeIVEstimate) = r.coef
StatsAPI.vcov(r::JudgeIVEstimate) = r.vcov
StatsAPI.coefnames(r::JudgeIVEstimate) = r.coefnames
StatsAPI.nobs(r::JudgeIVEstimate) = r.nobs
StatsAPI.dof_residual(r::JudgeIVEstimate) = r.dof_residual
StatsAPI.confint(r::JudgeIVEstimate; level::Real=r.level) =
    invoke(StatsAPI.confint, Tuple{CausalEstimate}, r; level=level)
estimand(r::JudgeIVEstimate) = r.estimand
method_name(r::JudgeIVEstimate) = r.method === :ujive ?
    "Judge design (UJIVE, judge indicators)" : "Judge design (2SLS, leave-one-out leniency)"

function show_details(io::IO, r::JudgeIVEstimate)
    println(io)
    println(io, "Covariance: ", r.se_type)
    c = r.cases_per_judge
    @printf(io, "Judges: %d; cases per judge: min %d, median %g, max %d\n",
            r.n_judges, c.min, c.median, c.max)
    fs = r.first_stage
    _iv_printf(io, "First stage: leniency coefficient %.4g (se %.3g), F = %.2f; " *
               "sd(leniency) = %.4g\n", fs.coef[1], sqrt(fs.vcov[1, 1]), fs.F,
               r.leniency_sd)
    println(io, "Estimand: ", r.estimand)
    println(io, "Note: ", r.estimand_note)
end

"""
    judge_iv(data, outcome, treatment, judge; strata=Symbol[], covariates=Symbol[],
             method=:leniency, residualize=true, cluster=judge, vcov=nothing,
             level=0.95) -> JudgeIVEstimate

Instrumental-variables estimate of the effect of a decision made by randomly assigned
judges, examiners or caseworkers.

Judge designs exploit the as-good-as-random assignment of cases to decision-makers who
differ in stringency: a case assigned to a more lenient judge is more likely to
receive the decision (for instance, pretrial release), and the judge's identity
affects the outcome only through that decision (Kling 2006; Dobbie, Goldin and Yang
2018; Bhuller, Dahl, Løken and Mogstad 2020). Let ``D_i(j)`` be case ``i``'s decision
if assigned to judge ``j`` and ``Y_i(d)`` its potential outcomes. The identifying
assumptions are (i) random assignment of judges, possibly within `strata` such as
court × period; (ii) exclusion, that the judge affects the outcome only through the
decision (judges do not, for example, also choose sentence length or bail
conditions that matter for the outcome); and (iii) monotonicity. Strict monotonicity
requires that a case treated by one judge is treated by every more stringent judge,
which is implausible if judges weigh case characteristics differently; Chan, Gentzkow
and Yu (2022) show that differences in skill across decision-makers violate it and
document such violations among radiologists. Frandsen, Lefgren and Leslie (2023)
replace it by *average monotonicity*: for every case, the covariance across judges
(weighted by assignment probabilities) between judge propensity ``p_j`` and the
case's decision ``D_i(j)`` is non-negative.

Under (i)–(iii) the 2SLS coefficient on the leniency instrument is a weighted average
of the effects ``Y_i(1) - Y_i(0)`` of cases whose decision depends on the judge drawn.
Under strict monotonicity the weights are non-negative (Imbens and Angrist 1994);
under average monotonicity they remain non-negative and are proportional to each
case's covariance between its decisions and judge propensities (Frandsen, Lefgren and
Leslie 2023). The weights are not complier shares, and the estimand is neither the
average effect among all cases nor that among the cases at the margin of a
particular judge. Random assignment has testable implications for pre-assignment
characteristics ([`judge_balance_test`](@ref)); exclusion and average monotonicity
jointly have testable implications for judge-level mean outcomes
([`judge_validity_test`](@ref)); and monotonicity implies non-negative first stages in
every subsample ([`judge_subsample_monotonicity`](@ref)). Non-rejection in these
checks does not establish the assumptions.

Two estimators are available. With `method = :leniency` (default) the instrument is
the leave-one-out leniency of the assigned judge ([`judge_leniency`](@ref),
residualized on strata and covariates when `residualize = true`), and 2SLS controls
for the `strata` fixed effects and the `covariates`. With `method = :ujive` the
instruments are one indicator per judge and the estimator is UJIVE (Kolesár 2013; see
[`jive`](@ref)), whose leave-one-out step is applied both to the full first stage and
to the controls, so that it stays consistent with many judges and many strata; its
standard errors treat the jackknife instrument as fixed. Both estimators target the
same weighted average under the assumptions above. Standard errors are clustered by
judge by default (`cluster = judge`), which accounts for the estimation error that
all cases of a judge share through the leniency measure; pass, for instance,
`cluster = [:judge, :defendant]` or `vcov` to change this. Report the first-stage
strength (`r.first_stage`), the number of judges and the cases per judge alongside
the estimate.

# Arguments
- `data::AbstractDataFrame`: case-level data; rows with missing values in used
  columns and judges with a single case are dropped.
- `outcome::Symbol`: the outcome.
- `treatment::Symbol`: the numeric decision (for example 0/1).
- `judge::Symbol`: the judge identifier.

# Keywords
- `strata::Vector{Symbol}`: fixed effects within which assignment is random (default
  none).
- `covariates::Vector{Symbol}`: case-level controls (default none).
- `method::Symbol`: `:leniency` (default) or `:ujive`.
- `residualize::Bool`: residualize the decisions on strata and covariates before
  computing leniency (default `true`).
- `cluster`: clustering variable(s) (default: the `judge` column).
- `vcov`: an explicit covariance specification, overriding `cluster` (default
  `nothing`).
- `level::Real`: confidence level (default 0.95).

# Returns
- A [`JudgeIVEstimate`](@ref); `estimate(r)` and `confint(r)` refer to the effect of
  the decision, `r.first_stage` is the leniency first stage and `r.iv` the leniency
  2SLS fit.

# Examples
```julia
using DrSnow, DataFrames, StableRNGs
rng = StableRNG(7)
n, J = 6_000, 60
judge = rand(rng, 1:J, n)                        # random assignment
court = (judge .- 1) .÷ 15 .+ 1                  # 4 courts of 15 judges
stringency = 0.25 .+ 0.5 .* rand(rng, J)         # judge-specific detention rate
u = rand(rng, n)                                 # case severity (unobserved)
d = Float64.(u .< stringency[judge])             # monotone threshold rule
y = 0.2 .+ 0.3 .* d .- 0.4 .* u .+ 0.3 .* randn(rng, n)
df = DataFrame(y=y, d=d, judge=judge, court=court)
r = judge_iv(df, :y, :d, :judge; strata=[:court])
r.first_stage.F, confint(r)
judge_iv(df, :y, :d, :judge; strata=[:court], method=:ujive)
```

# References
- Imbens, G. W., & Angrist, J. D. (1994). Identification and estimation of local
  average treatment effects. *Econometrica*, 62(2), 467–475.
- Kling, J. R. (2006). Incarceration length, employment, and earnings. *American
  Economic Review*, 96(3), 863–876.
- Dobbie, W., Goldin, J., & Yang, C. S. (2018). The effects of pretrial detention on
  conviction, future crime, and employment: Evidence from randomly assigned judges.
  *American Economic Review*, 108(2), 201–240.
- Bhuller, M., Dahl, G. B., Løken, K. V., & Mogstad, M. (2020). Incarceration,
  recidivism, and employment. *Journal of Political Economy*, 128(4), 1269–1324.
- Chan, D. C., Gentzkow, M., & Yu, C. (2022). Selection with variation in diagnostic
  skill: Evidence from radiologists. *Quarterly Journal of Economics*, 137(2),
  729–783.
- Frandsen, B., Lefgren, L., & Leslie, E. (2023). Judging judge fixed effects.
  *American Economic Review*, 113(1), 253–277.
- Kolesár, M. (2013). Estimation in an instrumental variables model with treatment
  effect heterogeneity (Working Paper No. 2013-2). Princeton University, Department
  of Economics.
"""
function judge_iv(data::AbstractDataFrame, outcome::Symbol, treatment::Symbol,
                  judge::Symbol; strata=Symbol[], covariates=Symbol[],
                  method::Symbol=:leniency, residualize::Bool=true, cluster=judge,
                  vcov=nothing, level::Real=0.95)
    method in (:leniency, :ujive) ||
        throw(ArgumentError("method must be :leniency or :ujive, got :$method"))
    st, covs = _as_symbols(strata), _as_symbols(covariates)
    vce = _iv_vcov_estimator(cluster, vcov)
    cols = unique(vcat([outcome, treatment, judge], st, covs, _iv_cluster_names(vce)))
    _iv_check_numeric(data, [outcome, treatment], "judge_iv")
    idx = _iv_judge_sample(data, cols, judge, "judge_iv")
    len = _iv_leniency_rows(data, idx, treatment, judge, st, covs, residualize)
    tmp = data[idx, cols]
    tmp[!, :__judge_leniency__] = disallowmissing(len[idx])
    std(tmp.__judge_leniency__) > 0 ||
        throw(ArgumentError("judge_iv: leniency does not vary across judges"))
    ivr = iv_regression(tmp, outcome, [treatment], [:__judge_leniency__];
                        covariates=covs, fe=st, vcov=vce, level=level)
    fs0 = ivr.first_stage.first_stage[1]
    fs = FirstStageResult(treatment, [:leniency], fs0.coef, fs0.vcov, fs0.F,
                          fs0.F_pvalue, fs0.F_homoskedastic, fs0.dof, fs0.partial_r2,
                          fs0.sanderson_windmeijer_F)
    if method === :ujive
        sub = disallowmissing(tmp[ivr.design.esample, :])
        jc = _iv_codes(sub[!, judge])
        J = maximum(jc)
        Zraw = zeros(nrow(sub), J - 1)
        for i in 1:nrow(sub)
            jc[i] > 1 && (Zraw[i, jc[i] - 1] = 1.0)
        end
        jd = _iv_jive_design(sub, outcome, [treatment], Zraw, covs, st, nothing, vce,
                             ivr.design.esample)
        β, V, label, dof, _, _ = _iv_jive_fit(jd, :ujive, :standard)
    else
        β, V, label, dof = ivr.coef[1:1], ivr.vcov[1:1, 1:1], ivr.vcov_type,
                           ivr.dof_residual
    end
    sub_j = tmp[ivr.design.esample, judge]
    cnts = collect(values(StatsBase.countmap(sub_j)))
    cpj = (min=minimum(cnts), median=median(cnts), max=maximum(cnts))
    note = "Weighted average of the effects of `$treatment` for cases whose decision " *
           "depends on the judge drawn, under as-good-as-random assignment of judges " *
           (isempty(st) ? "" : "within strata ") * "and exclusion; the weights are " *
           "non-negative under strict monotonicity or under the average monotonicity " *
           "of Frandsen, Lefgren & Leslie (2023). It is not the average effect for " *
           "all cases."
    return JudgeIVEstimate(β, Matrix(V), [string(treatment)], ivr.nobs, dof, method,
                           label, outcome, treatment, judge, st, covs,
                           length(unique(sub_j)), cpj, fs,
                           std(tmp.__judge_leniency__[ivr.design.esample]), float(level),
                           "weighted average of treatment effects for marginal cases",
                           note, ivr, len)
end

# ---------------------------------------------------------------------------
# Balance / randomization check
# ---------------------------------------------------------------------------

"""
    judge_balance_test(data, treatment, judge, characteristics; strata=Symbol[],
                       cluster=judge, vcov=nothing) -> DiagnosticTest

Randomization (balance) check for a judge design: do pre-assignment case
characteristics predict the leniency of the assigned judge?

If cases are assigned to judges as good as randomly within `strata`, characteristics
determined before assignment are independent of the assigned judge and hence of the
judge's leniency. The function regresses the leave-one-out leniency instrument
([`judge_leniency`](@ref), residualized on the strata) on the `characteristics`, with
`strata` fixed effects, and reports the Wald F test that all coefficients are zero,
using judge-clustered covariance by default and an ``F(q, \\text{dof})`` reference
distribution (``q`` characteristics; ``G - 1`` denominator degrees of freedom under
clustering). This is the standard randomization check of the judge-design literature
(Dobbie, Goldin and Yang 2018; Bhuller, Dahl, Løken and Mogstad 2020). For contrast,
`details` also reports the same regression with the decision itself as the dependent
variable: characteristics usually predict the decision strongly, and the comparison
shows that they are nonetheless unrelated to the judge assignment.

A rejection is evidence against random assignment (or of a stratification that does
not match the assignment mechanism). Non-rejection does not establish random
assignment, since the test only examines the observed characteristics and has
limited power with few judges, and no balance test can speak to exclusion or
monotonicity; see [`judge_validity_test`](@ref) and
[`judge_subsample_monotonicity`](@ref) for those.

# Arguments
- `data::AbstractDataFrame`: case-level data.
- `treatment::Symbol`: the numeric decision from which leniency is computed.
- `judge::Symbol`: the judge identifier.
- `characteristics`: pre-assignment case characteristics, a `Symbol` or vector
  (numeric columns).

# Keywords
- `strata::Vector{Symbol}`: fixed effects within which assignment is random (default
  none).
- `cluster`: clustering variable(s) (default: the `judge` column).
- `vcov`: an explicit covariance specification, overriding `cluster` (default
  `nothing`).

# Returns
- A [`DiagnosticTest`](@ref) with the Wald F statistic, its degrees of freedom and
  p-value; `details` has `characteristics`, `coef_leniency`, `se_leniency`,
  `F_treatment` and `p_treatment`.

# Examples
```julia
using DrSnow, DataFrames, StableRNGs
rng = StableRNG(7)
n, J = 6_000, 60
judge = rand(rng, 1:J, n)                        # random assignment
court = (judge .- 1) .÷ 15 .+ 1
stringency = 0.25 .+ 0.5 .* rand(rng, J)
age = 20 .+ 30 .* rand(rng, n)
d = Float64.(rand(rng, n) .< stringency[judge] .+ 0.004 .* (age .- 35))
df = DataFrame(d=d, judge=judge, court=court, age=age, female=rand(rng, 0:1, n))
judge_balance_test(df, :d, :judge, [:age, :female]; strata=[:court])
```

# References
- Dobbie, W., Goldin, J., & Yang, C. S. (2018). The effects of pretrial detention on
  conviction, future crime, and employment: Evidence from randomly assigned judges.
  *American Economic Review*, 108(2), 201–240.
- Bhuller, M., Dahl, G. B., Løken, K. V., & Mogstad, M. (2020). Incarceration,
  recidivism, and employment. *Journal of Political Economy*, 128(4), 1269–1324.
- Frandsen, B., Lefgren, L., & Leslie, E. (2023). Judging judge fixed effects.
  *American Economic Review*, 113(1), 253–277.
"""
function judge_balance_test(data::AbstractDataFrame, treatment::Symbol, judge::Symbol,
                            characteristics; strata=Symbol[], cluster=judge,
                            vcov=nothing)
    chars = _as_symbols(characteristics)
    isempty(chars) && throw(ArgumentError("at least one characteristic is required"))
    st = _as_symbols(strata)
    vce = _iv_vcov_estimator(cluster, vcov)
    cols = unique(vcat([treatment, judge], chars, st, _iv_cluster_names(vce)))
    _iv_check_numeric(data, vcat([treatment], chars), "judge_balance_test")
    idx = _iv_judge_sample(data, cols, judge, "judge_balance_test")
    len = _iv_leniency_rows(data, idx, treatment, judge, st, Symbol[], true)
    tmp = data[idx, cols]
    tmp[!, :__judge_leniency__] = disallowmissing(len[idx])
    res = map([:__judge_leniency__, treatment]) do lhs
        des = _iv_ols_design(tmp, [lhs], chars, Symbol[], st, nothing, vce,
                             "judge_balance_test")
        B, V, _ = _iv_ols(des, des.y, des.Z)
        q = length(chars)
        F = _iv_wald_F(vec(B), V)
        dof = _iv_ref_dof(des, q)
        (b=vec(B), V=V, F=F, p=ccdf(FDist(q, dof), F), dof=dof)
    end
    L, Dr = res
    q = length(chars)
    return DiagnosticTest("Judge-design balance test",
                          "case characteristics do not predict the leniency of the " *
                          "assigned judge" * (isempty(st) ? "" : " (within strata)"),
                          L.F, L.p; dof=(q, L.dof),
                          method="Wald F, leniency on characteristics, " *
                                 _iv_vcov_label(_iv_ols_design(tmp, [:__judge_leniency__],
                                                               chars, Symbol[], st,
                                                               nothing, vce,
                                                               "judge_balance_test")),
                          note="Non-rejection is not evidence of random assignment; " *
                               "exclusion and monotonicity are not tested.",
                          details=(characteristics=chars, coef_leniency=L.b,
                                   se_leniency=sqrt.(diag(L.V)), F_treatment=Dr.F,
                                   p_treatment=Dr.p))
end

# ---------------------------------------------------------------------------
# Frandsen, Lefgren & Leslie (2023): exclusion + average monotonicity
# ---------------------------------------------------------------------------

"""Box-constrained weighted least squares by enumeration of active sets (exact for
the small number of slope parameters used here). Column 1 is unconstrained."""
function _iv_box_wls(X::Matrix{Float64}, y::Vector{Float64}, w::Vector{Float64},
                     lo::Float64, hi::Float64)
    S = size(X, 2) - 1
    sw = sqrt.(w)
    Xw, yw = X .* sw, y .* sw
    best, bestobj = zeros(S + 1), Inf
    for code in 0:(3^S - 1)
        state = digits(code; base=3, pad=S)          # 0 free, 1 at lo, 2 at hi
        b = zeros(S + 1)
        fixed = [s + 1 for s in 1:S if state[s] != 0]
        for s in 1:S
            state[s] == 1 && (b[s + 1] = lo)
            state[s] == 2 && (b[s + 1] = hi)
        end
        free = setdiff(1:(S + 1), fixed)
        r = yw - Xw[:, fixed] * b[fixed]
        b[free] = Xw[:, free] \ r
        all(lo - 1e-12 <= b[s + 1] <= hi + 1e-12 for s in 1:S) || continue
        obj = sum(abs2, yw - Xw * b)
        if obj < bestobj
            bestobj, best = obj, b
        end
    end
    return best, bestobj
end

"""Linear-spline basis with slopes as coefficients: [1, φ₁(p), …, φ_S(p)]."""
function _iv_spline_basis(p::Vector{Float64}, knots::Vector{Float64})
    S = length(knots) - 1
    X = ones(length(p), S + 1)
    for s in 1:S
        X[:, s + 1] = clamp.(p .- knots[s], 0.0, knots[s + 1] - knots[s])
    end
    return X
end

function _iv_spline_slope(p::Vector{Float64}, knots::Vector{Float64}, b)
    S = length(knots) - 1
    return [b[clamp(searchsortedlast(knots, x), 1, S)] for x in p]
end

"""Constrained minimum-distance fit of judge means on propensities (two FGLS
iterations accounting for the sampling error in both)."""
function _iv_fll_stat(ȳ, p̂, vy, vp, cyp, knots, B)
    X = _iv_spline_basis(p̂, knots)
    slope = zeros(length(ȳ))
    b, T = zeros(size(X, 2)), Inf
    for _ in 1:3
        v = max.(vy .- 2 .* slope .* cyp .+ slope .^ 2 .* vp, 1e-12 .* maximum(vy))
        b, T = _iv_box_wls(X, ȳ, 1 ./ v, -B, B)
        slope = _iv_spline_slope(p̂, knots, b[2:end])
    end
    return b, T, slope
end

"""
    judge_validity_test(data, outcome, treatment, judge; strata=Symbol[],
                        covariates=Symbol[], method=:fll, n_knots=5, omega=0.9,
                        outcome_bounds=nothing, n_simulations=10_000,
                        n_segments=3, n_bootstrap=999,
                        rng=Random.default_rng()) -> DiagnosticTest

Joint test of the exclusion restriction and (average) monotonicity in a judge design
(Frandsen, Lefgren and Leslie 2023).

Frandsen, Lefgren and Leslie (2023, FLL) show that, under random assignment, exclusion
and average monotonicity (see [`judge_iv`](@ref)), the mean outcome of the cases of
judge ``j`` depends on the judge only through the judge's treatment propensity,
``E[Y \\mid J = j] = \\varphi(p_j)``, and that ``\\varphi`` is Lipschitz with constant
``K = y_{\\max} - y_{\\min}``, since no treatment effect can exceed the range of the
outcome (FLL, Theorem 2). Two kinds of evidence therefore contradict the null: judges
with similar propensities but different mean outcomes (the *fit* component), and an
outcome–propensity relation steeper than any treatment effect could produce (the
*slope* component). The test has power against violations of exclusion (judges
affecting outcomes through other channels) and of monotonicity, but it cannot tell
them apart, and it takes random assignment as given (check it with
[`judge_balance_test`](@ref)).

With `method = :fll` (default) the function implements FLL's test (their Section 3 and
appendix). Case-level outcome and treatment are first residualized on the
`covariates` (overall means added back; the `strata` enter the regressions as
indicators); ``\\hat p_i`` is the treatment rate of case ``i``'s judge and
``\\hat v_i = D_i - \\hat p_i``.

1. **Fit component.** Regress ``Y_i`` on a quadratic B-spline ``\\hat S_i = S(\\hat p_i)``
   with `n_knots` knots ``t_0 < \\dots < t_{m-1}`` (boundary knots at the smallest and
   largest judge propensity, interior knots at quantiles of the judge propensities;
   ``m + 1`` basis functions), take the residuals ``\\hat u_i`` and regress them on
   judge indicators: ``\\hat\\gamma_j`` is the mean of ``\\hat u`` among judge ``j``'s
   cases. The Wald statistic ``T = n\\hat\\gamma'\\hat\\Omega^{+}\\hat\\gamma`` is compared
   with ``\\chi^2(J - m - 1)``. Here
   ``\\hat\\Omega = Q_W^{-1}E_n[\\psi\\psi']Q_W^{-1}`` with
   ``\\psi_i = W_i(\\hat u_i - \\hat\\varphi'(\\hat p_i)\\hat v_i) -
   AQ_S^{-1}\\hat S_i(\\hat u_i - \\hat\\varphi'(\\hat p_i)\\hat v_i)``,
   ``A = E_n[W_i\\hat S_i']``,
   ``Q_S = E_n[\\hat S_i\\hat S_i']`` and ``Q_W = E_n[W_iW_i']``, which accounts for the
   estimation of the propensities (the linearization of FLL's appendix). The effect of
   ``\\hat p`` on the spline coefficients is propagated as well, which makes
   ``\\hat\\Omega`` singular in exactly the ``m + 1`` directions in which
   ``\\hat\\gamma`` is identically zero; hence the pseudo-inverse and the degrees of
   freedom.
2. **Slope component.** The slopes of the quadratic spline at the knots,
   ``\\hat\\varphi'(t_l) = 2(\\hat\\delta_{l+1} - \\hat\\delta_l)/(t_{l+1} - t_{l-1})``,
   must lie in ``[-K, K]``. FLL test these ``2m`` inequalities with the
   generalized-moment-selection procedure of Andrews and Soares (2010): the
   modified-method-of-moments statistic ``\\hat M = \\sum_l [(K - \\hat\\varphi'_l)
   /\\text{se}_l]_-^2 + [(K + \\hat\\varphi'_l)/\\text{se}_l]_-^2``, with inequalities
   whose standardized slack exceeds ``\\sqrt{\\ln n}`` dropped, and a p-value
   simulated from the normal approximation (`n_simulations` draws with `rng`).
3. **Joint p-value** (weighted Bonferroni): ``p = \\min(1, p_{\\text{fit}}/\\omega,
   p_{\\text{slope}}/(1 - \\omega))``. With ``\\omega = 1`` only the fit component is
   used (FLL's choice with many judges), with ``\\omega = 0`` only the slope component
   (appropriate with very few judges); the default ``\\omega = 0.9`` directs most of
   the power to the fit component.

The reported statistic is the fit Wald statistic ``T``; `details` holds both
components. ``K`` is the sample range of the outcome unless
`outcome_bounds = (lo, hi)` gives its logical bounds. The sample range can
understate the logical range, which makes the slope component too strict; supplying
the bounds of a bounded outcome avoids this. Cases are treated as independent, and
there must be more judges than spline terms (``J >`` `n_knots` + 1).

With `method = :minimum_distance` the function uses an earlier approximation, kept for
comparison: the judge means are fitted by a continuous piecewise-linear function of
the propensity with `n_segments` segments and slopes constrained to ``[-K, K]`` by
minimum distance, with weights that account for the sampling error of both judge
means and propensities; the statistic is
``\\sum_j (\\bar y_j - \\hat\\mu(\\hat p_j))^2/v_j`` and its p-value comes from a
parametric bootstrap under the fitted null (`n_bootstrap` draws).

For either method, a rejection is evidence against exclusion or average
monotonicity, given random assignment. Non-rejection does not establish them: power
is limited with few judges or few cases per judge, and violations that preserve the
relation between judge means and propensities are not detectable.

# Arguments
- `data::AbstractDataFrame`: case-level data.
- `outcome::Symbol`: the outcome.
- `treatment::Symbol`: the numeric decision.
- `judge::Symbol`: the judge identifier.

# Keywords
- `strata`, `covariates`: controls, as in [`judge_iv`](@ref) (default none).
- `method::Symbol`: `:fll` (default) or `:minimum_distance`.
- `n_knots::Int`: number of spline knots for `:fll` (default 5, at least 2).
- `omega::Real`: weight ``\\omega \\in [0, 1]`` of the fit component in the weighted
  Bonferroni combination (default 0.9).
- `outcome_bounds`: `nothing` (default; use the sample range) or `(lo, hi)`, the
  logical bounds of the outcome, which set the slope bound ``K = hi - lo``.
- `n_simulations::Int`: draws for the simulated slope p-value (default 10 000, at
  least 999).
- `n_segments::Int`: segments of the piecewise-linear fit for `:minimum_distance`
  (default 3).
- `n_bootstrap::Int`: parametric-bootstrap draws for `:minimum_distance` (default
  999, at least 99).
- `rng::AbstractRNG`: random number generator for the simulations (default
  `Random.default_rng()`; pass a seeded generator for reproducibility).

# Returns
- A [`DiagnosticTest`](@ref). For `:fll`, `details` has `fit_statistic`, `fit_dof`,
  `fit_pvalue`, `slope_statistic`, `slope_pvalue`, `omega`, `slopes` (at the knots),
  `slope_se`, `knots`, `slope_bound`, `judge_means` (judge, n, propensity, mean
  outcome, fitted value, `gamma`) and `n_judges`. For `:minimum_distance`, `details`
  has `judge_means`, `slopes`, `knots`, `slope_bound`, `pvalue_chisq` and `n_judges`.

# Examples
```julia
using DrSnow, DataFrames, StableRNGs
rng = StableRNG(7)
n, J = 6_000, 60
judge = rand(rng, 1:J, n)
court = (judge .- 1) .÷ 15 .+ 1
stringency = 0.25 .+ 0.5 .* rand(rng, J)
u = rand(rng, n)
d = Float64.(u .< stringency[judge])
y = 0.2 .+ 0.3 .* d .- 0.4 .* u .+ 0.3 .* randn(rng, n)
df = DataFrame(y=y, d=d, judge=judge, court=court)
judge_validity_test(df, :y, :d, :judge; strata=[:court], rng=StableRNG(1))
```

# References
- Frandsen, B., Lefgren, L., & Leslie, E. (2023). Judging judge fixed effects.
  *American Economic Review*, 113(1), 253–277.
- Andrews, D. W. K., & Soares, G. (2010). Inference for parameters defined by moment
  inequalities using generalized moment selection. *Econometrica*, 78(1), 119–157.
"""
function judge_validity_test(data::AbstractDataFrame, outcome::Symbol, treatment::Symbol,
                             judge::Symbol; strata=Symbol[], covariates=Symbol[],
                             method::Symbol=:fll, n_knots::Int=5, omega::Real=0.9,
                             n_simulations::Int=10_000,
                             n_segments::Int=3, outcome_bounds=nothing,
                             n_bootstrap::Int=999,
                             rng::AbstractRNG=Random.default_rng())
    method in (:fll, :minimum_distance) ||
        throw(ArgumentError("method must be :fll or :minimum_distance, got :$method"))
    if method === :fll
        n_knots >= 2 || throw(ArgumentError("n_knots must be at least 2"))
        0 <= omega <= 1 || throw(ArgumentError("omega must be in [0, 1]"))
        n_simulations >= 999 ||
            throw(ArgumentError("n_simulations must be at least 999"))
    else
        n_segments >= 1 || throw(ArgumentError("n_segments must be at least 1"))
        n_bootstrap >= 99 || throw(ArgumentError("n_bootstrap must be at least 99"))
    end
    st, covs = _as_symbols(strata), _as_symbols(covariates)
    cols = unique(vcat([outcome, treatment, judge], st, covs))
    _iv_check_numeric(data, [outcome, treatment], "judge_validity_test")
    idx = _iv_judge_sample(data, cols, judge, "judge_validity_test")
    sub = disallowmissing(data[idx, cols])
    yraw = Float64.(sub[!, outcome])
    draw = Float64.(sub[!, treatment])
    y = _iv_residualize(sub, yraw, st, covs) .+ mean(yraw)
    d = _iv_residualize(sub, draw, st, covs) .+ mean(draw)
    jc = _iv_codes(sub[!, judge])
    J = maximum(jc)
    B = if outcome_bounds === nothing
        maximum(yraw) - minimum(yraw)
    else
        length(outcome_bounds) == 2 && outcome_bounds[2] > outcome_bounds[1] ||
            throw(ArgumentError("outcome_bounds must be (lo, hi) with hi > lo"))
        float(outcome_bounds[2] - outcome_bounds[1])
    end
    B > 0 || throw(ArgumentError("judge_validity_test: the outcome does not vary"))
    if method === :fll
        J > n_knots + 1 ||
            throw(ArgumentError("judge_validity_test: need more judges ($J) than " *
                                "spline terms (n_knots + 1 = $(n_knots + 1))"))
        # strata enter the regressions as indicators (see _iv_fll_test); only the
        # covariates are residualized
        yc = _iv_residualize(sub, yraw, Symbol[], covs) .+ mean(yraw)
        dc = _iv_residualize(sub, draw, Symbol[], covs) .+ mean(draw)
        sc = isempty(st) ? ones(Int, nrow(sub)) :
             _iv_codes([Tuple(sub[i, c] for c in st) for i in 1:nrow(sub)])
        return _iv_fll_test(sub, judge, yc, dc, jc, J, sc, B, n_knots, float(omega),
                            n_simulations, rng)
    end
    J >= n_segments + 3 ||
        throw(ArgumentError("judge_validity_test: need at least n_segments + 3 = " *
                            "$(n_segments + 3) judges, found $J"))
    nj = zeros(Int, J)
    for g in jc
        nj[g] += 1
    end
    members = [Int[] for _ in 1:J]
    for (i, g) in enumerate(jc)
        push!(members[g], i)
    end
    ȳ = [mean(y[m]) for m in members]
    p̂ = [mean(d[m]) for m in members]
    vy = [var(y[m]) / length(m) for m in members]
    vp = [var(d[m]) / length(m) for m in members]
    cyp = [cov(y[m], d[m]) / length(m) for m in members]
    maximum(p̂) - minimum(p̂) > 0 ||
        throw(ArgumentError("judge_validity_test: judge propensities do not vary"))
    knots = quantile(p̂, range(0, 1; length=n_segments + 1))
    knots = unique(knots)
    length(knots) == n_segments + 1 ||
        throw(ArgumentError("judge_validity_test: too few distinct propensities for " *
                            "$n_segments segments"))
    b, T, slope = _iv_fll_stat(ȳ, p̂, vy, vp, cyp, knots, B)
    μ̂ = _iv_spline_basis(p̂, knots) * b
    # parametric bootstrap under the fitted null
    seeds = task_seeds(rng, n_bootstrap)
    Tb = zeros(n_bootstrap)
    for r in 1:n_bootstrap
        brng = Xoshiro(seeds[r])
        ys, ps = similar(ȳ), similar(p̂)
        for j in 1:J
            C = [vy[j] cyp[j]; cyp[j] vp[j]]
            L = cholesky(Symmetric(C + 1e-14 * I); check=false)
            e = issuccess(L) ? L.L * randn(brng, 2) :
                [sqrt(vy[j]) * randn(brng), sqrt(vp[j]) * randn(brng)]
            ys[j] = μ̂[j] + e[1]
            ps[j] = p̂[j] + e[2]
        end
        kb = quantile(ps, range(0, 1; length=n_segments + 1))
        if length(unique(kb)) < n_segments + 1
            kb = knots
        end
        _, Tb[r], _ = _iv_fll_stat(ys, ps, vy, vp, cyp, kb, B)
    end
    pval = (1 + count(>=(T), Tb)) / (1 + n_bootstrap)
    dofc = J - 1 - n_segments
    table = DataFrame(judge=[sub[members[j][1], judge] for j in 1:J], n=nj,
                      propensity=p̂, mean_outcome=ȳ, fitted=μ̂)
    return DiagnosticTest("Judge-design test of exclusion and average monotonicity " *
                          "(Frandsen, Lefgren & Leslie 2023)",
                          "judge mean outcomes are a function of judge propensity " *
                          "with slope in [−$(round(B; sigdigits=4)), " *
                          "$(round(B; sigdigits=4))]", T, pval; dof=(dofc,),
                          method="constrained minimum distance, linear spline with " *
                                 "$n_segments segment(s); parametric bootstrap " *
                                 "($n_bootstrap draws)",
                          note="Minimum-distance approximation to FLL's " *
                               "procedure (method = :minimum_distance). A rejection is " *
                               "evidence against exclusion and/or average " *
                               "monotonicity (given random assignment); non-rejection " *
                               "does not establish them, and power is limited with few " *
                               "judges or few cases per judge.",
                          details=(judge_means=table, slopes=b[2:end], knots=knots,
                                   slope_bound=B, pvalue_chisq=ccdf(Chisq(dofc), T),
                                   n_judges=J))
end

# ---------------------------------------------------------------------------
# Frandsen, Lefgren & Leslie (2023) test
# ---------------------------------------------------------------------------

"""
Clamped B-spline basis of degree `deg` with knots `t` (boundary knots `t[1]`,
`t[end]`) at `x`, and its derivative. Returns `(S, dS)`, `length(x) × (m + deg − 1)`
with `m = length(t)`.
"""
function _iv_bspline(x::AbstractVector, t::Vector{Float64}, deg::Int)
    τ = vcat(fill(t[1], deg), t, fill(t[end], deg))
    nb = length(τ) - deg - 1
    function basis(xv, p)
        # Cox–de Boor recursion; right-closed last interval
        B = zeros(length(xv), length(τ) - 1)
        for (i, v) in enumerate(xv)
            j = v >= τ[end] ? findlast(k -> τ[k] < τ[k + 1], 1:(length(τ) - 1)) :
                searchsortedlast(τ, v)
            j = clamp(j, 1, length(τ) - 1)
            B[i, j] = 1.0
        end
        for q in 1:p
            Bn = zeros(length(xv), length(τ) - 1 - q)
            for k in 1:(length(τ) - 1 - q)
                d1 = τ[k + q] - τ[k]
                d2 = τ[k + q + 1] - τ[k + 1]
                for i in eachindex(xv)
                    a = d1 > 0 ? (xv[i] - τ[k]) / d1 * B[i, k] : 0.0
                    b = d2 > 0 ? (τ[k + q + 1] - xv[i]) / d2 * B[i, k + 1] : 0.0
                    Bn[i, k] = a + b
                end
            end
            B = Bn
        end
        return B
    end
    S = basis(x, deg)
    size(S, 2) == nb || error("internal: B-spline basis size")
    Bl = basis(x, deg - 1)                  # degree deg − 1 on the same knots
    dS = zeros(length(x), nb)
    for k in 1:nb
        d1 = τ[k + deg] - τ[k]
        d2 = τ[k + deg + 1] - τ[k + 1]
        d1 > 0 && (dS[:, k] .+= (deg / d1) .* Bl[:, k])
        d2 > 0 && (dS[:, k] .-= (deg / d2) .* Bl[:, k + 1])
    end
    return S, dS
end

"""FLL (2023) fit + slope test with weighted-Bonferroni combination."""
function _iv_fll_test(sub, judge, y, d, jc, J, sc, K, n_knots, ω, n_sim, rng)
    n = length(y)
    nj = zeros(Int, J)
    for g in jc
        nj[g] += 1
    end
    pj = zeros(J)
    for (i, g) in enumerate(jc)
        pj[g] += d[i]
    end
    pj ./= nj
    maximum(pj) - minimum(pj) > 0 ||
        throw(ArgumentError("judge_validity_test: judge propensities do not vary"))
    knots = unique(quantile(pj, range(0, 1; length=n_knots)))
    length(knots) == n_knots ||
        throw(ArgumentError("judge_validity_test: too few distinct judge propensities " *
                            "for $n_knots knots"))
    p̂ = pj[jc]
    v̂ = d .- p̂
    Sj, dSj = _iv_bspline(pj, knots, 2)
    q = size(Sj, 2)                                  # m + 1 spline terms
    _iv_check_rank(Sj, "spline terms", "at the judge propensities; use fewer knots")
    # regressors: spline terms plus strata indicators (first level dropped; the
    # B-spline basis sums to one)
    C = maximum(sc)
    S = hcat(Sj[jc, :], [Float64(sc[i] == c) for i in 1:n, c in 2:C])
    _iv_check_rank(S, "spline terms and strata indicators", "; use fewer knots")
    QS = Symmetric(S' * S) ./ n
    QSi = inv(QS)
    δa = (S' * S) \ (S' * y)
    δ̂ = δa[1:q]
    û = y .- S * δa
    φp = (dSj * δ̂)[jc]                                # φ̂'(p̂ᵢ)
    γ̂ = zeros(J)
    for (i, g) in enumerate(jc)
        γ̂[g] += û[i]
    end
    γ̂ ./= nj
    # influence functions: ψᵢ (J-vector) for √n γ̂ (before Q_W⁻¹), ψδ for δ̂
    ε = û .- φp .* v̂
    A = zeros(J, size(S, 2))                         # E_n[Wᵢ Sᵢ']
    for (i, g) in enumerate(jc)
        A[g, :] .+= view(S, i, :)
    end
    A ./= n
    Sε = S .* ε
    corr = Sε * QSi * A'                             # n × J
    Ψ = -corr
    for (i, g) in enumerate(jc)
        Ψ[i, g] += ε[i]
    end
    Meat = (Ψ' * Ψ) ./ n
    QWi = n ./ nj
    Ω = Symmetric(QWi .* Meat .* QWi')
    E = eigen(Ω)
    tol = maximum(abs, E.values) * J * 1e-10
    keep = E.values .> tol
    dof = count(keep)
    dof >= 1 || throw(ArgumentError("judge_validity_test: no overidentifying variation " *
                                    "left (too few judges for the spline and strata)"))
    Ωp = E.vectors[:, keep] * Diagonal(1 ./ E.values[keep]) * E.vectors[:, keep]'
    T = n * dot(γ̂, Ωp * γ̂)
    p_fit = ccdf(Chisq(dof), T)
    # slope component (Andrews & Soares GMS, MMM statistic)
    m = n_knots
    tt = vcat(knots[1], knots, knots[end])           # t₋₁ = t₀, t_m = t_{m−1}
    Dm = zeros(m, q)
    for l in 1:m
        c = 2 / (tt[l + 2] - tt[l])
        Dm[l, l + 1] = c
        Dm[l, l] = -c
    end
    slopes = Dm * δ̂
    Ψδ = ((S .* ε) * QSi)[:, 1:q]                    # spline block, n × q
    Vδ = (Ψδ' * Ψδ) ./ n^2
    Vs = Symmetric(Dm * Vδ * Dm')
    se = sqrt.(max.(diag(Vs), 0.0))
    all(>(0), se) || throw(ArgumentError("judge_validity_test: zero standard error of " *
                                         "a spline slope; use fewer knots"))
    tlo = (K .- slopes) ./ se                        # ≥ 0 under H₀
    thi = (K .+ slopes) ./ se
    M̂ = sum(min.(tlo, 0.0) .^ 2) + sum(min.(thi, 0.0) .^ 2)
    κn = sqrt(log(n))
    Lm = findall(<=(κn), tlo)
    Lp = findall(<=(κn), thi)
    p_slope = if M̂ <= 0
        1.0
    elseif isempty(Lm) && isempty(Lp)
        0.0
    else
        Cm = Matrix(Vs) ./ (se * se')
        Ev = eigen(Symmetric(Cm))
        L = Ev.vectors * Diagonal(sqrt.(max.(Ev.values, 0.0)))
        seed = rand(rng, UInt64)
        srng = Xoshiro(seed)
        cnt = 0
        zs = zeros(m)
        for _ in 1:n_sim
            mul!(zs, L, randn(srng, m))
            Ms = 0.0
            for l in Lm
                Ms += min(zs[l], 0.0)^2
            end
            for l in Lp
                Ms += min(-zs[l], 0.0)^2
            end
            cnt += Ms >= M̂
        end
        (1 + cnt) / (1 + n_sim)
    end
    pval = ω == 1 ? p_fit : ω == 0 ? p_slope :
           min(1.0, p_fit / ω, p_slope / (1 - ω))
    strata_part = S[:, (q + 1):end] * δa[(q + 1):end]
    fitted = Sj * δ̂
    for (i, g) in enumerate(jc)
        fitted[g] += strata_part[i] / nj[g]
    end
    ȳ = zeros(J)
    for (i, g) in enumerate(jc)
        ȳ[g] += y[i]
    end
    ȳ ./= nj
    first_row = zeros(Int, J)
    for (i, g) in enumerate(jc)
        first_row[g] == 0 && (first_row[g] = i)
    end
    table = DataFrame(judge=[sub[first_row[g], judge] for g in 1:J], n=nj,
                      propensity=pj, mean_outcome=ȳ, fitted=fitted, gamma=γ̂)
    return DiagnosticTest("Judge-design test of exclusion and monotonicity " *
                          "(Frandsen, Lefgren & Leslie 2023)",
                          "judge mean outcomes are a continuous function of judge " *
                          "propensity with slope in [−$(round(K; sigdigits=4)), " *
                          "$(round(K; sigdigits=4))]", T, pval; dof=(dof,),
                          method="FLL: quadratic B-spline with $n_knots knots; fit Wald " *
                                 "χ²($dof) and GMS slope test combined by weighted " *
                                 "Bonferroni (ω = $ω)",
                          note="A rejection is evidence against exclusion and/or " *
                               "monotonicity (given random assignment); non-rejection " *
                               "does not establish them. Independent cases assumed.",
                          details=(fit_statistic=T, fit_dof=dof, fit_pvalue=p_fit,
                                   slope_statistic=M̂, slope_pvalue=p_slope, omega=ω,
                                   slopes=slopes, slope_se=se, knots=knots,
                                   slope_bound=K, judge_means=table, n_judges=J))
end

# ---------------------------------------------------------------------------
# Monotonicity by subsample
# ---------------------------------------------------------------------------

"""
    judge_subsample_monotonicity(data, treatment, judge, subgroups;
                                 strata=Symbol[], reverse=true, cluster=judge,
                                 vcov=nothing) -> DiagnosticTest

Checks of monotonicity in a judge design by subsample (Dobbie, Goldin and Yang 2018;
Bhuller, Dahl, Løken and Mogstad 2020).

Monotonicity in a judge design means that a judge who is more lenient overall is
(weakly) more lenient for every case. A testable implication is that the first stage
of the decision on judge leniency is non-negative within every subgroup of cases
defined by pre-assignment characteristics (race, offense type, age group). For every
level ``g`` of every `subgroups` column, the function estimates the first stage (the
decision on the leave-one-out leniency, with `strata` fixed effects and
judge-clustered standard errors) on the cases in ``g``, in two variants. In the
*standard* variant the leniency is computed from all cases. In the *reverse-sample*
variant (`reverse = true`) it is computed only from the judge's cases *outside*
``g``, so that the instrument for group ``g`` is built without the group's own
decisions; a positive reverse-sample first stage shows that judges who are lenient
with other cases are also lenient with group ``g``, which is closer to the
monotonicity condition. Levels with fewer than 10 usable cases are skipped.

The reported test is of ``H_0``: every first-stage coefficient is non-negative, with
statistic ``\\min_k t_k`` over the ``m`` estimated coefficients and the one-sided
Bonferroni p-value ``\\min(1, m\\,\\Phi(\\min_k t_k))``. The check is informal in two
respects. Non-negative first stages in every subgroup are necessary but not
sufficient for monotonicity, which restricts decisions case by case, and a
subgroup-level check cannot detect violations within subgroups; average monotonicity
(Frandsen, Lefgren and Leslie 2023) is the weaker condition that the 2SLS
interpretation actually requires, and its implications are tested jointly with
exclusion by [`judge_validity_test`](@ref). The Bonferroni adjustment is conservative
when the coefficients are positively correlated, as the standard and reverse-sample
estimates for the same group are.

# Arguments
- `data::AbstractDataFrame`: case-level data.
- `treatment::Symbol`: the numeric decision.
- `judge::Symbol`: the judge identifier.
- `subgroups`: categorical case characteristics defining the subsamples, a `Symbol`
  or vector.

# Keywords
- `strata::Vector{Symbol}`: fixed effects within which assignment is random (default
  none).
- `reverse::Bool`: also estimate the reverse-sample first stages (default `true`).
- `cluster`: clustering variable(s) (default: the `judge` column).
- `vcov`: an explicit covariance specification, overriding `cluster` (default
  `nothing`).

# Returns
- A [`DiagnosticTest`](@ref) with statistic ``\\min_k t_k`` and the Bonferroni p-value;
  `details.table` is a `DataFrame` with columns `subgroup`, `level`, `sample`
  (`"standard"` or `"reverse"`), `coef`, `se`, `t` and `n`.

# Examples
```julia
using DrSnow, DataFrames, StableRNGs
rng = StableRNG(7)
n, J = 6_000, 60
judge = rand(rng, 1:J, n)
court = (judge .- 1) .÷ 15 .+ 1
stringency = 0.25 .+ 0.5 .* rand(rng, J)
d = Float64.(rand(rng, n) .< stringency[judge])
df = DataFrame(d=d, judge=judge, court=court, female=rand(rng, 0:1, n))
t = judge_subsample_monotonicity(df, :d, :judge, [:female]; strata=[:court])
t.details.table
```

# References
- Dobbie, W., Goldin, J., & Yang, C. S. (2018). The effects of pretrial detention on
  conviction, future crime, and employment: Evidence from randomly assigned judges.
  *American Economic Review*, 108(2), 201–240.
- Bhuller, M., Dahl, G. B., Løken, K. V., & Mogstad, M. (2020). Incarceration,
  recidivism, and employment. *Journal of Political Economy*, 128(4), 1269–1324.
- Frandsen, B., Lefgren, L., & Leslie, E. (2023). Judging judge fixed effects.
  *American Economic Review*, 113(1), 253–277.
"""
function judge_subsample_monotonicity(data::AbstractDataFrame, treatment::Symbol,
                                      judge::Symbol, subgroups; strata=Symbol[],
                                      reverse::Bool=true, cluster=judge, vcov=nothing)
    groups = _as_symbols(subgroups)
    isempty(groups) && throw(ArgumentError("at least one subgroup column is required"))
    st = _as_symbols(strata)
    vce = _iv_vcov_estimator(cluster, vcov)
    cols = unique(vcat([treatment, judge], groups, st, _iv_cluster_names(vce)))
    _iv_check_numeric(data, [treatment], "judge_subsample_monotonicity")
    idx = _iv_judge_sample(data, cols, judge, "judge_subsample_monotonicity")
    full = _iv_leniency_rows(data, idx, treatment, judge, st, Symbol[], true)
    tab = DataFrame(subgroup=Symbol[], level=Any[], sample=String[], coef=Float64[],
                    se=Float64[], t=Float64[], n=Int[])
    inidx = falses(nrow(data))
    inidx[idx] .= true
    for gcol in groups
        for lev in sort(unique(data[idx, gcol]); by=string)
            ing = inidx .& coalesce.(data[!, gcol] .== lev, false)
            count(ing) >= 10 || continue
            variants = [("standard", full)]
            if reverse
                push!(variants, ("reverse",
                                 _iv_leniency_rows(data, idx, treatment, judge, st,
                                                   Symbol[], true; source=.!ing)))
            end
            for (lab, z) in variants
                rows = ing .& .!ismissing.(z)
                count(rows) >= 10 || continue
                tmp = data[rows, cols]
                tmp[!, :__judge_leniency__] = Float64.(z[rows])
                std(tmp.__judge_leniency__) > 0 || continue
                des = try
                    _iv_ols_design(tmp, [treatment], [:__judge_leniency__], Symbol[], st,
                                   nothing, vce, "judge_subsample_monotonicity")
                catch err
                    err isa ArgumentError || rethrow()
                    continue
                end
                Bc, V, _ = _iv_ols(des, des.y, des.Z)
                se = sqrt(V[1, 1])
                push!(tab, (gcol, lev, lab, Bc[1], se, Bc[1] / se, des.n))
            end
        end
    end
    nrow(tab) > 0 || throw(ArgumentError("judge_subsample_monotonicity: no subgroup " *
                                         "had enough cases to estimate a first stage"))
    tmin = minimum(tab.t)
    m = nrow(tab)
    p = min(1.0, m * cdf(Normal(), tmin))
    return DiagnosticTest("Judge-design monotonicity check by subsample",
                          "the first-stage leniency coefficient is non-negative in " *
                          "every subsample", tmin, p;
                          method="minimum t over $m first stages; one-sided, " *
                                 "Bonferroni-adjusted",
                          note="Checks one implication of monotonicity; positive " *
                               "first stages in every subsample do not establish " *
                               "monotonicity.",
                          details=(table=tab,))
end
