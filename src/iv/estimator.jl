# Two-stage least squares on FixedEffectModels, with honest estimand labelling.

"""
    FirstStageResult

First-stage regression of one endogenous regressor on the excluded instruments.

The object summarizes the reduced-form relation ``D = Z\\pi + W\\delta + v`` between one
endogenous regressor ``D`` and the excluded instruments ``Z`` after the included
exogenous regressors ``W``, the absorbed fixed effects and the weights have been
partialled out (Frisch–Waugh–Lovell). It is the building block of the weak-instrument
diagnostics in [`WeakIVDiagnostics`](@ref): relevance, ``\\pi \\neq 0``, is the one IV
assumption the data speak to directly, and the strength of the first stage governs the
quality of the normal approximation to the 2SLS estimator (Staiger and Stock 1997;
Andrews, Stock and Sun 2019).

The Wald statistic `F` uses the covariance estimator of the IV model
(heteroskedasticity- or cluster-robust when requested) and is therefore the robust
first-stage F; `F_homoskedastic` is the conventional statistic that assumes
homoskedastic, serially uncorrelated errors. With one instrument the robust F equals
the Montiel Olea–Pflueger effective F. With several endogenous regressors the
individual F statistics do not measure identification of the full coefficient vector;
the Sanderson–Windmeijer conditional F does, for each regressor given the others.

# Fields
- `endogenous::Symbol`: the endogenous regressor.
- `instruments::Vector{Symbol}`: the excluded instruments.
- `coef::Vector{Float64}`, `vcov::Matrix{Float64}`: first-stage coefficients on the
  excluded instruments and their covariance, of the same type as the IV model's.
- `F::Float64`, `F_pvalue::Float64`: Wald F for the joint exclusion of the instruments
  from the first stage, computed with the model's covariance estimator, and its
  p-value from ``F(k, \\text{dof})``.
- `F_homoskedastic::Float64`: the conventional (non-robust) first-stage F.
- `dof::Tuple{Int,Float64}`: numerator and denominator degrees of freedom of `F`
  (the denominator is ``G - 1`` under clustering).
- `partial_r2::Float64`: partial ``R^2`` of the excluded instruments.
- `sanderson_windmeijer_F::Float64`: the conditional first-stage F of Sanderson and
  Windmeijer (2016) in its homoskedastic form; it equals `F_homoskedastic` when there
  is one endogenous regressor.

# References
- Staiger, D., & Stock, J. H. (1997). Instrumental variables regression with weak
  instruments. *Econometrica*, 65(3), 557–586.
- Sanderson, E., & Windmeijer, F. (2016). A weak instrument F-test in linear IV
  models with multiple endogenous variables. *Journal of Econometrics*, 190(2),
  212–221.
- Andrews, I., Stock, J. H., & Sun, L. (2019). Weak instruments in instrumental
  variables regression: Theory and practice. *Annual Review of Economics*, 11,
  727–753.
"""
struct FirstStageResult
    endogenous::Symbol
    instruments::Vector{Symbol}
    coef::Vector{Float64}
    vcov::Matrix{Float64}
    F::Float64
    F_pvalue::Float64
    F_homoskedastic::Float64
    dof::Tuple{Int,Float64}
    partial_r2::Float64
    sanderson_windmeijer_F::Float64
end

"""
    WeakIVDiagnostics

Weak-instrument diagnostics of a linear IV model (see
[`first_stage_diagnostics`](@ref)).

The object collects the first-stage statistics that practitioners report to judge
whether conventional 2SLS inference is reliable. Per endogenous regressor it holds a
[`FirstStageResult`](@ref); jointly it holds the Cragg–Donald minimum-eigenvalue F
(homoskedastic) and the Kleibergen–Paap rk Wald F (robust to the model's covariance
type), which test the rank condition for the full coefficient vector. With one
endogenous regressor it also holds the effective F of Montiel Olea and Pflueger
(2013) with its simplified critical values. The effective F is the pre-test
recommended under heteroskedasticity, clustering or serial correlation (Andrews,
Stock and Sun 2019), because the Stock and Yogo (2005) critical values are derived
for homoskedastic errors and are not valid for robust F statistics.

A large first-stage statistic does not validate the instrument: it speaks to
relevance only, never to exclusion or independence. A small one does not call for
discarding the specification, since screening on the first stage distorts
inference (Andrews, Stock and Sun 2019); it calls for inference that is robust to
weak identification ([`weak_iv_confidence_set`](@ref), [`tf_confint`](@ref)).

# Fields
- `first_stage::Vector{FirstStageResult}`: one entry per endogenous regressor.
- `cragg_donald_F::Float64`: Cragg–Donald minimum-eigenvalue F (homoskedastic).
- `kleibergen_paap_F::Float64`: Kleibergen–Paap rk Wald F computed by
  FixedEffectModels with the model's covariance type.
- `effective_F::Union{Nothing,Float64}`: Montiel Olea–Pflueger effective F
  ``F_{\\text{eff}} = \\hat\\pi'Q\\hat\\pi / \\operatorname{tr}(\\hat V Q)``, with
  ``Q = Z'Z`` of the partialled instruments (one endogenous regressor only).
- `op_critical_values`: `nothing` or a `NamedTuple` `(tau_5, tau_10, tau_20, tau_30,
  K_eff)` of the simplified (conservative) Montiel Olea–Pflueger critical values for a
  5%-level test of the null that the worst-case Nagar bias of 2SLS exceeds a fraction
  τ of a benchmark bias, together with the effective degrees of freedom
  ``K_{\\text{eff}}``.
- `stock_yogo`: `nothing` or a `NamedTuple` with the Stock–Yogo (2005) critical values
  for the Cragg–Donald F: `size` (maximal size 10/15/20/25% of a nominal 5% Wald
  test) and `bias` (relative bias 5/10/20/30%, available with at least three
  instruments). They are valid only under homoskedastic, serially uncorrelated errors,
  with one endogenous regressor and up to 10 instruments.
- `n_instruments::Int`, `n_endogenous::Int`: dimensions of the model.
- `vcov_type::String`: covariance estimator behind the robust statistics.

# References
- Stock, J. H., & Yogo, M. (2005). Testing for weak instruments in linear IV
  regression. In D. W. K. Andrews & J. H. Stock (Eds.), *Identification and
  Inference for Econometric Models: Essays in Honor of Thomas Rothenberg*
  (pp. 80–108). Cambridge University Press.
- Kleibergen, F., & Paap, R. (2006). Generalized reduced rank tests using the
  singular value decomposition. *Journal of Econometrics*, 133(1), 97–126.
- Montiel Olea, J. L., & Pflueger, C. (2013). A robust test for weak instruments.
  *Journal of Business & Economic Statistics*, 31(3), 358–369.
- Andrews, I., Stock, J. H., & Sun, L. (2019). Weak instruments in instrumental
  variables regression: Theory and practice. *Annual Review of Economics*, 11,
  727–753.
"""
struct WeakIVDiagnostics
    first_stage::Vector{FirstStageResult}
    cragg_donald_F::Float64
    kleibergen_paap_F::Float64
    effective_F::Union{Nothing,Float64}
    op_critical_values::Union{Nothing,NamedTuple}
    stock_yogo::Union{Nothing,NamedTuple}
    n_instruments::Int
    n_endogenous::Int
    vcov_type::String
end

"""
    IVEstimate <: CausalEstimate

Result of [`iv_regression`](@ref) and [`late_2sls`](@ref): a two-stage least squares
fit together with its first-stage diagnostics and a description of its estimand.

Coefficients are ordered with the endogenous regressors first, followed by the
included exogenous regressors (covariates, and the intercept when no fixed effects are
absorbed), so that `estimate(r)` is the coefficient on the first endogenous regressor.
`vcov(r)` is the full covariance matrix computed by `FixedEffectModels.reg`. The
fields `estimand` and `estimand_note` state what the coefficient identifies in the
specification at hand under heterogeneous treatment effects (a LATE, an average
causal response, or a weighted average of such parameters) and under which
assumptions; see [`late_2sls`](@ref) for the full taxonomy. The object supports the
`StatsAPI` accessors `coef`, `vcov`, `stderror`, `confint`, `coeftable`, `nobs` and
`dof_residual`, and it is the input to [`weak_iv_test`](@ref),
[`weak_iv_confidence_set`](@ref), [`tf_confint`](@ref),
[`overidentification_test`](@ref), [`endogeneity_test`](@ref) and
[`plausibly_exogenous`](@ref).

# Fields
- `coef::Vector{Float64}`, `vcov::Matrix{Float64}`, `coefnames::Vector{String}`:
  coefficients, their covariance and names.
- `nobs::Int`: number of observations used.
- `dof_residual::Float64`: residual degrees of freedom (``G - 1`` under clustering).
- `outcome::Symbol`, `endogenous`, `instruments`, `covariates`, `fe`
  (`Vector{Symbol}`), `weights::Union{Nothing,Symbol}`: the specification.
- `vcov_type::String`: the covariance estimator, e.g.
  `"heteroskedasticity-robust (HC1)"`.
- `level::Float64`: default confidence level of `confint` and of printing.
- `estimand::String`, `estimand_note::String`: what 2SLS identifies in this
  specification and under which assumptions.
- `first_stage::WeakIVDiagnostics`: first-stage and weak-instrument statistics.
- `model`: the underlying `FixedEffectModels.FixedEffectModel`.
- `design`: the internal partialled-out design reused by the diagnostics (not part of
  the public interface).
"""
struct IVEstimate <: CausalEstimate
    coef::Vector{Float64}
    vcov::Matrix{Float64}
    coefnames::Vector{String}
    nobs::Int
    dof_residual::Float64
    outcome::Symbol
    endogenous::Vector{Symbol}
    instruments::Vector{Symbol}
    covariates::Vector{Symbol}
    fe::Vector{Symbol}
    weights::Union{Nothing,Symbol}
    vcov_type::String
    level::Float64
    estimand::String
    estimand_note::String
    first_stage::WeakIVDiagnostics
    model::FixedEffectModels.FixedEffectModel
    design::_IVDesign
end

StatsAPI.coef(r::IVEstimate) = r.coef
StatsAPI.vcov(r::IVEstimate) = r.vcov
StatsAPI.coefnames(r::IVEstimate) = r.coefnames
StatsAPI.nobs(r::IVEstimate) = r.nobs
StatsAPI.dof_residual(r::IVEstimate) = r.dof_residual
StatsAPI.confint(r::IVEstimate; level::Real=r.level) =
    invoke(StatsAPI.confint, Tuple{CausalEstimate}, r; level=level)
estimand(r::IVEstimate) = r.estimand
method_name(r::IVEstimate) = "2SLS"

"""
    iv_regression(data, outcome, endogenous, instruments;
                  covariates=Symbol[], fe=Symbol[], weights=nothing,
                  cluster=nothing, vcov=nothing, level=0.95,
                  drop_singletons=true) -> IVEstimate

Linear instrumental-variables regression estimated by two-stage least squares (2SLS).

The function fits the linear model ``Y = D'\\beta + W'\\gamma + \\alpha + u`` in which the
regressors ``D`` are endogenous, ``E[Du] \\neq 0``, using excluded instruments ``Z``
that satisfy the moment condition ``E[Zu] = 0`` and the rank condition that
``E[ZD']`` (after partialling out ``W`` and the fixed effects ``\\alpha``) has full
column rank. Several endogenous regressors and instruments, exogenous covariates,
high-dimensional fixed effects absorbed by `FixedEffectModels.reg`, and analytic
weights are supported. In a constant-coefficient model ``\\beta`` is the structural
causal effect; with heterogeneous effects and a single endogenous regressor the 2SLS
coefficient is instead a weighted average of local effects whose interpretation
depends on the instruments and controls, as described for [`late_2sls`](@ref) and
recorded in `estimand(r)`. With several endogenous regressors no general LATE-type
interpretation is available.

The estimator is ``\\hat\\beta = (\\hat D'\\tilde D)^{-1}\\hat D'\\tilde Y``, where the
tilde denotes residualization on the controls and ``\\hat D = P_{\\tilde Z}\\tilde D``
the first-stage fitted values. It is consistent and asymptotically normal for a fixed
number of instruments that are strong in the sense of Staiger and Stock (1997), and it
is biased towards OLS in finite samples, increasingly so with many or weak
instruments (Bound, Jaeger and Baker 1995). The covariance is homoskedastic,
heteroskedasticity-robust (HC1, the default) or one- or multiway cluster-robust with
reference distribution ``t(G - 1)``. Conventional t-based intervals can be badly
undersized when the first stage is weak, and Young (2022) documents in a large
sample of published studies that clustered and robust 2SLS inference is sensitive to
a few observations or clusters; Keane and Neal (2023) show that the 2SLS t-test
has a power asymmetry: it rejects spuriously often when the estimate is biased
towards OLS, because the estimated standard error is then too small.

In practice, report the first-stage statistics stored in `r.first_stage`
([`WeakIVDiagnostics`](@ref), in particular the Montiel Olea–Pflueger effective F),
and accompany the Wald interval with an interval that is robust to weak
identification: the Anderson–Rubin set of [`weak_iv_confidence_set`](@ref), valid
whatever the strength of the instruments, or, with one instrument, the tF interval of
[`tf_confint`](@ref). With many instruments prefer [`kclass_iv`](@ref) (LIML, HFUL) or
[`jive`](@ref); overidentifying restrictions and the exogeneity of ``D`` can be
examined with [`overidentification_test`](@ref) and [`endogeneity_test`](@ref).

# Arguments
- `data::AbstractDataFrame`: the data; rows with a missing value in any column used by
  the specification are dropped.
- `outcome::Symbol`: the dependent variable ``Y``.
- `endogenous`: the endogenous regressor(s) ``D``, a `Symbol` or a vector of them.
- `instruments`: the excluded instrument(s) ``Z``, a `Symbol` or a vector; there must
  be at least as many instruments as endogenous regressors.

# Keywords
- `covariates::Vector{Symbol}`: included exogenous regressors ``W`` (default none);
  categorical columns are dummy-coded.
- `fe::Vector{Symbol}`: fixed effects to absorb (default none); with fixed effects the
  intercept is not reported.
- `weights::Union{Nothing,Symbol}`: column of strictly positive analytic or
  probability weights (default `nothing`, unweighted).
- `cluster`: a `Symbol` or vector of `Symbol`s for one- or multiway cluster-robust
  covariance (default `nothing`).
- `vcov`: `Vcov.simple()`, `Vcov.robust()` or `Vcov.cluster(...)`; overrides `cluster`.
  The default `nothing` means `Vcov.robust()` (HC1) unless `cluster` is given.
- `level::Real`: confidence level used by `confint(r)` and printing (default 0.95).
- `drop_singletons::Bool`: drop singleton fixed-effect groups, as `reg` does (default
  `true`); singletons carry no within-group information but inflate the degrees of
  freedom.

# Returns
- An [`IVEstimate`](@ref); `estimate(r)` is the coefficient on the first endogenous
  regressor, `coeftable(r)` the full table, `r.first_stage` the weak-instrument
  diagnostics and `estimand(r)` the estimand label.

# Examples
```julia
using DrSnow, DataFrames, StableRNGs
rng = StableRNG(1)
n = 2_000
z1, z2, w = randn(rng, n), randn(rng, n), randn(rng, n)
v = randn(rng, n)
d = 0.5 .* z1 .+ 0.3 .* z2 .+ 0.4 .* w .+ v
y = 1.0 .* d .+ w .+ 0.6 .* v .+ randn(rng, n)        # D is endogenous through v
df = DataFrame(y=y, d=d, z1=z1, z2=z2, w=w, g=rand(rng, 1:50, n))
r = iv_regression(df, :y, :d, [:z1, :z2]; covariates=[:w], cluster=:g)
coeftable(r)
r.first_stage                   # effective F, Kleibergen–Paap F, ...
weak_iv_confidence_set(r)       # Anderson–Rubin set
```

# References
- Staiger, D., & Stock, J. H. (1997). Instrumental variables regression with weak
  instruments. *Econometrica*, 65(3), 557–586.
- Bound, J., Jaeger, D. A., & Baker, R. M. (1995). Problems with instrumental
  variables estimation when the correlation between the instruments and the
  endogenous explanatory variable is weak. *Journal of the American Statistical
  Association*, 90(430), 443–450.
- Angrist, J. D., & Pischke, J.-S. (2009). *Mostly Harmless Econometrics: An
  Empiricist's Companion*. Princeton University Press.
- Andrews, I., Stock, J. H., & Sun, L. (2019). Weak instruments in instrumental
  variables regression: Theory and practice. *Annual Review of Economics*, 11,
  727–753.
- Young, A. (2022). Consistency without inference: Instrumental variables in
  practical application. *European Economic Review*, 147, 104112.
- Keane, M., & Neal, T. (2023). Instrument strength in IV estimation and inference: A
  guide to theory and practice. *Journal of Econometrics*, 235(2), 1625–1653.
"""
function iv_regression(data::AbstractDataFrame, outcome::Symbol, endogenous, instruments;
                       covariates=Symbol[], fe=Symbol[],
                       weights::Union{Nothing,Symbol}=nothing,
                       cluster=nothing, vcov=nothing, level::Real=0.95,
                       drop_singletons::Bool=true)
    endo = _as_symbols(endogenous)
    inst = _as_symbols(instruments)
    covs = _as_symbols(covariates)
    fes = _as_symbols(fe)
    0 < level < 1 || throw(ArgumentError("level must be in (0, 1), got $level"))
    vce = _iv_vcov_estimator(cluster, vcov)
    _iv_validate_spec(data, outcome, endo, inst, covs, fes, weights, vce)

    f = make_formula(outcome, covs; fe=fes, endogenous=endo, instruments=inst)
    m = FixedEffectModels.reg(data, f, vce; weights=weights, tol=_IV_FE_TOL,
                              maxiter=100_000, drop_singletons=drop_singletons,
                              progress_bar=false)
    esample = BitVector(m.esample)
    used = unique(vcat(outcome, endo, inst, covs, fes, _iv_cluster_names(vce),
                       weights === nothing ? Symbol[] : [weights]))
    sub = disallowmissing(data[esample, used])
    des = _iv_build_design(sub, outcome, endo, inst, covs, fes, weights, vce, esample)

    # reorder: endogenous first (read by name; errors if dropped as collinear)
    names_m = StatsAPI.coefnames(m)
    idx_endo = [coef_index(m, string(e)) for e in endo]
    idx_rest = [i for i in eachindex(names_m) if !(i in idx_endo) &&
                isfinite(StatsAPI.vcov(m)[i, i]) &&
                !(iszero(StatsAPI.vcov(m)[i, i]) && iszero(StatsAPI.coef(m)[i]))]
    idx = vcat(idx_endo, idx_rest)
    b = StatsAPI.coef(m)[idx]
    V = Matrix(StatsAPI.vcov(m))[idx, idx]

    _iv_check_against_fem(des, b, V, endo)
    fs = _iv_weak_iv_diagnostics(des, endo, inst, m.F_kp)
    est, note = _iv_estimand(sub, endo, inst, covs, fes)
    return IVEstimate(b, V, names_m[idx], StatsAPI.nobs(m),
                      float(StatsAPI.dof_residual(m)), outcome, endo, inst, covs, fes,
                      weights, _iv_vcov_label(des), float(level), est, note, fs, m, des)
end

"""
Guard: the 2SLS coefficients recomputed on the partialled design must match
FixedEffectModels. They differ when `reg` re-classifies an endogenous regressor as
exogenous (it does so, with only an `@info`, when an endogenous variable is
collinear with the instruments, controls and other endogenous regressors).
"""
function _iv_check_against_fem(des::_IVDesign, b, V, endo)
    p = length(endo)
    β, _, _, _ = _iv_tsls(des, des.y, des.D, des.Z)
    tol = 1e-6 .* (abs.(b[1:p]) .+ sqrt.(max.(diag(V)[1:p], 0.0)) .+ 1e-8)
    if any(abs.(β .- b[1:p]) .> tol)
        throw(ArgumentError("the model is not identified as specified: an endogenous " *
                            "regressor is (nearly) collinear with the instruments, " *
                            "controls and other endogenous regressors (e.g. " *
                            "experience = age − education − c with age as an " *
                            "instrument). Respecify the endogenous regressors."))
    end
    return nothing
end

function _iv_validate_spec(data, outcome, endo, inst, covs, fes, weights, vce)
    isempty(endo) && throw(ArgumentError("at least one endogenous regressor is required"))
    isempty(inst) && throw(ArgumentError("at least one instrument is required"))
    length(inst) >= length(endo) ||
        throw(ArgumentError("under-identified: $(length(endo)) endogenous regressor(s) " *
                            "but only $(length(inst)) instrument(s)"))
    roles = vcat([outcome], endo, inst, covs)
    length(unique(roles)) == length(roles) ||
        throw(ArgumentError("the outcome, endogenous regressors, instruments and " *
                            "covariates must be distinct columns"))
    require_columns(data, vcat(roles, fes, _iv_cluster_names(vce),
                               weights === nothing ? Symbol[] : [weights]);
                    context="iv_regression")
    _iv_check_numeric(data, vcat([outcome], endo, inst), "iv_regression")
    if weights !== nothing
        _iv_check_numeric(data, [weights], "iv_regression")
        all(w -> ismissing(w) || w > 0, data[!, weights]) ||
            throw(ArgumentError("iv_regression: weights must be strictly positive"))
    end
    return nothing
end

_iv_isbinary(v) = all(x -> x == 0 || x == 1, v) && length(unique(v)) == 2

"""Describe what 2SLS identifies in this specification (returns `(label, note)`)."""
function _iv_estimand(sub, endo, inst, covs, fes)
    if length(endo) > 1
        return ("structural coefficients (constant-effects model)",
                "With several endogenous regressors, 2SLS recovers causal parameters " *
                "only in a linear model with constant (or suitably restricted) " *
                "effects; there is no general LATE interpretation.")
    end
    dbin = _iv_isbinary(sub[!, endo[1]])
    zbin = length(inst) == 1 && _iv_isbinary(sub[!, inst[1]])
    nocontrols = isempty(covs) && isempty(fes)
    saturated = isempty(covs) && length(fes) == 1
    assumptions = "independence, exclusion, monotonicity and relevance of the " *
                  "instrument"
    if zbin && dbin && nocontrols
        return ("LATE",
                "Local average treatment effect for compliers (Imbens & Angrist 1994) " *
                "under $assumptions.")
    elseif zbin && dbin && saturated
        return ("weighted average of covariate-cell LATEs",
                "Binary instrument with controls entered as a single set of cell fixed " *
                "effects (saturated): 2SLS is a weighted average of within-cell LATEs " *
                "with non-negative weights proportional to Var(Z | cell) × cell first " *
                "stage, under $assumptions within cells (Angrist & Imbens 1995; " *
                "Blandhol et al. 2026). The weights differ from complier shares; use " *
                "`late_ipw` for the unconditional LATE.")
    elseif zbin && dbin
        return ("covariate-adjusted 2SLS (weighted combination of conditional LATEs)",
                "With non-saturated covariates, 2SLS is a non-negatively weighted " *
                "average of conditional LATEs only if E[Z | X] is linear in the included " *
                "controls; otherwise some weights can be negative (Blandhol, Bonney, " *
                "Mogstad & Torgovitsky 2026; Słoczyński 2020). Consider saturating the " *
                "controls or `late_ipw` (Abadie 2003; Frölich 2007).")
    elseif zbin && nocontrols
        return ("ACR (average causal response)",
                "Non-binary treatment with a binary instrument: under $assumptions, " *
                "2SLS is a weighted average of per-unit causal responses to incremental " *
                "changes in the treatment among units whose treatment is shifted by " *
                "the instrument (Angrist & Imbens 1995).")
    elseif zbin && saturated
        return ("weighted average of covariate-cell ACRs",
                "Non-binary treatment with a binary instrument and controls entered as " *
                "a single set of cell fixed effects (saturated): 2SLS is a weighted " *
                "average of within-cell average causal responses (Angrist & Imbens " *
                "1995), with non-negative weights only if the cell first stages all " *
                "have the same sign, under $assumptions within cells.")
    elseif zbin
        return ("covariate-adjusted 2SLS (weighted combination of conditional ACRs)",
                "Non-binary treatment with a binary instrument: without covariates, " *
                "2SLS is an average causal response (Angrist & Imbens 1995). With " *
                "non-saturated covariates it is a non-negatively weighted average of " *
                "conditional ACRs only if E[Z | X] is linear in the included controls; " *
                "otherwise some weights can be negative (Blandhol, Bonney, Mogstad & " *
                "Torgovitsky 2026; Słoczyński 2020). Consider saturating the controls.")
    else
        base = length(inst) > 1 ?
               "With several instruments, 2SLS is a weighted average of the " *
               "instrument-specific IV estimands" :
               "With a multi-valued instrument, 2SLS is a weighted average of Wald " *
               "estimands between adjacent instrument values"
        cov_note = nocontrols ? "" :
                   " Covariates add the caveat that weights can be negative unless " *
                   "E[Z | X] is linear in the controls (Blandhol et al. 2026)."
        return ("weighted average of LATEs/ACRs",
                base * " (Imbens & Angrist 1994); the weights are non-negative only " *
                "when monotonicity holds for the combined first-stage index " *
                "(Mogstad, Torgovitsky & Walters 2021)." * cov_note)
    end
end

"""
    late_2sls(data, outcome, treatment, instrument;
              covariates=Symbol[], fe=Symbol[], weights=nothing,
              cluster=nothing, vcov=nothing, level=0.95,
              drop_singletons=true) -> IVEstimate

Two-stage least squares for one endogenous treatment, with the estimand described
under heterogeneous treatment effects.

This is [`iv_regression`](@ref) specialized to a single treatment ``D`` and one or more
instruments, the setting of the local average treatment effect (LATE) framework.
Let ``Y_i(d)`` denote potential outcomes and ``D_i(z)`` potential treatments. With a
binary instrument and a binary treatment, Imbens and Angrist (1994) and Angrist,
Imbens and Rubin (1996) show that under (i) independence, ``(Y_i(0), Y_i(1), D_i(0),
D_i(1)) \\perp Z_i``, (ii) exclusion, that ``Z`` affects ``Y`` only through ``D``,
(iii) monotonicity, ``D_i(1) \\ge D_i(0)`` for all ``i`` (no defiers), and (iv)
relevance, ``P(D_i(1) > D_i(0)) > 0``, the Wald ratio identifies the average effect
for compliers,

```math
\\beta_{\\text{IV}}
  = \\frac{E[Y \\mid Z=1] - E[Y \\mid Z=0]}{E[D \\mid Z=1] - E[D \\mid Z=0]}
  = E[Y_i(1) - Y_i(0) \\mid D_i(1) > D_i(0)] .
```

Compliers are not identifiable individually, and the LATE is specific to the
instrument: a different instrument moves a different subpopulation. Relevance is
testable ([`first_stage_diagnostics`](@ref)); independence and exclusion are not,
although they have testable implications (joint with monotonicity) that
[`instrument_validity_test`](@ref) and [`huber_mellace_test`](@ref) examine, and
[`plausibly_exogenous`](@ref) quantifies the consequences of violations.

The reported estimand, `estimand(r)` with explanation in `r.estimand_note`, depends on
the specification. With a binary treatment, a binary instrument and no controls it is
the LATE above. With a multi-valued treatment ``D \\in \\{0, 1, \\dots, J\\}`` and a
binary instrument, Angrist and Imbens (1995) show that 2SLS identifies the average
causal response (ACR), ``\\sum_{j=1}^J \\omega_j E[Y_i(j) - Y_i(j-1) \\mid D_i(1) \\ge j
> D_i(0)]`` with weights ``\\omega_j \\propto P(D_i(1) \\ge j > D_i(0))`` that are
non-negative and sum to one under monotonicity; it averages unit causal responses
along the part of the treatment scale that the instrument shifts and counts units
whose treatment moves by several units several times. When the controls enter as one
set of cell fixed effects (saturated), 2SLS with the instrument entered once is a
weighted average of within-cell LATEs with weights proportional to ``\\operatorname{Var}(Z
\\mid X) \\cdot \\pi(X)``, where ``\\pi(X)`` is the cell first stage (Angrist and Imbens
1995); these weights are non-negative only if the cell first stages all have the same
sign, which holds under the unconditional monotonicity ``D_i(1) \\ge D_i(0)`` but fails
if there are compliers but no defiers in some cells and defiers but no compliers in
others (Słoczyński 2020). The weights differ from complier shares, so the estimand
is not the unconditional LATE. With covariates that are not saturated, Blandhol,
Bonney, Mogstad and Torgovitsky (2026) show that 2SLS is a non-negatively weighted
average of conditional LATEs essentially only when ``E[Z \\mid X]`` is linear in the
included controls; otherwise some weights can be negative and the coefficient need
not lie between the smallest and the largest conditional effect. With a multi-valued
instrument or several instruments, 2SLS is a weighted average of Wald estimands of
pairs of instrument values (Imbens and Angrist 1994); with several instruments
Mogstad, Torgovitsky and Walters (2021) show that the conventional monotonicity
condition is plausible only if choice behaviour is effectively homogeneous, and they
give empirically verifiable conditions for non-negative weights under the weaker
partial monotonicity ([`multiple_iv_weights`](@ref)).

Inference is as in [`iv_regression`](@ref): a Wald interval with the chosen covariance
(HC1 by default), and robust alternatives in [`tf_confint`](@ref) and
[`weak_iv_confidence_set`](@ref). When the instrument is valid only conditionally on
covariates and the unconditional LATE is the target, use the κ-weighting estimator
[`late_ipw`](@ref) (Abadie 2003) or, with high-dimensional covariates,
the DML estimators of the machine-learning area. Describe the complier population with
[`estimate_compliance`](@ref) and [`complier_characteristics`](@ref) so that readers can
judge to whom the LATE applies.

# Arguments
- `data::AbstractDataFrame`: the data; rows with missing values in used columns are
  dropped.
- `outcome::Symbol`: the outcome ``Y``.
- `treatment::Symbol`: the endogenous treatment ``D``, binary (0/1) for a LATE or
  ordered and multi-valued for an ACR.
- `instrument`: the excluded instrument(s) ``Z``, a `Symbol` or a vector of `Symbol`s.

# Keywords
- `covariates`, `fe`, `weights`, `cluster`, `vcov`, `level`, `drop_singletons`: as
  in [`iv_regression`](@ref), with the same defaults (no controls, unweighted, HC1
  covariance, `level = 0.95`, singletons dropped). How the controls enter changes the
  estimand, as described above.

# Returns
- An [`IVEstimate`](@ref); `estimate(r)` is the 2SLS coefficient on `treatment`,
  `estimand(r)` and `r.estimand_note` describe what it identifies.

# Examples
```julia
using DrSnow, DataFrames, StableRNGs
rng = StableRNG(1)
n = 2_000
z = rand(rng, 0:1, n)                                    # randomized offer
u = rand(rng, n)                                         # compliance type
d = ifelse.(u .< 0.2, 1, ifelse.(u .< 0.7, z, 0))        # 20% AT, 50% C, 30% NT
y = 1.0 .+ 2.0 .* d .+ 0.8 .* (u .< 0.2) .+ randn(rng, n)
df = DataFrame(y=y, d=d, z=z, site=rand(rng, 1:40, n))
r = late_2sls(df, :y, :d, :z; cluster=:site)
estimate(r), confint(r)
estimand(r)                       # "LATE"
tf_confint(r)                     # Lee et al. (2022) tF interval
weak_iv_confidence_set(r)         # Anderson–Rubin set
```

# References
- Imbens, G. W., & Angrist, J. D. (1994). Identification and estimation of local
  average treatment effects. *Econometrica*, 62(2), 467–475.
- Angrist, J. D., & Imbens, G. W. (1995). Two-stage least squares estimation of
  average causal effects in models with variable treatment intensity. *Journal of
  the American Statistical Association*, 90(430), 431–442.
- Angrist, J. D., Imbens, G. W., & Rubin, D. B. (1996). Identification of causal
  effects using instrumental variables. *Journal of the American Statistical
  Association*, 91(434), 444–455.
- Blandhol, C., Bonney, J., Mogstad, M., & Torgovitsky, A. (2026). When is TSLS
  actually LATE? *Review of Economic Studies*, advance online publication.
  https://doi.org/10.1093/restud/rdag029
- Słoczyński, T. (2020). When should we (not) interpret linear IV estimands as LATE?
  arXiv:2011.06695.
- Mogstad, M., Torgovitsky, A., & Walters, C. R. (2021). The causal interpretation of
  two-stage least squares with multiple instrumental variables. *American Economic
  Review*, 111(11), 3663–3698.
- Abadie, A. (2003). Semiparametric instrumental variable estimation of treatment
  response models. *Journal of Econometrics*, 113(2), 231–263.
"""
function late_2sls(data::AbstractDataFrame, outcome::Symbol, treatment::Symbol,
                   instrument; kwargs...)
    return iv_regression(data, outcome, [treatment], instrument; kwargs...)
end

function show_details(io::IO, r::IVEstimate)
    println(io)
    println(io, "Covariance: ", r.vcov_type, "; residual dof: ",
            @sprintf("%g", r.dof_residual))
    fs = r.first_stage
    for s in fs.first_stage
        @printf(io, "First stage (%s): F = %.2f (%s), partial R² = %.4f\n",
                s.endogenous, s.F, r.vcov_type, s.partial_r2)
    end
    if fs.effective_F !== nothing
        _iv_printf(io, "Olea–Pflueger effective F = %.2f (5%% critical value " *
                "for τ = 10%%: " *
                "%.2f)\n", fs.effective_F, fs.op_critical_values.tau_10)
    end
    if length(r.endogenous) > 1 || length(r.instruments) > 1
        @printf(io, "Kleibergen–Paap rk Wald F = %.2f; Cragg–Donald F = %.2f\n",
                fs.kleibergen_paap_F, fs.cragg_donald_F)
    end
    println(io, "Estimand: ", r.estimand)
    println(io, "Note: ", r.estimand_note)
end
