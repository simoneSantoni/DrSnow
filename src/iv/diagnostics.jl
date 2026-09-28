# Falsification and specification tests for IV designs. Every function returns a
# `DiagnosticTest`; non-rejection is never reported as evidence that an assumption
# holds, and untestable assumptions are never described as tested.

"""Design for OLS of `lhs` columns on `regs` (+ controls) on a complete-case sample."""
function _iv_ols_design(data::AbstractDataFrame, lhs::Vector{Symbol},
                        regs::Vector{Symbol}, covariates::Vector{Symbol},
                        fe::Vector{Symbol}, weights, vce, context::String;
                        rows::Union{Nothing,AbstractVector{Bool}}=nothing)
    cols = unique(vcat(lhs, regs, covariates, fe, _iv_cluster_names(vce),
                       weights === nothing ? Symbol[] : [weights]))
    require_columns(data, cols; context=context)
    _iv_check_numeric(data, vcat(lhs, regs), context)
    mask = trues(nrow(data))
    rows === nothing || (mask .&= rows)
    for c in cols
        mask .&= .!ismissing.(data[!, c])
    end
    weights === nothing || (mask .&= coalesce.(data[!, weights] .> 0, false))
    sub = disallowmissing(data[mask, cols])
    nrow(sub) > length(regs) + length(covariates) + 1 ||
        throw(ArgumentError("$context: too few complete observations ($(nrow(sub)))"))
    return _iv_build_design(sub, lhs[1], lhs, regs, covariates, fe, weights, vce,
                            BitVector(mask); check_endogenous=false)
end

_iv_rows(data, subset::Symbol) = coalesce.(data[!, subset] .== true, false)
function _iv_rows(data, subset::AbstractVector)
    length(subset) == nrow(data) ||
        throw(DimensionMismatch("subset must have one entry per row of data"))
    return coalesce.(subset .== true, false)
end

# ---------------------------------------------------------------------------
# Instrument balance / placebo outcomes
# ---------------------------------------------------------------------------

"""
    instrument_balance(data, instrument, variables; covariates=Symbol[], fe=Symbol[],
                       weights=nothing, cluster=nothing, vcov=nothing)
        -> DiagnosticTest

Joint test that the instrument(s) are unrelated to pre-determined variables or
placebo outcomes.

The independence assumption of IV, ``(Y_i(\\cdot), D_i(\\cdot)) \\perp Z_i`` (possibly
conditional on controls), is not testable directly, but it implies that the
instrument is unrelated to any characteristic fixed before the instrument was
assigned, and to placebo outcomes that neither the instrument nor the treatment can
affect. A balance test of this implication is the IV analogue of the covariate balance
table of a randomized experiment and is routinely reported in natural-experiment
studies (Angrist and Pischke 2009; Imbens 2014).

Each variable in `variables` is regressed on the instruments and the controls
(`covariates`, absorbed fixed effects `fe`, weights). The Wald statistic tests that
all instrument coefficients in all equations are zero, using the joint cross-equation
covariance of the coefficients (heteroskedasticity-robust HC1 by default, or
cluster-robust), so that correlation among the variables is accounted for; the
reference distribution is ``F(q, \\text{dof})`` with ``q`` the number of instruments
times the number of variables and denominator degrees of freedom ``G - 1`` under
clustering.

A rejection is evidence against as-good-as-random assignment of the instrument given
the controls. Non-rejection is not evidence of independence: balance on observed
characteristics says nothing about unobserved ones, and the test has limited power
with many variables. No balance test can speak to the exclusion restriction, which
concerns the channels through which the instrument affects the outcome; see
[`zero_first_stage_test`](@ref) and [`plausibly_exogenous`](@ref) for that.

# Arguments
- `data::AbstractDataFrame`: the data; incomplete rows are dropped.
- `instrument`: the instrument(s), a `Symbol` or a vector of `Symbol`s.
- `variables`: numeric pre-determined characteristics or placebo outcomes (a `Symbol`
  or vector).

# Keywords
- `covariates::Vector{Symbol}`: controls conditional on which the instrument is
  assumed to be as good as random (default none).
- `fe::Vector{Symbol}`: fixed effects to absorb, e.g. the strata of a stratified
  lottery (default none).
- `weights::Union{Nothing,Symbol}`: analytic weights (default `nothing`).
- `cluster`: clustering variable(s) (default `nothing`).
- `vcov`: an explicit `Vcov` estimator that overrides `cluster` (default HC1).

# Returns
- A [`DiagnosticTest`](@ref); `pvalue(t)` is the joint p-value and `t.details.table`
  holds the per-variable, per-instrument coefficients, standard errors and
  p-values; `t.details.nobs` the sample size.

# Examples
```julia
using DrSnow, DataFrames, StableRNGs
rng = StableRNG(4)
n = 2_000
df = DataFrame(z=rand(rng, 0:1, n), age=rand(rng, 20:60, n),
               female=rand(rng, 0:1, n), prior=randn(rng, n),
               school=rand(rng, 1:50, n))
instrument_balance(df, :z, [:age, :female, :prior]; cluster=:school)
```

# References
- Angrist, J. D., & Pischke, J.-S. (2009). *Mostly Harmless Econometrics: An
  Empiricist's Companion*. Princeton University Press.
- Imbens, G. W. (2014). Instrumental variables: An econometrician's perspective.
  *Statistical Science*, 29(3), 323–358.
"""
function instrument_balance(data::AbstractDataFrame, instrument, variables;
                            covariates=Symbol[], fe=Symbol[],
                            weights::Union{Nothing,Symbol}=nothing, cluster=nothing,
                            vcov=nothing)
    inst = _as_symbols(instrument)
    vars = _as_symbols(variables)
    isempty(vars) && throw(ArgumentError("instrument_balance: no variables given"))
    vce = _iv_vcov_estimator(cluster, vcov)
    des = _iv_ols_design(data, vars, inst, _as_symbols(covariates), _as_symbols(fe),
                         weights, vce, "instrument_balance")
    k, m = length(inst), length(vars)
    B, V, _ = _iv_ols(des, des.D, des.Z)
    b = vec(B)
    F = _iv_wald_F(b, V)
    dof = _iv_ref_dof(des, k)
    q = k * m
    se = sqrt.(max.(diag(V), 0.0))
    tab = DataFrame(variable=repeat(vars; inner=k), instrument=repeat(inst; outer=m),
                    coef=b, se=se, pvalue=two_sided_pvalue.(b ./ se, dof))
    return DiagnosticTest("Instrument balance test",
                          "the instrument(s) are unrelated to " * join(vars, ", ") *
                          " (conditional on controls)", F, ccdf(FDist(q, dof), F);
                          dof=(q, dof), method="joint Wald, " * _iv_vcov_label(des),
                          note="Non-rejection does not show that the instrument is " *
                               "as good as randomly assigned, and balance says nothing " *
                               "about the exclusion restriction.",
                          details=(table=tab, nobs=des.n))
end

# ---------------------------------------------------------------------------
# First-stage sign reversal (monotonicity falsification)
# ---------------------------------------------------------------------------

"""
    first_stage_sign_test(data, treatment, instrument, subgroups;
                          covariates=Symbol[], fe=Symbol[], weights=nothing,
                          cluster=nothing, vcov=nothing, min_obs=30)
        -> DiagnosticTest

Falsification test of monotonicity: is the first stage of the opposite sign in some
subgroup?

Monotonicity, ``D_i(1) \\ge D_i(0)`` for all ``i`` (no defiers; Imbens and Angrist
1994), is what makes the IV estimand a non-negatively weighted average of treatment
effects. It is not testable in general, but together with independence and exclusion
holding within subgroups it implies that the first stage
``\\pi_g = E[D \\mid Z = 1, g] - E[D \\mid Z = 0, g]`` has the same sign as the pooled
first stage (or is zero) in every subgroup ``g`` defined by pre-determined
characteristics. A sign reversal indicates defiers in that subgroup. It also matters
for estimation with covariates: when there are compliers but no defiers in some
covariate cells and defiers but no compliers in others, 2SLS with the instrument
entered once places negative weight on some conditional LATEs (Słoczyński 2020).

For each level of each column in `subgroups`, the treatment is regressed on the
instrument and the controls within the subgroup, and the one-sided null
``H_0: s \\cdot \\pi_g \\ge 0`` (``s`` the sign of the pooled first stage) is tested
with a t test (``t(G-1)`` reference under clustering, normal otherwise). The one-sided
p-values are adjusted for multiplicity with Holm's (1979) step-down method, which
controls the family-wise error rate; the reported statistic is the smallest oriented
t-statistic and the reported p-value the smallest Holm-adjusted p-value.

A rejection is evidence of defiers in at least one subgroup (given independence and
exclusion within subgroups). Non-rejection does not establish monotonicity: defiers
can coexist with a net first stage of the pooled sign within every subgroup, and the
test has little power in small subgroups. Kitagawa's
[`instrument_validity_test`](@ref) and the [`huber_mellace_test`](@ref) use the
outcome distribution to detect other violations.

# Arguments
- `data::AbstractDataFrame`: the data.
- `treatment::Symbol`: the treatment (numeric; binary in the LATE setting).
- `instrument::Symbol`: a single instrument.
- `subgroups`: column(s) defining the subgroups (a `Symbol` or vector, any element
  type); each distinct non-missing level of each column is one subgroup, so the
  columns are examined one at a time rather than crossed.

# Keywords
- `covariates`, `fe`, `weights`, `cluster`, `vcov`: controls and covariance, as in
  [`instrument_balance`](@ref) (default no controls, HC1).
- `min_obs::Integer`: minimum number of rows in a subgroup (default 30); smaller
  subgroups, and subgroups in which the regression cannot be run (e.g. no variation
  in the instrument), are skipped and listed in `details.skipped`.

# Returns
- A [`DiagnosticTest`](@ref); `t.details.table` lists, per subgroup, the number of
  observations, the first stage, its standard error, the oriented t-statistic, and
  the one-sided and Holm-adjusted p-values; `t.details.pooled_first_stage` is the
  pooled coefficient.

# Examples
```julia
using DrSnow, DataFrames, StableRNGs
rng = StableRNG(5)
n = 3_000
female = rand(rng, 0:1, n)
region = rand(rng, 1:4, n)
z = rand(rng, 0:1, n)
d = Int.(rand(rng, n) .< 0.2 .+ 0.4 .* z)
df = DataFrame(d=d, z=z, female=female, region=region)
first_stage_sign_test(df, :d, :z, [:female, :region])
```

# References
- Imbens, G. W., & Angrist, J. D. (1994). Identification and estimation of local
  average treatment effects. *Econometrica*, 62(2), 467–475.
- Angrist, J. D., Imbens, G. W., & Rubin, D. B. (1996). Identification of causal
  effects using instrumental variables. *Journal of the American Statistical
  Association*, 91(434), 444–455.
- Holm, S. (1979). A simple sequentially rejective multiple test procedure.
  *Scandinavian Journal of Statistics*, 6(2), 65–70.
- Słoczyński, T. (2020). When should we (not) interpret linear IV estimands as LATE?
  arXiv:2011.06695.
"""
function first_stage_sign_test(data::AbstractDataFrame, treatment::Symbol,
                               instrument::Symbol, subgroups; covariates=Symbol[],
                               fe=Symbol[], weights::Union{Nothing,Symbol}=nothing,
                               cluster=nothing, vcov=nothing, min_obs::Integer=30)
    sg = _as_symbols(subgroups)
    isempty(sg) && throw(ArgumentError("first_stage_sign_test: no subgroup columns"))
    require_columns(data, sg; context="first_stage_sign_test")
    covs, fes = _as_symbols(covariates), _as_symbols(fe)
    vce = _iv_vcov_estimator(cluster, vcov)
    ctx = "first_stage_sign_test"
    pooled = _iv_ols_design(data, [treatment], [instrument], covs, fes, weights, vce, ctx)
    Bp, _, _ = _iv_ols(pooled, pooled.D, pooled.Z)
    s = sign(Bp[1, 1])
    s == 0 && throw(ArgumentError("$ctx: the pooled first stage is exactly zero"))
    rows = NamedTuple[]
    skipped = String[]
    for c in sg
        col = data[!, c]
        for lev in unique(skipmissing(col))
            sel = coalesce.(col .== lev, false)
            label = "$(c) = $(lev)"
            local des
            try
                sum(sel) >= min_obs || error("too few observations")
                des = _iv_ols_design(data, [treatment], [instrument], covs, fes, weights,
                                     vce, ctx; rows=sel)
            catch err
                (err isa ArgumentError || err isa ErrorException) || rethrow()
                push!(skipped, label)
                continue
            end
            B, V, _ = _iv_ols(des, des.D, des.Z)
            se = sqrt(V[1, 1])
            dof = _iv_ref_dof(des, 1)
            t = s * B[1, 1] / se
            p1 = isfinite(dof) ? cdf(TDist(dof), t) : cdf(Normal(), t)
            push!(rows, (subgroup=label, nobs=des.n, first_stage=B[1, 1], se=se,
                         t_oriented=t, pvalue_one_sided=p1))
        end
    end
    isempty(rows) && throw(ArgumentError("$ctx: no subgroup had enough observations"))
    tab = DataFrame(rows)
    m = nrow(tab)
    ord = sortperm(tab.pvalue_one_sided)
    adj = zeros(m)
    running = 0.0
    for (i, j) in enumerate(ord)
        running = max(running, min(1.0, (m - i + 1) * tab.pvalue_one_sided[j]))
        adj[j] = running
    end
    tab.pvalue_holm = adj
    return DiagnosticTest("First-stage sign-reversal test (monotonicity falsification)",
                          "the first stage has the pooled sign ($(s > 0 ? "+" : "−")) " *
                          "or is zero in every subgroup", minimum(tab.t_oriented),
                          minimum(adj); dof=(m,),
                          method="one-sided t tests by subgroup, Holm adjustment, " *
                                 _iv_vcov_label(pooled),
                          note="Statistic = smallest oriented t. A rejection indicates " *
                               "defiers in some subgroup; non-rejection does not " *
                               "establish monotonicity.",
                          details=(table=tab, skipped=skipped, pooled_first_stage=Bp[1, 1]))
end

# ---------------------------------------------------------------------------
# Zero-first-stage reduced-form test (van Kippersluis & Rietveld 2018)
# ---------------------------------------------------------------------------

"""
    zero_first_stage_test(data, outcome, treatment, instrument; subset,
                          covariates=Symbol[], fe=Symbol[], weights=nothing,
                          cluster=nothing, vcov=nothing) -> DiagnosticTest

Falsification test of the exclusion restriction in a subsample in which the
instrument does not move the treatment.

The exclusion restriction states that the instrument affects the outcome only through
the treatment. In a "zero-first-stage" subsample, a group for which the instrument is
known on a priori grounds not to affect the treatment (e.g. units ineligible for the
program the instrument promotes), the reduced-form effect of the instrument on the
outcome must therefore be zero. A non-zero reduced form in that group reveals a direct
effect of the instrument, or a failure of independence, at least for that group (van
Kippersluis and Rietveld 2018). The function regresses the outcome on the
instrument(s) and the controls within `subset` and tests that the instrument
coefficients are jointly zero with a Wald F test (HC1 by default, or cluster-robust;
reference ``F(k, \\text{dof})``).

The test is informative only if the first stage in `subset` really is zero, so the
first-stage F statistic and p-value in that subsample are reported alongside. It
speaks to the exclusion restriction in the subsample; carrying the conclusion to the
rest of the population requires that the direct effect be the same in both groups.
Non-rejection does not establish exclusion. Following van Kippersluis and Rietveld
(2018), the estimated reduced form (in outcome units per unit of the instrument) can
serve as the centre of a prior for the direct effect ``\\gamma`` in
[`plausibly_exogenous`](@ref) (Conley, Hansen and Rossi 2012). With one instrument, a
direct effect ``\\gamma`` biases the IV estimand in the full sample by ``\\gamma / \\pi``,
where ``\\pi`` is the first stage, so its sign relative to the first stage, not its sign
alone, determines the direction of the bias.

# Arguments
- `data::AbstractDataFrame`: the data.
- `outcome::Symbol`: the outcome.
- `treatment::Symbol`: the treatment, used to report the subsample first stage.
- `instrument`: the instrument(s), a `Symbol` or vector.

# Keywords
- `subset`: required; a `Bool` column name or a `Bool` vector with one entry per row
  marking the zero-first-stage subsample (`missing` counts as `false`).
- `covariates`, `fe`, `weights`, `cluster`, `vcov`: controls and covariance as in
  [`instrument_balance`](@ref) (default no controls, HC1).

# Returns
- A [`DiagnosticTest`](@ref) with `details = (reduced_form, reduced_form_se,
  first_stage, first_stage_F, first_stage_pvalue, nobs)`.

# Examples
```julia
using DrSnow, DataFrames, StableRNGs
rng = StableRNG(6)
n = 4_000
eligible = rand(rng, Bool, n)
z = rand(rng, 0:1, n)
d = Int.(eligible .& (rand(rng, n) .< 0.3 .+ 0.4 .* z))   # no first stage if ineligible
y = 1.0 .* d .+ 0.1 .* z .+ randn(rng, n)                  # direct effect of 0.1
df = DataFrame(y=y, d=d, z=z, ineligible=.!eligible)
t = zero_first_stage_test(df, :y, :d, :z; subset=:ineligible)
t.details.reduced_form, t.details.first_stage_pvalue
```

# References
- van Kippersluis, H., & Rietveld, C. A. (2018). Beyond plausibly exogenous. *The
  Econometrics Journal*, 21(3), 316–331.
- Conley, T. G., Hansen, C. B., & Rossi, P. E. (2012). Plausibly exogenous. *Review
  of Economics and Statistics*, 94(1), 260–272.
"""
function zero_first_stage_test(data::AbstractDataFrame, outcome::Symbol,
                               treatment::Symbol, instrument; subset,
                               covariates=Symbol[], fe=Symbol[],
                               weights::Union{Nothing,Symbol}=nothing, cluster=nothing,
                               vcov=nothing)
    inst = _as_symbols(instrument)
    vce = _iv_vcov_estimator(cluster, vcov)
    subset isa Symbol && require_columns(data, [subset]; context="zero_first_stage_test")
    rows = _iv_rows(data, subset)
    any(rows) || throw(ArgumentError("zero_first_stage_test: the subset is empty"))
    des = _iv_ols_design(data, [outcome, treatment], inst, _as_symbols(covariates),
                         _as_symbols(fe), weights, vce, "zero_first_stage_test";
                         rows=rows)
    k = length(inst)
    B, V, _ = _iv_ols(des, des.D, des.Z)
    brf, Vrf = B[:, 1], V[1:k, 1:k]
    bfs, Vfs = B[:, 2], V[(k + 1):(2k), (k + 1):(2k)]
    dof = _iv_ref_dof(des, k)
    F = _iv_wald_F(brf, Vrf)
    Ffs = _iv_wald_F(bfs, Vfs)
    return DiagnosticTest("Zero-first-stage reduced-form test",
                          "the instrument has no effect on $(outcome) in the " *
                          "zero-first-stage subsample", F, ccdf(FDist(k, dof), F);
                          dof=(k, dof), method="Wald, " * _iv_vcov_label(des),
                          note="Informative only if the first stage is zero in this " *
                               "subsample (see details.first_stage_pvalue); " *
                               "non-rejection does not establish the exclusion " *
                               "restriction elsewhere.",
                          details=(reduced_form=brf, reduced_form_se=sqrt.(diag(Vrf)),
                                   first_stage=bfs, first_stage_F=Ffs,
                                   first_stage_pvalue=ccdf(FDist(k, dof), Ffs),
                                   nobs=des.n))
end

# ---------------------------------------------------------------------------
# Overidentification and endogeneity
# ---------------------------------------------------------------------------

"""
    overidentification_test(r::IVEstimate) -> DiagnosticTest

Test of the overidentifying restrictions of an IV model with more instruments than
endogenous regressors.

With ``k`` instruments and ``p < k`` endogenous regressors, the moment conditions
``E[Z_i u_i] = 0`` impose ``k - p`` restrictions beyond those needed to identify
``\\beta``: in a constant-effects model every just-identifying subset of instruments
must estimate the same ``\\beta``. When the model was fitted with homoskedastic
covariance (`Vcov.simple()`) the function reports Sargan's (1958) statistic
``n R^2`` from the regression of the 2SLS residuals on the instruments (after
partialling out the controls). Otherwise it reports Hansen's (1982) J statistic, the
minimized criterion of two-step efficient GMM with a weight matrix built from the
2SLS residuals that is heteroskedasticity- or cluster-robust, matching the model's
covariance type, without small-sample correction. Both are compared with
``\\chi^2(k - p)``; `details` of the J test also contain the GMM and 2SLS coefficients.

The null hypothesis is that all instruments are valid given that a just-identifying
subset is; the test cannot detect invalidity that is common to all instruments, for
example if every instrument has a direct effect that biases its estimand by the same
amount. Under heterogeneous treatment effects the test has a different
interpretation: different instruments identify LATEs for different complier
populations, so a rejection may reflect effect heterogeneity rather than invalid
instruments (Angrist and Fernández-Val 2013), and non-rejection does not validate the
instruments. The statistic is also unreliable with many or weak instruments.

# Arguments
- `r::IVEstimate`: an overidentified model from [`iv_regression`](@ref) or
  [`late_2sls`](@ref) (more instruments than endogenous regressors); the test uses its
  sample, controls, weights and covariance type.

# Returns
- A [`DiagnosticTest`](@ref) with `dof = (k - p,)`.

# Examples
```julia
using DrSnow, DataFrames, StableRNGs
rng = StableRNG(7)
n = 2_000
z1, z2, v = randn(rng, n), randn(rng, n), randn(rng, n)
d = 0.5 .* z1 .+ 0.5 .* z2 .+ v
y = d .+ 0.5 .* v .+ randn(rng, n)
df = DataFrame(y=y, d=d, z1=z1, z2=z2)
r = iv_regression(df, :y, :d, [:z1, :z2])
overidentification_test(r)
```

# References
- Sargan, J. D. (1958). The estimation of economic relationships using instrumental
  variables. *Econometrica*, 26(3), 393–415.
- Hansen, L. P. (1982). Large sample properties of generalized method of moments
  estimators. *Econometrica*, 50(4), 1029–1054.
- Angrist, J. D., & Fernández-Val, I. (2013). ExtrapoLATE-ing: External validity and
  overidentification in the LATE framework. In D. Acemoglu, M. Arellano, &
  E. Dekel (Eds.), *Advances in Economics and Econometrics: Tenth World Congress*
  (Vol. III, pp. 401–434). Cambridge University Press.
- Imbens, G. W. (2014). Instrumental variables: An econometrician's perspective.
  *Statistical Science*, 29(3), 323–358.
"""
function overidentification_test(r::IVEstimate)
    des = r.design
    y, D, Z = des.y, des.D, des.Z
    k, p = size(Z, 2), size(D, 2)
    k > p || throw(ArgumentError("overidentification_test needs more instruments " *
                                 "($k) than endogenous regressors ($p)"))
    β, _, e, _ = _iv_tsls(des, y, D, Z)
    null = "all instruments satisfy the exclusion restriction, given that " *
           "a just-identifying subset does"
    note = "Only differences between instrument-specific estimands are detectable; " *
           "non-rejection does not validate the instruments."
    if des.vcov_kind === :simple
        Pe = Z * (Z \ e)
        stat = des.n * dot(e, Pe) / sum(abs2, e)
        return DiagnosticTest("Sargan overidentification test", null, stat,
                              ccdf(Chisq(k - p), stat); dof=(k - p,),
                              method="Sargan n·R², homoskedastic", note=note)
    end
    S = Z .* e
    Ω = des.vcov_kind === :robust ? Matrix(Symmetric(S' * S)) :
        _iv_psd(_iv_cluster_sum(S, des.groups))
    Wm = pinv(Ω)
    ZD, Zy = Z' * D, Z' * y
    βg = (ZD' * Wm * ZD) \ (ZD' * Wm * Zy)
    g = Z' * (y - D * βg)
    J = dot(g, Wm * g)
    return DiagnosticTest("Hansen J overidentification test", null, J,
                          ccdf(Chisq(k - p), J); dof=(k - p,),
                          method="two-step efficient GMM, " * _iv_vcov_label(des),
                          note=note, details=(gmm_coef=βg, tsls_coef=β))
end

"""
    endogeneity_test(r::IVEstimate; endogenous=r.endogenous) -> DiagnosticTest

Durbin–Wu–Hausman test of whether selected regressors can be treated as exogenous.

Hausman (1978) proposed comparing an estimator that is consistent under both the null
and the alternative (here 2SLS) with one that is efficient under the null but
inconsistent under the alternative (OLS). The function uses the equivalent
control-function (regression) form: the outcome is regressed on the endogenous
regressors, the controls and the first-stage residuals of the tested regressors, and
the coefficients on the residuals are tested jointly with a Wald F test using the
model's covariance estimator, so that the test is heteroskedasticity- or
cluster-robust when the model is (Wooldridge 2010). Under homoskedastic covariance it
is the Wu–Hausman F test. The reference distribution is ``F(q, \\text{dof})`` with
``q`` the number of tested regressors.

The test presumes that the instruments are valid and relevant; with weak instruments
it has low power, and a non-rejection is then uninformative. A rejection says that
the OLS and IV estimands differ. In a constant-effects model this is evidence of
endogeneity; with heterogeneous effects the two estimands target different weighted
averages of effects (OLS a population-wide regression coefficient, IV a LATE-type
average for compliers), so a difference can arise even if the regressor is
exogenous, and the test should not be read as a test of selection bias alone.

# Arguments
- `r::IVEstimate`: a fitted IV model.

# Keywords
- `endogenous`: the subset of `r.endogenous` to test (default all endogenous
  regressors).

# Returns
- A [`DiagnosticTest`](@ref); `details.control_function_coef` holds the coefficients
  on the first-stage residuals.

# Examples
```julia
using DrSnow, DataFrames, StableRNGs
rng = StableRNG(8)
n = 2_000
z, v = randn(rng, n), randn(rng, n)
d = 0.5 .* z .+ v
y = d .+ 0.5 .* v .+ randn(rng, n)          # d is endogenous through v
df = DataFrame(y=y, d=d, z=z)
endogeneity_test(late_2sls(df, :y, :d, :z))
```

# References
- Hausman, J. A. (1978). Specification tests in econometrics. *Econometrica*, 46(6),
  1251–1271.
- Wooldridge, J. M. (2010). *Econometric Analysis of Cross Section and Panel Data*
  (2nd ed.). MIT Press.
"""
function endogeneity_test(r::IVEstimate; endogenous=r.endogenous)
    tested = _as_symbols(endogenous)
    idx = [findfirst(==(e), r.endogenous) for e in tested]
    any(isnothing, idx) && throw(ArgumentError("endogeneity_test: $(tested) must be a " *
                                               "subset of $(r.endogenous)"))
    des = r.design
    D, Z = des.D, des.Z
    Vres = D[:, idx] - Z * (Z \ D[:, idx])
    X = hcat(D, Vres)
    p = size(D, 2)
    q = length(idx)
    B, V, _ = _iv_ols(des, des.y, X)
    b = B[(p + 1):(p + q), 1]
    Vb = V[(p + 1):(p + q), (p + 1):(p + q)]
    F = _iv_wald_F(b, Vb)
    dof = _iv_ref_dof(des, p + q)
    return DiagnosticTest("Durbin–Wu–Hausman endogeneity test",
                          join(tested, ", ") * " exogenous (OLS and IV estimands " *
                          "coincide)", F, ccdf(FDist(q, dof), F); dof=(q, dof),
                          method="control-function Wald, " * _iv_vcov_label(des),
                          note="Presumes valid instruments. Non-rejection does not show " *
                               "exogeneity (power is low with weak instruments).",
                          details=(control_function_coef=b,))
end

# ---------------------------------------------------------------------------
# Kitagawa (2015) instrument-validity test
# ---------------------------------------------------------------------------

"""
    instrument_validity_test(data, outcome, treatment, instrument; covariates=Symbol[],
                             trimming=0.07, n_bootstrap=999, max_points=100,
                             rng=Random.default_rng()) -> DiagnosticTest

Kitagawa's (2015) test of the testable implications of instrument validity with a
binary treatment and a binary instrument.

Independence, exclusion and monotonicity are not testable individually, but jointly
they have a sharp testable implication (Kitagawa 2015; related implications appear in
Imbens and Rubin 1997 and Heckman and Vytlacil 2005): the complier densities of
``Y(1)`` and ``Y(0)`` must be non-negative. With ``Z`` oriented to raise take-up, for
every interval ``B``

```math
P(Y \\in B, D = 1 \\mid Z = 1) - P(Y \\in B, D = 1 \\mid Z = 0) \\ge 0, \\qquad
P(Y \\in B, D = 0 \\mid Z = 0) - P(Y \\in B, D = 0 \\mid Z = 1) \\ge 0 .
```

The statistic is a variance-weighted Kolmogorov–Smirnov-type supremum of the scaled
violations over closed intervals whose endpoints lie on a grid of observed outcome
values (all distinct values, or `max_points` quantiles), with weights
``1/\\max(\\xi, \\hat\\sigma(B, d))``, where ``\\xi`` is the `trimming` constant. Critical
values come from Kitagawa's bootstrap, which resamples both instrument arms from the
pooled sample and so imposes the least-favourable null; the p-value is the share of
bootstrap statistics at least as large as the observed one.

**Conditioning covariates** (Kitagawa 2015, Section 3.2). When the instrument is
valid only conditional on discrete covariates ``X`` (`covariates`, whose combinations
define cells ``x``), the implications hold within cells and are tested through the
unconditional moment inequalities
``E[\\kappa_d(D, Z, X)\\, 1\\{Y \\in B, X = x\\}] \\le 0`` with

```math
\\kappa_1 = \\frac{D\\,(p(X) - Z)}{p(X)(1 - p(X))}, \\qquad
\\kappa_0 = \\frac{(1 - D)(Z - p(X))}{p(X)(1 - p(X))},
```

and ``p(X) = P(Z = 1 \\mid X)`` estimated by the cell frequencies, so that no
functional form is imposed. The statistic is ``T = \\sqrt N \\max_{d, B, x}
E_N[\\hat\\kappa_d g] / \\max(\\xi, \\hat\\sigma_d(g))`` over intervals on the outcome grid
and all cells, with ``\\hat\\sigma_d(g)`` the sample standard deviation of
``\\hat\\kappa_d g``. Critical values come from the nonparametric bootstrap of the
recentred statistic ``\\sqrt N \\max (E^*_N - E_N)[\\hat\\kappa_d g] / \\max(\\xi,
\\hat\\sigma_d(g))``, which, as in Kitagawa, ignores the estimation of ``p(X)``.
Kitagawa studentizes the bootstrap statistic with the bootstrap-sample standard
deviation; with κ-weights, bootstrap samples that omit the few observations of a
narrow interval then have ``\\hat\\sigma^* = 0`` and inflate the critical value, so the
full-sample ``\\hat\\sigma_d(g)`` is used instead, in the spirit of Andrews and Shi
(2013). The bootstrap does not select moments, so the test is conservative when many
inequalities are slack. Every cell must contain both instrument values. With a single
cell the two versions differ slightly in finite samples, because the pooled
(least-favourable) bootstrap of the unconditional test is not used here.

A rejection is evidence that at least one of independence (given the covariates),
exclusion and monotonicity fails. Non-rejection is not evidence of validity: many
violations, for example a direct effect that shifts outcomes without producing
negative complier densities, are undetectable by any test of these implications.
Observations are treated as independent (no clustering). Related tests are the
mean-based [`huber_mellace_test`](@ref) and the approach of Mourifié and Wan (2017).
The ``\\xi`` trade-off follows Kitagawa: small values weight intervals with small
variance more heavily, and ``\\xi = 1`` gives approximately the unweighted statistic.

# Arguments
- `data::AbstractDataFrame`: the data; incomplete rows are dropped.
- `outcome::Symbol`: the outcome ``Y``.
- `treatment::Symbol`: binary (0/1) treatment.
- `instrument::Symbol`: binary (0/1) instrument.

# Keywords
- `covariates::Vector{Symbol}`: discrete conditioning covariates defining cells
  (default none: the unconditional test).
- `trimming::Real`: the trimming constant ``\\xi > 0`` (default 0.07; Kitagawa
  considers 0.07, 0.3 and 1).
- `n_bootstrap::Integer`: bootstrap replications (default 999, at least 99).
- `max_points::Integer`: maximum number of grid points for interval endpoints
  (default 100, at least 2); the number of intervals grows quadratically.
- `rng::AbstractRNG`: random-number generator for the bootstrap (default
  `Random.default_rng()`; pass e.g. `StableRNG(1)` for reproducibility).

# Returns
- A [`DiagnosticTest`](@ref); `details` has `critical_value_95`, the arm sizes
  (`n_z1`, `n_z0`) or the number of cells (`n_cells`), and `grid_points`.

# Examples
```julia
using DrSnow, DataFrames, StableRNGs
rng = StableRNG(9)
n = 2_000
z = rand(rng, 0:1, n)
u = rand(rng, n)
d = ifelse.(u .< 0.2, 1, ifelse.(u .< 0.7, z, 0))
y = randn(rng, n) .+ d
df = DataFrame(y=y, d=d, z=z)
instrument_validity_test(df, :y, :d, :z; n_bootstrap=499, rng=StableRNG(1))
```

# References
- Kitagawa, T. (2015). A test for instrument validity. *Econometrica*, 83(5),
  2043–2063.
- Imbens, G. W., & Rubin, D. B. (1997). Estimating outcome distributions for
  compliers in instrumental variables models. *Review of Economic Studies*, 64(4),
  555–574.
- Heckman, J. J., & Vytlacil, E. (2005). Structural equations, treatment effects, and
  econometric policy evaluation. *Econometrica*, 73(3), 669–738.
- Andrews, D. W. K., & Shi, X. (2013). Inference based on conditional moment
  inequalities. *Econometrica*, 81(2), 609–666.
- Mourifié, I., & Wan, Y. (2017). Testing local average treatment effect
  assumptions. *Review of Economics and Statistics*, 99(2), 305–313.
"""
function instrument_validity_test(data::AbstractDataFrame, outcome::Symbol,
                                  treatment::Symbol, instrument::Symbol;
                                  covariates=Symbol[],
                                  trimming::Real=0.07, n_bootstrap::Integer=999,
                                  max_points::Integer=100,
                                  rng::AbstractRNG=Random.default_rng())
    trimming > 0 || throw(ArgumentError("trimming (ξ) must be positive"))
    n_bootstrap >= 99 || throw(ArgumentError("use at least 99 bootstrap draws"))
    max_points >= 2 || throw(ArgumentError("max_points must be at least 2"))
    ctx = "instrument_validity_test"
    covs = _as_symbols(covariates)
    if !isempty(covs)
        return _iv_kitagawa_conditional(data, outcome, treatment, instrument, covs,
                                        float(trimming), Int(n_bootstrap),
                                        Int(max_points), rng, ctx)
    end
    prep = _iv_binary_prep(data, treatment, instrument, [outcome], Symbol[], nothing,
                           nothing, ctx)
    y = Float64.(prep.sub[!, outcome])
    d = prep.d .== 1
    z = prep.z .== 1
    if mean(d[z]) < mean(d[.!z])
        z = .!z
    end
    grid = sort(unique(y))
    if length(grid) > max_points
        grid = unique(quantile(y, range(0, 1; length=max_points)))
    end
    m, n = count(z), count(.!z)
    N = m + n
    λ = m / N
    P1, P0 = _iv_kitagawa_probs(y[z], d[z], grid)     # Z = 1 arm: (d=1, d=0)
    Q1, Q0 = _iv_kitagawa_probs(y[.!z], d[.!z], grid)
    σ1 = sqrt.(max.(λ .* P1 .* (1 .- P1) .+ (1 - λ) .* Q1 .* (1 .- Q1), 0.0))
    σ0 = sqrt.(max.(λ .* P0 .* (1 .- P0) .+ (1 - λ) .* Q0 .* (1 .- Q0), 0.0))
    w1 = 1 ./ max.(trimming, σ1)
    w0 = 1 ./ max.(trimming, σ0)
    scale = sqrt(m * n / N)
    stat(P1, P0, Q1, Q0) = scale * max(maximum((Q1 .- P1) .* w1),
                                       maximum((P0 .- Q0) .* w0))
    T = stat(P1, P0, Q1, Q0)
    Tb = Vector{Float64}(undef, n_bootstrap)
    for b in 1:n_bootstrap
        i1 = rand(rng, 1:N, m)
        i0 = rand(rng, 1:N, n)
        bP1, bP0 = _iv_kitagawa_probs(y[i1], d[i1], grid)
        bQ1, bQ0 = _iv_kitagawa_probs(y[i0], d[i0], grid)
        Tb[b] = stat(bP1, bP0, bQ1, bQ0)
    end
    pval = (1 + count(>=(T - 1e-12), Tb)) / (1 + n_bootstrap)
    return DiagnosticTest("Kitagawa instrument-validity test",
                          "independence, exclusion and monotonicity hold jointly " *
                          "(non-negative complier outcome densities)", T, pval;
                          method="variance-weighted KS over $(length(grid)) grid " *
                                 "points, ξ = $(trimming), pooled bootstrap " *
                                 "($(n_bootstrap) draws)",
                          note="Detects only violations that produce negative complier " *
                               "densities; non-rejection does not validate the " *
                               "instrument. iid sampling assumed.",
                          details=(critical_value_95=quantile(Tb, 0.95), n_z1=m, n_z0=n,
                                   grid_points=length(grid)))
end

"""
Probabilities `P(Y ∈ [g_a, g_b], D = d)` for all grid intervals `a ≤ b`, returned as
vectors (d = 1, d = 0) over the upper-triangular index set.
"""
function _iv_kitagawa_probs(y::AbstractVector, d::AbstractVector{Bool},
                            grid::Vector{Float64})
    M = length(grid)
    nobs = length(y)
    le1 = zeros(Int, M)
    lt1 = zeros(Int, M)
    le0 = zeros(Int, M)
    lt0 = zeros(Int, M)
    for i in 1:nobs
        # counts of observations with y ≤ g_j and y < g_j
        jle = searchsortedfirst(grid, y[i])     # first grid point ≥ y
        jlt = searchsortedlast(grid, y[i]) + 1  # first grid point > y
        if d[i]
            jle <= M && (le1[jle] += 1)
            jlt <= M && (lt1[jlt] += 1)
        else
            jle <= M && (le0[jle] += 1)
            jlt <= M && (lt0[jlt] += 1)
        end
    end
    cumsum!(le1, le1)
    cumsum!(lt1, lt1)
    cumsum!(le0, le0)
    cumsum!(lt0, lt0)
    L = M * (M + 1) ÷ 2
    p1 = Vector{Float64}(undef, L)
    p0 = Vector{Float64}(undef, L)
    idx = 0
    for a in 1:M, b in a:M
        idx += 1
        p1[idx] = (le1[b] - lt1[a]) / nobs
        p0[idx] = (le0[b] - lt0[a]) / nobs
    end
    return p1, p0
end

"""
Sums `Σ_{i: y_i ∈ [g_a, g_b]} v_i` over all grid intervals `a ≤ b` (same ordering as
`_iv_kitagawa_probs`), for each column of `V`.
"""
function _iv_kit_interval_sums(y::AbstractVector, V::AbstractMatrix,
                               grid::Vector{Float64})
    M = length(grid)
    q = size(V, 2)
    le = zeros(M, q)
    lt = zeros(M, q)
    for i in eachindex(y)
        jle = searchsortedfirst(grid, y[i])
        jlt = searchsortedlast(grid, y[i]) + 1
        for c in 1:q
            jle <= M && (le[jle, c] += V[i, c])
            jlt <= M && (lt[jlt, c] += V[i, c])
        end
    end
    cumsum!(le, le; dims=1)
    cumsum!(lt, lt; dims=1)
    L = M * (M + 1) ÷ 2
    out = Matrix{Float64}(undef, L, q)
    idx = 0
    for a in 1:M, b in a:M
        idx += 1
        for c in 1:q
            out[idx, c] = le[b, c] - lt[a, c]
        end
    end
    return out
end

"""
Kitagawa (2015, Section 3.2) test with discrete conditioning covariates: variance-
weighted KS statistic over the moment inequalities `E[κ_d g] ≤ 0`, bootstrap of the
recentred statistic.
"""
function _iv_kitagawa_conditional(data, outcome, treatment, instrument, covs, ξ, B,
                                  max_points, rng, ctx)
    prep = _iv_binary_prep(data, treatment, instrument, [outcome], Symbol[], nothing,
                           nothing, ctx; groupings=covs)
    sub = prep.sub
    y = Float64.(sub[!, outcome])
    d = prep.d
    z = prep.z
    if mean(d[z .== 1]) < mean(d[z .== 0])
        z = 1 .- z
    end
    N = prep.n
    keys_ = [Tuple(sub[i, c] for c in covs) for i in 1:N]
    levels_ = sort(unique(keys_); by=string)
    cid = Dict(k => j for (j, k) in enumerate(levels_))
    cell = [cid[k] for k in keys_]
    C = length(levels_)
    nc = zeros(C)
    n1 = zeros(C)
    for i in 1:N
        nc[cell[i]] += 1
        n1[cell[i]] += z[i]
    end
    bad = findall(c -> n1[c] == 0 || n1[c] == nc[c], 1:C)
    isempty(bad) || throw(ArgumentError("$ctx: covariate cell(s) $(levels_[bad]) have " *
                                        "only one instrument value; P(Z = 1 | X) is 0 " *
                                        "or 1 there — merge or drop these cells (the " *
                                        "covariates must be discrete)"))
    px = (n1 ./ nc)[cell]
    κ1 = d .* (px .- z) ./ (px .* (1 .- px))
    κ0 = (1 .- d) .* (z .- px) ./ (px .* (1 .- px))
    grid = sort(unique(y))
    if length(grid) > max_points
        grid = unique(quantile(y, range(0, 1; length=max_points)))
    end
    # columns: (κ1, κ0) × cells, and their squares
    function moments(wts)
        V = zeros(N, 4C)
        for i in 1:N
            c = cell[i]
            V[i, c] = wts[i] * κ1[i]
            V[i, C + c] = wts[i] * κ0[i]
            V[i, 2C + c] = wts[i] * κ1[i]^2
            V[i, 3C + c] = wts[i] * κ0[i]^2
        end
        S = _iv_kit_interval_sums(y, V, grid) ./ N
        mom = S[:, 1:(2C)]
        return mom, sqrt.(max.(S[:, (2C + 1):(4C)] .- mom .^ 2, 0.0))
    end
    m, σ = moments(ones(N))
    T = sqrt(N) * maximum(m ./ max.(ξ, σ))
    seeds = task_seeds(rng, B)
    Tb = Vector{Float64}(undef, B)
    cnt = zeros(N)
    for b in 1:B
        brng = Xoshiro(seeds[b])
        fill!(cnt, 0.0)
        for _ in 1:N
            cnt[rand(brng, 1:N)] += 1
        end
        mb, σb = moments(cnt)
        Tb[b] = sqrt(N) * maximum((mb .- m) ./ max.(ξ, σ))
    end
    pval = (1 + count(>=(T - 1e-12), Tb)) / (1 + B)
    return DiagnosticTest("Kitagawa instrument-validity test with covariates",
                          "independence given the covariates, exclusion and " *
                          "monotonicity hold jointly (non-negative complier outcome " *
                          "densities within covariate cells)", T, pval;
                          method="variance-weighted KS over $(length(grid)) grid " *
                                 "points × $C covariate cells, ξ = $(ξ), " *
                                 "κ-weighted moments with cell propensities, " *
                                 "recentred bootstrap ($B draws), full-sample " *
                                 "studentization",
                          note="Detects only violations that produce negative complier " *
                               "densities within cells; non-rejection does not " *
                               "validate the instrument. iid sampling assumed; the " *
                               "estimation of P(Z = 1 | X) is ignored by the bootstrap.",
                          details=(critical_value_95=quantile(Tb, 0.95), n_cells=C,
                                   grid_points=length(grid), covariates=covs))
end
