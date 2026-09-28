# Fisher randomization test of a sharp null hypothesis.

"""
    RandomizationTestResult

Result of a Fisher randomization test of a sharp null hypothesis, returned by
[`randomization_test`](@ref).

The object holds the observed statistic, the randomization p-value and the complete
reference distribution: every assignment of the design's support with its
probability when the test is exact, or the observed assignment followed by the Monte
Carlo draws (each with weight one) otherwise. Keeping the reference set makes the
test auditable and lets users plot the randomization distribution or compute other
tail probabilities from it.

# Fields
- `statistic_name::String`: the test statistic.
- `observed::Float64`: its value on the realized assignment, computed on outcomes
  adjusted for the sharp null (``Y - τ_0 Z``).
- `pvalue::Float64`: randomization p-value.
- `mc_se::Float64`: Monte Carlo standard error of `pvalue`,
  ``\\sqrt{p(1-p)/B}``; `0.0` for exact enumeration.
- `alternative::Symbol`: `:two_sided` (compares ``|T|``), `:greater` or `:less`.
- `null::String`: the sharp null hypothesis, in words.
- `exact::Bool`: `true` when the p-value comes from complete enumeration.
- `n_draws::Int`: number of enumerated assignments, or of Monte Carlo draws ``B``.
- `n_dropped::Int`: assignments of the reference set on which the statistic was
  undefined (e.g. an empty arm within every stratum); they are excluded, so the
  p-value conditions on the statistic being defined.
- `distribution::Vector{Float64}`, `weights::Vector{Float64}`: the reference set.
- `mechanism::String`: description of the assignment mechanism.
- `nobs::Int`, `n_treated::Int`: numbers of units and of treated units.

# Accessors
- `pvalue(r)`, `nobs(r)`, [`rejects`](@ref)`(r; alpha)`,
  [`randomization_distribution`](@ref)`(r)`, and `DiagnosticTest(r)` to convert to
  the package-wide [`DiagnosticTest`](@ref).
"""
struct RandomizationTestResult
    statistic_name::String
    observed::Float64
    pvalue::Float64
    mc_se::Float64
    alternative::Symbol
    null::String
    exact::Bool
    n_draws::Int
    n_dropped::Int
    distribution::Vector{Float64}
    weights::Vector{Float64}
    mechanism::String
    nobs::Int
    n_treated::Int
end

StatsAPI.pvalue(r::RandomizationTestResult) = r.pvalue
StatsAPI.nobs(r::RandomizationTestResult) = r.nobs

"""
    randomization_distribution(r) -> (values, weights)

Randomization (reference) distribution stored in a randomization-inference result:
the values of the test statistic over the reference set of assignments and their
weights.

For exact results the reference set is the support of the design and the weights are
assignment probabilities; for Monte Carlo results the first value is the observed
statistic and all weights are one. The randomization p-value is the weighted share
of the reference set at least as extreme as the observed statistic, so any tail
probability can be recomputed from this output. For
[`MultipleTestingResult`](@ref) and [`RIRegressionResult`](@ref) `values` is a
matrix with one row per assignment and one column per statistic (the joint
distribution used for multiplicity adjustments and joint tests).

# Arguments
- `r`: a [`RandomizationTestResult`](@ref), [`MultipleTestingResult`](@ref) or
  [`RIRegressionResult`](@ref).

# Returns
- `Tuple` of the values (`Vector{Float64}` or `Matrix{Float64}`) and the weights
  (`Vector{Float64}`).

# Examples
```julia
using DrSnow, DataFrames, StableRNGs
rng = StableRNG(1)
df = DataFrame(d=[trues(15); falses(15)])
df.y = 0.8 .* df.d .+ randn(rng, 30)
r = randomization_test(df, :y, :d; nperm=999, rng=rng)
v, w = randomization_distribution(r)
sum(w[abs.(v) .>= abs(r.observed) - 1e-9]) / sum(w) ≈ pvalue(r)   # true
```
"""
randomization_distribution(r::RandomizationTestResult) = (r.distribution, r.weights)

"""
    rejects(r::RandomizationTestResult; alpha=0.05) -> Bool

Whether the sharp null hypothesis of the randomization test `r` is rejected at level
`alpha`, i.e. whether `pvalue(r) < alpha`.

For exact tests the rejection rule has size at most `alpha` under the sharp null;
Monte Carlo p-values of the form ``(1 + \\#)/(1 + B)`` preserve this guarantee. A
non-rejection is not evidence of no effect.

# Arguments
- `r::RandomizationTestResult`: a randomization test result.

# Keywords
- `alpha::Real`: significance level; default 0.05.

# Returns
- `Bool`.

# Examples
```julia
using DrSnow, DataFrames, StableRNGs
rng = StableRNG(1)
df = DataFrame(d=[trues(15); falses(15)])
df.y = 0.8 .* df.d .+ randn(rng, 30)
rejects(randomization_test(df, :y, :d; nperm=999, rng=rng); alpha=0.10)
```
"""
rejects(r::RandomizationTestResult; alpha::Real=0.05) = r.pvalue < alpha

"""
    DiagnosticTest(r::RandomizationTestResult) -> DiagnosticTest

Convert a randomization test result to the package-wide [`DiagnosticTest`](@ref)
type, for uniform printing and tabulation with [`tidy`](@ref).

The method string records whether the p-value is exact or Monte Carlo (with its
Monte Carlo standard error), the note records the assignment mechanism and the
alternative, and the original result is kept in `details.result`.

# Arguments
- `r::RandomizationTestResult`: a randomization test result.

# Returns
- `DiagnosticTest`.

# Examples
```julia
using DrSnow, DataFrames, StableRNGs
rng = StableRNG(1)
df = DataFrame(d=[trues(15); falses(15)])
df.y = 0.8 .* df.d .+ randn(rng, 30)
t = DiagnosticTest(randomization_test(df, :y, :d; nperm=999, rng=rng))
tidy(t)
```
"""
function DiagnosticTest(r::RandomizationTestResult)
    method = r.exact ?
        "randomization inference, exact enumeration of $(r.n_draws) assignments" :
        @sprintf("randomization inference, Monte Carlo with %d draws (MC s.e. of p = %.2g)",
                 r.n_draws, r.mc_se)
    return DiagnosticTest("Fisher randomization test ($(r.statistic_name))", r.null,
                          r.observed, r.pvalue; method=method,
                          note="Assignment mechanism: $(r.mechanism). Alternative: " *
                               "$(r.alternative).",
                          details=(result=r,))
end

function Base.show(io::IO, ::MIME"text/plain", r::RandomizationTestResult)
    println(io, "Fisher randomization test")
    println(io, "H₀ (sharp): ", r.null)
    println(io, "Statistic: ", r.statistic_name, "; alternative: ", r.alternative)
    println(io, "Assignment mechanism: ", r.mechanism)
    @printf(io, "Units: %d (%d treated)\n", r.nobs, r.n_treated)
    if r.exact
        println(io, "Reference distribution: exact, $(r.n_draws) assignments")
        @printf(io, "Observed statistic = %.6g, exact p-value = %.4g\n", r.observed,
                r.pvalue)
    else
        println(io, "Reference distribution: Monte Carlo, $(r.n_draws) draws")
        @printf(io, "Observed statistic = %.6g, p-value = %.4g (MC s.e. %.2g)\n",
                r.observed, r.pvalue, r.mc_se)
    end
    r.n_dropped > 0 &&
        println(io, "Assignments with undefined statistic (excluded): ", r.n_dropped)
    print(io, r.pvalue < 0.05 ? "H₀ rejected at the 5% level." :
          "H₀ not rejected at the 5% level (non-rejection is not evidence of no effect).")
end

Base.show(io::IO, r::RandomizationTestResult) =
    @printf(io, "RandomizationTestResult(stat = %.4g, p = %.4g%s)", r.observed, r.pvalue,
            r.exact ? ", exact" : "")

function _ri_null_text(tau0, outcome)
    if tau0 isa Symbol
        return "Y_i(1) - Y_i(0) = $(tau0)_i for every unit i (outcome $outcome)"
    end
    t = float(tau0)
    return iszero(t) ? "no effect of treatment on $outcome for any unit" :
           "Y_i(1) - Y_i(0) = $(t) for every unit i (outcome $outcome)"
end

"""
    randomization_test(data, outcome, treatment; statistic=:diff_means, tau0=0.0,
                       alternative=:two_sided, mechanism=nothing, strata=nothing,
                       cluster=nothing, id=nothing, covariates=Symbol[],
                       nperm=10_000, exact=:auto, rng=Random.default_rng(),
                       threaded=Threads.nthreads() > 1) -> RandomizationTestResult

Fisher randomization test of a sharp null hypothesis about unit-level treatment
effects, with the reference distribution generated by the known assignment
mechanism.

A sharp null hypothesis specifies every unit's treatment effect,
```math
H_0:\\; Y_i(1) - Y_i(0) = τ_{0i} \\quad \\text{for every unit } i,
```
by default ``τ_{0i} = 0`` (no effect for any unit), otherwise a constant ``τ_0`` or
unit-specific values taken from a column. Under such a null the missing potential
outcomes are known: the adjusted outcomes ``Y_i - τ_{0i} Z_i`` equal ``Y_i(0)`` for
every unit whatever the assignment. The distribution of any statistic
``T(Z, Y - τ_0 Z)`` over assignments drawn from the design is therefore known
exactly, and comparing the observed statistic with it gives a p-value that is valid
in finite samples, for any sample size and any outcome distribution, with no model
for the outcomes and no large-sample approximation (Fisher 1935; Rosenbaum 2002;
Imbens and Rubin 2015, ch. 5). The only assumption is that the stated mechanism is
the one that assigned treatment (together with no interference, which is implicit
in writing ``Y_i(z_i)``).

Exactness is a property of the sharp null. The weak null of a zero *average* effect,
``\\bar τ = 0``, leaves unit-level effects unrestricted, and a randomization test
based on the plain difference in means need not control its size, even
asymptotically, when effects are heterogeneous and arm sizes differ. Studentizing
the statistic repairs this: the `:studentized` and `:lin_studentized` statistics
give tests that remain exact under the sharp null and are asymptotically valid for
the weak null (Wu and Ding 2021; Zhao and Ding 2021; for the analogous result for
permutation tests of equality of means, Chung and Romano 2013). Prefer them when the
hypothesis of interest is about an average effect.

The reference distribution is computed exactly by enumerating the support of the
design when it has at most `nperm` assignments (`exact = :auto`) or when
`exact = true`; otherwise from `nperm` Monte Carlo draws, and the p-value is
``(1 + \\#\\{\\text{draws at least as extreme}\\}) / (1 + B)``, which is itself a
valid p-value (see [`permutation_pvalue`](@ref)); its Monte Carlo standard error is
reported. With `alternative = :two_sided` a draw is "at least as extreme" when
``|T| ≥ |T_{obs}|``; this is not twice the smaller one-sided p-value, and it is
therefore not the test that [`ri_confint`](@ref) inverts (its two-sided interval is
equal-tailed).

If `mechanism` is not supplied it is built from the data, conditionally on the
observed treated counts: complete randomization with the observed number of treated
units, or stratified, cluster or blocked cluster randomization with the observed
counts per stratum. Conditioning on the number treated is appropriate for complete
randomization and also valid under a Bernoulli design with a common probability
(given the number treated, every assignment is equally likely), but not under
Bernoulli designs with unit-specific probabilities or under rerandomization; pass
the actual mechanism in those cases. Choose the statistic before looking at the
outcomes: rank statistics are robust to outliers, the Kolmogorov–Smirnov statistic
targets distributional differences, and covariate adjustment (`:lin`) can increase
power substantially (Lin 2013). For regression coefficients and several treatment
arms see [`ri_regression`](@ref); for several outcomes [`ri_multiple_testing`](@ref);
for interval estimates [`ri_confint`](@ref). Report the statistic, the mechanism,
and whether the p-value is exact or based on ``B`` draws.

# Arguments
- `data`: a `DataFrame` with one row per randomized unit. Missing values are not
  allowed in the columns used, because dropping units changes the assignment
  mechanism.
- `outcome::Symbol`: outcome column.
- `treatment::Symbol`: treatment column, coded 0/1 or `Bool`.

# Keywords
- `statistic`: the test statistic. `:diff_means` (default; the
  stratum-size-weighted average of within-stratum differences in means),
  `:studentized` (the same divided by its Neyman / cluster-robust standard error;
  matched pairs use the variance of pair differences), `:rank_sum` (Wilcoxon
  rank-sum, as the difference in mean within-stratum ranks scaled by
  ``1/(n_s + 1)``), `:ks` (two-sample Kolmogorov–Smirnov distance; upper tail
  only), `:lin` / `:lin_studentized` (Lin 2013 regression-adjusted difference using
  `covariates`, or its HC2 / cluster-robust t-statistic), or a function
  `f(y::Vector{Float64}, z::BitVector) -> Real` of the adjusted outcomes and an
  assignment.
- `tau0`: hypothesized constant additive effect (`Real`, default 0) or the name of
  a column of unit-specific effects (`Symbol`).
- `alternative::Symbol`: `:two_sided` (default), `:greater` or `:less`.
- `mechanism::AssignmentMechanism`: the design, over units in the row order of
  `data` (or of rows sorted by `id`). Mutually exclusive with `strata` / `cluster`.
- `strata`, `cluster::Union{Nothing,Symbol}`: randomization strata and clusters
  used to build the design from the data (see above).
- `id::Union{Nothing,Symbol}`: unique unit identifier fixing the unit order when a
  `mechanism` is supplied.
- `covariates::Vector{Symbol}`: pre-treatment covariates for the `:lin` statistics
  (not allowed with other statistics).
- `nperm::Integer`: number of Monte Carlo draws, and the enumeration threshold for
  `exact = :auto`; default 10 000.
- `exact`: `:auto` (default), `true` (always enumerate; errors when infeasible) or
  `false` (always Monte Carlo).
- `rng::AbstractRNG`: random number generator; per-chunk seeds are drawn up front,
  so results are reproducible and independent of the number of threads.
- `threaded::Bool`: evaluate draws on several threads (user statistics must then be
  thread-safe).

# Returns
- [`RandomizationTestResult`](@ref); `pvalue(r)` gives the p-value.

# Examples
```julia
using DrSnow, DataFrames, StableRNGs
rng = StableRNG(1)
df = DataFrame(block=repeat(1:4; inner=10), d=repeat([1, 1, 1, 1, 0, 0, 0, 0, 0, 0], 4))
df.x = randn(rng, 40)
df.y = 0.5 .* df.block .+ 0.7 .* df.d .+ df.x .+ randn(rng, 40)
r = randomization_test(df, :y, :d; strata=:block, nperm=5_000, rng=rng)
pvalue(r)
randomization_test(df, :y, :d; strata=:block, statistic=:studentized, rng=rng)
randomization_test(df, :y, :d; tau0=0.5, statistic=:rank_sum, strata=:block, rng=rng)
randomization_test(df, :y, :d; statistic=:lin, covariates=[:x], strata=:block, rng=rng)
```

# References
- Fisher, R. A. (1935). *The Design of Experiments*. Oliver & Boyd.
- Rosenbaum, P. R. (2002). *Observational Studies* (2nd ed.), ch. 2. Springer.
- Imbens, G. W., & Rubin, D. B. (2015). *Causal Inference for Statistics, Social,
  and Biomedical Sciences: An Introduction*, ch. 5. Cambridge University Press.
- Chung, E., & Romano, J. P. (2013). Exact and asymptotically robust permutation
  tests. *Annals of Statistics*, 41(2), 484–507.
- Wu, J., & Ding, P. (2021). Randomization tests for weak null hypotheses in
  randomized experiments. *Journal of the American Statistical Association*,
  116(536), 1898–1913.
- Zhao, A., & Ding, P. (2021). Covariate-adjusted Fisher randomization tests for the
  average treatment effect. *Journal of Econometrics*, 225(2), 278–294.
- Lin, W. (2013). Agnostic notes on regression adjustments to experimental data:
  Reexamining Freedman's critique. *Annals of Applied Statistics*, 7(1), 295–318.
- Young, A. (2019). Channeling Fisher: Randomization tests and the statistical
  insignificance of seemingly significant experimental results. *Quarterly Journal
  of Economics*, 134(2), 557–598.
"""
function randomization_test(data, outcome::Symbol, treatment::Symbol;
                            statistic=:diff_means, tau0::Union{Real,Symbol}=0.0,
                            alternative::Symbol=:two_sided, mechanism=nothing,
                            strata::Union{Nothing,Symbol}=nothing,
                            cluster::Union{Nothing,Symbol}=nothing,
                            id::Union{Nothing,Symbol}=nothing,
                            covariates::Vector{Symbol}=Symbol[],
                            nperm::Integer=10_000, exact=:auto,
                            rng::AbstractRNG=Random.default_rng(),
                            threaded::Bool=Threads.nthreads() > 1)
    ctxname = "randomization_test"
    _ri_check_alternative(alternative)
    stat = _ri_parse_statistic(statistic)
    _ri_check_covariates(stat, covariates, ctxname)
    extra = tau0 isa Symbol ? [outcome, tau0] : [outcome]
    design = _ri_design(data, treatment; mechanism, strata, cluster, id, covariates,
                        sortcols=extra, context=ctxname)
    y = _ri_adjusted_outcome(data, outcome, design, tau0)
    all(isfinite, y) || throw(ArgumentError("$ctxname: outcome has non-finite values"))
    alt = _ri_signed(stat) ? alternative : _ri_unsigned_alternative(alternative, stat)
    plan = _ri_plan(design.mech, nperm, exact, rng)
    prep = _ri_prepare(stat, y, design.ctx)
    ctx = design.ctx
    vals = vec(_ri_map(z -> _ri_eval(stat, prep, z, ctx), plan, design.mech, design.z, 1;
                       threaded))
    obs = _ri_eval(stat, prep, design.z, ctx)
    isnan(obs) && throw(ArgumentError("$ctxname: the statistic is undefined for the " *
                                      "observed assignment"))
    w = _ri_weights(plan)
    p, _, dropped = _ri_pvalue(obs, vals, w, alt)
    se = plan.exact ? 0.0 : _ri_mc_se(p, plan.B - dropped)
    return RandomizationTestResult(_ri_name(stat), obs, p, se, alt,
                                   _ri_null_text(tau0, outcome), plan.exact, plan.B,
                                   dropped, vals, w, _ri_describe(design.mech),
                                   length(design.z), count(design.z))
end

function _ri_check_covariates(stat, covariates, ctxname)
    if !isempty(covariates) && !_ri_uses_covariates(stat)
        throw(ArgumentError("$ctxname: `covariates` are only used by the :lin and " *
                            ":lin_studentized statistics"))
    end
end

function _ri_unsigned_alternative(alternative, stat)
    alternative === :less &&
        throw(ArgumentError("the $(_ri_name(stat)) statistic is non-negative; only the " *
                            "upper tail (:greater or :two_sided) is meaningful"))
    return :greater
end
