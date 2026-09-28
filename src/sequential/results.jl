# Result types of the anytime-valid part of the area: a confidence-sequence path
# (`ConfidenceSequence`, a `CausalEstimate`) and a sequential test / e-process path
# (`SequentialTest`, convertible to a `DiagnosticTest`).

"""
    ConfidenceSequence <: CausalEstimate

Path of an anytime-valid confidence sequence for a scalar parameter (a mean, a
difference in means or an average treatment effect), together with the e-process of
the dual sequential test.

A ``(1-\\alpha)`` confidence sequence (CS) is a sequence of intervals
``(C_t)_{t \\ge 1}``, each computed from the first ``t`` observations, such that

```math
P\\left(\\theta \\in C_t \\ \\text{for all } t \\ge 1\\right) \\ge 1 - \\alpha .
```

The guarantee is *time-uniform*: it holds simultaneously over all sample sizes, and
therefore at any stopping time ``\\tau``, however ``\\tau`` depends on the data
(Darling and Robbins 1967; Lai 1976; Howard et al. 2021). The analyst may inspect the
interval after every observation, stop when it excludes zero, when it is narrow
enough, or for reasons unrelated to the data, and still report ``C_\\tau`` with
coverage ``1-\\alpha``. A fixed-``n`` confidence interval monitored in the same way
has a crossing probability that tends to one as the number of looks grows (the
"sampling to a foregone conclusion" problem of Armitage, McPherson and Rowe 1969).
Every CS in the package is obtained by inverting a nonnegative (super)martingale or
e-process ``E_t(m)``: ``C_t = \\{m : E_t(m) < 1/\\alpha\\}``, and Ville's (1939)
inequality ``P(\\sup_t E_t(\\theta) \\ge 1/\\alpha) \\le \\alpha`` delivers the
coverage (Ramdas et al. 2023).

The price of time-uniformity is width. A boundary is tuned to be tightest at one
sample size ``t_{\\text{opt}}``; for the Gaussian (normal-mixture) boundaries at level
0.95 the CS is about 1.55 times as wide as the fixed-``n`` Wald interval at
``t = t_{\\text{opt}}``, the ratio grows slowly (like ``\\sqrt{\\log t}``) beyond it,
and quickly before it (about 2 times at ``t_{\\text{opt}}/10``). Whether the coverage
is exact (nonasymptotic) or asymptotic depends on the boundary recorded in
`boundary`; see [`MeanMonitor`](@ref) and [`ATEMonitor`](@ref).

Accessors: `coef(r)` is the final point estimate; `confint(r)` the CS at the final
sample size; `pvalues(r)` the always-valid p-value for ``H_0: \\theta =`` `r.null`
(the smallest ``\\alpha`` at which some ``C_s``, ``s \\le n``, would have excluded
`r.null`); `stderror(r)` the naive fixed-sample standard error, reported for
reference only and not a basis for sequential inference. The whole path is returned by
[`sequence_path`](@ref); [`sequential_test`](@ref) turns it into a test and
[`stopping`](@ref) summarises the first crossing.

# Fields
- `estimand::String`: the parameter (`"mean"` or `"ATE"`).
- `method::String`: description of the boundary and, for ATEs, of the nuisance
  models.
- `boundary::Symbol`: `:asymptotic`, `:normal_mixture`, `:hoeffding`,
  `:empirical_bernstein`, `:betting` or `:aipw`.
- `n::Vector{Int}`: sample size at each step of the path.
- `estimate::Vector{Float64}`: running point estimate (sample mean or mean of the
  AIPW pseudo-outcomes).
- `lower::Vector{Float64}`, `upper::Vector{Float64}`: the confidence sequence;
  `-Inf`/`Inf` before monitoring starts.
- `sigma::Vector{Float64}`: running standard deviation of the observations (of the
  pseudo-outcomes for ATEs), or the known ``\\sigma`` of the normal-mixture boundary.
- `evalue::Vector{Float64}`: e-process for ``H_0: \\theta =`` `null` at each step.
- `null::Float64`: the null value of the e-process.
- `level::Float64`: the coverage ``1-\\alpha``.
- `t_opt::Float64`: the sample size for which the boundary was tuned (`NaN` for the
  untuned bounded-data boundaries).
- `rho::Float64`: normal-mixture precision parameter (`NaN` when not applicable).
- `running_intersection::Bool`: whether the reported CS is the running intersection
  ``\\cap_{s \\le t} C_s``.
- `note::String`: assumptions and caveats of the boundary.

# References
- Darling, D. A., & Robbins, H. (1967). Confidence sequences for mean, variance, and
  median. *Proceedings of the National Academy of Sciences*, 58(1), 66–68.
- Lai, T. L. (1976). On confidence sequences. *Annals of Statistics*, 4(2), 265–280.
- Howard, S. R., Ramdas, A., McAuliffe, J., & Sekhon, J. (2021). Time-uniform,
  nonparametric, nonasymptotic confidence sequences. *Annals of Statistics*, 49(2),
  1055–1080.
- Ramdas, A., Grünwald, P., Vovk, V., & Shafer, G. (2023). Game-theoretic statistics
  and safe anytime-valid inference. *Statistical Science*, 38(4), 576–601.
- Armitage, P., McPherson, C. K., & Rowe, B. C. (1969). Repeated significance tests on
  accumulating data. *Journal of the Royal Statistical Society: Series A*, 132(2),
  235–244.
- Ville, J. (1939). *Étude critique de la notion de collectif*. Gauthier-Villars.
"""
struct ConfidenceSequence <: CausalEstimate
    estimand::String
    method::String
    boundary::Symbol
    n::Vector{Int}
    estimate::Vector{Float64}
    lower::Vector{Float64}
    upper::Vector{Float64}
    sigma::Vector{Float64}
    evalue::Vector{Float64}
    null::Float64
    level::Float64
    t_opt::Float64
    rho::Float64
    running_intersection::Bool
    note::String
end

function _seq_check_nonempty(r::ConfidenceSequence)
    isempty(r.n) && throw(ArgumentError("the confidence sequence has no observations"))
    return nothing
end

StatsAPI.coef(r::ConfidenceSequence) = (_seq_check_nonempty(r); [r.estimate[end]])
function StatsAPI.vcov(r::ConfidenceSequence)
    _seq_check_nonempty(r)
    return fill(r.sigma[end]^2 / r.n[end], 1, 1)
end
StatsAPI.coefnames(r::ConfidenceSequence) = [r.estimand]
StatsAPI.nobs(r::ConfidenceSequence) = isempty(r.n) ? 0 : r.n[end]
estimand(r::ConfidenceSequence) = r.estimand
method_name(r::ConfidenceSequence) = r.method

"""
    confint(r::ConfidenceSequence; level=r.level) -> Matrix{Float64}

The confidence sequence at the final sample size, as a ``1 \\times 2`` matrix.

Because the coverage of a confidence sequence is time-uniform, the interval returned
here is valid whether the final sample size was fixed in advance or reached by a
data-dependent stopping rule. For the Gaussian boundaries (`:asymptotic`,
`:normal_mixture` and the AIPW sequence of [`confseq_ate`](@ref)) a different `level`
recomputes the whole path from the stored running estimates and scales, re-optimising
the mixture for the same `t_opt`, so that the running intersection is taken at the new
level. The bounded-data boundaries (`:hoeffding`, `:empirical_bernstein`, `:betting`)
depend on the level through the bets placed at every step; they cannot be recomputed
after the fact and throw an `ArgumentError` for a different `level`.

# Arguments
- `r::ConfidenceSequence`: the confidence-sequence path.

# Keywords
- `level::Real = r.level`: coverage of the interval; must lie in ``(0, 1)``. Choosing
  the level after looking at the data is not covered by the guarantee; report the
  level fixed at the design stage.

# Returns
- `Matrix{Float64}` of size ``1 \\times 2``: the lower and upper limit at `nobs(r)`.

# Examples
```julia
using DrSnow, StableRNGs
x = randn(StableRNG(1), 400) .+ 0.2
cs = confseq_mean(x; method=:asymptotic, t_opt=400)
confint(cs)                    # 95% CS at n = 400
confint(cs; level=0.90)        # recomputed path at level 0.90
```
"""
function StatsAPI.confint(r::ConfidenceSequence; level::Real=r.level)
    _seq_check_nonempty(r)
    if isapprox(level, r.level; atol=1e-12)
        return [r.lower[end] r.upper[end]]
    end
    lo, hi = _seq_recompute_gaussian(r, level)
    return [lo[end] hi[end]]
end

function _seq_recompute_gaussian(r::ConfidenceSequence, level::Real)
    (0 < level < 1) || throw(ArgumentError("level must be in (0, 1), got $level"))
    r.boundary in (:asymptotic, :normal_mixture, :aipw) ||
        throw(ArgumentError("the $(r.boundary) confidence sequence was computed at " *
                            "level $(r.level); recompute it with `level = $level`"))
    alpha = 1 - level
    rho = _seq_nm_rho(r.t_opt, alpha)
    lo = similar(r.lower)
    hi = similar(r.upper)
    for i in eachindex(r.n)
        s = r.sigma[i]
        # not yet started (lower = -Inf) or no scale yet: unbounded
        rad = isnan(s) || r.lower[i] == -Inf ? Inf :
              s * _seq_nm_bound(r.n[i], alpha, rho) / r.n[i]
        lo[i] = r.estimate[i] - rad
        hi[i] = r.estimate[i] + rad
    end
    if r.running_intersection
        accumulate!(max, lo, lo)
        accumulate!(min, hi, hi)
    end
    return lo, hi
end

_seq_running_pvalue(ev::AbstractVector) =
    isempty(ev) ? Float64[] : min.(1.0, 1 ./ accumulate(max, ev))

"""
    pvalues(r::ConfidenceSequence) -> Vector{Float64}

Always-valid p-value for ``H_0: \\theta =`` `r.null` at the final sample size.

With ``E_s`` the e-process stored in `r.evalue`, the always-valid p-value is

```math
p_n = \\min\\left(1, \\frac{1}{\\max_{s \\le n} E_s}\\right),
```

which is non-increasing in ``n``. Under the null, Ville's inequality gives
``P(\\exists n: p_n \\le \\alpha) \\le \\alpha``, so ``p_\\tau`` is a valid p-value at
any stopping time ``\\tau`` (Johari et al. 2022; Ramdas et al. 2023). The p-value is
dual to the confidence sequence: ``p_n \\le \\alpha`` exactly when some ``C_s`` with
``s \\le n`` has excluded `r.null` at level ``1-\\alpha`` (up to the grid resolution
of the betting boundary). For the asymptotic boundaries the validity is asymptotic in
the same sense as the coverage of the CS. A large p-value is not evidence for the null.

# Arguments
- `r::ConfidenceSequence`: the confidence-sequence path.

# Returns
- `Vector{Float64}` with one element, the always-valid p-value after `nobs(r)`
  observations.

# Examples
```julia
using DrSnow, StableRNGs
cs = confseq_mean(rand(StableRNG(2), 500); method=:betting, bounds=(0, 1), null=0.4)
pvalues(cs)
```

# References
- Johari, R., Koomen, P., Pekelis, L., & Walsh, D. (2022). Always valid inference:
  Continuous monitoring of A/B tests. *Operations Research*, 70(3), 1806–1821.
- Ramdas, A., Grünwald, P., Vovk, V., & Shafer, G. (2023). Game-theoretic statistics
  and safe anytime-valid inference. *Statistical Science*, 38(4), 576–601.
"""
pvalues(r::ConfidenceSequence) = (_seq_check_nonempty(r);
                                  [_seq_running_pvalue(r.evalue)[end]])

function show_details(io::IO, r::ConfidenceSequence)
    isempty(r.n) && return
    lv = round(100 * r.level; digits=1)
    n = r.n[end]
    println(io)
    @printf(io, "Anytime-valid %g%% confidence sequence (%s boundary", lv, r.boundary)
    isfinite(r.t_opt) && @printf(io, ", optimised for n = %g", r.t_opt)
    println(io, ")")
    @printf(io, "At n = %d: [%.6g, %.6g]", n, r.lower[end], r.upper[end])
    s = r.sigma[end]
    if isfinite(s) && s > 0 && isfinite(r.upper[end] - r.lower[end])
        fixed = 2 * critical_value(r.level) * s / sqrt(n)
        @printf(io, "; width %.3g× the fixed-n %g%% CI", (r.upper[end] -
                                                           r.lower[end]) / fixed, lv)
    end
    println(io)
    @printf(io, "Always-valid p-value for H₀: θ = %g: %.4g\n", r.null, pvalues(r)[1])
    println(io, "The interval is valid simultaneously over all sample sizes (continuous " *
                "monitoring and optional stopping allowed); the extra width is the " *
                "price of that guarantee.")
    isempty(r.note) || println(io, "Note: ", r.note)
end

"""
    sequence_path(r) -> DataFrame

The full path of a sequential result, one row per observation, for tables and plots.

A sequential analysis is summarised not only by its final interval but by the path
that led to it: when the interval first excluded the null, how the e-process grew,
and how the width shrank with the sample size. This function returns that path in a
tidy table. The columns depend on the input:

- [`ConfidenceSequence`](@ref): `n`, `estimate`, `lower`, `upper`, `sigma`, `evalue`
  (for ``H_0: \\theta =`` `r.null`) and `pvalue` (always-valid, non-increasing).
- [`SequentialTest`](@ref): `n`, `estimate`, `evalue`, `pvalue`, `lower`, `upper`
  (the always-valid interval; `NaN` when the test has none).
- A monitor ([`MeanMonitor`](@ref), [`ATEMonitor`](@ref), [`MSPRTMonitor`](@ref)): the
  path of its result, which requires the monitor to have been created with
  `record = true`.

Every row of the path is valid simultaneously with every other row; reporting the
interval at an intermediate row chosen after inspecting the path is therefore
legitimate, unlike for a sequence of fixed-``n`` intervals.

# Arguments
- `r`: a [`ConfidenceSequence`](@ref), a [`SequentialTest`](@ref) or a
  [`SequentialMonitor`](@ref).

# Returns
- `DataFrame` with one row per processed observation (per unit for the two-sample
  procedures).

# Examples
```julia
using DrSnow, StableRNGs
cs = confseq_mean(randn(StableRNG(3), 500); method=:asymptotic, t_opt=500)
p = sequence_path(cs)
last(p, 3)
```
"""
function sequence_path(r::ConfidenceSequence)
    return DataFrame(n=copy(r.n), estimate=copy(r.estimate), lower=copy(r.lower),
                     upper=copy(r.upper), sigma=copy(r.sigma), evalue=copy(r.evalue),
                     pvalue=_seq_running_pvalue(r.evalue))
end

# ---------------------------------------------------------------------------
# Sequential tests / e-processes
# ---------------------------------------------------------------------------

"""
    SequentialTest

Path of an anytime-valid sequential test: an e-process for a null hypothesis, the
always-valid p-value derived from it and, when available, the dual always-valid
interval.

An e-process for ``H_0`` is a nonnegative process ``(E_t)`` with
``E[E_\\tau] \\le 1`` under ``H_0`` for every stopping time ``\\tau``; a test
martingale (a nonnegative martingale with initial value one, such as a likelihood
ratio or a mixture of likelihood ratios) is the leading example (Ramdas et al. 2023;
Grünwald, de Heide and Koolen 2024). The always-valid p-value
``p_t = \\min(1, 1/\\max_{s \\le t} E_s)`` then satisfies
``P_{H_0}(\\exists t: p_t \\le \\alpha) \\le \\alpha`` by Ville's inequality. The test may
therefore be evaluated after every observation and stopped at any time, and the
evidence of independent studies can be combined by multiplying e-values. An
e-process has a betting interpretation: ``E_t`` is the wealth of a
gambler who started with one unit and bet against the null at fair odds.

A `SequentialTest` is produced by [`msprt_test`](@ref), by
[`sequential_test`](@ref) from a confidence sequence or a monitor, and is summarised
by [`stopping`](@ref); [`DiagnosticTest`](@ref)`(t)` converts its final state into the
package's standard test result, and `pvalue(t)` / `nobs(t)` return the final
always-valid p-value and sample size.

# Fields
- `name::String`: name of the test.
- `null::String`: the null hypothesis in words.
- `method::String`: the construction (e.g. the boundary or the mixture).
- `n::Vector{Int}`: sample size at each step.
- `estimate::Vector{Float64}`: running point estimate.
- `evalue::Vector{Float64}`: the e-process ``E_t``.
- `pvalue::Vector{Float64}`: the always-valid p-value ``p_t`` (non-increasing).
- `lower::Vector{Float64}`, `upper::Vector{Float64}`: the always-valid interval path
  (`NaN` when the test does not provide one).
- `alpha::Float64`: the design level, used by `stopping` by default.
- `note::String`: assumptions and caveats.

# References
- Wald, A. (1945). Sequential tests of statistical hypotheses. *Annals of
  Mathematical Statistics*, 16(2), 117–186.
- Ramdas, A., Grünwald, P., Vovk, V., & Shafer, G. (2023). Game-theoretic statistics
  and safe anytime-valid inference. *Statistical Science*, 38(4), 576–601.
- Grünwald, P., de Heide, R., & Koolen, W. (2024). Safe testing. *Journal of the Royal
  Statistical Society: Series B*, 86(5), 1091–1128.
- Johari, R., Koomen, P., Pekelis, L., & Walsh, D. (2022). Always valid inference:
  Continuous monitoring of A/B tests. *Operations Research*, 70(3), 1806–1821.
"""
struct SequentialTest
    name::String
    null::String
    method::String
    n::Vector{Int}
    estimate::Vector{Float64}
    evalue::Vector{Float64}
    pvalue::Vector{Float64}
    lower::Vector{Float64}
    upper::Vector{Float64}
    alpha::Float64
    note::String
end

StatsAPI.pvalue(t::SequentialTest) = isempty(t.pvalue) ? 1.0 : t.pvalue[end]
StatsAPI.nobs(t::SequentialTest) = isempty(t.n) ? 0 : t.n[end]

"""
    stopping(t::SequentialTest; alpha=t.alpha) -> NamedTuple
    stopping(r::ConfidenceSequence; alpha=1 - r.level) -> NamedTuple

Stopping summary of a sequential test monitored continuously at level `alpha`: the
first step at which the always-valid p-value fell to `alpha` or below, equivalently
the first step at which the e-process reached ``1/\\alpha``.

The rule "stop and reject ``H_0`` at ``\\tau = \\inf\\{t: E_t \\ge 1/\\alpha\\}``" is a
level-``\\alpha`` sequential test whatever the sampling plan: by Ville's inequality
the probability under ``H_0`` that ``\\tau`` is ever finite is at most ``\\alpha``
(Wald 1945; Ramdas et al. 2023). Continuing past ``\\tau`` is also allowed, because
the always-valid p-value can only decrease. For a [`ConfidenceSequence`](@ref) the
first crossing is the first sample size at which the CS at level ``1-\\alpha``
excluded `r.null`.

When the null was never rejected, the summary reports the final state. Non-rejection
is not evidence for the null: it may reflect a small effect, a small sample, or a
boundary tuned for a much larger `t_opt`. The point estimate at the stopping time is
biased away from the null (stopping when the estimate happens to be extreme selects
large estimates), so report the confidence sequence at ``\\tau`` rather than the
estimate alone.

# Arguments
- `t::SequentialTest` or `r::ConfidenceSequence`: the sequential result.

# Keywords
- `alpha::Real`: the monitoring level in ``(0, 1)``; defaults to the design level of
  the test (`t.alpha`) or ``1 -`` `r.level` for a confidence sequence. A level chosen
  after seeing the path is not covered by the guarantee.

# Returns
A `NamedTuple` with fields `rejected::Bool`; `step` (index into the path, `nothing`
when not rejected); `n` (sample size at rejection, or at the end); `estimate`,
`evalue` and `pvalue` at that step; `max_evalue` (running maximum of the e-process
over the whole path) and `final_pvalue`.

# Examples
```julia
using DrSnow, DataFrames, StableRNGs
rng = StableRNG(4)
d = rand(rng, 0:1, 2000)
df = DataFrame(y=0.3 .* d .+ randn(rng, 2000), d=d)
t = msprt_test(df, :y, :d; n_opt=2000)
s = stopping(t)
s.rejected && println("first rejection at n = ", s.n)
```

# References
- Wald, A. (1945). Sequential tests of statistical hypotheses. *Annals of
  Mathematical Statistics*, 16(2), 117–186.
- Ramdas, A., Grünwald, P., Vovk, V., & Shafer, G. (2023). Game-theoretic statistics
  and safe anytime-valid inference. *Statistical Science*, 38(4), 576–601.
"""
function stopping(t::SequentialTest; alpha::Real=t.alpha)
    (0 < alpha < 1) || throw(ArgumentError("alpha must be in (0, 1), got $alpha"))
    return _seq_stopping(t.n, t.estimate, t.evalue, t.pvalue, alpha)
end

function _seq_stopping(n, est, ev, pv, alpha)
    isempty(n) && throw(ArgumentError("the sequential test has no observations"))
    k = findfirst(<=(alpha), pv)
    i = k === nothing ? length(n) : k
    return (rejected=k !== nothing, step=k, n=n[i], estimate=est[i], evalue=ev[i],
            pvalue=pv[i], max_evalue=maximum(ev), final_pvalue=pv[end])
end

function stopping(r::ConfidenceSequence; alpha::Real=1 - r.level)
    (0 < alpha < 1) || throw(ArgumentError("alpha must be in (0, 1), got $alpha"))
    return _seq_stopping(r.n, r.estimate, r.evalue, _seq_running_pvalue(r.evalue), alpha)
end

"""
    sequential_test(r::ConfidenceSequence) -> SequentialTest
    sequential_test(m::SequentialMonitor) -> SequentialTest

The anytime-valid test of ``H_0: \\theta =`` `r.null` dual to a confidence sequence,
or the test accumulated by a streaming monitor.

Confidence sequences and sequential tests are two views of the same object: the CS
is the set of parameter values whose e-process has not yet reached ``1/\\alpha``, and
the test of a single value ``\\theta_0`` rejects as soon as ``\\theta_0`` leaves the
CS (Howard et al. 2021; Ramdas et al. 2023). This function exposes the e-process, the
always-valid p-values and the CS itself (as the interval path) of a
[`ConfidenceSequence`](@ref) in the form of a [`SequentialTest`](@ref), with the
design level ``1 -`` `r.level`. For a monitor it is applied to the monitor's recorded
path; for an [`MSPRTMonitor`](@ref), which is a test rather than a CS, it returns the
mixture-SPRT path directly.

# Arguments
- `r::ConfidenceSequence`: a confidence-sequence path.
- `m::SequentialMonitor`: a monitor created with `record = true`.

# Returns
- [`SequentialTest`](@ref).

# Examples
```julia
using DrSnow, StableRNGs
x = rand(StableRNG(5), 800) .< 0.58          # Bernoulli(0.58) stream
cs = confseq_mean(x; method=:betting, bounds=(0, 1), null=0.5)
t = sequential_test(cs)
stopping(t)
```

# References
- Howard, S. R., Ramdas, A., McAuliffe, J., & Sekhon, J. (2021). Time-uniform,
  nonparametric, nonasymptotic confidence sequences. *Annals of Statistics*, 49(2),
  1055–1080.
- Ramdas, A., Grünwald, P., Vovk, V., & Shafer, G. (2023). Game-theoretic statistics
  and safe anytime-valid inference. *Statistical Science*, 38(4), 576–601.
"""
function sequential_test(r::ConfidenceSequence)
    return SequentialTest("Anytime-valid test ($(r.method))",
                          "$(r.estimand) = $(r.null)", r.method, copy(r.n),
                          copy(r.estimate), copy(r.evalue),
                          _seq_running_pvalue(r.evalue), copy(r.lower), copy(r.upper),
                          1 - r.level, r.note)
end

function sequence_path(t::SequentialTest)
    return DataFrame(n=copy(t.n), estimate=copy(t.estimate), evalue=copy(t.evalue),
                     pvalue=copy(t.pvalue), lower=copy(t.lower), upper=copy(t.upper))
end

"""
    DiagnosticTest(t::SequentialTest) -> DiagnosticTest

Final state of a sequential test as a [`DiagnosticTest`](@ref), the package's
standard test result.

The statistic is the running maximum of the e-process, ``\\max_{s \\le n} E_s``, and the
p-value the always-valid p-value ``\\min(1, 1/\\max_{s \\le n} E_s)`` after the last
observation; both remain valid under continuous monitoring and optional stopping. The
`details` field holds the final sample size, estimate and e-value and the
[`stopping`](@ref) summary. Use this conversion to report a sequential test alongside
the other diagnostics of an analysis; the full path remains available from
[`sequence_path`](@ref).

# Arguments
- `t::SequentialTest`: a sequential test with at least one observation.

# Returns
- [`DiagnosticTest`](@ref) with the name, null and method of `t`.

# Examples
```julia
using DrSnow, StableRNGs
cs = confseq_mean(randn(StableRNG(6), 300) .+ 0.3; method=:asymptotic, t_opt=300)
DiagnosticTest(sequential_test(cs))
```
"""
function DiagnosticTest(t::SequentialTest)
    isempty(t.n) && throw(ArgumentError("the sequential test has no observations"))
    note = "Always-valid p-value: valid under continuous monitoring and optional " *
           "stopping (Ville's inequality). Statistic = running maximum of the " *
           "e-process after n = $(t.n[end]) observations."
    isempty(t.note) || (note *= " " * t.note)
    return DiagnosticTest(t.name, t.null, maximum(t.evalue), t.pvalue[end];
                          method=t.method, note=note,
                          details=(n=t.n[end], estimate=t.estimate[end],
                                   evalue=t.evalue[end], stopping=stopping(t)))
end

function Base.show(io::IO, ::MIME"text/plain", t::SequentialTest)
    println(io, t.name)
    println(io, "H₀: ", t.null)
    isempty(t.method) || println(io, "Method: ", t.method)
    if isempty(t.n)
        println(io, "No observations yet.")
        return
    end
    @printf(io, "n = %d, estimate = %.5g, e-value = %.4g (max %.4g), ",
            t.n[end], t.estimate[end], t.evalue[end], maximum(t.evalue))
    @printf(io, "always-valid p = %.4g\n", t.pvalue[end])
    if isfinite(t.lower[end]) || isfinite(t.upper[end])
        @printf(io, "Always-valid %g%% interval: [%.5g, %.5g]\n",
                100 * (1 - t.alpha), t.lower[end], t.upper[end])
    end
    s = stopping(t)
    if s.rejected
        @printf(io, "H₀ rejected at α = %g, first at n = %d (e-value %.4g ≥ 1/α).\n",
                t.alpha, s.n, s.evalue)
    else
        @printf(io, "H₀ not rejected at α = %g so far.", t.alpha)
        println(io, " Non-rejection is not evidence that H₀ is true.")
    end
    isempty(t.note) || println(io, "Note: ", t.note)
end

Base.show(io::IO, t::SequentialTest) =
    @printf(io, "%s: n = %d, p = %.4g", t.name, nobs(t), pvalue(t))
