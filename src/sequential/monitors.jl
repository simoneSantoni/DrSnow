# Streaming API: monitors updated observation by observation (or batch by batch)
# with `fit!`, queried with `snapshot`, and turned into full-path results with
# `confidence_sequence` / `sequential_test` / `sequence_path`.

"""
    SequentialMonitor

Abstract supertype of the streaming objects of the sequential-inference area.

A monitor implements the sequential analysis of a stream of data as it arrives: it is
created before the first observation with its design choices fixed (boundary, level,
``t_{\\text{opt}}``, null value), updated with `fit!(monitor, data...)` one
observation or one batch at a time, and queried at any time with
[`snapshot`](@ref), which returns the current anytime-valid interval and
always-valid p-value. Because the guarantees of confidence sequences and e-processes
are time-uniform, the snapshot may be read after every update and used to decide
whether to continue, without any correction for the number of looks (Howard et al.
2021; Ramdas et al. 2023). Fixing the design choices at construction, before the data
are seen, is what makes the guarantee meaningful; the batch functions
[`confseq_mean`](@ref), [`confseq_ate`](@ref) and [`msprt_test`](@ref) feed the same
monitors and give identical results.

When created with `record = true`, a monitor keeps its path, which
[`confidence_sequence`](@ref), [`sequential_test`](@ref) and [`sequence_path`](@ref)
return as a full result. Subtypes: [`MeanMonitor`](@ref) (mean of one stream),
[`ATEMonitor`](@ref) (average treatment effect, AIPW) and [`MSPRTMonitor`](@ref)
(mixture SPRT for a two-arm difference in means).

# References
- Howard, S. R., Ramdas, A., McAuliffe, J., & Sekhon, J. (2021). Time-uniform,
  nonparametric, nonasymptotic confidence sequences. *Annals of Statistics*, 49(2),
  1055–1080.
- Ramdas, A., Grünwald, P., Vovk, V., & Shafer, G. (2023). Game-theoretic statistics
  and safe anytime-valid inference. *Statistical Science*, 38(4), 576–601.
"""
abstract type SequentialMonitor end

# Shared path recorder.
mutable struct _SeqPath
    record::Bool
    n::Vector{Int}
    estimate::Vector{Float64}
    lower::Vector{Float64}
    upper::Vector{Float64}
    sigma::Vector{Float64}
    evalue::Vector{Float64}
end
_SeqPath(record::Bool) = _SeqPath(record, Int[], Float64[], Float64[], Float64[],
                                  Float64[], Float64[])

function _seq_record!(p::_SeqPath, n, est, lo, hi, sig, ev)
    push!(p.n, n); push!(p.estimate, est); push!(p.lower, lo); push!(p.upper, hi)
    push!(p.sigma, sig); push!(p.evalue, ev)
    return p
end

function _seq_check_level(level)
    (0 < level < 1) || throw(ArgumentError("level must be in (0, 1), got $level"))
    return nothing
end

function _seq_check_topt(t_opt)
    t_opt === nothing && return NaN
    (t_opt isa Real && isfinite(t_opt) && t_opt > 0) ||
        throw(ArgumentError("t_opt must be a positive number, got $t_opt"))
    return float(t_opt)
end

"""
    MeanMonitor(; method=:asymptotic, level=0.95, t_opt=nothing, bounds=nothing,
                sigma=nothing, null=0.0, running_intersection=true, breaks=1000,
                min_n=20, record=true, truncation=nothing) -> MeanMonitor

Streaming anytime-valid confidence sequence for the mean of a stream of observations.

The estimand is the common mean ``\\mu`` of observations ``X_1, X_2, \\ldots``
arriving one at a time. For the Gaussian boundaries the observations are assumed
i.i.d. (finite variance for `:asymptotic`, ``\\sigma``-sub-Gaussian for
`:normal_mixture`); for the bounded-data boundaries it suffices that each observation
lies in the known range `bounds` and has conditional mean ``\\mu`` given the past,
which allows dependence and heterogeneity over time (Waudby-Smith and Ramdas 2024).
The monitor returns, after every observation ``t``, an interval ``C_t`` with
``P(\\mu \\in C_t \\text{ for all } t) \\ge 1 - \\alpha``, and the e-process and
always-valid p-value for ``H_0: \\mu =`` `null`. Feed it with `fit!(m, x)` (a number
or a vector); read the current state with [`snapshot`](@ref) and the path with
[`confidence_sequence`](@ref).

**Boundaries (`method`).** Five constructions are available, in two families.

- `:asymptotic`: the asymptotic confidence sequence of Waudby-Smith et al. (2024),
  the time-uniform analogue of the CLT interval,
  ``\\hat\\mu_t \\pm \\hat\\sigma_t \\sqrt{(t+\\rho)\\{\\log(1+t/\\rho) +
  2\\log(1/\\alpha)\\}}/t``, with ``\\hat\\sigma_t`` the running standard deviation.
  It needs no bound on the data, but its guarantee is asymptotic: the coverage
  approaches ``1-\\alpha`` as monitoring starts later (see `min_n`).
- `:normal_mixture`: Robbins' (1970) two-sided normal-mixture boundary with the known
  scale ``\\sigma`` in place of ``\\hat\\sigma_t`` (Howard et al. 2021). Exact for
  ``\\sigma``-sub-Gaussian observations; it undercovers if the true scale exceeds
  `sigma`.
- `:hoeffding`: the predictable-mixture Hoeffding CS of Waudby-Smith and Ramdas
  (2024, Thm 2). Exact and nonasymptotic for data in `bounds`; it ignores the
  variance and is therefore conservative for low-variance outcomes.
- `:empirical_bernstein`: the predictable-mixture empirical-Bernstein CS
  (Waudby-Smith and Ramdas 2024). Exact; adapts to the variance through a running
  variance estimate.
- `:betting`: the hedged-capital betting CS (Waudby-Smith and Ramdas 2024). Exact
  and, in their comparisons, the tightest of the bounded-data boundaries. The interval
  is computed on a grid of `breaks + 1` points of the rescaled range, so its
  resolution is ``(\\text{hi} - \\text{lo})/\\text{breaks}`` and the reported interval
  is widened by one grid step on each side.

Use a bounded-data boundary whenever the outcome has known bounds (binary outcomes,
proportions, ratings, capped revenue): the guarantee then holds exactly at every
``n``. Use `:asymptotic` for unbounded outcomes, and `:normal_mixture` only when the
scale is genuinely known.

**Tuning (`t_opt`).** No confidence sequence is uniformly tightest: each boundary is
tuned to be tightest around one sample size, ``t_{\\text{opt}}``, typically the
planned sample size or the sample size at which a decision is most likely. For the
Gaussian boundaries ``t_{\\text{opt}}`` sets the mixture precision

```math
\\rho = \\frac{t_{\\text{opt}}}{2\\log(1/\\alpha) + \\log\\{1 + 2\\log(1/\\alpha)\\}},
```

which minimises the boundary at ``t = t_{\\text{opt}}`` (Howard et al. 2021); it is
required for these methods. For the bounded-data boundaries, `t_opt = nothing` uses
bets ``\\lambda_t \\propto 1/\\sqrt{t \\log(1+t)}``, which perform well over a wide
range of ``n``, while a number tunes the bets to that sample size (tighter near it,
looser far from it). The guarantee requires ``t_{\\text{opt}}`` to be fixed before the
data are seen; choosing it after the fact, for instance equal to a sample size at
which the analyst stopped because the results looked favourable, makes the boundary
data-dependent. Pre-register it with the design.

**Inference and practice.** With `running_intersection = true` the reported interval
is ``\\cap_{s \\le t} C_s``, which is still a valid CS and never wider. The e-process
used for the p-value is the normal-mixture martingale (Gaussian boundaries) or the
capital process of the bets (bounded-data boundaries), so the p-value and the CS are
dual. For two-arm experiments use [`ATEMonitor`](@ref) or [`MSPRTMonitor`](@ref);
for a fixed number of planned interim analyses a group-sequential design
([`gs_design`](@ref)) is more powerful. Report the boundary, the level and
``t_{\\text{opt}}`` together with the interval.

# Keywords
- `method::Symbol = :asymptotic`: the boundary, one of `:asymptotic`,
  `:normal_mixture`, `:hoeffding`, `:empirical_bernstein`, `:betting`.
- `level::Real = 0.95`: coverage ``1-\\alpha`` of the confidence sequence.
- `t_opt = nothing`: sample size at which the boundary is tightest. Required for the
  Gaussian boundaries; optional for the bounded ones (`nothing` gives untuned bets).
- `bounds = nothing`: known range `(lo, hi)` of the observations, required by the
  bounded-data boundaries; an observation outside it throws an error.
- `sigma = nothing`: known sub-Gaussian scale, required by `:normal_mixture`.
- `null::Real = 0.0`: null value of the e-process and the always-valid p-value; it
  must lie within `bounds` for the bounded-data boundaries.
- `running_intersection::Bool = true`: report ``\\cap_{s \\le t} C_s`` rather than
  ``C_t``.
- `breaks::Integer = 1000`: grid resolution of the betting CS.
- `min_n::Integer = 20`: `:asymptotic` only. The CS and the e-process start at
  ``n =`` `min_n`; before that the interval is ``(-\\infty, \\infty)`` and the e-value
  one. The asymptotic guarantee is for monitoring that starts late enough for
  ``\\hat\\sigma_t`` to be reliable: in the package's simulations, starting at
  ``n = 1`` gives time-uniform coverage of about 0.91 (normal data) and 0.83
  (exponential data) at nominal 0.95, starting at ``n = 20`` about 0.97 and 0.95.
- `record::Bool = true`: keep the path, needed by `confidence_sequence` and
  `sequence_path`; `false` saves memory for long streams.
- `truncation = nothing`: `:hoeffding` and `:empirical_bernstein` only; the largest
  bet ``\\lambda_t`` allowed (default 0.5 for empirical Bernstein and 1 for Hoeffding,
  the defaults of the `confseq` reference implementation of Howard et al.).

# Returns
- `MeanMonitor`, a [`SequentialMonitor`](@ref).

# Examples
```julia
using DrSnow, StableRNGs
rng = StableRNG(7)
m = MeanMonitor(; method=:betting, bounds=(0, 1), null=0.5)
for i in 1:5_000
    fit!(m, rand(rng) < 0.56)          # binary outcome with mean 0.56
    s = snapshot(m)
    s.lower > 0.5 && break             # stopping on the interval is allowed
end
cs = confidence_sequence(m)
confint(cs), nobs(cs)
```

# References
- Robbins, H. (1970). Statistical methods related to the law of the iterated
  logarithm. *Annals of Mathematical Statistics*, 41(5), 1397–1409.
- Darling, D. A., & Robbins, H. (1967). Confidence sequences for mean, variance, and
  median. *Proceedings of the National Academy of Sciences*, 58(1), 66–68.
- Howard, S. R., Ramdas, A., McAuliffe, J., & Sekhon, J. (2020). Time-uniform
  Chernoff bounds via nonnegative supermartingales. *Probability Surveys*, 17,
  257–317.
- Howard, S. R., Ramdas, A., McAuliffe, J., & Sekhon, J. (2021). Time-uniform,
  nonparametric, nonasymptotic confidence sequences. *Annals of Statistics*, 49(2),
  1055–1080.
- Waudby-Smith, I., & Ramdas, A. (2024). Estimating means of bounded random variables
  by betting. *Journal of the Royal Statistical Society: Series B*, 86(1), 1–27.
- Waudby-Smith, I., Arbour, D., Sinha, R., Kennedy, E. H., & Ramdas, A. (2024).
  Time-uniform central limit theory and asymptotic confidence sequences. *Annals of
  Statistics*, 52(6), 2613–2640.
"""
mutable struct MeanMonitor{E<:_SeqEngine} <: SequentialMonitor
    method::Symbol
    level::Float64
    t_opt::Float64
    lo::Float64
    hi::Float64
    null::Float64
    running_intersection::Bool
    engine::E
    moments::_SeqGaussEngine     # raw-scale running mean / sd (for reporting)
    cur_lower::Float64
    cur_upper::Float64
    max_logev::Float64
    path::_SeqPath
end

function MeanMonitor(; method::Symbol=:asymptotic, level::Real=0.95, t_opt=nothing,
                     bounds=nothing, sigma=nothing, null::Real=0.0,
                     running_intersection::Bool=true, breaks::Integer=1000,
                     min_n::Integer=20, record::Bool=true, truncation=nothing)
    _seq_check_level(level)
    method in _SEQ_BOUNDARIES ||
        throw(ArgumentError("method must be one of $(_SEQ_BOUNDARIES), got :$method"))
    alpha = 1 - level
    topt = _seq_check_topt(t_opt)
    lo, hi = -Inf, Inf
    if method in (:hoeffding, :empirical_bernstein, :betting)
        bounds === nothing &&
            throw(ArgumentError("method = :$method needs the known range of the " *
                                "data: bounds = (lo, hi)"))
        lo, hi = float.(bounds)
        (isfinite(lo) && isfinite(hi) && lo < hi) ||
            throw(ArgumentError("bounds must be finite with lo < hi, got $bounds"))
        (lo <= null <= hi) ||
            throw(ArgumentError("null = $null lies outside bounds $bounds"))
        null01 = (null - lo) / (hi - lo)
        engine = method === :betting ? _SeqBettingEngine(alpha, topt, null01, breaks) :
                 _SeqPredmixEngine(method === :empirical_bernstein, alpha, topt, null01;
                                   truncation=truncation)
    else
        isnan(topt) &&
            throw(ArgumentError("method = :$method needs t_opt, the sample size at " *
                                "which the boundary should be tightest"))
        sig = NaN
        if method === :normal_mixture
            sigma === nothing &&
                throw(ArgumentError("method = :normal_mixture needs the known " *
                                    "sub-Gaussian scale `sigma`; use :asymptotic to " *
                                    "estimate it"))
            (sigma > 0 && isfinite(sigma)) ||
                throw(ArgumentError("sigma must be positive, got $sigma"))
            sig = float(sigma)
        end
        min_n >= 2 || throw(ArgumentError("min_n must be at least 2"))
        engine = _SeqGaussEngine(sig, _seq_nm_rho(topt, alpha), alpha, float(null);
                                 min_n=min_n)
    end
    return MeanMonitor(method, float(level), topt, lo, hi, float(null),
                       running_intersection, engine,
                       _SeqGaussEngine(NaN, 1.0, alpha, 0.0), -Inf, Inf, 0.0,
                       _SeqPath(record))
end

"""
    fit!(m::MeanMonitor, x) -> m
    fit!(m::ATEMonitor, y, d[, X]; propensity=nothing) -> m
    fit!(m::ATEMonitor) -> m
    fit!(m::MSPRTMonitor, y, d) -> m

Update a sequential monitor with newly arrived data, one observation or a batch.

The monitors are online algorithms: each call processes the new observations in the
order given, updates the running estimates, the e-process and the confidence
sequence, and leaves the monitor ready for the next arrival. Feeding a batch gives
exactly the same state as feeding its observations one at a time, so data may be
passed in whatever chunks they arrive in. The validity of the anytime-valid guarantees
does not depend on how often the monitor is updated or read.

The methods are:

- `fit!(m::MeanMonitor, x)`: `x` a number or a vector of numbers. For the bounded
  boundaries every observation must lie within the monitor's `bounds`.
- `fit!(m::ATEMonitor, y, d[, X]; propensity)`: outcome(s) `y`, 0/1 treatment
  indicator(s) `d` and optional covariates (a vector for one unit, a matrix with one
  row per unit for a batch). `propensity` gives the known assignment probability of
  each unit when it varies across units (a number for one unit, a vector for a batch).
  With nuisance learners the units are processed in blocks of `refit_every`;
  `fit!(m)` with no data processes the pending units immediately.
- `fit!(m::MSPRTMonitor, y, d)`: outcome(s) and 0/1 arm indicator(s); binary outcomes
  must be coded 0/1.

# Arguments
- `m::SequentialMonitor`: the monitor to update (mutated in place).
- `x`, `y`, `d`, `X`: the new data, as described above; all values must be finite.

# Keywords
- `propensity = nothing`: [`ATEMonitor`](@ref) only; known per-unit assignment
  probabilities in ``(0, 1)``. Cannot be combined with a constant `propensity` or a
  `propensity_learner` given at construction.

# Returns
- The updated monitor `m`.

# Examples
```julia
using DrSnow, StableRNGs
rng = StableRNG(8)
m = MeanMonitor(; method=:asymptotic, t_opt=1000)
fit!(m, randn(rng, 100))           # a batch
fit!(m, 0.3)                       # a single observation
snapshot(m)
```
"""
StatsAPI.fit!(m::SequentialMonitor, args...) =
    throw(ArgumentError("fit! is not defined for $(typeof(m)) with these arguments"))

function StatsAPI.fit!(m::MeanMonitor, x::Real)
    isfinite(x) || throw(ArgumentError("observations must be finite, got $x"))
    if isfinite(m.lo)
        (m.lo <= x <= m.hi) ||
            throw(ArgumentError("observation $x lies outside bounds ($(m.lo), $(m.hi))"))
        _seq_push!(m.engine, (x - m.lo) / (m.hi - m.lo))
    else
        _seq_push!(m.engine, x)
    end
    _seq_push!(m.moments, x)
    l, u = _seq_interval(m.engine)
    if isfinite(m.lo)
        l, u = m.lo + (m.hi - m.lo) * l, m.lo + (m.hi - m.lo) * u
    end
    if m.running_intersection
        m.cur_lower = max(m.cur_lower, l)
        m.cur_upper = min(m.cur_upper, u)
    else
        m.cur_lower, m.cur_upper = l, u
    end
    lev = _seq_logevalue(m.engine)
    m.max_logev = max(m.max_logev, lev)
    if m.path.record
        sig = m.method === :normal_mixture ? m.engine.sigma : _seq_scale(m.moments)
        _seq_record!(m.path, m.moments.n, m.moments.mean, m.cur_lower, m.cur_upper, sig,
                     exp(lev))
    end
    return m
end

function StatsAPI.fit!(m::MeanMonitor, x::AbstractVector{<:Real})
    for xi in x
        fit!(m, xi)
    end
    return m
end

StatsAPI.nobs(m::MeanMonitor) = m.moments.n

"""
    snapshot(m::SequentialMonitor) -> NamedTuple

Current state of a sequential monitor: the estimate, the anytime-valid interval and
the always-valid p-value after the observations processed so far.

The snapshot is the object an analyst consults when monitoring an experiment
continuously. Reading it does not change the monitor and may be done after every
observation; decisions taken on the basis of it (stop, continue, stop when the
interval excludes zero or is narrower than a target precision) do not invalidate the
coverage of the interval or the level of the p-value, because both guarantees hold
uniformly over time (Howard et al. 2021; Johari et al. 2022).

The fields are `n` (observations or units processed), `estimate`, `lower` and
`upper` (the current anytime-valid interval, ``(-\\infty, \\infty)`` before monitoring
starts), `evalue` (the current value of the e-process for the null) and `pvalue` (the
always-valid p-value, ``\\min(1, 1/\\max_{s \\le n} E_s)``). An [`ATEMonitor`](@ref)
adds `n_treated`, `n_control` and `n_pending` (units received but not yet processed
because the current block of the nuisance learners is incomplete); an
[`MSPRTMonitor`](@ref) adds `n_treated`, `n_control` and `tau` (the mixing scale, `NaN`
before it is tuned at the end of the burn-in).

# Arguments
- `m::SequentialMonitor`: a [`MeanMonitor`](@ref), [`ATEMonitor`](@ref) or
  [`MSPRTMonitor`](@ref).

# Returns
- `NamedTuple` with the fields listed above.

# Examples
```julia
using DrSnow, StableRNGs
m = MeanMonitor(; method=:asymptotic, t_opt=500)
fit!(m, randn(StableRNG(9), 50))
s = snapshot(m)
s.lower, s.upper, s.pvalue
```

# References
- Howard, S. R., Ramdas, A., McAuliffe, J., & Sekhon, J. (2021). Time-uniform,
  nonparametric, nonasymptotic confidence sequences. *Annals of Statistics*, 49(2),
  1055–1080.
- Johari, R., Koomen, P., Pekelis, L., & Walsh, D. (2022). Always valid inference:
  Continuous monitoring of A/B tests. *Operations Research*, 70(3), 1806–1821.
"""
function snapshot(m::MeanMonitor)
    return (n=m.moments.n, estimate=_seq_estimate(m.moments), lower=m.cur_lower,
            upper=m.cur_upper, evalue=exp(_seq_logevalue(m.engine)),
            pvalue=min(1.0, exp(-m.max_logev)))
end

const _SEQ_METHOD_NAMES = Dict(
    :asymptotic => "Asymptotic confidence sequence",
    :normal_mixture => "Normal-mixture confidence sequence (known σ)",
    :hoeffding => "Predictable-mixture Hoeffding confidence sequence",
    :empirical_bernstein => "Predictable-mixture empirical-Bernstein confidence sequence",
    :betting => "Betting (hedged capital) confidence sequence")

const _SEQ_METHOD_NOTES = Dict(
    :asymptotic => "Asymptotic CS: time-uniform coverage holds approximately (in the " *
                   "limit of a late start), assuming i.i.d. data with finite variance.",
    :normal_mixture => "Exact for σ-sub-Gaussian observations with the stated σ; " *
                       "undercovers if the true scale exceeds σ.",
    :hoeffding => "Exact (nonasymptotic) for observations within the stated bounds.",
    :empirical_bernstein => "Exact (nonasymptotic) for observations within the stated " *
                            "bounds.",
    :betting => "Exact (nonasymptotic) for observations within the stated bounds; " *
                "interval resolved on a grid.")

"""
    confidence_sequence(m::MeanMonitor) -> ConfidenceSequence
    confidence_sequence(m::ATEMonitor) -> ConfidenceSequence

Full-path result of a streaming monitor: the confidence sequence after every
observation processed so far.

The monitor must have been created with `record = true`. The result carries the
boundary, level, ``t_{\\text{opt}}`` and caveats of the monitor, and supports the
standard accessors (`coef`, `confint`, `pvalues`, `nobs`), [`sequence_path`](@ref),
[`sequential_test`](@ref) and [`stopping`](@ref). It may be requested at any time,
including while data are still arriving; the path returned then ends at the current
observation. For an [`ATEMonitor`](@ref) with nuisance learners, units of an
incomplete block are not included until they are processed (call `fit!(m)` first);
the note of the result reports how many are pending. An [`MSPRTMonitor`](@ref) is a
test rather than a confidence sequence: use [`sequential_test`](@ref) for it.

# Arguments
- `m`: a [`MeanMonitor`](@ref) or [`ATEMonitor`](@ref) created with `record = true`.

# Returns
- [`ConfidenceSequence`](@ref).

# Examples
```julia
using DrSnow, StableRNGs
m = MeanMonitor(; method=:betting, bounds=(0, 1))
fit!(m, rand(StableRNG(10), 200))
cs = confidence_sequence(m)
last(sequence_path(cs), 2)
```
"""
function confidence_sequence(m::MeanMonitor)
    m.path.record ||
        throw(ArgumentError("the monitor was created with record = false; no path is " *
                            "stored (use `snapshot` for the current state)"))
    p = m.path
    rho = m.engine isa _SeqGaussEngine ? m.engine.rho : NaN
    return ConfidenceSequence("mean", _SEQ_METHOD_NAMES[m.method], m.method, copy(p.n),
                              copy(p.estimate), copy(p.lower), copy(p.upper),
                              copy(p.sigma), copy(p.evalue), m.null, m.level, m.t_opt,
                              rho, m.running_intersection, _SEQ_METHOD_NOTES[m.method])
end

sequential_test(m::SequentialMonitor) = sequential_test(confidence_sequence(m))
sequence_path(m::SequentialMonitor) = sequence_path(confidence_sequence(m))

"""
    confseq_mean(x; method=:asymptotic, level=0.95, t_opt=nothing, bounds=nothing,
                 sigma=nothing, null=0.0, running_intersection=true, breaks=1000,
                 min_n=20, truncation=nothing) -> ConfidenceSequence

Anytime-valid confidence sequence for the mean of the observations `x`, taken in the
order given (the order of arrival).

This is the batch interface to [`MeanMonitor`](@ref): the observations are fed to a
monitor one at a time and the full path is returned, so the result is identical to
monitoring the stream as it arrived. The estimand, the five boundaries and their
assumptions are described in [`MeanMonitor`](@ref). The interval at every prefix of
`x` is valid simultaneously, so the path shows what an analyst monitoring
continuously would have seen and when the interval first excluded `null`.

**Default tuning to the realized sample size.** For the Gaussian boundaries
(`:asymptotic`, `:normal_mixture`), `t_opt = nothing` sets ``t_{\\text{opt}}`` to
`length(x)`, so the boundary is tightest at the end of the data at hand. This is
harmless when the number of observations was fixed in advance independently of the
data, but when `length(x)` is itself the outcome of monitoring (the data were
collected until the results looked conclusive) the boundary becomes data-dependent
and the time-uniform guarantee no longer strictly applies. For confirmatory analyses,
pass the pre-registered planned sample size as `t_opt`; for interim analyses, pass the
planned final sample size rather than the current one. The bounded-data boundaries
are untuned by default (`t_opt = nothing`) and are not affected.

# Arguments
- `x::AbstractVector{<:Real}`: observations in arrival order; must be non-empty and
  finite.

# Keywords
- `method`, `level`, `bounds`, `sigma`, `null`, `running_intersection`, `breaks`,
  `min_n`, `truncation`: see [`MeanMonitor`](@ref).
- `t_opt = nothing`: sample size at which the boundary is tightest; for the Gaussian
  boundaries `nothing` means `length(x)` (see above).

# Returns
- [`ConfidenceSequence`](@ref) with estimand `"mean"`.

# Examples
```julia
using DrSnow, StableRNGs
rng = StableRNG(11)
x = rand(rng, 1000) .< 0.3                      # Bernoulli(0.3) stream
cs = confseq_mean(x; method=:betting, bounds=(0, 1))
confint(cs)                                      # CS at n = 1000
y = randn(rng, 1000) .+ 1
cs2 = confseq_mean(y; method=:asymptotic, t_opt=2000)   # pre-registered t_opt
stopping(cs2)                                    # first n excluding 0
```

# References
- Howard, S. R., Ramdas, A., McAuliffe, J., & Sekhon, J. (2021). Time-uniform,
  nonparametric, nonasymptotic confidence sequences. *Annals of Statistics*, 49(2),
  1055–1080.
- Waudby-Smith, I., & Ramdas, A. (2024). Estimating means of bounded random variables
  by betting. *Journal of the Royal Statistical Society: Series B*, 86(1), 1–27.
- Waudby-Smith, I., Arbour, D., Sinha, R., Kennedy, E. H., & Ramdas, A. (2024).
  Time-uniform central limit theory and asymptotic confidence sequences. *Annals of
  Statistics*, 52(6), 2613–2640.
"""
function confseq_mean(x::AbstractVector{<:Real}; method::Symbol=:asymptotic,
                      level::Real=0.95, t_opt=nothing, kwargs...)
    isempty(x) && throw(ArgumentError("x is empty"))
    if t_opt === nothing && method in (:asymptotic, :normal_mixture)
        t_opt = length(x)
    end
    m = MeanMonitor(; method=method, level=level, t_opt=t_opt, kwargs...)
    fit!(m, x)
    return confidence_sequence(m)
end
