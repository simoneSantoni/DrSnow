# Mixture sequential probability ratio test (mSPRT) for two-sample A/B tests:
# always-valid p-values and confidence intervals (Johari, Koomen, Pekelis & Walsh
# 2022; Robbins 1970).
#
# With θ̂ the difference in arm means, V its variance and a N(θ₀, τ²) mixing
# distribution over the alternative, the mixture likelihood ratio is
#     Λ = √(V / (V + τ²)) · exp(τ² (θ̂ - θ₀)² / (2 V (V + τ²))),
# a test martingale under H₀: θ = θ₀ for Gaussian data (paired arrivals); the
# always-valid p-value is p_n = min(p_{n-1}, 1/Λ_n), and inverting Λ < 1/α gives the
# always-valid interval
#     θ̂ ± √(V (V + τ²) / τ² · (2 log(1/α) + log((V + τ²)/V))).

"""
    MSPRTMonitor(; outcome_type=:continuous, null=0.0, alpha=0.05, tau=nothing,
                 n_opt=nothing, sigma=nothing, burn_in=10, record=true)
        -> MSPRTMonitor

Streaming mixture sequential probability ratio test (mSPRT) of ``H_0: \\theta =``
`null` for the difference in means ``\\theta = E[Y \\mid D = 1] - E[Y \\mid D = 0]``
of a two-arm experiment, with always-valid p-values and confidence intervals.

The mSPRT, popularised for online A/B testing by Johari et al. (2022), replaces the
single alternative of Wald's (1945) SPRT by a normal mixture over effect sizes,
``\\theta \\sim N(\\theta_0, \\tau^2)`` (Robbins 1970). With ``\\hat\\theta_n`` the
difference in arm means after ``n`` units and ``V_n`` its variance, the mixture
likelihood ratio is

```math
\\Lambda_n = \\sqrt{\\frac{V_n}{V_n + \\tau^2}}
  \\exp\\left\\{\\frac{\\tau^2 (\\hat\\theta_n - \\theta_0)^2}
  {2 V_n (V_n + \\tau^2)}\\right\\},
```

the always-valid p-value is ``p_n = \\min(p_{n-1}, 1/\\Lambda_n)`` and inverting
``\\Lambda_n < 1/\\alpha`` gives the always-valid interval

```math
\\hat\\theta_n \\pm \\sqrt{\\frac{V_n (V_n + \\tau^2)}{\\tau^2}
  \\left\\{2\\log(1/\\alpha) + \\log\\frac{V_n + \\tau^2}{V_n}\\right\\}},
```

intersected over time. Update the monitor with `fit!(m, y, d)` (`d = 1` treated,
`0` control) and read the current p-value and interval with [`snapshot`](@ref).

**Assumptions.** ``\\Lambda_n`` is an exact test martingale for Gaussian outcomes
with known variance arriving in treated/control pairs. With estimated variances,
unbalanced arrivals or non-Gaussian outcomes (including binary conversions) the
guarantee holds only approximately, through the central limit theorem, as in Johari et
al. (2022); the quality of the approximation depends on the sample size, the burn-in
and the outcome distribution (rare conversions are the hardest case), and no
finite-sample bound is available.
[`confseq_ate`](@ref) provides an asymptotic confidence sequence with a formal
time-uniform guarantee under i.i.d. arrivals, and [`MeanMonitor`](@ref) with a
bounded-data boundary gives exact guarantees for bounded outcomes.

**Tuning (`tau`, `n_opt`).** The mixing scale ``\\tau`` determines which effects the
test detects quickly: it is most powerful for effects of order ``\\tau``. When `tau`
is not given it is tuned to the planned total sample size ``n_{\\text{opt}}`` (both
arms): ``\\tau^2 = V_{\\text{opt}}\\{2\\log(1/\\alpha) + \\log(1 + 2\\log(1/\\alpha))\\}``
with ``V_{\\text{opt}} = 4\\sigma^2/n_{\\text{opt}}`` the variance of
``\\hat\\theta`` at ``n_{\\text{opt}}`` under equal allocation, which makes the
interval tightest at ``n_{\\text{opt}}`` (the tuning of the normal-mixture
confidence sequence of Howard et al. 2021). Here ``\\sigma^2`` is the known `sigma`
squared or the pooled variance of the first `burn_in` units per arm, fixed thereafter.
Both `tau` and `n_opt` must be set before the data are seen: a mixing scale chosen
after inspecting the data voids the guarantee.

# Keywords
- `outcome_type::Symbol = :continuous`: `:continuous` or `:binary` (0/1 conversions;
  the variance of each arm is ``\\tilde p(1-\\tilde p)`` with
  ``\\tilde p = (\\text{successes} + 1/2)/(n + 1)``).
- `null::Real = 0.0`: the null difference ``\\theta_0``.
- `alpha::Real = 0.05`: level of the always-valid interval and default level of
  [`stopping`](@ref).
- `tau = nothing`: standard deviation of the normal mixing distribution. One of
  `tau` and `n_opt` is required.
- `n_opt = nothing`: planned total sample size used to tune `tau` when `tau` is not
  given.
- `sigma = nothing`: known common outcome standard deviation (continuous outcomes
  only); otherwise the arm variances are estimated from the data seen so far.
- `burn_in::Integer = 10`: units per arm before the test starts (``\\Lambda = 1``
  before); must be at least 2.
- `record::Bool = true`: keep the path for [`sequential_test`](@ref).

# Returns
- `MSPRTMonitor`, a [`SequentialMonitor`](@ref).

# Examples
```julia
using DrSnow, StableRNGs
rng = StableRNG(14)
m = MSPRTMonitor(; outcome_type=:binary, n_opt=20_000)
for i in 1:20_000
    d = rand(rng) < 0.5
    y = rand(rng) < (d ? 0.12 : 0.10)
    fit!(m, y, d)
    snapshot(m).pvalue < 0.05 && break
end
t = sequential_test(m)
stopping(t)
```

# References
- Johari, R., Koomen, P., Pekelis, L., & Walsh, D. (2022). Always valid inference:
  Continuous monitoring of A/B tests. *Operations Research*, 70(3), 1806–1821.
- Robbins, H. (1970). Statistical methods related to the law of the iterated
  logarithm. *Annals of Mathematical Statistics*, 41(5), 1397–1409.
- Wald, A. (1945). Sequential tests of statistical hypotheses. *Annals of
  Mathematical Statistics*, 16(2), 117–186.
- Howard, S. R., Ramdas, A., McAuliffe, J., & Sekhon, J. (2021). Time-uniform,
  nonparametric, nonasymptotic confidence sequences. *Annals of Statistics*, 49(2),
  1055–1080.
"""
mutable struct MSPRTMonitor <: SequentialMonitor
    binary::Bool
    null::Float64
    alpha::Float64
    tau2::Float64               # NaN until tuned
    n_opt::Float64
    sigma::Float64              # NaN: estimate
    burn_in::Int
    n::NTuple{2,Int}            # (control, treated)
    mean::NTuple{2,Float64}
    m2::NTuple{2,Float64}
    cur_lower::Float64
    cur_upper::Float64
    pval::Float64
    loglr::Float64
    path::_SeqPath
end

function MSPRTMonitor(; outcome_type::Symbol=:continuous, null::Real=0.0,
                      alpha::Real=0.05, tau=nothing, n_opt=nothing, sigma=nothing,
                      burn_in::Integer=10, record::Bool=true)
    outcome_type in (:continuous, :binary) ||
        throw(ArgumentError("outcome_type must be :continuous or :binary"))
    (0 < alpha < 1) || throw(ArgumentError("alpha must be in (0, 1), got $alpha"))
    burn_in >= 2 || throw(ArgumentError("burn_in must be at least 2"))
    sig = NaN
    if sigma !== nothing
        outcome_type === :binary &&
            throw(ArgumentError("sigma is not used for binary outcomes"))
        (sigma > 0 && isfinite(sigma)) || throw(ArgumentError("sigma must be positive"))
        sig = float(sigma)
    end
    tau2 = NaN
    if tau !== nothing
        (tau > 0 && isfinite(tau)) || throw(ArgumentError("tau must be positive"))
        tau2 = float(tau)^2
    end
    nopt = n_opt === nothing ? NaN : float(n_opt)
    tau === nothing && n_opt === nothing &&
        throw(ArgumentError("give the mixing scale `tau` or the planned total sample " *
                            "size `n_opt` it should be tuned for"))
    isnan(nopt) || nopt > 0 || throw(ArgumentError("n_opt must be positive"))
    m = MSPRTMonitor(outcome_type === :binary, float(null), float(alpha), tau2, nopt,
                     sig, Int(burn_in), (0, 0), (0.0, 0.0), (0.0, 0.0), -Inf, Inf, 1.0,
                     0.0, _SeqPath(record))
    isnan(tau2) && !isnan(sig) && (m.tau2 = _seq_msprt_tau2(m, sig^2))
    return m
end

_seq_msprt_tau2(m::MSPRTMonitor, s2) =
    4 * s2 / m.n_opt * (2 * log(1 / m.alpha) + log(1 + 2 * log(1 / m.alpha)))

function _seq_arm_var(m::MSPRTMonitor, a::Int)
    n = m.n[a]
    if m.binary
        p = (m.mean[a] * n + 0.5) / (n + 1)
        return p * (1 - p)
    end
    isnan(m.sigma) || return m.sigma^2
    return n < 2 ? NaN : m.m2[a] / (n - 1)
end

_seq_msprt_estimate(m::MSPRTMonitor) =
    (m.n[1] == 0 || m.n[2] == 0) ? NaN : m.mean[2] - m.mean[1]

function StatsAPI.fit!(m::MSPRTMonitor, y::Real, d::Real)
    isfinite(y) || throw(ArgumentError("outcomes must be finite, got $y"))
    a = _seq_check_d(d) ? 2 : 1
    m.binary && !(y == 0 || y == 1) &&
        throw(ArgumentError("binary outcomes must be 0/1, got $y"))
    n = m.n[a] + 1
    dlt = y - m.mean[a]
    mu = m.mean[a] + dlt / n
    m2 = m.m2[a] + dlt * (y - mu)
    m.n = a == 1 ? (n, m.n[2]) : (m.n[1], n)
    m.mean = a == 1 ? (mu, m.mean[2]) : (m.mean[1], mu)
    m.m2 = a == 1 ? (m2, m.m2[2]) : (m.m2[1], m2)
    started = min(m.n...) >= m.burn_in
    if started && isnan(m.tau2)
        s2 = (_seq_arm_var(m, 1) * (m.n[1] - 1) + _seq_arm_var(m, 2) * (m.n[2] - 1)) /
             (m.n[1] + m.n[2] - 2)
        m.tau2 = _seq_msprt_tau2(m, s2)
    end
    est = _seq_msprt_estimate(m)
    loglr = 0.0
    lo, hi = -Inf, Inf
    V = NaN
    if started
        V = _seq_arm_var(m, 1) / m.n[1] + _seq_arm_var(m, 2) / m.n[2]
        if V > 0
            t2 = m.tau2
            loglr = 0.5 * log(V / (V + t2)) + t2 * (est - m.null)^2 / (2V * (V + t2))
            rad = sqrt(V * (V + t2) / t2 * (2 * log(1 / m.alpha) + log((V + t2) / V)))
            lo, hi = est - rad, est + rad
        end
    end
    m.loglr = loglr
    m.pval = min(m.pval, exp(-loglr))
    m.cur_lower = max(m.cur_lower, lo)
    m.cur_upper = min(m.cur_upper, hi)
    if m.path.record
        _seq_record!(m.path, sum(m.n), est, m.cur_lower, m.cur_upper,
                     isnan(V) ? NaN : sqrt(V), exp(loglr))
    end
    return m
end

function StatsAPI.fit!(m::MSPRTMonitor, y::AbstractVector{<:Real},
                       d::AbstractVector{<:Real})
    length(y) == length(d) || throw(DimensionMismatch("y and d must have equal length"))
    for i in eachindex(y, d)
        fit!(m, y[i], d[i])
    end
    return m
end

StatsAPI.nobs(m::MSPRTMonitor) = sum(m.n)

function snapshot(m::MSPRTMonitor)
    return (n=sum(m.n), estimate=_seq_msprt_estimate(m), lower=m.cur_lower,
            upper=m.cur_upper, evalue=exp(m.loglr), pvalue=m.pval,
            n_treated=m.n[2], n_control=m.n[1],
            tau=isnan(m.tau2) ? NaN : sqrt(m.tau2))
end

function sequential_test(m::MSPRTMonitor)
    m.path.record ||
        throw(ArgumentError("the monitor was created with record = false; no path is " *
                            "stored (use `snapshot` for the current state)"))
    p = m.path
    kind = m.binary ? "difference in proportions" : "difference in means"
    tau = isnan(m.tau2) ? "not yet tuned" : @sprintf("%.4g", sqrt(m.tau2))
    note = "mSPRT with normal mixing distribution (τ = $tau). Exact for Gaussian " *
           "outcomes with known variance and paired arrivals; otherwise valid " *
           "approximately (CLT)."
    return SequentialTest("Mixture SPRT, $kind", "$kind = $(m.null)",
                          "mSPRT (Johari et al. 2022)", copy(p.n), copy(p.estimate),
                          copy(p.evalue), _seq_running_pvalue(p.evalue), copy(p.lower),
                          copy(p.upper), m.alpha, note)
end

confidence_sequence(m::MSPRTMonitor) =
    throw(ArgumentError("an MSPRTMonitor is a test: use `sequential_test(m)`, whose " *
                        "`lower`/`upper` fields hold the always-valid interval"))
sequence_path(m::MSPRTMonitor) = sequence_path(sequential_test(m))

"""
    msprt_test(data, outcome, treatment; outcome_type=:continuous, order=nothing,
               null=0.0, alpha=0.05, tau=nothing, n_opt=nothing, sigma=nothing,
               burn_in=10) -> SequentialTest

Mixture sequential probability ratio test (mSPRT) of ``H_0: \\theta =`` `null` for
the difference in means (or conversion rates) between treated and control units,
computed over the units in arrival order.

This is the batch interface to [`MSPRTMonitor`](@ref), which defines the mixture
likelihood ratio, its tuning and its assumptions (exact for Gaussian outcomes with
known variance and paired arrivals, approximate otherwise; Johari et al. 2022). The
result holds the always-valid p-value and interval after every unit, so it shows when
a continuously monitored A/B test would first have rejected ``H_0`` at level
`alpha`, and the p-value after the last unit is valid whether or not the stopping
point was chosen by looking at the data.

**Default tuning to the realized sample size.** When neither `tau` nor `n_opt` is
given, the mixing scale is tuned to `n_opt = nrow(data)`, the realized number of
units. This is appropriate only when the number of units was fixed in advance
independently of the results; when the data end where monitoring stopped, a mixture
tuned to the realized size is chosen with knowledge of the stopping time and the
always-valid guarantee no longer strictly applies. Pass the pre-registered planned
sample size as `n_opt` (or a pre-specified `tau`), and report it.

For designs with a small number of planned interim analyses, a group-sequential test
([`gs_design`](@ref), [`gs_analysis`](@ref)) is more powerful than a continuously
monitored mSPRT; the mSPRT is preferable when the analysis schedule cannot be fixed.

# Arguments
- `data`: a table with one row per unit.
- `outcome::Symbol`: outcome column (0/1 for `outcome_type = :binary`).
- `treatment::Symbol`: 0/1 treatment column.

# Keywords
- `order = nothing`: arrival-time column; `nothing` means the row order is the
  arrival order.
- `n_opt = nothing`: planned total sample size for tuning `tau`; defaults to
  `nrow(data)` when `tau` is also `nothing` (see above).
- `outcome_type`, `null`, `alpha`, `tau`, `sigma`, `burn_in`: see
  [`MSPRTMonitor`](@ref).

# Returns
- [`SequentialTest`](@ref); `pvalue(t)` is the always-valid p-value after the last
  unit, [`stopping`](@ref) gives the first rejection and [`sequence_path`](@ref) the
  path.

# Examples
```julia
using DrSnow, DataFrames, StableRNGs
rng = StableRNG(15)
n = 10_000
treated = Int.(rand(rng, n) .< 0.5)
converted = Int.(rand(rng, n) .< ifelse.(treated .== 1, 0.115, 0.10))
df = DataFrame(converted=converted, treated=treated)
t = msprt_test(df, :converted, :treated; outcome_type=:binary, n_opt=10_000)
pvalue(t)                    # always-valid p-value after the last unit
stopping(t)                  # when the test would have stopped at α = 0.05
```

# References
- Johari, R., Koomen, P., Pekelis, L., & Walsh, D. (2022). Always valid inference:
  Continuous monitoring of A/B tests. *Operations Research*, 70(3), 1806–1821.
- Robbins, H. (1970). Statistical methods related to the law of the iterated
  logarithm. *Annals of Mathematical Statistics*, 41(5), 1397–1409.
- Wald, A. (1945). Sequential tests of statistical hypotheses. *Annals of
  Mathematical Statistics*, 16(2), 117–186.
"""
function msprt_test(data, outcome::Symbol, treatment::Symbol; order=nothing,
                    tau=nothing, n_opt=nothing, kwargs...)
    require_columns(data, [outcome, treatment])
    perm = _seq_arrival_order(data, order)
    y = _seq_numeric_column(data, outcome, "outcome")[perm]
    d = _seq_numeric_column(data, treatment, "treatment")[perm]
    all(v -> v == 0 || v == 1, d) ||
        throw(ArgumentError("treatment column :$treatment must be 0/1"))
    nopt = tau === nothing && n_opt === nothing ? length(y) : n_opt
    m = MSPRTMonitor(; tau=tau, n_opt=nopt, kwargs...)
    fit!(m, y, d)
    return sequential_test(m)
end
