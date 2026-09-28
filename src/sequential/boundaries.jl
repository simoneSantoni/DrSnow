# Time-uniform boundaries and the per-observation engines behind every confidence
# sequence of the area.
#
# Every confidence sequence (CS) here is the set of parameter values `m` whose
# e-process `E_t(m)` has not reached `1/α`:
#     C_t = {m : E_t(m) < 1/α}        (Ville's inequality ⇒ P(∃t: μ ∉ C_t) ≤ α),
# and the always-valid p-value for `H₀: μ = m₀` is `p_t = min(1, 1 / max_{s≤t} E_s(m₀))`.
# An engine consumes one observation at a time (`_seq_push!`), reports the current
# interval (`_seq_interval`) and the log e-value at the null (`_seq_logevalue`). The
# monitors in `monitors.jl` wrap engines; the batch functions feed monitors, so
# streaming and batch results are identical by construction.

const _SEQ_BOUNDARIES = (:asymptotic, :normal_mixture, :hoeffding,
                         :empirical_bernstein, :betting)

# ---------------------------------------------------------------------------
# Two-sided normal mixture (Robbins 1970; Howard et al. 2021, eq. (14))
# ---------------------------------------------------------------------------

"""
    _seq_nm_rho(v_opt, alpha)

Mixture precision `ρ` of the two-sided normal-mixture boundary that minimises the
boundary at intrinsic time `v_opt` (Howard et al. 2021, Prop. 3 / confseq
`TwoSidedNormalMixture::best_rho`).
"""
function _seq_nm_rho(v_opt::Real, alpha::Real)
    v_opt > 0 || throw(ArgumentError("t_opt must be positive, got $v_opt"))
    l = log(1 / alpha)
    return v_opt / (2l + log(1 + 2l))
end

"""
    _seq_nm_bound(v, alpha, rho)

Two-sided normal-mixture uniform boundary `u(v) = √((v+ρ)(log(1+v/ρ) + 2 log(1/α)))`:
for a σ²=1 sub-Gaussian sum `S_t` with intrinsic time `v = t`,
`P(∃t: |S_t| ≥ u(t)) ≤ α`.
"""
_seq_nm_bound(v::Real, alpha::Real, rho::Real) =
    sqrt((v + rho) * (log1p(v / rho) + 2 * log(1 / alpha)))

"""Log of the two-sided normal-mixture martingale at sum `s` and intrinsic time `v`."""
_seq_nm_logmart(s::Real, v::Real, rho::Real) =
    0.5 * log(rho / (v + rho)) + s^2 / (2 * (v + rho))

# ---------------------------------------------------------------------------
# Engine: Gaussian (asymptotic CS with estimated σ, or sub-Gaussian with known σ)
# ---------------------------------------------------------------------------

abstract type _SeqEngine end

mutable struct _SeqGaussEngine <: _SeqEngine
    sigma::Float64          # known σ (normal mixture) or NaN (estimate: asymptotic CS)
    rho::Float64            # mixture parameter on the unit-variance time scale
    alpha::Float64
    null::Float64
    min_n::Int              # estimated σ: monitoring starts at n = min_n
    n::Int
    mean::Float64
    m2::Float64
end

_SeqGaussEngine(sigma, rho, alpha, null; min_n::Integer=2) =
    _SeqGaussEngine(sigma, rho, alpha, null, max(Int(min_n), 2), 0, 0.0, 0.0)

function _seq_push!(e::_SeqGaussEngine, x::Real)
    e.n += 1
    d = x - e.mean
    e.mean += d / e.n
    e.m2 += d * (x - e.mean)
    return e
end

# Scale used by the boundary: the known σ or the running (1/t) standard deviation.
function _seq_scale(e::_SeqGaussEngine)
    isnan(e.sigma) || return e.sigma
    return e.n < 2 ? NaN : sqrt(max(e.m2, 0.0) / e.n)
end

# With an estimated σ the boundary is used from n = min_n on (asymptotic CS: the
# guarantee is for a late start; σ̂ from a handful of points is unreliable).
_seq_started(e::_SeqGaussEngine) = !isnan(e.sigma) || e.n >= e.min_n

function _seq_interval(e::_SeqGaussEngine)
    (e.n == 0 || !_seq_started(e)) && return (-Inf, Inf)
    s = _seq_scale(e)
    isnan(s) && return (-Inf, Inf)
    r = s * _seq_nm_bound(e.n, e.alpha, e.rho) / e.n
    return (e.mean - r, e.mean + r)
end

function _seq_logevalue(e::_SeqGaussEngine)
    s = _seq_scale(e)
    (e.n == 0 || isnan(s) || !_seq_started(e)) && return 0.0
    dev = e.n * (e.mean - e.null)
    if s == 0
        return dev == 0 ? 0.0 : Inf
    end
    return _seq_nm_logmart(dev / s, e.n, e.rho)
end

_seq_estimate(e::_SeqGaussEngine) = e.n == 0 ? NaN : e.mean

# ---------------------------------------------------------------------------
# Predictable-mixture Hoeffding and empirical-Bernstein CS (Waudby-Smith & Ramdas
# 2024, Thm 2 and Thm 3), for observations in [0, 1]. One `_SeqPredmixSide` is the
# lower CS of one stream (x for the lower bound, 1 - x for the upper bound); the
# arithmetic mirrors confseq's `predmix_lower_cs` step by step.
# ---------------------------------------------------------------------------

mutable struct _SeqPredmixSide
    eb::Bool                # empirical Bernstein (true) or Hoeffding (false)
    alpha::Float64          # one-sided level (α/2 of the two-sided CS)
    truncation::Float64
    t_opt::Float64          # NaN: the untuned λ_t ∝ 1/√(t log(1+t)) schedule
    t::Int
    s::Float64              # Σ y
    sig_acc::Float64        # Σ (y_i - μ̃_i)², μ̃ the regularised running mean
    sum_lambda::Float64
    sum_lambda_y::Float64
    sum_vpsi::Float64
    lower_max::Float64      # running maximum of the lower bound
end

_SeqPredmixSide(eb, alpha, truncation, t_opt) =
    _SeqPredmixSide(eb, alpha, truncation, t_opt, 0, 0.0, 0.0, 0.0, 0.0, 0.0, -Inf)

# λ_t from data up to t-1 (confseq `lambda_predmix_eb` with prior mean 1/2,
# prior variance 1/4 and one fake observation).
function _seq_predmix_eb_lambda(t::Int, sig_acc_prev::Float64, alpha::Float64,
                                t_opt::Float64, truncation::Float64)
    sigma2 = (0.25 + sig_acc_prev) / t          # (1·¼ + Σ_{i<t}) / ((t-1) + 1)
    lam = isnan(t_opt) ? sqrt(2 * log(1 / alpha) / (t * log(1 + t) * sigma2)) :
          sqrt(2 * log(1 / alpha) / (t_opt * sigma2))
    isnan(lam) && (lam = 0.0)
    return min(truncation, lam)
end

function _seq_push!(p::_SeqPredmixSide, y::Real)
    t = p.t + 1
    lam = if p.eb
        _seq_predmix_eb_lambda(t, p.sig_acc, p.alpha, p.t_opt, p.truncation)
    elseif isnan(p.t_opt)
        min(sqrt(8 * log(1 / p.alpha) / (t * log(1 + t))), p.truncation)
    else
        sqrt(8 * log(1 / p.alpha) / p.t_opt)
    end
    if p.eb
        mu_prev = p.t == 0 ? 0.0 : p.s / p.t   # unregularised mean, 0 at t = 1
        v = (y - mu_prev)^2
        psi = -log1p(-lam) - lam
    else
        v = 1.0
        psi = lam^2 / 8
    end
    p.t = t
    p.s += y
    mu_reg = min((0.5 + p.s) / (t + 1), 1.0)
    p.sig_acc += (y - mu_reg)^2
    p.sum_lambda += lam
    p.sum_lambda_y += lam * y
    p.sum_vpsi += v * psi
    return p
end

function _seq_lower(p::_SeqPredmixSide)
    p.sum_lambda > 0 || return 0.0
    l = (p.sum_lambda_y - log(1 / p.alpha) - p.sum_vpsi) / p.sum_lambda
    return max(l, 0.0)
end

# log E⁺_t(m) = Σ λ_i (y_i - m) - Σ v_i ψ(λ_i)
_seq_logmart(p::_SeqPredmixSide, m::Real) = p.sum_lambda_y - m * p.sum_lambda - p.sum_vpsi

mutable struct _SeqPredmixEngine <: _SeqEngine
    lo::_SeqPredmixSide     # lower CS from x
    hi::_SeqPredmixSide     # lower CS from 1 - x (gives the upper bound)
    null::Float64           # on the [0, 1] scale
    n::Int
    mean::Float64
end

function _SeqPredmixEngine(eb::Bool, alpha, t_opt, null; truncation=nothing)
    tr = truncation === nothing ? (eb ? 0.5 : 1.0) : float(truncation)
    return _SeqPredmixEngine(_SeqPredmixSide(eb, alpha / 2, tr, t_opt),
                             _SeqPredmixSide(eb, alpha / 2, tr, t_opt), null, 0, 0.0)
end

function _seq_push!(e::_SeqPredmixEngine, x::Real)
    _seq_push!(e.lo, x)
    _seq_push!(e.hi, 1 - x)
    e.n += 1
    e.mean += (x - e.mean) / e.n
    return e
end

_seq_interval(e::_SeqPredmixEngine) =
    e.n == 0 ? (0.0, 1.0) : (_seq_lower(e.lo), 1 - _seq_lower(e.hi))

# The CS excludes m iff E⁺(m) ≥ 2/α or E⁻(m) ≥ 2/α, i.e. iff max(E⁺, E⁻)/2 ≥ 1/α; this
# e-process is dominated by the test supermartingale (E⁺ + E⁻)/2.
_seq_logevalue(e::_SeqPredmixEngine) =
    e.n == 0 ? 0.0 : max(_seq_logmart(e.lo, e.null), _seq_logmart(e.hi, 1 - e.null)) -
                     log(2)

_seq_estimate(e::_SeqPredmixEngine) = e.n == 0 ? NaN : e.mean

# ---------------------------------------------------------------------------
# Betting (hedged capital) CS (Waudby-Smith & Ramdas 2024, Thm 3 / Sec. 4), for
# observations in [0, 1], computed on the grid {0, 1/B, ..., 1} exactly as confseq's
# `hedged_cs` (θ = 1/2, truncation 1/2, max of the two capital processes).
# ---------------------------------------------------------------------------

mutable struct _SeqBettingEngine <: _SeqEngine
    alpha::Float64
    t_opt::Float64
    grid::Vector{Float64}
    logk_pos::Vector{Float64}
    logk_neg::Vector{Float64}
    null::Float64
    logk_null::NTuple{2,Float64}
    t::Int
    s::Float64
    sig_acc::Float64
    mean::Float64
end

function _SeqBettingEngine(alpha, t_opt, null, breaks::Integer)
    breaks >= 2 || throw(ArgumentError("breaks must be at least 2, got $breaks"))
    grid = collect(0:breaks) ./ breaks
    return _SeqBettingEngine(alpha, t_opt, grid, zeros(length(grid)),
                             zeros(length(grid)), null, (0.0, 0.0), 0, 0.0, 0.0, 0.0)
end

const _SEQ_BET_TRUNC = 0.5

@inline function _seq_bet_factors(lam::Float64, x::Float64, m::Float64)
    lp = max(min(lam, _SEQ_BET_TRUNC / m), -_SEQ_BET_TRUNC / (1 - m))
    ln = max(min(lam, _SEQ_BET_TRUNC / (1 - m)), -_SEQ_BET_TRUNC / m)
    return log1p(lp * (x - m)), log1p(-ln * (x - m))
end

function _seq_push!(e::_SeqBettingEngine, x::Real)
    t = e.t + 1
    # Bets use α·θ = α/2 and no truncation beyond the m-dependent one.
    lam = _seq_predmix_eb_lambda(t, e.sig_acc, e.alpha / 2, e.t_opt, Inf)
    xf = float(x)
    @inbounds for j in eachindex(e.grid)
        fp, fn = _seq_bet_factors(lam, xf, e.grid[j])
        e.logk_pos[j] += fp
        e.logk_neg[j] += fn
    end
    fp, fn = _seq_bet_factors(lam, xf, e.null)
    e.logk_null = (e.logk_null[1] + fp, e.logk_null[2] + fn)
    e.t = t
    e.s += xf
    mu_reg = min((0.5 + e.s) / (t + 1), 1.0)
    e.sig_acc += (xf - mu_reg)^2
    e.mean += (xf - e.mean) / t
    return e
end

function _seq_interval(e::_SeqBettingEngine)
    e.t == 0 && return (0.0, 1.0)
    thr = log(1 / e.alpha)
    lh = log(0.5)
    first = 0
    last = 0
    @inbounds for j in eachindex(e.grid)
        if max(lh + e.logk_pos[j], lh + e.logk_neg[j]) <= thr
            first == 0 && (first = j)
            last = j
        end
    end
    step = e.grid[2] - e.grid[1]
    l, u = first == 0 ? (0.0, 1.0) : (e.grid[first], e.grid[last])
    return (max(0.0, l - step), min(1.0, u + step))
end

_seq_logevalue(e::_SeqBettingEngine) =
    e.t == 0 ? 0.0 : log(0.5) + max(e.logk_null[1], e.logk_null[2])

_seq_estimate(e::_SeqBettingEngine) = e.t == 0 ? NaN : e.mean
