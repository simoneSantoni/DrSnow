# Power for regression discontinuity designs (Cattaneo, Titiunik & Vazquez-Bare 2019,
# the `rdpower` package): the variance of the local polynomial estimator in a new
# sample of size ñ with bandwidths (h₋, h₊) is V₋/(ñ h₋^(1+2ν)) + V₊/(ñ h₊^(1+2ν)),
# with the per-side constants V estimated from pilot data by `rd_estimate`.

function _des_rd_power(tau, se, alpha, alternative)
    return _des_power_ncp(tau / se, Inf, alpha, alternative, :normal)
end

# Assemble the result shared by the data and the parameter interfaces.
function _des_rd_result(effect, power, n, Vrb::NTuple{2,Float64}, Vcl::NTuple{2,Float64},
                        bias::NTuple{2,Float64}, h::NTuple{2,Float64}, p::Int, deriv::Int,
                        alpha, alternative, extra::NamedTuple)
    _des_check_alpha(alpha, alternative, :normal)
    all(>(0), Vrb) && all(>(0), Vcl) ||
        throw(ArgumentError("power_rd: variances must be positive"))
    all(>(0), h) || throw(ArgumentError("power_rd: bandwidths must be positive"))
    _des_pos(n, "n")
    e = 1 + 2 * deriv
    vrb = Vrb[1] / h[1]^e + Vrb[2] / h[2]^e
    vcl = Vcl[1] / h[1]^e + Vcl[2] / h[2]^e
    bsum = bias[2] * h[2]^(1 + p - deriv) + bias[1] * h[1]^(1 + p - deriv)
    sefun = q -> (sqrt(vrb / q.n), Inf)
    ranges = Dict{Symbol,Tuple}(:n => (1e-8, Inf, true))
    des = "regression discontinuity (local polynomial p = $p, robust bias-corrected " *
          "test)"
    r = _des_linear(des, (effect=effect, n=n), power, :effect, sefun, ranges, alpha,
                    alternative, :normal;
                    note="the new sample is assumed to have the pilot's distribution " *
                         "of the running variable; `n` is the total new sample size")
    ñ = r.parameters.n
    se_cl = sqrt(vcl / ñ)
    pw_cl = _des_power_ncp((r.effect + bsum) / se_cl, Inf, alpha, alternative, :normal)
    params = merge(r.parameters, (power_conventional=pw_cl, se_conventional=se_cl,
                                  bias=bsum, h_left=h[1], h_right=h[2],
                                  V_rb_sum=Vrb[1] + Vrb[2]), extra)
    return PowerAnalysis(r.design, r.solved, r.power, r.effect, params, r.se, Inf,
                         r.alpha, r.alternative, :normal, r.mde_multiplier, r.note)
end

_des_pair(x, name) = x isa Real ? (float(x), float(x)) :
                     (length(x) == 2 ? (float(x[1]), float(x[2])) :
                      throw(ArgumentError("$name must be a number or a pair")))

"""
    power_rd(data, outcome, running; effect=nothing, power=nothing, n=nothing,
             sampsi=nothing, samph=nothing, cutoff=0.0, alpha=0.05,
             alternative=:two_sided, rd_kwargs...) -> PowerAnalysis
    power_rd(; effect=nothing, power=nothing, n=nothing, variance, bias=(0.0, 0.0),
             samph, variance_conventional=variance, p=1, deriv=0, alpha=0.05,
             alternative=:two_sided) -> PowerAnalysis

Power, minimum detectable effect or sample size for a sharp regression discontinuity
design, following the `rdpower` approach of Cattaneo, Titiunik & Vazquez-Bare (2019).

In a sharp RD design treatment switches on when a running variable crosses a cutoff,
and the estimand is the jump ``\\tau = E[Y(1) - Y(0) \\mid X = c]`` (or a jump in the
``\\nu``-th derivative for kink designs) at the cutoff ``c``, identified under
continuity of the conditional expectations of the potential outcomes at ``c``. It is
estimated by local polynomial regression of order ``p`` on each side within
bandwidths ``(h_-, h_+)``, with inference based on the robust bias-corrected statistic
of Calonico, Cattaneo & Titiunik (2014). Because only observations near the cutoff
contribute, power depends on the density of the running variable there, the
bandwidths and the conditional variance of the outcome, which are best learned from
pilot data.

In a new sample of ``n`` units drawn from the same distribution as the pilot, the
variance of the estimator is approximated by

```math
\\frac{V_-}{n\\, h_-^{1 + 2\\nu}} + \\frac{V_+}{n\\, h_+^{1 + 2\\nu}},
```

with per-side constants ``V_\\pm``, and power is that of the two-sided normal test on
the robust bias-corrected statistic,

```math
1 - \\Phi(z_{1-\\alpha/2} - \\tau/\\text{se}) + \\Phi(-z_{1-\\alpha/2} - \\tau/\\text{se}).
```

The power of the conventional test, which is shifted by its leading bias, is reported
in `parameters.power_conventional`.

With pilot `data`, the constants are backed out from [`rd_estimate`](@ref) on the pilot
(``V = N h^{1+2\\nu} \\times \\text{variance}``, ``N`` the pilot size); the new
bandwidths default to the pilot bandwidths and `n` to the pilot size. As in `rdpower`,
`sampsi = (ñ₋, ñ₊)` specifies the new sample by the number of observations within the
bandwidth on each side instead of by `n`. `rd_estimate` does not store per-side
variances, so with data the pilot and new bandwidths must be symmetric; use the
parameter method (`variance = (V₋, V₊)`) for asymmetric designs. The calculation
treats the pilot's variance estimates and bandwidths as known, so with a small pilot
the power is itself noisy; report it over a range of plausible variances.

# Arguments
- `data::AbstractDataFrame`: pilot data for a sharp design (data method only).
- `outcome::Symbol`: outcome column of the pilot.
- `running::Symbol`: running-variable column of the pilot.

# Keywords
- `effect`: jump at the cutoff in outcome units; the MDE when solved for.
- `power`: target power, in `(alpha, 1)`.
- `n`: total size of the new sample (continuous when solved for). Leave exactly one of
  `effect`, `power` and `n` as `nothing`; with data, `n` defaults to the pilot size
  unless both `effect` and `power` are given.
- `sampsi = nothing`: new sample sizes within the bandwidth, `(left, right)` or one
  number, as an alternative to `n` (data method).
- `samph`: new bandwidths `(left, right)` or one number; defaults to the pilot
  bandwidths with data, required without.
- `cutoff::Real = 0.0`: the RD cutoff (data method).
- `rd_kwargs...`: further keywords passed to [`rd_estimate`](@ref) on the pilot, e.g.
  the polynomial order, kernel, bandwidth selector, covariates or `cluster`.
- `variance`: `(V₋, V₊)` for the robust bias-corrected estimator (parameter method).
- `variance_conventional = variance`: the same constants for the conventional
  estimator.
- `bias = (0.0, 0.0)`: leading-bias constants `(B₋, B₊)` of the conventional
  estimator.
- `p::Integer = 1`, `deriv::Integer = 0`: polynomial order and derivative ``\\nu``
  (parameter method).
- `alpha::Real = 0.05`, `alternative::Symbol = :two_sided`: level and direction of the
  test.

# Returns
- [`PowerAnalysis`](@ref) with a normal reference; `parameters` also holds
  `power_conventional`, `se_conventional`, `bias`, the bandwidths and, with data, the
  pilot sample sizes.

# Examples
```julia
using DrSnow, DataFrames, StableRNGs
rng = StableRNG(1)
x = 2 .* rand(rng, 1000) .- 1
pilot = DataFrame(x=x, y=0.5 .* x .+ 0.2 .* (x .>= 0) .+ 0.3 .* randn(rng, 1000))
power_rd(pilot, :y, :x; effect=0.2)                        # power at the pilot size
power_rd(pilot, :y, :x; effect=0.2, power=0.8)             # sample size needed
power_rd(variance=(40.0, 35.0), samph=0.3, n=2000, power=0.8)   # MDE (parameters)
```

# References
- Calonico, S., Cattaneo, M. D., & Titiunik, R. (2014). Robust nonparametric
  confidence intervals for regression-discontinuity designs. *Econometrica*, 82(6),
  2295–2326.
- Cattaneo, M. D., Titiunik, R., & Vazquez-Bare, G. (2019). Power calculations for
  regression-discontinuity designs. *Stata Journal*, 19(1), 210–245.
"""
function power_rd(; effect=nothing, power=nothing, n=nothing, variance, bias=(0.0, 0.0),
                  samph, variance_conventional=variance, p::Integer=1,
                  deriv::Integer=0, alpha::Real=0.05, alternative::Symbol=:two_sided)
    return _des_rd_result(effect, power, n, _des_pair(variance, "variance"),
                          _des_pair(variance_conventional, "variance_conventional"),
                          _des_pair(bias, "bias"), _des_pair(samph, "samph"), Int(p),
                          Int(deriv), alpha, alternative, NamedTuple())
end

function power_rd(data::AbstractDataFrame, outcome::Symbol, running::Symbol;
                  effect=nothing, power=nothing, n=nothing, sampsi=nothing,
                  samph=nothing, cutoff::Real=0.0, alpha::Real=0.05,
                  alternative::Symbol=:two_sided, rd_kwargs...)
    ctx = "power_rd"
    haskey(rd_kwargs, :treatment) && rd_kwargs[:treatment] !== nothing &&
        throw(ArgumentError("$ctx: only sharp designs are supported"))
    require_columns(data, [outcome, running]; context=ctx)
    est = rd_estimate(data, outcome, running; cutoff=cutoff, rd_kwargs...)
    est.design in (:sharp, :sharp_kink, :sharp_deriv) ||
        throw(ArgumentError("$ctx: only sharp designs are supported"))
    est.h_left ≈ est.h_right ||
        throw(ArgumentError("$ctx: the pilot bandwidths differ on the two sides; " *
                            "per-side variances are needed (use the `variance` method)"))
    d = est.deriv
    e = 1 + 2 * d
    cl = get(rd_kwargs, :cluster, nothing)
    x = data[!, running]
    ok = .!ismissing.(data[!, outcome]) .& .!ismissing.(x)
    count_units(mask) = cl === nothing ? count(mask) :
                        length(unique(data[mask, cl]))
    right = ok .& (x .>= cutoff)
    left = ok .& (x .< cutoff)
    N = count_units(ok)
    nplus, nminus = count_units(right), count_units(left)
    hnew = samph === nothing ? (est.h_left, est.h_right) : _des_pair(samph, "samph")
    hnew[1] ≈ hnew[2] ||
        throw(ArgumentError("$ctx: asymmetric `samph` requires per-side variances " *
                            "(use the `variance` method)"))
    h = est.h_left
    # V₋ + V₊ = N h^(1+2ν) Var; split evenly (only the sum matters when h₋ = h₊)
    Vrb = N * h^e * est.se_robust^2 / 2
    Vcl = N * h^e * est.se_conventional^2 / 2
    nh_r = count_units(right .& (x .<= cutoff + hnew[2]))
    nh_l = count_units(left .& (x .>= cutoff - hnew[1]))
    if sampsi !== nothing
        n === nothing || throw(ArgumentError("$ctx: give either `n` or `sampsi`"))
        sl, sr = _des_pair(sampsi, "sampsi")
        n = nplus * (sr / nh_r) + nminus * (sl / nh_l)
    elseif n === nothing && !(effect !== nothing && power !== nothing)
        n = float(N)
    end
    extra = (n_pilot=N, n_pilot_left=nminus, n_pilot_right=nplus,
             n_pilot_h_left=nh_l, n_pilot_h_right=nh_r, pilot_h=h)
    return _des_rd_result(effect, power, n, (Vrb, Vrb), (Vcl, Vcl),
                          (est.bias_left / h^(1 + est.p - d),
                           est.bias_right / h^(1 + est.p - d)), hnew, est.p, d,
                          alpha, alternative, extra)
end
