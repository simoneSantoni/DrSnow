# Sensitivity to violations of the exclusion restriction: Conley, Hansen & Rossi
# (2012) "plausibly exogenous" union-of-confidence-intervals (UCI) and
# local-to-zero (LTZ) Gaussian-prior methods.
#
# Model: y = D β + Z γ + W δ + u. The exclusion restriction is γ = 0; γ is the direct
# effect of the instruments on the outcome in units of y per unit of Z. Writing
# A = (D̂'D̂)⁻¹ D̂'Z (D̂ = P_Z D, all partialled), 2SLS of y on D converges to β + Aγ,
# so for a given γ the bias-corrected estimator is β̂(γ) = β̂ − Aγ (2SLS of y − Zγ).

"""
    PlausiblyExogenousResult

Result of [`plausibly_exogenous`](@ref): inference on the coefficient of the
endogenous regressor that allows for a specified direct effect of the instruments on
the outcome.

The object reports a confidence interval that remains valid when the exclusion
restriction is relaxed from ``\\gamma = 0`` to a set of values (union of confidence
intervals) or a prior distribution (local-to-zero) for the direct effect ``\\gamma``
(Conley, Hansen and Rossi 2012), together with the matrix ``A`` that maps a direct
effect into the asymptotic bias of 2SLS.

# Fields
- `method::Symbol`: `:uci` (union of confidence intervals) or `:ltz`
  (local-to-zero).
- `level::Float64`: confidence level.
- `lower::Float64`, `upper::Float64`: confidence interval for the coefficient of the
  endogenous regressor that allows for the specified direct effect.
- `estimate::Float64`: for LTZ the prior-mean-corrected point estimate
  ``\\hat\\beta - A\\mu``; for UCI the unadjusted 2SLS estimate.
- `se::Float64`: LTZ standard error (`NaN` for UCI, which has no single standard
  error).
- `estimate_range::Tuple{Float64,Float64}`: for UCI the range of the bias-corrected
  estimates ``\\hat\\beta(\\gamma)`` over the support of ``\\gamma``; for LTZ
  `(estimate, estimate)`.
- `gamma`: the support of ``\\gamma`` (UCI: a vector of `(lo, hi)` per instrument) or
  the prior `(mean, vcov)` (LTZ).
- `adjustment::Matrix{Float64}`: ``A = (\\hat D'\\hat D)^{-1}\\hat D'Z`` (``1 \\times
  k``), mapping ``\\gamma`` to the asymptotic bias ``A\\gamma`` of 2SLS; with one
  instrument ``A = 1/\\hat\\pi``, the inverse of the first-stage coefficient.
- `instruments::Vector{Symbol}`, `outcome::Symbol`, `endogenous::Symbol`: the
  specification.

# References
- Conley, T. G., Hansen, C. B., & Rossi, P. E. (2012). Plausibly exogenous. *Review
  of Economics and Statistics*, 94(1), 260–272.
"""
struct PlausiblyExogenousResult
    method::Symbol
    level::Float64
    lower::Float64
    upper::Float64
    estimate::Float64
    se::Float64
    estimate_range::Tuple{Float64,Float64}
    gamma::Any
    adjustment::Matrix{Float64}
    instruments::Vector{Symbol}
    outcome::Symbol
    endogenous::Symbol
end

function Base.show(io::IO, ::MIME"text/plain", r::PlausiblyExogenousResult)
    lv = @sprintf("%g%%", 100 * r.level)
    if r.method === :uci
        println(io, "Plausibly exogenous IV (Conley, Hansen & Rossi 2012): union of ",
                "confidence intervals")
        for (j, s) in enumerate(r.instruments)
            @printf(io, "  direct effect of %s on %s: γ ∈ [%.4g, %.4g]\n", s, r.outcome,
                    r.gamma[j]...)
        end
        @printf(io, "  bias-corrected estimates range over [%.4g, %.4g]\n",
                r.estimate_range...)
    else
        println(io, "Plausibly exogenous IV (Conley, Hansen & Rossi 2012): local-to-zero ",
                "prior γ ~ N(μ, Ω)")
        @printf(io, "  estimate %.4g (se %.4g)\n", r.estimate, r.se)
    end
    @printf(io, "  %s interval for the %s coefficient: [%.4g, %.4g]\n", lv, r.endogenous,
            r.lower, r.upper)
end

"""
    plausibly_exogenous(r::IVEstimate; method=:uci, gamma=nothing,
                        gamma_mean=nothing, gamma_vcov=nothing, level=r.level)
        -> PlausiblyExogenousResult

Sensitivity of IV inference to violations of the exclusion restriction: the
"plausibly exogenous" confidence intervals of Conley, Hansen and Rossi (2012).

The exclusion restriction is rarely beyond doubt, and it cannot be tested with a
just-identified model. Conley, Hansen and Rossi (2012) replace it by the structural
model

```math
Y = D\\beta + Z'\\gamma + W'\\delta + u, \\qquad E[Zu] = 0,
```

in which ``\\gamma`` is a direct effect of the instruments on the outcome, measured in
outcome units per unit of the instrument (the units of a reduced-form coefficient).
Exact exclusion is ``\\gamma = 0``; the researcher states how large a violation is
plausible. For a given ``\\gamma``, 2SLS of ``Y - Z'\\gamma`` on ``D`` is consistent for
``\\beta``; equivalently, 2SLS of ``Y`` converges to ``\\beta + A\\gamma`` with
``A = (\\hat D'\\hat D)^{-1}\\hat D'Z`` (all variables partialled on the controls). With
one instrument ``A = 1/\\pi``, where ``\\pi`` is the first-stage coefficient, so the
bias is ``\\gamma/\\pi``: a positive direct effect biases 2SLS upward when the first
stage is positive and downward when it is negative, and the bias is large when the
first stage is weak. The sign conventions of `gamma` therefore refer to the
instrument's coding, not to the treatment.

Two procedures are available. The union of confidence intervals (`method = :uci`)
takes a support for ``\\gamma``, a box with one `(lo, hi)` interval per instrument,
and returns the union over the support of the 2SLS confidence intervals for ``\\beta``
computed from ``Y - Z'\\gamma``, each with the model's covariance estimator. The
point estimate is affine in ``\\gamma`` and the standard error is the norm of an
affine function of ``\\gamma``, hence convex, so the upper endpoint is convex and
the lower endpoint concave in ``\\gamma``; the union is therefore attained at the
vertices of the box and is computed exactly. It has coverage of at least `level` for
every ``\\gamma`` in the support and is conservative. The local-to-zero method
(`method = :ltz`) treats ``\\gamma`` as random with prior
``\\gamma \\sim N(\\mu, \\Omega)`` and returns ``\\hat\\beta - A\\mu \\pm
c\\,\\sqrt{\\hat V + A\\Omega A'}``, with ``\\hat V`` the model's 2SLS variance and ``c``
the critical value (``t(G-1)`` under clustering). A prior can be informed by a
zero-first-stage subsample ([`zero_first_stage_test`](@ref); van Kippersluis and
Rietveld 2018).

Report the results as a sensitivity analysis, for example the largest direct effect
at which the interval still excludes zero. The intervals rely on conventional 2SLS
asymptotics and inherit the problems of 2SLS intervals when the instruments are weak;
with heterogeneous effects ``\\beta`` is the LATE-type estimand of the model (see
[`late_2sls`](@ref)).

# Arguments
- `r::IVEstimate`: a fitted IV model with one endogenous regressor.

# Keywords
- `method::Symbol`: `:uci` (default) or `:ltz`.
- `gamma`: for UCI (required there), the support of ``\\gamma``: a `(lo, hi)` tuple
  with one instrument, or a vector of such tuples, one per instrument (default
  `nothing`).
- `gamma_mean`, `gamma_vcov`: for LTZ (required there), the prior mean (scalar or
  vector) and variance (scalar or positive semi-definite matrix) of ``\\gamma``
  (default `nothing`).
- `level::Real`: confidence level (default `r.level`).

# Returns
- A [`PlausiblyExogenousResult`](@ref); `(res.lower, res.upper)` is the interval,
  `res.adjustment` the bias map ``A``.

# Examples
```julia
using DrSnow, DataFrames, StableRNGs
rng = StableRNG(11)
n = 2_000
z = rand(rng, 0:1, n)
v = randn(rng, n)
d = Int.(0.4 .* z .+ 0.3 .* v .+ rand(rng, n) .> 0.8)
y = 1.0 .* d .+ 0.05 .* z .+ v .+ randn(rng, n)      # small direct effect of Z
df = DataFrame(y=y, d=d, z=z)
r = late_2sls(df, :y, :d, :z)
plausibly_exogenous(r; method=:uci, gamma=(0.0, 0.1))
plausibly_exogenous(r; method=:ltz, gamma_mean=0.05, gamma_vcov=0.025^2)
```

# References
- Conley, T. G., Hansen, C. B., & Rossi, P. E. (2012). Plausibly exogenous. *Review
  of Economics and Statistics*, 94(1), 260–272.
- van Kippersluis, H., & Rietveld, C. A. (2018). Beyond plausibly exogenous. *The
  Econometrics Journal*, 21(3), 316–331.
- Imbens, G. W. (2014). Instrumental variables: An econometrician's perspective.
  *Statistical Science*, 29(3), 323–358.
"""
function plausibly_exogenous(r::IVEstimate; method::Symbol=:uci, gamma=nothing,
                             gamma_mean=nothing, gamma_vcov=nothing,
                             level::Real=r.level)
    des = r.design
    size(des.D, 2) == 1 || throw(ArgumentError("plausibly_exogenous requires one " *
                                               "endogenous regressor"))
    0 < level < 1 || throw(ArgumentError("level must be in (0, 1)"))
    k = size(des.Z, 2)
    y, D, Z = des.y, des.D, des.Z
    Dhat = Z * (Z \ D)
    A = (Dhat' * Dhat) \ (Dhat' * Z)             # 1 × k
    crit = critical_value(level, r.dof_residual)
    b = r.coef[1]
    if method === :uci
        gamma === nothing && throw(ArgumentError("method = :uci requires `gamma` " *
                                                 "(support of the direct effect)"))
        box = gamma isa Tuple ? [gamma] : collect(gamma)
        length(box) == k || throw(ArgumentError("`gamma` needs one (lo, hi) per " *
                                                "instrument ($k)"))
        box = [(float(lo), float(hi)) for (lo, hi) in box]
        all(lo <= hi for (lo, hi) in box) ||
            throw(ArgumentError("each gamma support must satisfy lo ≤ hi"))
        lo_ci, hi_ci = Inf, -Inf
        lo_b, hi_b = Inf, -Inf
        for vertex in Iterators.product(box...)
            γ = collect(vertex)
            β, V, _, _ = _iv_tsls(des, y - Z * γ, D, Z)
            se = sqrt(V[1, 1])
            lo_ci = min(lo_ci, β[1] - crit * se)
            hi_ci = max(hi_ci, β[1] + crit * se)
            lo_b = min(lo_b, β[1])
            hi_b = max(hi_b, β[1])
        end
        return PlausiblyExogenousResult(:uci, float(level), lo_ci, hi_ci, b, NaN,
                                        (lo_b, hi_b), box, Matrix(A), r.instruments,
                                        r.outcome, r.endogenous[1])
    elseif method === :ltz
        (gamma_mean === nothing || gamma_vcov === nothing) &&
            throw(ArgumentError("method = :ltz requires `gamma_mean` and `gamma_vcov`"))
        μ = gamma_mean isa Real ? fill(float(gamma_mean), 1) : float.(collect(gamma_mean))
        Ω = gamma_vcov isa Real ? fill(float(gamma_vcov), 1, 1) :
            Matrix{Float64}(gamma_vcov)
        (length(μ) == k && size(Ω) == (k, k)) ||
            throw(ArgumentError("prior dimensions must match the $k instrument(s)"))
        all(>=(-1e-12 * max(1.0, maximum(abs, Ω))), eigvals(Symmetric(Ω))) ||
            throw(ArgumentError("gamma_vcov must be positive semi-definite"))
        est = b - dot(vec(A), μ)
        se = sqrt(r.vcov[1, 1] + dot(vec(A), Ω * vec(A)))
        return PlausiblyExogenousResult(:ltz, float(level), est - crit * se,
                                        est + crit * se, est, se, (est, est), (μ, Ω),
                                        Matrix(A), r.instruments, r.outcome,
                                        r.endogenous[1])
    end
    throw(ArgumentError("method must be :uci or :ltz, got :$method"))
end
