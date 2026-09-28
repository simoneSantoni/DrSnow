# Error-spending functions (Lan & DeMets 1983; Kim & DeMets 1987; Hwang, Shih &
# DeCani 1990). A spending function gives the cumulative error `f(t; α)` spent by
# information fraction `t ∈ [0, 1]`, with `f(0) = 0` and `f(1) = α`.

"""
    SpendingFunction

Abstract supertype of error-spending functions for group-sequential designs.

A spending function ``f(t; \\alpha)`` gives the cumulative type-I error (or, used for
futility, type-II error) that a group-sequential test may have spent by information
fraction ``t = I/I_{\\max} \\in [0, 1]``; it increases from ``f(0) = 0`` to
``f(1) = \\alpha``. At an analysis with information fraction ``t_k`` the efficacy
boundary ``c_k`` is chosen so that the probability under ``H_0`` of crossing for the
first time at look ``k`` equals ``f(t_k) - f(t_{k-1})`` (Lan and DeMets 1983). The
boundaries are computed under the canonical joint distribution of the standardized
statistics (independent Gaussian increments in information; Jennison and Turnbull
2000, ch. 3), which holds exactly for normal data with known variance and
asymptotically for most regular estimators.

The advantage over the classical Pocock (1977) and O'Brien and Fleming (1979)
boundaries is flexibility: only the information fractions of the analyses that are
actually performed are needed, so the number and timing of interim analyses need not
be fixed in advance, provided they are not chosen on the basis of the accumulating
results (Lan and DeMets 1989). The shape of the function governs the trade-off
between early stopping and the maximum sample size: conservative functions
([`OBFSpending`](@ref), [`PowerSpending`](@ref) with ``\\rho = 3``,
[`HSDSpending`](@ref) with ``\\gamma = -4``) spend little error early and keep the
final critical value close to the fixed-sample one; aggressive functions
([`PocockSpending`](@ref), ``\\rho = 1``, ``\\gamma = 1``) stop earlier on average at
the cost of a larger maximum sample size. Evaluate a function with
[`spending`](@ref); use it in [`gs_design`](@ref).

# References
- Lan, K. K. G., & DeMets, D. L. (1983). Discrete sequential boundaries for clinical
  trials. *Biometrika*, 70(3), 659–663.
- Lan, K. K. G., & DeMets, D. L. (1989). Changing frequency of interim analysis in
  sequential monitoring. *Biometrics*, 45(3), 1017–1020.
- Jennison, C., & Turnbull, B. W. (2000). *Group Sequential Methods with Applications
  to Clinical Trials*. Chapman & Hall/CRC.
- Proschan, M. A., Lan, K. K. G., & Wittes, J. T. (2006). *Statistical Monitoring of
  Clinical Trials: A Unified Approach*. Springer.
"""
abstract type SpendingFunction end

"""
    OBFSpending() -> OBFSpending

Lan–DeMets spending function that approximates the O'Brien–Fleming boundary.

The function is

```math
f(t; \\alpha) = 2 - 2\\Phi\\left(\\frac{\\Phi^{-1}(1 - \\alpha/2)}{\\sqrt{t}}\\right),
```

proposed by Lan and DeMets (1983) as a spending analogue of the O'Brien and Fleming
(1979) boundary ``c/\\sqrt{t_k}`` (gsDesign `sfLDOF`, rpact `asOF`). It spends almost
no error early: interim boundaries are very high (the first of three equally spaced
looks at one-sided ``\\alpha = 0.025`` requires ``z`` above 3.7), so trials stop early
only for large effects, and the final critical value is close to the fixed-sample one
(about 2.0 instead of 1.96). The inflation of the maximum sample size over the
fixed-sample design is therefore small. It is the conventional default in
confirmatory trials, where early stopping on modest evidence is undesirable.

# Returns
- `OBFSpending`, a [`SpendingFunction`](@ref); evaluate it with [`spending`](@ref).

# Examples
```julia
using DrSnow
spending(OBFSpending(), [0.25, 0.5, 1.0], 0.025)
gs_design(; k=3, efficacy=OBFSpending()).efficacy_z
```

# References
- Lan, K. K. G., & DeMets, D. L. (1983). Discrete sequential boundaries for clinical
  trials. *Biometrika*, 70(3), 659–663.
- O'Brien, P. C., & Fleming, T. R. (1979). A multiple testing procedure for clinical
  trials. *Biometrics*, 35(3), 549–556.
- Jennison, C., & Turnbull, B. W. (2000). *Group Sequential Methods with Applications
  to Clinical Trials*. Chapman & Hall/CRC.
"""
struct OBFSpending <: SpendingFunction end

"""
    PocockSpending() -> PocockSpending

Lan–DeMets spending function that approximates the Pocock boundary.

The function is

```math
f(t; \\alpha) = \\alpha \\log\\{1 + (e - 1)t\\},
```

the spending analogue of the constant boundary of Pocock (1977) proposed by Lan and
DeMets (1983) (gsDesign `sfLDPocock`, rpact `asP`). It spends error roughly evenly
over the information scale, which yields nearly constant boundaries on the ``z``
scale (about 2.28–2.30 for three equally spaced looks at one-sided
``\\alpha = 0.025``). Early stopping is more likely than with
[`OBFSpending`](@ref), but the final critical value is well above the fixed-sample
one, so a trial that runs to the end needs a larger maximum sample size (an inflation
of roughly 15% for three looks at 90% power) and loses power if the effect is small.
It suits settings where early stopping has high value, for instance when an effect,
if present, is expected to be large.

# Returns
- `PocockSpending`, a [`SpendingFunction`](@ref); evaluate it with
  [`spending`](@ref).

# Examples
```julia
using DrSnow
spending(PocockSpending(), 0.5, 0.025)
gs_design(; k=3, efficacy=PocockSpending()).inflation
```

# References
- Lan, K. K. G., & DeMets, D. L. (1983). Discrete sequential boundaries for clinical
  trials. *Biometrika*, 70(3), 659–663.
- Pocock, S. J. (1977). Group sequential methods in the design and analysis of
  clinical trials. *Biometrika*, 64(2), 191–199.
- Jennison, C., & Turnbull, B. W. (2000). *Group Sequential Methods with Applications
  to Clinical Trials*. Chapman & Hall/CRC.
"""
struct PocockSpending <: SpendingFunction end

"""
    PowerSpending(rho) -> PowerSpending

Kim–DeMets power family of spending functions.

The function is

```math
f(t; \\alpha) = \\alpha\\, t^{\\rho}, \\qquad \\rho > 0,
```

introduced by Kim and DeMets (1987) (gsDesign `sfPower`, rpact `asKD` with
`gammaA` ``= \\rho``). The single parameter ``\\rho`` interpolates between aggressive
and conservative spending: ``\\rho = 1`` spends error linearly in information and
gives boundaries similar to Pocock's, ``\\rho = 3`` gives boundaries similar to
O'Brien and Fleming's, and larger values spend even less early (Jennison and Turnbull
2000). The family is also a common choice for β-spending futility
boundaries.

# Arguments
- `rho::Real`: the exponent ``\\rho``, positive and finite. Larger values spend less
  error at early looks.

# Returns
- `PowerSpending`, a [`SpendingFunction`](@ref); evaluate it with
  [`spending`](@ref).

# Examples
```julia
using DrSnow
spending(PowerSpending(3), 0.5, 0.025)     # 0.025 × 0.5³ = 0.003125
gs_design(; k=4, efficacy=PowerSpending(2)).efficacy_z
```

# References
- Kim, K., & DeMets, D. L. (1987). Design and analysis of group sequential tests
  based on the type I error spending rate function. *Biometrika*, 74(1), 149–154.
- Lan, K. K. G., & DeMets, D. L. (1983). Discrete sequential boundaries for clinical
  trials. *Biometrika*, 70(3), 659–663.
- Jennison, C., & Turnbull, B. W. (2000). *Group Sequential Methods with Applications
  to Clinical Trials*. Chapman & Hall/CRC.
"""
struct PowerSpending <: SpendingFunction
    rho::Float64
    function PowerSpending(rho::Real)
        (rho > 0 && isfinite(rho)) || throw(ArgumentError("rho must be positive"))
        return new(float(rho))
    end
end

"""
    HSDSpending(gamma) -> HSDSpending

Hwang–Shih–DeCani family of spending functions.

The function is

```math
f(t; \\alpha) = \\alpha\\, \\frac{1 - e^{-\\gamma t}}{1 - e^{-\\gamma}}
  \\quad (\\gamma \\ne 0), \\qquad f(t; \\alpha) = \\alpha t \\quad (\\gamma = 0),
```

proposed by Hwang, Shih and DeCani (1990) (gsDesign `sfHSD`, rpact `asHSD`). Negative
values of ``\\gamma`` spend little error early (``\\gamma = -4`` resembles
O'Brien–Fleming), positive values spend it early (``\\gamma = 1`` resembles Pocock),
and ``\\gamma = 0`` spends it linearly. Because the family is continuous in
``\\gamma``, it is convenient for tuning a design to a desired balance between
expected and maximum sample size; ``\\gamma = -2`` is a common choice for β-spending
futility boundaries (the gsDesign default for the lower bound).

# Arguments
- `gamma::Real`: the shape parameter ``\\gamma``, any finite number.

# Returns
- `HSDSpending`, a [`SpendingFunction`](@ref); evaluate it with [`spending`](@ref).

# Examples
```julia
using DrSnow
spending(HSDSpending(-4), 0.5, 0.025)
gs_design(; k=3, futility=HSDSpending(-2)).futility_z
```

# References
- Hwang, I. K., Shih, W. J., & DeCani, J. S. (1990). Group sequential designs using a
  family of type I error probability spending functions. *Statistics in Medicine*,
  9(12), 1439–1445.
- Jennison, C., & Turnbull, B. W. (2000). *Group Sequential Methods with Applications
  to Clinical Trials*. Chapman & Hall/CRC.
"""
struct HSDSpending <: SpendingFunction
    gamma::Float64
    HSDSpending(gamma::Real) = isfinite(gamma) ? new(float(gamma)) :
                               throw(ArgumentError("gamma must be finite"))
end

"""
    spending(sf::SpendingFunction, t, alpha) -> Float64 or Vector{Float64}

Cumulative error spent by information fraction `t` out of a total `alpha`.

Evaluates ``f(t; \\alpha)`` for a [`SpendingFunction`](@ref). In a group-sequential
design the increment ``f(t_k; \\alpha) - f(t_{k-1}; \\alpha)`` is the probability, under
``H_0``, of crossing the efficacy boundary for the first time at look ``k`` (or,
for β-spending, of crossing the futility boundary for the first time under the
design alternative). Tabulating the function at the planned information fractions
shows how a design distributes its error over the analyses, which is useful for
choosing between spending functions and for reporting the design.

# Arguments
- `sf::SpendingFunction`: the spending function.
- `t`: information fraction(s) in ``[0, 1]``, a number or a vector.
- `alpha::Real`: the total error to spend, in ``(0, 1)``.

# Returns
- The cumulative error spent at `t`: a `Float64` for a number, a
  `Vector{Float64}` for a vector.

# Examples
```julia
using DrSnow
t = [1/3, 2/3, 1]
spending(OBFSpending(), t, 0.025)
diff([0; spending(PocockSpending(), t, 0.025)])    # error spent at each look
```

# References
- Lan, K. K. G., & DeMets, D. L. (1983). Discrete sequential boundaries for clinical
  trials. *Biometrika*, 70(3), 659–663.
- Kim, K., & DeMets, D. L. (1987). Design and analysis of group sequential tests
  based on the type I error spending rate function. *Biometrika*, 74(1), 149–154.
- Hwang, I. K., Shih, W. J., & DeCani, J. S. (1990). Group sequential designs using a
  family of type I error probability spending functions. *Statistics in Medicine*,
  9(12), 1439–1445.
"""
spending(sf::SpendingFunction, t::AbstractVector, alpha::Real) =
    [spending(sf, ti, alpha) for ti in t]

function _seq_check_t(t, alpha)
    (0 <= t <= 1) || throw(ArgumentError("information fraction must be in [0, 1], " *
                                         "got $t"))
    (0 < alpha < 1) || throw(ArgumentError("alpha must be in (0, 1), got $alpha"))
    return nothing
end

function spending(::OBFSpending, t::Real, alpha::Real)
    _seq_check_t(t, alpha)
    t == 0 && return 0.0
    return 2 * ccdf(Normal(), quantile(Normal(), 1 - alpha / 2) / sqrt(t))
end

function spending(::PocockSpending, t::Real, alpha::Real)
    _seq_check_t(t, alpha)
    return alpha * log(1 + (exp(1) - 1) * t)
end

function spending(sf::PowerSpending, t::Real, alpha::Real)
    _seq_check_t(t, alpha)
    return alpha * t^sf.rho
end

function spending(sf::HSDSpending, t::Real, alpha::Real)
    _seq_check_t(t, alpha)
    g = sf.gamma
    abs(g) < 1e-10 && return alpha * t
    return alpha * (-expm1(-g * t)) / (-expm1(-g))
end

_seq_sf_name(::OBFSpending) = "Lan–DeMets O'Brien–Fleming spending"
_seq_sf_name(::PocockSpending) = "Lan–DeMets Pocock spending"
_seq_sf_name(sf::PowerSpending) = "Kim–DeMets power spending (ρ = $(sf.rho))"
_seq_sf_name(sf::HSDSpending) = "Hwang–Shih–DeCani spending (γ = $(sf.gamma))"
_seq_sf_name(s::Symbol) = s === :pocock ? "Pocock (constant) boundary" :
                          s === :obrien_fleming ? "O'Brien–Fleming boundary" :
                          "Haybittle–Peto boundary"
