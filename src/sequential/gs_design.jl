# Group-sequential designs: efficacy boundaries (classical Pocock / O'Brien–Fleming /
# Haybittle–Peto or Lan–DeMets alpha spending), optional beta-spending futility
# boundaries (non-binding by default), maximum-information inflation and operating
# characteristics.

"""
    GroupSequentialDesign

A group-sequential design with ``K`` planned analyses, created by
[`gs_design`](@ref): efficacy (and optionally futility) boundaries, the maximum
information relative to a fixed-sample design, and the operating characteristics.

Boundaries are on the ``Z`` scale: the standardized statistic
``Z_k = \\hat\\theta_k / \\mathrm{se}(\\hat\\theta_k)`` at look ``k``, positive values
favouring the alternative. All probabilities are computed under the canonical joint
distribution of ``(Z_1, \\ldots, Z_K)``: multivariate normal with
``E[Z_k] = \\theta\\sqrt{I_k}`` and ``\\mathrm{Cov}(Z_j, Z_k) = \\sqrt{I_j/I_k}`` for
``j \\le k`` (Jennison and Turnbull 2000, ch. 3), which holds exactly for normal data
with known variance and asymptotically for efficient estimators in most regular models.
The design is reported relative to the fixed-sample design with the same ``\\alpha``
and power, so that it can be combined with any sample-size formula (for instance
those of [`power_means`](@ref) or [`power_proportions`](@ref)) by multiplying the
fixed sample size by `inflation`. Pass the design to [`gs_analysis`](@ref) at each
look.

# Fields
- `k::Int`: number of planned analyses.
- `sided::Int`: 1 for a one-sided efficacy test of ``H_0: \\theta \\le 0``; 2 for a
  two-sided symmetric test of ``H_0: \\theta = 0``.
- `alpha::Float64`: total type-I error; `beta::Float64`: type-II error at the design
  alternative.
- `timing::Vector{Float64}`: planned information fractions ``t_k = I_k / I_{\\max}``.
- `efficacy`: the efficacy rule, a [`SpendingFunction`](@ref) or one of `:pocock`,
  `:obrien_fleming`, `:haybittle_peto`.
- `futility`: a [`SpendingFunction`](@ref) for β-spending, or `nothing`.
- `binding::Bool`: whether the futility boundary is binding (enters the type-I error
  calculation).
- `efficacy_z::Vector{Float64}`: efficacy boundaries; for two-sided designs the lower
  boundary is `-efficacy_z`.
- `futility_z::Vector{Float64}`: futility boundaries (`-Inf` without futility).
- `nominal_p::Vector{Float64}`: one-sided nominal p-values ``1 - \\Phi(c_k)`` of the
  efficacy boundaries.
- `alpha_spent::Vector{Float64}`: cumulative probability under ``H_0`` of crossing
  the efficacy boundary (ignoring a non-binding futility boundary; both sides when
  `sided = 2`).
- `beta_spent::Vector{Float64}`: cumulative probability under the alternative of
  crossing the futility boundary (zeros without futility).
- `drift::Float64`: ``\\theta\\sqrt{I_{\\max}}`` at the design alternative.
- `inflation::Float64`: ``I_{\\max}/I_{\\text{fixed}}``, the maximum sample size
  relative to the fixed-sample design.
- `power::Float64`: probability of crossing the efficacy boundary under the
  alternative.
- `prob_efficacy_h0`, `prob_efficacy_h1`, `prob_futility_h0`,
  `prob_futility_h1::Vector{Float64}`: per-look crossing probabilities under ``H_0``
  and the alternative, with the futility boundary obeyed.
- `expected_information_h0`, `expected_information_h1::Float64`: expected information
  (average sample number) relative to the fixed design, under ``H_0`` and the
  alternative.
- `n_fixed::Float64`: fixed-design sample size (`NaN` if not given);
  `n::Vector{Float64}`: cumulative sample size at each look,
  ``n_{\\text{fixed}} \\times \\text{inflation} \\times t_k``.
- `hp_bound::Float64`: interim boundary of Haybittle–Peto designs.

# References
- Jennison, C., & Turnbull, B. W. (2000). *Group Sequential Methods with Applications
  to Clinical Trials*. Chapman & Hall/CRC.
- Wassmer, G., & Brannath, W. (2016). *Group Sequential and Confirmatory Adaptive
  Designs in Clinical Trials*. Springer.
"""
struct GroupSequentialDesign
    k::Int
    sided::Int
    alpha::Float64
    beta::Float64
    timing::Vector{Float64}
    efficacy::Union{SpendingFunction,Symbol}
    futility::Union{Nothing,SpendingFunction}
    binding::Bool
    efficacy_z::Vector{Float64}
    futility_z::Vector{Float64}
    nominal_p::Vector{Float64}
    alpha_spent::Vector{Float64}
    beta_spent::Vector{Float64}
    drift::Float64
    inflation::Float64
    power::Float64
    prob_efficacy_h0::Vector{Float64}
    prob_efficacy_h1::Vector{Float64}
    prob_futility_h0::Vector{Float64}
    prob_futility_h1::Vector{Float64}
    expected_information_h0::Float64
    expected_information_h1::Float64
    n_fixed::Float64
    n::Vector{Float64}
    hp_bound::Float64
end

"""
    _seq_gs_efficacy(rule, timing, info, alpha_side, sided; hp_bound, lower=nothing)

Efficacy bounds for looks with information fractions `timing` and informations `info`
(any units) at one-sided level `alpha_side` per side. `lower` optionally gives binding
lower bounds (one-sided designs).
"""
function _seq_gs_efficacy(rule, timing::AbstractVector, info::AbstractVector,
                          alpha_side::Float64, sided::Int; hp_bound::Float64=3.0,
                          lower=nothing)
    K = length(timing)
    lowb(k, b) = sided == 2 ? -b : (lower === nothing ? -Inf : float(lower[k]))
    if rule isa SpendingFunction
        b = zeros(K)
        s = _seq_gs_start()
        prev = 0.0
        for k in 1:K
            cum = spending(rule, timing[k], alpha_side)
            target = max(cum - prev, 0.0)
            prev = cum
            if sided == 2
                # symmetric: P(Z ≥ b) = P(Z ≤ -b) under H₀ by symmetry
                fs(x) = _seq_gs_up(s, 0.0, float(info[k]), x) - target
                b[k] = target <= 1e-300 ? Inf :
                       (fs(0.0) < 0 ? 0.0 : _seq_root(fs, 0.0, _SEQ_GS_ZMAX))
            else
                b[k] = _seq_gs_solve_upper(s, 0.0, float(info[k]), target)
            end
            k < K && (s = _seq_gs_advance(s, 0.0, float(info[k]), lowb(k, b[k]), b[k]))
        end
        return b
    elseif rule === :haybittle_peto
        b = fill(hp_bound, K)
        K == 1 && return [quantile(Normal(), 1 - alpha_side)]
        s = _seq_gs_start()
        spent = 0.0
        for k in 1:(K - 1)
            spent += _seq_gs_up(s, 0.0, float(info[k]), b[k])
            s = _seq_gs_advance(s, 0.0, float(info[k]), lowb(k, b[k]), b[k])
        end
        target = alpha_side - spent
        if target <= 0
            b[K] = Inf              # the interim bound alone exhausts α
        elseif sided == 2
            ff(x) = _seq_gs_up(s, 0.0, float(info[K]), x) - target
            b[K] = _seq_root(ff, 0.0, _SEQ_GS_ZMAX)
        else
            b[K] = _seq_gs_solve_upper(s, 0.0, float(info[K]), target)
        end
        return b
    else
        # Wang–Tsiatis family: b_k = C t_k^(Δ - 1/2), Δ = 0 (OBF) or 1/2 (Pocock)
        shape = rule === :pocock ? ones(K) : 1 ./ sqrt.(timing ./ timing[end])
        function total(c)
            bb = c .* shape
            a = [lowb(k, bb[k]) for k in 1:K]
            up, _ = _seq_gs_probs(a, bb, info, 0.0)
            return sum(up) - alpha_side
        end
        c = _seq_root(total, 0.0, _SEQ_GS_ZMAX)
        return c .* shape
    end
end

function _seq_gs_rule(efficacy)
    efficacy isa SpendingFunction && return efficacy
    efficacy in (:pocock, :obrien_fleming, :haybittle_peto) ||
        throw(ArgumentError("efficacy must be a SpendingFunction or one of :pocock, " *
                            ":obrien_fleming, :haybittle_peto; got $efficacy"))
    return efficacy
end

"""
    gs_design(; k=3, alpha=0.025, beta=0.1, sided=1, timing=nothing,
              efficacy=OBFSpending(), futility=nothing, binding=false,
              hp_bound=3.0, n_fixed=nothing) -> GroupSequentialDesign

Group-sequential design with `k` analyses at information fractions `timing`,
controlling the type-I error at `alpha` and with power ``1 - \\beta`` at the design
alternative.

A group-sequential trial analyses the accumulating data at ``K`` pre-planned looks
and stops early when the evidence is strong enough (Armitage, McPherson and Rowe
1969; Pocock 1977). Testing each look at the nominal level would inflate the type-I
error; the boundaries ``c_1, \\ldots, c_K`` are therefore chosen so that the
probability under ``H_0`` of ever crossing is `alpha`. With the canonical joint
distribution of the ``Z`` statistics (see [`GroupSequentialDesign`](@ref)) the
crossing probabilities are computed by recursive numerical integration (Armitage,
McPherson and Rowe 1969; Jennison and Turnbull 2000, ch. 19), on the grid used by the
gsDesign package. The maximum information ``I_{\\max}`` is then chosen so that the
power at the design alternative is ``1 - \\beta``, and is reported relative to the
fixed-sample design as `inflation`. Under the canonical normal joint distribution the
design's type-I error equals `alpha` up to numerical integration error; when the
statistics are only asymptotically normal (estimated variances, binary or survival
outcomes) it is `alpha` only approximately.

**Efficacy boundaries (`efficacy`).**
- A [`SpendingFunction`](@ref): Lan–DeMets error spending, the boundary at look ``k``
  spends ``f(t_k) - f(t_{k-1})``. [`OBFSpending`](@ref) is the usual default:
  practically no early stopping unless the effect is large, and a final critical
  value close to the fixed-sample one. Spending designs remain valid when the number
  or timing of the looks departs from the plan (see [`gs_analysis`](@ref)).
- `:obrien_fleming` or `:pocock`: the classical boundaries ``c/\\sqrt{t_k}``
  (O'Brien and Fleming 1979) and constant ``c`` (Pocock 1977), members of the
  Wang and Tsiatis (1987) family, with ``c`` chosen to give total type-I error
  `alpha`. They assume the planned number and timing of the looks.
- `:haybittle_peto`: the fixed boundary `hp_bound` (default 3) at every interim look
  (Haybittle 1971) and a final boundary adjusted so that the total type-I error under
  the canonical distribution is `alpha`.

**Futility (`futility`).** A [`SpendingFunction`](@ref) for β-spending (e.g.
`HSDSpending(-2)`): the futility boundary at look ``k`` is set so that the
probability of stopping for futility under the alternative, by look ``k``, is
``f(t_k; \\beta)``, and the maximum information is chosen so that the efficacy and
futility boundaries meet at the final analysis (gsDesign `test.type = 4` for
non-binding and `3` for binding futility). With `binding = false` (recommended) the
efficacy boundaries ignore the futility boundary: the type-I error is at most `alpha`
whether or not the futility boundary is obeyed, and strictly below `alpha` when it is
(the stated `alpha` is then conservative). With `binding = true` the efficacy
boundaries are lowered to exploit the futility stops, and type-I error control
requires stopping whenever the futility boundary is crossed, a commitment that is
rarely credible when the decision rests with a monitoring committee. Futility is
supported for one-sided designs only.

**Practical guidance.** Choose the number of looks and the spending function before
the trial, from the value of early stopping and the tolerable increase in maximum
sample size (compare `inflation` and `expected_information_h1` across candidates). A
continuously monitored design ([`confseq_ate`](@ref), [`msprt_test`](@ref)) is the
alternative when the analysis schedule cannot be planned; for a few planned looks the
group-sequential design is more efficient.

# Keywords
- `k::Integer = 3`: number of analyses (at least 1; `k = 1` is the fixed design).
- `alpha::Real = 0.025`: total type-I error; one-sided for `sided = 1`, split equally
  over the two sides for `sided = 2`.
- `beta::Real = 0.1`: type-II error at the design alternative (power 0.9).
- `sided::Integer = 1`: 1 (one-sided efficacy test) or 2 (two-sided symmetric test).
- `timing = nothing`: strictly increasing planned information fractions ending at 1;
  `nothing` means equally spaced looks.
- `efficacy = OBFSpending()`: the efficacy rule described above.
- `futility = nothing`: a β-spending function for a futility boundary, or `nothing`.
- `binding::Bool = false`: whether the futility boundary is binding; requires a
  futility spending function and an efficacy spending function.
- `hp_bound::Real = 3.0`: Haybittle–Peto interim boundary.
- `n_fixed = nothing`: sample size of the fixed design with the same `alpha` and
  power; when given, the cumulative sample sizes per look are reported in `n`.

# Returns
- [`GroupSequentialDesign`](@ref).

# Examples
```julia
using DrSnow
d = gs_design(; k=3, alpha=0.025, beta=0.1)                 # OBF-type spending
d.efficacy_z, d.inflation
gs_design(; k=4, efficacy=:pocock, n_fixed=500).n           # sample size per look
gs_design(; k=3, futility=HSDSpending(-2))                  # non-binding futility
```

# References
- Pocock, S. J. (1977). Group sequential methods in the design and analysis of
  clinical trials. *Biometrika*, 64(2), 191–199.
- O'Brien, P. C., & Fleming, T. R. (1979). A multiple testing procedure for clinical
  trials. *Biometrics*, 35(3), 549–556.
- Haybittle, J. L. (1971). Repeated assessment of results in clinical trials of cancer
  treatment. *British Journal of Radiology*, 44(526), 793–797.
- Wang, S. K., & Tsiatis, A. A. (1987). Approximately optimal one-parameter boundaries
  for group sequential trials. *Biometrics*, 43(1), 193–199.
- Lan, K. K. G., & DeMets, D. L. (1983). Discrete sequential boundaries for clinical
  trials. *Biometrika*, 70(3), 659–663.
- Armitage, P., McPherson, C. K., & Rowe, B. C. (1969). Repeated significance tests on
  accumulating data. *Journal of the Royal Statistical Society: Series A*, 132(2),
  235–244.
- Jennison, C., & Turnbull, B. W. (2000). *Group Sequential Methods with Applications
  to Clinical Trials*. Chapman & Hall/CRC.
- Wassmer, G., & Brannath, W. (2016). *Group Sequential and Confirmatory Adaptive
  Designs in Clinical Trials*. Springer.
"""
function gs_design(; k::Integer=3, alpha::Real=0.025, beta::Real=0.1,
                   sided::Integer=1, timing=nothing, efficacy=OBFSpending(),
                   futility=nothing, binding::Bool=false, hp_bound::Real=3.0,
                   n_fixed=nothing)
    k >= 1 || throw(ArgumentError("k must be at least 1"))
    sided in (1, 2) || throw(ArgumentError("sided must be 1 or 2"))
    (0 < alpha < (sided == 1 ? 0.5 : 1)) ||
        throw(ArgumentError("alpha must be in (0, 0.5) for one-sided designs"))
    (0 < beta < 1 - alpha / sided) || throw(ArgumentError("beta must be in (0, 1-α)"))
    t = timing === nothing ? collect(1:k) ./ k : float.(collect(timing))
    length(t) == k || throw(DimensionMismatch("timing must have k = $k entries"))
    (all(>(0), t) && issorted(t; lt=<=) && isapprox(t[end], 1; atol=1e-12)) ||
        throw(ArgumentError("timing must be strictly increasing, positive and end at 1"))
    t[end] = 1.0
    rule = _seq_gs_rule(efficacy)
    futility === nothing || futility isa SpendingFunction ||
        throw(ArgumentError("futility must be a SpendingFunction or nothing"))
    futility !== nothing && sided == 2 &&
        throw(ArgumentError("futility bounds are supported for one-sided designs only"))
    binding && futility === nothing &&
        throw(ArgumentError("binding = true needs a futility spending function"))
    binding && !(rule isa SpendingFunction) &&
        throw(ArgumentError("binding futility requires an efficacy spending function"))
    n_fixed === nothing || n_fixed > 0 || throw(ArgumentError("n_fixed must be positive"))
    a_side = sided == 2 ? alpha / 2 : float(alpha)
    hp = float(hp_bound)
    zfix = quantile(Normal(), 1 - a_side) + quantile(Normal(), 1 - beta)

    b = zeros(k)
    a = fill(-Inf, k)
    drift = 0.0
    if futility === nothing
        b = _seq_gs_efficacy(rule, t, t, a_side, sided; hp_bound=hp)
        isfinite(b[end]) ||
            throw(ArgumentError("the Haybittle–Peto interim bound $hp_bound already " *
                                "spends the whole α; raise hp_bound"))
        a = sided == 2 ? -b : fill(-Inf, k)
        pw(th) = sum(_seq_gs_probs(a, b, t, th)[1]) - (1 - beta)
        drift = _seq_root(pw, 0.0, 3 * zfix + 10; tol=1e-10)
    else
        bnb = binding ? nothing : _seq_gs_efficacy(rule, t, t, a_side, 1; hp_bound=hp)
        function final_gap(th)
            bb, aa = _seq_gs_futility_bounds(rule, futility, t, a_side, float(beta), th,
                                             bnb)
            return aa, bb
        end
        # β spent at the end (a_K = b_K) minus β: decreasing in the drift
        function g(th)
            aa, bb = final_gap(th)
            _, lo = _seq_gs_probs(aa, bb, t, th)
            return sum(lo) - beta
        end
        drift = _seq_root(g, 1e-6, 3 * zfix + 10; tol=1e-10)
        a, b = final_gap(drift)
    end

    # H₀ probabilities: efficacy ignoring a non-binding futility bound
    a_h0 = sided == 2 ? -b : (binding ? a : fill(-Inf, k))
    up0, lo0 = _seq_gs_probs(a_h0, b, t, 0.0)
    alpha_spent = cumsum(sided == 2 ? up0 .+ lo0 : up0)
    up0f, lo0f = _seq_gs_probs(a, b, t, 0.0)       # with futility obeyed
    up1, lo1 = _seq_gs_probs(a, b, t, drift)
    R = (drift / zfix)^2
    en(up, lo) = R * sum(t[1:(k - 1)] .* (up[1:(k - 1)] .+ lo[1:(k - 1)]); init=0.0) +
                 R * (1 - sum(up[1:(k - 1)] .+ lo[1:(k - 1)]; init=0.0))
    nf = n_fixed === nothing ? NaN : float(n_fixed)
    return GroupSequentialDesign(k, sided, float(alpha), float(beta), t, rule, futility,
                                 binding, b, a, ccdf.(Normal(), b), alpha_spent,
                                 futility === nothing ? zeros(k) : cumsum(lo1), drift, R,
                                 sum(up1), up0f, up1, lo0f, lo1, en(up0f, lo0f),
                                 en(up1, lo1), nf, nf .* R .* t, hp)
end

# Efficacy and futility bounds for drift `th` (stagewise; gsDesign test.type 3/4).
# `bnb`: precomputed non-binding efficacy bounds, or `nothing` for binding futility.
function _seq_gs_futility_bounds(rule, fut::SpendingFunction, t, a_side, beta, th, bnb)
    K = length(t)
    a = fill(-Inf, K)
    b = bnb === nothing ? zeros(K) : copy(bnb)
    s1 = _seq_gs_start()                 # under the drift
    s0 = _seq_gs_start()                 # under H₀ (binding only)
    prev_a = 0.0
    prev_b = 0.0
    for k in 1:K
        if bnb === nothing
            cum = spending(rule, t[k], a_side)
            b[k] = _seq_gs_solve_upper(s0, 0.0, t[k], max(cum - prev_a, 0.0))
            prev_a = cum
        end
        if k < K
            cumb = spending(fut, t[k], beta)
            a[k] = min(_seq_gs_solve_lower(s1, th, t[k], max(cumb - prev_b, 0.0)), b[k])
            prev_b = cumb
            s1 = _seq_gs_advance(s1, th, t[k], a[k], b[k])
            bnb === nothing && (s0 = _seq_gs_advance(s0, 0.0, t[k], a[k], b[k]))
        else
            a[k] = b[k]
        end
    end
    return b, a
end

function Base.show(io::IO, ::MIME"text/plain", d::GroupSequentialDesign)
    side = d.sided == 1 ? "one-sided" : "two-sided symmetric"
    println(io, "Group-sequential design: $(d.k) analyses, $side, α = $(d.alpha), " *
                "power = $(round(1 - d.beta; digits=4))")
    println(io, "Efficacy: ", _seq_sf_name(d.efficacy))
    d.futility === nothing ||
        println(io, "Futility: ", _seq_sf_name(d.futility), " (β-spending, ",
                d.binding ? "binding" : "non-binding", ")")
    @printf(io, "Inflation factor (I_max / I_fixed): %.4f\n", d.inflation)
    @printf(io, "Expected information / fixed design: %.4f under H₀, %.4f under H₁\n",
            d.expected_information_h0, d.expected_information_h1)
    hasn = !isnan(d.n_fixed)
    println(io, rpad("look", 6), rpad("t", 8), hasn ? rpad("n", 10) : "",
            rpad("efficacy z", 12), rpad("nominal p", 12), rpad("α spent", 10),
            d.futility === nothing ? "" : rpad("futility z", 12),
            rpad("P(eff|H₁)", 10))
    for k in 1:d.k
        print(io, rpad(k, 6), rpad(@sprintf("%.3f", d.timing[k]), 8))
        hasn && print(io, rpad(@sprintf("%.1f", d.n[k]), 10))
        print(io, rpad(@sprintf("%.4f", d.efficacy_z[k]), 12),
              rpad(@sprintf("%.5f", d.nominal_p[k]), 12),
              rpad(@sprintf("%.5f", d.alpha_spent[k]), 10))
        d.futility === nothing || print(io, rpad(@sprintf("%.4f", d.futility_z[k]), 12))
        println(io, rpad(@sprintf("%.4f", d.prob_efficacy_h1[k]), 10))
    end
end

Base.show(io::IO, d::GroupSequentialDesign) =
    print(io, "GroupSequentialDesign(k = $(d.k), α = $(d.alpha), inflation = ",
          @sprintf("%.4f", d.inflation), ")")
