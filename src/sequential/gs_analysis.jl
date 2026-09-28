# Analysis of a group-sequential trial at its interim / final looks: boundaries at
# the observed information, stopping decision, repeated confidence intervals and
# repeated p-values (Jennison & Turnbull 1989), and after stopping the adjusted
# p-value, median-unbiased estimate and confidence interval under the stagewise
# ordering (Armitage 1957; Tsiatis, Rosner & Mehta 1984).

"""
    GroupSequentialAnalysis <: CausalEstimate

Result of [`gs_analysis`](@ref): the boundaries at the observed looks, the stopping
decision, repeated confidence intervals and p-values, and, once the trial has
stopped, inference that accounts for the sequential design.

The accessors switch meaning when the trial stops, because the appropriate inference
differs. Before stopping, `coef(r)` is the naive estimate at the latest look,
`confint(r)` the repeated confidence interval (RCI) there and `pvalues(r)` the
repeated p-value; these are valid whatever stopping rule is eventually followed
(Jennison and Turnbull 1989). Once the trial has stopped (efficacy boundary crossed,
or final look reached), `coef(r)` is the median-unbiased estimate, `confint(r)` the
stagewise-ordering confidence interval and `pvalues(r)` the stagewise-ordering
p-value (Tsiatis, Rosner and Mehta 1984). The naive estimate at the stopping look is
biased away from zero after an early efficacy stop and should not be reported as the
effect. `stderror(r)` is the naive standard error at the latest look, for reference.

# Fields
- `design::GroupSequentialDesign`: the design analysed.
- `estimates`, `std_errors`, `z`, `information`, `timing::Vector{Float64}`: per look,
  the estimate, its standard error, ``Z_k``, ``I_k = 1/\\mathrm{se}_k^2`` and the
  information fraction used for spending.
- `efficacy_z`, `futility_z::Vector{Float64}`: boundaries at the observed looks.
- `rci_lower`, `rci_upper::Vector{Float64}`: repeated confidence intervals.
- `repeated_pvalue::Vector{Float64}`: repeated p-value after each look.
- `decision::Symbol`: `:efficacy`, `:futility` (non-binding boundary crossed),
  `:continue`, or `:final_no_rejection`.
- `stop_look::Int`: look at which the decision was reached (the latest look for
  `:continue`).
- `stopped::Bool`: `true` for `:efficacy` and `:final_no_rejection`.
- `pvalue_adjusted::Union{Nothing,Float64}`,
  `estimate_median_unbiased::Union{Nothing,Float64}`,
  `ci_adjusted::Union{Nothing,NTuple{2,Float64}}`: stagewise-ordering inference
  (`nothing` while the trial continues or after a futility stop).
- `level::Float64`: level of the adjusted interval.
- `nobs::Int`: cumulative sample size at the last look (0 when not given).

# References
- Jennison, C., & Turnbull, B. W. (1989). Interim analyses: The repeated confidence
  interval approach. *Journal of the Royal Statistical Society: Series B*, 51(3),
  305–334.
- Tsiatis, A. A., Rosner, G. L., & Mehta, C. R. (1984). Exact confidence intervals
  following a group sequential test. *Biometrics*, 40(3), 797–803.
"""
struct GroupSequentialAnalysis <: CausalEstimate
    design::GroupSequentialDesign
    estimates::Vector{Float64}
    std_errors::Vector{Float64}
    z::Vector{Float64}
    information::Vector{Float64}
    timing::Vector{Float64}
    efficacy_z::Vector{Float64}
    futility_z::Vector{Float64}
    rci_lower::Vector{Float64}
    rci_upper::Vector{Float64}
    repeated_pvalue::Vector{Float64}
    decision::Symbol
    stop_look::Int
    stopped::Bool
    pvalue_adjusted::Union{Nothing,Float64}
    estimate_median_unbiased::Union{Nothing,Float64}
    ci_adjusted::Union{Nothing,NTuple{2,Float64}}
    level::Float64
    nobs::Int
end

function StatsAPI.coef(r::GroupSequentialAnalysis)
    r.estimate_median_unbiased === nothing && return [r.estimates[r.stop_look]]
    return [r.estimate_median_unbiased]
end
StatsAPI.vcov(r::GroupSequentialAnalysis) = fill(r.std_errors[r.stop_look]^2, 1, 1)
StatsAPI.coefnames(r::GroupSequentialAnalysis) = ["θ"]
StatsAPI.nobs(r::GroupSequentialAnalysis) = r.nobs
estimand(::GroupSequentialAnalysis) = "treatment effect θ"
method_name(r::GroupSequentialAnalysis) =
    "Group-sequential analysis (" * _seq_sf_name(r.design.efficacy) * ")"

function StatsAPI.confint(r::GroupSequentialAnalysis; level::Real=r.level)
    if r.ci_adjusted !== nothing
        isapprox(level, r.level; atol=1e-12) &&
            return [r.ci_adjusted[1] r.ci_adjusted[2]]
        lo, hi = _seq_gs_adjusted_ci(r.design, r.information, r.efficacy_z,
                                     r.stop_look, r.z[r.stop_look], level)
        return [lo hi]
    end
    rl = 1 - (r.design.sided == 1 ? 2 : 1) * r.design.alpha
    isapprox(level, rl; atol=1e-12) ||
        throw(ArgumentError("repeated confidence intervals are tied to the design " *
                            "level $(rl); got level = $level"))
    return [r.rci_lower[r.stop_look] r.rci_upper[r.stop_look]]
end

pvalues(r::GroupSequentialAnalysis) =
    [r.pvalue_adjusted === nothing ? r.repeated_pvalue[r.stop_look] : r.pvalue_adjusted]

# Probability, under effect θ, of an outcome at least as extreme as stopping at
# look `ks` with statistic `zs` in the stagewise ordering (upper direction).
function _seq_gs_stagewise(theta::Float64, info, b, ks::Int, zs::Float64, sided::Int)
    s = _seq_gs_start()
    p = 0.0
    for k in 1:ks
        ik = float(info[k])
        if k < ks
            p += _seq_gs_up(s, theta, ik, b[k])
            s = _seq_gs_advance(s, theta, ik, sided == 2 ? -b[k] : -Inf, b[k])
        else
            p += _seq_gs_up(s, theta, ik, zs)
        end
    end
    return p
end

# θ with stagewise probability `target` (increasing in θ).
function _seq_gs_invert(info, b, ks, zs, sided, target)
    se = 1 / sqrt(info[ks])
    center = zs * se
    f(th) = _seq_gs_stagewise(th, info, b, ks, zs, sided) - target
    lo = center - 5se
    hi = center + 5se
    it = 0
    while f(lo) > 0 && it < 50
        lo -= 5se; it += 1
    end
    it = 0
    while f(hi) < 0 && it < 50
        hi += 5se; it += 1
    end
    return _seq_root(f, lo, hi; tol=1e-10 * se)
end

function _seq_gs_adjusted_ci(d::GroupSequentialDesign, info, b, ks, zs, level)
    (0 < level < 1) || throw(ArgumentError("level must be in (0, 1), got $level"))
    q = (1 - level) / 2
    return (_seq_gs_invert(info, b, ks, zs, d.sided, q),
            _seq_gs_invert(info, b, ks, zs, d.sided, 1 - q))
end

# Efficacy bounds at the observed looks, for total level `alpha` of the design's rule.
function _seq_gs_bounds_at(d::GroupSequentialDesign, t, info, alpha)
    a_side = d.sided == 2 ? alpha / 2 : alpha
    d.efficacy === :haybittle_peto && length(t) < d.k &&
        return fill(d.hp_bound, length(t))
    if d.efficacy === :pocock || d.efficacy === :obrien_fleming
        # classical boundaries: the planned shape at the observed timing
        K = d.k
        tt = copy(d.timing)
        tt[1:length(t)] .= t
        ii = copy(tt)
        ii[1:length(t)] .= info ./ info[end] .* t[end]
        return _seq_gs_efficacy(d.efficacy, tt, ii, a_side, d.sided;
                                hp_bound=d.hp_bound)[1:length(t)]
    end
    lower = d.binding ? d.futility_z[1:length(t)] : nothing
    return _seq_gs_efficacy(d.efficacy, t, info, a_side, d.sided; hp_bound=d.hp_bound,
                            lower=lower)
end

"""
    gs_analysis(design, estimates, std_errors; information_fraction=nothing,
                level=nothing, n=nothing) -> GroupSequentialAnalysis

Analyse a group-sequential trial after the looks performed so far: boundaries at the
observed information, stopping decision, repeated confidence intervals and, after
stopping, design-adjusted estimates, confidence intervals and p-values.

At each look ``k`` the user supplies the cumulative-data estimate ``\\hat\\theta_k`` and
its standard error; the standardized statistic is
``Z_k = \\hat\\theta_k/\\mathrm{se}_k`` and the information ``I_k = 1/\\mathrm{se}_k^2``.
The joint distribution of ``(Z_1, \\ldots, Z_K)`` is taken to be the canonical one,
with correlations ``\\sqrt{I_j/I_k}`` computed from the *observed* information
(Jennison and Turnbull 2000, ch. 3). The efficacy boundaries are recomputed at the
observed looks. For a spending design, look ``k`` spends
``f(t_k) - f(t_{k-1})``, where ``t_k`` is the information fraction (Lan and DeMets
1983); at the final planned look (`length(estimates) == design.k`) ``t_K`` is set to
one, so that all remaining ``\\alpha`` is spent. Classical Pocock and O'Brien–Fleming
boundaries keep their planned shape at the observed timing, and futility boundaries
are the planned, non-binding ones.

**Information fractions and over- or under-running information.** By default the
spending fractions are the *planned* fractions `design.timing`, whatever information
was actually observed; only the correlations use the observed information. To spend
according to the information actually accrued, supply `information_fraction`
``t_k = I_k / I_{\\max}``, where ``I_{\\max}`` is the maximum information fixed at the
design stage (the fixed-sample information times `design.inflation`); ``I_{\\max}``
must be pre-specified, since a maximum information derived from the interim results
would make the boundaries data-dependent. Because any pre-specified sequence of
fractions yields a valid spending test, both choices control the type-I error provided
the timing of the looks does not depend on the observed effect (Lan and DeMets 1989);
they differ in how ``\\alpha`` is distributed. When the information at the final look
falls short of ``I_{\\max}`` (under-running), the remaining ``\\alpha`` is still
spent and the type-I error is preserved, but the power is lower than planned. When
information exceeds ``I_{\\max}`` before the last planned look (over-running), cap
the fraction at one: that look then spends all remaining ``\\alpha`` and acts as the
final analysis, and no further looks can be analysed; its decision is reported as
`:continue` unless a boundary is crossed, because the analysis is not the `k`-th
planned look.

**Repeated confidence intervals and p-values.** For every look the result reports
the RCI ``\\hat\\theta_k \\pm c_k\\,\\mathrm{se}_k``, with ``c_k`` the efficacy
boundary, and the repeated p-value (the smallest ``\\alpha`` at which some look up to
``k`` would have crossed). Under the canonical distribution the RCIs have
simultaneous coverage ``1 - 2\\alpha`` (one-sided designs) or ``1 - \\alpha``
(two-sided designs) over all looks, whatever stopping rule is actually followed
(Jennison and Turnbull 1989); with a binding futility boundary the efficacy
boundaries are lowered and this coverage no longer holds exactly. The decision is
taken at the first look whose statistic crosses a boundary.

**Inference after stopping.** Once the trial has stopped (efficacy crossing, or the
final look reached without crossing), the stagewise ordering of the sample space
(Armitage 1957; Tsiatis, Rosner and Mehta 1984) gives the adjusted p-value, the
median-unbiased estimate (Kim 1989) and a `level` confidence interval that account for
the sequential design; non-binding futility boundaries are ignored in these
calculations, which is the conservative convention. Report the adjusted quantities
rather than the naive estimate and fixed-sample interval at the stopping look, which
overstate the effect after an early stop. All guarantees are exact under the
canonical normal distribution and approximate when the statistics are only
asymptotically normal.

# Arguments
- `design::GroupSequentialDesign`: the design fixed before the trial.
- `estimates::AbstractVector`: effect estimate at each look performed, on cumulative
  data, oriented so that positive values favour the alternative.
- `std_errors::AbstractVector`: standard error of each estimate; the implied
  information ``1/\\mathrm{se}_k^2`` must increase strictly across looks.

# Keywords
- `information_fraction = nothing`: observed information fractions
  ``t_k = I_k/I_{\\max}`` in ``(0, 1]``, strictly increasing; `nothing` uses the
  planned `design.timing` (see above).
- `level = nothing`: level of the adjusted confidence interval; defaults to
  ``1 - 2\\alpha`` for one-sided designs and ``1 - \\alpha`` for two-sided designs,
  the level of the RCIs.
- `n = nothing`: cumulative sample sizes per look, used only for `nobs`.

# Returns
- [`GroupSequentialAnalysis`](@ref).

# Examples
```julia
using DrSnow
d = gs_design(; k=3, alpha=0.025, beta=0.1)
a = gs_analysis(d, [0.21, 0.33], [0.15, 0.105])          # after two looks
a.decision, confint(a)                                     # RCI at look 2
# information accrued faster than planned: spend by the observed fractions
a2 = gs_analysis(d, [0.21, 0.33], [0.15, 0.105]; information_fraction=[0.40, 0.80])
a2.efficacy_z
```

# References
- Jennison, C., & Turnbull, B. W. (1989). Interim analyses: The repeated confidence
  interval approach. *Journal of the Royal Statistical Society: Series B*, 51(3),
  305–334.
- Tsiatis, A. A., Rosner, G. L., & Mehta, C. R. (1984). Exact confidence intervals
  following a group sequential test. *Biometrics*, 40(3), 797–803.
- Armitage, P. (1957). Restricted sequential procedures. *Biometrika*, 44(1–2),
  9–26.
- Kim, K. (1989). Point estimation following group sequential tests. *Biometrics*,
  45(2), 613–617.
- Lan, K. K. G., & DeMets, D. L. (1983). Discrete sequential boundaries for clinical
  trials. *Biometrika*, 70(3), 659–663.
- Lan, K. K. G., & DeMets, D. L. (1989). Changing frequency of interim analysis in
  sequential monitoring. *Biometrics*, 45(3), 1017–1020.
- Jennison, C., & Turnbull, B. W. (2000). *Group Sequential Methods with Applications
  to Clinical Trials*. Chapman & Hall/CRC.
- Wassmer, G., & Brannath, W. (2016). *Group Sequential and Confirmatory Adaptive
  Designs in Clinical Trials*. Springer.
"""
function gs_analysis(design::GroupSequentialDesign, estimates::AbstractVector{<:Real},
                     std_errors::AbstractVector{<:Real}; information_fraction=nothing,
                     level=nothing, n=nothing)
    K = length(estimates)
    K >= 1 || throw(ArgumentError("need at least one look"))
    length(std_errors) == K ||
        throw(DimensionMismatch("estimates and std_errors must have equal length"))
    K <= design.k || throw(ArgumentError("more looks ($K) than planned " *
                                         "($(design.k)); extend the design"))
    all(s -> s > 0 && isfinite(s), std_errors) ||
        throw(ArgumentError("standard errors must be positive and finite"))
    all(isfinite, estimates) || throw(ArgumentError("estimates must be finite"))
    est = float.(collect(estimates))
    se = float.(collect(std_errors))
    info = 1 ./ se .^ 2
    issorted(info; lt=<=) ||
        throw(ArgumentError("information (1/se²) must increase strictly across looks"))
    t = information_fraction === nothing ? design.timing[1:K] :
        float.(collect(information_fraction))
    length(t) == K || throw(DimensionMismatch("information_fraction must have one " *
                                              "entry per look"))
    (all(x -> 0 < x <= 1, t) && issorted(t; lt=<=)) ||
        throw(ArgumentError("information fractions must increase within (0, 1]"))
    K == design.k && (t[end] = 1.0)
    lv = level === nothing ? 1 - (design.sided == 1 ? 2 : 1) * design.alpha :
         float(level)
    (0 < lv < 1) || throw(ArgumentError("level must be in (0, 1), got $lv"))
    nn = n === nothing ? 0 : Int(n[end])

    b = _seq_gs_bounds_at(design, t, info, design.alpha)
    a = design.sided == 2 ? -b : copy(design.futility_z[1:K])
    K < design.k && design.sided == 1 && (a[end] = min(a[end], b[end]))
    z = est ./ se
    rlo = est .- b .* se
    rhi = est .+ b .* se

    decision = :continue
    ks = K
    for k in 1:K
        if (design.sided == 1 ? z[k] >= b[k] : abs(z[k]) >= b[k])
            decision = :efficacy; ks = k; break
        elseif design.sided == 1 && z[k] <= a[k] && k < design.k
            decision = :futility; ks = k; break
        end
    end
    decision === :continue && K == design.k && (decision = :final_no_rejection)
    stopped = decision in (:efficacy, :final_no_rejection)

    rep = _seq_gs_repeated_pvalues(design, t, info, z)
    padj = nothing; mue = nothing; ci = nothing
    if stopped
        bb = b[1:ks]
        zs = z[ks]
        pu = _seq_gs_stagewise(0.0, info, bb, ks, zs, design.sided)
        if design.sided == 2
            # lower-direction ordering, by symmetry of the bounds under θ = 0
            pl = _seq_gs_stagewise(0.0, info, bb, ks, -zs, 2)
            padj = min(1.0, 2 * min(pu, pl))
        else
            padj = pu
        end
        mue = _seq_gs_invert(info, bb, ks, zs, design.sided, 0.5)
        ci = _seq_gs_adjusted_ci(design, info, bb, ks, zs, lv)
    end
    return GroupSequentialAnalysis(design, est, se, z, info, t, b, a, rlo, rhi, rep,
                                   decision, ks, stopped, padj, mue, ci, lv, nn)
end

# Repeated p-value after each look: smallest α at which some look ≤ k crosses.
function _seq_gs_repeated_pvalues(d::GroupSequentialDesign, t, info, z)
    K = length(t)
    pj = zeros(K)
    amax = d.sided == 1 ? 0.5 : 1.0
    lmin, lmax = log(1e-15), log(amax * (1 - 1e-9))
    for j in 1:K
        zj = d.sided == 1 ? z[j] : abs(z[j])
        if d.efficacy === :haybittle_peto && j < d.k
            pj[j] = 1.0         # the fixed interim bound does not index a level
        elseif d.efficacy === :pocock || d.efficacy === :obrien_fleming
            pj[j] = _seq_gs_wt_level(d, t, info, j, zj)
        else
            g(la) = _seq_gs_bounds_at(d, t[1:j], info[1:j], exp(la))[j] - zj
            pj[j] = g(lmax) > 0 ? 1.0 : g(lmin) <= 0 ? exp(lmin) :
                    exp(_seq_root(g, lmin, lmax; tol=1e-10))
        end
    end
    return accumulate(min, pj)
end

# Wang–Tsiatis designs: b_k(α) = C(α) · shape_k, so look j crosses at level α iff
# z_j / shape_j ≥ C(α); the repeated p-value is the level whose constant is
# z_j / shape_j, i.e. the H₀ crossing probability of the scaled boundary.
function _seq_gs_wt_level(d::GroupSequentialDesign, t, info, j, zj)
    zj <= 0 && return 1.0
    tt = copy(d.timing); tt[1:length(t)] .= t
    ii = copy(tt); ii[1:length(t)] .= info ./ info[end] .* t[end]
    b = _seq_gs_efficacy(d.efficacy, tt, ii, d.sided == 2 ? d.alpha / 2 : d.alpha,
                         d.sided)
    bb = (zj / b[j]) .* b
    a = d.sided == 2 ? -bb : fill(-Inf, length(bb))
    up, lo = _seq_gs_probs(a, bb, ii, 0.0)
    return min(1.0, d.sided == 2 ? sum(up) + sum(lo) : sum(up))
end

function Base.show(io::IO, ::MIME"text/plain", r::GroupSequentialAnalysis)
    d = r.design
    println(io, method_name(r))
    println(io, "Looks analysed: $(length(r.z)) of $(d.k) planned")
    println(io, rpad("look", 6), rpad("t", 8), rpad("estimate", 11), rpad("z", 9),
            rpad("efficacy z", 12), d.sided == 1 && d.futility !== nothing ?
                                    rpad("futility z", 12) : "",
            "RCI (", round(100 * (1 - (d.sided == 1 ? 2 : 1) * d.alpha); digits=1),
            "%)")
    for k in eachindex(r.z)
        print(io, rpad(k, 6), rpad(@sprintf("%.3f", r.timing[k]), 8),
              rpad(@sprintf("%.4g", r.estimates[k]), 11),
              rpad(@sprintf("%.3f", r.z[k]), 9),
              rpad(@sprintf("%.4f", r.efficacy_z[k]), 12))
        d.sided == 1 && d.futility !== nothing &&
            print(io, rpad(@sprintf("%.4f", r.futility_z[k]), 12))
        @printf(io, "[%.4g, %.4g]\n", r.rci_lower[k], r.rci_upper[k])
    end
    msg = r.decision === :efficacy ? "efficacy bound crossed at look $(r.stop_look)" :
          r.decision === :futility ?
          "futility bound crossed at look $(r.stop_look) (non-binding: stopping for " *
          "futility is allowed, continuing does not inflate the type-I error)" :
          r.decision === :continue ? "no bound crossed; continue to the next look" :
          "final look reached without crossing the efficacy bound"
    println(io, "Decision: ", msg)
    @printf(io, "Repeated p-value: %.4g\n", r.repeated_pvalue[end])
    if r.pvalue_adjusted !== nothing
        lv = round(100 * r.level; digits=1)
        @printf(io, "Stagewise-ordering inference: p = %.4g, ", r.pvalue_adjusted)
        @printf(io, "median-unbiased estimate %.4g, %g%% CI [%.4g, %.4g]\n",
                r.estimate_median_unbiased, lv, r.ci_adjusted[1], r.ci_adjusted[2])
    end
end
