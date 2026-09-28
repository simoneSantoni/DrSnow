# First-stage strength: conventional / robust F, partial R², Cragg–Donald,
# Kleibergen–Paap, Sanderson–Windmeijer, Olea–Pflueger effective F with critical
# values, Stock–Yogo critical values, and the Lee–McCrary–Moreira–Porter tF interval.

"""
    first_stage_diagnostics(r::IVEstimate) -> WeakIVDiagnostics
    first_stage_diagnostics(data, treatment, instrument; outcome=nothing,
                            covariates=Symbol[], fe=Symbol[], weights=nothing,
                            cluster=nothing, vcov=nothing,
                            drop_singletons=true) -> WeakIVDiagnostics

First-stage strength and weak-instrument diagnostics of a linear IV specification.

Relevance of the instruments is the only IV assumption that the data can confirm, but
mere statistical significance of the first stage is not enough: when the
concentration parameter is small relative to the number of instruments, the 2SLS
estimator is biased towards OLS and its t-statistic is far from normal (Staiger and
Stock 1997; Stock and Yogo 2005). This function reports, for each endogenous
regressor, the first-stage coefficients, the Wald F computed with the covariance
estimator of the model (robust or clustered when requested) and the conventional F,
the partial ``R^2`` and the conditional F of Sanderson and Windmeijer (2016); and,
jointly, the Cragg–Donald and the Kleibergen–Paap (2006) rk Wald F statistics, which
test the rank condition for the whole coefficient vector.

With one endogenous regressor it also reports the effective F of Montiel Olea and
Pflueger (2013), ``F_{\\text{eff}} = \\hat\\pi'Q\\hat\\pi / \\operatorname{tr}(\\hat V Q)``
with ``Q = Z'Z`` of the partialled instruments and ``\\hat V`` the robust covariance of
the first-stage coefficients, together with its simplified critical values. The
effective F tests the null that the Nagar approximation to the 2SLS bias exceeds a
fraction τ of a worst-case benchmark, and it remains valid under heteroskedasticity,
clustering and serial correlation; it is the pre-test recommended by Andrews, Stock
and Sun (2019). The simplified critical values are conservative: they bound the
generalized critical values from above. The Stock and Yogo (2005) critical values
are reported only as a reference for the homoskedastic Cragg–Donald F and are not
valid for robust or clustered statistics; the popular rule of thumb F > 10 is an
approximation to their 10% relative-bias threshold and is likewise a homoskedastic
benchmark. With one instrument and one endogenous regressor, Lee, McCrary, Moreira and
Porter (2022) show that even F > 10 leaves the 5% t-test substantially oversized and
that the F needed for a correctly sized t-test is about 104.7.

A small first-stage F calls for weak-identification-robust inference, not for
discarding the specification: selecting specifications on the first-stage F is itself
a source of size distortion (Andrews, Stock and Sun 2019). Use
[`weak_iv_confidence_set`](@ref) (Anderson–Rubin) or [`tf_confint`](@ref) and report
the diagnostics alongside. A strong first stage says nothing about exclusion or
independence.

# Arguments
- `r::IVEstimate`: a fitted model; its stored diagnostics are returned.
- `data::AbstractDataFrame`, `treatment`, `instrument`: alternatively, the data, the
  endogenous regressor(s) and the excluded instrument(s), as in
  [`iv_regression`](@ref).

# Keywords
- `outcome::Union{Nothing,Symbol}`: the outcome of the IV model (default `nothing`).
  First-stage statistics do not depend on it, so when it is omitted a placeholder
  outcome is used; supplying it only matters for the estimation sample, since rows
  with a missing outcome are then dropped.
- `covariates`, `fe`, `weights`, `cluster`, `vcov`, `drop_singletons`: as in
  [`iv_regression`](@ref) (default no controls, HC1 covariance). The covariance
  choice determines the robust F statistics.

# Returns
- A [`WeakIVDiagnostics`](@ref); `d.first_stage[j].F` is the robust F of the `j`-th
  endogenous regressor, `d.effective_F` and `d.op_critical_values` the Montiel
  Olea–Pflueger statistic and critical values, `d.stock_yogo` the Stock–Yogo table.

# Examples
```julia
using DrSnow, DataFrames, StableRNGs
rng = StableRNG(2)
n = 1_000
z1, z2 = randn(rng, n), randn(rng, n)
d = 0.3 .* z1 .+ 0.1 .* z2 .+ randn(rng, n) .* (1 .+ abs.(z1))     # heteroskedastic
df = DataFrame(d=d, z1=z1, z2=z2)
fs = first_stage_diagnostics(df, :d, [:z1, :z2])
fs.effective_F, fs.op_critical_values.tau_10
```

# References
- Staiger, D., & Stock, J. H. (1997). Instrumental variables regression with weak
  instruments. *Econometrica*, 65(3), 557–586.
- Stock, J. H., & Yogo, M. (2005). Testing for weak instruments in linear IV
  regression. In D. W. K. Andrews & J. H. Stock (Eds.), *Identification and
  Inference for Econometric Models: Essays in Honor of Thomas Rothenberg*
  (pp. 80–108). Cambridge University Press.
- Kleibergen, F., & Paap, R. (2006). Generalized reduced rank tests using the
  singular value decomposition. *Journal of Econometrics*, 133(1), 97–126.
- Montiel Olea, J. L., & Pflueger, C. (2013). A robust test for weak instruments.
  *Journal of Business & Economic Statistics*, 31(3), 358–369.
- Sanderson, E., & Windmeijer, F. (2016). A weak instrument F-test in linear IV
  models with multiple endogenous variables. *Journal of Econometrics*, 190(2),
  212–221.
- Andrews, I., Stock, J. H., & Sun, L. (2019). Weak instruments in instrumental
  variables regression: Theory and practice. *Annual Review of Economics*, 11,
  727–753.
- Lee, D. S., McCrary, J., Moreira, M. J., & Porter, J. (2022). Valid t-ratio
  inference for IV. *American Economic Review*, 112(10), 3260–3290.
"""
first_stage_diagnostics(r::IVEstimate) = r.first_stage

function first_stage_diagnostics(data::AbstractDataFrame, treatment, instrument;
                                 outcome::Union{Nothing,Symbol}=nothing, kwargs...)
    if outcome === nothing
        tmp = copy(data; copycols=false)
        # any non-degenerate outcome: first-stage statistics do not depend on it
        tmp[!, :__iv_placeholder_outcome__] = collect(1.0:nrow(tmp))
        return iv_regression(tmp, :__iv_placeholder_outcome__, treatment, instrument;
                             kwargs...).first_stage
    end
    return iv_regression(data, outcome, treatment, instrument; kwargs...).first_stage
end

function _iv_weak_iv_diagnostics(des::_IVDesign, endo::Vector{Symbol},
                                 inst::Vector{Symbol}, F_kp::Real)
    Z, D = des.Z, des.D
    n, k = size(Z)
    p = size(D, 2)
    B, V, E = _iv_ols(des, D, Z)            # first stages, joint covariance
    dof_h = _iv_resid_dof(des, k)
    ref = _iv_ref_dof(des, k)
    ZtZ = Symmetric(Z' * Z)
    results = FirstStageResult[]
    for j in 1:p
        blk = ((j - 1) * k + 1):(j * k)
        b = B[:, j]
        Vj = V[blk, blk]
        F = _iv_wald_F(b, Vj)
        rss = sum(abs2, E[:, j])
        ess = sum(abs2, D[:, j]) - rss
        Fh = (ess / k) / (rss / dof_h)
        pr2 = ess / sum(abs2, D[:, j])
        sw = p == 1 ? Fh : _iv_sanderson_windmeijer(des, j)
        push!(results, FirstStageResult(endo[j], copy(inst), b, Vj, F,
                                        ccdf(FDist(k, ref), F), Fh, (k, ref), pr2, sw))
    end
    # Cragg–Donald minimum eigenvalue statistic
    Σvv = Symmetric((E' * E) ./ dof_h)
    PZD = Z * (ZtZ \ (Z' * D))
    Cv = cholesky(Σvv; check=false)
    cd = if issuccess(Cv)
        L = Cv.L
        minimum(eigvals(Symmetric(L \ ((D' * PZD) / L')))) / k
    else
        Inf     # some combination of first-stage errors has zero variance
    end

    effF, opcv = nothing, nothing
    if p == 1
        effF, opcv = _iv_effective_F(B[:, 1], V, Matrix(ZtZ))
    end
    sy = (p == 1 && k <= 10) ? _iv_stock_yogo(k) : nothing
    return WeakIVDiagnostics(results, cd, float(F_kp), effF, opcv, sy, k, p,
                             _iv_vcov_label(des))
end

"""Sanderson–Windmeijer conditional F for endogenous regressor `j` (homoskedastic)."""
function _iv_sanderson_windmeijer(des::_IVDesign, j::Int)
    D, Z = des.D, des.Z
    k, p = size(Z, 2), size(D, 2)
    others = [i for i in 1:p if i != j]
    Do = D[:, others]
    Dhat = Z * (Z \ Do)
    δ = (Dhat' * Do) \ (Dhat' * D[:, j])
    ε = D[:, j] - Do * δ
    Pε = Z * (Z \ ε)
    num = dot(ε, Pε) / (k - p + 1)
    den = (sum(abs2, ε) - dot(ε, Pε)) / _iv_resid_dof(des, k)
    return num / den
end

"""
Effective F (Montiel Olea & Pflueger 2013) and simplified critical values.

`π̂` first-stage coefficients, `V` their covariance (any type), `Q = Z'Z` of the
partialled instruments. `F_eff = π̂'Qπ̂ / tr(V Q)`. Critical values: with
`S = Q^{1/2} V Q^{1/2}` and `x = 1/τ`,
`K_eff = tr(S)²(1 + 2x) / (tr(S'S) + 2x tr(S) λ_max(S))` and
`c(τ) = χ²_{K_eff}(x K_eff)⁻¹(0.95) / K_eff` (non-central χ² quantile).
"""
function _iv_effective_F(π::AbstractVector, V::AbstractMatrix, Q::AbstractMatrix)
    Feff = dot(π, Q * π) / tr(V * Q)
    Qh = sqrt(Symmetric(Q))
    S = Symmetric(Qh * V * Qh)
    trS = tr(S)
    trSS = sum(abs2, S)
    λmax = maximum(eigvals(S))
    cv(τ) = begin
        x = 1 / τ
        Keff = trS^2 * (1 + 2x) / (trSS + 2x * trS * λmax)
        (quantile(NoncentralChisq(Keff, x * Keff), 0.95) / Keff, Keff)
    end
    c5, _ = cv(0.05)
    c10, Keff = cv(0.10)
    c20, _ = cv(0.20)
    c30, _ = cv(0.30)
    return Feff, (tau_5=c5, tau_10=c10, tau_20=c20, tau_30=c30, K_eff=Keff)
end

# Stock & Yogo (2005), Tables 5.1 (TSLS relative bias; K ≥ 3) and 5.2 (TSLS size of
# a nominal 5% Wald test), one endogenous regressor, K = number of instruments.
const _IV_SY_SIZE = Dict(
    1 => (16.38, 8.96, 6.66, 5.53), 2 => (19.93, 11.59, 8.75, 7.25),
    3 => (22.30, 12.83, 9.54, 7.80), 4 => (24.58, 13.96, 10.26, 8.31),
    5 => (26.87, 15.09, 10.98, 8.84), 6 => (29.18, 16.23, 11.72, 9.38),
    7 => (31.50, 17.38, 12.48, 9.93), 8 => (33.84, 18.54, 13.24, 10.50),
    9 => (36.19, 19.71, 14.01, 11.07), 10 => (38.54, 20.88, 14.78, 11.65))
const _IV_SY_BIAS = Dict(
    3 => (13.91, 9.08, 6.46, 5.39), 4 => (16.85, 10.27, 6.71, 5.34),
    5 => (18.37, 10.83, 6.77, 5.25), 6 => (19.28, 11.12, 6.76, 5.15),
    7 => (19.86, 11.29, 6.73, 5.07), 8 => (20.25, 11.39, 6.69, 4.99),
    9 => (20.53, 11.46, 6.65, 4.92), 10 => (20.74, 11.49, 6.61, 4.86))

function _iv_stock_yogo(k::Int)
    s = _IV_SY_SIZE[k]
    b = get(_IV_SY_BIAS, k, nothing)
    size_nt = (size_10=s[1], size_15=s[2], size_20=s[3], size_25=s[4])
    bias_nt = b === nothing ? nothing :
              (bias_5=b[1], bias_10=b[2], bias_20=b[3], bias_30=b[4])
    return (size=size_nt, bias=bias_nt,
            note="Stock–Yogo critical values apply to the Cragg–Donald F under " *
                 "homoskedastic, serially uncorrelated errors only.")
end

function Base.show(io::IO, ::MIME"text/plain", d::WeakIVDiagnostics)
    println(io, "Weak-instrument diagnostics (", d.n_endogenous, " endogenous, ",
            d.n_instruments, " instrument", d.n_instruments == 1 ? "" : "s", ")")
    println(io, "Covariance: ", d.vcov_type)
    for s in d.first_stage
        _iv_printf(io, "  %s: F = %.3f (p = %.4g), conventional F = %.3f, partial R² = " *
                "%.4f", s.endogenous, s.F, s.F_pvalue, s.F_homoskedastic, s.partial_r2)
        d.n_endogenous > 1 && @printf(io, ", Sanderson–Windmeijer F = %.3f",
                                      s.sanderson_windmeijer_F)
        println(io)
    end
    @printf(io, "Cragg–Donald F = %.3f; Kleibergen–Paap rk Wald F = %.3f\n",
            d.cragg_donald_F, d.kleibergen_paap_F)
    if d.effective_F !== nothing
        c = d.op_critical_values
        _iv_printf(io, "Olea–Pflueger effective F = %.3f; simplified 5%% critical values " *
                "(τ = 5/10/20/30%%): %.2f / %.2f / %.2f / %.2f\n", d.effective_F,
                c.tau_5, c.tau_10, c.tau_20, c.tau_30)
    end
    if d.stock_yogo !== nothing
        s = d.stock_yogo.size
        _iv_printf(io, "Stock–Yogo (iid only) size critical values 10/15/20/25%%: " *
                "%.2f / %.2f / %.2f / %.2f\n", s.size_10, s.size_15, s.size_20,
                s.size_25)
    end
end

# ---------------------------------------------------------------------------
# tF (Lee, McCrary, Moreira & Porter 2022)
# ---------------------------------------------------------------------------

# LMMP (2022), Table 3: 5%-level tF critical values c(F) on a grid of sqrt(F) from
# 2.0 to 10.3 in steps of 0.1 (F from 4 to 106.09). Above the grid c = 1.96.
const _IV_TF_SQRTF = collect(2.0:0.1:10.3)
const _IV_TF_CRIT = [
    18.66, 9.74, 7.37, 6.18, 5.43, 4.92, 4.54, 4.25, 4.01, 3.82, 3.65, 3.51, 3.39,
    3.29, 3.19, 3.11, 3.03, 2.97, 2.91, 2.85, 2.80, 2.75, 2.71, 2.67, 2.63, 2.60,
    2.57, 2.54, 2.51, 2.48, 2.46, 2.43, 2.41, 2.39, 2.37, 2.35, 2.33, 2.32, 2.30,
    2.29, 2.27, 2.26, 2.24, 2.23, 2.22, 2.21, 2.20, 2.19, 2.17, 2.16, 2.16, 2.15,
    2.14, 2.13, 2.12, 2.11, 2.10, 2.10, 2.09, 2.08, 2.08, 2.07, 2.06, 2.06, 2.05,
    2.04, 2.04, 2.03, 2.03, 2.02, 2.02, 2.01, 2.01, 2.00, 2.00, 1.99, 1.99, 1.99,
    1.98, 1.98, 1.97, 1.97, 1.97, 1.96]

"""5%-level tF critical value (linear interpolation in sqrt(F)); `Inf` for F ≤ 4."""
function _iv_tf_critical_value(F::Real)
    F > 0 || return Inf
    s = sqrt(F)
    s <= _IV_TF_SQRTF[1] && return s == _IV_TF_SQRTF[1] ? _IV_TF_CRIT[1] : Inf
    s >= _IV_TF_SQRTF[end] && return 1.96
    j = searchsortedlast(_IV_TF_SQRTF, s)
    t = (s - _IV_TF_SQRTF[j]) / (_IV_TF_SQRTF[j + 1] - _IV_TF_SQRTF[j])
    return (1 - t) * _IV_TF_CRIT[j] + t * _IV_TF_CRIT[j + 1]
end

"""
    tf_confint(r::IVEstimate; level=0.95) -> NamedTuple

The tF confidence interval of Lee, McCrary, Moreira and Porter (2022) for a
just-identified IV model.

In a model with one endogenous regressor and one instrument, the usual practice of
reporting the 2SLS t-ratio interval ``\\hat\\beta \\pm 1.96\\,\\widehat{\\text{se}}`` after
checking that the first-stage F exceeds 10 does not control size: the actual rejection
rate of the nominal 5% t-test can exceed 5% substantially for F between 10 and about
100, because the t-ratio and the first-stage F are correlated when the degree of
endogeneity is high. Lee et al. (2022) derive a smooth adjustment: the interval
``\\hat\\beta \\pm c(F)\\,\\widehat{\\text{se}}``, with a critical value ``c(F)`` that
decreases in the first-stage F, has correct 5% size for every degree of endogeneity.
The first-stage F must be computed with the same covariance estimator as the 2SLS
standard error (robust or clustered), which is how it is stored in the
[`IVEstimate`](@ref).

The function interpolates ``c(F)`` linearly in ``\\sqrt F`` from Table 3 of Lee et al.
(2022). For F above 104.7 the critical value equals 1.96 and the tF interval coincides
with the conventional one. For F at or below 4 the interval is taken to be the whole
real line: Lee et al. show that the 95% tF interval is unbounded when F < 3.84, and
values in (3.84, 4] lie below the published table and are treated conservatively.
The procedure is an asymptotic approximation under the usual first-order theory with a
fixed number of instruments.

The tF interval is always an interval, which makes it easy to report, but it is not
the most powerful robust procedure; the Anderson–Rubin set from
[`weak_iv_confidence_set`](@ref) is efficient in the just-identified model (Andrews,
Stock and Sun 2019) and may be unbounded when identification is weak, which is
informative. Lee et al. (2022) recommend reporting the tF interval, or the AR set,
in place of the F > 10 rule.

# Arguments
- `r::IVEstimate`: a just-identified IV estimate (one endogenous regressor, one
  instrument), typically from [`late_2sls`](@ref).

# Keywords
- `level::Real`: confidence level; only `0.95` is supported (default), because the
  published critical values are for 5% tests.

# Returns
- A `NamedTuple` `(estimate, se, F, critical_value, lower, upper)`: the 2SLS
  estimate and standard error, the first-stage F, the adjusted critical value
  ``c(F)`` (`Inf` when the interval is unbounded) and the interval endpoints.

# Examples
```julia
using DrSnow, DataFrames, StableRNGs
rng = StableRNG(3)
n = 1_000
z = randn(rng, n)
v = randn(rng, n)
d = 0.15 .* z .+ v
y = 0.5 .* d .+ 0.8 .* v .+ 0.6 .* randn(rng, n)
df = DataFrame(y=y, d=d, z=z)
r = late_2sls(df, :y, :d, :z)
tf_confint(r)                # wider than confint(r) when F is moderate
```

# References
- Lee, D. S., McCrary, J., Moreira, M. J., & Porter, J. (2022). Valid t-ratio
  inference for IV. *American Economic Review*, 112(10), 3260–3290.
- Staiger, D., & Stock, J. H. (1997). Instrumental variables regression with weak
  instruments. *Econometrica*, 65(3), 557–586.
- Andrews, I., Stock, J. H., & Sun, L. (2019). Weak instruments in instrumental
  variables regression: Theory and practice. *Annual Review of Economics*, 11,
  727–753.
- Keane, M., & Neal, T. (2023). Instrument strength in IV estimation and inference: A
  guide to theory and practice. *Journal of Econometrics*, 235(2), 1625–1653.
"""
function tf_confint(r::IVEstimate; level::Real=0.95)
    (length(r.endogenous) == 1 && length(r.instruments) == 1) ||
        throw(ArgumentError("tf_confint requires a just-identified model (one " *
                            "endogenous regressor and one instrument)"))
    isapprox(level, 0.95) ||
        throw(ArgumentError("tf_confint supports only level = 0.95 (LMMP Table 3)"))
    F = r.first_stage.first_stage[1].F
    c = _iv_tf_critical_value(F)
    b = r.coef[1]
    se = sqrt(r.vcov[1, 1])
    lo, hi = isfinite(c) ? (b - c * se, b + c * se) : (-Inf, Inf)
    return (estimate=b, se=se, F=F, critical_value=c, lower=lo, upper=hi)
end
