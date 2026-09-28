# McCrary (2008) density discontinuity test, following `DCdensity` of the R package
# `rdd` (Dimmery, version 0.57) step by step: histogram with bin width
# `2 sd(x) n^{-1/2}`, bandwidth from global quartic fits on each side, and local linear
# (triangular kernel) smoothing of the histogram on each side of the cutoff.

# Weighted least-squares intercept of `y` on `[1 x]` (weights ≥ 0).
function _rd_wls_intercept(x::AbstractVector, y::AbstractVector, w::AbstractVector)
    keep = w .> 0
    count(keep) >= 2 || throw(ArgumentError(
        "rd_mccrary_test: fewer than two histogram bins with positive kernel weight on " *
        "one side of the cutoff; use a larger bandwidth"))
    sw = sqrt.(w[keep])
    X = hcat(sw, sw .* x[keep])
    F = qr(X, ColumnNorm())
    rank(X) == 2 || throw(ArgumentError(
        "rd_mccrary_test: local linear fit of the histogram is not identified"))
    return (F \ (sw .* y[keep]))[1]
end

# Global quartic fit of `y` on `x` (OLS): coefficients and residual variance.
function _rd_quartic_fit(x::AbstractVector, y::AbstractVector)
    n = length(x)
    n > 5 || throw(ArgumentError(
        "rd_mccrary_test: too few histogram bins on one side of the cutoff to select " *
        "the bandwidth; pass `bandwidth`"))
    X = _rd_vander(x, 4)
    beta = qr(X, ColumnNorm()) \ y
    res = y .- X * beta
    return beta, sum(abs2, res) / (n - 5)
end

"""
    rd_mccrary_test(data, running; cutoff=0.0, bin=nothing,
                    bandwidth=nothing) -> DiagnosticTest
    rd_mccrary_test(x::AbstractVector; cutoff=0.0, bin=nothing,
                    bandwidth=nothing) -> DiagnosticTest

McCrary (2008) test of continuity of the density of the running variable at the
cutoff, equivalent to `DCdensity` of the R package `rdd`.

The test addresses the same question as [`rd_density_test`](@ref): whether units sorted
themselves across the cutoff, which would show up as a jump in the density ``f`` of the
running variable at ``c``. McCrary's construction proceeds in two steps. First, the
running variable is binned into a histogram with bin width `bin` (default
``2\\,\\hat\\sigma_X n^{-1/2}``), with no bin straddling the cutoff. Second, the
normalised bin heights are smoothed separately on each side by local linear regression
(triangular kernel, bandwidth `bandwidth`) and extrapolated to the cutoff, giving
``\\hat f^-`` and ``\\hat f^+``. The statistic is the log difference
```math
\\hat\\theta = \\log \\hat f^+ - \\log \\hat f^-, \\qquad
\\widehat{\\text{se}}(\\hat\\theta) = \\sqrt{\\frac{1}{n h}\\,\\frac{24}{5}
    \\left(\\frac{1}{\\hat f^+} + \\frac{1}{\\hat f^-}\\right)},
```
referred to the standard normal distribution. The default bandwidth is McCrary's rule of
thumb. On each side it is ``3.348\\,[\\hat\\sigma^2 (\\text{range}) /
\\sum \\hat f''(b_j)^2]^{1/5}``, from a global quartic fit to the histogram, and the two
sides are averaged.

This is the original manipulation test. It relies on user-tuned binning and
smoothing, and its inference ignores smoothing bias. The local polynomial test of
Cattaneo, Jansson and Ma (2020) needs no pre-binning, selects its bandwidths from the
data and uses robust bias-corrected inference, so [`rd_density_test`](@ref) is
preferable in applications. Use this function to reproduce or compare with published
McCrary-type results. Like every density test, it cannot detect manipulation that
leaves the density continuous. A non-rejection does not show that there was no
manipulation, and a rejection does not by itself identify its direction or cause.

# Arguments
- `data::AbstractDataFrame` and `running::Symbol`, or a vector `x`: the running
  variable. Missing and `NaN` values are dropped.

# Keywords
- `cutoff::Real=0.0`: the RD threshold. It must lie strictly inside the range of the
  data.
- `bin::Union{Nothing,Real}=nothing`: histogram bin width (default
  ``2\\,\\hat\\sigma_X n^{-1/2}``). Narrower bins leave more of the smoothing to the
  local linear step.
- `bandwidth::Union{Nothing,Real}=nothing`: bandwidth of the local linear smoother
  (default: McCrary's rule of thumb). Results can be sensitive to it, so report the
  value used.

# Returns
- `DiagnosticTest` with statistic ``\\hat\\theta / \\widehat{\\text{se}}`` and its
  two-sided normal p-value. `details` holds `theta` (log difference in density heights),
  `se`, `f_left`, `f_right`, `bin`, `bandwidth`, `n`, and `histogram` (a `DataFrame` of
  bin midpoints `midpoint` and normalised heights `density`).

# Examples
```julia
using DrSnow, CSV, DataFrames
senate = CSV.read(pkgdir(DrSnow, "test", "validation", "rd", "senate.csv"), DataFrame)
t = rd_mccrary_test(senate, :margin)
t.details.theta, t.pvalue
rd_mccrary_test(senate.margin; bin=1, bandwidth=20)
```

# References
- McCrary, J. (2008). Manipulation of the running variable in the regression
  discontinuity design: A density test. *Journal of Econometrics*, 142(2), 698–714.
- Cattaneo, M. D., Jansson, M., & Ma, X. (2020). Simple local polynomial density
  estimators. *Journal of the American Statistical Association*, 115(531), 1449–1455.
- Cattaneo, M. D., Idrobo, N., & Titiunik, R. (2020). *A Practical Introduction to
  Regression Discontinuity Designs: Foundations*. Cambridge University Press.
"""
function rd_mccrary_test(x::AbstractVector; cutoff::Real=0.0,
                         bin::Union{Nothing,Real}=nothing,
                         bandwidth::Union{Nothing,Real}=nothing)
    ctx = "rd_mccrary_test"
    xs = Float64[v for v in x if !ismissing(v) && !(v isa AbstractFloat && isnan(v))]
    rn = length(xs)
    rn >= 3 || throw(ArgumentError("$ctx: need at least three observations"))
    c = Float64(cutoff)
    rsd = std(xs)
    rmin, rmax = extrema(xs)
    (c <= rmin || c >= rmax) && throw(ArgumentError(
        "$ctx: the cutoff $c must lie strictly inside the range of the running " *
        "variable [$rmin, $rmax]"))
    if bin === nothing
        bin = 2 * rsd * rn^(-1 / 2)
    end
    bin = Float64(bin)
    bin > 0 || throw(ArgumentError("$ctx: bin must be positive"))
    mid(v) = floor((v - c) / bin) * bin + bin / 2 + c
    l = mid(rmin)
    r = mid(rmax)
    lc = c - bin / 2
    rc = c + bin / 2
    j = floor(Int, (rmax - rmin) / bin) + 2
    cellval = zeros(j)
    for v in xs
        k = round(Int, (mid(v) - l) / bin + 1)
        k > j && (append!(cellval, zeros(k - j)); j = k)
        cellval[k] += 1
    end
    cellval = (cellval ./ rn) ./ bin
    cellmp = [floor(((l + (k - 1) * bin) - c) / bin) * bin + bin / 2 + c for k in 1:j]
    if bandwidth === nothing
        leftofc = round(Int, (mid(lc) - l) / bin + 1)
        rightofc = round(Int, (mid(rc) - l) / bin + 1)
        rightofc - leftofc == 1 || error("$ctx: bin grid does not split at the cutoff")
        il = findall(<(c), cellmp)
        ir = findall(>=(c), cellmp)
        lcoef, mse_l = _rd_quartic_fit(cellmp[il], cellval[il])
        mpl = cellmp[1:leftofc]
        fl = 2 .* lcoef[3] .+ 6 .* lcoef[4] .* mpl .+ 12 .* lcoef[5] .* mpl .* mpl
        hleft = 3.348 * (mse_l * (c - l) / sum(fl .* fl))^(1 / 5)
        rcoef, mse_r = _rd_quartic_fit(cellmp[ir], cellval[ir])
        mpr = cellmp[rightofc:j]
        fr = 2 .* rcoef[3] .+ 6 .* rcoef[4] .* mpr .+ 12 .* rcoef[5] .* mpr .* mpr
        hright = 3.348 * (mse_r * (r - c) / sum(fr .* fr))^(1 / 5)
        bandwidth = 0.5 * (hleft + hright)
    end
    bw = Float64(bandwidth)
    (isfinite(bw) && bw > 0) || throw(ArgumentError(
        "$ctx: the bandwidth must be positive and finite (got $bw)"))
    if count(v -> c - bw < v < c, xs) == 0 || count(v -> c <= v < c + bw, xs) == 0
        throw(ArgumentError("$ctx: insufficient data within the bandwidth"))
    end
    pad = ceil(Int, bw / bin)
    jp = j + 2 * pad
    cmp, cval = cellmp, cellval
    if pad >= 1
        cval = vcat(zeros(pad), cellval, zeros(pad))
        cmp = vcat(_rd_seq_by(l - pad * bin, l - bin, bin), cellmp,
                   _rd_seq_by(r + bin, r + pad * bin, bin))
        length(cmp) == jp || error("$ctx: internal error in histogram padding")
    end
    dist = cmp .- c
    function side_fit(right::Bool)
        w = 1 .- abs.(dist ./ bw)
        w = [w[k] > 0 ? w[k] * ((cmp[k] >= c) == right) : 0.0 for k in eachindex(w)]
        sw = sum(w)
        sw > 0 || throw(ArgumentError("$ctx: no histogram bins within the bandwidth"))
        w = (w ./ sw) .* jp
        return _rd_wls_intercept(dist, cval, w)
    end
    fhatl = side_fit(false)
    fhatr = side_fit(true)
    (fhatl > 0 && fhatr > 0) || throw(ArgumentError(
        "$ctx: the estimated density at the cutoff is not positive on both sides " *
        "(f_left = $fhatl, f_right = $fhatr); the log difference is undefined"))
    theta = log(fhatr) - log(fhatl)
    se = sqrt((1 / (rn * bw)) * (24 / 5) * ((1 / fhatr) + (1 / fhatl)))
    z = theta / se
    p = two_sided_pvalue(z)
    note = "McCrary (2008) binned local linear test (bin = $(round(bin; sigdigits=4)), " *
           "bandwidth = $(round(bw; sigdigits=4))). The local polynomial density test " *
           "of Cattaneo, Jansson & Ma (2020) (rd_density_test) avoids pre-binning and " *
           "uses robust bias-corrected inference; it is recommended in applications. " *
           "A non-rejection does not show that the running variable was not " *
           "manipulated."
    return DiagnosticTest("McCrary density test", "the density of the running " *
                          "variable is continuous at the cutoff", z, p;
                          method="binned local linear (triangular kernel), normal " *
                                 "approximation",
                          note=note,
                          details=(theta=theta, se=se, f_left=fhatl, f_right=fhatr,
                                   bin=bin, bandwidth=bw, n=rn,
                                   histogram=DataFrame(midpoint=cellmp,
                                                       density=cellval)))
end

function rd_mccrary_test(data::AbstractDataFrame, running::Symbol; kwargs...)
    require_columns(data, [running]; context="rd_mccrary_test")
    return rd_mccrary_test(data[!, running]; kwargs...)
end
