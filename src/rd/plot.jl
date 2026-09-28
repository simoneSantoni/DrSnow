# Data for RD plots (rdplot equivalent): binned means with data-driven bin selection and
# a global polynomial fit on each side of the cutoff. Plotting itself is left to a
# plotting layer (e.g. a Makie extension) that consumes `RDPlotData`.

"""
    RDPlotData

Result of [`rd_plot_data`](@ref): the binned means and global polynomial fits that make
up a regression discontinuity plot.

The object separates computation from drawing. [`plot_rd`](@ref), available when a
Makie backend is loaded, draws `bins` as a scatter of local means, optionally with
their confidence intervals, and `poly` as two curves that meet the cutoff from each
side. Both components are descriptive summaries of the data. They are not the RD
estimate, which comes from [`rd_estimate`](@ref).

# Fields
- `bins::DataFrame`: one row per non-empty bin with columns `side` (`:left`/`:right`),
  `bin` (negative on the left, positive on the right), `bin_mid` (bin midpoint),
  `mean_x`, `mean_y`, `bin_left`, `bin_right` (edges), `n`, `se_y`, `ci_lower`,
  `ci_upper` (``t``-based interval for the bin mean at `level`).
- `poly::DataFrame`: global polynomial fit evaluated on a grid (`side`, `x`, `y`).
- `coef_left`, `coef_right`: polynomial coefficients in powers of `x - cutoff`.
- `nbins::Tuple{Int,Int}`: numbers of bins used (left, right).
- `nbins_imse`, `nbins_mv`: IMSE-optimal and mimicking-variance numbers of bins.
- `binselect::Symbol`, `binselect_description::String`: bin selection method.
- `scale`, `rscale`: requested scaling of the number of bins, and the implied scaling
  relative to the IMSE-optimal choice.
- `bin_avg`, `bin_med`: average and median bin length on each side.
- `cutoff`, `p`, `h`, `n`, `n_h`, `kernel`, `level`: settings and sample sizes.
"""
struct RDPlotData
    bins::DataFrame
    poly::DataFrame
    coef_left::Vector{Float64}
    coef_right::Vector{Float64}
    nbins::Tuple{Int,Int}
    nbins_imse::Tuple{Int,Int}
    nbins_mv::Tuple{Int,Int}
    binselect::Symbol
    binselect_description::String
    scale::Tuple{Int,Int}
    rscale::Tuple{Float64,Float64}
    bin_avg::Tuple{Float64,Float64}
    bin_med::Tuple{Float64,Float64}
    cutoff::Float64
    p::Int
    h::Tuple{Float64,Float64}
    n::Tuple{Int,Int}
    n_h::Tuple{Int,Int}
    kernel::Symbol
    level::Float64
end

function Base.show(io::IO, ::MIME"text/plain", r::RDPlotData)
    println(io, "RD plot data, cutoff = $(r.cutoff)")
    println(io, "Bins (left, right): $(r.nbins) — $(r.binselect_description)")
    println(io, "IMSE-optimal bins: $(r.nbins_imse); mimicking-variance bins: " *
                "$(r.nbins_mv)")
    println(io, "Global polynomial of order $(r.p), $(r.kernel) kernel, h = $(r.h)")
    print(io, "Observations (left, right): $(r.n)")
end

Base.show(io::IO, r::RDPlotData) = print(io, "RDPlotData(nbins = $(r.nbins))")

const _RD_BINSELECT = Dict(
    :es => ("es", "IMSE-optimal evenly-spaced method using spacings estimators"),
    :espr => ("es", "IMSE-optimal evenly-spaced method using polynomial regression"),
    :esmv => ("es", "mimicking variance evenly-spaced method using spacings estimators"),
    :esmvpr => ("es", "mimicking variance evenly-spaced method using polynomial " *
                      "regression"),
    :qs => ("qs", "IMSE-optimal quantile-spaced method using spacings estimators"),
    :qspr => ("qs", "IMSE-optimal quantile-spaced method using polynomial regression"),
    :qsmv => ("qs", "mimicking variance quantile-spaced method using spacings " *
                    "estimators"),
    :qsmvpr => ("qs", "mimicking variance quantile-spaced method using polynomial " *
                      "regression"))

_rd_Jfun(B, V, n) = ceil(Int, (((2 * B) / V) * n)^(1 / 3))

_rd_powers(x, k) = [xi^j for xi in x, j in 0:k]

"""
    rd_plot_data(data, outcome, running; cutoff=0.0, p=4, nbins=nothing,
                 binselect=:esmv, scale=nothing, kernel=:uniform, weights=nothing,
                 h=nothing, covariates=Symbol[], covs_eval=:mean, covs_drop=true,
                 support=nothing, masspoints=:adjust, level=0.95) -> RDPlotData

Data for a regression discontinuity plot, equivalent to `rdplot` (R/Stata): means of the
outcome in bins of the running variable, with the number of bins chosen as in Calonico,
Cattaneo and Titiunik (2015), and a global polynomial fit on each side of the cutoff.

An RD plot shows the raw relationship between the outcome and the score and makes a
discontinuity at the cutoff visible, or its absence apparent. Calonico, Cattaneo and
Titiunik (2015) treat the binned means as a nonparametric estimator of the regression
function and derive two data-driven choices of the number of bins. The *IMSE-optimal*
number (`:es`, `:qs` and their `pr` variants) minimises the integrated mean squared error
of that estimator, trading off bias against variance. It tends to produce few bins and
a smooth picture of the underlying regression function. The *mimicking-variance* number
(`:esmv`, `:qsmv`, and variants, the default) chooses more bins, so that the scatter of
binned means has about the same variability as the raw data. It conveys the noise
in the data around the fitted curves. Bins are evenly spaced (`es`) or quantile spaced
(`qs`, equal numbers of observations per bin). The required variance and bias constants
are computed with spacings estimators or, in the `pr` variants, with polynomial
regression.

The overlaid curves are global polynomials of order `p` (4 by default, as in `rdplot`),
fitted separately on each side with kernel `kernel` over bandwidth `h` (by default the
whole support). They are a visual aid only. Global high-order polynomial fits are
sensitive to observations far from the cutoff and give misleading estimates of the
jump (Gelman & Imbens 2019). Estimate and test with [`rd_estimate`](@ref) and use the
plot to show the data behind that estimate. Report the binning method and the number of
bins with the figure.

# Arguments
- `data::AbstractDataFrame`: one row per unit. Rows with a missing value in any used
  column are dropped.
- `outcome::Symbol`: outcome column.
- `running::Symbol`: running variable. At least 20 observations are required.

# Keywords
- `cutoff::Real=0.0`: the RD threshold. It must lie strictly inside the range of the
  running variable.
- `p::Integer=4`: order of the global polynomial fit (`0` gives side-specific means).
- `nbins=nothing`: number of bins, as a scalar or a `(left, right)` pair. It overrides
  the number chosen by `binselect`, while the spacing (even or quantile) still follows
  `binselect`.
- `binselect=:esmv`: `:es`/`:qs` (IMSE-optimal, evenly or quantile spaced, spacings
  estimators), `:espr`/`:qspr` (IMSE-optimal, polynomial-regression estimators), and
  `:esmv`/`:qsmv`/`:esmvpr`/`:qsmvpr` (mimicking variance). With
  `masspoints = :adjust` and mass points detected, spacings methods switch to their
  polynomial-regression versions, as in `rdplot`.
- `scale=nothing`: integer multiplier(s) of the selected number of bins.
- `kernel=:uniform`: kernel weights of the polynomial fit (`:uniform`, `:triangular`
  or `:epanechnikov`).
- `weights::Union{Nothing,Symbol}=nothing`: observation weights for the polynomial fit.
- `h=nothing`: bandwidth of the polynomial fit (scalar or `(left, right)`); by default
  the whole support on each side.
- `covariates::Vector{Symbol}=Symbol[]`: covariates added linearly to the polynomial
  fit. Binned means are unaffected.
- `covs_eval=:mean`: value at which covariates are held when the fitted curve is
  evaluated: `:mean` (sample means) or `:zero`.
- `covs_drop::Bool=true`: drop covariates collinear with earlier ones.
- `support=nothing`: `(lower, upper)` support used to build evenly spaced bins, when it
  extends beyond the observed range.
- `masspoints=:adjust`: handling of repeated values of the running variable (`:adjust`,
  `:check` or `:off`).
- `level::Real=0.95`: confidence level of the bin-mean intervals.

# Returns
- [`RDPlotData`](@ref) with the binned means in `bins` and the fitted curves in `poly`.

# Examples
```julia
using DrSnow, CSV, DataFrames
senate = CSV.read(pkgdir(DrSnow, "test", "validation", "rd", "senate.csv"), DataFrame)
pd = rd_plot_data(senate, :vote, :margin)
pd.nbins, pd.nbins_imse                 # mimicking-variance vs IMSE-optimal bins
first(pd.bins, 3)
rd_plot_data(senate, :vote, :margin; binselect=:qs, p=1, h=20, kernel=:triangular)
```

# References
- Calonico, S., Cattaneo, M. D., & Titiunik, R. (2015). Optimal data-driven regression
  discontinuity plots. *Journal of the American Statistical Association*, 110(512),
  1753–1769.
- Gelman, A., & Imbens, G. (2019). Why high-order polynomials should not be used in
  regression discontinuity designs. *Journal of Business & Economic Statistics*, 37(3),
  447–456.
- Cattaneo, M. D., Idrobo, N., & Titiunik, R. (2020). *A Practical Introduction to
  Regression Discontinuity Designs: Foundations*. Cambridge University Press.
"""
function rd_plot_data(data::AbstractDataFrame, outcome::Symbol, running::Symbol;
                      cutoff::Real=0.0, p::Integer=4, nbins=nothing, binselect=:esmv,
                      scale=nothing, kernel=:uniform, weights=nothing, h=nothing,
                      covariates=Symbol[], covs_eval=:mean, covs_drop::Bool=true,
                      support=nothing, masspoints=:adjust, level::Real=0.95)
    ctx = "rd_plot_data"
    0 < level < 1 || throw(ArgumentError("$ctx: level must be in (0, 1)"))
    p >= 0 || throw(ArgumentError("$ctx: p must be non-negative"))
    E = _rd_extract(data, outcome, running; covariates, weights, context=ctx)
    Z, znames = _rd_prepare_covariates(E.Z, E.covariates, covs_drop)
    bs = Symbol(lowercase(string(binselect)))
    haskey(_RD_BINSELECT, bs) || throw(ArgumentError(
        "$ctx: binselect must be one of " *
        join(sort(collect(keys(_RD_BINSELECT))), ", ")))
    ce = Symbol(lowercase(string(covs_eval)))
    ce in (:mean, :zero) || throw(ArgumentError("$ctx: covs_eval must be :mean or :zero"))
    kern = _rd_kernel(kernel)
    mp = _rd_masspoints(masspoints)
    c = Float64(cutoff)
    x = E.x .+ 0.0
    y = E.y
    n = length(x)
    n >= 20 || throw(ArgumentError("$ctx: at least 20 observations are needed"))
    x_min, x_max = extrema(x)
    il = x .< c
    ir = .!il
    x_l, x_r, y_l, y_r = x[il], x[ir], y[il], y[ir]
    if support !== nothing
        sl, sr = _rd_pair(support)
        sl < x_min && (x_min = sl)
        sr > x_max && (x_max = sr)
    end
    (c <= x_min || c >= x_max) && throw(ArgumentError(
        "$ctx: the cutoff must lie strictly inside the range of the running variable"))
    range_l = c - x_min
    range_r = x_max - c
    n_l, n_r = length(x_l), length(x_r)
    (n_l >= 2 && n_r >= 2) || throw(ArgumentError(
        "$ctx: need at least two observations on each side of the cutoff"))
    scale_l, scale_r = scale === nothing ? (1, 1) : Int.(_rd_pair(scale))
    (scale_l > 0 && scale_r > 0) || throw(ArgumentError("$ctx: scale must be positive"))
    h_l, h_r = h === nothing ? (range_l, range_r) : _rd_pair(h)
    if mp !== :off
        mass_l = 1 - length(unique(x_l)) / n_l
        mass_r = 1 - length(unique(x_r)) / n_r
        if mass_l >= 0.2 || mass_r >= 0.2
            if mp === :check
                @warn "$ctx: mass points detected in the running variable"
            else
                bs = bs === :es ? :espr : bs === :esmv ? :esmvpr :
                     bs === :qs ? :qspr : bs === :qsmv ? :qsmvpr : bs
            end
        end
    end

    # Global polynomial fit (weighted by kernel and optional weights).
    W_l = _rd_kweight(x_l, c, h_l, kern)
    W_r = _rd_kweight(x_r, c, h_r, kern)
    n_h = (count(>(0), W_l), count(>(0), W_r))
    if E.W !== nothing
        W_l .*= E.W[il]
        W_r .*= E.W[ir]
    end
    R_l = _rd_powers(x_l .- c, p)
    R_r = _rd_powers(x_r .- c, p)
    invG_l = _rd_xxinv(sqrt.(W_l) .* R_l)
    invG_r = _rd_xxinv(sqrt.(W_r) .* R_r)
    gammaZ = 0.0
    if Z === nothing
        g_l = invG_l * ((R_l .* W_l)' * y_l)
        g_r = invG_r * ((R_r .* W_r)' * y_r)
    else
        dZ = size(Z, 2)
        D_l = hcat(y_l, Z[il, :]); D_r = hcat(y_r, Z[ir, :])
        cols = 2:(1 + dZ)
        function zp(R, W, invG, D)
            U = (R .* W)' * D
            ZWD = (D[:, cols] .* W)' * D
            UiGU = U[:, cols]' * (invG * U)
            return ZWD[:, cols] .- UiGU[:, cols], ZWD[:, 1] .- UiGU[:, 1]
        end
        a_l, b_l = zp(R_l, W_l, invG_l, D_l)
        a_r, b_r = zp(R_r, W_r, invG_r, D_r)
        gamma = _rd_ginv(a_l .+ a_r) * (b_l .+ b_r)
        s = vcat(1.0, -gamma)
        g_l = (invG_l * ((R_l .* W_l)' * D_l)) * s
        g_r = (invG_r * ((R_r .* W_r)' * D_r)) * s
        ce === :mean && (gammaZ = dot(vec(mean(Z; dims=1)), gamma))
    end
    nplot = 500
    grid(a, b) = [k == nplot - 1 ? b : a + k * ((b - a) / (nplot - 1))
                  for k in 0:(nplot - 1)]
    xp_l = grid(c - h_l, c)
    xp_r = grid(c, c + h_r)
    yp_l = _rd_powers(xp_l .- c, p) * g_l .+ gammaZ
    yp_r = _rd_powers(xp_r .- c, p) * g_r .+ gammaZ

    # Number of bins (Calonico, Cattaneo & Titiunik 2015).
    k = 4
    function kfit(xs, ys)
        rk = _rd_powers(xs, k)
        invG = _rd_xxinv(rk)
        return rk, invG * (rk' * ys), invG * (rk' * (ys .^ 2))
    end
    rk_l, g1_l, g2_l = kfit(x_l, y_l)
    rk_r, g1_r, g2_r = kfit(x_r, y_r)
    dpow(xs) = [j * xi^(j - 1) for xi in xs, j in 1:k]
    ol, or_ = sortperm(x_l; alg=MergeSort), sortperm(x_r; alg=MergeSort)
    xi_l, yi_l, xi_r, yi_r = x_l[ol], y_l[ol], x_r[or_], y_r[or_]
    dxi_l = diff(xi_l); dyi_l = diff(yi_l)
    dxi_r = diff(xi_r); dyi_r = diff(yi_r)
    xbar_l = (xi_l[2:end] .+ xi_l[1:(end - 1)]) ./ 2
    xbar_r = (xi_r[2:end] .+ xi_r[1:(end - 1)]) ./ 2
    rki_l = _rd_powers(xbar_l, k); rki_r = _rd_powers(xbar_r, k)
    mu0i_l = rki_l * g1_l; mu0i_r = rki_r * g1_r
    mu2i_l = rki_l * g2_l; mu2i_r = rki_r * g2_r
    mu0_l = rk_l * g1_l; mu0_r = rk_r * g1_r
    mu2_l = rk_l * g2_l; mu2_r = rk_r * g2_r
    mu1_l = dpow(x_l) * g1_l[2:end]; mu1_r = dpow(x_r) * g1_r[2:end]
    mu1i_l = dpow(xbar_l) * g1_l[2:end]; mu1i_r = dpow(xbar_r) * g1_r[2:end]
    var_y_l, var_y_r = var(y_l), var(y_r)
    s2bar_l = mu2i_l .- mu0i_l .^ 2; s2bar_l[s2bar_l .< 0] .= var_y_l
    s2bar_r = mu2i_r .- mu0i_r .^ 2; s2bar_r[s2bar_r .< 0] .= var_y_r
    s2_l = mu2_l .- mu0_l .^ 2; s2_l[s2_l .< 0] .= var_y_l
    s2_r = mu2_r .- mu0_r .^ 2; s2_r[s2_r .< 0] .= var_y_r
    B_es = (((c - x_min)^2 / (12 * n)) * sum(mu1_l .^ 2),
            ((x_max - c)^2 / (12 * n)) * sum(mu1_r .^ 2))
    V_es_hat = ((0.5 / (c - x_min)) * sum(dxi_l .* dyi_l .^ 2),
                (0.5 / (x_max - c)) * sum(dxi_r .* dyi_r .^ 2))
    V_es_chk = ((1 / (c - x_min)) * sum(dxi_l .* s2bar_l),
                (1 / (x_max - c)) * sum(dxi_r .* s2bar_r))
    B_qs = ((n_l^2 / (24 * n)) * sum(dxi_l .^ 2 .* mu1i_l .^ 2),
            (n_r^2 / (24 * n)) * sum(dxi_r .^ 2 .* mu1i_r .^ 2))
    V_qs_hat = ((1 / (2 * n_l)) * sum(dyi_l .^ 2), (1 / (2 * n_r)) * sum(dyi_r .^ 2))
    V_qs_chk = ((1 / n_l) * sum(s2_l), (1 / n_r) * sum(s2_r))
    J2(B, V) = (_rd_Jfun(B[1], V[1], n), _rd_Jfun(B[2], V[2], n))
    mv(V) = (ceil(Int, (var_y_l / V[1]) * (n / log(n)^2)),
             ceil(Int, (var_y_r / V[2]) * (n / log(n)^2)))
    J_imse, J_mv = if bs in (:es, :esmv)
        J2(B_es, V_es_hat), mv(V_es_hat)
    elseif bs in (:espr, :esmvpr)
        J2(B_es, V_es_chk), mv(V_es_chk)
    elseif bs in (:qs, :qsmv)
        J2(B_qs, V_qs_hat), mv(V_qs_hat)
    else
        J2(B_qs, V_qs_chk), mv(V_qs_chk)
    end
    meth, desc = _RD_BINSELECT[bs]
    J_orig = bs in (:es, :espr, :qs, :qspr) ? J_imse : J_mv
    J_l, J_r = scale_l * J_orig[1], scale_r * J_orig[2]
    if nbins !== nothing
        J_l, J_r = Int.(_rd_pair(nbins))
        (J_l > 0 && J_r > 0) || throw(ArgumentError("$ctx: nbins must be positive"))
        desc = "manually selected number of bins, " *
               (meth == "es" ? "evenly spaced" : "quantile spaced")
    end
    if var_y_l == 0
        J_l = 1
        @warn "$ctx: no variability in the outcome below the cutoff"
    end
    if var_y_r == 0
        J_r = 1
        @warn "$ctx: no variability in the outcome above the cutoff"
    end
    rscale = (J_l / J_imse[1], J_r / J_imse[2])
    if meth == "es"
        jumps_l = _rd_seq_by(x_min, c, range_l / J_l)
        jumps_r = _rd_seq_by(c, x_max, range_r / J_r)
    else
        jumps_l = _rd_quantile_type7(x_l, _rd_seq_by(0.0, 1.0, 1 / J_l))
        jumps_r = _rd_quantile_type7(x_r, _rd_seq_by(0.0, 1.0, 1 / J_r))
    end
    bin_l = [_rd_find_interval(v, jumps_l) - J_l - 1 for v in x_l]
    bin_r = [_rd_find_interval(v, jumps_r) for v in x_r]
    tq(nb) = quantile(TDist(max(nb - 1, 1)), 1 - (1 - level) / 2)
    rows = DataFrame(side=Symbol[], bin=Int[], bin_mid=Float64[], mean_x=Float64[],
                     mean_y=Float64[], bin_left=Float64[], bin_right=Float64[], n=Int[],
                     se_y=Float64[], ci_lower=Float64[], ci_upper=Float64[])
    for (side, bins, xs, ys, jumps, J) in ((:left, bin_l, x_l, y_l, jumps_l, J_l),
                                           (:right, bin_r, x_r, y_r, jumps_r, J_r))
        for bnum in sort(unique(bins))
            idx = findall(==(bnum), bins)
            pos = side === :left ? bnum + J + 1 : bnum     # position among the J bins
            lo, hi = jumps[pos], jumps[pos + 1]
            nb = length(idx)
            my = mean(ys[idx])
            sdv = nb > 1 ? std(ys[idx]) : 0.0
            se = sdv / sqrt(nb)
            q = tq(nb)
            push!(rows, (side, bnum, (lo + hi) / 2, mean(xs[idx]), my, lo, hi, nb, se,
                         my - q * se, my + q * se))
        end
    end
    len_l = jumps_l[2:(J_l + 1)] .- jumps_l[1:J_l]
    len_r = jumps_r[2:(J_r + 1)] .- jumps_r[1:J_r]
    poly = DataFrame(side=vcat(fill(:left, nplot), fill(:right, nplot)),
                     x=vcat(xp_l, xp_r), y=vcat(yp_l, yp_r))
    return RDPlotData(rows, poly, g_l, g_r, (J_l, J_r), J_imse, J_mv, bs, desc,
                      (scale_l, scale_r), rscale, (mean(len_l), mean(len_r)),
                      (median(len_l), median(len_r)), c, Int(p), (h_l, h_r), (n_l, n_r),
                      n_h, kern, Float64(level))
end
