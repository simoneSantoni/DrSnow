# Augmented synthetic control (Ben-Michael, Feller & Rothstein 2021) with a ridge outcome
# model, following the R package augsynth (`progfunc = "Ridge"`), and conformal
# inference for synthetic-control-type estimators (Chernozhukov, Wüthrich & Zhu 2021).

struct _sc_AscmOpts
    lambda::Union{Nothing,Float64}
    ridge::Bool
    scm::Bool
    fixed_effects::Bool
    n_lambda::Int
    lambda_min_ratio::Float64
    holdout_length::Int
    min_1se::Bool
end

"""
    AugmentedSCEstimate <: CausalEstimate

Result of [`augmented_synthetic_control`](@ref): a ridge-augmented (or plain) synthetic
control estimate for one or more treated units with a common adoption date.

The coefficient `"ATT"` is the average post-treatment gap between the treated unit (the
average of the treated units) and its augmented synthetic control. `vcov` is the
placebo variance (default) or `augsynth`'s leave-one-unit-out jackknife variance, both
with the ridge penalty held fixed. It throws when `se_method = :none`. Per-period
design-based inference over time is available through
[`synth_conformal_inference`](@ref).

# Fields
- `att::Float64`: average post-treatment gap.
- `se::Union{Nothing,Float64}`, `se_method::Symbol`: standard error and its method.
- `replicate_estimates::Vector{Float64}`: placebo estimates, if any.
- `weights::Vector{Float64}`: augmented weights on the never-treated units (they sum to
  one and may be negative); `scm_weights::Vector{Float64}`: the underlying synthetic
  control weights.
- `lambda::Union{Nothing,Float64}`: ridge penalty (`nothing` without ridge
  augmentation); `cv::Union{Nothing,DataFrame}`: cross-validation curve (`lambda`,
  `error`, `error_se`).
- `att_path::Vector{Float64}`: gap in every period; `se_path`: jackknife standard
  errors of the post-treatment gaps (or `nothing`).
- `treated_path`, `synthetic_path`: outcome paths; `n_pre::Int`: number of
  pre-treatment periods; `panel::SynthPanel`: the data used.
"""
struct AugmentedSCEstimate{P<:SynthPanel} <: CausalEstimate
    att::Float64
    se::Union{Nothing,Float64}
    se_method::Symbol
    replicate_estimates::Vector{Float64}
    panel::P
    weights::Vector{Float64}
    scm_weights::Vector{Float64}
    lambda::Union{Nothing,Float64}
    att_path::Vector{Float64}
    se_path::Union{Nothing,Vector{Float64}}
    treated_path::Vector{Float64}
    synthetic_path::Vector{Float64}
    n_pre::Int
    cv::Union{Nothing,DataFrame}
    opts::_sc_AscmOpts
end

StatsAPI.coef(r::AugmentedSCEstimate) = [r.att]
StatsAPI.coefnames(::AugmentedSCEstimate) = ["ATT"]
StatsAPI.nobs(r::AugmentedSCEstimate) = length(r.panel.Y)
function StatsAPI.vcov(r::AugmentedSCEstimate)
    r.se === nothing && throw(ArgumentError("no standard error was computed; re-run " *
                                            "with se_method = :placebo or :jackknife"))
    return fill(r.se^2, 1, 1)
end
estimand(::AugmentedSCEstimate) = "ATT (average post-period gap of the treated units)"
method_name(r::AugmentedSCEstimate) =
    r.opts.ridge ? "Augmented synthetic control (ridge)" : "Synthetic control (augsynth)"

# ---------------------------------------------------------------------------------------
# Core (augsynth::fit_ridgeaug_formatted / predict.augsynth)
# ---------------------------------------------------------------------------------------

_sc_ridge_weights(Xc, x1, syn, lambda) =
    Xc * ((Xc' * Xc + lambda * I) \ (x1 .- Xc' * syn))

function _sc_ascm_cv(Xc::Matrix{Float64}, x1::Vector{Float64}, o::_sc_AscmOpts)
    lambda_max = svdvals(Xc)[1]^2
    scaler = o.lambda_min_ratio^(1 / o.n_lambda)
    lambdas = lambda_max .* scaler .^ (0:o.n_lambda)
    T0 = size(Xc, 2)
    h = o.holdout_length
    nfold = T0 - h
    nfold >= 2 || throw(ArgumentError("augmented_synthetic_control: too few " *
                                      "pre-treatment periods to cross-validate lambda"))
    errors = zeros(nfold, length(lambdas))
    for i in 1:nfold
        hold = i:(i + h - 1)
        keep = setdiff(1:T0, hold)
        X0 = Xc[:, keep]
        x1k = x1[keep]
        syn = o.scm ? _sc_simplex_ls(Matrix(X0'), x1k) : fill(1 / size(Xc, 1), size(Xc, 1))
        for (j, lam) in enumerate(lambdas)
            w = syn .+ _sc_ridge_weights(X0, x1k, syn, lam)
            errors[i, j] = sum(abs2, x1[hold] .- Xc[:, hold]' * w)
        end
    end
    err = vec(mean(errors; dims=1))
    err_se = vec(std(errors; dims=1)) ./ sqrt(nfold)
    imin = argmin(err)
    lambda_1se = maximum(lambdas[err .<= err[imin] + err_se[imin]])
    lambda = o.min_1se ? lambda_1se : lambdas[imin]
    return lambda, DataFrame(lambda=lambdas, error=err, error_se=err_se)
end

# X: N × T0 "pre" outcomes, y: N × T1 remaining periods, trt: treated rows.
# Returns weights, counterfactual path and gaps (over [X y] columns).
function _sc_ascm_core(X::Matrix{Float64}, y::Matrix{Float64}, trt::AbstractVector{Bool},
                       o::_sc_AscmOpts)
    ctrl = .!trt
    N0 = count(ctrl)
    comb = hcat(X, y)
    if o.fixed_effects
        means = mean(X; dims=2)
        X = X .- means
        comb = comb .- means
    end
    Xcent = X .- mean(X[ctrl, :]; dims=1)
    Xc = Xcent[ctrl, :]
    x1 = vec(mean(Xcent[trt, :]; dims=1))
    syn = o.scm ? _sc_simplex_ls(Matrix(Xc'), x1) : fill(1 / N0, N0)
    lambda = o.lambda
    cv = nothing
    w = copy(syn)
    if o.ridge
        if lambda === nothing
            lambda, cv = _sc_ascm_cv(Xc, x1, o)
        end
        w = syn .+ _sc_ridge_weights(Xc, x1, syn, lambda)
    end
    treated = vec(mean(comb[trt, :]; dims=1))
    synthetic = vec(comb[ctrl, :]' * w)
    return (weights=w, syn=syn, lambda=lambda, cv=cv, treated=treated,
            synthetic=synthetic, gap=treated .- synthetic)
end

function _sc_ascm_jackknife(X, y, trt, weights, o::_sc_AscmOpts, n_pre)
    n = size(X, 1)
    nnz = zeros(Bool, n)
    nnz[findall(.!trt)] .= round.(weights; digits=3) .!= 0
    count(trt) > 1 && (nnz[trt] .= true)
    idx = findall(nnz)
    T1 = size(y, 2)
    ests = zeros(length(idx), T1 + 1)
    for (k, i) in enumerate(idx)
        keep = [j for j in 1:n if j != i]
        f = _sc_ascm_core(X[keep, :], y[keep, :], trt[keep], o)
        e = f.gap[(n_pre + 1):end]
        ests[k, :] = vcat(e, mean(e))
    end
    se = [sqrt((n - 1) / n * sum(abs2, c .- mean(c))) for c in eachcol(ests)]
    return se[end], se[1:T1]
end

# Placebo variance (Arkhangelsky et al. 2021, Algorithm 4): the treated units' role is
# given to randomly chosen never-treated units, with λ held at its estimated value.
function _sc_ascm_placebo(X, y, trt, o::_sc_AscmOpts, n_pre, B, rng)
    ctrl = findall(.!trt)
    n1, nc = count(trt), length(ctrl)
    nc > n1 + 1 ||
        throw(ArgumentError("augmented_synthetic_control: the placebo standard error " *
                            "needs at least two more never-treated than treated units"))
    seeds = task_seeds(rng, B)
    draws = zeros(B)
    Threads.@threads for b in 1:B
        perm = randperm(Random.Xoshiro(seeds[b]), nc)
        rows = ctrl[perm]
        t = vcat(falses(nc - n1), trues(n1))
        f = _sc_ascm_core(X[rows, :], y[rows, :], t, o)
        draws[b] = mean(f.gap[(n_pre + 1):end])
    end
    return draws
end

"""
    augmented_synthetic_control(data, outcome, treatment, unit, time; lambda=nothing,
                                ridge=true, fixed_effects=false, se_method=:placebo,
                                replications=200, n_lambda=20, lambda_min_ratio=1e-8,
                                holdout_length=1, min_1se=true,
                                rng=Random.default_rng()) -> AugmentedSCEstimate
    augmented_synthetic_control(panel::SynthPanel; kwargs...) -> AugmentedSCEstimate

Ridge-augmented synthetic control (Ben-Michael, Feller & Rothstein 2021), following the R
package `augsynth` (`progfunc = "Ridge", scm = TRUE`).

The estimand is the average post-treatment effect on the treated unit (or on the average
of several treated units that adopt together). The synthetic control method is
justified when the weighted donors reproduce the treated unit's pre-treatment outcomes
closely (Abadie, Diamond & Hainmueller 2010). When the fit is imperfect, which is common
when the treated unit lies near or outside the boundary of the donors' convex hull, the
remaining imbalance biases the estimate (Ferman & Pinto 2021). Ben-Michael, Feller and
Rothstein (2021) correct this bias with an outcome model, in the spirit of augmented
inverse-propensity weighting. The augmented estimator is
``\\hat Y_1(0) = \\sum_j \\hat\\gamma_j Y_j + (X_1 - \\sum_j \\hat\\gamma_j X_j)^\\top
\\hat\\eta``, where ``X`` are pre-treatment outcomes and ``\\hat\\eta`` is a ridge
regression of post- on pre-treatment outcomes among the donors. With ridge regression
the estimator is itself a weighting estimator, with weights
```math
\\hat\\gamma^{\\text{aug}} = \\hat\\gamma^{\\text{scm}} + X_0 (X_0^\\top X_0 +
    \\lambda I)^{-1} (x_1 - X_0^\\top \\hat\\gamma^{\\text{scm}}),
```
with outcomes centred at the control means. The weights still sum to one but can be
negative: the correction extrapolates beyond the convex hull. Ben-Michael, Feller and
Rothstein (2021) bound the resulting bias under a linear factor model and show how the
penalty ``\\lambda`` trades imbalance against extrapolation. Large negative weights
signal heavy reliance on the outcome model. The printed summary reports their sum.

``\\lambda`` is chosen by cross-validation over the pre-treatment periods. Each fold
holds out a block of `holdout_length` periods, and the one-standard-error rule is
applied by default. `fixed_effects = true` first de-means each unit by its pre-treatment
average, which allows a level difference between the treated unit and the donors, as in
[`synthetic_did`](@ref). With `ridge = false` the function returns the plain
outcome-only synthetic control of `augsynth`, useful as the input for conformal
inference. All treated units must adopt in the same period. For staggered adoption use
[`synthetic_did`](@ref) or [`matrix_completion`](@ref). Ben-Michael, Feller and
Rothstein (2022) develop partially pooled synthetic controls for that case, which are
not implemented here.

Inference: the default *placebo* standard error re-runs the estimator with the treated
units' role given to randomly chosen never-treated units (Arkhangelsky et al. 2021,
Algorithm 4), with ``\\lambda`` held fixed. It assumes that treated and control units
share the same noise distribution, and it does not account for the selection of
``\\lambda``. The *jackknife* is `augsynth`'s leave-one-unit-out estimator over units
with non-zero weight (and over treated units when there are several), again with
``\\lambda`` fixed. With a single treated unit, that unit is never left out, so the
standard error ignores the noise in the treated unit's own outcomes and understates
uncertainty. Use the jackknife only with several treated units. For a single treated
unit, per-period conformal inference over time is available through
[`synth_conformal_inference`](@ref) (Chernozhukov, Wüthrich & Zhu 2021). Report the
pre-treatment fit, the penalty and the sum of negative weights, and compare with the
unaugmented fit (`ridge = false`).

# Arguments
- `data`: long panel (see [`synth_panel`](@ref)), with columns `outcome`, `treatment`,
  `unit` and `time`; all treated units must adopt in the same period. Or
- `panel::SynthPanel`: a prepared panel.

# Keywords
- `lambda=nothing`: ridge penalty. By default it is cross-validated.
- `ridge::Bool=true`: `false` gives the plain (outcome-only) synthetic control of
  `augsynth`.
- `fixed_effects::Bool=false`: de-mean each unit by its pre-treatment average first
  (`augsynth`'s `fixedeff = TRUE`).
- `se_method::Symbol=:placebo`: `:placebo`, `:jackknife` or `:none`. The placebo
  method needs at least two more never-treated than treated units.
- `replications::Integer=200`: number of placebo draws.
- `n_lambda::Integer=20`: number of grid points below the largest penalty.
- `lambda_min_ratio::Real=1e-8`: ratio of the smallest to the largest penalty on the
  grid.
- `holdout_length::Integer=1`: number of consecutive pre-treatment periods held out in
  each cross-validation fold.
- `min_1se::Bool=true`: choose the largest ``\\lambda`` within one standard error of
  the minimum cross-validation error (`false`: the minimiser).
- `rng::AbstractRNG=Random.default_rng()`: generator for the placebo draws.

# Returns
- [`AugmentedSCEstimate`](@ref).

# Examples
```julia
using DrSnow, CSV, DataFrames, StableRNGs
prop99 = CSV.read(pkgdir(DrSnow, "test", "validation", "synth", "prop99.csv"), DataFrame)
r = augmented_synthetic_control(prop99, :PacksPerCapita, :treated, :State, :Year;
                                rng=StableRNG(1))
coef(r), stderror(r), r.lambda
sum(min.(synth_weights(r).weight, 0))          # total negative weight
r0 = augmented_synthetic_control(prop99, :PacksPerCapita, :treated, :State, :Year;
                                 ridge=false, se_method=:none)
coef(r0)
```

# References
- Ben-Michael, E., Feller, A., & Rothstein, J. (2021). The augmented synthetic control
  method. *Journal of the American Statistical Association*, 116(536), 1789–1803.
- Ben-Michael, E., Feller, A., & Rothstein, J. (2022). Synthetic controls with
  staggered adoption. *Journal of the Royal Statistical Society Series B: Statistical
  Methodology*, 84(2), 351–381.
- Abadie, A., Diamond, A., & Hainmueller, J. (2010). Synthetic control methods for
  comparative case studies: Estimating the effect of California's tobacco control
  program. *Journal of the American Statistical Association*, 105(490), 493–505.
- Ferman, B., & Pinto, C. (2021). Synthetic controls with imperfect pretreatment fit.
  *Quantitative Economics*, 12(4), 1197–1221.
- Arkhangelsky, D., Athey, S., Hirshberg, D. A., Imbens, G. W., & Wager, S. (2021).
  Synthetic difference-in-differences. *American Economic Review*, 111(12), 4088–4118.
- Chernozhukov, V., Wüthrich, K., & Zhu, Y. (2021). An exact and robust conformal
  inference method for counterfactual and synthetic controls. *Journal of the American
  Statistical Association*, 116(536), 1849–1864.
"""
function augmented_synthetic_control(data, outcome::Symbol, treatment::Symbol,
                                     unit::Symbol, time::Symbol; kwargs...)
    return augmented_synthetic_control(synth_panel(data, outcome, treatment, unit, time);
                                       kwargs...)
end

function augmented_synthetic_control(panel::SynthPanel; lambda=nothing,
                                     ridge::Bool=true, fixed_effects::Bool=false,
                                     se_method::Symbol=:placebo,
                                     replications::Integer=200, n_lambda::Integer=20,
                                     lambda_min_ratio::Real=1e-8,
                                     holdout_length::Integer=1, min_1se::Bool=true,
                                     rng::AbstractRNG=Random.default_rng())
    _sc_is_block(panel) ||
        throw(ArgumentError("augmented_synthetic_control: all treated units must adopt " *
                            "in the same period; for staggered adoption use " *
                            "synthetic_did or matrix_completion"))
    se_method in (:placebo, :jackknife, :none) ||
        throw(ArgumentError("augmented_synthetic_control: se_method must be :placebo, " *
                            ":jackknife or :none"))
    se_method === :placebo && replications < 2 &&
        throw(ArgumentError("augmented_synthetic_control: replications must be ≥ 2"))
    lambda === nothing || lambda >= 0 ||
        throw(ArgumentError("augmented_synthetic_control: lambda must be non-negative"))
    n_pre = panel.adoption[end] - 1
    n_pre >= 2 || throw(ArgumentError("augmented_synthetic_control: at least two " *
                                      "pre-treatment periods are needed"))
    panel.n_control >= 2 || throw(ArgumentError("augmented_synthetic_control: at least " *
                                                "two never-treated units are needed"))
    o = _sc_AscmOpts(lambda === nothing ? nothing : Float64(lambda), ridge, true,
                     fixed_effects, n_lambda, lambda_min_ratio, holdout_length, min_1se)
    X = panel.Y[:, 1:n_pre]
    y = panel.Y[:, (n_pre + 1):end]
    trt = panel.adoption .> 0
    f = _sc_ascm_core(X, y, trt, o)
    ofix = _sc_AscmOpts(f.lambda, ridge, true, fixed_effects, n_lambda, lambda_min_ratio,
                        holdout_length, min_1se)
    att = mean(f.gap[(n_pre + 1):end])
    se, se_path = nothing, nothing
    draws = Float64[]
    if se_method === :jackknife
        se, se_path = _sc_ascm_jackknife(X, y, trt, f.weights, ofix, n_pre)
    elseif se_method === :placebo
        draws = _sc_ascm_placebo(X, y, trt, ofix, n_pre, replications, rng)
        se = _sc_replicate_se(draws)
    end
    return AugmentedSCEstimate(att, se, se_method, draws, panel, f.weights, f.syn,
                               f.lambda, f.gap, se_path, f.treated, f.synthetic, n_pre,
                               f.cv, ofix)
end

# ---------------------------------------------------------------------------------------
# Conformal inference (Chernozhukov, Wüthrich & Zhu 2021; augsynth::conformal_inf)
# ---------------------------------------------------------------------------------------

_sc_conformal_stat(x, q) = (sum(abs.(x) .^ q) / sqrt(length(x)))^(1 / q)

# p-value of H0: effect = h0 in the last `post_length` columns of X.
function _sc_conformal_pvalue(X, trt, o, h0, post_length, type, q, ns, rng)
    Xh = copy(X)
    tpost = size(X, 2)
    t0 = tpost - post_length
    Xh[trt, (t0 + 1):tpost] .-= h0
    f = _sc_ascm_core(Xh, zeros(size(X, 1), 0), trt, o)
    resid = f.gap
    obs = _sc_conformal_stat(resid[(t0 + 1):tpost], q)
    if type === :block
        stats = [_sc_conformal_stat([resid[mod(s + t0 + k - 1, tpost) + 1]
                                     for k in 1:post_length], q) for s in 0:(tpost - 1)]
        return mean(obs .<= stats)
    elseif post_length == 1
        # all permutations: the post residual is equally likely to be any residual
        return mean(obs .<= abs.(resid))
    else
        stats = [_sc_conformal_stat(shuffle(rng, resid)[(t0 + 1):tpost], q) for _ in 1:ns]
        return mean(obs .<= stats)
    end
end

"""
    synth_conformal_inference(r::AugmentedSCEstimate; level=0.95, type=:block, q=1,
                              grid_size=50, ns=1000,
                              rng=Random.default_rng()) -> NamedTuple

Conformal inference for synthetic-control-type estimators (Chernozhukov, Wüthrich & Zhu
2021), as implemented in `augsynth`: per-period confidence intervals for the effect on
the treated unit and a joint test of no effect in all post-treatment periods.

Chernozhukov, Wüthrich and Zhu (2021) treat the counterfactual as a prediction problem
and obtain inference by permuting residuals *over time*, not across units. To test
``H_0: \\tau_t = \\tau_0`` for a post-treatment period ``t``, the treated outcome in
``t`` is adjusted by ``\\tau_0``, and the estimator is re-fitted on the pre-treatment
periods plus period ``t``. The ridge penalty is held at its estimated value. Under the
null, the residual of period ``t`` should look like the pre-treatment residuals. The
statistic ``S(u) = (\\sum |u_s|^q / \\sqrt{n})^{1/q}`` of the post-treatment residuals
is compared with its values under permutations of the residual series, which gives a
p-value. Inverting the test over a grid of ``\\tau_0`` values yields the confidence set.
The grid has `grid_size` points spanning ``\\hat\\tau_t \\pm 2 s``, where ``s`` is the
root mean square of the post-treatment gap estimates. When the accepted set reaches the
edge of the grid (`truncated = true`) the interval is only known to contain the reported
range. The joint test imposes ``\\tau_t = 0`` in all post-treatment periods at once.

Validity rests on assumptions about the residuals over time, not about the assignment
of units. With `type = :iid` the residuals must be exchangeable over time. With
`type = :block` (moving-block permutations, the default) they must be stationary and
weakly dependent. In both cases the estimator of the counterfactual must be stable
(consistent, or not unduly sensitive to the post-treatment observation). Under these
conditions the test is exact with exchangeable residuals and approximately valid
otherwise (Chernozhukov, Wüthrich & Zhu 2021). Trends or structural breaks in the
pre-treatment residuals undermine it.

**Acceptance convention and coarseness.** Following `augsynth`, a value ``\\tau_0`` is
retained when its p-value satisfies ``p \\ge \\alpha = 1 - \\text{level}``, with a small
numerical tolerance. Since block p-values are multiples of ``1/(T_0 + 1)`` for a single
period, this convention is conservative. A value with ``p = \\alpha`` exactly is kept,
and no value can be rejected unless ``1/(T_0 + 1) < \\alpha``. With 19 pre-treatment
periods, as in the Prop 99 example below, the smallest p-value is 0.05, so 95% intervals
cover the whole grid (all rows `truncated`). In that case use a lower level or report
the p-values. The per-period `p_value` column tests ``\\tau_t = 0``.

# Arguments
- `r::AugmentedSCEstimate`: a fitted ridge-augmented or plain synthetic control from
  [`augmented_synthetic_control`](@ref).

# Keywords
- `level::Real=0.95`: confidence level of the per-period intervals.
- `type::Symbol=:block`: `:block` (moving-block permutations, deterministic) or `:iid`
  (random permutations; all permutations are enumerated for single periods).
- `q::Real=1`: exponent of the test statistic.
- `grid_size::Integer=50`: number of candidate values ``\\tau_0`` per period (at least
  2).
- `ns::Integer=1000`: number of random permutations for the joint `:iid` test.
- `rng::AbstractRNG=Random.default_rng()`: generator for the random permutations.

# Returns
- `NamedTuple` with fields
  - `per_period::DataFrame`: `time`, `estimate`, `lower`, `upper`, `truncated` and
    `p_value` (for ``\\tau_t = 0``); the bounds are `missing` if no grid value is
    accepted;
  - `joint::DiagnosticTest`: the joint test of no effect in all post-treatment periods.

# Examples
```julia
using DrSnow, CSV, DataFrames
prop99 = CSV.read(pkgdir(DrSnow, "test", "validation", "synth", "prop99.csv"), DataFrame)
r = augmented_synthetic_control(prop99, :PacksPerCapita, :treated, :State, :Year;
                                se_method=:none)
ci = synth_conformal_inference(r)
ci.per_period                 # 19 pre-periods: p-values are multiples of 1/20
ci.joint.pvalue
synth_conformal_inference(r; level=0.90).per_period
```

# References
- Chernozhukov, V., Wüthrich, K., & Zhu, Y. (2021). An exact and robust conformal
  inference method for counterfactual and synthetic controls. *Journal of the American
  Statistical Association*, 116(536), 1849–1864.
- Ben-Michael, E., Feller, A., & Rothstein, J. (2021). The augmented synthetic control
  method. *Journal of the American Statistical Association*, 116(536), 1789–1803.
- Abadie, A. (2021). Using synthetic controls: Feasibility, data requirements, and
  methodological aspects. *Journal of Economic Literature*, 59(2), 391–425.
"""
function synth_conformal_inference(r::AugmentedSCEstimate; level::Real=0.95,
                                   type::Symbol=:block, q::Real=1,
                                   grid_size::Integer=50, ns::Integer=1000,
                                   rng::AbstractRNG=Random.default_rng())
    _sc_check_level(level)
    type in (:block, :iid) || throw(ArgumentError("type must be :block or :iid"))
    grid_size >= 2 || throw(ArgumentError("grid_size must be at least 2"))
    alpha = 1 - level
    p = r.panel
    n_pre = r.n_pre
    T = size(p.Y, 2)
    trt = p.adoption .> 0
    post = r.att_path[(n_pre + 1):T]
    post_sd = sqrt(mean(abs2, post))
    T1 = T - n_pre
    lower = Vector{Union{Missing,Float64}}(missing, T1)
    upper = Vector{Union{Missing,Float64}}(missing, T1)
    truncated = falses(T1)
    pv = zeros(T1)
    for j in 1:T1
        Xj = hcat(p.Y[:, 1:n_pre], p.Y[:, n_pre + j])
        est = post[j]
        grid = vcat(collect(range(est - 2 * post_sd, est + 2 * post_sd;
                                  length=grid_size)), 0.0)
        ps = [_sc_conformal_pvalue(Xj, trt, r.opts, h, 1, type, q, ns, rng) for h in grid]
        # p-values are multiples of 1/(T0+1): compare with a tolerance so that e.g.
        # p = 1/20 is accepted at level 0.95 despite 1 - 0.95 > 0.05 in floating point
        accept = ps .>= alpha - 1e-10
        acc = grid[accept]
        if !isempty(acc)
            lower[j], upper[j] = minimum(acc), maximum(acc)
            truncated[j] = accept[1] || accept[grid_size]
        end
        pv[j] = ps[end]
    end
    joint_p = _sc_conformal_pvalue(p.Y, trt, r.opts, 0.0, T1, type, q, ns, rng)
    per = DataFrame(time=p.times[(n_pre + 1):T], estimate=post, lower=lower,
                    upper=upper, truncated=truncated, p_value=pv)
    stat = _sc_conformal_stat(_sc_ascm_core(p.Y, zeros(size(p.Y, 1), 0), trt,
                                            r.opts).gap[(n_pre + 1):T], q)
    joint = DiagnosticTest("Conformal test of no effect in all post-treatment periods",
                           "the treatment effect is zero in every post-treatment period",
                           stat, joint_p;
                           method="conformal inference, $(type) permutations " *
                                  "(Chernozhukov, Wüthrich & Zhu 2021)",
                           note="Assumes " * (type === :block ? "stationary, weakly " *
                                "dependent residuals" : "exchangeable residuals") *
                                " over time.",
                           details=(n_pre=n_pre, n_post=T1))
    return (per_period=per, joint=joint)
end

function synth_weights(r::AugmentedSCEstimate)
    return DataFrame(unit=r.panel.units[1:r.panel.n_control], weight=r.weights,
                     scm_weight=r.scm_weights)
end

function synth_gaps(r::AugmentedSCEstimate)
    T = length(r.att_path)
    return DataFrame(time=r.panel.times, treated=r.treated_path,
                     synthetic=r.treated_path .- r.att_path, gap=r.att_path,
                     post=(1:T) .> r.n_pre)
end

function Base.show(io::IO, ::MIME"text/plain", r::AugmentedSCEstimate)
    p = r.panel
    println(io, method_name(r), " — estimand: ", estimand(r))
    label = r.se_method === :placebo ?
            "placebo, $(length(r.replicate_estimates)) replications, λ fixed" :
            "jackknife over units, λ fixed"
    _sc_show_se_line(io, r.att, r.se, label)
    r.lambda === nothing || @printf(io, "  Ridge penalty λ: %.4g%s\n", r.lambda,
                                    r.cv === nothing ? " (user-supplied)" :
                                    " (cross-validated)")
    @printf(io, "  Units: %d never-treated, %d treated; pre-periods %d, post %d\n",
            p.n_control, size(p.Y, 1) - p.n_control, r.n_pre, size(p.Y, 2) - r.n_pre)
    @printf(io, "  Pre-treatment RMSE of the fit: %.4f\n",
            sqrt(mean(abs2, r.att_path[1:r.n_pre])))
    @printf(io, "  Sum of negative weights: %.4f\n", sum(min.(r.weights, 0)))
end

Base.show(io::IO, r::AugmentedSCEstimate) =
    print(io, method_name(r), "(", @sprintf("%.4g", r.att),
          r.se === nothing ? ")" : @sprintf(" (se %.3g))", r.se))
