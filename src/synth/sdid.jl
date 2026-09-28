# Synthetic difference-in-differences (Arkhangelsky, Athey, Hirshberg, Imbens & Wager
# 2021), following the authors' R package `synthdid` exactly for block designs:
# unit weights ω and time weights λ with ridge regularisation ζ, intercepts, Frank–Wolfe
# with sparsification, and the SC and DiD estimators obtained from the same machinery.
# Staggered adoption is handled cohort by cohort (never-treated units + one adoption
# cohort) and aggregated with weights proportional to treated unit-periods, as in the
# paper's appendix and Clarke, Pailañir, Athey & Imbens (2023).

struct _sc_SdidOpts
    zeta_omega::Float64
    zeta_lambda::Float64
    omega_intercept::Bool
    lambda_intercept::Bool
    update_omega::Bool
    update_lambda::Bool
    min_decrease::Float64
    max_iter::Int
    sparsify::Bool
    max_iter_pre_sparsify::Int
end

function _sc_with_updates(o::_sc_SdidOpts, update::Bool)
    update && return o
    return _sc_SdidOpts(o.zeta_omega, o.zeta_lambda, o.omega_intercept,
                        o.lambda_intercept, false, false, o.min_decrease, o.max_iter,
                        o.sparsify, o.max_iter_pre_sparsify)
end

struct _sc_CohortFit
    adoption::Int                 # column index of the first treated period
    rows::Vector{Int}             # treated rows of the panel
    tau::Float64
    omega::Vector{Float64}        # weights on the panel's never-treated rows
    lambda::Vector{Float64}       # weights on periods 1:(adoption - 1)
    beta::Vector{Float64}         # covariate coefficients (optimized method)
    opts::_sc_SdidOpts
    noise_level::Float64
    treated_path::Vector{Float64} # average (covariate-adjusted) outcome of the cohort
    gap::Vector{Float64}          # synthdid effect curve extended to all periods
end

"""
    SyntheticDiDEstimate <: CausalEstimate

Result of [`synthetic_did`](@ref): a synthetic difference-in-differences (or `synthdid`
synthetic control or DiD) estimate of the average treatment effect on the treated.

The object implements the `CausalEstimate` interface (`coef`, `vcov`, `stderror`,
`confint`, `coeftable`, `nobs`) with a single coefficient `"ATT"`. Intervals use the
normal approximation with the placebo, bootstrap or jackknife variance, as in
Arkhangelsky et al. (2021). `vcov` throws when `se_method = :none`. Under staggered
adoption the ATT is the weighted average of cohort-specific estimates, which
[`synth_cohorts`](@ref) lists.

# Fields
- `att::Float64`: estimated average treatment effect on the treated.
- `se::Union{Nothing,Float64}`: standard error (`nothing` when `se_method = :none`).
- `se_method::Symbol`: `:placebo`, `:bootstrap`, `:jackknife` or `:none`.
- `replications::Int`: number of placebo or bootstrap replications (0 otherwise).
- `replicate_estimates::Vector{Float64}`: the replicate estimates behind `se`.
- `method::Symbol`: `:sdid`, `:sc` (`synthdid`'s penalised SC) or `:did`.
- `covariate_method::Symbol`: `:none`, `:optimized` or `:projected`.
- `beta::Vector{Float64}`: covariate coefficients of the projected method (for the
  optimized method they are stored with each cohort).
- `cohorts::Vector`: per-adoption-cohort fits (weights, effect, regularisation).
- `panel::SynthPanel`: the data used.

Inspect the fit with [`synth_weights`](@ref), [`synth_time_weights`](@ref),
[`synth_gaps`](@ref) and [`synth_cohorts`](@ref).
"""
struct SyntheticDiDEstimate{P<:SynthPanel} <: CausalEstimate
    att::Float64
    se::Union{Nothing,Float64}
    se_method::Symbol
    replications::Int
    replicate_estimates::Vector{Float64}
    method::Symbol
    covariate_method::Symbol
    beta::Vector{Float64}
    cohorts::Vector{_sc_CohortFit}
    panel::P
end

StatsAPI.coef(r::SyntheticDiDEstimate) = [r.att]
StatsAPI.coefnames(::SyntheticDiDEstimate) = ["ATT"]
StatsAPI.nobs(r::SyntheticDiDEstimate) = length(r.panel.Y)
function StatsAPI.vcov(r::SyntheticDiDEstimate)
    r.se === nothing && throw(ArgumentError("no standard error was computed; re-run " *
                                            "with se_method = :placebo, :bootstrap or " *
                                            ":jackknife"))
    return fill(r.se^2, 1, 1)
end
estimand(::SyntheticDiDEstimate) = "ATT (average over treated units and post periods)"
method_name(r::SyntheticDiDEstimate) =
    r.method === :sdid ? "Synthetic difference-in-differences" :
    r.method === :sc ? "Synthetic control (synthdid penalised variant)" :
    "Difference-in-differences (synthdid)"

# ---------------------------------------------------------------------------------------
# Core block-design estimator (synthdid::synthdid_estimate)
# ---------------------------------------------------------------------------------------

function _sc_sdid_default_opts(Y::AbstractMatrix, N0::Integer, T0::Integer,
                               method::Symbol; zeta_omega=nothing, zeta_lambda=nothing,
                               sparsify::Bool=true, max_iter::Integer=10_000)
    T0 >= 2 || throw(ArgumentError("synthetic_did: at least two pre-treatment periods " *
                                   "are needed (found $T0)"))
    N0 >= 2 || throw(ArgumentError("synthetic_did: at least two never-treated units " *
                                   "are needed (found $N0)"))
    N1 = size(Y, 1) - N0
    T1 = size(Y, 2) - T0
    noise = std(vec(diff(Y[1:N0, 1:T0]; dims=2)))
    eta_omega = method === :sc ? 1e-6 : (N1 * T1)^(1 / 4)
    eta_lambda = 1e-6
    zo = zeta_omega === nothing ? eta_omega * noise : Float64(zeta_omega)
    zl = zeta_lambda === nothing ? eta_lambda * noise : Float64(zeta_lambda)
    update_omega = method !== :did
    update_lambda = method === :sdid
    omega_intercept = method !== :sc
    opts = _sc_SdidOpts(zo, zl, omega_intercept, true, update_omega, update_lambda,
                        1e-5 * noise, max_iter, sparsify, 100)
    omega0 = method === :did ? fill(1 / N0, N0) : nothing
    lambda0 = method === :sdid ? nothing : method === :sc ? zeros(T0) : fill(1 / T0, T0)
    return opts, noise, omega0, lambda0
end

function _sc_synthdid_core(Y::AbstractMatrix, N0::Integer, T0::Integer, X,
                           opts::_sc_SdidOpts; omega=nothing, lambda=nothing)
    N, T = size(Y)
    N1, T1 = N - N0, T - T0
    K = X === nothing ? 0 : size(X, 3)
    beta = zeros(K)
    if K == 0
        Yc = _sc_collapsed_form(Y, N0, T0)
        pre_iter = opts.sparsify ? opts.max_iter_pre_sparsify : opts.max_iter
        if opts.update_lambda
            lam, _ = _sc_weight_fw(Yc[1:N0, :], opts.zeta_lambda;
                                   intercept=opts.lambda_intercept, init=lambda,
                                   min_decrease=opts.min_decrease, max_iter=pre_iter)
            if opts.sparsify
                lam, _ = _sc_weight_fw(Yc[1:N0, :], opts.zeta_lambda;
                                       intercept=opts.lambda_intercept,
                                       init=_sc_sparsify(lam),
                                       min_decrease=opts.min_decrease,
                                       max_iter=opts.max_iter)
            end
            lambda = lam
        end
        if opts.update_omega
            Yo = Matrix(transpose(Yc[:, 1:T0]))
            om, _ = _sc_weight_fw(Yo, opts.zeta_omega; intercept=opts.omega_intercept,
                                  init=omega, min_decrease=opts.min_decrease,
                                  max_iter=pre_iter)
            if opts.sparsify
                om, _ = _sc_weight_fw(Yo, opts.zeta_omega;
                                      intercept=opts.omega_intercept,
                                      init=_sc_sparsify(om),
                                      min_decrease=opts.min_decrease,
                                      max_iter=opts.max_iter)
            end
            omega = om
        end
    else
        Yc = _sc_collapsed_form(Y, N0, T0)
        Xc = cat((_sc_collapsed_form(view(X, :, :, k), N0, T0) for k in 1:K)...; dims=3)
        fit = _sc_weight_fw_covariates(Yc, Xc; zeta_lambda=opts.zeta_lambda,
                                       zeta_omega=opts.zeta_omega,
                                       lambda_intercept=opts.lambda_intercept,
                                       omega_intercept=opts.omega_intercept,
                                       min_decrease=opts.min_decrease,
                                       max_iter=opts.max_iter, lambda=lambda,
                                       omega=omega, update_lambda=opts.update_lambda,
                                       update_omega=opts.update_omega)
        lambda, omega, beta = fit.lambda, fit.omega, fit.beta
    end
    Yadj = K == 0 ? Y : Y .- _sc_contract3(X, beta)
    uvec = vcat(-omega, fill(1 / N1, N1))
    tvec = vcat(-lambda, fill(1 / T1, T1))
    tau = dot(uvec, Yadj * tvec)
    return (tau=tau, omega=omega, lambda=lambda, beta=beta, Yadj=Yadj)
end

# ---------------------------------------------------------------------------------------
# Covariates: projection method (Kranz 2022; `method(projected)` in Stata sdid)
# ---------------------------------------------------------------------------------------

# Two-way within regression of Y on X using the given (never-treated) rows; exact for
# a balanced sub-panel. Returns β.
function _sc_project_beta(Y::AbstractMatrix, X::AbstractArray{<:Real,3},
                          rows::AbstractVector{<:Integer})
    K = size(X, 3)
    dm(M) = M .- mean(M; dims=1) .- mean(M; dims=2) .+ mean(M)
    y = vec(dm(Y[rows, :]))
    Z = hcat((vec(dm(X[rows, :, k])) for k in 1:K)...)
    rank(Z) == K || throw(ArgumentError("synthetic_did: covariates are collinear with " *
                                        "unit and time fixed effects among " *
                                        "never-treated units"))
    return Z \ y
end

# ---------------------------------------------------------------------------------------
# Cohort-wise estimation and replicates
# ---------------------------------------------------------------------------------------

function _sc_cohort_curve(Yadj, N0, T0, omega, lambda)
    N1 = size(Yadj, 1) - N0
    tau_sc = vec(transpose(vcat(-omega, fill(1 / N1, N1))) * Yadj)
    treated = vec(mean(Yadj[(N0 + 1):end, :]; dims=1))
    gap = tau_sc .- dot(tau_sc[1:T0], lambda)
    return treated, gap
end

# Estimate one cohort from scratch on the (never-treated + cohort) sub-panel.
function _sc_fit_cohort(Y, X, n_control, rows, a, method; zeta_omega, zeta_lambda,
                        sparsify, max_iter)
    sub = vcat(1:n_control, rows)
    Ya = Y[sub, :]
    Xa = X === nothing ? nothing : X[sub, :, :]
    opts, noise, omega0, lambda0 = _sc_sdid_default_opts(Ya, n_control, a - 1, method;
                                                         zeta_omega, zeta_lambda,
                                                         sparsify, max_iter)
    res = _sc_synthdid_core(Ya, n_control, a - 1, Xa, opts; omega=omega0,
                            lambda=lambda0)
    treated, gap = _sc_cohort_curve(res.Yadj, n_control, a - 1, res.omega, res.lambda)
    return _sc_CohortFit(a, collect(rows), res.tau, res.omega, res.lambda, res.beta,
                         opts, noise, treated, gap)
end

_sc_cohort_weight(n_treated, a, T) = n_treated * (T - a + 1)

# Aggregate estimate on a replicate sample. `ctrl` are panel rows used as controls
# (original never-treated rows, possibly repeated / permuted); `trt` are rows used as
# treated with adoption columns `trt_adopt`. Weights are warm-started from the
# original cohort fits (as synthdid does) and not updated when `update = false`.
function _sc_sdid_replicate(Y, X, cohorts::Vector{_sc_CohortFit}, ctrl::Vector{Int},
                            trt::Vector{Int}, trt_adopt::Vector{Int}; update::Bool=true)
    T = size(Y, 2)
    num = 0.0
    den = 0.0
    for c in cohorts
        sel = trt[trt_adopt .== c.adoption]
        isempty(sel) && continue
        sub = vcat(ctrl, sel)
        Xa = X === nothing ? nothing : X[sub, :, :]
        opts = _sc_with_updates(c.opts, update)
        res = _sc_synthdid_core(Y[sub, :], length(ctrl), c.adoption - 1, Xa, opts;
                                omega=_sc_sum_normalize(c.omega[ctrl]), lambda=c.lambda)
        w = _sc_cohort_weight(length(sel), c.adoption, T)
        num += w * res.tau
        den += w
    end
    den > 0 || error("replicate without treated units")
    return num / den
end

function _sc_sdid_adjusted_Y(Y, X, covariate_method, ctrl)
    covariate_method === :projected || return Y, nothing, Float64[]
    beta = _sc_project_beta(Y, X, ctrl)
    return Y .- _sc_contract3(X, beta), nothing, beta
end

function _sc_sdid_theta(Y, Xraw, covariate_method, cohorts, ctrl, trt, trt_adopt;
                        update::Bool=true)
    if covariate_method === :projected
        Yadj, _, _ = _sc_sdid_adjusted_Y(Y, Xraw, covariate_method, ctrl)
        return _sc_sdid_replicate(Yadj, nothing, cohorts, ctrl, trt, trt_adopt;
                                  update=update)
    end
    X = covariate_method === :optimized ? Xraw : nothing
    return _sc_sdid_replicate(Y, X, cohorts, ctrl, trt, trt_adopt; update=update)
end

function _sc_sdid_placebo(Y, X, cm, cohorts, n_control, adoption, B, rng)
    n_treated = length(adoption) - n_control
    n_control > n_treated ||
        throw(ArgumentError("synthetic_did: the placebo standard error needs more " *
                            "never-treated units ($n_control) than treated units " *
                            "($n_treated)"))
    trt_adopt = adoption[(n_control + 1):end]
    seeds = task_seeds(rng, B)
    draws = Vector{Float64}(undef, B)
    n0 = n_control - n_treated
    Threads.@threads for b in 1:B
        r = Random.Xoshiro(seeds[b])
        perm = randperm(r, n_control)
        draws[b] = _sc_sdid_theta(Y, X, cm, cohorts, perm[1:n0], perm[(n0 + 1):end],
                                  trt_adopt)
    end
    return draws
end

function _sc_sdid_bootstrap(Y, X, cm, cohorts, n_control, adoption, B, rng)
    N = length(adoption)
    N - n_control >= 2 ||
        throw(ArgumentError("synthetic_did: the bootstrap standard error needs at " *
                            "least two treated units; use se_method = :placebo"))
    seeds = task_seeds(rng, B)
    draws = Vector{Float64}(undef, B)
    Threads.@threads for b in 1:B
        r = Random.Xoshiro(seeds[b])
        while true
            ind = sort(rand(r, 1:N, N))
            ctrl = ind[ind .<= n_control]
            trt = ind[ind .> n_control]
            (isempty(ctrl) || isempty(trt)) && continue
            draws[b] = _sc_sdid_theta(Y, X, cm, cohorts, ctrl, trt, adoption[trt])
            break
        end
    end
    return draws
end

function _sc_sdid_jackknife(Y, X, cm, cohorts, n_control, adoption)
    N = length(adoption)
    for c in cohorts
        length(c.rows) >= 2 ||
            throw(ArgumentError("synthetic_did: the jackknife standard error needs at " *
                                "least two treated units in every adoption cohort; " *
                                "use se_method = :placebo or :bootstrap"))
        count(!iszero, c.omega) >= 2 ||
            throw(ArgumentError("synthetic_did: the jackknife standard error is not " *
                                "defined when only one control unit has positive weight"))
    end
    u = Vector{Float64}(undef, N)
    trt_all = collect((n_control + 1):N)
    Threads.@threads for i in 1:N
        ctrl = [j for j in 1:n_control if j != i]
        trt = [j for j in trt_all if j != i]
        u[i] = _sc_sdid_theta(Y, X, cm, cohorts, ctrl, trt, adoption[trt]; update=false)
    end
    return u, sqrt((N - 1) / N * sum(abs2, u .- mean(u)))
end

_sc_replicate_se(draws) = sqrt((length(draws) - 1) / length(draws)) * std(draws)

# ---------------------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------------------

"""
    synthetic_did(data, outcome, treatment, unit, time; covariates=Symbol[], kwargs...)
    synthetic_did(panel::SynthPanel; method=:sdid, se_method=:placebo,
                  replications=200, covariate_method=:optimized, zeta_omega=nothing,
                  zeta_lambda=nothing, sparsify=true, max_iter=10_000,
                  rng=Random.default_rng()) -> SyntheticDiDEstimate

Synthetic difference-in-differences (SDID) estimator of Arkhangelsky, Athey, Hirshberg,
Imbens and Wager (2021), reproducing the R package `synthdid`. With `method = :sc` or
`:did` the same machinery gives `synthdid`'s penalised synthetic control and
difference-in-differences estimators.

The estimand is the average treatment effect on the treated over treated unit-periods,
``\\tau = \\frac{1}{N_1 T_1}\\sum_{i \\text{ treated}}\\sum_{t > T_0}
(Y_{it}(1) - Y_{it}(0))``. The missing untreated outcomes are imputed from never-treated
units. Difference-in-differences does this by assuming parallel trends for the
unweighted averages of units. Synthetic control reweights control units to match the
treated unit's pre-treatment path, without an intercept. SDID combines the two. Unit
weights ``\\hat\\omega`` make the weighted control units' pre-treatment trend parallel
to that of the treated units, up to a constant. Time weights ``\\hat\\lambda`` make the
weighted pre-treatment periods predictive of the post-treatment periods for control
units. The estimate is a weighted two-way fixed-effects regression,
```math
(\\hat\\tau, \\hat\\mu, \\hat\\alpha, \\hat\\beta) = \\arg\\min \\sum_{i,t}
    \\big(Y_{it} - \\mu - \\alpha_i - \\beta_t - W_{it}\\tau\\big)^2
    \\hat\\omega_i \\hat\\lambda_t,
```
equivalently ``\\hat\\tau = (\\bar y_{\\text{tr,post}} - \\sum_t \\hat\\lambda_t
\\bar y_{\\text{tr},t}) - \\sum_i \\hat\\omega_i (\\bar y_{i,\\text{post}} - \\sum_t
\\hat\\lambda_t y_{it})``. The theoretical justification is a latent factor model
``Y_{it}(0) = \\alpha_i + \\beta_t + \\gamma_i^\\top \\upsilon_t + \\varepsilon_{it}``.
Arkhangelsky et al. (2021) show that the double weighting removes bias from the
interactive component under conditions on the number of units and periods and on the
quality of the weights. The assumption that treatment adoption is unrelated to the
idiosyncratic errors cannot be tested. The pre-treatment fit of the weighted controls,
visible through [`synth_gaps`](@ref), is the main diagnostic.

The weights solve the regularised quadratic programmes of the paper by Frank–Wolfe
iterations with a sparsification step, as in `synthdid`. The default ridge penalties
are ``\\zeta_\\omega = (N_1 T_1)^{1/4}\\hat\\sigma`` (``10^{-6}\\hat\\sigma`` for
`method = :sc`) and ``\\zeta_\\lambda = 10^{-6}\\hat\\sigma``, where ``\\hat\\sigma`` is
the standard deviation of first differences of the never-treated outcomes before
treatment. Staggered adoption is handled cohort by cohort.
Each adoption cohort is compared with the never-treated units over all periods, and the
cohort estimates are averaged with weights proportional to the number of treated
unit-periods (Arkhangelsky et al. 2021, appendix; Clarke, Pailañir, Athey & Imbens
2024). Not-yet-treated units are never used as controls. Time-varying covariates enter
either jointly with the weights, as in `synthdid` (`:optimized`), or by first
residualising outcomes on covariates in a two-way fixed-effects regression on
never-treated units (`:projected`, Kranz 2021).

Inference uses the normal approximation with one of the variance estimators of
Arkhangelsky et al. (2021). The *placebo* variance (their Algorithm 4) re-assigns the
treated units' adoption pattern to randomly chosen never-treated units. It is the only
option with a single treated unit, and it assumes that treated and control units have
the same error distribution (homoskedasticity across units). The *bootstrap* (their
Algorithm 2) resamples units and needs several treated units. The *jackknife* (their
Algorithm 3) leaves out one unit at a time with the weights held fixed; it needs at
least two treated units per cohort. With one or very few
treated units, no method gives reliable inference, and the interval should be read as
indicative. Compared with [`synthetic_control`](@ref), SDID allows a level difference
between treated units and controls and averages over a weighted pre-period. Compared
with a two-way fixed-effects regression, it down-weights dissimilar controls and
periods. [`augmented_synthetic_control`](@ref) and [`matrix_completion`](@ref) are
alternatives under similar factor-model assumptions. Report the unit and time weights
([`synth_weights`](@ref), [`synth_time_weights`](@ref)), the pre-treatment fit and,
under staggered adoption, the cohort estimates ([`synth_cohorts`](@ref)).

# Arguments
- `data`: long panel (see [`synth_panel`](@ref)), with columns `outcome`, `treatment`,
  `unit` and `time`; or
- `panel::SynthPanel`: a prepared panel.

# Keywords
- `covariates::Vector{Symbol}=Symbol[]`: time-varying covariates (long-data method
  only); they must have no missing values.
- `method::Symbol=:sdid`: `:sdid`, `:sc` (`synthdid`'s penalised synthetic control, no
  time weights or intercept) or `:did` (equal weights).
- `se_method::Symbol=:placebo`: `:placebo`, `:bootstrap` (at least 2 treated units),
  `:jackknife` (at least 2 treated units per cohort) or `:none`. The placebo method
  needs more never-treated than treated units.
- `replications::Integer=200`: placebo or bootstrap replications (the `synthdid`
  default).
- `covariate_method::Symbol=:optimized`: `:optimized` (coefficients estimated jointly
  with the weights, as in `synthdid`) or `:projected` (Kranz 2021).
- `zeta_omega=nothing`, `zeta_lambda=nothing`: override the regularisation levels of the
  unit and time weights.
- `sparsify::Bool=true`: `synthdid`'s sparsification step (weights below a quarter of
  the largest weight are set to zero and the problem is re-solved).
- `max_iter::Integer=10_000`: maximum number of Frank–Wolfe iterations.
- `rng::AbstractRNG=Random.default_rng()`: random number generator for placebo and
  bootstrap draws.

# Returns
- [`SyntheticDiDEstimate`](@ref).

# Examples
```julia
using DrSnow, CSV, DataFrames, StableRNGs
prop99 = CSV.read(pkgdir(DrSnow, "test", "validation", "synth", "prop99.csv"), DataFrame)
r = synthetic_did(prop99, :PacksPerCapita, :treated, :State, :Year; rng=StableRNG(1))
coef(r), stderror(r), confint(r)
first(sort(synth_weights(r), :weight; rev=true), 5)
synth_time_weights(r)
synthetic_did(prop99, :PacksPerCapita, :treated, :State, :Year; method=:did,
              rng=StableRNG(1))
```

# References
- Arkhangelsky, D., Athey, S., Hirshberg, D. A., Imbens, G. W., & Wager, S. (2021).
  Synthetic difference-in-differences. *American Economic Review*, 111(12), 4088–4118.
- Clarke, D., Pailañir, D., Athey, S., & Imbens, G. (2024). On synthetic
  difference-in-differences and related estimation methods in Stata. *The Stata
  Journal*, 24(4), 557–598.
- Doudchenko, N., & Imbens, G. W. (2016). Balancing, regression,
  difference-in-differences and synthetic control methods: A synthesis. NBER Working
  Paper 22791.
- Abadie, A., Diamond, A., & Hainmueller, J. (2010). Synthetic control methods for
  comparative case studies: Estimating the effect of California's tobacco control
  program. *Journal of the American Statistical Association*, 105(490), 493–505.
- Kranz, S. (2021). Synthetic difference-in-differences with time-varying covariates.
  Mimeo.
"""
function synthetic_did(data, outcome::Symbol, treatment::Symbol, unit::Symbol,
                       time::Symbol; covariates::Vector{Symbol}=Symbol[], kwargs...)
    panel = synth_panel(data, outcome, treatment, unit, time; covariates=covariates)
    return synthetic_did(panel; kwargs...)
end

function synthetic_did(panel::SynthPanel; method::Symbol=:sdid,
                       se_method::Symbol=:placebo, replications::Integer=200,
                       covariate_method::Symbol=:optimized, zeta_omega=nothing,
                       zeta_lambda=nothing, sparsify::Bool=true,
                       max_iter::Integer=10_000, rng::AbstractRNG=Random.default_rng())
    method in (:sdid, :sc, :did) ||
        throw(ArgumentError("synthetic_did: method must be :sdid, :sc or :did"))
    se_method in (:placebo, :bootstrap, :jackknife, :none) ||
        throw(ArgumentError("synthetic_did: se_method must be :placebo, :bootstrap, " *
                            ":jackknife or :none"))
    covariate_method in (:optimized, :projected) ||
        throw(ArgumentError("synthetic_did: covariate_method must be :optimized or " *
                            ":projected"))
    se_method in (:placebo, :bootstrap) && replications < 2 &&
        throw(ArgumentError("synthetic_did: replications must be at least 2"))
    Y = panel.Y
    n_control = panel.n_control
    has_cov = !isempty(panel.covariates)
    Xraw = has_cov ? _sc_complete_covariates(panel, "synthetic_did") : nothing
    cm = has_cov ? covariate_method : :none
    ctrl0 = collect(1:n_control)
    beta = Float64[]
    Yfit, Xfit = Y, (cm === :optimized ? Xraw : nothing)
    if cm === :projected
        beta = _sc_project_beta(Y, Xraw, ctrl0)
        Yfit = Y .- _sc_contract3(Xraw, beta)
    end
    cohorts = _sc_CohortFit[]
    for a in _sc_adoption_indices(panel)
        rows = findall(==(a), panel.adoption)
        push!(cohorts, _sc_fit_cohort(Yfit, Xfit, n_control, rows, a, method;
                                      zeta_omega, zeta_lambda, sparsify, max_iter))
    end
    T = size(Y, 2)
    wts = [_sc_cohort_weight(length(c.rows), c.adoption, T) for c in cohorts]
    att = sum(wts .* [c.tau for c in cohorts]) / sum(wts)

    se = nothing
    draws = Float64[]
    reps = 0
    Xrep = cm === :none ? nothing : Xraw
    if se_method === :placebo
        draws = _sc_sdid_placebo(Y, Xrep, cm, cohorts, n_control, panel.adoption,
                                 replications, rng)
        se = _sc_replicate_se(draws)
        reps = replications
    elseif se_method === :bootstrap
        draws = _sc_sdid_bootstrap(Y, Xrep, cm, cohorts, n_control, panel.adoption,
                                   replications, rng)
        se = _sc_replicate_se(draws)
        reps = replications
    elseif se_method === :jackknife
        draws, se = _sc_sdid_jackknife(Y, Xrep, cm, cohorts, n_control, panel.adoption)
    end
    return SyntheticDiDEstimate(att, se, se_method, reps, draws, method, cm, beta,
                                cohorts, panel)
end

"""
    synth_cohorts(r::SyntheticDiDEstimate) -> DataFrame

Cohort-level results of a synthetic difference-in-differences estimate.

Under staggered adoption, [`synthetic_did`](@ref) estimates one SDID effect per adoption
cohort, each against the never-treated units. The overall ATT averages these effects
with weights proportional to the number of treated unit-periods,
``n_{\\text{treated}} \\times n_{\\text{post}}`` (Arkhangelsky et al. 2021, appendix;
Clarke, Pailañir, Athey & Imbens 2024). The table shows that decomposition. It also
shows the regularisation of each cohort's weights, since the noise level
``\\hat\\sigma`` and the penalties are computed from each cohort's own pre-treatment
period. Heterogeneous cohort effects are informative in their own right, since the ATT
weights later cohorts less because they have fewer post-treatment periods. Block
designs have one row.

# Arguments
- `r::SyntheticDiDEstimate`: result of [`synthetic_did`](@ref).

# Returns
- `DataFrame` with one row per cohort and columns `adoption` (first treated period),
  `n_treated`, `n_post` (post-treatment periods), `estimate` (cohort ATT), `weight`
  (share in the overall ATT), `zeta_omega`, `zeta_lambda` and `noise_level`.

# Examples
```julia
using DrSnow, CSV, DataFrames
prop99 = CSV.read(pkgdir(DrSnow, "test", "validation", "synth", "prop99.csv"), DataFrame)
r = synthetic_did(prop99, :PacksPerCapita, :treated, :State, :Year; se_method=:none)
synth_cohorts(r)
```

# References
- Arkhangelsky, D., Athey, S., Hirshberg, D. A., Imbens, G. W., & Wager, S. (2021).
  Synthetic difference-in-differences. *American Economic Review*, 111(12), 4088–4118.
- Clarke, D., Pailañir, D., Athey, S., & Imbens, G. (2024). On synthetic
  difference-in-differences and related estimation methods in Stata. *The Stata
  Journal*, 24(4), 557–598.
"""
function synth_cohorts(r::SyntheticDiDEstimate)
    T = size(r.panel.Y, 2)
    w = [_sc_cohort_weight(length(c.rows), c.adoption, T) for c in r.cohorts]
    return DataFrame(adoption=[r.panel.times[c.adoption] for c in r.cohorts],
                     n_treated=[length(c.rows) for c in r.cohorts],
                     n_post=[T - c.adoption + 1 for c in r.cohorts],
                     estimate=[c.tau for c in r.cohorts], weight=w ./ sum(w),
                     zeta_omega=[c.opts.zeta_omega for c in r.cohorts],
                     zeta_lambda=[c.opts.zeta_lambda for c in r.cohorts],
                     noise_level=[c.noise_level for c in r.cohorts])
end

function synth_weights(r::SyntheticDiDEstimate)
    p = r.panel
    ctrl = p.units[1:p.n_control]
    if length(r.cohorts) == 1
        return DataFrame(unit=ctrl, weight=r.cohorts[1].omega)
    end
    return vcat([DataFrame(cohort=fill(p.times[c.adoption], p.n_control), unit=ctrl,
                           weight=c.omega) for c in r.cohorts]...)
end

function synth_time_weights(r::SyntheticDiDEstimate)
    p = r.panel
    if length(r.cohorts) == 1
        c = r.cohorts[1]
        return DataFrame(time=p.times[1:(c.adoption - 1)], weight=c.lambda)
    end
    return vcat([DataFrame(cohort=fill(p.times[c.adoption], c.adoption - 1),
                           time=p.times[1:(c.adoption - 1)], weight=c.lambda)
                 for c in r.cohorts]...)
end

function synth_gaps(r::SyntheticDiDEstimate)
    p = r.panel
    T = length(p.times)
    frames = map(r.cohorts) do c
        DataFrame(cohort=fill(p.times[c.adoption], T), time=p.times,
                  treated=c.treated_path, synthetic=c.treated_path .- c.gap,
                  gap=c.gap, post=(1:T) .>= c.adoption)
    end
    df = vcat(frames...)
    length(r.cohorts) == 1 && select!(df, Not(:cohort))
    return df
end

function Base.show(io::IO, ::MIME"text/plain", r::SyntheticDiDEstimate)
    p = r.panel
    println(io, method_name(r), " — estimand: ", estimand(r))
    label = r.se_method === :jackknife ? "jackknife" :
            "$(r.se_method), $(r.replications) replications"
    _sc_show_se_line(io, r.att, r.se, label)
    @printf(io, "  Units: %d never-treated, %d treated; periods: %d\n", p.n_control,
            size(p.Y, 1) - p.n_control, size(p.Y, 2))
    if length(r.cohorts) == 1
        c = r.cohorts[1]
        @printf(io, "  Effective number of controls (1/Σω²): %.2f\n",
                1 / sum(abs2, c.omega))
        r.method === :sdid &&
            @printf(io, "  Effective number of pre-periods (1/Σλ²): %.2f\n",
                    1 / sum(abs2, c.lambda))
    else
        println(io, "  Staggered adoption: ", length(r.cohorts), " cohorts " *
                    "(see synth_cohorts)")
    end
    r.covariate_method === :none ||
        println(io, "  Covariates: ", join(p.covariates, ", "), " (",
                r.covariate_method, ")")
end

Base.show(io::IO, r::SyntheticDiDEstimate) =
    print(io, method_name(r), "(", @sprintf("%.4g", r.att),
          r.se === nothing ? ")" : @sprintf(" (se %.3g))", r.se))
