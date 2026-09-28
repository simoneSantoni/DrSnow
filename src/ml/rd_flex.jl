# Flexible (machine-learning) covariate adjustment in regression discontinuity designs
# (Noack, Olma & Rothe 2024), as in DoubleML's `RDFlex`: the outcome (and, in fuzzy
# designs, the treatment) is adjusted by a cross-fitted prediction η(X) from the
# covariates, and the adjusted variable is analysed by local linear RD with
# robust bias-corrected inference (`rd_estimate`). It lives in the ml area because it
# needs the learners and cross-fitting; it is documented with the RD methods.

"""
    RDFlexEstimate <: CausalEstimate

Result of [`rd_flex`](@ref): a local linear RD estimate on outcomes (and treatments)
adjusted by cross-fitted machine-learning predictions from covariates.

The headline follows the same convention as [`RDEstimate`](@ref) and `rdrobust`:
`coef` is the conventional estimate (`tau_conventional`), while `stderror`, `pvalues`
and `confint` give robust bias-corrected inference, with the interval centred on the
bias-corrected estimate (`tau_bias_corrected`); [`rd_inference_table`](@ref) shows all
three rows. With `n_rep > 1` repetitions of the cross-fitting, point estimates are
medians over repetitions ``\\tilde\\theta = \\text{median}_r\\,\\hat\\theta_r``, and
variances are ``\\text{median}_r(\\text{se}_r^2 + (\\hat\\theta_r - \\tilde\\theta)^2)``,
which adds the variability due to the random fold split (Chernozhukov et al. 2018).

# Fields
- `design::Symbol`: `:sharp` or `:fuzzy`.
- `tau_conventional`, `tau_bias_corrected`: aggregated point estimates.
- `se_conventional`, `se_robust`: aggregated standard errors.
- `fits::Vector{RDEstimate}`: the final local polynomial fit of each repetition, with
  its bandwidths, effective sample sizes and first stage.
- `all_coef`, `all_se`: `3 × n_rep` matrices of estimates and standard errors per
  repetition (rows: conventional, bias-corrected, robust).
- `h_fs::Float64`: initial bandwidth of the kernel weights used to fit the nuisance
  learners.
- `h`, `b`: final main and pilot bandwidths per repetition (the larger of the left and
  right values).
- `eta_y::Matrix{Float64}`, `eta_d`: cross-fitted adjustments ``\\hat\\eta(W)`` of the
  outcome and, in fuzzy designs, of the treatment, per observation (complete cases in
  row order) and repetition.
- `folds::Matrix{Int}`: fold ids per observation and repetition.
- `covariates::Vector{Symbol}`, `learners::Vector{Pair{Symbol,String}}`,
  `fs_specification::Symbol`, `fs_kernel::Symbol`, `n_iterations::Int`: settings used.
- `cutoff::Float64`, `n_left::Int`, `n_right::Int`, `n_clusters::Tuple{Int,Int}`,
  `level::Float64`.
"""
struct RDFlexEstimate <: CausalEstimate
    design::Symbol
    tau_conventional::Float64
    tau_bias_corrected::Float64
    se_conventional::Float64
    se_robust::Float64
    fits::Vector{RDEstimate}
    all_coef::Matrix{Float64}
    all_se::Matrix{Float64}
    h_fs::Float64
    h::Vector{Float64}
    b::Vector{Float64}
    eta_y::Matrix{Float64}
    eta_d::Union{Nothing,Matrix{Float64}}
    folds::Matrix{Int}
    covariates::Vector{Symbol}
    learners::Vector{Pair{Symbol,String}}
    fs_specification::Symbol
    fs_kernel::Symbol
    n_iterations::Int
    cutoff::Float64
    n_left::Int
    n_right::Int
    n_clusters::Tuple{Int,Int}
    level::Float64
end

# Same reporting convention as `RDEstimate` (rdrobust): conventional point estimate,
# robust bias-corrected standard error, t statistic, p-value and interval.
StatsAPI.coef(r::RDFlexEstimate) = [r.tau_conventional]
StatsAPI.vcov(r::RDFlexEstimate) = fill(r.se_robust^2, 1, 1)
StatsAPI.coefnames(r::RDFlexEstimate) = ["RD effect (ML-adjusted)"]
tstats(r::RDFlexEstimate) = [r.tau_bias_corrected / r.se_robust]
function StatsAPI.confint(r::RDFlexEstimate; level::Real=0.95)
    c = critical_value(level, Inf)
    return [r.tau_bias_corrected - c * r.se_robust r.tau_bias_corrected + c * r.se_robust]
end
StatsAPI.nobs(r::RDFlexEstimate) = r.n_left + r.n_right
estimand(r::RDFlexEstimate) = r.design === :sharp ?
    "ATE at the cutoff (sharp RD)" : "LATE for compliers at the cutoff (fuzzy RD)"
method_name(r::RDFlexEstimate) =
    "Local linear RD with cross-fitted ML covariate adjustment (" *
    "$(first(r.fits).kernel) kernel, $(uppercase(string(first(r.fits).vce))) variance)"

function rd_inference_table(r::RDFlexEstimate; level::Real=r.level)
    z = critical_value(level)
    est = [r.tau_conventional, r.tau_bias_corrected, r.tau_bias_corrected]
    se = [r.se_conventional, r.se_conventional, r.se_robust]
    t = est ./ se
    return DataFrame(method=["Conventional", "Bias-corrected", "Robust"], estimate=est,
                     se=se, z=t, pvalue=two_sided_pvalue.(t), ci_lower=est .- z .* se,
                     ci_upper=est .+ z .* se)
end

function show_details(io::IO, r::RDFlexEstimate)
    println(io)
    f = first(r.fits)
    @printf(io, "Cutoff = %g; final bandwidths h = %.4g, b = %.4g; first-stage h = %.4g\n",
            r.cutoff, r.h[1], r.b[1], r.h_fs)
    @printf(io, "Observations left/right: %d / %d; within h: %d / %d\n", r.n_left,
            r.n_right, f.n_h_left, f.n_h_right)
    r.n_clusters != (0, 0) && @printf(io, "Clusters left/right: %d / %d\n", r.n_clusters...)
    println(io, "Covariates (", r.fs_specification, " specification): ",
            join(r.covariates, ", "))
    println(io, "Learners: ", join(("$(k) = $(v)" for (k, v) in r.learners), ", "),
            "; folds: ", maximum(r.folds), "; repetitions: ", size(r.folds, 2))
    tab = rd_inference_table(r)
    lv = round(Int, 100 * r.level)
    println(io, "Inference ($lv% CI):")
    for row in eachrow(tab)
        @printf(io, "  %-15s %10.4f  se %8.4f  p %8.4g  [%.4f, %.4f]\n", row.method,
                row.estimate, row.se, row.pvalue, row.ci_lower, row.ci_upper)
    end
    if f.first_stage !== nothing
        @printf(io, "First stage (jump in adjusted treatment): %.4f (se %.4f)\n",
                f.first_stage.tau_conventional, f.first_stage.se_conventional)
    end
end

const _ML_RDF_SPECS = (:cutoff, :cutoff_and_score, :interacted_cutoff_and_score)

"""
    rd_flex(data, outcome, running; covariates, cutoff=0.0, treatment=nothing,
            outcome_learner=LassoLearner(), treatment_learner=PenalizedLogisticLearner(),
            fs_specification=:cutoff, fs_kernel=:triangular, h_fs=nothing,
            n_iterations=2, n_folds=5, n_rep=1, folds=nothing, cluster=nothing,
            level=0.95, rng=Random.default_rng(), parallel=true,
            kwargs...) -> RDFlexEstimate

Regression discontinuity estimation with flexible, machine-learning covariate adjustment
(Noack, Olma & Rothe 2026), as in DoubleML's `RDFlex`.

Predetermined covariates ``W`` do not help identify an RD effect, but they can explain
much of the outcome's variation and so sharpen the estimate. Calonico, Cattaneo, Farrell
and Titiunik (2019) add them linearly ([`rd_estimate`](@ref) with `covariates`). Noack,
Olma and Rothe (2026) generalise this. They subtract from the outcome an adjustment
function ``\\eta(W)`` and apply a standard local linear RD analysis to
``M = Y - \\eta(W)``. If the distribution of ``W`` is continuous at the cutoff, the
jump in ``E[M \\mid X]`` equals the jump in ``E[Y \\mid X]`` for *any* fixed function
``\\eta``, so the estimand is unchanged. Only the variance depends on ``\\eta``. A
variance-minimising choice is
``\\eta(W) = (\\mu_+(W) + \\mu_-(W))/2`` with ``\\mu_\\pm(W) = E[Y \\mid W, X = c^\\pm]``.
This function estimates it with an arbitrary learner. In a fuzzy design the treatment is
adjusted in the same way (``D - \\eta_D(W)``), and the estimate is the ratio of the two
jumps.

The adjustment is estimated by **cross-fitting**, repeated for each of `n_rep` random
splits into `n_folds` folds.
1. On the training folds, fit the learner with kernel weights
   ``K((X - c)/h_{\\text{fs}})`` and features ``(W, Z)``, where ``Z = 1\\{X \\ge c\\}``,
   optionally with the score terms of `fs_specification`.
2. On the held-out fold, predict at ``Z = 0`` and ``Z = 1`` with score terms set to
   zero, and average the two predictions to obtain ``\\hat\\eta``.
3. Estimate the RD effect on ``Y - \\hat\\eta(W)`` with [`rd_estimate`](@ref), using
   robust bias-corrected inference.
4. For `n_iterations ≥ 2`, replace ``h_{\\text{fs}}`` by the bandwidth selected in step
   3 and repeat. The final estimate uses the last bandwidths with the last adjustment.
Because each ``\\hat\\eta(W_i)`` is fitted without observation ``i``, the estimation
error of the learner enters only at second order. The usual RD standard errors of the
adjusted variables remain valid when the learner converges, even slowly, to some fixed
function. That function need not be the optimal one: a poor learner costs precision
rather than validity.

The precision gain grows with the share of outcome variation that the covariates
explain near the cutoff and that the learner captures. With a linear learner the method
is close to the adjustment of Calonico, Cattaneo, Farrell and Titiunik (2019). It can
improve on it when covariates act non-linearly, and do worse when a flexible learner
overfits small local samples. The package's Monte Carlo study used 1,000 replications
per design (`test/validation/rd/flex_montecarlo.jl`). In a sharp design with non-linear
covariate effects at ``n = 1{,}000``, random-forest adjustment reduced the standard
deviation of the estimate from 0.60 (no covariates) to 0.40, with 95% coverage 0.935
and standard errors about 9% below the Monte Carlo standard deviation. At
``n = 4{,}000`` the coverage was 0.948. Identification still requires RD continuity,
and the covariates must be predetermined. Check this with
[`rd_covariate_balance`](@ref); the adjustment is not a substitute for continuity.

# Arguments
- `data::AbstractDataFrame`: one row per unit. Rows with missing or `NaN` values in the
  used columns are dropped.
- `outcome::Symbol`: outcome column.
- `running::Symbol`: running variable (treated side: `running ≥ cutoff`).

# Keywords
- `covariates::Vector{Symbol}`: numeric predetermined covariates (required; without
  covariates use [`rd_estimate`](@ref)).
- `cutoff::Real=0.0`: the RD threshold. It must lie strictly inside the range of the
  running variable.
- `treatment::Union{Nothing,Symbol}=nothing`: binary treatment take-up. Supplying it
  makes the design fuzzy.
- `outcome_learner=LassoLearner()`: learner for the outcome adjustment (a
  [`NuisanceLearner`](@ref) that accepts observation `weights`).
- `treatment_learner=PenalizedLogisticLearner()`: probabilistic learner for the
  treatment adjustment in fuzzy designs, used through [`fitpredict_proba`](@ref).
- `fs_specification::Symbol=:cutoff`: features added to the covariates in the nuisance
  fit: `:cutoff` (``Z``), `:cutoff_and_score` (``Z, X - c``) or
  `:interacted_cutoff_and_score` (``Z, Z(X - c), X - c``).
- `fs_kernel=:triangular`: kernel of the nuisance weights (`:triangular`,
  `:epanechnikov` or `:uniform`).
- `h_fs::Union{Nothing,Real}=nothing`: initial bandwidth of the nuisance weights. By
  default it is the largest of the MSE-optimal `h` and `b` without covariates, as in
  `RDFlex`.
- `n_iterations::Integer=2`: number of adjustment rounds. With `1`, the bandwidth is not
  updated, and the final bandwidth is selected on the adjusted data.
- `n_folds::Integer=5`, `n_rep::Integer=1`: cross-fitting folds and repetitions.
- `folds=nothing`: user-supplied fold ids, as a column name, a vector of length
  `nrow(data)`, or a matrix with one column per repetition.
- `cluster::Union{Nothing,Symbol}=nothing`: cluster identifier. Folds are drawn at the
  cluster level, and the RD variance is cluster-robust.
- `level::Real=0.95`: confidence level used when printing.
- `rng::AbstractRNG=Random.default_rng()`: random number generator for folds and learner
  seeds. Folds are drawn on the observations sorted by running variable and outcome, so
  results do not depend on the row order of `data`.
- `parallel::Bool=true`: fit the folds on threads (results are identical).
- `kwargs...`: passed to [`rd_estimate`](@ref) for the final fit (for example `kernel`,
  `vce`, `p`, `bwselect`, `h`, `b`).

# Returns
- [`RDFlexEstimate`](@ref).

# Examples
```julia
using DrSnow, CSV, DataFrames, StableRNGs
df = CSV.read(pkgdir(DrSnow, "test", "validation", "rd", "flex_data.csv"), DataFrame)
covs = [:z1, :z2, :z3, :z4]
r = rd_flex(df, :y_sharp, :x; covariates=covs, rng=StableRNG(1))
confint(r)
rd_inference_table(r)
confint(rd_estimate(df, :y_sharp, :x))    # unadjusted, for comparison
rd_flex(df, :y_fuzzy, :x; covariates=covs, treatment=:d, rng=StableRNG(1))
```

# References
- Noack, C., Olma, T., & Rothe, C. (2026). Flexible covariate adjustments in regression
  discontinuity designs. *Journal of Econometrics*, 257, 106298.
- Calonico, S., Cattaneo, M. D., Farrell, M. H., & Titiunik, R. (2019). Regression
  discontinuity designs using covariates. *Review of Economics and Statistics*,
  101(3), 442–451.
- Calonico, S., Cattaneo, M. D., & Titiunik, R. (2014). Robust nonparametric
  confidence intervals for regression-discontinuity designs. *Econometrica*, 82(6),
  2295–2326.
- Chernozhukov, V., Chetverikov, D., Demirer, M., Duflo, E., Hansen, C., Newey, W., &
  Robins, J. (2018). Double/debiased machine learning for treatment and structural
  parameters. *The Econometrics Journal*, 21(1), C1–C68.
"""
function rd_flex(data::AbstractDataFrame, outcome::Symbol, running::Symbol;
                 covariates::Vector{Symbol}, cutoff::Real=0.0,
                 treatment::Union{Nothing,Symbol}=nothing,
                 outcome_learner=LassoLearner(),
                 treatment_learner=PenalizedLogisticLearner(),
                 fs_specification::Symbol=:cutoff, fs_kernel=:triangular,
                 h_fs::Union{Nothing,Real}=nothing, n_iterations::Integer=2,
                 n_folds::Integer=5, n_rep::Integer=1, folds=nothing,
                 cluster::Union{Nothing,Symbol}=nothing, level::Real=0.95,
                 rng::AbstractRNG=Random.default_rng(), parallel::Bool=true, kwargs...)
    ctx = "rd_flex"
    isempty(covariates) && throw(ArgumentError(
        "$ctx: `covariates` must name at least one covariate (without covariates use " *
        "rd_estimate)"))
    fs_specification in _ML_RDF_SPECS || throw(ArgumentError(
        "$ctx: fs_specification must be one of $(join(_ML_RDF_SPECS, ", "))"))
    fk = _rd_kernel(fs_kernel)
    n_iterations >= 1 || throw(ArgumentError("$ctx: n_iterations must be ≥ 1"))
    _ml_check_common(n_folds, n_rep)
    0 < level < 1 || throw(ArgumentError("$ctx: level must be in (0, 1)"))
    h_fs === nothing || h_fs > 0 || throw(ArgumentError("$ctx: h_fs must be positive"))
    fcol = folds isa Symbol ? folds : nothing
    cols = Symbol[c for c in Any[outcome, running, treatment, cluster, fcol]
                  if c !== nothing]
    append!(cols, covariates)
    require_columns(data, cols; context=ctx)
    keep = trues(nrow(data))
    for c in cols
        v = data[!, c]
        keep .&= .!ismissing.(v)
        keep .&= [!(x isa AbstractFloat && isnan(x)) for x in v]
    end
    any(keep) || throw(ArgumentError("$ctx: no complete observations"))
    sub = data[keep, :]
    n = nrow(sub)
    y = _ml_column(sub, outcome; context=ctx)
    x = _ml_column(sub, running; context=ctx)
    X = _ml_matrix(sub, covariates; context=ctx)
    c = Float64(cutoff)
    (minimum(x) < c < maximum(x)) || throw(ArgumentError(
        "$ctx: the cutoff $c must lie strictly inside the range of the running variable"))
    s = x .- c
    Z = Float64.(s .>= 0)
    D = nothing
    if treatment !== nothing
        D = _ml_column(sub, treatment; context=ctx)
        _ml_check_binary_col(D, treatment, ctx)
    end
    # clusters and folds (folds drawn on a row-order-free ordering of the observations)
    cid = nothing
    if cluster !== nothing
        cid, _ = _ml_group_index(sub[!, cluster])
    end
    F = if folds === nothing
        ord = sortperm(collect(zip(x, y, eachcol(X)...)))
        strata = D === nothing ? Z : 2 .* Z .+ D
        Fo = crossfit_folds(n, n_folds, n_rep; rng=rng, strata=strata[ord],
                            groups=cid === nothing ? nothing : cid[ord])
        Fs = similar(Fo)
        Fs[ord, :] = Fo
        Fs
    else
        fv = folds isa Symbol ? sub[!, folds] :
             folds isa AbstractVector ? _ml_rdf_rows(folds, keep, nrow(data), ctx) :
             folds isa AbstractMatrix ? _ml_rdf_rows(folds, keep, nrow(data), ctx) :
             throw(ArgumentError("$ctx: folds must be a column name, vector or matrix"))
        _ml_resolve_folds(sub, fv isa AbstractVector ? Int.(collect(fv)) : Int.(fv), n,
                          n_folds, n_rep, rng, nothing, cid; context=ctx)
    end
    R = size(F, 2)
    K = maximum(F)
    # initial first-stage bandwidth: largest MSE-optimal h / b without covariates
    base = DataFrame(_y=y, _x=x)
    D === nothing || (base._d = D)
    cluster === nothing || (base._c = sub[!, cluster])
    ccol = cluster === nothing ? nothing : :_c
    if h_fs === nothing
        bw = rd_bandwidth(base, :_y, :_x; cutoff=c,
                          treatment=D === nothing ? nothing : :_d, cluster=ccol)
        h_fs = max(bw.h_left, bw.h_right, bw.b_left, bw.b_right)
    end
    h_fs = Float64(h_fs)
    _ml_rdf_check_sign(D, s, _rd_kweight(x, c, h_fs, fk))
    feats = _ml_rdf_features(fs_specification, Z, s, X)
    seeds = _ml_seeds(rng, K, 2 * n_iterations, R)
    all_coef = zeros(3, R)
    all_se = zeros(3, R)
    hs = zeros(R)
    bs = zeros(R)
    fits = RDEstimate[]
    eta_y = zeros(n, R)
    eta_d = D === nothing ? nothing : zeros(n, R)
    for r in 1:R
        w = _rd_kweight(x, c, h_fs, fk)
        h = b = nothing
        fit = nothing
        for it in 1:n_iterations
            sd = view(seeds, :, (2it - 1):(2it), r)
            ηy = _ml_rdf_eta(outcome_learner, y, false, feats, w, F[:, r], sd[:, 1],
                             parallel, ctx, :outcome)
            eta_y[:, r] = ηy
            adj = DataFrame(_y=y .- ηy, _x=x)
            if D !== nothing
                ηd = _ml_rdf_eta(treatment_learner, D, true, feats, w, F[:, r],
                                 sd[:, 2], parallel, ctx, :treatment)
                eta_d[:, r] = ηd
                adj._d = D .- ηd
            end
            cluster === nothing || (adj._c = sub[!, cluster])
            tcol = D === nothing ? nothing : :_d
            if it < n_iterations
                f0 = rd_estimate(adj, :_y, :_x; cutoff=c, treatment=tcol, cluster=ccol,
                                 level, kwargs...)
                h = max(f0.h_left, f0.h_right)
                b = max(f0.b_left, f0.b_right)
                w = _rd_kweight(x, c, h, fk)
            else
                # as RDFlex: the bandwidths of the last update, unless given by the user
                kw = h === nothing ? values(kwargs) : merge((; h, b), values(kwargs))
                fit = rd_estimate(adj, :_y, :_x; cutoff=c, treatment=tcol, cluster=ccol,
                                  level, kw...)
            end
        end
        push!(fits, fit)
        hs[r] = max(fit.h_left, fit.h_right)
        bs[r] = max(fit.b_left, fit.b_right)
        all_coef[:, r] = [fit.tau_conventional, fit.tau_bias_corrected,
                          fit.tau_bias_corrected]
        all_se[:, r] = [fit.se_conventional, fit.se_conventional, fit.se_robust]
    end
    θ = [median(all_coef[j, :]) for j in 1:3]
    se = [sqrt(median(all_se[j, :] .^ 2 .+ (all_coef[j, :] .- θ[j]) .^ 2)) for j in 1:3]
    learners = Pair{Symbol,String}[:ml_g => _ml_learner_name(outcome_learner)]
    D === nothing || push!(learners, :ml_m => _ml_learner_name(treatment_learner))
    f1 = first(fits)
    return RDFlexEstimate(D === nothing ? :sharp : :fuzzy, θ[1], θ[2], se[1], se[3],
                          fits, all_coef, all_se, h_fs, hs, bs, eta_y, eta_d, F,
                          copy(covariates), learners, fs_specification, fk,
                          Int(n_iterations), c, f1.n_left, f1.n_right, f1.n_clusters,
                          Float64(level))
end

# User-supplied folds of length nrow(data) restricted to the complete cases.
function _ml_rdf_rows(f, keep, N, ctx)
    size(f, 1) == N || throw(DimensionMismatch(
        "$ctx: folds must have one row per row of `data` ($N), got $(size(f, 1))"))
    return f isa AbstractVector ? f[keep] : f[keep, :]
end

# Warn when the (weighted) take-up is higher on the left of the cutoff.
function _ml_rdf_check_sign(D, s, w)
    D === nothing && return nothing
    l = (s .< 0) .& (w .> 0)
    r = (s .> 0) .& (w .> 0)
    (any(l) && any(r)) || return nothing
    pl = sum(D[l] .* w[l]) / sum(w[l])
    pr = sum(D[r] .* w[r]) / sum(w[r])
    pl - pr > 1e-6 && @warn "rd_flex: treatment take-up near the cutoff is higher on " *
                            "the left than on the right; check which side is treated"
    return nothing
end

# Feature matrix of the first stage and its counterparts at Z = 0 / Z = 1 (score
# terms set to zero), as in RDFlex.
function _ml_rdf_features(spec, Z, s, X)
    n = length(Z)
    o, z = ones(n), zeros(n)
    if spec === :cutoff
        return (hcat(Z, X), hcat(z, X), hcat(o, X))
    elseif spec === :cutoff_and_score
        return (hcat(Z, s, X), hcat(z, z, X), hcat(o, z, X))
    else
        return (hcat(Z, Z .* s, s, X), hcat(z, z, z, X), hcat(o, z, z, X))
    end
end

# Cross-fitted adjustment η = (μ̂(Z=0, X) + μ̂(Z=1, X)) / 2 with kernel weights `w`.
function _ml_rdf_eta(learner, target, proba, feats, w, fold, seeds, parallel, ctx, what)
    ZX, ZX0, ZX1 = feats
    n = length(target)
    K = maximum(fold)
    out = Vector{Float64}(undef, n)
    run = function (k)
        test = fold .== k
        train = .!test .& (w .> 0)
        count(train) > 0 || throw(ArgumentError(
            "$ctx: no training observations with positive first-stage kernel weight " *
            "in fold $k; increase h_fs or use fewer folds"))
        ytr = target[train]
        if proba && (all(==(0), ytr) || all(==(1), ytr))
            throw(ArgumentError("$ctx: the $what takes a single value in the " *
                                "first-stage training sample of fold $k"))
        end
        nt = count(test)
        Xnew = vcat(ZX0[test, :], ZX1[test, :])
        trng = Random.Xoshiro(seeds[k])
        pred = proba ?
            fitpredict_proba(learner, ZX[train, :], ytr, Xnew; rng=trng,
                             weights=w[train]) :
            fitpredict(learner, ZX[train, :], ytr, Xnew; rng=trng, weights=w[train])
        length(pred) == 2nt || throw(DimensionMismatch(
            "$ctx: the $what learner returned $(length(pred)) predictions for $(2nt) rows"))
        all(isfinite, pred) || throw(ArgumentError(
            "$ctx: the $what learner returned non-finite predictions in fold $k"))
        out[test] = (pred[1:nt] .+ pred[(nt + 1):end]) ./ 2
        return nothing
    end
    if parallel && Threads.nthreads() > 1 && K > 1
        tasks = [Threads.@spawn run(k) for k in 1:K]
        for t in tasks
            try
                fetch(t)
            catch e
                e isa TaskFailedException ? throw(e.task.exception) : rethrow()
            end
        end
    else
        foreach(run, 1:K)
    end
    return out
end
