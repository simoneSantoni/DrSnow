# Covariate balance assessed with the randomization distribution.

"""
    ri_balance_test(data, treatment, covariates; statistic=:mahalanobis,
                    method=:minp, mechanism=nothing, strata=nothing, cluster=nothing,
                    id=nothing, nperm=10_000, exact=:auto, rng=Random.default_rng(),
                    threaded=Threads.nthreads() > 1) -> DiagnosticTest

Randomization test of covariate balance: whether the observed differences in
pre-treatment covariates between arms are unusual relative to their randomization
distribution under the stated assignment mechanism.

Pre-treatment covariates cannot be affected by treatment, so the sharp null of "no
effect" holds for them by construction, and their randomization distribution under
the stated design is known exactly (Hansen and Bowers 2008). For each covariate the
(stratum-size-weighted) difference in means between arms is computed on the realized
assignment and on every assignment of the reference set. The omnibus statistic is
either the Mahalanobis distance ``(Δ - μ)' Σ^{+} (Δ - μ)``, where ``μ`` and ``Σ``
are the mean vector and covariance matrix of the differences over the reference set
(`:mahalanobis`), or the largest absolute standardized difference
``\\max_j |Δ_j - μ_j| / \\sqrt{Σ_{jj}}`` (`:max_abs_z`). Because ``μ`` and ``Σ`` are
computed from the reference set itself, which is exchangeable with the observed
assignment, the test remains exact under Monte Carlo draws. Per-covariate two-sided
randomization p-values are reported with a Westfall–Young step-down adjustment.

What the test can and cannot show: a small p-value indicates imbalance that is
unusual under the stated mechanism — a rare chance imbalance, a problem in the
implementation of the randomization, or a mechanism different from the one declared
(for example an "as-if random" natural experiment that is not). A large p-value does
not show that assignment followed the mechanism, and says nothing about balance on
unobserved characteristics. Balance tests are not a substitute for adjusting for
prognostic covariates; chance imbalance in a correctly randomized experiment is
better handled by pre-specified covariate adjustment (e.g. the `:lin` statistic of
[`randomization_test`](@ref)) than by conditioning the analysis on the test.

# Arguments
- `data`: a `DataFrame` with one row per randomized unit.
- `treatment::Symbol`: 0/1 treatment column.
- `covariates::Vector{Symbol}`: numeric pre-treatment covariates (dummy-encode
  categorical ones); no missing values.

# Keywords
- `statistic::Symbol`: omnibus statistic, `:mahalanobis` (default) or
  `:max_abs_z`.
- `method::Symbol`: Westfall–Young variant for the per-covariate p-values, `:minp`
  (default) or `:maxt`.
- `mechanism`, `strata`, `cluster`, `id`, `nperm`, `exact`, `rng`, `threaded`: as
  in [`randomization_test`](@ref).

# Returns
- [`DiagnosticTest`](@ref) whose `details` hold `per_covariate` (a `DataFrame` with
  arm means, differences, standardized differences, and unadjusted and
  Westfall–Young p-values), the reference distribution of the omnibus statistic
  (`distribution`, `weights`), `exact` and `n_draws`.

# Examples
```julia
using DrSnow, DataFrames, StableRNGs
rng = StableRNG(6)
df = DataFrame(block=repeat(1:4; inner=10), d=repeat([1, 1, 1, 1, 1, 0, 0, 0, 0, 0], 4))
df.age = 30 .+ 5 .* randn(rng, 40)
df.income = 20 .+ 4 .* randn(rng, 40)
df.female = Int.(rand(rng, 40) .< 0.5)
bt = ri_balance_test(df, :d, [:age, :income, :female]; strata=:block, rng=rng)
bt.details.per_covariate
```

# References
- Hansen, B. B., & Bowers, J. (2008). Covariate balance in simple, stratified and
  clustered comparative studies. *Statistical Science*, 23(2), 219–236.
- Imbens, G. W., & Rubin, D. B. (2015). *Causal Inference for Statistics, Social,
  and Biomedical Sciences: An Introduction*. Cambridge University Press.
- Morgan, K. L., & Rubin, D. B. (2012). Rerandomization to improve covariate balance
  in experiments. *Annals of Statistics*, 40(2), 1263–1282.
- Westfall, P. H., & Young, S. S. (1993). *Resampling-Based Multiple Testing:
  Examples and Methods for p-Value Adjustment*. Wiley.
"""
function ri_balance_test(data, treatment::Symbol, covariates::Vector{Symbol};
                         statistic::Symbol=:mahalanobis, method::Symbol=:minp,
                         mechanism=nothing, strata::Union{Nothing,Symbol}=nothing,
                         cluster::Union{Nothing,Symbol}=nothing,
                         id::Union{Nothing,Symbol}=nothing, nperm::Integer=10_000,
                         exact=:auto, rng::AbstractRNG=Random.default_rng(),
                         threaded::Bool=Threads.nthreads() > 1)
    ctxname = "ri_balance_test"
    isempty(covariates) && throw(ArgumentError("$ctxname: need at least one covariate"))
    allunique(covariates) || throw(ArgumentError("$ctxname: duplicate covariates"))
    statistic in (:mahalanobis, :max_abs_z) ||
        throw(ArgumentError("$ctxname: statistic must be :mahalanobis or :max_abs_z"))
    method in (:minp, :maxt) || throw(ArgumentError("method must be :minp or :maxt"))
    design = _ri_design(data, treatment; mechanism, strata, cluster, id,
                        sortcols=covariates, context=ctxname)
    ctx = design.ctx
    X = try
        _ri_numeric_matrix(data, covariates, design.rows)
    catch err
        err isa MethodError || err isa InexactError || rethrow()
        throw(ArgumentError("$ctxname: covariates must be numeric (dummy-encode " *
                            "categorical covariates)"))
    end
    all(isfinite, X) || throw(ArgumentError("$ctxname: covariates have non-finite values"))
    p = length(covariates)
    cols = [X[:, j] for j in 1:p]
    f = z -> [_ri_stratified_dim(cols[j], z, ctx) for j in 1:p]
    plan = _ri_plan(design.mech, nperm, exact, rng)
    R = _ri_map(f, plan, design.mech, design.z, p; threaded)
    obs = f(design.z)
    any(isnan, obs) && throw(ArgumentError("$ctxname: differences undefined for the " *
                                           "observed assignment"))
    w = _ri_weights(plan)
    keep = [all(!isnan, view(R, i, :)) for i in axes(R, 1)]
    Rk = R[keep, :]
    wk = w[keep]
    W = sum(wk)
    mu = vec(sum(Rk .* wk; dims=1)) ./ W
    C = Rk .- mu'
    Sigma = (C' * (C .* wk)) ./ W
    if statistic === :mahalanobis
        Si = pinv(Sigma)
        omni = d -> (v = d .- mu; dot(v, Si * v))
        sname = "Mahalanobis distance of mean differences"
    else
        sds = sqrt.(max.(diag(Sigma), 0.0))
        omni = d -> maximum(j -> sds[j] > 0 ? abs(d[j] - mu[j]) / sds[j] : 0.0, 1:p)
        sname = "maximum absolute standardized difference"
    end
    dist = [omni(view(Rk, i, :)) for i in axes(Rk, 1)]
    o = omni(obs)
    pv, _, _ = _ri_pvalue(o, dist, wk, :greater)
    wy = _ri_westfall_young(obs, Rk, wk, :two_sided, method)
    z = design.z
    m1 = [mean(X[z, j]) for j in 1:p]
    m0 = [mean(X[.!z, j]) for j in 1:p]
    sdp = [sqrt((var(X[z, j]) + var(X[.!z, j])) / 2) for j in 1:p]
    per = DataFrame(covariate=covariates, mean_treated=m1, mean_control=m0,
                    difference=obs,
                    std_difference=[sdp[j] > 0 ? obs[j] / sdp[j] : 0.0 for j in 1:p],
                    pvalue=wy.raw, pvalue_westfall_young=wy.adjusted)
    mstr = plan.exact ?
        "randomization inference, exact enumeration of $(plan.B) assignments" :
        "randomization inference, Monte Carlo with $(plan.B) draws"
    note = "A small p-value indicates covariate imbalance that is unusual under the " *
           "stated assignment mechanism (chance imbalance, an implementation " *
           "problem, or a misspecified mechanism). A large p-value does not show " *
           "that assignment followed the mechanism, nor that the arms are comparable " *
           "on unobserved characteristics. Differences are " *
           (length(ctx.strata) > 1 ? "stratum-size-weighted within-stratum " : "") *
           "differences in means; mechanism: $(_ri_describe(design.mech))."
    return DiagnosticTest("Randomization balance test ($sname)",
                          "the covariate differences between arms are a draw from the " *
                          "stated assignment mechanism", o, pv; dof=(p,),
                          method=mstr, note=note,
                          details=(per_covariate=per, distribution=dist, weights=wk,
                                   exact=plan.exact, n_draws=plan.B))
end
