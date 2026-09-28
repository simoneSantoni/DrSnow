# Generic machine-learning inference on heterogeneous treatment effects in randomized
# experiments (Chernozhukov, Demirer, Duflo & Fernández-Val 2025, "CDDF"): best
# linear predictor (BLP), sorted group average treatment effects (GATES) and
# classification analysis (CLAN), with repeated auxiliary/main sample splits and the
# median aggregation rule (intervals at nominal level from per-split level 1 - α/2,
# p-values 2 × median).

"""
    GenericMLInference

Result of [`generic_ml`](@ref): best linear predictor (BLP), sorted group average
treatment effects (GATES) and classification analysis (CLAN) of heterogeneous
treatment effects, aggregated over repeated sample splits by the median rule of
Chernozhukov, Demirer, Duflo and Fernández-Val (2025).

Each table reports, for every parameter, the median over splits of the per-split
estimate, the medians of the per-split lower and upper confidence bounds, and the
split-adjusted p-value ``\\min(1, 2 \\times \\text{median } p)``. The per-split
intervals are computed at level ``1 - \\alpha/2`` so that the reported intervals have
nominal level `level` ``= 1 - \\alpha``; the resulting inference is valid, typically
conservative (see [`generic_ml`](@ref)). Query the object with [`blp`](@ref),
[`blp_test`](@ref), [`gates`](@ref) and [`clan`](@ref).

# Fields
- `blp::DataFrame`: BLP table with rows `"ATE (β₁)"` and `"HET (β₂)"` and columns
  `term`, `estimate`, `lower`, `upper`, `pvalue`.
- `gates::DataFrame`: GATES table with one row per group `G1`, …, `GK` (G1 least,
  GK most affected by the ML proxy) and the difference `"GK - G1"`; columns `group`,
  `estimate`, `lower`, `upper`, `pvalue`.
- `clan::DataFrame`: CLAN table with columns `covariate`, `most_affected`,
  `least_affected`, `difference`, `lower`, `upper`, `pvalue`.
- `lambda::Float64`: median over splits of the BLP fit criterion
  ``\\Lambda = \\hat\\beta_2^2 \\widehat{\\mathrm{Var}}(S)``, used to compare ML proxies
  (larger is better).
- `lambda_bar::Float64`: median over splits of the GATES fit criterion
  ``\\bar\\Lambda = \\sum_k \\hat\\gamma_k^2 / K``.
- `n_splits::Int`: number of auxiliary/main sample splits.
- `n_groups::Int`: number of GATES groups ``K``.
- `level::Float64`: nominal confidence level of the reported intervals.
- `learner::String`: description of the ML proxy learner.
- `n_degenerate::Int`: number of splits in which the CATE proxy was constant on the
  main sample and was jittered to form groups.
- `n::Int`: number of observations.
- `split_estimates::Dict{Symbol,Matrix{Float64}}`: per-split results, with keys
  `:blp_estimate` and `:blp_pvalue` (`n_splits × 2`), `:gates_estimate`
  (`n_splits × (K + 1)`) and `:clan_difference` (`n_splits × p`); useful to inspect
  the variability induced by sample splitting.
"""
struct GenericMLInference
    blp::DataFrame
    gates::DataFrame
    clan::DataFrame
    lambda::Float64
    lambda_bar::Float64
    n_splits::Int
    n_groups::Int
    level::Float64
    learner::String
    n_degenerate::Int
    n::Int
    split_estimates::Dict{Symbol,Matrix{Float64}}
end

function Base.show(io::IO, ::MIME"text/plain", g::GenericMLInference)
    println(io, "Generic ML inference on heterogeneous effects (CDDF)")
    println(io, "Observations: $(g.n); $(g.n_splits) auxiliary/main splits; ML proxy: " *
                "$(g.learner); $(round(Int, 100g.level))% intervals (median rule)")
    println(io, "\nBest linear predictor:")
    show(io, g.blp)
    println(io, "\n\nSorted group average treatment effects (G1 = least affected):")
    show(io, g.gates)
    println(io, "\n\nClassification analysis (most minus least affected group):")
    show(io, g.clan)
    @printf(io, "\n\nFit of the ML proxy: Λ = %.4g (BLP), Λ̄ = %.4g (GATES)\n",
            g.lambda, g.lambda_bar)
    g.n_degenerate > 0 && println(io, "Note: the ML proxy was constant on the main " *
                                      "sample in $(g.n_degenerate) split(s).")
    return nothing
end

Base.show(io::IO, g::GenericMLInference) =
    print(io, "GenericMLInference(", g.n_splits, " splits)")

"""
    generic_ml(data, outcome, treatment; covariates, proxy_learner=LassoLearner(),
               propensity=nothing, n_splits=100, n_groups=5,
               clan_covariates=covariates, level=0.95, cluster=nothing,
               rng=Random.default_rng(), parallel=true) -> GenericMLInference

Generic machine-learning inference on heterogeneous treatment effects in randomized
experiments (Chernozhukov, Demirer, Duflo & Fernández-Val 2025): the best linear
predictor, sorted group average treatment effects and classification analysis.

The research question is whether, and for whom, the effect of a randomized binary
treatment varies with pre-treatment characteristics ``X``, when the conditional
average treatment effect ``s_0(x) = E[Y(1) - Y(0) \\mid X = x]`` may be complex and
high-dimensional and no machine-learning estimator of it is consistent or has a
known distribution. Instead of ``s_0`` itself, the method targets features of
``s_0`` that are defined relative to an ML proxy ``S(X)`` built on an independent
auxiliary sample: the **best linear predictor** of ``s_0(X)`` by ``S(X)``,

```math
\\mathrm{BLP}[s_0(X) \\mid S(X)] = \\beta_1 + \\beta_2\\{S(X) - E\\,S(X)\\},
```

where ``\\beta_1`` is the average treatment effect and ``\\beta_2 \\neq 0`` means that
the proxy captures real heterogeneity (``\\beta_2 = 1`` if ``S = s_0``); the
**GATES** ``\\gamma_k = E[s_0(X) \\mid S(X) \\in G_k]`` for the ``K`` quantile groups
``G_k`` of the proxy (G1 least, GK most affected); and **CLAN**, the average
characteristics of the most and least affected groups. The only identifying
assumption is that treatment is randomly assigned with known probabilities
``p(X) \\in (0, 1)`` (and SUTVA); no assumption on the quality of the ML proxy is
needed, but a poor proxy yields ``\\beta_2 \\approx 0`` and flat GATES even when
effects are heterogeneous. The estimands are conditional on the proxy, and thus on
the auxiliary sample.

For each of `n_splits` random splits into an auxiliary and a main half (stratified by
treatment, keeping clusters together):

1. On the auxiliary half, `proxy_learner` fits ``E[Y \\mid D = d, X]`` separately for
   ``d = 0, 1``; the proxies are the baseline ``B(X) = \\hat\\mu_0(X)`` and the CATE
   proxy ``S(X) = \\hat\\mu_1(X) - \\hat\\mu_0(X)``, evaluated on the main half.
2. On the main half, weighted least squares with weights ``1/\\{p(X)(1 - p(X))\\}``
   of the BLP and GATES regressions displayed below estimates
   ``(\\beta_1, \\beta_2)`` and ``(\\gamma_1, \\dots, \\gamma_K)``, with HC1 or, with
   `cluster`, CR1 standard errors; CLAN compares the means of `clan_covariates`
   between GK and G1 with Welch standard errors.
3. Point estimates are the medians over splits of the per-split estimates.

```math
\\begin{aligned}
Y &= \\alpha_0 + \\alpha_1 B + \\beta_1 (D - p)
     + \\beta_2 (D - p)(S - \\bar S) + \\varepsilon, \\\\
Y &= \\alpha_0 + \\alpha_1 B + \\textstyle\\sum_{k=1}^K \\gamma_k (D - p)\\,
     1\\{S \\in G_k\\} + \\varepsilon.
\\end{aligned}
```

Inference follows the variational (median) approach of Chernozhukov et al. (2025),
which conditions on the data and accounts for the randomness of sample splitting. In
their notation, the median of per-split confidence bounds computed at level
``1 - \\alpha`` gives an interval with coverage at least ``1 - 2\\alpha``, and the
split-adjusted p-value is ``\\min(1, 2 \\times \\text{median } p)``. `generic_ml`
therefore computes per-split intervals at level ``1 - \\alpha/2`` with
``\\alpha = 1 -`` `level`, so that the reported intervals have nominal level `level`,
and doubles the median p-value. This inference is valid, typically conservative:
the factor-of-two adjustment is a worst-case bound over the dependence between
splits. The number of splits reduces the Monte Carlo noise of the medians but not
this conservativeness. Use the method for randomized experiments (including
stratified designs with known, unit-specific assignment probabilities); for
observational data use doubly robust scores with [`cate_projection`](@ref),
[`best_linear_projection`](@ref) or [`subgroup_effects`](@ref). Compare proxy
learners through `lambda` and `lambda_bar` rather than through the test p-values.

# Arguments
- `data`: a `DataFrame` with one row per experimental unit.
- `outcome::Symbol`: numeric outcome column ``Y``.
- `treatment::Symbol`: randomized treatment column ``D``, coded `0`/`1`.

# Keywords
- `covariates::Vector{Symbol} = Symbol[]`: pre-treatment variables used by the proxy
  learner.
- `proxy_learner = LassoLearner()`: [`NuisanceLearner`](@ref) fitted separately to
  treated and control units of the auxiliary sample.
- `propensity = nothing`: known assignment probabilities; `nothing` uses the share
  treated (complete randomization), a number sets a common probability, and a
  column name gives each unit's own probability (all must lie in ``(0, 1)``).
- `n_splits::Integer = 100`: number of auxiliary/main splits; more splits stabilize
  the medians at proportional computational cost.
- `n_groups::Integer = 5`: number of GATES groups (quantile groups of ``S``).
- `clan_covariates::Vector{Symbol} = covariates`: variables compared between the
  most and least affected groups (they need not be used by the proxy).
- `level::Real = 0.95`: nominal level of the reported (median-rule) intervals.
- `cluster = nothing`: cluster column; splits keep clusters together and the
  regressions use cluster-robust standard errors.
- `rng::AbstractRNG = Random.default_rng()`: random-number generator; per-split
  seeds are drawn up front so results do not depend on threading.
- `parallel::Bool = true`: run splits on several threads when available.

# Returns
- [`GenericMLInference`](@ref): tables through [`blp`](@ref), [`blp_test`](@ref),
  [`gates`](@ref) and [`clan`](@ref); per-split results in `split_estimates`.

# Examples
```julia
using DrSnow, DataFrames, StableRNGs
rng = StableRNG(1)
n = 1000
df = DataFrame(x1=randn(rng, n), x2=randn(rng, n), x3=randn(rng, n))
df.d = Float64.(rand(rng, n) .< 0.5)
df.y = df.x2 .+ (1 .+ df.x1) .* df.d .+ randn(rng, n)
g = generic_ml(df, :y, :d; covariates=[:x1, :x2, :x3], n_splits=20,
               rng=StableRNG(2))
blp_test(g)
gates(g)
clan(g)
```

# References
- Chernozhukov, V., Demirer, M., Duflo, E., & Fernández-Val, I. (2025). Fisher–Schultz
  lecture: Generic machine learning inference on heterogeneous treatment effects in
  randomized experiments, with an application to immunization in India.
  *Econometrica*, 93(4), 1121–1164.
- Semenova, V., & Chernozhukov, V. (2021). Debiased machine learning of conditional
  average treatment effects and other causal functions. *The Econometrics Journal*,
  24(2), 264–289.
- Athey, S., & Imbens, G. (2016). Recursive partitioning for heterogeneous causal
  effects. *Proceedings of the National Academy of Sciences*, 113(27), 7353–7360.
- Imbens, G. W., & Rubin, D. B. (2015). *Causal Inference for Statistics, Social, and
  Biomedical Sciences: An Introduction*. Cambridge University Press.
"""
function generic_ml(data, outcome::Symbol, treatment::Symbol; covariates=Symbol[],
                    proxy_learner=LassoLearner(), propensity=nothing,
                    n_splits::Integer=100, n_groups::Integer=5,
                    clan_covariates=covariates, level::Real=0.95, cluster=nothing,
                    rng::AbstractRNG=Random.default_rng(), parallel::Bool=true)
    ctx = "generic_ml"
    0 < level < 1 || throw(ArgumentError("$ctx: level must be in (0, 1)"))
    n_splits >= 1 || throw(ArgumentError("$ctx: n_splits must be positive"))
    n_groups >= 2 || throw(ArgumentError("$ctx: n_groups must be at least 2"))
    covs = Symbol.(collect(covariates))
    ccovs = Symbol.(collect(clan_covariates))
    pcol = propensity isa Symbol ? [propensity] : Symbol[]
    cl = cluster === nothing ? Symbol[] : [Symbol(cluster)]
    require_columns(data, vcat(outcome, treatment, covs, ccovs, pcol, cl); context=ctx)
    n = nrow(data)
    y = _ml_column(data, outcome; context=ctx)
    d = _ml_column(data, treatment; context=ctx)
    _ml_check_binary_col(d, treatment, ctx)
    X = _ml_matrix(data, covs; context=ctx)
    C = _ml_matrix(data, ccovs; context=ctx)
    p = if propensity === nothing
        fill(mean(d), n)
    elseif propensity isa Real
        fill(Float64(propensity), n)
    else
        _ml_column(data, propensity; context=ctx)
    end
    all(x -> 0 < x < 1, p) ||
        throw(ArgumentError("$ctx: assignment probabilities must lie in (0, 1)"))
    cid, G = _ml_cluster_ids(data, cluster; context=ctx)
    α = 1 - level
    lvl = 1 - α / 2
    K = n_groups
    seeds = task_seeds(rng, n_splits)
    nc = length(ccovs)
    # per split: estimates, lower, upper, p-values
    blp_est = zeros(n_splits, 2, 4)
    gates_est = zeros(n_splits, K + 1, 4)
    clan_est = zeros(n_splits, nc, 6)
    lam = zeros(n_splits)
    lamb = zeros(n_splits)
    degenerate = falses(n_splits)
    run = function (s)
        srng = Random.Xoshiro(seeds[s])
        half = crossfit_folds(n, 2, 1; rng=srng, strata=d, groups=cid)[:, 1]
        aux = half .== 1
        main = .!aux
        t0 = aux .& (d .== 0)
        t1 = aux .& (d .== 1)
        (any(t0) && any(t1)) ||
            throw(ArgumentError("$ctx: an auxiliary sample lacks treated or control units"))
        Xm = X[main, :]
        r0, r1 = Random.Xoshiro(rand(srng, UInt64)), Random.Xoshiro(rand(srng, UInt64))
        μ0 = fitpredict(proxy_learner, X[t0, :], y[t0], Xm; rng=r0)
        μ1 = fitpredict(proxy_learner, X[t1, :], y[t1], Xm; rng=r1)
        S = μ1 .- μ0
        if var(S) <= 1e-12 * max(var(y), eps())
            degenerate[s] = true
            S = S .+ 1e-6 * std(y) .* randn(srng, length(S))
        end
        res = _ml_cddf_split(y[main], d[main], p[main], μ0, S, C[main, :], K, lvl,
                             cid === nothing ? nothing : cid[main], ctx)
        blp_est[s, :, :] .= res.blp
        gates_est[s, :, :] .= res.gates
        clan_est[s, :, :] .= res.clan
        lam[s] = res.lambda
        lamb[s] = res.lambda_bar
        return nothing
    end
    if parallel && Threads.nthreads() > 1 && n_splits > 1
        tasks = [Threads.@spawn run(s) for s in 1:n_splits]
        for t in tasks
            try
                fetch(t)
            catch e
                e isa TaskFailedException ? throw(e.task.exception) : rethrow()
            end
        end
    else
        foreach(run, 1:n_splits)
    end
    agg(A) = (median(A[:, 1]), median(A[:, 2]), median(A[:, 3]),
              min(1.0, 2 * median(A[:, 4])))
    rows = [agg(blp_est[:, j, :]) for j in 1:2]
    blp_tab = DataFrame(term=["ATE (β₁)", "HET (β₂)"], estimate=first.(rows),
                        lower=getindex.(rows, 2), upper=getindex.(rows, 3),
                        pvalue=last.(rows))
    grows = [agg(gates_est[:, j, :]) for j in 1:(K + 1)]
    gates_tab = DataFrame(group=vcat(["G$k" for k in 1:K], ["G$K - G1"]),
                          estimate=first.(grows), lower=getindex.(grows, 2),
                          upper=getindex.(grows, 3), pvalue=last.(grows))
    crow = [begin
                A = clan_est[:, j, :]
                (median(A[:, 1]), median(A[:, 2]), median(A[:, 3]), median(A[:, 4]),
                 median(A[:, 5]), min(1.0, 2 * median(A[:, 6])))
            end for j in 1:nc]
    clan_tab = DataFrame(covariate=string.(ccovs), most_affected=getindex.(crow, 1),
                         least_affected=getindex.(crow, 2), difference=getindex.(crow, 3),
                         lower=getindex.(crow, 4), upper=getindex.(crow, 5),
                         pvalue=getindex.(crow, 6))
    splits = Dict(:blp_estimate => blp_est[:, :, 1], :blp_pvalue => blp_est[:, :, 4],
                  :gates_estimate => gates_est[:, :, 1],
                  :clan_difference => clan_est[:, :, 3])
    return GenericMLInference(blp_tab, gates_tab, clan_tab, median(lam), median(lamb),
                              n_splits, K, Float64(level), _ml_learner_name(proxy_learner),
                              count(degenerate), n, splits)
end

# BLP, GATES and CLAN on one main sample. Returns arrays of
# (estimate, lower, upper, p-value) at per-split level `lvl`.
function _ml_cddf_split(y, d, p, B, S, C, K, lvl, cluster, ctx)
    n = length(y)
    w = 1 ./ (p .* (1 .- p))
    dp = d .- p
    cl, G = cluster === nothing ? (nothing, 0) : _ml_group_index(cluster)
    # a constant baseline proxy is absorbed by the intercept
    base = var(B) > 1e-12 * max(var(y), eps()) ? hcat(ones(n), B) : ones(n, 1)
    nb = size(base, 2)
    # BLP
    Sc = S .- mean(S)
    Z = hcat(base, dp, dp .* Sc)
    β, V = _ml_wls_sandwich(Z, y, w, cl, G)
    dof = cl === nothing ? n - size(Z, 2) : G - 1
    jb = (nb + 1):(nb + 2)
    blp = _ml_split_rows(β[jb], sqrt.(diag(V)[jb]), lvl, dof)
    # GATES: quantile groups of S (ties broken by position after a stable sort)
    ord = sortperm(S)
    grp = zeros(Int, n)
    for (r, i) in enumerate(ord)
        grp[i] = min(K, fld((r - 1) * K, n) + 1)
    end
    Zg = hcat(base, [dp .* (grp .== k) for k in 1:K]...)
    γ, Vg = _ml_wls_sandwich(Zg, y, w, cl, G)
    dofg = cl === nothing ? n - size(Zg, 2) : G - 1
    g1 = nb + 1
    γk = γ[g1:end]
    est = vcat(γk, γ[end] - γ[g1])
    se = vcat(sqrt.(diag(Vg)[g1:end]),
              sqrt(max(Vg[end, end] + Vg[g1, g1] - 2Vg[end, g1], 0.0)))
    gates = _ml_split_rows(est, se, lvl, dofg)
    # CLAN: most (G_K) minus least (G_1) affected group means (Welch)
    top = grp .== K
    bot = grp .== 1
    clan = zeros(size(C, 2), 6)
    for j in axes(C, 2)
        a = C[top, j]
        b = C[bot, j]
        diffj = mean(a) - mean(b)
        sej = sqrt(var(a) / length(a) + var(b) / length(b))
        z = critical_value(lvl)
        clan[j, :] .= (mean(a), mean(b), diffj, diffj - z * sej, diffj + z * sej,
                       sej > 0 ? two_sided_pvalue(diffj / sej) : (diffj == 0 ? 1.0 : 0.0))
    end
    lambda = β[nb + 2]^2 * var(S)
    lambda_bar = sum(abs2, γk) / K
    return (blp=blp, gates=gates, clan=clan, lambda=lambda, lambda_bar=lambda_bar)
end

function _ml_split_rows(est, se, lvl, dof)
    c = critical_value(lvl, dof)
    out = zeros(length(est), 4)
    for j in eachindex(est)
        out[j, :] .= (est[j], est[j] - c * se[j], est[j] + c * se[j],
                      two_sided_pvalue(est[j] / se[j], dof))
    end
    return out
end

"""
Weighted least squares with HC1 (or CR1: `G/(G-1)·(n-1)/(n-k)`) sandwich covariance,
identical to `FixedEffectModels.reg(...; weights)` with `Vcov.robust()` /
`Vcov.cluster`.
"""
function _ml_wls_sandwich(Z, y, w, cluster, G)
    n, k = size(Z)
    sw = sqrt.(w)
    Zw = Z .* sw
    F = qr(Zw, ColumnNorm())
    rank(F.R) == k || throw(ArgumentError("collinear regressors in a least-squares fit " *
                                          "(too few observations per group?)"))
    β = F \ (y .* sw)
    e = y .- Z * β
    A = inv(Symmetric(Zw' * Zw))
    S = Z .* (w .* e)
    M = if cluster === nothing
        n / (n - k) .* (S' * S)
    else
        Sg = zeros(G, k)
        for i in 1:n
            @views Sg[cluster[i], :] .+= S[i, :]
        end
        G / (G - 1) * (n - 1) / (n - k) .* (Sg' * Sg)
    end
    return β, Matrix(Symmetric(A * M * A))
end

"""
    blp(g::GenericMLInference) -> DataFrame

Best linear predictor table of [`generic_ml`](@ref): the average treatment effect
``\\beta_1`` and the heterogeneity loading ``\\beta_2`` of the BLP of the CATE given the
ML proxy, ``\\beta_1 + \\beta_2\\{S(X) - E\\,S(X)\\}``.

Estimates are medians over sample splits; `lower` and `upper` are the medians of the
per-split bounds computed at level ``1 - \\alpha/2`` (reported level `g.level`
``= 1 - \\alpha``), and `pvalue` is ``\\min(1, 2 \\times \\text{median } p)``, following
Chernozhukov, Demirer, Duflo and Fernández-Val (2025). This inference is valid,
typically conservative. ``\\beta_2`` is interpretable only relative to the proxy: a
value near 1 indicates a proxy that is well calibrated for the CATE, a value near 0 a
proxy that carries no information about effect heterogeneity.

# Arguments
- `g::GenericMLInference`: result of [`generic_ml`](@ref).

# Returns
- `DataFrame` with columns `term` (`"ATE (β₁)"`, `"HET (β₂)"`), `estimate`, `lower`,
  `upper`, `pvalue` (a copy; modifying it does not change `g`).

# Examples
```julia
using DrSnow, DataFrames, StableRNGs
rng = StableRNG(1)
n = 1000
df = DataFrame(x1=randn(rng, n), x2=randn(rng, n))
df.d = Float64.(rand(rng, n) .< 0.5)
df.y = df.x2 .+ (1 .+ df.x1) .* df.d .+ randn(rng, n)
g = generic_ml(df, :y, :d; covariates=[:x1, :x2], n_splits=20, rng=StableRNG(2))
blp(g)
```

# References
- Chernozhukov, V., Demirer, M., Duflo, E., & Fernández-Val, I. (2025). Fisher–Schultz
  lecture: Generic machine learning inference on heterogeneous treatment effects in
  randomized experiments, with an application to immunization in India.
  *Econometrica*, 93(4), 1121–1164.
"""
blp(g::GenericMLInference) = copy(g.blp)

"""
    blp_test(g::GenericMLInference) -> DiagnosticTest

Test of the null hypothesis ``H_0: \\beta_2 = 0`` in the best linear predictor of
[`generic_ml`](@ref), i.e. that the ML proxy ``S(X)`` does not predict
treatment-effect heterogeneity.

Rejection is evidence that the CATE varies with ``X`` (at least along the direction
the proxy captures). The p-value is the split-adjusted
``\\min(1, 2 \\times \\text{median } p)`` of Chernozhukov, Demirer, Duflo and
Fernández-Val (2025), which is valid, typically conservative, and the reported
statistic is the median estimate of ``\\beta_2``. A non-rejection does not show that
effects are homogeneous: the proxy may be too weak (e.g. a heavily regularized
learner or a small auxiliary sample) to detect heterogeneity that exists. Compare
GATES with [`gates`](@ref) and consider richer proxy learners before concluding.

# Arguments
- `g::GenericMLInference`: result of [`generic_ml`](@ref).

# Returns
- `DiagnosticTest`: statistic = median over splits of ``\\hat\\beta_2``; `pvalue` =
  split-adjusted p-value; `details` holds the BLP table and `lambda`.

# Examples
```julia
using DrSnow, DataFrames, StableRNGs
rng = StableRNG(1)
n = 1000
df = DataFrame(x1=randn(rng, n), x2=randn(rng, n))
df.d = Float64.(rand(rng, n) .< 0.5)
df.y = df.x2 .+ (1 .+ df.x1) .* df.d .+ randn(rng, n)
g = generic_ml(df, :y, :d; covariates=[:x1, :x2], n_splits=20, rng=StableRNG(2))
t = blp_test(g)
t.pvalue
```

# References
- Chernozhukov, V., Demirer, M., Duflo, E., & Fernández-Val, I. (2025). Fisher–Schultz
  lecture: Generic machine learning inference on heterogeneous treatment effects in
  randomized experiments, with an application to immunization in India.
  *Econometrica*, 93(4), 1121–1164.
"""
function blp_test(g::GenericMLInference)
    row = g.blp[2, :]
    return DiagnosticTest("BLP heterogeneity test (generic ML)",
                          "β₂ = 0: the ML proxy does not predict effect heterogeneity",
                          row.estimate, row.pvalue;
                          method="median over $(g.n_splits) sample splits " *
                                 "(p = min(1, 2 × median p))",
                          note="The statistic is the median estimate of β₂. " *
                               "Non-rejection can reflect a weak ML proxy and does " *
                               "not imply homogeneous effects.",
                          details=(table=copy(g.blp), lambda=g.lambda))
end

"""
    gates(g::GenericMLInference) -> DataFrame

Sorted group average treatment effects (GATES) from [`generic_ml`](@ref).

For the ``K`` quantile groups ``G_1, \\dots, G_K`` of the CATE proxy ``S(X)`` (G1
least, GK most affected according to the proxy), the estimand is
``\\gamma_k = E[Y(1) - Y(0) \\mid S(X) \\in G_k]``, plus the difference
``\\gamma_K - \\gamma_1``. Groups are defined by the proxy, so the GATES describe
the heterogeneity the proxy is able to sort, not the true CATE distribution; with a
good proxy they are monotone and the difference is large. Estimates are medians over
splits with median-rule intervals (per-split level ``1 - \\alpha/2``, reported level
`g.level`) and split-adjusted p-values ``\\min(1, 2 \\times \\text{median } p)``,
which are valid, typically conservative (Chernozhukov, Demirer, Duflo &
Fernández-Val 2025).

# Arguments
- `g::GenericMLInference`: result of [`generic_ml`](@ref).

# Returns
- `DataFrame` with columns `group` (`"G1"`, …, `"GK"`, `"GK - G1"`), `estimate`,
  `lower`, `upper`, `pvalue` (a copy).

# Examples
```julia
using DrSnow, DataFrames, StableRNGs
rng = StableRNG(1)
n = 1000
df = DataFrame(x1=randn(rng, n), x2=randn(rng, n))
df.d = Float64.(rand(rng, n) .< 0.5)
df.y = df.x2 .+ (1 .+ df.x1) .* df.d .+ randn(rng, n)
g = generic_ml(df, :y, :d; covariates=[:x1, :x2], n_splits=20, n_groups=4,
               rng=StableRNG(2))
gates(g)
```

# References
- Chernozhukov, V., Demirer, M., Duflo, E., & Fernández-Val, I. (2025). Fisher–Schultz
  lecture: Generic machine learning inference on heterogeneous treatment effects in
  randomized experiments, with an application to immunization in India.
  *Econometrica*, 93(4), 1121–1164.
"""
gates(g::GenericMLInference) = copy(g.gates)

"""
    clan(g::GenericMLInference) -> DataFrame

Classification analysis (CLAN) from [`generic_ml`](@ref): average characteristics of
the most and least affected groups.

For each CLAN covariate ``Z``, the estimands are ``E[Z \\mid S(X) \\in G_K]`` and
``E[Z \\mid S(X) \\in G_1]``, the means in the most and least affected groups defined
by the CATE proxy, and their difference. CLAN describes who the most and least
affected units are; it does not identify which characteristics cause effect
heterogeneity, since covariates are correlated with each other and with the proxy.
Reported values are medians over splits; the interval and p-value of the difference
use Welch standard errors within each split, with per-split level
``1 - \\alpha/2`` and split-adjusted p-value ``\\min(1, 2 \\times \\text{median } p)``,
valid, typically conservative (Chernozhukov, Demirer, Duflo & Fernández-Val 2025).

# Arguments
- `g::GenericMLInference`: result of [`generic_ml`](@ref).

# Returns
- `DataFrame` with columns `covariate`, `most_affected`, `least_affected`,
  `difference`, `lower`, `upper`, `pvalue` (a copy).

# Examples
```julia
using DrSnow, DataFrames, StableRNGs
rng = StableRNG(1)
n = 1000
df = DataFrame(x1=randn(rng, n), x2=randn(rng, n), age=rand(rng, 20:60, n))
df.d = Float64.(rand(rng, n) .< 0.5)
df.y = df.x2 .+ (1 .+ df.x1) .* df.d .+ randn(rng, n)
g = generic_ml(df, :y, :d; covariates=[:x1, :x2], clan_covariates=[:x1, :age],
               n_splits=20, rng=StableRNG(2))
clan(g)
```

# References
- Chernozhukov, V., Demirer, M., Duflo, E., & Fernández-Val, I. (2025). Fisher–Schultz
  lecture: Generic machine learning inference on heterogeneous treatment effects in
  randomized experiments, with an application to immunization in India.
  *Econometrica*, 93(4), 1121–1164.
"""
clan(g::GenericMLInference) = copy(g.clan)
