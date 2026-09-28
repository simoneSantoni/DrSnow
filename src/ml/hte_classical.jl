# Classical heterogeneity analyses: pre-specified subgroup effects (with a joint test
# of equal effects and multiplicity-adjusted p-values) and treatment × moderator
# interactions, for randomized experiments (regression) and for unconfoundedness
# designs (cross-fitted doubly robust scores), all regressions through
# FixedEffectModels.

function _ml_fresh(df, base::AbstractString)
    names_ = Set(propertynames(df))
    s = Symbol(base)
    k = 0
    while s in names_
        k += 1
        s = Symbol(base, "_", k)
    end
    return s
end

function _ml_hte_vcov(cluster, vcov)
    vcov === nothing || return vcov
    cluster === nothing && return Vcov.robust()
    cs = cluster isa AbstractVector ? Symbol.(cluster) : [Symbol(cluster)]
    return Vcov.cluster(cs...)
end

"""Cross-fitted AIPW scores φ for the binary treatment (rows of `data`)."""
function _ml_hte_scores(data, outcome, treatment, covariates, cluster, outcome_learner,
                        propensity_learner, trim, n_folds, folds, rng, parallel, ctx)
    trim = _ml_check_trim(trim)
    d = _ml_column(data, treatment; context=ctx)
    _ml_check_binary_col(d, treatment, ctx)
    isempty(covariates) &&
        throw(ArgumentError("$(ctx): method = :aipw needs covariates for the nuisances"))
    covs, X, cid, G, F = _ml_setup(data, [outcome, treatment], covariates, cluster, folds,
                                   n_folds, 1, rng, d; context=ctx)
    size(F, 2) == 1 || throw(ArgumentError("$(ctx): supply a single column of fold ids"))
    y = _ml_column(data, outcome; context=ctx)
    seeds = _ml_seeds(rng, maximum(F), 3, 1)
    _, _, _, nt, φ = _ml_aipw_nuisances(y, d, X, F[:, 1], view(seeds, :, :, 1),
                                        outcome_learner, propensity_learner, trim,
                                        parallel, ctx)
    return φ, nt
end

function _ml_hte_adjust(p, adjust, ctx)
    adjust === :holm && return holm_adjust(p)
    adjust === :bh && return bh_adjust(p)
    adjust === :bonferroni && return min.(1.0, length(p) .* p)
    adjust === :none && return copy(p)
    throw(ArgumentError("$(ctx): adjust must be :holm, :bh, :bonferroni or :none"))
end

"""
    subgroup_effects(data, outcome, treatment, subgroup; method=:regression,
                     covariates=Symbol[], cluster=nothing, vcov=nothing,
                     weights=nothing, adjust=:holm, outcome_learner=LassoLearner(),
                     propensity_learner=PenalizedLogisticLearner(), trim=0.01,
                     n_folds=5, folds=nothing, rng=Random.default_rng(),
                     parallel=true) -> HTEEstimate
    subgroup_effects(f::CausalForest, groups; adjust=:holm) -> HTEEstimate

Average treatment effects within pre-specified subgroups, with a joint test of equal
effects and p-values adjusted for testing one effect per subgroup.

For a discrete subgroup variable ``G`` with levels ``g = 1, \\dots, K``, the estimands
are the subgroup average treatment effects

```math
\\tau_g = E[Y(1) - Y(0) \\mid G = g], \\qquad g = 1, \\dots, K,
```

and the joint null ``H_0: \\tau_1 = \\dots = \\tau_K``. The subgroups must be defined by
pre-treatment characteristics and fixed before the outcome data are examined; the
tests are invalid for subgroups chosen after looking at the outcomes, a central
point of the reporting guidelines of Wang et al. (2007). For data-driven groups with
valid inference see [`generic_ml`](@ref) (GATES) or honest partitioning in the spirit
of Athey and Imbens (2016).

Two estimators are available. With `method = :regression`, for randomized
experiments, ``Y`` is regressed on the treatment interacted with subgroup indicators,
subgroup fixed effects and, optionally, `covariates` with common slopes, via
FixedEffectModels. The coefficient on ``D \\cdot 1\\{G = g\\}`` estimates ``\\tau_g``;
randomization (possibly with subgroup-specific assignment probabilities) is the only
identifying assumption, and covariate adjustment only affects precision (Lin 2013
discusses its properties and the benefits of full interactions). With
`method = :aipw`, for observational data under unconfoundedness given `covariates`
and overlap, cross-fitted doubly robust scores

```math
\\hat\\varphi_i = \\hat\\mu_1(X_i) - \\hat\\mu_0(X_i)
  + \\frac{D_i\\{Y_i - \\hat\\mu_1(X_i)\\}}{\\hat e(X_i)}
  - \\frac{(1 - D_i)\\{Y_i - \\hat\\mu_0(X_i)\\}}{1 - \\hat e(X_i)}
```

are regressed on the subgroup indicators, so each coefficient is the AIPW estimate of
``\\tau_g``, a special case of the projections of Semenova and Chernozhukov (2021).
The forest method uses the doubly robust scores [`get_scores`](@ref) of a fitted
[`causal_forest`](@ref) in the same way (with the forest's observation weights and
clusters). Propensities are clipped to `[trim, 1 - trim]`, which keeps the scores
finite but does not repair limited overlap.

Standard errors are heteroskedasticity-robust, cluster-robust with `cluster`, or as
given by `vcov`. `heterogeneity_test(r)` returns the Wald test of equal subgroup
effects (``F`` with finite degrees of freedom, otherwise ``\\chi^2``); non-rejection
does not show that effects are homogeneous, since subgroup tests have low power and
say nothing about heterogeneity along other variables. Per-subgroup p-values are
adjusted for multiplicity with Holm's (1979) step-down procedure (family-wise error
rate) by default, or by Benjamini–Hochberg (false discovery rate), Bonferroni, or not
at all. Report all pre-specified subgroups, the joint test and the adjusted p-values.

# Arguments
- `data`: a `DataFrame` with one row per unit.
- `outcome::Symbol`: numeric outcome column ``Y``.
- `treatment::Symbol`: treatment column ``D`` (binary `0`/`1` for `:aipw`; must vary
  within each subgroup for `:regression`).
- `subgroup::Symbol`: column with the subgroup labels (any sortable values, no
  missing values, at least two levels).
- `f::CausalForest` (forest method): a fitted [`causal_forest`](@ref).
- `groups` (forest method): one subgroup label per training observation of `f`.

# Keywords
- `method::Symbol = :regression`: `:regression` (randomized experiments) or `:aipw`
  (unconfoundedness; needs `covariates`).
- `covariates::Vector{Symbol} = Symbol[]`: adjustment covariates (`:regression`) or
  confounders for the nuisance learners (`:aipw`).
- `cluster = nothing`: cluster column(s) for cluster-robust standard errors (and
  grouped folds for `:aipw`).
- `vcov = nothing`: a `FixedEffectModels` covariance estimator that overrides
  `cluster`.
- `weights = nothing`: column of regression weights.
- `adjust::Symbol = :holm`: p-value adjustment, `:holm`, `:bh`, `:bonferroni` or
  `:none`.
- `outcome_learner = LassoLearner()`, `propensity_learner =
  PenalizedLogisticLearner()`, `trim::Real = 0.01`, `n_folds::Integer = 5`,
  `folds = nothing`, `rng::AbstractRNG = Random.default_rng()`,
  `parallel::Bool = true`: cross-fitting of the `:aipw` scores, as in
  [`dml_irm`](@ref) with one repetition.

# Returns
- [`HTEEstimate`](@ref) with one coefficient per subgroup (`"<subgroup> = <label>"`;
  `"group = <label>"` for the forest method). `heterogeneity_test(r)` gives the joint
  test; `r.details` holds `p_raw`, `p_adjusted`, `adjust`, `labels`, `method`,
  `n_trimmed` and the test as `heterogeneity`.

# Examples
```julia
using DrSnow, DataFrames, StableRNGs
rng = StableRNG(1)
n = 1200
df = DataFrame(x1=randn(rng, n), x2=randn(rng, n),
               region=rand(rng, ["north", "south", "west"], n))
df.d = Float64.(rand(rng, n) .< 0.5)
df.y = df.x1 .+ (1 .+ (df.region .== "south")) .* df.d .+ randn(rng, n)
r = subgroup_effects(df, :y, :d, :region; covariates=[:x1, :x2])
heterogeneity_test(r)
r.details.p_adjusted
```

# References
- Wang, R., Lagakos, S. W., Ware, J. H., Hunter, D. J., & Drazen, J. M. (2007).
  Statistics in medicine — reporting of subgroup analyses in clinical trials. *New
  England Journal of Medicine*, 357(21), 2189–2194.
- Holm, S. (1979). A simple sequentially rejective multiple test procedure.
  *Scandinavian Journal of Statistics*, 6(2), 65–70.
- Lin, W. (2013). Agnostic notes on regression adjustments to experimental data:
  Reexamining Freedman's critique. *The Annals of Applied Statistics*, 7(1),
  295–318.
- Semenova, V., & Chernozhukov, V. (2021). Debiased machine learning of conditional
  average treatment effects and other causal functions. *The Econometrics Journal*,
  24(2), 264–289.
- Athey, S., & Imbens, G. (2016). Recursive partitioning for heterogeneous causal
  effects. *Proceedings of the National Academy of Sciences*, 113(27), 7353–7360.
"""
function subgroup_effects(data, outcome::Symbol, treatment::Symbol, subgroup::Symbol;
                          method::Symbol=:regression, covariates=Symbol[],
                          cluster=nothing, vcov=nothing, weights=nothing,
                          adjust::Symbol=:holm, outcome_learner=LassoLearner(),
                          propensity_learner=PenalizedLogisticLearner(),
                          trim::Real=0.01, n_folds::Integer=5, folds=nothing,
                          rng::AbstractRNG=Random.default_rng(), parallel::Bool=true)
    ctx = "subgroup_effects"
    method in (:regression, :aipw) ||
        throw(ArgumentError("$(ctx): method must be :regression or :aipw"))
    covs = Symbol.(collect(covariates))
    cl = cluster === nothing ? Symbol[] :
         (cluster isa AbstractVector ? Symbol.(cluster) : [Symbol(cluster)])
    wc = weights === nothing ? Symbol[] : [Symbol(weights)]
    require_columns(data, vcat(outcome, treatment, subgroup, covs, cl, wc); context=ctx)
    any(ismissing, data[!, subgroup]) &&
        throw(ArgumentError("$(ctx): the subgroup column has missing values"))
    labels = sort(unique(data[!, subgroup]))
    length(labels) >= 2 || throw(ArgumentError("$(ctx): need at least two subgroups"))
    grp = data[!, subgroup]
    names = ["$(subgroup) = $(l)" for l in labels]
    base = DataFrame()
    for c in unique(vcat(cl, wc))
        base[!, c] = data[!, c]
    end
    dcols = [_ml_fresh(base, "_ml_sg$(k)") for k in eachindex(labels)]
    vc = _ml_hte_vcov(cluster, vcov)
    nt = 0
    if method === :regression
        y = _ml_column(data, outcome; context=ctx)
        d = _ml_column(data, treatment; context=ctx)
        yc = _ml_fresh(base, "_ml_y")
        base[!, yc] = y
        gc = _ml_fresh(base, "_ml_group")
        base[!, gc] = [findfirst(==(g), labels) for g in grp]
        for c in covs
            base[!, c] = _ml_column(data, c; context=ctx)
        end
        for (k, l) in enumerate(labels)
            base[!, dcols[k]] = d .* (grp .== l)
        end
        for (k, l) in enumerate(labels)
            dk = d[grp .== l]
            length(unique(dk)) >= 2 ||
                throw(ArgumentError("$(ctx): the treatment does not vary in subgroup $(l)"))
        end
        f = make_formula(yc, vcat(dcols, covs); fe=[gc])
        m = FixedEffectModels.reg(base, f, vc; weights=weights, progress_bar=false)
        idx = [coef_index(m, c) for c in dcols]
        b = coef(m)[idx]
        V = Matrix(StatsAPI.vcov(m)[idx, idx])
        dof = Float64(StatsAPI.dof_residual(m))
        mname = "Subgroup effects (regression with subgroup fixed effects)"
    else
        φ, nt = _ml_hte_scores(data, outcome, treatment, covs, cluster, outcome_learner,
                               propensity_learner, trim, n_folds, folds, rng, parallel,
                               ctx)
        b, V, dof = _ml_hte_score_regression(base, φ, grp, labels, dcols, vc, weights, ctx)
        mname = "Subgroup effects (cross-fitted AIPW scores)"
    end
    return _ml_subgroup_result(names, b, V, dof, nrow(data), adjust, mname,
                               (n_trimmed=nt, method=method, labels=labels), ctx)
end

function subgroup_effects(f::CausalForest, groups; adjust::Symbol=:holm)
    ctx = "subgroup_effects"
    length(groups) == nobs(f) ||
        throw(DimensionMismatch("$(ctx): groups must have one label per observation"))
    any(ismissing, groups) && throw(ArgumentError("$(ctx): groups have missing values"))
    labels = sort(unique(groups))
    length(labels) >= 2 || throw(ArgumentError("$(ctx): need at least two subgroups"))
    Γ = get_scores(f)
    w = _ml_grf_obs_weights(f)
    Z = hcat([Float64.(groups .== l) for l in labels]...)
    cl = f.cluster
    b, V = _ml_wls_sandwich(Z, Γ, w, cl, f.n_clusters)
    dof = cl === nothing ? nobs(f) - size(Z, 2) : f.n_clusters - 1.0
    return _ml_subgroup_result(["group = $(l)" for l in labels], b, V, dof, nobs(f),
                               adjust, "Subgroup effects (causal forest AIPW scores)",
                               (n_trimmed=0, method=:forest, labels=labels), ctx)
end

function _ml_hte_score_regression(base, φ, grp, labels, dcols, vc, weights, ctx)
    yc = _ml_fresh(base, "_ml_phi")
    base[!, yc] = φ
    for (k, l) in enumerate(labels)
        base[!, dcols[k]] = Float64.(grp .== l)
    end
    f = make_formula(yc, dcols; intercept=false)
    m = FixedEffectModels.reg(base, f, vc; weights=weights, progress_bar=false)
    idx = [coef_index(m, c) for c in dcols]
    return coef(m)[idx], Matrix(StatsAPI.vcov(m)[idx, idx]),
           Float64(StatsAPI.dof_residual(m))
end

function _ml_subgroup_result(names, b, V, dof, n, adjust, mname, extra, ctx)
    k = length(b)
    se = sqrt.(max.(diag(V), 0.0))
    p = two_sided_pvalue.(b ./ se, dof)
    padj = _ml_hte_adjust(p, adjust, ctx)
    R = hcat(ones(k - 1), -Matrix{Float64}(I, k - 1, k - 1))
    wt = wald_test(b, V; R=R, dof=dof)
    het = DiagnosticTest("Test of equal subgroup effects",
                         "the average treatment effect is the same in every subgroup",
                         wt.statistic, wt.pvalue;
                         dof=isfinite(dof) ? (k - 1, dof) : (k - 1,),
                         method=isfinite(dof) ? "Wald F test" : "Wald χ² test",
                         note="Non-rejection does not show that effects are " *
                              "homogeneous; tests of pre-specified subgroups have low " *
                              "power against heterogeneity along other variables.",
                         details=(chi2=wt.chi2,))
    return HTEEstimate(names, b, Matrix(Symmetric(V)), n, dof,
                       "average treatment effect by subgroup", mname,
                       merge(extra, (heterogeneity=het, p_raw=p, p_adjusted=padj,
                                     adjust=adjust)))
end

"""
    interaction_effects(data, outcome, treatment, moderators; method=:regression,
                        covariates=Symbol[], center=true, cluster=nothing,
                        vcov=nothing, weights=nothing, outcome_learner=LassoLearner(),
                        propensity_learner=PenalizedLogisticLearner(), trim=0.01,
                        n_folds=5, folds=nothing, rng=Random.default_rng(),
                        parallel=true) -> HTEEstimate

Treatment-effect heterogeneity along pre-specified continuous or binary moderators,
estimated by linear treatment × moderator interactions.

For moderators ``M = (M_1, \\dots, M_J)``, the target is the linear approximation of
the conditional average treatment effect
``\\tau(m) = E[Y(1) - Y(0) \\mid M = m]``,

```math
\\tau(m) \\approx \\tau + \\sum_{j=1}^J \\gamma_j (m_j - \\bar m_j),
```

where ``\\gamma_j`` is the change in the effect per unit of ``M_j`` holding the other
moderators fixed and, with centred moderators, ``\\tau`` is the effect at the moderator
means (the average treatment effect when the CATE is linear in ``M``). Interactions
describe effect modification: how the effect of the treatment co-varies with the
moderators. They are not estimates of the causal effect of changing a moderator,
which would require the moderator itself to be as good as randomly assigned.

With `method = :regression`, for randomized experiments, FixedEffectModels estimates

```math
Y = \\alpha + \\tau D + \\sum_j \\gamma_j D(M_j - \\bar M_j) + \\sum_j \\delta_j M_j
  + \\beta' Z + \\varepsilon,
```

with optional adjustment covariates ``Z`` (Lin 2013). Randomization identifies the
coefficients as the best linear approximation above; the linear specification is an
approximation when the CATE is nonlinear in ``M``. With `method = :aipw`, for
observational data under unconfoundedness given `covariates` and overlap,
cross-fitted doubly robust scores (as in [`subgroup_effects`](@ref)) are regressed on
``(1, M - \\bar M)``; the coefficients estimate the best linear projection of the CATE
on the moderators, which is valid without assuming linearity (Semenova & Chernozhukov
2021; compare [`cate_projection`](@ref) and [`best_linear_projection`](@ref)).

Standard errors are heteroskedasticity-robust, cluster-robust with `cluster`, or as
given by `vcov`. `heterogeneity_test(r)` is the joint Wald test that all interaction
coefficients are zero; non-rejection does not show that effects are homogeneous,
because they may vary nonlinearly in ``M`` or along other variables. Moderators must be
pre-treatment variables chosen before the outcome data are examined; post-treatment
moderators can be affected by the treatment and break the interpretation.

# Arguments
- `data`: a `DataFrame` with one row per unit.
- `outcome::Symbol`: numeric outcome column ``Y``.
- `treatment::Symbol`: treatment column ``D`` (binary `0`/`1` for `:aipw`).
- `moderators::Vector{Symbol}`: numeric moderator columns (encode categories as
  dummies).

# Keywords
- `method::Symbol = :regression`: `:regression` (randomized experiments) or `:aipw`
  (unconfoundedness; needs `covariates`).
- `covariates::Vector{Symbol} = Symbol[]`: adjustment covariates (`:regression`;
  moderators are skipped) or confounders for the nuisance learners (`:aipw`).
- `center::Bool = true`: centre the moderators at their (weighted) means, so the main
  effect is the effect at the means; with `false` it is the effect at ``M = 0``.
- `cluster = nothing`: cluster column(s) for cluster-robust standard errors.
- `vcov = nothing`: a `FixedEffectModels` covariance estimator that overrides
  `cluster`.
- `weights = nothing`: column of regression weights (also used for centring).
- `outcome_learner = LassoLearner()`, `propensity_learner =
  PenalizedLogisticLearner()`, `trim::Real = 0.01`, `n_folds::Integer = 5`,
  `folds = nothing`, `rng::AbstractRNG = Random.default_rng()`,
  `parallel::Bool = true`: cross-fitting of the `:aipw` scores, as in
  [`dml_irm`](@ref) with one repetition.

# Returns
- [`HTEEstimate`](@ref) with coefficients `"<treatment>"` and
  `"<treatment> × <moderator>"`; `heterogeneity_test(r)` gives the joint test and
  `r.details` holds `method`, `moderators`, `centered` and `n_trimmed`.

# Examples
```julia
using DrSnow, DataFrames, StableRNGs
rng = StableRNG(1)
n = 1000
df = DataFrame(age=rand(rng, 20:60, n), female=Float64.(rand(rng, n) .< 0.5),
               school=rand(rng, 1:40, n))
df.d = Float64.(rand(rng, n) .< 0.5)
df.y = (1 .+ 0.05 .* (df.age .- 40)) .* df.d .+ randn(rng, n)
r = interaction_effects(df, :y, :d, [:age, :female]; cluster=:school)
coeftable(r)
heterogeneity_test(r)
```

# References
- Lin, W. (2013). Agnostic notes on regression adjustments to experimental data:
  Reexamining Freedman's critique. *The Annals of Applied Statistics*, 7(1),
  295–318.
- Semenova, V., & Chernozhukov, V. (2021). Debiased machine learning of conditional
  average treatment effects and other causal functions. *The Econometrics Journal*,
  24(2), 264–289.
- Wang, R., Lagakos, S. W., Ware, J. H., Hunter, D. J., & Drazen, J. M. (2007).
  Statistics in medicine — reporting of subgroup analyses in clinical trials. *New
  England Journal of Medicine*, 357(21), 2189–2194.
"""
function interaction_effects(data, outcome::Symbol, treatment::Symbol, moderators;
                             method::Symbol=:regression, covariates=Symbol[],
                             center::Bool=true, cluster=nothing, vcov=nothing,
                             weights=nothing, outcome_learner=LassoLearner(),
                             propensity_learner=PenalizedLogisticLearner(),
                             trim::Real=0.01, n_folds::Integer=5, folds=nothing,
                             rng::AbstractRNG=Random.default_rng(), parallel::Bool=true)
    ctx = "interaction_effects"
    method in (:regression, :aipw) ||
        throw(ArgumentError("$(ctx): method must be :regression or :aipw"))
    mods = Symbol.(collect(moderators))
    isempty(mods) && throw(ArgumentError("$(ctx): give at least one moderator"))
    covs = Symbol.(collect(covariates))
    cl = cluster === nothing ? Symbol[] :
         (cluster isa AbstractVector ? Symbol.(cluster) : [Symbol(cluster)])
    wc = weights === nothing ? Symbol[] : [Symbol(weights)]
    require_columns(data, vcat(outcome, treatment, mods, covs, cl, wc); context=ctx)
    base = DataFrame()
    for c in unique(vcat(cl, wc))
        base[!, c] = data[!, c]
    end
    M = _ml_matrix(data, mods; context=ctx)
    wv = weights === nothing ? ones(nrow(data)) : _ml_column(data, Symbol(weights);
                                                             context=ctx)
    Mc = center ? M .- (sum(M .* wv; dims=1) ./ sum(wv)) : M
    mcols = [_ml_fresh(base, "_ml_m$(j)") for j in eachindex(mods)]
    for j in eachindex(mods)
        base[!, mcols[j]] = Mc[:, j]
    end
    vc = _ml_hte_vcov(cluster, vcov)
    names = vcat(string(treatment), ["$(treatment) × $(m)" for m in mods])
    nt = 0
    if method === :regression
        y = _ml_column(data, outcome; context=ctx)
        d = _ml_column(data, treatment; context=ctx)
        yc = _ml_fresh(base, "_ml_y")
        dc = _ml_fresh(base, "_ml_d")
        base[!, yc] = y
        base[!, dc] = d
        icols = [_ml_fresh(base, "_ml_dm$(j)") for j in eachindex(mods)]
        for j in eachindex(mods)
            base[!, icols[j]] = d .* Mc[:, j]
        end
        ccols = Symbol[]
        for c in covs
            c in mods && continue
            cc = _ml_fresh(base, "_ml_c_$(c)")
            base[!, cc] = _ml_column(data, c; context=ctx)
            push!(ccols, cc)
        end
        f = make_formula(yc, vcat(dc, icols, mcols, ccols))
        m = FixedEffectModels.reg(base, f, vc; weights=weights, progress_bar=false)
        idx = [coef_index(m, c) for c in vcat(dc, icols)]
        mname = "Treatment × moderator interactions (regression)"
    else
        φ, nt = _ml_hte_scores(data, outcome, treatment, covs, cluster, outcome_learner,
                               propensity_learner, trim, n_folds, folds, rng, parallel,
                               ctx)
        yc = _ml_fresh(base, "_ml_phi")
        base[!, yc] = φ
        f = make_formula(yc, mcols)
        m = FixedEffectModels.reg(base, f, vc; weights=weights, progress_bar=false)
        idx = vcat(coef_index(m, "(Intercept)"), [coef_index(m, c) for c in mcols])
        mname = "Best linear projection of the CATE on moderators (AIPW scores)"
    end
    b = coef(m)[idx]
    V = Matrix(StatsAPI.vcov(m)[idx, idx])
    dof = Float64(StatsAPI.dof_residual(m))
    k = length(mods)
    R = hcat(zeros(k), Matrix{Float64}(I, k, k))
    wt = wald_test(b, V; R=R, dof=dof)
    het = DiagnosticTest("Test of no effect modification",
                         "all treatment × moderator coefficients are zero",
                         wt.statistic, wt.pvalue;
                         dof=isfinite(dof) ? (k, dof) : (k,),
                         method=isfinite(dof) ? "Wald F test" : "Wald χ² test",
                         note="Non-rejection does not show that effects are " *
                              "homogeneous: effects may vary nonlinearly or along " *
                              "other variables.",
                         details=(chi2=wt.chi2,))
    est = method === :regression ?
          (center ? "effect at the moderator means and effect modification" :
           "effect at moderators = 0 and effect modification") :
          "best linear projection of the CATE on the moderators"
    return HTEEstimate(names, b, Matrix(Symmetric(V)), nrow(data), dof, est, mname,
                       (heterogeneity=het, n_trimmed=nt, method=method,
                        moderators=mods, centered=center))
end
