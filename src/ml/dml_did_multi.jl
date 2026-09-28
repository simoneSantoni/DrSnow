# Double/debiased machine learning for staggered / multi-period difference-in-
# differences: group-time average treatment effects ATT(g,t) (Callaway & Sant'Anna
# 2021) with cross-fitted nuisance functions in the doubly-robust scores of Sant'Anna
# & Zhao (2020) (Chang 2020 for the 2×2 building block), as in DoubleML's
# `DoubleMLDIDMulti`.
#
# Every ATT(g,t) is a 2×2 comparison between cohort g and its comparison group
# (never-treated or not-yet-treated units) between period t and a base period. Its
# nuisances (outcome regression(s) of the comparison group and the generalized
# propensity score P(G = g | X, cohort g or comparison group)) are cross-fitted on the
# units of that comparison with folds drawn once at the unit (or cluster) level, so all
# cells share the same sample split. The result is a `CallawaySantAnnaEstimate`, so
# `aggregate_att`, `pre_trend_test`, uniform bands, `honest_did` and plotting apply.

"""
    dml_did_multi(data, outcome, treatment, unit, time; covariates=Symbol[],
                  control_group=:never_treated, anticipation=0, base_period=:varying,
                  outcome_learner=LassoLearner(),
                  propensity_learner=PenalizedLogisticLearner(), trim=0.01,
                  n_folds=5, n_rep=1, folds=nothing, stratify=true, crossfit=true,
                  cluster=nothing, bootstrap=true, biters=999,
                  rng=Random.default_rng(), parallel=true) -> CallawaySantAnnaEstimate

Double/debiased machine-learning estimator of the group-time average treatment
effects of a staggered-adoption difference-in-differences design.

With cohorts ``g`` defined by the period of first treatment and potential outcomes
``Y_t(g)`` (treated from ``g`` on) and ``Y_t(∞)`` (never treated), the targets are
the group-time average treatment effects of Callaway and Sant'Anna (2021),

```math
\\mathrm{ATT}(g, t) = E[Y_t(g) - Y_t(∞) \\mid G = g],
```

and their aggregations into overall, cohort, calendar-time and event-time effects.
Identification requires, for each cohort, parallel trends conditional on
`covariates` with respect to the chosen comparison group (never-treated or
not-yet-treated units), no anticipation beyond `anticipation` periods, irreversible
(absorbing) treatment, and overlap (the generalized propensity score bounded away
from one). Pre-treatment ``\\mathrm{ATT}(g, t)`` estimates can be used to assess
parallel trends before treatment ([`pre_trend_test`](@ref)); a non-rejection does
not establish parallel trends after treatment, which [`honest_did`](@ref) relaxes.

Each ``\\mathrm{ATT}(g, t)`` is a two-group, two-period comparison between cohort
``g`` and its comparison group between period ``t`` and a base period, estimated with
the doubly robust score of Sant'Anna and Zhao (2020) with normalized (Hájek) weights
and machine-learned, cross-fitted nuisances (Chang 2020; Chernozhukov et al. 2018),
as in DoubleML's `DoubleMLDIDMulti`. For panel data, with ``ΔY = Y_t - Y_{base}``,
``g_0(X) = E[ΔY \\mid \\text{comparison}, X]`` and the generalized propensity score
``m(X) = P(G = g \\mid X)`` among cohort-``g`` and comparison units, the cell estimate
is ``E_n[w_1(ΔY - g_0)] - E_n[w_0(ΔY - g_0)]`` with ``w_1 ∝ 1\\{G = g\\}`` and
``w_0 ∝ m(1 - 1\\{G = g\\})/(1 - m)``. For repeated cross-sections (`unit = nothing`,
`treatment` a [`FirstTreated`](@ref) column) the locally efficient score with four
outcome regressions (cohort and comparison group, periods ``t`` and base) is used.
The nuisances of every cell are cross-fitted on that cell's observations with folds
drawn once over units (stratified by cohort and grouped by `cluster`), so that all
cells share one sample split and the joint influence-function covariance is valid.
The learners need only estimate the nuisances consistently with the product of the
outcome-regression and propensity errors vanishing faster than ``n^{-1/2}``. With
`OLSLearner()`, `LogisticLearner()` and `crossfit = false`, the point estimates
reproduce `did_callaway_santanna(...; method = :dr)` when that function's propensity
trimming does not bind.

Standard errors come from the influence functions (at the unit level, or summed
within clusters), and the multiplier bootstrap provides sup-t critical values for
simultaneous bands, as in [`did_callaway_santanna`](@ref). With `n_rep > 1`, each
``\\mathrm{ATT}(g, t)`` is the mean of its estimates over repetitions and the
influence function is the mean of the per-repetition influence functions; the
covariance is computed from this averaged influence function. Unlike the median
rule of [`DMLEstimate`](@ref), no split-to-split dispersion term is added, so the
standard errors do not reflect residual dependence on the sample split; compare the
per-repetition estimates in `settings.all_coef` to gauge it. The result is a
[`CallawaySantAnnaEstimate`](@ref) with `settings.method = :dml`, so
[`aggregate_att`](@ref), [`pre_trend_test`](@ref), `confint(r; uniform=true)`,
[`honest_did`](@ref) (on an event study with `base_period = :universal`) and the
plotting recipes apply unchanged. Propensity clipping at `trim` keeps the weights
finite but does not repair limited overlap (see [`dml_irm`](@ref)).

# Arguments
- `data`: a long `DataFrame` (a balanced panel; units missing some periods are
  dropped with a warning).
- `outcome::Symbol`: the numeric outcome column.
- `treatment`: an absorbing 0/1 treatment indicator column, or
  [`FirstTreated`](@ref)`(column)` giving each unit's first treatment period (never-treated
  units are coded `0` by default).
- `unit`: the unit identifier column, or `nothing` for repeated cross-sections.
- `time::Symbol`: the period column.

# Keywords
- `covariates::Vector{Symbol} = Symbol[]`: numeric pre-treatment covariates (values
  from the earlier period of each comparison, as in R's `did`).
- `control_group::Symbol = :never_treated`: `:never_treated` or `:not_yet_treated`.
- `anticipation::Integer = 0`: number of anticipation periods; the base period of
  cohort ``g`` is ``g - 1 - \\mathrm{anticipation}``.
- `base_period::Symbol = :varying`: `:varying` or `:universal` (see
  [`did_callaway_santanna`](@ref)).
- `outcome_learner = LassoLearner()`, `propensity_learner = PenalizedLogisticLearner()`:
  nuisance learners (any [`NuisanceLearner`](@ref), e.g. [`ForestLearner`](@ref) or
  [`MLJLearner`](@ref)); the propensity learner must implement
  [`fitpredict_proba`](@ref).
- `trim::Real = 0.01`: propensity scores are clipped to ``[\\mathrm{trim},
  1 - \\mathrm{trim}]``.
- `n_folds::Integer = 5`, `n_rep::Integer = 1`: folds and repetitions of
  cross-fitting (repetitions are averaged, see above).
- `folds = nothing`: a user-supplied sample split: a column name (constant within
  unit), a vector of fold ids for the units in sorted order of `unit` (for the
  observations in row order with repeated cross-sections), or a matrix with one
  column per repetition.
- `stratify::Bool = true`: stratify folds by cohort (cohort × period for repeated
  cross-sections).
- `crossfit::Bool = true`: `false` fits every nuisance on the full cell sample
  without sample splitting; only meant as a diagnostic cross-check.
- `cluster::Union{Nothing,Symbol} = nothing`: a unit-invariant cluster column; folds
  are drawn at the cluster level and influence functions are summed within
  clusters.
- `bootstrap::Bool = true`, `biters::Integer = 999`: multiplier bootstrap
  (Rademacher weights at the cluster level) for uniform bands.
- `rng::AbstractRNG = Random.default_rng()`: random-number generator for folds,
  learner seeds and the bootstrap.
- `parallel::Bool = true`: fit folds × nuisances on threads (results are
  identical).

# Returns
- [`CallawaySantAnnaEstimate`](@ref) whose `settings` additionally hold `learners`,
  `n_folds`, `n_rep`, `folds` (fold ids per unit or observation and repetition),
  `crossfit`, `trim`, `n_trimmed`, `all_coef` (``\\mathrm{ATT}(g,t)`` ×
  repetitions) and `nuisance_loss` (a `DataFrame` of out-of-fold RMSE and log loss
  per cell).

# Examples
```julia
using DrSnow, DataFrames, StableRNGs
rng = StableRNG(1)
N, T = 300, 4
x1 = randn(rng, N)
cohort = [rand(rng) < 0.3 ? 0 : rand(rng, 3:4) for _ in 1:N]   # 0 = never treated
df = DataFrame(id=repeat(1:N; inner=T), year=repeat(1:T, N),
               first_treat=repeat(cohort; inner=T), x1=repeat(x1; inner=T))
df.y = 0.2 .* df.year .+ df.year .* 0.3 .* df.x1 .+
       ((df.first_treat .> 0) .& (df.year .>= df.first_treat)) .+ randn(rng, N * T)
r = dml_did_multi(df, :y, FirstTreated(:first_treat), :id, :year;
                  covariates=[:x1], n_folds=3, rng=StableRNG(2))
es = aggregate_att(r, :dynamic)           # event study
confint(es; uniform=true)                 # simultaneous bands
aggregate_att(r, :group)
```

# References
- Callaway, B., & Sant'Anna, P. H. C. (2021). Difference-in-differences with multiple
  time periods. *Journal of Econometrics*, 225(2), 200–230.
- Sant'Anna, P. H. C., & Zhao, J. (2020). Doubly robust difference-in-differences
  estimators. *Journal of Econometrics*, 219(1), 101–122.
- Chang, N.-C. (2020). Double/debiased machine learning for difference-in-differences
  models. *The Econometrics Journal*, 23(2), 177–191.
- Chernozhukov, V., Chetverikov, D., Demirer, M., Duflo, E., Hansen, C., Newey, W., &
  Robins, J. (2018). Double/debiased machine learning for treatment and structural
  parameters. *The Econometrics Journal*, 21(1), C1–C68.
- Bach, P., Chernozhukov, V., Kurz, M. S., & Spindler, M. (2022). DoubleML – An
  object-oriented implementation of double machine learning in Python. *Journal of
  Machine Learning Research*, 23(53), 1–6.
"""
function dml_did_multi(data, outcome::Symbol, treatment, unit, time::Symbol;
                       covariates::Vector{Symbol}=Symbol[],
                       control_group::Symbol=:never_treated, anticipation::Integer=0,
                       base_period::Symbol=:varying, outcome_learner=LassoLearner(),
                       propensity_learner=PenalizedLogisticLearner(), trim::Real=0.01,
                       n_folds::Integer=5, n_rep::Integer=1, folds=nothing,
                       stratify::Bool=true, crossfit::Bool=true,
                       cluster::Union{Nothing,Symbol}=nothing, bootstrap::Bool=true,
                       biters::Integer=999, rng::AbstractRNG=Random.default_rng(),
                       parallel::Bool=true)
    ctx = "dml_did_multi"
    _did_check_control_group(control_group)
    base_period in (:varying, :universal) ||
        throw(ArgumentError("$ctx: base_period must be :varying or :universal"))
    anticipation >= 0 || throw(ArgumentError("$ctx: anticipation must be ≥ 0"))
    trim = _ml_check_trim(trim)
    _ml_check_common(n_folds, n_rep)
    biters > 0 || throw(ArgumentError("$ctx: biters must be positive"))
    δ = Int(anticipation)
    tcol = _did_treatment_column(treatment)
    fcol = folds isa Symbol ? folds : nothing
    opts = (; control_group, δ, base_period, outcome_learner, propensity_learner, trim,
            n_folds=Int(n_folds), n_rep=Int(n_rep), folds, stratify, crossfit, cluster,
            bootstrap, biters=Int(biters), rng, parallel, covariates, ctx)
    if unit === nothing
        cols = [outcome, tcol, time, covariates..., cluster, fcol]
        df, tm, G, Tuse, excl = _did_cs_rc_sample(data, cols, treatment, time,
                                                  control_group, δ; context=ctx)
        return _ml_didm_rc(df, tm, G, Tuse, excl, outcome, opts)
    end
    cols = [outcome, tcol, unit, time, covariates..., cluster, fcol]
    df, tm, G, Tuse, excl = _did_cs_panel_sample(data, cols, treatment, unit, time,
                                                 control_group, δ; context=ctx)
    return _ml_didm_panel(df, tm, G, Tuse, excl, outcome, opts)
end

# Group-time cells (g, t, base) in the order of did_callaway_santanna; cells without
# comparison units are skipped with a warning.
function _ml_didm_cells(G, Tuse, excl, o, periods)
    glist = sort!(unique(filter(g -> g > 0 && g != excl, G)))
    isempty(glist) && throw(ArgumentError("$(o.ctx): no treated cohorts"))
    cells = Vector{NamedTuple{(:g, :t, :base),Tuple{Int,Int,Int}}}()
    skipped = String[]
    for g in glist
        pret_g = g - o.δ - 1
        for t in (o.base_period === :varying ? (2:Tuse) : (1:Tuse))
            base = (o.base_period === :universal || g <= t) ? pret_g : t - 1
            base == t && continue
            ctrl = _did_control_units(G, g, t, base, o.control_group, o.δ)
            if !any(ctrl)
                push!(skipped, "(g=$(periods[g]), t=$(periods[t]))")
                continue
            end
            push!(cells, (g=g, t=t, base=base))
        end
    end
    isempty(skipped) || @warn "$(o.ctx): no comparison units for " *
                              join(skipped, ", ") * "; these ATT(g,t) are not reported"
    isempty(cells) && throw(ArgumentError("$(o.ctx): no estimable ATT(g,t)"))
    return cells
end

# Cluster value of each unit (observation for repeated cross-sections), checked to be
# constant within unit; returns (values, index 1:G, G).
function _ml_didm_clusters(df, cluster, rowkey, N, labels, ctx)
    cluster === nothing && return nothing, nothing, 0
    cv = df[!, cluster]
    ucl = Vector{eltype(cv)}(undef, N)
    seen = falses(N)
    for i in 1:nrow(df)
        u = rowkey[i]
        if seen[u] && !isequal(ucl[u], cv[i])
            throw(ArgumentError("$ctx: cluster variable `$cluster` varies within unit " *
                                "$(labels[u]); it must be unit-invariant"))
        end
        ucl[u] = cv[i]
        seen[u] = true
    end
    cid, G = _ml_group_index(ucl)
    return ucl, cid, G
end

# Fold matrix over the N units (observations for repeated cross-sections).
function _ml_didm_folds(df, o, rowkey, N, strata, cid)
    f = o.folds
    if f isa Symbol
        fv = df[!, f]
        any(ismissing, fv) &&
            throw(ArgumentError("$(o.ctx): fold column has missing values"))
        uf = zeros(Int, N)
        for i in 1:nrow(df)
            u = rowkey[i]
            (uf[u] == 0 || uf[u] == fv[i]) ||
                throw(ArgumentError("$(o.ctx): fold ids must be constant within unit"))
            uf[u] = Int(fv[i])
        end
        f = uf
    end
    return _ml_resolve_folds(df, f, N, o.n_folds, o.n_rep, o.rng,
                             o.stratify ? strata : nothing, cid; context=o.ctx)
end

# Relabel the fold ids of a cell to 1:K' (folds absent from the cell are dropped).
function _ml_didm_cell_folds(f::AbstractVector{Int}, label, ctx)
    ks = sort!(unique(f))
    length(ks) >= 2 ||
        throw(ArgumentError("$ctx: $label: all observations of the cell fall in one " *
                            "fold; use fewer folds"))
    pos = Dict(k => i for (i, k) in enumerate(ks))
    return [pos[k] for k in f]
end

# Nuisances fitted on the whole (masked) cell sample and predicted in-sample: the
# no-cross-fitting diagnostic (`crossfit = false`).
function _ml_didm_insample(specs::Vector{_MLNuisance}, seeds::AbstractMatrix{UInt64},
                           ctx)
    n = length(specs[1].target)
    out = Matrix{Float64}(undef, n, length(specs))
    for (s, sp) in enumerate(specs)
        train = sp.train_mask === nothing ? trues(n) : sp.train_mask
        any(train) || throw(ArgumentError("$ctx: no training observations for " *
                                          "nuisance $(sp.name)"))
        trng = Random.Xoshiro(seeds[1, s])
        out[:, s] = sp.proba ?
            fitpredict_proba(sp.learner, sp.X[train, :], sp.target[train], sp.X;
                             rng=trng) :
            fitpredict(sp.learner, sp.X[train, :], sp.target[train], sp.X; rng=trng)
        all(isfinite, view(out, :, s)) || throw(ArgumentError(
            "$ctx: learner for $(sp.name) returned non-finite predictions"))
    end
    return out
end

# Out-of-fold (or in-sample) nuisance predictions of one cell; errors name the cell.
function _ml_didm_fit_cell(specs, fold, seeds, o, label)
    try
        return o.crossfit ?
            _ml_crossfit(specs, _ml_didm_cell_folds(fold, label, o.ctx), seeds;
                         parallel=o.parallel, context=o.ctx) :
            _ml_didm_insample(specs, seeds, o.ctx)
    catch err
        err isa ArgumentError || rethrow()
        occursin(label, err.msg) && rethrow()
        throw(ArgumentError("$(o.ctx): $label: " * replace(err.msg, "$(o.ctx): " => "")))
    end
end

function _ml_didm_panel(df, tm, G, Tuse, excl, outcome, o)
    ctx = o.ctx
    N, T = length(tm.units), length(tm.periods)
    k = length(o.covariates)
    y = _ml_column(df, outcome; context=ctx)
    Xr = _ml_matrix(df, o.covariates; context=ctx)
    Y = fill(NaN, N, T)
    Xt = zeros(N, k, T)
    for i in 1:nrow(df)
        u, p = tm.row_unit[i], tm.row_period[i]
        Y[u, p] = y[i]
        Xt[u, :, p] = Xr[i, :]
    end
    ucl, cid, Gc = _ml_didm_clusters(df, o.cluster, tm.row_unit, N, tm.units, ctx)
    F = _ml_didm_folds(df, o, tm.row_unit, N, G, cid)
    R = size(F, 2)
    cells = _ml_didm_cells(G, Tuse, excl, o, tm.periods)
    C = length(cells)
    seeds = _ml_seeds(o.rng, maximum(F), 2, R * C)
    att = zeros(C, R)
    Ψ = zeros(N, C)
    loss = zeros(C, 2)
    ntrim = 0
    for (c, cell) in enumerate(cells)
        g, t, base = cell
        label = "ATT(g=$(tm.periods[g]), t=$(tm.periods[t]))"
        treated = G .== g
        idx = findall(treated .| _did_control_units(G, g, t, base, o.control_group, o.δ))
        d = Float64.(treated[idx])
        dy = Y[idx, t] .- Y[idx, base]
        X = Xt[idx, :, min(t, base)]
        ctrl = BitVector(d .== 0)
        for r in 1:R
            specs = [_MLNuisance(:ml_g0, o.outcome_learner, dy, X, false, ctrl),
                     _MLNuisance(:ml_m, o.propensity_learner, d, X, true)]
            P = _ml_didm_fit_cell(specs, F[idx, r], view(seeds, :, :, (r - 1) * C + c),
                                  o, label)
            g0, m = P[:, 1], P[:, 2]
            loss[c, 1] += _ml_rmse(dy[ctrl], g0[ctrl]) / R
            loss[c, 2] += _ml_logloss(d, m) / R
            ntrim += _ml_clip!(m, o.trim)
            _, psi_b = _ml_did_panel_score(dy, d, g0, m)
            θ = mean(psi_b)
            att[c, r] = θ
            Ψ[idx, c] .+= (N / length(idx)) .* (psi_b .- θ) ./ R
        end
    end
    learners = [:ml_g0 => _ml_learner_name(o.outcome_learner),
                :ml_m => _ml_learner_name(o.propensity_learner)]
    return _ml_didm_result(cells, att, Ψ, tm, G, ones(N), ucl, N * Tuse, Tuse, F,
                           learners, loss, ntrim, o, true)
end

function _ml_didm_rc(df, tm, G, Tuse, excl, outcome, o)
    ctx = o.ctx
    n = nrow(df)
    P = tm.row_period
    y = _ml_column(df, outcome; context=ctx)
    X = _ml_matrix(df, o.covariates; context=ctx)
    rowkey = collect(1:n)
    ucl, cid, Gc = _ml_didm_clusters(df, o.cluster, rowkey, n, rowkey, ctx)
    strata = G .* (length(tm.periods) + 1) .+ P
    F = _ml_didm_folds(df, o, rowkey, n, strata, cid)
    R = size(F, 2)
    cells = _ml_didm_cells(G, Tuse, excl, o, tm.periods)
    C = length(cells)
    seeds = _ml_seeds(o.rng, maximum(F), 5, R * C)
    names = [:ml_g_d0_t0, :ml_g_d0_t1, :ml_g_d1_t0, :ml_g_d1_t1]
    att = zeros(C, R)
    Ψ = zeros(n, C)
    loss = zeros(C, 5)
    ntrim = 0
    for (c, cell) in enumerate(cells)
        g, t, base = cell
        label = "ATT(g=$(tm.periods[g]), t=$(tm.periods[t]))"
        ctrl = _did_control_units(G, g, t, base, o.control_group, o.δ)
        idx = findall(((G .== g) .| ctrl) .& ((P .== t) .| (P .== base)))
        D = Float64.(G[idx] .== g)
        post = Float64.(P[idx] .== t)
        yc = y[idx]
        Xc = X[idx, :]
        masks = [BitVector((D .== a) .& (post .== b))
                 for (a, b) in ((0, 0), (0, 1), (1, 0), (1, 1))]
        for (j, mk) in enumerate(masks)
            any(mk) || throw(ArgumentError(
                "$ctx: $label: the (group, period) cell of nuisance $(names[j]) is empty"))
        end
        for r in 1:R
            specs = [_MLNuisance(names[j], o.outcome_learner, yc, Xc, false, masks[j])
                     for j in 1:4]
            push!(specs, _MLNuisance(:ml_m, o.propensity_learner, D, Xc, true))
            Pr = _ml_didm_fit_cell(specs, F[idx, r],
                                   view(seeds, :, :, (r - 1) * C + c), o, label)
            m = Pr[:, 5]
            for j in 1:4
                loss[c, j] += _ml_rmse(yc[masks[j]], Pr[masks[j], j]) / R
            end
            loss[c, 5] += _ml_logloss(D, m) / R
            ntrim += _ml_clip!(m, o.trim)
            _, psi_b = _ml_did_rcs_score(yc, D, post, Pr[:, 1], Pr[:, 2], Pr[:, 3],
                                         Pr[:, 4], m)
            θ = mean(psi_b)
            att[c, r] = θ
            Ψ[idx, c] .+= (n / length(idx)) .* (psi_b .- θ) ./ R
        end
    end
    learners = Pair{Symbol,String}[nm => _ml_learner_name(o.outcome_learner)
                                   for nm in names]
    push!(learners, :ml_m => _ml_learner_name(o.propensity_learner))
    return _ml_didm_result(cells, att, Ψ, tm, G, ones(n), ucl, n, Tuse, F, learners,
                           loss, ntrim, o, false)
end

function _ml_didm_result(cells, att, Ψ, tm, G, uw, ucl, nobs, Tuse, F, learners, loss,
                         ntrim, o, panel)
    groups = [c.g for c in cells]
    times = [c.t for c in cells]
    bases = [c.base for c in cells]
    coef = vec(mean(att; dims=2))
    V, ncl = _if_vcov(Ψ, ucl)
    supt = o.bootstrap ? last(_multiplier_bootstrap(o.rng, Ψ, ucl, o.biters)) :
           Float64[]
    lossdf = DataFrame(group=collect(tm.periods[groups]), time=collect(tm.periods[times]))
    for (j, (nm, _)) in enumerate(learners)
        lossdf[!, nm] = loss[:, j]
    end
    settings = (control_group=o.control_group, anticipation=o.δ,
                base_period=o.base_period, method=:dml, covariates=o.covariates,
                bootstrap=o.bootstrap, biters=o.biters,
                units=panel ? tm.units : Int[], n_periods_used=Tuse, panel=panel,
                learners=learners, n_folds=maximum(F), n_rep=size(F, 2), folds=F,
                crossfit=o.crossfit, trim=o.trim, n_trimmed=ntrim, all_coef=att,
                nuisance_loss=lossdf)
    return CallawaySantAnnaEstimate(groups, times, bases, coef, V, Ψ, tm.periods, G, uw,
                                    ucl, o.cluster === nothing ? 0 : ncl, nobs, supt,
                                    settings)
end
