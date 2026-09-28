# Double/debiased ML difference-in-differences for the 2×2 design (Chang 2020) using
# the doubly-robust scores of Sant'Anna & Zhao (2020) with cross-fitted nuisances.
#
# All weights are normalized to mean one (Hájek / "in-sample normalization"). The
# estimator is θ = Σ_c s_c mean(w_c e_c) over score components c, and the score used
# for inference is its exact influence function Σ_c s_c w_c (e_c - a_c) with
# a_c = mean(w_c e_c), which accounts for the estimated normalizing constants.
# Nuisance-estimation effects vanish by Neyman orthogonality and cross-fitting.

"""
Panel DR-DiD score (Sant'Anna & Zhao 2020, eq. 3.1 with normalized weights):
θ = E_n[w₁(ΔY - g₀)] - E_n[w₀(ΔY - g₀)], w₁ = D/mean(D),
w₀ ∝ m(1-D)/(1-m). Returns `(psi_a, psi_b)` of the linear score.
"""
function _ml_did_panel_score(dy, D, g0, m)
    e = dy .- g0
    w1 = D ./ mean(D)
    prop = m .* (1 .- D) ./ (1 .- m)
    w0 = prop ./ mean(prop)
    return _ml_hajek_score([(1.0, w1, e), (-1.0, w0, e)])
end

"""
Repeated cross-section locally efficient DR-DiD score (Sant'Anna & Zhao 2020,
eq. 3.4 with normalized weights), as in `DRDID::drdid_rc`.
"""
function _ml_did_rcs_score(y, D, T, g00, g01, g10, g11, m)
    g0y = T .* g01 .+ (1 .- T) .* g00
    e = y .- g0y
    prop = m .* (1 .- D) ./ (1 .- m)
    nz(w) = (s = mean(w); s > 0 ? w ./ s :
             throw(ArgumentError("dml_did: a (group, period) cell is empty")))
    w_tpost = nz(D .* T)
    w_tpre = nz(D .* (1 .- T))
    w_cpost = nz(prop .* T)
    w_cpre = nz(prop .* (1 .- T))
    w_d = nz(D)
    Δpost = g11 .- g01
    Δpre = g10 .- g00
    comps = [(1.0, w_tpost, e), (-1.0, w_tpre, e), (-1.0, w_cpost, e), (1.0, w_cpre, e),
             (1.0, w_d, Δpost), (-1.0, w_tpost, Δpost), (-1.0, w_d, Δpre),
             (1.0, w_tpre, Δpre)]
    return _ml_hajek_score(comps)
end

# ψ_a = -1, ψ_b = Σ_c s_c [w_c (e_c - a_c) + a_c]: mean(ψ_b) = θ and ψ_b - θ is the
# influence function of the Hájek-normalized estimator.
function _ml_hajek_score(comps)
    n = length(comps[1][2])
    psi_b = zeros(n)
    for (s, w, e) in comps
        a = mean(w .* e)
        psi_b .+= s .* (w .* (e .- a) .+ a)
    end
    return fill(-1.0, n), psi_b
end

"""
    dml_did(data, outcome, treatment; time, unit=nothing, covariates=Symbol[],
            outcome_learner=LassoLearner(), propensity_learner=PenalizedLogisticLearner(),
            trim=0.01, stratify=true, n_folds=5, n_rep=1, folds=nothing,
            cluster=nothing, rng=Random.default_rng(), parallel=true) -> DMLEstimate

Double/debiased machine-learning estimator of the average treatment effect on the
treated in the two-group, two-period difference-in-differences design with
covariates.

With periods ``t \\in \\{0, 1\\}``, a treatment group ``D = 1`` treated only in period
1, and potential outcomes ``Y_t(0)``, ``Y_t(1)``, the target is

```math
θ_0 = \\mathrm{ATT} = E[Y_1(1) - Y_1(0) \\mid D = 1].
```

It is identified under conditional parallel trends,
``E[Y_1(0) - Y_0(0) \\mid D = 1, X] = E[Y_1(0) - Y_0(0) \\mid D = 0, X]``, no
anticipation (``Y_0(1) = Y_0(0)`` for the treated), overlap
(``P(D = 1 \\mid X) < 1``) and, with repeated cross-sections, a joint distribution of
``(D, X)`` that is stable across the two periods. Conditioning on ``X`` allows the
untreated trend to depend on covariates, as in Abadie (2005). With two periods,
parallel trends cannot be tested; with more periods use [`dml_did_multi`](@ref) and
its pre-trend diagnostics.

The estimator uses the doubly robust scores of Sant'Anna and Zhao (2020), with
nuisances learned by machine learning and cross-fitted (Chang 2020). For panel data
(`unit` given), with ``ΔY = Y_1 - Y_0``, ``g_0(X) = E[ΔY \\mid D = 0, X]`` and
``m(X) = P(D = 1 \\mid X)``,

```math
\\hat θ = E_n\\big[w_1 \\{ΔY - g_0(X)\\}\\big] - E_n\\big[w_0 \\{ΔY - g_0(X)\\}\\big],
\\quad w_1 = \\frac{D}{E_n[D]}, \\quad
w_0 = \\frac{m(X)(1 - D)/\\{1 - m(X)\\}}{E_n[m(X)(1 - D)/\\{1 - m(X)\\}]},
```

that is, their equation (3.1) with normalized (Hájek) weights. Covariates are taken
from the pre-treatment period and the folds are drawn over units. For repeated
cross-sections (`unit = nothing`) the estimator is the locally efficient doubly
robust estimator of Sant'Anna and Zhao (2020, equation 3.4), as in R's
`DRDID::drdid_rc`, with outcome regressions ``g_{d,t}(X)`` fitted in each (group,
period) cell and folds stratified by the four cells. Both estimators are consistent
if either the outcome regressions or the propensity score are estimated
consistently, and the score is Neyman orthogonal, so ``\\sqrt{n}`` inference holds
when the product of the nuisance errors is ``o(n^{-1/2})``.

The variance is computed from the exact influence function of the Hájek-normalized
estimator, which accounts for the estimated normalizing constants; with `cluster`,
influence functions are summed within clusters and intervals use a ``t_{G-1}``
reference. Aggregation over repeated splits follows [`DMLEstimate`](@ref).
Propensities are clipped to ``[\\mathrm{trim}, 1 - \\mathrm{trim}]``; clipping keeps
the control weights finite but does not repair limited overlap (see
[`dml_irm`](@ref)). The parametric doubly robust estimator is [`did_drdid`](@ref);
comparing both shows how much the flexible nuisance models matter.

# Arguments
- `data`: a long `DataFrame`, one row per unit and period (panel) or per
  observation (repeated cross-sections).
- `outcome::Symbol`: the numeric outcome column.
- `treatment::Symbol`: the binary treatment-group indicator (1 for units treated in
  the post period, in both periods; constant within unit for panels).

# Keywords
- `time::Symbol`: the period column (required); it must take exactly two values,
  and the larger is the post period.
- `unit::Union{Nothing,Symbol} = nothing`: the unit identifier for a balanced panel;
  `nothing` treats the data as repeated cross-sections.
- `covariates::Vector{Symbol} = Symbol[]`: numeric covariates (pre-period values
  are used for panels).
- `outcome_learner = LassoLearner()`: the learner for the outcome regressions.
- `propensity_learner = PenalizedLogisticLearner()`: the learner for
  ``P(D = 1 \\mid X)``; it must implement [`fitpredict_proba`](@ref).
- `trim::Real = 0.01`: propensity clipping threshold in ``[0, 0.5)``.
- `stratify::Bool = true`: stratify folds by group (panels) or by group × period
  (cross-sections).
- `n_folds::Integer = 5`, `n_rep::Integer = 1`: folds and repetitions of
  cross-fitting.
- `folds = nothing`: for panels, a column name (constant within unit) or fold ids
  for the units in sorted order of `unit`; for cross-sections as in
  [`dml_plr`](@ref).
- `cluster = nothing`: one cluster column (constant within unit for panels).
- `rng = Random.default_rng()`, `parallel::Bool = true`: as in [`dml_plr`](@ref).

# Returns
- [`DMLEstimate`](@ref) with `model = :did` (panel, `nobs` = number of units) or
  `model = :did_cs` (repeated cross-sections).

# Examples
```julia
using DrSnow, DataFrames, StableRNGs
rng = StableRNG(1)
N = 600
x1, x2 = randn(rng, N), randn(rng, N)
treated = Float64.(rand(rng, N) .< 1 ./ (1 .+ exp.(-0.5 .* x1)))
α = randn(rng, N)
y0 = α .+ x1 .+ randn(rng, N)
y1 = α .+ x1 .+ 0.5 .+ 0.5 .* x1 .+ 1.0 .* treated .+ randn(rng, N)   # ATT = 1
panel = DataFrame(id=repeat(1:N, 2), year=repeat([2019, 2020]; inner=N),
                  y=vcat(y0, y1), treated=repeat(treated, 2),
                  x1=repeat(x1, 2), x2=repeat(x2, 2))
r = dml_did(panel, :y, :treated; time=:year, unit=:id, covariates=[:x1, :x2],
            rng=StableRNG(2))
r_cs = dml_did(panel, :y, :treated; time=:year, covariates=[:x1, :x2],
               rng=StableRNG(2))   # the same rows treated as cross-sections
```

# References
- Chang, N.-C. (2020). Double/debiased machine learning for difference-in-differences
  models. *The Econometrics Journal*, 23(2), 177–191.
- Sant'Anna, P. H. C., & Zhao, J. (2020). Doubly robust difference-in-differences
  estimators. *Journal of Econometrics*, 219(1), 101–122.
- Abadie, A. (2005). Semiparametric difference-in-differences estimators. *The
  Review of Economic Studies*, 72(1), 1–19.
- Chernozhukov, V., Chetverikov, D., Demirer, M., Duflo, E., Hansen, C., Newey, W., &
  Robins, J. (2018). Double/debiased machine learning for treatment and structural
  parameters. *The Econometrics Journal*, 21(1), C1–C68.
"""
function dml_did(data, outcome::Symbol, treatment::Symbol; time::Symbol,
                 unit::Union{Nothing,Symbol}=nothing, covariates=Symbol[],
                 outcome_learner=LassoLearner(),
                 propensity_learner=PenalizedLogisticLearner(), trim::Real=0.01,
                 stratify::Bool=true, n_folds::Integer=5, n_rep::Integer=1,
                 folds=nothing, cluster=nothing, rng::AbstractRNG=Random.default_rng(),
                 parallel::Bool=true)
    ctx = "dml_did"
    trim = _ml_check_trim(trim)
    _ml_check_common(n_folds, n_rep)
    covs = Symbol.(collect(covariates))
    cl = cluster === nothing ? Symbol[] :
         (cluster isa AbstractVector ? Symbol.(cluster) : [Symbol(cluster)])
    length(cl) <= 1 || throw(ArgumentError("$ctx: only one-way clustering is supported"))
    fcol = folds isa Symbol ? [folds] : Symbol[]
    require_columns(data, vcat(outcome, treatment, time, unit === nothing ? Symbol[] : unit,
                               covs, cl, fcol); context=ctx)
    tv = data[!, time]
    any(ismissing, tv) && throw(ArgumentError("$ctx: time column has missing values"))
    periods = sort(unique(tv))
    length(periods) == 2 ||
        throw(ArgumentError("$ctx: `time` must take exactly two values (2×2 design); " *
                            "got $(length(periods))"))
    post = Float64.(tv .== periods[2])
    y = _ml_column(data, outcome; context=ctx)
    D = _ml_column(data, treatment; context=ctx)
    _ml_check_binary_col(D, treatment, ctx)
    X = _ml_matrix(data, covs; context=ctx)
    if unit === nothing
        return _ml_did_rcs(data, y, D, post, X, cl, folds, n_folds, n_rep, rng,
                           outcome_learner, propensity_learner, trim, stratify, parallel,
                           treatment)
    end
    return _ml_did_panel(data, y, D, post, X, unit, cl, folds, n_folds, n_rep, rng,
                         outcome_learner, propensity_learner, trim, stratify, parallel,
                         treatment)
end

function _ml_did_panel(data, y, D, post, X, unit, cl, folds, n_folds, n_rep, rng,
                       outcome_learner, propensity_learner, trim, stratify, parallel,
                       treatment)
    ctx = "dml_did"
    uv = data[!, unit]
    any(ismissing, uv) && throw(ArgumentError("$ctx: unit column has missing values"))
    uidx, N = _ml_group_index(uv)
    pre_row = zeros(Int, N)
    post_row = zeros(Int, N)
    for i in eachindex(uidx)
        slot = post[i] == 1 ? post_row : pre_row
        slot[uidx[i]] == 0 ||
            throw(ArgumentError("$ctx: unit $(uv[i]) has several rows in the same period"))
        slot[uidx[i]] = i
    end
    if any(==(0), pre_row) || any(==(0), post_row)
        throw(ArgumentError("$ctx: the panel is unbalanced (some units are observed in " *
                            "only one period); drop them or use `unit = nothing` for " *
                            "repeated cross-sections"))
    end
    D[pre_row] == D[post_row] ||
        throw(ArgumentError("$ctx: the treatment-group indicator must be constant " *
                            "within unit"))
    d = D[pre_row]
    _ml_check_binary_col(d, "treatment group", ctx)
    dy = y[post_row] .- y[pre_row]
    Xu = X[pre_row, :]
    cid, G = nothing, 0
    if !isempty(cl)
        cv = data[!, cl[1]]
        any(ismissing, cv) &&
            throw(ArgumentError("$ctx: cluster column has missing values"))
        all(cv[pre_row] .== cv[post_row]) ||
            throw(ArgumentError("$ctx: the cluster variable must be constant within unit"))
        cid, G = _ml_group_index(cv[pre_row])
    end
    F = if folds isa Symbol
        fv = data[!, folds]
        fv[pre_row] == fv[post_row] ||
            throw(ArgumentError("$ctx: fold ids must be constant within unit"))
        _ml_resolve_folds(data, Int.(fv[pre_row]), N, n_folds, n_rep, rng, nothing,
                          cid; context=ctx)
    else
        _ml_resolve_folds(data, folds, N, n_folds, n_rep, rng, stratify ? d : nothing,
                          cid; context=ctx)
    end
    K = maximum(F)
    R = size(F, 2)
    seeds = _ml_seeds(rng, K, 2, R)
    ctrl = BitVector(d .== 0)
    rep_fn = function (r, _)
        specs = [_MLNuisance(:ml_g0, outcome_learner, dy, Xu, false, ctrl),
                 _MLNuisance(:ml_m, propensity_learner, d, Xu, true)]
        P = _ml_crossfit(specs, F[:, r], view(seeds, :, :, r); parallel=parallel,
                         context=ctx)
        g0, m = P[:, 1], P[:, 2]
        losses = [:ml_g0 => _ml_rmse(dy[ctrl], g0[ctrl]), :ml_m => _ml_logloss(d, m)]
        nt = _ml_clip!(m, trim)
        psi_a, psi_b = _ml_did_panel_score(dy, d, g0, m)
        return (psi_a=psi_a, psi_b=psi_b, preds=[:ml_g0 => g0, :ml_m => m],
                losses=losses, ntrim=nt)
    end
    out = _ml_run_dml(rep_fn, 1, F, cid, G)
    learners = [:ml_g0 => _ml_learner_name(outcome_learner),
                :ml_m => _ml_learner_name(propensity_learner)]
    return _ml_make_estimate(out, :did, :ATT, [string(treatment)], F, cid, G, N,
                             learners, trim, seeds, "ATT (2×2 DiD, panel)",
                             "DML difference-in-differences (panel, DR score)")
end

function _ml_did_rcs(data, y, D, post, X, cl, folds, n_folds, n_rep, rng,
                     outcome_learner, propensity_learner, trim, stratify, parallel,
                     treatment)
    ctx = "dml_did"
    n = length(y)
    cid, G = _ml_cluster_ids(data, isempty(cl) ? nothing : cl[1]; context=ctx)
    cells = 2 .* D .+ post
    for c in 0:3
        any(==(c), cells) ||
            throw(ArgumentError("$ctx: the (group, period) cell D = $(c ÷ 2), " *
                                "post = $(c % 2) is empty"))
    end
    F = _ml_resolve_folds(data, folds, n, n_folds, n_rep, rng,
                          stratify ? cells : nothing, cid; context=ctx)
    K = maximum(F)
    R = size(F, 2)
    seeds = _ml_seeds(rng, K, 5, R)
    masks = [BitVector(cells .== c) for c in 0:3]   # (d,t) = (0,0),(0,1),(1,0),(1,1)
    names = [:ml_g_d0_t0, :ml_g_d0_t1, :ml_g_d1_t0, :ml_g_d1_t1]
    rep_fn = function (r, _)
        specs = [_MLNuisance(names[c], outcome_learner, y, X, false, masks[c])
                 for c in 1:4]
        push!(specs, _MLNuisance(:ml_m, propensity_learner, D, X, true))
        P = _ml_crossfit(specs, F[:, r], view(seeds, :, :, r); parallel=parallel,
                         context=ctx)
        m = P[:, 5]
        losses = Pair{Symbol,Float64}[names[c] => _ml_rmse(y[masks[c]], P[masks[c], c])
                                      for c in 1:4]
        push!(losses, :ml_m => _ml_logloss(D, m))
        nt = _ml_clip!(m, trim)
        psi_a, psi_b = _ml_did_rcs_score(y, D, post, P[:, 1], P[:, 2], P[:, 3], P[:, 4], m)
        preds = Pair{Symbol,Vector{Float64}}[names[c] => P[:, c] for c in 1:4]
        push!(preds, :ml_m => m)
        return (psi_a=psi_a, psi_b=psi_b, preds=preds, losses=losses, ntrim=nt)
    end
    out = _ml_run_dml(rep_fn, 1, F, cid, G)
    learners = Pair{Symbol,String}[nm => _ml_learner_name(outcome_learner) for nm in names]
    push!(learners, :ml_m => _ml_learner_name(propensity_learner))
    return _ml_make_estimate(out, :did_cs, :ATT, [string(treatment)], F, cid, G, n,
                             learners, trim, seeds,
                             "ATT (2×2 DiD, repeated cross-sections)",
                             "DML difference-in-differences (repeated cross-sections, " *
                             "locally efficient DR score)")
end
