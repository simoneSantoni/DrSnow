# Analysis of (blocked) randomized experiments with the design's blocks and scores.

"""
    ExperimentEstimate <: CausalEstimate

Estimated average treatment effect of a (blocked) randomized experiment with its
standard error; the result of [`experiment_estimate`](@ref).

The object stores one scalar estimate, the standard error of the chosen variance
estimator and the degrees of freedom of its reference distribution, and supports the
package's [`CausalEstimate`](@ref) interface: `coef`, `stderror`, `vcov`,
`confint(r; level)`, `pvalues`, `nobs`, `dof_residual` and `tidy`. Intervals and
p-values use a t reference with `dof` degrees of freedom, or the normal when `dof` is
`Inf`.

# Fields
- `estimate::Float64`: estimated average treatment effect.
- `se::Float64`: standard error.
- `dof::Float64`: degrees of freedom of the t reference (`Inf`: normal reference).
- `method::Symbol`: `:difference`, `:block_fe`, `:lin` or `:dml`.
- `variance::String`: description of the variance estimator.
- `n::Int`: number of units.
- `n_treated::Int`: number of treated units.
- `n_blocks::Int`: number of blocks (0 without blocks).
- `covariates::Vector{Symbol}`: adjustment covariates, including the design score
  (named `:prognostic_score`) when it was reused.
- `block_estimates::Vector{Float64}`: within-block differences in means (`:difference`
  with blocks), empty otherwise.
"""
struct ExperimentEstimate <: CausalEstimate
    estimate::Float64
    se::Float64
    dof::Float64
    method::Symbol
    variance::String
    n::Int
    n_treated::Int
    n_blocks::Int
    covariates::Vector{Symbol}
    block_estimates::Vector{Float64}
end

StatsAPI.coef(r::ExperimentEstimate) = [r.estimate]
StatsAPI.vcov(r::ExperimentEstimate) = fill(r.se^2, 1, 1)
StatsAPI.coefnames(r::ExperimentEstimate) = ["ATE"]
StatsAPI.nobs(r::ExperimentEstimate) = r.n
StatsAPI.dof_residual(r::ExperimentEstimate) = r.dof
estimand(::ExperimentEstimate) = "ATE (sample average treatment effect)"
method_name(r::ExperimentEstimate) =
    Dict(:difference => "Difference in means", :block_fe => "Block fixed effects",
         :lin => "Lin (interacted) regression adjustment",
         :dml => "Cross-fitted AIPW (known assignment probabilities)")[r.method] *
    (r.n_blocks > 0 ? " ($(r.n_blocks) blocks)" : "")

function show_details(io::IO, r::ExperimentEstimate)
    println(io)
    print(io, "Variance: ", r.variance, "; treated ", r.n_treated, " of ", r.n)
    isempty(r.covariates) || print(io, "\nCovariates: ", join(r.covariates, ", "))
end

# Neyman difference in means, optionally within blocks (estimatr's difference_in_means).
function _des_neyman(y, z, blocks)
    if blocks === nothing
        y1, y0 = y[z .== 1], y[z .== 0]
        n1, n0 = length(y1), length(y0)
        (n1 >= 2 && n0 >= 2) ||
            throw(ArgumentError("experiment_estimate: need at least two treated and " *
                                "two control units"))
        v1, v0 = var(y1) / n1, var(y0) / n0
        se = sqrt(v1 + v0)
        dof = (v1 + v0)^2 / (v1^2 / (n1 - 1) + v0^2 / (n0 - 1))
        return mean(y1) - mean(y0), se, dof, "Neyman (Welch degrees of freedom)",
               Float64[]
    end
    g = _ri_groups(blocks)
    N = length(y)
    J = length(g)
    taus = Float64[]
    ws = Float64[]
    pairs = true
    for (_, mem) in g
        zb = z[mem]
        n1, n0 = count(==(1), zb), count(==(0), zb)
        (n1 >= 1 && n0 >= 1) ||
            throw(ArgumentError("experiment_estimate: every block needs treated and " *
                                "control units"))
        push!(taus, mean(y[mem][zb .== 1]) - mean(y[mem][zb .== 0]))
        push!(ws, length(mem))
        pairs &= length(mem) == 2
    end
    est = sum(ws .* taus) / N
    if pairs
        J >= 2 || throw(ArgumentError("experiment_estimate: need at least two pairs"))
        # Imai, King & Nall (2009) / estimatr: conservative matched-pair variance
        v = J / ((J - 1) * N^2) * sum((ws .* taus .- N * est / J) .^ 2)
        return est, sqrt(v), J - 1.0, "matched pairs (Imai, King & Nall 2009)", taus
    end
    v = 0.0
    for (k, (_, mem)) in enumerate(g)
        zb = z[mem]
        n1, n0 = count(==(1), zb), count(==(0), zb)
        (n1 >= 2 && n0 >= 2) ||
            throw(ArgumentError("experiment_estimate: the blocked Neyman variance " *
                                "needs two treated and two control units per block " *
                                "(or all blocks pairs); use method = :block_fe or :lin"))
        yb = y[mem]
        v += (ws[k] / N)^2 * (var(yb[zb .== 1]) / n1 + var(yb[zb .== 0]) / n0)
    end
    return est, sqrt(v), N - 2.0 * J, "blocked Neyman", taus
end

"""
    experiment_estimate(data, outcome, treatment; method=:difference, blocks=nothing,
                        covariates=Symbol[], id=nothing, learner=RidgeLearner(),
                        n_folds=5, vcov=Vcov.robust(), rng=Random.default_rng(),
                        probabilities=nothing) -> ExperimentEstimate
    experiment_estimate(data, outcome, treatment, bd::BlockingDesign;
                        method=:block_fe, use_score=true, kwargs...)
        -> ExperimentEstimate

Estimate the average treatment effect of a completely randomized, blocked or
matched-pair experiment, with or without covariate adjustment.

The estimand is the average treatment effect ``\\tau = E[Y(1) - Y(0)]`` over the
experimental units (or the population they were sampled from). Identification rests
on the design alone: assignment is random, possibly within blocks, with known
probabilities, and potential outcomes do not depend on other units' assignments (no
interference). Covariates and blocks never change the estimand; they only change
precision, and they must be measured before treatment. Four estimators are available.
`:difference` is the difference in means with the Neyman variance and Welch degrees of
freedom; with blocks it is the block-size-weighted average of within-block
differences, with the blocked Neyman variance and a t reference on ``N - 2B``
degrees of freedom. Both variances are conservative under effect heterogeneity, and
they reproduce `estimatr::difference_in_means`. `:block_fe` regresses the outcome on
treatment, block fixed effects and covariates with a heteroskedasticity-robust
variance by default. `:lin` is the regression of Lin (2013), with covariates centred
at their means and interacted with treatment (plus block fixed effects when there are
blocks). It answers Freedman's (2008) critique of ANCOVA: it is consistent for the
ATE, is asymptotically no less precise than the difference in means, and its robust
variance is conservative; without blocks it reproduces `estimatr::lm_lin`. `:dml` is
cross-fitted augmented inverse-probability weighting (Robins, Rotnitzky & Zhao 1994;
Chernozhukov et al. 2018) with the *known* assignment probabilities: outcome models
are fitted by `learner` separately in each arm and the covariates enter flexibly.
Because the propensity is known, the estimator is consistent for the ATE whatever
the learner (cf. Wager et al. 2016); a good learner only improves precision. Inference
uses the influence-function variance and a normal reference.

Matched pairs need special care. When every block is a pair, `:difference` uses the
variance of Imai, King & Nall (2009), which equals the variance of the "matched
pairs" t-test, with a t reference on ``J - 1`` degrees of freedom (``J`` pairs).
Bai, Romano & Shaikh (2022) show that, when pairs are formed on covariates that
predict the outcome, this test and the two-sample t-test are conservative: their
limiting rejection probability is at most, and typically below, the nominal level.
The asymptotically exact alternative, their adjusted variance built from "pairs of
pairs", is not implemented here; `randomization_test` with the design's mechanism
gives exact finite-sample tests of the sharp null. With `:block_fe` and treated shares
that differ across blocks, fixed effects weight blocks by ``n_b p_b (1 - p_b)``
rather than by size, so the coefficient is not the ATE under heterogeneous effects;
use `:difference`, `:lin` or `:dml` then.

With a [`BlockingDesign`](@ref), the design's blocks are used, and for `:lin`,
`:block_fe` and `:dml` with `use_score = true` the prognostic score that formed the
blocks enters as the covariate `:prognostic_score`, so the same prediction serves the
design and the analysis. Units are matched to the design by `bd.id`, and the observed
assignment must be possible under the design (the right number of treated units in
every block). Report the estimator, the variance estimator and the degrees of freedom
together with the estimate.

# Arguments
- `data::AbstractDataFrame`: one row per experimental unit.
- `outcome::Symbol`: outcome column (numeric).
- `treatment::Symbol`: 0/1 treatment column.
- `bd::BlockingDesign`: the design that generated the assignment (second method).

# Keywords
- `method::Symbol`: `:difference` (default without a design), `:block_fe` (default
  with a design), `:lin` or `:dml`.
- `blocks::Union{Nothing,Symbol} = nothing`: block column (not allowed with a design,
  whose blocks are used).
- `covariates::Vector{Symbol} = Symbol[]`: numeric baseline covariates; not allowed
  with `:difference`, required with `:dml` (unless the design supplies a score).
- `id::Union{Nothing,Symbol} = nothing`: unit identifier; fixes the unit order for the
  `:dml` folds so that the result does not depend on the row order.
- `learner = RidgeLearner()`: [`NuisanceLearner`](@ref) for the `:dml` outcome models.
- `n_folds::Integer = 5`: cross-fitting folds for `:dml`.
- `vcov = Vcov.robust()`: covariance estimator for `:block_fe` and `:lin`, e.g.
  `Vcov.cluster(:school)` for cluster-randomized designs.
- `rng::AbstractRNG = Random.default_rng()`: random-number generator for the `:dml`
  folds and learner.
- `probabilities = nothing`: known assignment probabilities per row for `:dml`
  (default: the treated share overall or within blocks; with a design, from its
  mechanism).
- `use_score::Bool = true`: with a design, add its prognostic score as a covariate
  (ignored for `:difference`).
- `kwargs...`: with a design, further keywords passed to the first method.

# Returns
- [`ExperimentEstimate`](@ref).

# Examples
```julia
using DrSnow, DataFrames, Random, StableRNGs
rng = StableRNG(1)
n = 200
df = DataFrame(id=1:n, x1=randn(rng, n), x2=randn(rng, n))
df.block = repeat(1:50, inner=4)
df.treated = vcat([shuffle(rng, [1, 1, 0, 0]) for _ in 1:50]...)
df.y = df.x1 .+ 0.5 .* df.x2 .+ 0.3 .* df.treated .+ randn(rng, n)
experiment_estimate(df, :y, :treated)                               # Neyman
experiment_estimate(df, :y, :treated; blocks=:block)                # blocked Neyman
experiment_estimate(df, :y, :treated; method=:lin, blocks=:block,
                    covariates=[:x1, :x2])                          # Lin (2013)
```

With a design built on a prognostic score (here from a baseline measurement):

```julia
using DrSnow, DataFrames, StableRNGs
rng = StableRNG(2)
n = 200
df = DataFrame(id=1:n, x1=randn(rng, n), x2=randn(rng, n))
df.y0 = df.x1 .+ 0.5 .* df.x2 .+ randn(rng, n)                     # baseline outcome
ps = prognostic_score(df, :y0; covariates=[:x1, :x2], id=:id, rng=StableRNG(3))
bd = block_design(df, ps; id=:id)
df2 = assign_treatment(bd, df; rng=StableRNG(4))
df2.y = df2.y0 .+ 0.3 .* df2.treated .+ 0.5 .* randn(rng, n)
experiment_estimate(df2, :y, :treated, bd; method=:lin)            # reuses the score
experiment_estimate(df2, :y, :treated, bd; method=:difference)     # matched pairs
```

# References
- Bai, Y., Romano, J. P., & Shaikh, A. M. (2022). Inference in experiments with
  matched pairs. *Journal of the American Statistical Association*, 117(540),
  1726–1737.
- Chernozhukov, V., Chetverikov, D., Demirer, M., Duflo, E., Hansen, C., Newey, W., &
  Robins, J. (2018). Double/debiased machine learning for treatment and structural
  parameters. *Econometrics Journal*, 21(1), C1–C68.
- Freedman, D. A. (2008). On regression adjustments to experimental data. *Advances in
  Applied Mathematics*, 40(2), 180–193.
- Imai, K., King, G., & Nall, C. (2009). The essential role of pair matching in
  cluster-randomized experiments, with application to the Mexican universal health
  insurance evaluation. *Statistical Science*, 24(1), 29–53.
- Lin, W. (2013). Agnostic notes on regression adjustments to experimental data:
  Reexamining Freedman's critique. *Annals of Applied Statistics*, 7(1), 295–318.
- Robins, J. M., Rotnitzky, A., & Zhao, L. P. (1994). Estimation of regression
  coefficients when some regressors are not always observed. *Journal of the
  American Statistical Association*, 89(427), 846–866.
- Wager, S., Du, W., Taylor, J., & Tibshirani, R. J. (2016). High-dimensional
  regression adjustments in randomized experiments. *Proceedings of the National
  Academy of Sciences*, 113(45), 12673–12678.
"""
function experiment_estimate(data::AbstractDataFrame, outcome::Symbol, treatment::Symbol;
                             method::Symbol=:difference,
                             blocks::Union{Nothing,Symbol}=nothing,
                             covariates::Vector{Symbol}=Symbol[],
                             id::Union{Nothing,Symbol}=nothing, learner=RidgeLearner(),
                             n_folds::Integer=5,
                             vcov::FixedEffectModels.CovarianceEstimator=Vcov.robust(),
                             rng::AbstractRNG=Random.default_rng(),
                             probabilities=nothing)
    ctx = "experiment_estimate"
    method in (:difference, :block_fe, :lin, :dml) ||
        throw(ArgumentError("$ctx: method must be :difference, :block_fe, :lin or :dml"))
    cols = vcat(outcome, treatment, covariates)
    blocks === nothing || push!(cols, blocks)
    id === nothing || push!(cols, id)
    require_columns(data, cols; context=ctx)
    (outcome in covariates || treatment in covariates) &&
        throw(ArgumentError("$ctx: outcome/treatment cannot be covariates"))
    y = _ml_column(data, outcome; context=ctx)
    z = _ml_column(data, treatment; context=ctx)
    _ml_check_binary_col(z, treatment, ctx)
    bl = blocks === nothing ? nothing : data[!, blocks]
    bl !== nothing && any(ismissing, bl) &&
        throw(ArgumentError("$ctx: block column has missing values"))
    n = length(y)
    n1 = count(==(1), z)
    nb = bl === nothing ? 0 : length(unique(bl))
    if method === :difference
        isempty(covariates) ||
            throw(ArgumentError("$ctx: method = :difference takes no covariates; use " *
                                ":lin, :block_fe or :dml"))
        est, se, dof, vdesc, taus = _des_neyman(y, z, bl)
        return ExperimentEstimate(est, se, dof, :difference, vdesc, n, n1, nb, Symbol[],
                                  taus)
    elseif method === :dml
        return _des_aipw(data, y, z, bl, covariates, id, learner, n_folds, rng,
                         probabilities, ctx)
    end
    # regression methods
    X = isempty(covariates) ? zeros(n, 0) : _ml_matrix(data, covariates; context=ctx)
    df = DataFrame(_des_y=y, _des_z=z)
    rhs = Symbol[:_des_z]
    for (j, c) in enumerate(covariates)
        xc = method === :lin ? X[:, j] .- mean(X[:, j]) : X[:, j]
        nm = Symbol("_des_x", j)
        df[!, nm] = xc
        push!(rhs, nm)
        if method === :lin
            ni = Symbol("_des_zx", j)
            df[!, ni] = z .* xc
            push!(rhs, ni)
        end
    end
    fe = Symbol[]
    if bl !== nothing
        df[!, :_des_block] = bl
        push!(fe, :_des_block)
    end
    vc_cols = _ri_vcov_columns(vcov)
    for c in vc_cols
        require_columns(data, [c]; context=ctx)
        hasproperty(df, c) || (df[!, c] = data[!, c])
    end
    m = FixedEffectModels.reg(df, make_formula(:_des_y, rhs; fe=fe), vcov;
                              progress_bar=false)
    i = coef_index(m, :_des_z)
    se = sqrt(StatsAPI.vcov(m)[i, i])
    dof = StatsAPI.dof_residual(m)
    vdesc = vcov isa Vcov.RobustCovariance ? "heteroskedasticity-robust (HC1)" :
            vcov isa Vcov.ClusterCovariance ? "cluster-robust" :
            vcov isa Vcov.SimpleCovariance ? "homoskedastic" :
            string(nameof(typeof(vcov)))
    return ExperimentEstimate(StatsAPI.coef(m)[i], se, float(dof), method, vdesc, n, n1,
                              nb, copy(covariates), Float64[])
end

# Cross-fitted AIPW with known assignment probabilities.
function _des_aipw(data, y, z, bl, covariates, id, learner, n_folds, rng, probs, ctx)
    isempty(covariates) &&
        throw(ArgumentError("$ctx: method = :dml needs covariates"))
    learner isa NuisanceLearner ||
        throw(ArgumentError("$ctx: learner must be a NuisanceLearner"))
    n = length(y)
    p = if probs !== nothing
        pv = Float64.(collect(probs))
        length(pv) == n || throw(DimensionMismatch("$ctx: probabilities length"))
        pv
    elseif bl === nothing
        fill(mean(z), n)
    else
        pb = Dict{Any,Float64}()
        for (b, mem) in _ri_groups(bl)
            pb[b] = mean(z[mem])
        end
        [pb[b] for b in bl]
    end
    all(x -> 0 < x < 1, p) ||
        throw(ArgumentError("$ctx: assignment probabilities must be in (0, 1)"))
    X = _ml_matrix(data, covariates; context=ctx)
    # canonical unit order for the folds
    rows = if id !== nothing
        sortperm(data[!, id]; by=_ri_sortkey)
    else
        tmp = DataFrame(X, :auto)
        xcols = propertynames(tmp)
        tmp[!, :_des_y] = y
        tmp[!, :_des_z] = z
        _ri_canonical_rows(tmp, vcat([:_des_y, :_des_z], xcols))
    end
    yr, zr, Xr, pr = y[rows], z[rows], X[rows, :], p[rows]
    F = crossfit_folds(n, n_folds, 1; rng=rng, strata=zr)
    seeds = _ml_seeds(rng, n_folds, 2, 1)
    specs = [_MLNuisance(:mu1, learner, yr, Xr, false, BitVector(zr .== 1)),
             _MLNuisance(:mu0, learner, yr, Xr, false, BitVector(zr .== 0))]
    P = _ml_crossfit(specs, F[:, 1], view(seeds, :, :, 1); parallel=true, context=ctx)
    μ1, μ0 = P[:, 1], P[:, 2]
    ψ = μ1 .- μ0 .+ zr .* (yr .- μ1) ./ pr .- (1 .- zr) .* (yr .- μ0) ./ (1 .- pr)
    θ = mean(ψ)
    se = sqrt(mean(abs2, ψ .- θ) / n)
    nb = bl === nothing ? 0 : length(unique(bl))
    return ExperimentEstimate(θ, se, Inf, :dml,
                              "influence function ($(n_folds)-fold cross-fitting, " *
                              _ml_learner_name(learner) * ")", n, count(==(1), z), nb,
                              copy(covariates), Float64[])
end

function experiment_estimate(data::AbstractDataFrame, outcome::Symbol, treatment::Symbol,
                             bd::BlockingDesign; method::Symbol=:block_fe,
                             use_score::Bool=true, covariates::Vector{Symbol}=Symbol[],
                             kwargs...)
    ctx = "experiment_estimate"
    haskey(kwargs, :blocks) &&
        throw(ArgumentError("$ctx: the blocks come from the design; do not pass `blocks`"))
    require_columns(data, [outcome, treatment]; context=ctx)
    idx = _des_design_index(bd, data, ctx)
    z = data[!, treatment]
    zc = falses(length(bd.ids))
    for (r, i) in enumerate(idx)
        zc[i] = z[r] == 1
    end
    _ri_in_support(bd.mechanism, zc) ||
        throw(ArgumentError("$ctx: the observed assignment is not possible under the " *
                            "design (wrong treated counts within blocks)"))
    df = DataFrame(data; copycols=true)
    df[!, :_des_design_block] = bd.blocks[idx]
    covs = copy(covariates)
    if use_score && bd.score !== nothing && method !== :difference
        :prognostic_score in propertynames(df) &&
            throw(ArgumentError("$ctx: data already has a column :prognostic_score"))
        df[!, :prognostic_score] = bd.score[idx]
        push!(covs, :prognostic_score)
    end
    probs = method === :dml ? treatment_probabilities(bd.mechanism)[idx] : nothing
    return experiment_estimate(df, outcome, treatment; method=method,
                               blocks=:_des_design_block, covariates=covs,
                               id=bd.id, probabilities=probs, kwargs...)
end
