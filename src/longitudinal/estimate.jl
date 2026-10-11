# Result type of the longitudinal g-methods.

"""
    LongitudinalEstimate <: CausalEstimate

Result of [`longitudinal_regime`](@ref): the estimated mean counterfactual outcome
``E[Y^d]`` under each treatment regime ``d``, with the influence-function (or
bootstrap) covariance of the estimates and the fitted nuisance quantities needed to
audit them.

The coefficients are one per regime, in the order the regimes were given, named by
their labels. The covariance is the sample covariance of the estimated efficient
influence functions divided by ``n`` (clustered when `cluster` was given), or the
covariance of the bootstrap replicates when `bootstrap > 0`. For the g-computation
estimator (`:gcomp`) no influence-function variance is reported: the plug-in
influence function of the sequential regressions is not the influence function of the
substitution estimator, so `vcov`, `stderror` and `confint` throw unless the estimate
was fitted with `bootstrap > 0` ([`tidy`](@ref) then reports `missing` inference).
Because the full influence-function matrix is stored, [`regime_contrast`](@ref) and
[`regime_msm`](@ref) can derive differences, ratios and marginal structural model
coefficients with the correct joint covariance.

Supports `coef`, `vcov`, `stderror`, `confint(r; level)`, `coeftable`, `nobs` (the
number of units), `dof_residual` (``G - 1`` with ``G`` clusters, otherwise `Inf`),
`estimand`, `method_name`, [`tidy`](@ref), [`glance`](@ref),
[`regime_weights`](@ref) and [`plot_regime_weights`](@ref).

# Fields
- `estimator::Symbol`: `:sdr`, `:tmle`, `:gcomp` or `:ipw`.
- `regimes::Vector{TreatmentRegime}`, `names::Vector{String}`: the regimes and their
  labels.
- `coef::Vector{Float64}`: the estimates ``\\hat ψ_d`` on the outcome's scale.
- `vcov::Union{Nothing,Matrix{Float64}}`: the reported covariance (`nothing` for
  `:gcomp` without bootstrap).
- `if_vcov::Union{Nothing,Matrix{Float64}}`: the influence-function covariance, kept
  when a bootstrap covariance is reported instead.
- `eif::Union{Nothing,Matrix{Float64}}`: `n × R` estimated efficient influence
  functions (first repetition of the sample split), scaled so that
  ``\\hat ψ - ψ ≈ n^{-1} \\sum_i \\mathrm{eif}_i``.
- `bootstrap_coef::Matrix{Float64}`: `B × R` bootstrap replicates (`0 × R` without
  bootstrap).
- `all_coef::Matrix{Float64}`: `R × n_rep` estimates of every repetition of the
  sample split (aggregated by the median rule of [`DMLEstimate`](@ref)).
- `outcome_regressions::Vector{Matrix{Float64}}`: per regime, the `n × (K + 1)`
  out-of-fold sequential regressions ``\\hat Q_1, …, \\hat Q_K`` (targeted for
  `:tmle`) and the outcome in the last column, on the outcome's scale; empty for
  `:ipw`.
- `weights::Vector{Matrix{Float64}}`: per regime, the `n × K` cumulative inverse
  probability weights ``1\\{\\bar A_t = \\bar d_t\\} / \\max(\\hat g_{1:t}, g_{bound})``.
- `cumulative_probability::Vector{Matrix{Float64}}`: per regime, the unbounded
  ``\\hat g_{1:t} = \\prod_{s ≤ t} \\hat g_s(d_s \\mid H_s)``.
- `followers::Vector{BitMatrix}`: per regime, whether each unit follows the regime
  through each period.
- `propensities::Matrix{Float64}`: `n × K` out-of-fold ``\\hat P(A_t = 1 \\mid H_t)``.
- `n_bounded::Matrix{Int}`: `R × K` number of followers whose cumulative probability
  was raised to `g_bound`.
- `treatment_logloss::Vector{Float64}`: out-of-fold log loss of the treatment
  mechanism in each period.
- `folds::Matrix{Int}`: fold ids of the units, `n × n_rep` (all ones without
  cross-fitting).
- `cluster::Union{Nothing,Vector{Int}}`, `n_clusters::Int`: cluster index of every
  unit and the number of clusters (0 without clustering).
- `n::Int`, `K::Int`: number of units and periods; `units`, `times`: their labels
  in the order of the rows and columns above.
- `learners::Vector{Pair{Symbol,String}}`: the outcome and treatment learners.
- `g_bound::Float64`, `q_bound::Float64`: the bounds used.
- `outcome_range::Tuple{Float64,Float64}`: the range the outcome was scaled with
  (`(0, 1)` when no scaling was needed).
- `normalize::Bool`: whether the IPW estimator was Hájek-normalized.
- `seeds::Vector{UInt64}`: the per-task seeds drawn from `rng`.
"""
struct LongitudinalEstimate <: CausalEstimate
    estimator::Symbol
    regimes::Vector{TreatmentRegime}
    names::Vector{String}
    coef::Vector{Float64}
    vcov::Union{Nothing,Matrix{Float64}}
    if_vcov::Union{Nothing,Matrix{Float64}}
    eif::Union{Nothing,Matrix{Float64}}
    bootstrap_coef::Matrix{Float64}
    all_coef::Matrix{Float64}
    outcome_regressions::Vector{Matrix{Float64}}
    weights::Vector{Matrix{Float64}}
    cumulative_probability::Vector{Matrix{Float64}}
    followers::Vector{BitMatrix}
    propensities::Matrix{Float64}
    n_bounded::Matrix{Int}
    treatment_logloss::Vector{Float64}
    folds::Matrix{Int}
    cluster::Union{Nothing,Vector{Int}}
    n_clusters::Int
    n::Int
    K::Int
    units::Vector{Any}
    times::Vector{Any}
    learners::Vector{Pair{Symbol,String}}
    g_bound::Float64
    q_bound::Float64
    outcome_range::Tuple{Float64,Float64}
    normalize::Bool
    seeds::Vector{UInt64}
end

const _LG_METHOD_NAMES = Dict(
    :sdr => "Sequentially doubly robust estimator (longitudinal)",
    :tmle => "Longitudinal targeted minimum loss-based estimator",
    :gcomp => "Iterated-regression g-computation",
    :ipw => "Inverse probability of treatment weighting")

StatsAPI.coef(r::LongitudinalEstimate) = r.coef
StatsAPI.coefnames(r::LongitudinalEstimate) = r.names
StatsAPI.nobs(r::LongitudinalEstimate) = r.n
StatsAPI.dof_residual(r::LongitudinalEstimate) =
    r.cluster === nothing ? Inf : r.n_clusters - 1.0
estimand(::LongitudinalEstimate) = "mean counterfactual outcome E[Y^d] under each regime"
method_name(r::LongitudinalEstimate) = _LG_METHOD_NAMES[r.estimator]

function StatsAPI.vcov(r::LongitudinalEstimate)
    r.vcov === nothing && throw(ArgumentError(
        "the g-computation estimator has no influence-function variance; refit with " *
        "`bootstrap = B` for bootstrap standard errors, or use estimator = :tmle or :sdr"))
    return r.vcov
end

# Without a variance (g-computation, no bootstrap) the coefficient table cannot be
# formed; print the estimates alone.
function Base.show(io::IO, ::MIME"text/plain", r::LongitudinalEstimate)
    if r.vcov !== nothing
        return invoke(Base.show, Tuple{IO,MIME"text/plain",CausalEstimate}, io,
                      MIME"text/plain"(), r)
    end
    println(io, method_name(r), " — estimand: ", estimand(r))
    println(io, "Observations: ", r.n)
    show(io, MIME"text/plain"(),
         StatsBase.CoefTable(reshape(r.coef, :, 1), ["Estimate"], r.names, 0, 0))
    show_details(io, r)
    return nothing
end

function show_details(io::IO, r::LongitudinalEstimate)
    println(io)
    println(io, "Units: ", r.n, "; periods: ", r.K)
    for (j, nm) in enumerate(r.names)
        f = r.followers[j]
        print(io, "  ", nm, ": ")
        @printf(io, "%d unit(s) (%.1f%%) follow the regime through the last period\n",
                count(f[:, end]), 100 * mean(f[:, end]))
    end
    nf = maximum(r.folds)
    R = size(r.folds, 2)
    if nf > 1
        print(io, "Cross-fitting: $nf folds × $R repetition", R == 1 ? "" : "s")
        R > 1 && print(io, " (median aggregation)")
        println(io)
    else
        println(io, "No cross-fitting (nuisances fitted in sample)")
    end
    r.cluster === nothing ||
        println(io, "Clusters: $(r.n_clusters) (folds grouped by cluster; " *
                    "t($(r.n_clusters - 1)))")
    for (nm, l) in r.learners
        r.estimator === :ipw && nm === :outcome && continue
        println(io, "  ", nm, " learner: ", l)
    end
    @printf(io, "Treatment mechanism out-of-fold log loss by period: %s\n",
            join([@sprintf("%.3g", v) for v in r.treatment_logloss], ", "))
    nb = sum(r.n_bounded)
    @printf(io, "Cumulative treatment probability bounded below at %.3g: ", r.g_bound)
    println(io, nb, " follower period(s) affected")
    if size(r.bootstrap_coef, 1) > 0
        println(io, "Standard errors: unit bootstrap with ", size(r.bootstrap_coef, 1),
                " replicates")
    elseif r.estimator === :gcomp
        println(io, "Standard errors: none (refit with `bootstrap = B`)")
    else
        println(io, "Standard errors: estimated efficient influence function")
    end
    return nothing
end
