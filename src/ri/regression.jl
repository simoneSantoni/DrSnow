# Randomization inference for regression coefficients (Young 2019): re-randomize
# the treatment according to the design, refit the regression with
# FixedEffectModels, and compare the observed coefficient (randomization-c) and its
# t-statistic (randomization-t) with their randomization distributions.

# Permutation of treatment labels across assignment units within strata; used when
# the treatment is not a single binary column (multiple arms, doses).
struct _ri_LabelPermutation
    units::Vector{Vector{Int}}      # rows of each assignment unit
    strata::Vector{Vector{Int}}     # assignment units in each stratum
    n::Int
end

# Row map: the treatment values of row i are taken from row src[i].
function _ri_draw_rowmap(rng::AbstractRNG, lp::_ri_LabelPermutation)
    src = collect(1:lp.n)
    for s in lp.strata
        perm = s[randperm(rng, length(s))]
        for (u, v) in zip(s, perm)
            r = lp.units[v][1]
            for i in lp.units[u]
                src[i] = r
            end
        end
    end
    return src
end

"""
    RIRegressionResult

Result of [`ri_regression`](@ref): conventional and randomization p-values for the
treatment coefficients of one or several regressions, with joint and
multiplicity-adjusted tests.

# Fields
- `table::DataFrame`: one row per (outcome, treatment coefficient) with columns
  `outcome`, `treatment`, `estimate`, `std_error`, `t`, `p_conventional` (from the
  regression's own covariance and reference distribution), `p_randomization_c`
  (randomization distribution of the coefficient), `p_randomization_t`
  (randomization distribution of the t-statistic) and `p_westfall_young`
  (Westfall–Young maxT step-down over all rows, based on the t-statistics).
- `joint::Vector{DiagnosticTest}`: for each outcome, a joint randomization test of
  all treatment coefficients based on their Wald statistic (only when there are
  several treatments).
- `omnibus::Union{Nothing,DiagnosticTest}`: joint randomization test across all
  outcomes and treatments, based on the sum of the per-outcome Wald statistics (only
  when there are several outcomes).
- `exact::Bool`, `n_draws::Int`, `n_dropped::Int`, `mechanism::String`: reference
  set and design; `n_dropped` counts assignments on which a coefficient was not
  identified.
- `vcov_type::String`: covariance estimator used for the t-statistics.
- `nobs::Int`: number of observations.
- `distribution::Matrix{Float64}`, `weights::Vector{Float64}`: reference set; the
  columns hold all coefficients, then all t-statistics, then the per-outcome Wald
  statistics.

# Accessors
- `coef`, `coefnames`, `nobs`, [`pvalues`](@ref) (randomization-t p-values),
  [`randomization_distribution`](@ref).
"""
struct RIRegressionResult
    table::DataFrame
    joint::Vector{DiagnosticTest}
    omnibus::Union{Nothing,DiagnosticTest}
    exact::Bool
    n_draws::Int
    n_dropped::Int
    mechanism::String
    vcov_type::String
    nobs::Int
    distribution::Matrix{Float64}
    weights::Vector{Float64}
end

StatsAPI.coef(r::RIRegressionResult) = Vector{Float64}(r.table.estimate)
StatsAPI.coefnames(r::RIRegressionResult) =
    ["$(o): $(t)" for (o, t) in zip(r.table.outcome, r.table.treatment)]
StatsAPI.nobs(r::RIRegressionResult) = r.nobs
pvalues(r::RIRegressionResult) = Vector{Float64}(r.table.p_randomization_t)
randomization_distribution(r::RIRegressionResult) = (r.distribution, r.weights)

function Base.show(io::IO, ::MIME"text/plain", r::RIRegressionResult)
    println(io, "Randomization inference for regression coefficients")
    println(io, "Assignment mechanism: ", r.mechanism)
    println(io, "Reference distribution: ", r.exact ? "exact, " : "Monte Carlo, ",
            r.n_draws, r.exact ? " assignments" : " draws")
    println(io, "Covariance for t-statistics: ", r.vcov_type, "; observations: ", r.nobs)
    r.n_dropped > 0 && println(io, "Assignments with a non-identified coefficient " *
                                   "(excluded): ", r.n_dropped)
    t = r.table
    wo = max(7, maximum(length ∘ string, t.outcome))
    wt = max(9, maximum(length ∘ string, t.treatment))
    @printf(io, "%-*s %-*s %11s %10s %8s %8s %8s %8s\n", wo, "Outcome", wt,
            "Treatment", "Estimate", "Std.Err.", "p conv", "p RI-c", "p RI-t", "p WY")
    for i in 1:nrow(t)
        @printf(io, "%-*s %-*s %11.5g %10.4g %8.4f %8.4f %8.4f %8.4f\n", wo,
                string(t.outcome[i]), wt, string(t.treatment[i]), t.estimate[i],
                t.std_error[i], t.p_conventional[i], t.p_randomization_c[i],
                t.p_randomization_t[i], t.p_westfall_young[i])
    end
    for j in r.joint
        @printf(io, "Joint (%s): Wald = %.4g, randomization p = %.4g\n", j.null,
                j.statistic, j.pvalue)
    end
    r.omnibus === nothing ||
        @printf(io, "Omnibus (all outcomes and treatments): stat = %.4g, randomization p = %.4g",
                r.omnibus.statistic, r.omnibus.pvalue)
end

Base.show(io::IO, r::RIRegressionResult) =
    print(io, "RIRegressionResult($(nrow(r.table)) coefficients)")

"""
    ri_regression(data, outcome, treatment; covariates=Symbol[], fe=Symbol[],
                  strata=nothing, cluster=nothing, vcov=nothing, weights=nothing,
                  mechanism=nothing, id=nothing, nperm=1_000, exact=:auto,
                  alternative=:two_sided, rng=Random.default_rng(),
                  threaded=Threads.nthreads() > 1) -> RIRegressionResult

Randomization inference for the treatment coefficients of a regression, following
Young (2019): the treatment is re-randomized according to the design, the
regression is refitted on every draw, and the observed coefficients and
t-statistics are compared with their randomization distributions.

Applied work typically analyses experiments with regressions of the form
`outcome ~ treatment + covariates + fe(...)` and conventional (robust or clustered)
standard errors, whose finite-sample size can be poor with few clusters, leverage
concentrated on a few observations or many regressors. Young (2019) shows, for a
large sample of published experiments, that randomization tests often lead to fewer
significant results. Two randomization p-values are reported per coefficient:
**randomization-c**, based on the distribution of the coefficient itself, and
**randomization-t**, based on the distribution of its t-statistic computed with the
same covariance estimator as the conventional test. Both are exact tests of the
sharp null that no treatment affects the outcome of any unit. The randomization-t
p-value, which Young recommends, is in addition asymptotically valid for the
corresponding weak null in the settings where the studentized statistic is
asymptotically pivotal (Chung and Romano 2013; Wu and Ding 2021); this is an
asymptotic, not an exact, property. Joint tests of several treatment coefficients
use the Wald statistic with their full covariance matrix, and a Westfall–Young
maxT adjustment across all outcomes and coefficients is reported.

Re-randomization follows the design. A single 0/1 treatment column is redrawn from
the assignment mechanism (`mechanism`, or the design built from `strata` /
`cluster` as in [`randomization_test`](@ref), conditional on the observed treated
counts), and exact enumeration is available. Several treatment columns (arm
dummies) or a non-binary treatment are permuted jointly as labels of assignment
units (clusters when `cluster` is given, rows otherwise) within `strata`, which
reproduces complete, stratified and cluster randomization with fixed arm sizes
(Monte Carlo only). With several arms the null tested by each coefficient's
randomization p-value is the joint sharp null that no arm has any effect, because
all arms are permuted together; a test of one arm while others may have effects is
not a sharp null and is not exact.

# Arguments
- `data`: a `DataFrame` with one row per observation; with `cluster`, the
  treatment must be constant within clusters. Missing values are not allowed in the
  columns used.
- `outcome`: a `Symbol` or a vector of `Symbol`s; one regression is fitted per
  outcome.
- `treatment`: a `Symbol` or a vector of `Symbol`s (the treatment regressors).

# Keywords
- `covariates::Vector{Symbol}`: control variables.
- `fe::Vector{Symbol}`: absorbed fixed effects, e.g. `fe=[:strata]` for stratified
  designs.
- `strata`, `cluster::Union{Nothing,Symbol}`: randomization strata and clusters.
  `cluster` also sets the default covariance, `Vcov.cluster(cluster)` (otherwise
  `Vcov.robust()`).
- `vcov::Union{Nothing,FixedEffectModels.CovarianceEstimator}`: covariance estimator
  for the t-statistics and the conventional p-values; takes precedence over the
  default.
- `weights::Union{Nothing,Symbol}`: regression weights.
- `mechanism`, `id`: as in [`randomization_test`](@ref) (single binary treatment
  only).
- `nperm::Integer`: number of Monte Carlo draws (and enumeration threshold);
  default 1 000. Every draw refits the regressions, so the cost is `nperm` times
  that of the fit.
- `exact`, `rng`, `threaded`: as in [`randomization_test`](@ref).
- `alternative::Symbol`: for the per-coefficient p-values; the joint and omnibus
  tests are upper-tailed.

# Returns
- [`RIRegressionResult`](@ref); `r.table` holds the per-coefficient results.

# Examples
```julia
using DrSnow, DataFrames, StableRNGs
rng = StableRNG(5)
df = DataFrame(school=repeat(1:12; inner=8), district=repeat(1:3; inner=32))
df.d = Int.(in.(df.school, Ref([1, 2, 5, 6, 9, 10])))
df.x = randn(rng, 96)
df.y1 = 0.4 .* df.d .+ df.x .+ randn(rng, 96)
df.y2 = randn(rng, 96)
r = ri_regression(df, [:y1, :y2], :d; covariates=[:x], cluster=:school,
                  strata=:district, fe=[:district], nperm=500, rng=rng)
r.table
```

# References
- Young, A. (2019). Channeling Fisher: Randomization tests and the statistical
  insignificance of seemingly significant experimental results. *Quarterly Journal
  of Economics*, 134(2), 557–598.
- Chung, E., & Romano, J. P. (2013). Exact and asymptotically robust permutation
  tests. *Annals of Statistics*, 41(2), 484–507.
- Wu, J., & Ding, P. (2021). Randomization tests for weak null hypotheses in
  randomized experiments. *Journal of the American Statistical Association*,
  116(536), 1898–1913.
- Imbens, G. W., & Rubin, D. B. (2015). *Causal Inference for Statistics, Social,
  and Biomedical Sciences: An Introduction*, ch. 5. Cambridge University Press.
- Westfall, P. H., & Young, S. S. (1993). *Resampling-Based Multiple Testing:
  Examples and Methods for p-Value Adjustment*. Wiley.
"""
function ri_regression(data, outcome::Union{Symbol,Vector{Symbol}},
                       treatment::Union{Symbol,Vector{Symbol}};
                       covariates::Vector{Symbol}=Symbol[], fe::Vector{Symbol}=Symbol[],
                       strata::Union{Nothing,Symbol}=nothing,
                       cluster::Union{Nothing,Symbol}=nothing,
                       vcov::Union{Nothing,FixedEffectModels.CovarianceEstimator}=nothing,
                       weights::Union{Nothing,Symbol}=nothing, mechanism=nothing,
                       id::Union{Nothing,Symbol}=nothing, nperm::Integer=1_000,
                       exact=:auto, alternative::Symbol=:two_sided,
                       rng::AbstractRNG=Random.default_rng(),
                       threaded::Bool=Threads.nthreads() > 1)
    ctxname = "ri_regression"
    _ri_check_alternative(alternative)
    ys = _as_symbols(outcome)
    ds = _as_symbols(treatment)
    (isempty(ys) || isempty(ds)) &&
        throw(ArgumentError("$ctxname: need at least one outcome and one treatment"))
    (allunique(ys) && allunique(ds)) ||
        throw(ArgumentError("$ctxname: duplicate outcome or treatment names"))
    isempty(intersect(ds, vcat(ys, covariates, fe))) ||
        throw(ArgumentError("$ctxname: treatment columns must not appear among the " *
                            "outcomes, covariates or fixed effects"))
    used = vcat(ys, ds, covariates, fe)
    weights === nothing || push!(used, weights)
    require_columns(data, vcat(used, [strata, cluster, id]); context=ctxname)
    _ri_check_complete(data, vcat(used, [strata, cluster, id]), ctxname)
    vc = vcov !== nothing ? vcov :
         cluster !== nothing ? Vcov.cluster(cluster) : Vcov.robust()
    vc_cols = _ri_vcov_columns(vc)
    single_binary = length(ds) == 1 && all(x -> x == 0 || x == 1, data[!, ds[1]])

    sortcols = unique(vcat(ys, ds, covariates, fe, vc_cols,
                           weights === nothing ? Symbol[] : [weights]))
    if single_binary
        design = _ri_design(data, ds[1]; mechanism, strata, cluster, id,
                            sortcols=setdiff(sortcols, [ds[1]]), context=ctxname)
        rows = design.rows
        plan = _ri_plan(design.mech, nperm, exact, rng)
        mdesc = _ri_describe(design.mech)
    else
        mechanism === nothing ||
            throw(ArgumentError("$ctxname: an explicit `mechanism` requires a single " *
                                "0/1 treatment; use `strata`/`cluster` for multiple " *
                                "or non-binary treatments"))
        exact === true && throw(ArgumentError("$ctxname: exact enumeration is only " *
                                              "available for a single 0/1 treatment"))
        rows, lp = _ri_label_permutation(data, ds, strata, cluster, id, sortcols,
                                         ctxname)
        nperm >= 1 || throw(ArgumentError("nperm must be positive"))
        plan = _ri_Plan(false, BitVector[], Float64[],
                        task_seeds(rng, cld(nperm, _RI_CHUNK)), Int(nperm))
        mdesc = "permutation of treatment labels across " *
                (cluster === nothing ? "units" : "clusters") *
                (strata === nothing ? "" : " within strata") *
                " ($(length(lp.units)) assignment units)"
    end

    local_cols = unique(vcat(used, vc_cols))
    base = DataFrame([c => data[rows, c] for c in local_cols])
    T0 = Matrix{Float64}(reduce(hcat, [Float64.(base[!, d]) for d in ds]))
    J = length(ds)
    K = length(ys)
    formulas = [make_formula(y, vcat(ds, covariates); fe=fe) for y in ys]
    fitkw = (weights=weights, progress_bar=false)

    # statistic vector: K*J coefficients, K*J t-statistics, K Wald statistics
    function stats_for(Tm::Matrix{Float64}, strict::Bool)
        df = copy(base; copycols=false)
        for (j, d) in enumerate(ds)
            df[!, d] = Tm[:, j]
        end
        out = fill(NaN, 2 * K * J + K)
        for k in 1:K
            m = FixedEffectModels.reg(df, formulas[k], vc; fitkw...)
            idx = strict ? [coef_index(m, d) for d in ds] : _ri_coef_indices(m, ds)
            idx === nothing && continue
            b = StatsAPI.coef(m)[idx]
            V = StatsAPI.vcov(m)[idx, idx]
            se = sqrt.(max.(diag(V), 0.0))
            for j in 1:J
                out[(k - 1) * J + j] = b[j]
                out[K * J + (k - 1) * J + j] = se[j] > 0 ? b[j] / se[j] : NaN
            end
            out[2 * K * J + k] = wald_test(b, V).chi2
        end
        return out
    end

    L = 2 * K * J + K
    obs = stats_for(T0, true)      # fails early if a coefficient is not identified
    if single_binary
        tofloat = z -> reshape(Float64.(z), :, 1)
        R = _ri_map(z -> stats_for(tofloat(z), false), plan, design.mech, design.z, L;
                    threaded)
    else
        R = _ri_map_mc(src -> stats_for(T0[src, :], false),
                       r -> _ri_draw_rowmap(r, lp), collect(1:length(rows)), plan.seeds,
                       plan.B, L; threaded)
    end
    w = _ri_weights(plan)
    keep = [all(!isnan, view(R, i, :)) for i in axes(R, 1)]
    dropped = count(!, keep)
    Rk = R[keep, :]
    wk = w[keep]

    # conventional results from the observed fits
    conv = [FixedEffectModels.reg(copy(base; copycols=false), formulas[k], vc; fitkw...)
            for k in 1:K]
    table = DataFrame(outcome=Symbol[], treatment=Symbol[], estimate=Float64[],
                      std_error=Float64[], t=Float64[], p_conventional=Float64[],
                      p_randomization_c=Float64[], p_randomization_t=Float64[],
                      p_westfall_young=Float64[])
    tcols = [K * J + i for i in 1:(K * J)]
    wy = _ri_westfall_young(obs[tcols], Rk[:, tcols], wk, alternative, :maxt)
    for k in 1:K, j in 1:J
        i = (k - 1) * J + j
        m = conv[k]
        ci = coef_index(m, ds[j])
        ct = StatsAPI.coeftable(m)       # rows looked up by name (order may differ)
        pconv = ct.cols[ct.pvalcol][findfirst(==(string(ds[j])), ct.rownms)]
        pc, _, _ = _ri_pvalue(obs[i], Rk[:, i], wk, alternative)
        pt, _, _ = _ri_pvalue(obs[K * J + i], Rk[:, K * J + i], wk, alternative)
        push!(table, (ys[k], ds[j], obs[i], sqrt(StatsAPI.vcov(m)[ci, ci]),
                      obs[K * J + i], pconv, pc, pt, wy.adjusted[j + (k - 1) * J]))
    end
    method = (plan.exact ? "randomization-t, exact enumeration of $(plan.B) assignments" :
              "randomization-t, Monte Carlo with $(plan.B) draws")
    joint = DiagnosticTest[]
    if J > 1
        for k in 1:K
            c = 2 * K * J + k
            p, _, _ = _ri_pvalue(obs[c], Rk[:, c], wk, :greater)
            push!(joint, DiagnosticTest("Joint randomization test of treatment " *
                                        "coefficients",
                                        "no effect of $(join(ds, ", ")) on $(ys[k]) " *
                                        "for any unit", obs[c], p; dof=(J,),
                                        method=method * "; Wald statistic",
                                        note="Wald statistic uses the full covariance " *
                                             "of the treatment coefficients."))
        end
    end
    omnibus = nothing
    if K > 1
        wcols = [2 * K * J + k for k in 1:K]
        so = sum(obs[wcols])
        sd = vec(sum(Rk[:, wcols]; dims=2))
        p, _, _ = _ri_pvalue(so, sd, wk, :greater)
        omnibus = DiagnosticTest("Omnibus randomization test",
                                 "no effect of any treatment on any outcome for any unit",
                                 so, p; method=method *
                                 "; sum over outcomes of Wald statistics",
                                 note="Cross-outcome covariances are not used in the " *
                                      "statistic; the randomization p-value is " *
                                      "nevertheless exact under the sharp null.")
    end
    return RIRegressionResult(table, joint, omnibus, plan.exact, plan.B, dropped, mdesc,
                              _ri_vcov_name(vc), length(rows), R, w)
end

_ri_vcov_columns(v) = hasproperty(v, :clusternames) ?
    Symbol[Symbol(c) for c in v.clusternames] : Symbol[]
_ri_vcov_name(v) = string(nameof(typeof(v)))

# Coefficient indices by name, or `nothing` if any is missing or not identified.
function _ri_coef_indices(m, ds)
    names = StatsAPI.coefnames(m)
    V = StatsAPI.vcov(m)
    idx = Int[]
    for d in ds
        i = findfirst(==(string(d)), names)
        i === nothing && return nothing
        v = V[i, i]
        (!isfinite(v) || v <= 0) && return nothing
        push!(idx, i)
    end
    return idx
end

function _ri_label_permutation(data, ds, strata, cluster, id, sortcols, ctxname)
    n = nrow(data)
    rows = if id !== nothing
        allunique(data[!, id]) || throw(ArgumentError("$ctxname: id has duplicates"))
        sortperm(data[!, id]; by=_ri_sortkey)
    else
        canon = Symbol[c for c in (cluster, strata) if c !== nothing]
        _ri_canonical_rows(data, unique(vcat(canon, sortcols)))
    end
    units = if cluster === nothing
        [[i] for i in 1:n]
    else
        [g for (_, g) in _ri_groups(data[rows, cluster])]
    end
    for g in units, d in ds
        v = data[rows[g], d]
        all(==(v[1]), v) ||
            throw(ArgumentError("$ctxname: treatment $d varies within a cluster"))
    end
    ustrata = if strata === nothing
        [collect(eachindex(units))]
    else
        lab = [data[rows[g[1]], strata] for g in units]
        for g in units
            all(==(data[rows[g[1]], strata]), data[rows[g], strata]) ||
                throw(ArgumentError("$ctxname: a cluster spans several strata"))
        end
        [gr for (_, gr) in _ri_groups(lab)]
    end
    return rows, _ri_LabelPermutation(units, ustrata, n)
end
