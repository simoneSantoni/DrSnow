# Learned measures inside natural experiments: difference-in-differences and sharp
# regression discontinuity with an ML-measured outcome, corrected for prediction
# error with a gold-labelled subsample.
#
# Design-based mode. With known labelling probabilities, the DSL pseudo-outcome
# Ỹ = ĝ + (R/π)(Y − ĝ) has the conditional mean of the true outcome in every
# treatment-group × period cell (or at every value of the running variable). The
# DiD and local-polynomial RD point estimators are linear in the outcome given the
# design, so applying them to Ỹ estimates the estimand defined with the true
# outcome under the usual identifying assumptions, and their cluster-robust / RD
# variance estimators applied to Ỹ include the labelling noise. No assumption on
# the structure of the prediction error is needed, but every cell must have a
# positive labelling probability.
#
# Stable-error mode. When labels exist only in some periods (or on one side of the
# cutoff), the error cannot be learned where it matters without an assumption: the
# measurement model E[prediction | Y, group, features] = a_g + b·Y + γ'features is
# assumed to be the same in all periods (on both sides of the cutoff). It is fitted
# on the labelled rows and inverted, Ŷ = (f − a_g − γ'features)/b, and a cluster
# bootstrap that refits it gives the variance. (The reverse regression E[Y | f] is
# not used: it changes whenever the distribution of Y changes, e.g. with treatment.)

"""
Stable-error correction: fit the linear measurement model `f = a + b Y + γ'W + u`
(`W` = `extra` columns and `features`) on the labelled rows and return
`(f − a − γ'W) / b` for every row.
"""
function _ml_meas_invert(d, outcome, pred, features, labeled, extra; context)
    Rb, _ = _ml_meas_labels(d, [outcome], labeled, nothing, false; context=context)
    yb = _ml_meas_gold(d, outcome, Rb; context=context)
    fb = _ml_column(d, pred; context=context)
    W = hcat(ones(nrow(d)), extra, _ml_matrix(d, features; context=context))
    Zl = hcat(yb, W)[Rb, :]
    rank(Zl) == size(Zl, 2) ||
        throw(ArgumentError("$(context): the measurement model is not identified on " *
                            "the labelled rows (collinear inputs)"))
    c = Zl \ fb[Rb]
    c[1] > 0 || throw(ArgumentError("$(context): among labelled rows the prediction " *
                                    "does not increase with the gold-standard outcome " *
                                    "(slope $(round(c[1]; sigdigits=3))); the " *
                                    "stable-error correction is not defined"))
    return (fb .- W * c[2:end]) ./ c[1]
end

"""
Coefficients on the scale used for inference. RD results report the conventional
estimate as `coef` but centre inference on the bias-corrected estimate, so the
bias-corrected estimate is used here.
"""
_ml_meas_infcoef(r) = r isa RDEstimate ? [r.tau_bias_corrected] :
                      Vector{Float64}(StatsAPI.coef(r))

"""Headline (estimate, standard error) of a design estimate, on the inference scale."""
function _ml_meas_headline(r)
    r isa RDEstimate && return r.tau_bias_corrected, r.se_robust
    if r isa CallawaySantAnnaEstimate
        a = aggregate_att(r, :simple; bootstrap=false)
        return StatsAPI.coef(a)[1], StatsAPI.stderror(a)[1]
    end
    return StatsAPI.coef(r)[1], StatsAPI.stderror(r)[1]
end

function _ml_meas_bias_test(diff_est, diff_se, method)
    z = diff_est / diff_se
    return DiagnosticTest("Naive-vs-corrected contrast (differential prediction error)",
                          "the design estimator applied to the prediction error " *
                          "(true minus predicted outcome) has mean zero, i.e. the " *
                          "naive estimate with the ML measurement is unbiased",
                          z, two_sided_pvalue(z); method=method,
                          note="Non-rejection is not evidence that the naive estimate " *
                               "is unbiased: the test has low power when few units " *
                               "are labelled. The corrected estimate does not rely " *
                               "on this test.",
                          details=(difference=diff_est, se=diff_se))
end

"""
Drop all rows of the groups (units / clusters) that contain a row flagged in
`training`. Returns (data, number of groups dropped).
"""
function _ml_meas_drop_training(df, training, groupcol; context)
    training === nothing && return df, 0
    t = df[!, training]
    all(x -> x == 0 || x == 1, t) ||
        throw(ArgumentError("$(context): measure_training column must be 0/1 or Bool"))
    any(==(1), t) || return df, 0
    if groupcol === nothing
        keep = .!(t .== 1)
        return df[keep, :], count(!, keep)
    end
    bad = Set(df[t .== 1, groupcol])
    keep = [!(g in bad) for g in df[!, groupcol]]
    any(keep) || throw(ArgumentError("$(context): every unit was used to train the " *
                                     "measure; nothing is left for the causal estimate"))
    return df[keep, :], length(bad)
end

"""Dummy columns (dropping the first level) of the values `v`."""
function _ml_meas_dummies(v)
    u = unique(v)
    try
        sort!(u)
    catch
    end
    length(u) <= 1 && return zeros(length(v), 0)
    return reduce(hcat, [Float64.(isequal.(v, x)) for x in u[2:end]])
end

"""
Cluster bootstrap for the stable-error mode: resample groups with replacement,
refit the calibration model on the labelled rows and re-run `fit` with the
calibrated outcome (and the naive one). Returns (draws of corrected coef, draws of
corrected − naive headline).
"""
function _ml_meas_stable_boot(df, groupcol, unitcol, fitfun, calib, naive_col, B, rng;
                              context)
    gv = groupcol === nothing ? collect(1:nrow(df)) : df[!, groupcol]
    u = unique(gv)
    rows_of = Dict{eltype(u),Vector{Int}}()
    for (i, g) in enumerate(gv)
        push!(get!(rows_of, g, Int[]), i)
    end
    G = length(u)
    draws = Vector{Vector{Float64}}()
    diffs = Float64[]
    failures = 0
    for b in 1:B
        pick = rand(rng, 1:G, G)
        idx = Int[]
        newid = Int[]
        for (j, gi) in enumerate(pick)
            rr = rows_of[u[gi]]
            append!(idx, rr)
            append!(newid, fill(j, length(rr)))
        end
        bdf = df[idx, :]
        # relabel resampled groups (and units within them) so duplicates are distinct
        if groupcol !== nothing
            bdf[!, groupcol] = newid
        end
        if unitcol !== nothing && unitcol != groupcol
            ukey = Dict{Tuple{Int,Any},Int}()
            bdf[!, unitcol] = [get!(ukey, (newid[i], bdf[i, unitcol]), length(ukey) + 1)
                               for i in 1:nrow(bdf)]
        end
        try
            bdf[!, :__ml_meas_y] = calib(bdf)
            rc = fitfun(bdf, :__ml_meas_y)
            rn = fitfun(bdf, naive_col)
            push!(draws, _ml_meas_infcoef(rc))
            push!(diffs, _ml_meas_headline(rc)[1] - _ml_meas_headline(rn)[1])
        catch err
            err isa InterruptException && rethrow()
            failures += 1
        end
    end
    length(draws) >= max(20, B ÷ 2) ||
        throw(ArgumentError("$(context): the bootstrap failed in $(failures) of $(B) " *
                            "replications (too few labelled groups?)"))
    return reduce(hcat, draws)', diffs, failures
end

"""
Shared driver. `cells` labels each row's design cell (for label checks);
`design_extra` / `stable_extra` are extra inputs of ĝ in the two modes; `fitfun(df,
ycol)` runs the design estimator; `fit_diff` re-runs it with the corrected fit's
tuning (e.g. RD bandwidths) for the naive and contrast fits.
"""
function _ml_meas_natural(df, outcome, preds, features, labeled, label_prob,
                          normalize_prob, learner, foldgroup, bootgroup, unitcol, cells,
                          design_extra, stable_extra, fitfun, fit_same, stable, n_folds,
                          folds, B, rng, parallel, ntrain; context)
    R, π = _ml_meas_labels(df, [outcome], labeled, label_prob, normalize_prob;
                           context=context)
    y = _ml_meas_gold(df, outcome, R; context=context)
    n = nrow(df)
    for c in (:__ml_meas_y, :__ml_meas_naive, :__ml_meas_diff)
        c in propertynames(df) &&
            throw(ArgumentError("$(context): reserved column name $(c) is in use"))
    end
    ucells = unique(cells)
    nolab = [c for c in ucells if !any(R[i] for i in 1:n if isequal(cells[i], c))]
    if !stable && label_prob === nothing && !isempty(nolab)
        throw(ArgumentError("$(context): no gold-labelled rows in cell(s) " *
                            "$(join(string.(nolab), ", ")). The labels were evidently " *
                            "not a simple random sample of rows: pass the labelling " *
                            "probabilities (`label_prob`, all positive), label rows in " *
                            "every cell, or set `assume_stable_error = true` to assume " *
                            "the calibration of the measure is the same in every cell."))
    end
    (learner === nothing && length(preds) != 1) &&
        throw(ArgumentError("$(context): learner = nothing needs one prediction column"))
    if !stable
        groups = foldgroup === nothing ? nothing : df[!, foldgroup]
        F = _ml_meas_folds(df, folds, n, n_folds, 1, rng, R, groups; context=context)
        g = learner === nothing ? _ml_column(df, preds[1]; context=context) :
            _ml_meas_crossfit([y], hcat(_ml_matrix(df, vcat(preds, features);
                                                   context=context), design_extra),
                              BitVector(R), learner, F, rng; parallel=parallel,
                              context=context)[:, 1, 1]
        naive = isempty(preds) ? g : _ml_column(df, preds[1]; context=context)
        df[!, :__ml_meas_y] = _ml_meas_pseudo(g, y, R, π)
        df[!, :__ml_meas_naive] = naive
        df[!, :__ml_meas_diff] = df.__ml_meas_y .- naive
        rc = fitfun(df, :__ml_meas_y)
        rn = fit_same(rc, df, :__ml_meas_naive)
        rd = fit_same(rc, df, :__ml_meas_diff)
        de, dse = _ml_meas_headline(rd)
        test = _ml_meas_bias_test(de, dse, "design estimator applied to the " *
                                  "pseudo-outcome minus the prediction")
        return MeasuredOutcomeEstimate(rc, rn, String.(StatsAPI.coefnames(rc)),
                                       Vector{Float64}(StatsAPI.coef(rc)),
                                       Matrix{Float64}(StatsAPI.vcov(rc)), :design, test,
                                       n, count(R), ntrain,
                                       (pseudo=df.__ml_meas_y, fitted=g, labeled=R,
                                        prob=π, folds=F[:, 1], contrast=rd,
                                        unlabeled_cells=nolab))
    end
    # stable-error mode: linear measurement model f = a + b·Y + γ'W + u, estimated on
    # the labelled rows and assumed identical in every cell; Ŷ = (f − a − γ'W) / b
    isempty(preds) && throw(ArgumentError("$(context): assume_stable_error needs a " *
                                          "`prediction` column"))
    calib = d -> _ml_meas_invert(d, outcome, preds[1], features, labeled,
                                 stable_extra[d.__ml_meas_row, :]; context=context)
    df[!, :__ml_meas_row] = collect(1:n)
    df[!, :__ml_meas_y] = calib(df)
    df[!, :__ml_meas_naive] = _ml_column(df, preds[1]; context=context)
    rc = fitfun(df, :__ml_meas_y)
    rn = fit_same(rc, df, :__ml_meas_naive)
    fitb = (d, c) -> fit_same(rc, d, c)
    draws, diffs, fails = _ml_meas_stable_boot(df, bootgroup, unitcol, fitb, calib,
                                               :__ml_meas_naive, B, rng; context=context)
    b = _ml_meas_infcoef(rc)
    V = Matrix(Symmetric(cov(Matrix(draws); corrected=true)))
    V = reshape(V, length(b), length(b))
    de = _ml_meas_headline(rc)[1] - _ml_meas_headline(rn)[1]
    test = _ml_meas_bias_test(de, std(diffs), "cluster bootstrap ($(length(diffs)) " *
                              "replications) of the corrected minus naive estimate")
    return MeasuredOutcomeEstimate(rc, rn, String.(StatsAPI.coefnames(rc)), b, V,
                                   :stable_error, test, n, count(R), ntrain,
                                   (calibrated=df.__ml_meas_y, labeled=R,
                                    bootstrap_draws=Matrix(draws),
                                    bootstrap_failures=fails, unlabeled_cells=nolab))
end

const _ML_MEAS_DESIGN_REFS = """
- Egami, N., Hinck, M., Stewart, B. M., & Wei, H. (2023). Using imperfect
  surrogates for downstream inference: Design-based supervised learning for
  social science applications of large language models. *Advances in Neural
  Information Processing Systems*, 36, 68589–68601.
- Angelopoulos, A. N., Bates, S., Fannjiang, C., Jordan, M. I., & Zrnic, T.
  (2023). Prediction-powered inference. *Science*, 382(6671), 669–674.
- Zrnic, T., & Candès, E. J. (2024). Cross-prediction-powered inference.
  *Proceedings of the National Academy of Sciences*, 121(15), e2322083121.
- Robins, J. M., Rotnitzky, A., & Zhao, L. P. (1994). Estimation of regression
  coefficients when some regressors are not always observed. *Journal of the
  American Statistical Association*, 89(427), 846–866.
- Carroll, R. J., Ruppert, D., Stefanski, L. A., & Crainiceanu, C. M. (2006).
  *Measurement Error in Nonlinear Models: A Modern Perspective* (2nd ed.).
  Chapman & Hall/CRC."""

"""
    did_with_predicted_outcome(data, outcome, treatment, unit, time;
                               prediction=nothing, estimator=:twfe,
                               features=Symbol[], learner=OLSLearner(),
                               labeled=nothing, label_prob=nothing,
                               normalize_prob=true, measure_training=nothing,
                               assume_stable_error=false, cluster=nothing, n_folds=5,
                               folds=nothing, bootstrap_reps=199,
                               rng=Random.default_rng(), parallel=false, kwargs...)
        -> MeasuredOutcomeEstimate

Difference-in-differences when the outcome is measured by an ML model or LLM
(for example the tone of documents coded by an LLM) and a subsample of
unit-periods carries expert (gold-standard) labels.

The estimand is the DiD estimand of the chosen estimator (the two-way
fixed-effects coefficient, the group-time ATTs of Callaway and Sant'Anna (2021),
or the two-period ATT of Sant'Anna and Zhao (2020)) defined with the *true*
outcome, and identification requires parallel trends of the true outcome. The
naive estimate that uses the ML measurement as outcome is biased whenever the
mean prediction error evolves differently over time in the treated and
comparison groups, for instance when treatment changes the language the model
reacts to, or when a classifier's error depends on the true outcome that
treatment moves. A group-specific error that is constant over time is
differenced out.

**Design-based mode** (default). Labels are sampled with probabilities that are
known by design (`label_prob`, or a simple random sample of rows) and bounded
away from zero in every group × period cell, and the gold labels are assumed
error-free. The DiD estimator is applied to the DSL pseudo-outcome
``\\tilde Y = \\hat g + (R/\\pi)(Y - \\hat g)`` of [`dsl_pseudo_outcome`](@ref),
with ``\\hat g`` a prediction of the gold standard from the measurement,
`features`, cohort and period indicators and the prediction × treatment
interaction, cross-fitted by unit (by `cluster` when given), so that the model
predicting a unit's rows never saw that unit's labels. Because
``E[\\tilde Y_{it} \\mid \\text{cohort}, \\text{period}] = E[Y_{it} \\mid
\\text{cohort}, \\text{period}]`` and the DiD estimators are linear in the
outcome given the design, the estimate targets the true-outcome estimand, and
the cluster-robust standard errors computed on ``\\tilde Y`` include the
labelling noise; no assumption is made about how prediction errors vary over
time or between groups. Without `label_prob` the labelled rows are treated as a
simple random sample of fixed size (``\\pi = n_{\\text{labeled}}/n``); the
default recalibration (``\\hat g`` includes cell indicators) keeps the
correction mean-zero in every cell, which makes this normalization innocuous,
whereas with `learner = nothing` and a strongly differential error,
Bernoulli-sampled labels should be analysed with `label_prob` and
`normalize_prob = false`.

**Stable-error mode** (`assume_stable_error = true`). When labels exist only in
some periods (for example only before treatment), the prediction error cannot
be learned where it matters without an assumption. This mode assumes that the
measurement model ``E[f \\mid Y, \\text{group}, W] = a_g + bY + \\gamma'W``
(group-specific intercept, common slope, `features` ``W``) is the same in every
period, including after treatment, so a treatment-induced change in how the
measure errs is ruled out by assumption and cannot be detected. The model is
fitted by least squares on the labelled rows and inverted,
``\\hat Y = (f - a_g - \\gamma'W)/b``, which undoes attenuation (``b < 1``) and
group-specific shifts, in the spirit of regression calibration (Carroll et al.
2006); `learner` is not used. The labels must be selected independently of the
prediction given the outcome. Standard errors come from a unit (cluster)
bootstrap that refits the measurement model. Supported for `:twfe` and
`:drdid`.

**Held-out discipline and reporting.** A measurement trained or fine-tuned on
units of the study must not be evaluated on those units: pass
`measure_training` (0/1 column flagging rows used to train the measure) and
every unit with a flagged row is removed from the analysis. Alternatively leave
`prediction = nothing` and pass the raw inputs as `features`: the measure is then
learned from the labelled rows by cross-fitting by unit, and the naive
comparison uses that out-of-fold measure (cross-prediction; Zrnic & Candès
2024). Report the corrected estimate, the naive estimate and `bias_test`; a
non-rejection of the latter is not evidence that the naive estimate is
unbiased. [`differential_error_test`](@ref) describes how the mean error
differs across cells.

# Arguments
- `data::AbstractDataFrame`: long panel, one row per unit and period.
- `outcome::Symbol`: gold-standard outcome, `missing` when not labelled (unless
  `labeled` is given).
- `treatment`: treatment column (0/1) or `FirstTreated(:g)`, as in the chosen
  estimator.
- `unit::Symbol`, `time::Symbol`: unit and period identifiers.

# Keywords
- `prediction::Union{Nothing,Symbol} = nothing`: ML measurement of the outcome on
  every row.
- `estimator::Symbol = :twfe`: `:twfe` ([`did_twfe`](@ref)), `:cs`
  ([`did_callaway_santanna`](@ref)) or `:drdid` ([`did_drdid`](@ref), two
  periods).
- `features::Vector{Symbol} = Symbol[]`, `learner = OLSLearner()`,
  `labeled = nothing`, `label_prob = nothing`, `normalize_prob = true`: as in
  [`dsl_pseudo_outcome`](@ref); `learner = nothing` uses the raw prediction as
  ``\\hat g``.
- `measure_training::Union{Nothing,Symbol} = nothing`: 0/1 column flagging rows
  used to train the measurement; their units are dropped.
- `assume_stable_error::Bool = false`: use the stable-error mode.
- `cluster::Union{Nothing,Symbol} = nothing`: clustering and cross-fitting groups
  (default: units); passed to the estimator.
- `n_folds = 5`, `folds = nothing`, `rng = Random.default_rng()`,
  `parallel = false`: cross-fitting options.
- `bootstrap_reps::Integer = 199`: bootstrap replications in the stable-error
  mode.
- `kwargs...`: passed to the DiD estimator (e.g. `covariates`,
  `control_group`).

# Returns
- [`MeasuredOutcomeEstimate`](@ref) wrapping the corrected and naive estimates;
  `bias_test` tests whether the estimator applied to the pseudo-outcome minus
  the prediction is zero. With `:cs`, aggregate with
  `aggregate_att(r.corrected, ...)`.

# Examples
```julia
using DrSnow, DataFrames, StableRNGs
rng = StableRNG(20)
N, T = 200, 4
id, year = repeat(1:N, inner=T), repeat(1:T, outer=N)
tr = repeat(Float64.(rand(rng, N) .< 0.5), inner=T)
treated = tr .* (year .>= 3)
tone = repeat(randn(rng, N), inner=T) .+ 0.3 .* year .+ treated .+
       randn(rng, N * T)
tone_llm = 0.2 .+ 0.9 .* tone .+ 0.6 .* treated .+ 0.5 .* randn(rng, N * T)
lab = rand(rng, N * T) .< 0.3
df = DataFrame(; id, year, treated, tone_llm,
               tone_expert=[l ? v : missing for (l, v) in zip(lab, tone)])
r = did_with_predicted_outcome(df, :tone_expert, :treated, :id, :year;
                               prediction=:tone_llm, rng=StableRNG(21))
coef(r), confint(r), r.bias_test
```

# References
$(_ML_MEAS_DESIGN_REFS)
- Callaway, B., & Sant'Anna, P. H. C. (2021). Difference-in-differences with
  multiple time periods. *Journal of Econometrics*, 225(2), 200–230.
- Sant'Anna, P. H. C., & Zhao, J. (2020). Doubly robust difference-in-differences
  estimators. *Journal of Econometrics*, 219(1), 101–122.
"""
function did_with_predicted_outcome(data::AbstractDataFrame, outcome::Symbol, treatment,
                                    unit::Symbol, time::Symbol;
                                    prediction::Union{Nothing,Symbol}=nothing,
                                    estimator::Symbol=:twfe,
                                    features::Vector{Symbol}=Symbol[],
                                    learner=OLSLearner(),
                                    labeled::Union{Nothing,Symbol}=nothing,
                                    label_prob::Union{Nothing,Symbol}=nothing,
                                    normalize_prob::Bool=true,
                                    measure_training::Union{Nothing,Symbol}=nothing,
                                    assume_stable_error::Bool=false,
                                    cluster::Union{Nothing,Symbol}=nothing,
                                    n_folds::Integer=5, folds=nothing,
                                    bootstrap_reps::Integer=199,
                                    rng::AbstractRNG=Random.default_rng(),
                                    parallel::Bool=false, kwargs...)
    ctx = "did_with_predicted_outcome"
    estimator in (:twfe, :cs, :drdid) ||
        throw(ArgumentError("$(ctx): estimator must be :twfe, :cs or :drdid"))
    (assume_stable_error && estimator === :cs) &&
        throw(ArgumentError("$(ctx): the stable-error mode supports :twfe and :drdid; " *
                            "for :cs label rows in every period (design-based mode)"))
    preds = _as_symbols(prediction)
    (isempty(preds) && isempty(features)) &&
        throw(ArgumentError("$(ctx): pass `prediction` and/or `features`"))
    tcol = _did_treatment_column(treatment)
    covs = get(kwargs, :covariates, Symbol[])
    need = unique(vcat(outcome, tcol, unit, time, preds, features, labeled, label_prob,
                       measure_training, cluster, covs,
                       folds isa Symbol ? folds : nothing))
    need = [c for c in need if c !== nothing]
    require_columns(data, need; context=ctx)
    _ml_meas_complete(data, [c for c in need if c != outcome]; context=ctx)
    df = DataFrame([c => data[!, c] for c in need])
    grp = cluster === nothing ? unit : cluster
    df, ntrain = _ml_meas_drop_training(df, measure_training, grp; context=ctx)
    # cohort of each unit (first period with treatment, or the FirstTreated value)
    cohort = _ml_meas_cohort(df, treatment, tcol, unit, time)
    ever = Float64[c !== nothing for c in cohort]
    cells = [(ever[i] == 1 ? "treated" : "comparison", df[i, time]) for i in 1:nrow(df)]
    dnow = treatment isa Symbol ? _ml_column(df, tcol; context=ctx) :
           Float64.([ever[i] == 1 && df[i, time] >= cohort[i] for i in 1:nrow(df)])
    f1 = isempty(preds) ? zeros(nrow(df)) : _ml_column(df, preds[1]; context=ctx)
    design_extra = hcat(_ml_meas_dummies([c === nothing ? "never" : string(c)
                                          for c in cohort]),
                        _ml_meas_dummies(df[!, time]), dnow, dnow .* f1)
    stable_extra = reshape(ever, :, 1)
    est_kw = cluster === nothing ? kwargs : merge(NamedTuple(kwargs), (cluster=cluster,))
    fitfun = function (d, ycol)
        if estimator === :twfe
            return did_twfe(d, ycol, treatment, unit, time; warn_heterogeneity=false,
                            est_kw...)
        elseif estimator === :cs
            return did_callaway_santanna(d, ycol, treatment, unit, time; bootstrap=false,
                                         est_kw...)
        else
            return did_drdid(d, ycol, treatment, unit, time; est_kw...)
        end
    end
    fit_same = (rc, d, ycol) -> fitfun(d, ycol)
    return _ml_meas_natural(df, outcome, preds, features, labeled, label_prob,
                            normalize_prob, learner, grp, grp, unit, cells, design_extra,
                            stable_extra, fitfun, fit_same, assume_stable_error, n_folds,
                            folds, bootstrap_reps, rng, parallel, ntrain; context=ctx)
end

"""Cohort (first treated period, `nothing` if never treated) of each row's unit."""
function _ml_meas_cohort(df, treatment, tcol, unit, time)
    n = nrow(df)
    if treatment isa FirstTreated
        v = df[!, tcol]
        return Any[v[i] == 0 ? nothing : v[i] for i in 1:n]
    end
    first_t = Dict{Any,Any}()
    for i in 1:n
        if df[i, tcol] == 1
            u = df[i, unit]
            t = df[i, time]
            first_t[u] = haskey(first_t, u) ? min(first_t[u], t) : t
        end
    end
    return Any[get(first_t, df[i, unit], nothing) for i in 1:n]
end

"""
    rd_with_predicted_outcome(data, outcome, running; prediction=nothing, cutoff=0.0,
                              features=Symbol[], learner=OLSLearner(),
                              labeled=nothing, label_prob=nothing,
                              normalize_prob=true, measure_training=nothing,
                              assume_stable_error=false, cluster=nothing, n_folds=5,
                              folds=nothing, bootstrap_reps=199,
                              rng=Random.default_rng(), parallel=false, kwargs...)
        -> MeasuredOutcomeEstimate

Sharp regression discontinuity when the outcome is measured by an ML model or
LLM and a subsample carries gold-standard labels.

The estimand is the jump at the cutoff of the conditional mean of the *true*
outcome, ``\\tau = \\lim_{x \\downarrow c} E[Y \\mid X = x] - \\lim_{x \\uparrow c}
E[Y \\mid X = x]``, identified under the usual continuity assumptions of the
sharp design. The naive RD estimate computed with the ML measurement is biased
when the mean prediction error jumps at the cutoff, for example because the
treatment changes what the model reacts to.

**Design-based mode** (default). [`rd_estimate`](@ref) is applied to the DSL
pseudo-outcome ``\\tilde Y = \\hat g + (R/\\pi)(Y - \\hat g)`` of
[`dsl_pseudo_outcome`](@ref), with ``\\hat g`` cross-fitted (by `cluster` when
given) from the prediction, `features`, the side of the cutoff, the centred
running variable and their interactions. The gold labels are assumed
error-free, and the labelling probabilities must be known by design and
bounded away from zero on both sides near the cutoff; they may depend on the
running variable, and oversampling near the cutoff is efficient. Local
polynomial estimates are linear in the outcome given the bandwidths, so the
estimate targets the discontinuity of the true outcome and the robust
bias-corrected inference of Calonico, Cattaneo and Titiunik (2014) computed on
``\\tilde Y`` accounts for the labelling. The bandwidths are selected by
[`rd_estimate`](@ref)'s data-driven selector applied to the pseudo-outcome,
not to the true outcome (which is unobserved for most units); since
``\\tilde Y`` is noisier than ``Y``, they generally differ from those the true
outcome would give, and inference is conditional on them as usual. Pass `h`
(and `b`) to fix them. The naive estimate and the contrast test reuse the
corrected fit's bandwidths, so the difference of their bias-corrected estimates
(`tau_bias_corrected`, the centre of robust inference) equals the contrast estimate.

**Stable-error mode** (`assume_stable_error = true`). With labels on one side
only (or unknown labelling probabilities), the linear measurement model
``E[f \\mid Y, W] = a + bY + \\gamma'W`` is assumed to be identical on both
sides of the cutoff, fitted on the labelled rows and inverted,
``\\hat Y = (f - a - \\gamma'W)/b``, which rescales the naive discontinuity by
``1/b``. A jump in the prediction error at the cutoff is ruled out by
assumption. Bootstrap inference re-estimates the measurement model with the
bandwidths fixed at the corrected fit's values.

Report the corrected and naive estimates, the bandwidths and `bias_test`
(whose non-rejection is not evidence that the naive estimate is unbiased).
Measurements trained on units of the study must be excluded with
`measure_training`, as in [`did_with_predicted_outcome`](@ref).

# Arguments
- `data::AbstractDataFrame`: one row per unit.
- `outcome::Symbol`: gold-standard outcome, `missing` when not labelled (unless
  `labeled` is given).
- `running::Symbol`: the running variable.

# Keywords
- `prediction::Union{Nothing,Symbol} = nothing`: ML measurement of the outcome on
  every row.
- `cutoff::Real = 0.0`: the cutoff; units at or above it are treated.
- `features`, `learner`, `labeled`, `label_prob`, `normalize_prob`,
  `measure_training`, `assume_stable_error`, `cluster`, `n_folds`, `folds`,
  `bootstrap_reps`, `rng`, `parallel`: as in
  [`did_with_predicted_outcome`](@ref); `measure_training` drops the flagged
  rows, or their clusters when `cluster` is given.
- `kwargs...`: passed to [`rd_estimate`](@ref) (e.g. `h`, `p`, `kernel`,
  `vce`); fuzzy designs are not supported.

# Returns
- [`MeasuredOutcomeEstimate`](@ref) wrapping `RDEstimate`s.

# Examples
```julia
using DrSnow, DataFrames, StableRNGs
rng = StableRNG(30)
n = 3000
margin = 2 .* rand(rng, n) .- 1
y = 0.5 .* margin .+ 1.0 .* (margin .>= 0) .+ 0.5 .* randn(rng, n)
y_llm = 0.9 .* y .+ 0.4 .* (margin .>= 0) .+ 0.3 .* randn(rng, n)
p_label = ifelse.(abs.(margin) .< 0.3, 0.5, 0.1)           # oversample near 0
lab = rand(rng, n) .< p_label
df = DataFrame(y_expert=[l ? v : missing for (l, v) in zip(lab, y)], y_llm=y_llm,
               margin=margin, p_label=p_label)
r = rd_with_predicted_outcome(df, :y_expert, :margin; prediction=:y_llm,
                              label_prob=:p_label, normalize_prob=false,
                              rng=StableRNG(31))
coef(r), confint(r), r.bias_test
```

# References
$(_ML_MEAS_DESIGN_REFS)
- Calonico, S., Cattaneo, M. D., & Titiunik, R. (2014). Robust nonparametric
  confidence intervals for regression-discontinuity designs. *Econometrica*,
  82(6), 2295–2326.
"""
function rd_with_predicted_outcome(data::AbstractDataFrame, outcome::Symbol,
                                   running::Symbol;
                                   prediction::Union{Nothing,Symbol}=nothing,
                                   cutoff::Real=0.0, features::Vector{Symbol}=Symbol[],
                                   learner=OLSLearner(),
                                   labeled::Union{Nothing,Symbol}=nothing,
                                   label_prob::Union{Nothing,Symbol}=nothing,
                                   normalize_prob::Bool=true,
                                   measure_training::Union{Nothing,Symbol}=nothing,
                                   assume_stable_error::Bool=false,
                                   cluster::Union{Nothing,Symbol}=nothing,
                                   n_folds::Integer=5, folds=nothing,
                                   bootstrap_reps::Integer=199,
                                   rng::AbstractRNG=Random.default_rng(),
                                   parallel::Bool=false, kwargs...)
    ctx = "rd_with_predicted_outcome"
    preds = _as_symbols(prediction)
    (isempty(preds) && isempty(features)) &&
        throw(ArgumentError("$(ctx): pass `prediction` and/or `features`"))
    haskey(kwargs, :treatment) &&
        throw(ArgumentError("$(ctx): only sharp designs are supported"))
    covs = get(kwargs, :covariates, Symbol[])
    need = unique(vcat(outcome, running, preds, features, labeled, label_prob,
                       measure_training, cluster, covs,
                       folds isa Symbol ? folds : nothing))
    need = [c for c in need if c !== nothing]
    require_columns(data, need; context=ctx)
    _ml_meas_complete(data, [c for c in need if c != outcome]; context=ctx)
    df = DataFrame([c => data[!, c] for c in need])
    df, ntrain = _ml_meas_drop_training(df, measure_training, cluster; context=ctx)
    x = _ml_column(df, running; context=ctx) .- cutoff
    side = Float64.(x .>= 0)
    cells = [s == 1 ? "right of cutoff" : "left of cutoff" for s in side]
    f1 = isempty(preds) ? zeros(nrow(df)) : _ml_column(df, preds[1]; context=ctx)
    design_extra = hcat(side, x, side .* x, side .* f1)
    stable_extra = zeros(nrow(df), 0)
    base_kw = cluster === nothing ? NamedTuple(kwargs) :
              merge(NamedTuple(kwargs), (cluster=cluster,))
    fitfun = (d, ycol) -> rd_estimate(d, ycol, running; cutoff=cutoff, base_kw...)
    fit_same = function (rc, d, ycol)
        kw = merge(Base.structdiff(base_kw, NamedTuple{(:rho, :bwselect)}),
                   (h=(rc.h_left, rc.h_right), b=(rc.b_left, rc.b_right)))
        return rd_estimate(d, ycol, running; cutoff=cutoff, kw...)
    end
    return _ml_meas_natural(df, outcome, preds, features, labeled, label_prob,
                            normalize_prob, learner, cluster, cluster, nothing, cells,
                            design_extra, stable_extra, fitfun, fit_same,
                            assume_stable_error, n_folds, folds, bootstrap_reps, rng,
                            parallel, ntrain; context=ctx)
end
