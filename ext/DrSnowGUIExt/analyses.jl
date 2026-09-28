# The analyses offered by the GUI. Each takes the session's table and the validated
# request options and returns a result dictionary (see `_gui_new_result`). Only the
# functions registered in `_GUI_ANALYSES` can be reached from a request.

_gui_cols_df(df::DataFrame, cols) = df[!, unique([c for c in cols if c !== nothing])]

# --- DiD: two-way fixed effects --------------------------------------------------------

function _gui_run_twfe(df::DataFrame, p)
    y = _gui_column(df, p, "outcome", "outcome"; numeric=true)
    treat, tcol, _ = _gui_did_treatment(df, p)
    unit = _gui_column(df, p, "unit", "unit identifier")
    time = _gui_column(df, p, "time", "time variable")
    covs = _gui_columns(df, p, "covariates", "covariates"; numeric=true)
    cl = _gui_cluster(df, p, unit)
    level = _gui_level(p)
    bacon = _gui_bool(p, "bacon", false)
    _gui_distinct!(y, tcol, unit, time)
    data = _gui_cols_df(df, [y, tcol, unit, time, covs..., cl])
    r = did_twfe(data, y, treat, unit, time; covariates=covs, cluster=cl)
    res = _gui_new_result("did_twfe", r)
    push!(res["tables"], _gui_coef_table(r, level))
    push!(res["notes"], "With staggered adoption or heterogeneous effects the TWFE " *
                        "coefficient can be a non-convex average of cohort-period " *
                        "effects; compare it with Callaway–Sant'Anna or an event study.")
    if bacon
        b = bacon_decomposition(data, y, treat, unit, time)
        push!(res["tables"], _gui_df_table(b.by_type, "Goodman-Bacon decomposition " *
            "(by comparison type)"; note="Weights sum to one; the weighted average of " *
            "the estimates equals the TWFE coefficient."))
        push!(res["tables"], _gui_df_table(b.comparisons, "Goodman-Bacon decomposition " *
            "(all 2×2 comparisons)"))
    end
    return res
end

# --- DiD: event study ------------------------------------------------------------------

function _gui_run_event_study(df::DataFrame, p)
    y = _gui_column(df, p, "outcome", "outcome"; numeric=true)
    treat, tcol, _ = _gui_did_treatment(df, p)
    unit = _gui_column(df, p, "unit", "unit identifier")
    time = _gui_column(df, p, "time", "time variable")
    covs = _gui_columns(df, p, "covariates", "covariates"; numeric=true)
    cl = _gui_cluster(df, p, unit)
    level = _gui_level(p)
    est = _gui_choice(p, "estimator",
                      (:auto, :twfe, :sun_abraham, :imputation, :callaway_santanna), :auto)
    max_pre = _gui_int(p, "max_pre", nothing, 0, 1000)
    max_post = _gui_int(p, "max_post", nothing, 0, 1000)
    _gui_distinct!(y, tcol, unit, time)
    data = _gui_cols_df(df, [y, tcol, unit, time, covs..., cl])
    kw = est === :callaway_santanna ? (; rng=_gui_rng(p)) : (;)
    es = event_study(data, y, treat, unit, time; estimator=est, max_pre=max_pre,
                     max_post=max_post, covariates=covs, cluster=cl, kw...)
    res = _gui_new_result("event_study", es)
    push!(res["tables"], _gui_coef_table(es, level; title="Event-time coefficients"))
    res["plot"] = _gui_event_plot(es, level)
    push!(res["notes"], "Reference period(s) normalized to zero: " *
                        join(es.reference, ", ") * ". Intervals are pointwise.")
    if any(<(0), relative_periods(es))
        push!(res["diagnostics"], _gui_diagnostic(pre_trend_test(es)))
    else
        push!(res["notes"], "No pre-treatment coefficients were estimated, so no " *
                            "pre-trend test is reported.")
    end
    return res
end

# --- DiD: Callaway–Sant'Anna -----------------------------------------------------------

function _gui_run_cs(df::DataFrame, p)
    y = _gui_column(df, p, "outcome", "outcome"; numeric=true)
    treat, tcol, _ = _gui_did_treatment(df, p)
    unit = _gui_column(df, p, "unit", "unit identifier")
    time = _gui_column(df, p, "time", "time variable")
    covs = _gui_columns(df, p, "covariates", "covariates"; numeric=true)
    cl = _gui_cluster(df, p, nothing)
    level = _gui_level(p)
    cg = _gui_choice(p, "control_group", (:never_treated, :not_yet_treated),
                     :never_treated)
    method = _gui_choice(p, "method", (:dr, :dr_improved, :ipw, :reg), :dr)
    agg = _gui_choice(p, "aggregation", (:simple, :group, :dynamic, :calendar), :dynamic)
    biters = _gui_int(p, "biters", 999, 99, 9999)
    _gui_distinct!(y, tcol, unit, time)
    data = _gui_cols_df(df, [y, tcol, unit, time, covs..., cl])
    rng = _gui_rng(p)
    cs = did_callaway_santanna(data, y, treat, unit, time; covariates=covs,
                               control_group=cg, method=method, cluster=cl,
                               biters=biters, rng=rng)
    a = aggregate_att(cs, agg; rng=rng)
    res = _gui_new_result("did_cs", a)
    res["title"] = method_name(cs)
    push!(res["tables"], _gui_coef_table(a, level;
                                          title="Aggregated ATT ($(agg) aggregation)"))
    a isa EventStudyEstimate && (res["plot"] = _gui_event_plot(a, level;
        title="Callaway–Sant'Anna ATT by event time"))
    push!(res["tables"], _gui_coef_table(cs, level; title="Group-time ATT(g, t)"))
    push!(res["notes"], "Pointwise intervals use analytic standard errors; the " *
                        "multiplier bootstrap ($(biters) draws, seed shown in the " *
                        "specification) is used for simultaneous bands only.")
    try
        push!(res["diagnostics"], _gui_diagnostic(pre_trend_test(cs)))
    catch err
        err isa ArgumentError || rethrow()
        push!(res["notes"], "Pre-trend test not available: " *
                            _gui_clean_message(err.msg))
    end
    res["summary"] = _gui_text(cs) * "\n\n" * _gui_text(a)
    return res
end

# --- Regression discontinuity ----------------------------------------------------------

function _gui_run_rd(df::DataFrame, p)
    y = _gui_column(df, p, "outcome", "outcome"; numeric=true)
    x = _gui_column(df, p, "running", "running variable"; numeric=true)
    d = _gui_column(df, p, "treatment", "treatment (fuzzy design)"; required=false,
                    numeric=true)
    cl = _gui_cluster(df, p, nothing)
    cutoff = _gui_float(p, "cutoff", 0.0, -1e15, 1e15)
    order = _gui_int(p, "p", 1, 0, 4)
    kernel = _gui_choice(p, "kernel", (:triangular, :epanechnikov, :uniform), :triangular)
    bw = _gui_choice(p, "bwselect", (:mserd, :msetwo, :cerrd, :certwo), :mserd)
    level = _gui_level(p)
    density = _gui_bool(p, "density_test", true)
    _gui_distinct!(y, x, d)
    data = _gui_cols_df(df, [y, x, d, cl])
    r = rd_estimate(data, y, x; cutoff=cutoff, treatment=d, p=order, kernel=kernel,
                    bwselect=bw, cluster=cl, level=level)
    res = _gui_new_result("rd", r)
    push!(res["tables"], _gui_coef_table(r, level))
    push!(res["tables"], _gui_df_table(rd_inference_table(r; level=level),
                                       "Conventional, bias-corrected and robust inference"))
    density && push!(res["diagnostics"], _gui_diagnostic(
        rd_density_test(data, x; cutoff=cutoff)))
    pd = rd_plot_data(data, y, x; cutoff=cutoff, level=level)
    res["plot"] = _gui_rd_plot(pd, y, x)
    push!(res["notes"], "The binned scatter and global polynomial are a visual summary; " *
                        "the estimate uses local polynomials within the bandwidth.")
    return res
end

# --- Instrumental variables ------------------------------------------------------------

function _gui_run_iv(df::DataFrame, p)
    y = _gui_column(df, p, "outcome", "outcome"; numeric=true)
    d = _gui_column(df, p, "endogenous", "endogenous treatment"; numeric=true)
    z = _gui_columns(df, p, "instruments", "instruments"; numeric=true, min=1)
    covs = _gui_columns(df, p, "covariates", "covariates"; numeric=true)
    fes = _gui_columns(df, p, "fe", "fixed effects")
    cl = _gui_cluster(df, p, nothing)
    level = _gui_level(p)
    _gui_distinct!(y, d, z...)
    data = _gui_cols_df(df, [y, d, z..., covs..., fes..., cl])
    r = iv_regression(data, y, [d], z; covariates=covs, fe=fes, cluster=cl, level=level)
    res = _gui_new_result("iv", r)
    push!(res["tables"], _gui_coef_table(r, level))
    fs = r.first_stage
    rows = [Any[string(s.endogenous), _gui_num(s.F), _gui_num(s.F_pvalue),
                _gui_num(s.F_homoskedastic), _gui_num(s.partial_r2)]
            for s in fs.first_stage]
    push!(res["tables"], _gui_table("First stage",
        ["Endogenous", "F ($(fs.vcov_type))", "p-value", "Conventional F", "Partial R²"],
        rows))
    push!(res["text_blocks"], _gui_text_block("Weak-instrument diagnostics", fs))
    cs = weak_iv_confidence_set(r; method=:ar, level=level)
    push!(res["text_blocks"], _gui_text_block("Anderson–Rubin confidence set " *
                                              "(robust to weak instruments)", cs))
    push!(res["diagnostics"], _gui_diagnostic(weak_iv_test(r; beta0=0.0, method=:ar)))
    isempty(r.estimand_note) || push!(res["notes"], r.estimand_note)
    return res
end

# --- Synthetic difference-in-differences -----------------------------------------------

function _gui_run_sdid(df::DataFrame, p)
    y = _gui_column(df, p, "outcome", "outcome"; numeric=true)
    d = _gui_column(df, p, "treatment", "treatment indicator"; numeric=true)
    unit = _gui_column(df, p, "unit", "unit identifier")
    time = _gui_column(df, p, "time", "time variable")
    method = _gui_choice(p, "method", (:sdid, :sc, :did), :sdid)
    se = _gui_choice(p, "se_method", (:placebo, :bootstrap, :jackknife, :none), :placebo)
    reps = _gui_int(p, "replications", 200, 10, 2000)
    level = _gui_level(p)
    _gui_distinct!(y, d, unit, time)
    data = _gui_cols_df(df, [y, d, unit, time])
    r = synthetic_did(data, y, d, unit, time; method=method, se_method=se,
                      replications=reps, rng=_gui_rng(p))
    res = _gui_new_result("sdid", r)
    if se === :none
        push!(res["tables"], _gui_table("Estimates", ["Term", "Estimate"],
                                        [Any["ATT", _gui_num(r.att)]]))
        push!(res["notes"], "No standard error was requested (se_method = none).")
    else
        push!(res["tables"], _gui_coef_table(r, level))
    end
    w = synth_weights(r)
    w = w[sortperm(w.weight; rev=true), :]
    push!(res["tables"], _gui_df_table(w, "Unit weights (largest first)"; max_rows=50))
    push!(res["tables"], _gui_df_table(synth_time_weights(r), "Time weights"))
    res["plot"] = _gui_sdid_plot(synth_gaps(r), y)
    return res
end

const _GUI_ANALYSES = Dict{String,Function}(
    "did_twfe" => _gui_run_twfe,
    "event_study" => _gui_run_event_study,
    "did_cs" => _gui_run_cs,
    "rd" => _gui_run_rd,
    "iv" => _gui_run_iv,
    "sdid" => _gui_run_sdid,
)

# --- Warning capture -------------------------------------------------------------------

"""Collects warnings emitted while an analysis runs (shown to the user) and forwards
every message to the server's logger."""
struct _GuiWarnCollector <: _GUI_LOG.AbstractLogger
    parent::_GUI_LOG.AbstractLogger
    messages::Vector{String}
    lock::ReentrantLock
end

_GUI_LOG.min_enabled_level(l::_GuiWarnCollector) = _GUI_LOG.min_enabled_level(l.parent)
_GUI_LOG.shouldlog(::_GuiWarnCollector, args...) = true
_GUI_LOG.catch_exceptions(::_GuiWarnCollector) = true

function _GUI_LOG.handle_message(l::_GuiWarnCollector, level, message, _module, group,
                                 id, args...; kwargs...)
    if level >= _GUI_LOG.Warn && level < _GUI_LOG.Error
        lock(l.lock) do
            length(l.messages) < 20 &&
                push!(l.messages, _gui_clean_message(string(message); maxlen=1000))
        end
    end
    if _GUI_LOG.shouldlog(l.parent, level, _module, group, id)
        _GUI_LOG.handle_message(l.parent, level, message, _module, group, id, args...;
                                kwargs...)
    end
    return nothing
end

"""Run a registered analysis on a worker thread, collecting warnings."""
function _gui_run_analysis(design::AbstractString, df::DataFrame, params)
    f = get(_GUI_ANALYSES, design, nothing)
    f === nothing && throw(_GuiUserError(400, "Unknown analysis."))
    logger = _GuiWarnCollector(_GUI_LOG.current_logger(), String[], ReentrantLock())
    res = _GUI_LOG.with_logger(logger) do
        fetch(Threads.@spawn f(df, params))
    end
    append!(res["warnings"], logger.messages)
    return res
end
