# Request routing. Every response goes through `_gui_respond` (security headers);
# every failure produces a generic JSON error.

struct _GuiState
    config::_GuiConfig
    store::_GuiStore
end

const _GUI_ANALYSIS_PREFIX = "/api/results/"

function _gui_handle(state::_GuiState, http::HTTP.Stream)
    req = http.message
    path = "/"
    try
        uri = HTTP.URI(req.target)
        path = uri.path
        if !_gui_host_ok(state.config, req)
            return _gui_reject_unread(http, 400, "Host not allowed.")
        end
        return _gui_route(state, http, req.method, path, HTTP.queryparams(uri))
    catch err
        err = _gui_unwrap(err)
        if err isa Base.IOError || err isa EOFError
            return nothing        # client went away
        end
        status, msg, internal = _gui_classify_error(err)
        if internal
            ref = _gui_log_internal(err, catch_backtrace(), "$(req.method) $path")
            msg *= " Reference: $ref."
        end
        HTTP.IOExtras.iswritable(http) && return nothing   # response already started
        return _gui_reject_unread(http, status, msg)
    end
end

function _gui_route(state::_GuiState, http::HTTP.Stream, method::AbstractString,
                    path::AbstractString, query)
    cfg = state.config
    if method == "GET"
        path == "/healthz" && return _gui_respond(http, 200, "ok",
                                                  "text/plain; charset=utf-8";
                                                  headers=("Cache-Control" => "no-store",))
        asset = get(cfg.assets, path, nothing)
        asset === nothing || return _gui_respond(http, 200, asset.body, asset.mime;
                                                 headers=("Cache-Control" => "no-cache",))
    end
    if path == "/api/session"
        method == "POST" && return _gui_api_session_create(state, http)
        method == "DELETE" && return _gui_api_session_delete(state, http)
        return _gui_method_not_allowed(http, "POST, DELETE")
    elseif path == "/api/upload"
        method == "POST" || return _gui_method_not_allowed(http, "POST")
        return _gui_api_upload(state, http, query)
    elseif path == "/api/data"
        method == "GET" || return _gui_method_not_allowed(http, "GET")
        return _gui_api_data(state, http)
    elseif path == "/api/analyze"
        method == "POST" || return _gui_method_not_allowed(http, "POST")
        return _gui_api_analyze(state, http)
    elseif startswith(path, _GUI_ANALYSIS_PREFIX)
        method == "GET" || return _gui_method_not_allowed(http, "GET")
        return _gui_api_export(state, http, path[(length(_GUI_ANALYSIS_PREFIX) + 1):end],
                               query)
    end
    return _gui_reject_unread(http, 404, "Not found.")
end

_gui_method_not_allowed(http, allow) =
    _gui_reject_unread(http, 405, "Method not allowed.")

"""Session + CSRF check for API requests. Returns the session, or `nothing` after
sending an error response."""
function _gui_authenticate(state::_GuiState, http::HTTP.Stream; state_changing::Bool)
    req = http.message
    if state_changing && !_gui_origin_ok(req)
        _gui_reject_unread(http, 403, "Cross-origin request refused.")
        return nothing
    end
    s = _gui_session_get(state.store, _gui_cookie(req, _GUI_SESSION_COOKIE),
                         state.config.session_ttl)
    if s === nothing
        _gui_reject_unread(http, 401, "No valid session: reload the page to start a " *
                                      "new session (sessions expire when idle).")
        return nothing
    end
    token = HTTP.header(req, "X-CSRF-Token", "")
    if isempty(token) || !_gui_secure_equals(token, s.csrf)
        _gui_reject_unread(http, 403, "Missing or invalid CSRF token.")
        return nothing
    end
    return s
end

function _gui_limits(cfg::_GuiConfig)
    return Dict("max_upload_bytes" => cfg.max_upload_bytes, "max_rows" => cfg.max_rows,
                "max_cols" => cfg.max_cols,
                "session_ttl_minutes" => round(cfg.session_ttl / 60; digits=1))
end

function _gui_api_session_create(state::_GuiState, http::HTTP.Stream)
    req = http.message
    _gui_origin_ok(req) || return _gui_reject_unread(http, 403,
                                                     "Cross-origin request refused.")
    s = _gui_session_create!(state.store, state.config)
    s === nothing && return _gui_reject_unread(http, 503,
                                               "The server is busy; try again later.")
    cookie = "$(_GUI_SESSION_COOKIE)=$(s.id); Path=/; HttpOnly; SameSite=Strict"
    return _gui_json(http, 200, Dict("csrf_token" => s.csrf,
                                     "limits" => _gui_limits(state.config),
                                     "analyses" => sort!(collect(keys(_GUI_ANALYSES))));
                     headers=("Set-Cookie" => cookie,))
end

function _gui_api_session_delete(state::_GuiState, http::HTTP.Stream)
    s = _gui_authenticate(state, http; state_changing=true)
    s === nothing && return nothing
    _gui_session_delete!(state.store, s.id)
    expired = "$(_GUI_SESSION_COOKIE)=; Path=/; HttpOnly; SameSite=Strict; Max-Age=0"
    return _gui_json(http, 200, Dict("ok" => true); headers=("Set-Cookie" => expired,))
end

function _gui_api_upload(state::_GuiState, http::HTTP.Stream, query)
    cfg = state.config
    s = _gui_authenticate(state, http; state_changing=true)
    s === nothing && return nothing
    limit_mb = @sprintf("%.3g", cfg.max_upload_bytes / 2^20)
    body = _gui_read_body(http, cfg.max_upload_bytes)
    body === nothing && return _gui_reject_unread(http, 413,
        "The file is larger than the upload limit ($limit_mb MB).")
    delim = get(query, "delim", "auto")
    df = fetch(Threads.@spawn _gui_parse_table(body, cfg, delim))
    name = _gui_clean_filename(HTTP.header(http.message, "X-Filename", "upload.csv"))
    _gui_set_data!(state.store, s, df, name)
    return _gui_json(http, 200, _gui_table_summary(df, name))
end

function _gui_api_data(state::_GuiState, http::HTTP.Stream)
    s = _gui_authenticate(state, http; state_changing=false)
    s === nothing && return nothing
    df = _gui_get_data(state.store, s)
    df === nothing && return _gui_error(http, 404, "No data uploaded yet.")
    return _gui_json(http, 200, _gui_table_summary(df, s.filename))
end

function _gui_api_analyze(state::_GuiState, http::HTTP.Stream)
    cfg = state.config
    s = _gui_authenticate(state, http; state_changing=true)
    s === nothing && return nothing
    obj = _gui_read_json(http, cfg)
    obj === nothing && return _gui_reject_unread(http, 413, "Request too large.")
    design = get(obj, :design, nothing)
    design isa AbstractString && haskey(_GUI_ANALYSES, design) ||
        throw(_GuiUserError(400, "Unknown analysis."))
    params = get(obj, :params, nothing)
    params === nothing && (params = JSON3.read("{}"))
    params isa JSON3.Object || throw(_GuiUserError(400, "Invalid analysis options."))
    df = _gui_get_data(state.store, s)
    df === nothing && throw(_GuiUserError(409, "Upload a data file first."))
    _gui_try_acquire!(state.store, s) ||
        throw(_GuiUserError(409, "An analysis is already running in this session."))
    res = try
        _gui_run_analysis(design, df, params)
    finally
        _gui_release!(state.store, s)
    end
    id = _gui_token(16)
    res["id"] = id
    res["spec"] = _gui_spec(params)
    _gui_add_result!(state.store, s, id, res, cfg.max_results)
    return _gui_json(http, 200, res)
end

"""Echo of the options used (JSON scalars and string lists only), for the record."""
function _gui_spec(params)
    out = Dict{String,Any}()
    for (k, v) in pairs(params)
        if v isa Union{AbstractString,Real,Bool}
            out[string(k)] = v isa AbstractString ? _gui_cell(v) : _gui_num(v)
        elseif v isa AbstractVector
            out[string(k)] = [_gui_cell(x) for x in v if x isa AbstractString]
        end
    end
    return out
end

function _gui_api_export(state::_GuiState, http::HTTP.Stream, id::AbstractString, query)
    s = _gui_authenticate(state, http; state_changing=false)
    s === nothing && return nothing
    res = _gui_get_result(state.store, s, id)
    res === nothing && return _gui_error(http, 404, "Result not found.")
    fmt = get(query, "format", "json")
    base = "drsnow_" * res["design"] * "_" * first(res["id"], 8)
    if fmt == "json"
        return _gui_respond(http, 200, JSON3.write(res), "application/json; charset=utf-8";
                            headers=("Cache-Control" => "no-store",
                                     "Content-Disposition" =>
                                         "attachment; filename=\"$base.json\""))
    elseif fmt == "csv"
        tables = res["tables"]
        k = tryparse(Int, get(query, "table", "1"))
        (k === nothing || !(1 <= k <= length(tables))) &&
            return _gui_error(http, 404, "Table not found.")
        return _gui_respond(http, 200, _gui_table_csv(tables[k]), "text/csv; charset=utf-8";
                            headers=("Cache-Control" => "no-store",
                                     "Content-Disposition" =>
                                         "attachment; filename=\"$(base)_table$k.csv\""))
    end
    return _gui_error(http, 400, "Unknown export format.")
end
