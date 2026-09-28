# Web GUI (package extension DrSnowGUIExt): server, security and analysis endpoints.
#
# A server is started asynchronously on a free port and every endpoint is exercised
# over real HTTP with HTTP.jl (and raw sockets where HTTP.jl would hide the behaviour).

@testset "using DrSnow alone does not load the GUI dependencies" begin
    code = """
        using DrSnow
        loaded = Set(string(k.name) for k in keys(Base.loaded_modules))
        print(any(in(loaded), ("HTTP", "JSON3", "CSV")), " ",
              Base.get_extension(DrSnow, :DrSnowGUIExt) === nothing, " ")
        try
            launch_gui()
        catch err
            print(occursin("using DrSnow, HTTP, JSON3, CSV", sprint(showerror, err)))
        end
        """
    project = Base.active_project()
    cmd = `$(Base.julia_cmd()) --startup-file=no --project=$project -e $code`
    @test readchomp(cmd) == "false true true"
end

using HTTP, JSON3, CSV, Sockets, SHA, Base64

const GUIExt = Base.get_extension(DrSnow, :DrSnowGUIExt)

@testset "extension loads with HTTP, JSON3 and CSV" begin
    @test GUIExt !== nothing
    @test stop_gui() == false
end

# --- test data ------------------------------------------------------------------------

function gui_csv(df; delim=',')
    io = IOBuffer()
    CSV.write(io, df; delim=delim)
    return take!(io)
end

function gui_panel(rng; N=60, T=8)
    rows = NamedTuple[]
    for i in 1:N, t in 1:T
        g = i <= 10 ? 5 : i <= 20 ? 6 : 0
        d = g > 0 && t >= g ? 1 : 0
        push!(rows, (unit=i, year=2000 + t, first=g == 0 ? 0 : 2000 + g, d=d,
                     y=i / 10 + t / 5 + 2d + randn(rng), x1=randn(rng)))
    end
    return DataFrame(rows)
end

function gui_rd(rng; n=600)
    x = 2 .* rand(rng, n) .- 1
    y = 1 .+ x .+ 1.5 .* (x .>= 0) .+ 0.5 .* randn(rng, n)
    return DataFrame(score=round.(x; digits=5), outcome=round.(y; digits=5))
end

function gui_iv(rng; n=800)
    z = rand(rng, 0:1, n)
    d = Float64.(z .+ randn(rng, n) .> 0.5)
    y = 1 .+ 2 .* d .+ randn(rng, n)
    return DataFrame(y=y, d=d, z=z, w=randn(rng, n))
end

# --- HTTP helpers ---------------------------------------------------------------------

const GUI_OPTS = (status_exception=false, cookies=false, retry=false, redirect=false,
                  readtimeout=300)

struct GuiClient
    base::String
    host::String
    sid::String
    csrf::String
end

function gui_session(base, host)
    r = HTTP.post(base * "api/session", ["Host" => host]; GUI_OPTS...)
    @assert r.status == 200
    m = match(r"drsnow_sid=([0-9a-f]+)", HTTP.header(r, "Set-Cookie"))
    return GuiClient(base, host, m.captures[1], JSON3.read(r.body).csrf_token)
end

gui_headers(c::GuiClient; csrf=c.csrf, extra=Pair{String,String}[]) =
    ["Host" => c.host, "Cookie" => "drsnow_sid=$(c.sid)", "X-CSRF-Token" => csrf, extra...]

gui_get(c::GuiClient, path; kw...) =
    HTTP.get(c.base * path, gui_headers(c; kw...); GUI_OPTS...)
gui_post(c::GuiClient, path, body; kw...) =
    HTTP.post(c.base * path, gui_headers(c; kw...), body; GUI_OPTS...)

gui_upload(c::GuiClient, bytes; delim="auto", kw...) =
    gui_post(c, "api/upload?delim=$delim", bytes; kw...)

function gui_analyze(c::GuiClient, design, params)
    r = gui_post(c, "api/analyze", JSON3.write(Dict("design" => design,
                                                    "params" => params));
                 extra=["Content-Type" => "application/json"])
    return r, JSON3.read(r.body)
end

jbody(r) = JSON3.read(r.body)

"""Send raw bytes on a fresh socket and return everything the server sends back."""
function gui_raw(port, request::Vector{UInt8}; timeout=20)
    sock = Sockets.connect(Sockets.localhost, port)
    try
        write(sock, request)
    catch
    end
    t = @async try
        read(sock, String)
    catch err
        "ERROR: " * sprint(showerror, err)
    end
    status = timedwait(() -> istaskdone(t), timeout)
    close(sock)
    return status === :ok ? fetch(t) : "TIMEOUT"
end

# Any response body must be free of stack traces and source paths.
leaks(body) = occursin("Stacktrace", body) || occursin(r"\.jl:\d+", body) ||
              occursin("/home/", body) || occursin("ERROR:", body)

const GUI_LIMIT_MB = 0.25
srv = launch_gui(; port=0, async=true, open_browser=false, verbose=false,
                 max_upload_mb=GUI_LIMIT_MB, max_rows=5000, max_cols=40)
const BASE = srv.url
const HOST = "127.0.0.1:$(srv.port)"

try

@testset "server lifecycle" begin
    @test srv.host == "127.0.0.1"
    @test srv.port > 0
    @test startswith(BASE, "http://127.0.0.1:")
    @test_throws ErrorException launch_gui(; port=0, async=true, open_browser=false,
                                           verbose=false)
    st = GUIExt._GUI_SERVER[]
    @test st isa GUIExt._GuiServer
    @test isopen(st.cleanup)                      # periodic session cleanup task
end

@testset "static assets and security headers" begin
    r = HTTP.get(BASE; GUI_OPTS...)
    @test r.status == 200
    @test startswith(HTTP.header(r, "Content-Type"), "text/html")
    csp = HTTP.header(r, "Content-Security-Policy")
    @test occursin("default-src 'none'", csp)
    @test occursin("script-src 'self'", csp)
    @test occursin("frame-ancestors 'none'", csp)
    @test !occursin("unsafe-eval", csp)
    @test !occursin("unsafe-inline", csp)
    @test HTTP.header(r, "X-Content-Type-Options") == "nosniff"
    @test HTTP.header(r, "X-Frame-Options") == "DENY"
    html = String(r.body)
    # no inline scripts or inline event handlers
    @test !occursin(r"<script(?![^>]*\bsrc=)"i, html)
    @test !occursin(r"\son[a-z]+\s*="i, html)
    # vendored Plotly pinned with SRI, and the hash matches the shipped file
    m = match(r"plotly-basic\.min\.js\" integrity=\"sha384-([^\"]+)\"", html)
    @test m !== nothing
    js = HTTP.get(BASE * "static/vendor/plotly-basic.min.js"; GUI_OPTS...)
    @test js.status == 200
    @test startswith(HTTP.header(js, "Content-Type"), "text/javascript")
    @test base64encode(sha384(js.body)) == m.captures[1]
    app = HTTP.get(BASE * "static/app.js"; GUI_OPTS...)
    @test app.status == 200
    @test !occursin(r"innerHTML|outerHTML|insertAdjacentHTML|document\.write",
                    String(app.body))
    @test !occursin(r"\beval\(", String(app.body))
    @test HTTP.get(BASE * "static/styles.css"; GUI_OPTS...).status == 200
    @test HTTP.get(BASE * "healthz"; GUI_OPTS...).status == 200
    # only whitelisted files are served; no path traversal
    for p in ("static/../Project.toml", "static/%2e%2e/Project.toml", "static/app.js/..",
              "Project.toml", "static/vendor/README.md", "ext/DrSnowGUIExt/server.jl")
        @test HTTP.get(BASE * p; GUI_OPTS...).status == 404
    end
    @test HTTP.request("PUT", BASE * "api/session"; GUI_OPTS...).status == 405
end

@testset "Host and Origin checks" begin
    r = HTTP.get(BASE, ["Host" => "evil.example:$(srv.port)"]; GUI_OPTS...)
    @test r.status == 400                         # DNS-rebinding protection
    @test HTTP.get(BASE, ["Host" => "localhost:$(srv.port)"]; GUI_OPTS...).status == 200
    r = HTTP.post(BASE * "api/session", ["Origin" => "http://evil.example"]; GUI_OPTS...)
    @test r.status == 403
    r = HTTP.post(BASE * "api/session", ["Sec-Fetch-Site" => "cross-site"]; GUI_OPTS...)
    @test r.status == 403
    r = HTTP.post(BASE * "api/session", ["Origin" => "http://$HOST"]; GUI_OPTS...)
    @test r.status == 200
    cookie = HTTP.header(r, "Set-Cookie")
    @test occursin("HttpOnly", cookie)
    @test occursin("SameSite=Strict", cookie)
    tok = jbody(r).csrf_token
    @test occursin(r"^[0-9a-f]{64}$", tok)
    @test occursin(r"drsnow_sid=[0-9a-f]{64};", cookie)
end

rng = StableRNG(20260927)
panel = gui_panel(rng)
c = gui_session(BASE, HOST)

@testset "authentication and CSRF" begin
    bytes = gui_csv(panel)
    # no session cookie
    r = HTTP.post(BASE * "api/upload", ["Host" => HOST, "X-CSRF-Token" => c.csrf], bytes;
                  GUI_OPTS...)
    @test r.status == 401
    # invalid session
    bad = GuiClient(BASE, HOST, repeat("0", 64), c.csrf)
    @test gui_upload(bad, bytes).status == 401
    @test gui_get(bad, "api/data").status == 401
    # missing / wrong CSRF token
    @test gui_upload(c, bytes; csrf="").status == 403
    @test gui_upload(c, bytes; csrf=repeat("a", 64)).status == 403
    @test gui_get(c, "api/data"; csrf="").status == 403
    # the CSRF token of another session does not work
    other = gui_session(BASE, HOST)
    @test gui_upload(c, bytes; csrf=other.csrf).status == 403
    # cross-origin state-changing request refused even with valid token
    r = gui_upload(c, bytes; extra=["Origin" => "http://evil.example"])
    @test r.status == 403
    @test gui_get(c, "api/data").status == 404    # nothing uploaded yet
end

@testset "upload and preview" begin
    r = gui_upload(c, gui_csv(panel); extra=["X-Filename" => "../../etc/panel<b>.csv"])
    @test r.status == 200
    s = jbody(r)
    @test s.n_rows == nrow(panel)
    @test s.n_cols == ncol(panel)
    @test [col.name for col in s.columns] == names(panel)
    @test s.columns[4].kind == "binary"
    @test length(s.preview) == 20
    @test !occursin("/", s.filename) && !occursin("<", s.filename)
    @test jbody(gui_get(c, "api/data")).n_rows == nrow(panel)
    # TSV with automatic and explicit delimiter
    tsv = gui_csv(panel; delim='\t')
    @test jbody(gui_upload(c, tsv)).n_cols == ncol(panel)
    @test jbody(gui_upload(c, tsv; delim="tab")).n_cols == ncol(panel)
    @test gui_upload(c, tsv; delim="pipe").status == 400
    # limits and malformed input
    @test gui_upload(c, UInt8[]).status == 422
    @test gui_upload(c, Vector{UInt8}("a,b\n")).status == 422         # no data rows
    @test gui_upload(c, UInt8[0x61, 0x2c, 0x62, 0x0a, 0xff, 0x2c, 0x31]).status == 422
    wide = join(("c$j" for j in 1:41), ",") * "\n" * join(("1" for j in 1:41), ",")
    r = gui_upload(c, Vector{UInt8}(wide))
    @test r.status == 422
    @test occursin("columns", jbody(r).error)
    long = "a,b\n" * repeat("1,2\n", 5001)
    r = gui_upload(c, Vector{UInt8}(long))
    @test r.status == 422
    @test occursin("rows", jbody(r).error)
    @test gui_upload(c, gui_csv(panel)).status == 200            # restore the panel
end

@testset "oversize uploads are rejected before being read" begin
    limit = round(Int, GUI_LIMIT_MB * 2^20)
    # Declared size above the limit: 413 without the body ever being sent.
    head = "POST /api/upload HTTP/1.1\r\nHost: $HOST\r\nCookie: drsnow_sid=$(c.sid)\r\n" *
           "X-CSRF-Token: $(c.csrf)\r\nContent-Type: text/csv\r\n" *
           "Content-Length: 1000000000\r\n\r\n"
    t0 = time()
    resp = gui_raw(srv.port, Vector{UInt8}(head * "unit,year\n1,2\n"))
    @test startswith(resp, "HTTP/1.1 413")
    @test occursin("upload limit", resp)
    @test time() - t0 < 10
    # Chunked body without Content-Length: stopped while streaming.
    chunk = repeat("1,2,3,4,5,6\n", 2000)             # 24 kB
    nchunks = cld(limit, length(chunk)) + 2
    body = IOBuffer()
    write(body, "POST /api/upload HTTP/1.1\r\nHost: $HOST\r\n",
          "Cookie: drsnow_sid=$(c.sid)\r\nX-CSRF-Token: $(c.csrf)\r\n",
          "Transfer-Encoding: chunked\r\n\r\n")
    write(body, "b\r\na,b,c,d,e,f\r\n")
    for _ in 1:nchunks
        write(body, string(length(chunk); base=16), "\r\n", chunk, "\r\n")
    end
    write(body, "0\r\n\r\n")
    resp = gui_raw(srv.port, take!(body))
    @test startswith(resp, "HTTP/1.1 413")
    # The session still holds the previous (valid) data.
    @test jbody(gui_get(c, "api/data")).n_rows == nrow(panel)
    # Oversized JSON bodies are rejected too.
    r = gui_post(c, "api/analyze", repeat("x", 300_000);
                 extra=["Content-Type" => "application/json"])
    @test r.status == 413
end

@testset "DiD: TWFE with Goodman-Bacon decomposition" begin
    r, res = gui_analyze(c, "did_twfe", Dict("outcome" => "y", "treatment" => "d",
                                             "unit" => "unit", "time" => "year",
                                             "bacon" => true, "level" => 0.9))
    @test r.status == 200
    ref = did_twfe(panel, :y, :d, :unit, :year; warn_heterogeneity=false)
    t = res.tables[1]
    @test t.columns[1] == "Term"
    @test t.columns[end] == "Upper 90%"
    @test t.rows[1][2] ≈ coef(ref)[1]
    @test t.rows[1][3] ≈ stderror(ref)[1]
    @test length(res.tables) == 3
    @test any(w -> occursin("staggered", w), res.warnings)   # estimator warning surfaced
    @test res.spec.level == 0.9
    @test occursin(r"^[0-9a-f]{32}$", res.id)
    # FirstTreated coding gives the same estimate
    r2, res2 = gui_analyze(c, "did_twfe", Dict("outcome" => "y", "treatment" => "first",
        "treatment_type" => "first_treated", "unit" => "unit", "time" => "year"))
    @test r2.status == 200
    @test res2.tables[1].rows[1][2] ≈ coef(ref)[1]
end

@testset "event study: coefficients, plot data and pre-trend test" begin
    r, res = gui_analyze(c, "event_study", Dict("outcome" => "y", "treatment" => "d",
        "unit" => "unit", "time" => "year", "estimator" => "sun_abraham",
        "max_pre" => 3, "max_post" => 2))
    @test r.status == 200
    p = res.plot
    @test p.kind == "event_study"
    @test length(p.x) == length(p.y) == length(p.lower) == length(p.upper)
    @test all(p.lower[i] <= p.y[i] <= p.upper[i] for i in eachindex(p.y))
    @test -1 in p.reference
    d = res.diagnostics[1]
    @test d.name == "Joint pre-trend test"
    @test 0 <= d.pvalue <= 1
    @test occursin("not evidence", d.verdict) || occursin("rejected", d.verdict)
    @test !isempty(d.note)
end

@testset "Callaway–Sant'Anna with aggregation" begin
    for agg in ("dynamic", "simple", "group")
        r, res = gui_analyze(c, "did_cs", Dict("outcome" => "y", "treatment" => "d",
            "unit" => "unit", "time" => "year", "aggregation" => agg, "biters" => 99,
            "seed" => 1))
        @test r.status == 200
        @test res.title == "Callaway–Sant'Anna (dr, never_treated controls, " *
                           "varying base period)"
        @test length(res.tables) == 2
        @test (res.plot !== nothing) == (agg == "dynamic")
        @test length(res.diagnostics) == 1
    end
end

@testset "regression discontinuity" begin
    rdd = gui_rd(StableRNG(3))
    @test gui_upload(c, gui_csv(rdd)).status == 200
    r, res = gui_analyze(c, "rd", Dict("outcome" => "outcome", "running" => "score",
                                       "cutoff" => 0.0))
    @test r.status == 200
    ref = rd_estimate(rdd, :outcome, :score)
    @test res.tables[1].rows[1][2] ≈ coef(ref)[1]
    @test res.diagnostics[1].name == "RD density discontinuity test"
    p = res.plot
    @test p.kind == "rd"
    @test p.cutoff == 0
    @test all(<(0), p.bins_left.x) && all(>=(0), p.bins_right.x)
    @test !isempty(p.poly_left.x)
end

@testset "IV: 2SLS, weak-IV diagnostics and Anderson–Rubin set" begin
    ivd = gui_iv(StableRNG(4))
    @test gui_upload(c, gui_csv(ivd)).status == 200
    r, res = gui_analyze(c, "iv", Dict("outcome" => "y", "endogenous" => "d",
                                       "instruments" => ["z"], "covariates" => ["w"]))
    @test r.status == 200
    ref = iv_regression(ivd, :y, [:d], [:z]; covariates=[:w])
    @test res.tables[1].rows[1][2] ≈ coef(ref)[1]
    @test res.tables[2].title == "First stage"
    @test any(b -> occursin("Anderson–Rubin", b.title), res.text_blocks)
    @test any(b -> occursin("Olea–Pflueger", b.text), res.text_blocks)
    @test occursin("LATE", res.estimand)
    r, _ = gui_analyze(c, "iv", Dict("outcome" => "y", "endogenous" => "d",
                                     "instruments" => String[]))
    @test r.status == 422
end

@testset "synthetic DiD" begin
    sd = filter(r -> r.unit > 10, panel)          # a single adoption cohort
    @test gui_upload(c, gui_csv(sd)).status == 200
    r, res = gui_analyze(c, "sdid", Dict("outcome" => "y", "treatment" => "d",
        "unit" => "unit", "time" => "year", "replications" => 20, "seed" => 7))
    @test r.status == 200
    @test res.tables[1].rows[1][1] == "ATT"
    p = res.plot
    @test p.kind == "sdid"
    @test length(p.series) == 1
    @test p.series[1].adoption == 2006
    @test length(p.series[1].time) == 8
    r, res = gui_analyze(c, "sdid", Dict("outcome" => "y", "treatment" => "d",
        "unit" => "unit", "time" => "year", "se_method" => "none"))
    @test r.status == 200
    @test res.tables[1].columns == ["Term", "Estimate"]
end

@testset "malicious column names are data, never code" begin
    evil_y = "y) + (println(\"x\")"
    evil_d = "<img src=x onerror=alert(1)>"
    evil_u = "u); DrSnowGuiPwned = 1; ("
    df = DataFrame(evil_y => panel.y, evil_d => panel.d, evil_u => panel.unit,
                   "=cmd|' /C calc'!A0" => panel.year)
    r = gui_upload(c, gui_csv(df))
    @test r.status == 200
    s = jbody(r)
    @test [col.name for col in s.columns] == names(df)   # returned verbatim, as JSON data
    # the raw response is JSON: the markup is inside a JSON string, never HTML
    @test startswith(HTTP.header(r, "Content-Type"), "application/json")
    r, res = gui_analyze(c, "did_twfe", Dict("outcome" => evil_y, "treatment" => evil_d,
        "unit" => evil_u, "time" => "=cmd|' /C calc'!A0"))
    @test r.status == 200
    @test res.tables[1].rows[1][1] == evil_d                # coefficient named by column
    @test !isdefined(Main, :DrSnowGuiPwned)
    # CSV export neutralizes spreadsheet formulas in text cells
    e = gui_get(c, "api/results/$(res.id)?format=csv&table=1")
    @test e.status == 200
    @test startswith(HTTP.header(e, "Content-Type"), "text/csv")
    @test occursin("attachment", HTTP.header(e, "Content-Disposition"))
    # names that are not columns are refused without echoing them
    r, res = gui_analyze(c, "did_twfe", Dict("outcome" => "nonexistent) + (1",
        "treatment" => evil_d, "unit" => evil_u, "time" => "=cmd|' /C calc'!A0"))
    @test r.status == 422
    @test !occursin("nonexistent", res.error)
    # options outside their allow-lists
    r, _ = gui_analyze(c, "did_twfe; rm -rf /", Dict())
    @test r.status == 400
    r, _ = gui_analyze(c, "event_study", Dict("outcome" => evil_y, "treatment" => evil_d,
        "unit" => evil_u, "time" => "=cmd|' /C calc'!A0", "estimator" => "Main.exit"))
    @test r.status == 400
    r, _ = gui_analyze(c, "did_twfe", Dict("outcome" => ["y"]))
    @test r.status == 400
    r, _ = gui_analyze(c, "did_twfe", Dict("outcome" => evil_y, "treatment" => evil_d,
        "unit" => evil_u, "time" => "=cmd|' /C calc'!A0", "level" => 2))
    @test r.status == 422
end

@testset "CSV injection neutralized in exports" begin
    df = DataFrame(y=panel.y, d=panel.d, unit=panel.unit, year=panel.year)
    rename!(df, :d => "=HYPERLINK(\"http://evil\")")
    @test gui_upload(c, gui_csv(df)).status == 200
    r, res = gui_analyze(c, "did_twfe", Dict("outcome" => "y",
        "treatment" => "=HYPERLINK(\"http://evil\")", "unit" => "unit", "time" => "year"))
    @test r.status == 200
    e = gui_get(c, "api/results/$(res.id)?format=csv&table=1")
    txt = String(e.body)
    @test occursin("'=HYPERLINK", txt)
    @test !occursin(r"(^|\n|,)\"?=HYPERLINK", txt)
    j = gui_get(c, "api/results/$(res.id)?format=json")
    @test j.status == 200
    @test JSON3.read(j.body).id == res.id
    @test gui_get(c, "api/results/$(res.id)?format=xml").status == 400
    @test gui_get(c, "api/results/$(res.id)?format=csv&table=99").status == 404
    @test gui_get(c, "api/results/deadbeef?format=json").status == 404
    # results are private to their session
    other = gui_session(BASE, HOST)
    @test gui_get(other, "api/results/$(res.id)?format=json").status == 404
end

@testset "errors are generic and never leak internals" begin
    r = gui_post(c, "api/analyze", "{not json";
                 extra=["Content-Type" => "application/json"])
    @test r.status == 400
    @test jbody(r).error == "The request body is not valid JSON."
    @test gui_post(c, "api/analyze", "[1,2]").status == 400
    # estimator validation errors are reported (message only, no trace)
    @test gui_upload(c, gui_csv(gui_iv(StableRNG(9)))).status == 200
    r, res = gui_analyze(c, "sdid", Dict("outcome" => "y", "treatment" => "z",
                                         "unit" => "d", "time" => "w"))
    @test r.status == 422
    @test !leaks(res.error)
    # internal errors: generic message with a reference, details only in the log
    status, msg, internal = GUIExt._gui_classify_error(
        ErrorException("secret at /home/user/x.jl:12"))
    @test status == 500
    @test internal
    @test !occursin("secret", msg)
    status, msg, internal = GUIExt._gui_classify_error(
        TaskFailedException(Task(() -> nothing)))  # unfinished task: falls to generic
    @test status == 500
    @test GUIExt._gui_classify_error(ArgumentError("bad column"))[1] == 422
    @test_logs (:error, r"reference") match_mode=:any GUIExt._gui_log_internal(
        ErrorException("boom"), backtrace(), "test")
end

@testset "session store: expiry, cleanup, capacity" begin
    cfg = GUIExt._gui_config(; port=0, session_ttl_minutes=1, max_sessions=2)
    store = GUIExt._GuiStore()
    s1 = GUIExt._gui_session_create!(store, cfg; now=0.0)
    s2 = GUIExt._gui_session_create!(store, cfg; now=10.0)
    @test s1.id != s2.id && s1.csrf != s2.csrf
    @test GUIExt._gui_session_get(store, s1.id, cfg.session_ttl; now=30.0) === s1
    # capacity: the least recently used idle session is evicted
    s3 = GUIExt._gui_session_create!(store, cfg; now=40.0)
    @test GUIExt._gui_session_get(store, s2.id, cfg.session_ttl; now=41.0) === nothing
    @test GUIExt._gui_session_get(store, s1.id, cfg.session_ttl; now=41.0) === s1
    # busy sessions are never evicted
    s1.busy = s3.busy = true
    @test GUIExt._gui_session_create!(store, cfg; now=42.0) === nothing
    s1.busy = s3.busy = false
    # idle expiry
    @test GUIExt._gui_cleanup!(store, cfg.session_ttl; now=1000.0) == 2
    @test isempty(store.sessions)
    @test GUIExt._gui_secure_equals("abc", "abc")
    @test !GUIExt._gui_secure_equals("abc", "abd")
    @test !GUIExt._gui_secure_equals("abc", "abcd")
end

@testset "session deletion" begin
    d = gui_session(BASE, HOST)
    r = HTTP.request("DELETE", BASE * "api/session", gui_headers(d); GUI_OPTS...)
    @test r.status == 200
    @test occursin("Max-Age=0", HTTP.header(r, "Set-Cookie"))
    @test gui_get(d, "api/data").status == 401
end

@testset "no response leaks internals" begin
    for r in (HTTP.get(BASE * "nope"; GUI_OPTS...),
              HTTP.post(BASE * "api/analyze", ["Host" => HOST], "x"; GUI_OPTS...),
              gui_post(c, "api/analyze", "{\"design\":\"rd\",\"params\":{}}"))
        @test r.status >= 400
        @test !leaks(String(r.body))
        @test HTTP.header(r, "X-Content-Type-Options") == "nosniff"
    end
end

finally
    @test stop_gui() == true
end

@testset "server stopped" begin
    @test stop_gui() == false
    @test GUIExt._GUI_SERVER[] === nothing
    @test_throws Exception HTTP.get(BASE * "healthz"; retry=false, connect_timeout=5,
                                    readtimeout=5)
    # stub arguments are validated before binding
    @test_throws ArgumentError launch_gui(; port=70000, async=true, open_browser=false)
end
