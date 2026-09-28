# Web GUI entry points. The implementation lives in the package extension
# `DrSnowGUIExt` (ext/DrSnowGUIExt/), loaded when HTTP, JSON3 and CSV are all loaded
# next to DrSnow. `using DrSnow` alone never loads any web dependency.

export launch_gui, stop_gui

"""Dispatch token: the GUI extension adds methods for it."""
struct _GuiBackend end

const _GUI_MISSING_MSG =
    "the DrSnow web GUI is provided by a package extension that loads when HTTP, " *
    "JSON3 and CSV are loaded. Install them once with " *
    "`import Pkg; Pkg.add([\"HTTP\", \"JSON3\", \"CSV\"])`, then run " *
    "`using DrSnow, HTTP, JSON3, CSV` before calling launch_gui()."

_gui_launch(::Any; kwargs...) = error("launch_gui: " * _GUI_MISSING_MSG)
_gui_stop(::Any) = error("stop_gui: " * _GUI_MISSING_MSG)

"""
    launch_gui(; host="127.0.0.1", port=8000, async=false, open_browser=true,
               verbose=true, max_upload_mb=50, max_rows=1_000_000, max_cols=500,
               session_ttl_minutes=120, max_sessions=64, max_results=20,
               cleanup_interval=60, allowed_hosts=nothing, max_json_kb=256)
        -> NamedTuple{(:url, :host, :port)}

Start DrSnow's local web interface: upload a CSV or TSV file, choose a research
design, run it, and inspect or export the coefficient tables, diagnostics and charts.

The interface is a convenience layer for exploratory work. It offers two-way
fixed-effects DiD ([`did_twfe`](@ref)), event studies ([`event_study`](@ref)),
Callaway–Sant'Anna DiD ([`did_callaway_santanna`](@ref) with
[`aggregate_att`](@ref) and [`pre_trend_test`](@ref)), regression discontinuity
([`rd_estimate`](@ref), with an RD plot and the density test), IV/2SLS
([`iv_regression`](@ref) with first-stage diagnostics) and synthetic DiD
([`synthetic_did`](@ref)). Every number it shows is computed by these documented
functions; the interface adds no methods of its own and exposes only a subset of
their options. Stochastic procedures use a seed that is shown in, and exported
with, the specification of each result, so a GUI result can be reproduced in code.
For research that will be reported, write the analysis with the Julia API.

The GUI is a package extension that loads only when HTTP.jl, JSON3.jl and CSV.jl are
all loaded next to DrSnow (`using DrSnow, HTTP, JSON3, CSV`); without them this
function throws an error explaining how to enable it. Only one server can run per
Julia session.

# Security model

The server has **no user accounts and no TLS**: anyone who can connect to its port
can use it, over plain HTTP. Its defaults are chosen for a single user on their own
machine.

- **Network exposure.** It binds the loopback interface `127.0.0.1` by default, so it
  is reachable only from this machine. Binding another address (e.g.
  `host="0.0.0.0"` inside a container) exposes an unauthenticated service to every
  machine that can reach that address, and logs a warning (when `verbose`). Do this
  only on a trusted network or behind a reverse proxy that adds TLS and
  authentication.
- **Host-header allow-list.** Requests whose `Host` header is not an allowed name
  are refused with HTTP 400, which blocks DNS-rebinding attacks by web pages open in
  the user's browser. See `allowed_hosts`.
- **Sessions and cross-site request forgery.** Each browser page gets a session
  identified by a 256-bit random token (from the operating system's random-number
  generator) in an `HttpOnly`, `SameSite=Strict` cookie, and a separate 256-bit
  CSRF token. Every API request must carry the session's CSRF token in an
  `X-CSRF-Token` header (compared in constant time); state-changing requests are
  also refused when `Sec-Fetch-Site` is cross-site or the `Origin` header does not
  match the `Host`.
- **No code evaluation.** Nothing sent by the browser is parsed or evaluated as Julia
  code: column names are looked up in the uploaded table and passed to the
  estimators as `Symbol`s, and the analysis name and every option are matched
  against fixed allow-lists.
- **Data handling.** Uploaded data and results are held in memory only, never
  written to disk, in a lock-protected store. Sessions idle for longer than
  `session_ttl_minutes` are deleted by a periodic cleanup task, and
  [`stop_gui`](@ref) deletes everything.
- **Resource limits.** Upload size, table dimensions, JSON request size, the number
  of sessions and of stored results are bounded by the keywords below; oversized
  uploads are rejected (HTTP 413) from the declared `Content-Length` or as soon as a
  streamed body exceeds the limit, without buffering the rest. Each session runs one
  analysis at a time.
- **Browser hardening.** Responses carry a strict Content-Security-Policy (scripts
  only from the server itself, no inline scripts, `frame-ancestors 'none'`),
  `X-Content-Type-Options: nosniff`, `X-Frame-Options: DENY` and
  `Referrer-Policy: no-referrer`; the page inserts data as text only, and its
  plotting library is served locally with Subresource Integrity.
- **Errors and exports.** Input problems return a short validation message;
  unexpected errors return a generic message with a reference code and are logged
  with details on the server only. CSV exports prefix text cells that begin with
  `=`, `+`, `-`, `@`, a tab or a carriage return with `'`, to neutralize
  spreadsheet formula injection.

# Keywords
- `host::AbstractString = "127.0.0.1"`: network interface (address or resolvable
  name) to bind. Anything other than the loopback exposes the server; see above.
- `port::Integer = 8000`: TCP port; `0` lets the operating system pick a free port,
  which is returned in the result.
- `async::Bool = false`: with `false` the call blocks until the server stops
  (Ctrl-C), then shuts it down and deletes all sessions; with `true` it returns
  immediately and the server keeps running in the background until
  [`stop_gui`](@ref).
- `open_browser::Bool = true`: try to open the page in the default browser.
- `verbose::Bool = true`: log the address when the server starts, and warn when it
  is bound to a non-loopback address or the Host check is disabled.
- `max_upload_mb::Real = 50`: maximum size of an uploaded file, in MiB.
- `max_rows::Integer = 1_000_000`, `max_cols::Integer = 500`: maximum number of data
  rows and columns of an uploaded table; larger tables are rejected.
- `session_ttl_minutes::Real = 120`: idle time after which a session, its data and
  its results are deleted.
- `max_sessions::Integer = 64`: maximum number of concurrent sessions. When it is
  reached, a new session evicts the least recently used idle session; if every
  session is running an analysis, the new session is refused (HTTP 503).
- `max_results::Integer = 20`: number of analysis results kept per session for
  display and export; older results are dropped first. Uploading a new file
  deletes the session's results.
- `cleanup_interval::Real = 60`: seconds between passes of the task that deletes
  expired sessions.
- `allowed_hosts = nothing`: host names accepted in the `Host` header. `nothing`
  accepts `localhost`, `127.0.0.1` and `[::1]`, plus `host` itself unless it is a
  wildcard address (`0.0.0.0` or `::`). A vector of names is accepted in
  addition to those (e.g. the name clients use to reach a container or a proxy);
  `["*"]` disables the check, which re-enables DNS rebinding.
- `max_json_kb::Real = 256`: maximum size of a JSON request body (analysis
  options), in KiB; larger requests are rejected with HTTP 413.

All size and count limits must be positive.

# Returns
- `(url, host, port)`: the address of the server (the browser URL, the bound host
  and the actual port). With `async = false` it is returned after the server has
  stopped.

# Examples
```julia
using DrSnow, HTTP, JSON3, CSV
launch_gui()                                  # blocks; open http://127.0.0.1:8000
srv = launch_gui(; port=0, async=true, open_browser=false)
srv.url
stop_gui()
# inside a container, reachable through a published port named "analysis.local":
# launch_gui(; host="0.0.0.0", allowed_hosts=["analysis.local"], open_browser=false)
```
"""
launch_gui(; kwargs...) = _gui_launch(_GuiBackend(); kwargs...)

"""
    stop_gui() -> Bool

Stop the web interface started by [`launch_gui`](@ref), delete every session together
with its uploaded data and results, and stop the session-cleanup task.

Because the interface keeps data in memory only, stopping the server is also the
way to make sure no uploaded data remains in the Julia session. Requires the GUI
extension (`using DrSnow, HTTP, JSON3, CSV`); without it this function throws an
error explaining how to enable it. A blocking server (`async = false`) is stopped
with Ctrl-C instead, which performs the same clean-up.

# Arguments
None.

# Returns
- `Bool`: `true` if a running server was stopped, `false` if none was running.

# Examples
```julia
using DrSnow, HTTP, JSON3, CSV
srv = launch_gui(; port=0, async=true, open_browser=false)
stop_gui()                                    # true
stop_gui()                                    # false: nothing running
```
"""
stop_gui() = _gui_stop(_GuiBackend())
