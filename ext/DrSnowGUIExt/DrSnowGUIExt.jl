# Web GUI for DrSnow, served with HTTP.jl. Loaded when HTTP, JSON3 and CSV are all
# loaded next to DrSnow (see `launch_gui`).
#
# Security model (see docs/src/gui.md):
# - binds 127.0.0.1 by default; `Host` header allow-list against DNS rebinding;
# - no user text is ever evaluated: column names are looked up in the uploaded table
#   and passed to estimators as `Symbol`s; analysis names and options are matched
#   against fixed allow-lists;
# - uploads are limited by size (checked from Content-Length and while streaming),
#   rows and columns;
# - per-session state in a lock-protected store with idle expiry and a periodic
#   cleanup task; session ids and CSRF tokens are 256-bit random values;
# - every API call needs the session cookie (HttpOnly, SameSite=Strict) and the
#   session's CSRF token in `X-CSRF-Token`; state-changing requests also need a
#   same-origin `Origin` (when sent);
# - strict CSP, nosniff, frame-ancestors 'none'; the frontend builds the DOM with
#   textContent only; Plotly is vendored and pinned with SRI;
# - error responses are generic; details go to the server log only.

module DrSnowGUIExt

using DrSnow
using DrSnow: CausalEstimate, DiagnosticTest, EventStudyEstimate, FirstTreated,
              method_name, estimand
using DataFrames
using Printf
using Random
using Statistics
using StatsAPI
using StatsBase
using HTTP
using JSON3
using CSV

const _GUI_LOG = Base.CoreLogging
const Sockets = HTTP.Sockets

include("config.jl")
include("sessions.jl")
include("http.jl")
include("upload.jl")
include("params.jl")
include("results.jl")
include("analyses.jl")
include("routes.jl")
include("server.jl")

end # module DrSnowGUIExt
