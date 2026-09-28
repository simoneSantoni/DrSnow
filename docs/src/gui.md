# Web interface

```@meta
CurrentModule = DrSnow
```

DrSnow ships an optional browser interface for exploratory work: upload a CSV/TSV file,
pick a research design, run it, and read the coefficient tables, diagnostics and
charts. Everything it shows is computed by the same functions documented in the rest
of this manual; the interface adds no methods of its own. For reproducible research,
use the Julia API (the "Full text output" and "Specification" panels show what was run).

## Installation and launch

The interface is a [package extension](https://pkgdocs.julialang.org/v1/creating-packages/#Conditional-loading-of-code-in-packages-(Extensions))
that loads only when **HTTP.jl, JSON3.jl and CSV.jl are all loaded** next to DrSnow.
`using DrSnow` alone never loads any web dependency.

```julia
import Pkg
Pkg.add(["HTTP", "JSON3", "CSV"])       # once

using DrSnow, HTTP, JSON3, CSV
launch_gui()                             # opens http://127.0.0.1:8000 and blocks
```

Stop it with Ctrl-C. To keep using the REPL while it runs:

```julia
srv = launch_gui(; async=true, port=0)   # port 0 = pick a free port
srv.url
stop_gui()
```

From a clone of the repository, `julia launch_gui.jl` sets up and uses the
deployment environment in `app/` (DrSnow from the checkout plus HTTP, JSON3 and CSV).
It reads `DRSNOW_GUI_HOST`, `DRSNOW_GUI_PORT`, `DRSNOW_GUI_OPEN_BROWSER`,
`DRSNOW_GUI_ALLOWED_HOSTS` and `DRSNOW_GUI_MAX_UPLOAD_MB`.

Long analyses (bootstrap, placebo standard errors) run on a worker thread; start
Julia with several threads (`julia -t auto`) so the interface stays responsive.

## Supported analyses

| Design | Functions used | Output |
|---|---|---|
| DiD, two-way fixed effects | [`did_twfe`](@ref), optionally [`bacon_decomposition`](@ref) | coefficient table, heterogeneity warnings, Goodman-Bacon comparisons |
| Event study | [`event_study`](@ref) (automatic, TWFE, Sun–Abraham, imputation, Callaway–Sant'Anna), [`pre_trend_test`](@ref) | event-time coefficients with pointwise intervals (chart and table), joint pre-trend test |
| Staggered DiD | [`did_callaway_santanna`](@ref), [`aggregate_att`](@ref) (overall, by cohort, event time, calendar time), [`pre_trend_test`](@ref) | aggregated ATT, all ATT(g, t), event-time chart |
| Regression discontinuity | [`rd_estimate`](@ref), [`rd_inference_table`](@ref), [`rd_density_test`](@ref), [`rd_plot_data`](@ref) | robust bias-corrected estimate, density test, binned scatter with global polynomial fit |
| Instrumental variables | [`iv_regression`](@ref), first-stage diagnostics, [`weak_iv_confidence_set`](@ref), [`weak_iv_test`](@ref) | 2SLS table, first-stage F statistics (Olea–Pflueger effective F), Anderson–Rubin confidence set |
| Synthetic DiD | [`synthetic_did`](@ref), [`synth_weights`](@ref), [`synth_time_weights`](@ref), [`synth_gaps`](@ref) | ATT, unit and time weights, treated and synthetic trajectories |

Warnings emitted by the estimators (for example the TWFE warning for staggered
adoption) are shown with the results. Diagnostics use the package's wording: a
non-rejection is reported as a non-rejection and is never presented as evidence that
an identifying assumption holds. Every table can be downloaded as CSV and every result
as JSON. Stochastic procedures use a seed shown in the specification, so results are
reproducible.

## Security model

The interface is meant to run on your own machine. It has **no user accounts**:
anyone who can reach the port can use it. Its defaults are chosen accordingly.

- **Network exposure.** The server binds `127.0.0.1` (this machine only). Requests
  whose `Host` header is not `localhost`, `127.0.0.1` or `[::1]` (or the bound
  address, or a name passed in `allowed_hosts`) are refused, which blocks DNS
  rebinding attacks from web pages you visit. Binding another address logs a warning.
- **No code evaluation.** Nothing sent by the browser is ever parsed or evaluated as
  Julia code. Column names are looked up in the uploaded table and passed to the
  estimators as `Symbol`s (the estimators build formulas with `StatsModels.term`);
  the analysis name and every option are matched against fixed lists. Column names
  such as `y) + (run(...))` or `<img src=x onerror=...>` are treated as plain text.
- **Sessions.** Each browser tab gets a session identified by a 256-bit random
  token in an `HttpOnly`, `SameSite=Strict` cookie. Uploaded data and results live
  in memory only, in a lock-protected store; sessions idle for longer than
  `session_ttl_minutes` (default 120) are deleted by a periodic cleanup task, and
  everything is deleted by [`stop_gui`](@ref).
- **Cross-site request forgery.** Every API call must carry the session's own
  random CSRF token in an `X-CSRF-Token` header (which other sites cannot read or
  set); state-changing requests are also refused when the `Origin` header is from
  another site or `Sec-Fetch-Site` is `cross-site`.
- **Resource limits.** Uploads above `max_upload_mb` (default 50) are rejected with
  HTTP 413 from the declared `Content-Length`, or as soon as a streamed body exceeds
  the limit, without buffering the rest. Tables are limited to `max_rows` rows and
  `max_cols` columns; JSON requests to 256 kB; one analysis runs at a time per
  session and at most 64 sessions are kept.
- **Browser hardening.** Responses carry a strict Content-Security-Policy (scripts
  only from the server itself, no inline scripts, no `eval`, `frame-ancestors
  'none'`), `X-Content-Type-Options: nosniff`, `X-Frame-Options: DENY` and
  `Referrer-Policy: no-referrer`. The page builds its content with `textContent`
  only, so data can never become markup. Plotly is vendored (no CDN) and loaded with
  Subresource Integrity.
- **Errors.** Input problems return a short message (for example the estimator's
  validation message). Unexpected errors return a generic message with a reference
  code; the details and stack trace go to the server log only.
- **Exports.** CSV exports prefix text cells that start with `=`, `+`, `-` or `@`
  with `'` to neutralize spreadsheet formula injection.

To make the interface reachable from other machines, pass `host="0.0.0.0"` and list
the names clients will use in `allowed_hosts`. Do this only on a trusted network, or
behind a reverse proxy that adds TLS and authentication.

## Docker

The repository contains a `Dockerfile` and `docker-compose.yml`. The image resolves
and precompiles the `app/` environment at build time, runs as an unprivileged user,
and binds `0.0.0.0` only inside the container (`DRSNOW_GUI_HOST`). The compose file
publishes the port on the host's loopback interface and runs the container with a
read-only filesystem and no capabilities.

```bash
docker compose up -d --build        # or: scripts/docker_quickstart.sh
# open http://localhost:8000
docker compose down
```

Without compose:

```bash
docker build -t drsnow-gui .
docker run --rm -p 127.0.0.1:8000:8000 drsnow-gui
```

The container's health check requests `/healthz` with `curl`.

The functions and types described on this page are documented in the [API reference](reference/gui.md).
