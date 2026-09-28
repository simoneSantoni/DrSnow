# Launch the DrSnow web GUI.
#
#     julia launch_gui.jl
#
# Uses the deployment environment in app/ (DrSnow from this repository plus HTTP,
# JSON3 and CSV). On first local use the environment is instantiated; in the Docker
# image (DRSNOW_ENV=production) it is already instantiated and precompiled at build
# time, and nothing is installed at runtime.
#
# Environment variables (all optional):
#   DRSNOW_GUI_HOST           interface to bind (default 127.0.0.1; the Docker image
#                             sets 0.0.0.0, which is only reachable through the
#                             container's published port)
#   DRSNOW_GUI_PORT           port (default 8000)
#   DRSNOW_GUI_OPEN_BROWSER   "true"/"false" (default true unless production)
#   DRSNOW_GUI_ALLOWED_HOSTS  comma-separated extra Host names accepted (e.g. a LAN
#                             name when publishing the container beyond localhost)
#   DRSNOW_GUI_MAX_UPLOAD_MB  upload limit in MB (default 50)

const APP = joinpath(@__DIR__, "app")
const PRODUCTION = get(ENV, "DRSNOW_ENV", "") == "production"

if PRODUCTION
    # The image runs `julia --project=app` with a read-only depot: no Pkg calls.
    dirname(something(Base.active_project(), "")) == APP ||
        error("run with `julia --project=app launch_gui.jl` in production")
else
    import Pkg
    Pkg.activate(APP; io=devnull)
    if !isfile(joinpath(APP, "Manifest.toml"))
        @info "Setting up the GUI environment in $APP (first run only)…"
        # Julia 1.10 does not read [sources]; link the repository explicitly.
        VERSION < v"1.11" && Pkg.develop(Pkg.PackageSpec(path=@__DIR__))
        Pkg.instantiate()
    end
end

using DrSnow, HTTP, JSON3, CSV

# Turn Ctrl-C into an InterruptException so launch_gui stops the server cleanly.
Base.exit_on_sigint(false)

env(key, default) = get(ENV, key, default)
hosts = filter(!isempty, strip.(split(env("DRSNOW_GUI_ALLOWED_HOSTS", ""), ",")))

launch_gui(; host=env("DRSNOW_GUI_HOST", "127.0.0.1"),
           port=parse(Int, env("DRSNOW_GUI_PORT", "8000")),
           open_browser=parse(Bool, env("DRSNOW_GUI_OPEN_BROWSER", string(!PRODUCTION))),
           max_upload_mb=parse(Float64, env("DRSNOW_GUI_MAX_UPLOAD_MB", "50")),
           allowed_hosts=isempty(hosts) ? nothing : hosts)
