# Server configuration and static assets.

"""A static file served from memory (loaded once at launch)."""
struct _GuiAsset
    body::Vector{UInt8}
    mime::String
end

struct _GuiConfig
    host::String
    port::Int
    max_upload_bytes::Int
    max_json_bytes::Int
    max_rows::Int
    max_cols::Int
    session_ttl::Float64            # seconds of inactivity before a session expires
    max_sessions::Int
    max_results::Int                # results kept per session (oldest dropped)
    cleanup_interval::Float64       # seconds between cleanup passes
    allowed_hosts::Union{Nothing,Set{String}}   # `nothing` = no Host check
    assets::Dict{String,_GuiAsset}  # request path => asset
end

const _GUI_STATIC_DIR = joinpath(@__DIR__, "static")

# Request path => (file relative to static/, MIME type). Only these files are served;
# request paths are never joined onto the filesystem.
const _GUI_STATIC_FILES = (
    "/" => ("index.html", "text/html; charset=utf-8"),
    "/static/app.js" => ("app.js", "text/javascript; charset=utf-8"),
    "/static/styles.css" => ("styles.css", "text/css; charset=utf-8"),
    "/static/vendor/plotly-basic.min.js" =>
        ("vendor/plotly-basic.min.js", "text/javascript; charset=utf-8"),
)

function _gui_load_assets()
    assets = Dict{String,_GuiAsset}()
    for (route, (file, mime)) in _GUI_STATIC_FILES
        path = joinpath(_GUI_STATIC_DIR, file)
        isfile(path) || error("DrSnow GUI asset missing: $path")
        assets[route] = _GuiAsset(read(path), mime)
    end
    assets["/index.html"] = assets["/"]
    return assets
end

const _GUI_LOOPBACK_NAMES = ("localhost", "127.0.0.1", "[::1]", "::1")

function _gui_allowed_hosts(host::AbstractString, allowed)
    if allowed !== nothing
        names = String[lowercase(strip(string(a))) for a in allowed]
        "*" in names && return nothing
    else
        names = String[]
    end
    set = Set{String}(_GUI_LOOPBACK_NAMES)
    union!(set, names)
    h = lowercase(host)
    h in ("0.0.0.0", "::", "[::]") || push!(set, occursin(':', h) ? "[" * h * "]" : h)
    return set
end

function _gui_config(; host::AbstractString="127.0.0.1", port::Integer=8000,
                     max_upload_mb::Real=50, max_rows::Integer=1_000_000,
                     max_cols::Integer=500, session_ttl_minutes::Real=120,
                     max_sessions::Integer=64, max_results::Integer=20,
                     cleanup_interval::Real=60, allowed_hosts=nothing,
                     max_json_kb::Real=256)
    0 <= port <= 65535 || throw(ArgumentError("launch_gui: port must be in 0:65535"))
    max_upload_mb > 0 || throw(ArgumentError("launch_gui: max_upload_mb must be positive"))
    max_rows > 0 || throw(ArgumentError("launch_gui: max_rows must be positive"))
    max_cols > 0 || throw(ArgumentError("launch_gui: max_cols must be positive"))
    session_ttl_minutes > 0 ||
        throw(ArgumentError("launch_gui: session_ttl_minutes must be positive"))
    max_sessions > 0 || throw(ArgumentError("launch_gui: max_sessions must be positive"))
    cleanup_interval > 0 ||
        throw(ArgumentError("launch_gui: cleanup_interval must be positive"))
    return _GuiConfig(String(host), Int(port), round(Int, max_upload_mb * 2^20),
                      round(Int, max_json_kb * 1024), Int(max_rows), Int(max_cols),
                      60.0 * session_ttl_minutes, Int(max_sessions), Int(max_results),
                      Float64(cleanup_interval), _gui_allowed_hosts(host, allowed_hosts),
                      _gui_load_assets())
end
