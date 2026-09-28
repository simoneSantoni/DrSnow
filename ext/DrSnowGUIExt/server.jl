# Server lifecycle: launch_gui / stop_gui.

mutable struct _GuiServer
    state::_GuiState
    server::HTTP.Server
    cleanup::Timer
    url::String
    host::String
    port::Int
end

const _GUI_SERVER = Ref{Union{Nothing,_GuiServer}}(nothing)
const _GUI_SERVER_LOCK = ReentrantLock()

function _gui_bind_address(host::AbstractString)
    h = startswith(host, "[") && endswith(host, "]") ? host[2:(end - 1)] : host
    try
        return parse(Sockets.IPAddr, h)
    catch
    end
    try
        return Sockets.getaddrinfo(host)
    catch
        throw(ArgumentError("launch_gui: cannot resolve host $(repr(host))"))
    end
end

function _gui_browser_url(host::AbstractString, port::Integer)
    h = host in ("0.0.0.0", "::", "[::]") ? "127.0.0.1" : host
    occursin(':', h) && !startswith(h, "[") && (h = "[" * h * "]")
    return "http://$h:$port/"
end

function _gui_open_browser(url::AbstractString)
    cmd = Sys.isapple() ? `open $url` :
          Sys.iswindows() ? `cmd /c start "" $url` : `xdg-open $url`
    try
        run(pipeline(cmd; stdout=devnull, stderr=devnull); wait=false)
    catch
        @info "DrSnow GUI: open $url in your browser."
    end
    return nothing
end

function DrSnow._gui_launch(::DrSnow._GuiBackend; host::AbstractString="127.0.0.1",
                            port::Integer=8000, async::Bool=false,
                            open_browser::Bool=true, verbose::Bool=true, kwargs...)
    cfg = _gui_config(; host=host, port=port, kwargs...)
    srv = lock(_GUI_SERVER_LOCK) do
        cur = _GUI_SERVER[]
        cur === nothing || error("launch_gui: a DrSnow GUI server is already running " *
                                 "at $(cur.url); call stop_gui() first")
        ip = _gui_bind_address(cfg.host)
        tcp = Sockets.listen(ip, cfg.port)
        actual = Int(Sockets.getsockname(tcp)[2])
        state = _GuiState(cfg, _GuiStore())
        server = HTTP.listen!(http -> _gui_handle(state, http), cfg.host, actual;
                              server=tcp, verbose=-1, readtimeout=0,
                              max_connections=256)
        cleanup = Timer(cfg.cleanup_interval; interval=cfg.cleanup_interval) do _
            try
                _gui_cleanup!(state.store, cfg.session_ttl)
            catch err
                @error "DrSnow GUI: session cleanup failed" exception = err
            end
        end
        s = _GuiServer(state, server, cleanup, _gui_browser_url(cfg.host, actual),
                       cfg.host, actual)
        _GUI_SERVER[] = s
        s
    end
    info = (url=srv.url, host=srv.host, port=srv.port)
    if verbose
        @info "DrSnow GUI listening on $(srv.url)" * (cfg.allowed_hosts === nothing ?
              " (Host header check disabled)" : "")
        cfg.host in ("127.0.0.1", "localhost", "::1") ||
            @warn "DrSnow GUI is bound to $(cfg.host): it has no login and is " *
                  "reachable from other machines that can connect to this address " *
                  "(in a container: through the published port)."
    end
    open_browser && _gui_open_browser(srv.url)
    async && return info
    try
        wait(srv.server)
    catch err
        err isa InterruptException || rethrow()
    finally
        _gui_stop_server(srv)
    end
    return info
end

function _gui_stop_server(srv::_GuiServer)
    removed = lock(_GUI_SERVER_LOCK) do
        _GUI_SERVER[] === srv || return false
        _GUI_SERVER[] = nothing
        return true
    end
    close(srv.cleanup)
    try
        HTTP.forceclose(srv.server)
    catch
    end
    _gui_clear!(srv.state.store)
    return removed
end

function DrSnow._gui_stop(::DrSnow._GuiBackend)
    srv = lock(() -> _GUI_SERVER[], _GUI_SERVER_LOCK)
    srv === nothing && return false
    return _gui_stop_server(srv)
end
