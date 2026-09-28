# HTTP helpers: responses, security headers, Host/Origin checks, bounded body reads,
# and error classification (generic messages to clients, details to the server log).

"""An error whose message is safe to show to the user."""
struct _GuiUserError <: Exception
    status::Int
    msg::String
end

const _GUI_CSP = join([
    "default-src 'none'",
    "script-src 'self'",
    # Plotly creates one empty <style> element and fills it through the CSSOM; the
    # hash is that of the empty string, so no inline style content is allowed.
    "style-src 'self' 'sha256-47DEQpj8HBSa+/TImW+5JCeuQeRkm5NMpJWZG3hSuFU='",
    "img-src 'self' data: blob:",
    "font-src 'self'",
    "connect-src 'self'",
    "base-uri 'none'",
    "form-action 'none'",
    "frame-ancestors 'none'",
], "; ")

const _GUI_SECURITY_HEADERS = (
    "Content-Security-Policy" => _GUI_CSP,
    "X-Content-Type-Options" => "nosniff",
    "X-Frame-Options" => "DENY",
    "Referrer-Policy" => "no-referrer",
    "Cross-Origin-Opener-Policy" => "same-origin",
    "Cross-Origin-Resource-Policy" => "same-origin",
    "Permissions-Policy" => "camera=(), microphone=(), geolocation=(), payment=()",
)

const _GUI_SESSION_COOKIE = "drsnow_sid"

function _gui_respond(http::HTTP.Stream, status::Integer, body, mime::AbstractString;
                      headers=(), close_connection::Bool=false)
    bytes = body isa AbstractVector{UInt8} ? body : Vector{UInt8}(codeunits(string(body)))
    HTTP.setstatus(http, status)
    for h in _GUI_SECURITY_HEADERS
        HTTP.setheader(http, h)
    end
    HTTP.setheader(http, "Content-Type" => mime)
    HTTP.setheader(http, "Content-Length" => string(length(bytes)))
    for h in headers
        HTTP.setheader(http, h)
    end
    close_connection && HTTP.setheader(http, "Connection" => "close")
    HTTP.startwrite(http)
    write(http, bytes)
    return nothing
end

_gui_json(http::HTTP.Stream, status::Integer, obj; headers=(), kw...) =
    _gui_respond(http, status, JSON3.write(obj), "application/json; charset=utf-8";
                 headers=(("Cache-Control" => "no-store"), headers...), kw...)

_gui_error(http::HTTP.Stream, status::Integer, msg::AbstractString; kw...) =
    _gui_json(http, status, Dict("error" => msg); kw...)

const _GUI_LINGER_BYTES = 1 << 20
const _GUI_LINGER_SECONDS = 1.0
const _GUI_DRAIN_BYTES = 1 << 20
const _GUI_DRAIN_SECONDS = 5.0

"""Read and discard the rest of a request body of known, small length (nothing is
kept). Returns `false` (connection closed) if the client is too slow."""
function _gui_drain!(http::HTTP.Stream)
    io = http.stream.io
    timer = Timer(_ -> close(io), _GUI_DRAIN_SECONDS)
    try
        while !eof(http)
            readavailable(http)
        end
        return isopen(io)
    catch
        return false
    finally
        close(timer)
    end
end

"""Send an error response to a request whose body may not have been read.

- Body fully read (or none): plain response, connection kept alive.
- Unread body of declared length up to `_GUI_DRAIN_BYTES`: the body is discarded
  (never stored), then a plain response is sent.
- Otherwise (oversized or chunked): respond with `Connection: close`, shut down the
  write side and discard at most `_GUI_LINGER_BYTES` for up to
  `_GUI_LINGER_SECONDS` ("lingering close", so the client can read the response
  instead of getting a connection reset), then close. Nothing is buffered."""
function _gui_reject_unread(http::HTTP.Stream, status::Integer, msg::AbstractString)
    http.ntoread > 0 || return _gui_error(http, status, msg)
    if !http.readchunked && http.ntoread <= _GUI_DRAIN_BYTES
        _gui_drain!(http) && return _gui_error(http, status, msg)
        return nothing
    end
    _gui_error(http, status, msg; close_connection=true)
    # Mark the request as consumed so HTTP.jl does not try to read the body itself.
    http.ntoread = 0
    conn = http.stream
    io = conn.io
    try
        flush(io)
        closewrite(io)            # half-close: the client sees the end of the response
        timer = Timer(_ -> close(io), _GUI_LINGER_SECONDS)
        try
            n = 0
            while n < _GUI_LINGER_BYTES && isopen(io) && !eof(io)
                n += length(readavailable(io))
            end
        finally
            close(timer)
        end
    catch
    end
    try
        close(conn)
    catch
    end
    return nothing
end

# --- Host / Origin ------------------------------------------------------------------

"""Host name (lower case, without port; IPv6 in brackets) of a `Host` header value."""
function _gui_hostname(hostport::AbstractString)
    h = lowercase(strip(hostport))
    if startswith(h, "[")
        i = findfirst(']', h)
        return i === nothing ? h : h[1:i]
    end
    i = findlast(':', h)
    return i === nothing ? h : h[1:(i - 1)]
end

function _gui_host_ok(cfg::_GuiConfig, req::HTTP.Request)
    host = HTTP.header(req, "Host", "")
    isempty(host) && return false
    cfg.allowed_hosts === nothing && return true
    return _gui_hostname(host) in cfg.allowed_hosts
end

"""Same-origin check for state-changing requests: a browser always sends `Origin` on
cross-origin POST/DELETE; when present it must match the `Host` the request was sent
to. `Sec-Fetch-Site: cross-site` is rejected as well."""
function _gui_origin_ok(req::HTTP.Request)
    site = lowercase(HTTP.header(req, "Sec-Fetch-Site", ""))
    site in ("", "same-origin", "none") || return false
    origin = HTTP.header(req, "Origin", "")
    isempty(origin) && return true
    host = lowercase(HTTP.header(req, "Host", ""))
    isempty(host) && return false
    o = lowercase(origin)
    return o == "http://" * host || o == "https://" * host
end

function _gui_cookie(req::HTTP.Request, name::AbstractString)
    for c in HTTP.Cookies.cookies(req)
        c.name == name && return c.value
    end
    return ""
end

# --- Bounded body reads ----------------------------------------------------------------

"""Read the request body, refusing to buffer more than `limit` bytes. Returns the
bytes, or `nothing` when the declared or streamed size exceeds `limit`. Throws
`_GuiUserError` for a malformed Content-Length."""
function _gui_read_body(http::HTTP.Stream, limit::Integer)
    cl = HTTP.header(http.message, "Content-Length", "")
    if !isempty(cl)
        n = tryparse(Int, strip(cl))
        (n === nothing || n < 0) && throw(_GuiUserError(400, "Invalid Content-Length."))
        n > limit && return nothing
    end
    buf = IOBuffer()
    total = 0
    while !eof(http)
        chunk = readavailable(http)
        total += length(chunk)
        total > limit && return nothing
        write(buf, chunk)
    end
    return take!(buf)
end

function _gui_read_json(http::HTTP.Stream, cfg::_GuiConfig)
    body = _gui_read_body(http, cfg.max_json_bytes)
    body === nothing && return nothing
    obj = try
        JSON3.read(body)
    catch
        throw(_GuiUserError(400, "The request body is not valid JSON."))
    end
    obj isa JSON3.Object || throw(_GuiUserError(400, "Expected a JSON object."))
    return obj
end

# --- Error classification ---------------------------------------------------------------

_gui_unwrap(e) = e
_gui_unwrap(e::TaskFailedException) = _gui_unwrap(e.task.exception)
_gui_unwrap(e::CompositeException) =
    isempty(e.exceptions) ? e : _gui_unwrap(first(e.exceptions))

"""Truncate a validation message and strip control characters."""
function _gui_clean_message(msg::AbstractString; maxlen::Int=600)
    s = replace(String(msg), r"[\x00-\x08\x0b-\x1f\x7f]" => " ")
    return length(s) > maxlen ? first(s, maxlen) * "…" : s
end

"""Map an exception to `(status, message)` for the client and whether it must be
logged as an internal error. Only `ArgumentError`s raised by the estimators'
input validation (which describe the user's data and options) and our own
`_GuiUserError`s are shown to the user; everything else gets a generic message."""
function _gui_classify_error(err)
    e = _gui_unwrap(err)
    e isa _GuiUserError && return (e.status, e.msg, false)
    e isa ArgumentError &&
        return (422, "The analysis could not be run: " * _gui_clean_message(e.msg), false)
    return (500, "The analysis failed because of an internal error. Details were " *
                 "written to the server log.", true)
end

"""Log an unexpected error with its backtrace under a random reference id and return
the generic message (with the reference) for the client."""
function _gui_log_internal(err, bt, context::AbstractString)
    ref = _gui_token(6)
    @error "DrSnow GUI internal error (reference $ref) in $context" exception =
        (err, bt)
    return ref
end
