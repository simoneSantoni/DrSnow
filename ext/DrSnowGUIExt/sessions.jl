# Per-session state in a lock-protected store with idle expiry.

mutable struct _GuiSession
    const id::String
    const csrf::String
    const created::Float64
    last_seen::Float64
    data::Union{Nothing,DataFrame}
    filename::String
    results::Vector{Pair{String,Dict{String,Any}}}   # oldest first
    busy::Bool
end

struct _GuiStore
    sessions::Dict{String,_GuiSession}
    lock::ReentrantLock
end
_GuiStore() = _GuiStore(Dict{String,_GuiSession}(), ReentrantLock())

"""256-bit token from the operating system's CSPRNG, hex encoded."""
_gui_token(nbytes::Integer=32) = bytes2hex(rand(Random.RandomDevice(), UInt8, nbytes))

"""Constant-time string comparison (for CSRF tokens)."""
function _gui_secure_equals(a::AbstractString, b::AbstractString)
    ca, cb = codeunits(a), codeunits(b)
    length(ca) == length(cb) || return false
    acc = 0x00
    @inbounds for i in eachindex(ca, cb)
        acc |= ca[i] ⊻ cb[i]
    end
    return acc == 0x00
end

_gui_expired(s::_GuiSession, ttl::Real, now::Real) = now - s.last_seen > ttl

"""Create a session, evicting expired ones and, at capacity, the least recently used
idle session. Returns `nothing` when every slot holds a busy session."""
function _gui_session_create!(store::_GuiStore, cfg::_GuiConfig; now::Real=time())
    lock(store.lock) do
        _gui_cleanup_locked!(store, cfg.session_ttl, now)
        if length(store.sessions) >= cfg.max_sessions
            idle = [s for s in values(store.sessions) if !s.busy]
            isempty(idle) && return nothing
            oldest = idle[argmin([s.last_seen for s in idle])]
            delete!(store.sessions, oldest.id)
        end
        id = _gui_token()
        while haskey(store.sessions, id)
            id = _gui_token()
        end
        s = _GuiSession(id, _gui_token(), now, now, nothing, "",
                        Pair{String,Dict{String,Any}}[], false)
        store.sessions[id] = s
        return s
    end
end

"""Look up a live session and refresh its idle timer; expired sessions are removed."""
function _gui_session_get(store::_GuiStore, id::AbstractString, ttl::Real;
                          now::Real=time())
    isempty(id) && return nothing
    lock(store.lock) do
        s = get(store.sessions, id, nothing)
        s === nothing && return nothing
        if _gui_expired(s, ttl, now) && !s.busy
            delete!(store.sessions, id)
            return nothing
        end
        s.last_seen = now
        return s
    end
end

function _gui_session_delete!(store::_GuiStore, id::AbstractString)
    lock(store.lock) do
        return pop!(store.sessions, id, nothing) !== nothing
    end
end

function _gui_cleanup_locked!(store::_GuiStore, ttl::Real, now::Real)
    n = 0
    for (id, s) in collect(store.sessions)
        if _gui_expired(s, ttl, now) && !s.busy
            delete!(store.sessions, id)
            n += 1
        end
    end
    return n
end

"""Remove expired sessions (and their data). Returns the number removed."""
_gui_cleanup!(store::_GuiStore, ttl::Real; now::Real=time()) =
    lock(() -> _gui_cleanup_locked!(store, ttl, now), store.lock)

function _gui_clear!(store::_GuiStore)
    lock(() -> empty!(store.sessions), store.lock)
    return nothing
end

"""Mark the session busy; returns `false` if an analysis is already running."""
function _gui_try_acquire!(store::_GuiStore, s::_GuiSession)
    lock(store.lock) do
        s.busy && return false
        s.busy = true
        return true
    end
end

function _gui_release!(store::_GuiStore, s::_GuiSession)
    lock(store.lock) do
        s.busy = false
        s.last_seen = time()
    end
    return nothing
end

function _gui_set_data!(store::_GuiStore, s::_GuiSession, df::DataFrame,
                        filename::AbstractString)
    lock(store.lock) do
        s.data = df
        s.filename = String(filename)
        empty!(s.results)
    end
    return nothing
end

_gui_get_data(store::_GuiStore, s::_GuiSession) = lock(() -> s.data, store.lock)

function _gui_add_result!(store::_GuiStore, s::_GuiSession, id::String,
                          result::Dict{String,Any}, max_results::Int)
    lock(store.lock) do
        push!(s.results, id => result)
        while length(s.results) > max_results
            popfirst!(s.results)
        end
    end
    return nothing
end

function _gui_get_result(store::_GuiStore, s::_GuiSession, id::AbstractString)
    lock(store.lock) do
        for (k, v) in s.results
            k == id && return v
        end
        return nothing
    end
end
