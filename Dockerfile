# DrSnow web GUI image.
#
#   docker build -t drsnow-gui .
#   docker run --rm -p 127.0.0.1:8000:8000 drsnow-gui     # http://localhost:8000
#
# The GUI environment (app/Project.toml: DrSnow from this checkout + HTTP, JSON3,
# CSV) is resolved, instantiated and precompiled at build time; nothing is installed
# at runtime. The server binds 0.0.0.0 inside the container only (DRSNOW_GUI_HOST);
# publish the port on 127.0.0.1 unless you intend to expose it (it has no login).

FROM julia:1.12-bookworm

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --uid 1000 drsnow
ENV JULIA_DEPOT_PATH=/opt/julia-depot \
    JULIA_PKG_PRECOMPILE_AUTO=0 \
    JULIA_NUM_THREADS=auto \
    DRSNOW_ENV=production \
    DRSNOW_GUI_HOST=0.0.0.0 \
    DRSNOW_GUI_PORT=8000 \
    DRSNOW_GUI_OPEN_BROWSER=false
RUN mkdir -p /opt/julia-depot /app && chown drsnow:drsnow /opt/julia-depot /app

USER drsnow
WORKDIR /app

# Package sources only (see .dockerignore); the app environment dev-links them.
COPY --chown=drsnow:drsnow Project.toml launch_gui.jl ./
COPY --chown=drsnow:drsnow src ./src
COPY --chown=drsnow:drsnow ext ./ext
COPY --chown=drsnow:drsnow app/Project.toml ./app/Project.toml

# Resolve and precompile DrSnow with its GUI extension.
RUN julia --project=app -e 'using Pkg; Pkg.instantiate(); Pkg.precompile(); \
        using DrSnow, HTTP, JSON3, CSV; \
        Base.get_extension(DrSnow, :DrSnowGUIExt) === nothing && error("GUI extension not loaded")'

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD curl -fsS http://localhost:8000/healthz || exit 1

CMD ["julia", "--project=app", "launch_gui.jl"]
