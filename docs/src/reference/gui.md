# Graphical Interface: API

```@meta
CurrentModule = DrSnow
```

Reference documentation for the functions and types described in [Web interface](../gui.md).

Both functions require the GUI extension, which loads when HTTP.jl, JSON3.jl and CSV.jl
are loaded next to DrSnow. The security model is described under
[`launch_gui`](@ref) and in the [Web interface](../gui.md#Security-model) guide.

## Starting and stopping the server

```@docs
launch_gui
stop_gui
```
