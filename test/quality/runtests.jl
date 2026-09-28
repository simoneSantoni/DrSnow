using Aqua

@testset "Aqua" begin
    Aqua.test_all(DrSnow; ambiguities=(recursive=false,), persistent_tasks=false)
end

@testset "No string evaluation in src" begin
    # Formulas must be built with `term`/`fe`; `eval`/`Meta.parse`/`include` at
    # runtime are code-injection and world-age hazards.
    srcdir = joinpath(pkgdir(DrSnow), "src")
    offenders = String[]
    dirs = filter(isdir, [srcdir, joinpath(pkgdir(DrSnow), "ext")])
    for dir in dirs, (root, _, files) in walkdir(dir), f in files
        endswith(f, ".jl") || continue
        path = joinpath(root, f)
        for (i, line) in enumerate(eachline(path))
            code = split(line, '#'; limit=2)[1]
            if occursin(r"\beval\(|Meta\.parse|@eval\b", code) ||
               (occursin(r"\binclude\(", code) && !endswith(root, "src") &&
                !endswith(f, basename(root) * ".jl") && f != "core.jl")
                push!(offenders, "$(relpath(path, pkgdir(DrSnow))):$i")
            end
        end
    end
    @test isempty(offenders)
    isempty(offenders) || @info "Offending lines" offenders
end
