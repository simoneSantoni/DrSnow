# Comparisons with external reference implementations. Fixtures live in
# test/validation/sutva/ (see the generate_*.jl and *_reference.R scripts there).

using FixedEffectModels: reg, @formula, fe

const _SV_VALDIR = joinpath(@__DIR__, "..", "validation", "sutva")

# Minimal CSV reader for the fixtures (header row, comma separated, optional quotes).
function _sv_read_csv(path)
    lines = readlines(path)
    header = Symbol.(strip.(split(lines[1], ','), '"'))
    cols = [String[] for _ in header]
    for l in lines[2:end]
        isempty(strip(l)) && continue
        for (k, v) in enumerate(split(l, ','))
            push!(cols[k], strip(v, '"'))
        end
    end
    df = DataFrame()
    for (h, c) in zip(header, cols)
        v = tryparse.(Float64, c)
        df[!, h] = any(isnothing, v) ? c : Float64.(v)
    end
    return df
end

@testset "Reference implementations" begin
    @testset "Aronow–Samii estimators vs R package `interference`" begin
        e = _sv_read_csv(joinpath(_SV_VALDIR, "as_edges.csv"))
        u = _sv_read_csv(joinpath(_SV_VALDIR, "as_units.csv"))
        draws = reduce(hcat, [parse.(Int, collect(l))
                              for l in eachline(joinpath(_SV_VALDIR, "as_draws.txt"))])
        ids = Int.(u.unit)
        g = NetworkStructure(ids, DataFrame(source=Int.(e.source), target=Int.(e.target)))
        P = exposure_probabilities(g, draws)
        ref = _sv_read_csv(joinpath(_SV_VALDIR, "as_reference.csv"))
        cons = [String(split(c, " - ")[1]) => "control_unexposed" for c in ref.contrast]
        df = DataFrame(unit=ids, z=Int.(u.z), y=u.y)
        ht = exposure_effects(df, :y, :z, g, P; unit=:unit, contrasts=cons,
                              estimator=:horvitz_thompson)
        hj = exposure_effects(df, :y, :z, g, P; unit=:unit, contrasts=cons)
        @test coef(ht) ≈ ref.tau_ht rtol = 1e-10
        @test diag(vcov(ht)) ≈ ref.var_tau_ht rtol = 1e-10
        @test coef(hj) ≈ ref.tau_h rtol = 1e-10
        @test diag(vcov(hj)) ≈ ref.var_tau_h rtol = 1e-10
        cond = _sv_read_csv(joinpath(_SV_VALDIR, "as_conditions.csv"))
        @test exposure_conditions(g, df.z) == cond.condition
        # the same values whatever the row order of the unit table
        ht2 = exposure_effects(df[randperm(StableRNG(1), nrow(df)), :], :y, :z, g, P;
                               unit=:unit, contrasts=cons, estimator=:horvitz_thompson)
        @test coef(ht2) ≈ coef(ht) && vcov(ht2) ≈ vcov(ht)
    end

    @testset "Conley standard errors vs fixest::vcov_conley" begin
        d = _sv_read_csv(joinpath(_SV_VALDIR, "conley_data.csv"))
        d.g = Int.(d.g)
        ref = _sv_read_csv(joinpath(_SV_VALDIR, "conley_reference.csv"))
        d1 = d[d.period .== 1, :]
        # fixest's great-circle distance uses an Earth radius of 6376 km
        cv(c) = ConleyVcov(; lat=:lat, lon=:lon, cutoff=c, earth_radius=6376,
                           fix_psd=false)
        fits = Dict("cs_100" => reg(d1, @formula(y ~ x1 + x2), cv(100)),
                    "cs_fe_250" => reg(d1, @formula(y ~ x1 + x2 + fe(g)), cv(250)),
                    "pool_200" => reg(d, @formula(y ~ x1 + x2 + fe(g)), cv(200)))
        for (spec, m) in fits
            rr = ref[ref.spec .== spec, :]
            names_ = replace.(coefnames(m), "(Intercept)" => "(Intercept)")
            for row in eachrow(rr)
                i = findfirst(==(row.row), names_)
                j = findfirst(==(row.col), names_)
                @test vcov(m)[i, j] ≈ row.vcov rtol = 1e-8
                @test coef(m)[i] ≈ row.coef_row rtol = 1e-8
            end
        end
        # the same regression through a SpatialStructure keyed by unit id
        s = SpatialStructure(d, :unit; lat=:lat, lon=:lon, earth_radius=6376)
        ms = reg(d, @formula(y ~ x1 + x2 + fe(g)),
                 ConleyVcov(s; unit=:unit, cutoff=200, fix_psd=false))
        @test vcov(ms) ≈ vcov(fits["pool_200"])
    end
end
