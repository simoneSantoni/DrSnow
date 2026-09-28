"""
DGP satisfying conditional effect ignorability: cells (g ∈ 1:3) × (f ∈ 0:1) with
cell-specific effects τ_c shared by all compliance types, cell-specific compliance
shares and a cell-specific instrument probability (instrument valid within cells).
"""
function iv_cells_dgp(rng; n=20_000)
    g = rand(rng, 1:3, n)
    f = rand(rng, 0:1, n)
    τ = [1.0 2.0; 3.0 0.5; -1.0 4.0]                  # τ[g, f + 1]
    pa = [0.1 0.3; 0.2 0.05; 0.3 0.1]
    pn = [0.5 0.2; 0.3 0.25; 0.1 0.4]
    pz = [0.3 0.5; 0.7 0.4; 0.5 0.6]
    u = rand(rng, n)
    idx = CartesianIndex.(g, f .+ 1)
    at = u .< pa[idx]
    nt = u .> 1 .- pn[idx]
    z = Float64.(rand(rng, n) .< pz[idx])
    d = Float64.(at .| (.!at .& .!nt .& (z .== 1)))
    y0 = g .+ f .+ 0.5 .* at .- 0.5 .* nt .+ randn(rng, n)   # levels differ by type
    eff = τ[idx] .+ 0.5 .* randn(rng, n)                       # same mean for all types
    y = y0 .+ d .* eff
    df = DataFrame(y=y, d=d, z=z, g=g, f=f, cell=string.(g, "-", f))
    # population targets (cells are equally likely by design)
    pc = fill(1 / 6, 3, 2)
    cs = 1 .- pa .- pn
    ptr = pa .+ cs .* pz
    tgt = (population=sum(pc .* τ) / sum(pc),
           treated=sum(pc .* ptr .* τ) / sum(pc .* ptr),
           untreated=sum(pc .* (1 .- ptr) .* τ) / sum(pc .* (1 .- ptr)),
           compliers=sum(pc .* cs .* τ) / sum(pc .* cs),
           always_takers=sum(pc .* pa .* τ) / sum(pc .* pa),
           never_takers=sum(pc .* pn .* τ) / sum(pc .* pn), τ=τ)
    return df, tgt
end

@testset "LATE extrapolation (Angrist & Fernández-Val 2013)" begin
    @testset "truth recovery under conditional effect ignorability" begin
        df, tgt = iv_cells_dgp(StableRNG(81); n=60_000)
        all_t = [:compliers, :population, :treated, :untreated, :always_takers,
                 :never_takers]
        ex = late_extrapolation(df, :y, :d, :z, [:g, :f]; targets=all_t)
        @test ex isa CausalEstimate
        @test coefnames(ex) == string.(all_t)
        for (j, t) in enumerate(all_t)
            @test abs(coef(ex)[j] - getfield(tgt, t)) < 4 * stderror(ex)[j]
        end
        @test nrow(ex.cells) == 6
        @test all(abs.(sum.(eachcol(ex.weights[:, 2:end])) .- 1) .< 1e-12)
        s = sprint(show, MIME"text/plain"(), ex)
        @test occursin("conditional effect ignorability", s)
        @test !occursin("CoefTable(", s)
    end

    @testset "compliers target = IPW LATE with saturated cells" begin
        df, _ = iv_cells_dgp(StableRNG(82); n=8000)
        ex = late_extrapolation(df, :y, :d, :z, [:cell]; targets=[:compliers])
        lw = late_ipw(df, :y, :d, :z; covariates=[:cell])
        @test coef(ex)[1] ≈ estimate(lw) rtol = 1e-8
        @test stderror(ex)[1] ≈ stderror(lw)[1] rtol = 1e-4
        # two cell columns give the same cells as the combined column
        ex2 = late_extrapolation(df, :y, :d, :z, [:g, :f]; targets=[:compliers])
        @test coef(ex2)[1] ≈ coef(ex)[1] rtol = 1e-10
    end

    @testset "external target, clustering, invariance" begin
        df, tgt = iv_cells_dgp(StableRNG(83); n=10_000)
        target = DataFrame(g=[1, 1, 3], f=[0, 0, 1])
        ex = late_extrapolation(df, :y, :d, :z, [:g, :f]; targets=Symbol[],
                                target_data=target)
        cl = ex.cells
        i10 = findfirst(==(string((1, 0))), cl.cell)
        i31 = findfirst(==(string((3, 1))), cl.cell)
        @test coef(ex)[1] ≈ (2 * cl.late[i10] + cl.late[i31]) / 3 rtol = 1e-10
        df.site = rand(StableRNG(84), 1:60, nrow(df))
        exc = late_extrapolation(df, :y, :d, :z, [:g, :f]; cluster=:site)
        @test dof_residual(exc) == 59
        perm = randperm(StableRNG(85), nrow(df))
        exp_ = late_extrapolation(df[perm, :], :y, :d, :z, [:g, :f]; cluster=:site)
        @test coef(exp_) ≈ coef(exc) rtol = 1e-10
        @test vcov(exp_) ≈ vcov(exc) rtol = 1e-8
    end

    @testset "input validation" begin
        df, _ = iv_cells_dgp(StableRNG(86); n=3000)
        @test_throws ArgumentError late_extrapolation(df, :y, :d, :z, Symbol[])
        @test_throws ArgumentError late_extrapolation(df, :y, :d, :z, [:g];
                                                      targets=[:everyone])
        @test_throws ArgumentError late_extrapolation(df, :y, :d, :z, [:g];
                                                      target_data=DataFrame(g=[9]))
        df2 = copy(df)
        df2.z[df2.g .== 1] .= 1.0                     # no instrument variation in g = 1
        @test_throws ArgumentError late_extrapolation(df2, :y, :d, :z, [:g])
        df3 = copy(df)
        df3.d[df3.g .== 2] .= 1.0                     # zero first stage in g = 2
        @test_throws ArgumentError late_extrapolation(df3, :y, :d, :z, [:g])
    end

    @testset "Monte Carlo: coverage of reweighted targets" begin
        R = mc_reps(1000, 150)
        rng = StableRNG(87)
        cover = zeros(3)
        _, tgt = iv_cells_dgp(StableRNG(88); n=10)
        for _ in 1:R
            df, _ = iv_cells_dgp(rng; n=4000)
            ex = late_extrapolation(df, :y, :d, :z, [:g, :f])
            ci = confint(ex)
            for (j, t) in enumerate((:compliers, :population, :treated))
                cover[j] += ci[j, 1] <= getfield(tgt, t) <= ci[j, 2]
            end
        end
        cover ./= R
        for j in 1:3
            @test abs(cover[j] - 0.95) < mc_tol(0.95, R; slack=0.015)
        end
    end
end
