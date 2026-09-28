# Parametric (continuous-covariate) LATE extrapolation (Angrist & Fernández-Val 2013).
# Reference: test/validation/iv/make_reference_robust_extrap.R (just-identified GMM
# with a numerical-Jacobian sandwich in R).

@testset "Parametric LATE extrapolation" begin
    all_t = [:compliers, :population, :treated, :untreated, :always_takers,
             :never_takers]

    @testset "validation against R (Card: somecol on nearc4, exper + black)" begin
        ex = late_extrapolation(CARD, :lwage, :somecol, :nearc4;
                                covariates=[:exper, :black], targets=all_t)
        n = nrow(CARD)
        @test ex.model === :linear
        for (j, t) in enumerate(coefnames(ex))
            @test coef(ex)[j] ≈ IV_REF_RX[("extrap_linear", "$(t)_coef")] rtol = 1e-9
            # R: HC0 sandwich; DrSnow: influence functions with n/(n − 1)
            @test stderror(ex)[j] * sqrt((n - 1) / n) ≈
                  IV_REF_RX[("extrap_linear", "$(t)_se_hc0")] rtol = 1e-6
        end
        for j in 1:3
            δref = IV_REF_RX[("extrap_linear", "delta_$j")]
            @test ex.cells.late_coef[j] ≈ δref rtol = 1e-9
        end
    end

    @testset "saturated covariates reproduce the cell version exactly" begin
        rng = StableRNG(501)
        df, _ = iv_binary_dgp(rng; n=4000)
        df.g = rand(rng, 1:3, nrow(df))
        df.f = rand(rng, 0:1, nrow(df))
        df.cell = string.(df.g, "_", df.f)
        df.site = rand(rng, 1:40, nrow(df))
        for kw in ((;), (cluster=:site,))
            a = late_extrapolation(df, :y, :d, :z, [:g, :f]; targets=all_t, kw...)
            b = late_extrapolation(df, :y, :d, :z; covariates=[:cell], targets=all_t,
                                   kw...)
            @test coef(b) ≈ coef(a) rtol = 1e-10
            @test stderror(b) ≈ stderror(a) rtol = 1e-7
            @test dof_residual(b) == dof_residual(a)
        end
        # weights
        df.w = 0.5 .+ rand(StableRNG(502), nrow(df))
        a = late_extrapolation(df, :y, :d, :z, [:g, :f]; targets=all_t, weights=:w)
        b = late_extrapolation(df, :y, :d, :z; covariates=[:cell], targets=all_t,
                               weights=:w)
        @test coef(b) ≈ coef(a) rtol = 1e-10
        @test stderror(b) ≈ stderror(a) rtol = 1e-7
    end

    @testset "external target, interface, invariance, errors" begin
        df, _ = iv_binary_dgp(StableRNG(503); n=3000)
        tgt = df[1:200, [:x]]
        ex = late_extrapolation(df, :y, :d, :z; covariates=[:x], target_data=tgt)
        @test coefnames(ex)[end] == "external"
        @test coef(ex)[end] ≈ sum(mean(Matrix(hcat(ones(200), tgt.x)); dims=1) .*
                                  ex.cells.late_coef') rtol = 1e-10
        @test occursin("linear LATE(x)", estimand(ex))
        @test occursin("Parametric model", sprint(show, MIME"text/plain"(), ex))
        @test nrow(ex.weights) == length(coefnames(ex))
        sh = df[shuffle(StableRNG(504), 1:nrow(df)), :]
        ex2 = late_extrapolation(sh, :y, :d, :z; covariates=[:x], target_data=tgt)
        @test coef(ex2) ≈ coef(ex) rtol = 1e-10
        @test stderror(ex2) ≈ stderror(ex) rtol = 1e-8
        @test_throws ArgumentError late_extrapolation(df, :y, :d, :z)
        @test_throws ArgumentError late_extrapolation(df, :y, :d, :z; covariates=[:x],
                                                      targets=[:bogus])
        @test_throws ArgumentError late_extrapolation(df, :y, :d, :z; covariates=[:x],
                                                      target_data=DataFrame(a=[1]))
        # instrument must vary given the covariates
        dz = copy(df)
        dz.xz = dz.z
        @test_throws ArgumentError late_extrapolation(dz, :y, :d, :z; covariates=[:xz])
    end

    @testset "Monte Carlo: coverage with a continuous covariate" begin
        R = mc_reps(1000, 150)
        rng = StableRNG(505)
        hit = zeros(3)
        for _ in 1:R
            # effect 2 + x for everybody (CEI holds); complier share linear in x
            n = 2000
            x = 4 .* (rand(rng, n) .- 0.5)
            z = Float64.(rand(rng, n) .< 0.5)
            u = rand(rng, n)
            pc = 0.45 .+ 0.1 .* x                               # in [0.25, 0.65]
            at = u .< 0.15
            co = .!at .& (u .< 0.15 .+ pc .* 0.85)
            d = Float64.(at .| (co .& (z .== 1)))
            y0 = x .+ randn(rng, n) .+ 0.3 .* at
            y = y0 .+ d .* (2 .+ x)
            df = DataFrame(y=y, d=d, z=z, x=x)
            ex = late_extrapolation(df, :y, :d, :z; covariates=[:x],
                                    targets=[:population, :compliers, :treated])
            ci = confint(ex)
            hit[1] += ci[1, 1] <= 2.0 <= ci[1, 2]                 # ATE = 2 + E[x] = 2
            τc = 2 + sum(pc .* x) / sum(pc)                       # complier weights
            hit[2] += ci[2, 1] <= τc <= ci[2, 2]
            τt = mean(2 .+ x[d .== 1])
            hit[3] += ci[3, 1] <= τt <= ci[3, 2]
        end
        hit ./= R
        for j in 1:3
            @test abs(hit[j] - 0.95) < mc_tol(0.95, R; slack=0.02)
        end
    end
end
