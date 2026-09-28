@testset "Two-way fixed effects DiD" begin
    function sim_2x2(rng; N=100, T=6, T0=3, att=3.0, share=0.5, sigma=1.0)
        nt = round(Int, N * share)
        df = sim_staggered(rng; N=N, T=T, effect=(g, e) -> att, sigma=sigma,
                           assign=vcat(fill(T0 + 1, nt), zeros(Int, N - nt)))
        return df
    end

    @testset "truth recovery and reference (FixedEffectModels)" begin
        rng = StableRNG(101)
        df = sim_2x2(rng; N=400)
        r = did_twfe(df, :y, :d, :unit, :time)
        @test r isa DiDEstimate
        @test abs(coef(r)[1] - 3.0) < 4 * stderror(r)[1]
        m = DrSnow.FixedEffectModels.reg(df,
            make_formula(:y, [:d]; fe=[:unit, :time]), Vcov.cluster(:unit))
        @test coef(r)[1] ≈ coef(m)[1]
        @test stderror(r)[1] ≈ stderror(m)[1]
        @test vcov(r) ≈ vcov(m)
        # unit counts: ever-treated vs never-treated units
        tu = unique(df.unit[df.g .> 0])
        @test r.n_treated == length(tu)
        @test r.n_control == length(unique(df.unit)) - length(tu)
        @test r.n_periods == 6
        # CI uses t(G - 1) and `level`
        G = length(unique(df.unit))
        @test dof_residual(r) == G - 1
        c = critical_value(0.9, G - 1)
        b, s = coef(r)[1], stderror(r)[1]
        @test confint(r; level=0.9) ≈ [b - c * s b + c * s]
        @test r.details.model isa DrSnow.FixedEffectModels.FixedEffectModel
        @test !r.details.staggered
        @test estimand(r) == "ATT"
        s = sprint(show, MIME"text/plain"(), r)
        @test occursin("Two-way fixed effects", s) && occursin("Treated units", s)
    end

    @testset "covariates, weights, clustering, vcov keyword" begin
        rng = StableRNG(102)
        df = sim_2x2(rng; N=200)
        df.z = randn(rng, nrow(df))
        df.y .+= 0.7 .* df.z
        df.w = rand(rng, nrow(df)) .+ 0.5
        df.state = (df.unit .- 1) .÷ 10
        r = did_twfe(df, :y, :d, :unit, :time; covariates=[:z], weights=:w, cluster=:state)
        m = DrSnow.FixedEffectModels.reg(df, make_formula(:y, [:d, :z]; fe=[:unit, :time]),
                                         Vcov.cluster(:state); weights=:w)
        @test coef(r)[1] ≈ coef(m)[1]
        @test stderror(r)[1] ≈ stderror(m)[1]
        @test r.n_clusters == 20
        @test dof_residual(r) == 19
        rr = did_twfe(df, :y, :d, :unit, :time; vcov=Vcov.robust(), cluster=:state)
        @test rr.n_clusters == 0          # vcov wins over cluster
        rn = did_twfe(df, :y, :d, :unit, :time; cluster=nothing)
        @test isfinite(stderror(rn)[1])
        # TreatmentPanel method and FirstTreated input give the same estimate
        p = TreatmentPanel(df, :y, :d, :unit, :time, [:z])
        @test coef(did_twfe(p))[1] ≈ coef(did_twfe(df, :y, :d, :unit, :time;
                                                   covariates=[:z]))[1]
        rf = did_twfe(df, :y, FirstTreated(:g), :unit, :time)
        @test coef(rf)[1] ≈ coef(did_twfe(df, :y, :d, :unit, :time))[1]
        # Column names with spaces/punctuation are safe (no string formulas)
        dn = rename(df, :y => Symbol("log wage"), :d => Symbol("d; println(1)"))
        rn2 = did_twfe(dn, Symbol("log wage"), Symbol("d; println(1)"), :unit, :time)
        @test coef(rn2)[1] ≈ coef(did_twfe(df, :y, :d, :unit, :time))[1]
    end

    @testset "error paths" begin
        rng = StableRNG(103)
        df = sim_2x2(rng; N=40)
        z = copy(df); z.d .= 0
        @test_throws ArgumentError did_twfe(z, :y, :d, :unit, :time)
        o = copy(df); o.d .= 1
        @test_throws ArgumentError did_twfe(o, :y, :d, :unit, :time)
        # treated units treated in every period → collinear with unit FE
        a = copy(df); a.d = Int.(a.g .> 0)
        @test_throws ArgumentError did_twfe(a, :y, :d, :unit, :time)
        @test_throws ArgumentError did_twfe(df, :y, :nope, :unit, :time)
        nb = copy(df); nb.d = nb.d .* 2
        @test_throws ArgumentError did_twfe(nb, :y, :d, :unit, :time)
    end

    @testset "staggered / non-absorbing designs warn" begin
        rng = StableRNG(104)
        df = sim_staggered(rng; N=120, T=8, cohorts=[0, 3, 6])
        r = @test_logs (:warn, r"staggered adoption") did_twfe(df, :y, :d, :unit, :time)
        @test r.details.staggered
        @test r.details.twfe_weights isa TWFEWeights
        @test occursin("negative", r.details.note)
        @test_logs did_twfe(df, :y, :d, :unit, :time; warn_heterogeneity=false)
        sw = copy(df)
        u = sw.unit[findfirst(sw.g .> 0)]
        sw.d[(sw.unit .== u) .& (sw.time .== maximum(sw.time))] .= 0
        r2 = @test_logs (:warn, r"non-absorbing") did_twfe(sw, :y, :d, :unit, :time)
        @test !r2.details.absorbing
    end

    @testset "row-shuffling invariance" begin
        rng = StableRNG(105)
        df = sim_2x2(rng; N=60)
        r1 = did_twfe(df, :y, :d, :unit, :time)
        r2 = did_twfe(shuffle_rows(rng, df), :y, :d, :unit, :time)
        @test coef(r1) ≈ coef(r2)
        @test vcov(r1) ≈ vcov(r2)
    end

    @testset "Monte Carlo: CI coverage with few clusters (t(G-1))" begin
        R = mc_reps(1000, 200)
        rng = StableRNG(106)
        hits = 0
        for _ in 1:R
            df = sim_2x2(rng; N=12, T=6, att=1.0, sigma=1.0)
            ci = confint(did_twfe(df, :y, :d, :unit, :time); level=0.95)
            hits += ci[1] <= 1.0 <= ci[2]
        end
        @test mc_close(hits / R, 0.95, R; slack=0.02)
    end
end
