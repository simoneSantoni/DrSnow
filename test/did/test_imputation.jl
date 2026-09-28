@testset "Borusyak–Jaravel–Spiess imputation" begin
    eff(g, e) = e >= 0 ? 1.0 + 0.5 * e + 0.6 * (g - 3) : 0.0

    @testset "truth recovery" begin
        rng = StableRNG(701)
        df = sim_staggered(rng; N=3000, T=7, cohorts=[0, 3, 5], effect=eff, sigma=0.5)
        a = did_imputation(df, :y, :d, :unit, :time)
        @test a isa DiDEstimate
        @test abs(coef(a)[1] - true_simple_att(df, eff)) < 4 * stderror(a)[1]
        @test dof_residual(a) == Inf
        e = did_imputation(df, :y, :d, :unit, :time; horizons=0:3, pretrends=2)
        @test relative_periods(e) == [-2, -1, 0, 1, 2, 3]
        @test isempty(e.reference)
        u = unique(df[df.g .> 0, [:unit, :g]])
        gs = u.g .- 1999
        θ(h) = mean(eff(g, h) for g in gs if g + h <= 7)
        post = 3:6
        @test all(abs.(coef(e)[post] .- θ.(0:3)) .< 4 .* stderror(e)[post])
        @test all(abs.(coef(e)[1:2]) .< 4 .* stderror(e)[1:2])
        @test pre_trend_test(e) isa DiagnosticTest
        @test coef(e.details.att) ≈ coef(a)
        # imputed effects table
        @test nrow(a.details.effects) == count(df.d .== 1)
    end

    @testset "closed forms and FixedEffectModels reference" begin
        rng = StableRNG(702)
        # single cohort, balanced: equals post-vs-pre DiD of means
        df = sim_staggered(rng; N=100, T=6, assign=vcat(fill(4, 50), zeros(Int, 50)))
        a = did_imputation(df, :y, :d, :unit, :time)
        m(mask) = mean(df.y[mask])
        tr = df.g .> 0; post = df.time .>= 2003
        dd = (m(tr .& post) - m(tr .& .!post)) - (m(.!tr .& post) - m(.!tr .& .!post))
        @test coef(a)[1] ≈ dd
        # staggered with a covariate: imputed Y(0) equals the FE fit on untreated rows
        st = sim_staggered(rng; N=80, T=6, cohorts=[0, 3, 5], effect=eff)
        st.z = randn(rng, nrow(st))
        st.y .+= 0.5 .* st.z
        b = did_imputation(st, :y, :d, :unit, :time; covariates=[:z])
        un = st[st.d .== 0, :]
        fm = DrSnow.FixedEffectModels.reg(un, make_formula(:y, [:z]; fe=[:unit, :time]);
                                          save=:fe, drop_singletons=false)
        fes = DrSnow.FixedEffectModels.fe(fm)
        αu = Dict(zip(un.unit, fes[!, 1])); λt = Dict(zip(un.time, fes[!, 2]))
        trd = st[st.d .== 1, :]
        yhat = [αu[r.unit] + λt[r.time] + coef(fm)[1] * r.z for r in eachrow(trd)]
        @test coef(b)[1] ≈ mean(trd.y .- yhat) atol = 1e-6
        # constant weights change nothing
        st.w = fill(2.0, nrow(st))
        @test coef(did_imputation(st, :y, :d, :unit, :time; covariates=[:z],
                                  weights=:w)) ≈ coef(b)
    end

    @testset "options, invariance and errors" begin
        rng = StableRNG(703)
        df = sim_staggered(rng; N=200, T=6, cohorts=[0, 3, 4], effect=eff)
        a = did_imputation(df, :y, :d, :unit, :time)
        s = did_imputation(shuffle_rows(rng, df), :y, :d, :unit, :time)
        @test coef(s) ≈ coef(a) && vcov(s) ≈ vcov(a)
        @test coef(did_imputation(df, :y, FirstTreated(:g), :unit, :time)) ≈ coef(a)
        df.state = (df.unit .- 1) .÷ 5
        c = did_imputation(df, :y, :d, :unit, :time; cluster=:state)
        @test c.n_clusters == 40 && coef(c) ≈ coef(a)
        @test_throws ArgumentError did_imputation(df, :y, :d, :unit, :time; horizons=[9])
        @test_throws ArgumentError did_imputation(df, :y, :d, :unit, :time; pretrends=9)
        sw = copy(df)
        u = sw.unit[findfirst(sw.g .== 2002)]
        sw.d[(sw.unit .== u) .& (sw.time .== maximum(sw.time))] .= 0   # leaves treatment
        @test_throws ArgumentError did_imputation(sw, :y, :d, :unit, :time)
        # all units treated at the same time: no untreated observations to impute from
        all_t = sim_staggered(rng; N=30, T=5, assign=fill(3, 30))
        @test_throws ArgumentError did_imputation(all_t, :y, :d, :unit, :time)
        # anticipation shifts the untreated sample and the horizons
        ea = did_imputation(df, :y, :d, :unit, :time; horizons=:all, anticipation=1)
        @test minimum(relative_periods(ea)) == -1
    end

    @testset "Monte Carlo: coverage (conservative variance) and pre-trend size" begin
        R = mc_reps(1000, 150)
        rng = StableRNG(704)
        hits = 0; hits0 = 0; rej = 0
        for _ in 1:R
            df = sim_staggered(rng; N=150, T=6, cohorts=[0, 3, 5], effect=eff)
            e = did_imputation(df, :y, :d, :unit, :time; horizons=[0], pretrends=2)
            ca = confint(e.details.att)
            hits += ca[1] <= true_simple_att(df, eff) <= ca[2]
            u = unique(df[df.g .> 0, [:unit, :g]])
            θ0 = mean(eff(g - 1999, 0) for g in u.g)
            c0 = confint(e)[3, :]
            hits0 += c0[1] <= θ0 <= c0[2]
            rej += pre_trend_test(e).pvalue < 0.05
        end
        # The BJS variance is conservative under effect heterogeneity: coverage ≥ 95%.
        @test hits / R >= 0.95 - 3.5 * sqrt(0.0475 / R) - 0.02
        @test hits0 / R >= 0.95 - 3.5 * sqrt(0.0475 / R) - 0.02
        @test mc_close(rej / R, 0.05, R; slack=0.02)
    end
end
