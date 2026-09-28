@testset "Sun–Abraham interaction-weighted event study" begin
    eff(g, e) = e >= 0 ? 1.0 + 0.5 * e + 0.6 * (g - 3) : 0.0

    @testset "truth recovery; TWFE event study is contaminated" begin
        rng = StableRNG(601)
        df = sim_staggered(rng; N=3000, T=7, cohorts=[0, 3, 5], effect=eff, sigma=0.5)
        sa = did_sun_abraham(df, :y, :d, :unit, :time)
        @test relative_periods(sa) == [-4, -3, -2, 0, 1, 2, 3, 4]
        @test sa.reference == [-1]
        u = unique(df[df.g .> 0, [:unit, :g]])
        gs = u.g .- 1999
        θ(e) = mean(eff(g, e) for g in gs if 1 <= g + e <= 7)
        @test all(abs.(coef(sa) .- θ.(relative_periods(sa))) .< 4 .* stderror(sa))
        att = sa.details.att
        @test abs(coef(att)[1] - true_simple_att(df, eff)) < 4 * stderror(att)[1]
        es = @test_logs (:warn,) event_study(df, :y, :d, :unit, :time; estimator=:twfe)
        pre = findall(<(0), relative_periods(es))
        @test maximum(abs.(coef(es)[pre]) ./ stderror(es)[pre]) > 4   # spurious pre-trends
        @test estimate(sa) ≈ mean(coef(sa)[relative_periods(sa) .>= 0])
    end

    @testset "closed form: CATT(g,e) = 2×2 DiD vs never treated (base g-1)" begin
        rng = StableRNG(602)
        df = sim_staggered(rng; N=400, T=6, cohorts=[0, 3, 4], effect=eff)
        sa = did_sun_abraham(df, :y, :d, :unit, :time)
        cs = did_callaway_santanna(df, :y, :d, :unit, :time; base_period=:universal,
                                   method=:reg, bootstrap=false)
        ce = sa.details.cohort_effects
        for (g, t, b) in zip(cs.groups, cs.times, cs.coef)
            k = findfirst(i -> ce.cohort[i] == cs.periods[g] && ce.e[i] == t - g,
                          1:nrow(ce))
            @test ce.estimate[k] ≈ b atol = 1e-10
        end
        # aggregation weights are cohort shares among cohorts observed at e
        W = sa.details.aggregation_weights
        @test all(sum(W; dims=2) .≈ 1)
        @test coef(sa) ≈ W * ce.estimate
        nc = nrow(ce)
        Vβ = vcov(sa.details.model)[1:nc, 1:nc]      # cell dummies come first
        @test sqrt.(diag(Vβ)) ≈ ce.std_error
        @test vcov(sa) ≈ W * Vβ * W'
    end

    @testset "options and error paths" begin
        rng = StableRNG(603)
        df = sim_staggered(rng; N=300, T=6, cohorts=[0, 3, 4], effect=eff)
        w = did_sun_abraham(df, :y, :d, :unit, :time; max_pre=2, max_post=1)
        @test relative_periods(w) == [-2, 0, 1]
        full = did_sun_abraham(df, :y, :d, :unit, :time)
        @test coef(w) ≈ coef(full)[[findfirst(==(e), relative_periods(full))
                                     for e in (-2, 0, 1)]]
        @test coef(did_sun_abraham(df, :y, FirstTreated(:g), :unit, :time)) ≈ coef(full)
        @test coef(did_sun_abraham(shuffle_rows(rng, df), :y, :d, :unit, :time)) ≈
              coef(full)
        # no never-treated units: last cohort becomes the comparison group
        nn = sim_staggered(rng; N=300, T=6, cohorts=[2, 4, 6], effect=eff)
        s2 = @test_logs (:info, r"no never-treated") did_sun_abraham(nn, :y, :d, :unit,
                                                                      :time)
        @test maximum(relative_periods(s2)) <= 3
        one = sim_staggered(rng; N=100, T=5, assign=fill(3, 100))
        @test_throws ArgumentError did_sun_abraham(one, :y, :d, :unit, :time)
        early = sim_staggered(rng; N=200, T=5, cohorts=[0, 1, 3])
        @test_logs (:warn, r"first period") did_sun_abraham(early, :y, :d, :unit, :time)
        df.z = randn(rng, nrow(df))
        @test length(coef(did_sun_abraham(df, :y, :d, :unit, :time; covariates=[:z]))) ==
              length(coef(full))
    end

    @testset "Monte Carlo: coverage of the e = 0 effect" begin
        R = mc_reps(1000, 150)
        rng = StableRNG(604)
        hits = 0; hits_att = 0
        for _ in 1:R
            df = sim_staggered(rng; N=150, T=5, cohorts=[0, 3, 4], effect=eff)
            sa = did_sun_abraham(df, :y, :d, :unit, :time)
            k = findfirst(==(0), relative_periods(sa))
            ci = confint(sa)
            u = unique(df[df.g .> 0, [:unit, :g]])
            θ0 = mean(eff(g - 1999, 0) for g in u.g)
            hits += ci[k, 1] <= θ0 <= ci[k, 2]
            ca = confint(sa.details.att)
            hits_att += ca[1] <= true_simple_att(df, eff) <= ca[2]
        end
        @test mc_close(hits / R, 0.95, R; slack=0.02)
        @test mc_close(hits_att / R, 0.95, R; slack=0.02)
    end
end
