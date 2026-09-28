@testset "TWFE decompositions" begin
    @testset "Goodman-Bacon: weights sum to one and reproduce TWFE" begin
        rng = StableRNG(301)
        # never-treated, three timing groups and an always-treated group
        assign = vcat(zeros(Int, 30), fill(3, 25), fill(5, 25), fill(7, 20), fill(1, 10))
        df = sim_staggered(rng; N=110, T=8, assign=assign,
                           effect=(g, e) -> 1.0 + 0.4e)
        b = bacon_decomposition(df, :y, :d, :unit, :time)
        tw = did_twfe(df, :y, :d, :unit, :time; warn_heterogeneity=false)
        @test b.twfe_estimate ≈ coef(tw)[1]
        @test sum(b.comparisons.weight) ≈ 1
        @test sum(b.comparisons.weight .* b.comparisons.estimate) ≈ coef(tw)[1]
        @test Set(b.comparisons.type) == Set([:treated_vs_never, :earlier_vs_later,
                                              :later_vs_earlier, :later_vs_always])
        @test sum(b.by_type.weight) ≈ 1
        # FirstTreated input and row order do not matter
        b2 = bacon_decomposition(shuffle_rows(rng, df), :y, FirstTreated(:g), :unit, :time)
        @test b2.twfe_estimate ≈ b.twfe_estimate
        @test sort(b2.comparisons.weight) ≈ sort(b.comparisons.weight)
        @test occursin("Goodman-Bacon", sprint(show, MIME"text/plain"(), b))
        # A 2x2 comparison equals the difference in mean changes (hand computation)
        k = findfirst(==(:treated_vs_never), b.comparisons.type)
        gk = b.comparisons.treated[k]
        sub = df[(df.g .== gk) .| (df.g .== 0), :]
        pre = sub.time .< gk
        m(x) = mean(x)
        dd = (m(sub.y[(sub.g .== gk) .& .!pre]) - m(sub.y[(sub.g .== gk) .& pre])) -
             (m(sub.y[(sub.g .== 0) .& .!pre]) - m(sub.y[(sub.g .== 0) .& pre]))
        @test b.comparisons.estimate[k] ≈ dd
    end

    @testset "Goodman-Bacon errors" begin
        rng = StableRNG(302)
        df = sim_staggered(rng; N=40, T=5, assign=repeat([0, 3], 20))
        @test_throws ArgumentError bacon_decomposition(df[2:end, :], :y, :d, :unit, :time)
        sw = copy(df)
        sw.d[(sw.unit .== 2) .& (sw.time .== maximum(sw.time))] .= 0   # leaves treatment
        @test_throws ArgumentError bacon_decomposition(sw, :y, :d, :unit, :time)
    end

    @testset "dCDH weights: exact decomposition of β_fe" begin
        rng = StableRNG(303)
        N, T = 60, 7
        assign = vcat(zeros(Int, 15), fill(2, 15), fill(4, 15), fill(6, 15))
        df = sim_staggered(rng; N=N, T=T, assign=assign, sigma=0.0,
                           effect=(g, e) -> 0.0)
        # heterogeneous cell effects, no noise: y = α_i + λ_t + Δ_it D_it
        Δ = [d == 1 ? 1.0 + 2.0 * (t - g) + 0.5 * (g - 2000) : 0.0
             for (d, t, g) in zip(df.d, df.time, df.g)]
        df.y .+= Δ
        w = twfe_weights(df, :y, :d, :unit, :time)
        key = Dict((u, t) => δ for (u, t, δ) in zip(df.unit, df.time, Δ))
        @test sum(w.weights.share .* [key[(u, t)] for (u, t) in
                                        zip(w.weights.unit, w.weights.time)]) ≈ w.beta_fe
        @test sum(w.weights.share) ≈ 1
        @test mean(w.weights.weight) ≈ 1
        @test w.n_negative > 0 && w.sum_negative < 0
        @test w.sum_negative + w.sum_positive ≈ 1
        @test w.beta_fe ≈ coef(did_twfe(df, :y, :d, :unit, :time;
                                        warn_heterogeneity=false))[1]
        # weighted and unbalanced: the identity still holds
        df.wt = 0.5 .+ rand(rng, nrow(df))
        dfu = df[setdiff(1:nrow(df), [3, 50, 77]), :]
        ww = twfe_weights(dfu, :y, :d, :unit, :time; weights=:wt)
        @test sum(ww.weights.share .* [key[(u, t)] for (u, t) in
                                         zip(ww.weights.unit, ww.weights.time)]) ≈
              ww.beta_fe
        @test ww.sigma_fe ≈ abs(ww.beta_fe) /
              sqrt(sum(ww.weights.share .* (ww.weights.weight .- 1)))  atol = 1e-8
        @test occursin("negative", sprint(show, MIME"text/plain"(), w))
        z = copy(df); z.d .= 0
        @test_throws ArgumentError twfe_weights(z, :y, :d, :unit, :time)
    end
end
