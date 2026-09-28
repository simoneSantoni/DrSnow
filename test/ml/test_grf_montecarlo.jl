# Monte Carlo coverage of forest-based inference (Wager & Athey 2018 designs), with
# `mc_reps(full, fast)` replications (DRSNOW_SLOW_TESTS=true for full counts).

grf_zeta(u) = 1 + 1 / (1 + exp(-20 * (u - 1 / 3)))

@testset "Forest inference: Monte Carlo coverage" begin
    @testset "confounded design, no effect (WA 2018, §5.1)" begin
        # e(x) = (1 + β₂,₄(x₁))/4, m(x) = 2x₁ - 1, τ = 0: ATE / ATT / overlap-ATE
        # intervals and pointwise CATE intervals
        reps = mc_reps(200, 12)
        n = 1000
        Xt = rand(StableRNG(99), 50, 2)
        cov = zeros(3)
        cov_pt = 0.0
        quiet(f) = Base.CoreLogging.with_logger(f, Base.CoreLogging.NullLogger())
        quiet() do
            for r in 1:reps
                rg = StableRNG(7000 + r)
                X = rand(rg, n, 2)
                e = 0.25 .* (1 .+ 20 .* X[:, 1] .* (1 .- X[:, 1]) .^ 3)
                W = Float64.(rand(rg, n) .< e)
                Y = 2 .* X[:, 1] .- 1 .+ randn(rg, n)
                cf = causal_forest(X, Y, W; num_trees=1000, rng=rg)
                for (j, t) in enumerate((:all, :treated, :overlap))
                    ci = confint(average_treatment_effect(cf; target=t))
                    cov[j] += ci[1] <= 0 <= ci[2]
                end
                pt = predict_interval(cf, Xt)
                cov_pt += mean(pt.conf_low .<= 0 .<= pt.conf_high)
            end
        end  # quiet: estimated propensities may trigger overlap warnings
        cov ./= reps
        cov_pt /= reps
        @info "Monte Carlo causal forest (WA design 1)" reps ate_att_ato = cov pointwise =
            cov_pt
        @test all(abs.(cov .- 0.95) .<= ml_cover_tol(reps))
        @test cov_pt >= 0.90
    end

    @testset "randomized design, heterogeneous effect (WA 2018, §5.2)" begin
        # τ(x) = ζ(x₁)ζ(x₂), d = 2: grf's own pointwise coverage here is about 0.93 at
        # n = 2000 (grf_forest_reference / docs); DrSnow matches it
        reps = mc_reps(100, 3)
        n = 2000
        Xt = rand(StableRNG(98), 100, 2)
        τt = grf_zeta.(Xt[:, 1]) .* grf_zeta.(Xt[:, 2])
        cov = zeros(100)
        for r in 1:reps
            rg = StableRNG(8000 + r)
            X = rand(rg, n, 2)
            W = Float64.(rand(rg, n) .< 0.5)
            Y = grf_zeta.(X[:, 1]) .* grf_zeta.(X[:, 2]) .* W .+ randn(rg, n)
            cf = causal_forest(X, Y, W; num_trees=2000, rng=rg)
            pt = predict_interval(cf, Xt)
            cov .+= pt.conf_low .<= τt .<= pt.conf_high
        end
        cov ./= reps
        @info "Monte Carlo causal forest (WA design 2)" reps mean(cov) minimum(cov)
        @test mean(cov) >= (SLOW_TESTS ? 0.90 : 0.85)
    end

    @testset "RATE bootstrap coverage without heterogeneity" begin
        # doubly robust scores with a constant effect and an uninformative priority:
        # the RATE is zero
        reps = mc_reps(500, 60)
        cov = zeros(2)
        for r in 1:reps
            rg = StableRNG(9000 + r)
            Γ = 1 .+ 2 .* randn(rg, 400)
            prio = rand(rg, 400)
            for (j, tg) in enumerate((:AUTOC, :QINI))
                ci = confint(rank_average_treatment_effect(Γ, prio; target=tg, R=100,
                                                           rng=rg))
                cov[j] += ci[1] <= 0 <= ci[2]
            end
        end
        cov ./= reps
        @info "Monte Carlo RATE coverage (AUTOC, QINI)" reps cov
        @test all(abs.(cov .- 0.95) .<= ml_cover_tol(reps))
    end
end
