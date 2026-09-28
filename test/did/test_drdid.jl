@testset "Sant'Anna–Zhao doubly robust DiD" begin
    # Two-period DGP with selection on x and x-dependent trends: parallel trends holds
    # only conditional on x. ATT = τ.
    function sim_sz(rng; n=1000, τ=2.0, panel=true)
        x1 = randn(rng, n); x2 = randn(rng, n)
        p = @. 1 / (1 + exp(-(0.4 * x1 - 0.3 * x2)))
        D = rand(rng, n) .< p
        α = 1.0 .+ x1 .+ randn(rng, n)
        trend = @. 1.0 + 1.5 * x1 + 0.5 * x2
        y0 = α .+ randn(rng, n)
        y1 = α .+ trend .+ τ .* D .+ randn(rng, n)
        if panel
            return DataFrame(id=repeat(1:n; inner=2), t=repeat([1, 2], n),
                             y=vec(permutedims(hcat(y0, y1))), d=repeat(Int.(D); inner=2),
                             x1=repeat(x1; inner=2), x2=repeat(x2; inner=2))
        end
        post = rand(rng, n) .< 0.5
        return DataFrame(t=Int.(post) .+ 1, y=ifelse.(post, y1, y0), d=Int.(D), x1=x1,
                         x2=x2)
    end
    methods = (:dr_improved, :dr, :ipw, :ipw_unnormalized, :reg)

    @testset "truth recovery (conditional parallel trends)" begin
        rng = StableRNG(401)
        df = sim_sz(rng; n=6000)
        for m in methods
            r = did_drdid(df, :y, :d, :id, :t; method=m, covariates=[:x1, :x2])
            @test abs(coef(r)[1] - 2.0) < 4 * stderror(r)[1]
            @test r.n_treated + r.n_control == 6000
        end
        naive = did_drdid(df, :y, :d, :id, :t)
        @test abs(coef(naive)[1] - 2.0) > 4 * stderror(naive)[1]   # unconditional: biased
        rc = sim_sz(rng; n=12000, panel=false)
        for m in methods
            r = did_drdid(rc, :y, :d, nothing, :t; method=m, covariates=[:x1, :x2])
            @test abs(coef(r)[1] - 2.0) < 4 * stderror(r)[1]
        end
    end

    @testset "closed forms without covariates" begin
        rng = StableRNG(402)
        df = sim_sz(rng; n=400)
        w = unstack(df, :id, :t, :y)
        dy = w[!, "2"] .- w[!, "1"]
        D = df.d[1:2:end] .== 1
        dd = mean(dy[D]) - mean(dy[.!D])
        n = length(dy); n1 = count(D); n0 = n - n1
        # influence-function variance of a difference in means
        se = sqrt(sum(abs2, dy[D] .- mean(dy[D])) / n1^2 +
                  sum(abs2, dy[.!D] .- mean(dy[.!D])) / n0^2)
        for m in methods
            r = did_drdid(df, :y, :d, :id, :t; method=m)
            @test coef(r)[1] ≈ dd
            @test stderror(r)[1] ≈ se
        end
        rc = sim_sz(rng; n=800, panel=false)
        cm(d, t) = mean(rc.y[(rc.d .== d) .& (rc.t .== t)])
        ddr = (cm(1, 2) - cm(1, 1)) - (cm(0, 2) - cm(0, 1))
        # (the unnormalized IPW estimator is not a difference of cell means)
        for m in (:dr_improved, :dr, :ipw, :reg)
            @test coef(did_drdid(rc, :y, :d, nothing, :t; method=m))[1] ≈ ddr
        end
        # outcome regression = imputation from a control-group OLS fit
        X = hcat(ones(n), df.x1[1:2:end], df.x2[1:2:end])
        β = X[.!D, :] \ dy[.!D]
        r = did_drdid(df, :y, :d, :id, :t; method=:reg, covariates=[:x1, :x2])
        @test coef(r)[1] ≈ mean(dy[D]) - mean(X[D, :] * β)
    end

    @testset "input handling, clustering, invariance" begin
        rng = StableRNG(403)
        df = sim_sz(rng; n=300)
        r = did_drdid(df, :y, :d, :id, :t; covariates=[:x1])
        dit = copy(df); dit.d = dit.d .* (dit.t .== 2)       # D_it instead of group
        @test coef(did_drdid(dit, :y, :d, :id, :t; covariates=[:x1])) ≈ coef(r)
        @test coef(did_drdid(shuffle_rows(rng, df), :y, :d, :id, :t;
                             covariates=[:x1])) ≈ coef(r)
        @test stderror(did_drdid(df, :y, :d, :id, :t; covariates=[:x1], cluster=:id)) ≈
              stderror(r)
        df.grp = (df.id .- 1) .÷ 10
        rc = did_drdid(df, :y, :d, :id, :t; covariates=[:x1], cluster=:grp)
        @test rc.n_clusters == 30
        @test coef(rc) ≈ coef(r)
        @test length(r.details.influence) == 300
        p = TreatmentPanel(df, :y, :d, :id, :t, [:x1])
        @test coef(did_drdid(p)) ≈ coef(r)
        # unbalanced: units without both periods are dropped with a warning
        @test_logs (:warn, r"dropped") match_mode = :any did_drdid(df[2:end, :], :y, :d,
                                                                    :id, :t)
    end

    @testset "error paths" begin
        rng = StableRNG(404)
        df = sim_sz(rng; n=100)
        three = vcat(df, DataFrame(id=1, t=3, y=0.0, d=0, x1=0.0, x2=0.0))
        @test_throws ArgumentError did_drdid(three, :y, :d, :id, :t)
        @test_throws ArgumentError did_drdid(df, :y, :d, :id, :t; method=:foo)
        z = copy(df); z.d .= 0
        @test_throws ArgumentError did_drdid(z, :y, :d, :id, :t)
        c = copy(df); c.x3 = 2 .* c.x1
        @test_throws ArgumentError did_drdid(c, :y, :d, :id, :t; covariates=[:x1, :x3])
        bad = copy(df); bad.d[1] = 1; bad.d[2] = 0     # treated only in period 1
        @test_throws ArgumentError did_drdid(bad, :y, :d, :id, :t)
    end

    @testset "Monte Carlo: coverage of influence-function CIs" begin
        R = mc_reps(1000, 200)
        rng = StableRNG(405)
        hits = Dict(m => 0 for m in (:dr_improved, :dr, :ipw))
        hits_rc = 0
        for _ in 1:R
            df = sim_sz(rng; n=500)
            for m in keys(hits)
                ci = confint(did_drdid(df, :y, :d, :id, :t; method=m,
                                       covariates=[:x1, :x2]))
                hits[m] += ci[1] <= 2.0 <= ci[2]
            end
            rc = sim_sz(rng; n=1000, panel=false)
            ci = confint(did_drdid(rc, :y, :d, nothing, :t; covariates=[:x1, :x2]))
            hits_rc += ci[1] <= 2.0 <= ci[2]
        end
        for m in keys(hits)
            @test mc_close(hits[m] / R, 0.95, R; slack=0.02)
        end
        @test mc_close(hits_rc / R, 0.95, R; slack=0.02)
    end
end
