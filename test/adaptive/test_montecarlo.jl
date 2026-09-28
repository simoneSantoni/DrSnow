# Monte Carlo coverage checks (reduced replication counts unless DRSNOW_SLOW_TESTS).
# The full tables are in test/validation/adaptive/hadad_montecarlo*.csv.

@testset "Monte Carlo: adaptively weighted arm values (Hadad et al. design)" begin
    reps = mc_reps(1000, 100)
    truth = [1.0, 1.0, 1.0]                    # "no signal" design, T = 1000
    tv = vcat(truth, 0.0, 0.0)
    seeds = DrSnow.task_seeds(StableRNG(2021), reps)
    cover = zeros(5)
    naive_bias = 0.0
    for i in 1:reps
        rng = Random.Xoshiro(seeds[i])
        Y = truth' .+ (2 .* rand(rng, 1000, 3) .- 1)
        log = run_adaptive_experiment(GaussianThompson(3; floor=1 / 3, floor_decay=0.7,
                                                       burnin=15), Y; rng=rng)
        r = adaptive_arm_values(log; weights=:two_point, reference=3)
        ci = confint(r; level=0.9)
        cover .+= ci[:, 1] .<= tv .<= ci[:, 2]
        naive_bias += mean(coef(naive_arm_means(log))[1:3] .- truth)
    end
    tol = ad_cover_tol(reps; level=0.9)
    @test all(abs.(cover ./ reps .- 0.9) .< tol)
    # sample means of Thompson-sampled arms are biased downward (Nie et al. 2018)
    @test naive_bias / reps < -0.015
end

function ad_sim_mrt(rng; n=30, T=40, binary=false)
    id = repeat(1:n; inner=T)
    dp = repeat(1:T, n) ./ T
    x = rand(rng, n * T)
    avail = Int.(rand(rng, n * T) .< 0.8)
    prob = ifelse.(x .> 0.5, 0.6, 0.3)
    a = Float64.(rand(rng, n * T) .< prob)
    y = if binary
        Float64.(rand(rng, n * T) .< 0.3 .* exp.(0.5 .* x) .* exp.(0.2 .* a))
    else
        u = repeat(randn(rng, n); inner=T)
        1 .+ x .+ u .+ a .* (0.3 .- 0.4 .* dp) .+ randn(rng, n * T)
    end
    return DataFrame(id=id, dp=dp, x=x, avail=avail, prob=prob, a=a, y=y)
end

@testset "Monte Carlo: WCLS and EMEE coverage" begin
    reps = mc_reps(1000, 150)
    cw = zeros(2)
    ce = 0
    for r in 1:reps
        d = ad_sim_mrt(StableRNG(r))
        f = wcls(d, :y, :a, :id; rand_prob=:prob, moderators=[:dp], controls=[:dp, :x],
                 availability=:avail)
        ci = confint(f)
        cw .+= ci[:, 1] .<= [0.3, -0.4] .<= ci[:, 2]
        b = ad_sim_mrt(StableRNG(10_000 + r); n=40, binary=true)
        g = emee(b, :y, :a, :id; rand_prob=:prob, controls=[:x], availability=:avail)
        cg = confint(g)
        ce += cg[1, 1] <= 0.2 <= cg[1, 2]
    end
    tol = ad_cover_tol(reps)
    @test all(abs.(cw ./ reps .- 0.95) .< tol)
    # the small-sample corrected EMEE variance is conservative with 40 participants
    @test 0.95 - tol < ce / reps < 0.995
end

@testset "Monte Carlo: off-policy evaluation coverage" begin
    reps = mc_reps(1000, 150)
    truth = 1 / sqrt(2π)                         # E[x₁ 1{x₁ > 0}]
    pol = x -> x[1] > 0 ? 2 : 1
    cv = zeros(3)
    for r in 1:reps
        df, _ = ad_logged_data(StableRNG(100 + r), 500)
        for (j, m) in enumerate((:dr, :ipw, :snipw))
            f = off_policy_value(df, :y, :a, pol; propensity=:p, covariates=[:x1, :x2],
                                 method=m, outcome_learner=OLSLearner(),
                                 rng=StableRNG(r))
            ci = confint(f)
            cv[j] += ci[1, 1] <= truth <= ci[1, 2]
        end
    end
    @test all(abs.(cv ./ reps .- 0.95) .< ad_cover_tol(reps))
end

@testset "Monte Carlo: contextual adaptive weighting (Zhan et al. design)" begin
    reps = mc_reps(400, 30)
    mf(x) = [0.0, x[1], -x[1], 0.5 * x[2]]
    truth = mean(maximum(mf(randn(StableRNG(10^6 + i), 2))) for i in 1:200_000)
    best = x -> argmax(mf(x))
    cover = 0
    for r in 1:reps
        env = ContextualBandit(mf, rng -> randn(rng, 2), 4)
        p = LinearThompson(4, 2; floor=0.25, floor_decay=0.5, burnin=100)
        log = run_adaptive_experiment(p, env, 600; batch_size=100, rng=StableRNG(r))
        f = adaptive_policy_value(log, best)
        ci = confint(f; level=0.9)
        cover += ci[1, 1] <= truth <= ci[1, 2]
    end
    @test abs(cover / reps - 0.9) < ad_cover_tol(reps; level=0.9)
end
