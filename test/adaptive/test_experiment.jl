# Running experiments: environments, step-through API, runner, the log, regret.

@testset "runner and log" begin
    p = GaussianThompson(3; floor=0.05, burnin=9)
    env = GaussianBandit([0.0, 0.5, 1.0]; sd=1.0)
    log = run_adaptive_experiment(p, env, 300; rng=StableRNG(1))
    @test nobs(log) == 300
    @test size(log.probabilities) == (300, 3)
    @test all(abs.(sum(log.probabilities; dims=2) .- 1) .< 1e-12)
    @test all(log.probabilities .>= 0.05 - 1e-12)
    @test all(log.probabilities[1:9, :] .== 1 / 3)             # burn-in
    @test log.batch == 1:300 && log.batch_start == 1:300
    @test isempty(log.snapshots)
    @test size(log.true_means) == (300, 3) && all(log.true_means[:, 3] .== 1.0)
    @test p.n == zeros(Int, 3)                                  # policy not mutated
    # reproducible
    log2 = run_adaptive_experiment(p, env, 300; rng=StableRNG(1))
    @test log2.arms == log.arms && log2.outcomes == log.outcomes
    # the adaptive policy moves toward the best arm
    @test mean(log.arms[201:300] .== 3) > 0.5
    # Tables.jl interface
    df = DataFrame(log)
    @test names(df) == ["t", "batch", "arm", "outcome", "p1", "p2", "p3"]
    @test df.p2 == log.probabilities[:, 2]
    @test occursin("3 arms", sprint(show, MIME"text/plain"(), log))
    # regret
    reg = cumulative_regret(log)
    @test length(reg) == 300 && all(diff(reg) .>= -1e-12)
    @test cumulative_regret(log; expected=false)[end] ≈
          sum(1.0 .- log.true_means[CartesianIndex.(1:300, log.arms)])
    # arms are drawn from the recorded probabilities
    big = run_adaptive_experiment(EpsilonGreedy(3; epsilon=0.5),
                                  BernoulliBandit([0.2, 0.5, 0.6]), 20_000;
                                  batch_size=500, rng=StableRNG(2))
    for k in 1:3
        dev = mean((big.arms .== k) .- big.probabilities[:, k])
        @test abs(dev) < 4 * sqrt(0.25 / 20_000)
    end
    @test all(y -> y == 0 || y == 1, big.outcomes)
end

@testset "potential-outcome matrix runner and batches" begin
    rng = StableRNG(3)
    Y = randn(rng, 120, 2) .+ [0.0 1.0]
    log = run_adaptive_experiment(SoftmaxPolicy(2; temperature=0.3, floor=0.1), Y;
                                  batch_size=[20, 40, 60], rng=StableRNG(4))
    @test log.outcomes == [Y[t, log.arms[t]] for t in 1:120]
    @test log.batch_start == [1, 21, 61]
    for u in (1:20, 21:60, 61:120)
        @test all(log.probabilities[t, :] == log.probabilities[u[1], :] for t in u)
    end
    @test log.true_means === nothing
    @test_throws ArgumentError cumulative_regret(log)
    @test_throws ArgumentError run_adaptive_experiment(EpsilonGreedy(2), Y;
                                                       batch_size=[50, 50])
    @test_throws DimensionMismatch run_adaptive_experiment(EpsilonGreedy(3), Y)
    @test_throws ArgumentError run_adaptive_experiment(EpsilonGreedy(2), Y;
                                                       batch_size=0)
end

@testset "step-through experiment" begin
    exp = AdaptiveExperiment(BetaBernoulliThompson(2; floor=0.1, ndraws=2000))
    rng = StableRNG(5)
    for b in 1:10
        arms = assign!(exp, 25; rng=rng)
        @test length(arms) == 25
        @test_throws ArgumentError assign!(exp, 5; rng=rng)     # outcomes pending
        @test_throws DimensionMismatch observe!(exp, zeros(3))
        observe!(exp, [rand(rng) < (a == 2 ? 0.7 : 0.4) for a in arms])
    end
    @test_throws ArgumentError observe!(exp, [1.0])
    log = experiment_log(exp)
    @test nobs(log) == 250 && length(log.batch_start) == 10
    @test occursin("250 outcomes", sprint(show, exp))
    @test_throws ArgumentError experiment_log(AdaptiveExperiment(EpsilonGreedy(2)))
    @test_throws ArgumentError observe!(AdaptiveExperiment(EpsilonGreedy(2)), [1.0])
end

@testset "contextual runs and policy snapshots" begin
    mf(x) = [0.0, x[1], -x[1]]
    env = ContextualBandit(mf, rng -> randn(rng, 2), 3; noise_sd=0.5)
    p = LinearThompson(3, 2; floor=0.3, floor_decay=0.5, burnin=60)
    log = run_adaptive_experiment(p, env, 400; batch_size=50, rng=StableRNG(6))
    @test size(log.contexts) == (400, 2)
    @test length(log.snapshots) == 8
    d = DrSnow._ad_data(log)
    err = maximum(maximum(abs.(d.probfn(log.batch[t], log.contexts[t, :]) .-
                               log.probabilities[t, :])) for t in 1:400)
    @test err < 1e-12
    @test names(DataFrame(log))[end-1:end] == ["x1", "x2"]
    exp = AdaptiveExperiment(p)
    @test_throws ArgumentError assign!(exp, 3)                  # needs contexts
    arms = assign!(exp; contexts=randn(StableRNG(1), 4, 2))
    @test length(arms) == 4
    bern = ContextualBandit(x -> [0.2, 0.8], rng -> [0.0], 2; outcome=:bernoulli)
    lb = run_adaptive_experiment(LinearThompson(2, 1), bern, 50; rng=StableRNG(2))
    @test all(y -> y in (0.0, 1.0), lb.outcomes)
    @test_throws ArgumentError ContextualBandit(mf, identity, 3; outcome=:poisson)
    @test_throws ArgumentError BernoulliBandit([0.5, 1.5])
end
