# Surrogate-model design optimization.

@testset "optimize_design finds the analytic sample size" begin
    d = des_simple_design()
    n_true = power_means(effect=0.3, power=0.8).parameters.n       # ≈ 350.8
    budget = mc_reps(6000, 3000)
    for (sur, tr) in ((:probit, :sqrt), (:logistic, :log), (:isotonic, :sqrt))
        opt = optimize_design(d; space=(n=20:4:800,), target_power=0.8,
                              surrogate=sur, transform=tr, max_sims=budget,
                              sims_per_point=200, rng=StableRNG(10))
        @test opt.reached
        @test abs(opt.best.n / n_true - 1) < 0.15
        @test opt.total_sims <= budget
        @test sum(opt.evaluations.sims) == opt.total_sims
        @test nrow(opt.predictions) == length(20:4:800)
        sur === :isotonic || @test opt.predicted_se > 0
    end
    # reproducible, identical with and without threads
    a = optimize_design(d; space=(n=20:10:600,), max_sims=1600, rng=StableRNG(11),
                        threaded=true)
    b = optimize_design(d; space=(n=20:10:600,), max_sims=1600, rng=StableRNG(11),
                        threaded=false)
    @test a.best == b.best && isequal(a.evaluations, b.evaluations)
    v = optimize_design(d; space=(n=20:10:600,), max_sims=1600, verify_sims=400,
                        rng=StableRNG(12))
    @test v.verification.sims == 400
    @test abs(v.verification.power - v.predicted_power) < 4 * v.verification.mc_se + 0.05
    @test occursin("Chosen design", sprint(show, MIME"text/plain"(), v))
end

@testset "optimize_design with two parameters and a cost" begin
    # cluster trial: n_clusters × cluster_size, clusters cost 10× an individual
    function cpop(rng, p)
        J, m = p.n_clusters, p.cluster_size
        u = repeat(sqrt(p.icc) .* randn(rng, J); inner=m)
        e = sqrt(1 - p.icc) .* randn(rng, J * m)
        DataFrame(cl=repeat(1:J; inner=m), Y0=u .+ e, Y1=u .+ e .+ p.effect)
    end
    function cest(data, p)
        # cluster-level analysis: difference in cluster means, t(J - 2)
        g = combine(groupby(data, :cl), :Y => mean => :Y, :Z => first => :Z)
        return experiment_estimate(g, :Y, :Z)
    end
    d = declare_design(cpop; params=(n_clusters=20, cluster_size=10, icc=0.1,
                                     effect=0.3),
                       assignment=(data, p) -> ClusterRandomization(data.cl,
                                                                    p.n_clusters ÷ 2),
                       estimators="cluster means" => cest)
    cost(p) = 10 * p.n_clusters + p.n_clusters * p.cluster_size
    opt = optimize_design(d; space=(n_clusters=10:4:70, cluster_size=[5, 10, 20, 40]),
                          cost=cost, target_power=0.8, max_sims=mc_reps(6000, 3000),
                          transform=p -> [sqrt(p.n_clusters /
                                               (0.1 + 0.9 / p.cluster_size))],
                          sims_per_point=150, n_initial=8, rng=StableRNG(13))
    @test opt.reached
    # analytic check of the chosen design (cluster-mean analysis = equal-size Bloom)
    pa = power_cluster(effect=0.3, icc=0.1, n_clusters=opt.best.n_clusters,
                       cluster_size=opt.best.cluster_size).power
    @test pa > 0.7
    # no cheaper candidate is analytically much above the target
    cheaper = [(J, m) for J in 10:4:70, m in [5, 10, 20, 40]
               if 10J + J * m < opt.cost - 1e-9]
    @test all(power_cluster(effect=0.3, icc=0.1, n_clusters=J, cluster_size=m).power <
              0.88 for (J, m) in cheaper)
    @test_throws ArgumentError optimize_design(d; space=(n_clusters=10:4:70,
                                                         cluster_size=[5, 10]),
                                               surrogate=:isotonic)
end

@testset "optimize_design errors" begin
    d = des_simple_design()
    @test_throws ArgumentError optimize_design(d; space=(n=20:10:100,), target_power=1.2)
    @test_throws ArgumentError optimize_design(d; space=(n=20:10:100,), surrogate=:gp)
    @test_throws ArgumentError optimize_design(d; space=(n=20:10:100,), max_sims=100)
    @test_throws ArgumentError optimize_design(d; space=(n=[20, 30],), n_initial=5)
    @test_throws ArgumentError optimize_design(d; space=(n=20:10:100,), estimator="x")
    @test_throws ArgumentError optimize_design(d; space=(n=["a", "b", "c", "d", "e"],))
    # unreachable target: reports the best candidate and reached = false
    r = optimize_design(d; space=(n=10:2:30,), max_sims=1200, rng=StableRNG(1))
    @test !r.reached
end

@testset "surrogate binomial fit matches GLM.jl" begin
    F = hcat(ones(6), sqrt.([20.0, 50, 100, 200, 400, 800]))
    k = [9.0, 22, 41, 88, 150, 196]
    m = fill(200.0, 6)
    for (link, L) in ((:probit, DrSnow.GLM.ProbitLink()), (:logit, DrSnow.GLM.LogitLink()))
        β, V, _ = DrSnow._des_fit_glm(F, k, m, link)
        g = DrSnow.GLM.glm(F, (k .+ 0.5) ./ (m .+ 1), DrSnow.Distributions.Binomial(), L;
                           weights=m .+ 1)
        @test β ≈ coef(g) rtol = 1e-7
        @test V ≈ vcov(g) rtol = 1e-4        # GLM.jl stops at a looser tolerance
    end
end
