# Assignment policies: floors, burn-in, probability computations, updates, errors.

@testset "floors" begin
    f = DrSnow._ad_apply_floor([1.0, 0.0, 0.0], 0.1)
    @test f ≈ [0.8, 0.1, 0.1]
    g = DrSnow._ad_apply_floor([0.5, 0.45, 0.05], 0.1)
    @test sum(g) ≈ 1
    @test all(g .>= 0.1 - 1e-12)
    @test g[1] > g[2] > g[3]
    @test DrSnow._ad_apply_floor([0.2, 0.8], 0.5) ≈ [0.5, 0.5]
    # decaying floor: c t^-α
    p = UCBPolicy(3; floor=0.3, floor_decay=0.5)
    for k in 1:3
        update_policy!(p, k, k == 1 ? 1.0 : 0.0)
    end
    for t in (1, 4, 100)
        e = assignment_probabilities(p, t)
        @test minimum(e) ≈ 0.3 * t^-0.5 atol = 1e-12
        @test sum(e) ≈ 1
    end
    @test_throws ArgumentError GaussianThompson(3; floor=0.4)
    @test_throws ArgumentError GaussianThompson(3; floor_decay=1.0)
    @test_throws ArgumentError GaussianThompson(1)
    @test_throws ArgumentError GaussianThompson(2; burnin=-1)
end

@testset "burn-in" begin
    p = EpsilonGreedy(4; epsilon=0.0, burnin=10)
    update_policy!(p, 1, 5.0)
    @test assignment_probabilities(p, 10) == fill(0.25, 4)
    @test assignment_probabilities(p, 11) != fill(0.25, 4)
    @test_throws ArgumentError assignment_probabilities(p, 0)
end

@testset "Thompson probabilities" begin
    # Gaussian: quadrature vs brute-force Monte Carlo of the posterior
    p = GaussianThompson(3; noise_var=1.0)
    rng = StableRNG(11)
    for (k, n, μ) in ((1, 5, 0.2), (2, 3, 0.5), (3, 8, 0.1))
        for _ in 1:n
            update_policy!(p, k, μ + randn(rng))
        end
    end
    e = assignment_probabilities(p)
    m, s = DrSnow._ad_posterior_normal(p)
    D = m' .+ s' .* randn(StableRNG(2), 400_000, 3)
    mc = DrSnow._ad_prob_best_draws(D)
    @test maximum(abs.(e .- mc)) < 0.006
    @test sum(e) ≈ 1
    # conjugate posterior
    q = GaussianThompson(2; prior_mean=1.0, prior_var=2.0, noise_var=4.0)
    update_policy!(q, 1, 3.0)
    update_policy!(q, 1, 5.0)
    m, s = DrSnow._ad_posterior_normal(q)
    prec = 1 / 2 + 2 / 4
    @test m[1] ≈ (1 / 2 + 8 / 4) / prec
    @test s[1] ≈ sqrt(1 / prec)
    @test m[2] ≈ 1.0 && s[2] ≈ sqrt(2.0)
    # symmetric arms get equal probability
    @test assignment_probabilities(GaussianThompson(4)) ≈ fill(0.25, 4) atol = 1e-12
    # estimated noise variance: pooled until an arm has two outcomes
    r = GaussianThompson(2)
    update_policy!(r, 1, 1.0)
    @test DrSnow._ad_noise_vars(r) == [1.0, 1.0]
    update_policy!(r, 1, 3.0)
    @test DrSnow._ad_noise_vars(r) ≈ [2.0, 2.0]

    # Beta-Bernoulli: Monte Carlo vs deterministic integration
    b1 = BetaBernoulliThompson(3; ndraws=200_000)
    b0 = BetaBernoulliThompson(3; ndraws=0)
    for (k, s, f) in ((1, 3, 7), (2, 6, 5), (3, 2, 2))
        for pol in (b1, b0)
            foreach(_ -> update_policy!(pol, k, 1), 1:s)
            foreach(_ -> update_policy!(pol, k, 0), 1:f)
        end
    end
    e1 = assignment_probabilities(b1; rng=StableRNG(3))
    e0 = assignment_probabilities(b0)
    @test maximum(abs.(e1 .- e0)) < 0.01
    @test_throws ArgumentError update_policy!(b1, 1, 0.5)
    @test_throws ArgumentError BetaBernoulliThompson(2; a=0.0)
    # closed form: θ₁ ~ U(0, 1), θ₂ ~ Beta(4, 1): P(θ₁ > θ₂) = ∫ x⁴ dx = 1/5
    c = BetaBernoulliThompson(2; ndraws=0)
    foreach(_ -> update_policy!(c, 2, 1), 1:3)
    @test assignment_probabilities(c) ≈ [0.2, 0.8] atol = 2e-3
end

@testset "top-two Thompson" begin
    base = GaussianThompson(3; noise_var=1.0)
    rng = StableRNG(5)
    for (k, μ) in ((1, 0.0), (2, 0.3), (3, 0.35)), _ in 1:6
        update_policy!(base, k, μ + randn(rng))
    end
    α = assignment_probabilities(base)
    tt = TopTwoThompson(base; beta=0.5)
    e = assignment_probabilities(tt)
    @test sum(e) ≈ 1
    # simulate the top-two procedure with draws from α (the "best arm" law)
    rng = StableRNG(9)
    cnt = zeros(3)
    N = 200_000
    for _ in 1:N
        i = DrSnow._ad_draw_arm(rng, α)
        if rand(rng) < 0.5
            cnt[i] += 1
        else
            j = i
            while j == i
                j = DrSnow._ad_draw_arm(rng, α)
            end
            cnt[j] += 1
        end
    end
    @test maximum(abs.(cnt ./ N .- e)) < 0.005
    # top-two spends more on the runner-up than Thompson sampling
    @test e[argmax(α)] < α[argmax(α)]
    @test_throws ArgumentError TopTwoThompson(base; beta=1.0)
    update_policy!(tt, 1, 1.0)
    @test tt.base.n[1] == 7
end

@testset "ε-greedy, softmax, UCB" begin
    p = EpsilonGreedy(3; epsilon=0.3)
    @test assignment_probabilities(p) ≈ fill(1 / 3, 3)        # all unobserved
    update_policy!(p, 1, 1.0)
    @test assignment_probabilities(p) ≈ [0.1, 0.45, 0.45]      # unobserved = best
    update_policy!(p, 2, 2.0)
    update_policy!(p, 3, 0.5)
    @test assignment_probabilities(p) ≈ [0.1, 0.8, 0.1]
    update_policy!(p, 1, 3.0)                                  # tie arm 1 and 2
    @test assignment_probabilities(p) ≈ [0.45, 0.45, 0.1]
    @test_throws ArgumentError EpsilonGreedy(2; epsilon=1.5)

    s = SoftmaxPolicy(2; temperature=0.5)
    update_policy!(s, 1, 1.0)
    update_policy!(s, 2, 0.0)
    @test assignment_probabilities(s) ≈ [exp(2), 1] ./ (exp(2) + 1)
    @test_throws ArgumentError SoftmaxPolicy(2; temperature=0)

    u = UCBPolicy(3; c=2.0)
    @test assignment_probabilities(u) ≈ fill(1 / 3, 3)
    update_policy!(u, 1, 1.0)
    update_policy!(u, 2, 0.0)
    @test assignment_probabilities(u) == [0.0, 0.0, 1.0]        # try arm 3 first
    update_policy!(u, 3, 0.0)
    idx = [1.0, 0.0, 0.0] .+ sqrt(2 * log(3))
    @test assignment_probabilities(u) == Float64.(idx .== maximum(idx))
    uf = UCBPolicy(3; floor=0.05)
    @test minimum(assignment_probabilities(uf)) >= 0.05 - 1e-12
    @test_throws ArgumentError update_policy!(u, 4, 1.0)
    @test_throws ArgumentError update_policy!(u, 1, Inf)
    @test n_arms(u) == 3
    @test occursin("UCB1", sprint(show, uf))
end

@testset "linear Thompson" begin
    p = LinearThompson(3, 2; prior_var=1.0, noise_var=0.5)
    @test_throws ArgumentError assignment_probabilities(p)
    @test_throws DimensionMismatch assignment_probabilities(p; context=[1.0])
    @test assignment_probabilities(p; context=[0.3, -1.0]) ≈ fill(1 / 3, 3) atol = 1e-12
    rng = StableRNG(21)
    for _ in 1:300
        x = randn(rng, 2)
        k = rand(rng, 1:3)
        μ = (0.0, x[1], -x[1])[k]
        update_policy!(p, k, μ + sqrt(0.5) * randn(rng); context=x)
    end
    @test_throws ArgumentError update_policy!(p, 1, 1.0)
    e_pos = assignment_probabilities(p; context=[2.0, 0.0])
    e_neg = assignment_probabilities(p; context=[-2.0, 0.0])
    @test argmax(e_pos) == 2 && argmax(e_neg) == 3
    # posterior of the arm mean at x: exact Bayesian linear regression
    z = [1.0, 0.5, 0.2]
    F = inv(p.A[2])
    m = dot(z, F * p.b[2])
    D = hcat([dot(z, (F * p.b[k])) .+ sqrt(0.5 * dot(z, inv(p.A[k]) * z)) .*
              randn(StableRNG(k), 300_000) for k in 1:3]...)
    @test maximum(abs.(assignment_probabilities(p; context=[0.5, 0.2]) .-
                       DrSnow._ad_prob_best_draws(D))) < 0.006
    @test isfinite(m)
    @test_throws ArgumentError LinearThompson(2, 0)
end
