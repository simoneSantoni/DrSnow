# Adaptively weighted AIPW: reference values from the authors' code (Hadad et al.
# 2021), agreement across interfaces, contextual weighting (Zhan et al. 2021), errors.

@testset "arm values match the Hadad et al. reference code" begin
    d = ad_read_csv("hadad_data.csv")
    ref = ad_read_csv("hadad_reference.csv")
    for w in (:two_point, :constant_allocation, :uniform), m in (:running_mean, :none)
        r = adaptive_arm_values(d, :reward, :arm; probabilities=[:p1, :p2, :p3],
                                time=:t, floor_decay=0.7, weights=w, outcome_model=m,
                                reference=3)
        rr = ref[(ref.weights .== string(w)) .& (ref.outcome_model .== string(m)), :]
        @test nrow(rr) == 5
        for x in eachrow(rr)
            if startswith(x.term, "value(arm 3) -")      # reference code: last - other
                k = x.term[end - 1]
                j = findfirst(==("value(arm $k) - value(arm 3)"), coefnames(r))
                @test -coef(r)[j] ≈ x.estimate rtol = 1e-10
            else
                j = findfirst(==(x.term), coefnames(r))
                @test coef(r)[j] ≈ x.estimate rtol = 1e-10
            end
            @test stderror(r)[j] ≈ x.stderr rtol = 1e-10
        end
    end
end

@testset "interfaces, invariance and naive means" begin
    p = GaussianThompson(3; floor=1 / 3, floor_decay=0.7, burnin=15)
    log = run_adaptive_experiment(p, GaussianBandit([0.9, 1.0, 1.1]), 600;
                                  rng=StableRNG(10))
    r = adaptive_arm_values(log)
    @test coefnames(r) == ["value(arm 1)", "value(arm 2)", "value(arm 3)",
                           "value(arm 2) - value(arm 1)", "value(arm 3) - value(arm 1)"]
    @test r.weights === :two_point && nobs(r) == 600
    df = DataFrame(log)
    r2 = adaptive_arm_values(df, :outcome, :arm; probabilities=[:p1, :p2, :p3],
                             time=:t, floor_decay=0.7)
    @test coef(r2) ≈ coef(r) && vcov(r2) ≈ vcov(r)
    shuffled = df[randperm(StableRNG(1), nrow(df)), :]
    r3 = adaptive_arm_values(shuffled, :outcome, :arm; probabilities=[:p1, :p2, :p3],
                             time=:t, floor_decay=0.7)
    @test coef(r3) ≈ coef(r) && vcov(r3) ≈ vcov(r)
    # contrasts are linear combinations with the joint covariance
    L = [-1 1 0; -1 0 1]
    @test coef(r)[4:5] ≈ L * coef(r)[1:3]
    @test vcov(r)[4:5, 4:5] ≈ L * vcov(r)[1:3, 1:3] * L'
    rp = adaptive_arm_values(log; contrasts=:pairwise)
    @test length(coef(rp)) == 6
    @test length(coef(adaptive_arm_values(log; contrasts=:none))) == 3
    # constant allocation = sqrt(e) weights; contextual weights reduce to it here
    rc = adaptive_arm_values(log; weights=:constant_allocation, contrasts=:none)
    Γ = r.scores
    for k in 1:3
        h = sqrt.(log.probabilities[:, k])
        @test coef(rc)[k] ≈ sum(h .* Γ[:, k]) / sum(h)
    end
    rx = adaptive_arm_values(log; weights=:contextual, contrasts=:none)
    @test coef(rx) ≈ coef(rc) && vcov(rx) ≈ vcov(rc)
    pv = adaptive_policy_value(log, (1, 2, 3); weights=:constant_allocation)
    @test coef(pv) ≈ coef(rc) && vcov(pv) ≈ vcov(rc)
    # IPW scores
    ri = adaptive_arm_values(log; weights=:uniform, outcome_model=:none, contrasts=:none)
    @test coef(ri)[2] ≈ mean((log.arms .== 2) .* log.outcomes ./ log.probabilities[:, 2])
    # naive means
    nm = naive_arm_means(log)
    @test coef(nm)[1] ≈ mean(log.outcomes[log.arms .== 1])
    n1 = count(==(1), log.arms)
    @test stderror(nm)[1] ≈ std(log.outcomes[log.arms .== 1]; corrected=false) / sqrt(n1)
    @test coef(naive_arm_means(df, :outcome, :arm)) ≈ coef(nm)
    @test occursin("not valid", method_name(nm))
    @test occursin("under-cover", sprint(show, MIME"text/plain"(),
                                         adaptive_arm_values(log; weights=:uniform)))
end

@testset "errors" begin
    d = ad_read_csv("hadad_data.csv")
    P = [:p1, :p2, :p3]
    @test_throws ArgumentError adaptive_arm_values(d, :reward, :arm; probabilities=P)
    @test_throws ArgumentError adaptive_arm_values(d, :reward, :arm; time=:t)
    @test_throws ArgumentError adaptive_arm_values(d, :reward, :arm; probabilities=P,
                                                   time=:t)          # floor_decay
    @test_throws ArgumentError adaptive_arm_values(d, :reward, :arm; probabilities=P,
                                                   time=:t, weights=:magic)
    bad = copy(d)
    bad.p1 .= 0.5
    @test_throws ArgumentError adaptive_arm_values(bad, :reward, :arm; probabilities=P,
                                                   time=:t, floor_decay=0.7)
    dup = copy(d)
    dup.t[2] = 1
    @test_throws ArgumentError adaptive_arm_values(dup, :reward, :arm; probabilities=P,
                                                   time=:t, floor_decay=0.7)
    arm4 = copy(d)
    arm4.arm = Vector{Int}(arm4.arm)
    arm4.arm[5] = 4
    @test_throws ArgumentError adaptive_arm_values(arm4, :reward, :arm;
                                                   probabilities=P, time=:t,
                                                   floor_decay=0.7)
    @test_throws ArgumentError adaptive_arm_values(d, :reward, :arm; probabilities=P,
                                                   time=:t, floor_decay=0.7,
                                                   outcome_model=RidgeLearner())
    # batched probabilities must be constant within a batch unless probability_fn
    b = copy(d)
    b.batch = fill(1, nrow(b))
    @test_throws ArgumentError adaptive_arm_values(b, :reward, :arm; probabilities=P,
                                                   time=:t, batch=:batch,
                                                   floor_decay=0.7)
    log = run_adaptive_experiment(EpsilonGreedy(2; epsilon=0.2),
                                  BernoulliBandit([0.3, 0.6]), 100; rng=StableRNG(1))
    @test_throws ArgumentError adaptive_policy_value(log, 1; weights=:two_point)
    @test_throws ArgumentError adaptive_policy_value(log, 3)
    @test_throws DimensionMismatch adaptive_policy_value(log, [1, 2])
    @test_throws ArgumentError adaptive_policy_value(log, x -> 5)
    @test_throws ArgumentError adaptive_policy_value(log, fill(0.3, 100, 2))
    @test_throws ArgumentError adaptive_arm_values(log; contrasts=:all)
    @test_throws ArgumentError naive_arm_means(log; reference=5)
end

@testset "contextual adaptive weighting (Zhan et al. 2021)" begin
    mf(x) = [0.0, x[1], -x[1], 0.5 * x[2]]
    env = ContextualBandit(mf, rng -> randn(rng, 2), 4)
    p = LinearThompson(4, 2; floor=0.25, floor_decay=0.5, burnin=100)
    log = run_adaptive_experiment(p, env, 500; batch_size=50, rng=StableRNG(3))
    best = x -> argmax(mf(x))
    r = adaptive_policy_value(log, (best, 1); names=["oracle", "arm 1"],
                              contrasts=:reference)
    @test coefnames(r) == ["value(oracle)", "value(arm 1)",
                           "value(arm 1) - value(oracle)"]
    # brute force: h[t, s] = h_t(X_s); Q = Σ_t h_t(X_t)/Z(X_t) Γ_t;
    # V = Σ_t (B_t - Σ_s h_t(X_s)/Z(X_s) B_s)²
    d = DrSnow._ad_data(log)
    T = 500
    Π = DrSnow._ad_policy_matrix(best, d, "test")
    Γ = DrSnow._ad_aipw_scores(d, DrSnow._ad_running_mean(d))
    h = [1 / sqrt(sum(Π[s, :] .^ 2 ./ d.probfn(d.batch[t], d.X[s, :])))
         for t in 1:T, s in 1:T]
    Z = vec(sum(h; dims=1))
    B = [h[t, t] / Z[t] * dot(Π[t, :], Γ[t, :]) for t in 1:T]
    @test coef(r)[1] ≈ sum(B)
    @test stderror(r)[1] ≈ sqrt(sum(abs2, B .- (h ./ Z') * B))
    # two-point weights are refused for contextual designs
    @test_throws ArgumentError adaptive_arm_values(log)
    # the table interface with probability_fn reproduces the log method
    df = DataFrame(log)
    snaps = log.snapshots
    starts = log.batch_start
    pf = (t, x) -> assignment_probabilities(snaps[findfirst(==(t), starts)], t;
                                            context=x)
    r2 = adaptive_policy_value(df, :outcome, :arm, (best, 1); probabilities=[:p1, :p2,
                               :p3, :p4], time=:t, batch=:batch,
                               covariates=[:x1, :x2], probability_fn=pf,
                               names=["oracle", "arm 1"], contrasts=:reference)
    @test coef(r2) ≈ coef(r) && vcov(r2) ≈ vcov(r)
    @test_throws ArgumentError adaptive_policy_value(df, :outcome, :arm, 1;
                                                     probabilities=[:p1, :p2, :p3,
                                                                    :p4], time=:t,
                                                     batch=:batch)
    # learner outcome model (sequential refits) and PolicyTree targets
    rl = adaptive_policy_value(log, best; outcome_model=RidgeLearner(), n_blocks=5,
                               rng=StableRNG(1))
    @test isfinite(coef(rl)[1]) && abs(coef(rl)[1] - coef(r)[1]) < 0.2
    Γs = bandit_dr_scores(log)
    @test Γs ≈ Γ
    tree = policy_tree(Γs, log.contexts; depth=1, actions=1:4, covariates=[:x1, :x2])
    rt = adaptive_policy_value(log, tree)
    @test isfinite(coef(rt)[1])
    rs = adaptive_policy_value(log, x -> fill(0.25, 4); weights=:constant_allocation)
    @test isfinite(stderror(rs)[1])
end
