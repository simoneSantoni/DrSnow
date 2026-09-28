# Off-policy evaluation from logged bandit data and DR scores for policy learning.

function ad_logged_data(rng, n)
    x1 = randn(rng, n)
    x2 = randn(rng, n)
    # logging policy: softmax in x1 over three actions labelled 1, 2, 3
    P = hcat(fill(1.0, n), exp.(0.5 .* x1), exp.(-0.5 .* x1))
    P ./= sum(P; dims=2)
    a = [DrSnow._ad_draw_arm(rng, P[i, :]) for i in 1:n]
    μ = hcat(zeros(n), x1, 0.5 .* x2)
    y = [μ[i, a[i]] for i in 1:n] .+ randn(rng, n)
    p = [P[i, a[i]] for i in 1:n]
    return DataFrame(x1=x1, x2=x2, a=a, y=y, p=p), μ
end

@testset "off-policy value: closed forms" begin
    df, μ = ad_logged_data(StableRNG(1), 800)
    pol = x -> x[1] > 0 ? 2 : 1
    Π = DrSnow._ad_policy_matrix(pol, Matrix(df[:, [:x1, :x2]]), [:x1, :x2],
                                 ["1", "2", "3"], 800, "t")
    w = [Π[i, df.a[i]] / df.p[i] for i in 1:800]
    r = off_policy_value(df, :y, :a, pol; propensity=:p, covariates=[:x1, :x2],
                         method=:ipw)
    @test coef(r)[1] ≈ mean(w .* df.y)
    @test stderror(r)[1] ≈ std(w .* df.y; corrected=false) / sqrt(800)
    s = off_policy_value(df, :y, :a, pol; propensity=:p, covariates=[:x1, :x2],
                         method=:snipw)
    θ = sum(w .* df.y) / sum(w)
    @test coef(s)[1] ≈ θ
    @test stderror(s)[1] ≈ sqrt(sum((w .* (df.y .- θ)) .^ 2)) / sum(w)
    folds = repeat(1:5, 160)
    dr = off_policy_value(df, :y, :a, pol; propensity=:p, covariates=[:x1, :x2],
                          outcome_learner=OLSLearner(), folds=folds, reference=1)
    @test coefnames(dr) == ["value(policy)", "value(reference)",
                            "value(policy) - value(reference)"]
    @test coef(dr)[3] ≈ coef(dr)[1] - coef(dr)[2]
    @test vcov(dr)[3, 3] ≈ vcov(dr)[1, 1] + vcov(dr)[2, 2] - 2vcov(dr)[1, 2]
    Γ = bandit_dr_scores(df, :y, :a; propensity=:p, covariates=[:x1, :x2],
                         outcome_learner=OLSLearner(), folds=folds)
    @test coef(dr)[1] ≈ mean(sum(Π .* Γ; dims=2))
    @test coef(dr)[2] ≈ mean(Γ[:, 1])
    # row-order invariance with fixed folds
    perm = randperm(StableRNG(2), 800)
    dr2 = off_policy_value(df[perm, :], :y, :a, pol; propensity=:p,
                           covariates=[:x1, :x2], outcome_learner=OLSLearner(),
                           folds=folds[perm], reference=1)
    @test coef(dr2) ≈ coef(dr) && vcov(dr2) ≈ vcov(dr)
    # policy learning hand-off
    tree = policy_tree(Γ, Matrix(df[:, [:x1, :x2]]); depth=1, actions=1:3,
                       covariates=[:x1, :x2])
    @test tree.covariates[tree.nodes[1].var] == :x1
    @test off_policy_value(df, :y, :a, tree; propensity=:p, covariates=[:x1, :x2],
                           outcome_learner=OLSLearner(), folds=folds).coef[1] > 0.2
    # policies as labels, matrices and probability functions
    r1 = off_policy_value(df, :y, :a, 2; propensity=:p, method=:ipw)
    r1b = off_policy_value(df, :y, :a, fill(2, 800); propensity=:p, method=:ipw)
    r1c = off_policy_value(df, :y, :a, x -> [0.0, 1.0, 0.0]; propensity=:p,
                           method=:ipw)
    @test coef(r1) ≈ coef(r1b) ≈ coef(r1c)
    # errors
    @test_throws ArgumentError off_policy_value(df, :y, :a, 1; method=:ipw)
    @test_throws ArgumentError off_policy_value(df, :y, :a, 1; propensity=:p)
    @test_throws ArgumentError off_policy_value(df, :y, :a, 1; propensity=:p,
                                                method=:magic)
    @test_throws ArgumentError off_policy_value(df, :y, :a, 7; propensity=:p,
                                                method=:ipw)
    bad = copy(df)
    bad.p[1] = 0.0
    @test_throws ArgumentError off_policy_value(bad, :y, :a, 1; propensity=:p,
                                                method=:ipw)
end
