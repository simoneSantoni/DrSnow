# The four estimators: closed forms, invariances, contrasts, MSM, bootstrap, panel
# method, clustering, cross-fitting with ML learners.

const LG_DF, LG_TRUTH = lg_sim(StableRNG(10), 500; K=3)
lg_fit(; kws...) = longitudinal_regime(LG_DF, :y, :a, :id, :t; covariates=[:l],
                                       baseline=[:w], outcome_learner=OLSLearner(),
                                       treatment_learner=LogisticLearner(), kws...)

@testset "IPW closed form and g-computation recursion" begin
    r = lg_fit(; regime=always_treat(3), estimator=:ipw, crossfit=false)
    d = DrSnow._lg_data(LG_DF, :y, :a, :id, :t; covariates=[:l], baseline=[:w],
                        cluster=nothing, context="t")
    # propensities by hand: logistic regression of a_t on the history
    W = zeros(d.n)
    cum = ones(d.n)
    follow = trues(d.n)
    for t in 1:3
        X = DrSnow._lg_design(d, t, nothing; current=false)
        p = fitpredict_proba(LogisticLearner(), X, d.A[:, t], X)
        cum .*= p
        follow .&= d.A[:, t] .== 1
    end
    W .= follow ./ max.(cum, 0.01)
    X1 = DrSnow._lg_design(d, 1, nothing; current=false)
    @test r.propensities[:, 1] ≈ fitpredict_proba(LogisticLearner(), X1, d.A[:, 1], X1)
    @test r.weights[1][:, 3] ≈ W
    @test coef(r)[1] ≈ sum(W .* d.y) / sum(W)
    ψ = coef(r)[1]
    eif = W .* (d.y .- ψ) ./ mean(W)
    @test stderror(r)[1] ≈ sqrt(sum(abs2, eif)) / d.n
    rh = lg_fit(; regime=always_treat(3), estimator=:ipw, crossfit=false, normalize=false)
    @test coef(rh)[1] ≈ mean(W .* d.y)
    @test isempty(r.outcome_regressions[1])
    # g-computation by hand (OLS on the scaled outcome, prediction at the regime)
    g = lg_fit(; regime=always_treat(3), estimator=:gcomp, crossfit=false, bootstrap=0)
    lo, hi = extrema(d.y)
    ys = (d.y .- lo) ./ (hi - lo)
    tgt = copy(ys)
    Q1 = nothing
    for t in 3:-1:1
        X = DrSnow._lg_design(d, t, nothing; current=true)
        Xd = DrSnow._lg_design(d, t, ones(d.n, 3); current=true)
        pred = clamp.(fitpredict(OLSLearner(), X, tgt, Xd), 1e-4, 1 - 1e-4)
        tgt = pred
        t == 1 && (Q1 = pred)
    end
    @test coef(g)[1] ≈ mean(Q1) * (hi - lo) + lo
    @test g.outcome_regressions[1][:, 1] ≈ Q1 .* (hi - lo) .+ lo
    @test g.outcome_regressions[1][:, 4] ≈ d.y
    @test_throws ArgumentError vcov(g)
    @test_throws ArgumentError stderror(g)
    @test all(ismissing, tidy(g).std_error)
    @test occursin("refit with `bootstrap", sprint(show, MIME"text/plain"(), g))
end

@testset "TMLE targeting solves the score equations" begin
    r = lg_fit(; regime=[always_treat(3), never_treat(3)], estimator=:tmle, crossfit=false)
    lo, hi = r.outcome_range
    for j in 1:2
        Q = (r.outcome_regressions[j] .- lo) ./ (hi - lo)
        W = r.weights[j]
        for t in 1:3
            @test abs(sum(W[:, t] .* (Q[:, t + 1] .- Q[:, t]))) < 1e-8
        end
        # the EIF has mean zero and the estimate is the mean of Q*_1
        @test abs(mean(r.eif[:, j])) < 1e-8
        @test coef(r)[j] ≈ mean(r.outcome_regressions[j][:, 1])
    end
    @test stderror(r) ≈ sqrt.(vec(sum(abs2, r.eif; dims=1))) ./ r.n
    @test vcov(r) ≈ r.eif' * r.eif ./ r.n^2
    @test r.if_vcov == vcov(r)
    @test estimand(r) != "" && occursin("targeted", method_name(r))
    @test nobs(r) == 500 && dof_residual(r) == Inf
    @test glance(r).n_clusters[1] === missing
    s = sprint(show, MIME"text/plain"(), r)
    @test occursin("No cross-fitting", s) && occursin("always treat", s)
    # the fluctuation solver
    y = [0.2, 0.9, 0.5, 0.1]
    q = [0.3, 0.6, 0.5, 0.2]
    w = [1.0, 2.0, 0.0, 1.5]
    ε = DrSnow._lg_fluctuate(y, q, w)
    expit(x) = 1 / (1 + exp(-x))
    @test abs(sum(w .* (y .- expit.(log.(q ./ (1 .- q)) .+ ε)))) < 1e-10
    @test DrSnow._lg_fluctuate(y, q, zeros(4)) == 0.0
end

@testset "SDR: estimate is the mean of the influence function; agreement" begin
    rs = lg_fit(; regime=[always_treat(3), never_treat(3)], estimator=:sdr, crossfit=false)
    @test abs(mean(rs.eif[:, 1])) < 1e-12
    both = [always_treat(3), never_treat(3)]
    rt = lg_fit(; regime=both, estimator=:tmle, crossfit=false)
    ri = lg_fit(; regime=both, estimator=:ipw, crossfit=false)
    rg = lg_fit(; regime=both, estimator=:gcomp, crossfit=false)
    truth = [LG_TRUTH(always_treat(3)), LG_TRUTH(never_treat(3))]
    for r in (rs, rt, ri)
        @test all(abs.(coef(r) .- truth) .< 3 .* stderror(r) .+ 0.05)
    end
    @test all(abs.(coef(rg) .- truth) .< 0.3)
    @test maximum(abs.(coef(rs) .- coef(rt))) < 0.15
end

@testset "Invariance, folds, repetitions, panel method, clusters" begin
    r = lg_fit(; regime=[always_treat(3), never_treat(3)], estimator=:sdr, n_folds=3,
               rng=StableRNG(4))
    perm = randperm(StableRNG(5), nrow(LG_DF))
    r2 = longitudinal_regime(LG_DF[perm, :], :y, :a, :id, :t; regime=[always_treat(3),
                             never_treat(3)], covariates=[:l], baseline=[:w],
                             outcome_learner=OLSLearner(),
                             treatment_learner=LogisticLearner(), estimator=:sdr,
                             n_folds=3, rng=StableRNG(4))
    @test coef(r2) ≈ coef(r) && vcov(r2) ≈ vcov(r)
    # threads do not change the result
    r3 = lg_fit(; regime=[always_treat(3), never_treat(3)], estimator=:sdr, n_folds=3,
                rng=StableRNG(4), parallel=false)
    @test coef(r3) == coef(r)
    # user folds: a vector over units, or a column
    F = crossfit_folds(500, 3; rng=StableRNG(4))
    r4 = lg_fit(; regime=always_treat(3), estimator=:sdr, folds=F, rng=StableRNG(4))
    @test r4.folds == reshape(F, :, 1)
    df = copy(LG_DF)
    df.fold = repeat(vec(F); inner=3)
    r5 = longitudinal_regime(df, :y, :a, :id, :t; regime=always_treat(3),
                             covariates=[:l], baseline=[:w], outcome_learner=OLSLearner(),
                             treatment_learner=LogisticLearner(), estimator=:sdr,
                             folds=:fold, rng=StableRNG(4))
    @test coef(r5) == coef(r4)
    @test_throws DimensionMismatch lg_fit(; regime=always_treat(3), folds=F[1:10])
    # repetitions: median aggregation
    r6 = lg_fit(; regime=always_treat(3), estimator=:tmle, n_folds=3, n_rep=3,
                rng=StableRNG(6))
    @test size(r6.all_coef) == (1, 3)
    @test coef(r6)[1] == median(r6.all_coef[1, :])
    @test occursin("3 repetitions", sprint(show, MIME"text/plain"(), r6))
    # TreatmentPanel method
    panel = TreatmentPanel(LG_DF, :y, :a, :id, :t, [:l])
    r7 = longitudinal_regime(panel; regime=always_treat(3), baseline=[:w],
                             outcome_learner=OLSLearner(),
                             treatment_learner=LogisticLearner(), crossfit=false)
    @test coef(r7) == coef(lg_fit(; regime=always_treat(3), crossfit=false))
    # clusters: folds grouped, t reference, cluster sums in the variance
    df.g = repeat(1:50; inner=30)
    rc = longitudinal_regime(df, :y, :a, :id, :t; regime=always_treat(3),
                             covariates=[:l], baseline=[:w], outcome_learner=OLSLearner(),
                             treatment_learner=LogisticLearner(), estimator=:tmle,
                             n_folds=3, cluster=:g, rng=StableRNG(7))
    @test rc.n_clusters == 50 && dof_residual(rc) == 49
    @test glance(rc).n_clusters[1] == 50
    for k in 1:3
        for g in 1:50
            @test allunique(rc.folds[rc.cluster .== g, 1][1:1]) # single fold per cluster
            @test length(unique(rc.folds[rc.cluster .== g, 1])) == 1
        end
    end
    S = [sum(rc.eif[rc.cluster .== g, 1]) for g in 1:50]
    @test stderror(rc)[1] ≈ sqrt(sum(abs2, S)) / rc.n
end

@testset "Contrasts and MSM projections" begin
    rs = [StaticRegime([0, 0, 0]), StaticRegime([1, 0, 0]), StaticRegime([1, 1, 0]),
          StaticRegime([1, 1, 1])]
    r = lg_fit(; regime=rs, estimator=:tmle, crossfit=false)
    c = regime_contrast(r)
    @test coefnames(c) == ["static(1,0,0) vs static(0,0,0)",
                           "static(1,1,0) vs static(0,0,0)",
                           "static(1,1,1) vs static(0,0,0)"]
    @test coef(c) ≈ coef(r)[2:4] .- coef(r)[1]
    A = [-1.0 1.0 0.0 0.0; -1.0 0.0 1.0 0.0; -1.0 0.0 0.0 1.0]
    @test vcov(c) ≈ A * vcov(r) * A'
    @test c.eif ≈ r.eif * A'
    c2 = regime_contrast(r; reference="static(1,1,1)")
    @test coef(c2) ≈ coef(r)[1:3] .- coef(r)[4]
    @test occursin("Reference regime: static(1,1,1)", sprint(show, MIME"text/plain"(), c2))
    @test_throws ArgumentError regime_contrast(r; reference=9)
    @test_throws ArgumentError regime_contrast(r; type=:foo)
    @test_throws ArgumentError regime_contrast(lg_fit(; regime=always_treat(3),
                                                      crossfit=false))
    # ratios need positive means; the outcome here is positive on average
    @test all(>(0), coef(r))
    lr = regime_contrast(r; type=:ratio)
    @test coef(lr) ≈ log.(coef(r)[2:4] ./ coef(r)[1])
    J = hcat(fill(-1 / coef(r)[1], 3), Diagonal(1 ./ coef(r)[2:4]))
    @test vcov(lr) ≈ J * vcov(r) * J'
    @test_throws ArgumentError regime_contrast(r; type=:odds_ratio)   # means > 1
    # saturated MSM = the contrasts; linear MSM = least-squares projection
    X = [1 0 0 0; 1 1 0 0; 1 0 1 0; 1 0 0 1]
    m = regime_msm(r, X)
    @test coef(m)[2:4] ≈ coef(c) && vcov(m)[2:4, 2:4] ≈ vcov(c)
    Xl = [1 0; 1 1; 1 2; 1 3]
    ml = regime_msm(r, Xl; names=["b0", "b1"])
    P = (Xl' * Xl) \ Xl'
    @test coef(ml) ≈ P * coef(r) && vcov(ml) ≈ P * vcov(r) * P'
    @test coefnames(ml) == ["b0", "b1"] && estimand(ml) != ""
    w = [4.0, 1.0, 1.0, 2.0]
    mw = regime_msm(r, Xl; weights=w)
    Pw = (Xl' * Diagonal(w) * Xl) \ (Xl' * Diagonal(w))
    @test coef(mw) ≈ Pw * coef(r)
    @test_throws ArgumentError regime_msm(r, [1 1; 1 1; 1 1; 1 1])
    @test_throws ArgumentError regime_msm(r, Xl[1:3, :])
    @test_throws DimensionMismatch regime_msm(r, Xl; names=["a"])
    # binary outcome: odds ratios and log risk ratios
    dfb, _ = lg_sim(StableRNG(11), 400; K=2, y_binary=true)
    rb = longitudinal_regime(dfb, :y, :a, :id, :t;
                             regime=[never_treat(2), always_treat(2)],
                             covariates=[:l], baseline=[:w],
                             outcome_learner=FractionalLogitLearner(),
                             treatment_learner=LogisticLearner(), estimator=:tmle,
                             crossfit=false)
    @test rb.outcome_range == (0.0, 1.0)
    @test all(v -> 0 < v < 1, coef(rb))
    or = regime_contrast(rb; type=:odds_ratio)
    ψ = coef(rb)
    @test coef(or)[1] ≈ log(ψ[2] / (1 - ψ[2])) - log(ψ[1] / (1 - ψ[1]))
    @test estimand(or) == "log odds ratio of mean counterfactual outcomes"
end

@testset "Bootstrap" begin
    r = lg_fit(; regime=[always_treat(3), never_treat(3)], estimator=:gcomp,
               crossfit=false, bootstrap=30, rng=StableRNG(12))
    @test size(r.bootstrap_coef) == (30, 2)
    @test vcov(r) ≈ cov(r.bootstrap_coef)
    @test r.if_vcov === nothing
    @test all(stderror(r) .> 0)
    @test occursin("bootstrap with 30", sprint(show, MIME"text/plain"(), r))
    c = regime_contrast(r)
    @test coef(c)[1] ≈ coef(r)[2] - coef(r)[1]
    @test vcov(c)[1, 1] ≈ var(r.bootstrap_coef[:, 2] .- r.bootstrap_coef[:, 1])
    @test c.eif === nothing && size(c.bootstrap_coef) == (30, 1)
    # bootstrap with the influence-function estimators keeps the IF covariance
    rt = lg_fit(; regime=always_treat(3), estimator=:tmle, crossfit=false, bootstrap=20,
                rng=StableRNG(13))
    @test rt.if_vcov !== nothing && vcov(rt) ≈ cov(rt.bootstrap_coef)
    @test abs(sqrt(rt.if_vcov[1, 1]) / stderror(rt)[1] - 1) < 0.6
    # reproducible
    r2 = lg_fit(; regime=[always_treat(3), never_treat(3)], estimator=:gcomp,
                crossfit=false, bootstrap=30, rng=StableRNG(12))
    @test r2.bootstrap_coef == r.bootstrap_coef
end

@testset "Dynamic regimes and ML learners with cross-fitting" begin
    dyn = DynamicRegime(h -> h.covariates[end].l > 0 ? 1 : 0; label="treat if l > 0")
    r = lg_fit(; regime=[dyn, never_treat(3)], estimator=:tmle, crossfit=false)
    truth = LG_TRUTH(dyn)
    @test abs(coef(r)[1] - truth) < 3 * stderror(r)[1] + 0.05
    d = DrSnow._lg_data(LG_DF, :y, :a, :id, :t; covariates=[:l], baseline=[:w],
                        cluster=nothing, context="t")
    @test r.followers[1][:, 1] == (d.A[:, 1] .== (d.L[1][:, 1] .> 0))
    rw = regime_weights(r)
    @test nrow(rw) == 6 && rw.n_following[3] == count(r.followers[1][:, 3])
    @test all(rw.share_following .<= 1) && all(rw.weight_max .>= rw.weight_mean)
    @test rw.weight_max[1] ≈ maximum(r.weights[1][r.followers[1][:, 1], 1])
    # ML learners, cross-fitted, default estimator
    rm = longitudinal_regime(LG_DF, :y, :a, :id, :t;
                             regime=[always_treat(3), never_treat(3)],
                             covariates=[:l], baseline=[:w], n_folds=3, rng=StableRNG(14))
    @test rm.estimator === :sdr
    @test rm.learners == [:outcome => "LassoLearner",
                          :treatment => "PenalizedLogisticLearner"]
    truth = [LG_TRUTH(always_treat(3)), LG_TRUTH(never_treat(3))]
    @test all(abs.(coef(rm) .- truth) .< 4 .* stderror(rm) .+ 0.1)
    # SDR targets are unbounded: a bounded-target learner is rejected with a clear error
    @test_throws ArgumentError lg_fit(; regime=always_treat(3), estimator=:sdr,
                                      crossfit=false,
                                      outcome_learner=FractionalLogitLearner())
    # outcome_bounds: the scaling changes nothing about the fit, only the scale
    lo, hi = extrema(skipmissing(LG_DF.y))
    rb = lg_fit(; regime=always_treat(3), estimator=:ipw, crossfit=false,
                outcome_bounds=(lo - 1, hi + 1))
    ri = lg_fit(; regime=always_treat(3), estimator=:ipw, crossfit=false)
    @test coef(rb) ≈ coef(ri) && stderror(rb) ≈ stderror(ri)
end
