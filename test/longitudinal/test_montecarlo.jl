# Monte Carlo coverage and double-robustness checks (reduced replication counts
# unless DRSNOW_SLOW_TESTS). The design has treatment–confounder feedback; the truth
# is simulated from the counterfactual data-generating process.

function lg_mc(reps, n; outcome, treatment, estimators, crossfit=false, seed=1)
    _, truth = lg_sim(StableRNG(0), 10; K=2, outcome=outcome, treatment=treatment)
    ψ1 = truth(always_treat(2); n_truth=300_000)
    ψ0 = truth(never_treat(2); n_truth=300_000)
    seeds = DrSnow.task_seeds(StableRNG(seed), reps)
    est = Dict(e => zeros(reps, 2) for e in estimators)
    cover = Dict(e => zeros(2) for e in estimators)
    for i in 1:reps
        rng = StableRNG(seeds[i])
        df, _ = lg_sim(rng, n; K=2, outcome=outcome, treatment=treatment)
        for e in estimators
            r = longitudinal_regime(df, :y, :a, :id, :t;
                                    regime=[always_treat(2), never_treat(2)],
                                    covariates=[:l], baseline=[:w], estimator=e,
                                    outcome_learner=OLSLearner(),
                                    treatment_learner=LogisticLearner(),
                                    crossfit=crossfit, n_folds=3, rng=rng)
            est[e][i, :] = coef(r)
            if e !== :gcomp
                ci = confint(r)
                cover[e] .+= (ci[:, 1] .<= [ψ1, ψ0] .<= ci[:, 2])
            end
        end
    end
    return (; ψ1, ψ0, est, cover)
end

@testset "Monte Carlo: coverage with correct parametric nuisances" begin
    reps = mc_reps(400, 60)
    mc = lg_mc(reps, 400; outcome=:linear, treatment=:linear,
               estimators=[:ipw, :tmle, :sdr, :gcomp])
    tol = lg_cover_tol(reps)
    for e in (:ipw, :tmle, :sdr)
        @test all(abs.(mc.cover[e] ./ reps .- 0.95) .< tol)
    end
    for e in (:ipw, :tmle, :sdr, :gcomp)
        bias = vec(mean(mc.est[e]; dims=1)) .- [mc.ψ1, mc.ψ0]
        mcse = vec(std(mc.est[e]; dims=1)) ./ sqrt(reps)
        @test all(abs.(bias) .< 4 .* mcse .+ 0.01)
    end
end

@testset "Monte Carlo: coverage with cross-fitting" begin
    reps = mc_reps(300, 40)
    mc = lg_mc(reps, 400; outcome=:linear, treatment=:linear, estimators=[:tmle, :sdr],
               crossfit=true, seed=2)
    tol = lg_cover_tol(reps)
    for e in (:tmle, :sdr)
        @test all(abs.(mc.cover[e] ./ reps .- 0.95) .< tol)
    end
end

@testset "Double robustness in a large sample" begin
    # One large sample per misspecified design: the single-robust estimator whose
    # model is wrong is biased by many standard errors, the doubly robust ones are not.
    n = 50_000
    function fit_all(outcome, treatment)
        df, truth = lg_sim(StableRNG(31), n; K=2, outcome=outcome, treatment=treatment)
        ψ = [truth(always_treat(2); n_truth=1_000_000),
             truth(never_treat(2); n_truth=1_000_000)]
        fits = Dict(e => longitudinal_regime(df, :y, :a, :id, :t;
                                             regime=[always_treat(2), never_treat(2)],
                                             covariates=[:l], baseline=[:w], estimator=e,
                                             outcome_learner=OLSLearner(),
                                             treatment_learner=LogisticLearner(),
                                             crossfit=false)
                    for e in (:gcomp, :ipw, :tmle, :sdr))
        return ψ, fits
    end
    ok(r, ψ) = all(abs.(coef(r) .- ψ) .< 4 .* stderror(r) .+ 0.01)
    # outcome regressions misspecified (treatment–covariate interaction), mechanism right
    ψ, f = fit_all(:interaction, :linear)
    @test ok(f[:ipw], ψ) && ok(f[:tmle], ψ) && ok(f[:sdr], ψ)
    @test maximum(abs.(coef(f[:gcomp]) .- ψ)) > 0.04
    # treatment mechanism misspecified (step in the covariate), outcome regressions right
    ψ, f = fit_all(:linear, :step)
    @test ok(f[:tmle], ψ) && ok(f[:sdr], ψ)
    @test maximum(abs.(coef(f[:gcomp]) .- ψ)) < 0.03
    @test maximum(abs.(coef(f[:ipw]) .- ψ) ./ stderror(f[:ipw])) > 3
end
