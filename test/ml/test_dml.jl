ml_logistic(x) = 1 / (1 + exp(-x))

function ml_plr_data(rng, n; θ=0.5, G=0)
    X = randn(rng, n, 5)
    a = zeros(n)
    b = zeros(n)
    cl = zeros(Int, n)
    if G > 0
        cl = repeat(1:G, inner=cld(n, G))[1:n]
        ca = randn(rng, G)
        cb = randn(rng, G)
        a = ca[cl]
        b = cb[cl]
    end
    d = 0.8 .* X[:, 1] .+ 0.4 .* X[:, 3] .+ randn(rng, n) .+ b
    y = θ .* d .+ X[:, 1] .+ 0.5 .* X[:, 2] .^ 2 .+ randn(rng, n) .+ a
    df = DataFrame(X, [:x1, :x2, :x3, :x4, :x5])
    df.d = d
    df.y = y
    df.g = cl
    return df
end

function ml_irm_data(rng, n; hetero=true)
    X = randn(rng, n, 3)
    m = ml_logistic.(0.5 .* X[:, 1] .- 0.5 .* X[:, 2])
    d = Float64.(rand(rng, n) .< m)
    τ = hetero ? 1.0 .+ 0.5 .* X[:, 2] : fill(1.0, n)
    y = d .* τ .+ X[:, 1] .+ 0.5 .* X[:, 3] .+ randn(rng, n)
    df = DataFrame(X, [:x1, :x2, :x3])
    df.d = d
    df.y = y
    return df
end

function ml_pliv_data(rng, n)
    X = randn(rng, n, 3)
    z = 0.5 .* X[:, 1] .+ randn(rng, n)
    u = randn(rng, n)
    d = 0.7 .* z .+ 0.5 .* X[:, 1] .+ u .+ 0.5 .* randn(rng, n)
    y = 0.6 .* d .+ X[:, 1] .- 0.5 .* X[:, 2] .+ 0.8 .* u .+ randn(rng, n)
    df = DataFrame(X, [:x1, :x2, :x3])
    df.z = z
    df.d = d
    df.y = y
    return df
end

function ml_iivm_data(rng, n; one_sided=false)
    X = randn(rng, n, 3)
    z = Float64.(rand(rng, n) .< ml_logistic.(0.4 .* X[:, 1]))
    v = randn(rng, n)
    d = one_sided ? z .* Float64.((0.5 .+ 0.4 .* X[:, 2] .+ v) .> 0) :
        Float64.((0.2 .+ 1.2 .* z .+ 0.4 .* X[:, 2] .+ v) .> 0.8)
    y = 0.8 .* d .+ X[:, 1] .+ 0.5 .* X[:, 3] .+ 0.6 .* v .+ randn(rng, n)
    df = DataFrame(X, [:x1, :x2, :x3])
    df.z = z
    df.d = d
    df.y = y
    return df
end

function ml_did_panel(rng, n; att=1.5)
    X = randn(rng, n, 3)
    dg = Float64.(rand(rng, n) .< ml_logistic.(-0.3 .+ 0.6 .* X[:, 1] .- 0.4 .* X[:, 2]))
    α = 0.5 .* X[:, 1] .+ randn(rng, n)
    y0 = α .+ X[:, 2] .+ randn(rng, n)
    y1 = α .+ X[:, 2] .+ 1.0 .+ 0.8 .* X[:, 1] .+ 0.4 .* X[:, 3] .+ att .* dg .+
         randn(rng, n)
    pre = DataFrame(id=1:n, t=0, y=y0, d=dg, x1=X[:, 1], x2=X[:, 2], x3=X[:, 3])
    post = DataFrame(id=1:n, t=1, y=y1, d=dg, x1=X[:, 1], x2=X[:, 2], x3=X[:, 3])
    return vcat(pre, post)
end

function ml_did_rcs(rng, n; att=1.5)
    X = randn(rng, n, 3)
    dc = Float64.(rand(rng, n) .< ml_logistic.(-0.3 .+ 0.6 .* X[:, 1] .- 0.4 .* X[:, 2]))
    tc = Float64.(rand(rng, n) .< 0.5)
    y = 0.5 .* X[:, 1] .+ X[:, 2] .+ 0.3 .* dc .+
        tc .* (1.0 .+ 0.8 .* X[:, 1] .+ 0.4 .* X[:, 3] .+ att .* dc) .+ randn(rng, n)
    return DataFrame(x1=X[:, 1], x2=X[:, 2], x3=X[:, 3], d=dc, post=tc, y=y)
end

"""Coverage, mean SE and empirical SD of `fit(rng)` estimates of `truth`."""
function ml_mc(fit, truth, reps, seed)
    est = zeros(reps)
    se = zeros(reps)
    cover = 0
    for r in 1:reps
        res = fit(StableRNG(seed + r))
        est[r] = coef(res)[1]
        se[r] = stderror(res)[1]
        ci = confint(res)
        cover += ci[1, 1] <= truth <= ci[1, 2]
    end
    return (coverage=cover / reps, bias=mean(est) - truth, se_ratio=mean(se) / std(est))
end

function ml_check_mc(mc, reps, name)
    @info "Monte Carlo ($name, $reps reps)" mc.coverage mc.bias mc.se_ratio
    @test abs(mc.coverage - 0.95) <= ml_cover_tol(reps)
    @test 0.75 <= mc.se_ratio <= 1.3
end

const ML_XS = [:x1, :x2, :x3, :x4, :x5]
const ML_OLS = OLSLearner()
const ML_LOGIT = LogisticLearner()

@testset "DML estimators" begin
    @testset "PLR basics, reproducibility and invariance" begin
        df = ml_plr_data(StableRNG(11), 400)
        kw = (covariates=ML_XS, outcome_learner=ML_OLS, treatment_learner=ML_OLS)
        r = dml_plr(df, :y, :d; kw..., n_rep=3, rng=StableRNG(1))
        @test r isa DMLEstimate
        @test coefnames(r) == ["d"] && nobs(r) == 400 && dof_residual(r) == Inf
        @test size(r.psi) == (400, 3, 1) && size(r.folds) == (400, 3)
        @test size(r.predictions[:ml_l]) == (400, 3, 1)
        @test abs(sum(r.psi[:, 1, 1])) < 1e-8
        @test coef(r)[1] == median(r.all_coef[1, :])
        r2 = dml_plr(df, :y, :d; kw..., n_rep=3, rng=StableRNG(1), parallel=false)
        @test coef(r2) == coef(r) && vcov(r2) == vcov(r)
        # CV-lasso learners use the task seeds: identical with and without threads
        rl = dml_plr(df, :y, :d; covariates=ML_XS, rng=StableRNG(2), parallel=true)
        rl2 = dml_plr(df, :y, :d; covariates=ML_XS, rng=StableRNG(2), parallel=false)
        @test coef(rl) == coef(rl2) && stderror(rl) == stderror(rl2)
        # row shuffling does not change results when folds are keyed to rows
        df.fold = crossfit_folds(400, 5; rng=StableRNG(3))[:, 1]
        a = dml_plr(df, :y, :d; kw..., folds=:fold)
        perm = randperm(StableRNG(4), 400)
        b = dml_plr(df[perm, :], :y, :d; kw..., folds=:fold)
        @test coef(a) ≈ coef(b) rtol = 1e-10
        @test stderror(a) ≈ stderror(b) rtol = 1e-10
        s = sprint(show, MIME"text/plain"(), r)
        @test occursin("Cross-fitting: 5 folds × 3 repetitions", s)
        nl = nuisance_loss(r)
        @test nl.nuisance == ["ml_l", "ml_m"] && all(nl.measure .== "RMSE")
        # IV-type with linear learners equals partialling out (OLS predictions are
        # linear in the target)
        riv = dml_plr(df, :y, :d; kw..., g_learner=ML_OLS, score=:iv_type,
                      rng=StableRNG(1))
        rpo = dml_plr(df, :y, :d; kw..., rng=StableRNG(1))
        @test coef(riv) ≈ coef(rpo) rtol = 1e-10
        @test haskey(riv.predictions, :ml_g)
    end

    @testset "error paths" begin
        df = ml_irm_data(StableRNG(12), 200)
        @test_throws ArgumentError dml_plr(df, :y, :nope; covariates=[:x1])
        @test_throws ArgumentError dml_plr(df, :y, :d; covariates=[:x1], score=:bad)
        @test_throws ArgumentError dml_plr(df, :y, :d; covariates=[:x1], n_folds=1)
        @test_throws ArgumentError dml_irm(df, :y, :x1; covariates=[:x2])
        @test_throws ArgumentError dml_irm(df, :y, :d; covariates=[:x1], score=:LATE)
        @test_throws ArgumentError dml_irm(df, :y, :d; covariates=[:x1], trim=0.7)
        @test_throws ArgumentError dml_irm(df, :y, :d; covariates=[:x1],
                                           propensity_learner=OLSLearner())
        dm = allowmissing(copy(df))
        dm.x1[3] = missing
        @test_throws ArgumentError dml_irm(dm, :y, :d; covariates=[:x1])
        ds = copy(df)
        ds.s = string.(ds.x1)
        @test_throws ArgumentError dml_plr(ds, :y, :d; covariates=[:s])
        di = ml_iivm_data(StableRNG(13), 200)
        @test_throws ArgumentError dml_iivm(di, :y, :d, :z; covariates=[:x1],
                                            always_takers=false)
        @test_throws ArgumentError dml_iivm(di, :y, :d, :x1; covariates=[:x2])
        dp = ml_pliv_data(StableRNG(14), 100)
        dp.z2 = randn(StableRNG(15), 100)
        @test_throws ArgumentError dml_pliv(dp, :y, :d, [:z, :z2]; covariates=[:x1],
                                            score=:iv_type)
        # a fold without treated units in training
        dd = DataFrame(y=randn(StableRNG(16), 20), d=[1.0; zeros(19)], x=randn(20))
        @test_throws ArgumentError dml_irm(dd, :y, :d; covariates=[:x], stratify=false,
                                           outcome_learner=ML_OLS,
                                           propensity_learner=ML_LOGIT)
    end

    @testset "clustering" begin
        df = ml_plr_data(StableRNG(17), 400; G=40)
        r = dml_plr(df, :y, :d; covariates=ML_XS, outcome_learner=ML_OLS,
                    treatment_learner=ML_OLS, cluster=:g, rng=StableRNG(1))
        @test dof_residual(r) == 39 && r.n_clusters == 40
        for c in 1:40
            @test length(unique(r.folds[df.g .== c, 1])) == 1
        end
        ψ = r.psi[:, 1, 1]
        S = [sum(ψ[df.g .== c]) for c in 1:40]
        @test vcov(r)[1, 1] ≈ 40 / 39 * sum(S .^ 2) / sum(r.psi_a[:, 1, 1])^2
        @test occursin("Clusters: 40", sprint(show, MIME"text/plain"(), r))
    end

    @testset "multiple treatments and simultaneous inference" begin
        rng = StableRNG(18)
        n = 500
        X = randn(rng, n, 3)
        D = X[:, 1:3] .+ randn(rng, n, 3)
        y = 0.5 .* D[:, 1] .+ X[:, 1] .+ randn(rng, n)
        df = DataFrame(hcat(X, D, y), [:x1, :x2, :x3, :d1, :d2, :d3, :y])
        r = dml_plr(df, :y, [:d1, :d2, :d3]; covariates=[:x1, :x2, :x3],
                    outcome_learner=ML_OLS, treatment_learner=ML_OLS, rng=StableRNG(1))
        @test coefnames(r) == ["d1", "d2", "d3"]
        @test size(vcov(r)) == (3, 3) && isposdef(Symmetric(vcov(r)))
        sc = simultaneous_confint(r; n_boot=2000, rng=StableRNG(2))
        @test sc.critical_value > critical_value(0.95)
        @test all(sc.lower .< sc.estimate .< sc.upper)
        @test sc.pvalues[1] < 0.01
        @test sc == simultaneous_confint(r; n_boot=2000, rng=StableRNG(2))
        for m in (:wild, :bayes)
            @test simultaneous_confint(r; n_boot=500, method=m, rng=StableRNG(3)) isa
                  NamedTuple
        end
        # a single coefficient: the sup-t critical value is the normal quantile
        r1 = dml_plr(df, :y, :d1; covariates=[:x1, :x2, :x3], outcome_learner=ML_OLS,
                     treatment_learner=ML_OLS, rng=StableRNG(1))
        c1 = simultaneous_confint(r1; n_boot=20_000, rng=StableRNG(4)).critical_value
        @test abs(c1 - 1.96) < 0.05
        @test_throws ArgumentError simultaneous_confint(r; method=:bad)
        # Monte Carlo: joint coverage of the three coefficients
        reps = mc_reps(400, 60)
        cover = 0
        for rep in 1:reps
            rg = StableRNG(1000 + rep)
            Xr = randn(rg, 300, 3)
            Dr = Xr .+ randn(rg, 300, 3)
            yr = 0.5 .* Dr[:, 1] .+ Xr[:, 1] .+ randn(rg, 300)
            dr = DataFrame(hcat(Xr, Dr, yr), [:x1, :x2, :x3, :d1, :d2, :d3, :y])
            rr = dml_plr(dr, :y, [:d1, :d2, :d3]; covariates=[:x1, :x2, :x3],
                         outcome_learner=ML_OLS, treatment_learner=ML_OLS, rng=rg)
            s = simultaneous_confint(rr; n_boot=500, rng=rg)
            cover += all(s.lower .<= [0.5, 0, 0] .<= s.upper)
        end
        @info "Monte Carlo joint coverage (sup-t)" cover / reps
        @test abs(cover / reps - 0.95) <= ml_cover_tol(reps)
    end

    @testset "Monte Carlo coverage" begin
        reps = mc_reps(500, 100)
        mc = ml_mc(rng -> dml_plr(ml_plr_data(rng, 400), :y, :d; covariates=ML_XS,
                                  outcome_learner=ML_OLS, treatment_learner=ML_OLS,
                                  rng=rng), 0.5, reps, 10_000)
        ml_check_mc(mc, reps, "PLR partialling out")
        mc = ml_mc(rng -> dml_plr(ml_plr_data(rng, 400; G=40), :y, :d; covariates=ML_XS,
                                  outcome_learner=ML_OLS, treatment_learner=ML_OLS,
                                  cluster=:g, rng=rng), 0.5, reps, 20_000)
        ml_check_mc(mc, reps, "PLR clustered")
        mc = ml_mc(rng -> dml_irm(ml_irm_data(rng, 500), :y, :d;
                                  covariates=[:x1, :x2, :x3], outcome_learner=ML_OLS,
                                  propensity_learner=ML_LOGIT, rng=rng), 1.0, reps,
                   30_000)
        ml_check_mc(mc, reps, "IRM ATE")
        mc = ml_mc(rng -> dml_irm(ml_irm_data(rng, 500; hetero=false), :y, :d;
                                  covariates=[:x1, :x2, :x3], outcome_learner=ML_OLS,
                                  propensity_learner=ML_LOGIT, score=:ATTE, rng=rng),
                   1.0, reps, 40_000)
        ml_check_mc(mc, reps, "IRM ATTE")
        mc = ml_mc(rng -> dml_pliv(ml_pliv_data(rng, 500), :y, :d, :z;
                                   covariates=[:x1, :x2, :x3], outcome_learner=ML_OLS,
                                   treatment_learner=ML_OLS, instrument_learner=ML_OLS,
                                   rng=rng), 0.6, reps, 50_000)
        ml_check_mc(mc, reps, "PLIV")
        mc = ml_mc(rng -> dml_iivm(ml_iivm_data(rng, 1000), :y, :d, :z;
                                   covariates=[:x1, :x2, :x3], outcome_learner=ML_OLS,
                                   instrument_learner=ML_LOGIT,
                                   treatment_learner=ML_LOGIT, rng=rng), 0.8, reps,
                   60_000)
        ml_check_mc(mc, reps, "IIVM LATE")
        mc = ml_mc(rng -> dml_iivm(ml_iivm_data(rng, 800; one_sided=true), :y, :d, :z;
                                   covariates=[:x1, :x2, :x3], outcome_learner=ML_OLS,
                                   instrument_learner=ML_LOGIT,
                                   treatment_learner=ML_LOGIT, always_takers=false,
                                   rng=rng), 0.8, reps, 70_000)
        ml_check_mc(mc, reps, "IIVM one-sided")
        # default (cross-validated lasso / penalized logistic) learners, 2 repetitions
        reps2 = mc_reps(300, 40)
        mc = ml_mc(rng -> dml_irm(ml_irm_data(rng, 500), :y, :d;
                                  covariates=[:x1, :x2, :x3], n_rep=2, rng=rng), 1.0,
                   reps2, 80_000)
        ml_check_mc(mc, reps2, "IRM ATE, lasso learners")
    end
end

@testset "DML difference-in-differences" begin
    @testset "scores match DRDID (in-sample nuisances)" begin
        vdir = joinpath(@__DIR__, "..", "validation", "ml")
        ref = ml_read_csv(joinpath(vdir, "drdid_reference.csv"))
        p = ml_read_csv(joinpath(vdir, "did_panel.csv"))
        X = Matrix(p[:, [:x1, :x2, :x3]])
        dy = p.y1 .- p.y0
        c0 = p.d .== 0
        g0 = fitpredict(ML_OLS, X[c0, :], dy[c0], X)
        m = fitpredict_proba(ML_LOGIT, X, p.d, X)
        _, ψb = DrSnow._ml_did_panel_score(dy, p.d, g0, m)
        @test mean(ψb) ≈ ref.att[ref.case .== "drdid_panel"][1] rtol = 1e-8
        # the orthogonal-score SE is close to DRDID's (which adds first-stage terms)
        @test std(ψb) / sqrt(length(ψb)) ≈ ref.se[ref.case .== "drdid_panel"][1] rtol = 0.05
        c = ml_read_csv(joinpath(vdir, "did_rcs.csv"))
        Xc = Matrix(c[:, [:x1, :x2, :x3]])
        g = [begin
                 s = (c.d .== a) .& (c.post .== b)
                 fitpredict(ML_OLS, Xc[s, :], c.y[s], Xc)
             end for (a, b) in ((0, 0), (0, 1), (1, 0), (1, 1))]
        mc = fitpredict_proba(ML_LOGIT, Xc, c.d, Xc)
        _, ψc = DrSnow._ml_did_rcs_score(c.y, c.d, c.post, g..., mc)
        @test mean(ψc) ≈ ref.att[ref.case .== "drdid_rc"][1] rtol = 1e-8
        @test std(ψc) / sqrt(length(ψc)) ≈ ref.se[ref.case .== "drdid_rc"][1] rtol = 0.05
    end

    @testset "panel and cross-section interface" begin
        long = ml_did_panel(StableRNG(21), 600)
        kw = (covariates=[:x1, :x2, :x3], outcome_learner=ML_OLS,
              propensity_learner=ML_LOGIT)
        r = dml_did(long, :y, :d; time=:t, unit=:id, kw..., n_rep=2, rng=StableRNG(1))
        @test nobs(r) == 600 && r.model === :did
        @test abs(coef(r)[1] - 1.5) < 4 * stderror(r)[1]
        # results do not depend on the row order of the long panel
        perm = randperm(StableRNG(2), nrow(long))
        r2 = dml_did(long[perm, :], :y, :d; time=:t, unit=:id, kw..., n_rep=2,
                     rng=StableRNG(1))
        @test coef(r2) ≈ coef(r) rtol = 1e-10
        @test stderror(r2) ≈ stderror(r) rtol = 1e-10
        # errors
        bad = copy(long)
        bad.t[1] = 2
        @test_throws ArgumentError dml_did(bad, :y, :d; time=:t, unit=:id, kw...)
        @test_throws ArgumentError dml_did(long[2:end, :], :y, :d; time=:t, unit=:id,
                                           kw...)
        b2 = copy(long)
        b2.d[1] = 1 - b2.d[1]
        @test_throws ArgumentError dml_did(b2, :y, :d; time=:t, unit=:id, kw...)
        cs = ml_did_rcs(StableRNG(22), 1200)
        rc = dml_did(cs, :y, :d; time=:post, kw..., rng=StableRNG(3))
        @test nobs(rc) == 1200 && rc.model === :did_cs
        @test length(rc.learners) == 5
        @test_throws ArgumentError dml_did(cs[cs.d .== 0, :], :y, :d; time=:post, kw...)
        # clustered panel
        long.state = repeat(1:30, 20)[long.id]
        rcl = dml_did(long, :y, :d; time=:t, unit=:id, cluster=:state, kw...,
                      rng=StableRNG(4))
        @test dof_residual(rcl) == 29
    end

    @testset "Monte Carlo coverage" begin
        reps = mc_reps(500, 100)
        kw = (covariates=[:x1, :x2, :x3], outcome_learner=ML_OLS,
              propensity_learner=ML_LOGIT)
        mc = ml_mc(rng -> dml_did(ml_did_panel(rng, 600), :y, :d; time=:t, unit=:id,
                                  kw..., rng=rng), 1.5, reps, 90_000)
        ml_check_mc(mc, reps, "DML-DiD panel")
        mc = ml_mc(rng -> dml_did(ml_did_rcs(rng, 1500), :y, :d; time=:post, kw...,
                                  rng=rng), 1.5, reps, 95_000)
        ml_check_mc(mc, reps, "DML-DiD repeated cross-sections")
    end
end
