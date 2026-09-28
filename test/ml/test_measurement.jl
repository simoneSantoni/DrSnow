# Inference with ML-measured variables: DSL, PPI for causal targets, natural
# experiments with a predicted outcome, regression calibration, differential-error
# test. Closed forms, reference packages, identities and error paths; Monte Carlo
# checks are in test_measurement_montecarlo.jl.

"""RCT with an ML-predicted outcome whose error depends on treatment (bias `delta`)."""
function ml_meas_rct(rng, n; share=0.25, delta=0.5, unequal=false, G=0)
    x = randn(rng, n)
    d = Float64.(rand(rng, n) .< 0.5)
    y = 1 .+ 1.0 .* d .+ 0.5 .* x .+ randn(rng, n)
    f = 0.3 .+ 0.8 .* y .+ delta .* d .+ 0.6 .* randn(rng, n)
    p = unequal ? clamp.(share .* (0.5 .+ (x .> 0) .+ 0.5 .* d), 0.02, 1.0) :
        fill(share, n)
    lab = rand(rng, n) .< p
    cl = G > 0 ? rand(rng, 1:G, n) : collect(1:n)
    return DataFrame(y=Union{Missing,Float64}[l ? v : missing for (l, v) in zip(lab, y)],
                     f=f, d=d, x=x, p=p, lab=Int.(lab), ytrue=y, cl=cl)
end

"""Staggered-free panel: treated half from period 3 of 4; LLM error shifts with D."""
function ml_meas_panel(rng, N; T=4, share=0.3, delta=0.6, tau=1.0)
    rows = N * T
    id = repeat(1:N, inner=T)
    t = repeat(1:T, outer=N)
    tr = repeat(Float64.(rand(rng, N) .< 0.5), inner=T)
    a = repeat(randn(rng, N), inner=T)
    D = tr .* (t .>= 3)
    y = a .+ 0.3 .* t .+ tau .* D .+ randn(rng, rows)
    f = 0.2 .+ 0.9 .* y .+ delta .* D .+ 0.5 .* randn(rng, rows)
    lab = rand(rng, rows) .< share
    return DataFrame(id=id, t=t, D=D, g=ifelse.(tr .== 1, 3, 0),
                     y=Union{Missing,Float64}[l ? v : missing for (l, v) in zip(lab, y)],
                     f=f, ytrue=y, lab=Int.(lab))
end

"""HC0 / CR0 sandwich of OLS of `y` on `X`."""
function ml_meas_hc0(X, y, cl=nothing)
    b = X \ y
    S = X .* (y .- X * b)
    if cl !== nothing
        u = unique(cl)
        S = reduce(vcat, [sum(S[cl .== c, :]; dims=1) for c in u])
    end
    B = inv(X' * X)
    return b, B * (S' * S) * B
end

@testset "Measurement: DSL closed forms" begin
    df = ml_meas_rct(StableRNG(11), 1500; G=60)
    n = nrow(df)
    R = df.lab .== 1
    π = mean(R)
    X = hcat(ones(n), df.d, df.x)
    yl = coalesce.(df.y, 0.0)
    @testset "raw prediction (learner = nothing) = PPI with λ = 1" begin
        r = dsl_regression(df, :y; covariates=[:d, :x], prediction=:f, learner=nothing)
        pseudo = df.f .+ (R ./ π) .* (yl .- df.f)
        b, V = ml_meas_hc0(X, pseudo)
        @test coef(r) ≈ b
        @test vcov(r) ≈ V
        @test coefnames(r) == ["(Intercept)", "d", "x"]
        @test r.n_labeled == count(R) && nobs(r) == n
        # the naive plug-in is the regression on the prediction
        bn, Vn = ml_meas_hc0(X, df.f)
        @test r.naive_coef ≈ bn && r.naive_vcov ≈ Vn
        # PPI (λ = 1) with the whole sample as "unlabeled" gives the same point estimate
        lab = DataFrame(y=Float64.(df.y[R]), f=df.f[R], d=df.d[R], x=df.x[R])
        p1 = ppi_ols(lab, df, :y, :f; covariates=[:d, :x], lambda=1)
        @test coef(p1) ≈ coef(r)
        # clustered: CR0 without small-sample factor
        rc = dsl_regression(df, :y; covariates=[:d, :x], prediction=:f, learner=nothing,
                            cluster=:cl)
        _, Vc = ml_meas_hc0(X, pseudo, df.cl)
        @test vcov(rc) ≈ Vc
        @test rc.n_clusters == 60
        # the pseudo-outcome helper agrees
        ps = dsl_pseudo_outcome(df, :y; prediction=:f, learner=nothing)
        @test ps.pseudo ≈ pseudo
    end
    @testset "cross-fitted OLS recalibration" begin
        folds = repeat(1:5, cld(n, 5))[1:n]
        df.fold = folds
        r = dsl_regression(df, :y; covariates=[:d, :x], prediction=:f, folds=:fold)
        g = zeros(n)
        for k in 1:5
            tr = (folds .!= k) .& R
            Z = hcat(ones(count(tr)), df.f[tr])
            c = Z \ yl[tr]
            g[folds .== k] = hcat(ones(count(folds .== k)), df.f[folds .== k]) * c
        end
        @test r.details.fitted[:, 1] ≈ g
        b, V = ml_meas_hc0(X, g .+ (R ./ π) .* (yl .- g))
        @test coef(r) ≈ b
        @test vcov(r) ≈ V
        # row-order invariance with a fold column
        perm = randperm(StableRNG(3), n)
        rs = dsl_regression(df[perm, :], :y; covariates=[:d, :x], prediction=:f,
                            folds=:fold)
        @test coef(rs) ≈ coef(r)
        @test vcov(rs) ≈ vcov(r)
        # labelling probabilities (normalized to the labelled share)
        df.pp = ifelse.(df.x .> 0, 0.3, 0.2)
        ru = dsl_regression(df, :y; covariates=[:d, :x], prediction=:f, folds=:fold,
                            label_prob=:pp)
        πn = df.pp .* (mean(R) / mean(df.pp))
        b2, _ = ml_meas_hc0(X, g .+ (R ./ πn) .* (yl .- g))
        @test coef(ru) ≈ b2
        ru2 = dsl_regression(df, :y; covariates=[:d, :x], prediction=:f, folds=:fold,
                             label_prob=:pp, normalize_prob=false)
        b3, _ = ml_meas_hc0(X, g .+ (R ./ df.pp) .* (yl .- g))
        @test coef(ru2) ≈ b3
        # repetitions: median over repetitions
        r3 = dsl_regression(df, :y; covariates=[:d, :x], prediction=:f, n_rep=3,
                            rng=StableRNG(5))
        @test size(r3.details.rep_coef) == (3, 3)
        @test coef(r3) ≈ vec(median(r3.details.rep_coef; dims=2))
    end
    @testset "fixed effects = LSDV sandwich" begin
        df2 = ml_meas_rct(StableRNG(12), 600; G=8)
        n2 = nrow(df2)
        R2 = df2.lab .== 1
        pseudo = df2.f .+ (R2 ./ mean(R2)) .* (coalesce.(df2.y, 0.0) .- df2.f)
        r = dsl_regression(df2, :y; covariates=[:d, :x], prediction=:f, learner=nothing,
                           fe=[:cl], cluster=:cl)
        Dm = reduce(hcat, [Float64.(df2.cl .== c) for c in sort(unique(df2.cl))])
        Z = hcat(df2.d, df2.x, Dm)
        b, V = ml_meas_hc0(Z, pseudo, df2.cl)
        @test coef(r) ≈ b[1:2] rtol = 1e-8
        @test vcov(r) ≈ V[1:2, 1:2] rtol = 1e-6
        @test coefnames(r) == ["d", "x"]
        @test occursin("fixed-effects", method_name(r))
    end
    @testset "logistic moment" begin
        rng = StableRNG(13)
        n3 = 3000
        x = randn(rng, n3)
        yb = Float64.(rand(rng, n3) .< 1 ./ (1 .+ exp.(-(0.3 .+ 0.8 .* x))))
        fb = clamp.(0.6 .* yb .+ 0.2 .+ 0.1 .* randn(rng, n3), 0, 1)
        lab = rand(rng, n3) .< 0.3
        d3 = DataFrame(y=Union{Missing,Float64}[l ? v : missing for (l, v) in zip(lab, yb)],
                       f=fb, x=x)
        r = dsl_regression(d3, :y; covariates=[:x], prediction=:f, learner=nothing,
                           family=:binomial)
        Ỹ = fb .+ (lab ./ mean(lab)) .* (coalesce.(d3.y, 0.0) .- fb)
        X3 = hcat(ones(n3), x)
        σ = 1 ./ (1 .+ exp.(-X3 * coef(r)))
        @test norm(X3' * (Ỹ .- σ)) / n3 < 1e-9
        J = X3' * (X3 .* (σ .* (1 .- σ)))
        S = X3 .* (Ỹ .- σ)
        @test vcov(r) ≈ inv(J) * (S' * S) * inv(J) rtol = 1e-8
        # all rows labelled: the logistic MLE
        d4 = DataFrame(y=yb, f=fb, x=x)
        rall = dsl_regression(d4, :y; covariates=[:x], prediction=:f, learner=nothing,
                              family=:binomial)
        m = DrSnow.GLM.glm(X3, yb, DrSnow.Binomial())
        @test coef(rall) ≈ coef(m) rtol = 1e-5
    end
    @testset "predicted covariate (linear moment)" begin
        rng = StableRNG(14)
        n4 = 2000
        xt = randn(rng, n4)
        y = 1 .+ 0.7 .* xt .+ randn(rng, n4)
        xp = 0.6 .* xt .+ 0.5 .* randn(rng, n4)
        lab = rand(rng, n4) .< 0.3
        d5 = DataFrame(y=y, x=Union{Missing,Float64}[l ? v : missing for (l, v) in
                                                     zip(lab, xt)], xp=xp)
        r = dsl_regression(d5, :y; covariates=[:x], prediction=:xp, predicted_vars=[:x],
                           learner=nothing)
        w = lab ./ mean(lab)
        Xo = hcat(ones(n4), coalesce.(d5.x, 0.0))
        Xp = hcat(ones(n4), xp)
        A = Xp' * (Xp .* (1 .- w)) .+ Xo' * (Xo .* w)
        b = Xp' * ((1 .- w) .* y) .+ Xo' * (w .* y)
        @test coef(r) ≈ A \ b
    end
    @testset "category proportions" begin
        rng = StableRNG(15)
        n5 = 2000
        truth = rand(rng, ["a", "b", "c"], n5)
        pred = [rand(rng) < 0.7 ? c : "a" for c in truth]
        lab = rand(rng, n5) .< 0.25
        yr = rand(rng, 1:2, n5)
        d6 = DataFrame(cat=Union{Missing,String}[l ? v : missing for (l, v) in
                                                 zip(lab, truth)], pc=pred, yr=yr)
        r = dsl_proportions(d6, :cat; prediction=:pc, learner=nothing)
        π6 = mean(lab)
        for (j, k) in enumerate(["a", "b", "c"])
            ind = Float64.(pred .== k)
            ỹ = ind .+ (lab ./ π6) .* (Float64.(lab .& (truth .== k)) .- ind)
            @test coef(r)[j] ≈ mean(ỹ)
            @test vcov(r)[j, j] ≈ var(ỹ; corrected=false) / n5
            @test r.naive_coef[j] ≈ mean(ind)
        end
        @test sum(coef(r)) ≈ 1
        @test coefnames(r) == ["a", "b", "c"]
        rb = dsl_proportions(d6, :cat; prediction=:pc, by=:yr, rng=StableRNG(1))
        @test length(coef(rb)) == 6
        @test coefnames(rb)[4] == "a | yr = 2"
        @test sum(coef(rb)[1:3]) ≈ 1 atol = 1e-8
    end
end

@testset "Measurement: R dsl parity" begin
    vdir = joinpath(@__DIR__, "..", "validation", "ml")
    ref = ml_read_csv(joinpath(vdir, "dsl_reference.csv"))
    refv = ml_read_csv(joinpath(vdir, "dsl_reference_vcov.csv"))
    load(case) = begin
        d = ml_read_csv(joinpath(vdir, "dsl_data_$(case).csv"))
        for c in propertynames(d)
            v = d[!, c]
            if eltype(v) <: AbstractString && c ∉ (:state,)
                d[!, c] = [isempty(s) ? missing : parse(Float64, s) for s in v]
            end
        end
        d.fold = Int.(d.fold)
        d
    end
    check(case, r; rtol_b=2e-5, rtol_se=2e-5) = begin
        rr = ref[ref.case .== case, :]
        @test coefnames(r) == String.(rr.term)
        @test coef(r) ≈ rr.estimate rtol = rtol_b
        @test stderror(r) ≈ rr.std_error rtol = rtol_se
        vv = refv[refv.case .== case, :]
        k = length(coef(r))
        V = zeros(k, k)
        for row in eachrow(vv)
            V[Int(row.row), Int(row.col)] = row.value
        end
        @test vcov(r) ≈ V rtol = 10 * rtol_se
    end
    d = load("lm")
    check("lm", dsl_regression(d, :Y; covariates=[:X1, :X2, :X3, :X4, :X5],
                               prediction=:pred_Y, folds=:fold))
    d = load("lm_cluster")
    check("lm_cluster", dsl_regression(d, :Y; covariates=[:X1, :X2, :X3],
                                       prediction=:pred_Y, folds=:fold, cluster=:cl))
    d = load("logit_xpred")
    check("logit_xpred", dsl_regression(d, :Y; covariates=[:X1, :X2, :X4],
                                        prediction=[:pred_Y, :pred_X1],
                                        predicted_vars=[:Y, :X1], family=:binomial,
                                        folds=:fold); rtol_b=1e-4, rtol_se=1e-4)
    d = load("logit_xpred")
    check("lm_xpred", dsl_regression(d, :Y; covariates=[:X1, :X2, :X4],
                                     prediction=[:pred_Y, :pred_X1],
                                     predicted_vars=[:Y, :X1], folds=:fold))
    d = load("logit_unequal")
    check("logit_unequal", dsl_regression(d, :Y; covariates=[:X1, :X2, :X3],
                                          prediction=:pred_Y, label_prob=:sample_prob,
                                          family=:binomial, folds=:fold);
          rtol_b=1e-4, rtol_se=1e-4)
    d = load("logit_unequal")
    check("lm_unequal", dsl_regression(d, :Y; covariates=[:X1, :X2, :X3],
                                       prediction=:pred_Y, label_prob=:sample_prob,
                                       folds=:fold))
    d = load("felm_twoways")
    # dsl's optimizer (L-BFGS-B on the squared moments) stops ~1e-3 (relative) from
    # the root in this nearly collinear two-way model; the exact within estimator on
    # the same pseudo-outcome (lm in R) agrees to machine precision.
    rtw = dsl_regression(d, :log_gsp; covariates=[:log_pcap, :log_pc, :unemp],
                         prediction=:pred_log_gsp, fe=[:state, :year], cluster=:state,
                         folds=:fold)
    check("felm_twoways", rtw; rtol_b=2e-3, rtol_se=2e-4)
    within = ml_read_csv(joinpath(vdir, "dsl_reference_within.csv"))
    @test coef(rtw) ≈ within.estimate rtol = 1e-9
    d = load("felm_oneway")
    check("felm_oneway", dsl_regression(d, :log_gsp;
                                        covariates=[:log_pcap, :log_pc, :unemp],
                                        prediction=:pred_log_gsp, fe=[:state],
                                        folds=:fold))
end

@testset "Measurement: PPI for causal targets" begin
    df = ml_meas_rct(StableRNG(21), 2000; G=80)
    n = nrow(df)
    R = df.lab .== 1
    π = mean(R)
    yl = coalesce.(df.y, 0.0)
    @testset "ATE closed forms" begin
        λ = 0.7
        r = ppi_ate(df, :y, :d, :f; lambda=λ)
        ỹ = λ .* df.f .+ (R ./ π) .* (yl .- λ .* df.f)
        t, c = df.d .== 1, df.d .== 0
        @test coef(r)[1] ≈ mean(ỹ[t]) - mean(ỹ[c])
        @test vcov(r)[1, 1] ≈ var(ỹ[t]; corrected=false) / count(t) +
                              var(ỹ[c]; corrected=false) / count(c)
        @test coefnames(r) == ["ATE"] && estimand(r) == "ATE"
        @test r.naive_coef[1] ≈ mean(df.f[t]) - mean(df.f[c])
        ro = ppi_ate(df, :y, :d, :f)
        r0 = ppi_ate(df, :y, :d, :f; lambda=0)
        r1 = ppi_ate(df, :y, :d, :f; lambda=1)
        @test ro.details.lambda >= 0
        @test vcov(ro)[1, 1] <= vcov(r0)[1, 1] * (1 + 1e-10)
        @test vcov(ro)[1, 1] <= vcov(r1)[1, 1] * (1 + 1e-10)
        # λ̂ minimizes the estimated variance exactly
        for δ in (-0.05, 0.05)
            rδ = ppi_ate(df, :y, :d, :f; lambda=max(ro.details.lambda + δ, 0))
            @test vcov(ro)[1, 1] <= vcov(rδ)[1, 1] * (1 + 1e-10)
        end
        # Lin regression adjustment and clustering run; covariates reduce variance
        rl = ppi_ate(df, :y, :d, :f; covariates=[:x])
        @test stderror(rl)[1] < stderror(ro)[1]
        @test rl.details.full_names == ["(Intercept)", "ATE", "x", "ATE × x"]
        rc = ppi_ate(df, :y, :d, :f; cluster=:cl)
        @test rc.n_clusters == 80
        # fixed effects: equals the within regression of the pseudo-outcome
        rf = ppi_ate(df, :y, :d, :f; fe=[:cl], lambda=0.5, cluster=:cl)
        ỹ5 = 0.5 .* df.f .+ (R ./ π) .* (yl .- 0.5 .* df.f)
        tmp = DataFrame(yt=ỹ5, d=df.d, cl=df.cl)
        m = DrSnow.FixedEffectModels.reg(tmp, DrSnow.make_formula(:yt, [:d]; fe=[:cl]);
                                         tol=1e-12)
        @test coef(rf)[1] ≈ coef(m)[1] rtol = 1e-8
        # invariance to row order
        perm = randperm(StableRNG(2), n)
        @test coef(ppi_ate(df[perm, :], :y, :d, :f; covariates=[:x])) ≈ coef(rl)
    end
    @testset "regression with λ = 1 matches ppi_ols" begin
        r = ppi_regression(df, :y, :f; covariates=[:d, :x], lambda=1)
        lab = DataFrame(y=Float64.(df.y[R]), f=df.f[R], d=df.d[R], x=df.x[R])
        @test coef(r) ≈ coef(ppi_ols(lab, df, :y, :f; covariates=[:d, :x], lambda=1))
        ro = ppi_regression(df, :y, :f; covariates=[:d, :x], target=:d)
        @test ro.details.lambda >= 0
        @test_throws ArgumentError ppi_regression(df, :y, :f; covariates=[:d],
                                                  target=:x)
    end
    @testset "cross-PPI" begin
        rng = StableRNG(22)
        n, N = 400, 4000
        z = randn(rng, n + N, 2)
        y = 1 .+ z[:, 1] .- 0.5 .* z[:, 2] .+ 0.5 .* randn(rng, n + N)
        lab = DataFrame(y=y[1:n], z1=z[1:n, 1], z2=z[1:n, 2])
        un = DataFrame(z1=z[(n + 1):end, 1], z2=z[(n + 1):end, 2])
        r = cross_ppi(lab, un, :y; features=[:z1, :z2], n_folds=5, rng=StableRNG(7))
        F = crossfit_folds(n, 5, 1; rng=StableRNG(7))[:, 1]
        oof = zeros(n)
        fu = zeros(N)
        for k in 1:5
            tr = F .!= k
            Z = hcat(ones(count(tr)), z[1:n, :][tr, :])
            c = Z \ y[1:n][tr]
            oof[.!tr] = hcat(ones(count(.!tr)), z[1:n, :][.!tr, :]) * c
            fu .+= hcat(ones(N), z[(n + 1):end, :]) * c ./ 5
        end
        @test r.details.oof_prediction ≈ oof
        @test r.details.unlabeled_prediction ≈ fu
        p = ppi_mean(y[1:n], oof, fu)
        @test coef(r) ≈ coef(p)
        @test vcov(r) ≈ vcov(p)
        @test estimand(r) == "population mean"
        @test stderror(r)[1] < sqrt(r.details.classical_vcov[1, 1])
        lab.d = Float64.(lab.z1 .> 0)
        un.d = Float64.(un.z1 .> 0)
        rr = cross_ppi(lab, un, :y; features=[:z1, :z2], covariates=[:d],
                       rng=StableRNG(8))
        @test coefnames(rr) == ["(Intercept)", "d"]
        lab.yb = Float64.(lab.y .> 1)
        rb = cross_ppi(lab, un, :yb; features=[:z1, :z2], covariates=[:d],
                       family=:binomial, learner=LogisticLearner(), rng=StableRNG(9))
        @test occursin("logistic", method_name(rb))
        @test_throws ArgumentError cross_ppi(lab, un, :y; features=Symbol[])
        @test_throws ArgumentError cross_ppi(lab, un, :y; features=[:z1], family=:poisson)
        @test_throws ArgumentError cross_ppi(lab, un, :y; features=[:z1],
                                             family=:binomial)
    end
end

@testset "Measurement: natural experiments" begin
    df = ml_meas_panel(StableRNG(31), 300)
    @testset "DiD (TWFE) design-based" begin
        r = did_with_predicted_outcome(df, :y, :D, :id, :t; prediction=:f,
                                       rng=StableRNG(1))
        tmp = copy(df)
        tmp.ps = r.details.pseudo
        rt = did_twfe(tmp, :ps, :D, :id, :t; warn_heterogeneity=false)
        @test coef(r) ≈ coef(rt)
        @test stderror(r) ≈ stderror(rt)
        @test dof_residual(r) == dof_residual(rt)
        @test coef(r.naive) ≈ coef(did_twfe(df, :f, :D, :id, :t; warn_heterogeneity=false))
        # linearity: corrected − naive = the estimator on (pseudo − prediction)
        @test coef(r)[1] - coef(r.naive)[1] ≈ r.bias_test.details.difference
        @test r.mode === :design
        @test confint(r) ≈ confint(rt)
        @test occursin("Prediction-corrected", method_name(r))
        @test occursin("Naive estimate", sprint(show, MIME"text/plain"(), r))
        # folds are grouped by unit
        f = r.details.folds
        @test all(length(unique(f[df.id .== i])) == 1 for i in 1:20)
        # raw prediction as ĝ
        r0 = did_with_predicted_outcome(df, :y, :D, :id, :t; prediction=:f,
                                        learner=nothing)
        R = df.lab .== 1
        ps = df.f .+ (R ./ mean(R)) .* (coalesce.(df.y, 0.0) .- df.f)
        tmp.ps = ps
        @test coef(r0) ≈ coef(did_twfe(tmp, :ps, :D, :id, :t; warn_heterogeneity=false))
        # cross-prediction: measure learned from features, by unit folds
        df.z = df.f .+ 0.1 .* randn(StableRNG(3), nrow(df))
        rf = did_with_predicted_outcome(df, :y, :D, :id, :t; features=[:z],
                                        rng=StableRNG(2))
        @test coef(rf.naive) ≈ coef(did_twfe(DataFrame(g=rf.details.fitted, D=df.D,
                                                       id=df.id, t=df.t), :g, :D, :id,
                                             :t; warn_heterogeneity=false))
        # row-order invariance with a fold column
        df.fold = mod1.(df.id, 5)
        a = did_with_predicted_outcome(df, :y, :D, :id, :t; prediction=:f, folds=:fold)
        perm = randperm(StableRNG(4), nrow(df))
        b = did_with_predicted_outcome(df[perm, :], :y, :D, :id, :t; prediction=:f,
                                       folds=:fold)
        @test coef(a) ≈ coef(b)
        @test stderror(a) ≈ stderror(b)
    end
    @testset "held-out discipline and label checks" begin
        d2 = copy(df)
        d2.train = Int.(d2.id .<= 30)
        r = did_with_predicted_outcome(d2, :y, :D, :id, :t; prediction=:f,
                                       measure_training=:train, rng=StableRNG(1))
        @test r.n_dropped_training == 30
        @test nobs(r) == nrow(df) - 30 * 4
        # labels only before treatment: design mode refuses, stable mode runs
        d3 = copy(df)
        d3.y = [t <= 2 ? v : missing for (t, v) in zip(d3.t, d3.ytrue)]
        err = try
            did_with_predicted_outcome(d3, :y, :D, :id, :t; prediction=:f)
            nothing
        catch e
            e
        end
        @test err isa ArgumentError
        @test occursin("assume_stable_error", sprint(showerror, err))
        rs = did_with_predicted_outcome(d3, :y, :D, :id, :t; prediction=:f,
                                        assume_stable_error=true, bootstrap_reps=40,
                                        rng=StableRNG(5))
        @test rs.mode === :stable_error
        @test all(stderror(rs) .> 0)
        @test isfinite(rs.bias_test.pvalue)
        @test size(confint(rs)) == (1, 2)
        @test_throws ArgumentError confint(rs; uniform=true)
        # with known probabilities the design mode accepts unlabelled cells only if π > 0
        d3.p = fill(0.5, nrow(d3))
        @test_throws ArgumentError did_with_predicted_outcome(d3, :y, :D, :id, :t;
                                                              prediction=:f,
                                                              estimator=:cs,
                                                              assume_stable_error=true)
        @test_throws ArgumentError did_with_predicted_outcome(df, :y, :D, :id, :t)
        @test_throws ArgumentError did_with_predicted_outcome(df, :y, :D, :id, :t;
                                                              prediction=:f,
                                                              estimator=:sa)
        @test_throws ArgumentError did_with_predicted_outcome(df, :y, :D, :id, :t;
                                                              prediction=:nope)
    end
    @testset "Callaway-Sant'Anna and DR-DiD" begin
        r = did_with_predicted_outcome(df, :y, FirstTreated(:g), :id, :t; prediction=:f,
                                       estimator=:cs, rng=StableRNG(1))
        tmp = copy(df)
        tmp.ps = r.details.pseudo
        cs = did_callaway_santanna(tmp, :ps, FirstTreated(:g), :id, :t; bootstrap=false)
        @test coef(r) ≈ coef(cs)
        @test estimate(r) ≈ estimate(cs)
        a = aggregate_att(r.corrected, :simple; bootstrap=false)
        an = aggregate_att(r.naive, :simple; bootstrap=false)
        @test coef(a)[1] - coef(an)[1] ≈ r.bias_test.details.difference
        d2 = df[df.t .>= 2 .&& df.t .<= 3, :]
        rd = did_with_predicted_outcome(d2, :y, :D, :id, :t; prediction=:f,
                                        estimator=:drdid, rng=StableRNG(2))
        @test length(coef(rd)) == 1
    end
    @testset "sharp RD" begin
        rng = StableRNG(41)
        n = 3000
        x = 2 .* rand(rng, n) .- 1
        s = Float64.(x .>= 0)
        y = 0.5 .* x .+ 1.0 .* s .+ 0.5 .* randn(rng, n)
        f = y .+ 0.4 .* s .+ 0.3 .* randn(rng, n)
        p = ifelse.(abs.(x) .< 0.3, 0.4, 0.1)
        lab = rand(rng, n) .< p
        d = DataFrame(x=x, f=f, p=p,
                      y=Union{Missing,Float64}[l ? v : missing for (l, v) in zip(lab, y)])
        r = rd_with_predicted_outcome(d, :y, :x; prediction=:f, label_prob=:p,
                                      rng=StableRNG(1))
        tmp = copy(d)
        tmp.ps = r.details.pseudo
        rr = rd_estimate(tmp, :ps, :x)
        @test coef(r) ≈ coef(rr)
        @test stderror(r) ≈ stderror(rr)
        @test r.naive.h_left == r.corrected.h_left
        @test r.corrected.tau_bias_corrected - r.naive.tau_bias_corrected ≈
              r.bias_test.details.difference rtol = 1e-8
        @test tstats(r) ≈ tstats(r.corrected)
        @test confint(r) ≈ confint(r.corrected)
        @test estimand(r) == estimand(rr)
        rs = rd_with_predicted_outcome(d, :y, :x; prediction=:f, assume_stable_error=true,
                                       bootstrap_reps=30, rng=StableRNG(2))
        @test rs.mode === :stable_error
        # labels on one side only without probabilities: refused in design mode
        d.y2 = [xi < 0 ? v : missing for (xi, v) in zip(d.x, y)]
        @test_throws ArgumentError rd_with_predicted_outcome(d, :y2, :x; prediction=:f)
        @test_throws ArgumentError rd_with_predicted_outcome(d, :y, :x; prediction=:f,
                                                             treatment=:s)
    end
end

@testset "Measurement error: calibration and differential-error test" begin
    rng = StableRNG(51)
    n = 3000
    x = randn(rng, n)
    z = randn(rng, n)
    y = 1 .+ 0.8 .* x .+ 0.3 .* z .+ randn(rng, n)
    xs = x .+ 0.8 .* randn(rng, n)
    lab = rand(rng, n) .< 0.2
    df = DataFrame(y=y, xs=xs, z=z,
                   x=Union{Missing,Float64}[l ? v : missing for (l, v) in zip(lab, x)])
    @testset "regression calibration" begin
        r = regression_calibration(df, :y, :xs, :x; covariates=[:z])
        W = hcat(ones(n), xs, z)
        γ = W[lab, :] \ x[lab]
        V = hcat(ones(n), W * γ, z)
        @test coef(r) ≈ V \ y
        @test r.details.calibration_coef ≈ γ
        @test coefnames(r) == ["(Intercept)", "x", "z"]
        @test r.naive_coef ≈ W \ y
        @test abs(r.naive_coef[2]) < abs(coef(r)[2])
        # all rows validated and X* = X: plain OLS with HC0
        d2 = DataFrame(y=y, xs=x, x=x, z=z)
        r2 = regression_calibration(d2, :y, :xs, :x; covariates=[:z])
        b, V2 = ml_meas_hc0(hcat(ones(n), x, z), y)
        @test coef(r2) ≈ b
        @test vcov(r2) ≈ V2 rtol = 1e-8
        # row-order invariance
        perm = randperm(StableRNG(1), n)
        rp = regression_calibration(df[perm, :], :y, :xs, :x; covariates=[:z])
        @test coef(rp) ≈ coef(r) && vcov(rp) ≈ vcov(r)
        @test_throws ArgumentError regression_calibration(df, :y, :xs, :nope)
    end
    @testset "differential-error test" begin
        d = ml_meas_rct(StableRNG(52), 3000; delta=0.5, unequal=true)
        t = differential_error_test(d, :y, :f; by=[:d], label_prob=:p)
        R = d.lab .== 1
        pn = d.p .* (mean(R) / mean(d.p))
        e = coalesce.(d.y, 0.0) .- d.f
        for (k, v) in enumerate((0.0, 1.0))
            s = R .& (d.d .== v)
            @test t.details.mean_error[k] ≈ sum(e[s] ./ pn[s]) / sum(1 ./ pn[s])
        end
        @test t.details.cells == ["d=0.0", "d=1.0"]
        @test rejects(t)
        @test t.dof == (1, count(R) - 2)
        tc = differential_error_test(d, :y, :f; by=[:d], cluster=:cl)
        @test tc.dof[2] == count(R) - 1   # each row its own cluster
        @test occursin("Non-rejection", t.note)
        @test_throws ArgumentError differential_error_test(d, :y, :f; by=Symbol[])
        d.one = ones(nrow(d))
        @test_throws ArgumentError differential_error_test(d, :y, :f; by=[:one])
    end
end

@testset "Measurement: input errors" begin
    df = ml_meas_rct(StableRNG(61), 400)
    @test_throws ArgumentError dsl_regression(df, :y; covariates=[:d], prediction=:f,
                                              family=:poisson)
    @test_throws ArgumentError dsl_regression(df, :y; covariates=[:d])
    @test_throws ArgumentError dsl_regression(df, :y; covariates=[:d], prediction=:f,
                                              predicted_vars=[:x2])
    @test_throws ArgumentError dsl_regression(df, :y; covariates=[:d, :x],
                                              prediction=[:f], fe=[:cl],
                                              predicted_vars=[:y, :x])
    @test_throws ArgumentError dsl_regression(df, :y; covariates=[:d], prediction=:f,
                                              family=:binomial)
    df.badp = fill(1.5, nrow(df))
    @test_throws ArgumentError dsl_regression(df, :y; covariates=[:d], prediction=:f,
                                              label_prob=:badp)
    @test_throws ArgumentError dsl_regression(df, :y; covariates=[:d], prediction=:f,
                                              learner=nothing, predicted_vars=[:y, :x])
    d2 = copy(df)
    d2.y = Vector{Union{Missing,Float64}}(missing, nrow(d2))
    @test_throws ArgumentError dsl_regression(d2, :y; covariates=[:d], prediction=:f)
    d3 = allowmissing(copy(df))
    d3.f[1] = missing
    @test_throws ArgumentError dsl_regression(d3, :y; covariates=[:d], prediction=:f)
    @test_throws ArgumentError ppi_ate(df, :y, :d, :f; lambda=-1)
    d4 = copy(df)
    d4.y = [d == 1 ? missing : v for (d, v) in zip(d4.d, d4.ytrue)]
    @test_throws ArgumentError ppi_ate(d4, :y, :d, :f)
    @test_throws ArgumentError ppi_ate(df, :y, :x, :f)
    @test_throws ArgumentError dsl_proportions(df, :y)
end
