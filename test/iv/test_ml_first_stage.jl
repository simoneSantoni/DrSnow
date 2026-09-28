using DrSnow: pvalue, FixedEffectModels, Chisq, FDist, ccdf
# Cross-fitted first-stage diagnostics and the DML Anderson–Rubin test / set.
# References: FixedEffectModels (robust / cluster Wald F of the residualized first
# stage), dml_irm (the IIVM first stage is the DML ATE of Z on D), closed forms.

"""PLIV DGP: `D = π'Z + x₁ + v`, `Y = θD + g(X) + u`, `corr(u, v) = ρ`."""
function mlfs_pliv_dgp(rng; n=600, pi=[0.3, 0.2], theta=1.0, rho=0.7, G=0)
    x1, x2 = randn(rng, n), randn(rng, n)
    k = length(pi)
    Z = randn(rng, n, k)
    Z[:, 1] .+= 0.5 .* x1
    v = randn(rng, n)
    u = rho .* v .+ sqrt(1 - rho^2) .* randn(rng, n)
    g = G > 0 ? rand(rng, 1:G, n) : collect(1:n)
    if G > 0
        a = randn(rng, G)
        u .+= 0.5 .* a[g]
        v .+= 0.5 .* a[g]
    end
    d = Z * pi .+ x1 .+ v
    y = theta .* d .+ x1 .- x2 .+ u
    df = DataFrame(y=y, d=d, x1=x1, x2=x2, g=g)
    for j in 1:k
        df[!, Symbol("z", j)] = Z[:, j]
    end
    return df
end

@testset "ML first stage and DML weak-IV-robust inference" begin
    @testset "PLIV first stage = FixedEffectModels on out-of-fold residuals" begin
        df = mlfs_pliv_dgp(StableRNG(11); G=40)
        F = crossfit_folds(nrow(df), 5, 2; rng=StableRNG(12), groups=df.g)
        ols = (treatment_learner=OLSLearner(), instrument_learner=OLSLearner())
        fs = ml_first_stage(df, :d, [:z1, :z2]; covariates=[:x1, :x2], folds=F, ols...)
        r = dml_pliv(df, :y, :d, [:z1, :z2]; covariates=[:x1, :x2], folds=F,
                     outcome_learner=OLSLearner(), ols...)
        fr = ml_first_stage(r, df)
        @test fr.F ≈ fs.F rtol = 1e-10
        @test fr.estimate ≈ fs.estimate rtol = 1e-10
        @test fs.instruments == ["z1", "z2"]
        for (k, clus) in ((1, nothing), (2, :g))
            w = df.d .- r.predictions[:ml_r][:, k, 1]
            V1 = df.z1 .- r.predictions[:ml_m_z1][:, k, 1]
            V2 = df.z2 .- r.predictions[:ml_m_z2][:, k, 1]
            tmp = DataFrame(w=w, v1=V1, v2=V2, g=df.g)
            f = make_formula(:w, [:v1, :v2]; intercept=false)
            vc = clus === nothing ? Vcov.robust() : Vcov.cluster(:g)
            m = FixedEffectModels.reg(tmp, f, vc)
            fk = ml_first_stage(df, :d, [:z1, :z2]; covariates=[:x1, :x2],
                                folds=F[:, k:k], cluster=clus, ols...)
            @test fk.F ≈ m.F rtol = 1e-8
            @test fk.estimate ≈ coef(m) rtol = 1e-10
            @test fk.vcov ≈ vcov(m) rtol = 1e-8
            mh = FixedEffectModels.reg(tmp, f, Vcov.simple())
            @test fk.F_homoskedastic ≈ mh.F rtol = 1e-8
            @test fk.partial_r2 ≈ DrSnow.StatsAPI.r2(mh) rtol = 1e-8
            @test fk.effective_F <= fk.F * 10    # well defined
        end
        @test length(fs.all_F) == 2
        @test fs.F ≈ median(fs.all_F)
        @test fs.optimal_instrument_F isa Float64 && fs.optimal_instrument_F > 0
        # one instrument: effective F = robust F
        f1 = ml_first_stage(df, :d, :z1; covariates=[:x1, :x2], folds=F[:, 1], ols...)
        @test f1.effective_F ≈ f1.F rtol = 1e-10
        @test f1.optimal_instrument_F === nothing
        io = IOBuffer()
        show(io, MIME"text/plain"(), fs)
        s = String(take!(io))
        @test occursin("Staiger", s) && occursin("104.7", s)
    end

    @testset "IIVM first stage = DML effect of Z on D" begin
        df, _ = iv_binary_dgp(StableRNG(13); n=3000, confounded=true)
        F = crossfit_folds(nrow(df), 5, 1; rng=StableRNG(14), strata=2 .* df.z .+ df.d)
        lg = LogisticLearner()
        fs = ml_first_stage(df, :d, :z; covariates=[:x], model=:iivm, folds=F,
                            treatment_learner=lg, instrument_learner=lg)
        ri = dml_irm(df, :d, :z; covariates=[:x], outcome_learner=lg,
                     propensity_learner=lg, folds=F)
        @test fs.estimate[1] ≈ coef(ri)[1] rtol = 1e-8
        @test sqrt(fs.vcov[1, 1]) ≈ stderror(ri)[1] rtol = 1e-8
        @test fs.F ≈ (coef(ri)[1] / stderror(ri)[1])^2 rtol = 1e-7
        @test fs.partial_r2 === nothing
        r = dml_iivm(df, :y, :d, :z; covariates=[:x], folds=F, treatment_learner=lg,
                     instrument_learner=lg, outcome_learner=OLSLearner())
        fr = ml_first_stage(r; instrument=:z)
        @test fr.estimate[1] ≈ fs.estimate[1] rtol = 1e-8
        @test fr.instruments == ["z"]
    end

    @testset "DML Anderson–Rubin: closed form and inversion" begin
        df = mlfs_pliv_dgp(StableRNG(15); pi=[0.4])
        r = dml_pliv(df, :y, :d, :z1; covariates=[:x1, :x2], rng=StableRNG(16),
                     outcome_learner=OLSLearner(), treatment_learner=OLSLearner(),
                     instrument_learner=OLSLearner())
        θ̂ = coef(r)[1]
        @test dml_weak_iv_test(r; beta0=θ̂).statistic ≈ 0 atol = 1e-16
        ψa, ψb = r.psi_a[:, 1, 1], r.psi_b[:, 1, 1]
        for b0 in (0.0, 0.8, 1.3)
            ψ = ψb .+ b0 .* ψa
            t = dml_weak_iv_test(r; beta0=b0)
            @test t.statistic ≈ sum(ψ)^2 / sum(abs2, ψ) rtol = 1e-12
            @test t.pvalue ≈ ccdf(Chisq(1), t.statistic) rtol = 1e-12
        end
        cs = dml_weak_iv_confidence_set(r; level=0.9)
        @test cs.kind === :bounded
        lo, hi = cs.intervals[1]
        @test lo < θ̂ < hi
        @test dml_weak_iv_test(r; beta0=lo).pvalue ≈ 0.1 atol = 1e-8
        @test dml_weak_iv_test(r; beta0=hi).pvalue ≈ 0.1 atol = 1e-8
        @test pvalue(cs, θ̂) ≈ 1.0
        # strong instrument: AR set ≈ Wald interval
        ci = confint(r; level=0.9)
        @test (hi - lo) ≈ (ci[2] - ci[1]) rtol = 0.2
        # clustered: F(1, G - 1) reference
        dfc = mlfs_pliv_dgp(StableRNG(17); pi=[0.4], G=30)
        rc = dml_pliv(dfc, :y, :d, :z1; covariates=[:x1, :x2], cluster=:g,
                      rng=StableRNG(18), outcome_learner=OLSLearner(),
                      treatment_learner=OLSLearner(), instrument_learner=OLSLearner())
        tc = dml_weak_iv_test(rc; beta0=1.0)
        @test tc.dof == (1, 29)
        @test tc.pvalue ≈ ccdf(FDist(1, 29), tc.statistic)
        # repetitions: a value is in the set iff at least half the repetitions accept
        rr = dml_pliv(df, :y, :d, :z1; covariates=[:x1, :x2], n_rep=4,
                      rng=StableRNG(19), outcome_learner=LassoLearner(),
                      treatment_learner=LassoLearner(), instrument_learner=LassoLearner())
        csr = dml_weak_iv_confidence_set(rr)
        tr = dml_weak_iv_test(rr; beta0=1.0)
        @test length(tr.details.pvalues) == 4
        for b in range(θ̂ - 1, θ̂ + 1; length=41)
            ps = dml_weak_iv_test(rr; beta0=b).details.pvalues
            @test (b in csr) == (count(>=(0.05), ps) >= 2)
        end
        # IIVM
        dfb, _ = iv_binary_dgp(StableRNG(20); n=2000)
        ri = dml_iivm(dfb, :y, :d, :z; covariates=[:x], rng=StableRNG(21))
        csi = dml_weak_iv_confidence_set(ri)
        @test coef(ri)[1] in csi
        # unidentified: zero first stage gives an unbounded set
        dz = mlfs_pliv_dgp(StableRNG(22); pi=[0.0], n=400)
        r0 = dml_pliv(dz, :y, :d, :z1; covariates=[:x1, :x2], rng=StableRNG(23),
                      outcome_learner=OLSLearner(), treatment_learner=OLSLearner(),
                      instrument_learner=OLSLearner())
        @test dml_weak_iv_confidence_set(r0).kind in (:union_of_rays, :real_line, :ray)
    end

    @testset "errors" begin
        df = mlfs_pliv_dgp(StableRNG(24))
        @test_throws ArgumentError ml_first_stage(df, :d, :z1; model=:foo)
        @test_throws ArgumentError ml_first_stage(df, :d, [:z1, :z2]; model=:iivm)
        @test_throws ArgumentError ml_first_stage(df, :d, :z1; model=:iivm)  # not binary
        @test_throws ArgumentError ml_first_stage(df, :d, Symbol[])
        @test_throws ArgumentError ml_first_stage(42)
        r2i = dml_pliv(df, :y, :d, [:z1, :z2]; covariates=[:x1], rng=StableRNG(1),
                       outcome_learner=OLSLearner(), treatment_learner=OLSLearner(),
                       instrument_learner=OLSLearner())
        @test_throws ArgumentError dml_weak_iv_test(r2i)
        @test_throws ArgumentError ml_first_stage(r2i)            # data required
        @test_throws DimensionMismatch ml_first_stage(r2i, df[1:10, :])
        riv = dml_pliv(df, :y, :d, :z1; covariates=[:x1], score=:iv_type,
                       rng=StableRNG(1), outcome_learner=OLSLearner(),
                       treatment_learner=OLSLearner(), instrument_learner=OLSLearner())
        @test_throws ArgumentError dml_weak_iv_confidence_set(riv)
        r1 = dml_pliv(df, :y, :d, :z1; covariates=[:x1], rng=StableRNG(1),
                      outcome_learner=OLSLearner(), treatment_learner=OLSLearner(),
                      instrument_learner=OLSLearner())
        @test_throws ArgumentError ml_first_stage(r1, df)          # instrument needed
        @test ml_first_stage(r1, df; instrument=:z1).instruments == ["z1"]
        @test_throws ArgumentError dml_weak_iv_confidence_set(r1; level=1.5)
        rp = dml_plr(df, :y, :d; covariates=[:x1], rng=StableRNG(1),
                     outcome_learner=OLSLearner(), treatment_learner=OLSLearner())
        @test_throws ArgumentError dml_weak_iv_test(rp)
    end

    @testset "Monte Carlo: DML-AR size under weak instruments" begin
        reps = mc_reps(1000, 150)
        rej_ar = 0
        rej_wald = 0
        cover_set = 0
        Fs = Float64[]
        for rep in 1:reps
            rng = StableRNG(3000 + rep)
            df = mlfs_pliv_dgp(rng; n=400, pi=[0.06], rho=0.9)
            ols = (outcome_learner=OLSLearner(), treatment_learner=OLSLearner(),
                   instrument_learner=OLSLearner())
            r = dml_pliv(df, :y, :d, :z1; covariates=[:x1, :x2], rng=rng, ols...)
            rej_ar += dml_weak_iv_test(r; beta0=1.0).pvalue < 0.05
            ci = confint(r)
            rej_wald += !(ci[1] <= 1.0 <= ci[2])
            cover_set += 1.0 in dml_weak_iv_confidence_set(r)
            push!(Fs, ml_first_stage(r, df; instrument=:z1).F)
        end
        se = sqrt(0.05 * 0.95 / reps)
        @test rej_ar / reps <= 0.05 + 3se
        @test rej_ar / reps >= 0.05 - 3se
        @test cover_set / reps >= 0.95 - 3se
        @test rej_wald / reps > 0.10          # the Wald interval is distorted
        @test median(Fs) < 10                 # and the diagnostic flags weakness
    end
end
