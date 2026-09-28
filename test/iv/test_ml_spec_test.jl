using DrSnow: pvalue, Normal, cdf, ccdf
import MLJDecisionTreeInterface   # loads the DrSnowMLJExt extension (test dependency)
# Residual prediction specification tests (Scheidegger, Londschien & Bühlmann 2025).
# Reference: test/validation/iv/make_reference_rpiv.R (the authors' R package RPIV,
# replayed step by step so that its sample split and random-forest predictions can be
# fed to DrSnow through a fixed-prediction learner).

const RPIV_DF = iv_read_csv(joinpath(IV_VALDIR, "rpiv.csv"))
const RPIV_REF = let r = iv_read_csv(joinpath(IV_VALDIR, "reference_rpiv.csv"))
    Dict((row.case, row.quantity) => row.value for row in eachrow(r))
end

"""Learner returning given predictions (auxiliary rows, then main rows)."""
struct IVFixedPredLearner <: DrSnow.NuisanceLearner
    train::Vector{Float64}
    test::Vector{Float64}
end
function DrSnow.fitpredict(l::IVFixedPredLearner, X, y, Xnew; kwargs...)
    size(Xnew, 1) == length(l.train) + length(l.test) || error("unexpected call")
    return vcat(l.train, l.test)
end

function rpiv_learner(case)
    aux = RPIV_DF[!, "aux_$case"] .== 1
    pr = RPIV_DF[!, "pred_$case"]
    return IVFixedPredLearner(pr[aux], pr[.!aux]), aux
end

"""Paper-style DGP: confounded endogenous regressor, optional violation terms."""
function rpiv_dgp(rng; n=400, pi=1.0, viol=0.0, direct=0.0, k=2, hetero=true)
    Z = randn(rng, n, k)
    c = 0.5 .* Z[:, 1] .+ randn(rng, n)
    h = randn(rng, n)
    x = pi .* tanh.(vec(sum(Z; dims=2)) ./ sqrt(k)) .+ 0.3 .* c .+ h .+ randn(rng, n)
    e = randn(rng, n) .* (hetero ? abs.(Z[:, 1]) : 1.0)
    y = 2 .- x .+ 0.5 .* c .- h .+ e .+ viol .* Z[:, 1] .^ 2 .+ direct .* Z[:, 1]
    df = DataFrame(y=y, x=x, c=c, g=rand(rng, 1:50, n))
    for j in 1:k
        df[!, Symbol("z", j)] = Z[:, j]
    end
    return df
end

@testset "Residual prediction specification test" begin
    @testset "validation against RPIV (strong identification)" begin
        for (case, y, cl, ves) in
            (("strong_null", :y_null, nothing, (:heteroskedastic, :homoskedastic)),
             ("strong_alt", :y_alt, nothing, (:heteroskedastic, :homoskedastic)),
             ("strong_cluster", :y_null, :cl, (:cluster, :heteroskedastic)))
            l, aux = rpiv_learner(case)
            for v in ves
                t = residual_prediction_test(RPIV_DF, y, :x, [:z1, :z2]; covariates=[:c],
                                             learner=l, aux_sample=aux, variance=v,
                                             cluster=cl)
                @test t.statistic ≈ RPIV_REF[(case, "T_$v")] rtol = 1e-10 atol = 1e-12
                @test t.pvalue ≈ RPIV_REF[(case, "p_$v")] rtol = 1e-9 atol = 1e-14
                @test t.details.var_fraction ≈ RPIV_REF[(case, "varfrac_$v")] rtol = 1e-10
                @test t.details.n_aux == count(aux)
            end
        end
    end

    @testset "validation against RPIV (weak-IV robust, at β₀)" begin
        for (case, y, b0, cl, ves) in
            (("weak_b1", :y_null, 1.0, nothing, (:heteroskedastic, :homoskedastic)),
             ("weak_b0", :y_null, 0.0, nothing, (:heteroskedastic, :homoskedastic)),
             ("weak_alt_b1", :y_alt, 1.0, nothing, (:heteroskedastic,)),
             ("weak_cluster_b1", :y_null, 1.0, :cl, (:cluster, :heteroskedastic)))
            l, aux = rpiv_learner(case)
            for v in ves
                t = residual_prediction_test(RPIV_DF, y, :x, [:z1, :z2]; covariates=[:c],
                                             learner=l, aux_sample=aux, variance=v,
                                             cluster=cl, beta0=b0)
                @test t.statistic ≈ RPIV_REF[(case, "T_$v")] rtol = 1e-10 atol = 1e-12
                @test t.pvalue ≈ 1 - cdf(Normal(), RPIV_REF[(case, "T_$v")]) atol = 1e-12
                @test t.details.beta == [b0]
            end
        end
    end

    @testset "interface, reproducibility and row order" begin
        df = rpiv_dgp(StableRNG(1); n=300)
        t1 = residual_prediction_test(df, :y, :x, [:z1, :z2]; covariates=[:c],
                                      rng=StableRNG(5))
        t2 = residual_prediction_test(df, :y, :x, [:z1, :z2]; covariates=[:c],
                                      rng=StableRNG(5))
        @test t1.statistic == t2.statistic
        @test t1 isa DiagnosticTest
        @test occursin("not evidence", t1.note)
        @test !rejects(t1) || t1.pvalue < 0.05
        # explicit split + deterministic learner: invariant to shuffling the rows
        aux = rand(StableRNG(2), 300) .< 0.45
        df.aux = aux
        kw = (covariates=[:c], learner=RidgeLearner(lambda=0.5), aux_sample=:aux)
        s0 = residual_prediction_test(df, :y, :x, [:z1, :z2]; kw...)
        w0 = residual_prediction_test(df, :y, :x, [:z1, :z2]; beta0=-1.0, kw...)
        perm = randperm(StableRNG(3), 300)
        dp = df[perm, :]
        @test residual_prediction_test(dp, :y, :x, [:z1, :z2]; kw...).statistic ≈
              s0.statistic rtol = 1e-8
        @test residual_prediction_test(dp, :y, :x, [:z1, :z2]; beta0=-1.0,
                                       kw...).statistic ≈ w0.statistic rtol = 1e-8
        # Bool vector split equals the column split
        @test residual_prediction_test(df, :y, :x, [:z1, :z2]; covariates=[:c],
                                       learner=RidgeLearner(lambda=0.5),
                                       aux_sample=aux).statistic == s0.statistic
        # :linear equals :refit for a linear smoother with a fixed penalty
        lr = residual_prediction_test(df, :y, :x, [:z1, :z2]; beta0=-0.5,
                                      weight_update=:linear, kw...)
        rf = residual_prediction_test(df, :y, :x, [:z1, :z2]; beta0=-0.5,
                                      weight_update=:refit, kw...)
        @test lr.statistic ≈ rf.statistic rtol = 1e-8
        @test residual_prediction_test(df, :y, :x, [:z1, :z2]; beta0=-0.5,
                                       weight_update=:fixed, kw...) isa DiagnosticTest
        # sign weights (clip_quantile = 0) and missing rows
        @test residual_prediction_test(df, :y, :x, [:z1, :z2]; clip_quantile=0,
                                       kw...) isa DiagnosticTest
        dm = allowmissing(df)
        dm[1, :y] = missing
        @test residual_prediction_test(dm, :y, :x, [:z1, :z2]; covariates=[:c],
                                       rng=StableRNG(1)).details.n_aux +
              residual_prediction_test(dm, :y, :x, [:z1, :z2]; covariates=[:c],
                                       rng=StableRNG(1)).details.n_main == 299
        # cluster-level split keeps clusters intact
        tc = residual_prediction_test(df, :y, :x, [:z1, :z2]; covariates=[:c],
                                      cluster=:g, rng=StableRNG(4))
        a = tc.details.aux_sample
        @test all(g -> length(unique(a[df.g .== g])) == 1, unique(df.g))
        @test tc.details.variance === :cluster
        # a learner that predicts nothing gives a floored, zero statistic
        z0 = residual_prediction_test(df, :y, :x, [:z1, :z2]; learner=MeanLearner(),
                                      beta0=-1.0, rng=StableRNG(1))
        @test isfinite(z0.statistic)
    end

    @testset "errors" begin
        df = rpiv_dgp(StableRNG(6); n=200)
        @test_throws ArgumentError residual_prediction_test(df, :y, :x, [:nope])
        @test_throws ArgumentError residual_prediction_test(df, :y, [:x, :c], :z1)
        @test_throws ArgumentError residual_prediction_test(df, :y, :x, :z1;
                                                            clip_quantile=1.5)
        @test_throws ArgumentError residual_prediction_test(df, :y, :x, :z1; gamma=-1)
        @test_throws ArgumentError residual_prediction_test(df, :y, :x, :z1;
                                                            variance=:cluster)
        @test_throws ArgumentError residual_prediction_test(df, :y, :x, :z1;
                                                            variance=:hc3)
        @test_throws ArgumentError residual_prediction_test(df, :y, :x, :z1;
                                                            aux_fraction=1.2)
        @test_throws DimensionMismatch residual_prediction_test(df, :y, :x, :z1;
                                                                beta0=[1.0, 2.0])
        @test_throws DimensionMismatch residual_prediction_test(df, :y, :x, :z1;
                                                                aux_sample=trues(3))
        @test_throws ArgumentError residual_prediction_test(df, :y, :x, :z1; beta0=1.0,
                                                            weight_update=:magic)
        @test_throws ArgumentError residual_prediction_test(df, :y, :x, :z1;
                                                            aux_fraction=0.001)
        @test_throws ArgumentError residual_prediction_confidence_set(df, :y, [:x, :c],
                                                                      [:z1, :z2])
        @test_throws ArgumentError residual_prediction_confidence_set(df, :y, :x, :z1;
                                                                      level=1.0)
    end

    @testset "Monte Carlo: size and power (forest weights)" begin
        reps = mc_reps(500, 80)
        rej_null = 0
        rej_alt = 0
        rej_lin = 0
        for r in 1:reps
            rng = StableRNG(1000 + r)
            d0 = rpiv_dgp(rng; n=500)
            t0 = residual_prediction_test(d0, :y, :x, [:z1, :z2]; covariates=[:c],
                                          learner=ForestLearner(num_trees=100), rng=rng)
            rej_null += t0.pvalue < 0.05
            d1 = rpiv_dgp(rng; n=500, viol=0.5)
            t1 = residual_prediction_test(d1, :y, :x, [:z1, :z2]; covariates=[:c],
                                          learner=ForestLearner(num_trees=100), rng=rng)
            rej_alt += t1.pvalue < 0.05
            # just identified with a *linear* direct effect: undetectable (Lemma 1)
            d2 = rpiv_dgp(rng; n=500, direct=0.5, k=1)
            t2 = residual_prediction_test(d2, :y, :x, :z1; covariates=[:c],
                                          learner=ForestLearner(num_trees=100), rng=rng)
            rej_lin += t2.pvalue < 0.05
        end
        se = sqrt(0.05 * 0.95 / reps)
        @test rej_null / reps <= 0.05 + 3se
        @test rej_lin / reps <= 0.05 + 3se
        @test rej_alt / reps >= 0.8
    end

    @testset "Monte Carlo: weak-IV-robust test and confidence set" begin
        reps = mc_reps(300, 40)
        rej = 0
        cover = 0
        empty_alt = 0
        for r in 1:reps
            rng = StableRNG(2000 + r)
            # very weak instruments: size of the test at the true β = −1
            dw = rpiv_dgp(rng; n=500, pi=0.1)
            tw = residual_prediction_test(dw, :y, :x, [:z1, :z2]; covariates=[:c],
                                          beta0=-1.0,
                                          learner=ForestLearner(num_trees=100), rng=rng)
            rej += tw.pvalue < 0.05
            ds = rpiv_dgp(rng; n=500, pi=1.5)
            cs = residual_prediction_confidence_set(ds, :y, :x, [:z1, :z2];
                                                    covariates=[:c], n_grid=31,
                                                    weight_update=:linear,
                                                    learner=RidgeLearner(lambda=1.0),
                                                    rng=rng)
            cover += (-1.0 in cs)
        end
        se = sqrt(0.05 * 0.95 / reps)
        @test rej / reps <= 0.05 + 3se
        @test cover / reps >= 0.95 - 3se
        # a strongly misspecified model gives an empty set
        d = rpiv_dgp(StableRNG(77); n=2000, pi=1.5, viol=1.5)
        cs = residual_prediction_confidence_set(d, :y, :x, [:z1, :z2]; covariates=[:c],
                                                weight_update=:linear,
                                                learner=ForestLearner(num_trees=100),
                                                rng=StableRNG(78))
        @test cs.kind === :empty
        @test pvalue(cs, cs.estimate) < 0.05
        # the p-value function agrees with the test at a point
        dd = rpiv_dgp(StableRNG(79); n=400)
        cs2 = residual_prediction_confidence_set(dd, :y, :x, [:z1, :z2]; covariates=[:c],
                                                 weight_update=:linear,
                                                 learner=RidgeLearner(lambda=1.0),
                                                 rng=StableRNG(80))
        t = residual_prediction_test(dd, :y, :x, [:z1, :z2]; covariates=[:c], beta0=-0.8,
                                     weight_update=:linear,
                                     learner=RidgeLearner(lambda=1.0), rng=StableRNG(80))
        @test pvalue(cs2, -0.8) ≈ t.pvalue rtol = 1e-10
        @test cs2.kind === :bounded
        lo, hi = cs2.intervals[1]
        @test pvalue(cs2, lo - 1e-6 * abs(lo)) < 0.05 + 1e-6
        @test pvalue(cs2, (lo + hi) / 2) >= 0.05
    end

    @testset "MLJ model as the residual learner" begin
        rf = MLJLearner(MLJDecisionTreeInterface.RandomForestRegressor(n_trees=50))
        d = rpiv_dgp(StableRNG(81); n=1000, viol=1.0)
        t = residual_prediction_test(d, :y, :x, [:z1, :z2]; covariates=[:c], learner=rf,
                                     rng=StableRNG(82))
        @test t.pvalue < 0.01
        @test occursin("RandomForestRegressor", t.details.learner)
    end
end
