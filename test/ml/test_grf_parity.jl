# Parity of the generalized random forest post-estimation with the R package grf
# (2.6.1). grf's formulas are evaluated in R on DrSnow's nuisance estimates and
# out-of-bag predictions (test/validation/ml/grf_reference.R), so the results must
# agree to numerical precision.

@testset "grf post-estimation parity (R reference)" begin
    vdir = joinpath(@__DIR__, "..", "validation", "ml")
    df = ml_read_csv(joinpath(vdir, "grf_data.csv"))
    ref = ml_read_csv(joinpath(vdir, "grf_reference.csv"))
    getref(case, q) = ref.value[(ref.case .== case) .& (ref.quantity .== q)]
    xs = [:x1, :x2, :x3, :x4, :x5]
    X = Matrix(df[:, xs])
    small = (num_trees=20, rng=StableRNG(1))

    variants = Dict("plain" => (;), "cluster" => (cluster=:cl,),
                    "weights" => (weights=:sw,),
                    "equalize" => (cluster=:cl, equalize_cluster_weights=true))
    for (v, kw) in variants
        @testset "$v" begin
            cf0 = causal_forest(df, :y, :w; covariates=xs, y_hat=df.y_hat,
                                w_hat=df.w_hat, small..., kw...)
            cf = DrSnow._ml_grf_replace(cf0; predictions=df.tau)
            for ts in (:all, :treated, :control, :overlap)
                a = average_treatment_effect(cf; target=ts)
                @test coef(a)[1] ≈ getref(v, "ate_$(ts)_est")[1] rtol = 1e-10
                @test stderror(a)[1] ≈ getref(v, "ate_$(ts)_se")[1] rtol = 1e-8
            end
            a = average_treatment_effect(cf; subset=df.x2 .> 0.3)
            @test coef(a)[1] ≈ getref(v, "ate_subset_est")[1] rtol = 1e-10
            @test stderror(a)[1] ≈ getref(v, "ate_subset_se")[1] rtol = 1e-8
            b = best_linear_projection(cf, X[:, 1:2])
            @test coef(b) ≈ getref(v, "blp_coef") rtol = 1e-9
            @test stderror(b) ≈ getref(v, "blp_se") rtol = 1e-8
            @test coefnames(b) == ["(Intercept)", "A1", "A2"]
            b1 = best_linear_projection(cf, [:x1, :x2]; vcov_type=:HC1)
            @test stderror(b1) ≈ getref(v, "blp_hc1_se") rtol = 1e-8
            @test coefnames(b1) == ["(Intercept)", "x1", "x2"]
            bo = best_linear_projection(cf, X[:, 1]; target=:overlap)
            @test coef(bo) ≈ getref(v, "blp_overlap_coef") rtol = 1e-9
            @test stderror(bo) ≈ getref(v, "blp_overlap_se") rtol = 1e-8
            t = test_calibration(cf)
            tab = t.details.table
            @test tab.estimate ≈ getref(v, "cal_coef") rtol = 1e-9
            @test tab.std_error ≈ getref(v, "cal_se") rtol = 1e-8
            @test tab.t ≈ getref(v, "cal_t") rtol = 1e-8
            @test tab.p_value_one_sided ≈ getref(v, "cal_p") rtol = 1e-6
            @test t.pvalue == tab.p_value_one_sided[2]
            @test get_scores(cf) ≈ getref(v, "scores") rtol = 1e-10
            for tg in (:AUTOC, :QINI)
                r = rank_average_treatment_effect(cf, hcat(df.tau, df.prio2); target=tg,
                                                  R=10, rng=StableRNG(2))
                @test coef(r) ≈ getref(v, "rate_$(tg)") rtol = 1e-9 atol = 1e-12
                @test r.toc.estimate ≈ getref(v, "toc_$(tg)") rtol = 1e-9 atol = 1e-12
                r2 = rank_average_treatment_effect(cf, df.prio2; target=tg,
                                                   q=[0.05, 0.25, 0.33, 0.5, 0.9, 1],
                                                   R=10, subset=findall(df.x4 .> 0.2),
                                                   rng=StableRNG(2))
                @test coef(r2) ≈ getref(v, "rate_ties_$(tg)") rtol = 1e-9 atol = 1e-12
                @test r2.toc.estimate ≈ getref(v, "toc_ties_$(tg)") rtol = 1e-9 atol = 1e-12
            end
        end
    end

    @testset "continuous treatment" begin
        cc0 = causal_forest(df, :yc, :wc; covariates=xs, y_hat=df.yc_hat,
                            w_hat=df.wc_hat, small...)
        cc = DrSnow._ml_grf_replace(cc0; predictions=df.tauc)
        a = average_treatment_effect(cc; debiasing_weights=df.gammac)
        @test coef(a)[1] ≈ getref("continuous", "ate_all_est")[1] rtol = 1e-10
        @test stderror(a)[1] ≈ getref("continuous", "ate_all_se")[1] rtol = 1e-8
        @test estimand(a) == "average partial effect"
        ao = average_treatment_effect(cc; target=:overlap)
        @test coef(ao)[1] ≈ getref("continuous", "ate_overlap_est")[1] rtol = 1e-10
        @test stderror(ao)[1] ≈ getref("continuous", "ate_overlap_se")[1] rtol = 1e-8
        b = best_linear_projection(cc, X[:, 1]; debiasing_weights=df.gammac)
        @test coef(b) ≈ getref("continuous", "blp_coef") rtol = 1e-9
        @test stderror(b) ≈ getref("continuous", "blp_se") rtol = 1e-8
        @test_throws ArgumentError average_treatment_effect(cc; target=:treated)
    end

    @testset "instrumental forest" begin
        iv0 = instrumental_forest(df, :yiv, :d, :z; covariates=xs, y_hat=df.yiv_hat,
                                  w_hat=df.d_hat, z_hat=df.z_hat, small...)
        iv = DrSnow._ml_grf_replace(iv0; predictions=df.tauiv)
        a = average_treatment_effect(iv; compliance_score=df.compliance)
        @test coef(a)[1] ≈ getref("iv", "ate_all_est")[1] rtol = 1e-10
        @test stderror(a)[1] ≈ getref("iv", "ate_all_se")[1] rtol = 1e-8
        b = best_linear_projection(iv, X[:, 1:2]; compliance_score=df.compliance)
        @test coef(b) ≈ getref("iv", "blp_coef") rtol = 1e-9
        @test stderror(b) ≈ getref("iv", "blp_se") rtol = 1e-8
        @test get_scores(iv; compliance_score=df.compliance) ≈ getref("iv", "scores") rtol =
            1e-10
        @test_throws ArgumentError average_treatment_effect(iv; target=:treated)
    end

    @testset "regression forest calibration" begin
        rf0 = regression_forest(df, :y; covariates=xs, small...)
        rf = DrSnow._ml_grf_replace(rf0; predictions=df.y_hat)
        tab = test_calibration(rf).details.table
        @test tab.estimate ≈ getref("regression", "cal_coef") rtol = 1e-9
        @test tab.std_error ≈ getref("regression", "cal_se") rtol = 1e-8
        @test tab.p_value_one_sided ≈ getref("regression", "cal_p") rtol = 1e-6
    end
end
