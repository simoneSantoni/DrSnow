# Parity with the R package DoubleML (1.0.2) using deterministic learners (OLS /
# logistic MLE) and identical sample splits. Reference values are produced by
# test/validation/ml/doubleml_reference.R from the committed CSV files.

@testset "DoubleML parity (R reference)" begin
    vdir = joinpath(@__DIR__, "..", "validation", "ml")
    df = ml_read_csv(joinpath(vdir, "dml_data.csv"))
    fdf = ml_read_csv(joinpath(vdir, "dml_folds.csv"))
    F = hcat(Int.(fdf.fold_rep1), Int.(fdf.fold_rep2))
    ref = ml_read_csv(joinpath(vdir, "dml_reference.csv"))
    xs = [:x1, :x2, :x3, :x4, :x5]
    ols = OLSLearner()
    logit = LogisticLearner()

    function check(case, r::DMLEstimate)
        rows = ref[ref.case .== case, :]
        for (j, nm) in enumerate(coefnames(r))
            sub = rows[rows.treatment .== nm, :]
            per = sort(sub[sub.rep .> 0, :], :rep)
            @test r.all_coef[j, :] ≈ per.coef rtol = 1e-8
            @test r.all_se[j, :] ≈ per.se rtol = 1e-7
            agg = sub[sub.rep .== 0, :]
            # point estimate: median over repetitions, as in DoubleML
            @test coef(r)[j] ≈ agg.coef[1] rtol = 1e-8
            # DoubleML 1.0.2 aggregates se² = median(se_r² + (θ_r - θ̃)²/n); DrSnow
            # uses the variance-scale rule median(se_r² + (θ_r - θ̃)²) (see docs), so
            # compare DoubleML's number with its formula applied to our per-rep values
            n = nobs(r)
            dev2 = (r.all_coef[j, :] .- coef(r)[j]) .^ 2
            dm = sqrt(median(r.all_se[j, :] .^ 2 .+ dev2 ./ n))
            @test dm ≈ agg.se[1] rtol = 1e-7
            ours = sqrt(median(r.all_se[j, :] .^ 2 .+ dev2))
            @test stderror(r)[j] ≈ ours rtol = 1e-12
        end
    end

    check("plr_po", dml_plr(df, :y_plr, :d1; covariates=xs, outcome_learner=ols,
                            treatment_learner=ols, folds=F))
    check("plr_ivtype", dml_plr(df, :y_plr, :d1; covariates=xs, outcome_learner=ols,
                                treatment_learner=ols, g_learner=ols, score=:iv_type,
                                folds=F))
    check("plr_multi", dml_plr(df, :y_plr, [:d1, :d2]; covariates=xs,
                               outcome_learner=ols, treatment_learner=ols, folds=F))
    check("irm_ate", dml_irm(df, :y_irm, :d_irm; covariates=xs, outcome_learner=ols,
                             propensity_learner=logit, trim=1e-12, folds=F))
    check("irm_atte", dml_irm(df, :y_irm, :d_irm; covariates=xs, outcome_learner=ols,
                              propensity_learner=logit, trim=1e-12, score=:ATTE,
                              folds=F))
    check("pliv_po", dml_pliv(df, :y_pliv, :d_pliv, :z1; covariates=xs,
                              outcome_learner=ols, treatment_learner=ols,
                              instrument_learner=ols, folds=F))
    check("pliv_ivtype", dml_pliv(df, :y_pliv, :d_pliv, :z1; covariates=xs,
                                  outcome_learner=ols, treatment_learner=ols,
                                  instrument_learner=ols, g_learner=ols,
                                  score=:iv_type, folds=F))
    check("pliv_2z", dml_pliv(df, :y_pliv, :d_pliv, [:z1, :z2]; covariates=xs,
                              outcome_learner=ols, treatment_learner=ols,
                              instrument_learner=ols, folds=F))
    check("iivm", dml_iivm(df, :y_iv, :d_iv, :z_iv; covariates=xs, outcome_learner=ols,
                           instrument_learner=logit, treatment_learner=logit,
                           trim=1e-12, folds=F))
    check("iivm_onesided", dml_iivm(df, :y_os, :d_os, :z_iv; covariates=xs,
                                    outcome_learner=ols, instrument_learner=logit,
                                    treatment_learner=logit, always_takers=false,
                                    trim=1e-12, folds=F))
end
