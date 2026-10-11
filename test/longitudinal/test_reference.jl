# Parity with ltmle 1.3.0 (exact) and a cross-check with lmtp 1.5.4 on the committed
# validation data (test/validation/longitudinal).

@testset "ltmle parity" begin
    df = lg_read_csv("longitudinal_data.csv")
    ref = lg_read_csv("ltmle_reference.csv")
    n = 1000
    adj = sqrt(n / (n - 1))                 # ltmle's var() divides by n - 1
    getref(case, est) = begin
        r = ref[(ref.case .== case) .& (ref.estimator .== est), :]
        (r.estimate[1], r.std_error[1])
    end
    fit(outcome, regime; est=:tmle) = longitudinal_regime(df, outcome, :a, :id, :t;
        regime=regime, covariates=[:l1, :l2], baseline=[:w1, :w2], estimator=est,
        outcome_learner=FractionalLogitLearner(), treatment_learner=LogisticLearner(),
        crossfit=false)
    dyn = DynamicRegime(h -> h.covariates[end].l1 > 0 ? 1 : 0; label="dynamic")
    cases = [("cont_always", :y_cont, always_treat(3)),
             ("cont_never", :y_cont, never_treat(3)),
             ("bin_always", :y_bin, always_treat(3)),
             ("bin_never", :y_bin, never_treat(3)),
             ("cont_treat_once", :y_cont, StaticRegime([1, 0, 0])),
             ("cont_dynamic", :y_cont, dyn),
             ("bin_dynamic", :y_bin, dyn)]
    for (case, oc, reg) in cases
        r = fit(oc, reg)
        e, s = getref(case, "tmle")
        @test coef(r)[1] ≈ e atol = 5e-8
        @test stderror(r)[1] * adj ≈ s rtol = 1e-7
        ri = fit(oc, reg; est=:ipw)
        e, s = getref(case, "iptw")
        @test coef(ri)[1] ≈ e atol = 5e-8
        # ltmle's IPTW influence curve omits the Hájek normalization by mean(w)
        mw = mean(ri.weights[1][:, 3])
        @test stderror(ri)[1] * mw * adj ≈ s rtol = 1e-7
    end
    for (case, oc) in [("cont_always_gcomp", :y_cont), ("bin_always_gcomp", :y_bin)]
        r = fit(oc, always_treat(3); est=:gcomp)
        @test coef(r)[1] ≈ getref(case, "gcomp")[1] atol = 5e-8
    end
    # two regimes: joint fit and effect measures
    for (case, oc) in [("cont_pair", :y_cont), ("bin_pair", :y_bin)]
        r = fit(oc, [always_treat(3), never_treat(3)])
        @test coef(r)[1] ≈ getref(case, "tmle")[1] atol = 5e-8
        @test coef(r)[2] ≈ getref(case * "_control", "tmle")[1] atol = 5e-8
        c = regime_contrast(r; reference=2)
        e, s = getref(case * "_ATE", "tmle")
        @test coef(c)[1] ≈ e atol = 5e-8
        @test stderror(c)[1] * adj ≈ s rtol = 1e-7
        if oc === :y_bin
            rr = regime_contrast(r; reference=2, type=:ratio)
            e, s = getref(case * "_logRR", "tmle")
            @test coef(rr)[1] ≈ e atol = 5e-8
            @test stderror(rr)[1] * adj ≈ s rtol = 1e-7
            or = regime_contrast(r; reference=2, type=:odds_ratio)
            e, s = getref(case * "_logOR", "tmle")
            @test coef(or)[1] ≈ e atol = 5e-8
            @test stderror(or)[1] * adj ≈ s rtol = 1e-7
        end
        ri = fit(oc, [always_treat(3), never_treat(3)]; est=:ipw)
        ci = regime_contrast(ri; reference=2)
        @test coef(ci)[1] ≈ getref(case * "_ATE", "iptw")[1] atol = 5e-8
    end
end

@testset "lmtp cross-check" begin
    # lmtp estimates density ratios by classification of stacked observed / shifted
    # data, not from propensity scores, so agreement is approximate.
    df = lg_read_csv("longitudinal_data.csv")
    lref = lg_read_csv("lmtp_reference.csv")
    for (case, oc, reg) in [("cont_always", :y_cont, always_treat(3)),
                            ("cont_never", :y_cont, never_treat(3)),
                            ("bin_always", :y_bin, always_treat(3)),
                            ("bin_never", :y_bin, never_treat(3))]
        for est in (:sdr, :tmle)
            r = longitudinal_regime(df, oc, :a, :id, :t; regime=reg,
                                    covariates=[:l1, :l2], baseline=[:w1, :w2],
                                    estimator=est, outcome_learner=OLSLearner(),
                                    treatment_learner=LogisticLearner(), crossfit=false)
            l = lref[(lref.case .== case) .& (lref.estimator .== string(est)), :]
            @test coef(r)[1] ≈ l.estimate[1] rtol = 0.05
            @test stderror(r)[1] ≈ l.std_error[1] rtol = 0.06
        end
    end
end
