# Validation of the IV area against R reference values
# (test/validation/iv/make_reference.R: AER, sandwich, ivmodel, fixest, momentfit).

const IV_VALDIR = joinpath(@__DIR__, "..", "validation", "iv")

"""Minimal reader for the numeric / quoted-string CSV files written by R."""
function iv_read_csv(path)
    lines = readlines(path)
    hdr = Symbol.(strip.(split(lines[1], ','), '"'))
    cols = [Any[] for _ in hdr]
    for ln in lines[2:end]
        isempty(strip(ln)) && continue
        for (j, f) in enumerate(split(ln, ','))
            s = strip(f, '"')
            v = tryparse(Float64, s)
            push!(cols[j], v === nothing ? (s == "NA" ? missing : String(s)) : v)
        end
    end
    df = DataFrame()
    for (h, c) in zip(hdr, cols)
        df[!, h] = all(x -> x isa Float64, c) ? Float64.(c) : c
    end
    return df
end

const IV_REF = let r = iv_read_csv(joinpath(IV_VALDIR, "reference.csv"))
    d = Dict{Tuple{String,String},Float64}()
    for row in eachrow(r)
        get!(d, (row.case, row.quantity), row.value)
    end
    d
end
ivref(case, q) = IV_REF[(case, q)]

const CARD = iv_read_csv(joinpath(IV_VALDIR, "card.csv"))
const MROZ = iv_read_csv(joinpath(IV_VALDIR, "mroz.csv"))
const CARD_CTRL = [:exper, :expersq, :black, :smsa, :south]

se_of(r, name) = stderror(r)[findfirst(==(name), coefnames(r))]

@testset "Validation against R (Card 1995, Mroz 1987)" begin
    @testset "2SLS coefficients and SEs (AER::ivreg + sandwich)" begin
        for (case, inst) in (("card_ji", [:nearc4]), ("card_oi", [:nearc4, :nearc2]))
            r_iid = late_2sls(CARD, :lwage, :educ, inst; covariates=CARD_CTRL,
                              vcov=Vcov.simple())
            r_hc1 = late_2sls(CARD, :lwage, :educ, inst; covariates=CARD_CTRL)
            r_cl = late_2sls(CARD, :lwage, :educ, inst; covariates=CARD_CTRL,
                             cluster=:region)
            @test coef(r_iid)[1] ≈ ivref(case, "coef_educ") rtol = 1e-9
            @test se_of(r_iid, "educ") ≈ ivref(case, "se_educ_iid") rtol = 1e-8
            @test se_of(r_hc1, "educ") ≈ ivref(case, "se_educ_hc1") rtol = 1e-8
            @test se_of(r_cl, "educ") ≈ ivref(case, "se_educ_cl") rtol = 1e-8
            if case == "card_ji"
                @test se_of(r_iid, "exper") ≈ ivref(case, "se_exper_iid") rtol = 1e-8
                @test se_of(r_hc1, "exper") ≈ ivref(case, "se_exper_hc1") rtol = 1e-8
                @test se_of(r_cl, "exper") ≈ ivref(case, "se_exper_cl") rtol = 1e-8
            end
            fs = r_iid.first_stage.first_stage[1]
            @test fs.F_homoskedastic ≈ ivref(case, "weak_F_iid") rtol = 1e-8
            @test fs.F ≈ ivref(case, "weak_F_iid") rtol = 1e-8
            wh = endogeneity_test(r_iid)
            @test wh.statistic ≈ ivref(case, "wu_hausman_F_iid") rtol = 1e-8
        end
        mz = iv_regression(MROZ, :lwage, :educ, [:fatheduc, :motheduc];
                           covariates=[:exper, :expersq], vcov=Vcov.simple())
        @test coef(mz)[1] ≈ ivref("mroz", "coef_educ") rtol = 1e-9
        @test se_of(mz, "educ") ≈ ivref("mroz", "se_educ_iid") rtol = 1e-8
        @test overidentification_test(mz).statistic ≈ ivref("mroz", "sargan") rtol = 1e-8
        @test endogeneity_test(mz).statistic ≈ ivref("mroz", "wu_hausman_F_iid") rtol = 1e-8
    end

    @testset "fixed effects, weights, clustering (fixest)" begin
        r = late_2sls(CARD, :lwage, :educ, [:nearc4, :nearc2]; covariates=CARD_CTRL,
                      fe=[:region], weights=:weight, cluster=:cl50)
        @test coef(r)[1] ≈ ivref("card_fe", "coef_educ") rtol = 1e-7
        @test se_of(r, "educ") ≈ ivref("card_fe", "se_educ") rtol = 1e-6
        @test se_of(r, "exper") ≈ ivref("card_fe", "se_exper") rtol = 1e-6
        r2 = late_2sls(CARD, :lwage, :educ, [:nearc4, :nearc2]; covariates=CARD_CTRL,
                       cluster=[:region, :cl50])
        @test coef(r2)[1] ≈ ivref("card_2w", "coef_educ") rtol = 1e-9
        @test se_of(r2, "educ") ≈ ivref("card_2w", "se_educ") rtol = 1e-6
    end

    @testset "two endogenous regressors" begin
        ctrl = [:exper, :black, :smsa, :south]
        inst = [:nearc4, :nearc2, :agesq]
        r = iv_regression(CARD, :lwage, [:educ, :expersq], inst; covariates=ctrl,
                          vcov=Vcov.simple())
        @test coef(r)[1:2] ≈ [ivref("card_p2", "coef_educ"),
                              ivref("card_p2", "coef_expersq")] rtol = 1e-8
        @test se_of(r, "educ") ≈ ivref("card_p2", "se_educ_iid") rtol = 1e-8
        fs = r.first_stage
        @test fs.cragg_donald_F ≈ ivref("card_p2", "cragg_donald_F") rtol = 1e-8
        @test fs.first_stage[1].sanderson_windmeijer_F ≈
              ivref("card_p2", "sw_F_educ") rtol = 1e-8
        @test fs.first_stage[2].sanderson_windmeijer_F ≈
              ivref("card_p2", "sw_F_expersq") rtol = 1e-8
        @test fs.effective_F === nothing
        rh = iv_regression(CARD, :lwage, [:educ, :expersq], inst; covariates=ctrl)
        @test se_of(rh, "expersq") ≈ ivref("card_p2", "se_expersq_hc1") rtol = 1e-8
        # exper = age - educ - 6: with age as an instrument FixedEffectModels would
        # silently treat exper as exogenous; DrSnow refuses the specification.
        @test_throws ArgumentError iv_regression(CARD, :lwage, [:educ, :exper],
                                                 [:nearc4, :nearc2, :age];
                                                 covariates=[:black, :smsa, :south])
    end

    @testset "Anderson–Rubin tests and sets (ivmodel; brute-force robust)" begin
        for (case, inst) in (("card_ji", [:nearc4]), ("card_oi", [:nearc4, :nearc2]))
            r = late_2sls(CARD, :lwage, :educ, inst; covariates=CARD_CTRL,
                          vcov=Vcov.simple())
            t = weak_iv_test(r; beta0=0.0)
            @test t.statistic ≈ ivref(case, "ar_F_iid_b0") rtol = 1e-8
            case == "card_ji" && @test t.pvalue ≈ ivref(case, "ar_p_iid_b0") rtol = 1e-6
            cs = weak_iv_confidence_set(r)
            @test cs.kind === :bounded
            @test cs.intervals[1][1] ≈ ivref(case, "ar_lower_iid") rtol = 1e-7
            @test cs.intervals[1][2] ≈ ivref(case, "ar_upper_iid") rtol = 1e-7
            for (tp, kw) in (("hc1", NamedTuple()), ("cl", (cluster=:region,)))
                rr = late_2sls(CARD, :lwage, :educ, inst; covariates=CARD_CTRL, kw...)
                @test weak_iv_test(rr).statistic ≈ ivref(case, "ar_F_$(tp)_b0") rtol = 1e-8
                s = weak_iv_confidence_set(rr)
                @test ivref(case, "ar_nroots_$tp") == 2
                @test ivref(case, "ar_left_accepted_$tp") == 0
                @test s.kind === :bounded
                @test s.intervals[1][1] ≈ ivref(case, "ar_root1_$tp") rtol = 1e-7
                @test s.intervals[1][2] ≈ ivref(case, "ar_root2_$tp") rtol = 1e-7
            end
        end
    end

    @testset "CLR (ivmodel)" begin
        r = late_2sls(CARD, :lwage, :educ, [:nearc4, :nearc2]; covariates=CARD_CTRL,
                      vcov=Vcov.simple())
        t = weak_iv_test(r; beta0=0.0, method=:clr)
        @test t.statistic ≈ ivref("card_oi", "clr_stat_b0") rtol = 1e-8
        @test t.pvalue ≈ ivref("card_oi", "clr_p_b0") rtol = 1e-4
        @test weak_iv_test(r; beta0=0.2, method=:clr).pvalue ≈
              ivref("card_oi", "clr_p_b02") rtol = 1e-4
        cs = weak_iv_confidence_set(r; method=:clr)
        @test cs.kind === :bounded
        @test cs.intervals[1][1] ≈ ivref("card_oi", "clr_lower") rtol = 1e-4
        @test cs.intervals[1][2] ≈ ivref("card_oi", "clr_upper") rtol = 1e-4
    end

    @testset "effective F, Olea–Pflueger critical values, tF, Hansen J" begin
        r = late_2sls(CARD, :lwage, :educ, [:nearc4, :nearc2]; covariates=CARD_CTRL)
        fs = r.first_stage
        @test fs.effective_F ≈ ivref("card_oi", "eff_F_hc1") rtol = 1e-8
        @test fs.first_stage[1].F ≈ ivref("card_oi", "fs_wald_F_hc1") rtol = 1e-8
        cv = fs.op_critical_values
        @test cv.tau_5 ≈ ivref("card_oi", "op_cv_0.05") rtol = 1e-6
        @test cv.tau_10 ≈ ivref("card_oi", "op_cv_0.1") rtol = 1e-6
        @test cv.tau_20 ≈ ivref("card_oi", "op_cv_0.2") rtol = 1e-6
        @test cv.tau_30 ≈ ivref("card_oi", "op_cv_0.3") rtol = 1e-6
        J = overidentification_test(r)
        @test J.statistic ≈ ivref("card_oi", "hansen_J") rtol = 1e-8
        @test J.details.gmm_coef[1] ≈ ivref("card_oi", "gmm_coef_educ") rtol = 1e-8
        rj = late_2sls(CARD, :lwage, :educ, :nearc4; covariates=CARD_CTRL)
        @test rj.first_stage.first_stage[1].F ≈ ivref("card_ji", "fs_F_hc1") rtol = 1e-8
        @test rj.first_stage.effective_F ≈ ivref("card_ji", "fs_F_hc1") rtol = 1e-8
        tf = tf_confint(rj)
        @test tf.critical_value ≈ ivref("card_ji", "tF_cF") rtol = 1e-9
        @test tf.lower ≈ ivref("card_ji", "tF_lower") atol = 1e-9
        @test tf.upper ≈ ivref("card_ji", "tF_upper") atol = 1e-9
    end

    @testset "plausibly exogenous (LTZ, UCI)" begin
        r = late_2sls(CARD, :lwage, :educ, :nearc4; covariates=CARD_CTRL)
        ltz = plausibly_exogenous(r; method=:ltz, gamma_mean=0.01, gamma_vcov=0.01^2)
        @test ltz.adjustment[1] ≈ ivref("card_ji", "ltz_A") rtol = 1e-8
        @test ltz.estimate ≈ ivref("card_ji", "ltz_est") rtol = 1e-8
        @test ltz.se ≈ ivref("card_ji", "ltz_se") rtol = 1e-8
        uci = plausibly_exogenous(r; method=:uci, gamma=(0.0, 0.02))
        @test uci.lower ≈ ivref("card_ji", "uci_lower") rtol = 1e-7
        @test uci.upper ≈ ivref("card_ji", "uci_upper") rtol = 1e-7
    end

    @testset "compliance, complier means, IPW LATE" begin
        ca = estimate_compliance(CARD, :somecol, :nearc4)
        @test coef(ca) ≈ [ivref("card_bin", "share_compliers"),
                          ivref("card_bin", "share_always"),
                          ivref("card_bin", "share_never")] rtol = 1e-10
        @test stderror(ca)[1] ≈ ivref("card_bin", "se_compliers") rtol = 1e-8
        @test stderror(ca)[2] ≈ ivref("card_bin", "se_always") rtol = 1e-8
        prof = complier_characteristics(CARD, :somecol, :nearc4, [:black, :exper, :south])
        for (i, v) in enumerate(("black", "exper", "south"))
            row = prof.table[i, :]
            @test row.complier_mean ≈ ivref("card_bin", "complier_mean_$v") rtol = 1e-9
            @test row.complier_se ≈ ivref("card_bin", "complier_se_$v") rtol = 1e-8
            @test row.always_taker_mean ≈ ivref("card_bin", "always_mean_$v") rtol = 1e-10
            @test row.never_taker_mean ≈ ivref("card_bin", "never_mean_$v") rtol = 1e-10
            @test row.population_mean ≈ ivref("card_bin", "pop_mean_$v") rtol = 1e-10
        end
        lw = late_ipw(CARD, :lwage, :somecol, :nearc4)
        @test estimate(lw) ≈ ivref("card_bin", "late_wald") rtol = 1e-9
        @test stderror(lw)[1] ≈ ivref("card_bin", "late_wald_se") rtol = 1e-8
        li = late_ipw(CARD, :lwage, :somecol, :nearc4;
                      covariates=[:black, :smsa, :south, :exper])
        @test coef(li) ≈ [ivref("card_bin", "ipw_late"), ivref("card_bin", "ipw_y1c"),
                          ivref("card_bin", "ipw_y0c")] rtol = 1e-7
        @test li.complier_share ≈ ivref("card_bin", "ipw_share") rtol = 1e-7
        # influence-function SEs vs 2000-draw nonparametric bootstrap in R
        # (robust bootstrap spread IQR/1.349; Monte Carlo coverage is checked in
        # test_compliance.jl)
        @test stderror(li)[1] ≈ ivref("card_bin", "ipw_late_boot_se") rtol = 0.1
        @test stderror(li)[2] ≈ ivref("card_bin", "ipw_y1c_boot_se") rtol = 0.1
        @test stderror(li)[3] ≈ ivref("card_bin", "ipw_y0c_boot_se") rtol = 0.1
        @test li.complier_share_se ≈ ivref("card_bin", "ipw_share_boot_se") rtol = 0.05
    end

    @testset "LATE extrapolation (cells = black)" begin
        ex = @test_logs (:warn, r"weak first stage") match_mode = :any begin
            late_extrapolation(CARD, :lwage, :somecol, :nearc4, [:black];
                               targets=[:compliers, :population, :treated])
        end
        for (j, t) in enumerate(("compliers", "population", "treated"))
            @test coef(ex)[j] ≈ ivref("card_ext", t) rtol = 1e-9
            @test stderror(ex)[j] ≈ ivref("card_ext", t * "_delta_se") rtol = 1e-5
            # one cell has a weak first stage (t ≈ 2.7), so the bootstrap distribution
            # of the ratio is heavy-tailed; the delta-method SE is only roughly equal
            # to the robust bootstrap spread here
            @test stderror(ex)[j] ≈ ivref("card_ext", t * "_boot_se") rtol = 0.2
        end
    end
end
