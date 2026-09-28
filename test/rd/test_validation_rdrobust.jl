# Validation of rd_estimate / rd_bandwidth against rdrobust 4.0.0 (R).
# Reference values: test/validation/rd/reference.csv (generate_reference.R).

const RD_SEN_COVS3 = [:presdemvoteshlag1, :demvoteshlag1, :demvoteshlag2]

# case => (data, outcome, running, keyword arguments)
const RD_ESTIMATE_CASES = Any[
    ("senate_default", RD_SENATE, :vote, :margin, (;)),
    [("senate_bw_$m", RD_SENATE, :vote, :margin, (; bwselect=m))
     for m in (:mserd, :msetwo, :msesum, :msecomb1, :msecomb2, :cerrd, :certwo, :cersum,
               :cercomb1, :cercomb2)]...,
    ("senate_epa", RD_SENATE, :vote, :margin, (; kernel=:epa)),
    ("senate_uni", RD_SENATE, :vote, :margin, (; kernel=:uni)),
    ("senate_p2", RD_SENATE, :vote, :margin, (; p=2)),
    ("senate_p0", RD_SENATE, :vote, :margin, (; p=0)),
    ("senate_p3q5", RD_SENATE, :vote, :margin, (; p=3, q=5)),
    [("senate_$v", RD_SENATE, :vote, :margin, (; vce=v))
     for v in (:hc0, :hc1, :hc2, :hc3)]...,
    ("senate_nn5", RD_SENATE, :vote, :margin, (; nnmatch=5)),
    ("senate_cluster", RD_SENATE, :vote, :margin, (; cluster=:state)),
    ("senate_covs", RD_SENATE, :vote, :margin, (; covariates=RD_SEN_COVS3)),
    ("senate_covs_cluster", RD_SENATE, :vote, :margin,
     (; covariates=[:presdemvoteshlag1, :demvoteshlag1], cluster=:state)),
    ("senate_covs_hc2", RD_SENATE, :vote, :margin,
     (; covariates=[:presdemvoteshlag1], vce=:hc2)),
    ("senate_h10", RD_SENATE, :vote, :margin, (; h=10)),
    ("senate_h_b_twosided", RD_SENATE, :vote, :margin, (; h=(10, 15), b=(20, 25))),
    ("senate_rho", RD_SENATE, :vote, :margin, (; h=12, rho=0.5)),
    ("senate_rho_est", RD_SENATE, :vote, :margin, (; rho=0.8)),
    ("senate_mass_off", RD_SENATE, :vote, :margin, (; masspoints=:off)),
    ("senate_mass_check", RD_SENATE, :vote, :margin, (; masspoints=:check)),
    ("senate_regul0", RD_SENATE, :vote, :margin, (; scaleregul=0)),
    ("senate_kink", RD_SENATE, :vote, :margin, (; deriv=1)),
    ("senate_weights", RD_SENATE, :vote, :margin, (; weights=:pop_m)),
    ("senate_cutoff5", RD_SENATE, :vote, :margin, (; cutoff=5)),
    ("senate_bwcheck", RD_SENATE, :vote, :margin, (; bwcheck=50)),
    ("senate_bwrestrict_off", RD_SENATE, :vote, :margin, (; bwrestrict=false)),
    ("senate_scalepar", RD_SENATE, :vote, :margin, (; scalepar=2)),
    ("sim_sharp", RD_SIM, :y_sharp, :x, (;)),
    ("sim_sharp_covs", RD_SIM, :y_sharp, :x, (; covariates=[:z1, :z2])),
    ("sim_sharp_covs_collinear", RD_SIM, :y_sharp, :x, (; covariates=[:z1, :z2, :z3])),
    ("sim_sharp_cluster", RD_SIM, :y_sharp, :x, (; cluster=:g)),
    ("sim_sharp_cluster_msetwo", RD_SIM, :y_sharp, :x, (; cluster=:g, bwselect=:msetwo)),
    ("sim_sharp_cluster_cerrd", RD_SIM, :y_sharp, :x, (; cluster=:g, bwselect=:cerrd)),
    ("sim_sharp_cluster_hb", RD_SIM, :y_sharp, :x, (; cluster=:g, h=0.3)),
    ("sim_sharp_weights", RD_SIM, :y_sharp, :x, (; weights=:w)),
    ("sim_sharp_hc3", RD_SIM, :y_sharp, :x, (; vce=:hc3)),
    ("sim_sharp_hc1_weights_covs", RD_SIM, :y_sharp, :x,
     (; vce=:hc1, weights=:w, covariates=[:z1])),
    ("sim_fuzzy", RD_SIM, :y_fuzzy, :x, (; treatment=:d)),
    ("sim_fuzzy_covs", RD_SIM, :y_fuzzy, :x, (; treatment=:d, covariates=[:z1, :z2])),
    ("sim_fuzzy_cluster", RD_SIM, :y_fuzzy, :x, (; treatment=:d, cluster=:g)),
    ("sim_fuzzy_covs_cluster", RD_SIM, :y_fuzzy, :x,
     (; treatment=:d, covariates=[:z1], cluster=:g)),
    ("sim_fuzzy_hc1", RD_SIM, :y_fuzzy, :x, (; treatment=:d, vce=:hc1)),
    ("sim_fuzzy_hc2_covs", RD_SIM, :y_fuzzy, :x, (; treatment=:d, vce=:hc2,
                                                   covariates=[:z1])),
    ("sim_fuzzy_sharpbw", RD_SIM, :y_fuzzy, :x, (; treatment=:d, sharpbw=true)),
    ("sim_fuzzy_msesum", RD_SIM, :y_fuzzy, :x, (; treatment=:d, bwselect=:msesum)),
    ("sim_fuzzy_certwo", RD_SIM, :y_fuzzy, :x, (; treatment=:d, bwselect=:certwo)),
    ("sim_fuzzy_onesided", RD_SIM, :y_one, :x, (; treatment=:d_one)),
    ("sim_kink", RD_SIM, :y_kink, :x, (; deriv=1)),
    ("sim_kink_uni", RD_SIM, :y_kink, :x, (; deriv=1, kernel=:uni)),
    ("sim_fuzzy_kink", RD_SIM, :y_fkink, :x, (; treatment=:d_kink, deriv=1)),
    ("sim_fuzzy_kink_covs", RD_SIM, :y_fkink, :x,
     (; treatment=:d_kink, deriv=1, covariates=[:z2])),
    ("sim_disc", RD_SIM, :y_disc, :x_disc, (;)),
    ("sim_disc_off", RD_SIM, :y_disc, :x_disc, (; masspoints=:off)),
    ("sim_disc_hc2", RD_SIM, :y_disc, :x_disc, (; vce=:hc2)),
    ("sim_disc_msecomb2", RD_SIM, :y_disc, :x_disc, (; bwselect=:msecomb2)),
    ("sim_p2_epa", RD_SIM, :y_sharp, :x, (; p=2, kernel=:epa)),
    ("senate_cr2", RD_SENATE, :vote, :margin, (; cluster=:state, vce=:cr2)),
    ("senate_cr3", RD_SENATE, :vote, :margin, (; cluster=:state, vce=:cr3)),
    ("sim_sharp_cr2", RD_SIM, :y_sharp, :x, (; cluster=:g, vce=:cr2)),
    ("sim_sharp_cr3", RD_SIM, :y_sharp, :x, (; cluster=:g, vce=:cr3)),
    ("sim_sharp_cr2_hb", RD_SIM, :y_sharp, :x, (; cluster=:g, vce=:cr2, h=0.3)),
    ("sim_sharp_cr3_hb", RD_SIM, :y_sharp, :x, (; cluster=:g, vce=:cr3, h=0.3)),
    ("sim_fuzzy_covs_cr3", RD_SIM, :y_fuzzy, :x,
     (; treatment=:d, covariates=[:z1], cluster=:g, vce=:cr3)),
    ("sim_fuzzy_cr2", RD_SIM, :y_fuzzy, :x, (; treatment=:d, cluster=:g, vce=:cr2)),
]

@testset "rd_estimate matches rdrobust ($(length(RD_ESTIMATE_CASES)) cases)" begin
    for (case, data, y, x, kw) in RD_ESTIMATE_CASES
        @testset "$case" begin
            r = Base.CoreLogging.with_logger(Base.CoreLogging.NullLogger()) do
                rd_estimate(data, y, x; kw...)
            end
            @test rdclose([r.h_left, r.h_right], rdref(case, "h"))
            @test rdclose([r.b_left, r.b_right], rdref(case, "b"))
            @test rdclose(r.tau_conventional, rdref(case, "tau_cl"))
            @test rdclose(r.tau_bias_corrected, rdref(case, "tau_bc"))
            @test rdclose(r.se_conventional, rdref(case, "se_cl"))
            @test rdclose(r.se_robust, rdref(case, "se_rb"))
            tab = rd_inference_table(r)
            @test rdclose(tab.pvalue, rdref(case, "pv"); atol=1e-12)
            @test rdclose(tab.ci_lower, rdref(case, "ci_lower"))
            @test rdclose(tab.ci_upper, rdref(case, "ci_upper"))
            @test [r.n_left, r.n_right] == rdref(case, "N")
            @test [r.n_h_left, r.n_h_right] == rdref(case, "N_h")
            @test [r.n_b_left, r.n_b_right] == rdref(case, "N_b")
            @test rdclose([r.bias_left, r.bias_right], rdref(case, "bias"); atol=1e-8)
            @test rdclose(r.beta_left, rdref(case, "beta_Y_p_l"); atol=1e-8)
            @test rdclose(r.beta_right, rdref(case, "beta_Y_p_r"); atol=1e-8)
            if hasref(case, "tau_T")
                fs = r.first_stage
                @test rdclose([fs.tau_conventional, fs.tau_bias_corrected,
                               fs.tau_bias_corrected], rdref(case, "tau_T"))
                @test rdclose([fs.se_conventional, fs.se_conventional, fs.se_robust],
                              rdref(case, "se_T"))
            end
            if hasref(case, "coef_covs")
                @test rdclose(vec(r.gamma), rdref(case, "coef_covs"); atol=1e-8)
            end
            # headline follows rdrobust: conventional estimate, robust inference
            @test coef(r)[1] == r.tau_conventional
            @test tstats(r)[1] ≈ r.tau_bias_corrected / r.se_robust
            @test confint(r)[1, 1] ≈ rdref(case, "ci_lower")[3] rtol = 1e-6
        end
    end
end

const RD_BW_CASES = Any[
    ("bw_senate", RD_SENATE, :vote, :margin, (;)),
    ("bw_senate_cluster", RD_SENATE, :vote, :margin, (; cluster=:state)),
    ("bw_senate_p2_uni", RD_SENATE, :vote, :margin, (; p=2, kernel=:uniform)),
    ("bw_sim_fuzzy_covs", RD_SIM, :y_fuzzy, :x, (; treatment=:d, covariates=[:z1, :z2])),
    ("bw_sim_fuzzy_covs_cluster", RD_SIM, :y_fuzzy, :x,
     (; treatment=:d, covariates=[:z1, :z2], cluster=:g)),
    ("bw_sim_kink_hc3", RD_SIM, :y_kink, :x, (; deriv=1, vce=:hc3)),
    ("bw_sim_disc", RD_SIM, :y_disc, :x_disc, (;)),
    ("bw_sim_cr2", RD_SIM, :y_sharp, :x, (; cluster=:g, vce=:cr2)),
    ("bw_senate_cr3", RD_SENATE, :vote, :margin, (; cluster=:state, vce=:cr3)),
]

@testset "rd_bandwidth(bwselect=:all) matches rdbwselect" begin
    for (case, data, y, x, kw) in RD_BW_CASES
        @testset "$case" begin
            bw = Base.CoreLogging.with_logger(Base.CoreLogging.NullLogger()) do
                rd_bandwidth(data, y, x; bwselect=:all, kw...)
            end
            @test bw.method === :all
            @test nrow(bw.table) == 10
            for row in eachrow(bw.table)
                m = string(row.method)
                @test rdclose([row.h_left, row.h_right], rdref(case, m * "_h"))
                @test rdclose([row.b_left, row.b_right], rdref(case, m * "_b"))
            end
        end
    end
    # single selector agrees with the corresponding row of :all
    one = rd_bandwidth(RD_SENATE, :vote, :margin; bwselect=:certwo)
    @test rdclose([one.h_left, one.h_right], rdref("bw_senate", "certwo_h"))
end
