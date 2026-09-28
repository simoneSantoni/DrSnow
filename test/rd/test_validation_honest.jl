# Validation of rd_honest / rd_honest_bme / rd_smoothness_bound against RDHonest, of
# rd_mccrary_test against rdd::DCdensity, and of the stdvars option against rdrobust.
# Reference values: test/validation/rd/reference_honest.csv
# (generate_reference_honest.R).

rdh_quiet(f) = Base.CoreLogging.with_logger(f, Base.CoreLogging.NullLogger())

const RD_LEE = rd_read_csv(joinpath(RD_VALIDATION_DIR, "lee08.csv"))
const RD_HS = rd_read_csv(joinpath(RD_VALIDATION_DIR, "headst.csv"))
const RD_RCP = rd_read_csv(joinpath(RD_VALIDATION_DIR, "rcp_sample.csv"))
const RD_CGHS = rd_read_csv(joinpath(RD_VALIDATION_DIR, "cghs_sample.csv"))
const RD_HREF = let d = rd_read_csv(joinpath(RD_VALIDATION_DIR, "reference_honest.csv"))
    Dict{Tuple{String,String},Float64}((r.case, r.key) => coalesce(r.value, NaN)
                                      for r in eachrow(d))
end

function rdhref(case, key)
    haskey(RD_HREF, (case, key)) && return RD_HREF[(case, key)]
    out = Float64[]
    i = 1
    while haskey(RD_HREF, (case, "$key[$i]"))
        push!(out, RD_HREF[(case, "$key[$i]")])
        i += 1
    end
    isempty(out) && error("no reference value for ($case, $key)")
    return out
end

const RD_HONEST_KEYS = (:estimate, :se, :max_bias, :conf_low, :conf_high,
                        :conf_low_onesided, :conf_high_onesided, :bandwidth, :eff_obs,
                        :leverage, :cv, :M, :pvalue)

function rdh_check(case, r; rtol)
    keys_ = collect(RD_HONEST_KEYS)
    r.design === :fuzzy && append!(keys_, [:first_stage, :M_rf, :M_fs])
    for k in keys_
        # absolute slack relative to the standard error for values near zero
        ok = rdclose(getproperty(r, k), rdhref(case, string(k)); rtol=rtol,
                     atol=max(1e-12, rtol * r.se))
        ok || @info "rd_honest mismatch" case k getproperty(r, k) rdhref(case, string(k))
        @test ok
    end
end

const RD_LEE_POS = RD_LEE[coalesce.(RD_LEE.margin .> 0, false), :]
const RD_HONEST_CASES = Any[
    # (case, data, outcome, running, kwargs, check at R's bandwidth?)
    ("lee_uni_h10", RD_LEE, :voteshare, :margin, (; kernel=:uniform, M=0.1, h=10), false),
    ("lee_default", RD_LEE, :voteshare, :margin, (;), true),
    ("lee_flci", RD_LEE, :voteshare, :margin, (; M=0.04, opt_criterion=:flci), true),
    ("lee_oci_epa", RD_LEE, :voteshare, :margin,
     (; M=0.04, kernel=:epanechnikov, opt_criterion=:oci), true),
    ("lee_uni_mse", RD_LEE, :voteshare, :margin, (; M=0.1, kernel=:uniform), true),
    ("lee_ehw_h10", RD_LEE, :voteshare, :margin, (; M=0.1, h=10, vce=:ehw), false),
    ("lee_taylor_flci", RD_LEE, :voteshare, :margin,
     (; M=0.1, sclass=:taylor, opt_criterion=:flci), true),
    ("lee_alpha10", RD_LEE, :voteshare, :margin,
     (; M=0.1, level=0.9, opt_criterion=:flci), true),
    ("lee_J5_h8", RD_LEE, :voteshare, :margin, (; M=0.1, h=8, J=5), false),
    ("lee_cutoff5", RD_LEE, :voteshare, :margin, (; M=0.1, cutoff=5), true),
    ("lee_ip_boundary", RD_LEE_POS, :voteshare, :margin,
     (; M=0.1, h=10, point_inference=true), false),
    ("lee_ip_rot", RD_LEE_POS, :voteshare, :margin, (; point_inference=true), true),
    ("senate_cluster", RD_SENATE, :vote, :margin, (; vce=:ehw, cluster=:state, M=0.1),
     true),
    ("senate_cluster_h", RD_SENATE, :vote, :margin,
     (; vce=:ehw, cluster=:state, M=0.1, h=15), false),
    ("hs_covs_rot", RD_HS, :mortHS, :povrate, (; covariates=[:urban, :black, :sch1417]),
     false),
    ("hs_covs_M2", RD_HS, :mortHS, :povrate,
     (; covariates=[:urban, :black, :sch1417], M=2), false),
    ("hs_covs_M2_h8", RD_HS, :mortHS, :povrate,
     (; covariates=[:urban, :black, :sch1417], M=2, h=8), false),
    ("hs_nocov", RD_HS, :mortHS, :povrate, (; M=2), true),
    ("hs_weights", RD_HS, :mortHS, :povrate, (; M=2, weights=:pop_k), true),
    ("hs_cluster_covs", RD_HS, :mortHS, :povrate,
     (; covariates=[:urban, :black], M=2, vce=:ehw, cluster=:statefp), false),
    ("rcp_fuzzy_h3", RD_RCP, :log_cn, :elig_year,
     (; treatment=:retired, M=(0.001, 0.002), h=3), false),
    ("rcp_fuzzy_mse", RD_RCP, :log_cn, :elig_year, (; treatment=:retired,
                                                    M=(0.001, 0.002)), true),
    ("rcp_fuzzy_rot", RD_RCP, :log_cn, :elig_year, (; treatment=:retired), true),
    ("rcp_fuzzy_ehw_flci", RD_RCP, :log_cn, :elig_year,
     (; treatment=:retired, M=(0.002, 0.004), vce=:ehw, opt_criterion=:flci), true),
    ("rcp_fuzzy_covs", RD_RCP, :log_cn, :elig_year,
     (; treatment=:retired, M=(0.001, 0.002), covariates=[:survey_year]), false),
    ("rcp_fuzzy_cluster", RD_RCP, :log_cn, :elig_year,
     (; treatment=:retired, M=(0.001, 0.002), vce=:ehw, cluster=:survey_year), true),
    ("sim_fuzzy_uni", RD_SIM, :y_fuzzy, :x, (; treatment=:d, M=(1, 1), kernel=:uniform),
     false),
    ("sim_disc", RD_SIM, :y_disc, :x_disc, (; M=1), true),
    ("sim_disc_uni", RD_SIM, :y_disc, :x_disc,
     (; M=1, kernel=:uniform, opt_criterion=:flci), false),
    ("cghs_h3", RD_CGHS, :log_earn, :yearat14, (; cutoff=1947, M=0.04, h=3), false),
    ("cghs_opt", RD_CGHS, :log_earn, :yearat14, (; cutoff=1947, M=0.04), true),
]

@testset "rd_honest vs RDHonest: $(c[1])" for c in RD_HONEST_CASES
    case, data, y, x, kw, at_ref_h = c
    r = rdh_quiet(() -> rd_honest(data, y, x; kw...))
    # Bandwidths chosen by numerical optimisation agree to the optimiser's tolerance
    # (R's `optimize`, ~1e-8 relative); with a fixed or grid-searched bandwidth every
    # quantity agrees to floating-point accuracy.
    rdh_check(case, r; rtol=haskey(kw, :h) || get(kw, :kernel, :x) === :uniform ?
                             1e-9 : 2e-6)
    if at_ref_h
        r2 = rdh_quiet(() -> rd_honest(data, y, x; kw..., h=rdhref(case, "bandwidth")))
        rdh_check(case, r2; rtol=1e-9)
    end
end

@testset "rd_honest fuzzy with T0 = first-round estimate" begin
    r0 = rd_honest(RD_RCP, :log_cn, :elig_year; treatment=:retired, M=(0.001, 0.002))
    r = rd_honest(RD_RCP, :log_cn, :elig_year; treatment=:retired, M=(0.001, 0.002),
                  T0=r0.estimate)
    rdh_check("rcp_fuzzy_T0", r; rtol=2e-6)
end

@testset "rd_honest at an interior point (Hölder bias by exact integration)" begin
    # RDHonest integrates the worst-case bias numerically (stats::integrate, relative
    # tolerance ~1e-4); DrSnow integrates the piecewise-linear integrand exactly.
    r = rd_honest(RD_LEE, :voteshare, :margin; M=0.1, h=10, cutoff=20,
                  point_inference=true)
    rdh_check("lee_ip_interior_h10", r; rtol=1e-5)
    r = rd_honest(RD_LEE, :voteshare, :margin; M=0.1, cutoff=20, point_inference=true)
    @test r.bandwidth ≈ rdhref("lee_ip_interior_opt", "bandwidth") rtol = 1e-3
    r = rd_honest(RD_LEE, :voteshare, :margin; M=0.1, cutoff=20, point_inference=true,
                  h=rdhref("lee_ip_interior_opt", "bandwidth"))
    rdh_check("lee_ip_interior_opt", r; rtol=1e-5)
end

@testset "critical values cv(B) vs RDHonest::CVb" begin
    Bs = [0, 0.1, 0.5, 1, 2, 5, 9.5, 12]
    @test rdclose([DrSnow._rd_cvb(b, 0.05) for b in Bs], rdhref("cvb_a05", "cv");
                  rtol=1e-10)
    @test rdclose([DrSnow._rd_cvb(b, 0.10) for b in Bs], rdhref("cvb_a10", "cv");
                  rtol=1e-10)
end

@testset "rd_honest_bme vs RDHonestBME: $(c[1])" for c in [
        ("cghs_bme_o0_h3", RD_CGHS, :log_earn, :yearat14, (; cutoff=1947, h=3, order=0)),
        ("cghs_bme_o1_h5", RD_CGHS, :log_earn, :yearat14, (; cutoff=1947, h=5, order=1)),
        ("cghs_bme_o2_all", RD_CGHS, :log_earn, :yearat14,
         (; cutoff=1947, order=2, level=0.9)),
        ("sim_disc_bme_o1", RD_SIM, :y_disc, :x_disc, (; h=0.2, order=1))]
    case, data, y, x, kw = c
    r = rd_honest_bme(data, y, x; kw...)
    for (k, v) in (("estimate", r.estimate), ("se", r.se), ("max_bias", r.max_bias),
                   ("conf_low", r.conf_low), ("conf_high", r.conf_high),
                   ("conf_low_onesided", r.conf_low_onesided),
                   ("conf_high_onesided", r.conf_high_onesided), ("eff_obs", r.n),
                   ("leverage", r.leverage))
        @test rdclose(v, rdhref(case, k); rtol=1e-9)
    end
end

@testset "rd_smoothness_bound vs RDSmoothnessBound" begin
    # single curvature estimate per side: deterministic
    sb = rd_smoothness_bound(RD_CGHS, :log_earn, :yearat14; cutoff=1947, s=2,
                             separate=true, multiple=false)
    @test rdclose(sb.estimate, rdhref("cghs_sb_s2_sep", "estimate"); rtol=1e-8)
    @test rdclose(sb.conf_low, rdhref("cghs_sb_s2_sep", "conf_low"); rtol=1e-8)
    sb = rd_smoothness_bound(RD_CGHS, :log_earn, :yearat14; cutoff=1947, s=1,
                             separate=true, multiple=false, sclass=:taylor)
    @test rdclose(sb.estimate, rdhref("cghs_sb_s1_T", "estimate"); rtol=1e-8)
    sb = rd_smoothness_bound(RD_LEE, :voteshare, :margin; s=100, separate=true,
                             multiple=false)
    @test rdclose(sb.estimate, rdhref("lee_sb_s100_sep", "estimate"); rtol=1e-8)
    @test rdclose(sb.conf_low, rdhref("lee_sb_s100_sep", "conf_low"); rtol=1e-8)
    @test sb.conf_low[1] > 0
    # multiple curvature estimates: simulated critical values (R uses its own draws)
    sb = rd_smoothness_bound(RD_LEE, :voteshare, :margin; s=100, rng=StableRNG(42))
    @test sb.estimate[1] ≈ rdhref("lee_sb_s100_multi", "estimate") rtol = 0.05
    @test sb.conf_low[1] <= sb.estimate[1]
end

@testset "rd_mccrary_test vs rdd::DCdensity: $(c[1])" for c in [
        ("mcc_senate", RD_SENATE.margin, (;)), ("mcc_lee", RD_LEE.margin, (;)),
        ("mcc_lee_bin_bw", RD_LEE.margin, (; bin=1, bandwidth=15)),
        ("mcc_sim_c02", RD_SIM.x, (; cutoff=0.2)), ("mcc_hs", RD_HS.povrate, (;))]
    case, x, kw = c
    t = rd_mccrary_test(x; kw...)
    @test rdclose(t.details.theta, rdhref(case, "theta"); rtol=1e-9)
    @test rdclose(t.details.se, rdhref(case, "se"); rtol=1e-9)
    @test rdclose(t.statistic, rdhref(case, "z"); rtol=1e-9)
    @test rdclose(t.pvalue, rdhref(case, "p"); rtol=1e-9)
    @test rdclose(t.details.bin, rdhref(case, "bin"); rtol=1e-12)
    @test rdclose(t.details.bandwidth, rdhref(case, "bw"); rtol=1e-9)
end

@testset "stdvars vs rdrobust(stdvars = TRUE): $(c[1])" for c in [
        ("std_senate", RD_SENATE, :vote, :margin, (;)),
        ("std_senate_msetwo", RD_SENATE, :vote, :margin, (; bwselect=:msetwo)),
        ("std_sim_fuzzy_covs", RD_SIM, :y_fuzzy, :x, (; treatment=:d, covariates=[:z1])),
        ("std_sim_disc", RD_SIM, :y_disc, :x_disc, (;)),
        ("std_sim_cluster_cer", RD_SIM, :y_sharp, :x, (; cluster=:g, bwselect=:cerrd))]
    case, data, y, x, kw = c
    r = rd_estimate(data, y, x; stdvars=true, kw...)
    @test rdclose(r.tau_conventional, rdhref(case, "tau_cl"))
    @test rdclose(r.tau_bias_corrected, rdhref(case, "tau_bc"))
    @test rdclose(r.se_conventional, rdhref(case, "se_cl"))
    @test rdclose(r.se_robust, rdhref(case, "se_rb"))
    @test rdclose([r.h_left, r.h_right], rdhref(case, "h"))
    @test rdclose([r.b_left, r.b_right], rdhref(case, "b"))
end

@testset "stdvars vs rdbwselect(stdvars = TRUE, all = TRUE)" begin
    bw = rd_bandwidth(RD_SENATE, :vote, :margin; bwselect=:all, stdvars=true)
    @test rdclose(vec(Matrix(bw.table[:, 2:5])'), rdhref("stdbw_senate_all", "bws"))
    bw = rd_bandwidth(RD_SIM, :y_kink, :x; deriv=1, bwselect=:all, stdvars=true)
    @test rdclose(vec(Matrix(bw.table[:, 2:5])'), rdhref("stdbw_sim_kink_all", "bws"))
end
