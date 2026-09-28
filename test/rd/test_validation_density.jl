# Validation of rd_density_test / rd_density_bandwidth against rddensity 3.0 (R).

const RD_DENS_CASES = Any[
    ("dens_senate", RD_SENATE.margin, (;)),
    ("dens_senate_all", RD_SENATE.margin, (;)),
    ("dens_senate_p1", RD_SENATE.margin, (; p=1)),
    ("dens_senate_uni", RD_SENATE.margin, (; kernel=:uniform)),
    ("dens_senate_epa", RD_SENATE.margin, (; kernel=:epanechnikov)),
    ("dens_senate_plugin", RD_SENATE.margin, (; vce=:plugin)),
    ("dens_senate_restricted", RD_SENATE.margin, (; fitselect=:restricted)),
    ("dens_senate_restricted_plugin", RD_SENATE.margin,
     (; fitselect=:restricted, vce=:plugin)),
    ("dens_senate_each", RD_SENATE.margin, (; bwselect=:each)),
    ("dens_senate_diff", RD_SENATE.margin, (; bwselect=:diff)),
    ("dens_senate_sum", RD_SENATE.margin, (; bwselect=:sum)),
    ("dens_senate_h", RD_SENATE.margin, (; h=(20, 30))),
    ("dens_senate_nomass", RD_SENATE.margin, (; masspoints=false)),
    ("dens_senate_c10", RD_SENATE.margin, (; cutoff=10)),
    ("dens_sim", RD_SIM.x, (;)),
    ("dens_sim_disc", RD_SIM.x_disc, (;)),
    ("dens_sim_binoW", RD_SIM.x, (; binomial_width=0.05, binomial_windows=5)),
]

# NaN in the reference (R's NA) must be NaN in DrSnow too.
rdclose_nan(a, b; kw...) = all(map((x, y) -> (isnan(x) && isnan(y)) ||
                                              isapprox(x, y; rtol=1e-7, atol=1e-10, kw...),
                                   a, b))

@testset "rd_density_test matches rddensity ($(length(RD_DENS_CASES)) cases)" begin
    for (case, x, kw) in RD_DENS_CASES
        @testset "$case" begin
            t = rd_density_test(x; kw...)
            d = t.details
            @test rdclose([d.h_left, d.h_right], rdref(case, "h"))
            @test rdclose_nan([d.f_left, d.f_right, d.f_diff], rdref(case, "hat"))
            N = rdref(case, "N")
            @test [d.n_left + d.n_right, d.n_left, d.n_right, d.n_eff_left,
                   d.n_eff_right] == N
            @test rdclose_nan([d.t_plugin, d.t_jackknife, d.p_plugin, d.p_jackknife],
                              rdref(case, "test"))
            if d.vce === :jackknife
                @test rdclose_nan([d.se_left, d.se_right, d.se_diff], rdref(case, "sd_jk"))
                @test t.statistic ≈ d.t_jackknife
            else
                @test rdclose_nan([d.se_left, d.se_right, d.se_diff],
                                  rdref(case, "sd_asy"))
                @test t.statistic ≈ d.t_plugin
            end
            b = d.binomial
            @test b.n_left == rdref(case, "bino_LN")
            @test b.n_right == rdref(case, "bino_RN")
            @test rdclose(b.window_left, rdref(case, "bino_LW"))
            @test rdclose(b.window_right, rdref(case, "bino_RW"))
            @test rdclose(b.pvalue, rdref(case, "bino_pval"); atol=1e-12)
            if hasref(case, "test_p")
                tp = rdref(case, "test_p")
                cv = d.conventional
                @test rdclose(cv.t, d.vce === :jackknife ? tp[2] : tp[1])
                @test rdclose([cv.f_left, cv.f_right, cv.f_diff], rdref(case, "hat_p"))
            end
        end
    end
end

const RD_BWD_CASES = Any[
    ("bwd_senate", RD_SENATE.margin, (;)),
    ("bwd_senate_restricted", RD_SENATE.margin, (; fitselect=:restricted)),
    ("bwd_senate_plugin_uni", RD_SENATE.margin, (; vce=:plugin, kernel=:uniform)),
    ("bwd_sim", RD_SIM.x, (; p=3)),
]

@testset "rd_density_bandwidth matches rdbwdensity" begin
    for (case, x, kw) in RD_BWD_CASES
        @testset "$case" begin
            tab = rd_density_bandwidth(x; kw...)
            @test rdclose(tab.bandwidth, rdref(case, "bw"))
            @test rdclose_nan(tab.variance, rdref(case, "variance"))
            @test rdclose_nan(tab.bias_squared, rdref(case, "biassq"))
        end
    end
    df = DataFrame(m=RD_SENATE.margin)
    @test rd_density_bandwidth(df, :m).bandwidth ≈ rdref("bwd_senate", "bw")
end
