# Validation of rd_plot_data against rdplot (rdrobust 4.0.0, R).

const RD_SEN_COMPLETE = dropmissing(RD_SENATE, [:vote, :margin])

const RD_PLOT_CASES = Any[
    [("plot_senate_$b", RD_SEN_COMPLETE, :vote, :margin, (; binselect=b))
     for b in (:es, :espr, :esmv, :esmvpr, :qs, :qspr, :qsmv, :qsmvpr)]...,
    ("plot_senate_nbins", RD_SEN_COMPLETE, :vote, :margin, (; nbins=(15, 25))),
    ("plot_senate_p1_scale", RD_SEN_COMPLETE, :vote, :margin, (; p=1, scale=2)),
    ("plot_senate_tri_h", RD_SEN_COMPLETE, :vote, :margin, (; kernel=:tri, h=40, p=2)),
    ("plot_senate_covs", RD_SENATE, :vote, :margin, (; covariates=[:presdemvoteshlag1])),
    ("plot_sim_disc", RD_SIM, :y_disc, :x_disc, (;)),
    ("plot_sim_support", RD_SIM, :y_sharp, :x, (; support=(-1.2, 1.2), binselect=:es)),
    ("plot_sim_weights", RD_SIM, :y_sharp, :x, (; weights=:w)),
]

@testset "rd_plot_data matches rdplot ($(length(RD_PLOT_CASES)) cases)" begin
    for (case, data, y, x, kw) in RD_PLOT_CASES
        @testset "$case" begin
            pd = Base.CoreLogging.with_logger(Base.CoreLogging.NullLogger()) do
                rd_plot_data(data, y, x; kw...)
            end
            @test collect(pd.nbins) == rdref(case, "J")
            @test collect(pd.nbins_imse) == rdref(case, "J_IMSE")
            @test collect(pd.nbins_mv) == rdref(case, "J_MV")
            @test rdclose(pd.coef_left, rdref(case, "coef_l"); rtol=1e-6)
            @test rdclose(pd.coef_right, rdref(case, "coef_r"); rtol=1e-6)
            b = pd.bins
            @test b.n == rdref(case, "N_bin")
            @test rdclose(b.bin_mid, rdref(case, "mean_bin"))
            @test rdclose(b.mean_x, rdref(case, "mean_x"))
            @test rdclose(b.mean_y, rdref(case, "mean_y"))
            @test rdclose(b.se_y, rdref(case, "se_y"))
            @test rdclose(b.ci_lower, rdref(case, "ci_l"))
            @test rdclose(b.ci_upper, rdref(case, "ci_r"))
            # rdplot mislabels left-bin edges when some left bins are empty; DrSnow
            # reports the edges of each bin, which agree whenever no bin is empty.
            if all(diff(b.bin[b.side .== :left]) .== 1) && b.bin[1] == -pd.nbins[1]
                @test rdclose(b.bin_left, rdref(case, "min_bin"))
                @test rdclose(b.bin_right, rdref(case, "max_bin"))
            end
            ipoly = [1, 250, 500, 501, 750, 1000]
            @test rdclose(pd.poly.x[ipoly], rdref(case, "poly_x"))
            @test rdclose(pd.poly.y[ipoly], rdref(case, "poly_y"); rtol=1e-6)
            @test rdclose(collect(pd.bin_avg), rdref(case, "bin_avg"))
            @test rdclose(collect(pd.bin_med), rdref(case, "bin_med"))
            @test rdclose(collect(pd.rscale), rdref(case, "rscale"))
        end
    end
end
