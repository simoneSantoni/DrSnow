# Backend-independent plot data for the design-diagnostic plots.

@testset "diagnostic plot data" begin
    f = VIZ_DIAG
    D = DrSnow

    @testset "trends" begin
        d = D._viz_trends_data(f.panel, :y, :d, :unit, :time)
        t = d.table
        @test unique(t.group) == ["Never treated", "Cohort 2005", "Cohort 2007"]
        # means match a direct computation
        g = f.panel.g
        for (lab, gv) in (("Never treated", 0), ("Cohort 2005", 5))
            s = t[t.group .== lab, :]
            ref = [mean(f.panel.y[(g .== gv) .& (f.panel.time .== tt)]) for tt in s.time]
            @test s.mean ≈ ref
            n = s.n[1]
            sd = std(f.panel.y[(g .== gv) .& (f.panel.time .== s.time[1])])
            @test s.conf_high[1] - s.mean[1] ≈
                  DrSnow.critical_value(0.95, n - 1) * sd / sqrt(n)
        end
        @test d.onsets.time == [2005, 2007]
        @test d.ticks === nothing && !d.adjusted
        # TreatmentTiming input gives the same table; results invariant to row order
        @test isequal(D._viz_trends_data(f.panel, :y, f.tm).table, t)
        perm = shuffle(StableRNG(3), 1:nrow(f.panel))
        ts = D._viz_trends_data(f.panel[perm, :], :y, :d, :unit, :time).table
        @test ts.mean ≈ t.mean && ts.group == t.group
        # ever-treated grouping pools the cohorts
        tt = D._viz_trends_data(f.panel, :y, :d, :unit, :time; by=:treated).table
        @test unique(tt.group) == ["Never treated", "Ever treated"]
        s = tt[(tt.group .== "Ever treated") .& (tt.time .== 2001), :]
        @test only(s.mean) ≈ mean(f.panel.y[(g .> 0) .& (f.panel.time .== 2001)])
        # cohort subset
        tc = D._viz_trends_data(f.panel, :y, :d, :unit, :time; cohorts=[2007]).table
        @test unique(tc.group) == ["Never treated", "Cohort 2007"]
        @test_throws ArgumentError D._viz_trends_data(f.panel, :y, :d, :unit, :time;
                                                      cohorts=[1999])
        # covariate adjustment: within-cell slope recovers the DGP coefficient (0.8)
        da = D._viz_trends_data(f.panel, :y, :d, :unit, :time; covariates=[:x])
        @test da.adjusted && abs(only(da.coefficients) - 0.8) < 0.1
        @test_throws ArgumentError D._viz_trends_data(f.panel, :y, :d, :unit, :time;
                                                      by=:nope)
        @test_throws ArgumentError D._viz_trends_data(f.panel[1:10, :], :y, f.tm)
        # non-numeric time: ticks carry the labels
        p2 = copy(f.panel)
        p2.tlab = "t" .* lpad.(string.(p2.time .- 2000), 2, '0')
        d2 = D._viz_trends_data(p2, :y, :d, :unit, :tlab)
        @test d2.ticks[2][1] == "t01" && d2.table.x[1] == 1.0
    end

    @testset "balance" begin
        b = D._viz_balance_data(f.pb)
        @test b.kind === :pretreatment && b.threshold == 0.1
        @test b.table.estimate == f.pb.std_diff
        @test sort(unique(b.table.group)) == ["Cohort 2005", "Cohort 2007"]
        @test all(ismissing, b.table.conf_low)
        r = D._viz_balance_data(f.rib)
        @test r.kind === :ri && r.test.pvalue == f.rib.pvalue
        pc = f.rib.details.per_covariate
        @test r.table.pvalue_adj == pc.pvalue_westfall_young
        @test isequal(D._viz_balance_data(pc).table, r.table)
        rd = D._viz_balance_data(f.rdcb)
        @test rd.kind === :rd && rd.table.conf_low == f.rdcb.ci_lower
        @test rd.table.pvalue_adj == f.rdcb.pvalue_holm
        rs = D._viz_balance_data(f.rdcb; data=f.rdd)
        @test rs.table.estimate ≈ f.rdcb.estimate ./ [std(f.rdd[!, c]) for c in
                                                       f.rdcb.covariate]
        @test_throws ArgumentError D._viz_balance_data(f.pb; data=f.rdd)
        @test_throws ArgumentError D._viz_balance_data(DataFrame(a=[1]))
        @test_throws ArgumentError D._viz_balance_data(f.rdden)
        @test_throws ArgumentError D._viz_balance_data(1.0)
    end

    @testset "RD falsification" begin
        p = D._viz_rd_placebo_data(f.rdpl; estimate=f.rd)
        @test nrow(p.table) == nrow(f.rdpl) + 1 && issorted(p.table.cutoff)
        act = p.table[p.table.true_cutoff, :]
        @test only(act.estimate) == f.rd.tau_bias_corrected
        @test only(act.conf_low) ≈ confint(f.rd)[1, 1]
        s = D._viz_rd_sensitivity_data(f.rdbw)
        @test s.kind === :bandwidth && count(s.table.baseline) == 1
        @test s.table.x ≈ f.rdbw.h_left && s.table.label[1] == "×0.5"
        dn = D._viz_rd_sensitivity_data(f.rddo)
        @test dn.kind === :donut && dn.table.baseline == (f.rddo.radius .== 0)
        @test_throws ArgumentError D._viz_rd_sensitivity_data(f.rdpl)
        @test_throws ArgumentError D._viz_rd_placebo_data(f.rdbw)

        dd = D._viz_rd_density_data(f.xr, f.rdden)
        det = f.rdden.details
        dl = dd.density[dd.density.side .== :left, :]
        dr = dd.density[dd.density.side .== :right, :]
        # at the cutoff the side estimates reproduce the test's order-p densities
        @test dl.x[end] == 0.0 && dr.x[1] == 0.0
        @test dl.f[end] ≈ det.conventional.f_left rtol = 1e-8
        @test dr.f[1] ≈ det.conventional.f_right rtol = 1e-8
        @test all(dd.density.conf_low .<= dd.density.f .<= dd.density.conf_high)
        # histogram: cutoff is an edge; densities integrate to the in-range share
        e = dd.hist.edges
        @test 0.0 in e
        inr = count(v -> dd.limits[1] <= v <= dd.limits[2], f.xr) / length(f.xr)
        @test sum(dd.hist.density .* diff(e)) ≈ inr
        # data-frame form and a well-behaved design
        d2 = D._viz_rd_density_data(f.rdd, :x, f.rdden_ok)
        l2 = d2.density[d2.density.side .== :left, :]
        @test l2.f[end] ≈ f.rdden_ok.details.conventional.f_left rtol = 1e-8
        @test_throws ArgumentError D._viz_rd_density_data(f.rdd.x, f.rdden)
        @test_throws ArgumentError D._viz_rd_density_data(f.xr, f.rib)
        @test_throws ArgumentError D._viz_rd_density_data(f.xr, f.rdden;
                                                          limits=(0.1, 0.5))
    end

    @testset "Honest DiD" begin
        h = D._viz_honest_data(f.hd_rm; breakdown=f.bd_rm)
        @test h.table.lb == f.hd_rm.lb && h.breakdown == f.bd_rm
        @test h.breakdown_source === :given && h.mname == "M̄"
        @test D._viz_honest_data(f.hd_sd).mname == "M"
        @test isequal(D._viz_honest_data(f.hd_rm).breakdown, f.hd_rm.breakdown)
        @test_throws ArgumentError D._viz_honest_data(f.rd)
    end

    @testset "IV designs" begin
        j = D._viz_judge_data(f.jd, f.jiv)
        @test nrow(j.bins) == 20 && sum(j.bins.n) == length(j.leniency)
        z = Float64.(f.jiv.leniency)
        @test j.fit.slope ≈ cov(z, f.jd.d) / var(z)
        # the leniency first-stage coefficient of judge_iv (no controls) is this slope
        @test j.fit.slope ≈ f.jiv.first_stage.coef[1] rtol = 1e-6
        j2 = D._viz_judge_data(f.jd, :d, f.jiv.leniency; nbins=10, trim=0.0)
        @test nrow(j2.bins) == 10 && length(j2.leniency) == nrow(f.jd)
        @test_throws ArgumentError D._viz_judge_data(f.jd, :d, z[1:10])
        @test_throws ArgumentError D._viz_judge_data(f.jd, :d, z; nbins=1)

        r = D._viz_rotemberg_data(f.rw; label=3)
        @test count(r.table.labelled) == 3
        @test r.table.sector[r.table.labelled] == string.(f.rw.table.sector[1:3])
        @test sum(r.table.alpha .* r.table.beta) ≈ f.rw.estimate

        m = D._viz_mte_data(f.mte_poly)
        @test m.curve.mte == f.mte_poly.curve.mte
        @test m.parameters.term == coefnames(f.mte_poly)
        @test nrow(D._viz_mte_data(f.mte_poly; parameters=false).parameters) == 0
        @test m.support.lower == f.mte_poly.support.lower
        @test_throws ArgumentError D._viz_mte_data(f.rd)
    end

    @testset "synthetic control backdating" begin
        s = D._viz_synth_in_time_data(f.sc, f.sc_back)
        # the backdated counterfactual reproduces the backdated fit on its sample
        T = length(f.sc_back.synthetic_path)
        @test s.table.backdated[1:T] ≈ f.sc_back.synthetic_path
        @test s.table.synthetic == f.sc.synthetic_path
        @test s.placebo_time == 1998 && s.treatment_time == 2003
        @test_throws ArgumentError D._viz_synth_in_time_data(f.sc, f.rd)
    end

    @testset "stubs without a backend" begin
        if Base.get_extension(DrSnow, :DrSnowMakieExt) === nothing
            for fn in (plot_trends, plot_balance, plot_rd_placebos, plot_rd_sensitivity,
                       plot_rd_density, plot_honest_did, plot_judge_first_stage,
                       plot_rotemberg, plot_mte, plot_synth_in_time, plot_trends!,
                       plot_balance!, plot_mte!, drsnow_theme)
                err = try
                    fn(f.pb)
                    nothing
                catch e
                    e
                end
                @test err isa ErrorException
                @test occursin("using CairoMakie", sprint(showerror, err))
            end
        end
    end
end
