@testset "plot data (backend independent)" begin
    f = VIZ_FIX

    @testset "event studies" begin
        es = f.es_twfe
        d = DrSnow._viz_event_study_data(es; level=0.9)
        @test issorted(d.rel_period)
        est = d[.!d.reference, :]
        @test est.rel_period == relative_periods(es)
        @test est.estimate == coef(es)
        ci = confint(es; level=0.9)
        @test est.conf_low ≈ ci[:, 1] && est.conf_high ≈ ci[:, 2]
        @test all(ismissing, d.uniform_low)
        ref = d[d.reference, :]
        @test ref.rel_period == es.reference && all(==(0.0), ref.estimate)
        @test all(ismissing, ref.conf_low)
        # binned endpoints are labelled
        binned = get(es.details, :binned, (false, false))
        binned[1] && @test startswith(est.label[1], "≤")
        # uniform bands contain the pointwise intervals
        u = DrSnow._viz_event_study_data(f.es_cs; uniform=true, rng=StableRNG(1))
        uu = u[.!u.reference, :]
        @test all(uu.uniform_low .<= uu.conf_low .+ 1e-12)
        @test all(uu.uniform_high .>= uu.conf_high .- 1e-12)
        # imputation event studies have no reference period
        @test !any(DrSnow._viz_event_study_data(f.es_imp).reference)
        # spillover event study: one series per group, reference -1 at zero
        s = DrSnow._viz_event_study_data(f.ring_es)
        @test Set(s.group) == Set(f.ring_es.table.group)
        for g in unique(s.group)
            sg = s[s.group .== g, :]
            @test issorted(sg.rel_period)
            @test sg.estimate[sg.reference] == [0.0] && sg.rel_period[sg.reference] == [-1]
        end
        @test "≤-3" in s.label && "≥2" in s.label
        @test_throws ArgumentError DrSnow._viz_event_study_data(f.ring_es; uniform=true)
        @test_throws ArgumentError DrSnow._viz_event_study_data(f.did)
    end

    @testset "coefficients" begin
        cd = DrSnow._viz_coef_data([f.did, f.sdid, f.rd]; labels=["a", "b", "c"])
        @test cd.single && cd.models == ["a", "b", "c"]
        cd2 = DrSnow._viz_coef_data([f.iv, f.ring_did])
        @test !cd2.single
        @test nrow(cd2.table) == length(coef(f.iv)) + length(coef(f.ring_did))
        cd3 = DrSnow._viz_coef_data(f.iv; terms=["d"])
        @test cd3.table.term == ["d"] && cd3.single
        @test nrow(DrSnow._viz_coef_data(f.ring_did; terms=r"spill").table) == 4
        @test nrow(DrSnow._viz_coef_data(f.iv; terms=t -> t != "(Intercept)").table) == 2
        @test_throws ArgumentError DrSnow._viz_coef_data(f.iv; terms=["nope"])
        @test_throws ArgumentError DrSnow._viz_coef_data(f.iv; terms=r"nope")
        @test_throws ArgumentError DrSnow._viz_coef_data(DrSnow.CausalEstimate[])
        @test_throws ArgumentError DrSnow._viz_coef_data([f.did, f.sdid]; labels=["a", "a"])
        @test_throws ArgumentError DrSnow._viz_coef_data(1.0)
        @test_logs (:warn, r"without a standard error") DrSnow._viz_coef_data(f.sdid_nose)
    end

    @testset "RD local fits" begin
        r = f.rd
        lf = DrSnow._viz_rd_local_fit(r)
        left = lf[lf.side .== :left, :]
        right = lf[lf.side .== :right, :]
        @test left.x[1] ≈ r.cutoff - r.h_left && right.x[end] ≈ r.cutoff + r.h_right
        # the jump of the local fits at the cutoff is the conventional estimate
        @test right.y[1] - left.y[end] ≈ r.tau_conventional rtol = 1e-8
    end

    @testset "synthetic control" begin
        sd = DrSnow._viz_synth_data(f.sc)
        @test sd.gaps.gap ≈ sd.gaps.treated .- sd.gaps.synthetic
        @test ismissing(sd.cohort) && sd.time_weights === nothing
        @test sd.onset == [sd.gaps.time[findfirst(sd.gaps.post)]]
        nplac = length(f.sc.placebo.units)
        @test nrow(sd.placebo_gaps) == nplac * nrow(sd.gaps)
        # the placebo cutoff drops poorly fitted donors
        cut = DrSnow._viz_synth_data(f.sc; placebo_cutoff=1.0)
        kept = count(f.sc.placebo.pre_rmspe .<= f.sc.pre_rmspe)
        @test nrow(cut.placebo_gaps) == kept * nrow(sd.gaps)
        s2 = DrSnow._viz_synth_data(f.sdid)
        @test s2.time_weights.weight ≈ synth_time_weights(f.sdid).weight
        @test s2.placebo_gaps === nothing
        @test DrSnow._viz_synth_data(f.mc).time_weights === nothing
        @test_throws ArgumentError DrSnow._viz_synth_data(f.sc; cohort=1999)
        @test_throws ArgumentError DrSnow._viz_synth_data(f.did)
    end

    @testset "randomization distributions" begin
        rd = DrSnow._viz_randomization_data(f.ri)
        v, w = randomization_distribution(f.ri)
        @test rd.values == v && rd.weights == w && rd.observed == f.ri.observed
        @test rd.pvalue == pvalue(f.ri) && rd.alternative === :two_sided
        pt = synth_in_space_placebo(f.sc)
        pd = DrSnow._viz_randomization_data(pt)
        @test pd.values == pt.details.placebo_statistics && pd.alternative === :greater
        @test_throws ArgumentError DrSnow._viz_randomization_data(pre_trend_test(f.es_cs))
        @test_throws ArgumentError DrSnow._viz_randomization_data(f.did)
    end

    @testset "GATES, exposure effects and confidence sets" begin
        gd = DrSnow._viz_gates_data(f.gml)
        @test nrow(gd.gates) == f.gml.n_groups
        @test !any(occursin.(" - ", gd.gates.group))
        @test gd.ate.estimate == blp(f.gml).estimate[1]
        @test_throws ArgumentError DrSnow._viz_gates_data(f.did)
        ed = DrSnow._viz_exposure_effect_data(f.ring_did)
        @test ed.kind == ["direct"; fill("spillover", 4)]
        @test_throws ArgumentError DrSnow._viz_exposure_effect_data(f.did)
        cs = DrSnow._viz_confidence_set_data(f.ar; npoints=101)
        @test length(cs.grid) == 101 && all(0 .<= cs.pvalue .<= 1)
        lo, hi = f.ar.intervals[1]
        @test cs.limits[1] < lo && cs.limits[2] > hi
        # the p-value crosses 1 - level at the set boundaries
        @test pvalue(f.ar, lo) ≈ 1 - f.ar.level atol = 1e-4
        inside = [any(iv -> iv[1] <= b <= iv[2], f.ar.intervals) for b in cs.grid]
        @test all(cs.pvalue[inside] .>= 1 - f.ar.level - 1e-6)
        @test all(cs.pvalue[.!inside] .<= 1 - f.ar.level + 1e-6)
        @test DrSnow._viz_confidence_set_data(f.ar; limits=(0, 1)).limits == (0.0, 1.0)
        @test_throws ArgumentError DrSnow._viz_confidence_set_data(f.ar; limits=(1, 0))
        @test_throws ArgumentError DrSnow._viz_confidence_set_data(f.did)
    end
end
