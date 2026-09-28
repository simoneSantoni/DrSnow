using Dates

@testset "Treatment timing layer" begin
    @testset "cohorts from an indicator, gaps and Dates" begin
        df = DataFrame(u=repeat(1:3; inner=4), t=repeat([2000, 2002, 2004, 2010], 3),
                       d=[0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1])
        tm = treatment_timing(df, :d, :u, :t)
        @test tm.periods == [2000, 2002, 2004, 2010]
        @test tm.unit_cohort == [3, 0, 2]          # period indices, 0 = never treated
        @test tm.absorbing && tm.balanced && tm.panel
        @test DrSnow._did_event_time(tm)[1:4] == [-2, -1, 0, 1]   # counted in periods
        dd = copy(df)
        dd.t = Date.(dd.t)
        tmd = treatment_timing(dd, :d, :u, :t)
        @test tmd.unit_cohort == tm.unit_cohort
        s = sprint(show, MIME"text/plain"(), tm)
        @test occursin("Never treated units: 1", s)
    end

    @testset "FirstTreated coding" begin
        df = DataFrame(u=repeat(1:5; inner=3), t=repeat(1:3, 5),
                       g=repeat([0.0, 2.0, Inf, 9.0, 1.5]; inner=3))
        tm = treatment_timing(df, FirstTreated(:g), :u, :t)
        # 0 and Inf → never; beyond the sample → never; 1.5 → next observed period 2
        @test tm.unit_cohort == [0, 2, 0, 0, 2]
        @test tm.row_treated == BitVector([0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1])
        dfm = DataFrame(u=[1, 1, 2, 2], t=[1, 2, 1, 2], g=[missing, missing, 2, 2])
        # `missing` codes never treated when passed through FirstTreated directly
        @test treatment_timing(dfm, FirstTreated(:g), :u, :t).unit_cohort == [0, 2]
        dfv = DataFrame(u=[1, 1], t=[1, 2], g=[2, 3])
        @test_throws ArgumentError treatment_timing(dfv, FirstTreated(:g), :u, :t)
        # custom never-treated code
        dfn = DataFrame(u=[1, 1, 2, 2], t=[1, 2, 1, 2], g=[-1, -1, 2, 2])
        @test treatment_timing(dfn, FirstTreated(:g; never=-1), :u, :t).unit_cohort ==
              [0, 2]
    end

    @testset "checks" begin
        df = DataFrame(u=[1, 1, 1, 2, 2, 2], t=[1, 2, 3, 1, 2, 3], d=[0, 1, 0, 0, 0, 0])
        @test !treatment_timing(df, :d, :u, :t).absorbing
        dup = DataFrame(u=[1, 1], t=[1, 1], d=[0, 1])
        @test_throws ArgumentError treatment_timing(dup, :d, :u, :t)
        bad = DataFrame(u=[1, 1], t=[1, 2], d=[0, 2])
        @test_throws ArgumentError treatment_timing(bad, :d, :u, :t)
        unb = DataFrame(u=[1, 1, 2], t=[1, 2, 1], d=[0, 1, 0])
        @test !treatment_timing(unb, :d, :u, :t).balanced
        # repeated cross-sections need cohorts, not a 0/1 indicator
        @test_throws ArgumentError treatment_timing(df, :d, nothing, :t)
        rc = DataFrame(t=[1, 2, 1, 2], g=[2, 2, 0, 0])
        tmr = treatment_timing(rc, FirstTreated(:g), nothing, :t)
        @test !tmr.panel && tmr.row_cohort == [2, 2, 0, 0]
        @test_throws ArgumentError treatment_timing(df, :d, :u, :t; anticipation=-1)
    end

    @testset "shuffle invariance and control groups" begin
        rng = StableRNG(11)
        df = sim_staggered(rng; N=30, T=5)
        tm1 = treatment_timing(df, :d, :unit, :time)
        tm2 = treatment_timing(shuffle_rows(rng, df), :d, :unit, :time)
        @test tm1.unit_cohort == tm2.unit_cohort
        G = [0, 3, 5, 4]
        @test DrSnow._did_control_units(G, 3, 3, 2, :never_treated, 0) == [1, 0, 0, 0]
        @test DrSnow._did_control_units(G, 3, 3, 2, :not_yet_treated, 0) == [1, 0, 1, 1]
        @test DrSnow._did_control_units(G, 3, 3, 2, :not_yet_treated, 1) == [1, 0, 1, 0]
    end

    @testset "TreatmentPanel and validate_panel" begin
        df = DataFrame(u=[1, 1, 2, 2], t=[1, 2, 1, 2], y=randn(StableRNG(1), 4),
                       d=[0, 1, 0, 0])
        p = TreatmentPanel(df; outcome=:y, treatment=:d, unit_id=:u, time=:t)
        @test p.unit_id == :u
        @test isempty(validate_panel(p))
        df2 = DataFrame(u=[1, 1, 2, 2], t=[1, 2, 1, 2], y=randn(StableRNG(1), 4),
                        d=[1, 0, 0, 0])
        @test any(occursin("Non-absorbing", m) for m in validate_panel(
            TreatmentPanel(df2, :y, :d, :u, :t)))
        df3 = DataFrame(u=[1, 1, 2, 2], t=[1, 1, 1, 2], y=zeros(4), d=zeros(Int, 4))
        msgs = validate_panel(TreatmentPanel(df3, :y, :d, :u, :t))
        @test any(occursin("Duplicated", m) for m in msgs)
        @test any(occursin("does not vary", m) for m in msgs)
    end
end
