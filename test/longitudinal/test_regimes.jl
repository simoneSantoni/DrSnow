# Regimes and data handling.

@testset "Treatment regimes" begin
    s = StaticRegime([1, 0, 1])
    @test s.actions == [1.0, 0.0, 1.0] && s.label == "static(1,0,1)"
    @test StaticRegime([true, false]; label="x").label == "x"
    @test always_treat(3) == StaticRegime(ones(3); label="always treat")
    @test never_treat(2).actions == zeros(2)
    @test_throws ArgumentError StaticRegime([1, 2])
    @test_throws ArgumentError StaticRegime(Int[])
    d = DynamicRegime(h -> h.t == 1 ? 1 : 0)
    @test d.label == "dynamic regime"
    @test occursin("static(1,0,1)", sprint(show, s))
    @test occursin("dynamic regime", sprint(show, d))
    # evaluation on data: history passed to the rule
    df, _ = lg_sim(StableRNG(1), 50; K=3)
    seen = Vector{Any}()
    rule = function (h)
        push!(seen, (h.t, length(h.covariates), length(h.treatment), keys(h.baseline)))
        h.covariates[end].l > 0 ? 1 : 0
    end
    dd = DrSnow._lg_data(df, :y, :a, :id, :t; covariates=[:l], baseline=[:w],
                         cluster=nothing, context="t")
    D = DrSnow._lg_actions(DynamicRegime(rule), dd, "t")
    @test size(D) == (50, 3)
    @test D == Float64.(hcat([dd.L[t][:, 1] .> 0 for t in 1:3]...))
    @test seen[1] == (1, 1, 0, (:w,)) && seen[3] == (3, 3, 2, (:w,))
    @test DrSnow._lg_actions(StaticRegime([1, 0, 0]), dd, "t") ==
          repeat([1.0 0.0 0.0], 50)
    @test_throws ArgumentError DrSnow._lg_actions(StaticRegime([1, 0]), dd, "t")
    bad = DynamicRegime(h -> 2)
    @test_throws ArgumentError DrSnow._lg_actions(bad, dd, "t")
end

@testset "Data ingestion and errors" begin
    df, _ = lg_sim(StableRNG(2), 40; K=3)
    kw = (covariates=[:l], baseline=[:w], cluster=nothing, context="t")
    d = DrSnow._lg_data(df, :y, :a, :id, :t; kw...)
    @test d.n == 40 && d.K == 3 && d.ids == collect(1:40) && d.times == [1, 2, 3]
    @test d.A[5, 2] == df.a[(5 - 1) * 3 + 2]
    @test d.L[3][7, 1] == df.l[(7 - 1) * 3 + 3]
    @test d.W[9, 1] == df.w[(9 - 1) * 3 + 1]
    @test d.y[11] == df.y[11 * 3]
    # the outcome is read from the last period; earlier rows may be missing
    @test all(ismissing, df.y[df.t .< 3])
    # shuffled rows give the same arrays
    perm = randperm(StableRNG(3), nrow(df))
    d2 = DrSnow._lg_data(df[perm, :], :y, :a, :id, :t; kw...)
    @test d2.A == d.A && d2.L[2] == d.L[2] && d2.y == d.y && d2.W == d.W
    # design matrices: history columns
    @test size(DrSnow._lg_design(d, 2, nothing; current=false)) == (40, 1 + 2 + 1)
    @test size(DrSnow._lg_design(d, 2, nothing; current=true)) == (40, 1 + 2 + 2)
    Dr = ones(40, 3)
    X = DrSnow._lg_design(d, 3, Dr; current=true)
    @test X[:, end - 2:end] == ones(40, 3)
    # errors
    @test_throws ArgumentError DrSnow._lg_data(df, :y, :a, :id, :t;
                                               covariates=[:nope], baseline=Symbol[],
                                               cluster=nothing, context="t")
    @test_throws ArgumentError DrSnow._lg_data(df, :y, :a, :id, :t; covariates=[:l],
                                               baseline=[:l], cluster=nothing,
                                               context="t")
    unb = df[1:end - 1, :]
    @test_throws ArgumentError DrSnow._lg_data(unb, :y, :a, :id, :t; kw...)
    dup = vcat(df, df[1:1, :])
    @test_throws ArgumentError DrSnow._lg_data(dup, :y, :a, :id, :t; kw...)
    bad = copy(df)
    bad.a[1] = 2.0
    @test_throws ArgumentError DrSnow._lg_data(bad, :y, :a, :id, :t; kw...)
    bad = copy(df)
    bad.w[2] += 1.0                         # baseline varies within a unit
    @test_throws ArgumentError DrSnow._lg_data(bad, :y, :a, :id, :t; kw...)
    bad = copy(df)
    bad.y[3] = missing                      # outcome missing in the last period
    @test_throws ArgumentError DrSnow._lg_data(bad, :y, :a, :id, :t; kw...)
    bad = copy(df)
    bad.a[bad.t .== 2] .= 1.0               # no variation in a period
    @test_throws ArgumentError DrSnow._lg_data(bad, :y, :a, :id, :t; kw...)
    # cluster constant within units
    df.g = repeat(1:8, 15)[1:nrow(df)]
    @test_throws ArgumentError DrSnow._lg_data(df, :y, :a, :id, :t; covariates=[:l],
                                               baseline=[:w], cluster=:g, context="t")
    df.g = repeat(1:8; inner=15)
    dc = DrSnow._lg_data(df, :y, :a, :id, :t; covariates=[:l], baseline=[:w],
                         cluster=:g, context="t")
    @test dc.n_clusters == 8 && length(dc.cluster) == 40
    # front-end argument errors
    r(; kws...) = longitudinal_regime(df, :y, :a, :id, :t; regime=always_treat(3),
                                      covariates=[:l], crossfit=false,
                                      outcome_learner=OLSLearner(),
                                      treatment_learner=LogisticLearner(), kws...)
    @test_throws ArgumentError r(estimator=:foo)
    @test_throws ArgumentError r(g_bound=0.0)
    @test_throws ArgumentError r(q_bound=0.5)
    @test_throws ArgumentError r(bootstrap=1)
    @test_throws ArgumentError r(regime=TreatmentRegime[])
    @test_throws ArgumentError r(regime=[always_treat(3), always_treat(3)])
    @test_throws ArgumentError r(n_rep=2)
    @test_throws ArgumentError r(regime=always_treat(2))
    @test_throws ArgumentError r(outcome_bounds=(5, 6))
    @test_throws ArgumentError r(outcome_learner=OLSLearner(),
                                 treatment_learner=OLSLearner())   # no fitpredict_proba
end
