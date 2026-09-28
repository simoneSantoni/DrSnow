function sim_convex_panel(rng; J=12, T0=15, T1=6, tau=-3.0, noise=0.1)
    T = T0 + T1
    F = hcat(ones(T), cumsum(randn(rng, T)), sin.((1:T) ./ 2))
    load = randn(rng, J, 3)
    Yd = load * F' .+ noise .* randn(rng, J, T)
    xd = randn(rng, J)                    # a time-invariant predictor
    w = zeros(J)
    w[[2, 5, 9]] = [0.5, 0.3, 0.2]
    y1 = vec(w' * Yd) .+ vcat(zeros(T0), fill(tau, T1))
    x1 = dot(w, xd)
    rows = NamedTuple[]
    for j in 1:J, t in 1:T
        push!(rows, (unit="d" * lpad(j, 2, '0'), t=t, y=Yd[j, t], d=0, x=xd[j]))
    end
    for t in 1:T
        push!(rows, (unit="treated", t=t, y=y1[t], d=t > T0 ? 1 : 0, x=x1))
    end
    return DataFrame(rows), w
end

@testset "exact recovery inside the convex hull" begin
    df, w = sim_convex_panel(StableRNG(31))
    r = synthetic_control(df, :y, :d, :unit, :t; placebo=false)
    @test r.weights ≈ w atol = 1e-6
    @test r.att ≈ -3.0 atol = 1e-6
    @test r.pre_rmspe < 1e-6
    @test_throws ArgumentError vcov(r)
    @test_throws ArgumentError synth_in_space_placebo(r)
    # predictors + special predictors with V optimisation
    r2 = synthetic_control(df, :y, :d, :unit, :t; predictors=[:x],
                           special_predictors=[:y => 1:5, :y => 6:10, :y => [15]],
                           placebo=false, rng=StableRNG(1))
    @test sum(r2.v) ≈ 1 && all(>=(0), r2.v)
    @test length(r2.predictor_names) == 4
    @test r2.att ≈ -3.0 atol = 1e-3
    @test nrow(r2.predictor_balance) == 4
    # user-supplied V
    r3 = synthetic_control(df, :y, :d, :unit, :t; predictors=[:x],
                           special_predictors=[:y => 1:5, :y => 6:10],
                           v=[1.0, 1.0, 1.0], placebo=false)
    @test r3.v ≈ fill(1 / 3, 3)
    @test_throws ArgumentError synthetic_control(df, :y, :d, :unit, :t;
                                                 predictors=[:x], v=[1.0, 2.0])
    @test_throws ArgumentError synthetic_control(df, :y, :d, :unit, :t;
                                                 special_predictors=[:y => 16:18])
end

@testset "placebo inference and robustness" begin
    df, _ = sim_convex_panel(StableRNG(32); noise=0.3, tau=-4.0)
    r = synthetic_control(df, :y, :d, :unit, :t; predictors=[:x],
                          special_predictors=[:y => 1:5, :y => 6:10, :y => 11:15],
                          rng=StableRNG(3))
    @test r.placebo !== nothing
    t = synth_in_space_placebo(r)
    @test t isa DiagnosticTest
    @test 1 / 13 <= t.pvalue <= 1
    @test t.details.n_placebos == 12
    ta = synth_in_space_placebo(r; statistic=:att)
    @test 1 / 13 <= ta.pvalue <= 1
    cut = sort(r.placebo.pre_rmspe)[6] / r.pre_rmspe
    tc = synth_in_space_placebo(r; pre_rmspe_cutoff=cut)
    @test tc.details.n_placebos == 6
    @test_throws ArgumentError synth_in_space_placebo(r; pre_rmspe_cutoff=1e-12)
    @test_throws ArgumentError synth_in_space_placebo(r; statistic=:foo)
    @test vcov(r)[1] ≈ var(r.placebo.att; corrected=false)
    @test stderror(r)[1] > 0
    s = sprint(show, MIME"text/plain"(), r)
    @test occursin("In-space placebo p-value", s)
    loo = synth_leave_one_out(r)
    @test nrow(loo.summary) == count(>(1e-6), r.weights)
    @test nrow(loo.gaps) == nrow(loo.summary) * 21
    # windows ending after the placebo date make the backdated design undefined
    @test_throws ArgumentError synth_in_time_placebo(r, 10)
    r_early = synthetic_control(df, :y, :d, :unit, :t; predictors=[:x],
                                special_predictors=[:y => 1:5, :y => 6:12],
                                placebo=false)
    bt = synth_in_time_placebo(r_early, 10)
    @test synth_in_time_placebo(r_early, 10).att == bt.att
    @test bt.n_pre == 9
    @test length(bt.treated_path) == 15
    @test abs(bt.att) < abs(r.att)
    @test_throws ArgumentError synth_in_time_placebo(r_early, 1)
    @test_throws ArgumentError synth_in_time_placebo(r_early, 18)
    g = synth_gaps(r)
    @test mean(g.gap[g.post]) ≈ r.att
    @test nrow(synth_weights(r)) == 12
end

@testset "row order and errors" begin
    df, _ = sim_convex_panel(StableRNG(33); noise=0.2)
    sh = df[randperm(StableRNG(1), nrow(df)), :]
    kw = (predictors=[:x], special_predictors=[:y => 1:5, :y => 6:15], placebo=false)
    a = synthetic_control(df, :y, :d, :unit, :t; kw..., rng=StableRNG(4))
    b = synthetic_control(sh, :y, :d, :unit, :t; kw..., rng=StableRNG(4))
    @test a.weights == b.weights && a.att == b.att
    two = copy(df)
    two.d[(two.unit .== "d01") .& (two.t .> 15)] .= 1
    @test_throws ArgumentError synthetic_control(two, :y, :d, :unit, :t)
    flat = copy(df)
    flat.z = ones(nrow(flat))
    @test_throws ArgumentError synthetic_control(flat, :y, :d, :unit, :t;
                                                 predictors=[:z])
end
