@testset "ridge ASCM basics" begin
    df = sim_block_panel(StableRNG(41); N0=25, N1=1, T0=15, T1=5, tau=-2.0, sigma=0.2)
    r = augmented_synthetic_control(df, :y, :d, :unit, :year; replications=50,
                                    rng=StableRNG(1))
    @test r isa AugmentedSCEstimate
    @test r.se_method == :placebo && length(r.replicate_estimates) == 50
    @test sum(r.weights) ≈ 1 atol = 1e-10
    @test sum(r.scm_weights) ≈ 1 atol = 1e-10
    @test all(>=(-1e-12), r.scm_weights)
    @test r.lambda > 0 && r.cv isa DataFrame
    @test abs(r.att + 2.0) < 1.0
    @test r.se > 0 && r.se_path === nothing
    rj = augmented_synthetic_control(df, :y, :d, :unit, :year; se_method=:jackknife)
    @test rj.se > 0 && length(rj.se_path) == 5
    @test rj.att == r.att
    @test coef(r) == [r.att] && nobs(r) == 26 * 20
    w = synth_weights(r)
    @test names(w) == ["unit", "weight", "scm_weight"]
    g = synth_gaps(r)
    @test mean(g.gap[g.post]) ≈ r.att
    @test occursin("ridge", sprint(show, MIME"text/plain"(), r))
    # λ → ∞ recovers plain SCM; ridge = false is plain SCM
    rs = augmented_synthetic_control(df, :y, :d, :unit, :year; ridge=false)
    rinf = augmented_synthetic_control(df, :y, :d, :unit, :year; lambda=1e12)
    @test rinf.att ≈ rs.att atol = 1e-6
    @test rs.lambda === nothing
    rn = augmented_synthetic_control(df, :y, :d, :unit, :year; se_method=:none)
    @test_throws ArgumentError vcov(rn)
    # row-order invariance
    sh = df[randperm(StableRNG(3), nrow(df)), :]
    rsh = augmented_synthetic_control(sh, :y, :d, :unit, :year; replications=50,
                                      rng=StableRNG(1))
    @test rsh.att == r.att && rsh.se == r.se
end

@testset "several treated units and errors" begin
    df = sim_block_panel(StableRNG(42); N0=20, N1=3)
    r = augmented_synthetic_control(df, :y, :d, :unit, :year; fixed_effects=true,
                                    se_method=:jackknife)
    @test r.se > 0
    stag = copy(df)
    stag.d[(stag.unit .== "u021") .& (stag.year .== 2016)] .= 0
    @test_throws ArgumentError augmented_synthetic_control(stag, :y, :d, :unit, :year)
    @test_throws ArgumentError augmented_synthetic_control(df, :y, :d, :unit, :year;
                                                           se_method=:bootstrap)
    @test_throws ArgumentError augmented_synthetic_control(df, :y, :d, :unit, :year;
                                                           replications=1)
    @test_throws ArgumentError augmented_synthetic_control(df, :y, :d, :unit, :year;
                                                           lambda=-1.0)
end

@testset "conformal inference" begin
    df = sim_block_panel(StableRNG(43); N0=20, N1=1, T0=18, T1=3, tau=0.0)
    r = augmented_synthetic_control(df, :y, :d, :unit, :year; se_method=:none)
    ci = synth_conformal_inference(r; grid_size=20)
    @test nrow(ci.per_period) == 3
    @test all(0 .< ci.per_period.p_value .<= 1)
    # block p-values are multiples of 1/(T0 + 1)
    @test all(isinteger.(round.(ci.per_period.p_value .* 19; digits=8)))
    @test ci.joint isa DiagnosticTest
    ok = .!ismissing.(ci.per_period.lower)
    @test all(ci.per_period.lower[ok] .<= ci.per_period.upper[ok])
    cii = synth_conformal_inference(r; type=:iid, grid_size=10, ns=200,
                                    rng=StableRNG(1))
    @test all(0 .< cii.per_period.p_value .<= 1)
    @test_throws ArgumentError synth_conformal_inference(r; type=:foo)
    @test_throws ArgumentError synth_conformal_inference(r; level=1.5)
end
