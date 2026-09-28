@testset "point estimates and interface" begin
    df = sim_block_panel(StableRNG(21); N0=30, N1=4, tau=3.0, sigma=0.3)
    r = synthetic_did(df, :y, :d, :unit, :year; se_method=:placebo, replications=100,
                      rng=StableRNG(1))
    @test r isa SyntheticDiDEstimate
    @test coef(r) == [r.att]
    @test coefnames(r) == ["ATT"]
    @test nobs(r) == 34 * 20
    @test abs(r.att - 3.0) < 4 * r.se
    @test stderror(r)[1] ≈ r.se
    @test confint(r)[1] < r.att < confint(r)[2]
    w = synth_weights(r)
    @test nrow(w) == 30 && sum(w.weight) ≈ 1 && all(>=(0), w.weight)
    tw = synth_time_weights(r)
    @test nrow(tw) == 15 && sum(tw.weight) ≈ 1
    g = synth_gaps(r)
    @test nrow(g) == 20 && mean(g.gap[g.post]) ≈ r.att
    @test nrow(synth_cohorts(r)) == 1
    s = sprint(show, MIME"text/plain"(), r)
    @test occursin("Synthetic difference-in-differences", s)
    @test occursin("placebo", s)
    @test !isempty(sprint(show, r))
    for m in (:sc, :did)
        rm = synthetic_did(df, :y, :d, :unit, :year; method=m, se_method=:none)
        @test rm.method == m
        @test_throws ArgumentError vcov(rm)
        @test occursin("not computed", sprint(show, MIME"text/plain"(), rm))
    end
    rdid = synthetic_did(df, :y, :d, :unit, :year; method=:did, se_method=:none)
    # DiD = difference of mean changes
    p = synth_panel(df, :y, :d, :unit, :year)
    Y = p.Y
    manual = (mean(Y[31:34, 16:20]) - mean(Y[31:34, 1:15])) -
             (mean(Y[1:30, 16:20]) - mean(Y[1:30, 1:15]))
    @test rdid.att ≈ manual
end

@testset "row-order invariance" begin
    df = sim_block_panel(StableRNG(22); N0=25, N1=3)
    shuffled = df[randperm(StableRNG(5), nrow(df)), :]
    for se in (:placebo, :bootstrap, :jackknife)
        a = synthetic_did(df, :y, :d, :unit, :year; se_method=se, replications=30,
                          rng=StableRNG(9))
        b = synthetic_did(shuffled, :y, :d, :unit, :year; se_method=se,
                          replications=30, rng=StableRNG(9))
        @test a.att == b.att
        @test a.se == b.se
        @test synth_weights(a) == synth_weights(b)
    end
end

@testset "staggered adoption equals cohort-wise aggregation" begin
    df = sim_block_panel(StableRNG(23); N0=25, N1=6, T0=12, T1=8)
    late = ["u029", "u030", "u031"]
    df.d[in.(df.unit, Ref(late)) .& (df.year .< 2016)] .= 0
    r = synthetic_did(df, :y, :d, :unit, :year; se_method=:none)
    c = synth_cohorts(r)
    @test nrow(c) == 2
    @test c.adoption == [2013, 2016]
    @test c.weight ≈ [3 * 8, 3 * 5] ./ (3 * 8 + 3 * 5)
    # each cohort equals a block design with never-treated units + that cohort
    for (k, tr) in enumerate([["u026", "u027", "u028"], late])
        sub = df[in.(df.unit, Ref(vcat(["u" * lpad(i, 3, '0') for i in 1:25], tr))), :]
        rb = synthetic_did(sub, :y, :d, :unit, :year; se_method=:none)
        @test rb.att ≈ c.estimate[k] rtol = 1e-10
    end
    @test r.att ≈ sum(c.weight .* c.estimate)
    @test nrow(synth_weights(r)) == 2 * 25
    @test :cohort in propertynames(synth_gaps(r))
    rp = synthetic_did(df, :y, :d, :unit, :year; se_method=:placebo, replications=20,
                       rng=StableRNG(2))
    @test rp.se > 0
    rb = synthetic_did(df, :y, :d, :unit, :year; se_method=:bootstrap, replications=20,
                       rng=StableRNG(2))
    @test rb.se > 0
    rj = synthetic_did(df, :y, :d, :unit, :year; se_method=:jackknife)
    @test rj.se > 0
end

@testset "covariates" begin
    rng = StableRNG(24)
    df = sim_block_panel(rng; N0=25, N1=3, tau=1.0)
    df.x2 = randn(rng, nrow(df))
    df.y .+= 2.0 .* df.x .- 1.0 .* df.x2
    rp = synthetic_did(df, :y, :d, :unit, :year; covariates=[:x, :x2],
                       covariate_method=:projected, se_method=:none)
    # projection β equals a two-way FE regression on never-treated units
    ctrl = df[df.unit .<= "u025", :]
    m = DrSnow.FixedEffectModels.reg(ctrl, make_formula(:y, [:x, :x2];
                                                        fe=[:unit, :year]))
    @test rp.beta ≈ coef(m) rtol = 1e-8
    @test rp.beta ≈ [2.0, -1.0] atol = 0.2
    ro = synthetic_did(df, :y, :d, :unit, :year; covariates=[:x, :x2], se_method=:none)
    @test ro.covariate_method == :optimized
    @test length(ro.cohorts[1].beta) == 2
    @test abs(ro.att - 1.0) < 1.0 && abs(rp.att - 1.0) < 1.0
    rj = synthetic_did(df, :y, :d, :unit, :year; covariates=[:x, :x2],
                       covariate_method=:projected, se_method=:jackknife)
    @test rj.se > 0
    dfm = allowmissing(copy(df))
    dfm.x[5] = missing
    @test_throws ArgumentError synthetic_did(dfm, :y, :d, :unit, :year; covariates=[:x])
end

@testset "errors" begin
    df = sim_block_panel(StableRNG(25); N0=10, N1=1)
    @test_throws ArgumentError synthetic_did(df, :y, :d, :unit, :year; method=:foo)
    @test_throws ArgumentError synthetic_did(df, :y, :d, :unit, :year; se_method=:foo)
    @test_throws ArgumentError synthetic_did(df, :y, :d, :unit, :year;
                                             se_method=:bootstrap)
    @test_throws ArgumentError synthetic_did(df, :y, :d, :unit, :year;
                                             se_method=:jackknife)
    few = sim_block_panel(StableRNG(26); N0=3, N1=3)
    @test_throws ArgumentError synthetic_did(few, :y, :d, :unit, :year)
    short = sim_block_panel(StableRNG(27); T0=1)
    @test_throws ArgumentError synthetic_did(short, :y, :d, :unit, :year)
    @test_throws ArgumentError synthetic_did(df, :y, :d, :unit, :year; replications=1)
end
