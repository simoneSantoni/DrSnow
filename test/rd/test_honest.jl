# Unit tests and Monte Carlo checks for honest RD inference (rd_honest,
# rd_honest_ar_confidence_set, rd_honest_bme, rd_smoothness_bound), the McCrary test
# and the stdvars option.

# Worst case of the Hölder class for the local linear RD estimator at the boundary:
# f(x) = s M x²/2 above and -s M x²/2 below the cutoff (Armstrong & Kolesár 2020).
function rdh_worst_case(rng, n; M=2.0, tau=0.5, s=1, discrete=false, sigma=0.5)
    x = 2 .* rand(rng, n) .- 1
    discrete && (x = round.(x .* 10) ./ 10)
    f = [xi >= 0 ? s * M / 2 * xi^2 : -s * M / 2 * xi^2 for xi in x]
    y = tau .* (x .>= 0) .+ f .+ sigma .* randn(rng, n)
    return DataFrame(y=y, x=x)
end

@testset "rd_honest: input validation" begin
    df = RD_LEE
    @test_throws ArgumentError rd_honest(df, :voteshare, :margin; M=-1)
    @test_throws ArgumentError rd_honest(df, :voteshare, :margin; M=(1, 2))
    @test_throws ArgumentError rd_honest(df, :voteshare, :margin; kernel=:gaussian)
    @test_throws ArgumentError rd_honest(df, :voteshare, :margin; sclass=:lipschitz)
    @test_throws ArgumentError rd_honest(df, :voteshare, :margin; opt_criterion=:cer)
    @test_throws ArgumentError rd_honest(df, :voteshare, :margin; M=0.1, h=-2)
    @test_throws ArgumentError rd_honest(df, :voteshare, :margin; M=0.1, level=95)
    @test_throws ArgumentError rd_honest(df, :voteshare, :margin; M=0.1, vce=:hc3)
    @test_throws ArgumentError rd_honest(df, :voteshare, :margin; M=0.1, cutoff=500)
    # bandwidth too small: the local linear fit is not identified
    @test_throws ArgumentError rd_honest(df, :voteshare, :margin; M=0.1, h=0.001)
    @test_throws ArgumentError rd_honest(RD_SENATE, :vote, :margin; M=0.1,
                                         cluster=:state, vce=:nn)
    @test_throws ArgumentError rd_honest(RD_RCP, :log_cn, :elig_year;
                                         treatment=:retired, M=0.1)
    @test_throws ArgumentError rd_honest(df, :voteshare, :margin; M=0.1,
                                         point_inference=true, treatment=:margin)
    @test_throws ArgumentError rd_honest(RD_HS, :mortHS, :povrate; M=1,
                                         point_inference=true, covariates=[:urban])
    @test_throws ArgumentError rd_honest(RD_HS, :mortHS, :povrate; M=1,
                                         weights=:black)   # zero weights
    @test_throws ArgumentError rd_honest(df, :nope, :margin)
    r = rd_honest(df, :voteshare, :margin; M=0.1, h=10)
    @test_throws ArgumentError rd_honest_ar_confidence_set(r)
    @test_throws ArgumentError confint(r; level=1.2)
    @test_throws ArgumentError rd_honest_bme(RD_CGHS, :log_earn, :yearat14;
                                             cutoff=1947, h=0.5)
    @test_throws ArgumentError rd_honest_bme(RD_CGHS, :log_earn, :yearat14;
                                             cutoff=1947, h=1, order=2)
    @test_throws ArgumentError rd_smoothness_bound(RD_CGHS, :log_earn, :yearat14;
                                                   cutoff=1947, s=100)
    @test_throws ArgumentError rd_mccrary_test(RD_LEE, :margin; cutoff=200)
    @test_throws ArgumentError rd_mccrary_test(RD_LEE, :margin; bandwidth=-1)
    @test_throws ArgumentError rd_mccrary_test(RD_LEE, :nope)
    @test_throws ArgumentError rd_mccrary_test(RD_LEE, :margin; bin=0)
end

@testset "rd_honest: result interface and honest interval" begin
    r = rd_honest(RD_LEE, :voteshare, :margin; M=0.1, h=10)
    @test r isa CausalEstimate
    @test coef(r) == [r.estimate]
    @test stderror(r) ≈ [r.se]
    @test nobs(r) == count(!ismissing, RD_LEE.voteshare)
    ci = confint(r)
    @test ci ≈ [r.conf_low r.conf_high]
    # the honest interval is wider than the naive one when the maximum bias is positive
    @test r.max_bias > 0
    @test ci[1, 2] - ci[1, 1] > 2 * 1.959963984540054 * r.se
    @test r.cv ≈ DrSnow._rd_cvb(r.max_bias / r.se, 0.05)
    ci90 = confint(r; level=0.9)
    @test ci90[1, 1] > ci[1, 1] && ci90[1, 2] < ci[1, 2]
    @test pvalues(r) == [r.pvalue]
    @test 0 <= r.pvalue <= 1
    # with M = 0 the interval is the usual normal interval
    r0 = rd_honest(RD_LEE, :voteshare, :margin; M=0, h=10)
    @test r0.max_bias == 0
    @test confint(r0) ≈ [r0.estimate - 1.959963984540054 * r0.se
                         r0.estimate + 1.959963984540054 * r0.se]'
    @test pvalues(r0)[1] ≈ 2 * ccdf(Normal(), abs(r0.estimate / r0.se)) rtol = 1e-10
    t = tidy(r)
    @test t.conf_low[1] ≈ r.conf_low && t.conf_high[1] ≈ r.conf_high
    @test t.p_value[1] ≈ r.pvalue
    @test occursin("Honest 95% CI", sprint(show, MIME"text/plain"(), r))
    @test occursin("maximum bias", sprint(show, MIME"text/plain"(), r))
    @test estimand(r) == "ATE at the cutoff (sharp RD)"
    rf = rd_honest(RD_RCP, :log_cn, :elig_year; treatment=:retired, M=(0.001, 0.002),
                   h=5)
    @test rf.design === :fuzzy
    @test rf.estimate ≈ rf.reduced_form / rf.first_stage
    @test rf.M ≈ (rf.M_rf + abs(rf.estimate) * rf.M_fs) / abs(rf.first_stage)
    @test occursin("First stage", sprint(show, MIME"text/plain"(), rf))
    # CVb: B = 0 gives the normal critical value; increasing in B; ≈ B + z for large B
    @test DrSnow._rd_cvb(0, 0.05) ≈ 1.959963984540054 rtol = 1e-12
    cvs = [DrSnow._rd_cvb(b, 0.05) for b in 0:0.25:12]
    @test issorted(cvs)
    @test DrSnow._rd_cvb(9.99, 0.05) ≈ 9.99 + quantile(Normal(), 0.95) rtol = 1e-8
end

@testset "rd_honest: row order does not matter" begin
    rng = StableRNG(5)
    for kw in ((; M=0.1), (; M=0.1, kernel=:uniform, h=8), (; M=0.1, vce=:ehw))
        a = rd_honest(RD_LEE, :voteshare, :margin; kw...)
        b = rd_honest(RD_LEE[randperm(rng, nrow(RD_LEE)), :], :voteshare, :margin; kw...)
        @test a.estimate ≈ b.estimate rtol = 1e-10
        @test a.se ≈ b.se rtol = 1e-10
        @test a.bandwidth ≈ b.bandwidth rtol = 1e-8
        @test a.conf_high ≈ b.conf_high rtol = 1e-8
    end
    a = rd_honest(RD_HS, :mortHS, :povrate; M=2, covariates=[:urban], cluster=:statefp)
    b = rd_honest(RD_HS[randperm(rng, nrow(RD_HS)), :], :mortHS, :povrate; M=2,
                  covariates=[:urban], cluster=:statefp)
    @test a.estimate ≈ b.estimate rtol = 1e-9
    @test a.se ≈ b.se rtol = 1e-9
    t1 = rd_mccrary_test(RD_LEE, :margin)
    t2 = rd_mccrary_test(RD_LEE[randperm(rng, nrow(RD_LEE)), :], :margin)
    @test t1.statistic ≈ t2.statistic rtol = 1e-10
    b1 = rd_honest_bme(RD_CGHS, :log_earn, :yearat14; cutoff=1947, h=3, order=1)
    b2 = rd_honest_bme(RD_CGHS[randperm(rng, nrow(RD_CGHS)), :], :log_earn, :yearat14;
                       cutoff=1947, h=3, order=1)
    @test b1.conf_low ≈ b2.conf_low rtol = 1e-9
end

@testset "rd_honest: worst-case bias at an interior point" begin
    # exact integral of the Hölder worst-case bias vs a fine Riemann sum
    rng = StableRNG(3)
    xx = sort(2 .* rand(rng, 60) .- 1)
    wt = randn(rng, 60) ./ 60
    c = DrSnow._rd_h_bias_constant(wt, xx, 1.0, :holder, false)
    grid = range(0, 1; length=200_001)
    gp(s) = abs(sum(wt[i] * (xx[i] - s) for i in eachindex(xx) if xx[i] >= s; init=0.0))
    gm(s) = abs(sum(wt[i] * (s - xx[i]) for i in eachindex(xx) if xx[i] <= s; init=0.0))
    riemann = sum(gp, grid) * step(grid) + sum(s -> gm(-s), grid) * step(grid)
    @test c ≈ riemann rtol = 1e-4
end

@testset "rd_honest: covariates, weights, discrete support" begin
    r = rd_honest(RD_HS, :mortHS, :povrate; M=2, h=8,
                  covariates=[:urban, :black, :sch1417])
    @test r.covariates == [:urban, :black, :sch1417]
    # a covariate collinear with others is dropped with a warning
    hs = copy(RD_HS)
    hs.dup = 2 .* hs.urban
    r2 = @test_logs (:warn,) match_mode = :any rd_honest(hs, :mortHS, :povrate; M=2,
                                                         h=8,
                                                         covariates=[:urban, :dup])
    @test r2.covariates == [:urban]
    # integer weights reproduce the estimate on the expanded data (up to NN variance)
    rng = StableRNG(9)
    df = rdh_worst_case(rng, 300)
    df.w = Float64.(rand(rng, 1:3, 300))
    expanded = df[vcat([fill(i, Int(df.w[i])) for i in 1:nrow(df)]...), :]
    a = rd_honest(df, :y, :x; M=2, h=0.5, weights=:w, vce=:ehw)
    b = rd_honest(expanded, :y, :x; M=2, h=0.5, vce=:ehw)
    @test a.estimate ≈ b.estimate rtol = 1e-10
    @test a.max_bias ≈ b.max_bias rtol = 1e-10
    # discrete running variable: bandwidth search over the support
    d = rdh_worst_case(StableRNG(2), 600; discrete=true)
    rd = rd_honest(d, :y, :x; M=2, kernel=:uniform)
    @test rd.bandwidth in unique(abs.(d.x))
end

@testset "rd_honest_ar_confidence_set" begin
    rf = rd_honest(RD_SIM, :y_fuzzy, :x; treatment=:d, M=(0.5, 0.2), h=0.5)
    s = rd_honest_ar_confidence_set(rf)
    @test s.kind === :interval
    lo, hi = s.intervals[1]
    @test lo < rf.estimate < hi
    # a weak first stage (retirement subsample, first stage t ≈ 1.3): the AR set is the
    # whole line although the delta-method interval is bounded
    rw = rd_honest(RD_RCP, :log_cn, :elig_year; treatment=:retired, M=(0, 0), h=8)
    @test rd_honest_ar_confidence_set(rw).kind === :real_line
    @test isfinite(rw.conf_low) && isfinite(rw.conf_high)
    # with M = 0 the set is the Fieller/Anderson–Rubin set, close to the delta-method
    # interval when the first stage is strong
    r0 = rd_honest(RD_SIM, :y_fuzzy, :x; treatment=:d, M=(0, 0), h=0.5)
    s0 = rd_honest_ar_confidence_set(r0)
    @test s0.kind === :interval
    @test s0.intervals[1][1] ≈ r0.conf_low rtol = 0.2
    @test s0.intervals[1][2] ≈ r0.conf_high rtol = 0.2
    # boundary points solve the acceptance equation
    V = r0.V
    for t in s0.intervals[1]
        sd = sqrt(V[1] - 2t * V[2] + t^2 * V[4])
        @test abs(r0.reduced_form - t * r0.first_stage) ≈ 1.959963984540054 * sd rtol=1e-6
    end
    @test rd_honest_ar_confidence_set(r0; level=0.99).intervals[1][2] > s0.intervals[1][2]
    # an uninformative first stage gives the whole line
    rng = StableRNG(4)
    n = 400
    x = 2 .* rand(rng, n) .- 1
    d = Float64.(rand(rng, n) .< 0.4)
    y = d .+ 0.5 .* randn(rng, n)
    rw = rd_honest(DataFrame(y=y, x=x, d=d), :y, :x; treatment=:d, M=(0.5, 0.5), h=0.5)
    @test rd_honest_ar_confidence_set(rw).kind in (:real_line, :union)
end

@testset "rd_honest_bme and rd_smoothness_bound: interface" begin
    r = rd_honest_bme(RD_CGHS, :log_earn, :yearat14; cutoff=1947, h=3, order=1)
    @test r isa CausalEstimate
    @test confint(r) ≈ [r.conf_low r.conf_high]
    c90 = confint(r; level=0.9)
    @test c90[1] >= r.conf_low && c90[2] <= r.conf_high
    # p-value is consistent with the interval (duality)
    for lv in (0.8, 0.9, 0.95, 0.99)
        ci = confint(r; level=lv)
        excludes = ci[1] > 0 || ci[2] < 0
        @test excludes == (r.pvalue < 1 - lv)
    end
    @test occursin("BME 95% CI", sprint(show, MIME"text/plain"(), r))
    sb = rd_smoothness_bound(RD_LEE, :voteshare, :margin; s=100, separate=true,
                             multiple=false)
    @test sb.side == ["below", "above"]
    @test all(sb.conf_low .<= sb.estimate)
    a = rd_smoothness_bound(RD_LEE, :voteshare, :margin; s=50, rng=StableRNG(1))
    b = rd_smoothness_bound(RD_LEE, :voteshare, :margin; s=50, rng=StableRNG(1))
    @test a == b
end

@testset "rd_mccrary_test: interface" begin
    t = rd_mccrary_test(RD_SENATE, :margin)
    @test t isa DiagnosticTest
    @test occursin("rd_density_test", t.note)
    @test t.details.histogram isa DataFrame
    @test t.statistic ≈ t.details.theta / t.details.se
    # user bandwidth / bin are honoured
    t2 = rd_mccrary_test(RD_SENATE.margin; bin=2, bandwidth=20)
    @test t2.details.bin == 2 && t2.details.bandwidth == 20
end

@testset "stdvars: bandwidths are equivariant to the units of the data" begin
    df = copy(RD_SENATE)
    df.margin100 = 100 .* df.margin
    df.vote10 = 10 .* df.vote
    a = rd_estimate(df, :vote, :margin; stdvars=true)
    b = rd_estimate(df, :vote10, :margin100; stdvars=true)
    @test b.h_left ≈ 100 * a.h_left rtol = 1e-8
    @test b.tau_conventional ≈ 10 * a.tau_conventional rtol = 1e-8
    # stdvars is ignored when h is supplied (as in rdrobust)
    @test rd_estimate(df, :vote, :margin; h=10, stdvars=true).tau_conventional ≈
          rd_estimate(df, :vote, :margin; h=10).tau_conventional
    bw = rd_bandwidth(df, :vote, :margin; stdvars=true)
    @test bw.h_left ≈ a.h_left rtol = 1e-8
end

@testset "Monte Carlo: honest CI coverage at the worst case of the Hölder class" begin
    # Coverage must be at least nominal uniformly over the class; at the least
    # favourable function it is close to nominal. 3 binomial s.e. + 0.01 slack.
    reps = mc_reps(2000, 300)
    tol = 3 * sqrt(0.95 * 0.05 / reps) + 0.01
    for (label, dgp, est) in (("MSE-optimal h, s = +1", (; s=1), (;)),
                              ("MSE-optimal h, s = -1", (; s=-1), (;)),
                              ("FLCI-optimal h", (; s=1), (; opt_criterion=:flci)),
                              ("fixed h = 0.3", (; s=-1), (; h=0.3)),
                              ("discrete running variable", (; s=1, discrete=true),
                               (; opt_criterion=:flci)))
        rng = StableRNG(2020)
        cover = 0
        naive = 0
        for _ in 1:reps
            d = rdh_worst_case(rng, 500; dgp...)
            r = rdh_quiet(() -> rd_honest(d, :y, :x; M=2.0, est...))
            cover += r.conf_low <= 0.5 <= r.conf_high
            naive += abs(r.estimate - 0.5) <= 1.959963984540054 * r.se
        end
        @info "Honest CI coverage ($label, $reps reps)" honest = cover / reps naive =
            naive / reps
        @test cover / reps >= 0.95 - tol
        # the naive interval ignores the bias and covers less at the worst case
        @test naive <= cover
    end
end

@testset "Monte Carlo: bias-aware AR set coverage with a weak first stage" begin
    reps = mc_reps(1000, 100)
    tol = 3 * sqrt(0.95 * 0.05 / reps) + 0.01
    for pi_ in (0.6, 0.1)
        rng = StableRNG(11)
        cover = 0
        for _ in 1:reps
            n = 1000
            x = 2 .* rand(rng, n) .- 1
            d = Float64.(rand(rng, n) .< 0.3 .+ pi_ .* (x .>= 0))
            g = [xi >= 0 ? xi^2 / 2 : -xi^2 / 2 for xi in x]
            y = d .+ g .+ 0.5 .* randn(rng, n)
            r = rdh_quiet(() -> rd_honest(DataFrame(y=y, x=x, d=d), :y, :x; treatment=:d,
                                          M=(1.0, 0.1), h=0.5))
            s = rd_honest_ar_confidence_set(r)
            cover += any(lo <= 1.0 <= hi for (lo, hi) in s.intervals)
        end
        @info "AR set coverage (first stage $pi_, $reps reps)" coverage = cover / reps
        @test cover / reps >= 0.95 - tol
    end
end

@testset "Monte Carlo: McCrary test size" begin
    reps = mc_reps(2000, 300)
    rng = StableRNG(8)
    rej = 0
    for _ in 1:reps
        x = randn(rng, 1000)
        rej += rd_mccrary_test(x).pvalue < 0.05
    end
    size_ = rej / reps
    @info "McCrary test size ($reps reps)" size = size_
    @test abs(size_ - 0.05) < 4 * sqrt(0.05 * 0.95 / reps) + 0.02
end
