function seq_ate_data(rng, n; tau=0.5, p=0.5)
    x1 = randn(rng, n); x2 = randn(rng, n)
    d = rand(rng, n) .< p
    y = x1 .+ 0.5 .* x2 .+ d .* (tau .+ 0.5 .* x1) .+ randn(rng, n)
    return DataFrame(y=y, d=d, x1=x1, x2=x2, arrival=1:n)
end

@testset "anytime-valid ATE (sequential AIPW)" begin
    rng = StableRNG(21)
    df = seq_ate_data(rng, 800)

    @testset "pseudo-outcomes by hand (arm means, known propensity)" begin
        cs = confseq_ate(df, :y, :d; propensity=0.5, running_intersection=false)
        # φ_i with running arm means from units before i
        n = nrow(df)
        phi = zeros(n)
        for i in 1:n
            past = 1:(i - 1)
            y1 = df.y[past][df.d[past]]; y0 = df.y[past][.!df.d[past]]
            ov = i == 1 ? 0.0 : mean(df.y[past])
            m1 = isempty(y1) ? ov : mean(y1); m0 = isempty(y0) ? ov : mean(y0)
            phi[i] = m1 - m0 + (df.d[i] ? (df.y[i] - m1) / 0.5 :
                                -(df.y[i] - m0) / 0.5)
        end
        ref = confseq_mean(phi; method=:asymptotic, t_opt=n, running_intersection=false)
        @test cs.estimate ≈ ref.estimate
        @test cs.lower[20:end] ≈ ref.lower[20:end]
        @test cs.upper[20:end] ≈ ref.upper[20:end]
        @test estimand(cs) == "ATE"
        @test cs.lower[end] < 0.5 < cs.upper[end]
    end

    @testset "regression adjustment narrows the CS" begin
        a = confseq_ate(df, :y, :d; propensity=0.5)
        b = confseq_ate(df, :y, :d; propensity=0.5, covariates=[:x1, :x2],
                        outcome_learner=OLSLearner(), refit_every=50)
        @test b.upper[end] - b.lower[end] < 0.85 * (a.upper[end] - a.lower[end])
        @test b.lower[end] < 0.5 < b.upper[end]
        @test occursin("OLSLearner", method_name(b))
        # estimated propensity (logistic) and per-unit known propensities
        c = confseq_ate(df, :y, :d; covariates=[:x1, :x2],
                        outcome_learner=OLSLearner(),
                        propensity_learner=LogisticLearner())
        @test c.lower[end] < 0.5 < c.upper[end]
        df.p = fill(0.5, nrow(df))
        e = confseq_ate(df, :y, :d; propensity=:p)
        @test e.estimate ≈ a.estimate && e.lower ≈ a.lower
        # default propensity: running treated share
        f = confseq_ate(df, :y, :d)
        @test occursin("running treated share", method_name(f))
    end

    @testset "arrival order, streaming and blocks" begin
        a = confseq_ate(df, :y, :d; propensity=0.5, covariates=[:x1],
                        outcome_learner=OLSLearner(), order=:arrival, refit_every=40)
        sh = df[randperm(StableRNG(3), nrow(df)), :]
        b = confseq_ate(sh, :y, :d; propensity=0.5, covariates=[:x1],
                        outcome_learner=OLSLearner(), order=:arrival, refit_every=40)
        @test a.lower ≈ b.lower && a.upper ≈ b.upper && a.estimate ≈ b.estimate
        m = ATEMonitor(; propensity=0.5, outcome_learner=OLSLearner(), refit_every=40,
                       t_opt=nrow(df))
        for i in 1:nrow(df)
            fit!(m, df.y[i], df.d[i], [df.x1[i]])
        end
        s = snapshot(m)
        @test s.n_pending == 0 && s.n == nrow(df)
        fit!(m)
        @test confidence_sequence(m).upper ≈ a.upper
        # pending units: processed on fit!(m)
        m2 = ATEMonitor(; propensity=0.5, outcome_learner=OLSLearner(), refit_every=40,
                        t_opt=100)
        fit!(m2, df.y[1:30], df.d[1:30], reshape(df.x1[1:30], :, 1))
        @test snapshot(m2).n_pending == 30 && snapshot(m2).n == 0
        fit!(m2)
        @test snapshot(m2).n == 30
        @test sequence_path(m2).n == 1:30
    end

    @testset "validation" begin
        @test_throws ArgumentError confseq_ate(df, :y, :nope)
        bad = copy(df); bad.d = 2 .* bad.d
        @test_throws ArgumentError confseq_ate(bad, :y, :d)
        @test_throws ArgumentError ATEMonitor(; t_opt=10, propensity=1.2)
        @test_throws ArgumentError ATEMonitor()
        @test_throws ArgumentError ATEMonitor(; t_opt=10, propensity=0.5,
                                              propensity_learner=LogisticLearner())
        m = ATEMonitor(; t_opt=10, propensity=0.5)
        @test_throws ArgumentError fit!(m, 1.0, 0.5)
        @test_throws ArgumentError fit!(m, 1.0, 1; propensity=0.3)
        fit!(m, 1.0, 1, [1.0])
        @test_throws DimensionMismatch fit!(m, 1.0, 1, [1.0, 2.0])
        tie = copy(df); tie.arrival .= 1
        @test_throws ArgumentError confseq_ate(tie, :y, :d; order=:arrival)
    end
end

@testset "mSPRT (always-valid p-values for A/B tests)" begin
    rng = StableRNG(31)
    n = 1000
    d = rand(rng, n) .< 0.5
    y = randn(rng, n) .+ 0.3 .* d
    df = DataFrame(y=y, d=d)

    @testset "closed form" begin
        tau = 0.4
        t = msprt_test(df, :y, :d; tau=tau, sigma=1.0)
        n1 = sum(d); n0 = n - n1
        V = 1 / n1 + 1 / n0
        est = mean(y[d]) - mean(y[.!d])
        lr = sqrt(V / (V + tau^2)) * exp(tau^2 * est^2 / (2V * (V + tau^2)))
        @test t.evalue[end] ≈ lr rtol = 1e-10
        @test t.estimate[end] ≈ est
        @test t.pvalue[end] ≈ min(1, minimum(1 ./ t.evalue))
        rad = sqrt(V * (V + tau^2) / tau^2 * (2log(20) + log((V + tau^2) / V)))
        @test t.upper[end] <= est + rad + 1e-12 && t.lower[end] >= est - rad - 1e-12
        # interval ⟺ test duality at each step (running versions)
        @test all((t.pvalue .<= 0.05) .== .!((t.lower .<= 0) .& (0 .<= t.upper)))
        @test stopping(t).rejected
        @test pvalue(t) < 0.05
    end

    @testset "binary outcomes, tuning and streaming" begin
        yb = Float64.(rand(rng, n) .< (0.2 .+ 0.08 .* d))
        dfb = DataFrame(y=yb, d=d)
        t = msprt_test(dfb, :y, :d; outcome_type=:binary)
        m = MSPRTMonitor(; outcome_type=:binary, n_opt=n)
        for i in 1:n
            fit!(m, yb[i], d[i])
        end
        @test sequential_test(m).pvalue ≈ t.pvalue
        s = snapshot(m)
        @test s.n == n && s.n_treated == sum(d) && isfinite(s.tau)
        @test s.pvalue ≈ t.pvalue[end]
        @test nrow(sequence_path(m)) == n
        @test_throws ArgumentError confidence_sequence(m)
        dt = DiagnosticTest(t)
        @test dt.pvalue == pvalue(t)
    end

    @testset "validation" begin
        @test_throws ArgumentError MSPRTMonitor()                  # no tau / n_opt
        @test_throws ArgumentError MSPRTMonitor(; tau=-1)
        @test_throws ArgumentError MSPRTMonitor(; n_opt=10, outcome_type=:foo)
        @test_throws ArgumentError MSPRTMonitor(; n_opt=10, outcome_type=:binary,
                                                sigma=1)
        m = MSPRTMonitor(; n_opt=10, outcome_type=:binary)
        @test_throws ArgumentError fit!(m, 0.5, 1)
        @test_throws ArgumentError fit!(m, 1, 3)
        @test_throws ArgumentError msprt_test(df, :nope, :d)
    end
end
