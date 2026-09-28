@testset "confidence sequences for means" begin
    data = CSV.read(joinpath(SEQ_VALDIR, "confseq_data.csv"), DataFrame)
    ref = CSV.read(joinpath(SEQ_VALDIR, "confseq_reference.csv"), DataFrame)

    @testset "normal-mixture boundary = confseq" begin
        nm = filter(:method => ==("nm_bound"), ref)
        ours = [DrSnow._seq_nm_bound(r.t, r.alpha, DrSnow._seq_nm_rho(r.t_opt, r.alpha))
                for r in eachrow(nm)]
        @test ours ≈ nm.lower rtol = 1e-12
    end

    for g in groupby(filter(:method => !=("nm_bound"), ref),
                     [:method, :data, :alpha, :t_opt, :running_intersection])
        k = g[1, :]
        @testset "bounded CS path = confseq ($(k.method), $(k.data), α=$(k.alpha))" begin
        x = data[!, Symbol(k.data)]
        cs = confseq_mean(x; method=Symbol(k.method), level=1 - k.alpha,
                          t_opt=k.t_opt < 0 ? nothing : k.t_opt, bounds=(0, 1),
                          running_intersection=k.running_intersection)
        t = round.(Int, g.t)
        @test cs.lower[t] ≈ g.lower atol = 1e-12
        @test cs.upper[t] ≈ g.upper atol = 1e-12
        end
    end

    @testset "asymptotic CS formula" begin
        rng = StableRNG(11)
        x = randn(rng, 300) .* 2 .+ 1
        cs = confseq_mean(x; method=:asymptotic, t_opt=100, running_intersection=false,
                          min_n=2)
        n = 150
        m = mean(x[1:n]); s = sqrt(mean((x[1:n] .- m) .^ 2))
        α = 0.05
        ρ = sqrt((-2log(α) + log(-2log(α) + 1)) / 100)      # Waudby-Smith et al. (2024)
        rad = s * sqrt(2 * (n * ρ^2 + 1) / (n^2 * ρ^2) * log(sqrt(n * ρ^2 + 1) / α))
        @test cs.lower[n] ≈ m - rad rtol = 1e-12
        @test cs.upper[n] ≈ m + rad rtol = 1e-12
        @test cs.estimate[n] ≈ m
        @test cs.sigma[n] ≈ s
        # known-σ normal mixture: same formula with σ
        cs2 = confseq_mean(x; method=:normal_mixture, sigma=2, t_opt=100,
                           running_intersection=false)
        rad2 = 2 * sqrt(2 * (n * ρ^2 + 1) / (n^2 * ρ^2) * log(sqrt(n * ρ^2 + 1) / α))
        @test cs2.upper[n] - cs2.estimate[n] ≈ rad2 rtol = 1e-12
        # min_n: unbounded before, finite after
        cs3 = confseq_mean(x; method=:asymptotic, t_opt=100)
        @test all(isinf, cs3.lower[1:19]) && all(isfinite, cs3.lower[20:end])
        @test all(==(1.0), cs3.evalue[1:19])
    end

    @testset "e-process and p-value duality" begin
        rng = StableRNG(12)
        for method in (:asymptotic, :hoeffding, :empirical_bernstein, :betting)
            x = rand(rng, Beta(2, 5), 400)
            null = 0.35
            kw = method === :asymptotic ? (; t_opt=400) : (; bounds=(0, 1))
            cs = confseq_mean(x; method=method, null=null, running_intersection=false,
                              breaks=2000, kw...)
            # null inside C_t  ⟺  E_t(null) < 1/α   (grid resolution for betting)
            inside = (cs.lower .<= null) .& (null .<= cs.upper)
            small = cs.evalue .< 20
            agree = mean(inside .== small)
            @test agree >= (method === :betting ? 0.98 : 1.0)
            p = sequence_path(cs).pvalue
            @test issorted(p; rev=true)
            @test pvalues(cs)[1] ≈ p[end]
            @test all(0 .<= p .<= 1)
        end
    end

    @testset "streaming = batch; running intersection" begin
        rng = StableRNG(13)
        x = rand(rng, 250)
        for method in (:asymptotic, :empirical_bernstein, :betting, :hoeffding)
            kw = method === :asymptotic ? (; t_opt=250) : (; bounds=(0, 1))
            m = MeanMonitor(; method=method, kw...)
            for i in 1:100
                fit!(m, x[i])
            end
            fit!(m, x[101:250])
            cs = confidence_sequence(m)
            cs2 = confseq_mean(x; method=method, kw...)
            @test cs.lower ≈ cs2.lower && cs.upper ≈ cs2.upper
            s = snapshot(m)
            @test s.n == 250 && s.lower == cs.lower[end] && s.upper == cs.upper[end]
            @test s.pvalue ≈ pvalues(cs)[1]
            @test issorted(cs.lower) && issorted(cs.upper; rev=true)
            @test nobs(m) == 250 && nobs(cs) == 250
        end
        # bounds rescaling: data on [10, 20]
        y = 10 .+ 10 .* x
        a = confseq_mean(x; method=:empirical_bernstein, bounds=(0, 1))
        b = confseq_mean(y; method=:empirical_bernstein, bounds=(10, 20), null=10)
        @test b.lower ≈ 10 .+ 10 .* a.lower && b.upper ≈ 10 .+ 10 .* a.upper
        # no-record monitor still reports
        m = MeanMonitor(; method=:betting, bounds=(0, 1), record=false)
        fit!(m, x)
        @test snapshot(m).n == 250
        @test_throws ArgumentError confidence_sequence(m)
    end

    @testset "result interface" begin
        rng = StableRNG(14)
        x = randn(rng, 400) .+ 0.5
        cs = confseq_mean(x; method=:asymptotic)
        @test coef(cs) == [cs.estimate[end]]
        @test stderror(cs)[1] ≈ cs.sigma[end] / sqrt(400)
        @test confint(cs) == [cs.lower[end] cs.upper[end]]
        # other level: recomputed path, nested intervals
        c90 = confint(cs; level=0.9)
        @test c90[1] > cs.lower[end] && c90[2] < cs.upper[end]
        # anytime-valid CS is wider than the fixed-n CI at n = 400
        fixed = 2 * 1.959964 * stderror(cs)[1]
        @test cs.upper[end] - cs.lower[end] > fixed
        @test estimand(cs) == "mean"
        @test occursin("confidence sequence", sprint(show, MIME"text/plain"(), cs))
        @test nrow(tidy(cs)) == 1
        b = confseq_mean(rand(rng, 50); method=:betting, bounds=(0, 1))
        @test_throws ArgumentError confint(b; level=0.9)
        t = sequential_test(cs)
        @test t isa SequentialTest
        @test pvalue(t) ≈ pvalues(cs)[1]
        s = stopping(cs)
        @test s.rejected == (pvalues(cs)[1] <= 0.05)
        # with the running intersection, rejecting 0 ⟺ the CS excludes 0
        excl = findfirst(i -> !(cs.lower[i] <= 0 <= cs.upper[i]), eachindex(cs.n))
        @test s.step == excl
        dt = DiagnosticTest(t)
        @test dt isa DiagnosticTest && dt.pvalue == pvalue(t)
        @test occursin("H₀", sprint(show, MIME"text/plain"(), t))
    end

    @testset "input validation" begin
        @test_throws ArgumentError confseq_mean(Float64[])
        @test_throws ArgumentError confseq_mean([0.5, 1.5]; method=:betting,
                                                bounds=(0, 1))
        @test_throws ArgumentError confseq_mean([0.5]; method=:hoeffding)
        @test_throws ArgumentError confseq_mean([0.5]; method=:normal_mixture)
        @test_throws ArgumentError confseq_mean([0.5]; method=:foo)
        @test_throws ArgumentError confseq_mean([0.5]; level=1.2)
        @test_throws ArgumentError MeanMonitor(; method=:asymptotic)       # no t_opt
        @test_throws ArgumentError MeanMonitor(; method=:betting, bounds=(1, 0))
        @test_throws ArgumentError MeanMonitor(; method=:betting, bounds=(0, 1),
                                               null=2)
        @test_throws ArgumentError fit!(MeanMonitor(; t_opt=10), NaN)
        @test_throws ArgumentError MeanMonitor(; t_opt=-1)
    end
end
