# Monte Carlo: time-uniform coverage of confidence sequences and type-I error under
# continuous monitoring, contrasted with naive repeated fixed-sample intervals.
# Reduced counts in CI (mc_reps); DRSNOW_SLOW_TESTS=true runs the full counts.

# Does the naive fixed-n 95% t-interval miss μ at some n ∈ n_from:T?
function seq_naive_misses(x, mu; n_from=10)
    s = 0.0; ss = 0.0
    for (i, xi) in enumerate(x)
        s += xi; ss += xi^2
        i < n_from && continue
        m = s / i
        v = max((ss - i * m^2) / (i - 1), 0.0)
        abs(m - mu) > 1.959964 * sqrt(v / i) && return true
    end
    return false
end

@testset "Monte Carlo: time-uniform coverage" begin
    R = mc_reps(1000, 150)
    T = 500
    rng = StableRNG(4242)
    # (method, data generator, true mean, keywords)
    cases = [
        (:asymptotic, n -> randn(rng, n) .+ 1, 1.0, (; t_opt=T)),
        (:normal_mixture, n -> randn(rng, n) .+ 1, 1.0, (; t_opt=T, sigma=1.0)),
        (:empirical_bernstein, n -> rand(rng, Beta(2, 5), n), 2 / 7, (; bounds=(0, 1))),
        (:hoeffding, n -> rand(rng, Beta(2, 5), n), 2 / 7, (; bounds=(0, 1))),
        (:betting, n -> rand(rng, Beta(2, 5), n), 2 / 7, (; bounds=(0, 1), breaks=200)),
    ]
    for (method, gen, mu, kw) in cases
        miss = 0; naive = 0
        for _ in 1:R
            x = gen(T)
            cs = confseq_mean(x; method=method, kw...)
            miss += any((cs.lower .> mu) .| (cs.upper .< mu))
            naive += seq_naive_misses(x, mu)
        end
        cov = 1 - miss / R
        # exact CSs: coverage ≥ 0.95 up to MC error; asymptotic: approximately
        slack = method === :asymptotic ? 0.03 : 0.0
        @test cov >= 0.95 - slack - 3 * sqrt(0.95 * 0.05 / R)
        # repeated fixed-n intervals fail badly under continuous monitoring
        @test 1 - naive / R < 0.8
        @info "time-uniform coverage" method cov naive_coverage = 1 - naive / R
    end

    # AIPW ATE confidence sequence (heterogeneous effects, OLS adjustment)
    Ra = mc_reps(400, 60)
    miss = 0
    for _ in 1:Ra
        df = seq_ate_data(rng, 400)
        cs = confseq_ate(df, :y, :d; propensity=0.5, covariates=[:x1, :x2],
                         outcome_learner=OLSLearner(), refit_every=50)
        miss += any((cs.lower .> 0.5) .| (cs.upper .< 0.5))
    end
    @test 1 - miss / Ra >= 0.92 - 3 * sqrt(0.05 * 0.95 / Ra)
    @info "ATE CS time-uniform coverage" coverage = 1 - miss / Ra
end

@testset "Monte Carlo: type-I error under continuous monitoring" begin
    R = mc_reps(1000, 150)
    T = 600
    rng = StableRNG(777)
    rej = 0; naive = 0
    for _ in 1:R
        d = rand(rng, T) .< 0.5
        y = randn(rng, T)
        t = msprt_test(DataFrame(y=y, d=d), :y, :d; sigma=1.0)
        rej += stopping(t).rejected
        # naive: two-sample z-test after every unit from n = 20
        s1 = s0 = 0.0; q1 = q0 = 0.0; n1 = n0 = 0
        for i in 1:T
            if d[i]
                s1 += y[i]; q1 += y[i]^2; n1 += 1
            else
                s0 += y[i]; q0 += y[i]^2; n0 += 1
            end
            (i < 20 || n1 < 2 || n0 < 2) && continue
            m1 = s1 / n1; m0 = s0 / n0
            v = (q1 - n1 * m1^2) / (n1 - 1) / n1 + (q0 - n0 * m0^2) / (n0 - 1) / n0
            if abs(m1 - m0) > 1.959964 * sqrt(v)
                naive += 1
                break
            end
        end
    end
    @test rej / R <= 0.05 + 3 * sqrt(0.05 * 0.95 / R)
    @test naive / R > 0.2
    @info "mSPRT type-I error (continuous monitoring)" msprt = rej / R naive = naive / R

    # group-sequential designs: simulated trials under H₀, five looks
    Rg = mc_reps(4000, 300)
    for des in (gs_design(; k=5), gs_design(; k=5, efficacy=:pocock),
                gs_design(; k=5, efficacy=:haybittle_peto))
        rej = 0; naive = 0
        for _ in 1:Rg
            x = randn(rng, 250)
            ests = [mean(x[1:(50k)]) for k in 1:5]
            ses = [1 / sqrt(50k) for k in 1:5]
            rej += gs_analysis(des, ests, ses).decision === :efficacy
            naive += any(ests ./ ses .>= 1.959964)
        end
        @test abs(rej / Rg - 0.025) <= 3.5 * sqrt(0.025 * 0.975 / Rg)
        @test naive / Rg > 0.05
        @info "group-sequential type-I error" des.efficacy rej / Rg naive / Rg
    end
end
