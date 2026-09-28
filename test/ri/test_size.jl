@testset "Monte Carlo size of Fisher tests" begin
    @testset "sharp null, several designs and statistics" begin
        reps = mc_reps(1000, 150)
        rng = StableRNG(606)
        rej = Dict{Symbol,Int}(:diff_means => 0, :studentized => 0, :rank_sum => 0,
                               :lin => 0, :cluster => 0)
        for rep in 1:reps
            N = 40
            x = randn(rng, N)
            y = x .+ randexp(rng, N)                   # skewed, no effect
            z = shuffle(rng, [trues(12); falses(28)])
            df = DataFrame(y=y, d=Int.(z), x=x, c=repeat(1:10, inner=4))
            for s in (:diff_means, :studentized, :rank_sum)
                r = randomization_test(df, :y, :d; statistic=s, nperm=199,
                                       rng=StableRNG(rep))
                rej[s] += r.pvalue <= 0.05
            end
            r = randomization_test(df, :y, :d; statistic=:lin, covariates=[:x],
                                   nperm=199, rng=StableRNG(rep))
            rej[:lin] += r.pvalue <= 0.05
            zc = shuffle(rng, [trues(4); falses(6)])
            df.dc = Int.(zc[df.c])
            r = randomization_test(df, :y, :dc; cluster=:c)      # exact, 210
            rej[:cluster] += r.pvalue <= 0.05
        end
        se = sqrt(0.05 * 0.95 / reps)
        for (k, v) in rej
            @test v / reps <= 0.05 + 3se
        end
        @info "Fisher test size under the sharp null (nominal 0.05)" reps rates =
            Dict(k => v / reps for (k, v) in rej)
    end

    @testset "weak null with heterogeneous effects (Wu & Ding 2021)" begin
        # ATE = 0 but effects vary: the studentized statistic keeps its size, the
        # plain difference in means does not when group sizes and variances differ.
        reps = mc_reps(1000, 200)
        rng = StableRNG(707)
        rej_t = 0; rej_d = 0
        for rep in 1:reps
            N = 100
            y0 = 0.5 .* randn(rng, N)
            y1 = 3.0 .* randn(rng, N)
            y1 .+= mean(y0) - mean(y1)                 # finite-population ATE = 0
            z = shuffle(rng, [trues(20); falses(80)])
            df = DataFrame(y=ifelse.(z, y1, y0), d=Int.(z))
            rej_t += randomization_test(df, :y, :d; statistic=:studentized, nperm=199,
                                        rng=StableRNG(rep)).pvalue <= 0.05
            rej_d += randomization_test(df, :y, :d; nperm=199,
                                        rng=StableRNG(rep)).pvalue <= 0.05
        end
        se = sqrt(0.05 * 0.95 / reps)
        @test rej_t / reps <= 0.05 + 3se + 0.01      # asymptotic validity
        @test rej_d / reps > rej_t / reps            # illustrates the failure mode
        @info "Weak-null size (nominal 0.05)" reps studentized = rej_t / reps diff_means =
            rej_d / reps
    end
end
