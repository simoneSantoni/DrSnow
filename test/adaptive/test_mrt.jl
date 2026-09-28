# Micro-randomized trials: WCLS and EMEE against MRTAnalysis 0.4.1, invariance,
# cross-check with FixedEffectModels, errors.

@testset "WCLS / EMEE match MRTAnalysis" begin
    ref = ad_read_csv("mrt_reference.csv")
    hs = ad_read_csv("mrt_heartsteps.csv")
    bin = ad_read_csv("mrt_binary.csv")
    sim = ad_read_csv("mrt_sim.csv")
    fits = Dict(
        "wcls_hs_marginal" => wcls(hs, :logstep_30min, :intervention, :userid;
                                   rand_prob=0.6, availability=:avail),
        "wcls_hs_moderated" => wcls(hs, :logstep_30min, :intervention, :userid;
                                    rand_prob=0.6, moderators=[:logstep_pre30min],
                                    controls=[:logstep_pre30min, :logstep_30min_lag1,
                                              :is_at_home_or_work],
                                    availability=:avail),
        "wcls_hs_numerator" => wcls(hs, :logstep_30min, :intervention, :userid;
                                    rand_prob=:rand_prob,
                                    moderators=[:is_at_home_or_work],
                                    controls=[:is_at_home_or_work, :logstep_pre30min],
                                    availability=:avail, numerator_prob=0.5),
        # 120 participants: MRTAnalysis applies no small-sample correction
        "wcls_sim" => wcls(sim, :y, :a, :id; rand_prob=:prob, moderators=[:dp],
                           controls=[:dp, :x], availability=:avail,
                           numerator_prob=:ptilde, small_sample=false),
        "emee_bin_marginal" => emee(bin, :Y, :A, :userid; rand_prob=:rand_prob,
                                    availability=:avail),
        "emee_bin_moderated" => emee(bin, :Y, :A, :userid; rand_prob=:rand_prob,
                                     moderators=[:time_var1],
                                     controls=[:time_var1, :time_var2],
                                     availability=:avail),
        "emee_bin_numerator" => emee(bin, :Y, :A, :userid; rand_prob=:rand_prob,
                                     moderators=[:time_var2],
                                     controls=[:time_var1, :time_var2],
                                     availability=:avail, numerator_prob=0.4))
    for r in eachrow(ref)
        f = fits[r.case]
        j = findfirst(==(r.term), coefnames(f))
        @test j !== nothing
        # R's multiroot tolerance limits EMEE agreement to ~1e-9
        @test coef(f)[j] ≈ r.estimate rtol = 1e-7 atol = 1e-9
        @test stderror(f)[j] ≈ r.std_error rtol = 1e-7
        @test dof_residual(f) == r.df
    end
    f = fits["wcls_hs_moderated"]
    @test nobs(f) == count(==(1), hs.avail)
    @test f.n_ids == 37
    @test glance(f).n_clusters[1] == 37
    @test occursin("small-sample corrected", sprint(show, MIME"text/plain"(), f))
    @test confint(f)[1, 2] ≈ coef(f)[1] + quantile(TDist(31), 0.975) * stderror(f)[1]
end

@testset "WCLS point estimate = weighted regression; invariance" begin
    sim = ad_read_csv("mrt_sim.csv")
    r = wcls(sim, :y, :a, :id; rand_prob=:prob, moderators=[:dp], controls=[:dp, :x],
             availability=:avail, numerator_prob=:ptilde)
    s = sim[sim.avail .== 1, :]
    s.w = ifelse.(s.a .== 1, s.ptilde ./ s.prob, (1 .- s.ptilde) ./ (1 .- s.prob))
    s.ac = s.a .- s.ptilde
    s.acdp = s.ac .* s.dp
    m = reg(s, @formula(y ~ dp + x + ac + acdp);
            weights=:w)
    @test coef(r) ≈ coef(m)[4:5]
    @test r.control_coef ≈ coef(m)[1:3]
    perm = randperm(StableRNG(3), nrow(sim))
    r2 = wcls(sim[perm, :], :y, :a, :id; rand_prob=:prob, moderators=[:dp],
              controls=[:dp, :x], availability=:avail, numerator_prob=:ptilde)
    @test coef(r2) ≈ coef(r) && vcov(r2) ≈ vcov(r)
    bin = ad_read_csv("mrt_binary.csv")
    perm = randperm(StableRNG(4), nrow(bin))
    e1 = emee(bin, :Y, :A, :userid; rand_prob=:rand_prob, moderators=[:time_var1],
              availability=:avail)
    e2 = emee(bin[perm, :], :Y, :A, :userid; rand_prob=:rand_prob,
              moderators=[:time_var1], availability=:avail)
    @test coef(e2) ≈ coef(e1) && vcov(e2) ≈ vcov(e1)
    # unavailable rows may have missing values
    sm = allowmissing(sim)
    sm.y[sm.avail .== 0] .= missing
    r3 = wcls(sm, :y, :a, :id; rand_prob=:prob, moderators=[:dp], controls=[:dp, :x],
              availability=:avail, numerator_prob=:ptilde)
    @test coef(r3) ≈ coef(r)
    # the uncorrected sandwich is smaller
    r4 = wcls(sim, :y, :a, :id; rand_prob=:prob, moderators=[:dp], controls=[:dp, :x],
              availability=:avail, numerator_prob=:ptilde, small_sample=false)
    @test all(stderror(r4) .< stderror(r))
end

@testset "MRT errors" begin
    sim = ad_read_csv("mrt_sim.csv")
    @test_throws ArgumentError wcls(sim, :y, :a, :id)
    @test_throws ArgumentError wcls(sim, :y, :a, :id; rand_prob=1.2)
    @test_throws ArgumentError wcls(sim, :y, :a, :id; rand_prob=:prob,
                                    numerator_prob=0.0)
    @test_throws ArgumentError wcls(sim, :y, :a, :id; rand_prob=:prob,
                                    moderators=[:nope])
    @test_throws ArgumentError emee(sim, :y, :a, :id; rand_prob=:prob)   # not binary
    s2 = copy(sim)
    s2.a = Float64.(s2.a)
    s2.a[1] = 2
    @test_throws ArgumentError wcls(s2, :y, :a, :id; rand_prob=:prob)
    s3 = copy(sim)
    s3.avail = Float64.(s3.avail)
    s3.avail[1] = 0.5
    @test_throws ArgumentError wcls(s3, :y, :a, :id; rand_prob=:prob,
                                    availability=:avail)
    tiny = sim[sim.id .<= 3, :]
    @test_throws ArgumentError wcls(tiny, :y, :a, :id; rand_prob=:prob,
                                    moderators=[:dp], controls=[:dp, :x])
    col = copy(sim)
    col.x2 = 2 .* col.x
    @test_throws ArgumentError wcls(col, :y, :a, :id; rand_prob=:prob,
                                    controls=[:x, :x2])
end
