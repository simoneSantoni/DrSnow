# Analytic power / MDE / sample size against pwr, PowerUpR, rdpower and closed forms.

using Distributions: Normal, TDist, cdf, quantile

const POWREF = let df = CSV.read(joinpath(DES_VALDIR, "reference_power.csv"), DataFrame)
    Dict((String(r.case), String(r.quantity)) => Float64(r.value) for r in eachrow(df))
end
pref(case, q) = POWREF[(case, q)]

@testset "two-sample means vs pwr" begin
    @test power_means(effect=0.5, n=128).power ≈ pref("t_power", "power") atol = 1e-12
    @test power_means(effect=0.3, n=100, alpha=0.01, alternative=:greater).power ≈
          pref("t_power_a01_greater", "power") atol = 1e-12
    @test power_means(effect=-0.4, n=80, alternative=:less).power ≈
          pref("t_power_less", "power") atol = 1e-12
    r = power_means(n=128, power=0.8)
    @test r.solved == :effect
    @test r.effect ≈ pref("t_mde", "d") atol = 1e-4        # pwr's uniroot tolerance
    @test power_means(effect=r.effect, n=128).power ≈ 0.8 atol = 1e-10
    r = power_means(effect=0.5, power=0.8)
    @test r.parameters.n / 2 ≈ pref("t_n", "n") atol = 1e-3
    @test power_means(effect=0.5, n=r.parameters.n).power ≈ 0.8 atol = 1e-10
    @test power_means(effect=0.6, n=100, p_treat=0.3).power ≈
          pref("t2n_power", "power") atol = 1e-12
    @test power_means(n=100, p_treat=0.3, power=0.9).effect ≈
          pref("t2n_mde", "d") atol = 1e-4
    # sd scaling and :less gives a negative MDE
    @test power_means(effect=1.0, sd=2.0, n=128).power ≈
          pref("t_power", "power") atol = 1e-12
    @test power_means(n=128, power=0.8, alternative=:less).effect < 0
    # covariate adjustment: R² shrinks the residual variance, covariates cost dof
    a = power_means(effect=0.3, n=200, r2=0.5, n_covariates=3)
    @test a.se ≈ sqrt(0.5 / (0.25 * 200)) && a.dof == 195
    # normal reference: closed form
    se = sqrt(1 / (0.25 * 200)); z = quantile(Normal(), 0.975)
    @test power_means(effect=0.3, n=200, distribution=:normal).power ≈
          cdf(Normal(), 0.3 / se - z) + cdf(Normal(), -0.3 / se - z)
    # Bloom multiplier MDE
    b = power_means(effect=0.5, n=128)
    t = TDist(126)
    @test b.mde_multiplier ≈ (quantile(t, 0.975) + quantile(t, b.power)) * b.se
end

@testset "proportions vs pwr and power.prop.test" begin
    @test power_proportions(p1=0.35, p0=0.25, n=600).power ≈
          pref("p2_power", "power") atol = 1e-12
    @test power_proportions(p1=0.35, p0=0.25, power=0.8).parameters.n / 2 ≈
          pref("p2_n", "n") atol = 1e-3
    @test power_proportions(p1=0.35, p0=0.25, n=600, p_treat=1 / 3).power ≈
          pref("p2n_power", "power") atol = 1e-12
    @test power_proportions(p1=0.35, p0=0.25, n=400, alternative=:greater).power ≈
          pref("p2_greater", "power") atol = 1e-12
    @test power_proportions(p1=0.35, p0=0.25, n=600, method=:pooled).power ≈
          pref("prop_test_power", "power") atol = 1e-12
    @test power_proportions(p1=0.35, p0=0.25, power=0.8, method=:pooled).parameters.n / 2 ≈
          pref("prop_test_n", "n") atol = 1e-6
    r = power_proportions(p0=0.25, n=600, power=0.8)
    @test r.effect > 0.25 && r.parameters.difference ≈ r.effect - 0.25
    @test power_proportions(p1=r.effect, p0=0.25, n=600).power ≈ 0.8 atol = 1e-9
    rl = power_proportions(p0=0.25, n=600, power=0.8, alternative=:less)
    @test rl.effect < 0.25
    @test_throws ArgumentError power_proportions(p1=1.2, p0=0.25, n=100)
    @test_throws ArgumentError power_proportions(p1=0.3, p0=0.25, n=100, method=:exact)
end

@testset "cluster and blocked designs vs PowerUpR" begin
    kw = (icc=0.1, cluster_size=20, r2_individual=0.3, r2_cluster=0.5,
          n_cluster_covariates=1)
    r = power_cluster(; effect=0.25, n_clusters=40, kw...)
    @test r.power ≈ pref("cra2_power", "power") atol = 1e-12
    @test r.dof == 37
    @test power_cluster(effect=0.3, icc=0.2, cluster_size=10, n_clusters=30, p_treat=0.4,
                        alternative=:greater).power ≈
          pref("cra2_power_onesided", "power") atol = 1e-12
    m = power_cluster(; n_clusters=40, power=0.8, kw...)
    @test m.mde_multiplier ≈ pref("cra2_mdes", "mdes") atol = 1e-12
    @test abs(m.effect - pref("cra2_mdes", "mdes")) < 1e-3     # exact vs multiplier
    j = power_cluster(; effect=0.25, power=0.8, kw...)
    @test ceil(Int, j.parameters.n_clusters) == Int(pref("cra2_mrss", "J"))
    # design effect and equal-size special case
    de = power_cluster(effect=0.2, icc=0.05, cluster_size=10, n_clusters=50)
    @test de.parameters.design_effect ≈ 1.45
    # unequal cluster sizes: Eldridge design effect 1 + ((cv² + 1) m - 1) ρ
    sz = [10, 20, 30, 40]
    u = power_cluster(effect=0.2, icc=0.05, cluster_sizes=sz, n_clusters=60)
    cv = std(sz; corrected=false) / mean(sz)
    @test u.parameters.design_effect ≈ 1 + ((cv^2 + 1) * 25 - 1) * 0.05
    @test u.se ≈ sqrt(u.parameters.design_effect / (0.25 * 60 * 25))
    @test u.power <
          power_cluster(effect=0.2, icc=0.05, cluster_size=25, n_clusters=60).power
    # cluster size cannot beat the cluster-level variance
    @test_throws ArgumentError power_cluster(effect=0.1, icc=0.3, n_clusters=20, power=0.9)
    cs = power_cluster(effect=0.3, icc=0.1, n_clusters=60, power=0.8)
    @test cs.solved == :cluster_size && cs.parameters.cluster_size > 0
    @test power_blocked(effect=0.2, n_blocks=100, block_size=2, r2=0.5).power ≈
          pref("bira_pairs_power", "power") atol = 1e-12
    @test power_blocked(effect=0.2, n_blocks=50, block_size=8, r2=0.3,
                        n_covariates=2).power ≈
          pref("bira_power", "power") atol = 1e-12
    @test power_blocked(n_blocks=50, block_size=8, r2=0.3, n_covariates=2,
                        power=0.8).mde_multiplier ≈ pref("bira_mdes", "mdes") atol = 1e-12
    @test power_blocked(effect=0.2, n_blocks=100, block_size=2).dof == 99
end

@testset "repeated measurements (McKenzie 2012)" begin
    z = quantile(Normal(), 0.975)
    pw(se, δ) = cdf(Normal(), δ / se - z) + cdf(Normal(), -δ / se - z)
    for (m, r, ρ) in ((1, 1, 0.5), (1, 3, 0.5), (3, 3, 0.3), (2, 5, 0.7), (4, 1, 0.0))
        post = (1 + (r - 1) * ρ) / r
        did = (1 + (m - 1) * ρ) / m + post - 2ρ
        anc = post - m * ρ^2 / (1 + (m - 1) * ρ)
        for (est, v) in ((:post, post), (:did, did), (:ancova, anc))
            a = power_did(effect=0.2, n=300, pre_periods=m, post_periods=r, rho=ρ,
                          estimator=est, distribution=:normal)
            @test a.parameters.variance_factor ≈ v
            @test a.power ≈ pw(sqrt(v / (0.25 * 300)), 0.2)
        end
        @test anc <= min(post, did) + 1e-12
    end
    # the explicit correlation matrix reproduces the equicorrelated case
    C = fill(0.4, 5, 5); C[diagind(C)] .= 1
    @test power_did(effect=0.2, n=300, pre_periods=2, post_periods=3, corr=C).power ≈
          power_did(effect=0.2, n=300, pre_periods=2, post_periods=3, rho=0.4).power
    @test power_did(effect=0.2, power=0.8, rho=0.5).solved == :n
    @test_throws DimensionMismatch power_did(effect=0.2, n=100, pre_periods=1,
                                             post_periods=1, corr=ones(3, 3))
    @test_throws ArgumentError power_did(effect=0.2, n=100, estimator=:foo)
end

@testset "repeated measurements: Monte Carlo with AR(1) errors" begin
    rng = StableRNG(11)
    n, m, r, φ, δ = 200, 2, 3, 0.6, 0.25
    T = m + r
    C = [φ^abs(s - t) for s in 1:T, t in 1:T]
    L = cholesky(C).L
    S = mc_reps(2000, 300)
    ests = Dict(:did => Float64[], :ancova => Float64[])
    for _ in 1:S
        E = (L * randn(rng, T, n))'
        d = zeros(Int, n); d[randperm(rng, n)[1:(n ÷ 2)]] .= 1
        Y = E .+ δ .* d .* (1:T .> m)'
        pre = vec(mean(Y[:, 1:m]; dims=2)); post = vec(mean(Y[:, (m + 1):T]; dims=2))
        ch = post .- pre
        push!(ests[:did], mean(ch[d .== 1]) - mean(ch[d .== 0]))
        X = hcat(ones(n), d, pre)
        push!(ests[:ancova], (X \ post)[2])
    end
    for est in (:did, :ancova)
        a = power_did(effect=δ, n=n, pre_periods=m, post_periods=r, corr=C, estimator=est)
        sd_mc = std(ests[est])
        @test abs(sd_mc / a.se - 1) < 4 / sqrt(2 * S)
        @test abs(mean(ests[est]) - δ) < 4 * sd_mc / sqrt(S)
    end
end

@testset "IV / encouragement design" begin
    a = power_iv(effect=0.5, compliance=0.4, n=1000)
    @test a.se ≈ sqrt(1 / (0.25 * 1000)) / 0.4
    @test a.parameters.itt ≈ 0.2
    @test a.power ≈ power_means(effect=0.2, n=1000).power
    @test power_iv(compliance=0.4, n=1000, power=0.8).effect ≈
          power_means(n=1000, power=0.8).effect / 0.4 rtol = 1e-8
    c = power_iv(effect=0.5, n=1000, power=0.8)
    @test c.solved == :compliance && 0 < c.parameters.compliance < 1
    @test_throws ArgumentError power_iv(effect=0.5, compliance=1.5, n=100)
    # Monte Carlo with 2SLS (one-sided noncompliance)
    rng = StableRNG(12)
    S = mc_reps(1000, 250)
    n, π, late = 600, 0.5, 0.5
    rej = 0
    for _ in 1:S
        zz = zeros(Int, n); zz[randperm(rng, n)[1:(n ÷ 2)]] .= 1
        complier = rand(rng, n) .< π
        dd = zz .* complier
        y = late .* dd .+ randn(rng, n)
        r = late_2sls(DataFrame(y=y, d=dd, z=zz), :y, :d, :z)
        rej += pvalues(r)[1] < 0.05
    end
    pa = power_iv(effect=late, compliance=π, n=n).power
    @test abs(rej / S - pa) < 4 * sqrt(pa * (1 - pa) / S) + 0.03
end

@testset "regression discontinuity vs rdpower" begin
    sen = dropmissing(CSV.read(joinpath(@__DIR__, "..", "validation", "rd", "senate.csv"),
                               DataFrame)[:, [:vote, :margin]])
    r = power_rd(sen, :vote, :margin; effect=5)
    @test r.power ≈ pref("rd_senate", "power.rbc") rtol = 1e-8
    @test r.se ≈ pref("rd_senate", "se.rbc") rtol = 1e-8
    @test r.parameters.power_conventional ≈ pref("rd_senate", "power.conv") rtol = 1e-8
    @test r.parameters.se_conventional ≈ pref("rd_senate", "se.conv") rtol = 1e-8
    h = pref("rd_senate", "samph.l")
    @test r.parameters.h_left ≈ h rtol = 1e-8
    @test r.parameters.bias ≈ (pref("rd_senate", "bias.l") + pref("rd_senate", "bias.r")) *
                             h^2 rtol = 1e-6
    @test r.parameters.V_rb_sum ≈
          pref("rd_senate", "Vl.rb") + pref("rd_senate", "Vr.rb") rtol = 1e-8
    @test r.parameters.n_pilot_left == pref("rd_senate", "N.l")
    @test r.parameters.n_pilot_h_right == pref("rd_senate", "Nh.r")
    s = power_rd(sen, :vote, :margin; effect=4, sampsi=(300, 350))
    @test s.power ≈ pref("rd_senate_sampsi", "power.rbc") rtol = 1e-8
    @test s.se ≈ pref("rd_senate_sampsi", "se.rbc") rtol = 1e-8
    @test s.parameters.power_conventional ≈
          pref("rd_senate_sampsi", "power.conv") rtol = 1e-8
    b = power_rd(sen, :vote, :margin; effect=4, samph=12)
    @test b.power ≈ pref("rd_senate_samph", "power.rbc") rtol = 1e-8
    @test b.se ≈ pref("rd_senate_samph", "se.rbc") rtol = 1e-8
    # parameter interface with rdpower's per-side constants
    p = power_rd(variance=(pref("rd_senate", "Vl.rb"), pref("rd_senate", "Vr.rb")),
                 samph=h, n=pref("rd_senate", "N.l") + pref("rd_senate", "N.r"), effect=5)
    @test p.power ≈ pref("rd_senate", "power.rbc") rtol = 1e-8
    # solving for the sample size and the MDE
    ns = power_rd(sen, :vote, :margin; effect=5, power=0.9)
    @test ns.solved == :n && ns.parameters.n > r.parameters.n
    @test power_rd(sen, :vote, :margin; effect=5, n=ns.parameters.n).power ≈ 0.9 atol = 1e-8
    md = power_rd(sen, :vote, :margin; power=0.8)
    @test md.solved == :effect && md.effect < 5
    @test_throws ArgumentError power_rd(sen, :vote, :margin; effect=4, samph=(10, 12))
    @test_throws ArgumentError power_rd(sen, :vote, :margin; effect=4, treatment=:vote)
end

@testset "solver and input errors" begin
    @test_throws ArgumentError power_means(n=100)                    # two unknowns
    @test_throws ArgumentError power_means(effect=0.3, n=100, power=0.8)
    @test_throws ArgumentError power_means(effect=0.3, n=100, alpha=1.2)
    @test_throws ArgumentError power_means(effect=0.3, n=100, alternative=:both)
    @test_throws ArgumentError power_means(effect=0.3, n=2)
    @test_throws ArgumentError power_means(effect=0.3, power=0.01)   # below alpha
    @test_throws ArgumentError power_means(effect=0.3, n=100, sd=-1)
    @test_throws ArgumentError power_means(effect=0.3, n=100, r2=1.0)
    @test_throws ArgumentError power_cluster(effect=0.2, icc=1.5, cluster_size=10,
                                             n_clusters=20)
    @test_throws ArgumentError power_blocked(effect=0.2, n_blocks=10, block_size=1)
    @test_throws ArgumentError power_cluster(effect=0.2, icc=0.1, cluster_size=10,
                                             cluster_sizes=[5, 10], n_clusters=20)
    r = power_means(effect=0.3, n=100)
    @test occursin("Solved for: power", sprint(show, MIME"text/plain"(), r))
    @test occursin("PowerAnalysis", sprint(show, r))
    @test occursin("round up", sprint(show, MIME"text/plain"(), power_means(effect=0.3,
                                                                           power=0.8)))
end
