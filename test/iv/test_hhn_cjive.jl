# Hansen–Hausman–Newey (2008) many-instrument variance for LIML / Fuller and the
# cluster jackknife IV estimator (CJIVE). References:
# test/validation/iv/make_reference_hhn_cjive.R (HHN formula with explicit n × n
# matrices; ManyIV's minimum-distance SE as a closeness check; clusterIV::cjive).

const IV_REF_HC = let r = iv_read_csv(joinpath(IV_VALDIR, "reference_hhn_cjive.csv"))
    Dict((row.case, row.quantity) => row.value for row in eachrow(r))
end
const MANYIV_EXT = iv_read_csv(joinpath(IV_VALDIR, "manyiv_ext.csv"))
const MANYIV_Z = [Symbol("z", j) for j in 2:150]

"""Group-indicator instruments (unbalanced), skewed errors."""
function iv_manygroups_dgp(rng; n=800, K=60, beta=0.5, str=0.3)
    grp = vcat(repeat(1:K; inner=3), rand(rng, 1:K, n - 3K))
    s = str .* randn(rng, K)
    e1 = (randn(rng, n) .^ 2 .+ randn(rng, n) .^ 2 .- 2) ./ 2       # centred χ²(2)/2
    e2 = 0.6 .* e1 .+ 0.8 .* (-log.(rand(rng, n)) .- 1)            # centred Exp(1)
    d = s[grp] .+ e2
    y = beta .* d .+ e1
    df = DataFrame(y=y, d=d, grp=grp)
    for j in 2:K
        df[!, Symbol("g", j)] = Float64.(grp .== j)
    end
    return df, [Symbol("g", j) for j in 2:K]
end

"""Clustered design: group k is drawn only in clusters k, …, k + span − 1 (instruments
concentrated within clusters), and a cluster shock enters both the first-stage and
the structural errors, so leave-one-out predictions pick up the shock."""
function iv_clustered_groups_dgp(rng; n=3000, K=60, G=150, span=5, beta=0.5, str=0.6)
    cl = rand(rng, 1:G, n)
    grp = [mod(c - 1 + rand(rng, 0:(span - 1)), K) + 1 for c in cl]
    s = str .* randn(rng, K)
    a = randn(rng, G)
    d = s[grp] .+ a[cl] .+ randn(rng, n)
    y = beta .* d .+ 0.8 .* a[cl] .+ randn(rng, n)
    df = DataFrame(y=y, d=d, cl=cl)
    for j in 2:K
        df[!, Symbol("g", j)] = Float64.(grp .== j)
    end
    return df, [Symbol("g", j) for j in 2:K]
end

@testset "HHN variance and CJIVE" begin
    @testset "HHN: validation against the explicit formula and ManyIV" begin
        df = MANYIV_EXT
        r = kclass_iv(df, :y, :d, MANYIV_Z; method=:liml, covariates=[:w], se=:hhn)
        @test coef(r)[1] ≈ IV_REF_HC[("hhn", "liml_coef")] rtol = 1e-9
        @test stderror(r)[1] ≈ IV_REF_HC[("hhn", "liml_se_hhn")] rtol = 1e-8
        @test occursin("Hansen, Hausman & Newey", r.se_type)
        rb = kclass_iv(df, :y, :d, MANYIV_Z; method=:liml, covariates=[:w], se=:bekker)
        @test stderror(rb)[1] ≈ IV_REF_HC[("hhn", "liml_se_bekker")] rtol = 1e-8
        # the non-normality corrections matter here (unbalanced instruments) ...
        @test abs(stderror(r)[1] / stderror(rb)[1] - 1) > 0.005
        # ... and HHN is close to Kolesár's (2018) minimum-distance SE (asymptotically
        # equivalent, not numerically identical)
        @test coef(r)[1] ≈ IV_REF_HC[("hhn", "manyiv_liml_coef")] rtol = 1e-9
        @test stderror(r)[1] ≈ IV_REF_HC[("hhn", "manyiv_liml_se_md")] rtol = 0.03
        rf = kclass_iv(df, :y, :d, MANYIV_Z; method=:fuller, covariates=[:w], se=:hhn)
        @test coef(rf)[1] ≈ IV_REF_HC[("hhn", "fuller1_coef")] rtol = 1e-9
        @test stderror(rf)[1] ≈ IV_REF_HC[("hhn", "fuller1_se_hhn")] rtol = 1e-8
        # fixed effects give the same answer as the corresponding dummies
        d2 = copy(df)
        d2.cat = string.(d2.cl .% 5)
        zsub = MANYIV_Z[1:20]          # (few instruments: fast fixed-effect compile)
        a = kclass_iv(d2, :y, :d, zsub; method=:liml, covariates=[:w], fe=[:cat],
                      se=:hhn)
        b = kclass_iv(d2, :y, :d, zsub; method=:liml, covariates=[:w, :cat],
                      se=:hhn)
        @test coef(a) ≈ coef(b) rtol = 1e-9
        @test stderror(a) ≈ stderror(b) rtol = 1e-8
        sh = df[shuffle(StableRNG(712), 1:nrow(df)), :]
        rsh = kclass_iv(sh, :y, :d, MANYIV_Z; method=:liml, covariates=[:w], se=:hhn)
        @test stderror(rsh) ≈ stderror(r) rtol = 1e-8
        @test_throws ArgumentError kclass_iv(df, :y, :d, MANYIV_Z; method=:hlim,
                                             se=:hhn)
        @test_throws ArgumentError kclass_iv(df, :y, :d, MANYIV_Z; method=:kclass,
                                             kappa=1.0, se=:hhn)
    end

    @testset "CJIVE: validation against clusterIV" begin
        df = copy(MANYIV_EXT)
        c = jive(df, :y, :d, MANYIV_Z; method=:cjive, cluster=:cl)
        @test coef(c)[1] ≈ IV_REF_HC[("cjive", "coef")] rtol = 1e-9
        @test stderror(c)[1] ≈ IV_REF_HC[("cjive", "se")] rtol = 1e-8
        @test dof_residual(c) == 79
        c = jive(df, :y, :d, MANYIV_Z; method=:cjive, cluster=:cl, covariates=[:w])
        @test coef(c)[1] ≈ IV_REF_HC[("cjive", "coef_w")] rtol = 1e-9
        @test stderror(c)[1] ≈ IV_REF_HC[("cjive", "se_w")] rtol = 1e-8
        df.wt = 0.5 .+ (df.cl .% 3) ./ 2
        c = jive(df, :y, :d, MANYIV_Z; method=:cjive, cluster=:cl, covariates=[:w],
                 weights=:wt)
        @test coef(c)[1] ≈ IV_REF_HC[("cjive", "coef_wt")] rtol = 1e-9
        @test stderror(c)[1] ≈ IV_REF_HC[("cjive", "se_wt")] rtol = 1e-8
        @test occursin("CJIVE", method_name(c))
        @test occursin("leave-cluster-out", c.se_type)
        sh = df[shuffle(StableRNG(701), 1:nrow(df)), :]
        c2 = jive(sh, :y, :d, MANYIV_Z; method=:cjive, cluster=:cl, covariates=[:w],
                  weights=:wt)
        @test coef(c2) ≈ coef(c) rtol = 1e-9
        @test stderror(c2) ≈ stderror(c) rtol = 1e-8
        @test_throws ArgumentError jive(df, :y, :d, MANYIV_Z; method=:cjive)
        @test_throws ArgumentError jive(df, :y, :d, MANYIV_Z; method=:cjive,
                                        cluster=:cl, se=:many_robust)
    end

    @testset "Monte Carlo: HHN coverage (many instruments, skewed errors)" begin
        R = mc_reps(600, 100)
        rng = StableRNG(702)
        hit = zeros(2)
        for _ in 1:R
            df, zs = iv_manygroups_dgp(rng; n=600, K=60)
            r = kclass_iv(df, :y, :d, zs; method=:liml, se=:hhn)
            ci = confint(r)
            hit[1] += ci[1, 1] <= 0.5 <= ci[1, 2]
            rs = kclass_iv(df, :y, :d, zs; method=:liml, vcov=Vcov.simple())
            cs = confint(rs)
            hit[2] += cs[1, 1] <= 0.5 <= cs[1, 2]
        end
        hit ./= R
        @test abs(hit[1] - 0.95) < mc_tol(0.95, R; slack=0.02)
        @test hit[2] < hit[1]                  # conventional LIML SEs undercover
    end

    @testset "Monte Carlo: CJIVE with clustered errors" begin
        R = mc_reps(400, 80)
        rng = StableRNG(703)
        est = zeros(R, 2)
        hit = 0
        for rep in 1:R
            df, zs = iv_clustered_groups_dgp(rng)
            c = jive(df, :y, :d, zs; method=:cjive, cluster=:cl)
            u = jive(df, :y, :d, zs; method=:ujive, cluster=:cl)
            est[rep, 1] = coef(c)[1]
            est[rep, 2] = coef(u)[1]
            ci = confint(c)
            hit += ci[1, 1] <= 0.5 <= ci[1, 2]
        end
        bias = vec(median(est; dims=1)) .- 0.5
        @test abs(bias[1]) < abs(bias[2])      # UJIVE keeps the own-cluster bias
        @test abs(hit / R - 0.95) < mc_tol(0.95, R; slack=0.04)
    end
end
