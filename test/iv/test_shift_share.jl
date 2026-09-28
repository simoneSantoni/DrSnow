# Shift-share IV: construction, AKM / AKM0 / BHJ inference, Rotemberg weights,
# recentering and randomization inference.

const SS_REF = let r = iv_read_csv(joinpath(IV_VALDIR, "reference_shiftshare.csv"))
    Dict((row.case, row.quantity) => row.value for row in eachrow(r))
end
const SS_DF = iv_read_csv(joinpath(IV_VALDIR, "shiftshare.csv"))
const SS_SHOCKS = iv_read_csv(joinpath(IV_VALDIR, "shiftshare_shocks.csv"))
const SS_SH = [Symbol("s", k) for k in 1:40]
SS_DF.ssum = vec(sum(Matrix(SS_DF[:, SS_SH]); dims=2))

"""Design-based shift-share DGP: shares `S` and the region-level errors `u`, `v` are
held fixed; only the shocks are redrawn (`u` has a sector component, so residuals are
correlated across regions with similar shares)."""
function iv_shift_share_draw(rng, S, u, v; beta=1.0, fs=0.8)
    K = size(S, 2)
    g = randn(rng, K)
    d = fs .* (S * g) .+ v
    df = DataFrame(y=beta .* d .+ u, d=d)
    for k in 1:K
        df[!, Symbol("s", k)] = S[:, k]
    end
    return df, g
end

@testset "shift-share IV" begin
    g = SS_SHOCKS.shock
    cl = Int.(SS_SHOCKS.cluster)
    draws = Matrix{Float64}(SS_SHOCKS[:, 5:end])

    @testset "construction" begin
        B = shift_share_instrument(SS_DF, SS_SH, g)
        S = Matrix(SS_DF[:, SS_SH])
        @test B ≈ S * g
        d = Dict(string(s) => g[k] for (k, s) in enumerate(SS_SH))
        @test shift_share_instrument(SS_DF, SS_SH, d) ≈ B
        @test_throws ArgumentError shift_share_instrument(SS_DF, SS_SH, g[1:3])
        @test_throws ArgumentError shift_share_instrument(SS_DF, SS_SH,
                                                          Dict("s1" => 1.0))
        dm = allowmissing(copy(SS_DF))
        dm[2, :s3] = missing
        @test ismissing(shift_share_instrument(dm, SS_SH, g)[2])
    end

    @testset "validation against ShiftShareSE and paper formulas" begin
        n, K = nrow(SS_DF), 4
        for (case, w, sc) in (("unw", nothing, nothing), ("w", :pop, nothing),
                              ("wcl", :pop, cl))
            r = shift_share_iv(SS_DF, :y, :d, SS_SH, g; covariates=[:x1], weights=w,
                               shock_clusters=sc)
            @test coef(r)[1] ≈ SS_REF[(case, "beta")] rtol = 1e-8
            @test stderror(r)[1] ≈ SS_REF[(case, "se_akm")] rtol = 1e-7
            inf = r.inference
            @test inf.se[1] * sqrt((n - K) / n) ≈ SS_REF[(case, "se_ehw")] rtol = 1e-7
            @test inf.se[3] ≈ SS_REF[(case, "bhj_se")] rtol = 1e-7
            @test r.akm0_set.kind === :bounded
            @test r.akm0_set.intervals[1][1] ≈ SS_REF[(case, "akm0_lower")] rtol = 1e-7
            @test r.akm0_set.intervals[1][2] ≈ SS_REF[(case, "akm0_upper")] rtol = 1e-7
            @test DrSnow.pvalue(r.akm0_set, 0.0) ≈ SS_REF[(case, "akm0_p_b0")] rtol = 1e-7
            rb = shift_share_iv(SS_DF, :y, :d, SS_SH, g; covariates=[:x1], weights=w,
                                shock_clusters=sc, se=:bhj)
            @test coef(rb)[1] ≈ SS_REF[(case, "bhj_beta")] rtol = 1e-8
            @test stderror(rb)[1] ≈ SS_REF[(case, "bhj_se")] rtol = 1e-7
            @test nrow(rb.shock_level) == 40
            rw = rotemberg_weights(SS_DF, :y, :d, SS_SH, g; covariates=[:x1, :ssum],
                                   weights=w)
            tab = rw.table
            α(s) = tab.alpha[findfirst(==(s), tab.sector)]
            @test sum(tab.alpha) ≈ 1.0
            @test α(:s1) ≈ SS_REF[(case, "rot_alpha_s1")] rtol = 1e-7
            @test α(:s7) ≈ SS_REF[(case, "rot_alpha_s7")] rtol = 1e-7
            @test tab.beta[findfirst(==(:s7), tab.sector)] ≈
                  SS_REF[(case, "rot_beta_s7")] rtol = 1e-7
            @test rw.negative_weight_sum ≈ SS_REF[(case, "rot_negsum")] rtol = 1e-7
            @test rw.estimate ≈ SS_REF[(case, "rot_est")] rtol = 1e-8
            rc = shift_share_iv(SS_DF, :y, :d, SS_SH, g; covariates=[:x1], weights=w,
                                shock_clusters=sc, shock_draws=draws)
            @test rc.recentered
            @test coef(rc)[1] ≈ SS_REF[(case, "rc_beta")] rtol = 1e-8
            @test rc.ri.pvalue ≈ SS_REF[(case, "ri_p_b0")] atol = 1e-12
            @test (0.0 in rc.ri.set) == (rc.ri.pvalue > 0.05)
        end
    end

    @testset "interface, printing, errors, invariance" begin
        r = shift_share_iv(SS_DF, :y, :d, SS_SH, g; covariates=[:x1])
        @test r isa ShiftShareIVEstimate && r isa CausalEstimate
        @test occursin("AKM", sprint(show, MIME"text/plain"(), r))
        @test occursin("weighted average", estimand(r))
        rs = shift_share_iv(SS_DF, :y, :d, SS_SH, g; covariates=[:x1], se=:standard)
        @test stderror(rs)[1] ≈ r.inference.se[1]
        rw = rotemberg_weights(SS_DF, :y, :d, SS_SH, g; covariates=[:x1, :ssum])
        @test occursin("Rotemberg", sprint(show, MIME"text/plain"(), rw))
        @test rw.estimate ≈ coef(r)[1] rtol = 1e-8
        @test issorted(abs.(rw.table.alpha); rev=true)
        # shuffled rows and permuted sector order give the same answers
        p = shuffle(StableRNG(301), 1:nrow(SS_DF))
        perm = shuffle(StableRNG(302), 1:40)
        r2 = shift_share_iv(SS_DF[p, :], :y, :d, SS_SH[perm], g[perm]; covariates=[:x1],
                            shock_clusters=cl[perm])
        r1 = shift_share_iv(SS_DF, :y, :d, SS_SH, g; covariates=[:x1], shock_clusters=cl)
        @test coef(r2) ≈ coef(r1) rtol = 1e-10
        @test stderror(r2) ≈ stderror(r1) rtol = 1e-8
        @test r2.inference.se ≈ r1.inference.se rtol = 1e-8
        @test_throws ArgumentError shift_share_iv(SS_DF, :y, :d, SS_SH, g; se=:foo)
        @test_throws ArgumentError shift_share_iv(SS_DF, :y, :d, SS_SH[1:1], g[1:1])
        @test_throws ArgumentError shift_share_iv(SS_DF, :y, :d, SS_SH, g;
                                                  shock_clusters=cl[1:5])
        @test_throws ArgumentError shift_share_iv(SS_DF, :y, :d, SS_SH, g;
                                                  shock_draws=draws[1:5, :])
        @test_throws ArgumentError shift_share_iv(SS_DF, :y, :d, SS_SH, g;
                                                  shock_draws=draws[:, 1:5])
        # BHJ requires the implied controls
        @test_throws ArgumentError shift_share_iv(SS_DF, :y, :d, SS_SH, g; se=:bhj,
                                                  add_share_sum=false)
        # shock-level covariates: region-level controls Σ s q are added
        q = SS_SHOCKS.q
        rq = shift_share_iv(SS_DF, :y, :d, SS_SH, g; covariates=[:x1], se=:bhj,
                            shock_covariates=reshape(q, :, 1))
        @test isfinite(coef(rq)[1])
    end

    @testset "Monte Carlo: exposure-robust coverage; RI size" begin
        R = mc_reps(500, 80)
        rng = StableRNG(303)
        n, K = 300, 40
        raw = rand(rng, n, K) .^ 8
        S = raw ./ sum(raw; dims=2)
        u = 3 .* (S * randn(rng, K)) .+ 0.5 .* randn(rng, n) .+ randn(rng, n)
        v = randn(rng, n)
        sh = [Symbol("s", k) for k in 1:K]
        cov = zeros(Bool, R, 4)
        rej_ri = zeros(Bool, R)
        for rep in 1:R
            df, gg = iv_shift_share_draw(rng, S, u, v)
            r = shift_share_iv(df, :y, :d, sh, gg)
            ci(lo, hi) = lo <= 1.0 <= hi
            cov[rep, 1] = ci(r.inference.lower[1], r.inference.upper[1])   # region HC1
            cov[rep, 2] = ci(r.inference.lower[2], r.inference.upper[2])   # AKM
            cov[rep, 3] = ci(r.inference.lower[3], r.inference.upper[3])   # BHJ
            cov[rep, 4] = 1.0 in r.akm0_set
            if rep <= R ÷ 2
                dr = hcat([gg[randperm(rng, K)] for _ in 1:99]...)
                rr = shift_share_iv(df, :y, :d, sh, gg; shock_draws=dr)
                rej_ri[rep] = DrSnow.pvalue(rr.ri.set, 1.0) <= 0.05
            end
        end
        @test mean(cov[:, 1]) < 0.92                    # conventional SEs undercover
        @test mean(cov[:, 1]) < mean(cov[:, 2])
        for j in 2:4
            @test abs(mean(cov[:, j]) - 0.95) < mc_tol(0.95, R; slack=0.05)
        end
        @test mean(rej_ri[1:(R ÷ 2)]) < 0.05 + mc_tol(0.05, R ÷ 2; slack=0.02)
    end
end
