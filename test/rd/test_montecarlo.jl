# Monte Carlo coverage and size checks. Replication counts use mc_reps(full, fast):
# DRSNOW_SLOW_TESTS=true runs the full counts.

# Calonico, Cattaneo & Titiunik (2014), Model 1 (calibrated to Lee 2008).
function rd_cct_dgp(rng, n)
    x = 2 .* [rand(rng, Beta(2, 4)) for _ in 1:n] .- 1
    mu_l(v) = 0.48 + 1.27v + 7.18v^2 + 20.21v^3 + 21.54v^4 + 7.33v^5
    mu_r(v) = 0.52 + 0.84v - 3.00v^2 + 7.99v^3 - 9.01v^4 + 3.56v^5
    y = [v < 0 ? mu_l(v) : mu_r(v) for v in x] .+ 0.1295 .* randn(rng, n)
    return DataFrame(y=y, x=x)            # true effect 0.52 - 0.48 = 0.04
end

# Tolerance for an estimated coverage/size: 4 binomial standard errors (plus slack for
# the known finite-sample distortion of the robust CI, ~1-2 points).
mc_tol(p, reps) = 4 * sqrt(p * (1 - p) / reps) + 0.02

@testset "Monte Carlo: coverage in the CCT (2014) model 1 matches rdrobust" begin
    # This high-curvature design is hard at n = 500: rdrobust 4.0.0 itself attains
    # 0.902 robust / 0.880 conventional coverage over 1,500 replications (see
    # docs/src/rd.md); DrSnow must reproduce that behaviour, not the nominal level.
    reps = mc_reps(2000, 150)
    rng = StableRNG(2014)
    cover_rb = 0
    cover_cl = 0
    for _ in 1:reps
        df = rd_cct_dgp(rng, 500)
        r = rd_estimate(df, :y, :x)
        ci = confint(r)
        cover_rb += ci[1, 1] <= 0.04 <= ci[1, 2]
        tab = rd_inference_table(r)
        cover_cl += tab.ci_lower[1] <= 0.04 <= tab.ci_upper[1]
    end
    cov = cover_rb / reps
    @test abs(cov - 0.902) < mc_tol(0.902, reps)
    @test cov >= cover_cl / reps - mc_tol(0.9, reps)
    @info "CCT model 1 coverage (n=500, $reps reps)" robust = cov conventional =
        cover_cl / reps
end

@testset "Monte Carlo: robust CI coverage ≈ nominal (smooth sharp and kink designs)" begin
    reps = mc_reps(2000, 150)
    rng = StableRNG(7)
    cover = 0
    cover_kink = 0
    for _ in 1:reps
        n = 500
        x = 2 .* rand(rng, n) .- 1
        y = 1 .+ x .- 0.5 .* x .^ 2 .+ 0.3 .* sin.(3 .* x) .+ 0.5 .* (x .>= 0) .+
            0.3 .* randn(rng, n)
        ci = confint(rd_estimate(DataFrame(y=y, x=x), :y, :x))
        cover += ci[1, 1] <= 0.5 <= ci[1, 2]
        yk = 1 .+ x .+ 0.7 .* x .* (x .>= 0) .- 0.5 .* x .^ 2 .+ 0.3 .* randn(rng, n)
        cik = confint(rd_estimate(DataFrame(y=yk, x=x), :y, :x; deriv=1))
        cover_kink += cik[1, 1] <= 0.7 <= cik[1, 2]
    end
    @test abs(cover / reps - 0.95) < mc_tol(0.95, reps)
    @test abs(cover_kink / reps - 0.95) < mc_tol(0.95, reps)
    @info "Smooth design coverage ($reps reps)" sharp = cover / reps kink =
        cover_kink / reps
end

@testset "Monte Carlo: fuzzy RD coverage and weak-IV set" begin
    reps = mc_reps(1000, 100)
    rng = StableRNG(2016)
    tau = 0.5
    cover = 0
    cover_ar = 0
    for _ in 1:reps
        n = 1000
        x = 2 .* rand(rng, n) .- 1
        u = randn(rng, n)
        d = Float64.((0.2 .+ 0.5 .* (x .>= 0) .+ 0.1 .* x .+ 0.1 .* u) .> rand(rng, n))
        y = 0.5 .+ 0.8 .* x .- 0.5 .* x .^ 2 .+ tau .* d .+ 0.3 .* (u .+ randn(rng, n))
        r = rd_estimate(DataFrame(y=y, x=x, d=d), :y, :x; treatment=:d)
        ci = confint(r)
        cover += ci[1, 1] <= tau <= ci[1, 2]
        cs = rd_weak_iv_confidence_set(r)
        cover_ar += any(iv -> iv[1] <= tau <= iv[2], cs.intervals)
    end
    @test abs(cover / reps - 0.95) < mc_tol(0.95, reps)
    @test abs(cover_ar / reps - 0.95) < mc_tol(0.95, reps)
    @info "Fuzzy RD coverage ($reps reps)" wald = cover / reps ar = cover_ar / reps
end

@testset "Monte Carlo: weak first stage, AR set keeps coverage" begin
    reps = mc_reps(1000, 100)
    rng = StableRNG(77)
    tau = 1.0
    cover_ar = 0
    cover_wald = 0
    for _ in 1:reps
        n = 1000
        x = 2 .* rand(rng, n) .- 1
        u = randn(rng, n)
        # first-stage jump of 0.08 with endogenous take-up
        d = Float64.((0.4 .+ 0.08 .* (x .>= 0) .+ 0.2 .* u) .> rand(rng, n))
        y = x .+ tau .* d .+ u
        r = rd_estimate(DataFrame(y=y, x=x, d=d), :y, :x; treatment=:d)
        ci = confint(r)
        cover_wald += ci[1, 1] <= tau <= ci[1, 2]
        cs = rd_weak_iv_confidence_set(r)
        cover_ar += any(iv -> iv[1] <= tau <= iv[2], cs.intervals)
    end
    @test cover_ar / reps > 0.95 - mc_tol(0.95, reps)
    @info "Weak first stage coverage ($reps reps)" ar = cover_ar / reps wald =
        cover_wald / reps
end

@testset "Monte Carlo: density test size" begin
    reps = mc_reps(1000, 100)
    rng = StableRNG(2020)
    rej = 0
    for _ in 1:reps
        x = 2 .* [rand(rng, Beta(2, 4)) for _ in 1:1000] .- 1
        rej += rd_density_test(x; binomial=false).pvalue < 0.05
    end
    @test rej / reps < 0.05 + mc_tol(0.05, reps)
    @info "Density test rejection rate under H0 ($reps reps)" rate = rej / reps
end

@testset "Monte Carlo: local randomization test size" begin
    reps = mc_reps(500, 60)
    rng = StableRNG(99)
    rej = 0
    for _ in 1:reps
        x = 2 .* rand(rng, 200) .- 1
        y = randn(rng, 200)                 # sharp null holds in any window
        t = rd_randomization_test(DataFrame(y=y, x=x), :y, :x; window=0.5, reps=199,
                                  rng=rng)
        rej += t.pvalue < 0.05
    end
    @test rej / reps < 0.05 + mc_tol(0.05, reps)
end
