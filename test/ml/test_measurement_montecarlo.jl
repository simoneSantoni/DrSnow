# Monte Carlo checks for inference with ML-measured variables. The key failure mode
# is differential prediction error: the ML measurement errs differently with
# treatment (or after treatment, or on one side of a cutoff). The naive plug-in is
# then biased and its intervals under-cover; the corrected estimators must cover.
# `mc_reps(full, fast)` replications (DRSNOW_SLOW_TESTS=true for full counts).

"""Share of intervals `ci` (reps × 2) covering `truth`."""
ml_meas_cover(ci, truth) = mean((ci[:, 1] .<= truth) .& (truth .<= ci[:, 2]))

@testset "Measurement Monte Carlo: RCT with differential LLM error" begin
    reps = mc_reps(1000, 80)
    rng = StableRNG(2024)
    ci = Dict(k => zeros(reps, 2) for k in (:dsl, :ppi, :ppi_lin, :naive, :dsl_logit))
    est = Dict(k => zeros(reps) for k in keys(ci))
    lam = zeros(reps)
    for b in 1:reps
        # unequal labelling probabilities depending on covariate and treatment
        df = ml_meas_rct(rng, 1000; share=0.2, delta=0.5, unequal=true)
        r = dsl_regression(df, :y; covariates=[:d, :x], prediction=:f, label_prob=:p,
                           rng=rng)
        ci[:dsl][b, :] = confint(r)[2, :]
        est[:dsl][b] = coef(r)[2]
        p = ppi_ate(df, :y, :d, :f; label_prob=:p)
        ci[:ppi][b, :] = confint(p)[1, :]
        est[:ppi][b] = coef(p)[1]
        lam[b] = p.details.lambda
        pl = ppi_ate(df, :y, :d, :f; label_prob=:p, covariates=[:x])
        ci[:ppi_lin][b, :] = confint(pl)[1, :]
        est[:ppi_lin][b] = coef(pl)[1]
        s = sqrt(r.naive_vcov[2, 2])
        c = critical_value(0.95)
        ci[:naive][b, :] = [r.naive_coef[2] - c * s, r.naive_coef[2] + c * s]
        est[:naive][b] = r.naive_coef[2]
        # binary outcome, logistic model on the true outcome; differential error
        yb = Float64.(df.ytrue .> 1.5)
        fb = clamp.(0.7 .* yb .+ 0.15 .+ 0.15 .* df.d .+ 0.1 .* randn(rng, nrow(df)),
                    0, 1)
        db = DataFrame(yb=Union{Missing,Float64}[l == 1 ? v : missing for (l, v) in
                                                 zip(df.lab, yb)],
                       fb=fb, d=df.d, x=df.x, p=df.p)
        rl = dsl_regression(db, :yb; covariates=[:d, :x], prediction=:fb,
                            family=:binomial, label_prob=:p, rng=rng)
        ci[:dsl_logit][b, :] = confint(rl)[2, :]
        est[:dsl_logit][b] = coef(rl)[2]
    end
    tol = ml_cover_tol(reps)
    @test ml_meas_cover(ci[:dsl], 1.0) >= 0.95 - tol
    @test ml_meas_cover(ci[:ppi], 1.0) >= 0.95 - tol
    @test ml_meas_cover(ci[:ppi_lin], 1.0) >= 0.95 - tol
    @test abs(mean(est[:dsl]) - 1.0) < 0.05
    @test abs(mean(est[:ppi]) - 1.0) < 0.05
    # naive plug-in: 0.8·1 + 0.5 = 1.3 on average; its intervals miss the ATE
    @test abs(mean(est[:naive]) - 1.3) < 0.05
    @test ml_meas_cover(ci[:naive], 1.0) < 0.5
    @test all(>=(0), lam)
    # logistic DSL: truth is the population logistic coefficient on d
    rngt = StableRNG(99)
    nt = 400_000
    xt = randn(rngt, nt)
    dt = Float64.(rand(rngt, nt) .< 0.5)
    yt = Float64.(1 .+ dt .+ 0.5 .* xt .+ randn(rngt, nt) .> 1.5)
    βt = coef(DrSnow.GLM.glm(hcat(ones(nt), dt, xt), yt, DrSnow.Binomial()))[2]
    @test ml_meas_cover(ci[:dsl_logit], βt) >= 0.95 - tol
end

@testset "Measurement Monte Carlo: cross-PPI mean" begin
    reps = mc_reps(1000, 80)
    rng = StableRNG(7)
    cov_ = zeros(Bool, reps)
    for b in 1:reps
        n, N = 300, 3000
        z = randn(rng, n + N, 2)
        y = 1 .+ z[:, 1] .- 0.5 .* z[:, 2] .^ 2 .+ 0.5 .* randn(rng, n + N)
        lab = DataFrame(y=y[1:n], z1=z[1:n, 1], z2=z[1:n, 2])
        un = DataFrame(z1=z[(n + 1):end, 1], z2=z[(n + 1):end, 2])
        r = cross_ppi(lab, un, :y; features=[:z1, :z2], n_folds=5, rng=rng)
        c = confint(r)
        cov_[b] = c[1, 1] <= 0.5 <= c[1, 2]      # E[y] = 1 − 0.5
    end
    @test mean(cov_) >= 0.95 - ml_cover_tol(reps)
end

@testset "Measurement Monte Carlo: DiD with treatment-induced measurement error" begin
    reps = mc_reps(500, 50)
    rng = StableRNG(31)
    cc = zeros(reps, 2)
    cn = zeros(reps, 2)
    ec = zeros(reps)
    en = zeros(reps)
    rej = zeros(Bool, reps)
    for b in 1:reps
        df = ml_meas_panel(rng, 200; delta=0.6, tau=1.0)
        r = did_with_predicted_outcome(df, :y, :D, :id, :t; prediction=:f, rng=rng)
        cc[b, :] = confint(r)[1, :]
        cn[b, :] = confint(r.naive)[1, :]
        ec[b] = coef(r)[1]
        en[b] = coef(r.naive)[1]
        rej[b] = rejects(r.bias_test)
    end
    tol = ml_cover_tol(reps)
    @test ml_meas_cover(cc, 1.0) >= 0.95 - tol
    @test abs(mean(ec) - 1.0) < 0.05
    # naive: 0.9·τ + 0.6 = 1.5
    @test abs(mean(en) - 1.5) < 0.05
    @test ml_meas_cover(cn, 1.0) < 0.2
    @test mean(rej) > 0.5
end

@testset "Measurement Monte Carlo: DiD size of the bias test (stable error)" begin
    reps = mc_reps(500, 50)
    rng = StableRNG(32)
    rej = zeros(Bool, reps)
    cc = zeros(reps, 2)
    for b in 1:reps
        # error independent of treatment and time: the naive DiD is unbiased
        df = ml_meas_panel(rng, 200; delta=0.0, tau=1.0)
        df.f = df.ytrue .+ 0.3 .+ 0.5 .* randn(rng, nrow(df))
        r = did_with_predicted_outcome(df, :y, :D, :id, :t; prediction=:f, rng=rng)
        rej[b] = rejects(r.bias_test)
        cc[b, :] = confint(r)[1, :]
    end
    @test mean(rej) <= 0.05 + ml_cover_tol(reps)
    @test ml_meas_cover(cc, 1.0) >= 0.95 - ml_cover_tol(reps)
end

@testset "Measurement Monte Carlo: sharp RD with an error jump at the cutoff" begin
    reps = mc_reps(400, 40)
    rng = StableRNG(41)
    cc = zeros(reps, 2)
    en = zeros(reps)
    for b in 1:reps
        n = 2000
        x = 2 .* rand(rng, n) .- 1
        s = Float64.(x .>= 0)
        y = 0.5 .* x .+ 0.3 .* x .^ 2 .+ 1.0 .* s .+ 0.5 .* randn(rng, n)
        f = 0.9 .* y .+ 0.4 .* s .+ 0.3 .* randn(rng, n)
        p = ifelse.(abs.(x) .< 0.3, 0.5, 0.1)
        lab = rand(rng, n) .< p
        d = DataFrame(x=x, f=f, p=p,
                      y=Union{Missing,Float64}[l ? v : missing for (l, v) in zip(lab, y)])
        r = rd_with_predicted_outcome(d, :y, :x; prediction=:f, label_prob=:p, rng=rng)
        cc[b, :] = confint(r)[1, :]
        en[b] = coef(r.naive)[1]
    end
    @test ml_meas_cover(cc, 1.0) >= 0.95 - ml_cover_tol(reps)
    @test abs(mean(en) - 1.3) < 0.1
end

@testset "Measurement Monte Carlo: regression calibration" begin
    reps = mc_reps(1000, 80)
    rng = StableRNG(51)
    cc = zeros(reps, 2)
    en = zeros(reps)
    for b in 1:reps
        n = 1000
        x = randn(rng, n)
        z = 0.5 .* x .+ randn(rng, n)
        y = 1 .+ 0.8 .* x .+ 0.3 .* z .+ randn(rng, n)
        xs = x .+ 0.8 .* randn(rng, n)
        lab = rand(rng, n) .< 0.2
        df = DataFrame(y=y, xs=xs, z=z,
                       x=Union{Missing,Float64}[l ? v : missing for (l, v) in zip(lab, x)])
        r = regression_calibration(df, :y, :xs, :x; covariates=[:z])
        cc[b, :] = confint(r)[2, :]
        en[b] = r.naive_coef[2]
    end
    @test ml_meas_cover(cc, 0.8) >= 0.95 - ml_cover_tol(reps)
    @test mean(en) < 0.6      # attenuated
end

@testset "Measurement Monte Carlo: differential-error test size and power" begin
    reps = mc_reps(1000, 100)
    rng = StableRNG(61)
    rej0 = zeros(Bool, reps)
    rej1 = zeros(Bool, reps)
    for b in 1:reps
        # additive error independent of treatment (an attenuated measure, f = a + bY,
        # would be differential: its mean error moves with the treatment effect)
        d0 = ml_meas_rct(rng, 1500; delta=0.0, unequal=true)
        d0.f = d0.ytrue .+ 0.3 .+ 0.6 .* randn(rng, nrow(d0))
        rej0[b] = rejects(differential_error_test(d0, :y, :f; by=[:d], label_prob=:p))
        d1 = ml_meas_rct(rng, 1500; delta=0.8, unequal=true)
        rej1[b] = rejects(differential_error_test(d1, :y, :f; by=[:d], label_prob=:p))
    end
    @test mean(rej0) <= 0.05 + ml_cover_tol(reps)
    @test mean(rej1) > 0.9
end
