# Full forest fits compared with the R package grf (2.6.1). Forests are random, so the
# comparison is distributional: test/validation/ml/grf_forest_reference.R fits grf
# with 20 seeds and stores the across-seed mean and standard deviation of average
# effects, standard errors, calibration and projection coefficients, variable
# importance, and CATE / regression / conditional LATE predictions (with variance
# estimates) at fixed test points. DrSnow's fits with `mc_reps(5, 1)` seeds must
# match these means within Monte Carlo error.

@testset "grf forest fits (R reference, Monte Carlo)" begin
    vdir = joinpath(@__DIR__, "..", "validation", "ml")
    d = ml_read_csv(joinpath(vdir, "grf_forest_data.csv"))
    tst = ml_read_csv(joinpath(vdir, "grf_forest_test.csv"))
    ref = ml_read_csv(joinpath(vdir, "grf_forest_reference.csv"))
    xs = [Symbol("x$j") for j in 1:6]
    S = mc_reps(5, 1)
    vals = Dict{String,Vector{Vector{Float64}}}()
    put(q, v) = push!(get!(vals, q, Vector{Float64}[]), collect(Float64, v))
    for s in 1:S
        cf = causal_forest(d, :y, :w; covariates=xs, num_trees=2000,
                           rng=StableRNG(100 + s))
        for ts in (:all, :treated, :overlap)
            a = average_treatment_effect(cf; target=ts)
            put("ate_$(ts)_est", coef(a))
            put("ate_$(ts)_se", stderror(a))
        end
        tab = test_calibration(cf).details.table
        put("cal_coef", tab.estimate)
        put("cal_se", tab.std_error)
        b = best_linear_projection(cf, [:x1, :x2])
        put("blp_coef", coef(b))
        put("blp_se", stderror(b))
        put("varimp", variable_importance(cf).importance)
        pt = predict_interval(cf, tst[:, xs])
        put("cate_test", pt.estimate)
        put("cate_test_var", pt.variance)
        po = predict_interval(cf)
        put("cate_oob_mean", [mean(po.estimate)])
        put("cate_oob_var_mean", [mean(po.variance)])
        put("Y_hat_mean_abs_resid", [mean(abs.(d.y .- cf.Y_hat))])
        put("W_hat_mean_abs_resid", [mean(abs.(d.w .- cf.W_hat))])
        rf = regression_forest(d, :y; covariates=xs, num_trees=2000,
                               rng=StableRNG(200 + s))
        rp = predict_interval(rf, tst[:, xs])
        put("reg_test", rp.estimate)
        put("reg_test_var", rp.variance)
        ivf = instrumental_forest(d, :yiv, :d, :z; covariates=xs, num_trees=2000,
                                  rng=StableRNG(300 + s))
        ip = predict_interval(ivf, tst[:, xs])
        put("iv_test", ip.estimate)
        put("iv_test_var", ip.variance)
        la = average_treatment_effect(ivf)
        put("late_est", coef(la))
        put("late_se", stderror(la))
    end
    worst = 0.0
    for (q, v) in vals
        M = reduce(hcat, v)                      # k × S
        sub = ref[ref.quantity .== q, :]
        sub = sub[sortperm(sub.index), :]
        @test nrow(sub) == size(M, 1)
        m_jl = vec(mean(M; dims=2))
        # Monte Carlo variance of the difference of the two seed averages
        v_mc = sub.sd .^ 2 .* (1 / S .+ 1 ./ sub.S)
        if size(M, 1) <= 6
            z = (m_jl .- sub.mean) ./ sqrt.(max.(v_mc, 1e-20))
            worst = max(worst, maximum(abs, z))
            @test all(abs.(z) .< 4.5)
        else
            # mean squared standardized difference over the test points (≈ 1)
            ratio = mean((m_jl .- sub.mean) .^ 2 ./ v_mc)
            worst = max(worst, ratio)
            @test ratio < 2.5
            # prediction curves (not the noisier variance curves) track grf's
            endswith(q, "_var") || @test cor(m_jl, sub.mean) > 0.95
        end
    end
    @info "grf forest parity: largest |z| (scalars) / MSE ratio (curves)" worst
end
