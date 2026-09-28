# Tests of rd_flex: RD with cross-fitted machine-learning covariate adjustment.

"""Sharp RD with a non-linear covariate effect g(z) and effect τ at the cutoff."""
function rd_flex_dgp(rng, n; τ=0.5, fuzzy=false)
    x = 2 .* rand(rng, n) .- 1
    Z = randn(rng, n, 4)
    gz = sin.(2 .* Z[:, 1]) .+ 0.5 .* Z[:, 2] .^ 2 .+ 0.5 .* Z[:, 3] .+
         0.3 .* Z[:, 1] .* Z[:, 4]
    side = x .>= 0
    v = randn(rng, n)
    d = fuzzy ? Float64.(0.2 .+ 0.5 .* side .+ 0.3 .* Z[:, 1] .+ 0.6 .* v .> 0.5) :
        Float64.(side)
    y = τ .* d .+ 0.8 .* x .- 0.3 .* x .^ 2 .+ gz .+ (fuzzy ? 0.3 .* v : 0.0) .+
        0.5 .* randn(rng, n)
    return DataFrame(x=x, y=y, d=d, z1=Z[:, 1], z2=Z[:, 2], z3=Z[:, 3], z4=Z[:, 4])
end

const RD_FLEX_Z = [:z1, :z2, :z3, :z4]
const RD_FLEX_LIN = (outcome_learner=OLSLearner(), treatment_learner=LogisticLearner())

@testset "DoubleML RDFlex parity (Python reference)" begin
    df = rd_read_csv(joinpath(RD_VALIDATION_DIR, "flex_data.csv"))
    for c in names(df)
        df[!, c] = Float64.(df[!, c])
    end
    ref = rd_read_csv(joinpath(RD_VALIDATION_DIR, "flex_reference.csv"))
    adj = rd_read_csv(joinpath(RD_VALIDATION_DIR, "flex_adjustment.csv"))
    kw = (covariates=RD_FLEX_Z, RD_FLEX_LIN...)
    cases = [
        "sharp" => rd_flex(df, :y_sharp, :x; kw..., folds=:fold_sharp),
        "sharp_score" => rd_flex(df, :y_sharp, :x; kw..., folds=:fold_sharp,
                                 fs_specification=:cutoff_and_score),
        "sharp_interacted" => rd_flex(df, :y_sharp, :x; kw..., folds=:fold_sharp,
                                      fs_specification=:interacted_cutoff_and_score),
        "sharp_iter1" => rd_flex(df, :y_sharp, :x; kw..., folds=:fold_sharp,
                                 n_iterations=1),
        "sharp_uniform" => rd_flex(df, :y_sharp, :x; kw..., folds=:fold_sharp,
                                   fs_kernel=:uniform),
        "fuzzy" => rd_flex(df, :y_fuzzy, :x; kw..., folds=:fold_fuzzy, treatment=:d)]
    for (case, r) in cases
        rr = ref[ref.case .== case, :]
        fz = case == "fuzzy"
        # the logistic first stage (sklearn, tol = 1e-12) limits the fuzzy agreement
        tol = fz ? 1e-7 : 1e-9
        @test r.design === (fz ? :fuzzy : :sharp)
        @test r.h_fs ≈ rr.h_fs[1] rtol = 1e-9
        @test r.h[1] ≈ rr.h[1] rtol = tol
        @test r.b[1] ≈ rr.b[1] rtol = tol
        @test r.tau_conventional ≈ rr.coef_conv[1] rtol = tol
        @test r.tau_bias_corrected ≈ rr.coef_rb[1] rtol = tol
        @test coef(r)[1] == r.tau_conventional
        @test r.se_conventional ≈ rr.se_conv[1] rtol = tol
        @test stderror(r)[1] ≈ rr.se_rb[1] rtol = tol
        @test (r.fits[1].n_h_left, r.fits[1].n_h_right) ==
              (Int(rr.n_h_left[1]), Int(rr.n_h_right[1]))
        yv = fz ? df.y_fuzzy : df.y_sharp
        @test yv .- r.eta_y[:, 1] ≈ adj[!, case * "_my"] atol = 1e-9
        fz && @test df.d .- r.eta_d[:, 1] ≈ adj[!, case * "_md"] atol = 1e-6
    end
end

@testset "interface" begin
    df = rd_flex_dgp(StableRNG(11), 1200)
    r = rd_flex(df, :y, :x; covariates=RD_FLEX_Z, RD_FLEX_LIN..., rng=StableRNG(1))
    @test r isa RDFlexEstimate && nobs(r) == 1200
    @test r.learners == [:ml_g => "OLSLearner"] && r.eta_d === nothing
    @test size(r.eta_y) == (1200, 1) && size(r.folds) == (1200, 1)
    @test occursin("sharp", estimand(r))
    tab = rd_inference_table(r)
    @test tab.estimate[3] == r.tau_bias_corrected && tab.se[3] == stderror(r)[1]
    @test tab.estimate[1] == coef(r)[1]
    @test confint(r)[1, 1] < r.tau_bias_corrected < confint(r)[1, 2]
    @test sum(confint(r)) / 2 ≈ r.tau_bias_corrected
    @test occursin("Robust", sprint(show, MIME"text/plain"(), r))
    # results do not depend on the row order of the data
    perm = randperm(StableRNG(2), nrow(df))
    r2 = rd_flex(df[perm, :], :y, :x; covariates=RD_FLEX_Z, RD_FLEX_LIN...,
                 rng=StableRNG(1))
    @test coef(r2) ≈ coef(r) rtol = 1e-10
    @test stderror(r2) ≈ stderror(r) rtol = 1e-10
    @test r2.eta_y[invperm(perm), 1] ≈ r.eta_y[:, 1] rtol = 1e-10
    # repetitions: medians and the variance-scale aggregation rule
    r3 = rd_flex(df, :y, :x; covariates=RD_FLEX_Z, RD_FLEX_LIN..., n_rep=3,
                 rng=StableRNG(3))
    @test size(r3.all_coef) == (3, 3) && length(r3.fits) == 3
    @test coef(r3)[1] == median(r3.all_coef[1, :])
    @test r3.tau_bias_corrected == median(r3.all_coef[3, :])
    θ = r3.tau_bias_corrected
    @test stderror(r3)[1] ≈ sqrt(median(r3.all_se[3, :] .^ 2 .+
                                        (r3.all_coef[3, :] .- θ) .^ 2))
    # options forwarded to rd_estimate
    r4 = rd_flex(df, :y, :x; covariates=RD_FLEX_Z, RD_FLEX_LIN..., rng=StableRNG(1),
                 p=2, kernel=:epanechnikov, vce=:hc1)
    @test r4.fits[1].p == 2 && r4.fits[1].kernel === :epanechnikov
    @test r4.fits[1].vce === :hc1
    r5 = rd_flex(df, :y, :x; covariates=RD_FLEX_Z, RD_FLEX_LIN..., rng=StableRNG(1),
                 h=0.3, h_fs=0.4)
    @test r5.h == [0.3] && r5.h_fs == 0.4 && r5.fits[1].h_left == 0.3
    # equivalent to rd_estimate on the adjusted outcome
    adj = DataFrame(y=df.y .- r.eta_y[:, 1], x=df.x)
    e = rd_estimate(adj, :y, :x; h=r.h[1], b=r.b[1])
    @test coef(e) ≈ coef(r) && stderror(e) ≈ stderror(r)
    # clusters: folds grouped by cluster, cluster-robust RD variance
    df.g = mod1.(1:nrow(df), 60)
    rc = rd_flex(df, :y, :x; covariates=RD_FLEX_Z, RD_FLEX_LIN..., cluster=:g,
                 rng=StableRNG(4))
    @test rc.n_clusters == (60, 60) && rc.fits[1].vce === :cr1
    @test all(length(unique(rc.folds[df.g .== k, 1])) == 1 for k in 1:60)
    # fuzzy design with a flexible learner
    fz = rd_flex_dgp(StableRNG(12), 1500; fuzzy=true, τ=1.0)
    rf = rd_flex(fz, :y, :x; covariates=RD_FLEX_Z, treatment=:d,
                 outcome_learner=ForestLearner(num_trees=100),
                 treatment_learner=ForestLearner(num_trees=100), rng=StableRNG(5))
    @test rf.design === :fuzzy && rf.fits[1].first_stage !== nothing
    @test abs(coef(rf)[1] - 1.0) < 4 * stderror(rf)[1]
    @test [first(p) for p in rf.learners] == [:ml_g, :ml_m]
    # missing values are dropped (folds given per row of the data)
    dm = allowmissing(df)
    dm.z1[5] = missing
    rm = rd_flex(dm, :y, :x; covariates=RD_FLEX_Z, RD_FLEX_LIN...,
                 folds=mod1.(1:nrow(dm), 5))
    @test nobs(rm) == nrow(df) - 1
end

@testset "errors" begin
    df = rd_flex_dgp(StableRNG(13), 600)
    kw = (covariates=RD_FLEX_Z, RD_FLEX_LIN...)
    @test_throws ArgumentError rd_flex(df, :y, :x; covariates=Symbol[])
    @test_throws ArgumentError rd_flex(df, :y, :x; kw..., fs_specification=:score)
    @test_throws ArgumentError rd_flex(df, :y, :x; kw..., fs_kernel=:gaussian)
    @test_throws ArgumentError rd_flex(df, :y, :x; kw..., cutoff=2.0)
    @test_throws ArgumentError rd_flex(df, :y, :x; kw..., n_iterations=0)
    @test_throws ArgumentError rd_flex(df, :y, :x; kw..., h_fs=-1.0)
    @test_throws ArgumentError rd_flex(df, :y, :x; kw..., n_folds=1)
    @test_throws ArgumentError rd_flex(df, :y, :x; kw..., treatment=:z1)
    @test_throws ArgumentError rd_flex(df, :y, :x; kw..., covariates=[:nope])
    @test_throws DimensionMismatch rd_flex(df, :y, :x; kw..., folds=[1, 2, 1])
    # a first-stage bandwidth so small that a training fold has no weight
    @test_throws ArgumentError rd_flex(df, :y, :x; kw..., h_fs=1e-4)
    # take-up higher on the left of the cutoff
    fz = rd_flex_dgp(StableRNG(14), 800; fuzzy=true)
    fz.d = 1 .- fz.d
    @test_logs (:warn, r"higher on the left") match_mode = :any rd_flex(
        fz, :y, :x; kw..., treatment=:d, rng=StableRNG(1))
end

"""Sharp RD in which the covariates act mostly non-linearly (effect 0.5)."""
function rd_flex_dgp_nl(rng, n)
    x = 2 .* rand(rng, n) .- 1
    Z = randn(rng, n, 3)
    y = 0.5 .* (x .>= 0) .+ 0.8 .* x .+ 1.5 .* sin.(2 .* Z[:, 1]) .+
        (Z[:, 2] .^ 2 .- 1) .+ 0.5 .* Z[:, 3] .+ 0.5 .* randn(rng, n)
    return DataFrame(x=x, y=y, z1=Z[:, 1], z2=Z[:, 2], z3=Z[:, 3])
end

@testset "Monte Carlo: coverage and variance reduction" begin
    reps = mc_reps(1000, 120)
    zs = [:z1, :z2, :z3]
    names_ = [:none, :linear, :flex_ols, :flex_forest]
    est = Dict(k => Float64[] for k in names_)
    se = Dict(k => Float64[] for k in names_)
    cover = Dict(k => 0 for k in names_)
    for rep in 1:reps
        rng = StableRNG(52_000 + rep)
        df = rd_flex_dgp_nl(rng, 1000)
        fits = (none=rd_estimate(df, :y, :x),
                linear=rd_estimate(df, :y, :x; covariates=zs),
                flex_ols=rd_flex(df, :y, :x; covariates=zs,
                                 outcome_learner=OLSLearner(), rng=rng),
                flex_forest=rd_flex(df, :y, :x; covariates=zs,
                                    outcome_learner=ForestLearner(num_trees=200),
                                    rng=rng))
        for k in names_
            r = fits[k]
            push!(est[k], coef(r)[1])
            push!(se[k], stderror(r)[1])
            ci = confint(r)
            cover[k] += ci[1, 1] <= 0.5 <= ci[1, 2]
        end
    end
    sds = Dict(k => std(est[k]) for k in names_)
    @info "rd_flex Monte Carlo (n = 1000, $reps reps)" coverage =
        [k => cover[k] / reps for k in names_] sd = [k => sds[k] for k in names_]
    for k in names_
        @test abs(cover[k] / reps - 0.95) < mc_tol(0.95, reps)
        @test 0.8 < mean(se[k]) / sds[k] < 1.25
    end
    # the flexible adjustment removes most of the covariate-driven variance
    @test sds[:flex_forest] < 0.8 * sds[:linear]
    @test sds[:flex_forest] < 0.8 * sds[:none]
    @test sds[:flex_ols] < 1.1 * sds[:linear]
end
