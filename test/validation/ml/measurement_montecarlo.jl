# Monte Carlo study of the estimators for ML-measured variables (the table in
# docs/src/ml.md). Every design has differential prediction error: the ML measure
# errs differently with treatment, after treatment, or on one side of a cutoff, so
# the naive plug-in is biased. Prints bias, empirical SD, mean SE and 95% coverage.
#
# Usage: julia --project=<env with DrSnow, DataFrames, StableRNGs> measurement_montecarlo.jl

using DrSnow, DataFrames, StableRNGs, Statistics, Printf, Random

const REPS = parse(Int, get(ENV, "MC_REPS", "1000"))

miss(lab, v) = Union{Missing,Float64}[l ? x : missing for (l, x) in zip(lab, v)]

function rct(rng, n)
    x = randn(rng, n)
    d = Float64.(rand(rng, n) .< 0.5)
    y = 1 .+ d .+ 0.5 .* x .+ randn(rng, n)
    f = 0.3 .+ 0.8 .* y .+ 0.5 .* d .+ 0.6 .* randn(rng, n)
    p = clamp.(0.25 .* (0.5 .+ (x .> 0) .+ 0.5 .* d), 0.02, 1.0)
    lab = rand(rng, n) .< p
    return DataFrame(y=miss(lab, y), f=f, d=d, x=x, p=p, ytrue=y)
end

function panel(rng, N; T=4)
    id = repeat(1:N, inner=T)
    t = repeat(1:T, outer=N)
    tr = repeat(Float64.(rand(rng, N) .< 0.5), inner=T)
    D = tr .* (t .>= 3)
    y = repeat(randn(rng, N), inner=T) .+ 0.3 .* t .+ D .+ randn(rng, N * T)
    f = 0.2 .+ 0.9 .* y .+ 0.6 .* D .+ 0.5 .* randn(rng, N * T)
    lab = rand(rng, N * T) .< 0.3
    return DataFrame(id=id, t=t, D=D, g=ifelse.(tr .== 1, 3, 0), y=miss(lab, y), f=f,
                     ytrue=y)
end

function rd(rng, n)
    x = 2 .* rand(rng, n) .- 1
    s = Float64.(x .>= 0)
    y = 0.5 .* x .+ 0.3 .* x .^ 2 .+ s .+ 0.5 .* randn(rng, n)
    f = 0.9 .* y .+ 0.4 .* s .+ 0.3 .* randn(rng, n)
    p = ifelse.(abs.(x) .< 0.3, 0.5, 0.1)
    lab = rand(rng, n) .< p
    return DataFrame(x=x, f=f, p=p, y=miss(lab, y))
end

const ROWS = Tuple{String,Float64,Vector{Float64},Vector{Float64},Vector{Bool}}[]

function record(name, truth, est, se, cover)
    push!(ROWS, (name, truth, est, se, cover))
    @printf("%-44s bias %+7.4f  sd %.4f  mean se %.4f  coverage %.3f\n", name,
            mean(est) - truth, std(est), mean(se), mean(cover))
    flush(stdout)
end

function run_mc(name, truth, reps, seed, f)
    rng = StableRNG(seed)
    est = zeros(reps)
    se = zeros(reps)
    cov_ = zeros(Bool, reps)
    for b in 1:reps
        e, s, lo, hi = f(rng)
        est[b], se[b], cov_[b] = e, s, lo <= truth <= hi
    end
    record(name, truth, est, se, cov_)
end

headline(r) = (coef(r)[1], stderror(r)[1], confint(r)[1, 1], confint(r)[1, 2])
function naive_ci(b, s)
    c = critical_value(0.95)
    return (b, s, b - c * s, b + c * s)
end

println("Monte Carlo, $(REPS) replications (DiD/RD: $(REPS ÷ 2))")
println("RCT, n = 1000, labels ~25% with probabilities depending on x and d; ATE = 1")
run_mc("  naive (LLM outcome as truth)", 1.0, REPS, 1, rng -> begin
    r = dsl_regression(rct(rng, 1000), :y; covariates=[:d, :x], prediction=:f,
                       label_prob=:p, rng=rng)
    naive_ci(r.naive_coef[2], sqrt(r.naive_vcov[2, 2]))
end)
run_mc("  DSL linear regression (OLS recalibration)", 1.0, REPS, 1, rng -> begin
    r = dsl_regression(rct(rng, 1000), :y; covariates=[:d, :x], prediction=:f,
                       label_prob=:p, rng=rng)
    (coef(r)[2], stderror(r)[2], confint(r)[2, 1], confint(r)[2, 2])
end)
run_mc("  PPI++ ATE (difference in means)", 1.0, REPS, 1,
       rng -> headline(ppi_ate(rct(rng, 1000), :y, :d, :f; label_prob=:p)))
run_mc("  PPI++ ATE (Lin adjustment)", 1.0, REPS, 1,
       rng -> headline(ppi_ate(rct(rng, 1000), :y, :d, :f; label_prob=:p,
                               covariates=[:x])))
run_mc("  labelled rows only (IPW, λ = 0)", 1.0, REPS, 1,
       rng -> headline(ppi_ate(rct(rng, 1000), :y, :d, :f; label_prob=:p, lambda=0)))

println("Cross-PPI mean, n = 300 labelled, N = 3000 unlabelled; mean = 0.5")
run_mc("  cross-PPI++ (OLS on features, 5 folds)", 0.5, REPS, 2, rng -> begin
    n, N = 300, 3000
    z = randn(rng, n + N, 2)
    y = 1 .+ z[:, 1] .- 0.5 .* z[:, 2] .^ 2 .+ 0.5 .* randn(rng, n + N)
    lab = DataFrame(y=y[1:n], z1=z[1:n, 1], z2=z[1:n, 2])
    un = DataFrame(z1=z[(n + 1):end, 1], z2=z[(n + 1):end, 2])
    headline(cross_ppi(lab, un, :y; features=[:z1, :z2], n_folds=5, rng=rng))
end)

R2 = REPS ÷ 2
println("DiD, 200 units × 4 periods, treatment in periods 3-4 for half the units, ",
        "30% of unit-periods labelled; LLM error shifts by +0.6 when treated; ATT = 1")
run_mc("  naive TWFE", 1.0, R2, 3,
       rng -> headline(did_with_predicted_outcome(panel(rng, 200), :y, :D, :id, :t;
                                                  prediction=:f, rng=rng).naive))
run_mc("  corrected TWFE (design-based)", 1.0, R2, 3,
       rng -> headline(did_with_predicted_outcome(panel(rng, 200), :y, :D, :id, :t;
                                                  prediction=:f, rng=rng)))
run_mc("  corrected Callaway-Sant'Anna (simple ATT)", 1.0, R2, 4, rng -> begin
    r = did_with_predicted_outcome(panel(rng, 200), :y, FirstTreated(:g), :id, :t;
                                   prediction=:f, estimator=:cs, rng=rng)
    a = aggregate_att(r.corrected, :simple; bootstrap=false)
    headline(a)
end)
run_mc("  corrected TWFE, raw prediction as ĝ", 1.0, R2, 5,
       rng -> headline(did_with_predicted_outcome(panel(rng, 200), :y, :D, :id, :t;
                                                  prediction=:f, learner=nothing)))
run_mc("  same, known π = 0.3 (normalize_prob=false)", 1.0, R2, 5, rng -> begin
    d = panel(rng, 200)
    d.p = fill(0.3, nrow(d))
    headline(did_with_predicted_outcome(d, :y, :D, :id, :t; prediction=:f,
                                        learner=nothing, label_prob=:p,
                                        normalize_prob=false))
end)
println("DiD, labels only in periods 1-2 (stable-error mode; assumption violated by ",
        "the treatment-induced error)")
run_mc("  stable-error calibration (violated)", 1.0, R2 ÷ 5, 6, rng -> begin
    d = panel(rng, 200)
    d.y = [t <= 2 ? v : missing for (t, v) in zip(d.t, d.ytrue)]
    headline(did_with_predicted_outcome(d, :y, :D, :id, :t; prediction=:f,
                                        assume_stable_error=true, bootstrap_reps=99,
                                        rng=rng))
end)
run_mc("  stable-error calibration (holds: no D effect)", 1.0, R2 ÷ 5, 7, rng -> begin
    d = panel(rng, 200)
    d.f = 0.2 .+ 0.9 .* d.ytrue .+ 0.5 .* randn(rng, nrow(d))
    d.y = [t <= 2 ? v : missing for (t, v) in zip(d.t, d.ytrue)]
    headline(did_with_predicted_outcome(d, :y, :D, :id, :t; prediction=:f,
                                        assume_stable_error=true, bootstrap_reps=99,
                                        rng=rng))
end)

println("Sharp RD, n = 2000, labels 50% within 0.3 of the cutoff, 10% elsewhere; ",
        "error jumps by 0.4 at the cutoff; effect = 1")
run_mc("  naive RD (robust bias-corrected)", 1.0, R2, 8,
       rng -> headline(rd_with_predicted_outcome(rd(rng, 2000), :y, :x; prediction=:f,
                                                 label_prob=:p, rng=rng).naive))
run_mc("  corrected RD (design-based)", 1.0, R2, 8,
       rng -> headline(rd_with_predicted_outcome(rd(rng, 2000), :y, :x; prediction=:f,
                                                 label_prob=:p, rng=rng)))

println("Regression calibration, n = 1000, 20% validation, reliability 0.61; β = 0.8")
run_mc("  naive (X* as regressor)", 0.8, REPS, 9, rng -> begin
    n = 1000
    x = randn(rng, n)
    z = 0.5 .* x .+ randn(rng, n)
    y = 1 .+ 0.8 .* x .+ 0.3 .* z .+ randn(rng, n)
    xs = x .+ 0.8 .* randn(rng, n)
    lab = rand(rng, n) .< 0.2
    r = regression_calibration(DataFrame(y=y, xs=xs, z=z, x=miss(lab, x)), :y, :xs, :x;
                               covariates=[:z])
    naive_ci(r.naive_coef[2], sqrt(r.naive_vcov[2, 2]))
end)
run_mc("  regression calibration", 0.8, REPS, 9, rng -> begin
    n = 1000
    x = randn(rng, n)
    z = 0.5 .* x .+ randn(rng, n)
    y = 1 .+ 0.8 .* x .+ 0.3 .* z .+ randn(rng, n)
    xs = x .+ 0.8 .* randn(rng, n)
    lab = rand(rng, n) .< 0.2
    r = regression_calibration(DataFrame(y=y, xs=xs, z=z, x=miss(lab, x)), :y, :xs, :x;
                               covariates=[:z])
    (coef(r)[2], stderror(r)[2], confint(r)[2, 1], confint(r)[2, 2])
end)
