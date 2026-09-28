# Sequential A/B test: continuous monitoring with anytime-valid inference, and a
# group-sequential design for the same experiment with three scheduled looks.
#
# Run from the repository root:  julia --project=. examples/sequential_ab_test_demo.jl

using DrSnow, DataFrames, Random, Statistics

rng = Xoshiro(2026)

# --- A simulated online experiment -------------------------------------------------
# 4000 users arrive one at a time and are randomized 50/50; revenue per user depends
# on two pre-treatment covariates; the treatment raises it by 0.15 on average.
n = 4000
df = DataFrame(arrival=1:n, x1=randn(rng, n), x2=randn(rng, n))
df.d = rand(rng, n) .< 0.5
df.y = 1 .+ 0.8 .* df.x1 .+ 0.4 .* df.x2 .+ 0.15 .* df.d .+ randn(rng, n)
df.converted = Float64.(rand(rng, n) .< 0.10 .+ 0.02 .* df.d)

# --- 1. Why naive peeking fails ---------------------------------------------------
# Under no effect, a 95% t-test checked after every user rejects far more than 5%.
function naive_peeking_rejects(rng; T=600)
    d = rand(rng, T) .< 0.5
    y = randn(rng, T)
    for i in 20:T
        a, b = y[1:i][d[1:i]], y[1:i][.!d[1:i]]
        se = sqrt(var(a) / length(a) + var(b) / length(b))
        abs(mean(a) - mean(b)) > 1.96 * se && return true
    end
    return false
end
rate = mean(naive_peeking_rejects(rng) for _ in 1:300)
println("Naive continuous peeking: false-positive rate ≈ ", round(rate; digits=3),
        " (nominal 0.05)")

# --- 2. Anytime-valid confidence sequence for the ATE ------------------------------
# Known randomization probability 0.5; difference-in-means type estimator.
cs = confseq_ate(df, :y, :d; propensity=0.5, order=:arrival, t_opt=n)
display(cs)

# Regression-adjusted (AIPW with OLS outcome models fitted on past users only).
cs_adj = confseq_ate(df, :y, :d; propensity=0.5, order=:arrival, t_opt=n,
                     covariates=[:x1, :x2], outcome_learner=OLSLearner(),
                     refit_every=200)
display(cs_adj)
w(c) = c.upper[end] - c.lower[end]
println("Width at n = $n: unadjusted ", round(w(cs); digits=3), ", adjusted ",
        round(w(cs_adj); digits=3))

s = stopping(cs_adj)
if s.rejected
    println("The adjusted CS first excluded 0 after n = ", s.n, " users (estimate ",
            round(s.estimate; digits=3), "). Stopping there is valid.")
else
    println("The adjusted CS has not excluded 0 yet.")
end

# --- 3. Streaming: the same analysis user by user ----------------------------------
m = ATEMonitor(; propensity=0.5, t_opt=n)
for i in 1:n
    fit!(m, df.y[i], df.d[i])
    st = snapshot(m)
    if st.lower > 0
        println("Streaming monitor: 0 excluded at n = ", st.n, ", CS [",
                round(st.lower; digits=3), ", ", round(st.upper; digits=3), "]")
        break
    end
end

# --- 4. Conversions: mSPRT with an always-valid p-value ----------------------------
t = msprt_test(df, :converted, :d; outcome_type=:binary, order=:arrival)
display(t)

# --- 5. Bounded outcome: betting confidence sequence for the control conversion rate
ctrl = df.converted[.!df.d]
cb = confseq_mean(ctrl; method=:betting, bounds=(0, 1))
println("Control conversion rate: ", round(cb.estimate[end]; digits=4),
        ", betting CS ", round.(vec(confint(cb)); digits=4))

# --- 6. Group-sequential alternative: three scheduled looks ------------------------
# A fixed-sample design would need n_fixed users; OBF-type spending with a
# non-binding futility bound inflates it slightly.
n_fixed = 3000
design = gs_design(; k=3, alpha=0.025, beta=0.1, efficacy=OBFSpending(),
                   futility=HSDSpending(-2), n_fixed=n_fixed)
display(design)

looks = round.(Int, design.n)
est = Float64[]; ses = Float64[]
for nk in looks
    sub = df[1:min(nk, n), :]
    a, b = sub.y[sub.d], sub.y[.!sub.d]
    push!(est, mean(a) - mean(b))
    push!(ses, sqrt(var(a) / length(a) + var(b) / length(b)))
    an = gs_analysis(design, est, ses)
    an.decision === :continue || break
end
analysis = gs_analysis(design, est, ses)
display(analysis)
