# Monte Carlo study of dml_did_multi in a staggered design where cohort selection and
# the covariate-specific trends are non-linear in the covariates: bias, standard
# deviation, mean standard error and coverage for the simple aggregated ATT and the
# ATT(g,t), and coverage of the simultaneous event-study band, against
# did_callaway_santanna (DR with linear/logit working models, and no covariates).
#
#   julia -t 4 --project=<env with DrSnow, StableRNGs> \
#         test/validation/ml/did_multi_montecarlo.jl
#
# Writes did_multi_montecarlo.csv (the numbers quoted in docs/src/ml.md).

using DrSnow, DataFrames, Statistics, StableRNGs, Printf

const REPS = parse(Int, get(ENV, "DIDM_MC_REPS", "300"))

att(g, t) = t >= g ? 1.0 + 0.5 * (t - g) : 0.0

function dgp(rng, N; T=5)
    x1 = randn(rng, N)
    x2 = randn(rng, N)
    f = 0.8 .* (x1 .^ 2 .- 1) .+ sin.(2 .* x2)          # non-linear index
    G = zeros(Int, N)
    for i in 1:N
        s = [0.0, 0.5 * f[i] - 0.2, 0.4 * f[i], 0.5 * f[i] - 0.1]
        p = cumsum(exp.(s) ./ sum(exp.(s)))
        G[i] = (0, 3, 4, 5)[min(4, searchsortedfirst(p, rand(rng)))]
    end
    α = x1 .+ randn(rng, N)
    rows = [(id=i, t=t, g=G[i], x1=x1[i], x2=x2[i],
             y=α[i] + 0.2t + 0.3t * f[i] + (G[i] > 0 ? att(G[i], t) : 0.0) +
               randn(rng))
            for i in 1:N for t in 1:T]
    return DataFrame(rows)
end

with_coef(r, b) =
    CallawaySantAnnaEstimate((k === :coef ? b : getfield(r, k)
                              for k in fieldnames(CallawaySantAnnaEstimate))...)

function fits(df, rng)
    ft = FirstTreated(:g)
    xs = [:x1, :x2]
    return [
        "CS, no covariates" => did_callaway_santanna(df, :y, ft, :id, :t; rng=rng),
        "CS, DR (linear/logit)" =>
            did_callaway_santanna(df, :y, ft, :id, :t; covariates=xs, rng=rng),
        "DML, OLS/logit" =>
            dml_did_multi(df, :y, ft, :id, :t; covariates=xs,
                          outcome_learner=OLSLearner(),
                          propensity_learner=LogisticLearner(), rng=rng),
        "DML, random forests" =>
            dml_did_multi(df, :y, ft, :id, :t; covariates=xs,
                          outcome_learner=ForestLearner(num_trees=300),
                          propensity_learner=ForestLearner(num_trees=300), rng=rng)]
end

rows = DataFrame(N=Int[], method=String[], bias=Float64[], sd=Float64[],
                 mean_se=Float64[], coverage_simple=Float64[], coverage_cells=Float64[],
                 coverage_band=Float64[])
for N in (2000, 5000)
    acc = Dict{String,Any}()
    labels = String[]
    for rep in 1:REPS
        rng = StableRNG(hash(("didm", N, rep)))
        df = dgp(rng, N)
        for (lab, r) in fits(df, rng)
            a = get!(acc, lab) do
                push!(labels, lab)
                (est=Float64[], se=Float64[], cov=Int[], cells=Float64[], band=Int[])
            end
            truth = att.(r.groups, r.times)
            rt = with_coef(r, truth)
            s = aggregate_att(r, :simple; bootstrap=false)
            θ = coef(aggregate_att(rt, :simple; bootstrap=false))[1]
            ci = confint(s)
            push!(a.est, coef(s)[1] - θ)
            push!(a.se, stderror(s)[1])
            push!(a.cov, ci[1, 1] <= θ <= ci[1, 2])
            cc = confint(r)
            push!(a.cells, mean((cc[:, 1] .<= truth) .& (truth .<= cc[:, 2])))
            es = aggregate_att(r, :dynamic; rng=rng)
            te = coef(aggregate_att(rt, :dynamic; bootstrap=false))
            cb = confint(es; uniform=true)
            push!(a.band, all((cb[:, 1] .<= te) .& (te .<= cb[:, 2])))
        end
    end
    for lab in labels
        a = acc[lab]
        push!(rows, (N, lab, mean(a.est), std(a.est), mean(a.se), mean(a.cov),
                     mean(a.cells), mean(a.band)))
    end
end
println("T = 5 periods, $REPS replications")
show(stdout, MIME"text/plain"(), rows; allrows=true)
println()
open(joinpath(@__DIR__, "did_multi_montecarlo.csv"), "w") do io
    println(io, join(names(rows), ","))
    for r in eachrow(rows)
        println(io, join([r.N, "\"$(r.method)\"",
                          (@sprintf("%.4f", r[k]) for k in 3:8)...], ","))
    end
end
