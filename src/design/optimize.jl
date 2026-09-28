# Surrogate-model design optimization (Zimmer & Debelak 2025; Zimmer et al. 2024):
# simulate power at a few candidate designs, fit a smooth monotone model of power on the
# design parameters, pick the cheapest design whose predicted power reaches the target,
# and spend further simulations near that boundary.

"""
    DesignOptimization

The cheapest design predicted to reach a target power, with the simulations and the
surrogate model that led to it; the result of [`optimize_design`](@ref).

The chosen design is the output of a stochastic search: its predicted power comes
from a surrogate model fitted to a finite number of simulations, so it carries
simulation error (`predicted_se`) and model error that the standard error does not
capture. The optional `verification` run, an independent simulation at the chosen
design, is the check that should be reported alongside the choice.

# Fields
- `best::NamedTuple`: the chosen design parameters.
- `cost::Float64`: cost of the chosen design.
- `predicted_power::Float64`: surrogate prediction of power at `best`.
- `predicted_se::Float64`: its standard error (delta method; `NaN` for the isotonic
  surrogate).
- `target::Float64`: target power.
- `reached::Bool`: whether any candidate is predicted to reach the target (otherwise
  `best` is the candidate with the highest predicted power).
- `estimator::String`: label of the estimator whose power was optimized.
- `evaluations::DataFrame`: one row per simulation batch, with the parameters,
  `round`, `sims`, `rejections`, `power` and `cost`.
- `predictions::DataFrame`: surrogate predictions at every candidate (parameters,
  `cost`, `power`, `se`).
- `surrogate::Symbol`: the surrogate model (`:probit`, `:logistic` or `:isotonic`).
- `transform`: the feature transform (a symbol or a function).
- `total_sims::Int`: simulations used by the search (excluding verification).
- `verification::Union{Nothing,NamedTuple}`: independent simulation at `best`, with
  `power`, `mc_se` and `sims`, when `verify_sims > 0`.
"""
struct DesignOptimization
    best::NamedTuple
    cost::Float64
    predicted_power::Float64
    predicted_se::Float64
    target::Float64
    reached::Bool
    estimator::String
    evaluations::DataFrame
    predictions::DataFrame
    surrogate::Symbol
    transform::Any
    total_sims::Int
    verification::Union{Nothing,NamedTuple}
end

function Base.show(io::IO, ::MIME"text/plain", r::DesignOptimization)
    println(io, "Surrogate design optimization (", r.surrogate, " surrogate, ",
            r.transform, " transform), estimator: ", r.estimator)
    @printf(io, "Target power %.3g %s\n", r.target,
            r.reached ? "reached" : "not reached by any candidate")
    println(io, "Chosen design: ", r.best, " (cost ", @sprintf("%.6g", r.cost), ")")
    se = isnan(r.predicted_se) ? "" : @sprintf(" (se %.3g)", r.predicted_se)
    @printf(io, "Predicted power: %.4f%s\n", r.predicted_power, se)
    @printf(io, "Simulations used: %d at %d design points\n", r.total_sims,
            length(unique(eachrow(r.evaluations[!, collect(keys(r.best))]))))
    v = r.verification
    v === nothing ||
        @printf(io, "Verification: simulated power %.4f (MC se %.3g, %d simulations)",
                v.power, v.mc_se, v.sims)
end

Base.show(io::IO, r::DesignOptimization) =
    print(io, "DesignOptimization(", r.best, ", predicted power ",
          @sprintf("%.3f", r.predicted_power), ")")

function _des_transform(x::Real, t::Symbol)
    t === :identity && return float(x)
    x > 0 || throw(ArgumentError("transform $t needs positive design parameters"))
    return t === :sqrt ? sqrt(float(x)) : log(float(x))
end

function _des_features(p::NamedTuple, names::Vector{Symbol}, t)
    if t isa Function
        f = t(p)
        (f isa AbstractVector{<:Real} || f isa Tuple) ||
            throw(ArgumentError("the transform function must return a numeric vector"))
        return vcat(1.0, Float64[x for x in f])
    end
    return vcat(1.0, [_des_transform(p[k], t) for k in names])
end

# Binomial GLM surrogate (probit or logit link) with 0.5 pseudo-successes and failures
# per design point, so that points with 0 or all rejections do not cause separation.
# Fitted by iteratively reweighted least squares on the grouped proportions (Fisher
# scoring; a few parameters and design points), which avoids depending on the
# weight-keyword API of a particular GLM.jl version. Returns (β, Var(β), link).
function _des_fit_glm(F::Matrix{Float64}, k::Vector{Float64}, m::Vector{Float64},
                      link::Symbol)
    y = (k .+ 0.5) ./ (m .+ 1.0)
    w = m .+ 1.0
    lk = link === :probit ? GLM.ProbitLink() : GLM.LogitLink()
    β = zeros(size(F, 2))
    η = F * β
    V = Matrix{Float64}(I, size(F, 2), size(F, 2))
    converged = false
    for _ in 1:200
        μ = clamp.(GLM.linkinv.(Ref(lk), η), 1e-12, 1 - 1e-12)
        dμ = max.(GLM.mueta.(Ref(lk), η), 1e-300)
        W = w .* dμ .^ 2 ./ (μ .* (1 .- μ))
        z = η .+ (y .- μ) ./ dμ
        XtW = F' .* W'
        A = Symmetric(XtW * F)
        βn = A \ (XtW * z)
        V = inv(A)
        done = maximum(abs.(βn .- β)) <= 1e-10 * max(1.0, maximum(abs.(βn)))
        β = βn
        η = F * β
        if done
            converged = true
            break
        end
    end
    converged || throw(ArgumentError("optimize_design: the surrogate model did not " *
                                     "converge; try another transform or surrogate"))
    μ = clamp.(GLM.linkinv.(Ref(lk), η), 1e-12, 1 - 1e-12)
    dμ = GLM.mueta.(Ref(lk), η)
    V = inv(Symmetric((F' .* (w .* dμ .^ 2 ./ (μ .* (1 .- μ)))') * F))
    return β, Matrix(V), lk
end

function _des_predict_glm(β, V, lk, F)
    η = F * β
    p = GLM.linkinv.(Ref(lk), η)
    dμ = GLM.mueta.(Ref(lk), η)
    seη = sqrt.(max.(vec(sum((F * V) .* F; dims=2)), 0.0))
    return p, abs.(dμ) .* seη
end

# Weighted pool-adjacent-violators (increasing), on points sorted by x.
function _des_pava(y::Vector{Float64}, w::Vector{Float64})
    vals = Float64[]
    wts = Float64[]
    cnt = Int[]
    for (yi, wi) in zip(y, w)
        push!(vals, yi); push!(wts, wi); push!(cnt, 1)
        while length(vals) >= 2 && vals[end - 1] > vals[end]
            v = (vals[end - 1] * wts[end - 1] + vals[end] * wts[end]) /
                (wts[end - 1] + wts[end])
            wsum = wts[end - 1] + wts[end]
            c = cnt[end - 1] + cnt[end]
            pop!(vals); pop!(wts); pop!(cnt)
            vals[end] = v; wts[end] = wsum; cnt[end] = c
        end
    end
    return reduce(vcat, [fill(v, c) for (v, c) in zip(vals, cnt)])
end

function _des_interp(x::Vector{Float64}, y::Vector{Float64}, xq::Real)
    xq <= x[1] && return y[1]
    xq >= x[end] && return y[end]
    i = searchsortedlast(x, xq)
    x[i] == xq && return y[i]
    t = (xq - x[i]) / (x[i + 1] - x[i])
    return y[i] + t * (y[i + 1] - y[i])
end

"""
    optimize_design(design; space, target_power=0.8, cost=nothing, estimator=nothing,
                    sims_per_point=200, n_initial=5, max_sims=4_000, n_refine=2,
                    surrogate=:probit, transform=:sqrt, alpha=0.05, level=0.95,
                    verify_sims=0, rng=Random.default_rng(),
                    threaded=Threads.nthreads() > 1) -> DesignOptimization

Find the cheapest design that reaches a target power with few simulations, by
surrogate modelling of simulated power.

When power can only be obtained by simulation, evaluating it on a fine grid of
designs ([`diagnose_grid`](@ref)) is expensive, and most simulations are spent on
designs far from the target. Zimmer & Debelak (2025) propose instead to fit a
surrogate model of power as a function of the design parameters to a few simulated
designs, to choose the next designs to simulate where the surrogate is most
informative about the target, and to iterate; Zimmer, Henninger & Debelak (2024)
describe the approach and its implementation in the R package `mlpwr`, including
cost-constrained designs with several parameters (e.g. clusters versus cluster size).
This function implements a simple version of that idea for a
[`DeclaredDesign`](@ref):

1. The candidate designs are the Cartesian product of the vectors in `space` (e.g.
   `(n = 20:10:600,)` or `(n_clusters = 10:2:80, cluster_size = [5, 10, 20, 40])`),
   each with a `cost` (default: the product of the parameters).
2. `n_initial` candidates spread evenly over the cost range are simulated
   `sims_per_point` times each, recording rejections of ``H_0: \\theta = 0`` at level
   `alpha` for `estimator`.
3. A surrogate is fitted to all simulated batches: a binomial regression of the
   rejection indicator on the transformed parameters (`surrogate = :probit` with
   `transform = :sqrt` mimics the normal-theory power curve
   ``\\Phi(a + b\\sqrt{n})``; `:logistic` uses the logit link; `transform = :log` or
   `:identity` are alternatives), or, for a single parameter, isotonic (monotone)
   regression interpolated linearly (`:isotonic`).
4. The cheapest candidate with predicted power at least `target_power` is the current
   optimum; it and the `n_refine - 1` other candidates whose predicted power is
   closest to the target are simulated next, refining the surrogate near the
   boundary. Steps 3 and 4 repeat until `max_sims` simulations have been used.
5. Optionally, `verify_sims` fresh simulations at the chosen design check the
   surrogate's prediction.

The surrogate borrows strength across designs, so far fewer simulations are needed
than for a full grid, but the answer inherits the surrogate's assumptions: a
parametric link that is misspecified over the whole candidate range can place the
boundary in the wrong place, which is why the verification run is recommended and
its simulated power (with Monte Carlo standard error) should be reported. The
chosen design is optimal only for the declared model and estimator; uncertainty
about the model (effect size, ICC) is better handled by repeating the search under
alternative declarations than by a single optimization.

# Arguments
- `design::DeclaredDesign`: a design from [`declare_design`](@ref).

# Keywords
- `space::NamedTuple`: candidate values of each numeric design parameter (required).
- `target_power::Real = 0.8`: power to reach, in `(0, 1)`.
- `cost = nothing`: `nothing` (product of the parameters) or a function
  `params::NamedTuple -> Real`.
- `estimator = nothing`: label of the estimator whose power is optimized (default:
  the first).
- `sims_per_point::Integer = 200`: simulations per evaluated candidate (at least 10).
- `n_initial::Integer = 5`: candidates simulated before the first surrogate fit.
- `max_sims::Integer = 4_000`: total simulation budget of the search.
- `n_refine::Integer = 2`: candidates simulated in each refinement round.
- `surrogate::Symbol = :probit`: `:probit`, `:logistic` or `:isotonic`.
- `transform = :sqrt`: `:sqrt`, `:log` or `:identity` applied to every parameter, or a
  function `params::NamedTuple -> Vector{Float64}` returning the surrogate's features
  (an intercept is added), e.g.
  `p -> [sqrt(p.n_clusters / (0.1 + 0.9 / p.cluster_size))]` for the standardized
  noncentrality of a cluster trial with ICC 0.1 (GLM surrogates).
- `alpha::Real = 0.05`: level of the test whose power is optimized.
- `level::Real = 0.95`: interval level passed to the estimators.
- `verify_sims::Integer = 0`: simulations of the final verification (0: none).
- `rng::AbstractRNG = Random.default_rng()`: source of the seeds; results are
  reproducible given `rng`.
- `threaded::Bool = Threads.nthreads() > 1`: run simulations on threads; results are
  identical with and without threads.

# Returns
- [`DesignOptimization`](@ref).

# Examples
```julia
using DrSnow, DataFrames, Statistics, StableRNGs
pop(rng, p) = (Y0 = randn(rng, p.n); DataFrame(Y0=Y0, Y1=Y0 .+ p.effect))
d = declare_design(pop; params=(n=100, effect=0.3),
                   assignment=(data, p) -> CompleteRandomization(p.n, p.n ÷ 2),
                   estimators="DiM" => (data, p) -> experiment_estimate(data, :Y, :Z))
opt = optimize_design(d; space=(n=20:4:600,), target_power=0.8, max_sims=2000,
                      verify_sims=1000, rng=StableRNG(1))
opt.best.n
```

# References
- Zimmer, F., & Debelak, R. (2025). Simulation-based design optimization for
  statistical power: Utilizing machine learning. *Psychological Methods*, 30(3),
  513–536.
- Zimmer, F., Henninger, M., & Debelak, R. (2024). Sample size planning for complex
  study designs: A tutorial for the mlpwr package. *Behavior Research Methods*, 56(5),
  5246–5263.
"""
function optimize_design(design::DeclaredDesign; space::NamedTuple,
                         target_power::Real=0.8, cost=nothing, estimator=nothing,
                         sims_per_point::Integer=200, n_initial::Integer=5,
                         max_sims::Integer=4_000, n_refine::Integer=2,
                         surrogate::Symbol=:probit, transform=:sqrt,
                         alpha::Real=0.05, level::Real=0.95, verify_sims::Integer=0,
                         rng::AbstractRNG=Random.default_rng(),
                         threaded::Bool=Threads.nthreads() > 1)
    ctx = "optimize_design"
    0 < target_power < 1 || throw(ArgumentError("$ctx: target_power must be in (0, 1)"))
    surrogate in (:probit, :logistic, :isotonic) ||
        throw(ArgumentError("$ctx: surrogate must be :probit, :logistic or :isotonic"))
    transform isa Function || transform in (:sqrt, :log, :identity) ||
        throw(ArgumentError("$ctx: transform must be :sqrt, :log, :identity or a " *
                            "function"))
    (sims_per_point >= 10 && n_initial >= 2 && n_refine >= 1) ||
        throw(ArgumentError("$ctx: need sims_per_point ≥ 10, n_initial ≥ 2, n_refine ≥ 1"))
    rows, pnames = _des_grid_rows(space)
    all(r -> all(v -> v isa Real, values(r)), rows) ||
        throw(ArgumentError("$ctx: design parameters in `space` must be numeric"))
    surrogate === :isotonic && length(pnames) > 1 &&
        throw(ArgumentError("$ctx: the isotonic surrogate needs a single parameter"))
    length(rows) >= n_initial ||
        throw(ArgumentError("$ctx: fewer candidates than n_initial"))
    max_sims >= n_initial * sims_per_point ||
        throw(ArgumentError("$ctx: max_sims is below n_initial × sims_per_point"))
    est = estimator === nothing ? design.estimators[1].first : string(estimator)
    est in first.(design.estimators) ||
        throw(ArgumentError("$ctx: unknown estimator $est"))
    costfun = cost === nothing ? (p -> prod(float(p[k]) for k in pnames)) : cost
    costs = [float(costfun(merge(design.params, r))) for r in rows]
    # candidates sorted by cost, ties by parameter values
    ord = sortperm(collect(eachindex(rows)); by=i -> (costs[i], Tuple(rows[i])))
    rows, costs = rows[ord], costs[ord]
    C = length(rows)
    F = reduce(vcat, [_des_features(r, pnames, transform)' for r in rows])

    kk = zeros(Float64, C)          # rejections per candidate
    mm = zeros(Float64, C)          # simulations per candidate
    evals = DataFrame([k => eltype(getfield.(rows, k))[] for k in pnames]...)
    evals.round = Int[]; evals.sims = Int[]; evals.rejections = Int[]
    evals.power = Float64[]; evals.cost = Float64[]
    used = 0
    seedrng = Random.Xoshiro(task_seeds(rng, 1)[1])

    function evaluate!(i, round)
        p = merge(design.params, rows[i])
        srng = Random.Xoshiro(rand(seedrng, UInt64))
        sim = _des_run_sims(design, p, sims_per_point, srng, level, threaded, :throw)
        sub = sim[sim.estimator .== est, :]
        r = count(sub.p_value .<= alpha)
        kk[i] += r
        mm[i] += nrow(sub)
        used += sims_per_point
        push!(evals, (values(rows[i])..., round, nrow(sub), r, r / nrow(sub), costs[i]))
        return nothing
    end

    function fit_predict()
        ev = findall(>(0), mm)
        if surrogate === :isotonic
            x = [float(rows[i][pnames[1]]) for i in ev]
            o = sortperm(x)
            xs, ys = x[o], (kk[ev] ./ mm[ev])[o]
            fitted = _des_pava(ys, mm[ev][o])
            pw = [_des_interp(xs, fitted, float(r[pnames[1]])) for r in rows]
            return pw, fill(NaN, C)
        end
        length(ev) > size(F, 2) ||
            throw(ArgumentError("$ctx: too few evaluated designs for the surrogate"))
        β, V, lk = _des_fit_glm(F[ev, :], kk[ev], mm[ev],
                                surrogate === :probit ? :probit : :logit)
        return _des_predict_glm(β, V, lk, F)
    end

    choose(pw) = (f = findfirst(>=(target_power), pw); f === nothing ?
                  (argmax(pw), false) : (f, true))

    init = unique(round.(Int, range(1, C; length=n_initial)))
    for i in init
        evaluate!(i, 0)
    end
    round_ = 0
    pw, se = fit_predict()
    while used + sims_per_point <= max_sims
        round_ += 1
        b, _ = choose(pw)
        cand = [b]
        others = sortperm(abs.(pw .- target_power))
        for i in others
            length(cand) >= n_refine && break
            i in cand || push!(cand, i)
        end
        for i in cand
            used + sims_per_point <= max_sims || break
            evaluate!(i, round_)
        end
        pw, se = fit_predict()
    end
    b, reached = choose(pw)
    ver = nothing
    if verify_sims > 0
        p = merge(design.params, rows[b])
        sim = _des_run_sims(design, p, verify_sims, Random.Xoshiro(rand(seedrng, UInt64)),
                            level, threaded, :throw)
        sub = sim[sim.estimator .== est, :]
        pv = mean(sub.p_value .<= alpha)
        ver = (power=pv, mc_se=sqrt(pv * (1 - pv) / nrow(sub)), sims=nrow(sub))
    end
    preds = DataFrame([k => [r[k] for r in rows] for k in pnames]...)
    preds.cost = costs
    preds.power = pw
    preds.se = se
    return DesignOptimization(rows[b], costs[b], pw[b], se[b], float(target_power),
                              reached, est, evals, preds, surrogate, transform, used, ver)
end
