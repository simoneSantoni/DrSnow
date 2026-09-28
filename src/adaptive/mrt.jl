# Micro-randomized trials: causal excursion effects with the weighted and centered
# least squares estimator (WCLS; Boruvka et al. 2018) for continuous outcomes and the
# estimator of the marginal excursion effect (EMEE; Qian et al. 2021) on the log
# relative-risk scale for binary outcomes, with the small-sample corrected sandwich
# variance of the R package MRTAnalysis.

"""
    ExcursionEffectEstimate <: CausalEstimate

Causal excursion effect of a micro-randomized trial, estimated by [`wcls`](@ref)
(difference scale, continuous outcomes) or [`emee`](@ref) (log relative-risk scale,
binary outcomes).

The effect is modelled as linear in the moderators, ``β(S_t) = (1, S_t')\\, β``; the
coefficients are the components of ``β``: `"(Intercept)"` is the excursion effect at
``S_t = 0`` and each moderator coefficient the change in the effect per unit of that
moderator. The control model ``Z_t'α`` (or ``\\exp(Z_t'α)`` for EMEE) only serves
to reduce variance; its coefficients are nuisance parameters without a causal
interpretation and are stored separately. Inference uses the participant-clustered
sandwich variance and ``t(n - p - q)`` reference distributions, where ``n`` is the
number of participants and ``p``, ``q`` the numbers of moderator and control terms
(intercepts included), as in the R package MRTAnalysis.

# Fields
- `coef::Vector{Float64}`, `vcov::Matrix{Float64}`: ``\\hat β`` and its covariance.
- `names::Vector{String}`: `"(Intercept)"` and the moderator names.
- `control_coef::Vector{Float64}`, `control_names::Vector{String}`: nuisance
  coefficients ``\\hat α`` of the control model.
- `full_vcov::Matrix{Float64}`: covariance of ``(\\hat α, \\hat β)``.
- `n_obs::Int`: available person-decision points used; `n_ids::Int`: participants.
- `method::Symbol`: `:wcls` or `:emee`.
- `small_sample::Bool`: whether the small-sample corrected variance was used.
- `moderator_ranges::Vector{Tuple{Float64,Float64}}`, `moderator_means`: range and
  mean of each moderator over the available decision points (used by
  [`plot_excursion_effect`](@ref)).

The StatsAPI accessors `coef`, `vcov`, `stderror`, `confint`, `coeftable`, `nobs`
and `dof_residual` apply; `exp.(coef(r))` gives relative risks for EMEE.
"""
struct ExcursionEffectEstimate <: CausalEstimate
    coef::Vector{Float64}
    vcov::Matrix{Float64}
    names::Vector{String}
    control_coef::Vector{Float64}
    control_names::Vector{String}
    full_vcov::Matrix{Float64}
    n_obs::Int
    n_ids::Int
    method::Symbol
    small_sample::Bool
    moderator_ranges::Vector{Tuple{Float64,Float64}}
    moderator_means::Vector{Float64}
end

StatsAPI.coef(r::ExcursionEffectEstimate) = r.coef
StatsAPI.vcov(r::ExcursionEffectEstimate) = r.vcov
StatsAPI.coefnames(r::ExcursionEffectEstimate) = r.names
StatsAPI.nobs(r::ExcursionEffectEstimate) = r.n_obs
StatsAPI.dof_residual(r::ExcursionEffectEstimate) =
    float(r.n_ids - length(r.coef) - length(r.control_coef))
estimand(r::ExcursionEffectEstimate) = r.method === :wcls ?
    "causal excursion effect (difference in mean proximal outcome)" :
    "causal excursion effect (log relative risk of the binary proximal outcome)"
method_name(r::ExcursionEffectEstimate) = r.method === :wcls ?
    "Weighted and centered least squares (WCLS)" :
    "Estimator of the marginal excursion effect (EMEE)"
_glance_nclusters(r::ExcursionEffectEstimate) = r.n_ids

function show_details(io::IO, r::ExcursionEffectEstimate)
    print(io, "\nParticipants: ", r.n_ids, "; t reference with ",
          Int(dof_residual(r)), " degrees of freedom; ",
          r.small_sample ? "small-sample corrected" : "uncorrected",
          " sandwich variance.")
    return nothing
end

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

struct _AdMRT
    id::Vector{Int}                 # participant index (1:n) of each available row
    y::Vector{Float64}
    a::Vector{Float64}
    p::Vector{Float64}
    ptilde::Vector{Float64}
    Z::Matrix{Float64}              # controls (with intercept)
    X::Matrix{Float64}              # moderators (with intercept)
    n_ids::Int
    groups::Vector{Vector{Int}}
end

function _ad_mrt_col(data, c, rows, context)
    v = data[rows, c]
    any(ismissing, v) &&
        throw(ArgumentError("$context: missing values in $c at available decision " *
                            "points"))
    T = nonmissingtype(eltype(v))
    T <: Real || throw(ArgumentError("$context: column $c must be numeric"))
    out = Float64.(v)
    all(isfinite, out) || throw(ArgumentError("$context: non-finite values in $c"))
    return out
end

function _ad_mrt_data(data, outcome, treatment, id, rand_prob, moderators, controls,
                      availability, numerator_prob, binary, context)
    mods = Symbol.(collect(moderators))
    ctrls = Symbol.(collect(controls))
    cols = Symbol[outcome, treatment, id]
    rand_prob isa Symbol && push!(cols, rand_prob)
    numerator_prob isa Symbol && push!(cols, numerator_prob)
    availability === nothing || push!(cols, availability)
    require_columns(data, unique(vcat(cols, mods, ctrls)); context=context)
    n = nrow(data)
    avail = if availability === nothing
        trues(n)
    else
        av = data[!, availability]
        any(ismissing, av) && throw(ArgumentError("$context: missing availability"))
        all(v -> v == 0 || v == 1, av) ||
            throw(ArgumentError("$context: availability must be 0/1"))
        av .== 1
    end
    rows = findall(avail)
    isempty(rows) && throw(ArgumentError("$context: no available decision points"))
    rawid = data[rows, id]
    any(ismissing, rawid) && throw(ArgumentError("$context: missing participant ids"))
    labels = sort(unique(rawid))
    idx = Dict(labels[i] => i for i in eachindex(labels))
    ids = [idx[v] for v in rawid]
    y = _ad_mrt_col(data, outcome, rows, context)
    a = _ad_mrt_col(data, treatment, rows, context)
    all(v -> v == 0 || v == 1, a) || throw(ArgumentError("$context: treatment must be 0/1"))
    binary && !all(v -> v == 0 || v == 1, y) &&
        throw(ArgumentError("$context: the outcome must be binary (0/1) for EMEE; use " *
                            "wcls for continuous outcomes"))
    getp(spec, what) = if spec isa Real
        fill(float(spec), length(rows))
    elseif spec isa Symbol
        _ad_mrt_col(data, spec, rows, context)
    else
        throw(ArgumentError("$context: $what must be a number or a column name"))
    end
    p = getp(rand_prob, "rand_prob")
    pt = getp(numerator_prob, "numerator_prob")
    all(v -> 0 < v < 1, p) ||
        throw(ArgumentError("$context: randomization probabilities must be in (0, 1)"))
    all(v -> 0 < v < 1, pt) ||
        throw(ArgumentError("$context: numerator probabilities must be in (0, 1)"))
    m = length(rows)
    Z = hcat(ones(m), [_ad_mrt_col(data, c, rows, context) for c in ctrls]...)
    X = hcat(ones(m), [_ad_mrt_col(data, c, rows, context) for c in mods]...)
    groups = [Int[] for _ in labels]
    for (r, g) in enumerate(ids)
        push!(groups[g], r)
    end
    # identification: the weighted design (Z, (A - p̃) X) must have full column rank
    w = [a[r] == 1 ? pt[r] / p[r] : (1 - pt[r]) / (1 - p[r]) for r in 1:m]
    D = hcat(Z, (a .- pt) .* X) .* sqrt.(w)
    rank(D) == size(D, 2) || throw(ArgumentError(
        "$context: the design (controls, centered treatment × moderators) is rank " *
        "deficient: collinear controls or moderators, or no variation in treatment"))
    return _AdMRT(ids, y, a, p, pt, Z, X, length(labels), groups), mods, ctrls
end

function _ad_mrt_names(mods, ctrls)
    return vcat("(Intercept)", string.(mods)), vcat("(Intercept)", string.(ctrls))
end

function _ad_mrt_result(θ, V, d::_AdMRT, mods, ctrls, method, small)
    q = size(d.Z, 2)
    p = size(d.X, 2)
    d.n_ids > p + q || throw(ArgumentError(
        "$(method): need more participants ($(d.n_ids)) than parameters ($(p + q))"))
    all(isfinite, θ) && all(isfinite, V) ||
        throw(ArgumentError("$(method): the estimate or its variance is not finite"))
    mn, cn = _ad_mrt_names(mods, ctrls)
    β = θ[(q + 1):end]
    Vb = Matrix(Symmetric(V[(q + 1):end, (q + 1):end]))
    S = d.X[:, 2:end]
    ranges = [extrema(view(S, :, j)) for j in axes(S, 2)]
    return ExcursionEffectEstimate(β, Vb, mn, θ[1:q], cn, Matrix(Symmetric(V)),
                                   length(d.y), d.n_ids, method, small, ranges,
                                   vec(mean(S; dims=1)))
end

# ---------------------------------------------------------------------------
# WCLS
# ---------------------------------------------------------------------------

"""
    wcls(data, outcome, treatment, id; rand_prob, moderators=Symbol[],
         controls=Symbol[], availability=nothing, numerator_prob=0.5,
         small_sample=true) -> ExcursionEffectEstimate

Causal excursion effect of a binary treatment on a continuous proximal outcome in a
micro-randomized trial, estimated by weighted and centered least squares (WCLS;
Boruvka, Almirall, Witkiewitz and Murphy 2018), as in `MRTAnalysis::wcls`.

**Design and estimand.** In a micro-randomized trial (Klasnja et al. 2015) each
participant is randomized many times, at every decision point ``t`` at which they are
available for treatment (``I_t = 1``), with a known probability
``p_t(H_t) = P(A_t = 1 \\mid H_t)`` that may depend on the participant's history
``H_t``. The causal excursion effect

```math
β(S_t) = E\\bigl[Y_{t+1}(\\bar A_{t-1}, 1) - Y_{t+1}(\\bar A_{t-1}, 0)
\\mid S_t, I_t = 1\\bigr]
```

is the effect of treating rather than not treating at ``t`` on the proximal outcome
``Y_{t+1}``, for available participants with moderator values ``S_t``, averaged over
the trial's own randomization of earlier treatments ``\\bar A_{t-1}`` and over
everything in the history that is not in ``S_t``. It is modelled as
``β(S_t) = (1, S_t')\\, β``. Because the effect is marginal over the treatment
policy of the trial, it describes an "excursion" from that policy and depends on it;
it is not the effect of a sustained treatment regime.

**Identification and estimator.** Randomization with known ``p_t`` and consistency
identify ``β(S_t)``; no model for the outcome is needed. WCLS solves

```math
\\min_{α, β} \\sum_{i, t} I_{it} W_{it} \\bigl(Y_{i,t+1} - Z_{it}'α
- (A_{it} - \\tilde p_{it}) (1, S_{it}')\\, β\\bigr)^2,
\\qquad
W_{it} = \\Bigl(\\frac{\\tilde p_{it}}{p_{it}}\\Bigr)^{A_{it}}
\\Bigl(\\frac{1 - \\tilde p_{it}}{1 - p_{it}}\\Bigr)^{1 - A_{it}},
```

where the numerator probability ``\\tilde p_t`` depends on ``S_t`` only and the
controls ``Z_t`` (intercept included) enter a working model for the main effects.
Centering the treatment at ``\\tilde p_t`` and weighting by ``W_t`` make ``\\hat β``
consistent even when ``Z_t'α`` is misspecified; good controls reduce the variance.
The controls should include the moderators so that their main effects are absorbed.

**Inference.** The variance is the sandwich estimator clustered by participant.
With `small_sample = true` the residuals of participant ``i`` are premultiplied by
``(I - H_{ii})^{-1}`` (Mancl and DeRouen 2001), and intervals use the
``t(n - p - q)`` distribution with ``n`` participants. MRTAnalysis applies the
correction only when there are at most 50 participants (its `small = 50` default);
set `small_sample = n ≤ 50` to reproduce it exactly. The package's estimates agree
with MRTAnalysis to `1e-7` or better, and a Monte Carlo study gives 95% coverage
with 30 participants. Report the moderators, the controls, the randomization
probabilities and the number of participants; unavailable decision points are
dropped and their outcomes may be missing.

# Arguments
- `data`: long table with one row per participant × decision point (any row order).
- `outcome::Symbol`: proximal outcome ``Y_{t+1}`` recorded on the row of decision
  point ``t``.
- `treatment::Symbol`: 0/1 treatment ``A_t``.
- `id::Symbol`: participant identifier (the clustering unit).

# Keywords
- `rand_prob` (required): column name or constant with the randomization
  probability ``p_t``, in `(0, 1)`.
- `moderators::Vector{Symbol} = Symbol[]`: numeric effect moderators ``S_t``; with
  none, the fully marginal excursion effect is estimated.
- `controls::Vector{Symbol} = Symbol[]`: numeric control variables ``Z_t`` of the
  working model (an intercept is always included).
- `availability = nothing`: 0/1 availability column (default: always available).
- `numerator_prob = 0.5`: constant or column with ``\\tilde p_t``; it must depend on
  the moderators only.
- `small_sample::Bool = true`: use the small-sample corrected variance.

# Returns
- An [`ExcursionEffectEstimate`](@ref).

# Examples
```julia
using DrSnow, DataFrames, StableRNGs
rng = StableRNG(1)
n, T = 40, 30
id = repeat(1:n; inner=T)
day = repeat(1:T, n) ./ T
x = rand(rng, n * T)
avail = Int.(rand(rng, n * T) .< 0.8)
prob = ifelse.(x .> 0.5, 0.6, 0.3)
a = Float64.(rand(rng, n * T) .< prob)
u = repeat(randn(rng, n); inner=T)
y = 1 .+ x .+ u .+ a .* (0.3 .- 0.4 .* day) .+ randn(rng, n * T)
df = DataFrame(id=id, day=day, x=x, avail=avail, prob=prob, a=a, y=y)
r = wcls(df, :y, :a, :id; rand_prob=:prob, moderators=[:day],
         controls=[:day, :x], availability=:avail)
coeftable(r)
```

# References
- Boruvka, A., Almirall, D., Witkiewitz, K., & Murphy, S. A. (2018). Assessing
  time-varying causal effect moderation in mobile health. *Journal of the American
  Statistical Association*, 113(523), 1112–1121.
- Klasnja, P., Hekler, E. B., Shiffman, S., Boruvka, A., Almirall, D., Tewari, A., &
  Murphy, S. A. (2015). Microrandomized trials: An experimental design for
  developing just-in-time adaptive interventions. *Health Psychology*, 34(Suppl.),
  1220–1228.
- Klasnja, P., Smith, S., Seewald, N. J., Lee, A., Hall, K., Luers, B., Hekler,
  E. B., & Murphy, S. A. (2019). Efficacy of contextually tailored suggestions for
  physical activity: A micro-randomized optimization trial of HeartSteps. *Annals of
  Behavioral Medicine*, 53(6), 573–582.
- Mancl, L. A., & DeRouen, T. A. (2001). A covariance estimator for GEE with
  improved small-sample properties. *Biometrics*, 57(1), 126–134.
- Qian, T., Yoo, H., Klasnja, P., Almirall, D., & Murphy, S. A. (2021). Estimating
  time-varying causal excursion effects in mobile health with binary outcomes.
  *Biometrika*, 108(3), 507–527.
"""
function wcls(data, outcome::Symbol, treatment::Symbol, id::Symbol; rand_prob=nothing,
              moderators=Symbol[], controls=Symbol[], availability=nothing,
              numerator_prob=0.5, small_sample::Bool=true)
    ctx = "wcls"
    rand_prob === nothing &&
        throw(ArgumentError("$ctx: pass `rand_prob` (a column or a constant)"))
    d, mods, ctrls = _ad_mrt_data(data, outcome, treatment, id, rand_prob, moderators,
                                  controls, availability, numerator_prob, false, ctx)
    w = [d.a[r] == 1 ? d.ptilde[r] / d.p[r] : (1 - d.ptilde[r]) / (1 - d.p[r])
         for r in eachindex(d.y)]
    D = hcat(d.Z, (d.a .- d.ptilde) .* d.X)
    DW = D .* w
    Bm = Symmetric(D' * DW)
    F = cholesky(Bm; check=false)
    issuccess(F) || throw(ArgumentError(
        "$ctx: the weighted design is singular (collinear controls/moderators or no " *
        "variation in treatment)"))
    θ = F \ (DW' * d.y)
    r = d.y .- D * θ
    Binv = inv(F)
    k = length(θ)
    meat = zeros(k, k)
    for g in d.groups
        Dg = D[g, :]
        rg = r[g]
        if small_sample
            Hg = Dg * Binv * (Dg .* w[g])'
            rg = (I - Hg) \ rg
        end
        u = Dg' * (w[g] .* rg)
        meat .+= u * u'
    end
    V = Binv * meat * Binv
    return _ad_mrt_result(θ, V, d, mods, ctrls, :wcls, small_sample)
end

# ---------------------------------------------------------------------------
# EMEE
# ---------------------------------------------------------------------------

"""
    emee(data, outcome, treatment, id; rand_prob, moderators=Symbol[],
         controls=Symbol[], availability=nothing, numerator_prob=0.5,
         maxiter=100, tol=1e-10) -> ExcursionEffectEstimate

Causal excursion effect of a binary treatment on a *binary* proximal outcome in a
micro-randomized trial, on the log relative-risk scale, estimated with the estimator
of the marginal excursion effect (EMEE) of Qian, Yoo, Klasnja, Almirall and Murphy
(2021), as in `MRTAnalysis::emee`.

**Estimand.** With the notation of [`wcls`](@ref) (availability ``I_t``,
randomization probability ``p_t``, moderators ``S_t``), the estimand is the causal
excursion effect on the relative-risk scale,

```math
\\log \\frac{E[Y_{t+1}(\\bar A_{t-1}, 1) \\mid S_t, I_t = 1]}
{E[Y_{t+1}(\\bar A_{t-1}, 0) \\mid S_t, I_t = 1]} = (1, S_t')\\, β,
```

again marginal over the trial's own treatment policy and over the part of the
history not in ``S_t``. The relative-risk scale suits binary outcomes whose baseline
probability varies across participants and time, where a difference-scale linear
model could leave the unit interval; `exp.(coef(r))` are relative risks.

**Estimator.** With ``X_t = (1, S_t')'``, weights ``W_t`` and centered treatment
``A_t - \\tilde p_t`` as in [`wcls`](@ref), EMEE solves the estimating equations

```math
\\sum_{i,t} I_{it} W_{it}\\, e^{-A_{it} X_{it}'β}
\\bigl(Y_{i,t+1} - e^{Z_{it}'α + A_{it} X_{it}'β}\\bigr)
\\begin{pmatrix} Z_{it} \\\\ (A_{it} - \\tilde p_{it}) X_{it} \\end{pmatrix} = 0
```

by damped Newton iterations from zero. The factor ``e^{-A X'β}`` removes the
treatment effect from the residual, so that, as for WCLS, ``\\hat β`` is consistent
even when the control model ``\\exp(Z_t'α)`` for the untreated risk is misspecified
(Qian et al. 2021). Convergence failure is reported as an error rather than
returned silently.

**Inference.** The variance is the participant-clustered sandwich with the
small-sample correction of MRTAnalysis (residuals of participant ``i`` premultiplied
by ``(I - H_{ii})^{-1}``, following Mancl and DeRouen 2001), and intervals use
``t(n - p - q)`` with ``n`` participants. The package's estimates agree with
MRTAnalysis to `1e-7` or better; in the package's Monte Carlo study with 40
participants the intervals were conservative (about 97% coverage at nominal 95%).

# Arguments
- `data`, `outcome::Symbol`, `treatment::Symbol`, `id::Symbol`: as in
  [`wcls`](@ref); the outcome must be 0/1.

# Keywords
- `rand_prob`, `moderators`, `controls`, `availability`, `numerator_prob`: as in
  [`wcls`](@ref).
- `maxiter::Integer = 100`: maximum number of Newton iterations.
- `tol::Real = 1e-10`: convergence tolerance on the maximum absolute estimating
  equation divided by the number of participants.

# Returns
- An [`ExcursionEffectEstimate`](@ref) on the log relative-risk scale
  (`exp.(confint(r))` gives relative-risk intervals).

# Examples
```julia
using DrSnow, DataFrames, StableRNGs
rng = StableRNG(2)
n, T = 40, 30
id = repeat(1:n; inner=T)
day = repeat(1:T, n) ./ T
x = rand(rng, n * T)
avail = Int.(rand(rng, n * T) .< 0.8)
prob = ifelse.(x .> 0.5, 0.6, 0.3)
a = Float64.(rand(rng, n * T) .< prob)
y = Float64.(rand(rng, n * T) .< 0.3 .* exp.(0.5 .* x) .* exp.(0.2 .* a))
df = DataFrame(id=id, day=day, x=x, avail=avail, prob=prob, a=a, y=y)
r = emee(df, :y, :a, :id; rand_prob=:prob, moderators=[:day],
         controls=[:day, :x], availability=:avail)
exp.(confint(r))
```

# References
- Qian, T., Yoo, H., Klasnja, P., Almirall, D., & Murphy, S. A. (2021). Estimating
  time-varying causal excursion effects in mobile health with binary outcomes.
  *Biometrika*, 108(3), 507–527.
- Boruvka, A., Almirall, D., Witkiewitz, K., & Murphy, S. A. (2018). Assessing
  time-varying causal effect moderation in mobile health. *Journal of the American
  Statistical Association*, 113(523), 1112–1121.
- Mancl, L. A., & DeRouen, T. A. (2001). A covariance estimator for GEE with
  improved small-sample properties. *Biometrics*, 57(1), 126–134.
- Klasnja, P., Hekler, E. B., Shiffman, S., Boruvka, A., Almirall, D., Tewari, A., &
  Murphy, S. A. (2015). Microrandomized trials: An experimental design for
  developing just-in-time adaptive interventions. *Health Psychology*, 34(Suppl.),
  1220–1228.
"""
function emee(data, outcome::Symbol, treatment::Symbol, id::Symbol; rand_prob=nothing,
              moderators=Symbol[], controls=Symbol[], availability=nothing,
              numerator_prob=0.5, maxiter::Integer=100, tol::Real=1e-10)
    ctx = "emee"
    rand_prob === nothing &&
        throw(ArgumentError("$ctx: pass `rand_prob` (a column or a constant)"))
    d, mods, ctrls = _ad_mrt_data(data, outcome, treatment, id, rand_prob, moderators,
                                  controls, availability, numerator_prob, true, ctx)
    q = size(d.Z, 2)
    p = size(d.X, 2)
    n = d.n_ids
    w = [d.a[r] == 1 ? d.ptilde[r] / d.p[r] : (1 - d.ptilde[r]) / (1 - d.p[r])
         for r in eachindex(d.y)]
    ca = d.a .- d.ptilde
    function ee(θ)
        α = θ[1:q]
        β = θ[(q + 1):end]
        Xβ = d.X * β
        Zα = d.Z * α
        U = zeros(q + p)
        J = zeros(q + p, q + p)
        for t in eachindex(d.y)
            pre = exp(-d.a[t] * Xβ[t]) * w[t]
            μ = exp(Zα[t] + d.a[t] * Xβ[t])
            rt = d.y[t] - μ
            Dt = pre .* vcat(d.Z[t, :], ca[t] .* d.X[t, :])
            U .+= Dt .* rt
            dr = -μ .* vcat(d.Z[t, :], d.a[t] .* d.X[t, :])
            # ∂D/∂θ · r: only the β-columns are non-zero
            J[1:q, (q + 1):end] .+= (-pre * d.a[t] * rt) .* (d.Z[t, :] * d.X[t, :]')
            J[(q + 1):end, (q + 1):end] .+= (-pre * d.a[t] * ca[t] * rt) .*
                                           (d.X[t, :] * d.X[t, :]')
            J .+= Dt * dr'
        end
        return U, J
    end
    θ = zeros(q + p)
    U, J = ee(θ)
    converged = maximum(abs, U) / n < tol
    iter = 0
    while !converged && iter < maxiter
        iter += 1
        step = try
            J \ U
        catch
            throw(ArgumentError("$ctx: singular Jacobian of the estimating equations " *
                                "(collinear terms or no treatment variation)"))
        end
        λ = 1.0
        cur = sum(abs2, U)
        local θn, Un, Jn
        while true
            θn = θ .- λ .* step
            Un, Jn = ee(θn)
            (all(isfinite, Un) && sum(abs2, Un) < cur) && break
            λ /= 2
            λ < 1e-8 && break
        end
        θ, U, J = θn, Un, Jn
        converged = maximum(abs, U) / n < tol
    end
    converged || throw(ArgumentError("$ctx: Newton's method did not converge in " *
                                     "$maxiter iterations"))
    Minv = inv(J)
    # Small-sample corrected meat (MRTAnalysis): D_i (I - H_ii)^{-1} r_i with
    # H_ii = (∂r_i/∂θ) M⁻¹ D_i, M the (unnormalized) Jacobian.
    α = θ[1:q]
    β = θ[(q + 1):end]
    meat = zeros(q + p, q + p)
    for g in d.groups
        m = length(g)
        Dg = zeros(q + p, m)
        Rg = zeros(m, q + p)
        rg = zeros(m)
        for (k, t) in enumerate(g)
            xb = dot(d.X[t, :], β)
            μ = exp(dot(d.Z[t, :], α) + d.a[t] * xb)
            pre = exp(-d.a[t] * xb) * w[t]
            Dg[:, k] = pre .* vcat(d.Z[t, :], ca[t] .* d.X[t, :])
            Rg[k, :] = -μ .* vcat(d.Z[t, :], d.a[t] .* d.X[t, :])
            rg[k] = d.y[t] - μ
        end
        Hg = Rg * Minv * Dg
        u = Dg * ((I - Hg) \ rg)
        meat .+= u * u'
    end
    V = Minv * meat * Minv'
    return _ad_mrt_result(θ, V, d, mods, ctrls, :emee, true)
end
