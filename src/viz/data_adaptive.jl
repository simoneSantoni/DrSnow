# Plot stubs and backend-independent plot data for the adaptive-experiments area:
# assignment probabilities over the course of an adaptive experiment, and causal
# excursion effects of micro-randomized trials as a function of a moderator.

"""
    plot_assignment_probabilities(log::AdaptiveLog; floor=true, figure=(;), axis=(;),
                                  colors=nothing) -> Makie.Figure

Assignment probability of every arm over the course of an adaptive experiment, with
the probability floor and the end of the burn-in.

In an adaptive experiment (e.g. a Thompson-sampling or other bandit design) the
probability with which each unit is assigned to each arm is updated as outcomes
accrue. The plot draws, for every arm, the probability used for each successive unit
(one step line per arm, distinguished by colour and line style); for contextual
policies, whose probabilities differ across units within a batch, it draws the mean
over the units of each batch. A dashed line shows the policy's floor
``\\text{floor} \\cdot t^{-\\text{floor\\_decay}}``, the minimum probability of
every arm, and a dotted vertical line marks the end of the equal-allocation burn-in.

The plot shows how quickly the design concentrated on the arms it judged best, and
how close the other arms' probabilities came to zero. The latter governs inference:
inverse-probability-weighted estimates of arm values after adaptive assignment have
variances driven by the inverse of these probabilities, and near-zero probabilities
make naive estimates unstable and their normal approximations unreliable, which is
why adaptively weighted estimators and probability floors are used (Hadad, Hirshberg,
Zhan, Wager and Athey 2021). Fast concentration is good for the participants and
for regret, but leaves little information about the losing arms (Offer-Westort,
Coppock and Green 2021). The probabilities are the design's, not estimates; a rise
in one arm's probability is not by itself evidence that the arm is better.

# Arguments
- `log::AdaptiveLog`: the record of an adaptive experiment, from
  [`run_adaptive_experiment`](@ref) or [`experiment_log`](@ref).

# Keywords
- `floor::Bool = true`: draw the probability floor when the policy has one.
$(_VIZ_COMMON_KW)

# Returns
- `Makie.Figure`.

# Examples
```julia
using DrSnow, CairoMakie, StableRNGs
policy = GaussianThompson(3; floor=0.1, burnin=30)
log = run_adaptive_experiment(policy, GaussianBandit([0.0, 0.2, 0.4]), 1000;
                              rng=StableRNG(1))
plot_assignment_probabilities(log)
```

# References
- Hadad, V., Hirshberg, D. A., Zhan, R., Wager, S., & Athey, S. (2021). Confidence
  intervals for policy evaluation in adaptive experiments. *Proceedings of the
  National Academy of Sciences*, 118(15), e2014602118.
- Offer-Westort, M., Coppock, A., & Green, D. P. (2021). Adaptive experimental
  design: Prospects and applications in political science. *American Journal of
  Political Science*, 65(4), 826–844.
"""
function plot_assignment_probabilities end
"""
    plot_assignment_probabilities!(ax, log::AdaptiveLog; floor=true,
                                   colors=nothing) -> ax

Draw the assignment probabilities of an adaptive experiment into an existing Makie
axis. This is the mutating counterpart of [`plot_assignment_probabilities`](@ref),
which describes the display; no legend or subtitle is added.

# Arguments
- `ax::Makie.AbstractAxis`: the axis to draw into.
- `log::AdaptiveLog`: from [`run_adaptive_experiment`](@ref) or
  [`experiment_log`](@ref).
- Keywords: those of [`plot_assignment_probabilities`](@ref) except `figure` and
  `axis`.

# Returns
- `ax`, the axis drawn into.
"""
function plot_assignment_probabilities! end
plot_assignment_probabilities(args...; kwargs...) =
    _viz_no_backend(plot_assignment_probabilities, args)
plot_assignment_probabilities!(args...; kwargs...) =
    _viz_no_backend(plot_assignment_probabilities!, args)

"""
    plot_excursion_effect(r::ExcursionEffectEstimate; moderator=nothing, values=nothing,
                          at=nothing, level=0.95, relative_risk=false, figure=(;),
                          axis=(;), colors=nothing) -> Makie.Figure

Causal excursion effect of a micro-randomized trial as a function of one moderator,
with a pointwise confidence band.

In a micro-randomized trial each participant is randomized repeatedly, at every
decision point, and the causal excursion effect is the effect on the proximal
outcome of treating at a decision point versus not, averaged over the treatment
policy actually followed at the other points and conditional on moderators
``S_t``; it is modelled as linear in the moderators, ``S_t'\\beta``, on the
difference scale for [`wcls`](@ref) (Boruvka, Almirall, Witkiewitz and Murphy 2018)
and on the log relative-risk scale for binary outcomes with [`emee`](@ref) (Qian,
Yoo, Klasnja, Almirall and Murphy 2021). The plot draws the fitted effect
``\\beta'(1, s)`` as `moderator` varies over `values`, holding the other moderators
at `at` (default: their means), with a pointwise `level` confidence band from the
participant-clustered variance and a t reference distribution with the estimator's
degrees of freedom. Without moderators the marginal excursion effect is drawn as a
single point with its interval. A dashed line marks no effect (0, or 1 on the
relative-risk scale).

The curve is only as flexible as the linear moderator model, so a declining line
reflects the specification as much as the data; the band is pointwise, not a
simultaneous band for the whole curve; and moderators must be measured before the
decision point for the effect to be causal. With `relative_risk = true` (EMEE only)
the effect and band are exponentiated to the relative-risk scale.

# Arguments
- `r::ExcursionEffectEstimate`: from [`wcls`](@ref) or [`emee`](@ref).

# Keywords
- `moderator::Union{Nothing,Symbol} = nothing`: the moderator on the horizontal
  axis (default: the first).
- `values = nothing`: moderator values at which to evaluate the effect (default:
  50 points over its observed range).
- `at = nothing`: a `Dict` or `NamedTuple` of values for the other moderators
  (default: their sample means).
- `level::Real = 0.95`: confidence level of the pointwise band.
- `relative_risk::Bool = false`: for EMEE, show the relative risk ``\\exp(S'\\beta)``
  instead of the log relative risk.
$(_VIZ_COMMON_KW)

# Returns
- `Makie.Figure`. The subtitle states the estimator, the number of participants,
  the reference distribution and the values of the other moderators.

# Examples
```julia
using DrSnow, CairoMakie, DataFrames, StableRNGs
rng = StableRNG(1)
n, T = 25, 20
df = DataFrame(id=repeat(1:n; inner=T), day=repeat(1:T, n) ./ T, x=rand(rng, n * T))
df.a = Float64.(rand(rng, n * T) .< 0.5)                     # randomized with p = 0.5
df.y = df.x .+ df.a .* (0.5 .- 0.5 .* df.day) .+ randn(rng, n * T)
r = wcls(df, :y, :a, :id; rand_prob=0.5, moderators=[:day], controls=[:day, :x])
plot_excursion_effect(r; moderator=:day)
```

# References
- Boruvka, A., Almirall, D., Witkiewitz, K., & Murphy, S. A. (2018). Assessing
  time-varying causal effect moderation in mobile health. *Journal of the American
  Statistical Association*, 113(523), 1112–1121.
- Qian, T., Yoo, H., Klasnja, P., Almirall, D., & Murphy, S. A. (2021). Estimating
  time-varying causal excursion effects in mobile health with binary outcomes.
  *Biometrika*, 108(3), 507–527.
"""
function plot_excursion_effect end
"""
    plot_excursion_effect!(ax, r::ExcursionEffectEstimate; moderator=nothing,
                           values=nothing, at=nothing, level=0.95,
                           relative_risk=false, colors=nothing) -> ax

Draw a causal excursion-effect curve into an existing Makie axis. This is the
mutating counterpart of [`plot_excursion_effect`](@ref), which describes the
display; no legend or subtitle is added.

# Arguments
- `ax::Makie.AbstractAxis`: the axis to draw into.
- `r::ExcursionEffectEstimate`: from [`wcls`](@ref) or [`emee`](@ref).
- Keywords: those of [`plot_excursion_effect`](@ref) except `figure` and `axis`.

# Returns
- `ax`, the axis drawn into.
"""
function plot_excursion_effect! end
plot_excursion_effect(args...; kwargs...) = _viz_no_backend(plot_excursion_effect, args)
plot_excursion_effect!(args...; kwargs...) =
    _viz_no_backend(plot_excursion_effect!, args)

"""
    _viz_assignment_data(log::AdaptiveLog)

NamedTuple with
- `table`: `t::Int` (first unit of the batch for contextual logs, every unit
  otherwise), `arm::Int`, `probability::Float64` (batch mean for contextual logs);
- `floor`: `t`, `floor` (empty when the policy has no floor);
- `burnin::Int`, `K::Int`, `policy::String`, `contextual::Bool`.
"""
function _viz_assignment_data(l::AdaptiveLog)
    T = nobs(l)
    ctx = l.contexts !== nothing
    ts = Int[]
    arms = Int[]
    ps = Float64[]
    if ctx
        for (b, s) in enumerate(l.batch_start)
            u = findall(==(b), l.batch)
            for k in 1:l.K
                push!(ts, s)
                push!(arms, k)
                push!(ps, mean(l.probabilities[u, k]))
            end
        end
    else
        for t in 1:T, k in 1:l.K
            push!(ts, t)
            push!(arms, k)
            push!(ps, l.probabilities[t, k])
        end
    end
    fl = if l.floor > 0
        tt = collect(1:T)
        DataFrame(t=tt, floor=l.floor .* float.(tt) .^ (-l.floor_decay))
    else
        DataFrame(t=Int[], floor=Float64[])
    end
    return (table=DataFrame(t=ts, arm=arms, probability=ps), floor=fl,
            burnin=l.burnin, K=l.K, policy=l.policy, contextual=ctx)
end

"""
    _viz_excursion_data(r::ExcursionEffectEstimate; moderator=nothing, values=nothing,
                        at=nothing, level=0.95, relative_risk=false)

NamedTuple with `table` (`x`, `estimate`, `conf_low`, `conf_high`), `moderator`
(`nothing` for a marginal effect), `level`, `scale` (`:difference`, `:log_rr` or
`:rr`), `method` and `held` (values of the other moderators).
"""
function _viz_excursion_data(r::ExcursionEffectEstimate; moderator=nothing,
                             values=nothing, at=nothing, level::Real=0.95,
                             relative_risk::Bool=false)
    0 < level < 1 || throw(ArgumentError("level must be in (0, 1)"))
    relative_risk && r.method !== :emee &&
        throw(ArgumentError("relative_risk applies to EMEE (binary outcomes) only"))
    mods = Symbol.(r.names[2:end])
    c = critical_value(level, dof_residual(r))
    scale = r.method === :wcls ? :difference : (relative_risk ? :rr : :log_rr)
    tr = relative_risk ? exp : identity
    if isempty(mods)
        moderator === nothing ||
            throw(ArgumentError("the estimate has no moderators"))
        b = r.coef[1]
        s = sqrt(r.vcov[1, 1])
        tab = DataFrame(x=[0.0], estimate=[tr(b)], conf_low=[tr(b - c * s)],
                        conf_high=[tr(b + c * s)])
        return (table=tab, moderator=nothing, level=level, scale=scale,
                method=method_name(r), held=NamedTuple())
    end
    m = moderator === nothing ? mods[1] : Symbol(moderator)
    j = findfirst(==(m), mods)
    j === nothing && throw(ArgumentError("$m is not a moderator of the estimate " *
                                         "($(join(mods, ", ")))"))
    xs = if values === nothing
        lo, hi = r.moderator_ranges[j]
        hi > lo ? collect(range(lo, hi; length=50)) : [lo]
    else
        Float64.(collect(values))
    end
    base = copy(r.moderator_means)
    held = Dict{Symbol,Float64}()
    if at !== nothing
        for (k, v) in pairs(at)
            i = findfirst(==(Symbol(k)), mods)
            i === nothing && throw(ArgumentError("`at`: $k is not a moderator"))
            base[i] = float(v)
        end
    end
    for (i, mm) in enumerate(mods)
        i == j || (held[mm] = base[i])
    end
    est = Float64[]
    lo = Float64[]
    hi = Float64[]
    for x in xs
        s = copy(base)
        s[j] = x
        z = vcat(1.0, s)
        b = dot(z, r.coef)
        se = sqrt(max(dot(z, r.vcov * z), 0.0))
        push!(est, tr(b))
        push!(lo, tr(b - c * se))
        push!(hi, tr(b + c * se))
    end
    return (table=DataFrame(x=xs, estimate=est, conf_low=lo, conf_high=hi),
            moderator=m, level=level, scale=scale, method=method_name(r),
            held=(; held...))
end
