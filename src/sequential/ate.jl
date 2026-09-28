# Anytime-valid inference for the average treatment effect in a sequentially
# observed randomized experiment: asymptotic confidence sequence on AIPW
# pseudo-outcomes whose nuisance estimates are fitted on past data only (predictable
# plug-ins; Waudby-Smith, Arbour, Sinha, Kennedy & Ramdas 2024, Sec. 3).
#
# Nuisance learners come from the ml area, which is loaded after this one: the
# learner calls (`fitpredict`, `fitpredict_proba`) are resolved at run time, so any
# `NuisanceLearner` (or any object with these methods) can be passed.

"""
    ATEMonitor(; propensity=nothing, outcome_learner=nothing,
               propensity_learner=nothing, refit_every=100, min_train=20,
               trim=0.01, level=0.95, t_opt, null=0.0, min_n=20,
               running_intersection=true, record=true,
               rng=Random.default_rng()) -> ATEMonitor

Streaming anytime-valid confidence sequence for the average treatment effect of a
binary treatment in a sequentially observed experiment.

Units ``i = 1, 2, \\ldots`` arrive in order, each with covariates ``X_i``, a treatment
``D_i \\in \\{0, 1\\}`` and an outcome ``Y_i = D_i Y_i(1) + (1 - D_i) Y_i(0)``. The
estimand is the average treatment effect ``\\tau = E[Y_i(1) - Y_i(0)]`` in the
superpopulation from which the units are drawn i.i.d. Following Waudby-Smith et al.
(2024), each arriving unit contributes the augmented inverse-probability-weighted
(AIPW) pseudo-outcome

```math
\\varphi_i = \\hat\\mu_1(X_i) - \\hat\\mu_0(X_i)
  + \\frac{D_i\\{Y_i - \\hat\\mu_1(X_i)\\}}{\\hat\\pi(X_i)}
  - \\frac{(1 - D_i)\\{Y_i - \\hat\\mu_0(X_i)\\}}{1 - \\hat\\pi(X_i)},
```

where the outcome models ``\\hat\\mu_0, \\hat\\mu_1`` and the propensity
``\\hat\\pi`` are fitted **on units that arrived earlier only** (predictable
plug-ins). The asymptotic confidence sequence of [`MeanMonitor`](@ref) is then
applied to ``\\varphi_1, \\varphi_2, \\ldots``. When the assignment probabilities are
known (a randomized experiment), each ``\\varphi_i`` has conditional mean ``\\tau``
given the past whatever the outcome models, because the plug-ins are fixed before
unit ``i`` arrives; the outcome models then affect only the width. A good regression
adjustment shrinks the confidence sequence, a poor one widens it but cannot bias it.

**Assumptions and inference.** The guarantee is that of the asymptotic confidence
sequence: time-uniform coverage holds approximately, for i.i.d. arrivals, a
consistent running variance of the pseudo-outcomes and monitoring that starts late
enough (`min_n`); it is not a finite-sample guarantee. With a known constant
propensity the identifying assumption is random assignment, which the design
guarantees. The default propensity, the running treated share
``(1 + \\sum_{j<i} D_j)/(i + 1)``, is appropriate for simple (Bernoulli) randomization
with a constant but unrecorded probability; it converges to the true probability,
and the pseudo-outcomes are then unbiased only asymptotically. A `propensity_learner`
is for observational streams: validity then additionally requires unconfoundedness,
overlap and consistent nuisance estimates, none of which the data can confirm. Known
per-unit propensities (passed to `fit!`) allow stratified or time-varying designs;
with adaptively chosen propensities that approach zero the pseudo-outcomes become
heavy-tailed and the asymptotic approximation can be poor, so the adaptive area's
estimators ([`adaptive_arm_values`](@ref)) are preferable there.

**Nuisance estimation.** With `outcome_learner = nothing` the outcome model is the
running arm means (a difference-in-means estimator updated after every unit).
Otherwise any [`NuisanceLearner`](@ref) (e.g. `OLSLearner()`, `ForestLearner()`,
`MLJLearner(model)`) is fitted separately in each arm on all earlier units. With
learners, the models are refitted every `refit_every` units: all units of a block use
models fitted on the units before the block, so the confidence sequence is updated
block by block (`snapshot(m).n_pending` counts the units not yet processed, and
`fit!(m)` with no data processes them now). Until each arm has `min_train` earlier
units, the arm means are used. Compared with a two-sample mixture SPRT
([`MSPRTMonitor`](@ref)), this monitor has a formal (asymptotic) guarantee without
Gaussian outcomes and can use covariates for precision.

# Keywords
- `propensity = nothing`: a known constant assignment probability in ``(0, 1)``;
  `nothing` uses per-unit probabilities passed to `fit!`, a `propensity_learner`, or
  the running treated share, in that order of precedence.
- `outcome_learner = nothing`: a [`NuisanceLearner`](@ref) for the arm-specific
  outcome regressions, or `nothing` for running arm means.
- `propensity_learner = nothing`: a classifier for the propensity (observational
  use); cannot be combined with a known `propensity`.
- `refit_every::Integer = 100`: block length between refits of the learners. Smaller
  blocks update the confidence sequence more often at a higher computational cost.
- `min_train::Integer = 20`: earlier units required in each arm before the learners
  are used.
- `trim::Real = 0.01`: estimated propensities are clipped to
  ``[\\text{trim}, 1 - \\text{trim}]``; known propensities are not clipped.
- `level::Real = 0.95`: coverage ``1-\\alpha``.
- `t_opt`: required; the sample size at which the confidence sequence is tightest,
  normally the planned number of units, fixed before the data are seen.
- `null::Real = 0.0`: null value of the e-process and the always-valid p-value.
- `min_n::Integer = 20`: number of units before monitoring starts (see
  [`MeanMonitor`](@ref)).
- `running_intersection::Bool = true`: report the running intersection of the
  intervals.
- `record::Bool = true`: keep the path for [`confidence_sequence`](@ref).
- `rng::AbstractRNG = Random.default_rng()`: source of the seeds of the learners'
  internal randomness (one seed per block, drawn in order), for reproducibility.

# Returns
- `ATEMonitor`, a [`SequentialMonitor`](@ref).

# Examples
```julia
using DrSnow, StableRNGs
rng = StableRNG(12)
m = ATEMonitor(; propensity=0.5, outcome_learner=OLSLearner(), t_opt=2000,
               rng=StableRNG(1))
for i in 1:2000
    x = randn(rng, 2)
    d = rand(rng) < 0.5
    y = 0.2 * d + x[1] - 0.5 * x[2] + randn(rng)
    fit!(m, y, d, x)
end
fit!(m)                              # process the last partial block
cs = confidence_sequence(m)
confint(cs)
```

# References
- Waudby-Smith, I., Arbour, D., Sinha, R., Kennedy, E. H., & Ramdas, A. (2024).
  Time-uniform central limit theory and asymptotic confidence sequences. *Annals of
  Statistics*, 52(6), 2613–2640.
- Howard, S. R., Ramdas, A., McAuliffe, J., & Sekhon, J. (2021). Time-uniform,
  nonparametric, nonasymptotic confidence sequences. *Annals of Statistics*, 49(2),
  1055–1080.
- Robins, J. M., Rotnitzky, A., & Zhao, L. P. (1994). Estimation of regression
  coefficients when some regressors are not always observed. *Journal of the American
  Statistical Association*, 89(427), 846–866.
"""
mutable struct ATEMonitor <: SequentialMonitor
    level::Float64
    t_opt::Float64
    null::Float64
    running_intersection::Bool
    propensity::Float64                 # NaN: not a known constant
    outcome_learner::Any
    propensity_learner::Any
    refit_every::Int
    min_train::Int
    trim::Float64
    rng::AbstractRNG
    y::Vector{Float64}
    d::Vector{Bool}
    X::Vector{Vector{Float64}}
    pknown::Vector{Float64}             # per-unit known propensity (NaN if none)
    n_done::Int
    engine::_SeqGaussEngine
    phi::Vector{Float64}
    cur_lower::Float64
    cur_upper::Float64
    max_logev::Float64
    path::_SeqPath
end

function ATEMonitor(; propensity=nothing, outcome_learner=nothing,
                    propensity_learner=nothing, refit_every::Integer=100,
                    min_train::Integer=20, trim::Real=0.01, level::Real=0.95,
                    t_opt=nothing, null::Real=0.0, min_n::Integer=20,
                    running_intersection::Bool=true,
                    record::Bool=true, rng::AbstractRNG=Random.default_rng())
    _seq_check_level(level)
    topt = _seq_check_topt(t_opt)
    isnan(topt) && throw(ArgumentError("ATEMonitor needs t_opt, the sample size at " *
                                       "which the confidence sequence should be " *
                                       "tightest (e.g. the planned sample size)"))
    p = NaN
    if propensity !== nothing
        propensity isa Real ||
            throw(ArgumentError("propensity must be a number in (0, 1) or nothing"))
        (0 < propensity < 1) ||
            throw(ArgumentError("propensity must be in (0, 1), got $propensity"))
        propensity_learner === nothing ||
            throw(ArgumentError("give either a known propensity or a " *
                                "propensity_learner, not both"))
        p = float(propensity)
    end
    refit_every >= 1 || throw(ArgumentError("refit_every must be at least 1"))
    min_train >= 1 || throw(ArgumentError("min_train must be at least 1"))
    (0 <= trim < 0.5) || throw(ArgumentError("trim must be in [0, 0.5)"))
    alpha = 1 - level
    return ATEMonitor(float(level), topt, float(null), running_intersection, p,
                      outcome_learner, propensity_learner, Int(refit_every),
                      Int(min_train), float(trim), rng, Float64[], Bool[],
                      Vector{Float64}[], Float64[], 0,
                      _SeqGaussEngine(NaN, _seq_nm_rho(topt, alpha), alpha,
                                      float(null); min_n=min_n),
                      Float64[], -Inf, Inf, 0.0, _SeqPath(record))
end

_seq_uses_blocks(m::ATEMonitor) =
    m.outcome_learner !== nothing || m.propensity_learner !== nothing

function _seq_check_d(d)
    (d == 0 || d == 1) || throw(ArgumentError("treatment must be 0/1, got $d"))
    return d == 1
end

function StatsAPI.fit!(m::ATEMonitor, y::Real, d::Real,
                       x::AbstractVector{<:Real}=Float64[]; propensity=nothing)
    isfinite(y) || throw(ArgumentError("outcomes must be finite, got $y"))
    dd = _seq_check_d(d)
    xv = collect(Float64, x)
    all(isfinite, xv) || throw(ArgumentError("covariates must be finite"))
    if !isempty(m.X) && length(xv) != length(m.X[1])
        throw(DimensionMismatch("expected $(length(m.X[1])) covariates, got " *
                                "$(length(xv))"))
    end
    pk = NaN
    if propensity !== nothing
        (0 < propensity < 1) ||
            throw(ArgumentError("propensity must be in (0, 1), got $propensity"))
        isnan(m.propensity) && m.propensity_learner === nothing ||
            throw(ArgumentError("a per-unit propensity cannot be combined with a " *
                                "constant propensity or a propensity_learner"))
        pk = float(propensity)
    end
    push!(m.y, y); push!(m.d, dd); push!(m.X, xv); push!(m.pknown, pk)
    if !_seq_uses_blocks(m) || length(m.y) - m.n_done >= m.refit_every
        _seq_process!(m)
    end
    return m
end

function StatsAPI.fit!(m::ATEMonitor, y::AbstractVector{<:Real},
                       d::AbstractVector{<:Real},
                       X::Union{Nothing,AbstractMatrix{<:Real}}=nothing;
                       propensity=nothing)
    n = length(y)
    length(d) == n || throw(DimensionMismatch("y and d must have the same length"))
    X === nothing || size(X, 1) == n ||
        throw(DimensionMismatch("X must have one row per observation"))
    propensity === nothing || propensity isa Real || length(propensity) == n ||
        throw(DimensionMismatch("propensity must be a number or one per observation"))
    for i in 1:n
        xi = X === nothing ? Float64[] : view(X, i, :)
        pi_ = propensity === nothing || propensity isa Real ? propensity : propensity[i]
        fit!(m, y[i], d[i], xi; propensity=pi_)
    end
    return m
end

# `fit!(m)`: process the pending units now (models fitted on the units before them).
function StatsAPI.fit!(m::ATEMonitor)
    length(m.y) > m.n_done && _seq_process!(m)
    return m
end

StatsAPI.nobs(m::ATEMonitor) = m.n_done

# Arm means of units 1:k.
function _seq_arm_means(m::ATEMonitor, k::Int)
    s1 = 0.0; n1 = 0; s0 = 0.0; n0 = 0
    @inbounds for j in 1:k
        if m.d[j]
            s1 += m.y[j]; n1 += 1
        else
            s0 += m.y[j]; n0 += 1
        end
    end
    overall = k == 0 ? 0.0 : (s1 + s0) / k
    return (n1 == 0 ? overall : s1 / n1), (n0 == 0 ? overall : s0 / n0), n1, n0
end

# Process units n_done+1 : end as one block with nuisances fitted on 1:n_done.
function _seq_process!(m::ATEMonitor)
    k = m.n_done
    idx = (k + 1):length(m.y)
    isempty(idx) && return m
    mu1_bar, mu0_bar, n1, n0 = _seq_arm_means(m, k)
    nb = length(idx)
    mu1 = fill(mu1_bar, nb)
    mu0 = fill(mu0_bar, nb)
    ps = fill((1 + (k == 0 ? 0 : count(view(m.d, 1:k)))) / (2 + k), nb)
    if _seq_uses_blocks(m)
        seed = rand(m.rng, UInt64)
        p = length(m.X[1])
        Xtr = p == 0 ? zeros(k, 0) : reduce(vcat, (permutedims(m.X[j]) for j in 1:k);
                                          init=zeros(0, p))
        Xnew = p == 0 ? zeros(nb, 0) : reduce(vcat, (permutedims(m.X[j]) for j in idx);
                                            init=zeros(0, p))
        if m.outcome_learner !== nothing && n1 >= m.min_train && n0 >= m.min_train
            tr1 = findall(view(m.d, 1:k))
            tr0 = findall(.!view(m.d, 1:k))
            mu1 = fitpredict(m.outcome_learner, Xtr[tr1, :], m.y[tr1], Xnew;
                             rng=Random.Xoshiro(seed), weights=nothing)
            mu0 = fitpredict(m.outcome_learner, Xtr[tr0, :], m.y[tr0], Xnew;
                             rng=Random.Xoshiro(seed + 0x1), weights=nothing)
        end
        if m.propensity_learner !== nothing && n1 >= m.min_train && n0 >= m.min_train
            ps = fitpredict_proba(m.propensity_learner, Xtr, Float64.(m.d[1:k]), Xnew;
                                  rng=Random.Xoshiro(seed + 0x2), weights=nothing)
        end
        ps = clamp.(ps, m.trim, 1 - m.trim)
    end
    for (b, i) in enumerate(idx)
        p_i = !isnan(m.pknown[i]) ? m.pknown[i] :
              !isnan(m.propensity) ? m.propensity : ps[b]
        yi = m.y[i]
        phi = mu1[b] - mu0[b] +
              (m.d[i] ? (yi - mu1[b]) / p_i : -(yi - mu0[b]) / (1 - p_i))
        push!(m.phi, phi)
        _seq_push!(m.engine, phi)
        l, u = _seq_interval(m.engine)
        if m.running_intersection
            m.cur_lower = max(m.cur_lower, l)
            m.cur_upper = min(m.cur_upper, u)
        else
            m.cur_lower, m.cur_upper = l, u
        end
        lev = _seq_logevalue(m.engine)
        m.max_logev = max(m.max_logev, lev)
        if m.path.record
            _seq_record!(m.path, m.engine.n, m.engine.mean, m.cur_lower, m.cur_upper,
                         _seq_scale(m.engine), exp(lev))
        end
    end
    m.n_done = length(m.y)
    return m
end

function snapshot(m::ATEMonitor)
    n1 = count(view(m.d, 1:m.n_done))
    return (n=m.n_done, estimate=_seq_estimate(m.engine), lower=m.cur_lower,
            upper=m.cur_upper, evalue=exp(_seq_logevalue(m.engine)),
            pvalue=min(1.0, exp(-m.max_logev)), n_treated=n1,
            n_control=m.n_done - n1, n_pending=length(m.y) - m.n_done)
end

function _seq_ate_method(m::ATEMonitor)
    om = m.outcome_learner === nothing ? "arm means" :
         string(nameof(typeof(m.outcome_learner)))
    ps = !isnan(m.propensity) ? "known propensity $(m.propensity)" :
         m.propensity_learner !== nothing ?
         "estimated propensity (" * string(nameof(typeof(m.propensity_learner))) * ")" :
         (any(!isnan, m.pknown) ? "known per-unit propensities" :
          "running treated share")
    return "Sequential AIPW ($om; $ps), asymptotic confidence sequence"
end

function confidence_sequence(m::ATEMonitor)
    m.path.record ||
        throw(ArgumentError("the monitor was created with record = false; no path is " *
                            "stored (use `snapshot` for the current state)"))
    p = m.path
    note = "Nuisances fitted on earlier units only (predictable plug-ins). Asymptotic " *
           "CS: time-uniform coverage holds approximately, for i.i.d. arrivals under " *
           "random assignment" *
           (m.propensity_learner === nothing ? "" :
            " (estimated propensities additionally need unconfoundedness and " *
            "consistent nuisance estimates)") * "."
    m.n_done < length(m.y) &&
        (note *= " $(length(m.y) - m.n_done) pending unit(s) not yet processed.")
    return ConfidenceSequence("ATE", _seq_ate_method(m), :aipw, copy(p.n),
                              copy(p.estimate), copy(p.lower), copy(p.upper),
                              copy(p.sigma), copy(p.evalue), m.null, m.level, m.t_opt,
                              m.engine.rho, m.running_intersection, note)
end

# Rows in arrival order (by the `order` column when given).
function _seq_arrival_order(data, order)
    order === nothing && return collect(1:nrow(data))
    require_columns(data, [order])
    col = data[!, order]
    any(ismissing, col) && throw(ArgumentError("order column :$order has missing values"))
    allunique(col) || throw(ArgumentError("order column :$order has ties; arrival " *
                                          "order must be strict"))
    return sortperm(col)
end

function _seq_numeric_column(data, col::Symbol, what::String)
    v = data[!, col]
    any(ismissing, v) && throw(ArgumentError("$what column :$col has missing values"))
    eltype(v) <: Union{Missing,Real} ||
        throw(ArgumentError("$what column :$col must be numeric"))
    return Float64.(v)
end

"""
    confseq_ate(data, outcome, treatment; covariates=Symbol[], order=nothing,
                propensity=nothing, outcome_learner=nothing,
                propensity_learner=nothing, refit_every=100, min_train=20,
                trim=0.01, level=0.95, t_opt=nothing, null=0.0, min_n=20,
                running_intersection=true, rng=Random.default_rng())
        -> ConfidenceSequence

Anytime-valid confidence sequence for the average treatment effect in a sequentially
observed randomized experiment (an A/B test), computed over the units in arrival
order.

This is the batch interface to [`ATEMonitor`](@ref), which defines the estimand
``\\tau = E[Y(1) - Y(0)]``, the AIPW pseudo-outcomes with predictable (past-only)
nuisance estimates, and the asymptotic confidence sequence of Waudby-Smith et al.
(2024) applied to them. The interval after every unit is valid simultaneously over
all sample sizes, so the path shows what continuous monitoring would have shown, and
the experiment may be stopped as soon as the interval excludes zero or is narrow
enough. The guarantee is asymptotic (see [`ATEMonitor`](@ref)): it presumes i.i.d.
arrivals and random assignment, and for estimated propensities additionally
unconfoundedness and consistent nuisance estimates.

The result depends on the arrival order by construction, because the nuisance models
for each unit are fitted on the earlier units only; with `order` it does not depend on
the row order of `data`. Relative to a fixed-``n`` difference in means or regression
estimate, the confidence sequence is wider at the planned sample size (see
[`ConfidenceSequence`](@ref)); that is the price of the freedom to stop early.

**Default tuning to the realized sample size.** `t_opt = nothing` tunes the boundary
to be tightest at the number of units in `data`. This is appropriate when that number
was fixed in advance; when the data set ends where monitoring stopped, a boundary tuned
to the realized size is data-dependent, and the pre-registered planned sample size
should be passed instead.

# Arguments
- `data`: a table with one row per unit.
- `outcome::Symbol`: numeric outcome column.
- `treatment::Symbol`: 0/1 treatment column.

# Keywords
- `covariates::Vector{Symbol} = Symbol[]`: pre-treatment covariates used by the
  learners (ignored when no learner is given).
- `order = nothing`: column giving the arrival time (strictly increasing after
  sorting, no ties or missing values); `nothing` means the rows are in arrival order.
- `propensity = nothing`: `nothing` (running treated share), a number (known
  constant probability), or a column of known per-unit probabilities.
- `outcome_learner`, `propensity_learner`, `refit_every`, `min_train`, `trim`,
  `rng`: nuisance estimation, see [`ATEMonitor`](@ref).
- `level`, `null`, `min_n`, `running_intersection`: as in [`MeanMonitor`](@ref).
- `t_opt = nothing`: sample size at which the confidence sequence is tightest;
  `nothing` means the number of units (see above).

# Returns
- [`ConfidenceSequence`](@ref) with estimand `"ATE"`; `pvalues` gives the
  always-valid p-value for ``H_0: \\tau =`` `null` and [`stopping`](@ref) the first
  unit at which the interval excluded it.

# Examples
```julia
using DrSnow, DataFrames, StableRNGs
rng = StableRNG(13)
n = 3000
x1, x2 = randn(rng, n), randn(rng, n)
d = Int.(rand(rng, n) .< 0.5)
df = DataFrame(y=0.15 .* d .+ x1 .+ randn(rng, n), d=d, x1=x1, x2=x2,
               arrival=1:n)
cs = confseq_ate(df, :y, :d; propensity=0.5, t_opt=4000)
cs2 = confseq_ate(df, :y, :d; covariates=[:x1, :x2], propensity=0.5, t_opt=4000,
                  outcome_learner=OLSLearner(), order=:arrival, rng=StableRNG(2))
confint(cs), confint(cs2)           # covariates narrow the sequence
stopping(cs2)                       # first n at which the CS excluded 0
```

# References
- Waudby-Smith, I., Arbour, D., Sinha, R., Kennedy, E. H., & Ramdas, A. (2024).
  Time-uniform central limit theory and asymptotic confidence sequences. *Annals of
  Statistics*, 52(6), 2613–2640.
- Howard, S. R., Ramdas, A., McAuliffe, J., & Sekhon, J. (2021). Time-uniform,
  nonparametric, nonasymptotic confidence sequences. *Annals of Statistics*, 49(2),
  1055–1080.
- Johari, R., Koomen, P., Pekelis, L., & Walsh, D. (2022). Always valid inference:
  Continuous monitoring of A/B tests. *Operations Research*, 70(3), 1806–1821.
"""
function confseq_ate(data, outcome::Symbol, treatment::Symbol;
                     covariates::Vector{Symbol}=Symbol[], order=nothing,
                     propensity=nothing, t_opt=nothing, kwargs...)
    require_columns(data, vcat([outcome, treatment], covariates))
    nrow(data) >= 2 || throw(ArgumentError("need at least two units"))
    perm = _seq_arrival_order(data, order)
    y = _seq_numeric_column(data, outcome, "outcome")[perm]
    d = _seq_numeric_column(data, treatment, "treatment")[perm]
    all(v -> v == 0 || v == 1, d) ||
        throw(ArgumentError("treatment column :$treatment must be 0/1"))
    X = isempty(covariates) ? nothing :
        reduce(hcat, [_seq_numeric_column(data, c, "covariate")[perm]
                      for c in covariates])
    known = nothing
    pconst = nothing
    if propensity isa Symbol
        known = _seq_numeric_column(data, propensity, "propensity")[perm]
    elseif propensity !== nothing
        pconst = propensity
    end
    m = ATEMonitor(; propensity=pconst, t_opt=t_opt === nothing ? length(y) : t_opt,
                   kwargs...)
    fit!(m, y, d, X; propensity=known)
    fit!(m)
    return confidence_sequence(m)
end
