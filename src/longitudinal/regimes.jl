# Treatment regimes: a static regime fixes the treatment sequence in advance, a dynamic
# regime chooses each period's treatment from the unit's history. Regimes are
# evaluated on the observed histories of a panel into an n × K matrix of actions.

"""
    TreatmentRegime

Abstract supertype of treatment regimes (strategies, rules) for a binary treatment
over ``K`` periods: [`StaticRegime`](@ref), where the sequence of actions
``\\bar a = (a_1, …, a_K)`` is fixed in advance, and [`DynamicRegime`](@ref), where
the action at period ``t`` is a function ``d_t(H_t)`` of the history observed up to
``t``.

A regime ``d`` defines the counterfactual outcome ``Y^d`` that a unit would have had
if its treatment at every period had been set by ``d``, and the target of the
longitudinal g-methods is its mean ``E[Y^d]`` (see
[`longitudinal_regime`](@ref)). Static regimes such as "always treat" and "never
treat" ([`always_treat`](@ref), [`never_treat`](@ref)) compare sustained strategies;
dynamic regimes such as "treat when the biomarker falls below a threshold" describe
strategies that react to the unit's evolving state, which is what most real policies
do (Robins 1986; Murphy 2003; Hernán and Robins 2020, ch. 19–21).

# References
- Robins, J. M. (1986). A new approach to causal inference in mortality studies with a
  sustained exposure period—application to control of the healthy worker survivor
  effect. *Mathematical Modelling*, 7(9–12), 1393–1512.
- Murphy, S. A. (2003). Optimal dynamic treatment regimes. *Journal of the Royal
  Statistical Society: Series B (Statistical Methodology)*, 65(2), 331–355.
- Hernán, M. A., & Robins, J. M. (2020). *Causal Inference: What If*. Chapman &
  Hall/CRC.
"""
abstract type TreatmentRegime end

"""
    StaticRegime(actions; label=nothing)

A fixed treatment sequence ``\\bar a = (a_1, …, a_K)`` with each ``a_t ∈ \\{0, 1\\}``.

A static regime assigns the same actions to every unit regardless of its history:
``(1, 1, 1)`` is "treat in every period", ``(1, 0, 0)`` is "treat once, then stop".
The mean counterfactual outcome ``E[Y^{\\bar a}]`` under a static regime is the
classical g-formula target (Robins 1986), and contrasts between static regimes are
the parameters of a saturated marginal structural model (Robins, Hernán and
Brumback 2000). Units whose observed treatment history coincides with ``\\bar a``
through period ``t`` are the "followers" through ``t``; inverse-probability and
targeted estimators use only followers' outcomes at each step, so a static regime
that few units follow (for example "always treat" over many periods when treatment
is rare) has little support in the data, which [`regime_weights`](@ref) reports.

# Arguments
- `actions`: a vector of length ``K`` with entries 0 or 1 (integers, floats or
  booleans). The length must equal the number of periods of the data the regime is
  applied to.

# Keywords
- `label = nothing`: the name used for the regime's coefficient; by default
  `"static(a_1,…,a_K)"`.

# Fields
- `actions::Vector{Float64}`: the sequence of actions.
- `label::String`: the coefficient name.

# Examples
```julia
using DrSnow
StaticRegime([1, 1, 1])                 # treat in all three periods
StaticRegime([1, 0, 0]; label="treat once")
always_treat(3) == StaticRegime(ones(3); label="always treat")
```

# References
- Robins, J. M. (1986). A new approach to causal inference in mortality studies with a
  sustained exposure period—application to control of the healthy worker survivor
  effect. *Mathematical Modelling*, 7(9–12), 1393–1512.
- Robins, J. M., Hernán, M. A., & Brumback, B. (2000). Marginal structural models and
  causal inference in epidemiology. *Epidemiology*, 11(5), 550–560.
"""
struct StaticRegime <: TreatmentRegime
    actions::Vector{Float64}
    label::String

    function StaticRegime(actions::AbstractVector; label=nothing)
        a = Float64[Float64(v) for v in actions]
        isempty(a) && throw(ArgumentError("StaticRegime: need at least one period"))
        all(v -> v == 0 || v == 1, a) ||
            throw(ArgumentError("StaticRegime: every action must be 0 or 1"))
        lbl = label === nothing ? "static(" * join(Int.(a), ",") * ")" : string(label)
        return new(a, lbl)
    end
end

Base.:(==)(a::StaticRegime, b::StaticRegime) = a.actions == b.actions && a.label == b.label

"""
    always_treat(K; label="always treat") -> StaticRegime
    never_treat(K; label="never treat") -> StaticRegime

The static regimes that treat (``a_t = 1``) or withhold treatment (``a_t = 0``) in
every one of `K` periods.

Their contrast ``E[Y^{\\bar 1}] - E[Y^{\\bar 0}]`` is the effect of sustained
treatment versus no treatment, the most common target of marginal structural and
g-formula analyses (Robins, Hernán and Brumback 2000); see
[`regime_contrast`](@ref).

# Arguments
- `K::Integer`: number of periods.

# Keywords
- `label::AbstractString`: the coefficient name.

# Returns
- [`StaticRegime`](@ref).

# Examples
```julia
using DrSnow
always_treat(4).actions      # [1.0, 1.0, 1.0, 1.0]
never_treat(4).label         # "never treat"
```

# References
- Robins, J. M., Hernán, M. A., & Brumback, B. (2000). Marginal structural models and
  causal inference in epidemiology. *Epidemiology*, 11(5), 550–560.
"""
always_treat(K::Integer; label::AbstractString="always treat") =
    StaticRegime(ones(K); label=label)

@doc (@doc always_treat)
never_treat(K::Integer; label::AbstractString="never treat") =
    StaticRegime(zeros(K); label=label)

"""
    DynamicRegime(rule; label="dynamic regime")

A treatment regime whose action at each period is a function of the unit's history.

The rule is called once per unit and period with a named tuple `h` describing the
history available when the period's treatment is decided:

- `h.t::Int`: the period (1 to ``K``);
- `h.baseline::NamedTuple`: the baseline covariates of the unit;
- `h.covariates::Vector{NamedTuple}`: the time-varying covariates of periods
  `1:t`, so `h.covariates[end]` holds the current period's values and
  `h.covariates[end - 1]` the previous period's;
- `h.treatment::Vector{Float64}`: the treatments of periods `1:t-1` (empty at
  `t = 1`).

It must return 0 or 1. The rule is evaluated on each unit's *observed* history, as in
the R packages ltmle and lmtp: a unit's prescribed action at ``t`` is
``d_t(H_t)`` with ``H_t`` its actual past, and the unit follows the regime through
``t`` when its observed treatments equal the prescribed ones at every period up to
``t``. The estimand ``E[Y^d]`` is the mean outcome had every unit been treated
according to the rule at every period; because the covariates that enter the rule
are themselves affected by earlier treatment, it is identified by the g-formula and
not by any single-period adjustment (Robins 1986; Murphy 2003; Hernán and Robins
2020, ch. 19).

# Arguments
- `rule`: a function of one argument (the history tuple above) returning 0 or 1.

# Keywords
- `label::AbstractString = "dynamic regime"`: the coefficient name.

# Fields
- `rule`: the rule function.
- `label::String`: the coefficient name.

# Examples
```julia
using DrSnow
# treat from the first period in which the biomarker `x` falls below 0.5, and keep
# treating afterwards
d = DynamicRegime(h -> (h.covariates[end].x < 0.5 || any(==(1), h.treatment)) ? 1 : 0;
                  label="treat when x < 0.5")
```

# References
- Robins, J. M. (1986). A new approach to causal inference in mortality studies with a
  sustained exposure period—application to control of the healthy worker survivor
  effect. *Mathematical Modelling*, 7(9–12), 1393–1512.
- Murphy, S. A. (2003). Optimal dynamic treatment regimes. *Journal of the Royal
  Statistical Society: Series B (Statistical Methodology)*, 65(2), 331–355.
- Hernán, M. A., & Robins, J. M. (2020). *Causal Inference: What If*. Chapman &
  Hall/CRC.
"""
struct DynamicRegime{F} <: TreatmentRegime
    rule::F
    label::String
end

DynamicRegime(rule; label::AbstractString="dynamic regime") =
    DynamicRegime{typeof(rule)}(rule, string(label))

"""Coefficient name of a regime."""
_lg_label(r::TreatmentRegime) = r.label

Base.show(io::IO, r::StaticRegime) =
    print(io, "StaticRegime(", Int.(r.actions), "; label=\"", r.label, "\")")
Base.show(io::IO, r::DynamicRegime) = print(io, "DynamicRegime(\"", r.label, "\")")

# ---------------------------------------------------------------------------
# Evaluation on a panel: n × K matrix of prescribed actions
# ---------------------------------------------------------------------------

function _lg_actions(r::StaticRegime, d, context)
    length(r.actions) == d.K ||
        throw(ArgumentError("$context: regime \"$(r.label)\" has $(length(r.actions)) " *
                            "actions but the data have $(d.K) periods"))
    return repeat(r.actions', d.n)
end

function _lg_actions(r::DynamicRegime, d, context)
    n, K = d.n, d.K
    D = Matrix{Float64}(undef, n, K)
    bnames = Tuple(d.baseline_names)
    cnames = Tuple(d.covariate_names)
    for i in 1:n
        base = NamedTuple{bnames}(Tuple(d.W[i, :]))
        covs = NamedTuple{cnames}[]
        for t in 1:K
            push!(covs, NamedTuple{cnames}(Tuple(d.L[t][i, :])))
            h = (t=t, baseline=base, covariates=covs, treatment=d.A[i, 1:(t - 1)])
            a = r.rule(h)
            (a isa Real && (a == 0 || a == 1)) ||
                throw(ArgumentError("$context: the rule of regime \"$(r.label)\" " *
                                    "returned $(repr(a)) for unit $(d.ids[i]) at " *
                                    "period $(d.times[t]); it must return 0 or 1"))
            D[i, t] = Float64(a)
        end
    end
    return D
end
