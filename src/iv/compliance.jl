# Binary instrument / binary treatment: compliance types, complier characteristics,
# complier potential outcomes and the IPW (κ-weighted) LATE.
#
# Everything here is a smooth function of instrument-arm means computed with
# normalized inverse-propensity weights,
#     μ₁(f) = Σ wᵢ Zᵢ fᵢ / p(Xᵢ) / Σ wᵢ Zᵢ / p(Xᵢ),
#     μ₀(f) = Σ wᵢ (1−Zᵢ) fᵢ / (1−p(Xᵢ)) / Σ wᵢ (1−Zᵢ) / (1−p(Xᵢ)),
# with p(X) = P(Z = 1 | X) from a (weighted) logit on the covariates (intercept only
# when there are none, so μ_z are plain arm means). Under conditional independence
# of Z given X, exclusion and monotonicity (Abadie 2003):
#     P(complier) = μ₁(D) − μ₀(D),  E[f | complier] = (μ₁(fD) − μ₀(fD)) / P(complier)
# and so on. Standard errors use influence functions that include the estimation
# error of the logit, aggregated by observation or by cluster.

"""Complete-case sample and binary checks shared by the binary-IV functions."""
function _iv_binary_prep(data::AbstractDataFrame, treatment::Symbol, instrument::Symbol,
                         extra::Vector{Symbol}, covariates::Vector{Symbol},
                         weights::Union{Nothing,Symbol}, cluster, context::String;
                         groupings::Vector{Symbol}=Symbol[])
    cl = _as_symbols(cluster)
    cols = unique(vcat([treatment, instrument], extra, covariates, cl, groupings,
                       weights === nothing ? Symbol[] : [weights]))
    require_columns(data, cols; context=context)
    _iv_check_numeric(data, vcat([treatment, instrument], extra), context)
    mask = trues(nrow(data))
    for c in cols
        mask .&= .!ismissing.(data[!, c])
    end
    weights === nothing || (mask .&= coalesce.(data[!, weights] .> 0, false))
    sub = disallowmissing(data[mask, cols])
    n = nrow(sub)
    n >= 4 || throw(ArgumentError("$context: too few complete observations ($n)"))
    d = Float64.(sub[!, treatment])
    z = Float64.(sub[!, instrument])
    all(x -> x == 0 || x == 1, d) ||
        throw(ArgumentError("$context: treatment `$treatment` must be binary (0/1)"))
    all(x -> x == 0 || x == 1, z) ||
        throw(ArgumentError("$context: instrument `$instrument` must be binary (0/1)"))
    (0 < sum(z) < n) || throw(ArgumentError("$context: the instrument does not vary"))
    X = _iv_exog_matrix(sub, covariates, true)
    Q, r = _iv_orthobasis(X)
    if r < size(X, 2)
        F = qr(X, ColumnNorm())
        X = X[:, sort(F.p[1:r])]
    end
    w = weights === nothing ? ones(n) : Float64.(sub[!, weights])
    groups = Vector{Int}[_iv_codes(sub[!, c]) for c in cl]
    if !isempty(groups)
        minimum(maximum.(groups)) >= 2 ||
            throw(ArgumentError("$context: need at least 2 clusters"))
    end
    return (sub=sub, d=d, z=z, X=X, w=w, groups=groups, n=n)
end

"""Weighted logit of `z` on `X` by Newton–Raphson; returns (p, H) with H the
negative Hessian `Σ wᵢ pᵢ(1−pᵢ) xᵢxᵢ'`."""
function _iv_logit(X::Matrix{Float64}, z::Vector{Float64}, w::Vector{Float64};
                   context::String="logit")
    q = size(X, 2)
    γ = zeros(q)
    zbar = sum(w .* z) / sum(w)
    γ[1] = log(zbar / (1 - zbar)) * (all(==(1.0), X[:, 1]) ? 1.0 : 0.0)
    p = similar(z)
    H = zeros(q, q)
    converged = false
    for _ in 1:200
        η = X * γ
        p .= 1 ./ (1 .+ exp.(-η))
        g = X' * (w .* (z .- p))
        H = X' * (X .* (w .* p .* (1 .- p)))
        C = cholesky(Symmetric(H); check=false)
        issuccess(C) || break                      # separation: H singular
        step = C \ g
        all(isfinite, step) || break
        γ .+= step
        if maximum(abs, step) < 1e-10 * (1 + maximum(abs, γ))
            converged = true
            break
        end
    end
    η = X * γ
    p .= 1 ./ (1 .+ exp.(-η))
    H = X' * (X .* (w .* p .* (1 .- p)))
    if !converged || any(x -> x < 1e-8 || x > 1 - 1e-8, p)
        throw(ArgumentError("$context: the instrument propensity P(Z=1|X) is (nearly) " *
                            "0 or 1 for some observations (perfect prediction / no " *
                            "overlap); reduce or coarsen the covariates"))
    end
    if minimum(p) < 0.01 || maximum(p) > 0.99
        @warn "$context: estimated instrument propensities range from " *
              "$(round(minimum(p); sigdigits=3)) to $(round(maximum(p); sigdigits=3)); " *
              "weighting estimates may be unstable (limited overlap)"
    end
    return p, Matrix(Symmetric(H))
end

"""
Normalized IPW arm means of the columns of `Fm` and their influence contributions
(`n × m` matrices whose column sums approximate the estimation errors).
"""
function _iv_ipw_means(Fm::Matrix{Float64}, z, w, X, p, H)
    S = X .* (w .* (z .- p))                      # logit scores, n × q
    SH = S / Symmetric(H)                         # n × q, rows Hinv * sᵢ
    a = w .* z ./ p
    A1 = sum(a)
    μ1 = vec(sum(a .* Fm; dims=1)) ./ A1
    R1 = Fm .- μ1'
    G1 = -(X' * (R1 .* (w .* z .* (1 .- p) ./ p)))   # q × m
    φ1 = (a .* R1 .+ SH * G1) ./ A1
    b = w .* (1 .- z) ./ (1 .- p)
    A0 = sum(b)
    μ0 = vec(sum(b .* Fm; dims=1)) ./ A0
    R0 = Fm .- μ0'
    G0 = X' * (R0 .* (w .* (1 .- z) .* p ./ (1 .- p)))
    φ0 = (b .* R0 .+ SH * G0) ./ A0
    return μ1, μ0, φ1, φ0
end

"""Inclusion–exclusion cluster cross-product of influence contributions."""
function _iv_if_vcov(Φ::AbstractMatrix, groups::Vector{Vector{Int}})
    n = size(Φ, 1)
    if isempty(groups)
        return Matrix(Symmetric(Φ' * Φ)) .* (n / (n - 1))
    end
    M = _iv_cluster_sum(Φ, groups)
    G = minimum(maximum.(groups))
    return _iv_psd(M .* (G / (G - 1)))
end

_iv_ipw_label(covariates) = isempty(covariates) ?
    "instrument-arm means (unconditional independence of Z)" :
    "IPW / κ-weighting with logit P(Z=1 | X) on " * join(covariates, ", ")

"""Fit propensity, orient the instrument, return means for `Fm_builder(d)`."""
function _iv_oriented_fit(prep, build::Function, context::String)
    p, H = _iv_logit(prep.X, prep.z, prep.w; context=context)
    μ1, μ0, _, _ = _iv_ipw_means(reshape(prep.d, :, 1), prep.z, prep.w, prep.X, p, H)
    fs = μ1[1] - μ0[1]
    abs(fs) > 1e-12 || throw(ArgumentError("$context: the first stage is zero; " *
                                           "compliers are not identified"))
    z = prep.z
    reversed = fs < 0
    if reversed
        z = 1 .- z
        p, H = 1 .- p, H
    end
    Fm = build(prep.d)
    μ1, μ0, φ1, φ0 = _iv_ipw_means(Fm, z, prep.w, prep.X, p, H)
    return (μ1=μ1, μ0=μ0, φ1=φ1, φ0=φ0, p=p, z=z, reversed=reversed)
end

# ---------------------------------------------------------------------------
# Compliance shares
# ---------------------------------------------------------------------------

"""
    ComplianceAnalysis <: CausalEstimate

Estimated population shares of the compliance types with a binary instrument and a
binary treatment (see [`estimate_compliance`](@ref)).

With potential treatments ``D_i(0), D_i(1)``, units are compliers
(``D_i(1) > D_i(0)``), always-takers (``D_i(0) = D_i(1) = 1``), never-takers
(``D_i(0) = D_i(1) = 0``) or defiers (``D_i(1) < D_i(0)``). Under independence,
exclusion and monotonicity (no defiers) the three remaining shares are identified from
the treatment rates in the two instrument arms (Angrist, Imbens and Rubin 1996). The
coefficients `compliers`, `always_takers` and `never_takers` sum to one by
construction, and their joint covariance is singular in that direction. The object
supports `coef`, `vcov`, `stderror`, `confint` and `coeftable`; with clustering the
reference distribution is ``t(G - 1)``, otherwise the standard normal.

# Fields
- `coef::Vector{Float64}`, `vcov::Matrix{Float64}`, `coefnames::Vector{String}`:
  the three shares, their influence-function covariance and their names.
- `nobs::Int`: number of observations used.
- `level::Float64`: default confidence level.
- `instrument_reversed::Bool`: `true` when the instrument lowered take-up and was
  recoded as ``1 - Z``; "compliers" are then the units that take the treatment only
  when the original instrument equals 0.
- `method::String`: how the instrument-arm means were computed (plain arm means, or
  inverse-propensity weighting on the listed covariates).
- `n_clusters::Int`: number of clusters (0 when not clustered).

# References
- Angrist, J. D., Imbens, G. W., & Rubin, D. B. (1996). Identification of causal
  effects using instrumental variables. *Journal of the American Statistical
  Association*, 91(434), 444–455.
"""
struct ComplianceAnalysis <: CausalEstimate
    coef::Vector{Float64}
    vcov::Matrix{Float64}
    coefnames::Vector{String}
    nobs::Int
    level::Float64
    instrument_reversed::Bool
    method::String
    n_clusters::Int
end

StatsAPI.coef(r::ComplianceAnalysis) = r.coef
StatsAPI.vcov(r::ComplianceAnalysis) = r.vcov
StatsAPI.coefnames(r::ComplianceAnalysis) = r.coefnames
StatsAPI.nobs(r::ComplianceAnalysis) = r.nobs
StatsAPI.dof_residual(r::ComplianceAnalysis) =
    r.n_clusters > 0 ? float(r.n_clusters - 1) : Inf
estimand(::ComplianceAnalysis) = "compliance-type shares (under monotonicity)"
method_name(::ComplianceAnalysis) = "Compliance analysis"

function show_details(io::IO, r::ComplianceAnalysis)
    println(io)
    println(io, "Method: ", r.method)
    r.instrument_reversed &&
        println(io, "The instrument reduces take-up; it was recoded as 1 − Z.")
    println(io, "Shares are identified under independence, exclusion and monotonicity.")
end

"""
    estimate_compliance(data, treatment, instrument; covariates=Symbol[],
                        weights=nothing, cluster=nothing, level=0.95)
        -> ComplianceAnalysis

Shares of compliers, always-takers and never-takers for a binary instrument and a
binary treatment.

In the LATE framework of Imbens and Angrist (1994) and Angrist, Imbens and Rubin
(1996) the population consists of compliance types defined by the potential
treatments ``(D_i(0), D_i(1))``. Individual types are not observed, but under
independence of ``Z`` from potential outcomes and treatments, exclusion and
monotonicity ``D_i(1) \\ge D_i(0)`` their shares are identified:

```math
P(\\text{C}) = E[D \\mid Z=1] - E[D \\mid Z=0], \\quad
P(\\text{A}) = E[D \\mid Z=0], \\quad
P(\\text{N}) = 1 - E[D \\mid Z=1] .
```

The complier share is the first stage, and it measures how large the population is to
which the LATE refers; reporting it (together with [`complier_characteristics`](@ref))
is a simple way to convey the external validity of an IV estimate (Angrist and Pischke
2009; Marbach and Hangartner 2020). Only the complier share is needed for the LATE;
the always- and never-taker shares additionally rely on the absence of defiers, which
the data cannot establish.

When the instrument is as good as randomly assigned only conditional on covariates
``X``, the arm means are replaced by normalized inverse-propensity-weighted means with
weights ``Z/p(X)`` and ``(1 - Z)/(1 - p(X))``, where ``p(X) = P(Z = 1 \\mid X)`` is
estimated by a logit on `covariates`. This yields the unconditional shares implied by
the κ-weighting of Abadie (2003); the logit must be a good approximation to the
instrument propensity, and propensities near 0 or 1 make the weights unstable (a
warning is issued below 0.01 or above 0.99, an error at perfect prediction). If the
instrument lowers take-up it is recoded as ``1 - Z`` (reported in
`instrument_reversed`). Standard errors come from the influence functions of the
weighted means, including the estimation error of the logit, aggregated by observation
(heteroskedasticity-robust) or by cluster.

# Arguments
- `data::AbstractDataFrame`: the data; rows with missing values in used columns, and
  rows with non-positive weights, are dropped.
- `treatment::Symbol`: binary (0/1) treatment ``D``.
- `instrument::Symbol`: binary (0/1) instrument ``Z``; it must vary.

# Keywords
- `covariates::Vector{Symbol}`: covariates conditional on which the instrument is
  independent (default none, so plain arm means are used); they enter the logit for
  ``P(Z = 1 \\mid X)`` linearly, with categorical columns dummy-coded.
- `weights::Union{Nothing,Symbol}`: sampling weights (default `nothing`).
- `cluster`: a `Symbol` or vector of `Symbol`s for cluster-robust standard errors
  (default `nothing`, heteroskedasticity-robust); at least two clusters are required.
- `level::Real`: default confidence level of `confint` and printing (default 0.95).

# Returns
- A [`ComplianceAnalysis`](@ref); `coef(ca)` holds the complier, always-taker and
  never-taker shares, `coeftable(ca)` their standard errors and intervals.

# Examples
```julia
using DrSnow, DataFrames, StableRNGs
rng = StableRNG(1)
n = 4_000
x = rand(rng, 0:1, n)                                   # binary covariate
z = Int.(rand(rng, n) .< ifelse.(x .== 1, 0.7, 0.3))    # P(Z = 1 | X) varies
u = rand(rng, n)
d = ifelse.(u .< 0.2, 1, ifelse.(u .< 0.7, z, 0))       # 20% A, 50% C, 30% N
y = 1 .+ x .+ (2 .+ x) .* d .+ randn(rng, n)
df = DataFrame(y=y, d=d, z=z, x=x)
ca = estimate_compliance(df, :d, :z; covariates=[:x])
coeftable(ca)
```

# References
- Imbens, G. W., & Angrist, J. D. (1994). Identification and estimation of local
  average treatment effects. *Econometrica*, 62(2), 467–475.
- Angrist, J. D., Imbens, G. W., & Rubin, D. B. (1996). Identification of causal
  effects using instrumental variables. *Journal of the American Statistical
  Association*, 91(434), 444–455.
- Abadie, A. (2003). Semiparametric instrumental variable estimation of treatment
  response models. *Journal of Econometrics*, 113(2), 231–263.
- Angrist, J. D., & Pischke, J.-S. (2009). *Mostly Harmless Econometrics: An
  Empiricist's Companion*. Princeton University Press.
- Marbach, M., & Hangartner, D. (2020). Profiling compliers and noncompliers for
  instrumental-variable analysis. *Political Analysis*, 28(3), 435–444.
"""
function estimate_compliance(data::AbstractDataFrame, treatment::Symbol,
                             instrument::Symbol; covariates=Symbol[],
                             weights::Union{Nothing,Symbol}=nothing, cluster=nothing,
                             level::Real=0.95)
    covs = _as_symbols(covariates)
    prep = _iv_binary_prep(data, treatment, instrument, Symbol[], covs, weights, cluster,
                           "estimate_compliance")
    fit = _iv_oriented_fit(prep, d -> reshape(d, :, 1), "estimate_compliance")
    b = [fit.μ1[1] - fit.μ0[1], fit.μ0[1], 1 - fit.μ1[1]]
    Φ = hcat(fit.φ1[:, 1] .- fit.φ0[:, 1], fit.φ0[:, 1], -fit.φ1[:, 1])
    V = _iv_if_vcov(Φ, prep.groups)
    G = isempty(prep.groups) ? 0 : minimum(maximum.(prep.groups))
    return ComplianceAnalysis(b, V, ["compliers", "always_takers", "never_takers"],
                              prep.n, float(level), fit.reversed, _iv_ipw_label(covs), G)
end

# ---------------------------------------------------------------------------
# Complier characteristics
# ---------------------------------------------------------------------------

"""
    ComplierProfile

Mean characteristics of the population and of each compliance type (see
[`complier_characteristics`](@ref)).

The LATE is an average over compliers, a subpopulation that cannot be listed unit by
unit. Its covariate means are nevertheless identified under the LATE assumptions, and
comparing them with the population means shows how the compliers differ from the
population to which a policy conclusion might be extrapolated (Abadie 2003; Angrist
and Pischke 2009; Marbach and Hangartner 2020). The object holds the table of means,
their influence-function standard errors, and the compliance shares on which they are
based.

# Fields
- `table::DataFrame`: one row per variable with columns `variable`,
  `population_mean`, `population_se`, `complier_mean`, `complier_se`,
  `always_taker_mean`, `always_taker_se`, `never_taker_mean`, `never_taker_se`,
  `complier_minus_population`, `difference_se` and `difference_pvalue` (two-sided,
  normal reference). Always-taker (never-taker) entries are `missing` when that type
  has zero estimated share.
- `shares::ComplianceAnalysis`: the compliance-type shares.
- `level::Float64`: confidence level.
- `method::String`: how the instrument-arm means were computed.

# References
- Abadie, A. (2003). Semiparametric instrumental variable estimation of treatment
  response models. *Journal of Econometrics*, 113(2), 231–263.
- Marbach, M., & Hangartner, D. (2020). Profiling compliers and noncompliers for
  instrumental-variable analysis. *Political Analysis*, 28(3), 435–444.
"""
struct ComplierProfile
    table::DataFrame
    shares::ComplianceAnalysis
    level::Float64
    method::String
end

function Base.show(io::IO, ::MIME"text/plain", p::ComplierProfile)
    println(io, "Complier characteristics (", p.method, ")")
    @printf(io, "Complier share: %.4f (se %.4f)\n", p.shares.coef[1],
            sqrt(p.shares.vcov[1, 1]))
    show(io, MIME"text/plain"(), p.table; allrows=true, allcols=true,
         summary=false, eltypes=false)
    println(io)
    println(io, "Means for compliance types are identified under independence, " *
                "exclusion and monotonicity.")
end

"""
    complier_characteristics(data, treatment, instrument, variables;
                             covariates=Symbol[], weights=nothing, cluster=nothing,
                             level=0.95) -> ComplierProfile

Mean characteristics of compliers, always-takers and never-takers for a binary
instrument and a binary treatment.

For a pre-determined characteristic ``X`` (one that neither the instrument nor the
treatment can affect), independence, exclusion and monotonicity identify its mean in
each compliance type. With ``Z`` oriented to raise take-up and ``\\mu_z(f)`` the
(reweighted) mean of ``f`` in instrument arm ``z``,

```math
E[X \\mid \\text{C}] = \\frac{\\mu_1(XD) - \\mu_0(XD)}{\\mu_1(D) - \\mu_0(D)}, \\quad
E[X \\mid \\text{A}] = \\frac{\\mu_0(XD)}{\\mu_0(D)}, \\quad
E[X \\mid \\text{N}] = \\frac{\\mu_1(X(1-D))}{\\mu_1(1-D)} .
```

The complier mean is Abadie's (2003) κ-weighted mean and equals the Wald ratio with
``XD`` as the outcome; always-takers are the treated units in the ``Z = 0`` arm and
never-takers the untreated units in the ``Z = 1`` arm. Comparing complier and
population means shows to whom the LATE applies, which is the first step of any
argument about external validity (Angrist and Pischke 2009; Marbach and Hangartner
2020; see [`late_extrapolation`](@ref) for reweighting LATEs to other populations).

The table reports the complier-minus-population difference with its standard error
and two-sided p-value, from influence functions that include the estimation of the
logit propensity when `covariates` are given; standard errors are
heteroskedasticity-robust or clustered. Under the assumptions every subgroup mean is a
weighted average of observed values; an estimated mean outside the sample range of the
variable is therefore impossible, and a warning is issued, since it points to a
violated assumption or a weak first stage. A regression of ``X`` on ``Z`` is a
different object: it tests instrument balance ([`instrument_balance`](@ref)) and does
not describe compliers.

# Arguments
- `data::AbstractDataFrame`: the data; rows with missing values in used columns are
  dropped.
- `treatment::Symbol`: binary (0/1) treatment.
- `instrument::Symbol`: binary (0/1) instrument.
- `variables`: numeric characteristics to profile (a `Symbol` or vector); they must
  be pre-determined, since the formulas are invalid for variables the instrument or
  the treatment can change.

# Keywords
- `covariates::Vector{Symbol}`: conditioning variables for instrument independence
  (default none); with covariates the arm means are reweighted by a logit
  ``P(Z = 1 \\mid X)``, as in [`estimate_compliance`](@ref).
- `weights::Union{Nothing,Symbol}`: sampling weights (default `nothing`).
- `cluster`: clustering variable(s) for the standard errors (default `nothing`).
- `level::Real`: confidence level stored in the result (default 0.95).

# Returns
- A [`ComplierProfile`](@ref); `prof.table` is the table of means and differences,
  `prof.shares` the compliance shares.

# Examples
```julia
using DrSnow, DataFrames, StableRNGs
rng = StableRNG(2)
n = 4_000
age = rand(rng, 20:60, n)
z = rand(rng, 0:1, n)
u = rand(rng, n)
complier = (u .> 0.2) .& (u .< 0.2 .+ 0.8 .* (age .< 40))   # compliers are young
d = ifelse.(u .< 0.2, 1, ifelse.(complier, z, 0))
df = DataFrame(d=d, z=z, age=age, female=rand(rng, 0:1, n))
prof = complier_characteristics(df, :d, :z, [:age, :female])
prof.table
```

# References
- Abadie, A. (2003). Semiparametric instrumental variable estimation of treatment
  response models. *Journal of Econometrics*, 113(2), 231–263.
- Angrist, J. D., Imbens, G. W., & Rubin, D. B. (1996). Identification of causal
  effects using instrumental variables. *Journal of the American Statistical
  Association*, 91(434), 444–455.
- Angrist, J. D., & Pischke, J.-S. (2009). *Mostly Harmless Econometrics: An
  Empiricist's Companion*. Princeton University Press.
- Marbach, M., & Hangartner, D. (2020). Profiling compliers and noncompliers for
  instrumental-variable analysis. *Political Analysis*, 28(3), 435–444.
"""
function complier_characteristics(data::AbstractDataFrame, treatment::Symbol,
                                  instrument::Symbol, variables;
                                  covariates=Symbol[],
                                  weights::Union{Nothing,Symbol}=nothing,
                                  cluster=nothing, level::Real=0.95)
    vars = _as_symbols(variables)
    isempty(vars) && throw(ArgumentError("complier_characteristics: no variables given"))
    covs = _as_symbols(covariates)
    ctx = "complier_characteristics"
    prep = _iv_binary_prep(data, treatment, instrument, vars, covs, weights, cluster, ctx)
    Xv = hcat([Float64.(prep.sub[!, v]) for v in vars]...)
    m = length(vars)
    # columns: D, 1−D, then for each variable: X·D, X·(1−D)
    build = d -> hcat(d, 1 .- d, Xv .* d, Xv .* (1 .- d))
    fit = _iv_oriented_fit(prep, build, ctx)
    μ1, μ0, φ1, φ0 = fit.μ1, fit.μ0, fit.φ1, fit.φ0
    w = prep.w
    W = sum(w)
    popμ = vec(sum(w .* Xv; dims=1)) ./ W
    popφ = (w .* (Xv .- popμ')) ./ W
    πc = μ1[1] - μ0[1]
    φc = φ1[:, 1] .- φ0[:, 1]
    πa, φa = μ0[1], φ0[:, 1]
    πn, φn = μ1[2], φ1[:, 2]
    rows = NamedTuple[]
    crit = critical_value(level)
    for j in 1:m
        iXD, iX1D = 2 + j, 2 + m + j
        # compliers
        numc = μ1[iXD] - μ0[iXD]
        mc = numc / πc
        φmc = ((φ1[:, iXD] .- φ0[:, iXD]) .- mc .* φc) ./ πc
        # always-takers / never-takers
        ma, φma = πa > 0 ? (μ0[iXD] / πa, (φ0[:, iXD] .- (μ0[iXD] / πa) .* φa) ./ πa) :
                  (missing, nothing)
        mn, φmn = πn > 0 ? (μ1[iX1D] / πn, (φ1[:, iX1D] .- (μ1[iX1D] / πn) .* φn) ./ πn) :
                  (missing, nothing)
        cols = Any[popφ[:, j], φmc, φmc .- popφ[:, j]]
        V = _iv_if_vcov(hcat(cols...), prep.groups)
        sea = φma === nothing ? missing : sqrt(_iv_if_vcov(reshape(φma, :, 1),
                                                           prep.groups)[1, 1])
        sen = φmn === nothing ? missing : sqrt(_iv_if_vcov(reshape(φmn, :, 1),
                                                           prep.groups)[1, 1])
        diff = mc - popμ[j]
        sed = sqrt(V[3, 3])
        push!(rows, (variable=vars[j], population_mean=popμ[j],
                     population_se=sqrt(V[1, 1]), complier_mean=mc,
                     complier_se=sqrt(V[2, 2]), always_taker_mean=ma,
                     always_taker_se=sea, never_taker_mean=mn, never_taker_se=sen,
                     complier_minus_population=diff, difference_se=sed,
                     difference_pvalue=two_sided_pvalue(diff / sed)))
    end
    _iv_warn_out_of_range(rows, Xv, vars, isempty(covs))
    Φs = hcat(φc, φa, -φ1[:, 1])
    G = isempty(prep.groups) ? 0 : minimum(maximum.(prep.groups))
    shares = ComplianceAnalysis([πc, πa, 1 - μ1[1]], _iv_if_vcov(Φs, prep.groups),
                                ["compliers", "always_takers", "never_takers"], prep.n,
                                float(level), fit.reversed, _iv_ipw_label(covs), G)
    return ComplierProfile(DataFrame(rows), shares, float(level), _iv_ipw_label(covs))
end

# ---------------------------------------------------------------------------
# IPW (κ-weighted) LATE and complier potential outcomes
# ---------------------------------------------------------------------------

"""
    IPWLATEEstimate <: CausalEstimate

Result of [`late_ipw`](@ref): the unconditional LATE and the complier means of the
two potential outcomes.

The coefficients are `LATE`, `E[Y(1) | complier]` and `E[Y(0) | complier]`, with a
joint covariance from their influence functions (including the logit step for the
instrument propensity). The LATE is the difference of the two complier means. The
object supports `coef`, `vcov`, `stderror`, `confint` and `coeftable`; with
clustering the reference distribution is ``t(G - 1)``.

# Fields
- `coef::Vector{Float64}`, `vcov::Matrix{Float64}`, `coefnames::Vector{String}`:
  the three estimates, their covariance and names.
- `nobs::Int`, `level::Float64`, `n_clusters::Int` (0 when not clustered).
- `complier_share::Float64`, `complier_share_se::Float64`: the first stage
  ``P(\\text{complier})`` and its standard error.
- `instrument_reversed::Bool`: `true` when the instrument was recoded as ``1 - Z``
  because it lowered take-up.
- `method::String`: plain arm means or IPW on the listed covariates.
- `propensity_range::Tuple{Float64,Float64}`: range of the fitted
  ``P(Z = 1 \\mid X)``, a quick overlap diagnostic.
"""
struct IPWLATEEstimate <: CausalEstimate
    coef::Vector{Float64}
    vcov::Matrix{Float64}
    coefnames::Vector{String}
    nobs::Int
    level::Float64
    n_clusters::Int
    complier_share::Float64
    complier_share_se::Float64
    instrument_reversed::Bool
    method::String
    propensity_range::Tuple{Float64,Float64}
end

StatsAPI.coef(r::IPWLATEEstimate) = r.coef
StatsAPI.vcov(r::IPWLATEEstimate) = r.vcov
StatsAPI.coefnames(r::IPWLATEEstimate) = r.coefnames
StatsAPI.nobs(r::IPWLATEEstimate) = r.nobs
StatsAPI.dof_residual(r::IPWLATEEstimate) =
    r.n_clusters > 0 ? float(r.n_clusters - 1) : Inf
StatsAPI.confint(r::IPWLATEEstimate; level::Real=r.level) =
    invoke(StatsAPI.confint, Tuple{CausalEstimate}, r; level=level)
estimand(::IPWLATEEstimate) = "LATE (unconditional, compliers)"
method_name(::IPWLATEEstimate) = "IPW / κ-weighted Wald estimator"

function show_details(io::IO, r::IPWLATEEstimate)
    println(io)
    println(io, "Method: ", r.method)
    @printf(io, "Complier share: %.4f (se %.4f); P(Z=1|X) range: [%.3f, %.3f]\n",
            r.complier_share, r.complier_share_se, r.propensity_range...)
    r.instrument_reversed &&
        println(io, "The instrument reduces take-up; it was recoded as 1 − Z.")
    indep = occursin("logit", r.method) ?
            "conditional independence of Z given the covariates" : "independence of Z"
    println(io, "Identified under ", indep, ", exclusion, monotonicity and a non-zero " *
                "first stage.")
end

"""
    late_ipw(data, outcome, treatment, instrument; covariates=Symbol[],
             weights=nothing, cluster=nothing, level=0.95) -> IPWLATEEstimate

Unconditional local average treatment effect with a binary instrument that is valid
conditional on covariates, estimated by inverse-propensity (κ) weighting.

In many designs the instrument is as good as randomly assigned only within strata of
covariates ``X`` (e.g. distance to college given region). Under conditional
independence of ``Z`` from potential outcomes and treatments given ``X``, exclusion,
monotonicity and overlap ``0 < p(X) = P(Z = 1 \\mid X) < 1``, the average effect for
all compliers is identified (Abadie 2003; Frölich 2007) as

```math
\\text{LATE} = E[Y(1) - Y(0) \\mid \\text{C}]
  = \\frac{\\mu_1(Y) - \\mu_0(Y)}{\\mu_1(D) - \\mu_0(D)}, \\qquad
\\mu_z(f) = \\frac{E[f \\, 1\\{Z=z\\} / P(Z=z \\mid X)]}{E[1\\{Z=z\\} / P(Z=z \\mid X)]} .
```

This is the Wald version of Abadie's (2003) κ-weighting. It targets a different and
more interpretable parameter than 2SLS with the same covariates: with covariates that
are not saturated, the 2SLS coefficient is a non-negatively weighted average of
conditional LATEs only under strong restrictions (essentially a linear ``E[Z \\mid
X]``; Blandhol, Bonney, Mogstad and Torgovitsky 2026), and even with saturated
covariates its weights are proportional to ``\\operatorname{Var}(Z \\mid X)`` times
the conditional first stage rather than to complier shares, and can be negative when
the first stage changes sign across covariate values (Słoczyński 2020). The function
also reports the complier means of the potential outcomes, ``E[Y(1) \\mid \\text{C}]``
and ``E[Y(0) \\mid \\text{C}]`` (Imbens and Rubin 1997). Without covariates it
reproduces the Wald (2SLS) estimate.

The propensity ``p(X)`` is estimated by a (weighted) logit that is linear in the
covariates, the weights are normalized within each instrument arm (Hájek form), and
the instrument is recoded as ``1 - Z`` if it lowers take-up. Standard errors come from
the influence function including the estimation error of the logit, robust or
clustered. The estimator is consistent only if the logit is correctly specified; a
misspecified propensity or poor overlap (fitted propensities near 0 or 1, reported in
`propensity_range` and warned about below 0.01 or above 0.99) biases the estimate and
destabilizes the weights. With many or high-dimensional covariates, doubly robust
estimators with machine-learned nuisance functions are preferable. The ratio form
inherits the weak-instrument problems of the Wald estimator when the complier share is
small.

# Arguments
- `data::AbstractDataFrame`: the data; incomplete rows are dropped.
- `outcome::Symbol`: the outcome ``Y`` (distinct from treatment and instrument).
- `treatment::Symbol`: binary (0/1) treatment ``D``.
- `instrument::Symbol`: binary (0/1) instrument ``Z``.

# Keywords
- `covariates::Vector{Symbol}`: variables conditional on which the instrument is valid
  (default none); they are the logit regressors, with categorical columns
  dummy-coded.
- `weights::Union{Nothing,Symbol}`: sampling weights (default `nothing`).
- `cluster`: clustering variable(s) for the standard errors (default `nothing`,
  heteroskedasticity-robust).
- `level::Real`: default confidence level (default 0.95).

# Returns
- An [`IPWLATEEstimate`](@ref); `estimate(r)` is the LATE, `coef(r)[2:3]` the
  complier means of ``Y(1)`` and ``Y(0)``, `r.complier_share` the first stage.

# Examples
```julia
using DrSnow, DataFrames, StableRNGs
rng = StableRNG(1)
n = 4_000
x = rand(rng, 0:1, n)
z = Int.(rand(rng, n) .< ifelse.(x .== 1, 0.7, 0.3))    # Z is random given X only
u = rand(rng, n)
d = ifelse.(u .< 0.2, 1, ifelse.(u .< 0.7, z, 0))
y = 1 .+ x .+ (2 .+ x) .* d .+ randn(rng, n)            # complier LATE = 2.5
df = DataFrame(y=y, d=d, z=z, x=x)
r = late_ipw(df, :y, :d, :z; covariates=[:x])
estimate(r), confint(r)
```

# References
- Abadie, A. (2003). Semiparametric instrumental variable estimation of treatment
  response models. *Journal of Econometrics*, 113(2), 231–263.
- Frölich, M. (2007). Nonparametric IV estimation of local average treatment effects
  with covariates. *Journal of Econometrics*, 139(1), 35–75.
- Imbens, G. W., & Rubin, D. B. (1997). Estimating outcome distributions for
  compliers in instrumental variables models. *Review of Economic Studies*, 64(4),
  555–574.
- Blandhol, C., Bonney, J., Mogstad, M., & Torgovitsky, A. (2026). When is TSLS
  actually LATE? *Review of Economic Studies*, advance online publication.
  https://doi.org/10.1093/restud/rdag029
- Słoczyński, T. (2020). When should we (not) interpret linear IV estimands as LATE?
  arXiv:2011.06695.
"""
function late_ipw(data::AbstractDataFrame, outcome::Symbol, treatment::Symbol,
                  instrument::Symbol; covariates=Symbol[],
                  weights::Union{Nothing,Symbol}=nothing, cluster=nothing,
                  level::Real=0.95)
    covs = _as_symbols(covariates)
    outcome in (treatment, instrument) &&
        throw(ArgumentError("late_ipw: outcome must differ from treatment/instrument"))
    prep = _iv_binary_prep(data, treatment, instrument, [outcome], covs, weights, cluster,
                           "late_ipw")
    y = Float64.(prep.sub[!, outcome])
    build = d -> hcat(d, y, y .* d, y .* (1 .- d))
    fit = _iv_oriented_fit(prep, build, "late_ipw")
    μ1, μ0, φ1, φ0 = fit.μ1, fit.μ0, fit.φ1, fit.φ0
    πc = μ1[1] - μ0[1]
    φc = φ1[:, 1] .- φ0[:, 1]
    late = (μ1[2] - μ0[2]) / πc
    φlate = ((φ1[:, 2] .- φ0[:, 2]) .- late .* φc) ./ πc
    m1 = (μ1[3] - μ0[3]) / πc
    φm1 = ((φ1[:, 3] .- φ0[:, 3]) .- m1 .* φc) ./ πc
    m0 = (μ0[4] - μ1[4]) / πc
    φm0 = ((φ0[:, 4] .- φ1[:, 4]) .- m0 .* φc) ./ πc
    V = _iv_if_vcov(hcat(φlate, φm1, φm0), prep.groups)
    vc = _iv_if_vcov(reshape(φc, :, 1), prep.groups)[1, 1]
    G = isempty(prep.groups) ? 0 : minimum(maximum.(prep.groups))
    return IPWLATEEstimate([late, m1, m0], V,
                           ["LATE", "E[Y(1) | complier]", "E[Y(0) | complier]"], prep.n,
                           float(level), G, πc, sqrt(vc), fit.reversed,
                           _iv_ipw_label(covs), extrema(fit.p))
end

"""
    complier_outcome_distribution(data, outcome, treatment, instrument;
                                  points=nothing, covariates=Symbol[],
                                  weights=nothing, cluster=nothing) -> DataFrame

Cumulative distribution functions of the potential outcomes ``Y(1)`` and ``Y(0)``
for compliers, with pointwise standard errors.

Imbens and Rubin (1997) showed that the LATE assumptions identify not only the mean
effect for compliers but the entire marginal distributions of their potential
outcomes. Applying the Wald ratio to the indicator ``1\\{Y \\le y\\}`` gives, with ``Z``
oriented to raise take-up and ``\\mu_z`` the (reweighted) arm means,

```math
F^{C}_1(y) = \\frac{\\mu_1(1\\{Y \\le y\\} D) - \\mu_0(1\\{Y \\le y\\} D)}{P(\\text{C})},
\\qquad
F^{C}_0(y) = \\frac{\\mu_0(1\\{Y \\le y\\}(1-D)) - \\mu_1(1\\{Y \\le y\\}(1-D))}
{P(\\text{C})} .
```

Differences between the two functions describe distributional effects for compliers;
their horizontal differences at a quantile are the local quantile treatment effects
(Abadie 2002; Frölich and Melly 2013), available with machine-learned nuisances in
[`dml_lqte`](@ref) and [`dml_complier_cdf`](@ref). With `covariates`, the arm means are
reweighted by a logit ``P(Z = 1 \\mid X)`` as in [`late_ipw`](@ref).

The estimates are sample analogues and are not constrained to be monotone or to lie
in ``[0, 1]``. Under the assumptions both functions are distribution functions, so a
material decrease is evidence against the joint validity of independence, exclusion
and monotonicity, which [`instrument_validity_test`](@ref) (Kitagawa 2015) tests
formally. Standard errors are pointwise, from influence functions (robust or
clustered); they do not give uniform confidence bands.

# Arguments
- `data::AbstractDataFrame`: the data; incomplete rows are dropped.
- `outcome::Symbol`: the outcome ``Y``.
- `treatment::Symbol`: binary (0/1) treatment.
- `instrument::Symbol`: binary (0/1) instrument.

# Keywords
- `points`: evaluation points ``y`` (default `nothing`: the 5th, 10th, …, 95th
  percentiles of the pooled outcome).
- `covariates`, `weights`, `cluster`: as in [`late_ipw`](@ref) (defaults: none).

# Returns
- A `DataFrame` with columns `y`, `cdf_treated`, `se_treated`, `cdf_untreated` and
  `se_untreated`, where "treated" refers to ``Y(1)`` and "untreated" to ``Y(0)``,
  both for compliers.

# Examples
```julia
using DrSnow, DataFrames, StableRNGs
rng = StableRNG(3)
n = 4_000
z = rand(rng, 0:1, n)
u = rand(rng, n)
d = ifelse.(u .< 0.2, 1, ifelse.(u .< 0.7, z, 0))
y = randn(rng, n) .+ d .* (1 .+ randn(rng, n))          # effect shifts and spreads
df = DataFrame(y=y, d=d, z=z)
cdfs = complier_outcome_distribution(df, :y, :d, :z; points=-2:0.5:3)
```

# References
- Imbens, G. W., & Rubin, D. B. (1997). Estimating outcome distributions for
  compliers in instrumental variables models. *Review of Economic Studies*, 64(4),
  555–574.
- Abadie, A. (2002). Bootstrap tests for distributional treatment effects in
  instrumental variable models. *Journal of the American Statistical Association*,
  97(457), 284–292.
- Abadie, A. (2003). Semiparametric instrumental variable estimation of treatment
  response models. *Journal of Econometrics*, 113(2), 231–263.
- Frölich, M., & Melly, B. (2013). Unconditional quantile treatment effects under
  endogeneity. *Journal of Business & Economic Statistics*, 31(3), 346–357.
- Kitagawa, T. (2015). A test for instrument validity. *Econometrica*, 83(5),
  2043–2063.
"""
function complier_outcome_distribution(data::AbstractDataFrame, outcome::Symbol,
                                       treatment::Symbol, instrument::Symbol;
                                       points=nothing, covariates=Symbol[],
                                       weights::Union{Nothing,Symbol}=nothing,
                                       cluster=nothing)
    covs = _as_symbols(covariates)
    ctx = "complier_outcome_distribution"
    prep = _iv_binary_prep(data, treatment, instrument, [outcome], covs, weights, cluster,
                           ctx)
    y = Float64.(prep.sub[!, outcome])
    pts = points === nothing ? quantile(y, 0.05:0.05:0.95) : float.(collect(points))
    isempty(pts) && throw(ArgumentError("$ctx: no evaluation points"))
    Ind = Float64.(y .<= pts')                    # n × G
    build = d -> hcat(d, Ind .* d, Ind .* (1 .- d))
    fit = _iv_oriented_fit(prep, build, ctx)
    μ1, μ0, φ1, φ0 = fit.μ1, fit.μ0, fit.φ1, fit.φ0
    πc = μ1[1] - μ0[1]
    φc = φ1[:, 1] .- φ0[:, 1]
    G = length(pts)
    out = DataFrame(y=pts, cdf_treated=zeros(G), se_treated=zeros(G),
                    cdf_untreated=zeros(G), se_untreated=zeros(G))
    for g in 1:G
        i1, i0 = 1 + g, 1 + G + g
        F1 = (μ1[i1] - μ0[i1]) / πc
        φF1 = ((φ1[:, i1] .- φ0[:, i1]) .- F1 .* φc) ./ πc
        F0 = (μ0[i0] - μ1[i0]) / πc
        φF0 = ((φ0[:, i0] .- φ1[:, i0]) .- F0 .* φc) ./ πc
        V = _iv_if_vcov(hcat(φF1, φF0), prep.groups)
        out.cdf_treated[g] = F1
        out.se_treated[g] = sqrt(V[1, 1])
        out.cdf_untreated[g] = F0
        out.se_untreated[g] = sqrt(V[2, 2])
    end
    return out
end

# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

# The generic `CausalEstimate` display in core passes the coefficient table to the
# 2-argument `show`, which prints the raw `CoefTable` struct; the IV result types
# use the text/plain rendering instead (with their own default `level`).
Base.show(io::IO, ::MIME"text/plain",
          r::Union{IVEstimate,ComplianceAnalysis,IPWLATEEstimate}) =
    _iv_show_estimate(io, r)

function _iv_show_estimate(io::IO, r::CausalEstimate)
    header = method_name(r)
    e = estimand(r)
    isempty(e) || (header *= " — estimand: " * e)
    println(io, header)
    println(io, "Observations: ", StatsAPI.nobs(r))
    show(io, MIME"text/plain"(), StatsAPI.coeftable(r; level=r.level))
    show_details(io, r)
end

# Warn when an estimated subgroup mean lies outside the sample range of the variable.
# Under instrument independence and monotonicity every subgroup mean is a weighted
# average of observed values, so this signals a violated assumption or a weak first
# stage rather than a meaningful profile.
function _iv_warn_out_of_range(rows, Xv, vars, no_covariates::Bool)
    bad = String[]
    for (j, r) in enumerate(rows)
        lo, hi = extrema(view(Xv, :, j))
        tol = 1e-8 * max(1.0, hi - lo)
        for (lab, v) in (("compliers", r.complier_mean),
                         ("always-takers", r.always_taker_mean),
                         ("never-takers", r.never_taker_mean))
            v === missing && continue
            (v < lo - tol || v > hi + tol) && push!(bad, "$(vars[j]) ($lab)")
        end
    end
    isempty(bad) && return nothing
    hint = no_covariates ?
        " If the instrument is only as good as random conditional on covariates, " *
        "pass them via `covariates` (κ-weighting)." : ""
    @warn "Estimated subgroup means lie outside the observed range for: " *
          join(bad, ", ") * ". This is impossible under instrument independence and " *
          "monotonicity, and indicates a violated assumption or a weak first stage." * hint
    return nothing
end
