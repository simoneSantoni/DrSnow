# Extrapolating LATEs to other populations by covariate reweighting
# (Angrist & Fernández-Val 2013), nonparametrically over discrete covariate cells.
#
# With a binary instrument that is valid within cells c of discrete covariates X,
# the cell Wald ratio identifies LATE(c). Under *conditional effect ignorability*
# (CEI) — within a cell, compliers, always-takers and never-takers have the same
# average treatment effect — LATE(c) is also the average effect of every type in the
# cell, and effects for other populations follow by reweighting:
#     θ(target) = Σ_c ω_c LATE(c) / Σ_c ω_c,
# with ω_c = P(c)·s_c for a target defined by the share s_c of that population in
# cell c (s_c = 1: everyone; P(D=1 | c): treated; complier share: compliers, ...).
# Without CEI only the complier target is identified.
#
# The parametric version (Angrist & Fernández-Val 2013, Section 4) replaces cells by a
# linear model in (possibly continuous) covariates X: under CEI with
# E[Y(0) | X] = X'α and LATE(X) = E[Y(1) − Y(0) | X] = X'δ, the interacted 2SLS of Y on
# (X, D·X) with instruments (X, Z·X) estimates (α, δ); the complier share is
# π(X) = X'π₁ from the interacted linear first stage D ~ (X, Z·X). Targets are
# weighted averages of X'δ with the target population's weights (delta-method SEs
# from the stacked influence functions).

"""
    LATEExtrapolation <: CausalEstimate

Result of [`late_extrapolation`](@ref): average treatment effects for target
populations other than (or including) compliers, obtained by reweighting
covariate-specific LATEs.

The coefficients are the target averages, named after the targets (`compliers`,
`population`, `treated`, `untreated`, `always_takers`, `never_takers`, `external`),
with their joint delta-method covariance. Except for the complier target, they rely on
conditional effect ignorability (Angrist and Fernández-Val 2013), which the data
cannot verify, and should be read as extrapolations. The object supports `coef`,
`vcov`, `stderror`, `confint` and `coeftable`; with clustering the reference
distribution is ``t(G - 1)``.

# Fields
- `coef::Vector{Float64}`, `vcov::Matrix{Float64}`, `coefnames::Vector{String}`:
  target averages, covariance and target names.
- `nobs::Int`, `level::Float64`, `n_clusters::Int` (0 when not clustered).
- `cells::DataFrame`: for `model = :cells`, one row per cell with the sample share,
  instrument share, complier share (the cell first stage), first-stage F, treatment
  rate, cell LATE and its standard error; for `model = :linear`, one row per term of
  ``x`` with the coefficients of ``\\text{LATE}(x) = x'\\delta`` (`late_coef`,
  `late_se`) and of the complier share ``x'\\pi_1`` (`complier_coef`,
  `complier_se`).
- `weights::DataFrame`: for `:cells`, the normalized weight of each cell in each
  target; for `:linear`, a summary of each target's observation weights
  (`min_weight`, `negative_share`), where negative weights indicate fitted type
  shares below zero.
- `cell_columns::Vector{Symbol}`: the cell columns (`:cells`) or the covariates
  (`:linear`).
- `model::Symbol`: `:cells` (nonparametric reweighting of cell LATEs) or `:linear`
  (parametric model with ``\\text{LATE}(x) = x'\\delta``).

# References
- Angrist, J. D., & Fernández-Val, I. (2013). ExtrapoLATE-ing: External validity and
  overidentification in the LATE framework. In D. Acemoglu, M. Arellano, &
  E. Dekel (Eds.), *Advances in Economics and Econometrics: Tenth World Congress*
  (Vol. III, pp. 401–434). Cambridge University Press.
"""
struct LATEExtrapolation <: CausalEstimate
    coef::Vector{Float64}
    vcov::Matrix{Float64}
    coefnames::Vector{String}
    nobs::Int
    level::Float64
    n_clusters::Int
    cells::DataFrame
    weights::DataFrame
    cell_columns::Vector{Symbol}
    model::Symbol
end

StatsAPI.coef(r::LATEExtrapolation) = r.coef
StatsAPI.vcov(r::LATEExtrapolation) = r.vcov
StatsAPI.coefnames(r::LATEExtrapolation) = r.coefnames
StatsAPI.nobs(r::LATEExtrapolation) = r.nobs
StatsAPI.dof_residual(r::LATEExtrapolation) =
    r.n_clusters > 0 ? float(r.n_clusters - 1) : Inf
StatsAPI.confint(r::LATEExtrapolation; level::Real=r.level) =
    invoke(StatsAPI.confint, Tuple{CausalEstimate}, r; level=level)
estimand(r::LATEExtrapolation) = r.model === :linear ?
    "target averages of a linear LATE(x) (non-complier targets require conditional " *
    "effect ignorability)" :
    "reweighted cell LATEs (non-complier targets require conditional effect " *
    "ignorability)"
method_name(::LATEExtrapolation) = "LATE extrapolation (Angrist & Fernández-Val 2013)"

Base.show(io::IO, ::MIME"text/plain", r::LATEExtrapolation) = _iv_show_estimate(io, r)

function show_details(io::IO, r::LATEExtrapolation)
    println(io)
    if r.model === :linear
        println(io, "Parametric model: LATE(x) = x'δ and complier share π(x) = x'π₁, ",
                "x = (1, ", join(r.cell_columns, ", "), ")")
        println(io, "Beyond the linear specification, the `compliers` target needs only " *
                    "instrument validity given the covariates; all other targets " *
                    "additionally assume conditional effect ignorability, which the " *
                    "data cannot verify.")
        return
    end
    println(io, "Cells: ", join(r.cell_columns, " × "), " (", nrow(r.cells), " cells)")
    println(io, "The `compliers` target needs only instrument validity within cells; " *
                "all other targets additionally assume conditional effect ignorability " *
                "(within a cell, compliers, always-takers and never-takers have the same " *
                "average effect), which the data cannot verify.")
end

const _IV_EXTRAP_TARGETS = (:compliers, :population, :treated, :untreated,
                            :always_takers, :never_takers)

"""
    late_extrapolation(data, outcome, treatment, instrument, cells;
                       targets=[:compliers, :population, :treated, :untreated],
                       target_data=nothing, weights=nothing, cluster=nothing,
                       level=0.95) -> LATEExtrapolation

Average treatment effects for populations other than compliers, obtained by
reweighting covariate-cell LATEs (Angrist and Fernández-Val 2013), for a binary
instrument and a binary treatment.

A LATE is specific to the compliers of a given instrument, whereas policy questions
often concern the whole population, the treated, or a population with a different
covariate mix. Angrist and Fernández-Val (2013) link the two through observed
covariates. If the instrument is valid within each cell ``c`` of the discrete
covariates `cells`, the cell Wald ratio identifies ``\\text{LATE}(c)``. Under
**conditional effect ignorability** (CEI), which states that within a cell compliers,
always-takers and never-takers have the same average treatment effect,
``\\text{LATE}(c)`` is also the cell's average effect for everybody, and the effect for
a target population follows by reweighting:

```math
\\theta(\\text{target})
  = \\frac{\\sum_c P(c)\\, s_c\\, \\text{LATE}(c)}{\\sum_c P(c)\\, s_c},
```

where ``s_c`` is the share of the target population in cell ``c``. The targets are
`:compliers` (``s_c`` the cell complier share; identified without CEI and equal to
[`late_ipw`](@ref) with the cells as saturated covariates), `:population`
(``s_c = 1``, the ATE under CEI), `:treated` and `:untreated` (``P(D = 1 \\mid c)``
and ``P(D = 0 \\mid c)``), and `:always_takers` and `:never_takers` (their cell
shares). With `target_data`, a data frame holding the cell columns for another
population, the additional target `:external` uses that population's cell
distribution, treated as known (its sampling error is ignored).

CEI is a substantive assumption: it rules out selection on gains within cells (the
Roy-model pattern in which those who expect larger gains take up treatment regardless
of the instrument) and it is not testable with a single binary instrument; the
non-complier targets are therefore extrapolations whose credibility rests on the
richness of the covariates. Every cell must contain both instrument values and a
non-zero first stage. The instrument is oriented to raise take-up in the pooled
sample; a warning is issued if the first stage is negative in some cells, which
contradicts monotonicity with a common direction. Standard errors come from the
influence functions of the cell means and the delta method (numerical Jacobian),
robust or clustered. They are unreliable when a cell's first stage is weak, because
each cell LATE is a ratio with a noisy denominator; a warning is issued when a cell
first-stage F is below 10, and coarser cells are then advisable. With continuous
covariates use the parametric method with the `covariates` keyword.

# Arguments
- `data::AbstractDataFrame`: the data; incomplete rows are dropped.
- `outcome::Symbol`: the outcome ``Y``.
- `treatment::Symbol`: binary (0/1) treatment.
- `instrument::Symbol`: binary (0/1) instrument.
- `cells`: discrete covariate(s) whose combinations define the cells (a `Symbol` or
  vector of `Symbol`s).

# Keywords
- `targets`: subset of `[:compliers, :population, :treated, :untreated,
  :always_takers, :never_takers]` (default the first four).
- `target_data::Union{Nothing,AbstractDataFrame}`: data on an external population
  containing the cell columns (default `nothing`); every cell present there must occur
  in the estimation sample.
- `weights::Union{Nothing,Symbol}`: sampling weights (default `nothing`).
- `cluster`: clustering variable(s) for the standard errors (default `nothing`).
- `level::Real`: default confidence level (default 0.95).

# Returns
- A [`LATEExtrapolation`](@ref) with `model = :cells`; `coeftable(ex)` reports the
  targets, `ex.cells` the cell LATEs and `ex.weights` the cell weights.

# Examples
```julia
using DrSnow, DataFrames, StableRNGs
rng = StableRNG(12)
n = 6_000
g = rand(rng, 1:3, n)                                   # covariate cells
z = rand(rng, 0:1, n)
u = rand(rng, n)
pa = [0.1, 0.2, 0.4][g]                                 # always-taker share by cell
d = ifelse.(u .< pa, 1, ifelse.(u .< pa .+ 0.5, z, 0))
y = randn(rng, n) .+ [1.0, 2.0, 3.0][g] .* d            # effect varies across cells
df = DataFrame(y=y, d=d, z=z, g=g)
ex = late_extrapolation(df, :y, :d, :z, [:g])
coeftable(ex)
ex.cells
```

# References
- Angrist, J. D., & Fernández-Val, I. (2013). ExtrapoLATE-ing: External validity and
  overidentification in the LATE framework. In D. Acemoglu, M. Arellano, &
  E. Dekel (Eds.), *Advances in Economics and Econometrics: Tenth World Congress*
  (Vol. III, pp. 401–434). Cambridge University Press.
- Imbens, G. W., & Angrist, J. D. (1994). Identification and estimation of local
  average treatment effects. *Econometrica*, 62(2), 467–475.
- Heckman, J. J., & Vytlacil, E. (2005). Structural equations, treatment effects, and
  econometric policy evaluation. *Econometrica*, 73(3), 669–738.
"""
function late_extrapolation(data::AbstractDataFrame, outcome::Symbol, treatment::Symbol,
                            instrument::Symbol, cells;
                            targets=[:compliers, :population, :treated, :untreated],
                            target_data::Union{Nothing,AbstractDataFrame}=nothing,
                            weights::Union{Nothing,Symbol}=nothing, cluster=nothing,
                            level::Real=0.95)
    ctx = "late_extrapolation"
    cellcols = _as_symbols(cells)
    isempty(cellcols) && throw(ArgumentError("$ctx: `cells` must name at least one " *
                                             "discrete covariate"))
    tg = Symbol[t for t in _as_symbols(targets)]
    for t in tg
        t in _IV_EXTRAP_TARGETS ||
            throw(ArgumentError("$ctx: unknown target :$t; use $(_IV_EXTRAP_TARGETS)"))
    end
    target_data === nothing || push!(tg, :external)
    isempty(tg) && throw(ArgumentError("$ctx: no targets requested"))
    prep = _iv_binary_prep(data, treatment, instrument, [outcome], Symbol[], weights,
                           cluster, ctx; groupings=cellcols)
    sub = prep.sub
    y = Float64.(sub[!, outcome])
    d, w, n = prep.d, prep.w, prep.n
    z = prep.z
    # orient the instrument to raise take-up (pooled)
    if sum(w .* z .* d) / sum(w .* z) < sum(w .* (1 .- z) .* d) / sum(w .* (1 .- z))
        z = 1 .- z
    end
    keys_ = [Tuple(sub[i, c] for c in cellcols) for i in 1:n]
    levels_ = sort(unique(keys_); by=string)
    cid = Dict(k => j for (j, k) in enumerate(levels_))
    c = [cid[k] for k in keys_]
    C = length(levels_)
    # basic moments θ = [p_{c,z} (2C); mY_{c,z} (2C); mD_{c,z} (2C)], index (c, z)
    ix(cc, zz) = 2 * (cc - 1) + zz + 1
    W = sum(w)
    θ = zeros(6C)
    Φ = zeros(n, 6C)
    for cc in 1:C, zz in 0:1
        sel = (c .== cc) .& (z .== zz)
        sw = sum(w[sel])
        sw > 0 || throw(ArgumentError("$ctx: cell $(levels_[cc]) has no observations " *
                                      "with instrument = $zz; merge or drop cells"))
        j = ix(cc, zz)
        θ[j] = sw / W
        Φ[:, j] .= w .* (sel .- θ[j]) ./ W
        θ[2C + j] = sum(w[sel] .* y[sel]) / sw
        Φ[:, 2C + j] .= w .* sel .* (y .- θ[2C + j]) ./ sw
        θ[4C + j] = sum(w[sel] .* d[sel]) / sw
        Φ[:, 4C + j] .= w .* sel .* (d .- θ[4C + j]) ./ sw
    end
    fs = [θ[4C + ix(cc, 1)] - θ[4C + ix(cc, 0)] for cc in 1:C]
    bad = findall(x -> abs(x) < 1e-10, fs)
    isempty(bad) || throw(ArgumentError("$ctx: zero first stage in cell(s) " *
                                        "$(levels_[bad]); the cell LATE is not " *
                                        "identified — merge cells"))
    any(<(0), fs) && @warn "$ctx: the first stage is negative in some cells " *
                           "($(count(<(0), fs)) of $C), which contradicts monotonicity " *
                           "with a common direction"
    q = if target_data === nothing
        nothing
    else
        require_columns(target_data, cellcols; context="$ctx target_data")
        tk = [Tuple(target_data[i, cc] for cc in cellcols) for i in 1:nrow(target_data)]
        miss = setdiff(unique(tk), levels_)
        isempty(miss) || throw(ArgumentError("$ctx: target_data has cells absent from " *
                                             "the estimation sample: $(miss)"))
        [count(==(k), tk) / length(tk) for k in levels_]
    end
    hfun = θv -> _iv_extrap_targets(θv, C, tg, q)
    est, _ = hfun(θ)
    J = _iv_numjac(t -> first(hfun(t)), θ)
    V = _iv_if_vcov(Φ * J', prep.groups)
    # cell table (cell LATE SEs via the same machinery)
    lates = [(θ[2C + ix(cc, 1)] - θ[2C + ix(cc, 0)]) / fs[cc] for cc in 1:C]
    Jc = _iv_numjac(t -> [(t[2C + ix(cc, 1)] - t[2C + ix(cc, 0)]) /
                          (t[4C + ix(cc, 1)] - t[4C + ix(cc, 0)]) for cc in 1:C], θ)
    Vc = _iv_if_vcov(Φ * Jc', prep.groups)
    pc = [θ[ix(cc, 0)] + θ[ix(cc, 1)] for cc in 1:C]
    ptreat = [(θ[ix(cc, 1)] * θ[4C + ix(cc, 1)] + θ[ix(cc, 0)] * θ[4C + ix(cc, 0)]) /
              pc[cc] for cc in 1:C]
    # cell first-stage strength: F = (FS / se(FS))²
    Jf = zeros(C, 6C)
    for cc in 1:C
        Jf[cc, 4C + ix(cc, 1)] = 1.0
        Jf[cc, 4C + ix(cc, 0)] = -1.0
    end
    Vf = _iv_if_vcov(Φ * Jf', prep.groups)
    cellF = fs .^ 2 ./ diag(Vf)
    if any(<(10), cellF)
        @warn "$ctx: weak first stage in $(count(<(10), cellF)) of $C cell(s) " *
              "(cell first-stage F < 10); cell LATEs are ratios with noisy " *
              "denominators and the delta-method standard errors may be unreliable. " *
              "Consider coarser cells."
    end
    celltab = DataFrame(cell=string.(levels_), share=pc,
                        instrument_share=[θ[ix(cc, 1)] / pc[cc] for cc in 1:C],
                        complier_share=fs, first_stage_F=cellF, treated_share=ptreat,
                        late=lates, late_se=sqrt.(max.(diag(Vc), 0.0)))
    _, wmat = hfun(θ)
    wtab = DataFrame(cell=string.(levels_))
    for (j, t) in enumerate(tg)
        wtab[!, t] = wmat[:, j]
    end
    G = isempty(prep.groups) ? 0 : minimum(maximum.(prep.groups))
    return LATEExtrapolation(est, V, string.(tg), n, float(level), G, celltab, wtab,
                             cellcols, :cells)
end

"""
    late_extrapolation(data, outcome, treatment, instrument; covariates,
                       targets=[:compliers, :population, :treated, :untreated],
                       target_data=nothing, weights=nothing, cluster=nothing,
                       level=0.95) -> LATEExtrapolation

Parametric version of the Angrist and Fernández-Val (2013) extrapolation, in which
covariate-specific LATEs are linear in (possibly continuous) covariates.

When the covariates are continuous or the cells too many, Angrist and Fernández-Val
(2013, Section 4) replace the cell-by-cell estimation by a linear model. With
``x = (1, X')'`` (categorical covariates dummy-coded), the first stage
``E[D \\mid X, Z] = x'\\pi_0 + Z\\, x'\\pi_1`` gives the complier share
``\\pi(x) = x'\\pi_1``, the always-taker share ``x'\\pi_0`` and the never-taker share
``1 - x'(\\pi_0 + \\pi_1)``. Under conditional effect ignorability (CEI) and the linear
specifications ``E[Y(0) \\mid X] = x'\\alpha`` and ``E[Y(1) - Y(0) \\mid X] =
x'\\delta``, the just-identified 2SLS of ``Y`` on ``(x, D\\,x)`` with instruments
``(x, Z\\,x)`` estimates ``(\\alpha, \\delta)``. Each target is then

```math
\\theta = \\frac{E[s(X)\\, x'\\delta]}{E[s(X)]},
```

estimated by its sample analogue, with ``s = \\pi(x)`` for `:compliers` (requiring
only instrument validity given ``X`` and the linear specification of the LATE
function), ``s = 1`` for `:population` (the ATE), ``s = D`` and ``s = 1 - D`` for
`:treated` and `:untreated`, ``s = x'\\pi_0`` for `:always_takers` and
``s = 1 - x'(\\pi_0 + \\pi_1)`` for `:never_takers`; with `target_data`, `:external`
is the average of ``x'\\delta`` over that population (treated as known). With a
saturated set of cell indicators as covariates the estimates coincide with the
nonparametric cell version.

Standard errors are delta-method standard errors from the stacked influence functions
of the first stage, the interacted 2SLS and the target averages (robust, or clustered
with `cluster`). The linear first stage does not constrain the fitted type shares to
``[0, 1]``; a warning is issued when the fitted complier share ``x'\\pi_1`` is not
positive for some observations, since the model then contradicts monotonicity or is
misspecified. The linearity of the LATE function and CEI are assumptions, and the
non-complier targets are extrapolations beyond what the instrument identifies; the
interacted 2SLS is only as reliable as the variation of the first stage with the
covariates.

# Arguments
- `data::AbstractDataFrame`: the data; incomplete rows are dropped.
- `outcome::Symbol`: the outcome ``Y``.
- `treatment::Symbol`: binary (0/1) treatment.
- `instrument::Symbol`: binary (0/1) instrument.

# Keywords
- `covariates`: the covariates of the linear model (required; a `Symbol` or vector).
- `targets`, `target_data`, `weights`, `cluster`, `level`: as in the cell version
  (default targets `[:compliers, :population, :treated, :untreated]`); `target_data`
  must contain the covariates, with categorical levels seen in the estimation sample.

# Returns
- A [`LATEExtrapolation`](@ref) with `model = :linear`; `ex.cells` holds the
  coefficients of ``\\text{LATE}(x)`` and of the complier share.

# Examples
```julia
using DrSnow, DataFrames, StableRNGs
rng = StableRNG(13)
n = 6_000
age = 20 .+ 40 .* rand(rng, n)
z = rand(rng, 0:1, n)
u = rand(rng, n)
d = ifelse.(u .< 0.2, 1, ifelse.(u .< 0.7, z, 0))
y = randn(rng, n) .+ (0.5 .+ 0.05 .* age) .* d          # LATE(x) linear in age
df = DataFrame(y=y, d=d, z=z, age=age)
ex = late_extrapolation(df, :y, :d, :z; covariates=[:age])
coeftable(ex)
ex.cells
```

# References
- Angrist, J. D., & Fernández-Val, I. (2013). ExtrapoLATE-ing: External validity and
  overidentification in the LATE framework. In D. Acemoglu, M. Arellano, &
  E. Dekel (Eds.), *Advances in Economics and Econometrics: Tenth World Congress*
  (Vol. III, pp. 401–434). Cambridge University Press.
- Abadie, A. (2003). Semiparametric instrumental variable estimation of treatment
  response models. *Journal of Econometrics*, 113(2), 231–263.
"""
function late_extrapolation(data::AbstractDataFrame, outcome::Symbol, treatment::Symbol,
                            instrument::Symbol;
                            covariates=Symbol[],
                            targets=[:compliers, :population, :treated, :untreated],
                            target_data::Union{Nothing,AbstractDataFrame}=nothing,
                            weights::Union{Nothing,Symbol}=nothing, cluster=nothing,
                            level::Real=0.95)
    ctx = "late_extrapolation"
    covs = _as_symbols(covariates)
    isempty(covs) && throw(ArgumentError("$ctx: the parametric version needs " *
                                         "`covariates` (or pass cell columns as the " *
                                         "fifth argument for the nonparametric version)"))
    0 < level < 1 || throw(ArgumentError("level must be in (0, 1)"))
    tg = Symbol[t for t in _as_symbols(targets)]
    for t in tg
        t in _IV_EXTRAP_TARGETS ||
            throw(ArgumentError("$ctx: unknown target :$t; use $(_IV_EXTRAP_TARGETS)"))
    end
    target_data === nothing || push!(tg, :external)
    isempty(tg) && throw(ArgumentError("$ctx: no targets requested"))
    prep = _iv_binary_prep(data, treatment, instrument, [outcome], Symbol[], weights,
                           cluster, ctx; groupings=covs)
    sub = prep.sub
    X, Xt, names_ = _iv_extrap_design(sub, covs, target_data, ctx)
    y = Float64.(sub[!, outcome])
    d, w, n = prep.d, prep.w, prep.n
    z = prep.z
    if sum(w .* z .* d) / sum(w .* z) < sum(w .* (1 .- z) .* d) / sum(w .* (1 .- z))
        z = 1 .- z
    end
    q = size(X, 2)
    # first stage: D on (x, z x)
    W1 = hcat(X, z .* X)
    _iv_check_rank(W1, "first-stage regressors (covariates and instrument × " *
                   "covariates)", "; the instrument must vary given the covariates")
    B1 = inv(Symmetric(W1' * (w .* W1)))
    π̂ = B1 * (W1' * (w .* d))
    v = d .- W1 * π̂
    Φπ = (w .* v .* W1) * B1                       # θ̂ − θ ≈ Σᵢ Φᵢ
    # interacted 2SLS: Y on (x, D x) with instruments (x, Z x)
    R = hcat(X, d .* X)
    A = W1' * (w .* R)
    rank(A) == 2q || throw(ArgumentError("$ctx: the interacted 2SLS is not identified " *
                                         "(the first stage does not vary enough with " *
                                         "the covariates)"))
    Ai = inv(A)
    θ̂ = Ai * (W1' * (w .* y))
    u = y .- R * θ̂
    Φθ = (w .* u .* W1) * Ai'
    δ = θ̂[(q + 1):end]
    π1 = π̂[(q + 1):end]
    cs = X * π1
    if any(<=(0), cs)
        @warn "$ctx: the fitted complier share x'π₁ is not positive for " *
              "$(count(<=(0), cs)) of $n observations; the linear first stage " *
              "contradicts monotonicity there or is misspecified"
    end
    sfun(t, π) = t === :compliers ? X * π[(q + 1):end] :
                 t === :population ? ones(n) :
                 t === :treated ? d :
                 t === :untreated ? 1 .- d :
                 t === :always_takers ? X * π[1:q] :
                 t === :never_takers ? 1 .- X * (π[1:q] .+ π[(q + 1):end]) :
                 error("internal: unknown target")
    m = length(tg)
    est = zeros(m)
    Φ = zeros(n, m)
    wtab = DataFrame(target=String[], min_weight=Float64[], negative_share=Float64[])
    for (j, t) in enumerate(tg)
        if t === :external
            xm = vec(mean(Xt; dims=1))
            est[j] = dot(xm, δ)
            Φ[:, j] = Φθ[:, (q + 1):end] * xm
            push!(wtab, ("external", 1 / size(Xt, 1), 0.0))
            continue
        end
        s = sfun(t, π̂)
        S = sum(w .* s)
        abs(S) > 1e-12 * sum(w) ||
            throw(ArgumentError("$ctx: the target population :$t has zero size"))
        xs = X' * (w .* s) ./ S                   # covariate mean of the target
        est[j] = dot(xs, δ)
        Φ[:, j] = w .* s .* (X * δ .- est[j]) ./ S  # sampling of the target average
        Φ[:, j] .+= Φθ[:, (q + 1):end] * xs         # estimation of δ
        if t in (:compliers, :always_takers, :never_takers)   # weights depend on π̂
            J = _iv_numjac(πv -> begin
                               sv = sfun(t, πv)
                               [dot(X' * (w .* sv), δ) / sum(w .* sv)]
                           end, π̂)
            Φ[:, j] .+= Φπ * vec(J)
        end
        push!(wtab, (string(t), minimum(w .* s) / S, sum(w[s .< 0]) / sum(w)))
    end
    V = _iv_if_vcov(Φ, prep.groups)
    Vθ = _iv_if_vcov(Φθ, prep.groups)
    Vπ = _iv_if_vcov(Φπ, prep.groups)
    ctab = DataFrame(term=names_, late_coef=δ,
                     late_se=sqrt.(max.(diag(Vθ)[(q + 1):end], 0.0)),
                     complier_coef=π1,
                     complier_se=sqrt.(max.(diag(Vπ)[(q + 1):end], 0.0)))
    G = isempty(prep.groups) ? 0 : minimum(maximum.(prep.groups))
    return LATEExtrapolation(est, V, string.(tg), n, float(level), G, ctab, wtab,
                             covs, :linear)
end

"""
Covariate design `(1, covariates)` for the parametric extrapolation, with the schema
learned on the estimation sample and applied to `target_data`; collinear columns are
dropped consistently. Returns `(X, Xtarget_or_nothing, column_names)`.
"""
function _iv_extrap_design(sub, covs, target_data, ctx)
    f = make_formula(:__iv_lhs__, covs; intercept=true)
    ts = f.rhs isa Tuple ? f.rhs : (f.rhs,)
    sch = StatsModels.schema(ts, sub)
    rhs = StatsModels.MatrixTerm(StatsModels.apply_schema(ts, sch,
                                                          StatsModels.StatisticalModel))
    tomat(M) = M isa AbstractVector ? reshape(Float64.(M), :, 1) : Matrix{Float64}(M)
    X = tomat(StatsModels.modelcols(rhs, sub))
    nm = StatsModels.coefnames(rhs)
    nm = nm isa AbstractString ? [nm] : collect(String, nm)
    F = qr(X, ColumnNorm())
    dR = abs.(diag(F.R))
    r = count(>(maximum(dR) * max(size(X)...) * eps() * 1e3), dR)
    keep = sort(F.p[1:r])
    Xt = nothing
    if target_data !== nothing
        require_columns(target_data, covs; context="$ctx target_data")
        tdf = dropmissing(target_data[:, covs])
        nrow(tdf) > 0 || throw(ArgumentError("$ctx: target_data has no complete rows"))
        Xt = try
            tomat(StatsModels.modelcols(rhs, tdf))
        catch err
            throw(ArgumentError("$ctx: target_data covariates are incompatible with " *
                                "the estimation sample (e.g. unseen categorical " *
                                "levels): $(sprint(showerror, err))"))
        end
        Xt = Xt[:, keep]
    end
    return X[:, keep], Xt, nm[keep]
end

"""Targets and normalized cell weights as functions of the basic moments."""
function _iv_extrap_targets(θ, C, tg, q)
    ix(cc, zz) = 2 * (cc - 1) + zz + 1
    p1 = [θ[ix(cc, 1)] for cc in 1:C]
    p0 = [θ[ix(cc, 0)] for cc in 1:C]
    pc = p0 .+ p1
    y1 = [θ[2C + ix(cc, 1)] for cc in 1:C]
    y0 = [θ[2C + ix(cc, 0)] for cc in 1:C]
    d1 = [θ[4C + ix(cc, 1)] for cc in 1:C]
    d0 = [θ[4C + ix(cc, 0)] for cc in 1:C]
    fs = d1 .- d0
    late = (y1 .- y0) ./ fs
    ptreat = (p1 .* d1 .+ p0 .* d0) ./ pc
    out = zeros(length(tg))
    wm = zeros(C, length(tg))
    for (j, t) in enumerate(tg)
        s = t === :compliers ? fs :
            t === :population ? ones(C) :
            t === :treated ? ptreat :
            t === :untreated ? 1 .- ptreat :
            t === :always_takers ? d0 :
            t === :never_takers ? 1 .- d1 : ones(C)
        ω = t === :external ? q : pc .* s
        ω = ω ./ sum(ω)
        wm[:, j] = ω
        out[j] = dot(ω, late)
    end
    return out, wm
end

"""Central finite-difference Jacobian of `f` at `x` (rows: outputs)."""
function _iv_numjac(f, x::Vector{Float64})
    f0 = f(x)
    J = zeros(length(f0), length(x))
    for j in eachindex(x)
        h = 1e-6 * max(1.0, abs(x[j]))
        xp = copy(x); xp[j] += h
        xm = copy(x); xm[j] -= h
        J[:, j] = (f(xp) .- f(xm)) ./ (2h)
    end
    return J
end
