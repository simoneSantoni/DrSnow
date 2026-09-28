# Extended two-way fixed effects (Wooldridge 2021, 2023, 2025): a saturated regression
# with one treatment dummy per treated cohort × period cell (and, with covariates,
# their interactions with covariates demeaned by cohort), estimated by pooled OLS
# with cohort (or unit) and period fixed effects, or by Poisson / logit quasi-maximum
# likelihood with cohort and period dummies. The design mirrors the R package etwfe
# (etwfe() + emfx()): cell effects are averaged over the treated observations of
# each cell and aggregated with delta-method standard errors.

"""
    ETWFEEstimate <: CausalEstimate

Cohort × period average treatment effects ``ATT(g,t)`` from Wooldridge's extended
two-way fixed effects regression, returned by [`did_etwfe`](@ref).

Each cell effect is the average, over the cell's treated observations, of the change
in the fitted conditional mean when the cell's treatment terms are switched on; for
the linear model without covariates this is the cohort × period coefficient
``\\tau_{gt}``. The covariance of the cell effects is obtained by the delta method
from the covariance of the model coefficients, treating the covariate values within
each cell as fixed. Summarize the cells with [`aggregate_att`](@ref) (`:simple`,
`:group`, `:calendar` or `:dynamic`).

# Fields
- `cohorts::Vector{Int}`, `times::Vector{Int}`: cohort and period of each cell as
  period indices into `periods`.
- `coef::Vector{Float64}`, `vcov::Matrix{Float64}`: cell effects and their
  delta-method covariance.
- `cell_weight::Vector{Float64}`: (weighted) number of observations per cell, used
  by the aggregations.
- `gradient::Matrix{Float64}`, `vcov_model::Matrix{Float64}`: Jacobian of the cell
  effects with respect to the model coefficients involved, and the covariance of
  those coefficients.
- `periods::Vector`: time labels.
- `nobs::Int`, `dof::Float64`, `n_clusters::Int`: observations, degrees of freedom
  of the t reference (``G - 1`` for the clustered linear model, `Inf` for the
  nonlinear models) and clusters.
- `settings::NamedTuple`: `control_group`, `family`, `covariates`, `fe`.
- `model`: the fitted `FixedEffectModel` (linear) or a `NamedTuple` with the
  coefficient vector, design column names and covariance (Poisson and logit).

# Accessors
`coef`, `vcov`, `stderror`, `coefnames`, `coeftable`, `confint(r; level)`, `nobs`,
`dof_residual`, [`estimate`](@ref) (the simple aggregated ATT) and
[`aggregate_att`](@ref).

# References
- Wooldridge, J. M. (2025). Two-way fixed effects, the two-way Mundlak regression,
  and difference-in-differences estimators. *Empirical Economics*, 69(5),
  2545–2587.
"""
struct ETWFEEstimate <: CausalEstimate
    cohorts::Vector{Int}
    times::Vector{Int}
    coef::Vector{Float64}
    vcov::Matrix{Float64}
    cell_weight::Vector{Float64}
    gradient::Matrix{Float64}
    vcov_model::Matrix{Float64}
    periods::Vector
    nobs::Int
    dof::Float64
    n_clusters::Int
    settings::NamedTuple
    model::Any
end

StatsAPI.coef(r::ETWFEEstimate) = r.coef
StatsAPI.vcov(r::ETWFEEstimate) = r.vcov
StatsAPI.nobs(r::ETWFEEstimate) = r.nobs
StatsAPI.dof_residual(r::ETWFEEstimate) = r.dof
StatsAPI.coefnames(r::ETWFEEstimate) =
    ["ATT(g=$(r.periods[g]), t=$(r.periods[t]))" for (g, t) in zip(r.cohorts, r.times)]
estimand(r::ETWFEEstimate) = "ATT(g,t): average effect in period t for cohort g" *
    (r.settings.family === :gaussian ? "" : " (on the response scale)")
method_name(r::ETWFEEstimate) =
    "Wooldridge extended TWFE ($(r.settings.family), " *
    "$(r.settings.control_group) controls)"

"""
    estimate(r::ETWFEEstimate) -> Float64

The simple aggregated ATT of an extended TWFE estimate: the observation-weighted
average of the post-treatment cell effects.

# Arguments
- `r::ETWFEEstimate`: cell effects from [`did_etwfe`](@ref).

# Returns
- `Float64`: `estimate(aggregate_att(r, :simple))`; use [`aggregate_att`](@ref) for
  its standard error.
"""
estimate(r::ETWFEEstimate) = estimate(aggregate_att(r, :simple))

function show_details(io::IO, r::ETWFEEstimate)
    println(io)
    isempty(r.settings.covariates) ||
        println(io, "Covariates (demeaned by cohort): ", join(r.settings.covariates, ", "))
    r.n_clusters > 0 && println(io, "Clusters: ", r.n_clusters)
    println(io, "Use aggregate_att(r, :simple / :group / :calendar / :dynamic) to ",
            "summarize.")
end

const _DID_ETWFE_FAMILIES = (:gaussian, :poisson, :logit)

"""
    did_etwfe(data, outcome, treatment, unit, time; covariates=Symbol[],
              control_group=:not_yet_treated, family=:gaussian, fe=:cohort,
              weights=nothing, cluster=unit, vcov=nothing) -> ETWFEEstimate
    did_etwfe(panel::TreatmentPanel; kwargs...) -> ETWFEEstimate

Extended two-way fixed effects (ETWFE) estimator of Wooldridge (2025) for staggered
adoption, with the nonlinear (Poisson and logit) extension of Wooldridge (2023).

The estimands are the cohort × period effects
``ATT(g,t) = E[Y_t(g) - Y_t(\\infty) \\mid G = g]`` for ``t \\ge g``. Wooldridge
(2025) shows that the problems of the single-coefficient TWFE regression disappear
once the treatment indicator is fully interacted with cohort and period dummies. In
the linear model without covariates the regression is

```math
Y_{it} = \\alpha_{G_i} + \\lambda_t
       + \\sum_{g} \\sum_{s \\ge g} \\tau_{gs}\\, 1\\{G_i = g\\}\\, 1\\{t = s\\}
       + \\varepsilon_{it},
```

with cohort effects ``\\alpha_{G_i}`` (or unit effects with `fe = :unit`), period
effects ``\\lambda_t``, and one coefficient ``\\tau_{gs}`` per treated cohort–period
cell, each of which estimates ``ATT(g,s)``. Covariates enter as in Wooldridge
(2025, 2023) and R's `etwfe`: demeaned within cohort and interacted with every
treatment cell, plus covariate × cohort and covariate × period terms, which allows
conditional parallel trends with trends that are linear in the covariates. The
covariates should not be affected by treatment; time-invariant (pre-treatment)
covariates are the safe choice. Because the regression is saturated in treatment
cells, no already-treated observation serves as a control and the negative-weighting
problem of [`did_twfe`](@ref) does not arise.

Identification depends on the comparison group. With the default
`control_group = :not_yet_treated`, all untreated observations, including
pre-treatment observations of later cohorts, identify the cohort and period effects;
this requires parallel trends and no anticipation in **all** periods, before as well
as after treatment, and gives the same estimates as the imputation estimator
[`did_imputation`](@ref) in balanced panels without covariates (Wooldridge, 2025;
Borusyak, Jaravel and Spiess, 2024). Under that assumption it is efficient in the
sense of the imputation estimator, but violations of parallel pre-trends feed
directly into the post-treatment estimates. With `:never_treated`, cells are
estimated for every period except the reference period ``g - 1``, so pre-treatment
cells become placebo effects and only never-treated units act as controls; parallel
trends is then needed only relative to the never-treated group, as for the
never-treated version of [`did_callaway_santanna`](@ref).

`family = :poisson` or `:logit` fits the nonlinear version (Wooldridge, 2023): an
exponential or logistic conditional mean with the same cohort, period and treatment
terms, estimated by quasi-maximum likelihood (the Poisson QMLE is consistent for any
nonnegative outcome with a correctly specified conditional mean; Gourieroux, Monfort
and Trognon, 1984). Parallel trends is then assumed on the scale of the linear
index, e.g. equal proportional trends for the Poisson model, which is a different
and not nested assumption from parallel trends in levels (Roth and Sant'Anna,
2023). Cell effects are average differences in the predicted response with and
without the cell's treatment terms.

Standard errors come from the delta method applied to the model covariance,
cluster-robust at the unit level by default; the linear model uses a t reference
with ``G - 1`` degrees of freedom under clustering and the nonlinear models normal
critical values. Units treated in the first period are dropped. Without
never-treated units and with `:not_yet_treated`, the last-treated cohort serves as
the comparison group and the periods from its treatment onwards are dropped (as in
`etwfe`). The implementation is validated against the R packages `etwfe` and
`fixest`.

# Arguments
- `data`: a long-format panel, or repeated cross-sections with `unit = nothing`.
- `outcome::Symbol`: outcome column (nonnegative for Poisson, 0/1 for logit).
- `treatment`: an absorbing 0/1 indicator or [`FirstTreated`](@ref)`(column)`.
- `unit`: unit identifier, or `nothing` for repeated cross-sections (then
  `treatment` must be `FirstTreated` and `fe = :cohort`).
- `time::Symbol`: time column.

# Keywords
- `covariates::Vector{Symbol} = Symbol[]`: numeric controls, entered with the
  interactions described above.
- `control_group::Symbol = :not_yet_treated`: `:not_yet_treated` or
  `:never_treated`, as described above.
- `family::Symbol = :gaussian`: `:gaussian` (least squares), `:poisson` or `:logit`.
- `fe::Symbol = :cohort`: `:cohort` (cohort and period fixed effects, the default of
  R's `etwfe`) or `:unit` (unit and period fixed effects; linear model only). In
  balanced panels without covariates both give the same treatment-effect estimates.
- `weights::Union{Nothing,Symbol} = nothing`: regression weights, also used in the
  aggregations.
- `cluster = unit`: clustering column; `nothing` gives heteroskedasticity-robust
  standard errors for the linear model and model-based (inverse-Hessian) standard
  errors for Poisson and logit.
- `vcov = nothing`: a `FixedEffectModels` covariance estimator for the linear model;
  takes precedence over `cluster`.

# Returns
- `ETWFEEstimate`: the ``ATT(g,t)`` cells; `aggregate_att(r, type)` gives the simple
  ATT and the cohort, calendar and event-time (`:dynamic`) aggregations.

# Examples
```julia
using DrSnow, CSV, DataFrames
file = joinpath(pkgdir(DrSnow), "test", "validation", "did", "mpdta.csv")
mpdta = CSV.read(file, DataFrame)
r = did_etwfe(mpdta, :lemp, FirstTreated(:first_treat), :countyreal, :year)
aggregate_att(r, :simple)
es = aggregate_att(r, :dynamic)
mpdta.emp = round.(exp.(mpdta.lemp))           # employment count
rp = did_etwfe(mpdta, :emp, FirstTreated(:first_treat), :countyreal, :year;
               family=:poisson)
aggregate_att(rp, :simple)
```

# References
- Wooldridge, J. M. (2025). Two-way fixed effects, the two-way Mundlak regression,
  and difference-in-differences estimators. *Empirical Economics*, 69(5),
  2545–2587.
- Wooldridge, J. M. (2023). Simple approaches to nonlinear difference-in-differences
  with panel data. *The Econometrics Journal*, 26(3), C31–C66.
- Borusyak, K., Jaravel, X., & Spiess, J. (2024). Revisiting event-study designs:
  Robust and efficient estimation. *Review of Economic Studies*, 91(6), 3253–3285.
- Mundlak, Y. (1978). On the pooling of time series and cross section data.
  *Econometrica*, 46(1), 69–85.
- Roth, J., & Sant'Anna, P. H. C. (2023). When is parallel trends sensitive to
  functional form? *Econometrica*, 91(2), 737–747.
- Gourieroux, C., Monfort, A., & Trognon, A. (1984). Pseudo maximum likelihood
  methods: Applications to Poisson models. *Econometrica*, 52(3), 701–720.
- McDermott, G. (2026). etwfe: Extended two-way fixed effects. R package version
  0.6.2.
"""
function did_etwfe(data, outcome::Symbol, treatment, unit, time::Symbol;
                   covariates::Vector{Symbol}=Symbol[],
                   control_group::Symbol=:not_yet_treated, family::Symbol=:gaussian,
                   fe::Symbol=:cohort, weights::Union{Nothing,Symbol}=nothing,
                   cluster::Union{Nothing,Symbol,Vector{Symbol}}=unit, vcov=nothing)
    _did_check_control_group(control_group)
    family in _DID_ETWFE_FAMILIES || throw(ArgumentError(
        "family must be :gaussian, :poisson or :logit"))
    fe in (:cohort, :unit) || throw(ArgumentError("fe must be :cohort or :unit"))
    unit === nothing && fe === :unit && throw(ArgumentError(
        "fe = :unit needs a unit identifier (repeated cross-sections use fe = :cohort)"))
    family === :gaussian || fe === :cohort || throw(ArgumentError(
        "the nonlinear ETWFE uses cohort and period dummies: fe must be :cohort"))
    family === :gaussian || vcov === nothing || throw(ArgumentError(
        "`vcov` applies to the linear model; use `cluster` for Poisson/logit"))
    tcol = _did_treatment_column(treatment)
    clcols = _did_cluster_symbols(cluster)
    df = _did_prepare(data, [outcome, tcol, unit, time, covariates..., weights,
                             clcols...]; context="did_etwfe", treatment=treatment)
    tm = treatment_timing(df, treatment, unit, time)
    tm.absorbing || throw(ArgumentError(
        "did_etwfe requires an absorbing (staggered-adoption) treatment; for " *
        "treatments that switch on and off see did_multiplegt_dyn"))
    early = (tm.row_cohort .> 0) .& (tm.row_cohort .<= 1)
    if any(early)
        what = unit === nothing ? "observation(s)" : "unit(s)"
        nunits = unit === nothing ? count(early) : length(unique(tm.row_unit[early]))
        @warn "did_etwfe: dropped $nunits $what already treated in the first period"
        df = df[.!early, :]
        tm = treatment_timing(df, treatment, unit, time)
    end
    G = copy(tm.row_cohort)
    P = tm.row_period
    if !any(==(0), G)
        control_group === :never_treated && throw(ArgumentError(
            "did_etwfe: control_group = :never_treated needs never-treated units"))
        last = maximum(G)
        @warn "did_etwfe: no never-treated units; the last-treated cohort " *
              "($(tm.periods[last])) is the control group and periods from " *
              "$(tm.periods[last]) on are dropped"
        keep = P .< last
        df = df[keep, :]
        G = G[keep]
        P = P[keep]
        G[G .== last] .= 0
    end
    any(>(0), G) || throw(ArgumentError("did_etwfe: no treated cohorts"))
    any(==(0), G) || throw(ArgumentError("did_etwfe: no control observations"))
    # treatment cells
    cells = Tuple{Int,Int}[]
    rowcell = zeros(Int, nrow(df))
    celldict = Dict{Tuple{Int,Int},Int}()
    for i in 1:nrow(df)
        g, t = G[i], P[i]
        g > 0 || continue
        incell = control_group === :not_yet_treated ? t >= g : t != g - 1
        incell || continue
        c = get!(celldict, (g, t)) do
            push!(cells, (g, t))
            length(cells)
        end
        rowcell[i] = c
    end
    ord = sortperm(cells)
    cells = cells[ord]
    remap = invperm(ord)
    rowcell = [c == 0 ? 0 : remap[c] for c in rowcell]
    w = weights === nothing ? ones(nrow(df)) : float.(df[!, weights])
    any(<(0), w) && throw(ArgumentError("weights must be ≥ 0"))
    # covariates demeaned by cohort (over all observations of the cohort)
    X = _did_design_matrix(df, covariates)[:, 2:end]
    k = length(covariates)
    Xdm = copy(X)
    for g in unique(G)
        idx = findall(==(g), G)
        Xdm[idx, :] .-= mean(X[idx, :]; dims=1)
    end
    common = (df=df, G=G, P=P, cells=cells, rowcell=rowcell, w=w, X=X, Xdm=Xdm, k=k,
              tm=tm, covariates=covariates, control_group=control_group,
              weights=weights, outcome=outcome, unit=unit, time=time,
              cluster=cluster, vcov=vcov, fe=fe, family=family)
    return family === :gaussian ? _did_etwfe_linear(common) : _did_etwfe_glm(common)
end

did_etwfe(panel::TreatmentPanel; kwargs...) =
    did_etwfe(panel.data, panel.outcome, panel.treatment, panel.unit_id, panel.time;
              covariates=panel.covariates, kwargs...)

# Column names for the regression; `pre` keeps them unique in `df`.
function _did_etwfe_linear(c)
    df = copy(c.df; copycols=false)
    ncell = length(c.cells)
    k = c.k
    taucols = Symbol[]
    gamcols = Matrix{Symbol}(undef, ncell, k)
    for (j, (g, t)) in enumerate(c.cells)
        nm = _did_fresh_name(df, "etwfe_tau_$(g)_$(t)")
        df[!, nm] = Float64.(c.rowcell .== j)
        push!(taucols, nm)
        for m in 1:k
            nm2 = _did_fresh_name(df, "etwfe_tau_$(g)_$(t)_x$(m)")
            df[!, nm2] = (c.rowcell .== j) .* c.Xdm[:, m]
            gamcols[j, m] = nm2
        end
    end
    nuis = Symbol[]
    gcol = _did_fresh_name(df, "etwfe_cohort")
    df[!, gcol] = c.G
    pcol = _did_fresh_name(df, "etwfe_period")
    df[!, pcol] = c.P
    if k > 0
        gl = sort(unique(c.G))
        pl = sort(unique(c.P))
        for m in 1:k
            push!(nuis, c.covariates[m])
            for g in gl
                g == 0 && continue
                nm = _did_fresh_name(df, "etwfe_x$(m)_g$(g)")
                df[!, nm] = (c.G .== g) .* c.X[:, m]
                push!(nuis, nm)
            end
            for t in pl[2:end]
                nm = _did_fresh_name(df, "etwfe_x$(m)_t$(t)")
                df[!, nm] = (c.P .== t) .* c.X[:, m]
                push!(nuis, nm)
            end
        end
    end
    fecols = c.fe === :unit ? [c.unit, pcol] : [gcol, pcol]
    rhs = vcat(taucols, vec(permutedims(gamcols)), nuis)
    f = make_formula(c.outcome, rhs; fe=fecols)
    vc = _did_vcov_estimator(c.cluster, c.vcov)
    wcol = nothing
    if c.weights !== nothing
        wcol = _did_fresh_name(df, "etwfe_w")
        df[!, wcol] = c.w
    end
    m = wcol === nothing ? reg(df, f, vc) : reg(df, f, vc; weights=wcol)
    # coefficients of the treatment cells and their covariate interactions
    idx = Int[]
    for (j, nm) in enumerate(taucols)
        push!(idx, try
            coef_index(m, nm)
        catch err
            err isa ErrorException || rethrow()
            g, t = c.cells[j]
            throw(ArgumentError("did_etwfe: ATT(g=$(c.tm.periods[g]), " *
                                "t=$(c.tm.periods[t])) is not identified (collinear)"))
        end)
    end
    gidx = Matrix{Int}(undef, ncell, k)
    for j in 1:ncell, mm in 1:k
        gidx[j, mm] = coef_index(m, gamcols[j, mm])
    end
    allidx = vcat(idx, vec(permutedims(gidx)))
    b = coef(m)[allidx]
    Vb = Matrix(StatsAPI.vcov(m)[allidx, allidx])
    # cell effects: averages over the cell's observations of τ + (x - x̄_g)'γ
    p = length(allidx)
    Jc = zeros(ncell, p)
    wc = zeros(ncell)
    for i in eachindex(c.rowcell)
        j = c.rowcell[i]
        j == 0 && continue
        wi = c.w[i]
        wc[j] += wi
        Jc[j, j] += wi
        for mm in 1:k
            Jc[j, ncell + (j - 1) * k + mm] += wi * c.Xdm[i, mm]
        end
    end
    Jc ./= wc
    att = Jc * b
    V = Matrix(Symmetric(Jc * Vb * Jc'))
    return ETWFEEstimate(first.(c.cells), last.(c.cells), att, V, wc, Jc, Vb,
                         c.tm.periods, nobs(m), dof_residual(m), _did_nclusters(m),
                         (control_group=c.control_group, family=:gaussian,
                          covariates=c.covariates, fe=c.fe), m)
end

# Poisson / logit QMLE with cohort and period dummies (dense design).
function _did_etwfe_glm(c)
    n = nrow(c.df)
    ncell = length(c.cells)
    k = c.k
    gl = sort(filter(>(0), unique(c.G)))
    pl = sort(unique(c.P))
    cols = Vector{Vector{Float64}}()
    names_ = String[]
    # treatment block first: τ cells, then γ (cell × covariate) interactions
    for (j, (g, t)) in enumerate(c.cells)
        push!(cols, Float64.(c.rowcell .== j))
        push!(names_, "tau_$(g)_$(t)")
    end
    for j in 1:ncell, mm in 1:k
        push!(cols, (c.rowcell .== j) .* c.Xdm[:, mm])
        push!(names_, "tau_$(j)_x$(mm)")
    end
    ntreat = length(cols)
    push!(cols, ones(n))
    push!(names_, "(Intercept)")
    for g in gl
        push!(cols, Float64.(c.G .== g))
        push!(names_, "cohort_$g")
    end
    for t in pl[2:end]
        push!(cols, Float64.(c.P .== t))
        push!(names_, "period_$t")
    end
    for mm in 1:k
        push!(cols, c.X[:, mm])
        push!(names_, "x$(mm)")
        for g in gl
            push!(cols, (c.G .== g) .* c.X[:, mm])
            push!(names_, "x$(mm)_g$(g)")
        end
        for t in pl[2:end]
            push!(cols, (c.P .== t) .* c.X[:, mm])
            push!(names_, "x$(mm)_t$(t)")
        end
    end
    Xd = reduce(hcat, cols)
    # drop collinear nuisance columns (the treatment block must be identified)
    keep = _did_etwfe_independent_columns(Xd, ntreat)
    Xd = Xd[:, keep]
    names_ = names_[keep]
    y = float.(c.df[!, c.outcome])
    w = c.w
    if c.family === :poisson
        any(<(0), y) &&
            throw(ArgumentError("family = :poisson needs a nonnegative outcome"))
    else
        all(v -> v == 0 || v == 1, y) || throw(ArgumentError(
            "family = :logit needs a 0/1 outcome"))
    end
    b, H = _did_etwfe_qmle(Xd, y, w, c.family)
    μ, μ′ = _did_etwfe_mean(Xd * b, c.family)
    Hinv = inv(Symmetric(H))
    scores = Xd .* (w .* (y .- μ))
    ncl = 0
    clcols = _did_cluster_symbols(c.cluster)
    if isempty(clcols)
        Vb = Matrix(Hinv)
    else
        length(clcols) == 1 || throw(ArgumentError(
            "Poisson/logit ETWFE supports one clustering variable"))
        S = _cluster_sums(scores, c.df[!, only(clcols)])
        ncl = size(S, 1)
        K = size(Xd, 2)
        adj = ncl / (ncl - 1) * (n - 1) / (n - K)
        Vb = Matrix(Symmetric(Hinv * (S' * S) * Hinv)) .* adj
    end
    # cell effects: mean over the cell of μ(η) - μ(η - treatment terms)
    ntk = count(<=(ntreat), keep)
    Jc = zeros(ncell, size(Xd, 2))
    att = zeros(ncell)
    wc = zeros(ncell)
    for i in 1:n
        j = c.rowcell[i]
        j == 0 && continue
        x1 = Xd[i, :]
        x0 = copy(x1)
        x0[1:ntk] .= 0.0
        m1, d1 = _did_etwfe_mean([dot(x1, b)], c.family)
        m0, d0 = _did_etwfe_mean([dot(x0, b)], c.family)
        wi = w[i]
        wc[j] += wi
        att[j] += wi * (m1[1] - m0[1])
        Jc[j, :] .+= wi .* (d1[1] .* x1 .- d0[1] .* x0)
    end
    att ./= wc
    Jc ./= wc
    V = Matrix(Symmetric(Jc * Vb * Jc'))
    model = (coef=b, coefnames=names_, vcov=Vb, family=c.family, converged=true)
    return ETWFEEstimate(first.(c.cells), last.(c.cells), att, V, wc, Jc, Vb,
                         c.tm.periods, n, Inf, ncl,
                         (control_group=c.control_group, family=c.family,
                          covariates=c.covariates, fe=:cohort), model)
end

function _did_etwfe_independent_columns(X, ntreat)
    F = qr(X, ColumnNorm())
    r = rank(X)
    keep = sort(F.p[1:r])
    all(in(keep), 1:ntreat) || throw(ArgumentError(
        "did_etwfe: some cohort × period treatment effects are not identified " *
        "(collinear with the cohort and period dummies or covariates)"))
    return keep
end

function _did_etwfe_mean(η, family)
    if family === :poisson
        μ = exp.(η)
        return μ, μ
    end
    p = _did_logistic.(η)
    return p, p .* (1 .- p)
end

function _did_etwfe_qmle(X, y, w, family)
    ȳ = sum(w .* y) / sum(w)
    b0 = zeros(size(X, 2))
    icol = findfirst(j -> all(==(1.0), view(X, :, j)), axes(X, 2))
    if family === :poisson
        ȳ > 0 || throw(ArgumentError("family = :poisson: the outcome is always zero"))
        icol === nothing || (b0[icol] = log(ȳ))
        function fp(b)
            η = X * b
            μ = exp.(η)
            return sum(w .* (μ .- y .* η)), X' * (w .* (μ .- y)), X' * (X .* (w .* μ))
        end
        b, ok = _did_newton_min(fp, b0; maxiter=500)
        ok || throw(ArgumentError("did_etwfe: Poisson QMLE did not converge"))
        return b, fp(b)[3]
    end
    0 < ȳ < 1 || throw(ArgumentError("family = :logit: the outcome does not vary"))
    icol === nothing || (b0[icol] = log(ȳ / (1 - ȳ)))
    function fl(b)
        η = X * b
        p = _did_logistic.(η)
        nll = -sum(w .* (y .* η .- log1p.(exp.(-abs.(η))) .- max.(η, 0)))
        return nll, -(X' * (w .* (y .- p))), X' * (X .* (w .* p .* (1 .- p)))
    end
    b, ok = _did_newton_min(fl, b0; maxiter=500)
    ok || throw(ArgumentError("did_etwfe: logit QMLE did not converge (perfect " *
                              "separation is a likely reason)"))
    return b, fl(b)[3]
end

# ---------------------------------------------------------------------------
# Aggregations (R etwfe::emfx)
# ---------------------------------------------------------------------------

function _did_etwfe_combine(r::ETWFEEstimate, sel)
    wv = r.cell_weight[sel]
    a = wv ./ sum(wv)
    θ = dot(a, r.coef[sel])
    grad = vec(a' * r.gradient[sel, :])
    return θ, grad
end

"""
    aggregate_att(r::ETWFEEstimate, type=:simple; min_e=nothing, max_e=nothing)
        -> Union{AggregatedATT, EventStudyEstimate}

Aggregate extended TWFE cell effects into summary parameters, as R's `etwfe::emfx`.

The aggregations are observation-weighted averages of the cell effects
``\\widehat{ATT}(g,t)``: over all treated cells (`:simple`, the ATT over treated
observations), by cohort (`:group`), by calendar period (`:calendar`), or by event
time ``e = t - g`` (`:dynamic`). The weights are the (weighted) numbers of
observations per cell and are treated as fixed, so standard errors come from the
delta method applied to the regression covariance of the cell effects; unlike
[`aggregate_att`](@ref) for Callaway–Sant'Anna, they do not account for sampling
variation in the cohort shares. For `:group` and `:calendar` the first (overall)
coefficient is the simple ATT. With `control_group = :never_treated`, `:dynamic`
also reports the pre-treatment (placebo) event times, relative to the omitted period
``e = -1``, which can be used with [`pre_trend_test`](@ref) and
[`honest_did`](@ref). As for any event-time aggregation, changes in the cohort
composition across ``e`` are mixed with the dynamics of the effects.

# Arguments
- `r::ETWFEEstimate`: cell effects from [`did_etwfe`](@ref).
- `type::Symbol = :simple`: `:simple`, `:group`, `:calendar` or `:dynamic`.

# Keywords
- `min_e`, `max_e = nothing`: for `:dynamic`, the range of event times reported.

# Returns
- `EventStudyEstimate` for `:dynamic`, with `details.overall` (the simple ATT);
- `AggregatedATT` otherwise.

# Examples
```julia
using DrSnow, CSV, DataFrames
file = joinpath(pkgdir(DrSnow), "test", "validation", "did", "mpdta.csv")
mpdta = CSV.read(file, DataFrame)
r = did_etwfe(mpdta, :lemp, FirstTreated(:first_treat), :countyreal, :year;
              control_group=:never_treated)
aggregate_att(r, :group)
es = aggregate_att(r, :dynamic; max_e=2)
pre_trend_test(es)
```

# References
- Wooldridge, J. M. (2025). Two-way fixed effects, the two-way Mundlak regression,
  and difference-in-differences estimators. *Empirical Economics*, 69(5),
  2545–2587.
- McDermott, G. (2026). etwfe: Extended two-way fixed effects. R package version
  0.6.2.
"""
function aggregate_att(r::ETWFEEstimate, type::Symbol=:simple;
                       min_e::Union{Nothing,Integer}=nothing,
                       max_e::Union{Nothing,Integer}=nothing)
    type in (:simple, :group, :calendar, :dynamic) || throw(ArgumentError(
        "type must be :simple, :group, :calendar or :dynamic"))
    e = r.times .- r.cohorts
    post = findall(>=(0), e)
    isempty(post) && throw(ArgumentError("no post-treatment cells"))
    θs, gs = _did_etwfe_combine(r, post)
    Vb = r.vcov_model
    meth = "Wooldridge extended TWFE aggregation ($type)"
    if type === :dynamic
        mine = min_e === nothing ? minimum(e) : Int(min_e)
        maxe = max_e === nothing ? maximum(e) : Int(max_e)
        es_ = sort(unique(filter(x -> mine <= x <= maxe, e)))
        isempty(es_) && throw(ArgumentError("no event times in [min_e, max_e]"))
        θ = Float64[]
        J = Vector{Vector{Float64}}()
        for x in es_
            t, g = _did_etwfe_combine(r, findall(==(x), e))
            push!(θ, t)
            push!(J, g)
        end
        Jm = reduce(hcat, J)'
        V = Matrix(Symmetric(Jm * Vb * Jm'))
        overall = DiDEstimate([θs], fill(dot(gs, Vb * gs), 1, 1), ["ATT"], r.nobs,
                              r.dof, r.n_clusters, -1, -1, length(r.periods),
                              meth * " — simple ATT", "ATT (simple average)",
                              (source=r,))
        ref = r.settings.control_group === :never_treated ? [-1] : Int[]
        return EventStudyEstimate(es_, θ, V, ref, r.nobs, r.dof, r.n_clusters,
            "Wooldridge extended TWFE event study ($(r.settings.family), " *
            "$(r.settings.control_group))",
            "observation-weighted average of ATT(g, g+e)" *
            (r.settings.family === :gaussian ? "" : " (response scale)"), Float64[],
            (overall=overall, binned=(false, false), source=r,
             n_treated=-1, n_control=-1, n_periods=length(r.periods),
             note=isempty(ref) ? "Not-yet-treated controls: pre-treatment periods " *
                                 "are pooled in the counterfactual (no single " *
                                 "reference period)." : ""))
    end
    if type === :simple
        return AggregatedATT(:simple, Any[], [θs], fill(dot(gs, Vb * gs), 1, 1),
                             r.nobs, r.n_clusters, Float64[], zeros(0, 1),
                             (source=r, method=meth, dof=r.dof,
                              estimand="ATT (observation-weighted average of " *
                                       "ATT(g,t), t ≥ g)"))
    end
    keyv = type === :group ? r.cohorts : r.times
    labs = sort(unique(keyv[post]))
    θ = [θs]
    J = [gs]
    for l in labs
        t, g = _did_etwfe_combine(r, [c for c in post if keyv[c] == l])
        push!(θ, t)
        push!(J, g)
    end
    Jm = reduce(hcat, J)'
    V = Matrix(Symmetric(Jm * Vb * Jm'))
    what = type === :group ? "cohort" : "calendar period"
    return AggregatedATT(type, collect(r.periods[labs]), θ, V, r.nobs, r.n_clusters,
                         Float64[], zeros(0, length(θ)),
                         (source=r, method=meth, dof=r.dof,
                          estimand="ATT by $what; overall = simple ATT"))
end
