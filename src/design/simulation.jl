# Simulation-based design diagnosis (Blair, Cooper, Coppock & Humphreys 2019): declare a
# design (data-generating process, assignment, estimand, estimators), simulate it many
# times with pre-drawn seeds, and summarize diagnosands (power, bias, RMSE, coverage,
# type-S and type-M errors) with Monte Carlo standard errors.

"""
    DeclaredDesign

A research design declared for simulation-based diagnosis; the result of
[`declare_design`](@ref).

Following the MIDA framework of Blair, Cooper, Coppock & Humphreys (2019), a design is
a model of the world (a data-generating process for the population and its potential
outcomes), an inquiry (the estimand), a data strategy (sampling and assignment) and
an answer strategy (one or more estimators). A `DeclaredDesign` stores these as Julia
functions together with default design parameters, so that
[`diagnose_design`](@ref), [`diagnose_grid`](@ref) and [`optimize_design`](@ref)
can simulate the whole design and evaluate the estimators against the estimand.

# Fields
- `name::String`: label of the design.
- `population`: function `(rng, params) -> DataFrame` generating one data set.
- `assignment`: `nothing`, an `AssignmentMechanism`, or a function
  `(data, params) -> AssignmentMechanism`.
- `estimand`: `nothing`, a number, or a function `(data, params) -> Real`
  evaluated on the generated data before assignment (e.g. the sample ATE from the
  potential outcomes).
- `estimators::Vector{Pair{String,Any}}`: `label => f` pairs, with `f(data, params)`
  (or `f(data, params, rng)`) returning a `CausalEstimate`.
- `potential_outcomes::Union{Nothing,Tuple{Symbol,Symbol}}`: the columns `(Y0, Y1)`
  from which the observed outcome is revealed after assignment.
- `outcome::Symbol`: name of the revealed outcome column.
- `treatment::Symbol`: name of the assigned treatment column.
- `term::Union{Int,String}`: the coefficient of each estimator's result that is
  evaluated (position or name).
- `params::NamedTuple`: default design parameters.

# References
- Blair, G., Cooper, J., Coppock, A., & Humphreys, M. (2019). Declaring and
  diagnosing research designs. *American Political Science Review*, 113(3), 838–859.
"""
struct DeclaredDesign
    name::String
    population::Any
    assignment::Any
    estimand::Any
    estimators::Vector{Pair{String,Any}}
    potential_outcomes::Union{Nothing,Tuple{Symbol,Symbol}}
    outcome::Symbol
    treatment::Symbol
    term::Union{Int,String}
    params::NamedTuple
end

function Base.show(io::IO, ::MIME"text/plain", d::DeclaredDesign)
    println(io, "Declared design: ", d.name)
    println(io, "Assignment: ", d.assignment === nothing ? "in the population function" :
                                d.assignment isa AssignmentMechanism ?
                                _ri_describe(d.assignment) : "function of data and params")
    println(io, "Estimand: ", d.estimand === nothing ? "none" :
                              d.estimand isa Real ? string(d.estimand) : "function")
    println(io, "Estimators: ", join(first.(d.estimators), ", "))
    print(io, "Parameters: ", isempty(d.params) ? "none" : string(d.params))
end

Base.show(io::IO, d::DeclaredDesign) = print(io, "DeclaredDesign(\"", d.name, "\")")

function _des_estimator_list(estimators)
    if estimators isa Pair
        return Pair{String,Any}[string(estimators.first) => estimators.second]
    elseif estimators isa AbstractVector
        out = Pair{String,Any}[]
        for (k, e) in enumerate(estimators)
            e isa Pair ? push!(out, string(e.first) => e.second) :
            push!(out, "estimator $k" => e)
        end
        return out
    elseif estimators isa NamedTuple
        return Pair{String,Any}[string(k) => v for (k, v) in pairs(estimators)]
    end
    return Pair{String,Any}["estimator" => estimators]
end

"""
    declare_design(population; estimators, assignment=nothing, estimand=nothing,
                   potential_outcomes=(:Y0, :Y1), outcome=:Y, treatment=:Z, term=1,
                   params=(;), name="design") -> DeclaredDesign

Declare a research design (data-generating process, assignment, estimand and
estimators) for simulation-based diagnosis.

Analytic power formulas cover a handful of canonical designs and estimators under
simplifying assumptions (constant effects, known variances, normality). Most applied
designs depart from them: blocked assignment with heterogeneous effects, attrition,
clustered panels, natural experiments analysed with DiD, RD or IV estimators,
multiple estimators to choose between. Blair, Cooper, Coppock & Humphreys (2019)
propose to *declare* the design completely, as a model of the world (M), an inquiry
or estimand (I), a data strategy (D) and an answer strategy (A), and to *diagnose* it
by Monte Carlo simulation of its properties (power, bias, RMSE, coverage). This
function implements the declaration, in the spirit of the R package `DeclareDesign`.

One simulation proceeds as follows. The population function draws a data set,
`data = population(rng, params)`; the estimand is computed on `data` before
assignment (so it can be a sample quantity such as the sample ATE from the potential
outcome columns); when `assignment` is given, an assignment `z` is drawn from the
mechanism, stored in column `treatment`, and the observed outcome
`outcome = z ? Y1 : Y0` is revealed from the `potential_outcomes` columns (when they
are present); finally every estimator is applied to the data. Natural experiments
(DiD, RD, IV) are declared with `assignment = nothing`, the population function then
generating the observed data directly. The conclusions of a diagnosis are only as
credible as the declared model: vary the uncertain parameters (effect size, ICC,
serial correlation, compliance) with [`diagnose_grid`](@ref) rather than relying on
a single guess.

# Arguments
- `population`: a function `(rng::AbstractRNG, params::NamedTuple) -> DataFrame`
  that draws one data set; it must use `rng` for all randomness so that simulations
  are reproducible.

# Keywords
- `estimators`: a function `(data, params) -> CausalEstimate` (optionally taking a
  third `rng` argument for estimators with internal randomness), a `label => f` pair,
  or a vector or `NamedTuple` of them (required). Any DrSnow estimator can be used,
  e.g. `(d, p) -> experiment_estimate(d, :Y, :Z)` or
  `(d, p) -> did_twfe(d, :y, :d, :unit, :year)`.
- `assignment = nothing`: `nothing`, an `AssignmentMechanism` whose units are the rows
  of the population data, or a function `(data, params) -> AssignmentMechanism`
  (e.g. blocks formed on the simulated covariates with [`block_design`](@ref)).
- `estimand = nothing`: `nothing`, a constant, or a function `(data, params) -> Real`.
  Without an estimand only power and the distribution of the estimates are diagnosed.
- `potential_outcomes = (:Y0, :Y1)`: columns of the untreated and treated potential
  outcomes, or `nothing`.
- `outcome::Symbol = :Y`, `treatment::Symbol = :Z`: names of the revealed outcome and
  the assigned treatment columns.
- `term = 1`: which coefficient of each result to evaluate (position or name).
- `params::NamedTuple = (;)`: default design parameters, overridden in
  [`diagnose_design`](@ref), [`diagnose_grid`](@ref) and [`optimize_design`](@ref).
- `name::AbstractString = "design"`: label of the design.

# Returns
- [`DeclaredDesign`](@ref).

# Examples
```julia
using DrSnow, DataFrames, Statistics, StableRNGs
pop(rng, p) = (Y0 = randn(rng, p.n); DataFrame(Y0=Y0, Y1=Y0 .+ p.effect))
d = declare_design(pop; params=(n=100, effect=0.3),
                   assignment=(data, p) -> CompleteRandomization(p.n, p.n ÷ 2),
                   estimand=(data, p) -> mean(data.Y1 .- data.Y0),
                   estimators="DiM" => (data, p) -> experiment_estimate(data, :Y, :Z))
diagnose_design(d; sims=500, rng=StableRNG(1))
```

# References
- Blair, G., Cooper, J., Coppock, A., & Humphreys, M. (2019). Declaring and
  diagnosing research designs. *American Political Science Review*, 113(3), 838–859.
"""
function declare_design(population; estimators, assignment=nothing, estimand=nothing,
                        potential_outcomes::Union{Nothing,Tuple{Symbol,Symbol}}=(:Y0, :Y1),
                        outcome::Symbol=:Y, treatment::Symbol=:Z,
                        term::Union{Integer,AbstractString,Symbol}=1,
                        params::NamedTuple=NamedTuple(), name::AbstractString="design")
    population isa Function ||
        throw(ArgumentError("declare_design: population must be a function (rng, params)"))
    assignment === nothing || assignment isa AssignmentMechanism ||
        assignment isa Function ||
        throw(ArgumentError("declare_design: assignment must be nothing, an " *
                            "AssignmentMechanism or a function (data, params)"))
    estimand === nothing || estimand isa Real || estimand isa Function ||
        throw(ArgumentError("declare_design: estimand must be nothing, a number or a " *
                            "function (data, params)"))
    ests = _des_estimator_list(estimators)
    isempty(ests) && throw(ArgumentError("declare_design: need at least one estimator"))
    all(e -> e.second isa Function, ests) ||
        throw(ArgumentError("declare_design: estimators must be functions"))
    allunique(first.(ests)) ||
        throw(ArgumentError("declare_design: estimator labels must be unique"))
    tm = term isa Integer ? Int(term) : string(term)
    return DeclaredDesign(string(name), population, assignment, estimand, ests,
                          potential_outcomes, outcome, treatment, tm, params)
end

# One simulated row per estimator: (estimand, estimate, se, p, lo, hi, dof).
function _des_simulate_once(d::DeclaredDesign, params::NamedTuple, seed::UInt64,
                            level::Real)
    rng = Random.Xoshiro(seed)
    data = d.population(rng, params)
    data isa AbstractDataFrame ||
        throw(ArgumentError("declare_design: population must return a DataFrame"))
    truth = d.estimand === nothing ? NaN :
            d.estimand isa Real ? float(d.estimand) : float(d.estimand(data, params))
    if d.assignment !== nothing
        mech = d.assignment isa AssignmentMechanism ? d.assignment :
               d.assignment(data, params)
        mech isa AssignmentMechanism ||
            throw(ArgumentError("declare_design: the assignment function must return " *
                                "an AssignmentMechanism"))
        n_units(mech) == nrow(data) ||
            throw(DimensionMismatch("assignment mechanism has $(n_units(mech)) units " *
                                    "but the population has $(nrow(data)) rows"))
        z = draw_assignment(rng, mech)
        data = DataFrame(data; copycols=false)
        data[!, d.treatment] = Int.(z)
        po = d.potential_outcomes
        if po !== nothing && hasproperty(data, po[1]) && hasproperty(data, po[2])
            data[!, d.outcome] = ifelse.(z, data[!, po[2]], data[!, po[1]])
        end
    end
    out = Vector{NTuple{7,Float64}}(undef, length(d.estimators))
    eseeds = rand(rng, UInt64, length(d.estimators))
    for (k, (_, f)) in enumerate(d.estimators)
        erng = Random.Xoshiro(eseeds[k])
        r = applicable(f, data, params, erng) ? f(data, params, erng) : f(data, params)
        r isa CausalEstimate ||
            throw(ArgumentError("declare_design: estimator $(d.estimators[k].first) " *
                                "must return a CausalEstimate, got $(typeof(r))"))
        i = d.term isa Int ? d.term : findfirst(==(d.term), StatsAPI.coefnames(r))
        (i === nothing || i > length(StatsAPI.coef(r))) &&
            throw(ArgumentError("declare_design: coefficient $(d.term) not found in " *
                                "the result of $(d.estimators[k].first)"))
        b = StatsAPI.coef(r)[i]
        se = StatsAPI.stderror(r)[i]
        dof = StatsAPI.dof_residual(r)
        p = pvalues(r)[i]
        ci = StatsAPI.confint(r; level=level)
        out[k] = (truth, b, se, p, ci[i, 1], ci[i, 2], float(dof))
    end
    return out
end

const _DES_SIM_COLS = (:estimand, :estimate, :std_error, :p_value, :conf_low,
                       :conf_high, :dof)

"""
Run `sims` simulations of `d` at `params` with seeds from `rng`; returns a DataFrame
with one row per (simulation, estimator). Identical with and without threads.
"""
function _des_run_sims(d::DeclaredDesign, params::NamedTuple, sims::Integer,
                       rng::AbstractRNG, level::Real, threaded::Bool, on_error::Symbol)
    sims >= 1 || throw(ArgumentError("sims must be positive"))
    seeds = task_seeds(rng, sims)
    K = length(d.estimators)
    res = Vector{Any}(undef, sims)
    function one(s)
        try
            res[s] = _des_simulate_once(d, params, seeds[s], level)
        catch err
            on_error === :record || rethrow()
            res[s] = sprint(showerror, err)
        end
        return nothing
    end
    if threaded && Threads.nthreads() > 1 && sims > 1
        chunks = collect(Iterators.partition(1:sims, cld(sims, 4 * Threads.nthreads())))
        tasks = [Threads.@spawn foreach(one, c) for c in chunks]
        for t in tasks
            try
                fetch(t)
            catch e
                e isa TaskFailedException ? throw(e.task.exception) : rethrow()
            end
        end
    else
        foreach(one, 1:sims)
    end
    n = sims * K
    cols = Dict(c => fill(NaN, n) for c in _DES_SIM_COLS)
    sim = zeros(Int, n)
    est = Vector{String}(undef, n)
    failed = falses(n)
    msg = fill("", n)
    row = 0
    for s in 1:sims, k in 1:K
        row += 1
        sim[row] = s
        est[row] = d.estimators[k].first
        r = res[s]
        if r isa String
            failed[row] = true
            msg[row] = r
        else
            for (j, c) in enumerate(_DES_SIM_COLS)
                cols[c][row] = r[k][j]
            end
        end
    end
    df = DataFrame(sim=sim, estimator=est)
    for c in _DES_SIM_COLS
        df[!, c] = cols[c]
    end
    df[!, :failed] = failed
    any(failed) && (df[!, :error] = msg)
    return df
end

# Diagnosands of one estimator's simulations (vectors of successful simulations).
function _des_diagnosands(t, b, se, p, lo, hi, alpha)
    S = length(b)
    out = Pair{Symbol,Union{Missing,Float64}}[]
    has_truth = !any(isnan, t)
    push!(out, :mean_estimate => mean(b))
    push!(out, :sd_estimate => S > 1 ? std(b) : missing)
    push!(out, :mean_se => mean(se))
    push!(out, :power => mean(p .<= alpha))
    if has_truth
        push!(out, :mean_estimand => mean(t))
        push!(out, :bias => mean(b .- t))
        push!(out, :rmse => sqrt(mean(abs2, b .- t)))
        push!(out, :coverage => mean((lo .<= t) .& (t .<= hi)))
        sig = p .<= alpha
        if any(sig) && all(!iszero, t[sig])
            push!(out, :type_s_rate => mean(sign.(b[sig]) .!= sign.(t[sig])))
            push!(out, :type_m => mean(abs.(b[sig]) ./ abs.(t[sig])))
            push!(out, :exaggeration_ratio => mean(b[sig] ./ t[sig]))
        else
            push!(out, :type_s_rate => missing)
            push!(out, :type_m => missing)
            push!(out, :exaggeration_ratio => missing)
        end
    end
    return out
end

function _des_summarize(sims::DataFrame, estimators::Vector{String}, alpha::Real,
                        bootstrap::Integer, rng::AbstractRNG)
    rows = DataFrame(estimator=String[], diagnosand=Symbol[],
                     value=Union{Missing,Float64}[], mc_se=Union{Missing,Float64}[],
                     n_sims=Int[], n_failed=Int[])
    bseeds = task_seeds(rng, length(estimators))
    for (k, e) in enumerate(estimators)
        sub = sims[(sims.estimator .== e) .& .!sims.failed, :]
        nf = count(sims.failed[sims.estimator .== e])
        S = nrow(sub)
        S >= 1 || throw(ArgumentError("every simulation of estimator $e failed"))
        cols = (sub.estimand, sub.estimate, sub.std_error, sub.p_value, sub.conf_low,
                sub.conf_high)
        base = _des_diagnosands(cols..., alpha)
        # bootstrap Monte Carlo standard errors (resampling simulations)
        boot = Dict{Symbol,Vector{Float64}}()
        if bootstrap > 0 && S > 1
            brng = Random.Xoshiro(bseeds[k])
            for _ in 1:bootstrap
                idx = rand(brng, 1:S, S)
                for (nm, v) in _des_diagnosands((c[idx] for c in cols)..., alpha)
                    v === missing && continue
                    push!(get!(boot, nm, Float64[]), v)
                end
            end
        end
        for (nm, v) in base
            bs = get(boot, nm, Float64[])
            se = length(bs) >= 2 ? std(bs) : missing
            push!(rows, (e, nm, v, se, S, nf))
        end
    end
    return rows
end

"""
    DesignDiagnosis

Monte Carlo diagnosands of a declared design, with their simulation standard errors;
the result of [`diagnose_design`](@ref) and [`diagnose_grid`](@ref).

A diagnosand is a summary of the sampling distribution of an estimator under the
declared design (Blair et al. 2019). Each is estimated from `sims` simulations and
comes with a bootstrap Monte Carlo standard error that measures simulation noise only,
not uncertainty about the declared model. Diagnosands that condition on statistical
significance follow Gelman & Carlin (2014): among significant estimates, the type-S
rate is the share with the wrong sign and the type-M error (exaggeration) is how much
their magnitude overstates the truth, both of which are large in underpowered
designs.

# Fields
- `design::String`: design name.
- `diagnosands::DataFrame`: long format, one row per (parameter combination,
  estimator, diagnosand), with columns for the varied parameters, `estimator`,
  `diagnosand`, `value`, `mc_se` (bootstrap Monte Carlo standard error), `n_sims` and
  `n_failed`. The diagnosands are `mean_estimate`, `sd_estimate`, `mean_se`, `power`
  (share of p-values at or below `alpha`) and, when the design has an estimand,
  `mean_estimand`, `bias`, `rmse`, `coverage` (of the `level` confidence interval),
  `type_s_rate` (share of significant estimates with the wrong sign), `type_m` (mean
  `|estimate| / |estimand|` among significant estimates) and `exaggeration_ratio`
  (mean `estimate / estimand` among significant estimates, as in `DeclareDesign`).
  Diagnosands conditional on significance are `missing` when no estimate is
  significant.
- `simulations::DataFrame`: one row per simulation and estimator (with the parameter
  columns for a grid), holding `estimand`, `estimate`, `std_error`, `p_value`,
  `conf_low`, `conf_high`, `dof` and `failed`.
- `parameters::Vector{Symbol}`: the varied parameters (empty for a single design).
- `sims::Int`: simulations per design point.
- `alpha::Float64`: significance level for power and the conditional diagnosands.
- `level::Float64`: confidence level for coverage.

# References
- Blair, G., Cooper, J., Coppock, A., & Humphreys, M. (2019). Declaring and
  diagnosing research designs. *American Political Science Review*, 113(3), 838–859.
- Gelman, A., & Carlin, J. (2014). Beyond power calculations: Assessing type S (sign)
  and type M (magnitude) errors. *Perspectives on Psychological Science*, 9(6),
  641–651.
"""
struct DesignDiagnosis
    design::String
    diagnosands::DataFrame
    simulations::DataFrame
    parameters::Vector{Symbol}
    sims::Int
    alpha::Float64
    level::Float64
end

function Base.show(io::IO, ::MIME"text/plain", r::DesignDiagnosis)
    println(io, "Design diagnosis: ", r.design)
    @printf(io, "Simulations: %d per design; alpha = %.4g; interval level = %.4g\n",
            r.sims, r.alpha, r.level)
    isempty(r.parameters) || println(io, "Varied parameters: ", join(r.parameters, ", "))
    d = r.diagnosands
    keyc = vcat(r.parameters, [:estimator])
    groups = unique(d[!, keyc])
    for g in eachrow(groups)
        mask = trues(nrow(d))
        for c in keyc
            mask .&= isequal.(d[!, c], g[c])
        end
        lab = join(["$(c) = $(g[c])" for c in keyc], ", ")
        println(io, lab)
        for row in eachrow(d[mask, :])
            v = row.value === missing ? "missing" : @sprintf("%.4g", row.value)
            se = row.mc_se === missing ? "" : @sprintf(" (MC se %.3g)", row.mc_se)
            println(io, "  ", rpad(string(row.diagnosand), 20), v, se)
        end
    end
end

Base.show(io::IO, r::DesignDiagnosis) =
    print(io, "DesignDiagnosis(\"", r.design, "\", ", nrow(r.diagnosands), " diagnosands)")

"""
    diagnose_design(design; sims=500, params=(;), alpha=0.05, level=0.95,
                    bootstrap=100, rng=Random.default_rng(),
                    threaded=Threads.nthreads() > 1, on_error=:throw) -> DesignDiagnosis

Simulate a declared design and estimate its diagnosands (power, bias, RMSE, coverage,
type-S and type-M errors) with Monte Carlo standard errors.

Each of the `sims` simulations draws data, the estimand and an assignment from the
[`DeclaredDesign`](@ref) and applies every estimator; the diagnosands summarize the
resulting sampling distributions (Blair, Cooper, Coppock & Humphreys 2019). Power is
the share of simulations whose two-sided p-value for ``H_0: \\theta = 0`` is at most
`alpha`; bias, RMSE and the coverage of the `level` confidence interval compare the
estimates with the simulated estimand; the type-S rate and type-M exaggeration
(Gelman & Carlin 2014) describe the estimates that reach significance. A diagnosis
is the natural complement of the analytic calculators ([`power_means`](@ref) and
related functions) whenever the design or estimator departs from their assumptions,
and it can reveal problems that no closed form shows, such as undercoverage of
cluster-robust intervals with few clusters or the bias of an estimator under
heterogeneous effects.

Monte Carlo standard errors come from `bootstrap` resamples of the simulations, as in
`DeclareDesign`; they quantify simulation noise only, and the number of simulations
should make them small relative to the differences of interest (for power near 0.8,
the binomial standard error is about ``0.4/\\sqrt{\\text{sims}}``). Reproducibility is
built in: one seed per simulation is drawn from `rng` up front, and each simulation
(population, assignment and estimators) uses its own `Xoshiro(seed)`, so the results
are identical with and without threads and do not depend on scheduling.

# Arguments
- `design::DeclaredDesign`: a design from [`declare_design`](@ref).

# Keywords
- `sims::Integer = 500`: number of simulations.
- `params::NamedTuple = (;)`: overrides of the design's default parameters.
- `alpha::Real = 0.05`: significance level for power and the diagnosands conditional
  on significance.
- `level::Real = 0.95`: confidence level of the intervals whose coverage is computed.
- `bootstrap::Integer = 100`: bootstrap resamples for the Monte Carlo standard errors
  (0: none).
- `rng::AbstractRNG = Random.default_rng()`: source of the per-simulation seeds.
- `threaded::Bool = Threads.nthreads() > 1`: run simulations on threads.
- `on_error::Symbol = :throw`: `:throw` stops at the first failing simulation;
  `:record` records failures (column `failed`, message in `error`), excludes them
  from the diagnosands and reports their number in `n_failed`. Frequent failures are
  themselves a diagnosis (e.g. empty arms in small blocks) and should be reported.

# Returns
- [`DesignDiagnosis`](@ref).

# Examples
```julia
using DrSnow, DataFrames, Statistics, StableRNGs
pop(rng, p) = (Y0 = randn(rng, p.n); DataFrame(Y0=Y0, Y1=Y0 .+ p.effect))
d = declare_design(pop; params=(n=100, effect=0.3),
                   assignment=(data, p) -> CompleteRandomization(p.n, p.n ÷ 2),
                   estimand=(data, p) -> mean(data.Y1 .- data.Y0),
                   estimators="DiM" => (data, p) -> experiment_estimate(data, :Y, :Z))
dx = diagnose_design(d; sims=1000, params=(n=200,), rng=StableRNG(1))
dx.diagnosands
```

# References
- Blair, G., Cooper, J., Coppock, A., & Humphreys, M. (2019). Declaring and
  diagnosing research designs. *American Political Science Review*, 113(3), 838–859.
- Gelman, A., & Carlin, J. (2014). Beyond power calculations: Assessing type S (sign)
  and type M (magnitude) errors. *Perspectives on Psychological Science*, 9(6),
  641–651.
"""
function diagnose_design(design::DeclaredDesign; sims::Integer=500,
                         params::NamedTuple=NamedTuple(), alpha::Real=0.05,
                         level::Real=0.95, bootstrap::Integer=100,
                         rng::AbstractRNG=Random.default_rng(),
                         threaded::Bool=Threads.nthreads() > 1, on_error::Symbol=:throw)
    _des_check_diag_args(alpha, level, bootstrap, on_error)
    p = merge(design.params, params)
    sim = _des_run_sims(design, p, sims, rng, level, threaded, on_error)
    diag = _des_summarize(sim, first.(design.estimators), alpha, bootstrap, rng)
    return DesignDiagnosis(design.name, diag, sim, Symbol[], Int(sims), float(alpha),
                           float(level))
end

function _des_check_diag_args(alpha, level, bootstrap, on_error)
    0 < alpha < 1 || throw(ArgumentError("alpha must be in (0, 1)"))
    0 < level < 1 || throw(ArgumentError("level must be in (0, 1)"))
    bootstrap >= 0 || throw(ArgumentError("bootstrap must be non-negative"))
    on_error in (:throw, :record) ||
        throw(ArgumentError("on_error must be :throw or :record"))
    return nothing
end

# Rows of a parameter grid: NamedTuple of vectors (Cartesian product) or DataFrame.
function _des_grid_rows(grid)
    if grid isa AbstractDataFrame
        nrow(grid) >= 1 || throw(ArgumentError("the parameter grid is empty"))
        names_ = Tuple(propertynames(grid))
        return [NamedTuple{names_}(Tuple(r)) for r in eachrow(grid)], collect(names_)
    elseif grid isa NamedTuple
        isempty(grid) && throw(ArgumentError("the parameter grid is empty"))
        ks = keys(grid)
        vals = [v isa AbstractVector || v isa AbstractRange || v isa Tuple ? collect(v) :
                [v] for v in values(grid)]
        any(isempty, vals) && throw(ArgumentError("a grid dimension is empty"))
        rows = [NamedTuple{ks}(Tuple(c)) for c in Iterators.product(vals...)]
        return vec(rows), collect(ks)
    end
    throw(ArgumentError("grid must be a NamedTuple of vectors or a DataFrame"))
end

"""
    diagnose_grid(design, grid; sims=500, alpha=0.05, level=0.95, bootstrap=100,
                  rng=Random.default_rng(), threaded=Threads.nthreads() > 1,
                  on_error=:throw) -> DesignDiagnosis

Diagnose a declared design at every combination of design parameters in a grid, for
power curves, sample-size choice and sensitivity to uncertain inputs.

The grid is either a `NamedTuple` of vectors, whose Cartesian product is taken, or a
`DataFrame` with one row per combination. Typical dimensions are the design sizes
(units, clusters, periods) and the uncertain features of the declared model (effect
size, ICC, serial correlation, compliance). At each point the design is simulated as
in [`diagnose_design`](@ref), with parameters not in the grid kept at their defaults;
the diagnosands are returned in long format with the grid parameters as columns,
ready for filtering, tabulation or plotting with [`plot_power_curve`](@ref).

Reporting the diagnosands over a range of plausible values of the uncertain inputs,
rather than at a single guess, is the recommended practice for design declaration
(Blair et al. 2019). A full grid spends the same number of simulations at every
point, including points far from the power target; when the aim is only to find the
cheapest design that reaches a target, [`optimize_design`](@ref) uses a surrogate
model to concentrate simulations near the boundary. One seed per grid point is drawn
from `rng` up front, so each point is reproducible on its own and the results do not
depend on threading.

# Arguments
- `design::DeclaredDesign`: a design from [`declare_design`](@ref).
- `grid`: a `NamedTuple` of vectors (Cartesian product) or a `DataFrame` of parameter
  combinations.

# Keywords
- `sims`, `alpha`, `level`, `bootstrap`, `rng`, `threaded`, `on_error`: as in
  [`diagnose_design`](@ref); `sims` is the number of simulations per grid point.

# Returns
- [`DesignDiagnosis`](@ref) with the grid parameters as columns of `diagnosands` and
  `simulations`, and listed in `parameters`.

# Examples
```julia
using DrSnow, DataFrames, Statistics, StableRNGs
pop(rng, p) = (Y0 = randn(rng, p.n); DataFrame(Y0=Y0, Y1=Y0 .+ p.effect))
d = declare_design(pop; params=(n=100, effect=0.3),
                   assignment=(data, p) -> CompleteRandomization(p.n, p.n ÷ 2),
                   estimand=(data, p) -> mean(data.Y1 .- data.Y0),
                   estimators="DiM" => (data, p) -> experiment_estimate(data, :Y, :Z))
g = diagnose_grid(d, (n=[50, 100, 200, 400], effect=[0.2, 0.3]); sims=300,
                  rng=StableRNG(1))
filter(r -> r.diagnosand == :power, g.diagnosands)
```

# References
- Blair, G., Cooper, J., Coppock, A., & Humphreys, M. (2019). Declaring and
  diagnosing research designs. *American Political Science Review*, 113(3), 838–859.
"""
function diagnose_grid(design::DeclaredDesign, grid; sims::Integer=500, alpha::Real=0.05,
                       level::Real=0.95, bootstrap::Integer=100,
                       rng::AbstractRNG=Random.default_rng(),
                       threaded::Bool=Threads.nthreads() > 1, on_error::Symbol=:throw)
    _des_check_diag_args(alpha, level, bootstrap, on_error)
    rows, pnames = _des_grid_rows(grid)
    seeds = task_seeds(rng, length(rows))
    diags = DataFrame[]
    simsdf = DataFrame[]
    for (g, row) in enumerate(rows)
        grng = Random.Xoshiro(seeds[g])
        p = merge(design.params, row)
        sim = _des_run_sims(design, p, sims, grng, level, threaded, on_error)
        dg = _des_summarize(sim, first.(design.estimators), alpha, bootstrap, grng)
        for (j, k) in enumerate(pnames)
            insertcols!(dg, j, k => fill(row[k], nrow(dg)))
            insertcols!(sim, j, k => fill(row[k], nrow(sim)))
        end
        push!(diags, dg)
        push!(simsdf, sim)
    end
    return DesignDiagnosis(design.name, vcat(diags...; cols=:union),
                           vcat(simsdf...; cols=:union), pnames, Int(sims), float(alpha),
                           float(level))
end
