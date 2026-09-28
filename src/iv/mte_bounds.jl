# Bounds on treatment-effect parameters from IV-like estimands and shape-restricted
# marginal treatment response functions (Mogstad, Santos & Torgovitsky 2018, MST).
#
# Model: D = 1{U ≤ p(Z, X)}, U ~ U(0, 1) given (Z, X); marginal treatment responses
# m_d(u, x) = E[Y(d) | U = u, X = x] are restricted to a finite-dimensional space
#   m_d(u, x) = Σ_k θ_dk b_k(u) + Σ_l α_dl x_l + Σ_{k,l'} ϑ_dkl' c_k(u) x_l'
# (b = Bernstein polynomials or B-splines in u; c the basis of the u-varying
# covariate effects). Every IV-like estimand
# β_s = E[s(D, Z, X) Y] is linear in (θ, α, ϑ):
#   β_s = E[ s(1, Z, X) ∫₀^{p} m₁(u, X) du + s(0, Z, X) ∫_{p}^1 m₀(u, X) du ],
# and so is every target parameter τ = E[∫ ω₁ m₁ + ω₀ m₀]. MST's bounds are the
# minimum and maximum of τ over MTRs that satisfy the shape restrictions and match
# the estimated IV-like estimands; following MST (Section 4) and `ivmte`, the sample
# version first minimizes the ℓ₁ criterion Q(θ) = Σ_s |Γ_s(θ) − β̂_s| and then
# optimizes τ over {θ : Q(θ) ≤ Q̂*(1 + criterion_tol)}. Shape restrictions are imposed
# on a grid of u (and on all observed covariate values). All programs are linear and
# solved with the simplex method of src/core/lp.jl (through the dual, whose basis has
# as many rows as there are MTR coefficients).

# ---------------------------------------------------------------------------
# Basis in u
# ---------------------------------------------------------------------------

"""Internal: basis of the MTRs in `u` (Bernstein polynomial or clamped B-spline)."""
struct _IVUBasis
    kind::Symbol                  # :bernstein or :spline
    degree::Int
    knots::Vector{Float64}        # interior knots (spline)
    breaks::Vector{Float64}       # 0, knots..., 1: the polynomial pieces
end

_iv_ub_size(b::_IVUBasis) = b.kind === :bernstein ? b.degree + 1 :
                            length(b.knots) + b.degree + 1

function _iv_ub_names(b::_IVUBasis)
    b.kind === :bernstein && return ["bernstein$(k),$(b.degree)(u)" for k in 0:b.degree]
    return ["bspline$(k)(u)" for k in 1:_iv_ub_size(b)]
end

"""Basis vector at `u ∈ [0, 1]`."""
function _iv_ub_eval(b::_IVUBasis, u::Real)
    if b.kind === :bernstein
        K = b.degree
        return [binomial(K, k) * u^k * (1 - u)^(K - k) for k in 0:K]
    end
    p = b.degree
    τ = vcat(zeros(p + 1), b.knots, ones(p + 1))
    nint = length(τ) - 1
    # degree-0 indicators, right-closed at u = 1
    B = zeros(nint)
    j = u >= 1 ? findlast(k -> τ[k] < τ[k + 1], 1:nint) :
        clamp(searchsortedlast(τ, u), 1, nint)
    B[j] = 1.0
    for q in 1:p
        Bn = zeros(nint - q)
        for k in 1:(nint - q)
            d1 = τ[k + q] - τ[k]
            d2 = τ[k + q + 1] - τ[k + 1]
            a = d1 > 0 ? (u - τ[k]) / d1 * B[k] : 0.0
            c = d2 > 0 ? (τ[k + q + 1] - u) / d2 * B[k + 1] : 0.0
            Bn[k] = a + c
        end
        B = Bn
    end
    return B
end

"""`∫_a^b basis(u) du` (signed: negative when `b < a`), exact by Gauss–Legendre on
each polynomial piece."""
function _iv_ub_integral(b::_IVUBasis, a::Real, c::Real)
    a == c && return zeros(_iv_ub_size(b))
    sgn = 1.0
    if c < a
        a, c = c, a
        sgn = -1.0
    end
    nodes, wts = _iv_gauss_legendre(b.degree ÷ 2 + 1)
    out = zeros(_iv_ub_size(b))
    br = b.breaks
    for i in 1:(length(br) - 1)
        lo, hi = max(a, br[i]), min(c, br[i + 1])
        hi > lo || continue
        h = (hi - lo) / 2
        for (x, w) in zip(nodes, wts)
            out .+= (w * h) .* _iv_ub_eval(b, lo + h * (x + 1))
        end
    end
    return sgn .* out
end

function _iv_ubasis(basis::Symbol, degree::Integer, knots)
    kn = sort!(Float64.(collect(knots)))
    if basis === :bernstein
        degree >= 0 || throw(ArgumentError("degree must be non-negative"))
        isempty(kn) || throw(ArgumentError("`knots` are only used with basis = :spline"))
        return _IVUBasis(:bernstein, Int(degree), Float64[], [0.0, 1.0])
    elseif basis === :spline
        degree >= 0 || throw(ArgumentError("degree must be non-negative"))
        all(k -> 0 < k < 1, kn) && allunique(kn) ||
            throw(ArgumentError("spline knots must be distinct points in (0, 1)"))
        return _IVUBasis(:spline, Int(degree), kn, vcat(0.0, kn, 1.0))
    end
    throw(ArgumentError("basis must be :bernstein or :spline, got :$basis"))
end

# ---------------------------------------------------------------------------
# IV-like specifications
# ---------------------------------------------------------------------------

_iv_mteb_term_name(t::Symbol) = string(t)
_iv_mteb_term_name(t::Tuple) = join(string.(t), "×")
_iv_mteb_term_name(t) = throw(ArgumentError("IV-like terms must be column names " *
                                            "(Symbol) or tuples of column names " *
                                            "(products), got $(repr(t))"))

"""Columns of an IV-like design (intercept first) with the treatment set to `dval`
(`nothing`: observed treatment)."""
function _iv_mteb_design(sub, terms, treatment::Symbol, dval)
    n = nrow(sub)
    col(s::Symbol) = s === treatment && dval !== nothing ? fill(float(dval), n) :
                     Float64.(sub[!, s])
    cols = [ones(n)]
    for t in terms
        if t isa Symbol
            push!(cols, col(t))
        elseif t isa Tuple
            push!(cols, reduce((a, s) -> a .* col(s), t; init=ones(n)))
        else
            _iv_mteb_term_name(t)
        end
    end
    return reduce(hcat, cols)
end

_iv_mteb_terms_symbols(terms) =
    unique(Symbol[s for t in terms for s in (t isa Tuple ? t : (t,)) if s isa Symbol])

function _iv_mteb_spec(spec)
    spec isa NamedTuple ||
        throw(ArgumentError("each IV-like specification must be a NamedTuple " *
                            "(regressors = [...], instruments = [...], " *
                            "components = [...])"))
    haskey(spec, :regressors) ||
        throw(ArgumentError("an IV-like specification needs `regressors`"))
    regs = collect(spec.regressors)
    insts = haskey(spec, :instruments) && spec.instruments !== nothing ?
            collect(spec.instruments) : nothing
    comps = haskey(spec, :components) && spec.components !== nothing ?
            collect(spec.components) : nothing
    return regs, insts, comps
end

"""
IV-like estimands of one specification: weighted OLS (no instruments), IV (as many
instruments as regressors) or 2SLS. Returns `(β̂, S₀, S₁, names)` where the columns of
`S_d` are `s(d, Zᵢ, Xᵢ)` for the selected components.
"""
function _iv_mteb_ivlike(sub, spec, treatment, y, w)
    regs, insts, comps = _iv_mteb_spec(spec)
    Xa = _iv_mteb_design(sub, regs, treatment, nothing)
    X0 = _iv_mteb_design(sub, regs, treatment, 0)
    X1 = _iv_mteb_design(sub, regs, treatment, 1)
    if insts === nothing
        Za, Z0, Z1 = Xa, X0, X1
    else
        Za = _iv_mteb_design(sub, insts, treatment, nothing)
        Z0 = _iv_mteb_design(sub, insts, treatment, 0)
        Z1 = _iv_mteb_design(sub, insts, treatment, 1)
    end
    W = sum(w)
    k, l = size(Xa, 2), size(Za, 2)
    l >= k || throw(ArgumentError("IV-like specification with fewer instruments than " *
                                  "regressors"))
    EZX = Za' * (w .* Xa) ./ W
    if l == k
        rank(EZX) == k || throw(ArgumentError("IV-like specification is not identified " *
                                              "(singular moment matrix)"))
        M = inv(EZX)
    else
        EZZ = Za' * (w .* Za) ./ W
        Π = EZX' / Symmetric(EZZ)
        A = Π * EZX
        rank(A) == k || throw(ArgumentError("IV-like 2SLS specification is not " *
                                            "identified"))
        M = A \ Π
    end
    names = vcat("intercept", [_iv_mteb_term_name(t) for t in regs])
    sel = if comps === nothing
        collect(1:k)
    else
        map(comps) do c
            nm = c === :intercept ? "intercept" : _iv_mteb_term_name(c)
            j = findfirst(==(nm), names)
            j === nothing && throw(ArgumentError("component $(repr(c)) is not a " *
                                                 "regressor of the IV-like " *
                                                 "specification"))
            j
        end
    end
    Sa = (Za * M')[:, sel]
    S0 = (Z0 * M')[:, sel]
    S1 = (Z1 * M')[:, sel]
    β = vec(sum(w .* Sa .* y; dims=1)) ./ W
    return β, S0, S1, names[sel]
end

# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------

"""
    MTEBounds

Result of [`mte_bounds`](@ref): estimated bounds on a target treatment-effect
parameter from IV-like estimands and shape-restricted marginal treatment response
functions (Mogstad, Santos and Torgovitsky 2018).

The bounds are the smallest and largest values of the target parameter over all
marginal treatment response (MTR) functions in the chosen finite-dimensional space
that satisfy the shape restrictions and reproduce the estimated IV-like estimands (up
to the tolerance on the minimum criterion). They are sharp given the specification
(the MTR basis, the shape restrictions and the IV-like estimands used), not sharp
with respect to all information in the data unless the specification is rich enough;
they are sample analogues, and no confidence interval is attached. When the upper and
lower bounds coincide numerically the parameter is point identified under the
specification. The fields `moments` and `mtr_coef` allow checking how well the
extreme MTRs fit the data.

# Fields
- `lower::Float64`, `upper::Float64`: the bounds (`±Inf` when the linear program is
  unbounded).
- `target::String`: description of the target parameter.
- `point_identified::Bool`: `true` when `upper − lower` is below numerical tolerance.
- `min_criterion::Float64`: the minimal ``\\ell_1`` distance ``\\sum_s |\\Gamma_s(\\theta)
  - \\hat\\beta_s|`` between the IV-like estimands implied by admissible MTRs and their
  estimates (0 when some admissible MTR matches all of them; a large value signals
  misspecification of the MTR model).
- `criterion_tol::Float64`: the bounds are taken over MTRs with criterion at most
  `min_criterion × (1 + criterion_tol)`.
- `moments::DataFrame`: the IV-like estimands (`spec`, `component`, `estimate`) and
  the values implied by the MTRs attaining the lower and the upper bound
  (`implied_lower`, `implied_upper`).
- `mtr_coef::NamedTuple`: MTR coefficients at the two optima, `(lower = (m0, m1),
  upper = (m0, m1))`; `mtr_terms::Vector{String}` names them.
- `basis`, `interact_basis`: the MTR bases in ``u`` (internal objects);
  `covariates::Vector{Symbol}`, `interact::Vector{Symbol}`: the MTR covariates and
  those with ``u``-varying effects.
- `u_grid::Vector{Float64}`: the grid on which the shape restrictions are imposed.
- `nobs::Int`: number of observations used.
- `propensity_range::Tuple{Float64,Float64}`: minimum and maximum fitted propensity.

# References
- Mogstad, M., Santos, A., & Torgovitsky, A. (2018). Using instrumental variables for
  inference about policy relevant treatment parameters. *Econometrica*, 86(5),
  1589–1619.
"""
struct MTEBounds
    lower::Float64
    upper::Float64
    target::String
    point_identified::Bool
    min_criterion::Float64
    criterion_tol::Float64
    moments::DataFrame
    mtr_coef::NamedTuple
    mtr_terms::Vector{String}
    basis::_IVUBasis
    interact_basis::_IVUBasis
    covariates::Vector{Symbol}
    interact::Vector{Symbol}
    u_grid::Vector{Float64}
    nobs::Int
    propensity_range::Tuple{Float64,Float64}
end

function Base.show(io::IO, ::MIME"text/plain", r::MTEBounds)
    println(io, "MTE bounds (Mogstad, Santos & Torgovitsky 2018)")
    println(io, "Target: ", r.target)
    if r.point_identified
        @printf(io, "Point identified: %.6g\n", r.lower)
    else
        println(io, "Bounds: [", _iv_fmt(r.lower), ", ", _iv_fmt(r.upper), "]")
    end
    b = r.basis
    println(io, "MTRs: ", b.kind === :bernstein ? "Bernstein polynomial of degree " *
                                                 "$(b.degree)" :
                "B-spline of degree $(b.degree) with knots $(b.knots)",
            isempty(r.covariates) ? "" : " + " * join(r.covariates, " + "),
            isempty(r.interact) ? "" : " (u-varying: " * join(r.interact, ", ") * ")")
    @printf(io, "IV-like estimands: %d; minimum criterion: %.4g; observations: %d\n",
            nrow(r.moments), r.min_criterion, r.nobs)
    @printf(io, "Propensity range: [%.3f, %.3f]; shape constraints on %d u-points\n",
            r.propensity_range[1], r.propensity_range[2], length(r.u_grid))
    println(io, "Note: estimated bounds (sample analogues); no confidence interval.")
end

Base.show(io::IO, r::MTEBounds) =
    print(io, "MTEBounds(", r.target, ": [", _iv_fmt(r.lower), ", ", _iv_fmt(r.upper), "])")

# ---------------------------------------------------------------------------
# Propensity
# ---------------------------------------------------------------------------

"""Weighted propensity model; returns fitted values and a predictor for
counterfactual instrument values."""
function _iv_mteb_propensity(sub, d, inst, covs, w, link)
    link in (:logit, :probit, :linear) ||
        throw(ArgumentError("link must be :logit, :probit or :linear, got :$link"))
    Xc = _iv_exog_matrix(sub, covs, true)
    Zi = Matrix{Float64}(hcat([Float64.(sub[!, c]) for c in inst]...))
    A = hcat(Xc, Zi)
    F = qr(A, ColumnNorm())
    dR = abs.(diag(F.R))
    r = count(>(dR[1] * 1e-10), dR)
    keep = sort(F.p[1:r])
    Ak = A[:, keep]
    if link === :linear
        coefv = (Ak' * (w .* Ak)) \ (Ak' * (w .* d))
        invlink = identity
    else
        lk = link === :probit ? ProbitLink() : LogitLink()
        m = GLM.glm(Ak, d, Binomial(), lk; wts=w)
        coefv = GLM.coef(m)
        invlink = η -> GLM.linkinv(lk, η)
    end
    p = invlink.(Ak * coefv)
    function predict(values::AbstractDict)
        Zc = copy(Zi)
        for (j, c) in enumerate(inst)
            haskey(values, c) && (Zc[:, j] .= float(values[c]))
        end
        return invlink.(hcat(Xc, Zc)[:, keep] * coefv)
    end
    return p, predict
end

_iv_mteb_dict(x::AbstractDict) = Dict{Symbol,Any}(Symbol(k) => v for (k, v) in x)
_iv_mteb_dict(x::NamedTuple) = Dict{Symbol,Any}(pairs(x))
_iv_mteb_dict(x) = throw(ArgumentError("late_from / late_to must map instrument " *
                                       "names to values (NamedTuple or Dict)"))

# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------

"""
    mte_bounds(data, outcome, treatment, instruments;
               target=:ate, ivlike=nothing, covariates=Symbol[], interact=Symbol[],
               interact_degree=nothing, link=:logit, propensity=nothing,
               basis=:bernstein, degree=2, knots=Float64[],
               mtr_bounds=:outcome_range, mte_range=nothing,
               m0_monotone=:none, m1_monotone=:none, mte_monotone=:none,
               late_from=nothing, late_to=nothing, u_interval=nothing,
               policy_propensity=nothing, u_grid=nothing, criterion_tol=1e-4,
               weights=nothing) -> MTEBounds

Bounds on treatment-effect parameters that the instrument does not point identify,
by linear programming over marginal treatment response functions (Mogstad, Santos
and Torgovitsky 2018).

An instrument point identifies the LATE of the compliers it moves, but policy
questions often concern other parameters: the ATE, the ATT, the effect of a policy
that shifts treatment take-up differently, or the LATE of a different instrument
change. Mogstad, Santos and Torgovitsky (2018, MST) show that, in the latent-index
model of Heckman and Vytlacil (2005), both the IV-like estimands the data identify
and such target parameters are linear functionals of the marginal treatment response
(MTR) functions ``m_d(u, x) = E[Y(d) \\mid U = u, X = x]``. The set of target values
consistent with the data and with a priori restrictions on the MTRs is therefore an
interval whose endpoints solve two linear programs. The bounds make explicit how
much a conclusion about the target owes to the data and how much to the assumptions,
and they collapse to a point when the assumptions are strong enough (for instance, a
linear MTE with a sufficiently rich instrument, as in Brinch, Mogstad and Wiswall
2017).

**Model and MTR specification.** Treatment is ``D = 1\\{U \\le p(Z, X)\\}`` with
``U \\sim U(0, 1)`` independent of the instruments given the covariates. The MTRs lie in
the linear space

```math
m_d(u, x) = \\sum_k \\theta_{dk} b_k(u) + \\sum_{l \\notin \\mathcal{I}} \\alpha_{dl} x_l
  + \\sum_{k,\\, l \\in \\mathcal{I}} \\vartheta_{dkl}\\, c_k(u) x_l ,
```

with ``b`` Bernstein polynomials of degree `degree` (`basis = :bernstein`; the same
space as polynomials in ``u``) or B-splines of degree `degree` with interior `knots`
(`basis = :spline`; `degree = 0` gives piecewise-constant MTRs, the nonparametric case
of MST when the knots include the propensity values). The `covariates` enter
additively (and also enter the propensity score); those listed in `interact`
(``\\mathcal{I}``) have effects that vary with ``u`` through a basis ``c(u)``: Bernstein
polynomials of degree `interact_degree` (`nothing`: `degree`; `interact_degree = 1`
gives ``x + u x``) or, for splines, the same spline basis; the level of an interacted
covariate is part of ``c(u) x``.

**IV-like estimands.** Each specification in `ivlike` is a linear regression whose
coefficients are IV-like estimands ``\\beta_s = E[s(D, Z, X) Y]``, written as a named
tuple `(regressors = [...], instruments = [...], components = [...])`. An intercept is
always included; terms are column names or tuples of column names (products, e.g.
`(:d, :z)`); the treatment column may appear among the regressors; `instruments =
nothing` (or absent) means OLS, as many instruments as regressors an IV regression,
more a 2SLS regression; `components` selects coefficients (`:intercept` for the
intercept; default all). The default is the IV regression of ``Y`` on ``(1, D)`` with
instruments ``(1, Z)``, which is 2SLS with several instruments. MST show that more,
and more flexible, IV-like estimands give (weakly) tighter bounds; with a discrete
instrument, a saturated OLS regression (``D``, instrument dummies and their products)
exhausts the information in ``E[Y \\mid D, Z]``.

**Targets** (`target`): `:ate`; `:att` and `:atu` (weights ``1\\{u \\le p\\}/P(D = 1)``
and ``1\\{u > p\\}/P(D = 0)``); `:late` for the instrument change `late_from →
late_to` (the average MTE between the two implied propensities, averaged over the
covariates as in `ivmte`); `:genlate` for the average MTE over `u_interval = (a, b)`;
and `:prte` for the policy that moves each observation's propensity to
`policy_propensity`.

**Shape restrictions**, imposed on the grid `u_grid` for every observed covariate
value: bounds on the MTRs (`mtr_bounds`, by default the range of the observed
outcome, as in `ivmte`, which is a logical restriction for a bounded outcome), bounds
on the MTE (`mte_range`), and monotonicity in ``u`` of ``m_0``, ``m_1`` and the MTE.
Such restrictions are assumptions, not implications of the data, and should be
motivated and varied.

**Estimation.** The propensity score is fitted by `link` on the covariates and
instruments, or taken from the column `propensity`. Following MST (Section 4) and the
R package `ivmte` (Shea and Torgovitsky 2023), the ``\\ell_1`` criterion ``Q(\\theta)
= \\sum_s |\\Gamma_s(\\theta) - \\hat\\beta_s|`` is first minimized over admissible
MTRs, and the bounds are the minimum and maximum of the target over admissible MTRs
with ``Q(\\theta) \\le \\hat Q^*(1 + \\tau)``, where ``\\tau`` is `criterion_tol`;
this guards against an empty identified set in finite samples. All programs are solved
by the simplex method. With `u_grid` equal to `ivmte`'s audit grid the results
reproduce `ivmte` (the constraints are the same finite set), which is how the function
is validated; a finer grid imposes the shape restrictions more densely and gives
(weakly) tighter bounds.

Only estimated bounds are reported: sampling uncertainty in the propensity score and
the IV-like estimates is not accounted for, and MST's inference procedure is not
implemented. Bounds that are very sensitive to the basis, the grid or
`criterion_tol` indicate weak identification. When the MTE is instead assumed to
have a parametric form, [`mte`](@ref) point estimates it by local IV.

# Arguments
- `data::AbstractDataFrame`: the data; incomplete rows are dropped.
- `outcome::Symbol`: the outcome ``Y``.
- `treatment::Symbol`: the binary (0/1) treatment ``D``.
- `instruments`: the numeric instrument column(s) ``Z``.

# Keywords
- `target::Symbol`: `:ate` (default), `:att`, `:atu`, `:late`, `:genlate` or `:prte`.
- `ivlike`: an IV-like specification (named tuple) or a vector of them; default
  `nothing`, the IV regression of ``Y`` on ``(1, D)`` with instruments ``(1, Z)``.
- `covariates::Vector{Symbol}`: MTR and propensity covariates (numeric or
  categorical; categorical ones are dummy-coded; default none).
- `interact::Vector{Symbol}`: the subset of `covariates` with ``u``-varying effects
  (default none); `interact_degree`: degree of their Bernstein basis (default
  `nothing`, i.e. `degree`).
- `link::Symbol`: propensity model, `:logit` (default), `:probit` or `:linear`.
- `propensity::Union{Nothing,Symbol}`: a column with precomputed propensity scores,
  used instead of `link` (default `nothing`); not available with `target = :late`.
- `basis::Symbol`: `:bernstein` (default) or `:spline`; `degree::Integer`: its degree
  (default 2); `knots`: interior spline knots (default none).
- `mtr_bounds`: `:outcome_range` (default), `nothing` (no bounds) or `(lo, hi)`.
- `mte_range`: `nothing` (default) or `(lo, hi)`, bounds on the MTE.
- `m0_monotone`, `m1_monotone`, `mte_monotone`: `:none` (default), `:increasing` or
  `:decreasing`, monotonicity in ``u``.
- `late_from`, `late_to`: instrument values (named tuples or dictionaries keyed by
  instrument name) defining the `:late` target.
- `u_interval`: `(a, b)` with ``0 \\le a < b \\le 1`` for the `:genlate` target.
- `policy_propensity`: for `:prte`, a vector with one propensity per estimation-sample
  row or a function of the fitted propensity vector.
- `u_grid`: the constraint grid in ``[0, 1]`` (default 101 equally spaced points plus
  the spline knots).
- `criterion_tol::Real`: relative tolerance on the minimum criterion (default
  `1e-4`).
- `weights::Union{Nothing,Symbol}`: frequency or sampling weights (default
  `nothing`).

# Returns
- An [`MTEBounds`](@ref); `b.lower` and `b.upper` are the bounds, `b.moments` compares
  the IV-like estimates with the values implied by the extreme MTRs.

# Examples
```julia
using DrSnow, DataFrames, StableRNGs
rng = StableRNG(6)
n = 5_000
z = rand(rng, 0:2, n)                              # three-valued instrument
ud = rand(rng, n)
d = Float64.(ud .<= [0.2, 0.45, 0.7][z .+ 1])
y1 = Float64.(rand(rng, n) .< 0.8 .- 0.4 .* ud)    # binary outcome, true ATE = 0.3
y0 = Float64.(rand(rng, n) .< 0.4 .- 0.2 .* ud)
df = DataFrame(y=ifelse.(d .== 1, y1, y0), d=d, z=z)
b = mte_bounds(df, :y, :d, :z; target=:ate, degree=2)
b.lower, b.upper
# more IV-like estimands and monotone MTRs tighten the bounds
spec = (regressors=[:d, :z, (:d, :z)],)
mte_bounds(df, :y, :d, :z; target=:ate, ivlike=spec, degree=2,
           m0_monotone=:decreasing, m1_monotone=:decreasing)
mte_bounds(df, :y, :d, :z; target=:late, late_from=(z=0,), late_to=(z=2,))
```

# References
- Mogstad, M., Santos, A., & Torgovitsky, A. (2018). Using instrumental variables for
  inference about policy relevant treatment parameters. *Econometrica*, 86(5),
  1589–1619.
- Heckman, J. J., & Vytlacil, E. (2005). Structural equations, treatment effects,
  and econometric policy evaluation. *Econometrica*, 73(3), 669–738.
- Shea, J., & Torgovitsky, A. (2023). ivmte: An R package for extrapolating
  instrumental variable estimates away from compliers. *Observational Studies*,
  9(2), 1–42.
- Brinch, C. N., Mogstad, M., & Wiswall, M. (2017). Beyond LATE with a discrete
  instrument. *Journal of Political Economy*, 125(4), 985–1039.
"""
function mte_bounds(data::AbstractDataFrame, outcome::Symbol, treatment::Symbol,
                    instruments; target::Symbol=:ate, ivlike=nothing,
                    covariates=Symbol[], interact=Symbol[],
                    interact_degree::Union{Nothing,Integer}=nothing,
                    link::Symbol=:logit,
                    propensity::Union{Nothing,Symbol}=nothing, basis::Symbol=:bernstein,
                    degree::Integer=2, knots=Float64[], mtr_bounds=:outcome_range,
                    mte_range=nothing, m0_monotone::Symbol=:none,
                    m1_monotone::Symbol=:none, mte_monotone::Symbol=:none,
                    late_from=nothing, late_to=nothing, u_interval=nothing,
                    policy_propensity=nothing, u_grid=nothing,
                    criterion_tol::Real=1e-4, weights::Union{Nothing,Symbol}=nothing)
    ctx = "mte_bounds"
    target in (:ate, :att, :atu, :late, :genlate, :prte) ||
        throw(ArgumentError("$ctx: target must be :ate, :att, :atu, :late, :genlate " *
                            "or :prte, got :$target"))
    for (nm, v) in (("m0_monotone", m0_monotone), ("m1_monotone", m1_monotone),
                    ("mte_monotone", mte_monotone))
        v in (:none, :increasing, :decreasing) ||
            throw(ArgumentError("$ctx: $nm must be :none, :increasing or :decreasing"))
    end
    criterion_tol >= 0 || throw(ArgumentError("$ctx: criterion_tol must be ≥ 0"))
    inst = _as_symbols(instruments)
    isempty(inst) && throw(ArgumentError("$ctx: at least one instrument is required"))
    covs, inter = _as_symbols(covariates), _as_symbols(interact)
    all(in(covs), inter) ||
        throw(ArgumentError("$ctx: `interact` must be a subset of `covariates`"))
    specs = ivlike === nothing ?
            [(regressors=[treatment], instruments=collect(inst))] :
            ivlike isa NamedTuple ? [ivlike] : collect(ivlike)
    isempty(specs) && throw(ArgumentError("$ctx: no IV-like specification"))
    spec_cols = Symbol[]
    for sp in specs
        regs, insts, _ = _iv_mteb_spec(sp)
        append!(spec_cols, _iv_mteb_terms_symbols(regs))
        insts === nothing || append!(spec_cols, _iv_mteb_terms_symbols(insts))
    end
    cols = unique(vcat([outcome, treatment], inst, covs, spec_cols,
                       weights === nothing ? Symbol[] : [weights],
                       propensity === nothing ? Symbol[] : [propensity]))
    require_columns(data, cols; context=ctx)
    _iv_check_numeric(data, vcat([outcome, treatment], inst, spec_cols), ctx)
    mask = trues(nrow(data))
    for c in cols
        mask .&= .!ismissing.(data[!, c])
    end
    sub = disallowmissing(data[mask, cols])
    n = nrow(sub)
    n >= 10 || throw(ArgumentError("$ctx: too few complete observations ($n)"))
    y = Float64.(sub[!, outcome])
    d = Float64.(sub[!, treatment])
    all(x -> x == 0 || x == 1, d) ||
        throw(ArgumentError("$ctx: treatment `$treatment` must be binary (0/1)"))
    0 < sum(d) < n || throw(ArgumentError("$ctx: the treatment does not vary"))
    w = weights === nothing ? ones(n) : Float64.(sub[!, weights])
    all(>(0), w) || throw(ArgumentError("$ctx: weights must be positive"))
    W = sum(w)
    # propensity
    if propensity === nothing
        p, pred = _iv_mteb_propensity(sub, d, inst, covs, w, link)
    else
        p = Float64.(sub[!, propensity])
        pred = nothing
    end
    all(x -> 0 <= x <= 1, p) ||
        throw(ArgumentError("$ctx: propensity scores outside [0, 1]; use a logit or " *
                            "probit link"))
    ub = _iv_ubasis(basis, degree, knots)
    nb = _iv_ub_size(ub)
    ubi = basis === :bernstein ?
          _iv_ubasis(:bernstein, something(interact_degree, degree), Float64[]) : ub
    nbi = _iv_ub_size(ubi)
    # covariate part of the MTR design (interacted covariates only through c(u)·x)
    addc = [c for c in covs if !(c in inter)]
    Xadd = isempty(addc) ? zeros(n, 0) : _iv_exog_matrix(sub, addc, false)
    Xint = isempty(inter) ? zeros(n, 0) : _iv_exog_matrix(sub, inter, false)
    nadd, nint = size(Xadd, 2), size(Xint, 2)
    q = nb + nadd + nbi * nint
    bnames = _iv_ub_names(ub)
    addnames = isempty(addc) ? String[] : _iv_mteb_colnames(sub, addc)
    intnames = isempty(inter) ? String[] : _iv_mteb_colnames(sub, inter)
    terms = vcat(bnames, addnames,
                 [b * "×" * x for x in intnames for b in _iv_ub_names(ubi)])
    # integrals of the MTR design over [a_i, b_i], using unique endpoints
    cache = Dict{Tuple{Float64,Float64},Vector{Float64}}()
    integ(a, c) = get!(() -> _iv_ub_integral(ub, a, c), cache, (a, c))
    cachei = Dict{Tuple{Float64,Float64},Vector{Float64}}()
    integi(a, c) = get!(() -> _iv_ub_integral(ubi, a, c), cachei, (a, c))
    function design_int(a::Vector{Float64}, c::Vector{Float64})
        R = zeros(n, q)
        for i in 1:n
            R[i, 1:nb] = integ(a[i], c[i])
            len = c[i] - a[i]
            for l in 1:nadd
                R[i, nb + l] = len * Xadd[i, l]
            end
            if nint > 0
                Ic = integi(a[i], c[i])
                for l in 1:nint, k in 1:nbi
                    R[i, nb + nadd + (l - 1) * nbi + k] = Ic[k] * Xint[i, l]
                end
            end
        end
        return R
    end
    R1 = design_int(zeros(n), p)          # ∫₀^p (treated)
    R0 = design_int(p, ones(n))           # ∫_p^1 (untreated)
    # IV-like estimands and their linear maps Γ
    βs = Float64[]
    Γ = zeros(0, 2q)
    mom = DataFrame(spec=Int[], component=String[], estimate=Float64[])
    for (j, sp) in enumerate(specs)
        β, S0, S1, names = _iv_mteb_ivlike(sub, sp, treatment, y, w)
        for c in eachindex(β)
            g0 = vec(sum(w .* S0[:, c] .* R0; dims=1)) ./ W
            g1 = vec(sum(w .* S1[:, c] .* R1; dims=1)) ./ W
            Γ = vcat(Γ, vcat(g0, g1)')
            push!(βs, β[c])
            push!(mom, (j, names[c], β[c]))
        end
    end
    # target
    tlabel, a_t, b_t, c_t = _iv_mteb_target(target, p, d, w, pred, late_from, late_to,
                                            u_interval, policy_propensity, ctx)
    Rt = design_int(a_t, b_t)
    τ1 = vec(sum(w .* c_t .* Rt; dims=1)) ./ W
    τ = vcat(-τ1, τ1)
    # constraint grid
    grid = if u_grid === nothing
        sort!(unique(vcat(collect(range(0, 1; length=101)), ub.knots)))
    else
        g = sort!(unique(Float64.(collect(u_grid))))
        all(x -> 0 <= x <= 1, g) || throw(ArgumentError("$ctx: u_grid must lie in [0, 1]"))
        length(g) >= 2 || throw(ArgumentError("$ctx: u_grid needs at least 2 points"))
        g
    end
    xs = unique(hcat(Xadd, Xint); dims=1)
    Bg = reduce(vcat, (_iv_ub_eval(ub, u)' for u in grid))       # Nu × nb
    Bgi = reduce(vcat, (_iv_ub_eval(ubi, u)' for u in grid))     # Nu × nbi
    Ab, bb = _iv_mteb_shape(Bg, Bgi, xs, nadd, nint, q, mtr_bounds, mte_range,
                            m0_monotone, m1_monotone, mte_monotone, y, ctx)
    lo, hi, qmin, θlo, θhi = _iv_mteb_solve(Γ, βs, τ, Ab, bb, float(criterion_tol), ctx)
    point = isfinite(lo) && isfinite(hi) && hi - lo <= 1e-7 * max(1.0, abs(lo), abs(hi))
    mom.implied_lower = θlo === nothing ? fill(NaN, nrow(mom)) : Γ * θlo
    mom.implied_upper = θhi === nothing ? fill(NaN, nrow(mom)) : Γ * θhi
    split_(θ) = θ === nothing ? (m0=Float64[], m1=Float64[]) :
                (m0=θ[1:q], m1=θ[(q + 1):end])
    return MTEBounds(lo, hi, tlabel, point, qmin, float(criterion_tol), mom,
                     (lower=split_(θlo), upper=split_(θhi)), terms, ub, ubi, covs,
                     inter, grid, n, (minimum(p), maximum(p)))
end

function _iv_mteb_colnames(sub, covs)
    f = make_formula(:__iv_lhs__, covs; intercept=false)
    ts = f.rhs isa Tuple ? f.rhs : (f.rhs,)
    sch = StatsModels.schema(ts, sub)
    rhs = StatsModels.MatrixTerm(StatsModels.apply_schema(ts, sch,
                                                          StatsModels.StatisticalModel))
    nm = StatsModels.coefnames(rhs)
    return nm isa AbstractString ? [nm] : collect(String, nm)
end

"""Target interval `[aᵢ, bᵢ]` and multiplier `cᵢ` per observation, so that
`τ = E_n[cᵢ ∫_{aᵢ}^{bᵢ} (m₁ − m₀)(u, Xᵢ) du]`."""
function _iv_mteb_target(target, p, d, w, pred, late_from, late_to, u_interval,
                         policy_propensity, ctx)
    n = length(p)
    W = sum(w)
    if target === :ate
        return "ATE", zeros(n), ones(n), ones(n)
    elseif target === :att
        return "ATT", zeros(n), copy(p), fill(W / sum(w .* d), n)
    elseif target === :atu
        return "ATU", copy(p), ones(n), fill(W / sum(w .* (1 .- d)), n)
    elseif target === :genlate
        u_interval === nothing &&
            throw(ArgumentError("$ctx: target = :genlate needs `u_interval = (a, b)`"))
        a, b = float.(u_interval)
        0 <= a < b <= 1 || throw(ArgumentError("$ctx: u_interval must satisfy " *
                                               "0 ≤ a < b ≤ 1"))
        return @sprintf("generalized LATE on u ∈ [%g, %g]", a, b), fill(a, n),
               fill(b, n), fill(1 / (b - a), n)
    elseif target === :late
        (late_from === nothing || late_to === nothing) &&
            throw(ArgumentError("$ctx: target = :late needs `late_from` and `late_to`"))
        pred === nothing &&
            throw(ArgumentError("$ctx: target = :late needs a fitted propensity model " *
                                "(not a `propensity` column)"))
        pf = pred(_iv_mteb_dict(late_from))
        pt = pred(_iv_mteb_dict(late_to))
        lo, hi = min.(pf, pt), max.(pf, pt)
        all(hi .- lo .> 1e-10) ||
            throw(ArgumentError("$ctx: the LATE instrument change does not move the " *
                                "propensity score for some observations"))
        return "LATE for $(late_from) → $(late_to)", lo, hi, 1 ./ (hi .- lo)
    else # :prte
        policy_propensity === nothing &&
            throw(ArgumentError("$ctx: target = :prte needs `policy_propensity`"))
        pp = policy_propensity isa Function ? Float64.(policy_propensity(copy(p))) :
             Float64.(collect(policy_propensity))
        length(pp) == n || throw(ArgumentError("$ctx: policy_propensity must have one " *
                                               "entry per estimation-sample row ($n)"))
        all(x -> 0 <= x <= 1, pp) ||
            throw(ArgumentError("$ctx: policy propensities must lie in [0, 1]"))
        Δ = sum(w .* (pp .- p)) / W
        abs(Δ) > 1e-10 || throw(ArgumentError("$ctx: the policy does not change the " *
                                              "average propensity"))
        return "PRTE", copy(p), pp, fill(1 / Δ, n)
    end
end

"""Shape constraints `A θ ≤ b` on the grid, θ = [θ₀; θ₁] (length 2q)."""
function _iv_mteb_shape(Bg, Bgi, xs, nadd, nint, q, mtr_bounds, mte_range, mono0,
                        mono1, monote, y, ctx)
    Nu, nb = size(Bg)
    nbi = size(Bgi, 2)
    nx = size(xs, 1)
    # design rows r(u, x) for all grid points × covariate values
    rows = Matrix{Float64}(undef, Nu * max(nx, 1), q)
    idx = 0
    for ix in 1:max(nx, 1), g in 1:Nu
        idx += 1
        rows[idx, 1:nb] = Bg[g, :]
        for l in 1:nadd
            rows[idx, nb + l] = xs[ix, l]
        end
        for l in 1:nint, k in 1:nbi
            rows[idx, nb + nadd + (l - 1) * nbi + k] = Bgi[g, k] * xs[ix, nadd + l]
        end
    end
    Z = zeros(size(rows))
    blocks = Matrix{Float64}[]
    rhs = Float64[]
    lohi = if mtr_bounds === :outcome_range
        (minimum(y), maximum(y))
    elseif mtr_bounds === nothing
        nothing
    else
        length(mtr_bounds) == 2 && mtr_bounds[1] <= mtr_bounds[2] ||
            throw(ArgumentError("$ctx: mtr_bounds must be (lo, hi) with lo ≤ hi, " *
                                ":outcome_range or nothing"))
        float.(Tuple(mtr_bounds))
    end
    if lohi !== nothing
        lo, hi = lohi
        for (A0, A1) in ((rows, Z), (Z, rows))          # m₀ then m₁
            push!(blocks, hcat(A0, A1)); append!(rhs, fill(hi, size(rows, 1)))
            push!(blocks, -hcat(A0, A1)); append!(rhs, fill(-lo, size(rows, 1)))
        end
    end
    if mte_range !== nothing
        length(mte_range) == 2 && mte_range[1] <= mte_range[2] ||
            throw(ArgumentError("$ctx: mte_range must be (lo, hi) with lo ≤ hi"))
        push!(blocks, hcat(-rows, rows)); append!(rhs, fill(float(mte_range[2]),
                                                            size(rows, 1)))
        push!(blocks, hcat(rows, -rows)); append!(rhs, fill(-float(mte_range[1]),
                                                            size(rows, 1)))
    end
    # monotonicity: differences between consecutive grid points (for each x only
    # when the u-profile depends on x)
    nxm = nint > 0 ? max(nx, 1) : 1
    Dif = Matrix{Float64}(undef, (Nu - 1) * nxm, q)
    idx = 0
    for ix in 1:nxm, g in 1:(Nu - 1)
        idx += 1
        r1 = rows[(ix - 1) * Nu + g + 1, :]
        r0 = rows[(ix - 1) * Nu + g, :]
        Dif[idx, :] = r1 .- r0
    end
    Zd = zeros(size(Dif))
    for (mono, blk) in ((mono0, hcat(Dif, Zd)), (mono1, hcat(Zd, Dif)),
                        (monote, hcat(-Dif, Dif)))
        mono === :none && continue
        push!(blocks, mono === :increasing ? -blk : blk)
        append!(rhs, zeros(size(blk, 1)))
    end
    A = isempty(blocks) ? zeros(0, 2q) : reduce(vcat, blocks)
    return A, rhs
end

"""
Criterion and bound linear programs. Variables `[θ (2q); t (S)]`; moments enter as
`±(Γθ − β̂) ≤ t`. Returns `(lower, upper, Q̂*, θ_lower, θ_upper)`.
"""
function _iv_mteb_solve(Γ, β, τ, Ab, bb, tol, ctx)
    # Drop MTR coefficients that are linear combinations of others in every row of
    # the problem (moments, target, shape constraints): the attainable values of the
    # target are unchanged, and the simplex works with a full-rank system.
    full = size(Γ, 2)
    Mall = vcat(Γ, τ', Ab)
    F = qr(Mall, ColumnNorm())
    dR = abs.(diag(F.R))
    r = count(>(maximum(dR; init=0.0) * max(size(Mall)...) * eps() * 1e2), dR)
    keepc = sort(F.p[1:r])
    Γ, τ, Ab = Γ[:, keepc], τ[keepc], Ab[:, keepc]
    expand(θ) = θ === nothing ? nothing : (v = zeros(full); v[keepc] = θ; v)
    lo, hi, qmin, θl, θh = _iv_mteb_solve_reduced(Γ, β, τ, Ab, bb, tol, ctx)
    return lo, hi, qmin, expand(θl), expand(θh)
end

function _iv_mteb_solve_reduced(Γ, β, τ, Ab, bb, tol, ctx)
    S, nθ = size(Γ)
    # column scaling of θ for numerical stability
    scale = [max(maximum(abs, view(Γ, :, j); init=0.0), abs(τ[j]),
                 maximum(abs, view(Ab, :, j); init=0.0)) for j in 1:nθ]
    scale = [s > 0 ? 1 / s : 1.0 for s in scale]
    Γs = Γ .* scale'
    Abs = Ab .* scale'
    τs = τ .* scale
    Is = Matrix{Float64}(I, S, S)
    A = vcat(hcat(Abs, zeros(size(Abs, 1), S)),
             hcat(Γs, -Is), hcat(-Γs, -Is),
             hcat(zeros(S, nθ), -Is))
    b = vcat(bb, β, -β, zeros(S))
    Aeq = zeros(0, nθ + S)
    beq = Float64[]
    c = vcat(zeros(nθ), ones(S))
    st, qmin, _ = _lp_free_dual(c, A, b, Aeq, beq; maximize=false)
    st === :infeasible &&
        throw(ArgumentError("$ctx: the shape restrictions are infeasible (no MTR " *
                            "satisfies them on the grid)"))
    st === :optimal ||
        throw(ErrorException("$ctx: the criterion linear program failed ($st)"))
    qmin = max(qmin, 0.0)
    cap = qmin * (1 + tol) + 1e-10 * max(1.0, maximum(abs, β; init=0.0))
    A2 = vcat(A, hcat(zeros(1, nθ), ones(1, S)))
    b2 = vcat(b, cap)
    cobj = vcat(τs, zeros(S))
    stl, lo, xl = _lp_free_dual(cobj, A2, b2, Aeq, beq; maximize=false)
    sth, hi, xh = _lp_free_dual(cobj, A2, b2, Aeq, beq; maximize=true)
    for s_ in (stl, sth)
        s_ in (:optimal, :unbounded) ||
            throw(ErrorException("$ctx: a bound linear program failed ($s_)"))
    end
    θl = stl === :optimal ? xl[1:nθ] .* scale : nothing
    θh = sth === :optimal ? xh[1:nθ] .* scale : nothing
    lo = stl === :optimal ? lo : -Inf
    hi = sth === :optimal ? hi : Inf
    return lo, hi, qmin, θl, θh
end
