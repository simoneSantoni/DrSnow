# Causal interpretation of 2SLS with several discrete instruments (Mogstad,
# Torgovitsky & Walters 2021, AER).
#
# With instrument vector Z taking K values (cells), exogeneity and exclusion, and a
# first stage ψ(Z) (the saturated propensity p(Z), or the linear projection of D on
# the instruments), the 2SLS estimand is
#     β = Cov(Y, ψ(Z)) / Cov(D, ψ(Z)) = Σ_g ω_g Δ_g,   ω_g = π_g c_g / Σ_h π_h c_h,
# where g runs over response groups (potential-treatment vectors (D(z))_z), π_g their
# shares, Δ_g their average effects and c_g = Cov(D_g(Z), ψ(Z)) (computed over the
# distribution of Z). Groups with constant D have c_g = 0. A group receives a negative
# weight iff it is present and c_g < 0. The admissible groups depend on the
# monotonicity assumption; the shares are not point identified, but they must
# reproduce p(z), so linear programming bounds each share and each weight.

const _IV_MIW_ASSUMPTIONS = (:pm, :vm, :iam)

"""
    MultipleIVWeights

Decomposition of a 2SLS estimand with several discrete instruments into weights on
response groups (see [`multiple_iv_weights`](@ref)).

The object describes, for a binary treatment and a vector of discrete instruments, the
instrument cells and their propensity scores, the response groups (vectors of
potential treatments across the cells) that are admissible under the chosen
monotonicity assumption, the identified sign of each group's 2SLS weight, linear
programming bounds on each group's share and weight, and a test of the null
hypothesis that no group that can be present receives a negative weight. The group
shares are not point identified, so the weights are reported as bounds.

# Fields
- `treatment::String`, `instruments::Vector{String}`: variable names.
- `assumption::Symbol`: `:pm` (partial monotonicity), `:vm` (vector monotonicity) or
  `:iam` (Imbens–Angrist monotonicity).
- `first_stage::Symbol`: `:saturated` (2SLS with a fully interacted first stage, that
  is, IV with the estimated propensity score as instrument) or `:linear` (instruments
  entered additively).
- `cells::DataFrame`: one row per instrument value, with the instrument values, `n`,
  `share` (``P(Z = z)``), `p` (``P(D = 1 \\mid Z = z)``), `psi` (the first-stage fitted
  value ``\\psi(z)``) and `rank` (position in increasing order of `p`).
- `groups::DataFrame`: one row per admissible response group that changes treatment
  status, with `group` (label), `pattern` (``D_g(z)`` over the cells in the order of
  `cells`), `c` (``c_g = \\operatorname{Cov}(D_g(Z), \\psi(Z))``), `se_c` (bootstrap
  standard error), `weight_sign` (sign of the weight when the group is present),
  `max_share` and `min_share` (linear-programming bounds on the group share given the
  observed propensities), `weight_lower` and `weight_upper` (bounds on the 2SLS
  weight), and `complier_at` / `defier_at` (the sets ``\\mathcal C_g`` and
  ``\\mathcal D_g`` of Mogstad, Torgovitsky and Walters 2021: the ranks ``k`` at which
  the group switches into, respectively out of, treatment between ``z^{k-1}`` and
  ``z^k``).
- `cov_d_psi::Float64`: ``\\operatorname{Cov}(D, \\psi(Z))``, the common denominator of
  the weights.
- `feasible::Bool`: whether some group shares consistent with the assumption
  reproduce the observed propensities (a linear program); when `false` the share and
  weight bounds are `NaN`.
- `negative_weight_possible::Bool`: some admissible group with ``c_g < 0`` can have a
  positive share.
- `negative_weight_bounds::Tuple{Float64,Float64}`: range of the total weight on
  negatively weighted groups over all group shares consistent with the data.
- `rectangular::Bool`: whether the support of ``Z`` is the product of the marginal
  supports.
- `tsls::Union{Nothing,Float64}`: the 2SLS estimate, when `outcome` was given.
- `test::DiagnosticTest`: the test of ``H_0: c_g \\ge 0`` for every group that can be
  present.
- `n::Int`: number of observations.

# References
- Mogstad, M., Torgovitsky, A., & Walters, C. R. (2021). The causal interpretation of
  two-stage least squares with multiple instrumental variables. *American Economic
  Review*, 111(11), 3663–3698.
"""
struct MultipleIVWeights
    treatment::String
    instruments::Vector{String}
    assumption::Symbol
    first_stage::Symbol
    cells::DataFrame
    groups::DataFrame
    cov_d_psi::Float64
    feasible::Bool
    negative_weight_possible::Bool
    negative_weight_bounds::Tuple{Float64,Float64}
    rectangular::Bool
    tsls::Union{Nothing,Float64}
    test::DiagnosticTest
    n::Int
end

StatsAPI.nobs(r::MultipleIVWeights) = r.n

function Base.show(io::IO, ::MIME"text/plain", r::MultipleIVWeights)
    aname = Dict(:pm => "partial monotonicity", :vm => "vector monotonicity",
                 :iam => "Imbens–Angrist monotonicity")[r.assumption]
    println(io, "2SLS weights on response groups (Mogstad, Torgovitsky & Walters 2021)")
    println(io, "Treatment ", r.treatment, "; instruments ", join(r.instruments, ", "),
            "; ", r.first_stage, " first stage; assumption: ", aname)
    println(io, "Observations: ", r.n, "; instrument cells: ", nrow(r.cells),
            r.rectangular ? "" : " (support not rectangular)")
    r.tsls === nothing || @printf(io, "2SLS estimate: %.4g\n", r.tsls)
    println(io, "Cells (ordered by propensity):")
    for row in eachrow(sort(r.cells, :rank))
        vals = join([string(row[Symbol(z)]) for z in r.instruments], ", ")
        @printf(io, "  z = (%s): n = %d, P(D=1|z) = %.4f, first stage ψ = %.4f\n", vals,
                row.n, row.p, row.psi)
    end
    println(io, "Groups that change treatment status:")
    for row in eachrow(r.groups)
        @printf(io, "  %-28s D = %s  c = %+.4g (se %.3g)", row.group, row.pattern,
                row.c, row.se_c)
        @printf(io, "  share ≤ %.3g  weight ∈ [%.3g, %.3g]\n", row.max_share,
                row.weight_lower, row.weight_upper)
    end
    if !r.feasible
        println(io, "The observed propensities are not compatible with ", aname,
                " (no group shares reproduce them).")
    elseif r.negative_weight_possible
        lo, hi = r.negative_weight_bounds
        @printf(io, "Negative weights are possible: total negative weight in [%.3g, %.3g]",
                lo, hi)
        println(io, " over the group shares consistent with the data.")
    else
        println(io, "No admissible group can receive a negative weight: under ", aname,
                ", exogeneity and exclusion, 2SLS is a non-negatively weighted average ",
                "of group effects.")
    end
    @printf(io, "%s: stat = %.4g, p = %.4g\n", r.test.name, r.test.statistic,
            r.test.pvalue)
end

Base.show(io::IO, r::MultipleIVWeights) =
    print(io, "MultipleIVWeights(", nrow(r.groups), " groups, negative weight possible: ",
          r.negative_weight_possible, ")")

# ---------------------------------------------------------------------------
# Cells, first stage and admissible response groups
# ---------------------------------------------------------------------------

function _iv_miw_first_stage(V::Matrix{Float64}, share::Vector{Float64}, p::Vector{Float64},
                             kind::Symbol)
    kind === :saturated && return copy(p)
    X = hcat(ones(size(V, 1)), V)
    sw = sqrt.(share)
    b = qr(X .* sw, ColumnNorm()) \ (p .* sw)
    return X * b
end

"""Constraints `D[i] ≥ D[j]` (pairs) implied by the monotonicity assumption."""
function _iv_miw_constraints(V::Matrix{Float64}, p::Vector{Float64}, assumption::Symbol,
                             dirs)
    K, L = size(V)
    cons = Tuple{Int,Int}[]
    tol = 1e-12
    if assumption === :iam
        for i in 1:K, j in 1:K
            i != j && p[i] >= p[j] - tol && push!(cons, (i, j))
        end
        return cons
    end
    for i in 1:K, j in (i + 1):K
        diff = findall(ℓ -> V[i, ℓ] != V[j, ℓ], 1:L)
        length(diff) == 1 || continue
        ℓ = diff[1]
        if assumption === :pm
            if p[i] > p[j] + tol
                push!(cons, (i, j))
            elseif p[j] > p[i] + tol
                push!(cons, (j, i))
            else
                push!(cons, (i, j)); push!(cons, (j, i))
            end
        else  # :vm, direction per instrument
            s = dirs[ℓ] * (V[i, ℓ] - V[j, ℓ])
            s > 0 ? push!(cons, (i, j)) : push!(cons, (j, i))
        end
    end
    return cons
end

"""All 0/1 vectors over the cells satisfying the constraints (depth-first search)."""
function _iv_miw_enumerate(K::Int, cons::Vector{Tuple{Int,Int}}, max_groups::Int, ctx)
    ge = [Int[] for _ in 1:K]      # ge[i]: j < i with D[i] ≥ D[j]
    le = [Int[] for _ in 1:K]      # le[i]: j < i with D[i] ≤ D[j]
    for (i, j) in cons
        if j < i
            push!(ge[i], j)
        elseif i < j
            push!(le[j], i)
        end
    end
    out = BitVector[]
    cur = falses(K)
    function rec(k)
        if k > K
            push!(out, copy(cur))
            length(out) > max_groups &&
                throw(ArgumentError("$ctx: more than $max_groups admissible response " *
                                    "groups; use fewer instrument values or a stronger " *
                                    "assumption"))
            return
        end
        for v in (false, true)
            ok = all(j -> v >= cur[j], ge[k]) && all(j -> v <= cur[j], le[k])
            ok || continue
            cur[k] = v
            rec(k + 1)
        end
        cur[k] = false
    end
    rec(1)
    return out
end

function _iv_miw_label(D::BitVector, V::Matrix{Float64}, names::Vector{String}, gidx)
    all(D) && return "always-taker"
    any(D) || return "never-taker"
    K, L = size(V)
    for ℓ in 1:L, a in unique(V[:, ℓ])
        D == BitVector(V[:, ℓ] .== a) && return "$(names[ℓ]) complier (=$(_iv_fmtnum(a)))"
    end
    if L == 2 && all(ℓ -> length(unique(V[:, ℓ])) == 2, 1:2)
        for a in unique(V[:, 1]), b in unique(V[:, 2])
            e1, e2 = V[:, 1] .== a, V[:, 2] .== b
            D == BitVector(e1 .| e2) && return "eager complier"
            D == BitVector(e1 .& e2) && return "reluctant complier"
        end
    end
    return "group $gidx"
end

_iv_fmtnum(a) = isinteger(a) ? string(Int(a)) : @sprintf("%.4g", a)

"""c_g for every group given cell shares and first-stage values."""
function _iv_miw_c(groups::Vector{BitVector}, share, psi)
    ψbar = dot(share, psi)
    wz = share .* (psi .- ψbar)
    return [dot(wz, D) for D in groups]
end

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

"""
    multiple_iv_weights(data, treatment, instruments; outcome=nothing,
                        first_stage=:saturated, assumption=:pm, directions=nothing,
                        n_boot=999, max_cells=64, max_groups=200_000,
                        rng=Random.default_rng()) -> MultipleIVWeights

Decompose the 2SLS estimand with several discrete instruments into weights on
response groups, and report which groups can receive a negative weight (Mogstad,
Torgovitsky and Walters 2021).

With a single binary instrument, the monotonicity condition of Imbens and Angrist
(1994) makes 2SLS a LATE. With several instruments, the analogous condition (IA
monotonicity: for any two instrument values, everybody's treatment moves in the same
direction) is much stronger than it looks: Mogstad, Torgovitsky and Walters (2021,
MTW) show that it requires choice behaviour to be effectively homogeneous across
individuals, since it rules out, for example, one person responding only to the first
instrument and another only to the second (Heckman and Vytlacil 2005 had observed that
IA monotonicity is a uniformity condition across individuals rather than monotonicity
in the instrument). MTW therefore study the weaker *partial
monotonicity* (PM): changing one instrument while holding the others fixed moves
everybody's treatment weakly in the same direction. Under PM, exogeneity and
exclusion, the 2SLS estimand is a weighted average of the average treatment effects
of the response groups that change treatment status, with weights that sum to one but
need not be non-negative; MTW characterize the signs of the weights in terms of
observable quantities, so that whether 2SLS is a positively weighted average of
group effects can be checked empirically.

Let ``Z`` take ``K`` values (cells) and let every individual have a response vector
``(D(z))_z``, which defines the response group ``g``. With first stage ``\\psi(Z)`` (the
saturated propensity score ``p(Z)``, or the linear projection of ``D`` on the
instruments), the 2SLS estimand is

```math
\\beta_{2SLS} = \\frac{\\operatorname{Cov}(Y, \\psi(Z))}{\\operatorname{Cov}(D, \\psi(Z))}
             = \\sum_g \\omega_g \\Delta_g , \\qquad
\\omega_g = \\frac{\\pi_g c_g}{\\operatorname{Cov}(D, \\psi(Z))} , \\quad
c_g = \\operatorname{Cov}(D_g(Z), \\psi(Z)) ,
```

where ``\\pi_g`` is the population share and ``\\Delta_g`` the average treatment effect
of group ``g``, and the covariance defining ``c_g`` is taken over the distribution of
``Z``. A group therefore receives a negative weight exactly when it is present and
``c_g < 0``; ``c_g`` is identified, ``\\pi_g`` is not. For the saturated first stage,
and ties in ``p`` aside,

```math
c_g = \\sum_{k=2}^{K} \\big(1[k \\in \\mathcal C_g] - 1[k \\in \\mathcal D_g]\\big)
      \\operatorname{Cov}\\big(D, 1[p(Z) \\ge p(z^k)]\\big) ,
```

where the cells are ordered by propensity and ``\\mathcal C_g`` (``\\mathcal D_g``) is
the set of ranks ``k`` at which group ``g`` switches into (out of) treatment between
``z^{k-1}`` and ``z^k``; the sign of ``c_g`` is therefore the sign given in MTW's
Proposition 7 (stated there for rectangular support). The admissible groups depend on
`assumption`:

- `:pm`, partial monotonicity (MTW): for any two cells that differ in one instrument
  only, everybody's treatment moves weakly in the same direction; the direction of
  each such comparison is identified and is read off the observed propensities.
- `:vm`, vector monotonicity (Goff 2024): treatment is monotone in each instrument in
  a direction common to all individuals and all values of the other instruments
  (`directions`, a vector of ±1; by default the sign of each instrument's coefficient
  in the linear projection of ``D`` on the instruments). With all directions positive
  this is MTW's "actual monotonicity"; it implies PM.
- `:iam`, IA monotonicity over all instrument values: the groups are thresholds in the
  propensity score, and with the saturated first stage all weights are non-negative
  (Imbens and Angrist 1994). With one instrument PM and IAM coincide.

The group shares must reproduce the observed propensities ``p(z)``; linear programs
give the largest and smallest share of each group, bounds on each weight and bounds
on the total negative weight. `test` is a least-favorable max-t bootstrap test
(rows resampled, `n_boot` draws) of ``H_0: c_g \\ge 0`` for every group that can be
present at the observed propensities, that is, of the null that 2SLS is a
non-negatively weighted average of group effects under the assumption; group
admissibility is determined at the point estimates and treated as fixed. This is
simpler and more conservative than the Romano–Shaikh–Wolf procedure that MTW favour
in their online appendix. Rejection indicates that a group that may be present would
receive a negative weight. Non-rejection does not establish positive weights, and
neither the monotonicity assumption nor exogeneity and exclusion are tested.

For two binary instruments under PM the groups are always- and never-takers, ``Z_1``
and ``Z_2`` compliers, and eager and reluctant compliers (MTW, Table 2). MTW's
Proposition 5 then shows that the weights of eager and reluctant compliers are always
non-negative, that the weight of the more common single-instrument complier group is
non-negative, and that the less common group, if present, has a negative weight if and
only if the regression of ``D`` on its own instrument alone has a negative slope;
Proposition 6 shows that this requires negatively correlated instruments, as with
mutually exclusive encouragement arms. The decomposition reproduces these results.
Covariates are not supported: condition by splitting the sample, or saturate discrete
covariates into the instrument cells.

# Arguments
- `data::AbstractDataFrame`: the data; rows with missing values in used columns are
  dropped.
- `treatment::Symbol`: the binary (0/1) treatment ``D``.
- `instruments`: the discrete numeric instrument columns (a `Symbol` or a vector).

# Keywords
- `outcome`: an optional outcome column (default `nothing`); when given, the 2SLS
  estimate with first stage ``\\psi`` is reported.
- `first_stage::Symbol`: `:saturated` (default; the fully interacted first stage of
  MTW's analysis) or `:linear` (2SLS with the instruments entered additively).
- `assumption::Symbol`: `:pm` (default), `:vm` or `:iam`, as described above.
- `directions`: for `:vm`, a vector of ±1, one per instrument (default `nothing`,
  estimated from the linear projection).
- `n_boot::Integer`: bootstrap draws for the test and for the standard errors of
  ``c_g`` (default 999, at least 19).
- `max_cells::Integer`, `max_groups::Integer`: guards against combinatorial explosion
  of the number of cells (default 64) and admissible groups (default 200 000).
- `rng::AbstractRNG`: random-number generator for the bootstrap.

# Returns
- A [`MultipleIVWeights`](@ref); `w.groups` lists ``c_g`` and the share and weight
  bounds per group, `w.negative_weight_possible` and `w.negative_weight_bounds`
  summarize the scope for negative weights, and `w.test` is the bootstrap test.

# Examples
```julia
using DrSnow, DataFrames, StableRNGs
rng = StableRNG(1)
n = 4_000
cell = rand(rng, n)                                  # P(z) = 0.1, 0.4, 0.4, 0.1
z1 = Float64.((0.1 .<= cell .< 0.5) .| (cell .>= 0.9))
z2 = Float64.(cell .>= 0.5)                          # cov(z1, z2) < 0
g = rand(rng, n)   # 15% always, 25% never, 35% Z1-, 10% Z2-, 10% eager, 5% reluctant
d = Float64.((g .< 0.15) .|
             ((0.40 .<= g .< 0.75) .& (z1 .== 1)) .|
             ((0.75 .<= g .< 0.85) .& (z2 .== 1)) .|
             ((0.85 .<= g .< 0.95) .& ((z1 .+ z2) .> 0)) .|
             ((g .>= 0.95) .& (z1 .+ z2 .== 2)))
y = d .* (1.0 .+ 2.0 .* (0.75 .<= g .< 0.85)) .+ randn(rng, n)
df = DataFrame(y=y, d=d, z1=z1, z2=z2)
w = multiple_iv_weights(df, :d, [:z1, :z2]; outcome=:y, n_boot=199, rng=StableRNG(2))
w.groups            # the Z2 compliers have c_g < 0: a negative weight if present
w.test              # H₀: no group that can be present has a negative weight
```

# References
- Imbens, G. W., & Angrist, J. D. (1994). Identification and estimation of local
  average treatment effects. *Econometrica*, 62(2), 467–475.
- Mogstad, M., Torgovitsky, A., & Walters, C. R. (2021). The causal interpretation of
  two-stage least squares with multiple instrumental variables. *American Economic
  Review*, 111(11), 3663–3698.
- Goff, L. (2024). A vector monotonicity assumption for multiple instruments.
  *Journal of Econometrics*, 241(1), 105735.
- Heckman, J. J., & Vytlacil, E. (2005). Structural equations, treatment effects, and
  econometric policy evaluation. *Econometrica*, 73(3), 669–738.
"""
function multiple_iv_weights(data::AbstractDataFrame, treatment::Symbol, instruments;
                             outcome=nothing, first_stage::Symbol=:saturated,
                             assumption::Symbol=:pm, directions=nothing,
                             n_boot::Integer=999, max_cells::Integer=64,
                             max_groups::Integer=200_000,
                             rng::AbstractRNG=Random.default_rng())
    ctx = "multiple_iv_weights"
    inst = _as_symbols(instruments)
    isempty(inst) && throw(ArgumentError("$ctx: at least one instrument is required"))
    first_stage in (:saturated, :linear) ||
        throw(ArgumentError("$ctx: first_stage must be :saturated or :linear"))
    assumption in _IV_MIW_ASSUMPTIONS ||
        throw(ArgumentError("$ctx: assumption must be :pm, :vm or :iam"))
    n_boot >= 19 || throw(ArgumentError("$ctx: n_boot must be at least 19"))
    cols = outcome === nothing ? vcat(treatment, inst) : vcat(outcome, treatment, inst)
    require_columns(data, cols; context=ctx)
    _iv_check_numeric(data, cols, ctx)
    keep = BitVector([all(c -> !ismissing(data[i, c]), cols) for i in 1:nrow(data)])
    sub = disallowmissing(data[keep, cols])
    n = nrow(sub)
    d = Float64.(sub[!, treatment])
    all(x -> x == 0 || x == 1, d) ||
        throw(ArgumentError("$ctx: treatment must be binary (0/1)"))
    Zr = Matrix{Float64}(hcat([Float64.(sub[!, z]) for z in inst]...))
    # cells in lexicographic order of instrument values (row-order invariant)
    keys_ = sort!(unique([Zr[i, :] for i in 1:n]))
    K = length(keys_)
    K <= max_cells || throw(ArgumentError("$ctx: $K instrument cells exceed max_cells " *
                                          "= $max_cells; the instruments must be " *
                                          "discrete"))
    K >= 2 || throw(ArgumentError("$ctx: the instruments take a single value"))
    cid = Dict(k => i for (i, k) in enumerate(keys_))
    cell = [cid[Zr[i, :]] for i in 1:n]
    V = Matrix{Float64}(reduce(vcat, [k' for k in keys_]))
    cnt = zeros(Int, K)
    sd = zeros(K)
    for i in 1:n
        cnt[cell[i]] += 1
        sd[cell[i]] += d[i]
    end
    share = cnt ./ n
    p = sd ./ cnt
    psi = _iv_miw_first_stage(V, share, p, first_stage)
    cov_dpsi = dot(share, (psi .- dot(share, psi)) .* p)
    cov_dpsi > 1e-12 ||
        throw(ArgumentError("$ctx: the first stage is zero (Cov(D, ψ(Z)) = 0); 2SLS is " *
                            "not identified"))
    L = length(inst)
    rect = K == prod(length(unique(V[:, ℓ])) for ℓ in 1:L)
    dirs = if assumption === :vm
        if directions === nothing
            X = hcat(ones(K), V)
            sw = sqrt.(share)
            b = (qr(X .* sw, ColumnNorm()) \ (p .* sw))[2:end]
            [bj >= 0 ? 1 : -1 for bj in b]
        else
            length(directions) == L && all(x -> x == 1 || x == -1, directions) ||
                throw(ArgumentError("$ctx: directions must be a vector of ±1, one per " *
                                    "instrument"))
            Int.(collect(directions))
        end
    else
        nothing
    end
    cons = _iv_miw_constraints(V, p, assumption, dirs)
    allg = _iv_miw_enumerate(K, cons, max_groups, ctx)
    # LP over all admissible groups (including always/never-takers)
    A = vcat(Matrix{Float64}(reduce(hcat, [Float64.(g) for g in allg])),
             ones(1, length(allg)))
    b = vcat(p, 1.0)
    cvec = _iv_miw_c(allg, share, psi)
    ng = length(allg)
    lpmax(obj) = begin
        res = _lp_simplex(obj, A, b)
        res.status === :optimal ? (true, res.objective) :
        (res.status === :infeasible ? (false, NaN) :
         error("internal: LP status $(res.status)"))
    end
    feasible, _ = lpmax(zeros(ng))
    maxs = fill(NaN, ng)
    mins = fill(NaN, ng)
    negb = (NaN, NaN)
    if feasible
        for g in 1:ng
            e = zeros(ng); e[g] = 1.0
            maxs[g] = max(lpmax(e)[2], 0.0)
            mins[g] = max(-lpmax(-e)[2], 0.0)
        end
        negc = [cvec[g] < -1e-12 ? cvec[g] : 0.0 for g in 1:ng]
        lo = -lpmax(-negc)[2] / cov_dpsi      # most negative total negative weight
        hi = lpmax(negc)[2] / cov_dpsi
        negb = (min(lo, hi) + 0.0, max(lo, hi) + 0.0)
    end
    # groups that change status
    movers = [g for g in 1:ng if any(allg[g]) && !all(allg[g])]
    # bootstrap of c_g (rows resampled)
    boot = zeros(n_boot, ng)
    for bi in 1:n_boot
        idx = rand(rng, 1:n, n)
        cb = zeros(Int, K)
        sb = zeros(K)
        for i in idx
            cb[cell[i]] += 1
            sb[cell[i]] += d[i]
        end
        shb = cb ./ n
        pb = [cb[k] > 0 ? sb[k] / cb[k] : 0.0 for k in 1:K]
        # empty cells have zero share, so they do not enter the linear projection
        psib = _iv_miw_first_stage(V, shb, pb, first_stage)
        boot[bi, :] = _iv_miw_c(allg, shb, psib)
    end
    se = vec(std(boot; dims=1))
    # ranks by propensity and MTW complier / defier sets
    ordr = sortperm(p)
    rank = zeros(Int, K)
    rank[ordr] = 1:K
    labels = String[]
    comp = Vector{Int}[]
    defi = Vector{Int}[]
    for (j, g) in enumerate(movers)
        D = allg[g]
        push!(labels, _iv_miw_label(D, V, string.(inst), j))
        push!(comp, [k for k in 2:K if D[ordr[k]] && !D[ordr[k - 1]]])
        push!(defi, [k for k in 2:K if !D[ordr[k]] && D[ordr[k - 1]]])
    end
    wsign = [cvec[g] > 1e-12 ? 1 : (cvec[g] < -1e-12 ? -1 : 0) for g in movers]
    wl = [feasible ? min(cvec[g] * mins[g], cvec[g] * maxs[g]) / cov_dpsi + 0.0 : NaN
          for g in movers]
    wu = [feasible ? max(cvec[g] * mins[g], cvec[g] * maxs[g]) / cov_dpsi + 0.0 : NaN
          for g in movers]
    groups = DataFrame(group=labels,
                       pattern=[join(Int.(allg[g])) for g in movers],
                       c=cvec[movers], se_c=se[movers], weight_sign=wsign,
                       max_share=maxs[movers], min_share=mins[movers],
                       weight_lower=wl, weight_upper=wu, complier_at=comp,
                       defier_at=defi)
    cells = DataFrame()
    for (ℓ, z) in enumerate(inst)
        cells[!, z] = V[:, ℓ]
    end
    cells.n = cnt
    cells.share = share
    cells.p = p
    cells.psi = psi
    cells.rank = rank
    negpossible = feasible && any(g -> cvec[g] < -1e-12 && maxs[g] > 1e-10, movers)
    # test H0: c_g ≥ 0 for groups that can be present
    present = [g for g in movers if !feasible || maxs[g] > 1e-10]
    present = [g for g in present if se[g] > 0]
    test = if isempty(present)
        DiagnosticTest("MTW test of non-negative 2SLS weights",
                       "every response group that can be present has c_g ≥ 0", 0.0, 1.0;
                       method="no admissible changing group with sampling variation",
                       note="Nothing to test: every admissible group has a " *
                            "deterministic non-negative weight.")
    else
        tstat = maximum(-cvec[g] / se[g] for g in present)
        bmax = [maximum(-(boot[bi, g] - cvec[g]) / se[g] for g in present)
                for bi in 1:n_boot]
        pv = (1 + count(>=(tstat), bmax)) / (1 + n_boot)
        DiagnosticTest("MTW test of non-negative 2SLS weights",
                       "every response group that can be present under the assumption " *
                       "has c_g = Cov(D_g(Z), ψ(Z)) ≥ 0 (no negative 2SLS weights)",
                       tstat, pv; method="least-favorable max-t bootstrap ($n_boot " *
                                         "draws, rows resampled), $(length(present)) " *
                                         "groups",
                       note="Group admissibility is taken from the point estimates of " *
                            "the propensities. Non-rejection does not show that the " *
                            "weights are positive; the monotonicity assumption, " *
                            "exogeneity and exclusion are maintained, not tested.",
                       details=(groups=groups.group[[findfirst(==(g), movers)
                                                     for g in present]],
                                t=[-cvec[g] / se[g] for g in present]))
    end
    tsls = nothing
    if outcome !== nothing
        y = Float64.(sub[!, outcome])
        ψi = psi[cell]
        tsls = cov(y, ψi) / cov(d, ψi)
    end
    return MultipleIVWeights(string(treatment), string.(inst), assumption, first_stage,
                             cells, groups, cov_dpsi, feasible, negpossible, negb, rect,
                             tsls, test, n)
end

"""
    _iv_miw_true_weights(V, share, patterns, π; first_stage=:saturated) -> Vector

Internal helper for simulations: population 2SLS weights `ω_g = π_g c_g / Σ_h π_h c_h`
for known response groups (0/1 patterns over the cells `V`) with known shares `π`.
"""
function _iv_miw_true_weights(V::Matrix{Float64}, share::Vector{Float64},
                              patterns::Vector{BitVector}, π::Vector{Float64};
                              first_stage::Symbol=:saturated)
    p = reduce(+, [π[g] .* Float64.(patterns[g]) for g in eachindex(patterns)])
    psi = _iv_miw_first_stage(V, share, p, first_stage)
    c = _iv_miw_c(patterns, share, psi)
    return π .* c ./ sum(π .* c)
end
