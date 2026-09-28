# Shared inference helpers: critical values, Wald tests, permutation p-values, RNG.

"""
    critical_value(level=0.95, dof=Inf) -> Float64

Two-sided critical value of a Student-t distribution with `dof` degrees of freedom,
or of the standard normal distribution when `dof` is infinite.

The function returns the quantile ``q`` such that an interval
``\\hat θ ± q ⋅ se(\\hat θ)`` has nominal coverage `level`, i.e.
``q = F^{-1}(1 - (1 - level)/2)`` with ``F`` the distribution function of ``t_{dof}``
or ``N(0, 1)``. It is the single place where DrSnow turns a confidence level into a
multiplier, so that no estimator hard-codes 1.96 and every result honours the
degrees of freedom it reports through `dof_residual`.

The normal reference is justified by a central limit theorem for the estimator. The
t reference with finite `dof` is a small-sample refinement: it is exact only for
normal homoskedastic linear models, and for cluster-robust inference with ``G``
clusters the conventional choice ``dof = G - 1`` is a heuristic that improves, but
does not guarantee, coverage when ``G`` is small (Cameron and Miller 2015).

# Arguments
- `level::Real`: nominal two-sided coverage, strictly between 0 and 1; values
  outside this range throw an `ArgumentError`.
- `dof::Real`: degrees of freedom of the t reference distribution. `Inf` (the
  default) selects the standard normal distribution.

# Returns
- `Float64`: the positive critical value.

# Examples
```julia
using DrSnow
critical_value(0.95)        # 1.959964 (normal)
critical_value(0.95, 9)     # 2.262157 (t with 9 degrees of freedom)
critical_value(0.90, 29)    # 1.699127
```

# References
- Cameron, A. C., & Miller, D. L. (2015). A practitioner's guide to cluster-robust
  inference. *Journal of Human Resources*, 50(2), 317–372.
"""
function critical_value(level::Real=0.95, dof::Real=Inf)
    0 < level < 1 || throw(ArgumentError("level must be in (0, 1), got $level"))
    q = 1 - (1 - level) / 2
    return isfinite(dof) ? quantile(TDist(dof), q) : quantile(Normal(), q)
end

"""
    two_sided_pvalue(t, dof=Inf) -> Float64

Two-sided p-value ``2 ⋅ P(|T| ≥ |t|)`` of a t or z statistic, with ``T`` Student-t
with `dof` degrees of freedom (standard normal when `dof` is infinite).

The tail probability is computed with the complementary distribution function
(`ccdf`) rather than as `1 - cdf`, so that very small p-values are not rounded to
zero. A non-finite statistic (for example a ratio with a zero standard error)
yields `NaN`, never a spurious 0 or 1. The p-value inherits the approximation that
underlies the reference distribution: it is asymptotic for a normal reference and
approximate, not exact, for a t reference outside the classical linear model (see
[`critical_value`](@ref)).

# Arguments
- `t::Real`: the test statistic, typically an estimate divided by its standard
  error.
- `dof::Real`: degrees of freedom of the t reference (`Inf` for the normal).

# Returns
- `Float64` in ``[0, 1]``, or `NaN` when `t` is not finite.

# Examples
```julia
using DrSnow
two_sided_pvalue(1.96)       # 0.0500
two_sided_pvalue(2.5, 12)    # 0.0279
```
"""
function two_sided_pvalue(t::Real, dof::Real=Inf)
    isfinite(t) || return NaN
    d = isfinite(dof) ? TDist(dof) : Normal()
    return min(1.0, 2 * ccdf(d, abs(t)))
end

"""
    WaldTest

Result of a joint Wald test of the linear restrictions ``Rβ = r``, returned by
[`wald_test`](@ref).

The object stores the test in both of its conventional forms: the quadratic form
``W = (R\\hat β - r)'(R \\hat V R')^{-1}(R\\hat β - r)``, which is asymptotically
``χ^2_q`` under the null, and, when a finite denominator degrees of freedom was
supplied, the F form ``W/q``, compared with ``F(q, dof_2)``. `pvalue(w)` is not
defined for this type; read the `pvalue` field directly. Printing shows the
statistic that was used for the p-value.

# Fields
- `statistic::Float64`: the F statistic ``W/q`` when `dof2` is finite, otherwise
  the χ² statistic ``W``.
- `dof1::Int`: number of restrictions ``q`` (rows of ``R``).
- `dof2::Float64`: denominator degrees of freedom of the F form, `Inf` for the χ²
  form.
- `pvalue::Float64`: p-value from the ``F(q, dof_2)`` or ``χ^2_q`` reference.
- `chi2::Float64`: the quadratic form ``W`` itself, whatever the reference used.
"""
struct WaldTest
    statistic::Float64
    dof1::Int
    dof2::Float64
    pvalue::Float64
    chi2::Float64
end

"""
    wald_test(b, V; R=I, r=0, dof=Inf) -> WaldTest

Joint Wald test (Wald 1943) of the linear hypothesis ``H_0: Rβ = r`` from an
estimate `b` of ``β`` and an estimate `V` of its full covariance matrix.

The statistic is the quadratic form
```math
W = (R b - r)' \\left(R V R'\\right)^{-1} (R b - r),
```
computed with a Moore–Penrose pseudo-inverse so that redundant restrictions (a
singular ``R V R'``) do not cause a failure; the number of restrictions ``q`` used
for the reference distribution is the number of rows of ``R`` in any case, so
redundant rows make the test conservative. If ``b`` is asymptotically normal and
``V`` is consistent for its covariance, ``W`` converges to ``χ^2_q`` under the null.
With a finite `dof` the test is reported in F form, ``W/q ∼ F(q, dof)``, the usual
small-sample refinement when ``V`` is a cluster-robust estimator with few clusters
(`dof = G - 1`); this refinement is a heuristic, not an exact result outside the
normal linear model.

DrSnow uses `wald_test` for every joint test (pre-trend tests of event studies,
joint significance of several treatment arms) so that the full covariance matrix,
never only the standard errors, enters the statistic. Testing coefficients one at a
time and combining the verdicts ignores their correlation and does not control the
joint size. A non-rejection is not evidence that the restrictions hold: joint tests
of many leads, in particular, can have low power against smooth violations.

This function has no Monte Carlo size study in the package; its validity rests on
the asymptotic normality of `b` and the consistency of `V` supplied by the caller.

# Arguments
- `b::AbstractVector`: the ``k`` estimated coefficients.
- `V::AbstractMatrix`: their ``k × k`` covariance matrix (e.g. `vcov(model)`), which
  should be the same estimator used for the coefficients' standard errors.

# Keywords
- `R::AbstractMatrix`: ``q × k`` restriction matrix. The default identity matrix
  tests that all ``k`` coefficients are zero.
- `r`: length-``q`` vector of hypothesized values of ``Rβ``; default zero.
- `dof::Real`: denominator degrees of freedom for the F form, e.g. the residual
  degrees of freedom or ``G - 1`` under clustering. `Inf` (default) gives the χ²
  form.

# Returns
- [`WaldTest`](@ref) with the statistic, degrees of freedom, p-value and the χ²
  quadratic form.

# Examples
```julia
using DrSnow
b = [0.40, -0.10, 0.25]
V = [0.04 0.01 0.00; 0.01 0.03 0.00; 0.00 0.00 0.05]
wald_test(b, V)                                   # all three coefficients zero
wald_test(b, V; R=[1.0 -1.0 0.0], dof=29)         # β₁ = β₂, F(1, 29) form
```

# References
- Wald, A. (1943). Tests of statistical hypotheses concerning several parameters
  when the number of observations is large. *Transactions of the American
  Mathematical Society*, 54(3), 426–482.
- Cameron, A. C., & Miller, D. L. (2015). A practitioner's guide to cluster-robust
  inference. *Journal of Human Resources*, 50(2), 317–372.
"""
function wald_test(b::AbstractVector, V::AbstractMatrix;
                   R::AbstractMatrix=Matrix{Float64}(I, length(b), length(b)),
                   r=zeros(size(R, 1)), dof::Real=Inf)
    q = size(R, 1)
    size(R, 2) == length(b) || throw(DimensionMismatch("R must have length(b) columns"))
    d = R * b .- r
    RVR = Symmetric(R * V * R')
    chi2 = float(dot(d, pinv(Matrix(RVR)) * d))
    if isfinite(dof)
        F = chi2 / q
        return WaldTest(F, q, dof, ccdf(FDist(q, dof), F), chi2)
    else
        return WaldTest(chi2, q, Inf, ccdf(Chisq(q), chi2), chi2)
    end
end

function Base.show(io::IO, w::WaldTest)
    if isfinite(w.dof2)
        @printf(io, "Wald test: F(%d, %g) = %.4f, p = %.4g", w.dof1, w.dof2,
                w.statistic, w.pvalue)
    else
        @printf(io, "Wald test: χ²(%d) = %.4f, p = %.4g", w.dof1, w.statistic, w.pvalue)
    end
end

"""
    permutation_pvalue(observed, draws; alternative=:two_sided) -> Float64

Monte Carlo randomization (permutation) p-value
``p = (1 + \\#\\{b : T_b \\text{ at least as extreme as } T_{obs}\\}) / (1 + B)``
from the observed statistic and ``B`` statistics computed on re-randomized
assignments.

Adding one to numerator and denominator counts the observed assignment as a member
of the reference set. Under the null hypothesis, if the ``B`` draws are independent
draws from the assignment mechanism (or, more generally, if the observed and drawn
statistics are exchangeable), this p-value satisfies ``P(p ≤ α) ≤ α`` exactly for
every ``B`` and every ``α``: it is a valid test, not an approximation to the exact
randomization p-value (Phipson and Smyth 2010). The naive proportion
``\\#\\{\\ldots\\}/B`` does not have this property and can equal zero. The price of
Monte Carlo is extra variability: the p-value has Monte Carlo standard error close
to ``\\sqrt{p(1-p)/B}`` around the p-value that complete enumeration would give, so
``B`` should be large enough that decisions at the chosen level are stable (e.g.
``B ≥ 1999`` at the 5% level, more near the threshold).

"At least as extreme" is decided with a relative tolerance of ``10^{-9}``, so that
draws that tie with the observed statistic up to floating-point noise count as ties,
as in R's `ri2`. With `alternative = :two_sided` the absolute values are compared,
which presumes a statistic centred near zero under the null; for statistics that are
not symmetric about zero use a one-sided alternative or an unsigned statistic.

# Arguments
- `observed::Real`: the statistic on the realized assignment.
- `draws::AbstractVector`: the statistic on ``B ≥ 1`` re-randomized assignments,
  drawn from the assignment mechanism under the null.

# Keywords
- `alternative::Symbol`: `:two_sided` (default; ``|T_b| ≥ |T_{obs}|``), `:greater`
  (``T_b ≥ T_{obs}``) or `:less` (``T_b ≤ T_{obs}``).

# Returns
- `Float64` in ``[1/(B+1), 1]``.

# Examples
```julia
using DrSnow, Random, StableRNGs
rng = StableRNG(1)
y = randn(rng, 40) .+ [fill(0.8, 20); zeros(20)]
z = [trues(20); falses(20)]
dim(z) = sum(y[z]) / count(z) - sum(y[.!z]) / count(.!z)
draws = [dim(z[randperm(rng, 40)]) for _ in 1:1999]
permutation_pvalue(dim(z), draws)
```

# References
- Phipson, B., & Smyth, G. K. (2010). Permutation p-values should never be zero:
  Calculating exact p-values when permutations are randomly drawn. *Statistical
  Applications in Genetics and Molecular Biology*, 9(1), Article 39.
- Lehmann, E. L., & Romano, J. P. (2005). *Testing Statistical Hypotheses* (3rd
  ed.). Springer.
- Imbens, G. W., & Rubin, D. B. (2015). *Causal Inference for Statistics, Social,
  and Biomedical Sciences: An Introduction*. Cambridge University Press.
"""
function permutation_pvalue(observed::Real, draws::AbstractVector;
                            alternative::Symbol=:two_sided)
    B = length(draws)
    B > 0 || throw(ArgumentError("need at least one permutation draw"))
    # Relative tolerance (as in R's ri2) so ties survive floating-point noise.
    tol(x) = 1e-9 * max(1.0, abs(x))
    n_extreme = if alternative === :two_sided
        count(d -> abs(d) >= abs(observed) - tol(observed), draws)
    elseif alternative === :greater
        count(d -> d >= observed - tol(observed), draws)
    elseif alternative === :less
        count(d -> d <= observed + tol(observed), draws)
    else
        throw(ArgumentError("alternative must be :two_sided, :greater or :less"))
    end
    return (1 + n_extreme) / (1 + B)
end

"""
    task_seeds(rng, n) -> Vector{UInt64}

Draw `n` independent seeds up front so that per-fold / per-replication work is
reproducible regardless of thread scheduling. Build a fresh `Xoshiro(seed)` per task.
"""
task_seeds(rng::AbstractRNG, n::Integer) = rand(rng, UInt64, n)

# ---------------------------------------------------------------------------
# Influence-function inference (shared by the DiD, ML and other areas)
# ---------------------------------------------------------------------------
#
# Convention: an influence-function matrix Ψ (n × k) is scaled so that
#     θ̂ - θ ≈ (1/n) Σ_i Ψ[i, :],
# i.e. the scaling used by R's DRDID/did packages. Then
#     V̂ = Σ_c S_c S_c' / n²,   S_c = Σ_{i ∈ c} Ψ[i, :],
# with each observation its own cluster when no cluster variable is given (this is
# exactly `sqrt(mean(ψ²)/n)` for a single estimate, as in R's `did`).

"""
    _cluster_sums(Ψ, clusters) -> Matrix

Row sums of `Ψ` within clusters (`clusters === nothing`: returns `Ψ`).
"""
function _cluster_sums(Ψ::AbstractMatrix, clusters)
    clusters === nothing && return Matrix{Float64}(Ψ)
    length(clusters) == size(Ψ, 1) ||
        throw(DimensionMismatch("cluster vector length must equal the number of rows"))
    keys_ = unique(clusters)
    try
        sort!(keys_)
    catch
    end
    idx = Dict(k => i for (i, k) in enumerate(keys_))
    S = zeros(length(keys_), size(Ψ, 2))
    @inbounds for i in axes(Ψ, 1)
        c = idx[clusters[i]]
        for j in axes(Ψ, 2)
            S[c, j] += Ψ[i, j]
        end
    end
    return S
end

"""
    _if_vcov(Ψ, clusters) -> (V, G)

Covariance `Σ_c S_c S_c' / n²` of the estimates whose influence functions are the
columns of `Ψ` (scaled as above), and the number of clusters `G`.
"""
function _if_vcov(Ψ::AbstractMatrix, clusters)
    n = size(Ψ, 1)
    S = _cluster_sums(Ψ, clusters)
    V = (S' * S) ./ n^2
    return Matrix(Symmetric(V)), size(S, 1)
end

"""
    _multiplier_bootstrap(rng, Ψ, clusters, biters) -> (draws, supt)

Multiplier bootstrap with Rademacher weights drawn at the cluster level:
`draws[b, :] = (1/n) Σ_c v_bc S_c` approximates the sampling distribution of
`θ̂ - θ`. `supt[b] = max_k |draws[b, k]| / se_k` over coefficients with positive
standard error (analytic), giving sup-t critical values for uniform bands.
"""
function _multiplier_bootstrap(rng::AbstractRNG, Ψ::AbstractMatrix, clusters,
                                   biters::Integer)
    biters > 0 || throw(ArgumentError("biters must be positive"))
    n = size(Ψ, 1)
    S = _cluster_sums(Ψ, clusters)
    G = size(S, 1)
    se = sqrt.(max.(vec(sum(abs2, S; dims=1)), 0.0)) ./ n
    k = size(S, 2)
    draws = Matrix{Float64}(undef, biters, k)
    # Draw in blocks to bound memory: v is (block × G).
    block = max(1, min(biters, div(2^24, max(G, 1))))
    b0 = 0
    while b0 < biters
        nb = min(block, biters - b0)
        v = Matrix{Float64}(undef, nb, G)
        @inbounds for j in 1:G, i in 1:nb
            v[i, j] = rand(rng, Bool) ? 1.0 : -1.0
        end
        draws[(b0 + 1):(b0 + nb), :] = (v * S) ./ n
        b0 += nb
    end
    pos = findall(>(0), se)
    supt = if isempty(pos)
        Float64[]
    else
        [maximum(abs(draws[b, j]) / se[j] for j in pos) for b in 1:biters]
    end
    return draws, supt
end

"""
    _uniform_critical_value(draws, V, dof, level, rng, ndraws, idx) -> Float64

Critical value for simultaneous (sup-t) bands over the coefficients `idx`: the
`level` quantile of the bootstrap sup-t `draws` when non-empty, otherwise of `ndraws`
simulated draws of `max_k |Z_k|` with `Z` Gaussian (Student-t when `dof` is finite)
with the correlation matrix implied by `V[idx, idx]`. Coefficients with zero variance
are excluded.
"""
function _uniform_critical_value(draws, V, dof, level, rng, ndraws, idx)
    0 < level < 1 || throw(ArgumentError("level must be in (0, 1), got $level"))
    if !isempty(draws)
        return quantile(draws, level)
    end
    Vs = Matrix(V[idx, idx])
    s = sqrt.(max.(diag(Vs), 0.0))
    keep = findall(>(0), s)
    isempty(keep) && return critical_value(level, dof)
    C = Vs[keep, keep] ./ (s[keep] * s[keep]')
    E = eigen(Symmetric(C))
    L = E.vectors * Diagonal(sqrt.(max.(E.values, 0.0)))
    k = length(keep)
    sims = Vector{Float64}(undef, ndraws)
    z = Vector{Float64}(undef, k)
    for b in 1:ndraws
        mul!(z, L, randn(rng, k))
        scale = isfinite(dof) ? sqrt(rand(rng, Chisq(dof)) / dof) : 1.0
        sims[b] = maximum(abs, z) / scale
    end
    return quantile(sims, level)
end
