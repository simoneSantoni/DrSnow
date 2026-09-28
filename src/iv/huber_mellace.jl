# Huber & Mellace (2015) test of instrument validity (binary instrument and
# treatment) based on inequality constraints on the outcome means of always- and
# never-takers.

function _iv_hm_trimmed(y::Vector{Float64}, share::Real, upper::Bool)
    m = length(y)
    k = clamp(round(Int, share * m), 1, m)
    s = sort(y)
    return upper ? mean(view(s, (m - k + 1):m)) : mean(view(s, 1:k))
end

"""The four Huber–Mellace moment inequalities θ ≤ 0 (instrument oriented so that the
first stage is positive)."""
function _iv_hm_theta(y::Vector{Float64}, d::Vector{Float64}, z::Vector{Float64})
    g11 = y[(d .== 1) .& (z .== 1)]
    g10 = y[(d .== 1) .& (z .== 0)]
    g01 = y[(d .== 0) .& (z .== 1)]
    g00 = y[(d .== 0) .& (z .== 0)]
    (isempty(g11) || isempty(g10) || isempty(g01) || isempty(g00)) &&
        throw(ArgumentError("every treatment × instrument cell must be non-empty"))
    p1 = length(g11) / count(==(1), z)
    p0 = length(g10) / count(==(0), z)
    q = p0 / p1                              # always-takers among treated with Z = 1
    r = (1 - p1) / (1 - p0)                  # never-takers among untreated with Z = 0
    (0 < q < 1 && 0 < r < 1) ||
        throw(ArgumentError("the first stage is zero or the compliance shares are " *
                            "degenerate"))
    μa, μn = mean(g10), mean(g01)
    θ = [_iv_hm_trimmed(g11, q, false) - μa, μa - _iv_hm_trimmed(g11, q, true),
         _iv_hm_trimmed(g00, r, false) - μn, μn - _iv_hm_trimmed(g00, r, true)]
    return θ, q, r
end

"""
    huber_mellace_test(data, outcome, treatment, instrument; n_bootstrap=999,
                       rng=Random.default_rng()) -> DiagnosticTest

Test of the joint validity of the LATE assumptions based on inequality constraints on
the mean outcomes of always-takers and never-takers (Huber and Mellace 2015).

With a binary instrument and a binary treatment, random assignment, exclusion and
monotonicity imply a mixture structure in the observed data. With ``Z`` oriented to
raise take-up, the treated units in the ``Z = 0`` arm are all always-takers, whereas
the treated units in the ``Z = 1`` arm are a mixture of always-takers, with share
``q = P(D=1 \\mid Z=0) / P(D=1 \\mid Z=1)``, and compliers. Because the always-takers
in the two arms have the same outcome distribution, their mean, observed in the
``Z = 0`` arm, must lie between the mean of the lowest and the mean of the highest
``q``-fraction of outcomes among treated units with ``Z = 1``. Symmetrically, the
never-takers' mean, observed among untreated units with ``Z = 1``, must lie within the
analogous trimmed means of the untreated with ``Z = 0``, with share
``r = P(D=0 \\mid Z=1) / P(D=0 \\mid Z=0)``. This yields four moment inequalities
``\\theta_j \\le 0``.

The statistic is ``T = \\max_j \\hat\\theta_j / \\hat\\sigma_j``, with ``\\hat\\sigma_j``
the nonparametric-bootstrap standard deviation of ``\\hat\\theta_j``. Its critical
value comes from the bootstrap distribution of ``\\max_j (\\hat\\theta^*_j -
\\hat\\theta_j) / \\hat\\sigma_j`` over the inequalities that are not far from binding,
``\\hat\\theta_j/\\hat\\sigma_j \\ge -\\sqrt{\\ln n}``, a generalized moment selection
step in the spirit of Andrews and Soares (2010); if no inequality is selected the
p-value is 1. This implementation differs from the smoothed-indicator procedure used
by Huber and Mellace (2015) but tests the same inequalities. Bootstrap samples in
which a treatment-by-instrument cell is empty are redrawn.

A rejection is evidence against at least one of random assignment, exclusion and
monotonicity. Non-rejection does not establish them: the test has power only against
violations that push the always- or never-takers' means outside these bounds, and it
uses less information than Kitagawa's (2015) distributional test in
[`instrument_validity_test`](@ref), which is generally more powerful against
violations that change the shape of the outcome distribution. Observations are
treated as independent.

# Arguments
- `data::AbstractDataFrame`: the data; incomplete rows are dropped.
- `outcome::Symbol`: the outcome ``Y``.
- `treatment::Symbol`: binary (0/1) treatment.
- `instrument::Symbol`: binary (0/1) instrument; every treatment-by-instrument cell
  must be non-empty and the first stage non-zero.

# Keywords
- `n_bootstrap::Int`: bootstrap replications (default 999, at least 99).
- `rng::AbstractRNG`: random-number generator (default `Random.default_rng()`; pass
  e.g. `StableRNG(1)` for reproducibility).

# Returns
- A [`DiagnosticTest`](@ref); `details` has `theta` (the four estimated inequality
  moments), `se`, the mixture shares `q` and `r`, and `selected` (the inequalities
  retained by moment selection).

# Examples
```julia
using DrSnow, DataFrames, StableRNGs
rng = StableRNG(10)
n = 3_000
z = rand(rng, 0:1, n)
u = rand(rng, n)
d = ifelse.(u .< 0.2, 1, ifelse.(u .< 0.7, z, 0))
y = randn(rng, n) .+ d .+ 0.5 .* (u .< 0.2)
df = DataFrame(y=y, d=d, z=z)
huber_mellace_test(df, :y, :d, :z; n_bootstrap=499, rng=StableRNG(1))
```

# References
- Huber, M., & Mellace, G. (2015). Testing instrument validity for LATE
  identification based on inequality moment constraints. *Review of Economics and
  Statistics*, 97(2), 398–411.
- Andrews, D. W. K., & Soares, G. (2010). Inference for parameters defined by moment
  inequalities using generalized moment selection. *Econometrica*, 78(1), 119–157.
- Kitagawa, T. (2015). A test for instrument validity. *Econometrica*, 83(5),
  2043–2063.
- Imbens, G. W., & Angrist, J. D. (1994). Identification and estimation of local
  average treatment effects. *Econometrica*, 62(2), 467–475.
"""
function huber_mellace_test(data::AbstractDataFrame, outcome::Symbol, treatment::Symbol,
                            instrument::Symbol; n_bootstrap::Int=999,
                            rng::AbstractRNG=Random.default_rng())
    n_bootstrap >= 99 || throw(ArgumentError("n_bootstrap must be at least 99"))
    prep = _iv_binary_prep(data, treatment, instrument, [outcome], Symbol[], nothing,
                           nothing, "huber_mellace_test")
    y = Float64.(prep.sub[!, outcome])
    d, z = prep.d, prep.z
    mean(d[z .== 1]) < mean(d[z .== 0]) && (z = 1 .- z)       # orient the instrument
    θ, q, r = _iv_hm_theta(y, d, z)
    n = length(y)
    seeds = task_seeds(rng, n_bootstrap)
    B = zeros(n_bootstrap, 4)
    b = 1
    tries = 0
    while b <= n_bootstrap
        tries += 1
        tries > 10 * n_bootstrap && error("huber_mellace_test: too many degenerate " *
                                          "bootstrap samples")
        brng = Xoshiro(seeds[b] + UInt64(tries))
        ib = rand(brng, 1:n, n)
        θb = try
            first(_iv_hm_theta(y[ib], d[ib], z[ib]))
        catch err
            err isa ArgumentError || rethrow()
            continue
        end
        B[b, :] = θb
        b += 1
    end
    σ = vec(std(B; dims=1))
    all(>(0), σ) || throw(ArgumentError("huber_mellace_test: a moment has zero " *
                                        "bootstrap variance"))
    t = θ ./ σ
    T = maximum(t)
    sel = t .>= -sqrt(log(n))
    # no inequality close to binding: the recentered bootstrap statistic is −∞
    p = if any(sel)
        Tb = [maximum(((B[j, :] .- θ) ./ σ)[sel]) for j in 1:n_bootstrap]
        (1 + count(>=(T), Tb)) / (1 + n_bootstrap)
    else
        1.0
    end
    return DiagnosticTest("Huber–Mellace test of instrument validity",
                          "always- and never-takers' mean outcomes lie within the " *
                          "bounds implied by independence, exclusion and monotonicity",
                          T, p;
                          method="max studentized violation of 4 moment inequalities; " *
                                 "bootstrap with moment selection ($n_bootstrap draws)",
                          note="Non-rejection does not establish instrument validity.",
                          details=(theta=θ, se=σ, q=q, r=r, selected=sel))
end
