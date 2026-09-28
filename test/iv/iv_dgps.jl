# Data-generating processes shared by the IV tests.

"""
Linear IV DGP with endogeneity: `D = Z π + v`, `Y = β D + x + u`,
`corr(u, v) = ρ`. Options: heteroskedastic errors (scale depends on Z₁), cluster
random effects in both equations (`G` clusters), a direct effect `γ` of Z₁ on Y.
"""
function iv_linear_dgp(rng; n=500, k=1, pi=0.5, beta=1.0, rho=0.6, hetero=false,
                       G=0, gamma=0.0)
    Z = randn(rng, n, k)
    x = randn(rng, n)
    v = randn(rng, n)
    u = rho .* v .+ sqrt(1 - rho^2) .* randn(rng, n)
    cl = G > 0 ? rand(rng, 1:G, n) : ones(Int, n)
    if G > 0
        a = randn(rng, G)
        b = randn(rng, G)
        u .+= a[cl]
        v .+= 0.5 .* a[cl] .+ b[cl]
        Z[:, 1] .+= randn(rng, G)[cl]
    end
    if hetero
        u .*= (0.5 .+ abs.(Z[:, 1]))
        v .*= (0.5 .+ 0.5 .* abs.(Z[:, 1]))
    end
    πv = pi isa Number ? fill(float(pi), k) : float.(pi)
    d = Z * πv .+ 0.3 .* x .+ v
    y = beta .* d .+ x .+ gamma .* Z[:, 1] .+ u
    df = DataFrame(y=y, d=d, x=x, g=cl)
    for j in 1:k
        df[!, Symbol("z", j)] = Z[:, j]
    end
    return df
end

"""
Binary-instrument / binary-treatment DGP with compliance types.

Compliance shares (compliers, always-takers, never-takers) may depend on a
covariate `x`; P(Z = 1 | x) is logistic in `x` when `confounded = true`. Potential
outcomes: `Y(0) = x + e`, `Y(1) = Y(0) + late + slope * x` for compliers (always- and
never-takers have level shifts). Returns the data and the true complier LATE /
complier means computed from the realized types.
"""
function iv_binary_dgp(rng; n=2000, pc=0.5, pa=0.2, late=2.0, slope=1.0,
                       confounded=false, defier_share=0.0, direct=0.0)
    x = randn(rng, n)
    pz = confounded ? 1 ./ (1 .+ exp.(-0.8 .* x)) : fill(0.5, n)
    z = Float64.(rand(rng, n) .< pz)
    pcx = clamp.(pc .+ 0.15 .* sign.(x), 0.0, 1.0)     # complier share varies with x
    u = rand(rng, n)
    at = u .< pa
    co = (u .>= pa) .& (u .< pa .+ pcx .* (1 - pa - defier_share))
    de = (u .>= 1 - defier_share)
    nt = .!(at .| co .| de)
    d = Float64.(at .| (co .& (z .== 1)) .| (de .& (z .== 0)))
    y0 = x .+ randn(rng, n) .+ 0.5 .* at .- 0.5 .* nt
    y1 = y0 .+ late .+ slope .* x
    y = ifelse.(d .== 1, y1, y0) .+ direct .* z
    df = DataFrame(y=y, d=d, z=z, x=x, pre=randn(rng, n))
    truth = (late=mean(late .+ slope .* x[co]), share=mean(co),
             xbar_c=mean(x[co]), xbar_a=mean(x[at]), xbar_n=mean(x[nt]))
    return df, truth
end

"""Binomial Monte Carlo tolerance: `c` standard errors plus slack."""
mc_tol(p, R; c=3.5, slack=0.01) = c * sqrt(p * (1 - p) / R) + slack

"""
Judge-design DGP: `n_courts` courts (the randomization strata), `judges` judges per
court, `cases` cases per judge on average (cases are assigned uniformly at random to
the judges of their court). Judge leniency `λ_j ~ U(0.2, 0.7)` plus a court shift.
Case resistance `U = Φ(0.7x + e)`; treatment `D = 1{U < λ_j}` for group `g = 0` and
`D = 1{U < (1 − defy) λ_j + defy (0.9 − λ_j)}` for `g = 1` (`defy > 0` breaks
monotonicity: lenient judges become strict for group 1). Outcome
`Y = 0.5x + D(τ + slope(U − 0.5)) + direct · γ_j + ε`
(`direct > 0` breaks exclusion).
"""
function iv_judge_dgp(rng; n_courts=10, judges=8, cases=60, tau=1.0, slope=0.0,
                      direct=0.0, defy=0.0)
    J = n_courts * judges
    court_of = repeat(1:n_courts; inner=judges)
    λ = 0.2 .+ 0.5 .* rand(rng, J) .+ 0.1 .* randn(rng, n_courts)[court_of]
    γ = randn(rng, J)
    n = J * cases
    court = rand(rng, 1:n_courts, n)
    judge = [(c - 1) * judges + rand(rng, 1:judges) for c in court]
    x = randn(rng, n)
    g = Float64.(rand(rng, n) .< 0.5)
    U = DrSnow.cdf.(DrSnow.Normal(), 0.7 .* x .+ randn(rng, n))
    λg = λ[judge] .+ defy .* g .* (0.9 .- 2 .* λ[judge])
    D = Float64.(U .< λg)
    Y = 0.5 .* x .+ D .* (tau .+ slope .* (U .- 0.5)) .+ direct .* γ[judge] .+
        randn(rng, n)
    return DataFrame(y=Y, d=D, judge=judge, court=court, x=x, g=g, u=U)
end

"""
Normal selection (Roy-type) DGP with a known MTE: `V ~ N(0, 1)`,
`D = 1{V ≤ γ₀ + γ₁ z + γₓ x}` (probit propensity), `Y₀ = x + U₀`,
`Y₁ = ate + (1 + bx) x + U₁`, `(U₀, U₁)` with `Cov(U₁ − U₀, V) = s`, so
`MTE(x, u) = ate + bx·x + s Φ⁻¹(u)` (decreasing in u when s < 0: units most likely
to take up the treatment gain most). Returns data and the truth (ATE, ATT, ATU from
the realized potential outcomes of a large auxiliary sample; `mte(x, u)`).
"""
function iv_mte_dgp(rng; n=3000, ate=1.0, s=-0.8, bx=0.5, γ=(0.0, 0.8, 0.3),
                    zdist=:normal)
    function draw(m)
        z = zdist === :normal ? randn(rng, m) : rand(rng, [-1.0, 0.0, 1.0], m)
        x = randn(rng, m)
        V = randn(rng, m)
        e0, e1 = randn(rng, m), randn(rng, m)
        U0 = 0.3 .* V .+ e0
        U1 = (0.3 + s) .* V .+ e1
        idx = γ[1] .+ γ[2] .* z .+ γ[3] .* x
        D = Float64.(V .<= idx)
        Y0 = x .+ U0
        Y1 = ate .+ (1 + bx) .* x .+ U1
        (z=z, x=x, D=D, Y=ifelse.(D .== 1, Y1, Y0), Δ=Y1 .- Y0,
         P=DrSnow.cdf.(DrSnow.Normal(), idx))
    end
    d = draw(n)
    df = DataFrame(y=d.Y, d=d.D, z=d.z, x=d.x)
    big = draw(400_000)
    t = big.D .== 1
    truth = (ATE=ate, ATT=mean(big.Δ[t]), ATU=mean(big.Δ[.!t]),
             mte=(x, u) -> ate + bx * x + s * DrSnow.quantile(DrSnow.Normal(), u),
             P=d.P)
    return df, truth
end
