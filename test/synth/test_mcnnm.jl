function sim_lowrank_panel(rng; N=30, T=20, r=2, tau=1.5, sigma=0.3, adopt=Dict())
    U = randn(rng, N, r)
    V = randn(rng, T, r)
    a = randn(rng, N)
    b = randn(rng, T)
    rows = NamedTuple[]
    for i in 1:N, t in 1:T
        at = get(adopt, i, 0)
        d = at > 0 && t >= at ? 1 : 0
        y = a[i] + b[t] + dot(U[i, :], V[t, :]) + tau * d + sigma * randn(rng)
        push!(rows, (unit=i, time=t, y=y, d=d))
    end
    return DataFrame(rows)
end

@testset "estimation and interface" begin
    adopt = Dict(27 => 15, 28 => 15, 29 => 18, 30 => 18)
    df = sim_lowrank_panel(StableRNG(51); adopt=adopt)
    r = matrix_completion(df, :y, :d, :unit, :time; replications=20, rng=StableRNG(2))
    @test r isa MatrixCompletionEstimate
    @test abs(r.att - 1.5) < 4 * r.se + 0.3
    @test r.cv isa DataFrame && r.lambda >= 0
    @test coef(r) == [r.att]
    @test nobs(r) == 600
    @test occursin("MC-NNM", sprint(show, MIME"text/plain"(), r))
    g = synth_gaps(r)
    @test nrow(g) == 20 && count(g.post) == 6
    # untreated cells are fitted, treated cells imputed
    p = r.panel
    W = DrSnow._sc_treatment_matrix(p)
    @test r.att ≈ mean((p.Y .- r.Y0hat)[W .== 1])
    # λ = 0 with FE only interpolates observed cells; large λ gives L = 0 (two-way FE)
    rbig = matrix_completion(df, :y, :d, :unit, :time; lambda=1e6, se_method=:none)
    @test rbig.rank == 0
    @test all(iszero, rbig.L)
    rb = matrix_completion(df, :y, :d, :unit, :time; lambda=r.lambda,
                           se_method=:bootstrap, replications=10, rng=StableRNG(3))
    @test rb.se > 0
    rn = matrix_completion(df, :y, :d, :unit, :time; lambda=r.lambda, se_method=:none)
    @test_throws ArgumentError vcov(rn)
    @test rn.att ≈ r.att
    sh = df[randperm(StableRNG(4), nrow(df)), :]
    rs = matrix_completion(sh, :y, :d, :unit, :time; lambda=r.lambda, se_method=:none)
    @test rs.att ≈ rn.att rtol = 1e-12
    rcv1 = matrix_completion(df, :y, :d, :unit, :time; se_method=:none,
                             rng=StableRNG(9))
    rcv2 = matrix_completion(sh, :y, :d, :unit, :time; se_method=:none,
                             rng=StableRNG(9))
    @test rcv1.lambda == rcv2.lambda
end

@testset "errors" begin
    df = sim_lowrank_panel(StableRNG(52); N=10, adopt=Dict(i => 10 for i in 5:10))
    @test_throws ArgumentError matrix_completion(df, :y, :d, :unit, :time)  # placebo
    @test_throws ArgumentError matrix_completion(df, :y, :d, :unit, :time;
                                                 se_method=:foo)
    @test_throws ArgumentError matrix_completion(df, :y, :d, :unit, :time;
                                                 lambda=-1.0)
    @test_throws ArgumentError matrix_completion(df, :y, :d, :unit, :time;
                                                 cv_ratio=1.5)
end
