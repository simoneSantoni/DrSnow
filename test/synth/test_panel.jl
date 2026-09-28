function sim_block_panel(rng; N0=20, N1=3, T0=15, T1=5, tau=2.0, sigma=0.5)
    N, T = N0 + N1, T0 + T1
    alpha = randn(rng, N)
    beta = cumsum(randn(rng, T))
    f = randn(rng, N)
    g = sin.((1:T) ./ 3)
    rows = NamedTuple[]
    for i in 1:N, t in 1:T
        d = i > N0 && t > T0 ? 1 : 0
        y = alpha[i] + beta[t] + f[i] * g[t] + tau * d + sigma * randn(rng)
        push!(rows, (unit="u" * lpad(i, 3, '0'), year=2000 + t, y=y, d=d,
                     x=randn(rng)))
    end
    return DataFrame(rows)
end

@testset "construction by key" begin
    df = sim_block_panel(StableRNG(1))
    p = synth_panel(df, :y, :d, :unit, :year; covariates=[:x])
    @test size(p.Y) == (23, 20)
    @test p.n_control == 20
    @test p.units[21:23] == ["u021", "u022", "u023"]
    @test all(p.adoption[1:20] .== 0) && all(p.adoption[21:23] .== 16)
    @test p.times == collect(2001:2020)
    # value lookup by key
    r = df[(df.unit .== "u005") .& (df.year .== 2007), :]
    @test p.Y[5, 7] == r.y[1]
    @test p.X[5, 7, 1] == r.x[1]
    shuffled = df[randperm(StableRNG(2), nrow(df)), :]
    q = synth_panel(shuffled, :y, :d, :unit, :year; covariates=[:x])
    @test q.Y == p.Y && q.units == p.units && isequal(q.X, p.X)
    s = sprint(show, MIME"text/plain"(), p)
    @test occursin("23 units × 20 periods", s) && occursin("block", s)
end

@testset "staggered adoption ordering" begin
    df = sim_block_panel(StableRNG(3))
    df.d[(df.unit .== "u005") .& (df.year .>= 2010)] .= 1
    p = synth_panel(df, :y, :d, :unit, :year)
    @test p.n_control == 19
    @test p.units[20] == "u005"          # earlier adopter first among treated
    @test p.adoption[20] == 10
    @test DrSnow._sc_adoption_indices(p) == [10, 16]
end

@testset "input errors" begin
    df = sim_block_panel(StableRNG(4))
    @test_throws ArgumentError synth_panel(df, :nope, :d, :unit, :year)
    unbalanced = df[2:end, :]
    err = try
        synth_panel(unbalanced, :y, :d, :unit, :year)
    catch e
        e
    end
    @test err isa ArgumentError && occursin("unbalanced", err.msg)
    @test_throws ArgumentError synth_panel(vcat(df, df[1:1, :]), :y, :d, :unit, :year)
    bad = copy(df)
    bad.d = Float64.(bad.d)
    bad.d[1] = 0.5
    @test_throws ArgumentError synth_panel(bad, :y, :d, :unit, :year)
    miss = allowmissing(copy(df))
    miss.y[3] = missing
    @test_throws ArgumentError synth_panel(miss, :y, :d, :unit, :year)
    switch = copy(df)
    switch.d[(switch.unit .== "u021") .& (switch.year .== 2020)] .= 0
    @test_throws ArgumentError synth_panel(switch, :y, :d, :unit, :year)
    first_period = copy(df)
    first_period.d[first_period.unit .== "u021"] .= 1
    @test_throws ArgumentError synth_panel(first_period, :y, :d, :unit, :year)
    allt = copy(df)
    allt.d .= allt.year .>= 2016
    @test_throws ArgumentError synth_panel(allt, :y, :d, :unit, :year)
    none = copy(df)
    none.d .= 0
    @test_throws ArgumentError synth_panel(none, :y, :d, :unit, :year)
end
