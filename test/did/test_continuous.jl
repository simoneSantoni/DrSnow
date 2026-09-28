# Continuous-treatment DiD (did_continuous): closed forms, recovery of a known
# dose-response, invariance, errors and Monte Carlo coverage.

# Two-period panel: a share of units untreated, the rest receive a dose in (0, 1];
# ATT(d) = f(d) for everybody (strong parallel trends), trends independent of dose.
function cont_sim(rng; n=600, p0=0.3, f=d -> 2d^2 + d, sigma=1.0, cl=nothing)
    D = [rand(rng) < p0 ? 0.0 : 0.05 + 0.95rand(rng) for _ in 1:n]
    a = randn(rng, n)
    y1 = a .+ sigma .* randn(rng, n)
    y2 = a .+ 0.5 .+ f.(D) .* (D .> 0) .+ sigma .* randn(rng, n)
    df = DataFrame(id=vcat(1:n, 1:n), year=vcat(fill(1, n), fill(2, n)),
                   y=vcat(y1, y2), dose=vcat(D, D))
    df.region = mod1.(df.id, 40)
    return df
end

@testset "did_continuous" begin
    rng = StableRNG(55)
    df = cont_sim(rng)

    @testset "closed forms (linear spline)" begin
        r = did_continuous(df, :y, :dose, :id, :year; degree=1)
        n = nrow(df) ÷ 2
        w = unstack(df, :id, :year, :y)
        d = df[df.year .== 2, :]
        d = d[sortperm(d.id), :]
        dy = w[!, "2"] .- w[!, "1"]
        tr = d.dose .> 0
        # ATT^o = difference in mean changes, HC0-type influence-function SE
        att = mean(dy[tr]) - mean(dy[.!tr])
        @test coef(r)[1] ≈ att atol = 1e-10
        se = sqrt(var(dy[tr]; corrected=false) / count(tr) +
                  var(dy[.!tr]; corrected=false) / count(.!tr))
        @test stderror(r)[1] ≈ se rtol = 1e-8
        # ACRT^o = OLS slope of ΔY on D among the treated
        sub = DataFrame(dy=dy[tr], dose=d.dose[tr])
        m = DrSnow.reg(sub, DrSnow.make_formula(:dy, [:dose]), DrSnow.Vcov.robust())
        @test coef(r)[2] ≈ coef(m)[2] atol = 1e-10
        n1 = count(tr)
        @test stderror(r)[2] * sqrt(n1 / (n1 - 2)) ≈ stderror(m)[2] rtol = 1e-8
        @test all(abs.(diff(r.acrt)) .< 1e-10)          # constant slope
    end

    @testset "recovers a quadratic dose response" begin
        r = did_continuous(df, :y, :dose, :id, :year; degree=2)
        f(d) = 2d^2 + d
        @test maximum(abs.(r.att .- f.(r.dose)) ./ (4 .* r.att_se)) < 1
        @test maximum(abs.(r.acrt .- (4 .* r.dose .+ 1)) ./ (4 .* r.acrt_se)) < 1
        rk = did_continuous(df, :y, :dose, :id, :year; degree=3, num_knots=2)
        @test length(rk.settings.knots) == 2
        @test coef(rk)[1] ≈ coef(r)[1] atol = 1e-10    # ATT^o is sieve-invariant
        # simultaneous bands are wider than pointwise ones
        cp = confint(r; curve=:att)
        cu = confint(r; curve=:att, uniform=true, rng=StableRNG(1))
        @test all(cu[:, 2] .- cu[:, 1] .>= cp[:, 2] .- cp[:, 1])
        # row order and clustering
        rs = did_continuous(shuffle_rows(StableRNG(2), df), :y, :dose, :id, :year;
                            degree=2)
        @test coef(rs) ≈ coef(r) atol = 1e-10
        @test vcov(rs) ≈ vcov(r) rtol = 1e-8
        rc = did_continuous(df, :y, :dose, :id, :year; degree=2, cluster=:region)
        @test rc.n_clusters == 40 && coef(rc) ≈ coef(r)
        io = IOBuffer()
        show(io, MIME"text/plain"(), r)
        @test occursin("strong parallel trends", String(take!(io)))
    end

    @testset "errors" begin
        @test_throws ArgumentError did_continuous(df, :y, :dose, :id, :year; degree=0)
        three = vcat(df, transform(df[df.year .== 2, :], :year => ByRow(_ -> 3) => :year))
        @test_throws ArgumentError did_continuous(three, :y, :dose, :id, :year)
        allt = df[df.dose .> 0, :]
        @test_throws ArgumentError did_continuous(allt, :y, :dose, :id, :year)
        same = copy(df)
        same.dose[same.dose .> 0] .= 0.5
        @test_throws ArgumentError did_continuous(same, :y, :dose, :id, :year)
        @test_throws ArgumentError did_continuous(df, :y, :dose, :id, :year;
                                                  knots=[2.0])
        neg = copy(df)
        neg.dose[1] = -1
        @test_throws ArgumentError did_continuous(neg, :y, :dose, :id, :year)
    end

    @testset "Monte Carlo coverage" begin
        R = mc_reps(500, 100)
        rng = StableRNG(909)
        hits = zeros(Int, 3)
        f(d) = 2d^2 + d
        for _ in 1:R
            d = cont_sim(rng; n=1000)
            r = did_continuous(d, :y, :dose, :id, :year; degree=2)
            tr = d.dose[d.year .== 2]
            tr = tr[tr .> 0]
            ci = confint(r)
            hits[1] += ci[1, 1] <= mean(f.(tr)) <= ci[1, 2]
            hits[2] += ci[2, 1] <= mean(4 .* tr .+ 1) <= ci[2, 2]
            cu = confint(r; curve=:acrt, uniform=true, rng=rng, ndraws=2000)
            hits[3] += all(cu[:, 1] .<= 4 .* r.dose .+ 1 .<= cu[:, 2])
        end
        # (HC0-type influence-function variances: with a few hundred treated units
        # bands undercover slightly near the dose boundaries, hence n = 1000 here)
        @test mc_close(hits[1] / R, 0.95, R)
        @test mc_close(hits[2] / R, 0.95, R)
        @test hits[3] / R >= 0.95 - 3.5 * sqrt(0.95 * 0.05 / R) - 0.01
    end
end
