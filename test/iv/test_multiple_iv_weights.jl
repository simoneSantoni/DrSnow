# 2SLS with several discrete instruments: weights on response groups
# (Mogstad, Torgovitsky & Walters 2021). Checked by construction on "population"
# datasets in which every (instrument cell, response group) combination appears with
# its exact frequency, so sample moments equal population moments; against
# FixedEffectModels 2SLS; and against MTW Propositions 5–7.

"""
Exact population: `cells` (K × L), cell shares `s` and group shares `π` in units of
1/20, response `patterns` (0/1 over cells), group effects `Δ` and baseline means `μ`.
"""
function miw_population(cells, s20, π20, patterns, Δ, μ; m=1)
    rows = NamedTuple[]
    for (k, z) in enumerate(eachrow(cells)), g in eachindex(π20)
        cnt = s20[k] * π20[g] * m
        dz = patterns[g][k]
        for _ in 1:cnt
            push!(rows, (z1=z[1], z2=z[2], d=Float64(dz), y=μ[g] + dz * Δ[g], grp=g))
        end
    end
    return DataFrame(rows)
end

const MIW_CELLS = [0.0 0.0; 0.0 1.0; 1.0 0.0; 1.0 1.0]
#                      at      nt      ec      rc      1c      2c
const MIW_PATTERNS = [[1, 1, 1, 1], [0, 0, 0, 0], [0, 1, 1, 1], [0, 0, 0, 1],
                      [0, 0, 1, 1], [0, 1, 0, 1]]

@testset "Multiple-instrument 2SLS weights (MTW 2021)" begin
    π20 = [2, 4, 3, 3, 5, 3]
    Δ = [0.0, 0.0, 1.0, 2.0, 3.0, -4.0]
    μ = [1.0, 0.0, 0.5, 0.2, -0.3, 0.1]

    @testset "two binary instruments, negatively correlated (MTW §III.B)" begin
        df = miw_population(MIW_CELLS, [1, 9, 9, 1], π20, MIW_PATTERNS, Δ, μ)
        w = multiple_iv_weights(df, :d, [:z1, :z2]; outcome=:y, rng=StableRNG(1),
                                n_boot=99)
        @test w.cells.p ≈ [0.1, 0.4, 0.5, 0.8]
        @test w.rectangular
        @test nrow(w.groups) == 4
        lbl = Dict(r.pattern => r.group for r in eachrow(w.groups))
        @test lbl["0111"] == "eager complier"
        @test lbl["0001"] == "reluctant complier"
        @test startswith(lbl["0011"], "z1 complier")
        @test startswith(lbl["0101"], "z2 complier")
        c = Dict(r.pattern => r.c for r in eachrow(w.groups))
        @test c["0011"] ≈ 0.04 atol = 1e-12
        @test c["0101"] ≈ -0.005 atol = 1e-12
        @test c["0111"] ≈ 0.0175 atol = 1e-12
        @test c["0001"] ≈ 0.0175 atol = 1e-12
        # Proposition 5: π1c ≥ π2c, so sgn(ω2c) = sgn(Cov(D, Z2)) and ω1c ≥ 0
        @test sign(c["0101"]) == sign(cov(df.d, df.z2))
        @test w.negative_weight_possible
        # exact decomposition of the 2SLS estimand with the true shares
        ωtrue = DrSnow._iv_miw_true_weights(MIW_CELLS, [1, 9, 9, 1] ./ 20,
                                            BitVector.(MIW_PATTERNS), π20 ./ 20)
        @test sum(ωtrue) ≈ 1
        @test w.tsls ≈ dot(ωtrue, Δ) rtol = 1e-10
        @test ωtrue[6] < 0
        # saturated 2SLS equals FixedEffectModels 2SLS with interacted instruments
        df.z12 = df.z1 .* df.z2
        @test w.tsls ≈ DrSnow.estimate(late_2sls(df, :y, :d, [:z1, :z2, :z12])) rtol = 1e-9
        # the true weights lie in the LP bounds; always-/never-taker shares identified
        for (j, pat) in enumerate(["0111", "0001", "0011", "0101"])
            row = w.groups[findfirst(==(pat), w.groups.pattern), :]
            g = findfirst(==(pat), join.(MIW_PATTERNS))
            @test row.min_share - 1e-9 <= π20[g] / 20 <= row.max_share + 1e-9
            @test row.weight_lower - 1e-9 <= ωtrue[g] <= row.weight_upper + 1e-9
        end
        @test w.negative_weight_bounds[1] - 1e-9 <= ωtrue[6] <=
              w.negative_weight_bounds[2] + 1e-9
        @test w.negative_weight_bounds[1] < 0
        # Proposition 7: c_g = Σ_k (1[k ∈ C_g] − 1[k ∈ D_g]) Cov(D, 1[p(Z) ≥ p(z^k)])
        pz = w.cells.p[[findfirst(r -> r.z1 == a && r.z2 == b, eachrow(w.cells))
                        for (a, b) in zip(df.z1, df.z2)]]
        ordp = sort(w.cells.p)
        for row in eachrow(w.groups)
            cz(k) = cov(df.d, Float64.(pz .>= ordp[k]))
            s = sum(cz, row.complier_at; init=0.0) - sum(cz, row.defier_at; init=0.0)
            @test s * (nrow(df) - 1) / nrow(df) ≈ row.c atol = 1e-12
        end
        @test w.groups.complier_at[findfirst(==("0101"), w.groups.pattern)] == [2, 4]
        @test w.groups.defier_at[findfirst(==("0101"), w.groups.pattern)] == [3]
        # linear first stage: equals the additive 2SLS estimate, and decomposes it too
        wl = multiple_iv_weights(df, :d, [:z1, :z2]; outcome=:y, first_stage=:linear,
                                 rng=StableRNG(2), n_boot=99)
        @test wl.tsls ≈ DrSnow.estimate(late_2sls(df, :y, :d, [:z1, :z2])) rtol = 1e-9
        ωl = DrSnow._iv_miw_true_weights(MIW_CELLS, [1, 9, 9, 1] ./ 20,
                                         BitVector.(MIW_PATTERNS), π20 ./ 20;
                                         first_stage=:linear)
        @test wl.tsls ≈ dot(ωl, Δ) rtol = 1e-10
        io = IOBuffer()
        show(io, MIME"text/plain"(), w)
        @test occursin("Negative weights are possible", String(take!(io)))
    end

    @testset "independent instruments: no negative weights (Proposition 6)" begin
        df = miw_population(MIW_CELLS, [5, 5, 5, 5], π20, MIW_PATTERNS, Δ, μ)
        w = multiple_iv_weights(df, :d, [:z1, :z2]; rng=StableRNG(3), n_boot=99)
        @test all(w.groups.c .> 0)
        @test !w.negative_weight_possible
        @test w.negative_weight_bounds == (0.0, 0.0)
        @test w.test.pvalue > 0.5
        # IAM (all groups thresholds in p): only nested groups
        wi = multiple_iv_weights(df, :d, [:z1, :z2]; assumption=:iam, rng=StableRNG(3),
                                 n_boot=99)
        @test nrow(wi.groups) == 3
        @test all(wi.groups.c .> 0)
        # IAM cannot reproduce propensities generated with both Z1 and Z2 compliers?
        # (it can: shares are only restricted through p) — the LP stays feasible here
        @test wi.feasible
    end

    @testset "vector monotonicity, directions and infeasibility" begin
        df = miw_population(MIW_CELLS, [1, 9, 9, 1], π20, MIW_PATTERNS, Δ, μ)
        wv = multiple_iv_weights(df, :d, [:z1, :z2]; assumption=:vm, rng=StableRNG(4),
                                 n_boot=99)
        @test nrow(wv.groups) == 4            # same six groups as PM here
        wr = multiple_iv_weights(df, :d, [:z1, :z2]; assumption=:vm, directions=[1, -1],
                                 rng=StableRNG(4), n_boot=99)
        @test !wr.feasible                    # wrong direction contradicts p(z)
        @test all(isnan, wr.groups.max_share)
        io = IOBuffer()
        show(io, MIME"text/plain"(), wr)
        @test occursin("not compatible", String(take!(io)))
    end

    @testset "judge design: one multivalued instrument (PM = IAM)" begin
        rng = StableRNG(5)
        n = 6000
        judge = rand(rng, 1:5, n)
        len = [0.2, 0.35, 0.5, 0.6, 0.8]
        u = rand(rng, n)
        d = Float64.(u .< len[judge])
        y = d .* (1 .+ u) .+ randn(rng, n)
        df = DataFrame(y=y, d=d, judge=judge)
        w = multiple_iv_weights(df, :d, :judge; outcome=:y, rng=StableRNG(6), n_boot=199)
        @test nrow(w.groups) == 4             # threshold groups between 5 judges
        @test all(w.groups.c .> 0)
        @test !w.negative_weight_possible
        wi = multiple_iv_weights(df, :d, :judge; assumption=:iam, rng=StableRNG(6),
                                 n_boot=199)
        @test sort(wi.groups.pattern) == sort(w.groups.pattern)
        # saturated 2SLS = judge-dummy 2SLS
        for j in 2:5
            df[!, Symbol("j", j)] = Float64.(df.judge .== j)
        end
        rj = late_2sls(df, :y, :d, [:j2, :j3, :j4, :j5])
        @test w.tsls ≈ DrSnow.estimate(rj) rtol = 1e-9
        # a *linear* first stage in the judge id is a misspecified index; still valid
        # weights are reported (and here remain positive: p is monotone in the id)
        wl = multiple_iv_weights(df, :d, :judge; first_stage=:linear, rng=StableRNG(6),
                                 n_boot=199)
        @test all(wl.groups.c .> 0)
    end

    @testset "row order, missing values and errors" begin
        df = miw_population(MIW_CELLS, [1, 9, 9, 1], π20, MIW_PATTERNS, Δ, μ)
        w1 = multiple_iv_weights(df, :d, [:z1, :z2]; rng=StableRNG(7), n_boot=49)
        w2 = multiple_iv_weights(df[randperm(StableRNG(8), nrow(df)), :], :d, [:z1, :z2];
                                 rng=StableRNG(7), n_boot=49)
        @test w1.groups.c ≈ w2.groups.c
        @test w1.groups.max_share ≈ w2.groups.max_share
        dm = allowmissing(df)
        dm[1, :z1] = missing
        @test multiple_iv_weights(dm, :d, [:z1, :z2]; n_boot=49).n == nrow(df) - 1
        @test_throws ArgumentError multiple_iv_weights(df, :y, [:z1, :z2])
        df.cont = randn(StableRNG(9), nrow(df))
        @test_throws ArgumentError multiple_iv_weights(df, :d, [:cont])   # > max_cells
        @test_throws ArgumentError multiple_iv_weights(df, :d, [:z1]; assumption=:foo)
        @test_throws ArgumentError multiple_iv_weights(df, :d, [:z1]; first_stage=:foo)
        @test_throws ArgumentError multiple_iv_weights(df, :d, Symbol[])
        @test_throws ArgumentError multiple_iv_weights(df, :d, [:z1]; n_boot=5)
        @test_throws ArgumentError multiple_iv_weights(df, :d, [:z1, :z2]; assumption=:vm,
                                                       directions=[1])
        df0 = DataFrame(d=[0.0, 1.0, 0.0, 1.0], z=[0.0, 0.0, 1.0, 1.0])
        @test_throws ArgumentError multiple_iv_weights(df0, :d, :z)       # no first stage
        @test_throws ArgumentError multiple_iv_weights(df0, :d, :nope)
    end

    @testset "Monte Carlo: test of non-negative weights" begin
        reps = mc_reps(300, 40)
        function draw(rng, n, s, π)
            cellp = cumsum(s)
            grp = cumsum(π)
            z = [findfirst(>=(rand(rng)), cellp) for _ in 1:n]
            g = [findfirst(>=(rand(rng)), grp) for _ in 1:n]
            d = [Float64(MIW_PATTERNS[g[i]][z[i]]) for i in 1:n]
            return DataFrame(z1=MIW_CELLS[z, 1], z2=MIW_CELLS[z, 2], d=d)
        end
        rej0 = 0
        rej1 = 0
        for r in 1:reps
            rng = StableRNG(9000 + r)
            # H0 true: independent instruments, all c_g > 0
            d0 = draw(rng, 2000, fill(0.25, 4), π20 ./ 20)
            rej0 += multiple_iv_weights(d0, :d, [:z1, :z2]; rng=rng,
                                        n_boot=199).test.pvalue < 0.05
            # H0 false: strongly negatively correlated instruments, many Z2 compliers
            d1 = draw(rng, 2000, [0.02, 0.48, 0.48, 0.02],
                      [0.1, 0.2, 0.05, 0.05, 0.35, 0.25])
            rej1 += multiple_iv_weights(d1, :d, [:z1, :z2]; rng=rng,
                                        n_boot=199).test.pvalue < 0.05
        end
        se = sqrt(0.05 * 0.95 / reps)
        @test rej0 / reps <= 0.05 + 3se
        @test rej1 / reps >= 0.7
    end
end
