# StratifiedVcov / stratified_vcov: identities with the Vcov estimators, the matrix
# interface, FWL with absorbed fixed effects, input checks, the R `survey` reference
# values in test/validation/core/, and a Monte Carlo coverage check under stratified
# randomization with stratum-specific effects.

using FixedEffectModels: reg, @formula, fe
using CSV

const _CORE_VALDIR = joinpath(@__DIR__, "..", "validation", "core")

# Explicit Σ_h n_h/(n_h-1) Σ_j (ψ_hj - ψ̄_h)(ψ_hj - ψ̄_h)' from row scores and labels.
function _strat_explicit(X, e, strata, cluster)
    S = X .* e
    units = cluster === nothing ? collect(1:length(e)) : cluster
    U = Dict{Any,Vector{Float64}}()
    hs = Dict{Any,Any}()
    for i in eachindex(e)
        U[units[i]] = get(U, units[i], zeros(size(X, 2))) .+ S[i, :]
        hs[units[i]] = strata[i]
    end
    meat = zeros(size(X, 2), size(X, 2))
    for h in unique(strata)
        us = [u for u in keys(U) if hs[u] == h]
        n_h = length(us)
        m = sum(U[u] for u in us) ./ n_h
        for u in us
            d = U[u] .- m
            meat .+= n_h / (n_h - 1) .* d * d'
        end
    end
    B = inv(X' * X)
    return B * meat * B
end

@testset "StratifiedVcov" begin
    rng = StableRNG(11)
    H, nh = 20, 10
    N = H * nh
    df = DataFrame(block=repeat(1:H, inner=nh), pair=repeat(1:(N ÷ 2), inner=2))
    df.z = vcat([shuffle(rng, [ones(5); zeros(5)]) for _ in 1:H]...)
    df.x = randn(rng, N)
    df.y = 0.3 .* (df.block .- 10) .* df.z .+ df.block ./ 4 .+ df.x .+ randn(rng, N)
    df.one = ones(Int, N)

    @testset "explicit formula, FWL and matrix interface" begin
        X = [ones(N) df.z df.x]
        b = X \ df.y
        e = df.y .- X * b
        m = reg(df, @formula(y ~ z + x), StratifiedVcov(:block; small_sample=false))
        @test coef(m) ≈ b
        @test vcov(m) ≈ _strat_explicit(X, e, df.block, nothing)
        @test stratified_vcov(X, e, df.block; small_sample=false) ≈ vcov(m)
        mc = reg(df, @formula(y ~ z + x), StratifiedVcov(:block; cluster=:pair,
                                                         small_sample=false))
        @test vcov(mc) ≈ _strat_explicit(X, e, df.block, df.pair)
        @test stratified_vcov(X, e, df.block; cluster=df.pair, small_sample=false) ≈
              vcov(mc)
        # small-sample factor (N - 1) / (N - K)
        ms = reg(df, @formula(y ~ z + x), StratifiedVcov(:block))
        @test vcov(ms) ≈ vcov(m) .* ((N - 1) / (N - 3))
        @test stratified_vcov(X, e, df.block) ≈ vcov(ms)
        # absorbed fixed effects: scores of the demeaned regression
        mf = reg(df, @formula(y ~ z + x + fe(block)), StratifiedVcov(:block;
                                                                   small_sample=false))
        Xd = copy(X[:, 2:3])
        yd = copy(df.y)
        for h in 1:H
            rows = df.block .== h
            Xd[rows, :] .-= mean(Xd[rows, :]; dims=1)
            yd[rows] .-= mean(yd[rows])
        end
        bd = Xd \ yd
        @test coef(mf) ≈ bd
        @test vcov(mf) ≈ _strat_explicit(Xd, yd .- Xd * bd, df.block, nothing)
        @test dof_residual(mf) == N - H
        # design degrees of freedom: units minus strata
        @test dof_residual(mc) == N ÷ 2 - H
    end

    @testset "reduces to Vcov.cluster / HC with one stratum" begin
        mcl = reg(df, @formula(y ~ z + x), Vcov.cluster(:pair))
        ms1 = reg(df, @formula(y ~ z + x), StratifiedVcov(:one; cluster=:pair))
        @test vcov(ms1) ≈ vcov(mcl)
        @test dof_residual(ms1) == dof_residual(mcl)
        mhc = reg(df, @formula(y ~ z + x), Vcov.robust())
        ms0 = reg(df, @formula(y ~ z + x), StratifiedVcov(:one; small_sample=false))
        @test vcov(ms0) ≈ vcov(mhc) .* ((N - 3) / N) .* (N / (N - 1))
    end

    @testset "row order, missing values and input checks" begin
        m = reg(df, @formula(y ~ z + x + fe(block)), StratifiedVcov(:block))
        perm = randperm(rng, N)
        mp = reg(df[perm, :], @formula(y ~ z + x + fe(block)), StratifiedVcov(:block))
        @test vcov(mp) ≈ vcov(m)
        @test coef(mp) ≈ coef(m)
        dm = allowmissing(df)
        dm.block[1] = missing
        mm = reg(dm, @formula(y ~ z + x), StratifiedVcov(:block))
        @test nobs(mm) == N - 1
        @test vcov(mm) ≈ vcov(reg(df[2:end, :], @formula(y ~ z + x),
                                  StratifiedVcov(:block)))
        # clusters must be nested in strata
        @test_throws ArgumentError reg(df, @formula(y ~ z),
                                       StratifiedVcov(:pair; cluster=:block))
        # every stratum needs two variance units
        df1 = copy(df)
        df1.block[1] = 99
        err = try
            reg(df1, @formula(y ~ z), StratifiedVcov(:block))
            nothing
        catch e
            e
        end
        @test err isa ArgumentError
        @test occursin("99", err.msg)
        X = [ones(N) df.z]
        @test_throws DimensionMismatch stratified_vcov(X, df.y, df.block[1:10])
        @test_throws ArgumentError stratified_vcov(X, df.y, df.block; dof=0)
        @test occursin("block", sprint(show, StratifiedVcov(:block; cluster=:pair)))
    end

    @testset "experiment_estimate and ri_regression accept it" begin
        r = experiment_estimate(df, :y, :z; method=:block_fe, blocks=:block,
                                covariates=[:x], vcov=StratifiedVcov(:block))
        m = reg(df, @formula(y ~ z + x + fe(block)), StratifiedVcov(:block))
        @test coef(r)[1] ≈ coef(m)[1]
        @test stderror(r)[1] ≈ stderror(m)[1]
        @test dof_residual(r) == N - H
        @test r.variance == "stratified sandwich"
        rr = ri_regression(df, :y, :z; strata=:block, fe=[:block],
                           vcov=StratifiedVcov(:block), nperm=20, rng=StableRNG(3))
        m0 = reg(df, @formula(y ~ z + fe(block)), StratifiedVcov(:block))
        @test rr.table.estimate[1] ≈ coef(m0)[1]
        @test rr.table.std_error[1] ≈ stderror(m0)[1]
    end

    @testset "reference: R survey (svyglm, strata and strata + PSUs)" begin
        d = CSV.read(joinpath(_CORE_VALDIR, "stratified_vcov_data.csv"), DataFrame)
        ref = CSV.read(joinpath(_CORE_VALDIR, "stratified_vcov_reference.csv"),
                       DataFrame)
        fits = Dict(
            "strata" => reg(d, @formula(y ~ z + x1 + x2),
                            StratifiedVcov(:stratum; small_sample=false)),
            "strata_fe" => reg(d, @formula(y ~ z + x1 + x2 + fe(stratum)),
                               StratifiedVcov(:stratum; small_sample=false)),
            "cluster" => reg(d, @formula(y ~ z + x1 + x2),
                             StratifiedVcov(:stratum; cluster=:cluster,
                                            small_sample=false)),
            "cluster_fe" => reg(d, @formula(y ~ z + x1 + x2 + fe(stratum)),
                                StratifiedVcov(:stratum; cluster=:cluster,
                                               small_sample=false)))
        for (nm, m) in fits
            cn = coefnames(m)
            V = vcov(m)
            for row in eachrow(ref[ref.spec .== nm, :])
                i = findfirst(==(String(row.row)), cn)
                j = findfirst(==(String(row.col)), cn)
                @test coef(m)[i] ≈ row.coef_row atol = 1e-12
                @test V[i, j] ≈ row.vcov atol = 1e-13
                @test dof_residual(m) == row.degf
            end
        end
    end

    @testset "Monte Carlo coverage under stratified randomization" begin
        # Constant effect within each stratum, effects differing across strata, fixed
        # potential outcomes, treatment re-randomized within strata: the stratified
        # estimator with stratum fixed effects is (asymptotically) exact, HC1 is
        # conservative because it keeps the between-stratum score variation.
        reps = mc_reps(2000, 300)
        mrng = StableRNG(2024)
        Hs, ns = 20, 10
        Ns = Hs * ns
        blk = repeat(1:Hs, inner=ns)
        tau_h = 0.3 .* ((1:Hs) .- 10.5)
        y0 = 0.5 .* blk .+ randn(mrng, Ns)
        y1 = y0 .+ tau_h[blk]
        truth = mean(y1 .- y0)
        cover_s = 0
        cover_h = 0
        for _ in 1:reps
            z = vcat([shuffle(mrng, [ones(5); zeros(5)]) for _ in 1:Hs]...)
            sim = DataFrame(blk=blk, z=z, y=ifelse.(z .== 1, y1, y0))
            for (est, acc) in ((StratifiedVcov(:blk), :s), (Vcov.robust(), :h))
                m = reg(sim, @formula(y ~ z + fe(blk)), est)
                lo, hi = confint(m)[1, :]
                hit = lo <= truth <= hi
                acc === :s ? (cover_s += hit) : (cover_h += hit)
            end
        end
        @test 0.92 <= cover_s / reps <= 0.98
        @test cover_h / reps >= 0.975
        @test cover_h > cover_s
    end
end
