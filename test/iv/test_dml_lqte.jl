# DML local quantile treatment effects and complier distributions.
# Reference: test/validation/iv/make_reference_lqte.py (Python DoubleML 0.11.4,
# DoubleMLLPQ / DoubleMLQTE framework difference, unpenalized logistic learners,
# identical outer folds; the nested splits and preliminary IPW quantiles DoubleML draws
# inside each fold are recorded and passed to DrSnow).

using DrSnow: Normal, quantile

const LQTE_DF = iv_read_csv(joinpath(IV_VALDIR, "lqte.csv"))
const LQTE_REF = let r = iv_read_csv(joinpath(IV_VALDIR, "reference_lqte.csv"))
    Dict((row.case, row.quantity) => row.value for row in eachrow(r))
end
const LQTE_PRE = iv_read_csv(joinpath(IV_VALDIR, "lqte_prelim.csv"))
const LQTE_IPW = iv_read_csv(joinpath(IV_VALDIR, "lqte_ipw.csv"))
const LQTE_PREDS = iv_read_csv(joinpath(IV_VALDIR, "lqte_preds.csv"))

"""DGP of the Python reference: compliers 60%, Y(0) ~ N(0.5x₁ − 0.3x₂, 1),
Y(1) = Y(0) + 1 + 0.8·Exp(1); Z depends on x₁."""
function lqte_dgp(rng; n=1200)
    x1, x2 = randn(rng, n), randn(rng, n)
    z = Float64.(rand(rng, n) .< 1 ./ (1 .+ exp.(-0.6 .* x1)))
    u = rand(rng, n)
    at = u .< 0.15
    nt = u .> 0.75
    co = .!(at .| nt)
    d = Float64.(at .| (co .& (z .== 1)))
    y0 = 0.5 .* x1 .- 0.3 .* x2 .+ randn(rng, n)
    y1 = y0 .+ 1.0 .+ 0.8 .* randexp(rng, n)
    return DataFrame(y=ifelse.(d .== 1, y1, y0), d=d, z=z, x1=x1, x2=x2)
end

"""True complier quantiles / CDFs of the DGP (compliers are independent of X)."""
const LQTE_TRUTH = let rng = StableRNG(99), N = 2_000_000
    y0 = sqrt(0.25 + 0.09 + 1.0) .* randn(rng, N)
    y1 = y0 .+ 1.0 .+ 0.8 .* randexp(rng, N)
    (q0=τ -> quantile(Normal(0, sqrt(1.34)), τ), q1=τ -> quantile(y1, τ),
     F0=y -> DrSnow.cdf(Normal(0, sqrt(1.34)), y), F1=y -> mean(y1 .<= y))
end

@testset "DML local quantile treatment effects" begin
    lg = LogisticLearner()
    qs = [0.25, 0.5, 0.75]
    K = 5
    prelim = [(Int.(LQTE_PRE.row[LQTE_PRE.fold .== k]),
               Int.(LQTE_PRE.inner[LQTE_PRE.fold .== k])) for k in 1:K]
    ipw = Dict((Int(r.fold), Int(r.treatment), findfirst(==(r.quantile), qs)) => r.ipw
               for r in eachrow(LQTE_IPW))
    folds = Int.(LQTE_DF.fold)

    @testset "validation against DoubleML (LPQ and QTE)" begin
        r = dml_lqte(LQTE_DF, :y, :d, :z; covariates=[:x1, :x2], quantiles=qs,
                     folds=folds, outcome_learner=lg, treatment_learner=lg,
                     instrument_learner=lg, variance=:doubleml, _prelim=prelim,
                     _ipw=ipw)
        for (j, q) in enumerate(qs)
            for (c, dv) in ((1, 0), (2, 1))
                case = "lpq_d$(dv)_q$q"
                @test r.lpq[j, c] ≈ LQTE_REF[(case, "coef")] atol = 1e-9
                @test r.lpq_se[j, c] ≈ LQTE_REF[(case, "se")] rtol = 1e-6
            end
            @test coef(r)[j] ≈ LQTE_REF[("lqte_q$q", "coef")] atol = 1e-9
            @test stderror(r)[j] ≈ LQTE_REF[("lqte_q$q", "se")] rtol = 1e-6
        end
        @test all(r.converged)
        # nuisance predictions of the (d = 1, τ = 0.5) model
        out = DrSnow._iv_lqte_rep(Float64.(LQTE_DF.y), Float64.(LQTE_DF.d),
                                  Float64.(LQTE_DF.z), Matrix(LQTE_DF[:, [:x1, :x2]]),
                                  folds, qs, lg, lg, lg, 0.01, true, :doubleml,
                                  DrSnow._iv_lqte_seeds(StableRNG(1), K, 3, 1)[:, :, 1],
                                  prelim, ipw, "test")
        @test out.mz ≈ LQTE_PREDS.ml_m_z rtol = 1e-6
        @test out.m0 ≈ LQTE_PREDS.ml_m_d_z0 rtol = 1e-6
        @test out.m1 ≈ LQTE_PREDS.ml_m_d_z1 rtol = 1e-6
        @test out.g0[:, 2, 2] ≈ LQTE_PREDS.ml_g_du_z0 rtol = 1e-5 atol = 1e-9
        @test out.g1[:, 2, 2] ≈ LQTE_PREDS.ml_g_du_z1 rtol = 1e-5 atol = 1e-9
        # the influence-function variance adds the complier-share term
        ri = dml_lqte(LQTE_DF, :y, :d, :z; covariates=[:x1, :x2], quantiles=qs,
                      folds=folds, outcome_learner=lg, treatment_learner=lg,
                      instrument_learner=lg, _prelim=prelim, _ipw=ipw)
        @test coef(ri) == coef(r)
        @test all(stderror(ri) .!= stderror(r))
        @test ri.variance === :influence
    end

    @testset "brentq transcription" begin
        f = x -> x^3 - 2x - 5
        @test DrSnow._iv_brentq(f, 2.0, 3.0) ≈ 2.0945514815423265 atol = 1e-11
        @test_throws ArgumentError DrSnow._iv_brentq(f, 3.0, 4.0)
        # step function: converges to the jump
        g = x -> x < 0.3 ? -1.0 : 1.0
        @test DrSnow._iv_brentq(g, 0.0, 1.0) ≈ 0.3 atol = 1e-11
    end

    @testset "interface" begin
        df = lqte_dgp(StableRNG(31); n=1500)
        r = dml_lqte(df, :y, :d, :z; covariates=[:x1, :x2], quantiles=[0.3, 0.6],
                     outcome_learner=lg, treatment_learner=lg, instrument_learner=lg,
                     n_rep=2, n_boot=200, rng=StableRNG(32))
        @test coefnames(r) == ["LQTE(0.3)", "LQTE(0.6)"]
        @test size(r.all_coef) == (2, 2)
        @test coef(r) ≈ vec(median(r.all_coef; dims=2))
        cu = confint(r; uniform=true)
        cp = confint(r)
        @test all(cu[:, 1] .<= cp[:, 1]) && all(cu[:, 2] .>= cp[:, 2])
        @test length(r.supt) == 400
        t = tidy(r; uniform=true)
        @test t.conf_low ≈ cu[:, 1]
        io = IOBuffer()
        show(io, MIME"text/plain"(), r)
        @test occursin("complier share", String(take!(io)))
        # same seed, same result
        r2 = dml_lqte(df, :y, :d, :z; covariates=[:x1, :x2], quantiles=[0.3, 0.6],
                      outcome_learner=lg, treatment_learner=lg, instrument_learner=lg,
                      n_rep=2, n_boot=200, rng=StableRNG(32))
        @test coef(r2) == coef(r)
        @test_throws ArgumentError dml_lqte(df, :y, :d, :z; quantiles=[1.2])
        @test_throws ArgumentError dml_lqte(df, :y, :d, :z; quantiles=Float64[])
        @test_throws ArgumentError dml_lqte(df, :y, :d, :z; variance=:hc3)
        @test_throws ArgumentError dml_lqte(df, :y, :x1, :z)
        @test_throws ArgumentError dml_lqte(df, :y, :d, :x1)
        @test_throws ArgumentError dml_complier_cdf(df, :y, :d, :z; level=2.0)
        @test_throws ArgumentError dml_complier_cdf(df, :y, :d, :z; grid=Float64[])
        # one-sided non-compliance: P(D = 1 | Z = 0) is degenerate
        d1 = copy(df)
        d1.d = d1.d .* d1.z
        @test_throws ArgumentError dml_lqte(d1, :y, :d, :z; covariates=[:x1],
                                            outcome_learner=lg, treatment_learner=lg,
                                            instrument_learner=lg, rng=StableRNG(1))
    end

    @testset "complier CDFs: agreement with the κ-weighting estimator" begin
        df = lqte_dgp(StableRNG(33); n=4000)
        pts = [-1.0, 0.0, 1.0, 2.0]
        c = dml_complier_cdf(df, :y, :d, :z; covariates=[:x1, :x2], grid=pts,
                             outcome_learner=lg, treatment_learner=lg,
                             instrument_learner=lg, rearrange=false, rng=StableRNG(34))
        k = complier_outcome_distribution(df, :y, :d, :z; covariates=[:x1, :x2],
                                          points=pts)
        @test maximum(abs.(c.table.cdf1 .- k.cdf_treated)) < 0.03
        @test maximum(abs.(c.table.cdf0 .- k.cdf_untreated)) < 0.03
        @test c.table.se1 ≈ k.se_treated rtol = 0.2
        @test c.table.se0 ≈ k.se_untreated rtol = 0.2
        @test all(c.critical_values .> critical_value(0.95))
        io = IOBuffer()
        show(io, MIME"text/plain"(), c)
        @test occursin("uniform bands", String(take!(io)))
        # rearranged: monotone and in [0, 1]
        cr = dml_complier_cdf(df, :y, :d, :z; covariates=[:x1, :x2],
                              outcome_learner=lg, treatment_learner=lg,
                              instrument_learner=lg, rng=StableRNG(34))
        @test issorted(cr.table.cdf0) && issorted(cr.table.cdf1)
        @test all(0 .<= cr.table.lower1 .<= cr.table.upper1 .<= 1)
    end

    @testset "Monte Carlo: coverage (pointwise, uniform, complier CDF bands)" begin
        reps = mc_reps(300, 30)
        qs2 = [0.25, 0.5, 0.75]
        truth = [LQTE_TRUTH.q1(τ) - LQTE_TRUTH.q0(τ) for τ in qs2]
        cov_inf = zeros(3)
        cov_unif = 0
        cov_cdf = 0
        pts = [-1.0, 0.0, 1.0, 2.0, 3.0]
        F1true = LQTE_TRUTH.F1.(pts)
        for rep in 1:reps
            rng = StableRNG(4000 + rep)
            df = lqte_dgp(rng; n=1500)
            r = dml_lqte(df, :y, :d, :z; covariates=[:x1, :x2], quantiles=qs2,
                         outcome_learner=lg, treatment_learner=lg,
                         instrument_learner=lg, n_boot=500, rng=rng)
            ci = confint(r)
            cov_inf .+= (ci[:, 1] .<= truth .<= ci[:, 2])
            cu = confint(r; uniform=true)
            cov_unif += all(cu[:, 1] .<= truth .<= cu[:, 2])
            c = dml_complier_cdf(df, :y, :d, :z; covariates=[:x1, :x2], grid=pts,
                                 outcome_learner=lg, treatment_learner=lg,
                                 instrument_learner=lg, n_boot=500, rng=rng)
            cov_cdf += all(c.table.lower1 .<= F1true .<= c.table.upper1)
        end
        se = sqrt(0.05 * 0.95 / reps)
        @test all(cov_inf ./ reps .>= 0.95 - 3se)
        @test cov_unif / reps >= 0.95 - 3se
        @test cov_cdf / reps >= 0.95 - 3se
    end
end
