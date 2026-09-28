@testset "pretreatment_balance" begin
    rng = StableRNG(801)
    df = sim_staggered(rng; N=300, T=6, cohorts=[0, 3, 5], confounded=true)
    b = pretreatment_balance(df, :d, :unit, :time; covariates=[:x, :xt])
    @test Set(b.cohort) == Set([2002, 2004])
    @test Set(b.covariate) == Set([:x, :xt])
    # manual computation for cohort 2004 (index 5): pre-periods t < 2004
    r = b[(b.cohort .== 2004) .& (b.covariate .== :x), :]
    tr = (df.g .== 2004) .& (df.time .< 2004)
    ct = (df.g .== 0) .& (df.time .< 2004)
    @test r.treated_mean[1] ≈ mean(df.x[tr])
    @test r.control_mean[1] ≈ mean(df.x[ct])
    @test r.n_treated[1] == count(tr)
    @test r.std_diff[1] ≈ (mean(df.x[tr]) - mean(df.x[ct])) /
                          sqrt((var(df.x[tr]) + var(df.x[ct])) / 2)
    @test all(b.std_diff[b.covariate .== :x] .> 0.4)    # confounded design
    # not-yet-treated comparison also uses the later cohort's pre-periods
    bn = pretreatment_balance(df, :d, :unit, :time; covariates=[:x],
                              control_group=:not_yet_treated)
    r2 = bn[bn.cohort .== 2002, :]
    @test r2.n_control[1] == count(((df.g .== 0) .| (df.g .== 2004)) .& (df.time .< 2002))
    p = TreatmentPanel(df, :y, :d, :unit, :time, [:x])
    @test pretreatment_balance(p).treated_mean ≈ b.treated_mean[b.covariate .== :x]
    @test_throws ArgumentError pretreatment_balance(df, :d, :unit, :time;
                                                    covariates=Symbol[])
end
