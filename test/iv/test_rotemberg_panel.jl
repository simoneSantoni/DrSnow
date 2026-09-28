# Panel aggregation of Rotemberg weights and the overidentified (2SLS) decomposition
# (Goldsmith-Pinkham, Sorkin & Swift 2020). Reference:
# test/validation/iv/make_reference_rotemberg.R (bartik.weight::bw on ADH data, GPSS
# aggregation across periods, AER::ivreg).

const IV_REF_ROT = let r = iv_read_csv(joinpath(IV_VALDIR, "reference_rotemberg.csv"))
    Dict((row.case, row.quantity) => row.value for row in eachrow(r))
end
const ADH = iv_read_csv(joinpath(IV_VALDIR, "adh_panel.csv"))
const ADH_SHOCKS = iv_read_csv(joinpath(IV_VALDIR, "adh_shocks.csv"))
const ADH_INDS = sort(unique(string.(Int.(ADH_SHOCKS.ind))))
const ADH_SH = Symbol.("sh_" .* ADH_INDS)
const ADH_CTRL = [:reg_midatl, :reg_encen, :reg_wncen, :reg_satl, :reg_escen,
                  :reg_wscen, :reg_mount, :reg_pacif, :l_sh_popedu_c, :l_sh_popfborn,
                  :l_sh_empl_f, :l_sh_routine33, :l_task_outsource, :t2,
                  :l_shind_manuf_cbp]

adh_shock(ind, yr) = ADH_SHOCKS.trade_[findfirst(i -> string(Int(ADH_SHOCKS.ind[i])) ==
                                                      ind && ADH_SHOCKS.year[i] == yr,
                                                  1:nrow(ADH_SHOCKS))]

@testset "Rotemberg weights: panels and 2SLS" begin
    shocks = Dict(t => [adh_shock(k, t) for k in ADH_INDS] for t in (1990.0, 2000.0))
    rrefb(q) = IV_REF_ROT[("adh_bartik", q)]

    @testset "Bartik panel decomposition vs bartik.weight (ADH)" begin
        rw = rotemberg_weights(ADH, :d_sh_empl_mfg, :d_tradeusch_pw, ADH_SH, shocks;
                               period=:year, covariates=ADH_CTRL, weights=:timepwt48)
        @test rw.estimate ≈ rrefb("estimate") rtol = 1e-9
        @test rw.estimate ≈ rrefb("ivreg") rtol = 1e-9
        bp = rw.by_period
        @test nrow(bp) == 50
        for j in 1:nrow(bp)
            ind = string(bp.sector[j])[4:end]
            yr = Int(bp.period[j])
            @test bp.alpha[j] ≈ rrefb("alpha_$(ind)_$(yr)") atol = 1e-10
            ismissing(bp.beta[j]) ||
                @test bp.beta[j] ≈ rrefb("beta_$(ind)_$(yr)") rtol = 1e-8
        end
        tab = rw.table
        @test nrow(tab) == 25
        @test sum(tab.alpha) ≈ 1.0
        @test issorted(abs.(tab.alpha); rev=true)
        for j in 1:nrow(tab)
            ind = string(tab.sector[j])[4:end]
            @test tab.alpha[j] ≈ rrefb("agg_alpha_$ind") atol = 1e-10
            @test tab.beta[j] ≈ rrefb("agg_beta_$ind") rtol = 1e-7
            @test tab.shock[j] ≈ rrefb("agg_g_$ind") rtol = 1e-7
        end
        @test sum(skipmissing(tab.alpha .* tab.beta)) ≈ rw.estimate rtol = 1e-8
        # shocks as a K × T matrix give the same result
        Gm = hcat(shocks[1990.0], shocks[2000.0])
        rm = rotemberg_weights(ADH, :d_sh_empl_mfg, :d_tradeusch_pw, ADH_SH, Gm;
                               period=:year, covariates=ADH_CTRL, weights=:timepwt48)
        @test rm.table.alpha ≈ rw.table.alpha rtol = 1e-12
        @test occursin("aggregated over 2 periods", sprint(show, MIME"text/plain"(), rw))
        sh = ADH[shuffle(StableRNG(711), 1:nrow(ADH)), :]
        rs = rotemberg_weights(sh, :d_sh_empl_mfg, :d_tradeusch_pw, ADH_SH, shocks;
                               period=:year, covariates=ADH_CTRL, weights=:timepwt48)
        @test rs.estimate ≈ rw.estimate rtol = 1e-10
        @test rs.table.alpha ≈ rw.table.alpha rtol = 1e-8
    end

    @testset "overidentified 2SLS decomposition" begin
        rt = rotemberg_weights(ADH, :d_sh_empl_mfg, :d_tradeusch_pw, ADH_SH, nothing;
                               period=:year, covariates=ADH_CTRL, weights=:timepwt48,
                               estimator=:tsls)
        @test rt.estimator === :tsls
        @test rt.estimate ≈ IV_REF_ROT[("adh_tsls", "ivreg")] rtol = 1e-9
        for j in 1:nrow(rt.table)
            ind = string(rt.table.sector[j])[4:end]
            aref = IV_REF_ROT[("adh_tsls", "agg_alpha_$ind")]
            @test rt.table.alpha[j] ≈ aref atol = 1e-10
            @test rt.table.beta[j] ≈ IV_REF_ROT[("adh_tsls", "agg_beta_$ind")] rtol = 1e-7
        end
        # cross-section: the decomposition reproduces 2SLS with all shares
        rs = rotemberg_weights(SS_DF, :y, :d, SS_SH, nothing; covariates=[:x1],
                               estimator=:tsls)
        r2 = iv_regression(SS_DF, :y, :d, SS_SH; covariates=[:x1])
        @test rs.estimate ≈ coef(r2)[1] rtol = 1e-8
        @test sum(rs.table.alpha) ≈ 1.0
        @test all(ismissing, rs.table.shock)
    end

    @testset "panel with one period equals the cross-section; errors" begin
        g = SS_SHOCKS.shock
        df = copy(SS_DF)
        df.period = fill(1, nrow(df))
        a = rotemberg_weights(SS_DF, :y, :d, SS_SH, g; covariates=[:x1, :ssum])
        b = rotemberg_weights(df, :y, :d, SS_SH, Dict(1 => g); period=:period,
                              covariates=[:x1, :ssum])
        @test a.estimate ≈ b.estimate rtol = 1e-12
        @test a.table.alpha ≈ b.table.alpha rtol = 1e-10
        @test_throws ArgumentError rotemberg_weights(df, :y, :d, SS_SH, g;
                                                     period=:period)
        @test_throws ArgumentError rotemberg_weights(df, :y, :d, SS_SH, Dict(2 => g);
                                                     period=:period)
        @test_throws ArgumentError rotemberg_weights(SS_DF, :y, :d, SS_SH, nothing)
        @test_throws ArgumentError rotemberg_weights(SS_DF, :y, :d, SS_SH, g;
                                                     estimator=:liml)
    end
end
