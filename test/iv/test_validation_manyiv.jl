# Validation of k-class / jackknife IV against R references
# (test/validation/iv/make_reference_manyiv.R: ivmodel, AER, sandwich, and formulas
# coded from the papers with explicit n × n matrices).

const IV_REF_MANY = let r = iv_read_csv(joinpath(IV_VALDIR, "reference_manyiv.csv"))
    d = Dict{Tuple{String,String},Float64}()
    for row in eachrow(r)
        d[(row.case, row.quantity)] = row.value
    end
    d
end
manyref(case, q) = IV_REF_MANY[(case, q)]

const CARD_MANY = let df = copy(CARD)
    for r in 1:9
        df[!, Symbol("z4r", r)] = df.nearc4 .* (df.region .== r)
        df[!, Symbol("z2r", r)] = df.nearc2 .* (df.region .== r)
    end
    df.w = df.weight ./ mean(df.weight)
    df
end
const MANY_Z = vcat([Symbol("z4r", r) for r in 1:9], [Symbol("z2r", r) for r in 1:9])

@testset "Validation: k-class and jackknife IV against R" begin
    @testset "LIML / Fuller vs ivmodel (Card, 2 instruments)" begin
        n = nrow(CARD)
        K = 1 + 1 + length(CARD_CTRL)
        G = length(unique(CARD.region))
        for (nm, kw) in (("liml", (method=:liml,)),
                         ("fuller1", (method=:fuller,)),
                         ("fuller4", (method=:fuller, fuller_alpha=4)))
            r = kclass_iv(CARD, :lwage, :educ, [:nearc4, :nearc2];
                          covariates=CARD_CTRL, vcov=Vcov.simple(), kw...)
            @test coef(r)[1] ≈ manyref("card_kc", nm * "_coef") rtol = 1e-8
            @test r.kappa ≈ manyref("card_kc", nm * "_kappa") rtol = 1e-10
            @test stderror(r)[1] ≈ manyref("card_kc", nm * "_se_iid") rtol = 1e-7
            rh = kclass_iv(CARD, :lwage, :educ, [:nearc4, :nearc2];
                           covariates=CARD_CTRL, kw...)
            @test stderror(rh)[1] * sqrt((n - K) / n) ≈
                  manyref("card_kc", nm * "_se_hc0") rtol = 1e-7
            rc = kclass_iv(CARD, :lwage, :educ, [:nearc4, :nearc2];
                           covariates=CARD_CTRL, cluster=:region, kw...)
            fac = (n - 1) / (n - K) * G / (G - 1)
            @test isapprox(stderror(rc)[1] / sqrt(fac), manyref("card_kc", nm * "_se_cr0");
                           rtol=1e-7)
            @test dof_residual(rc) == G - 1
        end
    end

    for (case, w) in (("card_many", nothing), ("card_many_w", :w))
        @testset "many instruments with fixed effects ($case)" begin
            common = (covariates=CARD_CTRL, fe=[:region], weights=w)
            for nm in ("liml", "fuller1")
                m = nm == "liml" ? :liml : :fuller
                r = kclass_iv(CARD_MANY, :lwage, :educ, MANY_Z; method=m,
                              vcov=Vcov.simple(), common...)
                @test coef(r)[1] ≈ manyref(case, nm * "_coef") rtol = 1e-7
                @test r.kappa ≈ manyref(case, nm * "_kappa") rtol = 1e-10
                @test stderror(r)[1] ≈ manyref(case, nm * "_se_iid") rtol = 1e-6
                rb = kclass_iv(CARD_MANY, :lwage, :educ, MANY_Z; method=m, se=:bekker,
                               common...)
                @test coef(rb)[1] ≈ coef(r)[1]
                @test stderror(rb)[1] ≈ manyref(case, nm * "_se_bekker") rtol = 1e-6
            end
            if w === nothing
                @test isapprox(manyref(case, "ivmodel_liml_coef"),
                               manyref(case, "liml_coef"); rtol=1e-9)
            end
            for nm in ("hlim", "hful")
                r = kclass_iv(CARD_MANY, :lwage, :educ, MANY_Z; method=Symbol(nm),
                              common...)
                @test coef(r)[1] ≈ manyref(case, nm * "_coef") rtol = 1e-7
                @test r.kappa ≈ manyref(case, nm * "_alpha") rtol = 1e-7
                @test stderror(r)[1] ≈ manyref(case, nm * "_se") rtol = 1e-6
            end
            for nm in ("jive1", "jive2")
                r = jive(CARD_MANY, :lwage, :educ, MANY_Z; method=Symbol(nm), common...)
                @test coef(r)[1] ≈ manyref(case, nm * "_coef") rtol = 1e-7
                @test stderror(r)[1] ≈ manyref(case, nm * "_se_hc1") rtol = 1e-6
                ri = jive(CARD_MANY, :lwage, :educ, MANY_Z; method=Symbol(nm),
                          vcov=Vcov.simple(), common...)
                @test stderror(ri)[1] ≈ manyref(case, nm * "_se_iid") rtol = 1e-6
            end
            r = jive(CARD_MANY, :lwage, :educ, MANY_Z; method=:ujive, common...)
            @test coef(r)[1] ≈ manyref(case, "ujive_coef") rtol = 1e-7
            @test stderror(r)[1] ≈ manyref(case, "ujive_se_hc1") rtol = 1e-6
            rm = jive(CARD_MANY, :lwage, :educ, MANY_Z; method=:ujive, se=:many_robust,
                      common...)
            @test stderror(rm)[1] ≈ manyref(case, "ujive_se_many") rtol = 1e-6
            for b0 in (0.0, 0.1)
                t = weak_iv_test(r; method=:jackknife_ar, beta0=b0)
                key = "jar_stat_b" * string(b0 == 0 ? 0 : b0)
                @test t.statistic ≈ manyref(case, key) rtol = 1e-7
            end
        end
    end
end
