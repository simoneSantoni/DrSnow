@testset "tidy, glance and the Tables.jl interface" begin
    f = VIZ_FIX
    cols = ["term", "estimate", "std_error", "statistic", "p_value", "conf_low",
            "conf_high"]

    @testset "tidy matches the StatsAPI accessors" begin
        for r in (f.did, f.iv, f.rd, f.dml, f.sdid, f.es_cs, f.ring_did, f.cs)
            t = tidy(r)
            @test names(t) == cols
            @test t.term == coefnames(r)
            @test t.estimate == coef(r)
            @test t.std_error ≈ stderror(r)
            @test t.statistic ≈ coef(r) ./ stderror(r)
            @test t.p_value ≈ pvalues(r)
            ci = confint(r; level=0.9)
            t90 = tidy(r; level=0.9)
            @test t90.conf_low ≈ ci[:, 1] && t90.conf_high ≈ ci[:, 2]
        end
        # t reference with G - 1 df under clustering
        t = tidy(f.did)
        c = critical_value(0.95, dof_residual(f.did))
        @test t.conf_high[1] ≈ t.estimate[1] + c * t.std_error[1]
        # estimator-specific confint options are forwarded
        tu = tidy(f.es_cs; uniform=true)
        tp = tidy(f.es_cs)
        @test all(tu.conf_high .- tu.conf_low .>= tp.conf_high .- tp.conf_low .- 1e-12)
        @test_throws ArgumentError tidy(f.did; level=1.2)
    end

    @testset "results without a variance" begin
        t = tidy(f.sdid_nose)
        @test t.estimate == coef(f.sdid_nose)
        @test all(ismissing, t.std_error) && all(ismissing, t.conf_low)
        @test all(ismissing, tidy(f.mc).p_value)
    end

    @testset "stacked results" begin
        t = tidy([f.did, f.sdid, f.iv]; names=["a", "b", "c"])
        @test names(t)[1] == "model"
        @test t.model == vcat("a", "b", fill("c", length(coef(f.iv))))
        @test nrow(t) == 2 + length(coef(f.iv))
        @test tidy([f.did]).model == [method_name(f.did)]
        @test nrow(tidy([f.did, f.sdid_nose])) == 2
        @test_throws DimensionMismatch tidy([f.did, f.sdid]; names=["a"])
        @test_throws ArgumentError tidy([f.did, 1.0])
    end

    @testset "DiagnosticTest" begin
        pt = pre_trend_test(f.es_cs)
        t = tidy(pt)
        @test nrow(t) == 1
        @test names(t) == ["test", "null", "statistic", "dof", "p_value", "method"]
        @test t.p_value[1] == pvalue(pt) && t.statistic[1] == pt.statistic
        @test t.dof[1] == join(string.(pt.dof), ", ")
        @test DataFrame(pt) == t
    end

    @testset "glance" begin
        g = glance(f.did)
        @test nrow(g) == 1
        @test names(g) == ["method", "estimand", "nobs", "n_coef", "dof_residual",
                           "n_clusters"]
        @test g.nobs[1] == nobs(f.did) && g.n_coef[1] == 1
        @test g.n_clusters[1] == f.did.n_clusters
        @test g.dof_residual[1] == dof_residual(f.did)
        @test ismissing(glance(f.dml).n_clusters[1])       # not clustered
        @test ismissing(glance(f.sdid).n_clusters[1])      # no cluster field
        @test isinf(glance(f.rd).dof_residual[1])
        ivc = iv_regression(viz_iv_data(StableRNG(3)) |>
                            d -> (d.g = repeat(1:25; inner=20); d), :y, :d, :z;
                            cluster=:g)
        @test glance(ivc).n_clusters[1] == 25
        @test ismissing(glance(f.iv).n_clusters[1])
        @test nrow(vcat(glance.([f.did, f.iv, f.rd])...)) == 3
    end

    @testset "Tables.jl" begin
        @test Tables.istable(f.did) && Tables.istable(typeof(f.es_cs))
        @test Tables.columnaccess(f.did)
        @test DataFrame(f.iv) == tidy(f.iv)
        ct = Tables.columntable(f.es_cs)
        @test ct.estimate == coef(f.es_cs)
        @test length(Tables.rowtable(f.iv)) == length(coef(f.iv))
        @test Tables.istable(pre_trend_test(f.es_cs))
    end
end
