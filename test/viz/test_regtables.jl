@testset "RegressionTables.jl" begin
    f = VIZ_FIX
    rs = (f.did, f.iv, f.rd, f.dml, f.sdid)
    fmt(x) = @sprintf("%.3f", x)

    @testset "text table" begin
        t = RegressionTables.regtable(rs...; render=RegressionTables.AsciiTable())
        s = string(t)
        for r in rs
            # every coefficient and its standard error are printed (3 digits)
            for (b, se) in zip(coef(r), stderror(r))
                @test occursin(fmt(abs(b)), s)
                @test occursin("(" * fmt(se) * ")", s)
            end
            @test occursin(method_name(r), s)
        end
        @test occursin("(5)", s)
        @test occursin("RD effect", s)
        @test occursin("First-stage F", s)
        # outcome names are used as the dependent-variable header when recorded
        @test occursin(" y ", s)
    end

    @testset "LaTeX table" begin
        t = RegressionTables.regtable(rs...; render=RegressionTables.LatexTable())
        s = string(t)
        @test occursin("\\begin{tabular}", s) && occursin("\\end{tabular}", s)
        @test occursin(fmt(coef(f.sdid)[1]), s)
    end

    @testset "p-values, intervals and mixing with FixedEffectModels" begin
        # stars follow DrSnow's p-values (normal reference for RD)
        s = string(RegressionTables.regtable(f.rd; render=RegressionTables.AsciiTable()))
        @test occursin(fmt(coef(f.rd)[1]) * "***", s) == (pvalues(f.rd)[1] < 0.01)
        s = string(RegressionTables.regtable(f.did;
                                             below_statistic=RegressionTables.ConfInt,
                                             render=RegressionTables.AsciiTable()))
        ci = confint(f.did)
        @test occursin("(" * fmt(ci[1]) * ", " * fmt(ci[2]) * ")", s)
        m = f.did.details.model
        s = string(RegressionTables.regtable(m, f.did;
                                             render=RegressionTables.AsciiTable()))
        @test occursin("(2)", s)
        s = string(RegressionTables.regtable(m, m, f.did, f.iv, m;
                                             render=RegressionTables.AsciiTable()))
        @test occursin("(5)", s) && occursin("2SLS", s)
        # a result without a variance cannot be tabulated
        @test_throws ArgumentError RegressionTables.regtable(f.sdid_nose)
    end
end
