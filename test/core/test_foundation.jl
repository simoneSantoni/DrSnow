@testset "Foundation" begin
    @testset "make_formula never parses strings" begin
        df = DataFrame(Symbol("log wage") => randn(StableRNG(1), 50),
                       Symbol("d; println(1)") => rand(StableRNG(2), 0:1, 50),
                       :g => repeat(1:10, 5))
        f = make_formula(Symbol("log wage"), [Symbol("d; println(1)")]; fe=[:g])
        m = DrSnow.FixedEffectModels.reg(df, f)
        @test length(coef(m)) == 1
        @test_throws ArgumentError make_formula(:y, [:x]; endogenous=[:d])
        @test_throws ArgumentError DrSnow.require_columns(df, [:nope])
    end

    @testset "critical values and p-values" begin
        @test critical_value(0.95) ≈ 1.959963984540054
        @test critical_value(0.95, 9) ≈ 2.2621571627409915
        @test two_sided_pvalue(1.959963984540054) ≈ 0.05
        @test isnan(two_sided_pvalue(NaN))
    end

    @testset "wald_test uses full covariance" begin
        b = [1.0, 1.0]
        V = [1.0 0.9; 0.9 1.0]
        w = wald_test(b, V)
        @test w.chi2 ≈ dot(b, V \ b)
        @test w.dof1 == 2
        wF = wald_test(b, V; dof=20)
        @test wF.statistic ≈ w.chi2 / 2
    end

    @testset "permutation_pvalue" begin
        @test permutation_pvalue(10.0, randn(StableRNG(3), 99)) == 0.01
        @test permutation_pvalue(0.0, [1.0, -1.0, 2.0]) == 1.0
    end

    @testset "DiagnosticTest never reports NaN as pass" begin
        t = DiagnosticTest("demo", "nothing happens", NaN, NaN)
        @test_throws ErrorException rejects(t)
        s = sprint(show, MIME"text/plain"(), DiagnosticTest("d", "H", 0.1, 0.9))
        @test occursin("not evidence", s)
    end
end
