@testset "Data Structures" begin
    
    @testset "TreatmentPanel construction" begin
        # Create simple test data
        data = DataFrame(
            unit = repeat(1:10, inner=5),
            time = repeat(1:5, outer=10),
            outcome = randn(50),
            treatment = repeat([0, 0, 0, 1, 1], outer=10),
            covar1 = randn(50)
        )
        
        # Test successful construction
        panel = TreatmentPanel(
            data, :outcome, :treatment, :unit, :time, [:covar1]
        )
        
        @test panel.outcome == :outcome
        @test panel.treatment == :treatment
        @test panel.unit_id == :unit
        @test panel.time == :time
        @test panel.covariates == [:covar1]
        
        # Test missing column error
        @test_throws ArgumentError TreatmentPanel(
            data, :missing_col, :treatment, :unit, :time
        )
        
        # Test missing covariate error
        @test_throws ArgumentError TreatmentPanel(
            data, :outcome, :treatment, :unit, :time, [:missing_covar]
        )
    end
    
    @testset "validate_panel" begin
        # Create test data with issues
        data = DataFrame(
            unit = [1, 1, 2, 2],
            time = [1, 2, 1, 2],
            outcome = [1.0, 2.0, 3.0, 4.0],
            treatment = [0, 1, 0, 2]  # Invalid treatment value
        )
        
        panel = TreatmentPanel(data, :outcome, :treatment, :unit, :time)
        warnings = validate_panel(panel)
        
        @test !isempty(warnings)
        @test any(contains.(warnings, "binary"))
    end
    
    @testset "preprocess_panel" begin
        # Create data with missing values
        data = DataFrame(
            unit = [1, 1, 2, 2, 3, 3],
            time = [1, 2, 1, 2, 1, 2],
            outcome = [1.0, 2.0, missing, 4.0, 5.0, 6.0],
            treatment = [0, 0, 1, 1, 0, 1]
        )
        
        # Test with drop_missing=true
        panel = preprocess_panel(data, :outcome, :treatment, :unit, :time, 
                                drop_missing=true)
        @test nrow(panel.data) == 5  # One row dropped
        
        # Test sorting
        @test issorted(panel.data, [:unit, :time])
    end
    
    @testset "DiDEstimate display" begin
        # DiDEstimate moved to src/did (result interface with full vcov).
        est = DiDEstimate([5.0], fill(1.0, 1, 1), ["ATT"], 100, 24.0, 25, 25, 75, 10,
                          "Test Method", "ATT", NamedTuple())
        output = sprint(show, MIME"text/plain"(), est)
        @test contains(output, "5.0")
        @test contains(output, "Test Method")
        @test confint(est)[1] ≈ 5.0 - critical_value(0.95, 24)
    end
end
