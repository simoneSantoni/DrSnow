# Synthetic control area: panel preparation, synthetic DiD, classic SC, augmented SC,
# conformal inference, matrix completion, TreatmentPanels interop.

include("helpers.jl")

@testset "panel" begin
    include("test_panel.jl")
end
@testset "solvers" begin
    include("test_solvers.jl")
end
@testset "synthetic DiD" begin
    include("test_sdid.jl")
end
@testset "classic synthetic control" begin
    include("test_adh.jl")
end
@testset "augmented SC and conformal" begin
    include("test_ascm.jl")
end
@testset "matrix completion" begin
    include("test_mcnnm.jl")
end
@testset "reference implementations (R)" begin
    include("test_reference.jl")
end
@testset "Monte Carlo" begin
    include("test_montecarlo.jl")
end
@testset "TreatmentPanels interop" begin
    include("test_interop.jl")
end
