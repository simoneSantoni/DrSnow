import TreatmentPanels

@testset "extension is loaded" begin
    @test Base.get_extension(DrSnow, :DrSnowTreatmentPanelsExt) !== nothing
end

@testset "BalancedPanel <-> SynthPanel" begin
    df = load_prop99()
    bp = TreatmentPanels.BalancedPanel(select(df, :State, :Year, :PacksPerCapita),
                                       "California" => 1989; id_var=:State,
                                       t_var=:Year, outcome_var=:PacksPerCapita)
    p = synth_panel(bp)
    q = synth_panel(df, :PacksPerCapita, :treated, :State, :Year)
    @test p.Y == q.Y && p.units == q.units && p.adoption == q.adoption
    r1 = synthetic_did(bp; se_method=:none)
    r2 = synthetic_did(q; se_method=:none)
    @test r1.att == r2.att
    @test augmented_synthetic_control(bp; se_method=:none).att ≈
          augmented_synthetic_control(q; se_method=:none).att
    @test synthetic_control(bp; placebo=false).att ≈ synthetic_control(q; placebo=false).att
    @test matrix_completion(bp; lambda=0.02, se_method=:none).att ≈
          matrix_completion(q; lambda=0.02, se_method=:none).att
    back = TreatmentPanels.BalancedPanel(q)
    @test back.Y == bp.Y
    @test back.W == bp.W
    # several treated units with the same adoption date
    df2 = copy(df)
    df2.treated[(df2.State .== "Utah") .& (df2.Year .>= 1989)] .= 1
    q2 = synth_panel(df2, :PacksPerCapita, :treated, :State, :Year)
    b2 = TreatmentPanels.BalancedPanel(q2)
    @test synth_panel(b2).Y == q2.Y
end
