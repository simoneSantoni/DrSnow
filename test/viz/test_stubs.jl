@testset "plot stubs without a Makie backend" begin
    if Base.get_extension(DrSnow, :DrSnowMakieExt) === nothing
        for fn in (plot_event_study, plot_coefficients, plot_rd, plot_synth,
                   plot_randomization_distribution, plot_bacon, plot_gates, plot_cate,
                   plot_spillover_rings, plot_confidence_set, plot_event_study!,
                   plot_synth!, plot_variable_importance, plot_variable_importance!,
                   plot_rate, plot_rate!, plot_assignment_probabilities,
                   plot_excursion_effect)
            err = try
                fn(VIZ_FIX.did)
                nothing
            catch e
                e
            end
            @test err isa ErrorException
            @test occursin("using CairoMakie", sprint(showerror, err))
        end
    else
        @info "Makie extension already loaded; skipping the no-backend stub test"
    end
end
