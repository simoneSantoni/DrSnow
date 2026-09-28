using FixedEffectModels: reg, @formula, fe, Vcov

# Spatial panel with pre/post adoption and ring spillovers.
function _sv_ring_panel(rng; N=500, T=6, adopt=4, share=0.1, direct=2.0,
                        sc=(1.0, 0.5), st=(0.3, 0.0), dyn=false, pretrend=0.0, L=100.0)
    x = L .* rand(rng, N)
    y = L .* rand(rng, N)
    s = SpatialStructure(["c$(i)" for i in 1:N]; x=x, y=y)
    tr = rand(rng, N) .< share
    panel = DataFrame(id=repeat(s.ids, T), t=repeat(1:T; inner=N))
    ui = repeat(1:N, T)
    panel.d = Int.(tr[ui] .& (panel.t .>= adopt))
    ex = compute_exposure(panel, :d, s, RingExposure([5.0, 10.0]); unit=:id, time=:t)
    a = randn(rng, N)
    l = randn(rng, T)
    k = panel.t .- adopt
    mult = dyn ? (1 .+ 0.5 .* max.(k, 0)) : ones(nrow(panel))
    # units that will be near treated units (fixed for pretrend violations)
    near = compute_exposure(s, tr, RingExposure([5.0, 10.0]))
    nearu = (near.ring_0_5 .> 0) .& .!tr
    panel.y = a[ui] .+ l[panel.t] .+ mult .* (direct .* panel.d .+
              (1 .- panel.d) .* (sc[1] .* ex.ring_0_5 .+ sc[2] .* ex.ring_5_10) .+
              panel.d .* (st[1] .* ex.ring_0_5 .+ st[2] .* ex.ring_5_10)) .+
              pretrend .* nearu[ui] .* panel.t .+ 0.5 .* randn(rng, nrow(panel))
    return panel, s
end

@testset "Spillover DiD, event study and exposure regression" begin
    rings = RingExposure([5.0, 10.0])

    @testset "truth recovery and equality with a direct FixedEffectModels fit" begin
        panel, s = _sv_ring_panel(StableRNG(1); N=800)
        r = spillover_did(panel, :y, :d, s; unit=:id, time=:t, exposure=rings)
        @test coefnames(r) == ["direct", "spill_control:ring_0_5",
                               "spill_treated:ring_0_5", "spill_control:ring_5_10",
                               "spill_treated:ring_5_10"]
        truth = [2.0, 1.0, 0.3, 0.5, 0.0]
        @test all(abs.(coef(r) .- truth) .<= 4 .* stderror(r))
        # manual regression with exposure columns from compute_exposure
        ex = compute_exposure(panel, :d, s, rings; unit=:id, time=:t)
        df = copy(panel)
        df.a1 = ex.ring_0_5 .* (1 .- df.d)
        df.b1 = ex.ring_0_5 .* df.d
        df.a2 = ex.ring_5_10 .* (1 .- df.d)
        df.b2 = ex.ring_5_10 .* df.d
        m = reg(df, @formula(y ~ d + a1 + b1 + a2 + b2 + fe(id) + fe(t)),
                Vcov.cluster(:id))
        @test coef(r) ≈ coef(m)
        @test vcov(r) ≈ vcov(m)
        @test dof_residual(r) == dof_residual(m)
        # shuffled rows: identical
        perm = randperm(StableRNG(2), nrow(panel))
        r2 = spillover_did(panel[perm, :], :y, :d, s; unit=:id, time=:t, exposure=rings)
        @test coef(r2) ≈ coef(r) && vcov(r2) ≈ vcov(r)
        # pooled spillover terms, Conley variance, printing
        rp = spillover_did(panel, :y, :d, s; unit=:id, time=:t, exposure=rings,
                           split_by_treatment=false,
                           vcov=ConleyVcov(s; unit=:id, cutoff=10.0, time=:t))
        @test coefnames(rp) == ["direct", "spill:ring_0_5", "spill:ring_5_10"]
        @test occursin("Spillover DiD", sprint(show, MIME"text/plain"(), rp))
    end

    @testset "Monte Carlo coverage of spillover-DiD intervals" begin
        # Every coefficient must be identified by many units (clusters): with dense
        # treatment few treated units lack treated neighbours and cluster-robust
        # intervals for `direct` undercover (few-treated-clusters problem).
        reps = mc_reps(300, 40)
        truth = [2.0, 1.0, 0.0, 0.5, 0.0]
        cover = zeros(5)
        for rep in 1:reps
            panel, s = _sv_ring_panel(StableRNG(100 + rep); N=1000, L=200.0,
                                      st=(0.0, 0.0))
            r = spillover_did(panel, :y, :d, s; unit=:id, time=:t, exposure=rings)
            ci = confint(r)
            cover .+= (ci[:, 1] .<= truth .<= ci[:, 2])
        end
        @info "spillover_did 95% coverage" cover ./ reps reps
        @test all(cover ./ reps .>= 0.85)
    end

    @testset "identification errors" begin
        panel, s = _sv_ring_panel(StableRNG(3); N=200)
        # time-invariant treatment: absorbed by unit fixed effects
        p2 = copy(panel)
        tr = Dict(u => any(==(1), panel.d[panel.id .== u]) for u in unique(panel.id))
        p2.d = Int.([tr[u] for u in p2.id])
        @test_throws ArgumentError spillover_did(p2, :y, :d, s; unit=:id, time=:t,
                                                 exposure=rings)
        # rings so wide that nobody is a clean control after adoption
        @test_throws ArgumentError spillover_did(panel, :y, :d, s; unit=:id, time=:t,
                                                 exposure=RingExposure([200.0]))
        p3 = copy(panel)
        p3.direct = zeros(nrow(p3))
        @test_throws ArgumentError spillover_did(p3, :y, :d, s; unit=:id, time=:t,
                                                 exposure=rings)
        @test_throws ArgumentError spillover_did(panel[panel.id .!= "c1", :], :y, :d, s;
                                                 unit=:id, time=:t, exposure=rings)
        @test_throws ArgumentError spillover_did(panel, :y, :nope, s; unit=:id, time=:t,
                                                 exposure=rings)
        # an exposure column with no variation is reported as not identified
        s_far = SpatialStructure(s.ids; x=1000 .* s.coords[:, 1], y=1000 .* s.coords[:, 2])
        @test_throws ArgumentError spillover_did(panel, :y, :d, s_far; unit=:id, time=:t,
                                                 exposure=rings)
    end

    @testset "network spillover DiD with hop exposure" begin
        rng = StableRNG(4)
        N = 600
        T = 5
        A = zeros(N, N)
        for i in 1:N, j in (i + 1):N
            rand(rng) < 3 / N && (A[i, j] = A[j, i] = 1)
        end
        g = NetworkStructure(1:N, A)
        tr = rand(rng, N) .< 0.1
        panel = DataFrame(id=repeat(1:N, T), t=repeat(1:T; inner=N))
        panel.d = Int.(tr[panel.id] .& (panel.t .>= 3))
        hx = compute_exposure(panel, :d, g, HopExposure(2; stat=:nearest,
                                                        isolates=:zero);
                              unit=:id, time=:t)
        panel.y = randn(rng, N)[panel.id] .+ 0.2 .* panel.t .+ 1.5 .* panel.d .+
                  (1 .- panel.d) .* (0.8 .* hx.hop1 .+ 0.3 .* hx.hop2) .+
                  0.5 .* randn(rng, nrow(panel))
        r = spillover_did(panel, :y, :d, g; unit=:id, time=:t,
                          exposure=HopExposure(2; stat=:nearest, isolates=:zero),
                          vcov=NetworkHACVcov(g; unit=:id, bandwidth=2, time=:t))
        b = coef(r)
        @test abs(b[1] - 1.5) <= 4 * stderror(r)[1]
        @test abs(b[2] - 0.8) <= 4 * stderror(r)[2]
        @test abs(b[4] - 0.3) <= 4 * stderror(r)[4]
    end

    @testset "event study: dynamics, binned endpoints, pre-trend test" begin
        panel, s = _sv_ring_panel(StableRNG(5); N=800, T=8, adopt=5, dyn=true)
        es = spillover_event_study(panel, :y, :d, s; unit=:id, time=:t, exposure=rings,
                                   leads=3, lags=2)
        tab = es.table
        @test Set(tab.group) == Set(["treated", "exposed:ring_0_5", "exposed:ring_5_10"])
        # treated effects: 2 (1 + 0.5 k) for k = 0, 1 and binned k ≥ 2 (mean of 2, 3)
        tt = tab[tab.group .== "treated", :]
        @test abs(tt.estimate[tt.event_time .== 0][1] - 2.0) < 0.35
        @test abs(tt.estimate[tt.event_time .== 1][1] - 3.0) < 0.35
        @test abs(tt.estimate[tt.event_time .== 2][1] - 4.5) < 0.35
        @test all(abs.(tt.estimate[tt.event_time .< -1]) .< 0.35)   # no contamination
        r1 = tab[tab.group .== "exposed:ring_0_5", :]
        @test abs(r1.estimate[r1.event_time .== 0][1] - 1.0) < 0.35
        t = spillover_pretrend_test(es)
        @test t.dof[1] == 6 && 0 <= t.pvalue <= 1
        @test spillover_pretrend_test(es; group="treated").dof[1] == 2
        @test_throws ArgumentError spillover_pretrend_test(es; group="nope")
        @test length(coef(es)) == nrow(tab)
        @test occursin("clean controls", sprint(show, MIME"text/plain"(), es))
        # shuffled rows
        perm = randperm(StableRNG(6), nrow(panel))
        es2 = spillover_event_study(panel[perm, :], :y, :d, s; unit=:id, time=:t,
                                    exposure=rings, leads=3, lags=2)
        @test coef(es2) ≈ coef(es)
        # a pre-trend among units near treated units is detected
        pv, sv = _sv_ring_panel(StableRNG(7); N=800, T=8, adopt=5, pretrend=0.4)
        esv = spillover_event_study(pv, :y, :d, sv; unit=:id, time=:t, exposure=rings,
                                    leads=3, lags=2)
        @test spillover_pretrend_test(esv; group="exposed:ring_0_5").pvalue < 0.01
    end

    @testset "event study: pre-trend test size" begin
        reps = mc_reps(300, 40)
        rej = 0
        for rep in 1:reps
            panel, s = _sv_ring_panel(StableRNG(500 + rep); N=300, T=7, adopt=4)
            es = spillover_event_study(panel, :y, :d, s; unit=:id, time=:t,
                                       exposure=rings, leads=3, lags=2)
            rej += spillover_pretrend_test(es).pvalue <= 0.05
        end
        @info "spillover_pretrend_test size at 5%" rej / reps reps
        @test rej / reps <= 0.05 + 3 * sqrt(0.05 * 0.95 / reps)
    end

    @testset "event study errors" begin
        panel, s = _sv_ring_panel(StableRNG(8); N=200)
        p2 = copy(panel)
        k = findfirst(r -> r.d == 1 && r.t == 6, eachrow(p2))
        p2.d[k] = 0                                   # switches off
        @test_throws ArgumentError spillover_event_study(p2, :y, :d, s; unit=:id,
                                                         time=:t, exposure=rings)
        @test_throws ArgumentError spillover_event_study(panel, :y, :d, s; unit=:id,
                                                         time=:t, exposure=rings,
                                                         leads=1)
        p3 = copy(panel)
        p3.d .= 0
        @test_throws ArgumentError spillover_event_study(p3, :y, :d, s; unit=:id,
                                                         time=:t, exposure=rings)
    end

    @testset "cross-sectional exposure regression" begin
        rng = StableRNG(9)
        N = 800
        A = zeros(N, N)
        for i in 1:N, j in (i + 1):N
            rand(rng) < 5 / N && (A[i, j] = A[j, i] = 1)
        end
        ids = ["v$(i)" for i in 1:N]
        g = NetworkStructure(ids, A)
        z = rand(rng, N) .< 0.3
        ex = compute_exposure(g, z, NeighborExposure(:share; isolates=:zero))
        u = (A + I) * randn(rng, N) ./ 2
        df = DataFrame(id=ids, z=Int.(z), y=1.0 .+ 2.0 .* z .+
                       1.5 .* ex.share .* (1 .- z) .+ 0.5 .* ex.share .* z .+ u)
        r = exposure_regression(df, :y, :z, g; unit=:id,
                                exposure=NeighborExposure(:share; isolates=:zero),
                                vcov=NetworkHACVcov(g; unit=:id, bandwidth=2))
        @test all(abs.(coef(r) .- [2.0, 1.5, 0.5]) .<= 4 .* stderror(r))
        df2 = copy(df)
        df2.a = ex.share .* (1 .- df2.z)
        df2.b = ex.share .* df2.z
        m = reg(df2, @formula(y ~ z + a + b), NetworkHACVcov(g; unit=:id, bandwidth=2))
        @test coef(r) ≈ coef(m)[2:4] && vcov(r) ≈ vcov(m)[2:4, 2:4]
        rs = exposure_regression(df[randperm(rng, N), :], :y, :z, g; unit=:id,
                                 exposure=NeighborExposure(:share; isolates=:zero),
                                 vcov=NetworkHACVcov(g; unit=:id, bandwidth=2))
        @test coef(rs) ≈ coef(r) && vcov(rs) ≈ vcov(r)
        # isolates (missing exposure by default) are dropped and counted
        rm = exposure_regression(df, :y, :z, g; unit=:id)
        @test rm.details.dropped == count(==(0), vec(sum(A; dims=2)))
        @test_throws ArgumentError exposure_regression(df[2:end, :], :y, :z, g; unit=:id)
    end
end
