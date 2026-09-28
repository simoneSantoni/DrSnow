# Tests of dml_did_multi: staggered DML difference-in-differences.

ml_didm_att(g, t) = t >= g ? 1.0 + 0.5 * (t - g) + 0.1 * (g - 3) : 0.0

"""
Balanced staggered panel with cohorts {3, 4, 5, never} drawn from a multinomial
logit in (x1, x2) and covariate-specific linear trends, so conditional (but not
unconditional) parallel trends hold; ATT(g,t) = `ml_didm_att(g, t)`.
"""
function ml_didm_panel(rng, N; T=5)
    x1 = randn(rng, N)
    x2 = randn(rng, N)
    G = zeros(Int, N)
    for i in 1:N
        s = [0.0, 0.5 * x1[i] - 0.3 * x2[i], 0.2 + 0.4 * x2[i], -0.1 + 0.6 * x1[i]]
        p = cumsum(exp.(s) ./ sum(exp.(s)))
        G[i] = (0, 3, 4, 5)[min(4, searchsortedfirst(p, rand(rng)))]
    end
    α = 0.5 .* x1 .+ randn(rng, N)
    rows = [(id=i, t=t, g=G[i], x1=x1[i], x2=x2[i],
             y=α[i] + 0.2t + t * (0.6 * x1[i] - 0.4 * x2[i]) +
               (G[i] > 0 ? ml_didm_att(G[i], t) : 0.0) + randn(rng))
            for i in 1:N for t in 1:T]
    df = DataFrame(rows)
    df.d = Int.((df.g .> 0) .& (df.t .>= df.g))
    return df
end

"""`r` with its ATT(g,t) replaced by `b` (to aggregate true effects with the same
sample weights as the estimates)."""
ml_didm_with_coef(r, b) =
    CallawaySantAnnaEstimate((f === :coef ? b : getfield(r, f)
                              for f in fieldnames(CallawaySantAnnaEstimate))...)

@testset "DML staggered DiD (dml_did_multi)" begin
    vdir = joinpath(@__DIR__, "..", "validation", "ml")
    lin = (outcome_learner=ML_OLS, propensity_learner=ML_LOGIT)

    @testset "DoubleMLDIDMulti parity (Python reference)" begin
        ref = ml_read_csv(joinpath(vdir, "did_multi_reference.csv"))
        panel = ml_read_csv(joinpath(vdir, "did_multi_panel.csv"))
        pf = ml_read_csv(joinpath(vdir, "did_multi_panel_folds.csv"))
        panel.fold = Int.(pf.fold[Int.(panel.id)])
        rcs = ml_read_csv(joinpath(vdir, "did_multi_rcs.csv"))
        rf = ml_read_csv(joinpath(vdir, "did_multi_rcs_folds.csv"))
        rcs.fold = Int.(rf.fold[Int.(rcs.id)])
        kw = (lin..., folds=:fold, bootstrap=false)
        xp = [:x1, :x2, :x3]
        fits = [
            "panel_never" => dml_did_multi(panel, :y, FirstTreated(:g), :id, :t;
                                           covariates=xp, kw...),
            "panel_notyet" => dml_did_multi(panel, :y, FirstTreated(:g), :id, :t;
                                            covariates=xp,
                                            control_group=:not_yet_treated, kw...),
            "panel_notyet_antic1" => dml_did_multi(panel, :y, FirstTreated(:g), :id, :t;
                                                   covariates=xp, anticipation=1,
                                                   control_group=:not_yet_treated,
                                                   kw...),
            "rcs_never" => dml_did_multi(rcs, :y, FirstTreated(:g), nothing, :t;
                                         covariates=[:x1, :x2], kw...),
            "rcs_notyet" => dml_did_multi(rcs, :y, FirstTreated(:g), nothing, :t;
                                          covariates=[:x1, :x2],
                                          control_group=:not_yet_treated, kw...)]
        for (case, r) in fits
            rr = ref[(ref.case .== case) .& (ref.kind .== "att_gt"), :]
            nmatch = 0
            for k in eachindex(coef(r))
                row = findfirst(i -> rr.g[i] == r.groups[k] && rr.t_eval[i] == r.times[k] &&
                                     rr.t_pre[i] == r.bases[k], 1:nrow(rr))
                row === nothing && continue
                nmatch += 1
                # sklearn's logistic solver stops at tol = 1e-12
                @test coef(r)[k] ≈ rr.coef[row] atol = 1e-7
                @test stderror(r)[k] ≈ rr.se_hajek[row] rtol = 1e-6
                # DoubleML's linear score (normalizing constant treated as known)
                @test stderror(r)[k] ≈ rr.se[row] rtol = 0.05
            end
            # with anticipation DoubleML's pre-treatment cells use long differences
            @test nmatch == (case == "panel_notyet_antic1" ? 6 : 12)
            for (kind, typ) in (("group", :group), ("time", :calendar))
                a = aggregate_att(r, typ; bootstrap=false)
                ra = ref[(ref.case .== case) .& (ref.kind .== kind), :]
                refc = vcat(ra.coef[end], ra.coef[1:(end - 1)])   # overall first
                @test coef(a) ≈ refc atol = 1e-7
                @test stderror(a) ≈ vcat(ra.se[end], ra.se[1:(end - 1)]) rtol = 0.05
            end
            case == "panel_notyet_antic1" && continue
            es = aggregate_att(r, :dynamic; bootstrap=false)
            ra = ref[(ref.case .== case) .& (ref.kind .== "eventstudy"), :]
            @test coef(es) ≈ ra.coef[1:(end - 1)] atol = 1e-7
            @test stderror(es) ≈ ra.se[1:(end - 1)] rtol = 0.05
            @test coef(es.details.overall)[1] ≈ ra.coef[end] atol = 1e-7
        end
    end

    @testset "reproduces did_callaway_santanna (DR) without cross-fitting" begin
        mp = ml_read_csv(joinpath(@__DIR__, "..", "validation", "did", "mpdta.csv"))
        for cg in (:never_treated, :not_yet_treated), unit in (:countyreal, nothing)
            cs = did_callaway_santanna(mp, :lemp, FirstTreated(:first_treat), unit, :year;
                                       covariates=[:lpop], control_group=cg,
                                       bootstrap=false)
            r = dml_did_multi(mp, :lemp, FirstTreated(:first_treat), unit, :year;
                              covariates=[:lpop], control_group=cg, lin...,
                              crossfit=false, trim=0.0, bootstrap=false)
            @test r.groups == cs.groups && r.times == cs.times && r.bases == cs.bases
            @test coef(r) ≈ coef(cs) atol = 1e-10
            # DRDID's influence function adds first-stage estimation terms
            @test stderror(r) ≈ stderror(cs) rtol = 0.03
            a1 = aggregate_att(r, :dynamic; bootstrap=false)
            a2 = aggregate_att(cs, :dynamic; bootstrap=false)
            @test coef(a1) ≈ coef(a2) atol = 1e-10
        end
        # anticipation and universal base period follow the same cells
        cs = did_callaway_santanna(mp, :lemp, FirstTreated(:first_treat), :countyreal,
                                   :year; covariates=[:lpop], anticipation=1,
                                   base_period=:universal, bootstrap=false)
        r = dml_did_multi(mp, :lemp, FirstTreated(:first_treat), :countyreal, :year;
                          covariates=[:lpop], anticipation=1, base_period=:universal,
                          lin..., crossfit=false, trim=0.0, bootstrap=false)
        @test coef(r) ≈ coef(cs) atol = 1e-10
        # cross-fitted estimates are close to the full-sample ones
        rc = dml_did_multi(mp, :lemp, FirstTreated(:first_treat), :countyreal, :year;
                           covariates=[:lpop], lin..., rng=StableRNG(3), bootstrap=false)
        cs0 = did_callaway_santanna(mp, :lemp, FirstTreated(:first_treat), :countyreal,
                                    :year; covariates=[:lpop], bootstrap=false)
        @test all(abs.(coef(rc) .- coef(cs0)) .< 0.5 .* stderror(cs0))
    end

    @testset "interface, aggregation and downstream tools" begin
        df = ml_didm_panel(StableRNG(31), 600)
        r = dml_did_multi(df, :y, :d, :id, :t; covariates=[:x1, :x2], lin...,
                          n_rep=2, rng=StableRNG(1), biters=299)
        @test r isa CallawaySantAnnaEstimate
        @test r.settings.method === :dml && r.settings.n_rep == 2
        @test size(r.settings.all_coef) == (length(coef(r)), 2)
        @test coef(r) ≈ vec(mean(r.settings.all_coef; dims=2))
        @test size(r.settings.folds) == (600, 2)
        @test names(r.settings.nuisance_loss) == ["group", "time", "ml_g0", "ml_m"]
        @test length(r.supt_draws) == 299
        @test nobs(r) == 600 * 5
        @test occursin("dml", method_name(r))
        post = findall(r.times .>= r.groups)
        @test all(abs.(coef(r)[post] .- ml_didm_att.(r.groups[post], r.times[post])) .<
                  4 .* stderror(r)[post])
        @test pre_trend_test(r).pvalue > 0.001
        ci = confint(r; uniform=true)
        @test all(ci[:, 2] .- ci[:, 1] .> 2 .* 1.95 .* stderror(r))
        es = aggregate_att(r, :dynamic; rng=StableRNG(2))
        @test es isa EventStudyEstimate && relative_periods(es) == -3:2
        @test confint(es; uniform=true) isa Matrix
        @test_throws ArgumentError honest_did(es; restriction=:relative_magnitudes)
        ru = dml_did_multi(df, :y, :d, :id, :t; covariates=[:x1, :x2], lin...,
                           base_period=:universal, rng=StableRNG(1), bootstrap=false)
        esu = aggregate_att(ru, :dynamic; bootstrap=false)
        @test esu.reference == [-1] && relative_periods(esu) == [-4, -3, -2, 0, 1, 2]
        hd = honest_did(esu; restriction=:relative_magnitudes, M=[0.0, 0.5])
        @test hd isa HonestDiDResult
        for typ in (:simple, :group, :calendar)
            @test aggregate_att(r, typ; bootstrap=false) isa AggregatedATT
        end
        @test tidy(r) isa DataFrame
        # FirstTreated coding gives the same result as the 0/1 indicator
        r2 = dml_did_multi(df, :y, FirstTreated(:g), :id, :t; covariates=[:x1, :x2],
                           lin..., n_rep=2, rng=StableRNG(1), biters=299)
        @test coef(r2) ≈ coef(r) rtol = 1e-12
        # results do not depend on the row order of the long panel
        perm = randperm(StableRNG(4), nrow(df))
        r3 = dml_did_multi(df[perm, :], :y, :d, :id, :t; covariates=[:x1, :x2], lin...,
                           n_rep=2, rng=StableRNG(1), biters=299)
        @test coef(r3) ≈ coef(r) rtol = 1e-10
        @test vcov(r3) ≈ vcov(r) rtol = 1e-10
        # repeated cross-sections are row-order invariant as well
        rc = df[df.t .== rand(StableRNG(5), 1:5, 600)[df.id], :]
        rc.obs = 1:nrow(rc)
        a = dml_did_multi(rc, :y, FirstTreated(:g), nothing, :t; covariates=[:x1, :x2],
                          lin..., rng=StableRNG(6), bootstrap=false, n_folds=3)
        pr = randperm(StableRNG(7), nrow(rc))
        folds = a.settings.folds[:, 1]
        b = dml_did_multi(rc[pr, :], :y, FirstTreated(:g), nothing, :t;
                          covariates=[:x1, :x2], lin..., folds=folds[pr],
                          bootstrap=false)
        @test coef(b) ≈ coef(a) rtol = 1e-10
        @test stderror(b) ≈ stderror(a) rtol = 1e-10
        @test !a.settings.panel && nobs(a) == nrow(rc)
        @test names(a.settings.nuisance_loss)[3:end] ==
              ["ml_g_d0_t0", "ml_g_d0_t1", "ml_g_d1_t0", "ml_g_d1_t1", "ml_m"]
    end

    @testset "clustering, flexible learners and errors" begin
        df = ml_didm_panel(StableRNG(41), 400)
        df.state = mod1.(df.id, 40)
        r = dml_did_multi(df, :y, :d, :id, :t; covariates=[:x1, :x2], lin...,
                          cluster=:state, rng=StableRNG(1), bootstrap=false)
        @test r.n_clusters == 40
        F = r.settings.folds[:, 1]
        st = mod1.(1:400, 40)
        @test all(length(unique(F[st .== s])) == 1 for s in 1:40)
        r0 = dml_did_multi(df, :y, :d, :id, :t; covariates=[:x1, :x2], lin...,
                           folds=F, bootstrap=false)
        @test coef(r0) ≈ coef(r) rtol = 1e-12
        @test all(stderror(r) .!= stderror(r0))
        # machine-learning nuisances
        rf = dml_did_multi(df, :y, :d, :id, :t; covariates=[:x1, :x2],
                           outcome_learner=ForestLearner(num_trees=100),
                           propensity_learner=LogisticLearner(), n_folds=3,
                           rng=StableRNG(2), bootstrap=false)
        post = findall(rf.times .>= rf.groups)
        @test all(abs.(coef(rf)[post] .- ml_didm_att.(rf.groups[post], rf.times[post])) .<
                  4.5 .* stderror(rf)[post])
        # no covariates: unconditional DR = simple DiD comparisons
        rn = dml_did_multi(df, :y, :d, :id, :t; lin..., crossfit=false, trim=0.0,
                           bootstrap=false)
        cn = did_callaway_santanna(df, :y, :d, :id, :t; bootstrap=false)
        @test coef(rn) ≈ coef(cn) atol = 1e-10
        # errors
        bad = copy(df)
        bad.d[(bad.id .== 1)] .= [0, 1, 0, 1, 0]
        @test_throws ArgumentError dml_did_multi(bad, :y, :d, :id, :t; lin...)
        @test_throws ArgumentError dml_did_multi(df, :y, :d, :id, :t; lin...,
                                                 control_group=:all)
        @test_throws ArgumentError dml_did_multi(df, :y, :d, :id, :t; lin...,
                                                 base_period=:fixed)
        @test_throws ArgumentError dml_did_multi(df, :y, :d, :id, :t; lin..., trim=0.6)
        @test_throws ArgumentError dml_did_multi(df, :y, :d, :id, :t; lin..., n_folds=1)
        @test_throws ArgumentError dml_did_multi(df, :y, :d, nothing, :t; lin...)
        @test_throws ArgumentError dml_did_multi(df, :y, :d, :id, :t; lin...,
                                                 covariates=[:nope])
        @test_throws DimensionMismatch dml_did_multi(df, :y, :d, :id, :t; lin...,
                                                     folds=[1, 2, 1])
        bf = copy(df)
        bf.f = mod1.(1:nrow(bf), 2)
        @test_throws ArgumentError dml_did_multi(bf, :y, :d, :id, :t; lin..., folds=:f)
        bc = copy(df)
        bc.c = 1:nrow(bc)
        @test_throws ArgumentError dml_did_multi(bc, :y, :d, :id, :t; lin..., cluster=:c)
        nx = copy(df)
        nx.x1 = Vector{Union{Missing,Float64}}(nx.x1)
        nx.x1[3] = NaN
        @test_throws ArgumentError dml_did_multi(nx, :y, :d, :id, :t; lin...,
                                                 covariates=[:x1])
        # a cohort with a single unit leaves training folds without treated units
        tiny = copy(df)
        ids5 = sort(unique(tiny.id[tiny.g .== 5]))
        tiny.g[in.(tiny.id, Ref(Set(ids5[2:end])))] .= 0
        err = try
            dml_did_multi(tiny, :y, FirstTreated(:g), :id, :t; covariates=[:x1, :x2],
                          lin..., rng=StableRNG(9))
        catch e
            e
        end
        @test err isa ArgumentError && occursin("ATT(g=5", err.msg)
    end

    @testset "Monte Carlo coverage" begin
        reps = mc_reps(400, 60)
        cover_cells = 0
        ncells = 0
        cover_simple = 0
        cover_band = 0
        est = Float64[]
        se = Float64[]
        for rep in 1:reps
            rng = StableRNG(70_000 + rep)
            df = ml_didm_panel(rng, 500)
            r = dml_did_multi(df, :y, :d, :id, :t; covariates=[:x1, :x2], lin...,
                              rng=rng, biters=499)
            truth = ml_didm_att.(r.groups, r.times)
            ci = confint(r)
            cover_cells += count((ci[:, 1] .<= truth) .& (truth .<= ci[:, 2]))
            ncells += length(truth)
            s = aggregate_att(r, :simple; bootstrap=false)
            rt = ml_didm_with_coef(r, truth)
            θ = coef(aggregate_att(rt, :simple; bootstrap=false))[1]
            cs = confint(s)
            cover_simple += cs[1, 1] <= θ <= cs[1, 2]
            push!(est, coef(s)[1] - θ)
            push!(se, stderror(s)[1])
            es = aggregate_att(r, :dynamic; rng=rng)
            cb = confint(es; uniform=true)
            te = coef(aggregate_att(rt, :dynamic; bootstrap=false))
            cover_band += all((cb[:, 1] .<= te) .& (te .<= cb[:, 2]))
        end
        tol = ml_cover_tol(reps)
        @info "Monte Carlo (dml_did_multi, $reps reps)" cells = cover_cells / ncells
        @info "Monte Carlo (dml_did_multi)" simple = cover_simple / reps band =
            cover_band / reps se_ratio = mean(se) / std(est)
        @test abs(cover_cells / ncells - 0.95) < ml_cover_tol(ncells)
        @test abs(cover_simple / reps - 0.95) < tol
        @test abs(cover_band / reps - 0.95) < tol
        @test 0.75 < mean(se) / std(est) < 1.3
    end
end
