# Wooldridge extended TWFE (did_etwfe) against R's etwfe/emfx
# (test/validation/did/generate_etwfe_references.R), exact equivalences with the
# imputation and Sun–Abraham estimators, invariance, errors and Monte Carlo coverage.

const ETWFE_TYPES = Dict("simple" => :simple, "group" => :group,
                         "calendar" => :calendar, "event" => :dynamic)

function etwfe_case(mp, c)
    T = FirstTreated(:first_treat)
    c == "lin" && return did_etwfe(mp, :lemp, T, :countyreal, :year)
    c == "lin_cov" && return did_etwfe(mp, :lemp, T, :countyreal, :year;
                                       covariates=[:lpop])
    c == "lin_never" && return did_etwfe(mp, :lemp, T, :countyreal, :year;
                                         control_group=:never_treated)
    c == "lin_unit" && return did_etwfe(mp, :lemp, T, :countyreal, :year; fe=:unit)
    c == "pois_iid" && return did_etwfe(mp, :emp, T, :countyreal, :year;
                                        family=:poisson, cluster=nothing)
    c == "pois" && return did_etwfe(mp, :emp, T, :countyreal, :year; family=:poisson)
    c == "pois_cov" && return did_etwfe(mp, :emp, T, :countyreal, :year;
                                        family=:poisson, covariates=[:lpop])
    c == "logit_iid" && return did_etwfe(mp, :high, T, :countyreal, :year;
                                         family=:logit, cluster=nothing)
    error("unknown case $c")
end

@testset "did_etwfe" begin
    mp = did_read_csv("mpdta.csv")
    mp.emp = round.(exp.(mp.lemp))
    mp.high = Int.(mp.lemp .> 5.5)
    ref = did_read_csv("r_etwfe.csv")
    refc = did_read_csv("r_etwfe_coef.csv")

    @testset "vs etwfe::emfx: $c" for c in unique(ref.case)
        r = etwfe_case(mp, c)
        rr = ref[ref.case .== c, :]
        glm = startswith(c, "pois") || startswith(c, "logit")
        # fixest scales the model-based GLM covariance by (n - 1) / (n - K)
        K = glm ? length(r.model.coef) : 0
        iidfac = endswith(c, "_iid") ? sqrt((nobs(r) - 1) / (nobs(r) - K)) : 1.0
        for ty in unique(rr.type)
            a = aggregate_att(r, ETWFE_TYPES[ty])
            b, s = coef(a), stderror(a)
            for row in eachrow(rr[rr.type .== ty, :])
                if ty == "event" && row.label == -1
                    # emfx reports the omitted reference period as 0 (never-treated
                    # controls); DrSnow records it as the reference instead
                    @test row.estimate == 0 && a.reference == [-1]
                    @test !(-1 in relative_periods(a))
                    continue
                end
                k = ty == "simple" ? 1 :
                    ty == "event" ? findfirst(==(Int(row.label)), relative_periods(a)) :
                    1 + findfirst(==(Int(row.label)), a.labels)
                if glm
                    @test b[k] ≈ row.estimate rtol = 1e-7
                    # fixest's IRLS stops at a deviance tolerance of 1e-8
                    @test s[k] * iidfac ≈ row.se rtol = endswith(c, "_iid") ? 2e-5 : 1e-4
                else
                    @test b[k] ≈ row.estimate atol = 1e-10
                    # FixedEffectModels' small-sample factor differs from fixest's
                    @test s[k] ≈ row.se rtol = 1e-3
                end
            end
        end
        if !glm && c != "lin_cov"
            # linear cell coefficients are the ATT(g,t)
            rc = refc[refc.case .== c, :]
            @test length(coef(r)) == nrow(rc)
            for row in eachrow(rc)
                m = match(r"first_treat::(\d+):year::(\d+)", row.term)
                g, t = parse(Int, m[1]), parse(Int, m[2])
                k = findfirst(j -> r.periods[r.cohorts[j]] == g &&
                                   r.periods[r.times[j]] == t, eachindex(r.coef))
                @test coef(r)[k] ≈ row.estimate atol = 1e-10
                @test stderror(r)[k] ≈ row.se rtol = 1e-3
            end
        end
    end

    @testset "equivalences" begin
        rng = StableRNG(77)
        df = sim_staggered(rng; N=300, T=7, cohorts=[0, 3, 4, 6],
                           effect=(g, e) -> 1.0 + 0.4e - 0.1g)
        # not-yet-treated ETWFE = imputation estimator (Wooldridge 2021; BJS 2024)
        r = did_etwfe(df, :y, :d, :unit, :time)
        es = aggregate_att(r, :dynamic)
        imp = did_imputation(df, :y, :d, :unit, :time; horizons=0:4)
        for e in 0:4
            @test coef(es)[findfirst(==(e), relative_periods(es))] ≈
                  coef(imp)[findfirst(==(e), relative_periods(imp))] atol = 1e-8
        end
        @test coef(aggregate_att(r, :simple))[1] ≈
              coef(did_imputation(df, :y, :d, :unit, :time))[1] atol = 1e-8
        # never-treated ETWFE = Sun–Abraham interaction-weighted event study
        rn = did_etwfe(df, :y, :d, :unit, :time; control_group=:never_treated)
        esn = aggregate_att(rn, :dynamic)
        sa = did_sun_abraham(df, :y, :d, :unit, :time)
        for e in relative_periods(sa)
            @test coef(esn)[findfirst(==(e), relative_periods(esn))] ≈
                  coef(sa)[findfirst(==(e), relative_periods(sa))] atol = 1e-8
        end
        @test esn.reference == [-1]
        # cohort and unit fixed effects give the same ATT(g,t) in a balanced panel
        ru = did_etwfe(df, :y, :d, :unit, :time; fe=:unit)
        @test coef(ru) ≈ coef(r) atol = 1e-8
        # row order does not matter
        rs = did_etwfe(shuffle_rows(StableRNG(3), df), :y, :d, :unit, :time)
        @test coef(rs) ≈ coef(r) atol = 1e-10
        @test vcov(rs) ≈ vcov(r) rtol = 1e-8
        # repeated cross-sections (cohort fixed effects, robust SEs)
        rc = did_etwfe(df, :y, FirstTreated(:g), nothing, :time; cluster=nothing)
        @test coef(rc) ≈ coef(r) atol = 1e-8
        # aggregations are consistent with the cell effects
        a = aggregate_att(r, :group)
        @test coef(a)[1] ≈ coef(aggregate_att(r, :simple))[1]
        @test dof_residual(a) == dof_residual(r)
        @test occursin("Wooldridge", method_name(a))
        @test length(coefnames(r)) == length(coef(r))
    end

    @testset "errors" begin
        T = FirstTreated(:first_treat)
        @test_throws ArgumentError did_etwfe(mp, :lemp, T, :countyreal, :year;
                                             family=:probit)
        @test_throws ArgumentError did_etwfe(mp, :lemp, T, :countyreal, :year; fe=:x)
        @test_throws ArgumentError did_etwfe(mp, :lemp, T, :countyreal, :year;
                                             family=:poisson, fe=:unit)
        @test_throws ArgumentError did_etwfe(mp, :lemp, T, :countyreal, :year;
                                             family=:logit)
        neg = copy(mp)
        neg.emp .-= 1e6
        @test_throws ArgumentError did_etwfe(neg, :emp, T, :countyreal, :year;
                                             family=:poisson)
        nonabs = DataFrame(u=repeat(1:30; inner=4), t=repeat(1:4; outer=30))
        nonabs.d = [(u % 3 == 0 && t in (2, 3)) ? 1 : 0 for (u, t) in
                    zip(nonabs.u, nonabs.t)]
        nonabs.y = randn(StableRNG(1), nrow(nonabs))
        @test_throws ArgumentError did_etwfe(nonabs, :y, :d, :u, :t)
        allt = mp[mp.first_treat .> 0, :]
        @test_throws ArgumentError did_etwfe(allt, :lemp, T, :countyreal, :year;
                                             control_group=:never_treated)
        @test_logs (:warn, r"last-treated cohort") did_etwfe(allt, :lemp, T,
                                                             :countyreal, :year)
    end

    @testset "Monte Carlo coverage" begin
        R = mc_reps(400, 80)
        rng = StableRNG(1234)
        hit_lin = hit_pois = 0
        for _ in 1:R
            df = sim_staggered(rng; N=200, T=6, cohorts=[0, 3, 5],
                               effect=(g, e) -> 1.0 + 0.5e)
            truth = true_simple_att(df, (g, e) -> 1.0 + 0.5e)
            a = aggregate_att(did_etwfe(df, :y, :d, :unit, :time), :simple)
            ci = confint(a)
            hit_lin += ci[1, 1] <= truth <= ci[1, 2]
            # Poisson DGP: E[y] = exp(α_i + λ_t + 0.3 D), ATT on the count scale
            n, T = 200, 5
            g = rand(rng, [0, 3, 4], n)
            α = 0.5 .+ 0.3 .* randn(rng, n)
            λ = [0.0, 0.1, 0.15, 0.3, 0.35]
            rows = [(i, t, Int(g[i] > 0 && t >= g[i])) for i in 1:n for t in 1:T]
            μ0 = [exp(α[i] + λ[t]) for (i, t, _) in rows]
            μ1 = μ0 .* exp(0.3)
            d = [x[3] for x in rows]
            ypois = [rand(rng, DrSnow.Poisson(d[k] == 1 ? μ1[k] : μ0[k]))
                     for k in eachindex(rows)]
            pdf = DataFrame(u=first.(rows), t=getindex.(rows, 2), d=d, y=ypois)
            tp = mean((μ1 .- μ0)[d .== 1])
            ap = aggregate_att(did_etwfe(pdf, :y, :d, :u, :t; family=:poisson), :simple)
            cp = confint(ap)
            hit_pois += cp[1, 1] <= tp <= cp[1, 2]
        end
        @test mc_close(hit_lin / R, 0.95, R)
        # the Poisson target is the sample ATT given the realized unit effects, so
        # the (unconditional) standard errors can only be conservative for it
        @test hit_pois / R >= 0.95 - 3.5 * sqrt(0.95 * 0.05 / R) - 0.01
    end
end
