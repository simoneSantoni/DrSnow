# Rambachan–Roth sensitivity analysis (honest_did) against the HonestDiD R package
# (reference values from test/validation/did/generate_honestdid_references.R), exact
# properties, integration with the event-study estimators, error paths, and Monte
# Carlo coverage.

const HD_DMAP = Dict(
    "DeltaSD" => (:smoothness, :parallel_trends, nothing, nothing),
    "DeltaSDPB" => (:smoothness, :parallel_trends, :positive, nothing),
    "DeltaSDNB" => (:smoothness, :parallel_trends, :negative, nothing),
    "DeltaSDI" => (:smoothness, :parallel_trends, nothing, :increasing),
    "DeltaRM" => (:relative_magnitudes, :parallel_trends, nothing, nothing),
    "DeltaRMPB" => (:relative_magnitudes, :parallel_trends, :positive, nothing),
    "DeltaRMD" => (:relative_magnitudes, :parallel_trends, nothing, :decreasing),
    "DeltaSDRM" => (:relative_magnitudes, :linear_trend, nothing, nothing))
const HD_MMAP = Dict("FLCI" => :flci, "Conditional" => :conditional,
                     "C-F" => :hybrid_flci, "C-LF" => :hybrid_lf)

hd_quiet(f) = Base.CoreLogging.with_logger(f, Base.CoreLogging.NullLogger())

hd_parse_vec(x) = x isa AbstractString ? parse.(Float64, split(x, ";")) : [float(x)]

function hd_inputs()
    b = did_read_csv("hd_bc_beta.csv").beta
    S = Matrix(did_read_csv("hd_bc_sigma.csv"))
    mp = did_read_csv("hd_mpdta_es.csv")
    V = Matrix(did_read_csv("hd_mpdta_sigma.csv"))
    keep = findall(!=(-1), mp.e)
    return (bc=(b, S, 4), bc1=(b[1:5], S[1:5, 1:5], 4),
            mpdta=(mp.att[keep], V[keep, keep], count(<(-1), mp.e)), mp=mp, V=V)
end

# Exact FLCI criterion (half-length with the exact folded-normal quantile) of the
# affine estimator with weights `v` on (β_pre, β_post).
function hd_flci_criterion(v, S, npre, M, α)
    h = sqrt(dot(v, S * v))
    z = cumsum(cumsum(v[1:npre]))
    l = v[(npre + 1):end]
    npost = length(l)
    sbar = dot(1:npost, l)
    C0 = sum(abs(dot(1:s, l[(npost - s + 1):npost])) for s in 1:npost) - sbar
    bias = C0 + sum(abs, z)
    return DrSnow._did_folded_normal_quantile(1 - α, M * bias / h) * h
end

@testset "honest_did" begin
    inp = hd_inputs()
    ref = did_read_csv("r_honestdid.csv")

    @testset "vs HonestDiD R package" begin
        nflci = ncond = nlf = 0
        for r in eachrow(ref)
            r.delta == "Original" && continue
            β, Σ, npre = getproperty(inp, Symbol(r.case))
            npost = length(β) - npre
            l = hd_parse_vec(r.l)
            res, bound, bs, mono = HD_DMAP[r.delta]
            m = HD_MMAP[r.method]
            kw = m === :flci ? (;) :
                 (grid_lb=r.grid_lb, grid_ub=r.grid_ub, grid_points=Int(r.grid_points),
                  refine=false)
            # (grids that end inside the confidence set trigger a warning)
            h = hd_quiet() do
                honest_did(β, Σ, npre, npost; restriction=res, bound=bound,
                           bias_sign=bs, monotonicity=mono, method=m, M=[r.M],
                           l_vec=l, rng=StableRNG(11), kw...)
            end
            if m === :flci
                nflci += 1
                # HonestDiD evaluates cv_α with 10⁶ simulated folded-normal draws and
                # optimizes by numerical bisection: half-lengths agree to 0.2%, and our
                # optimum is at least as short as R's weights under the exact criterion.
                hl = (h.ub[1] - h.lb[1]) / 2
                @test hl ≈ r.flci_halflength rtol = 2e-3
                vR = hd_parse_vec(r.flci_vec)
                @test hl <= hd_flci_criterion(vR, Σ, npre, r.M, 0.05) + 1e-10
                @test h.lb[1] ≈ r.lb atol = 0.02 * (r.ub - r.lb)
                @test h.ub[1] ≈ r.ub atol = 0.02 * (r.ub - r.lb)
            else
                step = (r.grid_ub - r.grid_lb) / (r.grid_points - 1)
                if m === :hybrid_lf
                    # simulated least-favorable critical value (1000 draws in both)
                    nlf += 1
                    @test abs(h.lb[1] - r.lb) <= 2.01step
                    @test abs(h.ub[1] - r.ub) <= 2.01step
                else
                    # deterministic: identical accepted grid points
                    ncond += 1
                    @test h.lb[1] ≈ r.lb atol = 1e-9
                    @test h.ub[1] ≈ r.ub atol = 1e-9
                end
            end
        end
        @test nflci >= 10 && ncond >= 40 && nlf >= 6
        # original confidence interval
        o = ref[ref.delta .== "Original", :][1, :]
        β, Σ, npre = inp.mpdta
        h = honest_did(β, Σ, npre, length(β) - npre; restriction=:smoothness, M=[0.0])
        @test h.original[1] ≈ o.lb atol = 1e-12
        @test h.original[2] ≈ o.ub atol = 1e-12
    end

    @testset "Callaway–Sant'Anna event study input" begin
        mp = did_read_csv("mpdta.csv")
        cs = did_callaway_santanna(mp, :lemp, FirstTreated(:first_treat), :countyreal,
                                   :year; base_period=:universal, bootstrap=false)
        es = aggregate_att(cs, :dynamic)
        @test es.reference == [-1]
        # same β and Σ as HonestDiD's honest_did.AGGTEobj
        rmp = inp.mp
        keep = findall(!=(-1), rmp.e)
        @test relative_periods(es) == Int.(rmp.e[keep])
        @test coef(es) ≈ rmp.att[keep] atol = 1e-10
        @test vcov(es) ≈ inp.V[keep, keep] rtol = 1e-7
        cond = ref[(ref.case .== "mpdta") .& (ref.method .== "Conditional"), :]
        h = honest_did(es; restriction=:relative_magnitudes, method=:conditional,
                       M=cond.M, grid_lb=cond.grid_lb[1], grid_ub=cond.grid_ub[1],
                       grid_points=Int(cond.grid_points[1]), refine=false)
        @test h.lb ≈ cond.lb atol = 1e-9
        @test h.ub ≈ cond.ub atol = 1e-9
        @test h.post_periods == [0, 1, 2, 3]
        @test h.pre_periods == [-4, -3, -2]
        # varying base period is rejected with an explanation
        csv = did_callaway_santanna(mp, :lemp, FirstTreated(:first_treat), :countyreal,
                                    :year; bootstrap=false)
        @test_throws ArgumentError honest_did(aggregate_att(csv, :dynamic))
    end

    @testset "targets, defaults and properties" begin
        β, Σ, npre = inp.bc
        npost = length(β) - npre
        # default M grids follow HonestDiD
        h = honest_did(β, Σ, npre, npost; restriction=:smoothness)
        @test length(h.M) == 10 && h.M[1] == 0
        @test h.method === :flci
        h2 = honest_did(β, Σ, npre, npost; restriction=:relative_magnitudes,
                        M=[0.0, 0.5, 1.0, 1.5], grid_points=300, rng=StableRNG(1))
        @test h2.method === :hybrid_lf
        # robust sets widen with M (nested restriction sets)
        @test all(diff(h2.ub) .>= -1e-8) && all(diff(h2.lb) .<= 1e-8)
        @test all(diff(h.ub .- h.lb) .>= -1e-8)
        # every robust set contains the M = 0 point estimate region's centre
        @test all(h2.lb .<= dot([1, 0, 0, 0], β[5:8]) .<= h2.ub)
        # refinement lands between the grid points it brackets
        hc = honest_did(β, Σ, npre, npost; restriction=:relative_magnitudes,
                        method=:conditional, M=[1.0], grid_points=200)
        hg = honest_did(β, Σ, npre, npost; restriction=:relative_magnitudes,
                        method=:conditional, M=[1.0], grid_points=200, refine=false)
        @test hc.lb[1] <= hg.lb[1] && hc.ub[1] >= hg.ub[1]
        @test (hc.ub[1] - hc.lb[1]) - (hg.ub[1] - hg.lb[1]) < 0.05 * (hg.ub[1] - hg.lb[1])
        # relative periods must be consecutive around the reference period
        esgap = EventStudyEstimate([-5, -4, -2, 0, 1, 2, 3], β[2:8], Σ[2:8, 2:8], [-1],
                                   1000, Inf, 0, "test", "", Float64[],
                                   (binned=(false, false),))
        @test_throws ArgumentError honest_did(esgap)
        es = EventStudyEstimate([-5, -4, -3, -2, 0, 1, 2, 3], β, Σ, [-1], 1000, Inf, 0,
                                "test", "", Float64[], (binned=(false, false),))
        ha = honest_did(es; restriction=:smoothness, M=[0.01], target=0:3)
        hr = honest_did(β, Σ, npre, npost; restriction=:smoothness, M=[0.01],
                        l_vec=fill(0.25, 4))
        @test ha.lb ≈ hr.lb && ha.ub ≈ hr.ub
        @test ha.estimate ≈ mean(β[5:8])
        he = honest_did(es; restriction=:smoothness, M=[0.01], target=2)
        @test he.l_vec == [0, 0, 1, 0]
        @test_throws ArgumentError honest_did(es; target=7)
        # show / confint
        io = IOBuffer()
        show(io, MIME"text/plain"(), ha)
        s = String(take!(io))
        @test occursin("DeltaSD", s) && occursin("original CI", s)
        @test size(confint(h2)) == (4, 2)
    end

    @testset "breakdown values" begin
        β, Σ, npre = inp.bc
        npost = length(β) - npre
        bd = honest_breakdown(β, Σ, npre, npost; restriction=:relative_magnitudes,
                              method=:conditional, tol=1e-4)
        @test 0 < bd < Inf
        lo = honest_did(β, Σ, npre, npost; restriction=:relative_magnitudes,
                        method=:conditional, M=[0.97bd], grid_points=400)
        hi = honest_did(β, Σ, npre, npost; restriction=:relative_magnitudes,
                        method=:conditional, M=[1.03bd], grid_points=400)
        @test lo.lb[1] > 0
        @test hi.lb[1] <= 0
        # grid breakdown is the first grid value whose set contains zero
        g = honest_did(β, Σ, npre, npost; restriction=:relative_magnitudes,
                       method=:conditional, M=[0.5, 1.0, 1.5, 2.0], grid_points=300)
        @test g.breakdown == first(g.M[(g.lb .<= 0) .& (g.ub .>= 0)])
        @test g.breakdown >= bd - 1e-3
        # FLCI breakdown: 0 is at the edge of the interval
        bs = honest_breakdown(β, Σ, npre, npost; restriction=:smoothness, tol=1e-6)
        f = honest_did(β, Σ, npre, npost; restriction=:smoothness, M=[bs])
        @test f.lb[1] ≈ 0 atol = 1e-4
    end

    @testset "TWFE and Sun–Abraham event studies" begin
        rng = StableRNG(2024)
        df = sim_staggered(rng; N=300, T=8, cohorts=[0, 5], effect=(g, e) -> 1.0)
        es = event_study(df, :y, :d, :unit, :time; estimator=:twfe)
        h = honest_did(es; restriction=:relative_magnitudes, M=[0.0, 1.0],
                       rng=StableRNG(1), grid_points=200)
        @test h.pre_periods == [-4, -3, -2] && h.post_periods == [0, 1, 2, 3]
        @test h.lb[1] < 1 < h.ub[1]
        esb = event_study(df, :y, :d, :unit, :time; estimator=:twfe, max_pre=2)
        @test_throws ArgumentError honest_did(esb)          # binned endpoint
        df2 = sim_staggered(StableRNG(5); N=300, T=8, cohorts=[0, 4, 6])
        sa = did_sun_abraham(df2, :y, :d, :unit, :time)
        hs = honest_did(sa; restriction=:smoothness, M=[0.0, 0.1])
        @test all(hs.lb .< hs.ub)
        imp = did_imputation(df2, :y, :d, :unit, :time; horizons=0:2, pretrends=3)
        @test_throws ArgumentError honest_did(imp)
        # the imputation estimator has no period normalized to zero between its
        # pre-trend and post-treatment coefficients
        @test_throws ArgumentError honest_did(imp; reference=-4)
    end

    @testset "input validation" begin
        β, Σ, npre = inp.bc
        @test_throws DimensionMismatch honest_did(β, Σ, 3, 4)
        @test_throws DimensionMismatch honest_did(β, Σ[1:7, 1:7], 4, 4)
        @test_throws ArgumentError honest_did(β, -Σ, 4, 4)
        @test_throws ArgumentError honest_did(β, Σ, 4, 4; restriction=:foo)
        @test_throws ArgumentError honest_did(β, Σ, 4, 4; method=:flci)   # RM
        @test_throws ArgumentError honest_did(β, Σ, 4, 4; bias_sign=:positive,
                                              monotonicity=:increasing)
        @test_throws ArgumentError honest_did(β, Σ, 4, 4; bias_sign=:up)
        @test_throws ArgumentError honest_did(β, Σ, 4, 4; M=[-1.0])
        @test_throws ArgumentError honest_did(β, Σ, 4, 4; l_vec=zeros(4))
        @test_throws DimensionMismatch honest_did(β, Σ, 4, 4; l_vec=ones(3))
        @test_throws ArgumentError honest_did(β[4:8], Σ[4:8, 4:8], 1, 4;
                                              bound=:linear_trend)
        @test_throws ArgumentError honest_did(β, Σ, 4, 4; level=1.2)
        @test_logs (:warn, r"ignores the sign") honest_did(β, Σ, 4, 4;
            restriction=:smoothness, method=:flci, bias_sign=:positive, M=[0.01])
    end

    @testset "numerical kernels" begin
        # simplex: small LP with known optimum and duals
        lp = DrSnow._did_simplex([3.0, 2.0, 0.0, 0.0], [1.0 1 1 0; 1 3 0 1], [4.0, 6.0])
        @test lp.status === :optimal
        @test lp.objective ≈ 12 && lp.x[1] ≈ 4
        @test DrSnow._did_simplex([1.0, 1.0], [1.0 -1.0], [1.0]).status === :unbounded
        @test DrSnow._did_simplex([1.0], [1.0;;], [-1.0]).status === :infeasible
        # lasso path: KKT conditions at every breakpoint
        Q = [2.0 0.5 0.1; 0.5 1.0 0.2; 0.1 0.2 0.5]
        r = [1.0, -0.4, 0.3]
        μs, Z = DrSnow._did_lasso_path(Q, r)
        @test μs[end] == 0 && Z[:, end] ≈ -(Q \ r)
        for k in eachindex(μs)
            g = Q * Z[:, k] .+ r
            for j in 1:3
                if abs(Z[j, k]) > 1e-10
                    @test g[j] ≈ -μs[k] * sign(Z[j, k]) atol = 1e-8
                else
                    @test abs(g[j]) <= μs[k] + 1e-8
                end
            end
        end
        # truncated normal quantile far in the tail and folded normal quantile
        @test DrSnow._did_truncnorm_quantile(0.5, 40.0, Inf) > 40
        @test DrSnow._did_truncnorm_quantile(0.5, -Inf, Inf) ≈ 0 atol = 1e-12
        @test DrSnow._did_truncnorm_quantile(0.3, -Inf, -38.0) < -38
        @test DrSnow._did_folded_normal_quantile(0.95, 0.0) ≈ 1.959963984540054
        c = DrSnow._did_folded_normal_quantile(0.9, 1.3)
        N01 = DrSnow.Normal()
        @test DrSnow.cdf(N01, c - 1.3) - DrSnow.cdf(N01, -c - 1.3) ≈ 0.9 atol = 1e-12
    end

    @testset "Monte Carlo coverage" begin
        # β̂ ~ N(τ + δ, Σ) with δ in the restriction set; coverage of θ = τ_post[1].
        rng = StableRNG(99)
        npre, npost = 4, 3
        T = npre + npost
        Σ = [0.01 * 0.5^abs(i - j) for i in 1:T, j in 1:T]
        τ = vcat(zeros(npre), [0.2, 0.3, 0.4])
        tt = vcat(-npre:-1, 1:npost)
        R = mc_reps(1000, 150)
        L = cholesky(Symmetric(Σ)).L
        # Δ^SD(M): quadratic trend with curvature exactly M (worst case for smoothness)
        M = 0.02
        δsd = 0.03 .* tt .+ (M / 2) .* tt .* (tt .+ 1) .* (tt .> 0)
        # Δ^RM(1): post changes equal to the largest pre change
        δrm = vcat([0.0, 0.05, 0.0, 0.05] .- 0.05, 0.05 .* (1:npost))
        cov_flci = cov_cond = cov_rm = 0
        for _ in 1:R
            e = L * randn(rng, T)
            b1 = τ .+ δsd .+ e
            f = DrSnow._did_hd_flci(Σ, M, npre, npost, [1.0, 0, 0], 0.05)
            c = dot(f.optimal_vec, b1)
            cov_flci += abs(c - τ[npre + 1]) <= f.halflength
            acc, _ = DrSnow._did_hd_acceptor(b1, Σ, npre, npost, [1.0, 0, 0], 0.05,
                                             :smoothness, M, :arp, nothing, nothing,
                                             true, 1000, 0.005, rng)
            cov_cond += acc(τ[npre + 1])
            b2 = τ .+ δrm .+ e
            acc2, _ = DrSnow._did_hd_acceptor(b2, Σ, npre, npost, [1.0, 0, 0], 0.05,
                                              :relative_magnitudes, 1.0, :lf, nothing,
                                              nothing, true, 200, 0.005, rng)
            cov_rm += acc2(τ[npre + 1])
        end
        # Robust sets are uniformly valid: coverage at least nominal up to MC error.
        for c in (cov_flci, cov_cond, cov_rm)
            @test c / R >= 0.95 - 3.5 * sqrt(0.95 * 0.05 / R) - 0.01
        end
        # FLCI coverage is exactly nominal at the worst-case bias for Δ^SD.
        @test mc_close(cov_flci / R, 0.95, R) || cov_flci / R > 0.95
    end
end
