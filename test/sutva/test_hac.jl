using FixedEffectModels: reg, @formula, fe, Vcov

# Explicit O(n²) sandwich with an observation-pair kernel matrix.
function _sv_explicit_sandwich(X, e, Kobs)
    B = inv(X' * X)
    S = X .* e
    return B * (S' * Kobs * S) * B
end

_sv_kw(kernel, d, c) = kernel === :uniform ? Float64(d <= c) : max(0.0, 1 - d / c)

@testset "Spatial and network HAC covariance" begin
    rng = StableRNG(31)
    n = 60
    x = 10 .* rand(rng, n)
    y = 10 .* rand(rng, n)
    s = SpatialStructure(["p$(i)" for i in 1:n]; x=x, y=y)
    D = pairwise_distances(s)
    X = hcat(ones(n), randn(rng, n), randn(rng, n))
    e = randn(rng, n)

    @testset "cross-section matches explicit formula" begin
        for kernel in (:uniform, :bartlett), c in (1.0, 3.0)
            K = [_sv_kw(kernel, D[i, j], c) for i in 1:n, j in 1:n]
            V = conley_vcov(X, e, s, s.ids; cutoff=c, kernel=kernel, fix_psd=false)
            @test V ≈ _sv_explicit_sandwich(X, e, K)
        end
        # rows in any order: ids are matched by key
        perm = randperm(rng, n)
        V1 = conley_vcov(X, e, s, s.ids; cutoff=2.0, fix_psd=false)
        V2 = conley_vcov(X[perm, :], e[perm], s, s.ids[perm]; cutoff=2.0, fix_psd=false)
        @test V1 ≈ V2
        # a cutoff below every distance gives HC0 (HC1 with small_sample=true)
        c0 = minimum(D[i, j] for i in 1:n for j in 1:n if i != j) / 2
        V0 = conley_vcov(X, e, s, s.ids; cutoff=c0, small_sample=true)
        B = inv(X' * X)
        @test V0 ≈ B * ((X .* e)' * (X .* e)) * B * n / (n - 3)
        @test_throws ArgumentError conley_vcov(X, e, s, ["zz"; s.ids[2:end]]; cutoff=1.0)
        @test_throws DimensionMismatch conley_vcov(X, e[1:5], s, s.ids; cutoff=1.0)
    end

    @testset "panel: within-period spatial + serial within unit" begin
        T = 3
        ids = repeat(s.ids, T)
        tt = repeat(1:T; inner=n)
        Xp = hcat(ones(n * T), randn(rng, n * T))
        ep = randn(rng, n * T)
        idx = repeat(1:n, T)
        for (lagc, lagk) in ((Inf, :bartlett), (1.0, :bartlett), (1.0, :uniform))
            K = zeros(n * T, n * T)
            for a in 1:(n * T), b in 1:(n * T)
                i, j = idx[a], idx[b]
                if tt[a] == tt[b]
                    K[a, b] = _sv_kw(:uniform, D[i, j], 2.5)
                elseif i == j
                    l = abs(tt[a] - tt[b])
                    K[a, b] = isinf(lagc) ? 1.0 : lagk === :uniform ? Float64(l <= lagc) :
                              max(0.0, 1 - l / (lagc + 1))
                end
            end
            V = conley_vcov(Xp, ep, s, ids; cutoff=2.5, time=tt, lag_cutoff=lagc,
                            lag_kernel=lagk, fix_psd=false)
            @test V ≈ _sv_explicit_sandwich(Xp, ep, K)
        end
    end

    @testset "ConleyVcov in FixedEffectModels.reg (FWL with fixed effects)" begin
        g = rand(rng, 1:5, n)
        df = DataFrame(id=s.ids, x=x, y=y, x1=X[:, 2], x2=X[:, 3], g=g,
                       out=X * [1.0, 0.5, -0.5] .+ 0.2 .* g .+ e)
        m = reg(df, @formula(out ~ x1 + x2), ConleyVcov(; x=:x, y=:y, cutoff=2.0,
                                                        fix_psd=false))
        r = df.out .- X * coef(m)
        @test vcov(m) ≈ conley_vcov(X, r, s, s.ids; cutoff=2.0, fix_psd=false)
        ms = reg(df, @formula(out ~ x1 + x2), ConleyVcov(s; unit=:id, cutoff=2.0,
                                                         fix_psd=false))
        @test vcov(ms) ≈ vcov(m)
        mf = reg(df, @formula(out ~ x1 + x2 + fe(g)), ConleyVcov(s; unit=:id, cutoff=2.0,
                                                                  fix_psd=false))
        # demean by group by hand
        Xd = copy(X[:, 2:3])
        yd = copy(df.out)
        for k in unique(g)
            rows = g .== k
            Xd[rows, :] .-= mean(Xd[rows, :]; dims=1)
            yd[rows] .-= mean(yd[rows])
        end
        bd = Xd \ yd
        @test coef(mf) ≈ bd
        @test vcov(mf) ≈ conley_vcov(Xd, yd .- Xd * bd, s, s.ids; cutoff=2.0,
                                     fix_psd=false)
        # shuffled rows: identical results
        perm = randperm(rng, n)
        mp = reg(df[perm, :], @formula(out ~ x1 + x2 + fe(g)),
                 ConleyVcov(s; unit=:id, cutoff=2.0, fix_psd=false))
        @test vcov(mp) ≈ vcov(mf)
        # missing coordinates drop rows through Vcov.completecases
        dm = allowmissing(df)
        dm.x[1] = missing
        mm = reg(dm, @formula(out ~ x1 + x2), ConleyVcov(; x=:x, y=:y, cutoff=2.0))
        @test nobs(mm) == n - 1
        @test_throws ArgumentError ConleyVcov(; x=:x, cutoff=1.0)
        @test_throws ArgumentError ConleyVcov(; x=:x, y=:y, lat=:a, lon=:b, cutoff=1.0)
        @test_throws ArgumentError ConleyVcov(; x=:x, y=:y, cutoff=-1.0)
        @test_throws ArgumentError ConleyVcov(; x=:x, y=:y, cutoff=1.0, kernel=:gauss)
        @test_throws ArgumentError ConleyVcov(; lat=:x, lon=:y, cutoff=1.0, units=:m)
        @test occursin("Conley", sprint(show, ConleyVcov(s; unit=:id, cutoff=1.0)))
    end

    @testset "IV (2SLS) regressions" begin
        zi = randn(rng, n)
        xi = 0.8 .* zi .+ randn(rng, n)
        df = DataFrame(id=s.ids, zi=zi, xi=xi, out=1.0 .+ 0.5 .* xi .+ e)
        m = reg(df, @formula(out ~ (xi ~ zi)), ConleyVcov(s; unit=:id, cutoff=2.0,
                                                          fix_psd=false))
        Z = hcat(ones(n), zi)
        Xhat = hcat(ones(n), Z * (Z \ xi))
        r = df.out .- hcat(ones(n), xi) * coef(m)
        @test vcov(m) ≈ conley_vcov(Xhat, r, s, s.ids; cutoff=2.0, fix_psd=false)
        @test isfinite(m.F_kp)
    end

    @testset "non-PSD uniform kernel is repaired with a warning" begin
        sl = SpatialStructure(1:3; x=[0.0, 1.0, 2.0], y=zeros(3))
        Xl = ones(3, 1)
        el = [1.0, -1.0, 1.0]           # s' K s = 3 − 4 = −1 with K tridiagonal ones
        Vraw = conley_vcov(Xl, el, sl, 1:3; cutoff=1.0, fix_psd=false)
        @test Vraw[1, 1] < 0
        V = @test_logs (:warn, r"not positive semidefinite") conley_vcov(Xl, el, sl, 1:3;
                                                                           cutoff=1.0)
        @test V[1, 1] == 0
    end

    @testset "network HAC matches explicit path-distance kernel" begin
        A = zeros(n, n)
        for i in 1:n, j in (i + 1):n
            rand(rng) < 0.05 && (A[i, j] = A[j, i] = 1)
        end
        gnet = NetworkStructure(1:n, A)
        H = Matrix(shortest_path_hops(gnet; max_hops=n))
        for (b, kernel) in ((0, :bartlett), (1, :uniform), (2, :bartlett), (3, :uniform))
            K = [i == j ? 1.0 : (H[i, j] == 0 || H[i, j] > b) ? 0.0 :
                 kernel === :uniform ? 1.0 : 1 - H[i, j] / (b + 1) for i in 1:n, j in 1:n]
            V = network_hac_vcov(X, e, gnet, 1:n; bandwidth=b, kernel=kernel,
                                 fix_psd=false)
            @test V ≈ _sv_explicit_sandwich(X, e, K)
        end
        df = DataFrame(id=1:n, x1=X[:, 2], x2=X[:, 3], out=X * [1.0, 0.5, -0.5] .+ e)
        m = reg(df, @formula(out ~ x1 + x2), NetworkHACVcov(gnet; unit=:id, bandwidth=2,
                                                            fix_psd=false))
        r = df.out .- X * coef(m)
        @test vcov(m) ≈ network_hac_vcov(X, r, gnet, 1:n; bandwidth=2, fix_psd=false)
        @test_throws ArgumentError NetworkHACVcov(gnet; unit=:id, bandwidth=-1)
        @test_throws ArgumentError network_hac_vcov(X, e, gnet, 1:n; bandwidth=-1)
        @test occursin("bandwidth 2", sprint(show, NetworkHACVcov(gnet; unit=:id,
                                                                  bandwidth=2)))
    end

    @testset "Monte Carlo coverage with spatially correlated data" begin
        rngc = StableRNG(77)
        # Spatial HAC is biased downward when the dependence range is large relative
        # to the domain (coverage ≈ 0.86 with N = 400 on a 20 × 20 square and range 3);
        # here the range (2) is small relative to the domain (30 × 30).
        N = 900
        sx = 30 .* rand(rngc, N)
        sy = 30 .* rand(rngc, N)
        sc = SpatialStructure(1:N; x=sx, y=sy)
        W = Matrix(neighbor_matrix(sc; radius=1.0)) + I     # moving-average weights
        reps = mc_reps(500, 100)
        cov_c = 0
        cov_r = 0
        for rep in 1:reps
            xv = W * randn(rngc, N)
            u = W * randn(rngc, N)
            df = DataFrame(id=1:N, sx=sx, sy=sy, x=xv, y=1.0 .* xv .+ u)
            # dependence of x and u reaches 2 × 1.0 = 2: a uniform kernel to 2
            mc = reg(df, @formula(y ~ x), ConleyVcov(; x=:sx, y=:sy, cutoff=2.0))
            mr = reg(df, @formula(y ~ x), Vcov.robust())
            b = coef(mc)[2]
            cov_c += abs(b - 1) <= 1.96 * sqrt(vcov(mc)[2, 2])
            cov_r += abs(b - 1) <= 1.96 * sqrt(vcov(mr)[2, 2])
        end
        @info "Conley vs HC1 coverage (95%)" cov_c / reps cov_r / reps reps
        @test cov_c / reps >= 0.88
        @test cov_r / reps < cov_c / reps
    end

    @testset "Monte Carlo coverage with network-correlated data" begin
        rngn = StableRNG(78)
        N = 800
        A = zeros(N, N)
        for i in 1:N, j in (i + 1):N
            rand(rngn) < 4 / N && (A[i, j] = A[j, i] = 1)
        end
        gn = NetworkStructure(1:N, A)
        W = A + I
        reps = mc_reps(500, 100)
        cov_n = 0
        cov_r = 0
        for rep in 1:reps
            xv = W * randn(rngn, N)
            u = W * randn(rngn, N)
            df = DataFrame(id=1:N, x=xv, y=1.0 .* xv .+ u)
            # scores are dependent up to path distance 2 (shared neighbours)
            mn = reg(df, @formula(y ~ x), NetworkHACVcov(gn; unit=:id, bandwidth=2,
                                                         kernel=:uniform))
            mr = reg(df, @formula(y ~ x), Vcov.robust())
            b = coef(mn)[2]
            cov_n += abs(b - 1) <= 1.96 * sqrt(vcov(mn)[2, 2])
            cov_r += abs(b - 1) <= 1.96 * sqrt(vcov(mr)[2, 2])
        end
        @info "Network HAC vs HC1 coverage (95%)" cov_n / reps cov_r / reps reps
        @test cov_n / reps >= 0.88
        @test cov_r / reps < cov_n / reps
    end
end
