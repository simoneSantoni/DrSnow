# Monte Carlo size / coverage checks. Replication counts use mc_reps(full, fast); the
# fast bounds allow for the Monte Carlo error of the reduced counts (≈ 3 s.e.).

# Interactive fixed effects DGP with exchangeable units; treated units drawn at random.
function mc_factor_panel(rng; N=31, N1=1, T0=15, T1=5, tau=1.0, sigma=1.0)
    T = T0 + T1
    load = randn(rng, N, 2)
    fac = hcat(cumsum(randn(rng, T)) ./ 2, sin.((1:T) ./ 3))
    Y = randn(rng, N) .+ load * fac' .+ sigma .* randn(rng, N, T)
    treated = randperm(rng, N)[1:N1]
    rows = NamedTuple[]
    for i in 1:N, t in 1:T
        d = (i in treated && t > T0) ? 1 : 0
        push!(rows, (unit=i, time=t, y=Y[i, t] + tau * d, d=d))
    end
    return DataFrame(rows)
end

_mc_se(p, n) = sqrt(p * (1 - p) / n)

function _coverage(f, reps, seed)
    hits = 0
    for b in 1:reps
        r = f(StableRNG(seed + b))
        lo, hi = confint(r)[1, :]
        hits += lo <= 1.0 <= hi
    end
    return hits / reps
end

@testset "SDID placebo SE coverage (one treated unit)" begin
    reps = mc_reps(400, 60)
    cov = _coverage(reps, 1000) do rng
        df = mc_factor_panel(rng)
        synthetic_did(df, :y, :d, :unit, :time; se_method=:placebo,
                      replications=mc_reps(200, 100), rng=rng)
    end
    @info "SDID placebo 95% CI coverage" cov reps
    @test cov >= 0.95 - 3 * _mc_se(0.95, reps) - 0.02
    @test cov <= 1.0
end

@testset "SDID bootstrap and jackknife SE coverage (five treated units)" begin
    reps = mc_reps(300, 40)
    covb = _coverage(reps, 2000) do rng
        df = mc_factor_panel(rng; N=40, N1=5)
        synthetic_did(df, :y, :d, :unit, :time; se_method=:bootstrap,
                      replications=mc_reps(200, 100), rng=rng)
    end
    @info "SDID bootstrap 95% CI coverage" covb reps
    @test covb >= 0.95 - 3 * _mc_se(0.95, reps) - 0.02
    repsj = mc_reps(600, 100)
    covj = _coverage(repsj, 3000) do rng
        df = mc_factor_panel(rng; N=40, N1=5)
        synthetic_did(df, :y, :d, :unit, :time; se_method=:jackknife)
    end
    @info "SDID jackknife 95% CI coverage" covj repsj
    @test covj >= 0.95 - 3 * _mc_se(0.95, repsj) - 0.02
end

@testset "In-space placebo test size (classic SC)" begin
    reps = mc_reps(500, 100)
    rej = 0
    for b in 1:reps
        df = mc_factor_panel(StableRNG(4000 + b); N=20, tau=0.0)
        r = synthetic_control(df, :y, :d, :unit, :time)
        rej += synth_in_space_placebo(r).pvalue <= 0.10
    end
    rate = rej / reps
    @info "In-space placebo rejection rate at 10%" rate reps
    @test rate <= 0.10 + 3 * _mc_se(0.10, reps)
end

@testset "Conformal test size (SC, block permutations)" begin
    reps = mc_reps(400, 80)
    rej = 0
    for b in 1:reps
        df = mc_factor_panel(StableRNG(5000 + b); N=20, T0=19, T1=1, tau=0.0)
        r = augmented_synthetic_control(df, :y, :d, :unit, :time; ridge=false,
                                        se_method=:none)
        ci = synth_conformal_inference(r; grid_size=2)
        rej += ci.per_period.p_value[1] <= 0.10
    end
    rate = rej / reps
    @info "Conformal rejection rate at 10%" rate reps
    @test rate <= 0.10 + 3 * _mc_se(0.10, reps)
end

@testset "ASCM placebo and jackknife SE coverage" begin
    reps = mc_reps(300, 50)
    covp = _coverage(reps, 7000) do rng
        df = mc_factor_panel(rng)
        augmented_synthetic_control(df, :y, :d, :unit, :time;
                                    replications=mc_reps(200, 100), rng=rng)
    end
    @info "ASCM placebo 95% CI coverage (one treated unit)" covp reps
    @test covp >= 0.95 - 3 * _mc_se(0.95, reps) - 0.02
    covj = _coverage(reps, 8000) do rng
        df = mc_factor_panel(rng; N=40, N1=5)
        augmented_synthetic_control(df, :y, :d, :unit, :time; se_method=:jackknife)
    end
    @info "ASCM jackknife 95% CI coverage (five treated units)" covj reps
    @test covj >= 0.95 - 3 * _mc_se(0.95, reps) - 0.02
end

@testset "Classic SC placebo variance coverage" begin
    reps = mc_reps(300, 50)
    cov = _coverage(reps, 9000) do rng
        synthetic_control(mc_factor_panel(rng; N=25), :y, :d, :unit, :time)
    end
    @info "Classic SC placebo-variance 95% CI coverage" cov reps
    @test cov >= 0.95 - 3 * _mc_se(0.95, reps) - 0.05
end

@testset "MC-NNM placebo SE coverage (approximate)" begin
    reps = mc_reps(200, 25)
    cov = _coverage(reps, 6000) do rng
        df = mc_factor_panel(rng; N=31, N1=1)
        matrix_completion(df, :y, :d, :unit, :time; n_lambda=10, n_folds=2,
                          replications=mc_reps(100, 40), rng=rng)
    end
    @info "MC-NNM placebo 95% CI coverage" cov reps
    @test cov >= 0.95 - 3 * _mc_se(0.95, reps) - 0.05
end
