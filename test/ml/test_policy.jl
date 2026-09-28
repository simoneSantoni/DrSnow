# Brute-force optimum over depth-1 / depth-2 trees for small problems.
function ml_brute_depth1(Γ, X, idx)
    best = maximum(sum(Γ[idx, :]; dims=1))
    for j in axes(X, 2), thr in unique(X[idx, j])
        l = idx[X[idx, j] .<= thr]
        r = idx[X[idx, j] .> thr]
        (isempty(l) || isempty(r)) && continue
        best = max(best, maximum(sum(Γ[l, :]; dims=1)) + maximum(sum(Γ[r, :]; dims=1)))
    end
    return best
end

function ml_brute_depth2(Γ, X)
    idx = collect(axes(Γ, 1))
    best = ml_brute_depth1(Γ, X, idx)
    for j in axes(X, 2), thr in unique(X[:, j])
        l = idx[X[:, j] .<= thr]
        r = idx[X[:, j] .> thr]
        (isempty(l) || isempty(r)) && continue
        best = max(best, ml_brute_depth1(Γ, X, l) + ml_brute_depth1(Γ, X, r))
    end
    return best
end

@testset "Policy learning" begin
    @testset "exhaustive tree search is optimal" begin
        for (s, A) in ((1, 2), (2, 3), (3, 2))
            rng = StableRNG(60 + s)
            n = 30
            X = round.(randn(rng, n, 2); digits=1)   # ties on purpose
            Γ = randn(rng, n, A)
            for depth in (1, 2)
                t = policy_tree(Γ, X; depth=depth, actions=collect(1:A))
                opt = depth == 1 ? ml_brute_depth1(Γ, X, collect(1:n)) :
                      ml_brute_depth2(Γ, X)
                @test t.reward * n ≈ opt
                acts = predict(t, X)
                @test sum(Γ[i, acts[i]] for i in 1:n) ≈ opt
            end
        end
        # min_node_size is respected
        rng = StableRNG(64)
        X = randn(rng, 40, 1)
        Γ = randn(rng, 40, 2)
        t = policy_tree(Γ, X; depth=1, min_node_size=15)
        acts = predict(t, X)
        if !t.nodes[1].leaf
            @test count(X[:, 1] .<= t.nodes[1].threshold) >= 15
            @test count(X[:, 1] .> t.nodes[1].threshold) >= 15
        end
        @test_throws ArgumentError policy_tree(Γ, X; depth=3)
        @test_throws DimensionMismatch policy_tree(Γ[1:5, :], X)
        @test_throws ArgumentError policy_tree(Γ[:, 1:1], X)
        @test occursin("PolicyTree (depth 1)", sprint(show, MIME"text/plain"(), t))
    end

    @testset "doubly-robust policy tree on data" begin
        rng = StableRNG(65)
        n = 1500
        X = randn(rng, n, 3)
        d = Float64.(rand(rng, n) .< 0.5)
        τ = ifelse.(X[:, 1] .> 0.0, 1.0, -1.0)
        y = d .* τ .+ X[:, 2] .+ randn(rng, n)
        df = DataFrame(X, [:x1, :x2, :x3])
        df.d = d
        df.y = y
        kw = (covariates=[:x1, :x2, :x3], outcome_learner=OLSLearner(),
              propensity_learner=LogisticLearner())
        r = policy_tree(df, :y, :d; kw..., depth=1, rng=StableRNG(1))
        @test r isa PolicyLearningResult
        @test r.tree.nodes[1].var == 1 && abs(r.tree.nodes[1].threshold) < 0.25
        @test predict(r, DataFrame(x1=[-1.0, 1.0], x2=0.0, x3=0.0)) == [0, 1]
        @test coefnames(r)[1] == "value(policy)"
        @test all(pvalues(r)[2:3] .< 0.01)
        @test length(r.fold_trees) == 5
        r2 = policy_tree(df, :y, :d; kw..., depth=2, policy_covariates=[:x1, :x3],
                         split_step=5, rng=StableRNG(1))
        @test r2.tree.depth == 2 && r2.tree.covariates == [:x1, :x3]
        @test occursin("Share assigned", sprint(show, MIME"text/plain"(), r2))
        @test_throws ArgumentError policy_tree(df, :y, :d; kw...,
                                               policy_covariates=Symbol[])

        # Monte Carlo: coverage of the out-of-fold policy value. The target is the
        # true value of the fold-specific rules, Σ_k (n_k/n) V(π₋ₖ), with
        # V(π) = E[X₂ + 1{π(X) = 1} τ(X)] = E[1{π(X)=1} τ(X)], computed on a large
        # independent sample.
        Xbig = randn(StableRNG(66), 200_000, 3)
        τbig = ifelse.(Xbig[:, 1] .> 0.0, 1.0, -1.0)
        value(t) = mean((predict(t, Xbig) .== 1) .* τbig)
        reps = mc_reps(300, 50)
        cover = 0
        for rep in 1:reps
            rg = StableRNG(7000 + rep)
            nr = 600
            Xr = randn(rg, nr, 3)
            dr = Float64.(rand(rg, nr) .< 0.5)
            yr = dr .* ifelse.(Xr[:, 1] .> 0.0, 1.0, -1.0) .+ Xr[:, 2] .+ randn(rg, nr)
            dfr = DataFrame(Xr, [:x1, :x2, :x3])
            dfr.d = dr
            dfr.y = yr
            rr = policy_tree(dfr, :y, :d; kw..., depth=1, rng=rg)
            nk = [count(==(k), rr.folds) for k in 1:5]
            truth = sum(nk[k] / nr * value(rr.fold_trees[k]) for k in 1:5)
            ci = confint(rr)
            cover += ci[1, 1] <= truth <= ci[1, 2]
        end
        @info "Monte Carlo policy value coverage" cover / reps
        @test abs(cover / reps - 0.95) <= ml_cover_tol(reps)
    end
end
