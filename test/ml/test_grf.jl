# Generalized random forests: algorithmic properties of the grf port, front ends,
# reproducibility and invariances, post-estimation interfaces and error paths.

function grf_data(rng, n; p=4, tau=x -> 1 + 2x[1], confounded=true)
    X = rand(rng, n, p)
    e = confounded ? 0.25 .+ 0.5 .* X[:, 3] : fill(0.5, n)
    w = Float64.(rand(rng, n) .< e)
    τ = [tau(view(X, i, :)) for i in 1:n]
    y = X[:, 3] .+ τ .* w .+ randn(rng, n)
    df = DataFrame(X, [Symbol("x$j") for j in 1:p])
    df.w = w
    df.y = y
    df.tau = τ
    return df
end

@testset "Generalized random forests" begin
    xs = [:x1, :x2, :x3, :x4]

    @testset "forest weights reproduce the predictions" begin
        df = grf_data(StableRNG(1), 400)
        cf = causal_forest(df, :y, :w; covariates=xs, num_trees=100, rng=StableRNG(2))
        rf = regression_forest(df, :y; covariates=xs, num_trees=100, rng=StableRNG(3))
        Xn = rand(StableRNG(4), 5, 4)
        for (f, kind) in ((cf, :causal), (rf, :regression))
            L = DrSnow._grf_leaf_matrix(f.forest, Xn, false, false)
            A = DrSnow._grf_forest_weights(f.forest, L, nobs(f))
            @test all(sum(A; dims=1) .≈ 1)
            o = f.order
            y = kind === :causal ? (f.Y .- f.Y_hat)[o] : f.Y[o]
            if kind === :regression
                @test predict(f, Xn) ≈ A' * y rtol = 1e-10
            else
                w = (f.W .- f.W_hat)[o]
                τ = [begin
                         a = A[:, j]
                         yb, wb = dot(a, y), dot(a, w)
                         dot(a, (w .- wb) .* (y .- yb)) / dot(a, (w .- wb) .^ 2)
                     end for j in 1:5]
                @test predict(f, Xn) ≈ τ rtol = 1e-8
            end
        end
    end

    @testset "honesty, pruning, clusters and split constraints" begin
        df = grf_data(StableRNG(5), 300)
        df.cl = repeat(1:30, inner=10)
        cf = causal_forest(df, :y, :w; covariates=xs, num_trees=40, cluster=:cl,
                           rng=StableRNG(6))
        o = cf.order
        cl_int = cf.cluster[o]
        for t in cf.forest.trees
            drawn = Set(t.drawn)
            # subsamples are unions of whole clusters
            @test all(i -> (i in drawn) == all(j -> j in drawn,
                                                findall(==(cl_int[i]), cl_int)),
                      unique(t.drawn))
            # honest leaves: populated only by in-bag units, no reachable empty leaf
            reach = Int[]
            stack = [t.root]
            while !isempty(stack)
                nd = pop!(stack)
                if DrSnow._grf_is_leaf(t, nd)
                    push!(reach, nd)
                else
                    push!(stack, t.left[nd], t.right[nd])
                end
            end
            @test all(nd -> !isempty(t.leaf_samples[nd]), reach)
            @test all(s -> s in drawn, reduce(vcat, t.leaf_samples[reach]))
        end
        # without honesty and with W̃ = ±0.5, every leaf holds at least min_node_size
        # treated and control units (the instrumental splitting rule)
        cf2 = causal_forest(df, :y, :w; covariates=xs, num_trees=20, w_hat=0.5,
                            honesty=false, min_node_size=7, rng=StableRNG(7))
        wi = cf2.W[cf2.order]
        ok = true
        for t in cf2.forest.trees, nd in eachindex(t.leaf_samples)
            L = t.leaf_samples[nd]
            (isempty(L) || nd == t.root) && continue
            ok &= count(==(1), wi[L]) >= 7 && count(==(0), wi[L]) >= 7
        end
        @test ok
        # a causal forest is an instrumental forest with the treatment as instrument
        iv = instrumental_forest(df, :y, :w, :w; covariates=xs, num_trees=40,
                                 y_hat=cf.Y_hat, w_hat=cf.W_hat, z_hat=cf.W_hat,
                                 rng=StableRNG(8))
        cfz = causal_forest(df, :y, :w; covariates=xs, num_trees=40, y_hat=cf.Y_hat,
                            w_hat=cf.W_hat, rng=StableRNG(8))
        @test predict(iv) == predict(cfz)
        # split frequencies at depth 1 sum to the number of split roots
        S = split_frequencies(cf; max_depth=2)
        @test size(S) == (2, 4) && sum(S[1, :]) <= length(cf.forest.trees)
        vi = variable_importance(cf)
        @test vi.variable == string.(xs) && sum(vi.importance) <= 1 + 1e-12
    end

    @testset "reproducibility and invariance" begin
        df = grf_data(StableRNG(9), 500)
        kw = (covariates=xs, num_trees=100)
        a = causal_forest(df, :y, :w; kw..., rng=StableRNG(10))
        b = causal_forest(df, :y, :w; kw..., rng=StableRNG(10), parallel=false)
        @test predict(a) == predict(b)
        @test predict_interval(a) == predict_interval(b; parallel=false)
        perm = randperm(StableRNG(11), nrow(df))
        c = causal_forest(df[perm, :], :y, :w; kw..., rng=StableRNG(10))
        @test predict(c) == predict(a)[perm]
        @test coef(average_treatment_effect(c)) ≈ coef(average_treatment_effect(a))
        d = causal_forest(df, :y, :w; kw..., rng=StableRNG(12))
        @test predict(d) != predict(a)
        # matrix and data-frame front ends agree
        m = causal_forest(Matrix(df[:, xs]), df.y, df.w; num_trees=100, rng=StableRNG(10))
        @test predict(m) == predict(a)
        @test predict(a, df[1:3, :]) == predict(a, Matrix(df[1:3, xs]))
    end

    @testset "accuracy, variance estimates and missing covariates" begin
        df = grf_data(StableRNG(13), 1500; confounded=false)
        rf = regression_forest(df, :y; covariates=xs, num_trees=300, rng=StableRNG(14))
        @test cor(predict(rf), df.x3 .+ df.tau .* 0.5) > 0.6
        cf = causal_forest(df, :y, :w; covariates=xs, num_trees=500, rng=StableRNG(15))
        @test cor(predict(cf), df.tau) > 0.9
        pi = predict_interval(cf, DataFrame(x1=[0.1, 0.9], x2=0.5, x3=0.5, x4=0.5);
                              level=0.9)
        @test all(pi.variance .> 0) && pi.std_error ≈ sqrt.(pi.variance)
        @test pi.conf_high .- pi.conf_low ≈ 2 .* critical_value(0.9) .* pi.std_error
        @test pi.estimate[2] > pi.estimate[1]
        @test nrow(predict_interval(cf)) == nrow(df)
        # missing values in a covariate are handled by the splits (MIA)
        dm = copy(df)
        dm.x2 = Vector{Union{Missing,Float64}}(dm.x2)
        dm.x2[1:200] .= missing
        cm = causal_forest(dm, :y, :w; covariates=xs, num_trees=200, rng=StableRNG(16))
        @test all(isfinite, predict(cm))
        @test all(isfinite, predict(cm, dm[1:10, :]))
        @test_throws ArgumentError causal_forest(dm, :y, :w; covariates=xs,
                                                 y_hat=OLSLearner(), num_trees=50)
        # learner-based local centering is cross-fitted
        cl = causal_forest(df, :y, :w; covariates=xs, num_trees=200,
                           y_hat=OLSLearner(), w_hat=LogisticLearner(), rng=StableRNG(17))
        @test occursin("cross-fitted", cl.nuisance[1].second)
        @test cor(predict(cl), df.tau) > 0.8
        # tuning by out-of-bag error
        ct = causal_forest(df[1:600, :], :y, :w; covariates=xs, num_trees=100,
                           tune_parameters=[:min_node_size, :alpha], tune_num_trees=50,
                           tune_num_reps=10, rng=StableRNG(18))
        @test ct.tuning.status in ("tuned", "default", "failure")
        @test keys(ct.tuning.params) == (:min_node_size, :alpha)
        @test ct.params.alpha == ct.tuning.params.alpha
        rt = regression_forest(df[1:600, :], :y; covariates=xs, num_trees=100,
                               tune_parameters=:all, tune_num_trees=30, tune_num_reps=8,
                               rng=StableRNG(19))
        @test rt.tuning !== nothing
        @test occursin("Tuning", sprint(show, MIME"text/plain"(), rt))
    end

    @testset "post-estimation interface" begin
        df = grf_data(StableRNG(20), 800)
        df.cl = repeat(1:80, inner=10)
        cf = causal_forest(df, :y, :w; covariates=xs, num_trees=300, cluster=:cl,
                           rng=StableRNG(21))
        a = average_treatment_effect(cf)
        @test a isa HTEEstimate && coefnames(a) == ["ATE"] && dof_residual(a) == 79
        @test coef(a)[1] ≈ mean(get_scores(cf))
        @test nrow(tidy(a)) == 1
        @test coefnames(average_treatment_effect(cf; target=:treated)) == ["ATT"]
        @test coefnames(average_treatment_effect(cf; target=:control)) == ["ATC"]
        sub = average_treatment_effect(cf; subset=df.x1 .> 0.5)
        @test nobs(sub) == count(df.x1 .> 0.5)
        b = best_linear_projection(cf, [:x1])
        @test b isa CATEProjection && coefnames(b) == ["(Intercept)", "x1"]
        @test dof_residual(b) == 79
        @test coef(cate_projection(cf; basis=[:x1])) == coef(b)
        t = test_calibration(cf)
        @test t isa DiagnosticTest && nrow(t.details.table) == 2
        Γ = double_robust_scores(cf)
        @test Γ[:, 2] .- Γ[:, 1] ≈ get_scores(cf)
        pv = policy_value(cf, predict(cf) .> median(predict(cf)))
        @test coefnames(pv)[1] == "value(policy)" && dof_residual(pv) == 79
        # value(treat all) - value(treat none) is the ATE
        pa = policy_value(cf, ones(Int, nrow(df)))
        @test coef(pa)[3] ≈ coef(a)[1] && abs(coef(pa)[2]) < 1e-12
        tree = policy_tree(Γ, Matrix(df[:, [:x1]]); depth=1, covariates=[:x1])
        @test policy_value(cf, tree) isa HTEEstimate
        r = rank_average_treatment_effect(cf, predict(cf); R=30, rng=StableRNG(22))
        @test r isa RATEEstimate && length(coef(r)) == 1 && r.R == 30
        @test nrow(r.toc) == 10 && r.toc.q[end] == 1.0
        r2 = rank_average_treatment_effect(cf, (predict(cf), df.x2); target=:QINI,
                                           R=20, rng=StableRNG(23))
        @test coefnames(r2)[3] == "priority1 - priority2 | QINI"
        @test coef(r2)[3] ≈ coef(r2)[1] - coef(r2)[2]
        @test size(vcov(r2)) == (3, 3)
        rs = rank_average_treatment_effect(get_scores(cf), predict(cf); R=20,
                                           cluster=df.cl, rng=StableRNG(24))
        @test coef(rs) ≈ coef(rank_average_treatment_effect(cf, predict(cf); R=2,
                                                            rng=StableRNG(1)))
        @test occursin("Causal forest", sprint(show, MIME"text/plain"(), cf))
        @test occursin("CausalForest", sprint(show, cf))
        # instrumental forest: LATE with an automatically fitted compliance forest
        ivd = grf_data(StableRNG(25), 800)
        ivd.z = Float64.(rand(StableRNG(26), 800) .< 0.5)
        u = rand(StableRNG(27), 800)
        ivd.d = Float64.(ifelse.(ivd.z .== 1, u .< 0.7, u .< 0.1))
        ivd.y = ivd.x3 .+ 2 .* ivd.d .+ randn(StableRNG(28), 800)
        ivf = instrumental_forest(ivd, :y, :d, :z; covariates=xs, num_trees=300,
                                  rng=StableRNG(29))
        la = average_treatment_effect(ivf; num_trees_for_weights=200)
        @test coefnames(la) == ["LATE"] && abs(coef(la)[1] - 2) < 4 * stderror(la)[1]
    end

    @testset "error paths" begin
        df = grf_data(StableRNG(30), 200)
        @test_throws ArgumentError causal_forest(df, :y, :w; covariates=Symbol[])
        @test_throws ArgumentError causal_forest(df, :y, :w; covariates=xs,
                                                 sample_fraction=0.7)
        @test_throws ArgumentError causal_forest(df, :y, :w; covariates=xs, mtry=5)
        @test_throws ArgumentError causal_forest(df, :y, :w; covariates=xs, alpha=0.3)
        @test_throws ArgumentError causal_forest(df, :y, :w; covariates=xs,
                                                 honesty_fraction=1.0)
        @test_throws ArgumentError causal_forest(df, :y, :w; covariates=xs,
                                                 tune_parameters=[:nope])
        @test_throws ArgumentError causal_forest(df, :y, :nope; covariates=xs)
        df.c = ones(nrow(df))
        @test_throws ArgumentError causal_forest(df, :y, :c; covariates=xs)
        @test_throws ArgumentError causal_forest(df, :y, :w; covariates=xs,
                                                 equalize_cluster_weights=true)
        @test_throws DimensionMismatch causal_forest(rand(10, 2), rand(9), rand(10))
        rf = regression_forest(df, :y; covariates=xs, num_trees=20, ci_group_size=1,
                               rng=StableRNG(31))
        @test_throws ArgumentError predict_interval(rf)
        @test_throws DimensionMismatch predict(rf, rand(3, 2))
        @test_throws ArgumentError predict(rf, DataFrame(x1=[0.1]))
        cf = causal_forest(df, :y, :w; covariates=xs, num_trees=20, rng=StableRNG(32))
        @test_throws ArgumentError average_treatment_effect(cf; target=:nope)
        @test_throws ArgumentError best_linear_projection(cf, [:nope])
        @test_throws ArgumentError rank_average_treatment_effect(cf, predict(cf);
                                                                 q=[0.5, 0.9])
        @test_throws ArgumentError rank_average_treatment_effect(cf, predict(cf);
                                                                 target=:nope)
        @test_throws DimensionMismatch rank_average_treatment_effect(cf, [1.0, 2.0])
        @test_throws ArgumentError policy_value(cf, fill(2, nrow(df)))
        @test_throws ArgumentError average_treatment_effect(cf; subset=Int[])
        @test_throws ArgumentError split_frequencies(cf; max_depth=0)
        @test_throws ArgumentError ForestLearner() |> l -> fitpredict(l, rand(2, 1),
                                                                     rand(2), rand(1, 1))
    end

    @testset "ForestLearner" begin
        rng = StableRNG(33)
        X = rand(rng, 600, 3)
        y = sin.(3 .* X[:, 1]) .+ 0.1 .* randn(rng, 600)
        p = fitpredict(ForestLearner(num_trees=200), X, y, X; rng=StableRNG(1))
        @test cor(p, sin.(3 .* X[:, 1])) > 0.9
        @test p == fitpredict(ForestLearner(num_trees=200), X, y, X; rng=StableRNG(1))
        d = Float64.(rand(rng, 600) .< X[:, 2])
        pr = fitpredict_proba(ForestLearner(num_trees=100), X, d, X; rng=StableRNG(2))
        @test all(0 .<= pr .<= 1) && cor(pr, X[:, 2]) > 0.7
        @test_throws ArgumentError fitpredict_proba(ForestLearner(), X, y, X)
        dd = DataFrame(x1=X[:, 1], x2=X[:, 2], d=d, y=y .+ d)
        r = dml_irm(dd, :y, :d; covariates=[:x1, :x2],
                    outcome_learner=ForestLearner(num_trees=100),
                    propensity_learner=ForestLearner(num_trees=100), rng=StableRNG(3))
        @test abs(coef(r)[1] - 1) < 4 * stderror(r)[1]
    end
end
