# The MLJ package extension. Only MLJModelInterface and a small model package are
# test dependencies, so MLJ's full data interface (needed for classifiers) is not
# loaded here: regressors are tested end to end and the classifier path must fail
# with an informative error.

import MLJModelInterface
import MLJDecisionTreeInterface

@testset "MLJ extension" begin
    @test Base.get_extension(DrSnow, :DrSnowMLJExt) !== nothing
    rng = StableRNG(91)
    n = 400
    X = randn(rng, n, 3)
    d = Float64.(rand(rng, n) .< 0.5)
    y = d .+ X[:, 1] .^ 2 .+ randn(rng, n)
    forest = MLJLearner(MLJDecisionTreeInterface.RandomForestRegressor(n_trees=10))
    a = fitpredict(forest, X, y, X[1:5, :]; rng=StableRNG(1))
    @test a == fitpredict(forest, X, y, X[1:5, :]; rng=StableRNG(1))
    @test a != fitpredict(forest, X, y, X[1:5, :]; rng=StableRNG(2))
    # the wrapped model is not mutated
    @test forest.model.rng isa Random.AbstractRNG || forest.model.rng isa Integer
    tree = MLJLearner(MLJDecisionTreeInterface.DecisionTreeRegressor(max_depth=2))
    p = fitpredict_proba(tree, X, d, X)
    @test all(0 .<= p .<= 1)
    @test_throws ArgumentError fitpredict_proba(tree, X, y, X)
    @test_throws ArgumentError fitpredict(tree, zeros(n, 0), y, zeros(2, 0))
    clf = MLJLearner(MLJDecisionTreeInterface.DecisionTreeClassifier())
    @test_throws ArgumentError fitpredict(clf, X, d, X)
    if MLJModelInterface.get_interface_mode() isa MLJModelInterface.LightInterface
        err = try
            fitpredict_proba(clf, X, d, X)
            nothing
        catch e
            e
        end
        @test err isa ArgumentError && occursin("MLJBase", err.msg)
    end
    @test DrSnow._ml_learner_name(forest) == "MLJLearner(RandomForestRegressor)"
    # end to end, reproducible, threads or not
    df = DataFrame(X, [:x1, :x2, :x3])
    df.d = d
    df.y = y
    r1 = dml_irm(df, :y, :d; covariates=[:x1, :x2, :x3], outcome_learner=forest,
                 propensity_learner=LogisticLearner(), rng=StableRNG(3))
    r2 = dml_irm(df, :y, :d; covariates=[:x1, :x2, :x3], outcome_learner=forest,
                 propensity_learner=LogisticLearner(), rng=StableRNG(3), parallel=false)
    @test coef(r1) == coef(r2) && stderror(r1) == stderror(r2)
    @test abs(coef(r1)[1] - 1) < 4 * stderror(r1)[1]
    # MLJ models in meta-learners and as causal-forest centering learners
    for f in (t_learner, x_learner, r_learner)
        kw = f === t_learner ? (;) : (propensity_learner=LogisticLearner(),)
        m = f(df, :y, :d; covariates=[:x1, :x2, :x3], outcome_learner=forest, kw...,
              rng=StableRNG(4))
        m2 = f(df, :y, :d; covariates=[:x1, :x2, :x3], outcome_learner=forest, kw...,
               rng=StableRNG(4), parallel=false)
        @test m.cate_oof == m2.cate_oof
        @test abs(m.ate - 1) < 0.5
    end
    cf = causal_forest(df, :y, :d; covariates=[:x1, :x2, :x3], y_hat=forest,
                       w_hat=LogisticLearner(), num_trees=200, rng=StableRNG(5))
    @test occursin("MLJLearner", cf.nuisance[1].second)
    @test abs(coef(average_treatment_effect(cf))[1] - 1) < 0.5
end
