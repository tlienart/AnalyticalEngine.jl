using AnalyticalEngine
using Base.Test

@testset "GLR-Types" begin

glr = GeneralizedLinearRegression()
ols = LinearRegression()
ridge = RidgeRegression()
lasso = LassoRegression()
logreg = LogisticRegression()

@test isa(glr, RegressionModel)
@test isa(glr, GLR{L2DistLoss, NoPenalty})

@test isa(ols, GLR{L2DistLoss, NoPenalty})
@test isa(ridge, GLR{L2DistLoss, ScaledPenalty{L2Penalty}})
@test isa(lasso, GLR{L2DistLoss, ScaledPenalty{L1Penalty}})
@test isa(logreg, GLR{LogisticLoss, ScaledPenalty{L2Penalty}})

# GLR hyperparameters, get, set!, set and deepcopy

hp = hyperparameters(lasso)

@test isa(get(hp[:loss]), L2DistLoss)
@test isa(get(hp[:penalty]), ScaledPenalty{L1Penalty})

set!(lasso, Dict(:fit_intercept=>false, :avg_loss=>false))

@test get(lasso.fit_intercept) == false
@test get(lasso.avg_loss) == false

other_lasso = set(lasso, Dict(:fit_intercept=>true))

@test get(lasso.fit_intercept) == false
@test get(other_lasso.fit_intercept) == true

lasso3 = deepcopy(lasso)
set!(lasso3, Dict(:penalty=>2.0*L1Penalty()))
@test get(lasso3.penalty).scale == 2.0

# TODO ability to change this
# ridge_from_lasso = deepcopy(lasso)
# set!(ridge_from_lasso, Dict(:penalty=>2.0*L2Penalty()))


end # testset
