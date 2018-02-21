using AnalyticalEngine
using Base.Test

@testset "GLR-Types" begin

glr = GeneralizedLinearRegression()
ols = LinearRegression()
ridge = RidgeRegression()
lasso = LassoRegression()
logreg = LogisticRegression()

@test isa(glr, RegressionModel)
@test isa(glr.loss, ScaledLoss{L2DistLoss})
@test isa(glr.penalty, NoPenalty)

@test isa(ols.loss, ScaledLoss{L2DistLoss})
@test isa(ols.penalty, NoPenalty)

@test isa(ridge.loss, ScaledLoss{L2DistLoss})
@test isa(ridge.penalty, ScaledPenalty{L2Penalty})

@test isa(lasso.loss, ScaledLoss{L2DistLoss})
@test isa(lasso.penalty, ScaledPenalty{L1Penalty})

@test isa(logreg.loss, ScaledLoss{LogisticLoss})
@test isa(logreg.penalty, ScaledPenalty{L2Penalty})

# GLR hyperparameters, set!, set and deepcopy

hp = hyperparameters(lasso)

@test issubset(hp, (:loss, :penalty, :fit_intercept, :avg_loss))

set!(lasso, fit_intercept=false, avg_loss=false)

@test lasso.fit_intercept == false
@test lasso.avg_loss == false

other_lasso = set(lasso, fit_intercept=true)

@test lasso.fit_intercept == false
@test other_lasso.fit_intercept == true

lasso3 = deepcopy(lasso)
set!(lasso3, penalty=2.0*L1Penalty())
@test lasso3.penalty.scale == 2.0

# TODO ability to change this
ridge_from_lasso = deepcopy(lasso)
set!(ridge_from_lasso, penalty=L2Penalty())

@test isa(ridge_from_lasso.penalty, ScaledPenalty{L2Penalty})
@test isa(lasso.penalty, ScaledPenalty{L1Penalty})

end # testset
