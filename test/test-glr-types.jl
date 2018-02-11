using AnalyticalEngine
using Base.Test

@testset "GLR-Types" begin

glr = GeneralizedLinearRegression()
ols = LinearRegression()
ridge = RidgeRegression()
lasso = LassoRegression()
logreg = LogisticRegression()

@test isa(glr, RegressionModel)
@test isa(glr, GLR{LPDistLoss{2}, NoPenalty})

@test isa(ols, GLR{LPDistLoss{2}, NoPenalty})
@test isa(ridge, GLR{LPDistLoss{2}, ScaledPenalty{L2Penalty}})
@test isa(lasso, GLR{LPDistLoss{2}, ScaledPenalty{L1Penalty}})
@test isa(logreg, GLR{LogisticLoss, ScaledPenalty{L2Penalty}})

end # testset
