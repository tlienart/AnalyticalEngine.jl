using Base.Test

@testset "GLR-Types" begin

glr = GeneralizedLinearRegression()
ols = LinearRegression()
ridge = RidgeRegression()
lasso = LassoRegression()

@test isa(glr, RegressionModel)
@test isa(glr, GLR{LPDistLoss{2}, NoPenalty})

@test isa(ols, GLR{LPDistLoss{2}, NoPenalty})
@test isa(ridge, GLR{LPDistLoss{2}, ScaledPenalty{L2Penalty}})
@test isa(lasso, GLR{LPDistLoss{2}, ScaledPenalty{L1Penalty}})

end # testset
