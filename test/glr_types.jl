include("_using.jl")

glr = GeneralizedLinearRegression()
ols = LinearRegression()
ridge = RidgeRegression()
lasso = LassoRegression()

@test isa(glr, RegressionModel)
@test isa(glr, GLR{LPDistLoss{2}, NoPenalty})

@test isa(ols, GLR{LPDistLoss{2}, NoPenalty})
@test isa(ridge, GLR{LPDistLoss{2}, SEP{T, L2Penalty}} where T<:Real)
@test isa(lasso, GLR{LPDistLoss{2}, SEP{T, L1Penalty}} where T<:Real)
