using AnalyticalEngine
using Base.Test

@testset "GLR-Predict" begin

srand(523)
n, p = 20, 3
X, y = randn(n, p), randn(n)
Xp = randn(n, p)

lr = LinearRegression(fit_intercept=false)
lr_no = LinearRegression()
fit!(lr, X, y)
fit!(lr_no, X, y)

@test predict(lr, Xp) ≈ Xp * lr.coefs
@test predict(lr_no, Xp) ≈ Xp * lr_no.coefs .+ lr_no.intercept

end # testsets
