include("_using.jl")

# ---------------
## OLS REGRESSION
# ---------------

ols = LinearRegression()
ols_no_intercept = LinearRegression(fit_intercept=false)

srand(1234)

n, p = 10, 3

X = randn(n, p)
y = randn(n)

fit!(ols, X, y)
fit!(ols_no_intercept, X, y)

@test ols.n_features == p
@test ols_no_intercept.n_features == p

@test isapprox(ols.coefficients, hcat(ones(n, 1), X) \ y)
@test isapprox(ols_no_intercept.coefficients, X \ y)

# -----------------
## RIDGE REGRESSION
# -----------------

λ = 2.0
ridge = RidgeRegression(λ, fit_intercept=false)

fit!(ridge, X, y)

@test isapprox(ridge.coefficients, (X' * X + λ * eye(p)) \ (X' * y))
