include("_using.jl")

@testset "Fit GLR" begin

# ---------------
## OLS REGRESSION (analytical)
# ---------------

ols = LinearRegression()
ols_no_intercept = LinearRegression(fit_intercept=false)

srand(1234)

n, p = 10, 3

X = randn(n, p)
X_ = hcat(ones(n, 1), X)
y = randn(n)

coefs = X_\y
coefs_noi = X\y

fit!(ols, X, y)
fit!(ols_no_intercept, X, y)

@test ols.n_features == p
@test ols_no_intercept.n_features == p

@test isapprox(ols.coefficients, coefs)
@test isapprox(ols_no_intercept.coefficients, coefs_noi)

# ---------------
## OLS REGRESSION (FLUX)
# ---------------

function update!(params) # basic gradient descent scheme
    for param in params
        param.data .-= 0.1param.grad
        param.grad .= 0
    end
    params
end

fit!(ols, X, y, solver="Flux", update! = update!, nsteps=30)

@test norm(ols.coefficients - coefs) <= 0.05

# -----------------
## RIDGE REGRESSION
# -----------------

λ = 2.0
ridge = RidgeRegression(λ, fit_intercept=false)

coefs_noi = (X' * X + λ * eye(p)) \ (X' * y)

fit!(ridge, X, y)

@test isapprox(ridge.coefficients, coefs_noi)

end # testset
