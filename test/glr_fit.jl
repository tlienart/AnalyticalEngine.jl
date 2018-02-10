using AnalyticalEngine
using Base.Test

@testset "GLR-Fit" begin

#=
OLS REGRESSION (analytical)
=#

ols = LinearRegression()
ols_no_intercept = LinearRegression(fit_intercept=false)

srand(1234)

n, p = 10, 3

X = randn(n, p)
X_ = hcat(ones(n, 1), X)
y = randn(n)

β = X_ \ y
intercept, coefs = β[1], β[2:end]
coefs_noi = X \ y

fit!(ols, X, y)
fit!(ols_no_intercept, X, y)

@test ols.n_features == p
@test ols_no_intercept.n_features == p

@test isapprox(ols.intercept, intercept)
@test isapprox(ols.coefs, coefs)
@test isapprox(ols_no_intercept.coefs, coefs_noi)

#=
OLS REGRESSION (FLUX)
=#

# basic gradient descent scheme
function basic_gd(δ=0.1, scale=false)
    function update!(params, i)
        for param in params
            param.data .-= δ*param.grad / (scale ? sqrt(i) : 1)
            param.grad .= 0
        end
        params
    end
end

ols_flux = LinearRegression()

fit!(ols_flux, X, y,
    solver="Flux", grad_step=basic_gd(), nsteps=50)

@test isapprox(
    ols.loss(ols(X), y),
    ols.loss(ols_flux(X), y),
    atol=1e-4)

#=
RIDGE REGRESSION
=#

λ = 2.0
ridge = RidgeRegression(λ, fit_intercept=false)
σ_gap = λ * n # avgloss, not avgpenalty
ridge_coefs_noi = (X' * X + σ_gap * eye(p)) \ (X' * y)

fit!(ridge, X, y)

@test isapprox(ridge.coefs, ridge_coefs_noi)

ridge_flux = RidgeRegression(λ, fit_intercept=false)

fit!(ridge_flux, X, y,
    solver="Flux", grad_step=basic_gd(0.1), nsteps=25)

# for this one, Flux converges to pretty much exactly the analytical solution
@test isapprox(
    ridge.loss(ridge(X), y) / n + ridge.penalty(ridge.coefs),
    ridge.loss(ridge_flux(X), y) / n + ridge.penalty(ridge_flux.coefs),
    atol=1e-12)

end # testset
