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

#=
LOGISTIC REGRESSION
=#

#= Sklearn for testing
import numpy as np
from sklearn.linear_model import LogisticRegression
X = np.array([
[0.8673472019512456, -0.5605013381807765, 1.5641682355362416],
[-0.9017438158568171, -0.019291781689849075, -1.3967353668333795],
[-0.4944787535042339, 0.12806443451512645, 1.1054978391059092],
[-0.9029142938652416, 1.852782957725545, -1.1067299135255761],
[0.8644013132535154, -0.8277634318169205, -3.2113596499239088],
[2.2118774995743475, 0.11009612632217552, -0.07401454242444336],
[0.5328132821695382, -0.2511757400198831, 0.1509756176321479],
[-0.27173539603462066, 0.3697140350317453, 0.7692782605345824],
[0.5023344963886675, 0.07211635315125874, -0.31015257323306406],
[-0.5169836206932686, -1.503429457351051, -0.6027068905147959]])
y = np.array([ 0.,  1.,  1.,  0.,  1.,  1.,  1.,  0.,  0.,  0.])
% lamba=2 -> C=0.5 but there's a factor 1/2 on top in their l2 penalty
lr = LogisticRegression(C=0.25, fit_intercept=False)
lr.fit(X, y)
=#

# coefficients returned by sklearn
# shouldn't be compared too closely as the methods are very different
# but, roughly speaking, we get similar coefficients.
skcoef = [0.20023141, -0.06924323, -0.22361009]

λ = 2.0
logreg_flux = LogisticRegression(λ, fit_intercept=false)
y_lr = (sign.(y) .+ 1)./2

fit!(logreg_flux, X, y_lr,
    solver="Flux", grad_step=basic_gd(0.1), nsteps=10)

@test norm(skcoef - logreg_flux.coefs) / norm(skcoef) <= 0.1

end # testset
