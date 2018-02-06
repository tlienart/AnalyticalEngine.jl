#=
Generalized Linear Regression (GLR) models with objective function:

    Loss(y, f(Xθ)) + λ Penalty(θ)

=#

export
    GeneralizedLinearRegression, GLR,
    LinearRegression,
    LassoRegression,
    RidgeRegression,
    LogisticRegression
    

"""
    GeneralizedLinearRegression{L<:Loss, P<:Penalty} <: RegressionModel

Generalized Linear Regression (GLR) model with objective function:


``L(y, f(Xθ)) + λ P(θ)``

where ``L`` is a loss function, ``P`` a penalty and ``f`` is a function.

Specific cases include:

* **OLS regression**: L2 loss, no penalty, identity ``f``
* **Ridge regression**: L2 loss, L2 penalty, identity ``f``
* **Lasso regression**: L2 loss, L1 penalty, identity ``f``
* **Logit/Probit regression**: L2 Loss, no/L1/L2 penalty, logit/probit ``f``
"""
mutable struct GeneralizedLinearRegression{L<:Loss, P<:Penalty} <: RegressionModel
    loss::L     # L(y, ŷ) where ŷ=Xθ
    penalty::P  # R(θ) contains the scaling
    fit_intercept::Bool
    n_features::Int
    coefficients::AbstractVector{Real}
end

# short alias
const GLR{L, P} = GeneralizedLinearRegression{L, P}


function GeneralizedLinearRegression(;
    loss=L2DistLoss(),
    penalty=NoPenalty(),
    fit_intercept=true)

    GeneralizedLinearRegression(
        loss,
        penalty,
        fit_intercept,
        zero(Int64),       # un-assigned number of features
        zeros(Real, 0))    # un-assigned coefficients
end


"""
    LinearRegression

Generalized Linear Regression model with objective function

``|y-Xθ|_2``
"""
function LinearRegression(;
    fit_intercept::Bool=true)

    GeneralizedLinearRegression(
        fit_intercept=fit_intercept)
end


"""
    RidgeRegression

Generalized Linear Regression model with objective function

``|y-Xθ|_2 + λ|θ|_2``
"""
function RidgeRegression(
    λ::Real=1.0;
    fit_intercept::Bool=true)

    GeneralizedLinearRegression(
        penalty=λ * L2Penalty(),
        fit_intercept=fit_intercept)
end


"""
    LassoRegression

Generalized Linear Regression model with objective function

``|y - Xθ|_2 + λ|θ|_1``
"""
function LassoRegression(
    λ::Real=1.0;
    fit_intercept::Bool=true)

    GeneralizedLinearRegression(
        penalty=λ * L1Penalty(),
        fit_intercept=fit_intercept)
end
