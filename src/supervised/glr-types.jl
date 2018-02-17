#=
GENERALIZED LINEAR REGRESSION (GLR) models

These are models with objective function:

    Loss(y, f(Xθ)) + λ Penalty(θ)

where y is a (n,) vector, X an (n, p) matrix, θ a (p, ) vector. Loss(y, ŷ)
is a function that measures the loss for a predicted ŷ and Penalty(θ) is a
penalty on the candidate coefficient vector θ.
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

``L(y, Xθ) + λ P(θ)``

where ``L`` is a loss function and ``P`` a penalty.

Specific cases include:

* **OLS regression**: L2 loss, no penalty
* **Ridge regression**: L2 loss, L2 penalty
* **Lasso regression**: L2 loss, L1 penalty
* **Logit/Probit regression**: Logit/Probit loss, no/L1/L2 penalty
"""
mutable struct GeneralizedLinearRegression <: RegressionModel
    # Parameters that can be tuned
    loss::Loss           # L(y, ŷ) where ŷ=Xθ
    penalty::Penalty     # R(θ) contains the scaling
    fit_intercept::Bool  # add intercept ? def=true
    avg_loss::Bool       # avg loss ? def=true
    # Fitted quantities
    n_features::Union{Void, Int}
    intercept::Union{Void, Real}
    coefs::Union{Void, AbstractVector{Real}}
end

const GLR = GeneralizedLinearRegression

# constructor
function GeneralizedLinearRegression(;
    loss=L2DistLoss(),
    penalty=NoPenalty(),
    fit_intercept=true,
    avg_loss=true)

    GeneralizedLinearRegression(
        loss,
        penalty,
        fit_intercept,
        avg_loss,
        nothing,        # un-assigned number of features
        nothing,        # un-assigned intercept
        nothing)        # un-assigned coefficients
end


# function that returns symbols corresponding to hyperparameters
hyperparameters(glr::GLR) =
    (:loss, :penalty, :fit_intercept, :avg_loss)

# function to copy a GLR object
copy(glr::GLR) = GeneralizedLinearRegression(
    map(deepcopy, (glr.loss, glr.penalty, glr.fit_intercept, glr.avg_loss,
                   glr.n_features, glr.intercept, glr.coefs))...)

#= ---------------------------------------------------------------------------
CONSTRUCTORS FOR STANDARD GLR MODELS
* OLS regression
* RIDGE regression
* LASSO regression
--------------------------------------------------------------------------- =#

"""
    LinearRegression

Generalized Linear Regression model with objective function

``|y-Xθ|₂``
"""
function LinearRegression(;
    fit_intercept::Bool=true,
    avg_loss::Bool=true)

    GeneralizedLinearRegression(
        fit_intercept=fit_intercept)
end


"""
    RidgeRegression

Generalized Linear Regression model with objective function

``|y-Xθ|₂ + λ|θ|₂``
"""
function RidgeRegression(λ::Real=1.0;
    fit_intercept::Bool=true,
    avg_loss::Bool=true)

    GeneralizedLinearRegression(
        penalty=L2Penalty() * λ,
        fit_intercept=fit_intercept,
        avg_loss=avg_loss)
end


"""
    LassoRegression

Generalized Linear Regression model with objective function

``|y - Xθ|₂ + λ|θ|₁``
"""
function LassoRegression(λ::Real=1.0;
    fit_intercept::Bool=true,
    avg_loss::Bool=true)

    GeneralizedLinearRegression(
        penalty=L1Penalty() * λ,
        fit_intercept=fit_intercept,
        avg_loss=avg_loss)
end


"""
    LogisticRegression
"""
function LogisticRegression(λ::Real=1.0;
    loss=LogisticLoss(),
    penalty::Union{NoPenalty, L1Penalty, L2Penalty}=L2Penalty(),
    fit_intercept::Bool=true,
    avg_loss::Bool=false) # it's usually not the averaged loss that's used

    GeneralizedLinearRegression(
        loss=loss,
        penalty=penalty * λ,
        fit_intercept=fit_intercept,
        avg_loss=avg_loss)
end
