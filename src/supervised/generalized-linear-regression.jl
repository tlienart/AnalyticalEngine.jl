#=
Generalized Linear Regression models with objective function:

    Loss(y, f(Xθ)) + λ Penalty(θ)
=#

export
    GeneralizedLinearRegression, GLR,
    LinearRegression,
    LassoRegression,
    RidgeRegression,
    LogisticRegression


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
        zero(Int64),
        zeros(Real, 0))
end


function LinearRegression(;
    fit_intercept::Bool=true)

    GeneralizedLinearRegression(
        fit_intercept=fit_intercept)
end


function RidgeRegression(
    λ::Real=1.0;
    fit_intercept::Bool=true)

    GeneralizedLinearRegression(
        penalty=scaled(L2Penalty(), λ),
        fit_intercept=fit_intercept)
end


function LassoRegression(
    λ::Real=1.0;
    fit_intercept::Bool=true)

    GeneralizedLinearRegression(
        penalty=scaled(L1Penalty(), λ),
        fit_intercept=fit_intercept)
end
