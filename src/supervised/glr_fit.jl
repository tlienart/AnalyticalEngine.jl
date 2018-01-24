function fit!(glr::GLR,
    X::AbstractMatrix{T},
    y::AbstractVector{T};
    solver::String="default") where T<:Real

    glr.n_features = size(X, 2)

    # XXX this is potentially very inefficient
    X_ = glr.fit_intercept ? hcat(ones(T, size(X, 1)), X) : X

    glr.coefficients = fit_(glr, X_, y, solver)
    glr
end

#=
OLS  regression
=#
function fit_(
    glr::GLR{LPDistLoss{2}, NoPenalty}, X, y, solver)

    if solver == "default"
        X \ y
    else
        throw(UnimplementedException())
    end
end

#=
Ridge regression
=#
function fit_(
    glr::GLR{LPDistLoss{2}, SEP{T, L2Penalty}}, X, y, solver) where T<: Real

    if solver == "default"
        (X' * X + glr.penalty.λ * eye(size(X, 2))) \ (X' * y)
    else
        throw(UnimplementedException())
    end
end
