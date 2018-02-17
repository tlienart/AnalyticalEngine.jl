function predict(glr::GLR, X::AbstractArray{T}) where T<:Real

    is_fitted = !any((glr.n_features, glr.intercept, glr.coefs) .== nothing)
    @assert is_fitted "Model was not fitted"

    p = size(X, 2)
    @assert p == glr.n_features "Dimension mismatch"

    if get(glr.fit_intercept)
        X * glr.coefs .+ glr.intercept
    else
        X * glr.coefs
    end
end


# convenience function to use a model directly as a function
(glr::GLR)(X) = predict(glr, X)
