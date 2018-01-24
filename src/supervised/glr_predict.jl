function predict(glr::GLR, X::AbstractMatrix{T}) where T<:Real

    n = size(X, 1)

    if glr.fit_intercept
        X * glr.coefficients[2:end] + glr.coefficients[1] * ones(T, n)
    else
        X * glr.coefs
    end
end
