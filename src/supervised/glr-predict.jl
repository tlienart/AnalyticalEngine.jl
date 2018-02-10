function predict(glr::GLR, X::AbstractArray{T}) where T<:Real

    p = size(X, 2)
    @assert p == glr.n_features "Dimension mismatch"

    if glr.fit_intercept
        X * glr.coefs .+ glr.intercept
    else
        X * glr.coefs
    end
end
(glr::GLR)(X) = predict(glr, X) 
