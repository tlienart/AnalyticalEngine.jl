function fit!(glr::GLR,
    X::AbstractMatrix{T},
    y::AbstractVector{T};
    solver::String="default",
    kwargs...) where T<:Real

    n, p = size(X)

    glr.n_features = p
    glr.coefficients = fit_(glr, X, y, n, p, solver; kwargs...)
    glr
end

#=
OLS  regression
=#
function fit_(
    glr::GLR{LPDistLoss{2}, NoPenalty},
    X, y, n, p, solver;
    # arguments for FLUX
    update!::Function=(p->p), nsteps=10)

    if solver == "default"
        # XXX this is potentially very inefficient
        X_ = glr.fit_intercept ? hcat(ones(n), X) : X
        X_ \ y
    elseif lowercase(solver) == "flux"
        θ = param(randn(p))
        b = param(randn(1))

        params = glr.fit_intercept ? (b, θ) : θ
        predict(X) = glr.fit_intercept ? X * θ .+ b : X * θ
        loss(X, y) = sum((predict(X) .- y).^2) / n

        for i = 1:nsteps
            back!(loss(X, y))
            update!(params)
        end
        vcat(b.data, θ.data)
    else
        throw(UnimplementedException())
    end
end

#=
Ridge regression
=#
function fit_(
    glr::GLR{LPDistLoss{2}, SEP{T, L2Penalty}},
    X, y, n, p, solver) where T<:Real

    if solver == "default"
        # XXX this is potentially very inefficient
        X_, p_ = glr.fit_intercept ? (hcat(ones(n), X), p+1) : (X, p)
        (X_' * X_ + glr.penalty.λ * eye(p_)) \ (X_' * y)
    else
        throw(UnimplementedException())
    end
end
