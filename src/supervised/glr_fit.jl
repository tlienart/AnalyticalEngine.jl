function fit!(glr::GLR, X::AbstractArray{S}, y::AbstractVector{T};
    solver::String="default", kwargs...) where {S <: Real, T <: Real}

    # retrieve the number of instances + features
    n, p = size(X)
    glr.n_features = p
    solver = lowercase(solver)

    β = []
    if solver == "flux"
        β = fit_flux(glr, X, y, n, p; kwargs...)
    else
        β = fit_(glr, X, y, n, p, solver; kwargs...)
    end
    glr.intercept, glr.coefs = glr.fit_intercept ? (β[1], β[2:end]) : (0, β)
    # return the fitted model
    glr
end

#=
OLS  regression
=#
function fit_(glr::GLR{LPDistLoss{2}, NoPenalty},
    X, y, n, p, solver;
    # arguments for FLUX
    update!::Function=(p->p), nsteps=10)

    if solver == "default"
        # XXX this is potentially very inefficient
        X_ = glr.fit_intercept ? hcat(ones(n), X) : X
        β = X_ \ y
    else
        throw(UnimplementedException())
    end
end

#=
Ridge regression
=#
function fit_(
    glr::GLR{LPDistLoss{2}, ScaledPenalty{L2Penalty}},
    X, y, n, p, solver)

    if solver == "default"
        # XXX this is potentially very inefficient
        X_, p_ = glr.fit_intercept ? (hcat(ones(n), X), p+1) : (X, p)
        (X_' * X_ + glr.penalty.scale * eye(p_)) \ (X_' * y)
    else
        throw(UnimplementedException())
    end
end

#=
Generalized Linear Regression with FLUX

NOTE will only work if things are differentiable and there it may be better to
just use analytical gradients (e.g.: logit).
=#
# function fit_flux(glr, X, y, n, p;
#     update!::Function=(p->p), nsteps=10)
#
#     θ = param(randn(p))
#     b = param(randn(1))
#
#     params = glr.fit_intercept ? (b, θ) : θ
#
#     predict(X) = glr.fit_intercept ? X * θ .+ b : X * θ
#     loss(X, y) = sum(glr.loss(predict(X) - y))/n +
#                  value(glr.penalty, θ)
#
#     for i = 1:nsteps
#         back!(loss(X, y))
#         update!(params)
#     end
#     vcat(b.data, θ.data)
# end
