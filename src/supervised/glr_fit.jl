function fit!(glr::GLR,
    X::AbstractArray{T},
    y::AbstractArray{T};
    solver::String="default",
    kwargs...) where T <: Real

    n, p = size(X)

    glr.n_features = p
    if lowercase(solver) == "flux"
        glr.coefficients = fit_flux(glr, X, y, n, p; kwargs...)
    else
        glr.coefficients = fit_(glr, X, y, n, p, lowercase(solver); kwargs...)
    end
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
