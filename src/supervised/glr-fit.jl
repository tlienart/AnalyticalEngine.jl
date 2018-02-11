function fit!(glr::GLR, X::AbstractArray{S}, y::AbstractVector{T};
    solver::String="default", kwargs...) where {S <: Real, T <: Real}

    # retrieve the number of instances + features
    n, p = size(X)
    glr.n_features = p
    solver = lowercase(solver)

    if solver == "flux"
        β = fit_flux(glr, X, y, n, p; kwargs...)
    else
        β = fit_(glr, X, y, n, p, solver; kwargs...)
    end

    glr.intercept, glr.coefs = glr.fit_intercept ? (β[1], β[2:end]) : (0, β)

    return glr
end

#=
OLS  regression
=#
function fit_(glr::GLR{LPDistLoss{2}, NoPenalty},
    X, y, n, p, solver)

    if solver ∈ ["default", "analytical"]
        # XXX this is potentially very inefficient
        X_ = glr.fit_intercept ? hcat(ones(n), X) : X
        β = X_ \ y
    else
        throw(UnimplementedException())
    end

    return β
end

#=
Ridge regression
=#
function fit_(glr::GLR{LPDistLoss{2}, ScaledPenalty{L2Penalty}},
    X, y, n, p, solver)

    if solver ∈ ["default", "analytical"]
        # XXX this is potentially very inefficient
        X_, p_ = glr.fit_intercept ? (hcat(ones(n), X), p+1) : (X, p)
        # depending on avgloss/avgpenalty, the objective function should take
        # n and p into account which all amounts to changing λ
        σ_gap = glr.penalty.scale
        σ_gap *= (glr.avgloss ? n : 1)
        σ_gap /= (glr.avgpenalty ? p : 1)
        β = (X_' * X_ + σ_gap * eye(p_)) \ (X_' * y)
    else
        throw(UnimplementedException())
    end

    return β
end

#=
Generalized Linear Regression with FLUX

NOTE will only work if things are differentiable and there it may be better to
just use analytical gradients (e.g.: logit).
=#
function fit_flux(glr, X, y, n, p;
    grad_step::Union{Void, Function}=nothing, nsteps=10)

    @assert typeof(grad_step) != Void "You need to specify an update mechanism"

    θ = param(randn(p))
    b = param(randn(1))

    params = glr.fit_intercept ? (b, θ) : (θ, )

    scale_l = glr.avgloss ? n : 1
    scale_p = glr.avgpenalty ? p : 1

    predict(X) = glr.fit_intercept ? (X * θ .+ b) : (X * θ)
    objfun(X, y) = glr.loss(predict(X), y)/scale_l + glr.penalty(θ)/scale_p

    for i = 1:nsteps
        back!(objfun(X, y))
        @show objfun(X, y)
        params = grad_step(params, i)
    end

    return glr.fit_intercept ? vcat(b.data, θ.data) : θ.data
end
