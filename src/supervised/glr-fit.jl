"""
    fit!(glr::GLR, X::AbstractArray{S}, y::AbstractVector{T}, ...)

Parent call to fit a `GeneralizedLinearRegression` model.
"""
function fit!(glr::GLR, X::AbstractArray{S}, y::AbstractVector{T};
    solver::String="default", kwargs...) where {S <: Real, T <: Real}

    # retrieve the number of instances + features and solver description
    n, p = size(X)
    glr.n_features = p
    solver = lowercase(solver)

    fit_intercept = get(glr.fit_intercept)

    if solver == "flux"
        # generic solver for differentiable programs
        β = fit_flux(glr, X, y, n, p, fit_intercept; kwargs...)
    else
        # specific solvers for different sub-cases
        β = fit_(glr, X, y, n, p, fit_intercept, solver; kwargs...)
    end

    glr.intercept, glr.coefs = fit_intercept ? (β[1], β[2:end]) : (0, β)

    return glr
end

#=
OLS  regression
=#
function fit_(glr::GLR{L2DistLoss, NoPenalty},
    X, y, n, p, fit_intercept, solver)

    if solver ∈ ["default", "analytical"]
        # XXX this is potentially very inefficient
        X_ = fit_intercept ? hcat(ones(n), X) : X
        β = X_ \ y
    else
        throw(UnimplementedException())
    end

    return β
end

#=
Ridge regression
=#
function fit_(glr::GLR{L2DistLoss, ScaledPenalty{L2Penalty}},
    X, y, n, p, fit_intercept, solver)

    # retrieve relevant hyperparameters
    penalty = get(glr.penalty)
    avg_loss = get(glr.avg_loss)

    if solver ∈ ["default", "analytical"]
        # XXX this is potentially very inefficient
        X_, p_ = fit_intercept ? (hcat(ones(n), X), p+1) : (X, p)
        # depending on avgloss/avgpenalty, the objective function should take
        # n and p into account which all amounts to changing λ
        σ_gap = penalty.scale
        σ_gap *= (avg_loss ? n : 1)
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
function fit_flux(glr, X, y, n, p, fit_intercept;
    grad_step::Union{Void, Function}=nothing, nsteps=10, showscore=false)

    @assert grad_step != nothing "You need to specify an update mechanism"

    # retrieve relevant hyperparameters
    loss = get(glr.loss)
    penalty = get(glr.penalty)
    avg_loss = get(glr.avg_loss)

    θ = param(randn(p))
    b = param(randn(1))

    params = fit_intercept ? (b, θ) : (θ, )

    scale_loss = (avg_loss ? length(y) : 1)

    predict(X) = fit_intercept ? (X * θ .+ b) : (X * θ)
    score(X, y) = loss(predict(X), y) / scale_loss + penalty(θ)

    for i = 1:nsteps
        back!(score(X, y))
        params = grad_step(params, i)
        if showscore
            @show score(X, y)
        end
    end

    return fit_intercept ? vcat(b.data, θ.data) : θ.data
end
