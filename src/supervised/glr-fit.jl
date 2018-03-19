"""
    fit!(glr::GLR, X::AbstractArray{S}, y::AbstractVector{T}, ...)

Parent call to fit a `GeneralizedLinearRegression` model.
"""
function fit!(glr::GLR, X::AbstractArray{<:Real}, y::AbstractVector{<:Real};
    solver::String="default", kwargs...)

    # retrieve the number of instances + features and solver description
    n, p = size(X)
    glr.n_features = p
    solver = lowercase(solver)

    # retrieve loss and penalty, deal accordingly
    loss = glr.loss
    penalty = glr.penalty

    if solver == "flux"
        # generic solver for differentiable programs
        β = fit_flux(loss, penalty, glr, X, y, n, p; kwargs...)
    else
        # specific solvers for different sub-cases
        β = fit_(loss, penalty, glr, X, y, n, p, solver; kwargs...)
    end

    glr.intercept, glr.coefs = glr.fit_intercept ? (β[1], β[2:end]) : (0, β)

    return glr
end

#=
OLS  regression

Loss: L2Loss
Penalty: NoPenalty
=#
function fit_(::ScaledLoss{L2DistLoss}, ::NoPenalty,
    glr, X, y, n, p, solver)

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

Loss: L2Loss
Penalty: L2Penalty
=#
function fit_(::ScaledLoss{L2DistLoss}, penalty::ScaledPenalty{L2Penalty},
    glr, X, y, n, p, solver)

    if solver ∈ ["default", "analytical"]
        # XXX this is potentially very inefficient
        X_, p_ = glr.fit_intercept ? (hcat(ones(n), X), p+1) : (X, p)
        # depending on avgloss/avgpenalty, the objective function should take
        # n and p into account which all amounts to changing λ
        σ_gap = penalty.scale
        σ_gap *= (glr.avg_loss ? n : 1)
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
function fit_flux(loss, penalty, glr, X, y, n, p;
    grad_step::Union{Void, Function}=nothing, nsteps=10, showscore=false)

    @assert grad_step != nothing "You need to specify an update mechanism"

    θ = param(randn(p))
    b = param(randn(1))

    params = glr.fit_intercept ? (b, θ) : (θ, )

    scale_loss = glr.avg_loss ? length(y) : 1

    predict(X) = glr.fit_intercept ? (X * θ .+ b) : (X * θ)
    score(X, y) = loss(predict(X), y) / scale_loss + penalty(θ)

    for i = 1:nsteps
        back!(score(X, y))
        params = grad_step(params, i)
        if showscore
            @show score(X, y)
        end
    end

    return glr.fit_intercept ? vcat(b.data, θ.data) : θ.data
end
