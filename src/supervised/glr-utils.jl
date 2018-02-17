
function score(glr::GLR,
    X::AbstractArray{S}, y::AbstractVector{T}) where {S <: Real, T <: Real}

    # retrieve relevant hyperparameters
    loss = get(glr.loss)
    penalty = get(glr.penalty)
    avg_loss = get(glr.avg_loss)

    loss = loss(predict(glr, X), y) / (avg_loss ? length(y) : 1)
    penalty = penalty(glr.coefs)

    return loss + penalty
end
