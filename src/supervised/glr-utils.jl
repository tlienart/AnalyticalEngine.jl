
function score(glr::GLR,
    X::AbstractArray{S}, y::AbstractVector{T}) where {S <: Real, T <: Real}

    loss = glr.loss(predict(glr, X), y) / (glr.avgloss ? length(y) : 1)
    penalty = glr.penalty(glr.coefs)
    
    return loss + penalty
end
