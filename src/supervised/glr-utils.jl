
function score(glr::GLR, X::AbstractArray{<:Real}, y::AbstractVector{<:Real})

    loss = glr.loss(predict(glr, X), y) / (glr.avg_loss ? length(y) : 1)
    penalty = glr.penalty(glr.coefs)

    return loss + penalty
end
