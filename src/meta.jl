chunk(xs, n) = collect(Iterators.partition(xs, ceil(Int, length(xs)/n)))

mutable struct Multiplex{M<:Tuple}
  models::M
  best::Int
end

Multiplex(models...) = Multiplex(models, 0)

function fit!(m::Multiplex, X, y)
  train, test = map(i -> (X[i,:],y[i]), chunk(1:size(X,1), 2))
  losses = map(m.models) do model
    fit!(model, train...)
    L2DistLoss()(predict(model, test[1]), test[2])
  end
  m.best = findmax(losses)[2]
  fit!(m.models[m.best], X, y)
  return m
end

predict(m::Multiplex, X) = predict(m.models[m.best], X)

# hyperparams(m::Multiplex) = union(hyperparams.(m.models)...)
