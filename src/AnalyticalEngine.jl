module AnalyticalEngine

using Flux.Tracker
#= ----------------------------------------------------------------------------
Things that should really be defined by either Flux.Tracker or NNLib and that
possibly are but not in tagged-versions and therefore currently not accessible.
=#
# from NNLib -- sigmoid and log sigmoid
sigmoid(x) = one(x) / (one(x) + exp(-x))
function logsigmoid(x)
  max_v = max(zero(x), -x)
  z = exp(-max_v) + exp(-x-max_v)
  -(max_v + log(z))
end
∇logsigmoid(Δ, x) = Δ * (1 - sigmoid(x))

# Extending Flux.Tracker
import Flux.Tracker: @back
logsigmoid(xs::TrackedArray) = track(logsigmoid, xs)
back(::typeof(logsigmoid), Δ, xs) = @back(xs, ∇logsigmoid(Δ, data(xs)))

# -----------------------------------------------------------------------------

export fit!, predict

struct UnimplementedException <: Exception end

include("types.jl")

## Loss functions and penalties

include("mlfun/loss-penalty-types.jl")
include("mlfun/loss-penalty-functions.jl")
include("mlfun/loss-penalty-utils.jl")

## SUPERVISED MODELS

include("supervised/glr-types.jl")     # generalized linear regression
include("supervised/glr-fit.jl")
include("supervised/glr-predict.jl")


end # AnalyticalEngine
