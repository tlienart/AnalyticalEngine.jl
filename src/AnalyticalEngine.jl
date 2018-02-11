module AnalyticalEngine

using Flux.Tracker
using NNlib
import Flux.Tracker: @back
import NNlib: logsigmoid, ∇logsigmoid
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
