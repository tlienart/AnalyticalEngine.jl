module AnalyticalEngine

using Flux.Tracker # Needs version > 0.4.1 (for norm)
using NNlib
import Flux.Tracker: @back

logsigmoid(xs::TrackedArray) = track(NNLib.logsigmoid, xs)

back(::typeof(logsigmoid), Δ, xs) = @back(xs, NNLib.∇logsigmoid(Δ, data(xs)))


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
