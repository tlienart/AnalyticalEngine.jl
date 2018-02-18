module AnalyticalEngine

# -----------------------------------------------------------------------------
# Extension to Flux and NNLib. These should eventually be integrated in those
# libraries by @mikeinnes
using Flux.Tracker
using NNlib
import Flux.Tracker: @back
import NNlib: logsigmoid, ∇logsigmoid
logsigmoid(xs::TrackedArray) = track(logsigmoid, xs)
back(::typeof(logsigmoid), Δ, xs) = @back(xs, ∇logsigmoid(Δ, data(xs)))
# -----------------------------------------------------------------------------

export
       # these functions should be implemented for all M <: SupervisedModel
       fit!, predict, score,
       set!, set, hyperparameters

struct UnimplementedException <: Exception end

include("types.jl")

## Loss functions and penalties

include("mlfun/loss-penalty-types.jl")
include("mlfun/loss-penalty-functions.jl")
include("mlfun/loss-penalty-utils.jl")

## SUPERVISED MODELS

include("supervised/sm-utils.jl")

#### generalized linear regression

include("supervised/glr-types.jl")
include("supervised/glr-utils.jl")
include("supervised/glr-fit.jl")
include("supervised/glr-predict.jl")



end # AnalyticalEngine
