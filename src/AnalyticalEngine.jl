module AnalyticalEngine

## HACK

struct UnimplementedException <: Exception end

## TODO evaluate whether necessary

using LearnBase
using LossFunctions
using PenaltyFunctions

const SEP{T, K} = PenaltyFunctions.ScaledElementPenalty{T, K}

import StatsBase.fit!

export fit!, predict, SEP

include("types.jl")

## SUPERVISED MODELS

include("supervised/glr.jl")     # generalized linear regression
include("supervised/glr_fit.jl")
include("supervised/glr_predict.jl")


end # AnalyticalEngine
