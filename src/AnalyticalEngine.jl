module AnalyticalEngine

## HACK

struct UnimplementedException <: Exception end

## TODO evaluate whether necessary

using LearnBase
using LossFunctions
using PenaltyFunctions

export fit, predict

include("types.jl")

## SUPERVISED

include("supervised/generalized-linear-regression.jl") # generalized linear regression



end # AnalyticalEngine
