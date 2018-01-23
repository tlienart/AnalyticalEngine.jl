using AnalyticalEngine
using Base.Test
using LossFunctions, PenaltyFunctions

const SEP{T, K} = PenaltyFunctions.ScaledElementPenalty{T, K}

@testset "Types GLR" begin include("types_glr.jl") end
