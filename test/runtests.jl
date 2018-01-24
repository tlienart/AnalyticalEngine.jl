include("_using.jl")

@testset "Types GLR" begin include("glr_types.jl") end
@testset "Fit GLR" begin include("glr_fit.jl") end
@testset "Predict GLR" begin include("glr_predict.jl") end
