include("_using.jl")

@testset "Types GLR" begin

lr = LinearRegression()

lr.coefficients = [1., 2., 3.]

end # testset
