using AnalyticalEngine
using Base.Test

@testset "Loss+Penalty" begin

srand(123)
x = randn(5)
y = randn(5)
θ = randn(5)

δ = x .- y
δ1 = norm(δ, 1)
δ2 = norm(δ, 2)^2
θ1 = norm(θ, 1)
θ2 = norm(θ, 2)^2

#=
TESTING RAW LOSS AND PENALTY FUNCTIONS
=#

noloss = NoLoss()
nopenalty = NoPenalty()

@test noloss(x, y) == 0
@test nopenalty(θ) == 0
@test isdifferentiable(noloss) == true
@test isdifferentiable(nopenalty) == true

l1 = L1DistLoss()
l2 = L2DistLoss()
p1 = L1Penalty()
p2 = L2Penalty()

@test l1(x, y) ≈ δ1
@test l2(x, y) ≈ δ2
@test p1(θ) ≈ θ1
@test p2(θ) ≈ θ2
@test isdifferentiable(l1) == false
@test isdifferentiable(l2) == true

#=
TESTING SCALING AND COMPOSITION OF LOSS AND PENALTY FUNCTIONS
=#

lc = l2 - l1
pc = p1 + p2

@test (2l2)(x, y) ≈ 2δ2
@test (l2+l1)(x, y) ≈ δ2 + δ1
@test (l2+2l1)(x, y) ≈ δ2 + 2δ1
@test (l2-l1)(x, y) ≈ δ2 - δ1
@test (l2+2l2-l1)(x, y) ≈ 3δ2-δ1
@test (lc + lc)(x, y) ≈ 2(δ2 - δ1)
@test (l2 * 2)(x, y) ≈ 2δ2
@test 2 * (2 * l2)(x, y) ≈ 4δ2
@test ((l1 + 2l2) + 2l1)(x, y) ≈ 3δ1 + 2δ2
@test (l1/3)(x, y) ≈ δ1/3
@test ((2l1)/3)(x, y) ≈ 2δ1/3
@test ((l1+l2)/3)(x, y) ≈ (δ1+δ2)/3
@test ((2l1)+l2)(x, y) ≈ 2δ1+δ2
@test ((2l1)-3(l1+l2))(x, y) ≈ -δ1 - 3δ2
@test ((2l1 + 3l2) - (l1 + 2l2))(x, y) ≈ δ1 + δ2

@test (2p1)(θ) ≈ 2θ1
@test (p1+p2)(θ) ≈ θ1 + θ2
@test (p1+2p2)(θ) ≈ θ1 + 2θ2
@test (p2-p1+3p2)(θ) ≈ θ2 - θ1 + 3θ2
@test (pc + pc)(θ) ≈ 2(θ1 + θ2)
@test (p1 * 2)(θ) ≈ 2θ1
@test 2 * (p1 * 3)(θ) ≈ 6θ1
@test ((6p1 + 3p2) - 5p1)(θ) ≈ θ1 + 3θ2
@test (p1/3)(θ) ≈ θ1/3
@test ((2p1)/4)(θ) ≈ θ1/2
@test ((2p1 - 5p2)/3)(θ) ≈ (2/3)θ1 - (5/3)θ2
@test ((2p1) + (p1 + 2p2))(θ) ≈ 3θ1 + 2θ2
@test ((2p1) - (p1 + p2))(θ) ≈ θ1 - θ2
@test ((p1 * 2) + 2(p1 - p2))(θ) ≈ 4θ1 - 2θ2
@test ((4p1 + p2) - (3p2 + p1))(θ) ≈ 3θ1 - 2θ2

end
