using Base.Test

@testset "Loss+Penalty" begin

srand(123)
x = randn(5)
y = randn(5)
θ = randn(5)

δ = x .- y
δ1 = norm(δ, 1)
δ2 = norm(δ, 2)
θ1 = norm(θ, 1)
θ2 = norm(θ, 2)

#=
TESTING RAW LOSS AND PENALTY FUNCTIONS
=#

noloss = NoLoss()
nopenalty = NoPenalty()

@test noloss(x, y) == 0
@test nopenalty(θ) == 0

l1 = L1DistLoss()
l2 = L2DistLoss()
p1 = L1Penalty()
p2 = L2Penalty()

@test l1(x, y) == δ1
@test l2(x, y) == δ2
@test p1(θ) == θ1
@test p2(θ) == θ2

#=
TESTING SCALING AND COMPOSITION OF LOSS AND PENALTY FUNCTIONS
=#

lc = l2 - l1
pc = p1 + p2

@test (2l2)(x, y) == 2δ2
@test (l2+l1)(x, y) == δ2 + δ1
@test (l2+2l1)(x, y) == δ2 + 2δ1
@test (l2-l1)(x, y) == δ2 - δ1
@test (l2+2l2-l1)(x, y) == 3δ2-δ1
@test (lc + lc)(x, y) == 2(δ2 - δ1)

@test (2p1)(θ) == 2θ1
@test (p1+p2)(θ) == θ1 + θ2
@test (p1+2p2)(θ) == θ1 + 2θ2
@test (p2-p1+3p2)(θ) == θ2 - θ1 + 3θ2
@test (pc + pc)(θ) == 2(θ1 + θ2)

end